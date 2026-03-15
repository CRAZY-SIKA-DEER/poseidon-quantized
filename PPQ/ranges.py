import torch
import torch.nn as nn


def compute_data_ranges_poseidon(
    model,
    dataloader,
    device,
    layer_names,
    percentile_prob: float = 1e-4,
):
    """
    Compute per-channel weight and activation ranges for target Linear layers.

    For weights:
        - one range per output channel

    For activations:
        - one range per input feature channel (pre-op activation, last dim)

    Returns:
        ranges_dict[layer_name] = {
            "weight_ranges":     tensor [out_features],
            "activation_ranges": tensor [in_features],
        }
    """
    model.eval()
    ranges_dict = {}

    z = torch.sqrt(torch.tensor(2.0, device=device)) * torch.special.erfinv(
        torch.tensor(1.0 - 2.0 * percentile_prob, device=device)
    )
    erf_inv_value = float(z)

    with torch.no_grad():
        # -------------------------
        # 1) Weight ranges
        # -------------------------
        for name, module in model.named_modules():
            if name in layer_names and isinstance(module, nn.Linear):
                weight = module.weight.data.to(device)  # [out_features, in_features]
                w_flat = weight.view(weight.size(0), -1)

                w_mean = w_flat.mean(dim=1)
                w_std = w_flat.std(dim=1, unbiased=False)

                tau = w_mean + w_std * erf_inv_value
                beta = tau
                alpha = 2 * w_mean - beta

                weight_ranges = (beta - alpha).clamp(min=1e-8)

                ranges_dict[name] = {
                    "weight_ranges": weight_ranges,
                    "act_stats": [],
                }

        # -------------------------
        # 2) Activation stats
        # -------------------------
        if callable(dataloader):
            dataloader = dataloader()

        for batch in dataloader:
            x = batch["pixel_values"].to(device)
            t = batch.get("time", None)
            pm = batch.get("pixel_mask", None)
            y = batch.get("labels", None)

            if t is not None:
                t = t.to(device)
            if pm is not None:
                pm = pm.to(device)
            if y is not None:
                y = y.to(device)

            def make_hook(name):
                def hook(module, inputs, output):
                    x_in = inputs[0]

                    if x_in.dim() == 2:
                        x_flat = x_in.transpose(0, 1)  # [C, B]
                    elif x_in.dim() == 3:
                        x_flat = x_in.permute(2, 0, 1).reshape(x_in.size(-1), -1)  # [C, B*seq]
                    elif x_in.dim() == 4:
                        x_flat = x_in.permute(3, 0, 1, 2).reshape(x_in.size(-1), -1)  # [C, B*H*W]
                    else:
                        x_flat = x_in.reshape(-1, x_in.size(-1)).transpose(0, 1)

                    x_mean = x_flat.mean(dim=1)
                    x_std = x_flat.std(dim=1, unbiased=False)

                    ranges_dict[name]["act_stats"].append(
                        {
                            "mean": x_mean.detach().cpu(),
                            "std": x_std.detach().cpu(),
                        }
                    )
                return hook

            handles = []
            for name, module in model.named_modules():
                if name in layer_names and isinstance(module, nn.Linear):
                    handles.append(module.register_forward_hook(make_hook(name)))

            _ = model(
                pixel_values=x,
                time=t,
                pixel_mask=pm,
                labels=y,
            )

            for h in handles:
                h.remove()

        # -------------------------
        # 3) Aggregate activation ranges
        # -------------------------
        for name in ranges_dict:
            act_stats = ranges_dict[name]["act_stats"]

            if len(act_stats) == 0:
                continue

            all_means = torch.stack([s["mean"] for s in act_stats])
            all_stds = torch.stack([s["std"] for s in act_stats])

            avg_mean = all_means.mean(dim=0).to(device)
            avg_std = all_stds.mean(dim=0).to(device)

            tau = avg_mean + avg_std * erf_inv_value
            beta = tau
            alpha = 2 * avg_mean - tau

            activation_ranges = (beta - alpha).clamp(min=1e-8)

            ranges_dict[name]["activation_ranges"] = activation_ranges
            del ranges_dict[name]["act_stats"]

    return ranges_dict


def evaluate_quantized_model_poseidon(
    model,
    dataloader,
    ranges,
    device,
    layer_names,
    num_bits: int = 8,
):
    """
    Fake-quant evaluation using per-channel weight + activation ranges.

    Returns:
        avg_loss: average MSE over given dataloader
    """
    model.eval()
    total_loss = 0.0
    count = 0
    num_levels = 2 ** num_bits - 1

    if callable(dataloader):
        dataloader = dataloader()

    with torch.no_grad():
        for batch in dataloader:
            x = batch["pixel_values"].to(device)
            t = batch.get("time", None)
            pm = batch.get("pixel_mask", None)
            y = batch.get("labels", None)

            if t is not None:
                t = t.to(device)
            if pm is not None:
                pm = pm.to(device)
            if y is not None:
                y = y.to(device)

            def make_quant_hook(name, w_ranges, a_ranges):
                def hook(mod, inp, out):
                    x_in = inp[0]

                    if x_in.dim() == 2:
                        a_step = a_ranges.view(1, -1) / num_levels
                    elif x_in.dim() == 3:
                        a_step = a_ranges.view(1, 1, -1) / num_levels
                    elif x_in.dim() == 4:
                        a_step = a_ranges.view(1, 1, 1, -1) / num_levels
                    else:
                        return out

                    x_quant = torch.round(x_in / a_step) * a_step

                    w_flat = mod.weight.view(mod.weight.size(0), -1)
                    w_step = w_ranges.view(-1, 1) / num_levels
                    w_quant = torch.round(w_flat / w_step) * w_step
                    w_quant = w_quant.view_as(mod.weight)

                    return torch.nn.functional.linear(x_quant, w_quant, mod.bias)
                return hook

            handles = []
            for name, module in model.named_modules():
                if (
                    name in layer_names
                    and isinstance(module, nn.Linear)
                    and name in ranges
                    and "weight_ranges" in ranges[name]
                    and "activation_ranges" in ranges[name]
                ):
                    handles.append(
                        module.register_forward_hook(
                            make_quant_hook(
                                name,
                                ranges[name]["weight_ranges"],
                                ranges[name]["activation_ranges"],
                            )
                        )
                    )

            outputs = model(
                pixel_values=x,
                time=t,
                pixel_mask=pm,
                labels=y,
            )

            pred = outputs.output
            loss = torch.nn.functional.mse_loss(pred, y)

            total_loss += loss.item()
            count += 1

            for h in handles:
                h.remove()

    return total_loss / count if count > 0 else float("inf")


def find_best_percentile_poseidon(
    model,
    dataloader,
    device,
    layer_names,
    percentile_candidates=(1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
    num_bits: int = 8,
):
    """
    Grid-search percentile_prob by fake-quant evaluation.
    """
    best_loss = float("inf")
    best_percentile = percentile_candidates[0]

    for p in percentile_candidates:
        ranges = compute_data_ranges_poseidon(
            model=model,
            dataloader=dataloader,
            device=device,
            layer_names=layer_names,
            percentile_prob=p,
        )

        loss = evaluate_quantized_model_poseidon(
            model=model,
            dataloader=dataloader,
            ranges=ranges,
            device=device,
            layer_names=layer_names,
            num_bits=num_bits,
        )

        if loss < best_loss:
            best_loss = loss
            best_percentile = p

    return best_percentile