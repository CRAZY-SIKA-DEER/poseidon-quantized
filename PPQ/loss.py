"""
Loss and objective utilities for Probabilistic Post-Training Quantization (PPQ).

This module implements the optimization objectives used to learn per-channel
quantization step sizes for the Poseidon (ScOT) model. The functions here define
both the likelihood term (via Monte-Carlo noise simulation) and the prior terms
(MDL prior and optional bit-budget penalty) used during PPQ optimization.

Main components
---------------

1. MDL Prior
   compute_mdl_prior(...)
   ---------------------
   Implements the Minimum Description Length (MDL) prior on weight step sizes.

       prior = gamma * Σ_{l,k} log2(R_{l,k} / S_{l,k})

   where:
       R_{l,k} = dynamic range of weight channel k in layer l
       S_{l,k} = learned quantization step size for that channel

   Intuition:
       Smaller step sizes imply more bits (higher precision), so the MDL prior
       penalizes overly small step sizes and encourages compression.


2. Network-wise Monte Carlo Likelihood
   compute_mc_loss_single_batch_network(...)
   -----------------------------------------
   Estimates the likelihood term by simulating quantization noise at the
   full-network level.

   Procedure:
       • Select one frozen calibration batch
       • Inject uniform quantization noise into Linear layer weights
       • Run a full forward pass of the model
       • Compare noisy output with the cached clean output
       • Repeat for multiple Monte-Carlo samples

   Likelihood approximation:

       E_mc [ ||y_noisy − y_clean||² / (2η) ]

   Notes:
       • Weight noise only (no activation noise)
       • Uses forward hooks to temporarily replace Linear layer weights
       • Frozen batches ensure identical inputs across MC samples


3. Network-wise MAP Objective
   compute_mc_loss_with_prior(...)
   -------------------------------
   Combines the likelihood term and the MDL prior:

       total_loss = likelihood + prior

   This corresponds to MAP estimation of the step sizes.


4. Layer-wise Monte Carlo Likelihood
   compute_mc_loss_single_batch(...)
   ---------------------------------
   A more efficient alternative to the network-wise loss.

   Instead of running the entire model, this version:
       • Uses cached per-layer clean inputs and outputs
       • Injects weight noise into one layer at a time
       • Computes output error locally for that layer

   The loss is averaged across all valid target layers.


5. Layer-wise MAP Objective
   compute_mc_loss_with_prior_layerwise(...)
   -----------------------------------------
   Combines the layer-wise likelihood with the MDL prior:

       total_loss = layerwise_likelihood + prior


6. Average Bit-Width Constraint
   prior_weighted_avg_bits_cap_poseidon(...)
   -----------------------------------------
   Adds a soft constraint to limit the average bit-width across channels.

   Steps:
       • Compute per-channel bit estimate

             bits = log2(R / S)

       • Compute parameter-weighted average bits
       • Apply smooth penalty if the average exceeds a target:

             penalty = λ * softplus(α * (avg_bits − target_bits))²

   This prevents the optimizer from allocating too many bits to sensitive
   channels and encourages global compression.


Key Concepts
------------

• Step sizes are learned per channel for each quantized layer.
• Quantization noise is modeled as uniform noise:

      noise ~ U(-Δ/2, Δ/2)

  where Δ is the step size.

• Monte-Carlo sampling approximates the expected quantization error.
• Frozen calibration batches ensure deterministic likelihood evaluation.
• The final optimization objective combines:
      likelihood + MDL prior + optional bit-budget penalty.

This module therefore provides the core mathematical objective used by the
PPQ optimization loop to learn quantization step sizes for Poseidon.
"""
import torch
import torch.nn.functional as F

from PPQ.noise import add_quantization_noise


def compute_mdl_prior(
    step_sizes_dict,
    ranges_dict,
    gamma: float = 0.001,
    eps: float = 1e-8,
):
    """
    MDL prior on WEIGHT step sizes only.

    prior = gamma * sum_{l,k} log2(R_{l,k} / S_{l,k})

    Args:
        step_sizes_dict:
            {layer_name: (weight_step_sizes, activation_step_sizes)}
        ranges_dict:
            {layer_name: {"weight_ranges": ..., "activation_ranges": ...}}
    """
    device = None
    for pair in step_sizes_dict.values():
        for p in pair:
            if p is not None:
                device = p.device
                break
        if device is not None:
            break

    if device is None:
        device = torch.device("cpu")

    prior_loss = torch.zeros((), device=device)

    for name, (weight_step_sizes, _activation_step_sizes) in step_sizes_dict.items():
        rec = ranges_dict.get(name, None)
        if rec is None or "weight_ranges" not in rec:
            continue

        w_ranges = rec["weight_ranges"].to(device)

        if w_ranges.numel() != weight_step_sizes.numel():
            raise ValueError(
                f"[{name}] weight_ranges.shape={tuple(w_ranges.shape)} "
                f"!= weight_step_sizes.shape={tuple(weight_step_sizes.shape)}"
            )

        w_term = torch.log2(
            torch.clamp(w_ranges, min=eps) /
            torch.clamp(weight_step_sizes, min=eps)
        )
        prior_loss = prior_loss + gamma * torch.sum(w_term)

    return prior_loss


def compute_mc_loss_single_batch_network(
    model,
    step_sizes_dict,
    frozen_batches,
    clean_net_outputs,
    batch_idx: int,
    num_mc_samples: int = 10,
    eta: float = 1e-4,
    device: str | torch.device = "cuda",
):
    """
    Network-wise MC likelihood for one frozen batch.

    Likelihood:
        E_mc [ ||y_noisy - y_clean||^2 / (2*eta) ]

    Current version:
        - weight noise only
        - no activation noise
    """
    if model is None:
        raise ValueError("model must not be None.")

    device = torch.device(device)
    model = model.to(device).eval()

    if batch_idx < 0 or batch_idx >= len(frozen_batches):
        raise IndexError(f"batch_idx={batch_idx} out of range for frozen_batches")
    if batch_idx < 0 or batch_idx >= len(clean_net_outputs):
        raise IndexError(f"batch_idx={batch_idx} out of range for clean_net_outputs")

    batch = frozen_batches[batch_idx]
    y_clean = clean_net_outputs[batch_idx].to(device)

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

    name2module = dict(model.named_modules())
    target_layers = [
        name for name in step_sizes_dict.keys()
        if isinstance(name2module.get(name, None), torch.nn.Linear)
    ]

    if len(target_layers) == 0:
        raise ValueError("No valid nn.Linear layers found in step_sizes_dict.")

    mc_losses = []

    for _ in range(num_mc_samples):
        handles = []

        def make_noisy_linear_hook(w_step_tensor):
            def hook(mod, inp, out):
                x_in = inp[0]
                w_clean = mod.weight
                w_noisy = add_quantization_noise(w_clean, w_step_tensor, channel_axis=0)
                return torch.nn.functional.linear(x_in, w_noisy, mod.bias)
            return hook

        for lname in target_layers:
            mod = name2module[lname]
            w_step, _a_step = step_sizes_dict[lname]

            if w_step.numel() != mod.weight.size(0):
                continue

            '''
            What register_forward_hook does
                            When you run:
                            handles.append(
                                mod.register_forward_hook(make_noisy_linear_hook(...))
                            )
                            PyTorch internally stores the hook inside the module:
                            module._forward_hooks
                            So the module now has:
                            Linear layer
                            ├ weight
                            ├ bias
                            └ forward_hook_list
                            2️⃣ What happens during forward pass
                            When PyTorch executes a module forward:
                            output = module(input)
                            internally PyTorch does something like:
                            out = module.forward(input)

                            for hook in module.forward_hooks:
                                out = hook(module, input, out)
                            So the hook automatically modifies the output.
                            3️⃣ What your hook does
                            Your hook:
                            def hook(mod, inp, out):
                                x_in = inp[0]
                                w_noisy = add_quantization_noise(...)
                                return linear(x_in, w_noisy)
                            So instead of returning the original
                            out = xW + b
                            it replaces it with
                            out = x(W + noise) + b
            '''
            handles.append(
                mod.register_forward_hook(
                    make_noisy_linear_hook(w_step.to(device))
                )
            )

        outputs = model(
            pixel_values=x,
            time=t,
            pixel_mask=pm,
            labels=y,
        )
        y_noisy = outputs.output

        loss_elem = torch.mean((y_noisy - y_clean) ** 2) / (2.0 * eta)
        mc_losses.append(loss_elem)

        for h in handles:
            h.remove()

    return torch.stack(mc_losses).mean()


def compute_mc_loss_with_prior(
    model,
    step_sizes_dict,
    frozen_batches,
    clean_net_outputs,
    ranges_dict,
    batch_idx: int,
    num_mc_samples: int = 10,
    eta: float = 1e-4,
    gamma: float = 0.005,
    device: str = "cuda",
):
    """
    Total MAP objective for one batch:
        total = likelihood + MDL prior
    """
    likelihood_loss = compute_mc_loss_single_batch_network(
        model=model,
        step_sizes_dict=step_sizes_dict,
        frozen_batches=frozen_batches,
        clean_net_outputs=clean_net_outputs,
        batch_idx=batch_idx,
        num_mc_samples=num_mc_samples,
        eta=eta,
        device=device,
    )

    prior_loss = compute_mdl_prior(
        step_sizes_dict=step_sizes_dict,
        ranges_dict=ranges_dict,
        gamma=gamma,
    )

    total_loss = likelihood_loss + prior_loss
    return total_loss, likelihood_loss, prior_loss


def compute_mc_loss_single_batch(
    model,
    step_sizes_dict,
    clean_inputs,
    clean_outputs,
    batch_idx: int,
    num_mc_samples: int = 10,
    eta: float = 1e-4,
    device: str | torch.device = "cuda",
):
    """
    Layer-wise MC likelihood for a single cached calibration batch.

    For each target layer:
        E_mc [ ||Y_noisy - Y_clean||^2 / (2*eta) ]

    Current version:
        - activation stays clean
        - weight noise only
        - averages over valid target layers
    """
    device = torch.device(device)
    if model is not None:
        model = model.to(device).eval()

    target_layers = [
        name for name in step_sizes_dict.keys()
        if name in clean_inputs and name in clean_outputs
    ]
    if not target_layers:
        raise ValueError("No overlapping layers between step_sizes_dict and clean IO caches.")

    name2module = dict(model.named_modules())

    batch_loss = None
    layer_contrib = 0

    for name in target_layers:
        module = name2module.get(name, None)
        if module is None or not isinstance(module, torch.nn.Linear):
            continue

        if batch_idx >= len(clean_inputs[name]) or batch_idx >= len(clean_outputs[name]):
            continue

        x_b = clean_inputs[name][batch_idx]
        y_b = clean_outputs[name][batch_idx]

        if x_b is None or y_b is None:
            continue

        x_clean = x_b.to(device)
        y_clean = y_b.to(device)
        w_clean = module.weight.to(device)

        w_step, _a_step = step_sizes_dict[name]
        w_step = w_step.to(device)

        if x_clean.shape[-1] != w_clean.shape[1]:
            raise ValueError(
                f"{name}: x_clean last dim {x_clean.shape[-1]} != w in_features {w_clean.shape[1]}"
            )
        if w_step.numel() != w_clean.shape[0]:
            raise ValueError(
                f"{name}: w_step numel {w_step.numel()} != out_features {w_clean.shape[0]}"
            )

        mc_losses = []
        for _ in range(num_mc_samples):
            w_noisy = add_quantization_noise(w_clean, w_step, channel_axis=0)
            y_noisy = torch.nn.functional.linear(x_clean, w_noisy, module.bias)
            loss_elem = torch.mean((y_noisy - y_clean) ** 2) / (2.0 * eta)
            mc_losses.append(loss_elem)

        layer_loss = torch.stack(mc_losses).mean()

        batch_loss = layer_loss if batch_loss is None else (batch_loss + layer_loss)
        layer_contrib += 1

    if layer_contrib == 0:
        return torch.zeros((), device=device)

    return batch_loss / layer_contrib


def compute_mc_loss_with_prior_layerwise(
    model,
    step_sizes_dict,
    clean_inputs,
    clean_outputs,
    ranges_dict,
    batch_idx: int,
    num_mc_samples: int = 10,
    eta: float = 1e-4,
    gamma: float = 0.005,
    device: str = "cuda",
):
    """
    Layer-wise MAP objective for one batch:
        total = layerwise_likelihood + MDL prior
    """
    likelihood_loss = compute_mc_loss_single_batch(
        model=model,
        step_sizes_dict=step_sizes_dict,
        clean_inputs=clean_inputs,
        clean_outputs=clean_outputs,
        batch_idx=batch_idx,
        num_mc_samples=num_mc_samples,
        eta=eta,
        device=device,
    )

    prior_loss = compute_mdl_prior(
        step_sizes_dict=step_sizes_dict,
        ranges_dict=ranges_dict,
        gamma=gamma,
    )

    total_loss = likelihood_loss + prior_loss
    return total_loss, likelihood_loss, prior_loss


def prior_weighted_avg_bits_cap_poseidon(
    step_sizes_dict,
    ranges_dict,
    channel_weights,
    target_bits: float = 4.0,
    lam: float = 1.0,
    alpha: float = 10.0,
    eps: float = 1e-8,
):
    """
    Smooth penalty if parameter-weighted average bits exceeds target_bits.

    penalty = lam * softplus(alpha * (avg_bits - target_bits))^2
    """
    total_bits_weighted = None
    total_weight = None

    for name, (w_step, _a_step) in step_sizes_dict.items():
        rec = ranges_dict.get(name, None)
        if rec is None or "weight_ranges" not in rec or w_step is None:
            continue

        w_range = rec["weight_ranges"].to(w_step.device)
        bits = torch.log2((w_range + eps) / (w_step + eps))

        if channel_weights is not None and name in channel_weights:
            w = channel_weights[name].to(bits.device)
            if w.numel() == 1:
                w = w.expand_as(bits)
        else:
            w = torch.ones_like(bits)

        bits_w = (bits * w).sum()
        w_sum = w.sum()

        total_bits_weighted = bits_w if total_bits_weighted is None else (total_bits_weighted + bits_w)
        total_weight = w_sum if total_weight is None else (total_weight + w_sum)

    if total_weight is None or float(total_weight.detach().cpu()) == 0.0:
        device = next(
            p.device for pair in step_sizes_dict.values() for p in pair if p is not None
        )
        return torch.zeros((), device=device)

    bits_avg = total_bits_weighted / (total_weight + eps)
    excess = bits_avg - target_bits
    penalty = lam * F.softplus(alpha * excess) ** 2
    return penalty