"""
Metrics and evaluation utilities for PPQ experiments on the Poseidon model.

This module provides helper functions used to analyze quantization behavior
and evaluate model performance after applying learned step sizes.

Main functionalities
--------------------

1. Channel parameter weighting
   build_channel_param_weights(...)
   --------------------------------
   Computes per-channel parameter counts for Linear layers. Each output channel
   of a Linear layer controls `in_features` weights, so this function returns a
   tensor indicating how many parameters are associated with each channel.
   These weights are later used for parameter-weighted statistics such as
   average bit-width.


2. Average effective bit-width estimation
   compute_avg_bits(...)
   ----------------------
   Computes the parameter-weighted average bit-width implied by the current
   step sizes:

       bits_{l,k} = log2(R_{l,k} / S_{l,k})

   where:
       R_{l,k} = dynamic range of weight channel k in layer l
       S_{l,k} = learned quantization step size

   This provides a global estimate of how many effective bits the model is
   using after quantization.


3. Dynamic quantization baseline
   compute_dynamic_stepsizes(...)
   -------------------------------
   Computes per-channel step sizes using a symmetric dynamic range rule:

       step_k = 2 * max(|w_k|) / (2^bits - 1)

   where `w_k` represents the weights of output channel k. This corresponds to
   symmetric max-absolute quantization and is used as a baseline for comparison
   with PPQ-learned step sizes.


4. Quantized model evaluation
   evaluate_with_stepsizes(...)
   ------------------------------
   Evaluates model performance under weight-only fake quantization.

   The function:
       • injects quantized weights using forward hooks
       • runs the model on a validation loader
       • computes error metrics (L1 and relative L1)

   Quantization is simulated using:

       w_quant = round(w / step) * step

   Metrics are computed using the Poseidon evaluation functions:
       - absolute L1 error
       - relative L1 error (percentage)

Overall role
------------

This module provides utilities for:
    • measuring effective bit usage
    • generating dynamic quantization baselines
    • evaluating quantized model accuracy

It is primarily used during PPQ experiments to monitor compression quality
and prediction accuracy when different quantization step sizes are applied.
"""

import torch
import torch.nn as nn
import numpy as np

from scOT.metrics import relative_lp_error, lp_error


def build_channel_param_weights(model: nn.Module, layer_names):
    """
    For each Linear layer, return a 1D tensor of length out_features where
    each entry is the number of parameters controlled by that output channel.
    For Linear, each output channel owns `in_features` weights.
    """
    name2mod = dict(model.named_modules())
    channel_weights = {}

    for name in layer_names:
        mod = name2mod.get(name, None)
        if not isinstance(mod, nn.Linear):
            continue

        in_features = mod.in_features
        out_features = mod.out_features
        channel_weights[name] = torch.full((out_features,), float(in_features))

    return channel_weights


def compute_avg_bits(
    step_sizes_dict,
    ranges_dict,
    channel_weights=None,
    eps: float = 1e-8,
) -> float:
    """
    Parameter-weighted average effective bit-width over weight channels only.

    bits_{l,k} = log2(R_{l,k} / S_{l,k})
    """
    total_bits_weighted = 0.0
    total_weight = 0.0

    for name, wa in step_sizes_dict.items():
        if name not in ranges_dict:
            continue

        w_step, _a_step = wa
        rec = ranges_dict[name]

        if "weight_ranges" not in rec or w_step is None:
            continue

        w_range = rec["weight_ranges"].to(w_step.device)
        bits = torch.log2((w_range + eps) / (w_step + eps))

        if channel_weights is not None and name in channel_weights:
            w = channel_weights[name].to(bits.device)
            if w.numel() == 1:
                w = w.expand_as(bits)
        else:
            w = torch.ones_like(bits)

        total_bits_weighted += float((bits * w).sum().item())
        total_weight += float(w.sum().item())

    if total_weight == 0.0:
        return float("nan")

    return total_bits_weighted / total_weight


def compute_dynamic_stepsizes(
    model: nn.Module,
    layer_names,
    num_bits: int = 8,
    device: str = "cuda",
):
    """
    Compute per-channel dynamic weight step sizes for Linear layers:

        step_k = 2 * max_abs(w_k) / (2^bits - 1)

    where w_k is one output-channel row of the weight matrix.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    name2mod = dict(model.named_modules())
    dynamic_steps = {}

    denom = (2 ** num_bits) - 1
    if denom <= 0:
        raise ValueError(f"Invalid num_bits={num_bits}")

    with torch.no_grad():
        for name in layer_names:
            mod = name2mod.get(name, None)
            if not isinstance(mod, nn.Linear):
                continue

            w = mod.weight.data.to(device)
            out_features = w.size(0)

            w_flat = w.view(out_features, -1)
            max_abs = w_flat.abs().max(dim=1).values
            step = (2.0 * max_abs) / float(denom)

            dynamic_steps[name] = step.cpu()

    return dynamic_steps


def evaluate_with_stepsizes(
    model: nn.Module,
    val_loader,
    weight_steps,
    act_steps,
    layer_names,
    device: str = "cuda",
):
    """
    Evaluate model with weight-only fake quantization.

    weight_steps may be:
      - PPQ format: {layer_name: (w_step, a_step)}
      - dynamic format: {layer_name: w_step_tensor}

    act_steps is kept only for API compatibility and is not used.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    name2mod = dict(model.named_modules())

    w_steps_per_layer = {}
    for name in layer_names:
        if name not in name2mod:
            continue
        if not isinstance(name2mod[name], nn.Linear):
            continue
        if name not in weight_steps:
            continue

        w_info = weight_steps[name]
        if isinstance(w_info, (tuple, list)) and len(w_info) >= 1:
            w_step = w_info[0]
        else:
            w_step = w_info

        if isinstance(w_step, torch.nn.Parameter):
            w_step = w_step.detach()
        if not isinstance(w_step, torch.Tensor):
            w_step = torch.tensor(w_step)

        w_steps_per_layer[name] = w_step.to(device)

    def make_weight_quant_hook(w_step_tensor):
        def hook(mod, inp, out):
            x = inp[0]
            w = mod.weight
            w_flat = w.view(w.size(0), -1)
            step = w_step_tensor.view(-1, 1)

            w_quant = torch.round(w_flat / step) * step
            w_quant = w_quant.view_as(w)

            return torch.nn.functional.linear(x, w_quant, mod.bias)
        return hook

    handles = []
    for name, mod in name2mod.items():
        if name in w_steps_per_layer and isinstance(mod, nn.Linear):
            handles.append(
                mod.register_forward_hook(
                    make_weight_quant_hook(w_steps_per_layer[name])
                )
            )

    loader = val_loader() if callable(val_loader) else val_loader

    rel_l1_list = []
    abs_l1_list = []
    
    print(f"[DEBUG] number of quant hooks registered = {len(handles)}")


    with torch.no_grad():
        for batch in loader:
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
            else:
                continue

            outputs = model(
                pixel_values=x,
                time=t,
                pixel_mask=pm,
                labels=y,
            )
            pred = outputs.output

            pred_np = pred.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()

            batch_rel = relative_lp_error(pred_np, y_np, p=1, return_percent=True)
            batch_abs = lp_error(pred_np, y_np, p=1)

            rel_l1_list.append(float(np.mean(batch_rel)))
            abs_l1_list.append(float(np.mean(batch_abs)))

    for h in handles:
        h.remove()

    if len(rel_l1_list) == 0:
        return {"l1": float("nan"), "rel_l1": float("nan")}

    return {
        "l1": float(sum(abs_l1_list) / len(abs_l1_list)),
        "rel_l1": float(sum(rel_l1_list) / len(rel_l1_list)),
    }