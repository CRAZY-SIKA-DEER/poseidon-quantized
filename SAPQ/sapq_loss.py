# SAPQ/sapq_loss.py
from __future__ import annotations

from contextlib import contextmanager
from typing import Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from PPQ.noise import add_quantization_noise
from BRECQ.quant.quant_layer import QuantModule
from BRECQ.quant.poseidon_data_utils import get_model_output_tensor


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _move_to_device(obj, device: torch.device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(_move_to_device(x, device) for x in obj)
    if isinstance(obj, list):
        return [_move_to_device(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


def _get_reconstruction_output(block_output):
    if isinstance(block_output, tuple):
        return block_output[0]
    return block_output


def _get_weight_steps(step_entry):
    """
    Accept either:
        step_entry = weight_step_tensor
    or
        step_entry = (weight_step_tensor, activation_step_tensor)
    """
    if isinstance(step_entry, tuple):
        return step_entry[0]
    return step_entry


def _named_quantmodules(block) -> dict[str, QuantModule]:
    return {
        name: module
        for name, module in block.named_modules()
        if isinstance(module, QuantModule)
    }


def _infer_device_from_steps(step_sizes_dict):
    for step_entry in step_sizes_dict.values():
        w_step = _get_weight_steps(step_entry)
        if w_step is not None:
            return w_step.device
    return torch.device("cpu")


# ---------------------------------------------------------------------
# likelihood energy
# ---------------------------------------------------------------------

def compute_block_fisher_diag_energy(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    grad: torch.Tensor,
) -> torch.Tensor:
    """
    Per-sample Fisher-diagonal weighted block reconstruction energy.

    Implements:
        E_i = sum( (pred - tgt)^2 * grad^2 )

    Args:
        pred: [B, ...]
        tgt:  [B, ...]
        grad: [B, ...]

    Returns:
        energy: [B]
    """
    if pred.shape != tgt.shape:
        raise ValueError(
            f"pred.shape={tuple(pred.shape)} != tgt.shape={tuple(tgt.shape)}"
        )
    if grad.shape != pred.shape:
        raise ValueError(
            f"grad.shape={tuple(grad.shape)} != pred.shape={tuple(pred.shape)}"
        )

    reduce_dims = tuple(range(1, pred.dim()))
    delta = pred - tgt
    energy = (delta.pow(2) * grad.pow(2)).sum(dim=reduce_dims)
    return energy


# ---------------------------------------------------------------------
# temporary noisy-weight injection
# ---------------------------------------------------------------------

@contextmanager
def temporary_block_noisy_weights(
    block,
    block_step_sizes_dict: Mapping[str, torch.Tensor | Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
):
    """
    Temporarily replace QuantModule.org_weight by:
        W_noisy = W + s * U(-1/2, 1/2)

    Important:
    - This uses the FP path of QuantModule (org_weight), not AdaRound.
    - This is for SAPQ step-size optimization.
    """
    name2module = _named_quantmodules(block)
    saved_org_weights = {}

    try:
        for local_name, step_entry in block_step_sizes_dict.items():
            if local_name not in name2module:
                continue

            module = name2module[local_name]
            w_step = _get_weight_steps(step_entry).to(device)

            w_clean = module.org_weight
            if w_clean.device != device:
                w_clean = w_clean.to(device)

            if w_step.numel() != w_clean.shape[0]:
                raise ValueError(
                    f"[{local_name}] step channels ({w_step.numel()}) "
                    f"!= weight out_channels ({w_clean.shape[0]})"
                )

            saved_org_weights[local_name] = module.org_weight
            w_noisy = add_quantization_noise(w_clean, w_step, channel_axis=0)
            module.org_weight = w_noisy

        yield

    finally:
        for local_name, original_weight in saved_org_weights.items():
            name2module[local_name].org_weight = original_weight


@contextmanager
def temporary_model_noisy_weights(
    model,
    step_sizes_dict: Mapping[str, torch.Tensor | Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
):
    """
    Temporarily replace nn.Linear weights in the whole model by:
        W_noisy = W + s * U(-1/2, 1/2)

    step_sizes_dict is keyed by ORIGINAL / model.named_modules() names.
    Supported value formats:
        step_sizes_dict[name] = weight_step_tensor
    or
        step_sizes_dict[name] = (weight_step_tensor, activation_step_tensor)
    """
    name2module = dict(model.named_modules())
    saved_weights = {}

    try:
        for layer_name, step_entry in step_sizes_dict.items():
            mod = name2module.get(layer_name, None)
            if mod is None or not isinstance(mod, nn.Linear):
                continue

            w_step = _get_weight_steps(step_entry).to(device)
            w_clean = mod.weight
            if w_clean.device != device:
                w_clean = w_clean.to(device)

            if w_step.numel() != w_clean.shape[0]:
                raise ValueError(
                    f"[{layer_name}] step channels ({w_step.numel()}) "
                    f"!= weight out_features ({w_clean.shape[0]})"
                )

            saved_weights[layer_name] = mod.weight
            w_noisy = add_quantization_noise(w_clean, w_step, channel_axis=0)
            mod.weight = nn.Parameter(w_noisy, requires_grad=False)

        yield

    finally:
        for layer_name, original_weight in saved_weights.items():
            name2module[layer_name].weight = original_weight


# ---------------------------------------------------------------------
# likelihoods
# ---------------------------------------------------------------------

def compute_block_mc_negative_loglikelihood(
    block,
    block_step_sizes_dict: Mapping[str, torch.Tensor | Tuple[torch.Tensor, torch.Tensor]],
    cached_block_inputs,
    cached_block_outputs,
    cached_block_grads,
    batch_idx: int,
    num_mc_samples: int = 10,
    device: str | torch.device = "cuda",
):
    """
    SAPQ block-wise Monte Carlo negative log-likelihood for one cached batch.

    Implements:
        - sum_i log( (1/M) sum_j exp( -1/2 * E_{i,j} ) )

    where E_{i,j} is the Fisher-diagonal weighted block reconstruction energy.
    """
    device = torch.device(device)
    block = block.to(device).eval()
    block.set_quant_state(False, False)

    if batch_idx < 0 or batch_idx >= len(cached_block_inputs):
        raise IndexError(f"batch_idx={batch_idx} out of range for cached_block_inputs")
    if batch_idx < 0 or batch_idx >= len(cached_block_outputs):
        raise IndexError(f"batch_idx={batch_idx} out of range for cached_block_outputs")
    if batch_idx < 0 or batch_idx >= len(cached_block_grads):
        raise IndexError(f"batch_idx={batch_idx} out of range for cached_block_grads")

    cur_inp = _move_to_device(cached_block_inputs[batch_idx], device)
    tgt = cached_block_outputs[batch_idx].to(device)
    grad = cached_block_grads[batch_idx].to(device)

    score_list = []
    for _ in range(num_mc_samples):
        with temporary_block_noisy_weights(
            block=block,
            block_step_sizes_dict=block_step_sizes_dict,
            device=device,
        ):
            pred = block(*cur_inp)
            pred = _get_reconstruction_output(pred)

        energy = compute_block_fisher_diag_energy(
            pred=pred,
            tgt=tgt,
            grad=grad,
        )  # [B]

        score = -0.5 * energy
        score_list.append(score)

    scores = torch.stack(score_list, dim=0)  # [M, B]
    log_prob_per_sample = torch.logsumexp(scores, dim=0) - torch.log(
        torch.tensor(float(num_mc_samples), device=device)
    )

    nll = -log_prob_per_sample.sum()
    return nll


def compute_mc_negative_loglikelihood_network(
    model,
    step_sizes_dict: Mapping[str, torch.Tensor | Tuple[torch.Tensor, torch.Tensor]],
    frozen_batches,
    clean_net_outputs,
    batch_idx: int,
    num_mc_samples: int = 10,
    eta: float = 1e-4,
    device: str | torch.device = "cuda",
):
    """
    Network-wise Monte Carlo negative log-likelihood for one frozen batch.

    Old temporary replacement version.
    Final mismatch is measured at the model output.
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

    score_list = []

    for _ in range(num_mc_samples):
        with temporary_model_noisy_weights(
            model=model,
            step_sizes_dict=step_sizes_dict,
            device=device,
        ):
            outputs = model(
                pixel_values=x,
                time=t,
                pixel_mask=pm,
                labels=y,
            )
            y_noisy = get_model_output_tensor(outputs)

        if y_noisy.shape != y_clean.shape:
            raise ValueError(
                f"y_noisy.shape={tuple(y_noisy.shape)} != y_clean.shape={tuple(y_clean.shape)}"
            )

        reduce_dims = tuple(range(1, y_noisy.dim()))
        sqerr_per_sample = (y_noisy - y_clean).pow(2).sum(dim=reduce_dims) / (2.0 * eta)
        score = -sqerr_per_sample
        score_list.append(score)

    scores = torch.stack(score_list, dim=0)  # [M, B]
    log_prob_per_sample = torch.logsumexp(scores, dim=0) - torch.log(
        torch.tensor(float(num_mc_samples), device=device)
    )

    nll = -log_prob_per_sample.sum()
    return nll


def compute_mc_negative_loglikelihood_network_global(
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
    Global / network-wise SAPQ Monte Carlo negative log-likelihood.

    Likelihood:
        - sum_i log( (1/M) sum_j exp( - ||y_noisy_i - y_clean_i||^2 / (2*eta) ) )

    Important:
    - whole model final output only
    - optimize all channel step sizes together
    - uses forward hooks on nn.Linear so gradient flows to w_step
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

    score_list = []

    for _ in range(num_mc_samples):
        handles = []

        def make_noisy_linear_hook(w_step_tensor):
            def hook(mod, inp, out):
                x_in = inp[0]
                w_clean = mod.weight
                w_noisy = add_quantization_noise(w_clean, w_step_tensor, channel_axis=0)
                return F.linear(x_in, w_noisy, mod.bias)
            return hook

        for lname in target_layers:
            mod = name2module[lname]
            step_entry = step_sizes_dict[lname]
            w_step = _get_weight_steps(step_entry).to(device)

            if w_step.numel() != mod.weight.size(0):
                continue

            handles.append(
                mod.register_forward_hook(
                    make_noisy_linear_hook(w_step)
                )
            )

        outputs = model(
            pixel_values=x,
            time=t,
            pixel_mask=pm,
            labels=y,
        )
        y_noisy = get_model_output_tensor(outputs)

        for h in handles:
            h.remove()

        if y_noisy.shape != y_clean.shape:
            raise ValueError(
                f"y_noisy.shape={tuple(y_noisy.shape)} != y_clean.shape={tuple(y_clean.shape)}"
            )

        reduce_dims = tuple(range(1, y_noisy.dim()))
        sqerr_per_sample = (y_noisy - y_clean).pow(2).mean(dim=reduce_dims) / (2.0 * eta)
        score = -sqerr_per_sample
        score_list.append(score)

    scores = torch.stack(score_list, dim=0)  # [M, B]
    log_prob_per_sample = torch.logsumexp(scores, dim=0) - torch.log(
        torch.tensor(float(num_mc_samples), device=device)
    )

    nll = -log_prob_per_sample.sum()
    return nll


# ---------------------------------------------------------------------
# priors
# ---------------------------------------------------------------------

def compute_ppq_prior(
    step_sizes_dict,
    ranges_dict,
    gamma: float = 0.005,
    eps: float = 1e-8,
):
    """
    PPQ MDL prior on WEIGHT step sizes only.

    prior = gamma * sum_{l,k} log2(R_{l,k} / S_{l,k})
    """
    device = _infer_device_from_steps(step_sizes_dict)
    prior_loss = torch.zeros((), device=device)

    for name, step_entry in step_sizes_dict.items():
        rec = ranges_dict.get(name, None)
        if rec is None or "weight_ranges" not in rec:
            continue

        w_step = _get_weight_steps(step_entry).to(device)
        w_ranges = rec["weight_ranges"].to(device)

        if w_ranges.numel() != w_step.numel():
            raise ValueError(
                f"[{name}] weight_ranges.shape={tuple(w_ranges.shape)} "
                f"!= weight_step_sizes.shape={tuple(w_step.shape)}"
            )

        bits_like_term = torch.log2(
            torch.clamp(w_ranges, min=eps) /
            torch.clamp(w_step, min=eps)
        )
        prior_loss = prior_loss + gamma * torch.sum(bits_like_term)

    return prior_loss


def compute_blockwise_bit_prior_no_sens(
    block_step_sizes_dict: Mapping[str, torch.Tensor | Tuple[torch.Tensor, torch.Tensor]],
    block_ranges_dict: Mapping[str, dict[str, torch.Tensor]],
    b_target: float = 4.0,
    sigma0: float = 0.5,
    prior_scale: float = 1.0,
    eps: float = 1e-8,
):
    """
    Blockwise Gaussian bit prior without sensitivity:

        L_prior
        =
        sum_c ( log2(R_c / s_c) - B_target )^2 / (2 sigma0^2)

    This is the control prior for isolating the sensitivity effect.
    """
    device = _infer_device_from_steps(block_step_sizes_dict)
    prior = torch.zeros((), device=device)

    sigma = torch.tensor(float(sigma0), device=device)
    sigma = torch.clamp(sigma, min=eps)

    for local_name, step_entry in block_step_sizes_dict.items():
        if local_name not in block_ranges_dict:
            continue
        if "weight_ranges" not in block_ranges_dict[local_name]:
            continue

        w_step = _get_weight_steps(step_entry).to(device)
        w_range = block_ranges_dict[local_name]["weight_ranges"].to(device)

        if w_step.numel() != w_range.numel():
            raise ValueError(
                f"[{local_name}] weight_ranges.shape={tuple(w_range.shape)} "
                f"!= weight_step_sizes.shape={tuple(w_step.shape)}"
            )

        bits = torch.log2(
            torch.clamp(w_range, min=eps) /
            torch.clamp(w_step, min=eps)
        )

        prior = prior + ((bits - b_target).pow(2) / (2.0 * sigma.pow(2))).sum()

    return prior_scale * prior


def compute_sensitivity_aware_bit_prior(
    block_step_sizes_dict: Mapping[str, torch.Tensor | Tuple[torch.Tensor, torch.Tensor]],
    block_ranges_dict: Mapping[str, dict[str, torch.Tensor]],
    block_sens_dict: Optional[Mapping[str, torch.Tensor]] = None,
    b_target: float = 4.0,
    sigma0: float = 0.5,
    alpha: float = 1.0,
    prior_scale: float = 1.0,
    eps: float = 1e-8,
):
    """
    Sensitivity-aware bitwidth prior:

        L_prior
        =
        sum_c ( log2(R_c / s_c) - B_target )^2 / (2 sigma_c^2)

    with
        sigma_c = sigma0 * (1 + alpha * sens_c)

    Notes:
    - all channels share the same center B_target
    - sensitivity only changes prior width, not center
    - block_sens_dict is assumed precomputed and normalized beforehand
    """
    device = _infer_device_from_steps(block_step_sizes_dict)
    prior = torch.zeros((), device=device)

    for local_name, step_entry in block_step_sizes_dict.items():
        if local_name not in block_ranges_dict:
            continue
        if "weight_ranges" not in block_ranges_dict[local_name]:
            continue

        w_step = _get_weight_steps(step_entry).to(device)
        w_range = block_ranges_dict[local_name]["weight_ranges"].to(device)

        if w_step.numel() != w_range.numel():
            raise ValueError(
                f"[{local_name}] weight_ranges.shape={tuple(w_range.shape)} "
                f"!= weight_step_sizes.shape={tuple(w_step.shape)}"
            )

        bits = torch.log2(
            torch.clamp(w_range, min=eps) /
            torch.clamp(w_step, min=eps)
        )

        if block_sens_dict is not None and local_name in block_sens_dict:
            sens = block_sens_dict[local_name].to(device)
            if sens.numel() == 1:
                sens = sens.expand_as(bits)
            if sens.shape != bits.shape:
                raise ValueError(
                    f"[{local_name}] sens.shape={tuple(sens.shape)} "
                    f"!= bits.shape={tuple(bits.shape)}"
                )
        else:
            sens = torch.zeros_like(bits)

        sigma = sigma0 * (1.0 + alpha * sens)
        sigma = torch.clamp(sigma, min=eps)

        prior = prior + ((bits - b_target).pow(2) / (2.0 * sigma.pow(2))).sum()

    return prior_scale * prior


def compute_sensitivity_aware_bit_prior_global(
    step_sizes_dict,
    ranges_dict,
    sens_dict=None,
    b_target: float = 4.0,
    sigma0: float = 0.5,
    alpha: float = 1.0,
    prior_scale: float = 1.0,
    eps: float = 1e-8,
):
    """
    Global SAPQ sensitivity-aware prior.
    """
    return compute_sensitivity_aware_bit_prior(
        block_step_sizes_dict=step_sizes_dict,
        block_ranges_dict=ranges_dict,
        block_sens_dict=sens_dict,
        b_target=b_target,
        sigma0=sigma0,
        alpha=alpha,
        prior_scale=prior_scale,
        eps=eps,
    )


def compute_prior_by_mode(
    prior_mode: str,
    step_sizes_dict,
    ranges_dict,
    sens_dict=None,
    gamma: float = 0.005,
    b_target: float = 4.0,
    sigma0: float = 0.5,
    alpha: float = 1.0,
    prior_scale: float = 1.0,
    eps: float = 1e-8,
):
    """
    Unified prior dispatcher.

    prior_mode:
        - "ppq"
        - "block_no_sens"
        - "block_sens"
    """
    if prior_mode == "ppq":
        return compute_ppq_prior(
            step_sizes_dict=step_sizes_dict,
            ranges_dict=ranges_dict,
            gamma=gamma,
            eps=eps,
        )

    if prior_mode == "block_no_sens":
        return compute_blockwise_bit_prior_no_sens(
            block_step_sizes_dict=step_sizes_dict,
            block_ranges_dict=ranges_dict,
            b_target=b_target,
            sigma0=sigma0,
            prior_scale=prior_scale,
            eps=eps,
        )

    if prior_mode == "block_sens":
        return compute_sensitivity_aware_bit_prior(
            block_step_sizes_dict=step_sizes_dict,
            block_ranges_dict=ranges_dict,
            block_sens_dict=sens_dict,
            b_target=b_target,
            sigma0=sigma0,
            alpha=alpha,
            prior_scale=prior_scale,
            eps=eps,
        )

    raise ValueError(
        f"Unknown prior_mode='{prior_mode}'. "
        f"Expected one of: 'ppq', 'block_no_sens', 'block_sens'."
    )


# ---------------------------------------------------------------------
# full SAPQ objectives
# ---------------------------------------------------------------------

def compute_sapq_loss_with_prior_network(
    model,
    step_sizes_dict: Mapping[str, torch.Tensor | Tuple[torch.Tensor, torch.Tensor]],
    frozen_batches,
    clean_net_outputs,
    batch_idx: int,
    ranges_dict: Mapping[str, dict[str, torch.Tensor]],
    sens_dict: Optional[Mapping[str, torch.Tensor]] = None,
    num_mc_samples: int = 10,
    eta: float = 1e-4,
    prior_mode: str = "block_sens",
    gamma: float = 0.005,
    b_target: float = 4.0,
    sigma0: float = 0.5,
    alpha: float = 1.0,
    prior_scale: float = 1.0,
    device: str | torch.device = "cuda",
):
    """
    Network-wise likelihood + selectable prior.
    Old temporary replacement version.
    """
    like_loss = compute_mc_negative_loglikelihood_network(
        model=model,
        step_sizes_dict=step_sizes_dict,
        frozen_batches=frozen_batches,
        clean_net_outputs=clean_net_outputs,
        batch_idx=batch_idx,
        num_mc_samples=num_mc_samples,
        eta=eta,
        device=device,
    )

    prior_loss = compute_prior_by_mode(
        prior_mode=prior_mode,
        step_sizes_dict=step_sizes_dict,
        ranges_dict=ranges_dict,
        sens_dict=sens_dict,
        gamma=gamma,
        b_target=b_target,
        sigma0=sigma0,
        alpha=alpha,
        prior_scale=prior_scale,
    )
    prior_weight = 1e-10

    total_loss = like_loss + prior_weight * prior_loss
    return total_loss, like_loss, prior_loss


def compute_sapq_loss_with_prior_global(
    model,
    step_sizes_dict,
    frozen_batches,
    clean_net_outputs,
    ranges_dict,
    sens_dict=None,
    batch_idx: int = 0,
    num_mc_samples: int = 10,
    eta: float = 1e-4,
    prior_mode: str = "block_sens",
    gamma: float = 0.005,
    b_target: float = 4.0,
    sigma0: float = 0.5,
    alpha: float = 1.0,
    prior_scale: float = 1.0,
    device: str | torch.device = "cuda",
):
    """
    Global network-wise SAPQ objective:

        total = network-wise likelihood + selectable prior

    - optimize all channel step sizes together
    - likelihood uses final model output
    """
    like_loss = compute_mc_negative_loglikelihood_network_global(
        model=model,
        step_sizes_dict=step_sizes_dict,
        frozen_batches=frozen_batches,
        clean_net_outputs=clean_net_outputs,
        batch_idx=batch_idx,
        num_mc_samples=num_mc_samples,
        eta=eta,
        device=device,
    )

    prior_loss = compute_prior_by_mode(
        prior_mode=prior_mode,
        step_sizes_dict=step_sizes_dict,
        ranges_dict=ranges_dict,
        sens_dict=sens_dict,
        gamma=gamma,
        b_target=b_target,
        sigma0=sigma0,
        alpha=alpha,
        prior_scale=prior_scale,
    )

    prior_weight = 1
    total_loss = like_loss + prior_weight * prior_loss
    return total_loss, like_loss, prior_loss


def compute_sapq_loss_with_prior(
    block,
    block_step_sizes_dict: Mapping[str, torch.Tensor | Tuple[torch.Tensor, torch.Tensor]],
    cached_block_inputs,
    cached_block_outputs,
    cached_block_grads,
    batch_idx: int,
    block_ranges_dict: Mapping[str, dict[str, torch.Tensor]],
    block_sens_dict: Optional[Mapping[str, torch.Tensor]] = None,
    num_mc_samples: int = 10,
    prior_mode: str = "block_sens",
    gamma: float = 0.005,
    b_target: float = 4.0,
    sigma0: float = 0.5,
    alpha: float = 1.0,
    prior_scale: float = 1.0,
    device: str | torch.device = "cuda",
):
    """
    Block-wise SAPQ objective for one cached batch:

        total = blockwise likelihood + selectable prior
    """
    like_loss = compute_block_mc_negative_loglikelihood(
        block=block,
        block_step_sizes_dict=block_step_sizes_dict,
        cached_block_inputs=cached_block_inputs,
        cached_block_outputs=cached_block_outputs,
        cached_block_grads=cached_block_grads,
        batch_idx=batch_idx,
        num_mc_samples=num_mc_samples,
        device=device,
    )

    prior_loss = compute_prior_by_mode(
        prior_mode=prior_mode,
        step_sizes_dict=block_step_sizes_dict,
        ranges_dict=block_ranges_dict,
        sens_dict=block_sens_dict,
        gamma=gamma,
        b_target=b_target,
        sigma0=sigma0,
        alpha=alpha,
        prior_scale=prior_scale,
    )

    prior_weight = 1e-4
    total_loss = like_loss + prior_weight * prior_loss
    return total_loss, like_loss, prior_loss