# SAPQ/sapq_loss.py
from __future__ import annotations

from contextlib import contextmanager
from typing import Mapping, Optional, Tuple

import torch

from PPQ.noise import add_quantization_noise
from BRECQ.quant.quant_layer import QuantModule


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

    Implements the SAPQ likelihood geometry:
        E_i = sum( (pred - tgt)^2 * grad^2 )

    where grad is the gradient of the FINAL output loss back to the
    current block output (cached beforehand).

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


# ---------------------------------------------------------------------
# likelihood
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

    Args:
        block:
            current block
        block_step_sizes_dict:
            block-local step sizes keyed by local QuantModule names
        cached_block_inputs:
            list of cached block input tuples
        cached_block_outputs:
            list of cached FP block outputs
        cached_block_grads:
            list of cached final-loss gradients wrt block output
        batch_idx:
            which cached batch to use

    Returns:
        scalar negative log-likelihood to MINIMIZE
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

    # [M, B]
    scores = torch.stack(score_list, dim=0)

    log_prob_per_sample = torch.logsumexp(scores, dim=0) - torch.log(
        torch.tensor(float(num_mc_samples), device=device)
    )

    nll = -log_prob_per_sample.sum()
    return nll


# ---------------------------------------------------------------------
# prior
# ---------------------------------------------------------------------

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
        sigma_c = sigma0 * (1 + alpha * sens_tilde_c)

    Notes:
    - all channels share the same center B_target
    - sensitivity only changes prior width, not center
    - block_sens_dict is assumed precomputed and normalized beforehand
    """
    device = None
    for step_entry in block_step_sizes_dict.values():
        w_step = _get_weight_steps(step_entry)
        if w_step is not None:
            device = w_step.device
            break
    if device is None:
        device = torch.device("cpu")

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


# ---------------------------------------------------------------------
# full SAPQ objective
# ---------------------------------------------------------------------

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
    b_target: float = 4.0,
    sigma0: float = 0.5,
    alpha: float = 1.0,
    prior_scale: float = 1.0,
    device: str | torch.device = "cuda",
):
    """
    Full SAPQ objective for one cached batch:

        total = likelihood + prior

    where:
        likelihood = block-wise MC negative log-likelihood
                     with Fisher-diagonal geometry
        prior      = sensitivity-aware bitwidth prior
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

    prior_loss = compute_sensitivity_aware_bit_prior(
        block_step_sizes_dict=block_step_sizes_dict,
        block_ranges_dict=block_ranges_dict,
        block_sens_dict=block_sens_dict,
        b_target=b_target,
        sigma0=sigma0,
        alpha=alpha,
        prior_scale=prior_scale,
    )

    total_loss = like_loss + prior_loss
    return total_loss, like_loss, prior_loss