# SAPQ/sapq_loss.py
"""
Structural-Aware Probabilistic Quantization (SAPQ) loss utilities.

This file implements the core SAPQ objective:

1) Block-wise probabilistic likelihood
   - optimize learnable channel-wise weight step sizes
   - inject PPQ-style additive uniform noise into weights
   - run the whole block forward
   - compare noisy block output with cached FP block output
   - support curvature weighting:
       * mse          : identity geometry
       * fisher_diag  : diagonal Fisher weighting
       * fisher_full  : BRECQ-style fuller approximation

2) Sensitivity-aware bitwidth prior
   - prior is centered at one shared target bitwidth B_target
   - sensitivity only changes prior width sigma_{b,c}
   - more sensitive channels => looser prior
   - less sensitive channels => tighter prior

Important design note
---------------------
This file is written for BLOCK-BY-BLOCK SAPQ.

So the trainer should pass block-local dictionaries:
    block_step_sizes_dict
    block_ranges_dict
    block_sens_dict

where the keys are local module names inside the current block
(e.g. names from block.named_modules()).

Typical usage inside a future trainer:
    total, like, prior = compute_sapq_loss_with_prior(
        block=block,
        block_step_sizes_dict=block_step_sizes_dict,
        cached_block_inputs=cached_inputs,
        cached_block_outputs=cached_outputs,
        cached_block_grads=cached_grads,
        block_ranges_dict=block_ranges_dict,
        block_sens_dict=block_sens_dict,
        batch_idx=batch_idx,
        num_mc_samples=cfg.num_mc_samples,
        rec_loss="fisher_diag",
        b_target=cfg.target_bits,
        sigma0=cfg.sigma0,
        alpha=cfg.alpha,
        prior_scale=cfg.prior_scale,
        device=device,
    )
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch

from PPQ.noise import add_quantization_noise
from BRECQ.quant.quant_layer import QuantModule


# ---------------------------------------------------------------------
# Small local helpers
# ---------------------------------------------------------------------

def _move_block_args_to_device(block_args, device: torch.device):
    if torch.is_tensor(block_args):
        return block_args.to(device)
    if isinstance(block_args, tuple):
        return tuple(_move_block_args_to_device(x, device) for x in block_args)
    if isinstance(block_args, list):
        return [_move_block_args_to_device(x, device) for x in block_args]
    if isinstance(block_args, dict):
        return {k: _move_block_args_to_device(v, device) for k, v in block_args.items()}
    return block_args


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


def _named_quantmodules(block) -> Dict[str, QuantModule]:
    return {
        name: module
        for name, module in block.named_modules()
        if isinstance(module, QuantModule)
    }


def _reduce_dims(x: torch.Tensor) -> Tuple[int, ...]:
    return tuple(range(1, x.dim()))


# ---------------------------------------------------------------------
# Curvature / reconstruction energy
# ---------------------------------------------------------------------

def compute_block_reconstruction_energy(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    grad: Optional[torch.Tensor] = None,
    rec_loss: str = "fisher_diag",
) -> torch.Tensor:
    """
    Returns per-sample reconstruction energy E_i.

    Shapes:
        pred, tgt, grad: [B, ...]
    Output:
        energy: [B]

    Supported:
        - mse
        - fisher_diag
        - fisher_full
    """
    if pred.shape != tgt.shape:
        raise ValueError(
            f"pred.shape={tuple(pred.shape)} != tgt.shape={tuple(tgt.shape)}"
        )

    rdims = _reduce_dims(pred)
    delta = pred - tgt

    if rec_loss == "mse":
        # Identity geometry
        energy = delta.pow(2).sum(dim=rdims)
        return energy

    if rec_loss == "fisher_diag":
        if grad is None:
            raise ValueError("grad must not be None when rec_loss='fisher_diag'")
        if grad.shape != pred.shape:
            raise ValueError(
                f"grad.shape={tuple(grad.shape)} != pred.shape={tuple(pred.shape)}"
            )
        energy = (delta.pow(2) * grad.pow(2)).sum(dim=rdims)
        return energy

    if rec_loss == "fisher_full":
        if grad is None:
            raise ValueError("grad must not be None when rec_loss='fisher_full'")
        if grad.shape != pred.shape:
            raise ValueError(
                f"grad.shape={tuple(grad.shape)} != pred.shape={tuple(pred.shape)}"
            )

        # BRECQ-style fuller approximation, converted to per-sample energy.
        # Original code uses:
        #   a = |pred - tgt|
        #   g = |grad|
        #   dot = sum(a * g)
        #   rec_loss = mean(dot * a * g) / 100
        #
        # Per sample version:
        #   energy_i = dot_i * mean(a_i * g_i) / 100
        a = delta.abs()
        g = grad.abs()
        ag = a * g
        dot = ag.sum(dim=rdims)          # [B]
        mean_ag = ag.mean(dim=rdims)     # [B]
        energy = (dot * mean_ag) / 100.0
        return energy

    raise ValueError(f"Unsupported rec_loss: {rec_loss}")


# ---------------------------------------------------------------------
# Temporary noisy-weight injection
# ---------------------------------------------------------------------

@contextmanager
def temporary_block_noisy_weights(
    block,
    block_step_sizes_dict: Mapping[str, torch.Tensor | Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
):
    """
    Temporarily replace QuantModule.org_weight inside the block by:
        W_noisy = W + step * U(-1/2, 1/2)

    Important:
    - This uses the FP path of QuantModule (org_weight), not AdaRound.
    - It is meant for SAPQ likelihood optimization over step sizes.
    - It preserves the original block structure.
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
                    f"[{local_name}] step size channels ({w_step.numel()}) "
                    f"!= weight out_channels/out_features ({w_clean.shape[0]})"
                )

            saved_org_weights[local_name] = module.org_weight
            w_noisy = add_quantization_noise(w_clean, w_step, channel_axis=0)
            module.org_weight = w_noisy

        yield

    finally:
        for local_name, original_weight in saved_org_weights.items():
            name2module[local_name].org_weight = original_weight


# ---------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------

def compute_block_mc_loglikelihood_single_batch(
    block,
    block_step_sizes_dict: Mapping[str, torch.Tensor | Tuple[torch.Tensor, torch.Tensor]],
    cached_block_inputs,
    cached_block_outputs,
    batch_idx: int,
    cached_block_grads=None,
    num_mc_samples: int = 10,
    rec_loss: str = "fisher_diag",
    device: str | torch.device = "cuda",
):
    """
    SAPQ block-wise likelihood for one cached batch.

    Implements:
        log( (1/M) sum_j exp( -1/2 * E_{i,j} ) )

    where E_{i,j} is the block reconstruction energy for sample i and MC draw j.

    Args:
        block:
            current Poseidon block
        block_step_sizes_dict:
            block-local step size dict keyed by local QuantModule names
        cached_block_inputs:
            list of cached block input tuples
        cached_block_outputs:
            list of cached FP block output tensors
        batch_idx:
            which cached batch to use
        cached_block_grads:
            optional list of cached gradient tensors, required for fisher_* modes
        num_mc_samples:
            number of MC noisy forward samples
        rec_loss:
            "mse" | "fisher_diag" | "fisher_full"

    Returns:
        scalar likelihood term to MINIMIZE:
            -(sum_i log mean_j exp(-1/2 * E_{i,j}))
    """
    device = torch.device(device)
    block = block.to(device).eval()

    if batch_idx < 0 or batch_idx >= len(cached_block_inputs):
        raise IndexError(f"batch_idx={batch_idx} out of range for cached_block_inputs")
    if batch_idx < 0 or batch_idx >= len(cached_block_outputs):
        raise IndexError(f"batch_idx={batch_idx} out of range for cached_block_outputs")

    block.set_quant_state(False, False)

    cur_inp = _move_block_args_to_device(cached_block_inputs[batch_idx], device)
    tgt = cached_block_outputs[batch_idx].to(device)

    if rec_loss != "mse":
        if cached_block_grads is None:
            raise ValueError("cached_block_grads must not be None for fisher-based loss")
        grad = cached_block_grads[batch_idx].to(device)
    else:
        grad = None

    score_list = []

    for _ in range(num_mc_samples):
        with temporary_block_noisy_weights(
            block=block,
            block_step_sizes_dict=block_step_sizes_dict,
            device=device,
        ):
            pred = block(*cur_inp)
            pred = _get_reconstruction_output(pred)

        energy = compute_block_reconstruction_energy(
            pred=pred,
            tgt=tgt,
            grad=grad,
            rec_loss=rec_loss,
        )  # [B]
        score = -0.5 * energy
        score_list.append(score)

    # [M, B]
    scores = torch.stack(score_list, dim=0)

    # log(1/M sum_j exp(score_j)) = logsumexp(scores) - log(M)
    log_prob_per_sample = torch.logsumexp(scores, dim=0) - torch.log(
        torch.tensor(float(num_mc_samples), device=device)
    )

    # We minimize negative log-likelihood
    nll = -log_prob_per_sample.sum()
    return nll


# ---------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------

def compute_sensitivity_aware_bit_prior(
    block_step_sizes_dict: Mapping[str, torch.Tensor | Tuple[torch.Tensor, torch.Tensor]],
    block_ranges_dict: Mapping[str, Dict[str, torch.Tensor]],
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
        sum_{c}
        ( log2(R_c / s_c) - B_target )^2 / (2 sigma_c^2)

    with
        sigma_c = sigma0 * (1 + alpha * sens_tilde_c)

    Notes:
    - All channels share the same center B_target.
    - Sensitivity changes only the width, not the center.
    - block_sens_dict is assumed already normalized (e.g. to [0,1] or similar).
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
            torch.clamp(w_range, min=eps) / torch.clamp(w_step, min=eps)
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
# Full SAPQ objective
# ---------------------------------------------------------------------

def compute_sapq_loss_with_prior(
    block,
    block_step_sizes_dict: Mapping[str, torch.Tensor | Tuple[torch.Tensor, torch.Tensor]],
    cached_block_inputs,
    cached_block_outputs,
    batch_idx: int,
    block_ranges_dict: Mapping[str, Dict[str, torch.Tensor]],
    block_sens_dict: Optional[Mapping[str, torch.Tensor]] = None,
    cached_block_grads=None,
    num_mc_samples: int = 10,
    rec_loss: str = "fisher_diag",
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
        likelihood = negative block-wise log Monte Carlo likelihood
        prior      = sensitivity-aware bitwidth prior
    """
    like_loss = compute_block_mc_loglikelihood_single_batch(
        block=block,
        block_step_sizes_dict=block_step_sizes_dict,
        cached_block_inputs=cached_block_inputs,
        cached_block_outputs=cached_block_outputs,
        batch_idx=batch_idx,
        cached_block_grads=cached_block_grads,
        num_mc_samples=num_mc_samples,
        rec_loss=rec_loss,
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