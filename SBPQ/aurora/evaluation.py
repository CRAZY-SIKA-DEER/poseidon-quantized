"""Evaluation utilities for Aurora."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from SBPQ.aurora.data_utils import iter_output_tensors, move_batch_to_device
from SBPQ.aurora.likelihood import make_noisy_linear_hook
from SBPQ.step_sizes import step_size_from_bitwidth


def latitude_weights(lat: torch.Tensor, device: torch.device) -> torch.Tensor:
    weights = torch.cos(torch.deg2rad(lat.to(device=device, dtype=torch.float32))).clamp_min(0)
    return weights / weights.mean().clamp_min(1e-12)


def _spatial_lat_weights(reference: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
    weights = latitude_weights(lat, reference.device).to(dtype=reference.dtype)
    shape = [1] * reference.ndim
    shape[-2] = weights.numel()
    return weights.reshape(shape)


def weighted_rmse(prediction: torch.Tensor, target: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
    weights = _spatial_lat_weights(prediction, lat)
    return torch.sqrt(((prediction - target).pow(2) * weights).mean())


def anomaly_correlation(prediction: torch.Tensor, target: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
    weights = _spatial_lat_weights(prediction, lat)
    pred_anom = prediction - prediction.mean(dim=(-2, -1), keepdim=True)
    targ_anom = target - target.mean(dim=(-2, -1), keepdim=True)
    numerator = (weights * pred_anom * targ_anom).sum()
    denominator = torch.sqrt((weights * pred_anom.pow(2)).sum() * (weights * targ_anom.pow(2)).sum())
    return numerator / denominator.clamp_min(1e-12)


def evaluate_output(prediction, target) -> dict[str, float]:
    metrics = {}
    lat = prediction.metadata.lat
    rmses = []
    accs = []
    for group_name, variable_name, pred_tensor in iter_output_tensors(prediction):
        target_group = target.surf_vars if group_name == "surf" else target.atmos_vars
        target_tensor = target_group[variable_name].to(pred_tensor.device, pred_tensor.dtype)
        rmse = weighted_rmse(pred_tensor, target_tensor, lat)
        acc = anomaly_correlation(pred_tensor, target_tensor, lat)
        key = f"{group_name}_{variable_name}"
        metrics[f"{key}_weighted_rmse"] = float(rmse.detach().cpu())
        metrics[f"{key}_acc"] = float(acc.detach().cpu())
        rmses.append(rmse.detach())
        accs.append(acc.detach())
    metrics["mean_weighted_rmse"] = float(torch.stack(rmses).mean().cpu())
    metrics["mean_acc"] = float(torch.stack(accs).mean().cpu())
    return metrics


@contextmanager
def fixed_bit_weight_noise(model: nn.Module, ranges_dict: Mapping, bits: float):
    handles = []
    modules = dict(model.named_modules())
    try:
        for layer_name, values in ranges_dict.items():
            module = modules[layer_name]
            step_size = step_size_from_bitwidth(values["weight_ranges"].to(module.weight.device), bits)
            handles.append(module.register_forward_hook(make_noisy_linear_hook(step_size)))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def learned_step_weight_noise(model: nn.Module, step_sizes: Mapping[str, torch.Tensor]):
    handles = []
    modules = dict(model.named_modules())
    try:
        for layer_name, step_size in step_sizes.items():
            module = modules[layer_name]
            handles.append(module.register_forward_hook(make_noisy_linear_hook(step_size.to(module.weight.device))))
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def evaluate_model(model, batch, target, device: str | torch.device = "cuda") -> dict[str, float]:
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()
    prediction = model(move_batch_to_device(batch, device))
    return evaluate_output(prediction, move_batch_to_device(target, device))


@torch.no_grad()
def evaluate_fixed_bits(model, batch, target, ranges_dict, bits: float, device: str | torch.device = "cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()
    with fixed_bit_weight_noise(model, ranges_dict, bits):
        prediction = model(move_batch_to_device(batch, device))
    return evaluate_output(prediction, move_batch_to_device(target, device))


@torch.no_grad()
def evaluate_learned_steps(model, batch, target, step_sizes, device: str | torch.device = "cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()
    with learned_step_weight_noise(model, step_sizes):
        prediction = model(move_batch_to_device(batch, device))
    return evaluate_output(prediction, move_batch_to_device(target, device))
