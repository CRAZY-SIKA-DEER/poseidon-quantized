"""Network-wise Monte Carlo likelihood for Aurora SBPQ."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from SBPQ.aurora.data_utils import (
    batch_squared_error_per_sample,
    move_batch_to_device,
)
from SBPQ.noise import add_weight_quantization_noise


def find_valid_noisy_linear_layers(
    model: nn.Module,
    step_sizes: Mapping[str, torch.Tensor],
) -> dict[str, nn.Linear]:
    modules = dict(model.named_modules())
    valid = {}
    for layer_name, step_size in step_sizes.items():
        module = modules.get(layer_name)
        if module is None:
            raise KeyError(f"Layer {layer_name} does not exist.")
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Layer {layer_name} is {type(module).__name__}, not Linear.")
        if step_size.numel() != module.weight.shape[0]:
            raise ValueError(f"Step-size shape mismatch for layer {layer_name}.")
        valid[layer_name] = module
    return valid


def make_noisy_linear_hook(step_size: torch.Tensor):
    def hook(module: nn.Linear, inputs, original_output):
        return F.linear(
            inputs[0],
            add_weight_quantization_noise(module.weight, step_size, channel_axis=0),
            module.bias,
        )
    return hook


def run_aurora_with_weight_noise(
    model: nn.Module,
    batch,
    step_sizes: Mapping[str, torch.Tensor],
    device: str | torch.device,
    autocast_dtype: torch.dtype | None = None,
):
    valid_layers = find_valid_noisy_linear_layers(model, step_sizes)
    handles = []
    try:
        for layer_name, module in valid_layers.items():
            handles.append(module.register_forward_hook(make_noisy_linear_hook(step_sizes[layer_name])))
        enabled = autocast_dtype is not None and torch.device(device).type == "cuda"
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=enabled):
            return model(move_batch_to_device(batch, device))
    finally:
        for handle in handles:
            handle.remove()


class AuroraNetworkLikelihood(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        frozen_batches: Sequence,
        clean_network_outputs: Sequence,
        num_mc_samples: int = 1,
        eta: float = 1e-3,
        device: str | torch.device = "cuda",
        autocast_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_mc_samples <= 0:
            raise ValueError("num_mc_samples must be positive.")
        if eta <= 0:
            raise ValueError("eta must be positive.")
        self.model = model
        self.frozen_batches = list(frozen_batches)
        self.clean_network_outputs = list(clean_network_outputs)
        self.num_mc_samples = int(num_mc_samples)
        self.eta = float(eta)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.autocast_dtype = autocast_dtype

    def forward(self, step_sizes: Mapping[str, torch.Tensor], batch_index: int) -> torch.Tensor:
        if not 0 <= batch_index < len(self.frozen_batches):
            raise IndexError("batch_index is out of range.")
        clean_output = move_batch_to_device(self.clean_network_outputs[batch_index], self.device)
        scores = []
        for _ in range(self.num_mc_samples):
            noisy_output = run_aurora_with_weight_noise(
                self.model,
                self.frozen_batches[batch_index],
                step_sizes,
                self.device,
                autocast_dtype=self.autocast_dtype,
            )
            squared_error = batch_squared_error_per_sample(noisy_output, clean_output)
            scores.append(-squared_error / (2.0 * self.eta))
        stacked = torch.stack(scores, dim=0)
        log_prob = torch.logsumexp(stacked, dim=0) - torch.log(
            torch.tensor(float(self.num_mc_samples), device=self.device, dtype=stacked.dtype)
        )
        return -log_prob.sum()
