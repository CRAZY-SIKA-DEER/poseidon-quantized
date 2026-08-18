"""
Poseidon evaluation utilities for SBPQ.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from scOT.metrics import lp_error, relative_lp_error

from SBPQ.poseidon.poseidon_utils import move_poseidon_batch_to_device


def build_channel_parameter_weights(
    model: nn.Module,
    layer_names,
) -> dict[str, torch.Tensor]:
    """
    For Linear layers, each output channel controls in_features weights.
    """
    name_to_module = dict(model.named_modules())
    weights = {}

    for layer_name in layer_names:
        module = name_to_module.get(layer_name)
        if not isinstance(module, nn.Linear):
            continue

        weights[layer_name] = torch.full(
            (module.out_features,),
            float(module.in_features),
            dtype=torch.float32,
        )

    return weights


def compute_dynamic_weight_step_sizes(
    model: nn.Module,
    layer_names,
    num_bits: int,
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    """
    Build fixed-bit symmetric max-absolute weight step sizes.
    """
    if num_bits <= 0:
        raise ValueError("num_bits must be positive.")

    device = torch.device(device)
    denominator = float(2**num_bits - 1)
    name_to_module = dict(model.named_modules())
    step_sizes = {}

    with torch.no_grad():
        for layer_name in layer_names:
            module = name_to_module.get(layer_name)
            if not isinstance(module, nn.Linear):
                continue

            weight = module.weight.detach().to(device)
            flattened = weight.reshape(weight.shape[0], -1)
            max_abs = flattened.abs().max(dim=1).values
            step_sizes[layer_name] = (
                2.0 * max_abs / denominator
            ).detach().cpu()

    return step_sizes


def _extract_weight_step(step_entry) -> torch.Tensor:
    if isinstance(step_entry, (tuple, list)):
        return step_entry[0]
    return step_entry


def _make_weight_quantization_hook(
    step_size: torch.Tensor,
):
    def hook(
        module: nn.Linear,
        inputs,
        output,
    ):
        if len(inputs) == 0:
            raise RuntimeError(
                "Linear layer received no input tensor."
            )

        layer_input = inputs[0]
        weight = module.weight
        step = step_size.to(
            device=weight.device,
            dtype=weight.dtype,
        ).reshape(-1, 1)

        quantized_weight = (
            torch.round(weight / step.clamp_min(1e-12))
            * step.clamp_min(1e-12)
        )

        return F.linear(
            layer_input,
            quantized_weight,
            module.bias,
        )

    return hook


def evaluate_poseidon_with_weight_steps(
    model: nn.Module,
    dataloader,
    weight_step_sizes: Mapping[str, torch.Tensor],
    layer_names,
    device: str | torch.device,
) -> dict[str, float]:
    """
    Evaluate Poseidon with optional weight-only fake quantization.
    """
    device = torch.device(device)
    model = model.to(device).eval()
    name_to_module = dict(model.named_modules())

    handles = []

    for layer_name in layer_names:
        if layer_name not in weight_step_sizes:
            continue

        module = name_to_module.get(layer_name)
        if not isinstance(module, nn.Linear):
            continue

        step_size = _extract_weight_step(
            weight_step_sizes[layer_name]
        )

        if isinstance(step_size, nn.Parameter):
            step_size = step_size.detach()

        handles.append(
            module.register_forward_hook(
                _make_weight_quantization_hook(step_size)
            )
        )

    loader = dataloader() if callable(dataloader) else dataloader
    l1_values = []
    relative_l1_values = []

    try:
        with torch.no_grad():
            for batch in loader:
                (
                    pixel_values,
                    time,
                    pixel_mask,
                    labels,
                ) = move_poseidon_batch_to_device(
                    batch=batch,
                    device=device,
                )

                if labels is None:
                    continue

                outputs = model(
                    pixel_values=pixel_values,
                    time=time,
                    pixel_mask=pixel_mask,
                    labels=labels,
                )

                prediction = outputs.output

                prediction_np = prediction.detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()

                l1_values.append(
                    float(lp_error(prediction_np, labels_np, p=1).mean())
                )
                relative_l1_values.append(
                    float(
                        relative_lp_error(
                            prediction_np,
                            labels_np,
                            p=1,
                            return_percent=True,
                        ).mean()
                    )
                )

    finally:
        for handle in handles:
            handle.remove()

    if len(l1_values) == 0:
        return {
            "l1": float("nan"),
            "rel_l1": float("nan"),
        }

    return {
        "l1": float(sum(l1_values) / len(l1_values)),
        "rel_l1": float(
            sum(relative_l1_values) / len(relative_l1_values)
        ),
    }
