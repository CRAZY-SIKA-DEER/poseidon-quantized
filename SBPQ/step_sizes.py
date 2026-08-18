"""
Generic step-size utilities for SBPQ.

This module handles:

1. Finding quantizable Linear layers.
2. Storing one learnable step-size tensor per layer.
3. Converting between step size and effective bitwidth.
4. Initializing step sizes from a target bitwidth.
5. Clamping step sizes so the effective bitwidth remains within bounds.

This file is model-independent.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Iterable

import torch
import torch.nn as nn


def find_quantizable_linear_layers(
    model: nn.Module,
) -> OrderedDict[str, nn.Linear]:
    """
    Find all Linear layers in a model.

    Returns:

        {
            layer_name: linear_module
        }
    """
    layers: OrderedDict[str, nn.Linear] = OrderedDict()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layers[name] = module

    return layers


def effective_bitwidth_from_step_size(
    quantization_range: torch.Tensor,
    step_size: torch.Tensor,
    minimum_step_size: float = 1e-12,
) -> torch.Tensor:
    """
    Calculate continuous effective bitwidth.

    Formula:

        B = log2(R / S)

    where:

        R = quantization range
        S = quantization step size
    """
    quantization_range = torch.as_tensor(
        quantization_range,
        device=step_size.device,
        dtype=step_size.dtype,
    )

    if torch.any(quantization_range <= 0):
        raise ValueError(
            "All quantization-range values must be positive."
        )

    safe_step_size = step_size.clamp_min(
        float(minimum_step_size)
    )

    return torch.log2(
        quantization_range / safe_step_size
    )


def step_size_from_bitwidth(
    quantization_range: torch.Tensor,
    bitwidth: float | torch.Tensor,
) -> torch.Tensor:
    """
    Convert effective bitwidth into step size.

    Formula:

        S = R / 2^B
    """
    quantization_range = torch.as_tensor(
        quantization_range,
    )

    bitwidth = torch.as_tensor(
        bitwidth,
        device=quantization_range.device,
        dtype=quantization_range.dtype,
    )

    if torch.any(quantization_range <= 0):
        raise ValueError(
            "All quantization-range values must be positive."
        )

    return quantization_range / torch.pow(
        torch.tensor(
            2.0,
            device=quantization_range.device,
            dtype=quantization_range.dtype,
        ),
        bitwidth,
    )


def get_step_size_bounds(
    quantization_range: torch.Tensor,
    minimum_bits: float,
    maximum_bits: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate valid step-size bounds.

    Because:

        S = R / 2^B

    the highest bitwidth gives the smallest step size, while the lowest
    bitwidth gives the largest step size.

    Returns:

        minimum_step_size:
            Corresponds to maximum_bits.

        maximum_step_size:
            Corresponds to minimum_bits.
    """
    if minimum_bits >= maximum_bits:
        raise ValueError(
            "minimum_bits must be smaller than maximum_bits."
        )

    minimum_step_size = step_size_from_bitwidth(
        quantization_range=quantization_range,
        bitwidth=maximum_bits,
    )

    maximum_step_size = step_size_from_bitwidth(
        quantization_range=quantization_range,
        bitwidth=minimum_bits,
    )

    return minimum_step_size, maximum_step_size


def clamp_step_size(
    step_size: torch.Tensor,
    quantization_range: torch.Tensor,
    minimum_bits: float,
    maximum_bits: float,
) -> torch.Tensor:
    """
    Clamp a step-size tensor so its bitwidth remains inside the allowed range.
    """
    quantization_range = torch.as_tensor(
        quantization_range,
        device=step_size.device,
        dtype=step_size.dtype,
    )

    minimum_step_size, maximum_step_size = get_step_size_bounds(
        quantization_range=quantization_range,
        minimum_bits=minimum_bits,
        maximum_bits=maximum_bits,
    )

    return torch.maximum(
        torch.minimum(
            step_size,
            maximum_step_size,
        ),
        minimum_step_size,
    )


def initialize_step_size(
    quantization_range: torch.Tensor,
    initial_bits: float,
    minimum_bits: float,
    maximum_bits: float,
) -> torch.Tensor:
    """
    Initialize a step-size tensor from an initial target bitwidth.
    """
    if not minimum_bits <= initial_bits <= maximum_bits:
        raise ValueError(
            "initial_bits must lie between minimum_bits "
            "and maximum_bits."
        )

    initial_step_size = step_size_from_bitwidth(
        quantization_range=quantization_range,
        bitwidth=initial_bits,
    )

    return initial_step_size.clone().detach()


class LearnableStepSizes(nn.Module):
    """
    Store one learnable step-size tensor for every quantized layer.

    The parameter names used by PyTorch cannot contain dots, so this class
    keeps a mapping between:

        original layer name
            encoder.layers.0.blocks.0.attention.query

    and:

        safe parameter name
            layer_0
    """

    def __init__(
        self,
        quantization_ranges: Mapping[str, torch.Tensor],
        initial_bits: float,
        minimum_bits: float,
        maximum_bits: float,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if len(quantization_ranges) == 0:
            raise ValueError(
                "quantization_ranges cannot be empty."
            )

        if not minimum_bits <= initial_bits <= maximum_bits:
            raise ValueError(
                "initial_bits must lie inside the allowed bitwidth range."
            )

        self.initial_bits = float(initial_bits)
        self.minimum_bits = float(minimum_bits)
        self.maximum_bits = float(maximum_bits)

        device = torch.device(device)

        self._layer_name_to_key: dict[str, str] = {}
        self._key_to_layer_name: dict[str, str] = {}

        for index, (
            layer_name,
            quantization_range,
        ) in enumerate(quantization_ranges.items()):
            safe_key = f"layer_{index}"

            quantization_range = torch.as_tensor(
                quantization_range,
                device=device,
                dtype=dtype,
            )

            if torch.any(quantization_range <= 0):
                raise ValueError(
                    f"Layer '{layer_name}' contains a non-positive "
                    "quantization range."
                )

            initial_step_size = initialize_step_size(
                quantization_range=quantization_range,
                initial_bits=self.initial_bits,
                minimum_bits=self.minimum_bits,
                maximum_bits=self.maximum_bits,
            )

            self.register_parameter(
                safe_key,
                nn.Parameter(initial_step_size),
            )

            self.register_buffer(
                f"{safe_key}_range",
                quantization_range.detach().clone(),
            )

            self._layer_name_to_key[layer_name] = safe_key
            self._key_to_layer_name[safe_key] = layer_name

    def get_step_sizes(
        self,
    ) -> OrderedDict[str, torch.Tensor]:
        """
        Return:

            {
                original_layer_name: learnable_step_size
            }
        """
        return OrderedDict(
            (
                layer_name,
                getattr(
                    self,
                    safe_key,
                ),
            )
            for layer_name, safe_key
            in self._layer_name_to_key.items()
        )

    def get_quantization_ranges(
        self,
    ) -> OrderedDict[str, torch.Tensor]:
        """
        Return:

            {
                original_layer_name: quantization_range
            }
        """
        return OrderedDict(
            (
                layer_name,
                getattr(
                    self,
                    f"{safe_key}_range",
                ),
            )
            for layer_name, safe_key
            in self._layer_name_to_key.items()
        )

    def get_effective_bitwidths(
        self,
    ) -> OrderedDict[str, torch.Tensor]:
        """
        Calculate the current effective bitwidth of every layer.
        """
        step_sizes = self.get_step_sizes()
        quantization_ranges = self.get_quantization_ranges()

        return OrderedDict(
            (
                layer_name,
                effective_bitwidth_from_step_size(
                    quantization_range=quantization_ranges[layer_name],
                    step_size=step_size,
                ),
            )
            for layer_name, step_size in step_sizes.items()
        )

    @torch.no_grad()
    def clamp_(
        self,
    ) -> None:
        """
        Clamp all learnable step sizes in place.

        Call this after every optimizer step.
        """
        quantization_ranges = self.get_quantization_ranges()

        for layer_name, safe_key in self._layer_name_to_key.items():
            step_size = getattr(
                self,
                safe_key,
            )

            clamped_step_size = clamp_step_size(
                step_size=step_size,
                quantization_range=quantization_ranges[layer_name],
                minimum_bits=self.minimum_bits,
                maximum_bits=self.maximum_bits,
            )

            step_size.copy_(
                clamped_step_size
            )

    @torch.no_grad()
    def reset_to_bitwidth(
        self,
        bitwidth: float,
    ) -> None:
        """
        Reset every layer to the same selected bitwidth.
        """
        if not self.minimum_bits <= bitwidth <= self.maximum_bits:
            raise ValueError(
                "bitwidth must lie inside the allowed range."
            )

        quantization_ranges = self.get_quantization_ranges()

        for layer_name, safe_key in self._layer_name_to_key.items():
            step_size = getattr(
                self,
                safe_key,
            )

            new_step_size = step_size_from_bitwidth(
                quantization_range=quantization_ranges[layer_name],
                bitwidth=bitwidth,
            )

            step_size.copy_(
                new_step_size
            )

    def extra_repr(self) -> str:
        """
        Text shown when printing the module.
        """
        return (
            f"layers={len(self._layer_name_to_key)}, "
            f"initial_bits={self.initial_bits}, "
            f"minimum_bits={self.minimum_bits}, "
            f"maximum_bits={self.maximum_bits}"
        )


def collect_trainable_step_size_parameters(
    step_size_module: LearnableStepSizes,
) -> Iterable[nn.Parameter]:
    """
    Return the parameters that should be given to the optimizer.
    """
    return step_size_module.parameters()