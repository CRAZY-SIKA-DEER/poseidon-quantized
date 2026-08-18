"""
Model-agnostic quantization-noise utilities for SBPQ.

The stochastic quantization relaxation is:

    epsilon = S * nu
    nu ~ Uniform(-1/2, 1/2)
"""

from __future__ import annotations

import torch


def sample_uniform_noise_like(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Sample nu with the same shape as tensor from U(-1/2, 1/2).
    """
    return torch.rand_like(tensor) - 0.5


def add_channelwise_uniform_noise(
    tensor: torch.Tensor,
    step_size: torch.Tensor,
    channel_axis: int,
) -> torch.Tensor:
    """
    Add S * nu to a tensor, with one step size per channel.
    """
    if tensor.ndim == 0:
        raise ValueError(
            "tensor must contain at least one dimension."
        )

    if channel_axis < 0:
        channel_axis += tensor.ndim

    if not 0 <= channel_axis < tensor.ndim:
        raise ValueError(
            f"Invalid channel_axis={channel_axis} for "
            f"tensor shape {tuple(tensor.shape)}."
        )

    step_size = torch.as_tensor(
        step_size,
        device=tensor.device,
        dtype=tensor.dtype,
    )

    expected_channels = tensor.shape[channel_axis]

    if step_size.numel() != expected_channels:
        raise ValueError(
            f"step_size contains {step_size.numel()} values, "
            f"but tensor axis {channel_axis} contains "
            f"{expected_channels} channels."
        )

    broadcast_shape = [1] * tensor.ndim
    broadcast_shape[channel_axis] = expected_channels

    return tensor + sample_uniform_noise_like(tensor) * step_size.reshape(
        broadcast_shape
    )


def add_weight_quantization_noise(
    weight: torch.Tensor,
    step_size: torch.Tensor,
    channel_axis: int = 0,
) -> torch.Tensor:
    """
    Add uniform quantization noise to a weight tensor.
    """
    return add_channelwise_uniform_noise(
        tensor=weight,
        step_size=step_size,
        channel_axis=channel_axis,
    )
