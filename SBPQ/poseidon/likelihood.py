"""
Poseidon-specific Monte Carlo likelihood for SBPQ.

The likelihood estimates the network-output error caused by simulated
weight quantization noise.

For each Monte Carlo sample:

    1. Take one fixed calibration batch.
    2. Add uniform noise to every selected Linear-layer weight.
    3. Run the complete Poseidon model.
    4. Compare the noisy output with the cached clean FP32 output.
    5. Estimate the network-wise negative log likelihood with log-sum-exp
       over Monte Carlo samples.

The simulated noise is:

    epsilon ~ Uniform(-S / 2, S / 2)

where S is the learnable per-output-channel weight step size.

This file computes only the likelihood. The Beta prior is calculated
separately by SBPQ/beta_prior.py.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from SBPQ.noise import add_weight_quantization_noise
from SBPQ.poseidon.poseidon_utils import (
    move_poseidon_batch_to_device,
)


def find_valid_noisy_linear_layers(
    model: nn.Module,
    step_sizes: Mapping[str, torch.Tensor],
) -> dict[str, nn.Linear]:
    """
    Find Linear layers that have corresponding learnable step sizes.

    Returns:

        {
            layer_name: linear_module
        }
    """
    name_to_module = dict(
        model.named_modules()
    )

    valid_layers: dict[str, nn.Linear] = {}

    for layer_name, step_size in step_sizes.items():
        module = name_to_module.get(layer_name)

        if module is None:
            raise KeyError(
                f"Layer '{layer_name}' from the step-size mapping "
                "does not exist in the model."
            )

        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"Layer '{layer_name}' has type "
                f"{type(module).__name__}, not nn.Linear."
            )

        if step_size.numel() != module.weight.shape[0]:
            raise ValueError(
                f"Layer '{layer_name}' has "
                f"{module.weight.shape[0]} output channels, but its "
                f"step-size tensor contains {step_size.numel()} values."
            )

        valid_layers[layer_name] = module

    if len(valid_layers) == 0:
        raise ValueError(
            "No valid noisy Linear layers were found."
        )

    return valid_layers


def make_noisy_linear_hook(
    step_size: torch.Tensor,
):
    """
    Create a forward hook that replaces a Linear layer's normal output.

    The normal output is:

        output = linear(input, clean_weight, bias)

    The hook returns:

        output = linear(input, noisy_weight, bias)
    """

    def noisy_linear_hook(
        module: nn.Linear,
        inputs,
        original_output,
    ) -> torch.Tensor:
        if len(inputs) == 0:
            raise RuntimeError(
                "The Linear layer received no input tensor."
            )

        layer_input = inputs[0]

        noisy_weight = add_weight_quantization_noise(
            weight=module.weight,
            step_size=step_size,
            channel_axis=0,
        )

        return F.linear(
            layer_input,
            noisy_weight,
            module.bias,
        )

    return noisy_linear_hook


def run_poseidon_with_weight_noise(
    model: nn.Module,
    batch,
    step_sizes: Mapping[str, torch.Tensor],
    device: str | torch.device,
) -> torch.Tensor:
    """
    Run one complete Poseidon forward pass with sampled weight noise.

    A new noise sample is generated each time this function is called.
    """
    device = torch.device(device)

    valid_layers = find_valid_noisy_linear_layers(
        model=model,
        step_sizes=step_sizes,
    )

    handles = []

    try:
        for layer_name, module in valid_layers.items():
            step_size = step_sizes[layer_name]

            handle = module.register_forward_hook(
                make_noisy_linear_hook(
                    step_size=step_size,
                )
            )

            handles.append(handle)

        (
            pixel_values,
            time,
            pixel_mask,
            labels,
        ) = move_poseidon_batch_to_device(
            batch=batch,
            device=device,
        )

        outputs = model(
            pixel_values=pixel_values,
            time=time,
            pixel_mask=pixel_mask,
            labels=labels,
        )

        prediction = outputs.output

    finally:
        # Hooks must always be removed. Otherwise, later forward passes
        # would continue injecting noise.
        for handle in handles:
            handle.remove()

    return prediction


def compute_network_mc_likelihood_single_batch(
    model: nn.Module,
    step_sizes: Mapping[str, torch.Tensor],
    frozen_batches: Sequence,
    clean_network_outputs: Sequence[torch.Tensor],
    batch_index: int,
    num_mc_samples: int = 10,
    eta: float = 1e-4,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """
    Compute the network-wise Monte Carlo negative log likelihood for one
    fixed batch.

    Likelihood approximation:

        -sum_i log(
            1/M sum_j exp(
                -||y_noisy_ij - y_clean_i||^2 / (2 eta)
            )
        )

    Args:
        model:
            Poseidon model.

        step_sizes:
            Mapping from Linear-layer name to learnable weight step size:

                {
                    layer_name: tensor [out_features]
                }

        frozen_batches:
            Fixed calibration batches.

        clean_network_outputs:
            Cached FP32 network output corresponding to every frozen batch.

        batch_index:
            Index of the selected frozen batch.

        num_mc_samples:
            Number of independent quantization-noise samples.

        eta:
            Likelihood variance or temperature parameter.

            Smaller eta gives stronger importance to reconstruction error.

        device:
            Device used for the forward passes.

    Returns:
        Scalar Monte Carlo negative log-likelihood.
    """
    if model is None:
        raise ValueError(
            "model must not be None."
        )

    if len(step_sizes) == 0:
        raise ValueError(
            "step_sizes cannot be empty."
        )

    if num_mc_samples <= 0:
        raise ValueError(
            "num_mc_samples must be positive."
        )

    if eta <= 0:
        raise ValueError(
            "eta must be positive."
        )

    if not 0 <= batch_index < len(frozen_batches):
        raise IndexError(
            f"batch_index={batch_index} is outside frozen_batches "
            f"with length {len(frozen_batches)}."
        )

    if not 0 <= batch_index < len(clean_network_outputs):
        raise IndexError(
            f"batch_index={batch_index} is outside "
            f"clean_network_outputs with length "
            f"{len(clean_network_outputs)}."
        )

    device = torch.device(device)

    model = model.to(device)
    model.eval()

    batch = frozen_batches[batch_index]

    clean_output = clean_network_outputs[
        batch_index
    ].to(device)

    monte_carlo_scores = []

    for _ in range(num_mc_samples):
        noisy_output = run_poseidon_with_weight_noise(
            model=model,
            batch=batch,
            step_sizes=step_sizes,
            device=device,
        )

        if noisy_output.shape != clean_output.shape:
            raise ValueError(
                "Noisy and clean network outputs have different shapes: "
                f"{tuple(noisy_output.shape)} and "
                f"{tuple(clean_output.shape)}."
            )

        reduce_dims = tuple(
            range(1, noisy_output.ndim)
        )

        squared_error = (
            noisy_output - clean_output
        ).pow(2).sum(
            dim=reduce_dims
        )

        sample_score = (
            -squared_error
            / (2.0 * float(eta))
        )

        monte_carlo_scores.append(
            sample_score
        )

    scores = torch.stack(
        monte_carlo_scores,
        dim=0,
    )

    log_probability_per_sample = (
        torch.logsumexp(
            scores,
            dim=0,
        )
        - torch.log(
            torch.tensor(
                float(num_mc_samples),
                device=device,
                dtype=scores.dtype,
            )
        )
    )

    return -log_probability_per_sample.sum()


def compute_network_mc_likelihood(
    model: nn.Module,
    step_sizes: Mapping[str, torch.Tensor],
    frozen_batches: Sequence,
    clean_network_outputs: Sequence[torch.Tensor],
    batch_indices: Sequence[int] | None = None,
    num_mc_samples: int = 10,
    eta: float = 1e-4,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """
    Compute the average Monte Carlo likelihood over multiple fixed batches.

    When batch_indices is None, every frozen batch is used.
    """
    if len(frozen_batches) != len(clean_network_outputs):
        raise ValueError(
            "frozen_batches and clean_network_outputs must contain "
            "the same number of entries."
        )

    if len(frozen_batches) == 0:
        raise ValueError(
            "frozen_batches cannot be empty."
        )

    if batch_indices is None:
        batch_indices = list(
            range(len(frozen_batches))
        )

    if len(batch_indices) == 0:
        raise ValueError(
            "batch_indices cannot be empty."
        )

    batch_losses = []

    for batch_index in batch_indices:
        batch_loss = compute_network_mc_likelihood_single_batch(
            model=model,
            step_sizes=step_sizes,
            frozen_batches=frozen_batches,
            clean_network_outputs=clean_network_outputs,
            batch_index=batch_index,
            num_mc_samples=num_mc_samples,
            eta=eta,
            device=device,
        )

        batch_losses.append(
            batch_loss
        )

    return torch.stack(
        batch_losses
    ).mean()


class PoseidonNetworkLikelihood(nn.Module):
    """
    Reusable module for the Poseidon network-wise MC likelihood.

    It stores the fixed calibration batches and their clean FP32 outputs.
    The learnable step sizes are supplied during each forward call.
    """

    def __init__(
        self,
        model: nn.Module,
        frozen_batches: Sequence,
        clean_network_outputs: Sequence[torch.Tensor],
        num_mc_samples: int = 10,
        eta: float = 1e-4,
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()

        if len(frozen_batches) == 0:
            raise ValueError(
                "frozen_batches cannot be empty."
            )

        if len(frozen_batches) != len(clean_network_outputs):
            raise ValueError(
                "frozen_batches and clean_network_outputs must have "
                "the same length."
            )

        if num_mc_samples <= 0:
            raise ValueError(
                "num_mc_samples must be positive."
            )

        if eta <= 0:
            raise ValueError(
                "eta must be positive."
            )

        self.model = model
        self.frozen_batches = list(
            frozen_batches
        )

        # Clean outputs are fixed optimization targets and do not require
        # gradients.
        self.clean_network_outputs = [
            output.detach().cpu()
            for output in clean_network_outputs
        ]

        self.num_mc_samples = int(
            num_mc_samples
        )

        self.eta = float(
            eta
        )

        self.device = torch.device(
            device
        )

    def forward(
        self,
        step_sizes: Mapping[str, torch.Tensor],
        batch_index: int,
    ) -> torch.Tensor:
        """
        Compute the MC likelihood for one selected frozen batch.
        """
        return compute_network_mc_likelihood_single_batch(
            model=self.model,
            step_sizes=step_sizes,
            frozen_batches=self.frozen_batches,
            clean_network_outputs=self.clean_network_outputs,
            batch_index=batch_index,
            num_mc_samples=self.num_mc_samples,
            eta=self.eta,
            device=self.device,
        )

    def all_batches(
        self,
        step_sizes: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the average MC likelihood over every frozen batch.
        """
        return compute_network_mc_likelihood(
            model=self.model,
            step_sizes=step_sizes,
            frozen_batches=self.frozen_batches,
            clean_network_outputs=self.clean_network_outputs,
            num_mc_samples=self.num_mc_samples,
            eta=self.eta,
            device=self.device,
        )
