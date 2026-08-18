"""
Training utilities for Sobolev-Guided Beta Probabilistic Quantization.

This trainer combines:

    Poseidon network-wise Monte Carlo likelihood
    +
    block-wise Beta prior

The optimization variables are only the channel-wise weight step sizes.
The original model weights and the precomputed Beta parameters remain fixed.

Objective:

    total_loss
        =
        likelihood_scale * likelihood_loss
        +
        prior_loss
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn as nn

from SBPQ.beta_prior import BlockwiseBetaPrior
from SBPQ.step_sizes import LearnableStepSizes
from SBPQ.poseidon.likelihood import PoseidonNetworkLikelihood


def extract_weight_ranges(
    ranges_dict: Mapping[
        str,
        Mapping[str, torch.Tensor],
    ],
) -> OrderedDict[str, torch.Tensor]:
    """
    Extract only weight ranges from the complete Poseidon range dictionary.

    Input:

        {
            layer_name: {
                "weight_ranges": tensor,
                "activation_ranges": tensor,
            }
        }

    Output:

        {
            layer_name: weight_range_tensor
        }
    """
    if len(ranges_dict) == 0:
        raise ValueError(
            "ranges_dict cannot be empty."
        )

    weight_ranges = OrderedDict()

    for layer_name, layer_ranges in ranges_dict.items():
        if "weight_ranges" not in layer_ranges:
            raise KeyError(
                f"Layer '{layer_name}' does not contain "
                "'weight_ranges'."
            )

        weight_range = torch.as_tensor(
            layer_ranges["weight_ranges"],
            dtype=torch.float32,
        )

        if weight_range.numel() == 0:
            raise ValueError(
                f"Layer '{layer_name}' has an empty weight range."
            )

        if torch.any(weight_range <= 0):
            raise ValueError(
                f"Layer '{layer_name}' contains non-positive "
                "weight-range values."
            )

        weight_ranges[layer_name] = weight_range

    return weight_ranges


def freeze_model_parameters(
    model: nn.Module,
) -> None:
    """
    Freeze all original model parameters.

    The model is still used in forward passes, but its weights are not
    optimized. Gradients are required only for the learnable step sizes.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    model.eval()


class SBPQTrainer:
    """
    Optimize channel-wise weight step sizes using the SBPQ objective.

    The trainer combines:

        likelihood_loss
            Network-output reconstruction under sampled weight noise.

        beta_prior_loss
            Block-wise Beta negative log-prior over effective bitwidths.
    """

    def __init__(
        self,
        model: nn.Module,
        frozen_batches: Sequence,
        clean_network_outputs: Sequence[torch.Tensor],
        ranges_dict: Mapping[
            str,
            Mapping[str, torch.Tensor],
        ],
        layer_to_block: Mapping[str, str],
        beta_parameter_path: str | Path,
        initial_bits: float,
        minimum_bits: float,
        maximum_bits: float,
        learning_rate: float = 1e-3,
        num_mc_samples: int = 10,
        eta: float = 1e-4,
        prior_scale: float = 1.0,
        likelihood_scale: float = 1.0,
        beta_reduction: str = "sum",
        beta_boundary_epsilon: float = 1e-6,
        weight_decay: float = 0.0,
        gradient_clip_norm: float | None = None,
        channel_weights: Mapping[str, torch.Tensor] | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        """
        Initialize the SBPQ trainer.

        Args:
            model:
                Full-precision Poseidon model.

            frozen_batches:
                Fixed calibration batches used during optimization.

            clean_network_outputs:
                Cached full-precision outputs corresponding to the frozen
                calibration batches.

            ranges_dict:
                Precomputed Poseidon weight and activation ranges.

            layer_to_block:
                Mapping from each quantized Linear layer to its structural
                Poseidon block.

            beta_parameter_path:
                Saved block-wise Beta-parameter file.

            initial_bits:
                Initial effective bitwidth used to initialize all step sizes.

            minimum_bits:
                Minimum allowed effective bitwidth.

            maximum_bits:
                Maximum allowed effective bitwidth.

            learning_rate:
                Step-size optimizer learning rate.

            num_mc_samples:
                Number of noisy network evaluations for each likelihood.

            eta:
                Likelihood variance or temperature parameter.

            prior_scale:
                Global multiplier for the Beta prior.

            likelihood_scale:
                Global multiplier for the likelihood.

            beta_reduction:
                How channel-wise Beta-prior values are combined:
                "sum" or "mean".

            beta_boundary_epsilon:
                Keeps normalized bitwidth away from exactly 0 and 1.

            weight_decay:
                AdamW weight decay applied to the step-size parameters.

            gradient_clip_norm:
                Optional maximum gradient norm.

            channel_weights:
                Optional per-layer per-output-channel parameter weights used
                for parameter-weighted average bitwidth reporting.

            device:
                Optimization device.
        """
        if len(frozen_batches) == 0:
            raise ValueError(
                "frozen_batches cannot be empty."
            )

        if len(frozen_batches) != len(clean_network_outputs):
            raise ValueError(
                "frozen_batches and clean_network_outputs must "
                "have the same length."
            )

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if likelihood_scale < 0:
            raise ValueError(
                "likelihood_scale must be non-negative."
            )

        if prior_scale < 0:
            raise ValueError(
                "prior_scale must be non-negative."
            )

        if (
            gradient_clip_norm is not None
            and gradient_clip_norm <= 0
        ):
            raise ValueError(
                "gradient_clip_norm must be positive or None."
            )

        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )

        self.model = model.to(self.device)

        freeze_model_parameters(
            self.model
        )

        self.frozen_batches = list(
            frozen_batches
        )

        self.clean_network_outputs = [
            output.detach().cpu()
            for output in clean_network_outputs
        ]

        self.minimum_bits = float(
            minimum_bits
        )

        self.maximum_bits = float(
            maximum_bits
        )

        self.initial_bits = float(
            initial_bits
        )

        self.learning_rate = float(
            learning_rate
        )

        self.likelihood_scale = float(
            likelihood_scale
        )

        self.prior_scale = float(
            prior_scale
        )

        self.gradient_clip_norm = (
            None
            if gradient_clip_norm is None
            else float(gradient_clip_norm)
        )

        # -----------------------------------------------------
        # 1. Extract the weight ranges used by the step sizes
        # -----------------------------------------------------
        weight_ranges = extract_weight_ranges(
            ranges_dict=ranges_dict,
        )

        # Only keep layers that have a block assignment.
        #
        # Layers outside structural blocks cannot use a block-wise
        # Beta prior, so they are excluded from SBPQ optimization.
        filtered_weight_ranges = OrderedDict(
            (
                layer_name,
                weight_range,
            )
            for layer_name, weight_range in weight_ranges.items()
            if layer_name in layer_to_block
        )

        if len(filtered_weight_ranges) == 0:
            raise RuntimeError(
                "No quantized layers have both a weight range and "
                "a block assignment."
            )

        self.layer_to_block = {
            layer_name: layer_to_block[layer_name]
            for layer_name in filtered_weight_ranges
        }

        self.channel_weights = None
        if channel_weights is not None:
            self.channel_weights = {
                layer_name: torch.as_tensor(
                    channel_weights[layer_name],
                    dtype=torch.float32,
                    device=self.device,
                )
                for layer_name in filtered_weight_ranges
                if layer_name in channel_weights
            }

        # -----------------------------------------------------
        # 2. Create learnable step sizes
        # -----------------------------------------------------
        self.step_size_module = LearnableStepSizes(
            quantization_ranges=filtered_weight_ranges,
            initial_bits=self.initial_bits,
            minimum_bits=self.minimum_bits,
            maximum_bits=self.maximum_bits,
            device=self.device,
        ).to(self.device)

        # -----------------------------------------------------
        # 3. Create network-wise Monte Carlo likelihood
        # -----------------------------------------------------
        self.likelihood = PoseidonNetworkLikelihood(
            model=self.model,
            frozen_batches=self.frozen_batches,
            clean_network_outputs=self.clean_network_outputs,
            num_mc_samples=num_mc_samples,
            eta=eta,
            device=self.device,
        )

        # -----------------------------------------------------
        # 4. Create block-wise Beta prior
        # -----------------------------------------------------
        self.beta_prior = BlockwiseBetaPrior(
            beta_parameter_path=beta_parameter_path,
            layer_to_block=self.layer_to_block,
            minimum_bits=self.minimum_bits,
            maximum_bits=self.maximum_bits,
            prior_scale=self.prior_scale,
            boundary_epsilon=beta_boundary_epsilon,
            reduction=beta_reduction,
            device=self.device,
        ).to(self.device)

        # -----------------------------------------------------
        # 5. Optimize only step-size parameters
        # -----------------------------------------------------
        self.optimizer = torch.optim.AdamW(
            self.step_size_module.parameters(),
            lr=self.learning_rate,
            weight_decay=float(weight_decay),
        )

        self.history: list[dict[str, float]] = []

        print(
            f"[INFO] SBPQ trainer initialized with "
            f"{len(filtered_weight_ranges)} quantized layers."
        )

    def get_step_sizes(
        self,
    ) -> OrderedDict[str, torch.Tensor]:
        """
        Return the current learnable step sizes.
        """
        return self.step_size_module.get_step_sizes()

    def get_quantization_ranges(
        self,
    ) -> OrderedDict[str, torch.Tensor]:
        """
        Return the weight ranges associated with the step sizes.
        """
        return self.step_size_module.get_quantization_ranges()

    def get_effective_bitwidths(
        self,
    ) -> OrderedDict[str, torch.Tensor]:
        """
        Return the current effective bitwidths.
        """
        return self.step_size_module.get_effective_bitwidths()

    def compute_losses(
        self,
        batch_index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Compute total, likelihood, and Beta-prior losses.

        Returns:

            total_loss,
            likelihood_loss,
            beta_prior_loss
        """
        step_sizes = self.get_step_sizes()

        quantization_ranges = (
            self.get_quantization_ranges()
        )

        likelihood_loss = self.likelihood(
            step_sizes=step_sizes,
            batch_index=batch_index,
        )

        beta_prior_loss = self.beta_prior(
            step_sizes=step_sizes,
            quantization_ranges=quantization_ranges,
        )

        total_loss = (
            self.likelihood_scale * likelihood_loss
            + beta_prior_loss
        )

        return (
            total_loss,
            likelihood_loss,
            beta_prior_loss,
        )

    def train_step(
        self,
        batch_index: int,
    ) -> dict[str, float]:
        """
        Perform one optimization step using one frozen batch.
        """
        if not 0 <= batch_index < len(self.frozen_batches):
            raise IndexError(
                f"batch_index={batch_index} is outside the valid "
                f"range [0, {len(self.frozen_batches) - 1}]."
            )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        (
            total_loss,
            likelihood_loss,
            beta_prior_loss,
        ) = self.compute_losses(
            batch_index=batch_index,
        )

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"Non-finite total loss detected: "
                f"{total_loss.item()}."
            )

        total_loss.backward()

        if self.gradient_clip_norm is not None:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.step_size_module.parameters(),
                max_norm=self.gradient_clip_norm,
            )
        else:
            gradient_norm = self._calculate_gradient_norm()

        self.optimizer.step()

        # Keep effective bitwidths inside the configured bounds.
        self.step_size_module.clamp_()

        average_bits = self.calculate_average_bitwidth(
            parameter_weighted=True,
        )

        result = {
            "batch_index": float(batch_index),
            "total_loss": float(
                total_loss.detach().cpu().item()
            ),
            "likelihood_loss": float(
                likelihood_loss.detach().cpu().item()
            ),
            "beta_prior_loss": float(
                beta_prior_loss.detach().cpu().item()
            ),
            "average_bits": float(average_bits),
            "gradient_norm": float(
                torch.as_tensor(
                    gradient_norm
                ).detach().cpu().item()
            ),
        }

        self.history.append(
            result
        )

        return result

    def optimize(
        self,
        number_of_steps: int,
        print_every: int = 1,
    ) -> list[dict[str, float]]:
        """
        Run SBPQ step-size optimization.

        Frozen calibration batches are selected cyclically:

            step 0 -> batch 0
            step 1 -> batch 1
            ...
            final batch -> return to batch 0
        """
        if number_of_steps <= 0:
            raise ValueError(
                "number_of_steps must be positive."
            )

        if print_every <= 0:
            raise ValueError(
                "print_every must be positive."
            )

        print("\n========== SBPQ OPTIMIZATION ==========")

        for step_index in range(number_of_steps):
            batch_index = (
                step_index % len(self.frozen_batches)
            )

            result = self.train_step(
                batch_index=batch_index,
            )

            if (
                (step_index + 1) % print_every == 0
                or step_index == 0
                or step_index + 1 == number_of_steps
            ):
                print(
                    f"step={step_index + 1}/{number_of_steps} "
                    f"| batch={batch_index} "
                    f"| total={result['total_loss']:.6e} "
                    f"| likelihood="
                    f"{result['likelihood_loss']:.6e} "
                    f"| beta_prior="
                    f"{result['beta_prior_loss']:.6e} "
                    f"| avg_bits="
                    f"{result['average_bits']:.4f} "
                    f"| grad_norm="
                    f"{result['gradient_norm']:.6e}"
                )

        print("=======================================\n")

        return self.history

    def calculate_average_bitwidth(
        self,
        parameter_weighted: bool = False,
    ) -> float:
        """
        Calculate effective bitwidth over all channels.

        When parameter_weighted=True and channel_weights were provided,
        each output channel is weighted by the number of weights it controls.
        """
        bitwidths = self.get_effective_bitwidths()

        if not parameter_weighted or self.channel_weights is None:
            flattened_bitwidths = [
                bitwidth.reshape(-1)
                for bitwidth in bitwidths.values()
            ]

            if len(flattened_bitwidths) == 0:
                raise RuntimeError(
                    "No effective bitwidth values are available."
                )

            all_bitwidths = torch.cat(
                flattened_bitwidths,
                dim=0,
            )

            return float(
                all_bitwidths.mean().detach().cpu().item()
            )

        weighted_sum = torch.zeros(
            (),
            device=self.device,
        )
        total_weight = torch.zeros(
            (),
            device=self.device,
        )

        for layer_name, bitwidth in bitwidths.items():
            weight = self.channel_weights.get(
                layer_name,
                None,
            )

            if weight is None:
                weight = torch.ones_like(bitwidth)
            else:
                weight = weight.to(
                    device=bitwidth.device,
                    dtype=bitwidth.dtype,
                )

            if weight.numel() == 1:
                weight = weight.expand_as(bitwidth)

            if weight.shape != bitwidth.shape:
                raise ValueError(
                    f"channel_weights for '{layer_name}' have shape "
                    f"{tuple(weight.shape)}, but bitwidths have shape "
                    f"{tuple(bitwidth.shape)}."
                )

            weighted_sum = weighted_sum + (bitwidth * weight).sum()
            total_weight = total_weight + weight.sum()

        if total_weight <= 0:
            raise RuntimeError(
                "Total bitwidth reporting weight is zero."
            )

        return float(
            (weighted_sum / total_weight).detach().cpu().item()
        )

    def _calculate_gradient_norm(
        self,
    ) -> torch.Tensor:
        """
        Calculate the total L2 norm of all step-size gradients.
        """
        squared_norms = []

        for parameter in self.step_size_module.parameters():
            if parameter.grad is None:
                continue

            squared_norms.append(
                parameter.grad.detach().pow(2).sum()
            )

        if len(squared_norms) == 0:
            return torch.zeros(
                (),
                device=self.device,
            )

        return torch.sqrt(
            torch.stack(squared_norms).sum()
        )

    def get_state_dict(
        self,
    ) -> dict:
        """
        Build a serializable SBPQ optimization state.
        """
        return {
            "step_size_state_dict": (
                self.step_size_module.state_dict()
            ),
            "optimizer_state_dict": (
                self.optimizer.state_dict()
            ),
            "history": list(self.history),
            "layer_to_block": dict(
                self.layer_to_block
            ),
            "initial_bits": self.initial_bits,
            "minimum_bits": self.minimum_bits,
            "maximum_bits": self.maximum_bits,
            "learning_rate": self.learning_rate,
            "likelihood_scale": self.likelihood_scale,
            "prior_scale": self.prior_scale,
            "parameter_weighted_average_bits": (
                self.calculate_average_bitwidth(
                    parameter_weighted=True,
                )
            ),
            "unweighted_average_bits": (
                self.calculate_average_bitwidth(
                    parameter_weighted=False,
                )
            ),
        }

    def save(
        self,
        save_path: str | Path,
        metadata: dict | None = None,
    ) -> None:
        """
        Save optimized step sizes and trainer state.
        """
        save_path = Path(
            save_path
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        state = self.get_state_dict()

        state["metadata"] = (
            metadata or {}
        )

        state["optimized_step_sizes"] = {
            layer_name: step_size.detach().cpu()
            for layer_name, step_size
            in self.get_step_sizes().items()
        }

        state["effective_bitwidths"] = {
            layer_name: bitwidth.detach().cpu()
            for layer_name, bitwidth
            in self.get_effective_bitwidths().items()
        }

        torch.save(
            state,
            save_path,
        )

        print(
            f"[INFO] Saved SBPQ trainer state to: {save_path}"
        )

    def load(
        self,
        checkpoint_path: str | Path,
        load_optimizer: bool = True,
    ) -> None:
        """
        Restore step sizes and optionally the optimizer state.
        """
        checkpoint_path = Path(
            checkpoint_path
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"SBPQ checkpoint was not found: "
                f"{checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        if "step_size_state_dict" not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain "
                "'step_size_state_dict'."
            )

        self.step_size_module.load_state_dict(
            checkpoint["step_size_state_dict"]
        )

        if (
            load_optimizer
            and "optimizer_state_dict" in checkpoint
        ):
            self.optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        self.history = list(
            checkpoint.get(
                "history",
                [],
            )
        )

        # Ensure the loaded step sizes satisfy the configured bounds.
        self.step_size_module.clamp_()

        print(
            f"[INFO] Loaded SBPQ checkpoint from: "
            f"{checkpoint_path}"
        )
