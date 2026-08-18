"""Aurora-specific SBPQ trainer."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn as nn

from SBPQ.beta_prior import BlockwiseBetaPrior
from SBPQ.step_sizes import LearnableStepSizes
from SBPQ.trainer import extract_weight_ranges, freeze_model_parameters
from SBPQ.aurora.likelihood import AuroraNetworkLikelihood


class AuroraSBPQTrainer:
    def __init__(
        self,
        model: nn.Module,
        frozen_batches: Sequence,
        clean_network_outputs: Sequence,
        ranges_dict: Mapping[str, Mapping[str, torch.Tensor]],
        layer_to_block: Mapping[str, str],
        beta_parameter_path: str | Path,
        initial_bits: float,
        minimum_bits: float,
        maximum_bits: float,
        learning_rate: float = 3e-5,
        num_mc_samples: int = 1,
        eta: float = 1e-3,
        likelihood_scale: float = 1.0,
        prior_scale: float = 1.0,
        weight_decay: float = 0.0,
        gradient_clip_norm: float | None = 1.0,
        autocast_dtype: torch.dtype | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        freeze_model_parameters(self.model)
        self.frozen_batches = list(frozen_batches)
        self.clean_network_outputs = list(clean_network_outputs)
        weight_ranges = extract_weight_ranges(ranges_dict)
        filtered = OrderedDict(
            (name, value)
            for name, value in weight_ranges.items()
            if name in layer_to_block
        )
        if not filtered:
            raise RuntimeError("No Aurora layers have ranges and block assignments.")
        self.layer_to_block = {name: layer_to_block[name] for name in filtered}
        self.step_size_module = LearnableStepSizes(
            filtered,
            initial_bits=initial_bits,
            minimum_bits=minimum_bits,
            maximum_bits=maximum_bits,
            device=self.device,
        ).to(self.device)
        self.likelihood = AuroraNetworkLikelihood(
            model=self.model,
            frozen_batches=self.frozen_batches,
            clean_network_outputs=self.clean_network_outputs,
            num_mc_samples=num_mc_samples,
            eta=eta,
            device=self.device,
            autocast_dtype=autocast_dtype,
        )
        self.beta_prior = BlockwiseBetaPrior(
            beta_parameter_path=beta_parameter_path,
            layer_to_block=self.layer_to_block,
            minimum_bits=minimum_bits,
            maximum_bits=maximum_bits,
            prior_scale=prior_scale,
            reduction="sum",
            device=self.device,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.step_size_module.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.gradient_clip_norm = gradient_clip_norm
        self.likelihood_scale = float(likelihood_scale)
        self.history: list[dict[str, float]] = []

    def _sanitize_nonfinite_gradients(self) -> int:
        count = 0
        for parameter in self.step_size_module.parameters():
            if parameter.grad is None:
                continue
            finite_mask = torch.isfinite(parameter.grad)
            bad_count = int((~finite_mask).sum().detach().cpu())
            if bad_count:
                parameter.grad = torch.where(
                    finite_mask,
                    parameter.grad,
                    torch.zeros_like(parameter.grad),
                )
                count += bad_count
        return count

    def get_step_sizes(self):
        return self.step_size_module.get_step_sizes()

    def train_step(self, batch_index: int = 0) -> dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        step_sizes = self.step_size_module.get_step_sizes()
        ranges = self.step_size_module.get_quantization_ranges()
        likelihood_loss = self.likelihood(step_sizes, batch_index)
        beta_loss = self.beta_prior(step_sizes, ranges)
        total_loss = self.likelihood_scale * likelihood_loss + beta_loss
        total_loss.backward()
        sanitized_gradients = self._sanitize_nonfinite_gradients()
        if self.gradient_clip_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.step_size_module.parameters(),
                self.gradient_clip_norm,
            )
        else:
            grad_norm = torch.tensor(0.0)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError(
                f"Non-finite Aurora step-size gradient norm: {grad_norm}."
            )
        self.optimizer.step()
        self.step_size_module.clamp_()
        bits = torch.cat([
            value.detach().flatten().cpu()
            for value in self.step_size_module.get_effective_bitwidths().values()
        ])
        record = {
            "total_loss": float(total_loss.detach().cpu()),
            "likelihood_loss": float(likelihood_loss.detach().cpu()),
            "beta_prior_loss": float(beta_loss.detach().cpu()),
            "gradient_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
            "sanitized_gradients": float(sanitized_gradients),
            "average_bits": float(bits.mean()),
        }
        self.history.append(record)
        return record
