

"""
Probabilistic Programming Quantization (PPQ) implementation inspired by
"Improving Post-Training Quantization via Probabilistic Programming".

This module provides a PyTorch-based reference implementation of the core
ideas described in the paper, covering:
  * Percentile-based activation clipping
  * Bayesian MAP optimisation of quantisation step sizes with an MDL prior
  * Monte-Carlo likelihood estimation with additive uniform noise
  * Validation-driven early stopping

The implementation is model-agnostic: any `torch.nn.Module` composed of
`nn.Conv2d` and `nn.Linear` layers can be calibrated using a small
calibration/validation split.

Example usage (assuming `calib_loader` and `val_loader` are prepared and
deliver `(inputs, labels)` batches):

```python
from PPQ.PPQ_L2 import PPQConfig, run_ppq_calibration

config = PPQConfig(
    bit_width=4,
    gamma=0.2,
    num_mc_samples=16,
    lr=1e-3,
    device="cuda"
)

quant_model, calibration_report = run_ppq_calibration(
    model=fp32_model,
    calib_loader=calib_loader,
    val_loader=val_loader,
    config=config,
)
```
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


ModelInput = Union[Tensor, Dict[str, Tensor]]


def _move_to_device(value, device: torch.device):
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    return value


def _prepare_model_inputs(batch, device: torch.device) -> ModelInput:
    """
    Normalize dataloader batches so downstream code can call the model either with
    positional tensors or keyword arguments.
    """
    if isinstance(batch, dict):
        return {k: _move_to_device(v, device) for k, v in batch.items() if v is not None}
    if isinstance(batch, (list, tuple)):
        return _prepare_model_inputs(batch[0], device)
    if isinstance(batch, Tensor):
        return batch.to(device)
    raise TypeError(f"Unsupported batch type {type(batch)}")


def _forward_model(model: nn.Module, model_inputs: ModelInput) -> Tensor:
    """
    Call a model with either positional or keyword arguments and always return
    the main tensor output.
    """
    if isinstance(model_inputs, dict):
        outputs = model(**model_inputs)
    else:
        outputs = model(model_inputs)

    if isinstance(outputs, Tensor):
        return outputs

    for attr in ("output", "logits"):
        if hasattr(outputs, attr):
            candidate = getattr(outputs, attr)
            if isinstance(candidate, Tensor):
                return candidate

    raise TypeError(
        f"Model returned unsupported output type {type(outputs)} – expected Tensor "
        "or an object with `.output`/`.logits` tensor attributes."
    )


class _RoundSTE(torch.autograd.Function):
    """Straight-through estimator (STE) for the rounding operation."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        return x.round()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output


@dataclass
class PPQConfig:
    """Hyper-parameters controlling the PPQ calibration routine."""

    bit_width: int = 4
    gamma: float = 0.2
    eta: float = 1e-2
    num_mc_samples: int = 8
    lr: float = 1e-3
    max_steps: int = 1000
    validation_interval: int = 2
    early_stopping_patience: int = 100
    percentile: float = 1e-3
    s_min: float = 1e-4
    s_max: float = 1.0
    device: str = "cuda"
    target_module_types: Tuple[type, ...] = (nn.Conv2d, nn.Linear)
    max_calibration_batches: Optional[int] = None
    max_validation_batches: Optional[int] = None
    clip_percentile_floor: float = 1e-12
    clip_percentile_ceil: float = 5e-1
    report_samples: int = 8

    def __post_init__(self) -> None:
        if not (0 < self.percentile < 0.5):
            raise ValueError("percentile must lie in (0, 0.5).")
        if self.bit_width < 2:
            raise ValueError("bit_width must be >= 2.")
        if self.num_mc_samples < 1:
            raise ValueError("num_mc_samples must be >= 1.")


class _StatsAccumulator:
    """Streaming accumulator for mean and variance estimation."""

    def __init__(self) -> None:
        self.sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

    def update(self, tensor: Tensor) -> None:
        flat = tensor.detach().flatten().to(dtype=torch.float64)
        self.sum += float(flat.sum().item())
        self.sq_sum += float((flat ** 2).sum().item())
        self.count += flat.numel()

    def mean_std(self) -> Tuple[float, float]:
        if self.count == 0:
            return 0.0, 0.0
        mean = self.sum / self.count
        mean_sq = self.sq_sum / self.count
        variance = max(mean_sq - mean ** 2, 0.0)
        std = math.sqrt(variance)
        return mean, std


class ClippingSelector:
    """
    Computes percentile-based clipping thresholds for layer activations.

    Activations are assumed to follow approximately Gaussian distributions.
    The selector tracks running mean and variance and applies the closed-form
    percentile threshold described in the paper.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: Sequence[Tuple[str, nn.Module]],
        device: torch.device,
        config: PPQConfig,
    ) -> None:
        self.model = model
        self.target_layers = target_layers
        self.device = device
        self.config = config
        self._accumulators: Dict[str, _StatsAccumulator] = {
            name: _StatsAccumulator() for name, _ in target_layers
        }

    def collect(self, dataloader: Iterable, max_batches: Optional[int]) -> None:
        handles = []
        for name, module in self.target_layers:
            handles.append(
                module.register_forward_hook(
                    self._make_hook(name),
                )
            )
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                model_inputs = _prepare_model_inputs(batch, self.device)
                if isinstance(model_inputs, dict):
                    _ = self.model(**model_inputs)
                else:
                    _ = self.model(model_inputs)
                if max_batches is not None and (batch_idx + 1) >= max_batches:
                    break
        for handle in handles:
            handle.remove()

    def compute_thresholds(self) -> Dict[str, Tuple[float, float]]:
        thresholds = {}
        percentile = float(self.config.percentile)
        percentile = min(max(percentile, self.config.clip_percentile_floor), self.config.clip_percentile_ceil)
        erf_arg = 1.0 - 2.0 * percentile
        erf_arg = min(max(erf_arg, -0.999999), 0.999999)
        erf_arg_tensor = torch.tensor(erf_arg, dtype=torch.float32)
        erfinv_value = torch.special.erfinv(erf_arg_tensor).item()

        for name, accumulator in self._accumulators.items():
            mean, std = accumulator.mean_std()
            if std == 0.0:
                # Degenerate case: fall back to symmetric bounds around the mean.
                alpha = mean - 1.0
                beta = mean + 1.0
            else:
                tau = 2.0 * std * erfinv_value + mean
                beta = tau
                alpha = 2.0 * mean - tau
                if beta <= alpha:
                    width = max(abs(mean), std)
                    alpha = mean - width
                    beta = mean + width
            thresholds[name] = (float(alpha), float(beta))
        return thresholds

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: Tuple[Tensor, ...], output: Tensor) -> None:
            self._accumulators[name].update(output.detach())

        return hook


class ProbabilisticQuantizer(nn.Module):
    """
    Uniform symmetric quantizer with additive-noise surrogate (MAP)
    and real quantize-dequantize for deployment/validation.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        bit_width: int,
        gamma: float,
        s_min: float,
        s_max: float,
        device: torch.device,
    ) -> None:
        super().__init__()

        if beta <= alpha:
            raise ValueError("beta must be greater than alpha for quantiser initialisation.")

        range_span = beta - alpha
        initial_step = range_span / (2 ** bit_width - 1)
        initial_step = float(min(max(initial_step, s_min), s_max))

        # Non-trainable buffers (move with .to(device), saved in state_dict)
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32, device=device))
        self.register_buffer("beta", torch.tensor(beta, dtype=torch.float32, device=device))
        self.register_buffer("range_span", torch.tensor(range_span, dtype=torch.float32, device=device))

        # Symmetric signed integer range (e.g., 4-bit -> ±7)
        self.qmax = float(2 ** (bit_width - 1) - 1)

        # Hyper-params
        self.gamma = gamma
        self.s_min = s_min
        self.s_max = s_max

        # Toggle: True during MC/MAP; False for validation/deploy
        self.sample_noise = True

        # Learnable step size (scalar here; can extend to per-channel)
        self.step = nn.Parameter(torch.tensor(initial_step, dtype=torch.float32, device=device))

    def clamp_step_(self) -> None:
        # Projected gradient step: keep S within [s_min, s_max]
        with torch.no_grad():
            self.step.clamp_(self.s_min, self.s_max)

    def log_prior(self) -> torch.Tensor:
        # MDL prior:  -gamma * log2(R / S)
        eps = 1e-12
        return -self.gamma * torch.log2(self.range_span / (self.step.abs() + eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        - Training/MAP (self.sample_noise == True):
            x_tilde = clip(x + S * U(-0.5, 0.5), alpha, beta)
            (no rounding; differentiable surrogate)
        - Validation/Deploy (self.sample_noise == False):
            Real quantize-dequantize with rounding.
        """
        S = self.step.abs()

        if self.sample_noise:
            # ---- MC/MAP mode: additive-noise surrogate only ----
            # ε ~ Uniform(-S/2, S/2) via reparameterization
            x = x + S * (torch.rand_like(x) - 0.5)
            # Percentile clipping
            x = torch.clamp(x, self.alpha, self.beta)
            # Return the (continuous) surrogate output (no rounding in MAP)
            return x
        else:
            # ---- Validation/Deploy: real quantize-dequantize ----
            # Clip first
            x = torch.clamp(x, self.alpha, self.beta)
            # Scale to grid
            z = x / (S + 1e-12)
            # Hard rounding (no STE needed here; typically eval/no_grad)
            z_q = torch.clamp(torch.round(z), -self.qmax, self.qmax)
            # De-normalize back
            return z_q * S



class PPQLayerWrapper(nn.Module):
    """
    Light-weight wrapper attaching a probabilistic quantiser to a module.
    """

    def __init__(self, module: nn.Module, quantiser: ProbabilisticQuantizer) -> None:
        super().__init__()
        self.module = module
        self.quantiser = quantiser

    def forward(self, x: Tensor) -> Tensor:
        y = self.module(x)
        return self.quantiser(y)


class ValidationManager:
    """Tracks validation loss and implements patience-based early stopping."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_loss: Optional[float] = None
        self.stalled_rounds = 0
        self.best_steps: Optional[List[Tensor]] = None

    def update(self, current_loss: float, quantisers: Sequence[ProbabilisticQuantizer]) -> None:
        if self.best_loss is None or current_loss < self.best_loss:
            self.best_loss = current_loss
            self.stalled_rounds = 0
            self.best_steps = [q.step.detach().clone() for q in quantisers]
        else:
            self.stalled_rounds += 1

    @property
    def should_stop(self) -> bool:
        return self.stalled_rounds >= self.patience

    def restore_best(self, quantisers: Sequence[ProbabilisticQuantizer]) -> None:
        if self.best_steps is None:
            return
        for param, best in zip(quantisers, self.best_steps):
            param.step.data.copy_(best.to(param.step.device))


class MAPOptimizer:
    """
    Performs joint MAP optimisation of all quantiser step sizes.

    The optimiser maximises the Monte Carlo estimated posterior by minimising
    the negative log-posterior with Adam, while enforcing projected bounds on
    the step sizes.
    """

    def __init__(
        self,
        fp_model: nn.Module,
        quant_model: nn.Module,
        quantisers: Sequence[ProbabilisticQuantizer],
        config: PPQConfig,
        device: torch.device,
    ) -> None:
        self.fp_model = fp_model
        self.quant_model = quant_model
        self.quantisers = list(quantisers)
        self.config = config
        self.device = device
        self.quantiser_names = [
            getattr(quantiser, "layer_name", f"layer_{idx}")
            for idx, quantiser in enumerate(self.quantisers)
        ]

        params = [q.step for q in self.quantisers]
        self.optim = torch.optim.Adam(params, lr=config.lr)

    def train(
        self,
        calib_loader: Iterable,
        val_loader: Iterable,
    ) -> Dict[str, float]:
        validation_manager = ValidationManager(self.config.early_stopping_patience)
        iteration = 0
        report: Dict[str, float] = {}

        for step in range(self.config.max_steps):
            step_loss = 0.0
            processed_batches = 0

            for batch_idx, batch in enumerate(calib_loader):
                model_inputs = _prepare_model_inputs(batch, self.device)

                with torch.no_grad():
                    fp_outputs = _forward_model(self.fp_model, model_inputs)

                mc_outputs = []
                for _ in range(self.config.num_mc_samples):
                    for quantiser in self.quantisers:
                        quantiser.sample_noise = True
                    mc_outputs.append(_forward_model(self.quant_model, model_inputs))

                stacked = torch.stack(mc_outputs, dim=0)  # (M, B, ...)
                fp_expanded = fp_outputs.unsqueeze(0)

                diff = stacked - fp_expanded
                diff = diff.flatten(start_dim=2)
                log_probs = -(diff.pow(2).sum(dim=2)) / (2 * self.config.eta)
                log_likelihood = torch.logsumexp(log_probs, dim=0) - math.log(self.config.num_mc_samples)
                loss_likelihood = -log_likelihood.mean()

                log_prior = torch.stack([quantiser.log_prior() for quantiser in self.quantisers]).sum()
                loss = loss_likelihood + log_prior

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                for quantiser in self.quantisers:
                    quantiser.clamp_step_()

                step_loss += float(loss.item())
                processed_batches += 1

                iteration += 1
                if iteration % self.config.validation_interval == 0:
                    val_loss = self.evaluate(val_loader)
                    validation_manager.update(val_loss, self.quantisers)
                    report[f"val_loss/{iteration}"] = val_loss
                    if validation_manager.should_stop:
                        validation_manager.restore_best(self.quantisers)
                        report["early_stop_iteration"] = float(iteration)
                        report["train_loss_last"] = step_loss / max(processed_batches, 1)
                        self._record_step_sizes(report)
                        return report

                if (
                    self.config.max_calibration_batches is not None
                    and (batch_idx + 1) >= self.config.max_calibration_batches
                ):
                    break

            if processed_batches == 0:
                break

        # Load best parameters observed during validation.
        validation_manager.restore_best(self.quantisers)
        report["train_loss_last"] = step_loss / max(processed_batches, 1)
        self._record_step_sizes(report)
        return report

    def evaluate(self, dataloader: Iterable) -> float:
        for quantiser in self.quantisers:
            quantiser.sample_noise = False

        total_loss = 0.0
        total_items = 0

        self.quant_model.eval()
        self.fp_model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                model_inputs = _prepare_model_inputs(batch, self.device)

                fp_outputs = _forward_model(self.fp_model, model_inputs)
                quant_outputs = _forward_model(self.quant_model, model_inputs)

                mse = F.mse_loss(quant_outputs, fp_outputs, reduction="sum")
                total_loss += float(mse.item())
                total_items += fp_outputs.numel()

                if (
                    self.config.max_validation_batches is not None
                    and (batch_idx + 1) >= self.config.max_validation_batches
                ):
                    break

        for quantiser in self.quantisers:
            quantiser.sample_noise = True

        if total_items == 0:
            return 0.0
        return total_loss / total_items

    def _record_step_sizes(self, report: Dict[str, float]) -> None:
        for name, quantiser in zip(self.quantiser_names, self.quantisers):
            report[f"step/{name}"] = float(quantiser.step.item())


def _clone_model(model: nn.Module) -> nn.Module:
    cloned = copy.deepcopy(model)
    for module in cloned.modules():
        if hasattr(module, "requires_grad_"):
            module.requires_grad_(False)
    cloned.eval()
    return cloned


def _find_target_layers(
    model: nn.Module, allowed_types: Tuple[type, ...]
) -> List[Tuple[str, nn.Module]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, allowed_types)]


def _apply_wrappers(
    model: nn.Module,
    thresholds: Dict[str, Tuple[float, float]],
    config: PPQConfig,
    device: torch.device,
) -> Tuple[nn.Module, List[ProbabilisticQuantizer]]:
    quantisers: List[ProbabilisticQuantizer] = []
    for name, module in model.named_modules():
        if name not in thresholds:
            continue
        parent, child_name = _split_parent_child(model, name)
        alpha, beta = thresholds[name]
        quantiser = ProbabilisticQuantizer(
            alpha=alpha,
            beta=beta,
            bit_width=config.bit_width,
            gamma=config.gamma,
            s_min=config.s_min,
            s_max=config.s_max,
            device=device,
        )
        setattr(quantiser, "layer_name", name)
        wrapper = PPQLayerWrapper(module, quantiser)
        setattr(parent, child_name, wrapper)
        quantisers.append(quantiser)
    return model, quantisers


def _split_parent_child(model: nn.Module, qualified_name: str) -> Tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def run_ppq_calibration(
    model: nn.Module,
    calib_loader: Iterable,
    val_loader: Iterable,
    config: PPQConfig,
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Execute PPQ calibration on a model.

    Args:
        model: Pre-trained float32 model.
        calib_loader: DataLoader (or iterable) for calibration samples.
        val_loader: DataLoader (or iterable) for validation/early stopping.
        config: Calibration hyper-parameters.

    Returns:
        A tuple consisting of the calibrated quantised model and a report with
        diagnostic statistics (validation losses, early stopping iteration, ...).
    """
    device = torch.device(config.device)

    fp_model = _clone_model(model).to(device)
    target_layers = _find_target_layers(fp_model, config.target_module_types)

    clipping_selector = ClippingSelector(fp_model, target_layers, device=device, config=config)
    clipping_selector.collect(calib_loader, config.max_calibration_batches)
    thresholds = clipping_selector.compute_thresholds()

    quant_model = _clone_model(model).to(device)
    quant_model, quantisers = _apply_wrappers(quant_model, thresholds, config, device)

    optimizer = MAPOptimizer(
        fp_model=fp_model,
        quant_model=quant_model,
        quantisers=quantisers,
        config=config,
        device=device,
    )
    report = optimizer.train(calib_loader, val_loader)

    for quantiser in quantisers:
        quantiser.sample_noise = False
    quant_model.eval()

    return quant_model, report


__all__ = [
    "PPQConfig",
    "run_ppq_calibration",
    "ProbabilisticQuantizer",
    "PPQLayerWrapper",
]
