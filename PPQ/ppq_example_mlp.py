"""
Example script showing how to calibrate a simple MLP with PPQ.

This script:
  * Defines a tiny feed-forward network (`TinyMLP`)
  * Builds synthetic calibration and validation data loaders
  * Runs the PPQ calibration pipeline on the model
  * Prints the resulting quantised step sizes and validation losses
"""

from __future__ import annotations

import itertools
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset

from PPQ.PPQ_L2 import PPQConfig, ProbabilisticQuantizer, run_ppq_calibration


class TinyMLP(nn.Module):
    """Minimal MLP used to demonstrate PPQ end-to-end."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 32, num_classes: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _build_dataloaders(
    num_features: int,
    calib_samples: int = 128,
    val_samples: int = 64,
    batch_size: int = 32,
    seed: int = 7,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create deterministic synthetic calibration/validation loaders.

    Labels are random but unused; they mimic the shape of image-classification
    datasets where only inputs are required for calibration.
    """
    generator = torch.Generator().manual_seed(seed)
    calib_inputs = torch.randn(calib_samples, num_features, generator=generator)
    calib_labels = torch.randint(0, 4, (calib_samples,), generator=generator)

    val_inputs = torch.randn(val_samples, num_features, generator=generator)
    val_labels = torch.randint(0, 4, (val_samples,), generator=generator)

    calib_loader = DataLoader(
        TensorDataset(calib_inputs, calib_labels),
        batch_size=batch_size,
        shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(val_inputs, val_labels),
        batch_size=batch_size,
        shuffle=False,
    )
    return calib_loader, val_loader


def _set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def _evaluate_cross_entropy(model: nn.Module, dataloader: DataLoader, device: str) -> float:
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_size = inputs.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
    if total_samples == 0:
        return 0.0
    return total_loss / total_samples


def _report_sort_key(item: str) -> Tuple[str, float]:
    if "/" in item:
        prefix, suffix = item.split("/")
        try:
            return prefix, float(suffix)
        except ValueError:
            return prefix, float("inf")
    return item, -1.0


def _describe_quantisers(model: nn.Module) -> None:
    for name, module in model.named_modules():
        if isinstance(module, ProbabilisticQuantizer):
            print(f"{name}: step={module.step.item():.6f}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = 16
    _set_global_seed(42)
    model = TinyMLP(input_dim=input_dim).to(device).eval()

    calib_loader, val_loader = _build_dataloaders(num_features=input_dim)

    config = PPQConfig(
        bit_width=4,
        gamma=0.2,
        num_mc_samples=8,
        lr=1e-3,
        percentile=1e-3,
        device=device,
        max_calibration_batches=4,
        max_validation_batches=2,
    )

    quant_model, report = run_ppq_calibration(
        model=model,
        calib_loader=calib_loader,
        val_loader=val_loader,
        config=config,
    )

    print("PPQ calibration finished. Validation log:")
    for key in sorted(report, key=_report_sort_key):
        print(f"  {key}: {report[key]:.6f}")

    fp_loss = _evaluate_cross_entropy(model, val_loader, device)
    quant_loss = _evaluate_cross_entropy(quant_model, val_loader, device)
    loss_increase = quant_loss - fp_loss
    relative_increase = loss_increase / (fp_loss if fp_loss != 0 else 1.0)
    print(f"\nValidation cross-entropy (FP32): {fp_loss:.6f}")
    print(f"Validation cross-entropy (Quantised): {quant_loss:.6f}")
    print(f"Loss increase due to quantisation: {loss_increase:.6f}")
    print(f"Relative loss increase: {relative_increase:.6%}")

    print("\nQuantiser step sizes:")
    for name, module in quant_model.named_modules():
        if isinstance(module, ProbabilisticQuantizer):
            print(f"  {name}: step={module.step.item():.6f}")

    with torch.no_grad():
        sample_inputs, _ = next(iter(itertools.cycle(val_loader)))
        sample_inputs = sample_inputs.to(device)
        outputs = quant_model(sample_inputs)
        print(f"\nQuantised model sample output stats: mean={outputs.mean():.4f}, std={outputs.std():.4f}")


if __name__ == "__main__":
    main()
