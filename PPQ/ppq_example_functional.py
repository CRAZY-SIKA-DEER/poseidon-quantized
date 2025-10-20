"""
Example demonstrating PPQ calibration on synthetic divergence-free functional data.

The script:
  * Generates 2D divergence-free vector fields from random stream functions.
  * Feeds them through a tiny convolutional network.
  * Calibrates PPQ step sizes.
  * Reports loss/relative loss increases and divergence norms before/after quantisation.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from PPQ.PPQ_L2 import PPQConfig, run_ppq_calibration


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def _make_grid(grid_size: int) -> Tuple[Tensor, Tensor, float, float]:
    xs = torch.linspace(0.0, 2 * math.pi, steps=grid_size)
    ys = torch.linspace(0.0, 2 * math.pi, steps=grid_size)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    return grid_x, grid_y, dx, dy


def _sample_div_free_fields(
    num_samples: int,
    grid_size: int,
    terms: int,
    seed: int,
) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    grid_x, grid_y, dx, dy = _make_grid(grid_size)
    fields = []
    for _ in range(num_samples):
        psi = torch.zeros_like(grid_x)
        for _ in range(terms):
            kx = torch.randint(1, 5, (1,), generator=generator).item()
            ky = torch.randint(1, 5, (1,), generator=generator).item()
            phase_x = float(2 * math.pi * torch.rand(1, generator=generator).item())
            phase_y = float(2 * math.pi * torch.rand(1, generator=generator).item())
            amplitude = float(torch.randn(1, generator=generator).item())
            psi = psi + amplitude * torch.sin(kx * grid_x + phase_x) * torch.sin(ky * grid_y + phase_y)
        dpsi_dx, dpsi_dy = torch.gradient(psi, spacing=(dx, dy), edge_order=2)
        u = dpsi_dy
        v = -dpsi_dx
        field = torch.stack([u, v], dim=0)
        fields.append(field)
    return torch.stack(fields, dim=0)


def _build_functional_dataloaders(
    grid_size: int = 32,
    calib_samples: int = 64,
    val_samples: int = 64,
    batch_size: int = 16,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader, float, float]:
    fields_calib = _sample_div_free_fields(calib_samples, grid_size, terms=3, seed=seed)
    fields_val = _sample_div_free_fields(val_samples, grid_size, terms=3, seed=seed + 1)
    calib_dataset = TensorDataset(fields_calib, fields_calib.clone())
    val_dataset = TensorDataset(fields_val, fields_val.clone())
    calib_loader = DataLoader(calib_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    _, _, dx, dy = _make_grid(grid_size)
    return calib_loader, val_loader, dx, dy


class FunctionalCNN(nn.Module):
    """Small conv net processing 2-channel vector fields."""

    def __init__(self, hidden_channels: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _mse_loss(model: nn.Module, loader: Iterable, device: torch.device) -> float:
    criterion = nn.MSELoss(reduction="none")
    total_loss = 0.0
    total_elems = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            batch_loss = criterion(outputs, targets)
            total_loss += float(batch_loss.sum().item())
            total_elems += batch_loss.numel()
    if total_elems == 0:
        return 0.0
    return total_loss / total_elems


def _divergence_norm(model: nn.Module, loader: Iterable, dx: float, dy: float, device: torch.device) -> float:
    total_norm = 0.0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            u, v = outputs[:, 0], outputs[:, 1]
            du_dx, du_dy = torch.gradient(u, spacing=(dx, dy), dim=(1, 2), edge_order=2)
            dv_dx, dv_dy = torch.gradient(v, spacing=(dx, dy), dim=(1, 2), edge_order=2)
            divergence = du_dx + dv_dy
            sample_norm = torch.linalg.vector_norm(divergence.reshape(divergence.size(0), -1), dim=1)
            total_norm += float(sample_norm.sum().item())
            total_samples += divergence.size(0)
    if total_samples == 0:
        return 0.0
    return total_norm / total_samples


def main() -> None:
    _set_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    calib_loader, val_loader, dx, dy = _build_functional_dataloaders()

    model = FunctionalCNN().to(device).eval()

    config = PPQConfig(
        bit_width=4,
        gamma=0.2,
        num_mc_samples=8,
        lr=1e-3,
        percentile=1e-3,
        device=str(device),
        target_module_types=(nn.Conv2d,),
        max_calibration_batches=8,
        max_validation_batches=8,
    )

    quant_model, report = run_ppq_calibration(
        model=model,
        calib_loader=calib_loader,
        val_loader=val_loader,
        config=config,
    )

    def _sort_key(entry: str) -> Tuple[str, float, str]:
        if "/" in entry:
            prefix, suffix = entry.split("/", 1)
            try:
                return prefix, float(suffix), ""
            except ValueError:
                return prefix, float("inf"), suffix
        return entry, float("inf"), ""

    print("PPQ calibration finished. Validation log:")
    for key in sorted(report, key=_sort_key):
        print(f"  {key}: {report[key]:.6f}")

    fp_loss = _mse_loss(model, val_loader, device)
    quant_loss = _mse_loss(quant_model, val_loader, device)
    loss_increase = quant_loss - fp_loss
    rel_loss_increase = loss_increase / (fp_loss if fp_loss != 0 else 1.0)

    fp_div = _divergence_norm(model, val_loader, dx, dy, device)
    quant_div = _divergence_norm(quant_model, val_loader, dx, dy, device)
    div_increase = quant_div - fp_div
    rel_div_increase = div_increase / (fp_div if fp_div != 0 else 1.0)

    print(f"\nValidation MSE (FP32): {fp_loss:.6f}")
    print(f"Validation MSE (Quantised): {quant_loss:.6f}")
    print(f"Loss increase due to quantisation: {loss_increase:.6f}")
    print(f"Relative loss increase: {rel_loss_increase:.6%}")

    print(f"\nDivergence norm (FP32): {fp_div:.6e}")
    print(f"Divergence norm (Quantised): {quant_div:.6e}")
    print(f"Divergence increase: {div_increase:.6e}")
    print(f"Relative divergence increase: {rel_div_increase:.6%}")


if __name__ == "__main__":
    main()
