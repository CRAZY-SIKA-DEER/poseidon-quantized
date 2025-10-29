"""
Calibrate the Poseidon NS-BB model with Probabilistic Programming Quantisation (PPQ).

Usage example:
python -m PPQ.PPQ_BB \
    --model-dir models/NS-BB \
    --dataset-dir dataset/NS-BB \
    --calib-trajectories 8 \
    --calib-samples 256 \
    --val-trajectories 4 \
    --val-samples 128 \
    --bit-width 4 \
    --device cuda
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from scOT.model import ScOT
from scOT.problems.base import get_dataset

from PPQ.PPQ_L2 import PPQConfig, run_ppq_calibration

DATASET_NAME = "fluids.incompressible.BrownianBridge"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPQ calibration on the NS-BB Poseidon model.")
    parser.add_argument("--model-dir", type=Path, default=Path("models/NS-BB"), help="Directory with config.json & pytorch_model.bin.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset/NS-BB"), help="Directory containing NS-BB.nc files.")
    parser.add_argument("--dataset-name", type=str, default=DATASET_NAME, help="Dataset identifier understood by scOT.")
    parser.add_argument("--calib-trajectories", type=int, default=8, help="Number of trajectories to draw for calibration.")
    parser.add_argument("--calib-samples", type=int, default=256, help="Max calibration samples (after temporal expansion).")
    parser.add_argument("--val-trajectories", type=int, default=4, help="Number of trajectories for validation.")
    parser.add_argument("--val-samples", type=int, default=128, help="Max validation samples.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for calibration/validation loaders.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")

    parser.add_argument("--bit-width", type=int, default=4, help="PPQ quantisation bit-width.")
    parser.add_argument("--gamma", type=float, default=0.2, help="MDL prior strength.")
    parser.add_argument("--eta", type=float, default=1e-2, help="Gaussian variance for likelihood.")
    parser.add_argument("--num-mc-samples", type=int, default=8, help="Monte-Carlo samples per iteration.")
    parser.add_argument("--percentile", type=float, default=1e-3, help="Percentile for activation clipping.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for step-size optimiser.")
    parser.add_argument("--max-steps", type=int, default=400, help="Maximum optimisation iterations.")
    parser.add_argument("--early-stopping", type=int, default=80, help="Validation patience.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device string (e.g. cuda, cuda:0, cpu).")

    parser.add_argument("--max-calibration-batches", type=int, default=None, help="Optional cap on calibration batches per epoch.")
    parser.add_argument("--max-validation-batches", type=int, default=None, help="Optional cap on validation batches.")
    parser.add_argument("--save-quantised", type=Path, default=None, help="Optional path to save the quantised state_dict.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_subset(dataset, max_samples: int | None):
    if max_samples is None:
        return dataset
    max_samples = min(max_samples, len(dataset))
    indices = list(range(max_samples))
    return Subset(dataset, indices)


def build_dataloader(
    dataset_name: str,
    dataset_dir: Path,
    which: str,
    num_trajectories: int,
    max_samples: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
) -> Tuple[DataLoader, int]:
    dataset = get_dataset(
        dataset_name,
        which=which,
        num_trajectories=max(1, num_trajectories),
        data_path=str(dataset_dir),
    )
    dataset = build_subset(dataset, max_samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    effective_batches = math.ceil(len(dataset) / batch_size) if len(dataset) > 0 else 0
    return loader, effective_batches


def forward_model(model: ScOT, batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    pixel_values = batch["pixel_values"].to(device)
    time = batch.get("time")
    time = time.to(device) if isinstance(time, torch.Tensor) else None
    pixel_mask = batch.get("pixel_mask")
    pixel_mask = pixel_mask.to(device) if isinstance(pixel_mask, torch.Tensor) else None
    labels = batch.get("labels")
    labels = labels.to(device) if isinstance(labels, torch.Tensor) else None

    outputs = model(pixel_values=pixel_values, time=time, pixel_mask=pixel_mask, labels=labels)
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "output"):
        return outputs.output
    raise TypeError(f"Unexpected output type {type(outputs)} from model forward.")


def mse_against_labels(model: ScOT, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_elems = 0
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(device)
            preds = forward_model(model, batch, device)
            total_loss += F.mse_loss(preds, labels, reduction="sum").item()
            total_elems += labels.numel()
    return total_loss / max(total_elems, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model directory {args.model_dir} does not exist.")
    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory {args.dataset_dir} does not exist.")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    pin_memory = device.type == "cuda"

    calib_loader, calib_batches = build_dataloader(
        dataset_name=args.dataset_name,
        dataset_dir=args.dataset_dir,
        which="train",
        num_trajectories=args.calib_trajectories,
        max_samples=args.calib_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=pin_memory,
    )

    val_loader, val_batches = build_dataloader(
        dataset_name=args.dataset_name,
        dataset_dir=args.dataset_dir,
        which="val",
        num_trajectories=args.val_trajectories,
        max_samples=args.val_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=pin_memory,
    )

    print(f"Calibration samples: {len(calib_loader.dataset)} ({calib_batches} batches)")
    print(f"Validation samples: {len(val_loader.dataset)} ({val_batches} batches)")

    base_model = ScOT.from_pretrained(str(args.model_dir))
    base_model.to(device).eval()

    max_calib_batches = args.max_calibration_batches or calib_batches or None
    max_val_batches = args.max_validation_batches or val_batches or None

    ppq_config = PPQConfig(
        bit_width=args.bit_width,
        gamma=args.gamma,
        eta=args.eta,
        num_mc_samples=args.num_mc_samples,
        lr=args.lr,
        max_steps=args.max_steps,
        early_stopping_patience=args.early_stopping,
        percentile=args.percentile,
        device=str(device),
        max_calibration_batches=max_calib_batches,
        max_validation_batches=max_val_batches,
    )

    quant_model, report = run_ppq_calibration(
        model=base_model,
        calib_loader=calib_loader,
        val_loader=val_loader,
        config=ppq_config,
    )

    print("\nPPQ calibration summary:")
    for key in sorted(report):
        print(f"  {key}: {report[key]:.6f}")

    fp_mse = mse_against_labels(base_model, val_loader, device)
    quant_mse = mse_against_labels(quant_model, val_loader, device)
    loss_increase = quant_mse - fp_mse
    rel_increase = loss_increase / fp_mse if fp_mse > 0 else 0.0

    print("\nValidation against ground-truth labels:")
    print(f"  FP32 MSE: {fp_mse:.6f}")
    print(f"  Quantised MSE: {quant_mse:.6f}")
    print(f"  Absolute increase: {loss_increase:.6f}")
    print(f"  Relative increase: {rel_increase:.6%}")

    if args.save_quantised is not None:
        args.save_quantised.parent.mkdir(parents=True, exist_ok=True)
        torch.save(quant_model.state_dict(), args.save_quantised)
        print(f"\nQuantised state_dict saved to {args.save_quantised}")


if __name__ == "__main__":
    main()
