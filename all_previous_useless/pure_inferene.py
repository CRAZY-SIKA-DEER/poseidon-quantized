"""
FP32 baseline evaluation for Poseidon/ScOT:
- load model: models/NS-PwC-T
- load dataset: dataset/NS-PwC (PiecewiseConstants)
- inference + metrics: relative_lp_error / lp_error
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from scOT.model import ScOT
from scOT.problems.base import get_dataset
from scOT.metrics import relative_lp_error, lp_error


# =========================
# Config (T model)
# =========================
MODEL_PATH   = "models/NS-PwC-T"
DATA_PATH    = "dataset/NS-PwC"
DATASET_NAME = "fluids.incompressible.PiecewiseConstants"

DEVICE = "cuda"
VAL_BATCHSIZE = 16
VAL_STEPS     = 50            # evaluate first N batches


def load_poseidon_model(model_path: str, device: str = "cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = ScOT.from_pretrained(model_path).to(device).eval()
    torch.set_float32_matmul_precision("high")
    print(f"[Model] Loaded from: {model_path}")
    print(f"[Model] Device: {device}")
    print(f"[Model] Params: {sum(p.numel() for p in model.parameters()):,}")
    return model, device


def build_val_iter(dataset_name: str, data_path: str, batch_size: int, steps: int):
    # try val split; fallback to test
    try:
        val_ds = get_dataset(dataset_name, which="val", num_trajectories=256, data_path=data_path)
        split = "val"
    except Exception:
        val_ds = get_dataset(dataset_name, which="test", num_trajectories=256, data_path=data_path)
        split = "test"

    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(os.cpu_count() or 0, 16),
        pin_memory=True,
    )

    def take_n_batches():
        for i, b in enumerate(loader):
            if i >= steps:
                break
            yield b

    print(f"[Data] Using split='{split}', dataset_len={len(val_ds)}")
    print(f"[Data] Evaluating {steps} batches, batch_size={batch_size}")
    return take_n_batches


@torch.no_grad()
def evaluate_fp32(model, val_iter_fn, device, num_batches: int):
    all_preds, all_labels = [], []

    for i, batch in enumerate(tqdm(val_iter_fn(), total=num_batches, desc="Evaluating FP32")):
        if i >= num_batches:
            break

        x  = batch["pixel_values"].to(device)
        t  = batch.get("time", None)
        pm = batch.get("pixel_mask", None)
        y  = batch.get("labels", None)

        if t is not None:  t  = t.to(device)
        if pm is not None: pm = pm.to(device)
        if y is None:
            raise RuntimeError("Batch has no 'labels' field; cannot evaluate.")
        y = y.to(device)

        out = model(
            pixel_values=x,
            time=t,
            pixel_mask=pm,
            labels=y,
        )
        pred = out.output

        all_preds.append(pred.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

    preds  = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    rel_l1 = relative_lp_error(preds, labels, p=1, return_percent=True)  # (%)
    l1     = lp_error(preds, labels, p=1)

    metrics = {
        "num_samples": int(preds.shape[0]),
        "l1_mean": float(np.mean(l1)),
        "l1_median": float(np.median(l1)),
        "rel_l1_mean_percent": float(np.mean(rel_l1)),
        "rel_l1_median_percent": float(np.median(rel_l1)),
    }

    print("\n================ FP32 BASELINE (NS-PwC-T) ================")
    print(f"Samples: {metrics['num_samples']}")
    print(f"L1 mean      : {metrics['l1_mean']:.6e}")
    print(f"L1 median    : {metrics['l1_median']:.6e}")
    print(f"RelL1 mean   : {metrics['rel_l1_mean_percent']:.4f}%")
    print(f"RelL1 median : {metrics['rel_l1_median_percent']:.4f}%")
    print("=========================================================\n")
    return metrics


if __name__ == "__main__":
    model, device = load_poseidon_model(MODEL_PATH, device=DEVICE)
    val_iter_fn = build_val_iter(DATASET_NAME, DATA_PATH, VAL_BATCHSIZE, VAL_STEPS)
    evaluate_fp32(model, val_iter_fn, device, num_batches=VAL_STEPS)
