#!/usr/bin/env python
import os, torch, numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ====== EDIT THESE ======
#for PwC_L
# MODEL_PATH   = "checkpoints/finetune_NS-PwC_L_2048/PoseidonFinetune_NS-PwC_L/finetune_NS-PwC_run_L_2048/checkpoint-368800"
# DATA_PATH    = "datasets/NS-PwC"
# DATASET_NAME = "fluids.incompressible.PiecewiseConstants"
#for BB_L
# MODEL_PATH   =  "checkpoints/finetune_NS-BB_L_2048/PoseidonFinetune_NS-BB_L/finetune_NS-BB_run_L_2048/checkpoint-368800"
# DATA_PATH    = "datasets/NS-BB"
# DATASET_NAME = "fluids.incompressible.BrownianBridge"
#for SVS_L
# MODEL_PATH = "checkpoints/finetune_NS-SVS_L_2048/PoseidonFinetune_NS-SVS_L/finetune_NS-SVS_run_L_2048/checkpoint-368800"
# DATA_PATH        = "datasets/NS-SVS"
# DATASET_NAME     = "fluids.incompressible.VortexSheet"
#for SL_L
# MODEL_PATH = "checkpoints/finetune_NS-SL_L_2048/PoseidonFinetune_NS-SL_L/finetune_NS-SL_run_L_2048/checkpoint-368800"
# DATA_PATH        = "datasets/NS-SL"
# DATASET_NAME     = "fluids.incompressible.ShearLayer"



#for PwC_B
# MODEL_PATH   = "checkpoints/finetune_NS-PwC_B_2048/PoseidonFinetune_NS-PwC_B/finetune_NS-PwC_run_B_2048/checkpoint-368800"
# DATA_PATH    = "datasets/NS-PwC"
# DATASET_NAME = "fluids.incompressible.PiecewiseConstants"
#for BB_B
# MODEL_PATH   =  "checkpoints/finetune_NS-BB_B_2048/PoseidonFinetune_NS-BB_B/finetune_NS-BB_run_B_2048/checkpoint-368800"
# DATA_PATH    = "datasets/NS-BB"
# DATASET_NAME = "fluids.incompressible.BrownianBridge"
#for SVS_B
# MODEL_PATH = "checkpoints/finetune_NS-SVS_B_2048/PoseidonFinetune_NS-SVS_B/finetune_NS-SVS_run_B_2048/checkpoint-368800"
# DATA_PATH        = "datasets/NS-SVS"
# DATASET_NAME     = "fluids.incompressible.VortexSheet"
#for SL_L
# MODEL_PATH = "checkpoints/finetune_NS-SL_B_2048/PoseidonFinetune_NS-SL_B/finetune_NS-SL_run_B_2048/checkpoint-368800"
# DATA_PATH        = "datasets/NS-SL"
# DATASET_NAME     = "fluids.incompressible.ShearLayer"


NUM_TRAJ     = 8      # test subset size
BATCH_SIZE   = 16
NUM_WORKERS  = min(os.cpu_count(), 4)
ASSUME_SL_LAYOUT = False  # set True for ShearLayer datasets (NS-SL)
USE_DENORM_FOR_L1 = True  # compute relative L1 on de-normalized fields (recommended)
# ========================

from scOT.model import ScOT
from scOT.problems.base import get_dataset
from scOT.problems.fluids.normalization_constants import CONSTANTS

@torch.no_grad()
def run_inference(model, dataset_name, data_path, num_traj, batch_size):
    device = next(model.parameters()).device
    ds = get_dataset(dataset=dataset_name, which="test",
                     num_trajectories=num_traj, data_path=data_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True)

    total_loss_elems, total_nel = 0.0, 0
    preds_all, labels_all = [], []

    for batch in dl:
        xb = batch["pixel_values"].to(device)
        yb = batch["labels"].to(device)
        tm = batch["time"].to(device)
        pm = batch["pixel_mask"].to(device)

        out   = model(pixel_values=xb, time=tm, pixel_mask=pm, labels=yb)
        loss  = out.loss          # Poseidon’s own loss (mean per element)
        preds = out.output        # (N, C, H, W) in normalized units

        n_el = preds.numel()
        total_loss_elems += loss.item() * n_el
        total_nel        += n_el

        preds_all.append(preds.cpu())
        labels_all.append(yb.cpu())

    preds_all  = torch.cat(preds_all,  dim=0)  # (N,C,H,W)
    labels_all = torch.cat(labels_all, dim=0)  # (N,C,H,W)
    dataset_loss = total_loss_elems / total_nel
    return dataset_loss, preds_all, labels_all

def denorm(x: torch.Tensor):
    """
    Denormalize Poseidon outputs using CONSTANTS['mean'], ['std'] with shape (C,1,1).
    x: (N,C,H,W) normalized → returns same shape de-normalized.
    """
    mean = CONSTANTS["mean"].to(x.device)  # (C,1,1)
    std  = CONSTANTS["std"].to(x.device)   # (C,1,1)
    return x * std.unsqueeze(0) + mean.unsqueeze(0)

@torch.no_grad()
def relative_l1(preds: torch.Tensor, targets: torch.Tensor, use_denorm: bool):
    """
    Relative L1:  mean( |ŷ - y| ) / mean( |y| )   over all elements and channels.
    If use_denorm=True, compute in physical units (recommended for reporting).
    """
    if use_denorm:
        preds   = denorm(preds)
        targets = denorm(targets)

    num = (preds - targets).abs().mean().item()
    den = targets.abs().mean().item() + 1e-12
    return num / den

@torch.no_grad()
def divergence_stats(preds: torch.Tensor, assume_sl_layout: bool = False):
    """
    Compute mean |∇·v| on the interior grid from de-normalized velocity (u,v).
    preds: (N,C,H,W) in normalized units.
    """
    x = denorm(preds)                         # de-normalize
    uv = x[:, 1:3, :, :]                      # channels 1,2 → (u,v)

    # axis handling
    if assume_sl_layout:
        u = uv[:, 0]        # (N,H,W)
        v = uv[:, 1]
    else:
        u = uv[:, 0].transpose(1, 2)  # (N,W,H)
        v = uv[:, 1].transpose(1, 2)  # (N,W,H)

    N, W, H = u.shape
    dx, dy = 1.0/(W-1), 1.0/(H-1)

    du_dx = (u[:, :, 2:] - u[:, :, :-2])/(2*dx)  # (N,W,H-2)
    dv_dy = (v[:, 2:, :] - v[:, :-2, :])/(2*dy)  # (N,W-2,H)

    du_dx = du_dx[:, 1:-1, :]  # → (N,W-2,H-2)
    dv_dy = dv_dy[:, :, 1:-1]  # → (N,W-2,H-2)

    div = du_dx + dv_dy                           # (N,W-2,H-2)
    per_sample = div.abs().flatten(1).mean(dim=1) # (N,)

    print("=== Divergence‐Free Check (de-normalized) ===")
    print(f"Mean abs ∇·v: {per_sample.mean().item():.6e}")
    print(f"Max  abs ∇·v: {per_sample.max().item():.6e}")
    print(f"Min  abs ∇·v: {per_sample.min().item():.6e}")
    return {
        "per_sample_mean_abs_div": per_sample,
        "mean": per_sample.mean().item(),
        "max":  per_sample.max().item(),
        "min":  per_sample.min().item(),
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScOT.from_pretrained(MODEL_PATH).to(device).eval()
    print("model is loaded")

    # 1) Poseidon’s own dataset-level loss (whatever the model defines as .loss)
    ds_loss, preds_all, labels_all = run_inference(
        model, DATASET_NAME, DATA_PATH, NUM_TRAJ, BATCH_SIZE
    )
    print(MODEL_PATH)
    print(f"[FP32] Poseidon dataset loss: {ds_loss:.6e}")

    # 2) Explicit relative L1 (normalized or de-normalized)
    rel_l1 = relative_l1(preds_all, labels_all, use_denorm=USE_DENORM_FOR_L1)
    unit = "de-normalized (physical)" if USE_DENORM_FOR_L1 else "normalized"
    print(f"[FP32] Relative L1 ({unit}): {rel_l1:.6e}")

    # 3) Divergence on de-normalized velocity
    _ = divergence_stats(preds_all, assume_sl_layout=ASSUME_SL_LAYOUT)

if __name__ == "__main__":
    main()