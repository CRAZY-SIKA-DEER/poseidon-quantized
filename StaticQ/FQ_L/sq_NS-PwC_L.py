#!/usr/bin/env python
"""
fq_poseidon.py  –– Fully‐Quantized (PTF) Post‐Training Quantization for Poseidon
"""

import os
import torch
from torch.utils.data import DataLoader
from scOT.model import ScOT
from scOT.problems.base import get_dataset

# 1) IMPORT YOUR PTF OBSERVER
# Adjust this import path to where you placed ptf.py
from FQ_ViT.ptf import PtfObserver  

# ─── BitType helper for PtfObserver ──────────────────────────────────────────
class BitType:
    def __init__(self, lower, upper):
        self.lower_bound = lower
        self.upper_bound = upper

# ──────────────────────────────────────────────────────────────────────────────
# 0. USER SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
MODEL_PATH   = "checkpoints/finetune_NS-PwC_B_2048/PoseidonFinetune_NS-PwC_B/finetune_NS-PwC_run_B_2048/checkpoint-368800"
DATA_PATH    = "datasets/NS-PwC"
DATASET_NAME = "fluids.incompressible.PiecewiseConstants"
SAVE_PATH    = "fq_poseidon_B_PwC_morebatch.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CALIB_BATCHES =  128    # how many batches to calibrate on
BATCH_SIZE    = 20     # batch size for calibration
# ──────────────────────────────────────────────────────────────────────────────

def quantize_tensor(x, scale, zp, qmin=0, qmax=255):
    """Affine‐quantize and dequantize a float tensor in‐place."""
    # x_q = clamp(round(x/scale + zp), qmin, qmax)
    # x_deq = (x_q - zp) * scale
    x_div = x / scale
    x_q   = (x_div + zp).round().clamp(qmin, qmax)
    return (x_q - zp) * scale

def main():
    # —── Load pretrained Poseidon model ─────────────────────────────────────────
    model = ScOT.from_pretrained(MODEL_PATH)
    model.to(DEVICE).eval()

    # —── Prepare a small validation set for calibration ────────────────────────
    val_ds = get_dataset(
        dataset=DATASET_NAME,
        which="val",        # use "train" or "val" split as you prefer
        num_trajectories=-1,  # full trajectories; just grabbing a few batches
        data_path=DATA_PATH,
    )
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # —── Initialize PTF Observer ───────────────────────────────────────────────
    # We hard‐code 8‐bit here; PtfObserver will use bit_type.upper_bound=255, lower_bound=0
    # calibration_mode="layer_wise" to share min/max across channels for scale8 computation.
    bit_type = BitType(0, 255)        # for 8-bit quantization
    ptf = PtfObserver(
        module_type="act",
        bit_type=bit_type,
        calibration_mode="channel_wise" 
    )
    # If your PtfObserver expects a custom BitType object, instantiate that instead.

    # —── Run calibration ────────────────────────────────────────────────────────
    print(f"🛠  Calibrating PTF observer on {CALIB_BATCHES} batches…")
    seen = 0
    with torch.no_grad():
        for batch in val_dl:
            xb = batch["pixel_values"].to(DEVICE)   # adjust key if different
            ptf.update(xb)
            seen += 1
            if seen >= CALIB_BATCHES:
                break

    # —── Compute scale & zero‐point ────────────────────────────────────────────
    scale, zp = ptf.get_quantization_params(xb)
    # scale.shape == [C], zp is scalar or shape [1]
    print("▶️  Computed per‐channel PTF scales and zero‐point.")

    # —── Quantize all weights in the model (per-layer MinMax) ─────────────────
    print("🔧  Quantizing model weights…")
    qmin, qmax = 0, 255
    eps = 1e-8
    for name, param in model.named_parameters():
        if "weight" in name and param.ndim >= 2:
            w = param.data

            # 1) compute this layer’s min/max
            w_min, w_max = w.min(), w.max()
            # 2) derive scale and zero-point
            scale_w = (w_max - w_min) / float(qmax - qmin)
            scale_w = torch.clamp(scale_w, min=eps)
            zero_point_w = (qmin - torch.round(w_min / scale_w)).clamp(qmin, qmax)

            # 3) quantize + dequantize
            w_div = w / scale_w
            w_q   = (w_div + zero_point_w).round().clamp(qmin, qmax)
            param.data = (w_q - zero_point_w) * scale_w

    # —── Save quantized checkpoint ────────────────────────────────────────────
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ Quantized model saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()