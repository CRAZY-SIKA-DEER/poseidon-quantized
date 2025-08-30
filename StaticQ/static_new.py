#!/usr/bin/env python
"""
static_poseidon.py  –– Post‑training static quantization script for a Poseidon/scOT model using QNNPACK on ARM.

• loads the float model from MODEL_PATH
• fuses Conv+BN(+ReLU) blocks
• attaches a static qconfig
• runs a calibration pass to collect activations
• converts and saves an int8 CPU‑ready model

Only three lines need editing: MODEL_PATH, DATA_PATH, DATASET_NAME.
"""

import torch

class QuantWrapper(torch.nn.Module):
    def __init__(self, quant_model):
        super().__init__()
        self.quant_model = quant_model

    def forward(self, xb, tm, pm, yb):
        # map your positional inputs to the right keywords
        return self.quant_model(
            pixel_values=xb,
            time=tm,
            pixel_mask=pm,
            labels=yb,
        )


# ================================================================
# 0. User‑editable paths
# ================================================================
MODEL_PATH   = "checkpoints/finetune_NS-PwC_L_2048/PoseidonFinetune_NS-PwC_L/finetune_NS-PwC_run_L_2048/checkpoint-368800"
DATA_PATH    = "datasets/NS-PwC"
DATASET_NAME = "fluids.incompressible.PiecewiseConstants"
SAVE_PATH    = "./static_int8_poseidon_NS-PwC_L_2048.pt"

# ================================================================
# 1. Imports
# ================================================================
import argparse
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.quantization import (
    get_default_qconfig,
    prepare,
    convert,
    fuse_modules,
)
from scOT.model import ScOT                                # Poseidon backbone
from scOT.problems.base import get_dataset

# ================================================================
# Monkey-patch fuse model
# ================================================================

# 2. Monkey‑patch fuse_model into ScOT
def apply_fuse(self):
    # iterate over your backbone’s blocks
    for name, block in self.base_model.named_children():
        if hasattr(block, 'conv1') and hasattr(block, 'bn1') and hasattr(block, 'relu1'):
            fuse_modules(block, ['conv1', 'bn1', 'relu1',
                                 'conv2', 'bn2', 'relu2'], inplace=True)

# inject it
ScOT.fuse_model = apply_fuse

# ================================================================
# 2. CLI for calibration / batch size
# ================================================================
p = argparse.ArgumentParser()
p.add_argument("--calibration_batches", type=int, default=20,
               help="Number of batches to run for calibration")
p.add_argument("--batch_size",           type=int, default=16)
p.add_argument("--device",               type=str, default="cpu")
args = p.parse_args()

# ================================================================
# 3. Set device and quant engine
# ================================================================
device = torch.device(args.device)
torch.backends.quantized.engine = 'qnnpack'   # ARM/mobile backend

# ================================================================
# 4. Prepare dataset for calibration
# ================================================================
calib_ds = get_dataset(
    dataset          = DATASET_NAME,
    which            = "train",
    num_trajectories = 256,      # smaller subset for calibration
    data_path        = DATA_PATH,
)
calib_loader = DataLoader(
    calib_ds,
    batch_size  = args.batch_size,
    shuffle     = True,
    num_workers = min(os.cpu_count(), 16),
    pin_memory  = True,
)

# ================================================================
# 5. Load FLOAT model & move to device
# ================================================================
model = ScOT.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# ================================================================
# 6. Fuse fusible layers
# ================================================================
# Make sure your ScOT class implements fuse_model() to fold Conv+BN(+ReLU)
model.fuse_model()

# ================================================================
# 7. Attach static quantization config & prepare
# ================================================================
model.qconfig = get_default_qconfig('qnnpack')
prepare(model, inplace=True)
print(f"🔧 Model prepared for static quantization (QNNPACK) with qconfig: {model.qconfig}")

# ================================================================
# 8. Calibration pass (collect statistics)
# ================================================================
with torch.no_grad():
    for i, batch in enumerate(calib_loader):
        if i >= args.calibration_batches:
            break
        xb = batch["pixel_values"].to(device)
        tm = batch["time"].to(device)
        pm = batch["pixel_mask"].to(device)
        # run forward to collect observer stats
        yb = batch["labels"].to(device) 
        _ = model(
            pixel_values=xb,
            time=tm,
            pixel_mask=pm,
            labels=yb,                         # ← pass them in
        )
print(f"✅ Completed {min(args.calibration_batches, i+1)} calibration batches.")

# ================================================================
# 9. Convert to quantized INT8 model
# ================================================================
model_int8 = convert(model, inplace=False)

# ── Replace all of the above with just this ───────────────────────────
# Trace the eager‐quantized model (on CPU) so that
# the int8 kernels + your original scales are baked in,
# and your Python control flow (“if”/“for”) still runs.
model_int8_cpu = model_int8.eval().to('cpu')

# pull one batch (CPU tensors) for tracing
batch = next(iter(calib_loader))
xb, tm, pm, yb = (batch["pixel_values"],
                 batch["time"],
                 batch["pixel_mask"],
                 batch["labels"])

# wrap to bind positional args to keywords
wrapped = QuantWrapper(model_int8_cpu)
traced_int8 = torch.jit.trace(wrapped, (xb, tm, pm, yb), strict=False)
torch.jit.save(traced_int8, "poseidon_int8_traced.pt")
print("✔ Saved traced quantized model to poseidon_int8_traced.pt")


# ================================================================
# 10. Save quantized model
# ================================================================
torch.save(model_int8.state_dict(), SAVE_PATH)
print(f"💾 Saved static int8 model to {SAVE_PATH}")


import torch

from torch.quantization import get_default_qconfig, prepare, convert, fuse_modules
from scOT.model import ScOT
from torch.utils.data import DataLoader
from scOT.problems.base import get_dataset

