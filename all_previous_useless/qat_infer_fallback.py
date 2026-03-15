#!/usr/bin/env python
"""
qat_infer_fallback.py  ––  Load your QAT int8 checkpoint, dequantize every
Conv2d/Linear back to float32, then run inference on GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.ao.nn import quantized as qnn
from scOT.model import ScOT, ScOTConfig
from scOT.problems.base import get_dataset

# === User fills in ===
FLOAT_MODEL_PATH = "checkpoints/finetune_NS-PwC_B/PoseidonFinetune_NS-PwC_B/finetune_NS-PwC_run_B"
QAT_MODEL_PATH   = "qat_int8_poseidon.pt"
DATA_PATH        = "datasets/NS-PwC"
DATASET_NAME     = "fluids.incompressible.PiecewiseConstants"
NUM_TRAJECTORIES = 4
BATCH_SIZE       = 10

# === 1) Build & convert a QAT model architecture (ignore float weights) ===
config = ScOTConfig.from_pretrained(FLOAT_MODEL_PATH)
model  = ScOT(config)

# must be in train mode for prepare_qat, but we won't actually train
torch.backends.quantized.engine = "qnnpack"
model.train()
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert
model.qconfig = get_default_qat_qconfig("qnnpack")
model = prepare_qat(model, inplace=False)
model = convert(model.eval(), inplace=False)

# load your saved int8 state dict
state = torch.load(QAT_MODEL_PATH, map_location="cpu")
model.load_state_dict(state)
model.eval()

# === 2) Recursively replace quantized ops with float modules ===
def dequantize_conv_modules(module: nn.Module):
    for name, child in list(module.named_children()):
        # regular conv2d
        if isinstance(child, qnn.Conv2d):
            w_q = child.weight()      # QuantizedTensor
            b_f = child.bias()        # float Tensor or None
            w_f = w_q.dequantize()    # FloatTensor

            fconv = nn.Conv2d(
                child.in_channels, child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=(b_f is not None),
            )
            fconv.weight.data.copy_(w_f)
            if b_f is not None:
                fconv.bias.data.copy_(b_f)
            setattr(module, name, fconv)

        # transpose (de)conv
        elif isinstance(child, qnn.ConvTranspose2d):
            w_q = child.weight()
            b_f = child.bias()
            w_f = w_q.dequantize()

            fconvT = nn.ConvTranspose2d(
                child.in_channels, child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                output_padding=child.output_padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=(b_f is not None),
            )
            fconvT.weight.data.copy_(w_f)
            if b_f is not None:
                fconvT.bias.data.copy_(b_f)
            setattr(module, name, fconvT)

        else:
            # recurse into submodules
            dequantize_conv_modules(child)

def dequantize_linear_modules(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, qnn.Linear):
            w_q = child.weight()
            b_f = child.bias()
            w_f = w_q.dequantize()

            flinear = nn.Linear(
                child.in_features, child.out_features,
                bias=(b_f is not None)
            )
            flinear.weight.data.copy_(w_f)
            if b_f is not None:
                flinear.bias.data.copy_(b_f)
            setattr(module, name, flinear)
        else:
            dequantize_linear_modules(child)

dequantize_conv_modules(model)
dequantize_linear_modules(model)

# === 3) Run inference on GPU as ordinary float model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

test_ds = get_dataset(
    dataset          = DATASET_NAME,
    which            = "test",
    num_trajectories = NUM_TRAJECTORIES,
    data_path        = DATA_PATH,
)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

total_loss = 0.0
n_batches  = 0

for batch in test_dl:
    xb = batch["pixel_values"].to(device)
    yb = batch["labels"].to(device)
    tm = batch["time"].to(device)
    pm = batch["pixel_mask"].to(device)

    with torch.no_grad():
        out   = model(pixel_values=xb, time=tm, labels=None)
        preds = out if isinstance(out, torch.Tensor) else out.output

    total_loss += F.mse_loss(preds, yb).item()
    n_batches  += 1

print(f"Average MSE (float fallback): {total_loss / n_batches:.6f}")
