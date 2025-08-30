#!/usr/bin/env python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert
from torch.ao.nn import quantized as qnn
from scOT.model import ScOT, ScOTConfig
from scOT.problems.base import get_dataset
import numpy as np
from scOT.problems.fluids.normalization_constants import CONSTANTS

# ── USER SETTINGS ──────────────────────────────────────────────────────────────
#model path
#NS-PwC-B
FLOAT_MODEL_PATH = "checkpoints/finetune_NS-PwC_B/PoseidonFinetune_NS-PwC_B/finetune_NS-PwC_run_B"
#NS-SVS-B
#FLOAT_MODEL_PATH = "checkpoints/finetune_NS-SVS_B/PoseidonFinetune_NS-SVS_B/finetune_NS-SVS_run_B"
#NS-BB-B
#FLOAT_MODEL_PATH = "checkpoints/finetune_NS-BB_B/PoseidonFinetune_NS-BB_B/finetune_NS-BB_run_B"
#NS-SL
#FLOAT_MODEL_PATH = "checkpoints/finetune_NS-SL_B/PoseidonFinetune_NS-SL_B/finetune_NS-SL_run_B"

#quantized model path
QAT_MODEL_PATH   = "qat_int8_poseidon_NS-PwC_B.pt"
#QAT_MODEL_PATH   = "qat_int8_poseidon_NS-SVS_B.pt"
#QAT_MODEL_PATH   = "qat_int8_poseidon_NS-BB_B.pt"
#QAT_MODEL_PATH   = "qat_int8_poseidon_NS-SL_B.pt"

#data path
DATA_PATH        = "datasets/NS-PwC"
#DATA_PATH        = "datasets/NS-SVS"
#DATA_PATH        = "datasets/NS-BB"
#DATA_PATH        = "datasets/NS-SL"

#dataset name
DATASET_NAME     = "fluids.incompressible.PiecewiseConstants"
#DATASET_NAME     = "fluids.incompressible.VortexSheet"
#DATASET_NAME     = "fluids.incompressible.BrownianBridge"
#DATASET_NAME     = "fluids.incompressible.ShearLayer"


NUM_TRAJECTORIES = 8
BATCH_SIZE       = 16
# ────────────────────────────────────────────────────────────────────────────────

def get_mb(path: str):
    return os.path.getsize(path) / (1024**2)

# chaneg for eval model comes here
def eval_model(model):
    """Compute dataset‐level MSE and then report divergence on all preds."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    ds = get_dataset(
        dataset=DATASET_NAME,
        which="test",
        num_trajectories=NUM_TRAJECTORIES,
        data_path=DATA_PATH,
    )
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    total_se   = 0.0
    total_nel  = 0
    all_preds  = []

    with torch.no_grad():
        for batch in dl:
            xb = batch["pixel_values"].to(device)
            yb = batch["labels"].to(device)
            tm = batch["time"].to(device)

            out   = model(pixel_values=xb, time=tm)
            preds = out if isinstance(out, torch.Tensor) else out.output

            total_se  += F.mse_loss(preds, yb, reduction="sum").item()
            total_nel += preds.numel()

            all_preds.append(preds.cpu())

    mse = total_se / total_nel
    print(f"→ Dataset‐level MSE: {mse:.6e}")

    # stack and compute divergence
    preds_all = torch.cat(all_preds, dim=0)
    divergence_stats(preds_all)

    return mse

def dequantize_conv_modules(module: nn.Module):
    for name, child in list(module.named_children()):
        # regular Conv2d
        if isinstance(child, qnn.Conv2d):
            w_q, b_f = child.weight(), child.bias()
            w_f = w_q.dequantize()
            fconv = nn.Conv2d(
                in_channels = child.in_channels,
                out_channels = child.out_channels,
                kernel_size = child.kernel_size,
                stride = child.stride,
                padding = child.padding,
                dilation = child.dilation,
                groups = child.groups,
                bias = (b_f is not None),
            )
            fconv.weight.data.copy_(w_f)
            if b_f is not None:
                fconv.bias.data.copy_(b_f)
            setattr(module, name, fconv)

        # ConvTranspose2d
        elif isinstance(child, qnn.ConvTranspose2d):
            w_q, b_f = child.weight(), child.bias()
            w_f = w_q.dequantize()
            fconvT = nn.ConvTranspose2d(
                in_channels = child.in_channels,
                out_channels = child.out_channels,
                kernel_size = child.kernel_size,
                stride = child.stride,
                padding = child.padding,
                output_padding = child.output_padding,
                dilation = child.dilation,
                groups = child.groups,
                bias = (b_f is not None),
            )
            fconvT.weight.data.copy_(w_f)
            if b_f is not None:
                fconvT.bias.data.copy_(b_f)
            setattr(module, name, fconvT)

        else:
            dequantize_conv_modules(child)

def dequantize_linear_modules(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, qnn.Linear):
            w_q, b_f = child.weight(), child.bias()
            w_f = w_q.dequantize()
            f = nn.Linear(child.in_features, child.out_features,
                          bias=(b_f is not None))
            f.weight.data.copy_(w_f)
            if b_f is not None:
                f.bias.data.copy_(b_f)
            setattr(module, name, f)
        else:
            dequantize_linear_modules(child)

def evaluate_qat():
    print("→ Building QAT model architecture…")
    # 1) skeleton
    cfg   = ScOTConfig.from_pretrained(FLOAT_MODEL_PATH)
    model = ScOT(cfg)
    # 2) attach QAT stubs & convert to quantized modules
    torch.backends.quantized.engine = "qnnpack"
    model.train()  # prepare_qat requires train mode
    model.qconfig = get_default_qat_qconfig("qnnpack")
    model = prepare_qat(model, inplace=False)
    model = convert(model.eval(), inplace=False)
    # 3) load your int8 weights
    sd = torch.load(QAT_MODEL_PATH, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    # 4) dequantize every quantized layer → pure float
    dequantize_conv_modules(model)
    dequantize_linear_modules(model)
    # 5) evaluate exactly as FP32
    return eval_model(model)



def divergence_stats(preds: torch.Tensor):
    """
    preds2: torch.Tensor of shape (N, C_out, H, W)
    """
    device = preds.device
    # 1) de-normalize u & v (channels 1,2)
    means = torch.tensor(CONSTANTS["mean"][1:3], device=device).view(1,2,1,1)
    stds  = torch.tensor(CONSTANTS["std" ][1:3], device=device).view(1,2,1,1)
    uv_norm = preds[:, 1:3, :, :]       # (N,2,H,W)
    uv      = uv_norm * stds + means    # broadcast

    u = uv[:, 0].transpose(1,2)         # (N,W,H)
    v = uv[:, 1].transpose(1,2)

    # 2) grid spacing
    N, W, H = u.shape
    dx = 1.0/(W-1);  dy = 1.0/(H-1)

    # 3) central differences
    du_dx = (u[:, :, 2:] - u[:, :, :-2])/(2*dx)   # (N,W,H-2)
    dv_dy = (v[:, 2:, :] - v[:, :-2, :])/(2*dy)   # (N,W-2,H)

    # 4) crop to common interior (W-2,H-2)
    du_dx = du_dx[:, 1:-1, :]
    dv_dy = dv_dy[:, :, 1:-1]

    # 5) divergence
    div = du_dx + dv_dy                       # (N, W-2, H-2)

    div_np   = div.cpu().numpy()             # shape (N, W-2, H-2)
    abs_div  = np.abs(div_np).reshape(N, -1) # take absolute, then flatten each sample
    m        = abs_div.mean(axis=1)          # per-sample mean(|∇·v|)
    print("=== Divergence‐Free Check ===")
    print(f"Mean abs ∇·v: {m.mean():.6e}")
    print(f"Max  abs ∇·v: {m.max():.6e}")
    print(f"Min  abs ∇·v: {m.min():.6e}")

def main():
    # find the .bin inside FLOAT_MODEL_PATH
    if os.path.isdir(FLOAT_MODEL_PATH):
        float_ckpt = os.path.join(FLOAT_MODEL_PATH, "pytorch_model.bin")
    else:
        float_ckpt = FLOAT_MODEL_PATH

    print(f"Float checkpoint: {float_ckpt}")
    print(f"  size = {get_mb(float_ckpt):.2f} MB")
    print(f"QAT checkpoint:   {QAT_MODEL_PATH}")
    print(f"  size = {get_mb(QAT_MODEL_PATH):.2f} MB\n")

    # 1) FP32
    print("→ Loading FP32 model…")
    fp32 = ScOT.from_pretrained(FLOAT_MODEL_PATH).eval()
    fp32_mse = eval_model(fp32)
    print(f"FP32 - dataset MSE: {fp32_mse:.6e}")

    # 2) QAT
    qat_mse = evaluate_qat()
    print(f"QAT - dataset MSE:  {qat_mse:.6e}\n")

    # 3) report
    increase = 100.0 * (qat_mse - fp32_mse) / fp32_mse
    print(f"Loss increase: {increase:.2f}%")

if __name__ == "__main__":
    main()