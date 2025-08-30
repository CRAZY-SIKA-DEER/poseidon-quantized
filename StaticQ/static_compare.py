#!/usr/bin/env python
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

# quant utilities
from torch.quantization import get_default_qconfig, prepare, convert, fuse_modules

# your model + dataset
from scOT.model import ScOT
from scOT.problems.base import get_dataset
from scOT.problems.fluids.normalization_constants import CONSTANTS

# ── USER SETTINGS ──────────────────────────────────────────────────────────────
FLOAT_MODEL_PATH  = "checkpoints/finetune_NS-PwC_L_2048/PoseidonFinetune_NS-PwC_L/finetune_NS-PwC_run_L_2048/checkpoint-368800"
STATIC_MODEL_PATH = "./static_int8_poseidon_NS-PwC_L_2048.pt"
DATA_PATH         = "datasets/NS-PwC"
DATASET_NAME      = "fluids.incompressible.PiecewiseConstants"

NUM_TRAJECTORIES  = 8
BATCH_SIZE        = 16
# ────────────────────────────────────────────────────────────────────────────────

def get_mb(path: str):
    return os.path.getsize(path) / (1024 ** 2)

def divergence_stats(preds: torch.Tensor):
    # denormalize u,v
    means = torch.as_tensor(CONSTANTS["mean"][1:3], device=preds.device).view(1,2,1,1)
    stds  = torch.as_tensor(CONSTANTS["std" ][1:3], device=preds.device).view(1,2,1,1)
    uv = preds[:,1:3] * stds + means

    u = uv[:,0].transpose(1,2)
    v = uv[:,1].transpose(1,2)
    N, W, H = u.shape
    dx, dy = 1.0/(W-1), 1.0/(H-1)

    du_dx = (u[:,:,2:] - u[:,:,:-2])/(2*dx)
    dv_dy = (v[:,2:,:] - v[:,:-2,:])/(2*dy)
    du_dx = du_dx[:,1:-1,:]
    dv_dy = dv_dy[:,:,1:-1]

    div = du_dx + dv_dy
    abs_div = np.abs(div.cpu().numpy().reshape(N, -1))

    m = abs_div.mean(axis=1)
    print("=== Divergence‐Free Check ===")
    print(f"Mean abs ∇·v: {m.mean():.6e}")
    print(f"Max  abs ∇·v: {m.max():.6e}")
    print(f"Min  abs ∇·v: {m.min():.6e}")

def eval_model_with_model_loss(model, device=None):
    # allow forcing CPU for quantized model
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model.to(device).eval()

    ds = get_dataset(
        dataset          = DATASET_NAME,
        which            = "test",
        num_trajectories = NUM_TRAJECTORIES,
        data_path        = DATA_PATH,
    )
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    total_loss_elems = 0.0
    total_nel        = 0
    all_preds        = []

    with torch.no_grad():
        for batch in dl:
            xb = batch["pixel_values"].to(device)
            yb = batch["labels"].to(device)
            tm = batch["time"].to(device)
            pm = batch["pixel_mask"].to(device)

            out = model(pixel_values=xb, time=tm, pixel_mask=pm, labels=yb)
            batch_loss = out.loss
            preds      = out.output

            n_elems = preds.numel()
            total_loss_elems += batch_loss.item() * n_elems
            total_nel        += n_elems
            all_preds.append(preds.cpu())

    dataset_loss = total_loss_elems / total_nel
    print(f"→ Dataset‐level loss: {dataset_loss:.6e}")
    preds_all = torch.cat(all_preds, dim=0)
    divergence_stats(preds_all)
    return dataset_loss

def load_static_quant_model():
    """Rebuilds the quantized model graph (with original scales) and loads the int8 weights."""
    # 1) Instantiate the float model
    model = ScOT.from_pretrained(FLOAT_MODEL_PATH).eval()

    # 2) Monkey‑patch / fuse
    def apply_fuse(self):
        for _, block in self.base_model.named_children():
            if hasattr(block, 'conv1') and hasattr(block, 'bn1') and hasattr(block, 'relu1'):
                fuse_modules(block, ['conv1','bn1','relu1','conv2','bn2','relu2'], inplace=True)
    ScOT.fuse_model = apply_fuse
    model.fuse_model()

    # 3) Prepare & convert to quantized graph _without_ calibration
    model.qconfig = get_default_qconfig('qnnpack')
    prepare(model, inplace=True)
    qmodel = convert(model, inplace=False)

    # 4) Load your saved int8 weights _and_ the stored scale/zero_point buffers
    sd = torch.load(STATIC_MODEL_PATH, map_location='cpu')
    qmodel.load_state_dict(sd)

    # 5) Force CPU for QNNPACK
    qmodel.to('cpu')
    return qmodel

def main():
    # 1) Size comparison
    print(f"Float ckpt  : {FLOAT_MODEL_PATH}    ({get_mb(FLOAT_MODEL_PATH):.2f} MB)")
    print(f"Static ckpt : {STATIC_MODEL_PATH}    ({get_mb(STATIC_MODEL_PATH):.2f} MB)")
    size_inc = 100.0 * (get_mb(STATIC_MODEL_PATH) - get_mb(FLOAT_MODEL_PATH)) / get_mb(FLOAT_MODEL_PATH)
    print(f"Size change : {size_inc:.2f}%\n")

    # 2) FP32 eval (on GPU if available)
    print("→ Evaluating FP32 model…")
    fp32 = ScOT.from_pretrained(FLOAT_MODEL_PATH)
    fp32_loss = eval_model_with_model_loss(fp32)
    print()

    # 3) INT8 eval (forced CPU)
    print("→ Evaluating static‑quant INT8 model…")
    int8 = load_static_quant_model()
    int8_loss = eval_model_with_model_loss(int8, device='cpu')
    print()

    # 4) Loss increase
    inc = 100.0 * (int8_loss - fp32_loss) / fp32_loss
    print(f"Loss increase (int8 vs fp32): {inc:.2f}%")

if __name__=="__main__":
    main()
