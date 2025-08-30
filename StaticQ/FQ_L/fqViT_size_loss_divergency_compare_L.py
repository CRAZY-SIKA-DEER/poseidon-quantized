#!/usr/bin/env python
import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from scOT.model import ScOT, ScOTConfig
from scOT.problems.base import get_dataset
from scOT.problems.fluids.normalization_constants import CONSTANTS

# 1.a) Import the quantized LayerNorm you copied over
from poseidon.FQ_ViT.qint_layernorm import QIntLayerNorm
# 1.b) Import your PTF observer
from poseidon.FQ_ViT.ptf.ptf_observer import PtfObserver

# 1.c) BitType helper (if not already defined)
class BitType:
    def __init__(self, lower, upper):
        self.lower_bound = lower
        self.upper_bound = upper

# 1.d) DummyQuantizer just holds the PTF scale
class DummyQuantizer:
    def __init__(self, scale):
        self.scale = scale

# 1.e) Recursively replace nn.LayerNorm → QIntLayerNorm
def replace_layernorms(module, in_q, out_q):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm):
            # create QIntLayerNorm same shape
            qln = QIntLayerNorm(child.normalized_shape[0],
                                 eps=child.eps,
                                 elementwise_affine=child.elementwise_affine)
            # copy learned params
            qln.weight.data = child.weight.data.clone()
            qln.bias.data   = child.bias.data.clone()
            # switch into integer mode
            qln.mode = 'int'
            # attach our PTF scale
            qln.in_quantizer  = in_q
            qln.out_quantizer = out_q
            setattr(module, name, qln)
        else:
            replace_layernorms(child, in_q, out_q)



# ── USER SETTINGS ──────────────────────────────────────────────────────────────
FLOAT_MODEL_PATH  = "checkpoints/finetune_NS-PwC_B_2048/PoseidonFinetune_NS-PwC_B/finetune_NS-PwC_run_B_2048/checkpoint-368800"
FQ_MODEL_PATH     = "./FQ_ViT/fq_poseidon_B_PwC_morebatch.pt"

DATA_PATH         = "datasets/NS-PwC"
DATASET_NAME      = "fluids.incompressible.PiecewiseConstants"
NUM_TRAJECTORIES  = 8
BATCH_SIZE        = 16
# ────────────────────────────────────────────────────────────────────────────────

def get_mb(path: str):
    return os.path.getsize(path) / (1024**2)

def divergence_stats(preds: torch.Tensor):
    device = preds.device
    means = torch.tensor(CONSTANTS["mean"][1:3], device=device).view(1,2,1,1)
    stds  = torch.tensor(CONSTANTS["std" ][1:3], device=device).view(1,2,1,1)

    # denormalize u,v
    uv = preds[:,1:3] * stds + means

    # swap so axis-1 = x, axis-2 = y
    u = uv[:,0].transpose(1,2)
    v = uv[:,1].transpose(1,2)

    N, W, H = u.shape
    dx, dy = 1.0/(W-1), 1.0/(H-1)

    du_dx = (u[:,:,2:] - u[:,:,:-2])/(2*dx)
    dv_dy = (v[:,2:,:] - v[:,:-2,:])/(2*dy)
    du_dx = du_dx[:,1:-1,:]
    dv_dy = dv_dy[:,:,1:-1]

    div = du_dx + dv_dy                      # shape (N, W-2, H-2)
    abs_div = np.abs(div.cpu().numpy().reshape(N, -1))

    m = abs_div.mean(axis=1)
    print("=== Divergence‐Free Check ===")
    print(f"Mean abs ∇·v: {m.mean():.6e}")
    print(f"Max  abs ∇·v: {m.max():.6e}")
    print(f"Min  abs ∇·v: {m.min():.6e}")

def eval_model_with_model_loss(model):
    """Eval using model’s own .loss, accum per‐element, then compute dataset‐avg."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    ds = get_dataset(
        dataset         = DATASET_NAME,
        which           = "test",
        num_trajectories= NUM_TRAJECTORIES,
        data_path       = DATA_PATH,
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

            out = model(
                pixel_values=xb,
                time        =tm,
                labels      =yb,
                pixel_mask  =pm,
            )
            # out.loss is the mean per‐element over this batch
            batch_loss = out.loss
            preds      = out.output

            n_elems = preds.numel()
            total_loss_elems += batch_loss.item() * n_elems
            total_nel        += n_elems

            all_preds.append(preds.cpu())

    dataset_loss = total_loss_elems / total_nel
    print(f"→ Dataset‐level loss (model’s): {dataset_loss:.6e}")

    preds_all = torch.cat(all_preds, dim=0)
    divergence_stats(preds_all)

    return dataset_loss

def evaluate_fq():
    print("→ Loading FQ (fake-quant) model…")
    cfg   = ScOTConfig.from_pretrained(FLOAT_MODEL_PATH)
    model = ScOT(cfg)
    sd    = torch.load(FQ_MODEL_PATH, map_location="cpu")
    model.load_state_dict(sd)
    model.to(DEVICE).eval()

    # ——— Calibrate PTF for LayerNorm scales ———————————————————————
    # (reuse a few batches from training or val)
    calib_ds = get_dataset(
      dataset         = DATASET_NAME,
      which           = "train",   # or "val" if preferred  
      num_trajectories=NUM_TRAJECTORIES,
      data_path       = DATA_PATH,
    )
    calib_dl = DataLoader(calib_ds, batch_size=BATCH_SIZE, shuffle=True)

    # init PTF observer for activations
    bit_type = BitType(0, 255)
    ptf_obs = PtfObserver(
      module_type="act",
      bit_type=bit_type,
      calibration_mode="layer_wise"
    )

    seen = 0
    with torch.no_grad():
        for batch in calib_dl:
            xb = batch["pixel_values"].to(DEVICE)
            ptf_obs.update(xb)
            seen += 1
            if seen >= CALIB_BATCHES:
                break

    scale, zp = ptf_obs.get_quantization_params(xb)
    print("▶️  PTF LayerNorm scales calibrated.")

    # build dummy quantizer carrying that scale
    dq = DummyQuantizer(scale.to(DEVICE))

    # patch all LayerNorm → QIntLayerNorm
    replace_layernorms(model, in_q=dq, out_q=dq)
    print("🔧  All LayerNorms replaced with QIntLayerNorm.")

    # ——— Now evaluate as usual ——————————————————————————————
    return eval_model_with_model_loss(model)


def main():
    # checkpoint size print
    float_ckpt = (os.path.isdir(FLOAT_MODEL_PATH)
                  and os.path.join(FLOAT_MODEL_PATH, "pytorch_model.bin")
                  or FLOAT_MODEL_PATH)

    print(f"Float ckpt : {float_ckpt}  ({get_mb(float_ckpt):.2f} MB)")
    print(f"FQ   ckpt   : {FQ_MODEL_PATH}  ({get_mb(FQ_MODEL_PATH):.2f} MB)\n")

    # 1) FP32
    print("→ Loading FP32 model…")
    fp32 = ScOT.from_pretrained(FLOAT_MODEL_PATH).eval()
    fp32_loss = eval_model_with_model_loss(fp32)
    print(f"FP32 - dataset loss: {fp32_loss:.6e}\n")

    # 2) FQ
    fq_loss = evaluate_fq()
    print(f"FQ   - dataset loss: {fq_loss:.6e}\n")

    # 3) relative increase
    inc = 100.0 * (fq_loss - fp32_loss) / fp32_loss
    print(f"Loss increase (FQ vs FP32): {inc:.2f}%")

if __name__ == "__main__":
    main()