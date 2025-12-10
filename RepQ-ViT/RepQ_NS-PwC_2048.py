#!/usr/bin/env python
# RepQ-ViT style PTQ for Poseidon / ScOT (no Poseidon source edits)

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os, sys, time, random, numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scOT.problems.fluids.normalization_constants import CONSTANTS

# ==== EDIT THESE ====
REPO_REPQ_CLASSIFICATION = "./RepQ-ViT/classification"   # path to the 'classification' dir of RepQ-ViT
#for PwC_L
MODEL_PATH   = "models/NS-PwC-B"
DATA_PATH    = "dataset/NS-PwC"
DATASET_NAME = "fluids.incompressible.PiecewiseConstants"
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



W_BITS, A_BITS = 8, 8
CALIB_BATCHSIZE = 8      # images per step for calib
CALIB_STEPS     = 8      # how many batches for calib
VAL_BATCHSIZE   = 16
VAL_STEPS       = 50
DEVICE = "cuda"
SEED   = 0
NUM_TRAJECTORIES = 8    # <-- add this (test split size)
BATCH_SIZE       = 16   # (you already use this below; keep it here with NUM_TRAJECTORIES)
# ====================

# import RepQ-ViT classification.quant
sys.path.insert(0, REPO_REPQ_CLASSIFICATION)
from quant.quant_model import quant_model, set_quant_state  # wraps modules with Quant* wrappers
# --- SKIP quant for time-conditioned LayerNorm hypernets (γ(t), β(t)) ---
from quant.quant_modules import QuantLinear  # adjust import path if needed

# Poseidon imports
from scOT.model import ScOT
from scOT.problems.base import get_dataset
import torch.nn.functional as F
from torch.nn import Parameter

def seed_all(seed=0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True

@torch.no_grad()
def repq_ln_reparam_generic(q_model: nn.Module):
    """
    RepQ-ViT LayerNorm reparameterization:
      for each LayerNorm 'norm1' / 'norm2' in a block,
      find the next Linear (attn.qkv OR attn.query/key/value; mlp.fc1; reduction),
      read its *per-channel* activation quant (delta, zero_point),
      convert to *layer-wise* by updating LN.{weight,bias} and compensating Linear.{weight,bias},
      then set that next input quantizer to layer-wise and force reinit of its weight quantizer.
    """
    module_map = {n:m for n,m in q_model.named_modules()}

    def get_parent(name):
        i = name.rfind('.')
        return (name[:i] if i!=-1 else ""), (name[i+1:] if i!=-1 else name)

    for name, ln in q_model.named_modules():
        if not isinstance(ln, nn.LayerNorm):
            continue
        parent_name, local = get_parent(name)
        parent = module_map.get(parent_name, None)
        if parent is None: 
            continue

        # collect candidate "next" modules
        candidates = []
        if local == "norm1":
            # attention side: prefer fused qkv; else q/k/v
            for path in (f"{parent_name}.attn.qkv",
                         f"{parent_name}.attn.query",
                         f"{parent_name}.attn.key",
                         f"{parent_name}.attn.value"):
                m = module_map.get(path, None)
                if isinstance(m, nn.Linear):
                    candidates.append(m)
        elif local == "norm2":
            m = module_map.get(f"{parent_name}.mlp.fc1", None)
            if isinstance(m, nn.Linear): candidates.append(m)
        else:
            # stage reduction norms (optional)
            m = module_map.get(f"{parent_name}.reduction", None)
            if isinstance(m, nn.Linear): candidates.append(m)

        if not candidates:
            continue

        # read per-channel activation quant stats (must be initialized via calib pass)
        deltas, zps = [], []
        for nm in candidates:
            iq = getattr(nm, "input_quantizer", None)
            if iq is None or not getattr(iq, "inited", False):
                # if not inited, skip this LN (you likely need more calib passes)
                deltas = []; break
            deltas.append(iq.delta.reshape(-1))
            zps.append(iq.zero_point.reshape(-1))
        if not deltas: 
            continue

        act_delta = torch.stack(deltas, 0).mean(0)        # [D]
        act_zp    = torch.stack(zps,    0).mean(0)        # [D]
        target_delta = act_delta.mean()                    # scalar
        target_zp    = act_zp.mean()                       # scalar

        act_min    = -act_zp * act_delta                   # [D]
        target_min = -target_zp * target_delta             # scalar
        r = (act_delta / (target_delta + 1e-12))           # r1, [D]
        b = (act_min / (r + 1e-12)) - target_min           # s*r2 term, [D]

        # update LN affine
        D = ln.normalized_shape[0]
        if r.numel() != D:
            r = r[:D]; b = b[:D]
        ln.weight.data = ln.weight.data / r
        ln.bias.data   = ln.bias.data   / r - b

        # compensate each next Linear (scale input channels; shift bias)
        for nm in candidates:
            W = nm.weight.data
            nm.weight.data = W * r.unsqueeze(0)           # scale columns
            if nm.bias is None:
                nm.bias = Parameter(torch.zeros(nm.out_features, device=W.device, dtype=W.dtype))
            nm.bias.data = nm.bias.data + (nm.weight.data @ b.reshape(-1,1)).reshape(-1)

            # flip that input quantizer to layer-wise and set target stats
            iq = nm.input_quantizer
            iq.channel_wise = False
            iq.delta = target_delta.to(iq.delta.device, iq.delta.dtype)
            iq.zero_point = target_zp.to(iq.zero_point.device, iq.zero_point.dtype)
            # weight quantizer must re-init after W change
            if hasattr(nm, "weight_quantizer"):
                nm.weight_quantizer.inited = False

def main():
    seed_all(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # -------- Poseidon datasets
    train_ds = get_dataset(DATASET_NAME, which="train", num_trajectories=2048, data_path=DATA_PATH)
    try:
        val_ds = get_dataset(DATASET_NAME, which="val",  num_trajectories=256, data_path=DATA_PATH)
    except Exception:
        val_ds = get_dataset(DATASET_NAME, which="test", num_trajectories=256, data_path=DATA_PATH)

    calib_loader = DataLoader(train_ds, batch_size=CALIB_BATCHSIZE, shuffle=True,
                              num_workers=min(os.cpu_count(), 16), pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=VAL_BATCHSIZE, shuffle=False,
                              num_workers=min(os.cpu_count(), 16), pin_memory=True)

    # -------- Build float model
    model = ScOT.from_pretrained(MODEL_PATH).to(device).eval()

    # -------- Wrap with RepQ-ViT quant modules
    wq_params = {'n_bits': W_BITS, 'channel_wise': True}
    aq_params = {'n_bits': A_BITS, 'channel_wise': False}
    q_model = quant_model(model, input_quant_params=aq_params, weight_quant_params=wq_params)
    q_model.to(device).eval()

    # -------- Initial quantization (calibration forward)
    set_quant_state(q_model, input_quant=True, weight_quant=False)
    with torch.no_grad():
        steps = 0
        for batch in calib_loader:
            if steps >= CALIB_STEPS: break
            x  = batch["pixel_values"].to(device)
            t  = batch.get("time", None)
            pm = batch.get("pixel_mask", None)
            y  = batch.get("labels", None)
            _ = q_model(pixel_values=x,
                        time=(t.to(device) if t is not None else None),
                        pixel_mask=(pm.to(device) if pm is not None else None),
                        labels=(y.to(device) if y is not None else None))
            steps += 1
    print(f"Calibrated quantizers on {steps} batches.")

    # -------- Scale reparameterization (LN -> next Linear)
    with torch.no_grad():
        repq_ln_reparam_generic(q_model)
    print("Applied RepQ scale reparameterization.")

    # -------- Guard quant on time-conditioned LN hypernets + zero-range weights
    from quant.quant_modules import QuantLinear  # adjust path if needed

    # 1) Disable quant on tiny linears that generate γ(t)/β(t) for time-conditioned norms
    disabled = 0
    for name, m in q_model.named_modules():
        if isinstance(m, QuantLinear):
            # heuristics: these layers usually sit under modules named like "*.norm.*"
            if (".norm." in name) and (name.endswith(".weight") or name.endswith(".bias")):
                m.set_quant_state(input_quant=False, weight_quant=False)
                disabled += 1
    print(f"Disabled quant on {disabled} time-conditioned norm hypernet linears.")

    # 2) For any remaining Linear with zero-range rows, avoid per-channel W-quant
    fallback = 0
    for name, m in q_model.named_modules():
        if isinstance(m, QuantLinear):
            W = m.weight.data
            if not torch.isfinite(W).all():
                # keep weights FP for pathological tensors
                m.set_quant_state(input_quant=True, weight_quant=False)
                fallback += 1
                continue
            row_rng = (W.max(dim=1).values - W.min(dim=1).values)
            if (row_rng == 0).any():
                m.weight_quantizer.channel_wise = False   # switch to layer-wise W quant
                m.weight_quantizer.inited = False         # re-init next pass
                fallback += 1
    print(f"Applied weight-quant fallbacks on {fallback} Linear layers.")


    # -------- Re-calibration (since we changed W/b)
    set_quant_state(q_model, input_quant=True, weight_quant=True)
    with torch.no_grad():
        steps = 0
        for batch in calib_loader:
            if steps >= max(2, CALIB_STEPS//2): break
            x  = batch["pixel_values"].to(device)
            t  = batch.get("time", None)
            pm = batch.get("pixel_mask", None)
            y  = batch.get("labels", None)
            _ = q_model(pixel_values=x,
                        time=(t.to(device) if t is not None else None),
                        pixel_mask=(pm.to(device) if pm is not None else None),
                        labels=(y.to(device) if y is not None else None))
            steps += 1
    print(f"Re-calibrated on {steps} batches.")

    # after: repq_ln_reparam_generic(q_model) + second calib
    # from torch.utils.data import DataLoader
    # from scOT.problems.base import get_dataset

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

        # for NS-SL only
        # u = uv[:, 0]       # (N,W,H)
        # v = uv[:, 1]

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




    def eval_model_poseidon(model, dataset_name, data_path, num_traj=8, batch_size=16):
        device = next(model.parameters()).device
        model.eval()

        ds = get_dataset(dataset=dataset_name, which="test",
                        num_trajectories=num_traj, data_path=data_path)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

        total_rel_l1, num_batches = 0.0, 0
        preds_all, labels_all = [], []

        with torch.no_grad():
            for batch in dl:
                xb = batch["pixel_values"].to(device)
                yb = batch["labels"].to(device)
                tm = batch["time"].to(device)
                pm = batch["pixel_mask"].to(device)

                out = model(pixel_values=xb, time=tm, pixel_mask=pm, labels=yb)

                preds = out.output   # (N, C, H, W)
                true  = yb           # (N, C, H, W)

                # ---- Relative L1 per batch ----
                l1_num = torch.sum(torch.abs(preds - true))
                l1_den = torch.sum(torch.abs(true)) + 1e-12
                rel_l1_batch = l1_num / l1_den

                total_rel_l1 += rel_l1_batch.item()
                num_batches  += 1

                preds_all.append(preds.cpu())
                labels_all.append(true.cpu())

        dataset_rel_l1 = total_rel_l1 / max(1, num_batches)
        preds_all  = torch.cat(preds_all,  dim=0)
        labels_all = torch.cat(labels_all, dim=0)

        # return mean relative L1 over batches, plus full preds/labels for divergence stats
        return dataset_rel_l1, preds_all, labels_all











   # this is the MSE evaluation


    # def eval_model_poseidon(model, dataset_name, data_path, num_traj=8, batch_size=16):
    #     device = next(model.parameters()).device
    #     model.eval()
    #     ds = get_dataset(dataset=dataset_name, which="test",
    #                     num_trajectories=num_traj, data_path=data_path)
    #     dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    #     total_loss_elems, total_nel = 0.0, 0
    #     preds_all = []
    #     with torch.no_grad():
    #         for batch in dl:
    #             xb = batch["pixel_values"].to(device)
    #             yb = batch["labels"].to(device)
    #             tm = batch["time"].to(device)
    #             pm = batch["pixel_mask"].to(device)
    #             out = model(pixel_values=xb, time=tm, pixel_mask=pm, labels=yb)
    #             loss  = out.loss          # Poseidon’s own per-element MSE
    #             preds = out.output        # predictions
    #             n_elems = preds.numel()
    #             total_loss_elems += loss.item() * n_elems
    #             total_nel        += n_elems
    #             preds_all.append(preds.cpu())
    #     dataset_loss = total_loss_elems / total_nel
    #     preds_all = torch.cat(preds_all, dim=0)
    #     return dataset_loss, preds_all

    # # 1) Evaluate quantized model with Poseidon’s loss + your divergence metric
    # repq_loss, repq_preds = eval_model_poseidon(
    #     q_model, DATASET_NAME, DATA_PATH, NUM_TRAJECTORIES, BATCH_SIZE
    # )
    # print("-----------------------------------------------------------------------------------------------------------------")
    # print(MODEL_PATH)
    # print(f"[RepQ] dataset loss: {repq_loss:.6e}")
    # divergence_stats(repq_preds)   # your function from earlier










class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val=0; self.sum=0; self.cnt=0; self.avg=0
    def update(self, v, n=1): self.val=v; self.sum+=v*n; self.cnt+=n; self.avg=self.sum/max(1,self.cnt)

if __name__ == "__main__":
    main()
