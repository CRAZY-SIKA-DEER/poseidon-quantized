#!/usr/bin/env python3
"""
eval_ppq_and_dynamic.py

Evaluate:
  - PPQ step sizes from a chosen run (gamma-style, saved as ppq_step_sizes-*.json)
  - Dynamic-k-bit step sizes (saved in dynamic_stats/NS-PwC-T-dynamic-stepsizes-*.json)

Uses weight-only fake quantization via evaluate_with_stepsizes().
but no deivergnce term is involved
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import json
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from scOT.model import ScOT
from scOT.problems.base import get_dataset
from scOT.metrics import relative_lp_error, lp_error


# =============================================================================
# CONFIG
# =============================================================================

# Paths
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))         # e.g. main/PPQ
PROJECT_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))    # e.g. main
INSPECT_DIR   = os.path.join(PROJECT_ROOT, "inspect_layers")
DYNAMIC_DIR   = os.path.join(PROJECT_ROOT, "dynamic_stats")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "ppq_artifacts", "NS-PwC-T")

MODEL_PATH    = os.path.join(PROJECT_ROOT, "models", "NS-PwC-T")
DATA_PATH     = os.path.join(PROJECT_ROOT, "dataset", "NS-PwC")
DATASET_NAME  = "fluids.incompressible.PiecewiseConstants"

DEVICE_STR    = "cuda"      # or "cpu"

# ---- choose which PPQ run & which dynamic bitwidths to compare ----
#PPQ_TAG      = "1-7"        # -> loads ppq_step_sizes-1-7.json; change as you like.  1-7 avergae 11.58 bitwidth
#PPQ_TAG      = "1-6"         # 1-6 average bitwidth is 10.26 bitwidth
PPQ_TAG      = "1-5"           # 1-5 average bitwidth is 8.49
#PPQ_TAG      = "1-9"           # 1-9 avergae bitwidht is 13.24
DYNAMIC_BITS = [7, 8, 9]       # dynamic 8-bit and 9-bit; change to [4,5,...,16] etc.


# =============================================================================
# BASIC HELPERS (model + data)
# =============================================================================

def load_poseidon_model(model_path: str, device: str = "cuda"):
    """
    Load the Poseidon ScOT model for quantization/evaluation.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = ScOT.from_pretrained(model_path).to(device)
    model.eval()
    torch.set_float32_matmul_precision("high")
    print(f"[INFO] Model loaded from '{model_path}' on device: {device}")
    return model, device


def build_poseidon_loaders(
    dataset_name: str,
    data_path: str,
    calib_batchsize: int = 8,
    calib_steps: int = 8,
    val_batchsize: int = 16,
    val_steps: int = 50,
):
    """
    Exactly the same pattern you used before: returns
      calib_loader, val_loader, calib_iter, val_iter
    """
    train_ds = get_dataset(
        dataset_name, which="train",
        num_trajectories=2048, data_path=data_path
    )
    try:
        val_ds = get_dataset(
            dataset_name, which="val",
            num_trajectories=256, data_path=data_path
        )
    except Exception:
        val_ds = get_dataset(
            dataset_name, which="test",
            num_trajectories=256, data_path=data_path
        )

    calib_loader = DataLoader(
        train_ds, batch_size=calib_batchsize, shuffle=True,
        num_workers=min(os.cpu_count() or 0, 16), pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=val_batchsize, shuffle=False,
        num_workers=min(os.cpu_count() or 0, 16), pin_memory=True
    )

    def take(loader, steps):
        for i, b in enumerate(loader):
            if i >= steps:
                break
            yield b

    calib_iter = lambda: take(calib_loader, calib_steps)
    val_iter   = lambda: take(val_loader,   val_steps)
    return calib_loader, val_loader, calib_iter, val_iter


def load_quantize_layer_names(path: str) -> List[str]:
    """
    Load layer names from T_quantize_layers.pt.
    Expected format: {"quantize_layers": [name1, name2, ...]}
    """
    print(f"[INFO] Loading quantize layer list from: {path}")
    data = torch.load(path, map_location="cpu")
    return data["quantize_layers"]


# =============================================================================
# EVALUATION FUNCTION (your version, unchanged)
# =============================================================================

def evaluate_with_stepsizes(
    model: nn.Module,
    val_loader,
    weight_steps,
    act_steps,              # kept for API compatibility; not used (we only quantize weights)
    layer_names,
    device: str = "cuda",
):
    """
    Evaluate the model with *weight-only* fake quantization using given step sizes.

    - model:       Poseidon / ScOT model
    - val_loader:  DataLoader OR callable that returns an iterator over validation batches
                   (same pattern as calib_iter / val_iter in your code)
    - weight_steps:
        PPQ case:   { layer_name: (w_step_param, a_step_tensor) }
        Dynamic case: { layer_name: 1D tensor OR list of step sizes [out_features] }
    - act_steps:   ignored (placeholder for future activation quantization)
    - layer_names: list of layer names to quantize
    - device:      "cuda" or "cpu"

    Returns:
        metrics: {
            "l1":     average absolute L1 error (using lp_error, p=1),
            "rel_l1": average relative L1 error (using relative_lp_error, p=1),
        }
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Map layer name -> module
    name2mod = dict(model.named_modules())

    # Prepare per-layer weight step tensors on device
    w_steps_per_layer = {}
    for name in layer_names:
        if name not in name2mod:
            continue
        if not isinstance(name2mod[name], nn.Linear):
            continue
        if name not in weight_steps:
            continue

        w_info = weight_steps[name]
        # PPQ case: (w_step_param, a_step)
        if isinstance(w_info, (tuple, list)) and len(w_info) >= 1:
            w_step = w_info[0]
        else:
            # Dynamic case: directly a tensor or list
            w_step = w_info

        if isinstance(w_step, torch.nn.Parameter):
            w_step = w_step.detach()

        if not isinstance(w_step, torch.Tensor):
            w_step = torch.tensor(w_step)

        w_steps_per_layer[name] = w_step.to(device)

    # Build quantization hooks (weights only)
    def make_weight_quant_hook(layer_name, w_step_tensor):
        def hook(mod, inp, out):
            x = inp[0]  # input activation (we do NOT quantize it here)

            # mod.weight: [out_features, in_features]
            w = mod.weight
            w_flat = w.view(w.size(0), -1)  # [out_features, *]
            step = w_step_tensor.view(-1, 1)  # [out_features, 1]

            w_quant = torch.round(w_flat / step) * step
            w_quant = w_quant.view_as(w)

            y = torch.nn.functional.linear(x, w_quant, mod.bias)
            return y
        return hook

    # Register hooks
    handles = []
    for name, mod in name2mod.items():
        if name in w_steps_per_layer and isinstance(mod, nn.Linear):
            h = mod.register_forward_hook(
                make_weight_quant_hook(name, w_steps_per_layer[name])
            )
            handles.append(h)

    # Prepare val iterator (callable or DataLoader)
    if callable(val_loader):
        loader = val_loader()
    else:
        loader = val_loader

    # Accumulate metrics over val set
    rel_l1_list = []
    abs_l1_list = []

    with torch.no_grad():
        for batch in loader:
            x  = batch["pixel_values"].to(device)
            t  = batch.get("time", None)
            pm = batch.get("pixel_mask", None)
            y  = batch.get("labels", None)

            if t is not None:
                t = t.to(device)
            if pm is not None:
                pm = pm.to(device)
            if y is not None:
                y = y.to(device)
            else:
                # for evaluation we expect labels
                continue

            outputs = model(
                pixel_values=x,
                time=t,
                pixel_mask=pm,
                labels=y,
            )
            pred = outputs.output

            # Poseidon's own metrics
            # p=1 → L1 and relative L1
            pred_np = pred.detach().cpu().numpy()
            y_np    = y.detach().cpu().numpy()

            batch_rel = relative_lp_error(pred_np, y_np, p=1, return_percent=True)
            batch_abs = lp_error(pred_np, y_np, p=1)

            rel_l1 = float(np.mean(batch_rel))
            abs_l1 = float(np.mean(batch_abs))

            rel_l1_list.append(rel_l1)
            abs_l1_list.append(abs_l1)

    # Remove hooks
    for h in handles:
        h.remove()

    # Average over all validation batches
    if len(rel_l1_list) == 0:
        metrics = {"l1": float("nan"), "rel_l1": float("nan")}
    else:
        metrics = {
            "l1":     float(sum(abs_l1_list) / len(abs_l1_list)),
            "rel_l1": float(sum(rel_l1_list) / len(rel_l1_list)),
        }

    print(f"[EVAL] L1={metrics['l1']:.6e} | RelL1={metrics['rel_l1']:.6e}")
    return metrics


# =============================================================================
# LOADERS FOR STEP-SIZE FILES
# =============================================================================

def load_ppq_step_sizes(json_path: str, device: torch.device) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Load PPQ step sizes from a JSON file:

      {
        "step_sizes": {
          layer_name: [w_list, a_list],
          ...
        },
        "meta": {...}
      }

    Returns:
        dict {layer_name: (w_step_tensor, a_step_tensor)}
    """
    print(f"[INFO] Loading PPQ step sizes from: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    step_sizes = {}
    for name, (w_list, a_list) in data["step_sizes"].items():
        w = torch.tensor(w_list, dtype=torch.float32, device=device)
        a = torch.tensor(a_list, dtype=torch.float32, device=device)
        step_sizes[name] = (w, a)

    print(f"[INFO] Loaded PPQ steps for {len(step_sizes)} layers.")
    return step_sizes


def load_dynamic_step_sizes(json_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Load dynamic per-channel weight step sizes from:

      {
        "num_bits": B,
        "step_sizes": {
          layer_name: [S_k, ...],
          ...
        }
      }

    Returns:
        dict {layer_name: w_step_tensor}
    """
    #print(f"[INFO] Loading dynamic step sizes from: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    steps = {
        name: torch.tensor(s, dtype=torch.float32, device=device)
        for name, s in data["step_sizes"].items()
    }
    #print(f"[INFO] Loaded dynamic steps for {len(steps)} layers (num_bits={data['num_bits']}).")
    return steps


# =============================================================================
# MAIN
# =============================================================================

def main():
    # 1) device, model, val loader
    device = torch.device(DEVICE_STR if torch.cuda.is_available() else "cpu")
    model, device = load_poseidon_model(MODEL_PATH, device=device)

    _, _, _, val_iter = build_poseidon_loaders(
        dataset_name=DATASET_NAME,
        data_path=DATA_PATH,
        calib_batchsize=2,
        calib_steps=64,
        val_batchsize=4,
        val_steps=40,
    )

    # 2) layer list (Linear only)
    layer_list_path = os.path.join(INSPECT_DIR, "T_quantize_layers.pt")
    quantize_layers = load_quantize_layer_names(layer_list_path)

    name2mod = dict(model.named_modules())
    cand_layers = [
        n for n in quantize_layers
        if isinstance(name2mod.get(n, None), nn.Linear)
    ]
    print(f"[INFO] {len(cand_layers)} candidate Linear layers for quantization.")

    # 3) Evaluate PPQ (gamma run) if desired
    if PPQ_TAG is not None:
        ppq_json = os.path.join(ARTIFACTS_DIR, f"ppq_step_sizes-{PPQ_TAG}.json")
        ppq_steps = load_ppq_step_sizes(ppq_json, device=device)


        print("\n================ PPQ EVALUATION ================")
        ppq_metrics = evaluate_with_stepsizes(
            model=model,
            val_loader=val_iter,       # callable
            weight_steps=ppq_steps,
            act_steps=None,
            layer_names=cand_layers,
            device=device,
        )
        print(f"[PPQ-{PPQ_TAG}] L1={ppq_metrics['l1']:.6e} | RelL1={ppq_metrics['rel_l1']:.6e}")

    # 4) Evaluate each dynamic bitwidth
    for bits in DYNAMIC_BITS:
        dyn_json = os.path.join(DYNAMIC_DIR, f"NS-PwC-T-dynamic-stepsizes-{bits}.json")
        if not os.path.exists(dyn_json):
            print(f"[WARN] Dynamic step-size file not found for {bits} bits: {dyn_json}")
            continue

        dyn_steps = load_dynamic_step_sizes(dyn_json, device=device)

        print(f"\n================ DYNAMIC {bits}-BIT EVALUATION ================")
        dyn_metrics = evaluate_with_stepsizes(
            model=model,
            val_loader=val_iter,
            weight_steps=dyn_steps,
            act_steps=None,
            layer_names=cand_layers,
            device=device,
        )
        print(f"[Dyn-{bits}] L1={dyn_metrics['l1']:.6e} | RelL1={dyn_metrics['rel_l1']:.6e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
