from __future__ import annotations

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn

from BRECQ.quant.poseidon_quant_model import PoseidonQuantModel
from BRECQ.quant.poseidon_quant_block import QuantScOTLayer, QuantConvNeXtBlock, QuantResNetBlock
from BRECQ.quant.poseidon_block_recon import poseidon_block_reconstruction

from scOT.metrics import relative_lp_error, lp_error
from PPQ.poseidon_utils import load_poseidon_model, build_poseidon_loaders
from BRECQ.quant.quant_layer import QuantModule
from BRECQ.quant.quant_block import BaseQuantBlock
import time


def get_weight_scale_path(args):
    cw_name = "channelwise" if args.channel_wise else "layerwise"
    model_name = Path(args.model_path).name

    scale_path = (
        REPO_ROOT
        / "brecq_artifacts"
        / model_name
        / "weight_scales"
        / f"w{args.n_bits_w}_{cw_name}_mse80.pt"
    )
    return scale_path


def get_recon_save_dir(args):
    model_name = Path(args.model_path).name

    save_dir = (
        REPO_ROOT
        / "brecq_artifacts"
        / model_name
        / "recon"
        / f"w{args.n_bits_w}"
        / f"iters{args.iters_w}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir

def save_brecq_recon_state(qnn, save_dir: Path, args, metrics=None):
    state = {}

    for name, m in qnn.model.named_modules():
        if not isinstance(m, QuantModule):
            continue

        q = m.weight_quantizer

        item = {
            "n_bits": q.n_bits,
            "delta": q.delta.detach().cpu(),
            "zero_point": q.zero_point.detach().cpu(),
        }

        if hasattr(q, "alpha") and q.alpha is not None:
            item["alpha"] = q.alpha.detach().cpu()
            item["soft_targets"] = bool(q.soft_targets)

        state[name] = item

    save_path = save_dir / "adaround_state.pt"
    torch.save(state, save_path)

    meta = {
        "model_path": args.model_path,
        "model_name": Path(args.model_path).name,
        "dataset_name": args.dataset_name,
        "data_path": args.data_path,
        "n_bits_w": args.n_bits_w,
        "channel_wise": args.channel_wise,
        "iters_w": args.iters_w,
        "opt_mode": args.opt_mode,
        "calib_batchsize": args.calib_batchsize,
        "calib_steps": args.calib_steps,
        "num_quant_modules": len(state),
        "metrics": metrics,
        "save_path": str(save_path),
        "real_save_path": str(save_path.resolve()),
    }

    meta_path = save_dir / "meta.json"
    import json
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Saved BRECQ AdaRound state -> {save_path}")
    print(f"[INFO] Saved meta -> {meta_path}")

def seed_all(seed: int = 1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def validate_poseidon(model, val_loader, device):
    model.eval()
    loader = val_loader() if callable(val_loader) else val_loader

    rel_l1_list = []
    abs_l1_list = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        x = batch["pixel_values"]
        t = batch.get("time", None)
        pm = batch.get("pixel_mask", None)
        y = batch.get("labels", None)

        if y is None:
            continue

        outputs = model(
            pixel_values=x,
            time=t,
            pixel_mask=pm,
            labels=y,
            return_dict=True,
        )
        pred = outputs.output

        pred_np = pred.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        batch_rel = relative_lp_error(pred_np, y_np, p=1, return_percent=True)
        batch_abs = lp_error(pred_np, y_np, p=1)

        rel_l1_list.append(float(np.mean(batch_rel)))
        abs_l1_list.append(float(np.mean(batch_abs)))

    if len(rel_l1_list) == 0:
        print("L1: nan")
        print("RelL1: nan")
        return {"l1": float("nan"), "rel_l1": float("nan")}

    mean_l1 = float(sum(abs_l1_list) / len(abs_l1_list))
    mean_rel_l1 = float(sum(rel_l1_list) / len(rel_l1_list))

    print(f"L1: {mean_l1:.6e}")
    print(f"RelL1: {mean_rel_l1:.6e}")
    return {"l1": mean_l1, "rel_l1": mean_rel_l1}


def recon_model(qnn: nn.Module, cali_data, args):
    for name, module in qnn.model.named_modules():
        if hasattr(module, "ignore_reconstruction") and module.ignore_reconstruction:
            print(f"Ignore reconstruction of block {name}")
            continue

        if isinstance(module, (QuantScOTLayer, QuantConvNeXtBlock, QuantResNetBlock)):
            print(f"Reconstruction for block {name} ({module.__class__.__name__})")
            poseidon_block_reconstruction(
                model=qnn,
                block=module,
                cali_data=cali_data,
                batch_size=args.batch_size_recon,
                iters=args.iters_w,
                weight=args.weight,
                opt_mode=args.opt_mode,
                asym=True,
                include_act_func=True,
                b_range=(args.b_start, args.b_end),
                warmup=args.warmup,
                act_quant=False,
                lr=args.lr,
                p=args.p,
            )


def load_precomputed_weight_scales(qnn, scale_path: str, device):
    state = torch.load(scale_path, map_location="cpu")

    loaded = 0
    missing = 0

    for name, m in qnn.model.named_modules():
        if not isinstance(m, QuantModule):
            continue

        if name not in state:
            print(f"[WARN] missing scale for {name}")
            missing += 1
            continue

        q = m.weight_quantizer
        q.delta = state[name]["delta"].to(device)
        q.zero_point = state[name]["zero_point"].to(device)
        q.inited = True
        loaded += 1

    print(f"[INFO] Loaded precomputed scales: {loaded}, missing: {missing}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=1005, type=int)
    parser.add_argument("--device", default="cuda", type=str)

    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--dataset_name", required=True, type=str)
    parser.add_argument("--data_path", required=True, type=str)

    parser.add_argument("--calib_batchsize", default=2, type=int)
    parser.add_argument("--calib_steps", default=2, type=int)
    parser.add_argument("--val_batchsize", default=2, type=int)
    parser.add_argument("--val_steps", default=2, type=int)

    parser.add_argument("--num_samples", default=-1, type=int)

    parser.add_argument("--n_bits_w", default=4, type=int)
    parser.add_argument("--channel_wise", action="store_true")

    parser.add_argument("--iters_w", default=10, type=int)
    parser.add_argument("--weight", default=0.01, type=float)
    parser.add_argument("--b_start", default=20, type=int)
    parser.add_argument("--b_end", default=2, type=int)
    parser.add_argument("--warmup", default=0.2, type=float)
    parser.add_argument("--lr", default=4e-5, type=float)
    parser.add_argument("--p", default=2.0, type=float)

    parser.add_argument("--batch_size_recon", default=2, type=int)
    parser.add_argument("--test_before_calibration", action="store_true")

    parser.add_argument(
    "--opt_mode",
    default="mse",
    choices=["mse", "fisher_diag", "fisher_full"],
    type=str,
    )

    args = parser.parse_args()

    seed_all(args.seed)

    print("Loading Poseidon model...")
    model, device = load_poseidon_model(args.model_path, args.device)

    print("Building Poseidon loaders...")
    _calib_loader, _val_loader, calib_iter, val_iter = build_poseidon_loaders(
        dataset_name=args.dataset_name,
        data_path=args.data_path,
        calib_batchsize=args.calib_batchsize,
        calib_steps=args.calib_steps,
        val_batchsize=args.val_batchsize,
        val_steps=args.val_steps,
    )


    cali_data = list(calib_iter())
    print("[DEBUG] num calibration batches =", len(cali_data))
    print("[DEBUG] batch sizes =", [b["pixel_values"].shape[0] for b in cali_data])
    print("[DEBUG] total calibration samples =", sum(b["pixel_values"].shape[0] for b in cali_data))
    if args.num_samples > 0:
        kept_batches = []
        total = 0
        for batch in cali_data:
            kept_batches.append(batch)
            total += int(batch["pixel_values"].shape[0])
            if total >= args.num_samples:
                break
        cali_data = kept_batches

    if len(cali_data) == 0:
        raise RuntimeError("Calibration data is empty.")

    print("Building quantized Poseidon model...")
    wq_params = {
        "n_bits": args.n_bits_w,
        "channel_wise": args.channel_wise,
        "scale_method": "max",
    }
    aq_params = {
        "n_bits": 8,
        "channel_wise": False,
        "scale_method": "mse",
        "leaf_param": False,
    }

    qnn = PoseidonQuantModel(
        model=model,
        weight_quant_params=wq_params,
        act_quant_params=aq_params,
    )
    qnn.to(device)
    qnn.eval()

    print("Num QuantModule:", sum(1 for m in qnn.modules() if isinstance(m, QuantModule)))
    print("Num BaseQuantBlock:", sum(1 for m in qnn.modules() if isinstance(m, BaseQuantBlock)))
    print("Num QuantScOTLayer:", sum(1 for m in qnn.modules() if isinstance(m, QuantScOTLayer)))
    print("Num QuantConvNeXtBlock:", sum(1 for m in qnn.modules() if isinstance(m, QuantConvNeXtBlock)))
    print("Num QuantResNetBlock:", sum(1 for m in qnn.modules() if isinstance(m, QuantResNetBlock)))

    print("Loading precomputed weight scales...")
    scale_path = get_weight_scale_path(args)
    print("[INFO] scale_path =", scale_path)
    print("[INFO] real scale_path =", scale_path.resolve())

    if not scale_path.exists():
        raise FileNotFoundError(f"Precomputed scale file not found: {scale_path}")

    load_precomputed_weight_scales(qnn, str(scale_path), device)

    print("Setting quantizer states...")
    qnn.set_quant_state(True, False)

    if args.test_before_calibration:
        print("Before reconstruction:")
        validate_poseidon(qnn, val_iter, device)

    print("Start block reconstruction...")
    recon_model(qnn, cali_data, args)

    qnn.set_quant_state(True, False)
    print("After reconstruction:")
    metrics = validate_poseidon(qnn, val_iter, device)

    save_dir = get_recon_save_dir(args)
    print("[INFO] recon save_dir =", save_dir)
    print("[INFO] real recon save_dir =", save_dir.resolve())
    save_brecq_recon_state(qnn, save_dir, args, metrics=metrics)


if __name__ == "__main__":
    main()