from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
VICON_ROOT = REPO_ROOT / "VICON"
SRC_DIR = VICON_ROOT / "src"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(VICON_ROOT))

import models
from cache_vicon_calib_layer_io import load_vicon_ckpt, vicon_preprocess_bc

from BRECQ.quant.vicon_quant_model import VICONQuantModel
from BRECQ.quant.vicon_block_recon import block_reconstruction
from BRECQ.quant.quant_layer import QuantModule
from BRECQ.quant.vicon_quant_block import QuantMultiheadAttention
from BRECQ.quant.quant_block import BaseQuantBlock


def parse_args():
    parser = argparse.ArgumentParser("Run BRECQ for VICON")

    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default="models/vicon/vicon.pth")

    parser.add_argument("--n_bits_w", type=int, default=4)
    parser.add_argument("--channel_wise", action="store_true")

    parser.add_argument("--calib_batchsize", type=int, default=2)
    parser.add_argument("--calib_steps", type=int, default=512)

    parser.add_argument("--recon_iters", type=int, default=20000)
    parser.add_argument("--recon_batch_size", type=int, default=32)

    parser.add_argument("--opt_mode", type=str, default="mse")
    parser.add_argument("--asym", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--save_root", type=str, default="brecq_artifacts/VICON")
    parser.add_argument("--init_batches", type=int, default=10)

    parser.add_argument(
        "--weight_scales_path",
        type=str,
        default=None,
        help="Optional precomputed MSE weight scale file.",
    )

    return parser.parse_args()


def build_model(args):
    cfg = OmegaConf.load(REPO_ROOT / "VICON/configs/model/default.yaml")
    model = models.ICON_UNCROPPED(cfg)
    load_vicon_ckpt(model, str(REPO_ROOT / args.ckpt_path))
    return model, cfg


def load_calibration(args):
    frozen_path = (
        REPO_ROOT
        / "ppq_artifacts"
        / f"{args.dataset_name}-calib"
        / "frozen_calibration_batches.pt"
    )

    if not frozen_path.exists():
        raise FileNotFoundError(f"Frozen calibration not found: {frozen_path}")

    frozen = torch.load(frozen_path, map_location="cpu")
    frozen = frozen[: args.calib_steps]

    cali_data = [batch["pairs"] for batch in frozen]

    print(f"[INFO] Loaded frozen calibration: {frozen_path}")
    print(f"[INFO] Num frozen batches used: {len(cali_data)}")
    print(f"[INFO] Approx samples: {len(cali_data) * args.calib_batchsize}")

    return cali_data, frozen_path


def build_quant_model(model, args):
    model.to(args.device)

    wq_params = {
        "n_bits": args.n_bits_w,
        "channel_wise": args.channel_wise,
        "scale_method": "max",
    }

    qnn = VICONQuantModel(
        model=model,
        weight_quant_params=wq_params,
        act_quant_params={},
    )

    qnn.to(args.device)
    qnn.eval()
    return qnn


def load_precomputed_weight_scales(qnn, args):
    if args.weight_scales_path is None:
        print("[INFO] No precomputed weight scales used.")
        return

    scale_path = REPO_ROOT / args.weight_scales_path
    if not scale_path.exists():
        raise FileNotFoundError(f"Weight scales file not found: {scale_path}")

    state = torch.load(scale_path, map_location="cpu")
    name2mod = dict(qnn.model.named_modules())

    loaded = 0
    skipped = []

    for name, obj in state.items():

        # ---------- Case 1: QuantModule ----------
        if name in name2mod:
            m = name2mod[name]

            if isinstance(m, QuantModule):
                delta = obj["delta"].to(args.device)
                zero_point = obj["zero_point"].to(args.device)

                m.weight_quantizer.delta = delta
                m.weight_quantizer.zero_point = zero_point
                m.weight_quantizer.n_bits = int(obj.get("n_bits", args.n_bits_w))
                m.weight_quantizer.n_levels = 2 ** m.weight_quantizer.n_bits
                m.weight_quantizer.channel_wise = bool(obj.get("channel_wise", args.channel_wise))
                m.weight_quantizer.scale_method = obj.get("scale_method", "mse")
                m.weight_quantizer.inited = True

                loaded += 1
                continue

        # ---------- Case 2: Attention weights ----------
        if ".in_proj_weight" in name:
            module_name = name.replace(".in_proj_weight", "")
            if module_name in name2mod and isinstance(name2mod[module_name], QuantMultiheadAttention):
                m = name2mod[module_name]

                m.in_proj_weight_quantizer.delta = obj["delta"].to(args.device)
                m.in_proj_weight_quantizer.zero_point = obj["zero_point"].to(args.device)
                m.in_proj_weight_quantizer.inited = True

                loaded += 1
                continue

        if ".out_proj_weight" in name:
            module_name = name.replace(".out_proj_weight", "")
            if module_name in name2mod and isinstance(name2mod[module_name], QuantMultiheadAttention):
                m = name2mod[module_name]

                m.out_proj_weight_quantizer.delta = obj["delta"].to(args.device)
                m.out_proj_weight_quantizer.zero_point = obj["zero_point"].to(args.device)
                m.out_proj_weight_quantizer.inited = True

                loaded += 1
                continue

        skipped.append(name)

    print(f"[INFO] Loaded precomputed weight scales: {scale_path}")
    print(f"[INFO] Loaded scales for QuantModule: {loaded}")
    if skipped:
        print(f"[WARN] Skipped scale entries: {len(skipped)}")


def init_quantizers(qnn, cali_data, model_cfg, args):
    print("========== INIT QUANT ==========")

    qnn.set_quant_state(weight_quant=True, act_quant=False)

    with torch.no_grad():
        for batch in cali_data[: args.init_batches]:
            data = vicon_preprocess_bc(batch, model_cfg)
            data = [x.to(args.device) for x in data]
            _ = qnn(data)

    qnn.set_quant_state(weight_quant=False, act_quant=False)


def get_reconstruction_blocks(qnn):
    return [
        (name, m)
        for name, m in qnn.named_modules()
        if isinstance(m, BaseQuantBlock)
    ]


def save_result(qnn, args, frozen_path):
    save_dir = (
        REPO_ROOT
        / args.save_root
        / args.dataset_name
        / f"w{args.n_bits_w}"
        / f"iters{args.recon_iters}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    state_path = save_dir / "adaround_state.pt"
    meta_path = save_dir / "meta.json"

    torch.save(qnn.state_dict(), state_path)

    meta = {
        "model": "VICON",
        "dataset_name": args.dataset_name,
        "ckpt_path": args.ckpt_path,
        "n_bits_w": args.n_bits_w,
        "channel_wise": args.channel_wise,
        "calib_batchsize": args.calib_batchsize,
        "calib_steps": args.calib_steps,
        "recon_iters": args.recon_iters,
        "recon_batch_size": args.recon_batch_size,
        "opt_mode": args.opt_mode,
        "asym": args.asym,
        "weight_scales_path": args.weight_scales_path,
        "frozen_calibration_path": str(frozen_path),
        "state_path": str(state_path),
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Saved state -> {state_path}")
    print(f"[INFO] Saved meta  -> {meta_path}")


def main():
    args = parse_args()

    print("\n========== VICON BRECQ CONFIG ==========")
    print(f"dataset_name:        {args.dataset_name}")
    print(f"ckpt_path:           {args.ckpt_path}")
    print(f"n_bits_w:            {args.n_bits_w}")
    print(f"channel_wise:        {args.channel_wise}")
    print(f"calib_batchsize:     {args.calib_batchsize}")
    print(f"calib_steps:         {args.calib_steps}")
    print(f"recon_iters:         {args.recon_iters}")
    print(f"recon_batch_size:    {args.recon_batch_size}")
    print(f"opt_mode:            {args.opt_mode}")
    print(f"asym:                {args.asym}")
    print(f"device:              {args.device}")
    print(f"save_root:           {args.save_root}")
    print(f"weight_scales_path:  {args.weight_scales_path}")
    print("========================================\n")

    print("========== BUILD MODEL ==========")
    model, model_cfg = build_model(args)

    print("========== LOAD CALIB ==========")
    cali_data, frozen_path = load_calibration(args)

    print("========== BUILD QUANT MODEL ==========")
    qnn = build_quant_model(model, args)

    print("========== LOAD WEIGHT SCALES ==========")
    load_precomputed_weight_scales(qnn, args)

    init_quantizers(qnn, cali_data, model_cfg, args)

    print("========== COLLECT BLOCKS ==========")
    blocks = get_reconstruction_blocks(qnn)
    print(f"[INFO] Num blocks: {len(blocks)}")

    for idx, (name, block) in enumerate(blocks):
        print(f"\n========== Block {idx + 1}/{len(blocks)}: {name} ==========")

        block_reconstruction(
            model=qnn,
            block=block,
            cali_data=cali_data,
            batch_size=args.recon_batch_size,
            iters=args.recon_iters,
            weight=0.01,
            opt_mode=args.opt_mode,
            asym=args.asym,
            act_quant=False,
        )

    save_result(qnn, args, frozen_path)
    print("\n[INFO] VICON BRECQ finished.")


if __name__ == "__main__":
    main()