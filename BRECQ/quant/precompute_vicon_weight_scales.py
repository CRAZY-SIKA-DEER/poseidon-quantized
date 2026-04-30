# BRECQ/quant/precompute_vicon_weight_scales.py
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
VICON_ROOT = REPO_ROOT / "VICON"
SRC_DIR = VICON_ROOT / "src"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(VICON_ROOT))

import models
from cache_vicon_calib_layer_io import load_vicon_ckpt

from BRECQ.quant.vicon_quant_model import VICONQuantModel
from BRECQ.quant.vicon_quant_block import QuantMultiheadAttention
from BRECQ.quant.quant_layer import QuantModule


def vectorized_mse_scale_search(
    w: torch.Tensor,
    n_bits: int,
    channel_wise: bool,
    mse_steps: int,
    p: float = 2.4,
):
    n_levels = 2 ** n_bits
    w = w.detach()

    if not channel_wise:
        x = w.reshape(1, -1)
    else:
        if w.dim() == 4:
            x = w.flatten(1)
        elif w.dim() == 2:
            x = w
        else:
            x = w.reshape(w.shape[0], -1)

    device = x.device
    dtype = x.dtype

    x_max = x.max(dim=1).values
    x_min = x.min(dim=1).values

    best_score = torch.full((x.shape[0],), float("inf"), device=device, dtype=dtype)
    best_delta = torch.empty_like(x_max)
    best_zero_point = torch.empty_like(x_max)

    for i in range(mse_steps):
        shrink = 1.0 - i * 0.01

        new_max = x_max * shrink
        new_min = x_min * shrink

        delta = (new_max - new_min) / (n_levels - 1)
        delta = torch.clamp(delta, min=1e-8)

        zero_point = (-new_min / delta).round()

        x_int = torch.round(x / delta[:, None])
        x_quant = torch.clamp(x_int + zero_point[:, None], 0, n_levels - 1)
        x_dequant = (x_quant - zero_point[:, None]) * delta[:, None]

        score = (x - x_dequant).abs().pow(p).sum(dim=1)

        better = score < best_score
        best_score[better] = score[better]
        best_delta[better] = delta[better]
        best_zero_point[better] = zero_point[better]

    if not channel_wise:
        return best_delta[0], best_zero_point[0]

    if w.dim() == 4:
        return best_delta.view(-1, 1, 1, 1), best_zero_point.view(-1, 1, 1, 1)

    return best_delta.view(-1, 1), best_zero_point.view(-1, 1)


def collect_vicon_quant_weights(qnn):
    """
    Collect all VICON weight tensors that should have precomputed BRECQ scales.

    Includes:
      - QuantModule weights:
          pre_proj, post_proj, linear1, linear2
      - QuantMultiheadAttention weights:
          self_attn.in_proj_weight
          self_attn.out_proj_weight
    """
    items = []

    for name, m in qnn.model.named_modules():
        if isinstance(m, QuantModule):
            items.append((name, m.weight))

        elif isinstance(m, QuantMultiheadAttention):
            items.append((f"{name}.in_proj_weight", m.in_proj_weight))
            items.append((f"{name}.out_proj_weight", m.out_proj_weight))

    return items



def parse_args():
    parser = argparse.ArgumentParser("Precompute VICON BRECQ weight MSE scales")

    parser.add_argument("--ckpt_path", type=str, default="models/vicon/vicon.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mse_search_steps", type=int, default=80)
    parser.add_argument("--channel_wise", action="store_true")
    parser.add_argument("--bits", nargs="+", type=int, default=[4, 8])
    parser.add_argument(
        "--save_root",
        type=str,
        default="brecq_artifacts/VICON/weight_scales",
    )

    return parser.parse_args()


def build_model(args):
    cfg = OmegaConf.load(REPO_ROOT / "VICON/configs/model/default.yaml")
    model = models.ICON_UNCROPPED(cfg)
    load_vicon_ckpt(model, str(REPO_ROOT / args.ckpt_path))
    model.to(args.device)
    model.eval()
    return model


def get_save_paths(args, n_bits_w: int):
    save_dir = REPO_ROOT / args.save_root
    save_dir.mkdir(parents=True, exist_ok=True)

    cw_name = "channelwise" if args.channel_wise else "layerwise"
    stem = f"w{n_bits_w}_{cw_name}_mse{args.mse_search_steps}"

    return save_dir / f"{stem}.pt", save_dir / f"{stem}_meta.json"


def main():
    args = parse_args()

    print("\n========== VICON WEIGHT SCALE PRECOMPUTE ==========")
    print(f"ckpt_path:        {args.ckpt_path}")
    print(f"device:           {args.device}")
    print(f"bits:             {args.bits}")
    print(f"channel_wise:     {args.channel_wise}")
    print(f"mse_search_steps: {args.mse_search_steps}")
    print(f"save_root:        {args.save_root}")
    print("===================================================\n")

    for n_bits_w in args.bits:
        print("\n" + "=" * 80)
        print(f"[INFO] Precomputing VICON weight scales: w{n_bits_w}")
        print("=" * 80)

        model = build_model(args)

        wq_params = {
            "n_bits": n_bits_w,
            "channel_wise": args.channel_wise,
            "scale_method": "mse",
        }

        qnn = VICONQuantModel(
            model=model,
            weight_quant_params=wq_params,
            act_quant_params={},
        ).to(args.device)
        qnn.eval()

        quant_weights = collect_vicon_quant_weights(qnn)

        print(f"[INFO] Num quantized weight tensors = {len(quant_weights)}")

        state = {}

        with torch.no_grad():
            for idx, (name, weight) in enumerate(quant_weights):
                w = weight.detach().to(args.device)

                delta, zero_point = vectorized_mse_scale_search(
                    w=w,
                    n_bits=n_bits_w,
                    channel_wise=args.channel_wise,
                    mse_steps=args.mse_search_steps,
                )

                state[name] = {
                    "delta": delta.detach().cpu(),
                    "zero_point": zero_point.detach().cpu(),
                    "n_bits": n_bits_w,
                    "channel_wise": args.channel_wise,
                    "scale_method": "mse",
                    "mse_search_steps": args.mse_search_steps,
                    "weight_shape": tuple(weight.shape),
                }

                if (idx + 1) % 10 == 0 or (idx + 1) == len(quant_weights):
                    print(f"[INFO] processed {idx + 1}/{len(quant_weights)}: {name}")

        save_path, meta_path = get_save_paths(args, n_bits_w)

        torch.save(state, save_path)

        meta = {
            "model": "VICON",
            "ckpt_path": args.ckpt_path,
            "n_bits_w": n_bits_w,
            "channel_wise": args.channel_wise,
            "scale_method": "mse",
            "mse_search_steps": args.mse_search_steps,
            "num_quant_weight_tensors": len(state),
            "save_path": str(save_path),
            "real_save_path": str(save_path.resolve()),
        }

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[INFO] Saved scales -> {save_path}")
        print(f"[INFO] Real path     -> {save_path.resolve()}")
        print(f"[INFO] Saved meta   -> {meta_path}")

        del model
        del qnn
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()