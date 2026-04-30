from __future__ import annotations

import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import torch
import torch.nn as nn

from BRECQ.quant.poseidon_quant_model import PoseidonQuantModel
from BRECQ.quant.quant_layer import QuantModule
from PPQ.poseidon_utils import load_poseidon_model


def vectorized_mse_scale_search(
    w: torch.Tensor,
    n_bits: int,
    channel_wise: bool,
    mse_steps: int,
    p: float = 2.4,
):
    """
    Vectorized version of original BRECQ MSE clipping search.

    Original logic:
        for each channel:
            for i in range(80):
                shrink min/max
                quantize
                choose best reconstruction error

    This does the same search, but all channels are handled together.
    """

    n_levels = 2 ** n_bits
    w = w.detach()

    if not channel_wise:
        x = w.reshape(1, -1)
    else:
        if w.dim() == 4:
            # Conv2d: [out_channels, in_channels, kh, kw]
            x = w.flatten(1)
        elif w.dim() == 2:
            # Linear: [out_features, in_features]
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
    else:
        return best_delta.view(-1, 1), best_zero_point.view(-1, 1)


def get_save_paths(model_path: str, n_bits_w: int, channel_wise: bool, mse_steps: int):
    model_name = Path(model_path).name
    cw_name = "channelwise" if channel_wise else "layerwise"

    save_dir = REPO_ROOT / "brecq_artifacts" / model_name / "weight_scales"
    save_dir.mkdir(parents=True, exist_ok=True)

    stem = f"w{n_bits_w}_{cw_name}_mse{mse_steps}"
    return save_dir / f"{stem}.pt", save_dir / f"{stem}_meta.json"


def get_target_model_paths():
    models_root = REPO_ROOT / "models"

    model_paths = []
    for p in sorted(models_root.iterdir()):
        if not p.is_dir():
            continue
        name = p.name

        if not name.endswith("-L"):
            continue
        if name == "CE-RM-L":
            continue

        model_paths.append(p)

    return model_paths

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--mse_search_steps", default=80, type=int)
    parser.add_argument("--channel_wise", action="store_true")

    args = parser.parse_args()

    model_paths = get_target_model_paths()
    bitwidths = [4, 8]

    print("[INFO] Target models:")
    for p in model_paths:
        print("  ", p)

    for model_path in model_paths:
        for n_bits_w in bitwidths:
            print("\n" + "=" * 80)
            print(f"[INFO] Precomputing scales for model={model_path.name}, w{n_bits_w}")
            print("=" * 80)

            print("Loading Poseidon model...")
            model, device = load_poseidon_model(str(model_path), args.device)

            print("Building quantized Poseidon model...")
            wq_params = {
                "n_bits": n_bits_w,
                "channel_wise": args.channel_wise,
                "scale_method": "mse",
            }
            aq_params = {
                "n_bits": 8,
                "channel_wise": False,
                "scale_method": "max",
                "leaf_param": False,
            }

            qnn = PoseidonQuantModel(
                model=model,
                weight_quant_params=wq_params,
                act_quant_params=aq_params,
            ).to(device)
            qnn.eval()

            print("Precomputing weight scales...")
            state = {}

            quant_modules = [
                (name, m)
                for name, m in qnn.model.named_modules()
                if isinstance(m, QuantModule)
            ]

            print(f"[INFO] Num QuantModule = {len(quant_modules)}")

            with torch.no_grad():
                for idx, (name, m) in enumerate(quant_modules):
                    w = m.weight.detach().to(device)

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
                        "weight_shape": tuple(m.weight.shape),
                    }

                    if (idx + 1) % 20 == 0 or (idx + 1) == len(quant_modules):
                        print(f"[INFO] processed {idx + 1}/{len(quant_modules)}: {name}")

            save_path, meta_path = get_save_paths(
                model_path=str(model_path),
                n_bits_w=n_bits_w,
                channel_wise=args.channel_wise,
                mse_steps=args.mse_search_steps,
            )

            torch.save(state, save_path)

            meta = {
                "model_path": str(model_path),
                "model_name": model_path.name,
                "n_bits_w": n_bits_w,
                "channel_wise": args.channel_wise,
                "scale_method": "mse",
                "mse_search_steps": args.mse_search_steps,
                "num_quant_modules": len(state),
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