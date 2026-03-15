from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from PPQ.config import PPQConfig
from PPQ.poseidon_utils import load_poseidon_model, build_poseidon_loaders
from PPQ.ranges import compute_data_ranges_poseidon
from PPQ.optimize import freeze_batches


def load_candidate_linear_layers(model, quant_layer_path: Path):
    """
    Load candidate quantization layers from file, then keep only real Linear layers.
    """
    print(f"[INFO] Loading quantize layer list from: {quant_layer_path}")
    layer_data = torch.load(quant_layer_path)

    name2mod = dict(model.named_modules())
    candidate_layers = [
        name for name in layer_data["quantize_layers"]
        if isinstance(name2mod.get(name, None), nn.Linear)
    ]

    print(f"[INFO] {len(candidate_layers)} candidate Linear layers")
    return candidate_layers


def build_init_step_sizes_from_ranges(
    ranges_dict,
    target_layers,
    init_bits: int,
    bmax_bits: int,
    device: str | torch.device = "cuda",
    eps: float = 1e-8,
):
    """
    Build plain tensor initial step sizes from ranges_dict.
    Same logic as initialize_step_sizes(), but without nn.Parameter wrapping.
    """
    device = torch.device(device)
    step_sizes_dict = {}

    for name in target_layers:
        if name not in ranges_dict:
            continue

        w_range = ranges_dict[name]["weight_ranges"].to(device)
        a_range = ranges_dict[name]["activation_ranges"].to(device)

        w_step_init = w_range / (2 ** init_bits)
        a_step_init = a_range / (2 ** init_bits)

        w_step_min = w_range / (2 ** bmax_bits)
        a_step_min = a_range / (2 ** bmax_bits)

        w_step_min = torch.maximum(w_step_min, torch.full_like(w_step_min, eps))
        a_step_min = torch.maximum(a_step_min, torch.full_like(a_step_min, eps))

        w_step_init = torch.clamp(w_step_init, min=w_step_min, max=w_range)
        a_step_init = torch.clamp(a_step_init, min=a_step_min, max=a_range)

        step_sizes_dict[name] = (
            w_step_init.detach().cpu(),
            a_step_init.detach().cpu(),
        )

    return step_sizes_dict


def main():
    cfg = PPQConfig()

    model, device = load_poseidon_model(cfg.model_path, cfg.device)

    _calib_loader, _val_loader, calib_iter, _val_iter = build_poseidon_loaders(
        dataset_name=cfg.dataset_name,
        data_path=cfg.data_path,
        calib_batchsize=cfg.calib_batchsize,
        calib_steps=cfg.calib_steps,
        val_batchsize=cfg.val_batchsize,
        val_steps=cfg.val_steps,
    )

    frozen_batches, frozen_iter = freeze_batches(calib_iter)
    print(f"[INFO] Frozen calibration batches: {len(frozen_batches)}")

    candidate_layers = load_candidate_linear_layers(
        model=model,
        quant_layer_path=cfg.quant_layer_path,
    )

    print(f"[INFO] Computing ranges with percentile_prob={cfg.percentile_prob} ...")
    ranges_dict = compute_data_ranges_poseidon(
        model=model,
        dataloader=frozen_iter,
        device=device,
        layer_names=candidate_layers,
        percentile_prob=cfg.percentile_prob,
    )

    model_name = Path(cfg.model_path).name
    percentile_tag = f"p{cfg.percentile_prob:.0e}"

    save_dir = cfg.repo_root / "initial_step_sizes" / model_name / percentile_tag
    save_dir.mkdir(parents=True, exist_ok=True)

    for bits in [4, 8, 16]:
        print(f"[INFO] Building {bits}-bit initial step sizes ...")

        step_sizes_dict = build_init_step_sizes_from_ranges(
            ranges_dict=ranges_dict,
            target_layers=candidate_layers,
            init_bits=bits,
            bmax_bits=cfg.bmax_bits,
            device=device,
        )

        save_obj = {
            "step_sizes": {
                name: (w_step, a_step)
                for name, (w_step, a_step) in step_sizes_dict.items()
            },
            "meta": {
                "model_path": cfg.model_path,
                "percentile_prob": float(cfg.percentile_prob),
                "init_bits": int(bits),
                "bmax_bits": int(cfg.bmax_bits),
                "layer_count": len(step_sizes_dict),
            },
        }

        pt_path = save_dir / f"{bits}bit.pt"
        json_path = save_dir / f"{bits}bit.json"

        torch.save(save_obj, pt_path)

        with open(json_path, "w") as f:
            json.dump(
                {
                    "step_sizes": {
                        name: {
                            "weight_step": w.tolist(),
                            "act_step": a.tolist(),
                        }
                        for name, (w, a) in step_sizes_dict.items()
                    },
                    "meta": save_obj["meta"],
                },
                f,
                indent=2,
            )

        print(f"[INFO] Saved -> {pt_path}")
        print(f"[INFO] Saved -> {json_path}")


if __name__ == "__main__":
    main()