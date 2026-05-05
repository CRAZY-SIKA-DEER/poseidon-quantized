# SAPQ/debug_likelihood_grad_by_layer.py
from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn as nn

from PPQ.config import PPQConfig
from PPQ.poseidon_utils import (
    load_poseidon_model,
    build_poseidon_loaders,
    get_clean_network_outputs_poseidon,
)
from PPQ.optimize import (
    freeze_batches,
    initialize_step_sizes,
    get_compatible_linear_layers,
)
from PPQ.metrics import build_channel_param_weights, compute_avg_bits
from SAPQ.sapq_loss import compute_mc_negative_loglikelihood_network_global


REPO = Path("/home/u6ey/yiheng.u6ey/poseidon-quantized")

MODEL_TAG = "NS-SVS-L"
DATASET_TAG = "NS-SVS"
DATASET_NAME = "fluids.incompressible.VortexSheet"

RANGES_PATH = REPO / "precalculated_ranges" / MODEL_TAG / "best" / "ranges.pt"
SENS_PATH = REPO / "SAPQ" / "prior_sensitivity_sobo" / MODEL_TAG / "prior_sensitivity.pt"

SAVE_DIR = REPO / "SAPQ" / "debug_grad_outputs" / MODEL_TAG
SAVE_DIR.mkdir(parents=True, exist_ok=True)

NUM_DEBUG_BATCHES = 20
NUM_MC_SAMPLES = 10
ETA = 1e-6

INIT_BITS = 8
BMAX_BITS = 20
EPS = 1e-12


def load_candidate_layers(model, quant_layer_path: Path):
    obj = torch.load(quant_layer_path, map_location="cpu")
    names = obj["quantize_layers"]
    name2mod = dict(model.named_modules())
    return [n for n in names if isinstance(name2mod.get(n), nn.Linear)]


def get_w_step(entry):
    if isinstance(entry, tuple):
        return entry[0]
    return entry


def zero_step_grads(step_sizes_dict):
    for entry in step_sizes_dict.values():
        s = get_w_step(entry)
        if s.grad is not None:
            s.grad.zero_()


def collect_likelihood_grad_rows(
    step_sizes_dict,
    ranges_dict,
    sens_dict,
    batch_idx: int,
    like_loss_value: float,
):
    rows = []

    for layer_name, entry in step_sizes_dict.items():
        if layer_name not in ranges_dict:
            continue

        s = get_w_step(entry)
        if s.grad is None:
            continue

        g = s.grad.detach().float().cpu()
        step = s.detach().float().cpu()
        r = ranges_dict[layer_name]["weight_ranges"].detach().float().cpu()

        if g.numel() != step.numel() or step.numel() != r.numel():
            continue

        bits = torch.log2(torch.clamp(r, min=EPS) / torch.clamp(step, min=EPS))

        sens = sens_dict.get(layer_name, None)
        if sens is not None:
            sens = sens.detach().float().cpu()
            if sens.numel() == 1:
                sens = sens.expand_as(bits)

            if sens.numel() == bits.numel():
                sens_mean = sens.mean().item()
                sens_max = sens.max().item()
            else:
                sens_mean = float("nan")
                sens_max = float("nan")
        else:
            sens_mean = float("nan")
            sens_max = float("nan")

        # Interpretation under loss minimization:
        # grad > 0  => S decreases => bitwidth increases
        # grad < 0  => S increases => bitwidth decreases
        rows.append({
            "batch_idx": batch_idx,
            "loss": like_loss_value,
            "layer": layer_name,
            "num_channels": int(g.numel()),

            "grad_mean": g.mean().item(),
            "grad_median": g.median().item(),
            "grad_sum": g.sum().item(),
            "grad_abs_mean": g.abs().mean().item(),
            "grad_abs_max": g.abs().max().item(),

            "frac_grad_positive": (g > 0).float().mean().item(),
            "frac_grad_negative": (g < 0).float().mean().item(),

            "bits_mean": bits.mean().item(),
            "bits_min": bits.min().item(),
            "bits_max": bits.max().item(),

            "step_mean": step.mean().item(),
            "range_mean": r.mean().item(),

            "sens_mean": sens_mean,
            "sens_max": sens_max,
        })

    return rows


def main():
    cfg = PPQConfig()

    cfg.model_path = str(REPO / "models" / MODEL_TAG)
    cfg.data_path = str(REPO / "dataset" / DATASET_TAG)
    cfg.dataset_name = DATASET_NAME

    cfg.calib_batchsize = 1
    cfg.calib_steps = NUM_DEBUG_BATCHES
    cfg.init_bits = INIT_BITS
    cfg.bmax_bits = BMAX_BITS

    print("Loading model...")
    model, device = load_poseidon_model(cfg.model_path, cfg.device)
    model = model.to(device).eval()

    print("Building calibration loader...")
    _, _, calib_iter, _ = build_poseidon_loaders(
        dataset_name=cfg.dataset_name,
        data_path=cfg.data_path,
        calib_batchsize=cfg.calib_batchsize,
        calib_steps=cfg.calib_steps,
        val_batchsize=1,
        val_steps=1,
    )

    frozen_batches, _ = freeze_batches(calib_iter())
    frozen_batches = frozen_batches[:NUM_DEBUG_BATCHES]
    print(f"[INFO] debug batches = {len(frozen_batches)}")

    print("Caching clean FP outputs...")
    clean_outputs = get_clean_network_outputs_poseidon(
        model=model,
        frozen_batches=frozen_batches,
        device=device,
    )

    print("Loading ranges...")
    ranges_obj = torch.load(RANGES_PATH, map_location="cpu")
    ranges_dict = ranges_obj["ranges_dict"]

    for rec in ranges_dict.values():
        rec["weight_ranges"] = rec["weight_ranges"].to(device)
        if "activation_ranges" in rec:
            rec["activation_ranges"] = rec["activation_ranges"].to(device)

    print("Loading sensitivity...")
    sens_obj = torch.load(SENS_PATH, map_location="cpu")
    sens_dict = {
        k: v.to(device)
        for k, v in sens_obj["layer_sensitivity_raw"].items()
    }

    print("Loading candidate layers...")
    candidate_layers = load_candidate_layers(model, Path(cfg.quant_layer_path))
    target_layers = get_compatible_linear_layers(model, candidate_layers, ranges_dict)
    print(f"[INFO] target layers = {len(target_layers)}")

    print("Initializing step sizes...")
    step_sizes_dict, _ = initialize_step_sizes(
        ranges_dict=ranges_dict,
        target_layers=target_layers,
        init_bits=INIT_BITS,
        bmax_bits=BMAX_BITS,
        device=device,
        model_path=cfg.model_path,
        percentile_prob=cfg.percentile_prob,
        repo_root=cfg.repo_root,
        weight_only=True,
    )

    channel_weights = build_channel_param_weights(model, target_layers)
    avg_bits = compute_avg_bits(step_sizes_dict, ranges_dict, channel_weights)
    print(f"[INIT] AvgBits = {avg_bits:.6f}")

    all_rows = []

    for batch_idx in range(len(frozen_batches)):
        print(f"\n========== batch {batch_idx} ==========")

        zero_step_grads(step_sizes_dict)

        like_loss = compute_mc_negative_loglikelihood_network_global(
            model=model,
            step_sizes_dict=step_sizes_dict,
            frozen_batches=frozen_batches,
            clean_net_outputs=clean_outputs,
            batch_idx=batch_idx,
            num_mc_samples=NUM_MC_SAMPLES,
            eta=ETA,
            device=device,
        )

        like_loss.backward()

        rows = collect_likelihood_grad_rows(
            step_sizes_dict=step_sizes_dict,
            ranges_dict=ranges_dict,
            sens_dict=sens_dict,
            batch_idx=batch_idx,
            like_loss_value=like_loss.item(),
        )

        all_rows.extend(rows)

        mean_grad = sum(r["grad_mean"] for r in rows) / max(len(rows), 1)
        mean_pos = sum(r["frac_grad_positive"] for r in rows) / max(len(rows), 1)
        mean_neg = sum(r["frac_grad_negative"] for r in rows) / max(len(rows), 1)

        print(
            f"[LIKE] loss={like_loss.item():.6e} | "
            f"mean_grad={mean_grad:.6e} | "
            f"pos_frac={mean_pos:.3f} | "
            f"neg_frac={mean_neg:.3f}"
        )

    csv_path = SAVE_DIR / "likelihood_grad_by_layer.csv"

    if len(all_rows) == 0:
        raise RuntimeError("No likelihood gradient rows collected.")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n========== SAVED ==========")
    print(f"Likelihood gradient CSV: {csv_path}")

    print("\n========== TOP LIKELIHOOD GRADIENT LAYERS, BATCH 0 ==========")
    batch0 = [r for r in all_rows if r["batch_idx"] == 0]
    batch0 = sorted(batch0, key=lambda x: abs(x["grad_mean"]), reverse=True)[:20]

    for r in batch0:
        direction = "bits_up" if r["grad_mean"] > 0 else "bits_down"
        print(
            f"{r['grad_mean']: .3e} | {direction:9s} | "
            f"B={r['bits_mean']:.2f} | "
            f"sens={r['sens_mean']:.3e} | "
            f"{r['layer']}"
        )


if __name__ == "__main__":
    main()