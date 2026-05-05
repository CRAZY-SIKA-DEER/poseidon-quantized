# SAPQ/debug_first_epoch_grads.py
from __future__ import annotations

import csv
import json
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from PPQ.config import PPQConfig
from PPQ.poseidon_utils import load_poseidon_model, get_clean_network_outputs_poseidon
from PPQ.metrics import build_channel_param_weights, compute_avg_bits
from PPQ.optimize import (
    get_lr_for_epoch,
    clamp_step_sizes_,
    initialize_step_sizes,
    get_compatible_linear_layers,
)

from SAPQ.run_sapq_network_global import (
    load_candidate_layers,
    maybe_load_or_compute_ranges,
    load_sapq_sensitivity,
    load_frozen_calibration_batches,
)

from SAPQ.sapq_loss import compute_sapq_loss_with_prior_global


# ==========================
# DEBUG SETTINGS
# ==========================
DEBUG_NUM_EPOCHS_FOR_LR = 20   # keep original LR schedule
DEBUG_RUN_BATCHES = 512        # collect summary for first 512 batches
RAW_SAVE_BATCHES = {0, 1, 2, 3}  # save full channel tensors only for these batches

PRIOR_WEIGHT = 1e-10           # must match sapq_loss.py


def clone_steps(step_sizes_dict):
    return {
        name: w_step.detach().clone()
        for name, (w_step, _a_step) in step_sizes_dict.items()
    }


def collect_grads(step_sizes_dict):
    out = {}
    for name, (w_step, _a_step) in step_sizes_dict.items():
        if w_step.grad is None:
            out[name] = torch.zeros_like(w_step.detach())
        else:
            out[name] = w_step.grad.detach().clone()
    return out


def zero_step_grads(step_sizes_dict):
    for _name, (w_step, _a_step) in step_sizes_dict.items():
        if w_step.grad is not None:
            w_step.grad.zero_()


def layer_avg_bits(name, w_step, ranges_dict, eps=1e-8):
    w_range = ranges_dict[name]["weight_ranges"].to(w_step.device)
    bits = torch.log2((w_range + eps) / (w_step.detach() + eps))
    return float(bits.mean().item())


def main():
    cfg = PPQConfig()

    # keep same important settings as original experiment
    cfg.prior_mode = "block_sens"
    cfg.num_epochs = DEBUG_NUM_EPOCHS_FOR_LR
    cfg.eval_every = None

    model_tag = Path(cfg.model_path).name
    dataset_tag = Path(cfg.data_path).name

    out_dir = (
        Path(cfg.repo_root)
        / "SAPQ"
        / "debug_first_epoch_grads_out"
        / model_tag
        / dataset_tag
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "layer_update_summary.csv"
    batch_json = out_dir / "batch_avg_bits_summary.json"
    raw_pt = out_dir / "raw_selected_batches.pt"

    print("Loading model...")
    model, device = load_poseidon_model(cfg.model_path, cfg.device)

    print("Loading candidate layers...")
    candidate_layers = load_candidate_layers(model, Path(cfg.quant_layer_path))

    print("Loading frozen calibration batches...")
    frozen_batches = load_frozen_calibration_batches(cfg, device=device)
    frozen_batches = frozen_batches[:DEBUG_RUN_BATCHES]

    def frozen_iter():
        for b in frozen_batches:
            yield b

    print("Loading ranges...")
    ranges_dict = maybe_load_or_compute_ranges(
        cfg=cfg,
        model=model,
        frozen_iter=frozen_iter,
        candidate_layers=candidate_layers,
        device=device,
    )

    print("Loading sensitivity...")
    sens_dict = load_sapq_sensitivity(cfg, device=device)

    print("Caching clean outputs...")
    model = model.to(device).eval()
    clean_net_outputs = get_clean_network_outputs_poseidon(
        model=model,
        frozen_batches=frozen_batches,
        device=device,
    )

    target_layers = get_compatible_linear_layers(
        model=model,
        candidate_layers=candidate_layers,
        ranges_dict=ranges_dict,
    )
    print(f"Optimizing/debugging {len(target_layers)} layers")

    step_sizes_dict, params = initialize_step_sizes(
        ranges_dict=ranges_dict,
        target_layers=target_layers,
        init_bits=cfg.init_bits,
        bmax_bits=cfg.bmax_bits,
        device=device,
        model_path=cfg.model_path,
        percentile_prob=cfg.percentile_prob,
        repo_root=cfg.repo_root,
        weight_only=cfg.weight_only,
    )

    channel_weights = build_channel_param_weights(model, target_layers)

    sens_dict = {
        name: sens_dict[name]
        for name in target_layers
        if name in sens_dict
    }

    optimizer = optim.Adam(params, lr=cfg.base_lr)

    epoch = 1
    lr_epoch = get_lr_for_epoch(
        epoch=epoch,
        base_lr=cfg.base_lr,
        num_epochs=DEBUG_NUM_EPOCHS_FOR_LR,
    )

    for pg in optimizer.param_groups:
        pg["lr"] = lr_epoch

    print(f"[DEBUG] LR used = {lr_epoch:.6e}")
    print(f"[DEBUG] batches = {len(frozen_batches)}")

    batch_records = []
    raw_records = {}

    fieldnames = [
        "batch_idx",
        "layer",
        "avg_bits_before_global",
        "avg_bits_after_global",
        "avg_bits_before_layer",
        "avg_bits_after_layer",
        "mean_delta_s",
        "frac_delta_s_pos",
        "frac_delta_s_neg",
        "mean_like_grad",
        "frac_like_grad_neg",
        "frac_like_grad_pos",
        "mean_prior_grad",
        "mean_prior_contrib_grad",
        "frac_prior_grad_neg",
        "frac_prior_grad_pos",
        "mean_total_grad",
        "frac_total_grad_neg",
        "frac_total_grad_pos",
        "like_loss",
        "prior_loss",
        "total_loss",
    ]

    with open(summary_csv, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        for batch_idx in range(len(frozen_batches)):
            avg_bits_before = compute_avg_bits(
                step_sizes_dict=step_sizes_dict,
                ranges_dict=ranges_dict,
                channel_weights=channel_weights,
            )
            s_before = clone_steps(step_sizes_dict)

            optimizer.zero_grad()
            zero_step_grads(step_sizes_dict)

            total_loss, like_loss, prior_loss = compute_sapq_loss_with_prior_global(
                model=model,
                step_sizes_dict=step_sizes_dict,
                frozen_batches=frozen_batches,
                clean_net_outputs=clean_net_outputs,
                ranges_dict=ranges_dict,
                sens_dict=sens_dict,
                batch_idx=batch_idx,
                num_mc_samples=cfg.num_mc_samples,
                eta=cfg.eta,
                prior_mode=cfg.prior_mode,
                b_target=float(cfg.target_bits),
                sigma0=float(cfg.sigma0),
                alpha=float(cfg.alpha),
                prior_scale=float(cfg.prior_scale),
                device=device,
            )

            # 1) likelihood gradient
            like_loss.backward(retain_graph=True)
            like_grads = collect_grads(step_sizes_dict)
            zero_step_grads(step_sizes_dict)

            # 2) prior gradient
            prior_loss.backward(retain_graph=True)
            prior_grads = collect_grads(step_sizes_dict)
            zero_step_grads(step_sizes_dict)

            # 3) actual total gradient used by Adam
            total_loss.backward()
            total_grads = collect_grads(step_sizes_dict)

            optimizer.step()

            clamp_step_sizes_(
                step_sizes_dict=step_sizes_dict,
                ranges_dict=ranges_dict,
                bmax_bits=cfg.bmax_bits,
                device=device,
                weight_only=cfg.weight_only,
            )

            avg_bits_after = compute_avg_bits(
                step_sizes_dict=step_sizes_dict,
                ranges_dict=ranges_dict,
                channel_weights=channel_weights,
            )

            batch_records.append(
                {
                    "batch_idx": int(batch_idx),
                    "avg_bits_before": float(avg_bits_before),
                    "avg_bits_after": float(avg_bits_after),
                    "like_loss": float(like_loss.item()),
                    "prior_loss": float(prior_loss.item()),
                    "total_loss": float(total_loss.item()),
                    "lr": float(lr_epoch),
                }
            )

            if batch_idx in RAW_SAVE_BATCHES:
                raw_records[batch_idx] = {}

            for name, (w_step_after, _a_step) in step_sizes_dict.items():
                if name not in ranges_dict:
                    continue

                sb = s_before[name].to(w_step_after.device)
                sa = w_step_after.detach()
                delta_s = sa - sb

                lg = like_grads[name].to(sa.device)
                pg = prior_grads[name].to(sa.device)
                tg = total_grads[name].to(sa.device)

                row = {
                    "batch_idx": int(batch_idx),
                    "layer": name,
                    "avg_bits_before_global": float(avg_bits_before),
                    "avg_bits_after_global": float(avg_bits_after),
                    "avg_bits_before_layer": layer_avg_bits(name, sb, ranges_dict),
                    "avg_bits_after_layer": layer_avg_bits(name, sa, ranges_dict),
                    "mean_delta_s": float(delta_s.mean().item()),
                    "frac_delta_s_pos": float((delta_s > 0).float().mean().item()),
                    "frac_delta_s_neg": float((delta_s < 0).float().mean().item()),
                    "mean_like_grad": float(lg.mean().item()),
                    "frac_like_grad_neg": float((lg < 0).float().mean().item()),
                    "frac_like_grad_pos": float((lg > 0).float().mean().item()),
                    "mean_prior_grad": float(pg.mean().item()),
                    "mean_prior_contrib_grad": float((PRIOR_WEIGHT * pg).mean().item()),
                    "frac_prior_grad_neg": float((pg < 0).float().mean().item()),
                    "frac_prior_grad_pos": float((pg > 0).float().mean().item()),
                    "mean_total_grad": float(tg.mean().item()),
                    "frac_total_grad_neg": float((tg < 0).float().mean().item()),
                    "frac_total_grad_pos": float((tg > 0).float().mean().item()),
                    "like_loss": float(like_loss.item()),
                    "prior_loss": float(prior_loss.item()),
                    "total_loss": float(total_loss.item()),
                }
                writer.writerow(row)

                if batch_idx in RAW_SAVE_BATCHES:
                    raw_records[batch_idx][name] = {
                        "s_before": sb.detach().cpu(),
                        "s_after": sa.detach().cpu(),
                        "delta_s": delta_s.detach().cpu(),
                        "like_grad": lg.detach().cpu(),
                        "prior_grad": pg.detach().cpu(),
                        "prior_contrib_grad": (PRIOR_WEIGHT * pg).detach().cpu(),
                        "total_grad": tg.detach().cpu(),
                    }

            if batch_idx % 1 == 0:
                print(
                    f"[BATCH {batch_idx:04d}] "
                    f"AvgBits {avg_bits_before:.4f} -> {avg_bits_after:.4f} | "
                    f"Like={like_loss.item():.6f} | Prior={prior_loss.item():.6f}"
                )

    with open(batch_json, "w") as f:
        json.dump(batch_records, f, indent=2)

    torch.save(raw_records, raw_pt)

    print(f"\nSaved layer summary CSV -> {summary_csv}")
    print(f"Saved batch avg JSON    -> {batch_json}")
    print(f"Saved raw tensors       -> {raw_pt}")


if __name__ == "__main__":
    main()