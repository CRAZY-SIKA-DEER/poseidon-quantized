# SAPQ/debug_batch4_5.py
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
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


DEBUG_NUM_EPOCHS_FOR_LR = 20
MAX_BATCHES = 6
SAVE_BATCHES = {4, 5}
PRIOR_WEIGHT = 1e-10


def zero_grads(step_sizes_dict):
    for _, (w_step, _) in step_sizes_dict.items():
        if w_step.grad is not None:
            w_step.grad.zero_()


def clone_steps(step_sizes_dict):
    return {
        name: w_step.detach().clone()
        for name, (w_step, _) in step_sizes_dict.items()
    }


def collect_grads(step_sizes_dict):
    out = {}
    for name, (w_step, _) in step_sizes_dict.items():
        if w_step.grad is None:
            out[name] = torch.zeros_like(w_step.detach())
        else:
            out[name] = w_step.grad.detach().clone()
    return out


def layer_bits(name, w_step, ranges_dict, eps=1e-8):
    w_range = ranges_dict[name]["weight_ranges"].to(w_step.device)
    return torch.log2((w_range + eps) / (w_step.detach() + eps))


def sign_agreement_delta_grad(delta_s, grad):
    mask = grad != 0
    if mask.sum() == 0:
        return float("nan")
    agree = ((grad < 0) & (delta_s > 0)) | ((grad > 0) & (delta_s < 0))
    return float(agree[mask].float().mean().item())


def bit_distribution(step_sizes_dict, ranges_dict):
    layer_mean_bits = []

    for name, (w_step, _) in step_sizes_dict.items():
        if name not in ranges_dict:
            continue
        bits = layer_bits(name, w_step.detach(), ranges_dict)
        layer_mean_bits.append(float(bits.mean().item()))

    layer_mean_bits_sorted = sorted(layer_mean_bits, reverse=True)

    return {
        "max_layer_bits": layer_mean_bits_sorted[0],
        "top10_mean_layer_bits": sum(layer_mean_bits_sorted[:10]) / min(10, len(layer_mean_bits_sorted)),
        "num_layers_above_12": sum(b > 12.0 for b in layer_mean_bits),
        "num_layers_above_10": sum(b > 10.0 for b in layer_mean_bits),
        "num_layers_above_8": sum(b > 8.0 for b in layer_mean_bits),
        "num_layers_below_4": sum(b < 4.0 for b in layer_mean_bits),
    }


def main():
    cfg = PPQConfig()
    cfg.prior_mode = "block_sens"
    cfg.num_epochs = DEBUG_NUM_EPOCHS_FOR_LR
    cfg.eval_every = None

    model_tag = Path(cfg.model_path).name
    dataset_tag = Path(cfg.data_path).name

    out_dir = (
        Path(cfg.repo_root)
        / "SAPQ"
        / "debug_batch4_5_out"
        / model_tag
        / dataset_tag
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "batch4_5_layer_grad_summary.csv"
    json_path = out_dir / "batch0_to_5_global_summary.json"
    raw_path = out_dir / "batch4_5_raw_tensors.pt"

    print("Loading model...")
    model, device = load_poseidon_model(cfg.model_path, cfg.device)
    model = model.to(device).eval()

    print("Loading candidate layers...")
    candidate_layers = load_candidate_layers(model, Path(cfg.quant_layer_path))

    print("Loading frozen batches...")
    frozen_batches = load_frozen_calibration_batches(cfg, device=device)
    frozen_batches = frozen_batches[:MAX_BATCHES]

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

    sens_dict = {
        name: sens_dict[name]
        for name in target_layers
        if name in sens_dict
    }

    channel_weights = build_channel_param_weights(model, target_layers)

    optimizer = optim.Adam(params, lr=cfg.base_lr)
    lr_epoch = get_lr_for_epoch(
        epoch=1,
        base_lr=cfg.base_lr,
        num_epochs=DEBUG_NUM_EPOCHS_FOR_LR,
    )

    for pg in optimizer.param_groups:
        pg["lr"] = lr_epoch

    print(f"[DEBUG] LR used = {lr_epoch:.6e}")

    global_records = []
    layer_rows = []
    raw_records = {}

    for batch_idx in range(MAX_BATCHES):
        avg_before = compute_avg_bits(step_sizes_dict, ranges_dict, channel_weights)
        dist_before = bit_distribution(step_sizes_dict, ranges_dict)
        s_before = clone_steps(step_sizes_dict)

        optimizer.zero_grad()
        zero_grads(step_sizes_dict)

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

        like_loss.backward(retain_graph=True)
        like_grads = collect_grads(step_sizes_dict)
        zero_grads(step_sizes_dict)

        prior_loss.backward(retain_graph=True)
        prior_grads = collect_grads(step_sizes_dict)
        zero_grads(step_sizes_dict)

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

        avg_after = compute_avg_bits(step_sizes_dict, ranges_dict, channel_weights)
        dist_after = bit_distribution(step_sizes_dict, ranges_dict)

        global_record = {
            "batch_idx": int(batch_idx),
            "avg_bits_before": float(avg_before),
            "avg_bits_after": float(avg_after),
            "delta_avg_bits": float(avg_after - avg_before),
            "like_loss": float(like_loss.item()),
            "prior_loss": float(prior_loss.item()),
            "prior_contribution": float((PRIOR_WEIGHT * prior_loss).item()),
            **{f"before_{k}": v for k, v in dist_before.items()},
            **{f"after_{k}": v for k, v in dist_after.items()},
        }
        global_records.append(global_record)

        print(
            f"[BATCH {batch_idx}] "
            f"AvgBits {avg_before:.4f}->{avg_after:.4f} | "
            f"Like={like_loss.item():.6f} | Prior={prior_loss.item():.6f} | "
            f"above8 {dist_before['num_layers_above_8']}->{dist_after['num_layers_above_8']} | "
            f"above12 {dist_before['num_layers_above_12']}->{dist_after['num_layers_above_12']}"
        )

        if batch_idx not in SAVE_BATCHES:
            continue

        raw_records[batch_idx] = {}

        for name, (w_step_after, _) in step_sizes_dict.items():
            if name not in ranges_dict:
                continue

            sb = s_before[name].to(device)
            sa = w_step_after.detach()

            delta_s = sa - sb

            like_g = like_grads[name].to(device)
            prior_g = prior_grads[name].to(device)
            prior_contrib_g = PRIOR_WEIGHT * prior_g
            total_g = total_grads[name].to(device)

            bits_before = layer_bits(name, sb, ranges_dict)
            bits_after = layer_bits(name, sa, ranges_dict)
            delta_bits = bits_after - bits_before

            row = {
                "batch_idx": int(batch_idx),
                "layer": name,
                "bits_before_mean": float(bits_before.mean().item()),
                "bits_after_mean": float(bits_after.mean().item()),
                "delta_bits_mean": float(delta_bits.mean().item()),

                "delta_s_mean": float(delta_s.mean().item()),
                "frac_delta_s_pos": float((delta_s > 0).float().mean().item()),
                "frac_delta_s_neg": float((delta_s < 0).float().mean().item()),

                "like_grad_mean": float(like_g.mean().item()),
                "like_grad_abs_mean": float(like_g.abs().mean().item()),
                "frac_like_grad_neg": float((like_g < 0).float().mean().item()),
                "frac_like_grad_pos": float((like_g > 0).float().mean().item()),

                "prior_grad_mean": float(prior_g.mean().item()),
                "prior_grad_abs_mean": float(prior_g.abs().mean().item()),
                "prior_contrib_grad_mean": float(prior_contrib_g.mean().item()),
                "prior_contrib_grad_abs_mean": float(prior_contrib_g.abs().mean().item()),
                "frac_prior_grad_neg": float((prior_g < 0).float().mean().item()),
                "frac_prior_grad_pos": float((prior_g > 0).float().mean().item()),

                "total_grad_mean": float(total_g.mean().item()),
                "total_grad_abs_mean": float(total_g.abs().mean().item()),

                "agree_like_with_delta": sign_agreement_delta_grad(delta_s, like_g),
                "agree_prior_with_delta": sign_agreement_delta_grad(delta_s, prior_g),
                "agree_total_with_delta": sign_agreement_delta_grad(delta_s, total_g),
            }

            layer_rows.append(row)

            raw_records[batch_idx][name] = {
                "s_before": sb.detach().cpu(),
                "s_after": sa.detach().cpu(),
                "delta_s": delta_s.detach().cpu(),
                "bits_before": bits_before.detach().cpu(),
                "bits_after": bits_after.detach().cpu(),
                "delta_bits": delta_bits.detach().cpu(),
                "like_grad": like_g.detach().cpu(),
                "prior_grad": prior_g.detach().cpu(),
                "prior_contrib_grad": prior_contrib_g.detach().cpu(),
                "total_grad": total_g.detach().cpu(),
            }

        print("\nTop 10 |delta_bits| layers for batch", batch_idx)
        for r in sorted(
            [x for x in layer_rows if x["batch_idx"] == batch_idx],
            key=lambda x: abs(x["delta_bits_mean"]),
            reverse=True,
        )[:10]:
            print(
                f"{r['layer']} | "
                f"bits {r['bits_before_mean']:.3f}->{r['bits_after_mean']:.3f} "
                f"Δbits={r['delta_bits_mean']:.3f} | "
                f"|like_g|={r['like_grad_abs_mean']:.3e} | "
                f"|prior_eff_g|={r['prior_contrib_grad_abs_mean']:.3e} | "
                f"agree_like={r['agree_like_with_delta']:.2f} | "
                f"agree_prior={r['agree_prior_with_delta']:.2f}"
            )

    with open(json_path, "w") as f:
        json.dump(global_records, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(layer_rows[0].keys()))
        writer.writeheader()
        writer.writerows(layer_rows)

    torch.save(raw_records, raw_path)

    print(f"\nSaved global summary -> {json_path}")
    print(f"Saved layer summary  -> {csv_path}")
    print(f"Saved raw tensors    -> {raw_path}")


if __name__ == "__main__":
    main()