# SAPQ/debug_batch0_1.py
from __future__ import annotations

import csv
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


def collect_adam_state(optimizer, step_sizes_dict):
    out = {}

    for name, (w_step, _) in step_sizes_dict.items():
        state = optimizer.state.get(w_step, None)

        if state is None or "exp_avg" not in state or "exp_avg_sq" not in state:
            out[name] = {
                "exp_avg": torch.zeros_like(w_step.detach()),
                "exp_avg_sq": torch.zeros_like(w_step.detach()),
            }
        else:
            out[name] = {
                "exp_avg": state["exp_avg"].detach().clone(),
                "exp_avg_sq": state["exp_avg_sq"].detach().clone(),
            }

    return out


def layer_bits(name, w_step, ranges_dict, eps=1e-8):
    w_range = ranges_dict[name]["weight_ranges"].to(w_step.device)
    bits = torch.log2((w_range + eps) / (w_step.detach() + eps))
    return bits


def summarize_tensor(x):
    x = x.detach().float().cpu()
    return {
        "mean": float(x.mean().item()),
        "median": float(x.median().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "frac_pos": float((x > 0).float().mean().item()),
        "frac_neg": float((x < 0).float().mean().item()),
        "frac_zero": float((x == 0).float().mean().item()),
    }


def sign_agreement_delta_grad(delta_s, grad):
    """
    Gradient descent:
        grad < 0 should give delta_s > 0
        grad > 0 should give delta_s < 0
    """
    delta_s = delta_s.detach()
    grad = grad.detach()

    mask = grad != 0
    if mask.sum() == 0:
        return float("nan")

    agree = ((grad < 0) & (delta_s > 0)) | ((grad > 0) & (delta_s < 0))
    return float(agree[mask].float().mean().item())


def print_global_batch_summary(batch_idx, avg_before, avg_after, like_loss, prior_loss):
    print("\n" + "=" * 80)
    print(f"BATCH {batch_idx}")
    print("=" * 80)
    print(f"AvgBits: {avg_before:.6f} -> {avg_after:.6f}")
    print(f"Like loss:  {like_loss.item():.6f}")
    print(f"Prior loss: {prior_loss.item():.6f}")
    print(f"Prior contribution to total: {(PRIOR_WEIGHT * prior_loss).item():.12e}")


def run_one_batch(
    *,
    batch_idx,
    model,
    step_sizes_dict,
    frozen_batches,
    clean_net_outputs,
    ranges_dict,
    sens_dict,
    optimizer,
    cfg,
    device,
    channel_weights,
    lr_epoch,
):
    avg_before = compute_avg_bits(
        step_sizes_dict=step_sizes_dict,
        ranges_dict=ranges_dict,
        channel_weights=channel_weights,
    )

    s_before = clone_steps(step_sizes_dict)
    adam_before = collect_adam_state(optimizer, step_sizes_dict)

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

    avg_after = compute_avg_bits(
        step_sizes_dict=step_sizes_dict,
        ranges_dict=ranges_dict,
        channel_weights=channel_weights,
    )

    s_after = clone_steps(step_sizes_dict)
    adam_after = collect_adam_state(optimizer, step_sizes_dict)

    print_global_batch_summary(batch_idx, avg_before, avg_after, like_loss, prior_loss)

    rows = []

    for name, (w_step, _) in step_sizes_dict.items():
        if name not in ranges_dict:
            continue

        sb = s_before[name].to(device)
        sa = s_after[name].to(device)

        delta_s = sa - sb
        like_g = like_grads[name].to(device)
        prior_g = prior_grads[name].to(device)
        prior_contrib_g = PRIOR_WEIGHT * prior_g
        total_g = total_grads[name].to(device)

        bits_before = layer_bits(name, sb, ranges_dict)
        bits_after = layer_bits(name, sa, ranges_dict)
        delta_bits = bits_after - bits_before

        adam_exp_avg_before = adam_before[name]["exp_avg"].to(device)
        adam_exp_avg_after = adam_after[name]["exp_avg"].to(device)

        row = {
            "batch_idx": batch_idx,
            "layer": name,

            "bits_before_mean": float(bits_before.mean().item()),
            "bits_after_mean": float(bits_after.mean().item()),
            "delta_bits_mean": float(delta_bits.mean().item()),

            "delta_s_mean": float(delta_s.mean().item()),
            "delta_s_median": float(delta_s.median().item()),
            "delta_s_min": float(delta_s.min().item()),
            "delta_s_max": float(delta_s.max().item()),
            "frac_delta_s_pos": float((delta_s > 0).float().mean().item()),
            "frac_delta_s_neg": float((delta_s < 0).float().mean().item()),

            "like_grad_mean": float(like_g.mean().item()),
            "like_grad_median": float(like_g.median().item()),
            "like_grad_min": float(like_g.min().item()),
            "like_grad_max": float(like_g.max().item()),
            "frac_like_grad_neg": float((like_g < 0).float().mean().item()),
            "frac_like_grad_pos": float((like_g > 0).float().mean().item()),

            "prior_grad_mean": float(prior_g.mean().item()),
            "prior_grad_median": float(prior_g.median().item()),
            "prior_grad_min": float(prior_g.min().item()),
            "prior_grad_max": float(prior_g.max().item()),
            "frac_prior_grad_neg": float((prior_g < 0).float().mean().item()),
            "frac_prior_grad_pos": float((prior_g > 0).float().mean().item()),

            "prior_contrib_grad_mean": float(prior_contrib_g.mean().item()),

            "total_grad_mean": float(total_g.mean().item()),
            "frac_total_grad_neg": float((total_g < 0).float().mean().item()),
            "frac_total_grad_pos": float((total_g > 0).float().mean().item()),

            "adam_exp_avg_before_mean": float(adam_exp_avg_before.mean().item()),
            "adam_exp_avg_after_mean": float(adam_exp_avg_after.mean().item()),

            "agree_like_grad_with_delta": sign_agreement_delta_grad(delta_s, like_g),
            "agree_prior_grad_with_delta": sign_agreement_delta_grad(delta_s, prior_g),
            "agree_total_grad_with_delta": sign_agreement_delta_grad(delta_s, total_g),

            "like_loss": float(like_loss.item()),
            "prior_loss": float(prior_loss.item()),
            "total_loss": float(total_loss.item()),
            "lr": float(lr_epoch),
        }

        rows.append(row)

    return rows


def print_top_layers(rows, batch_idx):
    print("\n" + "-" * 80)
    print(f"Batch {batch_idx}: top 10 layers with largest bitwidth DROP")
    print("-" * 80)
    for r in sorted(rows, key=lambda x: x["delta_bits_mean"])[:10]:
        print(
            f"{r['layer']} | "
            f"bits {r['bits_before_mean']:.3f}->{r['bits_after_mean']:.3f} "
            f"Δbits={r['delta_bits_mean']:.3f} | "
            f"frac ΔS+={r['frac_delta_s_pos']:.2f} | "
            f"prior_grad_mean={r['prior_grad_mean']:.3e} | "
            f"like_grad_mean={r['like_grad_mean']:.3e} | "
            f"agree_prior={r['agree_prior_grad_with_delta']:.2f} | "
            f"agree_like={r['agree_like_grad_with_delta']:.2f}"
        )

    print("\n" + "-" * 80)
    print(f"Batch {batch_idx}: top 10 layers with largest bitwidth INCREASE")
    print("-" * 80)
    for r in sorted(rows, key=lambda x: x["delta_bits_mean"], reverse=True)[:10]:
        print(
            f"{r['layer']} | "
            f"bits {r['bits_before_mean']:.3f}->{r['bits_after_mean']:.3f} "
            f"Δbits={r['delta_bits_mean']:.3f} | "
            f"frac ΔS+={r['frac_delta_s_pos']:.2f} | "
            f"prior_grad_mean={r['prior_grad_mean']:.3e} | "
            f"like_grad_mean={r['like_grad_mean']:.3e} | "
            f"agree_prior={r['agree_prior_grad_with_delta']:.2f} | "
            f"agree_like={r['agree_like_grad_with_delta']:.2f}"
        )

    print("\n" + "-" * 80)
    print(f"Batch {batch_idx}: top 10 layers with largest |raw prior grad|")
    print("-" * 80)
    for r in sorted(rows, key=lambda x: abs(x["prior_grad_mean"]), reverse=True)[:10]:
        print(
            f"{r['layer']} | "
            f"bits {r['bits_before_mean']:.3f}->{r['bits_after_mean']:.3f} "
            f"Δbits={r['delta_bits_mean']:.3f} | "
            f"prior_grad_mean={r['prior_grad_mean']:.3e} | "
            f"prior_contrib={r['prior_contrib_grad_mean']:.3e} | "
            f"frac prior-={r['frac_prior_grad_neg']:.2f} | "
            f"frac ΔS+={r['frac_delta_s_pos']:.2f}"
        )


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
        / "debug_batch0_1_out"
        / model_tag
        / dataset_tag
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "batch0_1_layer_summary.csv"
    raw_path = out_dir / "batch0_1_rows.pt"

    print("Loading model...")
    model, device = load_poseidon_model(cfg.model_path, cfg.device)
    model = model.to(device).eval()

    print("Loading candidate layers...")
    candidate_layers = load_candidate_layers(model, Path(cfg.quant_layer_path))

    print("Loading frozen batches...")
    frozen_batches = load_frozen_calibration_batches(cfg, device=device)
    frozen_batches = frozen_batches[:2]

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

    print("Caching clean outputs for batch 0 and 1...")
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

    all_rows = []

    rows0 = run_one_batch(
        batch_idx=0,
        model=model,
        step_sizes_dict=step_sizes_dict,
        frozen_batches=frozen_batches,
        clean_net_outputs=clean_net_outputs,
        ranges_dict=ranges_dict,
        sens_dict=sens_dict,
        optimizer=optimizer,
        cfg=cfg,
        device=device,
        channel_weights=channel_weights,
        lr_epoch=lr_epoch,
    )
    print_top_layers(rows0, batch_idx=0)
    all_rows.extend(rows0)

    rows1 = run_one_batch(
        batch_idx=1,
        model=model,
        step_sizes_dict=step_sizes_dict,
        frozen_batches=frozen_batches,
        clean_net_outputs=clean_net_outputs,
        ranges_dict=ranges_dict,
        sens_dict=sens_dict,
        optimizer=optimizer,
        cfg=cfg,
        device=device,
        channel_weights=channel_weights,
        lr_epoch=lr_epoch,
    )
    print_top_layers(rows1, batch_idx=1)
    all_rows.extend(rows1)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    torch.save(all_rows, raw_path)

    print(f"\nSaved CSV -> {csv_path}")
    print(f"Saved raw rows -> {raw_path}")


if __name__ == "__main__":
    main()