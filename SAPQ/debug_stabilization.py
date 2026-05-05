# SAPQ/debug_stabilization.py
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
MAX_BATCHES = 65
SELECTED_BATCHES = {0, 1, 2, 3, 4, 8, 16, 32, 64}
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


def get_adam_update_proxy(optimizer, w_step, eps=1e-8):
    state = optimizer.state.get(w_step, None)
    if state is None or "exp_avg" not in state or "exp_avg_sq" not in state:
        return torch.zeros_like(w_step.detach())
    return state["exp_avg"].detach() / (state["exp_avg_sq"].detach().sqrt() + eps)


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
        / "debug_stabilization_out"
        / model_tag
        / dataset_tag
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_json = out_dir / "batch_global_summary.json"
    selected_csv = out_dir / "selected_batch_layer_summary.csv"

    print("Loading model...")
    model, device = load_poseidon_model(cfg.model_path, cfg.device)
    model = model.to(device).eval()

    candidate_layers = load_candidate_layers(model, Path(cfg.quant_layer_path))

    frozen_batches = load_frozen_calibration_batches(cfg, device=device)
    frozen_batches = frozen_batches[:MAX_BATCHES]

    def frozen_iter():
        for b in frozen_batches:
            yield b

    ranges_dict = maybe_load_or_compute_ranges(
        cfg=cfg,
        model=model,
        frozen_iter=frozen_iter,
        candidate_layers=candidate_layers,
        device=device,
    )

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

    sens_dict = {name: sens_dict[name] for name in target_layers if name in sens_dict}
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

    batch_records = []
    selected_rows = []

    for batch_idx in range(MAX_BATCHES):
        avg_before = compute_avg_bits(step_sizes_dict, ranges_dict, channel_weights)
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

        # Global diagnostics
        delta_s_all = []
        delta_bits_all = []
        like_abs_all = []
        prior_contrib_abs_all = []
        adam_proxy_abs_all = []

        num_layers_above_8 = 0
        num_layers_below_4 = 0

        layer_rows_this_batch = []

        for name, (w_step_after, _) in step_sizes_dict.items():
            if name not in ranges_dict:
                continue

            sb = s_before[name].to(device)
            sa = w_step_after.detach()
            delta_s = sa - sb

            bits_before = layer_bits(name, sb, ranges_dict)
            bits_after = layer_bits(name, sa, ranges_dict)
            delta_bits = bits_after - bits_before

            like_g = like_grads[name].to(device)
            prior_g = prior_grads[name].to(device)
            prior_contrib_g = PRIOR_WEIGHT * prior_g
            total_g = total_grads[name].to(device)

            adam_proxy = get_adam_update_proxy(optimizer, w_step_after).to(device)

            if bits_after.mean().item() > 8.0:
                num_layers_above_8 += 1
            if bits_after.mean().item() < 4.0:
                num_layers_below_4 += 1

            delta_s_all.append(delta_s.flatten())
            delta_bits_all.append(delta_bits.flatten())
            like_abs_all.append(like_g.abs().flatten())
            prior_contrib_abs_all.append(prior_contrib_g.abs().flatten())
            adam_proxy_abs_all.append(adam_proxy.abs().flatten())

            if batch_idx in SELECTED_BATCHES:
                layer_rows_this_batch.append(
                    {
                        "batch_idx": batch_idx,
                        "layer": name,
                        "bits_before_mean": float(bits_before.mean().item()),
                        "bits_after_mean": float(bits_after.mean().item()),
                        "delta_bits_mean": float(delta_bits.mean().item()),
                        "delta_s_mean": float(delta_s.mean().item()),
                        "frac_delta_s_pos": float((delta_s > 0).float().mean().item()),
                        "frac_delta_s_neg": float((delta_s < 0).float().mean().item()),
                        "like_grad_mean": float(like_g.mean().item()),
                        "prior_grad_mean": float(prior_g.mean().item()),
                        "prior_contrib_grad_mean": float(prior_contrib_g.mean().item()),
                        "total_grad_mean": float(total_g.mean().item()),
                        "mean_abs_like_grad": float(like_g.abs().mean().item()),
                        "mean_abs_prior_contrib_grad": float(prior_contrib_g.abs().mean().item()),
                        "mean_abs_adam_proxy": float(adam_proxy.abs().mean().item()),
                        "agree_like_with_delta": sign_agreement_delta_grad(delta_s, like_g),
                        "agree_prior_with_delta": sign_agreement_delta_grad(delta_s, prior_g),
                        "agree_total_with_delta": sign_agreement_delta_grad(delta_s, total_g),
                    }
                )

        delta_s_all = torch.cat(delta_s_all)
        delta_bits_all = torch.cat(delta_bits_all)
        like_abs_all = torch.cat(like_abs_all)
        prior_contrib_abs_all = torch.cat(prior_contrib_abs_all)
        adam_proxy_abs_all = torch.cat(adam_proxy_abs_all)

        batch_records.append(
            {
                "batch_idx": batch_idx,
                "avg_bits_before": float(avg_before),
                "avg_bits_after": float(avg_after),
                "delta_avg_bits": float(avg_after - avg_before),
                "like_loss": float(like_loss.item()),
                "prior_loss": float(prior_loss.item()),
                "prior_contribution": float((PRIOR_WEIGHT * prior_loss).item()),
                "mean_abs_delta_s": float(delta_s_all.abs().mean().item()),
                "mean_abs_delta_bits": float(delta_bits_all.abs().mean().item()),
                "frac_delta_s_pos": float((delta_s_all > 0).float().mean().item()),
                "frac_delta_s_neg": float((delta_s_all < 0).float().mean().item()),
                "mean_abs_like_grad": float(like_abs_all.mean().item()),
                "mean_abs_prior_contrib_grad": float(prior_contrib_abs_all.mean().item()),
                "mean_abs_adam_proxy": float(adam_proxy_abs_all.mean().item()),
                "num_layers_above_8": int(num_layers_above_8),
                "num_layers_below_4": int(num_layers_below_4),
            }
        )

        if batch_idx in SELECTED_BATCHES:
            selected_rows.extend(layer_rows_this_batch)

            print("\n" + "=" * 80)
            print(f"SELECTED BATCH {batch_idx}")
            print("=" * 80)
            print(f"AvgBits {avg_before:.4f} -> {avg_after:.4f}")
            print(f"Like={like_loss.item():.6f} | Prior={prior_loss.item():.6f}")
            print(f"mean_abs_delta_bits={delta_bits_all.abs().mean().item():.6f}")
            print(f"mean_abs_like_grad={like_abs_all.mean().item():.6e}")
            print(f"mean_abs_prior_contrib_grad={prior_contrib_abs_all.mean().item():.6e}")
            print(f"layers_above_8={num_layers_above_8}, layers_below_4={num_layers_below_4}")

            print("\nTop 8 largest |delta_bits| layers:")
            for r in sorted(layer_rows_this_batch, key=lambda x: abs(x["delta_bits_mean"]), reverse=True)[:8]:
                print(
                    f"{r['layer']} | "
                    f"bits {r['bits_before_mean']:.2f}->{r['bits_after_mean']:.2f} "
                    f"Δbits={r['delta_bits_mean']:.2f} | "
                    f"agree_prior={r['agree_prior_with_delta']:.2f} "
                    f"agree_like={r['agree_like_with_delta']:.2f} | "
                    f"|like_g|={r['mean_abs_like_grad']:.2e} "
                    f"|prior_g_eff|={r['mean_abs_prior_contrib_grad']:.2e}"
                )

        else:
            print(
                f"[BATCH {batch_idx:04d}] "
                f"AvgBits {avg_before:.4f}->{avg_after:.4f} | "
                f"|Δbits|={delta_bits_all.abs().mean().item():.4f} | "
                f"above8={num_layers_above_8} below4={num_layers_below_4}"
            )

    with open(batch_json, "w") as f:
        json.dump(batch_records, f, indent=2)

    with open(selected_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(selected_rows[0].keys()))
        writer.writeheader()
        writer.writerows(selected_rows)

    print(f"\nSaved batch summary -> {batch_json}")
    print(f"Saved selected layer summary -> {selected_csv}")


if __name__ == "__main__":
    main()