# SAPQ/debug_gradient_direction.py
from __future__ import annotations

import copy
from pathlib import Path
import torch
import torch.nn as nn

from PPQ.config import PPQConfig
from PPQ.poseidon_utils import load_poseidon_model, build_poseidon_loaders, get_clean_network_outputs_poseidon
from PPQ.optimize import initialize_step_sizes, freeze_batches, get_compatible_linear_layers
from PPQ.metrics import build_channel_param_weights, compute_avg_bits
from SAPQ.sapq_loss import compute_mc_negative_loglikelihood_network_global


def load_candidate_layers(model, path):
    obj = torch.load(path, map_location="cpu")
    names = obj["quantize_layers"]
    name2mod = dict(model.named_modules())
    return [n for n in names if isinstance(name2mod.get(n, None), nn.Linear)]


def main():
    cfg = PPQConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.model_path = "/home/u6ey/yiheng.u6ey/poseidon-quantized/models/NS-SVS-L"
    cfg.data_path = "/home/u6ey/yiheng.u6ey/poseidon-quantized/dataset/NS-SVS"
    cfg.dataset_name = "fluids.incompressible.VortexSheet"

    ranges_path = Path("/home/u6ey/yiheng.u6ey/poseidon-quantized/precalculated_ranges/NS-SVS-L/best/ranges.pt")

    print("Loading model...")
    model, device = load_poseidon_model(cfg.model_path, cfg.device)
    model = model.to(device).eval()

    print("Building loaders...")
    calib_loader, _, calib_iter, _ = build_poseidon_loaders(
        dataset_name=cfg.dataset_name,
        data_path=cfg.data_path,
        calib_batchsize=cfg.calib_batchsize,
        calib_steps=4,          # only small debug
        val_batchsize=cfg.val_batchsize,
        val_steps=1,
    )

    frozen_batches, _ = freeze_batches(calib_iter())
    frozen_batches = frozen_batches[:4]

    print("Loading ranges...")
    ranges_obj = torch.load(ranges_path, map_location="cpu")
    ranges_dict = ranges_obj["ranges_dict"]
    for v in ranges_dict.values():
        v["weight_ranges"] = v["weight_ranges"].to(device)
        v["activation_ranges"] = v["activation_ranges"].to(device)

    candidate_layers = load_candidate_layers(model, Path(cfg.quant_layer_path))
    target_layers = get_compatible_linear_layers(model, candidate_layers, ranges_dict)

    print("Initializing 8-bit step sizes...")
    step_sizes_dict, params = initialize_step_sizes(
        ranges_dict=ranges_dict,
        target_layers=target_layers,
        init_bits=8,
        bmax_bits=20,
        device=device,
        model_path=cfg.model_path,
        percentile_prob=cfg.percentile_prob,
        repo_root=cfg.repo_root,
        weight_only=True,
    )

    channel_weights = build_channel_param_weights(model, target_layers)

    with torch.no_grad():
        avg_bits_before = compute_avg_bits(step_sizes_dict, ranges_dict, channel_weights)
    print(f"[Before] AvgBits = {avg_bits_before:.6f}")

    print("Computing likelihood gradient at initialization...")
    loss = compute_mc_negative_loglikelihood_network_global(
        model=model,
        step_sizes_dict=step_sizes_dict,
        frozen_batches=frozen_batches,
        clean_net_outputs=get_clean_network_outputs_poseidon(model, frozen_batches, device),
        batch_idx=0,
        num_mc_samples=10,
        eta=1e-6,
        device=device,
    )

    loss.backward()

    # total = 0
    # push_S_up = 0
    # push_S_down = 0

    # # Adam/SGD update: S_new = S - lr * grad
    # # grad < 0 means S increases, bits decrease.
    # for name, entry in step_sizes_dict.items():
    #     w_step = entry[0] if isinstance(entry, tuple) else entry
    #     if w_step.grad is None:
    #         continue

    #     g = w_step.grad.detach()
    #     total += g.numel()
    #     push_S_up += (g < 0).sum().item()
    #     push_S_down += (g > 0).sum().item()

    # print("\n========== GRADIENT DIRECTION ==========")
    # print(f"loss: {loss.item():.6e}")
    # print(f"channels checked: {total}")
    # print(f"grad < 0  => S up, bits down: {push_S_up} ({push_S_up / total * 100:.2f}%)")
    # print(f"grad > 0  => S down, bits up: {push_S_down} ({push_S_down / total * 100:.2f}%)")

    # print("\nInterpretation:") 
    # print("If most gradients are negative, the likelihood itself pushes step sizes larger, so bitwidth drops.")
    total = 0
    push_S_up = 0
    push_S_down = 0

    sum_grad_neg_abs = 0.0   # grad < 0 => S increases => bits decrease
    sum_grad_pos_abs = 0.0   # grad > 0 => S decreases => bits increase

    weighted_delta_S = 0.0
    weighted_delta_bits_proxy = 0.0

    for name, entry in step_sizes_dict.items():
        w_step = entry[0] if isinstance(entry, tuple) else entry
        if w_step.grad is None:
            continue

        g = w_step.grad.detach()
        s = w_step.detach()

        total += g.numel()

        neg = g < 0
        pos = g > 0

        push_S_up += neg.sum().item()
        push_S_down += pos.sum().item()

        sum_grad_neg_abs += g[neg].abs().sum().item()
        sum_grad_pos_abs += g[pos].abs().sum().item()

        # proxy for bit change:
        # B = log2(R/S), so dB/dS = -1 / (S ln 2)
        # SGD update: dS = -lr * grad
        # therefore dB ≈ grad / (S ln 2)
        weighted_delta_bits_proxy += (g / torch.clamp(s, min=1e-12)).sum().item()

    print("\n========== GRADIENT DIRECTION + MAGNITUDE ==========")
    print(f"loss: {loss.item():.6e}")
    print(f"channels checked: {total}")

    print(f"grad < 0 => S up, bits down: {push_S_up} ({push_S_up / total * 100:.2f}%)")
    print(f"grad > 0 => S down, bits up:   {push_S_down} ({push_S_down / total * 100:.2f}%)")

    print("\nMagnitude:")
    print(f"sum |grad| where grad < 0: {sum_grad_neg_abs:.6e}")
    print(f"sum |grad| where grad > 0: {sum_grad_pos_abs:.6e}")

    ratio = sum_grad_neg_abs / max(sum_grad_pos_abs, 1e-30)
    print(f"neg/pos magnitude ratio:  {ratio:.6f}")

    print("\nBit-change proxy:")
    print(f"sum grad/S = {weighted_delta_bits_proxy:.6e}")
    if weighted_delta_bits_proxy < 0:
        print("Prediction: average bitwidth will DECREASE.")
    else:
        print("Prediction: average bitwidth will INCREASE.")


if __name__ == "__main__":
    main()