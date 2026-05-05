# SAPQ/debug_one_epoch_bit_drop.py
from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

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
    clamp_step_sizes_,
)
from PPQ.metrics import build_channel_param_weights, compute_avg_bits
from SAPQ.sapq_loss import compute_sapq_loss_with_prior_global


REPO = Path("/home/u6ey/yiheng.u6ey/poseidon-quantized")
RANGES_PATH = REPO / "precalculated_ranges/NS-SVS-L/best/ranges.pt"
SENS_PATH = REPO / "SAPQ/prior_sensitivity_sobo/NS-SVS-L/prior_sensitivity.pt"


def load_candidate_layers(model, quant_layer_path):
    obj = torch.load(quant_layer_path, map_location="cpu")
    layer_names = obj["quantize_layers"]
    name2mod = dict(model.named_modules())
    return [n for n in layer_names if isinstance(name2mod.get(n), nn.Linear)]


def main():
    cfg = PPQConfig()
    cfg.model_path = str(REPO / "models/NS-SVS-L")
    cfg.data_path = str(REPO / "dataset/NS-SVS")
    cfg.dataset_name = "fluids.incompressible.VortexSheet"

    cfg.calib_batchsize = 1
    cfg.calib_steps = 512
    cfg.init_bits = 8
    cfg.bmax_bits = 20
    cfg.target_bits = 8
    cfg.num_mc_samples = 10
    cfg.eta = 1e-6
    cfg.base_lr = 9.1e-4
    cfg.prior_mode = "block_sens"
    cfg.sigma0 = 0.5
    cfg.alpha = 1.0
    cfg.prior_scale = 0.0001

    model, device = load_poseidon_model(cfg.model_path, cfg.device)
    model = model.to(device).eval()

    calib_loader, _, calib_iter, _ = build_poseidon_loaders(
        dataset_name=cfg.dataset_name,
        data_path=cfg.data_path,
        calib_batchsize=cfg.calib_batchsize,
        calib_steps=cfg.calib_steps,
        val_batchsize=1,
        val_steps=1,
    )

    frozen_batches, _ = freeze_batches(calib_iter())
    print(f"[INFO] frozen batches = {len(frozen_batches)}")

    clean_outputs = get_clean_network_outputs_poseidon(
        model=model,
        frozen_batches=frozen_batches,
        device=device,
    )

    ranges_obj = torch.load(RANGES_PATH, map_location="cpu")
    ranges_dict = ranges_obj["ranges_dict"]
    for v in ranges_dict.values():
        v["weight_ranges"] = v["weight_ranges"].to(device)
        v["activation_ranges"] = v["activation_ranges"].to(device)

    sens_obj = torch.load(SENS_PATH, map_location="cpu")
    sens_dict = {k: v.to(device) for k, v in sens_obj["layer_sensitivity_raw"].items()}

    candidate_layers = load_candidate_layers(model, Path(cfg.quant_layer_path))
    target_layers = get_compatible_linear_layers(model, candidate_layers, ranges_dict)

    step_sizes_dict, params = initialize_step_sizes(
        ranges_dict=ranges_dict,
        target_layers=target_layers,
        init_bits=cfg.init_bits,
        bmax_bits=cfg.bmax_bits,
        device=device,
        model_path=cfg.model_path,
        percentile_prob=cfg.percentile_prob,
        repo_root=cfg.repo_root,
        weight_only=True,
    )

    channel_weights = build_channel_param_weights(model, target_layers)

    def avg_bits():
        return compute_avg_bits(step_sizes_dict, ranges_dict, channel_weights)

    optimizer = optim.Adam(params, lr=cfg.base_lr)

    print(f"[INIT] AvgBits = {avg_bits():.6f}")

    # ---- one update only ----
    optimizer.zero_grad()
    total_loss, like_loss, prior_loss = compute_sapq_loss_with_prior_global(
        model=model,
        step_sizes_dict=step_sizes_dict,
        frozen_batches=frozen_batches,
        clean_net_outputs=clean_outputs,
        ranges_dict=ranges_dict,
        sens_dict=sens_dict,
        batch_idx=0,
        num_mc_samples=cfg.num_mc_samples,
        eta=cfg.eta,
        prior_mode=cfg.prior_mode,
        b_target=cfg.target_bits,
        sigma0=cfg.sigma0,
        alpha=cfg.alpha,
        prior_scale=cfg.prior_scale,
        device=device,
    )
    total_loss.backward()

    # gradient magnitude diagnostic
    grad_abs_sum = 0.0
    grad_abs_mean_list = []
    for p in params:
        if p.grad is not None:
            grad_abs_sum += p.grad.abs().sum().item()
            grad_abs_mean_list.append(p.grad.abs().mean().item())

    print(f"[ONE STEP BEFORE] loss={total_loss.item():.6e}, like={like_loss.item():.6e}, prior={prior_loss.item():.6e}")
    print(f"[GRAD] abs_sum={grad_abs_sum:.6e}, abs_mean={sum(grad_abs_mean_list)/len(grad_abs_mean_list):.6e}")

    optimizer.step()
    clamp_step_sizes_(
        step_sizes_dict=step_sizes_dict,
        ranges_dict=ranges_dict,
        bmax_bits=cfg.bmax_bits,
        device=device,
        weight_only=True,
    )

    print(f"[AFTER 1 UPDATE] AvgBits = {avg_bits():.6f}")

    # ---- continue one epoch ----
    for batch_idx in range(1, len(frozen_batches)):
        optimizer.zero_grad()
        total_loss, like_loss, prior_loss = compute_sapq_loss_with_prior_global(
            model=model,
            step_sizes_dict=step_sizes_dict,
            frozen_batches=frozen_batches,
            clean_net_outputs=clean_outputs,
            ranges_dict=ranges_dict,
            sens_dict=sens_dict,
            batch_idx=batch_idx,
            num_mc_samples=cfg.num_mc_samples,
            eta=cfg.eta,
            prior_mode=cfg.prior_mode,
            b_target=cfg.target_bits,
            sigma0=cfg.sigma0,
            alpha=cfg.alpha,
            prior_scale=cfg.prior_scale,
            device=device,
        )
        total_loss.backward()
        optimizer.step()
        clamp_step_sizes_(
            step_sizes_dict=step_sizes_dict,
            ranges_dict=ranges_dict,
            bmax_bits=cfg.bmax_bits,
            device=device,
            weight_only=True,
        )

        if batch_idx in [9, 49, 99, 199, 511]:
            print(f"[AFTER {batch_idx+1:4d} UPDATES] AvgBits = {avg_bits():.6f}")

    print(f"[AFTER 1 EPOCH] AvgBits = {avg_bits():.6f}")


if __name__ == "__main__":
    main()