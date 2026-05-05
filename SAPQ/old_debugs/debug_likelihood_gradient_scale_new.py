# SAPQ/debug_likelihood_gradient_scale.py
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
from SAPQ.sapq_loss import compute_mc_negative_loglikelihood_network_global


REPO = Path("/home/u6ey/yiheng.u6ey/poseidon-quantized")
RANGES_PATH = REPO / "precalculated_ranges/NS-SVS-L/best/ranges.pt"


def load_candidate_layers(model, quant_layer_path):
    obj = torch.load(quant_layer_path, map_location="cpu")
    names = obj["quantize_layers"]
    name2mod = dict(model.named_modules())
    return [n for n in names if isinstance(name2mod.get(n), nn.Linear)]


def get_w_step(entry):
    if isinstance(entry, tuple):
        return entry[0]
    return entry


def collect_stats(step_sizes_dict, ranges_dict):
    grads, steps, ranges = [], [], []

    for name, entry in step_sizes_dict.items():
        if name not in ranges_dict:
            continue

        s = get_w_step(entry)
        if s.grad is None:
            continue

        r = ranges_dict[name]["weight_ranges"]

        grads.append(s.grad.detach().flatten().cpu())
        steps.append(s.detach().flatten().cpu())
        ranges.append(r.detach().flatten().cpu())

    g = torch.cat(grads)
    s = torch.cat(steps)
    r = torch.cat(ranges)

    bit_proxy = g / torch.clamp(s, min=1e-12)
    bits = torch.log2(torch.clamp(r, min=1e-12) / torch.clamp(s, min=1e-12))

    print("\n========== GRADIENT STATS ==========")
    print(f"grad abs mean:     {g.abs().mean().item():.6e}")
    print(f"grad abs max:      {g.abs().max().item():.6e}")
    print(f"grad abs q99:      {torch.quantile(g.abs(), 0.99).item():.6e}")

    print("\n========== STEP SIZE STATS ==========")
    print(f"S mean:            {s.mean().item():.6e}")
    print(f"S median:          {s.median().item():.6e}")
    print(f"S min:             {s.min().item():.6e}")
    print(f"S max:             {s.max().item():.6e}")

    print("\n========== RANGE (R) STATS ==========")
    print(f"R mean:            {r.mean().item():.6e}")
    print(f"R median:          {r.median().item():.6e}")
    print(f"R min:             {r.min().item():.6e}")
    print(f"R max:             {r.max().item():.6e}")

    print("\n========== R/S (BIT SCALE) ==========")
    rs = r / torch.clamp(s, min=1e-12)
    print(f"(R/S) mean:        {rs.mean().item():.6e}")
    print(f"(R/S) median:      {rs.median().item():.6e}")
    print(f"(R/S) max:         {rs.max().item():.6e}")

    print("\n========== BITWIDTH STATS ==========")
    print(f"B mean:            {bits.mean().item():.6f}")
    print(f"B min:             {bits.min().item():.6f}")
    print(f"B max:             {bits.max().item():.6f}")

    print("\n========== BITWIDTH CHANGE PROXY ==========")
    print(f"sum(g/S):          {bit_proxy.sum().item():.6e}")
    print(f"mean(g/S):         {bit_proxy.mean().item():.6e}")

    if bit_proxy.sum() > 0:
        print("[PREDICT] bitwidth ↑")
    else:
        print("[PREDICT] bitwidth ↓")


def main():
    cfg = PPQConfig()
    cfg.model_path = str(REPO / "models/NS-SVS-L")
    cfg.data_path = str(REPO / "dataset/NS-SVS")
    cfg.dataset_name = "fluids.incompressible.VortexSheet"

    cfg.calib_batchsize = 1
    cfg.calib_steps = 4
    cfg.init_bits = 8
    cfg.bmax_bits = 20
    cfg.num_mc_samples = 10
    cfg.eta = 1e-6
    cfg.base_lr = 9.1e-4

    model, device = load_poseidon_model(cfg.model_path, cfg.device)
    model = model.to(device).eval()

    _, _, calib_iter, _ = build_poseidon_loaders(
        dataset_name=cfg.dataset_name,
        data_path=cfg.data_path,
        calib_batchsize=cfg.calib_batchsize,
        calib_steps=cfg.calib_steps,
        val_batchsize=1,
        val_steps=1,
    )

    frozen_batches, _ = freeze_batches(calib_iter())
    clean_outputs = get_clean_network_outputs_poseidon(model, frozen_batches, device)

    ranges_obj = torch.load(RANGES_PATH, map_location="cpu")
    ranges_dict = ranges_obj["ranges_dict"]
    for v in ranges_dict.values():
        v["weight_ranges"] = v["weight_ranges"].to(device)
        v["activation_ranges"] = v["activation_ranges"].to(device)

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

    print(f"[INIT] AvgBits = {compute_avg_bits(step_sizes_dict, ranges_dict, channel_weights):.6f}")

    loss = compute_mc_negative_loglikelihood_network_global(
        model=model,
        step_sizes_dict=step_sizes_dict,
        frozen_batches=frozen_batches,
        clean_net_outputs=clean_outputs,
        batch_idx=0,
        num_mc_samples=cfg.num_mc_samples,
        eta=cfg.eta,
        device=device,
    )

    loss.backward()

    print(f"\n[LIKELIHOOD LOSS] {loss.item():.6e}")
    collect_stats(step_sizes_dict, ranges_dict)

    optimizer = optim.Adam(params, lr=cfg.base_lr)

    before_bits = compute_avg_bits(step_sizes_dict, ranges_dict, channel_weights)
    optimizer.step()
    clamp_step_sizes_(step_sizes_dict, ranges_dict, cfg.bmax_bits, device, True)
    after_bits = compute_avg_bits(step_sizes_dict, ranges_dict, channel_weights)

    print("\n========== ONE STEP EFFECT ==========")
    print(f"AvgBits before: {before_bits:.6f}")
    print(f"AvgBits after:  {after_bits:.6f}")
    print(f"Delta Bits:     {after_bits - before_bits:.6f}")


if __name__ == "__main__":
    main()