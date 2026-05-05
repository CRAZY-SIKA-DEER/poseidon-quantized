# SAPQ/debug_prior_bits_ns_svs.py
from __future__ import annotations

import torch
from pathlib import Path

REPO = Path("/home/u6ey/yiheng.u6ey/poseidon-quantized")

STEP_PATH = REPO / "sapq_experiments/NS-SVS-L/NS-SVS/network_block_sens_sobo/raw/sapq_global_step_sizes.pt"
RANGE_PATH = REPO / "precalculated_ranges/NS-SVS-L/best/ranges.pt"
SENS_PATH = REPO / "SAPQ/prior_sensitivity_sobo/NS-SVS-L/prior_sensitivity.pt"

TARGET_BITS = 8.0
SIGMA0 = 0.5
ALPHA = 1.0
EPS = 1e-8


def get_w_step(entry):
    if isinstance(entry, tuple):
        return entry[0]
    return entry


def compute_layer_stats(name, w_step, w_range, sens=None):
    w_step = w_step.float().cpu()
    w_range = w_range.float().cpu()

    bits = torch.log2(torch.clamp(w_range, min=EPS) / torch.clamp(w_step, min=EPS))

    if sens is None:
        sens = torch.zeros_like(bits)
    else:
        sens = sens.float().cpu()
        if sens.numel() == 1:
            sens = sens.expand_as(bits)

    # paper-correct variance:
    var = (SIGMA0 ** 2) * (1.0 + ALPHA * sens)
    var = torch.clamp(var, min=EPS)

    prior_per_channel = (bits - TARGET_BITS).pow(2) / (2.0 * var)

    return {
        "name": name,
        "num_channels": bits.numel(),
        "mean_bits": bits.mean().item(),
        "min_bits": bits.min().item(),
        "max_bits": bits.max().item(),
        "prior": prior_per_channel.sum().item(),
        "mean_sens": sens.mean().item(),
        "max_sens": sens.max().item(),
    }


def main():
    print(f"[LOAD] steps: {STEP_PATH}")
    step_obj = torch.load(STEP_PATH, map_location="cpu")
    final_steps = step_obj["step_sizes_dict"]

    print(f"[LOAD] ranges: {RANGE_PATH}")
    range_obj = torch.load(RANGE_PATH, map_location="cpu")
    ranges_dict = range_obj["ranges_dict"]

    print(f"[LOAD] sensitivity: {SENS_PATH}")
    sens_obj = torch.load(SENS_PATH, map_location="cpu")
    sens_dict = sens_obj["layer_sensitivity_raw"]

    rows = []

    total_prior = 0.0
    total_weighted_bits = 0.0
    total_params = 0.0

    for name, entry in final_steps.items():
        if name not in ranges_dict:
            continue
        if "weight_ranges" not in ranges_dict[name]:
            continue

        w_step = get_w_step(entry)
        w_range = ranges_dict[name]["weight_ranges"]

        if w_step.numel() != w_range.numel():
            print(f"[SKIP shape mismatch] {name}: step={w_step.shape}, range={w_range.shape}")
            continue

        sens = sens_dict.get(name, None)

        stat = compute_layer_stats(name, w_step, w_range, sens)
        rows.append(stat)

        # weighted avg bits: each output channel weighted by in_features approximately.
        # If no model is loaded, use channel count only here.
        # This is unweighted-by-parameter, but good for diagnosis.
        bits = torch.log2(torch.clamp(w_range.float(), min=EPS) / torch.clamp(w_step.float(), min=EPS))
        total_weighted_bits += bits.sum().item()
        total_params += bits.numel()
        total_prior += stat["prior"]

    avg_bits = total_weighted_bits / max(total_params, 1.0)

    print("\n========== GLOBAL SUMMARY ==========")
    print(f"num_layers used:          {len(rows)}")
    print(f"unweighted avg bits:      {avg_bits:.6f}")
    print(f"total prior loss:         {total_prior:.6f}")
    print(f"target bits:              {TARGET_BITS}")
    print(f"sigma0:                   {SIGMA0}")
    print(f"alpha:                    {ALPHA}")

    print("\n========== TOP 20 LOWEST MEAN BIT LAYERS ==========")
    for r in sorted(rows, key=lambda x: x["mean_bits"])[:20]:
        print(
            f"{r['mean_bits']:8.4f} bits | "
            f"prior={r['prior']:12.3f} | "
            f"sens_mean={r['mean_sens']:.3e} | "
            f"sens_max={r['max_sens']:.3e} | "
            f"{r['name']}"
        )

    print("\n========== TOP 20 HIGHEST PRIOR CONTRIBUTION LAYERS ==========")
    for r in sorted(rows, key=lambda x: x["prior"], reverse=True)[:20]:
        print(
            f"prior={r['prior']:12.3f} | "
            f"bits_mean={r['mean_bits']:8.4f} | "
            f"bits_min={r['min_bits']:8.4f} | "
            f"bits_max={r['max_bits']:8.4f} | "
            f"sens_mean={r['mean_sens']:.3e} | "
            f"{r['name']}"
        )

    print("\n========== TOP 20 HIGHEST SENSITIVITY LAYERS ==========")
    for r in sorted(rows, key=lambda x: x["mean_sens"], reverse=True)[:20]:
        print(
            f"sens_mean={r['mean_sens']:.3e} | "
            f"sens_max={r['max_sens']:.3e} | "
            f"bits_mean={r['mean_bits']:8.4f} | "
            f"prior={r['prior']:12.3f} | "
            f"{r['name']}"
        )


if __name__ == "__main__":
    main()