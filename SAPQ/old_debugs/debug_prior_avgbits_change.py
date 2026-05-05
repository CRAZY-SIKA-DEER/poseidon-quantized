# SAPQ/debug_prior_avgbits_change.py
from __future__ import annotations

from pathlib import Path
import torch

REPO = Path("/home/u6ey/yiheng.u6ey/poseidon-quantized")

FINAL_STEP_PATH = REPO / "sapq_experiments/NS-SVS-L/NS-SVS/network_block_sens_sobo/raw/sapq_global_step_sizes.pt"
RANGE_PATH = REPO / "precalculated_ranges/NS-SVS-L/best/ranges.pt"

TARGET_BITS = 8.0
SIGMA0 = 0.5
EPS = 1e-8


def get_w_step(entry):
    if isinstance(entry, tuple):
        return entry[0]
    return entry


def compute_bits_from_steps(step_sizes_dict, ranges_dict):
    all_bits = []
    layer_rows = []

    for name, entry in step_sizes_dict.items():
        if name not in ranges_dict:
            continue
        if "weight_ranges" not in ranges_dict[name]:
            continue

        w_step = get_w_step(entry).float().cpu()
        w_range = ranges_dict[name]["weight_ranges"].float().cpu()

        if w_step.numel() != w_range.numel():
            continue

        bits = torch.log2(torch.clamp(w_range, min=EPS) / torch.clamp(w_step, min=EPS))
        all_bits.append(bits)

        prior = ((bits - TARGET_BITS).pow(2) / (2.0 * (SIGMA0 ** 2))).sum()

        layer_rows.append({
            "name": name,
            "mean_bits": bits.mean().item(),
            "min_bits": bits.min().item(),
            "max_bits": bits.max().item(),
            "prior": prior.item(),
        })

    all_bits = torch.cat(all_bits)
    total_prior = ((all_bits - TARGET_BITS).pow(2) / (2.0 * (SIGMA0 ** 2))).sum()

    return all_bits, total_prior.item(), layer_rows


def summarize(tag, bits, prior):
    print(f"\n========== {tag} ==========")
    print(f"avg bits:       {bits.mean().item():.6f}")
    print(f"median bits:    {bits.median().item():.6f}")
    print(f"std bits:       {bits.std().item():.6f}")
    print(f"min bits:       {bits.min().item():.6f}")
    print(f"max bits:       {bits.max().item():.6f}")
    print(f"prior loss:     {prior:.6f}")
    print(f"mean |B-8|:     {(bits - TARGET_BITS).abs().mean().item():.6f}")
    print(f"mean (B-8)^2:   {(bits - TARGET_BITS).pow(2).mean().item():.6f}")

    for q in [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]:
        print(f"q{int(q*100):02d}:            {torch.quantile(bits, q).item():.6f}")


def main():
    print(f"[LOAD] final steps: {FINAL_STEP_PATH}")
    final_obj = torch.load(FINAL_STEP_PATH, map_location="cpu")
    final_steps = final_obj["step_sizes_dict"]

    print(f"[LOAD] ranges: {RANGE_PATH}")
    range_obj = torch.load(RANGE_PATH, map_location="cpu")
    ranges_dict = range_obj["ranges_dict"]

    # Build initial 8-bit steps directly from ranges:
    # B = log2(R/S), so S = R / 2^B
    init_steps = {}
    for name, rec in ranges_dict.items():
        if "weight_ranges" not in rec:
            continue
        w_range = rec["weight_ranges"].float().cpu()
        init_w_step = w_range / (2.0 ** TARGET_BITS)
        init_steps[name] = (init_w_step, None)

    init_bits, init_prior, init_rows = compute_bits_from_steps(init_steps, ranges_dict)
    final_bits, final_prior, final_rows = compute_bits_from_steps(final_steps, ranges_dict)

    summarize("INITIAL 8-BIT", init_bits, init_prior)
    summarize("FINAL TRAINED", final_bits, final_prior)

    print("\n========== CHANGE ==========")
    print(f"avg bits change:       {final_bits.mean().item() - init_bits.mean().item():.6f}")
    print(f"prior loss change:     {final_prior - init_prior:.6f}")
    print(f"std bits change:       {final_bits.std().item() - init_bits.std().item():.6f}")
    print(f"mean |B-8| change:     {(final_bits - TARGET_BITS).abs().mean().item() - (init_bits - TARGET_BITS).abs().mean().item():.6f}")
    print(f"mean (B-8)^2 change:   {(final_bits - TARGET_BITS).pow(2).mean().item() - (init_bits - TARGET_BITS).pow(2).mean().item():.6f}")

    if final_bits.mean() < init_bits.mean() and final_prior < init_prior:
        print("\n[RESULT] Both average bitwidth and prior loss decreased.")
    elif final_bits.mean() < init_bits.mean() and final_prior > init_prior:
        print("\n[RESULT] Average bitwidth decreased, but prior loss increased.")
    else:
        print("\n[RESULT] Check printed values above.")

    print("\n========== TOP 20 LAYERS WITH BIGGEST BIT DROP ==========")
    init_map = {r["name"]: r for r in init_rows}
    changes = []
    for r in final_rows:
        name = r["name"]
        if name not in init_map:
            continue
        changes.append({
            "name": name,
            "init_mean_bits": init_map[name]["mean_bits"],
            "final_mean_bits": r["mean_bits"],
            "delta_bits": r["mean_bits"] - init_map[name]["mean_bits"],
            "init_prior": init_map[name]["prior"],
            "final_prior": r["prior"],
            "delta_prior": r["prior"] - init_map[name]["prior"],
        })

    for r in sorted(changes, key=lambda x: x["delta_bits"])[:20]:
        print(
            f"dB={r['delta_bits']:8.4f} | "
            f"B: {r['init_mean_bits']:7.3f} -> {r['final_mean_bits']:7.3f} | "
            f"dPrior={r['delta_prior']:12.3f} | "
            f"{r['name']}"
        )

    print("\n========== TOP 20 LAYERS WITH BIGGEST PRIOR DECREASE ==========")
    for r in sorted(changes, key=lambda x: x["delta_prior"])[:20]:
        print(
            f"dPrior={r['delta_prior']:12.3f} | "
            f"B: {r['init_mean_bits']:7.3f} -> {r['final_mean_bits']:7.3f} | "
            f"dB={r['delta_bits']:8.4f} | "
            f"{r['name']}"
        )


if __name__ == "__main__":
    main()