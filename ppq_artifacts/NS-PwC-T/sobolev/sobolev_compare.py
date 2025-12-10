#!/usr/bin/env python
"""
Compare:
  - Full-precision Poseidon (NS-PwC-T)
  - PPQ (original step sizes)
  - PPQ (Sobolev-augmented step sizes)

Metrics:
  - l1_mean
  - relative_l1_mean
  - l2_mean
  - relative_l2_mean
  - divergence_ratio

This script lives at:
  ppq_artifacts/NS-PwC-T/sobolev/sobolev_compare.py
"""

import os
import sys
import json
import numpy as np
import torch

# ---------------------------------------------------------------------
# Path setup: go from sobolev/ → NS-PwC-T → ppq_artifacts → repo root
# ---------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PPQ_T_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))          # ppq_artifacts/NS-PwC-T
PROJECT_ROOT = os.path.abspath(os.path.join(PPQ_T_DIR, "..", ".."))     # repo root

# Make repo root importable
sys.path.insert(0, PROJECT_ROOT)

# Import helpers from the B_test script
from PPQ.B_test import (
    load_poseidon_model,
    build_poseidon_loaders,
    evaluate_model,
    load_ppq_step_sizes,
    load_quantize_layers,
    make_quantized_copy,
)

# ---------------------------------------------------------------------
# NS-PwC-T specific config (EDITABLE)
# ---------------------------------------------------------------------
# Model + data config for NS-PwC-T
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "NS-PwC-T")
DATA_PATH    = os.path.join(PROJECT_ROOT, "dataset", "NS-PwC")
DATASET_NAME = "fluids.incompressible.PiecewiseConstants"

# PPQ artifacts directory for NS-PwC-T
PPQ_DIR_T = PPQ_T_DIR   # i.e. ppq_artifacts/NS-PwC-T

# Quantize-layer list for NS-PwC-T
# If your file is different, just edit this path:
QUANT_LAYERS_PATH_T = os.path.join(
    PROJECT_ROOT, "inspect_layers", "T_quantize_layers.pt"
)
# If you are reusing B_quantize_layers for T as well, then use:
# QUANT_LAYERS_PATH_T = os.path.join(PROJECT_ROOT, "inspect_layers", "B_quantize_layers.pt")

# Directory holding the step-size .pt files (can change to a subfolder if you like)
STEP_DIR = PPQ_DIR_T
# STEP_DIR = os.path.join(PPQ_DIR_T, "sobolev")  # if you later move them into sobolev/

# Filenames for the two PPQ variants to compare (EDITABLE)
PPQ_STEPS_FILE_ORIG = "ppq_step_sizes-1-5-conv.pt"
PPQ_STEPS_FILE_SOB  = "sobolev/ppq_step_sizes-1-5-conv-sobolev.pt"

PPQ_STEPS_PATH_ORIG = os.path.join(STEP_DIR, PPQ_STEPS_FILE_ORIG)
PPQ_STEPS_PATH_SOB  = os.path.join(STEP_DIR, PPQ_STEPS_FILE_SOB)

# Where to save the comparison JSON
OUT_JSON_PATH = os.path.join(STEP_DIR, "sobolev_ppq_comparison.json")


def main():
    print("=" * 80)
    print("SOBOLEV-PPQ COMPARISON: FP vs PPQ-orig vs PPQ-Sobolev (NS-PwC-T)")
    print("=" * 80)
    print(f"Project root:      {PROJECT_ROOT}")
    print(f"PPQ artifacts (T): {PPQ_DIR_T}")
    print(f"Step dir:          {STEP_DIR}")
    print(f"Original steps :   {PPQ_STEPS_PATH_ORIG}")
    print(f"Sobolev steps  :   {PPQ_STEPS_PATH_SOB}")
    print(f"Quant layers   :   {QUANT_LAYERS_PATH_T}")
    print(f"Model path     :   {MODEL_PATH}")
    print(f"Data path      :   {DATA_PATH}")
    print(f"Dataset name   :   {DATASET_NAME}")

    # -----------------------------------------------------------------
    # 1) Load model & data loaders
    # -----------------------------------------------------------------
    model, device = load_poseidon_model(MODEL_PATH)

    calib_loader, val_loader, calib_iter, val_iter = build_poseidon_loaders(
        dataset_name=DATASET_NAME,
        data_path=DATA_PATH,
        calib_batchsize=8,
        calib_steps=8,
        val_batchsize=16,
        val_steps=50,
    )

    # -----------------------------------------------------------------
    # 2) Evaluate full-precision baseline
    # -----------------------------------------------------------------
    baseline_metrics = evaluate_model(
        model=model,
        loader_iter=val_iter,
        device=device,
        num_batches=50,
        description="Full Precision Baseline (NS-PwC-T)",
    )

    # -----------------------------------------------------------------
    # 3) Quantize layer names for NS-PwC-T
    # -----------------------------------------------------------------
    quantize_names = load_quantize_layers(QUANT_LAYERS_PATH_T)

    # -----------------------------------------------------------------
    # 4) Build & evaluate PPQ (original step sizes)
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Building PPQ Fake-Quant model (ORIGINAL step sizes)…")
    print("=" * 80)

    if not os.path.isfile(PPQ_STEPS_PATH_ORIG):
        raise FileNotFoundError(f"Original PPQ step sizes not found: {PPQ_STEPS_PATH_ORIG}")

    step_sizes_orig = load_ppq_step_sizes(
        path_pt=PPQ_STEPS_PATH_ORIG,
        path_json="__missing__",  # we don't expect JSON here
        device=device,
    )

    model_ppq_orig = make_quantized_copy(
        model=model,
        step_sizes=step_sizes_orig,
        quantize_names=quantize_names,
        device=device,
    )

    ppq_orig_metrics = evaluate_model(
        model=model_ppq_orig,
        loader_iter=val_iter,
        device=device,
        num_batches=50,
        description="PPQ Fake-Quant (ORIGINAL, NS-PwC-T)",
    )

    # -----------------------------------------------------------------
    # 5) Build & evaluate PPQ (Sobolev step sizes)
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Building PPQ Fake-Quant model (SOBOLEV step sizes)…")
    print("=" * 80)

    if not os.path.isfile(PPQ_STEPS_PATH_SOB):
        raise FileNotFoundError(f"Sobolev PPQ step sizes not found: {PPQ_STEPS_PATH_SOB}")

    step_sizes_sob = load_ppq_step_sizes(
        path_pt=PPQ_STEPS_PATH_SOB,
        path_json="__missing__",
        device=device,
    )

    model_ppq_sob = make_quantized_copy(
        model=model,
        step_sizes=step_sizes_sob,
        quantize_names=quantize_names,
        device=device,
    )

    ppq_sob_metrics = evaluate_model(
        model=model_ppq_sob,
        loader_iter=val_iter,
        device=device,
        num_batches=50,
        description="PPQ Fake-Quant (SOBOLEV, NS-PwC-T)",
    )

    # -----------------------------------------------------------------
    # 6) Compare key metrics: L1/L2 + divergence_ratio
    # -----------------------------------------------------------------
    keys_to_compare = [
        "l1_mean",
        "relative_l1_mean",
        "l2_mean",
        "relative_l2_mean",
        "divergence_ratio",
    ]

    print("\n" + "=" * 80)
    print("COMPARISON: FP vs PPQ-orig vs PPQ-Sobolev (NS-PwC-T)")
    print("=" * 80)

    comparison = {}

    for k in keys_to_compare:
        base_val = float(baseline_metrics[k])
        orig_val = float(ppq_orig_metrics[k])
        sob_val  = float(ppq_sob_metrics[k])

        orig_diff = orig_val - base_val
        sob_diff  = sob_val  - base_val

        orig_ratio = orig_val / (base_val + 1e-12)
        sob_ratio  = sob_val  / (base_val + 1e-12)

        comparison[k] = {
            "baseline": base_val,
            "ppq_orig": orig_val,
            "ppq_sobolev": sob_val,
            "ppq_orig_diff": orig_diff,
            "ppq_orig_ratio": orig_ratio,
            "ppq_sobolev_diff": sob_diff,
            "ppq_sobolev_ratio": sob_ratio,
        }

        print(f"\nMetric: {k}")
        print(f"  Baseline        : {base_val:.6f}")
        print(f"  PPQ (orig)      : {orig_val:.6f}  "
              f"(diff {orig_diff:+.6f}, ratio {orig_ratio:.4f}x)")
        print(f"  PPQ (Sobolev)   : {sob_val:.6f}  "
              f"(diff {sob_diff:+.6f}, ratio {sob_ratio:.4f}x)")

    print("\n" + "=" * 80)
    print("SUMMARY (ratio vs baseline):")
    for k in keys_to_compare:
        r_orig = comparison[k]["ppq_orig_ratio"]
        r_sob  = comparison[k]["ppq_sobolev_ratio"]
        print(f"  {k:23s}: PPQ-orig={r_orig:.4f}x, PPQ-Sobolev={r_sob:.4f}x")
    print("=" * 80 + "\n")

    # -----------------------------------------------------------------
    # 7) Save all metrics + comparison to JSON
    # -----------------------------------------------------------------
    baseline_metrics_json = {
        key: float(value) if isinstance(value, (np.floating, np.integer)) else value
        for key, value in baseline_metrics.items()
    }
    ppq_orig_metrics_json = {
        key: float(value) if isinstance(value, (np.floating, np.integer)) else value
        for key, value in ppq_orig_metrics.items()
    }
    ppq_sob_metrics_json = {
        key: float(value) if isinstance(value, (np.floating, np.integer)) else value
        for key, value in ppq_sob_metrics.items()
    }

    out = {
        "baseline_metrics": baseline_metrics_json,
        "ppq_original_metrics": ppq_orig_metrics_json,
        "ppq_sobolev_metrics": ppq_sob_metrics_json,
        "comparison_vs_baseline": comparison,
    }

    os.makedirs(STEP_DIR, exist_ok=True)
    with open(OUT_JSON_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n✓ Comparison JSON saved to: {OUT_JSON_PATH}")
    print("Done ✔")


if __name__ == "__main__":
    main()
