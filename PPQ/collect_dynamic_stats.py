#!/usr/bin/env python3
"""
collect_dynamic.py

This script collects per-channel *dynamic* weight step sizes for the NS-PwC-T model.

For each Linear layer in T_quantize_layers.pt, we compute a dynamic
uniform quantization step size for each output channel, using a standard
per-channel symmetric scheme:

    - For each out_channel k:
        w_k = weight[k, :]        # shape [in_features]
        max_abs = max(|w_k|)
        step_k = 2 * max_abs / (2^bits - 1)

These step sizes are then saved to:

    PROJECT_ROOT/dynamic_stats/NS-PwC-T-dynamic-stepsizes.pt
    PROJECT_ROOT/dynamic_stats/NS-PwC-T-dynamic-stepsizes.json

You can later load these step sizes and use them as the initial w_step
in your PPQ optimization, so that PPQ starts exactly from the dynamic
quantization solution.

Adjust MODEL_PATH / LAYER_LIST_PATH / NUM_BITS as needed.
"""

import os
import sys
import json
from typing import Dict, Any

import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# Paths / setup
# --------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # .../main/PPQ or similar
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))   # .../main
INSPECT_DIR = os.path.join(PROJECT_ROOT, "inspect_layers")       # .../inspect_layers
DYNAMIC_DIR = os.path.join(PROJECT_ROOT, "dynamic_stats")        # .../dynamic_stats

os.makedirs(DYNAMIC_DIR, exist_ok=True)

# Make sure we can import scOT from project root
sys.path.insert(0, PROJECT_ROOT)

from scOT.model import ScOT  # type: ignore


# --------------------------------------------------------------------------
# Configuration (edit these if needed)
# --------------------------------------------------------------------------

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "NS-PwC-T")  # adjust if different
LAYER_LIST_PATH = os.path.join(INSPECT_DIR, "T_quantize_layers.pt")
OUTPUT_BASENAME = "NS-PwC-T-dynamic-stepsizes"
NUM_BITS = 4  # dynamic quantization bit-width for weights


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def load_model(model_path: str, device: str = "cpu") -> torch.nn.Module:
    """
    Load the Poseidon ScOT T-model from a given path.
    """
    device = torch.device(device)
    model = ScOT.from_pretrained(model_path).to(device)
    model.eval()
    print(f"Loaded model from '{model_path}' on device {device}")
    return model


def load_quantize_layer_names(path: str) -> Dict[str, Any]:
    """
    Load layer names from T_quantize_layers.pt (or similar).
    Expected format:
        {"quantize_layers": [name1, name2, ...]}
    """
    print(f"Loading quantize layer list from: {path}")
    data = torch.load(path, map_location="cpu")
    if "quantize_layers" not in data:
        raise KeyError(f"'quantize_layers' not found in {path}")
    return data["quantize_layers"]


def compute_per_channel_dynamic_steps(weight: torch.Tensor, num_bits: int) -> torch.Tensor:
    """
    Compute a per-channel symmetric dynamic step size for a Linear weight.

    Args:
        weight: [out_features, in_features] tensor
        num_bits: number of quantization bits

    Returns:
        step_sizes: [out_features] tensor, one step per output channel
    """
    # weight is [out_features, in_features]
    # For each out_channel k: step_k = 2 * max_abs_k / (2^bits - 1)
    with torch.no_grad():
        out_features = weight.size(0)
        w_flat = weight.view(out_features, -1)              # [out_features, in_features]
        max_abs = w_flat.abs().max(dim=1).values            # [out_features]
        denom = (2 ** num_bits) - 1
        step_sizes = (2.0 * max_abs) / max(denom, 1)
    return step_sizes


def collect_dynamic_stepsizes(
    model: nn.Module,
    layer_names,
    num_bits: int = NUM_BITS,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    For each Linear layer in layer_names, compute per-channel dynamic step sizes.

    Returns:
        dynamic_steps: {layer_name: step_sizes[out_features]}
    """
    device = torch.device(device)
    model = model.to(device)
    name2mod = dict(model.named_modules())

    dynamic_steps: Dict[str, torch.Tensor] = {}

    print(f"\nCollecting dynamic per-channel weight step sizes (num_bits={num_bits})...")
    for name in layer_names:
        mod = name2mod.get(name, None)
        if not isinstance(mod, nn.Linear):
            # We only care about Linear layers for weights here
            continue

        w = mod.weight.data.to(device)  # [out_features, in_features]
        step = compute_per_channel_dynamic_steps(w, num_bits=num_bits)  # [out_features]

        dynamic_steps[name] = step.cpu()
        print(f"  - {name}: out_features={w.size(0)}, step_sizes shape={tuple(step.shape)}")

    print(f"\n✓ Collected dynamic step sizes for {len(dynamic_steps)} Linear layers.")
    return dynamic_steps


def save_dynamic_steps(dynamic_steps: Dict[str, torch.Tensor], num_bits: int):
    """
    Save dynamic step sizes to both .pt and .json under DYNAMIC_DIR.
    """
    pt_path = os.path.join(DYNAMIC_DIR, OUTPUT_BASENAME + ".pt")
    json_path = os.path.join(DYNAMIC_DIR, OUTPUT_BASENAME + ".json")

    # Save as .pt (tensors intact)
    save_obj = {
        "num_bits": num_bits,
        "step_sizes": {k: v.clone() for k, v in dynamic_steps.items()},
    }
    torch.save(save_obj, pt_path)
    print(f"✓ Saved dynamic step sizes (tensors) → {pt_path}")

    # Save as .json (tensors → lists)
    json_ready = {
        "num_bits": num_bits,
        "step_sizes": {k: v.tolist() for k, v in dynamic_steps.items()},
    }
    with open(json_path, "w") as f:
        json.dump(json_ready, f, indent=2)
    print(f"✓ Saved dynamic step sizes (JSON) → {json_path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    # 1) Load model
    model = load_model(MODEL_PATH, device="cpu")

    # 2) Load quantize layer names
    quantize_layers = load_quantize_layer_names(LAYER_LIST_PATH)
    print(f"Found {len(quantize_layers)} layers in {os.path.basename(LAYER_LIST_PATH)}")

    # 3) Collect dynamic per-channel weight step sizes for Linear layers
    dynamic_steps = collect_dynamic_stepsizes(
        model=model,
        layer_names=quantize_layers,
        num_bits=NUM_BITS,
        device="cpu",
    )

    # 4) Save to disk
    save_dynamic_steps(dynamic_steps, num_bits=NUM_BITS)

    print("\nDone. Dynamic weight step sizes are ready for PPQ initialization.")


if __name__ == "__main__":
    main()
