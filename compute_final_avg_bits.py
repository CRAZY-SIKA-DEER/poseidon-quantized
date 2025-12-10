import os
import json
import torch
import torch.nn as nn

# ---------------------------------------------------------------------
# Import helpers from your PPQ_weight_only.py
# (adjust the import path if this file is in a different folder)
# ---------------------------------------------------------------------
from PPQ.PPQ_weight_only import (
    load_poseidon_model,
    build_poseidon_loaders,
    compute_data_ranges_poseidon,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR 
INSPECT_DIR = os.path.join(PROJECT_ROOT, "inspect_layers")

# Path to your saved PPQ step sizes (JSON)
PPQ_JSON_PATH = os.path.join(
    PROJECT_ROOT,
    "ppq_artifacts",
    "NS-PwC-T",
    "ppq_step_sizes.json",   # <-- change if filename differs
)

MODEL_PATH   = "models/NS-PwC-T"
DATA_PATH    = "dataset/NS-PwC"
DATASET_NAME = "fluids.incompressible.PiecewiseConstants"
DEVICE_STR   = "cuda"
CALIB_BATCHSIZE = 2
CALIB_STEPS     = 64   # same as you used in main() ideally


def load_ppq_step_sizes_from_json(json_path: str, device: torch.device):
    """
    Load weight step sizes from the PPQ JSON file.

    JSON structure (per layer):

        "layer_name": [
          [ w_step_0, w_step_1, ... ],   # weights
          [ a_step_0, a_step_1, ... ]    # activations (ignored here)
        ]
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    step_sizes = {}
    for name, pair in data["step_sizes"].items():
        w_list = pair[0]  # first list = weight step sizes
        # we ignore the second list (activation steps) for weight-only bits

        w_tensor = torch.tensor(w_list, dtype=torch.float32, device=device)
        step_sizes[name] = w_tensor

    meta = data.get("meta", {})
    return step_sizes, meta


def compute_weighted_avg_bits(
    model: nn.Module,
    weight_steps: dict[str, torch.Tensor],
    ranges_dict: dict[str, dict[str, torch.Tensor]],
    eps: float = 1e-8,
) -> float:
    """
    Compute parameter-weighted average bit-width for weights only.

    For each linear layer and each output channel k:

        bits_k = log2( R_w_k / S_k )

    where:
        R_w_k = weight_ranges[name][k]
        S_k   = weight_steps[name][k]

    Each output channel k corresponds to 'in_features' parameters
    (one per input feature). So its contribution is:

        contribution_k = bits_k * in_features

    Global average:
        avg_bits = (sum over all layers, all channels of contribution_k)
                   / (total number of weight parameters in those layers)
    """
    name2mod = dict(model.named_modules())

    total_bits_times_params = 0.0
    total_params = 0

    for name, w_step in weight_steps.items():
        if name not in ranges_dict:
            continue
        if name not in name2mod:
            continue

        mod = name2mod[name]
        if not isinstance(mod, nn.Linear):
            continue

        # weight_ranges: [out_features]
        w_range = ranges_dict[name].get("weight_ranges", None)
        if w_range is None:
            continue

        w_range = w_range.to(w_step.device)

        if w_range.numel() != w_step.numel():
            print(f"[WARN] {name}: mismatch w_range({w_range.numel()}) vs w_step({w_step.numel()}); skipping.")
            continue

        # bits per output channel
        bits = torch.log2((w_range + eps) / (w_step + eps))

        # number of parameters per output channel
        in_features = mod.weight.shape[1]
        params_per_channel = in_features

        # contribution of this layer to global average
        layer_bits_times_params = (bits * params_per_channel).sum().item()
        layer_params = bits.numel() * params_per_channel

        total_bits_times_params += layer_bits_times_params
        total_params += layer_params

    if total_params == 0:
        return float("nan")

    avg_bits = total_bits_times_params / float(total_params)
    return avg_bits


def main():
    # ---------------------------------------------------------
    # 1) Load model + device
    # ---------------------------------------------------------
    model, device = load_poseidon_model(MODEL_PATH, device=DEVICE_STR)
    name2mod = dict(model.named_modules())

    # ---------------------------------------------------------
    # 2) Load final PPQ step sizes from JSON
    # ---------------------------------------------------------
    weight_steps, meta = load_ppq_step_sizes_from_json(PPQ_JSON_PATH, device)
    print(f"[INFO] Loaded step sizes for {len(weight_steps)} layers from {PPQ_JSON_PATH}")

    percentile_prob = meta.get("percentile_prob", 1e-4)
    print(f"[INFO] Using percentile_prob={percentile_prob} to recompute ranges.")

    # ---------------------------------------------------------
    # 3) Build calibration iterator to recompute ranges_dict
    #    (same way as in your PPQ script)
    # ---------------------------------------------------------
    _, _, calib_iter, _ = build_poseidon_loaders(
        dataset_name=DATASET_NAME,
        data_path=DATA_PATH,
        calib_batchsize=CALIB_BATCHSIZE,
        calib_steps=CALIB_STEPS,
        val_batchsize=16,
        val_steps=50,
    )

    # We only need Linear layers that appear in weight_steps
    target_layers = [n for n in weight_steps.keys() if isinstance(name2mod.get(n, None), nn.Linear)]
    print(f"[INFO] Recomputing ranges for {len(target_layers)} Linear layers.")

    ranges_dict = compute_data_ranges_poseidon(
        model=model,
        dataloader=calib_iter,
        device=device,
        layer_names=target_layers,
        percentile_prob=percentile_prob,
    )

    # ---------------------------------------------------------
    # 4) Compute parameter-weighted average bit-width
    # ---------------------------------------------------------
    avg_bits = compute_weighted_avg_bits(
        model=model,
        weight_steps=weight_steps,
        ranges_dict=ranges_dict,
    )

    print("\n==================== FINAL RESULT ====================")
    print(f"Parameter-weighted average weight bit-width: {avg_bits:.4f} bits")
    print("======================================================\n")


if __name__ == "__main__":
    main()
