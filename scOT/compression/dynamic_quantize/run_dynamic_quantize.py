#!/usr/bin/env python
import os
# 1) Hide all GPUs from PyTorch so quantized kernels dispatch on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import numpy as np
import random

from scOT.model import ScOT
from scOT.trainer import TrainingArguments, Trainer
from scOT.problems.base import get_dataset
from scOT.metrics import relative_lp_error, lp_error

# ──────────────────────────────────────────────────────────────────────────────
#  Settings
# ──────────────────────────────────────────────────────────────────────────────

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

QUANTIZED_MODEL_PATH = "./quantized_poseidon_t_dynamic.pth"
DATA_PATH            = "./datasets/NS-Sines"
DATASET_NAME         = "fluids.incompressible.Sines"
BATCH_SIZE           = 16

# ──────────────────────────────────────────────────────────────────────────────
#  Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_quantized_model(path: str) -> torch.nn.Module:
    print(f"[INFO] Reconstructing and quantizing Poseidon-T …")

    # 1) Build the original architecture from HF
    model = ScOT.from_pretrained("camlab-ethz/Poseidon-T")
    model.eval()

    # 2) Apply dynamic quantization (CPU‐only)
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # 3) Load your saved quantized weights (state_dict)
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)

    # 4) Ensure model is on CPU
    return model.to("cpu")

# ──────────────────────────────────────────────────────────────────────────────
#  Data loading
# ──────────────────────────────────────────────────────────────────────────────

def get_test_dataset(name: str, data_path: str):
    return get_dataset(
        dataset=name,
        which="test",
        num_trajectories=1,
        data_path=data_path,
        move_to_local_scratch=None
    )

# ──────────────────────────────────────────────────────────────────────────────
#  Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_eval_metrics(preds: np.ndarray, labels: np.ndarray):
    rel_err = relative_lp_error(preds, labels, p=1, return_percent=True)
    l1_err  = lp_error         (preds, labels, p=1)
    return float(np.mean(rel_err)), float(np.mean(l1_err))

# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Load quantized model
    model   = load_quantized_model(QUANTIZED_MODEL_PATH)

    # Load dataset (this auto‐handles the 3→4 channel mismatch)
    dataset = get_test_dataset(DATASET_NAME, DATA_PATH)

    # Build a Trainer that *will stay on CPU*
    args = TrainingArguments(
        output_dir=".",
        per_device_eval_batch_size=BATCH_SIZE,
        dataloader_num_workers=0,
        no_cuda=True,           # <-- reinforce CPU‐only
    )
    trainer = Trainer(model=model, args=args)

    # Run inference
    print("[INFO] Running inference on quantized model…")
    output = trainer.predict(dataset)

    preds  = output.predictions   # shape (N, C*H*W) or (N, C, H, W) depending internally
    labels = output.label_ids

    # Compute and print metrics
    mean_rel, mean_l1 = compute_eval_metrics(preds, labels)
    print("\n=== Quantized Model Evaluation Results ===")
    print(f"Mean Relative L1 Error (%): {mean_rel:.4f}")
    print(f"Mean L1 Error:            {mean_l1:.4f}")

if __name__ == "__main__":
    main()
