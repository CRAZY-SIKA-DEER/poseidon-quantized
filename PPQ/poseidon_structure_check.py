from __future__ import annotations
import numpy as np
import sys

import json
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from scOT.metrics import relative_lp_error, lp_error

from PPQ.config import PPQConfig
from PPQ.poseidon_utils import load_poseidon_model, build_poseidon_loaders
from PPQ.trainer import PPQTrainer
from PPQ.metrics import (
    evaluate_with_stepsizes,
    compute_avg_bits,
    compute_dynamic_stepsizes,
)


def main():
    cfg = PPQConfig()

    # --------------------------------------------------
    # 1) Load model
    # --------------------------------------------------
    model, device = load_poseidon_model(cfg.model_path, cfg.device)

    print(model.config)
    print("residual_model:", model.config.residual_model)
    print("depths:", model.config.depths)
    print("embed_dim:", model.config.embed_dim)
    print("num_heads:", model.config.num_heads)
    print("skip_connections:", model.config.skip_connections)

    # --------------------------------------------------
    # 2) Build data loaders
    # --------------------------------------------------
    _calib_loader, _val_loader, calib_iter, val_iter = build_poseidon_loaders(
        dataset_name=cfg.dataset_name,
        data_path=cfg.data_path,
        calib_batchsize=cfg.calib_batchsize,
        calib_steps=cfg.calib_steps,
        val_batchsize=cfg.val_batchsize,
        val_steps=cfg.val_steps,
    )



if __name__ == "__main__":
    main()