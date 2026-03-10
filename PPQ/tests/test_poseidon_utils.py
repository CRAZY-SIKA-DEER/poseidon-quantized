import numpy as np

from PPQ.config import PPQConfig
from PPQ.poseidon_utils import (
    load_poseidon_model,
    build_poseidon_loaders,
    poseidon_forward,
)
from scOT.metrics import relative_lp_error, lp_error


def test_poseidon_load_and_infer_once():
    cfg = PPQConfig()

    model, device = load_poseidon_model(cfg.model_path, cfg.device)
    _, _, calib_iter, _ = build_poseidon_loaders(
        dataset_name=cfg.dataset_name,
        data_path=cfg.data_path,
        calib_batchsize=cfg.calib_batchsize,
        calib_steps=1,
        val_batchsize=cfg.val_batchsize,
        val_steps=1,
    )

    batch = next(calib_iter())
    pred = poseidon_forward(model, batch, device)
    y = batch["labels"].to(device)

    pred_np = pred.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    rel_l1 = float(np.mean(relative_lp_error(pred_np, y_np, p=1, return_percent=True)))
    l1 = float(np.mean(lp_error(pred_np, y_np, p=1)))

    print("pred shape:", pred.shape)
    print("label shape:", y.shape)
    print("L1:", l1)
    print("RelL1:", rel_l1)

    assert pred.shape == y.shape
    assert np.isfinite(l1)
    assert np.isfinite(rel_l1)