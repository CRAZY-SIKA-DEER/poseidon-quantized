import numpy as np
import torch
import torch.nn as nn

from PPQ.config import PPQConfig
from PPQ.poseidon_utils import (
    load_poseidon_model,
    build_poseidon_loaders,
    poseidon_forward,
    get_clean_outputs_poseidon,
    get_clean_network_outputs_poseidon,
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


def test_clean_outputs_poseidon():
    cfg = PPQConfig()

    model, device = load_poseidon_model(cfg.model_path, cfg.device)
    _, _, calib_iter, _ = build_poseidon_loaders(
        dataset_name=cfg.dataset_name,
        data_path=cfg.data_path,
        calib_batchsize=cfg.calib_batchsize,
        calib_steps=2,
        val_batchsize=cfg.val_batchsize,
        val_steps=1,
    )

    name2mod = dict(model.named_modules())
    layer_names = [
        name for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
    ][:5]

    clean_inputs, clean_outputs = get_clean_outputs_poseidon(
        model=model,
        dataloader=calib_iter,
        device=device,
        layer_names=layer_names,
    )

    print("num target layers:", len(layer_names))

    for name in layer_names:
        print(f"\nLayer: {name}")
        print("  num input batches :", len(clean_inputs[name]))
        print("  num output batches:", len(clean_outputs[name]))

        assert len(clean_inputs[name]) == 2
        assert len(clean_outputs[name]) == 2

        first_x = clean_inputs[name][0]
        first_y = clean_outputs[name][0]

        if first_x is not None and first_y is not None:
            print("  X shape:", tuple(first_x.shape))
            print("  Y shape:", tuple(first_y.shape))
            assert first_x.shape[-1] == name2mod[name].in_features
            assert first_y.shape[-1] == name2mod[name].out_features


def test_clean_network_outputs_poseidon():
    cfg = PPQConfig()

    model, device = load_poseidon_model(cfg.model_path, cfg.device)
    _, _, calib_iter, _ = build_poseidon_loaders(
        dataset_name=cfg.dataset_name,
        data_path=cfg.data_path,
        calib_batchsize=cfg.calib_batchsize,
        calib_steps=2,
        val_batchsize=cfg.val_batchsize,
        val_steps=1,
    )

    frozen_batches = list(calib_iter())
    clean_net_outputs = get_clean_network_outputs_poseidon(
        model=model,
        frozen_batches=frozen_batches,
        device=device,
    )

    print("num frozen batches:", len(frozen_batches))
    print("num cached outputs:", len(clean_net_outputs))

    assert len(clean_net_outputs) == len(frozen_batches)

    for i, out in enumerate(clean_net_outputs):
        y = frozen_batches[i]["labels"]
        print(f"batch {i} output shape:", tuple(out.shape))
        print(f"batch {i} label  shape:", tuple(y.shape))
        assert tuple(out.shape) == tuple(y.shape)
        assert torch.isfinite(out).all()