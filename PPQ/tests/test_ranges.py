import torch
import torch.nn as nn

from PPQ.config import PPQConfig
from PPQ.poseidon_utils import load_poseidon_model, build_poseidon_loaders
from PPQ.ranges import (
    compute_data_ranges_poseidon,
    evaluate_quantized_model_poseidon,
)


def test_compute_data_ranges_poseidon():
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

    layer_names = [
        name for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
    ][:5]

    ranges = compute_data_ranges_poseidon(
        model=model,
        dataloader=calib_iter,
        device=device,
        layer_names=layer_names,
        percentile_prob=1e-4,
    )

    assert len(ranges) > 0

    for name in layer_names:
        assert name in ranges
        assert "weight_ranges" in ranges[name]
        assert "activation_ranges" in ranges[name]

        w = ranges[name]["weight_ranges"]
        a = ranges[name]["activation_ranges"]

        print(f"{name}: weight={tuple(w.shape)}, act={tuple(a.shape)}")

        assert torch.isfinite(w).all()
        assert torch.isfinite(a).all()
        assert (w > 0).all()
        assert (a > 0).all()


def test_evaluate_quantized_model_poseidon():
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

    layer_names = [
        name for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
    ][:5]

    ranges = compute_data_ranges_poseidon(
        model=model,
        dataloader=calib_iter,
        device=device,
        layer_names=layer_names,
        percentile_prob=1e-4,
    )

    loss = evaluate_quantized_model_poseidon(
        model=model,
        dataloader=calib_iter,
        ranges=ranges,
        device=device,
        layer_names=layer_names,
        num_bits=8,
    )

    print("fake-quant eval loss:", loss)

    assert isinstance(loss, float)
    assert loss == loss
    assert loss >= 0.0