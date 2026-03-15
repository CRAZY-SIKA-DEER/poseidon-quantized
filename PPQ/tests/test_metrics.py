import torch
import torch.nn as nn

from PPQ.config import PPQConfig
from PPQ.poseidon_utils import load_poseidon_model, build_poseidon_loaders
from PPQ.ranges import compute_data_ranges_poseidon
from PPQ.metrics import (
    build_channel_param_weights,
    compute_avg_bits,
    compute_dynamic_stepsizes,
    evaluate_with_stepsizes,
)


def test_build_channel_param_weights_real_poseidon():
    cfg = PPQConfig()
    model, _device = load_poseidon_model(cfg.model_path, cfg.device)

    layer_names = [
        name for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
    ][:5]

    weights = build_channel_param_weights(model, layer_names)

    assert len(weights) > 0
    for name in layer_names:
        assert name in weights
        assert weights[name].ndim == 1
        assert torch.isfinite(weights[name]).all()


def test_compute_avg_bits_runs():
    step_sizes_dict = {
        "layer1": (torch.tensor([1.0, 2.0]), None)
    }
    ranges_dict = {
        "layer1": {"weight_ranges": torch.tensor([4.0, 8.0])}
    }

    avg_bits = compute_avg_bits(step_sizes_dict, ranges_dict)
    print("avg_bits:", avg_bits)

    assert isinstance(avg_bits, float)


def test_compute_dynamic_stepsizes_real_poseidon():
    cfg = PPQConfig()
    model, device = load_poseidon_model(cfg.model_path, cfg.device)

    layer_names = [
        name for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
    ][:5]

    dyn_steps = compute_dynamic_stepsizes(
        model=model,
        layer_names=layer_names,
        num_bits=8,
        device=device,
    )

    assert len(dyn_steps) > 0
    for name in layer_names:
        assert name in dyn_steps
        assert dyn_steps[name].ndim == 1
        assert torch.isfinite(dyn_steps[name]).all()
        assert (dyn_steps[name] > 0).all()


def test_evaluate_with_stepsizes_real_poseidon():
    cfg = PPQConfig()

    model, device = load_poseidon_model(cfg.model_path, cfg.device)
    _, _, calib_iter, val_iter = build_poseidon_loaders(
        dataset_name=cfg.dataset_name,
        data_path=cfg.data_path,
        calib_batchsize=cfg.calib_batchsize,
        calib_steps=2,
        val_batchsize=cfg.val_batchsize,
        val_steps=2,
    )

    layer_names = [
        name for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
    ][:5]

    ranges_dict = compute_data_ranges_poseidon(
        model=model,
        dataloader=calib_iter,
        device=device,
        layer_names=layer_names,
        percentile_prob=cfg.percentile_prob,
    )

    step_sizes_dict = {}
    for name in layer_names:
        if name not in ranges_dict:
            continue

        w_range = ranges_dict[name]["weight_ranges"].to(device)
        a_range = ranges_dict[name]["activation_ranges"].to(device)

        w_step = (w_range / (2 ** cfg.init_bits)).clone().detach()
        a_step = (a_range / (2 ** cfg.init_bits)).clone().detach()

        step_sizes_dict[name] = (w_step, a_step)

    metrics = evaluate_with_stepsizes(
        model=model,
        val_loader=val_iter,
        weight_steps=step_sizes_dict,
        act_steps=None,
        layer_names=layer_names,
        device=device,
    )

    print("metrics:", metrics)

    assert "l1" in metrics
    assert "rel_l1" in metrics
    assert metrics["l1"] == metrics["l1"]
    assert metrics["rel_l1"] == metrics["rel_l1"]