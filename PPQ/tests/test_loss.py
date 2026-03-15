import torch
import torch.nn as nn

from PPQ.config import PPQConfig
from PPQ.poseidon_utils import (
    load_poseidon_model,
    build_poseidon_loaders,
    get_clean_network_outputs_poseidon,
    get_clean_outputs_poseidon
)
from PPQ.ranges import compute_data_ranges_poseidon
from PPQ.loss import (
    compute_mdl_prior,
    prior_weighted_avg_bits_cap_poseidon,
    compute_mc_loss_with_prior,
    compute_mc_loss_with_prior_layerwise,
)


def test_compute_mdl_prior_runs_and_backprops():
    w1 = torch.tensor([2.0, 4.0, 8.0])
    a1 = torch.tensor([1.0, 3.0, 5.0])
    w2 = torch.tensor([4.0, 8.0])
    a2 = torch.tensor([2.0, 10.0])

    w1_step = torch.tensor([0.5, 1.0, 2.0], requires_grad=True)
    a1_step = torch.tensor([0.2, 1.0, 2.0], requires_grad=True)
    w2_step = torch.tensor([1.0, 3.0], requires_grad=True)
    a2_step = torch.tensor([0.5, 1.0], requires_grad=True)

    ranges_dict = {
        "layer1": {"weight_ranges": w1, "activation_ranges": a1},
        "layer2": {"weight_ranges": w2, "activation_ranges": a2},
    }
    step_sizes_dict = {
        "layer1": (w1_step, a1_step),
        "layer2": (w2_step, a2_step),
    }

    prior = compute_mdl_prior(step_sizes_dict, ranges_dict, gamma=0.01)
    print("prior:", prior.item())

    assert prior.ndim == 0
    assert torch.isfinite(prior)

    prior.backward()

    assert w1_step.grad is not None
    assert w2_step.grad is not None
    assert a1_step.grad is None
    assert a2_step.grad is None


def test_prior_weighted_avg_bits_cap_poseidon_runs():
    step_sizes_dict = {
        "layer1": (
            torch.tensor([0.5, 1.0, 2.0], requires_grad=True),
            torch.tensor([0.1, 0.1, 0.1], requires_grad=True),
        ),
        "layer2": (
            torch.tensor([1.0, 3.0], requires_grad=True),
            torch.tensor([0.1, 0.1], requires_grad=True),
        ),
    }

    ranges_dict = {
        "layer1": {"weight_ranges": torch.tensor([2.0, 4.0, 8.0])},
        "layer2": {"weight_ranges": torch.tensor([4.0, 8.0])},
    }

    channel_weights = {
        "layer1": torch.tensor([10.0, 10.0, 10.0]),
        "layer2": torch.tensor([20.0, 20.0]),
    }

    penalty = prior_weighted_avg_bits_cap_poseidon(
        step_sizes_dict=step_sizes_dict,
        ranges_dict=ranges_dict,
        channel_weights=channel_weights,
        target_bits=4.0,
        lam=1.0,
        alpha=10.0,
    )

    print("penalty:", penalty.item())

    assert penalty.ndim == 0
    assert torch.isfinite(penalty)

    penalty.backward()

    assert step_sizes_dict["layer1"][0].grad is not None
    assert step_sizes_dict["layer2"][0].grad is not None

def test_mc_loss_with_prior_real_poseidon():
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

    layer_names = [
        name for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
    ][:5]

    ranges_dict = compute_data_ranges_poseidon(
        model=model,
        dataloader=frozen_batches,
        device=device,
        layer_names=layer_names,
        percentile_prob=cfg.percentile_prob,
    )

    clean_net_outputs = get_clean_network_outputs_poseidon(
        model=model,
        frozen_batches=frozen_batches,
        device=device,
    )

    step_sizes_dict = {}
    for name in layer_names:
        if name not in ranges_dict:
            continue

        w_range = ranges_dict[name]["weight_ranges"].to(device)
        a_range = ranges_dict[name]["activation_ranges"].to(device)

        w_step = (w_range / (2 ** cfg.init_bits)).clone().detach().requires_grad_(True)
        a_step = (a_range / (2 ** cfg.init_bits)).clone().detach().requires_grad_(True)

        step_sizes_dict[name] = (w_step, a_step)

    total_loss, likelihood_loss, prior_loss = compute_mc_loss_with_prior(
        model=model,
        step_sizes_dict=step_sizes_dict,
        frozen_batches=frozen_batches,
        clean_net_outputs=clean_net_outputs,
        ranges_dict=ranges_dict,
        batch_idx=0,
        num_mc_samples=2,
        eta=cfg.eta,
        gamma=0.0,
        device=device,
    )

    print("total_loss:", float(total_loss))
    print("likelihood_loss:", float(likelihood_loss))
    print("prior_loss:", float(prior_loss))

    assert total_loss.ndim == 0
    assert likelihood_loss.ndim == 0
    assert prior_loss.ndim == 0
    assert torch.isfinite(total_loss)
    assert torch.isfinite(likelihood_loss)
    assert torch.isfinite(prior_loss)

    total_loss.backward()

    first_name = next(iter(step_sizes_dict))
    assert step_sizes_dict[first_name][0].grad is not None

def test_mc_loss_with_prior_layerwise_real_poseidon():
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

    layer_names = [
        name for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
    ][:5]

    ranges_dict = compute_data_ranges_poseidon(
        model=model,
        dataloader=frozen_batches,
        device=device,
        layer_names=layer_names,
        percentile_prob=cfg.percentile_prob,
    )

    clean_inputs, clean_outputs = get_clean_outputs_poseidon(
        model=model,
        dataloader=frozen_batches,
        device=device,
        layer_names=layer_names,
    )

    step_sizes_dict = {}
    for name in layer_names:
        if name not in ranges_dict:
            continue
        if clean_inputs[name][0] is None or clean_outputs[name][0] is None:
            continue

        w_range = ranges_dict[name]["weight_ranges"].to(device)
        a_range = ranges_dict[name]["activation_ranges"].to(device)

        w_step = (w_range / (2 ** cfg.init_bits)).clone().detach().requires_grad_(True)
        a_step = (a_range / (2 ** cfg.init_bits)).clone().detach().requires_grad_(True)
        step_sizes_dict[name] = (w_step, a_step)

    total_loss, likelihood_loss, prior_loss = compute_mc_loss_with_prior_layerwise(
        model=model,
        step_sizes_dict=step_sizes_dict,
        clean_inputs=clean_inputs,
        clean_outputs=clean_outputs,
        ranges_dict=ranges_dict,
        batch_idx=0,
        num_mc_samples=2,
        eta=cfg.eta,
        gamma=0.0,
        device=device,
    )

    print("layerwise total_loss:", float(total_loss))
    print("layerwise likelihood_loss:", float(likelihood_loss))
    print("layerwise prior_loss:", float(prior_loss))

    assert total_loss.ndim == 0
    assert torch.isfinite(total_loss)

    total_loss.backward()

    first_name = next(iter(step_sizes_dict))
    assert step_sizes_dict[first_name][0].grad is not None