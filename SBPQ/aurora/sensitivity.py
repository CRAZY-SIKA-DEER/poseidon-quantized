"""Sobolev/gradient-aware sensitivity collection for Aurora blocks."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from SBPQ.aurora.data_utils import batch_mse, move_batch_to_device


def _module_output_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for value in output:
            tensor = _module_output_tensor(value)
            if tensor is not None:
                return tensor
    if isinstance(output, dict):
        for value in output.values():
            tensor = _module_output_tensor(value)
            if tensor is not None:
                return tensor
    return None


def block_root_names(block_mapping: OrderedDict[str, list[str]]) -> OrderedDict[str, str]:
    roots = OrderedDict()
    for block_name, layers in block_mapping.items():
        first = layers[0]
        if first.startswith("backbone.encoder_layers."):
            parts = first.split(".")
            root = ".".join(parts[:3])
        elif first.startswith("backbone.decoder_layers."):
            parts = first.split(".")
            root = ".".join(parts[:3])
        elif first.startswith("encoder.level_agg."):
            root = "encoder.level_agg"
        elif first.startswith("encoder."):
            root = "encoder"
        elif first.startswith("backbone.time_mlp."):
            root = "backbone.time_mlp"
        elif first.startswith("decoder."):
            root = "decoder"
        else:
            root = ".".join(first.split(".")[:-1])
        roots[block_name] = root
    return roots


def proxy_weight_sensitivity(
    model: nn.Module,
    block_mapping: OrderedDict[str, list[str]],
) -> OrderedDict[str, torch.Tensor]:
    modules = dict(model.named_modules())
    sensitivity = OrderedDict()
    for block_name, layers in block_mapping.items():
        values = []
        for layer_name in layers:
            weight = modules[layer_name].weight.detach().float()
            values.append(weight.pow(2).mean())
        sensitivity[block_name] = torch.stack(values).mean().cpu()
    return sensitivity


def compute_aurora_block_sensitivity(
    model: nn.Module,
    batch,
    target,
    block_mapping: OrderedDict[str, list[str]],
    device: str | torch.device = "cuda",
) -> OrderedDict[str, torch.Tensor]:
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    roots = block_root_names(block_mapping)
    modules = dict(model.named_modules())
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(block_name):
        def hook(module, inputs, output):
            tensor = _module_output_tensor(output)
            if tensor is not None and tensor.is_floating_point():
                tensor.retain_grad()
                captured[block_name] = tensor
        return hook

    try:
        for block_name, root_name in roots.items():
            module = modules.get(root_name)
            if module is not None:
                handles.append(module.register_forward_hook(make_hook(block_name)))

        model.zero_grad(set_to_none=True)
        prediction = model(move_batch_to_device(batch, device))
        loss = batch_mse(prediction, move_batch_to_device(target, device))
        loss.backward()

        sensitivity = OrderedDict()
        for block_name in block_mapping:
            tensor = captured.get(block_name)
            if tensor is None or tensor.grad is None:
                sensitivity[block_name] = torch.tensor(0.0)
            else:
                sensitivity[block_name] = tensor.grad.detach().float().pow(2).mean().cpu()
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)

    return sensitivity


def save_block_sensitivity(
    sensitivity: OrderedDict[str, torch.Tensor],
    path,
    metadata: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "raw_sensitivity": OrderedDict(
                (k, torch.as_tensor(v).detach().cpu()) for k, v in sensitivity.items()
            ),
            "metadata": metadata or {},
        },
        path,
    )

