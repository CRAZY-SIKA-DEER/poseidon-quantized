"""Structural block definitions for Aurora."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


def build_aurora_block_mapping(model: nn.Module) -> OrderedDict[str, list[str]]:
    quant_layers = {
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }

    mapping: OrderedDict[str, list[str]] = OrderedDict()
    mapping["encoder_input"] = sorted(
        name for name in quant_layers
        if name.startswith("encoder.") and not name.startswith("encoder.level_agg.")
    )
    mapping["encoder_level_agg"] = sorted(
        name for name in quant_layers
        if name.startswith("encoder.level_agg.")
    )

    for stage_index, stage in enumerate(getattr(model.backbone, "encoder_layers", [])):
        prefix = f"backbone.encoder_layers.{stage_index}."
        mapping[f"backbone_encoder_stage_{stage_index}"] = sorted(
            name for name in quant_layers if name.startswith(prefix)
        )

    for stage_index, stage in enumerate(getattr(model.backbone, "decoder_layers", [])):
        prefix = f"backbone.decoder_layers.{stage_index}."
        mapping[f"backbone_decoder_stage_{stage_index}"] = sorted(
            name for name in quant_layers if name.startswith(prefix)
        )

    mapping["backbone_time"] = sorted(
        name for name in quant_layers if name.startswith("backbone.time_mlp.")
    )
    mapping["decoder_output"] = sorted(
        name for name in quant_layers if name.startswith("decoder.")
    )

    return OrderedDict((block, layers) for block, layers in mapping.items() if layers)


def build_layer_to_block(block_mapping: OrderedDict[str, list[str]]) -> dict[str, str]:
    return {
        layer_name: block_name
        for block_name, layers in block_mapping.items()
        for layer_name in layers
    }


def select_quant_layers(
    block_mapping: OrderedDict[str, list[str]],
    max_quant_layers: int | None = None,
) -> list[str]:
    layers = [
        layer_name
        for block_layers in block_mapping.values()
        for layer_name in block_layers
    ]
    if max_quant_layers is not None:
        layers = layers[: int(max_quant_layers)]
    return layers


def compute_block_parameter_counts(
    model: nn.Module,
    block_mapping: OrderedDict[str, list[str]],
) -> OrderedDict[str, torch.Tensor]:
    modules = dict(model.named_modules())
    counts = OrderedDict()
    for block_name, layer_names in block_mapping.items():
        total = 0
        for layer_name in layer_names:
            layer = modules[layer_name]
            total += int(layer.weight.numel())
        counts[block_name] = torch.tensor(float(total), dtype=torch.float32)
    return counts

