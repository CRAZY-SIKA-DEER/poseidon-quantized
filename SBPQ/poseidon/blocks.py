"""
Poseidon block utilities for SBPQ.

This module:

1. Finds Poseidon structural blocks:
   - ScOTLayer
   - ConvNeXtBlock
   - ResNetBlock

2. Finds quantizable Linear layers.

3. Maps each Linear layer to the nearest structural block that contains it.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from scOT.model import ScOTLayer, ConvNeXtBlock, ResNetBlock


POSEIDON_BLOCK_TYPES = (
    ScOTLayer,
    ConvNeXtBlock,
    ResNetBlock,
)


def find_poseidon_blocks(
    model: nn.Module,
) -> "OrderedDict[str, nn.Module]":
    """
    Find all structural blocks in the Poseidon model.

    Returns:
        {
            block_name: block_module
        }
    """
    blocks = OrderedDict()

    for name, module in model.named_modules():
        if isinstance(module, POSEIDON_BLOCK_TYPES):
            blocks[name] = module

    return blocks


def find_linear_layers(
    model: nn.Module,
) -> "OrderedDict[str, nn.Linear]":
    """
    Find all Linear layers in the Poseidon model.

    Returns:
        {
            layer_name: linear_module
        }
    """
    linear_layers = OrderedDict()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers[name] = module

    return linear_layers


def find_nearest_parent_block(
    layer_name: str,
    block_names: List[str],
) -> Optional[str]:
    """
    Find the nearest structural block containing a layer.

    Example:

        block:
            encoder.stages.0.blocks.1

        layer:
            encoder.stages.0.blocks.1.attention.self.query

    The layer belongs to the block because its name begins with
    the block name.

    If several block names match, the longest name is selected,
    because it is the nearest parent block.
    """
    matching_blocks = []

    for block_name in block_names:
        belongs_to_block = (
            layer_name == block_name
            or layer_name.startswith(block_name + ".")
        )

        if belongs_to_block:
            matching_blocks.append(block_name)

    if not matching_blocks:
        return None

    return max(matching_blocks, key=len)


def build_poseidon_block_mapping(
    model: nn.Module,
) -> Tuple[
    "OrderedDict[str, nn.Module]",
    Dict[str, List[str]],
    Dict[str, str],
    List[str],
]:
    """
    Build the mapping between Poseidon blocks and Linear layers.

    Returns:
        blocks:
            {
                block_name: block_module
            }

        block_to_layers:
            {
                block_name: [
                    linear_layer_name_1,
                    linear_layer_name_2,
                ]
            }

        layer_to_block:
            {
                linear_layer_name: block_name
            }

        unassigned_layers:
            Linear layers outside all detected structural blocks.
    """
    blocks = find_poseidon_blocks(model)
    linear_layers = find_linear_layers(model)

    block_names = list(blocks.keys())

    block_to_layers = {
        block_name: []
        for block_name in block_names
    }

    layer_to_block = {}
    unassigned_layers = []

    for layer_name in linear_layers.keys():
        parent_block = find_nearest_parent_block(
            layer_name=layer_name,
            block_names=block_names,
        )

        if parent_block is None:
            unassigned_layers.append(layer_name)
            continue

        block_to_layers[parent_block].append(layer_name)
        layer_to_block[layer_name] = parent_block

    return (
        blocks,
        block_to_layers,
        layer_to_block,
        unassigned_layers,
    )


def count_linear_weight_parameters(
    linear_layer: nn.Linear,
) -> int:
    """
    Count quantized weights controlled by one Linear layer.
    """
    return int(linear_layer.weight.numel())


def compute_block_parameter_counts(
    model: nn.Module,
    block_to_layers: Dict[str, List[str]],
    layer_names: List[str] | Tuple[str, ...] | set[str] | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Count the number of quantized Linear weights in every structural block.

    The Beta-prior centering in the paper uses n_b so that preferred
    bitwidth shifts are balanced by parameter count, not by block count.
    """
    name_to_module = dict(model.named_modules())
    allowed_layers = None if layer_names is None else set(layer_names)
    counts: Dict[str, torch.Tensor] = {}

    for block_name, layers in block_to_layers.items():
        count = 0

        for layer_name in layers:
            if allowed_layers is not None and layer_name not in allowed_layers:
                continue

            module = name_to_module.get(layer_name)
            if not isinstance(module, nn.Linear):
                continue

            count += count_linear_weight_parameters(module)

        counts[block_name] = torch.tensor(
            float(count),
            dtype=torch.float32,
        )

    return counts


def print_poseidon_block_summary(
    blocks: "OrderedDict[str, nn.Module]",
    block_to_layers: Dict[str, List[str]],
    unassigned_layers: List[str],
) -> None:
    """
    Print the detected block structure for verification.
    """
    print(f"[INFO] Found {len(blocks)} Poseidon structural blocks.")

    for block_name, block_module in blocks.items():
        assigned_layers = block_to_layers.get(block_name, [])

        print(
            f"[BLOCK] {block_name} "
            f"| type={type(block_module).__name__} "
            f"| linear_layers={len(assigned_layers)}"
        )

        for layer_name in assigned_layers:
            print(f"    - {layer_name}")

    print(
        f"[INFO] Found {len(unassigned_layers)} Linear layers "
        "outside the detected structural blocks."
    )

    for layer_name in unassigned_layers:
        print(f"    - {layer_name}")
