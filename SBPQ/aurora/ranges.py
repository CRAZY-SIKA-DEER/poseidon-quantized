"""Weight range utilities for Aurora SBPQ."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from SBPQ.poseidon.ranges import percentile_range_from_samples


def compute_weight_ranges_aurora(
    model: nn.Module,
    layer_names: list[str],
    percentile_prob: float = 1e-4,
    minimum_range: float = 1e-12,
) -> OrderedDict[str, dict[str, torch.Tensor]]:
    layer_set = set(layer_names)
    ranges = OrderedDict()
    for name, module in model.named_modules():
        if name not in layer_set:
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Layer {name} is not nn.Linear.")
        weight = module.weight.detach().float().reshape(module.weight.shape[0], -1)
        ranges[name] = {
            "weight_ranges": percentile_range_from_samples(
                weight,
                percentile_prob=percentile_prob,
                minimum_range=minimum_range,
            ).cpu(),
        }
    missing = layer_set - set(ranges)
    if missing:
        raise KeyError(f"Missing Aurora Linear layers: {sorted(missing)}")
    return ranges

