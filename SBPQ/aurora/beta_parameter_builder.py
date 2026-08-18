"""Build Beta prior parameters for Aurora blocks."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import torch

from SBPQ.poseidon.beta_parameter_builder import build_poseidon_beta_parameters


def build_aurora_beta_parameters(
    sensitivity: Mapping[str, torch.Tensor],
    minimum_bits: float,
    maximum_bits: float,
    reference_bits: float,
    delta_bits: float,
    beta_kappa: float,
    block_parameter_counts: Mapping[str, torch.Tensor] | None = None,
    mean_epsilon: float = 1e-4,
    relative_epsilon: float = 1e-12,
) -> OrderedDict[str, dict[str, torch.Tensor]]:
    return build_poseidon_beta_parameters(
        sensitivity=sensitivity,
        minimum_bits=minimum_bits,
        maximum_bits=maximum_bits,
        reference_bits=reference_bits,
        delta_bits=delta_bits,
        beta_kappa=beta_kappa,
        block_parameter_counts=block_parameter_counts,
        mean_epsilon=mean_epsilon,
        relative_epsilon=relative_epsilon,
    )


def save_aurora_beta_parameters(
    beta_parameters: Mapping[str, Mapping[str, torch.Tensor]],
    save_path: str | Path,
    metadata: dict | None = None,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = OrderedDict()
    for block_name, parameters in beta_parameters.items():
        serializable[block_name] = {
            key: torch.as_tensor(value).detach().cpu()
            for key, value in parameters.items()
        }
    torch.save(
        {
            "block_beta_parameters": serializable,
            "metadata": metadata or {},
        },
        save_path,
    )

