"""
Build and save Poseidon block-wise Beta prior parameters.

This file converts precomputed block sensitivity values into one Beta
distribution for every Poseidon block.

Pipeline:

    block sensitivity r_b
        ↓
    relative sensitivity d_b in [-1, 1]
        ↓
    preferred bitwidth B_pref_b
        ↓
    normalized Beta mean mu_b in (0, 1)
        ↓
    Beta parameters a_b and b_b
        ↓
    save all parameters before optimization

The Beta parameters are model- and dataset-specific. They are calculated
once and loaded later during quantization optimization.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import torch


def load_saved_block_sensitivity(
    sensitivity_path: str | Path,
    sensitivity_key: str = "raw_sensitivity",
) -> OrderedDict[str, torch.Tensor]:
    """
    Load previously saved block-sensitivity values.

    Expected sensitivity file structure:

        {
            "raw_sensitivity": {
                block_name: scalar_tensor,
                ...
            },
            "normalized_sensitivity": {
                block_name: scalar_tensor,
                ...
            },
            "metadata": {...}
        }

    Args:
        sensitivity_path:
            Path to the saved sensitivity .pt file.

        sensitivity_key:
            Which sensitivity dictionary to use.

            Usually use:

                "raw_sensitivity"

    Returns:
        Ordered dictionary:

            {
                block_name: scalar sensitivity tensor
            }
    """
    sensitivity_path = Path(sensitivity_path)

    if not sensitivity_path.exists():
        raise FileNotFoundError(
            f"Sensitivity file was not found: {sensitivity_path}"
        )

    saved_object = torch.load(
        sensitivity_path,
        map_location="cpu",
    )

    if sensitivity_key not in saved_object:
        raise KeyError(
            f"Key '{sensitivity_key}' was not found in "
            f"{sensitivity_path}."
        )

    loaded_sensitivity = saved_object[sensitivity_key]

    if not isinstance(loaded_sensitivity, Mapping):
        raise TypeError(
            f"Expected '{sensitivity_key}' to be a mapping, "
            f"but received {type(loaded_sensitivity)}."
        )

    sensitivity = OrderedDict()

    for block_name, value in loaded_sensitivity.items():
        value = torch.as_tensor(
            value,
            dtype=torch.float32,
        ).reshape(())

        if not torch.isfinite(value):
            raise ValueError(
                f"Non-finite sensitivity found for block "
                f"'{block_name}': {value.item()}."
            )

        sensitivity[block_name] = value

    if len(sensitivity) == 0:
        raise ValueError(
            "The loaded sensitivity dictionary is empty."
        )

    return sensitivity



def normalize_raw_sensitivity(
    sensitivity: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """
    First normalization step.

    Convert raw block sensitivities phi_b into r_b in [0, 1]:

        r_b = (phi_b - min(phi)) /
              (max(phi) - min(phi))

    Float64 is used because block sensitivities can be extremely small.
    """
    if len(sensitivity) == 0:
        raise ValueError(
            "Cannot normalize an empty sensitivity mapping."
        )

    block_names = list(sensitivity.keys())

    values = torch.stack(
        [
            torch.as_tensor(
                sensitivity[block_name],
                dtype=torch.float64,
            ).reshape(())
            for block_name in block_names
        ]
    )

    if not torch.isfinite(values).all():
        raise ValueError(
            "Sensitivity values contain NaN or infinity."
        )

    minimum = values.min()
    maximum = values.max()
    value_range = maximum - minimum

    # Only treat the values as identical when the range is exactly zero.
    if value_range == 0:
        normalized_values = torch.full_like(
            values,
            fill_value=0.5,
        )
    else:
        normalized_values = (
            values - minimum
        ) / value_range

    return OrderedDict(
        (
            block_name,
            normalized_values[index].float(),
        )
        for index, block_name in enumerate(block_names)
    )


def compute_relative_sensitivity(
    sensitivity: Mapping[str, torch.Tensor],
    block_parameter_counts: Mapping[str, torch.Tensor] | None = None,
    epsilon: float = 1e-12,
) -> OrderedDict[str, torch.Tensor]:
    """
    Second normalization step.

    Convert r_b into centered relative sensitivity d_b:

        d_b = (r_b - r_bar_w) /
              (max_j |r_j - r_bar_w| + epsilon)

    where r_bar_w is the parameter-weighted mean when n_b values are
    supplied:

        r_bar_w = sum_b n_b r_b / sum_b n_b

    This produces values in [-1, 1].
    """
    if len(sensitivity) == 0:
        raise ValueError(
            "Cannot calculate relative sensitivity "
            "from an empty mapping."
        )

    block_names = list(sensitivity.keys())

    values = torch.stack(
        [
            torch.as_tensor(
                sensitivity[block_name],
                dtype=torch.float64,
            ).reshape(())
            for block_name in block_names
        ]
    )

    if block_parameter_counts is None:
        weights = torch.ones_like(values)
    else:
        weights = torch.stack(
            [
                torch.as_tensor(
                    block_parameter_counts[block_name],
                    dtype=torch.float64,
                ).reshape(())
                for block_name in block_names
            ]
        )

        if not torch.isfinite(weights).all():
            raise ValueError(
                "Block parameter counts contain NaN or infinity."
            )

        if torch.any(weights < 0):
            raise ValueError(
                "Block parameter counts must be non-negative."
            )

        if weights.sum() <= 0:
            raise ValueError(
                "At least one block must have a positive parameter count."
            )

    weighted_mean_sensitivity = (
        (weights * values).sum() / weights.sum()
    )

    centered_values = values - weighted_mean_sensitivity

    maximum_absolute_deviation = centered_values.abs().max()

    if maximum_absolute_deviation <= 0:
        relative_values = torch.zeros_like(values)
    else:
        relative_values = (
            centered_values
            / (maximum_absolute_deviation + float(epsilon))
        )

    relative_values = relative_values.clamp(
        min=-1.0,
        max=1.0,
    )

    return OrderedDict(
        (
            block_name,
            relative_values[index].float(),
        )
        for index, block_name in enumerate(block_names)
    )


def compute_parameter_weighted_average(
    values: Mapping[str, torch.Tensor],
    block_parameter_counts: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """
    Compute sum_b n_b value_b / sum_b n_b.
    """
    if len(values) == 0:
        raise ValueError("Cannot average an empty value mapping.")

    total_weighted_value = torch.zeros((), dtype=torch.float64)
    total_weight = torch.zeros((), dtype=torch.float64)

    for block_name, value in values.items():
        if block_name not in block_parameter_counts:
            raise KeyError(
                f"Missing parameter count for block '{block_name}'."
            )

        weight = torch.as_tensor(
            block_parameter_counts[block_name],
            dtype=torch.float64,
        ).reshape(())

        if weight < 0:
            raise ValueError(
                f"Block '{block_name}' has a negative parameter count."
            )

        total_weighted_value = total_weighted_value + weight * torch.as_tensor(
            value,
            dtype=torch.float64,
        ).reshape(())
        total_weight = total_weight + weight

    if total_weight <= 0:
        raise ValueError(
            "At least one block must have a positive parameter count."
        )

    return (total_weighted_value / total_weight).float()


def compute_preferred_bitwidths(
    relative_sensitivity: Mapping[str, torch.Tensor],
    reference_bits: float,
    delta_bits: float,
    minimum_bits: float,
    maximum_bits: float,
) -> OrderedDict[str, torch.Tensor]:
    """
    Convert relative sensitivity into preferred block bitwidth.

    Formula:

        B_pref_b = B_reference + delta_bits * d_b

    Therefore:

        sensitive block:
            d_b > 0
            -> preferred bitwidth above the reference

        insensitive block:
            d_b < 0
            -> preferred bitwidth below the reference
    """
    if minimum_bits >= maximum_bits:
        raise ValueError(
            "minimum_bits must be smaller than maximum_bits."
        )

    if delta_bits < 0:
        raise ValueError(
            "delta_bits must be non-negative."
        )

    preferred_bitwidths = OrderedDict()

    for block_name, relative_value in relative_sensitivity.items():
        preferred_bits = (
            float(reference_bits)
            + float(delta_bits)
            * torch.as_tensor(
                relative_value,
                dtype=torch.float32,
            )
        )

        preferred_bits = preferred_bits.clamp(
            min=float(minimum_bits),
            max=float(maximum_bits),
        )

        preferred_bitwidths[block_name] = preferred_bits.reshape(())

    return preferred_bitwidths


def normalize_preferred_bitwidths(
    preferred_bitwidths: Mapping[str, torch.Tensor],
    minimum_bits: float,
    maximum_bits: float,
    mean_epsilon: float = 1e-4,
) -> OrderedDict[str, torch.Tensor]:
    """
    Convert preferred bitwidth into the Beta-distribution mean.

    Formula:

        mu_b =
            (B_pref_b - B_min) /
            (B_max - B_min)

    The result is clipped slightly away from 0 and 1 because a valid
    Beta distribution requires positive a_b and b_b.
    """
    if minimum_bits >= maximum_bits:
        raise ValueError(
            "minimum_bits must be smaller than maximum_bits."
        )

    if not 0.0 < mean_epsilon < 0.5:
        raise ValueError(
            "mean_epsilon must be between 0 and 0.5."
        )

    bitwidth_range = float(maximum_bits - minimum_bits)

    beta_means = OrderedDict()

    for block_name, preferred_bits in preferred_bitwidths.items():
        beta_mean = (
            torch.as_tensor(
                preferred_bits,
                dtype=torch.float32,
            )
            - float(minimum_bits)
        ) / bitwidth_range

        beta_mean = beta_mean.clamp(
            min=mean_epsilon,
            max=1.0 - mean_epsilon,
        )

        beta_means[block_name] = beta_mean.reshape(())

    return beta_means


def compute_beta_shape_parameters(
    beta_means: Mapping[str, torch.Tensor],
    beta_kappa: float,
) -> OrderedDict[str, dict[str, torch.Tensor]]:
    """
    Convert each Beta mean into shape parameters a_b and b_b.

    Formula:

        a_b = kappa * mu_b
        b_b = kappa * (1 - mu_b)

    Here, kappa controls the strength or concentration of the prior.

    Larger kappa:
        narrower and stronger Beta preference

    Smaller kappa:
        wider and weaker Beta preference
    """
    if beta_kappa <= 0:
        raise ValueError(
            "beta_kappa must be positive."
        )

    beta_parameters = OrderedDict()

    for block_name, beta_mean in beta_means.items():
        beta_mean = torch.as_tensor(
            beta_mean,
            dtype=torch.float32,
        ).reshape(())

        alpha = float(beta_kappa) * beta_mean
        beta = float(beta_kappa) * (1.0 - beta_mean)

        if alpha <= 0 or beta <= 0:
            raise ValueError(
                f"Invalid Beta parameters for block '{block_name}': "
                f"alpha={alpha.item()}, beta={beta.item()}."
            )

        beta_parameters[block_name] = {
            "alpha": alpha,
            "beta": beta,
        }

    return beta_parameters


def build_poseidon_beta_parameters(
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
    """
    Build complete Beta information for every Poseidon block.

    Returns:

        {
            block_name: {
                "sensitivity": ...,
                "relative_sensitivity": ...,
                "preferred_bits": ...,
                "beta_mean": ...,
                "alpha": ...,
                "beta": ...,
            }
        }
    """
    if not minimum_bits <= reference_bits <= maximum_bits:
        raise ValueError(
            "reference_bits must lie between minimum_bits "
            "and maximum_bits."
        )

    normalized_sensitivity = normalize_raw_sensitivity(
        sensitivity=sensitivity,
    )

    relative_sensitivity = compute_relative_sensitivity(
        sensitivity=normalized_sensitivity,
        block_parameter_counts=block_parameter_counts,
        epsilon=relative_epsilon,
    )

    preferred_bitwidths = compute_preferred_bitwidths(
        relative_sensitivity=relative_sensitivity,
        reference_bits=reference_bits,
        delta_bits=delta_bits,
        minimum_bits=minimum_bits,
        maximum_bits=maximum_bits,
    )

    beta_means = normalize_preferred_bitwidths(
        preferred_bitwidths=preferred_bitwidths,
        minimum_bits=minimum_bits,
        maximum_bits=maximum_bits,
        mean_epsilon=mean_epsilon,
    )

    shape_parameters = compute_beta_shape_parameters(
        beta_means=beta_means,
        beta_kappa=beta_kappa,
    )

    complete_parameters = OrderedDict()

    weighted_reference_bits = None
    if block_parameter_counts is not None:
        weighted_reference_bits = compute_parameter_weighted_average(
            values=preferred_bitwidths,
            block_parameter_counts=block_parameter_counts,
        )

    for block_name in sensitivity:
        complete_parameters[block_name] = {
            "raw_sensitivity": torch.as_tensor(
                sensitivity[block_name],
                dtype=torch.float32,
            ).reshape(()),

            # First normalization: r_b in [0, 1]
            "normalized_sensitivity": normalized_sensitivity[block_name],

            # Second normalization: d_b in [-1, 1]
            "relative_sensitivity": relative_sensitivity[block_name],

            "preferred_bits": preferred_bitwidths[block_name],
            "beta_mean": beta_means[block_name],
            "alpha": shape_parameters[block_name]["alpha"],
            "beta": shape_parameters[block_name]["beta"],
        }

        if block_parameter_counts is not None:
            complete_parameters[block_name]["parameter_count"] = torch.as_tensor(
                block_parameter_counts[block_name],
                dtype=torch.float32,
            ).reshape(())

            complete_parameters[block_name][
                "weighted_average_preferred_bits"
            ] = weighted_reference_bits

    return complete_parameters


def save_poseidon_beta_parameters(
    beta_parameters: Mapping[
        str,
        Mapping[str, torch.Tensor],
    ],
    save_path: str | Path,
    metadata: dict | None = None,
) -> None:
    """
    Save precomputed block-wise Beta parameters.

    These parameters can later be loaded during SBPQ optimization without
    recalculating block sensitivity.
    """
    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serializable_parameters = OrderedDict()

    for block_name, block_parameters in beta_parameters.items():
        serializable_parameters[block_name] = {
            parameter_name: torch.as_tensor(parameter_value)
            .detach()
            .cpu()
            for parameter_name, parameter_value
            in block_parameters.items()
        }

    save_object = {
        "block_beta_parameters": serializable_parameters,
        "metadata": metadata or {},
    }

    torch.save(
        save_object,
        save_path,
    )

    print(
        f"[INFO] Saved Poseidon Beta parameters to: {save_path}"
    )


def load_poseidon_beta_parameters(
    parameter_path: str | Path,
) -> dict:
    """
    Load precomputed Poseidon Beta parameters.
    """
    parameter_path = Path(parameter_path)

    if not parameter_path.exists():
        raise FileNotFoundError(
            f"Beta-parameter file was not found: {parameter_path}"
        )

    saved_object = torch.load(
        parameter_path,
        map_location="cpu",
    )

    if "block_beta_parameters" not in saved_object:
        raise KeyError(
            "Saved file does not contain "
            "'block_beta_parameters'."
        )

    return saved_object


def print_beta_parameter_summary(
    beta_parameters: Mapping[
        str,
        Mapping[str, torch.Tensor],
    ],
) -> None:
    """
    Print the generated Beta parameters for inspection.
    """
    print("\n========== BETA PARAMETER SUMMARY ==========")

    for block_name, parameters in beta_parameters.items():
        raw_sensitivity = parameters["raw_sensitivity"].item()
        normalized_sensitivity = parameters[
            "normalized_sensitivity"
        ].item()
        relative = parameters["relative_sensitivity"].item()
        preferred_bits = parameters["preferred_bits"].item()
        beta_mean = parameters["beta_mean"].item()
        alpha = parameters["alpha"].item()
        beta = parameters["beta"].item()
        parameter_count = parameters.get("parameter_count", None)
        weighted_average_preferred_bits = parameters.get(
            "weighted_average_preferred_bits",
            None,
        )

        print(
            f"{block_name}\n"
            f"    raw sensitivity       = {raw_sensitivity:.6e}\n"
            f"    normalized r_b        = {normalized_sensitivity:.6f}\n"
            f"    relative d_b          = {relative:.6f}\n"
            f"    preferred bits      = {preferred_bits:.4f}\n"
            f"    beta mean           = {beta_mean:.6f}\n"
            f"    alpha               = {alpha:.6f}\n"
            f"    beta                = {beta:.6f}"
        )

        if parameter_count is not None:
            print(
                f"    parameter count     = "
                f"{float(torch.as_tensor(parameter_count).item()):.0f}"
            )

        if weighted_average_preferred_bits is not None:
            print(
                f"    weighted avg B_pref = "
                f"{float(torch.as_tensor(weighted_average_preferred_bits).item()):.6f}"
            )

    print("============================================\n")


def build_and_save_from_sensitivity_file(
    sensitivity_path: str | Path,
    save_path: str | Path,
    minimum_bits: float,
    maximum_bits: float,
    reference_bits: float,
    delta_bits: float,
    beta_kappa: float,
    block_parameter_counts: Mapping[str, torch.Tensor] | None = None,
    mean_epsilon: float = 1e-4,
    relative_epsilon: float = 1e-12,
    metadata: dict | None = None,
) -> OrderedDict[str, dict[str, torch.Tensor]]:
    """
    Convenience function for the complete offline workflow:

        load sensitivity
        -> calculate Beta parameters
        -> print parameters
        -> save parameters
    """
    sensitivity = load_saved_block_sensitivity(
        sensitivity_path=sensitivity_path,
        sensitivity_key="raw_sensitivity",
    )

    beta_parameters = build_poseidon_beta_parameters(
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

    print_beta_parameter_summary(
        beta_parameters=beta_parameters,
    )

    complete_metadata = {
        "sensitivity_path": str(sensitivity_path),
        "minimum_bits": float(minimum_bits),
        "maximum_bits": float(maximum_bits),
        "reference_bits": float(reference_bits),
        "delta_bits": float(delta_bits),
        "beta_kappa": float(beta_kappa),
        "mean_epsilon": float(mean_epsilon),
        "relative_epsilon": float(relative_epsilon),
        "parameter_weighted_centering": block_parameter_counts is not None,
        "number_of_blocks": len(beta_parameters),
    }

    if metadata is not None:
        complete_metadata.update(metadata)

    save_poseidon_beta_parameters(
        beta_parameters=beta_parameters,
        save_path=save_path,
        metadata=complete_metadata,
    )

    return beta_parameters
