import torch

from SBPQ.poseidon.beta_parameter_builder import (
    build_poseidon_beta_parameters,
    compute_parameter_weighted_average,
    compute_relative_sensitivity,
    normalize_raw_sensitivity,
)


def test_relative_sensitivity_uses_parameter_weighted_centering():
    raw = {
        "large": torch.tensor(1.0),
        "small": torch.tensor(0.0),
    }
    normalized = normalize_raw_sensitivity(raw)
    counts = {
        "large": torch.tensor(999.0),
        "small": torch.tensor(1.0),
    }

    relative = compute_relative_sensitivity(
        sensitivity=normalized,
        block_parameter_counts=counts,
        epsilon=0.0,
    )

    weighted_sum = (
        counts["large"] * relative["large"]
        + counts["small"] * relative["small"]
    )

    assert torch.allclose(weighted_sum, torch.tensor(0.0), atol=1e-5)
    assert relative["large"] > 0
    assert relative["small"] < 0


def test_beta_parameters_preserve_weighted_reference_bitwidth():
    sensitivity = {
        "large": torch.tensor(1.0),
        "small": torch.tensor(0.0),
    }
    counts = {
        "large": torch.tensor(999.0),
        "small": torch.tensor(1.0),
    }

    beta_parameters = build_poseidon_beta_parameters(
        sensitivity=sensitivity,
        minimum_bits=1.0,
        maximum_bits=16.0,
        reference_bits=8.0,
        delta_bits=2.0,
        beta_kappa=10.0,
        block_parameter_counts=counts,
        relative_epsilon=0.0,
    )

    preferred = {
        block_name: parameters["preferred_bits"]
        for block_name, parameters in beta_parameters.items()
    }

    weighted_average = compute_parameter_weighted_average(
        values=preferred,
        block_parameter_counts=counts,
    )

    assert torch.allclose(weighted_average, torch.tensor(8.0), atol=1e-5)
    assert (
        beta_parameters["large"]["preferred_bits"]
        > beta_parameters["small"]["preferred_bits"]
    )
    assert (
        beta_parameters["large"]["beta_mean"]
        > beta_parameters["small"]["beta_mean"]
    )
