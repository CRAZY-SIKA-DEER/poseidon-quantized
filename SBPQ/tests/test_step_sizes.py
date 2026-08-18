import torch

from SBPQ.step_sizes import (
    clamp_step_size,
    effective_bitwidth_from_step_size,
)


def test_clamp_step_size_enforces_bitwidth_bounds():
    quantization_range = torch.tensor([16.0, 16.0])
    step_size = torch.tensor([100.0, 1e-6])

    clamped = clamp_step_size(
        step_size=step_size,
        quantization_range=quantization_range,
        minimum_bits=2.0,
        maximum_bits=4.0,
    )

    bits = effective_bitwidth_from_step_size(
        quantization_range=quantization_range,
        step_size=clamped,
    )

    assert torch.all(bits >= 2.0)
    assert torch.all(bits <= 4.0)
    assert torch.allclose(bits, torch.tensor([2.0, 4.0]))
