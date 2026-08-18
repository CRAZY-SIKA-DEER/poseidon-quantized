from types import SimpleNamespace

import torch
import torch.nn as nn

from SBPQ.poseidon.likelihood import (
    compute_network_mc_likelihood_single_batch,
)


class TinyPoseidonLikeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)
        nn.init.constant_(self.linear.weight, 1.0)

    def forward(
        self,
        pixel_values,
        time=None,
        pixel_mask=None,
        labels=None,
    ):
        output = self.linear(pixel_values).reshape(-1, 1, 1, 1)
        return SimpleNamespace(output=output)


def test_network_likelihood_backpropagates_to_step_size():
    torch.manual_seed(7)

    model = TinyPoseidonLikeModel()
    batch = {
        "pixel_values": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "labels": torch.zeros(2, 1, 1, 1),
    }
    clean_output = torch.zeros(2, 1, 1, 1)
    step_size = nn.Parameter(torch.tensor([0.25]))

    loss = compute_network_mc_likelihood_single_batch(
        model=model,
        step_sizes={"linear": step_size},
        frozen_batches=[batch],
        clean_network_outputs=[clean_output],
        batch_index=0,
        num_mc_samples=3,
        eta=1.0,
        device="cpu",
    )

    loss.backward()

    assert torch.isfinite(loss)
    assert step_size.grad is not None
    assert torch.isfinite(step_size.grad).all()
