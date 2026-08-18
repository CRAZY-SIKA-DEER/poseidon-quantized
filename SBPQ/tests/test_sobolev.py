import torch

from SBPQ.poseidon.sobolev import compute_spatial_sobolev_loss


def test_spatial_sobolev_l1_order_one_matches_manual_value():
    prediction = torch.tensor(
        [[[[1.0, 3.0], [2.0, 6.0]]]]
    )
    reference = torch.zeros_like(prediction)

    loss = compute_spatial_sobolev_loss(
        prediction=prediction,
        reference=reference,
        max_order=1,
        order_weights=(1.0, 1.0),
        norm="l1",
    )

    value_loss = prediction.abs().mean()
    dx_loss = torch.tensor([[[[2.0], [4.0]]]]).abs().mean()
    dy_loss = torch.tensor([[[[1.0, 3.0]]]]).abs().mean()
    expected = value_loss + (dx_loss + dy_loss) / 2.0

    assert torch.allclose(loss, expected)


def test_spatial_sobolev_rejects_unknown_norm():
    prediction = torch.zeros(1, 1, 2, 2)

    try:
        compute_spatial_sobolev_loss(
            prediction=prediction,
            reference=prediction,
            norm="unknown",
        )
    except ValueError as error:
        assert "Unsupported Sobolev norm" in str(error)
    else:
        raise AssertionError("Expected ValueError.")
