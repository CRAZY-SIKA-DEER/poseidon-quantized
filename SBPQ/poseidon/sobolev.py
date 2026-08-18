"""
Spatial Sobolev loss for Poseidon outputs.

The loss supports derivative orders from 0 up to a chosen maximum order.

Order 0:
    Compare the field values directly.

Order 1:
    Compare first spatial derivatives.

Order 2:
    Compare second and mixed spatial derivatives.

And similarly for higher orders.

Only spatial derivatives are included.
Time derivatives are not considered.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def prepare_channel_statistics(
    mean,
    std,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert normalization statistics into tensors shaped [1, C, 1, 1].
    """
    mean = torch.as_tensor(
        mean,
        device=device,
        dtype=dtype,
    )

    std = torch.as_tensor(
        std,
        device=device,
        dtype=dtype,
    )

    if mean.ndim == 1:
        mean = mean.view(1, -1, 1, 1)

    if std.ndim == 1:
        std = std.view(1, -1, 1, 1)

    return mean, std


def denormalize_field(
    field: torch.Tensor,
    mean,
    std,
) -> torch.Tensor:
    """
    Convert a normalized field back to its physical scale.
    """
    mean, std = prepare_channel_statistics(
        mean=mean,
        std=std,
        device=field.device,
        dtype=field.dtype,
    )

    return field * std + mean


def select_physical_channels(
    field: torch.Tensor,
    dataset_name: str,
) -> torch.Tensor:
    """
    Select physically meaningful output channels.

    For the current incompressible Poseidon datasets:

        [rho, u, v, p] -> [u, v, p]

    Other datasets currently keep all channels.
    """
    number_of_channels = field.shape[1]
    dataset_name = dataset_name.lower()

    if "incompressible" in dataset_name or "ns" in dataset_name:
        if number_of_channels == 4:
            channel_indices = [1, 2, 3]
        else:
            channel_indices = list(range(number_of_channels))
    else:
        channel_indices = list(range(number_of_channels))

    return field[:, channel_indices, ...].contiguous()


def spatial_difference(
    field: torch.Tensor,
    axis: str,
) -> torch.Tensor:
    """
    Apply one forward finite difference along one spatial axis.

    Expected field shape:
        [batch, channels, height, width]
    """
    if axis == "x":
        return field[..., 1:] - field[..., :-1]

    if axis == "y":
        return field[..., 1:, :] - field[..., :-1, :]

    raise ValueError(
        f"Unsupported spatial axis '{axis}'. Use 'x' or 'y'."
    )


def spatial_derivative(
    field: torch.Tensor,
    x_order: int,
    y_order: int,
) -> torch.Tensor:
    """
    Compute D_x^x_order D_y^y_order using repeated finite differences.

    Example:

        x_order=2, y_order=0
            -> D_xx

        x_order=1, y_order=1
            -> D_xy

        x_order=0, y_order=2
            -> D_yy
    """
    if x_order < 0 or y_order < 0:
        raise ValueError("Derivative orders must be non-negative.")

    derivative = field

    for _ in range(x_order):
        derivative = spatial_difference(
            derivative,
            axis="x",
        )

    for _ in range(y_order):
        derivative = spatial_difference(
            derivative,
            axis="y",
        )

    return derivative


def get_order_weights(
    max_order: int,
    order_weights: float | Sequence[float],
) -> list[float]:
    """
    Produce one weight for every order from 0 to max_order.

    Examples:

        order_weights=1.0
            -> [1.0, 1.0, ..., 1.0]

        order_weights=[1.0, 0.5, 0.1]
            -> weights for orders 0, 1, and 2
    """
    if isinstance(order_weights, (int, float)):
        return [
            float(order_weights)
            for _ in range(max_order + 1)
        ]

    weights = list(order_weights)

    if len(weights) != max_order + 1:
        raise ValueError(
            "order_weights must contain exactly "
            f"{max_order + 1} values, but received {len(weights)}."
        )

    return [float(weight) for weight in weights]


def compute_spatial_sobolev_loss(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    max_order: int = 1,
    order_weights: float | Sequence[float] = 1.0,
    norm: str = "l1",
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute a spatial Sobolev loss up to max_order.

    For each derivative order k, all derivatives satisfying

        x_order + y_order = k

    are included.

    For example, order 2 includes:

        D_xx, D_xy, D_yy

    Args:
        prediction:
            Predicted field shaped [B, C, H, W].

        reference:
            Reference field shaped [B, C, H, W].

        max_order:
            Highest spatial derivative order to include.
            Supported range: 0 to 8.

        order_weights:
            Either one shared scalar or one value per derivative order.

        norm:
            Pointwise discrepancy norm. The paper default is "l1",
            matching the finite-difference Sobolev norm. "mse" is kept
            for ablations and compatibility.

        reduction:
            Reduction used after calculating pointwise discrepancy.

    Returns:
        Scalar Sobolev loss.
    """
    if prediction.shape != reference.shape:
        raise ValueError(
            "prediction and reference must have the same shape, "
            f"but received {prediction.shape} and {reference.shape}."
        )

    if prediction.ndim != 4:
        raise ValueError(
            "Sobolev loss expects tensors shaped [B, C, H, W], "
            f"but received {prediction.shape}."
        )

    if not 0 <= max_order <= 8:
        raise ValueError(
            f"max_order must be between 0 and 8, but received {max_order}."
        )

    if norm not in {"l1", "mse"}:
        raise ValueError(
            f"Unsupported Sobolev norm '{norm}'. Use 'l1' or 'mse'."
        )

    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(
            f"Unsupported reduction '{reduction}'."
        )

    height = prediction.shape[-2]
    width = prediction.shape[-1]

    if max_order >= height or max_order >= width:
        raise ValueError(
            f"max_order={max_order} is too large for spatial size "
            f"{height}x{width}."
        )

    weights = get_order_weights(
        max_order=max_order,
        order_weights=order_weights,
    )

    total_loss = prediction.new_zeros(())

    for total_order in range(max_order + 1):
        order_loss = prediction.new_zeros(())
        derivative_count = 0

        for x_order in range(total_order + 1):
            y_order = total_order - x_order

            prediction_derivative = spatial_derivative(
                field=prediction,
                x_order=x_order,
                y_order=y_order,
            )

            reference_derivative = spatial_derivative(
                field=reference,
                x_order=x_order,
                y_order=y_order,
            )

            if norm == "l1":
                derivative_loss = F.l1_loss(
                    prediction_derivative,
                    reference_derivative,
                    reduction=reduction,
                )
            else:
                derivative_loss = F.mse_loss(
                    prediction_derivative,
                    reference_derivative,
                    reduction=reduction,
                )

            if derivative_loss.ndim != 0:
                derivative_loss = derivative_loss.mean()

            order_loss = order_loss + derivative_loss
            derivative_count += 1

        # Average derivatives belonging to the same total order.
        order_loss = order_loss / derivative_count

        total_loss = (
            total_loss
            + weights[total_order] * order_loss
        )

    return total_loss


def compute_poseidon_sobolev_loss(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    dataset,
    dataset_name: str,
    max_order: int = 1,
    order_weights: float | Sequence[float] = 1.0,
    norm: str = "l1",
    transpose_spatial_axes: bool = False,
) -> torch.Tensor:
    """
    Compute the denormalized spatial Sobolev loss for Poseidon.

    The dataset must provide:

        dataset.constants["mean"]
        dataset.constants["std"]

    Time derivatives are not included.
    """
    if not hasattr(dataset, "constants"):
        raise ValueError(
            "The supplied dataset does not contain normalization constants."
        )

    constants = dataset.constants

    if "mean" not in constants or "std" not in constants:
        raise ValueError(
            "dataset.constants must contain 'mean' and 'std'."
        )

    prediction = denormalize_field(
        field=prediction,
        mean=constants["mean"],
        std=constants["std"],
    )

    reference = denormalize_field(
        field=reference,
        mean=constants["mean"],
        std=constants["std"],
    )

    if transpose_spatial_axes:
        prediction = prediction.transpose(-2, -1)
        reference = reference.transpose(-2, -1)

    prediction = select_physical_channels(
        field=prediction,
        dataset_name=dataset_name,
    )

    reference = select_physical_channels(
        field=reference,
        dataset_name=dataset_name,
    )

    return compute_spatial_sobolev_loss(
        prediction=prediction,
        reference=reference,
        max_order=max_order,
        order_weights=order_weights,
        norm=norm,
    )
