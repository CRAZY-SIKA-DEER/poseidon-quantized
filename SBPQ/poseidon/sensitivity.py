"""
Poseidon block-sensitivity calculation for SBPQ.

For every Poseidon structural block, we calculate:

    sensitivity_b = mean((d loss / d block_output)^2)

The loss function is passed from run_sbpq_poseidon.py. Therefore, this file
does not hard-code the Sobolev order or its weights.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable
from pathlib import Path

import torch
import torch.nn as nn

from SBPQ.poseidon.blocks import find_poseidon_blocks
from SBPQ.poseidon.poseidon_utils import poseidon_forward


class BlockOutputGradientHook:
    """
    Capture the gradient of the loss with respect to a block's output.
    """

    def __init__(self) -> None:
        self.gradient: torch.Tensor | None = None

    def clear(self) -> None:
        """
        Remove the gradient stored from the previous batch.
        """
        self.gradient = None

    def __call__(
        self,
        module: nn.Module,
        grad_input,
        grad_output,
    ) -> None:
        """
        grad_output contains the gradients with respect to the module outputs.

        Usually:

            grad_output[0] = d loss / d block_output
        """
        if grad_output is None or len(grad_output) == 0:
            self.gradient = None
            return

        gradient = grad_output[0]

        # Some blocks may return multiple outputs inside a tuple.
        if isinstance(gradient, (tuple, list)):
            gradient = gradient[0]

        if gradient is None:
            self.gradient = None
            return

        self.gradient = gradient.detach()


def reduce_gradient_to_scalar(
    gradient: torch.Tensor,
) -> torch.Tensor:
    """
    Convert a block-output gradient tensor into one sensitivity value.

    We square every gradient element and then take the mean:

        sensitivity = mean(gradient^2)
    """
    if gradient.numel() == 0:
        raise ValueError("Cannot reduce an empty gradient tensor.")

    return gradient.float().pow(2).mean()


def _prepare_loader(
    dataloader,
) -> Iterable:
    """
    Return an iterator over calibration batches.

    The dataloader can be either:

        1. A normal DataLoader/list.
        2. A callable that creates a new iterator.
    """
    if callable(dataloader):
        return dataloader()

    return dataloader


def compute_poseidon_block_sensitivity(
    model: nn.Module,
    dataloader,
    device: str | torch.device,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    max_batches: int | None = None,
    block_names: Iterable[str] | None = None,
) -> OrderedDict[str, torch.Tensor]:
    """
    Compute one sensitivity score for every Poseidon structural block.

    For each block:

        sensitivity = mean(
            (d loss / d block_output) ** 2
        )

    The loss function must have the interface:

        loss = loss_function(prediction, reference)
    """
    device = torch.device(device)

    model = model.to(device)
    model.eval()

    blocks = find_poseidon_blocks(model)
    if block_names is not None:
        requested_block_names = set(block_names)
        blocks = OrderedDict(
            (
                block_name,
                block_module,
            )
            for block_name, block_module in blocks.items()
            if block_name in requested_block_names
        )

    if len(blocks) == 0:
        raise RuntimeError(
            "No Poseidon structural blocks were found in the model."
        )

    sensitivity_results: OrderedDict[str, torch.Tensor] = OrderedDict()

    print(
        f"[INFO] Calculating sensitivity for "
        f"{len(blocks)} Poseidon blocks."
    )

    # Process one block at a time.
    for block_index, (block_name, block_module) in enumerate(
        blocks.items(),
        start=1,
    ):
        print(
            f"[INFO] Block {block_index}/{len(blocks)}: "
            f"{block_name}"
        )

        gradient_hook = BlockOutputGradientHook()

        hook_handle = block_module.register_full_backward_hook(
            gradient_hook
        )

        block_sensitivity_sum = torch.zeros(
            (),
            dtype=torch.float64,
        )

        processed_batches = 0

        try:
            block_loader = _prepare_loader(dataloader)

            for batch_index, batch in enumerate(
                block_loader,
                start=1,
            ):
                if (
                    max_batches is not None
                    and batch_index > max_batches
                ):
                    break

                model.zero_grad(set_to_none=True)
                gradient_hook.clear()

                # The reference field used by the Sobolev loss.
                labels = batch.get("labels", None)

                if labels is None:
                    raise ValueError(
                        "Block-sensitivity calculation requires "
                        "batch['labels']."
                    )

                labels = labels.to(device)

                # poseidon_forward() internally moves pixel_values,
                # time, pixel_mask, and labels to the device.
                prediction = poseidon_forward(
                    model=model,
                    batch=batch,
                    device=device,
                )

                loss = loss_function(
                    prediction,
                    labels,
                )

                if loss.ndim != 0:
                    raise ValueError(
                        "loss_function must return one scalar loss, "
                        f"but returned shape {loss.shape}."
                    )

                loss.backward()

                if gradient_hook.gradient is None:
                    raise RuntimeError(
                        "Failed to capture the output gradient for "
                        f"block '{block_name}'."
                    )

                current_sensitivity = reduce_gradient_to_scalar(
                    gradient_hook.gradient
                )

                block_sensitivity_sum += (
                    current_sensitivity
                    .detach()
                    .cpu()
                    .double()
                )

                processed_batches += 1

                print(
                    f"    batch={batch_index} "
                    f"| loss={loss.item():.6e} "
                    f"| sensitivity={current_sensitivity.item():.6e}"
                )

        finally:
            hook_handle.remove()

        if processed_batches == 0:
            raise RuntimeError(
                f"No calibration batches were processed for "
                f"block '{block_name}'."
            )

        average_sensitivity = (
            block_sensitivity_sum / processed_batches
        ).float()

        sensitivity_results[block_name] = average_sensitivity

        print(
            f"    average sensitivity="
            f"{average_sensitivity.item():.6e}"
        )

    return sensitivity_results

def normalize_block_sensitivity(
    sensitivity: dict[str, torch.Tensor],
    epsilon: float = 1e-8,
) -> OrderedDict[str, torch.Tensor]:
    """
    Min-max normalize sensitivity across all blocks into [0, 1].

    The least sensitive block becomes approximately 0.
    The most sensitive block becomes approximately 1.
    """
    if len(sensitivity) == 0:
        raise ValueError(
            "Cannot normalize an empty sensitivity dictionary."
        )

    block_names = list(sensitivity.keys())

    sensitivity_values = torch.stack(
        [
            torch.as_tensor(
                sensitivity[block_name],
                dtype=torch.float32,
            ).reshape(())
            for block_name in block_names
        ]
    )

    minimum = sensitivity_values.min()
    maximum = sensitivity_values.max()

    value_range = maximum - minimum

    if value_range.abs() < epsilon:
        # All blocks have effectively the same sensitivity.
        normalized_values = torch.full_like(
            sensitivity_values,
            fill_value=0.5,
        )
    else:
        normalized_values = (
            sensitivity_values - minimum
        ) / (
            value_range + epsilon
        )

    return OrderedDict(
        (
            block_name,
            normalized_values[index],
        )
        for index, block_name in enumerate(block_names)
    )


def save_block_sensitivity(
    raw_sensitivity: dict[str, torch.Tensor],
    normalized_sensitivity: dict[str, torch.Tensor],
    save_path: str | Path,
    metadata: dict | None = None,
) -> None:
    """
    Save raw and normalized block sensitivities.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_object = {
        "raw_sensitivity": {
            block_name: torch.as_tensor(value).detach().cpu()
            for block_name, value in raw_sensitivity.items()
        },
        "normalized_sensitivity": {
            block_name: torch.as_tensor(value).detach().cpu()
            for block_name, value in normalized_sensitivity.items()
        },
        "blocks": {
            block_name: {
                "block_sensitivity": torch.as_tensor(
                    raw_sensitivity[block_name]
                ).detach().cpu(),
                "normalized_sensitivity": torch.as_tensor(
                    normalized_sensitivity[block_name]
                ).detach().cpu(),
                "layers": {},
            }
            for block_name in raw_sensitivity
        },
        "metadata": metadata or {},
    }

    torch.save(
        save_object,
        save_path,
    )

    print(
        f"[INFO] Saved block sensitivity to: {save_path}"
    )


def load_block_sensitivity(
    sensitivity_path: str | Path,
) -> dict:
    """
    Load a previously saved block-sensitivity file.
    """
    sensitivity_path = Path(sensitivity_path)

    if not sensitivity_path.exists():
        raise FileNotFoundError(
            f"Sensitivity file not found: {sensitivity_path}"
        )

    return torch.load(
        sensitivity_path,
        map_location="cpu",
    )
