"""Data helpers for Aurora SBPQ experiments."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F
from aurora import Batch, Metadata


def _tensor_dict(values: Mapping[str, object]) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value, dtype=torch.float32)
        for key, value in values.items()
    }


def batch_from_pickle_dict(saved: Mapping[str, object]) -> Batch:
    metadata = saved["metadata"]
    return Batch(
        surf_vars=_tensor_dict(saved.get("surf_vars", {})),
        static_vars=_tensor_dict(saved.get("static_vars", {})),
        atmos_vars=_tensor_dict(saved.get("atmos_vars", {})),
        metadata=Metadata(
            lat=torch.as_tensor(metadata["lat"], dtype=torch.float32),
            lon=torch.as_tensor(metadata["lon"], dtype=torch.float32),
            time=tuple(metadata["time"]),
            atmos_levels=tuple(metadata["atmos_levels"]),
        ),
    )


def load_aurora_pickle_batch(path: str | Path) -> Batch:
    with Path(path).open("rb") as handle:
        saved = pickle.load(handle)
    return batch_from_pickle_dict(saved)


def load_static_vars(path: str | Path) -> dict[str, torch.Tensor]:
    with Path(path).open("rb") as handle:
        saved = pickle.load(handle)
    return _tensor_dict(saved)


def attach_static_vars(
    batch: Batch,
    static_vars: Mapping[str, torch.Tensor],
) -> Batch:
    height, width = batch.spatial_shape
    resized = {}
    for name, value in static_vars.items():
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tuple(tensor.shape[-2:]) != (int(height), int(width)):
            tensor = F.interpolate(
                tensor.reshape(1, 1, *tensor.shape[-2:]),
                size=(int(height), int(width)),
                mode="bilinear",
                align_corners=False,
            ).reshape(int(height), int(width))
        resized[name] = tensor
    return Batch(
        surf_vars=batch.surf_vars,
        static_vars=resized,
        atmos_vars=batch.atmos_vars,
        metadata=batch.metadata,
    )


def spatial_crop_batch(
    batch: Batch,
    height: int | None = None,
    width: int | None = None,
) -> Batch:
    if height is None and width is None:
        return batch
    original_height, original_width = batch.spatial_shape
    height = int(height or original_height)
    width = int(width or original_width)
    if height > original_height or width > original_width:
        raise ValueError("Crop size cannot exceed the original Aurora batch size.")
    top = int((original_height - height) // 2)
    left = int((original_width - width) // 2)
    bottom = top + height
    right = left + width
    return Batch(
        surf_vars={k: v[..., top:bottom, left:right] for k, v in batch.surf_vars.items()},
        static_vars={k: v[..., top:bottom, left:right] for k, v in batch.static_vars.items()},
        atmos_vars={k: v[..., top:bottom, left:right] for k, v in batch.atmos_vars.items()},
        metadata=Metadata(
            lat=batch.metadata.lat[top:bottom],
            lon=batch.metadata.lon[left:right],
            time=batch.metadata.time,
            atmos_levels=batch.metadata.atmos_levels,
            rollout_step=batch.metadata.rollout_step,
        ),
    )


def move_batch_to_device(batch: Batch, device: str | torch.device) -> Batch:
    return batch.to(torch.device(device))


def detach_batch_to_cpu(batch: Batch) -> Batch:
    return Batch(
        surf_vars={k: v.detach().cpu() for k, v in batch.surf_vars.items()},
        static_vars={k: v.detach().cpu() for k, v in batch.static_vars.items()},
        atmos_vars={k: v.detach().cpu() for k, v in batch.atmos_vars.items()},
        metadata=batch.metadata,
    )


def clone_batch(batch: Batch) -> Batch:
    return Batch(
        surf_vars={k: v.clone() for k, v in batch.surf_vars.items()},
        static_vars={k: v.clone() for k, v in batch.static_vars.items()},
        atmos_vars={k: v.clone() for k, v in batch.atmos_vars.items()},
        metadata=batch.metadata,
    )


def iter_output_tensors(batch: Batch):
    for group_name, group in (
        ("surf", batch.surf_vars),
        ("atmos", batch.atmos_vars),
    ):
        for variable_name, tensor in group.items():
            yield group_name, variable_name, tensor


def batch_mse(prediction: Batch, target: Batch) -> torch.Tensor:
    losses = []
    for group_name, variable_name, pred_tensor in iter_output_tensors(prediction):
        target_group = target.surf_vars if group_name == "surf" else target.atmos_vars
        target_tensor = target_group[variable_name].to(
            pred_tensor.device,
            dtype=pred_tensor.dtype,
        )
        losses.append((pred_tensor - target_tensor).pow(2).mean())
    if not losses:
        raise ValueError("Aurora output batch contains no comparable tensors.")
    return torch.stack(losses).mean()


def batch_squared_error_per_sample(prediction: Batch, target: Batch) -> torch.Tensor:
    errors = []
    for group_name, variable_name, pred_tensor in iter_output_tensors(prediction):
        target_group = target.surf_vars if group_name == "surf" else target.atmos_vars
        target_tensor = target_group[variable_name].to(
            pred_tensor.device,
            dtype=pred_tensor.dtype,
        )
        reduce_dims = tuple(range(1, pred_tensor.ndim))
        errors.append((pred_tensor - target_tensor).pow(2).sum(dim=reduce_dims))
    return torch.stack(errors, dim=0).sum(dim=0)
