"""Build Aurora Batch windows from downloaded ERA5 NetCDF files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import xarray as xr
from aurora import Batch, Metadata

from SBPQ.aurora.data_utils import spatial_crop_batch


@dataclass(frozen=True)
class ERA5Window:
    input_batch: Batch
    target_batch: Batch
    input_times: tuple[object, object]
    target_time: object


def _time_values(dataset) -> list:
    key = "valid_time" if "valid_time" in dataset.coords else "time"
    return list(dataset[key].values)


def _datetime_from_value(value):
    return value.astype("datetime64[s]").tolist()


def _read_days(raw_dir: str | Path, day_tags: list[str]):
    raw_dir = Path(raw_dir)
    static_ds = xr.open_dataset(raw_dir / "static.nc", engine="netcdf4")
    surf_datasets = [
        xr.open_dataset(raw_dir / f"{day}-surface-level.nc", engine="netcdf4")
        for day in day_tags
    ]
    atmos_datasets = [
        xr.open_dataset(raw_dir / f"{day}-atmospheric.nc", engine="netcdf4")
        for day in day_tags
    ]
    surf_ds = xr.concat(surf_datasets, dim="valid_time")
    atmos_ds = xr.concat(atmos_datasets, dim="valid_time")
    for dataset in surf_datasets + atmos_datasets:
        dataset.close()
    return static_ds, surf_ds, atmos_ds


def _make_batch(static_ds, surf_ds, atmos_ds, input_indices: list[int]) -> Batch:
    metadata_time = _datetime_from_value(_time_values(surf_ds)[input_indices[-1]])
    return Batch(
        surf_vars={
            "2t": torch.from_numpy(surf_ds["t2m"].values[input_indices][None]).float(),
            "10u": torch.from_numpy(surf_ds["u10"].values[input_indices][None]).float(),
            "10v": torch.from_numpy(surf_ds["v10"].values[input_indices][None]).float(),
            "msl": torch.from_numpy(surf_ds["msl"].values[input_indices][None]).float(),
        },
        static_vars={
            "z": torch.from_numpy(static_ds["z"].values[0]).float(),
            "slt": torch.from_numpy(static_ds["slt"].values[0]).float(),
            "lsm": torch.from_numpy(static_ds["lsm"].values[0]).float(),
        },
        atmos_vars={
            "t": torch.from_numpy(atmos_ds["t"].values[input_indices][None]).float(),
            "u": torch.from_numpy(atmos_ds["u"].values[input_indices][None]).float(),
            "v": torch.from_numpy(atmos_ds["v"].values[input_indices][None]).float(),
            "q": torch.from_numpy(atmos_ds["q"].values[input_indices][None]).float(),
            "z": torch.from_numpy(atmos_ds["z"].values[input_indices][None]).float(),
        },
        metadata=Metadata(
            lat=torch.from_numpy(surf_ds.latitude.values).float(),
            lon=torch.from_numpy(surf_ds.longitude.values).float(),
            time=(metadata_time,),
            atmos_levels=tuple(int(level) for level in atmos_ds.pressure_level.values),
        ),
    )


def build_era5_windows(
    raw_dir: str | Path,
    day_tags: list[str],
    max_windows: int | None = None,
    crop_height: int | None = None,
    crop_width: int | None = None,
) -> list[ERA5Window]:
    static_ds, surf_ds, atmos_ds = _read_days(raw_dir, day_tags)
    try:
        times = _time_values(surf_ds)
        if len(times) != len(_time_values(atmos_ds)):
            raise ValueError("Surface and atmospheric ERA5 files have different time counts.")
        windows = []
        for start in range(max(0, len(times) - 2)):
            input_batch = _make_batch(static_ds, surf_ds, atmos_ds, [start, start + 1])
            target_batch = _make_batch(static_ds, surf_ds, atmos_ds, [start + 2])
            input_batch = spatial_crop_batch(input_batch, crop_height, crop_width)
            target_batch = spatial_crop_batch(target_batch, crop_height, crop_width)
            windows.append(
                ERA5Window(
                    input_batch=input_batch,
                    target_batch=target_batch,
                    input_times=(
                        _datetime_from_value(times[start]),
                        _datetime_from_value(times[start + 1]),
                    ),
                    target_time=_datetime_from_value(times[start + 2]),
                )
            )
            if max_windows is not None and len(windows) >= max_windows:
                break
        return windows
    finally:
        static_ds.close()
        surf_ds.close()
        atmos_ds.close()

