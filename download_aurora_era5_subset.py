"""Download an Aurora-compatible ERA5 subset with CDS API.

This intentionally downloads selected variables and times only. Full ERA5 is
far too large for SBPQ calibration experiments.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import cdsapi


SURFACE_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]

STATIC_VARIABLES = [
    "geopotential",
    "land_sea_mask",
    "soil_type",
]

ATMOS_VARIABLES = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "geopotential",
]

PRESSURE_LEVELS = [
    "50",
    "100",
    "150",
    "200",
    "250",
    "300",
    "400",
    "500",
    "600",
    "700",
    "850",
    "925",
    "1000",
]

DEFAULT_TIMES = ["00:00", "06:00", "12:00", "18:00"]


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def retrieve_if_missing(client: cdsapi.Client, dataset: str, request: dict, output: Path) -> None:
    if output.exists() and output.stat().st_size > 0:
        print(f"[SKIP] {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DOWNLOAD] {dataset} -> {output}")
    client.retrieve(dataset, request, str(output))
    print(f"[DONE] {output} ({output.stat().st_size} bytes)")


def build_base_request(day: date, times: list[str]) -> dict:
    return {
        "product_type": "reanalysis",
        "year": f"{day.year:04d}",
        "month": f"{day.month:02d}",
        "day": f"{day.day:02d}",
        "time": times,
        "format": "netcdf",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="aurora_artifacts/datasets/era5_025/raw",
    )
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--num-days", type=int, default=1)
    parser.add_argument(
        "--times",
        nargs="+",
        default=DEFAULT_TIMES,
        help="ERA5 times to download per day, e.g. 00:00 06:00 12:00 18:00.",
    )
    parser.add_argument(
        "--skip-static",
        action="store_true",
        help="Do not download static fields.",
    )
    args = parser.parse_args()

    if args.num_days <= 0:
        raise ValueError("--num-days must be positive.")

    output_dir = Path(args.output_dir)
    start = parse_date(args.start_date)
    client = cdsapi.Client()

    if not args.skip_static:
        static_request = {
            "product_type": "reanalysis",
            "variable": STATIC_VARIABLES,
            "year": f"{start.year:04d}",
            "month": f"{start.month:02d}",
            "day": f"{start.day:02d}",
            "time": "00:00",
            "format": "netcdf",
        }
        retrieve_if_missing(
            client,
            "reanalysis-era5-single-levels",
            static_request,
            output_dir / "static.nc",
        )

    for offset in range(args.num_days):
        current = start + timedelta(days=offset)
        day_tag = current.isoformat()

        surface_request = build_base_request(current, args.times)
        surface_request["variable"] = SURFACE_VARIABLES
        retrieve_if_missing(
            client,
            "reanalysis-era5-single-levels",
            surface_request,
            output_dir / f"{day_tag}-surface-level.nc",
        )

        atmos_request = build_base_request(current, args.times)
        atmos_request["variable"] = ATMOS_VARIABLES
        atmos_request["pressure_level"] = PRESSURE_LEVELS
        retrieve_if_missing(
            client,
            "reanalysis-era5-pressure-levels",
            atmos_request,
            output_dir / f"{day_tag}-atmospheric.nc",
        )

    print(f"[DONE] ERA5 subset saved under {output_dir}")


if __name__ == "__main__":
    main()

