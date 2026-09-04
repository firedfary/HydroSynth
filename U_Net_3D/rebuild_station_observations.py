"""Rebuild gridded CMA monthly precipitation without test-period leakage.

The existing ``observe_data24.csv`` contains trustworthy monthly totals but
stores coordinates as degree-minute integers and computes climatology from the
entire record.  This script corrects the coordinates and uses a fixed early
reference period, so every rolling validation and outer-test target uses the
same climatology that was available before validation began.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from project_paths import OBSERVATION_FILE, STATION_TABLE_FILE

DEFAULT_EXCLUDED = ("2011-09-01", "2011-10-01", "2017-01-01")


def degree_minute_to_decimal(values: pd.Series) -> pd.Series:
    raw = values.round().astype("int64")
    degrees = raw.abs() // 100
    minutes = raw.abs() % 100
    # A few CMA records encode an exact carry as 2960 or 11960.  The formula
    # below correctly normalizes these to 30.0 and 120.0 degrees.
    if (minutes > 60).any():
        bad = raw[minutes > 60].head().tolist()
        raise ValueError(f"Invalid degree-minute coordinates: {bad}")
    decimal = degrees + minutes / 60.0
    return decimal.where(raw >= 0, -decimal)


def build_station_month_table(
    csv_path: Path,
    reference_start: int,
    reference_end: int,
    minimum_reference_years: int,
    climatology_floor_mm: float,
) -> pd.DataFrame:
    columns = ["Stn_No", "Year", "Month", "Lat", "Long", "Precip", "time"]
    observations = pd.read_csv(csv_path, usecols=columns)
    observations["time"] = pd.to_datetime(observations["time"])
    observations["Lat"] = degree_minute_to_decimal(observations["Lat"])
    observations["Long"] = degree_minute_to_decimal(observations["Long"])
    observations = (
        observations.groupby(["Stn_No", "time"], as_index=False)
        .agg(
            Year=("Year", "first"),
            Month=("Month", "first"),
            Lat=("Lat", "last"),
            Long=("Long", "last"),
            Precip=("Precip", "mean"),
        )
        .sort_values(["time", "Stn_No"])
    )

    reference = observations[
        observations["Year"].between(reference_start, reference_end)
    ]
    climatology = (
        reference.groupby(["Stn_No", "Month"])["Precip"]
        .agg(climatology_mm="mean", reference_years="count")
        .reset_index()
    )
    climatology = climatology[
        climatology["reference_years"] >= minimum_reference_years
    ]
    observations = observations.merge(
        climatology, on=["Stn_No", "Month"], how="inner", validate="many_to_one"
    )
    denominator = np.maximum(
        observations["climatology_mm"].to_numpy(), climatology_floor_mm
    )
    observations["anomaly_fraction"] = (
        observations["Precip"].to_numpy()
        - observations["climatology_mm"].to_numpy()
    ) / denominator
    return observations


def interpolate_months(
    observations: pd.DataFrame,
    dates: pd.DatetimeIndex,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_lon, grid_lat = np.meshgrid(longitudes, latitudes)

    # Use the reference-network convex hull as one stable evaluation domain.
    station_meta = observations.groupby("Stn_No")[["Long", "Lat"]].median()
    static_hull = np.isfinite(
        griddata(
            station_meta[["Long", "Lat"]].to_numpy(),
            np.ones(len(station_meta)),
            (grid_lon, grid_lat),
            method="linear",
        )
    )

    precipitation = np.full((len(dates), *grid_lon.shape), np.nan, dtype=np.float32)
    anomaly = np.full_like(precipitation, np.nan)
    station_counts = np.zeros(len(dates), dtype=np.int32)
    grouped = {date: frame for date, frame in observations.groupby("time")}

    for index, date in enumerate(dates):
        current = grouped.get(date)
        if current is None or len(current) < 3:
            continue
        points = current[["Long", "Lat"]].to_numpy(dtype=np.float64)
        station_counts[index] = len(current)
        for output, column in (
            (precipitation, "Precip"),
            (anomaly, "anomaly_fraction"),
        ):
            values = current[column].to_numpy(dtype=np.float64)
            linear = griddata(points, values, (grid_lon, grid_lat), method="linear")
            # Coordinate updates can move a monthly convex hull by one edge
            # cell.  Fill only inside the fixed station-network domain.
            if np.any(static_hull & ~np.isfinite(linear)):
                nearest = griddata(
                    points, values, (grid_lon, grid_lat), method="nearest"
                )
                linear = np.where(static_hull & ~np.isfinite(linear), nearest, linear)
            output[index, static_hull] = linear[static_hull].astype(np.float32)
    return precipitation, anomaly, station_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=Path, default=STATION_TABLE_FILE
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OBSERVATION_FILE,
    )
    parser.add_argument("--reference-start", type=int, default=1994)
    parser.add_argument("--reference-end", type=int, default=2010)
    parser.add_argument("--minimum-reference-years", type=int, default=10)
    parser.add_argument("--climatology-floor-mm", type=float, default=0.1)
    parser.add_argument("--start", default="1994-01-01")
    parser.add_argument("--end", default="2024-12-01")
    args = parser.parse_args()

    observations = build_station_month_table(
        args.csv,
        args.reference_start,
        args.reference_end,
        args.minimum_reference_years,
        args.climatology_floor_mm,
    )
    dates = pd.date_range(args.start, args.end, freq="MS")
    latitudes = np.arange(60.0, 0.0, -0.5, dtype=np.float32)
    longitudes = np.arange(70.0, 140.0, 0.5, dtype=np.float32)
    precipitation, anomaly, station_counts = interpolate_months(
        observations, dates, latitudes, longitudes
    )
    mask = np.all(np.isfinite(anomaly), axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        dates=np.asarray(dates.strftime("%Y-%m-%d"), dtype="<U10"),
        latitudes=latitudes,
        longitudes=longitudes,
        precipitation_mm=precipitation,
        anomaly_fraction=anomaly,
        valid_mask=mask,
        station_counts=station_counts,
    )

    excluded = set(pd.to_datetime(DEFAULT_EXCLUDED))
    aligned = (dates <= pd.Timestamp("2024-09-01")) & ~dates.isin(excluded)
    compatible_path = args.output.with_name(args.output.stem + "_aligned.npy")
    np.save(compatible_path, anomaly[aligned])
    metadata = {
        "source": str(args.csv),
        "reference_period": [args.reference_start, args.reference_end],
        "minimum_reference_years": args.minimum_reference_years,
        "climatology_floor_mm": args.climatology_floor_mm,
        "date_count": len(dates),
        "aligned_date_count": int(aligned.sum()),
        "valid_grid_cells": int(mask.sum()),
        "station_count_range": [int(station_counts.min()), int(station_counts.max())],
        "coordinate_encoding": "CMA degree-minute converted to decimal degrees",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved {args.output}")
    print(f"Saved trainer-compatible target {compatible_path}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
