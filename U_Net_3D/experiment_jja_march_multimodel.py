"""Evaluate March-initialized ECMWF/NCEP/BCC forecasts for JJA rainfall.

The blend is selected only from rolling 2011-2020 forecasts.  Years
2021-2024 are held out for the final comparison.
"""

from __future__ import annotations

import argparse
import calendar
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from project_paths import MODEL_DATA_DIR, OBSERVATION_FILE, experiment_path

MODEL_ROOT = MODEL_DATA_DIR
OBS_PATH = OBSERVATION_FILE
TRAIN_YEARS = tuple(range(2001, 2021))
OOF_YEARS = tuple(range(2011, 2021))
TEST_YEARS = tuple(range(2021, 2025))


def issue_year(path: Path) -> int:
    match = re.search(r"((?:19|20)\d{2})03", path.name)
    if match is None:
        raise ValueError(f"No March issue year in {path.name}")
    return int(match.group(1))


def collect_march_files(model: str) -> dict[int, Path]:
    if model == "ECMWF":
        paths = (MODEL_ROOT / "MODESv21_ecmwf_seas51").glob("*.nc")
        return {
            issue_year(path): path
            for path in paths
            if re.search(r"\d{4}03_monthly", path.name)
        }
    if model == "NCEP":
        result = {}
        for directory in ("MODESv2_ncep_cfs2", "MODESv21_ncep_cfs2"):
            paths = (MODEL_ROOT / directory).glob("*.nc")
            result.update(
                {
                    issue_year(path): path
                    for path in paths
                    if re.search(r"\d{4}03_monthly", path.name)
                }
            )
        return result
    if model == "BCC":
        paths = (MODEL_ROOT / "BCC-CPSV3" / "cpsv3").glob("*PRECT.nc")
        return {issue_year(path): path for path in paths}
    raise ValueError(model)


def month_number(value) -> int:
    if hasattr(value, "month"):
        return int(value.month)
    return int(pd.Timestamp(value).month)


def read_jja_total(
    path: Path,
    model: str,
    target_latitudes: np.ndarray,
    target_longitudes: np.ndarray,
) -> np.ndarray:
    with xr.open_dataset(path) as dataset:
        if "PRECT" in dataset:
            variable = "PRECT"
            rate_to_mm_day = 86_400.0 * 1_000.0
        elif "tp" in dataset:
            variable = "tp"
            rate_to_mm_day = 86_400.0 * 1_000.0
        elif "precsfc" in dataset:
            variable = "precsfc"
            rate_to_mm_day = 1.0
        else:
            raise KeyError(f"No supported precipitation variable in {path.name}")
        values = dataset[variable]
        indices = [
            index
            for index, date in enumerate(values["time"].values)
            if month_number(date) in (6, 7, 8)
        ]
        if len(indices) != 3:
            raise ValueError(f"Expected June-August in {path.name}, found {len(indices)}")
        days = np.asarray(
            [calendar.monthrange(issue_year(path), month)[1] for month in (6, 7, 8)],
            dtype=np.float64,
        )
        total = (values.isel(time=indices) * days[:, None, None]).sum("time")
        total = total * rate_to_mm_day
        latitude_name = "latitude" if "latitude" in total.coords else "lat"
        longitude_name = "longitude" if "longitude" in total.coords else "lon"
        interpolated = total.interp(
            {
                latitude_name: xr.DataArray(target_latitudes, dims=latitude_name),
                longitude_name: xr.DataArray(target_longitudes, dims=longitude_name),
            },
            method="linear",
        )
        result = np.asarray(interpolated.values, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"Non-finite interpolated values in {path.name}")
    return result


def load_observation_totals(path: Path) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        dates = pd.to_datetime(data["dates"].astype(str))
        precipitation = np.asarray(data["precipitation_mm"], dtype=np.float32)
        mask = np.asarray(data["valid_mask"], dtype=bool)
        latitudes = np.asarray(data["latitudes"], dtype=np.float32)
        longitudes = np.asarray(data["longitudes"], dtype=np.float32)
    result = {}
    for year in range(min(TRAIN_YEARS), max(TEST_YEARS) + 1):
        indices = np.where((dates.year == year) & np.isin(dates.month, (6, 7, 8)))[0]
        if len(indices) == 3:
            result[year] = precipitation[indices].sum(axis=0)
    return result, mask, latitudes, longitudes


def signed_log_anomaly(field: np.ndarray, climatology: np.ndarray) -> np.ndarray:
    fraction = (field - climatology) / (climatology + 1e-6)
    return np.sign(fraction) * np.log1p(np.abs(fraction))


def weighted_pattern(field: np.ndarray, mask: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(field[mask], dtype=np.float64)
    point_weights = weights[mask]
    mean = np.sum(values * point_weights) / np.sum(point_weights)
    centered = values - mean
    variance = np.sum(point_weights * centered**2) / np.sum(point_weights)
    return (centered / np.sqrt(variance + 1e-12)).astype(np.float32)


def pattern_for_year(
    fields: dict[int, np.ndarray],
    year: int,
    climatology_years: list[int] | tuple[int, ...],
    mask: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    climatology = np.mean([fields[item] for item in climatology_years], axis=0)
    return weighted_pattern(
        signed_log_anomaly(fields[year], climatology), mask, weights
    )


def weighted_acc(prediction: np.ndarray, target: np.ndarray, point_weights: np.ndarray) -> float:
    numerator = np.sum(point_weights * prediction * target)
    denominator = np.sqrt(
        np.sum(point_weights * prediction**2) * np.sum(point_weights * target**2)
    )
    return float(numerator / denominator) if denominator > 0 else 0.0


def simplex_weights(step: float = 0.05) -> np.ndarray:
    units = int(round(1.0 / step))
    rows = []
    for ec, ncep in itertools.product(range(units + 1), repeat=2):
        if ec + ncep > units:
            continue
        rows.append((ec, ncep, units - ec - ncep))
    return np.asarray(rows, dtype=np.float32) / units


def evaluate(output_prefix: Path) -> tuple[pd.DataFrame, dict]:
    observations, mask, latitudes, longitudes = load_observation_totals(OBS_PATH)
    area = np.cos(np.deg2rad(latitudes))[:, None] * np.ones((1, len(longitudes)))
    point_weights = area[mask].astype(np.float64)

    model_fields = {}
    required_years = set(TRAIN_YEARS) | set(TEST_YEARS)
    for model in ("ECMWF", "NCEP", "BCC"):
        files = collect_march_files(model)
        missing = sorted(required_years - set(files))
        if missing:
            raise ValueError(f"{model} is missing March forecasts for {missing}")
        model_fields[model] = {
            year: read_jja_total(files[year], model, latitudes, longitudes)
            for year in sorted(required_years)
        }

    oof_sources = []
    oof_targets = []
    for year in OOF_YEARS:
        history = list(range(min(TRAIN_YEARS), year))
        oof_sources.append(
            np.stack(
                [
                    pattern_for_year(model_fields[model], year, history, mask, area)
                    for model in ("ECMWF", "NCEP", "BCC")
                ]
            )
        )
        oof_targets.append(pattern_for_year(observations, year, history, mask, area))
    oof_sources = np.stack(oof_sources)
    oof_targets = np.stack(oof_targets)

    candidates = simplex_weights()
    candidate_scores = []
    for candidate in candidates:
        predictions = np.einsum("k,nkp->np", candidate, oof_sources)
        candidate_scores.append(
            np.mean(
                [
                    weighted_acc(predictions[i], oof_targets[i], point_weights)
                    for i in range(len(OOF_YEARS))
                ]
            )
        )
    selected = candidates[int(np.argmax(candidate_scores))]

    test_sources = np.stack(
        [
            np.stack(
                [
                    pattern_for_year(
                        model_fields[model], year, TRAIN_YEARS, mask, area
                    )
                    for model in ("ECMWF", "NCEP", "BCC")
                ]
            )
            for year in TEST_YEARS
        ]
    )
    test_targets = np.stack(
        [pattern_for_year(observations, year, TRAIN_YEARS, mask, area) for year in TEST_YEARS]
    )
    blended = np.einsum("k,nkp->np", selected, test_sources)

    rows = []
    labels = ("ECMWF", "NCEP", "BCC", "OOF_blend")
    predictions = (*[test_sources[:, index] for index in range(3)], blended)
    for label, forecast in zip(labels, predictions):
        annual_scores = [
            weighted_acc(forecast[index], test_targets[index], point_weights)
            for index in range(len(TEST_YEARS))
        ]
        rows.append(
            {
                "model": label,
                "mean_test_acc": float(np.mean(annual_scores)),
                **{
                    f"acc_{year}": score
                    for year, score in zip(TEST_YEARS, annual_scores)
                },
            }
        )
    table = pd.DataFrame(rows)
    metadata = {
        "train_years": [min(TRAIN_YEARS), max(TRAIN_YEARS)],
        "oof_years": list(OOF_YEARS),
        "test_years": list(TEST_YEARS),
        "source_order": ["ECMWF", "NCEP", "BCC"],
        "selected_weights": selected.tolist(),
        "selected_oof_acc": float(np.max(candidate_scores)),
        "ecmwf_oof_acc": float(candidate_scores[np.where((candidates == [1, 0, 0]).all(axis=1))[0][0]]),
        "target": "March-initialized JJA total precipitation signed-log fractional anomaly pattern",
    }
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_prefix.with_suffix(".csv"), index=False)
    output_prefix.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output_prefix.with_suffix(".npz"),
        years=np.asarray(TEST_YEARS),
        predictions=np.stack(predictions),
        observations=test_targets,
        selected_weights=selected,
        valid_mask=mask,
        latitudes=latitudes,
        longitudes=longitudes,
    )
    return table, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=experiment_path("jja_march_multimodel", "metrics"),
    )
    args = parser.parse_args()
    table, metadata = evaluate(args.output_prefix)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
