"""Evaluate multi-lead skill by season and broad China diagnostic region."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import OBSERVATION_FILE, paths, experiment_path

from evaluate_multilead_metrics import spatial_acc, transform_fractional_anomaly


SEASONS = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}

# Rectangles are diagnostics, not official Chinese climate-region boundaries.
REGIONS = {
    "Northwest": (35.0, 50.0, 73.0, 105.0),
    "North": (35.0, 42.0, 105.0, 120.0),
    "Northeast": (40.0, 54.0, 120.0, 135.0),
    "Yangtze": (27.0, 35.0, 105.0, 122.0),
    "South": (20.0, 27.0, 105.0, 123.0),
    "Southwest": (22.0, 35.0, 97.0, 105.0),
}


def subset_metrics(model, ec, obs, weights):
    model_acc = spatial_acc(model, obs, weights)
    ec_acc = spatial_acc(ec, obs, weights)
    normalized_weights = weights / weights.sum()
    model_mse = np.sum((model - obs) ** 2 * normalized_weights[None, :], axis=1)
    ec_mse = np.sum((ec - obs) ** 2 * normalized_weights[None, :], axis=1)
    return {
        "acc": float(model_acc.mean()),
        "ec_acc": float(ec_acc.mean()),
        "acc_gain": float((model_acc - ec_acc).mean()),
        "rmse": float(np.sqrt(model_mse.mean())),
        "ec_rmse": float(np.sqrt(ec_mse.mean())),
        "rmse_skill_pct": float(
            100.0 * (np.sqrt(ec_mse.mean()) - np.sqrt(model_mse.mean()))
            / np.sqrt(ec_mse.mean())
        ),
    }


def evaluate(data_dir, prediction_file, observation_file, test_months=21):
    model_all = np.load(prediction_file, mmap_mode="r")
    ec_all = np.load(
        data_dir / "multi_lead_ec_precip_anom_results.npy", mmap_mode="r"
    )
    production_dates = np.load(data_dir / "multi_lead_dates.npy").astype(str)
    with np.load(observation_file) as data:
        observations = transform_fractional_anomaly(
            np.asarray(data["anomaly_fraction"], dtype=np.float64), "signed_log1p"
        )
        observation_dates = data["dates"].astype(str)
        valid_mask = np.asarray(data["valid_mask"], dtype=bool)
        latitudes = np.asarray(data["latitudes"], dtype=np.float64)
        longitudes = np.asarray(data["longitudes"], dtype=np.float64)
    observation_index = {date: index for index, date in enumerate(observation_dates)}
    latitude_grid, longitude_grid = np.meshgrid(
        latitudes, longitudes, indexing="ij"
    )
    area = np.cos(np.deg2rad(latitude_grid))

    scopes = [("national", "China", valid_mask)]
    for name, (lat_min, lat_max, lon_min, lon_max) in REGIONS.items():
        region_mask = (
            valid_mask
            & (latitude_grid >= lat_min)
            & (latitude_grid < lat_max)
            & (longitude_grid >= lon_min)
            & (longitude_grid < lon_max)
        )
        scopes.append(("region", name, region_mask))

    rows = []
    for lead in range(model_all.shape[1]):
        valid_indices = np.flatnonzero(np.isfinite(ec_all[:, lead, 0, 0]))
        indices = valid_indices[-test_months:]
        dates = pd.to_datetime(production_dates[indices])
        obs_indices = [observation_index[date] for date in production_dates[indices]]
        obs_fields = observations[obs_indices]
        model_fields = np.asarray(model_all[indices, lead], dtype=np.float64)
        ec_fields = np.asarray(ec_all[indices, lead], dtype=np.float64)

        for scope_type, scope, spatial_mask in scopes:
            if spatial_mask.sum() < 20:
                continue
            temporal_groups = {"ALL": np.ones(len(indices), dtype=bool)}
            if scope_type == "national":
                temporal_groups.update(
                    {
                        season: np.isin(dates.month, months)
                        for season, months in SEASONS.items()
                    }
                )
            for period, temporal_mask in temporal_groups.items():
                if temporal_mask.sum() < 2:
                    continue
                metrics = subset_metrics(
                    model_fields[temporal_mask][:, spatial_mask],
                    ec_fields[temporal_mask][:, spatial_mask],
                    obs_fields[temporal_mask][:, spatial_mask],
                    area[spatial_mask],
                )
                rows.append(
                    {
                        "scope_type": scope_type,
                        "scope": scope,
                        "period": period,
                        "lead": lead,
                        "months": int(temporal_mask.sum()),
                        "grid_cells": int(spatial_mask.sum()),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=paths.results_dir / "signed_log_run"
    )
    parser.add_argument(
        "--prediction-file",
        type=Path,
        default=paths.results_dir
        / "signed_log_run"
        / "multi_lead_predict_results_ensemble_safe.npy",
    )
    parser.add_argument("--observation-file", type=Path, default=OBSERVATION_FILE)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=experiment_path("stratified_evaluation", "metrics.csv"),
    )
    parser.add_argument("--test-months", type=int, default=21)
    args = parser.parse_args()
    table = evaluate(
        args.data_dir,
        args.prediction_file,
        args.observation_file,
        args.test_months,
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_file, index=False)
    national_seasons = table[
        (table.scope_type == "national") & (table.period != "ALL")
    ]
    print(
        national_seasons.pivot(
            index="lead", columns="period", values="acc"
        ).to_string(float_format=lambda value: f"{value:.3f}")
    )


if __name__ == "__main__":
    main()
