"""Evaluate deterministic multi-lead precipitation-anomaly forecasts.

The target and forecasts are fractional monthly precipitation anomalies, so
RMSE/MAE values are dimensionless (multiply by 100 for percentage points).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import OBSERVATION_FILE, paths, experiment_path


def weighted_mean(values, weights, axis=None):
    return np.sum(values * weights, axis=axis) / np.sum(weights, axis=axis)


def spatial_acc(fields, target, weights):
    weights = weights / weights.sum()
    field_mean = np.sum(fields * weights[None, :], axis=1, keepdims=True)
    target_mean = np.sum(target * weights[None, :], axis=1, keepdims=True)
    field_anom = fields - field_mean
    target_anom = target - target_mean
    covariance = np.sum(weights[None, :] * field_anom * target_anom, axis=1)
    variance_product = (
        np.sum(weights[None, :] * field_anom**2, axis=1)
        * np.sum(weights[None, :] * target_anom**2, axis=1)
    )
    return covariance / np.sqrt(np.maximum(variance_product, 1e-20))


def temporal_correlation(fields, target, weights):
    field_anom = fields - fields.mean(axis=0, keepdims=True)
    target_anom = target - target.mean(axis=0, keepdims=True)
    numerator = np.sum(field_anom * target_anom, axis=0)
    denominator = np.sqrt(
        np.sum(field_anom**2, axis=0) * np.sum(target_anom**2, axis=0)
    )
    valid = denominator > 1e-12
    correlations = numerator[valid] / denominator[valid]
    valid_weights = weights[valid]
    # Fisher-z averaging is less biased than directly averaging correlations.
    fisher_z = np.arctanh(np.clip(correlations, -0.999999, 0.999999))
    fisher_mean = np.tanh(np.average(fisher_z, weights=valid_weights))
    return float(fisher_mean), float(np.median(correlations))


def willmott_index(fields, target, weights):
    target_mean = np.average(target, weights=np.broadcast_to(weights, target.shape))
    numerator = np.sum((fields - target) ** 2 * weights[None, :])
    denominator = np.sum(
        (np.abs(fields - target_mean) + np.abs(target - target_mean)) ** 2
        * weights[None, :]
    )
    return float(1.0 - numerator / max(denominator, 1e-20))


def critical_success_index(forecast_event, observed_event, weights):
    weights_2d = np.broadcast_to(weights, forecast_event.shape)
    hits = np.sum(weights_2d * (forecast_event & observed_event))
    misses = np.sum(weights_2d * (~forecast_event & observed_event))
    false_alarms = np.sum(weights_2d * (forecast_event & ~observed_event))
    return float(hits / max(hits + misses + false_alarms, 1e-20))


def event_frequency(events, weights):
    return float(np.average(events, weights=np.broadcast_to(weights, events.shape)))


def moving_block_indices(rng, sample_count, block_length):
    block_count = int(np.ceil(sample_count / block_length))
    starts = rng.integers(0, sample_count, size=block_count)
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % sample_count).ravel()[:sample_count]


def transform_fractional_anomaly(values, transform):
    if transform == "none":
        return values
    if transform == "signed_log1p":
        return np.sign(values) * np.log1p(np.abs(values))
    raise ValueError(f"Unknown observation transform: {transform}")


def bootstrap_intervals(monthly_by_lead, iterations, seed, block_length=3):
    rng = np.random.default_rng(seed)
    lead_count, sample_count = monthly_by_lead["acc_gain"].shape
    draws = {name: np.empty((iterations, lead_count)) for name in monthly_by_lead}
    macro_acc_gain = np.empty(iterations)
    macro_msess = np.empty(iterations)

    for iteration in range(iterations):
        indices = moving_block_indices(rng, sample_count, block_length)
        for name, values in monthly_by_lead.items():
            if name == "msess":
                model_mse, ec_mse = values
                draws[name][iteration] = 1.0 - (
                    model_mse[:, indices].mean(axis=1)
                    / ec_mse[:, indices].mean(axis=1)
                )
            else:
                draws[name][iteration] = values[:, indices].mean(axis=1)
        macro_acc_gain[iteration] = draws["acc_gain"][iteration].mean()
        model_mse, ec_mse = monthly_by_lead["msess"]
        macro_msess[iteration] = 1.0 - (
            model_mse[:, indices].mean() / ec_mse[:, indices].mean()
        )

    intervals = {}
    for lead in range(lead_count):
        intervals[str(lead)] = {
            "acc_gain_ci95": np.quantile(
                draws["acc_gain"][:, lead], [0.025, 0.975]
            ).tolist(),
            "acc_gain_probability_positive": float(
                np.mean(draws["acc_gain"][:, lead] > 0.0)
            ),
            "msess_ci95": np.quantile(
                draws["msess"][:, lead], [0.025, 0.975]
            ).tolist(),
        }
    intervals["macro"] = {
        "acc_gain_ci95": np.quantile(macro_acc_gain, [0.025, 0.975]).tolist(),
        "acc_gain_probability_positive": float(np.mean(macro_acc_gain > 0.0)),
        "msess_ci95": np.quantile(macro_msess, [0.025, 0.975]).tolist(),
        "msess_probability_positive": float(np.mean(macro_msess > 0.0)),
    }
    return intervals


def evaluate(
    data_dir,
    prediction_file,
    observation_file,
    test_months,
    bootstrap_iterations,
    seed,
    observation_transform="none",
):
    ec_all = np.load(
        data_dir / "multi_lead_ec_precip_anom_results.npy", mmap_mode="r"
    )
    model_all = np.load(prediction_file, mmap_mode="r")
    dates = np.load(data_dir / "multi_lead_dates.npy")
    external_observations = None
    external_date_to_index = None
    if observation_file is None:
        obs_all = np.load(data_dir / "multi_lead_obs_results.npy", mmap_mode="r")
        raw_obs = np.load(data_dir / "hr_unet" / "hr_data.npy", mmap_mode="r")
        # The trainer historically replaced intermittent missing targets by zero.
        # Evaluate only cells observed throughout the outer test window instead.
        land_mask = np.all(np.isfinite(raw_obs[-test_months:]), axis=0)
    else:
        obs_all = None
        with np.load(observation_file) as observation_data:
            external_observations = observation_data["anomaly_fraction"]
            external_dates = observation_data["dates"].astype(str)
            land_mask = observation_data["valid_mask"].astype(bool)
        external_observations = transform_fractional_anomaly(
            external_observations, observation_transform
        )
        external_date_to_index = {
            date: index for index, date in enumerate(external_dates)
        }

    latitudes = np.arange(60.0, 0.0, -0.5)
    area_weights_2d = np.cos(np.deg2rad(latitudes))[:, None] * np.ones((1, 140))
    area_weights = area_weights_2d[land_mask]
    area_weights = area_weights / area_weights.sum()

    rows = []
    monthly_acc_gain = []
    monthly_model_mse = []
    monthly_ec_mse = []

    for lead in range(model_all.shape[1]):
        valid_dates = np.flatnonzero(np.isfinite(ec_all[:, lead, 0, 0]))
        test_indices = valid_dates[-test_months:]
        if external_observations is None:
            obs = np.asarray(
                obs_all[test_indices, lead][:, land_mask], dtype=np.float64
            )
        else:
            missing = [
                str(dates[index])
                for index in test_indices
                if str(dates[index]) not in external_date_to_index
            ]
            if missing:
                raise ValueError(f"Observation file is missing dates: {missing}")
            observation_indices = [
                external_date_to_index[str(dates[index])] for index in test_indices
            ]
            obs = np.asarray(
                external_observations[observation_indices][:, land_mask],
                dtype=np.float64,
            )
        model = np.asarray(model_all[test_indices, lead][:, land_mask], dtype=np.float64)
        ec = np.asarray(ec_all[test_indices, lead][:, land_mask], dtype=np.float64)

        model_error = model - obs
        ec_error = ec - obs
        model_mse_by_month = np.sum(model_error**2 * area_weights[None, :], axis=1)
        ec_mse_by_month = np.sum(ec_error**2 * area_weights[None, :], axis=1)
        model_mae_by_month = np.sum(np.abs(model_error) * area_weights[None, :], axis=1)
        ec_mae_by_month = np.sum(np.abs(ec_error) * area_weights[None, :], axis=1)

        model_acc_by_month = spatial_acc(model, obs, area_weights)
        ec_acc_by_month = spatial_acc(ec, obs, area_weights)
        model_tcc, model_tcc_median = temporal_correlation(model, obs, area_weights)
        ec_tcc, ec_tcc_median = temporal_correlation(ec, obs, area_weights)

        model_centered = model - np.sum(
            model * area_weights[None, :], axis=1, keepdims=True
        )
        obs_centered = obs - np.sum(
            obs * area_weights[None, :], axis=1, keepdims=True
        )
        ec_centered = ec - np.sum(
            ec * area_weights[None, :], axis=1, keepdims=True
        )
        model_spatial_spread = float(
            np.mean(np.sqrt(np.sum(model_centered**2 * area_weights[None, :], axis=1)))
        )
        ec_spatial_spread = float(
            np.mean(np.sqrt(np.sum(ec_centered**2 * area_weights[None, :], axis=1)))
        )
        obs_spatial_spread = float(
            np.mean(np.sqrt(np.sum(obs_centered**2 * area_weights[None, :], axis=1)))
        )
        model_grid_rmse = np.sqrt(np.mean(model_error**2, axis=0))
        ec_grid_rmse = np.sqrt(np.mean(ec_error**2, axis=0))
        model_rmse = float(np.sqrt(model_mse_by_month.mean()))
        ec_rmse = float(np.sqrt(ec_mse_by_month.mean()))
        model_mae = float(model_mae_by_month.mean())
        ec_mae = float(ec_mae_by_month.mean())

        rows.append(
            {
                "lead": lead,
                "test_start": str(dates[test_indices[0]]),
                "test_end": str(dates[test_indices[-1]]),
                "spatial_acc": float(model_acc_by_month.mean()),
                "ec_spatial_acc": float(ec_acc_by_month.mean()),
                "acc_gain": float((model_acc_by_month - ec_acc_by_month).mean()),
                "pooled_rmse": model_rmse,
                "ec_pooled_rmse": ec_rmse,
                "rmse_skill_pct": 100.0 * (ec_rmse - model_rmse) / ec_rmse,
                "msess": 1.0 - model_mse_by_month.mean() / ec_mse_by_month.mean(),
                "mean_grid_rmse": float(np.average(model_grid_rmse, weights=area_weights)),
                "ec_mean_grid_rmse": float(np.average(ec_grid_rmse, weights=area_weights)),
                "mae": model_mae,
                "ec_mae": ec_mae,
                "mae_skill_pct": 100.0 * (ec_mae - model_mae) / ec_mae,
                "bias": float(np.mean(np.sum(model_error * area_weights[None, :], axis=1))),
                "ec_bias": float(np.mean(np.sum(ec_error * area_weights[None, :], axis=1))),
                "centered_rmse": float(
                    np.sqrt(np.mean(np.sum((model_centered - obs_centered) ** 2 * area_weights[None, :], axis=1)))
                ),
                "spatial_spread": model_spatial_spread,
                "ec_spatial_spread": ec_spatial_spread,
                "obs_spatial_spread": obs_spatial_spread,
                "spread_ratio": model_spatial_spread / obs_spatial_spread,
                "ec_spread_ratio": ec_spatial_spread / obs_spatial_spread,
                "willmott_index": willmott_index(model, obs, area_weights),
                "ec_willmott_index": willmott_index(ec, obs, area_weights),
                "tcc_fisher_mean": model_tcc,
                "ec_tcc_fisher_mean": ec_tcc,
                "tcc_median": model_tcc_median,
                "ec_tcc_median": ec_tcc_median,
                "wet_csi_gt_0": critical_success_index(model > 0.0, obs > 0.0, area_weights),
                "ec_wet_csi_gt_0": critical_success_index(ec > 0.0, obs > 0.0, area_weights),
                "wet_frequency": event_frequency(model > 0.0, area_weights),
                "ec_wet_frequency": event_frequency(ec > 0.0, area_weights),
                "obs_wet_frequency": event_frequency(obs > 0.0, area_weights),
                "strong_wet_csi_gt_1": critical_success_index(model > 1.0, obs > 1.0, area_weights),
                "ec_strong_wet_csi_gt_1": critical_success_index(ec > 1.0, obs > 1.0, area_weights),
                "strong_wet_frequency": event_frequency(model > 1.0, area_weights),
                "ec_strong_wet_frequency": event_frequency(ec > 1.0, area_weights),
                "obs_strong_wet_frequency": event_frequency(obs > 1.0, area_weights),
                "dry_csi_lt_m0_5": critical_success_index(model < -0.5, obs < -0.5, area_weights),
                "ec_dry_csi_lt_m0_5": critical_success_index(ec < -0.5, obs < -0.5, area_weights),
                "dry_frequency": event_frequency(model < -0.5, area_weights),
                "ec_dry_frequency": event_frequency(ec < -0.5, area_weights),
                "obs_dry_frequency": event_frequency(obs < -0.5, area_weights),
            }
        )
        monthly_acc_gain.append(model_acc_by_month - ec_acc_by_month)
        monthly_model_mse.append(model_mse_by_month)
        monthly_ec_mse.append(ec_mse_by_month)

    metrics = pd.DataFrame(rows)
    intervals = bootstrap_intervals(
        {
            "acc_gain": np.asarray(monthly_acc_gain),
            "msess": (
                np.asarray(monthly_model_mse),
                np.asarray(monthly_ec_mse),
            ),
        },
        iterations=bootstrap_iterations,
        seed=seed,
    )
    return metrics, intervals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=paths.results_dir / "ams_multilead"
    )
    parser.add_argument("--test-months", type=int, default=21)
    parser.add_argument(
        "--prediction-file",
        type=Path,
        default=OBSERVATION_FILE,
        help="Forecast array to evaluate (default: DATA_DIR/multi_lead_predict_results.npy)",
    )
    parser.add_argument(
        "--observation-file",
        type=Path,
        default=None,
        help="Dated observation NPZ from rebuild_station_observations.py",
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--observation-transform",
        choices=("none", "signed_log1p"),
        default="none",
        help="Apply the same fixed transform used by the forecast trainer",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=experiment_path("evaluation", "multilead_evaluation_metrics"),
    )
    args = parser.parse_args()

    prediction_file = args.prediction_file or (
        args.data_dir / "multi_lead_predict_results.npy"
    )
    metrics, intervals = evaluate(
        args.data_dir,
        prediction_file,
        args.observation_file,
        args.test_months,
        args.bootstrap,
        args.seed,
        args.observation_transform,
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_prefix.with_suffix(".csv"), index=False)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(intervals, indent=2), encoding="utf-8"
    )
    columns = [
        "lead", "spatial_acc", "ec_spatial_acc", "acc_gain", "pooled_rmse",
        "ec_pooled_rmse", "rmse_skill_pct", "mae", "ec_mae",
        "tcc_fisher_mean", "ec_tcc_fisher_mean", "willmott_index",
        "ec_willmott_index",
    ]
    print(metrics[columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nBootstrap intervals:")
    print(json.dumps(intervals, indent=2))


if __name__ == "__main__":
    main()
