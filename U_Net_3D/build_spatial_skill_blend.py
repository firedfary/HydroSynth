"""Blend an AMS forecast with ECMWF using nested-OOF spatial skill weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from project_paths import OBSERVATION_FILE, paths, experiment_path

from build_fixed_ensemble import (
    spatial_acc_vectors,
    standardize_vectors,
    transform_fractional_anomaly,
)


PENALTIES = (
    0.0,
    1.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
    200.0,
    500.0,
    1_000.0,
    5_000.0,
    1e12,  # Numerically equivalent to the model-only limiting case.
)


def fit_local_weights(
    model: np.ndarray,
    ec: np.ndarray,
    target: np.ndarray,
    penalty: float,
    prior: float = 1.0,
) -> np.ndarray:
    difference = model - ec
    numerator = np.sum(difference * (target - ec), axis=0) + penalty * prior
    denominator = np.sum(difference**2, axis=0) + penalty
    weights = np.divide(
        numerator,
        denominator,
        out=np.full(model.shape[1], prior, dtype=np.float64),
        where=denominator > 1e-12,
    )
    return np.clip(weights, 0.0, 1.0)


def smooth_masked_weights(
    weights: np.ndarray,
    mask: np.ndarray,
    sigma: float,
) -> np.ndarray:
    field = np.zeros(mask.shape, dtype=np.float64)
    field[mask] = weights
    numerator = gaussian_filter(field, sigma=sigma, mode="nearest")
    denominator = gaussian_filter(mask.astype(np.float64), sigma=sigma, mode="nearest")
    result = np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=denominator > 1e-8,
    )
    return np.clip(result[mask], 0.0, 1.0)


def select_penalty(
    model: np.ndarray,
    ec: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    area_weights: np.ndarray,
    sigma: float,
) -> tuple[float, dict]:
    scores = {penalty: [] for penalty in PENALTIES}
    # Each validation block is later than all data used to estimate its map.
    for train_end in (42, 63, 84):
        validation = slice(train_end, train_end + 21)
        for penalty in PENALTIES:
            local_weights = fit_local_weights(
                model[:train_end], ec[:train_end], target[:train_end], penalty
            )
            local_weights = smooth_masked_weights(local_weights, mask, sigma)
            prediction = (
                local_weights[None, :] * model[validation]
                + (1.0 - local_weights[None, :]) * ec[validation]
            )
            prediction = standardize_vectors(prediction)
            scores[penalty].append(
                float(
                    np.mean(
                        spatial_acc_vectors(
                            prediction, target[validation], area_weights
                        )
                    )
                )
            )
    fold_weights = np.asarray([1.0, 4.0, 9.0])
    fold_weights /= fold_weights.sum()
    weighted_scores = {
        penalty: float(np.average(values, weights=fold_weights))
        for penalty, values in scores.items()
    }
    best = max(weighted_scores, key=weighted_scores.get)
    return float(best), {
        "selected_penalty": float(best),
        "weighted_validation_acc": weighted_scores[best],
        "scores": {str(key): value for key, value in scores.items()},
    }


def build_spatial_blend(
    data_dir: Path,
    prediction_file: Path | None,
    oof_file: Path,
    observation_file: Path,
    output_file: Path,
    sigma: float,
    observation_transform: str = "none",
) -> None:
    dates = np.load(data_dir / "multi_lead_dates.npy").astype(str)
    date_index = {date: index for index, date in enumerate(dates)}
    model_all = np.load(
        prediction_file or data_dir / "multi_lead_predict_results.npy",
        mmap_mode="r",
    )
    ec_all = np.load(
        data_dir / "multi_lead_ec_precip_anom_results.npy", mmap_mode="r"
    )
    with np.load(oof_file) as oof:
        oof_dates = oof["dates"].astype(str)
        model_oof = np.asarray(oof["predictions"], dtype=np.float64)
    with np.load(observation_file) as observations:
        target_all = transform_fractional_anomaly(
            observations["anomaly_fraction"], observation_transform
        )
        target_dates = observations["dates"].astype(str)
        mask = observations["valid_mask"].astype(bool)
    target_index = {date: index for index, date in enumerate(target_dates)}
    latitudes = np.arange(60.0, 0.0, -0.5)
    area_weights = (
        np.cos(np.deg2rad(latitudes))[:, None] * np.ones((1, mask.shape[1]))
    )[mask]
    area_weights /= area_weights.sum()

    result = np.asarray(model_all).copy()
    diagnostics = {}
    for lead in range(model_all.shape[1]):
        lead_dates = oof_dates[:, lead]
        production_indices = np.asarray([date_index[date] for date in lead_dates])
        observation_indices = np.asarray([target_index[date] for date in lead_dates])
        ec_oof = standardize_vectors(
            np.asarray(ec_all[production_indices, lead][:, mask], dtype=np.float64)
        )
        target_oof = standardize_vectors(
            np.asarray(target_all[observation_indices][:, mask], dtype=np.float64)
        )
        lead_model_oof = standardize_vectors(model_oof[:, lead])
        penalty, lead_diagnostics = select_penalty(
            lead_model_oof,
            ec_oof,
            target_oof,
            mask,
            area_weights,
            sigma,
        )
        local_weights = fit_local_weights(
            lead_model_oof, ec_oof, target_oof, penalty
        )
        local_weights = smooth_masked_weights(local_weights, mask, sigma)
        valid_rows = np.flatnonzero(np.isfinite(ec_all[:, lead, 0, 0]))
        model = np.asarray(model_all[valid_rows, lead][:, mask], dtype=np.float64)
        ec = np.asarray(ec_all[valid_rows, lead][:, mask], dtype=np.float64)
        model_patterns = standardize_vectors(model)
        ec_patterns = standardize_vectors(ec)
        blended = standardize_vectors(
            local_weights[None, :] * model_patterns
            + (1.0 - local_weights[None, :]) * ec_patterns
        )
        ec_means = ec.mean(axis=1, keepdims=True)
        ec_scales = ec.std(axis=1, keepdims=True)
        restored = blended * np.maximum(ec_scales, 1e-8) + ec_means
        fields = np.zeros((len(valid_rows), *mask.shape), dtype=np.float32)
        fields[:, mask] = restored.astype(np.float32)
        result[valid_rows, lead] = fields
        lead_diagnostics.update(
            {
                "mean_model_weight": float(np.average(local_weights, weights=area_weights)),
                "min_model_weight": float(local_weights.min()),
                "max_model_weight": float(local_weights.max()),
            }
        )
        diagnostics[str(lead)] = lead_diagnostics

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, result)
    output_file.with_suffix(".json").write_text(
        json.dumps(
            {
                "smoothing_sigma": sigma,
                "observation_transform": observation_transform,
                "prediction_file": str(
                    prediction_file or data_dir / "multi_lead_predict_results.npy"
                ),
                "leads": diagnostics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved spatial skill blend to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=paths.results_dir / "signed_log_run"
    )
    parser.add_argument("--prediction-file", type=Path, default=None)
    parser.add_argument(
        "--oof-file",
        type=Path,
        default=paths.results_dir / "signed_log_run" / "ams_oof_patterns.npz",
    )
    parser.add_argument("--observation-file", type=Path, default=OBSERVATION_FILE)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=experiment_path(
            "spatial_skill_blend", "multi_lead_predict_results_spatial.npy"
        ),
    )
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument(
        "--observation-transform",
        choices=("none", "signed_log1p"),
        default="none",
    )
    args = parser.parse_args()
    build_spatial_blend(
        args.data_dir,
        args.prediction_file,
        args.oof_file,
        args.observation_file,
        args.output_file,
        args.sigma,
        args.observation_transform,
    )


if __name__ == "__main__":
    main()
