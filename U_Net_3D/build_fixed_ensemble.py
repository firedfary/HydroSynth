"""Build an OOF-selected AMS + seasonal-stacking multi-lead product.

Seasonal stacking stores unit-variance spatial patterns, so the AMS pattern
is put on the same scale before blending.  Weights can be fixed or selected
separately by lead from rolling out-of-fold months.  The final field is either
restored with the ECMWF amplitude convention or calibrated from OOF targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from project_paths import OBSERVATION_FILE, paths, experiment_path


SEASON_BY_MONTH = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


def transform_fractional_anomaly(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "none":
        return values
    if transform == "signed_log1p":
        return np.sign(values) * np.log1p(np.abs(values))
    raise ValueError(f"Unknown observation transform: {transform}")


def standardize_vectors(values: np.ndarray) -> np.ndarray:
    means = values.mean(axis=1, keepdims=True)
    scales = values.std(axis=1, keepdims=True)
    return (values - means) / np.maximum(scales, 1e-8)


def spatial_acc_vectors(
    predictions: np.ndarray,
    targets: np.ndarray,
    area_weights: np.ndarray,
) -> np.ndarray:
    prediction_means = predictions @ area_weights
    target_means = targets @ area_weights
    prediction_anomalies = predictions - prediction_means[:, None]
    target_anomalies = targets - target_means[:, None]
    numerator = np.sum(
        prediction_anomalies * target_anomalies * area_weights[None, :], axis=1
    )
    denominator = np.sqrt(
        np.sum(prediction_anomalies**2 * area_weights[None, :], axis=1)
        * np.sum(target_anomalies**2 * area_weights[None, :], axis=1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )


def select_oof_ams_weights(
    ams_predictions: np.ndarray,
    stack_predictions: np.ndarray,
    targets: np.ndarray,
    area_weights: np.ndarray,
    fold_size: int = 21,
) -> tuple[np.ndarray, list[dict]]:
    """Choose one coarse blend weight per lead from rolling OOF predictions."""
    if ams_predictions.shape != stack_predictions.shape or targets.shape != ams_predictions.shape:
        raise ValueError("AMS, stacking, and OOF target shapes must match")

    candidates = np.linspace(0.0, 1.0, 21)
    sample_count, lead_count, _ = ams_predictions.shape
    fold_indices = np.minimum(np.arange(sample_count) // fold_size, 4)
    sample_weights = (fold_indices + 1.0) ** 2
    sample_weights /= sample_weights.sum()
    selected = np.empty(lead_count, dtype=np.float64)
    diagnostics = []

    for lead in range(lead_count):
        ams_patterns = standardize_vectors(ams_predictions[:, lead])
        stack_patterns = standardize_vectors(stack_predictions[:, lead])
        candidate_scores = []
        for ams_weight in candidates:
            patterns = standardize_vectors(
                ams_weight * ams_patterns + (1.0 - ams_weight) * stack_patterns
            )
            monthly_acc = spatial_acc_vectors(
                patterns, targets[:, lead], area_weights
            )
            candidate_scores.append(float(np.sum(monthly_acc * sample_weights)))
        best_index = int(np.argmax(candidate_scores))
        selected[lead] = candidates[best_index]
        diagnostics.append(
            {
                "ams_weight": float(selected[lead]),
                "stack_weight": float(1.0 - selected[lead]),
                "weighted_oof_acc": candidate_scores[best_index],
                "ams_only_oof_acc": candidate_scores[-1],
                "stack_only_oof_acc": candidate_scores[0],
            }
        )
    return selected, diagnostics


def select_oof_ec_safety_seasons(
    blended_predictions: np.ndarray,
    ec_predictions: np.ndarray,
    targets: np.ndarray,
    oof_dates: np.ndarray,
    area_weights: np.ndarray,
    fold_size: int = 21,
) -> tuple[list[set[str]], list[dict]]:
    """Flag seasons whose blend fails to beat ECMWF in every OOF fold."""
    if blended_predictions.shape != ec_predictions.shape or targets.shape != ec_predictions.shape:
        raise ValueError("Blended, ECMWF, and target OOF shapes must match")
    sample_count, lead_count, _ = targets.shape
    fold_indices = np.arange(sample_count) // fold_size
    seasons = ("DJF", "MAM", "JJA", "SON")
    fallback_seasons = [set() for _ in range(lead_count)]
    diagnostics = []

    for lead in range(lead_count):
        blend_acc = spatial_acc_vectors(
            blended_predictions[:, lead], targets[:, lead], area_weights
        )
        ec_acc = spatial_acc_vectors(
            ec_predictions[:, lead], targets[:, lead], area_weights
        )
        month_seasons = np.asarray(
            [SEASON_BY_MONTH[int(str(date)[5:7])] for date in oof_dates[:, lead]]
        )
        for season in seasons:
            season_mask = month_seasons == season
            fold_gains = []
            for fold in np.unique(fold_indices):
                fold_mask = season_mask & (fold_indices == fold)
                if np.any(fold_mask):
                    fold_gains.append(float(np.mean(blend_acc[fold_mask] - ec_acc[fold_mask])))
            mean_gain = float(np.mean(blend_acc[season_mask] - ec_acc[season_mask]))
            mean_blend_acc = float(np.mean(blend_acc[season_mask]))
            mean_ec_acc = float(np.mean(ec_acc[season_mask]))
            positive_folds = int(np.sum(np.asarray(fold_gains) > 0.0))
            fallback = bool(
                fold_gains and positive_folds == 0 and mean_gain < -1e-12
            )
            if fallback:
                fallback_seasons[lead].add(season)
            diagnostics.append(
                {
                    "lead": lead,
                    "season": season,
                    "mean_oof_acc_gain": mean_gain,
                    "mean_oof_model_acc": mean_blend_acc,
                    "mean_oof_ec_acc": mean_ec_acc,
                    "positive_folds": positive_folds,
                    "fold_count": len(fold_gains),
                    "fallback_to_ecmwf": fallback,
                }
            )
    return fallback_seasons, diagnostics


def fit_oof_amplitude_calibration(
    patterns: np.ndarray,
    ec_vectors: np.ndarray,
    target_vectors: np.ndarray,
    area_weights: np.ndarray,
) -> tuple[float, float]:
    """Fit two nonnegative OOF coefficients for map mean and spatial pattern."""
    ec_area_means = ec_vectors @ area_weights
    target_area_means = target_vectors @ area_weights
    mean_denominator = float(np.dot(ec_area_means, ec_area_means))
    mean_coefficient = float(
        np.dot(ec_area_means, target_area_means) / max(mean_denominator, 1e-12)
    )

    centered_patterns = patterns - patterns @ area_weights[:, None]
    ec_scales = ec_vectors.std(axis=1)
    spatial_predictor = centered_patterns * ec_scales[:, None]
    centered_targets = target_vectors - target_area_means[:, None]
    spatial_numerator = np.sum(
        spatial_predictor * centered_targets * area_weights[None, :]
    )
    spatial_denominator = np.sum(
        spatial_predictor**2 * area_weights[None, :]
    )
    spatial_coefficient = float(
        spatial_numerator / max(spatial_denominator, 1e-12)
    )
    # Negative amplitudes reverse the forecast pattern.  Large extrapolations
    # are also unstable with only 105 independent validation months.
    return (
        float(np.clip(mean_coefficient, 0.0, 1.5)),
        float(np.clip(spatial_coefficient, 0.0, 1.5)),
    )


def restore_calibrated_patterns(
    patterns: np.ndarray,
    ec_vectors: np.ndarray,
    area_weights: np.ndarray,
    mean_coefficient: float,
    spatial_coefficient: float,
) -> np.ndarray:
    ec_area_means = ec_vectors @ area_weights
    ec_scales = ec_vectors.std(axis=1)
    centered_patterns = patterns - patterns @ area_weights[:, None]
    return (
        mean_coefficient * ec_area_means[:, None]
        + spatial_coefficient * ec_scales[:, None] * centered_patterns
    )


def build_ensemble(
    data_dir: Path,
    stack_file: Path,
    output_file: Path,
    ams_weight: float,
    ams_oof_file: Path | None = None,
    calibrate_amplitude: bool = False,
    observation_file: Path | None = None,
    select_weights_from_oof: bool = False,
    seasonal_ec_safety: bool = False,
    observation_transform: str = "none",
) -> Path:
    if not 0.0 <= ams_weight <= 1.0:
        raise ValueError("--ams-weight must be between 0 and 1")

    ams_path = data_dir / "multi_lead_predict_results.npy"
    ec_path = data_dir / "multi_lead_ec_precip_anom_results.npy"
    dates_path = data_dir / "multi_lead_dates.npy"

    ams = np.load(ams_path, mmap_mode="r")
    ec = np.load(ec_path, mmap_mode="r")
    production_dates = np.load(dates_path).astype(str)
    external_observations = None
    external_date_to_index = None
    if observation_file is None:
        raw_obs = np.load(data_dir / "hr_unet" / "hr_data.npy", mmap_mode="r")
        mask = np.isfinite(raw_obs[0])
    else:
        with np.load(observation_file) as observation_data:
            external_observations = np.asarray(
                observation_data["anomaly_fraction"], dtype=np.float64
            )
            external_dates = observation_data["dates"].astype(str)
            mask = observation_data["valid_mask"].astype(bool)
        external_date_to_index = {
            date: index for index, date in enumerate(external_dates)
        }
        external_observations = transform_fractional_anomaly(
            external_observations, observation_transform
        )
    latitudes = np.arange(60.0, 0.0, -0.5)
    area_weights_2d = np.cos(np.deg2rad(latitudes))[:, None] * np.ones((1, 140))
    area_weights = area_weights_2d[mask]
    area_weights /= area_weights.sum()

    with np.load(stack_file) as stack:
        stack_dates = stack["dates"].astype(str)
        stack_patterns = stack["predictions"]
        stack_oof_dates = stack["oof_dates"].astype(str)
        stack_oof_patterns = stack["oof_predictions"]
        if "valid_mask" in stack and not np.array_equal(stack["valid_mask"], mask):
            raise ValueError("Stacking and observation masks differ")

    expected_shape = (len(stack_dates), ams.shape[1], int(mask.sum()))
    if stack_patterns.shape != expected_shape:
        raise ValueError(
            f"Unexpected stacking shape {stack_patterns.shape}; expected {expected_shape}"
        )
    if ams.shape != ec.shape:
        raise ValueError(f"AMS and ECMWF shapes differ: {ams.shape} vs {ec.shape}")
    if len(np.unique(stack_dates)) != len(stack_dates):
        raise ValueError("Stacking test dates are not unique")

    date_to_index = {date: index for index, date in enumerate(production_dates)}
    missing_dates = sorted(set(stack_dates) - set(date_to_index))
    if missing_dates:
        raise ValueError(f"Stacking dates missing from production arrays: {missing_dates}")

    ams_oof = None
    if calibrate_amplitude or select_weights_from_oof or seasonal_ec_safety:
        if ams_oof_file is None:
            raise ValueError("OOF operation requested without --ams-oof-file")
        ams_oof = np.load(ams_oof_file)

    obs_all = np.load(data_dir / "multi_lead_obs_results.npy", mmap_mode="r")
    result = np.asarray(ams).copy()
    calibration_by_lead = {}
    ams_weights = np.full(ams.shape[1], ams_weight, dtype=np.float64)
    weight_diagnostics = []
    oof_targets = None
    oof_ec = None
    if select_weights_from_oof:
        if not np.array_equal(ams_oof["dates"].astype(str), stack_oof_dates):
            raise ValueError("AMS and stacking OOF dates differ")
        oof_targets = np.empty_like(ams_oof["predictions"], dtype=np.float64)
        oof_ec = np.empty_like(ams_oof["predictions"], dtype=np.float64)
        for lead in range(ams.shape[1]):
            lead_oof_dates = stack_oof_dates[:, lead]
            oof_indices = np.asarray([date_to_index[date] for date in lead_oof_dates])
            oof_ec[:, lead] = ec[oof_indices, lead][:, mask]
            if external_observations is None:
                oof_indices = np.asarray([date_to_index[date] for date in lead_oof_dates])
                oof_targets[:, lead] = obs_all[oof_indices, lead][:, mask]
            else:
                missing = [date for date in lead_oof_dates if date not in external_date_to_index]
                if missing:
                    raise ValueError(f"Observation file is missing OOF dates: {missing[:3]}")
                observation_indices = [external_date_to_index[date] for date in lead_oof_dates]
                oof_targets[:, lead] = external_observations[observation_indices][:, mask]
        ams_weights, weight_diagnostics = select_oof_ams_weights(
            np.asarray(ams_oof["predictions"], dtype=np.float64),
            np.asarray(stack_oof_patterns, dtype=np.float64),
            oof_targets,
            area_weights,
        )

    fallback_seasons = [set() for _ in range(ams.shape[1])]
    safety_diagnostics = []
    if seasonal_ec_safety:
        if not select_weights_from_oof:
            raise ValueError("--seasonal-ec-safety requires --select-weights-from-oof")
        blended_oof = np.empty_like(oof_targets)
        for lead in range(ams.shape[1]):
            blended_oof[:, lead] = standardize_vectors(
                ams_weights[lead]
                * standardize_vectors(ams_oof["predictions"][:, lead])
                + (1.0 - ams_weights[lead])
                * standardize_vectors(stack_oof_patterns[:, lead])
            )
        fallback_seasons, safety_diagnostics = select_oof_ec_safety_seasons(
            blended_oof,
            oof_ec,
            oof_targets,
            stack_oof_dates,
            area_weights,
        )

    for lead in range(ams.shape[1]):
        lead_ams_weight = float(ams_weights[lead])
        stack_weight = 1.0 - lead_ams_weight
        indices = np.asarray([date_to_index[date] for date in stack_dates])
        if not np.all(np.isfinite(ams[indices, lead][:, mask])):
            raise ValueError(f"AMS contains missing test values for Lead-{lead}")
        if not np.all(np.isfinite(ec[indices, lead][:, mask])):
            raise ValueError(f"ECMWF contains missing test values for Lead-{lead}")

        ams_vectors = np.asarray(ams[indices, lead][:, mask], dtype=np.float64)
        ec_vectors = np.asarray(ec[indices, lead][:, mask], dtype=np.float64)
        ams_patterns = standardize_vectors(ams_vectors)
        blended_patterns = (
            lead_ams_weight * ams_patterns
            + stack_weight * np.asarray(stack_patterns[:, lead], dtype=np.float64)
        )
        blended_patterns = standardize_vectors(blended_patterns)

        if calibrate_amplitude:
            oof_dates = ams_oof["dates"][:, lead].astype(str)
            if not np.array_equal(oof_dates, stack_oof_dates[:, lead]):
                raise ValueError(f"OOF date mismatch for Lead-{lead}")
            oof_indices = np.asarray([date_to_index[date] for date in oof_dates])
            oof_patterns = standardize_vectors(
                lead_ams_weight * standardize_vectors(ams_oof["predictions"][:, lead])
                + stack_weight * stack_oof_patterns[:, lead]
            )
            oof_ec = np.asarray(ec[oof_indices, lead][:, mask], dtype=np.float64)
            if external_observations is None:
                oof_target = np.asarray(
                    obs_all[oof_indices, lead][:, mask], dtype=np.float64
                )
            else:
                observation_indices = [
                    external_date_to_index[date] for date in oof_dates
                ]
                oof_target = external_observations[observation_indices][:, mask]
            mean_coefficient, spatial_coefficient = fit_oof_amplitude_calibration(
                oof_patterns, oof_ec, oof_target, area_weights
            )
            restored = restore_calibrated_patterns(
                blended_patterns,
                ec_vectors,
                area_weights,
                mean_coefficient,
                spatial_coefficient,
            )
            calibration_by_lead[str(lead)] = {
                "mean_coefficient": mean_coefficient,
                "spatial_coefficient": spatial_coefficient,
            }
        else:
            ec_means = ec_vectors.mean(axis=1, keepdims=True)
            ec_scales = ec_vectors.std(axis=1, keepdims=True)
            restored = blended_patterns * np.maximum(ec_scales, 1e-8) + ec_means
        if fallback_seasons[lead]:
            fallback_rows = np.asarray(
                [
                    SEASON_BY_MONTH[int(str(date)[5:7])] in fallback_seasons[lead]
                    for date in stack_dates
                ]
            )
            restored[fallback_rows] = ec_vectors[fallback_rows]
        lead_fields = np.zeros((len(indices), *mask.shape), dtype=np.float32)
        lead_fields[:, mask] = restored.astype(np.float32)
        result[indices, lead] = lead_fields

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, result)
    metadata = {
        "ams_weights": ams_weights.tolist(),
        "seasonal_stack_weights": (1.0 - ams_weights).tolist(),
        "weights_selected_from_oof": select_weights_from_oof,
        "weight_diagnostics": weight_diagnostics,
        "seasonal_ec_safety": seasonal_ec_safety,
        "observation_transform": observation_transform,
        "fallback_seasons_by_lead": {
            str(lead): sorted(seasons)
            for lead, seasons in enumerate(fallback_seasons)
        },
        "safety_diagnostics": safety_diagnostics,
        "amplitude_calibrated_from_oof": calibrate_amplitude,
        "calibration_by_lead": calibration_by_lead,
    }
    output_file.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=paths.results_dir / "ams_multilead"
    )
    parser.add_argument(
        "--stack-file",
        type=Path,
        default=paths.results_dir
        / "seasonal_stacking"
        / "seasonal_stacking_test_patterns.npz",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=experiment_path(
            "fixed_ensemble", "multi_lead_predict_results_ensemble.npy"
        ),
    )
    parser.add_argument("--ams-weight", type=float, default=0.5)
    parser.add_argument(
        "--ams-oof-file",
        type=Path,
        default=paths.results_dir / "ams_multilead" / "ams_oof_patterns.npz",
    )
    parser.add_argument(
        "--calibrate-amplitude",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fit map-mean and spatial amplitude coefficients from historical OOF predictions",
    )
    parser.add_argument(
        "--observation-file",
        type=Path,
        default=OBSERVATION_FILE,
        help="Dated corrected observation NPZ used for mask, OOF selection, and calibration",
    )
    parser.add_argument(
        "--select-weights-from-oof",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Select a separate AMS/stacking blend weight for each lead from historical OOF months",
    )
    parser.add_argument(
        "--seasonal-ec-safety",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fall back to ECMWF when no rolling OOF fold supports a lead-season blend",
    )
    parser.add_argument(
        "--observation-transform",
        choices=("none", "signed_log1p"),
        default="none",
        help="Apply the same fixed anomaly transform used by trainer and stacking",
    )
    args = parser.parse_args()
    output = build_ensemble(
        args.data_dir,
        args.stack_file,
        args.output_file,
        args.ams_weight,
        args.ams_oof_file,
        args.calibrate_amplitude,
        args.observation_file,
        args.select_weights_from_oof,
        args.seasonal_ec_safety,
        args.observation_transform,
    )
    print(f"Saved fixed ensemble to {output}")


if __name__ == "__main__":
    main()
