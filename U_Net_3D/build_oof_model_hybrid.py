"""Build a lead-season hybrid from two AMS forecast products using OOF skill."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from project_paths import OBSERVATION_FILE, paths, experiment_dir

from build_fixed_ensemble import (
    SEASON_BY_MONTH,
    spatial_acc_vectors,
    standardize_vectors,
    transform_fractional_anomaly,
)


def select_candidate_seasons(
    base_oof: np.ndarray,
    candidate_oof: np.ndarray,
    targets: np.ndarray,
    dates: np.ndarray,
    area_weights: np.ndarray,
    minimum_positive_folds: int = 4,
    minimum_mean_gain: float = 0.005,
    fold_size: int = 21,
) -> tuple[list[set[str]], list[dict]]:
    if base_oof.shape != candidate_oof.shape or targets.shape != base_oof.shape:
        raise ValueError("Base, candidate, and target OOF shapes must match")
    if dates.shape != base_oof.shape[:2]:
        raise ValueError("OOF dates shape must match sample and lead dimensions")

    sample_count, lead_count, _ = base_oof.shape
    folds = np.arange(sample_count) // fold_size
    selected = [set() for _ in range(lead_count)]
    diagnostics = []
    for lead in range(lead_count):
        base_acc = spatial_acc_vectors(
            standardize_vectors(base_oof[:, lead]), targets[:, lead], area_weights
        )
        candidate_acc = spatial_acc_vectors(
            standardize_vectors(candidate_oof[:, lead]), targets[:, lead], area_weights
        )
        sample_seasons = np.asarray(
            [SEASON_BY_MONTH[int(str(date)[5:7])] for date in dates[:, lead]]
        )
        for season in ("DJF", "MAM", "JJA", "SON"):
            season_mask = sample_seasons == season
            fold_gains = []
            for fold in np.unique(folds):
                fold_mask = season_mask & (folds == fold)
                if np.any(fold_mask):
                    fold_gains.append(
                        float(np.mean(candidate_acc[fold_mask] - base_acc[fold_mask]))
                    )
            mean_gain = float(
                np.mean(candidate_acc[season_mask] - base_acc[season_mask])
            )
            positive_folds = int(np.sum(np.asarray(fold_gains) > 0.0))
            use_candidate = bool(
                mean_gain >= minimum_mean_gain
                and positive_folds >= minimum_positive_folds
            )
            if use_candidate:
                selected[lead].add(season)
            diagnostics.append(
                {
                    "lead": lead,
                    "season": season,
                    "mean_oof_acc_gain": mean_gain,
                    "positive_folds": positive_folds,
                    "fold_count": len(fold_gains),
                    "use_candidate": use_candidate,
                }
            )
    return selected, diagnostics


def build_hybrid(
    base_dir: Path,
    candidate_dir: Path,
    base_oof_file: Path,
    candidate_oof_file: Path,
    observation_file: Path,
    output_dir: Path,
    output_oof_file: Path,
    minimum_positive_folds: int,
    minimum_mean_gain: float,
    observation_transform: str = "none",
) -> None:
    names = (
        "multi_lead_dates.npy",
        "multi_lead_obs_results.npy",
        "multi_lead_ec_precip_anom_results.npy",
        "multi_lead_predict_results.npy",
    )
    base_arrays = {name: np.load(base_dir / name, mmap_mode="r") for name in names}
    candidate_arrays = {
        name: np.load(candidate_dir / name, mmap_mode="r") for name in names
    }
    for name in names:
        if base_arrays[name].shape != candidate_arrays[name].shape:
            raise ValueError(f"Shape mismatch for {name}")
    if not np.array_equal(
        base_arrays["multi_lead_dates.npy"], candidate_arrays["multi_lead_dates.npy"]
    ):
        raise ValueError("Base and candidate production dates differ")

    with np.load(base_oof_file) as data:
        base_dates = data["dates"].astype(str)
        base_oof = np.asarray(data["predictions"], dtype=np.float64)
    with np.load(candidate_oof_file) as data:
        candidate_dates = data["dates"].astype(str)
        candidate_oof = np.asarray(data["predictions"], dtype=np.float64)
    if not np.array_equal(base_dates, candidate_dates):
        raise ValueError("Base and candidate OOF dates differ")

    with np.load(observation_file) as observations:
        obs = transform_fractional_anomaly(
            observations["anomaly_fraction"], observation_transform
        )
        obs_dates = observations["dates"].astype(str)
        mask = observations["valid_mask"].astype(bool)
    obs_index = {date: index for index, date in enumerate(obs_dates)}
    targets = np.empty_like(base_oof)
    for lead in range(base_oof.shape[1]):
        targets[:, lead] = obs[[obs_index[date] for date in base_dates[:, lead]]][:, mask]

    latitudes = np.arange(60.0, 0.0, -0.5)
    area_weights = (
        np.cos(np.deg2rad(latitudes))[:, None] * np.ones((1, mask.shape[1]))
    )[mask]
    area_weights /= area_weights.sum()
    selected, diagnostics = select_candidate_seasons(
        base_oof,
        candidate_oof,
        targets,
        base_dates,
        area_weights,
        minimum_positive_folds,
        minimum_mean_gain,
    )

    hybrid_oof = base_oof.copy()
    for lead, seasons in enumerate(selected):
        rows = np.asarray(
            [SEASON_BY_MONTH[int(str(date)[5:7])] in seasons for date in base_dates[:, lead]]
        )
        hybrid_oof[rows, lead] = candidate_oof[rows, lead]

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in names[:-1]:
        np.save(output_dir / name, np.asarray(base_arrays[name]))
    production_dates = base_arrays["multi_lead_dates.npy"].astype(str)
    hybrid_forecast = np.asarray(base_arrays["multi_lead_predict_results.npy"]).copy()
    candidate_forecast = candidate_arrays["multi_lead_predict_results.npy"]
    for lead, seasons in enumerate(selected):
        rows = np.asarray(
            [SEASON_BY_MONTH[int(str(date)[5:7])] in seasons for date in production_dates]
        )
        hybrid_forecast[rows, lead] = candidate_forecast[rows, lead]
    np.save(output_dir / "multi_lead_predict_results.npy", hybrid_forecast)
    np.savez_compressed(
        output_oof_file,
        dates=base_dates,
        predictions=hybrid_oof.astype(np.float32),
    )
    metadata = {
        "base_dir": str(base_dir),
        "candidate_dir": str(candidate_dir),
        "minimum_positive_folds": minimum_positive_folds,
        "minimum_mean_gain": minimum_mean_gain,
        "observation_transform": observation_transform,
        "candidate_seasons_by_lead": {
            str(lead): sorted(seasons) for lead, seasons in enumerate(selected)
        },
        "diagnostics": diagnostics,
    }
    (output_dir / "oof_model_hybrid.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata["candidate_seasons_by_lead"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir", type=Path, default=paths.results_dir / "signed_log_run"
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=paths.results_dir / "model_as_sample_transfer_run",
    )
    parser.add_argument(
        "--base-oof-file",
        type=Path,
        default=paths.results_dir / "signed_log_run" / "ams_oof_patterns.npz",
    )
    parser.add_argument(
        "--candidate-oof-file",
        type=Path,
        default=paths.results_dir
        / "model_as_sample_transfer_run"
        / "model_as_sample_transfer_oof.npz",
    )
    parser.add_argument("--observation-file", type=Path, default=OBSERVATION_FILE)
    parser.add_argument(
        "--output-dir", type=Path, default=experiment_dir("oof_model_hybrid")
    )
    parser.add_argument(
        "--output-oof-file",
        type=Path,
        default=paths.results_dir / "oof_model_hybrid" / "oof_patterns.npz",
    )
    parser.add_argument("--minimum-positive-folds", type=int, default=4)
    parser.add_argument("--minimum-mean-gain", type=float, default=0.005)
    parser.add_argument(
        "--observation-transform",
        choices=("none", "signed_log1p"),
        default="none",
    )
    args = parser.parse_args()
    build_hybrid(
        args.base_dir,
        args.candidate_dir,
        args.base_oof_file,
        args.candidate_oof_file,
        args.observation_file,
        args.output_dir,
        args.output_oof_file,
        args.minimum_positive_folds,
        args.minimum_mean_gain,
        args.observation_transform,
    )


if __name__ == "__main__":
    main()
