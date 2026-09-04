"""Nested rolling validation for season-dependent convex forecast stacking."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from project_paths import experiment_path
from analyze_multimodel_raw import (
    EXCLUDED,
    OBS_PATH,
    START,
    TEST_END,
    aligned_dates,
    anomalies_for_dates,
    build_model_fields,
    standardize,
)


NUM_TEST = 21
SEASON_BY_MONTH = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}
PENALTIES = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05)


def simplex_weights(step=0.05):
    units = int(round(1.0 / step))
    rows = []
    for ec in range(units + 1):
        for ncep in range(units - ec + 1):
            for recent in range(units - ec - ncep + 1):
                annual = units - ec - ncep - recent
                rows.append((ec, ncep, recent, annual))
    return np.asarray(rows, dtype=np.float32) / units


WEIGHTS = simplex_weights()


def transform_fractional_anomaly(values, transform):
    if transform == "none":
        return values
    if transform == "signed_log1p":
        return np.sign(values) * np.log1p(np.abs(values))
    raise ValueError(f"Unknown observation transform: {transform}")


def load_observations(path: Path, transform="none"):
    """Load either the legacy aligned array or a dated corrected product."""
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            observations = np.asarray(data["anomaly_fraction"], dtype=np.float32)
            observation_dates = pd.to_datetime(data["dates"].astype(str))
            mask = np.asarray(data["valid_mask"], dtype=bool)
    else:
        observations = np.asarray(np.load(path), dtype=np.float32)
        observation_dates = [
            date
            for date in pd.date_range(START, TEST_END, freq="MS")
            if date not in EXCLUDED
        ]
        mask = np.isfinite(observations[0])

    if observations.ndim != 3 or observations.shape[1:] != mask.shape:
        raise ValueError(
            f"Observation shape {observations.shape} is incompatible with mask {mask.shape}"
        )
    if len(observation_dates) != len(observations):
        raise ValueError("Observation dates and fields have different lengths")
    if not np.all(np.isfinite(observations[:, mask])):
        raise ValueError("Observation product contains missing values inside valid_mask")

    date_to_idx = {
        pd.Timestamp(date): index for index, date in enumerate(observation_dates)
    }
    observations = transform_fractional_anomaly(observations, transform)
    return np.nan_to_num(observations, nan=0.0), date_to_idx, mask


def transform_field_dict(fields, transform):
    return {
        date: transform_fractional_anomaly(field, transform)
        for date, field in fields.items()
    }


def vector(field, mask):
    return standardize(np.nan_to_num(field, nan=0.0), mask)[mask]


def build_arrays(dates, lead, ec_fields, ncep_fields, observations, date_to_idx, mask):
    ec = np.stack([vector(ec_fields[date], mask) for date in dates])
    ncep = np.stack([vector(ncep_fields[date], mask) for date in dates])
    recent = []
    annual = []
    target = []
    for date in dates:
        recent_fields = [
            vector(
                observations[
                    date_to_idx[date - pd.DateOffset(months=lead + offset)]
                ],
                mask,
            )
            for offset in (1, 2, 3)
        ]
        recent.append(np.mean(recent_fields, axis=0))
        annual.append(
            vector(observations[date_to_idx[date - pd.DateOffset(months=12)]], mask)
        )
        target.append(vector(observations[date_to_idx[date]], mask))
    sources = np.stack([ec, ncep, np.stack(recent), np.stack(annual)], axis=1)
    return sources.astype(np.float32), np.stack(target).astype(np.float32)


def score_weights(sources, target, weights):
    # Sufficient statistics avoid materializing every candidate forecast map.
    sources = sources - sources.mean(axis=2, keepdims=True)
    target = target - target.mean(axis=1, keepdims=True)
    source_target = np.einsum("nkp,np->nk", sources, target, optimize=True)
    source_gram = np.einsum("nkp,nlp->nkl", sources, sources, optimize=True)
    target_norm2 = np.sum(target**2, axis=1)
    numerator = source_target @ weights.T
    prediction_norm2 = np.einsum(
        "mk,nkl,ml->nm", weights, source_gram, weights, optimize=True
    )
    denominator = np.sqrt(prediction_norm2 * target_norm2[:, None])
    correlations = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    return correlations.mean(axis=0), correlations


def select_weights(dates, sources, target, penalty):
    selected = {}
    for season in ("DJF", "MAM", "JJA", "SON"):
        indices = np.asarray(
            [index for index, date in enumerate(dates) if SEASON_BY_MONTH[date.month] == season]
        )
        mean_acc, _ = score_weights(sources[indices], target[indices], WEIGHTS)
        shrinkage = np.sum((WEIGHTS - np.asarray([1, 0, 0, 0])) ** 2, axis=1)
        objective = mean_acc - penalty * shrinkage
        selected[season] = WEIGHTS[int(np.argmax(objective))]
    return selected


def evaluate_selected(dates, sources, target, selected):
    scores = []
    for index, date in enumerate(dates):
        weight = selected[SEASON_BY_MONTH[date.month]]
        prediction = np.sum(sources[index] * weight[:, None], axis=0, keepdims=True)
        _, correlations = score_weights(
            prediction[:, None, :], target[index:index + 1], np.ones((1, 1))
        )
        scores.append(float(correlations[0, 0]))
    return np.asarray(scores)


def evaluate_lead(
    lead,
    ec_raw,
    ncep_raw,
    observations,
    date_to_idx,
    mask,
    observation_transform="none",
):
    dates = aligned_dates(ec_raw, lead)
    train_dates = dates[:-NUM_TEST]
    test_dates = dates[-NUM_TEST:]
    cv_scores = {penalty: [] for penalty in PENALTIES}
    cv_ec_scores = []
    fold_cache = []
    splitter = TimeSeriesSplit(n_splits=5, test_size=NUM_TEST)

    for train_idx, val_idx in splitter.split(train_dates):
        fold_train = [train_dates[index] for index in train_idx]
        fold_val = [train_dates[index] for index in val_idx]
        all_dates = fold_train + fold_val
        ec_fields = transform_field_dict(
            anomalies_for_dates(ec_raw, lead, fold_train, all_dates),
            observation_transform,
        )
        ncep_fields = transform_field_dict(
            anomalies_for_dates(ncep_raw, lead, fold_train, all_dates),
            observation_transform,
        )
        sources, target = build_arrays(
            all_dates,
            lead,
            ec_fields,
            ncep_fields,
            observations,
            date_to_idx,
            mask,
        )
        split = len(fold_train)
        ec_scores = evaluate_selected(
            fold_val,
            sources[split:],
            target[split:],
            {season: np.asarray([1, 0, 0, 0]) for season in ("DJF", "MAM", "JJA", "SON")},
        )
        cv_ec_scores.append(float(ec_scores.mean()))
        for penalty in PENALTIES:
            selected = select_weights(
                fold_train, sources[:split], target[:split], penalty
            )
            scores = evaluate_selected(
                fold_val, sources[split:], target[split:], selected
            )
            cv_scores[penalty].append(float(scores.mean()))
        fold_cache.append((fold_train, fold_val, sources, target, split))

    fold_weights = np.arange(1.0, 6.0) ** 2
    fold_weights /= fold_weights.sum()
    weighted_scores = {
        penalty: float(np.average(values, weights=fold_weights))
        for penalty, values in cv_scores.items()
    }
    best_penalty = max(weighted_scores, key=weighted_scores.get)
    oof_dates = []
    oof_predictions = []
    for fold_train, fold_val, fold_sources, fold_target, fold_split in fold_cache:
        fold_selected = select_weights(
            fold_train,
            fold_sources[:fold_split],
            fold_target[:fold_split],
            best_penalty,
        )
        oof_dates.extend(fold_val)
        oof_predictions.extend(
            np.sum(
                fold_sources[fold_split + index]
                * fold_selected[SEASON_BY_MONTH[date.month]][:, None],
                axis=0,
            )
            for index, date in enumerate(fold_val)
        )

    all_dates = train_dates + test_dates
    ec_fields = transform_field_dict(
        anomalies_for_dates(ec_raw, lead, train_dates, all_dates),
        observation_transform,
    )
    ncep_fields = transform_field_dict(
        anomalies_for_dates(ncep_raw, lead, train_dates, all_dates),
        observation_transform,
    )
    sources, target = build_arrays(
        all_dates,
        lead,
        ec_fields,
        ncep_fields,
        observations,
        date_to_idx,
        mask,
    )
    split = len(train_dates)
    selected = select_weights(
        train_dates, sources[:split], target[:split], best_penalty
    )
    test_scores = evaluate_selected(
        test_dates, sources[split:], target[split:], selected
    )
    ec_selected = {
        season: np.asarray([1, 0, 0, 0]) for season in ("DJF", "MAM", "JJA", "SON")
    }
    ec_test_scores = evaluate_selected(
        test_dates, sources[split:], target[split:], ec_selected
    )
    return {
        "lead": lead,
        "penalty": best_penalty,
        "cv_acc": weighted_scores[best_penalty],
        "ec_cv_acc": float(np.average(cv_ec_scores, weights=fold_weights)),
        "fold_gains": np.asarray(cv_scores[best_penalty]) - np.asarray(cv_ec_scores),
        "test_acc": float(test_scores.mean()),
        "ec_test_acc": float(ec_test_scores.mean()),
        "weights": selected,
        "test_dates": test_dates,
        "test_prediction": np.stack(
            [
                np.sum(
                    sources[split + index]
                    * selected[SEASON_BY_MONTH[date.month]][:, None],
                    axis=0,
                )
                for index, date in enumerate(test_dates)
            ]
        ),
        "oof_dates": oof_dates,
        "oof_prediction": np.stack(oof_predictions),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-file", type=Path, default=OBS_PATH)
    parser.add_argument(
        "--observation-transform",
        choices=("none", "signed_log1p"),
        default="none",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=experiment_path(
            "seasonal_stacking", "seasonal_stacking_test_patterns.npz"
        ),
    )
    args = parser.parse_args()

    observations, date_to_idx, mask = load_observations(
        args.observation_file, args.observation_transform
    )
    _, _, _, ec_raw = build_model_fields("ECMWF")
    _, _, _, ncep_raw = build_model_fields("NCEP")

    print("lead,penalty,cv_acc,ec_cv_acc,fold_gains,test_acc,ec_test_acc,season_weights")
    results = []
    for lead in range(6):
        result = evaluate_lead(
            lead,
            ec_raw,
            ncep_raw,
            observations,
            date_to_idx,
            mask,
            args.observation_transform,
        )
        results.append(result)
        gains = "/".join(f"{value:+.3f}" for value in result["fold_gains"])
        weights = ";".join(
            f'{season}:{"/".join(f"{value:.2f}" for value in weight)}'
            for season, weight in result["weights"].items()
        )
        print(
            f'{result["lead"]},{result["penalty"]:.3f},{result["cv_acc"]:.6f},'
            f'{result["ec_cv_acc"]:.6f},{gains},{result["test_acc"]:.6f},'
            f'{result["ec_test_acc"]:.6f},{weights}'
        )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_file,
        dates=np.asarray(
            [date.strftime("%Y-%m-%d") for date in results[0]["test_dates"]]
        ),
        predictions=np.stack([r["test_prediction"] for r in results], axis=1),
        oof_dates=np.stack(
            [
                np.asarray([date.strftime("%Y-%m-%d") for date in r["oof_dates"]])
                for r in results
            ],
            axis=1,
        ),
        oof_predictions=np.stack([r["oof_prediction"] for r in results], axis=1),
        valid_mask=mask,
        source_names=np.asarray(["ECMWF", "NCEP", "recent_obs", "annual_obs"]),
    )
    print(
        f'macro_test_acc={np.mean([r["test_acc"] for r in results]):.6f},'
        f'macro_ec_acc={np.mean([r["ec_test_acc"] for r in results]):.6f}'
    )
    print(f"saved={args.output_file}")


if __name__ == "__main__":
    main()
