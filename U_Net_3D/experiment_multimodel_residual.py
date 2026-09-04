"""Leak-free EOF/Ridge residual correction using ECMWF, NCEP, and observations.

The model predicts only the spatial-pattern error of ECMWF.  Forecast
climatologies, EOFs, and Ridge models are refitted inside every expanding
validation window.  The final 2023-01..2024-09 period is never used for model
or hyperparameter selection.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from analyze_multimodel_raw import (
    EXCLUDED,
    OBS_PATH,
    START,
    TEST_END,
    acc,
    aligned_dates,
    anomalies_for_dates,
    build_model_fields,
    standardize,
)
from experiment_seasonal_stacking import load_observations


N_COMPONENTS = (20, 40, 80)
ALPHAS = (10.0, 100.0, 1000.0, 10000.0)
CORRECTION_WEIGHTS = tuple(np.linspace(0.0, 1.0, 11))
NUM_TEST = 21


def pattern_vector(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return standardize(np.nan_to_num(field, nan=0.0), mask)[mask]


def target_matrix(dates, observations, date_to_idx, mask):
    return np.stack(
        [pattern_vector(observations[date_to_idx[date]], mask) for date in dates]
    ).astype(np.float32)


def source_matrix(dates, lead, ec_fields, ncep_fields, observations, date_to_idx, mask):
    rows = []
    calendar = []
    for target in dates:
        lag_dates = (
            target - pd.DateOffset(months=lead + 1),
            target - pd.DateOffset(months=lead + 2),
            target - pd.DateOffset(months=lead + 3),
            target - pd.DateOffset(months=12),
        )
        source_fields = [ec_fields[target], ncep_fields[target]] + [
            observations[date_to_idx[date]] for date in lag_dates
        ]
        rows.append(np.concatenate([pattern_vector(field, mask) for field in source_fields]))
        angle = 2.0 * np.pi * target.month / 12.0
        calendar.append((np.sin(angle), np.cos(angle)))
    return np.stack(rows).astype(np.float32), np.asarray(calendar, dtype=np.float32)


def row_acc(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_centered = prediction - prediction.mean(axis=1, keepdims=True)
    target_centered = target - target.mean(axis=1, keepdims=True)
    numerator = np.sum(pred_centered * target_centered, axis=1)
    denominator = np.sqrt(
        np.sum(pred_centered**2, axis=1) * np.sum(target_centered**2, axis=1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )


def fit_eof(x_train, x_eval, calendar_train, calendar_eval):
    max_components = min(max(N_COMPONENTS), len(x_train) - 1, x_train.shape[1])
    pca = PCA(
        n_components=max_components,
        svd_solver="randomized",
        whiten=True,
        random_state=42,
    )
    train_scores = pca.fit_transform(x_train)
    eval_scores = pca.transform(x_eval)
    return (
        np.concatenate([train_scores, calendar_train], axis=1),
        np.concatenate([eval_scores, calendar_eval], axis=1),
        max_components,
    )


def evaluate_lead(lead, ec_raw, ncep_raw, observations, date_to_idx, mask):
    dates = aligned_dates(ec_raw, lead)
    train_dates = dates[:-NUM_TEST]
    test_dates = dates[-NUM_TEST:]
    splitter = TimeSeriesSplit(n_splits=5, test_size=NUM_TEST)
    scores = defaultdict(list)

    for train_idx, val_idx in splitter.split(train_dates):
        fold_train = [train_dates[index] for index in train_idx]
        fold_val = [train_dates[index] for index in val_idx]
        all_fold_dates = fold_train + fold_val
        ec_fields = anomalies_for_dates(ec_raw, lead, fold_train, all_fold_dates)
        ncep_fields = anomalies_for_dates(ncep_raw, lead, fold_train, all_fold_dates)
        if not all(date in ncep_fields for date in all_fold_dates):
            raise ValueError(f"NCEP is incomplete in Lead-{lead} validation data")

        x_maps, calendar = source_matrix(
            all_fold_dates,
            lead,
            ec_fields,
            ncep_fields,
            observations,
            date_to_idx,
            mask,
        )
        split = len(fold_train)
        x_train, x_val, max_components = fit_eof(
            x_maps[:split], x_maps[split:], calendar[:split], calendar[split:]
        )
        y_train = target_matrix(fold_train, observations, date_to_idx, mask)
        y_val = target_matrix(fold_val, observations, date_to_idx, mask)
        ec_train = np.stack([pattern_vector(ec_fields[d], mask) for d in fold_train])
        ec_val = np.stack([pattern_vector(ec_fields[d], mask) for d in fold_val])
        residual_train = y_train - ec_train

        for n_components in N_COMPONENTS:
            if n_components > max_components:
                continue
            # Keep the two calendar columns after the selected EOF scores.
            selected_train = np.concatenate(
                [x_train[:, :n_components], x_train[:, -2:]], axis=1
            )
            selected_val = np.concatenate(
                [x_val[:, :n_components], x_val[:, -2:]], axis=1
            )
            for alpha in ALPHAS:
                model = Ridge(alpha=alpha)
                model.fit(selected_train, residual_train)
                residual_val = model.predict(selected_val)
                for weight in CORRECTION_WEIGHTS:
                    prediction = ec_val + weight * residual_val
                    scores[(n_components, alpha, float(weight))].append(
                        float(row_acc(prediction, y_val).mean())
                    )

    fold_weights = np.arange(1.0, 6.0) ** 2
    fold_weights /= fold_weights.sum()
    weighted_scores = {
        spec: float(np.average(fold_scores, weights=fold_weights))
        for spec, fold_scores in scores.items()
    }
    best_spec = max(weighted_scores, key=weighted_scores.get)

    all_dates = train_dates + test_dates
    ec_fields = anomalies_for_dates(ec_raw, lead, train_dates, all_dates)
    ncep_fields = anomalies_for_dates(ncep_raw, lead, train_dates, all_dates)
    x_maps, calendar = source_matrix(
        all_dates,
        lead,
        ec_fields,
        ncep_fields,
        observations,
        date_to_idx,
        mask,
    )
    split = len(train_dates)
    x_train, x_test, _ = fit_eof(
        x_maps[:split], x_maps[split:], calendar[:split], calendar[split:]
    )
    n_components, alpha, correction_weight = best_spec
    selected_train = np.concatenate(
        [x_train[:, :n_components], x_train[:, -2:]], axis=1
    )
    selected_test = np.concatenate(
        [x_test[:, :n_components], x_test[:, -2:]], axis=1
    )
    y_train = target_matrix(train_dates, observations, date_to_idx, mask)
    y_test = target_matrix(test_dates, observations, date_to_idx, mask)
    ec_train = np.stack([pattern_vector(ec_fields[d], mask) for d in train_dates])
    ec_test = np.stack([pattern_vector(ec_fields[d], mask) for d in test_dates])
    model = Ridge(alpha=alpha)
    model.fit(selected_train, y_train - ec_train)
    prediction = ec_test + correction_weight * model.predict(selected_test)

    fold_scores = np.asarray(scores[best_spec])
    baseline_folds = np.asarray(scores[(N_COMPONENTS[0], ALPHAS[0], 0.0)])
    return {
        "lead": lead,
        "n_components": n_components,
        "alpha": alpha,
        "weight": correction_weight,
        "cv_acc": weighted_scores[best_spec],
        "ec_cv_acc": float(np.average(baseline_folds, weights=fold_weights)),
        "fold_gains": fold_scores - baseline_folds,
        "test_acc": float(row_acc(prediction, y_test).mean()),
        "ec_test_acc": float(row_acc(ec_test, y_test).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-file", type=Path, default=OBS_PATH)
    args = parser.parse_args()
    observations, date_to_idx, mask = load_observations(args.observation_file)

    _, _, ec_failures, ec_raw = build_model_fields("ECMWF")
    _, _, ncep_failures, ncep_raw = build_model_fields("NCEP")
    if ec_failures or ncep_failures:
        raise RuntimeError(
            f"Forecast read failures: ECMWF={ec_failures[:2]}, NCEP={ncep_failures[:2]}"
        )

    print("lead,n_components,alpha,weight,cv_acc,ec_cv_acc,fold_gains,test_acc,ec_test_acc")
    results = []
    for lead in range(6):
        result = evaluate_lead(
            lead, ec_raw, ncep_raw, observations, date_to_idx, mask
        )
        results.append(result)
        gains = "/".join(f"{gain:+.3f}" for gain in result["fold_gains"])
        print(
            f'{lead},{result["n_components"]},{result["alpha"]:.0f},'
            f'{result["weight"]:.1f},{result["cv_acc"]:.6f},'
            f'{result["ec_cv_acc"]:.6f},{gains},{result["test_acc"]:.6f},'
            f'{result["ec_test_acc"]:.6f}'
        )
    print(
        "macro_test_acc="
        f'{np.mean([result["test_acc"] for result in results]):.6f},'
        "macro_ec_acc="
        f'{np.mean([result["ec_test_acc"] for result in results]):.6f}'
    )


if __name__ == "__main__":
    main()
