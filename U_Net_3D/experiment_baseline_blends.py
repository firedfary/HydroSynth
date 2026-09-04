"""Select leak-free convex blends of ECMWF, NCEP, and persistence baselines."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

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
WEIGHTS = tuple(np.linspace(0.0, 1.0, 21))


def vector(field, mask):
    return standardize(np.nan_to_num(field, nan=0.0), mask)[mask]


def row_acc(prediction, target):
    prediction = prediction - prediction.mean(axis=1, keepdims=True)
    target = target - target.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(prediction**2, axis=1) * np.sum(target**2, axis=1)
    )
    return np.divide(
        np.sum(prediction * target, axis=1),
        denominator,
        out=np.zeros(len(prediction), dtype=np.float32),
        where=denominator > 0,
    )


def build_sources(dates, lead, ec_fields, ncep_fields, observations, date_to_idx, mask):
    sources = {
        "ECMWF": np.stack([vector(ec_fields[date], mask) for date in dates]),
        "NCEP": np.stack([vector(ncep_fields[date], mask) for date in dates]),
    }
    lag_vectors = []
    for name, offset in (
        ("Lag1", lead + 1),
        ("Lag2", lead + 2),
        ("Lag3", lead + 3),
        ("Lag12", 12),
    ):
        values = np.stack(
            [
                vector(observations[date_to_idx[date - pd.DateOffset(months=offset)]], mask)
                for date in dates
            ]
        )
        sources[name] = values
        lag_vectors.append(values)
    sources["LagMean"] = np.mean(lag_vectors[:3], axis=0)
    return sources


def evaluate_lead(lead, ec_raw, ncep_raw, observations, date_to_idx, mask):
    dates = aligned_dates(ec_raw, lead)
    train_dates = dates[:-NUM_TEST]
    test_dates = dates[-NUM_TEST:]
    scores = defaultdict(list)
    splitter = TimeSeriesSplit(n_splits=5, test_size=NUM_TEST)

    for train_idx, val_idx in splitter.split(train_dates):
        fold_train = [train_dates[index] for index in train_idx]
        fold_val = [train_dates[index] for index in val_idx]
        ec_fields = anomalies_for_dates(ec_raw, lead, fold_train, fold_val)
        ncep_fields = anomalies_for_dates(ncep_raw, lead, fold_train, fold_val)
        sources = build_sources(
            fold_val,
            lead,
            ec_fields,
            ncep_fields,
            observations,
            date_to_idx,
            mask,
        )
        target = np.stack([vector(observations[date_to_idx[date]], mask) for date in fold_val])
        ec = sources["ECMWF"]
        for source_name, source in sources.items():
            for weight in WEIGHTS:
                prediction = (1.0 - weight) * ec + weight * source
                scores[(source_name, float(weight))].append(
                    float(row_acc(prediction, target).mean())
                )

    fold_weights = np.arange(1.0, 6.0) ** 2
    fold_weights /= fold_weights.sum()
    weighted_scores = {
        spec: float(np.average(values, weights=fold_weights))
        for spec, values in scores.items()
    }
    best_spec = max(weighted_scores, key=weighted_scores.get)

    ec_fields = anomalies_for_dates(ec_raw, lead, train_dates, test_dates)
    ncep_fields = anomalies_for_dates(ncep_raw, lead, train_dates, test_dates)
    sources = build_sources(
        test_dates,
        lead,
        ec_fields,
        ncep_fields,
        observations,
        date_to_idx,
        mask,
    )
    target = np.stack([vector(observations[date_to_idx[date]], mask) for date in test_dates])
    source_name, weight = best_spec
    prediction = (1.0 - weight) * sources["ECMWF"] + weight * sources[source_name]
    baseline_spec = ("ECMWF", 0.0)
    return {
        "lead": lead,
        "source": source_name,
        "weight": weight,
        "cv_acc": weighted_scores[best_spec],
        "ec_cv_acc": weighted_scores[baseline_spec],
        "fold_gains": np.asarray(scores[best_spec]) - np.asarray(scores[baseline_spec]),
        "test_acc": float(row_acc(prediction, target).mean()),
        "ec_test_acc": float(row_acc(sources["ECMWF"], target).mean()),
    }


def main():
    valid_dates = [
        date
        for date in pd.date_range(START, TEST_END, freq="MS")
        if date not in EXCLUDED
    ]
    date_to_idx = {date: index for index, date in enumerate(valid_dates)}
    raw_observations = np.load(OBS_PATH)
    mask = ~np.isnan(raw_observations[0])
    observations = np.nan_to_num(raw_observations, nan=0.0).astype(np.float32)
    _, _, _, ec_raw = build_model_fields("ECMWF")
    _, _, _, ncep_raw = build_model_fields("NCEP")

    print("lead,source,weight,cv_acc,ec_cv_acc,fold_gains,test_acc,ec_test_acc")
    results = []
    for lead in range(6):
        result = evaluate_lead(
            lead, ec_raw, ncep_raw, observations, date_to_idx, mask
        )
        results.append(result)
        gains = "/".join(f"{value:+.3f}" for value in result["fold_gains"])
        print(
            f'{lead},{result["source"]},{result["weight"]:.2f},'
            f'{result["cv_acc"]:.6f},{result["ec_cv_acc"]:.6f},{gains},'
            f'{result["test_acc"]:.6f},{result["ec_test_acc"]:.6f}'
        )
    print(
        f'macro_test_acc={np.mean([r["test_acc"] for r in results]):.6f},'
        f'macro_ec_acc={np.mean([r["ec_test_acc"] for r in results]):.6f}'
    )


if __name__ == "__main__":
    main()
