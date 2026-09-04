"""Use auxiliary seasonal models as training samples for ECMWF correction.

Each NCEP/JMA forecast is paired with the verifying observation as an
additional training example. Validation and final testing remain in the
ECMWF domain. All forecast climatologies and PCA maps are refitted within
each expanding time split.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from project_paths import OBSERVATION_FILE, paths, experiment_path

from analyze_multimodel_raw import (
    aligned_dates,
    anomalies_for_dates,
    build_model_fields,
)
from experiment_seasonal_stacking import load_observations, transform_fractional_anomaly


OBS_PATH = OBSERVATION_FILE
BASE_DATA_DIR = paths.results_dir / "signed_log_run"
OUTPUT_DIR = paths.results_dir / "model_as_sample_transfer_run"
NUM_TEST = 21
N_COMPONENTS = (10, 20, 40, 80)
ALPHAS = (1.0, 10.0, 100.0, 1000.0)
AUXILIARY_WEIGHTS = (0.25, 0.5, 1.0)
MAPPED_WEIGHTS = (0.25, 0.5, 0.75, 1.0)
RECENCY_HALFLIVES = (0, 60, 120)
SOURCE_NAMES = tuple(
    name.strip()
    for name in os.getenv("AMS_TRANSFER_SOURCES", "ECMWF,NCEP,JMA").split(",")
    if name.strip()
)
if not SOURCE_NAMES or SOURCE_NAMES[0] != "ECMWF":
    raise ValueError("AMS_TRANSFER_SOURCES must start with ECMWF")


def weighted_pattern_vector(
    field: np.ndarray, mask: np.ndarray, point_weights: np.ndarray
) -> np.ndarray:
    values = np.nan_to_num(field, nan=0.0)[mask].astype(np.float64)
    mean = np.sum(values * point_weights) / np.sum(point_weights)
    centered = values - mean
    variance = np.sum(point_weights * centered**2) / np.sum(point_weights)
    return (centered / np.sqrt(variance + 1e-12)).astype(np.float32)


def weighted_row_acc(
    prediction: np.ndarray, target: np.ndarray, point_weights: np.ndarray
) -> np.ndarray:
    pred_mean = np.sum(prediction * point_weights, axis=1, keepdims=True) / np.sum(
        point_weights
    )
    target_mean = np.sum(target * point_weights, axis=1, keepdims=True) / np.sum(
        point_weights
    )
    pred_centered = prediction - pred_mean
    target_centered = target - target_mean
    numerator = np.sum(point_weights * pred_centered * target_centered, axis=1)
    denominator = np.sqrt(
        np.sum(point_weights * pred_centered**2, axis=1)
        * np.sum(point_weights * target_centered**2, axis=1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )


def target_matrix(
    dates, observations, date_to_idx, mask, point_weights
) -> np.ndarray:
    return np.stack(
        [
            weighted_pattern_vector(
                observations[date_to_idx[date]], mask, point_weights
            )
            for date in dates
        ]
    )


def calendar_features(dates) -> np.ndarray:
    angles = np.asarray([2.0 * np.pi * date.month / 12.0 for date in dates])
    return np.column_stack([np.sin(angles), np.cos(angles)]).astype(np.float32)


def model_anomalies(raw_models, lead, climatology_dates, requested_dates):
    return {
        model: {
            date: transform_fractional_anomaly(field, "signed_log1p")
            for date, field in anomalies_for_dates(
                raw_models[model], lead, climatology_dates, requested_dates
            ).items()
        }
        for model in SOURCE_NAMES
    }


def build_training_samples(
    dates,
    fields,
    observations,
    date_to_idx,
    mask,
    point_weights,
):
    input_rows = []
    targets = []
    source_indices = []
    sample_dates = []
    for source_index, model in enumerate(SOURCE_NAMES):
        for date in dates:
            if date not in fields[model]:
                continue
            input_rows.append(
                weighted_pattern_vector(fields[model][date], mask, point_weights)
            )
            targets.append(
                weighted_pattern_vector(
                    observations[date_to_idx[date]], mask, point_weights
                )
            )
            source_indices.append(source_index)
            sample_dates.append(date)
    return (
        np.stack(input_rows),
        np.stack(targets),
        np.asarray(source_indices),
        sample_dates,
    )


def append_metadata(scores, dates, source_indices) -> np.ndarray:
    one_hot = np.eye(len(SOURCE_NAMES), dtype=np.float32)[source_indices]
    return np.concatenate([scores, calendar_features(dates), one_hot], axis=1)


def prepare_fold(
    lead,
    train_dates,
    eval_dates,
    raw_models,
    observations,
    date_to_idx,
    mask,
    point_weights,
):
    requested = train_dates + eval_dates
    fields = model_anomalies(raw_models, lead, train_dates, requested)
    train_input, train_target, train_sources, sample_dates = build_training_samples(
        train_dates,
        fields,
        observations,
        date_to_idx,
        mask,
        point_weights,
    )
    eval_input = np.stack(
        [
            weighted_pattern_vector(fields["ECMWF"][date], mask, point_weights)
            for date in eval_dates
        ]
    )
    eval_target = target_matrix(
        eval_dates, observations, date_to_idx, mask, point_weights
    )
    max_components = min(max(N_COMPONENTS), len(train_input) - 1, train_input.shape[1])
    input_pca = PCA(
        n_components=max_components,
        svd_solver="randomized",
        whiten=True,
        random_state=42,
    )
    train_scores = input_pca.fit_transform(train_input)
    eval_scores = input_pca.transform(eval_input)

    # Fit the target basis on unique dates so model availability does not
    # implicitly give some verifying months more weight than others.
    first_index_by_date = {}
    for index, date in enumerate(sample_dates):
        first_index_by_date.setdefault(date, index)
    unique_indices = np.asarray(list(first_index_by_date.values()))
    target_pca = PCA(n_components=0.40, svd_solver="full")
    target_pca.fit(train_target[unique_indices])
    return {
        "train_scores": train_scores,
        "target_scores": target_pca.transform(train_target),
        "train_sources": train_sources,
        "sample_dates": sample_dates,
        "target_pca": target_pca,
        "eval_scores": eval_scores,
        "eval_input": eval_input,
        "eval_target": eval_target,
        "eval_dates": eval_dates,
    }


def mapped_prediction(fold, spec, point_weights):
    n_components, alpha, auxiliary_weight, mapped_weight, recency_halflife = spec
    train_features = append_metadata(
        fold["train_scores"][:, :n_components],
        fold["sample_dates"],
        fold["train_sources"],
    )
    eval_sources = np.zeros(len(fold["eval_dates"]), dtype=np.int64)
    eval_features = append_metadata(
        fold["eval_scores"][:, :n_components], fold["eval_dates"], eval_sources
    )
    sample_weight = np.where(
        fold["train_sources"] == 0, 1.0, auxiliary_weight
    )
    if recency_halflife > 0:
        latest = max(fold["sample_dates"])
        ages = np.asarray(
            [
                (latest.year - date.year) * 12 + latest.month - date.month
                for date in fold["sample_dates"]
            ],
            dtype=np.float64,
        )
        sample_weight *= np.power(0.5, ages / recency_halflife)
        sample_weight *= len(sample_weight) / sample_weight.sum()
    model = Ridge(alpha=alpha)
    model.fit(train_features, fold["target_scores"], sample_weight=sample_weight)
    mapped = fold["target_pca"].inverse_transform(model.predict(eval_features))
    mapped = np.stack(
        [
            weighted_pattern_vector_row(row, point_weights)
            for row in mapped
        ]
    )
    return (1.0 - mapped_weight) * fold["eval_input"] + mapped_weight * mapped


def weighted_pattern_vector_row(values: np.ndarray, point_weights: np.ndarray) -> np.ndarray:
    mean = np.sum(values * point_weights) / np.sum(point_weights)
    centered = values - mean
    variance = np.sum(point_weights * centered**2) / np.sum(point_weights)
    return (centered / np.sqrt(variance + 1e-12)).astype(np.float32)


def evaluate_lead(
    lead,
    raw_models,
    observations,
    date_to_idx,
    mask,
    point_weights,
):
    selection_fold_count = int(os.getenv("AMS_SELECTION_FOLD_COUNT", "5"))
    if not 1 <= selection_fold_count <= 5:
        raise ValueError("AMS_SELECTION_FOLD_COUNT must be between 1 and 5")
    dates = aligned_dates(raw_models["ECMWF"], lead)
    train_dates = dates[:-NUM_TEST]
    test_dates = dates[-NUM_TEST:]
    splitter = TimeSeriesSplit(n_splits=5, test_size=NUM_TEST)
    folds = []
    for train_idx, val_idx in splitter.split(train_dates):
        fold_train = [train_dates[index] for index in train_idx]
        fold_val = [train_dates[index] for index in val_idx]
        folds.append(
            prepare_fold(
                lead,
                fold_train,
                fold_val,
                raw_models,
                observations,
                date_to_idx,
                mask,
                point_weights,
            )
        )

    specs = [
        (
            n_components,
            alpha,
            auxiliary_weight,
            mapped_weight,
            recency_halflife,
        )
        for n_components in N_COMPONENTS
        for alpha in ALPHAS
        for auxiliary_weight in AUXILIARY_WEIGHTS
        for mapped_weight in MAPPED_WEIGHTS
        for recency_halflife in RECENCY_HALFLIVES
    ]
    fold_scores = defaultdict(list)
    for fold in folds:
        # PCA is the expensive part. This straightforward implementation is
        # intentionally explicit for auditability; the dataset is small.
        for spec in specs:
            if spec[0] > fold["train_scores"].shape[1]:
                continue
            prediction = mapped_prediction(fold, spec, point_weights)
            fold_scores[spec].append(
                float(
                    weighted_row_acc(
                        prediction, fold["eval_target"], point_weights
                    ).mean()
                )
            )
    recency_weights = np.arange(1.0, selection_fold_count + 1.0) ** 2
    recency_weights /= recency_weights.sum()
    cv_scores = {
        spec: float(
            np.average(scores[:selection_fold_count], weights=recency_weights)
        )
        for spec, scores in fold_scores.items()
    }
    best_spec = max(cv_scores, key=cv_scores.get)

    oof_predictions = []
    oof_dates = []
    for fold in folds:
        fold_prediction = mapped_prediction(fold, best_spec, point_weights)
        oof_predictions.append(
            np.stack(
                [weighted_pattern_vector_row(row, point_weights) for row in fold_prediction]
            )
        )
        oof_dates.extend(fold["eval_dates"])

    test_fold = prepare_fold(
        lead,
        train_dates,
        test_dates,
        raw_models,
        observations,
        date_to_idx,
        mask,
        point_weights,
    )
    prediction = mapped_prediction(test_fold, best_spec, point_weights)
    prediction_patterns = np.stack(
        [weighted_pattern_vector_row(row, point_weights) for row in prediction]
    )
    test_scores = weighted_row_acc(
        prediction, test_fold["eval_target"], point_weights
    )
    ec_scores = weighted_row_acc(
        test_fold["eval_input"], test_fold["eval_target"], point_weights
    )
    baseline_folds = np.asarray([
        float(weighted_row_acc(fold["eval_input"], fold["eval_target"], point_weights).mean())
        for fold in folds
    ])
    selected_fold_scores = np.asarray(fold_scores[best_spec])
    holdout_slice = slice(selection_fold_count, None)
    holdout_acc = (
        float(selected_fold_scores[holdout_slice].mean())
        if selection_fold_count < len(folds)
        else None
    )
    holdout_ec_acc = (
        float(baseline_folds[holdout_slice].mean())
        if selection_fold_count < len(folds)
        else None
    )
    return {
        "lead": lead,
        "n_components": best_spec[0],
        "alpha": best_spec[1],
        "auxiliary_weight": best_spec[2],
        "mapped_weight": best_spec[3],
        "recency_halflife_months": best_spec[4],
        "cv_acc": cv_scores[best_spec],
        "ec_cv_acc": float(
            np.average(
                baseline_folds[:selection_fold_count], weights=recency_weights
            )
        ),
        "selection_fold_count": selection_fold_count,
        "holdout_acc": holdout_acc,
        "holdout_ec_acc": holdout_ec_acc,
        "fold_gains": selected_fold_scores - baseline_folds,
        "test_acc": float(test_scores.mean()),
        "ec_test_acc": float(ec_scores.mean()),
        "oof_dates": oof_dates,
        "oof_predictions": np.concatenate(oof_predictions),
        "test_dates": test_dates,
        "test_predictions": prediction_patterns,
    }


def restore_with_ecmwf_amplitude(
    patterns: np.ndarray, ec_fields: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    ec_vectors = np.asarray(ec_fields[:, mask], dtype=np.float64)
    means = ec_vectors.mean(axis=1, keepdims=True)
    scales = ec_vectors.std(axis=1, keepdims=True)
    centered = patterns - patterns.mean(axis=1, keepdims=True)
    normalized = centered / np.maximum(centered.std(axis=1, keepdims=True), 1e-8)
    result = np.zeros((len(patterns), *mask.shape), dtype=np.float32)
    result[:, mask] = (normalized * np.maximum(scales, 1e-8) + means).astype(
        np.float32
    )
    return result


def save_candidate_product(
    results,
    base_data_dir: Path,
    output_dir: Path,
    oof_file: Path,
    mask: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    production_dates = np.load(base_data_dir / "multi_lead_dates.npy").astype(str)
    date_to_index = {date: index for index, date in enumerate(production_dates)}
    observations = np.load(
        base_data_dir / "multi_lead_obs_results.npy", mmap_mode="r"
    )
    ec_fields = np.load(
        base_data_dir / "multi_lead_ec_precip_anom_results.npy", mmap_mode="r"
    )
    base_prediction = np.load(
        base_data_dir / "multi_lead_predict_results.npy", mmap_mode="r"
    )
    candidate_prediction = np.asarray(base_prediction).copy()
    for result in results:
        lead = result["lead"]
        dates = np.asarray([date.strftime("%Y-%m-%d") for date in result["test_dates"]])
        indices = np.asarray([date_to_index[date] for date in dates])
        restored = restore_with_ecmwf_amplitude(
            result["test_predictions"], ec_fields[indices, lead], mask
        )
        candidate_prediction[indices, lead] = restored

    np.save(output_dir / "multi_lead_dates.npy", production_dates)
    np.save(output_dir / "multi_lead_obs_results.npy", np.asarray(observations))
    np.save(
        output_dir / "multi_lead_ec_precip_anom_results.npy", np.asarray(ec_fields)
    )
    np.save(output_dir / "multi_lead_predict_results.npy", candidate_prediction)
    np.savez_compressed(
        oof_file,
        dates=np.stack(
            [
                np.asarray([date.strftime("%Y-%m-%d") for date in result["oof_dates"]])
                for result in results
            ],
            axis=1,
        ),
        predictions=np.stack(
            [result["oof_predictions"] for result in results], axis=1
        ).astype(np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-file", type=Path, default=OBS_PATH)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=experiment_path(
            "model_as_sample_transfer_run", "model_as_sample_transfer_metrics.json"
        ),
    )
    parser.add_argument("--base-data-dir", type=Path, default=BASE_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--oof-file",
        type=Path,
        default=experiment_path(
            "model_as_sample_transfer_run", "model_as_sample_transfer_oof.npz"
        ),
    )
    args = parser.parse_args()

    observations, date_to_idx, mask = load_observations(
        args.observation_file, "signed_log1p"
    )
    with np.load(args.observation_file) as data:
        latitudes = np.asarray(data["latitudes"], dtype=np.float64)
    area = np.cos(np.deg2rad(latitudes))[:, None] * np.ones(mask.shape[1])[None, :]
    point_weights = area[mask]

    raw_models = {}
    for model in SOURCE_NAMES:
        _, _, failures, raw_models[model] = build_model_fields(model)
        if failures:
            raise RuntimeError(f"{model} read failures: {failures[:3]}")

    results = []
    print(
        "lead,n_components,alpha,aux_weight,mapped_weight,recency_halflife,cv_acc,ec_cv_acc,"
        "holdout_acc,holdout_ec_acc,fold_gains,test_acc,ec_test_acc"
    )
    for lead in range(6):
        result = evaluate_lead(
            lead,
            raw_models,
            observations,
            date_to_idx,
            mask,
            point_weights,
        )
        results.append(result)
        gains = "/".join(f"{value:+.3f}" for value in result["fold_gains"])
        print(
            f'{lead},{result["n_components"]},{result["alpha"]:.0f},'
            f'{result["auxiliary_weight"]:.2f},{result["mapped_weight"]:.2f},'
            f'{result["recency_halflife_months"]},'
            f'{result["cv_acc"]:.6f},{result["ec_cv_acc"]:.6f},'
            f'{result["holdout_acc"]},{result["holdout_ec_acc"]},{gains},'
            f'{result["test_acc"]:.6f},{result["ec_test_acc"]:.6f}'
        )
    save_candidate_product(
        results, args.base_data_dir, args.output_dir, args.oof_file, mask
    )
    array_keys = {
        "oof_dates", "oof_predictions", "test_dates", "test_predictions"
    }
    serializable = []
    for result in results:
        row = {key: value for key, value in result.items() if key not in array_keys}
        row["fold_gains"] = result["fold_gains"].tolist()
        serializable.append(row)
    args.output_file.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(
        f'macro_test_acc={np.mean([item["test_acc"] for item in results]):.6f},'
        f'macro_ec_acc={np.mean([item["ec_test_acc"] for item in results]):.6f}'
    )


if __name__ == "__main__":
    main()
