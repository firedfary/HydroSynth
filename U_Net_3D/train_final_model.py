"""Final Academic Production Model for Multi-Lead Precipitation Forecasting.

This script contains ONLY the final recommended model architecture and pipeline,
with all baseline/comparison models (FiLM-UNet, KernelRidge, SVR, RF, XGBoost,
unregularized residual models) completely stripped away:

1. Data Loading & Alignment:
   - CMA rebuilt station observations (1994-2010 fixed reference period).
   - Multi-model raw dynamical forecasts (ECMWF SEAS5, NCEP CFSv2, JMA CPS3).
   - Parameter-free signed-log1p fractional anomaly transform: z = sign(x)*log1p(|x|).

2. Engine 1: Recency-Decayed Multi-Model Transfer Learning:
   - Paired dynamical forecast -> observation training samples.
   - Expanding TimeSeriesSplit(n_splits=5, test_size=21).
   - In-fold model climatologies, input PCA (10-80 components), target PCA (40% variance).
   - Ridge regression with lead-dependent exponential recency decay (0, 60, 120 months)
     and auxiliary source weighting, tuned via rolling OOF spatial ACC.

3. Engine 2: Seasonal Forecast Stacking:
   - Four physical predictors: ECMWF, NCEP, recent 1-3 month lag mean, 12-month annual lag.
   - Season-dependent (DJF, MAM, JJA, SON) simplex convex optimization with L2 shrinkage
     towards ECMWF, tuned via rolling OOF.

4. Academic Pure Model Blending & Amplitude Restoration:
   - Lead-dependent OOF ACC-weighted convex blend of Transfer and Stacking predictions.
   - ECMWF spatial pattern amplitude restoration (spatial mean and variance).
   - Note: Ad-hoc seasonal safety fallback to ECMWF is completely removed per academic
     research conventions, ensuring 100% pure autonomous model output.

5. Verification & Persistence:
   - Saves final multi-lead arrays, dates, observations, and configuration JSON.
   - Evaluates and displays area-weighted ACC, RMSE, and ECMWF baseline metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

# Ensure project paths and root imports
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import OBSERVATION_FILE, paths
from analyze_multimodel_raw import (
    EXCLUDED,
    START,
    TEST_END,
    aligned_dates,
    anomalies_for_dates,
    build_model_fields,
    standardize,
)

# Core Constants
SOURCE_NAMES = ("ECMWF", "NCEP", "JMA")
NUM_TEST = 21
N_COMPONENTS_CANDIDATES = (10, 20, 40, 80)
ALPHAS = (1.0, 10.0, 100.0, 1000.0)
AUXILIARY_WEIGHTS = (0.25, 0.5, 1.0)
MAPPED_WEIGHTS = (0.25, 0.5, 0.75, 1.0)
RECENCY_HALFLIVES = (0, 60, 120)  # months: 0 means no recency decay
STACKING_PENALTIES = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05)

SEASON_BY_MONTH = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


# ==============================================================================
# 1. Mathematical & Spatial Utilities
# ==============================================================================

def transform_fractional_anomaly(values: np.ndarray, transform: str = "signed_log1p") -> np.ndarray:
    """Apply parameter-free, monotonic signed-log1p transform."""
    if transform == "none":
        return values
    if transform == "signed_log1p":
        return np.sign(values) * np.log1p(np.abs(values))
    raise ValueError(f"Unknown transform: {transform}")


def load_observations(path: Path, transform: str = "signed_log1p") -> tuple[np.ndarray, dict, np.ndarray]:
    """Load reconstructed observation target with fixed 1994-2010 reference."""
    if not path.exists():
        raise FileNotFoundError(
            f"Observation file not found: {path}\n"
            f"Please run rebuild_station_observations.py first."
        )
    with np.load(path) as data:
        observations = np.asarray(data["anomaly_fraction"], dtype=np.float32)
        observation_dates = pd.to_datetime(data["dates"].astype(str))
        mask = np.asarray(data["valid_mask"], dtype=bool)

    date_to_idx = {
        pd.Timestamp(date): index for index, date in enumerate(observation_dates)
    }
    transformed_obs = transform_fractional_anomaly(observations, transform)
    return np.nan_to_num(transformed_obs, nan=0.0), date_to_idx, mask


def get_area_weights(mask: np.ndarray) -> np.ndarray:
    """Calculate normalized cosine latitude area weights for valid grid cells."""
    latitudes = np.arange(59.75, -0.25, -0.5) if mask.shape[0] == 120 else np.arange(60.0, 0.0, -0.5)
    if len(latitudes) != mask.shape[0]:
        latitudes = np.linspace(59.75, 0.25, mask.shape[0])
    area_2d = np.cos(np.deg2rad(latitudes))[:, None] * np.ones(mask.shape[1])[None, :]
    point_weights = area_2d[mask].astype(np.float64)
    return point_weights / point_weights.sum()


def weighted_pattern_vector(
    field: np.ndarray, mask: np.ndarray, point_weights: np.ndarray
) -> np.ndarray:
    """Extract spatial pattern normalized to zero area-weighted mean and unit variance."""
    values = np.nan_to_num(field, nan=0.0)[mask].astype(np.float64)
    mean = np.sum(values * point_weights) / np.sum(point_weights)
    centered = values - mean
    variance = np.sum(point_weights * centered**2) / np.sum(point_weights)
    return (centered / np.sqrt(variance + 1e-12)).astype(np.float32)


def weighted_pattern_vector_row(row: np.ndarray, point_weights: np.ndarray) -> np.ndarray:
    """Normalize a 1D vector to zero weighted mean and unit variance."""
    mean = np.sum(row * point_weights) / np.sum(point_weights)
    centered = row - mean
    variance = np.sum(point_weights * centered**2) / np.sum(point_weights)
    return (centered / np.sqrt(variance + 1e-12)).astype(np.float32)


def weighted_row_acc(
    prediction: np.ndarray, target: np.ndarray, point_weights: np.ndarray
) -> np.ndarray:
    """Calculate area-weighted spatial anomaly correlation coefficient (ACC)."""
    pred_mean = np.sum(prediction * point_weights, axis=1, keepdims=True) / np.sum(point_weights)
    target_mean = np.sum(target * point_weights, axis=1, keepdims=True) / np.sum(point_weights)
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
        where=denominator > 1e-12,
    )


def standardize_vectors(values: np.ndarray) -> np.ndarray:
    """Standardize each spatial vector along axis 1."""
    means = values.mean(axis=1, keepdims=True)
    scales = values.std(axis=1, keepdims=True)
    return (values - means) / np.maximum(scales, 1e-8)


def restore_with_ecmwf_amplitude(
    patterns: np.ndarray, ec_fields: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Restore standardized patterns to ECMWF spatial mean and variance scale."""
    ec_vectors = np.asarray(ec_fields[:, mask], dtype=np.float64)
    means = ec_vectors.mean(axis=1, keepdims=True)
    scales = ec_vectors.std(axis=1, keepdims=True)
    centered = patterns - patterns.mean(axis=1, keepdims=True)
    normalized = centered / np.maximum(centered.std(axis=1, keepdims=True), 1e-8)
    result = np.zeros((len(patterns), *mask.shape), dtype=np.float32)
    result[:, mask] = (normalized * np.maximum(scales, 1e-8) + means).astype(np.float32)
    return result


def calendar_features(dates: list[pd.Timestamp]) -> np.ndarray:
    """Cyclical month sinusoidal embedding."""
    angles = np.asarray([2.0 * np.pi * date.month / 12.0 for date in dates])
    return np.column_stack([np.sin(angles), np.cos(angles)]).astype(np.float32)


# ==============================================================================
# 2. Multi-Model Forecast Caching & Loading
# ==============================================================================

def load_cached_model_fields(model: str) -> dict[pd.Timestamp, np.ndarray]:
    """Load raw model fields from cache if available, otherwise read and cache."""
    cache_path = paths.cache_dir / f"{model}_raw_fields_cache.npz"
    if cache_path.exists():
        print(f"Loading {model} fields from cache: {cache_path.name}")
        with np.load(cache_path, allow_pickle=True) as cache:
            date_strings = cache["dates"]
            arrays = cache["fields"]
            return {
                pd.Timestamp(str(d)): arr for d, arr in zip(date_strings, arrays)
            }

    print(f"Building {model} fields from raw NetCDF files...")
    _, _, failures, raw_by_issue = build_model_fields(model)
    if failures:
        print(f"Warning: {len(failures)} file reading failures for {model}: {failures[:2]}")

    try:
        paths.cache_dir.mkdir(parents=True, exist_ok=True)
        sorted_items = sorted(raw_by_issue.items())
        cache_dates = np.asarray([str(d) for d, _ in sorted_items])
        cache_arrays = np.asarray([arr for _, arr in sorted_items], dtype=np.float32)
        np.savez_compressed(cache_path, dates=cache_dates, fields=cache_arrays)
        print(f"Cached {model} fields to {cache_path}")
    except Exception as exc:
        print(f"Notice: Cache saving skipped ({exc})")

    return raw_by_issue


# ==============================================================================
# 3. Engine 1: Recency-Decayed Multi-Model Transfer Learning
# ==============================================================================

class RecencyTransferEngine:
    """Model-as-sample transfer learning with lead-dependent recency decay."""

    def __init__(
        self,
        raw_models: dict[str, dict[pd.Timestamp, np.ndarray]],
        observations: np.ndarray,
        date_to_idx: dict[pd.Timestamp, int],
        mask: np.ndarray,
        point_weights: np.ndarray,
    ):
        self.raw_models = raw_models
        self.observations = observations
        self.date_to_idx = date_to_idx
        self.mask = mask
        self.point_weights = point_weights

    def _model_anomalies(self, lead: int, climatology_dates: list[pd.Timestamp], requested_dates: list[pd.Timestamp]):
        return {
            model: {
                date: transform_fractional_anomaly(field, "signed_log1p")
                for date, field in anomalies_for_dates(
                    self.raw_models[model], lead, climatology_dates, requested_dates
                ).items()
            }
            for model in SOURCE_NAMES
        }

    def _build_training_samples(self, dates: list[pd.Timestamp], fields: dict):
        input_rows = []
        targets = []
        source_indices = []
        sample_dates = []
        for source_index, model in enumerate(SOURCE_NAMES):
            for date in dates:
                if date not in fields[model]:
                    continue
                input_rows.append(
                    weighted_pattern_vector(fields[model][date], self.mask, self.point_weights)
                )
                targets.append(
                    weighted_pattern_vector(
                        self.observations[self.date_to_idx[date]], self.mask, self.point_weights
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

    def _prepare_fold(self, lead: int, train_dates: list[pd.Timestamp], eval_dates: list[pd.Timestamp]):
        requested = train_dates + eval_dates
        fields = self._model_anomalies(lead, train_dates, requested)
        train_input, train_target, train_sources, sample_dates = self._build_training_samples(
            train_dates, fields
        )
        eval_input = np.stack(
            [
                weighted_pattern_vector(fields["ECMWF"][date], self.mask, self.point_weights)
                for date in eval_dates
            ]
        )
        eval_target = np.stack(
            [
                weighted_pattern_vector(
                    self.observations[self.date_to_idx[date]], self.mask, self.point_weights
                )
                for date in eval_dates
            ]
        )

        max_components = min(max(N_COMPONENTS_CANDIDATES), len(train_input) - 1, train_input.shape[1])
        input_pca = PCA(
            n_components=max_components,
            svd_solver="randomized",
            whiten=True,
            random_state=42,
        )
        train_scores = input_pca.fit_transform(train_input)
        eval_scores = input_pca.transform(eval_input)

        # Unique dates target PCA to avoid biasing target basis towards multi-model availability
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

    def _mapped_prediction(self, fold: dict, spec: tuple):
        n_components, alpha, auxiliary_weight, mapped_weight, recency_halflife = spec
        one_hot_train = np.eye(len(SOURCE_NAMES), dtype=np.float32)[fold["train_sources"]]
        train_features = np.concatenate(
            [fold["train_scores"][:, :n_components], calendar_features(fold["sample_dates"]), one_hot_train],
            axis=1,
        )

        eval_sources = np.zeros(len(fold["eval_dates"]), dtype=np.int64)
        one_hot_eval = np.eye(len(SOURCE_NAMES), dtype=np.float32)[eval_sources]
        eval_features = np.concatenate(
            [fold["eval_scores"][:, :n_components], calendar_features(fold["eval_dates"]), one_hot_eval],
            axis=1,
        )

        sample_weight = np.where(fold["train_sources"] == 0, 1.0, auxiliary_weight)
        if recency_halflife > 0:
            latest = max(fold["sample_dates"])
            ages = np.asarray(
                [(latest.year - d.year) * 12 + latest.month - d.month for d in fold["sample_dates"]],
                dtype=np.float64,
            )
            sample_weight *= np.power(0.5, ages / recency_halflife)
            sample_weight *= len(sample_weight) / sample_weight.sum()

        model = Ridge(alpha=alpha)
        model.fit(train_features, fold["target_scores"], sample_weight=sample_weight)
        mapped = fold["target_pca"].inverse_transform(model.predict(eval_features))
        mapped = np.stack(
            [weighted_pattern_vector_row(row, self.point_weights) for row in mapped]
        )
        return (1.0 - mapped_weight) * fold["eval_input"] + mapped_weight * mapped

    def evaluate_lead(self, lead: int) -> dict:
        dates = aligned_dates(self.raw_models["ECMWF"], lead)
        all_train_dates = dates[:-NUM_TEST]
        test_dates = dates[-NUM_TEST:]

        # 5-fold expanding time-series split
        tscv = TimeSeriesSplit(n_splits=5, test_size=NUM_TEST)
        folds = [
            self._prepare_fold(lead, [all_train_dates[i] for i in train_idx], [all_train_dates[i] for i in val_idx])
            for train_idx, val_idx in tscv.split(all_train_dates)
        ]
        test_fold = self._prepare_fold(lead, all_train_dates, test_dates)

        fold_weights = (np.arange(len(folds)) + 1.0) ** 2
        fold_weights /= fold_weights.sum()

        # Grid search over candidate hyperparameter specifications
        specs = []
        for n_comp in N_COMPONENTS_CANDIDATES:
            for alpha in ALPHAS:
                for aux_w in AUXILIARY_WEIGHTS:
                    for map_w in MAPPED_WEIGHTS:
                        for hl in RECENCY_HALFLIVES:
                            specs.append((n_comp, alpha, aux_w, map_w, hl))

        spec_scores = []
        for spec in specs:
            fold_accs = []
            for fold in folds:
                pred = self._mapped_prediction(fold, spec)
                acc = float(np.mean(weighted_row_acc(pred, fold["eval_target"], self.point_weights)))
                fold_accs.append(acc)
            score = float(np.sum(np.asarray(fold_accs) * fold_weights))
            spec_scores.append(score)

        best_spec = specs[int(np.argmax(spec_scores))]
        best_fold_accs = [
            float(np.mean(weighted_row_acc(self._mapped_prediction(fold, best_spec), fold["eval_target"], self.point_weights)))
            for fold in folds
        ]

        # Generate Test Predictions & Out-of-fold (OOF) Predictions
        test_pred = self._mapped_prediction(test_fold, best_spec)
        oof_preds = [self._mapped_prediction(fold, best_spec) for fold in folds]
        oof_dates = [d for fold in folds for d in fold["eval_dates"]]

        return {
            "lead": lead,
            "n_components": best_spec[0],
            "alpha": best_spec[1],
            "auxiliary_weight": best_spec[2],
            "mapped_weight": best_spec[3],
            "recency_halflife_months": best_spec[4],
            "cv_acc": float(np.sum(np.asarray(best_fold_accs) * fold_weights)),
            "test_dates": test_dates,
            "test_predictions": test_pred,
            "oof_dates": oof_dates,
            "oof_predictions": np.concatenate(oof_preds, axis=0),
        }


# ==============================================================================
# 4. Engine 2: Seasonal Forecast Stacking
# ==============================================================================

def simplex_weights_grid(step: float = 0.05) -> np.ndarray:
    """Generate discrete simplex grid for 4 weights (sum=1)."""
    units = int(round(1.0 / step))
    rows = []
    for ec in range(units + 1):
        for ncep in range(units - ec + 1):
            for recent in range(units - ec - ncep + 1):
                annual = units - ec - ncep - recent
                rows.append((ec, ncep, recent, annual))
    return np.asarray(rows, dtype=np.float32) / units


class SeasonalStackingEngine:
    """Season-dependent convex forecast stacking with shrinkage to ECMWF."""

    def __init__(
        self,
        raw_models: dict[str, dict[pd.Timestamp, np.ndarray]],
        observations: np.ndarray,
        date_to_idx: dict[pd.Timestamp, int],
        mask: np.ndarray,
    ):
        self.raw_models = raw_models
        self.observations = observations
        self.date_to_idx = date_to_idx
        self.mask = mask
        self.weights = simplex_weights_grid(0.05)

    def _build_arrays(self, dates: list[pd.Timestamp], lead: int):
        train_dates = dates[:-NUM_TEST]
        ec_anoms = anomalies_for_dates(self.raw_models["ECMWF"], lead, train_dates, dates)
        ncep_anoms = anomalies_for_dates(self.raw_models["NCEP"], lead, train_dates, dates)

        ec_vectors = np.stack(
            [standardize(np.nan_to_num(ec_anoms[d], nan=0.0), self.mask)[self.mask] for d in dates]
        )
        ncep_vectors = np.stack(
            [standardize(np.nan_to_num(ncep_anoms[d], nan=0.0), self.mask)[self.mask] for d in dates]
        )

        recent_vectors = []
        annual_vectors = []
        target_vectors = []

        for d in dates:
            recents = [
                standardize(
                    self.observations[self.date_to_idx[d - pd.DateOffset(months=lead + offset)]],
                    self.mask,
                )[self.mask]
                for offset in (1, 2, 3)
            ]
            recent_vectors.append(np.mean(recents, axis=0))
            annual_vectors.append(
                standardize(
                    self.observations[self.date_to_idx[d - pd.DateOffset(months=12)]],
                    self.mask,
                )[self.mask]
            )
            target_vectors.append(
                standardize(self.observations[self.date_to_idx[d]], self.mask)[self.mask]
            )

        return (
            ec_vectors,
            ncep_vectors,
            np.stack(recent_vectors),
            np.stack(annual_vectors),
            np.stack(target_vectors),
        )

    def evaluate_lead(self, lead: int) -> dict:
        dates = aligned_dates(self.raw_models["ECMWF"], lead)
        all_train_dates = dates[:-NUM_TEST]
        test_dates = dates[-NUM_TEST:]

        ec, ncep, recent, annual, target = self._build_arrays(dates, lead)
        train_count = len(all_train_dates)

        # 5-fold expanding time-series split
        tscv = TimeSeriesSplit(n_splits=5, test_size=NUM_TEST)
        oof_predictions = np.zeros((5 * NUM_TEST, int(self.mask.sum())), dtype=np.float32)
        oof_dates = []
        selected_weights_by_season = {}

        for season in ("DJF", "MAM", "JJA", "SON"):
            season_mask = np.asarray([SEASON_BY_MONTH[d.month] == season for d in all_train_dates])
            best_penalty = 0.0
            best_score = -1e9
            best_w = None

            for penalty in STACKING_PENALTIES:
                penalty_scores = []
                for train_idx, val_idx in tscv.split(all_train_dates):
                    s_train = train_idx[season_mask[train_idx]]
                    if len(s_train) == 0:
                        continue
                    # Candidate evaluation
                    X_pool = np.stack(
                        [ec[s_train], ncep[s_train], recent[s_train], annual[s_train]], axis=1
                    )
                    # Optimize on training slice
                    shrink_diff = self.weights - np.asarray([1.0, 0.0, 0.0, 0.0])
                    reg = penalty * np.sum(shrink_diff**2, axis=1)

                    # Maximize mean spatial correlation
                    accs = []
                    for w in self.weights:
                        pred = (
                            w[0] * ec[s_train]
                            + w[1] * ncep[s_train]
                            + w[2] * recent[s_train]
                            + w[3] * annual[s_train]
                        )
                        pred = standardize_vectors(pred)
                        dot = np.mean(np.sum(pred * target[s_train], axis=1) / target[s_train].shape[1])
                        accs.append(dot)
                    w_opt = self.weights[int(np.argmax(np.asarray(accs) - reg))]

                    # Validate on validation slice
                    s_val = val_idx[season_mask[val_idx]]
                    if len(s_val) > 0:
                        pred_val = standardize_vectors(
                            w_opt[0] * ec[s_val]
                            + w_opt[1] * ncep[s_val]
                            + w_opt[2] * recent[s_val]
                            + w_opt[3] * annual[s_val]
                        )
                        val_acc = np.mean(np.sum(pred_val * target[s_val], axis=1) / target[s_val].shape[1])
                        penalty_scores.append(val_acc)

                if penalty_scores and np.mean(penalty_scores) > best_score:
                    best_score = float(np.mean(penalty_scores))
                    best_penalty = penalty

            # Fit optimal weights using full training set for this season
            s_all = np.where(season_mask)[0]
            shrink_diff = self.weights - np.asarray([1.0, 0.0, 0.0, 0.0])
            reg = best_penalty * np.sum(shrink_diff**2, axis=1)
            full_accs = []
            for w in self.weights:
                p = (
                    w[0] * ec[s_all]
                    + w[1] * ncep[s_all]
                    + w[2] * recent[s_all]
                    + w[3] * annual[s_all]
                )
                p = standardize_vectors(p)
                dot = np.mean(np.sum(p * target[s_all], axis=1) / target[s_all].shape[1])
                full_accs.append(dot)
            best_w = self.weights[int(np.argmax(np.asarray(full_accs) - reg))]
            selected_weights_by_season[season] = best_w.tolist()

        # Fill OOF predictions across the 5 validation folds
        oof_pointer = 0
        for train_idx, val_idx in tscv.split(all_train_dates):
            for idx in val_idx:
                d = all_train_dates[idx]
                season = SEASON_BY_MONTH[d.month]
                w = np.asarray(selected_weights_by_season[season])
                oof_predictions[oof_pointer] = standardize_vectors(
                    (w[0] * ec[idx] + w[1] * ncep[idx] + w[2] * recent[idx] + w[3] * annual[idx])[None, :]
                )[0]
                oof_dates.append(d)
                oof_pointer += 1

        # Test set predictions
        test_predictions = np.zeros((NUM_TEST, int(self.mask.sum())), dtype=np.float32)
        for i, d in enumerate(test_dates):
            global_idx = train_count + i
            season = SEASON_BY_MONTH[d.month]
            w = np.asarray(selected_weights_by_season[season])
            test_predictions[i] = standardize_vectors(
                (w[0] * ec[global_idx] + w[1] * ncep[global_idx] + w[2] * recent[global_idx] + w[3] * annual[global_idx])[None, :]
            )[0]

        return {
            "lead": lead,
            "selected_weights_by_season": selected_weights_by_season,
            "oof_dates": oof_dates,
            "oof_predictions": oof_predictions,
            "test_dates": test_dates,
            "test_predictions": test_predictions,
        }


# ==============================================================================
# 5. Integration, Academic Ensemble & Execution Pipeline
# ==============================================================================

def select_oof_blend_weights(
    transfer_oof: np.ndarray,
    stacking_oof: np.ndarray,
    targets: np.ndarray,
    area_weights: np.ndarray,
    fold_size: int = NUM_TEST,
) -> np.ndarray:
    """Find convex blend weight w in [0, 1] maximizing weighted OOF spatial ACC."""
    candidates = np.linspace(0.0, 1.0, 21)
    sample_count, lead_count, _ = transfer_oof.shape
    fold_indices = np.minimum(np.arange(sample_count) // fold_size, 4)
    sample_weights = (fold_indices + 1.0) ** 2
    sample_weights /= sample_weights.sum()

    selected_weights = np.empty(lead_count, dtype=np.float64)
    for lead in range(lead_count):
        trans_p = standardize_vectors(transfer_oof[:, lead])
        stack_p = standardize_vectors(stacking_oof[:, lead])
        scores = []
        for w in candidates:
            blended = standardize_vectors(w * trans_p + (1.0 - w) * stack_p)
            acc = weighted_row_acc(blended, targets[:, lead], area_weights)
            scores.append(float(np.sum(acc * sample_weights)))
        best_idx = int(np.argmax(scores))
        selected_weights[lead] = candidates[best_idx]
    return selected_weights


def main():
    parser = argparse.ArgumentParser(
        description="Train the final academic precipitation model without comparison baselines."
    )
    parser.add_argument(
        "--observation-file",
        type=Path,
        default=OBSERVATION_FILE,
        help="Path to reconstructed observation NPZ file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.get_exp_dir("final_model"),
        help="Directory to save final model outputs and evaluation summaries.",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("HydroSynth U_Net_3D Final Academic Model Pipeline")
    print(f"Observation file: {args.observation_file}")
    print(f"Output directory: {out_dir}")
    print("=" * 80)

    # 1. Load Observations
    observations, date_to_idx, mask = load_observations(args.observation_file, "signed_log1p")
    area_weights = get_area_weights(mask)
    print(f"Loaded observations: {len(observations)} months, {int(mask.sum())} valid land points.")

    # 2. Load Raw Dynamical Forecasts (with disk cache)
    raw_models = {}
    for model in SOURCE_NAMES:
        raw_models[model] = load_cached_model_fields(model)

    # 3. Train Engine 1: Recency-Decayed Multi-Model Transfer
    print("\n" + "-" * 80)
    print("Training Engine 1: Recency-Decayed Multi-Model Transfer Learning (ECMWF/NCEP/JMA)...")
    print("-" * 80)
    transfer_engine = RecencyTransferEngine(
        raw_models, observations, date_to_idx, mask, area_weights
    )
    transfer_results = []
    for lead in range(6):
        t0 = time.time()
        res = transfer_engine.evaluate_lead(lead)
        transfer_results.append(res)
        print(
            f"Lead {lead} (took {time.time()-t0:.1f}s) -> "
            f"PCA: {res['n_components']} | Alpha: {res['alpha']:.0f} | "
            f"AuxWeight: {res['auxiliary_weight']:.2f} | HalfLife: {res['recency_halflife_months']}m | "
            f"OOF ACC: {res['cv_acc']:.4f}"
        )

    # 4. Train Engine 2: Seasonal Forecast Stacking
    print("\n" + "-" * 80)
    print("Training Engine 2: Seasonal Forecast Stacking (ECMWF + NCEP + Observation Lags)...")
    print("-" * 80)
    stacking_engine = SeasonalStackingEngine(raw_models, observations, date_to_idx, mask)
    stacking_results = []
    for lead in range(6):
        t0 = time.time()
        res = stacking_engine.evaluate_lead(lead)
        stacking_results.append(res)
        print(f"Lead {lead} (took {time.time()-t0:.1f}s) -> Seasonal weights fitted.")

    # 5. Integrate & Ensemble (Academic Pure Version: NO Seasonal Fallback Gate)
    print("\n" + "-" * 80)
    print("Ensemble Integration & Spatial Amplitude Restoration...")
    print("-" * 80)

    # Assemble OOF Arrays
    oof_transfer = np.stack([r["oof_predictions"] for r in transfer_results], axis=1)
    oof_stacking = np.stack([r["oof_predictions"] for r in stacking_results], axis=1)
    test_transfer = np.stack([r["test_predictions"] for r in transfer_results], axis=1)
    test_stacking = np.stack([r["test_predictions"] for r in stacking_results], axis=1)
    test_dates = transfer_results[0]["test_dates"]

    # Assemble OOF targets
    oof_dates = transfer_results[0]["oof_dates"]
    oof_targets = np.stack(
        [
            np.stack([observations[date_to_idx[d]][mask] for d in oof_dates])
            for _ in range(6)
        ],
        axis=1,
    )

    # Select optimal blend weights on historical OOF
    transfer_blend_weights = select_oof_blend_weights(
        oof_transfer, oof_stacking, oof_targets, area_weights
    )

    # Assemble ECMWF benchmark fields and Observation targets on test set
    test_obs = np.stack([observations[date_to_idx[d]] for d in test_dates])  # (21, 120, 140)
    test_ec_raw = np.zeros((NUM_TEST, 6, *mask.shape), dtype=np.float32)
    train_dates = aligned_dates(raw_models["ECMWF"], 0)[:-NUM_TEST]

    for lead in range(6):
        ec_anoms = anomalies_for_dates(raw_models["ECMWF"], lead, train_dates, test_dates)
        for i, d in enumerate(test_dates):
            field = transform_fractional_anomaly(ec_anoms[d], "signed_log1p")
            test_ec_raw[i, lead] = field

    # Generate Final Forecast Fields
    final_predictions = np.zeros((NUM_TEST, 6, *mask.shape), dtype=np.float32)
    obs_all_leads = np.zeros((NUM_TEST, 6, *mask.shape), dtype=np.float32)

    for lead in range(6):
        w = transfer_blend_weights[lead]
        p_trans = standardize_vectors(test_transfer[:, lead])
        p_stack = standardize_vectors(test_stacking[:, lead])
        blended = standardize_vectors(w * p_trans + (1.0 - w) * p_stack)

        # ECMWF Amplitude Restoration (mean and standard deviation)
        ec_lead = test_ec_raw[:, lead]
        restored = restore_with_ecmwf_amplitude(blended, ec_lead, mask)
        final_predictions[:, lead] = restored
        obs_all_leads[:, lead] = test_obs

    # 6. Save Artifacts
    date_strs = np.asarray([d.strftime("%Y-%m-%d") for d in test_dates])
    pred_file = out_dir / "multi_lead_predict_results_final.npy"
    dates_file = out_dir / "multi_lead_dates.npy"
    obs_file = out_dir / "multi_lead_obs_results.npy"
    ec_file = out_dir / "multi_lead_ec_precip_anom_results.npy"
    config_file = out_dir / "final_model_config_and_metrics.json"

    np.save(pred_file, final_predictions)
    np.save(dates_file, date_strs)
    np.save(obs_file, obs_all_leads)
    np.save(ec_file, test_ec_raw)

    # 7. Compute & Print Comprehensive Evaluation
    print("\n" + "=" * 85)
    print(f"{'Lead':<6}{'Transfer W':<12}{'Model ACC':<12}{'ECMWF ACC':<12}{'ACC Gain':<12}{'Model RMSE':<12}{'ECMWF RMSE':<12}")
    print("=" * 85)

    metrics_summary = []
    for lead in range(6):
        w = transfer_blend_weights[lead]
        pred_vec = final_predictions[:, lead][:, mask]
        ec_vec = test_ec_raw[:, lead][:, mask]
        target_vec = test_obs[:, mask]

        model_acc = float(np.mean(weighted_row_acc(pred_vec, target_vec, area_weights)))
        ec_acc = float(np.mean(weighted_row_acc(ec_vec, target_vec, area_weights)))
        acc_gain = model_acc - ec_acc

        model_rmse = float(np.sqrt(np.mean(np.sum(area_weights[None, :] * (pred_vec - target_vec)**2, axis=1))))
        ec_rmse = float(np.sqrt(np.mean(np.sum(area_weights[None, :] * (ec_vec - target_vec)**2, axis=1))))

        metrics_summary.append({
            "lead": lead,
            "transfer_weight": float(w),
            "stacking_weight": float(1.0 - w),
            "model_acc": model_acc,
            "ecmwf_acc": ec_acc,
            "acc_gain": acc_gain,
            "model_rmse": model_rmse,
            "ecmwf_rmse": ec_rmse,
            "transfer_hyperparams": {
                "n_components": transfer_results[lead]["n_components"],
                "alpha": transfer_results[lead]["alpha"],
                "aux_weight": transfer_results[lead]["auxiliary_weight"],
                "recency_halflife_months": transfer_results[lead]["recency_halflife_months"],
            },
        })
        print(
            f"Lead {lead:<2}{w:<12.2f}{model_acc:<12.4f}{ec_acc:<12.4f}{acc_gain:<+12.4f}{model_rmse:<12.4f}{ec_rmse:<12.4f}"
        )

    macro_model_acc = float(np.mean([m["model_acc"] for m in metrics_summary]))
    macro_ec_acc = float(np.mean([m["ecmwf_acc"] for m in metrics_summary]))
    macro_acc_gain = macro_model_acc - macro_ec_acc
    macro_model_rmse = float(np.mean([m["model_rmse"] for m in metrics_summary]))
    macro_ec_rmse = float(np.mean([m["ecmwf_rmse"] for m in metrics_summary]))

    print("-" * 85)
    print(
        f"{'Macro':<6}{'--':<12}{macro_model_acc:<12.4f}{macro_ec_acc:<12.4f}{macro_acc_gain:<+12.4f}{macro_model_rmse:<12.4f}{macro_ec_rmse:<12.4f}"
    )
    print("=" * 85)

    full_manifest = {
        "model_name": "HydroSynth_U_Net_3D_Final_Model",
        "description": "Recency-Decayed Multi-Model Transfer + Seasonal Stacking Pure Ensemble (No Fallback Gate)",
        "test_period": ["2023-01-01", "2024-09-01"],
        "macro_metrics": {
            "model_acc": macro_model_acc,
            "ecmwf_acc": macro_ec_acc,
            "acc_gain": macro_acc_gain,
            "model_rmse": macro_model_rmse,
            "ecmwf_rmse": macro_ec_rmse,
        },
        "lead_metrics": metrics_summary,
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(full_manifest, f, indent=2)

    print(f"\nAll artifacts successfully written to: {out_dir}")
    print(f"- Final Predictions: {pred_file.name}")
    print(f"- Config & Metrics:  {config_file.name}")


if __name__ == "__main__":
    main()
