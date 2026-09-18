"""Diagnostics, significance testing, and stratified analysis for precipitation verification.

Includes:
- Moving block bootstrap for time-correlated climate and weather series (95% CI and P(gain > 0)).
- Grid-point paired significance testing (t-test / Wilcoxon) for map stippling.
- Spatiotemporal stratified skill evaluation (by lead, season, and geographic region).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from .config import REGIONS, SEASONS
from .metrics_continuous import (
    spatial_acc,
    pooled_rmse,
    msess,
    rmse_skill_pct,
    normalize_weights,
)


def moving_block_indices(
    rng: np.random.Generator, sample_count: int, block_length: int
) -> np.ndarray:
    """Generate circular moving block bootstrap indices."""
    block_count = int(np.ceil(sample_count / block_length))
    starts = rng.integers(0, sample_count, size=block_count)
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % sample_count).ravel()[:sample_count]


def bootstrap_metric_intervals(
    model_series: np.ndarray,
    obs_series: np.ndarray,
    ref_series: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    iterations: int = 2000,
    block_length: int = 3,
    seed: int = 42,
) -> Dict[str, Dict[str, Union[float, List[float]]]]:
    """Compute moving-block bootstrap confidence intervals and gain probabilities.

    Parameters:
    -----------
    model_series : np.ndarray, shape (T, N_space)
    obs_series : np.ndarray, shape (T, N_space)
    ref_series : np.ndarray, shape (T, N_space), optional
    weights : np.ndarray, shape (N_space,), optional
    iterations : int, default 2000
    block_length : int, default 3
    seed : int, default 42

    Returns:
    --------
    dict with 'acc', 'rmse', and if ref provided, 'acc_gain', 'msess', 'rmse_skill_pct'.
    """
    rng = np.random.default_rng(seed)
    t_count = len(model_series)
    n_space = model_series.shape[-1]
    w = normalize_weights(weights, (n_space,))

    # Precalculate per-month metrics
    model_acc_by_t = spatial_acc(model_series, obs_series, w)
    model_mse_by_t = np.sum((model_series - obs_series) ** 2 * w, axis=-1)

    has_ref = ref_series is not None
    if has_ref:
        ref_acc_by_t = spatial_acc(ref_series, obs_series, w)
        ref_mse_by_t = np.sum((ref_series - obs_series) ** 2 * w, axis=-1)
        acc_gain_by_t = model_acc_by_t - ref_acc_by_t

    draws_acc = np.empty(iterations, dtype=np.float64)
    draws_rmse = np.empty(iterations, dtype=np.float64)
    if has_ref:
        draws_gain = np.empty(iterations, dtype=np.float64)
        draws_msess = np.empty(iterations, dtype=np.float64)
        draws_rmse_skill = np.empty(iterations, dtype=np.float64)

    for b in range(iterations):
        idx = moving_block_indices(rng, t_count, block_length)
        b_acc = float(np.mean(model_acc_by_t[idx]))
        b_mse = float(np.mean(model_mse_by_t[idx]))
        draws_acc[b] = b_acc
        draws_rmse[b] = np.sqrt(b_mse)

        if has_ref:
            draws_gain[b] = float(np.mean(acc_gain_by_t[idx]))
            ref_b_mse = float(np.mean(ref_mse_by_t[idx]))
            draws_msess[b] = 1.0 - (b_mse / max(ref_b_mse, 1e-20))
            ref_b_rmse = np.sqrt(ref_b_mse)
            draws_rmse_skill[b] = (ref_b_rmse - np.sqrt(b_mse)) / max(ref_b_rmse, 1e-20) * 100.0

    res = {
        "acc": {
            "mean": float(np.mean(draws_acc)),
            "ci95": np.quantile(draws_acc, [0.025, 0.975]).tolist(),
            "std": float(np.std(draws_acc)),
        },
        "rmse": {
            "mean": float(np.mean(draws_rmse)),
            "ci95": np.quantile(draws_rmse, [0.025, 0.975]).tolist(),
            "std": float(np.std(draws_rmse)),
        },
    }

    if has_ref:
        res["acc_gain"] = {
            "mean": float(np.mean(draws_gain)),
            "ci95": np.quantile(draws_gain, [0.025, 0.975]).tolist(),
            "std": float(np.std(draws_gain)),
            "prob_positive": float(np.mean(draws_gain > 0.0)),
        }
        res["msess"] = {
            "mean": float(np.mean(draws_msess)),
            "ci95": np.quantile(draws_msess, [0.025, 0.975]).tolist(),
            "std": float(np.std(draws_msess)),
            "prob_positive": float(np.mean(draws_msess > 0.0)),
        }
        res["rmse_skill_pct"] = {
            "mean": float(np.mean(draws_rmse_skill)),
            "ci95": np.quantile(draws_rmse_skill, [0.025, 0.975]).tolist(),
            "std": float(np.std(draws_rmse_skill)),
            "prob_positive": float(np.mean(draws_rmse_skill > 0.0)),
        }

    return res


def grid_point_significance(
    model_series: np.ndarray,
    ref_series: np.ndarray,
    obs_series: np.ndarray,
    metric: str = "absolute_error",
    test_type: str = "ttest",
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform paired hypothesis testing at each grid point / station.

    Tests whether model error is statistically significantly smaller than reference error.

    Parameters:
    -----------
    model_series : np.ndarray, shape (T, ...)
    ref_series : np.ndarray, shape (T, ...)
    obs_series : np.ndarray, shape (T, ...)
    metric : str, 'absolute_error' or 'squared_error'
    test_type : str, 'ttest' or 'wilcoxon'
    alpha : float, significance level (default 0.05)

    Returns:
    --------
    diff_mean : np.ndarray
        Mean difference (error_ref - error_model). Positive means model is better.
    p_values : np.ndarray
        Two-sided p-values.
    significant_mask : np.ndarray, bool
        True where model is significantly superior (p < alpha and diff_mean > 0).
    """
    model_series = np.asarray(model_series, dtype=np.float64)
    ref_series = np.asarray(ref_series, dtype=np.float64)
    obs_series = np.asarray(obs_series, dtype=np.float64)

    if metric == "absolute_error":
        err_m = np.abs(model_series - obs_series)
        err_r = np.abs(ref_series - obs_series)
    elif metric == "squared_error":
        err_m = (model_series - obs_series) ** 2
        err_r = (ref_series - obs_series) ** 2
    else:
        raise ValueError(f"Unknown metric: {metric}")

    diff = err_r - err_m  # Positive = model error smaller = model better
    t_len = diff.shape[0]
    spatial_shape = diff.shape[1:]

    diff_flat = diff.reshape(t_len, -1)
    n_pts = diff_flat.shape[1]

    diff_mean = np.mean(diff_flat, axis=0)
    p_values = np.ones(n_pts, dtype=np.float64)

    for i in range(n_pts):
        d_i = diff_flat[:, i]
        if np.all(d_i == 0) or np.isnan(d_i).all():
            p_values[i] = 1.0
            continue
        valid_d = d_i[np.isfinite(d_i)]
        if len(valid_d) < 3:
            p_values[i] = 1.0
            continue

        if test_type == "ttest":
            _, p_val = stats.ttest_1samp(valid_d, 0.0)
            p_values[i] = p_val if np.isfinite(p_val) else 1.0
        elif test_type == "wilcoxon":
            try:
                res = stats.wilcoxon(valid_d, alternative="two-sided")
                p_values[i] = res.pvalue
            except Exception:
                p_values[i] = 1.0

    diff_mean_grid = diff_mean.reshape(spatial_shape)
    p_values_grid = p_values.reshape(spatial_shape)
    sig_mask_grid = (p_values_grid < alpha) & (diff_mean_grid > 0)

    return diff_mean_grid, p_values_grid, sig_mask_grid


def evaluate_stratified_by_season_and_region(
    model_fields: np.ndarray,
    obs_fields: np.ndarray,
    dates: Union[pd.DatetimeIndex, List[str], np.ndarray],
    ref_fields: Optional[np.ndarray] = None,
    latitudes: Optional[np.ndarray] = None,
    longitudes: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    regions: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    seasons: Optional[Dict[str, Tuple[int, ...]]] = None,
) -> pd.DataFrame:
    """Compute comprehensive stratified evaluation metrics across seasons and geographic regions.

    Parameters:
    -----------
    model_fields : np.ndarray, shape (T, H, W) or (T, N_stations)
    obs_fields : np.ndarray, shape (T, H, W) or (T, N_stations)
    dates : sequence of dates corresponding to axis 0
    ref_fields : np.ndarray, optional
    latitudes : np.ndarray (H,)
    longitudes : np.ndarray (W,)
    valid_mask : np.ndarray (H, W), optional
    regions : dict of region boundaries, optional
    seasons : dict of season months, optional

    Returns:
    --------
    pd.DataFrame with stratified performance metrics.
    """
    if regions is None:
        regions = REGIONS
    if seasons is None:
        seasons = SEASONS

    dates_dt = pd.to_datetime(dates)
    has_ref = ref_fields is not None

    is_grid = model_fields.ndim == 3
    if is_grid:
        h, w = model_fields.shape[1], model_fields.shape[2]
        if latitudes is None or longitudes is None:
            raise ValueError("latitudes and longitudes required for 2D grid stratified evaluation")
        lat_2d, lon_2d = np.meshgrid(latitudes, longitudes, indexing="ij")
        area_weights_2d = np.cos(np.deg2rad(lat_2d))
        if valid_mask is None:
            valid_mask = np.ones((h, w), dtype=bool)

        scopes = [("National", "China", valid_mask)]
        for r_name, (lat_min, lat_max, lon_min, lon_max) in regions.items():
            r_mask = (
                valid_mask
                & (lat_2d >= lat_min)
                & (lat_2d < lat_max)
                & (lon_2d >= lon_min)
                & (lon_2d < lon_max)
            )
            scopes.append(("Region", r_name, r_mask))
    else:
        # Station array (T, N)
        scopes = [("National", "China", np.ones(model_fields.shape[-1], dtype=bool))]

    rows = []
    for scope_type, scope_name, s_mask in scopes:
        s_count = int(np.sum(s_mask))
        if s_count < 10:
            continue

        if is_grid:
            sub_m = model_fields[:, s_mask]
            sub_o = obs_fields[:, s_mask]
            sub_w = area_weights_2d[s_mask]
            sub_r = ref_fields[:, s_mask] if has_ref else None
        else:
            sub_m = model_fields[:, s_mask]
            sub_o = obs_fields[:, s_mask]
            sub_w = np.ones(s_count, dtype=np.float64)
            sub_r = ref_fields[:, s_mask] if has_ref else None

        # Periods: ALL plus seasons
        periods = {"ALL": np.ones(len(dates_dt), dtype=bool)}
        for s_name, s_months in seasons.items():
            periods[s_name] = np.isin(dates_dt.month, s_months)

        for period_name, t_mask in periods.items():
            t_count = int(np.sum(t_mask))
            if t_count < 2:
                continue

            pm_m = sub_m[t_mask]
            pm_o = sub_o[t_mask]
            pm_r = sub_r[t_mask] if has_ref else None

            acc_series = spatial_acc(pm_m, pm_o, sub_w)
            m_acc = float(np.mean(acc_series))
            m_rmse = pooled_rmse(pm_m, pm_o, sub_w)

            row = {
                "scope_type": scope_type,
                "region": scope_name,
                "period": period_name,
                "months": t_count,
                "stations_or_cells": s_count,
                "spatial_acc": m_acc,
                "pooled_rmse": m_rmse,
            }

            if has_ref:
                ref_acc_series = spatial_acc(pm_r, pm_o, sub_w)
                r_acc = float(np.mean(ref_acc_series))
                r_rmse = pooled_rmse(pm_r, pm_o, sub_w)
                row["ref_acc"] = r_acc
                row["acc_gain"] = m_acc - r_acc
                row["ref_rmse"] = r_rmse
                row["msess"] = msess(pm_m, pm_o, pm_r, sub_w)
                row["rmse_skill_pct"] = rmse_skill_pct(pm_m, pm_o, pm_r, sub_w)

            rows.append(row)

    return pd.DataFrame(rows)
