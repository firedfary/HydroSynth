"""Continuous field verification metrics for precipitation forecasting.

Supports area-weighted spatial calculations, temporal correlations,
error decomposition, skill scores relative to baselines, and hydrological metrics.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import numpy as np


def normalize_weights(
    weights: Optional[np.ndarray], shape: Tuple[int, ...]
) -> np.ndarray:
    """Ensure weights array matches spatial dimensions and sums to 1."""
    if weights is None:
        weights = np.ones(shape, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    total = np.sum(weights)
    if total <= 0:
        raise ValueError("Weights sum must be strictly positive.")
    return weights / total


def weighted_mean(
    values: np.ndarray,
    weights: Optional[np.ndarray] = None,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    """Compute weighted mean along specified axes."""
    values = np.asarray(values, dtype=np.float64)
    if weights is None:
        return np.mean(values, axis=axis, keepdims=keepdims)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != values.shape:
        weights = np.broadcast_to(weights, values.shape)
    weighted_sum = np.sum(values * weights, axis=axis, keepdims=keepdims)
    sum_weights = np.sum(weights, axis=axis, keepdims=keepdims)
    return weighted_sum / np.maximum(sum_weights, 1e-20)


def spatial_acc(
    pred: np.ndarray,
    obs: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Calculate Anomaly Correlation Coefficient (ACC) across the spatial dimension.

    Parameters:
    -----------
    pred : np.ndarray, shape (..., N_space)
        Forecast anomaly fields.
    obs : np.ndarray, shape (..., N_space)
        Observed anomaly fields.
    weights : np.ndarray, shape (N_space,), optional
        Area weights (e.g. cos(latitude)).

    Returns:
    --------
    acc : np.ndarray, shape (...)
        Spatial ACC for each leading dimension, clamped to [-1.0, 1.0].
    """
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    if pred.shape != obs.shape:
        raise ValueError(f"pred and obs must have the same shape, got {pred.shape} vs {obs.shape}")

    n_space = pred.shape[-1]
    w = normalize_weights(weights, (n_space,))

    # Weighted spatial mean for each slice
    pred_mean = np.sum(pred * w, axis=-1, keepdims=True)
    obs_mean = np.sum(obs * w, axis=-1, keepdims=True)

    pred_anom = pred - pred_mean
    obs_anom = obs - obs_mean

    covariance = np.sum(w * pred_anom * obs_anom, axis=-1)
    variance_pred = np.sum(w * pred_anom**2, axis=-1)
    variance_obs = np.sum(w * obs_anom**2, axis=-1)

    denom = np.sqrt(np.maximum(variance_pred * variance_obs, 1e-20))
    acc = covariance / denom
    return np.clip(acc, -1.0, 1.0)


def temporal_correlation(
    pred: np.ndarray,
    obs: np.ndarray,
    weights: Optional[np.ndarray] = None,
    axis: int = 0,
) -> Tuple[np.ndarray, float, float]:
    """Calculate Temporal Correlation Coefficient (TCC) for each spatial grid point.

    Parameters:
    -----------
    pred : np.ndarray, shape (N_time, N_space)
        Predicted anomaly series.
    obs : np.ndarray, shape (N_time, N_space)
        Observed anomaly series.
    weights : np.ndarray, shape (N_space,), optional
        Area weights for Fisher-z spatial averaging.
    axis : int, default 0
        Time axis.

    Returns:
    --------
    grid_tcc : np.ndarray, shape (N_space,)
        Correlation at each grid point.
    fisher_mean : float
        Fisher-z transformed weighted average correlation across space.
    median_tcc : float
        Median correlation across space.
    """
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)

    pred_anom = pred - np.mean(pred, axis=axis, keepdims=True)
    obs_anom = obs - np.mean(obs, axis=axis, keepdims=True)

    numerator = np.sum(pred_anom * obs_anom, axis=axis)
    denominator = np.sqrt(
        np.sum(pred_anom**2, axis=axis) * np.sum(obs_anom**2, axis=axis)
    )

    valid = denominator > 1e-12
    grid_tcc = np.full(numerator.shape, np.nan, dtype=np.float64)
    grid_tcc[valid] = np.clip(numerator[valid] / denominator[valid], -1.0, 1.0)

    if weights is None:
        w = np.ones(numerator.shape, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)

    valid_mask = valid & np.isfinite(grid_tcc)
    if np.any(valid_mask):
        valid_corrs = grid_tcc[valid_mask]
        valid_weights = w[valid_mask]
        fisher_z = np.arctanh(np.clip(valid_corrs, -0.999999, 0.999999))
        fisher_mean = float(np.tanh(np.average(fisher_z, weights=valid_weights)))
        median_tcc = float(np.median(valid_corrs))
    else:
        fisher_mean = float("nan")
        median_tcc = float("nan")

    return grid_tcc, fisher_mean, median_tcc


def pooled_rmse(
    pred: np.ndarray,
    obs: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Calculate overall area-weighted pooled Root Mean Squared Error."""
    error = np.asarray(pred, dtype=np.float64) - np.asarray(obs, dtype=np.float64)
    n_space = error.shape[-1]
    w = normalize_weights(weights, (n_space,))
    # Weighted mean squared error across space, then average across time
    mse_by_time = np.sum(error**2 * w, axis=-1)
    return float(np.sqrt(np.mean(mse_by_time)))


def centered_rmse(
    pred: np.ndarray,
    obs: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Calculate Centered (unbiased) Root Mean Squared Error (CRMSE).

    CRMSE^2 = RMSE^2 - Bias^2. Key metric used in Taylor diagrams.
    """
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    n_space = pred.shape[-1]
    w = normalize_weights(weights, (n_space,))

    pred_mean = np.sum(pred * w, axis=-1, keepdims=True)
    obs_mean = np.sum(obs * w, axis=-1, keepdims=True)

    pred_centered = pred - pred_mean
    obs_centered = obs - obs_mean

    c_mse_by_time = np.sum((pred_centered - obs_centered) ** 2 * w, axis=-1)
    return float(np.sqrt(np.mean(c_mse_by_time)))


def mae(
    pred: np.ndarray,
    obs: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Calculate area-weighted Mean Absolute Error."""
    error = np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(obs, dtype=np.float64))
    n_space = error.shape[-1]
    w = normalize_weights(weights, (n_space,))
    mae_by_time = np.sum(error * w, axis=-1)
    return float(np.mean(mae_by_time))


def mean_bias(
    pred: np.ndarray,
    obs: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Calculate area-weighted mean systematic bias (pred - obs)."""
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(obs, dtype=np.float64)
    n_space = diff.shape[-1]
    w = normalize_weights(weights, (n_space,))
    bias_by_time = np.sum(diff * w, axis=-1)
    return float(np.mean(bias_by_time))


def msess(
    pred: np.ndarray,
    obs: np.ndarray,
    ref: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Calculate Mean Squared Error Skill Score (MSESS) against a reference.

    MSESS = 1 - MSE(pred, obs) / MSE(ref, obs)
    Positive values indicate superior skill over reference; 1.0 is perfect.
    """
    error_pred = np.asarray(pred, dtype=np.float64) - np.asarray(obs, dtype=np.float64)
    error_ref = np.asarray(ref, dtype=np.float64) - np.asarray(obs, dtype=np.float64)
    n_space = error_pred.shape[-1]
    w = normalize_weights(weights, (n_space,))

    mse_pred = np.mean(np.sum(error_pred**2 * w, axis=-1))
    mse_ref = np.mean(np.sum(error_ref**2 * w, axis=-1))

    return float(1.0 - mse_pred / max(mse_ref, 1e-20))


def rmse_skill_pct(
    pred: np.ndarray,
    obs: np.ndarray,
    ref: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Calculate percentage reduction in RMSE relative to reference.

    (RMSE_ref - RMSE_pred) / RMSE_ref * 100%
    """
    val_pred = pooled_rmse(pred, obs, weights)
    val_ref = pooled_rmse(ref, obs, weights)
    return float(100.0 * (val_ref - val_pred) / max(val_ref, 1e-20))


def willmott_index(
    pred: np.ndarray,
    obs: np.ndarray,
    weights: Optional[np.ndarray] = None,
    modified: bool = False,
) -> float:
    """Calculate Willmott's Index of Agreement (d or d_mod).

    Standard d: 1 - sum(w * (pred - obs)^2) / sum(w * (|pred - obs_mean| + |obs - obs_mean|)^2)
    Modified d_mod: 1 - sum(w * |pred - obs|) / sum(w * (|pred - obs_mean| + |obs - obs_mean|))
    """
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    n_space = pred.shape[-1]
    w = normalize_weights(weights, (n_space,))

    obs_mean = float(np.average(obs, weights=np.broadcast_to(w, obs.shape)))

    if not modified:
        num = np.sum((pred - obs) ** 2 * w)
        den = np.sum((np.abs(pred - obs_mean) + np.abs(obs - obs_mean)) ** 2 * w)
        return float(1.0 - num / max(den, 1e-20))
    else:
        num = np.sum(np.abs(pred - obs) * w)
        den = np.sum((np.abs(pred - obs_mean) + np.abs(obs - obs_mean)) * w)
        return float(1.0 - num / max(den, 1e-20))


def kge(
    pred: np.ndarray,
    obs: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float]:
    """Calculate Kling-Gupta Efficiency (KGE) and its three components.

    KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
    where:
      r = Pearson correlation coefficient
      alpha = std(pred) / std(obs) (variability ratio)
      beta = mean(pred) / mean(obs) (bias ratio)

    Returns:
    --------
    kge : float
    r : float
    alpha : float
    beta : float
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    obs = np.asarray(obs, dtype=np.float64).ravel()
    if weights is not None:
        w = np.broadcast_to(weights, (len(pred) // len(weights), len(weights))).ravel()
        w = w / np.sum(w)
    else:
        w = np.ones_like(pred) / len(pred)

    mean_pred = float(np.sum(pred * w))
    mean_obs = float(np.sum(obs * w))

    std_pred = float(np.sqrt(np.sum((pred - mean_pred) ** 2 * w)))
    std_obs = float(np.sqrt(np.sum((obs - mean_obs) ** 2 * w)))

    cov = float(np.sum((pred - mean_pred) * (obs - mean_obs) * w))
    r = cov / max(std_pred * std_obs, 1e-20)
    r = float(np.clip(r, -1.0, 1.0))

    alpha = std_pred / max(std_obs, 1e-20)
    beta = (mean_pred + 1.0) / (mean_obs + 1.0) if abs(mean_obs) < 0.1 else mean_pred / mean_obs

    kge_val = float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    return kge_val, r, alpha, beta


def nse(
    pred: np.ndarray,
    obs: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Calculate Nash-Sutcliffe Efficiency (NSE).

    NSE = 1 - sum((pred - obs)^2) / sum((obs - mean(obs))^2)
    """
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    n_space = pred.shape[-1]
    w = normalize_weights(weights, (n_space,))

    obs_mean = np.average(obs, weights=np.broadcast_to(w, obs.shape))
    num = np.sum((pred - obs) ** 2 * w)
    den = np.sum((obs - obs_mean) ** 2 * w)
    return float(1.0 - num / max(den, 1e-20))


def spatial_spread(
    fields: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Calculate average spatial standard deviation (spread amplitude)."""
    fields = np.asarray(fields, dtype=np.float64)
    n_space = fields.shape[-1]
    w = normalize_weights(weights, (n_space,))
    field_mean = np.sum(fields * w, axis=-1, keepdims=True)
    variance = np.sum((fields - field_mean) ** 2 * w, axis=-1)
    return float(np.mean(np.sqrt(np.maximum(variance, 0.0))))


def compute_all_continuous_metrics(
    pred: np.ndarray,
    obs: np.ndarray,
    ref: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute all standard continuous evaluation metrics in a single unified dict."""
    acc_series = spatial_acc(pred, obs, weights)
    mean_acc = float(np.mean(acc_series))

    _, fisher_tcc, median_tcc = temporal_correlation(pred, obs, weights)
    p_rmse = pooled_rmse(pred, obs, weights)
    c_rmse = centered_rmse(pred, obs, weights)
    mae_val = mae(pred, obs, weights)
    bias_val = mean_bias(pred, obs, weights)
    w_index = willmott_index(pred, obs, weights)
    kge_val, r_val, alpha_val, beta_val = kge(pred, obs, weights)
    nse_val = nse(pred, obs, weights)
    spread = spatial_spread(pred, weights)
    obs_spread = spatial_spread(obs, weights)

    result = {
        "spatial_acc": mean_acc,
        "tcc_fisher_mean": fisher_tcc,
        "tcc_median": median_tcc,
        "pooled_rmse": p_rmse,
        "centered_rmse": c_rmse,
        "mae": mae_val,
        "mean_bias": bias_val,
        "willmott_index": w_index,
        "kge": kge_val,
        "kge_r": r_val,
        "kge_alpha": alpha_val,
        "kge_beta": beta_val,
        "nse": nse_val,
        "spatial_spread": spread,
        "spread_ratio": spread / max(obs_spread, 1e-20),
    }

    if ref is not None:
        ref_acc_series = spatial_acc(ref, obs, weights)
        ref_acc = float(np.mean(ref_acc_series))
        result["ref_spatial_acc"] = ref_acc
        result["acc_gain"] = mean_acc - ref_acc
        result["ref_pooled_rmse"] = pooled_rmse(ref, obs, weights)
        result["msess"] = msess(pred, obs, ref, weights)
        result["rmse_skill_pct"] = rmse_skill_pct(pred, obs, ref, weights)
        ref_spread = spatial_spread(ref, weights)
        result["ref_spread_ratio"] = ref_spread / max(obs_spread, 1e-20)

    return result
