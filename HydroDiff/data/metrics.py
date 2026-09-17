"""Meteorological and Hydrological Spatial Evaluation Metrics.

Computes objective spatial verification scores strictly over valid mask pixels:
- ACC (Anomaly Correlation Coefficient)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Amplitude Ratio (Predicted Std / Target Std)
- Bias (Predicted Mean - Target Mean)
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
import torch


def to_numpy(tensor_or_array: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert tensor to numpy array if needed."""
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.detach().cpu().numpy()
    return np.asarray(tensor_or_array)


def compute_acc(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    mask: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Compute Anomaly Correlation Coefficient (ACC) over valid masked pixels.
    
    Formula:
        ACC = sum((p - mean(p)) * (t - mean(t))) / sqrt(sum((p - mean(p))^2) * sum((t - mean(t))^2))
    """
    p = to_numpy(pred)
    t = to_numpy(target)
    m = to_numpy(mask).astype(bool)

    valid_p = p[m]
    valid_t = t[m]

    if valid_p.size == 0 or valid_t.size == 0:
        return 0.0

    p_anom = valid_p - np.mean(valid_p)
    t_anom = valid_t - np.mean(valid_t)

    var_p = np.sum(p_anom**2)
    var_t = np.sum(t_anom**2)

    if var_p < 1e-12 or var_t < 1e-12:
        return 0.0

    acc = np.sum(p_anom * t_anom) / np.sqrt(var_p * var_t)
    return float(np.clip(acc, -1.0, 1.0))


def compute_rmse(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    mask: Union[torch.Tensor, np.ndarray],
) -> float:
    """Compute Root Mean Squared Error (RMSE) over valid masked pixels."""
    p = to_numpy(pred)
    t = to_numpy(target)
    m = to_numpy(mask).astype(bool)

    diff = p[m] - t[m]
    if diff.size == 0:
        return 0.0

    return float(np.sqrt(np.mean(diff**2)))


def compute_mae(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    mask: Union[torch.Tensor, np.ndarray],
) -> float:
    """Compute Mean Absolute Error (MAE) over valid masked pixels."""
    p = to_numpy(pred)
    t = to_numpy(target)
    m = to_numpy(mask).astype(bool)

    diff = np.abs(p[m] - t[m])
    if diff.size == 0:
        return 0.0

    return float(np.mean(diff))


def compute_spatial_metrics(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    mask: Union[torch.Tensor, np.ndarray],
) -> Dict[str, float]:
    """
    Compute a complete dictionary of spatial evaluation metrics.
    """
    p = to_numpy(pred)
    t = to_numpy(target)
    m = to_numpy(mask).astype(bool)

    valid_p = p[m]
    valid_t = t[m]

    if valid_p.size == 0 or valid_t.size == 0:
        return {
            "acc": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
            "gt_mean": 0.0,
            "gt_std": 0.0,
            "pred_mean": 0.0,
            "pred_std": 0.0,
            "bias": 0.0,
            "amplitude_ratio": 1.0,
        }

    acc = compute_acc(p, t, m)
    rmse = compute_rmse(p, t, m)
    mae = compute_mae(p, t, m)

    gt_mean = float(np.mean(valid_t))
    gt_std = float(np.std(valid_t))
    pred_mean = float(np.mean(valid_p))
    pred_std = float(np.std(valid_p))

    amplitude_ratio = float(pred_std / gt_std) if gt_std > 1e-6 else 1.0
    bias = float(pred_mean - gt_mean)

    return {
        "acc": acc,
        "rmse": rmse,
        "mae": mae,
        "gt_mean": gt_mean,
        "gt_std": gt_std,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "bias": bias,
        "amplitude_ratio": amplitude_ratio,
    }
