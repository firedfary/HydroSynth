"""Categorical verification metrics and extreme event skills for precipitation forecasting.

Implements:
- 2x2 contingency table (Hits, Misses, False Alarms, Correct Negatives) with area weighting.
- Threat Score / Critical Success Index (TS / CSI).
- Equitable Threat Score / Gilbert Skill Score (ETS).
- Heidke Skill Score (HSS).
- Probability of Detection (POD / Hit Rate).
- False Alarm Ratio (FAR).
- Probability of False Detection (POFD).
- Frequency Bias (BIAS).
- CMA (China Meteorological Administration) Operational Precipitation Prediction Score (PS 评分).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import numpy as np


def contingency_table(
    pred_event: np.ndarray,
    obs_event: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float]:
    """Calculate area-weighted contingency table counts.

    Returns:
    --------
    hits, misses, false_alarms, correct_negatives : float
    """
    pred_b = np.asarray(pred_event, dtype=bool)
    obs_b = np.asarray(obs_event, dtype=bool)

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != pred_b.shape:
            w = np.broadcast_to(w, pred_b.shape)
    else:
        w = np.ones(pred_b.shape, dtype=np.float64)

    hits = float(np.sum(w * (pred_b & obs_b)))
    misses = float(np.sum(w * (~pred_b & obs_b)))
    false_alarms = float(np.sum(w * (pred_b & ~obs_b)))
    correct_negatives = float(np.sum(w * (~pred_b & ~obs_b)))

    return hits, misses, false_alarms, correct_negatives


def critical_success_index(
    hits: float, misses: float, false_alarms: float
) -> float:
    """Calculate Threat Score (TS) / Critical Success Index (CSI).

    CSI = H / (H + M + F)
    """
    den = hits + misses + false_alarms
    return float(hits / den) if den > 1e-20 else 0.0


def equitable_threat_score(
    hits: float, misses: float, false_alarms: float, correct_negatives: float
) -> float:
    """Calculate Equitable Threat Score (ETS) / Gilbert Skill Score.

    ETS = (H - H_r) / (H + M + F - H_r)
    where H_r = (H + M)(H + F) / Total
    """
    total = hits + misses + false_alarms + correct_negatives
    if total <= 1e-20:
        return 0.0
    h_r = (hits + misses) * (hits + false_alarms) / total
    den = hits + misses + false_alarms - h_r
    return float((hits - h_r) / den) if abs(den) > 1e-20 else 0.0


def heidke_skill_score(
    hits: float, misses: float, false_alarms: float, correct_negatives: float
) -> float:
    """Calculate Heidke Skill Score (HSS).

    HSS = 2 * (H * CN - M * F) / [(H + M)(M + CN) + (H + F)(F + CN)]
    """
    num = 2.0 * (hits * correct_negatives - misses * false_alarms)
    den = (hits + misses) * (misses + correct_negatives) + (hits + false_alarms) * (
        false_alarms + correct_negatives
    )
    return float(num / den) if abs(den) > 1e-20 else 0.0


def probability_of_detection(hits: float, misses: float) -> float:
    """Calculate Probability of Detection (POD / Recall / Hit Rate).

    POD = H / (H + M)
    """
    den = hits + misses
    return float(hits / den) if den > 1e-20 else 0.0


def false_alarm_ratio(hits: float, false_alarms: float) -> float:
    """Calculate False Alarm Ratio (FAR).

    FAR = F / (H + F)
    """
    den = hits + false_alarms
    return float(false_alarms / den) if den > 1e-20 else 0.0


def probability_of_false_detection(
    false_alarms: float, correct_negatives: float
) -> float:
    """Calculate Probability of False Detection (POFD / Fall-out).

    POFD = F / (F + CN)
    """
    den = false_alarms + correct_negatives
    return float(false_alarms / den) if den > 1e-20 else 0.0


def frequency_bias(hits: float, misses: float, false_alarms: float) -> float:
    """Calculate Frequency Bias (BIAS).

    BIAS = (H + F) / (H + M)
    """
    den = hits + misses
    return float((hits + false_alarms) / den) if den > 1e-20 else 0.0


def evaluate_event_metrics(
    pred: np.ndarray,
    obs: np.ndarray,
    operator: str,
    threshold: float,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Evaluate full suite of 2x2 contingency metrics for a given event threshold.

    Parameters:
    -----------
    pred : np.ndarray
        Forecast values.
    obs : np.ndarray
        Observed values.
    operator : str
        Comparison operator: '>' or '<'.
    threshold : float
        Threshold value.
    weights : np.ndarray, optional
        Spatial area weights.

    Returns:
    --------
    dict with csi, ets, hss, pod, far, pofd, bias, hits, misses, false_alarms, obs_freq, pred_freq.
    """
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)

    if operator == ">":
        pred_e = pred > threshold
        obs_e = obs > threshold
    elif operator == "<":
        pred_e = pred < threshold
        obs_e = obs < threshold
    else:
        raise ValueError(f"Unsupported operator '{operator}'. Use '>' or '<'.")

    h, m, f, cn = contingency_table(pred_e, obs_e, weights)
    total = h + m + f + cn

    csi = critical_success_index(h, m, f)
    ets = equitable_threat_score(h, m, f, cn)
    hss = heidke_skill_score(h, m, f, cn)
    pod = probability_of_detection(h, m)
    far = false_alarm_ratio(h, f)
    pofd = probability_of_false_detection(f, cn)
    fbias = frequency_bias(h, m, f)

    return {
        "csi": csi,
        "ets": ets,
        "hss": hss,
        "pod": pod,
        "far": far,
        "pofd": pofd,
        "bias": fbias,
        "hits": h,
        "misses": m,
        "false_alarms": f,
        "correct_negatives": cn,
        "obs_freq": float((h + m) / total) if total > 0 else 0.0,
        "pred_freq": float((h + f) / total) if total > 0 else 0.0,
    }


def cma_ps_score(
    pred_anom: np.ndarray,
    obs_anom: np.ndarray,
    weights: Optional[np.ndarray] = None,
    a: float = 1.0,
    b: float = 2.0,
) -> Dict[str, float]:
    """Calculate China Meteorological Administration (CMA) operational Ps score.

    Based on CMA seasonal/monthly precipitation anomaly 5-category evaluation standard:
    Cat 1: <= -50% (severe drought / 1类旱)
    Cat 2: (-50%, -20%] (moderate drought / 2类旱)
    Cat 3: (-20%, +20%) (normal / 正常)
    Cat 4: [+20%, +50%) (moderate wet / 2类涝)
    Cat 5: >= +50% (severe wet / 1类涝)

    Formula:
    Ps = (a * M0 + b * M1) / (a * N0 + b * Nf - M0) * 100%

    Parameters:
    -----------
    pred_anom : np.ndarray
        Forecast precipitation anomaly fraction.
    obs_anom : np.ndarray
        Observed precipitation anomaly fraction.
    weights : np.ndarray, optional
        Station/area weights.
    a : float, default 1.0
        Weight for general trend agreement.
    b : float, default 2.0
        Weight for extreme first-class anomaly (1类异常) hit.

    Returns:
    --------
    dict with 'ps_score', 'M0_trend_hits', 'M1_extreme_hits', 'N0_total', 'Nf_extremes'.
    """
    pred = np.asarray(pred_anom, dtype=np.float64)
    obs = np.asarray(obs_anom, dtype=np.float64)

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != pred.shape:
            w = np.broadcast_to(w, pred.shape)
    else:
        w = np.ones(pred.shape, dtype=np.float64)

    def classify_5(arr):
        cats = np.zeros(arr.shape, dtype=int)
        cats[arr <= -0.50] = 1
        cats[(arr > -0.50) & (arr <= -0.20)] = 2
        cats[(arr > -0.20) & (arr < 0.20)] = 3
        cats[(arr >= 0.20) & (arr < 0.50)] = 4
        cats[arr >= 0.50] = 5
        return cats

    pred_cat = classify_5(pred)
    obs_cat = classify_5(obs)

    # Direction/Trend agreement (M0):
    # Same sign: (pred dry & obs dry) or (pred normal & obs normal) or (pred wet & obs wet)
    dry_agree = (pred_cat <= 2) & (obs_cat <= 2)
    norm_agree = (pred_cat == 3) & (obs_cat == 3)
    wet_agree = (pred_cat >= 4) & (obs_cat >= 4)
    trend_agree = dry_agree | norm_agree | wet_agree

    # Extreme first-class anomaly (Cat 1 or Cat 5)
    is_obs_extreme = (obs_cat == 1) | (obs_cat == 5)
    is_pred_extreme = (pred_cat == 1) | (pred_cat == 5)
    extreme_hit = is_obs_extreme & (pred_cat == obs_cat)

    n0 = float(np.sum(w))
    nf = float(np.sum(w * is_obs_extreme))
    m0 = float(np.sum(w * trend_agree))
    m1 = float(np.sum(w * extreme_hit))

    den = a * n0 + b * nf
    ps = float((a * m0 + b * m1) / den * 100.0) if den > 1e-12 else 0.0

    return {
        "cma_ps": ps,
        "trend_agreement_m0": m0,
        "extreme_hits_m1": m1,
        "total_weight_n0": n0,
        "extreme_count_nf": nf,
        "trend_accuracy_pct": float(m0 / max(n0, 1e-12) * 100.0),
        "extreme_hit_rate_pct": float(m1 / max(nf, 1e-12) * 100.0) if nf > 0 else 0.0,
    }
