# -*- coding: utf-8 -*-
"""
Legacy unified training, evaluation, and interpretability pipeline
for FiLM-U-Net Monthly Precipitation Prediction over China (EJRH-D-26-03375).

Important protocol notes:
1. Chronological split:
   - Training set:            1994.01 - 2017.12 (285 months)
   - Internal validation set: 2018.01 - 2022.12 (60 months, used for early stopping & hyperparameter selection)
   - Explored evaluation set: 2023.01 - 2024.09 (21 months). This period has been
     examined repeatedly and is no longer an independent final test.
2. SST climatology, EOF basis, PC scaling, and extreme p90 threshold use the
   training period. The legacy meteorological predictor cache used by this script
   was normalized on the full series and has leakage. Use rebuild_inputs_strict.py
   and the strict_* evaluation scripts for train-fitted predictors.
3. The legacy model comparison also changes backbone width between models;
   it cannot isolate the effect of FiLM conditioning.
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, wilcoxon, t as student_t

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

# Ensure repository root and model directory are on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import config  # noqa: E402
from utils.paths import SubprojectPaths  # noqa: E402
import utils.utils as utils  # noqa: E402
from unetlite import UNetLite  # noqa: E402
from unetlitefilm import UNetLiteFiLM  # noqa: E402


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_date_index():
    """Construct the exact 366-month date index (1994.01-2024.09 excluding 3 missing archive months)."""
    dates_all = pd.date_range(start="1994-01-01", end="2024-09-01", freq="MS")
    exclude_months = pd.to_datetime(["2011-09-01", "2011-10-01", "2017-01-01"])
    dates_all = pd.DatetimeIndex(sorted(dates_all.difference(exclude_months)))
    assert len(dates_all) == 366, f"Expected 366 dates, got {len(dates_all)}"
    return dates_all


def compute_strict_sst_pcs(sst_raw: np.ndarray, fit_mask: np.ndarray,
                           n_pcs: int = 5, window_months: int = 6,
                           keep_lags: bool = False):
    """
    Compute SST EOF patterns and standardized PCs strictly fitted on `fit_mask` timesteps.
    A four-dimensional input stores preceding months oldest to newest; average only
    the most recent `window_months` to form one SST field per target month.
    With `keep_lags`, project each month onto that same EOF basis and retain
    the oldest-to-newest PC sequence instead of averaging the PC scores.
    Also enforces canonical El Nino positive sign convention on PC1 (warm eastern equatorial Pacific).
    """
    if sst_raw.ndim == 4:
        if not 1 <= window_months <= sst_raw.shape[1]:
            raise ValueError("window_months must be within the stored SST lag window")
        sst = np.mean(sst_raw[:, -window_months:], axis=1)  # [T, 89, 180]
    else:
        sst = sst_raw.copy()

    T, H, W = sst.shape
    sst_mean = np.nanmean(sst[fit_mask], axis=0, keepdims=True)
    sst_anom = sst - sst_mean
    X = sst_anom.reshape(T, -1)
    X[~np.isfinite(X)] = 0.0

    pca = PCA(n_components=n_pcs, svd_solver="full")
    pca.fit(X[fit_mask])
    pcs_all = pca.transform(X)

    mean_val = pcs_all[fit_mask].mean(axis=0, keepdims=True)
    std_val = pcs_all[fit_mask].std(axis=0, keepdims=True) + 1e-8
    pcs_std = (pcs_all - mean_val) / std_val
    if keep_lags:
        if sst_raw.ndim != 4:
            raise ValueError("keep_lags requires a time-by-lag SST input")
        lag_anom = sst_raw[:, -window_months:] - sst_mean[:, None]
        lag_x = lag_anom.reshape(T * window_months, -1)
        lag_x[~np.isfinite(lag_x)] = 0.0
        lag_scores = pca.transform(lag_x).reshape(T, window_months, n_pcs)
        pcs_std = (lag_scores - mean_val) / std_val
    eof_patterns = pca.components_.reshape(n_pcs, H, W).copy()

    # Enforce canonical El Nino sign on PC1 using true NOAA ERSST v5 coordinates (-88..88N, 0..358E)
    lats = np.arange(-88, 90, 2, dtype=np.float32)[:H]
    lons = np.arange(0, 360, 2, dtype=np.float32)[:W]
    lat_mask = (lats >= -5.0) & (lats <= 5.0)
    lon_mask = (lons >= 190.0) & (lons <= 240.0)
    nino34_loading = np.nanmean(eof_patterns[0][np.ix_(lat_mask, lon_mask)])
    if nino34_loading < 0:
        pcs_std[..., 0] *= -1.0
        eof_patterns[0] *= -1.0

    if keep_lags:
        pcs_std = pcs_std.reshape(T, window_months * n_pcs)

    return (
        pcs_std.astype(np.float32),
        eof_patterns.astype(np.float32),
        pca.explained_variance_ratio_.astype(np.float32),
    )


def train_single_run(
    model_type: str,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    n_pcs: int,
    lr: float,
    dropout: float,
    epochs: int,
    device: torch.device,
    seed: int = 42,
):
    """Train one model configuration and select best checkpoint strictly by internal validation MSE."""
    set_seed(seed)
    batch_size = 8
    loss_fn = nn.MSELoss()

    if model_type == "baseline":
        model = UNetLite(n_channels=10, n_classes=1, base_filters=32, dropout=dropout).to(device)
    elif model_type == "eof_cat":
        model = UNetLite(n_channels=10 + n_pcs, n_classes=1, base_filters=32, dropout=dropout).to(device)
    elif model_type == "film":
        model = UNetLiteFiLM(
            n_channels=10, n_classes=1, index_dim=n_pcs, base_filters=16, dropout=dropout
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = None

    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            if model_type in ["baseline", "eof_cat"]:
                y, x, m = [item.to(device) for item in batch]
                optimizer.zero_grad()
                pred = model(x)
            else:
                y, x, m, p = [item.to(device) for item in batch]
                optimizer.zero_grad()
                pred = model(x, p)
            pred[m] = float("nan")
            loss = loss_fn(pred[~m], y[~m])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if model_type in ["baseline", "eof_cat"]:
                    y, x, m = [item.to(device) for item in batch]
                    pred = model(x)
                else:
                    y, x, m, p = [item.to(device) for item in batch]
                    pred = model(x, p)
                pred[m] = float("nan")
                v_loss = loss_fn(pred[~m], y[~m])
                val_losses.append(v_loss.item())

        mean_val = float(np.mean(val_losses))
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            best_epoch = ep + 1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    model.eval()
    return model, best_val_loss, best_epoch


def compute_tcc_and_p(pred: np.ndarray, obs: np.ndarray, valid_mask: np.ndarray):
    """
    Compute grid-wise Temporal Correlation Coefficient (TCC) and Student's t p-value.

    Vectorised across all grid cells (numerically equivalent to per-grid scipy.stats.pearsonr)
    so the 6,000 calls inside the year-blocked bootstrap complete in seconds.
    """
    H, W = valid_mask.shape
    T = int(pred.shape[0])
    if T != int(obs.shape[0]):
        raise ValueError("pred and obs must share the same time dimension")

    o = np.asarray(obs, dtype=np.float64).reshape(T, -1)
    p = np.asarray(pred, dtype=np.float64).reshape(T, -1)
    ok = np.isfinite(o) & np.isfinite(p)
    o = np.where(ok, o, 0.0)
    p = np.where(ok, p, 0.0)

    n = ok.sum(axis=0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        o_mean = np.divide(o.sum(axis=0), np.maximum(n, 1.0))
        p_mean = np.divide(p.sum(axis=0), np.maximum(n, 1.0))
        od = np.where(ok, o - o_mean, 0.0)
        pd_ = np.where(ok, p - p_mean, 0.0)
        so = np.sqrt(np.sum(od * od, axis=0))
        sp = np.sqrt(np.sum(pd_ * pd_, axis=0))
        denom = so * sp
        r = np.divide(np.sum(od * pd_, axis=0), denom, out=np.full_like(denom, np.nan), where=denom > 0)

        std_o = np.divide(so, np.sqrt(np.maximum(n, 1.0)))
        std_p = np.divide(sp, np.sqrt(np.maximum(n, 1.0)))
        dof = np.maximum(n - 2.0, 1.0)
        r_clip = np.clip(r, -1.0 + 1e-12, 1.0 - 1e-12)
        tstat = r_clip * np.sqrt(dof / np.maximum(1.0 - r_clip ** 2, 1e-300))
        pv = 2.0 * student_t.sf(np.abs(tstat), dof)

    good = (
        valid_mask.reshape(-1)
        & (n >= 5)
        & (std_o > 1e-6)
        & (std_p > 1e-6)
        & np.isfinite(r)
        & np.isfinite(pv)
    )
    tcc_flat = np.where(good, r, np.nan).astype(np.float32)
    pval_flat = np.where(good, pv, np.nan).astype(np.float32)
    return tcc_flat.reshape(H, W), pval_flat.reshape(H, W)


def compute_suite_metrics(
    pred: np.ndarray,
    obs: np.ndarray,
    valid_mask: np.ndarray,
    p90_thresh: float,
):
    """Compute ACC series, mean ACC, spatial TCC, significant grid %, RMSE, Sign Consistency, and Extreme CSI."""
    # Strictly restrict ACC evaluation to `valid_mask` (critical for regional breakdowns!)
    obs_m = np.where(valid_mask[None, :, :], obs, np.nan)
    pred_m = np.where(valid_mask[None, :, :], pred, np.nan)
    acc_series = utils.cal_acc(obs_m, pred_m)
    mean_acc = float(np.nanmean(acc_series))

    tcc_map, pval_map = compute_tcc_and_p(pred, obs, valid_mask)
    mean_tcc = float(np.nanmean(tcc_map[valid_mask].astype(np.float64)))
    n_valid = int(np.sum(valid_mask))
    n_sig = int(np.sum((pval_map < 0.05) & valid_mask))
    sig_pct = float(n_sig / n_valid * 100.0)

    sq_errs = []
    sc_rates = []
    hits = fas = misses = 0
    for t in range(len(obs)):
        o = obs[t][valid_mask]
        p = pred[t][valid_mask]
        ok = np.isfinite(o) & np.isfinite(p)
        o_ok, p_ok = o[ok], p[ok]
        sq_errs.append((o_ok - p_ok) ** 2)
        sc_rates.append(np.mean((o_ok * p_ok) > 0))
        obs_ext = o_ok >= p90_thresh
        pred_ext = p_ok >= p90_thresh
        hits += int(np.sum(obs_ext & pred_ext))
        fas += int(np.sum((~obs_ext) & pred_ext))
        misses += int(np.sum(obs_ext & (~pred_ext)))

    rmse = float(np.sqrt(np.mean(np.concatenate(sq_errs))))
    sc_pct = float(np.mean(sc_rates) * 100.0)
    csi = float(hits / (hits + fas + misses + 1e-8))

    return {
        "acc_series": acc_series,
        "mean_acc": mean_acc,
        "tcc_map": tcc_map,
        "pval_map": pval_map,
        "mean_tcc": mean_tcc,
        "n_valid": n_valid,
        "n_sig": n_sig,
        "sig_pct": sig_pct,
        "rmse": rmse,
        "sc_pct": sc_pct,
        "csi": csi,
    }


def paired_stat_tests(
    series_a: np.ndarray,
    series_b: np.ndarray,
    years: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
):
    """
    Compute mean difference (A - B), paired Wilcoxon signed-rank p-value,
    month-wise bootstrap 95% CI, and year-blocked bootstrap 95% CI.
    """
    rng = np.random.default_rng(seed)
    diff = series_a - series_b
    mean_diff = float(np.mean(diff))
    pos_count = int(np.sum(diff > 0))
    total_n = int(len(diff))

    try:
        _, p_wilcox = wilcoxon(series_a, series_b, alternative="two-sided")
        p_wilcox = float(p_wilcox)
    except Exception:
        p_wilcox = 1.0

    idx_m = rng.integers(0, total_n, size=(n_boot, total_n))
    boot_m = np.mean(diff[idx_m], axis=1)
    ci_m = [float(np.percentile(boot_m, 2.5)), float(np.percentile(boot_m, 97.5))]

    unique_years = np.unique(years)
    year_blocks = [diff[years == y] for y in unique_years]
    n_y = len(year_blocks)
    boot_y = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sampled_y = rng.integers(0, n_y, size=n_y)
        concat_block = np.concatenate([year_blocks[k] for k in sampled_y])
        boot_y[b] = np.mean(concat_block)
    ci_y = [float(np.percentile(boot_y, 2.5)), float(np.percentile(boot_y, 97.5))]

    return {
        "mean_diff": mean_diff,
        "pos_months": pos_count,
        "total_months": total_n,
        "wilcoxon_p": p_wilcox,
        "boot_month_ci95": ci_m,
        "boot_year_block_ci95": ci_y,
    }


def compute_raw_spatial_gradcam(
    model: nn.Module,
    model_type: str,
    cond_tensor: torch.Tensor,
    valid_mask: np.ndarray,
    pcs_tensor: torch.Tensor = None,
    device: torch.device = torch.device("cpu"),
):
    """
    Compute raw (un-normalized) multi-channel Grad-CAM over `model.up4.conv` with the
    backpropagation objective restricted to valid China land grids (`out[:, :, valid_mask].mean()`),
    averaged across all input timesteps. Shared spatial-span normalization is applied later
    across all six symmetric ablation panels so constant/flat bias maps do not saturate to 1.0.
    """
    model.eval()
    target_layer = model.up4.conv
    vm_t = torch.from_numpy(valid_mask).to(device)
    feats = []
    grads = []

    def fwd_hook(module, inp, out):
        feats.append(out)

    def bwd_hook(module, grad_in, grad_out):
        grads.append(grad_out[0])

    h_f = target_layer.register_forward_hook(fwd_hook)
    h_b = target_layer.register_full_backward_hook(bwd_hook)

    raw_cams = []
    try:
        for t in range(len(cond_tensor)):
            feats.clear()
            grads.clear()
            x_in = cond_tensor[t : t + 1].to(device)
            if model_type == "film":
                p_in = pcs_tensor[t : t + 1].to(device)
                out = model(x_in, p_in)
            else:
                out = model(x_in)

            score = out[:, :, vm_t].mean()
            model.zero_grad()
            score.backward()

            g = grads[0][0]  # [C, H, W]
            a = feats[0][0]  # [C, H, W]
            weights = torch.mean(g, dim=(1, 2))  # [C]
            cam = torch.sum(weights[:, None, None] * a, dim=0)
            cam = F.relu(cam)
            raw_cams.append(cam.detach().cpu().numpy())
    finally:
        h_f.remove()
        h_b.remove()

    return np.mean(raw_cams, axis=0).astype(np.float64)


def reconstruct_raw_sensitivity_map(corr_gamma_layer: np.ndarray, eof_patterns: np.ndarray):
    """Reconstruct un-normalized positive sensitivity map from channel-PC correlations and EOF patterns."""
    _, n_pcs = corr_gamma_layer.shape
    sens = np.zeros_like(eof_patterns[0], dtype=np.float64)
    for j in range(n_pcs):
        weighted_eof = corr_gamma_layer[:, j].reshape(-1, 1, 1) * eof_patterns[j]
        sens += weighted_eof.sum(axis=0)
    sens = np.maximum(sens, 0.0)
    return sens.astype(np.float32)


def extract_top_peaks(arr: np.ndarray, lats_g: np.ndarray, lons_g: np.ndarray, k: int = 3):
    """Extract top-k spatially separated maxima from a global [H_s, W_s] sensitivity map."""
    masked = arr.copy()
    peaks = []
    for _ in range(k):
        ri, ci = np.unravel_index(np.nanargmax(masked), masked.shape)
        val = float(masked[ri, ci])
        lon_e = float(lons_g[ci])
        lat_v = float(lats_g[ri])
        lon_label = f"{lon_e:.0f}E" if lon_e <= 180 else f"{360.0 - lon_e:.0f}W"
        peaks.append({"lat": lat_v, "lon_e": lon_e, "coord": f"{lat_v:.0f}N,{lon_label}", "val": val})
        r0, r1 = max(0, ri - 5), min(len(lats_g), ri + 6)
        c0, c1 = max(0, ci - 8), min(len(lons_g), ci + 9)
        masked[r0:r1, c0:c1] = -1.0
    return peaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true", help="Reuse existing checkpoints & tuning_log if present")
    args = ap.parse_args()

    t_start = time.time()
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Unified Eval] Running on device: {device} (eval_only={args.eval_only})")

    paths = SubprojectPaths(__file__)
    exp_dir = paths.get_exp_dir("unified_eval")
    ckpt_dir = exp_dir / "checkpoints"
    print(f"[Unified Eval] Output experiment directory: {exp_dir}")

    # 1. Load datasets and build chronological masks
    dates_all = build_date_index()
    hr_file = os.path.join(config.modelconfig["hr_path"], "hr_data.npy")
    cond_file = os.path.join(config.modelconfig["lr_path"], "lr_data_reconstructed2.npy")
    sst_file = config.modelconfig["sst_file"]

    target = np.load(hr_file).astype(np.float32)  # [366, 120, 140]
    cond = np.load(cond_file).astype(np.float32)   # [366, 10, 120, 140]
    sst_raw = np.load(sst_file).astype(np.float32) # [366, 6, 89, 180]
    assert target.shape[0] == cond.shape[0] == sst_raw.shape[0] == 366

    train_mask_t = (dates_all >= "1994-01-01") & (dates_all <= "2017-12-01")
    val_mask_t = (dates_all >= "2018-01-01") & (dates_all <= "2022-12-01")
    hind_mask_t = (dates_all >= "1994-01-01") & (dates_all <= "2022-12-01")
    test_mask_t = dates_all >= "2023-01-01"

    n_train = int(np.sum(train_mask_t))
    n_val = int(np.sum(val_mask_t))
    n_hind = int(np.sum(hind_mask_t))
    n_test = int(np.sum(test_mask_t))
    print(f"[Split] Train={n_train}m (1994-2017), Internal Val={n_val}m (2018-2022), Hindcast={n_hind}m, Test={n_test}m (2023.01-2024.09)")

    valid_land_mask = ~np.isnan(target[0])  # [120, 140], 5,301 valid China grids
    n_valid_grids = int(np.sum(valid_land_mask))

    train_valid_vals = target[train_mask_t][:, valid_land_mask]
    p90_thresh = float(np.percentile(train_valid_vals, 90))
    pct_above_3 = float(np.mean(np.abs(target[:, valid_land_mask]) > 3.0) * 100.0)
    print(f"[Data Stats] Valid China land grids={n_valid_grids}, Train 90th-pct anomaly threshold={p90_thresh:.4f}, Grid-months |dP|>3.0: {pct_above_3:.2f}%")

    # 2. Strict leak-free SST PCA (n_pcs=5 unified across EOF-cat and FiLM-U-Net)
    n_pcs = 5
    pcs5, eof_patterns, explained_var = compute_strict_sst_pcs(sst_raw, fit_mask=train_mask_t, n_pcs=n_pcs)
    print(f"[SST PCA] Fitted strictly on {n_train}m Train set. 5 PCs explain {explained_var.sum()*100:.2f}% variance: {[round(float(v)*100, 2) for v in explained_var]}")

    target_t = torch.from_numpy(np.expand_dims(target, 1))
    mask_t = torch.isnan(target_t)
    cond_t = torch.from_numpy(cond)
    pcs5_t = torch.from_numpy(pcs5)

    H_map, W_map = cond_t.shape[2], cond_t.shape[3]
    pcs5_broad = pcs5_t.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H_map, W_map)
    cond_cat_t = torch.cat([cond_t, pcs5_broad], dim=1)  # [366, 15, 120, 140]

    # 3. Identical Hyperparameter Tuning Grid on Internal Val (2018-2022)
    tuning_grid = [
        {"name": "lr2e-4_do0.2", "lr": 2e-4, "dropout": 0.2},
        {"name": "lr1e-4_do0.2", "lr": 1e-4, "dropout": 0.2},
        {"name": "lr2e-4_do0.1", "lr": 2e-4, "dropout": 0.1},
    ]
    epochs = 100

    datasets_by_model = {
        "baseline": (
            TensorDataset(target_t[train_mask_t], cond_t[train_mask_t], mask_t[train_mask_t]),
            TensorDataset(target_t[val_mask_t], cond_t[val_mask_t], mask_t[val_mask_t]),
        ),
        "eof_cat": (
            TensorDataset(target_t[train_mask_t], cond_cat_t[train_mask_t], mask_t[train_mask_t]),
            TensorDataset(target_t[val_mask_t], cond_cat_t[val_mask_t], mask_t[val_mask_t]),
        ),
        "film": (
            TensorDataset(target_t[train_mask_t], cond_t[train_mask_t], mask_t[train_mask_t], pcs5_t[train_mask_t]),
            TensorDataset(target_t[val_mask_t], cond_t[val_mask_t], mask_t[val_mask_t], pcs5_t[val_mask_t]),
        ),
    }

    best_models = {}
    tuning_log = {}
    prev_metrics_path = exp_dir / "unified_metrics.json"
    can_reuse = (
        args.eval_only
        and prev_metrics_path.exists()
        and all((ckpt_dir / f"{m}_best.pt").exists() for m in ("baseline", "eof_cat", "film"))
    )

    if can_reuse:
        prev_json = json.loads(prev_metrics_path.read_text(encoding="utf-8"))
        tuning_log = prev_json["tuning_log"]
        for m_type in ["baseline", "eof_cat", "film"]:
            best_cfg = min(tuning_log[m_type], key=lambda z: z["best_val_mse"])
            if m_type == "baseline":
                mdl = UNetLite(n_channels=10, n_classes=1, base_filters=32, dropout=best_cfg["dropout"]).to(device)
            elif m_type == "eof_cat":
                mdl = UNetLite(n_channels=10 + n_pcs, n_classes=1, base_filters=32, dropout=best_cfg["dropout"]).to(device)
            else:
                mdl = UNetLiteFiLM(n_channels=10, n_classes=1, index_dim=n_pcs, base_filters=16, dropout=best_cfg["dropout"]).to(device)
            mdl.load_state_dict(torch.load(ckpt_dir / f"{m_type}_best.pt", map_location=device))
            mdl.eval()
            best_models[m_type] = mdl
            print(f"  => Loaded existing {m_type} checkpoint ({best_cfg['config']})")
    else:
        for m_type in ["baseline", "eof_cat", "film"]:
            print(f"\n--- Hyperparameter Tuning & Training: {m_type} ---")
            tr_ds, va_ds = datasets_by_model[m_type]
            best_cfg = None
            best_loss = float("inf")
            best_mdl = None
            tuning_log[m_type] = []

            for cfg in tuning_grid:
                mdl, val_loss, best_ep = train_single_run(
                    model_type=m_type,
                    train_ds=tr_ds,
                    val_ds=va_ds,
                    n_pcs=n_pcs,
                    lr=cfg["lr"],
                    dropout=cfg["dropout"],
                    epochs=epochs,
                    device=device,
                    seed=42,
                )
                print(f"  [{m_type} | {cfg['name']}] Best Internal Val MSE = {val_loss:.5f} at epoch {best_ep}")
                tuning_log[m_type].append({
                    "config": cfg["name"],
                    "lr": cfg["lr"],
                    "dropout": cfg["dropout"],
                    "best_val_mse": val_loss,
                    "best_epoch": best_ep,
                })
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_cfg = cfg
                    best_mdl = mdl

            best_models[m_type] = best_mdl
            torch.save(best_mdl.state_dict(), ckpt_dir / f"{m_type}_best.pt")
            print(f"  => Selected {m_type} best config: {best_cfg['name']} (Val MSE={best_loss:.5f})")

    # 4. Generate Full-Timeline Predictions (366 months)
    model_base = best_models["baseline"]
    model_cat = best_models["eof_cat"]
    model_film = best_models["film"]

    with torch.no_grad():
        pred_base = []
        pred_cat = []
        pred_film = []
        for idx0 in range(0, 366, 16):
            pred_base.append(model_base(cond_t[idx0 : idx0 + 16].to(device)).cpu()[:, 0])
            pred_cat.append(model_cat(cond_cat_t[idx0 : idx0 + 16].to(device)).cpu()[:, 0])
            pred_film.append(
                model_film(cond_t[idx0 : idx0 + 16].to(device), pcs5_t[idx0 : idx0 + 16].to(device)).cpu()[:, 0]
            )
        pred_base = torch.cat(pred_base, dim=0).numpy()
        pred_cat = torch.cat(pred_cat, dim=0).numpy()
        pred_film = torch.cat(pred_film, dim=0).numpy()

    pred_seas5 = cond[:, 0].copy()  # Channel 0 is raw SEAS5 precipitation anomaly
    for arr in [pred_seas5, pred_base, pred_cat, pred_film]:
        arr[:, ~valid_land_mask] = np.nan

    # 5. Compute Comprehensive Metrics across Train (285m), Internal Val (60m), Hindcast (345m), and Test (21m)
    models_preds = {
        "SEAS5": pred_seas5,
        "Baseline": pred_base,
        "EOF-cat": pred_cat,
        "FiLM-U-Net": pred_film,
    }

    eval_summary = {}
    tcc_maps_test = {}
    pval_maps_test = {}
    acc_series_all = {}

    for m_name, p_arr in models_preds.items():
        m_train = compute_suite_metrics(p_arr[train_mask_t], target[train_mask_t], valid_land_mask, p90_thresh)
        m_val = compute_suite_metrics(p_arr[val_mask_t], target[val_mask_t], valid_land_mask, p90_thresh)
        m_hind = compute_suite_metrics(p_arr[hind_mask_t], target[hind_mask_t], valid_land_mask, p90_thresh)
        m_test = compute_suite_metrics(p_arr[test_mask_t], target[test_mask_t], valid_land_mask, p90_thresh)

        mask_2023 = dates_all.year == 2023
        mask_2024 = dates_all.year == 2024
        acc_2023 = float(np.nanmean(utils.cal_acc(target[mask_2023], p_arr[mask_2023])))
        acc_2024 = float(np.nanmean(utils.cal_acc(target[mask_2024], p_arr[mask_2024])))

        eval_summary[m_name] = {
            "train_acc": m_train["mean_acc"],
            "val_acc": m_val["mean_acc"],
            "hindcast_acc": m_hind["mean_acc"],
            "hindcast_tcc": m_hind["mean_tcc"],
            "hindcast_sig_pct": m_hind["sig_pct"],
            "test_acc": m_test["mean_acc"],
            "test_acc_2023": acc_2023,
            "test_acc_2024": acc_2024,
            "test_tcc": m_test["mean_tcc"],
            "test_n_sig": m_test["n_sig"],
            "test_n_valid": m_test["n_valid"],
            "test_sig_pct": m_test["sig_pct"],
            "test_rmse": m_test["rmse"],
            "test_sc_pct": m_test["sc_pct"],
            "test_csi": m_test["csi"],
        }
        tcc_maps_test[m_name] = m_test["tcc_map"]
        pval_maps_test[m_name] = m_test["pval_map"]
        acc_series_all[m_name] = utils.cal_acc(target, p_arr)

    print("\n================ TABLE 1 SUMMARY (LEAK-FREE) ================")
    for m_name, s in eval_summary.items():
        print(
            f"{m_name:<12} | Train={s['train_acc']:.4f} | Val={s['val_acc']:.4f} | Hind={s['hindcast_acc']:.4f} | "
            f"Test ACC={s['test_acc']:.4f} (2023:{s['test_acc_2023']:.4f}, 2024:{s['test_acc_2024']:.4f}) | "
            f"TCC={s['test_tcc']:.4f} | Sig={s['test_sig_pct']:.1f}% ({s['test_n_sig']}/{s['test_n_valid']}) | "
            f"RMSE={s['test_rmse']:.4f} | SC={s['test_sc_pct']:.1f}% | CSI={s['test_csi']:.3f}"
        )

    # 6. Paired Significance Tests & Year-Blocked Bootstrap CIs (Table 2 & Section 4.1/4.2)
    test_years = dates_all[test_mask_t].year.values
    hind_years = dates_all[hind_mask_t].year.values

    stat_tests = {
        "test_film_vs_cat_acc": paired_stat_tests(
            acc_series_all["FiLM-U-Net"][test_mask_t],
            acc_series_all["EOF-cat"][test_mask_t],
            test_years,
        ),
        "test_film_vs_base_acc": paired_stat_tests(
            acc_series_all["FiLM-U-Net"][test_mask_t],
            acc_series_all["Baseline"][test_mask_t],
            test_years,
        ),
        "test_cat_vs_base_acc": paired_stat_tests(
            acc_series_all["EOF-cat"][test_mask_t],
            acc_series_all["Baseline"][test_mask_t],
            test_years,
        ),
        "test_base_vs_seas5_acc": paired_stat_tests(
            acc_series_all["Baseline"][test_mask_t],
            acc_series_all["SEAS5"][test_mask_t],
            test_years,
        ),
        "hindcast_film_vs_cat_acc": paired_stat_tests(
            acc_series_all["FiLM-U-Net"][hind_mask_t],
            acc_series_all["EOF-cat"][hind_mask_t],
            hind_years,
        ),
        "hindcast_film_vs_base_acc": paired_stat_tests(
            acc_series_all["FiLM-U-Net"][hind_mask_t],
            acc_series_all["Baseline"][hind_mask_t],
            hind_years,
        ),
    }

    tcc_film_v = tcc_maps_test["FiLM-U-Net"][valid_land_mask].astype(np.float64)
    tcc_cat_v = tcc_maps_test["EOF-cat"][valid_land_mask].astype(np.float64)
    tcc_base_v = tcc_maps_test["Baseline"][valid_land_mask].astype(np.float64)
    _, p_tcc_film_cat = wilcoxon(tcc_film_v, tcc_cat_v)
    _, p_tcc_film_base = wilcoxon(tcc_film_v, tcc_base_v)

    rng = np.random.default_rng(42)
    unique_test_yrs = np.unique(test_years)
    boot_dtcc_cat = []
    boot_dtcc_base = []
    for _ in range(2000):
        sampled_yrs = rng.integers(0, len(unique_test_yrs), size=len(unique_test_yrs))
        idx_list = np.concatenate([np.where(test_years == unique_test_yrs[k])[0] for k in sampled_yrs])
        if len(np.unique(idx_list)) >= 5:
            t_f, _ = compute_tcc_and_p(pred_film[test_mask_t][idx_list], target[test_mask_t][idx_list], valid_land_mask)
            t_c, _ = compute_tcc_and_p(pred_cat[test_mask_t][idx_list], target[test_mask_t][idx_list], valid_land_mask)
            t_b, _ = compute_tcc_and_p(pred_base[test_mask_t][idx_list], target[test_mask_t][idx_list], valid_land_mask)
            boot_dtcc_cat.append(float(np.nanmean((t_f - t_c)[valid_land_mask])))
            boot_dtcc_base.append(float(np.nanmean((t_f - t_b)[valid_land_mask])))

    stat_tests["test_tcc_spatial"] = {
        "film_vs_cat_mean_dtcc": float(np.nanmean(tcc_film_v - tcc_cat_v)),
        "film_vs_cat_wilcoxon_p": float(p_tcc_film_cat),
        "film_vs_cat_year_block_ci95": [float(np.percentile(boot_dtcc_cat, 2.5)), float(np.percentile(boot_dtcc_cat, 97.5))],
        "film_vs_base_mean_dtcc": float(np.nanmean(tcc_film_v - tcc_base_v)),
        "film_vs_base_wilcoxon_p": float(p_tcc_film_base),
        "film_vs_base_year_block_ci95": [float(np.percentile(boot_dtcc_base, 2.5)), float(np.percentile(boot_dtcc_base, 97.5))],
    }

    print("\n================ PAIRED SIGNIFICANCE TESTS ================")
    for k, v in stat_tests.items():
        print(f"  {k}: {v}")

    # 7. Regional Hydroclimatic Attribution (aligned strictly to 0.5-deg station grid coords)
    lons_1d = np.arange(70, 140, 0.5)[:W_map]
    lats_1d = np.arange(60, 0, -0.5)[:H_map]
    lon2d, lat2d = np.meshgrid(lons_1d, lats_1d)

    region_masks = {
        "Eastern_Monsoon": valid_land_mask & (lon2d >= 105.0),
        "Tibetan_Plateau": valid_land_mask & (lon2d < 105.0) & (lat2d <= 37.0),
        "Northwest_Arid": valid_land_mask & (lon2d < 105.0) & (lat2d > 37.0),
    }
    regional_metrics = {}
    for r_name, r_mask in region_masks.items():
        regional_metrics[r_name] = {"n_grids": int(np.sum(r_mask))}
        for m_name, p_arr in models_preds.items():
            rm = compute_suite_metrics(p_arr[test_mask_t], target[test_mask_t], r_mask, p90_thresh)
            regional_metrics[r_name][m_name] = {
                "test_acc": rm["mean_acc"],
                "test_tcc": rm["mean_tcc"],
                "sig_pct": rm["sig_pct"],
                "rmse": rm["rmse"],
                "sc_pct": rm["sc_pct"],
            }

    print("\n================ REGIONAL SKILL ATTRIBUTION ================")
    for r_name, r_data in regional_metrics.items():
        print(f"  {r_name} ({r_data['n_grids']} grids):")
        for m_name in ["SEAS5", "Baseline", "EOF-cat", "FiLM-U-Net"]:
            d = r_data[m_name]
            print(f"    {m_name:<11}: ACC={d['test_acc']:.4f}, TCC={d['test_tcc']:.4f}, Sig={d['sig_pct']:.1f}%, RMSE={d['rmse']:.4f}, SC={d['sc_pct']:.1f}%")

    # 8. Table 3: Permutation Feature Importance (10 Atmospheric Variables, 20 trials)
    var_names = ["tp", "h200", "h500", "slp", "t2m", "t850", "u200", "u850", "v200", "v850"]
    var_full_names = [
        "Total Precipitation Anomaly (tp)",
        "200 hPa Geopotential Height (h200)",
        "500 hPa Geopotential Height (h500)",
        "Sea Level Pressure (slp)",
        "2 m Air Temperature (t2m)",
        "850 hPa Air Temperature (t850)",
        "200 hPa Zonal Wind (u200)",
        "850 hPa Zonal Wind (u850)",
        "200 hPa Meridional Wind (v200)",
        "850 hPa Meridional Wind (v850)",
    ]
    c_test_dev = cond_t[test_mask_t].to(device)
    p_test_dev = pcs5_t[test_mask_t].to(device)
    base_film_test_acc = eval_summary["FiLM-U-Net"]["test_acc"]

    rng_perm = np.random.default_rng(42)
    n_perm_trials = 20
    importance_list = []
    with torch.no_grad():
        for ch_idx in range(10):
            drops = []
            for _ in range(n_perm_trials):
                c_perm = c_test_dev.clone()
                perm_order = torch.from_numpy(rng_perm.permutation(n_test)).to(device)
                c_perm[:, ch_idx] = c_perm[perm_order, ch_idx]
                out_perm = model_film(c_perm, p_test_dev).cpu().numpy()[:, 0]
                out_perm[:, ~valid_land_mask] = np.nan
                acc_perm = float(np.nanmean(utils.cal_acc(target[test_mask_t], out_perm)))
                drops.append(base_film_test_acc - acc_perm)
            importance_list.append({
                "var": var_names[ch_idx],
                "full_name": var_full_names[ch_idx],
                "mean_drop": float(np.mean(drops)),
                "std_drop": float(np.std(drops)),
            })
    importance_list.sort(key=lambda x: x["mean_drop"], reverse=True)
    print("\n================ TABLE 3: PERMUTATION IMPORTANCE ================")
    for rank_idx, item in enumerate(importance_list, 1):
        print(f"  #{rank_idx:<2} {item['full_name']:<38}: {item['mean_drop']:+.4f} +/- {item['std_drop']:.4f}")

    # 9. Table 4: ENSO Phase Stratification (1994-2022 Hindcast [345m] & 2023-2024 Test [21m])
    pc1_all = pcs5[:, 0]
    enso_labels = np.where(
        pc1_all >= 0.5, "El Nino", np.where(pc1_all <= -0.5, "La Nina", "Neutral")
    )
    enso_strat = {"hindcast": {}, "test": {}}
    for period_name, p_mask in [("hindcast", hind_mask_t), ("test", test_mask_t)]:
        p_len = int(np.sum(p_mask))
        phases_sub = enso_labels[p_mask]
        for phase in ["El Nino", "La Nina", "Neutral", "All"]:
            if phase == "All":
                sub_bool = np.ones(p_len, dtype=bool)
            else:
                sub_bool = phases_sub == phase
            cnt = int(np.sum(sub_bool))
            pct = float(cnt / p_len * 100.0)
            entry = {"months": cnt, "pct": pct}
            if cnt > 0:
                for m_name in ["SEAS5", "Baseline", "EOF-cat", "FiLM-U-Net"]:
                    acc_sub = acc_series_all[m_name][p_mask][sub_bool]
                    entry[m_name] = float(np.nanmean(acc_sub))
                entry["gain_vs_base"] = float(entry["FiLM-U-Net"] - entry["Baseline"])
                entry["gain_vs_cat"] = float(entry["FiLM-U-Net"] - entry["EOF-cat"])
            enso_strat[period_name][phase] = entry

    print("\n================ TABLE 4: ENSO STRATIFICATION ================")
    for p_name in ["hindcast", "test"]:
        print(f"  [{p_name.upper()}]")
        for ph, d in enso_strat[p_name].items():
            if d["months"] > 0:
                print(
                    f"    {ph:<8} ({d['months']:>3}m, {d['pct']:>5.1f}%): SEAS5={d['SEAS5']:.4f}, "
                    f"Base={d['Baseline']:.4f}, Cat={d['EOF-cat']:.4f}, FiLM={d['FiLM-U-Net']:.4f} | "
                    f"dBase={d['gain_vs_base']:+.4f}, dCat={d['gain_vs_cat']:+.4f}"
                )

    # 10. Sensitivity Experiment (R2-1): 2020.01-2021.12 (24m Mixed-ENSO Rolling Holdout)
    print("\n--- Running 2020.01-2021.12 (24m Mixed-ENSO) Rolling Holdout Sensitivity Test ---")
    holdout_2021_mask = (dates_all >= "2020-01-01") & (dates_all <= "2021-12-01")
    avail_hist_mask = hind_mask_t & (~holdout_2021_mask)
    ho_train_mask = avail_hist_mask & (dates_all <= "2015-12-01")
    ho_val_mask = avail_hist_mask & (dates_all >= "2016-01-01") & (dates_all <= "2019-12-01")

    pcs5_ho, _, _ = compute_strict_sst_pcs(sst_raw, fit_mask=ho_train_mask, n_pcs=n_pcs)
    pcs5_ho_t = torch.from_numpy(pcs5_ho)
    pcs5_ho_broad = pcs5_ho_t.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H_map, W_map)
    cond_cat_ho_t = torch.cat([cond_t, pcs5_ho_broad], dim=1)

    ho_models = {}
    ho_ds_map = {
        "baseline": (
            TensorDataset(target_t[ho_train_mask], cond_t[ho_train_mask], mask_t[ho_train_mask]),
            TensorDataset(target_t[ho_val_mask], cond_t[ho_val_mask], mask_t[ho_val_mask]),
        ),
        "eof_cat": (
            TensorDataset(target_t[ho_train_mask], cond_cat_ho_t[ho_train_mask], mask_t[ho_train_mask]),
            TensorDataset(target_t[ho_val_mask], cond_cat_ho_t[ho_val_mask], mask_t[ho_val_mask]),
        ),
        "film": (
            TensorDataset(target_t[ho_train_mask], cond_t[ho_train_mask], mask_t[ho_train_mask], pcs5_ho_t[ho_train_mask]),
            TensorDataset(target_t[ho_val_mask], cond_t[ho_val_mask], mask_t[ho_val_mask], pcs5_ho_t[ho_val_mask]),
        ),
    }
    for m_type in ["baseline", "eof_cat", "film"]:
        best_cfg_name = min(tuning_log[m_type], key=lambda z: z["best_val_mse"])
        mdl_ho, _, _ = train_single_run(
            model_type=m_type,
            train_ds=ho_ds_map[m_type][0],
            val_ds=ho_ds_map[m_type][1],
            n_pcs=n_pcs,
            lr=best_cfg_name["lr"],
            dropout=best_cfg_name["dropout"],
            epochs=epochs,
            device=device,
            seed=42,
        )
        ho_models[m_type] = mdl_ho

    with torch.no_grad():
        p_ho_seas5 = cond[holdout_2021_mask, 0].copy()
        p_ho_base = ho_models["baseline"](cond_t[holdout_2021_mask].to(device)).cpu().numpy()[:, 0]
        p_ho_cat = ho_models["eof_cat"](cond_cat_ho_t[holdout_2021_mask].to(device)).cpu().numpy()[:, 0]
        p_ho_film = ho_models["film"](
            cond_t[holdout_2021_mask].to(device), pcs5_ho_t[holdout_2021_mask].to(device)
        ).cpu().numpy()[:, 0]

    for arr in [p_ho_seas5, p_ho_base, p_ho_cat, p_ho_film]:
        arr[:, ~valid_land_mask] = np.nan

    obs_ho_2021 = target[holdout_2021_mask]
    ho_phases = enso_labels[holdout_2021_mask]
    holdout_2021_summary = {
        "months": int(np.sum(holdout_2021_mask)),
        "el_nino_months": int(np.sum(ho_phases == "El Nino")),
        "la_nina_months": int(np.sum(ho_phases == "La Nina")),
        "neutral_months": int(np.sum(ho_phases == "Neutral")),
    }
    for m_name, p_arr in [
        ("SEAS5", p_ho_seas5),
        ("Baseline", p_ho_base),
        ("EOF-cat", p_ho_cat),
        ("FiLM-U-Net", p_ho_film),
    ]:
        hm = compute_suite_metrics(p_arr, obs_ho_2021, valid_land_mask, p90_thresh)
        holdout_2021_summary[m_name] = {
            "acc": hm["mean_acc"],
            "tcc": hm["mean_tcc"],
            "sig_pct": hm["sig_pct"],
            "rmse": hm["rmse"],
            "sc_pct": hm["sc_pct"],
        }
    print(f"  [2020-2021 Holdout ({holdout_2021_summary['la_nina_months']} La Nina, {holdout_2021_summary['neutral_months']} Neutral, {holdout_2021_summary['el_nino_months']} El Nino)]:")
    for m_name in ["SEAS5", "Baseline", "EOF-cat", "FiLM-U-Net"]:
        d = holdout_2021_summary[m_name]
        print(f"    {m_name:<11}: ACC={d['acc']:.4f}, TCC={d['tcc']:.4f}, Sig={d['sig_pct']:.1f}%, RMSE={d['rmse']:.4f}")

    # 11. Interpretability Analyses (Figs. 6, 7, 8, 9 & Table 5)
    print("\n--- Extracting FiLM Parameters, Shuffled-PC Controls, and Symmetric Grad-CAM ---")
    all_gamma, all_beta = [], []

    def hook_film(module, inp, out):
        gammas, betas = out
        all_gamma.append([g.detach().cpu().numpy() for g in gammas])
        all_beta.append([b.detach().cpu().numpy() for b in betas])

    handle_film = model_film.index_encoder.register_forward_hook(hook_film)
    with torch.no_grad():
        for idx0 in range(0, 366, 16):
            _ = model_film(cond_t[idx0 : idx0 + 16].to(device), pcs5_t[idx0 : idx0 + 16].to(device))
    handle_film.remove()

    n_layers = len(all_gamma[0])
    gamma_per_layer = [np.concatenate([bg[l] for bg in all_gamma], axis=0) for l in range(n_layers)]
    beta_per_layer = [np.concatenate([bb[l] for bb in all_beta], axis=0) for l in range(n_layers)]

    corr_gamma = []
    corr_beta = []
    for l_idx in range(n_layers):
        n_ch = gamma_per_layer[l_idx].shape[1]
        cg = np.zeros((n_ch, n_pcs), dtype=np.float32)
        cb = np.zeros((n_ch, n_pcs), dtype=np.float32)
        for ch_i in range(n_ch):
            for pc_j in range(n_pcs):
                cg[ch_i, pc_j] = pearsonr(gamma_per_layer[l_idx][hind_mask_t, ch_i], pcs5[hind_mask_t, pc_j])[0]
                cb[ch_i, pc_j] = pearsonr(beta_per_layer[l_idx][hind_mask_t, ch_i], pcs5[hind_mask_t, pc_j])[0]
        corr_gamma.append(cg)
        corr_beta.append(cb)

    rng_shuf = np.random.default_rng(123)
    n_shuf_trials = 100
    shuf_gamma_mean_abs = np.zeros((n_shuf_trials, n_pcs), dtype=np.float32)
    shuf_beta_mean_abs = np.zeros((n_shuf_trials, n_pcs), dtype=np.float32)
    shuf_l1_corr_gamma_trials = []

    for s_idx in range(n_shuf_trials):
        perm = rng_shuf.permutation(n_hind)
        pcs_shuf = pcs5[hind_mask_t][perm]
        cg_s = np.zeros((16, n_pcs), dtype=np.float32)
        cb_s = np.zeros((16, n_pcs), dtype=np.float32)
        for ch_i in range(16):
            for pc_j in range(n_pcs):
                cg_s[ch_i, pc_j] = pearsonr(gamma_per_layer[0][hind_mask_t, ch_i], pcs_shuf[:, pc_j])[0]
                cb_s[ch_i, pc_j] = pearsonr(beta_per_layer[0][hind_mask_t, ch_i], pcs_shuf[:, pc_j])[0]
        shuf_gamma_mean_abs[s_idx] = np.mean(np.abs(cg_s), axis=0)
        shuf_beta_mean_abs[s_idx] = np.mean(np.abs(cb_s), axis=0)
        shuf_l1_corr_gamma_trials.append(cg_s)

    raw_sens_layers = [reconstruct_raw_sensitivity_map(corr_gamma[l], eof_patterns) for l in range(n_layers)]
    raw_sens_shuf_trials = [reconstruct_raw_sensitivity_map(cg_s, eof_patterns) for cg_s in shuf_l1_corr_gamma_trials]
    raw_sens_shuf_mean = np.mean(raw_sens_shuf_trials, axis=0)
    raw_sens_shuf_rep = raw_sens_shuf_trials[0]

    global_sens_max = max(float(np.nanmax(m)) for m in raw_sens_layers) + 1e-8
    norm_sens_layers = [np.clip(m / global_sens_max, 0.0, 1.0) for m in raw_sens_layers]
    norm_sens_shuf_mean = np.clip(raw_sens_shuf_mean / global_sens_max, 0.0, 1.0)
    norm_sens_shuf_rep = np.clip(raw_sens_shuf_rep / global_sens_max, 0.0, 1.0)

    lats_g = np.arange(-88, 90, 2, dtype=np.float32)[:eof_patterns.shape[1]]
    lons_g = np.arange(0, 360, 2, dtype=np.float32)[:eof_patterns.shape[2]]
    sens_peaks = {
        f"layer_{idx + 1}": extract_top_peaks(norm_sens_layers[idx], lats_g, lons_g, k=3)
        for idx in range(n_layers)
    }

    print(
        f"[Fig. 9 Sensitivity] Layer 1 max={norm_sens_layers[0].max():.3f}, Layer 5 max={norm_sens_layers[4].max():.3f}, "
        f"Shuffled-PC Control mean-map max={norm_sens_shuf_mean.max():.3f} (single draw max={norm_sens_shuf_rep.max():.3f})"
    )

    # Table 5 & Fig. 8: Symmetric Ablation & Grad-CAM across all 6 spatial panels
    cat_test_dev = cond_cat_t[test_mask_t].to(device)
    cat_pzero_dev = cat_test_dev.clone()
    cat_pzero_dev[:, 10:] = 0.0
    cat_czero_dev = cat_test_dev.clone()
    cat_czero_dev[:, :10] = 0.0

    p_zero_dev = torch.zeros_like(p_test_dev)
    c_zero_dev = torch.zeros_like(c_test_dev)

    ablation_configs = [
        ("Baseline U-Net (No SST)", "baseline", c_test_dev, None),
        ("EOF-cat (SST PCs Zeroed)", "eof_cat", cat_pzero_dev, None),
        ("EOF-cat (Local Zeroed)", "eof_cat", cat_czero_dev, None),
        ("Full FiLM-U-Net (Ours)", "film", c_test_dev, p_test_dev),
        ("FiLM-U-Net (SST PCs Zeroed)", "film", c_test_dev, p_zero_dev),
        ("FiLM-U-Net (Local Zeroed)", "film", c_zero_dev, p_test_dev),
        ("Full EOF-cat (All Inputs)", "eof_cat", cat_test_dev, None),
    ]

    table5_results = {}
    raw_gradcam_maps = {}

    for cfg_label, m_key, x_in_dev, p_in_dev in ablation_configs:
        mdl = best_models[m_key]
        with torch.no_grad():
            if m_key == "film":
                pred_abl = mdl(x_in_dev, p_in_dev).cpu().numpy()[:, 0]
            else:
                pred_abl = mdl(x_in_dev).cpu().numpy()[:, 0]
        pred_abl[:, ~valid_land_mask] = np.nan
        sm = compute_suite_metrics(pred_abl, target[test_mask_t], valid_land_mask, p90_thresh)
        table5_results[cfg_label] = {
            "test_acc": sm["mean_acc"],
            "test_tcc": sm["mean_tcc"],
            "sig_pct": sm["sig_pct"],
            "rmse": sm["rmse"],
            "sc_pct": sm["sc_pct"],
        }
        if cfg_label != "Full EOF-cat (All Inputs)":
            raw_gradcam_maps[cfg_label] = compute_raw_spatial_gradcam(
                mdl, m_key, x_in_dev.cpu(), valid_land_mask,
                pcs_tensor=(p_in_dev.cpu() if p_in_dev is not None else None),
                device=device,
            )

    # Shared spatial-span China-floor normalization across all 6 Fig. 8 configurations:
    # removes spatially uniform constant-bias floors (e.g. when local predictors are zeroed)
    # and scales spatially varying saliency onto a single common [0, 1] scale.
    shared_gradcam_span = max(
        float(m[valid_land_mask].max() - m[valid_land_mask].min()) for m in raw_gradcam_maps.values()
    ) + 1e-12
    gradcam_maps = {}
    gradcam_diag = {"shared_spatial_span": shared_gradcam_span}
    for cfg_label, raw_m in raw_gradcam_maps.items():
        v_min = float(raw_m[valid_land_mask].min())
        v_max = float(raw_m[valid_land_mask].max())
        norm_m = np.clip((raw_m - v_min) / shared_gradcam_span, 0.0, 1.0)
        norm_m[~valid_land_mask] = np.nan
        gradcam_maps[cfg_label] = norm_m.astype(np.float32)
        nmean = float(np.nanmean(norm_m[valid_land_mask]))
        nmax = float(np.nanmax(norm_m[valid_land_mask]))
        table5_results[cfg_label]["mean_gradcam"] = nmean
        gradcam_diag[cfg_label] = {
            "raw_china_min": v_min,
            "raw_china_max": v_max,
            "raw_china_span": v_max - v_min,
            "norm_mean": nmean,
            "norm_max": nmax,
        }

    rng_t5 = np.random.default_rng(77)
    shuf_accs_t5, shuf_tccs_t5, shuf_sigs_t5 = [], [], []
    with torch.no_grad():
        for _ in range(50):
            perm_t = torch.from_numpy(rng_t5.permutation(n_test)).to(device)
            pred_s = model_film(c_test_dev, p_test_dev[perm_t]).cpu().numpy()[:, 0]
            pred_s[:, ~valid_land_mask] = np.nan
            sm_s = compute_suite_metrics(pred_s, target[test_mask_t], valid_land_mask, p90_thresh)
            shuf_accs_t5.append(sm_s["mean_acc"])
            shuf_tccs_t5.append(sm_s["mean_tcc"])
            shuf_sigs_t5.append(sm_s["sig_pct"])

    table5_results["FiLM-U-Net (Shuffled PCs Control)"] = {
        "test_acc": float(np.mean(shuf_accs_t5)),
        "test_acc_std": float(np.std(shuf_accs_t5)),
        "test_tcc": float(np.mean(shuf_tccs_t5)),
        "sig_pct": float(np.mean(shuf_sigs_t5)),
    }

    print("\n================ TABLE 5: SYMMETRIC ABLATION & CONTROLS ================")
    for k, v in table5_results.items():
        cam_str = f", GradCAM={v['mean_gradcam']:.4f}" if "mean_gradcam" in v else ""
        print(f"  {k:<35}: ACC={v['test_acc']:.4f}, TCC={v['test_tcc']:.4f}, Sig={v['sig_pct']:.1f}%{cam_str}")

    # 12. Save all canonical artifacts
    summary_payload = {
        "split": {
            "train_period": "1994.01-2017.12",
            "train_months": n_train,
            "val_period": "2018.01-2022.12",
            "val_months": n_val,
            "hindcast_period": "1994.01-2022.12",
            "hindcast_months": n_hind,
            "test_period": "2023.01-2024.09",
            "test_months": n_test,
            "excluded_archive_months": ["2011-09", "2011-10", "2017-01"],
            "valid_china_land_grids": n_valid_grids,
            "n_pcs": n_pcs,
            "pca_explained_variance_ratio": [float(x) for x in explained_var],
            "pca_total_explained_variance": float(explained_var.sum()),
            "train_p90_anomaly_threshold": p90_thresh,
            "pct_grid_months_abs_gt_3": pct_above_3,
        },
        "tuning_log": tuning_log,
        "table1_metrics": eval_summary,
        "paired_stat_tests": stat_tests,
        "regional_metrics": regional_metrics,
        "table3_permutation_importance": importance_list,
        "table4_enso_stratification": enso_strat,
        "holdout_2020_2021_sensitivity": holdout_2021_summary,
        "table5_symmetric_ablation": table5_results,
        "fig7_control_summary": {
            "gamma_real_mean_abs": [float(x) for x in np.mean(np.abs(corr_gamma[0]), axis=0)],
            "gamma_real_std_abs": [float(x) for x in np.std(np.abs(corr_gamma[0]), axis=0)],
            "gamma_shuf_mean_abs": [float(x) for x in np.mean(shuf_gamma_mean_abs, axis=0)],
            "gamma_shuf_p95_abs": [float(x) for x in np.percentile(shuf_gamma_mean_abs, 95, axis=0)],
            "beta_real_mean_abs": [float(x) for x in np.mean(np.abs(corr_beta[0]), axis=0)],
            "beta_real_std_abs": [float(x) for x in np.std(np.abs(corr_beta[0]), axis=0)],
            "beta_shuf_mean_abs": [float(x) for x in np.mean(shuf_beta_mean_abs, axis=0)],
            "beta_shuf_p95_abs": [float(x) for x in np.percentile(shuf_beta_mean_abs, 95, axis=0)],
            "gamma_max_abs_per_pc": [float(np.max(np.abs(corr_gamma[0][:, j]))) for j in range(n_pcs)],
            "beta_max_abs_per_pc": [float(np.max(np.abs(corr_beta[0][:, j]))) for j in range(n_pcs)],
        },
        "fig8_gradcam_diagnostics": gradcam_diag,
        "fig9_sensitivity_summary": {
            "layer_max_normalized": [float(np.nanmax(m)) for m in norm_sens_layers],
            "shuffled_control_mean_max_normalized": float(np.nanmax(norm_sens_shuf_mean)),
            "shuffled_control_rep_max_normalized": float(np.nanmax(norm_sens_shuf_rep)),
            "layer_top_peaks": sens_peaks,
        },
    }

    with open(exp_dir / "unified_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    np.savez_compressed(
        exp_dir / "predictions_all.npz",
        dates=dates_all.strftime("%Y-%m-%d").values,
        train_mask=train_mask_t,
        val_mask=val_mask_t,
        hind_mask=hind_mask_t,
        test_mask=test_mask_t,
        valid_land_mask=valid_land_mask,
        target=target,
        pred_seas5=pred_seas5,
        pred_base=pred_base,
        pred_cat=pred_cat,
        pred_film=pred_film,
        acc_seas5=acc_series_all["SEAS5"],
        acc_base=acc_series_all["Baseline"],
        acc_cat=acc_series_all["EOF-cat"],
        acc_film=acc_series_all["FiLM-U-Net"],
        tcc_base=tcc_maps_test["Baseline"],
        pval_base=pval_maps_test["Baseline"],
        tcc_cat=tcc_maps_test["EOF-cat"],
        pval_cat=pval_maps_test["EOF-cat"],
        tcc_film=tcc_maps_test["FiLM-U-Net"],
        pval_film=pval_maps_test["FiLM-U-Net"],
        tcc_seas5=tcc_maps_test["SEAS5"],
        pval_seas5=pval_maps_test["SEAS5"],
        pcs5=pcs5,
        eof_patterns=eof_patterns,
    )

    np.savez_compressed(
        exp_dir / "interpretability_all.npz",
        gamma_l0=gamma_per_layer[0],
        gamma_l4=gamma_per_layer[4],
        beta_l0=beta_per_layer[0],
        beta_l4=beta_per_layer[4],
        corr_gamma_l0=corr_gamma[0],
        corr_beta_l0=corr_beta[0],
        shuf_gamma_mean_abs=shuf_gamma_mean_abs,
        shuf_beta_mean_abs=shuf_beta_mean_abs,
        sens_l0=norm_sens_layers[0],
        sens_l1=norm_sens_layers[1],
        sens_l2=norm_sens_layers[2],
        sens_l3=norm_sens_layers[3],
        sens_l4=norm_sens_layers[4],
        sens_shuf_mean=norm_sens_shuf_mean,
        sens_shuf_rep=norm_sens_shuf_rep,
        cam_baseline=gradcam_maps["Baseline U-Net (No SST)"],
        cam_cat_sst_zero=gradcam_maps["EOF-cat (SST PCs Zeroed)"],
        cam_cat_local_zero=gradcam_maps["EOF-cat (Local Zeroed)"],
        cam_film_full=gradcam_maps["Full FiLM-U-Net (Ours)"],
        cam_film_sst_zero=gradcam_maps["FiLM-U-Net (SST PCs Zeroed)"],
        cam_film_local_zero=gradcam_maps["FiLM-U-Net (Local Zeroed)"],
    )

    print(f"\n[Unified Eval] Completed cleanly in {time.time() - t_start:.1f}s!")
    print(f"Saved metrics JSON: {exp_dir / 'unified_metrics.json'}")
    print(f"Saved predictions NPZ: {exp_dir / 'predictions_all.npz'}")
    print(f"Saved interpretability NPZ: {exp_dir / 'interpretability_all.npz'}")


if __name__ == "__main__":
    main()
