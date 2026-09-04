import os
import sys
import json
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
from scipy.ndimage import gaussian_filter
import tqdm
import warnings
warnings.filterwarnings("ignore")

# Ensure project parent is in sys.path
_curr_file = os.path.abspath(__file__)
_project_root = os.path.dirname(os.path.dirname(_curr_file))
_project_parent = os.path.dirname(_project_root)
if _project_parent not in sys.path:
    sys.path.insert(0, _project_parent)

from HydroSynth.utils import utils
from HydroSynth.utils.observe_norm import DataNormalizer
from U_Net_model.unetlitefilm import UNetLiteFiLM
from project_paths import (
    ALIGNED_OBSERVATION_FILE,
    ERSST_DATA_DIR,
    MODEL_DATA_DIR,
    experiment_dir,
)

def cal_acc_np(pred, target, mask):
    N, H, W = pred.shape
    acc_list = []
    for i in range(N):
        p = pred[i][mask]
        t = target[i][mask]
        p_anom = p - np.mean(p)
        t_anom = t - np.mean(t)
        cov = np.sum(p_anom * t_anom)
        var_p = np.sum(p_anom ** 2)
        var_t = np.sum(t_anom ** 2)
        if var_p == 0 or var_t == 0:
            acc_list.append(0.0)
        else:
            acc_list.append(cov / np.sqrt(var_p * var_t))
    return np.array(acc_list)


def transform_fractional_anomaly(values, transform):
    if transform == "none":
        return values
    if transform == "signed_log1p":
        return np.sign(values) * np.log1p(np.abs(values))
    raise ValueError(f"Unknown AMS_TARGET_TRANSFORM: {transform}")


def spatial_standardize_np(fields, mask):
    """Standardize each map over valid grid cells, matching ACC invariances."""
    standardized = np.zeros_like(fields, dtype=np.float32)
    valid = fields[:, mask]
    means = valid.mean(axis=1, keepdims=True)
    scales = valid.std(axis=1, keepdims=True) + 1e-8
    standardized[:, mask] = (valid - means) / scales
    return standardized, means[:, 0], scales[:, 0]


def regional_mean_np(fields, latitudes, longitudes, lat_bounds, lon_bounds):
    """Area-weighted regional mean for fields shaped [sample, lat, lon]."""
    latitudes = np.asarray(latitudes)
    longitudes = np.mod(np.asarray(longitudes), 360.0)
    lat_low, lat_high = lat_bounds
    lon_low, lon_high = lon_bounds
    lat_mask = (latitudes >= lat_low) & (latitudes <= lat_high)
    if lon_low <= lon_high:
        lon_mask = (longitudes >= lon_low) & (longitudes <= lon_high)
    else:
        lon_mask = (longitudes >= lon_low) | (longitudes <= lon_high)
    if not np.any(lat_mask) or not np.any(lon_mask):
        raise ValueError(
            f"Empty region lat={lat_bounds}, lon={lon_bounds} for supplied grid"
        )
    region = fields[:, lat_mask][:, :, lon_mask]
    weights = np.cos(np.deg2rad(latitudes[lat_mask]))
    weights = weights / weights.sum()
    return np.nanmean(region, axis=2) @ weights


def build_physical_indices(
    forecast_fields,
    sst_fields,
    forecast_latitudes,
    forecast_longitudes,
    sst_latitudes,
    sst_longitudes,
):
    """Construct low-dimensional circulation and SST bridge predictors."""
    h200 = forecast_fields[:, 1]
    h500 = forecast_fields[:, 2]
    slp = forecast_fields[:, 3]
    u200 = forecast_fields[:, 6]
    u850 = forecast_fields[:, 7]
    v850 = forecast_fields[:, 9]

    wnp_h500 = regional_mean_np(
        h500, forecast_latitudes, forecast_longitudes, (15, 30), (110, 150)
    )
    sah_h200 = regional_mean_np(
        h200, forecast_latitudes, forecast_longitudes, (20, 35), (70, 110)
    )
    east_asia_jet = regional_mean_np(
        u200, forecast_latitudes, forecast_longitudes, (25, 40), (100, 140)
    )
    monsoon_v850 = regional_mean_np(
        v850, forecast_latitudes, forecast_longitudes, (10, 30), (105, 130)
    )
    somali_jet = regional_mean_np(
        u850, forecast_latitudes, forecast_longitudes, (0, 15), (40, 70)
    )
    wnp_slp = regional_mean_np(
        slp, forecast_latitudes, forecast_longitudes, (10, 25), (120, 150)
    )
    east_china_slp = regional_mean_np(
        slp, forecast_latitudes, forecast_longitudes, (25, 40), (105, 125)
    )
    wnp_slp_gradient = wnp_slp - east_china_slp

    latest_sst = sst_fields[:, -1]
    earliest_sst = sst_fields[:, 0]
    nino34 = regional_mean_np(
        latest_sst, sst_latitudes, sst_longitudes, (-5, 5), (190, 240)
    )
    nino34_early = regional_mean_np(
        earliest_sst, sst_latitudes, sst_longitudes, (-5, 5), (190, 240)
    )
    iod_west = regional_mean_np(
        latest_sst, sst_latitudes, sst_longitudes, (-10, 10), (50, 70)
    )
    iod_east = regional_mean_np(
        latest_sst, sst_latitudes, sst_longitudes, (-10, 0), (90, 110)
    )
    tropical_indian = regional_mean_np(
        latest_sst, sst_latitudes, sst_longitudes, (-20, 20), (40, 110)
    )
    warm_pool = regional_mean_np(
        latest_sst, sst_latitudes, sst_longitudes, (-5, 15), (120, 160)
    )
    indices = [
        wnp_h500,
        sah_h200,
        east_asia_jet,
        monsoon_v850,
        somali_jet,
        wnp_slp_gradient,
        nino34,
        nino34 - nino34_early,
        iod_west - iod_east,
        tropical_indian,
        warm_pool,
    ]
    if forecast_fields.shape[1] > 10:
        forecast_sst = forecast_fields[:, 10]
        forecast_nino34 = regional_mean_np(
            forecast_sst,
            forecast_latitudes,
            forecast_longitudes,
            (-5, 5),
            (190, 240),
        )
        forecast_iod_west = regional_mean_np(
            forecast_sst,
            forecast_latitudes,
            forecast_longitudes,
            (-10, 10),
            (50, 70),
        )
        forecast_iod_east = regional_mean_np(
            forecast_sst,
            forecast_latitudes,
            forecast_longitudes,
            (-10, 0),
            (90, 110),
        )
        indices.extend(
            [
                forecast_nino34,
                forecast_nino34 - nino34,
                forecast_iod_west - forecast_iod_east,
                regional_mean_np(
                    forecast_sst,
                    forecast_latitudes,
                    forecast_longitudes,
                    (-20, 20),
                    (40, 110),
                ),
                regional_mean_np(
                    forecast_sst,
                    forecast_latitudes,
                    forecast_longitudes,
                    (-5, 15),
                    (120, 160),
                ),
            ]
        )
    return np.column_stack(indices).astype(np.float32)


def normalize_monthly_indices(indices, months, train_end):
    """Fold-local monthly standardization for physical scalar predictors."""
    result = np.zeros_like(indices, dtype=np.float32)
    train_rows = np.arange(len(indices)) < train_end
    for month in range(1, 13):
        all_month = months == month
        fit_month = all_month & train_rows
        if not np.any(fit_month):
            continue
        mean = indices[fit_month].mean(axis=0)
        scale = indices[fit_month].std(axis=0)
        result[all_month] = np.clip(
            (indices[all_month] - mean) / np.maximum(scale, 1e-6), -3.0, 3.0
        )
    return result

def prepare_mca_target(Y, mask_y, fit_train_only=True, train_end=342):
    """Precompute the target-side SVD shared by every predictor field."""
    T = Y.shape[0]
    flat_y = Y.reshape(T, -1)[:, mask_y.flatten()]
    flat_y = np.nan_to_num(flat_y, nan=0.0, posinf=0.0, neginf=0.0)
    T_fit = train_end if fit_train_only else T
    mean_y = flat_y[:T_fit].mean(axis=0) if fit_train_only else flat_y.mean(axis=0)
    flat_y_centered = flat_y - mean_y
    U_y, s_y, Vt_y = np.linalg.svd(flat_y_centered[:T_fit], full_matrices=False)
    return flat_y_centered, U_y, s_y, Vt_y


def compute_mca_np(
    X, Y, mask_x, mask_y, fit_train_only=True, train_end=342,
    prepared_target=None, return_right_expansion=True
):
    T = X.shape[0]
    flat_x = X.reshape(T, -1)[:, mask_x.flatten()]
    flat_y = Y.reshape(T, -1)[:, mask_y.flatten()]
    
    # Clean NaNs and Inf before SVD covariance computation
    flat_x = np.nan_to_num(flat_x, nan=0.0, posinf=0.0, neginf=0.0)
    flat_y = np.nan_to_num(flat_y, nan=0.0, posinf=0.0, neginf=0.0)
    
    if fit_train_only:
        mean_x = flat_x[:train_end].mean(axis=0)
        T_fit = train_end
    else:
        mean_x = flat_x.mean(axis=0)
        T_fit = T
        
    flat_x_centered = flat_x - mean_x
    # Efficient SVD/MCA using thin SVD of X_fit and Y_fit (T_fit << P_x, P_y)
    X_fit = flat_x_centered[:T_fit]
    U_x, s_x, Vt_x = np.linalg.svd(X_fit, full_matrices=False)

    if prepared_target is None:
        flat_y_centered, U_y, s_y, Vt_y = prepare_mca_target(
            Y, mask_y, fit_train_only=fit_train_only, train_end=train_end
        )
    else:
        flat_y_centered, U_y, s_y, Vt_y = prepared_target
    
    # Core covariance matrix of size [T_fit, T_fit]
    M = (U_x.T @ U_y) * s_x[:, None] * s_y[None, :] / (T_fit - 1)
    
    U_m, s_m, Vt_m = np.linalg.svd(M, full_matrices=False)
    
    # Recover spatial patterns: U = Vt_x.T @ U_m and V = Vt_y.T @ Vt_m.T
    U = Vt_x.T @ U_m
    le = flat_x_centered @ U
    if return_right_expansion:
        V = Vt_y.T @ Vt_m.T
        re = flat_y_centered @ V
    else:
        re = None
    
    if fit_train_only:
        std_le = le[:train_end].std(axis=0) + 1e-8
    else:
        std_le = le.std(axis=0) + 1e-8
    le = le / std_le
    
    return le, re, s_m

class FiLMUNetWrapper:
    """
    Wrapper for UNetLiteFiLM to provide unified fit and predict interface.
    """
    def __init__(self, in_channels=17, index_dim=50, base_filters=16, dropout=0.1, lr=1e-3, epochs=35, batch_size=16, device=None, mask=None):
        self.in_channels = in_channels
        self.index_dim = index_dim
        self.base_filters = base_filters
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mask = mask
        self.model = None

    def fit(self, X_spatial, X_pcs, y_spatial):
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.model = UNetLiteFiLM(
            n_channels=self.in_channels,
            n_classes=1,
            index_dim=self.index_dim,
            base_filters=self.base_filters,
            dropout=self.dropout
        ).to(self.device)
        
        X_sp_t = torch.from_numpy(X_spatial.astype(np.float32))
        X_pc_t = torch.from_numpy(X_pcs.astype(np.float32))
        y_sp_t = torch.from_numpy(y_spatial.astype(np.float32)).unsqueeze(1)  # [N, 1, H, W]
        
        dataset = TensorDataset(X_sp_t, X_pc_t, y_sp_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)
        loss_fn = nn.MSELoss()
        
        mask_t = torch.from_numpy(self.mask).to(self.device) if self.mask is not None else None
        
        self.model.train()
        for epoch in range(self.epochs):
            for xb, pb, yb in loader:
                xb, pb, yb = xb.to(self.device), pb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = self.model(xb, pb)  # [B, 1, H, W]
                if mask_t is not None:
                    loss = loss_fn(out[:, 0][:, mask_t], yb[:, 0][:, mask_t])
                else:
                    loss = loss_fn(out, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
        return self

    def predict(self, X_spatial, X_pcs):
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_spatial), self.batch_size):
                xb = torch.from_numpy(X_spatial[i:i+self.batch_size].astype(np.float32)).to(self.device)
                pb = torch.from_numpy(X_pcs[i:i+self.batch_size].astype(np.float32)).to(self.device)
                out = self.model(xb, pb)  # [B, 1, H, W]
                preds.append(out.squeeze(1).cpu().numpy())
        pred_arr = np.concatenate(preds, axis=0)  # [N, H, W]
        if self.mask is not None:
            pred_arr[:, ~self.mask] = 0.0
        return pred_arr

def main():
    print("================ Multi-Lead Spatio-Temporal AMS Prediction System (FiLM-UNet + PCR) ================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Computation Device: {device}")
    
    output_dir = os.getenv("AMS_OUTPUT_DIR") or str(
        experiment_dir(os.getenv("AMS_EXPERIMENT_NAME", "ams_multilead"))
    )
    os.makedirs(output_dir, exist_ok=True)
    modes_data_path = os.getenv("MODESV21_DATA_PATH") or str(
        MODEL_DATA_DIR / "MODESv21_ecmwf_seas51"
    )
    
    # 1. Base Dates Alignment
    all_dates = pd.date_range(start='1994-01-01', end='2024-09-01', freq='MS')
    exclude_dates = [pd.to_datetime(d) for d in ['2017-01-01', '2011-09-01', '2011-10-01']]
    valid_dates = [d for d in all_dates if d not in exclude_dates]
    date_to_idx = {d: i for i, d in enumerate(valid_dates)}  # 366
    
    # Load High-Resolution Observation grid
    hr_obs_path = os.getenv(
        "AMS_OBS_PATH", str(ALIGNED_OBSERVATION_FILE)
    )
    hr_obs_full = np.load(hr_obs_path)  # [366, 120, 140]
    if hr_obs_full.shape != (len(valid_dates), 120, 140):
        raise ValueError(
            f"Observation target must have shape {(len(valid_dates), 120, 140)}, "
            f"got {hr_obs_full.shape} from {hr_obs_path}"
        )
    mask = ~np.isnan(hr_obs_full[0])
    target_transform = os.getenv("AMS_TARGET_TRANSFORM", "none")
    hr_obs_full = transform_fractional_anomaly(
        hr_obs_full, target_transform
    ).astype(np.float32)
    print(f"Target anomaly transform: {target_transform}")
    
    # Load SST config and paths
    ersst_data_path = os.getenv("ERSST_DATA_PATH") or str(ERSST_DATA_DIR)
    
    # Calculate all ERSST dates needed across target dates and leads
    all_sst_dates = set()
    for target_date in valid_dates:
        for lead in range(6):
            issue_date = target_date - pd.DateOffset(months=lead)
            for m in range(1, 7):  # Preceding 6 months
                all_sst_dates.add(issue_date - pd.DateOffset(months=m))
    all_sst_dates = sorted(list(all_sst_dates))
    
    print("\nPre-caching ERSST monthly files to memory...")
    sst_cache = {}
    mask_sst = None
    sst_latitudes = None
    sst_longitudes = None
    for sst_date in tqdm.tqdm(all_sst_dates, desc="Caching SST files"):
        ym_str = sst_date.strftime("%Y%m")
        fname = f"ersst.v5.{ym_str}.nc"
        fpath = os.path.join(ersst_data_path, fname)
        if not os.path.exists(fpath):
            continue
        try:
            ds = xr.open_dataset(fpath)
            ssta = np.squeeze(ds['ssta'].values)
            sst_cache[sst_date] = ssta
            if mask_sst is None:
                mask_sst = ~np.isnan(ssta)
                sst_latitudes = ds['lat'].values
                sst_longitudes = ds['lon'].values
            ds.close()
        except Exception as e:
            print(f"Warning: Failed to load SST {fname}: {e}")
            
    if mask_sst is None:
        raise ValueError("Error: Could not determine mask_sst as no SST files were loaded successfully.")

    # 2. Pre-cache all NC files to memory to avoid redundant disk I/O
    print("\nPre-caching ECMWF forecast files to memory...")
    modes_cache = {}
    forecast_latitudes = None
    forecast_longitudes = None
    use_forecast_sst = os.getenv("AMS_USE_FORECAST_SST", "0") == "1"
    use_forecast_sst_field = os.getenv(
        "AMS_USE_FORECAST_SST_FIELD", "0"
    ) == "1"
    
    all_issue_dates = sorted(list({
        target_date - pd.DateOffset(months=lead)
        for target_date in valid_dates
        for lead in range(6)
    }))
    
    for issue_date in tqdm.tqdm(all_issue_dates, desc="Caching NC files"):
        if issue_date < pd.to_datetime('1994-01-01') or issue_date in exclude_dates:
            continue
        ym_str = issue_date.strftime("%Y%m")
        fname = f"MODESv21_ecmwf_seas51_{ym_str}_monthly_em.nc"
        fpath = os.path.join(modes_data_path, fname)
        if not os.path.exists(fpath):
            continue
        try:
            ds = xr.open_dataset(fpath)
            # Keep the historical channel order stable; optional forecast SST
            # is appended so the wind-component indices remain unchanged.
            selected_names = [
                'tp', 'h200', 'h500', 'slp', 't2m', 't850',
                'u200', 'u850', 'v200', 'v850'
            ]
            if use_forecast_sst:
                selected_names.append('sst')
            selected = ds[selected_names]
            cond_data = selected.isel(time=slice(0, 6)).to_array().to_numpy().astype(np.float32)
            cond_data = np.swapaxes(cond_data, 0, 1)  # [6, 10, 180, 360]
            modes_cache[issue_date] = cond_data
            if forecast_latitudes is None:
                forecast_latitudes = ds['latitude'].values
                forecast_longitudes = ds['longitude'].values
            ds.close()
        except Exception as e:
            print(f"Warning: Failed to load {fname}: {e}")

    lead_avg_accs = {}
    lead_avg_ec_accs = {}
    lead_avg_pcr_rmse = {}
    lead_avg_ec_rmse = {}
    lead_avg_prmse_overall = {}
    lead_avg_prmse_grid = {}
    selected_model_by_lead = {}
    selected_weight_by_lead = {}
    best_cv_score_by_lead = {}
    obs_results_by_lead = {}
    ec_precip_anom_results_by_lead = {}
    predict_results_by_lead = {}
    target_dates_by_lead = {}
    oof_patterns_by_lead = {}
    oof_dates_by_lead = {}
    
    # MCA/PCA Hyperparameters and settings
    lead_hyperparams = {
        0: {"sigma": 1.0},
        1: {"sigma": 1.5},
        2: {"sigma": 1.8},
        3: {"sigma": 2.0},
        4: {"sigma": 2.2},
        5: {"sigma": 2.5}
    }
    fit_on_full = False
    num_test = 21
    # An expensive nested-CV run can be replayed without recomputing its five
    # folds by supplying its raw (pre-safety-fallback) selections as JSON.
    # This is an evaluation/runtime cache only; the default remains full CV.
    cached_selections = json.loads(
        os.getenv("AMS_PRESELECTED_SELECTIONS", "{}")
    )
    validate_cached_selections = os.getenv(
        "AMS_VALIDATE_PRESELECTED", "0"
    ) == "1"
    oof_output_path = os.getenv("AMS_OOF_PATH")
    # The primary objective is validation-selected ACC. A stricter one-SE
    # fallback remains available for conservative operational deployments.
    enable_safety_fallback = os.getenv(
        "AMS_ENABLE_SAFETY_FALLBACK", "0"
    ) == "1"
    
    # Loop over all 6 leads (Lead 0 to Lead 5)
    for lead in range(6):
        hparams = lead_hyperparams[lead]
        sigma = hparams["sigma"]
        print(f"\n==================== Processing Lead-{lead} Model Training & AMS ====================")
        
        # Step A: Find aligned target dates for Lead-L without leakage
        aligned_target_dates = []
        for target_date in valid_dates:
            issue_date = target_date - pd.DateOffset(months=lead)
            if issue_date not in modes_cache:
                continue
            
            # Check lag persistence availability
            lag_dates = [
                target_date - pd.DateOffset(months=lead + 1),
                target_date - pd.DateOffset(months=lead + 2),
                target_date - pd.DateOffset(months=lead + 3),
                target_date - pd.DateOffset(months=12)
            ]
            if any(ld not in date_to_idx for ld in lag_dates):
                continue
                
            aligned_target_dates.append(target_date)
            
        N = len(aligned_target_dates)
        if N <= num_test:
            print(f"Skipping Lead-{lead} due to insufficient samples ({N}).")
            continue
            
        print(f"Lead-{lead}: Aligned samples = {N} (Train={N-num_test}, Test={num_test})")
        
        # Step B: Dynamically build predictors and target matrices
        cond_list = []
        physical_forecast_list = []
        target_list = []
        sst_list = []
        lags_list = []
        
        for target_date in aligned_target_dates:
            issue_date = target_date - pd.DateOffset(months=lead)
            
            # 1. Forecast variables (Extracted from memory cache)
            cond_lead = modes_cache[issue_date][lead]
            physical_forecast_list.append(cond_lead)
            model_cond_lead = (
                cond_lead if use_forecast_sst_field else cond_lead[:10]
            )
            
            # Compute new physical variables:
            # 10: low-level wind speed (u850=7, v850=9)
            v850_speed = np.sqrt(model_cond_lead[7]**2 + model_cond_lead[9]**2)
            # 11: high-level wind speed (u200=6, v200=8)
            v200_speed = np.sqrt(model_cond_lead[6]**2 + model_cond_lead[8]**2)
            # 12: vertical wind shear magnitude
            wind_shear = np.sqrt(
                (model_cond_lead[6] - model_cond_lead[7])**2
                + (model_cond_lead[8] - model_cond_lead[9])**2
            )
            
            # Concatenate wind-speed/shear diagnostics after the base fields.
            cond_lead_ext = np.concatenate(
                [
                    model_cond_lead,
                    v850_speed[None],
                    v200_speed[None],
                    wind_shear[None],
                ],
                axis=0,
            )
            cond_list.append(cond_lead_ext)
            
            # 2. SST (No leakage: relative to issue_date, preceding 6 months)
            sst_seq = []
            for m in range(6, 0, -1):
                sst_date = issue_date - pd.DateOffset(months=m)
                if sst_date in sst_cache:
                    sst_seq.append(sst_cache[sst_date])
                else:
                    sst_seq.append(np.zeros_like(mask_sst, dtype=np.float32))
            sst_list.append(np.stack(sst_seq))  # [6, 89, 180]
            
            # 3. Lags (Leak-free lagged persistence relative to target month T)
            lag_1 = target_date - pd.DateOffset(months=lead + 1)
            lag_2 = target_date - pd.DateOffset(months=lead + 2)
            lag_3 = target_date - pd.DateOffset(months=lead + 3)
            lag_12 = target_date - pd.DateOffset(months=12)
            
            lags = [
                hr_obs_full[date_to_idx[lag_1]],
                hr_obs_full[date_to_idx[lag_2]],
                hr_obs_full[date_to_idx[lag_3]],
                hr_obs_full[date_to_idx[lag_12]]
            ]
            lags_list.append(np.stack(lags))  # [4, 120, 140]
            
            # 4. Target
            target_list.append(hr_obs_full[date_to_idx[target_date]])
            
        cond_arr = np.stack(cond_list)
        physical_forecast_arr = np.stack(physical_forecast_list)
        sst_arr = np.nan_to_num(np.stack(sst_list), nan=0.0)      # [N, 6, 89, 180]
        lags_arr = np.nan_to_num(np.stack(lags_list), nan=0.0)    # [N, 4, 120, 140]
        target_arr = np.nan_to_num(np.stack(target_list), nan=0.0)  # [N, 120, 140]
        
        # Process global forecast fields
        cond_interp = cond_arr.copy()
        cond_interp[:, 0] = cond_interp[:, 0] * 31 * 24 * 60 * 60 * 1000  # mm/month
        cond_interp[:, 0] = np.clip(cond_interp[:, 0], 0.0, None)
        use_physical_indices = os.getenv("AMS_USE_PHYSICAL_INDICES", "0") == "1"
        physical_indices_raw = None
        if use_physical_indices:
            physical_indices_raw = build_physical_indices(
                physical_forecast_arr,
                sst_arr,
                forecast_latitudes,
                forecast_longitudes,
                sst_latitudes,
                sst_longitudes,
            )
            print(
                f"Physical bridge predictors enabled: {physical_indices_raw.shape[1]} indices"
            )
        
        # Train/Test boundaries
        train_end = N - num_test
        target_months = np.array([d.month for d in aligned_target_dates])
        train_months = target_months[:train_end]
        
        # Compute anomalies (fitted on train only to avoid leakage)
        cond_anom = np.zeros_like(cond_interp)
        for c in range(cond_interp.shape[1]):
            for m in range(1, 13):
                train_idx_m = np.where(train_months == m)[0]
                all_idx_m = np.where(target_months == m)[0]
                if len(train_idx_m) == 0:
                    continue
                clim = cond_interp[train_idx_m, c].mean(axis=0)
                if c == 0:  # percent anomaly
                    cond_anom[all_idx_m, c] = (cond_interp[all_idx_m, c] - clim) / (cond_interp[train_idx_m, c].mean(axis=0) + 1e-6)
                else:
                    cond_anom[all_idx_m, c] = cond_interp[all_idx_m, c] - clim
        cond_anom[:, 0] = transform_fractional_anomaly(
            cond_anom[:, 0], target_transform
        )
                    
        # Apply Z-score + 3-sigma DataNormalizer (fitted on train only)
        cond_norm = np.zeros_like(cond_anom)
        for c in range(cond_anom.shape[1]):
            normalizer = DataNormalizer(clip_sigma=3.0)
            normalizer.fit(cond_anom[:train_end, c])
            cond_norm[:, c] = np.nan_to_num(normalizer.transform(cond_anom[:, c]), nan=0.0)
            
        # Normalize lagged persistence
        lags_norm = np.zeros_like(lags_arr)
        for c in range(4):
            normalizer_lag = DataNormalizer(clip_sigma=3.0)
            normalizer_lag.fit(lags_arr[:train_end, c])
            lags_norm[:, c] = np.nan_to_num(normalizer_lag.transform(lags_arr[:, c]), nan=0.0)
            
        # Construct forecast plus four lag channels for FiLM-UNet.
        cond_reg = cond_norm[:, :, 30:90, 70:140]
        cond_reg_tensor = torch.from_numpy(cond_reg)
        cond_reg_interp = F.interpolate(cond_reg_tensor, size=(120, 140), mode='bicubic', align_corners=True).numpy()
        spatial_arr = np.concatenate([cond_reg_interp, lags_norm], axis=1)
        
        # Climatology Target dates sin/cos
        months_sin = np.sin(2 * np.pi * target_months / 12.0)
        months_cos = np.cos(2 * np.pi * target_months / 12.0)

        # Put the raw EC precipitation anomaly on the observation grid for
        # validation-time blending and baseline fallback.
        cond_anom_china = cond_anom[:, 0, 30:90, 70:140]
        cond_anom_china_tensor = torch.from_numpy(cond_anom_china[:, None])
        cond_anom_china_interp = F.interpolate(
            cond_anom_china_tensor, size=(120, 140), mode='bicubic', align_corners=False
        ).numpy()[:, 0]
        target_pattern_arr, _, _ = spatial_standardize_np(target_arr, mask)
        ec_pattern_arr, ec_spatial_means, ec_spatial_scales = spatial_standardize_np(
            cond_anom_china_interp, mask
        )
        
        # Step C: Max Covariance Analysis (MCA)
        # Define global and regional forecast masks
        mask_cond_global = ~np.isnan(cond_norm[0, 0])
        mask_cond_regional = np.ones((60, 70), dtype=bool)
        derived_start = cond_arr.shape[1] - 3
        regional_channels = [0, 4, 5, 7, 9] + list(
            range(derived_start, cond_arr.shape[1])
        )
        if use_forecast_sst_field:
            regional_channels.append(10)

        def select_mca_components(left_expansion, singular_values):
            scf = singular_values**2 / (np.sum(singular_values**2) + 1e-12)
            n_comp = np.searchsorted(np.cumsum(scf), 0.80) + 1
            n_comp = int(np.clip(n_comp, 4, 12))
            return left_expansion[:, :n_comp]

        def build_predictor_pcs(mca_train_end):
            """Build MCA features without using targets after mca_train_end."""
            prepared_target = prepare_mca_target(
                target_pattern_arr, mask, fit_train_only=True, train_end=mca_train_end
            )
            le_ssts = []
            for ch in range(6):
                le_sst_ch, _, singular_values = compute_mca_np(
                    sst_arr[:, ch], target_pattern_arr, mask_sst, mask,
                    fit_train_only=True, train_end=mca_train_end,
                    prepared_target=prepared_target, return_right_expansion=False
                )
                le_ssts.append(select_mca_components(le_sst_ch, singular_values))

            le_channels = []
            for c in range(cond_norm.shape[1]):
                if c in regional_channels:
                    predictor_field = cond_norm[:, c, 30:90, 70:140]
                    predictor_mask = mask_cond_regional
                else:
                    predictor_field = cond_norm[:, c]
                    predictor_mask = mask_cond_global
                le_c, _, singular_values = compute_mca_np(
                    predictor_field, target_pattern_arr, predictor_mask, mask,
                    fit_train_only=True, train_end=mca_train_end,
                    prepared_target=prepared_target, return_right_expansion=False
                )
                le_channels.append(select_mca_components(le_c, singular_values))

            le_lags = []
            for c in range(4):
                le_l, _, singular_values = compute_mca_np(
                    lags_norm[:, c], target_pattern_arr, mask, mask,
                    fit_train_only=True, train_end=mca_train_end,
                    prepared_target=prepared_target, return_right_expansion=False
                )
                le_lags.append(select_mca_components(le_l, singular_values))

            feature_blocks = (
                le_ssts
                + [months_sin[:, None], months_cos[:, None]]
                + le_channels
                + le_lags
            )
            if physical_indices_raw is not None:
                normalized_physical_indices = normalize_monthly_indices(
                    physical_indices_raw, target_months, mca_train_end
                )
                feature_blocks.append(normalized_physical_indices)
                if os.getenv("AMS_USE_SEASON_INTERACTIONS", "0") == "1":
                    feature_blocks.extend(
                        [
                            normalized_physical_indices * months_sin[:, None],
                            normalized_physical_indices * months_cos[:, None],
                        ]
                    )
            return np.concatenate(feature_blocks, axis=1)

        # Final-test features may use every training target, but never a test
        # target.  Each AMS fold below builds its own expanding-window MCA map.
        predictor_pcs = build_predictor_pcs(train_end)
        index_dim = predictor_pcs.shape[1]
        
        # Training data slices
        X_train_pcs = predictor_pcs[:train_end]
        X_train_spatial = spatial_arr[:train_end]
        y_train_spatial = target_pattern_arr[:train_end]
        
        # Target PCA (explaining 40% of target variance)
        pca_target = PCA(n_components=0.40, svd_solver='full')
        pca_target.fit(y_train_spatial[:, mask])
        y_train_pcs = pca_target.transform(y_train_spatial[:, mask])

        print(
            f"Predictor PCs dimension: {index_dim} | Target PCA components: {pca_target.n_components_}"
        )
        
        # Step D: Adaptive Model Selection (AMS) via 5-Fold Cross-Validation
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor

        ridge_specs = {
            f"Ridge_p{int(variance * 100)}_a{alpha:g}": (variance, alpha)
            for variance in (0.40, 0.60, 0.80)
            for alpha in (1.0, 5.0, 20.0, 100.0)
        }
        target_variance_by_model = {
            name: variance for name, (variance, _) in ridge_specs.items()
        }
        pls_specs = {
            f"PLS_{components}": components for components in (2, 4, 8)
        }
        target_variance_by_model.update(
            {name: 0.60 for name in pls_specs}
        )

        def make_candidate_factories(film_index_dim):
            factories = {
                "FiLM_UNet": lambda: FiLMUNetWrapper(
                    in_channels=spatial_arr.shape[1],
                    index_dim=film_index_dim,
                    base_filters=16,
                    dropout=0.1,
                    lr=1e-3,
                    epochs=35,
                    batch_size=16,
                    device=device,
                    mask=mask
                ),
                "RandomForest": lambda: RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
                "LightGBM": lambda: MultiOutputRegressor(LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)),
                "XGBoost": lambda: MultiOutputRegressor(XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1)),
                "KernelRidge_Poly": lambda: KernelRidge(alpha=1.0, kernel='poly', degree=2),
                "SVR_RBF": lambda: MultiOutputRegressor(SVR(kernel='rbf', C=10.0, epsilon=0.1)),
            }
            for name, components in pls_specs.items():
                factories[name] = lambda components=components: PLSRegression(
                    n_components=components,
                    scale=True,
                    max_iter=1000,
                )
            for name, (_, alpha) in ridge_specs.items():
                factories[name] = lambda alpha=alpha: Ridge(alpha=alpha)
            factories["EC_Baseline"] = None
            return factories

        candidate_names = list(make_candidate_factories(index_dim).keys())
        requested_candidates = os.getenv("AMS_CANDIDATES", "").strip()
        if requested_candidates:
            requested = [
                name.strip() for name in requested_candidates.split(",") if name.strip()
            ]
            unknown = sorted(set(requested) - set(candidate_names))
            if unknown:
                raise ValueError(f"Unknown AMS_CANDIDATES: {unknown}")
            candidate_names = list(dict.fromkeys(requested + ["EC_Baseline"]))
        cached_selection = cached_selections.get(str(lead))
        if cached_selection is not None:
            cached_model_name = cached_selection["model"]
            if cached_model_name not in candidate_names:
                raise ValueError(
                    f"Unknown cached model for Lead-{lead}: {cached_model_name}"
                )
            candidate_names = list(dict.fromkeys([cached_model_name, "EC_Baseline"]))

        print(f"\n--- Running AMS 5-Fold Time-Series Cross-Validation for Lead-{lead} ---")
        # The production test set is the last `num_test` chronological
        # samples.  Validation must therefore use contiguous, past-to-future
        # windows of the same size rather than shuffled random folds.
        cv_splits = (
            TimeSeriesSplit(n_splits=5, test_size=num_test).split(X_train_pcs)
            if cached_selection is None or validate_cached_selections
            else []
        )
        cv_fold_weights = np.arange(1.0, 6.0) ** 2
        cv_fold_weights /= cv_fold_weights.sum()
        blend_weights = np.linspace(0.0, 1.0, 11)  # model contribution; 0 = EC only
        cv_scores_by_weight = {
            name: {float(w): [] for w in blend_weights}
            for name in candidate_names
        }
        cv_oof_by_model = (
            {
                name: np.full((train_end, 120, 140), np.nan, dtype=np.float32)
                for name in candidate_names
            }
            if oof_output_path and (cached_selection is None or validate_cached_selections)
            else None
        )

        if cached_selection is not None and not validate_cached_selections:
            cached_weight = float(cached_selection["weight"])
            cached_cv_acc = float(cached_selection["cv_acc"])
            cached_ec_cv_acc = float(cached_selection["ec_cv_acc"])
            if not np.any(np.isclose(blend_weights, cached_weight)):
                raise ValueError(
                    f"Cached weight for Lead-{lead} is not on the blend grid: "
                    f"{cached_weight}"
                )
            # Populate the score table so the ordinary selection/reporting
            # path can be reused. Non-selected weights are only sentinels;
            # their scores are not interpreted as fresh CV measurements.
            for weight in blend_weights:
                cv_scores_by_weight[cached_model_name][float(weight)] = [
                    cached_cv_acc - abs(float(weight) - cached_weight)
                ] * 5
                cv_scores_by_weight["EC_Baseline"][float(weight)] = [
                    cached_ec_cv_acc
                ] * 5
            print(
                f"  Reusing cached leak-free CV selection: {cached_model_name}, "
                f"weight={cached_weight:.1f}"
            )

        for fold, (trn_idx, val_idx) in enumerate(cv_splits):
            # MCA is supervised (it uses Y), so it belongs inside the CV fold.
            # Fitting it once before splitting leaks validation targets and was
            # the largest source of the old optimistic CV scores.
            fold_predictor_pcs = build_predictor_pcs(len(trn_idx))
            fold_factories = make_candidate_factories(fold_predictor_pcs.shape[1])
            # Fit every requested target-PCA threshold on the training fold.
            pca_by_variance = {}
            y_pcs_by_variance = {}
            for target_variance in (0.40, 0.60, 0.80):
                pca_for_variance = PCA(n_components=target_variance, svd_solver='full')
                y_pcs_for_variance = pca_for_variance.fit_transform(
                    y_train_spatial[trn_idx][:, mask]
                )
                pca_by_variance[target_variance] = pca_for_variance
                y_pcs_by_variance[target_variance] = y_pcs_for_variance
            ec_val = ec_pattern_arr[val_idx]
            
            for name in candidate_names:
                factory = fold_factories[name]
                if name == "EC_Baseline":
                    pred_val_smoothed = ec_val
                else:
                    m = factory()
                    if name == "FiLM_UNet":
                        m.fit(X_train_spatial[trn_idx], fold_predictor_pcs[trn_idx], y_train_spatial[trn_idx])
                        pred_val = m.predict(X_train_spatial[val_idx], fold_predictor_pcs[val_idx])
                    else:
                        target_variance = target_variance_by_model.get(name, 0.40)
                        pca_fold = pca_by_variance[target_variance]
                        y_trn_pcs_fold = y_pcs_by_variance[target_variance]
                        m.fit(fold_predictor_pcs[trn_idx], y_trn_pcs_fold)
                        pred_val_pcs = m.predict(fold_predictor_pcs[val_idx])
                        pred_val_flat = pca_fold.inverse_transform(pred_val_pcs)
                        pred_val = np.zeros((len(val_idx), 120, 140), dtype=np.float32)
                        pred_val[:, mask] = pred_val_flat

                    pred_val_smoothed = np.zeros_like(pred_val)
                    for i in range(len(val_idx)):
                        pred_val_smoothed[i] = gaussian_filter(pred_val[i], sigma=sigma)
                    pred_val_smoothed[:, ~mask] = 0.0
                    pred_val_smoothed, _, _ = spatial_standardize_np(
                        pred_val_smoothed, mask
                    )

                if cv_oof_by_model is not None:
                    cv_oof_by_model[name][val_idx] = pred_val_smoothed

                # Score blends against the same ACC used for the final report.
                # weight=0 is exactly raw EC and weight=1 is the model field.
                for weight in blend_weights:
                    blended = weight * pred_val_smoothed + (1.0 - weight) * ec_val
                    fold_acc = np.mean(cal_acc_np(blended, target_pattern_arr[val_idx], mask))
                    cv_scores_by_weight[name][float(weight)].append(fold_acc)

        mean_cv_acc = {}
        best_weight_by_model = {}
        for name, scores_by_weight in cv_scores_by_weight.items():
            mean_by_weight = {
                w: float(np.average(scores, weights=cv_fold_weights))
                for w, scores in scores_by_weight.items()
            }
            best_weight = max(mean_by_weight, key=mean_by_weight.get)
            best_weight_by_model[name] = best_weight
            mean_cv_acc[name] = mean_by_weight[best_weight]
            print(
                f"  Candidate: {name:<20} | Recency CV ACC: {mean_cv_acc[name]:.5f} "
                f"| Model Weight: {best_weight:.1f}"
            )

        best_model_name = max(mean_cv_acc, key=mean_cv_acc.get)
        best_cv_score = mean_cv_acc[best_model_name]
        best_blend_weight = best_weight_by_model[best_model_name]
        ec_cv_score = mean_cv_acc["EC_Baseline"]
        selected_fold_scores = np.asarray(
            cv_scores_by_weight[best_model_name][best_blend_weight]
        )
        ec_fold_scores = np.asarray(cv_scores_by_weight["EC_Baseline"][0.0])
        paired_gains = selected_fold_scores - ec_fold_scores
        effective_folds = 1.0 / np.sum(cv_fold_weights**2)
        weighted_gain = float(np.average(paired_gains, weights=cv_fold_weights))
        weighted_gain_variance = np.sum(
            cv_fold_weights * (paired_gains - weighted_gain) ** 2
        ) / (1.0 - np.sum(cv_fold_weights**2))
        gain_standard_error = np.sqrt(weighted_gain_variance / effective_folds)
        required_gain = max(0.005, gain_standard_error)
        observed_gain = weighted_gain
        print(
            "  Selected fold ACCs: "
            + ", ".join(f"{score:.4f}" for score in selected_fold_scores)
            + " | EC: "
            + ", ".join(f"{score:.4f}" for score in ec_fold_scores)
        )

        if cv_oof_by_model is not None:
            raw_oof = cv_oof_by_model[best_model_name]
            ec_oof = cv_oof_by_model["EC_Baseline"]
            available = np.isfinite(raw_oof[:, 0, 0])
            blended_oof = (
                best_blend_weight * raw_oof[available]
                + (1.0 - best_blend_weight) * ec_oof[available]
            )
            blended_oof, _, _ = spatial_standardize_np(blended_oof, mask)
            oof_patterns_by_lead[lead] = blended_oof[:, mask]
            oof_dates_by_lead[lead] = np.asarray(
                [aligned_target_dates[index].strftime("%Y-%m-%d") for index in np.where(available)[0]]
            )

        # Multiple model/weight comparisons on five folds have a substantial
        # winner's curse.  Require a practically meaningful paired gain that
        # also clears one standard error; otherwise retain raw EC.
        if (
            enable_safety_fallback
            and (best_blend_weight == 0.0 or observed_gain <= required_gain)
        ):
            if best_model_name != "EC_Baseline":
                print(
                    f"  Safety fallback to EC: gain={observed_gain:.5f}, "
                    f"required>{required_gain:.5f}"
                )
            best_model_name = "EC_Baseline"
            best_cv_score = ec_cv_score
            best_blend_weight = 0.0
        selected_model_by_lead[lead] = best_model_name
        selected_weight_by_lead[lead] = best_blend_weight
        best_cv_score_by_lead[lead] = best_cv_score
        print(
            f">>> AMS Selected Best Model for Lead-{lead}: {best_model_name} "
            f"(CV ACC: {best_cv_score:.5f}, Model Weight: {best_blend_weight:.1f})"
        )
        
        # Step E: Full Training of Selected Best Model
        if best_model_name == "EC_Baseline":
            pred_all_pattern = ec_pattern_arr.copy()
        else:
            final_model_factories = make_candidate_factories(index_dim)
            best_model = final_model_factories[best_model_name]()
            if best_model_name == "FiLM_UNet":
                best_model.fit(X_train_spatial, X_train_pcs, y_train_spatial)
                pred_all_recon = best_model.predict(spatial_arr, predictor_pcs)
            else:
                selected_target_variance = target_variance_by_model.get(best_model_name, 0.40)
                if selected_target_variance == 0.40:
                    selected_pca_target = pca_target
                    selected_y_train_pcs = y_train_pcs
                else:
                    selected_pca_target = PCA(
                        n_components=selected_target_variance, svd_solver='full'
                    )
                    selected_y_train_pcs = selected_pca_target.fit_transform(
                        y_train_spatial[:, mask]
                    )
                best_model.fit(X_train_pcs, selected_y_train_pcs)
                pred_all_pcs = best_model.predict(predictor_pcs)
                pred_all_flat = selected_pca_target.inverse_transform(pred_all_pcs)
                pred_all_recon = np.zeros_like(target_arr)
                pred_all_recon[:, mask] = pred_all_flat

            pred_all_smoothed = np.zeros_like(pred_all_recon)
            for i in range(N):
                pred_all_smoothed[i] = gaussian_filter(pred_all_recon[i], sigma=sigma)
            pred_all_smoothed[:, ~mask] = 0.0
            pred_all_pattern, _, _ = spatial_standardize_np(pred_all_smoothed, mask)

        blended_pattern = (
            best_blend_weight * pred_all_pattern
            + (1.0 - best_blend_weight) * ec_pattern_arr
        )
        blended_pattern, _, _ = spatial_standardize_np(blended_pattern, mask)
        pred_all_smoothed = np.zeros_like(target_arr)
        pred_all_smoothed[:, mask] = (
            blended_pattern[:, mask] * ec_spatial_scales[:, None]
            + ec_spatial_means[:, None]
        )
        pred_test_smoothed = pred_all_smoothed[train_end:]
        
        # Retrieve test actual observations and EC baseline
        obs_test = target_arr[train_end:]  # [21, 120, 140]
        mod_test = cond_anom_china_interp[train_end:]  # EC anomaly [21, 120, 140]
        
        # 1. Evaluate ACC
        test_accs = cal_acc_np(pred_test_smoothed, obs_test, mask)
        avg_acc = np.mean(test_accs)
        lead_avg_accs[lead] = avg_acc
        
        test_ec_accs = cal_acc_np(mod_test, obs_test, mask)
        avg_ec_acc = np.mean(test_ec_accs)
        lead_avg_ec_accs[lead] = avg_ec_acc
        
        # 2. Evaluate RMSE & P-RMSE decrease rates
        sq_err_pre = (pred_test_smoothed - obs_test) ** 2
        sq_err_mod = (mod_test - obs_test) ** 2
        
        pre_rmse_grid = np.sqrt(np.mean(sq_err_pre, axis=0))
        mod_rmse_grid = np.sqrt(np.mean(sq_err_mod, axis=0))
        
        pre_rmse_masked = pre_rmse_grid[mask]
        mod_rmse_masked = mod_rmse_grid[mask]
        
        avg_pcr_rmse = np.mean(pre_rmse_masked)
        avg_ec_rmse = np.mean(mod_rmse_masked)
        
        prmse_overall = ((avg_ec_rmse - avg_pcr_rmse) / (avg_ec_rmse + 1e-8)) * 100
        prmse_grid_masked = ((mod_rmse_masked - pre_rmse_masked) / (mod_rmse_masked + 1e-8)) * 100
        avg_prmse_grid = np.mean(prmse_grid_masked)
        
        lead_avg_pcr_rmse[lead] = avg_pcr_rmse
        lead_avg_ec_rmse[lead] = avg_ec_rmse
        lead_avg_prmse_overall[lead] = prmse_overall
        lead_avg_prmse_grid[lead] = avg_prmse_grid
        
        print(f"Lead-{lead} [{best_model_name}] Average Test ACC: {avg_acc:.6f} | EC ACC: {avg_ec_acc:.6f} | Model RMSE: {avg_pcr_rmse:.6f} | EC RMSE: {avg_ec_rmse:.6f} | Overall P-RMSE Decr: {prmse_overall:.2f}% | Grid P-RMSE: {avg_prmse_grid:.2f}%")
        
        target_dates_by_lead[lead] = aligned_target_dates
        obs_results_by_lead[lead] = target_arr
        ec_precip_anom_results_by_lead[lead] = cond_anom_china_interp
        predict_results_by_lead[lead] = pred_all_smoothed
    
    # Save multi-lead arrays
    result_dates = sorted({d for dates in target_dates_by_lead.values() for d in dates})
    result_date_to_idx = {d: i for i, d in enumerate(result_dates)}
    result_shape = (len(result_dates), 6, 120, 140)
    
    obs_results = np.full(result_shape, np.nan, dtype=np.float32)
    ec_precip_anom_results = np.full(result_shape, np.nan, dtype=np.float32)
    predict_results = np.full(result_shape, np.nan, dtype=np.float32)
    
    for lead, dates in target_dates_by_lead.items():
        date_indices = [result_date_to_idx[d] for d in dates]
        obs_results[date_indices, lead] = obs_results_by_lead[lead]
        ec_precip_anom_results[date_indices, lead] = ec_precip_anom_results_by_lead[lead]
        predict_results[date_indices, lead] = predict_results_by_lead[lead]
    
    np.save(os.path.join(output_dir, "multi_lead_obs_results.npy"), obs_results)
    np.save(os.path.join(output_dir, "multi_lead_ec_precip_anom_results.npy"), ec_precip_anom_results)
    np.save(os.path.join(output_dir, "multi_lead_predict_results.npy"), predict_results)
    result_dates_str = np.array([d.strftime("%Y-%m-%d") for d in result_dates])
    np.save(os.path.join(output_dir, "multi_lead_dates.npy"), result_dates_str)
    if oof_output_path:
        missing = sorted(set(range(6)) - set(oof_patterns_by_lead))
        if missing:
            raise RuntimeError(f"OOF predictions were not generated for leads: {missing}")
        np.savez_compressed(
            oof_output_path,
            dates=np.stack([oof_dates_by_lead[lead] for lead in range(6)], axis=1),
            predictions=np.stack(
                [oof_patterns_by_lead[lead] for lead in range(6)], axis=1
            ),
        )
    
    print("\n============================ Multi-Lead Test Performance & AMS Summary ============================")
    print(f"  {'Lead':<6} | {'Selected Model':<18} | {'Weight':<6} | {'CV ACC':<8} | {'Test ACC':<10} | {'EC ACC':<10} | {'Model RMSE':<10} | {'EC RMSE':<10} | {'P-RMSE Decr':<12}")
    print("-" * 116)
    for lead in sorted(lead_avg_accs.keys()):
        print(f"  Lead-{lead:<1} | {selected_model_by_lead[lead]:<18} | {selected_weight_by_lead[lead]:.1f}    | {best_cv_score_by_lead[lead]:.5f}  | {lead_avg_accs[lead]:.6f} | {lead_avg_ec_accs[lead]:.6f} | {lead_avg_pcr_rmse[lead]:.6f} | {lead_avg_ec_rmse[lead]:.6f} | {lead_avg_prmse_overall[lead]:.2f}%")

    macro_test_acc = np.mean(list(lead_avg_accs.values()))
    macro_ec_acc = np.mean(list(lead_avg_ec_accs.values()))
    print(
        f"Macro Test ACC: {macro_test_acc:.6f} | Macro EC ACC: {macro_ec_acc:.6f} "
        f"| Absolute gain: {macro_test_acc - macro_ec_acc:+.6f}"
    )
    
if __name__ == "__main__":
    main()
