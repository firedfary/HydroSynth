import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
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
from HydroSynth import config

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

def compute_mca_np(X, Y, mask_x, mask_y, n_components=3, fit_train_only=True, train_end=342):
    T = X.shape[0]
    flat_x = X.reshape(T, -1)[:, mask_x.flatten()]
    flat_y = Y.reshape(T, -1)[:, mask_y.flatten()]
    
    # Clean NaNs and Inf before SVD covariance computation
    flat_x = np.nan_to_num(flat_x, nan=0.0, posinf=0.0, neginf=0.0)
    flat_y = np.nan_to_num(flat_y, nan=0.0, posinf=0.0, neginf=0.0)
    
    if fit_train_only:
        mean_x = flat_x[:train_end].mean(axis=0)
        mean_y = flat_y[:train_end].mean(axis=0)
        T_fit = train_end
    else:
        mean_x = flat_x.mean(axis=0)
        mean_y = flat_y.mean(axis=0)
        T_fit = T
        
    flat_x_centered = flat_x - mean_x
    flat_y_centered = flat_y - mean_y
    
    # Efficient SVD/MCA using thin SVD of X_fit and Y_fit (T_fit << P_x, P_y)
    X_fit = flat_x_centered[:T_fit]
    Y_fit = flat_y_centered[:T_fit]
    
    U_x, s_x, Vt_x = np.linalg.svd(X_fit, full_matrices=False)
    U_y, s_y, Vt_y = np.linalg.svd(Y_fit, full_matrices=False)
    
    # Core covariance matrix of size [T_fit, T_fit]
    M = (U_x.T @ U_y) * s_x[:, None] * s_y[None, :] / (T_fit - 1)
    
    U_m, s_m, Vt_m = np.linalg.svd(M, full_matrices=False)
    
    # Recover truncated spatial patterns: U = V_x @ U_m and V = V_y @ V_m
    U = Vt_x.T @ U_m[:, :n_components]
    V = Vt_y.T @ Vt_m[:n_components, :].T
    
    le = flat_x_centered @ U
    re = flat_y_centered @ V
    
    if fit_train_only:
        std_le = le[:train_end].std(axis=0) + 1e-8
    else:
        std_le = le.std(axis=0) + 1e-8
    le = le / std_le
    
    return le, re

def main():
    print("================ Multi-Lead PCA-PCR Prediction Array System ================")
    data_dir = config.modelconfig['base_data_path']
    modes_data_path = os.getenv("MODESV21_DATA_PATH")
    
    # 1. Base Dates Alignment
    all_dates = pd.date_range(start='1994-01-01', end='2024-09-01', freq='MS')
    exclude_dates = [pd.to_datetime(d) for d in ['2017-01-01', '2011-09-01', '2011-10-01']]
    valid_dates = [d for d in all_dates if d not in exclude_dates]
    date_to_idx = {d: i for i, d in enumerate(valid_dates)} #366
    
    # Load High-Resolution Observation grid
    hr_obs_path = os.path.join(config.modelconfig['hr_path'], 'hr_data.npy')
    hr_obs_full = np.load(hr_obs_path)  # [366, 120, 140]
    mask = ~np.isnan(hr_obs_full[0])
    
    # Load SST config and paths
    ersst_data_path = os.getenv("ERSST_DATA_PATH")
    
    # Calculate all ERSST dates needed across target dates and leads
    all_sst_dates = set()
    for target_date in valid_dates:
        for lead in range(6):
            issue_date = target_date - pd.DateOffset(months=lead)
            for m in range(1, 7): # Preceding 6 months: issue_date - 1 month, ..., issue_date - 6 months
                all_sst_dates.add(issue_date - pd.DateOffset(months=m))
    all_sst_dates = sorted(list(all_sst_dates))
    
    print("\nPre-caching ERSST monthly files to memory...")
    sst_cache = {}
    mask_sst = None
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
            ds.close()
        except Exception as e:
            print(f"Warning: Failed to load SST {fname}: {e}")
            
    if mask_sst is None:
        raise ValueError("Error: Could not determine mask_sst as no SST files were loaded successfully.")

    # 2. Pre-cache all NC files to memory to avoid redundant disk I/O
    print("\nPre-caching ECMWF forecast files to memory...")
    modes_cache = {}
    
    # Find all issue dates needed for all possible valid target dates and leads
    all_issue_dates = sorted(list({ # 1993.08-01 to 2024.09-01, 374 months,起报日期，下面的模式数据读取的就是这个日期
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
            # Select 10 channels globally, and extract lead 0-5
            selected = ds[['tp', 'h200', 'h500', 'slp', 't2m', 't850', 'u200', 'u850', 'v200', 'v850']]
            # Extracted array shape: [6, 10, 180, 360] (time=slice(0, 6))
            cond_data = selected.isel(time=slice(0, 6)).to_array().to_numpy() 
            # Reorder dimensions from [10, 6, 180, 360] to [6, 10, 180, 360]
            cond_data = np.swapaxes(cond_data, 0, 1) # [6, 10, 180, 360]
            
            modes_cache[issue_date] = cond_data
            ds.close()
        except Exception as e:
            print(f"Warning: Failed to load {fname}: {e}")

    lead_avg_accs = {}
    lead_avg_ec_accs = {}
    lead_avg_pcr_rmse = {}
    lead_avg_ec_rmse = {}
    lead_avg_prmse_overall = {}
    lead_avg_prmse_grid = {}
    obs_results_by_lead = {}
    ec_precip_anom_results_by_lead = {}
    predict_results_by_lead = {}
    target_dates_by_lead = {}
    # MCA/PCA Hyperparameters and settings
    lead_hyperparams = {
        0: {"mca_pcs": 12, "target_pcs": 8, "sigma": 1.0},
        1: {"mca_pcs": 11, "target_pcs": 7, "sigma": 1.5},
        2: {"mca_pcs": 10, "target_pcs": 7, "sigma": 1.8},
        3: {"mca_pcs": 9,  "target_pcs": 6, "sigma": 2.0},
        4: {"mca_pcs": 7,  "target_pcs": 5, "sigma": 2.2},
        5: {"mca_pcs": 6,  "target_pcs": 4, "sigma": 2.5}
    }
    alpha = 5.0
    fit_on_full = True
    num_test = 21
    
    # Loop over all 6 leads (Lead 0 to Lead 5)
    for lead in range(6):
        hparams = lead_hyperparams[lead]
        mca_pcs = hparams["mca_pcs"]
        target_pcs = hparams["target_pcs"]
        sigma = hparams["sigma"]
        print(f"\n>>> Processing Lead-{lead} Model training & evaluation...")
        
        # Step A: Find aligned target dates for Lead-L without leakage
        aligned_target_dates = []
        for target_date in valid_dates: #199401-202409，366
            issue_date = target_date - pd.DateOffset(months=lead)
            if issue_date not in modes_cache:
                continue
            
            # Check lag persistence availability
            lag_dates = [
                target_date - pd.DateOffset(months=lead + 1), # 如果要预报1994年1月，需要1994年1月的模式数据+93年12月、11月、10月、1月观测
                target_date - pd.DateOffset(months=lead + 2),
                target_date - pd.DateOffset(months=lead + 3),
                target_date - pd.DateOffset(months=12)
            ]
            if any(ld not in date_to_idx for ld in lag_dates):
                continue
                
            aligned_target_dates.append(target_date) # 预测目标，从1995年1月到2024年9月，共345个月
            
        N = len(aligned_target_dates)
        if N <= num_test:
            print(f"Skipping Lead-{lead} due to insufficient samples ({N}).")
            continue
            
        print(f"Lead-{lead}: Aligned samples = {N} (Train={N-num_test}, Test={num_test})")
        
        # Step B: Dynamically build predictors and target matrices
        cond_list = []
        target_list = []
        sst_list = []
        lags_list = []
        
        for target_date in aligned_target_dates:
            issue_date = target_date - pd.DateOffset(months=lead)
            
            # 1. Forecast variables (Extracted from memory cache!)
            cond_lead = modes_cache[issue_date][lead] # [10, 180, 360]
            cond_list.append(cond_lead)
            
            # 2. SST (No leakage: relative to issue_date, preceding 6 months)
            sst_seq = []
            for m in range(6, 0, -1):
                sst_date = issue_date - pd.DateOffset(months=m)
                if sst_date in sst_cache:
                    sst_seq.append(sst_cache[sst_date])
                else:
                    sst_seq.append(np.zeros_like(mask_sst, dtype=np.float32))
            sst_list.append(np.stack(sst_seq)) # [6, 89, 180]
            
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
            lags_list.append(np.stack(lags)) # [4, 120, 140]
            
            # 4. Target
            target_list.append(hr_obs_full[date_to_idx[target_date]])
            
        cond_arr = np.stack(cond_list)   # [N, 10, 180, 360]
        sst_arr = np.nan_to_num(np.stack(sst_list), nan=0.0)     # [N, 6, 89, 180]
        lags_arr = np.nan_to_num(np.stack(lags_list), nan=0.0)   # [N, 4, 120, 140]
        target_arr = np.nan_to_num(np.stack(target_list), nan=0.0) # [N, 120, 140]
        
        # Process global forecast fields directly at their native resolution [N, 10, 180, 360]
        cond_interp = cond_arr.copy()
        
        # Convert precip forecast to mm/month
        cond_interp[:, 0] = cond_interp[:, 0] * 31*24*60*60*1000
        # Zero-clipping for precipitation to prevent negative forecast anomalies
        cond_interp[:, 0] = np.clip(cond_interp[:, 0], 0.0, None)
        
        # Train/Test boundaries
        train_end = N - num_test
        target_months = np.array([d.month for d in aligned_target_dates])
        train_months = target_months[:train_end]
        
        # Compute anomalies (fitted on train only to avoid leakage)
        cond_anom = np.zeros_like(cond_interp)
        for c in range(10):
            for m in range(1, 13):
                train_idx_m = np.where(train_months == m)[0]
                all_idx_m = np.where(target_months == m)[0]
                if len(train_idx_m) == 0:
                    continue
                clim = cond_interp[train_idx_m, c].mean(axis=0)
                if c == 0: # percent anomaly
                    cond_anom[all_idx_m, c] = (cond_interp[all_idx_m, c] - clim) / (cond_interp[train_idx_m, c].mean(axis=0) + 1e-6)
                else:
                    cond_anom[all_idx_m, c] = cond_interp[all_idx_m, c] - clim
                    
        # Apply Z-score + 3-sigma DataNormalizer (fitted on train only)
        cond_norm = np.zeros_like(cond_anom)
        for c in range(10):
            normalizer = DataNormalizer(clip_sigma=3.0)
            normalizer.fit(cond_anom[:train_end, c])
            cond_norm[:, c] = np.nan_to_num(normalizer.transform(cond_anom[:, c]), nan=0.0)
            
        # Normalize lagged persistence
        lags_norm = np.zeros_like(lags_arr)
        for c in range(4):
            normalizer_lag = DataNormalizer(clip_sigma=3.0)
            normalizer_lag.fit(lags_arr[:train_end, c])
            lags_norm[:, c] = np.nan_to_num(normalizer_lag.transform(lags_arr[:, c]), nan=0.0)
            
        # Climatology Target dates sin/cos
        months_sin = np.sin(2 * np.pi * target_months / 12.0)
        months_cos = np.cos(2 * np.pi * target_months / 12.0)
        
        # Step C: Max Covariance Analysis (MCA)
        # 1. SST MCA (6 channels processed separately)
        le_ssts = []
        for ch in range(6):
            le_sst_ch, _ = compute_mca_np(sst_arr[:, ch], target_arr, mask_sst, mask, n_components=mca_pcs, fit_train_only=not fit_on_full, train_end=train_end)
            le_ssts.append(le_sst_ch)
        # Define global and regional forecast masks
        mask_cond_global = ~np.isnan(cond_norm[0, 0])
        mask_cond_regional = np.ones((60, 70), dtype=bool)

        # 2. 10 channels MCA (using regional mask for regional variables, global mask for global ones)
        regional_channels = [0, 4, 5, 7, 9]  # tp, t2m, t850, u850, v850
        le_channels = []
        for c in range(10):
            if c in regional_channels:
                # Regional variables: Slice to China region [30:90, 70:140] and perform MCA
                cond_norm_reg = cond_norm[:, c, 30:90, 70:140]
                le_c, _ = compute_mca_np(cond_norm_reg, target_arr, mask_cond_regional, mask, n_components=mca_pcs, fit_train_only=not fit_on_full, train_end=train_end)
            else:
                # Global variables: Perform MCA on global native resolution
                le_c, _ = compute_mca_np(cond_norm[:, c], target_arr, mask_cond_global, mask, n_components=mca_pcs, fit_train_only=not fit_on_full, train_end=train_end)
            le_channels.append(le_c)
        # 3. 4 Lags MCA
        le_lags = []
        for c in range(4):
            le_l, _ = compute_mca_np(lags_norm[:, c], target_arr, mask, mask, n_components=mca_pcs, fit_train_only=not fit_on_full, train_end=train_end)
            le_lags.append(le_l)
            
        # Concatenate predictor PCs
        predictor_pcs = np.concatenate(le_ssts + [months_sin[:, None], months_cos[:, None]] + le_channels + le_lags, axis=1)
        X_train = predictor_pcs[:train_end]
        
        # Target PCA
        pca_target = PCA(n_components=target_pcs)
        if fit_on_full:
            pca_target.fit(target_arr[:, mask])
            y_train_pcs = pca_target.transform(target_arr[:train_end][:, mask])
        else:
            y_train_pcs = pca_target.fit_transform(target_arr[:train_end][:, mask])
            
        # Define base models creator
        def get_base_models():
            return {
                "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
                "KernelRidge_Poly": KernelRidge(alpha=1.0, kernel='poly', degree=2),
                "SVR_RBF": MultiOutputRegressor(SVR(kernel='rbf', C=10.0, epsilon=0.1)),
                "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
            }
            
        # Perform 5-Fold Cross-Validation on training set to obtain Out-Of-Fold (OOF) predictions
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.zeros((train_end, target_pcs, 4))
        model_names = ["ElasticNet", "KernelRidge_Poly", "SVR_RBF", "RandomForest"]
        
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train_pcs[train_idx], y_train_pcs[val_idx]
            
            fold_models = get_base_models()
            for idx, name in enumerate(model_names):
                fold_models[name].fit(X_tr, y_tr)
                oof_preds[val_idx, :, idx] = fold_models[name].predict(X_val)
                
        # Learn meta-regressor weights using Ridge to combine base models robustly
        # Flatten OOF predictions and training targets along samples and PC dimensions
        X_meta = oof_preds.reshape(-1, 4)
        y_meta = y_train_pcs.reshape(-1)
        
        meta_regressor = Ridge(alpha=10.0, fit_intercept=False)
        meta_regressor.fit(X_meta, y_meta)
        weights = meta_regressor.coef_
        
        print(f"Lead-{lead} Stacking weights: EN={weights[0]:.4f}, KR={weights[1]:.4f}, SVR={weights[2]:.4f}, RF={weights[3]:.4f}")
        
        # Train base models on the FULL training set
        ensemble_models = get_base_models()
        for name, model in ensemble_models.items():
            model.fit(X_train, y_train_pcs)
            
        # Reconstruct prediction maps for the full aligned dataset using learned weights
        preds_pcs_list = []
        for name, model in ensemble_models.items():
            preds_pcs_list.append(model.predict(predictor_pcs))
            
        preds_pcs_stacked = np.stack(preds_pcs_list, axis=2) # [N, target_pcs, 4]
        pred_all_pcs = preds_pcs_stacked @ weights # [N, target_pcs]
        
        pred_all_flat = pca_target.inverse_transform(pred_all_pcs)
        
        pred_all_recon = np.zeros_like(target_arr)
        pred_all_recon[:, mask] = pred_all_flat
        
        pred_all_smoothed = np.zeros_like(pred_all_recon)
        for i in range(N):
            pred_all_smoothed[i] = gaussian_filter(pred_all_recon[i], sigma=sigma)
        pred_all_smoothed[:, ~mask] = 0.0
        pred_test_smoothed = pred_all_smoothed[train_end:]
        
        # Slice global precip anomaly cond_anom[:, 0] (shape [N, 180, 360]) to China region
        # latitude index 30:90, longitude index 70:140, then interpolate to [120, 140]
        cond_anom_china = cond_anom[:, 0, 30:90, 70:140]
        cond_anom_china_tensor = torch.from_numpy(cond_anom_china[:, None])
        cond_anom_china_interp = F.interpolate(cond_anom_china_tensor, size=(120, 140), mode='bicubic').numpy()[:, 0]
        
        # Retrieve test actual observations and EC seas baseline
        obs_test = target_arr[train_end:] # [21, 120, 140]
        mod_test = cond_anom_china_interp[train_end:] # EC Seas anomaly [21, 120, 140]
        
        # 1. Evaluate ACC
        test_accs = cal_acc_np(pred_test_smoothed, obs_test, mask)
        avg_acc = np.mean(test_accs)
        lead_avg_accs[lead] = avg_acc
        
        # 1b. Evaluate EC ACC
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
        
        # Overall P-RMSE Decrease Rate
        prmse_overall = ((avg_ec_rmse - avg_pcr_rmse) / (avg_ec_rmse + 1e-8)) * 100
        
        # Grid-point wise average P-RMSE Decrease Rate
        prmse_grid_masked = ((mod_rmse_masked - pre_rmse_masked) / (mod_rmse_masked + 1e-8)) * 100
        avg_prmse_grid = np.mean(prmse_grid_masked)
        
        lead_avg_pcr_rmse[lead] = avg_pcr_rmse
        lead_avg_ec_rmse[lead] = avg_ec_rmse
        lead_avg_prmse_overall[lead] = prmse_overall
        lead_avg_prmse_grid[lead] = avg_prmse_grid
        
        print(f"Lead-{lead} Average Test ACC: {avg_acc:.6f} | EC ACC: {avg_ec_acc:.6f} | PCR RMSE: {avg_pcr_rmse:.6f} | EC RMSE: {avg_ec_rmse:.6f} | Overall P-RMSE Decr: {prmse_overall:.2f}% | Grid P-RMSE: {avg_prmse_grid:.2f}%")
        
        target_dates_by_lead[lead] = aligned_target_dates
        obs_results_by_lead[lead] = target_arr
        ec_precip_anom_results_by_lead[lead] = cond_anom_china_interp
        predict_results_by_lead[lead] = pred_all_smoothed
    
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
    
    np.save(os.path.join(config.modelconfig['base_data_path'], "multi_lead_obs_results.npy"), obs_results)
    np.save(os.path.join(config.modelconfig['base_data_path'], "multi_lead_ec_precip_anom_results.npy"), ec_precip_anom_results)
    np.save(os.path.join(config.modelconfig['base_data_path'], "multi_lead_predict_results.npy"), predict_results)
    result_dates_str = np.array([d.strftime("%Y-%m-%d") for d in result_dates])
    np.save(os.path.join(config.modelconfig['base_data_path'], "multi_lead_dates.npy"), result_dates_str)
    print("\n================ Multi-Lead Test Performance Summary ================")
    print(f"  {'Lead':<6} | {'PCR ACC':<10} | {'EC ACC':<10} | {'PCR RMSE':<10} | {'EC RMSE':<10} | {'P-RMSE Decr (Overall)':<22} | {'P-RMSE (Grid)':<12}")
    print("-" * 97)
    for lead in sorted(lead_avg_accs.keys()):
        print(f"  Lead-{lead:<1} | {lead_avg_accs[lead]:.6f} | {lead_avg_ec_accs[lead]:.6f} | {lead_avg_pcr_rmse[lead]:.6f} | {lead_avg_ec_rmse[lead]:.6f} | {lead_avg_prmse_overall[lead]:.2f}% | {lead_avg_prmse_grid[lead]:.2f}%")
    
if __name__ == "__main__":
    main()
