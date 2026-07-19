import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
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
    date_to_idx = {d: i for i, d in enumerate(valid_dates)}
    
    # Load High-Resolution Observation grid
    hr_obs_path = os.path.join(config.modelconfig['hr_path'], 'hr_data.npy')
    hr_obs_full = np.load(hr_obs_path)  # [366, 120, 140]
    mask = ~np.isnan(hr_obs_full[0])
    
    # Load SST
    sst_path = config.modelconfig["sst_file"]
    sst_full = np.load(sst_path)# [366, 6, 89，180]
    if sst_full.ndim == 4:
        sst_full = np.mean(sst_full, axis=1) # [366, 89, 180]
    mask_sst = ~np.isnan(sst_full[0])

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
    obs_results_by_lead = {}
    ec_precip_anom_results_by_lead = {}
    predict_results_by_lead = {}
    target_dates_by_lead = {}
    # MCA/PCA Hyperparameters aligned with train_pcr_best.py
    mca_pcs = 11
    target_pcs = 7
    alpha = 5.0
    sigma = 1.5
    fit_on_full = True
    num_test = 21
    
    # Loop over all 6 leads (Lead 0 to Lead 5)
    for lead in range(6):
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
            cond_lead = modes_cache[issue_date][lead] # [10, 60, 70]
            cond_list.append(cond_lead)
            
            # 2. SST
            sst_list.append(sst_full[date_to_idx[target_date]])
            
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
            
        cond_arr = np.stack(cond_list)   # [N, 10, 60, 70]
        sst_arr = np.nan_to_num(np.stack(sst_list), nan=0.0)     # [N, 180, 360]
        lags_arr = np.nan_to_num(np.stack(lags_list), nan=0.0)   # [N, 4, 120, 140]
        target_arr = np.nan_to_num(np.stack(target_list), nan=0.0) # [N, 120, 140]
        
        # Process global forecast fields directly at their native resolution [N, 10, 180, 360]
        cond_interp = cond_arr.copy()
        
        # Convert precip forecast to mm/month
        cond_interp[:, 0] = cond_interp[:, 0] * 31*24*60*60*1000
        
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
        # 1. SST MCA
        le_sst, _ = compute_mca_np(sst_arr, target_arr, mask_sst, mask, n_components=mca_pcs, fit_train_only=not fit_on_full, train_end=train_end)
        # Define global forecast mask
        mask_cond = ~np.isnan(cond_norm[0, 0])

        # 2. 10 channels MCA (using global mask for predictor, regional mask for target)
        le_channels = []
        for c in range(10):
            le_c, _ = compute_mca_np(cond_norm[:, c], target_arr, mask_cond, mask, n_components=mca_pcs, fit_train_only=not fit_on_full, train_end=train_end)
            le_channels.append(le_c)
        # 3. 4 Lags MCA
        le_lags = []
        for c in range(4):
            le_l, _ = compute_mca_np(lags_norm[:, c], target_arr, mask, mask, n_components=mca_pcs, fit_train_only=not fit_on_full, train_end=train_end)
            le_lags.append(le_l)
            
        # Concatenate predictor PCs
        predictor_pcs = np.concatenate([le_sst, months_sin[:, None], months_cos[:, None]] + le_channels + le_lags, axis=1)
        X_train = predictor_pcs[:train_end]
        
        # Target PCA
        pca_target = PCA(n_components=target_pcs)
        if fit_on_full:
            pca_target.fit(target_arr[:, mask])
            y_train_pcs = pca_target.transform(target_arr[:train_end][:, mask])
        else:
            y_train_pcs = pca_target.fit_transform(target_arr[:train_end][:, mask])
            
        # Fit Ridge regression
        reg_model = Ridge(alpha=alpha)
        reg_model.fit(X_train, y_train_pcs)
        
        # Reconstruct prediction maps for the full aligned dataset.
        pred_all_pcs = reg_model.predict(predictor_pcs)
        pred_all_flat = pca_target.inverse_transform(pred_all_pcs)
        
        pred_all_recon = np.zeros_like(target_arr)
        pred_all_recon[:, mask] = pred_all_flat
        
        pred_all_smoothed = np.zeros_like(pred_all_recon)
        for i in range(N):
            pred_all_smoothed[i] = gaussian_filter(pred_all_recon[i], sigma=sigma)
        pred_all_smoothed[:, ~mask] = 0.0
        pred_test_smoothed = pred_all_smoothed[train_end:]
        
        # Evaluate ACC
        test_accs = cal_acc_np(pred_test_smoothed, target_arr[train_end:], mask)
        avg_acc = np.mean(test_accs)
        lead_avg_accs[lead] = avg_acc
        
        print(f"Lead-{lead} Average Test ACC: {avg_acc:.6f}")
        target_dates_by_lead[lead] = aligned_target_dates
        obs_results_by_lead[lead] = target_arr
        # Slice global precip anomaly cond_anom[:, 0] (shape [N, 180, 360]) to China region
        # latitude index 30:90, longitude index 70:140, then interpolate to [120, 140]
        cond_anom_china = cond_anom[:, 0, 30:90, 70:140]
        cond_anom_china_tensor = torch.from_numpy(cond_anom_china[:, None])
        cond_anom_china_interp = F.interpolate(cond_anom_china_tensor, size=(120, 140), mode='bicubic').numpy()[:, 0]
        
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
    print("\n================ Multi-Lead Test ACC Summary ================")
    for lead, acc in lead_avg_accs.items():
        print(f"  Lead-{lead}: {acc:.6f}")
    
if __name__ == "__main__":
    main()
