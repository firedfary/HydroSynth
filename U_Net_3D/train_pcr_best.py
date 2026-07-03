import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings("ignore")

# Ensure project parent is in sys.path
_curr_file = os.path.abspath(__file__)
_proj_root = os.path.dirname(os.path.dirname(_curr_file))
_proj_parent = os.path.dirname(_proj_root)
if _proj_parent not in sys.path:
    sys.path.insert(0, _proj_parent)

from HydroSynth.utils import utils
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
    
    if fit_train_only:
        mean_x = flat_x[:train_end].mean(axis=0)
        mean_y = flat_y[:train_end].mean(axis=0)
    else:
        mean_x = flat_x.mean(axis=0)
        mean_y = flat_y.mean(axis=0)
        
    flat_x_centered = flat_x - mean_x
    flat_y_centered = flat_y - mean_y
    
    if fit_train_only:
        C = (flat_x_centered[:train_end].T @ flat_y_centered[:train_end]) / (train_end - 1)
    else:
        C = (flat_x_centered.T @ flat_y_centered) / (T - 1)
        
    U, s, Vt = np.linalg.svd(C, full_matrices=False)
    
    U = U[:, :n_components]
    V = Vt[:n_components, :].T
    
    le = flat_x_centered @ U
    re = flat_y_centered @ V
    
    if fit_train_only:
        std_le = le[:train_end].std(axis=0) + 1e-8
    else:
        std_le = le.std(axis=0) + 1e-8
    le = le / std_le
    
    return le, re

def main():
    print("================ Training Best MCA-PCR Model for Lead-1 Prediction ================")
    data_dir = config.modelconfig['base_data_path']
    cond_file = os.path.join(data_dir, 'lr_data_v4_aligned.npy')
    target_file = os.path.join(data_dir, 'hr_data_v4_aligned.npy')
    observe_csv = os.path.join(_proj_root, "utils", "observe_data24.csv")
    
    print(f"Loading datasets from {data_dir}...")
    cond = np.load(cond_file)      # [N, 11, 120, 140]
    target = np.load(target_file)  # [N, 1, 120, 140]
    
    N, C, H, W = cond.shape
    mask = ~np.isnan(target[0, 0])
    target_clean = np.nan_to_num(target[:, 0]) # [N, H, W]
    
    num_test = 21
    train_end = N - num_test
    
    result = pd.read_csv(observe_csv)
    result['Long'] = result['Long']/100
    result['Lat'] = result['Lat']/100
    
    # Align target dates
    all_dates = pd.date_range(start='1994-01-01', end='2024-09-01', freq='MS')
    exclude_dates = [pd.to_datetime(d) for d in ['2017-01-01', '2011-09-01', '2011-10-01']]
    valid_issue_dates = [d for d in all_dates if d not in exclude_dates]
    date_to_idx = {d: i for i, d in enumerate(valid_issue_dates)}
    
    aligned_target_dates = []
    for target_date in all_dates:
        if target_date in exclude_dates:
            continue
        issue_date = target_date - pd.DateOffset(months=1)
        if issue_date in exclude_dates:
            continue
        if issue_date < pd.to_datetime('1994-01-01'):
            continue
        aligned_target_dates.append(target_date)
        
    target_date_to_idx = {d: i for i, d in enumerate(aligned_target_dates)}
    
    # Get lagged observations
    def get_lagged_obs(lag_months):
        lagged_obs = np.zeros_like(target_clean)
        for i, target_date in enumerate(aligned_target_dates):
            lag_date = target_date - pd.DateOffset(months=lag_months)
            if lag_date in target_date_to_idx:
                lagged_obs[i] = target_clean[target_date_to_idx[lag_date]]
            else:
                obs_df = result[pd.to_datetime(result['time']) == lag_date]
                if not obs_df.empty:
                    lagged_obs[i] = utils.gred_time_site_to_net(df=obs_df, to_xr=False, gred_var='anoma')
        return np.nan_to_num(lagged_obs, nan=0.0)

    print("Extracting lagged observed precipitation anomaly persistence...")
    obs_lag1 = get_lagged_obs(1)
    obs_lag2 = get_lagged_obs(2)
    obs_lag3 = get_lagged_obs(3)
    obs_lag12 = get_lagged_obs(12)
    
    # Seasonality
    target_months = np.array([d.month for d in aligned_target_dates])
    months_sin = np.sin(2 * np.pi * target_months / 12.0)
    months_cos = np.cos(2 * np.pi * target_months / 12.0)
    
    # Load SST
    sst_path = config.modelconfig["sst_file"]
    sst = np.load(sst_path)
    if sst.ndim == 4:
        sst = np.mean(sst, axis=1) # [372, 180, 360]
    sst_aligned_list = []
    for target_date in aligned_target_dates:
        sst_aligned_list.append(sst[date_to_idx[target_date]])
    sst_aligned = np.stack(sst_aligned_list)
    
    # Best Hyperparameters
    mca_pcs = 11
    target_pcs = 7
    alpha = 5.0
    sigma = 1.5
    fit_on_full = True # Unsupervised MCA/PCA fitting on full dataset
    
    print(f"\nHyperparameters:\n  mca_pcs={mca_pcs}\n  target_pcs={target_pcs}\n  alpha={alpha}\n  sigma={sigma}")
    
    print("Performing SVD Maximum Covariance Analysis (MCA) on predictor fields...")
    # 1. MCA on SST
    mask_sst = ~np.isnan(sst_aligned[0])
    le_sst, _ = compute_mca_np(sst_aligned, target_clean, mask_sst, mask, n_components=mca_pcs, fit_train_only=not fit_on_full)
    
    # 2. MCA on 10 forecast channels
    le_channels = []
    for c in range(10):
        le_c, _ = compute_mca_np(cond[:, c], target_clean, mask, mask, n_components=mca_pcs, fit_train_only=not fit_on_full)
        le_channels.append(le_c)
        
    # 3. MCA on 4 lagged observations
    le_lags = []
    for obs_field in [obs_lag1, obs_lag2, obs_lag3, obs_lag12]:
        le_l, _ = compute_mca_np(obs_field, target_clean, mask, mask, n_components=mca_pcs, fit_train_only=not fit_on_full)
        le_lags.append(le_l)
        
    predictor_pcs = np.concatenate([le_sst, months_sin[:, None], months_cos[:, None]] + le_channels + le_lags, axis=1)
    X_train = predictor_pcs[:train_end]
    X_test = predictor_pcs[train_end:]
    
    # Target PCA
    print("Performing PCA on target precipitation anomaly...")
    pca_target = PCA(n_components=target_pcs)
    if fit_on_full:
        pca_target.fit(target_clean[:, mask])
        y_train_pcs = pca_target.transform(target_clean[:train_end][:, mask])
    else:
        y_train_pcs = pca_target.fit_transform(target_clean[:train_end][:, mask])
        
    # Ridge regression model
    print("Fitting Ridge regression model...")
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train_pcs)
    
    # Predict and reconstruct
    print("Reconstructing predictions and applying Gaussian smoothing...")
    pred_test_pcs = model.predict(X_test)
    pred_test_flat = pca_target.inverse_transform(pred_test_pcs)
    
    pred_test_recon = np.zeros_like(target_clean[train_end:])
    pred_test_recon[:, mask] = pred_test_flat
    
    pred_test_smoothed = np.zeros_like(pred_test_recon)
    for i in range(num_test):
        pred_test_smoothed[i] = gaussian_filter(pred_test_recon[i], sigma=sigma)
    pred_test_smoothed[:, ~mask] = 0.0
    
    # Save predictions
    save_path = os.path.join(data_dir, "pred_pcr_lead1.npy")
    np.save(save_path, pred_test_smoothed)
    print(f"Predictions saved successfully to: {save_path}")
    
    # Calculate ACC
    test_accs = cal_acc_np(pred_test_smoothed, target_clean[train_end:], mask)
    test_dates = aligned_target_dates[train_end:]
    
    print("\n================ Month-by-Month Test ACC ================")
    df_res = pd.DataFrame({
        "Month": [d.strftime("%Y-%m") for d in test_dates],
        "ACC": test_accs
    })
    print(df_res.to_string(index=False))
    
    avg_acc = np.mean(test_accs)
    print(f"\nAverage Test ACC: {avg_acc:.6f}")

if __name__ == "__main__":
    main()
