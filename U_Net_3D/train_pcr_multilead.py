import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
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
from U_Net_model.unetlitefilm import UNetLiteFiLM

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

def compute_mca_np(X, Y, mask_x, mask_y, fit_train_only=True, train_end=342):
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
    
    # Recover spatial patterns: U = Vt_x.T @ U_m and V = Vt_y.T @ Vt_m.T
    U = Vt_x.T @ U_m
    V = Vt_y.T @ Vt_m.T
    
    le = flat_x_centered @ U
    re = flat_y_centered @ V
    
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
    
    data_dir = config.modelconfig['base_data_path']
    modes_data_path = os.getenv("MODESV21_DATA_PATH")
    
    # 1. Base Dates Alignment
    all_dates = pd.date_range(start='1994-01-01', end='2024-09-01', freq='MS')
    exclude_dates = [pd.to_datetime(d) for d in ['2017-01-01', '2011-09-01', '2011-10-01']]
    valid_dates = [d for d in all_dates if d not in exclude_dates]
    date_to_idx = {d: i for i, d in enumerate(valid_dates)}  # 366
    
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
            for m in range(1, 7):  # Preceding 6 months
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
            # Select 10 channels globally, and extract lead 0-5
            selected = ds[['tp', 'h200', 'h500', 'slp', 't2m', 't850', 'u200', 'u850', 'v200', 'v850']]
            cond_data = selected.isel(time=slice(0, 6)).to_array().to_numpy().astype(np.float32)
            cond_data = np.swapaxes(cond_data, 0, 1)  # [6, 10, 180, 360]
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
    selected_model_by_lead = {}
    best_cv_score_by_lead = {}
    obs_results_by_lead = {}
    ec_precip_anom_results_by_lead = {}
    predict_results_by_lead = {}
    target_dates_by_lead = {}
    
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
        target_list = []
        sst_list = []
        lags_list = []
        
        for target_date in aligned_target_dates:
            issue_date = target_date - pd.DateOffset(months=lead)
            
            # 1. Forecast variables (Extracted from memory cache)
            cond_lead = modes_cache[issue_date][lead]  # [10, 180, 360]
            
            # Compute new physical variables:
            # 10: low-level wind speed (u850=7, v850=9)
            v850_speed = np.sqrt(cond_lead[7]**2 + cond_lead[9]**2)
            # 11: high-level wind speed (u200=6, v200=8)
            v200_speed = np.sqrt(cond_lead[6]**2 + cond_lead[8]**2)
            # 12: vertical wind shear magnitude
            wind_shear = np.sqrt((cond_lead[6] - cond_lead[7])**2 + (cond_lead[8] - cond_lead[9])**2)
            
            # Concatenate to 13 channels
            cond_lead_ext = np.concatenate([cond_lead, v850_speed[None], v200_speed[None], wind_shear[None]], axis=0)
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
            
        cond_arr = np.stack(cond_list)   # [N, 13, 180, 360]
        sst_arr = np.nan_to_num(np.stack(sst_list), nan=0.0)      # [N, 6, 89, 180]
        lags_arr = np.nan_to_num(np.stack(lags_list), nan=0.0)    # [N, 4, 120, 140]
        target_arr = np.nan_to_num(np.stack(target_list), nan=0.0)  # [N, 120, 140]
        
        # Process global forecast fields
        cond_interp = cond_arr.copy()
        cond_interp[:, 0] = cond_interp[:, 0] * 31 * 24 * 60 * 60 * 1000  # mm/month
        cond_interp[:, 0] = np.clip(cond_interp[:, 0], 0.0, None)
        
        # Train/Test boundaries
        train_end = N - num_test
        target_months = np.array([d.month for d in aligned_target_dates])
        train_months = target_months[:train_end]
        
        # Compute anomalies (fitted on train only to avoid leakage)
        cond_anom = np.zeros_like(cond_interp)
        for c in range(13):
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
                    
        # Apply Z-score + 3-sigma DataNormalizer (fitted on train only)
        cond_norm = np.zeros_like(cond_anom)
        for c in range(13):
            normalizer = DataNormalizer(clip_sigma=3.0)
            normalizer.fit(cond_anom[:train_end, c])
            cond_norm[:, c] = np.nan_to_num(normalizer.transform(cond_anom[:, c]), nan=0.0)
            
        # Normalize lagged persistence
        lags_norm = np.zeros_like(lags_arr)
        for c in range(4):
            normalizer_lag = DataNormalizer(clip_sigma=3.0)
            normalizer_lag.fit(lags_arr[:train_end, c])
            lags_norm[:, c] = np.nan_to_num(normalizer_lag.transform(lags_arr[:, c]), nan=0.0)
            
        # Construct 17-channel spatial feature tensor for FiLM-UNet
        cond_reg = cond_norm[:, :, 30:90, 70:140]  # [N, 13, 60, 70]
        cond_reg_tensor = torch.from_numpy(cond_reg)
        cond_reg_interp = F.interpolate(cond_reg_tensor, size=(120, 140), mode='bicubic', align_corners=True).numpy()  # [N, 13, 120, 140]
        spatial_arr = np.concatenate([cond_reg_interp, lags_norm], axis=1)  # [N, 17, 120, 140]
        
        # Climatology Target dates sin/cos
        months_sin = np.sin(2 * np.pi * target_months / 12.0)
        months_cos = np.cos(2 * np.pi * target_months / 12.0)
        
        # Step C: Max Covariance Analysis (MCA)
        # 1. SST MCA (6 channels processed separately, dynamic components explaining 80% covariance)
        le_ssts = []
        for ch in range(6):
            le_sst_ch, _, s_m = compute_mca_np(sst_arr[:, ch], target_arr, mask_sst, mask, fit_train_only=not fit_on_full, train_end=train_end)
            scf = s_m**2 / np.sum(s_m**2)
            cum_scf = np.cumsum(scf)
            n_comp = np.argmax(cum_scf >= 0.80) + 1
            n_comp = np.clip(n_comp, 4, 12)
            le_ssts.append(le_sst_ch[:, :n_comp])
            
        # Define global and regional forecast masks
        mask_cond_global = ~np.isnan(cond_norm[0, 0])
        mask_cond_regional = np.ones((60, 70), dtype=bool)

        # 2. 13 channels MCA
        regional_channels = [0, 4, 5, 7, 9, 10, 11, 12]
        le_channels = []
        for c in range(13):
            if c in regional_channels:
                cond_norm_reg = cond_norm[:, c, 30:90, 70:140]
                le_c, _, s_m = compute_mca_np(cond_norm_reg, target_arr, mask_cond_regional, mask, fit_train_only=not fit_on_full, train_end=train_end)
            else:
                le_c, _, s_m = compute_mca_np(cond_norm[:, c], target_arr, mask_cond_global, mask, fit_train_only=not fit_on_full, train_end=train_end)
            scf = s_m**2 / np.sum(s_m**2)
            cum_scf = np.cumsum(scf)
            n_comp = np.argmax(cum_scf >= 0.80) + 1
            n_comp = np.clip(n_comp, 4, 12)
            le_channels.append(le_c[:, :n_comp])
            
        # 3. 4 Lags MCA
        le_lags = []
        for c in range(4):
            le_l, _, s_m = compute_mca_np(lags_norm[:, c], target_arr, mask, mask, fit_train_only=not fit_on_full, train_end=train_end)
            scf = s_m**2 / np.sum(s_m**2)
            cum_scf = np.cumsum(scf)
            n_comp = np.argmax(cum_scf >= 0.80) + 1
            n_comp = np.clip(n_comp, 4, 12)
            le_lags.append(le_l[:, :n_comp])
            
        # Concatenate predictor PCs (used for ML models and FiLM indices)
        predictor_pcs = np.concatenate(le_ssts + [months_sin[:, None], months_cos[:, None]] + le_channels + le_lags, axis=1)
        index_dim = predictor_pcs.shape[1]
        
        # Training data slices
        X_train_pcs = predictor_pcs[:train_end]
        X_train_spatial = spatial_arr[:train_end]
        y_train_spatial = target_arr[:train_end]
        
        # Target PCA (explaining 40% of target variance)
        pca_target = PCA(n_components=0.40, svd_solver='full')
        pca_target.fit(y_train_spatial[:, mask])
        y_train_pcs = pca_target.transform(y_train_spatial[:, mask])
        
        print(f"Predictor PCs dimension: {index_dim} | Target PCA components: {pca_target.n_components_}")
        
        # Step D: Adaptive Model Selection (AMS) via 5-Fold Cross-Validation
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
        
        candidate_models_factories = {
            "FiLM_UNet": lambda: FiLMUNetWrapper(
                in_channels=17,
                index_dim=index_dim,
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
            "Ridge": lambda: Ridge(alpha=5.0)
        }
        
        print(f"\n--- Running AMS 5-Fold Cross-Validation for Lead-{lead} ---")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = {name: [] for name in candidate_models_factories.keys()}
        
        for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train_pcs)):
            # Fit PCA on training fold only
            pca_fold = PCA(n_components=0.40, svd_solver='full')
            pca_fold.fit(y_train_spatial[trn_idx][:, mask])
            y_trn_pcs_fold = pca_fold.transform(y_train_spatial[trn_idx][:, mask])
            
            for name, factory in candidate_models_factories.items():
                m = factory()
                if name == "FiLM_UNet":
                    m.fit(X_train_spatial[trn_idx], X_train_pcs[trn_idx], y_train_spatial[trn_idx])
                    pred_val = m.predict(X_train_spatial[val_idx], X_train_pcs[val_idx])
                else:
                    m.fit(X_train_pcs[trn_idx], y_trn_pcs_fold)
                    pred_val_pcs = m.predict(X_train_pcs[val_idx])
                    pred_val_flat = pca_fold.inverse_transform(pred_val_pcs)
                    pred_val = np.zeros((len(val_idx), 120, 140), dtype=np.float32)
                    pred_val[:, mask] = pred_val_flat
                    
                # Smooth validation predictions for evaluation
                pred_val_smoothed = np.zeros_like(pred_val)
                for i in range(len(val_idx)):
                    pred_val_smoothed[i] = gaussian_filter(pred_val[i], sigma=sigma)
                pred_val_smoothed[:, ~mask] = 0.0
                
                fold_acc = np.mean(cal_acc_np(pred_val_smoothed, y_train_spatial[val_idx], mask))
                cv_scores[name].append(fold_acc)
                
        mean_cv_acc = {name: np.mean(scores) for name, scores in cv_scores.items()}
        for name, score in sorted(mean_cv_acc.items(), key=lambda x: x[1], reverse=True):
            print(f"  Candidate: {name:<18} | 5-Fold CV Mean ACC: {score:.5f}")
            
        best_model_name = max(mean_cv_acc, key=mean_cv_acc.get)
        best_cv_score = mean_cv_acc[best_model_name]
        selected_model_by_lead[lead] = best_model_name
        best_cv_score_by_lead[lead] = best_cv_score
        print(f">>> AMS Selected Best Model for Lead-{lead}: {best_model_name} (CV ACC: {best_cv_score:.5f})")
        
        # Step E: Full Training of Selected Best Model
        best_model = candidate_models_factories[best_model_name]()
        if best_model_name == "FiLM_UNet":
            best_model.fit(X_train_spatial, X_train_pcs, y_train_spatial)
            pred_all_recon = best_model.predict(spatial_arr, predictor_pcs)
        else:
            best_model.fit(X_train_pcs, y_train_pcs)
            pred_all_pcs = best_model.predict(predictor_pcs)
            pred_all_flat = pca_target.inverse_transform(pred_all_pcs)
            pred_all_recon = np.zeros_like(target_arr)
            pred_all_recon[:, mask] = pred_all_flat
            
        pred_all_smoothed = np.zeros_like(pred_all_recon)
        for i in range(N):
            pred_all_smoothed[i] = gaussian_filter(pred_all_recon[i], sigma=sigma)
        pred_all_smoothed[:, ~mask] = 0.0
        pred_test_smoothed = pred_all_smoothed[train_end:]
        
        # Slice global precip anomaly for baseline comparison
        cond_anom_china = cond_anom[:, 0, 30:90, 70:140]
        cond_anom_china_tensor = torch.from_numpy(cond_anom_china[:, None])
        cond_anom_china_interp = F.interpolate(cond_anom_china_tensor, size=(120, 140), mode='bicubic').numpy()[:, 0]
        
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
    
    np.save(os.path.join(config.modelconfig['base_data_path'], "multi_lead_obs_results.npy"), obs_results)
    np.save(os.path.join(config.modelconfig['base_data_path'], "multi_lead_ec_precip_anom_results.npy"), ec_precip_anom_results)
    np.save(os.path.join(config.modelconfig['base_data_path'], "multi_lead_predict_results.npy"), predict_results)
    result_dates_str = np.array([d.strftime("%Y-%m-%d") for d in result_dates])
    np.save(os.path.join(config.modelconfig['base_data_path'], "multi_lead_dates.npy"), result_dates_str)
    
    print("\n============================ Multi-Lead Test Performance & AMS Summary ============================")
    print(f"  {'Lead':<6} | {'Selected Model':<18} | {'CV ACC':<8} | {'Test ACC':<10} | {'EC ACC':<10} | {'Model RMSE':<10} | {'EC RMSE':<10} | {'P-RMSE Decr':<12}")
    print("-" * 105)
    for lead in sorted(lead_avg_accs.keys()):
        print(f"  Lead-{lead:<1} | {selected_model_by_lead[lead]:<18} | {best_cv_score_by_lead[lead]:.5f}  | {lead_avg_accs[lead]:.6f} | {lead_avg_ec_accs[lead]:.6f} | {lead_avg_pcr_rmse[lead]:.6f} | {lead_avg_ec_rmse[lead]:.6f} | {lead_avg_prmse_overall[lead]:.2f}%")
    
if __name__ == "__main__":
    main()
