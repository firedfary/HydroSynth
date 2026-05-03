import os
os.environ["OMP_NUM_THREADS"] = "6"  # i5-10500 有 6 个物理核心
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
import xarray as xr

from utils_monthly import *
import warnings
warnings.filterwarnings('ignore')

# Configuration
OBS_CSV = r'E:\HydroSynth\utils\observe_data24.csv'
MODEL_PATH = r'D:\MODESv21_ecmwf_seas51'
ERSST_PATH = r'D:\ersst_data'
OUTPUT_DIR = r'E:\HydroSynth\XGBoost_Monthly\results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS = range(1994, 2024)
MONTHS = range(1, 13)
N_COMPONENTS = 20

# Spatial Ranges for Predictors (strictly from 1.ipynb)
VAR_RANGES = {
    'tp': {'lon': slice(70, 140), 'lat': slice(60, 0)},
    'slp': {'lon': slice(60, 180), 'lat': slice(60, 0)},
    'h500': {'lon': slice(70, 180), 'lat': slice(60, -30)},
    'sst': {'lon': slice(160, 210), 'lat': slice(10, -10)},
    'ersst': {'lon': slice(160, 210), 'lat': slice(-10, 10)}
}

# Load all observations once
print("Loading observed data...")
obs_df_all = load_observed_precip(OBS_CSV)

def train_and_predict_month(target_month):
    print(f"\nProcessing month {target_month}...")
    
    # 1. Prepare Target (Observed Precip)
    obs_matrix = get_monthly_obs_matrix(obs_df_all, target_month)
    available_years = sorted(list(set(YEARS) & set(obs_matrix.index)))
    obs_matrix = obs_matrix.loc[available_years]
    
    obs_matrix_filled = obs_matrix.fillna(obs_matrix.mean()).dropna(axis=1)
    obs_matrix_filled = obs_matrix_filled.loc[:, obs_matrix_filled.std() > 0]
    
    if obs_matrix_filled.empty:
        print(f"No valid observation data for month {target_month}")
        return

    obs_da = xr.DataArray(obs_matrix_filled.values, 
                         dims=['time', 'Stn_No'],
                         coords={'time': pd.to_datetime([f"{y}-01-01" for y in available_years]), 
                                 'Stn_No': obs_matrix_filled.columns})
    
    obs_filtered_da, obs_le, obs_lp = eof_filter(obs_da, n_modes=N_COMPONENTS, is_precip=False)
    
    Y = obs_le.values.T 
    num_pcs = Y.shape[1]
    
    # 2. Prepare Predictors
    model_vars = ['tp', 'slp', 'h500', 'sst']
    predictor_names = model_vars + ['ersst']
    predictor_das = {name: [] for name in predictor_names}
    valid_years = []
    
    for year in available_years:
        ds_m = load_model_nc(MODEL_PATH, year, target_month, lead=3)
        
        prev_month = target_month - 3
        prev_year = year
        if prev_month <= 0:
            prev_month += 12
            prev_year -= 1
        ds_e = load_ersst_nc(ERSST_PATH, prev_year, prev_month)
        
        if ds_m is not None and ds_e is not None:
            if all(v in ds_m for v in model_vars) and 'sst' in ds_e:
                try:
                    current_year_das = {}
                    skip = False
                    for var in model_vars:
                        da = ds_m[var]
                        # Use helper to get correct coordinate names
                        lat_key, lon_key = get_coord_names(da)
                        selector = {lon_key: VAR_RANGES[var]['lon'], lat_key: VAR_RANGES[var]['lat']}
                        sub = da.sel(**selector)
                        if sub.size == 0: skip = True; break
                        current_year_das[var] = sub
                    
                    if not skip:
                        da_e = ds_e['sst']
                        lat_key_e, lon_key_e = get_coord_names(da_e)
                        sub_e = da_e.sel({lon_key_e: VAR_RANGES['ersst']['lon'], lat_key_e: VAR_RANGES['ersst']['lat']})
                        if sub_e.size == 0: skip = True
                    
                    if not skip:
                        valid_years.append(year)
                        for var in model_vars: predictor_das[var].append(current_year_das[var])
                        predictor_das['ersst'].append(sub_e)
                except Exception as e:
                    # print(f"Skipping {year} due to {e}")
                    continue

    if len(valid_years) < 2:
        print(f"Not enough valid data (years: {len(valid_years)})")
        return

    time_coords = pd.to_datetime([f"{y}-01-01" for y in valid_years])
    idx_valid = [available_years.index(y) for y in valid_years]
    Y = Y[idx_valid]
    obs_filtered_da_valid = obs_filtered_da.isel(time=idx_valid)
    obs_filtered_da_valid['time'] = time_coords

    all_predictor_features = []
    
    print(f"--- Predictor MCA Coupling Analysis for month {target_month} ---")
    for name in predictor_names:
        da_field = xr.concat(predictor_das[name], dim='time')
        da_field['time'] = time_coords
        
        is_p = (name == 'tp')
        da_filtered, _, _ = eof_filter(da_field, n_modes=N_COMPONENTS, is_precip=is_p)
        
        le, re, svd = perform_mca(da_filtered, obs_filtered_da_valid, n_components=N_COMPONENTS)
        
        if le is None:
            print(f"MCA failed for {name}")
            return
            
        corr = np.corrcoef(le[:, 0], re[:, 0])[0, 1]
        print(f"Correlation (MCA {name}_EC1 vs Obs_EC1): {corr:.3f}")
        all_predictor_features.append(le)

    X = np.concatenate(all_predictor_features, axis=1)
    print("------------------------------------------------------------------")
    
    # 3. XGBoost Training with LOOCV
    n_years = len(valid_years)
    kf = KFold(n_splits=n_years)
    y_pred_all = np.zeros_like(Y)
    
    params = {
        'booster': 'gbtree',
        'objective': 'reg:squarederror',
        'max_depth': 3,
        'learning_rate': 0.1,
        'nthread': 4,
        'eval_metric': 'rmse'
    }
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
        for i in range(num_pcs):
            dtrain = xgb.DMatrix(X_train, label=Y_train[:, i])
            dval = xgb.DMatrix(X_val)
            bst = xgb.train(params, dtrain, num_boost_round=50)
            y_pred_all[val_idx, i] = bst.predict(dval)
            
    # 4. Reconstruction
    obs_mean = obs_da.mean(dim='time').values
    pred_anoma = np.dot(y_pred_all, obs_lp.values) + obs_mean
    
    # 5. Evaluation (ACC)
    accs = []
    # obs_matrix_filled contains years for available_years, we only want valid_years
    obs_valid = obs_matrix_filled.loc[valid_years].values
    for i in range(n_years):
        obs = obs_valid[i]
        pred = pred_anoma[i] 
        if np.std(obs) > 0 and np.std(pred) > 0:
            accs.append(np.corrcoef(obs, pred)[0, 1])
        else: accs.append(0)
    
    print(f"Mean ACC for month {target_month}: {np.mean(accs):.3f}")
    
    # Save results
    res_df = pd.DataFrame({'Year': valid_years, 'ACC': accs})
    res_df.to_csv(os.path.join(OUTPUT_DIR, f'acc_month_{target_month}.csv'), index=False)
    
    pred_df = pd.DataFrame(pred_anoma, index=valid_years, columns=obs_matrix_filled.columns)
    pred_df.to_csv(os.path.join(OUTPUT_DIR, f'pred_month_{target_month}.csv'))

if __name__ == "__main__":
    for m in MONTHS:
        try:
            train_and_predict_month(m)
        except Exception as e:
            print(f"Error processing month {m}: {e}")
            import traceback
            traceback.print_exc()
