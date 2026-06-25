import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
from tqdm import tqdm

# Ensure project root is in sys.path
_curr_file = os.path.abspath(__file__)
_proj_root = os.path.dirname(os.path.dirname(_curr_file))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import config
from utils.observe_norm import DataNormalizer
import utils.utils as utils

def min_max_normalize_channels_4D(lr_data):
    """
    Global min-max normalization to [-1, 1] for all channels.
    """
    channel_mins = torch.amin(lr_data, dim=(0, 2, 3), keepdim=True)
    channel_maxs = torch.amax(lr_data, dim=(0, 2, 3), keepdim=True)
    normalized_data = 2 * (lr_data - channel_mins) / (channel_maxs - channel_mins + 1e-8) - 1
    return normalized_data

def main():
    print("Starting data preparation for Lead-1 (V3 - 11-channel Focused Features)...")
    
    # 1. Paths
    data_path = os.getenv("MODESV21_DATA_PATH") or "/Volumes/Game/MODESv21_ecmwf_seas51"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")
        
    observe_csv = os.path.join(_proj_root, "utils", "observe_data24.csv")
    if not os.path.exists(observe_csv):
        raise FileNotFoundError(f"Observe CSV not found: {observe_csv}")
    
    result = pd.read_csv(observe_csv)
    result['Long'] = result['Long']/100
    result['Lat'] = result['Lat']/100
    
    # Exclude months
    exclude_dates = ['2017-01-01', '2011-09-01', '2011-10-01']
    exclude_dates_dt = [pd.to_datetime(d) for d in exclude_dates]
    
    # Target dates range: 1994-01-01 to 2024-09-01
    target_dates = pd.date_range(start='1994-01-01', end='2024-09-01', freq='MS')
    
    cond_list = []
    target_obs_list = []
    aligned_target_months = []
    
    print("Reading NetCDF files and aligning with target dates...")
    for target_date in tqdm(target_dates, desc="Aligning dates"):
        if target_date in exclude_dates_dt:
            continue
            
        issue_date = target_date - pd.DateOffset(months=1)
        if issue_date in exclude_dates_dt:
            continue
            
        # Skip if previous month has no observations (before 1994-01-01)
        if issue_date < pd.to_datetime('1994-01-01'):
            continue
            
        ym_str = issue_date.strftime("%Y%m")
        fname = f"MODESv21_ecmwf_seas51_{ym_str}_monthly_em.nc"
        fpath = os.path.join(data_path, fname)
        
        if not os.path.exists(fpath):
            print(f"\nWarning: issue file {fname} not found, skipping target date {target_date.strftime('%Y-%m')}")
            continue
            
        try:
            ds = xr.open_dataset(fpath)
            # Select 10 channels and slice
            selected = ds[['tp', 'h200', 'h500', 'slp', 't2m', 't850', 'u200', 'u850', 'v200', 'v850']].sel(
                longitude=slice(70, 140), 
                latitude=slice(60, 0)
            )
            # Extract only lead 1 (forecast for target month t)
            cond_lead1 = selected.isel(time=1).to_array().to_numpy() # [10, 60, 70]
            
            # Load observation for target_date (current month t)
            obs_df = result[pd.to_datetime(result['time']) == target_date]
            if obs_df.empty:
                print(f"\nWarning: no observation for target date {target_date.strftime('%Y-%m')}, skipping")
                continue
            obs_grid = utils.gred_time_site_to_net(df=obs_df, to_xr=False, gred_var='anoma') # [120, 140]
            
            # Load observation for issue_date (previous month t-1)
            obs_prev_df = result[pd.to_datetime(result['time']) == issue_date]
            if obs_prev_df.empty:
                print(f"\nWarning: no observation for previous date {issue_date.strftime('%Y-%m')}, skipping")
                continue
            obs_prev_grid = utils.gred_time_site_to_net(df=obs_prev_df, to_xr=False, gred_var='anoma') # [120, 140]
            
            cond_list.append((cond_lead1, obs_prev_grid))
            target_obs_list.append(obs_grid)
            aligned_target_months.append(target_date.month)
            
        except Exception as e:
            print(f"\nError processing target date {target_date.strftime('%Y-%m')}: {e}")
            
    # Process condition and interpolation
    cond_lead1_list = []
    obs_prev_list = []
    for cond_lead1, obs_prev in cond_list:
        cond_lead1_list.append(cond_lead1)
        obs_prev_list.append(obs_prev)
        
    cond_lead1_dataset = np.stack(cond_lead1_list) # [N, 10, 60, 70]
    obs_prev_dataset = np.stack(obs_prev_list)   # [N, 1, 120, 140] (from gred_time_site_to_net)
    target_obs_dataset = np.stack(target_obs_list) # [N, 1, 120, 140]
    target_months = np.array(aligned_target_months)
    
    print(f"\nAligned samples: {cond_lead1_dataset.shape[0]}")
    
    # Bicubic interpolation of forecast condition to double resolution
    print("Interpolating condition to double resolution...")
    cond_tensor = torch.from_numpy(cond_lead1_dataset)
    lr_data = F.interpolate(cond_tensor, scale_factor=2.0, mode='bicubic').numpy() # [N, 10, 120, 140]
    print(f"Interpolated shape: {lr_data.shape}")
    
    # Convert precip forecast (channel 0) to mm/month
    print("Computing precipitation anomaly percentage for lead-1 forecast...")
    lr_data[:, 0] = lr_data[:, 0] * 31*24*60*60*1000
    
    # Compute climatology for lead 1
    anomaly_lead1 = np.zeros_like(lr_data[:, 0])
    for m in range(1, 13):
        idx = np.where(target_months == m)[0]
        if len(idx) == 0:
            continue
        clim = lr_data[idx, 0].mean(axis=0)
        anomaly_lead1[idx] = (lr_data[idx, 0] - clim) / (clim + 1e-6)
        
    # Normalize precipitation anomaly
    normalizer_lead1 = DataNormalizer(clip_sigma=3.0)
    normalizer_lead1.fit(anomaly_lead1)
    lr_data[:, 0] = normalizer_lead1.transform(anomaly_lead1)
    
    # Global min-max normalization for all 10 forecast channels
    print("Applying global min-max normalization for forecast channels...")
    lr_tensor = torch.from_numpy(lr_data)
    lr_norm = min_max_normalize_channels_4D(lr_tensor).numpy()
    
    # Normalize the previous observed precipitation using Z-score
    print("Normalizing previous observed precipitation anomaly channel...")
    normalizer_obs = DataNormalizer(clip_sigma=3.0)
    normalizer_obs.fit(obs_prev_dataset)
    obs_prev_norm = normalizer_obs.transform(obs_prev_dataset) # [N, 1, 120, 140]
    obs_prev_norm[np.isnan(obs_prev_norm)] = 0.0
    
    # Concatenate forecast and observed persistence to get 11 channels
    final_cond = np.concatenate([lr_norm, obs_prev_norm], axis=1) # [N, 11, 120, 140]
    
    # Save the aligned datasets
    save_dir = "/Users/huawei/workplace/unet3D"
    os.makedirs(save_dir, exist_ok=True)
    
    cond_save_path = os.path.join(save_dir, 'lr_data_v3_aligned.npy')
    target_save_path = os.path.join(save_dir, 'hr_data_v3_aligned.npy')
    months_save_path = os.path.join(save_dir, 'months_v3_aligned.npy')
    
    np.save(cond_save_path, final_cond)
    np.save(target_save_path, target_obs_dataset)
    np.save(months_save_path, target_months)
    
    print(f"Data preparation complete! Saved to:")
    print(f"  Condition: {cond_save_path} (shape={final_cond.shape})")
    print(f"  Target: {target_save_path} (shape={target_obs_dataset.shape})")

if __name__ == '__main__':
    main()
