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
    Global min-max normalization to [-1, 1] for channels 1-9.
    """
    channel_mins = torch.amin(lr_data, dim=(0, 2, 3), keepdim=True)
    channel_maxs = torch.amax(lr_data, dim=(0, 2, 3), keepdim=True)
    normalized_data = 2 * (lr_data - channel_mins) / (channel_maxs - channel_mins + 1e-8) - 1
    return normalized_data

def main():
    print("Starting data preparation for Lead-1...")
    
    # 1. Paths
    data_path = os.getenv("MODESV21_DATA_PATH") or "/Volumes/Game/MODESv21_ecmwf_seas51"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")
        
    # Load observed precipitation data
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
        # 1. Check if target date is excluded
        if target_date in exclude_dates_dt:
            continue
            
        # 2. Issue date is target_date - 1 month
        issue_date = target_date - pd.DateOffset(months=1)
        
        # 3. Check if issue date is in exclude_dates
        if issue_date in exclude_dates_dt:
            continue
            
        # 4. Read forecast NetCDF for issue_date
        ym_str = issue_date.strftime("%Y%m")
        fname = f"MODESv21_ecmwf_seas51_{ym_str}_monthly_em.nc"
        fpath = os.path.join(data_path, fname)
        
        if not os.path.exists(fpath):
            print(f"\nWarning: issue file {fname} not found, skipping target date {target_date.strftime('%Y-%m')}")
            continue
            
        try:
            ds = xr.open_dataset(fpath)
            # Select 10 channels, lead 1 (time=1), and slice
            selected = ds[['tp', 'h200', 'h500', 'slp', 't2m', 't850', 'u200', 'u850', 'v200', 'v850']].sel(
                longitude=slice(70, 140), 
                latitude=slice(60, 0)
            )
            # nth_cond shape: [10, 60, 70]
            nth_cond = selected.isel(time=1).to_array().to_numpy()
            
            # Load observation for target_date
            obs_df = result[pd.to_datetime(result['time']) == target_date]
            if obs_df.empty:
                print(f"\nWarning: no observation for target date {target_date.strftime('%Y-%m')}, skipping")
                continue
                
            obs_grid = utils.gred_time_site_to_net(df=obs_df, to_xr=False, gred_var='anoma')
            # obs_grid shape: [120, 140] (since gred_time_site_to_net converts to grid)
            
            cond_list.append(nth_cond)
            target_obs_list.append(obs_grid)
            aligned_target_months.append(target_date.month)
            
        except Exception as e:
            print(f"\nError processing target date {target_date.strftime('%Y-%m')}: {e}")
            
    cond_dataset = np.stack(cond_list) # [N, 10, 60, 70]
    target_obs_dataset = np.stack(target_obs_list) # [N, 120, 140]
    target_months = np.array(aligned_target_months)
    
    print(f"\nAligned samples: {cond_dataset.shape[0]}")
    print(f"Condition shape: {cond_dataset.shape}")
    print(f"Target shape: {target_obs_dataset.shape}")
    
    # Bicubic interpolation of forecast condition to double resolution
    print("Interpolating condition to double resolution...")
    cond_tensor = torch.from_numpy(cond_dataset)
    lr_data = F.interpolate(cond_tensor, scale_factor=2.0, mode='bicubic').numpy() # [N, 10, 120, 140]
    print(f"Interpolated shape: {lr_data.shape}")
    
    # Compute precip anomaly percentage for forecast (Channel 0)
    print("Computing precipitation anomaly percentage for lead-1 forecast...")
    lr_data[:, 0] = lr_data[:, 0] * 31*24*60*60*1000 # Convert to mm/month
    
    anomaly_lead1 = np.zeros_like(lr_data[:, 0])
    for m in range(1, 13):
        idx = np.where(target_months == m)[0]
        if len(idx) == 0:
            continue
        # Climatology of forecast at lead 1 for target month m
        clim = lr_data[idx, 0].mean(axis=0)
        anomaly_lead1[idx] = (lr_data[idx, 0] - clim) / (clim + 1e-6)
        
    # Z-score + 3-sigma clipping normalization
    print("Applying Z-score + 3-sigma normalization to precip anomalies...")
    normalizer = DataNormalizer(clip_sigma=3.0)
    normalizer.fit(anomaly_lead1)
    norm_anomaly = normalizer.transform(anomaly_lead1)
    lr_data[:, 0] = norm_anomaly
    
    # Global min-max normalization for other channels (1-9)
    print("Applying global min-max normalization for other channels...")
    lr_tensor = torch.from_numpy(lr_data)
    lr_norm = min_max_normalize_channels_4D(lr_tensor).numpy()
    
    # Save the aligned datasets under config.modelconfig['lr_path'] / unet3D
    save_dir = "/Users/huawei/workplace/unet3D"
    os.makedirs(save_dir, exist_ok=True)
    
    cond_save_path = os.path.join(save_dir, 'lr_data_lead1_aligned.npy')
    target_save_path = os.path.join(save_dir, 'hr_data_lead1_aligned.npy')
    months_save_path = os.path.join(save_dir, 'months_lead1_aligned.npy')
    
    np.save(cond_save_path, lr_norm)
    np.save(target_save_path, target_obs_dataset)
    np.save(months_save_path, target_months)
    
    print(f"Data preparation complete! Saved to:")
    print(f"  Condition: {cond_save_path}")
    print(f"  Target: {target_save_path}")
    print(f"  Target months: {months_save_path}")

if __name__ == '__main__':
    main()
