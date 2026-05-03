import pandas as pd
import numpy as np
import xarray as xr
import os
from xMCA import xMCA

def get_coord_names(da):
    """
    Finds the actual names for latitude and longitude coordinates.
    """
    lat_name = None
    lon_name = None
    
    for c in da.coords:
        if 'lat' in c.lower():
            lat_name = c
        if 'lon' in c.lower() or 'lng' in c.lower():
            lon_name = c
            
    if lat_name is None or lon_name is None:
        lat_name = 'lat' if 'lat' in da.coords else ('latitude' if 'latitude' in da.coords else 'lat')
        lon_name = 'lon' if 'lon' in da.coords else ('longitude' if 'longitude' in da.coords else 'lon')
        
    return lat_name, lon_name

def load_observed_precip(csv_path):
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'])
    return df

def get_monthly_obs_matrix(df, month):
    df_m = df[df['Month'] == month].copy()
    df_m = df_m.sort_values(['Year', 'Stn_No'])
    pivot_df = df_m.pivot_table(index='Year', columns='Stn_No', values='anoma', aggfunc='mean')
    return pivot_df

def load_model_nc(base_path, year, month, lead=1):
    init_month = month - lead
    init_year = year
    if init_month <= 0:
        init_month += 12
        init_year -= 1
    
    file_name = f"MODESv21_ecmwf_seas51_{init_year}{str(init_month).zfill(2)}_monthly_em.nc"
    file_path = os.path.join(base_path, file_name)
    
    if not os.path.exists(file_path):
        return None
    
    ds = xr.open_dataset(file_path)
    if lead < ds.dims['time']:
        return ds.isel(time=lead)
    else:
        return None

def load_ersst_nc(base_path, year, month):
    file_name = f"ersst.v5.{year}{str(month).zfill(2)}.nc"
    file_path = os.path.join(base_path, file_name)
    if not os.path.exists(file_path):
        return None
    ds = xr.open_dataset(file_path)
    return ds

def apply_lat_weights(da):
    lat_name, _ = get_coord_names(da)
    if lat_name in da.coords:
        coslat = np.cos(np.deg2rad(da.coords[lat_name].values))
        wgts = np.sqrt(coslat)
        da_weighted = da * xr.DataArray(wgts, coords={lat_name: da.coords[lat_name]}, dims=[lat_name])
        return da_weighted
    return da

def eof_filter(da, n_modes=20, is_precip=False):
    """
    EOF filtering using xMCA.
    xMCA uses 'n' as the default dimension name for components.
    """
    time_mean = da.mean(dim='time')
    if is_precip:
        # Match 1.ipynb: percentage anomaly for precipitation
        ano = (da - time_mean) / time_mean.where(time_mean != 0, 1.0)
    else:
        ano = da - time_mean
    
    # Apply lat weights for gridded data
    if len(da.dims) >= 3:
        ano = apply_lat_weights(ano)
    
    try:
        # EOF is xMCA on the same field
        svd = xMCA(ano, ano)
        svd.solver()
        actual_n = min(n_modes, da.shape[0], da.shape[1])
        lp, rp = svd.patterns(n=actual_n)
        le, re = svd.expansionCoefs(n=actual_n)
        
        # xMCA default dimension for modes is 'n'
        # Reconstruction: Sum over 'n' dimension
        mode_dim = 'n'
        if mode_dim not in le.dims:
            # Fallback in case of different xMCA version/configuration
            mode_dim = le.dims[0]
            
        reconstructed = (le * lp).sum(dim=mode_dim)
        return reconstructed, le, lp
    except Exception as e:
        print(f"EOF Filter Error: {e}")
        return None, None, None

def perform_mca(da1, da2, n_components=20):
    try:
        svd = xMCA(da1, da2)
        svd.solver()
        actual_n = min(n_components, da1.shape[0])
        le, re = svd.expansionCoefs(n=actual_n)
        # Return time as the first dimension for XGBoost: (time, n)
        return le.values.T, re.values.T, svd
    except Exception as e:
        print(f"MCA Error: {e}")
        return None, None, None

def reconstruct_from_eof(pcs, eofs, pca_mean):
    return np.dot(pcs, eofs) + pca_mean
