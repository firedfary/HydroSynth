import os
import glob
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from tqdm import tqdm


class DynamicalModelAligner:
    """
    Extracts, spatially aligns, and temporally disaggregates monthly dynamical climate
    model predictions (ECMWF, NCEP, BCC, UKMO) to 2371 station coordinates.
    """
    def __init__(
        self,
        coords: np.ndarray,
        model_dir: str = r"E:\DATA\model_data",
        cache_dir: str = "./cache"
    ):
        self.coords = coords          # (N, 2) [lat, lon]
        self.num_nodes = coords.shape[0]
        self.model_dir = model_dir
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.model_cache_file = os.path.join(self.cache_dir, f"model_features_N{self.num_nodes}.npz")

    @staticmethod
    def _extract_grid_to_stations(ds: xr.Dataset, var_name: str, coords: np.ndarray) -> np.ndarray:
        """
        Spatially extract NetCDF regular grid variable to station coordinates using bilinear interpolation.
        Returns: (Leads, Num_Stations)
        """
        # Identify lat/lon coordinate names in dataset
        lat_names = [c for c in ds.coords if c.lower() in ["lat", "latitude", "ylat"]]
        lon_names = [c for c in ds.coords if c.lower() in ["lon", "longitude", "xlon"]]
        if not lat_names or not lon_names:
            raise KeyError(f"Could not find lat/lon coordinates in dataset: {list(ds.coords)}")
            
        lat_coord = ds[lat_names[0]].values
        lon_coord = ds[lon_names[0]].values
        
        # Normalize longitudes to [0, 360) or match station lon range
        stn_lats = coords[:, 0]
        stn_lons = coords[:, 1]
        if lon_coord.max() > 180.0 and stn_lons.min() >= 0:
            stn_lons = np.where(stn_lons < 0, stn_lons + 360.0, stn_lons)
            
        data = ds[var_name].values
        # Data shape might be (lead, lat, lon) or (lead, member, lat, lon)
        if data.ndim == 4:  # ensemble dimension present
            data = np.nanmean(data, axis=1)  # ensemble mean
            
        leads = data.shape[0]
        stn_data = np.zeros((leads, len(coords)), dtype=np.float32)
        
        # RegularGridInterpolator expects sorted coordinates
        lat_sort_idx = np.argsort(lat_coord)
        lon_sort_idx = np.argsort(lon_coord)
        lat_sorted = lat_coord[lat_sort_idx]
        lon_sorted = lon_coord[lon_sort_idx]
        
        for l in range(leads):
            grid_slice = data[l][lat_sort_idx, :][:, lon_sort_idx]
            interp = RegularGridInterpolator(
                (lat_sorted, lon_sorted),
                grid_slice,
                method="linear",
                bounds_error=False,
                fill_value=None
            )
            pts = np.column_stack([stn_lats, stn_lons])
            stn_data[l] = interp(pts).astype(np.float32)
            
        return stn_data

    def build_or_load_station_features(
        self,
        target_dates: pd.DatetimeIndex,
        force_recompute: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Extract multi-model features across all target months and cache.
        Returns:
            dict containing:
                'macro_features': (T_months, Num_Stations, Num_Features)
                'daily_trend': (T_days, Num_Stations) Cubic spline base trend
                'dates_monthly': pd.DatetimeIndex
        """
        if not force_recompute and os.path.exists(self.model_cache_file):
            print(f"[DynamicalModelAligner] Loading cached model features from {self.model_cache_file}...")
            cached = np.load(self.model_cache_file, allow_pickle=True)
            return {k: cached[k] for k in cached.files}

        print(f"[DynamicalModelAligner] Scanning model files in {self.model_dir}...")
        nc_files = glob.glob(os.path.join(self.model_dir, "**", "*.nc"), recursive=True)
        print(f"[DynamicalModelAligner] Found {len(nc_files)} NetCDF model forecast files.")
        
        # Process monthly timeline
        monthly_dates = pd.date_range(start=target_dates.min(), end=target_dates.max(), freq="MS")
        num_months = len(monthly_dates)
        
        # Multi-model feature container (T_months, N_stations, Channels)
        # Channels: [ECMWF_precip_lead1, NCEP_precip_lead1, BCC_precip_lead1, UKMO_precip_lead1, MultiModel_Mean]
        num_channels = 5
        macro_features = np.zeros((num_months, self.num_nodes, num_channels), dtype=np.float32)
        
        # Build month lookup dictionary
        month_to_idx = {d.strftime("%Y%m"): idx for idx, d in enumerate(monthly_dates)}
        
        for nc_path in tqdm(nc_files, desc="Extracting dynamical model forecasts"):
            try:
                # Extract year-month from filename (e.g. *_200103* -> 200103)
                basename = os.path.basename(nc_path)
                match = pd.Series(basename).str.extract(r"(\d{6})")[0].values[0]
                if pd.isna(match) or match not in month_to_idx:
                    continue
                    
                m_idx = month_to_idx[match]
                ds = xr.open_dataset(nc_path)
                
                # Check variable name for precipitation
                precip_var = None
                for v in ["PRECT", "tp", "precip", "total_precipitation", "pr"]:
                    if v in ds.data_vars:
                        precip_var = v
                        break
                        
                if precip_var is not None:
                    stn_leads = self._extract_grid_to_stations(ds, precip_var, self.coords)
                    # Convert units if in m/s or m/day to mm/day
                    if stn_leads.max() < 0.1:  # m/s -> mm/day
                        stn_leads = stn_leads * 86400000.0
                    elif stn_leads.max() < 1.0:  # m/day -> mm/day
                        stn_leads = stn_leads * 1000.0
                        
                    lead1_val = stn_leads[0]  # lead 1
                    
                    # Distribute to appropriate channel
                    if "ecmwf" in nc_path.lower():
                        macro_features[m_idx, :, 0] = lead1_val
                    elif "ncep" in nc_path.lower() or "cfs" in nc_path.lower():
                        macro_features[m_idx, :, 1] = lead1_val
                    elif "bcc" in nc_path.lower() or "csm" in nc_path.lower():
                        macro_features[m_idx, :, 2] = lead1_val
                    elif "ukmo" in nc_path.lower() or "glosea" in nc_path.lower():
                        macro_features[m_idx, :, 3] = lead1_val
                ds.close()
            except Exception as e:
                pass  # Skip corrupted or non-standard format files
                
        # Multi-model ensemble mean (Channel 4)
        active_models = np.where(macro_features[:, :, :4] > 0, macro_features[:, :, :4], np.nan)
        macro_features[:, :, 4] = np.nan_to_num(np.nanmean(active_models, axis=-1), nan=0.0)

        # Generate continuous daily spline trend
        print("[DynamicalModelAligner] Interpolating monthly forecasts to daily baseline trend via Cubic Spline...")
        total_days = len(target_dates)
        daily_trend = np.zeros((total_days, self.num_nodes), dtype=np.float32)
        
        # Monthly representative x coordinates (middle of each month)
        m_day_indices = np.array([(d - target_dates.min()).days + 15 for d in monthly_dates])
        all_day_indices = np.arange(total_days)
        
        ensemble_monthly = macro_features[:, :, 4]  # (T_months, N)
        cs = CubicSpline(m_day_indices, ensemble_monthly, axis=0, bc_type="natural")
        daily_trend = np.maximum(cs(all_day_indices), 0.0).astype(np.float32)

        results = {
            "macro_features": macro_features,
            "daily_trend": daily_trend,
            "dates_monthly": monthly_dates.strftime("%Y-%m-%d").values
        }
        
        np.savez_compressed(self.model_cache_file, **results)
        print(f"[DynamicalModelAligner] Saved aligned features to {self.model_cache_file}.")
        return results
