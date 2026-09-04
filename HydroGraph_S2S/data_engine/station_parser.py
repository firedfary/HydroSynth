import os
import glob
import re
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm


class StationParser:
    """
    Parser for CMA 2371 national weather stations daily precipitation observations.
    Handles coordinate conversion, elevation, trace rain, missing values,
    and caches dense time-series matrices and spatial metadata.
    """
    def __init__(self, data_dir: str = r"E:\DATA\原始站点资料（有华南）", cache_dir: str = "./cache"):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.meta_cache_file = os.path.join(self.cache_dir, "station_meta.npz")
        self.daily_cache_file = os.path.join(self.cache_dir, "daily_precip.npz")
        self.pentad_cache_file = os.path.join(self.cache_dir, "pentad_precip.npz")
        
        self.station_ids = None
        self.coords = None        # (N, 2) [lat, lon] in decimal degrees
        self.elevations = None    # (N,) in meters
        self.daily_precip = None  # (T_days, N)
        self.dates = None         # (T_days,) pd.DatetimeIndex

    @staticmethod
    def _degmin_to_decimal(val: float) -> float:
        """Convert degree-minute format (e.g., 5328 -> 53 deg 28 min -> 53.4667 deg)."""
        val_int = int(round(val))
        deg = val_int // 100
        minute = val_int % 100
        return deg + minute / 60.0

    @staticmethod
    def _clean_precip_val(val: float) -> float:
        """
        Clean CMA daily precipitation values:
        - 32700: trace precipitation -> 0.05 mm
        - 30000 - 32766: missing / error / equipment maintenance -> np.nan
        - Normal values: 0.1 mm unit -> val * 0.1 mm
        """
        if pd.isna(val) or val >= 31000:
            if val == 32700:
                return 0.05
            return np.nan
        if val < 0:
            return 0.0
        return val * 0.1

    def parse_and_cache(self, force_recompute: bool = False) -> None:
        """Parse all monthly TXT files or load from cache."""
        if not force_recompute and os.path.exists(self.meta_cache_file) and os.path.exists(self.daily_cache_file):
            print(f"[StationParser] Loading cached station data from {self.cache_dir}...")
            meta_data = np.load(self.meta_cache_file, allow_pickle=True)
            self.station_ids = meta_data["station_ids"]
            self.coords = meta_data["coords"]
            self.elevations = meta_data["elevations"]
            
            daily_data = np.load(self.daily_cache_file, allow_pickle=True)
            self.daily_precip = daily_data["daily_precip"]
            self.dates = pd.to_datetime(daily_data["dates"])
            print(f"[StationParser] Loaded {len(self.station_ids)} stations, {len(self.dates)} days ({self.dates[0].date()} to {self.dates[-1].date()}).")
            return

        print(f"[StationParser] Parsing raw station TXT files from {self.data_dir}...")
        txt_files = sorted(glob.glob(os.path.join(self.data_dir, "SURF_CLI_CHN_MUL_DAY-PRE-*.TXT")))
        if not txt_files:
            raise FileNotFoundError(f"No station TXT files found in {self.data_dir}")

        station_info_dict = {}  # id -> {lat, lon, elev}
        daily_records = []      # list of dataframes
        
        for file_path in tqdm(txt_files, desc="Parsing monthly station files"):
            try:
                # Format: [Station_ID, Lat, Lon, Elev, Year, Month, Day, 20-8, 8-20, 20-20, Q1, Q2, Q3]
                df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="c", usecols=[0, 1, 2, 3, 4, 5, 6, 9])
                df.columns = ["station_id", "lat_raw", "lon_raw", "elev_raw", "year", "month", "day", "precip_raw"]
                
                # Update station meta
                unique_stns = df[["station_id", "lat_raw", "lon_raw", "elev_raw"]].drop_duplicates(subset=["station_id"])
                for _, row in unique_stns.iterrows():
                    sid = int(row["station_id"])
                    if sid not in station_info_dict:
                        station_info_dict[sid] = {
                            "lat": self._degmin_to_decimal(row["lat_raw"]),
                            "lon": self._degmin_to_decimal(row["lon_raw"]),
                            "elev": float(row["elev_raw"]) * 0.1  # to meters
                        }
                
                # Clean precipitation
                df["precip"] = df["precip_raw"].apply(self._clean_precip_val)
                # Create date column
                df["date"] = pd.to_datetime(df[["year", "month", "day"]])
                daily_records.append(df[["station_id", "date", "precip"]])
            except Exception as e:
                print(f"[StationParser] Error parsing {os.path.basename(file_path)}: {e}")

        all_records = pd.concat(daily_records, ignore_index=True)
        # Deduplicate to handle overlapping station files
        all_records = all_records.drop_duplicates(subset=["station_id", "date"], keep="last")
        
        # Sort stations consistently
        sorted_station_ids = np.array(sorted(station_info_dict.keys()), dtype=np.int64)
        num_stations = len(sorted_station_ids)
        coords = np.zeros((num_stations, 2), dtype=np.float32)
        elevations = np.zeros((num_stations,), dtype=np.float32)
        
        for idx, sid in enumerate(sorted_station_ids):
            coords[idx, 0] = station_info_dict[sid]["lat"]
            coords[idx, 1] = station_info_dict[sid]["lon"]
            elevations[idx] = station_info_dict[sid]["elev"]

        # Pivot table to dense matrix (Dates x Stations)
        pivot_df = all_records.pivot(index="date", columns="station_id", values="precip")
        
        # Reindex to complete continuous date range
        full_date_range = pd.date_range(start=pivot_df.index.min(), end=pivot_df.index.max(), freq="D")
        pivot_df = pivot_df.reindex(index=full_date_range, columns=sorted_station_ids)
        
        # Missing value imputation: fill remaining NaNs with 0.0 or linear interpolation
        daily_mat = pivot_df.values.astype(np.float32)
        daily_mat = np.nan_to_num(daily_mat, nan=0.0)

        self.station_ids = sorted_station_ids
        self.coords = coords
        self.elevations = elevations
        self.daily_precip = daily_mat
        self.dates = full_date_range

        # Save to cache
        np.savez_compressed(
            self.meta_cache_file,
            station_ids=self.station_ids,
            coords=self.coords,
            elevations=self.elevations
        )
        np.savez_compressed(
            self.daily_cache_file,
            daily_precip=self.daily_precip,
            dates=self.dates.strftime("%Y-%m-%d").values
        )
        print(f"[StationParser] Successfully parsed and cached {num_stations} stations and {len(self.dates)} days.")

    def get_pentad_precip(self) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        """
        Aggregate daily precipitation into 6 pentads per month:
        - Pentad 1: Days 1-5
        - Pentad 2: Days 6-10
        - Pentad 3: Days 11-15
        - Pentad 4: Days 16-20
        - Pentad 5: Days 21-25
        - Pentad 6: Days 26 to end of month
        Returns:
            pentad_mat: (Num_Pentads, Num_Stations) cumulative precipitation in mm
            pentad_dates: pd.DatetimeIndex of pentad starting dates
        """
        if os.path.exists(self.pentad_cache_file):
            cached = np.load(self.pentad_cache_file, allow_pickle=True)
            return cached["pentad_precip"], pd.to_datetime(cached["dates"])

        if self.daily_precip is None:
            self.parse_and_cache()

        df = pd.DataFrame(self.daily_precip, index=self.dates, columns=self.station_ids)
        
        def assign_pentad(dt):
            day = dt.day
            p_idx = min((day - 1) // 5 + 1, 6)
            return f"{dt.year}-{dt.month:02d}-P{p_idx}"

        pentad_labels = df.index.map(assign_pentad)
        pentad_df = df.groupby(pentad_labels, sort=False).sum()
        
        # Representative date for each pentad (1st, 6th, 11th, 16th, 21st, 26th)
        pentad_dates = []
        for label in pentad_df.index:
            y, m, p = label.split("-")
            p_num = int(p[1])
            d = (p_num - 1) * 5 + 1
            pentad_dates.append(datetime(int(y), int(m), d))
            
        pentad_mat = pentad_df.values.astype(np.float32)
        pentad_dates = pd.DatetimeIndex(pentad_dates)

        np.savez_compressed(
            self.pentad_cache_file,
            pentad_precip=pentad_mat,
            dates=pentad_dates.strftime("%Y-%m-%d").values
        )
        return pentad_mat, pentad_dates

    def compute_dynamic_features(self, daily_series: np.ndarray) -> np.ndarray:
        """
        Compute multi-channel dynamic features for ST-GNN:
        Channel 0: raw daily precipitation (mm)
        Channel 1: log-transformed precipitation log(1 + p)
        Channel 2: 7-day rolling cumulative precipitation
        Channel 3: Consecutive Dry Days (CDD) count
        Shape: (T, N, 4)
        """
        cache_file = os.path.join(self.cache_dir, "dynamic_features.npz")
        if os.path.exists(cache_file):
            return np.load(cache_file)["features"]

        print("[StationParser] Computing dynamic features (log-precip, rolling 7d, CDD)...")
        T, N = daily_series.shape
        features = np.zeros((T, N, 4), dtype=np.float32)
        
        features[:, :, 0] = daily_series
        features[:, :, 1] = np.log1p(np.maximum(daily_series, 0.0))
        
        # Fast vectorized rolling 7-day cumulative sum
        cs = np.cumsum(daily_series, axis=0)
        roll7 = np.zeros_like(daily_series)
        roll7[:7] = cs[:7]
        roll7[7:] = cs[7:] - cs[:-7]
        features[:, :, 2] = roll7
            
        # CDD (Consecutive Dry Days, p < 0.1 mm)
        cdd = np.zeros((T, N), dtype=np.float32)
        for t in range(1, T):
            dry_mask = daily_series[t] < 0.1
            cdd[t] = np.where(dry_mask, cdd[t-1] + 1.0, 0.0)
        features[:, :, 3] = cdd / 30.0  # Normalized
        
        np.savez(cache_file, features=features)
        print(f"[StationParser] Cached dynamic features to {cache_file}.")
        return features
