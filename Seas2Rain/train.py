import os
import glob
import random
import re
import sys
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy.interpolate import griddata
from torch.utils.tensorboard import SummaryWriter
import tqdm
from dataclasses import dataclass

# Ensure repo parent is on sys.path so absolute imports like 'HydroSynth' work.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_repo_root = os.path.normpath(_repo_root)
_repo_parent = os.path.dirname(_repo_root)
if _repo_parent not in sys.path:
    sys.path.insert(0, _repo_parent)

from HydroSynth import config
import HydroSynth.utils.utils as utils

try:
    from HydroSynth.Seas2Rain import model
except Exception:
    import model  # fallback for direct script execution from HydroSynth/Seas2Rain

config.enable_auto_create_folders()

LEADS = 6
COND_VARS = ["h500", "slp"]
TARGET_HW = (60, 70)
MODES_CACHE_VERSION = "20260504_cond_h500_slp_ranges_v1"


def _cond_vars_signature() -> str:
    return "_".join(COND_VARS)
BATCH_KEYS = ("cond", "seas_anom", "ec_base", "obs_target", "obs_mask", "sst_hist")

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_init_dates() -> pd.DatetimeIndex:
    dates = pd.date_range(start="1994-01-01", end="2024-09-01", freq="MS")
    drop = {pd.Timestamp("2011-09-01"), pd.Timestamp("2011-10-01")}
    dates = pd.DatetimeIndex([d for d in dates if d not in drop])
    return dates

def split_indices_by_date(init_dates: pd.DatetimeIndex) -> Dict[str, np.ndarray]:
    train_end = pd.Timestamp("2015-12-01")
    val_start = pd.Timestamp("2016-01-01")
    val_end = pd.Timestamp("2019-12-01")
    test_start = pd.Timestamp("2020-01-01")

    train_idx = np.where(init_dates <= train_end)[0]
    val_idx = np.where((init_dates >= val_start) & (init_dates <= val_end))[0]
    test_idx = np.where(init_dates >= test_start)[0]

    return {"train": train_idx, "val": val_idx, "test": test_idx}

def _normalize_station_coords(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df["Long"].abs().max() > 180:
        df["Long"] = df["Long"] / 100.0
    if df["Lat"].abs().max() > 90:
        df["Lat"] = df["Lat"] / 100.0
    return df

def _month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)

def _parse_date_from_path(path: str) -> Optional[pd.Timestamp]:
    name = os.path.basename(path)
    match = re.search(r"_(\d{6})_", name)
    if not match:
        return None
    yyyymm = match.group(1)
    year = int(yyyymm[:4])
    month = int(yyyymm[4:])
    return pd.Timestamp(year=year, month=month, day=1)

def _get_lat_lon(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    if "latitude" in ds.coords:
        lat = ds.coords["latitude"].to_numpy()
    elif "lat" in ds.coords:
        lat = ds.coords["lat"].to_numpy()
    else:
        raise KeyError("Latitude coordinate not found in dataset.")

    if "longitude" in ds.coords:
        lon = ds.coords["longitude"].to_numpy()
    elif "lon" in ds.coords:
        lon = ds.coords["lon"].to_numpy()
    else:
        raise KeyError("Longitude coordinate not found in dataset.")

    return lat, lon

def _build_target_grid(lat: np.ndarray, lon: np.ndarray, target_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    if len(lat) == target_hw[0] and len(lon) == target_hw[1]:
        return lat.astype(np.float32), lon.astype(np.float32)

    lat_min, lat_max = float(lat.min()), float(lat.max())
    lon_min, lon_max = float(lon.min()), float(lon.max())
    if lat[0] > lat[-1]:
        grid_lats = np.linspace(lat_max, lat_min, target_hw[0], dtype=np.float32)
    else:
        grid_lats = np.linspace(lat_min, lat_max, target_hw[0], dtype=np.float32)
    grid_lons = np.linspace(lon_min, lon_max, target_hw[1], dtype=np.float32)
    return grid_lats, grid_lons

def read_modes_data(
    cache_path: str,
    target_hw: Tuple[int, int] = TARGET_HW,
    seas_nc_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        tp_raw = cached["tp_raw"].astype(np.float32)
        cond_raw = cached["cond_raw"].astype(np.float32)
        init_dates = pd.to_datetime(cached["init_dates"].astype(str))
        grid_lats = cached["grid_lats"].astype(np.float32)
        grid_lons = cached["grid_lons"].astype(np.float32)
        cached_cond_vars = cached["cond_vars"].tolist() if "cond_vars" in cached.files else None
        cached_version = str(cached["cache_version"]) if "cache_version" in cached.files else None

        cache_valid = (
            cond_raw.ndim == 5
            and cond_raw.shape[2] == len(COND_VARS)
            and tp_raw.shape[-2:] == target_hw
            and tuple(grid_lats.shape) == (target_hw[0],)
            and tuple(grid_lons.shape) == (target_hw[1],)
            and cached_cond_vars == COND_VARS
            and cached_version == MODES_CACHE_VERSION
        )
        if cache_valid:
            return tp_raw, cond_raw, pd.DatetimeIndex(init_dates), grid_lats, grid_lons

        print(
            f"Ignore stale MODES cache: {cache_path}. "
            f"cond_channels={cond_raw.shape[2] if cond_raw.ndim == 5 else 'invalid'}, "
            f"expected={len(COND_VARS)}, cache_version={cached_version}, "
            f"cond_vars={cached_cond_vars}"
        )

    file_list = utils.read_nc_to_npy(199401, 202409, data_path=seas_nc_path or "D:\\MODESv21_ecmwf_seas51")# 读取SERS5模式数据

    tp_list: List[np.ndarray] = []
    cond_list: List[np.ndarray] = []
    date_list: List[pd.Timestamp] = []
    grid_lats: Optional[np.ndarray] = None
    grid_lons: Optional[np.ndarray] = None

    forbidden = {pd.Timestamp("2011-09-01"), pd.Timestamp("2011-10-01")}

    for f in tqdm.tqdm(file_list, desc="Read MODESv21"):
        date = _parse_date_from_path(f)
        if date is None or date in forbidden:
            continue
        try:
            with xr.open_dataset(f) as ds:
                # Coordinate names standardization
                if "longitude" in ds.coords: ds = ds.rename({"longitude": "lon", "latitude": "lat"})

                # Precipitation range: lon=70-140, lat=60-0
                tp_ds = ds[["tp"]].sel(lon=slice(70, 140), lat=slice(60, 0))
                
                lat, lon = _get_lat_lon(tp_ds)
                if grid_lats is None or grid_lons is None:
                    grid_lats, grid_lons = _build_target_grid(lat, lon, target_hw)

                tp_arr = tp_ds.interp(lat=grid_lats, lon=grid_lons, method="linear").to_array().to_numpy().astype(np.float32) # [1, L, H, W]

                # Condition variables ranges
                # h500: lon=70-180, lat=60--30
                # slp: lon=60-180, lat=60-0
                h500_ds = ds[["h500"]].sel(lon=slice(70, 180), lat=slice(60, -30))
                slp_ds = ds[["slp"]].sel(lon=slice(60, 180), lat=slice(60, 0))

                h500_interp = h500_ds.interp(lat=grid_lats, lon=grid_lons, method="linear")["h500"].to_numpy()
                slp_interp = slp_ds.interp(lat=grid_lats, lon=grid_lons, method="linear")["slp"].to_numpy()

                cond_arr = np.stack([h500_interp, slp_interp], axis=0).astype(np.float32) # [2, L, H, W]

                if tp_arr.shape[1] != LEADS:
                    raise ValueError(f"tp lead mismatch: expected {LEADS}, got {tp_arr.shape[1]}")
                if cond_arr.shape[1] != LEADS:
                    raise ValueError(f"cond lead mismatch: expected {LEADS}, got {cond_arr.shape[1]}")

                tp_arr = np.nan_to_num(tp_arr, nan=0.0, posinf=0.0, neginf=0.0)
                cond_arr = np.nan_to_num(cond_arr, nan=0.0, posinf=0.0, neginf=0.0)

                tp_list.append(tp_arr.squeeze(0))  # [L, H, W]
                cond_list.append(cond_arr)  # [2, L, H, W]
                date_list.append(date)
        except Exception as e:
            print(f"Skip file {f}: {e}")
            continue

    if len(tp_list) == 0:
        raise RuntimeError("No valid MODESv21 files found.")

    tp_raw = np.stack(tp_list, axis=0)  # [N, L, H, W]
    cond_raw = np.stack(cond_list, axis=0)  # [N, 2, L, H, W]
    cond_raw = np.transpose(cond_raw, (0, 2, 1, 3, 4))  # [N, L, 2, H, W]

    init_dates = pd.DatetimeIndex(date_list)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        tp_raw=tp_raw.astype(np.float32),
        cond_raw=cond_raw.astype(np.float32),
        init_dates=np.array([d.strftime("%Y-%m-%d") for d in init_dates]),
        grid_lats=grid_lats.astype(np.float32),
        grid_lons=grid_lons.astype(np.float32),
        cond_vars=np.array(COND_VARS, dtype="<U32"),
        cache_version=np.array(MODES_CACHE_VERSION),
    )
    return tp_raw.astype(np.float32), cond_raw.astype(np.float32), init_dates, grid_lats, grid_lons

def _parse_ersst_date(path: str) -> Optional[pd.Timestamp]:
    name = os.path.basename(path)
    match = re.search(r"(\d{6})", name)
    if not match:
        return None
    yyyymm = match.group(1)
    year = int(yyyymm[:4])
    month = int(yyyymm[4:])
    return pd.Timestamp(year=year, month=month, day=1)

def read_ersst_data(
    ersst_dir: str,
    cache_path: str,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        sst = cached["ssta"].astype(np.float32)
        dates = pd.to_datetime(cached["dates"].astype(str))
        return sst, pd.DatetimeIndex(dates)

    files = sorted(glob.glob(os.path.join(ersst_dir, "ersst.v5.*.nc")))
    if len(files) == 0:
        raise FileNotFoundError(f"No ERSST files found under {ersst_dir}")

    sst_list: List[np.ndarray] = []
    date_list: List[pd.Timestamp] = []
    ref_shape: Optional[Tuple[int, int]] = None

    for f in tqdm.tqdm(files, desc="Read ERSST ssta"):
        date = _parse_ersst_date(f)
        if date is None:
            continue
        try:
            with xr.open_dataset(f) as ds:
                if "ssta" not in ds:
                    raise KeyError("ssta not found in ERSST file")
                
                # SST range: lon=160-210, lat=10--10
                if "longitude" in ds.coords: ds = ds.rename({"longitude": "lon", "latitude": "lat"})
                ds_ssta = ds.sel(lon=slice(160, 210), lat=slice(10, -10))

                arr = np.asarray(ds_ssta["ssta"].to_numpy())
                if arr.ndim == 4:
                    # Expected (time=1, zlev=1, lat, lon)
                    if arr.shape[0] != 1 or arr.shape[1] != 1:
                        raise ValueError(f"ssta shape not supported: {arr.shape}")
                    arr = arr[0, 0]
                elif arr.ndim == 3:
                    # Expected (time=1, lat, lon)
                    if arr.shape[0] != 1:
                        raise ValueError(f"ssta shape not supported: {arr.shape}")
                    arr = arr[0]
                elif arr.ndim != 2:
                    raise ValueError(f"ssta shape not supported: {arr.shape}")
                if ref_shape is None:
                    ref_shape = arr.shape
                elif arr.shape != ref_shape:
                    raise ValueError(f"Inconsistent ssta shape: {arr.shape} vs {ref_shape}")

                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                sst_list.append(arr)
                date_list.append(date)
        except Exception as e:
            print(f"Skip ERSST file {f}: {e}")
            continue

    if len(sst_list) == 0:
        raise RuntimeError("No valid ERSST ssta data found.")

    sst = np.stack(sst_list, axis=0)  # [T, H, W]
    dates = pd.DatetimeIndex(date_list)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        ssta=sst.astype(np.float32),
        dates=np.array([d.strftime("%Y-%m-%d") for d in dates]),
    )
    return sst.astype(np.float32), dates

def build_sst_history(
    init_dates: pd.DatetimeIndex,
    sst: np.ndarray,
    sst_dates: pd.DatetimeIndex,
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    sst_index = {_month_start(d): i for i, d in enumerate(sst_dates)}
    keep: List[int] = []
    hist: List[np.ndarray] = []

    for i, d in enumerate(init_dates):
        seq: List[np.ndarray] = []
        ok = True
        for m in range(window, 0, -1):
            md = _month_start(d - pd.DateOffset(months=m))
            idx = sst_index.get(md, None)
            if idx is None:
                ok = False
                break
            seq.append(sst[idx])
        if ok:
            keep.append(i)
            hist.append(np.stack(seq, axis=0))

    if len(hist) == 0:
        raise RuntimeError("No init_dates have complete SST history window.")

    return np.stack(hist, axis=0).astype(np.float32), np.asarray(keep, dtype=np.int64)

def normalize_sst_hist(sst_hist: np.ndarray, train_idx: np.ndarray) -> Tuple[np.ndarray, float, float]:
    train_vals = sst_hist[train_idx]
    mean = float(train_vals.mean())
    std = float(train_vals.std())
    std = max(std, 1e-6)
    sst_norm = (sst_hist - mean) / std
    return sst_norm.astype(np.float32), mean, std

def calc_precip_percent_anomaly(
    tp_tensor: torch.Tensor,
    init_dates: pd.DatetimeIndex,
    eps: float = 1e-6,
    lead_dependent: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # tp_tensor: [T, L, H, W]
    T, L, H, W = tp_tensor.shape
    if len(init_dates) != T:
        raise ValueError(f"init_dates length mismatch: {len(init_dates)} vs {T}")

    target_month_idx = torch.empty((T, L), dtype=torch.long)
    for t, d in enumerate(init_dates):
        for l in range(L):
            target = d + pd.DateOffset(months=l)
            target_month_idx[t, l] = target.month

    pa = torch.zeros_like(tp_tensor)

    if lead_dependent:
        climatology = torch.zeros((L, 12, H, W), dtype=tp_tensor.dtype, device=tp_tensor.device)
        for l in range(L):
            for m in range(1, 13):
                idx_t = (target_month_idx[:, l] == m).nonzero(as_tuple=True)[0]
                if idx_t.numel() == 0:
                    continue
                clim = tp_tensor[idx_t, l].mean(dim=0)
                climatology[l, m - 1] = clim
                pa[idx_t, l] = (tp_tensor[idx_t, l] - clim) / (clim + eps) * 100.0
    else:
        climatology = torch.zeros((12, H, W), dtype=tp_tensor.dtype, device=tp_tensor.device)
        for m in range(1, 13):
            mask_m = (target_month_idx == m)
            vals = tp_tensor[mask_m]
            if vals.numel() == 0:
                continue
            clim = vals.mean(dim=0)
            climatology[m - 1] = clim
            pa[mask_m] = (vals - clim) / (clim + eps) * 100.0

    return pa, climatology

def compute_cond_stats(cond_raw: np.ndarray, train_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # cond_raw: [N, L, C, H, W]
    lead_dim = cond_raw.shape[1]
    cond_dim = cond_raw.shape[2]
    sum_x = np.zeros((lead_dim, cond_dim), dtype=np.float64)
    sum_x2 = np.zeros((lead_dim, cond_dim), dtype=np.float64)
    count = np.zeros((lead_dim, cond_dim), dtype=np.float64)

    for idx in tqdm.tqdm(train_idx, desc="Cond stats"):
        x = cond_raw[int(idx)].astype(np.float32)  # [L, 7, H, W]
        valid = np.isfinite(x) & (x > -9000.0)
        xv = np.where(valid, x, 0.0)
        sum_x += xv.sum(axis=(2, 3))
        sum_x2 += (xv * xv).sum(axis=(2, 3))
        count += valid.sum(axis=(2, 3))

    count = np.maximum(count, 1.0)
    mean = sum_x / count
    var = sum_x2 / count - mean * mean
    var = np.maximum(var, 1e-8)
    std = np.sqrt(var)
    std = np.maximum(std, 1e-4)
    return mean.astype(np.float32), std.astype(np.float32)

def normalize_cond(cond_raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    # cond_raw: [N, L, C, H, W], mean/std: [L, C]
    x = (cond_raw - mean[None, :, :, None, None]) / std[None, :, :, None, None]
    x = np.clip(x, -6.0, 6.0)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32)

def prepare_observe_data(
    csv_path: str,
    cache_path: str,
    grid_lons: np.ndarray,
    grid_lats: np.ndarray,
    start_date: str = "1994-01-01",
    end_date: str = "2024-12-01",
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=False)
        obs_grid = cached["obs_grid"].astype(np.float32)
        obs_mask = cached["obs_mask"].astype(np.float32)
        obs_dates = pd.to_datetime(cached["obs_dates"].astype(str))
        return obs_grid, obs_mask, pd.DatetimeIndex(obs_dates)

    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        raise ValueError(f"Column 'time' not found in {csv_path}")
    if "anoma" not in df.columns:
        raise ValueError(f"Column 'anoma' not found in {csv_path}")
    if "Long" not in df.columns or "Lat" not in df.columns:
        raise ValueError(f"Columns 'Long' and 'Lat' are required in {csv_path}")

    df["time"] = pd.to_datetime(df["time"])
    df["time"] = df["time"].apply(_month_start)
    df = _normalize_station_coords(df)

    all_months = pd.date_range(start=start_date, end=end_date, freq="MS")
    grid_lon2d, grid_lat2d = np.meshgrid(grid_lons, grid_lats)

    grid_all = []
    mask_all = []
    for cur_month in tqdm.tqdm(all_months, desc="Interpolate observations"):
        cur = df[df["time"] == cur_month]
        if len(cur) < 3:
            grid_all.append(np.zeros(TARGET_HW, dtype=np.float32))
            mask_all.append(np.zeros(TARGET_HW, dtype=np.float32))
            continue

        points = cur[["Long", "Lat"]].to_numpy(dtype=np.float32)
        values = cur["anoma"].to_numpy(dtype=np.float32)

        try:
            linear = griddata(points, values, (grid_lon2d, grid_lat2d), method="linear")
        except Exception:
            linear = np.full(TARGET_HW, np.nan, dtype=np.float32)

        try:
            nearest = griddata(points, values, (grid_lon2d, grid_lat2d), method="nearest")
        except Exception:
            nearest = np.zeros(TARGET_HW, dtype=np.float32)

        valid = np.isfinite(linear)
        merged = np.where(valid, linear, nearest)
        merged = np.nan_to_num(merged, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        mask = valid.astype(np.float32)

        grid_all.append(merged)
        mask_all.append(mask)

    obs_grid = np.stack(grid_all, axis=0).astype(np.float32)
    obs_mask = np.stack(mask_all, axis=0).astype(np.float32)
    obs_dates = pd.DatetimeIndex(all_months)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        obs_grid=obs_grid,
        obs_mask=obs_mask,
        obs_dates=np.array([d.strftime("%Y-%m-%d") for d in obs_dates]),
    )
    return obs_grid, obs_mask, obs_dates

def build_obs_targets(
    init_dates: pd.DatetimeIndex,
    obs_grid: np.ndarray,
    obs_mask: np.ndarray,
    obs_dates: pd.DatetimeIndex,
    leads: int = LEADS,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(init_dates)
    target = np.zeros((n, leads, TARGET_HW[0], TARGET_HW[1]), dtype=np.float32)
    mask = np.zeros((n, leads, TARGET_HW[0], TARGET_HW[1]), dtype=np.float32)

    obs_index = {_month_start(d): i for i, d in enumerate(obs_dates)}
    for i, init_d in enumerate(init_dates):
        for l in range(leads):
            tgt_d = _month_start(init_d + pd.DateOffset(months=l))
            j = obs_index.get(tgt_d, None)
            if j is None:
                continue
            target[i, l] = obs_grid[j]
            mask[i, l] = obs_mask[j]
    return target, mask

def prepare_data() -> Dict[str, np.ndarray]:
    cfg = config.modelconfig
    cache_dir = cfg.get("seas2rain_cache_dir", os.path.join(cfg["lr_path"], "seas2rain_cache"))
    os.makedirs(cache_dir, exist_ok=True)

    cond_cache_tag = _cond_vars_signature()
    modes_cache = os.path.join(cache_dir, f"modes_tp_cond_60x70_{cond_cache_tag}.npz")
    seas_nc_path = cfg.get("seas_nc_path")
    tp_unit_scale = float(cfg.get("tp_unit_scale", 1.0))

    tp_raw, cond_raw, init_dates, grid_lats, grid_lons = read_modes_data(
        cache_path=modes_cache,
        target_hw=TARGET_HW,
        seas_nc_path=seas_nc_path,
    )

    init_dates_full = build_init_dates()
    allowed = set(init_dates_full)
    keep_mask = np.array([d in allowed for d in init_dates], dtype=bool)
    tp_raw = tp_raw[keep_mask]
    cond_raw = cond_raw[keep_mask]
    init_dates = pd.DatetimeIndex([d for d in init_dates if d in allowed])

    ersst_dir = cfg.get("ersst_dir", r"D:\ersst_data")# 海温数据，ERSST5.0
    sst_window = int(cfg.get("sst_window", 12))
    ersst_cache = os.path.join(cache_dir, "ersst_ssta_cache.npz")
    sst_raw, sst_dates = read_ersst_data(ersst_dir=ersst_dir, cache_path=ersst_cache)
    sst_hist, keep_idx = build_sst_history(init_dates, sst_raw, sst_dates, sst_window)

    tp_raw = tp_raw[keep_idx]
    cond_raw = cond_raw[keep_idx]
    init_dates = pd.DatetimeIndex([init_dates[i] for i in keep_idx])

    if len(init_dates) < 300:
        raise ValueError(f"Aligned sample count too small: {len(init_dates)}")

    splits = split_indices_by_date(init_dates)

    anom_cache = os.path.join(cache_dir, f"seas_anom_leaddep_n{len(init_dates)}.npy")
    if os.path.exists(anom_cache):
        seas_anom = np.load(anom_cache).astype(np.float32)
        if seas_anom.shape != tp_raw.shape:
            seas_anom = None
    else:
        seas_anom = None

    if seas_anom is None:
        tp_tensor = torch.from_numpy(tp_raw * tp_unit_scale)
        seas_anom_t, _ = calc_precip_percent_anomaly(tp_tensor, init_dates, lead_dependent=True)
        seas_anom = seas_anom_t.cpu().numpy().astype(np.float32)
        np.save(anom_cache, seas_anom)

    ec_base = seas_anom[:, :, None, :, :].astype(np.float32)
    seas_anom = seas_anom[:, :, None, :, :].astype(np.float32)

    cond_mean, cond_std = compute_cond_stats(cond_raw=cond_raw, train_idx=splits["train"])
    cond_norm_cache = os.path.join(
        cache_dir,
        f"cond_norm_{cond_cache_tag}_{MODES_CACHE_VERSION}_n{len(init_dates)}.npy",
    )
    if os.path.exists(cond_norm_cache):
        cond_norm = np.load(cond_norm_cache).astype(np.float32)
        if cond_norm.shape != cond_raw.shape:
            cond_norm = None
    else:
        cond_norm = None

    if cond_norm is None:
        cond_norm = normalize_cond(cond_raw, cond_mean, cond_std)
        np.save(cond_norm_cache, cond_norm)

    sst_hist_cache = os.path.join(cache_dir, f"ersst_ssta_hist_w{sst_window}_n{len(init_dates)}.npz")
    sst_mean = None
    sst_std = None
    if os.path.exists(sst_hist_cache):
        cached = np.load(sst_hist_cache, allow_pickle=True)
        cached_dates = cached["init_dates"] if "init_dates" in cached else None
        if cached_dates is not None:
            cached_dates = pd.to_datetime(cached_dates.astype(str))
        if cached_dates is not None and len(cached_dates) == len(init_dates) and all(cached_dates == init_dates):
            sst_hist = cached["sst_hist"].astype(np.float32)
            sst_mean = float(cached["sst_mean"])
            sst_std = float(cached["sst_std"])
    if sst_mean is None or sst_std is None:
        sst_hist, sst_mean, sst_std = normalize_sst_hist(sst_hist, splits["train"])
        np.savez_compressed(
            sst_hist_cache,
            sst_hist=sst_hist.astype(np.float32),
            sst_mean=np.array(sst_mean, dtype=np.float32),
            sst_std=np.array(sst_std, dtype=np.float32),
            init_dates=np.array([d.strftime("%Y-%m-%d") for d in init_dates]),
        )

    obs_csv_path = cfg.get(
        "observe_csv_path",
        os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "utils", "observe_data24.csv")),#中国站点观测降水数据
    )
    obs_cache_path = os.path.join(cache_dir, "observe_grid_cache_199401_202412_60x70.npz")
    obs_grid, obs_month_mask, obs_dates = prepare_observe_data(
        csv_path=obs_csv_path,
        cache_path=obs_cache_path,
        grid_lons=grid_lons,
        grid_lats=grid_lats,
        start_date="1994-01-01",
        end_date="2024-12-01",
    )

    obs_target, obs_mask = build_obs_targets(
        init_dates=init_dates,
        obs_grid=obs_grid,
        obs_mask=obs_month_mask,
        obs_dates=obs_dates,
        leads=LEADS,
    )

    obs_target = obs_target[:, :, None, :, :].astype(np.float32)
    obs_mask = obs_mask[:, :, None, :, :].astype(np.float32)

    data = {
        "cond": cond_norm.astype(np.float32),
        "seas_anom": seas_anom.astype(np.float32),
        "ec_base": ec_base.astype(np.float32),
        "sst_hist": sst_hist.astype(np.float32),
        "sst_mean": float(sst_mean),
        "sst_std": float(sst_std),
        "obs_target": obs_target.astype(np.float32),
        "obs_mask": obs_mask.astype(np.float32),
        "init_dates": np.array([d.strftime("%Y-%m-%d") for d in init_dates]),
        "split_indices": splits,
        "cond_mean": cond_mean,
        "cond_std": cond_std,
    }
    validate_data_bundle(data)
    return data

def validate_data_bundle(data: Dict[str, np.ndarray]) -> None:
    n = len(data["init_dates"])
    for k in ("cond", "seas_anom", "ec_base", "sst_hist", "obs_target", "obs_mask"):
        if data[k].shape[0] != n:
            raise ValueError(f"{k} length mismatch: {data[k].shape[0]} vs {n}")

    if data["cond"].ndim != 5 or data["cond"].shape[1] != LEADS or data["cond"].shape[2] != len(COND_VARS):
        raise ValueError(f"cond shape invalid: {data['cond'].shape}")
    if data["sst_hist"].ndim != 4:
        raise ValueError(f"sst_hist shape invalid: {data['sst_hist'].shape}")
    if tuple(data["seas_anom"].shape[1:]) != (LEADS, 1, TARGET_HW[0], TARGET_HW[1]):
        raise ValueError(f"seas_anom shape invalid: {data['seas_anom'].shape}")
    if tuple(data["ec_base"].shape[1:]) != (LEADS, 1, TARGET_HW[0], TARGET_HW[1]):
        raise ValueError(f"ec_base shape invalid: {data['ec_base'].shape}")
    if tuple(data["obs_target"].shape[1:]) != (LEADS, 1, TARGET_HW[0], TARGET_HW[1]):
        raise ValueError(f"obs_target shape invalid: {data['obs_target'].shape}")
    if tuple(data["obs_mask"].shape[1:]) != (LEADS, 1, TARGET_HW[0], TARGET_HW[1]):
        raise ValueError(f"obs_mask shape invalid: {data['obs_mask'].shape}")

@dataclass
class TensorBatchStore:
    tensors: Dict[str, torch.Tensor]
    split_indices: Dict[str, torch.Tensor]
    batch_size: int
    storage_device: torch.device
    train_device: torch.device

    def split_size(self, split: str) -> int:
        return int(self.split_indices[split].shape[0])


def _resolve_data_storage_device(train_device: torch.device) -> torch.device:
    storage_mode = str(config.modelconfig.get("dataset_storage_device", "auto")).lower()
    if storage_mode in ("auto", "device", "train", "gpu", "cuda"):
        return train_device if train_device.type == "cuda" else torch.device("cpu")
    if storage_mode == "cpu":
        return torch.device("cpu")
    return torch.device(storage_mode)


def _materialize_batch_tensors(
    data: Dict[str, np.ndarray],
    storage_device: torch.device,
    pin_memory: bool,
) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, torch.Tensor] = {}
    for key in BATCH_KEYS:
        tensor = torch.from_numpy(np.ascontiguousarray(data[key]))
        if storage_device.type == "cpu":
            if pin_memory:
                tensor = tensor.pin_memory()
        else:
            tensor = tensor.to(storage_device)
        tensors[key] = tensor
    return tensors


def build_batch_store(data: Dict[str, np.ndarray], train_device: torch.device) -> TensorBatchStore:
    batch_size = int(config.modelconfig.get("batch_size", 8))
    storage_device = _resolve_data_storage_device(train_device)
    pin_memory = storage_device.type == "cpu" and train_device.type == "cuda"

    try:
        tensors = _materialize_batch_tensors(data, storage_device=storage_device, pin_memory=pin_memory)
        split_indices = {
            split: torch.as_tensor(indices, dtype=torch.long, device=storage_device)
            for split, indices in data["split_indices"].items()
        }
    except RuntimeError as e:
        if storage_device.type != "cuda" or "out of memory" not in str(e).lower():
            raise
        print("Preloading full dataset to CUDA failed; falling back to pinned CPU tensors.")
        torch.cuda.empty_cache()
        storage_device = torch.device("cpu")
        tensors = _materialize_batch_tensors(data, storage_device=storage_device, pin_memory=train_device.type == "cuda")
        split_indices = {
            split: torch.as_tensor(indices, dtype=torch.long, device=storage_device)
            for split, indices in data["split_indices"].items()
        }

    return TensorBatchStore(
        tensors=tensors,
        split_indices=split_indices,
        batch_size=batch_size,
        storage_device=storage_device,
        train_device=train_device,
    )


def count_split_batches(batch_store: TensorBatchStore, split: str) -> int:
    size = batch_store.split_size(split)
    return (size + batch_store.batch_size - 1) // batch_store.batch_size


def iter_split_batches(
    batch_store: TensorBatchStore,
    split: str,
    shuffle: bool = False,
) -> Iterator[Dict[str, torch.Tensor]]:
    indices = batch_store.split_indices[split]
    if shuffle and indices.numel() > 1:
        perm = torch.randperm(indices.shape[0], device=indices.device)
        ordered_indices = indices.index_select(0, perm)
    else:
        ordered_indices = indices

    for start in range(0, int(ordered_indices.shape[0]), batch_store.batch_size):
        batch_indices = ordered_indices[start : start + batch_store.batch_size]
        batch: Dict[str, torch.Tensor] = {}
        for key, tensor in batch_store.tensors.items():
            batch_tensor = tensor.index_select(0, batch_indices)
            if batch_tensor.device != batch_store.train_device:
                batch_tensor = batch_tensor.to(batch_store.train_device, non_blocking=True)
            batch[key] = batch_tensor
        yield batch

def build_model(device: torch.device) -> model.Seas2RainModel:
    cfg = config.modelconfig
    hidden_dim = int(cfg.get("seas2rain_hidden_dim", 64))
    num_layers = int(cfg.get("seas2rain_num_layers", 1))
    decoder_channels = int(cfg.get("seas2rain_decoder_channels", 32))
    encoder_channels = int(cfg.get("seas2rain_encoder_channels", 64))
    ps_scale = int(cfg.get("seas2rain_ps_scale", 2))
    spade_hidden = int(cfg.get("spade_hidden", 16))
    lead_embed_dim = int(cfg.get("lead_embed_dim", 8))
    lead_gate_hidden = int(cfg.get("lead_gate_hidden", max(lead_embed_dim, 32)))
    lead_gate_init_bias = float(cfg.get("lead_gate_init_bias", 4.0))
    enc_spade1_hidden = int(cfg.get("enc_spade1_hidden", spade_hidden))
    enc_spade2_hidden = int(cfg.get("enc_spade2_hidden", spade_hidden))
    dec_spade_hidden = int(cfg.get("dec_spade_hidden", spade_hidden))
    sst_window = int(cfg.get("sst_window", 12))
    dropout = float(cfg.get("dropout", 0.5))
    cond_dropout = float(cfg.get("cond_dropout", 0.5))

    net = model.Seas2RainModel(
        cond_channels=len(COND_VARS),
        sst_hist_channels=sst_window,
        spade_hidden=spade_hidden,
        lead_embed_dim=lead_embed_dim,
        lead_gate_hidden=lead_gate_hidden,
        lead_gate_init_bias=lead_gate_init_bias,
        enc_spade1_hidden=enc_spade1_hidden,
        enc_spade2_hidden=enc_spade2_hidden,
        dec_spade_hidden=dec_spade_hidden,
        dropout=dropout,
        cond_dropout=cond_dropout,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
        ps_scale=ps_scale,
    ).to(device)
    return net

def _ssr_ratio(epoch: int, start: float, end: float, decay_epochs: int) -> float:
    if decay_epochs <= 0:
        return float(end)
    if start <= end:
        return float(end)
    progress = min(max(epoch, 0), decay_epochs) / float(decay_epochs)
    return float(max(end, start - (start - end) * progress))

def _init_prev_pred(ec_base: torch.Tensor, seas_anom: torch.Tensor, mode: str) -> torch.Tensor:
    # ec_base/seas_anom: [B, T, 1, H, W]
    if mode == "ec_base":
        return ec_base[:, 0, 0]
    if mode == "seas":
        return seas_anom[:, 0, 0]
    if mode == "zero":
        return ec_base.new_zeros((ec_base.shape[0], ec_base.shape[-2], ec_base.shape[-1]))
    raise ValueError(f"Unknown prev_pred_init mode: {mode}")

def autoregressive_rollout(
    net: model.Seas2RainModel,
    cond: torch.Tensor,
    seas_anom: torch.Tensor,
    ec_base: torch.Tensor,
    sst_hist: torch.Tensor,
    target: Optional[torch.Tensor],
    teacher_forcing_ratio: float,
    detach_rollout: bool,
    prev_pred_init: str,
) -> torch.Tensor:
    # cond: [B, T, 7, H, W]
    bsz, tdim = cond.shape[0], cond.shape[1]
    prev_pred = _init_prev_pred(ec_base, seas_anom, prev_pred_init)
    preds: List[torch.Tensor] = []

    state = None
    for t in range(tdim):
        if t == 0:
            prev_in = prev_pred
        else:
            if target is not None and teacher_forcing_ratio > 0.0:
                use_teacher = torch.rand(bsz, device=cond.device) < teacher_forcing_ratio
                prev_in = torch.where(use_teacher[:, None, None], target[:, t - 1, 0], prev_pred)
            else:
                prev_in = prev_pred
        lead_idx = torch.full((bsz,), t, device=cond.device, dtype=torch.long)

        pred_t, state = net.forward_step(
            cond_t=cond[:, t],
            seas_anom_t=seas_anom[:, t],
            ec_base_t=ec_base[:, t],
            prev_pred=prev_in,
            sst_hist=sst_hist,
            lead_idx=lead_idx,
            state=state,
        )
        preds.append(pred_t)
        prev_pred = pred_t.detach() if detach_rollout else pred_t

    return torch.stack(preds, dim=1)

def differentiable_acc_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    rmse_weight: float = 0.0,
    epsilon: float = 1e-8,
) -> Tuple[torch.Tensor, int]:
    # pred/target: [B, T, H, W], mask: [B, T, H, W] bool
    m = mask.float()
    count = m.sum(dim=(2, 3), keepdim=True)
    valid = count > 1.0
    count = torch.clamp(count, min=1.0)

    pred_mean = (pred * m).sum(dim=(2, 3), keepdim=True) / count
    target_mean = (target * m).sum(dim=(2, 3), keepdim=True) / count

    pred_anom = (pred - pred_mean) * m
    target_anom = (target - target_mean) * m

    cov = (pred_anom * target_anom).sum(dim=(2, 3))
    pred_std = torch.sqrt((pred_anom ** 2).sum(dim=(2, 3)) + epsilon)
    target_std = torch.sqrt((target_anom ** 2).sum(dim=(2, 3)) + epsilon)

    acc = cov / (pred_std * target_std + epsilon)
    acc = torch.where(valid.squeeze(-1).squeeze(-1), acc, torch.zeros_like(acc))
    acc_loss = 1.0 - acc.mean()

    if rmse_weight > 0.0:
        mse = ((pred - target) ** 2 * m).sum(dim=(2, 3), keepdim=True) / count
        rmse = torch.sqrt(mse + epsilon)
        rmse = torch.where(valid, rmse, torch.zeros_like(rmse))
        rmse_loss = rmse.mean()
        loss = acc_loss + rmse_weight * rmse_loss
    else:
        loss = acc_loss
    valid_count = int(valid.sum().item())
    return loss, valid_count

def init_metric_state(
    leads: int = LEADS,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    metric_device = device if device is not None else torch.device("cpu")
    return {
        "mse_sum": torch.zeros(leads, dtype=torch.float64, device=metric_device),
        "count": torch.zeros(leads, dtype=torch.float64, device=metric_device),
        "acc_sum": torch.zeros(leads, dtype=torch.float64, device=metric_device),
        "acc_cnt": torch.zeros(leads, dtype=torch.float64, device=metric_device),
    }

def update_metrics(
    state: Dict[str, torch.Tensor],
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    # pred/target/mask: [B, T, H, W]
    m = mask.float()
    sq_err = (pred - target) ** 2
    state["mse_sum"] += (sq_err * m).sum(dim=(0, 2, 3)).double()
    state["count"] += m.sum(dim=(0, 2, 3)).double()

    count = m.sum(dim=(2, 3))
    valid = count > 1.0
    count = count.clamp_min(1.0)

    pred_mean = (pred * m).sum(dim=(2, 3)) / count
    target_mean = (target * m).sum(dim=(2, 3)) / count

    pred_anom = (pred - pred_mean[:, :, None, None]) * m
    target_anom = (target - target_mean[:, :, None, None]) * m

    cov = (pred_anom * target_anom).sum(dim=(2, 3))
    pred_std = torch.sqrt((pred_anom ** 2).sum(dim=(2, 3)) + 1e-12)
    target_std = torch.sqrt((target_anom ** 2).sum(dim=(2, 3)) + 1e-12)
    acc = cov / (pred_std * target_std + 1e-12)
    acc = torch.where(valid, acc, torch.zeros_like(acc))

    state["acc_sum"] += acc.sum(dim=0).double()
    state["acc_cnt"] += valid.sum(dim=0).double()

def finalize_metrics(state: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    eps = 1e-12
    count = torch.clamp(state["count"], min=eps)
    rmse = torch.sqrt(state["mse_sum"] / count)
    acc = state["acc_sum"] / torch.clamp(state["acc_cnt"], min=1.0)
    acc = torch.where(state["acc_cnt"] > 0, acc, torch.full_like(acc, float("nan")))
    return {
        "rmse": rmse.detach().cpu().numpy(),
        "acc": acc.detach().cpu().numpy(),
    }

def evaluate_baseline(batch_store: TensorBatchStore, split: str) -> Dict[str, np.ndarray]:
    state = init_metric_state(device=batch_store.train_device)
    with torch.inference_mode():
        for batch in iter_split_batches(batch_store, split=split, shuffle=False):
            pred = batch["ec_base"][:, :, 0]
            target2 = batch["obs_target"][:, :, 0]
            mask2 = batch["obs_mask"][:, :, 0] > 0.5
            update_metrics(state, pred, target2, mask2)
    return finalize_metrics(state)

def format_metric_line(name: str, values: np.ndarray) -> str:
    parts = [f"L{i+1}:{float(values[i]):.4f}" if np.isfinite(values[i]) else f"L{i+1}:nan" for i in range(len(values))]
    return f"{name}: " + ", ".join(parts)

def train() -> None:
    seed = int(config.modelconfig.get("seed", 42))
    set_seed(seed)

    device = config.modelconfig["device"]
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass

    data = prepare_data()
    batch_store = build_batch_store(data, train_device=device)

    net = build_model(device=device)

    lr = float(config.modelconfig.get("lr", 1e-4))
    weight_decay = float(config.modelconfig.get("weight_decay", 1e-4))
    epochs = int(config.modelconfig.get("epoch", 200))
    grad_clip = float(config.modelconfig.get("grad_clip", 2.0))
    save_every = int(config.modelconfig.get("save_every", 5))
    patience = int(config.modelconfig.get("patience", 12))
    min_delta = float(config.modelconfig.get("early_stop_min_delta", 0.0001))
    autoregressive = bool(config.modelconfig.get("autoregressive", True))
    ssr_start = float(config.modelconfig.get("ssr_start", 1.0))
    ssr_end = float(config.modelconfig.get("ssr_end", 0.0))
    ssr_decay_epochs = int(config.modelconfig.get("ssr_decay_epochs", 30))
    prev_pred_init = str(config.modelconfig.get("prev_pred_init", "ec_base"))
    detach_rollout = bool(config.modelconfig.get("detach_rollout", False))
    input_noise_std = float(config.modelconfig.get("input_noise_std", 0.0))
    rmse_weight = float(config.modelconfig.get("rmse_weight", 0))
    progress_log_interval = max(1, int(config.modelconfig.get("progress_log_interval", 10)))

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    writer = SummaryWriter(log_dir=config.modelconfig["log_path"])

    print(
        "Train setup:",
        {
            "device": str(device),
            "batch_size": batch_store.batch_size,
            "dataset_storage": str(batch_store.storage_device),
            "train_samples": batch_store.split_size("train"),
            "val_samples": batch_store.split_size("val"),
            "test_samples": batch_store.split_size("test"),
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "autoregressive": autoregressive,
        },
    )

    baseline_val = evaluate_baseline(batch_store, split="val")
    baseline_test = evaluate_baseline(batch_store, split="test")
    print(format_metric_line("Baseline VAL RMSE", baseline_val["rmse"]))
    print(format_metric_line("Baseline TEST RMSE", baseline_test["rmse"]))
    print(format_metric_line("Baseline VAL ACC", baseline_val["acc"]))
    print(format_metric_line("Baseline TEST ACC", baseline_test["acc"]))

    best_val_loss = float("inf")
    best_epoch = -1
    stale_epochs = 0
    train_num_batches = count_split_batches(batch_store, "train")

    for epoch in range(epochs):
        net.train()
        train_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        train_loss_batches = 0
        train_state = init_metric_state(device=device)
        skipped_batches = 0
        teacher_forcing_ratio = _ssr_ratio(epoch, ssr_start, ssr_end, ssr_decay_epochs) if autoregressive else 0.0

        pbar = tqdm.tqdm(
            iter_split_batches(batch_store, split="train", shuffle=True),
            desc=f"Epoch {epoch}",
            total=train_num_batches,
            mininterval=1.0,
        )
        for batch_idx, batch in enumerate(pbar, start=1):
            cond = batch["cond"]
            seas_anom = batch["seas_anom"]
            ec_base = batch["ec_base"]
            target = batch["obs_target"]
            obs_mask = batch["obs_mask"]
            sst_hist = batch["sst_hist"]
            if input_noise_std > 0.0:
                cond = cond + torch.randn_like(cond) * input_noise_std
                sst_hist = sst_hist + torch.randn_like(sst_hist) * input_noise_std

            if autoregressive:
                pred = autoregressive_rollout(
                    net=net,
                    cond=cond,
                    seas_anom=seas_anom,
                    ec_base=ec_base,
                    sst_hist=sst_hist,
                    target=target,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                    detach_rollout=detach_rollout,
                    prev_pred_init=prev_pred_init,
                )
            else:
                pred = net(cond, seas_anom=seas_anom, ec_base=ec_base, sst_hist=sst_hist)

            mask2 = obs_mask[:, :, 0] > 0.5
            loss, valid_count = differentiable_acc_loss(pred, target[:, :, 0], mask2, rmse_weight=rmse_weight)

            if valid_count == 0:
                skipped_batches += 1
                continue

            if not torch.isfinite(loss.detach()).item():
                raise FloatingPointError("Non-finite loss detected in training.")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()

            train_loss_sum = train_loss_sum + loss.detach()
            train_loss_batches += 1
            update_metrics(train_state, pred.detach(), target[:, :, 0].detach(), mask2.detach())

            if batch_idx % progress_log_interval == 0 or batch_idx == train_num_batches:
                avg_loss = (
                    float((train_loss_sum / train_loss_batches).item())
                    if train_loss_batches > 0
                    else float("nan")
                )
                pbar.set_postfix(loss=f"{avg_loss:.4f}", skipped=skipped_batches)

        train_metrics = finalize_metrics(train_state)
        train_loss = (
            float((train_loss_sum / train_loss_batches).item())
            if train_loss_batches > 0
            else float("inf")
        )

        net.eval()
        val_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        val_loss_batches = 0
        val_state = init_metric_state(device=device)
        with torch.inference_mode():
            for batch in iter_split_batches(batch_store, split="val", shuffle=False):
                cond = batch["cond"]
                seas_anom = batch["seas_anom"]
                ec_base = batch["ec_base"]
                target = batch["obs_target"]
                obs_mask = batch["obs_mask"]
                sst_hist = batch["sst_hist"]
                if autoregressive:
                    pred = autoregressive_rollout(
                        net=net,
                        cond=cond,
                        seas_anom=seas_anom,
                        ec_base=ec_base,
                        sst_hist=sst_hist,
                        target=None,
                        teacher_forcing_ratio=0.0,
                        detach_rollout=True,
                        prev_pred_init=prev_pred_init,
                    )
                else:
                    pred = net(cond, seas_anom=seas_anom, ec_base=ec_base, sst_hist=sst_hist)

                mask2 = obs_mask[:, :, 0] > 0.5
                loss, valid_count = differentiable_acc_loss(pred, target[:, :, 0], mask2, rmse_weight=rmse_weight)
                if valid_count == 0:
                    continue
                val_loss_sum = val_loss_sum + loss.detach()
                val_loss_batches += 1
                update_metrics(val_state, pred, target[:, :, 0], mask2)

        val_metrics = finalize_metrics(val_state)
        val_loss = (
            float((val_loss_sum / val_loss_batches).item())
            if val_loss_batches > 0
            else float("inf")
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        for lead in range(LEADS):
            writer.add_scalar(f"RMSE/train_lead{lead+1}", train_metrics["rmse"][lead], epoch)
            writer.add_scalar(f"RMSE/val_lead{lead+1}", val_metrics["rmse"][lead], epoch)
            writer.add_scalar(f"ACC/train_lead{lead+1}", train_metrics["acc"][lead], epoch)
            writer.add_scalar(f"ACC/val_lead{lead+1}", val_metrics["acc"][lead], epoch)

        print(f"Epoch {epoch}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}, skipped={skipped_batches}")
        print(format_metric_line("VAL RMSE", val_metrics["rmse"]))
        print(format_metric_line("VAL ACC", val_metrics["acc"]))
        skill = baseline_val["rmse"] - val_metrics["rmse"]
        acc_diff = val_metrics["acc"] - baseline_val["acc"]
        print(format_metric_line("VAL RMSE Skill (baseline-model)", skill))
        print(format_metric_line("VAL ACC Skill (baseline-model)", acc_diff))

        if epoch % max(1, save_every) == 0:
            ckpt_path = os.path.join(config.modelconfig["save_weight_path"], f"epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_path,
            )

        if val_loss + min_delta < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            best_path = os.path.join(config.modelconfig["save_weight_path"], "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping at epoch {epoch}. best_epoch={best_epoch}, best_val_loss={best_val_loss:.6f}")
                # break

    best_path = os.path.join(config.modelconfig["save_weight_path"], "best.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        net.load_state_dict(ckpt["model_state_dict"], strict=False)

    net.eval()
    test_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    test_loss_batches = 0
    test_state = init_metric_state(device=device)
    with torch.inference_mode():
        for batch in iter_split_batches(batch_store, split="test", shuffle=False):
            cond = batch["cond"]
            seas_anom = batch["seas_anom"]
            ec_base = batch["ec_base"]
            target = batch["obs_target"]
            obs_mask = batch["obs_mask"]
            sst_hist = batch["sst_hist"]
            if autoregressive:
                pred = autoregressive_rollout(
                    net=net,
                    cond=cond,
                    seas_anom=seas_anom,
                    ec_base=ec_base,
                    sst_hist=sst_hist,
                    target=None,
                    teacher_forcing_ratio=0.0,
                    detach_rollout=True,
                    prev_pred_init=prev_pred_init,
                )
            else:
                pred = net(cond, seas_anom=seas_anom, ec_base=ec_base, sst_hist=sst_hist)

            mask2 = obs_mask[:, :, 0] > 0.5
            loss, valid_count = differentiable_acc_loss(pred, target[:, :, 0], mask2, rmse_weight=rmse_weight)
            if valid_count == 0:
                continue
            test_loss_sum = test_loss_sum + loss.detach()
            test_loss_batches += 1
            update_metrics(test_state, pred, target[:, :, 0], mask2)

    test_metrics = finalize_metrics(test_state)
    test_loss = (
        float((test_loss_sum / test_loss_batches).item())
        if test_loss_batches > 0
        else float("inf")
    )
    print(f"Final TEST loss={test_loss:.5f}")
    print(format_metric_line("TEST RMSE", test_metrics["rmse"]))
    print(format_metric_line("TEST ACC", test_metrics["acc"]))
    test_skill = baseline_test["rmse"] - test_metrics["rmse"]
    print(format_metric_line("TEST RMSE Skill (baseline-model)", test_skill))

    writer.close()

if __name__ == "__main__":
    train()
