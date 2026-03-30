import os
import random
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
from scipy.interpolate import griddata
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import tqdm

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
COND_VARS = ["h500", "slp", "t2m", "t850", "u850", "v850", "sst"]
TARGET_HW = (60, 70)


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


def _sel_region(ds: xr.Dataset) -> xr.Dataset:
    if "longitude" in ds.coords and "latitude" in ds.coords:
        return ds.sel(longitude=slice(70, 140), latitude=slice(60, 0))
    if "lon" in ds.coords and "lat" in ds.coords:
        return ds.sel(lon=slice(70, 140), lat=slice(60, 0))
    return ds


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
        return tp_raw, cond_raw, pd.DatetimeIndex(init_dates), grid_lats, grid_lons

    file_list = utils.read_nc_to_npy(199401, 202409, data_path=seas_nc_path or "D:\\MODESv21_ecmwf_seas51")

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
                ds_tp = _sel_region(ds)
                tp = ds_tp[["tp"]]
                cond = ds[COND_VARS]

                tp_arr = tp.to_array().to_numpy().astype(np.float32)  # [1, L, H, W]
                cond_arr = cond.to_array().to_numpy().astype(np.float32)  # [7, L, H, W]

                if tp_arr.shape[1] != LEADS:
                    raise ValueError(f"tp lead mismatch: expected {LEADS}, got {tp_arr.shape[1]}")
                if cond_arr.shape[1] != LEADS:
                    raise ValueError(f"cond lead mismatch: expected {LEADS}, got {cond_arr.shape[1]}")

                lat, lon = _get_lat_lon(tp)
                if grid_lats is None or grid_lons is None:
                    grid_lats, grid_lons = _build_target_grid(lat, lon, target_hw)

                if tp_arr.shape[-2:] != target_hw:
                    raise ValueError(
                        f"tp spatial shape mismatch: expected {target_hw}, got {tp_arr.shape[-2:]}"
                    )

                tp_arr = np.nan_to_num(tp_arr, nan=0.0, posinf=0.0, neginf=0.0)
                cond_arr = np.nan_to_num(cond_arr, nan=0.0, posinf=0.0, neginf=0.0)

                tp_list.append(tp_arr.squeeze(0))  # [L, H, W]
                cond_list.append(cond_arr)  # [7, L, H, W]
                date_list.append(date)
        except Exception as e:
            print(f"Skip file {f}: {e}")
            continue

    if len(tp_list) == 0:
        raise RuntimeError("No valid MODESv21 files found.")

    tp_raw = np.stack(tp_list, axis=0)  # [N, L, H, W]
    cond_raw = np.stack(cond_list, axis=0)  # [N, 7, L, H, W]
    cond_raw = np.transpose(cond_raw, (0, 2, 1, 3, 4))  # [N, L, 7, H, W]

    init_dates = pd.DatetimeIndex(date_list)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        tp_raw=tp_raw.astype(np.float32),
        cond_raw=cond_raw.astype(np.float32),
        init_dates=np.array([d.strftime("%Y-%m-%d") for d in init_dates]),
        grid_lats=grid_lats.astype(np.float32),
        grid_lons=grid_lons.astype(np.float32),
    )
    return tp_raw.astype(np.float32), cond_raw.astype(np.float32), init_dates, grid_lats, grid_lons


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
    # cond_raw: [N, L, 7, H, W]
    sum_x = np.zeros((LEADS, len(COND_VARS)), dtype=np.float64)
    sum_x2 = np.zeros((LEADS, len(COND_VARS)), dtype=np.float64)
    count = np.zeros((LEADS, len(COND_VARS)), dtype=np.float64)

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
    # cond_raw: [N, L, 7, H, W], mean/std: [L, 7]
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

    modes_cache = os.path.join(cache_dir, "modes_tp_cond_60x70.npz")
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
    cond_norm_cache = os.path.join(cache_dir, f"cond_norm_n{len(init_dates)}.npy")
    if os.path.exists(cond_norm_cache):
        cond_norm = np.load(cond_norm_cache).astype(np.float32)
        if cond_norm.shape != cond_raw.shape:
            cond_norm = None
    else:
        cond_norm = None

    if cond_norm is None:
        cond_norm = normalize_cond(cond_raw, cond_mean, cond_std)
        np.save(cond_norm_cache, cond_norm)

    obs_csv_path = cfg.get(
        "observe_csv_path",
        os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "utils", "observe_data24.csv")),
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
        "cond": cond_norm.astype(np.float32),  # 来自EC的条件场数据，已经过归一化处理，形状为[N, L, 7, H, W]
        "seas_anom": seas_anom.astype(np.float32),  # 来自EC的降水异常数据，形状为[N, L, 1, H, W]
        "ec_base": ec_base.astype(np.float32),  # 来自EC的降水异常数据，形状为[N, L, 1, H, W]
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
    for k in ("cond", "seas_anom", "ec_base", "obs_target", "obs_mask"):
        if data[k].shape[0] != n:
            raise ValueError(f"{k} length mismatch: {data[k].shape[0]} vs {n}")

    if data["cond"].ndim != 5 or data["cond"].shape[1] != LEADS or data["cond"].shape[2] != len(COND_VARS):
        raise ValueError(f"cond shape invalid: {data['cond'].shape}")
    if tuple(data["seas_anom"].shape[1:]) != (LEADS, 1, TARGET_HW[0], TARGET_HW[1]):
        raise ValueError(f"seas_anom shape invalid: {data['seas_anom'].shape}")
    if tuple(data["ec_base"].shape[1:]) != (LEADS, 1, TARGET_HW[0], TARGET_HW[1]):
        raise ValueError(f"ec_base shape invalid: {data['ec_base'].shape}")
    if tuple(data["obs_target"].shape[1:]) != (LEADS, 1, TARGET_HW[0], TARGET_HW[1]):
        raise ValueError(f"obs_target shape invalid: {data['obs_target'].shape}")
    if tuple(data["obs_mask"].shape[1:]) != (LEADS, 1, TARGET_HW[0], TARGET_HW[1]):
        raise ValueError(f"obs_mask shape invalid: {data['obs_mask'].shape}")


class Seas2RainDataset(Dataset):
    def __init__(self, data: Dict[str, np.ndarray], indices: np.ndarray):
        self.cond = data["cond"]
        self.seas_anom = data["seas_anom"]
        self.ec_base = data["ec_base"]
        self.obs_target = data["obs_target"]
        self.obs_mask = data["obs_mask"]
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        cond = torch.from_numpy(self.cond[idx])
        seas_anom = torch.from_numpy(self.seas_anom[idx])
        ec_base = torch.from_numpy(self.ec_base[idx])
        obs_target = torch.from_numpy(self.obs_target[idx])
        obs_mask = torch.from_numpy(self.obs_mask[idx])
        return cond, seas_anom, ec_base, obs_target, obs_mask


def build_dataloaders(data: Dict[str, np.ndarray], device: torch.device):
    train_ds = Seas2RainDataset(data, data["split_indices"]["train"])
    val_ds = Seas2RainDataset(data, data["split_indices"]["val"])
    test_ds = Seas2RainDataset(data, data["split_indices"]["test"])

    batch_size = int(config.modelconfig.get("batch_size", 8))
    num_workers = int(config.modelconfig.get("num_workers", 0))
    pin_memory = str(device).startswith("cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader


def build_model(device: torch.device) -> model.Seas2RainModel:
    cfg = config.modelconfig
    hidden_dim = int(cfg.get("seas2rain_hidden_dim", 64))
    num_layers = int(cfg.get("seas2rain_num_layers", 1))
    decoder_channels = int(cfg.get("seas2rain_decoder_channels", 32))
    encoder_channels = int(cfg.get("seas2rain_encoder_channels", 64))
    ps_scale = int(cfg.get("seas2rain_ps_scale", 2))

    net = model.Seas2RainModel(
        cond_channels=len(COND_VARS),
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
    target: Optional[torch.Tensor],
    teacher_forcing_ratio: float,
    detach_rollout: bool,
    prev_pred_init: str,
) -> torch.Tensor:
    # cond: [B, T, 7, 60, 70]
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

        pred_t, state = net.forward_step(
            cond_t=cond[:, t],
            seas_anom_t=seas_anom[:, t],
            ec_base_t=ec_base[:, t],
            prev_pred=prev_in,
            state=state,
        )
        preds.append(pred_t)
        prev_pred = pred_t.detach() if detach_rollout else pred_t

    return torch.stack(preds, dim=1)


def differentiable_acc_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
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
    loss = 1.0 - acc.mean()
    valid_count = int(valid.sum().item())
    return loss, valid_count


def init_metric_state(leads: int = LEADS) -> Dict[str, np.ndarray]:
    return {
        "mse_sum": np.zeros(leads, dtype=np.float64),
        "count": np.zeros(leads, dtype=np.float64),
        "acc_sum": np.zeros(leads, dtype=np.float64),
        "acc_cnt": np.zeros(leads, dtype=np.float64),
    }


def _corrcoef_1d(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).sum() * (y * y).sum())
    if denom <= 1e-12:
        return float("nan")
    return float(((x * y).sum() / denom).item())


def update_metrics(
    state: Dict[str, np.ndarray],
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    # pred/target/mask: [B, T, H, W]
    bsz, leads, _, _ = pred.shape
    err = pred - target
    sq_err = err * err

    for t in range(leads):
        mt = mask[:, t]
        count = float(mt.sum().item())
        if count <= 0:
            continue
        state["mse_sum"][t] += float(sq_err[:, t][mt].sum().item())
        state["count"][t] += count

    for b in range(bsz):
        for t in range(leads):
            mt = mask[b, t]
            n_valid = int(mt.sum().item())
            if n_valid < 2:
                continue
            c = _corrcoef_1d(pred[b, t][mt], target[b, t][mt])
            if np.isfinite(c):
                state["acc_sum"][t] += c
                state["acc_cnt"][t] += 1.0


def finalize_metrics(state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    eps = 1e-12
    rmse = np.sqrt(state["mse_sum"] / np.maximum(state["count"], eps))
    acc = state["acc_sum"] / np.maximum(state["acc_cnt"], 1.0)
    acc[state["acc_cnt"] < 1] = np.nan
    return {"rmse": rmse, "acc": acc}


def evaluate_baseline(loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    state = init_metric_state()
    with torch.no_grad():
        for _, _, ec_base, target, obs_mask in loader:
            ec_base = ec_base.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            obs_mask = obs_mask.to(device, non_blocking=True)
            pred = ec_base[:, :, 0]
            target2 = target[:, :, 0]
            mask2 = obs_mask[:, :, 0] > 0.5
            update_metrics(state, pred, target2, mask2)
    return finalize_metrics(state)


def format_metric_line(name: str, values: np.ndarray) -> str:
    parts = [f"L{i+1}:{float(values[i]):.4f}" if np.isfinite(values[i]) else f"L{i+1}:nan" for i in range(len(values))]
    return f"{name}: " + ", ".join(parts)


def train() -> None:
    seed = int(config.modelconfig.get("seed", 42))
    set_seed(seed)

    device = config.modelconfig["device"]
    data = prepare_data()
    train_loader, val_loader, test_loader = build_dataloaders(data, device=device)

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

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    writer = SummaryWriter(log_dir=config.modelconfig["log_path"])

    print(
        "Train setup:",
        {
            "device": str(device),
            "batch_size": train_loader.batch_size,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "autoregressive": autoregressive,
        },
    )

    baseline_val = evaluate_baseline(val_loader, device=device)
    baseline_test = evaluate_baseline(test_loader, device=device)
    print(format_metric_line("Baseline VAL RMSE", baseline_val["rmse"]))
    print(format_metric_line("Baseline TEST RMSE", baseline_test["rmse"]))
    print(format_metric_line("Baseline VAL ACC", baseline_val["acc"]))
    print(format_metric_line("Baseline TEST ACC", baseline_test["acc"]))

    best_val_loss = float("inf")
    best_epoch = -1
    stale_epochs = 0

    for epoch in range(epochs):
        net.train()
        train_losses: List[float] = []
        train_state = init_metric_state()
        skipped_batches = 0
        teacher_forcing_ratio = _ssr_ratio(epoch, ssr_start, ssr_end, ssr_decay_epochs) if autoregressive else 0.0

        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}")
        for cond, seas_anom, ec_base, target, obs_mask in pbar:
            cond = cond.to(device, non_blocking=True)
            seas_anom = seas_anom.to(device, non_blocking=True)
            ec_base = ec_base.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            obs_mask = obs_mask.to(device, non_blocking=True)

            if autoregressive:
                pred = autoregressive_rollout(
                    net=net,
                    cond=cond,
                    seas_anom=seas_anom,
                    ec_base=ec_base,
                    target=target,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                    detach_rollout=detach_rollout,
                    prev_pred_init=prev_pred_init,
                )
            else:
                pred = net(cond, seas_anom=seas_anom, ec_base=ec_base)

            mask2 = obs_mask[:, :, 0] > 0.5
            loss, valid_count = differentiable_acc_loss(pred, target[:, :, 0], mask2)

            if valid_count == 0:
                skipped_batches += 1
                continue

            if (not torch.isfinite(pred).all().item()) or (not torch.isfinite(loss).item()):
                raise FloatingPointError("Non-finite values detected in training.")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()

            train_losses.append(float(loss.item()))
            update_metrics(train_state, pred.detach(), target[:, :, 0].detach(), mask2.detach())

            pbar.set_postfix(loss=f"{loss.item():.4f}", skipped=skipped_batches)

        train_metrics = finalize_metrics(train_state)
        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")

        net.eval()
        val_losses: List[float] = []
        val_state = init_metric_state()
        with torch.no_grad():
            for cond, seas_anom, ec_base, target, obs_mask in val_loader:
                cond = cond.to(device, non_blocking=True)
                seas_anom = seas_anom.to(device, non_blocking=True)
                ec_base = ec_base.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                obs_mask = obs_mask.to(device, non_blocking=True)

                if autoregressive:
                    pred = autoregressive_rollout(
                        net=net,
                        cond=cond,
                        seas_anom=seas_anom,
                        ec_base=ec_base,
                        target=None,
                        teacher_forcing_ratio=0.0,
                        detach_rollout=True,
                        prev_pred_init=prev_pred_init,
                    )
                else:
                    pred = net(cond, seas_anom=seas_anom, ec_base=ec_base)

                mask2 = obs_mask[:, :, 0] > 0.5
                loss, valid_count = differentiable_acc_loss(pred, target[:, :, 0], mask2)
                if valid_count == 0:
                    continue
                val_losses.append(float(loss.item()))
                update_metrics(val_state, pred, target[:, :, 0], mask2)

        val_metrics = finalize_metrics(val_state)
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

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
    test_losses: List[float] = []
    test_state = init_metric_state()
    with torch.no_grad():
        for cond, seas_anom, ec_base, target, obs_mask in test_loader:
            cond = cond.to(device, non_blocking=True)
            seas_anom = seas_anom.to(device, non_blocking=True)
            ec_base = ec_base.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            obs_mask = obs_mask.to(device, non_blocking=True)

            if autoregressive:
                pred = autoregressive_rollout(
                    net=net,
                    cond=cond,
                    seas_anom=seas_anom,
                    ec_base=ec_base,
                    target=None,
                    teacher_forcing_ratio=0.0,
                    detach_rollout=True,
                    prev_pred_init=prev_pred_init,
                )
            else:
                pred = net(cond, seas_anom=seas_anom, ec_base=ec_base)

            mask2 = obs_mask[:, :, 0] > 0.5
            loss, valid_count = differentiable_acc_loss(pred, target[:, :, 0], mask2)
            if valid_count == 0:
                continue
            test_losses.append(float(loss.item()))
            update_metrics(test_state, pred, target[:, :, 0], mask2)

    test_metrics = finalize_metrics(test_state)
    test_loss = float(np.mean(test_losses)) if test_losses else float("inf")
    print(f"Final TEST loss={test_loss:.5f}")
    print(format_metric_line("TEST RMSE", test_metrics["rmse"]))
    print(format_metric_line("TEST ACC", test_metrics["acc"]))
    test_skill = baseline_test["rmse"] - test_metrics["rmse"]
    print(format_metric_line("TEST RMSE Skill (baseline-model)", test_skill))

    writer.close()


if __name__ == "__main__":
    train()
