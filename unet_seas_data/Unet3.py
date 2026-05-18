import sys
import os
import glob
import random
import re
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy.interpolate import griddata
import tqdm
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader

# Ensure repo parent is on sys.path so absolute imports like 'HydroSynth' work.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_repo_root = os.path.normpath(_repo_root)
_repo_parent = os.path.dirname(_repo_root)
if _repo_parent not in sys.path:
    sys.path.insert(0, _repo_parent)

import HydroSynth.config as config
from HydroSynth.utils import utils
from unetlitefilm import UNetLiteFiLM

config.auto_save_config()

LEADS = 6
COND_VARS = ["h500", "slp", "t2m", "t850", "u850", "v850", "u200", "v200", "t200", "sst"]
TARGET_HW = (60, 70)
MODES_CACHE_VERSION = "20260505_all_vars_global_sst_v1"

def _cond_vars_signature() -> str:
    return "_".join(COND_VARS)

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
                if "longitude" in ds.coords: ds = ds.rename({"longitude": "lon", "latitude": "lat"})
                tp_ds = ds[["tp"]].sel(lon=slice(70, 140), lat=slice(60, 0))
                lat, lon = _get_lat_lon(tp_ds)
                if grid_lats is None or grid_lons is None:
                    grid_lats, grid_lons = _build_target_grid(lat, lon, target_hw)
                tp_arr = tp_ds.interp(lat=grid_lats, lon=grid_lons, method="linear").to_array().to_numpy().astype(np.float32)
                
                cond_layers = []
                for v in COND_VARS:
                    if v in ds:
                        v_ds = ds[[v]]
                        v_interp = v_ds.interp(lat=grid_lats, lon=grid_lons, method="linear")[v].to_numpy()
                        cond_layers.append(v_interp)
                    else:
                        # Fallback for missing variables - skip this file or fill with zeros?
                        # Consistency is key for stacking, so we fill with zeros if missing.
                        cond_layers.append(np.zeros((LEADS, target_hw[0], target_hw[1]), dtype=np.float32))
                
                cond_arr = np.stack(cond_layers, axis=0).astype(np.float32)

                if tp_arr.shape[1] != LEADS or cond_arr.shape[1] != LEADS:
                    continue

                tp_arr = np.nan_to_num(tp_arr, nan=0.0, posinf=0.0, neginf=0.0)
                cond_arr = np.nan_to_num(cond_arr, nan=0.0, posinf=0.0, neginf=0.0)
                tp_list.append(tp_arr.squeeze(0))
                cond_list.append(cond_arr)
                date_list.append(date)
        except Exception:
            continue

    if len(tp_list) == 0:
        raise RuntimeError("No valid MODESv21 files found.")

    tp_raw = np.stack(tp_list, axis=0)
    cond_raw = np.stack(cond_list, axis=0)
    cond_raw = np.transpose(cond_raw, (0, 2, 1, 3, 4))
    init_dates = pd.DatetimeIndex(date_list)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        tp_raw=tp_raw,
        cond_raw=cond_raw,
        init_dates=np.array([d.strftime("%Y-%m-%d") for d in init_dates]),
        grid_lats=grid_lats,
        grid_lons=grid_lons,
        cond_vars=np.array(COND_VARS, dtype="<U32"),
        cache_version=np.array(MODES_CACHE_VERSION),
    )
    return tp_raw, cond_raw, init_dates, grid_lats, grid_lons

def _parse_ersst_date(path: str) -> Optional[pd.Timestamp]:
    name = os.path.basename(path)
    match = re.search(r"(\d{6})", name)
    if not match: return None
    yyyymm = match.group(1)
    return pd.Timestamp(year=int(yyyymm[:4]), month=int(yyyymm[4:]), day=1)

def read_ersst_data(ersst_dir: str, cache_path: str) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        return cached["ssta"].astype(np.float32), pd.to_datetime(cached["dates"].astype(str))
    
    files = sorted(glob.glob(os.path.join(ersst_dir, "ersst.v5.*.nc")))
    sst_list, date_list = [], []
    for f in tqdm.tqdm(files, desc="Read ERSST"):
        date = _parse_ersst_date(f)
        if date is None: continue
        try:
            with xr.open_dataset(f) as ds:
                if "longitude" in ds.coords: ds = ds.rename({"longitude": "lon", "latitude": "lat"})
                
                # Global sea temperature - no slicing
                ds_ssta = ds

                if "ssta" not in ds_ssta:
                    continue
                
                arr = np.asarray(ds_ssta["ssta"].to_numpy())
                if arr.size == 0:
                    continue
                
                if arr.ndim == 4: arr = arr[0, 0]
                elif arr.ndim == 3: arr = arr[0]
                
                sst_list.append(np.nan_to_num(arr, nan=0.0).astype(np.float32))
                date_list.append(date)
        except Exception: continue
    if len(sst_list) == 0:
        raise RuntimeError(f"No valid ERSST ssta data found in {ersst_dir}.")

    sst = np.stack(sst_list, axis=0)
    dates = pd.DatetimeIndex(date_list)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, ssta=sst, dates=np.array([d.strftime("%Y-%m-%d") for d in dates]))
    return sst, dates

def build_sst_history(init_dates, sst, sst_dates, window) -> Tuple[np.ndarray, np.ndarray]:
    sst_index = {_month_start(d): i for i, d in enumerate(sst_dates)}
    keep, hist = [], []
    for i, d in enumerate(init_dates):
        seq, ok = [], True
        for m in range(window, 0, -1):
            md = _month_start(d - pd.DateOffset(months=m))
            idx = sst_index.get(md, None)
            if idx is None: ok = False; break
            seq.append(sst[idx])
        if ok:
            keep.append(i)
            hist.append(np.stack(seq, axis=0))
    return np.stack(hist, axis=0).astype(np.float32), np.asarray(keep, dtype=np.int64)

def normalize_sst_hist(sst_hist, train_idx) -> Tuple[np.ndarray, float, float]:
    train_vals = sst_hist[train_idx]
    mean, std = float(train_vals.mean()), float(train_vals.std())
    return ((sst_hist - mean) / max(std, 1e-6)).astype(np.float32), mean, std

def calc_precip_percent_anomaly(tp_tensor, init_dates, lead_dependent=True) -> Tuple[torch.Tensor, torch.Tensor]:
    T, L, H, W = tp_tensor.shape
    target_month_idx = torch.empty((T, L), dtype=torch.long)
    for t, d in enumerate(init_dates):
        for l in range(L):
            target_month_idx[t, l] = (d + pd.DateOffset(months=l)).month
    pa = torch.zeros_like(tp_tensor)
    if lead_dependent:
        climatology = torch.zeros((L, 12, H, W), dtype=tp_tensor.dtype, device=tp_tensor.device)
        for l in range(L):
            for m in range(1, 13):
                idx_t = (target_month_idx[:, l] == m).nonzero(as_tuple=True)[0]
                if idx_t.numel() == 0: continue
                clim = tp_tensor[idx_t, l].mean(dim=0)
                climatology[l, m - 1] = clim
                pa[idx_t, l] = (tp_tensor[idx_t, l] - clim) / (clim + 1e-6) * 100.0
    return pa, climatology

def compute_cond_stats(cond_raw, train_idx) -> Tuple[np.ndarray, np.ndarray]:
    lead_dim, cond_dim = cond_raw.shape[1], cond_raw.shape[2]
    sum_x, sum_x2, count = np.zeros((lead_dim, cond_dim)), np.zeros((lead_dim, cond_dim)), np.zeros((lead_dim, cond_dim))
    for idx in train_idx:
        x = cond_raw[int(idx)]
        valid = np.isfinite(x) & (x > -9000.0)
        xv = np.where(valid, x, 0.0)
        sum_x += xv.sum(axis=(2, 3)); sum_x2 += (xv * xv).sum(axis=(2, 3)); count += valid.sum(axis=(2, 3))
    mean = sum_x / np.maximum(count, 1.0)
    std = np.sqrt(np.maximum(sum_x2 / np.maximum(count, 1.0) - mean * mean, 1e-8))
    return mean.astype(np.float32), np.maximum(std, 1e-4).astype(np.float32)

def normalize_cond(cond_raw, mean, std) -> np.ndarray:
    x = (cond_raw - mean[None, :, :, None, None]) / std[None, :, :, None, None]
    return np.nan_to_num(np.clip(x, -6.0, 6.0), nan=0.0).astype(np.float32)

def prepare_observe_data(csv_path, cache_path, grid_lons, grid_lats) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=False)
        return cached["obs_grid"], cached["obs_mask"], pd.to_datetime(cached["obs_dates"].astype(str))
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"]).apply(_month_start)
    df = _normalize_station_coords(df)
    all_months = pd.date_range(start="1994-01-01", end="2024-12-01", freq="MS")
    grid_lon2d, grid_lat2d = np.meshgrid(grid_lons, grid_lats)
    grid_all, mask_all = [], []
    for cur_month in tqdm.tqdm(all_months, desc="Obs Interpolation"):
        cur = df[df["time"] == cur_month]
        if len(cur) < 3:
            grid_all.append(np.zeros(TARGET_HW, dtype=np.float32)); mask_all.append(np.zeros(TARGET_HW, dtype=np.float32))
            continue
        pts, vals = cur[["Long", "Lat"]].to_numpy(), cur["anoma"].to_numpy()
        try:
            linear = griddata(pts, vals, (grid_lon2d, grid_lat2d), method="linear")
            nearest = griddata(pts, vals, (grid_lon2d, grid_lat2d), method="nearest")
            valid = np.isfinite(linear)
            grid_all.append(np.nan_to_num(np.where(valid, linear, nearest), nan=0.0).astype(np.float32))
            mask_all.append(valid.astype(np.float32))
        except Exception:
            grid_all.append(np.zeros(TARGET_HW, dtype=np.float32)); mask_all.append(np.zeros(TARGET_HW, dtype=np.float32))
    obs_grid, obs_mask = np.stack(grid_all, axis=0), np.stack(mask_all, axis=0)
    np.savez_compressed(cache_path, obs_grid=obs_grid, obs_mask=obs_mask, obs_dates=np.array([d.strftime("%Y-%m-%d") for d in all_months]))
    return obs_grid, obs_mask, all_months

def build_obs_targets(init_dates, obs_grid, obs_mask, obs_dates, leads=LEADS) -> Tuple[np.ndarray, np.ndarray]:
    n = len(init_dates)
    target, mask = np.zeros((n, leads, TARGET_HW[0], TARGET_HW[1]), dtype=np.float32), np.zeros((n, leads, TARGET_HW[0], TARGET_HW[1]), dtype=np.float32)
    obs_index = {_month_start(d): i for i, d in enumerate(obs_dates)}
    for i, init_d in enumerate(init_dates):
        for l in range(leads):
            idx = obs_index.get(_month_start(init_d + pd.DateOffset(months=l)), None)
            if idx is not None: target[i, l], mask[i, l] = obs_grid[idx], obs_mask[idx]
    return target, mask

def validate_data_bundle(data: Dict[str, np.ndarray]) -> None:
    n = len(data["init_dates"])
    for k in ("cond", "seas_anom", "sst_hist", "obs_target", "obs_mask"):
        if data[k].shape[0] != n: raise ValueError(f"{k} length mismatch")

def compute_pcs_from_sst_hist(sst_hist, n_pcs=10):
    N = sst_hist.shape[0]
    X = sst_hist.reshape(N, -1)
    X[~np.isfinite(X)] = 0.0
    pca = PCA(n_components=n_pcs)
    pcs = pca.fit_transform(X)
    pcs = (pcs - pcs.mean(0, keepdims=True)) / (pcs.std(0, keepdims=True) + 1e-8)
    return pcs.astype(np.float32)

def prepare_data():
    cfg = config.modelconfig
    cache_dir = cfg.get("seas2rain_cache_dir", os.path.join(cfg.get("lr_path", "."), "seas2rain_cache"))
    os.makedirs(cache_dir, exist_ok=True)
    modes_cache = os.path.join(cache_dir, f"modes_tp_cond_60x70_{_cond_vars_signature()}.npz")
    tp_raw, cond_raw, init_dates, lats, lons = read_modes_data(modes_cache, TARGET_HW, cfg.get("seas_nc_path"))
    
    init_dates_full = build_init_dates()
    allowed = set(init_dates_full)
    keep_mask = np.array([d in allowed for d in init_dates])
    tp_raw, cond_raw, init_dates = tp_raw[keep_mask], cond_raw[keep_mask], init_dates[keep_mask]
    
    sst_raw, sst_dates = read_ersst_data(cfg.get("ersst_dir", "D:\\ersst_data"), os.path.join(cache_dir, "ersst_ssta_cache.npz"))
    if sst_raw.shape[-1] == 0 or sst_raw.shape[-2] == 0:
        cache_file = os.path.join(cache_dir, "ersst_ssta_cache.npz")
        raise ValueError(f"SST data has zero spatial dimension {sst_raw.shape}.")
    sst_hist, keep_idx = build_sst_history(init_dates, sst_raw, sst_dates, int(cfg.get("sst_window", 12)))
    tp_raw, cond_raw, init_dates = tp_raw[keep_idx], cond_raw[keep_idx], init_dates[keep_idx]
    
    splits = split_indices_by_date(init_dates)
    seas_anom, _ = calc_precip_percent_anomaly(torch.from_numpy(tp_raw * float(cfg.get("tp_unit_scale", 1.0))), init_dates)
    seas_anom = seas_anom.cpu().numpy()[:, :, None, :, :]
    
    cond_mean, cond_std = compute_cond_stats(cond_raw, splits["train"])
    cond_norm = normalize_cond(cond_raw, cond_mean, cond_std)
    sst_hist, _, _ = normalize_sst_hist(sst_hist, splits["train"])
    
    obs_csv = cfg.get("observe_csv_path", os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "utils", "observe_data24.csv")))
    obs_grid, obs_mask, obs_dates = prepare_observe_data(obs_csv, os.path.join(cache_dir, "obs_cache.npz"), lons, lats)
    obs_target, obs_mask = build_obs_targets(init_dates, obs_grid, obs_mask, obs_dates)
    
    data = {
        "cond": cond_norm, "seas_anom": seas_anom, "sst_hist": sst_hist,
        "obs_target": obs_target[:, :, None, :, :], "obs_mask": obs_mask[:, :, None, :, :],
        "init_dates": init_dates, "split_indices": splits
    }
    validate_data_bundle(data)
    
    target_lead = int(cfg.get("target_lead", 0))
    n_pcs = int(cfg.get("n_pcs", 10))
    
    target_t = torch.from_numpy(data["obs_target"][:, target_lead])
    cond_t = torch.from_numpy(data["cond"][:, target_lead])
    mask_t = torch.from_numpy(data["obs_mask"][:, target_lead] < 0.5)
    pcs_t = torch.from_numpy(compute_pcs_from_sst_hist(data["sst_hist"], n_pcs))
    
    train_idx, test_idx = splits["train"], splits["test"]
    train_set = TensorDataset(target_t[train_idx], cond_t[train_idx], mask_t[train_idx], pcs_t[train_idx])
    test_set = TensorDataset(target_t[test_idx], cond_t[test_idx], mask_t[test_idx], pcs_t[test_idx])
    return train_set, test_set

def train():
    device = torch.device(config.modelconfig["device"])
    set_seed(int(config.modelconfig.get("seed", 42)))
    train_set, test_set = prepare_data()

    train_loader = DataLoader(train_set, batch_size=config.modelconfig["batch_size"], shuffle=True, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=config.modelconfig["batch_size"], shuffle=False, pin_memory=True)

    input_channels = train_set[0][1].shape[0]
    index_dim = train_set[0][3].shape[0]
    model = UNetLiteFiLM(n_channels=input_channels, n_classes=1, index_dim=index_dim, base_filters=16, dropout=config.modelconfig["dropout"]).to(device)

    if config.modelconfig.get("train_load_weight"):
        weight_path = os.path.join(config.modelconfig["save_weight_path"], config.modelconfig["train_load_weight"])
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.modelconfig["lr"], weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.modelconfig["epoch"], eta_min=0)
    writer = SummaryWriter(config.modelconfig["log_path"])

    for e in range(config.modelconfig["epoch"]):
        model.train()
        train_losses, train_accs = [], []
        for x_0, cond, mask, pcs in tqdm.tqdm(train_loader, desc=f"Train {e}"):
            x_0, cond, mask, pcs = x_0.to(device), cond.to(device), mask.to(device), pcs.to(device)
            optimizer.zero_grad()
            out = model(cond, pcs)
            valid = ~mask
            if valid.any():
                loss = loss_fn(out[valid], x_0[valid])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.modelconfig["grad_clip"])
                optimizer.step()
                train_losses.append(loss.item())
            
            acc = utils.cal_acc(x_0.squeeze(1)*100.0, out.squeeze(1)*100.0).mean()
            train_accs.append(acc.item())

        scheduler.step()
        model.eval()
        test_losses, test_accs = [], []
        with torch.no_grad():
            for x_0, cond, mask, pcs in tqdm.tqdm(test_loader, desc=f"Test {e}"):
                x_0, cond, mask, pcs = x_0.to(device), cond.to(device), mask.to(device), pcs.to(device)
                out = model(cond, pcs)
                valid = ~mask
                if valid.any():
                    loss = loss_fn(out[valid], x_0[valid])
                    test_losses.append(loss.item())
                acc = utils.cal_acc(x_0.squeeze(1)*100.0, out.squeeze(1)*100.0).mean()
                test_accs.append(acc.item())

        print(f"Epoch {e}: TrainLoss={np.mean(train_losses) if train_losses else 0:.4f}, TestLoss={np.mean(test_losses) if test_losses else 0:.4f}, TrainAcc={np.mean(train_accs):.3f}, TestAcc={np.mean(test_accs):.3f}")
        writer.add_scalar("Loss/train", np.mean(train_losses) if train_losses else 0, e)
        writer.add_scalar("Loss/test", np.mean(test_losses) if test_losses else 0, e)
        if e % 10 == 0:
            torch.save(model.state_dict(), os.path.join(config.modelconfig["save_weight_path"], f"epoch_{e}.pt"))
    writer.close()

if __name__ == "__main__":
    train()
