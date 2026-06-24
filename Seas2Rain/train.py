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

config.enable_auto_create_folders(False)

LEADS = 6
COND_VARS = ["h500", "slp", "t850", "u850", "v850", "sst"]
TARGET_HW = (60, 70)
MODES_CACHE_VERSION = "20260529_dtd_v2"

def _cond_vars_signature() -> str:
    return "_".join(COND_VARS)
BATCH_KEYS = ("cond", "seas_anom", "ec_base", "obs_target", "obs_mask", "sst_hist", "init_month")

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
    if not match: return None
    yyyymm = match.group(1)
    return pd.Timestamp(year=int(yyyymm[:4]), month=int(yyyymm[4:]), day=1)

def _parse_ersst_date(path: str) -> Optional[pd.Timestamp]:
    name = os.path.basename(path)
    match = re.search(r"(\d{6})", name)
    if not match: return None
    yyyymm = match.group(1)
    return pd.Timestamp(year=int(yyyymm[:4]), month=int(yyyymm[4:]), day=1)

def _get_lat_lon(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    if "latitude" in ds.coords:
        lat = ds.coords["latitude"].to_numpy()
    elif "lat" in ds.coords:
        lat = ds.coords["lat"].to_numpy()
    else: raise KeyError("Latitude coordinate not found in dataset.")

    if "longitude" in ds.coords:
        lon = ds.coords["longitude"].to_numpy()
    elif "lon" in ds.coords:
        lon = ds.coords["lon"].to_numpy()
    else: raise KeyError("Longitude coordinate not found in dataset.")

    return lat, lon

def _build_target_grid(lat: np.ndarray, lon: np.ndarray, target_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    if len(lat) == target_hw[0] and len(lon) == target_hw[1]: return lat.astype(np.float32), lon.astype(np.float32)
    lat_min, lat_max = float(lat.min()), float(lat.max())
    lon_min, lon_max = float(lon.min()), float(lon.max())
    if lat[0] > lat[-1]: grid_lats = np.linspace(lat_max, lat_min, target_hw[0], dtype=np.float32)
    else: grid_lats = np.linspace(lat_min, lat_max, target_hw[0], dtype=np.float32)
    grid_lons = np.linspace(lon_min, lon_max, target_hw[1], dtype=np.float32)
    return grid_lats, grid_lons

def read_modes_data(cache_path: str, target_hw: Tuple[int, int] = TARGET_HW, seas_nc_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        tp_raw = cached["tp_raw"].astype(np.float32)
        cond_raw = cached["cond_raw"].astype(np.float32)
        init_dates = pd.to_datetime(cached["init_dates"].astype(str))
        grid_lats = cached["grid_lats"].astype(np.float32)
        grid_lons = cached["grid_lons"].astype(np.float32)
        cached_cond_vars = cached["cond_vars"].tolist() if "cond_vars" in cached.files else None
        cached_version = str(cached["cache_version"]) if "cache_version" in cached.files else None
        if (cond_raw.ndim == 5 and cond_raw.shape[2] == len(COND_VARS) and tp_raw.shape[-2:] == target_hw 
            and tuple(grid_lats.shape) == (target_hw[0],) and tuple(grid_lons.shape) == (target_hw[1],) 
            and cached_cond_vars == COND_VARS and cached_version == MODES_CACHE_VERSION):
            return tp_raw, cond_raw, pd.DatetimeIndex(init_dates), grid_lats, grid_lons

    file_list = utils.read_nc_to_npy(199401, 202409, data_path=seas_nc_path)
    tp_list, cond_list, date_list = [], [], []
    grid_lats, grid_lons = None, None
    forbidden = {pd.Timestamp("2011-09-01"), pd.Timestamp("2011-10-01")}

    for f in tqdm.tqdm(file_list, desc="Read MODESv21"):
        date = _parse_date_from_path(f)
        if date is None or date in forbidden: continue
        try:
            with xr.open_dataset(f) as ds:
                if "longitude" in ds.coords: ds = ds.rename({"longitude": "lon", "latitude": "lat"})
                tp_ds = ds[["tp"]].sel(lon=slice(70, 140), lat=slice(60, 0))
                lat, lon = _get_lat_lon(tp_ds)
                if grid_lats is None or grid_lons is None: grid_lats, grid_lons = _build_target_grid(lat, lon, target_hw)
                tp_arr = tp_ds.interp(lat=grid_lats, lon=grid_lons, method="linear").to_array().to_numpy().astype(np.float32) 
                v_arrays = []
                for v in COND_VARS:
                    v_ds = ds[[v]].sel(lon=slice(60, 180), lat=slice(60, -30))
                    v_arrays.append(v_ds.interp(lat=grid_lats, lon=grid_lons, method="linear")[v].to_numpy())
                cond_arr = np.stack(v_arrays, axis=0).astype(np.float32) 
                tp_list.append(np.nan_to_num(tp_arr, nan=0.0).squeeze(0))  
                cond_list.append(np.nan_to_num(cond_arr, nan=0.0))  
                date_list.append(date)
        except Exception as e:
            print(f"Skip file {f}: {e}")
            continue

    tp_raw, cond_raw = np.stack(tp_list, axis=0), np.transpose(np.stack(cond_list, axis=0), (0, 2, 1, 3, 4))  
    init_dates = pd.DatetimeIndex(date_list)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, tp_raw=tp_raw, cond_raw=cond_raw, init_dates=np.array([d.strftime("%Y-%m-%d") for d in init_dates]), grid_lats=grid_lats, grid_lons=grid_lons, cond_vars=np.array(COND_VARS, dtype="<U32"), cache_version=np.array(MODES_CACHE_VERSION))
    return tp_raw, cond_raw, init_dates, grid_lats, grid_lons

def read_ersst_data(ersst_dir: str, cache_path: str) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        return cached["ssta"].astype(np.float32), pd.to_datetime(cached["dates"].astype(str))
    files, sst_list, date_list, ref_shape = sorted(glob.glob(os.path.join(ersst_dir, "ersst.v5.*.nc"))), [], [], None
    for f in tqdm.tqdm(files, desc="Read ERSST ssta"):
        date = _parse_ersst_date(f)
        if date is None: continue
        try:
            with xr.open_dataset(f) as ds:
                if "longitude" in ds.coords: ds = ds.rename({"longitude": "lon", "latitude": "lat"})
                lat_slice = slice(30, -30) if ds.lat[0] > ds.lat[-1] else slice(-30, 30)
                ds_ssta = ds.sel(lon=slice(30, 290), lat=lat_slice)
                arr = np.asarray(ds_ssta["ssta"].to_numpy())
                if arr.ndim == 4: arr = arr[0, 0]
                elif arr.ndim == 3: arr = arr[0]
                if ref_shape is None: ref_shape = arr.shape
                elif arr.shape != ref_shape:
                    arr = ds_ssta["ssta"].interp(lat=np.linspace(lat_slice.start, lat_slice.stop, ref_shape[0]), lon=np.linspace(30, 290, ref_shape[1])).to_numpy().squeeze()
                sst_list.append(np.nan_to_num(arr, nan=0.0).astype(np.float32))
                date_list.append(date)
        except Exception: continue
    sst, dates = np.stack(sst_list, axis=0), pd.DatetimeIndex(date_list)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, ssta=sst, dates=np.array([d.strftime("%Y-%m-%d") for d in dates]))
    return sst, dates

def build_sst_history(init_dates, sst, sst_dates, window):
    sst_index = {_month_start(d): i for i, d in enumerate(sst_dates)}
    keep, hist = [], []
    for i, d in enumerate(init_dates):
        seq, ok = [], True
        for m in range(window, 0, -1):
            idx = sst_index.get(_month_start(d - pd.DateOffset(months=m)))
            if idx is None: ok = False; break
            seq.append(sst[idx])
        if ok: keep.append(i); hist.append(np.stack(seq, axis=0))
    return np.stack(hist, axis=0), np.asarray(keep, dtype=np.int64)

def normalize_sst_hist(sst_hist, train_idx):
    train_vals = sst_hist[train_idx]
    mean, std = float(train_vals.mean()), max(float(train_vals.std()), 1e-6)
    return (sst_hist - mean) / std, mean, std

def calc_precip_percent_anomaly(tp_tensor, init_dates, eps=1e-6):
    T, L, H, W = tp_tensor.shape
    target_month_idx = torch.empty((T, L), dtype=torch.long)
    for t, d in enumerate(init_dates):
        for l in range(L): target_month_idx[t, l] = (d + pd.DateOffset(months=l)).month
    pa = torch.zeros_like(tp_tensor)
    climatology = torch.zeros((L, 12, H, W), dtype=tp_tensor.dtype, device=tp_tensor.device)
    for l in range(L):
        for m in range(1, 13):
            idx_t = (target_month_idx[:, l] == m).nonzero(as_tuple=True)[0]
            if idx_t.numel() == 0: continue
            clim = tp_tensor[idx_t, l].mean(dim=0)
            climatology[l, m - 1] = clim
            pa[idx_t, l] = (tp_tensor[idx_t, l] - clim) / (clim + eps)
    return pa, climatology

def compute_cond_stats(cond_raw, train_idx):
    sum_x = np.zeros((cond_raw.shape[1], cond_raw.shape[2]), dtype=np.float64)
    sum_x2 = np.zeros_like(sum_x)
    count = np.zeros_like(sum_x)
    for idx in tqdm.tqdm(train_idx, desc="Cond stats"):
        x = cond_raw[int(idx)].astype(np.float32)
        valid = np.isfinite(x) & (x > -9000.0)
        xv = np.where(valid, x, 0.0)
        sum_x += xv.sum(axis=(2, 3)); sum_x2 += (xv * xv).sum(axis=(2, 3)); count += valid.sum(axis=(2, 3))
    count = np.maximum(count, 1.0)
    mean, std = sum_x / count, np.sqrt(np.maximum(sum_x2 / count - (sum_x / count) ** 2, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)

def normalize_cond(cond_raw, mean, std):
    return np.nan_to_num(np.clip((cond_raw - mean[None, :, :, None, None]) / std[None, :, :, None, None], -6.0, 6.0), nan=0.0).astype(np.float32)

def prepare_observe_data(csv_path, cache_path, grid_lons, grid_lats):
    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=False)
        return cached["obs_grid"], cached["obs_mask"], pd.to_datetime(cached["obs_dates"].astype(str))
    df = _normalize_station_coords(pd.read_csv(csv_path))
    df["time"] = pd.to_datetime(df["time"]).apply(_month_start)
    all_months = pd.date_range(start="1994-01-01", end="2024-12-01", freq="MS")
    grid_lon2d, grid_lat2d = np.meshgrid(grid_lons, grid_lats)
    grid_all, mask_all = [], []
    for cur_month in tqdm.tqdm(all_months, desc="Interpolate observations"):
        cur = df[df["time"] == cur_month]
        if len(cur) < 3:
            grid_all.append(np.zeros(TARGET_HW, dtype=np.float32)); mask_all.append(np.zeros(TARGET_HW, dtype=np.float32)); continue
        p, v = cur[["Long", "Lat"]].to_numpy(dtype=np.float32), cur["anoma"].to_numpy(dtype=np.float32)
        try:
            linear = griddata(p, v, (grid_lon2d, grid_lat2d), method="linear")
            nearest = griddata(p, v, (grid_lon2d, grid_lat2d), method="nearest")
            valid = np.isfinite(linear)
            grid_all.append(np.nan_to_num(np.where(valid, linear, nearest), nan=0.0)); mask_all.append(valid.astype(np.float32))
        except Exception:
            grid_all.append(np.zeros(TARGET_HW, dtype=np.float32)); mask_all.append(np.zeros(TARGET_HW, dtype=np.float32))
    obs_grid, obs_mask = np.stack(grid_all), np.stack(mask_all)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, obs_grid=obs_grid, obs_mask=obs_mask, obs_dates=np.array([d.strftime("%Y-%m-%d") for d in all_months]))
    return obs_grid, obs_mask, all_months

def build_obs_targets(init_dates, obs_grid, obs_mask, obs_dates):
    target, mask = np.zeros((len(init_dates), LEADS, *TARGET_HW), dtype=np.float32), np.zeros((len(init_dates), LEADS, *TARGET_HW), dtype=np.float32)
    obs_index = {_month_start(d): i for i, d in enumerate(obs_dates)}
    for i, init_d in enumerate(init_dates):
        for l in range(LEADS):
            j = obs_index.get(_month_start(init_d + pd.DateOffset(months=l)))
            if j is not None: target[i, l], mask[i, l] = obs_grid[j], obs_mask[j]
    return target, mask

def prepare_data():
    cfg = config.modelconfig
    cache_dir = cfg.get("seas2rain_cache_dir", os.path.join(cfg["lr_path"], "seas2rain_cache"))
    os.makedirs(cache_dir, exist_ok=True)
    tp_raw, cond_raw, init_dates, grid_lats, grid_lons = read_modes_data(os.path.join(cache_dir, f"modes_tp_cond_60x70_{_cond_vars_signature()}.npz"), TARGET_HW, cfg.get("seas_nc_path"))
    ersst_dir = cfg.get("ersst_dir", r"D:\ersst_data")
    sst_raw, sst_dates = read_ersst_data(ersst_dir, os.path.join(cache_dir, "ersst_ssta_cache_expanded.npz"))
    sst_hist, keep_idx = build_sst_history(init_dates, sst_raw, sst_dates, int(cfg.get("sst_window", 12)))
    tp_raw, cond_raw, init_dates = tp_raw[keep_idx], cond_raw[keep_idx], init_dates[keep_idx]
    splits = split_indices_by_date(init_dates)
    seas_anom, _ = calc_precip_percent_anomaly(torch.from_numpy(tp_raw), init_dates)
    cond_mean, cond_std = compute_cond_stats(cond_raw, splits["train"])
    obs_grid, obs_mask_raw, obs_dates = prepare_observe_data(cfg.get("observe_csv_path", os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "utils", "observe_data24.csv"))), os.path.join(cache_dir, "observe_grid_cache_60x70.npz"), grid_lons, grid_lats)
    obs_target, obs_mask = build_obs_targets(init_dates, obs_grid, obs_mask_raw, obs_dates)
    sst_hist, _, _ = normalize_sst_hist(sst_hist, splits["train"])
    return {
        "cond": normalize_cond(cond_raw, cond_mean, cond_std), "seas_anom": seas_anom.cpu().numpy()[:, :, None], "ec_base": seas_anom.cpu().numpy()[:, :, None],
        "sst_hist": sst_hist, "obs_target": obs_target[:, :, None], "obs_mask": obs_mask[:, :, None],
        "init_dates": np.array([d.strftime("%Y-%m-%d") for d in init_dates]), "init_month": init_dates.month.values.astype(np.int64), "split_indices": splits,
    }

@dataclass
class TensorBatchStore:
    tensors: Dict[str, torch.Tensor]; split_indices: Dict[str, torch.Tensor]; batch_size: int; train_device: torch.device
    def split_size(self, split: str) -> int: return int(self.split_indices[split].shape[0])

def build_batch_store(data, train_device):
    tensors = {k: torch.from_numpy(data[k]).to(torch.device("cpu")).pin_memory() for k in BATCH_KEYS}
    split_indices = {k: torch.as_tensor(v, dtype=torch.long) for k, v in data["split_indices"].items()}
    return TensorBatchStore(tensors, split_indices, int(config.modelconfig.get("batch_size", 8)), train_device)

def iter_split_batches(batch_store, split, shuffle=False):
    indices = batch_store.split_indices[split]
    if shuffle: indices = indices[torch.randperm(indices.size(0))]
    for start in range(0, indices.size(0), batch_store.batch_size):
        idx = indices[start : start + batch_store.batch_size]
        yield {k: v[idx].to(batch_store.train_device, non_blocking=True) for k, v in batch_store.tensors.items()}

def build_model(device):
    cfg = config.modelconfig
    return model.Seas2RainModel(
        cond_channels=len(COND_VARS), 
        hidden_dim=int(cfg.get("seas2rain_hidden_dim", 32)), 
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)

def acc_hybrid_loss(pred, target, mask, mse_weight=0.1):
    m = mask.float()
    count = m.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    
    # Calculate ACC
    p_mean = (pred * m).sum(dim=(2, 3), keepdim=True) / count
    t_mean = (target * m).sum(dim=(2, 3), keepdim=True) / count
    
    pa = (pred - p_mean) * m
    ta = (target - t_mean) * m
    
    cov = (pa * ta).sum(dim=(2, 3))
    p_var = (pa ** 2).sum(dim=(2, 3))
    t_var = (ta ** 2).sum(dim=(2, 3))
    
    acc = cov / torch.sqrt(p_var * t_var + 1e-6)
    
    # We want to minimize (1 - ACC)
    acc_loss = (1.0 - acc).mean()
    
    # Calculate MSE to anchor the scale
    mse_loss = ((pred - target)**2 * m).sum() / m.sum().clamp_min(1.0)
    
    return acc_loss + mse_weight * mse_loss

def init_metric_state(leads=LEADS, device=None):
    return {k: torch.zeros(leads, device=device) for k in ["mse_sum", "count", "acc_sum", "acc_cnt"]}

def update_metrics(state, pred, target, mask):
    m = mask.float(); state["mse_sum"] += ((pred - target)**2 * m).sum(dim=(0, 2, 3)); state["count"] += m.sum(dim=(0, 2, 3))
    count = m.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    pa, ta = (pred - (pred*m).sum(dim=(2, 3), keepdim=True)/count)*m, (target - (target*m).sum(dim=(2, 3), keepdim=True)/count)*m
    acc = (pa*ta).sum(dim=(2, 3)) / (torch.sqrt((pa**2).sum(dim=(2, 3))* (ta**2).sum(dim=(2, 3)) + 1e-12))
    state["acc_sum"] += acc.sum(dim=0); state["acc_cnt"] += (m.sum(dim=(2, 3)) > 0).float().sum(dim=0)

def finalize_metrics(state):
    return {"rmse": torch.sqrt(state["mse_sum"] / state["count"].clamp_min(1.0)).cpu().numpy(),
            "acc": (state["acc_sum"] / state["acc_cnt"].clamp_min(1.0)).cpu().numpy()}

def train():
    print("Starting DTD training pipeline (ACC-Driven Model)...")
    set_seed(int(config.modelconfig.get("seed", 42)))
    device = config.modelconfig["device"]
    batch_store = build_batch_store(prepare_data(), device)
    net = build_model(device)
    
    lr = 5e-4 # Slightly higher LR for ACC loss
    epochs = int(config.modelconfig.get("epoch", 50))
    
    # Strong regularization
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    writer = SummaryWriter(log_dir=config.modelconfig["log_path"])
    best_val_acc = -1.0
    
    base_state = init_metric_state(device=device)
    for b in iter_split_batches(batch_store, "val"):
        update_metrics(base_state, b["ec_base"][:,:,0], b["obs_target"][:,:,0], b["obs_mask"][:,:,0])
    base_metrics = finalize_metrics(base_state)
    print(f"Baseline (ECMWF) VAL ACC: {base_metrics['acc']}")

    for epoch in range(epochs):
        net.train(); train_state = init_metric_state(device=device)
        pbar = tqdm.tqdm(iter_split_batches(batch_store, "train", True), desc=f"Epoch {epoch}")
        for b in pbar:
            pred = net(b["cond"], b["seas_anom"], b["ec_base"], b["sst_hist"], b["init_month"])
            loss = acc_hybrid_loss(pred, b["obs_target"][:,:,0], b["obs_mask"][:,:,0], mse_weight=0.05)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            update_metrics(train_state, pred.detach(), b["obs_target"][:,:,0], b["obs_mask"][:,:,0])
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        scheduler.step()
        
        net.eval(); val_state = init_metric_state(device=device)
        with torch.no_grad():
            for b in iter_split_batches(batch_store, "val"):
                pred = net(b["cond"], b["seas_anom"], b["ec_base"], b["sst_hist"], b["init_month"])
                update_metrics(val_state, pred, b["obs_target"][:,:,0], b["obs_mask"][:,:,0])
        metrics = finalize_metrics(val_state); avg_val_acc = metrics['acc'].mean()
        print(f"Epoch {epoch} VAL ACC: {metrics['acc']}")
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(net.state_dict(), os.path.join(config.modelconfig["save_weight_path"], "best.pt"))
    writer.close()

if __name__ == "__main__":
    train()
