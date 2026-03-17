import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import griddata
from sklearn.decomposition import PCA
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
try:
    from HydroSynth.FNO import model1
except Exception:
    import model1  # fallback for direct script execution from HydroSynth/FNO

config.enable_auto_create_folders()


LEADS = 6
COND_VARS = ["h500", "slp", "t2m", "t850", "u850", "v850", "sst"]
OBS_GRID_LONS = np.arange(70.0, 140.0, 0.5, dtype=np.float32)  # 140
OBS_GRID_LATS = np.arange(60.0, 0.0, -0.5, dtype=np.float32)   # 120
OBS_GRID_LON2D, OBS_GRID_LAT2D = np.meshgrid(OBS_GRID_LONS, OBS_GRID_LATS)


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


def prepare_observe_data(
    csv_path: str,
    cache_path: str,
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
    grid_all = []
    mask_all = []
    for cur_month in tqdm.tqdm(all_months, desc="Interpolate observations"):
        cur = df[df["time"] == cur_month]
        if len(cur) < 3:
            grid_all.append(np.zeros((120, 140), dtype=np.float32))
            mask_all.append(np.zeros((120, 140), dtype=np.float32))
            continue

        points = cur[["Long", "Lat"]].to_numpy(dtype=np.float32)
        values = cur["anoma"].to_numpy(dtype=np.float32)

        try:
            linear = griddata(points, values, (OBS_GRID_LON2D, OBS_GRID_LAT2D), method="linear")
        except Exception:
            linear = np.full((120, 140), np.nan, dtype=np.float32)

        try:
            nearest = griddata(points, values, (OBS_GRID_LON2D, OBS_GRID_LAT2D), method="nearest")
        except Exception:
            nearest = np.zeros((120, 140), dtype=np.float32)

        valid = np.isfinite(linear)
        merged = np.where(valid, linear, nearest)
        merged = np.nan_to_num(merged, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        mask = valid.astype(np.float32)

        grid_all.append(merged)
        mask_all.append(mask)

    obs_grid = np.stack(grid_all, axis=0).astype(np.float32)
    obs_mask = np.stack(mask_all, axis=0).astype(np.float32)
    obs_dates = pd.DatetimeIndex(all_months)

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
    target = np.zeros((n, leads, 120, 140), dtype=np.float32)
    mask = np.zeros((n, leads, 120, 140), dtype=np.float32)

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


def compute_cond_stats(cond_raw: np.ndarray, train_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_var = cond_raw.shape[1]
    leads = cond_raw.shape[2]
    sum_x = np.zeros((n_var, leads), dtype=np.float64)
    sum_x2 = np.zeros((n_var, leads), dtype=np.float64)
    count = np.zeros((n_var, leads), dtype=np.float64)

    for idx in tqdm.tqdm(train_idx, desc="Cond stats"):
        x = cond_raw[int(idx)].astype(np.float32)  # [7,6,180,360]
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


def build_or_load_cond_global(
    cond_raw: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    n_samples: int,
    cache_path: str,
) -> np.ndarray:
    expected_shape = (n_samples, LEADS, len(COND_VARS) + 1, 180, 360)
    if os.path.exists(cache_path):
        cached = np.load(cache_path, mmap_mode="r")
        if tuple(cached.shape) == expected_shape:
            return cached

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    out = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float16,
        shape=expected_shape,
    )

    for i in tqdm.tqdm(range(n_samples), desc="Build cond_global cache"):
        x = cond_raw[i].astype(np.float32)  # [7,6,180,360]
        invalid = (~np.isfinite(x)) | (x <= -9000.0)
        sst_valid = (~invalid[6]).astype(np.float32)  # [6,180,360]

        x[invalid] = np.nan
        x = (x - mean[:, :, None, None]) / std[:, :, None, None]
        x = np.clip(x, -6.0, 6.0)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        x = np.concatenate([x, sst_valid[None, ...]], axis=0)  # [8,6,180,360]
        x = np.transpose(x, (1, 0, 2, 3))  # [6,8,180,360]
        out[i] = x.astype(np.float16)

    out.flush()
    del out
    return np.load(cache_path, mmap_mode="r")


def compute_sst_pcs(
    sst_raw: np.ndarray,
    train_idx: np.ndarray,
    n_pcs: int,
    cache_path: str,
) -> np.ndarray:
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached.shape[:2] == sst_raw.shape[:2] and cached.shape[2] == n_pcs:
            return cached.astype(np.float32)

    n, t, h, w = sst_raw.shape
    pcs = np.zeros((n, t, n_pcs), dtype=np.float32)

    min_valid_train = max(2, int(0.1 * len(train_idx)))
    for lead in tqdm.tqdm(range(t), desc="SST EOF PCs"):
        x = sst_raw[:, lead].reshape(n, h * w).astype(np.float32)
        invalid = (~np.isfinite(x)) | (x <= -9000.0)
        x[invalid] = np.nan

        x_train = x[train_idx]
        valid_cols = np.isfinite(x_train).sum(axis=0) >= min_valid_train
        if valid_cols.sum() < 2:
            continue

        x = x[:, valid_cols]
        x_train = x[train_idx]

        col_mean = np.nanmean(x_train, axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)

        nan_pos = ~np.isfinite(x)
        if nan_pos.any():
            x[nan_pos] = col_mean[np.where(nan_pos)[1]]

        max_comp = min(n_pcs, len(train_idx), x.shape[1])
        if max_comp < 1:
            continue

        pca = PCA(n_components=max_comp, svd_solver="randomized", random_state=42)
        pca.fit(x[train_idx])
        z = pca.transform(x)

        mu = z[train_idx].mean(axis=0, keepdims=True)
        sd = z[train_idx].std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-6, 1.0, sd)
        z = (z - mu) / sd

        pcs[:, lead, :max_comp] = z.astype(np.float32)

    np.save(cache_path, pcs.astype(np.float32))
    return pcs


def prepare_data() -> Dict[str, np.ndarray]:
    lr_path = config.modelconfig["lr_path"]
    cond_path = os.path.join(lr_path, "cond.npy")
    anomaly_path = os.path.join(lr_path, "anomaly.npy")
    sst_path = config.modelconfig["sst_file"]

    cond_raw = np.load(cond_path, mmap_mode="r")
    ec_anomaly = np.load(anomaly_path, mmap_mode="r")
    sst_raw = np.load(sst_path, mmap_mode="r")

    if cond_raw.ndim != 5 or cond_raw.shape[1:3] != (7, 6):
        raise ValueError(f"cond.npy shape must be [N,7,6,180,360], got {cond_raw.shape}")
    if ec_anomaly.ndim != 4 or ec_anomaly.shape[1] != 6:
        raise ValueError(f"anomaly.npy shape must be [N,6,120,140], got {ec_anomaly.shape}")
    if sst_raw.ndim != 4 or sst_raw.shape[1] != 6:
        raise ValueError(f"sst_file shape must be [N,6,H,W], got {sst_raw.shape}")

    init_dates_full = build_init_dates()
    n = min(cond_raw.shape[0], ec_anomaly.shape[0], sst_raw.shape[0], len(init_dates_full))
    if n < 300:
        raise ValueError(f"Aligned sample count too small: {n}")

    init_dates = pd.DatetimeIndex(init_dates_full[:n])
    cond_raw = cond_raw[:n]
    ec_anomaly = ec_anomaly[:n].astype(np.float32)
    sst_raw = sst_raw[:n].astype(np.float32)

    splits = split_indices_by_date(init_dates)

    obs_csv_path = config.modelconfig.get(
        "observe_csv_path",
        os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "utils", "observe_data24.csv")),
    )
    obs_cache_path = os.path.join(lr_path, "observe_grid_cache_199401_202412.npz")
    obs_grid, obs_month_mask, obs_dates = prepare_observe_data(
        csv_path=obs_csv_path,
        cache_path=obs_cache_path,
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

    ec_base = (ec_anomaly / 100.0)[:, :, None, :, :]      # [N,6,1,120,140]
    obs_target = (obs_target / 100.0)[:, :, None, :, :]   # [N,6,1,120,140]
    obs_mask = obs_mask[:, :, None, :, :].astype(np.float32)

    cond_mean, cond_std = compute_cond_stats(cond_raw=cond_raw, train_idx=splits["train"])
    cond_cache_path = os.path.join(lr_path, f"cond_global_norm_clip6_fp16_n{n}.npy")
    cond_global = build_or_load_cond_global(
        cond_raw=cond_raw,
        mean=cond_mean,
        std=cond_std,
        n_samples=n,
        cache_path=cond_cache_path,
    )  # [N,6,8,180,360], float16 memmap

    n_pcs = int(config.modelconfig["n_pcs"])
    pcs_cache_path = os.path.join(lr_path, f"sst_pcs_eof_k{n_pcs}_n{n}.npy")
    sst_pcs = compute_sst_pcs(
        sst_raw=sst_raw,
        train_idx=splits["train"],
        n_pcs=n_pcs,
        cache_path=pcs_cache_path,
    )  # [N,6,K]

    data = {
        "cond_global": cond_global,
        "ec_base": ec_base.astype(np.float32),
        "obs_target": obs_target.astype(np.float32),
        "obs_mask": obs_mask.astype(np.float32),
        "sst_pcs": sst_pcs.astype(np.float32),
        "init_dates": np.array([d.strftime("%Y-%m-%d") for d in init_dates]),
        "split_indices": splits,
        "cond_mean": cond_mean,
        "cond_std": cond_std,
    }
    validate_data_bundle(data)
    return data


def validate_data_bundle(data: Dict[str, np.ndarray]) -> None:
    n = len(data["init_dates"])
    for k in ("cond_global", "ec_base", "obs_target", "obs_mask", "sst_pcs"):
        if data[k].shape[0] != n:
            raise ValueError(f"{k} length mismatch: {data[k].shape[0]} vs {n}")

    if tuple(data["cond_global"].shape[1:]) != (6, 8, 180, 360):
        raise ValueError(f"cond_global shape invalid: {data['cond_global'].shape}")
    if tuple(data["ec_base"].shape[1:]) != (6, 1, 120, 140):
        raise ValueError(f"ec_base shape invalid: {data['ec_base'].shape}")
    if tuple(data["obs_target"].shape[1:]) != (6, 1, 120, 140):
        raise ValueError(f"obs_target shape invalid: {data['obs_target'].shape}")
    if tuple(data["obs_mask"].shape[1:]) != (6, 1, 120, 140):
        raise ValueError(f"obs_mask shape invalid: {data['obs_mask'].shape}")

    init_dates = pd.to_datetime(data["init_dates"])
    forbidden = {pd.Timestamp("2011-09-01"), pd.Timestamp("2011-10-01")}
    if any(d in forbidden for d in init_dates):
        raise ValueError("init_dates still include 2011-09/2011-10, which must be excluded.")

    # Fast finite checks on slices.
    idx = [0, len(init_dates) // 2, len(init_dates) - 1]
    for i in idx:
        cg = np.asarray(data["cond_global"][i], dtype=np.float32)
        if not np.isfinite(cg).all():
            raise ValueError(f"Non-finite values found in cond_global sample {i}")
        if not np.isfinite(data["ec_base"][i]).all():
            raise ValueError(f"Non-finite values found in ec_base sample {i}")
        if not np.isfinite(data["sst_pcs"][i]).all():
            raise ValueError(f"Non-finite values found in sst_pcs sample {i}")


class Hydro6LeadDataset(Dataset):
    def __init__(self, data: Dict[str, np.ndarray], indices: np.ndarray):
        self.cond_global = data["cond_global"]
        self.ec_base = data["ec_base"]
        self.obs_target = data["obs_target"]
        self.obs_mask = data["obs_mask"]
        self.sst_pcs = data["sst_pcs"]
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        cond = torch.from_numpy(np.asarray(self.cond_global[idx], dtype=np.float32))
        ec_base = torch.from_numpy(self.ec_base[idx])
        obs_target = torch.from_numpy(self.obs_target[idx])
        obs_mask = torch.from_numpy(self.obs_mask[idx])
        sst_pcs = torch.from_numpy(self.sst_pcs[idx])
        return cond, ec_base, obs_target, obs_mask, sst_pcs


def build_dataloaders(data: Dict[str, np.ndarray], device: torch.device):
    train_ds = Hydro6LeadDataset(data, data["split_indices"]["train"])
    val_ds = Hydro6LeadDataset(data, data["split_indices"]["val"])
    test_ds = Hydro6LeadDataset(data, data["split_indices"]["test"])

    batch_size = int(config.modelconfig["batch_size"])
    num_workers = int(config.modelconfig["num_workers"])
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


def build_model(data: Dict[str, np.ndarray], device: torch.device) -> model1.GlobalResidualUNet6:
    cond_channels = int(data["cond_global"].shape[2])
    pcs_dim = int(data["sst_pcs"].shape[2])
    model = model1.GlobalResidualUNet6(
        cond_channels=cond_channels,
        leads=LEADS,
        pcs_dim=pcs_dim,
        channels=tuple(config.modelconfig["channels"]),
        lead_embed_dim=int(config.modelconfig["lead_embed_dim"]),
        global_dim=int(config.modelconfig["global_dim"]),
    ).to(device)
    return model


def maybe_load_weights(model: torch.nn.Module, device: torch.device) -> None:
    # weight_name = "epoch_500.pt"
    weight_name = None
    if not weight_name:
        return
    load_dir = r"D:\workplace\conv_data\weight_t0\run_20260316_184706"
    ckpt_path = os.path.join(load_dir, weight_name)
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    else:
        state_dict = ckpt.state_dict()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"Loaded checkpoint: {ckpt_path}\n"
        f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
    )


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    ec_base: torch.Tensor,
    obs_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    # pred: [B,6,120,140]
    # target/ec_base/obs_mask: [B,6,1,120,140]
    target2 = target[:, :, 0]
    ec2 = ec_base[:, :, 0]
    mask2 = obs_mask[:, :, 0] > 0.5
    valid_count = int(mask2.sum().item())
    if valid_count == 0:
        zero = pred.new_tensor(0.0)
        return zero, zero, zero, 0

    huber = model1.masked_huber_loss(pred, target2, mask2, delta=1.0)
    mse_res = model1.masked_mse_loss(pred - ec2, target2 - ec2, mask2)
    loss = huber + 0.3 * mse_res
    return loss, huber, mse_res, valid_count


def init_metric_state(leads: int = LEADS) -> Dict[str, np.ndarray]:
    return {
        "mae_sum": np.zeros(leads, dtype=np.float64),
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
    # pred/target/mask: [B,6,120,140], mask bool
    bsz, leads, _, _ = pred.shape
    err = pred - target
    abs_err = torch.abs(err)
    sq_err = err * err

    for t in range(leads):
        mt = mask[:, t]
        count = float(mt.sum().item())
        if count <= 0:
            continue
        state["mae_sum"][t] += float(abs_err[:, t][mt].sum().item())
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
    mae = state["mae_sum"] / np.maximum(state["count"], eps)
    rmse = np.sqrt(state["mse_sum"] / np.maximum(state["count"], eps))
    acc = state["acc_sum"] / np.maximum(state["acc_cnt"], 1.0)
    acc[state["acc_cnt"] < 1] = np.nan
    return {"mae": mae, "rmse": rmse, "acc": acc}


def evaluate_baseline(loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    state = init_metric_state()
    with torch.no_grad():
        for _, ec_base, target, obs_mask, _ in loader:
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
    seed = int(config.modelconfig["seed"])
    set_seed(seed)

    device = config.modelconfig["device"]
    data = prepare_data()
    train_loader, val_loader, test_loader = build_dataloaders(data, device=device)

    model = build_model(data, device=device)
    maybe_load_weights(model, device=device)

    lr = float(config.modelconfig["lr"])
    weight_decay = float(config.modelconfig["weight_decay"])
    epochs = int(config.modelconfig["epoch"])
    grad_accum = int(config.modelconfig["grad_accum"])
    grad_clip = float(config.modelconfig["grad_clip"])
    save_every = int(config.modelconfig["save_every"])
    patience = int(config.modelconfig["patience"])
    min_delta = float(config.modelconfig["early_stop_min_delta"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    writer = SummaryWriter(log_dir=config.modelconfig["log_path"])

    print(
        "Train setup:",
        {
            "device": str(device),
            "batch_size": train_loader.batch_size,
            "epochs": epochs,
            "grad_accum": grad_accum,
            "lr": lr,
            "weight_decay": weight_decay,
        },
    )

    baseline_val = evaluate_baseline(val_loader, device=device)
    baseline_test = evaluate_baseline(test_loader, device=device)
    print(format_metric_line("Baseline VAL RMSE", baseline_val["rmse"]))
    print(format_metric_line("Baseline TEST RMSE", baseline_test["rmse"]))

    best_val_loss = float("inf")
    best_epoch = -1
    stale_epochs = 0

    global_step = 0
    for epoch in range(epochs):
        model.train()
        train_losses: List[float] = []
        train_state = init_metric_state()
        skipped_batches = 0

        optimizer.zero_grad(set_to_none=True)
        accum_counter = 0
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}")
        for step, (cond, ec_base, target, obs_mask, sst_pcs) in enumerate(pbar):
            cond = cond.to(device, non_blocking=True)
            ec_base = ec_base.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            obs_mask = obs_mask.to(device, non_blocking=True)
            sst_pcs = sst_pcs.to(device, non_blocking=True)

            pred = model(cond, ec_base=ec_base, sst_pcs=sst_pcs)
            loss, huber, mse_res, valid_count = compute_loss(pred, target, ec_base, obs_mask)

            if valid_count == 0:
                skipped_batches += 1
                continue

            if (not torch.isfinite(pred).all().item()) or (not torch.isfinite(loss).item()):
                raise FloatingPointError(
                    "Non-finite values detected in training. "
                    "Check input normalization."
                )

            scaled_loss = loss / max(1, grad_accum)
            scaled_loss.backward()
            accum_counter += 1

            do_step = accum_counter >= max(1, grad_accum)
            if do_step:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0

            train_losses.append(float(loss.item()))
            pred_det = pred.detach()
            target_det = target[:, :, 0].detach()
            mask_det = (obs_mask[:, :, 0] > 0.5).detach()
            update_metrics(train_state, pred_det, target_det, mask_det)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                huber=f"{huber.item():.4f}",
                mse=f"{mse_res.item():.4f}",
                skipped=skipped_batches,
            )
            global_step += 1

        if accum_counter > 0:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_metrics = finalize_metrics(train_state)
        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")

        model.eval()
        val_losses: List[float] = []
        val_state = init_metric_state()
        with torch.no_grad():
            for cond, ec_base, target, obs_mask, sst_pcs in val_loader:
                cond = cond.to(device, non_blocking=True)
                ec_base = ec_base.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                obs_mask = obs_mask.to(device, non_blocking=True)
                sst_pcs = sst_pcs.to(device, non_blocking=True)

                pred = model(cond, ec_base=ec_base, sst_pcs=sst_pcs)
                loss, _, _, valid_count = compute_loss(pred, target, ec_base, obs_mask)

                if valid_count == 0:
                    continue
                if (not torch.isfinite(pred).all().item()) or (not torch.isfinite(loss).item()):
                    raise FloatingPointError(
                        "Non-finite values detected in validation. "
                        "Check input normalization."
                    )

                val_losses.append(float(loss.item()))
                update_metrics(val_state, pred, target[:, :, 0], obs_mask[:, :, 0] > 0.5)

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
        print(format_metric_line("VAL RMSE Skill (baseline-model)", skill)) # skill > 0 means model is better than baseline
        print(format_metric_line("VAL ACC Skill (baseline-model)", acc_diff)) # acc_diff > 0 means model is better than baseline

        if epoch % max(1, save_every) == 0:
            ckpt_path = os.path.join(config.modelconfig["save_weight_path"], f"epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
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
                    "model_state_dict": model.state_dict(),
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

    # Final test eval with best checkpoint.
    best_path = os.path.join(config.modelconfig["save_weight_path"], "best.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    model.eval()
    test_losses: List[float] = []
    test_state = init_metric_state()
    with torch.no_grad():
        for cond, ec_base, target, obs_mask, sst_pcs in test_loader:
            cond = cond.to(device, non_blocking=True)
            ec_base = ec_base.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            obs_mask = obs_mask.to(device, non_blocking=True)
            sst_pcs = sst_pcs.to(device, non_blocking=True)

            pred = model(cond, ec_base=ec_base, sst_pcs=sst_pcs)
            loss, _, _, valid_count = compute_loss(pred, target, ec_base, obs_mask)
            if valid_count == 0:
                continue
            test_losses.append(float(loss.item()))
            update_metrics(test_state, pred, target[:, :, 0], obs_mask[:, :, 0] > 0.5)

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
