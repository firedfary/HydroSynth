"""Rebuild SEAS5 predictors with preprocessing fitted only on training months."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.paths import SubprojectPaths
from run_unified_eval import build_date_index


VARIABLES = ("tp", "h200", "h500", "slp", "t2m", "t850", "u200", "u850", "v200", "v850")


def transform_train_only(fields, months, train_mask):
    """Apply the former transform order while fitting every statistic on train."""
    if fields.ndim != 4 or fields.shape[1] != len(VARIABLES):
        raise ValueError("fields must be [time, 10, height, width]")
    if len(months) != len(fields) or len(train_mask) != len(fields):
        raise ValueError("time masks must match fields")
    if not np.isfinite(fields).all():
        raise ValueError("SEAS5 fields contain non-finite values")
    if not np.all(np.isin(np.arange(1, 13), months[train_mask])):
        raise ValueError("training period must include all calendar months")

    result = fields.copy()
    result[:, 0] *= 31 * 24 * 60 * 60 * 1000
    monthly_clim = np.stack([
        result[train_mask & (months == month), 0].mean(axis=0)
        for month in range(1, 13)
    ])
    anomaly = (result[:, 0] - monthly_clim[months - 1]) / (monthly_clim[months - 1] + 1e-6)
    train_anomaly = anomaly[train_mask]
    anomaly_mean = float(train_anomaly.mean(dtype=np.float64))
    anomaly_std = float(train_anomaly.std(dtype=np.float64))
    if anomaly_std <= 0:
        raise ValueError("training precipitation anomaly has zero variance")
    result[:, 0] = np.clip((anomaly - anomaly_mean) / (anomaly_std + 1e-8), -3, 3) / 3

    channel_min = result[train_mask].min(axis=(0, 2, 3), keepdims=True)
    channel_max = result[train_mask].max(axis=(0, 2, 3), keepdims=True)
    if np.any(channel_max <= channel_min):
        raise ValueError("a training channel has zero range")
    normalized = 2 * (result - channel_min) / (channel_max - channel_min + 1e-8) - 1
    params = {
        "monthly_tp_climatology": monthly_clim.astype(np.float32),
        "tp_anomaly_mean": np.float64(anomaly_mean),
        "tp_anomaly_std": np.float64(anomaly_std),
        "channel_min": channel_min.astype(np.float32),
        "channel_max": channel_max.astype(np.float32),
    }
    return normalized.astype(np.float32), params


def load_raw_fields(raw_dir, dates):
    fields = []
    for index, date in enumerate(dates):
        path = raw_dir / f"MODESv21_ecmwf_seas51_{date:%Y%m}_monthly_em.nc"
        if not path.is_file():
            raise FileNotFoundError(path)
        with xr.open_dataset(path) as ds:
            actual = pd.Timestamp(ds.time.values[0])
            if actual.year != date.year or actual.month != date.month:
                raise ValueError(f"time mismatch in {path}: {actual}")
            field = ds[list(VARIABLES)].sel(
                longitude=slice(70, 140), latitude=slice(60, 0)
            ).isel(time=0).to_array().values.astype(np.float32)
        if field.shape != (10, 60, 70):
            raise ValueError(f"unexpected spatial shape in {path}: {field.shape}")
        fields.append(field)
        if (index + 1) % 50 == 0:
            print(f"Read {index + 1}/{len(dates)} monthly files", flush=True)
    raw = torch.from_numpy(np.stack(fields))
    return F.interpolate(raw, scale_factor=2.0, mode="bicubic").numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--fit-end", default="2017-12-01")
    parser.add_argument("--validation-end", default="2022-12-01")
    parser.add_argument("--cache-name", default="strict_inputs")
    args = parser.parse_args()
    dates = build_date_index()
    if (args.cache_name in ("", ".", "..") or
            "/" in args.cache_name or "\\" in args.cache_name or
            pd.Timestamp(args.fit_end) >= pd.Timestamp(args.validation_end)):
        parser.error("cache-name must be a directory name and fit-end must precede validation-end")
    train_mask = np.asarray(dates <= args.fit_end)
    val_mask = np.asarray((dates > args.fit_end) & (dates <= args.validation_end))
    test_mask = np.asarray(dates > args.validation_end)
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        parser.error("fit, validation, and later periods must all contain months")
    paths = SubprojectPaths(__file__)
    cache_dir = paths.cache_dir / args.cache_name
    input_path = cache_dir / "cond_trainfit.npy"
    params_path = cache_dir / "preprocess_params.npz"
    manifest_path = cache_dir / "manifest.json"
    if args.verify_only:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        legacy_validation = "2018-01/2022-12" if args.cache_name == "strict_inputs" else None
        if (manifest["train_period"] != f"1994-01/{args.fit_end[:7]}" or
                manifest.get("validation_period", legacy_validation) !=
                f"{dates[val_mask][0]:%Y-%m}/{args.validation_end[:7]}"):
            raise ValueError("cache manifest belongs to a different time fold")
        values = np.load(input_path, mmap_mode="r")
        assert values.shape == (366, 10, 120, 140)
        assert np.isfinite(values).all()
        assert hashlib.sha256(input_path.read_bytes()).hexdigest() == manifest["sha256"]
        print(f"Verified {input_path}", flush=True)
        return

    if any(path.exists() for path in (input_path, params_path, manifest_path)):
        raise FileExistsError(f"strict input cache already exists: {cache_dir}")

    raw_env = os.environ.get("MODESV21_DATA_PATH")
    if not raw_env:
        raise RuntimeError("MODESV21_DATA_PATH must point to the raw monthly NetCDF folder")
    raw_dir = Path(raw_env)
    fields = load_raw_fields(raw_dir, dates)
    months = np.asarray(dates.month)
    normalized, params = transform_train_only(fields, months, train_mask)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(input_path, normalized)
    np.savez_compressed(params_path, **params)
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    manifest = {
        "source": str(raw_dir), "period": "1994-01/2024-09", "months": len(dates),
        "train_period": f"1994-01/{args.fit_end[:7]}", "train_months": int(train_mask.sum()),
        "validation_period": f"{dates[val_mask][0]:%Y-%m}/{args.validation_end[:7]}",
        "excluded_months": ["2011-09", "2011-10", "2017-01"],
        "variables": list(VARIABLES), "interpolation": "bicubic, factor 2",
        "precipitation_conversion": "fixed 31-day factor, matching prior pipeline",
        "preprocessing": "train-only monthly tp climatology, train-only tp z-score and 3-sigma clip, train-only per-channel min-max",
        "shape": list(normalized.shape), "sha256": digest,
        "train_channel_range": [
            [float(normalized[train_mask, channel].min()), float(normalized[train_mask, channel].max())]
            for channel in range(len(VARIABLES))
        ],
        "val_outside_unit_pct": float((np.abs(normalized[val_mask]) > 1).mean() * 100),
        "test_outside_unit_pct": float((np.abs(normalized[test_mask]) > 1).mean() * 100),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {input_path} with sha256 {digest}", flush=True)
    print(f"Validation/test outside [-1,1]: {manifest['val_outside_unit_pct']:.3f}% / {manifest['test_outside_unit_pct']:.3f}%", flush=True)


if __name__ == "__main__":
    main()
