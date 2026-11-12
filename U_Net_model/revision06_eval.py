"""Re-evaluate matched three-seed models against an unscaled SEAS5 anomaly.

The 2023-2024 period has already informed development, so every result from
this script is exploratory. Outputs belong to the external experiment workspace.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from utils.paths import SubprojectPaths
from run_unified_eval import build_date_index, compute_strict_sst_pcs, paired_stat_tests
from strict_fair_eval import make_model, predict, summary


SEEDS = (42, 43, 44)
SOURCES = {42: "strict_fair_v1", 43: "strict_fair_s43", 44: "strict_fair_s44"}
MODELS = {"baseline": "baseline32", "eof_cat": "eof_cat32", "film": "film32_identity"}
PERIODS = {
    "validation": ("2018-01-01", "2022-12-01"),
    "test": ("2023-01-01", "2024-09-01"),
}


def raw_seas5_anomaly(dates, raw_dir, reference_end="2010-12-01",
                      interpolation="bilinear"):
    """Use the observed target's fixed 1994-2010 climatology period."""
    monthly = []
    for index, date in enumerate(dates):
        path = raw_dir / f"MODESv21_ecmwf_seas51_{date:%Y%m}_monthly_em.nc"
        if not path.is_file():
            raise FileNotFoundError(path)
        with xr.open_dataset(path) as ds:
            actual = np.datetime64(ds.time.values[0], "M")
            if actual != np.datetime64(date, "M"):
                raise ValueError(f"time mismatch: {path}")
            field = ds["tp"].sel(longitude=slice(70, 140), latitude=slice(60, 0)).isel(time=0)
            values = field.values.astype(np.float32)
        if values.shape != (60, 70) or not np.isfinite(values).all():
            raise ValueError(f"invalid precipitation field: {path}")
        monthly.append(values)
        if (index + 1) % 100 == 0:
            print(f"Read {index + 1}/{len(dates)} SEAS5 months", flush=True)
    # Bilinear interpolation preserves non-negative precipitation fields. The
    # bicubic option is retained to quantify sensitivity to the old input grid.
    raw = F.interpolate(torch.from_numpy(np.stack(monthly))[:, None],
                        scale_factor=2, mode=interpolation)[:, 0].numpy()
    raw = np.maximum(raw, 0)
    raw *= 31 * 24 * 60 * 60 * 1000
    months = np.asarray(dates.month)
    reference = np.asarray(dates <= reference_end)
    climatology = np.stack([raw[reference & (months == m)].mean(axis=0)
                            for m in range(1, 13)])
    anomaly = (raw - climatology[months - 1]) / (climatology[months - 1] + 1e-4)
    return anomaly.astype(np.float32), climatology.astype(np.float32)


def metrics_row(pred, obs, valid, p90):
    row, acc = summary(pred, obs, valid, p90)
    return row, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="revision06_eval")
    parser.add_argument("--interpolation", choices=("bilinear", "bicubic"),
                        default="bilinear")
    args = parser.parse_args()
    if not args.experiment or any(c in args.experiment for c in "/\\"):
        parser.error("experiment must be a directory name")

    paths = SubprojectPaths(__file__)
    output = paths.results_dir / args.experiment
    if output.exists():
        raise FileExistsError(output)
    dates = build_date_index()
    raw_dir_name = os.environ.get("MODESV21_DATA_PATH")
    if not raw_dir_name:
        raise RuntimeError("MODESV21_DATA_PATH is required")
    raw, climatology = raw_seas5_anomaly(dates, Path(raw_dir_name),
                                        interpolation=args.interpolation)
    y = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    if y.shape != raw.shape:
        raise ValueError("target and SEAS5 chronology differ")
    valid = np.isfinite(y).all(axis=0)
    p90 = float(np.percentile(y[np.asarray(dates <= "2017-12-01")][:, valid], 90))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_file = paths.cache_dir / "strict_inputs" / "cond_trainfit.npy"
    manifest = json.loads((input_file.parent / "manifest.json").read_text(encoding="utf-8"))
    input_hash = hashlib.sha256(input_file.read_bytes()).hexdigest()
    if input_hash != manifest["sha256"]:
        raise ValueError("strict input hash mismatch")
    x = torch.from_numpy(np.load(input_file, mmap_mode="r").copy())
    sst = np.load(config.modelconfig["sst_file"], mmap_mode="r")
    pc_all, _, _ = compute_strict_sst_pcs(sst, np.asarray(dates <= "2017-12-01"), 5)

    output = Path(paths.get_exp_dir(args.experiment))
    np.savez_compressed(output / "seas5_raw.npz", dates=dates.strftime("%Y-%m-%d").values,
                        raw_anomaly=raw, climatology=climatology)
    payload = {
        "protocol": {
            "seeds": list(SEEDS), "train": "1994-01/2017-12",
            "validation": "2018-01/2022-12", "test": "2023-01/2024-09",
            "reference_climatology": "1994-01/2010-12 for SEAS5 and observations",
            "seas5": f"first monthly tp field in each archive file, {args.interpolation} grid, non-negative fractional anomaly without input standardization; initialization date and operational lead label are not present in file metadata",
            "models": "32-channel U-Net; 5 PCs for EOF-cat and FiLM",
            "checkpoint_selection": "validation spatial ACC, one per seed and model",
            "ensemble": "equal-weight arithmetic mean of three seed predictions",
            "evidence_status": "exploratory; 2023-2024 was inspected during model development",
        },
        "input_sha256": input_hash,
        "valid_grid_count": int(valid.sum()), "train_p90": p90,
        "periods": {},
    }
    for period, (start, end) in PERIODS.items():
        mask = np.asarray((dates >= start) & (dates <= end))
        years = dates[mask].year.values
        predictions = {"seas5": raw[mask]}
        members = {name: [] for name in MODELS}
        seed_rows = {}
        for seed in SEEDS:
            source = paths.results_dir / SOURCES[seed]
            source_metrics = json.loads((source / "strict_fair_metrics.json").read_text(encoding="utf-8"))
            if source_metrics["input_sha256"] != input_hash:
                raise ValueError(f"input mismatch: {source}")
            seed_rows[str(seed)] = {}
            for name, model_name in MODELS.items():
                model = make_model(model_name, 0.1).to(device)
                state = torch.load(source / "checkpoints" / f"{model_name}.pt",
                                   map_location=device, weights_only=True)
                model.load_state_dict(state)
                # PCs are recorded in the test NPZ; validation PCs are generated
                # below from the train-fitted EOF basis to avoid refitting.
                if period == "test":
                    with np.load(source / "test_predictions.npz") as previous:
                        pred = previous[model_name].copy()
                        if not np.allclose(previous["target"], y[mask], equal_nan=True):
                            raise ValueError(f"target mismatch: {source}")
                        if not np.allclose(previous["pcs"], pc_all[mask], atol=1e-5):
                            raise ValueError(f"SST PC mismatch: {source}")
                else:
                    pred = predict(model, model_name, x[mask],
                                   torch.from_numpy(pc_all[mask].copy()), device)
                del model, state
                members[name].append(pred)
                seed_rows[str(seed)][name] = metrics_row(pred, y[mask], valid, p90)[0]
        for name, values in members.items():
            predictions[name] = np.mean(values, axis=0)
        metrics, acc = {}, {}
        for name, pred in predictions.items():
            metrics[name], acc[name] = metrics_row(pred, y[mask], valid, p90)
        paired = {f"{name}_minus_{other}": paired_stat_tests(acc[name], acc[other], years)
                  for name in MODELS for other in ("seas5", "baseline", "eof_cat")
                  if name != other}
        payload["periods"][period] = {
            "months": int(mask.sum()), "ensemble": metrics,
            "seed_metrics": seed_rows, "paired_acc": paired,
        }
        np.savez_compressed(output / f"{period}_predictions.npz",
                            dates=dates[mask].strftime("%Y-%m-%d").values,
                            target=y[mask], valid=valid, **predictions)
        print(period, {k: v["mean_acc"] for k, v in metrics.items()}, flush=True)
    (output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {output / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
