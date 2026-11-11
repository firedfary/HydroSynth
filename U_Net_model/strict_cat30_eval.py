"""Evaluate input concatenation with the same 30 lagged SST PCs as FiLM."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from utils.paths import SubprojectPaths
from unetlite import UNetLite
from run_unified_eval import build_date_index, compute_strict_sst_pcs, paired_stat_tests
from strict_fair_eval import predict, summary, train
from strict_sst_window_eval import FOLDS, SEEDS, baseline_weights, fold_data


def checked_prediction(file, key, expected_dates, expected_target):
    with np.load(file, allow_pickle=True) as source:
        if (not np.array_equal(source["dates"], expected_dates) or
                source[key].shape != expected_target.shape or
                not np.allclose(source["target"], expected_target, equal_nan=True)):
            raise ValueError(f"source prediction chronology or target mismatch: {file}")
        return source[key].copy()


def metric_row(pred, target, valid, p90):
    metrics, acc = summary(pred, target, valid, p90)
    return metrics, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--experiment", default="strict_cat30_v1")
    args = parser.parse_args()
    if args.epochs <= 0 or not args.experiment or "/" in args.experiment or "\\" in args.experiment:
        parser.error("epochs must be positive and experiment must be a directory name")

    paths = SubprojectPaths(__file__)
    if (paths.results_dir / args.experiment).exists():
        raise FileExistsError(paths.results_dir / args.experiment)
    dates = build_date_index()
    sst = np.load(config.modelconfig["sst_file"]).astype(np.float32)
    y = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    exp_dir = Path(paths.get_exp_dir(args.experiment))
    result_file = exp_dir / "cat30_metrics.json"
    payload = {
        "protocol": {
            "conditioning": "six oldest-to-newest monthly vectors of five train-fitted SST EOF PCs",
            "architecture": "32-channel U-Net with 30 input-broadcast PC maps, trained from scratch",
            "epochs": args.epochs, "seeds": list(SEEDS), "batch_size": 8,
            "lr": 2e-4, "dropout": 0.1, "loss": "masked MSE",
            "checkpoint_selection": "validation spatial ACC",
            "comparison": "fixed three-seed ensemble of early FiLM and matched Baseline",
            "evidence_status": "all holdout periods were previously inspected; exploratory only",
        }, "folds": {},
    }

    for fold, spec in FOLDS.items():
        x, digest, train_mask, val_mask, holdout_mask, valid, p90, _ = fold_data(
            paths, spec, sst, y, dates, windows=(6,),
        )
        pcs, _, explained = compute_strict_sst_pcs(sst, train_mask, 5, 6, keep_lags=True)
        if pcs.shape != (len(dates), 30) or not np.isfinite(pcs).all():
            raise ValueError(f"invalid 30-PC features for {fold}")
        source_metrics = json.loads((paths.results_dir / "film_early_v1" /
                                     "trajectory_metrics.json").read_text(encoding="utf-8"))
        source_fold = source_metrics["folds"][fold]
        pcs_hash = hashlib.sha256(pcs.tobytes()).hexdigest()
        if (source_fold["input_sha256"] != digest or
                source_fold["trajectory_sha256"] != pcs_hash):
            raise ValueError(f"FiLM source input or SST PC mismatch for {fold}")
        x_t, pcs_t = torch.from_numpy(x), torch.from_numpy(pcs)
        y_t = torch.from_numpy(y[:, None])
        mask_t = torch.isnan(y_t) | torch.from_numpy(~valid)[None, None]
        train_ds = TensorDataset(y_t[train_mask], x_t[train_mask],
                                 mask_t[train_mask], pcs_t[train_mask])
        val_ds = TensorDataset(y_t[val_mask], x_t[val_mask],
                               mask_t[val_mask], pcs_t[val_mask])
        val_dates = dates[val_mask].strftime("%Y-%m-%d").values
        holdout_dates = dates[holdout_mask].strftime("%Y-%m-%d").values
        fold_result = {"input_sha256": digest, "trajectory_sha256": pcs_hash,
                       "sst_pca_explained_variance": [float(v) for v in explained],
                       "seeds": {}}
        payload["folds"][fold] = fold_result
        ensemble = {period: {name: [] for name in ("cat30", "film", "baseline")}
                    for period in ("validation", "holdout")}
        for seed in SEEDS:
            model, info = train("eof_cat32", train_ds, val_ds, device,
                                args.epochs, 2e-4, 0.1, seed, index_dim=30)
            torch.save(model.state_dict(), exp_dir / "checkpoints" / f"{fold}_s{seed}.pt")
            cat_val = predict(model, "eof_cat32", x_t[val_mask], pcs_t[val_mask], device)
            cat_holdout = predict(model, "eof_cat32", x_t[holdout_mask],
                                  pcs_t[holdout_mask], device)
            film_val_file = paths.results_dir / "film_early_v1" / f"{fold}_s{seed}_validation.npz"
            with np.load(film_val_file, allow_pickle=True) as source:
                if (not np.array_equal(source["dates"], val_dates) or
                        source["prediction"].shape != cat_val.shape):
                    raise ValueError(f"FiLM validation mismatch: {film_val_file}")
                film_val = source["prediction"].copy()
            film_holdout_file = paths.results_dir / "film_early_v1" / f"{fold}_s{seed}_holdout.npz"
            film_holdout = checked_prediction(film_holdout_file, "trajectory",
                                              holdout_dates, y[holdout_mask])
            base_holdout = checked_prediction(film_holdout_file, "baseline",
                                              holdout_dates, y[holdout_mask])
            baseline = UNetLite(n_channels=10, base_filters=32, dropout=0.1).to(device)
            baseline.load_state_dict(baseline_weights(paths, fold, seed))
            base_val = predict(baseline, "baseline32", x_t[val_mask], pcs_t[val_mask], device)
            del baseline
            for period, arrays in (("validation", (cat_val, film_val, base_val)),
                                   ("holdout", (cat_holdout, film_holdout, base_holdout))):
                for name, arr in zip(("cat30", "film", "baseline"), arrays):
                    if arr is not None:
                        ensemble[period][name].append(arr)
            fold_result["seeds"][str(seed)] = {
                "training": info,
                "validation": {
                    "cat30": metric_row(cat_val, y[val_mask], valid, p90)[0],
                    "film": metric_row(film_val, y[val_mask], valid, p90)[0],
                    "baseline": metric_row(base_val, y[val_mask], valid, p90)[0],
                },
                "exploratory_holdout": {
                    "cat30": metric_row(cat_holdout, y[holdout_mask], valid, p90)[0],
                    "film": metric_row(film_holdout, y[holdout_mask], valid, p90)[0],
                    "baseline": metric_row(base_holdout, y[holdout_mask], valid, p90)[0],
                },
            }
            np.savez_compressed(exp_dir / f"{fold}_s{seed}_predictions.npz",
                                validation_dates=val_dates, holdout_dates=holdout_dates,
                                cat30_validation=cat_val, cat30_holdout=cat_holdout)
            result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"{fold} seed={seed} validation ACC cat30="
                  f"{fold_result['seeds'][str(seed)]['validation']['cat30']['mean_acc']:.5f}",
                  flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for period, mask in (("validation", val_mask), ("holdout", holdout_mask)):
            target = y[mask]
            metrics = {}
            acc_series = {}
            for name, predictions in ensemble[period].items():
                if not predictions:
                    continue
                metrics[name], acc_series[name] = metric_row(
                    np.mean(predictions, axis=0), target, valid, p90,
                )
            paired = {other: paired_stat_tests(acc_series["film"], acc_series[other],
                                               dates[mask].year.values)
                      for other in metrics if other != "film"}
            fold_result["ensemble_" + period] = {"metrics": metrics,
                                                  "paired_film_minus": paired}
            print(f"{fold} {period} ensemble ACC " +
                  " ".join(f"{name}={row['mean_acc']:.5f}"
                           for name, row in metrics.items()), flush=True)
        result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {result_file}", flush=True)


if __name__ == "__main__":
    main()
