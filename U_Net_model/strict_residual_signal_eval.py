"""Diagnose whether lagged SST PCs explain held-validation U-Net residuals."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from utils.paths import SubprojectPaths
from unetlite import UNetLite
from run_unified_eval import build_date_index, compute_strict_sst_pcs
from strict_fair_eval import predict, summary
from strict_sst_window_eval import FOLDS, SEEDS, baseline_weights, fold_data


RIDGE_ALPHA = 100.0
SST_GAINS = (0.05, 0.1, 0.25, 0.5, 1.0)


def design_matrix(pcs, dates, train_mask):
    mean = pcs[train_mask].mean(axis=0)
    scale = pcs[train_mask].std(axis=0)
    if np.any(scale <= 0):
        raise ValueError("SST PC has zero training variance")
    months = np.asarray(dates.month, dtype=np.float64)
    phase = 2 * np.pi * (months - 1) / 12
    calendar = np.column_stack((np.ones(len(pcs)), np.sin(phase), np.cos(phase)))
    standard_pcs = (pcs - mean) / scale
    return calendar, np.column_stack((calendar, standard_pcs))


def ridge_predict(x_train, residual, x_val, alpha):
    penalty = np.eye(x_train.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(
        x_train.T @ x_train + penalty, x_train.T @ residual
    )
    return x_val @ coefficients, x_train @ coefficients


def corrected_metrics(base_prediction, correction, y, valid, p90):
    prediction = base_prediction.copy()
    prediction[:, valid] += correction.astype(np.float32)
    metrics, _ = summary(prediction, y, valid, p90)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="strict_residual_signal_v1")
    args = parser.parse_args()
    if not args.experiment or "/" in args.experiment or "\\" in args.experiment:
        parser.error("experiment must be a directory name")
    paths = SubprojectPaths(__file__)
    exp_dir = paths.results_dir / args.experiment
    if exp_dir.exists():
        raise FileExistsError(exp_dir)
    exp_dir = Path(paths.get_exp_dir(args.experiment))
    result_file = exp_dir / "residual_signal_metrics.json"

    dates = build_date_index()
    sst = np.load(config.modelconfig["sst_file"]).astype(np.float32)
    y = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = {
        "protocol": {
            "purpose": "diagnose incremental SST information in Baseline residuals",
            "control": "ridge correction from intercept and month sine/cosine",
            "candidate": "same control plus six-lag 30-PC SST trajectory",
            "ridge_alpha": RIDGE_ALPHA,
            "ridge_alpha_selection": "fixed before validation; no tuning",
            "sst_gains": list(SST_GAINS),
            "gain_role": "diagnostic sensitivity only; no model selected",
            "baseline": "existing fold-specific selected checkpoints; training predictions are in sample",
            "validation_status": "exploratory because Baseline checkpoints were selected using these periods",
            "holdout_status": "no holdout metrics; shared valid-grid mask checks all-period missingness",
        },
        "folds": {},
    }
    for fold, spec in FOLDS.items():
        x, digest, train_mask, val_mask, _, valid, p90, _ = fold_data(
            paths, spec, sst, y, dates, windows=(6,),
        )
        pcs, _, _ = compute_strict_sst_pcs(sst, train_mask, 5, 6, keep_lags=True)
        calendar, all_features = design_matrix(pcs, dates, train_mask)
        fold_result = {
            "input_sha256": digest,
            "pc_sha256": hashlib.sha256(pcs.tobytes()).hexdigest(),
            "train_months": int(train_mask.sum()),
            "validation_months": int(val_mask.sum()),
            "seeds": {},
        }
        for seed in SEEDS:
            model = UNetLite(n_channels=10, base_filters=32, dropout=0.1).to(device)
            model.load_state_dict(baseline_weights(paths, fold, seed))
            base_train = predict(model, "baseline32", torch.from_numpy(x[train_mask]),
                                 torch.from_numpy(pcs[train_mask]), device)
            base_val = predict(model, "baseline32", torch.from_numpy(x[val_mask]),
                               torch.from_numpy(pcs[val_mask]), device)
            del model
            residual = (y[train_mask][:, valid] - base_train[:, valid]).astype(np.float64)
            base_metrics, _ = summary(base_val, y[val_mask], valid, p90)
            modes = {"calendar": calendar, "sst": all_features}
            metrics = {"baseline": base_metrics}
            train_mse = {"baseline": float(np.mean(residual ** 2))}
            corrections = {}
            for name, features in modes.items():
                correction, fitted = ridge_predict(features[train_mask], residual,
                                                    features[val_mask], RIDGE_ALPHA)
                corrections[name] = correction
                metrics[name] = corrected_metrics(base_val, correction, y[val_mask],
                                                  valid, p90)
                train_mse[name] = float(np.mean((residual - fitted) ** 2))
            sst_increment = corrections["sst"] - corrections["calendar"]
            sst_gain_metrics = {
                str(gain): corrected_metrics(
                    base_val, corrections["calendar"] + gain * sst_increment,
                    y[val_mask], valid, p90,
                ) for gain in SST_GAINS
            }
            val_residual_after_calendar = (y[val_mask][:, valid] -
                                           base_val[:, valid] - corrections["calendar"])
            optimal_mse_gain = float(np.sum(sst_increment * val_residual_after_calendar) /
                                     np.sum(sst_increment ** 2))
            fold_result["seeds"][str(seed)] = {
                "validation": metrics, "training_residual_mse": train_mse,
                "sst_gain_sensitivity": sst_gain_metrics,
                "sst_increment_rms": float(np.sqrt(np.mean(sst_increment ** 2))),
                "posthoc_validation_mse_optimal_gain": optimal_mse_gain,
                "sst_minus_calendar_val_acc": metrics["sst"]["mean_acc"] -
                                               metrics["calendar"]["mean_acc"],
                "sst_minus_calendar_val_rmse": metrics["sst"]["rmse"] -
                                                metrics["calendar"]["rmse"],
            }
            payload["folds"][fold] = fold_result
            result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"{fold} seed={seed} validation ACC "
                  f"base={metrics['baseline']['mean_acc']:.5f} "
                  f"calendar={metrics['calendar']['mean_acc']:.5f} "
                  f"SST={metrics['sst']['mean_acc']:.5f}", flush=True)
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {result_file}", flush=True)


if __name__ == "__main__":
    main()
