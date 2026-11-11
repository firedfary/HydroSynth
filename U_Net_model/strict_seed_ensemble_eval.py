"""Compare fixed three-seed FiLM and Baseline ensembles on inspected periods."""

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
from run_unified_eval import build_date_index, paired_stat_tests
from strict_fair_eval import predict, summary
from strict_sst_window_eval import FOLDS, SEEDS, baseline_weights


SOURCES = {"deep": "strict_trajectory_v1", "early": "film_early_v1"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="film_ens_v1")
    args = parser.parse_args()
    if not args.experiment or "/" in args.experiment or "\\" in args.experiment:
        parser.error("experiment must be a directory name")
    paths = SubprojectPaths(__file__)
    if (paths.results_dir / args.experiment).exists():
        raise FileExistsError(paths.results_dir / args.experiment)
    exp_dir = Path(paths.get_exp_dir(args.experiment))
    dates = build_date_index()
    y = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    valid = np.isfinite(y).all(axis=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = {
        "protocol": {
            "members": list(SEEDS),
            "aggregation": "equal-weight arithmetic mean of three preselected checkpoints",
            "control": "same three-seed Baseline ensemble",
            "source_selection": "none; compare all available deep and early FiLM ensembles",
            "evidence_status": "exploratory because both evaluation periods were repeatedly inspected",
        }, "folds": {},
    }
    for fold, spec in FOLDS.items():
        train_mask = np.asarray(dates <= spec["train_end"])
        period_mask = np.asarray((dates > spec["val_end"]) &
                                 (dates <= spec["holdout_end"]))
        p90 = float(np.percentile(y[train_mask][:, valid], 90))
        fold_rows = {}
        for name, source in SOURCES.items():
            films, baselines = [], []
            for seed in SEEDS:
                file = paths.results_dir / source / f"{fold}_s{seed}_holdout.npz"
                with np.load(file) as data:
                    if (data["target"].shape != y[period_mask].shape or
                            not np.allclose(data["target"], y[period_mask], equal_nan=True)):
                        raise ValueError(f"target mismatch in {file}")
                    films.append(data["trajectory"])
                    baselines.append(data["baseline"])
            film_pred = np.mean(films, axis=0)
            base_pred = np.mean(baselines, axis=0)
            film_metrics, film_acc = summary(film_pred, y[period_mask], valid, p90)
            base_metrics, base_acc = summary(base_pred, y[period_mask], valid, p90)
            fold_rows[name] = {
                "film": film_metrics, "baseline": base_metrics,
                "paired_acc": paired_stat_tests(film_acc, base_acc,
                                                dates[period_mask].year.values),
            }
            print(f"{fold} {name} ensemble ACC FiLM={film_metrics['mean_acc']:.5f} "
                  f"Baseline={base_metrics['mean_acc']:.5f}", flush=True)
        payload["folds"][fold] = fold_rows
        validation_mask = np.asarray((dates > spec["train_end"]) &
                                     (dates <= spec["val_end"]))
        x_file = paths.cache_dir / spec["input"] / "cond_trainfit.npy"
        input_hash = hashlib.sha256(x_file.read_bytes()).hexdigest()
        for source in SOURCES.values():
            metrics_file = paths.results_dir / source / "trajectory_metrics.json"
            source_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            if source_metrics["folds"][fold]["input_sha256"] != input_hash:
                raise ValueError(f"input cache differs from {source} on {fold}")
        x_val = torch.from_numpy(np.load(x_file, mmap_mode="r")[validation_mask].copy())
        baseline_val = []
        for seed in SEEDS:
            model = UNetLite(n_channels=10, base_filters=32, dropout=0.1).to(device)
            model.load_state_dict(baseline_weights(paths, fold, seed))
            baseline_val.append(predict(model, "baseline32", x_val,
                                        torch.zeros(len(x_val), 30), device))
            del model
        baseline_val_pred = np.mean(baseline_val, axis=0)
        baseline_val_metrics, baseline_val_acc = summary(
            baseline_val_pred, y[validation_mask], valid, p90
        )
        validation_rows = {}
        for name, source in SOURCES.items():
            films = []
            for seed in SEEDS:
                file = paths.results_dir / source / f"{fold}_s{seed}_validation.npz"
                with np.load(file) as data:
                    prediction = data["prediction"]
                    if prediction.shape != y[validation_mask].shape:
                        raise ValueError(f"validation shape mismatch in {file}")
                    films.append(prediction)
            film_pred = np.mean(films, axis=0)
            film_metrics, film_acc = summary(film_pred, y[validation_mask], valid, p90)
            validation_rows[name] = {
                "film": film_metrics, "baseline": baseline_val_metrics,
                "paired_acc": paired_stat_tests(film_acc, baseline_val_acc,
                                                dates[validation_mask].year.values),
            }
            print(f"{fold} {name} validation ensemble ACC "
                  f"FiLM={film_metrics['mean_acc']:.5f} "
                  f"Baseline={baseline_val_metrics['mean_acc']:.5f}", flush=True)
        payload["folds"][fold]["validation_ensemble"] = validation_rows
    result_file = exp_dir / "ensemble_metrics.json"
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {result_file}", flush=True)


if __name__ == "__main__":
    main()
