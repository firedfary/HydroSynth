"""Compare recent SST averaging windows using two train-fitted validation folds."""

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
from unetlitefilm import UNetLiteFiLM
from run_unified_eval import build_date_index, compute_strict_sst_pcs, paired_stat_tests
from strict_centered_eval import train_candidate
from strict_fair_eval import predict, summary


FOLDS = {
    "recent": {"input": "strict_inputs", "train_end": "2017-12-01",
               "val_end": "2022-12-01", "holdout_end": "2024-09-01",
               "prior": "strict_centered_v1"},
    "historical": {"input": "strict_inputs_pre2011", "train_end": "2010-12-01",
                   "val_end": "2013-12-01", "holdout_end": "2017-12-01",
                   "prior": "strict_replication_v1"},
}
FAIR_EXPERIMENTS = {42: "strict_fair_v1", 43: "strict_fair_s43", 44: "strict_fair_s44"}
SEEDS = (42, 43, 44)
WINDOWS = (1, 3, 6)


def fold_data(paths, spec, sst, y, dates, windows=WINDOWS):
    input_file = paths.cache_dir / spec["input"] / "cond_trainfit.npy"
    manifest = json.loads((input_file.parent / "manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(input_file.read_bytes()).hexdigest()
    if digest != manifest["sha256"] or manifest["train_period"] != f"1994-01/{spec['train_end'][:7]}":
        raise ValueError(f"predictor cache does not match fold {spec['input']}")
    train_mask = np.asarray(dates <= spec["train_end"])
    val_mask = np.asarray((dates > spec["train_end"]) & (dates <= spec["val_end"]))
    holdout_mask = np.asarray((dates > spec["val_end"]) & (dates <= spec["holdout_end"]))
    x = np.load(input_file).astype(np.float32)
    valid = np.isfinite(y).all(axis=0)
    p90 = float(np.percentile(y[train_mask][:, valid], 90))
    pcs = {window: compute_strict_sst_pcs(sst, train_mask, 5, window)[0]
           for window in windows}
    return x, digest, train_mask, val_mask, holdout_mask, valid, p90, pcs


def baseline_weights(paths, fold, seed):
    if fold == "recent":
        path = paths.results_dir / FAIR_EXPERIMENTS[seed] / "checkpoints" / "baseline32.pt"
    else:
        path = paths.results_dir / "strict_replication_v1" / "checkpoints" / f"baseline_s{seed}.pt"
    return torch.load(path, map_location="cpu", weights_only=True)


def prior_validation(paths, fold, seed):
    prior_dir = paths.results_dir / FOLDS[fold]["prior"]
    if fold == "recent":
        prior = json.loads((prior_dir / "strict_centered_metrics.json").read_text(encoding="utf-8"))
        row = prior["seeds"][str(seed)]
        return row["baseline_val_acc"], row["candidates"]["deep_mse"]["validation"]["mean_acc"]
    prior = json.loads((prior_dir / "strict_replication_metrics.json").read_text(encoding="utf-8"))
    row = prior["seeds"][str(seed)]
    return row["baseline_validation"]["mean_acc"], row["film"]["validation"]["mean_acc"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--experiment", default="strict_sst_window_v1")
    args = parser.parse_args()
    if args.epochs <= 0 or not args.experiment or "/" in args.experiment or "\\" in args.experiment:
        parser.error("epochs must be positive and experiment must be a directory name")
    paths = SubprojectPaths(__file__)
    exp_dir = paths.results_dir / args.experiment
    if exp_dir.exists():
        raise FileExistsError(exp_dir)
    exp_dir = Path(paths.get_exp_dir(args.experiment))
    result_file = exp_dir / "sst_window_metrics.json"

    dates = build_date_index()
    sst = np.load(config.modelconfig["sst_file"]).astype(np.float32)
    y = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = {
        "protocol": {"windows": list(WINDOWS), "seeds": list(SEEDS),
                     "architecture": "deep zero-PC-identity FiLM, frozen 32-channel U-Net",
                     "fit": "fold-specific meteorological normalization and SST EOF",
                     "epochs": args.epochs, "checkpoint_selection": "validation ACC",
                     "window_selection": "both fold mean validation ACC must exceed W6 and Baseline",
                     "holdout_status": "only validation-selected W1 or W3 is evaluated; previously inspected periods are exploratory"},
        "folds": {},
    }
    for fold, spec in FOLDS.items():
        x, digest, train_mask, val_mask, _, valid, p90, pcs = fold_data(
            paths, spec, sst, y, dates,
        )
        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y[:, None])
        mask_t = torch.isnan(y_t) | torch.from_numpy(~valid)[None, None]
        fold_result = {"input_sha256": digest,
                       "pc_sha256": {str(w): hashlib.sha256(pcs[w].tobytes()).hexdigest()
                                     for w in WINDOWS},
                       "seeds": {}}
        for seed in SEEDS:
            base_weights = baseline_weights(paths, fold, seed)
            baseline = UNetLite(n_channels=10, base_filters=32, dropout=0.1).to(device)
            baseline.load_state_dict(base_weights)
            base_val_pred = predict(baseline, "baseline32", x_t[val_mask],
                                    torch.from_numpy(pcs[6][val_mask]), device)
            base_val, _ = summary(base_val_pred, y[val_mask], valid, p90)
            prior_base, prior_w6 = prior_validation(paths, fold, seed)
            if abs(base_val["mean_acc"] - prior_base) > 1e-5:
                raise ValueError(f"Baseline validation mismatch: {fold}, {seed}")
            del baseline
            seed_result = {"baseline_val_acc": base_val["mean_acc"],
                           "windows": {"6": {"val_acc": prior_w6,
                                             "source": spec["prior"]}}}
            for window in (1, 3):
                pcs_t = torch.from_numpy(pcs[window])
                train_ds = TensorDataset(y_t[train_mask], x_t[train_mask],
                                         mask_t[train_mask], pcs_t[train_mask])
                val_arrays = (x_t[val_mask], pcs_t[val_mask], y[val_mask], valid, p90)
                model, info, val_pred = train_candidate(
                    f"w{window}", (3, 4), 0.0, base_weights, train_ds,
                    val_arrays, base_val["mean_acc"], device, args.epochs, seed,
                )
                torch.save(model.state_dict(),
                           exp_dir / "checkpoints" / f"{fold}_s{seed}_w{window}.pt")
                np.savez_compressed(exp_dir / f"{fold}_s{seed}_w{window}_validation.npz",
                                    dates=dates[val_mask].strftime("%Y-%m-%d").values,
                                    prediction=val_pred)
                seed_result["windows"][str(window)] = {
                    "val_acc": info["validation"]["mean_acc"],
                    "val_rmse": info["validation"]["rmse"],
                    "val_pc_zero_acc": info["validation_pc_zero"]["mean_acc"],
                    "best_epoch": info["best_epoch"], "history": info["history"],
                }
                del model
            fold_result["seeds"][str(seed)] = seed_result
            payload["folds"][fold] = fold_result
            result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fold_means = {
        fold: {str(window): float(np.mean([
            payload["folds"][fold]["seeds"][str(seed)]["windows"][str(window)]["val_acc"]
            for seed in SEEDS])) for window in WINDOWS}
        for fold in FOLDS
    }
    baseline_means = {
        fold: float(np.mean([payload["folds"][fold]["seeds"][str(seed)]["baseline_val_acc"]
                             for seed in SEEDS])) for fold in FOLDS
    }
    eligible = [window for window in (1, 3)
                if all(fold_means[fold][str(window)] > max(
                    fold_means[fold]["6"], baseline_means[fold]) for fold in FOLDS)]
    selected = max(eligible, key=lambda w: sum(fold_means[f][str(w)] for f in FOLDS)) if eligible else None
    payload["selection"] = {"fold_mean_val_acc": fold_means,
                            "fold_mean_baseline_acc": baseline_means,
                            "selected_new_window": selected}
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"window selection: {payload['selection']}", flush=True)
    if selected is None:
        print("No new window passed the two-fold validation gate.", flush=True)
        return

    for fold, spec in FOLDS.items():
        x, _, train_mask, _, holdout_mask, valid, p90, pcs = fold_data(
            paths, spec, sst, y, dates,
        )
        x_t = torch.from_numpy(x)
        pcs_t = torch.from_numpy(pcs[selected])
        for seed in SEEDS:
            base_weights = baseline_weights(paths, fold, seed)
            baseline = UNetLite(n_channels=10, base_filters=32, dropout=0.1).to(device)
            baseline.load_state_dict(base_weights)
            model = UNetLiteFiLM(
                n_channels=10, index_dim=5, base_filters=32, index_hidden=16,
                dropout=0.1, film_layers=(3, 4), identity_init=True,
                zero_pc_identity=True,
            ).to(device)
            model.load_state_dict(torch.load(
                exp_dir / "checkpoints" / f"{fold}_s{seed}_w{selected}.pt",
                map_location=device, weights_only=True,
            ))
            xx, pp = x_t[holdout_mask], pcs_t[holdout_mask]
            base_pred = predict(baseline, "baseline32", xx, pp, device)
            film_pred = predict(model, "film32_identity", xx, pp, device)
            zero_pred = predict(model, "film32_identity", xx, torch.zeros_like(pp), device)
            base, base_acc = summary(base_pred, y[holdout_mask], valid, p90)
            film, film_acc = summary(film_pred, y[holdout_mask], valid, p90)
            zero, _ = summary(zero_pred, y[holdout_mask], valid, p90)
            payload["folds"][fold]["seeds"][str(seed)]["exploratory_holdout"] = {
                "baseline": base, "film": film, "film_pc_zero": zero,
                "paired_acc": paired_stat_tests(film_acc, base_acc,
                                                 dates[holdout_mask].year.values),
            }
            np.savez_compressed(exp_dir / f"{fold}_s{seed}_w{selected}_holdout.npz",
                                dates=dates[holdout_mask].strftime("%Y-%m-%d").values,
                                target=y[holdout_mask], baseline=base_pred,
                                film=film_pred, film_pc_zero=zero_pred)
            print(f"{fold} seed={seed} exploratory holdout ACC "
                  f"FiLM={film['mean_acc']:.4f} Baseline={base['mean_acc']:.4f}", flush=True)
            del model, baseline
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {result_file}", flush=True)


if __name__ == "__main__":
    main()
