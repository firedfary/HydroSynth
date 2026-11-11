"""Evaluate six lagged SST EOF vectors as one FiLM conditioning sequence."""

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
from strict_sst_window_eval import FOLDS, SEEDS, baseline_weights, fold_data, prior_validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--experiment", default="strict_trajectory_v1")
    parser.add_argument("--condition-dropout", type=float, default=0.0)
    parser.add_argument("--film-layers", default="3,4")
    parser.add_argument("--loss", choices=("mse", "huber1"), default="mse")
    parser.add_argument("--season", action="store_true")
    args = parser.parse_args()
    try:
        layers = tuple(int(value) for value in args.film_layers.split(","))
    except ValueError:
        parser.error("film-layers must be comma-separated integers")
    if (args.epochs <= 0 or not args.experiment or "/" in args.experiment or
            "\\" in args.experiment or not 0.0 <= args.condition_dropout < 1.0 or
            not layers or len(set(layers)) != len(layers) or
            any(level < 0 or level > 8 for level in layers)):
        parser.error("invalid experiment, dropout, or film-layers")
    paths = SubprojectPaths(__file__)
    exp_dir = paths.results_dir / args.experiment
    if exp_dir.exists():
        raise FileExistsError(exp_dir)
    exp_dir = Path(paths.get_exp_dir(args.experiment))
    result_file = exp_dir / "trajectory_metrics.json"

    dates = build_date_index()
    sst = np.load(config.modelconfig["sst_file"]).astype(np.float32)
    y = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = {
        "protocol": {"conditioning": "six oldest-to-newest monthly vectors of five PCs",
                     "eof_fit": "six-month mean field on each fold's training period only",
                     "pc_scale": "training six-month-mean PC standard deviation",
                     "architecture": "zero-PC-identity FiLM, hidden 16, frozen Baseline",
                     "film_layers": list(layers),
                     "loss": args.loss,
                     "season_features": "target-month sine and cosine" if args.season else None,
                     "epochs": args.epochs, "seeds": list(SEEDS),
                     "condition_dropout": args.condition_dropout,
                     "checkpoint_selection": "validation ACC",
                     "holdout_gate": "both fold mean validation ACC exceed averaged-W6, Baseline, and the no-dropout trajectory when applicable",
                     "holdout_status": "previously inspected periods are exploratory"},
        "folds": {},
    }
    for fold, spec in FOLDS.items():
        x, digest, train_mask, val_mask, _, valid, p90, averaged = fold_data(
            paths, spec, sst, y, dates, windows=(6,),
        )
        trajectory, _, explained = compute_strict_sst_pcs(
            sst, train_mask, 5, 6, keep_lags=True,
        )
        reconstruction_diff = float(np.max(np.abs(
            trajectory.reshape(len(dates), 6, 5).mean(axis=1) - averaged[6]
        )))
        if reconstruction_diff > 1e-4:
            raise ValueError(f"lagged PCs do not reconstruct averaged PC: {reconstruction_diff}")
        if args.season:
            angle = 2 * np.pi * (np.asarray(dates.month) - 1) / 12
            trajectory = np.column_stack((trajectory, np.sin(angle), np.cos(angle))).astype(np.float32)
        x_t, pcs_t = torch.from_numpy(x), torch.from_numpy(trajectory)
        y_t = torch.from_numpy(y[:, None])
        mask_t = torch.isnan(y_t) | torch.from_numpy(~valid)[None, None]
        train_ds = TensorDataset(y_t[train_mask], x_t[train_mask],
                                 mask_t[train_mask], pcs_t[train_mask])
        val_arrays = (x_t[val_mask], pcs_t[val_mask], y[val_mask], valid, p90)
        fold_result = {"input_sha256": digest,
                       "trajectory_sha256": hashlib.sha256(trajectory.tobytes()).hexdigest(),
                       "mean_pc_reconstruction_max_abs_diff": reconstruction_diff,
                       "sst_pca_explained_variance": [float(v) for v in explained],
                       "seeds": {}}
        prior_trajectory = None
        if args.condition_dropout or layers != (3, 4) or args.loss != "mse" or args.season:
            prior_trajectory = json.loads((paths.results_dir / "strict_trajectory_v1" /
                                           "trajectory_metrics.json").read_text(encoding="utf-8"))
        for seed in SEEDS:
            prior_base, prior_avg6 = prior_validation(paths, fold, seed)
            model, info, val_pred = train_candidate(
                "trajectory6", layers, 0.0, baseline_weights(paths, fold, seed),
                train_ds, val_arrays, prior_base, device, args.epochs, seed,
                condition_dropout=args.condition_dropout,
                loss_name=args.loss,
                reference_zero_dims=30 if args.season else None,
            )
            torch.save(model.state_dict(), exp_dir / "checkpoints" / f"{fold}_s{seed}.pt")
            np.savez_compressed(exp_dir / f"{fold}_s{seed}_validation.npz",
                                dates=dates[val_mask].strftime("%Y-%m-%d").values,
                                prediction=val_pred)
            fold_result["seeds"][str(seed)] = {
                "baseline_val_acc": prior_base, "averaged_w6_val_acc": prior_avg6,
                "trajectory": info,
            }
            if prior_trajectory is not None:
                fold_result["seeds"][str(seed)]["no_dropout_val_acc"] = (
                    prior_trajectory["folds"][fold]["seeds"][str(seed)]
                    ["trajectory"]["validation"]["mean_acc"]
                )
            payload["folds"][fold] = fold_result
            result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            del model

    fold_means = {}
    for fold in FOLDS:
        rows = payload["folds"][fold]["seeds"]
        fold_means[fold] = {
            "baseline": float(np.mean([rows[str(seed)]["baseline_val_acc"] for seed in SEEDS])),
            "averaged_w6": float(np.mean([rows[str(seed)]["averaged_w6_val_acc"] for seed in SEEDS])),
            "trajectory6": float(np.mean([rows[str(seed)]["trajectory"]["validation"]["mean_acc"]
                                          for seed in SEEDS])),
        }
        if args.condition_dropout or layers != (3, 4) or args.loss != "mse" or args.season:
            fold_means[fold]["no_dropout"] = float(np.mean([
                rows[str(seed)]["no_dropout_val_acc"] for seed in SEEDS
            ]))
    passed = all(row["trajectory6"] > max(
        row["baseline"], row["averaged_w6"], row.get("no_dropout", -float("inf")))
                 for row in fold_means.values())
    payload["selection"] = {"fold_mean_val_acc": fold_means,
                            "passed_two_fold_gate": passed}
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"trajectory selection: {payload['selection']}", flush=True)
    if not passed:
        print("Trajectory did not pass the two-fold validation gate.", flush=True)
        return

    for fold, spec in FOLDS.items():
        x, _, train_mask, _, holdout_mask, valid, p90, _ = fold_data(
            paths, spec, sst, y, dates, windows=(6,),
        )
        trajectory, _, _ = compute_strict_sst_pcs(
            sst, train_mask, 5, 6, keep_lags=True,
        )
        if args.season:
            angle = 2 * np.pi * (np.asarray(dates.month) - 1) / 12
            trajectory = np.column_stack((trajectory, np.sin(angle), np.cos(angle))).astype(np.float32)
        xx = torch.from_numpy(x[holdout_mask])
        pp = torch.from_numpy(trajectory[holdout_mask])
        for seed in SEEDS:
            baseline = UNetLite(n_channels=10, base_filters=32, dropout=0.1).to(device)
            baseline.load_state_dict(baseline_weights(paths, fold, seed))
            model = UNetLiteFiLM(
                n_channels=10, index_dim=trajectory.shape[1], base_filters=32, index_hidden=16,
                dropout=0.1, film_layers=layers, identity_init=True,
                zero_pc_identity=True,
                reference_zero_dims=30 if args.season else None,
            ).to(device)
            model.load_state_dict(torch.load(
                exp_dir / "checkpoints" / f"{fold}_s{seed}.pt",
                map_location=device, weights_only=True,
            ))
            base_pred = predict(baseline, "baseline32", xx, pp, device)
            film_pred = predict(model, "film32_identity", xx, pp, device)
            zero_pred = predict(model, "film32_identity", xx,
                                model.reference_indices(pp), device)
            base, base_acc = summary(base_pred, y[holdout_mask], valid, p90)
            film, film_acc = summary(film_pred, y[holdout_mask], valid, p90)
            zero, _ = summary(zero_pred, y[holdout_mask], valid, p90)
            payload["folds"][fold]["seeds"][str(seed)]["exploratory_holdout"] = {
                "baseline": base, "trajectory": film, "trajectory_pc_zero": zero,
                "paired_acc": paired_stat_tests(film_acc, base_acc,
                                                 dates[holdout_mask].year.values),
            }
            np.savez_compressed(exp_dir / f"{fold}_s{seed}_holdout.npz",
                                dates=dates[holdout_mask].strftime("%Y-%m-%d").values,
                                target=y[holdout_mask], baseline=base_pred,
                                trajectory=film_pred, trajectory_pc_zero=zero_pred)
            print(f"{fold} seed={seed} exploratory holdout ACC "
                  f"FiLM={film['mean_acc']:.4f} Baseline={base['mean_acc']:.4f}", flush=True)
            del baseline, model
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {result_file}", flush=True)


if __name__ == "__main__":
    main()
