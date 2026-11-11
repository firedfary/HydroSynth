"""Compare compact temporal EOF features with the six-lag FiLM reference."""

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


PC_COUNTS = (1, 2, 3)
FEATURES = tuple(f"lag_pc{count}" for count in PC_COUNTS)


def temporal_features(trajectory, train_mask):
    """Return train-scaled trajectories containing one, two, or three leading PCs."""
    lags = trajectory.reshape(len(trajectory), 6, 5)
    features, scales = {}, {}
    for count in PC_COUNTS:
        selected = lags[:, :, :count]
        lag_std = selected[train_mask].std(axis=0)
        if np.any(lag_std <= 0):
            raise ValueError(f"a lagged PC has zero training variance for {count} PCs")
        features[f"lag_pc{count}"] = (selected / lag_std).reshape(
            len(lags), 6 * count
        ).astype(np.float32)
        scales[f"lag_pc{count}_std"] = lag_std.tolist()
    return features, scales


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--experiment", default="strict_temporal_features_v1")
    args = parser.parse_args()
    if args.epochs <= 0 or not args.experiment or "/" in args.experiment or "\\" in args.experiment:
        parser.error("epochs must be positive and experiment must be a directory name")

    paths = SubprojectPaths(__file__)
    exp_dir = paths.results_dir / args.experiment
    if exp_dir.exists():
        raise FileExistsError(exp_dir)
    exp_dir = Path(paths.get_exp_dir(args.experiment))
    result_file = exp_dir / "temporal_features_metrics.json"
    reference = json.loads((paths.results_dir / "strict_trajectory_v1" /
                            "trajectory_metrics.json").read_text(encoding="utf-8"))
    dates = build_date_index()
    sst = np.load(config.modelconfig["sst_file"]).astype(np.float32)
    y = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = {
        "protocol": {
            "features": list(FEATURES),
            "eof_fit": "fold training period only, using six-month-mean SST fields",
            "low_mode_ablation": "retain the first one, two, or three EOF PCs at each of six lags",
            "feature_scaling": "divide each retained lag-PC slot by its training-period standard deviation",
            "architecture": "frozen matched Baseline, zero-PC-identity FiLM at levels 3 and 4",
            "epochs": args.epochs, "seeds": list(SEEDS),
            "checkpoint_selection": "validation ACC",
            "candidate_selection": "both folds' seed-mean validation ACC must exceed the 30-PC reference, averaged-W6, and Baseline",
            "holdout_status": "previously inspected periods are exploratory",
        },
        "folds": {},
    }
    for fold, spec in FOLDS.items():
        x, digest, train_mask, val_mask, _, valid, p90, averaged = fold_data(
            paths, spec, sst, y, dates, windows=(6,),
        )
        trajectory, _, _ = compute_strict_sst_pcs(
            sst, train_mask, 5, 6, keep_lags=True,
        )
        features, scales = temporal_features(trajectory, train_mask)
        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y[:, None])
        mask_t = torch.isnan(y_t) | torch.from_numpy(~valid)[None, None]
        fold_result = {
            "input_sha256": digest,
            "feature_sha256": {name: hashlib.sha256(value.tobytes()).hexdigest()
                               for name, value in features.items()},
            "train_feature_scales": scales,
            "seeds": {},
        }
        for seed in SEEDS:
            prior_base, prior_avg6 = prior_validation(paths, fold, seed)
            prior_trajectory = reference["folds"][fold]["seeds"][str(seed)][
                "trajectory"]["validation"]["mean_acc"]
            rows = {
                "baseline_val_acc": prior_base,
                "averaged_w6_val_acc": prior_avg6,
                "trajectory30_val_acc": prior_trajectory,
                "candidates": {},
            }
            for name in FEATURES:
                pcs_t = torch.from_numpy(features[name])
                train_ds = TensorDataset(y_t[train_mask], x_t[train_mask],
                                         mask_t[train_mask], pcs_t[train_mask])
                val_arrays = (x_t[val_mask], pcs_t[val_mask], y[val_mask], valid, p90)
                model, info, val_pred = train_candidate(
                    name, (3, 4), 0.0, baseline_weights(paths, fold, seed),
                    train_ds, val_arrays, prior_base, device, args.epochs, seed,
                )
                torch.save(model.state_dict(),
                           exp_dir / "checkpoints" / f"{fold}_s{seed}_{name}.pt")
                np.savez_compressed(exp_dir / f"{fold}_s{seed}_{name}_validation.npz",
                                    dates=dates[val_mask].strftime("%Y-%m-%d").values,
                                    prediction=val_pred)
                rows["candidates"][name] = info
                del model
            fold_result["seeds"][str(seed)] = rows
            payload["folds"][fold] = fold_result
            result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fold_means = {}
    for fold in FOLDS:
        rows = payload["folds"][fold]["seeds"]
        fold_means[fold] = {
            name: float(np.mean([
                rows[str(seed)]["candidates"][name]["validation"]["mean_acc"]
                for seed in SEEDS
            ])) for name in FEATURES
        }
        for name, key in (("baseline", "baseline_val_acc"),
                          ("averaged_w6", "averaged_w6_val_acc"),
                          ("trajectory30", "trajectory30_val_acc")):
            fold_means[fold][name] = float(np.mean([
                rows[str(seed)][key] for seed in SEEDS
            ]))
    eligible = [name for name in FEATURES if all(
        fold_means[fold][name] > max(
            fold_means[fold]["baseline"],
            fold_means[fold]["averaged_w6"],
            fold_means[fold]["trajectory30"],
        ) for fold in FOLDS
    )]
    selected = max(eligible, key=lambda name: sum(fold_means[fold][name]
                                                   for fold in FOLDS)) if eligible else None
    payload["selection"] = {
        "fold_mean_val_acc": fold_means,
        "selected_new_feature": selected,
    }
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"temporal feature selection: {payload['selection']}", flush=True)
    if selected is None:
        print("No new temporal feature passed the two-fold validation gate.", flush=True)
        return

    for fold, spec in FOLDS.items():
        x, _, train_mask, _, holdout_mask, valid, p90, _ = fold_data(
            paths, spec, sst, y, dates, windows=(6,),
        )
        trajectory, _, _ = compute_strict_sst_pcs(
            sst, train_mask, 5, 6, keep_lags=True,
        )
        features, _ = temporal_features(trajectory, train_mask)
        xx = torch.from_numpy(x[holdout_mask])
        pp = torch.from_numpy(features[selected][holdout_mask])
        for seed in SEEDS:
            base_weights = baseline_weights(paths, fold, seed)
            baseline = UNetLite(n_channels=10, base_filters=32, dropout=0.1).to(device)
            baseline.load_state_dict(base_weights)
            model = UNetLiteFiLM(
                n_channels=10, index_dim=pp.shape[1], base_filters=32,
                index_hidden=16, dropout=0.1, film_layers=(3, 4),
                identity_init=True, zero_pc_identity=True,
            ).to(device)
            model.load_state_dict(torch.load(
                exp_dir / "checkpoints" / f"{fold}_s{seed}_{selected}.pt",
                map_location=device, weights_only=True,
            ))
            base_pred = predict(baseline, "baseline32", xx, pp, device)
            film_pred = predict(model, "film32_identity", xx, pp, device)
            zero_pred = predict(model, "film32_identity", xx,
                                torch.zeros_like(pp), device)
            base, base_acc = summary(base_pred, y[holdout_mask], valid, p90)
            film, film_acc = summary(film_pred, y[holdout_mask], valid, p90)
            zero, _ = summary(zero_pred, y[holdout_mask], valid, p90)
            payload["folds"][fold]["seeds"][str(seed)]["exploratory_holdout"] = {
                "baseline": base, "film": film, "film_pc_zero": zero,
                "paired_acc": paired_stat_tests(film_acc, base_acc,
                                                 dates[holdout_mask].year.values),
            }
            np.savez_compressed(exp_dir / f"{fold}_s{seed}_{selected}_holdout.npz",
                                dates=dates[holdout_mask].strftime("%Y-%m-%d").values,
                                target=y[holdout_mask], baseline=base_pred,
                                film=film_pred, film_pc_zero=zero_pred)
            print(f"{fold} seed={seed} exploratory holdout ACC "
                  f"FiLM={film['mean_acc']:.4f} Baseline={base['mean_acc']:.4f}", flush=True)
            del baseline, model
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {result_file}", flush=True)


if __name__ == "__main__":
    main()
