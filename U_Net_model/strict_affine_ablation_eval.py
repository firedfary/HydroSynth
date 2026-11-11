"""Compare scale-only and shift-only FiLM with matched full-affine FiLM."""

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


MODES = ("scale", "shift")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--experiment", default="strict_affine_ablation_v1")
    args = parser.parse_args()
    if args.epochs <= 0 or not args.experiment or "/" in args.experiment or "\\" in args.experiment:
        parser.error("epochs must be positive and experiment must be a directory name")

    paths = SubprojectPaths(__file__)
    exp_dir = paths.results_dir / args.experiment
    if exp_dir.exists():
        raise FileExistsError(exp_dir)
    exp_dir = Path(paths.get_exp_dir(args.experiment))
    result_file = exp_dir / "affine_ablation_metrics.json"
    reference = json.loads((paths.results_dir / "strict_trajectory_v1" /
                            "trajectory_metrics.json").read_text(encoding="utf-8"))
    dates = build_date_index()
    sst = np.load(config.modelconfig["sst_file"]).astype(np.float32)
    y = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = {
        "protocol": {
            "modes": list(MODES), "control": "full-affine 30-PC FiLM",
            "conditioning": "six monthly SST fields projected onto five EOFs fitted on fold training period",
            "architecture": "32-channel frozen matched U-Net, FiLM on levels 3 and 4, zero-PC identity",
            "epochs": args.epochs, "seeds": list(SEEDS),
            "checkpoint_selection": "validation ACC",
            "candidate_selection": "both folds' mean validation ACC must exceed full FiLM and Baseline",
            "holdout_status": "previously inspected periods are exploratory",
        },
        "folds": {},
    }
    for fold, spec in FOLDS.items():
        x, digest, train_mask, val_mask, _, valid, p90, _ = fold_data(
            paths, spec, sst, y, dates, windows=(6,),
        )
        pcs, _, _ = compute_strict_sst_pcs(sst, train_mask, 5, 6, keep_lags=True)
        x_t, pcs_t = torch.from_numpy(x), torch.from_numpy(pcs)
        y_t = torch.from_numpy(y[:, None])
        mask_t = torch.isnan(y_t) | torch.from_numpy(~valid)[None, None]
        train_ds = TensorDataset(y_t[train_mask], x_t[train_mask],
                                 mask_t[train_mask], pcs_t[train_mask])
        val_arrays = (x_t[val_mask], pcs_t[val_mask], y[val_mask], valid, p90)
        rows = {
            "input_sha256": digest,
            "pc_sha256": hashlib.sha256(pcs.tobytes()).hexdigest(),
            "seeds": {},
        }
        for seed in SEEDS:
            baseline_val, _ = prior_validation(paths, fold, seed)
            full_val = reference["folds"][fold]["seeds"][str(seed)][
                "trajectory"]["validation"]["mean_acc"]
            seed_rows = {"baseline_val_acc": baseline_val,
                         "full_val_acc": full_val, "candidates": {}}
            for mode in MODES:
                model, info, val_pred = train_candidate(
                    mode, (3, 4), 0.0, baseline_weights(paths, fold, seed),
                    train_ds, val_arrays, baseline_val, device, args.epochs, seed,
                    affine_mode=mode,
                )
                torch.save(model.state_dict(),
                           exp_dir / "checkpoints" / f"{fold}_s{seed}_{mode}.pt")
                np.savez_compressed(exp_dir / f"{fold}_s{seed}_{mode}_validation.npz",
                                    dates=dates[val_mask].strftime("%Y-%m-%d").values,
                                    prediction=val_pred)
                seed_rows["candidates"][mode] = info
                del model
            rows["seeds"][str(seed)] = seed_rows
            payload["folds"][fold] = rows
            result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fold_means = {}
    for fold in FOLDS:
        rows = payload["folds"][fold]["seeds"]
        fold_means[fold] = {
            mode: float(np.mean([
                rows[str(seed)]["candidates"][mode]["validation"]["mean_acc"]
                for seed in SEEDS
            ])) for mode in MODES
        }
        for name, key in (("baseline", "baseline_val_acc"),
                          ("full", "full_val_acc")):
            fold_means[fold][name] = float(np.mean([
                rows[str(seed)][key] for seed in SEEDS
            ]))
    eligible = [mode for mode in MODES if all(
        fold_means[fold][mode] > max(fold_means[fold]["full"],
                                     fold_means[fold]["baseline"])
        for fold in FOLDS
    )]
    selected = max(eligible, key=lambda mode: sum(fold_means[fold][mode]
                                                  for fold in FOLDS)) if eligible else None
    payload["selection"] = {"fold_mean_val_acc": fold_means,
                            "selected_new_mode": selected}
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"affine mode selection: {payload['selection']}", flush=True)
    if selected is None:
        print("Neither affine mode passed the two-fold validation gate.", flush=True)
        return

    for fold, spec in FOLDS.items():
        x, _, train_mask, _, holdout_mask, valid, p90, _ = fold_data(
            paths, spec, sst, y, dates, windows=(6,),
        )
        pcs, _, _ = compute_strict_sst_pcs(sst, train_mask, 5, 6, keep_lags=True)
        xx = torch.from_numpy(x[holdout_mask])
        pp = torch.from_numpy(pcs[holdout_mask])
        for seed in SEEDS:
            base = UNetLite(n_channels=10, base_filters=32, dropout=0.1).to(device)
            base.load_state_dict(baseline_weights(paths, fold, seed))
            model = UNetLiteFiLM(
                n_channels=10, index_dim=30, base_filters=32, index_hidden=16,
                dropout=0.1, film_layers=(3, 4), identity_init=True,
                zero_pc_identity=True, affine_mode=selected,
            ).to(device)
            model.load_state_dict(torch.load(
                exp_dir / "checkpoints" / f"{fold}_s{seed}_{selected}.pt",
                map_location=device, weights_only=True,
            ))
            base_pred = predict(base, "baseline32", xx, pp, device)
            film_pred = predict(model, "film32_identity", xx, pp, device)
            zero_pred = predict(model, "film32_identity", xx,
                                torch.zeros_like(pp), device)
            base_metrics, base_acc = summary(base_pred, y[holdout_mask], valid, p90)
            film_metrics, film_acc = summary(film_pred, y[holdout_mask], valid, p90)
            zero_metrics, _ = summary(zero_pred, y[holdout_mask], valid, p90)
            payload["folds"][fold]["seeds"][str(seed)]["exploratory_holdout"] = {
                "baseline": base_metrics, "film": film_metrics,
                "film_pc_zero": zero_metrics,
                "paired_acc": paired_stat_tests(film_acc, base_acc,
                                                dates[holdout_mask].year.values),
            }
            np.savez_compressed(exp_dir / f"{fold}_s{seed}_{selected}_holdout.npz",
                                dates=dates[holdout_mask].strftime("%Y-%m-%d").values,
                                target=y[holdout_mask], baseline=base_pred,
                                film=film_pred, film_pc_zero=zero_pred)
            print(f"{fold} seed={seed} exploratory holdout ACC "
                  f"FiLM={film_metrics['mean_acc']:.4f} "
                  f"Baseline={base_metrics['mean_acc']:.4f}", flush=True)
            del base, model
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {result_file}", flush=True)


if __name__ == "__main__":
    main()
