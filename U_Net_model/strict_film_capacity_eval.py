"""Test smaller FiLM conditioning heads using two validation folds."""

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


WIDTHS = (4, 8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--experiment", default="film_head_width_v1")
    args = parser.parse_args()
    if args.epochs <= 0 or not args.experiment or "/" in args.experiment or "\\" in args.experiment:
        parser.error("epochs must be positive and experiment must be a directory name")
    paths = SubprojectPaths(__file__)
    if (paths.results_dir / args.experiment).exists():
        raise FileExistsError(paths.results_dir / args.experiment)
    exp_dir = Path(paths.get_exp_dir(args.experiment))
    result_file = exp_dir / "capacity_metrics.json"
    dates = build_date_index()
    sst = np.load(config.modelconfig["sst_file"]).astype(np.float32)
    y = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    original = json.loads((paths.results_dir / "film_early_v1" /
                           "trajectory_metrics.json").read_text(encoding="utf-8"))
    payload = {"protocol": {
        "widths": list(WIDTHS), "reference_width": 16,
        "model": "30-PC zero-identity FiLM at encoder levels 1-4; frozen Baseline",
        "epochs": args.epochs, "seeds": list(SEEDS), "batch_size": 8,
        "lr": 2e-4, "weight_decay": 1e-4, "penalty_weight": 0.1,
        "loss": "masked MSE", "checkpoint_selection": "validation ACC",
        "candidate_gate": "both fold seed-mean validation ACC exceed width 16 and Baseline",
        "holdout_status": "previously inspected; exploratory only",
    }, "folds": {}}

    for fold, spec in FOLDS.items():
        x, digest, train_mask, val_mask, _, valid, p90, _ = fold_data(
            paths, spec, sst, y, dates, windows=(6,),
        )
        pcs, _, _ = compute_strict_sst_pcs(sst, train_mask, 5, 6, keep_lags=True)
        pcs_hash = hashlib.sha256(pcs.tobytes()).hexdigest()
        if (original["folds"][fold]["input_sha256"] != digest or
                original["folds"][fold]["trajectory_sha256"] != pcs_hash):
            raise ValueError(f"source mismatch for {fold}")
        x_t, pcs_t = torch.from_numpy(x), torch.from_numpy(pcs)
        y_t = torch.from_numpy(y[:, None])
        mask_t = torch.isnan(y_t) | torch.from_numpy(~valid)[None, None]
        train_ds = TensorDataset(y_t[train_mask], x_t[train_mask],
                                 mask_t[train_mask], pcs_t[train_mask])
        val_arrays = (x_t[val_mask], pcs_t[val_mask], y[val_mask], valid, p90)
        fold_result = {"input_sha256": digest, "trajectory_sha256": pcs_hash,
                       "seeds": {}}
        payload["folds"][fold] = fold_result
        for seed in SEEDS:
            baseline_val, _ = prior_validation(paths, fold, seed)
            source_row = original["folds"][fold]["seeds"][str(seed)]
            original_val = source_row["trajectory"]["validation"]["mean_acc"]
            row = {"baseline_val_acc": baseline_val,
                   "width16_val_acc": original_val, "widths": {}}
            for width in WIDTHS:
                model, info, pred = train_candidate(
                    f"head{width}", (1, 2, 3, 4), 0.0,
                    baseline_weights(paths, fold, seed), train_ds, val_arrays,
                    baseline_val, device, args.epochs, seed, index_hidden=width,
                )
                info["trainable_parameters"] = sum(
                    parameter.numel() for parameter in model.index_encoder.parameters()
                )
                torch.save(model.state_dict(),
                           exp_dir / "checkpoints" / f"{fold}_s{seed}_w{width}.pt")
                np.savez_compressed(exp_dir / f"{fold}_s{seed}_w{width}_validation.npz",
                                    dates=dates[val_mask].strftime("%Y-%m-%d").values,
                                    prediction=pred)
                row["widths"][str(width)] = info
                del model
            fold_result["seeds"][str(seed)] = row
            result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    means = {}
    for fold in FOLDS:
        rows = payload["folds"][fold]["seeds"]
        means[fold] = {
            "baseline": float(np.mean([rows[str(s)]["baseline_val_acc"] for s in SEEDS])),
            "width16": float(np.mean([rows[str(s)]["width16_val_acc"] for s in SEEDS])),
        }
        for width in WIDTHS:
            means[fold][str(width)] = float(np.mean([
                rows[str(s)]["widths"][str(width)]["validation"]["mean_acc"]
                for s in SEEDS
            ]))
    eligible = [width for width in WIDTHS if all(
        means[fold][str(width)] > max(means[fold]["baseline"], means[fold]["width16"])
        for fold in FOLDS
    )]
    selected = max(eligible, key=lambda width: sum(
        means[fold][str(width)] for fold in FOLDS
    )) if eligible else None
    payload["selection"] = {"fold_mean_validation_acc": means,
                            "eligible": eligible, "selected_width": selected}
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"capacity selection: {payload['selection']}", flush=True)
    if selected is None:
        return

    for fold, spec in FOLDS.items():
        x, _, train_mask, _, holdout_mask, valid, p90, _ = fold_data(
            paths, spec, sst, y, dates, windows=(6,),
        )
        pcs, _, _ = compute_strict_sst_pcs(sst, train_mask, 5, 6, keep_lags=True)
        xx, pp = torch.from_numpy(x[holdout_mask]), torch.from_numpy(pcs[holdout_mask])
        members = {"candidate": [], "baseline": [], "width16": []}
        for seed in SEEDS:
            model = UNetLiteFiLM(
                n_channels=10, index_dim=30, base_filters=32, index_hidden=selected,
                dropout=0.1, film_layers=(1, 2, 3, 4), identity_init=True,
                zero_pc_identity=True,
            ).to(device)
            model.load_state_dict(torch.load(
                exp_dir / "checkpoints" / f"{fold}_s{seed}_w{selected}.pt",
                map_location=device, weights_only=True,
            ))
            members["candidate"].append(predict(model, "film32_identity", xx, pp, device))
            del model
            baseline = UNetLite(n_channels=10, base_filters=32, dropout=0.1).to(device)
            baseline.load_state_dict(baseline_weights(paths, fold, seed))
            members["baseline"].append(predict(baseline, "baseline32", xx, pp, device))
            del baseline
            prior_file = paths.results_dir / "film_early_v1" / f"{fold}_s{seed}_holdout.npz"
            with np.load(prior_file, allow_pickle=True) as prior:
                if (prior["trajectory"].shape != y[holdout_mask].shape or
                        not np.allclose(prior["target"], y[holdout_mask], equal_nan=True)):
                    raise ValueError(f"prior target mismatch: {prior_file}")
                members["width16"].append(prior["trajectory"].copy())
        metrics, acc = {}, {}
        predictions = {name: np.mean(preds, axis=0) for name, preds in members.items()}
        for name, pred in predictions.items():
            metrics[name], acc[name] = summary(pred, y[holdout_mask], valid, p90)
        payload["folds"][fold]["exploratory_holdout"] = {
            "metrics": metrics,
            "paired_candidate_minus_baseline": paired_stat_tests(
                acc["candidate"], acc["baseline"], dates[holdout_mask].year.values),
            "paired_candidate_minus_width16": paired_stat_tests(
                acc["candidate"], acc["width16"], dates[holdout_mask].year.values),
        }
        np.savez_compressed(exp_dir / f"{fold}_holdout.npz",
                            dates=dates[holdout_mask].strftime("%Y-%m-%d").values,
                            target=y[holdout_mask], **predictions)
        print(f"{fold} exploratory holdout ACC " + " ".join(
            f"{name}={row['mean_acc']:.5f}" for name, row in metrics.items()
        ), flush=True)
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {result_file}", flush=True)


if __name__ == "__main__":
    main()
