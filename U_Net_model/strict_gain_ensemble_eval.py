"""Select one shared modulation gain for the fixed three-seed early FiLM ensemble."""

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
from unetlitefilm import UNetLiteFiLM
from film_gain_eval import predict_gain
from run_unified_eval import build_date_index, compute_strict_sst_pcs, paired_stat_tests
from strict_fair_eval import predict, summary
from strict_sst_window_eval import FOLDS, SEEDS, baseline_weights, fold_data


GAINS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)


def load_film(paths, fold, seed, device):
    model = UNetLiteFiLM(
        n_channels=10, index_dim=30, base_filters=32, index_hidden=16,
        dropout=0.1, film_layers=(1, 2, 3, 4), identity_init=True,
        zero_pc_identity=True,
    ).to(device)
    checkpoint = paths.results_dir / "film_early_v1" / "checkpoints" / f"{fold}_s{seed}.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="film_gain_ensemble_v1")
    args = parser.parse_args()
    if not args.experiment or "/" in args.experiment or "\\" in args.experiment:
        parser.error("experiment must be a directory name")
    paths = SubprojectPaths(__file__)
    if (paths.results_dir / args.experiment).exists():
        raise FileExistsError(paths.results_dir / args.experiment)
    exp_dir = Path(paths.get_exp_dir(args.experiment))
    result_file = exp_dir / "gain_metrics.json"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dates = build_date_index()
    sst = np.load(config.modelconfig["sst_file"]).astype(np.float32)
    y = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    source = json.loads((paths.results_dir / "film_early_v1" /
                         "trajectory_metrics.json").read_text(encoding="utf-8"))
    payload = {"protocol": {
        "gains": list(GAINS), "seeds": list(SEEDS),
        "model": "frozen 32-channel Baseline with 30-PC FiLM at levels 1-4",
        "selection": "shared gain must improve ensemble validation ACC over gain 1 in both folds; maximize summed validation ACC",
        "holdout_status": "previously inspected; exploratory only",
    }, "folds": {}}

    for fold, spec in FOLDS.items():
        x, digest, train_mask, val_mask, holdout_mask, valid, p90, _ = fold_data(
            paths, spec, sst, y, dates, windows=(6,),
        )
        pcs, _, _ = compute_strict_sst_pcs(sst, train_mask, 5, 6, keep_lags=True)
        pcs_hash = hashlib.sha256(pcs.tobytes()).hexdigest()
        if (source["folds"][fold]["input_sha256"] != digest or
                source["folds"][fold]["trajectory_sha256"] != pcs_hash):
            raise ValueError(f"source mismatch for {fold}")
        xx, pp = torch.from_numpy(x[val_mask]), torch.from_numpy(pcs[val_mask])
        members = {gain: [] for gain in GAINS}
        base_members = []
        for seed in SEEDS:
            model = load_film(paths, fold, seed, device)
            baseline = UNetLite(n_channels=10, base_filters=32, dropout=0.1).to(device)
            baseline.load_state_dict(baseline_weights(paths, fold, seed))
            base_pred = predict(baseline, "baseline32", xx, pp, device)
            base_members.append(base_pred)
            for gain in GAINS:
                pred = predict_gain(model, xx, pp, gain, device)
                if gain == 0.0 and float(np.max(np.abs(pred - base_pred))) > 1e-5:
                    raise ValueError(f"zero-gain FiLM differs from Baseline: {fold}, {seed}")
                members[gain].append(pred)
            del model, baseline
        base_pred = np.mean(base_members, axis=0)
        base_metrics, _ = summary(base_pred, y[val_mask], valid, p90)
        gain_metrics = {}
        for gain in GAINS:
            ensemble_pred = np.mean(members[gain], axis=0)
            gain_metrics[str(gain)], _ = summary(ensemble_pred, y[val_mask], valid, p90)
        payload["folds"][fold] = {
            "input_sha256": digest, "trajectory_sha256": pcs_hash,
            "baseline_validation": base_metrics, "gains_validation": gain_metrics,
        }
        result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"{fold} validation ACC " + " ".join(
            f"{gain:g}={gain_metrics[str(gain)]['mean_acc']:.5f}" for gain in GAINS
        ), flush=True)

    eligible = [gain for gain in GAINS if gain != 1.0 and all(
        payload["folds"][fold]["gains_validation"][str(gain)]["mean_acc"] >
        payload["folds"][fold]["gains_validation"]["1.0"]["mean_acc"]
        for fold in FOLDS
    )]
    selected = max(eligible, key=lambda gain: sum(
        payload["folds"][fold]["gains_validation"][str(gain)]["mean_acc"]
        for fold in FOLDS
    )) if eligible else None
    payload["selection"] = {"eligible": eligible, "selected_gain": selected}
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"gain selection: {payload['selection']}", flush=True)
    if selected is None:
        return

    for fold, spec in FOLDS.items():
        x, _, train_mask, _, holdout_mask, valid, p90, _ = fold_data(
            paths, spec, sst, y, dates, windows=(6,),
        )
        pcs, _, _ = compute_strict_sst_pcs(sst, train_mask, 5, 6, keep_lags=True)
        xx, pp = torch.from_numpy(x[holdout_mask]), torch.from_numpy(pcs[holdout_mask])
        selected_members, original_members, base_members = [], [], []
        for seed in SEEDS:
            model = load_film(paths, fold, seed, device)
            selected_members.append(predict_gain(model, xx, pp, selected, device))
            original_members.append(predict_gain(model, xx, pp, 1.0, device))
            baseline = UNetLite(n_channels=10, base_filters=32, dropout=0.1).to(device)
            baseline.load_state_dict(baseline_weights(paths, fold, seed))
            base_members.append(predict(baseline, "baseline32", xx, pp, device))
            del model, baseline
        predictions = {"selected": np.mean(selected_members, axis=0),
                       "original": np.mean(original_members, axis=0),
                       "baseline": np.mean(base_members, axis=0)}
        metrics, acc = {}, {}
        for name, pred in predictions.items():
            metrics[name], acc[name] = summary(pred, y[holdout_mask], valid, p90)
        payload["folds"][fold]["exploratory_holdout"] = {
            "metrics": metrics,
            "paired_selected_minus_original": paired_stat_tests(
                acc["selected"], acc["original"], dates[holdout_mask].year.values),
            "paired_selected_minus_baseline": paired_stat_tests(
                acc["selected"], acc["baseline"], dates[holdout_mask].year.values),
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
