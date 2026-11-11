"""Historical rolling-origin replication of the validation-selected FiLM design."""

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
from run_unified_eval import build_date_index, compute_strict_sst_pcs, paired_stat_tests
from strict_centered_eval import train_candidate
from strict_fair_eval import predict, summary, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--experiment", default="strict_replication_v1")
    args = parser.parse_args()
    if args.epochs <= 0 or not args.experiment or "/" in args.experiment or "\\" in args.experiment:
        parser.error("epochs must be positive and experiment must be a directory name")
    paths = SubprojectPaths(__file__)
    input_dir = paths.cache_dir / "strict_inputs_pre2011"
    input_file = input_dir / "cond_trainfit.npy"
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    input_hash = hashlib.sha256(input_file.read_bytes()).hexdigest()
    if input_hash != manifest["sha256"] or manifest["train_period"] != "1994-01/2010-12":
        raise ValueError("historical predictor preprocessing is not train-fitted")
    exp_dir = paths.results_dir / args.experiment
    if exp_dir.exists():
        raise FileExistsError(exp_dir)
    exp_dir = Path(paths.get_exp_dir(args.experiment))

    dates = build_date_index()
    train_mask = np.asarray(dates <= "2010-12-01")
    val_mask = np.asarray((dates >= "2011-01-01") & (dates <= "2013-12-01"))
    holdout_mask = np.asarray((dates >= "2014-01-01") & (dates <= "2017-12-01"))
    x = np.load(input_file).astype(np.float32)
    y = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    sst = np.load(config.modelconfig["sst_file"]).astype(np.float32)
    pcs, _, explained = compute_strict_sst_pcs(sst, train_mask, 5)
    valid = np.isfinite(y).all(axis=0)
    p90 = float(np.percentile(y[train_mask][:, valid], 90))
    x_t, pcs_t = torch.from_numpy(x), torch.from_numpy(pcs)
    y_t = torch.from_numpy(y[:, None])
    mask_t = torch.isnan(y_t) | torch.from_numpy(~valid)[None, None]
    train_ds = TensorDataset(y_t[train_mask], x_t[train_mask],
                             mask_t[train_mask], pcs_t[train_mask])
    val_ds = TensorDataset(y_t[val_mask], x_t[val_mask],
                           mask_t[val_mask], pcs_t[val_mask])
    val_arrays = (x_t[val_mask], pcs_t[val_mask], y[val_mask], valid, p90)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = {
        "protocol": {"fit_and_train": "1994-01/2010-12",
                     "checkpoint_validation": "2011-01/2013-12",
                     "historical_holdout": "2014-01/2017-12",
                     "epochs": args.epochs, "seeds": [42, 43, 44],
                     "film_design": "zero-PC identity, levels 3 and 4, hidden 16, frozen baseline, MSE loss",
                     "design_status": "fixed after the later 2018-2022 validation experiment"},
        "input_sha256": input_hash,
        "sst_pca_explained_variance": [float(value) for value in explained],
        "seeds": {},
    }
    for seed in (42, 43, 44):
        baseline, base_info = train("baseline32", train_ds, val_ds, device,
                                    args.epochs, 2e-4, 0.1, seed)
        base_weights = {key: value.detach().cpu().clone()
                        for key, value in baseline.state_dict().items()}
        torch.save(base_weights, exp_dir / "checkpoints" / f"baseline_s{seed}.pt")
        base_val_pred = predict(baseline, "baseline32", x_t[val_mask], pcs_t[val_mask], device)
        base_val, _ = summary(base_val_pred, y[val_mask], valid, p90)
        base_holdout_pred = predict(baseline, "baseline32",
                                    x_t[holdout_mask], pcs_t[holdout_mask], device)
        base_holdout, base_acc = summary(base_holdout_pred, y[holdout_mask], valid, p90)
        del baseline

        film, film_info, film_val_pred = train_candidate(
            "deep_mse", (3, 4), 0.0, base_weights, train_ds,
            val_arrays, base_val["mean_acc"], device, args.epochs, seed,
        )
        torch.save(film.state_dict(), exp_dir / "checkpoints" / f"film_s{seed}.pt")
        film_holdout_pred = predict(film, "film32_identity",
                                     x_t[holdout_mask], pcs_t[holdout_mask], device)
        film_zero_pred = predict(film, "film32_identity", x_t[holdout_mask],
                                  torch.zeros_like(pcs_t[holdout_mask]), device)
        film_holdout, film_acc = summary(film_holdout_pred, y[holdout_mask], valid, p90)
        zero_holdout, _ = summary(film_zero_pred, y[holdout_mask], valid, p90)
        np.savez_compressed(exp_dir / f"s{seed}_predictions.npz",
                            val_dates=dates[val_mask].strftime("%Y-%m-%d").values,
                            base_val=base_val_pred, film_val=film_val_pred,
                            holdout_dates=dates[holdout_mask].strftime("%Y-%m-%d").values,
                            target=y[holdout_mask], baseline=base_holdout_pred,
                            film=film_holdout_pred, film_pc_zero=film_zero_pred)
        payload["seeds"][str(seed)] = {
            "baseline_checkpoint_epoch": base_info["best_epoch"],
            "baseline_validation": base_val, "baseline_holdout": base_holdout,
            "film": film_info, "film_holdout": film_holdout,
            "film_pc_zero_holdout": zero_holdout,
            "paired_acc": paired_stat_tests(film_acc, base_acc,
                                             dates[holdout_mask].year.values),
        }
        (exp_dir / "strict_replication_metrics.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
        print(f"seed={seed} historical ACC FiLM={film_holdout['mean_acc']:.4f} "
              f"Baseline={base_holdout['mean_acc']:.4f}", flush=True)
        del film
    result_file = exp_dir / "strict_replication_metrics.json"
    print(f"Saved {result_file}", flush=True)


if __name__ == "__main__":
    main()
