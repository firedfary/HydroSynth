"""Probe SST dependence of the early-origin EOF-cat holdout ensemble."""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="film_early_origin_v1")
    parser.add_argument("--experiment", default="early_cat_ablate_v1")
    parser.add_argument("--permutations", type=int, default=100)
    args = parser.parse_args()
    if (args.permutations <= 0 or not args.experiment or "/" in args.experiment or
            "\\" in args.experiment):
        parser.error("invalid experiment or permutation count")
    paths = SubprojectPaths(__file__)
    if (paths.results_dir / args.experiment).exists():
        raise FileExistsError(paths.results_dir / args.experiment)
    source = paths.results_dir / args.source
    previous = json.loads((source / "early_origin_metrics.json").read_text(encoding="utf-8"))
    input_file = paths.cache_dir / "strict_pre2005" / "cond_trainfit.npy"
    digest = hashlib.sha256(input_file.read_bytes()).hexdigest()
    if digest != previous["input_sha256"]:
        raise ValueError("predictor cache differs from source run")
    dates = build_date_index()
    train_mask = np.asarray(dates <= "2004-12-01")
    holdout_mask = np.asarray((dates >= "2008-01-01") & (dates <= "2010-12-01"))
    x = np.load(input_file).astype(np.float32)
    sst = np.load(config.modelconfig["sst_file"]).astype(np.float32)
    y = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    pcs, _, _ = compute_strict_sst_pcs(sst, train_mask, 5, 6, keep_lags=True)
    if hashlib.sha256(pcs.tobytes()).hexdigest() != previous["trajectory_sha256"]:
        raise ValueError("SST PCs differ from source run")
    valid = np.isfinite(y).all(axis=0)
    p90 = float(np.percentile(y[train_mask][:, valid], 90))
    xx = torch.from_numpy(x[holdout_mask])
    pp = pcs[holdout_mask]
    yy = y[holdout_mask]
    month = np.asarray(dates[holdout_mask].month)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = []
    for seed in (42, 43, 44):
        model = UNetLite(n_channels=40, base_filters=32, dropout=0.1).to(device)
        model.load_state_dict(torch.load(source / "checkpoints" / f"eof_cat_s{seed}.pt",
                                         map_location=device, weights_only=True))
        models.append(model.eval())

    def ensemble(pc_values):
        pc_t = torch.from_numpy(pc_values.copy())
        return np.mean([predict(model, "eof_cat32", xx, pc_t, device)
                        for model in models], axis=0)

    with np.load(source / "ensemble_holdout.npz", allow_pickle=True) as saved:
        if (not np.array_equal(saved["dates"],
                               dates[holdout_mask].strftime("%Y-%m-%d").values) or
                not np.allclose(saved["target"], yy, equal_nan=True)):
            raise ValueError("source holdout target or dates mismatch")
        original_pred = saved["eof_cat"].copy()
    reproduced = ensemble(pp)
    max_diff = float(np.max(np.abs(reproduced - original_pred)))
    if max_diff > 1e-5:
        raise ValueError(f"source EOF-cat predictions differ: {max_diff}")
    actual, _ = summary(original_pred, yy, valid, p90)
    zero, _ = summary(ensemble(np.zeros_like(pp)), yy, valid, p90)
    rng = np.random.default_rng(20261112)
    permuted_acc = []
    for index in range(args.permutations):
        permutation = np.arange(len(pp))
        for m in range(1, 13):
            positions = np.flatnonzero(month == m)
            permutation[positions] = rng.permutation(positions)
        pred = ensemble(pp[permutation])
        metrics, _ = summary(pred, yy, valid, p90)
        permuted_acc.append(metrics["mean_acc"])
        if (index + 1) % 20 == 0:
            print(f"Completed {index + 1}/{args.permutations} permutations", flush=True)
    exp_dir = Path(paths.get_exp_dir(args.experiment))
    payload = {
        "protocol": {
            "source": args.source, "period": "2008-01/2010-12",
            "ensemble": "same three EOF-cat checkpoints, fixed equal weights",
            "zero": "all 30 PC features set to zero",
            "permutation": "shuffle PC vectors across years within each calendar month",
            "permutations": args.permutations, "rng_seed": 20261112,
            "status": "mechanism diagnostic after holdout inspection; not model selection",
        },
        "input_sha256": digest,
        "trajectory_sha256": previous["trajectory_sha256"],
        "prediction_reproduction_max_abs_diff": max_diff,
        "actual": actual, "pc_zero": zero,
        "permutation": {
            "mean_acc": float(np.mean(permuted_acc)),
            "std_acc": float(np.std(permuted_acc)),
            "min_acc": float(np.min(permuted_acc)),
            "max_acc": float(np.max(permuted_acc)),
            "pct_at_least_actual": float(np.mean(np.asarray(permuted_acc) >= actual["mean_acc"])),
            "acc_values": permuted_acc,
        },
    }
    result_file = exp_dir / "ablation_metrics.json"
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"ACC actual={actual['mean_acc']:.5f}, zero={zero['mean_acc']:.5f}, "
          f"permuted mean={np.mean(permuted_acc):.5f}", flush=True)
    print(f"Saved {result_file}", flush=True)


if __name__ == "__main__":
    main()
