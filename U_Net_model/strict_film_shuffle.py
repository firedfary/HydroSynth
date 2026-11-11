"""Check whether shallow FiLM validation skill needs correctly paired SST PCs."""

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
from run_unified_eval import build_date_index, compute_strict_sst_pcs
from strict_decoder_continue import FOLDS, SEEDS, SOURCE, make_model, source_info
from strict_fair_eval import predict, summary
from strict_sst_window_eval import fold_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--experiment", default="film_shuffle_v1")
    args = parser.parse_args()
    if (args.permutations <= 0 or not args.experiment or "/" in args.experiment or
            "\\" in args.experiment):
        parser.error("invalid permutations or experiment name")
    paths = SubprojectPaths(__file__)
    if (paths.results_dir / args.experiment).exists():
        raise FileExistsError(paths.results_dir / args.experiment)
    dates = build_date_index()
    sst = np.load(config.modelconfig["sst_file"]).astype(np.float32)
    y = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(20261112)
    payload = {
        "protocol": {
            "sources": SOURCE, "periods": "validation only", "seeds": list(SEEDS),
            "ensemble": "equal-weight predictions from three fixed checkpoints",
            "permutation": "shuffle SST PC vectors across years within calendar month",
            "permutations": args.permutations, "rng_seed": 20261112,
            "status": "post-selection mechanism diagnostic; no holdout accessed",
        }, "folds": {},
    }
    for fold, spec in FOLDS.items():
        x, digest, train_mask, val_mask, _, valid, p90, _ = fold_data(
            paths, spec, sst, y, dates, windows=(6,),
        )
        pcs, _, _ = compute_strict_sst_pcs(sst, train_mask, 5, 6, keep_lags=True)
        pc_digest = hashlib.sha256(pcs.tobytes()).hexdigest()
        source = source_info(paths, fold, digest, pc_digest)
        prior_file = ("early_origin_metrics.json" if fold == "early" else
                      "trajectory_metrics.json")
        prior = json.loads((source / prior_file).read_text(encoding="utf-8"))
        xx = torch.from_numpy(x[val_mask])
        pp = pcs[val_mask]
        yy = y[val_mask]
        month = np.asarray(dates[val_mask].month)
        models = [make_model(paths, fold, seed, "film", source, device).eval()
                  for seed in SEEDS]

        def ensemble(pc_values):
            pc_t = torch.from_numpy(pc_values.copy())
            return np.mean([predict(model, "film32_identity", xx, pc_t, device)
                            for model in models], axis=0)

        per_seed_acc = []
        for seed, model in zip(SEEDS, models):
            pred = predict(model, "film32_identity", xx, torch.from_numpy(pp), device)
            acc = summary(pred, yy, valid, p90)[0]["mean_acc"]
            reported = (prior["seeds"][str(seed)]["validation"]["metrics"]
                        ["film"]["mean_acc"] if fold == "early" else
                        prior["folds"][fold]["seeds"][str(seed)]
                        ["trajectory"]["validation"]["mean_acc"])
            if abs(acc - reported) > 1e-5:
                raise ValueError(f"{fold} seed={seed} source validation ACC differs")
            per_seed_acc.append(acc)
        correct = summary(ensemble(pp), yy, valid, p90)[0]
        zero = summary(ensemble(np.zeros_like(pp)), yy, valid, p90)[0]
        permuted_acc = []
        for index in range(args.permutations):
            permutation = np.arange(len(pp))
            for m in range(1, 13):
                positions = np.flatnonzero(month == m)
                permutation[positions] = rng.permutation(positions)
            acc = summary(ensemble(pp[permutation]), yy, valid, p90)[0]["mean_acc"]
            permuted_acc.append(acc)
            if (index + 1) % 20 == 0:
                print(f"{fold} permutation {index + 1}/{args.permutations}", flush=True)
        arr = np.asarray(permuted_acc)
        payload["folds"][fold] = {
            "input_sha256": digest, "trajectory_sha256": pc_digest,
            "per_seed_correct_acc": per_seed_acc,
            "correct": correct, "pc_zero": zero,
            "permutation": {
                "mean_acc": float(np.mean(arr)), "std_acc": float(np.std(arr)),
                "pct_at_least_correct": float(np.mean(arr >= correct["mean_acc"])),
                "acc_values": permuted_acc,
            },
        }
        print(f"{fold} ACC correct={correct['mean_acc']:.5f} "
              f"zero={zero['mean_acc']:.5f} "
              f"shuffled={np.mean(arr):.5f}", flush=True)
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()
    exp_dir = Path(paths.get_exp_dir(args.experiment))
    result_file = exp_dir / "shuffle_metrics.json"
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {result_file}", flush=True)


if __name__ == "__main__":
    main()
