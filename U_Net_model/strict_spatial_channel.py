"""Three-fold validation of channel-specific bounded spatial FiLM."""

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
from run_unified_eval import build_date_index, compute_strict_sst_pcs
from strict_centered_eval import train_candidate
from strict_decoder_continue import FOLDS, SEEDS, SOURCE, source_info
from strict_sst_window_eval import baseline_weights, fold_data


def prior_validation(prior, fold, seed):
    if fold == "early":
        metrics = prior["seeds"][str(seed)]["validation"]["metrics"]
        return metrics["baseline"]["mean_acc"], metrics["film"]["mean_acc"]
    row = prior["folds"][fold]["seeds"][str(seed)]
    return row["baseline_val_acc"], row["trajectory"]["validation"]["mean_acc"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--experiment", default="film_spatial_ch_v1")
    args = parser.parse_args()
    if (args.epochs <= 0 or not args.experiment or "/" in args.experiment or
            "\\" in args.experiment):
        parser.error("invalid epochs or experiment name")
    paths = SubprojectPaths(__file__)
    if (paths.results_dir / args.experiment).exists():
        raise FileExistsError(paths.results_dir / args.experiment)
    exp_dir = Path(paths.get_exp_dir(args.experiment))
    result_file = exp_dir / "spatial_metrics.json"
    dates = build_date_index()
    sst = np.load(config.modelconfig["sst_file"]).astype(np.float32)
    y = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = {
        "protocol": {
            "candidate": "channel-specific bounded latitude-longitude FiLM at levels 1-4",
            "sources": SOURCE, "seeds": list(SEEDS), "epochs": args.epochs,
            "backbone": "frozen 32-channel Baseline",
            "conditioning": "six monthly vectors of five SST EOF PCs",
            "loss": "masked MSE plus 0.1 FiLM penalty",
            "checkpoint_selection": "validation ACC for each seed",
            "selection_gate": "candidate exceeds original shallow FiLM and Baseline on all three fold mean validation ACC values",
            "holdout_status": "not accessed by this experiment",
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
        x_t, pcs_t = torch.from_numpy(x), torch.from_numpy(pcs)
        y_t = torch.from_numpy(y[:, None])
        mask_t = torch.isnan(y_t) | torch.from_numpy(~valid)[None, None]
        dataset = TensorDataset(y_t[train_mask], x_t[train_mask],
                                mask_t[train_mask], pcs_t[train_mask])
        val_arrays = (x_t[val_mask], pcs_t[val_mask], y[val_mask], valid, p90)
        row = {"input_sha256": digest, "trajectory_sha256": pc_digest, "seeds": {}}
        for seed in SEEDS:
            baseline_acc, old_film_acc = prior_validation(prior, fold, seed)
            if fold == "early":
                base_weights = torch.load(source / "checkpoints" /
                                          f"baseline_s{seed}.pt", map_location="cpu",
                                          weights_only=True)
            else:
                base_weights = baseline_weights(paths, fold, seed)
            model, info, _ = train_candidate(
                "spatial_channel", (1, 2, 3, 4), 0.0, base_weights,
                dataset, val_arrays, baseline_acc, device, args.epochs, seed,
                spatial_film=True, spatial_per_channel=True,
            )
            torch.save(model.state_dict(), exp_dir / "checkpoints" /
                       f"{fold}_s{seed}.pt")
            row["seeds"][str(seed)] = {
                "baseline_val_acc": baseline_acc,
                "old_film_val_acc": old_film_acc,
                "candidate": info,
            }
            payload["folds"][fold] = row
            result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"completed {fold}", flush=True)
    means = {}
    for fold, row in payload["folds"].items():
        means[fold] = {
            name: float(np.mean([row["seeds"][str(seed)][key] for seed in SEEDS]))
            for name, key in (("baseline", "baseline_val_acc"),
                              ("old_film", "old_film_val_acc"))
        }
        means[fold]["candidate"] = float(np.mean([
            row["seeds"][str(seed)]["candidate"]["validation"]["mean_acc"]
            for seed in SEEDS
        ]))
    payload["selection"] = {
        "fold_mean_val_acc": means,
        "passed_three_fold_gate": all(
            value["candidate"] > max(value["baseline"], value["old_film"])
            for value in means.values()
        ),
    }
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"selection: {payload['selection']}", flush=True)


if __name__ == "__main__":
    main()
