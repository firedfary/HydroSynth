"""Retrospective rolling-origin check of the fixed early FiLM design."""

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
from strict_fair_eval import predict, summary, train


SEEDS = (42, 43, 44)
PERIODS = {
    "train": ("1994-01-01", "2004-12-01"),
    "validation": ("2005-01-01", "2007-12-01"),
    "holdout": ("2008-01-01", "2010-12-01"),
}


def metric_rows(predictions, obs, valid, p90, years):
    metrics, acc = {}, {}
    for name, pred in predictions.items():
        metrics[name], acc[name] = summary(pred, obs, valid, p90)
    paired = {name: paired_stat_tests(acc["film"], acc[name], years)
              for name in ("baseline", "eof_cat")}
    return {"metrics": metrics, "paired_film_minus": paired}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--experiment", default="film_early_origin_v1")
    args = parser.parse_args()
    if args.epochs <= 0 or not args.experiment or "/" in args.experiment or "\\" in args.experiment:
        parser.error("epochs must be positive and experiment must be a directory name")
    paths = SubprojectPaths(__file__)
    if (paths.results_dir / args.experiment).exists():
        raise FileExistsError(paths.results_dir / args.experiment)
    cache_dir = paths.cache_dir / "strict_pre2005"
    input_file = cache_dir / "cond_trainfit.npy"
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    input_hash = hashlib.sha256(input_file.read_bytes()).hexdigest()
    if (input_hash != manifest["sha256"] or
            manifest["train_period"] != "1994-01/2004-12" or
            manifest["validation_period"] != "2005-01/2007-12"):
        raise ValueError("new rolling-origin predictors do not match the fixed protocol")
    exp_dir = Path(paths.get_exp_dir(args.experiment))
    result_file = exp_dir / "early_origin_metrics.json"
    dates = build_date_index()
    masks = {name: np.asarray((dates >= start) & (dates <= end))
             for name, (start, end) in PERIODS.items()}
    x = np.load(input_file).astype(np.float32)
    y = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    sst = np.load(config.modelconfig["sst_file"]).astype(np.float32)
    if (x.shape != (len(dates), 10, 120, 140) or
            y.shape != (len(dates), 120, 140) or
            sst.shape != (len(dates), 6, 89, 180)):
        raise ValueError("input, target, or SST chronology differs from the fixed protocol")
    valid = np.isfinite(y).all(axis=0)
    p90 = float(np.percentile(y[masks["train"]][:, valid], 90))
    pcs, _, explained = compute_strict_sst_pcs(sst, masks["train"], 5, 6, keep_lags=True)
    if pcs.shape != (len(dates), 30) or not np.isfinite(pcs).all():
        raise ValueError("invalid 30-PC conditioning")
    x_t, pcs_t = torch.from_numpy(x), torch.from_numpy(pcs)
    y_t = torch.from_numpy(y[:, None])
    mask_t = torch.isnan(y_t) | torch.from_numpy(~valid)[None, None]
    datasets = {name: TensorDataset(y_t[mask], x_t[mask], mask_t[mask], pcs_t[mask])
                for name, mask in masks.items() if name != "holdout"}
    val_arrays = (x_t[masks["validation"]], pcs_t[masks["validation"]],
                  y[masks["validation"]], valid, p90)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = {
        "protocol": {
            "periods": {name: f"{start[:7]}/{end[:7]}"
                        for name, (start, end) in PERIODS.items()},
            "seeds": list(SEEDS), "epochs": args.epochs,
            "conditioning": "six monthly SST EOF vectors, five PCs per month",
            "eof_fit": "1994-01/2004-12 only",
            "models": {
                "baseline": "32-channel U-Net, from scratch",
                "eof_cat": "32-channel U-Net with 30 broadcast PC maps, from scratch",
                "film": "32-channel frozen Baseline plus zero-PC-identity FiLM at levels 1-4",
            },
            "backbone_lr": 2e-4, "film_head_lr": 2e-4, "dropout": 0.1,
            "batch_size": 8, "loss": "masked MSE; FiLM head penalty 0.1",
            "checkpoint_selection": "validation ACC for each seed and model",
            "ensemble": "equal-weight predictions from all three fixed seeds",
            "evidence_status": "retrospective robustness; model was developed using later periods",
        },
        "input_sha256": input_hash,
        "trajectory_sha256": hashlib.sha256(pcs.tobytes()).hexdigest(),
        "sst_pca_explained_variance": [float(v) for v in explained],
        "train_p90": p90, "seeds": {},
    }
    members = {period: {name: [] for name in ("baseline", "eof_cat", "film")}
               for period in ("validation", "holdout")}
    for seed in SEEDS:
        row = {}
        baseline, info = train("baseline32", datasets["train"], datasets["validation"],
                               device, args.epochs, 2e-4, 0.1, seed)
        base_weights = {key: value.detach().cpu().clone()
                        for key, value in baseline.state_dict().items()}
        torch.save(base_weights, exp_dir / "checkpoints" / f"baseline_s{seed}.pt")
        row["baseline"] = {"training": info}
        members["validation"]["baseline"].append(
            predict(baseline, "baseline32", x_t[masks["validation"]],
                    pcs_t[masks["validation"]], device)
        )
        base_val, _ = summary(members["validation"]["baseline"][-1],
                              y[masks["validation"]], valid, p90)
        del baseline

        cat, info = train("eof_cat32", datasets["train"], datasets["validation"],
                          device, args.epochs, 2e-4, 0.1, seed, index_dim=30)
        torch.save(cat.state_dict(), exp_dir / "checkpoints" / f"eof_cat_s{seed}.pt")
        row["eof_cat"] = {"training": info}
        members["validation"]["eof_cat"].append(
            predict(cat, "eof_cat32", x_t[masks["validation"]],
                    pcs_t[masks["validation"]], device)
        )
        del cat

        film, info, val_pred = train_candidate(
            "early_film", (1, 2, 3, 4), 0.0, base_weights,
            datasets["train"], val_arrays, base_val["mean_acc"],
            device, args.epochs, seed,
        )
        torch.save(film.state_dict(), exp_dir / "checkpoints" / f"film_s{seed}.pt")
        row["film"] = {"training": info}
        members["validation"]["film"].append(val_pred)
        del film
        mask = masks["validation"]
        predictions = {name: members["validation"][name][-1]
                       for name in members["validation"]}
        row["validation"] = metric_rows(predictions, y[mask], valid, p90,
                                        dates[mask].year.values)
        np.savez_compressed(exp_dir / f"s{seed}_validation.npz",
                            dates=dates[mask].strftime("%Y-%m-%d").values,
                            target=y[mask], **predictions)
        payload["seeds"][str(seed)] = row
        result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"seed={seed} validation ACC " + " ".join(
            f"{name}={row['validation']['metrics'][name]['mean_acc']:.5f}"
            for name in ("baseline", "eof_cat", "film")
        ), flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("All checkpoints fixed; starting retrospective holdout evaluation", flush=True)
    mask = masks["holdout"]
    for seed in SEEDS:
        models = {
            "baseline": (UNetLite(n_channels=10, base_filters=32, dropout=0.1),
                         "baseline32", f"baseline_s{seed}.pt"),
            "eof_cat": (UNetLite(n_channels=40, base_filters=32, dropout=0.1),
                        "eof_cat32", f"eof_cat_s{seed}.pt"),
            "film": (UNetLiteFiLM(
                n_channels=10, index_dim=30, base_filters=32, index_hidden=16,
                dropout=0.1, film_layers=(1, 2, 3, 4), identity_init=True,
                zero_pc_identity=True,
            ), "film32_identity", f"film_s{seed}.pt"),
        }
        for name, (model, model_name, checkpoint) in models.items():
            model.load_state_dict(torch.load(exp_dir / "checkpoints" / checkpoint,
                                             map_location="cpu", weights_only=True))
            model = model.to(device)
            members["holdout"][name].append(
                predict(model, model_name, x_t[mask], pcs_t[mask], device)
            )
            del model
        predictions = {name: members["holdout"][name][-1]
                       for name in members["holdout"]}
        payload["seeds"][str(seed)]["holdout"] = metric_rows(
            predictions, y[mask], valid, p90, dates[mask].year.values,
        )
        np.savez_compressed(exp_dir / f"s{seed}_holdout.npz",
                            dates=dates[mask].strftime("%Y-%m-%d").values,
                            target=y[mask], **predictions)
        result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"seed={seed} holdout ACC " + " ".join(
            f"{name}={payload['seeds'][str(seed)]['holdout']['metrics'][name]['mean_acc']:.5f}"
            for name in ("baseline", "eof_cat", "film")
        ), flush=True)

    payload["ensemble"] = {}
    for period in members:
        mask = masks[period]
        predictions = {name: np.mean(preds, axis=0)
                       for name, preds in members[period].items()}
        payload["ensemble"][period] = metric_rows(
            predictions, y[mask], valid, p90, dates[mask].year.values,
        )
        np.savez_compressed(exp_dir / f"ensemble_{period}.npz",
                            dates=dates[mask].strftime("%Y-%m-%d").values,
                            target=y[mask], **predictions)
        print(f"ensemble {period} ACC " + " ".join(
            f"{name}={payload['ensemble'][period]['metrics'][name]['mean_acc']:.5f}"
            for name in ("baseline", "eof_cat", "film")
        ), flush=True)
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {result_file}", flush=True)


if __name__ == "__main__":
    main()
