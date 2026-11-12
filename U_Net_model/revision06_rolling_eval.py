"""Retrospective 2020-2021 rolling-origin check for matched model variants."""

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
from strict_fair_eval import predict, summary, train


SEEDS = (42, 43, 44)
MODELS = {"baseline": "baseline32", "eof_cat": "eof_cat32",
          "film": "film32_identity"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--experiment", default="revision06_rolling")
    args = parser.parse_args()
    if args.epochs < 1 or not args.experiment or any(c in args.experiment for c in "/\\"):
        parser.error("invalid experiment or epoch count")
    paths = SubprojectPaths(__file__)
    output = paths.results_dir / args.experiment
    if output.exists():
        raise FileExistsError(output)
    dates = build_date_index()
    masks = {
        "train": np.asarray(dates <= "2015-12-01"),
        "validation": np.asarray((dates >= "2016-01-01") & (dates <= "2019-12-01")),
        "holdout": np.asarray((dates >= "2020-01-01") & (dates <= "2021-12-01")),
    }
    input_file = paths.cache_dir / "strict_pre2016" / "cond_trainfit.npy"
    manifest = json.loads((input_file.parent / "manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(input_file.read_bytes()).hexdigest()
    if (digest != manifest["sha256"] or
            manifest["train_period"] != "1994-01/2015-12" or
            manifest["validation_period"] != "2016-01/2019-12"):
        raise ValueError("rolling fold input mismatch")
    x = np.load(input_file).astype(np.float32)
    y = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    sst = np.load(config.modelconfig["sst_file"], mmap_mode="r")
    pcs, _, explained = compute_strict_sst_pcs(sst, masks["train"], 5)
    valid = np.isfinite(y).all(axis=0)
    p90 = float(np.percentile(y[masks["train"]][:, valid], 90))
    with np.load(paths.results_dir / "revision06_eval_v2" / "seas5_raw.npz",
                 allow_pickle=True) as raw_file:
        if not np.array_equal(raw_file["dates"], dates.strftime("%Y-%m-%d").values):
            raise ValueError("SEAS5 date mismatch")
        seas5 = raw_file["raw_anomaly"][masks["holdout"]]
    x_t = torch.from_numpy(x)
    pc_t = torch.from_numpy(pcs)
    y_t = torch.from_numpy(y[:, None])
    nan_t = torch.isnan(y_t) | torch.from_numpy(~valid)[None, None]
    dataset = TensorDataset(y_t, x_t, nan_t, pc_t)
    train_ds = torch.utils.data.Subset(dataset, np.flatnonzero(masks["train"]))
    val_ds = torch.utils.data.Subset(dataset, np.flatnonzero(masks["validation"]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output = Path(paths.get_exp_dir(args.experiment))
    payload = {
        "protocol": {
            "train": "1994-01/2015-12", "validation": "2016-01/2019-12",
            "holdout": "2020-01/2021-12", "seeds": list(SEEDS),
            "models": MODELS, "epochs": args.epochs,
            "checkpoint_selection": "mean spatial ACC on validation",
            "evidence_status": "retrospective: this period has been inspected during development",
        },
        "input_sha256": digest,
        "sst_explained_variance": [float(v) for v in explained],
        "seeds": {},
    }
    members = {name: [] for name in MODELS}
    for seed in SEEDS:
        row = {}
        for name, model_name in MODELS.items():
            model, info = train(model_name, train_ds, val_ds, device,
                                args.epochs, 2e-4, 0.1, seed)
            torch.save(model.state_dict(), output / "checkpoints" / f"{name}_s{seed}.pt")
            val_pred = predict(model, model_name, x_t[masks["validation"]],
                               pc_t[masks["validation"]], device)
            row[name] = {"training": info,
                         "validation": summary(val_pred, y[masks["validation"]], valid, p90)[0]}
            del model
        payload["seeds"][str(seed)] = row
        (output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"selected seed {seed}", flush=True)
    print("All checkpoints fixed; evaluating rolling holdout", flush=True)
    from strict_fair_eval import make_model
    for seed in SEEDS:
        for name, model_name in MODELS.items():
            model = make_model(model_name, 0.1).to(device)
            model.load_state_dict(torch.load(output / "checkpoints" / f"{name}_s{seed}.pt",
                                             map_location=device, weights_only=True))
            pred = predict(model, model_name, x_t[masks["holdout"]],
                           pc_t[masks["holdout"]], device)
            members[name].append(pred)
            payload["seeds"][str(seed)][name]["holdout"] = summary(
                pred, y[masks["holdout"]], valid, p90)[0]
            del model
    predictions = {name: np.mean(values, axis=0) for name, values in members.items()}
    predictions["seas5"] = seas5
    acc, metrics = {}, {}
    for name, pred in predictions.items():
        metrics[name], acc[name] = summary(pred, y[masks["holdout"]], valid, p90)
    paired = {f"{name}_minus_{other}": paired_stat_tests(
        acc[name], acc[other], dates[masks["holdout"]].year.values)
        for name in MODELS for other in ("seas5", "baseline", "eof_cat") if name != other}
    payload["ensemble"] = {"holdout": metrics, "paired_acc": paired}
    np.savez_compressed(output / "holdout_predictions.npz",
                        dates=dates[masks["holdout"]].strftime("%Y-%m-%d").values,
                        target=y[masks["holdout"]], valid=valid, **predictions)
    (output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print({name: row["mean_acc"] for name, row in metrics.items()}, flush=True)
    print(f"Saved {output / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
