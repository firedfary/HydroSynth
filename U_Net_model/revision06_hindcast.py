"""Matched retrospective 2014-2022 hindcast for revision-06 models."""

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
from strict_fair_eval import make_model, predict, summary, train
from revision06_enso_eval import read_oni, stratify


SEEDS = (42, 43, 44)
MODELS = {"baseline": "baseline32", "eof_cat": "eof_cat32",
          "film": "film32_identity"}
FOLDS = {
    "2014_2019": {
        "cache": "strict_inputs_pre2011", "fit_end": "2010-12-01",
        "validation": ("2011-01-01", "2013-12-01"),
        "holdout": ("2014-01-01", "2019-12-01"),
    },
    "2020_2022": {
        "cache": "strict_pre2016", "fit_end": "2015-12-01",
        "validation": ("2016-01-01", "2019-12-01"),
        "holdout": ("2020-01-01", "2022-12-01"),
    },
}


def arrays(paths, dates, spec, y, sst):
    folder = paths.cache_dir / spec["cache"]
    input_file = folder / "cond_trainfit.npy"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(input_file.read_bytes()).hexdigest()
    if (digest != manifest["sha256"] or
            manifest["train_period"] != f"1994-01/{spec['fit_end'][:7]}" or
            manifest["validation_period"] !=
            f"{spec['validation'][0][:7]}/{spec['validation'][1][:7]}"):
        raise ValueError(f"Input cache does not match fold: {folder}")
    x = np.load(input_file).astype(np.float32)
    if x.shape != (len(dates), 10, 120, 140):
        raise ValueError("Unexpected atmospheric input shape")
    train_mask = np.asarray(dates <= spec["fit_end"])
    val_mask = np.asarray((dates >= spec["validation"][0]) &
                          (dates <= spec["validation"][1]))
    hold_mask = np.asarray((dates >= spec["holdout"][0]) &
                           (dates <= spec["holdout"][1]))
    pcs, _, explained = compute_strict_sst_pcs(sst, train_mask, 5)
    valid = np.isfinite(y).all(axis=0)
    p90 = float(np.percentile(y[train_mask][:, valid], 90))
    return x, pcs, valid, p90, train_mask, val_mask, hold_mask, digest, explained


def predict_fold(paths, output, dates, spec, y, sst, raw, device, train_models):
    (x, pcs, valid, p90, train_mask, val_mask, hold_mask, digest,
     explained) = arrays(paths, dates, spec, y, sst)
    x_t, pcs_t = torch.from_numpy(x), torch.from_numpy(pcs)
    members = {name: [] for name in MODELS}
    training = {}

    if train_models:
        y_t = torch.from_numpy(y[:, None])
        mask_t = torch.isnan(y_t) | torch.from_numpy(~valid)[None, None]
        dataset = TensorDataset(y_t, x_t, mask_t, pcs_t)
        train_ds = torch.utils.data.Subset(dataset, np.flatnonzero(train_mask))
        val_ds = torch.utils.data.Subset(dataset, np.flatnonzero(val_mask))
        checkpoint_dir = output / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    else:
        checkpoint_dir = paths.results_dir / "revision06_rolling_v1" / "checkpoints"

    for seed in SEEDS:
        training[str(seed)] = {}
        for name, model_name in MODELS.items():
            checkpoint = checkpoint_dir / f"{name}_s{seed}.pt"
            if train_models:
                if checkpoint.exists():
                    raise FileExistsError(checkpoint)
                model, info = train(model_name, train_ds, val_ds, device,
                                    100, 2e-4, 0.1, seed)
                torch.save(model.state_dict(), checkpoint)
                training[str(seed)][name] = {
                    "best_epoch": info["best_epoch"],
                    "best_validation_acc": info["best_val_acc"],
                    "parameters": info["parameters"],
                }
            else:
                model = make_model(model_name, 0.1).to(device)
                model.load_state_dict(torch.load(checkpoint, map_location=device,
                                                 weights_only=True))
            members[name].append(predict(model, model_name, x_t[hold_mask],
                                         pcs_t[hold_mask], device))
            del model
            print(f"{spec['holdout'][0][:4]}-{spec['holdout'][1][:4]} "
                  f"seed={seed} model={name} done", flush=True)
    predictions = {name: np.mean(values, axis=0) for name, values in members.items()}
    predictions["seas5"] = raw[hold_mask]
    pred_dates = dates[hold_mask].strftime("%Y-%m-%d").values
    pred_path = output / f"fold_{spec['holdout'][0][:4]}_{spec['holdout'][1][:4]}.npz"
    np.savez_compressed(pred_path, dates=pred_dates, target=y[hold_mask],
                        valid=valid, **predictions)
    metrics, acc = {}, {}
    for name, pred in predictions.items():
        metrics[name], acc[name] = summary(pred, y[hold_mask], valid, p90)
    paired = {f"film_minus_{name}": paired_stat_tests(
        acc["film"], acc[name], dates[hold_mask].year.values)
        for name in ("baseline", "eof_cat", "seas5")}
    return pred_path, {
        "training_window": f"1994-01/{spec['fit_end'][:7]}",
        "validation_window": "/".join(v[:7] for v in spec["validation"]),
        "holdout_window": "/".join(v[:7] for v in spec["holdout"]),
        "months": int(hold_mask.sum()), "input_sha256": digest,
        "sst_pca_explained_variance": [float(v) for v in explained],
        "train_p90": p90, "training": training, "metrics": metrics,
        "paired_acc": paired,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="revision06_hindcast_v1")
    args = parser.parse_args()
    if not args.experiment or any(c in args.experiment for c in "/\\"):
        parser.error("experiment must be one directory name")
    paths = SubprojectPaths(__file__)
    output = paths.results_dir / args.experiment
    if output.exists() and (
            (output / "metrics.json").exists() or
            any((output / "checkpoints").glob("*.pt"))):
        raise FileExistsError(output)
    output = Path(paths.get_exp_dir(args.experiment))
    dates = build_date_index()
    y = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    sst = np.load(config.modelconfig["sst_file"], mmap_mode="r")
    if y.shape != (len(dates), 120, 140) or sst.shape != (len(dates), 6, 89, 180):
        raise ValueError("Target or SST chronology differs from expected input")
    with np.load(paths.results_dir / "revision06_eval_v2" / "seas5_raw.npz",
                 allow_pickle=True) as source:
        if not np.array_equal(source["dates"], dates.strftime("%Y-%m-%d").values):
            raise ValueError("Raw SEAS5 chronology mismatch")
        raw = source["raw_anomaly"].copy()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The matched retrospective training requires a CUDA GPU")
    payload = {
        "protocol": {
            "models": MODELS, "seeds": list(SEEDS), "epochs": 100,
            "optimizer": "AdamW, 0.0002, weight decay 0.00001, cosine schedule",
            "checkpoint_selection": "validation mean spatial ACC",
            "ensemble": "equal-weight mean of three preselected seeds",
            "sst": "five standardized training-fitted EOF scores from preceding six-month mean",
            "seas5": "first target-month archive field; bilinear, nonnegative, 1994-2010 climatology",
            "status": "retrospective; design and these years were previously examined",
        },
        "folds": {},
    }
    files = []
    for key, spec in FOLDS.items():
        file, row = predict_fold(paths, output, dates, spec, y, sst, raw,
                                 device, train_models=key == "2014_2019")
        files.append(file)
        payload["folds"][key] = row
        (output / "metrics.json").write_text(json.dumps(payload, indent=2),
                                               encoding="utf-8")
    arrays_to_merge = []
    for file in files:
        with np.load(file, allow_pickle=True) as data:
            arrays_to_merge.append({key: data[key].copy() for key in
                                    ("dates", "target", *MODELS, "seas5")})
    combined = {key: np.concatenate([row[key] for row in arrays_to_merge])
                for key in arrays_to_merge[0]}
    combined["valid"] = np.isfinite(y).all(axis=0)
    pred_dates = combined["dates"]
    if len(pred_dates) != len(set(pred_dates)) or len(pred_dates) != 107:
        raise ValueError("Combined rolling holdout must have 107 unique months")
    combined_file = output / "hindcast_predictions.npz"
    np.savez_compressed(combined_file, **combined)
    oni = read_oni(ROOT / "ref" / "ONI_v5.html")
    phases = stratify(combined_file, oni, 0.5)
    acc = {name: summary(combined[name], combined["target"],
                         combined["valid"], 0.9843)[1]
           for name in ("seas5", *MODELS)}
    years = np.asarray([int(date[:4]) for date in pred_dates])
    paired = {f"film_minus_{name}": paired_stat_tests(acc["film"], acc[name], years)
              for name in ("baseline", "eof_cat", "seas5")}
    payload["pooled"] = {"months": len(pred_dates), "phases": phases,
                         "paired_acc": paired}
    (output / "metrics.json").write_text(json.dumps(payload, indent=2),
                                           encoding="utf-8")
    print(f"Saved {output / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
