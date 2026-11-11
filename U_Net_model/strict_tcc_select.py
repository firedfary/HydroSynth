"""Check TCC-selected shallow FiLM on three train-fitted validation folds."""

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
from run_unified_eval import build_date_index, compute_strict_sst_pcs
from strict_centered_eval import train_candidate
from strict_decoder_continue import FOLDS, SEEDS, SOURCE, make_model, source_info
from strict_fair_eval import predict, summary
from strict_sst_window_eval import baseline_weights, fold_data


def reference_predictions(paths, fold, seed, source, x_val, pc_val, device):
    predictions = {}
    for name in ("baseline", "film"):
        model = make_model(paths, fold, seed, name, source, device)
        mode = "baseline32" if name == "baseline" else "film32_identity"
        predictions["old_film" if name == "film" else name] = predict(
            model, mode, x_val, pc_val, device,
        )
        del model
    cat = UNetLite(n_channels=40, base_filters=32, dropout=0.1).to(device)
    cat_source = (source if fold == "early" else
                  paths.results_dir / "strict_cat30_v2")
    filename = (f"eof_cat_s{seed}.pt" if fold == "early" else
                f"{fold}_s{seed}.pt")
    cat.load_state_dict(torch.load(cat_source / "checkpoints" / filename,
                                   map_location=device, weights_only=True))
    predictions["eof_cat"] = predict(cat, "eof_cat32", x_val, pc_val, device)
    del cat
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--experiment", default="film_tcc_select_v1")
    args = parser.parse_args()
    if (args.epochs <= 0 or not args.experiment or "/" in args.experiment or
            "\\" in args.experiment):
        parser.error("invalid epochs or experiment name")
    paths = SubprojectPaths(__file__)
    if (paths.results_dir / args.experiment).exists():
        raise FileExistsError(paths.results_dir / args.experiment)
    exp_dir = Path(paths.get_exp_dir(args.experiment))
    result_file = exp_dir / "tcc_metrics.json"
    dates = build_date_index()
    sst = np.load(config.modelconfig["sst_file"]).astype(np.float32)
    y = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = {
        "protocol": {
            "candidate": "shallow zero-PC-identity FiLM with TCC checkpoint selection",
            "sources": SOURCE, "seeds": list(SEEDS), "epochs": args.epochs,
            "architecture": "frozen 32-channel Baseline, FiLM levels 1-4, 30 SST PCs",
            "training": "masked MSE plus 0.1 FiLM penalty; AdamW 2e-4",
            "checkpoint_selection": "validation mean TCC, including epoch 0",
            "references": "existing ACC-selected Baseline, 30-PC EOF-cat, and shallow FiLM",
            "gate": "candidate ensemble TCC and significant-grid coverage exceed all references on every fold",
            "holdout_status": "not accessed by this experiment",
        }, "folds": {},
    }
    cat_prior = json.loads((paths.results_dir / "strict_cat30_v2" /
                            "cat30_metrics.json").read_text(encoding="utf-8"))
    for fold, spec in FOLDS.items():
        x, digest, train_mask, val_mask, _, valid, p90, _ = fold_data(
            paths, spec, sst, y, dates, windows=(6,),
        )
        pcs, _, _ = compute_strict_sst_pcs(sst, train_mask, 5, 6, keep_lags=True)
        pc_digest = hashlib.sha256(pcs.tobytes()).hexdigest()
        source = source_info(paths, fold, digest, pc_digest)
        if fold != "early":
            cat_row = cat_prior["folds"][fold]
            if (cat_row["input_sha256"] != digest or
                    cat_row["trajectory_sha256"] != pc_digest):
                raise ValueError(f"EOF-cat reference input differs for {fold}")
        else:
            early_prior = json.loads((source / "early_origin_metrics.json")
                                     .read_text(encoding="utf-8"))
        x_t, pcs_t = torch.from_numpy(x), torch.from_numpy(pcs)
        y_t = torch.from_numpy(y[:, None])
        mask_t = torch.isnan(y_t) | torch.from_numpy(~valid)[None, None]
        dataset = TensorDataset(y_t[train_mask], x_t[train_mask],
                                mask_t[train_mask], pcs_t[train_mask])
        val_arrays = (x_t[val_mask], pcs_t[val_mask], y[val_mask], valid, p90)
        members = {name: [] for name in ("baseline", "eof_cat", "old_film", "candidate")}
        row = {"input_sha256": digest, "trajectory_sha256": pc_digest, "seeds": {}}
        for seed in SEEDS:
            references = reference_predictions(
                paths, fold, seed, source, val_arrays[0], val_arrays[1], device,
            )
            ref_metrics = {name: summary(pred, val_arrays[2], valid, p90)[0]
                           for name, pred in references.items()}
            reported = (early_prior["seeds"][str(seed)]["validation"]["metrics"]
                        if fold == "early" else
                        cat_prior["folds"][fold]["seeds"][str(seed)]["validation"])
            labels = {"baseline": "baseline", "eof_cat": "eof_cat" if fold == "early" else "cat30",
                      "old_film": "film"}
            for name, source_name in labels.items():
                if abs(ref_metrics[name]["mean_acc"] - reported[source_name]["mean_acc"]) > 1e-5:
                    raise ValueError(f"{fold} seed={seed} {name} reference differs")
                members[name].append(references[name])
            if fold == "early":
                base_weights = torch.load(source / "checkpoints" /
                                          f"baseline_s{seed}.pt", map_location="cpu",
                                          weights_only=True)
            else:
                base_weights = baseline_weights(paths, fold, seed)
            model, info, candidate_pred = train_candidate(
                "tcc_selected", (1, 2, 3, 4), 0.0, base_weights,
                dataset, val_arrays, ref_metrics["baseline"]["mean_acc"],
                device, args.epochs, seed, selection_metric="mean_tcc",
                include_baseline=True,
            )
            torch.save(model.state_dict(), exp_dir / "checkpoints" /
                       f"{fold}_s{seed}.pt")
            members["candidate"].append(candidate_pred)
            row["seeds"][str(seed)] = {"references": ref_metrics, "candidate": info}
            payload["folds"][fold] = row
            result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        ensemble = {name: np.mean(preds, axis=0) for name, preds in members.items()}
        row["ensemble_validation"] = {
            name: summary(pred, val_arrays[2], valid, p90)[0]
            for name, pred in ensemble.items()
        }
        payload["folds"][fold] = row
        result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"{fold} TCC " + " ".join(
            f"{name}={metric['mean_tcc']:.5f}"
            for name, metric in row["ensemble_validation"].items()
        ), flush=True)
    gate = all(
        all(row["ensemble_validation"]["candidate"][key] > max(
            row["ensemble_validation"][name][key]
            for name in ("baseline", "eof_cat", "old_film")
        ) for key in ("mean_tcc", "sig_pct"))
        for row in payload["folds"].values()
    )
    payload["selection"] = {"passed_three_fold_gate": gate}
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"selection: {payload['selection']}", flush=True)


if __name__ == "__main__":
    main()
