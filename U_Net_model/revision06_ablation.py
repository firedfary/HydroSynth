"""Symmetric ablations and local-predictor importance for revision 06."""

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
from strict_fair_eval import make_model, predict, summary
from utils.utils import cal_acc


SEEDS = (42, 43, 44)
SOURCES = {42: "strict_fair_v1", 43: "strict_fair_s43", 44: "strict_fair_s44"}
MODELS = {"baseline": "baseline32", "eof_cat": "eof_cat32", "film": "film32_identity"}
VARIABLES = ("tp", "h200", "h500", "slp", "t2m", "t850", "u200", "u850", "v200", "v850")


def eval_condition(models, x, pc, device):
    return {name: predict(model, MODELS[name], x, pc, device)
            for name, model in models.items()}


def mean_acc(pred, obs, valid):
    return float(np.nanmean(cal_acc(np.where(valid, obs, np.nan),
                                    np.where(valid, pred, np.nan))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="revision06_ablation")
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    if args.trials < 1 or not args.experiment or any(c in args.experiment for c in "/\\"):
        parser.error("invalid experiment or trial count")
    paths = SubprojectPaths(__file__)
    output = paths.results_dir / args.experiment
    if output.exists():
        raise FileExistsError(output)
    dates = build_date_index()
    mask = np.asarray(dates >= "2023-01-01")
    train = np.asarray(dates <= "2017-12-01")
    pool = np.asarray((dates >= "2018-01-01") & (dates <= "2022-12-01"))
    target = np.load(Path(config.modelconfig["hr_path"]) / "hr_data.npy").astype(np.float32)
    valid = np.isfinite(target).all(axis=0)
    p90 = float(np.percentile(target[train][:, valid], 90))
    x_file = paths.cache_dir / "strict_inputs" / "cond_trainfit.npy"
    manifest = json.loads((x_file.parent / "manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(x_file.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError("strict input hash mismatch")
    x_all = np.load(x_file, mmap_mode="r")
    x = torch.from_numpy(x_all[mask].copy())
    sst = np.load(config.modelconfig["sst_file"], mmap_mode="r")
    pc_all, _, _ = compute_strict_sst_pcs(sst, train, 5)
    pc = torch.from_numpy(pc_all[mask].copy())
    y = target[mask]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(6026)
    output = Path(paths.get_exp_dir(args.experiment))
    members = {case: {name: [] for name in MODELS}
               for case in ("full", "pc_zero", "local_zero")}
    shuffled = {name: [] for name in ("eof_cat", "film")}
    importance = {name: [] for name in VARIABLES}
    row_seeds = {}

    for seed in SEEDS:
        source = paths.results_dir / SOURCES[seed]
        info = json.loads((source / "strict_fair_metrics.json").read_text(encoding="utf-8"))
        if info["input_sha256"] != digest:
            raise ValueError(f"source input differs: {source}")
        models = {}
        for name, model_name in MODELS.items():
            model = make_model(model_name, 0.1).to(device)
            model.load_state_dict(torch.load(source / "checkpoints" / f"{model_name}.pt",
                                             map_location=device, weights_only=True))
            model.eval()
            models[name] = model
        seed_rows = {}
        for case, changed_x, changed_pc in (
            ("full", x, pc),
            ("pc_zero", x, torch.zeros_like(pc)),
            ("local_zero", torch.zeros_like(x), pc),
        ):
            preds = eval_condition(models, changed_x, changed_pc, device)
            for name, values in preds.items():
                members[case][name].append(values)
            seed_rows[case] = {name: summary(pred, y, valid, p90)[0]
                               for name, pred in preds.items()}
        with np.load(source / "test_predictions.npz", allow_pickle=True) as previous:
            for name, model_name in MODELS.items():
                if not np.allclose(previous[model_name], members["full"][name][-1], atol=1e-5):
                    raise ValueError(f"checkpoint prediction mismatch: {source} {name}")

        shuffle_preds = {name: [] for name in shuffled}
        importance_pred = {name: [] for name in VARIABLES}
        for trial in range(args.trials):
            source_idx = np.array([
                rng.choice(np.flatnonzero(pool & (np.asarray(dates.month) == month)))
                for month in np.asarray(dates.month)[mask]
            ])
            shuffled_pc = torch.from_numpy(pc_all[source_idx].copy())
            for name in shuffled:
                shuffle_preds[name].append(
                    predict(models[name], MODELS[name], x, shuffled_pc, device))
            for channel, name in enumerate(VARIABLES):
                changed = x.clone()
                changed[:, channel] = torch.from_numpy(x_all[source_idx, channel].copy())
                importance_pred[name].append(
                    predict(models["film"], MODELS["film"], changed, pc, device))
            if (trial + 1) % 5 == 0:
                print(f"seed={seed} perturbation={trial + 1}/{args.trials}", flush=True)
        for name in shuffled:
            shuffled[name].append(np.stack(shuffle_preds[name]))
        for name in VARIABLES:
            importance[name].append(np.stack(importance_pred[name]))
        row_seeds[str(seed)] = seed_rows
        del models

    ensemble = {}
    for case in members:
        ensemble[case] = {
            name: summary(np.mean(values, axis=0), y, valid, p90)[0]
            for name, values in members[case].items()
        }
    shuffle_rows = {}
    for name, member_trials in shuffled.items():
        trial_preds = np.mean(np.stack(member_trials), axis=0)
        scores = [mean_acc(pred, y, valid) for pred in trial_preds]
        shuffle_rows[name] = {
            "mean_acc": float(np.mean(scores)),
            "sd_acc": float(np.std(scores, ddof=1)) if args.trials > 1 else None,
        }
    importance_rows = []
    full_acc = ensemble["full"]["film"]["mean_acc"]
    for name, member_trials in importance.items():
        trial_preds = np.mean(np.stack(member_trials), axis=0)
        drops = [full_acc - mean_acc(pred, y, valid) for pred in trial_preds]
        importance_rows.append({"variable": name, "mean_acc_drop": float(np.mean(drops)),
                                "sd_acc_drop": float(np.std(drops, ddof=1)) if args.trials > 1 else None})
    importance_rows.sort(key=lambda row: row["mean_acc_drop"], reverse=True)
    payload = {
        "protocol": {"seeds": list(SEEDS), "trials": args.trials,
                     "pc_shuffle": "same calendar month drawn from 2018-2022",
                     "local_permutation": "replace one full predictor map with a same-calendar-month 2018-2022 map",
                     "period": "2023-01/2024-09; exploratory after repeated inspection",
                     "pc_zero": "all standardized PCs set to zero",
                     "local_zero": "all ten normalized atmospheric channels set to zero"},
        "input_sha256": digest, "seed_metrics": row_seeds, "ensemble": ensemble,
        "pc_shuffle": shuffle_rows, "film_predictor_importance": importance_rows,
    }
    (output / "ablation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {output / 'ablation.json'}", flush=True)


if __name__ == "__main__":
    main()
