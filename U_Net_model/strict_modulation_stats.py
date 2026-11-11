"""Inspect validation-period FiLM modulation magnitudes for deep and early layers."""

import argparse
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
from unetlitefilm import UNetLiteFiLM
from run_unified_eval import build_date_index, compute_strict_sst_pcs
from strict_sst_window_eval import FOLDS, SEEDS


MODELS = {
    "deep": ("strict_trajectory_v1", (3, 4)),
    "early": ("film_early_v1", (1, 2, 3, 4)),
}


def describe(values):
    flat = values.detach().cpu().numpy().ravel()
    return {"rms": float(np.sqrt(np.mean(flat ** 2))),
            "p99_abs": float(np.percentile(np.abs(flat), 99)),
            "max_abs": float(np.max(np.abs(flat)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="film_amp_v1")
    args = parser.parse_args()
    if not args.experiment or "/" in args.experiment or "\\" in args.experiment:
        parser.error("experiment must be a directory name")
    paths = SubprojectPaths(__file__)
    if (paths.results_dir / args.experiment).exists():
        raise FileExistsError(paths.results_dir / args.experiment)
    exp_dir = Path(paths.get_exp_dir(args.experiment))
    dates = build_date_index()
    sst = np.load(config.modelconfig["sst_file"]).astype(np.float32)
    payload = {"protocol": "validation PCs only; train-fitted EOF per fold",
               "folds": {}}
    for fold, spec in FOLDS.items():
        train_mask = np.asarray(dates <= spec["train_end"])
        val_mask = np.asarray((dates > spec["train_end"]) &
                              (dates <= spec["val_end"]))
        pcs, _, _ = compute_strict_sst_pcs(sst, train_mask, 5, 6, keep_lags=True)
        indices = torch.from_numpy(pcs[val_mask])
        rows = {}
        for seed in SEEDS:
            seed_rows = {}
            for name, (source, layers) in MODELS.items():
                model = UNetLiteFiLM(
                    n_channels=10, index_dim=30, base_filters=32,
                    index_hidden=16, dropout=0.1, film_layers=layers,
                    identity_init=True, zero_pc_identity=True,
                )
                checkpoint = (paths.results_dir / source / "checkpoints" /
                              f"{fold}_s{seed}.pt")
                model.load_state_dict(torch.load(checkpoint, map_location="cpu",
                                                 weights_only=True))
                model.eval()
                with torch.no_grad():
                    gammas, betas = model.index_encoder(indices)
                    zero_gammas, zero_betas = model.index_encoder(
                        torch.zeros_like(indices)
                    )
                seed_rows[name] = {
                    str(level): {
                        "gamma": describe(gammas[level] - zero_gammas[level]),
                        "beta": describe(betas[level] - zero_betas[level]),
                    } for level in layers
                }
            rows[str(seed)] = seed_rows
        payload["folds"][fold] = rows
    result_file = exp_dir / "modulation_stats.json"
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {result_file}", flush=True)


if __name__ == "__main__":
    main()
