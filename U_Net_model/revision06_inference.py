"""Paired temporal-correlation and annual-ACC checks for revision 06.

The 2023-2024 comparison has only two calendar-year blocks. Its exhaustive
block-resampling range is descriptive and must not be read as a confirmatory CI.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.paths import SubprojectPaths
from utils.utils import cal_acc


MODELS = ("seas5", "baseline", "eof_cat", "film")
COMPARATORS = ("seas5", "baseline", "eof_cat")


def temporal_correlation(pred, obs):
    """Pearson correlation along time, averaged over commonly valid grid cells."""
    pred = pred.astype(np.float64, copy=False)
    obs = obs.astype(np.float64, copy=False)
    pred = pred - pred.mean(axis=0)
    obs = obs - obs.mean(axis=0)
    numerator = np.sum(pred * obs, axis=0)
    denominator = np.sqrt(np.sum(pred ** 2, axis=0) * np.sum(obs ** 2, axis=0))
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
                     where=denominator > 1e-12)


def paired_tcc(predictions, obs, indices):
    correlations = {name: temporal_correlation(pred[indices], obs[indices])
                    for name, pred in predictions.items()}
    return {name: float(np.nanmean(correlations["film"] - correlations[name]))
            for name in COMPARATORS}


def analyze(path, n_boot, seed):
    with np.load(path, allow_pickle=True) as source:
        dates = np.asarray(source["dates"], dtype="datetime64[M]")
        valid = source["valid"].astype(bool)
        obs = source["target"][:, valid]
        predictions = {name: source[name][:, valid] for name in MODELS}
    if not np.isfinite(obs).all() or any(not np.isfinite(p).all()
                                          for p in predictions.values()):
        raise ValueError(f"Nonfinite value in valid cells: {path}")
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    unique_years = np.unique(years)
    blocks = [np.flatnonzero(years == year) for year in unique_years]
    full = np.arange(len(dates))
    point = paired_tcc(predictions, obs, full)
    rng = np.random.default_rng(seed)
    if len(blocks) == 2:
        choices = ((0, 0), (0, 1), (1, 1))
    else:
        choices = [tuple(rng.integers(len(blocks), size=len(blocks)))
                   for _ in range(n_boot)]
    boot = {name: [] for name in COMPARATORS}
    for choice in choices:
        sampled = np.concatenate([blocks[index] for index in choice])
        result = paired_tcc(predictions, obs, sampled)
        for name in COMPARATORS:
            boot[name].append(result[name])
    tcc = {}
    for name in COMPARATORS:
        values = np.asarray(boot[name])
        tcc[name] = {
            "film_minus_comparator": point[name],
            "year_block_range" if len(blocks) == 2 else "year_block_boot95":
                [float(np.min(values)), float(np.max(values))] if len(blocks) == 2
                else [float(v) for v in np.percentile(values, [2.5, 97.5])],
            "n_resamples": len(values),
        }
    # Compute spatial ACC on the compressed common-land grid; cal_acc accepts
    # time-by-1-by-cell arrays and uses the paper's Pearson definition.
    acc = {name: np.asarray(cal_acc(obs[:, None], pred[:, None]), dtype=float)
           for name, pred in predictions.items()}
    annual = {}
    for year in unique_years:
        selected = years == year
        annual[str(year)] = {
            "months": int(selected.sum()),
            "mean_acc": {name: float(np.mean(values[selected]))
                         for name, values in acc.items()},
            "film_minus": {name: float(np.mean((acc["film"] - acc[name])[selected]))
                           for name in COMPARATORS},
        }
    return {
        "period": f"{dates[0]}/{dates[-1]}",
        "years": [int(year) for year in unique_years],
        "months": len(dates),
        "valid_cells": int(valid.sum()),
        "tcc_differences": tcc,
        "annual_acc": annual,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="revision06_eval_v2")
    parser.add_argument("--experiment", default="revision06_inference_v1")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()
    if args.n_boot < 1 or not args.experiment or any(c in args.experiment for c in "/\\"):
        parser.error("invalid experiment or bootstrap count")
    paths = SubprojectPaths(__file__)
    output = paths.results_dir / args.experiment
    if output.exists():
        raise FileExistsError(output)
    source = paths.results_dir / args.source
    payload = {
        "protocol": "Paired temporal-correlation differences; resample intact calendar years. Two-year ranges enumerate all distinct unordered block combinations and are descriptive.",
        "validation": analyze(source / "validation_predictions.npz", args.n_boot, 6106),
        "exploratory": analyze(source / "test_predictions.npz", args.n_boot, 6206),
    }
    output = Path(paths.get_exp_dir(args.experiment))
    (output / "inference.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for period in ("validation", "exploratory"):
        print(period, payload[period]["tcc_differences"], flush=True)
    print(f"Saved {output / 'inference.json'}")


if __name__ == "__main__":
    main()
