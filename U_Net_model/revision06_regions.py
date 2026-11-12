"""Describe regional skill from the saved revision-06 prediction arrays."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.paths import SubprojectPaths
from strict_fair_eval import summary


MODELS = ("seas5", "baseline", "eof_cat", "film")
REGIONS = (
    ("East", "longitude >= 105 E"),
    ("Southwest", "longitude < 105 E and latitude <= 37 N"),
    ("Northwest", "longitude < 105 E and latitude > 37 N"),
)


def analyze(path, p90):
    with np.load(path, allow_pickle=True) as source:
        valid = source["valid"].astype(bool)
        target = source["target"]
        predictions = {name: source[name] for name in MODELS}
    latitude, longitude = np.meshgrid(
        np.arange(60, 0, -0.5), np.arange(70, 140, 0.5), indexing="ij"
    )
    if valid.shape != longitude.shape:
        raise ValueError("Unexpected grid shape")
    region_masks = (
        longitude >= 105,
        (longitude < 105) & (latitude <= 37),
        (longitude < 105) & (latitude > 37),
    )
    if not np.array_equal(np.sum(region_masks, axis=0), np.ones_like(valid)):
        raise ValueError("Regions do not partition the evaluation grid")
    rows = []
    for (name, definition), region_mask in zip(REGIONS, region_masks):
        mask = valid & region_mask
        for model, pred in predictions.items():
            metrics, _ = summary(pred, target, mask, p90)
            rows.append({
                "region": name, "definition": definition,
                "valid_cells": int(mask.sum()), "months": int(len(target)),
                "model": model, "acc": metrics["mean_acc"],
                "tcc": metrics["mean_tcc"], "rmse": metrics["rmse"],
                "sign_pct": metrics["sc_pct"], "csi": metrics["csi"],
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="revision06_eval_v2")
    parser.add_argument("--experiment", default="revision06_regions_v1")
    args = parser.parse_args()
    if any(not name or any(char in name for char in "/\\")
           for name in (args.source, args.experiment)):
        parser.error("source and experiment must be directory names")
    paths = SubprojectPaths(__file__)
    output = paths.results_dir / args.experiment
    if output.exists():
        raise FileExistsError(output)
    source = paths.results_dir / args.source
    metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    p90 = float(metrics["train_p90"])
    rows = []
    for period in ("validation", "test"):
        for row in analyze(source / f"{period}_predictions.npz", p90):
            rows.append({"period": period, **row})
    output = Path(paths.get_exp_dir(args.experiment))
    (output / "regional_metrics.json").write_text(
        json.dumps({"protocol": "Descriptive, post hoc geographic partitions on the common valid grid; no independent test or regional significance inference.",
                    "p90": p90, "rows": rows}, indent=2), encoding="utf-8"
    )
    with (output / "regional_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['period']:10s} {row['region']:9s} {row['model']:8s} "
              f"ACC={row['acc']:.3f} RMSE={row['rmse']:.3f}")


if __name__ == "__main__":
    main()
