"""Describe revision-06 skill by NOAA CPC ONI phase without test-set tuning."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.paths import SubprojectPaths
from strict_fair_eval import summary
from utils.utils import cal_acc


def read_oni(path):
    tables = pd.read_html(path, match="DJF")
    table = next(t for t in tables if str(t.iloc[0, 0]).strip() == "Year")
    columns = list(table.iloc[0])
    if columns[:2] != ["Year", "DJF"] or len(columns) != 13:
        raise ValueError("Unexpected NOAA ONI table layout")
    values = {}
    for _, row in table.iloc[1:].iterrows():
        if str(row.iloc[0]).strip() == "Year":
            continue
        year = int(row.iloc[0])
        for month in range(1, 13):
            value = pd.to_numeric(row.iloc[month], errors="coerce")
            if np.isfinite(value):
                values[(year, month)] = float(value)
    return values


def stratify(prediction_file, oni, threshold):
    with np.load(prediction_file, allow_pickle=True) as data:
        dates = pd.to_datetime(data["dates"])
        obs = data["target"]
        valid = data["valid"]
        predictions = {name: data[name] for name in
                       ("seas5", "baseline", "eof_cat", "film")}
    indices = np.array([oni[(date.year, date.month)] for date in dates])
    labels = np.where(indices >= threshold, "warm",
                      np.where(indices <= -threshold, "cold", "neutral"))
    rows = {}
    for label in ("warm", "cold", "neutral", "all"):
        mask = labels == label if label != "all" else np.ones(len(dates), bool)
        count = int(mask.sum())
        values = {}
        for name, pred in predictions.items():
            if count == 0:
                values[name] = {"mean_acc": None, "mean_tcc": None}
                continue
            acc = float(np.nanmean(cal_acc(
                np.where(valid, obs[mask], np.nan),
                np.where(valid, pred[mask], np.nan))))
            tcc = summary(pred[mask], obs[mask], valid, 0.9843)[0]["mean_tcc"] if count >= 5 else None
            values[name] = {"mean_acc": acc, "mean_tcc": tcc}
        rows[label] = {"months": count, "metrics": values}
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oni", type=Path, required=True)
    parser.add_argument("--experiment", default="revision06_eval_v2")
    parser.add_argument("--rolling", default="revision06_rolling_v1")
    args = parser.parse_args()
    paths = SubprojectPaths(__file__)
    source = paths.results_dir / args.experiment
    oni = read_oni(args.oni)
    periods = {name: stratify(source / f"{name}_predictions.npz", oni, 0.5)
               for name in ("validation", "test")}
    rolling = paths.results_dir / args.rolling / "holdout_predictions.npz"
    if rolling.is_file():
        periods["rolling_2020_2021"] = stratify(rolling, oni, 0.5)
    payload = {
        "source": "NOAA CPC historical ONI v5 table",
        "definition": "Each calendar month uses its centered 3-month ONI; warm >= +0.5, cold <= -0.5, otherwise neutral. These are threshold categories, not episode labels.",
        "caution": "TCC omitted where fewer than five months; all phases and periods are descriptive, and the test set was explored during development.",
        "periods": periods,
    }
    output = source / "enso_metrics.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({period: {phase: row["months"] for phase, row in phases.items()}
                      for period, phases in periods.items()}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
