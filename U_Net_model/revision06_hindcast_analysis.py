"""Analyze the completed two-origin retrospective hindcast without retraining."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.paths import SubprojectPaths
from utils.utils import cal_acc
from revision06_enso_eval import read_oni
from revision06_inference import analyze
from run_unified_eval import paired_stat_tests


MODELS = ("seas5", "baseline", "eof_cat", "film")


def main():
    folder = SubprojectPaths(__file__).results_dir / "revision06_hindcast_v1"
    source = folder / "hindcast_predictions.npz"
    with np.load(source, allow_pickle=True) as data:
        dates = pd.to_datetime(data["dates"])
        valid = data["valid"].astype(bool)
        target = data["target"]
        predictions = {name: data[name] for name in MODELS}
    oni = read_oni(ROOT / "ref" / "ONI_v5.html")
    values = np.asarray([oni[(date.year, date.month)] for date in dates])
    phase = np.where(values >= 0.5, "warm",
                     np.where(values <= -0.5, "cold", "neutral"))
    obs = np.where(valid, target, np.nan)
    acc = {name: cal_acc(obs, np.where(valid, pred, np.nan))
           for name, pred in predictions.items()}
    contrasts = {}
    for label in ("warm", "cold", "neutral", "all"):
        mask = phase == label if label != "all" else np.ones(len(dates), bool)
        contrasts[label] = {
            "months": int(mask.sum()),
            "years": sorted(set(int(y) for y in dates.year[mask])),
            "paired_acc": {
                f"film_minus_{name}": paired_stat_tests(
                    acc["film"][mask], acc[name][mask], dates.year[mask].values,
                    n_boot=10000, seed=7031)
                for name in ("baseline", "eof_cat", "seas5")
            },
        }
    payload = {
        "status": "retrospective and previously examined; intervals describe stability, not confirmatory testing",
        "protocol": "Paired monthly spatial ACC by ONI threshold category; calendar-year blocks for ACC and TCC intervals.",
        "phase_contrasts": contrasts,
        "pooled_tcc_and_annual_acc": analyze(source, n_boot=1000, seed=7032),
    }
    output = folder / "paired_analysis.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
