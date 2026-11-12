"""Plot annual ACC from the two-origin revision-06 retrospective hindcast."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from revision06_figures import COLOR, LABEL, ORDER, save
from utils.paths import SubprojectPaths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delivery", type=Path, required=True)
    args = parser.parse_args()
    source = SubprojectPaths(__file__).results_dir / "revision06_hindcast_v1"
    payload = json.loads((source / "paired_analysis.json").read_text(encoding="utf-8"))
    annual = payload["pooled_tcc_and_annual_acc"]["annual_acc"]
    years = np.array(sorted(int(year) for year in annual))
    if len(years) != 9 or years[0] != 2014 or years[-1] != 2022:
        raise ValueError("Expected nine annual retrospective summaries")
    figures = source / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    args.delivery.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 6.6), sharex=True,
                             gridspec_kw={"height_ratios": (2, 1)},
                             constrained_layout=True)
    for name in ORDER:
        values = [annual[str(year)]["mean_acc"][name] for year in years]
        for part in (slice(None, 6), slice(6, None)):
            axes[0].plot(years[part], np.asarray(values)[part], marker="o", ms=5,
                         lw=1.7, color=COLOR[name],
                         label=LABEL[name] if part.start is None else None)
    axes[0].set_ylabel("Mean monthly spatial ACC")
    axes[0].set_ylim(0.1, 0.45)
    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.19),
                   frameon=False)
    axes[0].set_title("(a) Annual retrospective holdout skill", loc="left")
    for name in ("seas5", "baseline", "eof_cat"):
        difference = [annual[str(year)]["film_minus"][name] for year in years]
        for part in (slice(None, 6), slice(6, None)):
            axes[1].plot(years[part], np.asarray(difference)[part], marker="o",
                         ms=5, lw=1.5, color=COLOR[name],
                         label=f"FiLM - {LABEL[name]}" if part.start is None else None)
    axes[1].axhline(0, color="#555555", lw=0.9)
    axes[1].set_ylabel("FiLM ACC difference")
    axes[1].set_title("(b) Paired annual differences", loc="left")
    axes[1].set_xticks(years)
    axes[1].set_xlabel("Holdout year")
    axes[1].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.27),
                   frameon=False)
    for ax in axes:
        ax.axvline(2019.5, color="#777777", lw=1, ls="--")
        ax.grid(axis="y", alpha=0.18)
    axes[0].text(2016.5, 0.115, "Origin 1", ha="center", color="#555555")
    axes[0].text(2021, 0.115, "Origin 2", ha="center", color="#555555")
    save(fig, "Fig8_annual_hindcast", figures, args.delivery)


if __name__ == "__main__":
    main()
