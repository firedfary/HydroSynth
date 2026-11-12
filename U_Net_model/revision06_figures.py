"""Build comparison figures from the verified revision-06 evaluation outputs."""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.paths import SubprojectPaths
from run_unified_eval import compute_tcc_and_p
from utils.utils import cal_acc


ORDER = ("seas5", "baseline", "eof_cat", "film")
LABEL = {"seas5": "SEAS5", "baseline": "U-Net", "eof_cat": "EOF-cat", "film": "FiLM-U-Net"}
COLOR = {"seas5": "#525C66", "baseline": "#126A8A", "eof_cat": "#D1872B", "film": "#B0425E"}


def save(fig, name, figures, delivery):
    for suffix in ("png", "svg"):
        path = figures / f"{name}.{suffix}"
        fig.savefig(path, dpi=360, bbox_inches="tight", facecolor="white")
        shutil.copy2(path, delivery / path.name)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="revision06_eval_v2")
    parser.add_argument("--sensitivity", default="revision06_bicubic_v1")
    parser.add_argument("--delivery", required=True)
    args = parser.parse_args()
    paths = SubprojectPaths(__file__)
    source = paths.results_dir / args.experiment
    sensitivity = paths.results_dir / args.sensitivity
    metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    alt = json.loads((sensitivity / "metrics.json").read_text(encoding="utf-8"))
    if not ("bilinear" in metrics["protocol"]["seas5"] and
            "bicubic" in alt["protocol"]["seas5"]):
        raise ValueError("main/sensitivity source order is incorrect")
    delivery = Path(args.delivery)
    figures = source / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    delivery.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})

    rows = []
    for period in ("validation", "test"):
        for model in ORDER:
            row = {"period": period, "model": LABEL[model]}
            row.update(metrics["periods"][period]["ensemble"][model])
            rows.append(row)
    with (delivery.parent / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with np.load(source / "test_predictions.npz", allow_pickle=True) as data:
        test = {key: data[key].copy() for key in data.files}
    dates = np.asarray(test["dates"], dtype="datetime64[M]")
    obs, valid = test["target"], test["valid"].astype(bool)
    series = {name: np.asarray(cal_acc(np.where(valid, obs, np.nan),
                                       np.where(valid, test[name], np.nan)), dtype=float)
              for name in ORDER}
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 7.2), sharex=True,
                             gridspec_kw={"height_ratios": (2, 1)})
    x = np.arange(len(dates))
    for name in ORDER:
        axes[0].plot(x, series[name], marker="o", ms=3.5, lw=1.7,
                     color=COLOR[name], label=LABEL[name])
    axes[0].axhline(0, color="#999999", lw=0.8)
    axes[0].set_ylabel("Monthly spatial ACC")
    axes[0].legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    axes[0].set_title("(a) Monthly spatial anomaly correlation", loc="left", fontsize=11)
    for name in ("baseline", "eof_cat", "film"):
        axes[1].plot(x, series[name] - series["seas5"], marker="o", ms=3,
                     lw=1.4, color=COLOR[name], label=LABEL[name])
    axes[1].axhline(0, color="#555555", lw=0.9)
    axes[1].set_ylabel("ACC difference vs SEAS5")
    axes[1].set_title("(b) Paired monthly differences", loc="left", fontsize=11)
    axes[1].set_xticks(x[::2], [str(v) for v in dates[::2]], rotation=45, ha="right")
    fig.tight_layout()
    save(fig, "Fig2_monthly_ACC", figures, delivery)

    keys = ("mean_acc", "mean_tcc", "rmse", "sc_pct", "csi")
    names = ("Spatial ACC", "Mean TCC", "RMSE", "Sign consistency (%)", "Extreme CSI")
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.3), constrained_layout=True)
    for ax, key, title in zip(axes.flat, keys, names):
        values = [metrics["periods"]["test"]["ensemble"][name][key] for name in ORDER]
        ax.bar(np.arange(4), values, color=[COLOR[name] for name in ORDER], width=0.72)
        ax.set_xticks(np.arange(4), [LABEL[name] for name in ORDER], rotation=55, ha="right")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.2)
        for i, value in enumerate(values):
            ax.text(i, value * 0.94,
                    f"{value:.3f}" if key != "sc_pct" else f"{value:.1f}",
                    ha="center", va="top", fontsize=9, color="white")
        span = max(values) - min(values)
        ax.set_ylim(0, max(values) + max(span * 0.35, 0.05))
    axes.flat[5].axis("off")
    fig.suptitle("Evaluation period: January 2023-September 2024 (21 months)")
    save(fig, "Fig3_metric_tradeoff", figures, delivery)

    tcc = {}
    pval = {}
    for name in ORDER:
        tcc[name], pval[name] = compute_tcc_and_p(test[name], obs, valid)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, name in zip(axes.flat, ORDER):
        im = ax.imshow(np.where(valid, tcc[name], np.nan), cmap="RdBu_r",
                       vmin=-0.8, vmax=0.8, origin="upper", extent=(70, 140, 0, 60))
        ax.set_title(f"{LABEL[name]} (mean {np.nanmean(tcc[name]):.3f})")
        ax.set_xlim(70, 140)
        ax.set_ylim(10, 60)
        ax.set_xlabel("Longitude (degrees E)")
        ax.set_ylabel("Latitude (degrees N)")
        points = np.where((pval[name][::5, ::5] < 0.05) & valid[::5, ::5])
        ax.scatter(70 + points[1] * 2.5, 60 - points[0] * 2.5,
                   s=1.3, c="black", alpha=0.5, rasterized=True)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.72,
                 location="bottom", label="Temporal correlation")
    fig.suptitle("Gridwise temporal correlation (21 months; descriptive)")
    save(fig, "Fig4_temporal_corr", figures, delivery)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, right in zip(axes, ("baseline", "eof_cat")):
        left = "film"
        difference = tcc[left] - tcc[right]
        im = ax.imshow(np.where(valid, difference, np.nan), cmap="RdBu_r",
                       vmin=-0.5, vmax=0.5, origin="upper", extent=(70, 140, 0, 60))
        ax.set_title(f"FiLM - {LABEL[right]} (mean {np.nanmean(difference):+.3f})")
        ax.set_xlim(70, 140)
        ax.set_ylim(10, 60)
        ax.set_xlabel("Longitude (degrees E)")
        ax.set_ylabel("Latitude (degrees N)")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7,
                 location="bottom", label="TCC difference")
    save(fig, "FigS2_TCC_difference", figures, delivery)

    representative = ("2023-01", "2023-04", "2023-07", "2023-10")
    fields = (("Observation", obs),) + tuple((LABEL[name], test[name]) for name in ORDER)
    fig, axes = plt.subplots(4, 5, figsize=(13.5, 10), sharex=True, sharey=True,
                             constrained_layout=True)
    for row, month in enumerate(representative):
        index = int(np.flatnonzero(dates == np.datetime64(month, "M"))[0])
        for col, (label, values) in enumerate(fields):
            ax = axes[row, col]
            image = ax.imshow(np.where(valid, values[index], np.nan),
                              cmap="BrBG", vmin=-1.5, vmax=1.5,
                              extent=(70, 140, 0, 60))
            ax.set_xlim(70, 140)
            ax.set_ylim(10, 60)
            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(month + "\nLatitude (N)")
            if row == 3:
                ax.set_xlabel("Longitude (E)")
    fig.colorbar(image, ax=axes.ravel().tolist(), location="bottom", shrink=0.75,
                 label="Fractional precipitation anomaly (display clipped to [-1.5, 1.5])")
    save(fig, "Fig5_spatial_examples", figures, delivery)

    ablation_file = paths.results_dir / "revision06_ablation_v1" / "ablation.json"
    if ablation_file.exists():
        ablation = json.loads(ablation_file.read_text(encoding="utf-8"))
        values = {
            name: [ablation["ensemble"][case][name]["mean_acc"]
                   for case in ("full", "pc_zero", "local_zero")]
            for name in ("eof_cat", "film")
        }
        values["eof_cat"].append(ablation["pc_shuffle"]["eof_cat"]["mean_acc"])
        values["film"].append(ablation["pc_shuffle"]["film"]["mean_acc"])
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        xloc = np.arange(4)
        width = 0.35
        for offset, name in ((-width / 2, "eof_cat"), (width / 2, "film")):
            ax.bar(xloc + offset, values[name], width, label=LABEL[name], color=COLOR[name])
        ax.set_xticks(xloc, ("Full input", "SST PCs zeroed", "Local fields zeroed",
                              "SST PCs replaced"))
        ax.set_ylabel("Test spatial ACC")
        ax.set_ylim(-0.06, 0.39)
        ax.axhline(0, color="#777777", lw=0.8)
        ax.legend(frameon=False)
        ax.set_title("Symmetric input ablations (three-seed ensembles)")
        fig.tight_layout()
        save(fig, "Fig6_symmetric_ablation", figures, delivery)

        ranking = ablation["film_predictor_importance"]
        fig, ax = plt.subplots(figsize=(7.5, 5))
        labels = [row["variable"] for row in ranking][::-1]
        scores = [row["mean_acc_drop"] for row in ranking][::-1]
        ax.barh(labels, scores, color=[COLOR["baseline"] if value > 0 else "#A4ADB4"
                                       for value in scores])
        ax.axvline(0, color="#777777", lw=0.8)
        ax.set_xlabel("Mean ACC decrease after same-month field replacement")
        ax.set_title("FiLM-U-Net atmospheric predictor importance")
        fig.tight_layout()
        save(fig, "Fig7_predictor_importance", figures, delivery)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    main = metrics["periods"]["test"]["ensemble"]["seas5"]["mean_acc"]
    bicubic = alt["periods"]["test"]["ensemble"]["seas5"]["mean_acc"]
    old = 0.2807
    values = (main, bicubic, old)
    labels = ("Bilinear + nonnegative\n(main)",
              "Bicubic + nonnegative\n(sensitivity)",
              "Old standardized input\n(invalid raw baseline)")
    ax.bar(np.arange(3), values, color=(COLOR["seas5"], "#A0A9B1", "#D6D9DC"))
    ax.set_xticks(np.arange(3), labels)
    ax.set_ylabel("Test spatial ACC")
    ax.set_ylim(0, 0.38)
    for i, value in enumerate(values):
        ax.text(i, value + 0.01, f"{value:.3f}", ha="center")
    ax.set_title("Raw-SEAS5 construction changes the comparison")
    fig.tight_layout()
    save(fig, "FigS1_SEAS5_sensitivity", figures, delivery)

    summary = {
        "main_source": str(source / "metrics.json"),
        "sensitivity_source": str(sensitivity / "metrics.json"),
        "metrics": metrics["periods"],
        "seas5_bicubic_test_acc": bicubic,
        "old_standardized_input_acc": old,
        "caution": "All 2023-2024 comparisons are exploratory after repeated inspection."
    }
    (delivery.parent / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved figures and summary to {delivery.parent}")


if __name__ == "__main__":
    main()
