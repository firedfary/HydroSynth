"""Build comparison figures from the verified revision-06 evaluation outputs."""

import argparse
import csv
import json
import shutil
import sys
from functools import lru_cache
from pathlib import Path

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import shapefile
from shapely.geometry import LineString, shape

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.paths import SubprojectPaths
from run_unified_eval import compute_tcc_and_p
from utils import maskout
from utils.utils import cal_acc


ORDER = ("seas5", "baseline", "eof_cat", "film")
LABEL = {"seas5": "SEAS5", "baseline": "U-Net", "eof_cat": "EOF-cat", "film": "FiLM-U-Net"}
COLOR = {"seas5": "#525C66", "baseline": "#126A8A", "eof_cat": "#D1872B", "film": "#B0425E"}
MAP_CRS = ccrs.PlateCarree()
MAP_DIR = ROOT / "utils"


@lru_cache(maxsize=1)
def china_features():
    with shapefile.Reader(str(MAP_DIR / "china0.shp")) as reader:
        outline = shape(reader.shape(0).__geo_interface__)
    borders = []
    for block in (MAP_DIR / "CN-border-La.gmt").read_text(encoding="utf-8").split(">"):
        values = np.fromstring(block, sep=" ")
        if values.size >= 4 and values.size % 2 == 0:
            line = LineString(values.reshape(-1, 2)).simplify(0.06)
            borders.append(np.asarray(line.coords))
    return outline, borders


def china_panel(ax, field, valid, *, vmin, vmax, cmap, stipple=None,
                show_x=True, show_y=True):
    lon = np.linspace(70, 140, field.shape[1])
    lat = np.linspace(60, 0, field.shape[0])
    outline, borders = china_features()
    ax.set_extent((70, 140, 10, 60), crs=MAP_CRS)
    ax.set_facecolor("white")
    image = ax.pcolormesh(lon, lat, np.ma.masked_invalid(np.where(valid, field, np.nan)),
                          transform=MAP_CRS, cmap=cmap, vmin=vmin, vmax=vmax,
                          shading="nearest", rasterized=True, zorder=1)
    clip = maskout.shp2clip(image, ax, str(MAP_DIR / "china0.shp"))
    ax.add_geometries([outline], MAP_CRS, facecolor="none", edgecolor="#24282B",
                      linewidth=0.5, zorder=3)
    ax.add_collection(LineCollection(borders, colors="#41484C", linewidths=0.24,
                                     transform=MAP_CRS, zorder=4))
    if stipple is not None:
        yy, xx = np.where(stipple[::5, ::5] & valid[::5, ::5])
        dots = ax.scatter(lon[::5][xx], lat[::5][yy], s=1.5, c="black", alpha=0.7,
                          transform=MAP_CRS, rasterized=True, zorder=5)
        dots.set_clip_path(clip)
    ax.set_xticks((80, 100, 120, 140), crs=MAP_CRS)
    ax.set_yticks((20, 40, 60), crs=MAP_CRS)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=7, length=2, labelbottom=show_x, labelleft=show_y)

    inset = ax.inset_axes((0.765, 0.02, 0.225, 0.29), projection=MAP_CRS)
    inset.set_extent((105, 125, 0, 25), crs=MAP_CRS)
    inset.set_facecolor("white")
    inset.add_geometries([outline], MAP_CRS, facecolor="#F4F5F3",
                         edgecolor="#41484C", linewidth=0.35)
    inset.add_collection(LineCollection(borders, colors="#41484C", linewidths=0.28,
                                        transform=MAP_CRS))
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor("#80888C")
        spine.set_linewidth(0.45)
    return image


def spatial_figures(test, dates, obs, valid, figures, delivery):
    tcc = {}
    pval = {}
    for name in ORDER:
        tcc[name], pval[name] = compute_tcc_and_p(test[name], obs, valid)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8),
                             subplot_kw={"projection": MAP_CRS}, constrained_layout=True)
    for i, (ax, name) in enumerate(zip(axes.flat, ORDER)):
        im = china_panel(ax, tcc[name], valid, vmin=-0.8, vmax=0.8,
                         cmap="RdBu_r", stipple=pval[name] < 0.05,
                         show_x=i >= 2, show_y=i % 2 == 0)
        ax.set_title(f"{LABEL[name]} (mean {np.nanmean(tcc[name]):.3f})", fontsize=10)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75,
                 location="bottom", label="Temporal correlation")
    fig.suptitle("Gridwise temporal correlation (21 months; descriptive)")
    save(fig, "Fig3_temporal_corr", figures, delivery)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7),
                             subplot_kw={"projection": MAP_CRS}, constrained_layout=True)
    for i, (ax, right) in enumerate(zip(axes, ("baseline", "eof_cat"))):
        difference = tcc["film"] - tcc[right]
        im = china_panel(ax, difference, valid, vmin=-0.5, vmax=0.5,
                         cmap="RdBu_r", show_y=i == 0)
        ax.set_title(f"FiLM - {LABEL[right]} (mean {np.nanmean(difference):+.3f})",
                     fontsize=10)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75,
                 location="bottom", label="TCC difference")
    save(fig, "Fig4_TCC_difference", figures, delivery)

    representative = ("2023-01", "2023-04", "2023-07", "2023-10")
    fields = (("Observation", obs),) + tuple((LABEL[name], test[name]) for name in ORDER)
    fig, axes = plt.subplots(4, 5, figsize=(13.5, 11.3),
                             subplot_kw={"projection": MAP_CRS}, constrained_layout=True)
    for row, month in enumerate(representative):
        index = int(np.flatnonzero(dates == np.datetime64(month, "M"))[0])
        for col, (label, values) in enumerate(fields):
            ax = axes[row, col]
            image = china_panel(ax, values[index], valid, vmin=-1.5, vmax=1.5,
                                cmap="BrBG", show_x=row == 3, show_y=col == 0)
            if row == 0:
                ax.set_title(label, fontsize=10)
            if col == 0:
                ax.text(-0.13, 0.5, month, rotation=90, va="center", ha="right",
                        transform=ax.transAxes, fontsize=8)
    fig.colorbar(image, ax=axes.ravel().tolist(), location="bottom", shrink=0.75,
                 label="Fractional precipitation anomaly (display clipped to [-1.5, 1.5])")
    save(fig, "Fig5_spatial_examples", figures, delivery)


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
    parser.add_argument("--spatial-only", action="store_true")
    args = parser.parse_args()
    paths = SubprojectPaths(__file__)
    source = paths.results_dir / args.experiment
    sensitivity = paths.results_dir / args.sensitivity
    delivery = Path(args.delivery)
    figures = source / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    delivery.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})

    if args.spatial_only:
        with np.load(source / "test_predictions.npz", allow_pickle=True) as data:
            test = {key: data[key].copy() for key in data.files}
        dates = np.asarray(test["dates"], dtype="datetime64[M]")
        spatial_figures(test, dates, test["target"], test["valid"].astype(bool),
                        figures, delivery)
        return

    metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    alt = json.loads((sensitivity / "metrics.json").read_text(encoding="utf-8"))
    if not ("bilinear" in metrics["protocol"]["seas5"] and
            "bicubic" in alt["protocol"]["seas5"]):
        raise ValueError("main/sensitivity source order is incorrect")

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
    axes[0].legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.14), frameon=False)
    axes[0].set_title("(a) Monthly spatial anomaly correlation", loc="left", fontsize=11)
    for name in ("baseline", "eof_cat", "seas5"):
        axes[1].plot(x, series["film"] - series[name], marker="o", ms=3,
                     lw=1.4, color=COLOR[name], label=f"FiLM - {LABEL[name]}")
    axes[1].axhline(0, color="#555555", lw=0.9)
    axes[1].set_ylabel("FiLM ACC difference")
    axes[1].set_title("(b) Paired monthly differences", loc="left", fontsize=11)
    axes[1].legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.14), frameon=False)
    axes[1].set_xticks(x[::2], [str(v) for v in dates[::2]], rotation=45, ha="right")
    fig.tight_layout()
    save(fig, "Fig1_monthly_ACC", figures, delivery)

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
    save(fig, "Fig2_metric_tradeoff", figures, delivery)

    spatial_figures(test, dates, obs, valid, figures, delivery)

    ablation_file = paths.results_dir / "revision06_ablation_v1" / "ablation.json"
    if ablation_file.exists():
        ablation = json.loads(ablation_file.read_text(encoding="utf-8"))
        values = {
            name: [ablation["ensemble"][case][name]["mean_acc"]
                   for case in ("full", "pc_zero", "local_zero")]
            for name in ("baseline", "eof_cat", "film")
        }
        values["baseline"].append(values["baseline"][0])
        values["eof_cat"].append(ablation["pc_shuffle"]["eof_cat"]["mean_acc"])
        values["film"].append(ablation["pc_shuffle"]["film"]["mean_acc"])
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        xloc = np.arange(4)
        width = 0.25
        for offset, name in ((-width, "baseline"), (0, "eof_cat"), (width, "film")):
            ax.bar(xloc + offset, values[name], width, label=LABEL[name], color=COLOR[name])
        ax.set_xticks(xloc, ("Full input", "SST PCs zeroed", "Local fields zeroed",
                              "SST PCs replaced"))
        ax.set_ylabel("Test spatial ACC")
        ax.set_ylim(-0.06, 0.39)
        ax.axhline(0, color="#777777", lw=0.8)
        ax.legend(frameon=False, ncol=3, loc="lower center",
                  bbox_to_anchor=(0.5, 1.01))
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
