# -*- coding: utf-8 -*-
"""
Publication-Grade Figure Generator for FiLM-U-Net Precipitation Prediction Manuscript (EJRH-D-26-03375).
Reads canonical outputs from `D:\\Hydro_Workspace\\results\\U_Net_model\\unified_eval\\`
and dual-archives high-resolution PNG (300 DPI) + vector SVG figures to:
  1. D:\\Hydro_Workspace\\results\\U_Net_model\\unified_eval\\figures\\
  2. C:\\Users\\fired\\OneDrive\\Desktop\\文献与汇报\\02_自研课题与论文\\FiLM调制UNet降水预测\\修改1\\05_返修交付包\\figures\\
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import shapefile
from shapely.geometry import Point as ShapelyPoint, shape as shapely_shape
from shapely.ops import unary_union
from shapely.prepared import prep
from sklearn.decomposition import PCA

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.paths import SubprojectPaths
import utils.maskout as maskout
from utils.eval_precip.visualizer import _load_borders, CHINA_SHP

# Global publication typography
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"


def save_dual(fig, basename: str, out_dirs: list):
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, f"{basename}.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(d, f"{basename}.svg"), bbox_inches="tight")
    print(f"[Figures] Saved {basename}.png and .svg")


def build_china_polygon_mask(lon2d: np.ndarray, lat2d: np.ndarray, shp_path: str) -> np.ndarray:
    """Strictly check whether each (lon, lat) point lies inside CHINA_SHP polygons."""
    sf = shapefile.Reader(str(shp_path))
    geoms = [shapely_shape(rec.shape.__geo_interface__) for rec in sf.shapeRecords()]
    poly_prep = prep(unary_union(geoms))
    inside = np.zeros(lon2d.shape, dtype=bool)
    H, W = lon2d.shape
    for r in range(H):
        for c in range(W):
            if poly_prep.contains(ShapelyPoint(float(lon2d[r, c]), float(lat2d[r, c]))):
                inside[r, c] = True
    return inside


def add_china_map_with_scs_inset(fig, ax, segments, proj, extent=(70, 140, 15, 55)):
    """Configure China Cartopy map axis with borders and bounded South China Sea inset."""
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#F8F8F8")
    lc = LineCollection(segments, colors="black", linewidths=0.45, transform=ccrs.Geodetic())
    ax.add_collection(lc)
    ax.set_extent(list(extent), crs=proj)

    pos = ax.get_position()
    sub_w = 0.165 * pos.width
    sub_h = 0.225 * pos.height
    sub_left = pos.x0 + pos.width - sub_w - 0.006 * pos.width
    sub_bot = pos.y0 + 0.015 * pos.height

    sub_ax = fig.add_axes([sub_left, sub_bot, sub_w, sub_h], projection=proj, facecolor="white")
    sub_ax.patch.set_facecolor("white")
    sub_ax.patch.set_edgecolor("black")
    sub_ax.patch.set_linewidth(0.8)
    sub_ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#F8F8F8")
    lc_sub = LineCollection(segments, colors="black", linewidths=0.35, transform=ccrs.Geodetic())
    sub_ax.add_collection(lc_sub)
    sub_ax.set_extent([105, 125, 0, 25], crs=proj)
    sub_ax.set_xticks([])
    sub_ax.set_yticks([])
    return sub_ax


def main():
    paths = SubprojectPaths(__file__)
    exp_dir = paths.get_exp_dir("unified_eval")
    ws_fig_dir = str(exp_dir / "figures")
    desk_fig_dir = r"C:\Users\fired\OneDrive\Desktop\文献与汇报\02_自研课题与论文\FiLM调制UNet降水预测\修改1\05_返修交付包\figures"
    out_dirs = [ws_fig_dir, desk_fig_dir]

    with open(exp_dir / "unified_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    preds_npz = np.load(exp_dir / "predictions_all.npz", allow_pickle=True)
    interp_npz = np.load(exp_dir / "interpretability_all.npz", allow_pickle=True)

    dates_all = pd.to_datetime(preds_npz["dates"])
    hind_mask = preds_npz["hind_mask"].astype(bool)
    test_mask = preds_npz["test_mask"].astype(bool)
    valid_land_mask = preds_npz["valid_land_mask"].astype(bool)

    time_hind = dates_all[hind_mask]
    time_test = dates_all[test_mask]

    target = preds_npz["target"]
    pred_seas5 = preds_npz["pred_seas5"]
    pred_base = preds_npz["pred_base"]
    pred_cat = preds_npz["pred_cat"]
    pred_film = preds_npz["pred_film"]

    acc_seas5 = preds_npz["acc_seas5"]
    acc_base = preds_npz["acc_base"]
    acc_cat = preds_npz["acc_cat"]
    acc_film = preds_npz["acc_film"]

    tcc_base = preds_npz["tcc_base"]
    pval_base = preds_npz["pval_base"]
    tcc_cat = preds_npz["tcc_cat"]
    pval_cat = preds_npz["pval_cat"]
    tcc_film = preds_npz["tcc_film"]
    pval_film = preds_npz["pval_film"]
    pcs5 = preds_npz["pcs5"]

    borders = _load_borders()
    segments = [line.reshape(-1, 2) for line in borders]
    proj = ccrs.PlateCarree(central_longitude=0.0)

    H_map, W_map = valid_land_mask.shape
    lons = np.arange(70, 140, 0.5)[:W_map]
    lats = np.arange(60, 0, -0.5)[:H_map]
    lon2d, lat2d = np.meshgrid(lons, lats)

    china_poly_mask = (
        build_china_polygon_mask(lon2d, lat2d, str(CHINA_SHP))
        if CHINA_SHP.exists()
        else valid_land_mask
    )

    # =========================================================================
    # FIGURE 2: Macro-Climatological Hindcast (1994-2022) & Test (2023-2024) ACC
    # =========================================================================
    print("Generating Fig. 2...")
    c_seas5_fill, c_seas5_line = "#7E99F4", "#2B5C8F"
    models_f2 = [
        ("Baseline U-Net", acc_base, "#E69F00", "#D55E00", "(a)", "(b)"),
        ("EOF-cat", acc_cat, "#009E73", "#006B4E", "(c)", "(d)"),
        ("FiLM-U-Net", acc_film, "#9E5EB0", "#6A1B9A", "(e)", "(f)"),
    ]

    fig2, axes2 = plt.subplots(3, 2, figsize=(15.5, 13.0), dpi=300)
    plt.subplots_adjust(hspace=0.38, wspace=0.16, top=0.93, bottom=0.06)

    for row, (m_name, acc_m, col_fill, col_line, tag_h, tag_f) in enumerate(models_f2):
        ax_h = axes2[row, 0]
        ax_f = axes2[row, 1]

        m_h_mean = float(np.nanmean(acc_m[hind_mask]))
        s_h_mean = float(np.nanmean(acc_seas5[hind_mask]))

        ax_h.axvspan(pd.Timestamp("2018-01-01"), pd.Timestamp("2022-12-01"), color="#E0E0E0", alpha=0.45, zorder=0)
        ax_h.fill_between(
            time_hind, -0.25, acc_m[hind_mask],
            label=f"{m_name} (Mean: {m_h_mean:.4f})", color=col_fill, alpha=0.82, linewidth=1.0
        )
        ax_h.fill_between(
            time_hind, -0.25, acc_seas5[hind_mask],
            label=f"ECMWF SEAS5 (Mean: {s_h_mean:.4f})", color=c_seas5_fill, alpha=0.75, linewidth=1.0
        )
        ax_h.axhline(y=m_h_mean, color=col_line, linestyle="--", linewidth=1.6)
        ax_h.axhline(y=s_h_mean, color=c_seas5_line, linestyle=":", linewidth=1.6)
        ax_h.axvline(x=pd.Timestamp("2018-01-01"), color="#424242", linestyle="-.", linewidth=1.2)
        ax_h.text(
            pd.Timestamp("2018-05-01"), 0.78, "Internal Val\n(2018–2022)",
            fontsize=8.3, color="#222222", fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#CCCCCC", alpha=0.92),
        )

        ax_h.set_ylim(-0.25, 0.90)
        ax_h.set_ylabel("ACC", fontsize=11, fontweight="bold")
        ax_h.grid(True, linestyle="--", alpha=0.35)
        ax_h.tick_params(axis="both", labelsize=10)
        ax_h.text(
            0.015, 0.93, f"{tag_h} Historical Hindcast (1994–2022)", transform=ax_h.transAxes,
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#DDDDDD", alpha=0.92),
        )
        ax_h.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
            fontsize=9.5, frameon=True, facecolor="white", edgecolor="#CCCCCC"
        )
        if row == 2:
            ax_h.set_xlabel("Historical Hindcast Period (1994.01–2022.12)", fontsize=11, fontweight="bold", labelpad=6)

        m_f_mean = float(np.nanmean(acc_m[test_mask]))
        s_f_mean = float(np.nanmean(acc_seas5[test_mask]))

        ax_f.plot(
            time_test, acc_m[test_mask], label=f"{m_name} (Mean: {m_f_mean:.4f})",
            color=col_line, linewidth=2.1, marker="o", markersize=4.8, alpha=0.92
        )
        ax_f.plot(
            time_test, acc_seas5[test_mask], label=f"ECMWF SEAS5 (Mean: {s_f_mean:.4f})",
            color=c_seas5_line, linewidth=1.9, marker="s", markersize=4.5, alpha=0.88
        )
        ax_f.axhline(y=m_f_mean, color=col_line, linestyle="--", linewidth=1.6)
        ax_f.axhline(y=s_f_mean, color=c_seas5_line, linestyle=":", linewidth=1.6)
        ax_f.axhline(y=0.0, color="gray", linestyle=":", linewidth=0.9)

        ax_f.set_ylim(-0.30, 0.85)
        ax_f.set_ylabel("ACC", fontsize=11, fontweight="bold")
        ax_f.grid(True, linestyle="--", alpha=0.35)
        ax_f.tick_params(axis="both", labelsize=10)
        ax_f.text(
            0.015, 0.93, f"{tag_f} Independent Test (2023.01–2024.09)", transform=ax_f.transAxes,
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#DDDDDD", alpha=0.92),
        )
        ax_f.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax_f.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax_f.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
            fontsize=9.5, frameon=True, facecolor="white", edgecolor="#CCCCCC"
        )
        if row == 2:
            ax_f.set_xlabel("Independent Test Period (2023.01–2024.09)", fontsize=11, fontweight="bold", labelpad=6)

    save_dual(fig2, "Fig2_回报与预测对比", out_dirs)
    plt.close(fig2)

    # =========================================================================
    # FIGURE 3: 4-Panel Comprehensive Temporal Evaluation
    # =========================================================================
    print("Generating Fig. 3...")
    df_hind = pd.DataFrame({
        "month": time_hind.month,
        "year": time_hind.year,
        "SEAS5": acc_seas5[hind_mask],
        "Baseline": acc_base[hind_mask],
        "EOF-cat": acc_cat[hind_mask],
        "FiLM-U-Net": acc_film[hind_mask],
    })
    monthly_mean = df_hind.groupby("month")[["SEAS5", "Baseline", "EOF-cat", "FiLM-U-Net"]].mean()
    yearly_mean = df_hind.groupby("year")[["SEAS5", "Baseline", "EOF-cat", "FiLM-U-Net"]].mean()

    fig3, (ax3a, ax3b, ax3c, ax3d) = plt.subplots(
        4, 1, figsize=(15.5, 19.5), dpi=300,
        gridspec_kw={"height_ratios": [1.15, 1.15, 1.05, 0.95]}
    )
    plt.subplots_adjust(hspace=0.40, top=0.95, bottom=0.05)

    models_f3 = ["SEAS5", "Baseline", "EOF-cat", "FiLM-U-Net"]
    colors_f3 = ["#7E99F4", "#E69F00", "#009E73", "#9E5EB0"]
    labels_f3 = ["ECMWF SEAS5", "Baseline U-Net", "EOF-cat", "FiLM-U-Net (Ours)"]

    x_m = np.arange(1, 13)
    w_m = 0.20
    for idx, (m_key, col, lab) in enumerate(zip(models_f3, colors_f3, labels_f3)):
        offset = (idx - 1.5) * w_m
        bars = ax3a.bar(x_m + offset, monthly_mean[m_key], width=w_m, color=col, label=lab, alpha=0.90)
        for bar in bars:
            h_val = bar.get_height()
            va_pos = "bottom" if h_val >= 0 else "top"
            ax3a.text(
                bar.get_x() + bar.get_width() / 2.0,
                h_val + (0.012 if h_val >= 0 else -0.025),
                f"{h_val:.2f}", ha="center", va=va_pos, fontsize=7.5, rotation=90
            )
    ax3a.set_ylabel("Mean ACC", fontsize=11, fontweight="bold")
    ax3a.set_xticks(x_m)
    ax3a.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=10.5)
    ax3a.set_ylim(-0.08, 0.65)
    ax3a.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax3a.set_title("(a) Multi-year mean ACC across calendar months (1994–2022, in-sample hindcast)", loc="left", fontsize=11.5, fontweight="bold", pad=26)
    ax3a.legend(loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=4, fontsize=9.5, frameon=True, edgecolor="#CCCCCC")

    years_h = yearly_mean.index.values
    x_y = np.arange(len(years_h))
    w_y = 0.20
    for idx, (m_key, col, lab) in enumerate(zip(models_f3, colors_f3, labels_f3)):
        offset = (idx - 1.5) * w_y
        ax3b.bar(x_y + offset, yearly_mean[m_key], width=w_y, color=col, label=lab, alpha=0.90)
    val_start_x = int(np.where(years_h == 2018)[0][0]) - 0.5
    ax3b.axvspan(val_start_x, len(years_h) - 0.5, color="#EEEEEE", alpha=0.7, zorder=0)
    ax3b.text(val_start_x + 0.2, 0.57, "Internal Val (2018–2022)", fontsize=9, fontweight="bold", color="#424242")
    ax3b.set_ylabel("Mean ACC", fontsize=11, fontweight="bold")
    ax3b.set_xticks(x_y[::2])
    ax3b.set_xticklabels(years_h[::2], fontsize=10, rotation=35)
    ax3b.set_ylim(-0.15, 0.65)
    ax3b.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax3b.set_title("(b) Annual mean ACC across historical years (1994–2022, in-sample hindcast)", loc="left", fontsize=11.5, fontweight="bold", pad=26)
    ax3b.legend(loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=4, fontsize=9.5, frameon=True, edgecolor="#CCCCCC")

    ts_cfgs = [
        ("SEAS5", acc_seas5[test_mask], "#2B5C8F", "ECMWF SEAS5", "s", "--", 1.8),
        ("Baseline", acc_base[test_mask], "#D55E00", "Baseline U-Net", "^", "-.", 2.0),
        ("EOF-cat", acc_cat[test_mask], "#0072B2", "EOF-cat", "D", ":", 2.0),
        ("FiLM-U-Net", acc_film[test_mask], "#78288C", "FiLM-U-Net (Ours)", "o", "-", 2.4),
    ]
    for m_key, s_arr, col, lab, mark, ls, lw in ts_cfgs:
        ax3c.plot(
            time_test, s_arr, label=f"{lab} ({np.nanmean(s_arr):.4f})",
            color=col, linewidth=lw, linestyle=ls, marker=mark, markersize=5.0, alpha=0.92
        )
        ax3c.axhline(y=float(np.nanmean(s_arr)), color=col, linestyle=":", linewidth=1.1, alpha=0.65)
    ax3c.axhline(y=0.0, color="gray", linestyle=":", linewidth=1.0)
    ax3c.set_ylabel("Test ACC", fontsize=11, fontweight="bold")
    ax3c.set_ylim(-0.35, 0.85)
    ax3c.grid(True, linestyle="--", alpha=0.35)
    test_ticks = pd.date_range(time_test[0], time_test[-1], freq="2MS")
    ax3c.set_xticks(test_ticks)
    ax3c.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax3c.set_xlim(time_test[0] - pd.Timedelta(days=5), time_test[-1] + pd.Timedelta(days=20))
    plt.setp(ax3c.get_xticklabels(), rotation=25)
    ax3c.set_title("(c) Monthly ACC trajectory across independent test period (2023.01–2024.09)", loc="left", fontsize=11.5, fontweight="bold", pad=26)
    ax3c.legend(loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=4, fontsize=9.0, frameon=True, edgecolor="#CCCCCC")

    delta_acc = acc_film[test_mask] - acc_cat[test_mask]
    mean_delta = float(np.nanmean(delta_acc))
    n_test_m = len(delta_acc)
    pos_cnt = int(np.sum(delta_acc > 0))
    x_test = np.arange(n_test_m)
    bar_cols = ["#78288C" if v >= 0 else "#D0021B" for v in delta_acc]
    ax3d.bar(x_test, delta_acc, width=0.72, color=bar_cols, alpha=0.86, edgecolor="black", linewidth=0.5)
    ax3d.axhline(y=0.0, color="black", linestyle="-", linewidth=1.0)
    ax3d.axhline(y=mean_delta, color="#78288C", linestyle="--", linewidth=1.6, label=f"Mean ΔACC: {mean_delta:+.4f} ({pos_cnt}/{n_test_m} months > 0)")
    ax3d.set_ylabel("ΔACC\n(FiLM − Cat)", fontsize=11, fontweight="bold")
    d_min = min(float(np.nanmin(delta_acc)), -0.36)
    d_max = max(float(np.nanmax(delta_acc)), 0.18)
    ax3d.set_ylim(d_min - 0.06, d_max + 0.06)
    ax3d.grid(True, linestyle="--", alpha=0.35)
    ax3d.set_xticks(x_test[::2])
    ax3d.set_xticklabels([pd.Timestamp(d).strftime("%b %Y") for d in time_test[::2]], rotation=25)
    ax3d.set_xlim(-0.6, n_test_m - 0.4)
    ax3d.set_xlabel("Independent Test Period (2023.01–2024.09, 21 months)", fontsize=11, fontweight="bold", labelpad=6)
    ax3d.set_title(f"(d) Monthly paired ΔACC (FiLM − EOF-cat); mean {mean_delta:+.4f} ({pos_cnt}/{n_test_m} months > 0)", loc="left", fontsize=11.5, fontweight="bold", pad=24)
    ax3d.legend(loc="lower right", bbox_to_anchor=(1.0, 1.01), fontsize=9.5, frameon=True, edgecolor="#CCCCCC")

    save_dual(fig3, "Fig3_综合时间尺度对比", out_dirs)
    plt.close(fig3)

    # =========================================================================
    # FIGURE 4: Spatial Forecast Comparison (4 Seasonal Months x 5 Columns)
    # =========================================================================
    print("Generating Fig. 4...")
    obs_test = target[test_mask]
    p_seas5_t = pred_seas5[test_mask]
    p_base_t = pred_base[test_mask]
    p_cat_t = pred_cat[test_mask]
    p_film_t = pred_film[test_mask]

    month_indices = [0, 3, 6, 9]
    month_names = ["Jan 2023", "Apr 2023", "Jul 2023", "Oct 2023"]
    col_names = ["Observation", "SEAS5 (Raw)", "Baseline U-Net", "EOF-cat", "FiLM-U-Net (Ours)"]

    fig4 = plt.figure(figsize=(16.0, 11.5), facecolor="white")
    gs4 = fig4.add_gridspec(4, 5, top=0.94, bottom=0.10, left=0.06, right=0.98, wspace=0.08, hspace=0.12)
    cmap_p = plt.cm.BrBG
    norm_p = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    levels_p = np.linspace(-1.0, 1.0, 11)
    last_cf4 = None

    for r_idx, (m_idx, m_name) in enumerate(zip(month_indices, month_names)):
        row_fields = [obs_test[m_idx], p_seas5_t[m_idx], p_base_t[m_idx], p_cat_t[m_idx], p_film_t[m_idx]]
        for c_idx, fld in enumerate(row_fields):
            ax = fig4.add_subplot(gs4[r_idx, c_idx], projection=proj)
            add_china_map_with_scs_inset(fig4, ax, segments, proj)
            if c_idx == 0:
                ax.set_yticks(range(20, 56, 10), crs=proj)
                ax.yaxis.set_major_formatter(LatitudeFormatter())
                ax.tick_params(axis="y", labelsize=8)
                ax.set_ylabel(m_name, fontsize=11, fontweight="bold", labelpad=6)
            else:
                ax.set_yticks([], crs=proj)
            if r_idx == 3:
                ax.set_xticks(range(70, 141, 20), crs=proj)
                ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))
                ax.tick_params(axis="x", labelsize=8)
            else:
                ax.set_xticks([], crs=proj)
            if r_idx == 0:
                ax.set_title(col_names[c_idx], fontsize=11, fontweight="bold", pad=8)

            cf = ax.contourf(lons, lats, fld, levels=levels_p, cmap=cmap_p, norm=norm_p, extend="both", transform=proj)
            last_cf4 = cf
            if CHINA_SHP.exists():
                try:
                    maskout.shp2clip(cf, ax, str(CHINA_SHP))
                except Exception:
                    pass

    cbar_ax4 = fig4.add_axes([0.25, 0.045, 0.50, 0.022])
    cb4 = fig4.colorbar(last_cf4, cax=cbar_ax4, orientation="horizontal", extend="both")
    cb4.set_label("Precipitation Anomaly Fraction (ΔP)", fontsize=10, fontweight="bold")
    cb4.ax.tick_params(labelsize=9)
    save_dual(fig4, "Fig4_空间预报对比", out_dirs)
    plt.close(fig4)

    # =========================================================================
    # FIGURE 5: 2x2 Spatial TCC & Significance Stippling (p < 0.05) + Difference
    # Strictly clips stippling dots inside China polygon boundary!
    # =========================================================================
    print("Generating Fig. 5...")
    tcc_diff = tcc_film - tcc_cat
    mean_tcc1 = float(np.nanmean(tcc_base[valid_land_mask].astype(np.float64)))
    mean_tcc2 = float(np.nanmean(tcc_cat[valid_land_mask].astype(np.float64)))
    mean_tcc3 = float(np.nanmean(tcc_film[valid_land_mask].astype(np.float64)))
    mean_diff_tcc = float(metrics["paired_stat_tests"]["test_tcc_spatial"]["film_vs_cat_mean_dtcc"])

    n_val_g = int(np.sum(valid_land_mask))
    sig_pct1 = float(np.sum((pval_base < 0.05) & valid_land_mask) / n_val_g * 100.0)
    sig_pct2 = float(np.sum((pval_cat < 0.05) & valid_land_mask) / n_val_g * 100.0)
    sig_pct3 = float(np.sum((pval_film < 0.05) & valid_land_mask) / n_val_g * 100.0)

    fig5 = plt.figure(figsize=(13.2, 10.5), facecolor="white")
    gs5 = fig5.add_gridspec(2, 2, top=0.95, bottom=0.13, left=0.08, right=0.96, wspace=0.12, hspace=0.22)

    norm_tcc = mcolors.Normalize(vmin=-0.6, vmax=0.6)
    levels_tcc = np.linspace(-0.6, 0.6, 13)
    norm_diff = mcolors.Normalize(vmin=-0.4, vmax=0.4)
    levels_diff = np.linspace(-0.4, 0.4, 9)

    panels5 = [
        (0, 0, tcc_base, f"(a) Baseline U-Net (Mean TCC: {mean_tcc1:.3f}, Sig: {sig_pct1:.1f}%)", pval_base, False),
        (0, 1, tcc_cat, f"(b) EOF-cat (Mean TCC: {mean_tcc2:.3f}, Sig: {sig_pct2:.1f}%)", pval_cat, False),
        (1, 0, tcc_film, f"(c) FiLM-U-Net (Mean TCC: {mean_tcc3:.3f}, Sig: {sig_pct3:.1f}%)", pval_film, False),
        (1, 1, tcc_diff, f"(d) Difference: FiLM − Cat (Mean ΔTCC: {mean_diff_tcc:+.4f})", None, True),
    ]

    last_cf5_tcc = None
    last_cf5_diff = None
    for r_idx, c_idx, data, title, pval, is_diff in panels5:
        ax = fig5.add_subplot(gs5[r_idx, c_idx], projection=proj)
        add_china_map_with_scs_inset(fig5, ax, segments, proj)
        ax.set_yticks(range(20, 56, 10), crs=proj)
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.set_xticks(range(70, 141, 20), crs=proj)
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))
        ax.tick_params(axis="both", labelsize=8.5)
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8)

        if not is_diff:
            cf = ax.contourf(lons, lats, data, levels=levels_tcc, cmap=plt.cm.RdBu_r, norm=norm_tcc, extend="both", transform=proj)
            last_cf5_tcc = cf
        else:
            cf = ax.contourf(lons, lats, data, levels=levels_diff, cmap=plt.cm.bwr, norm=norm_diff, extend="both", transform=proj)
            last_cf5_diff = cf

        if CHINA_SHP.exists():
            try:
                maskout.shp2clip(cf, ax, str(CHINA_SHP))
            except Exception:
                pass

        if pval is not None:
            sig_mask = (pval < 0.05) & valid_land_mask & china_poly_mask
            step = 3
            sub_sig = sig_mask[::step, ::step]
            sc = ax.scatter(
                lon2d[::step, ::step][sub_sig], lat2d[::step, ::step][sub_sig],
                s=2.2, color="black", alpha=0.85, transform=proj, zorder=4
            )
            if CHINA_SHP.exists():
                try:
                    maskout.shp2clip(sc, ax, str(CHINA_SHP), vcplot=True)
                except Exception:
                    pass

    cbar_ax5_1 = fig5.add_axes([0.12, 0.055, 0.35, 0.022])
    cb5_1 = fig5.colorbar(last_cf5_tcc, cax=cbar_ax5_1, orientation="horizontal", extend="both", ticks=[-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    cb5_1.set_label("Temporal Correlation Coefficient (TCC)", fontsize=9.5, fontweight="bold")
    cb5_1.ax.tick_params(labelsize=8.5)

    cbar_ax5_2 = fig5.add_axes([0.56, 0.055, 0.35, 0.022])
    cb5_2 = fig5.colorbar(last_cf5_diff, cax=cbar_ax5_2, orientation="horizontal", extend="both", ticks=[-0.4, -0.2, 0.0, 0.2, 0.4])
    cb5_2.set_label("TCC Difference (ΔTCC: FiLM − Cat)", fontsize=9.5, fontweight="bold")
    cb5_2.ax.tick_params(labelsize=8.5)

    save_dual(fig5, "Fig5_TCC空间检验", out_dirs)
    plt.close(fig5)

    # =========================================================================
    # FIGURE 6: 2D PCA Projection of Layer-5 FiLM Parameters (gamma, beta)
    # =========================================================================
    print("Generating Fig. 6...")
    gamma_l4 = interp_npz["gamma_l4"][hind_mask]
    beta_l4 = interp_npz["beta_l4"][hind_mask]
    pc1_hind = pcs5[hind_mask, 0]

    proj_g = PCA(n_components=2).fit_transform(gamma_l4)
    proj_b = PCA(n_components=2).fit_transform(beta_l4)

    fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=300)
    sc_a = ax6a.scatter(proj_g[:, 0], proj_g[:, 1], c=pc1_hind, cmap="viridis", alpha=0.85, s=26, linewidth=0.3, edgecolors="k")
    cb6a = fig6.colorbar(sc_a, ax=ax6a)
    cb6a.set_label("PC1 (ENSO)", fontsize=10.5, fontweight="bold")
    ax6a.set_xlabel("PCA 1", fontsize=11, fontweight="bold")
    ax6a.set_ylabel("PCA 2", fontsize=11, fontweight="bold")
    ax6a.set_title("(a) PCA of γ (Layer 5) colored by PC1", fontsize=11.5, fontweight="bold")
    ax6a.grid(True, linestyle="--", alpha=0.3)

    sc_b = ax6b.scatter(proj_b[:, 0], proj_b[:, 1], c=pc1_hind, cmap="viridis", alpha=0.85, s=26, linewidth=0.3, edgecolors="k")
    cb6b = fig6.colorbar(sc_b, ax=ax6b)
    cb6b.set_label("PC1 (ENSO)", fontsize=10.5, fontweight="bold")
    ax6b.set_xlabel("PCA 1", fontsize=11, fontweight="bold")
    ax6b.set_ylabel("PCA 2", fontsize=11, fontweight="bold")
    ax6b.set_title("(b) PCA of β (Layer 5) colored by PC1", fontsize=11.5, fontweight="bold")
    ax6b.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    save_dual(fig6, "Fig6_参数PCA投影", out_dirs)
    plt.close(fig6)

    # =========================================================================
    # FIGURE 7: Channel-PC Correlations + Shuffled-PC Negative Control
    # =========================================================================
    print("Generating Fig. 7...")
    corr_gamma_l0 = interp_npz["corr_gamma_l0"]
    corr_beta_l0 = interp_npz["corr_beta_l0"]
    shuf_g_abs = interp_npz["shuf_gamma_mean_abs"]
    shuf_b_abs = interp_npz["shuf_beta_mean_abs"]

    fig7 = plt.figure(figsize=(15.5, 8.2), dpi=300)
    gs7 = fig7.add_gridspec(2, 3, width_ratios=[1.1, 1.1, 1.25], wspace=0.32, hspace=0.36)

    ax7a = fig7.add_subplot(gs7[:, 0])
    ax7b = fig7.add_subplot(gs7[:, 1])
    ax7c = fig7.add_subplot(gs7[0, 2])
    ax7d = fig7.add_subplot(gs7[1, 2])

    ch_labels = [f"Ch {i+1}" for i in range(16)]
    pc_labels = [f"PC{j+1}" for j in range(5)]

    vmax_hm = max(float(np.max(np.abs(corr_gamma_l0))), float(np.max(np.abs(corr_beta_l0))))
    vmax_hm = max(0.75, round(vmax_hm, 2))

    im7a = ax7a.imshow(corr_gamma_l0, cmap="RdBu_r", vmin=-vmax_hm, vmax=vmax_hm, aspect="auto")
    ax7a.set_xticks(range(5))
    ax7a.set_xticklabels(pc_labels, fontsize=9.5, fontweight="bold")
    ax7a.set_yticks(range(16))
    ax7a.set_yticklabels(ch_labels, fontsize=9)
    ax7a.set_xlabel("Principal Components", fontsize=10.5, fontweight="bold")
    ax7a.set_ylabel("Encoder Layer 1 Channels", fontsize=10.5, fontweight="bold")
    ax7a.set_title("(a) Correlation: FiLM γ vs. SST PCs", fontsize=11, fontweight="bold", pad=10)
    for r_i in range(16):
        for c_j in range(5):
            val = corr_gamma_l0[r_i, c_j]
            ax7a.text(c_j, r_i, f"{val:.2f}", ha="center", va="center", fontsize=7.5,
                      color="white" if abs(val) > 0.45 * vmax_hm else "black")
    cb7a = fig7.colorbar(im7a, ax=ax7a, fraction=0.046, pad=0.04)
    cb7a.set_label("Pearson r", fontsize=9.5)

    im7b = ax7b.imshow(corr_beta_l0, cmap="RdBu_r", vmin=-vmax_hm, vmax=vmax_hm, aspect="auto")
    ax7b.set_xticks(range(5))
    ax7b.set_xticklabels(pc_labels, fontsize=9.5, fontweight="bold")
    ax7b.set_yticks(range(16))
    ax7b.set_yticklabels(ch_labels, fontsize=9)
    ax7b.set_xlabel("Principal Components", fontsize=10.5, fontweight="bold")
    ax7b.set_title("(b) Correlation: FiLM β vs. SST PCs", fontsize=11, fontweight="bold", pad=10)
    for r_i in range(16):
        for c_j in range(5):
            val = corr_beta_l0[r_i, c_j]
            ax7b.text(c_j, r_i, f"{val:.2f}", ha="center", va="center", fontsize=7.5,
                      color="white" if abs(val) > 0.45 * vmax_hm else "black")
    cb7b = fig7.colorbar(im7b, ax=ax7b, fraction=0.046, pad=0.04)
    cb7b.set_label("Pearson r", fontsize=9.5)

    x_pc = np.arange(5)
    w_bar = 0.36
    g_real_m = np.mean(np.abs(corr_gamma_l0), axis=0)
    g_real_s = np.std(np.abs(corr_gamma_l0), axis=0)
    g_shuf_m = np.mean(shuf_g_abs, axis=0)
    g_shuf_s = np.std(shuf_g_abs, axis=0)
    g_shuf_p95 = float(np.max(np.percentile(shuf_g_abs, 95, axis=0)))

    ax7c.bar(x_pc - w_bar / 2, g_real_m, yerr=g_real_s, width=w_bar, capsize=3.5,
             color="#4A7BB7", edgecolor="#1D3557", label="Observed SST PCs")
    ax7c.bar(x_pc + w_bar / 2, g_shuf_m, yerr=g_shuf_s, width=w_bar, capsize=3.5,
             color="#B0BEC5", edgecolor="#455A64", hatch="//", label="Shuffled PCs Control")
    ax7c.axhline(y=g_shuf_p95, color="#D32F2F", linestyle="--", linewidth=1.2, label=f"95% Noise Ceiling ({g_shuf_p95:.2f})")
    for i_p, v_p in enumerate(g_real_m):
        ax7c.text(i_p - w_bar / 2, v_p + g_real_s[i_p] + 0.015, f"{v_p:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax7c.set_xticks(x_pc)
    ax7c.set_xticklabels(pc_labels, fontsize=9.5, fontweight="bold")
    ax7c.set_ylim(0, max(g_real_m + g_real_s) * 1.25)
    ax7c.set_ylabel("Mean |r| across channels", fontsize=9.5, fontweight="bold")
    ax7c.set_title("(c) Layer 1 FiLM γ–PC Strength vs. Control", fontsize=10.5, fontweight="bold")
    ax7c.grid(True, linestyle="--", alpha=0.3)
    ax7c.legend(loc="upper right", fontsize=8, frameon=True, edgecolor="#CCCCCC")

    b_real_m = np.mean(np.abs(corr_beta_l0), axis=0)
    b_real_s = np.std(np.abs(corr_beta_l0), axis=0)
    b_shuf_m = np.mean(shuf_b_abs, axis=0)
    b_shuf_s = np.std(shuf_b_abs, axis=0)
    b_shuf_p95 = float(np.max(np.percentile(shuf_b_abs, 95, axis=0)))

    ax7d.bar(x_pc - w_bar / 2, b_real_m, yerr=b_real_s, width=w_bar, capsize=3.5,
             color="#4A7BB7", edgecolor="#1D3557", label="Observed SST PCs")
    ax7d.bar(x_pc + w_bar / 2, b_shuf_m, yerr=b_shuf_s, width=w_bar, capsize=3.5,
             color="#B0BEC5", edgecolor="#455A64", hatch="//", label="Shuffled PCs Control")
    ax7d.axhline(y=b_shuf_p95, color="#D32F2F", linestyle="--", linewidth=1.2, label=f"95% Noise Ceiling ({b_shuf_p95:.2f})")
    for i_p, v_p in enumerate(b_real_m):
        ax7d.text(i_p - w_bar / 2, v_p + b_real_s[i_p] + 0.015, f"{v_p:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax7d.set_xticks(x_pc)
    ax7d.set_xticklabels(pc_labels, fontsize=9.5, fontweight="bold")
    ax7d.set_ylim(0, max(b_real_m + b_real_s) * 1.25)
    ax7d.set_ylabel("Mean |r| across channels", fontsize=9.5, fontweight="bold")
    ax7d.set_title("(d) Layer 1 FiLM β–PC Strength vs. Control", fontsize=10.5, fontweight="bold")
    ax7d.grid(True, linestyle="--", alpha=0.3)
    ax7d.legend(loc="upper right", fontsize=8, frameon=True, edgecolor="#CCCCCC")

    save_dual(fig7, "Fig7_参数相关性", out_dirs)
    plt.close(fig7)

    # =========================================================================
    # FIGURE 8: 6-Panel Symmetric Grad-CAM Interpretability Ablation over China
    # Uses common China spatial-span normalization so constant-bias local-zeroed
    # maps accurately collapse to [0.0, 0.2] dark blue.
    # =========================================================================
    print("Generating Fig. 8 (6-panel symmetric Grad-CAM)...")
    t5 = metrics["table5_symmetric_ablation"]
    panels8 = [
        (0, 0, interp_npz["cam_baseline"], f"(a) Baseline U-Net (ACC={t5['Baseline U-Net (No SST)']['test_acc']:.4f})"),
        (0, 1, interp_npz["cam_cat_sst_zero"], f"(b) EOF-cat: SST Zeroed (ACC={t5['EOF-cat (SST PCs Zeroed)']['test_acc']:.4f})"),
        (0, 2, interp_npz["cam_cat_local_zero"], f"(c) EOF-cat: Local Zeroed (ACC={t5['EOF-cat (Local Zeroed)']['test_acc']:.4f})"),
        (1, 0, interp_npz["cam_film_full"], f"(d) Full FiLM-U-Net (ACC={t5['Full FiLM-U-Net (Ours)']['test_acc']:.4f})"),
        (1, 1, interp_npz["cam_film_sst_zero"], f"(e) FiLM: SST Zeroed (ACC={t5['FiLM-U-Net (SST PCs Zeroed)']['test_acc']:.4f})"),
        (1, 2, interp_npz["cam_film_local_zero"], f"(f) FiLM: Local Zeroed (ACC={t5['FiLM-U-Net (Local Zeroed)']['test_acc']:.4f})"),
    ]

    cam_colors = ["#2B83BA", "#5EADE2", "#ABEBC6", "#F9E79F", "#F5B041", "#E74C3C"]
    cmap_cam = mcolors.ListedColormap(cam_colors)
    bounds_cam = [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]
    norm_cam = mcolors.BoundaryNorm(bounds_cam, cmap_cam.N)

    fig8 = plt.figure(figsize=(16.5, 9.2), facecolor="white")
    gs8 = fig8.add_gridspec(2, 3, top=0.93, bottom=0.12, left=0.05, right=0.98, wspace=0.10, hspace=0.22)
    last_cf8 = None

    for r_idx, c_idx, cam_data, title in panels8:
        ax = fig8.add_subplot(gs8[r_idx, c_idx], projection=proj)
        add_china_map_with_scs_inset(fig8, ax, segments, proj)
        ax.set_yticks(range(20, 56, 10), crs=proj)
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.set_xticks(range(70, 141, 20), crs=proj)
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))
        ax.tick_params(axis="both", labelsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

        cf = ax.contourf(lons, lats, cam_data, levels=bounds_cam, cmap=cmap_cam, norm=norm_cam, extend="neither", transform=proj)
        last_cf8 = cf
        if CHINA_SHP.exists():
            try:
                maskout.shp2clip(cf, ax, str(CHINA_SHP))
            except Exception:
                pass

    cbar_ax8 = fig8.add_axes([0.28, 0.05, 0.44, 0.024])
    cb8 = fig8.colorbar(last_cf8, cax=cbar_ax8, orientation="horizontal", ticks=bounds_cam)
    cb8.set_label("Normalized Spatial Grad-CAM Intensity (Common China Span)", fontsize=10, fontweight="bold")
    cb8.ax.tick_params(labelsize=9)

    save_dual(fig8, "Fig8_对称消融", out_dirs)
    plt.close(fig8)

    # =========================================================================
    # FIGURE 9: Global FiLM-EOF Teleconnection Sensitivity Maps (Layers 1-5 + Shuffled Control)
    # Fixed: True NOAA ERSST v5 grid (0..358E, -88..88N) + add_cyclic_point + Pacific-centered
    # projection + sequential YlOrRd colormap [0, 1] with extend='neither'!
    # =========================================================================
    print("Generating Fig. 9 (Layers 1-5 + Shuffled-PC Control, fixed 0-360E & YlOrRd)...")
    sens_list = [
        (interp_npz["sens_l0"], "(a) Layer 1 — 16 channels"),
        (interp_npz["sens_l1"], "(b) Layer 2 — 32 channels"),
        (interp_npz["sens_l2"], "(c) Layer 3 — 64 channels"),
        (interp_npz["sens_l3"], "(d) Layer 4 — 128 channels"),
        (interp_npz["sens_l4"], "(e) Layer 5 — 128 channels"),
        (interp_npz["sens_shuf_mean"], "(f) Shuffled-PC Control (Layer 1, 100-trial mean)"),
    ]

    H_s, W_s = sens_list[0][0].shape
    lats_g = np.arange(-88, 90, 2, dtype=np.float32)[:H_s]
    lons_g = np.arange(0, 360, 2, dtype=np.float32)[:W_s]

    fig9 = plt.figure(figsize=(14.5, 10.5), facecolor="white")
    gs9 = fig9.add_gridspec(3, 2, top=0.95, bottom=0.06, left=0.06, right=0.88, wspace=0.12, hspace=0.26)
    levels_s = np.linspace(0.0, 1.0, 21)
    norm_s = mcolors.Normalize(vmin=0.0, vmax=1.0)
    proj9 = ccrs.PlateCarree(central_longitude=180.0)
    last_cf9 = None

    for idx_s, (s_map, title_s) in enumerate(sens_list):
        r_i, c_i = idx_s // 2, idx_s % 2
        ax = fig9.add_subplot(gs9[r_i, c_i], projection=proj9)
        ax.set_global()
        s_cyc, lon_cyc = add_cyclic_point(s_map, coord=lons_g)
        cf = ax.contourf(
            lon_cyc, lats_g, s_cyc, levels=levels_s, cmap="YlOrRd",
            norm=norm_s, extend="neither", transform=ccrs.PlateCarree()
        )
        last_cf9 = cf
        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#EBEBEB", edgecolor="#444444", linewidth=0.5, zorder=2)
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.5, edgecolor="#333333", zorder=3)
        ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
        ax.set_yticks([-60, -30, 0, 30, 60], crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(axis="both", labelsize=8.5)
        ax.set_title(title_s, fontsize=10.5, fontweight="bold", pad=6)

    cbar_ax9 = fig9.add_axes([0.90, 0.14, 0.022, 0.72])
    cb9 = fig9.colorbar(last_cf9, cax=cbar_ax9, orientation="vertical", ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb9.set_label("Sensitivity (Normalized to Global Peak)", fontsize=10.5, fontweight="bold")
    cb9.ax.tick_params(labelsize=9)

    save_dual(fig9, "Fig9_遥相关敏感性", out_dirs)
    plt.close(fig9)

    # =========================================================================
    # FIGURE_IMPORTANCE: Permutation Feature Importance (10 Atmospheric Predictors)
    # =========================================================================
    print("Generating Fig_特征重要性...")
    imp_items = metrics["table3_permutation_importance"]
    var_disp = [it["full_name"] for it in imp_items]
    m_drops = [it["mean_drop"] for it in imp_items]
    s_drops = [it["std_drop"] for it in imp_items]

    fig_imp, ax_imp = plt.subplots(figsize=(9.5, 5.2), dpi=300)
    y_pos = np.arange(len(var_disp))
    cols_imp = ["#C62828" if v > 0.03 else ("#1565C0" if v > 0 else "#757575") for v in m_drops]
    ax_imp.barh(
        y_pos, m_drops, xerr=s_drops, align="center", color=cols_imp, alpha=0.88,
        edgecolor="black", linewidth=0.8, capsize=4, error_kw={"elinewidth": 1.1, "ecolor": "#333333"}
    )
    ax_imp.set_yticks(y_pos)
    ax_imp.set_yticklabels(var_disp, fontsize=9.5, fontweight="bold")
    ax_imp.invert_yaxis()
    ax_imp.axvline(0.0, color="black", linewidth=1.0)
    ax_imp.set_xlabel("Permutation Feature Importance (ΔACC Drop When Shuffled)", fontsize=10.5, fontweight="bold")
    ax_imp.set_title("Permutation Feature Importance of 10 Atmospheric Predictors (Test Set)", fontsize=11.5, fontweight="bold", pad=10)
    ax_imp.grid(True, linestyle="--", alpha=0.45, axis="x")
    max_x = max(m + s for m, s in zip(m_drops, s_drops))
    min_x = min(0.0, min(m - s for m, s in zip(m_drops, s_drops)))
    ax_imp.set_xlim(min_x - 0.085, max_x * 1.18)

    for idx_v, (v_m, v_s) in enumerate(zip(m_drops, s_drops)):
        x_txt = (v_m + v_s + 0.006) if v_m >= 0 else (v_m - v_s - 0.006)
        ha_align = "left" if v_m >= 0 else "right"
        ax_imp.text(x_txt, idx_v, f"{v_m:+.4f}", va="center", ha=ha_align, fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    save_dual(fig_imp, "Fig_特征重要性", out_dirs)
    plt.close(fig_imp)

    print("[Figures] All figures successfully generated and dual-archived!")


if __name__ == "__main__":
    main()
