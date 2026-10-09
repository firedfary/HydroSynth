"""Multi-lead spatial comparison and comprehensive diagnostic pipeline.

Generates:
1. 42 quarterly spatial comparison maps (6 leads x 7 quarters) covering all 21 test months.
2. Macro ACC gain matrix heatmap (21 months x 6 leads).
3. Top breakthrough and summer monsoon case comparison figures.
4. Comprehensive CSV and metrics summary.
Synchronizes all figures to Desktop research repository.
"""

from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Project root and local imports
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

from utils.paths import SubprojectPaths
from utils.eval_precip.metrics_continuous import spatial_acc
from utils.eval_precip.visualizer import plot_spatial_comparison
from project_paths import OBSERVATION_FILE


def render_single_quarter(args):
    (
        lead,
        q_name,
        t_indices,
        month_labels,
        obs_slice,
        ec_slice,
        pred_slice,
        valid_mask,
        weights,
        lats,
        lons,
        save_path_native,
        save_path_desktop,
    ) = args

    fields = []
    titles = []
    letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    p_idx = 0

    for i, t_idx in enumerate(t_indices):
        m_str = month_labels[i]
        o = obs_slice[i]
        e = ec_slice[i]
        p = pred_slice[i]

        acc_e = float(spatial_acc(e[valid_mask], o[valid_mask], weights))
        acc_p = float(spatial_acc(p[valid_mask], o[valid_mask], weights))
        gain = acc_p - acc_e

        fields.append(o)
        titles.append(f"({letters[p_idx]}) Observation ({m_str})")
        p_idx += 1

        fields.append(e)
        titles.append(f"({letters[p_idx]}) SEAS5 (ACC={acc_e:+.2f})")
        p_idx += 1

        fields.append(p)
        titles.append(f"({letters[p_idx]}) ReMAP (ACC={acc_p:+.2f}, \u0394={gain:+.2f})")
        p_idx += 1

    # Compute joint model p98 for logging the selected discrete tier
    e_p98 = float(np.percentile(np.abs(ec_slice[:, valid_mask]), 98.0))
    p_p98 = float(np.percentile(np.abs(pred_slice[:, valid_mask]), 98.0))
    ref_amp = max(e_p98, p_p98)
    chosen_tier = 1.0
    for t_val in (0.3, 0.5, 0.8, 1.0):
        if ref_amp <= t_val * 1.05:
            chosen_tier = t_val
            break

    plot_spatial_comparison(
        fields=fields,
        titles=titles,
        latitudes=lats,
        longitudes=lons,
        ncols=3,
        vmaxs=None,
        color_modes=["diverging"] * 9,
        save_path=save_path_native,
        dpi=300,
        shared_colorbar="col",
    )

    if save_path_desktop:
        save_path_desktop.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(save_path_native, save_path_desktop)

    return f"Lead {lead} - {q_name} [Model Tier: \u00b1{chosen_tier:.1f} (p98={ref_amp:.2f})] -> {save_path_native.name}"


def plot_acc_gain_heatmap(
    df: pd.DataFrame,
    save_path_native: Path,
    save_path_desktop: Path,
):
    """Plot high-resolution 21-month x 6-lead ACC gain matrix."""
    pivot_gain = df.pivot(index="date", columns="lead", values="acc_gain")
    pivot_remap = df.pivot(index="date", columns="lead", values="acc_remap")
    pivot_ec = df.pivot(index="date", columns="lead", values="acc_ec")

    dates = pivot_gain.index.tolist()
    leads = [f"Lead {c}" for c in pivot_gain.columns]
    data = pivot_gain.values

    fig, ax = plt.subplots(figsize=(10.5, 9.5), facecolor="white")

    # Diverging colormap centered at 0
    norm = mcolors.TwoSlopeNorm(vmin=-0.35, vcenter=0.0, vmax=0.75)
    cmap = plt.get_cmap("RdBu_r")

    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(leads)))
    ax.set_xticklabels(leads, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(dates)))
    ax.set_yticklabels(dates, fontsize=10.5)

    # In cell text annotation
    for r in range(len(dates)):
        for c in range(len(leads)):
            gain_val = data[r, c]
            remap_val = pivot_remap.values[r, c]
            ec_val = pivot_ec.values[r, c]

            text_color = "white" if abs(gain_val) > 0.40 else "black"
            text_str = f"{gain_val:+.2f}\n({remap_val:+.2f}/{ec_val:+.2f})"
            ax.text(
                c,
                r,
                text_str,
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.5,
                fontweight="normal",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Spatial ACC Gain (\u0394ACC = ReMAP - SEAS5)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    ax.set_title(
        "ReMAP vs. ECMWF SEAS5 Multi-Lead Spatial ACC Gain Matrix (2023-01 to 2024-09)\n"
        "Cell Text: \u0394ACC (ReMAP ACC / SEAS5 ACC)",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )

    fig.tight_layout()
    save_path_native.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path_native, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if save_path_desktop:
        save_path_desktop.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(save_path_native, save_path_desktop)
    print(f"Heatmap saved to: {save_path_native}")


def render_highlight_case_figure(
    case_specs: list[dict],
    save_path_native: Path,
    save_path_desktop: Path,
    obs_all,
    ec_all,
    pred_all,
    dates,
    lats,
    lons,
    valid_mask,
    weights,
):
    """Render selected highlight comparison cases in a 3x3 layout."""
    fields = []
    titles = []
    letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    p_idx = 0

    for spec in case_specs:
        t_idx = spec["t_idx"]
        lead = spec["lead"]
        d_str = dates[t_idx][:7]

        o = obs_all[t_idx, lead]
        e = ec_all[t_idx, lead]
        p = pred_all[t_idx, lead]

        acc_e = float(spatial_acc(e[valid_mask], o[valid_mask], weights))
        acc_p = float(spatial_acc(p[valid_mask], o[valid_mask], weights))
        gain = acc_p - acc_e

        fields.append(o)
        titles.append(f"({letters[p_idx]}) Observation ({d_str})")
        p_idx += 1

        fields.append(e)
        titles.append(f"({letters[p_idx]}) SEAS5 Lead {lead} (ACC={acc_e:+.2f})")
        p_idx += 1

        fields.append(p)
        titles.append(f"({letters[p_idx]}) ReMAP Lead {lead} (ACC={acc_p:+.2f}, \u0394={gain:+.2f})")
        p_idx += 1

    plot_spatial_comparison(
        fields=fields,
        titles=titles,
        latitudes=lats,
        longitudes=lons,
        ncols=3,
        vmaxs=None,
        color_modes=["diverging"] * 9,
        save_path=save_path_native,
        dpi=300,
        shared_colorbar="col",
    )

    if save_path_desktop:
        save_path_desktop.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(save_path_native, save_path_desktop)
    print(f"Highlight figure saved: {save_path_native.name}")


def main():
    paths = SubprojectPaths("U_Net_3D")
    exp_dir = paths.get_exp_dir("final_model")

    native_base = exp_dir / "evaluation" / "figures" / "spatial_leads"
    desktop_base = Path(r"C:\Users\fired\OneDrive\Desktop\文献与汇报\02_自研课题与论文\ReMAP次季节降水预测\figures\动态色标空间对比")

    native_base.mkdir(parents=True, exist_ok=True)
    desktop_base.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HydroSynth U_Net_3D Multi-Lead Full Period Spatial Visualizer")
    print(f"Native Target:  {native_base}")
    print(f"Desktop Target: {desktop_base}")
    print("=" * 70)

    # 1. Load Data
    dates = np.load(exp_dir / "multi_lead_dates.npy").astype(str)
    obs = np.load(exp_dir / "multi_lead_obs_results.npy")
    ec = np.load(exp_dir / "multi_lead_ec_precip_anom_results.npy")
    pred = np.load(exp_dir / "multi_lead_predict_results_final.npy")

    with np.load(OBSERVATION_FILE) as npz:
        valid_mask = np.asarray(npz["valid_mask"], dtype=bool)
        lats = np.asarray(npz["latitudes"], dtype=np.float64)
        lons = np.asarray(npz["longitudes"], dtype=np.float64)

    weights = np.cos(np.deg2rad(lats))[:, None] * np.ones((1, 140))
    weights = weights[valid_mask]
    weights /= weights.mean()

    n_dates = len(dates)
    n_leads = obs.shape[1]

    # 2. Compute Full Metrics Table
    records = []
    for l in range(n_leads):
        for t in range(n_dates):
            o = obs[t, l][valid_mask]
            e = ec[t, l][valid_mask]
            p = pred[t, l][valid_mask]

            acc_e = float(spatial_acc(e, o, weights))
            acc_p = float(spatial_acc(p, o, weights))
            sim_ep = float(spatial_acc(e, p, weights))
            rmse_e = float(np.sqrt(np.mean(weights * (e - o) ** 2)))
            rmse_p = float(np.sqrt(np.mean(weights * (p - o) ** 2)))

            records.append({
                "lead": l,
                "date": dates[t][:7],
                "acc_ec": acc_e,
                "acc_remap": acc_p,
                "acc_gain": acc_p - acc_e,
                "sim_ec_remap": sim_ep,
                "rmse_ec": rmse_e,
                "rmse_remap": rmse_p,
                "rmse_gain": rmse_e - rmse_p,
            })

    df = pd.DataFrame(records)
    csv_native = exp_dir / "evaluation" / "tables" / "monthly_lead_metrics.csv"
    csv_desktop = desktop_base / "month_lead_metric.csv"
    csv_native.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_native, index=False)
    df.to_csv(csv_desktop, index=False)
    print(f"Metrics table exported to: {csv_native} and {csv_desktop}")

    # 3. Macro ACC Gain Heatmap
    heatmap_native = native_base / "acc_gain_heatmap.png"
    heatmap_desktop = desktop_base / "acc_gain_heatmap.png"
    plot_acc_gain_heatmap(df, heatmap_native, heatmap_desktop)

    # 4. Highlight Figures
    # A) Top Breakthrough Cases
    top_cases = [
        {"t_idx": 10, "lead": 1},  # 2023-11 Lead 1 (+0.72)
        {"t_idx": 10, "lead": 3},  # 2023-11 Lead 3 (+0.70)
        {"t_idx": 8, "lead": 4},   # 2023-09 Lead 4 (+0.55)
    ]
    render_highlight_case_figure(
        top_cases,
        native_base / "top_gain_cases.png",
        desktop_base / "top_gain_cases.png",
        obs, ec, pred, dates, lats, lons, valid_mask, weights
    )

    # B) Summer Monsoon / Flood Season Cases
    summer_cases = [
        {"t_idx": 5, "lead": 5},   # 2023-06 Lead 5 (+0.42)
        {"t_idx": 7, "lead": 4},   # 2023-08 Lead 4 (+0.39)
        {"t_idx": 17, "lead": 1},  # 2024-06 Lead 1 (+0.36)
    ]
    render_highlight_case_figure(
        summer_cases,
        native_base / "summer_gain_case.png",
        desktop_base / "summer_gain_case.png",
        obs, ec, pred, dates, lats, lons, valid_mask, weights
    )

    # 5. Define Quarters for 21 months
    quarter_defs = [
        ("2023_Q1", [0, 1, 2]),
        ("2023_Q2", [3, 4, 5]),
        ("2023_Q3", [6, 7, 8]),
        ("2023_Q4", [9, 10, 11]),
        ("2024_Q1", [12, 13, 14]),
        ("2024_Q2", [15, 16, 17]),
        ("2024_Q3", [18, 19, 20]),
    ]

    tasks = []
    for l in range(n_leads):
        for q_name, t_idxs in quarter_defs:
            fname = f"L{l}_{q_name}.png"
            p_native = native_base / fname
            p_desktop = desktop_base / fname
            m_labels = [dates[idx][:7] for idx in t_idxs]
            o_sub = obs[t_idxs, l]
            e_sub = ec[t_idxs, l]
            p_sub = pred[t_idxs, l]

            tasks.append((
                l,
                q_name,
                t_idxs,
                m_labels,
                o_sub,
                e_sub,
                p_sub,
                valid_mask,
                weights,
                lats,
                lons,
                p_native,
                p_desktop,
            ))

    print(f"\nDispatching {len(tasks)} quarter rendering tasks using multiprocessing (max_workers=6)...")
    with ProcessPoolExecutor(max_workers=6) as executor:
        for res in executor.map(render_single_quarter, tasks):
            print(f"  {res}")

    print("\n" + "=" * 70)
    print("All 42 multi-lead quarterly comparison figures + 2 highlight case figures + 1 heatmap generated!")
    print(f"Native location:  {native_base}")
    print(f"Desktop location: {desktop_base}")
    print("=" * 70)


if __name__ == "__main__":
    main()
