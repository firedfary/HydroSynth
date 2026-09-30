"""Scientific Visualization and Table Generator for ReMAP Ablation & Sensitivity Analysis.

Produces:
1. High-resolution multi-panel publication figure:
   `ref/figures/fig7_ablation_and_sensitivity.png` (300 DPI, minimalist light style).
   - Panel (a): Core Architecture Mechanism Ablation (ΔACC relative to Full ReMAP)
   - Panel (b): 11 Physical Factors Contribution Rate Ranking (Atmospheric vs Oceanic)
   - Panel (c): Hyperparameter Sensitivity Sweeps (Recency Half-life & Auxiliary Weight)
   - Panel (d): Convex Ensemble Blend Weight Response Curves across Leads
2. Markdown and LaTeX standard three-line tables:
   - Table 3: Core Mechanism Ablation Comparison
   - Table 4: 11 Physical Factors Multi-Lead Contribution Rate Decomposition
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
FIGURES_DIR = PROJECT_ROOT / "ref" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

from project_paths import paths

RESULTS_DIR = paths.get_exp_dir("ablation_and_sensitivity")
JSON_RESULTS_FILE = RESULTS_DIR / "ablation_and_sensitivity_results.json"


def setup_matplotlib_style():
    """Configure publication-ready minimalist light style."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["SimHei", "Microsoft YaHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "semibold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "legend.frameon": True,
        "legend.edgecolor": "#cccccc",
        "legend.facecolor": "#ffffff",
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 1.0,
        "grid.color": "#ebebeb",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
    })


VARIANT_ENGLISH = {
    "Full_Model": "ReMAP (Full Model)",
    "M0_ECMWF_Baseline": "ECMWF SEAS5 Baseline",
    "A1_wo_MAS": "w/o MAS Transfer",
    "A2_wo_Recency": "w/o Recency Decay",
    "A3_Fixed_Recency_60": "Fixed 5-Yr Half-life",
    "A4_Fixed_Recency_120": "Fixed 10-Yr Half-life",
    "A5_Transfer_Only": "Transfer Engine Only",
    "A6_Stacking_Only": "Stacking Engine Only",
    "A7_wo_SignedLog1p": "w/o Signed-log1p",
    "A8_wo_Amplitude": "w/o Amplitude Rest.",
}


def plot_ablation_and_sensitivity(data: dict, output_path: Path):
    """Generate 4-panel publication figure."""
    setup_matplotlib_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), dpi=300)
    plt.subplots_adjust(hspace=0.28, wspace=0.22, top=0.94, bottom=0.08, left=0.08, right=0.96)

    # --------------------------------------------------------------------------
    # Subplot (a): Core Mechanism Ablation (Macro ΔACC relative to Full ReMAP)
    # --------------------------------------------------------------------------
    ax_a = axes[0, 0]
    ablation_items = data.get("ablation_summary", [])
    
    # Sort or keep logical order: Full Model first, then ablations, then ECMWF Baseline
    names = [VARIANT_ENGLISH.get(item["variant_key"], item["variant_display"]) for item in ablation_items]
    macro_accs = [item["macro_acc"] for item in ablation_items]
    delta_accs = [item["delta_acc"] for item in ablation_items]

    y_pos = np.arange(len(names))[::-1]
    colors = []
    for item in ablation_items:
        key = item["variant_key"]
        if key == "Full_Model":
            colors.append("#1b9e77")  # Teal green for full model
        elif key == "M0_ECMWF_Baseline":
            colors.append("#e7298a")  # Magenta for raw GCM baseline
        elif "wo_MAS" in key or "wo_Recency" in key:
            colors.append("#d95f02")  # Warm orange for core mechanisms
        else:
            colors.append("#7570b3")  # Soft purple for structural ablations

    bars = ax_a.barh(y_pos, delta_accs, color=colors, height=0.62, edgecolor="#333333", linewidth=0.8, alpha=0.9)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(names, fontweight="medium")
    ax_a.axvline(0.0, color="#333333", linestyle="-", linewidth=1.0)
    ax_a.grid(True, axis="x")
    ax_a.set_xlabel("Skill Difference relative to Full ReMAP (Δ Mean ACC)")
    ax_a.set_title("(a) Core Architectural Mechanism Ablation", loc="left")

    # Add text labels on bars with smart inside/outside placement
    for bar, d_acc, m_acc in zip(bars, delta_accs, macro_accs):
        x_val = bar.get_width()
        text_str = f"ACC: {m_acc:.4f} ({d_acc:+.4f})" if d_acc != 0 else f"ACC: {m_acc:.4f} (Ref)"
        if x_val < -0.06:
            # Place inside the bar in bold white
            ax_a.text(x_val + 0.003, bar.get_y() + bar.get_height() / 2, text_str, va="center", ha="left", fontsize=8.2, color="#ffffff", fontweight="bold")
        else:
            offset = -0.003 if x_val < 0 else 0.003
            ha = "right" if x_val < 0 else "left"
            ax_a.text(x_val + offset, bar.get_y() + bar.get_height() / 2, text_str, va="center", ha=ha, fontsize=8.2, color="#222222")

    ax_a.set_xlim(-0.135, 0.035)

    # --------------------------------------------------------------------------
    # Subplot (b): 11 Physical Factors Relative Contribution Rate Ranking
    # --------------------------------------------------------------------------
    ax_b = axes[0, 1]
    factors = data.get("factor_importance", [])
    
    # Sort factors by mean contribution rate descending
    factors_sorted = sorted(factors, key=lambda x: x["mean_contribution_pct"], reverse=True)
    f_names = [f["factor_display"] for f in factors_sorted]
    f_pcts = [f["mean_contribution_pct"] for f in factors_sorted]
    f_types = [f["category"] for f in factors_sorted]

    y_pos_b = np.arange(len(f_names))[::-1]
    
    # Coral for Atmosphere, Steel Blue for Ocean
    cat_colors = {"Atmosphere": "#E26D5C", "Ocean": "#386CB0"}
    f_bar_colors = [cat_colors.get(t, "#888888") for t in f_types]

    bars_b = ax_b.barh(y_pos_b, f_pcts, color=f_bar_colors, height=0.62, edgecolor="#333333", linewidth=0.8, alpha=0.9)
    ax_b.set_yticks(y_pos_b)
    ax_b.set_yticklabels(f_names, fontweight="medium")
    ax_b.grid(True, axis="x")
    ax_b.set_xlabel("Relative Predictive Contribution Rate (%)")
    ax_b.set_title("(b) 11-D Physical Prior Feature Contribution Ranking", loc="left")

    for bar, val in zip(bars_b, f_pcts):
        ax_b.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", ha="left", fontsize=8.5, fontweight="bold", color="#333333")

    ax_b.set_xlim(0, max(f_pcts) * 1.22)
    
    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#E26D5C", edgecolor="#333333", label="Atmospheric Dynamics (6-D)"),
        Patch(facecolor="#386CB0", edgecolor="#333333", label="Oceanic Boundary Forcing (5-D)"),
    ]
    ax_b.legend(handles=legend_elements, loc="lower right", framealpha=0.95)

    # --------------------------------------------------------------------------
    # Subplot (c): Hyperparameter Sensitivity Sweeps (Half-Life & Aux Weight)
    # --------------------------------------------------------------------------
    ax_c = axes[1, 0]
    sens = data.get("sensitivity_sweeps", {})
    
    tau_sweep = sens.get("recency_halflife", {})
    aux_sweep = sens.get("auxiliary_weight", {})

    tau_vals = tau_sweep.get("param_values", [])
    tau_scores = tau_sweep.get("macro_acc", [])
    
    # Map 0 to Inf label for plotting
    x_tau = np.arange(len(tau_vals))
    tau_labels = [str(v) if v > 0 else "∞ (No-decay)" for v in tau_vals]

    color_tau = "#2b83ba"
    color_aux = "#d7191c"

    line1 = ax_c.plot(x_tau, tau_scores, marker="o", color=color_tau, linewidth=2.0, markersize=6, label="Decadal Half-life τ (months)")
    ax_c.set_xticks(x_tau)
    ax_c.set_xticklabels(tau_labels, rotation=25)
    ax_c.set_xlabel("Candidate Half-life Parameter τ (months)")
    ax_c.set_ylabel("Mean ACC (Rolling OOF & Test)", color=color_tau)
    ax_c.tick_params(axis="y", labelcolor=color_tau)
    ax_c.grid(True)
    ax_c.set_title("(c) Parameter Sensitivity: Time Half-Life & Auxiliary Weight", loc="left")

    # Secondary x-axis or twin axis for aux weight
    ax_c2 = ax_c.twiny()
    aux_vals = aux_sweep.get("param_values", [])
    aux_scores = aux_sweep.get("macro_acc", [])
    
    if aux_vals and aux_scores:
        line2 = ax_c2.plot(aux_vals, aux_scores, marker="s", color=color_aux, linewidth=2.0, linestyle="--", markersize=6, label="Aux Weight w_aux (NCEP/JMA)")
        ax_c2.set_xlabel("Auxiliary Model Weight w_aux", color=color_aux)
        ax_c2.tick_params(axis="x", labelcolor=color_aux)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax_c.legend(lines, labels, loc="lower right", framealpha=0.95)

    # --------------------------------------------------------------------------
    # Subplot (d): Dual-Engine Convex Blend Weight (α_L) Response across Leads
    # --------------------------------------------------------------------------
    ax_d = axes[1, 1]
    blend_sweep = sens.get("blend_weights", {})
    alphas = blend_sweep.get("alpha_candidates", np.linspace(0, 1, 21).tolist())
    lead_curves = blend_sweep.get("lead_curves", {})

    lead_palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
    for lead in range(6):
        lead_key = f"Lead_{lead}"
        if lead_key in lead_curves:
            scores = lead_curves[lead_key]
            best_idx = int(np.argmax(scores))
            opt_alpha = alphas[best_idx]
            opt_acc = scores[best_idx]
            ax_d.plot(alphas, scores, color=lead_palette[lead], linewidth=1.6, label=f"Lead {lead} (Opt α={opt_alpha:.2f})")
            ax_d.scatter([opt_alpha], [opt_acc], color=lead_palette[lead], s=35, zorder=5)

    if "Macro_Mean" in lead_curves:
        macro_curve = lead_curves["Macro_Mean"]
        best_m_idx = int(np.argmax(macro_curve))
        ax_d.plot(alphas, macro_curve, color="#111111", linewidth=2.5, linestyle="-.", label=f"Overall Mean (Opt α={alphas[best_m_idx]:.2f})")
        ax_d.scatter([alphas[best_m_idx]], [macro_curve[best_m_idx]], color="#111111", s=50, zorder=6)

    ax_d.set_xlabel("Transfer Engine Convex Weight α_L  [1.0 = Pure Transfer, 0.0 = Pure Stacking]")
    ax_d.set_ylabel("Spatial Pattern ACC")
    ax_d.grid(True)
    ax_d.set_title("(d) Multi-Lead Convex Blend Optimization Response", loc="left")
    ax_d.legend(loc="lower center", ncol=2, framealpha=0.95, fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Successfully generated publication figure: {output_path}")


def generate_tables(data: dict) -> tuple[str, str]:
    """Generate Markdown and LaTeX standard three-line tables for Table 3 and Table 4."""
    # --------------------------------------------------------------------------
    # Table 3: Core Architectural Mechanism Ablation Table
    # --------------------------------------------------------------------------
    ablation_items = data.get("ablation_summary", [])
    
    md_t3 = []
    md_t3.append("**表 3: ReMAP 模型核心机制消融实验结果对比表 (Ablation Study)**\n")
    md_t3.append("| 实验配置与模型变体 | Lead 0 ACC | Lead 1 ACC | Lead 2 ACC | Lead 3 ACC | Lead 4 ACC | Lead 5 ACC | 平均 ACC | 相对基准增益 (ΔACC) | 平均 RMSE |")
    md_t3.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for item in ablation_items:
        name = item["variant_display"]
        l_accs = item["lead_accs"]
        m_acc = item["macro_acc"]
        d_acc = item["delta_acc"]
        m_rmse = item["macro_rmse"]
        
        d_str = "基准 Ref" if d_acc == 0 else f"{d_acc:+.4f}"
        bold_wrap = lambda s: f"**{s}**" if item["variant_key"] == "Full_Model" else s
        
        row = f"| {bold_wrap(name)} | {l_accs[0]:.4f} | {l_accs[1]:.4f} | {l_accs[2]:.4f} | {l_accs[3]:.4f} | {l_accs[4]:.4f} | {l_accs[5]:.4f} | {bold_wrap(f'{m_acc:.4f}')} | {d_str} | {m_rmse:.4f} |"
        md_t3.append(row)

    md_table_3 = "\n".join(md_t3)

    # --------------------------------------------------------------------------
    # Table 4: 11 Physical Factors Multi-Lead Contribution Rate Decomposition
    # --------------------------------------------------------------------------
    factors = data.get("factor_importance", [])
    factors_sorted = sorted(factors, key=lambda x: x["mean_contribution_pct"], reverse=True)

    md_t4 = []
    md_t4.append("**表 4: 11 维物理前兆因子多预见期相对贡献率分解表 (%)**\n")
    md_t4.append("| 因子编号 | 前兆因子名称与动力学涵义 | 物理分类 | Lead 0 | Lead 1 | Lead 2 | Lead 3 | Lead 4 | Lead 5 | 全时段平均 (%) | 主导响应季节 |")
    md_t4.append("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for idx, f in enumerate(factors_sorted, 1):
        name = f["factor_display"]
        cat = "大气动力" if f["category"] == "Atmosphere" else "海洋外强迫"
        l_pcts = f["lead_contributions_pct"]
        m_pct = f["mean_contribution_pct"]
        season = f.get("dominant_season", "夏/冬季")
        
        row = f"| F{idx} | {name} | {cat} | {l_pcts[0]:.1f}% | {l_pcts[1]:.1f}% | {l_pcts[2]:.1f}% | {l_pcts[3]:.1f}% | {l_pcts[4]:.1f}% | {l_pcts[5]:.1f}% | **{m_pct:.2f}%** | {season} |"
        md_t4.append(row)

    md_table_4 = "\n".join(md_t4)

    return md_table_3, md_table_4


def main():
    if not JSON_RESULTS_FILE.exists():
        print(f"Results file not found: {JSON_RESULTS_FILE}")
        print("Please run experiment_ablation_and_sensitivity.py first.")
        return

    with open(JSON_RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Generate Figures
    fig_path = FIGURES_DIR / "fig7_ablation_and_sensitivity.png"
    plot_ablation_and_sensitivity(data, fig_path)

    # 2. Generate Tables
    t3, t4 = generate_tables(data)
    
    # Save markdown tables
    tables_file = RESULTS_DIR / "generated_tables_markdown.txt"
    with open(tables_file, "w", encoding="utf-8") as f:
        f.write(t3 + "\n\n" + t4 + "\n")
    
    print(f"Generated Markdown tables written to: {tables_file}")
    print("\n" + "=" * 80)
    print("Table 3 Preview:")
    print("=" * 80)
    print(t3)
    print("\n" + "=" * 80)
    print("Table 4 Preview:")
    print("=" * 80)
    print(t4)


if __name__ == "__main__":
    main()
