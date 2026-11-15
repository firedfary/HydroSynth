"""Two new figures for revision 07: training variability and output amplitude.

Figure 2 shows what repeating an identical training configuration does to the
reported scores, which is the evidence behind the manuscript's claim that a
single run cannot resolve the differences it reports.

Figure 3 shows that extreme-event CSI in this setting is largely a statement
about output sharpness, and that equalizing amplitude reverses the published
ranking.

Both read only from verified result files; nothing is recomputed here.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.paths import SubprojectPaths

PATHS = SubprojectPaths(__file__)
RESULTS = PATHS.results_dir
REPLICATES = ("film_fix_s1", "film_fix_rep", "film_fix_rep3", "film_fix_rep4", "film_fix_rep5")
STAGE = "s0_current"
MODELS = (("baseline32", "Baseline"), ("eof_cat32", "EOF-cat"), ("film32_identity", "FiLM"))
COLOURS = {"Baseline": "#4C72B0", "EOF-cat": "#DD8452", "FiLM": "#55A868", "SEAS5": "#8C8C8C"}
CHANCE_CSI = 0.122


def load_replicates():
    runs = []
    for name in REPLICATES:
        payload = json.loads((RESULTS / name / "metrics.json").read_text(encoding="utf-8"))
        block = payload["results"][STAGE]
        runs.append({
            "ensemble": block["periods"]["validation"]["ensemble"],
            "training": block["training"],
        })
    return runs


def figure_variability(runs, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.1))
    n = len(runs)
    jitter = np.linspace(-0.17, 0.17, n)

    for position, (key, label) in enumerate(MODELS):
        values = np.array([r["ensemble"][key]["mean_acc"] for r in runs])
        ax1.scatter(position + jitter, values, s=46, zorder=3,
                    color=COLOURS[label], edgecolor="white", linewidth=0.8)
        ax1.hlines(values.mean(), position - 0.30, position + 0.30,
                   color=COLOURS[label], linewidth=2.0, zorder=2)
        ax1.annotate(f"spread\n{values.max() - values.min():.4f}",
                     (position, values.min()), textcoords="offset points",
                     xytext=(0, -30), ha="center", fontsize=8.5, color="#444444")
    ax1.set_xticks(range(len(MODELS)))
    ax1.set_xticklabels([label for _, label in MODELS])
    ax1.set_ylabel("Validation mean spatial ACC")
    ax1.set_title(f"(a) {n} retrainings of an identical configuration", fontsize=10.5)
    ax1.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax1.set_axisbelow(True)
    ax1.margins(y=0.22)

    gaps = np.array([r["ensemble"]["film32_identity"]["mean_acc"]
                     - r["ensemble"]["baseline32"]["mean_acc"] for r in runs])
    per_seed = np.array([[r["training"][s]["film32_identity"]["best_validation_acc"]
                          - r["training"][s]["baseline32"]["best_validation_acc"]
                          for s in ("42", "43", "44")] for r in runs])
    mean, sem = gaps.mean(), gaps.std(ddof=1) / np.sqrt(len(gaps))
    crit = 2.776  # t, 4 degrees of freedom

    for index, seed in enumerate(("42", "43", "44")):
        ax2.scatter(per_seed[:, index], np.full(len(runs), 1.0) + np.linspace(-0.12, 0.12, len(runs)),
                    s=28, alpha=0.75, color="#999999", zorder=2,
                    label="individual seeds" if index == 0 else None)
    ax2.scatter(gaps, np.zeros(len(gaps)), s=58, color=COLOURS["FiLM"], zorder=4,
                edgecolor="white", linewidth=0.8, label="three-seed ensembles")
    ax2.errorbar(mean, -0.75, xerr=crit * sem, fmt="D", color="#333333", capsize=5,
                 markersize=7, zorder=5, label="mean, 95% interval")
    ax2.axvline(0.0, color="#B33A3A", linewidth=1.2, linestyle="--", zorder=1)
    ax2.set_yticks([1.0, 0.0, -0.75])
    ax2.set_yticklabels(["per seed", "per ensemble", "mean"])
    ax2.set_ylim(-1.5, 2.3)
    ax2.set_xlabel("FiLM minus Baseline, validation mean spatial ACC")
    ax2.set_title("(b) The gap, and what resolves it", fontsize=10.5)
    ax2.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax2.set_axisbelow(True)
    # Legend sits in the band above the data so it cannot cover the seed points.
    ax2.legend(fontsize=8, loc="upper center", ncol=3, framealpha=0.0,
               borderpad=0.2, handletextpad=0.4, columnspacing=1.2)
    ax2.annotate(f"mean {mean:+.4f} ± {sem:.4f}", (mean, -0.75),
                 textcoords="offset points", xytext=(0, -22), ha="center", fontsize=8.5)

    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(out_dir / f"Fig2_training_variability.{suffix}", dpi=360,
                    bbox_inches="tight")
    plt.close(fig)
    return {"n": len(gaps), "mean_gap": float(mean), "sem": float(sem),
            "ci95": [float(mean - crit * sem), float(mean + crit * sem)],
            "model_spreads": {label: float(np.ptp([r["ensemble"][key]["mean_acc"] for r in runs]))
                              for key, label in MODELS}}


def figure_amplitude(out_dir):
    ablation = json.loads((RESULTS / "revision06_ablation_amplitude"
                           / "ablation_amplitude.json").read_text(encoding="utf-8"))
    rows = {r["row"]: r for r in ablation["rows"]}
    eval_metrics = json.loads((RESULTS / "revision06_eval_v2" / "metrics.json")
                              .read_text(encoding="utf-8"))["periods"]["test"]["ensemble"]

    with np.load(RESULTS / "revision06_eval_v2" / "test_predictions.npz") as data:
        valid, target = data["valid"], data["target"]
        amplitudes = {}
        for key, label in (("seas5", "SEAS5"), ("baseline", "Baseline"),
                           ("eof_cat", "EOF-cat"), ("film", "FiLM")):
            field = data[key]
            amplitudes[label] = float(np.mean([np.nanstd(field[t][valid])
                                               for t in range(len(field))]))
        obs = float(np.mean([np.nanstd(target[t][valid]) for t in range(len(target))]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.1))
    labels = ["SEAS5", "Baseline", "EOF-cat", "FiLM"]
    ratios = [amplitudes[k] / obs for k in labels]
    ax1.bar(labels, ratios, color=[COLOURS[k] for k in labels], width=0.6, zorder=3)
    ax1.axhline(1.0, color="#B33A3A", linewidth=1.3, linestyle="--", zorder=4)
    ax1.annotate("observed amplitude", (3.42, 1.0), xytext=(0, 5),
                 textcoords="offset points", ha="right", fontsize=8.5, color="#B33A3A")
    for index, value in enumerate(ratios):
        ax1.annotate(f"{value:.2f}", (index, value), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=9)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Predicted / observed monthly spatial SD")
    ax1.set_title("(a) Every system is strongly underdispersive", fontsize=10.5)
    ax1.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax1.set_axisbelow(True)

    names = ("Baseline U-Net", "EOF-cat (full)", "FiLM-U-Net (full)")
    short = ("Baseline", "EOF-cat", "FiLM")
    raw = [rows[n]["csi"] for n in names]
    equalised = [rows[n]["csi_at_obs_amplitude"] for n in names]
    x = np.arange(len(names))
    ax2.bar(x - 0.19, raw, width=0.36, color=[COLOURS[s] for s in short],
            alpha=0.45, zorder=3)
    ax2.bar(x + 0.19, equalised, width=0.36, color=[COLOURS[s] for s in short],
            zorder=3)
    ax2.axhline(CHANCE_CSI, color="#555555", linewidth=1.1, linestyle=":", zorder=4)
    # Place the chance-level label clear of the bars and their value labels.
    ax2.annotate(f"chance level {CHANCE_CSI:.3f}", (-0.46, CHANCE_CSI), xytext=(0, 5),
                 textcoords="offset points", ha="left", fontsize=8.5, color="#555555")
    # Neutral swatches: colour encodes the model, shading encodes the treatment.
    ax2.legend(handles=[Patch(facecolor="#777777", alpha=0.45, label="as reported"),
                        Patch(facecolor="#777777", label="at observed amplitude")],
               fontsize=8, loc="upper left", framealpha=0.9)
    for index in range(len(names)):
        ax2.annotate(f"{raw[index]:.3f}", (index - 0.19, raw[index]), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=8.5)
        ax2.annotate(f"{equalised[index]:.3f}", (index + 0.19, equalised[index]),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short)
    ax2.set_ylabel("Extreme-event CSI")
    ax2.set_ylim(0, 0.33)
    ax2.set_title("(b) Equalising amplitude reverses the ranking", fontsize=10.5)
    ax2.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(out_dir / f"Fig3_amplitude_and_CSI.{suffix}", dpi=360,
                    bbox_inches="tight")
    plt.close(fig)

    raw_order = [short[i] for i in np.argsort(raw)[::-1]]
    eq_order = [short[i] for i in np.argsort(equalised)[::-1]]
    return {"observed_amplitude": obs,
            "amplitude_ratio": {k: amplitudes[k] / obs for k in labels},
            "raw_csi_ranking": raw_order, "equalised_csi_ranking": eq_order,
            "ranking_reverses": raw_order == eq_order[::-1]}


def main():
    delivery = (Path.home() / "OneDrive/Desktop/文献与汇报/02_自研课题与论文/"
                "FiLM调制UNet降水预测/修改1/07_可信度返修/figures")
    delivery.mkdir(parents=True, exist_ok=True)
    native = Path(PATHS.get_exp_dir("revision07_figures")) / "figures"
    native.mkdir(parents=True, exist_ok=True)

    runs = load_replicates()
    for out_dir in (native, delivery):
        stats = figure_variability(runs, out_dir)
        amp = figure_amplitude(out_dir)
    summary = {"training_variability": stats, "amplitude": amp}
    (native.parent / "figure_values.json").write_text(json.dumps(summary, indent=2),
                                                      encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved figures to {native} and {delivery}")


if __name__ == "__main__":
    main()
