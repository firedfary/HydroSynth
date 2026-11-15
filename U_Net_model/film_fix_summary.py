"""Pool repeated trainings of one configuration and report the reproducibility floor.

Why
---
`film_fix_eval.py` trains three architectures with seeds 42/43/44 and averages
them, which is the protocol behind every number in the manuscript. Seeds,
batch order, data and hyperparameters are fixed, so rerunning a stage from
scratch should in principle reproduce it exactly. It does not: cuDNN's
convolution backward uses nonatomic reductions, and
`torch.use_deterministic_algorithms` is not enabled, so each run lands on a
slightly different model.

This script quantifies that spread and separates two questions that the
manuscript currently conflates:

1. **What can one training run resolve?** The spread of the ensemble score
   across independent reruns of an identical configuration. Any model-versus-
   model difference smaller than this cannot be claimed from a single run --
   which is what the manuscript, and most of this literature, reports.

2. **What can repeated training resolve?** The mean difference over n reruns,
   whose standard error shrinks as sqrt(n). A difference far below the
   single-run spread can still be real and estimable this way.

These give different answers and both belong in the paper. Reporting only (1)
would overstate the case -- "unresolvable by one run" is not "zero". Reporting
only (2) would hide that no published single-run comparison, including the
earlier versions of this manuscript, had the resolution to see it.

Inputs
------
Every `metrics.json` written by `film_fix_eval.py` under the experiment
directories matched by `--pattern`. A stage is pooled across whichever
experiments contain it; stages present in only one experiment are reported
but flagged, since no spread can be estimated from a single run.

Usage
-----
    python film_fix_summary.py
    python film_fix_summary.py --pattern "film_fix_*" --stage s0_current
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.paths import SubprojectPaths  # noqa: E402

MODELS = {"baseline32": "Baseline", "eof_cat32": "EOF-cat", "film32_identity": "FiLM"}
METRICS = ("mean_acc", "mean_tcc", "rmse", "sc_pct", "csi")
PERIOD = "validation"
# Differences the manuscript drafts asked the reader to interpret.
REPORTED = {
    "revision-06 FiLM minus Baseline": -0.0076,
    "revision-05 FiLM minus EOF-cat": -0.0048,
}


def load(pattern, results_dir):
    """Collect {stage: {experiment: ensemble-metrics}} from every metrics.json."""
    runs = {}
    for path in sorted(results_dir.glob(pattern)):
        metrics = path / "metrics.json"
        if not metrics.is_file():
            continue
        try:
            payload = json.loads(metrics.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  skipping unreadable {metrics}", file=sys.stderr)
            continue
        if "results" not in payload:
            continue
        for stage, block in payload["results"].items():
            period = block.get("periods", {}).get(PERIOD)
            if period is None:
                continue
            runs.setdefault(stage, {})[path.name] = {
                "ensemble": period["ensemble"],
                "training": block.get("training", {}),
            }
    return runs


def t_test(values):
    """Two-sided one-sample t-test against zero, on the t distribution.

    With n of about five the normal approximation is materially wrong: its
    two-sided tail and its 1.96 multiplier both understate the uncertainty, so
    the t distribution with n-1 degrees of freedom is used for the p-value and
    for the interval multiplier reported alongside it. A sign test is included
    because it assumes only exchangeability, not normality, which matters when
    five points cannot establish the shape of the distribution.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return None
    from math import comb
    positives = int((values > 0).sum())
    tail = sum(comb(n, k) for k in range(min(positives, n - positives) + 1))
    sign_p = min(1.0, 2.0 * tail / (2 ** n))
    sd = values.std(ddof=1)
    if sd == 0:
        return {"n": n, "mean": float(values.mean()), "sem": 0.0,
                "t": float("inf"), "p": 0.0, "sign_test_p": sign_p}
    sem = sd / np.sqrt(n)
    t = values.mean() / sem
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df=n - 1))
        crit = float(stats.t.ppf(0.975, df=n - 1))
    except ImportError:  # normal fallback, flagged in the output
        from math import erfc, sqrt
        p, crit = float(erfc(abs(t) / sqrt(2))), 1.96
    return {"n": n, "mean": float(values.mean()), "sd": float(sd), "sem": float(sem),
            "t": float(t), "p": p, "t_crit_95": crit, "sign_test_p": sign_p}


def variance_sources(experiments):
    """Split the FiLM-minus-Baseline spread into its two sources.

    Each run trains the same three fixed seeds, so the per-seed gap indexed by
    (run, seed) separates two things the ensemble does not remove:

    * rerunning an identical configuration with the SAME seed varies only
      through cuDNN's nondeterministic reductions;
    * changing the seed varies the trained model itself.

    The published ensemble averages three specific seeds, 42/43/44. Anyone
    reproducing the work picks their own, so the seed term is the one that
    governs whether an independent group would see the same ranking. Uses each
    seed's best validation ACC, the quantity the checkpoint rule maximizes.
    """
    gaps = {}
    for name, block in experiments.items():
        for seed, models in block.get("training", {}).items():
            if "film32_identity" in models and "baseline32" in models:
                gaps[(name, seed)] = (models["film32_identity"]["best_validation_acc"]
                                      - models["baseline32"]["best_validation_acc"])
    if not gaps:
        return None
    runs = sorted({run for run, _ in gaps})
    seeds = sorted({seed for _, seed in gaps})
    if len(runs) < 2 or len(seeds) < 2:
        return None
    table = np.array([[gaps.get((run, seed), np.nan) for seed in seeds] for run in runs])
    if not np.isfinite(table).all():
        return None
    # Same seed, different run: cuDNN only. Different seed, same run: seed + cuDNN.
    across_runs = float(np.mean([table[:, j].std(ddof=1) for j in range(len(seeds))]))
    across_seeds = float(np.mean([table[i].std(ddof=1) for i in range(len(runs))]))
    seed_only = float(np.sqrt(max(across_seeds ** 2 - across_runs ** 2, 0.0)))
    return {"seeds": seeds, "runs": runs, "per_seed_gap": table.tolist(),
            "sd_across_runs_same_seed": across_runs,
            "sd_across_seeds_same_run": across_seeds,
            "sd_seed_component": seed_only,
            "sd_of_3_seed_ensemble_if_reseeded": seed_only / np.sqrt(len(seeds))}


def summarize(values):
    values = np.asarray(values, dtype=float)
    out = {"n": len(values), "mean": float(values.mean()),
           "min": float(values.min()), "max": float(values.max()),
           "spread": float(values.max() - values.min())}
    if len(values) > 1:
        out["sd"] = float(values.std(ddof=1))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="film_fix_*")
    parser.add_argument("--stage", action="append",
                        help="limit to these stages; repeatable")
    parser.add_argument("--output", default="film_fix_summary",
                        help="experiment directory for the written summary")
    args = parser.parse_args()

    paths = SubprojectPaths(__file__)
    runs = load(args.pattern, paths.results_dir)
    stages = args.stage or sorted(runs)
    missing = [s for s in stages if s not in runs]
    if missing:
        parser.error(f"no runs found for stages: {missing}")

    report = {"period": PERIOD, "pattern": args.pattern, "stages": {}}

    for stage in stages:
        experiments = runs[stage]
        names = sorted(experiments)
        print("=" * 86)
        print(f"{stage}: {len(names)} independent run(s) -- {', '.join(names)}")
        if len(names) == 1:
            print("  only one run; no reproducibility spread can be estimated")

        print(f"\n  per-run ensemble {PERIOD} ACC")
        print("  %-18s" % "run" + "".join("%12s" % label for label in MODELS.values())
              + "%16s" % "FiLM-Baseline")
        gaps, per_model = [], {m: [] for m in MODELS}
        for name in names:
            ens = experiments[name]["ensemble"]
            row = "  %-18s" % name
            for key in MODELS:
                value = ens[key]["mean_acc"]
                per_model[key].append(value)
                row += "%12.4f" % value
            gap = ens["film32_identity"]["mean_acc"] - ens["baseline32"]["mean_acc"]
            gaps.append(gap)
            print(row + "%+16.4f" % gap)

        stage_report = {"runs": names, "per_model_acc": {}, "gap_film_minus_baseline": {}}
        if len(names) > 1:
            print("\n  spread of an identical configuration")
            print("  %-18s%10s%10s%10s%12s" % ("quantity", "mean", "sd", "spread", "range"))
            for key, label in MODELS.items():
                s = summarize(per_model[key])
                stage_report["per_model_acc"][label] = s
                print("  %-18s%10.4f%10.4f%10.4f   [%.4f, %.4f]"
                      % (label + " ACC", s["mean"], s["sd"], s["spread"], s["min"], s["max"]))
            g = summarize(gaps)
            stage_report["gap_film_minus_baseline"]["spread"] = g
            print("  %-18s%+10.4f%10.4f%10.4f   [%+.4f, %+.4f]"
                  % ("FiLM-Baseline", g["mean"], g["sd"], g["spread"], g["min"], g["max"]))

            floor = max(summarize(per_model[k])["spread"] for k in MODELS)
            stage_report["single_run_floor_acc"] = floor
            print(f"\n  SINGLE-RUN FLOOR: an identical configuration moves ensemble ACC "
                  f"by up to {floor:.4f}")
            for label, value in REPORTED.items():
                verdict = "below the floor" if abs(value) < floor else "above the floor"
                print(f"    {label}: {value:+.4f} -- {verdict}")

            test = t_test(gaps)
            if test is not None:
                stage_report["gap_film_minus_baseline"]["mean_test"] = test
                print(f"\n  MEAN OVER {test['n']} RUNS: {test['mean']:+.4f} "
                      f"+/- {test['sem']:.4f} (sem), two-sided t p={test['p']:.4f}, "
                      f"sign-test p={test['sign_test_p']:.4f}")
                crit = test.get("t_crit_95", 1.96)
                lo = test["mean"] - crit * test["sem"]
                hi = test["mean"] + crit * test["sem"]
                stage_report["gap_film_minus_baseline"]["ci95_normal"] = [lo, hi]
                crosses = lo * hi <= 0
                print(f"    approximate 95% interval on the mean gap: [{lo:+.4f}, {hi:+.4f}]"
                      f" -- {'includes' if crosses else 'excludes'} zero")
                print("    Repeated training estimates the mean gap more precisely than any"
                      "\n    single run; a gap under the single-run floor can still be real.")
        sources = variance_sources(experiments)
        if sources is not None:
            stage_report["variance_sources"] = sources
            print("\n  where the spread comes from (per-seed gap, FiLM minus Baseline)")
            print("  %-18s" % "run" + "".join("%12s" % f"seed {s}" for s in sources["seeds"]))
            for name, row in zip(sources["runs"], sources["per_seed_gap"]):
                print("  %-18s" % name + "".join("%+12.4f" % v for v in row))
            print("  %-42s%.4f" % ("same seed, different run (cuDNN only):",
                                   sources["sd_across_runs_same_seed"]))
            print("  %-42s%.4f" % ("different seed, same run:",
                                   sources["sd_across_seeds_same_run"]))
            print("  %-42s%.4f" % ("seed component alone:", sources["sd_seed_component"]))
            reseed = sources["sd_of_3_seed_ensemble_if_reseeded"]
            print("  %-42s%.4f" % ("a 3-seed ensemble on DIFFERENT seeds:", reseed))
            print(f"    The published ensemble fixes seeds 42/43/44. An independent group\n"
                  f"    choosing its own three seeds would see the gap move by about"
                  f" {reseed:.4f},\n    which is the floor that governs external reproducibility.")
        report["stages"][stage] = stage_report
        print()

    exp_dir = Path(paths.get_exp_dir(args.output))
    out_file = exp_dir / "reproducibility_floor.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
