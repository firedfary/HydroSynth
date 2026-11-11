# -*- coding: utf-8 -*-
"""
Additional statistics required by Reviewer 2:
  1. TCC stratified by ENSO phase, for hindcast and independent test periods.
  2. Multiple-comparison control on the grid-point significance maps (Benjamini-Hochberg FDR).

Reads predictions_all.npz from the unified evaluation and writes 补充统计.json.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.paths import SubprojectPaths  # noqa: E402


def tcc_p_map(pred, obs, valid_mask):
    T = pred.shape[0]
    H, W = valid_mask.shape
    o = np.asarray(obs, dtype=np.float64).reshape(T, -1)
    p = np.asarray(pred, dtype=np.float64).reshape(T, -1)
    ok = np.isfinite(o) & np.isfinite(p)
    o = np.where(ok, o, 0.0)
    p = np.where(ok, p, 0.0)
    n = ok.sum(axis=0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        om = np.divide(o.sum(axis=0), np.maximum(n, 1.0))
        pm = np.divide(p.sum(axis=0), np.maximum(n, 1.0))
        od = np.where(ok, o - om, 0.0)
        pd_ = np.where(ok, p - pm, 0.0)
        so = np.sqrt(np.sum(od * od, axis=0))
        sp = np.sqrt(np.sum(pd_ * pd_, axis=0))
        denom = so * sp
        r = np.divide(np.sum(od * pd_, axis=0), denom,
                      out=np.full_like(denom, np.nan), where=denom > 0)
        dof = np.maximum(n - 2.0, 1.0)
        rc = np.clip(r, -1.0 + 1e-12, 1.0 - 1e-12)
        tstat = rc * np.sqrt(dof / np.maximum(1.0 - rc ** 2, 1e-300))
        pv = 2.0 * student_t.sf(np.abs(tstat), dof)
    good = valid_mask.reshape(-1) & (n >= 5) & (so / np.sqrt(np.maximum(n, 1.0)) > 1e-6) \
        & (sp / np.sqrt(np.maximum(n, 1.0)) > 1e-6) & np.isfinite(r) & np.isfinite(pv)
    return (np.where(good, r, np.nan).astype(np.float32).reshape(H, W),
            np.where(good, pv, np.nan).astype(np.float32).reshape(H, W))


def bh_fdr(pvals, alpha=0.05):
    flat = np.asarray(pvals, dtype=np.float64).reshape(-1)
    ok = np.isfinite(flat)
    p = flat[ok]
    n = len(p)
    if n == 0:
        return 0, 0.0, n
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresh
    k = int(np.max(np.where(passed)[0]) + 1) if np.any(passed) else 0
    return k, float(k / n * 100.0), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Delivery folder (absolute path)")
    args = ap.parse_args()

    exp_dir = Path(SubprojectPaths(__file__).get_exp_dir("unified_eval"))
    npz_path = exp_dir / "predictions_all.npz"
    if not npz_path.exists():
        raise SystemExit(f"Cannot find {npz_path}; run run_unified_eval.py first.")
    d = np.load(npz_path, allow_pickle=True)

    hind = d["hind_mask"].astype(bool)
    test = d["test_mask"].astype(bool)
    vm = d["valid_land_mask"].astype(bool)
    obs = d["target"]
    pcs = d["pcs5"]
    pc1 = pcs[:, 0]
    labels = np.where(pc1 >= 0.5, "El Nino", np.where(pc1 <= -0.5, "La Nina", "Neutral"))

    preds = {
        "SEAS5": d["pred_seas5"],
        "Baseline": d["pred_base"],
        "EOF-cat": d["pred_cat"],
        "FiLM-U-Net": d["pred_film"],
    }
    n_grids = int(vm.sum())
    result = {"n_valid_grids": n_grids, "enso_tcc": {}, "fdr": {}}

    for period_name, pmask in [("hindcast", hind), ("test", test)]:
        result["enso_tcc"][period_name] = {}
        idx_all = np.where(pmask)[0]
        for phase in ["El Nino", "La Nina", "Neutral", "All"]:
            idx = idx_all if phase == "All" else idx_all[labels[pmask] == phase]
            entry = {"months": int(len(idx))}
            for m, arr in preds.items():
                if len(idx) < 5:
                    entry[m] = None
                    continue
                tmap, pvmap = tcc_p_map(arr[idx], obs[idx], vm)
                entry[m] = {
                    "mean_tcc": float(np.nanmean(tmap[vm].astype(np.float64))),
                    "sig_pct": float(np.sum((pvmap < 0.05) & vm) / n_grids * 100.0),
                }
            result["enso_tcc"][period_name][phase] = entry

    for m in ["Baseline", "EOF-cat", "FiLM-U-Net"]:
        key = {"Baseline": "pval_base", "EOF-cat": "pval_cat", "FiLM-U-Net": "pval_film"}[m]
        pv = d[key]
        k, pct_val, n = bh_fdr(pv[vm], alpha=0.05)
        raw = int(np.sum((pv[vm] < 0.05)))
        result["fdr"][m] = {
            "n_grids": n,
            "raw_p_lt_005": raw,
            "raw_pct": float(raw / n * 100.0),
            "bh_fdr_005_count": k,
            "bh_fdr_005_pct": pct_val,
            "expected_false_positives_at_alpha_005": int(round(n * 0.05)),
        }

    out = Path(args.out) / "结果表" / "补充统计.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)
    print(f"[ExtraStats] wrote {out}")
    for period in ("hindcast", "test"):
        for phase in ("El Nino", "La Nina", "Neutral", "All"):
            e = result["enso_tcc"][period][phase]
            print(f"  {period:9s} {phase:8s} ({e['months']:3d}m): " +
                  " ".join(f"{m}={e[m]['mean_tcc']:.4f}" if e[m] else f"{m}=n/a"
                           for m in preds))
    for m, v in result["fdr"].items():
        print(f"  FDR {m:12s}: raw p<0.05 {v['raw_pct']:.1f}% -> BH-FDR q<0.05 {v['bh_fdr_005_pct']:.1f}%")


if __name__ == "__main__":
    main()
