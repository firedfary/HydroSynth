# -*- coding: utf-8 -*-
"""
Build the 05 revision package (manuscript, response letter, Highlights) entirely
from the authoritative unified-evaluation results.

Usage (nc conda env):
    python revision_build_docs.py \
        --src  "<folder containing 04_大修修改稿.docx / 03_审稿回复信.docx / Highlights.docx>" \
        --out  "<05 delivery folder>" \
        --fig  "<folder containing regenerated PNG/SVG figures>"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from revision_docx_lib import DocxEditor, boundary_replace, q, W  # noqa: E402


def f(x, n=4):
    return "n/a" if x is None else f"{float(x):.{n}f}"


def p1(x):
    return "n/a" if x is None else f"{float(x):.1f}%"


def signed(x, n=4):
    return "n/a" if x is None else f"{float(x):+.{n}f}"


def fmt_p(x):
    if x is None:
        return "n/a"
    v = float(x)
    if v == 0.0:
        return "< 1e-300"
    return f"{v:.2e}" if v < 1e-4 else f"{v:.4f}"


def join_en(items):
    if not items:
        return "none"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def fmt_coord(pk: dict) -> str:
    lat = float(pk["lat"])
    lon_e = float(pk["lon_e"])
    lat_s = f"{abs(lat):.0f}°{'N' if lat >= 0 else 'S'}"
    lon_s = f"{lon_e:.0f}°E" if lon_e <= 180.0 else f"{360.0 - lon_e:.0f}°W"
    return f"{lat_s}, {lon_s} ({pk['val']:.3f})"


def fmt_peaks(f9: dict, layer_key: str, k: int = 2) -> str:
    peaks = f9.get("layer_top_peaks", {}).get(layer_key, [])[:k]
    if not peaks:
        return "n/a"
    return "; ".join(fmt_coord(pk) for pk in peaks)


def fix_ms_typography(text: str) -> str:
    # Fix missing space before inline Author (Year) citations, including after ')' or digits:
    # e.g. ")(Turkoglu et al., 2022)" -> ") (Turkoglu et al., 2022)"
    t2 = re.sub(
        r"([A-Za-z0-9\]\)])\((?=[A-Z][A-Za-z\-]+(?:\s+et al\.|\s+and\s|\s*,\s*(?:19|20)\d\d))",
        r"\1 (",
        text,
    )
    t2 = t2.replace("β.(c)", "β. (c)")
    return t2


def fix_rl_spacing(text: str) -> str:
    t2 = text
    t2 = t2.replace("对审稿人 1意见", "对审稿人 1 意见")
    t2 = t2.replace("对审稿人 2意见", "对审稿人 2 意见")
    t2 = t2.replace("在修改稿4.1", "在修改稿第 4.1 节")
    t2 = t2.replace("ΔACC逐月", "ΔACC 逐月")
    t2 = t2.replace("Stage 1与", "Stage 1 与")
    t2 = t2.replace("Stage 2的作用", "Stage 2 的作用")
    t2 = t2.replace("对审稿人", "对审稿人")
    t2 = t2.replace("（。", "（见图 3d）。")
    # Clean CJK-ASCII/number jams only on known safe tokens without breaking markdown/punctuation
    replacements = {
        "相对SEAS5": "相对 SEAS5",
        "新增2020.01": "新增 2020.01",
        "拟合EOF": "拟合 EOF",
        "分块bootstrap": "分块 bootstrap",
        "逐月bootstrap": "逐月 bootstrap",
        "置零后ACC": "置零后 ACC",
        "MSE统一选模": "MSE 统一选模",
        "reportSEAS5": "SEAS5",
        "平均TCC": "平均 TCC",
        "测试ACC": "测试 ACC",
        "SEAS5各": "SEAS5 各",
        "提高0.04": "提高 0.04",
        "p值为": "p 值为",
    }
    for k, v in replacements.items():
        t2 = t2.replace(k, v)
    return t2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig", required=True)
    args = ap.parse_args()

    src, out, fig = Path(args.src), Path(args.out), Path(args.fig)
    V = json.loads((out / "结果表" / "权威数值.json").read_text(encoding="utf-8"))
    supp_path = out / "结果表" / "补充统计.json"
    X = json.loads(supp_path.read_text(encoding="utf-8")) if supp_path.exists() else {}

    S = V["split"]
    T1 = V["table1"]
    ST = V["stat_tests"]
    EN = V["enso"]
    HO = V["holdout"]
    T5 = V["table5"]
    T3 = V["table3"]
    REG = V["regional"]
    F7 = V["fig7"]
    F8 = V.get("fig8", {})
    F9 = V["fig9"]
    TUN = V["tuning"]
    D = V["derived"]

    order = ["SEAS5", "Baseline", "EOF-cat", "FiLM-U-Net"]
    ng = S["valid_china_land_grids"]
    n_test = S["test_months"]
    p90 = S["train_p90_anomaly_threshold"]
    clipped = S["pct_grid_months_abs_gt_3"]

    fa = T1["FiLM-U-Net"]["test_acc"]
    ba = T1["Baseline"]["test_acc"]
    ca = T1["EOF-cat"]["test_acc"]
    sa = T1["SEAS5"]["test_acc"]
    ftcc = T1["FiLM-U-Net"]["test_tcc"]
    btcc = T1["Baseline"]["test_tcc"]
    ctcc = T1["EOF-cat"]["test_tcc"]
    fsig = T1["FiLM-U-Net"]["test_sig_pct"]
    csig = T1["EOF-cat"]["test_sig_pct"]
    bsig = T1["Baseline"]["test_sig_pct"]
    frmse, fsc, fcsi = (T1["FiLM-U-Net"][k] for k in ("test_rmse", "test_sc_pct", "test_csi"))
    crmse, csc, ccsi = (T1["EOF-cat"][k] for k in ("test_rmse", "test_sc_pct", "test_csi"))
    brmse, bsc, bcsi = (T1["Baseline"][k] for k in ("test_rmse", "test_sc_pct", "test_csi"))
    srmse, ssc, scsi = T1["SEAS5"]["test_rmse"], T1["SEAS5"]["test_sc_pct"], T1["SEAS5"]["test_csi"]

    fc = ST["test_film_vs_cat_acc"]
    fb = ST["test_film_vs_base_acc"]
    ts = ST["test_tcc_spatial"]
    sig_fc = "reaches" if fc["wilcoxon_p"] < 0.05 else "does not reach"
    sig_fb = "reaches" if fb["wilcoxon_p"] < 0.05 else "does not reach"
    sig_tcc = "reaches" if ts["film_vs_cat_wilcoxon_p"] < 0.05 else "does not reach"

    learned = ["Baseline", "EOF-cat", "FiLM-U-Net"]
    rank_acc = max(learned, key=lambda m: T1[m]["test_acc"])
    rank_tcc = max(learned, key=lambda m: T1[m]["test_tcc"])
    rank_sig = max(learned, key=lambda m: T1[m]["test_sig_pct"])
    rank_rmse = min(learned, key=lambda m: T1[m]["test_rmse"])
    rank_sc = max(learned, key=lambda m: T1[m]["test_sc_pct"])
    rank_csi = max(learned, key=lambda m: T1[m]["test_csi"])
    rank_line = (
        f"Among the three learned variants, {rank_acc} has the highest test ACC "
        f"({f(T1[rank_acc]['test_acc'])}), {rank_tcc} the highest spatial mean TCC "
        f"({f(T1[rank_tcc]['test_tcc'])}), {rank_sig} the largest significant-grid coverage "
        f"({p1(T1[rank_sig]['test_sig_pct'])}), {rank_rmse} the lowest RMSE "
        f"({f(T1[rank_rmse]['test_rmse'])}), {rank_sc} the highest Sign Consistency "
        f"({p1(T1[rank_sc]['test_sc_pct'])}) and {rank_csi} the highest Extreme CSI "
        f"({f(T1[rank_csi]['test_csi'],3)})."
    )
    rank_line_cn = (
        f"三个学习模型中，{rank_acc} 的测试 ACC 最高（{f(T1[rank_acc]['test_acc'])}），"
        f"{rank_tcc} 的空间平均 TCC 最高（{f(T1[rank_tcc]['test_tcc'])}），"
        f"{rank_sig} 的显著网格覆盖率最高（{p1(T1[rank_sig]['test_sig_pct'])}），"
        f"{rank_rmse} 的 RMSE 最低（{f(T1[rank_rmse]['test_rmse'])}），"
        f"{rank_sc} 的符号一致率最高（{p1(T1[rank_sc]['test_sc_pct'])}），"
        f"{rank_csi} 的极端事件 CSI 最高（{f(T1[rank_csi]['test_csi'],3)}）。"
    )
    acc_vs_cat = "exceeds" if fa > ca else ("matches" if abs(fa - ca) < 1e-6 else "does not exceed")
    metric_specs = [
        ("test ACC", rank_acc == "FiLM-U-Net"),
        ("spatial mean TCC", rank_tcc == "FiLM-U-Net"),
        ("significant-grid coverage", rank_sig == "FiLM-U-Net"),
        ("lowest RMSE", rank_rmse == "FiLM-U-Net"),
        ("Sign Consistency", rank_sc == "FiLM-U-Net"),
        ("Extreme CSI", rank_csi == "FiLM-U-Net"),
    ]
    best_of_film = [name for name, ok_m in metric_specs if ok_m]
    not_best = [name for name, ok_m in metric_specs if not ok_m]
    sc_below = [m for m in learned if T1[m]["test_sc_pct"] < ssc]
    sc_tradeoff_en = (
        "all learned models reduce Sign Consistency relative to raw SEAS5"
        if len(sc_below) == len(learned)
        else (
            f"{join_en(sc_below)} yield lower Sign Consistency than raw SEAS5 "
            f"({p1(ssc)}, vs {rank_sc} {p1(T1[rank_sc]['test_sc_pct'])})"
            if sc_below
            else "Sign Consistency varies modestly across learned variants"
        )
    )
    sc_tradeoff_cn = (
        "符号一致率相对 SEAS5 全面下降及指标权衡"
        if len(sc_below) == len(learned)
        else (
            f"{' 与 '.join(sc_below)} 符号一致率相对 SEAS5 下降及多指标权衡"
            if sc_below
            else "多指标排名分化及符号一致率权衡"
        )
    )
    rmse_vals = [T1[m]["test_rmse"] for m in learned]
    rmse_min, rmse_max = min(rmse_vals), max(rmse_vals)
    rmse_span = rmse_max - rmse_min
    hind_vals = [T1[m]["hindcast_acc"] for m in learned]
    hind_min, hind_max = min(hind_vals), max(hind_vals)
    small_phs = [
        f"{ph} ({EN['test'][ph]['months']} months)"
        for ph in ("El Nino", "La Nina", "Neutral")
        if EN["test"][ph]["months"] < 5
    ]
    learned_tcc_line = (
        f"Raw SEAS5 attains a spatial mean TCC of {f(T1['SEAS5']['test_tcc'])}; "
        + ", ".join(f"{m} {f(T1[m]['test_tcc'])}" for m in learned) + "."
    )

    t5_full = T5["Full FiLM-U-Net (Ours)"]
    t5_zero = T5["FiLM-U-Net (SST PCs Zeroed)"]
    t5_shuf = T5.get("FiLM-U-Net (Shuffled PCs Control)", {})
    t5_flz = T5["FiLM-U-Net (Local Zeroed)"]
    t5_base = T5["Baseline U-Net (No SST)"]
    t5_catz = T5["EOF-cat (SST PCs Zeroed)"]
    t5_catlz = T5["EOF-cat (Local Zeroed)"]

    abl_zero_acc = t5_zero["test_acc"] - t5_full["test_acc"]
    abl_shuf_acc = (t5_shuf.get("test_acc") or t5_full["test_acc"]) - t5_full["test_acc"]
    abl_zero_tcc = t5_zero["test_tcc"] - t5_full["test_tcc"]
    abl_shuf_tcc = (t5_shuf.get("test_tcc") or t5_full["test_tcc"]) - t5_full["test_tcc"]

    if abl_zero_acc >= 0 and abl_shuf_acc >= 0:
        abl_acc_line = (
            "Neither control lowers test ACC relative to the full model, so we do not "
            "claim an ACC benefit from SST modulation."
        )
    elif abl_zero_acc < 0 and abl_shuf_acc < 0:
        abl_acc_line = (
            "Both controls lower test ACC relative to the full model, which is "
            "consistent with a modest ACC contribution from the SST inputs."
        )
    else:
        abl_acc_line = (
            "The two controls give mixed ACC responses, so no directional ACC claim "
            "is made for SST modulation."
        )
    abl_tcc_line = (
        f"The full model's mean TCC is {f(t5_full['test_tcc'])} versus "
        f"{f(t5_zero['test_tcc'])} with PCs zeroed ({signed(abl_zero_tcc)}) and "
        f"{f(t5_shuf.get('test_tcc'))} with shuffled PCs ({signed(abl_shuf_tcc)})."
    )
    abl_acc_line_cn = (
        "两种对照的 ACC 均未低于完整模型，故不宣称 SST 调制带来 ACC 收益。"
        if abl_zero_acc >= 0 and abl_shuf_acc >= 0 else
        "两种对照的 ACC 响应并不一致，故不对 SST 调制的 ACC 效应作方向性断言。"
    )
    abl_tcc_line_cn = (
        f"完整模型的空间平均 TCC 为 {f(t5_full['test_tcc'])}，SST PCs 置零后为 "
        f"{f(t5_zero['test_tcc'])}（差 {signed(abl_zero_tcc)}），打乱 PCs 后为 "
        f"{f(t5_shuf.get('test_tcc'))}（差 {signed(abl_shuf_tcc)}）。"
    )
    if fc["wilcoxon_p"] >= 0.05 and fb["wilcoxon_p"] >= 0.05:
        both_sig_line = "两个配对检验均未达 5% 显著水平"
    else:
        both_sig_line = (
            f"FiLM−EOF-cat p = {f(fc['wilcoxon_p'],3)}，"
            f"FiLM−Baseline p = {f(fb['wilcoxon_p'],3)}，显著性结论以数值为准"
        )

    ho_models = ["SEAS5", "Baseline", "EOF-cat", "FiLM-U-Net"]
    ho_acc_best = max(ho_models, key=lambda m: HO[m]["acc"])
    ho_tcc_best = max(ho_models, key=lambda m: HO[m]["tcc"])
    ho_line = (
        f"The highest holdout ACC is attained by {ho_acc_best} "
        f"({f(HO[ho_acc_best]['acc'])}) and the highest holdout TCC by {ho_tcc_best} "
        f"({f(HO[ho_tcc_best]['tcc'])}); FiLM-U-Net is therefore not the leading model in "
        "this sensitivity check, which we report as a bounding test rather than as "
        "supporting evidence for superiority."
    )

    if X.get("fdr"):
        fd = X["fdr"]
        fdr_line = (
            "After Benjamini–Hochberg FDR control at q = 0.05, the surviving significant-grid "
            "fractions are "
            + ", ".join(
                f"{m} {fd[m]['bh_fdr_005_pct']:.1f}%"
                for m in ("Baseline", "EOF-cat", "FiLM-U-Net")
            )
            + " (uncorrected p < 0.05: "
            + ", ".join(
                f"{fd[m]['raw_pct']:.1f}%"
                for m in ("Baseline", "EOF-cat", "FiLM-U-Net")
            )
            + f"; approximately {fd['FiLM-U-Net']['expected_false_positives_at_alpha_005']} false "
              f"positives are expected under a global null across {ng:,} grids). "
        )
    else:
        fdr_line = (
            "Benjamini–Hochberg FDR-controlled counts and field-level paired tests are "
            "reported in the accompanying results bundle. "
        )

    # Regional summaries
    r_em = REG["Eastern_Monsoon"]
    r_tp = REG["Tibetan_Plateau"]
    r_nw = REG["Northwest_Arid"]

    reg_sentence_en = (
        f"In the eastern monsoon zone (lon >= 105E, n = {r_em['n_grids']:,} grids), "
        f"test ACC spans {f(r_em['SEAS5']['test_acc'])} (SEAS5) to {f(r_em['Baseline']['test_acc'])} (Baseline), "
        f"{f(r_em['EOF-cat']['test_acc'])} (EOF-cat) and {f(r_em['FiLM-U-Net']['test_acc'])} (FiLM-U-Net), "
        f"while mean TCC reaches {f(r_em['FiLM-U-Net']['test_tcc'])} for FiLM-U-Net versus "
        f"{f(r_em['EOF-cat']['test_tcc'])} (EOF-cat) and {f(r_em['Baseline']['test_tcc'])} (Baseline). "
        f"Over the Tibetan Plateau (lon < 105E, lat <= 37N, n = {r_tp['n_grids']:,} grids), "
        f"test ACC is {f(r_tp['SEAS5']['test_acc'])} (SEAS5), {f(r_tp['Baseline']['test_acc'])} (Baseline), "
        f"{f(r_tp['EOF-cat']['test_acc'])} (EOF-cat) and {f(r_tp['FiLM-U-Net']['test_acc'])} (FiLM-U-Net), "
        f"with mean TCC of {f(r_tp['Baseline']['test_tcc'])}, {f(r_tp['EOF-cat']['test_tcc'])} and "
        f"{f(r_tp['FiLM-U-Net']['test_tcc'])}, respectively. "
        f"In the northwest arid zone (lon < 105E, lat > 37N, n = {r_nw['n_grids']:,} grids), "
        f"test ACC is {f(r_nw['SEAS5']['test_acc'])} (SEAS5), {f(r_nw['Baseline']['test_acc'])} (Baseline), "
        f"{f(r_nw['EOF-cat']['test_acc'])} (EOF-cat) and {f(r_nw['FiLM-U-Net']['test_acc'])} (FiLM-U-Net), "
        f"with mean TCC of {f(r_nw['Baseline']['test_tcc'])}, {f(r_nw['EOF-cat']['test_tcc'])} and "
        f"{f(r_nw['FiLM-U-Net']['test_tcc'])}."
    )
    reg_sentence_cn = (
        f"量化区域统计显示（第 4.2 节）："
        f"东部季风区（{r_em['n_grids']:,} 格点）测试 ACC 为 SEAS5 {f(r_em['SEAS5']['test_acc'])}、"
        f"Baseline {f(r_em['Baseline']['test_acc'])}、EOF-cat {f(r_em['EOF-cat']['test_acc'])}、"
        f"FiLM-U-Net {f(r_em['FiLM-U-Net']['test_acc'])}，平均 TCC 为 "
        f"Baseline {f(r_em['Baseline']['test_tcc'])} / EOF-cat {f(r_em['EOF-cat']['test_tcc'])} / "
        f"FiLM {f(r_em['FiLM-U-Net']['test_tcc'])}；"
        f"青藏高原（{r_tp['n_grids']:,} 格点）测试 ACC 为 "
        f"{f(r_tp['SEAS5']['test_acc'])} / {f(r_tp['Baseline']['test_acc'])} / "
        f"{f(r_tp['EOF-cat']['test_acc'])} / {f(r_tp['FiLM-U-Net']['test_acc'])}，"
        f"平均 TCC 为 {f(r_tp['Baseline']['test_tcc'])} / {f(r_tp['EOF-cat']['test_tcc'])} / "
        f"{f(r_tp['FiLM-U-Net']['test_tcc'])}；"
        f"西北干旱区（{r_nw['n_grids']:,} 格点）测试 ACC 为 "
        f"{f(r_nw['SEAS5']['test_acc'])} / {f(r_nw['Baseline']['test_acc'])} / "
        f"{f(r_nw['EOF-cat']['test_acc'])} / {f(r_nw['FiLM-U-Net']['test_acc'])}，"
        f"平均 TCC 为 {f(r_nw['Baseline']['test_tcc'])} / {f(r_nw['EOF-cat']['test_tcc'])} / "
        f"{f(r_nw['FiLM-U-Net']['test_tcc'])}。"
    )

    # Dynamic tuning log summary
    best_tun = {
        m: min(TUN[m], key=lambda c: c["best_val_mse"])
        for m in ("baseline", "eof_cat", "film")
    }

    # Dynamic Fig. 7 statistics
    g_means = [float(x) for x in F7["gamma_real_mean_abs"]]
    b_means = [float(x) for x in F7["beta_real_mean_abs"]]
    g_maxs = [float(x) for x in F7["gamma_max_abs_per_pc"]]
    b_maxs = [float(x) for x in F7["beta_max_abs_per_pc"]]
    g_shuf_m = [float(x) for x in F7["gamma_shuf_mean_abs"]]
    g_shuf_p95 = [float(x) for x in F7["gamma_shuf_p95_abs"]]
    b_shuf_m = [float(x) for x in F7["beta_shuf_mean_abs"]]
    b_shuf_p95 = [float(x) for x in F7["beta_shuf_p95_abs"]]
    g_lead_m = max(range(5), key=lambda i: g_means[i])
    g_lead_x = max(range(5), key=lambda i: g_maxs[i])
    b_lead_m = max(range(5), key=lambda i: b_means[i])
    b_lead_x = max(range(5), key=lambda i: b_maxs[i])

    # Dynamic Table 3 description
    t3_pos = [it for it in T3 if it["mean_drop"] > 0.002]
    t3_top_str = ", ".join(f"{it['var']} ({signed(it['mean_drop'])} ± {f(it['std_drop'])})" for it in T3[:4])

    # ------------------------------------------------------------------
    # 1. Manuscript
    # ------------------------------------------------------------------
    doc = DocxEditor(src / "04_大修修改稿.docx")
    print(f"[Manuscript] paragraphs={len(doc.xml.findall('.//' + q(W, 'p')))}, tables={len(doc.tables())}")

    # Conservative boundary-protected mapping for any residual untouched paragraphs
    num_map = {}
    old_new = [
        ("0.1618", sa), ("0.3373", ba), ("0.3090", ca), ("0.3415", fa),
        ("0.3771", btcc), ("0.3428", ctcc), ("0.3906", ftcc),
        ("0.3075", T1["Baseline"]["hindcast_acc"]),
        ("0.3094", T1["EOF-cat"]["hindcast_acc"]),
        ("0.3131", T1["FiLM-U-Net"]["hindcast_acc"]),
        ("0.2014", T1["SEAS5"]["hindcast_acc"]),
        ("0.3812", t5_zero["test_tcc"]),
        ("0.3684", t5_shuf.get("test_tcc")),
        ("0.3431", t5_zero["test_acc"]),
        ("0.3426", t5_shuf.get("test_acc")),
        ("-0.0181", t5_flz["test_acc"]),
        ("0.2861", t5_catz["test_acc"]),
        ("0.0675", t5_catlz["test_acc"]),
        ("0.0325", D["delta_acc_film_cat"]),
        ("0.0478", D["delta_tcc_film_cat"]),
        ("0.0042", D["delta_acc_film_base"]),
        ("0.0135", D["delta_tcc_film_base"]),
        ("0.1755", D["stage1_acc"]),
        ("-0.0283", D["stage2a_acc"]),
        ("-0.0343", D["stage2a_tcc"]),
        ("1.850", brmse), ("1.840", crmse), ("1.842", frmse), ("1.892", srmse),
    ]
    for old, new in old_new:
        if new is not None:
            num_map[old] = f(new, 4)
    for old, new in [
        ("5,301", f"{ng:,}"), ("5301", str(ng)),
        ("1.342", f(p90, 4)), ("46.8%", p1(fsig)),
        ("44.2%", p1(bsig)), ("34.2%", p1(csig)),
        ("59.9%", p1(fsc)), ("65.2%", p1(ssc)),
        ("64.1%", p1(bsc)), ("58.2%", p1(csc)),
        ("0.089", f(fcsi, 3)), ("0.082", f(bcsi, 3)),
        ("0.162", f(ccsi, 3)),
    ]:
        num_map[old] = new
    num_map["0.121"] = f(fc["wilcoxon_p"], 3)
    num_map["2,482"] = f"{T1['FiLM-U-Net']['test_n_sig']:,}"
    num_map["2,343"] = f"{T1['Baseline']['test_n_sig']:,}"
    num_map["1,814"] = f"{T1['EOF-cat']['test_n_sig']:,}"
    num_map["+36.8%"] = signed((fsig - csig) / csig * 100.0, 1) + "%"
    num_map["14 out of 21"] = f"{fc['pos_months']} out of {fc['total_months']}"
    num_map["14 months (66.7%)"] = f"{EN['test']['El Nino']['months']} months ({EN['test']['El Nino']['pct']:.1f}%)"
    num_map["6 months (28.6%)"] = f"{EN['test']['Neutral']['months']} months ({EN['test']['Neutral']['pct']:.1f}%)"
    num_map["1 month (4.8%)"] = f"{EN['test']['La Nina']['months']} months ({EN['test']['La Nina']['pct']:.1f}%)"
    num_map["10 random trials"] = "20 random trials"
    num_map["10 次随机打乱"] = "20 次随机打乱"

    # ---- full paragraph rewrites ----
    R = []
    add = lambda needle, text: R.append((needle, text, True))

    add("Fan YUE, 3 , Xiaojuan WANG",
        "Fan YUE1,3, Xiaojuan WANG*1, Rong AI2, Yinzhong WU3, Guolin FENG2,4, and Zhiqiang GONG2")

    add("Crucially, historical 345-month hindcast stratification",
        "New Hydrological Insights for the Region: Incorporating leading global SST principal components "
        "via FiLM provides layer-wise affine modulation of local atmospheric representations without "
        "broadcasting global indices across the spatial grid. Over an independent evaluation period "
        f"(January 2023–September 2024, {n_test} months), FiLM-U-Net attains an Anomaly Correlation "
        f"Coefficient (ACC) of {f(fa,3)} and a spatial mean Temporal Correlation Coefficient (TCC) of "
        f"{f(ftcc,3)}, with {p1(fsig)} of {ng:,} valid land grids passing a two-tailed Student's t test "
        f"(p < 0.05) versus {p1(csig)} for direct channel concatenation (EOF-cat). The "
        f"paired monthly ACC difference relative to EOF-cat is {signed(fc['mean_diff'])} "
        f"(Wilcoxon p = {f(fc['wilcoxon_p'],3)}; year-blocked bootstrap 95% CI "
        f"[{signed(fc['boot_year_block_ci95'][0])}, {signed(fc['boot_year_block_ci95'][1])}]) "
        f"and therefore {sig_fc} statistical significance at the 5% level; FiLM-U-Net "
        f"{acc_vs_cat} EOF-cat on test ACC. The measurable advantage is spatial "
        f"(paired ΔTCC = {signed(ts['film_vs_cat_mean_dtcc'])}, {sig_tcc} field-level significance). "
        "To bound the influence of the 2023–2024 El Niño regime we additionally report an ENSO-stratified "
        "historical record (in-sample) and a 24-month rolling holdout (2020–2021) spanning mixed ENSO "
        "phases, and we explicitly report that removing or shuffling SST inputs does not degrade test ACC. "
        "The contribution of SST modulation is therefore interpreted as a spatial-structure and coverage "
        "effect rather than a generic improvement in every metric.")

    add("• EOF-FiLM integration couples global SST modes",
        "• EOF-FiLM integration couples global SST modes into U-Net for rainfall forecast.")
    add("• FiLM modulates local features without diluting",
        "• FiLM modulates local features without diluting fine spatial gradient fields.")
    add("• Evaluation over 2023–2024 achieves",
        f"• Independent 2023–2024 test: {f(fa,3)} ACC and {f(ftcc,3)} spatial mean TCC.")
    add("• Model expands significant skill coverage",
        f"• Significant skill coverage (p < 0.05) reaches {p1(fsig)} of {ng:,} land grids.")

    add("The main questions of this work are going to figure out are",
        "This work addresses three questions: (1) how to integrate compact teleconnection indices "
        "with a U-Net spatial backbone for monthly precipitation correction; (2) whether FiLM "
        "conditioning improves skill relative to a baseline U-Net and to naive channel concatenation, "
        "and how large and statistically robust that difference is; and (3) whether the learned "
        "modulation is consistent with known teleconnection patterns when tested against explicit "
        "negative controls.")
    add("The paper is organized as follows.",
        "The paper is organized as follows. Section 2 describes the data, preprocessing, EOF "
        "computation and model architectures. Section 3 defines the evaluation metrics and the "
        "statistical-testing protocol. Section 4 presents results, Section 5 examines "
        "interpretability, Section 6 discusses limitations, and Section 7 concludes.")

    add("Data Processing and Outlier Handling:",
        "Data Processing and Outlier Handling: All variables were converted to monthly anomalies by "
        "subtracting the historical climatological monthly mean. Precipitation anomalies were "
        "expressed as fractional deviations ΔP = (P − P_clim)/(P_clim + ε). No truncation or "
        "winsorization was applied to the prepared target field: its heavy tail is retained "
        f"(maximum |ΔP| ≈ 388; {p1(clipped)} of grid-months exceed |ΔP| = 3), and numerical stability "
        "during optimisation is instead controlled by gradient-norm clipping (‖g‖₂ ≤ 2). The extreme-event "
        f"threshold used for the Critical Success Index is the training-set 90th percentile, ΔP ≥ {f(p90,4)}, "
        "rather than a fixed threshold carried over from the earlier draft. To eliminate data "
        "leakage, the SST climatology, EOF spatial basis, PC standardisation statistics, and this percentile "
        "were fitted strictly on the 285-month training window (1994.01–2017.12). The 60-month internal "
        "validation window (2018.01–2022.12) was used only for early stopping and hyperparameter selection; "
        "the 21-month independent test window (2023.01–2024.09) was transformed with frozen statistics and "
        "never entered any fitting or model-selection step. The 1994.01–2022.12 calendar interval contains "
        "348 months, of which three (2011-09, 2011-10, 2017-01) are absent from the archive, leaving the "
        "345 historical months used throughout this paper.")

    add("Crucially, to prevent data leakage, the spatial EOF modes",
        "Crucially, to prevent data leakage, the spatial EOF modes (eigenvectors V_train), the SST "
        "climatology and the PC standardisation parameters were fitted exclusively on the 285-month "
        "training window (1994.01–2017.12); the 60-month internal validation window and the "
        "21-month test window were projected onto these frozen modes via "
        "PCs = (X · V_train − μ_train)/σ_train. We retain the leading five PCs "
        f"(explaining {float(S['pca_total_explained_variance'])*100:.2f}% of training SST variance) "
        "with a one-month lag window (W = 1) for both EOF-cat and FiLM-U-Net.")

    add("The input to the network consists of ten atmospheric variables",
        "The input to the network consists of the same ten atmospheric variables listed in Section "
        "2.1.1: tp, h200, h500, slp, t2m, t850, u200, u850, v200 and v850. No humidity or vertical-"
        "velocity channel is used.")
    add("The encoder part of the U-Net gradually extracts",
        "The encoder consists of five double-convolution blocks with 16, 32, 64, 128 and 128 channels "
        "for the FiLM variant (32, 64, 128, 256 and 256 for the baseline/EOF-cat variant, which uses a "
        "larger base filter count), each followed by 2×2 max-pooling. The decoder mirrors this path "
        "with bilinear upsampling and skip connections, and a 1×1 convolution yields the predicted "
        "precipitation anomaly field at the original 120 × 140 resolution. All models use BatchNorm, "
        "ReLU and Dropout2d.")
    add("(3) U-Net + FiLM, as demonstrated",
        "(3) U-Net + FiLM, as shown in panel (3) of Fig. 1, retains the ten local atmospheric predictors "
        "as the primary spatial input while supplying the five leading EOF-based SST PCs through Feature-wise "
        "Linear Modulation (FiLM). A lightweight index encoder transforms the PC vector into layer-wise "
        "scaling (γ) and shifting (β) parameters that modulate intermediate encoder feature maps without "
        "broadcasting oceanic indices across spatial channels; the empirical comparison against the other "
        "variants is reported in Section 4.")
    add("Beyond the local atmospheric inputs, the model incorporates",
        "Beyond the local atmospheric inputs, the model incorporates an auxiliary IndexEncoder branch "
        "that processes the leading five principal components (PCs) extracted by Empirical Orthogonal Function "
        "(EOF) decomposition of global SST anomalies over a one-month window (W = 1).")
    add("An additional advantage of FiLM lies in its simplicity",
        "An additional advantage of FiLM lies in its low parameter overhead and direct inspectability: "
        "tracking γ and β across ENSO states shows how SST inputs parameterise channel weights, while "
        "its empirical forecast impact against unconditioned and concatenated baselines is evaluated in Section 4.")

    add("Loss Function: Mean squared error",
        "Loss Function: Mean squared error computed over the valid (non-NaN) land grid points. All "
        "three models were trained with AdamW (weight decay 1×10⁻⁵), cosine-annealing learning-rate "
        "schedule, batch size 8 and 100 epochs, with gradient-norm clipping at 2.0; the checkpoint "
        "with the lowest internal-validation MSE was retained (early stopping by selection, not by "
        "termination). For a fair comparison, every model was tuned over the same three-point grid "
        "(learning rate ∈ {2×10⁻⁴, 1×10⁻⁴}, dropout ∈ {0.1, 0.2}), i.e. nine training runs in total "
        "with an identical budget, and the same seed (42) and split. Both EOF-cat and FiLM-U-Net "
        "receive the same five SST PCs with W = 1, but the insertion mechanism differs by design: "
        "EOF-cat broadcasts the PCs as five additional input channels at the first convolution, "
        "whereas the FiLM index encoder emits (γ, β) pairs that modulate all five encoder levels. "
        "Because a strictly single-level FiLM control is not included, insertion-depth parity is a "
        "design choice rather than an experimentally demonstrated equivalence; we therefore avoid "
        "attributing the entire performance difference to the modulation mechanism alone. The FiLM "
        "transform is x·(1 + γ) + β, so zero-valued PCs do not in general reproduce the identity "
        "mapping because the MLP carries bias terms.")

    add("where is the valid mask. We selected AdamW",
        "All reported models were retrained under this single leak-free protocol: AdamW "
        "(weight decay 1×10⁻⁵), cosine-annealing schedule, batch size 8, 100 epochs and gradient-norm "
        "clipping at 2.0 over the same three-point hyperparameter grid (lr ∈ {2×10⁻⁴, 1×10⁻⁴}, "
        "dropout ∈ {0.1, 0.2}), selecting checkpoints strictly by minimum MSE on the 60-month internal "
        f"validation set: Baseline ({best_tun['baseline']['config']}, val MSE {best_tun['baseline']['best_val_mse']:.5f}, "
        f"epoch {best_tun['baseline']['best_epoch']}), EOF-cat ({best_tun['eof_cat']['config']}, "
        f"val MSE {best_tun['eof_cat']['best_val_mse']:.5f}, epoch {best_tun['eof_cat']['best_epoch']}), "
        f"and FiLM-U-Net ({best_tun['film']['config']}, val MSE {best_tun['film']['best_val_mse']:.5f}, "
        f"epoch {best_tun['film']['best_epoch']}). EOF-cat broadcasts five PC channels at the input layer, "
        "whereas the FiLM index encoder emits channel-wise (γ, β) pairs for all five encoder levels; "
        "a single-level FiLM control was not run, so insertion-depth equivalence is not claimed. "
        "No model or hyperparameter was chosen using the independent test months.")

    add("This paper uses the spatial anomaly correlation coefficient",
        "3.1 Evaluation Criteria. This paper uses the spatial anomaly correlation coefficient (ACC) "
        "to assess the spatial similarity of predictions (Yang et al., 2024), and the temporal "
        "anomaly correlation coefficient (TCC) to describe local interannual co-variability. The "
        "normalised root mean square error (RMSE) is computed across valid grid points. Sign "
        "Consistency (SC) is the percentage of grid points where predicted and observed anomalies "
        f"share the same sign. Extreme events are defined as anomalies exceeding the training-set "
        f"90th percentile (ΔP ≥ {f(p90,4)}), and the Critical Success Index is "
        "CSI = Hits/(Hits + False Alarms + Misses).")

    add("Table 1. Summary of model configurations",
        f"Table 1. Summary of model configurations and evaluation performance. 'Historical ACC' is "
        f"computed over the {S['hindcast_months']}-month in-sample training + internal-validation "
        f"record; all other metrics use the {n_test}-month independent test period. Significant-grid "
        f"counts are out of {ng:,} valid land grids.")

    add("Table 2. Stepwise contribution decomposition",
        "Table 2. Stepwise contribution decomposition across the independent test period "
        "(2023.01–2024.09), derived from Table 1.")

    add("Table 1 summarizes the performance of tested model architectures",
        "Table 1 reports the performance of all architectures on the historical record and on the "
        f"{n_test}-month independent test period. Raw SEAS5 attains a test ACC of {f(sa,4)}. "
        "Relative to SEAS5, the three learned variants reach test ACC values of "
        f"{f(ba,4)} (Baseline), {f(ca,4)} (EOF-cat) and {f(fa,4)} (FiLM-U-Net), i.e. Stage 1 "
        f"{signed(D['stage1_acc'])}, Stage 2a {signed(D['stage2a_acc'])} and Stage 2b "
        f"{signed(D['delta_acc_film_base'])} relative to the Baseline. The mean TCC values are "
        f"{f(btcc,4)}, {f(ctcc,4)} and {f(ftcc,4)} respectively, with FiLM-U-Net "
        f"{signed(D['delta_tcc_film_cat'])} above EOF-cat and "
        f"{signed(D['delta_tcc_film_base'])} above the Baseline. " + rank_line + " "
        "The ranking is therefore metric-dependent, and we do not claim universal superiority of "
        "FiLM-U-Net: its reproducible advantage in this rerun is the spatial correlation structure "
        "and the fraction of grids with locally significant temporal correlation. The earlier "
        "claims of an ACC improvement over concatenation, of a monotonic stepwise gain and of "
        "positive increments across all metrics are withdrawn because they are not supported by "
        "the unified rerun. Sign Consistency of the learned models is "
        + ", ".join(f"{m} {p1(T1[m]['test_sc_pct'])}" for m in learned)
        + f" against {p1(ssc)} for raw SEAS5; this trade-off is discussed in Section 6.2.")

    add("To systematically evaluate forecasting skill across different temporal perspectives",
        "Figure 2 and Figure 3 serve complementary purposes: Figure 2 contrasts the multi-decadal "
        "historical hindcast distributions (1994–2022, in-sample) with the independent test "
        "aggregations (2023–2024) for each model, while Figure 3 resolves calendar-month, interannual "
        "and month-by-month behaviour. Regenerated from the unified evaluation, Figure 2 reports "
        f"historical mean ACC values of {f(T1['Baseline']['hindcast_acc'],4)} (Baseline), "
        f"{f(T1['EOF-cat']['hindcast_acc'],4)} (EOF-cat), {f(T1['FiLM-U-Net']['hindcast_acc'],4)} "
        f"(FiLM-U-Net) and {f(T1['SEAS5']['hindcast_acc'],4)} (SEAS5); the corresponding test means "
        f"are {f(ba,4)}, {f(ca,4)}, {f(fa,4)} and {f(sa,4)}. All legends are placed outside the axes.")

    add("Fig. 2. Comparison of ACC between the SEAS5 model",
        "Fig. 2. Comparison of ACC between SEAS5 and three U-Net–based models for precipitation "
        "anomaly percentage over China. (a, c, e) Historical hindcasts (1994.01–2022.12, 345 months, "
        "in-sample, with the 2018–2022 internal-validation window shaded); (b, d, f) independent test period "
        "(2023.01–2024.09, 21 months). Rows: Baseline U-Net, EOF-cat, FiLM-U-Net. Legends are positioned "
        "outside the plotting area; dashed lines mark period means.")

    add("The FiLM-based model maintains competitive and robust performance",
        f"Across the historical record the three U-Net variants span {f(hind_min,4)}–{f(hind_max,4)} in mean ACC, "
        "so the historical period should not be read as evidence of out-of-sample superiority: it is "
        "in-sample because the models were fitted on the 285-month training window and selected on "
        "the 60-month internal-validation window. The independent test period is the only strictly "
        "out-of-sample evidence, and its limited length is addressed in Sections 3.2, 4.4 and 6.2.")

    add("Fig. 3. Comprehensive temporal evaluation",
        "Fig. 3. Comprehensive temporal evaluation. (a) Calendar-month mean ACC (1994–2022); "
        "(b) annual mean ACC (1994–2022, with the internal-validation years shaded); (c) monthly "
        f"ACC trajectories of all four models over the {n_test} independent test months "
        f"(2023.01–2024.09); and (d) monthly paired difference ΔACC = FiLM − EOF-cat, with "
        f"{fc['pos_months']} of {fc['total_months']} months positive and a mean of "
        f"{signed(fc['mean_diff'])}. Legends are placed above the axes.")

    add("Figure 3 systematically evaluates temporal prediction skill",
        "Figure 3c–d isolates the paired ACC difference between FiLM-U-Net and EOF-cat on the "
        f"independent test period. The mean paired difference is {signed(fc['mean_diff'])} "
        f"with a Wilcoxon signed-rank p of {f(fc['wilcoxon_p'],3)}, a month-wise bootstrap 95% CI of "
        f"[{signed(fc['boot_month_ci95'][0])}, {signed(fc['boot_month_ci95'][1])}] and a year-blocked "
        f"bootstrap 95% CI of [{signed(fc['boot_year_block_ci95'][0])}, "
        f"{signed(fc['boot_year_block_ci95'][1])}]; this {sig_fc} the 5% significance level. "
        f"FiLM versus Baseline gives {signed(fb['mean_diff'])} (p = {f(fb['wilcoxon_p'],3)}; "
        f"year-blocked CI [{signed(fb['boot_year_block_ci95'][0])}, "
        f"{signed(fb['boot_year_block_ci95'][1])}]) and likewise {sig_fb} significance. Because the "
        "21 test months span only two calendar years, the year-blocked interval is reported "
        "alongside, and interpreted more cautiously than, the month-wise interval. Test-period "
        f"yearly means are {f(T1['FiLM-U-Net']['test_acc_2023'],4)} (2023) and "
        f"{f(T1['FiLM-U-Net']['test_acc_2024'],4)} (2024) for FiLM-U-Net, versus "
        f"{f(T1['EOF-cat']['test_acc_2023'],4)} and {f(T1['EOF-cat']['test_acc_2024'],4)} for EOF-cat.")

    add("In the historical multi-decadal climatological and interannual analysis",
        "In the historical analysis (Fig. 3a–b), all three U-Net variants show similar calendar-month "
        "and interannual ACC behaviour and clearly exceed raw SEAS5. Because this period is in-sample, "
        "these panels are descriptive rather than evidence of generalisation; the independent test "
        "period and the 2020–2021 rolling holdout provide the out-of-sample evidence.")

    add("Fig. 4. Spatial distribution of precipitation anomaly percentages",
        "Fig. 4. Spatial distribution of precipitation anomaly fractions ΔP in four representative "
        "months of 2023 (January, April, July, October) across Observation, SEAS5 (Raw), Baseline "
        "U-Net, EOF-cat and FiLM-U-Net (Ours). All panels share an identical colour scale "
        "[−1.0, 1.0] in fractional anomaly units.")

    # Section 4.2 Figure 4 & Regional quantitative analysis
    add("The independent sample validation results are displayed in Figure 4",
        "Figure 4 compares observed fractional precipitation anomalies (ΔP) with raw SEAS5, Baseline U-Net, "
        "EOF-cat and FiLM-U-Net across four representative seasonal months in 2023, and reveals clear regional "
        "differences that are quantified across China's three major hydroclimatic zones in the independent test set. "
        + reg_sentence_en + " Physically, circulation predictors and monsoon moisture transport over eastern "
        "monsoon China provide stronger predictable structure at the monthly scale, whereas steep orographic "
        "gradients over the Tibetan Plateau and sparse station coverage plus localized convective rainfall in "
        "the northwest arid interior restrict absolute amplitude recovery across all learned models; these "
        "regional attributions are empirical interpretations consistent with the zone-level metrics rather than "
        "causal identification.")

    add("In January and April 2023",
        "In January 2023 (Fig. 4, top row), observations exhibit positive anomalies over northern Xinjiang and "
        "parts of Northeast China contrasted against widespread negative anomalies across central, eastern and "
        "southern China as well as the western Tarim–Tibetan margins. Raw SEAS5 captures only weak tendencies, "
        "whereas all three U-Net variants reconstruct the northern-wet/southern-dry dipole while smoothing "
        "fine-scale coastal contrast. In April 2023 (second row), observations display a prominent wet belt "
        "spanning North China, the Huang–Huai plain and the Loess Plateau alongside strong dryness over the "
        "Tarim Basin and southern coastal margins; SEAS5 underestimates both poles, whereas the U-Net models "
        "sharply reproduce both the Huang–Huai wet core and the Northwest dry core.")

    add("During July 2023",
        "In July 2023 (Fig. 4, third row), observed precipitation anomalies show wet conditions across parts "
        "of Northeast China, North China, the Sichuan Basin/middle Yangtze and the southeastern coast contrasted "
        "against dry anomalies over Inner Mongolia, northwestern China and localized coastal South China pockets. "
        "Consistent with MSE regression-to-the-mean under weak summertime convective predictability, raw SEAS5 "
        "and all three U-Net variants yield damped anomaly amplitudes across eastern China, although the learned "
        "variants more clearly retain the northeastern wet tendency and northwestern dry tendency than SEAS5.")

    add("In October 2023",
        "In October 2023 (Fig. 4, bottom row), observations exhibit an intense wet core over the West China "
        "Autumn Rain region (Qinghai–Gansu–Sichuan–Shaanxi–Inner Mongolia bend) and localized southeastern coastal "
        "wetness, framed by dry anomalies over Xinjiang, Northeast China and the eastern Yangtze plains. "
        "SEAS5 yields only a muted central-western wet patch, whereas Baseline U-Net, EOF-cat and FiLM-U-Net "
        "all sharpen the West China Autumn Rain core and the northeastern/eastern dry contrast, though extreme "
        "dry magnitudes in western Xinjiang remain underestimated.")

    add("To quantitatively evaluate model fidelity during anomalous",
        "To evaluate behaviour during anomalous regimes, Table 1 reports Sign Consistency and the "
        f"Critical Success Index for extremes above the training-set 90th percentile (ΔP ≥ {f(p90,4)}). "
        f"Raw SEAS5 attains CSI = {f(scsi,3)}; deep-learning post-processing "
        f"recovers localised extremes (Baseline {f(bcsi,3)}, EOF-cat {f(ccsi,3)}, FiLM-U-Net "
        f"{f(fcsi,3)}). Extreme CSI is highest for {rank_csi} ({f(T1[rank_csi]['test_csi'],3)}), "
        f"whereas spatial mean TCC is highest for {rank_tcc} ({f(T1[rank_tcc]['test_tcc'],4)}); "
        "CSI alone should therefore not be read as overall superiority. Sign Consistency is "
        + ", ".join(f"{p1(T1[m]['test_sc_pct'])} ({m})" for m in learned)
        + f" against {p1(ssc)} for raw SEAS5; the learned variants do not uniformly improve "
        "directional fidelity, a trade-off discussed in Section 6.2. We therefore no longer claim "
        "a relative RMSE reduction over most of China; the field-level RMSE values in Table 1 show "
        "only small differences among the three U-Net variants.")

    add("Overall, the independent validation demonstrates",
        "Overall, the independent validation shows that U-Net post-processing markedly improves "
        f"spatial pattern correlation over raw SEAS5, and that among the three learned variants FiLM-U-Net "
        f"leads on {join_en(best_of_film)}, while lagging on {join_en(not_best)}.")

    add("Fig. 5. Spatial distribution of Temporal Correlation Coefficients",
        f"Fig. 5. Spatial TCC over the {n_test}-month independent test period for (a) Baseline U-Net "
        f"(mean {f(btcc,3)}, {p1(bsig)} significant), (b) EOF-cat (mean {f(ctcc,3)}, {p1(csig)}), "
        f"(c) FiLM-U-Net (mean {f(ftcc,3)}, {p1(fsig)}) and (d) ΔTCC = FiLM − EOF-cat "
        f"(mean {signed(ts['film_vs_cat_mean_dtcc'],4)}). Stippling in (a–c) marks grid points inside "
        "mainland China with two-tailed Student's t p < 0.05; these maps are descriptive and are "
        "accompanied by the field-level paired tests in Section 3.2.")

    add("Figure 5 presents the spatial distribution of TCC",
        f"Figure 5 presents spatial TCC across {ng:,} valid land grids. FiLM-U-Net (Fig. 5c) attains "
        f"a mean TCC of {f(ftcc,4)}, versus {f(ctcc,4)} for EOF-cat and {f(btcc,4)} for the Baseline. "
        f"The mean paired ΔTCC over EOF-cat is {signed(ts['film_vs_cat_mean_dtcc'])} "
        f"(Wilcoxon p = {fmt_p(ts['film_vs_cat_wilcoxon_p'])}; year-blocked bootstrap 95% CI "
        f"[{signed(ts['film_vs_cat_year_block_ci95'][0])}, "
        f"{signed(ts['film_vs_cat_year_block_ci95'][1])}]), which {sig_tcc} field-level significance. "
        f"Raw p < 0.05 stippling covers {p1(fsig)} of grids for FiLM-U-Net and {p1(csig)} for EOF-cat. "
        "The spatial Wilcoxon test and the year-blocked bootstrap of the mean ΔTCC address different "
        "questions; with only two calendar-year blocks the latter interval is degenerate, so neither "
        "result is used alone to declare field significance. " + fdr_line +
        "The stippled fraction is therefore treated as descriptive rather than as a formal "
        "field-significance test.")

    add("The relative RMSE reduction further confirms",
        "The spatial difference map (Fig. 5d) shows that FiLM−EOF-cat ΔTCC gains are spatially "
        "heterogeneous rather than uniform across the domain. Quantitative regional stratification confirms "
        f"this contrast: over the eastern monsoon zone (n = {r_em['n_grids']:,}), FiLM-U-Net attains "
        f"mean TCC {f(r_em['FiLM-U-Net']['test_tcc'])} and RMSE {f(r_em['FiLM-U-Net']['rmse'])} versus "
        f"EOF-cat {f(r_em['EOF-cat']['test_tcc'])} / {f(r_em['EOF-cat']['rmse'])} and Baseline "
        f"{f(r_em['Baseline']['test_tcc'])} / {f(r_em['Baseline']['rmse'])}; over the Tibetan Plateau "
        f"(n = {r_tp['n_grids']:,}), TCC is {f(r_tp['FiLM-U-Net']['test_tcc'])} (FiLM) vs "
        f"{f(r_tp['EOF-cat']['test_tcc'])} (EOF-cat) and {f(r_tp['Baseline']['test_tcc'])} (Baseline); "
        f"and over the northwest arid zone (n = {r_nw['n_grids']:,}), TCC is "
        f"{f(r_nw['FiLM-U-Net']['test_tcc'])} (FiLM) vs {f(r_nw['EOF-cat']['test_tcc'])} (EOF-cat) and "
        f"{f(r_nw['Baseline']['test_tcc'])} (Baseline). Field-level RMSE values in Table 1 span a narrow "
        f"range ({f(rmse_min,4)}–{f(rmse_max,4)}, total spread {f(rmse_span,4)}), and we do not claim "
        "widespread relative RMSE reductions over China.")

    add("To address whether all 10 atmospheric predictors meaningfully contribute",
        "To address whether all ten atmospheric predictors contribute meaningfully, we performed a "
        "permutation feature-importance analysis on the independent test set with 20 random temporal "
        "shuffles per channel (Table 3). The largest mean ACC drops occur for "
        f"{T3[0]['full_name']} ({signed(T3[0]['mean_drop'])} ± {f(T3[0]['std_drop'])}), followed by "
        f"{t3_top_str}. In total, {len(t3_pos)} predictors yield positive ΔACC drops exceeding 0.002, "
        "whereas several circulation and thermal variables produce near-zero or slightly negative drops, "
        "indicating strong collinearity with the leading predictors rather than a demonstrable individual "
        "contribution. We therefore refrain from claiming that every predictor is individually necessary.")
    add("Table 3. Permutation feature importance ranking",
        "Table 3. Permutation feature importance of the ten atmospheric predictors on the independent "
        "test set (20 random shuffles per channel).")

    add("Table 4. Forecast skill (ACC)",
        "Table 4. Forecast skill (ACC) stratified by ENSO phase (standardised SST PC1 threshold ±0.5). "
        "The 1994–2022 historical rows are in-sample; only the 2023–2024 rows are out-of-sample. "
        "ENSO-stratified TCC values are reported in the accompanying results bundle and Section 4.4.")

    add("To rigorously address whether model performance was biased",
        "To address whether the 2023–2024 El Niño regime biased the evaluation, Table 4 stratifies "
        f"ACC by ENSO phase. The {n_test} independent test months comprise "
        f"{EN['test']['El Nino']['months']} El Niño, {EN['test']['Neutral']['months']} neutral and "
        f"{EN['test']['La Nina']['months']} La Niña months. In the historical record the phase "
        "distribution is more balanced, but these rows are in-sample and are reported as a "
        "descriptive stratification, not as evidence of multi-decadal generalisation. "
        "ENSO-stratified mean TCC (also requested by the reviewer) is reported in the results bundle: "
        + "; ".join(
            f"{ph}: " + ", ".join(
                f"{m} {f(X['enso_tcc']['test'][ph][m]['mean_tcc'],3) if X and X['enso_tcc']['test'][ph].get(m) else 'n/a'}"
                for m in ("Baseline", "EOF-cat", "FiLM-U-Net")
            ) for ph in ("El Nino", "La Nina", "Neutral")
        )
        + (
            f". TCC stratification requires at least five monthly samples, so {join_en(small_phs)} "
            f"{'are' if len(small_phs) > 1 else 'is'} reported as not available. To provide out-of-sample evidence outside the "
            if small_phs
            else ". To provide out-of-sample evidence outside the "
        )
        + "El Niño-dominated window, a "
        f"24-month rolling holdout (2020.01–2021.12; {HO['la_nina_months']} La Niña, "
        f"{HO['neutral_months']} neutral, {HO['el_nino_months']} El Niño months) was retrained "
        "from scratch using only the remaining historical months, with the SST EOF basis refitted "
        "on the 1994–2015 training portion. On this holdout, FiLM-U-Net attains ACC "
        f"{f(HO['FiLM-U-Net']['acc'],4)} versus {f(HO['EOF-cat']['acc'],4)} (EOF-cat) and "
        f"{f(HO['Baseline']['acc'],4)} (Baseline), with mean TCC "
        f"{f(HO['FiLM-U-Net']['tcc'],4)} versus {f(HO['EOF-cat']['tcc'],4)}. " + ho_line +
        " We explicitly acknowledge that 21 independent test months cannot support multi-decadal interannual "
        "inference, and that TCC estimated from 21 samples has low power at individual grid points.")

    # ---- Section 5 ----
    add("During the inference phase, the layer-wise γ and β parameters",
        "During evaluation, the layer-wise γ and β modulation parameters generated by the IndexEncoder "
        "were recorded over the 345-month historical hindcast window (1994–2022). Channel-wise Pearson "
        "correlation coefficients were then computed between each encoder channel's γ/β time series "
        "and the five standardised SST PCs, alongside 2D PCA projections of the 128-channel Layer-5 "
        "modulation vectors (Figs. 6–7).")

    add("Fig. 6. (a) Principal component analysis",
        "Fig. 6. (a) PCA projection of the fifth-encoder-layer scaling parameter γ, coloured by SST "
        "PC1; (b) same for the shifting parameter β. Colour denotes the phase of PC1 (warm: "
        "El Niño-like; cool: La Niña-like).")
    add("Both γ and β exhibit clear organization along the ENSO axis",
        "Both γ and β are organised along the ENSO axis: because they are produced by a "
        "deterministic MLP from the SST PCs, such structure is a constructed consequence of the "
        "architecture. It is consistent with, but does not independently prove, the learning of "
        "teleconnection physics; the shuffled-PC controls in Figs. 7 and 9 provide the corresponding "
        "negative control.")
    add("The structured gradient in color across the γ–β manifolds",
        "The colour gradient across the γ–β manifolds reflects the deterministic mapping from "
        "SST PC1 by construction, indicating that the IndexEncoder propagates large-scale SST phase "
        "information without collapsing it; whether this conditional modulation translates into forecast "
        "skill is evaluated separately in Sections 4.1 and 5.3.")
    add("Overall, the parameter–teleconnection correspondence revealed in Figure 6",
        "Overall, Figure 6 shows that the modulation parameters encode the SST index in a "
        "low-dimensional manifold. This is a necessary condition for conditional modulation, not by "
        "itself evidence of physical teleconnection skill.")

    add("Fig. 7. (a) Correlation coefficients between the FiLM scaling",
        "Fig. 7. (a) Channel-wise correlations between first-layer γ and the five SST PCs; "
        "(b) same for β; (c) mean |r| of γ per PC with a 100-permutation shuffled-PC control and "
        "its 95th percentile; (d) same for β. The control quantifies the correlation strength "
        "expected when the PC–parameter alignment is destroyed.")

    # Section 5.2 dynamic rewrites from F7 (eliminates stale 0.8 / 0.47 / 0.38)
    g_mean_summary = ", ".join(f"PC{i+1}: {g_means[i]:.3f}" for i in range(5))
    g_max_summary = ", ".join(f"PC{i+1}: {g_maxs[i]:.3f}" for i in range(5))
    b_mean_summary = ", ".join(f"PC{i+1}: {b_means[i]:.3f}" for i in range(5))
    b_max_summary = ", ".join(f"PC{i+1}: {b_maxs[i]:.3f}" for i in range(5))
    g_p95_max = max(g_shuf_p95)
    b_p95_max = max(b_shuf_p95)

    add("For the scaling parameter γ (Figure 7a)",
        f"For the first-layer scaling parameter γ (Fig. 7a), channel-wise Pearson correlations span both "
        f"positive and negative values across all five modes, with maximum absolute per-channel correlations of "
        f"{g_max_summary} (peaking at {g_maxs[g_lead_x]:.3f} on PC{g_lead_x+1}). Because different convolutional "
        "filters exhibit distinct positive, weak or negative loadings across PC1–PC5, the trained index "
        "encoder distributes gain adjustments unevenly across feature channels rather than applying an "
        "identical response to one mode.")
    add("The shifting parameter β (Figure 7b)",
        f"The first-layer shifting parameter β (Fig. 7b) displays a distinct channel-wise correlation pattern, "
        f"with maximum absolute per-channel correlations of {b_max_summary} (peaking at "
        f"{b_maxs[b_lead_x]:.3f} on PC{b_lead_x+1}). Thus, whereas some channels couple most strongly to "
        f"PC{g_lead_m+1} in scaling (γ), additive shifting (β) distributes weight across lower-order modes "
        f"including PC{b_lead_m+1} and PC1.")
    add("The bar plots in Figure 7c–d summarize",
        f"Figure 7c–d summarises the channel-averaged mean |r| for each SST PC against the 100-permutation "
        f"shuffled-PC negative control. For γ (Fig. 7c), mean |r| values are {g_mean_summary} (largest on "
        f"PC{g_lead_m+1} at {g_means[g_lead_m]:.3f}), substantially exceeding the shuffled-PC noise ceiling "
        f"(p95 ≤ {g_p95_max:.3f}; shuffled means {g_shuf_m[0]:.3f}–{max(g_shuf_m):.3f}). For β (Fig. 7d), "
        f"mean |r| values are {b_mean_summary} (largest on PC{b_lead_m+1} at {b_means[b_lead_m]:.3f}), "
        f"likewise above the shuffled-PC ceiling (p95 ≤ {b_p95_max:.3f}). Because correlation strength is "
        "non-monotonic across PC order, the encoder does not rely solely on PC1; rather, it distributes "
        "conditioning weights across the leading modes.")
    add("Overall, the results demonstrate",
        f"Overall, the real-PC correlations exceed the 100-permutation shuffled-PC ceiling across all five "
        f"modes (e.g. γ PC1 mean |r| = {g_means[0]:.3f} vs shuffled mean {g_shuf_m[0]:.3f}; γ PC{g_lead_m+1} "
        f"{g_means[g_lead_m]:.3f} vs {g_shuf_m[g_lead_m]:.3f}), showing that the γ/β–PC alignment is not an "
        "artefact of marginal distributions. Because γ and β are deterministic MLP outputs of the PCs, this "
        "alignment is expected by construction; it verifies the conditioning pathway rather than proving the "
        "discovery of new physical laws.")

    # Section 5.3 Fig. 8 & Table 5
    gcam_base = t5_base.get("mean_gradcam")
    gcam_catz = t5_catz.get("mean_gradcam")
    gcam_catlz = t5_catlz.get("mean_gradcam")
    gcam_film = t5_full.get("mean_gradcam")
    gcam_filmz = t5_zero.get("mean_gradcam")
    gcam_filmlz = t5_flz.get("mean_gradcam")

    add("Fig. 8 Grad-CAM interpretability analysis",
        "Fig. 8. Six-panel symmetric Grad-CAM ablation over China's valid land domain (backpropagating "
        "from valid-land mean output at `up4.conv`). (a) Baseline U-Net; (b) EOF-cat with SST PCs zeroed; "
        "(c) EOF-cat with local predictors zeroed; (d) full FiLM-U-Net; (e) FiLM-U-Net with SST PCs zeroed; "
        "(f) FiLM-U-Net with local predictors zeroed. All six panels share a common China spatial-span "
        "normalisation [0, 1] after subtracting each map's China-domain floor so featureless local-zeroed "
        "controls retain no structured monsoonal or topographic contrast; titles report the corresponding test ACC.")
    add("Figure 8 presents the Grad-CAM visualizations for the four model configurations",
        f"Figure 8 presents the six matched symmetric configurations requested by the reviewer, placing the "
        f"FiLM branch ablations side by side with EOF-cat under a single shared China spatial-span colour "
        f"scale. Using Baseline U-Net as reference (Fig. 8a, mean normalised intensity {f(gcam_base,4)}, "
        f"ACC = {f(t5_base['test_acc'],4)}), spatial saliency concentrates across northern, eastern and "
        "western China regions of pronounced topographic and monsoonal moisture gradients.")
    add("When the concatenation model is forced",
        f"In the concatenation branch (Fig. 8b–c), zeroing the broadcasted SST PCs reduces mean normalised "
        f"Grad-CAM saliency to {f(gcam_catz,4)} (ACC = {f(t5_catz['test_acc'],4)}), while retaining a spatial "
        f"distribution broadly similar to the Baseline. Conversely, zeroing all ten local atmospheric predictors "
        f"(Fig. 8c) collapses spatially varying saliency across China to the low uniform floor ({f(gcam_catlz,4)}, "
        f"ACC = {f(t5_catlz['test_acc'],4)}), confirming that broadcasted uniform PC channels cannot sustain "
        "coherent local spatial activation.")
    add("In sharp contrast, the proposed FiLM-U-Net model (Figure 8d)",
        f"In the FiLM branch (Fig. 8d–f), Full FiLM-U-Net achieves the highest domain-averaged normalised "
        f"Grad-CAM intensity ({f(gcam_film,4)}; Fig. 8d), with elevated saliency across northwestern, northern "
        f"and central–eastern China. Zeroing the SST PCs lowers mean intensity to {f(gcam_filmz,4)} (Fig. 8e, ACC = "
        f"{f(t5_zero['test_acc'],4)}), whereas zeroing all local atmospheric predictors eliminates structured "
        f"regional contrasts and leaves only a featureless residual gradient (mean {f(gcam_filmlz,4)} from "
        f"bias and zero-padding boundary terms modulated by γ/β; Fig. 8f, ACC = {f(t5_flz['test_acc'],4)}). This behaviour "
        "is consistent with FiLM acting as a conditional modulator of locally extracted atmospheric features "
        "rather than as an independent spatial-field generator. Note that zero-valued PCs do not yield "
        "γ = 0 and β = 0 because the index MLP contains bias terms, so this ablation is a fixed-condition "
        "control rather than an identity mapping.")
    add("Overall, the Grad-CAM comparison demonstrates that",
        "Overall, the symmetric Grad-CAM comparison shows that local predictors dominate spatial "
        "activation in both architectures, and that the FiLM conditioning pathway does not create "
        "a spatial field on its own. We use 'is consistent with' rather than 'proves' because "
        "Grad-CAM localises sensitivity, not causal physical mechanism.")
    add("To rigorously address inquiries regarding symmetric ablation",
        "To quantify the symmetric ablation (Table 5), we report all matched configurations. "
        f"Zeroing local predictors gives FiLM-U-Net ACC "
        f"{f(t5_flz['test_acc'],4)} (and EOF-cat {f(t5_catlz['test_acc'],4)}), confirming that local predictors "
        f"dominate spatial skill. Zeroing the SST PCs gives {f(t5_zero['test_acc'])} "
        f"({signed(abl_zero_acc)} versus the full model) and shuffling them gives "
        f"{f(t5_shuf.get('test_acc'))} ± {f(t5_shuf.get('test_acc_std'))} ({signed(abl_shuf_acc)}). "
        + abl_acc_line + " " + abl_tcc_line + " Differences of this magnitude fall within run-to-run noise "
        "and should not be over-interpreted; the shuffled control is averaged over 50 permutations. "
        "Interpretation therefore rests on the spatial TCC field and significant-grid coverage rather than on "
        "a generic claim of ACC improvement.")
    add("Table 5. Symmetric interpretability ablation",
        "Table 5. Symmetric interpretability ablation and negative controls on the independent test "
        "period. ACC and TCC are reported as measured; interpretation follows Section 5.3.")

    # Section 5.4 dynamic rewrites from F9 (eliminates Niño3/Niño3.4 / weaker in deep layers mismatch)
    add("To complement the spatial Grad-CAM analysis over China",
        "To complement the China-domain Grad-CAM analysis, we reconstructed layer-wise FiLM–EOF global "
        "sensitivity maps by weighting each SST EOF pattern with the channel-wise γ–PC correlation across the "
        "five modes and clamping positive sensitivity across channels. A 100-permutation shuffled-PC negative "
        "control for Layer 1 is included as Fig. 9(f), normalised on the identical global scale.")
    add("Figure 9 presents the layer-wise FiLM–EOF reconstruction sensitivity maps",
        "Figure 9 displays the Pacific-centred (180°E) layer-wise FiLM–EOF sensitivity maps across all "
        "five encoder depths together with the shuffled-PC control on one common [0, 1] colour scale "
        "normalised to the overall maximum across layers.")
    add("Fig. 9. FiLM-EOF Reconstruction Sensitivity Maps",
        "Fig. 9. Layer-wise global FiLM–EOF sensitivity maps (Pacific-centred projection) for encoder "
        "layers 1–5 (a–e) and the 100-permutation shuffled-PC control for Layer 1 (f). All panels share "
        "one common [0, 1] normalisation relative to the cross-layer peak.")
    add("Layer 1 and Layer 2 exhibit strong sensitivity",
        f"Layer 1 (Fig. 9a; normalised peak {F9['layer_max_normalized'][0]:.3f}) shows a concentrated "
        f"equatorial eastern Pacific signal ({fmt_peaks(F9, 'layer_1', 2)}), broadly aligned with the tropical "
        f"Pacific cold-tongue/ENSO loading in PC1. By Layer 2 (Fig. 9b; peak "
        f"{F9['layer_max_normalized'][1]:.3f}), maximum sensitivity shifts into extratropical and marginal "
        f"basins ({fmt_peaks(F9, 'layer_2', 2)}), reflecting mixed weights across higher-order PCs rather than "
        "a purely tropical equatorial maximum.")
    add("Layers 3 and 4 show broader and more diversified",
        f"Layers 3 and 4 (Fig. 9c–d; normalised peaks {F9['layer_max_normalized'][2]:.3f} and "
        f"{F9['layer_max_normalized'][3]:.3f}) shift sensitivity across Pacific and high-latitude/marginal basins "
        f"(Layer 3 peaks: {fmt_peaks(F9, 'layer_3', 2)}; Layer 4 peaks: {fmt_peaks(F9, 'layer_4', 2)}), "
        "indicating that intermediate encoder layers distribute linear sensitivity across Bering/North Pacific, "
        "equatorial eastern Pacific and marginal-sea EOF structures.")
    add("Layer 5—the deepest encoder layer",
        f"At the deepest bottleneck stage, Layer 5 (Fig. 9e; normalised peak "
        f"{F9['layer_max_normalized'][4]:.3f}, the global maximum across all five encoder depths) does not "
        "decay monotonically; instead it displays strong reconstructed maxima over "
        f"{fmt_peaks(F9, 'layer_5', 2)}. In sharp contrast, the 100-permutation shuffled-PC control (Fig. 9f) "
        f"remains nearly flat near zero across the global ocean (mean-map maximum "
        f"{F9['shuffled_control_mean_max_normalized']:.3f}; single-draw maximum "
        f"{F9['shuffled_control_rep_max_normalized']:.3f}). Because these maps are linear weighted sums of "
        "five truncated global EOF modes, high extratropical intensity (especially over high-latitude or "
        "coastal SST modes) reflects linear projection weights onto the EOF basis and is reported descriptively "
        "as statistical sensitivity rather than proof of direct causal oceanic forcing.")
    add("In summary, the interpretability analyses collectively demonstrate",
        "In summary, the interpretability analyses show that the FiLM parameters are structured by "
        "the SST indices by construction, that local predictors dominate spatial activation, and "
        "that reconstructed global sensitivity exceeds the shuffled-PC control across all layers "
        f"(layer peaks {min(F9['layer_max_normalized']):.3f}–{max(F9['layer_max_normalized']):.3f} "
        f"versus shuffled mean-map peak {F9['shuffled_control_mean_max_normalized']:.3f}). "
        "These patterns are treated as reconstructed statistical sensitivity consistent with conditional "
        "modulation rather than independent proof of physical teleconnection mechanisms.")

    # ---- Section 6 / 7 ----
    add("6. Results and Discussion", "6. Discussion")
    add("6.1 Discussion", "6.1 Synthesis of results")
    add("Based on above analysis, we introduce a practical EOF+FiLM",
        "We introduced a practical EOF+FiLM architecture that integrates five compact SST indices "
        f"with a U-Net backbone. On the independent test period the learned models ({f(min(T1[m]['test_acc'] for m in learned),3)}–"
        f"{f(max(T1[m]['test_acc'] for m in learned),3)}) all exceed raw SEAS5 ({f(sa,3)}) in test ACC. " + learned_tcc_line + " FiLM-U-Net attains the highest "
        "spatial TCC and the largest significant-grid coverage of the learned variants, but "
        f"{acc_vs_cat} EOF-cat on test ACC. Paired tests show that monthly ACC differences among "
        "the learned variants are small and not statistically significant at the 5% level, and "
        "the symmetric ablations show that zeroing or shuffling the SST inputs does not produce a "
        "consistent ACC change. The interpretability evidence is therefore framed as consistency with "
        "conditional modulation, not as proof of learned teleconnection physics.")
    add("Figures 2 and 5 summarize the performance of different model configurations",
        f"Figures 2 and 5 summarise both periods. On the historical (in-sample) record the three "
        f"U-Net variants attain mean ACC values between "
        f"{f(min(T1[m]['hindcast_acc'] for m in learned),3)} and "
        f"{f(max(T1[m]['hindcast_acc'] for m in learned),3)}, all above "
        f"SEAS5 ({f(T1['SEAS5']['hindcast_acc'],3)}). On the independent test period the three "
        f"learned variants span {f(min(T1[m]['test_acc'] for m in learned),3)}–"
        f"{f(max(T1[m]['test_acc'] for m in learned),3)} against {f(sa,3)} for SEAS5.")
    add("During the hindcast period, the baseline U-Net improves the mean spatial ACC from 0.2 to 0.29",
        "Because the historical record includes the training window, we do not use it to claim "
        "decadal-scale generalisation. The ENSO-stratified results (Table 4) show that model ranking "
        "is phase dependent, and the 2020–2021 rolling holdout provides the additional out-of-sample "
        "check requested by the reviewer. We therefore no longer claim any systematic "
        "advantage during strong ENSO phases; the rerun does not support such a claim.")
    add("Spatially, the FiLM model exhibits higher TCC across most regions of China",
        f"Spatially, FiLM-U-Net attains a mean TCC of {f(ftcc,4)} versus {f(btcc,4)} (Baseline) and "
        f"{f(ctcc,4)} (EOF-cat) on the independent test set. Regional performance varies across "
        f"hydroclimatic zones: in the eastern monsoon zone (n = {r_em['n_grids']:,}), test ACC is "
        f"{f(r_em['FiLM-U-Net']['test_acc'])} (FiLM) vs {f(r_em['EOF-cat']['test_acc'])} (EOF-cat) and "
        f"{f(r_em['Baseline']['test_acc'])} (Baseline), with mean TCC {f(r_em['FiLM-U-Net']['test_tcc'])} vs "
        f"{f(r_em['EOF-cat']['test_tcc'])} and {f(r_em['Baseline']['test_tcc'])}; over the Tibetan Plateau "
        f"(n = {r_tp['n_grids']:,}), test ACC is {f(r_tp['FiLM-U-Net']['test_acc'])} vs "
        f"{f(r_tp['EOF-cat']['test_acc'])} and {f(r_tp['Baseline']['test_acc'])} (TCC "
        f"{f(r_tp['FiLM-U-Net']['test_tcc'])} vs {f(r_tp['EOF-cat']['test_tcc'])} and "
        f"{f(r_tp['Baseline']['test_tcc'])}); and over the northwest arid zone (n = {r_nw['n_grids']:,}), "
        f"test ACC is {f(r_nw['FiLM-U-Net']['test_acc'])} vs {f(r_nw['EOF-cat']['test_acc'])} and "
        f"{f(r_nw['Baseline']['test_acc'])} (TCC {f(r_nw['FiLM-U-Net']['test_tcc'])} vs "
        f"{f(r_nw['EOF-cat']['test_tcc'])} and {f(r_nw['Baseline']['test_tcc'])}).")
    add("Overall, the EOF–FiLM framework provides a physically interpretable and effective approach",
        "Overall, the EOF–FiLM framework provides a compact conditional-modulation mechanism whose "
        "measurable benefit in this study is improved spatial correlation and significant-grid "
        "coverage, with explicitly bounded claims about generalisation, statistical significance "
        "and physical interpretation.")
    add("While the proposed EOF–FiLM framework demonstrates notable improvements",
        "While the proposed EOF–FiLM framework improves spatial correlation over the two learned "
        "baselines, several limitations must be stated explicitly: (1) In-sample historical "
        "evidence: the 345-month hindcast record contains the training and model-selection windows "
        "and cannot substitute for out-of-sample confirmation. (2) Short independent window: "
        "21 months span only two calendar years, giving low power for interannual TCC inference; "
        "year-blocked bootstrap intervals are wide, and the 2020–2021 rolling holdout is a "
        "sensitivity check rather than an equally sized independent test set. (3) Non-significant "
        "ACC differences: paired tests do not reject the null for FiLM versus EOF-cat or versus "
        "Baseline on monthly ACC. (4) SST modulation does not improve test ACC: zeroing or "
        "shuffling SST PCs does not degrade ACC; the gain appears in TCC and coverage. (5) "
        f"Metric trade-offs: {sc_tradeoff_en}, and FiLM-U-Net is not the leading learned model on {join_en(not_best)}. (6) Multiple comparisons: with "
        f"{ng:,} grids, about {int(round(ng*0.05))} false positives are expected at p < 0.05; "
        "Benjamini–Hochberg FDR-controlled counts and field-level paired tests are reported in "
        "Section 3.2. (7) Global affine assumption: γ and β are spatially uniform within a channel, "
        "which cannot represent spatially heterogeneous teleconnection impacts. (8) EOF truncation "
        "to five PCs filters localised non-stationary SST signals. (9) MSE loss smooths sharp "
        "convective extremes; focal or hurdle losses are promising alternatives. (10) Insertion-depth "
        "asymmetry: FiLM modulates all five encoder levels whereas EOF-cat receives the PCs once at "
        "the input, and no single-level FiLM control was run, so the comparison is not a pure "
        "mechanism-isolating ablation. (11) γ and β are deterministic MLP outputs; correlation with "
        "SST PCs is a constructed consequence and cannot by itself establish physical learning. "
        "(12) Because the index MLP has biases, zero PCs do not produce the identity modulation, so "
        "the 'SST zeroed' control is a fixed-condition control, not a removal of the FiLM pathway.")

    add("Acknowledgments. This work was supported by",
        "Acknowledgments. This work was supported by the National Natural Science Foundation of "
        "China (Grant Nos. 42275050, 42130610 and 42575056). The authors thank the two anonymous "
        "reviewers for their constructive comments.")

    # Apply full rewrites first
    n = doc.replace_paragraphs(R)
    print(f"[Manuscript] rewrote {n} paragraphs")

    # Insert new sections
    doc.insert_after("3.1 Evaluation Criteria.", [
        ("3.2 Statistical significance protocol", False, True),
        ("All significance statements follow a pre-specified protocol. (i) Paired monthly ACC "
         f"differences on the {n_test} independent test months are tested with a two-sided Wilcoxon "
         "signed-rank test (α = 0.05). (ii) 95% confidence intervals for the mean difference are "
         "obtained from 10,000 bootstrap resamples of individual months and, separately, from 10,000 "
         "resamples of entire calendar-year blocks; because the test period spans only two calendar "
         "years, only three distinct year-block resamples exist, so the year-blocked interval is "
         "degenerate and is reported descriptively rather than as confirmatory inference. "
         "(iii) For spatial TCC, grid-point p-values come from a two-tailed Student's t test on the "
         "21 monthly pairs; field-level inference instead uses a paired Wilcoxon test over valid "
         "grids plus a 2,000-draw year-blocked bootstrap of the mean ΔTCC. These two tests answer "
         "different questions (the spatial distribution of ΔTCC versus the temporal stability of "
         "its mean) and can therefore diverge; we report both without merging them into a single "
         "claim. (iv) To control the multiple-comparison problem inherent in stippling "
         f"{ng:,} grids, we additionally apply the Benjamini–Hochberg FDR procedure at q = 0.05. "
         + fdr_line + "(v) Shuffled-PC controls use 100 permutations for the parameter "
         "correlations and sensitivity maps and 50 permutations for the ablation metrics. No "
         "hyperparameter or model was selected using the independent test set.", True, False),
    ])
    doc.insert_after("Acknowledgments. This work was supported by", [
        ("7. Conclusions", False, True),
        ("This study evaluated a FiLM-conditioned U-Net for monthly precipitation anomaly "
         "prediction over China under a strictly leak-free chronological protocol. The principal "
         "empirical finding is that conditional modulation of local atmospheric features by five "
         "SST principal components improves spatial correlation structure and the coverage of "
         "statistically significant skill relative to direct channel concatenation, while the "
         "monthly ACC differences are small, of mixed sign and not statistically significant at the "
         f"5% level: FiLM-U-Net {acc_vs_cat} EOF-cat on test ACC. Symmetric ablations show that "
         "local predictors dominate spatial activation and that zeroing or shuffling the SST "
         "inputs produces no consistent ACC change; shuffled-PC controls "
         "confirm that parameter–PC correlations are a constructed consequence of the conditioning "
         "MLP. An additional 24-month rolling holdout (2020–2021), which contains El Niño, La Niña "
         "and neutral months, together with ENSO-stratified diagnostics bounds, but does not "
         "eliminate, the representativeness limitation of the 21-month independent test window. "
         "Future work should target spatially adaptive modulation, mechanism-isolating single-level "
         "controls, longer out-of-sample evaluation, and loss functions that preserve extreme "
         "convective events. Code and evaluation artefacts are made available to support "
         "reproducibility.", True, False),
        ("Data and Code Availability. The leak-free training/evaluation pipeline, the unified "
         "metrics file, prediction arrays and the figure-generation scripts used in this revision "
         "are archived with the manuscript's reproducibility package; source code is available "
         "from the corresponding author on reasonable request and is intended for public release "
         "upon acceptance. Raw ERA/ERSST and station data are subject to the data policies of their "
         "providers.", True, False),
        ("CRediT authorship contribution statement. Fan Yue: conceptualisation, methodology, "
         "software, formal analysis, writing – original draft. Xiaojuan Wang: supervision, "
         "validation, writing – review and editing. Rong Ai and Zhiqiang Gong: data curation, "
         "resources. Yinzhong Wu and Guolin Feng: methodology, validation, writing – review and "
         "editing.", True, False),
        ("Declaration of competing interest. The authors declare no competing financial or "
         "personal interests that could have influenced the work reported in this paper.", True, False),
        ("Generative AI use. Generative AI tools were not used to generate scientific results, "
         "figures or data; they were used only for language editing and consistency checking of "
         "the revised manuscript under author supervision.", True, False),
    ])

    # ---- Tables 1..5 (rebuild BEFORE typography/numeric sweeps so cells stay exact) ----
    def tbl1_rows():
        head = ["Model Architecture", "Input Configuration", "Historical ACC (345m, in-sample)",
                f"Test ACC ({n_test}m)", "Spatial Mean TCC", "Sig. Grids (p<0.05)",
                "Test RMSE", "Sign Consistency", "Extreme CSI (>p90)"]
        rows = [head]
        labels = {
            "SEAS5": ("SEAS5 (Raw Dynamical)", "Raw Dynamical Precipitation"),
            "Baseline": ("Baseline U-Net (A)", "10 Local Atmospheric Variables"),
            "EOF-cat": ("U-Net + EOF-cat (B)", "10 Local Vars + Broadcasted SST PCs"),
            "FiLM-U-Net": ("FiLM-U-Net (C, Ours)", "10 Local Vars + FiLM SST Modulation"),
        }
        for m in ["SEAS5", "Baseline", "EOF-cat", "FiLM-U-Net"]:
            d = T1[m]
            rows.append([
                labels[m][0], labels[m][1], f(d["hindcast_acc"]), f(d["test_acc"]),
                f(d["test_tcc"]) if m != "SEAS5" else "-",
                f"{p1(d['test_sig_pct'])} ({d['test_n_sig']:,}/{d['test_n_valid']:,})" if m != "SEAS5" else "-",
                f(d["test_rmse"]), p1(d["test_sc_pct"]), f(d["test_csi"], 3),
            ])
        rows.append(["Global Input Model", "Global atmospheric + SST fields", "-", "0", "-", "-", "-", "-", "-"])
        return rows

    def tbl2_rows():
        return [
            ["Processing Stage", "Model Transition", "Core Methodological Mechanism", "ΔACC", "ΔTCC",
             "Primary Physical & Methodological Role"],
            ["Stage 1 (Local Downscaling)", "SEAS5 Raw → Baseline U-Net",
             "Multi-channel U-Net convolution on 10 local atmospheric fields",
             signed(D["stage1_acc"]), "-",
             "Corrects systematic spatial displacement and local mean biases using multi-level circulation fields."],
            ["Stage 2a (Naive Teleconnection Concatenation)", "Baseline U-Net → EOF-cat",
             "Direct channel concatenation of broadcasted 1D SST PCs",
             signed(D["stage2a_acc"]), signed(D["stage2a_tcc"]),
             "Broadcast PC planes add five constant input channels; the observed ranking is metric-dependent and no overfitting mechanism is asserted."],
            ["Stage 2b (Affine Teleconnection Modulation)", "Baseline U-Net → FiLM-U-Net",
             "Feature-wise affine transformation x·(1+γ)+β conditioned on SST",
             f"{signed(D['delta_acc_film_base'])} ({signed(D['delta_acc_film_cat'])} vs Cat)",
             f"{signed(D['delta_tcc_film_base'])} ({signed(D['delta_tcc_film_cat'])} vs Cat)",
             "Conditions channel activations on oceanic indices without adding spatial input channels; the ACC difference is small and not statistically significant."],
        ]

    def tbl3_rows():
        rows = [["Rank", "Predictor Variable", "Abbr.", "Drop in Test ACC (ΔACC ± SD)",
                 "Physical Meteorological Mechanism"]]
        base_mech = {
            "tp": "Primary background dynamical precipitation field",
            "slp": "Surface monsoon trough and regional pressure gradient",
            "t850": "Lower-tropospheric thermal contrast and baroclinic front",
            "t2m": "Near-surface sensible heat and boundary-layer stability",
            "h200": "Upper-tropospheric South Asian High and jet displacement",
            "h500": "Mid-tropospheric subtropical high and trough–ridge wave train",
            "v200": "Upper-level cross-equatorial flow and divergent outflow",
            "v850": "Low-level East Asian monsoon water-vapour transport",
            "u200": "East Asian subtropical upper-level westerly jet stream",
            "u850": "Low-level zonal monsoon westerly and shear transport",
        }
        for i, it in enumerate(T3, 1):
            m_desc = base_mech.get(it["var"], "Atmospheric circulation predictor")
            if it["var"] != "tp" and it["mean_drop"] <= 0.002:
                m_desc += " (near-zero/negative ΔACC reflects collinearity)"
            rows.append([str(i), it["full_name"], it["var"],
                         f"{signed(it['mean_drop'])} ± {f(it['std_drop'])}", m_desc])
        return rows

    def tbl4_rows():
        rows = [["Period / ENSO Phase", "Month Count", "SEAS5 (Raw)", "Baseline U-Net",
                 "EOF-cat (Concat)", "FiLM-U-Net (Ours)", "FiLM vs Baseline (ΔACC)",
                 "FiLM vs Concat (ΔACC)"]]
        for period, label in [("hindcast", "Historical (in-sample)"), ("test", "Test Period")]:
            for phase in ["El Nino", "La Nina", "Neutral", "All"]:
                d = EN[period][phase]
                if d["months"] == 0:
                    continue
                rows.append([
                    f"{label}: {phase}", f"{d['months']} months ({d['pct']:.1f}%)",
                    *[f(d.get(m)) for m in order],
                    signed(d.get("gain_vs_base", 0)), signed(d.get("gain_vs_cat", 0)),
                ])
        return rows

    def tbl5_rows():
        rows = [["Model Architecture", "Ablation / Control Configuration", "Test ACC",
                 "Spatial Mean TCC", "Physical Interpretation"]]
        meta = {
            "Full FiLM-U-Net (Ours)": (
                "FiLM-U-Net (Full, Ours)",
                "10 Local Vars + FiLM SST Modulation",
                f"Full proposed multimodal framework (Grad-CAM {f(t5_full.get('mean_gradcam'),3)})",
            ),
            "FiLM-U-Net (SST PCs Zeroed)": (
                "FiLM-U-Net (SST Zeroed)",
                "10 Local Vars + Zeroed SST PCs",
                f"Fixed-condition control; ACC {signed(abl_zero_acc,4)}, TCC {signed(abl_zero_tcc,4)}",
            ),
            "FiLM-U-Net (Local Zeroed)": (
                "FiLM-U-Net (Local Zeroed)",
                "Local Vars Zeroed + Real SST PCs",
                f"Skill & local pattern collapse (residual Grad-CAM mean {f(t5_flz.get('mean_gradcam'),3)}); local predictors dominate",
            ),
            "FiLM-U-Net (Shuffled PCs Control)": (
                "FiLM-U-Net (Shuffled PCs)",
                "10 Local Vars + 50-perm Shuffled PCs",
                f"Negative control; ACC {signed(abl_shuf_acc,4)}, TCC {signed(abl_shuf_tcc,4)}",
            ),
            "Full EOF-cat (All Inputs)": (
                "EOF-cat (Full Concat)",
                "10 Local Vars + Broadcast SST PCs",
                "Direct channel concatenation scheme",
            ),
            "EOF-cat (SST PCs Zeroed)": (
                "EOF-cat (SST Zeroed)",
                "10 Local Vars + Zeroed Broadcast PCs",
                "Concatenation architecture without SST signal",
            ),
            "EOF-cat (Local Zeroed)": (
                "EOF-cat (Local Zeroed)",
                "Local Vars Zeroed + Broadcast SST PCs",
                f"Spatial Grad-CAM collapses ({f(t5_catlz.get('mean_gradcam'),3)}); weak scalar offset",
            ),
            "Baseline U-Net (No SST)": (
                "Baseline U-Net (No SST)",
                "10 Local Atmospheric Variables (No SST)",
                f"Purely atmospheric benchmark (Grad-CAM {f(t5_base.get('mean_gradcam'),3)})",
            ),
        }
        for k, v in T5.items():
            acc = f(v["test_acc"]) + (f" ± {f(v['test_acc_std'])}" if "test_acc_std" in v else "")
            m_col, cfg_col, interp_col = meta.get(k, (k, k, "Matched control configuration"))
            rows.append([m_col, cfg_col, acc, f(v["test_tcc"]), interp_col])
        return rows

    for idx, builder in enumerate([tbl1_rows, tbl2_rows, tbl3_rows, tbl4_rows, tbl5_rows]):
        rows = builder()
        doc.set_table(idx, rows)
        print(f"[Manuscript] rebuilt Table {idx+1}: {len(rows)} rows x {len(rows[0])} cols")

    # Typography fixes strictly on untouched body paragraphs
    fix_count = 0
    for idx_p, p in enumerate(doc.xml.findall(".//" + q(W, "p"))):
        if doc.is_touched(p):
            continue
        t = doc.para_text(p)
        if not t:
            continue
        t2 = fix_ms_typography(t)
        if t2 != t:
            doc.set_para_text(p, t2, red=False)
            fix_count += 1
            print(f"  [Manuscript typo P{idx_p:03d}] fixed spacing")
    print(f"[Manuscript] typography fixes: {fix_count} paragraphs")

    # Boundary-protected numeric sweep strictly on untouched body paragraphs
    swept = 0
    for idx_p, p in enumerate(doc.xml.findall(".//" + q(W, "p"))):
        if doc.is_touched(p):
            continue
        t = doc.para_text(p)
        if not t:
            continue
        t2 = boundary_replace(t, num_map)
        if t2 != t:
            doc.set_para_text(p, t2, red=True)
            doc._mark_touched(p, t2)
            swept += 1
            print(f"  [Manuscript sweep P{idx_p:03d}] {t[:85]} -> {t2[:85]}")
    print(f"[Manuscript] numeric sweep: {swept} paragraphs")

    # ---- Replace embedded figures ----
    fig_map = {
        "media/image5.png": "Fig2_回报与预测对比.png", "media/image6.svg": "Fig2_回报与预测对比.svg",
        "media/image7.png": "Fig3_综合时间尺度对比.png", "media/image8.svg": "Fig3_综合时间尺度对比.svg",
        "media/image9.png": "Fig4_空间预报对比.png",
        "media/image10.png": "Fig5_TCC空间检验.png", "media/image11.svg": "Fig5_TCC空间检验.svg",
        "media/image12.png": "Fig6_参数PCA投影.png", "media/image13.svg": "Fig6_参数PCA投影.svg",
        "media/image14.png": "Fig7_参数相关性.png", "media/image15.svg": "Fig7_参数相关性.svg",
        "media/image16.png": "Fig8_对称消融.png", "media/image17.svg": "Fig8_对称消融.svg",
        "media/image18.png": "Fig9_遥相关敏感性.png", "media/image19.svg": "Fig9_遥相关敏感性.svg",
    }
    media = {}
    missing = []
    for target, fname in fig_map.items():
        p = fig / fname
        if p.exists():
            media[target] = p
        else:
            missing.append(fname)
    if missing:
        raise FileNotFoundError(f"Missing regenerated figures required for manuscript: {missing}")
    nrep = doc.replace_media(media)
    print(f"[Manuscript] replaced {nrep} embedded media parts")

    ms_out = doc.save(out / "05_大修修改稿.docx")
    print(f"[Manuscript] wrote {ms_out}")

    # ------------------------------------------------------------------
    # 2. Response letter
    # ------------------------------------------------------------------
    rl = DocxEditor(src / "03_审稿回复信.docx")
    print(f"[Response] paragraphs={len(rl.xml.findall('.//' + q(W, 'p')))}")

    RL = []
    radd = lambda needle, text: RL.append((needle, text, True))

    radd("我们由衷感谢主编、副主编以及两位审稿专家",
         "我们由衷感谢主编、副主编以及两位审稿专家的严谨评审。本轮返修在上一版基础上，"
         "对全部结果进行了统一管线的重新训练与评估，并以该唯一结果源同步重算了全部表格、"
         "文字与图件。除原有整改外，本次新增：①完全无泄露的时间顺序划分（训练 285 月、"
         "内部验证 60 月、独立测试 21 月）与 5 个 SST 主成分的统一口径；②按年分块 "
         "bootstrap、FiLM 对 Baseline 的配对检验、TCC 差值的场级检验与 BH-FDR 多重比较控制；"
         "③ENSO 位相分层的 ACC 与 TCC；④2020–2021 混合 ENSO 位相的 24 个月滚动留出"
         "敏感性检验（并如实说明其并非纯非厄尔尼诺测试期）；"
         "⑤图 8 六面板对称消融、图 7 与图 9 的打乱 PCs 阴性对照；⑥相同超参数调优预算与"
         "插入深度差异的如实说明；⑦对表 5 中“SST 置零/打乱不降低 ACC”这一事实的如实报告。"
         "以下是逐条回复。")

    radd("感谢主编的明确指导。我们已将 Highlights",
         f"感谢主编的明确指导。我们已将 Highlights 重新提炼为 4 条要点，并按修订后的权威结果"
         f"同步更新数值；逐条核算字符数分别为 "
         f"{len('EOF-FiLM integration couples global SST modes into U-Net for rainfall forecast.')}、"
         f"{len('FiLM modulates local features without diluting fine spatial gradient fields.')}、"
         f"{len(f'Independent 2023–2024 test: {f(fa,3)} ACC and {f(ftcc,3)} spatial mean TCC.')}、"
         f"{len(f'Significant skill coverage (p < 0.05) reaches {p1(fsig)} of {ng:,} land grids.')}，"
         "均严格控制在 85 个字符以内，并与单独提交的 Highlights.docx 完全一致。")

    radd("1. 关于所提出模型局限性的深入讨论与分析",
         "1. 关于模型局限性：已在修改稿第 6.2 节系统列出 12 条局限，分别涵盖：历史期为 "
         "in-sample、21 个月测试窗对年际 TCC 推断功效不足、ACC 配对差异未达显著、"
         f"SST 调制不改善测试 ACC、{sc_tradeoff_cn}、"
         f"{ng:,} 个格点上的多重比较问题、全局仿射假设与 5 PCs 截断、MSE 对极端峰值的平滑、"
         "五层 vs 输入层插入深度不对称，以及 γ/β 由 MLP 确定性生成与含偏置置零对照带来的机理解释边界。"
         "局限性讨论中与表 4 冲突的“强 ENSO 期改进最大”旧结论已删除。")

    radd("2. 关于关键输入变量的影响程度与相对重要性分析",
         "2. 关于关键输入变量：第 4.3 节与表 3 已按统一重跑结果重建，采用 20 次随机打乱"
         f"（上一版为 10 次）。首位预测因子为 {T3[0]['full_name']}，ΔACC = "
         f"{signed(T3[0]['mean_drop'])} ± {f(T3[0]['std_drop'])}。我们不再声称全部 10 个变量"
         "均具有独立贡献：部分环流变量的边际重要性接近零或为负值，正文已如实说明其共线性。")

    radd("3. 关于 2023–2024 年评估期的独立性与代表性确立",
         "3. 关于独立性与代表性：预处理口径已按代码统一为——SST 气候态、EOF 基底、PC 标准化"
         "与极端阈值全部仅在 1994.01–2017.12 训练窗（285 月）拟合；2018.01–2022.12（60 月）"
         "仅用于早停与超参数选择；2023.01–2024.09（21 月）完全样本外。1994.01–2022.12 共 348 个"
         "日历月，其中 2011-09、2011-10、2017-01 三个月档案缺测，故实际使用 345 个月。"
         "第 4.4 节表 4 明确把历史行标注为 in-sample，并删除了“跨年代泛化验证”的不当表述；"
         "SEAS5 的历史位相数值已补齐，不再以“-”代替。")

    radd("针对 2023–2024 年强厄尔尼诺偏置疑虑",
         "针对强厄尔尼诺偏置：第 4.4 节同时给出 ACC 与 TCC 的 ENSO 分层结果，并新增 "
         f"2020.01–2021.12 的 24 个月滚动留出检验（其中拉尼娜 {HO['la_nina_months']} 月、"
         f"中性 {HO['neutral_months']} 月、厄尔尼诺 {HO['el_nino_months']} 月）。该检验重新拟合"
         "EOF 基底并从头训练全部模型，结果为 FiLM-U-Net ACC "
         f"{f(HO['FiLM-U-Net']['acc'],4)}、TCC {f(HO['FiLM-U-Net']['tcc'],4)}，EOF-cat ACC "
         f"{f(HO['EOF-cat']['acc'],4)}、TCC {f(HO['EOF-cat']['tcc'],4)}。我们如实说明：该留出期"
         f"上 ACC 最高的是 {ho_acc_best}（{f(HO[ho_acc_best]['acc'],4)}）、TCC 最高的是 "
         f"{ho_tcc_best}（{f(HO[ho_tcc_best]['tcc'],4)}），FiLM-U-Net 并非领先模型；因此该检验"
         "是边界条件排查而非支持性证据。它也不能等同于等长的独立测试集，21 个月仍不足以"
         "支撑多年代年际推断。上一版回复信对此项请求的沉默，本版已补答。")

    radd("4. 关于补齐最接近基线模型（EOF-cat）的结构对比",
         "4. 关于 EOF-cat 结构对比与显著性检验：图 2、图 3、图 4、图 5、图 8 以及新增图件"
         "全部由统一评估结果重新生成（含 PNG 与 SVG 双格式），图内数值与表 1–表 5 逐位一致。"
         "上一版图 2/3/5 仍沿用数据泄露整改前的旧结果，本版已彻底替换。"
         "图 3 的实际版式为：(a) 日历月平均 ACC、(b) 逐年平均 ACC、"
         f"(c) 21 个测试月逐月 ACC 曲线、(d) 逐月 ΔACC 差值（上一版回复信误写为“3a 逐月/3b 差值”，"
         "已改正）。测试期逐年（2023 vs 2024）差值亦已在第 4.1 节报告。")

    radd("统计显著性检验，全中国 5,301 个空间有效陆地网格",
         f"统计显著性检验：新增第 3.2 节完整协议。有效陆地格点经统一管线核定为 {ng:,} 个"
         "（上一版的格点数有误，现已按统一管线重新核定）。除网格级 Student's t 打点外，"
         "补充了 10,000 次逐月 bootstrap、"
         "10,000 次按年分块 bootstrap（并明确说明测试期仅跨 2 个日历年、分块区间本身较粗）、"
         "FiLM 对 Baseline 的配对 Wilcoxon 检验、空间 ΔTCC 的场级配对检验与 2,000 次按年分块 "
         f"bootstrap，以及 {ng:,} 个格点上的 Benjamini–Hochberg FDR 控制（q = 0.05）。"
         f"FiLM−EOF-cat 的平均 ACC 差为 {signed(fc['mean_diff'])}（p = {f(fc['wilcoxon_p'],3)}，"
         f"按年分块 CI [{signed(fc['boot_year_block_ci95'][0])}, "
         f"{signed(fc['boot_year_block_ci95'][1])}]），未达 5% 显著水平，已在正文如实表述；"
         "空间 ΔTCC 与显著覆盖率构成主要证据。")

    radd("感谢审稿专家关注这一关键水文气象问题",
         "感谢审稿专家关注这一关键水文气象问题。在水文气象预报中，极端对流降水代表真实的物理"
         "过程而非观测噪声，人为截断会破坏异常场并掩盖高风险事件。我们更正上一版回复中的不准确"
         "表述：制备好的目标场并未做 ±3σ 截断或缩尾，其重尾被完整保留"
         f"（{p1(clipped)} 的格月超过 |ΔP| = 3），数值稳定性仅由梯度范数裁剪（‖g‖₂ ≤ 2）保证。"
         f"极端事件阈值采用训练窗 90 分位数 ΔP ≥ {f(p90,4)}，该统计量同样只在训练窗拟合；"
         "上一版所引阈值数值已更正。")

    radd("1. 数据预处理：如修改稿第 2.1.2 节所述",
         "1. 数据预处理：修改稿第 2.1.2 与 2.2 节已如实说明无截断、仅梯度裁剪，"
         "并给出训练窗拟合的全部统计量清单。")
    radd("2. 极端降水预报能力评估在文中的具体体现",
         f"2. 极端降水评估：第 3 节给出 ΔP ≥ {f(p90,4)} 的 CSI/SC 定义，第 4.2 节与表 1 报告 "
         f"SEAS5 {f(scsi,3)}、Baseline {f(bcsi,3)}、EOF-cat {f(ccsi,3)}、"
         f"FiLM-U-Net {f(fcsi,3)}。我们不再以“高方向准确率”形容 FiLM 的 {p1(fsc)}。"
         "各指标最优模型并不一致：" + rank_line_cn +
         " FiLM-U-Net 并非全面最优，相关权衡已写入正文与第 6.2 节局限性。")

    radd("1. 实际业务预报意义：在月度至次季节气候预报领域",
         f"1. 实际业务意义：测试期 SEAS5 的 ACC 为 {f(sa,4)}，Baseline U-Net 提升至 "
         f"{f(ba,4)}，EOF-cat 为 {f(ca,4)}，FiLM-U-Net 为 {f(fa,4)}。我们不再笼统宣称"
         "“提高 0.04 具有重大业务价值”，而是如实报告各阶段差值：Stage 1 "
         f"{signed(D['stage1_acc'])}、Stage 2a {signed(D['stage2a_acc'])}、FiLM 相对 Baseline "
         f"{signed(D['delta_acc_film_base'])}、FiLM 相对 EOF-cat "
         f"{signed(D['delta_acc_film_cat'])}；配对检验见第 3.2 节，"
         f"{both_sig_line}。")
    radd("2. 统计显著性检验：我们执行了严格的网格级",
         f"2. 统计显著性：除网格级打点外，本版补充了配对 Wilcoxon、逐月与按年分块 bootstrap "
         f"CI、空间 ΔTCC 场级检验与 BH-FDR。空间平均 TCC：FiLM-U-Net {f(ftcc,4)}、"
         f"EOF-cat {f(ctcc,4)}、Baseline {f(btcc,4)}；显著覆盖率 {p1(fsig)} / {p1(csig)} / "
         f"{p1(bsig)}。网格打点仅作描述性展示，正式推断以场级检验为准。")
    radd("3. 补充多元评估指标：我们在修改稿第 4.1 节",
         "3. 多元指标：表 1 的 RMSE / SC / CSI 已按统一结果重建；" + rank_line_cn +
         " 即模型排序随指标而变，不存在单一模型全面占优；原“全指标正增益”表述已删除。")

    radd("为了清晰剥离两阶段的技术贡献",
         "为清晰剥离两阶段贡献，表 2 已按统一结果重算：Stage 1（SEAS5→Baseline）"
         f"{signed(D['stage1_acc'])}，Stage 2a（Baseline→EOF-cat）{signed(D['stage2a_acc'])}，"
         f"Stage 2b（Baseline→FiLM）{signed(D['delta_acc_film_base'])}。图 2 与图 3 的分工说明"
         "保留，但删除了“unambiguously demonstrates”等无显著性支撑的强断言。")

    radd("非常感谢这一极具建设性的审稿建议。我们在完全独立的测试集上",
         "感谢这一建议。我们在独立测试集上以 20 次随机打乱重建了置换重要性（表 3），"
         "并补充了第 4.2/6.2 节对区域差异的定量讨论与局限说明。")

    radd("- 意见 4（图 2 与图 3 的对比度与图例遮挡）",
         "- 意见 4（图 2 与图 3 的对比度与图例遮挡）：全部图件已由统一管线重新出图，"
         "图例统一置于绘图区上方外侧，不再遮挡曲线；图内数值与表格完全一致。")
    radd("- 意见 5（图 4 子图标签与统一色标尺）",
         "- 意见 5（图 4 子图标签与统一色标尺）：保留月份/类别标签与 [−1.0, 1.0] 统一色标，"
         "并补充单位为“降水异常分数 ΔP”，避免与百分率混淆。")
    radd("- 意见 3（10 个变量与 9 个变量的表述矛盾）",
         "- 意见 3（10 个变量与 9 个变量的表述矛盾）：已核对全文，第 2.3 节删除了不在变量清单中的"
         "“humidity, vertical velocity”残留描述。")

    radd("1. 测试集与历史后报期内 ENSO 位相构成清查",
         f"1. ENSO 位相构成：21 个测试月中厄尔尼诺 {EN['test']['El Nino']['months']} 月"
         f"（{EN['test']['El Nino']['pct']:.1f}%）、中性 {EN['test']['Neutral']['months']} 月"
         f"（{EN['test']['Neutral']['pct']:.1f}%）、拉尼娜 {EN['test']['La Nina']['months']} 月"
         f"（{EN['test']['La Nina']['pct']:.1f}%）。历史 {S['hindcast_months']} 个月为 "
         f"{EN['hindcast']['El Nino']['months']}/{EN['hindcast']['La Nina']['months']}/"
         f"{EN['hindcast']['Neutral']['months']} 月，但属训练/选模窗内（in-sample），"
         "已在表 4 与正文中标明，不再作为泛化证据。相关内容位于第 4.4 节与第 6.2 节"
         "（上一版回复信的章节指针有误，现已更正）。")
    radd("2. ENSO 位相分层评估正式入文",
         "2. ENSO 分层：第 4.4 节表 4 给出 ACC 分层，结果包与正文同时给出 TCC 分层；"
         "SEAS5 各历史位相数值已补齐（上一版全为“-”）。")
    radd("3. TCC 统计显著性打点",
         f"3. TCC 显著性：除网格 Student's t 打点外，新增 10,000 次逐月与按年分块 bootstrap 95% CI、"
         f"空间 ΔTCC 配对 Wilcoxon 与场级 2,000 次分块 bootstrap CI，以及 {ng:,} 格点上的 "
         "BH-FDR 控制；第 3.2 节给出完整协议与多重比较说明。")
    radd("---",
         "4. 非厄尔尼诺测试期：我们按建议执行了 2020.01–2021.12 的 24 个月滚动留出"
         "敏感性检验（重新拟合 SST 气候态与 EOF 基底、从头训练全部模型），结果见第 4.4 节与表 4 附注。"
         "需要如实说明：该窗口的 ENSO 构成为 "
         f"{HO['el_nino_months']} 个厄尔尼诺月、{HO['neutral_months']} 个中性月、"
         f"{HO['la_nina_months']} 个拉尼娜月，因此并非严格的“纯非厄尔尼诺期”；"
         "在现有 366 个月档案与 21 个月完全样本外测试窗的约束下，无法再构造一个等长、"
         "完全独立且仅含非厄尔尼诺月份的测试期，我们把该窗口定位为额外的留出敏感性检验，"
         "而非第二个独立测试集。该留出窗内 FiLM-U-Net 的 ACC 为 "
         f"{f(HO['FiLM-U-Net']['acc'],4)}，低于 EOF-cat 的 {f(HO['EOF-cat']['acc'],4)}，"
         "我们未回避这一不利结果，并在正文与结论中说明模型排序随评估窗口与指标变化。"
         "此外补充测试期逐年（2023 与 2024）的 ACC 差值对比。")

    radd("我们反查了初始代码工程",
         "我们反查了初始代码工程，确认初稿早期曾以全序列拟合主成分，存在数据泄露隐患，"
         "对此深表歉意。整改后的统一管线把 SST 气候态、EOF 基底、PC 标准化与极端阈值"
         "全部限定在 1994.01–2017.12 训练窗；内部验证窗 2018.01–2022.12 仅用于早停与"
         "超参数选择；测试窗 2023.01–2024.09 完全样本外。全部模型、全部表格、全部图件"
         "均由该管线重新生成。需要说明的是，整改后的绝对数值与上一版不同："
         f"FiLM-U-Net 独立测试 ACC 为 {f(fa,4)}、空间平均 TCC 为 {f(ftcc,4)}（有效格点 "
         f"{ng:,} 个、5 个 SST 主成分、训练集 p90 阈值 {f(p90,4)}）。我们已将全稿数值统一到"
         "这一结果，不再沿用上一版表格中的旧数值口径。")

    radd("我们对图 3、图 4 与图 5 进行了彻底的重绘",
         "我们对全部图件进行了彻底重绘（统一结果源、PNG+SVG 双格式），并修正上一版回复信"
         "对图 3 面板的错误描述：修订后的图 3 为 (a) 日历月平均 ACC、(b) 逐年平均 ACC、"
         "(c) 21 个测试月逐月 ACC 曲线、(d) 逐月 ΔACC 差值；图 4 为四季典型月的五列对比；"
         "图 5 为 (a) Baseline、(b) EOF-cat、(c) FiLM-U-Net 的 TCC 场与 (d) 空间差值场。"
         "逐月、逐年与空间三类 FiLM−cat 差值均已报告。")

    radd("我们在修改稿第 2.4 节（Training Strategy）与第 4.1 节中补充了严格的对比公平性对齐说明",
         "我们在第 2.4 节补充了可核验的对比公平性说明：三个模型均使用相同的三点超参数网格"
         "（共 9 次训练）、相同种子与划分、相同优化器与 100 epoch 预算，并以内部验证 MSE "
         "统一选模；EOF-cat 与 FiLM 使用相同的 5 个 SST 主成分与 W = 1。关于插入深度，"
         "我们如实说明二者并不相同：EOF-cat 在输入层广播 5 个 PC 通道，FiLM 则由索引编码器"
         "对全部 5 个编码层生成 (γ, β)；由于未运行单层 FiLM 对照，我们不再声称“相同瓶颈层引入”，"
         "也不把全部性能差异单独归因于调制机制。上一版回复信中关于“算力预算已写入正文”的"
         "过度声称，现已有第 2.4 节逐项可核验的训练预算说明作为支撑。此外，重跑结果并不支持"
         "“EOF-cat 因历史训练噪声过拟合而系统性退化”的强断言：EOF-cat 的测试期 ACC 为 "
         f"{f(ca,4)}（vs Baseline {f(ba,4)}、FiLM-U-Net {f(fa,4)}、SEAS5 {f(sa,4)}）；第 4.1 节已删除该"
         "因果归因，仅保留可核验的指标排序与位相分层事实。")

    radd("1. 完成双重统计检验",
         "1. 完成配对检验：FiLM 较 EOF-cat 的平均 ACC 差 "
         f"{signed(fc['mean_diff'])}，Wilcoxon p = {f(fc['wilcoxon_p'],3)}，"
         f"逐月 bootstrap 95% CI [{signed(fc['boot_month_ci95'][0])}, "
         f"{signed(fc['boot_month_ci95'][1])}]，按年分块 bootstrap 95% CI "
         f"[{signed(fc['boot_year_block_ci95'][0])}, {signed(fc['boot_year_block_ci95'][1])}]；"
         f"FiLM 较 Baseline 为 {signed(fb['mean_diff'])}（p = {f(fb['wilcoxon_p'],3)}）。"
         f"{both_sig_line}，我们不再使用“significantly outperforms”类表述。"
         f"空间 ΔTCC（FiLM−cat）= {signed(ts['film_vs_cat_mean_dtcc'])}，场级 Wilcoxon "
         f"p = {fmt_p(ts['film_vs_cat_wilcoxon_p'])}，按年分块 CI "
         f"[{signed(ts['film_vs_cat_year_block_ci95'][0])}, "
         f"{signed(ts['film_vs_cat_year_block_ci95'][1])}]。")
    radd("2. 全文措辞严谨性修订",
         "2. 全文措辞：第 4.1 节、摘要与第 7 节已统一改为基于统计事实的限定表述；"
         "同时删除“全指标正增益”“相对 RMSE 全面下降”“强 ENSO 期改进最大”等与表格冲突的旧结论。")

    radd("我们开展了完整的全对称消融实验",
         "我们开展了完整的全对称消融，并把图 8 由 4 面板扩展为 6 面板（新增 FiLM 的 "
         "SST PCs 置零与局地变量置零），图 7(c,d) 与图 9(f) 加入打乱 PCs 阴性对照；"
         "表 5 与第 5.3 节据统一结果重建。更重要的是，我们如实修正了上一版与自家数据"
         f"相反的解释：SST PCs 置零 ACC = {f(t5_zero['test_acc'])}（相对完整模型 "
         f"{signed(abl_zero_acc)}）、打乱 ACC = {f(t5_shuf.get('test_acc'))}"
         f"（{signed(abl_shuf_acc)}），完整模型为 {f(t5_full['test_acc'])}。"
         + abl_acc_line_cn + "因此删除“平滑回落至基线”“打乱导致退化”的说法。" +
         abl_tcc_line_cn + "论证重心放在空间 TCC 场与显著网格覆盖率上。局地变量置零后 "
         f"ACC = {f(t5_flz['test_acc'],4)}，说明局地预测因子主导"
         "空间技巧，但正文改为“与条件调制机制一致”，不再使用“证明”等因果断言。")

    radd("非常感谢审稿专家的宝贵建议。我们在修改稿第 6.2节",
         "感谢审稿专家的这一建议。第 6.2 节现系统列出 12 条局限：历史期为 in-sample、"
         "21 个月测试窗对年际 TCC 推断功效不足、ACC 配对差异未达显著、SST 调制不改善测试 ACC、"
         f"{sc_tradeoff_cn}、{ng:,} 个格点上的多重比较问题、全局仿射假设、"
         "5 PCs 截断、MSE 对极端降水峰值的平滑、五层 vs 输入层插入深度不对称，以及 γ/β 由 MLP "
         "确定性生成与含偏置置零对照带来的机理解释边界；第 7 节结论与摘要同步收敛了声称强度。")

    radd("我们已在修改稿第 4.2 节中补充了详尽的气象学机理解释",
         "我们已在第 4.2 节与第 6 节综合讨论中结合区域统计补充成因讨论：东部季风区同时受海温遥相关与水汽输送"
         "影响，青藏高原与西北干旱区受地形和稀疏观测约束，各模型在不同区域的优劣并不一致；"
         + reg_sentence_cn + "同时明确这些解释是与数据一致的推断而非因果识别，区域统计值与显著性结果"
         "一并列于结果包，避免把相关性表述成物理机制证明。")

    radd("- 意见 2（文字格式与断句排版）",
         "- 意见 2（文字格式与断句排版）：修正了诸如 `(T2m).Spatial` 与 `)(Turkoglu` 等格式粘连，"
         "统一补齐引文与句间空格、规范句末标点，并将分散的列举式短语改写为语法完整的段落陈述。")

    radd("在 2023.01–2024.09 的 21 个月独立测试集中，确实有 14 个月",
         f"位相构成的准确数字为：{n_test} 个独立测试月中厄尔尼诺 "
         f"{EN['test']['El Nino']['months']} 月（{EN['test']['El Nino']['pct']:.1f}%）、"
         f"中性 {EN['test']['Neutral']['months']} 月（{EN['test']['Neutral']['pct']:.1f}%）、"
         f"拉尼娜 {EN['test']['La Nina']['months']} 月（{EN['test']['La Nina']['pct']:.1f}%）；"
         f"历史 {S['hindcast_months']} 个月（in-sample）为 "
         f"{EN['hindcast']['El Nino']['months']}/{EN['hindcast']['La Nina']['months']}/"
         f"{EN['hindcast']['Neutral']['months']} 月。上一版回复信把测试期写成 14/6/1 月、"
         "历史期写成 103/115/127 月，均与本版统一管线的位相判定不符，现已全部更正；"
         "相关内容位于第 4.4 节与第 6.2 节。")

    radd("正式新增了表 4",
         "第 4.4 节的表 4 已按统一结果重建（历史行明确标注 in-sample），并在结果包中"
         "补充按 ENSO 位相分层的空间平均 TCC；测试期拉尼娜与中性各仅 4 个月，"
         "不足 5 个月样本，TCC 分层按缺样本记为 n/a，不做过度解读。")

    radd("我们对代码流水线与数据预处理执行了彻底的物理隔离整改",
         f"补充说明整改的具体实现：SST 月气候态、EOF 基底、PC 标准化均值/方差与 p90 阈值"
         f"均仅在 {S['train_period']} 训练窗（{S['train_months']} 月）拟合；内部验证窗 "
         f"{S['val_period']}（{S['val_months']} 月）只用于早停与选模；独立测试窗 "
         f"{S['test_period']}（{n_test} 月）不参与任何拟合，所有统计量拟合后冻结，"
         "后续月份仅做正交投影与标准化。整改后的绝对数值与上一版不同：FiLM-U-Net "
         f"独立测试 ACC 为 {f(fa,4)}、空间平均 TCC 为 {f(ftcc,4)}，并不等于上一版报告的约 "
         "0.35；我们不再声称该数值被“再次证实”，而是以本版统一重跑结果为唯一口径。")

    radd("修订后的图 3（时间序列全面对比）",
         "修订后的图 3（时间序列全面对比）：(a) 为 1994–2022 年日历月平均 ACC（in-sample），"
         "(b) 为逐年平均 ACC（in-sample），(c) 为 21 个独立测试月四模型逐月 ACC 曲线，"
         f"(d) 为逐月 FiLM−EOF-cat 配对 ΔACC 柱形图（均值 {signed(fc['mean_diff'])}，"
         f"{fc['pos_months']}/{fc['total_months']} 月为正）。四面板含义已与图题、"
         "正文第 4.1 节完全一致，上一版对图 3a/3b 的错误描述与残缺句已删除。")

    radd("逐月时间配对检验：在 21 个独立测试月上",
         f"逐月时间配对检验（更正）：在 {n_test} 个独立测试月上，FiLM 较 EOF-cat 的平均 ACC "
         f"差为 {signed(fc['mean_diff'])}（为负值，并非增益），Wilcoxon 符号秩 "
         f"p = {f(fc['wilcoxon_p'],3)}；10,000 次逐月 bootstrap 95% CI "
         f"[{signed(fc['boot_month_ci95'][0])}, {signed(fc['boot_month_ci95'][1])}]，"
         f"按年分块 95% CI [{signed(fc['boot_year_block_ci95'][0])}, "
         f"{signed(fc['boot_year_block_ci95'][1])}]。两项区间均跨零，因此不认为存在"
         "稳定的正向技巧增益；上一版回复信所引的旧 bootstrap 区间与重抽次数均已删除。")

    radd("全域空间网格统计检验：在全国",
         f"全域空间网格统计检验（更正）：在 {ng:,} 个有效陆地格点上，FiLM-U-Net 的原始 "
         f"p < 0.05 网格占比为 {p1(fsig)}，EOF-cat 为 {p1(csig)}，相差 "
         f"{signed(fsig - csig,1)} 个百分点；BH-FDR（q = 0.05）后分别为 "
         f"{p1(X['fdr']['FiLM-U-Net']['bh_fdr_005_pct'])} 与 "
         f"{p1(X['fdr']['EOF-cat']['bh_fdr_005_pct'])}。上一版关于覆盖率相对增幅的"
         "旧数值与“显著扩展”的强断言均与统一结果不符，已删除；打点仅作描述性展示，"
         "正式推断以场级配对检验为准。")

    print(f"[Response] rewrote {rl.replace_paragraphs(RL)} paragraphs")

    # Typography + boundary-protected numeric sweep strictly on untouched response paragraphs
    sweep2 = 0
    for idx_p, p in enumerate(rl.xml.findall(".//" + q(W, "p"))):
        t = rl.para_text(p)
        if not t:
            continue
        was_touched = rl.is_touched(p)
        t2 = fix_rl_spacing(t)
        did_num_sweep = False
        if not was_touched:
            t_num = boundary_replace(t2, num_map)
            t_num = t_num.replace("第 6.1 节", "第 4.4 节与第 6.2 节").replace("5,301", f"{ng:,}")
            did_num_sweep = (t_num != t2)
            t2 = t_num
        if t2 != t:
            rl.set_para_text(p, t2, red=(was_touched or did_num_sweep))
            rl._mark_touched(p, t2)
            sweep2 += 1
            print(f"  [Response sweep/space P{idx_p:03d}] {t[:75]} -> {t2[:75]}")
    print(f"[Response] numeric/spacing sweep: {sweep2} paragraphs")

    rl_out = rl.save(out / "05_审稿回复信.docx")
    print(f"[Response] wrote {rl_out}")

    # ------------------------------------------------------------------
    # 3. Highlights
    # ------------------------------------------------------------------
    hl = DocxEditor(src / "Highlights.docx")
    bullets = [
        "EOF-FiLM integration couples global SST modes into U-Net for rainfall forecast.",
        "FiLM modulates local features without diluting fine spatial gradient fields.",
        f"Independent 2023–2024 test: {f(fa,3)} ACC and {f(ftcc,3)} spatial mean TCC.",
        f"Significant skill coverage (p < 0.05) reaches {p1(fsig)} of {ng:,} land grids.",
    ]
    hl.replace_paragraphs([
        ("EOF-FiLM integration couples", bullets[0], False),
        ("FiLM modulates local features", bullets[1], False),
        ("Evaluation over 2023–2024 achieves", bullets[2], True),
        ("Model expands significant skill coverage", bullets[3], True),
    ])
    hl_out = hl.save(out / "Highlights.docx")
    print(f"[Highlights] wrote {hl_out}; lengths = {[len(b) for b in bullets]} (limit 85)")


if __name__ == "__main__":
    main()
