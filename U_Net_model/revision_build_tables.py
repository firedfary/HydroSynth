# -*- coding: utf-8 -*-
"""
Build the single authoritative results bundle from the leak-free unified evaluation.

Usage (nc conda env):
    python revision_build_tables.py --out "<delivery folder>"
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.paths import SubprojectPaths  # noqa: E402


def f4(x):
    return "-" if x is None else f"{float(x):.4f}"


def f3(x):
    return "-" if x is None else f"{float(x):.3f}"


def pct(x):
    return "-" if x is None else f"{float(x):.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Delivery folder (absolute path)")
    args = ap.parse_args()

    exp_dir = Path(SubprojectPaths(__file__).get_exp_dir("unified_eval"))
    metrics_file = exp_dir / "unified_metrics.json"
    if not metrics_file.exists():
        raise SystemExit(f"Cannot find {metrics_file}; run run_unified_eval.py first.")

    with metrics_file.open("r", encoding="utf-8") as fp:
        M = json.load(fp)

    out = Path(args.out)
    tbl_dir = out / "结果表"
    val_dir = out / "校验"
    tbl_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    split = M["split"]
    t1 = M["table1_metrics"]
    st = M["paired_stat_tests"]
    ts = st["test_tcc_spatial"]
    ens = M["table4_enso_stratification"]
    hold = M["holdout_2020_2021_sensitivity"]
    t5 = M["table5_symmetric_ablation"]
    t3 = M["table3_permutation_importance"]
    reg = M["regional_metrics"]
    f7 = M["fig7_control_summary"]
    f8 = M.get("fig8_gradcam_diagnostics", {})
    f9 = M["fig9_sensitivity_summary"]
    tuning = M["tuning_log"]

    order = ["SEAS5", "Baseline", "EOF-cat", "FiLM-U-Net"]
    lines = []
    lines.append("# EJRH-D-26-03375 权威结果总表（唯一数值源）")
    lines.append("")
    lines.append("来源：`run_unified_eval.py` 在标准工作区代码中的统一无泄露重跑，")
    lines.append(f"产物：`{metrics_file}`。任何图、正文、摘要、Highlights、回复信的数值必须与本表一致。")
    lines.append("")
    lines.append("## 0. 数据划分与实验口径")
    lines.append("")
    lines.append("| 项目 | 值 |")
    lines.append("|---|---|")
    for k, label in [
        ("train_period", "训练期"), ("train_months", "训练月数"),
        ("val_period", "内部验证期"), ("val_months", "内部验证月数"),
        ("hindcast_period", "历史期（训练+内部验证，in-sample）"), ("hindcast_months", "历史月数"),
        ("test_period", "独立测试期"), ("test_months", "独立测试月数"),
        ("valid_china_land_grids", "有效中国陆地格点"), ("n_pcs", "SST 主成分个数"),
        ("train_p90_anomaly_threshold", "训练集 p90 异常阈值"),
        ("pct_grid_months_abs_gt_3", "|ΔP|>3 的格月占比"),
    ]:
        lines.append(f"| {label} | {split[k]} |")
    lines.append(f"| PCA 累计解释方差 | {float(split['pca_total_explained_variance'])*100:.2f}% |")
    lines.append(f"| 归档缺测月份 | {', '.join(split['excluded_archive_months'])} |")
    lines.append("")
    lines.append("## 1. 表 1 主结果（独立测试期）")
    lines.append("")
    lines.append("| 模型 | 历史 ACC (in-sample) | 内部验证 ACC | 测试 ACC | 2023 ACC | 2024 ACC | 测试 TCC | 显著格点占比 | RMSE | SC | CSI |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in order:
        d = t1[m]
        lines.append(
            f"| {m} | {f4(d['hindcast_acc'])} | {f4(d['val_acc'])} | {f4(d['test_acc'])} | "
            f"{f4(d['test_acc_2023'])} | {f4(d['test_acc_2024'])} | {f4(d['test_tcc'])} | "
            f"{pct(d['test_sig_pct'])} ({d['test_n_sig']}/{d['test_n_valid']}) | "
            f"{f4(d['test_rmse'])} | {pct(d['test_sc_pct'])} | {f3(d['test_csi'])} |"
        )
    lines.append("")
    lines.append("## 2. 表 2 阶梯式贡献分解（独立测试期）")
    lines.append("")
    dcat = t1["FiLM-U-Net"]["test_acc"] - t1["EOF-cat"]["test_acc"]
    dbase = t1["FiLM-U-Net"]["test_acc"] - t1["Baseline"]["test_acc"]
    dtcc_cat = ts["film_vs_cat_mean_dtcc"]
    dtcc_base = ts["film_vs_base_mean_dtcc"]
    s1_acc = t1["Baseline"]["test_acc"] - t1["SEAS5"]["test_acc"]
    s2a_acc = t1["EOF-cat"]["test_acc"] - t1["Baseline"]["test_acc"]
    s2a_tcc = t1["EOF-cat"]["test_tcc"] - t1["Baseline"]["test_tcc"]
    lines.append("| 阶段 | 模型过渡 | ΔACC | ΔTCC |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Stage 1 局地降尺度 | SEAS5 → Baseline | {s1_acc:+.4f} | - |")
    lines.append(f"| Stage 2a 直接拼接 | Baseline → EOF-cat | {s2a_acc:+.4f} | {s2a_tcc:+.4f} |")
    lines.append(f"| Stage 2b FiLM 调制 | Baseline → FiLM-U-Net | {dbase:+.4f} (vs Cat {dcat:+.4f}) | {dtcc_base:+.4f} (vs Cat {dtcc_cat:+.4f}) |")
    lines.append("")
    lines.append("## 3. 配对统计检验（含按年分块 bootstrap）")
    lines.append("")
    lines.append("| 检验 | 均值差 | 正增益月数 | Wilcoxon p | 逐月 bootstrap 95% CI | 按年分块 bootstrap 95% CI |")
    lines.append("|---|---|---|---|---|---|")
    for k, v in st.items():
        if k == "test_tcc_spatial":
            continue
        ci_m = v["boot_month_ci95"]
        ci_y = v["boot_year_block_ci95"]
        lines.append(
            f"| {k} | {v['mean_diff']:+.4f} | {v['pos_months']}/{v['total_months']} | "
            f"{v['wilcoxon_p']:.4f} | [{ci_m[0]:+.4f}, {ci_m[1]:+.4f}] | "
            f"[{ci_y[0]:+.4f}, {ci_y[1]:+.4f}] |"
        )
    lines.append("")
    lines.append("### 空间 TCC 差值检验")
    lines.append("")
    lines.append(f"- FiLM − EOF-cat 平均 ΔTCC = {ts['film_vs_cat_mean_dtcc']:+.4f}，Wilcoxon p = {ts['film_vs_cat_wilcoxon_p']:.3g}，按年分块 bootstrap 95% CI = [{ts['film_vs_cat_year_block_ci95'][0]:+.4f}, {ts['film_vs_cat_year_block_ci95'][1]:+.4f}]")
    lines.append(f"- FiLM − Baseline 平均 ΔTCC = {ts['film_vs_base_mean_dtcc']:+.4f}，Wilcoxon p = {ts['film_vs_base_wilcoxon_p']:.3g}，按年分块 bootstrap 95% CI = [{ts['film_vs_base_year_block_ci95'][0]:+.4f}, {ts['film_vs_base_year_block_ci95'][1]:+.4f}]")
    lines.append("")
    lines.append("## 4. 表 3 置换特征重要性（独立测试期，20 次打乱）")
    lines.append("")
    lines.append("| 排名 | 变量 | ΔACC 均值 | ΔACC 标准差 |")
    lines.append("|---|---|---|---|")
    for i, it in enumerate(t3, 1):
        lines.append(f"| {i} | {it['full_name']} | {it['mean_drop']:+.4f} | {it['std_drop']:.4f} |")
    lines.append("")
    lines.append("## 5. 表 4 ENSO 位相分层（ACC）")
    lines.append("")
    lines.append("| 时期 / 位相 | 月数 | 占比 | SEAS5 | Baseline | EOF-cat | FiLM | ΔBase | ΔCat |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for period, label in [("hindcast", "历史（in-sample）"), ("test", "独立测试")]:
        for phase in ["El Nino", "La Nina", "Neutral", "All"]:
            d = ens[period][phase]
            if d["months"] == 0:
                continue
            cells = " | ".join(f4(d.get(mm)) for mm in order)
            lines.append(
                f"| {label}: {phase} | {d['months']} | {d['pct']:.1f}% | {cells} | "
                f"{d.get('gain_vs_base', 0):+.4f} | {d.get('gain_vs_cat', 0):+.4f} |"
            )
    lines.append("")
    lines.append("## 6. 2020–2021 非厄尔尼诺滚动留出敏感性检验")
    lines.append("")
    lines.append(f"留出月数 {hold['months']}；厄尔尼诺 {hold['el_nino_months']} 月，拉尼娜 {hold['la_nina_months']} 月，中性 {hold['neutral_months']} 月。")
    lines.append("")
    lines.append("| 模型 | ACC | TCC | 显著格点占比 | RMSE | SC |")
    lines.append("|---|---|---|---|---|---|")
    for m in order:
        d = hold[m]
        lines.append(f"| {m} | {f4(d['acc'])} | {f4(d['tcc'])} | {pct(d['sig_pct'])} | {f4(d['rmse'])} | {pct(d['sc_pct'])} |")
    lines.append("")
    lines.append("## 7. 表 5 对称消融与阴性对照（独立测试期）")
    lines.append("")
    lines.append("| 配置 | Test ACC | 空间平均 TCC | 显著格点占比 | RMSE | SC | Grad-CAM均值 |")
    lines.append("|---|---|---|---|---|---|---|")
    for k, v in t5.items():
        cam_s = f4(v["mean_gradcam"]) if "mean_gradcam" in v else "-"
        lines.append(
            f"| {k} | {f4(v['test_acc'])}"
            + (f" ± {f4(v['test_acc_std'])}" if "test_acc_std" in v else "")
            + f" | {f4(v['test_tcc'])} | {pct(v.get('sig_pct'))} | "
            + (f4(v['rmse']) if 'rmse' in v else "-") + " | "
            + (pct(v['sc_pct']) if 'sc_pct' in v else "-") + f" | {cam_s} |"
        )
    lines.append("")
    lines.append("## 8. 分区域技巧（独立测试期）")
    lines.append("")
    lines.append("| 区域 | 格点数 | 模型 | ACC | TCC | 显著格点占比 | RMSE | SC |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r_name, rd in reg.items():
        for m in order:
            d = rd[m]
            lines.append(f"| {r_name} | {rd['n_grids']} | {m} | {f4(d['test_acc'])} | {f4(d['test_tcc'])} | {pct(d['sig_pct'])} | {f4(d['rmse'])} | {pct(d['sc_pct'])} |")
    lines.append("")
    lines.append("## 9. 阴性对照与超参数调优预算")
    lines.append("")
    lines.append(f"- Fig. 7 γ 真实 |r| 均值（PC1–PC5）：{[round(x, 4) for x in f7['gamma_real_mean_abs']]}；shuffled 均值：{[round(x, 4) for x in f7['gamma_shuf_mean_abs']]}；shuffled p95：{[round(x, 4) for x in f7['gamma_shuf_p95_abs']]}")
    lines.append(f"- Fig. 7 β 真实 |r| 均值（PC1–PC5）：{[round(x, 4) for x in f7['beta_real_mean_abs']]}；shuffled 均值：{[round(x, 4) for x in f7['beta_shuf_mean_abs']]}")
    lines.append(f"- Fig. 9 各层归一化灵敏度峰值：{[round(x, 4) for x in f9['layer_max_normalized']]}；shuffled 对照均值图峰值 {f9['shuffled_control_mean_max_normalized']:.4f}")
    for m, cfgs in tuning.items():
        lines.append(f"- {m} 调优网格：{len(cfgs)} 组，最佳 {min(cfgs, key=lambda c: c['best_val_mse'])['config']}（内部验证 MSE={min(c['best_val_mse'] for c in cfgs):.5f}）")
    lines.append("")
    lines.append("## 10. 长格式 CSV 字段说明")
    lines.append("")
    lines.append("同目录 `权威结果总表.csv` 以 (表格, 行, 列, 值) 长格式保存全部数值，供自动一致性校验脚本读取。")

    (tbl_dir / "权威结果总表.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    flat = {
        "split": split,
        "table1": t1,
        "stat_tests": st,
        "enso": ens,
        "holdout": hold,
        "table5": t5,
        "table3": t3,
        "regional": reg,
        "fig7": f7,
        "fig8": f8,
        "fig9": f9,
        "tuning": tuning,
        "derived": {
            "delta_acc_film_cat": dcat,
            "delta_acc_film_base": dbase,
            "delta_tcc_film_cat": dtcc_cat,
            "delta_tcc_film_base": dtcc_base,
            "stage1_acc": s1_acc,
            "stage2a_acc": s2a_acc,
            "stage2a_tcc": s2a_tcc,
        },
    }
    with (tbl_dir / "权威数值.json").open("w", encoding="utf-8") as fp:
        json.dump(flat, fp, indent=2, ensure_ascii=False)

    rows = []
    for m in order:
        for k, v in t1[m].items():
            rows.append(("Table1", m, k, v))
    for period in ("hindcast", "test"):
        for phase, d in ens[period].items():
            for k, v in d.items():
                rows.append((f"Table4_{period}", phase, k, v))
    for k, v in t5.items():
        for kk, vv in v.items():
            rows.append(("Table5", k, kk, vv))
    for r_name, rd in reg.items():
        for m in order:
            for k, v in rd[m].items():
                rows.append((f"Regional_{r_name}", m, k, v))
    for m in order:
        for k, v in hold[m].items():
            rows.append(("Holdout2020_2021", m, k, v))
    for k, v in st.items():
        for kk, vv in v.items():
            rows.append(("Stats", k, kk, vv))
    with (tbl_dir / "权威结果总表.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["表格", "行", "列", "值"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], json.dumps(r[3], ensure_ascii=False) if isinstance(r[3], (list, dict)) else r[3]])

    print(f"[Tables] wrote {tbl_dir / '权威结果总表.md'}")
    print(f"[Tables] wrote {tbl_dir / '权威结果总表.csv'} ({len(rows)} rows)")
    print(f"[Tables] wrote {tbl_dir / '权威数值.json'}")
    print(f"[Tables] split: valid grids={split['valid_china_land_grids']}, n_pcs={split['n_pcs']}, test ACC FiLM={t1['FiLM-U-Net']['test_acc']:.4f}")


if __name__ == "__main__":
    main()
