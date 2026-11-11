# -*- coding: utf-8 -*-
"""
Build the reviewer-comment <-> manuscript <-> response-letter traceability table
for the 05 revision package.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from revision_docx_lib import DocxEditor, q, W  # noqa: E402


def doc_text(path: Path) -> str:
    if not path.exists():
        return ""
    ed = DocxEditor(path)
    return "\n".join(ed.para_text(p) for p in ed.xml.findall(".//" + q(W, "p")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)

    V = json.loads((out / "结果表" / "权威数值.json").read_text(encoding="utf-8"))
    X = json.loads((out / "结果表" / "补充统计.json").read_text(encoding="utf-8"))
    T1, ST, EN, HO, T5, REG = V["table1"], V["stat_tests"], V["enso"], V["holdout"], V["table5"], V["regional"]
    S, D = V["split"], V["derived"]
    ng, n_test = S["valid_china_land_grids"], S["test_months"]
    fc, fb = ST["test_film_vs_cat_acc"], ST["test_film_vs_base_acc"]
    ts = ST["test_tcc_spatial"]
    f4 = lambda x: f"{x:.4f}"
    f3 = lambda x: f"{x:.3f}"
    p1 = lambda x: f"{x:.1f}%"
    sig = lambda x: f"{x:+.4f}"
    fmt_p = lambda x: "< 1e-300" if float(x) == 0.0 else (f"{float(x):.2e}" if float(x) < 1e-4 else f"{float(x):.4f}")

    ms_text = doc_text(out / "05_大修修改稿.docx")
    rl_text = doc_text(out / "05_审稿回复信.docx")

    rows = [
        ("EiC", "1",
         "摘要须分为 Study Region / Study Focus / New Hydrological Insights for the Region 三节",
         "摘要（结构化三段）",
         f"测试期 {f4(T1['FiLM-U-Net']['test_acc'])} ACC / {f4(T1['FiLM-U-Net']['test_tcc'])} TCC",
         "格式问题答复 · 摘要结构", "New Hydrological Insights", "摘要"),
        ("EiC", "2",
         "Highlights 3–5 条，每条 ≤85 字符（含空格）",
         "Highlights.docx（独立交付）",
         "4 条，长度由构建脚本打印并校验",
         "格式问题答复 · Highlights", "Highlights", "Highlights"),
        ("AE", "综合",
         "局限性、关键输入变量影响与相对重要性、2023–2024 评估独立性与代表性、"
         "与最近基线 EOF-cat 的对比及显著性检验",
         "§3.2 / §4.1 / §4.4 / §5.3 / §6.2 / §7；表 1、表 3、表 4；图 3–5、图 8–9",
         f"EOF-cat 测试 ACC {f4(T1['EOF-cat']['test_acc'])} vs FiLM {f4(T1['FiLM-U-Net']['test_acc'])}",
         "总述 + 各审稿人逐条答复", "3.2 Statistical significance", "EOF-cat"),
        ("R1", "Major 1",
         "更详细地讨论 FiLM 模型的局限性",
         "§6.2 Limitations and Future Perspectives（12 条）",
         "含 in-sample 历史期、21 月短窗、非显著 ACC、多重比较、对称调制边界等",
         "意见 1", "Short independent window", "12 条局限"),
        ("R1", "Major 2",
         "说明极端降水异常值/离群值的预处理方式",
         "§2.1 / §3.2 / 回复信对应段",
         f"目标场重尾保留，|ΔP|>3 的格月占 {p1(S['pct_grid_months_abs_gt_3'])}；"
         "仅使用梯度范数裁剪",
         "意见 2", "gradient-norm clipping", "截断"),
        ("R1", "Major 3",
         "ACC 约 0.03 的提升是否显著/有实际意义，需补充评估指标",
         "表 1 / §4.1 / §3.2",
         f"ACC 差 {sig(fc['mean_diff'])}，Wilcoxon p={f3(fc['wilcoxon_p'])}；"
         "另报 TCC、RMSE、Sign Consistency、Extreme CSI",
         "意见 3 / 统计检验答复", "Wilcoxon", "Wilcoxon"),
        ("R1", "Major 4",
         "区分 U-Net 后处理本身贡献与 FiLM 遥连接整合的增量贡献",
         "§4.1 两阶段分解 / 表 2 / 图 3(d)",
         f"Stage 1 {sig(D['stage1_acc'])}，Stage 2a {sig(D['stage2a_acc'])}，"
         f"Stage 2b {sig(D['delta_acc_film_base'])}",
         "意见 4", "Stage 1", "Stage 1"),
        ("R1", "Major 5",
         "对 10 个大气预报因子做系统的特征重要性/消融分析",
         "表 3 / §4.3 / 图（置换重要性）",
         "20 次随机打乱的置换重要性排序",
         "意见 5", "Permutation feature importance", "置换重要性"),
        ("R1", "Major 6",
         "讨论图 4 所示区域差异的可能成因",
         "§4.2 / §6.1 / §6.2",
         f"定量分区域统计入文（东部季风区 FiLM TCC {f4(REG['Eastern_Monsoon']['FiLM-U-Net']['test_tcc'])}、"
         f"青藏高原 {f4(REG['Tibetan_Plateau']['FiLM-U-Net']['test_tcc'])}、"
         f"西北干旱区 {f4(REG['Northwest_Arid']['FiLM-U-Net']['test_tcc'])}）",
         "意见 6", "Eastern_Monsoon" if False else "eastern monsoon zone", "东部季风区"),
        ("R1", "Minor 1",
         "首次出现处定义 ENSO、PDO、BCC-CSM、SEAS5 等缩写",
         "§1 / §2 全文术语",
         "缩写表与首次定义检查",
         "格式修订条目", "ENSO", "缩写"),
        ("R1", "Minor 2",
         "统一引文与句间空格、句式（如 (T2m).Spatial）",
         "全文排版扫描",
         "构建脚本执行 space-before-citation 正则修正",
         "格式修订条目", "(T2m)", "空格"),
        ("R1", "Minor 3",
         "核对“10 个大气变量”与实际列出 9 个的矛盾，删除多余变量描述",
         "§2.3 变量清单",
         "删除 humidity / vertical velocity 残留",
         "意见 3（变量清单）", "ten atmospheric predictors", "变量"),
        ("R1", "Minor 4",
         "图 2/3 配色区分度、图 2 图例不得遮挡曲线",
         "图 2、图 3（全部重出）",
         "四模型统一配色，图例外置于绘图区上方",
         "意见 4（图件）", "Fig. 2", "图例"),
        ("R1", "Minor 5",
         "图 4 加月份/类别标签并统一色标范围",
         "图 4 与图注",
         "四季典型月 × 五列；统一色标并注明 ΔP 单位",
         "意见 5（图件）", "Fig. 4", "色标"),
        ("R1", "Minor 6",
         "澄清图 2c 中 EOF-CAT 的含义",
         "§2.4 / 图 2 图注",
         "明确为 5 个 SST 主成分在输入层通道拼接的对照模型",
         "结构对比答复", "EOF-cat", "EOF-cat"),
        ("R2", "1",
         "21 月不足以支撑年际结论；TCC 需分块 bootstrap 显著性；"
         "按 ENSO 位相分层 ACC/TCC；给出测试期位相月数；如可能增加非厄尔尼诺测试期",
         "§3.2 / §4.4 / 表 4 / §6.2；补充统计.json",
         f"测试期 {EN['test']['El Nino']['months']} El Niño / "
         f"{EN['test']['Neutral']['months']} 中性 / {EN['test']['La Nina']['months']} La Niña；"
         f"2020–2021 留出窗 FiLM ACC {f4(HO['FiLM-U-Net']['acc'])} < EOF-cat {f4(HO['EOF-cat']['acc'])}",
         "答复 1–4（ENSO / 留出敏感性）", "ENSO", "非厄尔尼诺"),
        ("R2", "2",
         "明确 EOF 基底、标准化与气候态的拟合窗口；若使用全序列须重跑",
         "§2.1.2 / §2.2 / §2.4 / 图 1",
         f"训练窗 {S['train_months']} 月拟合；内部验证 {S['val_months']} 月；"
         f"独立测试 {n_test} 月完全样本外；有效格点 {ng:,}",
         "数据泄露整改答复", "1994.01–2017.12", "泄露"),
        ("R2", "3",
         "图 3–5 及正文须加入 Baseline 与 EOF-cat，报告逐月/逐年/空间 FiLM−cat 差值",
         "图 2–5 / §4.1 / 表 1–2",
         f"逐月平均 ΔACC {sig(fc['mean_diff'])}（{fc['pos_months']}/{fc['total_months']} 月 > 0）；"
         f"空间平均 ΔTCC {sig(ts['film_vs_cat_mean_dtcc'])}",
         "意见 4（图件）/ 统计检验答复", "Fig. 3", "逐月"),
        ("R2", "4",
         "解释 EOF-cat 退化并证明对比公平性（PC 数、W、插入深度、调参预算）",
         "§2.4 / §4.1 / 回复信",
         f"相同 5 PC / W=1 / 3 点网格共 9 次训练 / 100 epoch；"
         f"如实说明插入深度不同；EOF-cat 测试 ACC {f4(T1['EOF-cat']['test_acc'])} 并非退化",
         "公平性答复", "hyperparameter", "超参数"),
        ("R2", "5",
         "配对检验（符号/Wilcoxon）、按年分块 bootstrap CI、图 5 TCC 显著性打点；"
         "无支撑则删除 significantly outperforms",
         "§3.2 / 表 1 / 图 5 / §4.1 / §7",
         f"FiLM−cat ACC p={f3(fc['wilcoxon_p'])}，逐月 CI "
         f"[{sig(fc['boot_month_ci95'][0])}, {sig(fc['boot_month_ci95'][1])}]；"
         f"空间 ΔTCC Wilcoxon p={fmt_p(ts['film_vs_cat_wilcoxon_p'])}；"
         f"FiLM 原始 p<0.05 {p1(T1['FiLM-U-Net']['test_sig_pct'])}、"
         f"BH-FDR q<0.05 {p1(X['fdr']['FiLM-U-Net']['bh_fdr_005_pct'])}",
         "统计检验答复", "Benjamini–Hochberg", "FDR"),
        ("R2", "6",
         "补齐 FiLM 的 SST PCs 置零 / 局地变量置零对称消融；"
         "图 7、图 9 增加打乱 PCs 阴性对照",
         "图 7(c,d) / 图 8（6 面板）/ 图 9(f) / 表 5 / §5.3",
         f"FiLM SST 置零 ACC {f4(T5['FiLM-U-Net (SST PCs Zeroed)']['test_acc'])}、"
         f"打乱 {f4(T5['FiLM-U-Net (Shuffled PCs Control)']['test_acc'])}、"
         f"完整 {f4(T5['Full FiLM-U-Net (Ours)']['test_acc'])}；"
         f"局地置零 {f4(T5['FiLM-U-Net (Local Zeroed)']['test_acc'])}",
         "消融答复", "Symmetric ablations", "消融"),
    ]

    lines = [
        "# 审稿意见 ↔ 修改稿 ↔ 回复信 三向对照表",
        "",
        "- 稿件编号：EJRH-D-26-03375",
        "- 唯一数值源：`结果表/权威数值.json`、`结果表/补充统计.json`",
        f"- 独立测试窗：2023.01–2024.09（{n_test} 月）；有效陆地格点 {ng:,}",
        "- 检出状态：以生成的 05 版修改稿 / 回复信全文做关键词核验；"
        "「⚠️」表示该行需在提交前人工复核。",
        "",
        "| 来源 | 条目 | 审稿要求 | 修改稿落实位置 | 关键数值 / 证据 | 回复信对应条目 | 修改稿检出 | 回复信检出 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    coverage = {"ms": 0, "rl": 0, "total": len(rows), "issues": []}
    for src, item, req, loc, ev, resp, kw_ms, kw_rl in rows:
        ms_ok = (not ms_text) or (kw_ms.lower() in ms_text.lower())
        rl_ok = (not rl_text) or (kw_rl in rl_text)
        ms_flag = "—（未生成）" if not ms_text else ("✅" if ms_ok else "⚠️")
        rl_flag = "—（未生成）" if not rl_text else ("✅" if rl_ok else "⚠️")
        if ms_text and not ms_ok:
            coverage["issues"].append(f"{src}-{item}: 修改稿未检出 `{kw_ms}`")
        if rl_text and not rl_ok:
            coverage["issues"].append(f"{src}-{item}: 回复信未检出 `{kw_rl}`")
        coverage["ms"] += int(bool(ms_text) and ms_ok)
        coverage["rl"] += int(bool(rl_text) and rl_ok)
        lines.append(
            f"| {src} | {item} | {req} | {loc} | {ev} | {resp} | {ms_flag} | {rl_flag} |"
        )

    lines += [
        "",
        "## 核验汇总",
        "",
        f"- 意见条目总数：{coverage['total']}",
        f"- 修改稿自动检出：{coverage['ms']}/{coverage['total']}",
        f"- 回复信自动检出：{coverage['rl']}/{coverage['total']}",
        f"- 待人工复核条目：{len(coverage['issues'])}",
    ]
    for issue in coverage["issues"]:
        lines.append(f"  - {issue}")
    lines += [
        "",
        "> 注：关键词检出仅证明对应内容存在，不等价于语义完整性；"
        "数值一致性与旧值残留由 `校验/一致性校验报告.md` 独立校验。",
        "",
    ]

    dest = out / "三向对照表.md"
    dest.write_text("\n".join(lines), encoding="utf-8")
    (out / "校验" / "三向对照核验.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[Traceability] wrote {dest}")
    print(f"[Traceability] ms {coverage['ms']}/{coverage['total']}, "
          f"rl {coverage['rl']}/{coverage['total']}, issues {len(coverage['issues'])}")
    for issue in coverage["issues"]:
        print(f"  [WARN] {issue}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
