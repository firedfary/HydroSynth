"""Build the revision-07 manuscript, highlights and reviewer responses.

Revision 07 keeps revision 06's corrected data handling -- training-period
preprocessing, a physically reconstructed raw SEAS5 comparator, two
retrospective origins -- and changes what the comparison is allowed to claim.

Three measurements drive the change, all verified in
``U_Net_model/FILM_FIX_FINDINGS.md``:

1. Five retrainings of an identical configuration give a FiLM-minus-Baseline
   ACC gap of -0.0078 +/- 0.0012, so the gap is real; but rerunning that same
   configuration moves ensemble ACC by up to 0.0090, so no single run can
   resolve it. Revision 06 reported -0.0076 from one run.
2. Extreme-event CSI is dominated by output sharpness here. Equalizing each
   month to the observed spatial standard deviation reverses the published
   ranking exactly: Baseline > FiLM > EOF-cat, not EOF-cat > FiLM > Baseline.
3. The FiLM conditioning pathway carries a measured modulation of 2-7 percent
   of feature magnitude, so revision 05's interpretability evidence describes
   a pathway that works as designed and contributes no measurable skill.

Revision 05's interpretability section is reinstated, with its conclusions
reversed, because reviewer 2 asked for it to be strengthened rather than
removed and its shuffled-PC controls answer that request directly.

Every number is read from a result file; none is typed in. Run
``revision07_verify.py`` after this to check the written documents against the
same sources.
"""

import json
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.paths import SubprojectPaths

BASE = (Path.home() / "OneDrive/Desktop/文献与汇报/02_自研课题与论文/"
        "FiLM调制UNet降水预测/修改1")
SOURCE = BASE / "06_对照研究返修"
DELIVERY = BASE / "07_可信度返修"
PATHS = SubprojectPaths(__file__)
RESULTS = PATHS.results_dir

METRICS = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))["metrics"]
ABLATION = json.loads((RESULTS / "revision06_ablation_v1/ablation.json").read_text(encoding="utf-8"))
AMPLITUDE = json.loads((RESULTS / "revision06_ablation_amplitude/ablation_amplitude.json")
                       .read_text(encoding="utf-8"))
ENSO = json.loads((RESULTS / "revision06_eval_v2/enso_metrics.json").read_text(encoding="utf-8"))
HINDCAST = json.loads((RESULTS / "revision06_hindcast_v1/metrics.json").read_text(encoding="utf-8"))
HINDCAST_PAIRED = json.loads((RESULTS / "revision06_hindcast_v1/paired_analysis.json")
                             .read_text(encoding="utf-8"))
INFERENCE = json.loads((RESULTS / "revision06_inference_v1/inference.json").read_text(encoding="utf-8"))
REGIONAL = json.loads((RESULTS / "revision06_regions_v1/regional_metrics.json")
                      .read_text(encoding="utf-8"))["rows"]
FLOOR = json.loads((RESULTS / "film_fix_summary/reproducibility_floor.json")
                   .read_text(encoding="utf-8"))["stages"]["s0_current"]
FIGVALS = json.loads((RESULTS / "revision07_figures/figure_values.json").read_text(encoding="utf-8"))

AMP_ROWS = {row["row"]: row for row in AMPLITUDE["rows"]}
CHANCE_CSI = 0.122
REPLICATES = ("film_fix_s1", "film_fix_rep", "film_fix_rep3", "film_fix_rep4", "film_fix_rep5")


def f(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def replicate_rows():
    """Per-run ensemble ACC for the five identical retrainings."""
    rows = []
    for index, name in enumerate(REPLICATES, start=1):
        payload = json.loads((RESULTS / name / "metrics.json").read_text(encoding="utf-8"))
        ens = payload["results"]["s0_current"]["periods"]["validation"]["ensemble"]
        gap = ens["film32_identity"]["mean_acc"] - ens["baseline32"]["mean_acc"]
        rows.append((f"Run {index}", f(ens["baseline32"]["mean_acc"], 4),
                     f(ens["eof_cat32"]["mean_acc"], 4),
                     f(ens["film32_identity"]["mean_acc"], 4), f(gap, 4)))
    spreads = FLOOR["per_model_acc"]
    rows.append(("Spread", f(spreads["Baseline"]["spread"], 4),
                 f(spreads["EOF-cat"]["spread"], 4), f(spreads["FiLM"]["spread"], 4),
                 f(FLOOR["gap_film_minus_baseline"]["spread"]["spread"], 4)))
    return rows


def setup(doc, chinese=False):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.85)
    section.right_margin = Inches(0.85)
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "SimSun" if chinese else "Times New Roman"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.12
    for name, size in (("Title", 15), ("Heading 1", 12), ("Heading 2", 11)):
        style = styles[name]
        style.font.name = "SimSun" if chinese else "Times New Roman"
        style.font.size = Pt(size)
        style.font.color.rgb = None
        style.paragraph_format.space_before = Pt(13 if name == "Heading 1" else 8)
        style.paragraph_format.space_after = Pt(5)
        ppr = style.element.pPr
        if ppr is not None:
            border = ppr.find(qn("w:pBdr"))
            if border is not None:
                ppr.remove(border)
        rpr = style.element.get_or_add_rPr()
        fonts = rpr.find(qn("w:rFonts"))
        if fonts is None:
            fonts = OxmlElement("w:rFonts")
            rpr.append(fonts)
        for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            fonts.set(qn(attr), "SimSun" if chinese else "Times New Roman")


def para(doc, text, style=None):
    return doc.add_paragraph(text, style=style)


def heading(doc, text, level=1):
    doc.add_heading(text, level=level)


def table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.autofit = False
    for i, value in enumerate(headers):
        t.rows[0].cells[i].text = str(value)
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = str(value)
    for row in t.rows:
        for i, cell in enumerate(row.cells):
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if widths:
                cell.width = Inches(widths[i])
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.space_before = Pt(2)
                p.paragraph_format.line_spacing = 1.0
                for run in p.runs:
                    run.font.size = Pt(8.5)
    for cell in t.rows[0].cells:
        for p in cell.paragraphs:
            for run in p.runs:
                run.bold = True
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    t.rows[0]._tr.get_or_add_trPr().append(repeat)
    doc.add_paragraph()
    return t


def figure(doc, filename, caption, width=6.6):
    path = DELIVERY / "figures" / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.line_spacing = 1.0
    p.add_run().add_picture(str(path), width=Inches(width))
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.line_spacing = 1.0
    for run in cap.runs:
        run.font.size = Pt(9)


def model_table(period, with_amplitude=False):
    result = METRICS[period]["ensemble"]
    order = (("SEAS5", "seas5"), ("Baseline U-Net", "baseline"),
             ("EOF-cat", "eof_cat"), ("FiLM-U-Net", "film"))
    ratios = FIGVALS["amplitude"]["amplitude_ratio"]
    short = {"SEAS5": "SEAS5", "Baseline U-Net": "Baseline",
             "EOF-cat": "EOF-cat", "FiLM-U-Net": "FiLM"}
    rows = []
    for label, key in order:
        v = result[key]
        row = [label, f(v["mean_acc"]), f(v["mean_tcc"]), f(v["rmse"]),
               f(v["sc_pct"], 1), f(v["csi"])]
        if with_amplitude:
            row.append(f(ratios[short[label]], 2))
        rows.append(tuple(row))
    return rows


def paired(label, key, period="test"):
    row = METRICS[period]["paired_acc"][key]
    return (label, f(row["mean_diff"], 4),
            f"{row['pos_months']}/{row['total_months']}", f(row["wilcoxon_p"], 3),
            f"[{f(row['boot_month_ci95'][0], 3)}, {f(row['boot_month_ci95'][1], 3)}]",
            f"[{f(row['boot_year_block_ci95'][0], 3)}, {f(row['boot_year_block_ci95'][1], 3)}]")
