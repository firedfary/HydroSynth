"""Build the revision-06 manuscript and bilingual reviewer responses."""

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

DELIVERY = (Path.home() / "OneDrive/Desktop/文献与汇报/02_自研课题与论文/"
            "FiLM调制UNet降水预测/修改1/06_对照研究返修")
PATHS = SubprojectPaths(__file__)
RESULTS = PATHS.results_dir
METRICS = json.loads((DELIVERY / "summary.json").read_text(encoding="utf-8"))["metrics"]
ABLATION = json.loads((RESULTS / "revision06_ablation_v1/ablation.json").read_text(encoding="utf-8"))
ENSO = json.loads((RESULTS / "revision06_eval_v2/enso_metrics.json").read_text(encoding="utf-8"))
HINDCAST = json.loads((RESULTS / "revision06_hindcast_v1/metrics.json").read_text(encoding="utf-8"))
HINDCAST_PAIRED = json.loads((RESULTS / "revision06_hindcast_v1/paired_analysis.json").read_text(encoding="utf-8"))
INFERENCE = json.loads((RESULTS / "revision06_inference_v1/inference.json").read_text(encoding="utf-8"))
REGIONAL = json.loads((RESULTS / "revision06_regions_v1/regional_metrics.json").read_text(encoding="utf-8"))["rows"]
def f(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


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
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("EJRH-D-26-03375  |  ")
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)


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
    p.add_run().add_picture(str(path), width=Inches(width))
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.keep_with_next = False
    for run in cap.runs:
        run.font.size = Pt(9)


def model_table(period):
    result = METRICS[period]["ensemble"]
    return [(name, f(v["mean_acc"]), f(v["mean_tcc"]), f(v["rmse"]),
             f(v["sc_pct"], 1), f(v["csi"]))
            for name, v in (("SEAS5", result["seas5"]),
                            ("Baseline U-Net", result["baseline"]),
                            ("EOF-cat", result["eof_cat"]),
                            ("FiLM-U-Net", result["film"]))]


def paired(label, key):
    row = METRICS["test"]["paired_acc"][key]
    return (label, f(row["mean_diff"], 4),
            f"{row['pos_months']}/{row['total_months']}", f(row["wilcoxon_p"], 3),
            f"[{f(row['boot_month_ci95'][0], 3)}, {f(row['boot_month_ci95'][1], 3)}]",
            f"[{f(row['boot_year_block_ci95'][0], 3)}, {f(row['boot_year_block_ci95'][1], 3)}]")


def manuscript():
    doc = Document()
    setup(doc)
    doc.styles["Normal"].paragraph_format.line_spacing = 2
    line_numbers = OxmlElement("w:lnNumType")
    line_numbers.set(qn("w:countBy"), "1")
    line_numbers.set(qn("w:start"), "1")
    line_numbers.set(qn("w:restart"), "continuous")
    doc.sections[0]._sectPr.append(line_numbers)
    para(doc, "Comparing U-Net Postprocessing and SST Conditioning for Monthly Precipitation over China", "Title")
    authors = doc.add_paragraph()
    for i, (name, affiliation) in enumerate((
        ("Fan YUE", "1,3"), ("Xiaojuan WANG", "1*"),
        ("Rong AI", "2"), ("Yinzhong WU", "3"),
        ("Guolin FENG", "2,4"), ("Zhiqiang GONG", "2"),
    )):
        if i:
            authors.add_run(", " if i < 5 else ", and ")
        authors.add_run(name)
        authors.add_run(affiliation).font.superscript = True
    for line in (
        "1 College of Electronic Information Engineering, Suzhou University of Technology, Suzhou 215500, China",
        "2 Laboratory for Climate Studies, National Climate Center, Beijing 100081, China",
        "3 College of Physics and Technology, Suzhou University of Science and Technology, Suzhou 215009, China",
        "4 College of Physical Science and Technology, Yangzhou University, Yangzhou 225009, China",
    ):
        para(doc, line)
    para(doc, "* Corresponding author: Xiaojuan WANG; email: mouse0903@126.com")
    heading(doc, "Abstract")
    test = METRICS["test"]["ensemble"]
    para(doc, "Study Region: Mainland China, where monsoon variability, aridity, and complex terrain challenge monthly precipitation prediction.")
    para(doc, "Study Focus: We compare direct precipitation from the fifth-generation ECMWF seasonal forecast system (SEAS5) with three matched 32-channel U-Net postprocessors: atmospheric predictors alone, the same predictors with five global sea surface temperature (SST) empirical orthogonal function (EOF) scores concatenated at the input, and the scores supplied through feature-wise linear modulation (FiLM). The main chronology fits preprocessing and models on 1994–2017, selects checkpoints on 2018–2022, and explores January 2023–September 2024. Two additional retrospective training origins yield nonoverlapping 2014–2019 and 2020–2022 holdouts.")
    para(doc, "New Hydrological Insights for the Region: On the 21-month evaluation, spatial anomaly correlation (ACC) is "
         f"{f(test['seas5']['mean_acc'])} for SEAS5, {f(test['baseline']['mean_acc'])} for the baseline, "
         f"{f(test['eof_cat']['mean_acc'])} for EOF-cat, and {f(test['film']['mean_acc'])} for FiLM. "
         "All three postprocessors lower root mean square error (RMSE) and improve extreme-event critical success index (CSI) in this period, but EOF-cat loses spatial ACC relative to SEAS5. Across three exploratory geographic partitions, the southwest has the lowest ACC for every system. In the 107-month retrospective holdout, FiLM ACC is 0.287 versus 0.289 for the baseline and 0.295 for SEAS5. Removing or temporally mismatching the SST scores does not reduce FiLM ACC in the explored period. These findings do not support a stable incremental benefit from SST modulation or universal postprocessor superiority.")
    para(doc, "Keywords: precipitation postprocessing; SEAS5; U-Net; empirical orthogonal functions; feature-wise linear modulation; China")

    heading(doc, "1 Introduction")
    para(doc, "Monthly precipitation has different sources of predictability across China. Previous studies associate summer rainfall in South China with distinct tropical Indian and Pacific Ocean influences in the early and late rainy seasons (Yuan et al., 2019), summer rainfall in Southwest China with moisture transport from several source regions (Yuan and Yang, 2020), and North China rainfall with the evolution of preceding SST anomalies rather than their level alone (Liang et al., 2023). These season- and region-specific findings motivate a regional assessment of monthly forecasts; they do not imply that one national SST summary should improve every month or location.")
    para(doc, "The fifth-generation ECMWF seasonal forecast system (SEAS5) provides the dynamical fields evaluated here (Johnson et al., 2019). Gridded atmospheric predictors can be postprocessed with a multiscale U-Net encoder–decoder (Ronneberger et al., 2015). Empirical orthogonal functions (EOFs) summarize dominant spatial SST variability (Hannachi et al., 2007), while feature-wise linear modulation (FiLM) conditions intermediate network features through learned channelwise scales and offsets (Perez et al., 2017). The hydrological question is whether either SST-conditioning design improves observed precipitation anomaly skill beyond atmospheric-field postprocessing, including in regions with different moisture and circulation controls.")
    para(doc, "Precipitation postprocessing over China has been studied with statistical calibration of ECMWF System4 seasonal output (Peng et al., 2014), deep learning for seasonal precipitation prediction (Jin et al., 2022), and a hybrid U-Net for subseasonal-to-seasonal summer ensemble forecasts (Lyu et al., 2024). These studies establish relevant alternatives, but their forecast systems, lead times, target definitions, and verification designs differ from the monthly archive used here. Their reported scores cannot be used as direct numerical benchmarks for our experiments. The specific contribution of this study is a matched comparison of an unconditioned U-Net, input-level EOF concatenation, and multi-level FiLM conditioning against the same reconstructed SEAS5 field.")
    para(doc, "A more expressive conditioning mechanism is useful only if it improves forecasts under matched inputs and a defensible comparator. The initial version of this study interpreted a standardized atmospheric input channel as raw SEAS5 precipitation and allowed full-record information into some preprocessing. Reconstructing the raw SEAS5 comparator and fitting transformations on the training period changes the ranking. We therefore ask which method performs better on each metric, whether FiLM adds value beyond direct EOF concatenation and the unconditioned U-Net, and whether any benefit survives a second chronological origin. These questions make negative and mixed results informative rather than treating architectural complexity as an advantage by itself.")

    heading(doc, "2 Data and methods")
    heading(doc, "2.1 Data and chronology", 2)
    para(doc, "Monthly SEAS5 archive files supplied by the National Climate Center of China from January 1994 through September 2024 supply ten China-domain atmospheric fields: total precipitation (tp), geopotential height at 200 and 500 hPa (h200, h500), sea-level pressure (slp), 2-m temperature (t2m), 850-hPa temperature (t850), and zonal and meridional winds at 200 and 850 hPa (u200, u850, v200, v850). The archived 1° fields are mapped to a nominal 0.5° grid of 120 × 140 cells over 70–140°E and 0–60°N; network predictors use bicubic interpolation, while the raw-precipitation comparator uses bilinear interpolation. Precipitation observations are based on 2,374 National Meteorological Information Center stations and their prepared gridded target field; 5,243 land cells contain valid observations across the analysis. Global NOAA Extended Reconstructed Sea Surface Temperature version 5 (ERSSTv5) fields provide SST predictors (Huang et al., 2017). Three archive months are absent (September and October 2011; January 2017), leaving 366 aligned months.")
    para(doc, "The main chronology assigns 285 months in 1994–2017 to fitting, 60 months in 2018–2022 to validation and checkpoint selection, and 21 months in January 2023–September 2024 to exploratory evaluation. All data-dependent atmospheric normalization, SST climatology, EOF basis, and score scaling are fitted on the first period. The observed and raw SEAS5 precipitation anomalies use a fixed 1994–2010 calendar-month reference climatology, which lies within training. Their fractional anomaly is precipitation minus the corresponding calendar-month mean, divided by that mean plus 0.0001. For the SEAS5 comparator, we select the first monthly tp field from each target-month archive file, bilinearly interpolate it, constrain it nonnegative, and apply a fixed 31-day conversion factor before anomaly conversion, matching the archived-data processing. The file metadata identify the first valid month but do not establish an operational initialization date or lead label. No standardized network input is used as the raw comparator.")
    para(doc, "The target's extreme anomalies are retained without winsorization. The 90th percentile of training anomalies, 0.9843, defines extreme hits in the main chronology. This fractional-anomaly scale can magnify small absolute precipitation errors where climatological rainfall is low, so aggregate skill and regional maps require caution. Two retrospective origins use 1994–2010 training and 2011–2013 validation for a 2014–2019 holdout (71 available months), and 1994–2015 training and 2016–2019 validation for a 2020–2022 holdout (36 months). Atmospheric transforms, SST EOFs and scaling, models, and extreme thresholds are refitted at each origin. The 2014–2022 combined holdout contains 107 distinct months and two different trained model sets. These years and the design were previously examined, so the combined analysis is descriptive rather than a pristine confirmatory test.")
    heading(doc, "2.2 SST representation and model comparison", 2)
    para(doc, "The stored SST predictor contains six preceding monthly fields for each target month. We average those six fields to one SST map, subtract a training-only SST climatology, fit EOFs on the training maps, and project later maps onto the frozen basis. The first five standardized principal-component (PC) scores are shared by EOF-cat and FiLM. Averaging removes within-window timing; the design should not be described as a six-step PC sequence or a one-month PC lag.")
    para(doc, "All neural variants use the same 32-channel base U-Net, with five encoder levels, a mirrored decoder, skip connections, batch normalization, ReLU activations, and spatial dropout. Baseline receives the ten atmospheric channels and has 3,352,289 trainable parameters. EOF-cat broadcasts the five PC scores as constant spatial maps and appends them to the ten input channels, giving 3,353,729 parameters. FiLM receives the ten atmospheric channels plus a two-layer index encoder that maps the five scores to channelwise scale and offset parameters at each encoder level, giving 3,452,513 parameters. Its transform applies the predicted scale to each feature channel and adds the predicted offset; zero initialization of the final conditioning layer starts near the unmodulated mapping. EOF-cat and FiLM use the same five PCs from the same preceding six-month SST mean, with the same split, optimizer, seeds, 100-epoch training budget, and validation-ACC checkpoint rule. The 98,784-parameter FiLM increase over EOF-cat and the difference between input concatenation and five-level modulation mean that insertion depth and capacity are not matched. The earlier exploratory search cannot be retroactively equalized, so this comparison does not isolate a single causal effect of the FiLM operator.")
    heading(doc, "2.3 Training and evaluation", 2)
    para(doc, "Each model is trained with seeds 42, 43, and 44 for 100 epochs, batch size eight, masked mean squared error over valid land cells, AdamW at learning rate 0.0002 and weight decay 0.00001, cosine decay, dropout 0.1, and gradient-norm clipping at 2. The best checkpoint for each model and seed is selected by mean validation spatial ACC. The three selected predictions are averaged with equal weight. These hyperparameters were fixed for the matched rerun; the earlier experimental search and repeated examination of 2023–2024 preclude treating the latter as an untouched final test.")
    para(doc, "Spatial ACC is the Pearson correlation across valid cells for each month, then averaged over months. Temporal correlation coefficient (TCC) is computed at each valid cell across months, then averaged spatially. RMSE is over all valid grid-month pairs; sign consistency is the fraction with matching anomaly sign; and extreme critical success index (CSI) is hits divided by hits plus false alarms plus misses above the training p90. Monthly ACC differences use paired Wilcoxon tests and month-resampled and calendar-year-block bootstrap intervals. With only two calendar years in the exploratory period, the year-block interval has extremely few distinct resamples and is descriptive. Gridwise TCC p-value stippling does not constitute field significance, and 21 months do not support strong interannual claims.")

    heading(doc, "3 Results")
    heading(doc, "3.1 Matched model scores", 2)
    para(doc, "Table 1. Main-model validation and exploratory evaluation. Higher ACC, TCC, sign consistency and CSI are better; lower RMSE is better. TCC from 21 months is descriptive.")
    table(doc, ("Period", "Model", "ACC", "TCC", "RMSE", "Sign %", "CSI"),
          [(period, *row) for period in ("Validation", "Exploratory")
           for row in model_table("validation" if period == "Validation" else "test")])
    para(doc, "On 2018–2022 validation, the baseline has the highest ACC (0.335) and raw SEAS5 the highest TCC (0.429). In the 21-month explored period, baseline leads ACC (0.333) and sign consistency (66.4%), whereas EOF-cat has the best TCC (0.407), RMSE (1.844), and extreme CSI (0.124). FiLM is intermediate on most measures. Although every neural variant has lower RMSE and higher CSI than SEAS5 in this period, EOF-cat ACC is lower than raw SEAS5 by 0.020. Thus a blanket superiority statement fails even before accounting for uncertainty.")
    figure(doc, "Fig1_monthly_ACC.png", "Figure 1. Monthly spatial ACC for all four systems and paired FiLM-minus-comparator differences in the 21-month exploratory evaluation.")
    figure(doc, "Fig2_metric_tradeoff.png", "Figure 2. Metric-dependent ranking of the four systems in the exploratory evaluation; metrics have different units and directions.")
    heading(doc, "3.2 Paired comparisons and spatial variation", 2)
    para(doc, "Table 2. Paired monthly spatial ACC differences in January 2023–September 2024. Confidence intervals are descriptive because the period was repeatedly examined; the calendar-year-block interval has only two available years.")
    table(doc, ("Comparison", "Mean Δ", "Positive", "Wilcoxon p", "Month 95% CI", "Year block 95% CI"),
          [paired("FiLM – baseline", "film_minus_baseline"),
           paired("FiLM – EOF-cat", "film_minus_eof_cat"),
           paired("FiLM – SEAS5", "film_minus_seas5"),
           paired("Baseline – SEAS5", "baseline_minus_seas5")])
    para(doc, "FiLM exceeds EOF-cat in 17 of 21 months on ACC, but its mean ACC remains below baseline by 0.016. FiLM's gain over raw SEAS5 is 0.0028 and the paired Wilcoxon p-value is 0.892. Its annual mean ACC differences from baseline, EOF-cat, and SEAS5 are -0.015, +0.015, and +0.025 in 2023 (12 months), versus -0.017, +0.034, and -0.026 in 2024 (nine months). Thus even the direction of the FiLM-SEAS5 difference changes across the two calendar years.")
    tcc_rows = []
    for label, key in (("FiLM – SEAS5", "seas5"),
                       ("FiLM – baseline", "baseline"),
                       ("FiLM – EOF-cat", "eof_cat")):
        val = INFERENCE["validation"]["tcc_differences"][key]
        test_tcc = INFERENCE["exploratory"]["tcc_differences"][key]
        tcc_rows.append((label, f(val["film_minus_comparator"]),
                         f"[{f(val['year_block_boot95'][0])}, {f(val['year_block_boot95'][1])}]",
                         f(test_tcc["film_minus_comparator"]),
                         f"[{f(test_tcc['year_block_range'][0])}, {f(test_tcc['year_block_range'][1])}]"))
    para(doc, "Table 3. Paired differences in spatially averaged TCC. Validation intervals use 1,000 calendar-year-block bootstrap samples over five years. The explored period has only two years; its ranges enumerate the three distinct block combinations and are not 95% confidence intervals. Validation also served checkpoint selection, so its intervals are descriptive conditional on those selections.")
    table(doc, ("Comparison", "2018–22 ΔTCC", "Year boot 95%", "2023–24 ΔTCC", "Two-year range"), tcc_rows)
    para(doc, "The 21-month spatial mean TCC values are 0.398 for SEAS5, 0.394 for baseline, 0.407 for EOF-cat, and 0.398 for FiLM. None of the five-year validation TCC-difference intervals in Table 3 excludes zero. The exploratory two-year ranges are too coarse for a significance claim. Maps show heterogeneous cells rather than a uniform FiLM gain.")
    figure(doc, "Fig3_temporal_corr.png", "Figure 3. Gridwise TCC for all four systems over 21 months. Dots indicate nominal two-sided p < 0.05 at displayed cells; they are not a spatial field-significance mask.")
    figure(doc, "Fig4_TCC_difference.png", "Figure 4. Gridwise FiLM-minus-baseline and FiLM-minus-EOF-cat TCC differences over 21 exploratory months. These maps are descriptive and carry no spatial significance claim.")
    figure(doc, "Fig5_spatial_examples.png", "Figure 5. Observed and predicted fractional precipitation anomalies in January, April, July, and October 2023. The same display range is used for every panel; values beyond it are clipped for visualization only.", 6.5)
    regional_rows = []
    for period, label in (("validation", "2018–22"), ("test", "2023–24")):
        for region in ("East", "Southwest", "Northwest"):
            group = {row["model"]: row for row in REGIONAL
                     if row["period"] == period and row["region"] == region}
            regional_rows.append((label, region, group["seas5"]["valid_cells"],
                                  *(f(group[model]["acc"]) for model in
                                    ("seas5", "baseline", "eof_cat", "film"))))
    para(doc, "Table 4. Descriptive regional spatial ACC on the common valid grid. East is ≥105°E; Southwest is <105°E and ≤37°N; Northwest is <105°E and >37°N. These simple geographic partitions are not formal hydroclimatic zones. Within-region ACC is computed across that region's cells for each month, then averaged; it need not average to the nationwide ACC.")
    table(doc, ("Period", "Region", "Cells", "SEAS5", "Baseline", "EOF-cat", "FiLM"), regional_rows)
    para(doc, "In the explored period, baseline ACC exceeds raw SEAS5 in each partition: 0.369 versus 0.354 in the east, 0.284 versus 0.254 in the southwest, and 0.325 versus 0.299 in the northwest. EOF-cat trails raw SEAS5 in all three, while FiLM is only slightly above it. The southwest has the lowest regional ACC for every system. Earlier work links southwest summer rainfall to moisture transport from multiple source regions (Yuan and Yang, 2020), but this study has not attributed its monthly model errors to those pathways. The partitions were inspected after model development and have no regional significance test. Terrain, station coverage, and amplified fractional anomalies in dry areas remain possible explanations rather than demonstrated causes.")
    heading(doc, "3.3 Conditioning controls and predictor importance", 2)
    ab = ABLATION["ensemble"]
    para(doc, "Table 5. Symmetric input controls on the exploratory period. PC zeroing and local-input zeroing are out-of-distribution perturbations, so their score changes diagnose dependence but are not replacement-model benchmarks.")
    table(doc, ("Input condition", "Baseline ACC", "EOF-cat ACC", "FiLM ACC"),
          [(label, f(ab[key]["baseline"]["mean_acc"]),
            f(ab[key]["eof_cat"]["mean_acc"]), f(ab[key]["film"]["mean_acc"]))
           for label, key in (("Full", "full"), ("SST PCs = 0", "pc_zero"),
                              ("Atmospheric maps = 0", "local_zero"))])
    para(doc, "Setting FiLM's standardized SST PCs to zero raises ACC from 0.316 to 0.318; replacing the scores with same-calendar-month values from 2018–2022 gives mean ACC 0.324 across ten trials. EOF-cat likewise rises from 0.293 to 0.319 when PCs are zeroed. Removing the local atmospheric maps sharply degrades all three architectures. These controls do not establish that simultaneous SST conditioning improves the evaluated forecast, although some other metrics can change in the opposite direction.")
    figure(doc, "Fig6_symmetric_ablation.png", "Figure 6. Symmetric SST and local-predictor input controls for baseline, EOF-cat, and FiLM. Baseline has no SST input, so its PC-zeroed and PC-replaced bars repeat the full-input prediction.")
    para(doc, "Replacing one atmospheric predictor map at a time with a same-calendar-month map from the validation period makes tp dominant: its FiLM ACC reduction averages 0.263 across ten trials. Every other predictor has an absolute mean effect below 0.005. Correlated circulation variables can substitute for one another, so a small single-variable permutation effect does not prove that the variable is useless. The test-period use of these controls remains exploratory.")
    figure(doc, "Fig7_predictor_importance.png", "Figure 7. FiLM predictor permutation importance, measured as mean decrease in spatial ACC when one full atmospheric map is replaced.", 5.8)
    heading(doc, "3.4 ENSO phase and retrospective origin", 2)
    para(doc, "We classify each month using the NOAA Climate Prediction Center historical Oceanic Niño Index (ONI) centered three-month value: warm at ≥ +0.5 °C, cold at ≤ −0.5 °C, and neutral otherwise. These threshold categories are not declarations of official ENSO episodes. In 2023–2024 there are 13 warm, one cold, and seven neutral months. The single cold month does not permit a temporal correlation estimate, and no phase-specific score from that period is a stable regime generalization. Tables 6 and 7 report the 2018–2022 validation, explored period, and pooled 2014–2022 retrospective holdout. We do not use 1994–2010 training predictions as evidence of out-of-sample phase skill: the first retrospective models were fitted on those same targets.")
    rows = []
    for period, key in (("2018–2022 validation", "validation"),
                        ("2023–2024 explored", "test"),
                        ("2014–2022 retrospective", "retrospective")):
        for phase in ("warm", "cold", "neutral"):
            row = (HINDCAST["pooled"]["phases"][phase] if key == "retrospective"
                   else ENSO["periods"][key][phase])
            rows.append((period, phase, row["months"],
                         *[f(row["metrics"][name]["mean_acc"]) for name in
                           ("seas5", "baseline", "eof_cat", "film")]))
    caption = para(doc, "Table 6. Spatial ACC by ONI threshold category. The retrospective rows pool predictions from two different training origins. The exploratory cold category contains only one month.")
    caption.paragraph_format.page_break_before = True
    caption.paragraph_format.keep_with_next = True
    table(doc, ("Period", "ONI", "n", "SEAS5", "Baseline", "EOF-cat", "FiLM"), rows)
    tcc_rows = []
    for period, key in (("2018–2022 validation", "validation"),
                        ("2023–2024 explored", "test"),
                        ("2014–2022 retrospective", "retrospective")):
        for phase in ("warm", "cold", "neutral"):
            row = (HINDCAST["pooled"]["phases"][phase] if key == "retrospective"
                   else ENSO["periods"][key][phase])
            tcc_rows.append((period, phase, row["months"],
                             *[f(row["metrics"][name]["mean_tcc"]) for name in
                               ("seas5", "baseline", "eof_cat", "film")]))
    caption = para(doc, "Table 7. Spatially averaged TCC within ONI threshold categories. The retrospective rows combine two origins; the seven-month exploratory neutral estimate is unstable, and TCC is omitted for one cold month.")
    caption.paragraph_format.page_break_before = True
    caption.paragraph_format.keep_with_next = True
    table(doc, ("Period", "ONI", "n", "SEAS5", "Baseline", "EOF-cat", "FiLM"), tcc_rows)
    caption = para(doc, "Table 8. Retrospective holdouts from two training origins and their pooled 107-month evaluation. Each origin refits the models and all training-dependent transformations. Both windows were examined during development.")
    caption.paragraph_format.page_break_before = True
    caption.paragraph_format.keep_with_next = True
    rows = []
    for period, key in (("2014–2019 (71)", "2014_2019"),
                        ("2020–2022 (36)", "2020_2022")):
        for label, name in (("SEAS5", "seas5"), ("Baseline", "baseline"),
                            ("EOF-cat", "eof_cat"), ("FiLM", "film")):
            score = HINDCAST["folds"][key]["metrics"][name]
            rows.append((period, label, f(score["mean_acc"]),
                         f(score["mean_tcc"])))
    for label, name in (("SEAS5", "seas5"), ("Baseline", "baseline"),
                        ("EOF-cat", "eof_cat"), ("FiLM", "film")):
        score = HINDCAST["pooled"]["phases"]["all"]["metrics"][name]
        rows.append(("2014–2022 (107)", label, f(score["mean_acc"]),
                     f(score["mean_tcc"])))
    table(doc, ("Holdout", "Model", "ACC", "TCC"), rows)
    warm = HINDCAST_PAIRED["phase_contrasts"]["warm"]["paired_acc"]["film_minus_baseline"]
    pooled = HINDCAST_PAIRED["phase_contrasts"]["all"]["paired_acc"]["film_minus_baseline"]
    tcc = HINDCAST_PAIRED["pooled_tcc_and_annual_acc"]["tcc_differences"]["baseline"]
    para(doc, "The pooled holdout includes 35 warm, 36 cold, and 36 neutral ONI-threshold months. "
         f"FiLM exceeds baseline ACC by {f(warm['mean_diff'])} in warm months, but its calendar-year-block 95% interval "
         f"[{f(warm['boot_year_block_ci95'][0])}, {f(warm['boot_year_block_ci95'][1])}] crosses zero. "
         f"Across all 107 months, FiLM minus baseline ACC is {f(pooled['mean_diff'])} "
         f"(year-block 95% interval [{f(pooled['boot_year_block_ci95'][0])}, {f(pooled['boot_year_block_ci95'][1])}]); "
         f"its paired spatial-mean TCC difference is {f(tcc['film_minus_comparator'])} "
         f"(year-block 95% interval [{f(tcc['year_block_boot95'][0])}, {f(tcc['year_block_boot95'][1])}]). "
         "FiLM exceeds EOF-cat ACC by 0.029, but no consistent advantage over baseline or SEAS5 emerges. "
         "The first fold favors FiLM slightly in ACC, whereas the second reverses the ranking. "
         "All contrasts are retrospective and descriptive because the model design and these years were examined during development.")

    heading(doc, "4 Discussion")
    para(doc, "The most direct empirical explanation for the modest value of SST conditioning is that the local tp field carries most of the information used by the network. Its permutation effect is more than an order of magnitude larger than any individual circulation channel, and SST score removal or temporal mismatch does not hurt FiLM spatial ACC. The six-month SST mean may suppress the evolution and seasonal timing relevant to some regional rainfall signals: Liang et al. (2023) emphasize winter-to-spring SST changes for North China, while Yuan et al. (2019) distinguish early- from late-season teleconnections in South China. Those studies motivate alternative SST representations but do not establish the cause of the present model ranking. A single five-score vector also cannot directly encode location-specific teleconnections. The training set contains 285 monthly maps but far fewer independent climate regimes; five-level conditional scaling may therefore be difficult to estimate robustly. Equal backbone width and training budget improve comparability, but EOF-cat changes the first input convolution while FiLM adds a multilayer branch at five depths, so architecture and parameter-count effects remain intertwined.")
    para(doc, "The corrected raw SEAS5 comparator materially changes the interpretation. Bilinear remapping with a nonnegative precipitation constraint yields 0.314 exploratory ACC. Nonnegative bicubic remapping gives 0.288, and the previously reported 0.281 came from a standardized neural input rather than original model precipitation (Supplementary Figure S1). The visual and statistical comparison must use the same explicitly stated physical reconstruction; interpolation sensitivity is a limitation rather than evidence for a desired model ranking.")
    para(doc, "The study has several additional limits. First, repeated examination of 2023–2024 prevents a confirmatory test claim. Second, a 21-month period and two calendar-year blocks give weak temporal and phase-specific inference, especially for gridwise TCC. Third, extreme CSI is based on a training threshold and does not quantify rainfall intensity calibration. Fourth, the station-derived gridded target has uneven sampling, and fractional anomalies amplify error in dry regions. Fifth, the 2014–2022 retrospective sequence joins two training origins and was also examined during development; its intervals describe sensitivity rather than independent confirmatory uncertainty. Sixth, the current ablations are inference-time perturbations outside the joint training distribution, not retrained feature-removal experiments. Finally, the archive lacks initialization metadata, so the operational forecast lead cannot be independently verified from these files. A future study should pre-register architecture and metrics, hold back newly acquired years, and evaluate alternate SST windows and regional conditioning while preserving the same input and compute budget.")
    heading(doc, "5 Conclusions")
    para(doc, "With training-fitted preprocessing and a physically reconstructed SEAS5 comparator, the three U-Net postprocessors show different metric trade-offs. Baseline has the highest spatial ACC and sign consistency in the explored 2023–2024 period; EOF-cat has the highest mean TCC and extreme CSI and the lowest RMSE. FiLM is not consistently best and shows no stable ACC benefit from simultaneous SST scores. Although all three reduce RMSE and increase CSI in that period, EOF-cat loses ACC relative to SEAS5. The comparison therefore does not support a universal superiority claim.")
    para(doc, "The 107-month retrospective evaluation gives ACC 0.295 for SEAS5, 0.289 for baseline, 0.258 for EOF-cat, and 0.287 for FiLM. FiLM's warm-phase mean ACC is higher than baseline, but its within-phase year-block interval includes zero, and the two chronological origins reverse the ACC ranking. These years and the 21-month exploratory period were examined during model development. The archive also lacks a verifiable operational initialization date and lead label. Independent forecast–observation years, confirmed lead metadata, and preregistered model selection are needed before a general or operational skill claim.")
    para(doc, "For regional precipitation applications, the appropriate postprocessor depends on the intended measure of skill: spatial anomaly pattern, temporal correlation, overall error, and extreme-event detection give different rankings here. Reporting these measures alongside the raw dynamical benchmark and matched neural controls makes the value of additional ocean conditioning testable. This framework can guide evaluation in other regions once independent forecast archives and observations are available.")
    heading(doc, "Data and code availability")
    para(doc, "Analysis scripts, aggregate metrics, and derived figures are retained with the revision materials and can be provided by the corresponding author. The SEAS5 archive was supplied by the National Climate Center of China; raw SEAS5, ERSST, and station observations remain subject to their providers' access conditions. The 2025 forecast–observation pairs needed for a new untouched final test were unavailable for this revision.")
    heading(doc, "Funding")
    para(doc, "This work was supported by the National Natural Science Foundation of China (grant numbers 42275050, 42130610, and 42575056).")
    heading(doc, "Declaration of generative AI and AI-assisted technologies in the writing process")
    para(doc, "OpenAI Codex assisted with manuscript drafting, language editing, and consistency checks. The authors are responsible for reviewing the analysis, interpretation, and final manuscript before submission.")
    heading(doc, "References")
    references = (
        "Hannachi, A., Jolliffe, I.T., Stephenson, D.B., 2007. Empirical orthogonal functions and related techniques in atmospheric science: A review. International Journal of Climatology 27, 1119–1152. https://doi.org/10.1002/joc.1499",
        "Huang, B., et al., 2017. Extended Reconstructed Sea Surface Temperature, Version 5 (ERSSTv5): Upgrades, Validations, and Intercomparisons. Journal of Climate 30, 8179–8205. https://doi.org/10.1175/JCLI-D-16-0836.1",
        "Jin, W., Luo, Y., Wu, T., Huang, X., Xue, W., Yu, C., 2022. Deep Learning for Seasonal Precipitation Prediction over China. Journal of Meteorological Research 36, 271–281. https://doi.org/10.1007/s13351-022-1174-7",
        "Johnson, S.J., et al., 2019. SEAS5: the new ECMWF seasonal forecast system. Geoscientific Model Development 12, 1087–1117. https://doi.org/10.5194/gmd-12-1087-2019",
        "Liang, Y., Fan, L., Yang, J., 2023. Prediction of Summer Precipitation in North China: Role of the Evolution of Sea Surface Temperature Anomalies from Boreal Winter to Spring. Journal of Climate 36, 3737–3747. https://doi.org/10.1175/JCLI-D-22-0559.1",
        "Lyu, Y., Zhu, S., Zhi, X., Wang, J., Ji, Y., Fan, Y., Dong, F., 2024. Significant advancement in subseasonal-to-seasonal summer precipitation ensemble forecast skills in China mainland through an innovative hybrid CSG-UNET method. Environmental Research Letters 19, 074055. https://doi.org/10.1088/1748-9326/ad5577",
        "NOAA Climate Prediction Center, 2026. Oceanic Niño Index historical episodes (ERSSTv5), accessed 12 November 2026. https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/oni/v5/",
        "Peng, Z., Wang, Q.J., Bennett, J.C., Schepen, A., Pappenberger, F., Pokhrel, P., Wang, Z., 2014. Statistical calibration and bridging of ECMWF System4 outputs for forecasting seasonal precipitation over China. Journal of Geophysical Research: Atmospheres 119, 7116–7135. https://doi.org/10.1002/2013JD021162",
        "Perez, E., Strub, F., Vries, H. de, Dumoulin, V., Courville, A., 2017. FiLM: Visual Reasoning with a General Conditioning Layer. https://doi.org/10.48550/arXiv.1709.07871",
        "Ronneberger, O., Fischer, P., Brox, T., 2015. U-Net: Convolutional Networks for Biomedical Image Segmentation, in: Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015. Springer, Cham, pp. 234–241. https://doi.org/10.1007/978-3-319-24574-4_28",
        "Yuan, C., Liu, J., Luo, J.-J., Guan, Z., 2019. Influences of Tropical Indian and Pacific Oceans on the Interannual Variations of Precipitation in the Early and Late Rainy Seasons in South China. Journal of Climate 32, 3681–3694. https://doi.org/10.1175/JCLI-D-18-0588.1",
        "Yuan, C., Yang, M., 2020. Interannual Variations in Summer Precipitation in Southwest China: Anomalies in Moisture Transport and the Role of the Tropical Atlantic. Journal of Climate 33, 5993–6007. https://doi.org/10.1175/JCLI-D-19-0809.1",
    )
    for reference in references:
        para(doc, reference)
    output = DELIVERY / "06_对照研究稿.docx"
    doc.save(output)
    return output


ISSUES = [
    ("EiC 1", "Structured abstract", "The abstract now has the three named sections Study Region, Study Focus, and New Hydrological Insights for the Region. Its conclusion is revised to the observed metric trade-offs."),
    ("EiC 2", "Highlights", "A separate Highlights file contains four statements. Each is at most 85 characters, including spaces, and no statement claims FiLM or all postprocessors are uniformly superior."),
    ("AE", "Limitations, predictor importance, test independence, and matched baseline", "We reconstructed raw SEAS5 precipitation, fitted atmospheric and SST transforms on training years, matched backbone width and three-seed budgets, added direct baseline and EOF-cat comparisons and symmetric ablations, and disclosed that 2023–2024 was inspected repeatedly. Two retrospective origins cover 107 nonoverlapping months in 2014–2022; they were also examined during development. The revised paper withdraws the former FiLM superiority claim."),
    ("R1 Major 1", "FiLM limitations", "Section 4 discusses limited independent climate regimes, six-month SST averaging, five-level conditional parameterization, different insertion depth, and the absence of a pristine later-year test. The negative SST controls are reported directly."),
    ("R1 Major 2", "Extreme outliers", "Section 2.1 states that target fractional anomalies were not truncated or winsorized. Training uses a valid-land mask and gradient clipping at norm 2. The extreme CSI threshold is the training-period 90th percentile, 0.9843; low-climatology amplification is acknowledged."),
    ("R1 Major 3", "ACC magnitude and additional metrics", "Table 1 reports ACC, TCC, RMSE, sign consistency, and extreme CSI. Table 2 reports paired monthly differences, Wilcoxon p-values, and two bootstrap intervals. FiLM minus SEAS5 ACC is +0.0028 (p = 0.892), and no robust FiLM ACC gain is claimed."),
    ("R1 Major 4", "U-Net and FiLM increments", "Baseline, EOF-cat, FiLM, and raw SEAS5 appear in Tables 1–2 and Figures 1–4. The explored-period baseline ACC is 0.333, FiLM 0.316, and EOF-cat 0.293. The revised conclusion attributes no incremental ACC benefit to FiLM."),
    ("R1 Major 5", "Ten predictor channels", "Section 3.3 and Figure 7 report matched-calendar-month, channelwise perturbations. Replacing tp lowers FiLM ACC by 0.263 on average; every other individual variable has an absolute effect below 0.005. Collinearity limits a necessity interpretation."),
    ("R1 Major 6", "Regional differences", "Figure 5 includes all methods for the same months with a common color range. New Table 4 reports spatial ACC for three nonoverlapping geographic partitions in both validation and explored periods. In the latter, baseline exceeds raw SEAS5 in each partition, EOF-cat trails it in each, and FiLM is only slightly above it. Southwest ACC is lowest for every system. The partitions are descriptive and post hoc; terrain, station coverage, and fractional-anomaly scaling remain hypotheses rather than demonstrated causes."),
    ("R1 Minor 1", "Abbreviations", "The revised introduction and methods define SST, EOF, PC, FiLM, ACC, TCC, RMSE, CSI, and ONI at first use. Unused BCC-CSM and PDO references were removed from the main argument."),
    ("R1 Minor 2", "Spacing and sentences", "The paper was reset as a new manuscript with complete prose descriptions of domain, period, and resolution. Captions and text were proofread for citation spacing and punctuation."),
    ("R1 Minor 3", "Predictor count", "Section 2.1 lists exactly ten fields: tp, h200, h500, slp, t2m, t850, u200, u850, v200, and v850."),
    ("R1 Minor 4", "Figure colors and legends", "Figures 1–2 were regenerated with distinguishable series and unobstructed legends; the direct monthly comparison now includes all four systems."),
    ("R1 Minor 5", "Spatial example labels and scale", "Figure 5 has direct month and method labels, including observation and all three neural models, and a shared display scale for corresponding panels."),
    ("R1 Minor 6", "EOF-cat label", "The first mention defines EOF-cat as input-channel concatenation of five EOF principal-component scores."),
    ("R2 1", "Short period, ENSO, and retrospective origins", "We agree that 21 months cannot support interannual claims. Section 3.4 and Tables 6–7 use NOAA CPC ONI rather than SST PC1 as a phase proxy. The 2018–2022 validation period contains 15 warm, 28 cold, and 17 neutral threshold months; the explored period has 13, one, and seven. We now add two refitted retrospective origins: 1994–2010 training/2011–2013 validation/2014–2019 holdout (71 months), and 1994–2015 training/2016–2019 validation/2020–2022 holdout (36 months). Tables 6–8 report their pooled 107-month ACC and TCC, including 35 warm, 36 cold, and 36 neutral months. FiLM minus baseline ACC is +0.019 in warm months, but its year-block 95% interval [−0.008, 0.050] crosses zero; the pooled difference is −0.002 [−0.016, 0.013]. Training months are not counted as out-of-sample tests. The years and model design were examined during development, so this is retrospective evidence, not an independent confirmatory test. The single explored-period cold month has no TCC estimate."),
    ("R2 2", "Preprocessing leakage", "The main analysis fits atmospheric normalization and SST climatology, EOF basis, and score scaling on 1994–2017 training only; the raw SEAS5 comparator uses its physical precipitation field and fixed 1994–2010 monthly reference. The retrospective origins refit these transformations on 1994–2010 and 1994–2015, respectively. The older numbers based on full-series normalization are withdrawn. Supplementary Figure S1 documents comparator-remapping sensitivity and identifies the invalid earlier standardized-input value."),
    ("R2 3", "Missing direct comparisons", "Figure 1 now plots the monthly FiLM-minus-baseline, FiLM-minus-EOF-cat, and FiLM-minus-SEAS5 ACC differences; the 2023 annual differences are -0.015, +0.015, and +0.025, respectively, versus -0.017, +0.034, and -0.026 in 2024. Figure 4 maps FiLM-minus-baseline and FiLM-minus-EOF-cat TCC on the same grid. Figure 5 shows all four systems for the same months and display range. Tables 1–3 quantify the direct comparisons. FiLM has no general advantage over baseline."),
    ("R2 4", "EOF-cat fairness and degradation", "The current comparison gives both SST models the same five scores from a six-month SST average, the same 32-channel backbone width, learning rate, dropout, 100 epochs, three seeds, and validation ACC checkpoint rule. Baseline, EOF-cat, and FiLM have 3,352,289, 3,353,729, and 3,452,513 trainable parameters, respectively. EOF-cat broadcasts scores at the input whereas FiLM modulates five encoder depths, so insertion depth and added capacity are not equivalent; prior exploratory tuning cannot be retroactively equalized. Section 2.2 states these differences rather than attributing the contrast solely to the FiLM operator. The SST-zero control raises EOF-cat ACC from 0.293 to 0.319, and we do not ascribe its loss uniquely to redundant representation."),
    ("R2 5", "Paired inference and TCC mask", "Table 2 gives paired monthly ACC differences, Wilcoxon p-values, and month and year-block bootstrap intervals. Table 3 adds paired spatial-mean TCC differences and year-block resampling: in 2018–2022, FiLM-minus-baseline is -0.010 (95% interval [-0.017, 0.002]) and FiLM-minus-EOF-cat is +0.004 ([-0.003, 0.026]). The 2023–2024 TCC block ranges have only three distinct combinations, so we report them descriptively rather than as confidence intervals. Figure 3 stippling denotes nominal gridwise p < 0.05, not field significance; Figure 4 maps the two direct TCC differences. No significant FiLM superiority is claimed."),
    ("R2 6", "Symmetric ablation and negative control", "Table 5 and Figure 6 evaluate full input, zeroed SST scores, and zeroed atmospheric maps for baseline, EOF-cat, and FiLM. Same-calendar-month SST-score replacement is also reported. FiLM ACC rises from 0.316 to 0.318 with zeros and 0.324 under replacement; therefore, SST-driven FiLM parameters are not presented as evidence of improved teleconnection skill."),
]

CN_ISSUES = [
    ("主编 1", "结构式摘要", "摘要已按 Study Region、Study Focus、New Hydrological Insights for the Region 三个具名部分重写，并将结论改为按指标区分的实证结果。"),
    ("主编 2", "Highlights", "另附四条 Highlights，每条均不超过含空格 85 字符；不再宣称 FiLM 或三种后处理方法全面优于 SEAS5。"),
    ("副主编", "局限、变量重要性和对照公平性", "重新构建原始 SEAS5 降水对照，训练期拟合大气及 SST 预处理，统一三种模型的主干宽度与三随机种子预算，补充直接对照和对称消融，并说明 2023—2024 年结果曾被反复查看。原先的 FiLM 优越性结论已撤回。"),
    ("审稿人 1 主要 1", "FiLM 局限", "讨论新增训练气候状态样本有限、六个月 SST 平均削弱时序信息、五层调制参数估计不稳定、插入深度不同以及缺少新年份未查看测试集等限制，并如实报告 SST 阴性对照。"),
    ("审稿人 1 主要 2", "极端值处理", "目标降水异常没有截尾或缩尾。训练使用有效陆地掩膜、梯度范数上限 2；极端 CSI 阈值取训练期第 90 百分位 0.9843，并讨论少雨区分母较小造成的异常放大。"),
    ("审稿人 1 主要 3", "ACC 幅度与其他指标", "表 1 同时报告 ACC、TCC、RMSE、符号一致率及极端 CSI；表 2 报告月度配对差、Wilcoxon p 值和两种 bootstrap 区间。FiLM 相对 SEAS5 的 ACC 仅高 0.0028，p=0.892，不作为稳健增益。"),
    ("审稿人 1 主要 4", "U-Net 与 FiLM 增量", "表 1—2 和图 1—4 直接展示 Baseline、EOF-cat、FiLM 与原始 SEAS5。探索期 ACC 分别为 0.333、0.293、0.316、0.314；没有将 FiLM 描述为相对基线的正增量。"),
    ("审稿人 1 主要 5", "十个大气变量", "图 7 给出同历月整幅变量场替换的重要性。tp 平均使 FiLM ACC 降低 0.263，其他变量各自绝对变化小于 0.005；相关变量之间可能互相替代，因此不据此判定变量无用。"),
    ("审稿人 1 主要 6", "区域差异", "图 5 将四种模型置于同一月份和同一色标；新增表 4，分别报告验证期与探索期三个互不重叠地理分区的区域 ACC。探索期 Baseline 在三个区均高于原始 SEAS5，EOF-cat 在三个区均低于原始 SEAS5，FiLM 仅略高；西南区四系统 ACC 均最低。分区分析属于事后描述，地形、观测站分布和少雨区比例异常仅为待检验解释。"),
    ("审稿人 1 次要 1", "缩写", "正文首次出现时解释 SST、EOF、PC、FiLM、ACC、TCC、RMSE、CSI、ONI；未使用的 BCC-CSM 和 PDO 表述已移除。"),
    ("审稿人 1 次要 2", "空格和完整句", "新稿统一重排，研究区域、时间和分辨率均改为完整句，并核对引文、标点与图注。"),
    ("审稿人 1 次要 3", "变量数量", "方法部分明确列出 tp、h200、h500、slp、t2m、t850、u200、u850、v200、v850 共十项。"),
    ("审稿人 1 次要 4", "颜色和图例", "重新绘制月度 ACC 和指标对照图，四系统采用可区分配色，图例不遮挡数据。"),
    ("审稿人 1 次要 5", "空间图标签", "图 5 直接标注月份、观测、SEAS5 和三种神经网络方法，各对应面板采用相同显示范围。"),
    ("审稿人 1 次要 6", "EOF-cat 定义", "首次出现时将 EOF-cat 定义为五个 EOF 主成分得分在输入通道拼接。"),
    ("审稿人 2 第 1 点", "短时段、ENSO 和回溯训练起点", "承认 21 个月不足以支持年际泛化。表 6—7 改用 NOAA CPC ONI 分组：2018—2022 年验证期含暖相 15、冷相 28、中性 17 个月；探索期为 13、1、7 个月。新增两个重训回溯起点：1994—2010 年训练、2011—2013 年验证、2014—2019 年留出 71 个月；1994—2015 年训练、2016—2019 年验证、2020—2022 年留出 36 个月。表 6—8 报告合并的 107 个月 ACC 与 TCC，其中暖相 35、冷相 36、中性 36 个月。暖相 FiLM 减 Baseline 的 ACC 为 +0.019，但按年分块 95% 区间 [−0.008, 0.050] 跨零；合并差值 −0.002，区间 [−0.016, 0.013]。训练月份不计为样本外测试；这些年份和模型设计曾在开发中查看，因此仅属事后回溯，不能视为独立确认性检验。探索期单个冷相月不计算 TCC。"),
    ("审稿人 2 第 2 点", "预处理泄露", "主分析的大气变量归一化及 SST 气候态、EOF 基底和主成分标准化仅在 1994—2017 年训练数据拟合；原始 SEAS5 降水使用物理量及 1994—2010 年固定月气候态。两个回溯起点分别在 1994—2010 年和 1994—2015 年重新拟合变换。撤回全时段归一化所得旧数值。补充图 S1 展示对照插值方式的敏感性，并标出旧稿误用的标准化输入数值。"),
    ("审稿人 2 第 3 点", "完整直接对照", "图 1 直接画出 FiLM 相对 Baseline、EOF-cat、SEAS5 的逐月 ACC 差；2023 年三者年均差分别为 -0.015、+0.015、+0.025，2024 年分别为 -0.017、+0.034、-0.026。图 4 显示 FiLM 相对 Baseline 和 EOF-cat 的逐格 TCC 差，图 5 对齐四种方法的相同月份及色标；表 1—3 给出整体、配对及 TCC 统计。"),
    ("审稿人 2 第 4 点", "EOF-cat 公平性与退化", "两种 SST 模型共享由六个月 SST 均值得出的五个得分，主干均为 32 通道，学习率、dropout、100 轮训练、三个随机种子及验证 ACC 选模规则相同。Baseline、EOF-cat、FiLM 的可训练参数量分别为 3,352,289、3,353,729、3,452,513。EOF-cat 在输入层拼接，FiLM 调制五层，插入深度及新增容量不等同；此前探索性调参也无法事后变成完全等预算。正文第 2.2 节如实说明这一点，不把性能差异唯一归因于 FiLM 算子。EOF-cat 的 SST 置零 ACC 由 0.293 升至 0.319，不能简单归因于冗余表示。"),
    ("审稿人 2 第 5 点", "配对推断与 TCC 显著性", "表 2 给出月度 ACC 配对检验及逐月、按年分块区间。表 3 补充空间平均 TCC 的配对年分块结果：2018—2022 年 FiLM 减 Baseline 为 -0.010，95% 区间 [-0.017, 0.002]；FiLM 减 EOF-cat 为 +0.004，区间 [-0.003, 0.026]。2023—2024 年仅有三种不同的两年分块组合，因此只报告描述性范围，不声称显著。图 3 点标仅代表逐格名义 p<0.05，图 4 展示两组直接 TCC 差值；已删除显著优于对照的表述。"),
    ("审稿人 2 第 6 点", "对称消融和阴性对照", "表 5、图 6 对 Baseline、EOF-cat、FiLM 同时进行完整输入、SST 得分置零和局地大气输入置零，并补充同历月 SST 得分替换。FiLM ACC 分别为 0.316、0.318 和替换后的 0.324，不将调制参数相关性解释成预报能力提升的证据。"),
]


def response(chinese=False):
    doc = Document()
    setup(doc, chinese)
    doc.styles["Normal"].font.size = Pt(10.5)
    doc.styles["Normal"].paragraph_format.space_after = Pt(4)
    doc.styles["Normal"].paragraph_format.line_spacing = 1.12
    doc.styles["Heading 2"].paragraph_format.space_before = Pt(8)
    doc.styles["Heading 2"].paragraph_format.space_after = Pt(3)
    if chinese:
        doc.sections[0].top_margin = Inches(0.6)
        doc.sections[0].bottom_margin = Inches(0.6)
    para(doc, "审稿意见逐项回复" if chinese else "Response to the Editor and Reviewers", "Title")
    para(doc, "Manuscript EJRH-D-26-03375")
    if chinese:
        para(doc, "感谢编辑和审稿人的意见。重算发现旧稿对原始 SEAS5 的定义及部分预处理存在问题，因此本轮撤回 FiLM 稳定领先的主张。2023—2024 年已被反复查看，不再称完全独立测试。正文和两份回复均采用同一重算数值。")
    else:
        para(doc, "We thank the editor and reviewers. The manuscript has been retitled to reflect its revised comparative objective. Reanalysis identified an incorrect raw-SEAS5 comparator and preprocessing that required training-only refitting. We withdraw the claim of stable FiLM superiority. The 2023–2024 period was inspected repeatedly during development and is now explicitly exploratory. All revised claims use the same corrected results.")
    if chinese:
        para(doc, "核心结果：2023—2024 年 SEAS5、Baseline、EOF-cat、FiLM 的 ACC 分别为 0.314、0.333、0.293、0.316。2014—2022 年两起点合并的 107 个月回溯值依次为 0.295、0.289、0.258、0.287。两组结果均不支持三种方法全面优于 SEAS5；回溯年份曾在开发中查看。")
    else:
        para(doc, "Core result: exploratory 2023–2024 ACC is 0.314 (SEAS5), 0.333 (baseline), 0.293 (EOF-cat), and 0.316 (FiLM). The two-origin retrospective 2014–2022 sequence gives 0.295, 0.289, 0.258, and 0.287 over 107 months in the same order. These years were previously examined and do not support universal postprocessor superiority.")
    for label, issue, answer in (CN_ISSUES if chinese else ISSUES):
        heading(doc, f"{label}  {issue}", 2)
        para(doc, answer)
    output = DELIVERY / ("06_审稿回复中文.docx" if chinese else "06_Response_EN.docx")
    doc.save(output)
    return output


def highlights():
    lines = [
        "Corrected raw SEAS5 comparison changes the postprocessing model ranking.",
        "U-Net leads spatial ACC; EOF-cat leads TCC, RMSE, and extreme CSI.",
        "FiLM shows no stable ACC gain from simultaneous SST conditioning.",
        "Two retrospective origins show no stable FiLM ACC advantage.",
    ]
    assert 3 <= len(lines) <= 5 and all(len(line) <= 85 for line in lines)
    doc = Document()
    setup(doc)
    para(doc, "Highlights", "Title")
    for line in lines:
        para(doc, line, "List Bullet")
    output = DELIVERY / "Highlights.docx"
    doc.save(output)
    return output


def supplement():
    doc = Document()
    setup(doc)
    para(doc, "Supplementary Material", "Title")
    para(doc, "Manuscript EJRH-D-26-03375")
    figure(
        doc,
        "FigS1_SEAS5_sensitivity.png",
        "Supplementary Figure S1. Sensitivity of exploratory-period spatial anomaly correlation to construction of the SEAS5 precipitation comparator. The primary comparator uses the first monthly total-precipitation field in each target-month archive, bilinear remapping, and a nonnegative constraint (ACC 0.314). Nonnegative bicubic remapping gives ACC 0.288. The earlier value of 0.281 was calculated from a standardized neural-network input channel and is not a valid raw-SEAS5 skill estimate.",
    )
    output = DELIVERY / "06_Supplement_S1.docx"
    doc.save(output)
    return output


def main():
    DELIVERY.mkdir(parents=True, exist_ok=True)
    outputs = [manuscript(), response(False), response(True), highlights(), supplement()]
    print("\n".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
