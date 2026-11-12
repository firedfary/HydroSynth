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
OLD = DELIVERY.parent / "05_返修交付包/05_大修修改稿.docx"
PATHS = SubprojectPaths(__file__)
RESULTS = PATHS.results_dir
METRICS = json.loads((DELIVERY / "summary.json").read_text(encoding="utf-8"))["metrics"]
ABLATION = json.loads((RESULTS / "revision06_ablation_v1/ablation.json").read_text(encoding="utf-8"))
ENSO = json.loads((RESULTS / "revision06_eval_v2/enso_metrics.json").read_text(encoding="utf-8"))
ROLLING = json.loads((RESULTS / "revision06_rolling_v1/metrics.json").read_text(encoding="utf-8"))
if "ensemble" not in ROLLING:
    raise RuntimeError("Rolling holdout has not finished")


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
    para(doc, "Comparing U-Net Postprocessing and SST Conditioning for Monthly Precipitation over China", "Title")
    para(doc, "Fan YUE1,3, Xiaojuan WANG1*, Rong AI2, Yinzhong WU3, Guolin FENG2,4, and Zhiqiang GONG2")
    for line in (
        "1 College of Electronic Information Engineering, Suzhou University of Technology, Suzhou 215500, China",
        "2 Laboratory for Climate Studies, National Climate Center, Beijing 100081, China",
        "3 College of Physics and Technology, Suzhou University of Science and Technology, Suzhou 215009, China",
        "4 College of Physical Science and Technology, Yangzhou University, Yangzhou 225009, China",
    ):
        para(doc, line)
    heading(doc, "Abstract")
    test = METRICS["test"]["ensemble"]
    para(doc, "Study Region: Mainland China, where monsoon variability, aridity, and complex terrain challenge monthly precipitation prediction.")
    para(doc, "Study Focus: We compare direct SEAS5 precipitation with three matched 32-channel U-Net postprocessors: atmospheric predictors alone, the same predictors with five global sea surface temperature (SST) empirical orthogonal function (EOF) scores concatenated at the input, and the scores supplied through feature-wise linear modulation (FiLM). Atmospheric preprocessing and SST EOFs are fitted on 1994–2017 data; model selection uses 2018–2022; January 2023–September 2024 is an exploratory evaluation because its outcomes were inspected during development. A separate 2020–2021 retrospective holdout is retrained from an earlier origin.")
    para(doc, "New Hydrological Insights for the Region: On the 21-month evaluation, spatial anomaly correlation (ACC) is "
         f"{f(test['seas5']['mean_acc'])} for SEAS5, {f(test['baseline']['mean_acc'])} for the baseline, "
         f"{f(test['eof_cat']['mean_acc'])} for EOF-cat, and {f(test['film']['mean_acc'])} for FiLM. "
         "All three postprocessors lower root mean square error and improve extreme-event critical success index in this period, but EOF-cat loses spatial ACC relative to SEAS5, and FiLM has no robust incremental ACC gain. Removing or temporally mismatching the SST scores does not reduce FiLM ACC. The results support a metric-dependent comparison of correction strategies, not a general claim that SST modulation or every postprocessor outperforms SEAS5.")
    para(doc, "Keywords: precipitation postprocessing; SEAS5; U-Net; empirical orthogonal functions; feature-wise linear modulation; China")

    heading(doc, "1 Introduction")
    para(doc, "Monthly precipitation anomalies over China depend on large-scale circulation and local terrain, while dynamical seasonal prediction systems retain substantial regional error. SEAS5 provides the dynamical fields evaluated here (Johnson et al., 2019). Neural postprocessing can learn spatial corrections from gridded atmospheric predictors. U-Net offers a multiscale encoder–decoder with skip connections (Ronneberger et al., 2015), and SST modes may carry information about remote ocean–atmosphere variability (Hannachi et al., 2007). FiLM can condition intermediate features on an external vector through learned channelwise scales and offsets (Perez et al., 2017).")
    para(doc, "A more expressive conditioning mechanism is useful only if it improves forecasts under matched inputs and a defensible comparator. The initial version of this study interpreted a standardized atmospheric input channel as raw SEAS5 precipitation and allowed full-record information into some preprocessing. Reconstructing the raw SEAS5 comparator and fitting transformations on the training period changes the ranking. We therefore ask which method performs better on each metric, whether FiLM adds value beyond direct EOF concatenation and the unconditioned U-Net, and whether any benefit survives a second chronological origin. These questions make negative and mixed results informative rather than treating architectural complexity as an advantage by itself.")

    heading(doc, "2 Data and methods")
    heading(doc, "2.1 Data and chronology", 2)
    para(doc, "Monthly SEAS5 archive files from January 1994 through September 2024 supply ten China-domain atmospheric fields: total precipitation (tp), geopotential height at 200 and 500 hPa (h200, h500), sea-level pressure (slp), 2-m temperature (t2m), 850-hPa temperature (t850), and zonal and meridional winds at 200 and 850 hPa (u200, u850, v200, v850). The archived 1° fields are bilinearly mapped to a 120 × 140 grid spanning 70–140°E and 0–60°N. Precipitation observations are based on 2,374 National Meteorological Information Center stations and their prepared gridded target field; 5,243 land cells contain valid observations across the analysis. Global NOAA ERSST v5 fields provide SST predictors. Three archive months are absent (September and October 2011; January 2017), leaving 366 aligned months.")
    para(doc, "The main chronology assigns 285 months in 1994–2017 to fitting, 60 months in 2018–2022 to validation and checkpoint selection, and 21 months in January 2023–September 2024 to exploratory evaluation. All data-dependent atmospheric normalization, SST climatology, EOF basis, and score scaling are fitted on the first period. The observed and raw SEAS5 precipitation anomalies use a fixed 1994–2010 calendar-month reference climatology, which lies within training. Their fractional anomaly is precipitation minus the corresponding calendar-month mean, divided by that mean plus 0.0001. For the SEAS5 comparator, we select the first monthly tp field from each target-month archive file, bilinearly interpolate it, and constrain it nonnegative before anomaly conversion. The file metadata identify the first valid month but do not establish an operational initialization date or lead label. No standardized network input is used as the raw comparator.")
    para(doc, "The target's extreme anomalies are retained without winsorization. The 90th percentile of training anomalies, 0.9843, defines extreme hits. This fractional-anomaly scale can magnify small absolute precipitation errors where climatological rainfall is low, so both aggregate skill and regional maps require caution. A 2020–2021 retrospective holdout uses a separate fitting origin (1994–2015 training, 2016–2019 validation) with all transforms and models refitted. That period was previously inspected and is a sensitivity check, not a pristine new test.")
    heading(doc, "2.2 SST representation and model comparison", 2)
    para(doc, "The stored SST predictor contains six preceding monthly fields for each target month. We average those six fields to one SST map, subtract a training-only SST climatology, fit EOFs on the training maps, and project later maps onto the frozen basis. The first five standardized principal-component (PC) scores are shared by EOF-cat and FiLM. Averaging removes within-window timing; the design should not be described as a six-step PC sequence or a one-month PC lag.")
    para(doc, "All neural variants use the same 32-channel base U-Net, with five encoder levels, a mirrored decoder, skip connections, batch normalization, ReLU activations, and spatial dropout. Baseline receives the ten atmospheric channels. EOF-cat broadcasts the five PC scores as constant spatial maps and appends them to the ten input channels. FiLM receives the ten atmospheric channels plus a two-layer index encoder that maps the five scores to channelwise scale and offset parameters at each encoder level. Its transform applies the predicted scale to each feature channel and adds the predicted offset; zero initialization of the final conditioning layer starts near the unmodulated mapping. The variants share backbone width, PCs, split, optimizer, and seed set, but input concatenation and five-level modulation still differ in insertion depth and parameter count. This comparison does not isolate a single causal effect of the FiLM operator.")
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
    figure(doc, "Fig2_monthly_ACC.png", "Figure 1. Monthly spatial ACC in the 21-month exploratory evaluation, including all three learned models and raw SEAS5.")
    figure(doc, "Fig3_metric_tradeoff.png", "Figure 2. Metric-dependent ranking of the four systems in the exploratory evaluation; metrics have different units and directions.")
    heading(doc, "3.2 Paired comparisons and spatial variation", 2)
    para(doc, "Table 2. Paired monthly spatial ACC differences in January 2023–September 2024. Confidence intervals are descriptive because the period was repeatedly examined; the calendar-year-block interval has only two available years.")
    table(doc, ("Comparison", "Mean Δ", "Positive", "Wilcoxon p", "Month 95% CI", "Year block 95% CI"),
          [paired("FiLM – baseline", "film_minus_baseline"),
           paired("FiLM – EOF-cat", "film_minus_eof_cat"),
           paired("FiLM – SEAS5", "film_minus_seas5"),
           paired("Baseline – SEAS5", "baseline_minus_seas5")])
    para(doc, "FiLM exceeds EOF-cat in 17 of 21 months on ACC, but its mean ACC remains below baseline by 0.016. FiLM's gain over raw SEAS5 is 0.0028 and the paired Wilcoxon p-value is 0.892. The 21-month spatial mean TCC values are 0.398 for SEAS5, 0.394 for baseline, 0.407 for EOF-cat, and 0.398 for FiLM. The differences are small and should not be interpreted as confirmed multi-year superiority. Maps show heterogeneous cells rather than a uniform FiLM gain.")
    figure(doc, "Fig4_temporal_corr.png", "Figure 3. Gridwise TCC for all four systems over 21 months. Dots indicate nominal two-sided p < 0.05 at displayed cells; they are not a spatial field-significance mask.")
    figure(doc, "Fig5_spatial_examples.png", "Figure 4. Observed and predicted fractional precipitation anomalies in January, April, July, and October 2023. The same display range is used for every panel; values beyond it are clipped for visualization only.", 6.5)
    para(doc, "The example maps reveal common smoothing and regional differences across the monsoon domain, Tibetan Plateau, and arid northwest. The strong dependence of skill on raw forecast precipitation, and the fractional target's sensitivity to low-rainfall climatology, are plausible contributors; these maps alone cannot identify a specific atmospheric mechanism or establish a regional causal attribution.")
    heading(doc, "3.3 Conditioning controls and predictor importance", 2)
    ab = ABLATION["ensemble"]
    para(doc, "Table 3. Symmetric input controls on the exploratory period. PC zeroing and local-input zeroing are out-of-distribution perturbations, so their score changes diagnose dependence but are not replacement-model benchmarks.")
    table(doc, ("Input condition", "Baseline ACC", "EOF-cat ACC", "FiLM ACC"),
          [(label, f(ab[key]["baseline"]["mean_acc"]),
            f(ab[key]["eof_cat"]["mean_acc"]), f(ab[key]["film"]["mean_acc"]))
           for label, key in (("Full", "full"), ("SST PCs = 0", "pc_zero"),
                              ("Atmospheric maps = 0", "local_zero"))])
    para(doc, "Setting FiLM's standardized SST PCs to zero raises ACC from 0.316 to 0.318; replacing the scores with same-calendar-month values from 2018–2022 gives mean ACC 0.324 across ten trials. EOF-cat likewise rises from 0.293 to 0.319 when PCs are zeroed. Removing the local atmospheric maps sharply degrades all three architectures. These controls do not establish that simultaneous SST conditioning improves the evaluated forecast, although some other metrics can change in the opposite direction.")
    figure(doc, "Fig6_symmetric_ablation.png", "Figure 5. Symmetric SST and local-predictor input controls for baseline, EOF-cat, and FiLM.")
    para(doc, "Replacing one atmospheric predictor map at a time with a same-calendar-month map from the validation period makes tp dominant: its FiLM ACC reduction averages 0.263 across ten trials. Every other predictor has an absolute mean effect below 0.005. Correlated circulation variables can substitute for one another, so a small single-variable permutation effect does not prove that the variable is useless. The test-period use of these controls remains exploratory.")
    figure(doc, "Fig7_predictor_importance.png", "Figure 6. FiLM predictor permutation importance, measured as mean decrease in spatial ACC when one full atmospheric map is replaced.", 5.8)
    heading(doc, "3.4 ENSO phase and retrospective origin", 2)
    para(doc, "We classify each month using the NOAA Climate Prediction Center historical Oceanic Niño Index (ONI) centered three-month value: warm at ≥ +0.5 °C, cold at ≤ −0.5 °C, and neutral otherwise. These threshold categories are not declarations of official ENSO episodes. In 2023–2024 there are 13 warm, one cold, and seven neutral months. The single cold month does not permit a temporal correlation estimate, and no phase-specific score from that period is a stable regime generalization. Table 4 reports the longer validation period and the explored period to show the sample composition and metric variability.")
    rows = []
    for period, key in (("2018–2022 validation", "validation"),
                        ("2023–2024 explored", "test")):
        for phase in ("warm", "cold", "neutral"):
            row = ENSO["periods"][key][phase]
            rows.append((period, phase, row["months"],
                         *[f(row["metrics"][name]["mean_acc"]) for name in
                           ("seas5", "baseline", "eof_cat", "film")]))
    para(doc, "Table 4. Spatial ACC by ONI threshold category. The exploratory cold category contains only one month.")
    table(doc, ("Period", "ONI", "n", "SEAS5", "Baseline", "EOF-cat", "FiLM"), rows)
    tcc_rows = []
    for period, key in (("2018–2022 validation", "validation"),
                        ("2023–2024 explored", "test")):
        for phase in ("warm", "cold", "neutral"):
            row = ENSO["periods"][key][phase]
            tcc_rows.append((period, phase, row["months"],
                             *[f(row["metrics"][name]["mean_tcc"]) for name in
                               ("seas5", "baseline", "eof_cat", "film")]))
    para(doc, "Table 5. Spatially averaged TCC within ONI threshold categories. The seven-month neutral estimate is unstable; TCC is omitted for one cold month.")
    table(doc, ("Period", "ONI", "n", "SEAS5", "Baseline", "EOF-cat", "FiLM"), tcc_rows)
    hold = ROLLING["ensemble"]["holdout"]
    caption = para(doc, "Table 6. Retrospective 2020–2021 holdout after refitting training-only transforms and all three models from the earlier origin. This window was previously inspected and is not a new confirmatory test.")
    caption.paragraph_format.page_break_before = True
    caption.paragraph_format.keep_with_next = True
    table(doc, ("Model", "ACC", "TCC", "RMSE", "Sign %", "CSI"),
          [(label, f(hold[name]["mean_acc"]), f(hold[name]["mean_tcc"]),
            f(hold[name]["rmse"]), f(hold[name]["sc_pct"], 1),
            f(hold[name]["csi"])) for label, name in
           (("SEAS5", "seas5"), ("Baseline", "baseline"),
            ("EOF-cat", "eof_cat"), ("FiLM", "film"))])
    para(doc, "The retrospective window contains three warm, thirteen cold, and eight neutral ONI-threshold months. Its rankings test whether the FiLM claim persists under a different origin. Both periods were inspected during model development, and both should be reported in full when drawing conclusions.")

    heading(doc, "4 Discussion")
    para(doc, "The most direct empirical explanation for the modest value of SST conditioning is that the local tp field carries most of the information used by the network. Its permutation effect is more than an order of magnitude larger than any individual circulation channel, and SST score removal or temporal mismatch does not hurt FiLM spatial ACC. The six-month SST mean may also suppress phase timing relevant to monthly rainfall, and a single five-score vector cannot describe regionally varying teleconnections. These latter points are plausible hypotheses rather than causal results of the present ablations. The training set contains 285 monthly maps but far fewer independent climate regimes; five-level conditional scaling may therefore be difficult to estimate robustly. Equal backbone width and training budget improve comparability, but EOF-cat changes the first input convolution while FiLM adds a multilayer branch at five depths, so architecture and parameter-count effects remain intertwined.")
    para(doc, "The corrected raw SEAS5 comparator materially changes the interpretation. Bilinear remapping with a nonnegative precipitation constraint yields 0.314 exploratory ACC. Nonnegative bicubic remapping gives 0.288, and the previously reported 0.281 came from a standardized neural input rather than original model precipitation. The visual and statistical comparison must use the same explicitly stated physical reconstruction; interpolation sensitivity is a limitation rather than evidence for a desired model ranking.")
    para(doc, "The study has several additional limits. First, repeated examination of 2023–2024 prevents a confirmatory test claim. Second, a 21-month period and two calendar-year blocks give weak temporal and phase-specific inference, especially for gridwise TCC. Third, extreme CSI is based on a single training threshold and does not quantify rainfall intensity calibration. Fourth, the station-derived gridded target has uneven sampling, and fractional anomalies amplify error in dry regions. Fifth, the 2020–2021 origin has a different training span and was also previously explored. Sixth, the current ablations are inference-time perturbations outside the joint training distribution, not retrained feature-removal experiments. Finally, the archive lacks initialization metadata, so the operational forecast lead cannot be independently verified from these files. A future study should pre-register architecture and metrics, compare multiple earlier rolling origins, hold back newly acquired years, and evaluate alternate SST windows and regional conditioning while preserving the same input and compute budget.")
    heading(doc, "5 Conclusions")
    para(doc, "Under training-fitted preprocessing and a physically reconstructed SEAS5 comparator, the three U-Net postprocessors show different metric trade-offs. Baseline has the best spatial ACC and sign consistency in the explored 2023–2024 period; EOF-cat has the best mean TCC, RMSE, and extreme CSI; FiLM is not consistently best and does not show a stable ACC benefit from simultaneous SST scores. All three improve RMSE and CSI in this period, but EOF-cat loses ACC relative to SEAS5 and none supports a universal superiority claim. The 2020–2021 retrained origin and short-period uncertainty further bound generalization. This comparison establishes a reproducible evaluation and identifies the conditions that any claimed FiLM advantage must withstand in a future independent test.")
    heading(doc, "Data and code availability")
    para(doc, "Analysis scripts, aggregate metrics, and derived figures are retained with the revision materials and can be provided by the corresponding author. Raw SEAS5, ERSST, and station observations remain subject to their providers' access conditions. The 2025 forecast–observation pairs needed for a new untouched final test were unavailable for this revision.")
    heading(doc, "Declarations")
    para(doc, "Funding: National Natural Science Foundation of China, grant numbers 42275050, 42130610, and 42575056. Competing interests: the authors declare none. Author contributions should be confirmed by all authors before submission. Generative AI tools assisted with drafting and consistency checks; all analyses, interpretations, and final wording require author verification.")
    heading(doc, "References")
    old = Document(OLD)
    prefixes = ("Hannachi", "Perez", "Ronneberger")
    references = []
    for p in old.paragraphs:
        if p.style.name == "Bibliography" and p.text.startswith(prefixes):
            references.append(p.text)
    references.extend((
        "Johnson, S.J., et al., 2019. SEAS5: the new ECMWF seasonal forecast system. Geoscientific Model Development 12, 1087–1117. https://doi.org/10.5194/gmd-12-1087-2019",
        "NOAA Climate Prediction Center, 2026. Oceanic Niño Index historical episodes (ERSSTv5), accessed 12 November 2026. https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php",
    ))
    for reference in sorted(references, key=str.casefold):
        para(doc, reference)
    output = DELIVERY / "06_对照研究稿.docx"
    doc.save(output)
    return output


ISSUES = [
    ("EiC 1", "Structured abstract", "The abstract now has the three named sections Study Region, Study Focus, and New Hydrological Insights for the Region. Its conclusion is revised to the observed metric trade-offs."),
    ("EiC 2", "Highlights", "A separate Highlights file contains four statements. Each is at most 85 characters, including spaces, and no statement claims FiLM or all postprocessors are uniformly superior."),
    ("AE", "Limitations, predictor importance, test independence, and matched baseline", "We reconstructed raw SEAS5 precipitation, fitted atmospheric and SST transforms on training years, matched backbone width and three-seed budgets, added direct baseline and EOF-cat comparisons and symmetric ablations, and disclosed that 2023–2024 was inspected repeatedly. The revised paper withdraws the former FiLM superiority claim."),
    ("R1 Major 1", "FiLM limitations", "Section 4 discusses limited independent climate regimes, six-month SST averaging, five-level conditional parameterization, different insertion depth, and the absence of a pristine later-year test. The negative SST controls are reported directly."),
    ("R1 Major 2", "Extreme outliers", "Section 2.1 states that target fractional anomalies were not truncated or winsorized. Training uses a valid-land mask and gradient clipping at norm 2. The extreme CSI threshold is the training-period 90th percentile, 0.9843; low-climatology amplification is acknowledged."),
    ("R1 Major 3", "ACC magnitude and additional metrics", "Table 1 reports ACC, TCC, RMSE, sign consistency, and extreme CSI. Table 2 reports paired monthly differences, Wilcoxon p-values, and two bootstrap intervals. FiLM minus SEAS5 ACC is +0.0028 (p = 0.892), and no robust FiLM ACC gain is claimed."),
    ("R1 Major 4", "U-Net and FiLM increments", "Baseline, EOF-cat, FiLM, and raw SEAS5 appear in Tables 1–2 and Figures 1–4. The explored-period baseline ACC is 0.333, FiLM 0.316, and EOF-cat 0.293. The revised conclusion attributes no incremental ACC benefit to FiLM."),
    ("R1 Major 5", "Ten predictor channels", "Section 3.3 and Figure 6 report matched-calendar-month, channelwise perturbations. Replacing tp lowers FiLM ACC by 0.263 on average; every other individual variable has an absolute effect below 0.005. Collinearity limits a necessity interpretation."),
    ("R1 Major 6", "Regional differences", "Figure 4 includes all methods for the same months with a common color range. Section 3.2 discusses heterogeneous regional error and identifies terrain, station coverage, and fractional-anomaly scaling as plausible explanations, without presenting these as causal findings."),
    ("R1 Minor 1", "Abbreviations", "The revised introduction and methods define SST, EOF, PC, FiLM, ACC, TCC, RMSE, CSI, and ONI at first use. Unused BCC-CSM and PDO references were removed from the main argument."),
    ("R1 Minor 2", "Spacing and sentences", "The paper was reset as a new manuscript with complete prose descriptions of domain, period, and resolution. Captions and text were proofread for citation spacing and punctuation."),
    ("R1 Minor 3", "Predictor count", "Section 2.1 lists exactly ten fields: tp, h200, h500, slp, t2m, t850, u200, u850, v200, and v850."),
    ("R1 Minor 4", "Figure colors and legends", "Figures 1–2 were regenerated with distinguishable series and unobstructed legends; the direct monthly comparison now includes all four systems."),
    ("R1 Minor 5", "Spatial example labels and scale", "Figure 4 has direct month and method labels, including observation and all three neural models, and a shared display scale for corresponding panels."),
    ("R1 Minor 6", "EOF-cat label", "The first mention defines EOF-cat as input-channel concatenation of five EOF principal-component scores."),
    ("R2 1", "Short period, ENSO, and another origin", "We agree that 21 months cannot support interannual claims. Section 3.4 uses NOAA CPC ONI rather than SST PC1 as a phase proxy. The explored period has 13 warm, one cold, and seven neutral threshold months. The retrained 2020–2021 origin has three warm, thirteen cold, and eight neutral months. We report all methods in that holdout and identify both periods as previously explored. The single cold month has no TCC estimate."),
    ("R2 2", "Preprocessing leakage", "The current analysis rebuilds all atmospheric normalization and SST climatology, EOF basis, and score scaling from 1994–2017 training only; the raw SEAS5 comparator uses its physical precipitation field and fixed 1994–2010 monthly reference. For the 2020–2021 origin, transforms are refitted on 1994–2015. The older numbers based on full-series normalization are withdrawn."),
    ("R2 3", "Missing direct comparisons", "Figures 1–4 and Tables 1–2 now show baseline, EOF-cat, FiLM, and SEAS5 together. We report FiLM minus EOF-cat and FiLM minus baseline monthly ACC and spatial TCC descriptions. FiLM has no general advantage over baseline."),
    ("R2 4", "EOF-cat fairness and degradation", "The current comparison gives both SST models the same five scores from a six-month SST average, the same 32-channel backbone width, learning rate, dropout, epochs, seed set, and validation ACC checkpoint rule. EOF-cat broadcasts scores at the input whereas FiLM modulates five encoder depths, so insertion depth and added parameters are not equivalent. Its explored ACC loss is described, and the SST-zero control raises EOF-cat ACC from 0.293 to 0.319; we do not ascribe the loss uniquely to redundant representation."),
    ("R2 5", "Paired inference and TCC mask", "Table 2 gives monthly paired Wilcoxon tests and month and calendar-year-block bootstrap intervals. The 21 months span only two years, making the year-block interval descriptive. Figure 3 includes a nominal gridwise TCC p < 0.05 stipple; it is not called field significance. All language asserting significant FiLM superiority has been removed."),
    ("R2 6", "Symmetric ablation and negative control", "Table 3 and Figure 5 evaluate full input, zeroed SST scores, and zeroed atmospheric maps for baseline, EOF-cat, and FiLM. Same-calendar-month SST-score replacement is also reported. FiLM ACC rises from 0.316 to 0.318 with zeros and 0.324 under replacement; therefore, SST-driven FiLM parameters are not presented as evidence of improved teleconnection skill."),
]

CN_ISSUES = [
    ("主编 1", "结构式摘要", "摘要已按 Study Region、Study Focus、New Hydrological Insights for the Region 三个具名部分重写，并将结论改为按指标区分的实证结果。"),
    ("主编 2", "Highlights", "另附四条 Highlights，每条均不超过含空格 85 字符；不再宣称 FiLM 或三种后处理方法全面优于 SEAS5。"),
    ("副主编", "局限、变量重要性和对照公平性", "重新构建原始 SEAS5 降水对照，训练期拟合大气及 SST 预处理，统一三种模型的主干宽度与三随机种子预算，补充直接对照和对称消融，并说明 2023—2024 年结果曾被反复查看。原先的 FiLM 优越性结论已撤回。"),
    ("审稿人 1 主要 1", "FiLM 局限", "讨论新增训练气候状态样本有限、六个月 SST 平均削弱时序信息、五层调制参数估计不稳定、插入深度不同以及缺少新年份未查看测试集等限制，并如实报告 SST 阴性对照。"),
    ("审稿人 1 主要 2", "极端值处理", "目标降水异常没有截尾或缩尾。训练使用有效陆地掩膜、梯度范数上限 2；极端 CSI 阈值取训练期第 90 百分位 0.9843，并讨论少雨区分母较小造成的异常放大。"),
    ("审稿人 1 主要 3", "ACC 幅度与其他指标", "表 1 同时报告 ACC、TCC、RMSE、符号一致率及极端 CSI；表 2 报告月度配对差、Wilcoxon p 值和两种 bootstrap 区间。FiLM 相对 SEAS5 的 ACC 仅高 0.0028，p=0.892，不作为稳健增益。"),
    ("审稿人 1 主要 4", "U-Net 与 FiLM 增量", "表 1—2 和图 1—4 直接展示 Baseline、EOF-cat、FiLM 与原始 SEAS5。探索期 ACC 分别为 0.333、0.293、0.316、0.314；没有将 FiLM 描述为相对基线的正增量。"),
    ("审稿人 1 主要 5", "十个大气变量", "图 6 给出同历月整幅变量场替换的重要性。tp 平均使 FiLM ACC 降低 0.263，其他变量各自绝对变化小于 0.005；相关变量之间可能互相替代，因此不据此判定变量无用。"),
    ("审稿人 1 主要 6", "区域差异", "图 4 将四种模型置于同一月份和同一色标；正文讨论地形、观测站分布和少雨区比例异常的可能作用，并明确其属于解释性假说。"),
    ("审稿人 1 次要 1", "缩写", "正文首次出现时解释 SST、EOF、PC、FiLM、ACC、TCC、RMSE、CSI、ONI；未使用的 BCC-CSM 和 PDO 表述已移除。"),
    ("审稿人 1 次要 2", "空格和完整句", "新稿统一重排，研究区域、时间和分辨率均改为完整句，并核对引文、标点与图注。"),
    ("审稿人 1 次要 3", "变量数量", "方法部分明确列出 tp、h200、h500、slp、t2m、t850、u200、u850、v200、v850 共十项。"),
    ("审稿人 1 次要 4", "颜色和图例", "重新绘制月度 ACC 和指标对照图，四系统采用可区分配色，图例不遮挡数据。"),
    ("审稿人 1 次要 5", "空间图标签", "图 4 直接标注月份、观测、SEAS5 和三种神经网络方法，各对应面板采用相同显示范围。"),
    ("审稿人 1 次要 6", "EOF-cat 定义", "首次出现时将 EOF-cat 定义为五个 EOF 主成分得分在输入通道拼接。"),
    ("审稿人 2 第 1 点", "短时段、ENSO 和另一训练起点", "承认 21 个月不足以支持年际泛化。改用 NOAA CPC ONI，而不是 SST PC1 代理分组。探索期为暖相 13、冷相 1、中性 7 个月；重新训练的 2020—2021 年留出期为暖相 3、冷相 13、中性 8 个月。两个时段均被历史探索查看；单个冷相月不计算 TCC。"),
    ("审稿人 2 第 2 点", "预处理泄露", "大气变量归一化及 SST 气候态、EOF 基底和主成分标准化仅在 1994—2017 年训练数据拟合；原始 SEAS5 降水使用物理量及 1994—2010 年固定月气候态。2020—2021 年试验重新在 1994—2015 年拟合。撤回全时段归一化所得旧数值。"),
    ("审稿人 2 第 3 点", "完整直接对照", "图 1—4、表 1—2 均同时展示 Baseline、EOF-cat、FiLM 和 SEAS5；报告 FiLM 对 EOF-cat 和 Baseline 的月度 ACC 配对差及空间 TCC 描述。"),
    ("审稿人 2 第 4 点", "EOF-cat 公平性与退化", "两种 SST 模型共享由六个月 SST 均值得出的五个得分，主干均为 32 通道，学习率、dropout、训练轮数、随机种子及验证 ACC 选模规则相同。但 EOF-cat 在输入层拼接，FiLM 调制五层，深度和新增参数仍不等同。EOF-cat 的 SST 置零 ACC 由 0.293 升至 0.319，不能简单归因于冗余表示。"),
    ("审稿人 2 第 5 点", "配对推断与 TCC 显著性", "表 2 给出月度配对 Wilcoxon 检验及逐月、按年分块 bootstrap 区间。由于探索期仅跨两年，按年分块区间仅作描述。图 3 的点标为逐格名义 p<0.05，不称整体空间显著；删除 FiLM 显著优于对照的表述。"),
    ("审稿人 2 第 6 点", "对称消融和阴性对照", "表 3、图 5 对 Baseline、EOF-cat、FiLM 同时进行完整输入、SST 得分置零和局地大气输入置零，并补充同历月 SST 得分替换。FiLM ACC 分别为 0.316、0.318 和替换后的 0.324，不将调制参数相关性解释成预报能力提升的证据。"),
]


def response(chinese=False):
    doc = Document()
    setup(doc, chinese)
    doc.styles["Normal"].font.size = Pt(9.6 if chinese else 10)
    doc.styles["Normal"].paragraph_format.space_after = Pt(1 if chinese else 3)
    doc.styles["Normal"].paragraph_format.line_spacing = 1 if chinese else 1.05
    doc.styles["Heading 2"].paragraph_format.space_before = Pt(4 if chinese else 7)
    doc.styles["Heading 2"].paragraph_format.space_after = Pt(1 if chinese else 2)
    if chinese:
        doc.sections[0].top_margin = Inches(0.6)
        doc.sections[0].bottom_margin = Inches(0.6)
    para(doc, "审稿意见逐项回复" if chinese else "Response to the Editor and Reviewers", "Title")
    para(doc, "Manuscript EJRH-D-26-03375")
    if chinese:
        para(doc, "感谢编辑和审稿人的意见。重算发现旧稿对原始 SEAS5 的定义及部分预处理存在问题，因此本轮撤回 FiLM 稳定领先的主张。2023—2024 年已被反复查看，不再称完全独立测试。正文和两份回复均采用同一重算数值。")
    else:
        para(doc, "We thank the editor and reviewers. Reanalysis identified an incorrect raw-SEAS5 comparator and preprocessing that required training-only refitting. We withdraw the claim of stable FiLM superiority. The 2023–2024 period was inspected repeatedly during development and is now explicitly exploratory. All revised claims use the same corrected results.")
    hold = ROLLING["ensemble"]["holdout"]
    if chinese:
        para(doc, "核心结果：2023—2024 年 SEAS5、Baseline、EOF-cat、FiLM 的 ACC 分别为 0.314、0.333、0.293、0.316。2020—2021 年重训留出期对应 ACC 为 " + ", ".join(f(hold[name]["mean_acc"]) for name in ("seas5", "baseline", "eof_cat", "film")) + "。两个时段均不支持三种方法全面优于 SEAS5。")
    else:
        para(doc, "Core result: exploratory 2023–2024 ACC is 0.314 (SEAS5), 0.333 (baseline), 0.293 (EOF-cat), and 0.316 (FiLM). The retrained 2020–2021 holdout gives " + ", ".join(f(hold[name]["mean_acc"]) for name in ("seas5", "baseline", "eof_cat", "film")) + " in the same order. These periods do not support universal postprocessor superiority.")
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
        "A second retrained origin limits claims of generalizable superiority.",
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


def main():
    DELIVERY.mkdir(parents=True, exist_ok=True)
    outputs = [manuscript(), response(False), response(True), highlights()]
    print("\n".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
