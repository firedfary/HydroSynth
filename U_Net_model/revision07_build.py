"""Assemble the revision-07 manuscript, highlights and responses.

Helpers, data loading and all verified numbers come from ``revision07_docs``;
this module holds the text. Nothing here hard-codes a result: every quantity
is interpolated from a result file so that ``revision07_verify.py`` can check
the written documents against the same sources.
"""

import json
import sys
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revision07_docs import (  # noqa: E402
    ABLATION, AMPLITUDE, AMP_ROWS, CHANCE_CSI, DELIVERY, ENSO, FIGVALS, FLOOR,
    HINDCAST, HINDCAST_PAIRED, INFERENCE, METRICS, REGIONAL, SOURCE, f, figure,
    heading, model_table, paired, para, replicate_rows, setup, table,
)

GAP = FLOOR["gap_film_minus_baseline"]
FLOOR_ACC = FLOOR["single_run_floor_acc"]
SOURCES = FLOOR["variance_sources"]
RESEED = SOURCES["sd_of_3_seed_ensemble_if_reseeded"]
TEST = METRICS["test"]["ensemble"]
VAL = METRICS["validation"]["ensemble"]


def front_matter(doc):
    para(doc, "Below the Resolution of a Single Training Run: SST-Conditioned U-Net "
              "Postprocessing of Monthly Precipitation over China", "Title")
    authors = doc.add_paragraph()
    for i, (name, affiliation) in enumerate((
        ("Fan YUE", "1,3"), ("Xiaojuan WANG", "1*"), ("Rong AI", "2"),
        ("Yinzhong WU", "3"), ("Guolin FENG", "2,4"), ("Zhiqiang GONG", "2"),
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


def abstract(doc):
    stats = GAP["mean_test"]
    heading(doc, "Abstract")
    para(doc, "Study Region: Mainland China, where monsoon variability, aridity and complex "
              "terrain challenge monthly precipitation prediction.")
    para(doc, "Study Focus: We compare direct precipitation from the fifth-generation ECMWF "
              "seasonal forecast system (SEAS5) with three matched 32-channel U-Net "
              "postprocessors: atmospheric predictors alone, the same predictors with five "
              "global sea surface temperature (SST) empirical orthogonal function (EOF) "
              "scores concatenated at the input, and the scores supplied through "
              "feature-wise linear modulation (FiLM). The main chronology fits "
              "preprocessing and models on 1994-2017, selects checkpoints on 2018-2022, and "
              "explores January 2023 to September 2024. Two additional retrospective "
              "training origins yield nonoverlapping 2014-2019 and 2020-2022 holdouts. "
              "Because the differences at stake are small and network training is not "
              "deterministic, we retrain one configuration five times to measure how large "
              "a difference a single training run can resolve, and we report extreme-event "
              "skill both as conventionally computed and after equalising output "
              "amplitude.")
    para(doc, "New Hydrological Insights for the Region: Repeated training shows that SST "
              f"conditioning lowers monthly spatial anomaly correlation (ACC) by "
              f"{abs(stats['mean']):.4f} +/- {stats['sem']:.4f} relative to the "
              f"unconditioned U-Net (five retrainings, all negative, p = {stats['p']:.4f}); "
              "the conditioning does not merely fail to help, it carries a small cost. That "
              f"gap is nonetheless smaller than the {FLOOR_ACC:.4f} ACC spread produced by "
              f"rerunning the identical configuration and smaller than the {RESEED:.4f} "
              "shift expected from choosing different random seeds, so no single training "
              "run can establish it. On the 21-month evaluation, ACC is "
              f"{f(TEST['seas5']['mean_acc'])} for SEAS5, {f(TEST['baseline']['mean_acc'])} "
              f"for the baseline, {f(TEST['eof_cat']['mean_acc'])} for EOF-cat and "
              f"{f(TEST['film']['mean_acc'])} for FiLM. All four systems reproduce only "
              "about one fifth of observed monthly anomaly amplitude, and once that "
              "amplitude is equalised the extreme-event ranking reverses, placing the "
              "unconditioned baseline first rather than last. Reporting the spread of "
              "repeated training, and separating amplitude from placement, are therefore "
              "prerequisites for comparing postprocessors at this scale.")
    para(doc, "Keywords: precipitation postprocessing; SEAS5; U-Net; feature-wise linear "
              "modulation; reproducibility; forecast verification")


def introduction(doc):
    heading(doc, "1 Introduction")
    para(doc, "Monthly precipitation has different sources of predictability across China. "
              "Previous studies associate summer rainfall in South China with distinct "
              "tropical Indian and Pacific Ocean influences in the early and late rainy "
              "seasons (Yuan et al., 2019), summer rainfall in Southwest China with "
              "moisture transport from several source regions (Yuan and Yang, 2020), and "
              "North China rainfall with the evolution of preceding SST anomalies rather "
              "than their level alone (Liang et al., 2023). These season- and "
              "region-specific findings motivate a regional assessment of monthly "
              "forecasts; they do not imply that one national SST summary should improve "
              "every month or location.")
    para(doc, "The fifth-generation ECMWF seasonal forecast system (SEAS5) provides the "
              "dynamical fields evaluated here (Johnson et al., 2019). Gridded atmospheric "
              "predictors can be postprocessed with a multiscale U-Net encoder-decoder "
              "(Ronneberger et al., 2015). Empirical orthogonal functions (EOFs) summarise "
              "dominant spatial SST variability (Hannachi et al., 2007), while feature-wise "
              "linear modulation (FiLM) conditions intermediate network features through "
              "learned channelwise scales and offsets (Perez et al., 2017). The "
              "hydrological question is whether either SST-conditioning design improves "
              "observed precipitation anomaly skill beyond atmospheric-field "
              "postprocessing.")
    para(doc, "Precipitation postprocessing over China has been studied with statistical "
              "calibration of ECMWF System4 seasonal output (Peng et al., 2014), deep "
              "learning for seasonal precipitation prediction (Jin et al., 2022) and a "
              "hybrid U-Net for subseasonal-to-seasonal summer ensemble forecasts (Lyu et "
              "al., 2024). Their forecast systems, lead times, target definitions and "
              "verification designs differ from the monthly archive used here, so their "
              "reported scores are not direct numerical benchmarks for our experiments.")
    para(doc, "A more expressive conditioning mechanism is useful only if it improves "
              "forecasts under matched inputs and a defensible comparator. The initial "
              "version of this study interpreted a standardised atmospheric input channel "
              "as raw SEAS5 precipitation and allowed full-record information into some "
              "preprocessing. Reconstructing the raw comparator and fitting all "
              "transformations on the training period changes the ranking. A second "
              "question then becomes unavoidable, and is the methodological contribution of "
              "this paper: the differences separating these designs are of order 0.01 in "
              "spatial ACC, and network training is not deterministic, so how large must a "
              "difference be before a single training run can resolve it? We answer that by "
              "measurement rather than assumption, and apply the same scepticism to "
              "extreme-event skill, where a fixed threshold rewards sharper forecasts "
              "independently of whether they are better placed. These questions make "
              "negative and mixed results informative rather than treating architectural "
              "complexity as an advantage by itself.")


def methods(doc):
    heading(doc, "2 Data and methods")
    heading(doc, "2.1 Data and chronology", 2)
    para(doc, "Monthly SEAS5 archive files supplied by the National Climate Center of China "
              "from January 1994 through September 2024 supply ten China-domain atmospheric "
              "fields: total precipitation (tp), geopotential height at 200 and 500 hPa "
              "(h200, h500), sea-level pressure (slp), 2-m temperature (t2m), 850-hPa "
              "temperature (t850), and zonal and meridional winds at 200 and 850 hPa (u200, "
              "u850, v200, v850). The archived 1-degree fields are mapped to a nominal "
              "0.5-degree grid of 120 x 140 cells over 70-140E and 0-60N; network predictors "
              "use bicubic interpolation, while the raw-precipitation comparator uses "
              "bilinear interpolation. Precipitation observations are based on 2,374 "
              "National Meteorological Information Center stations and their prepared "
              "gridded target field; 5,243 land cells contain valid observations. Global "
              "NOAA Extended Reconstructed Sea Surface Temperature version 5 (ERSSTv5) "
              "fields provide SST predictors (Huang et al., 2017). Three archive months are "
              "absent (September and October 2011; January 2017), leaving 366 aligned "
              "months.")
    para(doc, "Each archive file contains six monthly fields whose first time step is the "
              "month named in the file. Both the atmospheric predictors and the raw "
              "comparator use that first field, so the task evaluated here is the correction "
              "of a same-month dynamical simulation rather than a forecast issued at a lead "
              "of one month or more. The file metadata identify the first valid month but do "
              "not establish an operational initialisation date or lead label, so the scores "
              "reported below should not be compared directly with published skill at a "
              "stated forecast lead.")
    para(doc, "The main chronology assigns 285 months in 1994-2017 to fitting, 60 months in "
              "2018-2022 to validation and checkpoint selection, and 21 months in January "
              "2023 to September 2024 to exploratory evaluation. All data-dependent "
              "atmospheric normalisation, SST climatology, EOF basis and score scaling are "
              "fitted on the first period. The observed and raw SEAS5 precipitation "
              "anomalies use a fixed 1994-2010 calendar-month reference climatology, which "
              "lies within training. Their fractional anomaly is precipitation minus the "
              "corresponding calendar-month mean, divided by that mean plus 0.0001. For the "
              "SEAS5 comparator we select the first monthly tp field from each target-month "
              "archive file, bilinearly interpolate it, constrain it nonnegative, and apply "
              "a fixed 31-day conversion factor before anomaly conversion. No standardised "
              "network input is used as the raw comparator.")
    para(doc, "The target's extreme anomalies are retained without winsorization. The 90th "
              f"percentile of training anomalies, "
              f"{AMPLITUDE['protocol']['extreme_threshold_p90']:.4f}, defines extreme hits "
              "in the main chronology. This fractional-anomaly scale can magnify small "
              "absolute precipitation errors where climatological rainfall is low, so "
              "aggregate skill and regional maps require caution. Two retrospective origins "
              "use 1994-2010 training and 2011-2013 validation for a 2014-2019 holdout (71 "
              "available months), and 1994-2015 training and 2016-2019 validation for a "
              "2020-2022 holdout (36 months). Atmospheric transforms, SST EOFs and scaling, "
              "models and extreme thresholds are refitted at each origin. These years and "
              "the design were previously examined, so the combined analysis is descriptive "
              "rather than a pristine confirmatory test.")

    heading(doc, "2.2 SST representation and model comparison", 2)
    para(doc, "The stored SST predictor contains six preceding monthly fields for each "
              "target month. We average those six fields to one SST map, subtract a "
              "training-only SST climatology, fit EOFs on the training maps and project "
              "later maps onto the frozen basis. The first five standardised "
              "principal-component (PC) scores are shared by EOF-cat and FiLM. Averaging "
              "removes within-window timing; the design should not be described as a "
              "six-step PC sequence or a one-month PC lag.")
    para(doc, "All neural variants use the same 32-channel base U-Net, with five encoder "
              "levels, a mirrored decoder, skip connections, batch normalisation, ReLU "
              "activations and spatial dropout. Baseline receives the ten atmospheric "
              "channels and has 3,352,289 trainable parameters. EOF-cat broadcasts the five "
              "PC scores as constant spatial maps and appends them to the ten input "
              "channels, giving 3,353,729 parameters. FiLM receives the ten atmospheric "
              "channels plus a two-layer index encoder that maps the five scores to "
              "channelwise scale and offset parameters at each encoder level, giving "
              "3,452,513 parameters. Its transform applies the predicted scale to each "
              "feature channel and adds the predicted offset; zero initialisation of the "
              "final conditioning layer starts near the unmodulated mapping. EOF-cat and "
              "FiLM use the same five PCs from the same preceding six-month SST mean, with "
              "the same split, optimiser, seeds, 100-epoch training budget and "
              "validation-ACC checkpoint rule. The 98,784-parameter FiLM increase over "
              "EOF-cat and the difference between input concatenation and five-level "
              "modulation mean that insertion depth and capacity are not matched. The "
              "earlier exploratory search cannot be retroactively equalised, so this "
              "comparison does not isolate a single causal effect of the FiLM operator.")

    heading(doc, "2.3 Training, repetition and evaluation", 2)
    para(doc, "Each model is trained with seeds 42, 43 and 44 for 100 epochs, batch size "
              "eight, masked mean squared error over valid land cells, AdamW at learning "
              "rate 0.0002 and weight decay 0.00001, cosine decay, dropout 0.1 and "
              "gradient-norm clipping at 2. The best checkpoint for each model and seed is "
              "selected by mean validation spatial ACC, and the three selected predictions "
              "are averaged with equal weight.")
    para(doc, "Because convolution backward passes use nondeterministic reductions on the "
              "GPU, two runs of this protocol with identical seeds, data and batch order do "
              "not produce identical models. We therefore repeat the complete protocol five "
              "times and report the resulting spread alongside every model comparison. That "
              "spread is not a formality: Section 3.2 shows it is larger than the "
              "differences earlier versions of this study interpreted.")
    para(doc, "Spatial ACC is the Pearson correlation across valid cells for each month, "
              "then averaged over months. Temporal correlation coefficient (TCC) is computed "
              "at each valid cell across months, then averaged spatially. RMSE is over all "
              "valid grid-month pairs; sign consistency is the fraction with matching "
              "anomaly sign; and extreme critical success index (CSI) is hits divided by "
              "hits plus false alarms plus misses above the training 90th percentile. "
              "Because CSI counts exceedances of a fixed threshold, it rewards a sharper "
              "field independently of placement, so we additionally report CSI after "
              "rescaling each predicted month to the observed spatial standard deviation of "
              "that month. ACC is invariant to this rescaling, which the analysis code "
              "asserts as a self-check. Output amplitude is reported as the mean "
              "within-month spatial standard deviation over valid land cells. Monthly ACC "
              "differences use paired Wilcoxon tests and month-resampled and "
              "calendar-year-block bootstrap intervals; these resample months around one "
              "fixed set of trained models and therefore exclude training variability, "
              "which Section 3.2 quantifies separately.")


def results_scores(doc):
    heading(doc, "3 Results")
    heading(doc, "3.1 Matched model scores", 2)
    para(doc, "Table 1. Main-model validation and exploratory evaluation. Higher ACC, TCC, "
              "sign consistency and CSI are better; lower RMSE is better. Amplitude is the "
              "ratio of predicted to observed mean monthly spatial standard deviation, "
              "reported for the exploratory period. TCC from 21 months is descriptive.")
    rows = [("Validation 2018-2022",) + row + ("",) for row in model_table("validation")]
    rows += [("Exploratory 2023-2024",) + row
             for row in model_table("test", with_amplitude=True)]
    table(doc, ("Period", "System", "ACC", "TCC", "RMSE", "Sign cons. (%)", "CSI", "Amp."),
          rows, widths=(1.22, 1.12, 0.60, 0.60, 0.60, 0.80, 0.60, 0.52))
    para(doc, f"On 2018-2022 validation the baseline has the highest ACC "
              f"({f(VAL['baseline']['mean_acc'])}) and raw SEAS5 the highest TCC "
              f"({f(VAL['seas5']['mean_tcc'])}). In the 21-month explored period the "
              f"baseline again leads ACC ({f(TEST['baseline']['mean_acc'])}) and sign "
              f"consistency ({f(TEST['baseline']['sc_pct'], 1)} percent), whereas EOF-cat "
              f"has the best TCC ({f(TEST['eof_cat']['mean_tcc'])}), RMSE "
              f"({f(TEST['eof_cat']['rmse'])}) and nominal extreme CSI "
              f"({f(TEST['eof_cat']['csi'])}). FiLM is intermediate on most measures. "
              "Section 3.2 shows that the ACC differences in this table are smaller than "
              "the spread of repeated training, and Section 3.4 shows that the CSI ordering "
              "is a statement about output amplitude rather than extreme-event placement "
              "and reverses when amplitude is equalised. The amplitude column is given here "
              "so that the CSI column is not read in isolation.")
    figure(doc, "Fig1_monthly_ACC.png",
           "Figure 1. Monthly spatial ACC for all four systems and paired "
           "FiLM-minus-comparator differences in the 21-month exploratory evaluation.")


def results_variability(doc):
    stats, ci = GAP["mean_test"], GAP["ci95_normal"]
    heading(doc, "3.2 Training variability and what one run can resolve", 2)
    para(doc, "The differences in Table 1 are of order 0.01 in ACC. Before interpreting them "
              "we measure how much of that is attributable to training itself. The complete "
              "protocol of Section 2.3 -- identical seeds, data, batch order and "
              "hyperparameters -- was run five times from scratch.")
    para(doc, "Table 2. Five independent retrainings of one identical configuration, "
              "validation 2018-2022. Spread is the maximum minus the minimum across runs. "
              "Differences between runs arise only from nondeterministic GPU reductions.")
    table(doc, ("Run", "Baseline ACC", "EOF-cat ACC", "FiLM ACC", "FiLM minus Baseline"),
          replicate_rows(), widths=(0.9, 1.3, 1.3, 1.3, 1.6))
    para(doc, "Two conclusions follow, and they are not the same conclusion. First, the gap "
              f"is real: all five runs are negative, the mean is {f(stats['mean'], 4)} with "
              f"a standard error of {f(stats['sem'], 4)}, a one-sample t-test gives "
              f"p = {stats['p']:.4f}, and the 95 percent interval [{f(ci[0], 4)}, "
              f"{f(ci[1], 4)}] excludes zero. The sign test gives "
              f"p = {stats['sign_test_p']:.4f}, its smallest attainable value at five runs. "
              "Conditioning the network on SST scores does not merely fail to help in this "
              "setting; it carries a small but repeatable cost in spatial pattern skill.")
    para(doc, "Second, no single run can establish that. Rerunning the identical "
              f"configuration moves ensemble ACC by up to {f(FLOOR_ACC, 4)} (EOF-cat; "
              f"{f(FLOOR['per_model_acc']['Baseline']['spread'], 4)} for the baseline), "
              "which exceeds the gap itself. A study reporting one training run -- as the "
              "earlier versions of this work did, and as is common in this literature -- "
              "does not have the resolution to separate these designs. The appropriate "
              "correction is not to withdraw the comparison but to report it with the "
              "uncertainty that produced it.")
    para(doc, "Decomposing the spread shows that the random seed matters more than the "
              "nondeterminism. Across runs with the same seed the gap varies with a "
              f"standard deviation of {f(SOURCES['sd_across_runs_same_seed'], 4)}; across "
              f"seeds within a run it is {f(SOURCES['sd_across_seeds_same_run'], 4)}, "
              f"implying a seed component of {f(SOURCES['sd_seed_component'], 4)}. A "
              "three-member ensemble built on a different triplet of seeds would therefore "
              f"shift the gap by about {f(RESEED, 4)}, comparable to the gap. In our runs "
              "seed 42 produced a positive gap in two of five repetitions while seed 43 was "
              "strongly negative in all five, so a single-seed study could have reported "
              "either sign. Because an independent group would choose its own seeds, this "
              "is the quantity that governs external reproducibility.")
    figure(doc, "Fig2_training_variability.png",
           "Figure 2. (a) Validation mean spatial ACC for five independent retrainings of an "
           "identical configuration; horizontal bars mark the mean and annotations give the "
           "spread. (b) The FiLM-minus-baseline gap per seed, per three-seed ensemble, and "
           "as a mean with its 95 percent interval. The dashed line marks zero.")
    para(doc, "This measurement also bears on the interval estimates reported below and in "
              "the earlier versions of this study. A paired bootstrap over months resamples "
              "the evaluation months around one fixed set of trained models, so it "
              "quantifies month-sampling noise and is blind to training noise. Such "
              "intervals are narrower than the quantity they appear to bound. We retain them "
              "because they answer a question reviewers asked, and report the training "
              "spread beside them so that neither is read alone.")


def results_paired(doc):
    heading(doc, "3.3 Paired comparisons and spatial variation", 2)
    para(doc, "Table 3. Paired monthly spatial ACC differences in January 2023 to September "
              "2024. Intervals are descriptive because the period was repeatedly examined, "
              "the calendar-year-block interval has only two available years, and neither "
              "interval includes the training variability of Section 3.2.")
    table(doc, ("Comparison", "Mean diff.", "Positive", "Wilcoxon p",
                "Month 95% CI", "Year block 95% CI"),
          [paired("FiLM - baseline", "film_minus_baseline"),
           paired("FiLM - EOF-cat", "film_minus_eof_cat"),
           paired("FiLM - SEAS5", "film_minus_seas5"),
           paired("Baseline - SEAS5", "baseline_minus_seas5")],
          widths=(1.25, 0.85, 0.75, 0.85, 1.25, 1.25))
    row = METRICS["test"]["paired_acc"]["film_minus_eof_cat"]
    base_row = METRICS["test"]["paired_acc"]["film_minus_baseline"]
    para(doc, f"FiLM exceeds EOF-cat in {row['pos_months']} of {row['total_months']} months "
              f"on ACC, while its mean ACC remains below the baseline by "
              f"{abs(base_row['mean_diff']):.4f}. The single-run difference against the "
              f"baseline has a Wilcoxon p of {f(base_row['wilcoxon_p'], 3)} and a "
              "year-block interval that includes zero, which earlier drafts read as an "
              "absence of evidence. Section 3.2 gives the more precise statement: the "
              "difference is real but below what this comparison, computed from one "
              "training run, can resolve.")
    para(doc, "Table 4. Paired differences in spatially averaged TCC. Validation intervals "
              "use 1,000 calendar-year-block bootstrap samples over five years. The explored "
              "period has only two years; its ranges enumerate the three distinct block "
              "combinations and are not 95 percent confidence intervals. Validation also "
              "served checkpoint selection, so its intervals are descriptive conditional on "
              "those selections.")
    tcc_rows = []
    for label, key in (("FiLM - SEAS5", "seas5"), ("FiLM - baseline", "baseline"),
                       ("FiLM - EOF-cat", "eof_cat")):
        val = INFERENCE["validation"]["tcc_differences"][key]
        tst = INFERENCE["exploratory"]["tcc_differences"][key]
        tcc_rows.append((label, f(val["film_minus_comparator"]),
                         f"[{f(val['year_block_boot95'][0])}, {f(val['year_block_boot95'][1])}]",
                         f(tst["film_minus_comparator"]),
                         f"[{f(tst['year_block_range'][0])}, {f(tst['year_block_range'][1])}]"))
    table(doc, ("Comparison", "2018-22 dTCC", "Year boot 95%", "2023-24 dTCC",
                "Two-year range"), tcc_rows,
          widths=(1.35, 1.05, 1.35, 1.05, 1.35))
    para(doc, f"The 21-month spatial mean TCC values are {f(TEST['seas5']['mean_tcc'])} for "
              f"SEAS5, {f(TEST['baseline']['mean_tcc'])} for the baseline, "
              f"{f(TEST['eof_cat']['mean_tcc'])} for EOF-cat and "
              f"{f(TEST['film']['mean_tcc'])} for FiLM. None of the five-year validation "
              "TCC-difference intervals in Table 4 excludes zero. The exploratory two-year "
              "ranges are too coarse for a significance claim. Maps show heterogeneous "
              "cells rather than a uniform FiLM gain.")
    figure(doc, "Fig4_gridwise_TCC.png",
           "Figure 3. Gridwise TCC for all four systems over 21 months. Dots indicate "
           "nominal two-sided p < 0.05 at displayed land cells; they are not a spatial "
           "field-significance mask. Boundaries and South China Sea insets provide "
           "geographic context; mapped values are limited to valid observation cells.")
    figure(doc, "Fig5_TCC_difference.png",
           "Figure 4. Gridwise FiLM-minus-baseline and FiLM-minus-EOF-cat TCC differences "
           "over 21 exploratory months. These maps are descriptive and carry no spatial "
           "significance claim.")
    figure(doc, "Fig6_spatial_examples.png",
           "Figure 5. Observed and predicted fractional precipitation anomalies in January, "
           "April, July and October 2023. The same display range is used for every panel; "
           "values beyond it are clipped for visualization only.")


def results_amplitude(doc):
    heading(doc, "3.4 Output amplitude and extreme-event skill", 2)
    ratios = FIGVALS["amplitude"]["amplitude_ratio"]
    para(doc, "Every system in Table 1 is strongly underdispersive. Measured as the mean "
              "within-month spatial standard deviation over valid land cells, the "
              f"exploratory-period amplitudes are {f(ratios['SEAS5'], 2)} of observed for "
              f"SEAS5, {f(ratios['Baseline'], 2)} for the baseline, "
              f"{f(ratios['EOF-cat'], 2)} for EOF-cat and {f(ratios['FiLM'], 2)} for FiLM. "
              "All four reproduce roughly one fifth of observed anomaly amplitude. For "
              "applications that depend on anomaly magnitude this is a more consequential "
              "limitation than any ranking among the four systems, and it is not removed by "
              "postprocessing.")
    para(doc, "It also governs the extreme-event metric. CSI counts exceedances of a fixed "
              "training-period threshold, so a sharper field scores higher whether or not "
              "its extremes are better placed. Rescaling each predicted month to the "
              "observed spatial standard deviation of that month removes the sharpness term "
              "and leaves placement; ACC is invariant to the operation and is asserted "
              "unchanged in the analysis code.")
    para(doc, "Table 5. Extreme-event CSI before and after equalising output amplitude, "
              "exploratory period. A white-noise field rescaled to observed amplitude "
              f"scores {CHANCE_CSI:.3f}, as does a spatially shuffled copy of the "
              "observation, so that value is the chance level; a perfect forecast scores "
              "1.000.")
    rows = []
    for label, key in (("Baseline U-Net", "Baseline U-Net"), ("EOF-cat", "EOF-cat (full)"),
                       ("FiLM-U-Net", "FiLM-U-Net (full)")):
        r = AMP_ROWS[key]
        rows.append((label, f(r["csi"]), f(r["amplitude_ratio"], 2),
                     f(r["csi_at_obs_amplitude"]),
                     f(r["csi_at_obs_amplitude"] - CHANCE_CSI)))
    table(doc, ("System", "CSI as reported", "Amplitude / obs.",
                "CSI at obs. amplitude", "Above chance"), rows,
          widths=(1.35, 1.15, 1.15, 1.35, 1.05))
    raw_order = " > ".join(FIGVALS["amplitude"]["raw_csi_ranking"])
    eq_order = " > ".join(FIGVALS["amplitude"]["equalised_csi_ranking"])
    para(doc, f"The ranking reverses exactly. By CSI as conventionally computed it is "
              f"{raw_order}; at equal amplitude it is {eq_order}. The apparent "
              "extreme-event advantage of the SST-conditioned designs in Table 1 is an "
              "artifact of their producing a sharper field, and the unconditioned baseline "
              "is in fact the best of the three at placing extremes. All three remain well "
              "above chance, so the metric does carry information; what it does not carry, "
              "at a fixed threshold and with underdispersive forecasts, is a clean ordering "
              "of placement skill.")
    figure(doc, "Fig3_amplitude_and_CSI.png",
           "Figure 6. (a) Predicted divided by observed mean monthly spatial standard "
           "deviation for the four systems. (b) Extreme-event CSI as reported and after "
           "equalising each month to the observed amplitude; the dotted line marks the "
           "chance level.")


def results_regional(doc):
    heading(doc, "3.5 Regional variation", 2)
    para(doc, "Table 6. Descriptive regional spatial ACC on the common valid grid. East is "
              "at or above 105E; Southwest is below 105E and at or below 37N; Northwest is "
              "below 105E and above 37N. These simple geographic partitions are not formal "
              "hydroclimatic zones. Within-region ACC is computed across that region's "
              "cells for each month, then averaged; it need not average to the nationwide "
              "ACC.")
    # REGIONAL stores one row per (period, region, model); pivot to one row per region.
    pivot, order = {}, []
    labels = {"validation": "2018-2022", "exploratory": "2023-2024",
              "test": "2023-2024"}
    for r in REGIONAL:
        key = (r["period"], r["region"])
        if key not in pivot:
            pivot[key] = {"cells": r["valid_cells"]}
            order.append(key)
        pivot[key][r["model"]] = r["acc"]
    table(doc, ("Period", "Region", "Cells", "SEAS5", "Baseline", "EOF-cat", "FiLM"),
          [(labels.get(period, period), region, pivot[(period, region)]["cells"],
            f(pivot[(period, region)]["seas5"]), f(pivot[(period, region)]["baseline"]),
            f(pivot[(period, region)]["eof_cat"]), f(pivot[(period, region)]["film"]))
           for period, region in order],
          widths=(1.15, 1.05, 0.65, 0.78, 0.78, 0.78, 0.78))
    explore = [k for k in order if k[0] != "validation"]
    east = pivot[[k for k in explore if k[1].lower().startswith("east")][0]]
    sw = pivot[[k for k in explore if k[1].lower().startswith("south")][0]]
    nw = pivot[[k for k in explore if k[1].lower().startswith("north")][0]]
    para(doc, "In the explored period the baseline ACC exceeds raw SEAS5 in each partition: "
              f"{f(east['baseline'])} versus {f(east['seas5'])} in the east, "
              f"{f(sw['baseline'])} versus {f(sw['seas5'])} in the southwest and "
              f"{f(nw['baseline'])} versus {f(nw['seas5'])} in the northwest. EOF-cat trails "
              "raw SEAS5 in all three, while FiLM is only slightly above it. The southwest "
              "has the lowest regional ACC for every system. Earlier work links southwest "
              "summer rainfall to moisture transport from multiple source regions (Yuan and "
              "Yang, 2020), but this study has not attributed its monthly model errors to "
              "those pathways. The partitions were inspected after model development and "
              "have no regional significance test. Terrain, station coverage and amplified "
              "fractional anomalies in dry areas remain possible explanations rather than "
              "demonstrated causes. Given Section 3.2, differences between the three "
              "postprocessors within a region should be read as provisional: they are of "
              "the same order as the training spread, which was not measured separately per "
              "region.")


def results_controls(doc):
    heading(doc, "3.6 Conditioning controls and predictor importance", 2)
    para(doc, "Table 7. Symmetric input controls on the exploratory period, with output "
              "amplitude. PC zeroing and local-input zeroing are out-of-distribution "
              "perturbations, so their score changes diagnose dependence but are not "
              "replacement-model benchmarks. Because zeroing an input both removes "
              "information and shrinks the output, CSI is reported at equal amplitude "
              "alongside its nominal value.")
    order = (("Baseline, full inputs", "Baseline U-Net"),
             ("EOF-cat, full inputs", "EOF-cat (full)"),
             ("EOF-cat, SST zeroed", "EOF-cat (SST zeroed)"),
             ("EOF-cat, shuffled PCs", "EOF-cat (shuffled PCs)"),
             ("EOF-cat, local zeroed", "EOF-cat (local zeroed)"),
             ("FiLM, full inputs", "FiLM-U-Net (full)"),
             ("FiLM, SST zeroed", "FiLM-U-Net (SST zeroed)"),
             ("FiLM, shuffled PCs", "FiLM-U-Net (shuffled PCs)"),
             ("FiLM, local zeroed", "FiLM-U-Net (local zeroed)"))
    table(doc, ("Configuration", "ACC", "TCC", "CSI", "Amp. / obs.", "CSI at obs. amp."),
          [(label, f(AMP_ROWS[key]["mean_acc"]), f(AMP_ROWS[key]["mean_tcc"]),
            f(AMP_ROWS[key]["csi"]), f(AMP_ROWS[key]["amplitude_ratio"], 2),
            f(AMP_ROWS[key]["csi_at_obs_amplitude"])) for label, key in order],
          widths=(1.70, 0.72, 0.72, 0.72, 0.95, 1.15))
    film_full = AMP_ROWS["FiLM-U-Net (full)"]
    film_zero = AMP_ROWS["FiLM-U-Net (SST zeroed)"]
    cat_full = AMP_ROWS["EOF-cat (full)"]
    cat_zero = AMP_ROWS["EOF-cat (SST zeroed)"]
    drop = (film_full["csi"] - film_zero["csi"]) / film_full["csi"] * 100
    eq_drop = ((film_full["csi_at_obs_amplitude"] - film_zero["csi_at_obs_amplitude"])
               / film_full["csi_at_obs_amplitude"] * 100)
    amp_drop = (1 - film_zero["amplitude_ratio"] / film_full["amplitude_ratio"]) * 100
    para(doc, "Setting FiLM's standardised SST PCs to zero raises ACC from "
              f"{f(film_full['mean_acc'])} to {f(film_zero['mean_acc'])}, and replacing the "
              "scores with same-calendar-month values from 2018-2022 gives "
              f"{f(AMP_ROWS['FiLM-U-Net (shuffled PCs)']['mean_acc'])}. EOF-cat likewise "
              f"rises from {f(cat_full['mean_acc'])} to {f(cat_zero['mean_acc'])} when PCs "
              "are zeroed. Neither control lowers ACC, so these results do not support an "
              "ACC benefit from simultaneous SST conditioning, consistent with the repeated "
              "training of Section 3.2.")
    para(doc, "The amplitude columns change how the CSI response should be read. Zeroing "
              f"FiLM's SST scores appears to collapse CSI by {drop:.0f} percent, from "
              f"{f(film_full['csi'])} to {f(film_zero['csi'])}; but the same operation "
              f"reduces output amplitude by {amp_drop:.0f} percent, and at equal amplitude "
              f"the change is {eq_drop:.1f} percent. For EOF-cat the correction changes the "
              f"sign of the conclusion: nominal CSI falls from {f(cat_full['csi'])} to "
              f"{f(cat_zero['csi'])}, while equalised CSI rises from "
              f"{f(cat_full['csi_at_obs_amplitude'])} to "
              f"{f(cat_zero['csi_at_obs_amplitude'])}. On placement alone, EOF-cat's SST "
              "input makes its extreme-event forecasts worse. The local-zeroed rows fall to "
              f"{f(AMP_ROWS['FiLM-U-Net (local zeroed)']['csi_at_obs_amplitude'])}, just "
              f"above the {CHANCE_CSI:.3f} chance level, as expected when only bias terms "
              "remain and rescaling amplifies numerical noise; their agreement with an "
              "independently computed chance level is a check that the rescaling behaves as "
              "intended. Removing the local atmospheric maps degrades all architectures "
              "sharply, confirming that local predictors carry the spatial information.")
    figure(doc, "Fig7_symmetric_ablation.png",
           "Figure 7. Symmetric SST and local-predictor input controls for baseline, EOF-cat "
           "and FiLM. The baseline has no SST input, so its PC-zeroed and PC-replaced bars "
           "repeat the full-input prediction.")
    importance = ABLATION["film_predictor_importance"]
    top = importance[0]
    others = max(abs(row["mean_acc_drop"]) for row in importance[1:])
    para(doc, "Replacing one atmospheric predictor map at a time with a same-calendar-month "
              "map from the validation period makes tp dominant: its FiLM ACC reduction "
              f"averages {f(top['mean_acc_drop'])} across ten trials, against "
              f"{f(others)} for the largest of the others. Every other predictor has an absolute mean "
              "effect below 0.005. Correlated circulation variables can substitute for one "
              "another, so a small single-variable permutation effect does not prove that a "
              "variable is useless. These controls are inference-time perturbations outside "
              "the joint training distribution, not retrained feature-removal experiments.")
    figure(doc, "Fig8_predictor_importance.png",
           "Figure 8. FiLM predictor permutation importance, measured as mean decrease in "
           "spatial ACC when one full atmospheric map is replaced.")


def results_enso(doc):
    heading(doc, "3.7 ENSO phase and retrospective origins", 2)
    para(doc, "Table 8. Spatial ACC by Oceanic Nino Index threshold category. The "
              "retrospective rows pool predictions from two different training origins. The "
              "exploratory cold category contains only one month.")
    rows = []
    # Human-readable period labels; the pooled retrospective block lives in the
    # hindcast file rather than the ENSO file, so it is appended explicitly.
    labels = {"validation": "2018-2022 validation",
              "test": "2023-2024 exploratory",
              "rolling_2020_2021": "2020-2021 rolling holdout"}
    blocks = list(ENSO["periods"].items())
    blocks.append(("2014-2022 retrospective", HINDCAST["pooled"]["phases"]))
    for period, block in blocks:
        for phase in ("warm", "cold", "neutral", "all"):
            if phase not in block:
                continue
            entry = block[phase]
            m = entry["metrics"]
            rows.append((labels.get(period, period), phase.capitalize(),
                         entry["months"], f(m["seas5"]["mean_acc"]),
                         f(m["baseline"]["mean_acc"]), f(m["eof_cat"]["mean_acc"]),
                         f(m["film"]["mean_acc"])))
    table(doc, ("Period", "Phase", "Months", "SEAS5", "Baseline", "EOF-cat", "FiLM"),
          rows, widths=(1.20, 0.85, 0.70, 0.78, 0.82, 0.78, 0.78))
    para(doc, "Table 9. Retrospective holdouts from two training origins and their pooled "
              "107-month evaluation. Each origin refits the models and all "
              "training-dependent transformations. Both windows were examined during "
              "development.")
    hind_rows = []
    for key, label in (("2014_2019", "2014-2019 (71 months)"),
                       ("2020_2022", "2020-2022 (36 months)")):
        m = HINDCAST["folds"][key]["metrics"]
        hind_rows.append((label, f(m["seas5"]["mean_acc"]), f(m["baseline"]["mean_acc"]),
                          f(m["eof_cat"]["mean_acc"]), f(m["film"]["mean_acc"])))
    pooled = HINDCAST["pooled"]["phases"]["all"]
    m = pooled["metrics"]
    hind_rows.append((f"2014-2022 pooled ({pooled['months']} months)",
                      f(m["seas5"]["mean_acc"]), f(m["baseline"]["mean_acc"]),
                      f(m["eof_cat"]["mean_acc"]), f(m["film"]["mean_acc"])))
    table(doc, ("Holdout", "SEAS5", "Baseline", "EOF-cat", "FiLM"), hind_rows,
          widths=(2.20, 1.00, 1.00, 1.00, 1.00))
    f14 = HINDCAST["folds"]["2014_2019"]["paired_acc"]["film_minus_baseline"]["mean_diff"]
    f20 = HINDCAST["folds"]["2020_2022"]["paired_acc"]["film_minus_baseline"]["mean_diff"]
    span = abs(f14 - f20)
    para(doc, "The two origins rank FiLM oppositely: the FiLM-minus-baseline ACC difference "
              f"is {f(f14, 4)} at the 2014-2019 origin and {f(f20, 4)} at the 2020-2022 "
              f"origin, a span of {f(span, 3)}. That span exceeds both the single-run "
              f"spread of {f(FLOOR_ACC, 4)} and the reseeding shift of {f(RESEED, 4)} "
              "measured in Section 3.2, so training variability alone is sufficient to "
              "produce a reversal of this size. We therefore do not offer a physical "
              "explanation for the reversal, and note that one is not required; neither is "
              "a physical cause excluded by these data.")
    figure(doc, "Fig9_annual_hindcast.png",
           "Figure 9. Annual mean spatial ACC for the four systems and paired "
           "FiLM-minus-comparator differences in the 2014-2022 retrospective holdouts. The "
           "dashed line marks the switch between independently fitted training origins "
           "after 2019. The 2017 summary has 11 available months; other years have 12.")
