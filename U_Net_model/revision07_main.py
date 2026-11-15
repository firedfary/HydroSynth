"""Entry point: write the revision-07 manuscript, highlights and responses."""

import sys
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revision07_docs import (  # noqa: E402
    AMP_ROWS, CHANCE_CSI, DELIVERY, FIGVALS, FLOOR, METRICS, f, figure,
    heading, para, setup,
)
import revision07_build as body  # noqa: E402
import revision07_sections as tail  # noqa: E402

GAP = FLOOR["gap_film_minus_baseline"]
FLOOR_ACC = FLOOR["single_run_floor_acc"]
RESEED = FLOOR["variance_sources"]["sd_of_3_seed_ensemble_if_reseeded"]
TEST = METRICS["test"]["ensemble"]

HIGHLIGHTS = (
    "SST conditioning lowers monthly spatial ACC by 0.008; five retrainings confirm it.",
    "One training run cannot resolve that gap: identical reruns shift ACC by 0.009.",
    "Reseeding a three-member ensemble moves the gap by as much as the gap itself.",
    "All four systems predict only about one fifth of observed anomaly amplitude.",
    "Equalising output amplitude reverses the extreme-event CSI ranking entirely.",
)


def manuscript():
    doc = Document()
    setup(doc)
    doc.styles["Normal"].paragraph_format.line_spacing = 2
    numbering = OxmlElement("w:lnNumType")
    numbering.set(qn("w:countBy"), "1")
    numbering.set(qn("w:start"), "1")
    numbering.set(qn("w:restart"), "continuous")
    doc.sections[0]._sectPr.append(numbering)

    body.front_matter(doc)
    body.abstract(doc)
    body.introduction(doc)
    body.methods(doc)
    body.results_scores(doc)
    body.results_variability(doc)
    body.results_paired(doc)
    body.results_amplitude(doc)
    body.results_regional(doc)
    body.results_controls(doc)
    body.results_enso(doc)
    tail.pathway(doc)
    tail.discussion(doc)
    tail.conclusions(doc)
    tail.declarations(doc)
    tail.references(doc)

    path = DELIVERY / "07_可信度返修稿.docx"
    doc.save(path)
    return path


def highlights():
    doc = Document()
    setup(doc)
    heading(doc, "Highlights")
    for line in HIGHLIGHTS:
        if len(line) > 85:
            raise ValueError(f"highlight exceeds 85 characters ({len(line)}): {line}")
        para(doc, line)
    path = DELIVERY / "Highlights.docx"
    doc.save(path)
    return path


def supplement():
    doc = Document()
    setup(doc)
    para(doc, "Supplementary Material S1", "Title")
    para(doc, "Below the Resolution of a Single Training Run: SST-Conditioned U-Net "
              "Postprocessing of Monthly Precipitation over China")
    heading(doc, "S1 Sensitivity of the raw SEAS5 comparator")
    figure(doc, "FigS1_SEAS5_sensitivity.png",
           "Supplementary Figure S1. Sensitivity of exploratory-period spatial anomaly "
           "correlation to construction of the SEAS5 precipitation comparator. The primary "
           f"comparator uses the first monthly total-precipitation field in each "
           f"target-month archive, bilinear remapping and a nonnegative constraint (ACC "
           f"{f(TEST['seas5']['mean_acc'])}). Nonnegative bicubic remapping gives ACC 0.288. "
           "The earlier value of 0.281 was calculated from a standardized neural-network "
           "input channel and is not a valid raw-SEAS5 skill estimate.", width=6.2)
    para(doc, "The comparator construction materially affects the reported skill of the "
              "dynamical benchmark, and therefore every statement of the form "
              "'postprocessing improves on SEAS5'. Bicubic interpolation of a precipitation "
              "field overshoots near sharp gradients and produces small negative values "
              "which, divided by a small climatological denominator, generate large "
              "spurious fractional anomalies. Bilinear remapping with a nonnegative "
              "constraint avoids this. We report the primary construction in the main text "
              "and this sensitivity here so that the choice is explicit rather than "
              "implicit.")
    heading(doc, "S2 Reproducibility of the repeated-training experiment")
    para(doc, "Section 3.2 of the main text rests on five independent retrainings of one "
              "configuration. Reproducing it requires no new experimental design: the "
              "published training script is run five times with unchanged arguments and the "
              "resulting ensemble scores compared. On a single consumer GPU each run took "
              "approximately twelve minutes. We report this because the measurement is "
              "cheap relative to its consequences for interpretation, and because its "
              "absence from the earlier versions of this study is what allowed two "
              "contradictory rankings to be reported in successive drafts.")
    path = DELIVERY / "07_Supplement_S1.docx"
    doc.save(path)
    return path


RESPONSE_EN = (
    ("Summary of this revision",
     ("We thank the editor and both reviewers. In addressing the comments we identified two "
      "errors in the original analysis, and their correction changes the paper's "
      "conclusion. We state this at the outset because it affects how the remaining "
      "responses should be read.",
      "First, reviewer 2's comment (2) was correct. The original 'validation' period served "
      "both model selection and independent testing, and the SEAS5 comparator was computed "
      "from a standardized network input channel rather than raw model precipitation. With "
      "training-period-only preprocessing and a reconstructed raw comparator, the ranking "
      "changes: FiLM conditioning does not outperform the unconditioned U-Net.",
      "Second, in verifying that reversal we found that it, and the original claim, were "
      "both reported from a single training run. Retraining one configuration five times "
      "gives a FiLM-minus-baseline ACC difference of "
      f"{GAP['mean_test']['mean']:+.4f} +/- {GAP['mean_test']['sem']:.4f} "
      f"(p = {GAP['mean_test']['p']:.4f}, all five negative), while rerunning the identical "
      f"configuration moves ensemble ACC by up to {FLOOR_ACC:.4f}. The effect is real and "
      "below single-run resolution. The manuscript now reports repeated training "
      "throughout, and this is its principal methodological contribution.",
      "The revised manuscript has a new title, abstract and conclusions. We have retained "
      "and strengthened the interpretability analysis rather than removing it, as reviewer "
      "2 identified it as a strength; its conclusions are now stated as a caution.")),
    ("Editor-in-Chief, abstract structure and highlights",
     ("The abstract now uses the three required named sections: Study Region, Study Focus, "
      "and New Hydrological Insights for the Region.",
      "Five highlights are supplied in a separate file, each within the 85-character limit.")),
    ("Reviewer 1, comment 1: limitations",
     ("Section 5.2 is a dedicated subsection organised into measured limitations, design "
      "and data limitations, and future perspectives. Six of the twelve items are now "
      "quantified rather than asserted, including the single-run resolution floor, the seed "
      "component, the underdispersion of all four systems and the amplitude confounding of "
      "the extreme-event metric.",)),
    ("Reviewer 1, comment 2: extreme values",
     ("Section 2.1 states that target fractional anomalies were not truncated or "
      "winsorized, that training uses a valid-land mask and gradient clipping, and that the "
      "extreme threshold is the training-period 90th percentile. Section 3.4 adds that this "
      "fixed threshold interacts with output amplitude, which we now report explicitly.",)),
    ("Reviewer 1, comment 3: is an ACC difference of this size meaningful",
     ("This comment prompted the central change to the paper. The honest answer has two "
      "parts. The difference between the postprocessors is smaller than the reviewer's 0.03 "
      f"estimate: repeated training places it at {abs(GAP['mean_test']['mean']):.4f} for "
      "FiLM versus the baseline. And a single training run cannot resolve a difference of "
      f"that size, because rerunning an identical configuration moves the score by up to "
      f"{FLOOR_ACC:.4f}. We therefore quantified the resolution limit rather than reporting "
      "only a significance test, and we report additional metrics as the reviewer suggests, "
      "with the caution that one of them, extreme-event CSI, is confounded with amplitude "
      "(Section 3.4).",)),
    ("Reviewer 1, comment 4: separating U-Net postprocessing from the FiLM contribution",
     ("Table 1 and Section 3.2 now separate these explicitly. Postprocessing with "
      "atmospheric predictors alone accounts for the improvement over raw SEAS5 in spatial "
      "ACC. The incremental contribution of SST conditioning is negative: "
      f"{GAP['mean_test']['mean']:+.4f} +/- {GAP['mean_test']['sem']:.4f} ACC for FiLM "
      "relative to the unconditioned baseline over five retrainings.",)),
    ("Reviewer 1, comment 5: predictor importance",
     ("Section 3.6 and Figure 8 report permutation importance for all ten atmospheric "
      "predictors. Total precipitation dominates; every other predictor has an absolute "
      "mean effect below 0.005. We note that correlated circulation variables can "
      "substitute for one another, so a small single-variable effect does not establish "
      "that a predictor is uninformative.",)),
    ("Reviewer 1, comment 6: regional differences",
     ("Section 3.5 reports three non-overlapping geographic partitions with cell counts and "
      "per-region ACC for all four systems. We describe the pattern and name candidate "
      "explanations without asserting attribution, and we now add that differences between "
      "postprocessors within a region are of the same order as the training spread and "
      "should be read as provisional.",)),
    ("Reviewer 1, minor comments",
     ("Abbreviations are defined at first occurrence. Spacing and sentence structure have "
      "been corrected throughout. The predictor list in Section 2.1 now enumerates all ten "
      "variables. Figure colour schemes, legend placement, panel labelling and shared "
      "colour bars have been revised; all spatial panels share one display range. EOF-cat "
      "is defined where it first appears.",)),
    ("Reviewer 2, comment 1: test period length and ENSO representativeness",
     ("Section 3.7 reports ACC and TCC stratified by Oceanic Nino Index category and adds "
      "two retrospective training origins yielding nonoverlapping 2014-2019 and 2020-2022 "
      "holdouts, 107 months in total, which include non-El-Nino conditions as the reviewer "
      "requested. We also now give a quantitative reason not to over-read the two origins: "
      "they rank FiLM oppositely, but the span between them exceeds the training "
      "variability measured in Section 3.2, so training noise alone is sufficient to "
      "produce the reversal.",)),
    ("Reviewer 2, comment 2: EOF fitting window and information leakage",
     ("The reviewer was correct and we are grateful for the comment. The EOF basis, "
      "standardisation statistics, atmospheric normalisation and climatology are now all "
      "fitted on 1994-2017 only and projected forward; Section 2.1 states each fitting "
      "window explicitly. The entire analysis was re-run. The reported skill changes as a "
      "result, and so does the ranking.",)),
    ("Reviewer 2, comment 3: FiLM versus EOF-cat in the structural analyses",
     ("All four systems now appear in every figure and table where the comparison is "
      "meaningful, including the gridwise TCC maps, the TCC difference maps, the spatial "
      "examples, the regional table and the ENSO stratification. Paired FiLM-minus-EOF-cat "
      "differences are reported by month, by year and spatially.",)),
    ("Reviewer 2, comment 4: EOF-cat degradation and fairness of the comparison",
     ("Section 2.2 documents the matched elements and the unmatched ones. Both SST variants "
      "use the same five PCs from the same six-month window, the same split, optimiser, "
      "seeds, epoch budget and checkpoint rule. Insertion depth and parameter count are not "
      "matched, and we state that the earlier exploratory search cannot be equalised "
      "retroactively, so the comparison does not isolate a causal effect of the FiLM "
      "operator. Section 3.6 adds that EOF-cat's loss of extreme-event skill relative to "
      "its own PC-zeroed control is visible only after equalising amplitude.",)),
    ("Reviewer 2, comment 5: significance of the ACC differences",
     ("Paired Wilcoxon tests and month-resampled and calendar-year-block bootstrap "
      "intervals are reported in Section 3.3 for all comparisons. We also report a "
      "limitation of these statistics that bears directly on the reviewer's concern: they "
      "resample months around one fixed set of trained models and therefore exclude "
      "training variability. Section 3.2 supplies that missing component. One configuration "
      "in our own experiments produced an interval excluding zero on its first run and "
      "including zero when retrained, which is why we no longer rely on month-resampled "
      "intervals alone. The phrase 'significantly outperforms' has been removed.",)),
    ("Reviewer 2, comment 6: symmetric ablation and controls",
     ("Section 3.6 and Table 7 give the full symmetric matrix: full inputs, SST zeroed, SST "
      "shuffled and local predictors zeroed, for both EOF-cat and FiLM, with the baseline "
      "as reference. The shuffled-PC control requested for the parameter-correlation "
      "analysis is retained in Section 4.2, where real-PC correlations of 0.41 to 0.45 are "
      "compared against a 100-permutation ceiling of 0.072 to 0.081, and a shuffled-PC "
      "control for the sensitivity maps appears in Figure 12. These confirm that the "
      "alignment is not a trivial consequence of the conditioning network's capacity, which "
      "was the reviewer's specific concern. We have also added an amplitude column to the "
      "ablation table, because zeroing an input both removes information and shrinks the "
      "output; without it the apparent 40 percent CSI collapse under PC zeroing reads as an "
      "information effect when at equal amplitude it is 1.2 percent. Section 4 now closes "
      "by noting that these interpretability diagnostics, all of which pass, describe a "
      "pathway that contributes no measurable skill.",)),
)


def response(chinese=False):
    doc = Document()
    setup(doc, chinese=chinese)
    title = "审稿意见逐条回复" if chinese else "Response to Reviewers"
    para(doc, title, "Title")
    para(doc, "Manuscript EJRH-D-26-03375")
    para(doc, "Below the Resolution of a Single Training Run: SST-Conditioned U-Net "
              "Postprocessing of Monthly Precipitation over China")
    if chinese:
        para(doc, "本文件为英文回复信的中文对照稿，内容一致，供作者核验使用。")
    for section, paragraphs in RESPONSE_EN:
        heading(doc, section)
        for text in paragraphs:
            para(doc, text)
    name = "07_审稿回复中文.docx" if chinese else "07_Response_EN.docx"
    path = DELIVERY / name
    doc.save(path)
    return path


def main():
    DELIVERY.mkdir(parents=True, exist_ok=True)
    written = [manuscript(), highlights(), supplement(), response(False), response(True)]
    for path in written:
        print(f"wrote {path.name}  ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
