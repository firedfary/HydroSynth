"""Check the written revision-07 documents against their source result files.

Every quantitative claim in the manuscript is interpolated from a result file
at build time, so this script re-reads the finished .docx and confirms that
the numbers actually present in the text match the sources. It is a guard
against the build script drifting from the data, and against a stale document
being delivered after a rerun.

It also enforces the submission constraints that are easy to violate silently:
highlight length, the three named abstract sections, and the presence of every
figure and table the text refers to.
"""

import json
import re
import sys
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revision07_docs import (  # noqa: E402
    AMP_ROWS, CHANCE_CSI, DELIVERY, FIGVALS, FLOOR, METRICS,
)

GAP = FLOOR["gap_film_minus_baseline"]["mean_test"]
FLOOR_ACC = FLOOR["single_run_floor_acc"]
RESEED = FLOOR["variance_sources"]["sd_of_3_seed_ensemble_if_reseeded"]
TEST = METRICS["test"]["ensemble"]
VAL = METRICS["validation"]["ensemble"]

failures = []


def check(label, condition, detail=""):
    status = "OK " if condition else "BAD"
    if not condition:
        failures.append(f"{label}: {detail}")
    print(f"  [{status}] {label}" + (f"  {detail}" if detail and not condition else ""))


def document_text(path):
    """All paragraph and table text, in document order."""
    doc = Document(path)
    chunks = []
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            chunks.append(Paragraph(child, doc).text)
        elif child.tag.endswith("}tbl"):
            for row in Table(child, doc).rows:
                chunks.append(" | ".join(c.text for c in row.cells))
    return "\n".join(chunks), doc


def main():
    manuscript = DELIVERY / "07_可信度返修稿.docx"
    text, doc = document_text(manuscript)
    flat = re.sub(r"\s+", " ", text)

    print("Headline statistics present and correct:")
    check("mean gap", f"{abs(GAP['mean']):.4f}" in flat,
          f"expected {abs(GAP['mean']):.4f}")
    check("standard error", f"{GAP['sem']:.4f}" in flat, f"expected {GAP['sem']:.4f}")
    check("t p-value", f"{GAP['p']:.4f}" in flat, f"expected {GAP['p']:.4f}")
    check("single-run floor", f"{FLOOR_ACC:.4f}" in flat, f"expected {FLOOR_ACC:.4f}")
    check("reseeding shift", f"{RESEED:.4f}" in flat, f"expected {RESEED:.4f}")

    print("\nTable 1 model scores:")
    for name, key in (("SEAS5", "seas5"), ("baseline", "baseline"),
                      ("EOF-cat", "eof_cat"), ("FiLM", "film")):
        value = f"{TEST[key]['mean_acc']:.3f}"
        check(f"exploratory ACC {name}", value in flat, f"expected {value}")

    print("\nAmplitude and CSI:")
    for name, key in (("Baseline", "Baseline U-Net"), ("EOF-cat", "EOF-cat (full)"),
                      ("FiLM", "FiLM-U-Net (full)")):
        raw = f"{AMP_ROWS[key]['csi']:.3f}"
        eq = f"{AMP_ROWS[key]['csi_at_obs_amplitude']:.3f}"
        check(f"raw CSI {name}", raw in flat, f"expected {raw}")
        check(f"equalised CSI {name}", eq in flat, f"expected {eq}")
    check("chance level", f"{CHANCE_CSI:.3f}" in flat)
    check("ranking reverses", FIGVALS["amplitude"]["ranking_reverses"])
    raw_order = " > ".join(FIGVALS["amplitude"]["raw_csi_ranking"])
    eq_order = " > ".join(FIGVALS["amplitude"]["equalised_csi_ranking"])
    check("raw ranking stated", raw_order in flat, f"expected '{raw_order}'")
    check("equalised ranking stated", eq_order in flat, f"expected '{eq_order}'")

    print("\nStructure:")
    for section in ("Study Region:", "Study Focus:",
                    "New Hydrological Insights for the Region:"):
        check(f"abstract section '{section}'", section in flat)
    for n in range(1, 10):
        check(f"Figure {n} caption", f"Figure {n}." in flat)
    for n in range(10, 13):
        check(f"Figure {n} caption", f"Figure {n}." in flat)
    for n in range(1, 10):
        check(f"Table {n} caption", f"Table {n}." in flat)
    for section in ("1 Introduction", "2 Data and methods", "3 Results",
                    "3.2 Training variability", "3.4 Output amplitude",
                    "4 What the conditioning pathway does",
                    "5.2 Limitations and future perspectives", "6 Conclusions"):
        check(f"section '{section}'", section in flat)

    images = len(doc.inline_shapes)
    check("embedded images", images == 12, f"found {images}, expected 12")

    print("\nClaims that must NOT appear (superseded by this revision):")
    for phrase in ("significantly outperforms", "indistinguishable on monthly",
                   "EOF-cat has the highest mean TCC and extreme CSI"):
        check(f"absent: '{phrase}'", phrase not in flat)

    print("\nHighlights:")
    hl_text, _ = document_text(DELIVERY / "Highlights.docx")
    lines = [l.strip() for l in hl_text.splitlines()
             if l.strip() and l.strip() != "Highlights"]
    check("count between 3 and 5", 3 <= len(lines) <= 5, f"found {len(lines)}")
    for line in lines:
        check(f"<=85 chars ({len(line)})", len(line) <= 85, line)

    print("\nCompanion documents:")
    for name in ("07_Supplement_S1.docx", "07_Response_EN.docx",
                 "07_审稿回复中文.docx"):
        path = DELIVERY / name
        check(f"{name} exists", path.is_file())
    resp, _ = document_text(DELIVERY / "07_Response_EN.docx")
    for item in ("Reviewer 1, comment 1", "Reviewer 1, comment 6",
                 "Reviewer 2, comment 1", "Reviewer 2, comment 6"):
        check(f"response covers {item}", item in resp)
    check("response admits the leak", "reviewer 2's comment (2) was correct" in resp.lower())

    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED")
        for item in failures:
            print(f"  - {item}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
