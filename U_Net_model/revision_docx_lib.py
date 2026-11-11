# -*- coding: utf-8 -*-
"""
Minimal, format-preserving DOCX editor used by the EJRH-D-26-03375 revision build.

It edits word/document.xml in place (lxml keeps the original namespace prefixes),
replaces embedded media by relationship target, and rebuilds table cells.
No file is ever modified in place: the caller always supplies a distinct output path.
"""
from __future__ import annotations

import copy
import io
import re
import zipfile
from pathlib import Path

from lxml import etree
from PIL import Image

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS = {"w": W, "r": R, "wp": WP, "a": A}
_TOUCH_ATTR = "_rev_touched"


def q(ns: str, tag: str) -> str:
    return "{%s}%s" % (ns, tag)


def boundary_replace(text: str, mapping: dict[str, str]) -> str:
    """
    Replace tokens with strict numeric/identifier boundary protection so e.g.
    "0.000" never corrupts "-0.0007" and "0.396" never matches "0.3967".
    """
    out = text
    for old in sorted(mapping, key=len, reverse=True):
        new = mapping[old]
        if not old or old == new:
            continue
        if old not in out:
            continue
        # If the token is purely ASCII numeric/punctuation (e.g. -0.0283, +36.8%, 5,301, 0.3286),
        # enforce boundaries against adjacent letters, digits, dots, or minus signs.
        if re.fullmatch(r"[+\-0-9.,%]+", old):
            pat = re.compile(r"(?<![0-9A-Za-z.\-])" + re.escape(old) + r"(?![0-9A-Za-z.%])")
            out = pat.sub(lambda _m: new, out)
        else:
            out = out.replace(old, new)
    return out


class DocxEditor:
    def __init__(self, src: str | Path):
        self.src = Path(src)
        self.zin = zipfile.ZipFile(self.src, "r")
        self.parts = {n: self.zin.read(n) for n in self.zin.namelist()}
        self.zin.close()
        self.xml = etree.fromstring(self.parts["word/document.xml"])
        self.rels = etree.fromstring(self.parts["word/_rels/document.xml.rels"])
        self.body = self.xml.find(q(W, "body"))
        self.rid_to_target = {
            rel.get("Id"): rel.get("Target") for rel in self.rels
        }
        self.changed_media: dict[str, bytes] = {}

    def is_touched(self, p) -> bool:
        return p.get(_TOUCH_ATTR) == "1"

    def _mark_touched(self, p, text: str = ""):
        p.set(_TOUCH_ATTR, "1")

    # ---------- text helpers ----------
    @staticmethod
    def para_text(p) -> str:
        return "".join(t.text or "" for t in p.findall(".//" + q(W, "t")))

    @staticmethod
    def set_para_text(p, text: str, red: bool = False, bold: bool = False):
        """Replace all runs of a paragraph with one run, keeping w:pPr."""
        for child in list(p):
            if child.tag != q(W, "pPr"):
                p.remove(child)
        r = etree.SubElement(p, q(W, "r"))
        rPr = etree.SubElement(r, q(W, "rPr"))
        if red:
            c = etree.SubElement(rPr, q(W, "color"))
            c.set(q(W, "val"), "DA291C")
        if bold:
            etree.SubElement(rPr, q(W, "b"))
        t = etree.SubElement(r, q(W, "t"))
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        t.text = text
        return p

    def new_para(self, text: str, template=None, red: bool = False, bold: bool = False):
        if template is None:
            template = next(
                (p for p in self.body if p.tag == q(W, "p") and len(self.para_text(p)) > 40),
                None,
            )
        p = copy.deepcopy(template) if template is not None else etree.Element(q(W, "p"))
        return self.set_para_text(p, text, red=red, bold=bold)

    # ---------- paragraph-level replacement ----------
    def replace_paragraphs(self, rules: list[tuple[str, str, bool]]):
        """rules: list of (unique original substring, new text, mark_red)."""
        applied = 0
        seen_needles = set()
        all_p = self.xml.findall(".//" + q(W, "p"))
        for needle, new_text, red in rules:
            hits = [p for p in all_p if needle in self.para_text(p)]
            if len(hits) == 0:
                print(f"  [WARN] no paragraph matched: {needle[:70]}")
                continue
            if len(hits) > 1:
                print(f"  [WARN] {len(hits)} paragraphs matched, using first: {needle[:70]}")
            p = hits[0]
            if p.get(_TOUCH_ATTR) == "1":
                print(f"  [WARN] paragraph targeted twice: {needle[:70]}")
            self.set_para_text(p, new_text, red=red)
            self._mark_touched(p, new_text)
            applied += 1
            seen_needles.add(needle)
        return applied

    def insert_after(self, needle: str, texts: list[tuple[str, bool, bool]], template=None):
        """Insert new paragraphs after the first paragraph containing needle."""
        all_p = [p for p in self.body if p.tag == q(W, "p")]
        target = next((p for p in all_p if needle in self.para_text(p)), None)
        if target is None:
            print(f"  [WARN] insert anchor not found: {needle[:70]}")
            return 0
        idx = list(self.body).index(target)
        for offset, (text, red, bold) in enumerate(texts, 1):
            new_p = self.new_para(text, template=template, red=red, bold=bold)
            self._mark_touched(new_p, text)
            self.body.insert(idx + offset, new_p)
        return len(texts)

    # ---------- tables ----------
    def tables(self):
        return [t for t in self.body if t.tag == q(W, "tbl")]

    def set_table(self, index: int, rows: list[list[str]], red: bool = True):
        tbls = self.tables()
        if index >= len(tbls):
            raise IndexError(f"table {index} not found (document has {len(tbls)} tables)")
        tbl = tbls[index]
        trs = tbl.findall(q(W, "tr"))
        if len(trs) != len(rows):
            raise ValueError(f"table {index}: expected {len(rows)} rows, found {len(trs)}")
        for tr, row in zip(trs, rows):
            tcs = tr.findall(q(W, "tc"))
            if len(tcs) != len(row):
                raise ValueError(
                    f"table {index}: expected {len(row)} cells, found {len(tcs)} in row"
                )
            for tc, value in zip(tcs, row):
                ps = tc.findall(q(W, "p"))
                self.set_para_text(ps[0], value, red=red)
                self._mark_touched(ps[0], value)
                for extra in ps[1:]:
                    self.set_para_text(extra, "", red=False)
                    self._mark_touched(extra, "")

    # ---------- media ----------
    def replace_media(self, mapping: dict[str, str | Path]):
        """mapping: media target ('media/image5.png') -> new image file path."""
        for target_path, new_file in mapping.items():
            key = "word/" + target_path.lstrip("/")
            if key not in self.parts:
                print(f"  [WARN] media missing in package: {key}")
                continue
            data = Path(new_file).read_bytes()
            self.changed_media[key] = data
            if key.lower().endswith(".png"):
                rid = next(
                    (rid for rid, tgt in self.rid_to_target.items()
                     if tgt and tgt.replace("\\", "/").lstrip("/") == target_path.lstrip("/")),
                    None,
                )
                if rid:
                    with Image.open(io.BytesIO(data)) as im:
                        self._update_extent(rid, im.width, im.height)
        return len(self.changed_media)

    def _update_extent(self, rid: str, px_w: int, px_h: int):
        for drawing in self.xml.iter():
            if drawing.tag not in (q(WP, "inline"), q(WP, "anchor")):
                continue
            blips = [
                b for b in drawing.iter()
                if b.tag in (q(A, "blip"), q(A, "svgBlip"), q(A, "imagedata"))
                and (b.get(q(R, "embed")) or b.get(q(R, "id")) or b.get(q(R, "link"))) == rid
            ]
            if not blips:
                continue
            ext = drawing.find(q(WP, "extent"))
            if ext is None:
                continue
            cx = int(ext.get("cx"))
            cy = int(round(cx * px_h / max(px_w, 1)))
            ext.set("cy", str(cy))
            for e in drawing.iter(q(A, "ext")):
                if e.get("cx"):
                    e.set("cy", str(cy))
                    e.set("cx", str(cx))
            break

    # ---------- output ----------
    def save(self, dst: str | Path):
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Strip temporary internal tracking attribute before serializing valid OOXML
        xml_copy = copy.deepcopy(self.xml)
        for elem in xml_copy.iter():
            if _TOUCH_ATTR in elem.attrib:
                del elem.attrib[_TOUCH_ATTR]
        self.parts["word/document.xml"] = etree.tostring(
            xml_copy, xml_declaration=True, encoding="UTF-8", standalone=True
        )
        for key, data in self.changed_media.items():
            self.parts[key] = data
        with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
            for name, data in self.parts.items():
                zout.writestr(name, data)
        return dst
