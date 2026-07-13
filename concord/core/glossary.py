"""
Glossary / termbase adherence checking.

Given an approved termbase (English term -> approved Arabic translation),
flag segments whose source contains the term but whose target does NOT use
the approved translation. This complements the discovery engine: discovery
finds *unknown* inconsistencies, the glossary enforces *known* decisions.

Termbase formats: CSV (source,target[,...]) or TBX (basic <termEntry>).
"""

from __future__ import annotations
import csv
import re
from dataclasses import dataclass
from typing import List, Tuple

from lxml import etree

from .textutil import normalize_ar, light_stem_ar


@dataclass
class GlossaryEntry:
    source: str          # English term
    target: str          # approved Arabic translation


@dataclass
class Violation:
    sid: str
    file: str
    source_term: str
    approved: str
    segment_source: str
    segment_target: str


_HEADER_HINTS = {"source", "english", "term", "en"}


def load_glossary_csv(path: str) -> List[GlossaryEntry]:
    entries: List[GlossaryEntry] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            s, t = row[0].strip(), row[1].strip()
            if not s or not t or s.lower() in _HEADER_HINTS:
                continue
            entries.append(GlossaryEntry(s, t))
    return entries


def load_glossary_tbx(path: str) -> List[GlossaryEntry]:
    """Minimal TBX reader: one English + one Arabic term per <termEntry>."""
    entries: List[GlossaryEntry] = []
    root = etree.parse(path).getroot()
    for entry in root.iter():
        if not entry.tag.endswith("termEntry"):
            continue
        by_lang = {}
        for ls in entry.iter():
            if not ls.tag.endswith(("langSet", "langSec")):
                continue
            lang = (ls.get("{http://www.w3.org/XML/1998/namespace}lang")
                    or ls.get("lang") or "").lower()
            terms = [t for t in ls.iter() if t.tag.endswith("term")]
            if terms and terms[0].text:
                by_lang[lang[:2]] = terms[0].text.strip()
        if by_lang.get("en") and by_lang.get("ar"):
            entries.append(GlossaryEntry(by_lang["en"], by_lang["ar"]))
    return entries


def load_glossary(path: str) -> List[GlossaryEntry]:
    if path.lower().endswith((".tbx", ".xml")):
        return load_glossary_tbx(path)
    return load_glossary_csv(path)


def _norm(s: str, fold_taa: bool, strip_clitics: bool,
          strip_diacritics: bool = True) -> str:
    s = normalize_ar(s, fold_taa, strip_diacritics)
    if strip_clitics:
        s = " ".join(light_stem_ar(w) for w in s.split())
    return s


def check_adherence(
    segments, entries: List[GlossaryEntry],
    fold_taa: bool = True, strip_clitics: bool = True,
    strip_diacritics: bool = True,
) -> List[Violation]:
    """Return segments that use a glossary source term but not its approved
    Arabic translation (compared under the SAME normalization as the engine —
    the caller must pass the analysis's fold_taa/strip_clitics/strip_diacritics
    so a term the engine treats as matching isn't reported as a violation)."""
    prepared: List[Tuple[GlossaryEntry, re.Pattern, str]] = []
    for e in entries:
        pat = re.compile(r"\b" + re.escape(e.source.lower()) + r"\b")
        appr = _norm(e.target, fold_taa, strip_clitics, strip_diacritics)
        if appr:
            prepared.append((e, pat, appr))

    viols: List[Violation] = []
    for seg in segments:
        src_low = seg.source.lower()
        tgt_norm = _norm(seg.target, fold_taa, strip_clitics, strip_diacritics)
        for e, pat, appr in prepared:
            if pat.search(src_low) and appr not in tgt_norm:
                viols.append(Violation(
                    sid=seg.sid, file=seg.file, source_term=e.source,
                    approved=e.target, segment_source=seg.source,
                    segment_target=seg.target,
                ))
    return viols
