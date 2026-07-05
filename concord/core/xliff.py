"""
XLIFF parsing + round-trip editing (1.2 and 2.0).

Keeps the parsed tree for each file so edited targets can be written back into
the original structure (namespaces, inline tags on untouched segments, all
other attributes preserved), then re-serialized to valid XLIFF.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Optional
from lxml import etree


# XLIFF inline elements (1.2 + 2.0) and literal placeholder patterns.
_INLINE_TAGS = {"ph", "x", "g", "bx", "ex", "bpt", "ept", "it",
                "mrk", "sub", "pc", "sc", "ec"}
_PH_TEXT_RE = re.compile(
    r"%\d+\$[sd@]|%[sd@]|\{\{\w+\}\}|\{\d+\}|\{\w+\}"
)


@dataclass
class Segment:
    sid: str               # stable id: "<filename>#<index>"
    source: str
    target: str
    unit: str
    file: str
    src_ph: List[str] = field(default_factory=list)   # source placeholder codes
    tgt_ph: List[str] = field(default_factory=list)   # target placeholder codes
    _target_el: object = field(repr=False, default=None)  # lxml element


@dataclass
class XliffFile:
    name: str
    path: str
    tree: object = field(repr=False, default=None)
    root: object = field(repr=False, default=None)
    segments: List[Segment] = field(default_factory=list)
    dirty: bool = False


def _localname(el) -> str:
    tag = el.tag
    if isinstance(tag, str) and "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _first_child(el, localname):
    for c in el:
        if _localname(c) == localname:
            return c
    # fall back to any descendant
    for c in el.iter():
        if c is not el and _localname(c) == localname:
            return c
    return None


def _text_of(el) -> str:
    if el is None:
        return ""
    return " ".join("".join(el.itertext()).split())


def extract_placeholders(el) -> List[str]:
    """Collect placeholder codes from an element: inline placeholder tags
    (identified by id / equiv-text) plus literal patterns (%s, {0}, {name}).
    Returned sorted so two segments can be compared as multisets."""
    if el is None:
        return []
    codes: List[str] = []
    for c in el.iter():
        if c is el:
            continue
        ln = _localname(c)
        if ln in _INLINE_TAGS:
            code = (c.get("id") or c.get("equiv-text")
                    or c.get("dataRef") or c.get("ctype") or "")
            codes.append(f"{ln}:{code}")
    codes.extend(_PH_TEXT_RE.findall("".join(el.itertext())))
    return sorted(codes)


def parse_xliff(path: str, name: Optional[str] = None) -> XliffFile:
    name = name or path.split("/")[-1].split("\\")[-1]
    parser = etree.XMLParser(remove_blank_text=False, recover=False)
    tree = etree.parse(path, parser)
    root = tree.getroot()
    xf = XliffFile(name=name, path=path, tree=tree, root=root)

    idx = 0
    # XLIFF 2.0: <unit><segment><source/><target/>
    units = [e for e in root.iter() if _localname(e) == "unit"]
    if units:
        for u in units:
            uid = u.get("id", "")
            segs = [e for e in u.iter() if _localname(e) == "segment"]
            holders = segs if segs else [u]
            for s in holders:
                src_el = _first_child(s, "source")
                tgt_el = _first_child(s, "target")
                if src_el is not None and tgt_el is not None:
                    xf.segments.append(Segment(
                        sid=f"{name}#{idx}", source=_text_of(src_el),
                        target=_text_of(tgt_el), unit=uid, file=name,
                        src_ph=extract_placeholders(src_el),
                        tgt_ph=extract_placeholders(tgt_el),
                        _target_el=tgt_el,
                    ))
                    idx += 1
        return xf

    # XLIFF 1.2: <trans-unit><source/><target/>
    for tu in [e for e in root.iter() if _localname(e) == "trans-unit"]:
        uid = tu.get("id", "")
        src_el = _first_child(tu, "source")
        tgt_el = _first_child(tu, "target")
        if src_el is not None and tgt_el is not None:
            xf.segments.append(Segment(
                sid=f"{name}#{idx}", source=_text_of(src_el),
                target=_text_of(tgt_el), unit=uid, file=name,
                src_ph=extract_placeholders(src_el),
                tgt_ph=extract_placeholders(tgt_el),
                _target_el=tgt_el,
            ))
            idx += 1
    return xf


def placeholder_issues(segments) -> List[Segment]:
    """Segments whose target placeholder set differs from the source's —
    a missing/extra/duplicated placeholder is a hard localization defect."""
    return [s for s in segments if sorted(s.src_ph) != sorted(s.tgt_ph)]


def set_target_text(seg: Segment, text: str):
    """
    Replace a target element's content with plain text.

    If the segment carries inline tags and the new text is unchanged from the
    current flattened text, the inline structure is preserved (a no-op edit no
    longer destroys formatting). When the text genuinely changes we must write
    plain text; the returned dict reports whether inline tags were dropped so
    callers can warn the user.
    """
    el = seg._target_el
    if el is None:
        return {"ok": False, "dropped_tags": False}
    had_tags = len(el) > 0
    if had_tags and text.strip() == _text_of(el):
        seg.target = _text_of(el)          # no real change — keep tags intact
        return {"ok": True, "dropped_tags": False}
    for child in list(el):
        el.remove(child)
    el.text = text
    seg.target = text
    return {"ok": True, "dropped_tags": had_tags}


def serialize(xf: XliffFile) -> bytes:
    return etree.tostring(
        xf.tree, xml_declaration=True, encoding="UTF-8", pretty_print=False
    )
