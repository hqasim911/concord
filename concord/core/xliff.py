"""
XLIFF parsing + round-trip editing (1.2 and 2.0).

Keeps the parsed tree for each file so edited targets can be written back into
the original structure (namespaces, inline tags on untouched segments, all
other attributes preserved), then re-serialized to valid XLIFF.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from lxml import etree


@dataclass
class Segment:
    sid: str               # stable id: "<filename>#<index>"
    source: str
    target: str
    unit: str
    file: str
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
                _target_el=tgt_el,
            ))
            idx += 1
    return xf


def set_target_text(seg: Segment, text: str):
    """Replace a target element's content with plain text (drops inline tags)."""
    el = seg._target_el
    if el is None:
        return
    for child in list(el):
        el.remove(child)
    el.text = text
    seg.target = text


def serialize(xf: XliffFile) -> bytes:
    return etree.tostring(
        xf.tree, xml_declaration=True, encoding="UTF-8", pretty_print=False
    )
