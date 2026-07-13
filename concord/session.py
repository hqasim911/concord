"""
ProjectState — the loaded documents and the reviewer's in-progress edits.

Extracted from the pywebview bridge so the document/edit model is a plain,
testable object with no dependency on the window or the JS transport. The
bridge (api.py) owns analysis-run state (flags, glossary, model identity) and
delegates all file/segment/edit operations here.
"""

from __future__ import annotations
import os
from typing import Dict, List

from .core.xliff import XliffFile, Segment


class ProjectState:
    def __init__(self):
        self.files: Dict[str, XliffFile] = {}
        self.segments: List[Segment] = []
        self.seg_by_sid: Dict[str, Segment] = {}
        self.edits: Dict[str, str] = {}        # sid -> new target

    # ---- files / segments ----
    def files_summary(self) -> List[dict]:
        return [{"name": n, "segments": len(f.segments)}
                for n, f in self.files.items()]

    def ingest(self, paths: List[str], parse_fn) -> dict:
        """Parse and add each path (parse_fn: path -> XliffFile), reporting any
        per-file errors, then rebuild the flattened segment list."""
        added = []
        for p in paths:
            try:
                xf = parse_fn(p)
                self.files[xf.name] = xf
                added.append({"name": xf.name, "segments": len(xf.segments)})
            except Exception as e:
                added.append({"name": os.path.basename(p), "error": str(e)})
        self.rebuild_segments()
        return {"files": self.files_summary(), "added": added}

    def remove_file(self, name: str) -> dict:
        self.files.pop(name, None)
        self.rebuild_segments()
        return {"files": self.files_summary()}

    def rebuild_segments(self):
        self.segments = []
        for xf in self.files.values():
            self.segments.extend(xf.segments)
        self.seg_by_sid = {s.sid: s for s in self.segments}
        # drop edits whose segment vanished
        for sid in list(self.edits):
            if sid not in self.seg_by_sid:
                self.edits.pop(sid)

    def list_segments(self, limit: int = 2000) -> dict:
        """Segments for the viewer (capped at `limit` rendered rows)."""
        segs = self.segments
        shown = segs[:limit] if limit else segs
        return {
            "total": len(segs), "shown": len(shown),
            "segments": [{"sid": s.sid, "file": s.file,
                          "source": s.source, "target": s.target}
                         for s in shown],
        }

    # ---- edits ----
    def target_of(self, occ) -> str:
        """The occurrence's current target: the reviewer's edit if any, else
        the original."""
        return self.edits.get(occ.sid, occ.target)

    def set_edit(self, sid: str, text: str) -> dict:
        seg = self.seg_by_sid.get(sid)
        if not seg:
            return {"ok": False}
        if text == seg.target:
            self.edits.pop(sid, None)
        else:
            self.edits[sid] = text
        return {"ok": True, "edits": len(self.edits)}

    def revert(self, sids: List[str]) -> dict:
        for sid in sids:
            self.edits.pop(sid, None)
        return {"ok": True, "edits": len(self.edits)}

    def revert_all(self) -> dict:
        self.edits.clear()
        return {"ok": True, "edits": 0}

    def edit_count(self) -> dict:
        files = {self.seg_by_sid[s].file
                 for s in self.edits if s in self.seg_by_sid}
        return {"edits": len(self.edits), "files": len(files)}

    @staticmethod
    def splice(target: str, lo, hi, corrected: str) -> str:
        """Replace only the aligned token range [lo, hi] in the target with the
        corrected term, keeping the rest of the segment intact."""
        toks = target.split()
        if lo is None or hi is None or lo < 0 or hi >= len(toks) or lo > hi:
            return corrected                       # can't locate → old behavior
        return " ".join(toks[:lo] + [corrected] + toks[hi + 1:])

    def apply_splice(self, occ, corrected: str) -> str:
        """Splice `corrected` into one occurrence's aligned span and record the
        edit (or clear it if the result equals the original). Returns the new
        target text."""
        new_t = self.splice(occ.target, occ.tgt_lo, occ.tgt_hi, corrected)
        if new_t == occ.target:
            self.edits.pop(occ.sid, None)
        else:
            self.edits[occ.sid] = new_t
        return new_t
