"""
pywebview backend API.

Exposes methods callable from the JS frontend via window.pywebview.api.*.
Heavy work (model load, analysis) runs in background threads and reports
progress back through the JS bridge.
"""

from __future__ import annotations
import os
import threading
import traceback
from typing import List, Dict, Optional

from .core.aligner import build_aligner
from .core.engine import ConsistencyEngine, EngineConfig
from .core.xliff import parse_xliff, set_target_text, serialize, XliffFile, Segment
from .core import llm as llm_mod


class ConcordAPI:
    def __init__(self):
        self._window = None
        self._aligner = None
        self._files: Dict[str, XliffFile] = {}
        self._segments: List[Segment] = []
        self._seg_by_sid: Dict[str, Segment] = {}
        self._edits: Dict[str, str] = {}       # sid -> new target
        self._flags = []
        self._reverse = []
        self._glossary = []
        self._llm_cfg: Optional[llm_mod.LLMConfig] = None
        self._model_kind = "simalign"
        self._model_name = "bert"

    def set_window(self, window):
        self._window = window

    # ---- JS bridge helpers ----
    def _emit(self, event: str, payload: dict):
        if not self._window:
            return
        import json
        js = f"window.dispatchEvent(new CustomEvent('{event}', {{detail: {json.dumps(payload)}}}))"
        try:
            self._window.evaluate_js(js)
        except Exception:
            pass

    # ---- model lifecycle ----
    def load_model(self, kind: str = "simalign", model: str = "bert") -> dict:
        """Load the alignment model (downloads on first run). Runs in a thread."""
        def work():
            try:
                self._emit("model-status", {"state": "loading", "model": model})
                self._aligner = build_aligner(kind, model=model)
                self._model_kind, self._model_name = kind, model
                self._emit("model-status", {"state": "ready", "model": model})
            except Exception as e:
                self._emit("model-status", {"state": "error", "error": str(e),
                                            "trace": traceback.format_exc()[-800:]})
        threading.Thread(target=work, daemon=True).start()
        return {"started": True}

    # ---- file intake ----
    def open_files(self) -> dict:
        """Native file picker -> parse selected XLIFF files."""
        from webview import OPEN_DIALOG
        paths = self._window.create_file_dialog(
            OPEN_DIALOG, allow_multiple=True,
            file_types=("XLIFF (*.xlf;*.xliff;*.mxliff;*.sdlxliff)", "All files (*.*)"),
        )
        if not paths:
            return {"files": []}
        return self._ingest(list(paths))

    def _ingest(self, paths: List[str]) -> dict:
        added = []
        for p in paths:
            try:
                xf = parse_xliff(p)
                self._files[xf.name] = xf
                added.append({"name": xf.name, "segments": len(xf.segments)})
            except Exception as e:
                added.append({"name": os.path.basename(p), "error": str(e)})
        self._rebuild_segments()
        return {"files": [{"name": n, "segments": len(f.segments)}
                          for n, f in self._files.items()],
                "added": added}

    def remove_file(self, name: str) -> dict:
        self._files.pop(name, None)
        self._rebuild_segments()
        return {"files": [{"name": n, "segments": len(f.segments)}
                          for n, f in self._files.items()]}

    def _rebuild_segments(self):
        self._segments = []
        for xf in self._files.values():
            self._segments.extend(xf.segments)
        self._seg_by_sid = {s.sid: s for s in self._segments}
        # drop edits whose segment vanished
        for sid in list(self._edits):
            if sid not in self._seg_by_sid:
                self._edits.pop(sid)

    # ---- analysis ----
    def analyze(self, cfg: dict) -> dict:
        if self._aligner is None:
            return {"error": "Model not loaded yet."}

        def work():
            try:
                ec = EngineConfig(
                    nmin=int(cfg.get("nmin", 2)),
                    nmax=int(cfg.get("nmax", 3)),
                    stop_mode=cfg.get("stop_mode", "trim"),
                    min_occurrences=int(cfg.get("min_occurrences", 2)),
                    fold_taa=bool(cfg.get("fold_taa", True)),
                    strip_clitics=bool(cfg.get("strip_clitics", True)),
                    cluster_spans=bool(cfg.get("cluster_spans", True)),
                    cluster_max_dist=float(cfg.get("cluster_max_dist", 0.2)),
                    min_variant_count=int(cfg.get("min_variant_count", 1)),
                )
                engine = ConsistencyEngine(self._aligner, ec)

                def prog(done, total):
                    self._emit("analyze-progress", {"done": done, "total": total})
                flags = engine.analyze(self._segments, progress=prog)
                self._flags = flags
                reverse = []
                if cfg.get("reverse"):
                    reverse = engine.analyze_reverse(self._segments)
                self._reverse = reverse
                self._emit("analyze-done", {
                    "segments": len(self._segments),
                    "files": len(self._files),
                    "flags": self._flags_to_json(flags),
                    "reverse": self._reverse_to_json(reverse),
                    "placeholder_issues": self._placeholder_count(),
                })
            except Exception as e:
                self._emit("analyze-error", {"error": str(e),
                                             "trace": traceback.format_exc()[-800:]})
        threading.Thread(target=work, daemon=True).start()
        return {"started": True}

    def _flags_to_json(self, flags) -> list:
        out = []
        for f in flags:
            out.append({
                "ngram": f.ngram, "distinct": f.distinct, "total": f.total,
                "score": round(f.score, 3),
                "variants": [{
                    "span": v.span, "count": v.count,
                    "occurrences": [{
                        "sid": o.sid, "file": o.file, "unit": o.unit,
                        "source": o.source,
                        "target": self._edits.get(o.sid, o.target),
                        "original": o.target, "span": o.span,
                    } for o in v.occurrences],
                } for v in f.variants],
            })
        return out

    def _reverse_to_json(self, rflags) -> list:
        out = []
        for f in rflags:
            out.append({
                "span": f.span, "distinct": f.distinct, "total": f.total,
                "score": round(f.score, 3),
                "uses": [{
                    "term": u.term, "count": u.count,
                    "occurrences": [{
                        "sid": o.sid, "file": o.file, "unit": o.unit,
                        "source": o.source, "span": o.span,
                        "target": self._edits.get(o.sid, o.target),
                    } for o in u.occurrences],
                } for u in f.uses],
            })
        return out

    def _placeholder_count(self) -> int:
        from .core.xliff import placeholder_issues
        return len(placeholder_issues(self._segments))

    # ---- editing ----
    def set_edit(self, sid: str, text: str) -> dict:
        seg = self._seg_by_sid.get(sid)
        if not seg:
            return {"ok": False}
        if text == seg.target:
            self._edits.pop(sid, None)
        else:
            self._edits[sid] = text
        return {"ok": True, "edits": len(self._edits)}

    def revert(self, sids: List[str]) -> dict:
        for sid in sids:
            self._edits.pop(sid, None)
        return {"ok": True, "edits": len(self._edits)}

    def revert_all(self) -> dict:
        self._edits.clear()
        return {"ok": True, "edits": 0}

    def edit_count(self) -> dict:
        files = {self._seg_by_sid[s].file for s in self._edits if s in self._seg_by_sid}
        return {"edits": len(self._edits), "files": len(files)}

    # ---- export ----
    def export(self) -> dict:
        from webview import FOLDER_DIALOG
        if not self._edits:
            return {"ok": False, "msg": "No edits to export."}
        folder = self._window.create_file_dialog(FOLDER_DIALOG)
        if not folder:
            return {"ok": False, "msg": "No folder chosen."}
        out_dir = folder[0] if isinstance(folder, (list, tuple)) else folder

        # apply edits to DOM
        dirty_files = set()
        dropped_tags = 0
        for sid, text in self._edits.items():
            seg = self._seg_by_sid.get(sid)
            if seg:
                res = set_target_text(seg, text)
                if res.get("dropped_tags"):
                    dropped_tags += 1
                dirty_files.add(seg.file)

        written = []
        for name in dirty_files:
            xf = self._files[name]
            base, ext = os.path.splitext(name)
            out_name = f"{base}-corrected{ext or '.xlf'}"
            out_path = os.path.join(out_dir, out_name)
            with open(out_path, "wb") as fh:
                fh.write(serialize(xf))
            written.append(out_name)
        return {"ok": True, "written": written, "dir": out_dir,
                "dropped_tags": dropped_tags}

    # ---- LLM ----
    def set_llm(self, base_url: str, api_key: str, model: str,
                provider: str = "auto") -> dict:
        if not (base_url and api_key and model):
            self._llm_cfg = None
            return {"ok": False, "msg": "Missing fields."}
        self._llm_cfg = llm_mod.LLMConfig(base_url, api_key, model,
                                          provider=provider)
        res = llm_mod.test_connection(self._llm_cfg)
        if not res.get("ok"):
            self._llm_cfg = None
        return res

    def llm_judge(self, ngram: str, spans: List[str]) -> dict:
        if not self._llm_cfg:
            return {"error": "LLM not configured."}
        return llm_mod.judge_group(self._llm_cfg, ngram, spans)

    def llm_judge_all(self) -> dict:
        """Run an LLM verdict over every current forward flag, concurrently."""
        if not self._llm_cfg:
            return {"error": "LLM not configured."}
        items = [{"ngram": f.ngram, "spans": [v.span for v in f.variants]}
                 for f in self._flags]
        verdicts = llm_mod.judge_all(self._llm_cfg, items)
        return {"verdicts": [
            {"ngram": it["ngram"], **v} for it, v in zip(items, verdicts)
        ]}

    # ---- glossary / termbase ----
    def load_glossary(self) -> dict:
        from webview import OPEN_DIALOG
        from .core import glossary as gl
        paths = self._window.create_file_dialog(
            OPEN_DIALOG, allow_multiple=False,
            file_types=("Glossary (*.csv;*.tbx;*.xml)", "All files (*.*)"),
        )
        if not paths:
            return {"ok": False, "entries": 0}
        path = paths[0] if isinstance(paths, (list, tuple)) else paths
        try:
            self._glossary = gl.load_glossary(path)
        except Exception as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "entries": len(self._glossary)}

    def check_glossary(self) -> dict:
        from .core import glossary as gl
        if not self._glossary:
            return {"ok": False, "msg": "No glossary loaded."}
        viols = gl.check_adherence(self._segments, self._glossary)
        return {"ok": True, "count": len(viols), "violations": [{
            "sid": v.sid, "file": v.file, "term": v.source_term,
            "approved": v.approved, "source": v.segment_source,
            "target": v.segment_target,
        } for v in viols]}

    # ---- placeholder QA ----
    def placeholder_report(self) -> dict:
        from .core.xliff import placeholder_issues
        issues = placeholder_issues(self._segments)
        return {"count": len(issues), "items": [{
            "sid": s.sid, "file": s.file, "source": s.source,
            "target": s.target, "src_ph": s.src_ph, "tgt_ph": s.tgt_ph,
        } for s in issues]}
