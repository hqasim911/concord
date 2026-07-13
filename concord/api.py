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
        self._mt = None
        self._embedder = None
        from .core.termbase import TermBase
        self._termbase = TermBase().load()
        from .core.decisions import Decisions
        self._decisions = Decisions().load()
        self._llm_cfg: Optional[llm_mod.LLMConfig] = None
        self._model_kind = "simalign"
        self._model_name = "bert"
        # normalization used by the last analysis, so glossary adherence
        # compares under the same rules the engine did (defaults = engine's)
        self._norm = {"fold_taa": True, "strip_clitics": True,
                      "strip_diacritics": True}

    def set_window(self, window):
        self._window = window

    # ---- JS bridge helpers ----
    def _log(self, msg: str):
        self._emit("analyze-log", {"msg": msg})

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
    def load_model(self, kind: str = "simalign", model: str = "bert",
                   mode: str = "intersect") -> dict:
        """Load the alignment model (downloads on first run). Runs in a thread."""
        def work():
            try:
                self._emit("model-status", {"state": "loading", "kind": kind,
                                            "model": model, "mode": mode})
                self._aligner = build_aligner(kind, model=model, mode=mode)
                self._model_kind, self._model_name = kind, model
                self._emit("model-status", {"state": "ready", "kind": kind,
                                            "model": model, "mode": mode})
            except Exception as e:
                self._emit("model-status", {"state": "error", "error": str(e),
                                            "kind": kind, "model": model,
                                            "trace": traceback.format_exc()[-800:]})
        threading.Thread(target=work, daemon=True).start()
        return {"started": True}

    # ---- verification models (LaBSE, MT) ----
    def _ensure_aux(self, key: str, attr: str, factory,
                    load_msg: str, ready_msg: str):
        """Lazily construct an auxiliary model (LaBSE / MT), caching it on
        `attr` and reporting load/ready/error through the aux-model-status
        bridge. Returns the (cached) model instance."""
        model = getattr(self, attr)
        if model is None:
            self._emit("aux-model-status", {"model": key, "state": "loading"})
            self._log(load_msg)
            try:
                model = factory()
            except Exception as e:
                self._emit("aux-model-status",
                           {"model": key, "state": "error", "error": str(e)})
                raise
            setattr(self, attr, model)
            self._emit("aux-model-status", {"model": key, "state": "ready"})
            self._log(ready_msg)
        return model

    def _ensure_labse(self):
        from .core import embed as emb
        return self._ensure_aux("labse", "_embedder", emb.Embedder,
                                "Loading LaBSE (~1.8GB)…", "LaBSE ready")

    def _ensure_mt(self):
        from .core import mt as mt_mod
        return self._ensure_aux("mt", "_mt", mt_mod.Translator,
                                "Loading MT model (opus-mt-ar-en, ~300MB)…",
                                "MT model ready")

    def load_labse(self) -> dict:
        threading.Thread(target=self._ensure_labse, daemon=True).start()
        return {"started": True}

    def load_mt(self) -> dict:
        threading.Thread(target=self._ensure_mt, daemon=True).start()
        return {"started": True}

    def models_status(self) -> dict:
        return {"labse": self._embedder is not None, "mt": self._mt is not None}

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

    def list_segments(self, limit: int = 2000) -> dict:
        """Segments for the viewer (capped at `limit` rendered rows)."""
        segs = self._segments
        shown = segs[:limit] if limit else segs
        return {
            "total": len(segs), "shown": len(shown),
            "segments": [{"sid": s.sid, "file": s.file,
                          "source": s.source, "target": s.target}
                         for s in shown],
        }

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
                    strip_diacritics=bool(cfg.get("strip_diacritics", True)),
                    cluster_spans=bool(cfg.get("cluster_spans", True)),
                    cluster_max_dist=float(cfg.get("cluster_max_dist", 0.2)),
                    merge_contained=bool(cfg.get("merge_contained", True)),
                    min_variant_count=int(cfg.get("min_variant_count", 1)),
                    include_consistent=bool(cfg.get("include_consistent", False)),
                    termbase=(self._termbase.check_map()
                              if cfg.get("check_termbase") else None),
                )
                self._norm = {"fold_taa": ec.fold_taa,
                              "strip_clitics": ec.strip_clitics,
                              "strip_diacritics": ec.strip_diacritics}
                size = int(cfg.get("batch_size", 0) or 0)
                bnum = max(int(cfg.get("batch_num", 1) or 1), 1)
                segs = self._segments
                if size > 0:
                    start = (bnum - 1) * size
                    segs = self._segments[start:start + size]
                    self._log(f"Batch {bnum}: segments {start}–"
                              f"{start + len(segs)} of {len(self._segments)}")

                self._log(f"Backend: {self._model_kind} / {self._model_name}")
                self._log(f"Analyzing {len(segs)} segment(s) "
                          f"across {len(self._files)} file(s)")
                self._log("Aligning unique sentence pairs…")
                engine = ConsistencyEngine(self._aligner, ec)

                def prog(done, total):
                    self._emit("analyze-progress",
                               {"done": done, "total": total, "phase": "align"})
                flags = engine.analyze(segs, progress=prog)
                if cfg.get("faithfulness_filter"):
                    flags = self._faithfulness_filter(
                        flags, ec.include_consistent,
                        float(cfg.get("faithfulness_threshold", 0.6)))
                if cfg.get("labse_prefilter"):
                    flags = self._labse_prefilter(
                        flags, ec.include_consistent,
                        float(cfg.get("prefilter_threshold", 0.98)))
                flags = self._apply_decisions(flags, ec.include_consistent)
                self._flags = flags
                inc = sum(1 for f in flags if self._is_inconsistent(f))
                self._log(f"Grouped {len(flags)} n-gram(s) — {inc} inconsistent")

                reverse = []
                if cfg.get("reverse"):
                    self._log("Reverse pass (AR→EN)…")
                    reverse = engine.analyze_reverse(segs)
                    self._log(f"Reverse: {len(reverse)} over-loaded span(s)")
                self._reverse = reverse

                from .core.xliff import placeholder_issues
                ph = len(placeholder_issues(segs))
                self._log(f"Placeholder issues: {ph}")
                self._log("Done.")
                self._emit("analyze-done", {
                    "segments": len(segs),
                    "files": len(self._files),
                    "inconsistent": inc,
                    "flags": self._flags_to_json(flags),
                    "reverse": self._reverse_to_json(reverse),
                    "placeholder_issues": ph,
                })
            except Exception as e:
                self._emit("analyze-error", {"error": str(e),
                                             "trace": traceback.format_exc()[-800:]})
        threading.Thread(target=work, daemon=True).start()
        return {"started": True}

    @staticmethod
    def _is_inconsistent(f) -> bool:
        """A flag is inconsistent if it has >=2 spans, the LaBSE pre-filter (if
        run) did not clear it as a near-identical duplicate, and the reviewer
        has not already decided it (accepted/dismissed)."""
        if f.distinct < 2 or f.decided:
            return False
        return not (f.verify and f.verify.get("cleared"))

    @staticmethod
    def _flag_items(flags, min_distinct: int = 2) -> list:
        """[{ngram, spans}] payload the verifiers (LaBSE / MT / LLM) consume."""
        return [{"ngram": f.ngram, "spans": [v.span for v in f.variants]}
                for f in flags if f.distinct >= min_distinct]

    def _sort_by_significance(self, flags) -> list:
        """Rank inconsistent flags first, then by score, then frequency."""
        flags.sort(key=lambda f: (self._is_inconsistent(f), f.score, f.total),
                   reverse=True)
        return flags

    def _apply_decisions(self, flags, include_consistent):
        """Suppress flags the reviewer already decided (accepted/dismissed) so
        they never resurface. In all-n-grams mode they stay, marked as decided;
        otherwise they are dropped from the results."""
        kept, n = [], 0
        for f in flags:
            st = self._decisions.status_of(f.ngram)
            if st:
                f.decided = st
                n += 1
                if not include_consistent:
                    continue
            kept.append(f)
        if n:
            self._log(f"Suppressed {n} previously-decided flag(s)")
        return kept

    def _faithfulness_filter(self, flags, include_consistent, threshold=0.6):
        """Drop variant spans that don't actually translate the source n-gram
        (aligner errors, e.g. brand -> لون). If a flag drops below two faithful
        variants it is no longer an inconsistency: removed, or (in include-all
        mode) kept as a consistent single-translation term."""
        from .core import embed as emb_mod
        from .core.engine import _entropy_score
        items = self._flag_items(flags)
        if not items:
            return flags
        self._ensure_labse()
        self._log(f"Faithfulness filter: checking {len(items)} flag(s) against "
                  f"the source term (threshold {threshold:.2f})…")
        rep = {r["ngram"]: r for r in
               emb_mod.term_faithfulness(self._embedder, items, threshold)}

        kept, n_drop, n_fp = [], 0, 0
        for f in flags:
            r = rep.get(f.ngram)
            if r is None:
                kept.append(f)
                continue
            faithful = [v for v, info in zip(f.variants, r["spans"])
                        if info["faithful"]]
            dropped = [{"span": info["span"], "sim": info["sim"]}
                       for info in r["spans"] if not info["faithful"]]
            if dropped:
                n_drop += len(dropped)
                f.variants = faithful
                f.dropped = dropped
                f.score = _entropy_score([v.count for v in faithful]) \
                    if len(faithful) >= 2 else 0.0
                if len(faithful) < 2:
                    n_fp += 1
            if len(faithful) >= 2 or (len(faithful) == 1 and include_consistent):
                kept.append(f)
        self._log(f"Faithfulness: dropped {n_drop} mis-aligned variant(s), "
                  f"removed {n_fp} false-positive flag(s)")
        return self._sort_by_significance(kept)

    def _labse_prefilter(self, flags, include_consistent, threshold=0.98):
        """Verify each candidate inconsistent flag with LaBSE before results are
        produced. Only flags whose variants are NEAR-IDENTICAL (similarity >=
        threshold — a pipeline duplicate/artifact) are cleared: dropped, or (in
        include-all mode) downgraded to consistent. Genuinely different spans
        stay flagged even if they are semantically acceptable synonyms."""
        from .core import embed as emb_mod
        items = self._flag_items(flags)
        if not items:
            return flags
        self._ensure_labse()
        self._log(f"LaBSE pre-filter: verifying {len(items)} flag(s) "
                  f"at identity threshold {threshold:.2f}…")
        vmap = {v["ngram"]: v
                for v in emb_mod.verify_all(self._embedder, items, threshold)}

        kept, cleared = [], 0
        for f in flags:
            v = vmap.get(f.ngram)
            if v is not None:
                v["cleared"] = (v["verdict"] == "duplicate")
                f.verify = v
                if v["cleared"]:
                    cleared += 1
                    if not include_consistent:
                        continue
            kept.append(f)
        self._log(f"LaBSE pre-filter: cleared {cleared} near-identical "
                  f"duplicate(s); kept the rest as inconsistent")
        return self._sort_by_significance(kept)

    def _occ_to_json(self, o) -> dict:
        """One occurrence -> dict, overlaying the reviewer's in-progress edit
        onto the target (falls back to the original target if unedited)."""
        return {
            "sid": o.sid, "file": o.file, "unit": o.unit,
            "source": o.source, "span": o.span,
            "target": self._edits.get(o.sid, o.target),
        }

    def _flags_to_json(self, flags) -> list:
        out = []
        for f in flags:
            out.append({
                "ngram": f.ngram, "distinct": f.distinct, "total": f.total,
                "score": round(f.score, 3),
                "inconsistent": self._is_inconsistent(f),
                "verify": f.verify, "dropped": f.dropped,
                "approved": f.termbase_approved,
                "tb_violation": f.termbase_violation,
                "decided": f.decided,
                "variants": [{
                    "span": v.span, "count": v.count,
                    "raw": (v.occurrences[0].raw if v.occurrences else v.span),
                    "occurrences": [{**self._occ_to_json(o),
                                     "original": o.target}
                                    for o in v.occurrences],
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
                    "occurrences": [self._occ_to_json(o)
                                    for o in u.occurrences],
                } for u in f.uses],
            })
        return out

    def _placeholder_count(self) -> int:
        from .core.xliff import placeholder_issues
        return len(placeholder_issues(self._segments))

    @staticmethod
    def _splice(target: str, lo, hi, corrected: str) -> str:
        """Replace only the aligned token range [lo, hi] in the target with the
        corrected term, keeping the rest of the segment intact."""
        toks = target.split()
        if lo is None or hi is None or lo < 0 or hi >= len(toks) or lo > hi:
            return corrected                       # can't locate → old behavior
        return " ".join(toks[:lo] + [corrected] + toks[hi + 1:])

    def apply_correction(self, ngram: str, corrected: str) -> dict:
        """Standardize a flagged term: splice `corrected` into every occurrence
        at its aligned span (not replace the whole segment). Returns the new
        target text per segment so the UI can update."""
        corrected = (corrected or "").strip()
        if not corrected:
            return {"ok": False}
        out = {}
        for f in self._flags:
            if f.ngram != ngram:
                continue
            for v in f.variants:
                for o in v.occurrences:
                    new_t = self._splice(o.target, o.tgt_lo, o.tgt_hi, corrected)
                    if new_t == o.target:
                        self._edits.pop(o.sid, None)
                    else:
                        self._edits[o.sid] = new_t
                    out[o.sid] = new_t
            break
        return {"ok": True, "targets": out, "edits": len(self._edits)}

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
        items = self._flag_items(self._flags, min_distinct=1)
        verdicts = llm_mod.judge_all(self._llm_cfg, items)
        return {"verdicts": [
            {"ngram": it["ngram"], **v} for it, v in zip(items, verdicts)
        ]}

    # ---- local verifier (MT back-translation | LaBSE embeddings) ----
    def verify_all(self, method: str = "labse") -> dict:
        """Verify every inconsistent flag with a local model. method='mt'
        back-translates variants (opus-mt-ar-en); method='labse' compares LaBSE
        embeddings. Loads the chosen model on first use. Returns per-flag
        verdicts in a shared shape (ngram, verdict, summary, rows)."""
        if not self._flags:
            return {"error": "Run an analysis first."}
        items = self._flag_items(self._flags)
        if not items:
            return {"verdicts": [], "method": method}
        try:
            if method == "labse":
                from .core import embed as emb_mod
                self._ensure_labse()
                self._log(f"Embedding {len(items)} flag(s)…")
                verdicts = emb_mod.verify_all(self._embedder, items)
            else:
                from .core import mt as mt_mod
                self._ensure_mt()
                n = sum(len(it["spans"]) for it in items)
                self._log(f"Back-translating {n} span(s)…")
                verdicts = mt_mod.verify_all(self._mt, items)
            self._log("Verification done")
            return {"verdicts": verdicts, "method": method}
        except Exception as e:
            return {"error": str(e), "trace": traceback.format_exc()[-600:]}

    # ---- per-flag decisions (accept / dismiss) ----
    def decide_flag(self, ngram: str, status: str, note: str = "") -> dict:
        """Record a reviewer verdict so this flag is not shown again."""
        return {"ok": True, "count": self._decisions.set(ngram, status, note)}

    def undecide_flag(self, key: str) -> dict:
        return {"ok": True, "count": self._decisions.remove(key)}

    def decisions_info(self) -> dict:
        return {"count": len(self._decisions),
                "entries": self._decisions.as_list()}

    def clear_decisions(self) -> dict:
        self._decisions.clear()
        return {"ok": True, "count": 0}

    # ---- approved term base (persistent) ----
    def termbase_info(self) -> dict:
        return {"count": len(self._termbase),
                "trash_count": len(self._termbase.trash),
                "path": self._termbase.path,
                "entries": self._termbase.as_list()}

    def approve_term(self, ngram: str, target: str) -> dict:
        """Record a resolved decision: source n-gram -> approved translation."""
        count = self._termbase.add(ngram, target)
        return {"ok": True, "count": count}

    def approve_all(self) -> dict:
        """Approve the dominant (most frequent) translation of every current
        inconsistent flag. The reviewer can prune the term base afterwards."""
        n = 0
        for f in self._flags:
            if f.distinct >= 2 and self._is_inconsistent(f):
                self._termbase.add(f.ngram, f.variants[0].span)
                n += 1
        return {"ok": True, "approved": n, "count": len(self._termbase)}

    def remove_term(self, key: str) -> dict:
        return {"ok": True, "count": self._termbase.remove(key)}

    def remove_terms(self, keys: List[str]) -> dict:
        return {"ok": True, "count": self._termbase.remove_many(keys)}

    def restore_term(self, key: str) -> dict:
        return {"ok": True, "count": self._termbase.restore(key)}

    def restore_all_terms(self) -> dict:
        return {"ok": True, "count": self._termbase.restore_all()}

    def empty_trash(self) -> dict:
        self._termbase.empty_trash()
        return {"ok": True}

    def trash_info(self) -> dict:
        return {"count": len(self._termbase.trash),
                "entries": self._termbase.trash_list()}

    def update_term(self, old_key: str, source: str, target: str) -> dict:
        """Edit an entry: rename the key if the source changed, set target."""
        if old_key and old_key != source.strip().lower():
            self._termbase.remove(old_key)
        self._termbase.add(source, target)
        return {"ok": True, "count": len(self._termbase)}

    def clear_termbase(self) -> dict:
        self._termbase.clear()
        return {"ok": True, "count": 0}

    def export_vault(self) -> dict:
        from webview import SAVE_DIALOG
        import csv
        path = self._window.create_file_dialog(
            SAVE_DIALOG, save_filename="ngram-vault.csv")
        if not path:
            return {"ok": False}
        p = path if isinstance(path, str) else path[0]
        with open(p, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["source", "target"])
            for e in self._termbase.as_list():
                w.writerow([e["source"], e["target"]])
        return {"ok": True, "path": p, "count": len(self._termbase)}

    def import_vault(self) -> dict:
        from webview import OPEN_DIALOG
        from .core import glossary as gl
        paths = self._window.create_file_dialog(
            OPEN_DIALOG, allow_multiple=False,
            file_types=("Vault (*.csv;*.json)", "All files (*.*)"))
        if not paths:
            return {"ok": False, "added": 0}
        p = paths[0] if isinstance(paths, (list, tuple)) else paths
        try:
            pairs = []
            if p.lower().endswith(".json"):
                import json
                data = json.load(open(p, encoding="utf-8"))
                entries = data.get("entries", data)
                for v in entries.values():
                    if isinstance(v, dict) and v.get("source") and v.get("target"):
                        pairs.append((v["source"], v["target"]))
            else:
                pairs = [(e.source, e.target) for e in gl.load_glossary_csv(p)]
            count = self._termbase.add_many(pairs)
        except Exception as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "added": len(pairs), "count": count}

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
        viols = gl.check_adherence(self._segments, self._glossary, **self._norm)
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
