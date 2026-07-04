"""
Consistency engine.

For every English source n-gram, finds the Arabic span that translates it in
each segment (via the aligner), then groups by n-gram and flags only when the
ALIGNED SPANS genuinely differ — not when whole targets differ.

This is the core improvement over whole-segment comparison.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional

from .textutil import (
    tokenize, ngrams_with_positions, target_span, DEFAULT_STOPWORDS,
)
from .aligner import Aligner


@dataclass
class SpanOccurrence:
    sid: str
    file: str
    unit: str
    source: str
    target: str          # full original target (for the editor)
    span: str            # normalized aligned span translating the n-gram


@dataclass
class Variant:
    span: str                         # the normalized aligned span (the key)
    occurrences: List[SpanOccurrence] = field(default_factory=list)

    @property
    def count(self):
        return len(self.occurrences)


@dataclass
class Flag:
    ngram: str
    variants: List[Variant]

    @property
    def distinct(self):
        return len(self.variants)

    @property
    def total(self):
        return sum(v.count for v in self.variants)


@dataclass
class EngineConfig:
    nmin: int = 2
    nmax: int = 3
    stop_mode: str = "trim"          # off | trim | strict
    min_occurrences: int = 2
    fold_taa: bool = True
    strip_clitics: bool = True       # fold الـ / clitics off target terms
    stopwords: Optional[set] = None


class ConsistencyEngine:
    def __init__(self, aligner: Aligner, config: EngineConfig = None):
        self.aligner = aligner
        self.cfg = config or EngineConfig()

    def analyze(
        self, segments, progress: Optional[Callable[[int, int], None]] = None
    ) -> List[Flag]:
        cfg = self.cfg
        stop = cfg.stopwords or DEFAULT_STOPWORDS

        # ngram_key -> { span -> Variant }, and remember display form
        groups: Dict[str, Dict[str, Variant]] = {}
        display: Dict[str, str] = {}

        total = len(segments)
        for i, seg in enumerate(segments):
            if progress and (i % 25 == 0 or i == total - 1):
                progress(i + 1, total)

            src_tokens = tokenize(seg.source)
            tgt_tokens = tokenize(seg.target)
            if not src_tokens or not tgt_tokens:
                continue

            ngs = ngrams_with_positions(
                src_tokens, cfg.nmin, cfg.nmax, cfg.stop_mode, stop
            )
            if not ngs:
                continue

            # one alignment per segment (cached); reused for all its n-grams
            alignment = self.aligner.align(src_tokens, tgt_tokens)

            seen_in_seg = set()
            for disp, start, length in ngs:
                key = disp.lower()
                if key in seen_in_seg:
                    continue
                seen_in_seg.add(key)
                display.setdefault(key, disp)

                span = target_span(
                    tgt_tokens, alignment, start, length,
                    normalize=True, fold_taa=cfg.fold_taa,
                    strip_clitics=cfg.strip_clitics,
                )
                if not span:
                    # alignment produced nothing for this n-gram; skip rather
                    # than guess (avoids spurious flags)
                    continue

                gv = groups.setdefault(key, {})
                var = gv.get(span)
                if var is None:
                    var = Variant(span=span)
                    gv[span] = var
                var.occurrences.append(SpanOccurrence(
                    sid=seg.sid, file=seg.file, unit=seg.unit,
                    source=seg.source, target=seg.target, span=span,
                ))

        # build flags: >=2 distinct spans AND total >= min_occurrences
        flags: List[Flag] = []
        for key, gv in groups.items():
            if len(gv) < 2:
                continue
            variants = sorted(gv.values(), key=lambda v: v.count, reverse=True)
            total_occ = sum(v.count for v in variants)
            if total_occ < cfg.min_occurrences:
                continue
            flags.append(Flag(ngram=display[key], variants=variants))

        flags.sort(key=lambda f: (f.distinct, f.total, len(f.ngram)), reverse=True)
        return flags
