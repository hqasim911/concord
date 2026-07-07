"""
Consistency engine.

Forward: for every English source n-gram, find the Arabic span that translates
it in each segment (via the aligner), group by n-gram, and flag only when the
ALIGNED SPANS genuinely differ — not when whole targets differ.

Reverse: also detect when one Arabic span is used to translate several distinct
English n-grams (an over-loaded / ambiguous target term).

Near-duplicate spans are clustered before counting distinctness, and each flag
carries an inconsistency score (distribution entropy) for ranking.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional

from .textutil import (
    tokenize, ngrams_with_positions, target_span, norm_edit_distance,
    DEFAULT_STOPWORDS,
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
    score: float = 0.0                # inconsistency score in [0, 1]
    verify: Optional[dict] = None     # local-verifier verdict (LaBSE pre-filter)
    dropped: Optional[list] = None    # variants removed as mis-aligned
    termbase_approved: Optional[str] = None   # approved translation, if any
    termbase_violation: bool = False          # deviates from the approved term
    decided: Optional[str] = None             # reviewer verdict: accepted|dismissed

    @property
    def distinct(self):
        return len(self.variants)

    @property
    def total(self):
        return sum(v.count for v in self.variants)


@dataclass
class TermUse:
    """One English n-gram that a given Arabic span was used to translate."""
    term: str
    occurrences: List[SpanOccurrence] = field(default_factory=list)

    @property
    def count(self):
        return len(self.occurrences)


@dataclass
class ReverseFlag:
    span: str                         # the Arabic span reused across terms
    uses: List[TermUse]
    score: float = 0.0

    @property
    def distinct(self):
        return len(self.uses)

    @property
    def total(self):
        return sum(u.count for u in self.uses)


@dataclass
class EngineConfig:
    nmin: int = 2
    nmax: int = 3
    stop_mode: str = "trim"          # off | trim | strict
    min_occurrences: int = 2
    fold_taa: bool = True
    strip_clitics: bool = True       # fold الـ / clitics off target terms
    cluster_spans: bool = True       # merge near-duplicate spans
    cluster_max_dist: float = 0.2    # normalized edit distance threshold
    merge_contained: bool = True     # fold partial-alignment fragment spans
    min_variant_count: int = 1       # drop variants seen fewer times (noise)
    reverse: bool = False            # also compute reverse (over-loaded) flags
    include_consistent: bool = False  # keep single-span (consistent) n-grams too
    termbase: Optional[dict] = None  # ngram_key -> approved target (persist check)
    stopwords: Optional[set] = None


def _entropy_score(counts: List[int]) -> float:
    """Normalized Shannon entropy of a count distribution, in [0, 1].
    1.0 = perfectly even split (most inconsistent); ~0 = one dominant variant."""
    total = sum(counts)
    n = len(counts)
    if total == 0 or n < 2:
        return 0.0
    ent = -sum((c / total) * math.log2(c / total) for c in counts if c)
    return ent / math.log2(n)


def _cluster_variants(variants: List[Variant], max_dist: float) -> List[Variant]:
    """Greedily merge spans whose normalized edit distance <= max_dist. The
    highest-count span in each cluster becomes its representative."""
    merged: List[Variant] = []
    for v in sorted(variants, key=lambda x: x.count, reverse=True):
        for c in merged:
            if norm_edit_distance(v.span, c.span) <= max_dist:
                c.occurrences.extend(v.occurrences)
                break
        else:
            merged.append(Variant(span=v.span, occurrences=list(v.occurrences)))
    return merged


def _is_contained(short_tokens: List[str], long_tokens: List[str]) -> bool:
    """True if short_tokens appear as a contiguous sub-sequence of long_tokens
    (and are strictly shorter). Partial alignments truncate a term to a
    contiguous fragment of the full span, so this catches them."""
    n, m = len(short_tokens), len(long_tokens)
    if n == 0 or n >= m:
        return False
    return any(long_tokens[i:i + n] == short_tokens for i in range(m - n + 1))


def _merge_contained(variants: List[Variant]) -> List[Variant]:
    """Fold a variant whose span is a contiguous fragment of a longer variant's
    span into that longer variant (a partial-alignment artifact, e.g. خط into
    خط اساس جدول زمني). The longer span stays as the representative."""
    kept: List[Variant] = []
    for v in sorted(variants, key=lambda x: len(x.span.split()), reverse=True):
        vtok = v.span.split()
        for k in kept:
            if _is_contained(vtok, k.span.split()):
                k.occurrences.extend(v.occurrences)
                break
        else:
            kept.append(v)
    return kept


class ConsistencyEngine:
    def __init__(self, aligner: Aligner, config: EngineConfig = None):
        self.aligner = aligner
        self.cfg = config or EngineConfig()

    # ---- single pass over the corpus ----
    def _collect(self, segments, progress):
        cfg = self.cfg
        stop = cfg.stopwords or DEFAULT_STOPWORDS

        # Pass 1: tokenize + extract n-grams; gather UNIQUE sentence pairs so
        # each distinct pair is aligned only once (translation memories repeat
        # heavily). Alignment is the expensive step.
        prepared = []                                 # (seg, src, tgt, ngs)
        uniq = {}                                     # (tsrc, ttgt) -> (src, tgt)
        for seg in segments:
            src_tokens = tokenize(seg.source)
            tgt_tokens = tokenize(seg.target)
            if not src_tokens or not tgt_tokens:
                continue
            ngs = ngrams_with_positions(
                src_tokens, cfg.nmin, cfg.nmax, cfg.stop_mode, stop
            )
            if not ngs:
                continue
            prepared.append((seg, src_tokens, tgt_tokens, ngs))
            uniq.setdefault((tuple(src_tokens), tuple(tgt_tokens)),
                            (src_tokens, tgt_tokens))

        # Align unique pairs in chunks (enables batching backends) w/ progress.
        pairs = list(uniq.values())
        align_map = {}
        total = len(pairs)
        for start in range(0, total, 32):
            chunk = pairs[start:start + 32]
            for (s, t), a in zip(chunk, self.aligner.align_batch(chunk)):
                align_map[(tuple(s), tuple(t))] = a
            if progress:
                progress(min(start + 32, total), total)
        if progress and total == 0:
            progress(0, 0)

        # Pass 2: build forward + reverse groups from the cached alignments.
        groups: Dict[str, Dict[str, Variant]] = {}   # ngram -> span -> Variant
        display: Dict[str, str] = {}                  # ngram key -> display
        rev: Dict[str, Dict[str, TermUse]] = {}       # span -> ngram -> TermUse
        for seg, src_tokens, tgt_tokens, ngs in prepared:
            alignment = align_map[(tuple(src_tokens), tuple(tgt_tokens))]
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
                    continue

                occ = SpanOccurrence(
                    sid=seg.sid, file=seg.file, unit=seg.unit,
                    source=seg.source, target=seg.target, span=span,
                )
                gv = groups.setdefault(key, {})
                gv.setdefault(span, Variant(span=span)).occurrences.append(occ)

                if cfg.reverse:
                    rg = rev.setdefault(span, {})
                    rg.setdefault(key, TermUse(term=disp)).occurrences.append(occ)

        return groups, display, rev

    def analyze(
        self, segments, progress: Optional[Callable[[int, int], None]] = None
    ) -> List[Flag]:
        """Forward flags: one English n-gram -> multiple Arabic spans."""
        cfg = self.cfg
        groups, display, _ = self._collect(segments, progress)

        flags: List[Flag] = []
        for key, gv in groups.items():
            raw = list(gv.values())
            # term-base check runs on the raw spans (before inconsistency
            # filters) so a single, in-file-consistent deviation still surfaces.
            approved = cfg.termbase.get(key) if cfg.termbase else None
            violation = approved is not None and any(v.span != approved
                                                     for v in raw)

            variants = raw
            if cfg.cluster_spans:
                variants = _cluster_variants(variants, cfg.cluster_max_dist)
            if cfg.merge_contained:
                variants = _merge_contained(variants)
            if cfg.min_variant_count > 1:
                variants = [v for v in variants
                            if v.count >= cfg.min_variant_count]
            if not variants:
                variants = raw
            if len(variants) < 2 and not cfg.include_consistent and not violation:
                continue
            variants.sort(key=lambda v: v.count, reverse=True)
            if sum(v.count for v in variants) < cfg.min_occurrences \
                    and not violation:
                continue
            score = _entropy_score([v.count for v in variants])
            flags.append(Flag(
                ngram=display[key], variants=variants, score=score,
                termbase_approved=approved, termbase_violation=violation))

        # term-base violations first, then in-file inconsistencies, then rest
        flags.sort(key=lambda f: (f.termbase_violation, f.distinct >= 2,
                                  f.score, f.total, f.distinct), reverse=True)
        return flags

    def analyze_reverse(
        self, segments, progress: Optional[Callable[[int, int], None]] = None
    ) -> List[ReverseFlag]:
        """Reverse flags: one Arabic span -> multiple distinct English n-grams."""
        prev = self.cfg.reverse
        self.cfg.reverse = True
        try:
            _, _, rev = self._collect(segments, progress)
        finally:
            self.cfg.reverse = prev

        flags: List[ReverseFlag] = []
        for span, uses_by_term in rev.items():
            uses = list(uses_by_term.values())
            if len(uses) < 2:
                continue
            uses.sort(key=lambda u: u.count, reverse=True)
            if sum(u.count for u in uses) < self.cfg.min_occurrences:
                continue
            score = _entropy_score([u.count for u in uses])
            flags.append(ReverseFlag(span=span, uses=uses, score=score))

        flags.sort(key=lambda f: (f.score, f.total, f.distinct), reverse=True)
        return flags
