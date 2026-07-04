"""
Word-alignment layer.

Defines a swappable Aligner interface so the consistency engine never depends
on a specific alignment backend. Implementations:

  - SimAlignAligner : real neural aligner (mBERT/XLM-R), local, accurate.
  - MockAligner     : hand-fed alignments, for tests / offline demos.

The engine asks each aligner for token-level (src_idx, tgt_idx) pairs.
"""

from __future__ import annotations
import functools
import hashlib
from typing import List, Tuple, Dict, Optional


Alignment = List[Tuple[int, int]]


class Aligner:
    """Abstract interface."""
    name = "base"

    def align(self, src_tokens: List[str], tgt_tokens: List[str]) -> Alignment:
        raise NotImplementedError


class SimAlignAligner(Aligner):
    """
    Neural word aligner via SimAlign (https://github.com/cisnlp/simalign).

    model: 'bert' (mBERT, ~700MB) or 'xlmr' (XLM-R, larger, sometimes better AR).
    matching_method codes: m=mwmf, a=inter(argmax), i=itermax.
    'mai' computes all three; we prefer 'itermax' then 'inter'.
    """
    name = "simalign"

    def __init__(self, model: str = "bert", matching_method: str = "mai",
                 device: str = "cpu"):
        from simalign import SentenceAligner
        self._aligner = SentenceAligner(
            model=model, token_type="bpe",
            matching_methods=matching_method, device=device,
        )
        self._pref = ["itermax", "inter", "mwmf"]

    def align(self, src_tokens, tgt_tokens) -> Alignment:
        if not src_tokens or not tgt_tokens:
            return []
        res: Dict[str, Alignment] = self._aligner.get_word_aligns(src_tokens, tgt_tokens)
        for k in self._pref:
            if k in res:
                return res[k]
        # fallback: whatever key exists
        return next(iter(res.values()), [])


class MockAligner(Aligner):
    """Lookup table aligner for tests / offline demos."""
    name = "mock"

    def __init__(self, table: Optional[Dict] = None):
        self._table = table or {}

    def add(self, src_tokens, tgt_tokens, alignment: Alignment):
        self._table[(tuple(src_tokens), tuple(tgt_tokens))] = alignment

    def align(self, src_tokens, tgt_tokens) -> Alignment:
        return self._table.get((tuple(src_tokens), tuple(tgt_tokens)), [])


class CachingAligner(Aligner):
    """
    Wraps any aligner with an in-memory cache keyed by the sentence pair.
    Neural alignment is expensive; the same segment is aligned once per n-gram,
    so caching by (src,tgt) avoids recomputing the same alignment repeatedly.
    """
    name = "caching"

    def __init__(self, inner: Aligner):
        self._inner = inner
        self._cache: Dict[str, Alignment] = {}

    @staticmethod
    def _key(src_tokens, tgt_tokens) -> str:
        h = hashlib.md5()
        h.update(("\u0001".join(src_tokens) + "\u0002" + "\u0001".join(tgt_tokens)).encode("utf-8"))
        return h.hexdigest()

    def align(self, src_tokens, tgt_tokens) -> Alignment:
        k = self._key(src_tokens, tgt_tokens)
        cached = self._cache.get(k)
        if cached is not None:
            return cached
        result = self._inner.align(src_tokens, tgt_tokens)
        self._cache[k] = result
        return result

    @property
    def inner_name(self) -> str:
        return self._inner.name


def build_aligner(kind: str = "simalign", **kwargs) -> Aligner:
    """Factory. kind in {'simalign','mock'}; always wrapped in caching."""
    if kind == "simalign":
        base = SimAlignAligner(**kwargs)
    elif kind == "mock":
        base = MockAligner(kwargs.get("table"))
    else:
        raise ValueError(f"Unknown aligner kind: {kind}")
    return CachingAligner(base)
