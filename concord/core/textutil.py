"""
Text utilities: tokenization, Arabic normalization, n-gram extraction.

Arabic normalization is ported from the Term-Extractor nlp_lib; the n-gram /
stopword logic mirrors Concord's browser engine (off / trim / strict modes)
so behavior is consistent between the two tools.
"""

from __future__ import annotations
import re
import functools
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Arabic normalization
# ---------------------------------------------------------------------------
_AR_DIACRITICS_TATWEEL = re.compile("[\u064B-\u0652\u0640]")

@functools.lru_cache(maxsize=50000)
def normalize_ar(text: str, fold_taa: bool = True) -> str:
    """Strip tatweel + diacritics, unify alef/hamza forms; optionally taa->haa."""
    text = _AR_DIACRITICS_TATWEEL.sub("", text)
    text = re.sub("[أآإ]", "ا", text)
    text = re.sub("[ؤ]", "و", text)
    text = re.sub("[ئ]", "ى", text)
    if fold_taa:
        text = re.sub("ة", "ه", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[^\s]+")

def tokenize(text: str) -> List[str]:
    """Whitespace tokenization preserving token order/index (alignment needs indices)."""
    return _WORD_RE.findall(text.strip())

def strip_edge_punct(tok: str) -> str:
    return re.sub(r"^[^\w\u0600-\u06FF]+|[^\w\u0600-\u06FF]+$", "", tok)


# ---------------------------------------------------------------------------
# Stopwords (English source side) — same default list as the browser app
# ---------------------------------------------------------------------------
DEFAULT_STOPWORDS = set((
    "a an the this that these those of to in on at by for with from into "
    "and or but if then else as is are was were be been being am do does did has have had "
    "will would shall should can could may might must it its he she they them his her "
    "their your our my me you we i us not no nor so than too very just also each every any "
    "all some such only own same up out off over under again further about against between"
).split())


# ---------------------------------------------------------------------------
# N-gram extraction (post-trim length enforced, like Concord's fixed engine)
# ---------------------------------------------------------------------------
def ngrams_with_positions(
    tokens: List[str], nmin: int, nmax: int,
    stop_mode: str = "trim", stopwords=None,
) -> List[Tuple[str, int, int]]:
    """
    Returns list of (ngram_text, start_index, length) where length is the FINAL
    (post-trim) token count, constrained to [nmin, nmax].

    stop_mode: 'off' | 'trim' | 'strict'
      off    -> keep n-gram as-is
      trim   -> shave leading/trailing stopwords; window widened so trimmed
                result can still land in range
      strict -> drop any n-gram containing a stopword anywhere
    """
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS
    # clean tokens for matching but keep original index mapping
    words = [strip_edge_punct(t) for t in tokens]
    out: List[Tuple[str, int, int]] = []
    seen = set()
    pad = 2 if stop_mode == "trim" else 0
    scan_max = min(len(words), nmax + pad)

    for n in range(nmin, scan_max + 1):
        for i in range(0, len(words) - n + 1):
            sl = words[i:i + n]
            # skip empties produced by punctuation stripping
            if any(w == "" for w in sl):
                continue
            start = i
            length = n
            if stop_mode == "strict":
                if any(w.lower() in stopwords for w in sl):
                    continue
            elif stop_mode == "trim":
                lo, hi = 0, len(sl)
                while lo < hi and sl[lo].lower() in stopwords:
                    lo += 1
                while hi > lo and sl[hi - 1].lower() in stopwords:
                    hi -= 1
                if hi - lo == 0:
                    continue
                start = i + lo
                sl = sl[lo:hi]
                length = len(sl)
            # enforce FINAL length within requested range
            if length < nmin or length > nmax:
                continue
            disp = " ".join(sl)
            key = disp.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append((disp, start, length))
    return out


# ---------------------------------------------------------------------------
# Target span extraction from alignment
# ---------------------------------------------------------------------------
def target_span(
    tgt_tokens: List[str], alignments, ng_start: int, ng_len: int,
    normalize: bool = True, fold_taa: bool = True,
) -> str:
    """
    Given the n-gram's source token range and the alignment, return the
    contiguous Arabic span (min..max aligned target index) that translates it.
    """
    ng_src = set(range(ng_start, ng_start + ng_len))
    tgt_idx = sorted({ti for (si, ti) in alignments if si in ng_src})
    if not tgt_idx:
        return ""
    lo, hi = tgt_idx[0], tgt_idx[-1]
    span = " ".join(tgt_tokens[lo:hi + 1])
    span = strip_edge_punct_span(span)
    return normalize_ar(span, fold_taa) if normalize else span


def strip_edge_punct_span(span: str) -> str:
    toks = span.split()
    toks = [strip_edge_punct(t) for t in toks]
    toks = [t for t in toks if t]
    return " ".join(toks)
