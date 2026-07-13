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
_AR_TATWEEL = re.compile("\u0640")
_AR_DIACRITICS = re.compile("[\u064B-\u0652]")


@functools.lru_cache(maxsize=50000)
def normalize_ar(text: str, fold_taa: bool = True,
                 strip_diacritics: bool = True) -> str:
    """Strip tatweel, unify alef/hamza forms; optionally taa->haa and diacritics.

    strip_diacritics: remove the tashkeel marks (fatha/damma/kasra/shadda/...)
    so a vowelled spelling and its bare form compare equal and don't false-flag.
    """
    text = _AR_TATWEEL.sub("", text)
    if strip_diacritics:
        text = _AR_DIACRITICS.sub("", text)
    text = re.sub("[أآإ]", "ا", text)
    text = re.sub("[ؤ]", "و", text)
    text = re.sub("[ئ]", "ى", text)
    if fold_taa:
        text = re.sub("ة", "ه", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Light stemming (collapse morphological variants of the SAME term)
# ---------------------------------------------------------------------------
# Definite article "ال" plus optional leading preposition/conjunction
# proclitics, longest-first. Deliberately conservative: only ARTICLE-BEARING
# prefixes are stripped, never a bare و/ب/ك/ل, because many ordinary words
# legitimately start with those letters (باب, كتاب, ولد). This keeps
# precision high — it merges الزر / بالزر / وبالزر -> زر without merging
# unrelated words.
_AR_ARTICLE_CLITICS = (
    "وبال", "فبال", "وكال", "فكال",
    "وال", "فال", "بال", "كال", "ولل", "فلل",
    "لل", "ال",
)


@functools.lru_cache(maxsize=50000)
def light_stem_ar(word: str) -> str:
    """
    Strip a leading definite-article clitic so morphological variants of a
    term compare equal (الزر / بالزر / وبالزر -> زر).

    Length-guarded to keep >= 2 stem characters, so short words are not
    over-stemmed (e.g. الف "thousand" -> "ف" would leave 1 char, so it is
    left untouched).
    """
    for p in _AR_ARTICLE_CLITICS:
        if word.startswith(p) and len(word) - len(p) >= 2:
            return word[len(p):]
    return word


# ---------------------------------------------------------------------------
# Key normalization (source n-gram -> lookup key)
# ---------------------------------------------------------------------------
def normalize_key(text: str) -> str:
    """Canonical lookup key for a source n-gram: lowercase, whitespace-collapsed.

    Shared by the term base and the decisions store, and MUST match the JS
    mirror `jsKey` in ui/app.js — decision/vault keys are matched across the
    JS↔Python bridge, so the two implementations have to agree exactly.
    """
    return " ".join(text.lower().split())


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
# Max gap (in tokens) tolerated between kept indices before an index counts
# as an outlier. 1 lets a single unaligned particle sit inside a span while
# still discarding far-flung spurious links.
_SPAN_MAX_GAP = 1


def trim_span_outliers(tgt_idx, max_gap: int = _SPAN_MAX_GAP):
    """
    From sorted unique target indices aligned to an n-gram, return the
    (lo, hi) bounds of the densest cluster: the run holding the most aligned
    indices where consecutive kept indices are at most max_gap tokens apart.

    A raw min..max hull balloons whenever a single source word mis-aligns to
    a distant target token (e.g. indices {2,3,15} -> a 14-token span). Keeping
    only the dominant cluster drops that outlier and returns {2,3}.
    """
    runs = []                       # (lo, hi, count)
    lo = prev = tgt_idx[0]
    count = 1
    for idx in tgt_idx[1:]:
        if idx - prev <= max_gap + 1:
            prev, count = idx, count + 1
        else:
            runs.append((lo, prev, count))
            lo = prev = idx
            count = 1
    runs.append((lo, prev, count))
    # most indices wins; tie-break: narrower span, then earlier position
    best = max(runs, key=lambda r: (r[2], -(r[1] - r[0]), -r[0]))
    return best[0], best[1]


def target_span(
    tgt_tokens: List[str], alignments, ng_start: int, ng_len: int,
    normalize: bool = True, fold_taa: bool = True,
    strip_clitics: bool = True, strip_diacritics: bool = True,
) -> str:
    """
    Given the n-gram's source token range and the alignment, return the
    Arabic span that translates it — the densest cluster of aligned target
    tokens (outlier links trimmed; see trim_span_outliers).

    strip_clitics: when normalizing, also fold the definite article/clitic
    off each word so morphological variants of the same term are compared
    as equal (see light_stem_ar).
    """
    return aligned_span(tgt_tokens, alignments, ng_start, ng_len,
                        normalize, fold_taa, strip_clitics,
                        strip_diacritics)["span"]


def aligned_span(
    tgt_tokens: List[str], alignments, ng_start: int, ng_len: int,
    normalize: bool = True, fold_taa: bool = True, strip_clitics: bool = True,
    strip_diacritics: bool = True,
) -> dict:
    """
    Like target_span, but returns {span, lo, hi, raw}:
      span — the normalized comparison span (what flags are grouped by)
      lo/hi — the target token range, so a correction can be spliced back
              into the exact place in the segment (not replace the whole target)
      raw — the un-normalized visible span text (readable term to insert)
    """
    ng_src = set(range(ng_start, ng_start + ng_len))
    tgt_idx = sorted({ti for (si, ti) in alignments if si in ng_src})
    if not tgt_idx:
        return {"span": "", "lo": None, "hi": None, "raw": ""}
    lo, hi = trim_span_outliers(tgt_idx)
    raw = strip_edge_punct_span(" ".join(tgt_tokens[lo:hi + 1]))
    span = raw
    if normalize:
        span = normalize_ar(span, fold_taa, strip_diacritics)
        if strip_clitics:
            span = " ".join(light_stem_ar(w) for w in span.split())
    return {"span": span, "lo": lo, "hi": hi, "raw": raw}


def strip_edge_punct_span(span: str) -> str:
    toks = span.split()
    toks = [strip_edge_punct(t) for t in toks]
    toks = [t for t in toks if t]
    return " ".join(toks)


# ---------------------------------------------------------------------------
# String distance (for clustering near-duplicate spans)
# ---------------------------------------------------------------------------
def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (iterative, O(len(a)*len(b)) time, O(len(b)) space)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,          # deletion
                cur[j - 1] + 1,       # insertion
                prev[j - 1] + (ca != cb),  # substitution
            ))
        prev = cur
    return prev[-1]


def norm_edit_distance(a: str, b: str) -> float:
    """Edit distance scaled to [0, 1] by the longer string length."""
    m = max(len(a), len(b))
    return 0.0 if m == 0 else edit_distance(a, b) / m
