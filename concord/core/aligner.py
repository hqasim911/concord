"""
Word-alignment layer.

Defines a swappable Aligner interface so the consistency engine never depends
on a specific alignment backend. Implementations:

  - SimAlignAligner    : neural aligner via SimAlign (mBERT/XLM-R), local.
  - AwesomeAlignAligner : awesome-align extraction method (Dou & Neubig 2021).
  - MockAligner        : hand-fed alignments, for tests / offline demos.

The engine asks each aligner for token-level (src_idx, tgt_idx) pairs.
"""

from __future__ import annotations
import hashlib
import itertools
from typing import List, Tuple, Dict, Optional


Alignment = List[Tuple[int, int]]


class Aligner:
    """Abstract interface."""
    name = "base"

    def align(self, src_tokens: List[str], tgt_tokens: List[str]) -> Alignment:
        raise NotImplementedError

    def align_batch(self, pairs) -> List[Alignment]:
        """Align many (src_tokens, tgt_tokens) pairs. Default is sequential;
        backends that support true batching or caching may override."""
        return [self.align(s, t) for (s, t) in pairs]


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


class AwesomeAlignAligner(Aligner):
    """
    Word aligner using the awesome-align extraction method (Dou & Neubig,
    2021), implemented directly on top of a multilingual Transformer.

    Instead of depending on the awesome-align package/CLI, this reproduces
    its inference algorithm: take contextual sub-word embeddings from an
    aligned hidden layer, build the src<->tgt similarity matrix, apply a
    softmax in both directions, keep cells above threshold in BOTH (the
    "intersection"), then map sub-word cells back up to word indices.

    model: an HF model id, or the shorthands 'bert' / 'xlmr' (mapped in
      build_aligner). Point it at a fine-tuned awesome-align checkpoint for
      the accuracy gains reported in the paper; the base multilingual model
      is roughly on par with SimAlign.
    """
    name = "awesome"

    def __init__(self, model: str = "bert-base-multilingual-cased",
                 device: str = "cpu", align_layer: int = 8,
                 threshold: float = 1e-3):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModel.from_pretrained(model).to(device).eval()
        self._device = device
        self._align_layer = align_layer
        self._threshold = threshold

    def align(self, src_tokens, tgt_tokens) -> Alignment:
        if not src_tokens or not tgt_tokens:
            return []
        torch = self._torch
        tok = self._tokenizer

        # sub-word tokenize each word, tracking which word each piece is from
        sub_src = [tok.tokenize(w) for w in src_tokens]
        sub_tgt = [tok.tokenize(w) for w in tgt_tokens]
        flat_src = list(itertools.chain.from_iterable(
            tok.convert_tokens_to_ids(s) for s in sub_src))
        flat_tgt = list(itertools.chain.from_iterable(
            tok.convert_tokens_to_ids(s) for s in sub_tgt))
        if not flat_src or not flat_tgt:
            return []

        enc_src = tok.prepare_for_model(
            flat_src, return_tensors="pt", truncation=True,
            max_length=tok.model_max_length)["input_ids"]
        enc_tgt = tok.prepare_for_model(
            flat_tgt, return_tensors="pt", truncation=True,
            max_length=tok.model_max_length)["input_ids"]

        # sub-word index -> word index (parallel to the flat id lists above)
        sub2word_src = [i for i, s in enumerate(sub_src) for _ in s]
        sub2word_tgt = [i for i, s in enumerate(sub_tgt) for _ in s]

        with torch.no_grad():
            # [1:-1] drops the [CLS]/[SEP] special tokens
            hs_src = self._model(
                enc_src.unsqueeze(0).to(self._device),
                output_hidden_states=True,
            ).hidden_states[self._align_layer][0, 1:-1]
            hs_tgt = self._model(
                enc_tgt.unsqueeze(0).to(self._device),
                output_hidden_states=True,
            ).hidden_states[self._align_layer][0, 1:-1]

            sim = torch.matmul(hs_src, hs_tgt.transpose(-1, -2))
            fwd = torch.nn.functional.softmax(sim, dim=-1)
            bwd = torch.nn.functional.softmax(sim, dim=-2)
            inter = (fwd > self._threshold) * (bwd > self._threshold)

        pairs = set()
        for idx in torch.nonzero(inter, as_tuple=False):
            i, j = int(idx[0]), int(idx[1])
            pairs.add((sub2word_src[i], sub2word_tgt[j]))
        return sorted(pairs)


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

    def align_batch(self, pairs) -> List[Alignment]:
        """Compute each distinct uncached pair exactly once (dedups within the
        batch AND against the cache), then cache the results."""
        results: List[Alignment] = [None] * len(pairs)
        miss_pair = {}                 # key -> (s, t)
        miss_pos = {}                  # key -> [result indices]
        for i, (s, t) in enumerate(pairs):
            k = self._key(s, t)
            cached = self._cache.get(k)
            if cached is not None:
                results[i] = cached
            else:
                miss_pair.setdefault(k, (s, t))
                miss_pos.setdefault(k, []).append(i)
        if miss_pair:
            keys = list(miss_pair)
            computed = self._inner.align_batch([miss_pair[k] for k in keys])
            for k, a in zip(keys, computed):
                self._cache[k] = a
                for i in miss_pos[k]:
                    results[i] = a
        return results

    @property
    def inner_name(self) -> str:
        return self._inner.name


class EnsembleAligner(Aligner):
    """
    Combine several aligners. mode='intersect' keeps only links all backends
    agree on (higher precision — the awesome-align ∩ SimAlign recipe);
    mode='union' keeps any link (higher recall).
    """
    name = "ensemble"

    def __init__(self, aligners: List[Aligner], mode: str = "intersect"):
        if not aligners:
            raise ValueError("EnsembleAligner needs at least one aligner")
        self._aligners = aligners
        self._mode = mode

    def align(self, src_tokens, tgt_tokens) -> Alignment:
        sets = [set(a.align(src_tokens, tgt_tokens)) for a in self._aligners]
        if self._mode == "union":
            out = set().union(*sets)
        else:
            out = sets[0].intersection(*sets[1:])
        return sorted(out)


# Shorthands so the same UI bert/xlmr toggle drives every backend.
_AWESOME_MODEL_ALIASES = {
    "bert": "bert-base-multilingual-cased",
    "xlmr": "xlm-roberta-base",
}


def build_aligner(kind: str = "simalign", **kwargs) -> Aligner:
    """Factory. kind in {'simalign','awesome','mock'}; wrapped in caching."""
    if kind == "simalign":
        base = SimAlignAligner(**kwargs)
    elif kind == "awesome":
        m = kwargs.get("model", "bert")
        kwargs["model"] = _AWESOME_MODEL_ALIASES.get(m, m)
        base = AwesomeAlignAligner(**kwargs)
    elif kind == "ensemble":
        m = kwargs.get("model", "bert")
        mode = kwargs.get("mode", "intersect")
        base = EnsembleAligner([
            SimAlignAligner(model=m),
            AwesomeAlignAligner(model=_AWESOME_MODEL_ALIASES.get(m, m)),
        ], mode=mode)
    elif kind == "mock":
        base = MockAligner(kwargs.get("table"))
    else:
        raise ValueError(f"Unknown aligner kind: {kind}")
    return CachingAligner(base)
