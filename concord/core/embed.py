"""
LaBSE embedding verifier (alternative to MT back-translation).

Embeds the English n-gram and each Arabic variant span with LaBSE
(language-agnostic BERT sentence embeddings) and compares them by cosine
similarity — no translation step, so it degrades less on short terms.

Verdict: if the variants are mutually similar (min pairwise cosine >=
threshold) they are probably equivalent (acceptable); a low-similarity
variant corroborates a genuine inconsistency. Heuristic, advisory only.

Uses setu4993/LaBSE via plain transformers (sentence embedding = the L2-
normalized pooler_output), so no sentence-transformers dependency.
"""

from __future__ import annotations
from typing import List, Dict

MODEL_NAME = "setu4993/LaBSE"
DEFAULT_THRESHOLD = 0.7   # min pairwise variant cosine to count as "acceptable"


class Embedder:
    def __init__(self, model: str = MODEL_NAME, device: str = "cpu"):
        import torch
        from transformers import BertModel, BertTokenizerFast
        self._torch = torch
        self._tok = BertTokenizerFast.from_pretrained(model)
        self._model = BertModel.from_pretrained(model).to(device).eval()
        self._device = device

    def embed(self, texts: List[str]):
        """Return an [n, d] tensor of L2-normalized LaBSE embeddings."""
        torch = self._torch
        enc = self._tok(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=64)
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            out = self._model(**enc)
        return torch.nn.functional.normalize(out.pooler_output, p=2, dim=1)


def verify_all(embedder: Embedder, items: List[Dict],
               threshold: float = DEFAULT_THRESHOLD) -> List[Dict]:
    """items: [{"ngram": str, "spans": [str, ...]}] (groups with >=2 spans).
    Returns one verdict dict per item, in order (unified shape shared with
    the MT verifier: ngram, verdict, summary, rows)."""
    torch = embedder._torch
    texts: List[str] = []
    slices = []
    for it in items:
        start = len(texts)
        texts.append(it["ngram"])
        texts.extend(it["spans"])
        slices.append((start, len(it["spans"])))

    emb = embedder.embed(texts) if texts else None

    out: List[Dict] = []
    for it, (start, ns) in zip(items, slices):
        term = emb[start]
        vecs = emb[start + 1:start + 1 + ns]
        pair = [float(torch.dot(vecs[i], vecs[j]))
                for i in range(ns) for j in range(i + 1, ns)]
        min_sim = min(pair) if pair else 1.0
        term_sim = [float(torch.dot(vecs[i], term)) for i in range(ns)]
        out.append({
            "ngram": it["ngram"],
            "verdict": "acceptable" if min_sim >= threshold else "inconsistent",
            "summary": f"min variant sim {round(min_sim, 2)}",
            "rows": [{"span": s, "note": f"sim to term {round(ts, 2)}"}
                     for s, ts in zip(it["spans"], term_sim)],
        })
    return out
