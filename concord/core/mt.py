"""
Local MT back-translation verifier.

Back-translates each flagged Arabic variant span to English with a small local
Marian model (Helsinki-NLP/opus-mt-ar-en). If a group's variants back-translate
to similar English, they are probably acceptable synonyms/inflections; if they
diverge, the inconsistency is corroborated.

This is a heuristic confidence signal, NOT ground truth — short spans lack
context, so treat the verdict as ranking/annotation, never an auto-decision.
"""

from __future__ import annotations
from typing import List, Dict

from .textutil import norm_edit_distance

MODEL_NAME = "Helsinki-NLP/opus-mt-ar-en"

# max pairwise normalized edit distance between back-translations for a group
# to still count as "acceptable" (the variants mean the same thing).
DEFAULT_THRESHOLD = 0.34


class Translator:
    """Lazily-loaded Marian ar->en translator (loads on construction)."""

    def __init__(self, model: str = MODEL_NAME, device: str = "cpu"):
        import torch
        from transformers import MarianMTModel, MarianTokenizer
        self._torch = torch
        self._tok = MarianTokenizer.from_pretrained(model)
        self._model = MarianMTModel.from_pretrained(model).to(device).eval()
        self._device = device

    def translate(self, texts: List[str]) -> List[str]:
        if not texts:
            return []
        torch = self._torch
        enc = self._tok(texts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            gen = self._model.generate(**enc, max_length=64, num_beams=1)
        return [self._tok.decode(g, skip_special_tokens=True).strip() for g in gen]


def _norm_en(s: str) -> str:
    return " ".join(s.lower().split())


def verify_all(translator: Translator, items: List[Dict],
               threshold: float = DEFAULT_THRESHOLD) -> List[Dict]:
    """
    items: [{"ngram": str, "spans": [str, ...]}] (only groups with >=2 spans).

    Back-translates every span in one batch, then per group decides:
      acceptable   -> all back-translations agree (variants are equivalent)
      inconsistent -> back-translations diverge (genuine inconsistency)
    Returns one verdict dict per item, in order.
    """
    flat: List[str] = []
    owner: List[int] = []
    for k, it in enumerate(items):
        for s in it["spans"]:
            flat.append(s)
            owner.append(k)

    bts = translator.translate(flat)

    per = [[] for _ in items]
    for k, span, bt in zip(owner, flat, bts):
        per[k].append((span, bt))

    out: List[Dict] = []
    for it, pairs in zip(items, per):
        norm = [_norm_en(bt) for (_, bt) in pairs]
        maxd = 0.0
        for i in range(len(norm)):
            for j in range(i + 1, len(norm)):
                maxd = max(maxd, norm_edit_distance(norm[i], norm[j]))
        out.append({
            "ngram": it["ngram"],
            "verdict": "acceptable" if maxd <= threshold else "inconsistent",
            "summary": f"max divergence {round(maxd, 2)}",
            "rows": [{"span": s, "note": f"“{bt}”"} for (s, bt) in pairs],
        })
    return out
