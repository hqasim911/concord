"""
Shared verdict schema for the local verifiers (LaBSE embeddings + MT
back-translation).

Both verifiers score a flagged group's variants for mutual agreement and
return one dict per group in the SAME shape, so the UI and the LaBSE
pre-filter can consume either interchangeably:

    {
      "ngram":     str,              # the English source n-gram
      "verdict":   "duplicate" | "distinct",
      "agreement": float,            # in [0, 1]; higher = variants agree more
      "summary":   str,              # human-readable one-liner
      "rows":      [{"span": str, "note": str}, ...],
    }

'duplicate' = the variants are essentially the same rendering (agreement at or
above the near-exact threshold) → safe to treat as one. 'distinct' = genuinely
different translations → stay flagged, even when they are valid synonyms.
"""

from __future__ import annotations
from typing import List, Dict


def build_verdict(ngram: str, agreement: float, threshold: float,
                  summary: str, rows: List[Dict]) -> Dict:
    """Assemble a verdict dict, deriving duplicate/distinct from the threshold."""
    return {
        "ngram": ngram,
        "verdict": "duplicate" if agreement >= threshold else "distinct",
        "agreement": round(agreement, 3),
        "summary": summary,
        "rows": rows,
    }
