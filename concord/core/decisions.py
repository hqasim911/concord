"""
Persistent per-flag decisions — the "silencer" for the consistency checker.

When a reviewer judges a flagged inconsistency, that judgement is recorded so
the same flag never has to be re-reviewed — in this file, a later batch, or a
different file weeks later. Two silencing verdicts:

  accepted  — the differing translations are all acceptable here; leave it.
  dismissed — the tool was wrong (e.g. a mis-alignment); a false positive.

(A third outcome, "resolved", is handled by the N-gram Vault: approving the
correct translation.) Saved to disk and auto-loaded every session.
"""

from __future__ import annotations
import os
from datetime import datetime
from typing import Optional

from .store import JsonKeyStore

DEFAULT_PATH = os.path.expanduser("~/.concord/decisions.json")

STATUSES = {"accepted", "dismissed"}


class Decisions(JsonKeyStore):
    def __init__(self, path: str = DEFAULT_PATH):
        super().__init__(path)   # entries: key -> {ngram, status, note, updated}

    def set(self, ngram: str, status: str, note: str = "") -> int:
        if status not in STATUSES:
            return len(self.entries)
        self.entries[self._key(ngram)] = {
            "ngram": ngram, "status": status, "note": note,
            "updated": datetime.now().isoformat(timespec="seconds"),
        }
        self.save()
        return len(self.entries)

    def status_of(self, ngram: str) -> Optional[str]:
        e = self.entries.get(self._key(ngram))
        return e["status"] if e else None
