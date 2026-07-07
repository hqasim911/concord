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
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

DEFAULT_PATH = os.path.expanduser("~/.concord/decisions.json")

STATUSES = {"accepted", "dismissed"}


def _key(ngram: str) -> str:
    return " ".join(ngram.lower().split())


class Decisions:
    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        self.entries: Dict[str, dict] = {}   # key -> {ngram, status, note, updated}

    def load(self) -> "Decisions":
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.entries = data.get("entries", {}) if isinstance(data, dict) else {}
        except (OSError, ValueError):
            self.entries = {}
        return self

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"entries": self.entries}, fh, ensure_ascii=False, indent=2)

    def set(self, ngram: str, status: str, note: str = "") -> int:
        if status not in STATUSES:
            return len(self.entries)
        self.entries[_key(ngram)] = {
            "ngram": ngram, "status": status, "note": note,
            "updated": datetime.now().isoformat(timespec="seconds"),
        }
        self.save()
        return len(self.entries)

    def remove(self, key: str) -> int:
        self.entries.pop(_key(key), None)
        self.save()
        return len(self.entries)

    def clear(self):
        self.entries = {}
        self.save()

    def status_of(self, ngram: str) -> Optional[str]:
        e = self.entries.get(_key(ngram))
        return e["status"] if e else None

    def as_list(self) -> List[dict]:
        return [{"key": k, **v} for k, v in sorted(self.entries.items())]

    def __len__(self):
        return len(self.entries)
