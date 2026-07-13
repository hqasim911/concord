"""
Shared JSON-backed key/value store.

Both the approved term base and the per-flag decisions store are a dict of
`key -> entry`, persisted as `{"entries": {...}}` JSON under ~/.concord and
auto-loaded each session. This base captures that skeleton (load/save, hard
remove, clear, listing) so the two stores only add what differs — decisions
add a status verdict; the term base adds a soft-delete recycle bin.
"""

from __future__ import annotations
import json
import os
from typing import Dict, List

from .textutil import normalize_key


class JsonKeyStore:
    def __init__(self, path: str):
        self.path = path
        self.entries: Dict[str, dict] = {}   # key -> entry dict

    @staticmethod
    def _key(text: str) -> str:
        return normalize_key(text)

    # ---- persistence (subclass hooks: _load_data / _reset / _extra_state) ----
    def _extra_state(self) -> dict:
        """Extra top-level sections to persist alongside 'entries'."""
        return {}

    def _load_data(self, data: dict):
        """Populate state from a parsed JSON dict. Override to read extras."""
        self.entries = data.get("entries", {})

    def _reset(self):
        self.entries = {}

    def load(self):
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                self._load_data(data)
            else:
                self._reset()
        except (OSError, ValueError):
            self._reset()
        return self

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"entries": self.entries, **self._extra_state()}, fh,
                      ensure_ascii=False, indent=2)

    # ---- common operations ----
    def remove(self, key: str) -> int:
        self.entries.pop(self._key(key), None)
        self.save()
        return len(self.entries)

    def clear(self):
        self.entries = {}
        self.save()

    def as_list(self) -> List[dict]:
        return [{"key": k, **v} for k, v in sorted(self.entries.items())]

    def __len__(self):
        return len(self.entries)
