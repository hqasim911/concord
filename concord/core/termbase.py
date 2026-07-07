"""
Persistent approved-term base.

Turns resolved inconsistencies into a durable record of decisions: each entry
maps a source n-gram to its approved target translation. Saved to disk and
auto-loaded every session, so a term standardized once stays standardized
across files and over time — even when files are processed in separate runs.

At analysis time it is checked as an option: any n-gram whose translation in
the new file deviates from its approved entry is flagged, even when that file
is internally consistent.
"""

from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Dict, List

DEFAULT_PATH = os.path.expanduser("~/.concord/termbase.json")


def _key(source: str) -> str:
    return " ".join(source.lower().split())


class TermBase:
    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        self.entries: Dict[str, dict] = {}   # key -> {source, target, updated}

    def load(self) -> "TermBase":
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                self.entries = data.get("entries", data)
        except (OSError, ValueError):
            self.entries = {}
        return self

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"entries": self.entries}, fh,
                      ensure_ascii=False, indent=2)

    def add(self, source: str, target: str) -> int:
        """Record source -> approved target (overwrites any prior decision)."""
        source, target = source.strip(), target.strip()
        if not source or not target:
            return len(self.entries)
        self.entries[_key(source)] = {
            "source": source, "target": target,
            "updated": datetime.now().isoformat(timespec="seconds"),
        }
        self.save()
        return len(self.entries)

    def add_many(self, pairs) -> int:
        """Add/overwrite many (source, target) pairs, saving once."""
        now = datetime.now().isoformat(timespec="seconds")
        for source, target in pairs:
            source, target = source.strip(), target.strip()
            if source and target:
                self.entries[_key(source)] = {
                    "source": source, "target": target, "updated": now}
        self.save()
        return len(self.entries)

    def remove(self, key: str) -> int:
        self.entries.pop(_key(key), None)
        self.save()
        return len(self.entries)

    def clear(self):
        self.entries = {}
        self.save()

    def check_map(self) -> Dict[str, str]:
        """{ngram_key -> approved target} for the engine's term-base check."""
        return {k: v["target"] for k, v in self.entries.items()}

    def as_list(self) -> List[dict]:
        return [{"key": k, **v} for k, v in sorted(self.entries.items())]

    def __len__(self):
        return len(self.entries)
