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
        self.trash: Dict[str, dict] = {}     # soft-deleted entries (recycle bin)

    def load(self) -> "TermBase":
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                self.entries = data.get("entries", data)
                self.trash = data.get("trash", {})
        except (OSError, ValueError):
            self.entries, self.trash = {}, {}
        return self

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"entries": self.entries, "trash": self.trash}, fh,
                      ensure_ascii=False, indent=2)

    def add(self, source: str, target: str) -> int:
        """Record source -> approved target (overwrites any prior decision)."""
        source, target = source.strip(), target.strip()
        if not source or not target:
            return len(self.entries)
        k = _key(source)
        self.entries[k] = {
            "source": source, "target": target,
            "updated": datetime.now().isoformat(timespec="seconds"),
        }
        self.trash.pop(k, None)
        self.save()
        return len(self.entries)

    def add_many(self, pairs) -> int:
        """Add/overwrite many (source, target) pairs, saving once."""
        now = datetime.now().isoformat(timespec="seconds")
        for source, target in pairs:
            source, target = source.strip(), target.strip()
            if source and target:
                k = _key(source)
                self.entries[k] = {
                    "source": source, "target": target, "updated": now}
                self.trash.pop(k, None)
        self.save()
        return len(self.entries)

    # ---- soft delete / recycle bin ----
    def remove(self, key: str) -> int:
        k = _key(key)
        e = self.entries.pop(k, None)
        if e is not None:
            self.trash[k] = e
            self.save()
        return len(self.entries)

    def remove_many(self, keys) -> int:
        moved = False
        for key in keys:
            k = _key(key)
            e = self.entries.pop(k, None)
            if e is not None:
                self.trash[k] = e
                moved = True
        if moved:
            self.save()
        return len(self.entries)

    def clear(self):
        self.trash.update(self.entries)
        self.entries = {}
        self.save()

    def restore(self, key: str) -> int:
        k = _key(key)
        e = self.trash.pop(k, None)
        if e is not None:
            self.entries[k] = e
            self.save()
        return len(self.entries)

    def restore_all(self) -> int:
        self.entries.update(self.trash)
        self.trash = {}
        self.save()
        return len(self.entries)

    def empty_trash(self):
        self.trash = {}
        self.save()

    def trash_list(self) -> List[dict]:
        return [{"key": k, **v} for k, v in sorted(self.trash.items())]

    def check_map(self) -> Dict[str, str]:
        """{ngram_key -> approved target} for the engine's term-base check."""
        return {k: v["target"] for k, v in self.entries.items()}

    def as_list(self) -> List[dict]:
        return [{"key": k, **v} for k, v in sorted(self.entries.items())]

    def __len__(self):
        return len(self.entries)
