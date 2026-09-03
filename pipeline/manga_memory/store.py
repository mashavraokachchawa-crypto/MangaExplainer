"""JSON-file backed store for Manga Memory Engine records.

Layout::

    <state_dir>/manga_memory/
        characters.json   -> {"records": [MemoryRecord.dict, ...], "version": 1}
        world.json
        story.json
        user_corrections.json

All writes are atomic (tmp file + os.replace) so a crash mid-write never
corrupts a memory file. Every read is guarded to degrade to an empty store
rather than raise into the pipeline.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from .models import MemoryRecord

LOG = logging.getLogger("mangaexplainer.manga_memory")

DIRNAME = "manga_memory"
VERSION = 1

FILE_BY_KIND = {
    "character": "characters.json",
    "world": "world.json",
    "story": "story.json",
    "correction": "user_corrections.json",
    "book": "book.json",
}


class MemoryStoreError(Exception):
    pass


def memory_dir(cfg) -> Path:
    """Absolute directory holding manga_memory json files (created on demand)."""
    try:
        state_dir = Path(cfg.pipeline.state.dir)
    except (AttributeError, TypeError):
        state_dir = Path("state")
    return state_dir / DIRNAME


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _read_records(path: Path) -> list[dict]:
    try:
        if not path.is_file():
            return []
        doc = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        LOG.warning("ignoring unreadable memory file %s", path)
        return []
    records = doc.get("records") if isinstance(doc, dict) else None
    return records if isinstance(records, list) else []


class MemoryStore:
    """Load/merge/persist a category of memory records from one JSON file."""

    def __init__(self, kind: str, path: Path):
        if kind not in FILE_BY_KIND:
            raise MemoryStoreError(f"unknown memory kind {kind!r}")
        self.kind = kind
        self.path = path
        self.records: dict[str, MemoryRecord] = {}

    # ----------------------------------------------------------------- load
    def load(self) -> "MemoryStore":
        self.records = {}
        for raw in _read_records(self.path):
            try:
                rec = MemoryRecord.from_dict(raw)
            except (TypeError, ValueError):
                continue
            if rec.kind and rec.key:
                self.records[rec.key] = rec
        return self

    def save(self) -> None:
        payload = {
            "version": VERSION,
            "kind": self.kind,
            "records": [rec.to_dict() for rec in self.records.values()],
        }
        _atomic_write(self.path, payload)

    # ---------------------------------------------------------------- access
    def get(self, key: str) -> MemoryRecord | None:
        return self.records.get(key)

    def items(self):
        return self.records.items()

    def keys(self):
        return list(self.records.keys())

    def count(self) -> int:
        return len(self.records)

    def all(self) -> list[MemoryRecord]:
        return list(self.records.values())

    # ---------------------------------------------------------------- mutate
    def upsert(
        self,
        rec: MemoryRecord,
        *,
        merge: bool = True,
        prefer_user: bool = True,
    ) -> MemoryRecord:
        """Insert or merge ``rec`` by key.

        ``prefer_user``: a user-corrected existing record is never downgraded
        by an auto pipeline record — the new value is rejected. Conflict is
        flagged as CONFLICTED when two non-user values differ materially.
        """
        existing = self.records.get(rec.key)
        if existing is None:
            self.records[rec.key] = rec
            return rec
        if prefer_user and existing.is_user:
            existing.touch(page=rec.page)
            return existing
        if merge:
            self._merge(existing, rec)
            return existing
        self.records[rec.key] = rec
        return rec

    def _merge(self, existing: MemoryRecord, new: MemoryRecord) -> None:
        if existing.value == new.value:
            existing.touch(page=new.page)
            return
        # Values differ. Keep the stronger of the two states.
        from .models import VerificationState

        if existing.is_verified:
            existing.touch(page=new.page)
            return
        # Both non-user and disagreeing: record conflict unless one is new.
        if existing.state == VerificationState.UNCERTAIN or new.state == VerificationState.UNCERTAIN:
            # Prefer the confident one, flag as conflict if both confident.
            if new.confidence > existing.confidence + 0.15:
                existing.value = new.value
                existing.state = VerificationState.CONFLICTED
                existing.source = new.source
            else:
                existing.state = VerificationState.CONFLICTED
            existing.touch(page=new.page)
            return
        if existing.state == VerificationState.CONFLICTED:
            # A third, consistent source can resolve: keep most recent.
            existing.value = new.value
            existing.state = VerificationState.AUTO
            existing.source = new.source
            existing.touch(page=new.page)
            return
        existing.state = VerificationState.CONFLICTED
        existing.extra["conflicting_values"] = list(
            dict.fromkeys(
                [existing.value, new.value]
                + (existing.extra.get("conflicting_values") or [])
            )
        )
        existing.touch(page=new.page)
        existing.updated_at = new.updated_at

    def delete(self, key: str) -> bool:
        return self.records.pop(key, None) is not None

    def mark_user_corrected(self, key: str, value, source="user") -> MemoryRecord | None:
        existing = self.records.get(key)
        if existing is not None:
            existing.value = value
            existing.source = source
            existing.state = "user_corrected"
            existing.confidence = 1.0
            existing.touch()
            return existing
        rec = MemoryRecord(
            kind=self.kind,
            key=key,
            value=value,
            source=source,
            confidence=1.0,
            state="user_corrected",
        )
        self.records[key] = rec
        return rec


# ------------------------------------------------------------------- facade


class MangaMemory:
    """Top-level handle to the whole memory engine (all categories)."""

    def __init__(self, cfg):
        base = memory_dir(cfg)
        self.base = base
        self.stores: dict[str, MemoryStore] = {}
        for kind in FILE_BY_KIND:
            self.stores[kind] = MemoryStore(kind, base / FILE_BY_KIND[kind])

    def load_all(self) -> "MangaMemory":
        for store in self.stores.values():
            store.load()
        return self

    def save_all(self) -> None:
        for store in self.stores.values():
            store.save()

    def store_for(self, kind: str) -> MemoryStore:
        if kind not in self.stores:
            kind = _normalize_kind(kind)
        return self.stores[kind]

    def lookup(self, kind: str = None):
        """Query convenience: returns (kind, store) pairs as a dict-of-stores."""
        if kind:
            return self.store_for(kind)
        return dict(self.stores)


def _normalize_kind(kind: str) -> str:
    aliases = {
        "characters": "character",
        "character": "character",
        "world": "world",
        "story": "story",
        "corrections": "correction",
        "correction": "correction",
        "user": "correction",
        "books": "book",
        "book": "book",
    }
    return aliases.get(str(kind).lower(), "character")


def open_memory(cfg, lazy: bool = True) -> MangaMemory:
    """Open all memory stores (loads records if not lazy)."""
    mem = MangaMemory(cfg)
    if not lazy:
        mem.load_all()
    return mem


def memory_info(cfg) -> dict:
    """Compact stats for the dashboard; never raises."""
    try:
        mem = open_memory(cfg, lazy=True).load_all()
        return {
            "characters": mem.store_for("character").count(),
            "world": mem.store_for("world").count(),
            "story": mem.store_for("story").count(),
            "corrections": mem.store_for("correction").count(),
            "books": mem.store_for("book").count(),
        }
    except Exception:
        return {"characters": 0, "world": 0, "story": 0, "corrections": 0, "books": 0}
