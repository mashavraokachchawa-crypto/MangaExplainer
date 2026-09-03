"""User corrections — the highest-priority layer of the memory engine.

A correction is a user-stated fact that overrides auto-learned memory. These
never get flagged as conflicts and always win in retrieval, so a reader can
fix a wrong name, a mis-remembered location, or correct reading direction.
"""
from __future__ import annotations

from .models import MemoryRecord, VerificationState

# Keyword -> correction kind. Store records under these stable keys so a user
# can correct e.g. the protagonist's name and have it win everywhere.
KINDS = {
    "character_name": "character_name",
    "location": "location",
    "fact": "fact",
    "reading_direction": "reading_direction",
    "relationship": "relationship",
}


class UserCorrectionMemory:
    def __init__(self, store):
        if store.kind != "correction":
            raise ValueError("UserCorrectionMemory needs the 'correction' store")
        self.store = store

    def add(
        self,
        target: str,
        correction: str,
        *,
        kind: str = "fact",
        source: str = "user",
        page: int | None = None,
    ) -> MemoryRecord:
        rec = MemoryRecord(
            kind="correction",
            key=f"correction::{kind}::{target}",
            value={"target": target, "correction": correction, "kind": kind},
            source=source,
            page=page,
            confidence=1.0,
            state=VerificationState.USER_CORRECTED,
            extra={"kind": kind, "target": target},
        )
        self.store.upsert(rec, merge=False)
        return rec

    def all_corrections(self) -> list[MemoryRecord]:
        return self.store.all()

    def for_kind(self, kind: str) -> list[MemoryRecord]:
        return [r for r in self.store.all() if (r.extra or {}).get("kind") == kind]
