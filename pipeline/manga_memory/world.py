"""World memory — places, objects, and recurring world facts."""
from __future__ import annotations

from .models import MemoryRecord, VerificationState


class WorldMemory:
    def __init__(self, store):
        if store.kind != "world":
            raise ValueError("WorldMemory needs the 'world' store")
        self.store = store

    def learn_place(
        self,
        name: str,
        *,
        source: str = "",
        page: int | None = None,
        description: str | None = None,
        confidence: float = 1.0,
    ) -> MemoryRecord:
        key = f"place::{name}"
        rec = self.store.get(key) or MemoryRecord(
            kind="world",
            key=key,
            value={"type": "place", "name": name, "description": description or ""},
            source=source,
            page=page,
            confidence=confidence,
        )
        value = rec.value if isinstance(rec.value, dict) else {"type": "place", "name": name}
        if description and description != value.get("description"):
            value["description"] = description
        rec.value = value
        rec.touch(page=page)
        self.store.records[key] = rec
        return rec

    def learn_object(
        self,
        name: str,
        *,
        source: str = "",
        page: int | None = None,
        description: str | None = None,
        confidence: float = 1.0,
    ) -> MemoryRecord:
        key = f"object::{name}"
        rec = self.store.get(key) or MemoryRecord(
            kind="world",
            key=key,
            value={"type": "object", "name": name, "description": description or ""},
            source=source,
            page=page,
            confidence=confidence,
        )
        value = rec.value if isinstance(rec.value, dict) else {"type": "object", "name": name}
        if description and description != value.get("description"):
            value["description"] = description
        rec.value = value
        rec.touch(page=page)
        self.store.records[key] = rec
        return rec

    def all_places(self):
        return [r for r in self.store.all() if (r.value or {}).get("type") == "place"]

    def all_objects(self):
        return [r for r in self.store.all() if (r.value or {}).get("type") == "object"]

    def all(self):
        return self.store.all()
