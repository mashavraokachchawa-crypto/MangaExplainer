"""Story memory — narrative events, timeline, and ongoing story threads."""
from __future__ import annotations

from .models import MemoryRecord, VerificationState


class StoryMemory:
    def __init__(self, store):
        if store.kind != "story":
            raise ValueError("StoryMemory needs the 'story' store")
        self.store = store

    def add_event(
        self,
        event: str,
        *,
        page: int | None = None,
        source: str = "",
        characters: list[str] | None = None,
        location: str | None = None,
        confidence: float = 1.0,
    ) -> MemoryRecord:
        import hashlib

        if not event:
            return None
        digest = hashlib.sha1(event.encode("utf-8")).hexdigest()[:12]
        key = f"event::{digest}"
        rec = self.store.get(key) or MemoryRecord(
            kind="story",
            key=key,
            value={
                "type": "event",
                "text": event,
                "characters": characters or [],
                "location": location,
            },
            source=source,
            page=page,
            confidence=confidence,
        )
        rec.touch(page=page)
        self.store.records[key] = rec
        return rec

    def set_ongoing_thread(self, title: str, detail: str, *, page=None, source="") -> MemoryRecord:
        key = f"thread::{title}"
        rec = self.store.get(key) or MemoryRecord(
            kind="story",
            key=key,
            value={"type": "thread", "title": title, "detail": detail},
            source=source,
            page=page,
        )
        value = rec.value if isinstance(rec.value, dict) else {"type": "thread", "title": title}
        value["detail"] = detail
        rec.value = value
        rec.touch(page=page)
        self.store.records[key] = rec
        return rec

    def events(self, limit: int | None = None) -> list[MemoryRecord]:
        evs = [r for r in self.store.all() if (r.value or {}).get("type") == "event"]
        evs.sort(key=lambda r: r.page or 0)
        return evs[-limit:] if limit else evs

    def threads(self) -> list[MemoryRecord]:
        return [r for r in self.store.all() if (r.value or {}).get("type") == "thread"]

    def all(self):
        return self.store.all()
