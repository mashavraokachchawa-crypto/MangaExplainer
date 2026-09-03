"""Book memory — durable facts about the manga itself.

Unlike character/world/story (learned page-by-page from understanding), a book
record is *externally sourced*: the reader types the manga's name once, the
app fetches the important facts from the internet (MangaDex, Wikipedia), and the
whole reference is remembered here as a single VERIFIED record. Every later
narration/script prompt sees it via the retriever (BOOK tag), so the narrator
knows the correct title, authors, genres, status and synopsis up front.

Key scheme: ``book::<slugged-title>``. Fetches for the same title refresh the
existing record (keep the first source URL; re-fill missing fields).
"""
from __future__ import annotations

import re
import unicodedata

from .models import MemoryRecord, VerificationState

MAX_SYNOPSIS = 400  # keep the record small enough to render in one prompt line


def _slug(title: str) -> str:
    t = unicodedata.normalize("NFKC", str(title or "")).lower()
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return t[:48] or "manga"


def _clean(info: dict) -> dict:
    """Normalise a fetched reference into the stored value shape."""
    def one(key):
        v = (info or {}).get(key)
        if isinstance(v, dict):  # localized value -> pick en, else first
            if v.get("en"):
                return str(v["en"]).strip()
            for _k, _v in v.items():
                if isinstance(_v, str) and _v.strip():
                    return _v.strip()
            return ""
        return str(v or "").strip()

    def many(key):
        v = (info or {}).get(key) or []
        if isinstance(v, str):
            v = [v]
        out = []
        for x in v:
            if isinstance(x, dict):
                for _k, _v in x.items():
                    if isinstance(_v, str) and _v.strip():
                        out.append(_v.strip())
            elif isinstance(x, str) and x.strip():
                out.append(x.strip())
        # dedupe (case-insensitive) — a synopsis can name a character repeatedly
        seen, uniq = set(), []
        for x in out:
            lk = x.lower()
            if lk not in seen:
                seen.add(lk)
                uniq.append(x)
        return uniq[:20]

    value = {
        "type": "book",
        "title": one("title") or "Untitled",
        "authors": many("authors"),
        "characters": many("characters"),
        "genres": many("genres"),
        "demographic": one("demographic"),
        "status": one("status"),
        "year": one("year"),
        "language": one("language"),
        "synopsis": one("synopsis")[:MAX_SYNOPSIS],
    }
    if info.get("source"):
        value["source"] = str(info["source"])
    if info.get("url"):
        value["url"] = str(info["url"])
    if info.get("cover_url"):
        value["cover_url"] = str(info["cover_url"])
    return {k: v for k, v in value.items() if v not in ("", [], None)}


class BookMemory:
    def __init__(self, store):
        if store.kind != "book":
            raise ValueError("BookMemory needs the 'book' store")
        self.store = store

    def remember(self, info: dict, *, source: str = "internet") -> MemoryRecord:
        """Create or refresh the VERIFIED book record for ``info['title']``."""
        title = str((info or {}).get("title") or "").strip()
        if not title:
            raise ValueError("book info needs a title")
        key = "book::" + _slug(title)
        value = _clean(info)
        rec = self.store.get(key)
        if rec is None:
            rec = MemoryRecord(
                kind="book",
                key=key,
                value=value,
                source=source or "internet",
                confidence=0.9,
                state=VerificationState.VERIFIED,
            )
        else:
            old = rec.value if isinstance(rec.value, dict) else {}
            for k, v in value.items():
                if v not in ("", [], None) and (old.get(k) in ("", [], None)):
                    old[k] = v
            if value.get("synopsis") and not old.get("synopsis"):
                old["synopsis"] = value["synopsis"]
            if value.get("title"):
                old["title"] = value["title"]
            rec.value = old
            rec.confidence = max(rec.confidence, 0.9)
            rec.touch()
        self.store.records[key] = rec
        return rec

    def by_title(self, title: str):
        return self.store.get("book::" + _slug(title))

    def all_books(self):
        return self.store.all()