"""Research cache: wraps internet_ref with persistent caching in the knowledge DB.

Every internet fetch result is stored in the research_cache table so the same
query is never repeated.  Cache entries track when they were fetched and can
be refreshed when stale.

Usage::

    cached = ResearchCache(db, manga_id)
    result = cached.fetch("characters", lambda: internet_ref.fetch_characters("Berserk"))
"""
from __future__ import annotations

import json
import logging
import time
from typing import Callable

LOG = logging.getLogger("mangaexplainer.research_cache")

# Cache TTLs per query type (seconds).  None = never expires.
_CACHE_TTL = {
    "book_ref": 86400 * 30,      # 30 days — manga metadata rarely changes
    "characters": 86400 * 30,    # 30 days
    "character_portraits": 86400 * 7,  # 7 days — images can change
    "chapter_list": 86400 * 7,   # 7 days
    "volume_list": 86400 * 7,    # 7 days
    "synopsis": 86400 * 30,      # 30 days
}


class ResearchCache:
    """Persistent research cache backed by the knowledge DB's research_cache table."""

    def __init__(self, db, manga_id: str):
        self.db = db
        self.manga_id = manga_id

    def get(self, query: str, source_url: str = "") -> dict | None:
        """Look up cached result.  Returns None on miss."""
        return self.db.get_cached_research(self.manga_id, query, source_url)

    def put(self, query: str, result: dict, source_url: str = ""):
        """Store a result in the cache."""
        self.db.cache_research(self.manga_id, query, source_url, result)

    def fetch(self, query: str, fetcher: Callable[[], dict],
              source_url: str = "", force: bool = False) -> dict | None:
        """Cache-first fetch: return cached result or call fetcher.

        ``fetcher`` is a callable that returns a dict (or None on failure).
        The result is cached on success.
        """
        if not force:
            cached = self.get(query, source_url)
            if cached is not None:
                # Check TTL
                fetched_at = cached.get("fetched_at") or cached.get("_fetched_at")
                ttl = _CACHE_TTL.get(query)
                if ttl and fetched_at:
                    try:
                        from datetime import datetime, timezone
                        fetched_dt = datetime.fromisoformat(fetched_at)
                        age = (datetime.now(timezone.utc) - fetched_dt).total_seconds()
                        if age > ttl:
                            cached = None  # Expired
                    except Exception:
                        pass
                if cached is not None:
                    LOG.debug("research cache hit: %s", query)
                    return cached

        try:
            result = fetcher()
        except Exception as e:
            LOG.warning("research fetch failed for %s: %s", query, e)
            return None

        if result is not None:
            # Add fetch timestamp
            result["_fetched_at"] = json.loads(
                json.dumps(result, default=str)
            ).get("_fetched_at")  # In case already present
            from datetime import datetime, timezone
            result["_fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self.put(query, result, source_url)
            LOG.info("research cached: %s (source=%s)", query, source_url or "auto")

        return result


def store_book_reference(db, manga_id: str, book_info: dict,
                          research_cache: ResearchCache | None = None):
    """Store a fetched book reference into both manga record and cache."""
    if not book_info or not book_info.get("title"):
        return

    # Update manga metadata from book reference
    manga_data = {
        "title": book_info.get("title", ""),
        "alt_titles": book_info.get("alt_titles") or [],
        "author": book_info.get("author", "") or ", ".join(book_info.get("authors") or []),
        "genres": book_info.get("genres") or [],
        "demographic": book_info.get("demographic", ""),
        "status": book_info.get("status", ""),
        "synopsis": book_info.get("synopsis", ""),
        "cover_url": book_info.get("cover_url", ""),
        "total_chapters": book_info.get("total_chapters"),
        "total_volumes": book_info.get("total_volumes"),
        "publisher": book_info.get("publisher", ""),
        "magazine": book_info.get("magazine", ""),
        "language": book_info.get("language", ""),
        "year": book_info.get("year", ""),
    }
    db.upsert_manga(manga_data, manga_id)

    # Record source evidence for each metadata field
    for field in ("title", "author", "genres", "status", "synopsis"):
        value = book_info.get(field)
        if value:
            db.add_source_evidence(manga_id, {
                "entity_type": "metadata",
                "entity_key": field,
                "source_type": "internet",
                "source_url": book_info.get("url", ""),
                "detail": str(value)[:500],
                "confidence": book_info.get("confidence", 0.8),
            })

    # Cache the full result
    if research_cache:
        research_cache.put(
            f"book_ref:{book_info.get('title', '')}",
            book_info,
            source_url=book_info.get("url", ""),
        )


def store_character_list(db, manga_id: str, characters: list[dict],
                          research_cache: ResearchCache | None = None,
                          source: str = "internet"):
    """Store internet-fetched character list into the knowledge DB.

    Each character gets a record with source="internet" so it can be
    cross-referenced with PDF-detected characters.
    """
    stored = 0
    for char in characters:
        name = char.get("name", "").strip()
        if not name:
            continue

        role = char.get("role", "")
        description = char.get("description", "")
        if isinstance(role, dict):
            role = str(role.get("en", "") or next(iter(role.values()), ""))
        if isinstance(description, dict):
            description = str(description.get("en", "") or next(iter(description.values()), ""))

        char_data = {
            "name": name,
            "description": str(description)[:200] if description else "",
            "role": str(role)[:100] if role else "",
            "confidence": 0.7,  # Internet source, not verified against PDF
            "source": source,
        }

        try:
            char_id = db.add_character(manga_id, char_data)
            db.add_source_evidence(manga_id, {
                "entity_type": "character",
                "entity_key": char_id,
                "source_type": "internet",
                "source_url": "",
                "detail": f"Name: {name}, Role: {role}",
                "confidence": 0.7,
            })
            stored += 1
        except Exception as e:
            LOG.warning("failed to store internet character %s: %s", name, e)

    if research_cache and characters:
        research_cache.put(
            f"characters:{db.get_manga(manga_id).get('title', '')}",
            {"characters": characters},
        )

    return stored
