"""Hierarchical story memory (levels 1-5) built on the knowledge database.

Implements the L1-L5 memory hierarchy:

  LEVEL 1: MANGA  — overall manga info (in knowledge_db.manga)
  LEVEL 2: ARC    — major story arcs and important context
  LEVEL 3: CHAPTER — summary + important events per chapter
  LEVEL 4: PAGE   — important info per page
  LEVEL 5: PANEL  — detailed info only when necessary

Retrieval only pulls *relevant* layers — never the whole manga — so the LLM
context stays bounded (low RAM / context win).
"""
from __future__ import annotations

import logging
from pathlib import Path

from .knowledge_db import open_knowledge_db

LOG = logging.getLogger("mangaexplainer.story_memory")

# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def generate_page_summary(db, manga_id: str, page: int, chapter_id: int | None = None,
                          max_events: int = 3) -> dict | None:
    """Build a page summary: events + characters + location from stored events.

    This is deterministic — it compresses what analysis already extracted,
    it does NOT call the LLM.  For richer summaries the pipeline calls the
    LLM separately (see generate_chapter_summaries).
    """
    events = db.get_events(manga_id, page=page)
    if not events:
        return None

    important = [e for e in events if e["importance"] >= 0.5]
    if not important:
        important = events[:1]

    # Deduplicate similar events on the page
    seen = set()
    events_out = []
    for e in important:
        key = e["description"].lower().strip()[:80]
        if key in seen:
            continue
        seen.add(key)
        events_out.append(e)

    # Extract characters present on this page
    char_names = []
    for e in events:
        for c in e.get("characters") or []:
            if c not in char_names:
                char_names.append(c)

    text = " | ".join(e["description"] for e in events_out[:max_events])

    summary = {
        "summary_type": "page",
        "page_number": page,
        "chapter_id": chapter_id,
        "text": text,
        "important_events": [e["description"] for e in events_out],
        "confidence": 0.8,
    }
    db.add_summary(manga_id, summary)
    return summary


def summarize_chapter(db, manga_id: str, chapter: dict,
                      llm, cfg, task="summary") -> dict:
    """Generate short/medium/detail summaries for a chapter using the LLM.

    Falls back to deterministic compression if the LLM is unavailable.
    """
    chapter_id = chapter["id"]
    events = db.get_events(manga_id, chapter_id=chapter_id,
                           min_importance=0.3)

    # Build context from stored events + characters
    context = _chapter_context(db, manga_id, chapter, events)

    detail = None
    if llm is not None:
        try:
            from .prompts import build_chapter_summary_prompt
            prompt = build_chapter_summary_prompt(chapter, context)
            raw = llm.generate(prompt)
            from .llm_provider import clean_text
            detail = clean_text(raw)
        except Exception as e:
            LOG.warning("LLM chapter summary failed: %s", e)
            detail = None

    # Deterministic fallback
    short = _short_summary(events, chapter)
    medium = _medium_summary(events, chapter)

    result = {
        "summary_type": "chapter_detail",
        "chapter_id": chapter_id,
        "text": detail or medium,
        "important_events": [e["description"] for e in events if e["importance"] >= 0.5],
        "new_characters": _new_characters_in_chapter(db, manga_id, chapter),
        "confidence": 0.7,
    }
    db.add_summary(manga_id, result)

    return {
        "chapter_id": chapter_id,
        "short": short,
        "medium": medium,
        "detail": detail or medium,
        "config": cfg,
    }


def _short_summary(events, chapter) -> str:
    """2-5 sentences summarizing the chapter."""
    if not events:
        return f"Chapter {chapter.get('chapter_number', '?')}: (no significant events detected)"
    top = [e for e in events if e["importance"] >= 0.6][:3] or events[:3]
    parts = [f"Chapter {chapter.get('chapter_number', '?')} covers {len(events)} key moments."]
    for e in top:
        parts.append(e["description"])
    # Roughly 2-5 sentences
    return " ".join(parts[:4])


def _medium_summary(events, chapter) -> str:
    """Important events and character actions, chronological."""
    if not events:
        return ""
    lines = []
    for e in events:
        chars = ", ".join(e.get("characters") or [])
        loc = e.get("location") or ""
        parts = []
        if chars:
            parts.append(chars)
        parts.append(e["description"])
        if loc:
            parts.append(f"(at {loc})")
        lines.append(" · ".join(parts))
    return "\n".join(lines)


def _new_characters_in_chapter(db, manga_id: str, chapter: dict) -> list[str]:
    """Characters whose first_appearance is within this chapter's page range."""
    start = chapter["pdf_page_start"]
    end = chapter["pdf_page_end"]
    out = []
    for ch in db.get_characters(manga_id):
        if ch.get("first_page") is not None and start <= ch["first_page"] <= end:
            out.append(ch["name"])
    return out


def _chapter_context(db, manga_id, chapter, events) -> str:
    """Compact text context for the LLM summary prompt."""
    parts = []
    parts.append(f"CHAPTER {chapter.get('chapter_number', '?')}: "
                 f"{chapter.get('title', 'Untitled')} "
                 f"(pages {chapter['pdf_page_start']}–{chapter['pdf_page_end']})")
    for e in events:
        chars = ", ".join(e.get("characters") or [])
        parts.append(f"- [{e.get('page_number')}] {e['description']}"
                     f"{' (' + chars + ')' if chars else ''}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Hierarchical retrieval
# ---------------------------------------------------------------------------

def retrieve_relevant_memory(db, manga_id: str, page: int | None = None,
                              chapter_id: int | None = None,
                              max_chars: int = 3000) -> str:
    """Build a bounded relevant-memory block for the LLM.

    Only pulls what's relevant to the current page/chapter:
    - manga-level facts (title, author, synopsis trimmed)
    - current chapter summary if available
    - events on current/nearby pages
    - characters appearing on this page
    Returns a compact string, never more than max_chars.
    """
    parts = []

    # Level 1: manga facts (compact)
    manga = db.get_manga(manga_id)
    if manga:
        meta = []
        if manga.get("title"):
            meta.append(f"Title: {manga['title']}")
        if manga.get("author"):
            meta.append(f"Author: {manga['author']}")
        if manga.get("genres"):
            meta.append(f"Genres: {', '.join(manga['genres'])}")
        if manga.get("status"):
            meta.append(f"Status: {manga['status']}")
        if meta:
            parts.append("MANGA: " + " | ".join(meta))
        if manga.get("synopsis"):
            parts.append(f"SYNOPSIS: {manga['synopsis'][:400]}")

    # Level 3: chapter summary if available
    if chapter_id is not None:
        summaries = db.get_summaries(manga_id, "chapter_detail", chapter_id)
        if summaries:
            parts.append(f"CHAPTER SUMMARY: {summaries[-1]['text'][:600]}")

    # Level 4: page memory
    if page is not None:
        # Events on this page (already in order)
        events = db.get_events(manga_id, page=page)
        if events:
            ev_lines = [f"- {e['description']}" for e in events[:5]]
            parts.append("THIS PAGE EVENTS:\n" + "\n".join(ev_lines))

        # Nearby pages (context window)
        for nearby in (page - 1, page + 1):
            if nearby >= 1:
                prev_events = db.get_events(manga_id, page=nearby)
                if prev_events:
                    parts.append(
                        f"NEARBY PAGE {nearby}:\n" +
                        "\n".join(f"- {e['description']}" for e in prev_events[:2])
                    )

    # Character context for this page
    if page is not None:
        chars_on_page = _characters_on_page(db, manga_id, page)
        if chars_on_page:
            char_lines = []
            for c in chars_on_page:
                role = f" ({c['role']})" if c.get("role") else ""
                desc = f": {c['description'][:120]}" if c.get("description") else ""
                char_lines.append(f"- {c['name']}{role}{desc}")
            parts.append("CHARACTERS HERE:\n" + "\n".join(char_lines))

    text = "\n\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def _characters_on_page(db, manga_id: str, page: int) -> list[dict]:
    """Characters who appear on this page (from source evidence)."""
    # Best effort: query events on this page for character names, resolve to IDs
    events = db.get_events(manga_id, page=page)
    names = set()
    for e in events:
        for c in e.get("characters") or []:
            names.add(c)

    found = []
    for name in names:
        char_id = db.resolve_character(manga_id, name)
        if char_id:
            char = db.get_character(manga_id, char_id)
            if char:
                found.append(char)
    return found


def get_story_memory(db, manga_id: str, page: int | None = None,
                      chapter_id: int | None = None) -> dict:
    """Return structured hierarchy for the Memory Explorer + retrieval."""
    return {
        "manga_level": db.get_manga(manga_id),
        "arc_level": db.get_events(manga_id, event_type="arc") if page is None else [],
        "chapter_level": db.get_summaries(manga_id, chapter_id=chapter_id) if chapter_id else [],
        "page_level": db.get_events(manga_id, page=page) if page else [],
        "panel_level": _panel_events(db, manga_id, page) if page else [],
    }


def _panel_events(db, manga_id, page):
    """Panel-level detail — only fetched when explicitly requested."""
    events = db.get_events(manga_id, page=page)
    return [e for e in events if e.get("extra", {}).get("panel_id")]


def build_layered_block(db, manga_id: str, page: int | None = None,
                         chapter_id: int | None = None) -> str:
    """Build the hierarchical memory block for the narrator prompt.

    Replaces the flat retrieval with the proper L1-L5 hierarchy while
    staying bounded.
    """
    return retrieve_relevant_memory(db, manga_id, page, chapter_id)
