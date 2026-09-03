"""Theme-aware context builder — shapes memory into compact prompt blocks.

The narration prompt already has a fixed structure in
``pipeline/prompts.py`` (SCENE_FACTS / PANEL_FACTS). This module builds an
ADDITIONAL, optional memory block that is inserted alongside the existing
project-memory block, so the LLM sees durable, corrected, and story-consistent
facts without bloating the context.

The output stays bounded: retrieve() caps the number of records, and each
record is rendered as one short line.
"""
from __future__ import annotations

from .retrieval import MemoryRetriever
from . import confidence


def _render(rec) -> str:
    tag = {
        "character": "CHAR",
        "world": "WORLD",
        "story": "STORY",
        "correction": "NOTE",
        "book": "BOOK",
    }.get(rec.kind, rec.kind.upper())
    value = rec.value
    if isinstance(value, dict):
        # compact: drop empty keys
        parts = []
        for k, v in value.items():
            if v in (None, "", [], {}):
                continue
            parts.append(f"{k}:{_one(v)}")
        text = " ".join(parts)
    else:
        text = str(value)
    page = f" p{rec.page}" if rec.page is not None else ""
    conf = f" cf:{confidence.effective_confidence(rec):.2f}"
    return f"[{tag}{page}] {text}{conf}"


def _one(v) -> str:
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return str(v)


def build_memory_block(
    memory,
    *,
    page: int | None = None,
    task: str = "narration",
    limit: int = 8,
    extra_text: str | None = None,
    heading: str = "MANGA MEMORY (durable, project-wide)",
) -> str:
    """Render a compact, bounded memory context block ('' if no records)."""
    try:
        retriever = MemoryRetriever(memory, page=page, task=task)
        records = retriever.retrieve(
            limit=limit,
            min_confidence=0.0,
            extra_text=extra_text,
        )
        if not records:
            return ""
        lines = [heading]
        lines.append("These are durable facts the narrator should stay "
                     "consistent with; trust NOTE/user-corrected entries "
                     "most, CONFLICTED entries least.")
        for rec in records:
            lines.append("- " + _render(rec))
        return "\n".join(lines)
    except Exception:
        return ""
