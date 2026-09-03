"""Ingest understood-page knowledge into the Manga Memory Engine.

Called after the per-page understanding stage (alongside the existing legacy
``context_memory.remember_project``), this walks one page's knowledge dict and
learns characters, places, objects, and story events into the durable stores.

It is fully optional and never raises: a corrupt page knowledge degrades to a
no-op, and a memory write failure never breaks the pipeline.
"""
from __future__ import annotations

import logging

LOG = logging.getLogger("mangaexplainer.manga_memory")


def _iter_panel_visuals(knowledge):
    if not isinstance(knowledge, dict):
        return
    for record in knowledge.get("panels") or []:
        if not isinstance(record, dict):
            continue
        visual = record.get("visual")
        if isinstance(visual, dict):
            yield record, visual


def ingest_page(memory, page, knowledge) -> int:
    """Learn durable memory from one page. Returns number of stores updated."""
    learned = 0
    chars = memory.store_for("character")
    world = memory.store_for("world")
    story = memory.store_for("story")
    for record, visual in _iter_panel_visuals(knowledge):
        for char in visual.get("characters") or []:
            if not isinstance(char, dict):
                continue
            name = str(char.get("name") or "").strip()
            if not name or name.lower() in {"unknown", "unk", "n/a"}:
                continue
            chars.learn(
                name,
                source=f"page_{int(page)}",
                page=int(page),
                description=str(char.get("description") or "")[:120].strip() or None,
                role=str(char.get("role") or char.get("relationship") or "").strip() or None,
            )
            learned += 1
        env = str(visual.get("environment") or "").strip()
        if env and env.lower() not in {"unknown", "unk", "n/a", "none"}:
            world.learn_place(
                env,
                source=f"page_{int(page)}",
                page=int(page),
                description=env,
            )
            learned += 1
        for obj in visual.get("objects") or []:
            name = str(obj.get("name") if isinstance(obj, dict) else obj or "").strip()
            if name and name.lower() not in {"unknown", "unk", "n/a", "none"}:
                world.learn_object(
                    name,
                    source=f"page_{int(page)}",
                    page=int(page),
                )
                learned += 1
        event = str(visual.get("important_event") or "").strip()
        if event and event.lower() not in {"unknown", "unk", "n/a", "none"}:
            ev_chars = []
            for c in visual.get("characters") or []:
                if isinstance(c, dict) and str(c.get("name") or "").strip():
                    ev_chars.append(str(c["name"]).strip())
            story.add_event(
                event,
                page=int(page),
                source=f"page_{int(page)}",
                characters=ev_chars,
                location=str(visual.get("environment") or "")[:80] or None,
            )
            learned += 1
    return learned


def remember_manga(cfg, page, knowledge=None, pdf_name=None):
    """Full entry point: load durable memory, ingest one page, persist.

    Mirrors ``context_memory.remember_project`` in spirit. Cannot raise into
    the pipeline.
    """
    try:
        if knowledge is None:
            from ..knowledge import KnowledgeError, load_page_knowledge

            try:
                knowledge = load_page_knowledge(cfg, page)
            except KnowledgeError:
                knowledge = None
        if not isinstance(knowledge, dict) or knowledge.get("result") == "error":
            return False
        from .store import open_memory

        memory = open_memory(cfg, lazy=True).load_all()
        ingest_page(memory, page, knowledge)
        memory.save_all()
        return True
    except Exception:
        LOG.exception("manga memory ingest failed for page %s", page)
        return False
