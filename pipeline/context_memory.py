"""Project + page-window memory for the explanation pipeline.

Two independent memories back the narration stage:

1. PROJECT MEMORY (durable, "type 1")
   state/project_memory.json — remembered entities (characters, places,
   objects) accumulated page by page during understanding. It belongs to the
   PROJECT, not to one PDF: resuming the project, or switching to a new manga
   PDF in the same project, keeps every fact already learned (character names,
   traits, roles, first appearances). The script stage receives these so the
   narrator stays consistent across the whole volume.

2. PAGE WINDOW ("type 2", context, only recently relevant)
   state/page_context.json — a rolling window of the LAST 10 understood pages
   (small per-page summaries). Older pages fall out; they are not lost from
   project memory, they just stop being re-fed to the LLM. Capped by default
   (window_size, default 10).

Neither memory EVER breaks the pipeline: every read is optional and every
write is guarded, so a corrupt/missing memory file degrades to "no context".
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger("mangaexplainer.context_memory")

PROJECT_MEMORY_FILENAME = "project_memory.json"
PAGE_WINDOW_FILENAME = "page_context.json"

DEFAULT_WINDOW_SIZE = 10

_ELIDED = {"", "unknown", "none", "n/a", "unknowns", "unk"}


def _clean(value, limit=160):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in _ELIDED:
        return None
    compact = " ".join(text.split())
    if len(compact) > limit:
        compact = compact[:limit].rstrip() + "…"
    return compact


def _state_dir(cfg):
    try:
        return Path(cfg.pipeline.state.dir)
    except (AttributeError, TypeError):
        return Path("state")


def memory_path(cfg):
    return _state_dir(cfg) / PROJECT_MEMORY_FILENAME


def window_path(cfg):
    return _state_dir(cfg) / PAGE_WINDOW_FILENAME


def window_size(cfg):
    try:
        value = int(getattr(cfg, "memory", {}).get("window_size",
                                                   DEFAULT_WINDOW_SIZE))
    except (AttributeError, TypeError, ValueError):
        return DEFAULT_WINDOW_SIZE
    return value if value > 0 else DEFAULT_WINDOW_SIZE


def _atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _read_json(path, default):
    try:
        if not path.is_file():
            return default
        doc = json.loads(path.read_text("utf-8"))
        return doc if isinstance(doc, dict) else default
    except (OSError, json.JSONDecodeError, ValueError):
        LOG.warning("ignoring unreadable memory file %s", path)
        return default


# ------------------------------------------------------------------- entities


def _iter_visuals(knowledge):
    if not isinstance(knowledge, dict):
        return
    for record in knowledge.get("panels") or []:
        if not isinstance(record, dict):
            continue
        visual = record.get("visual")
        if isinstance(visual, dict):
            yield record, visual


def _merge_character(target, page, record):
    name = _clean(record.get("name"), limit=40)
    if name is None:
        return
    entry = target.setdefault(
        name,
        {"first_seen_page": int(page), "appearances": 0,
         "descriptions": [], "roles": []},
    )
    entry["first_seen_page"] = min(int(entry["first_seen_page"]), int(page))
    entry["appearances"] = entry.get("appearances", 0) + 1
    desc = _clean(record.get("description"), limit=120)
    if desc and len(entry["descriptions"]) < 6 and desc not in entry["descriptions"]:
        entry["descriptions"].append(desc)
    role = _clean(record.get("role") or record.get("relationship"), limit=60)
    if role and len(entry["roles"]) < 4 and role not in entry["roles"]:
        entry["roles"].append(role)


def _merge_entity(target, name, page):
    if name is None:
        return
    entry = target.setdefault(
        name, {"first_seen_page": int(page), "appearances": 0})
    entry["first_seen_page"] = min(int(entry["first_seen_page"]), int(page))
    entry["appearances"] = entry.get("appearances", 0) + 1


# ------------------------------------------------------------- page window


def build_page_summary(page, knowledge):
    """One compact line describing a page, for the recent-pages window."""
    parts = []
    for record, visual in _iter_visuals(knowledge):
        bits = []
        chars = sorted({
            _clean(c.get("name"), limit=40) for c in visual.get("characters") or []
            if isinstance(c, dict)
        } - {None})
        if chars:
            bits.append(f"chars:[{', '.join(chars)}]")
        event = _clean(visual.get("important_event"))
        if event:
            bits.append(f"event:{event}")
        text = _clean((record.get("ocr") or {}).get("text"), limit=120)
        if text:
            bits.append(f"text:{text!r}")
        if bits:
            parts.append(f"{record.get('panel_id', '?')} " + " ".join(bits))
    if not parts:
        return f"Page {int(page)}: (no panel information recorded)"
    return f"Page {int(page)}: " + " | ".join(parts)


class PageWindow:
    """Rolling summary of the most recent understood pages (type-2 memory)."""

    def __init__(self, pages=None):
        self.pages = pages if isinstance(pages, dict) else {}

    @classmethod
    def load(cls, cfg):
        doc = _read_json(window_path(cfg), {})
        if isinstance(doc, dict) and isinstance(doc.get("pages"), dict):
            return cls(doc["pages"])
        # legacy bare {page: summary} form
        if isinstance(doc, dict) and doc:
            return cls(doc)
        return cls()

    def add_page(self, page, knowledge, size=None):
        page = int(page)
        size = size or DEFAULT_WINDOW_SIZE
        self.pages[str(page)] = build_page_summary(page, knowledge)
        overflow = sorted((int(k) for k in self.pages), reverse=True)[size:]
        for stale in overflow:
            self.pages.pop(str(stale), None)

    def save(self, cfg):
        _atomic_write(window_path(cfg), {"pages": self.pages})

    def to_prompt(self):
        if not self.pages:
            return ""
        ordered = sorted((int(k) for k in self.pages), reverse=True)
        lines = [
            "RECENT STORY CONTEXT — the last understood pages, newest last. "
            "Use this to keep continuity; older pages are in project memory "
            "(see below) if you need them.",
        ]
        for page in ordered:
            lines.append("- " + self.pages[str(page)])
        return "\n".join(lines)


# ----------------------------------------------------------- project memory


class ProjectMemory:
    """Durable project-level memory (type 1) keyed by entity."""

    def __init__(self, doc=None):
        doc = doc or {}
        self.characters = doc.get("characters") or {}
        self.places = doc.get("places") or {}
        self.objects = doc.get("objects") or {}
        self.facts = doc.get("facts") or []
        self.total_pages_seen = int(doc.get("total_pages_seen") or 0)
        self.first_pdf = doc.get("first_pdf") or ""
        self.updated_at = doc.get("updated_at") or ""

    @classmethod
    def load(cls, cfg):
        return cls(_read_json(memory_path(cfg), {}))

    @classmethod
    def for_pdf(cls, cfg, pdf_name=None):
        """Load project memory, seeding it with the current PDF identity."""
        mem = cls.load(cfg)
        if pdf_name and not mem.first_pdf:
            mem.first_pdf = str(pdf_name)
        return mem

    def remember_page(self, page, knowledge, pdf_name=None):
        page = int(page)
        self.total_pages_seen = max(self.total_pages_seen, page)
        if pdf_name and not self.first_pdf:
            self.first_pdf = str(pdf_name)
        for record, visual in _iter_visuals(knowledge):
            for char in visual.get("characters") or []:
                if isinstance(char, dict):
                    _merge_character(self.characters, page, char)
            _merge_entity(self.places, _clean(visual.get("environment")), page)
            for obj in visual.get("objects") or []:
                _merge_entity(
                    self.objects,
                    _clean(obj.get("name") if isinstance(obj, dict) else obj),
                    page,
                )
        self.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def add_fact(self, fact, page=None):
        fact = _clean(fact, limit=220)
        if not fact:
            return
        line = f"page p.{int(page)}: {fact}" if page else fact
        if line not in self.facts:
            self.facts.append(line)
            self.facts = self.facts[-40:]

    def save(self, cfg):
        _atomic_write(memory_path(cfg), {
            "characters": self.characters,
            "places": self.places,
            "objects": self.objects,
            "facts": self.facts,
            "total_pages_seen": self.total_pages_seen,
            "first_pdf": self.first_pdf,
            "updated_at": self.updated_at,
        })

    def to_prompt(self):
        blocks = []
        if self.characters:
            lines = ["PROJECT MEMORY — characters (durable across the whole "
                     "project, incl. earlier volumes):"]
            for name, entry in sorted(self.characters.items()):
                descs = "; ".join(entry.get("descriptions") or [])
                roles = ", ".join(entry.get("roles") or [])
                extra = []
                if roles:
                    extra.append(roles)
                if descs:
                    extra.append(descs)
                detail = (f" ({'; '.join(extra)})" if extra else "")
                lines.append(
                    f"- {name}{detail} (first seen page "
                    f"{entry.get('first_seen_page')}, {entry.get('appearances', 0)} "
                    f"panel appearance(s))")
            blocks.append("\n".join(lines))
        for key, label in (("places", "PLACES"), ("objects", "OBJECTS")):
            table = getattr(self, key)
            if table:
                lines = [f"PROJECT MEMORY — {label} (recurring throughout the "
                         "project):"]
                for name, entry in sorted(table.items()):
                    lines.append(
                        f"- {name} (first seen page "
                        f"{entry.get('first_seen_page')}, "
                        f"{entry.get('appearances', 0)} appearance(s))")
                blocks.append("\n".join(lines))
        if self.facts:
            lines = ["PROJECT MEMORY — narrative facts:"]
            lines.extend(f"- {fact}" for fact in self.facts)
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)


# ------------------------------------------------------------------- wiring


def remember_project(cfg, page, knowledge=None, pdf_name=None):
    """Update BOTH memories from one understood page (never raises).

    Called after the knowledge stage per page; both memories persist so a
    resume (or a new PDF in the same project) keeps what was learned.
    """
    try:
        if knowledge is None:
            from .knowledge import KnowledgeError, load_page_knowledge
            try:
                knowledge = load_page_knowledge(cfg, page)
            except KnowledgeError:
                knowledge = None
        if not isinstance(knowledge, dict) or knowledge.get("result") == "error":
            return False
        mem = ProjectMemory.load(cfg)
        mem.remember_page(page, knowledge, pdf_name=pdf_name)
        mem.save(cfg)

        win = PageWindow.load(cfg)
        win.add_page(page, knowledge, size=window_size(cfg))
        win.save(cfg)
        return True
    except Exception:  # never let memory break the pipeline
        LOG.exception("context memory update failed for page %s", page)
        return False


def script_context(cfg):
    """(memory_block, window_block) prompt fragments, optional and safe."""
    memory_block = str()
    window_block = str()
    try:
        memory_block = ProjectMemory.load(cfg).to_prompt()
    except Exception:
        memory_block = str()
    try:
        window_block = PageWindow.load(cfg).to_prompt()
    except Exception:
        window_block = str()
    return memory_block, window_block


def memory_info(cfg):
    """Compact dashboard info for /api/live or /api/status; never raises."""
    try:
        mem = ProjectMemory.load(cfg)
        win = PageWindow.load(cfg)
        info = {
            "characters": len(mem.characters),
            "places": len(mem.places),
            "objects": len(mem.objects),
            "facts": len(mem.facts),
            "total_pages_seen": mem.total_pages_seen,
            "window_pages": len(win.pages),
            "window": sorted((int(k) for k in win.pages), reverse=True),
        }
        return info
    except Exception:
        return None


def manga_memory_block(cfg, page=None, task="narration", limit=8, extra_text=None):
    """Render the durable Manga Memory Engine block ('' if empty/disabled).

    Composes atop :meth:`ProjectMemory.to_prompt` so the narration stage gets
    both the flat legacy tables and the richer, corrected, story-aware records.
    """
    try:
        from .manga_memory.store import open_memory
        from .manga_memory.context_builder import build_memory_block

        memory = open_memory(cfg, lazy=True).load_all()
        return build_memory_block(
            memory,
            page=page,
            task=task,
            limit=limit,
            extra_text=extra_text,
        )
    except Exception:
        LOG.exception("manga_memory block build failed")
        return ""


def manga_memory_info(cfg):
    """Rich memory stats (durable engine) for the dashboard; never raises."""
    try:
        from .manga_memory.store import open_memory, memory_info as _mi

        return _mi(cfg)
    except Exception:
        return None