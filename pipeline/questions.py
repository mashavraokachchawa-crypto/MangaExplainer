"""Information Needed — compulsory manga-name question.

This module used to scan analysis JSONs for unknown characters/places/events
and surface them as questions. That approach has been replaced by a single
compulsory question: **"What manga is this?"** The user types the manga name,
the app fetches characters, genres, synopsis, and other facts from MangaDex /
Wikipedia, and stores them as durable memory so the narrator uses correct
names, places and context.

保留 (keep) minimal scaffolding so existing imports don't break.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

QUESTIONS_FILE = "questions.json"

_EMPTY = {"pending": [], "answered": [], "updated_at": None, "last_scan": None}


def _state_path(cfg) -> Path:
    try:
        base = Path(cfg.pipeline.state.dir)
    except Exception:
        base = Path(getattr(cfg, "root_dir", ".")) / "state"
    base.mkdir(parents=True, exist_ok=True)
    return base / QUESTIONS_FILE


def _load(cfg) -> dict:
    path = _state_path(cfg)
    if not path.is_file():
        return dict(_EMPTY)
    try:
        doc = json.loads(path.read_text("utf-8"))
        return doc if isinstance(doc, dict) else dict(_EMPTY)
    except Exception:
        return dict(_EMPTY)


def list_questions(cfg):
    """Never-raises snapshot for the dashboard."""
    try:
        doc = _load(cfg)
        return {
            "pending": doc.get("pending", []),
            "answered": doc.get("answered", []),
            "last_scan": doc.get("last_scan"),
            "updated_at": doc.get("updated_at"),
        }
    except Exception:
        return {"pending": [], "answered": [], "last_scan": None,
                "updated_at": None}


def stats(cfg):
    """Counts for the status block; never raises."""
    doc = _load(cfg)
    return {"pending": len(doc.get("pending", [])),
            "answered": len(doc.get("answered", []))}
