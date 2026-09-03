"""Lightweight multi-project registry for MangaExplainer.

The pipeline itself stays single (one config, one active PDF), but the app can
hold MANY named projects — one per manga — so the home screen can list what
has been worked on and a "new project" can be started with one name.

Each project record keeps the durable identity + the memory we gathered about
the manga:

    {
      "slug":           "one-piece",
      "name":           "One Piece",
      "cover_file":     "cover.jpg",          # relative to image store dir
      "characters":     {"Luffy": "luffy.jpg", ...},  # name -> image file
      "toggles":        {"tts": true, "music": false},
      "book":           {...} or null,        # fetched reference snapshot
      "state_dir":      "state",
      "created_at":     "...",                # ISO
      "updated_at":     "...",
    }

Persistence is one JSON file (``state/projects.json``) so it works across
restarts and can round-trip through backups. All reads are tolerant of a
missing/corrupt file (returns []); all writes are atomic-ish (write+rename).

Toggles here are *preferences* — the pipeline honours ``tts.enabled`` /
``music.enabled`` by forwarding the project's choice into the config at run
start (see webui). This module only stores them.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

PROJECTS_FILE = "projects.json"


def _slugify(name: str) -> str:
    t = re.sub(r"[^A-Za-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return t[:48] or "manga"


def _now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def projects_path(state_dir: str | Path) -> Path:
    return Path(state_dir) / PROJECTS_FILE


def list_projects(state_dir: str | Path) -> list[dict]:
    """All projects, newest first. Never raises."""
    path = projects_path(state_dir)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return []
    projects = data.get("projects") if isinstance(data, dict) else data
    if not isinstance(projects, list):
        return []
    projects = [p for p in projects if isinstance(p, dict)]
    projects.sort(key=lambda p: (p.get("updated_at") or ""), reverse=True)
    return projects


def get_project(state_dir: str | Path, slug: str) -> dict | None:
    for p in list_projects(state_dir):
        if p.get("slug") == slug:
            return p
    return None


def save_projects(state_dir: str | Path, projects: list[dict]) -> None:
    path = projects_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(
        {"projects": projects}, indent=2, ensure_ascii=False), "utf-8")
    tmp.replace(path)


def upsert_project(state_dir: str | Path, record: dict) -> dict:
    """Insert a new project or merge ``record`` fields into an existing slug."""
    slug = record.get("slug") or _slugify(record.get("name") or "")
    record["slug"] = slug
    projects = list_projects(state_dir)
    existing = next((p for p in projects if p.get("slug") == slug), None)
    if existing is None:
        record["created_at"] = record.get("created_at") or _now()
        projects.append(record)
    else:
        merged = dict(existing)
        merged.update({k: v for k, v in record.items() if v is not None})
        projects = [merged if p.get("slug") == slug else p for p in projects]
    record["updated_at"] = _now()
    save_projects(state_dir, projects)
    return get_project(state_dir, slug) or record


def delete_project(state_dir: str | Path, slug: str) -> bool:
    projects = list_projects(state_dir)
    kept = [p for p in projects if p.get("slug") != slug]
    if len(kept) == len(projects):
        return False
    save_projects(state_dir, kept)
    return True


def set_toggle(state_dir: str | Path, slug: str, key: str, value: bool) -> dict | None:
    """Persist one per-project toggle (tts / music). Returns the updated record."""
    if key not in ("tts", "music"):
        return None
    existing = get_project(state_dir, slug)
    if existing is None:
        return None
    toggles = dict(existing.get("toggles") or {})
    toggles[key] = bool(value)
    existing["toggles"] = toggles
    return upsert_project(state_dir, {"slug": slug, "toggles": toggles})


def default_toggles() -> dict:
    return {"tts": True, "music": False}