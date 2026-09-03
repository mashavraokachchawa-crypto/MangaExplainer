"""Knowledge extraction: bridges the existing PDF analysis pipeline into the
persistent knowledge database.

Called per-page after VLM understanding completes.  Extracts structured data
from the analysis JSON (which already has characters, environment, events, etc.)
and stores it in the SQLite knowledge DB with source tracking and confidence.

This module is deliberately lightweight — it reads existing analysis artifacts,
it does NOT re-run VLM or OCR.  It feeds into the knowledge_db for persistent
storage and later retrieval.

Flow per page:
    analysis/page_NNN_panel_YYY.json  (already produced)
        → extract characters, environments, events, objects
        → resolve character IDs (deduplicate via aliases)
        → store in knowledge DB with source=pdf_page_NNN
        → record source evidence
        → update manga memory
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

LOG = logging.getLogger("mangaexplainer.knowledge_extract")

_ELIDED = {"", "unknown", "unk", "n/a", "none", "unknowns", "tbd"}


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


def _extract_page_knowledge(cfg, page: int) -> dict | None:
    """Load the analysis knowledge for a single page (already built)."""
    analysis_dir = Path(cfg.output.analysis_dir)
    knowledge_file = analysis_dir / f"page_{page:03d}_knowledge.json"
    if not knowledge_file.is_file():
        return None
    try:
        return json.loads(knowledge_file.read_text("utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def extract_characters_from_page(page_data: dict, page_num: int) -> list[dict]:
    """Extract character information from one page's panel analyses.

    Returns a list of character records suitable for knowledge_db.add_character().
    """
    characters = []
    seen_names = set()

    for panel in page_data.get("panels") or []:
        visual = panel.get("visual") or {}
        panel_id = panel.get("panel_id", "")
        for char in visual.get("characters") or []:
            if not isinstance(char, dict):
                continue
            name = _clean(char.get("name"), limit=40)
            if not name or name.lower() in _ELIDED:
                # Unknown character — still track for merging later
                name = f"Unknown_{panel_id}"
                char_id = None
            else:
                char_id = None  # Let DB resolve

            if name in seen_names:
                # Reinforce existing character on this page
                for c in characters:
                    if c["name"] == name:
                        c["appearance_count"] = c.get("appearance_count", 0) + 1
                        break
                continue
            seen_names.add(name)

            description = _clean(char.get("description"), limit=120)
            role = _clean(char.get("role") or char.get("relationship"), limit=60)
            action = _clean(char.get("action"), limit=80)
            emotion = _clean(char.get("emotion"), limit=40)

            characters.append({
                "name": name,
                "character_id": char_id,
                "description": description or "",
                "role": role or "",
                "visual_traits": {
                    "action": action or "",
                    "emotion": emotion or "",
                },
                "first_page": page_num,
                "last_page": page_num,
                "appearance_count": 1,
                "confidence": float(visual.get("confidence", 0.5)),
                "source": f"pdf_page_{page_num:03d}",
                "extra": {
                    "panel_id": panel_id,
                    "raw_description": description or "",
                },
            })

    return characters


def extract_locations_from_page(page_data: dict, page_num: int) -> list[dict]:
    """Extract location information from one page's analyses."""
    locations = []
    seen = set()

    for panel in page_data.get("panels") or []:
        visual = panel.get("visual") or {}
        env = _clean(visual.get("environment"), limit=100)
        if not env or env.lower() in _ELIDED:
            continue
        if env in seen:
            continue
        seen.add(env)

        locations.append({
            "name": env,
            "description": env,  # VLM often gives brief description as environment
            "location_type": "",  # Not typically provided by VLM
            "first_page": page_num,
            "last_page": page_num,
            "appearance_count": 1,
            "confidence": float(visual.get("confidence", 0.5)),
            "source": f"pdf_page_{page_num:03d}",
        })

    return locations


def extract_events_from_page(page_data: dict, page_num: int,
                              chapter_id: int | None = None) -> list[dict]:
    """Extract story events from one page's analyses."""
    events = []
    seen = set()

    for panel in page_data.get("panels") or []:
        visual = panel.get("visual") or {}
        panel_id = panel.get("panel_id", "")
        event_text = _clean(visual.get("important_event"), limit=200)
        if not event_text or event_text.lower() in _ELIDED:
            continue

        # Deduplicate similar events on same page
        event_hash = re.sub(r"\s+", " ", event_text.lower())[:80]
        if event_hash in seen:
            continue
        seen.add(event_hash)

        # Extract character IDs from this panel's characters
        char_names = []
        for c in visual.get("characters") or []:
            if isinstance(c, dict):
                name = _clean(c.get("name"), limit=40)
                if name and name.lower() not in _ELIDED:
                    char_names.append(name)

        # Determine importance from story_relevance
        relevance = str(visual.get("story_relevance", "")).lower()
        importance = 0.5
        if any(w in relevance for w in ("critical", "important", "key", "major")):
            importance = 0.8
        elif any(w in relevance for w in ("minor", "background", "low")):
            importance = 0.3

        events.append({
            "event_type": "page",
            "page_number": page_num,
            "chapter_id": chapter_id,
            "characters": char_names,
            "location": _clean(visual.get("environment"), limit=80) or "",
            "description": event_text,
            "importance": importance,
            "confidence": float(visual.get("confidence", 0.5)),
            "source": f"pdf_page_{page_num:03d}",
            "extra": {
                "panel_id": panel_id,
                "actions": visual.get("actions", []),
                "objects": visual.get("objects", []),
                "visual_effects": visual.get("visual_effects", []),
            },
        })

    return events


def extract_objects_from_page(page_data: dict, page_num: int) -> list[dict]:
    """Extract notable objects from one page."""
    objects = []
    seen = set()

    for panel in page_data.get("panels") or []:
        visual = panel.get("visual") or {}
        for obj in visual.get("objects") or []:
            name = _clean(
                obj.get("name") if isinstance(obj, dict) else str(obj),
                limit=60,
            )
            if not name or name.lower() in _ELIDED or name in seen:
                continue
            seen.add(name)
            objects.append({
                "name": name,
                "first_page": page_num,
                "source": f"pdf_page_{page_num:03d}",
            })

    return objects


def ingest_page_to_knowledge_db(db, manga_id: str, cfg, page: int) -> dict:
    """Extract knowledge from one page and store it in the knowledge DB.

    Returns a summary: {"characters": N, "locations": N, "events": N, "objects": N}
    """
    page_data = _extract_page_knowledge(cfg, page)
    if page_data is None:
        return {"characters": 0, "locations": 0, "events": 0, "objects": 0,
                "status": "no_knowledge"}

    # Find chapter for this page
    chapter = db.chapter_for_page(manga_id, page)
    chapter_id = chapter.get("id") if chapter else None

    # Extract structured data
    characters = extract_characters_from_page(page_data, page)
    locations = extract_locations_from_page(page_data, page)
    events = extract_events_from_page(page_data, page, chapter_id)
    objects = extract_objects_from_page(page_data, page)

    # Store characters
    char_count = 0
    for char in characters:
        try:
            char_id = db.add_character(manga_id, char)
            # Record source evidence
            db.add_source_evidence(manga_id, {
                "entity_type": "character",
                "entity_key": char_id,
                "source_type": "vlm",
                "pdf_page": page,
                "panel_id": char.get("extra", {}).get("panel_id", ""),
                "detail": char.get("description", "") or char.get("name", ""),
                "confidence": char.get("confidence", 0.5),
            })
            char_count += 1
        except Exception as e:
            LOG.warning("failed to store character %s for page %d: %s",
                       char.get("name"), page, e)

    # Store locations
    loc_count = 0
    for loc in locations:
        try:
            loc_id = db.add_location(manga_id, loc)
            loc_count += 1
        except Exception as e:
            LOG.warning("failed to store location %s for page %d: %s",
                       loc.get("name"), page, e)

    # Store events
    evt_count = 0
    for evt in events:
        try:
            db.add_event(manga_id, evt)
            evt_count += 1
        except Exception as e:
            LOG.warning("failed to store event for page %d: %s", page, e)

    # Store objects as world memory
    for obj in objects:
        try:
            db.add_location(manga_id, {
                "name": obj["name"],
                "description": "",
                "location_type": "object",
                "first_page": page,
                "last_page": page,
                "appearance_count": 1,
                "confidence": 0.5,
                "source": obj.get("source", f"pdf_page_{page:03d}"),
            })
        except Exception:
            pass

    # Mark checkpoint
    db.checkpoint_set(manga_id, "extract", "completed", page=page)

    return {
        "characters": char_count,
        "locations": loc_count,
        "events": evt_count,
        "objects": len(objects),
        "status": "ok",
    }


def extract_ocr_text_to_db(db, manga_id: str, cfg, page: int, panel: int):
    """Store OCR text as source evidence for later use."""
    ocr_dir = Path(cfg.output.ocr_dir)
    ocr_file = ocr_dir / f"page_{page:03d}_panel_{panel:03d}.json"
    if not ocr_file.is_file():
        return
    try:
        data = json.loads(ocr_file.read_text("utf-8"))
    except Exception:
        return

    combined = (data.get("combined_text") or "").strip()
    if not combined:
        return

    # Store OCR text as evidence for this page
    for block in data.get("text_blocks") or []:
        text = (block.get("text") or "").strip()
        if not text:
            continue
        db.add_source_evidence(manga_id, {
            "entity_type": "page_text",
            "entity_key": f"page_{page:03d}_panel_{panel:03d}",
            "source_type": "ocr",
            "pdf_page": page,
            "panel_id": f"p{page:03d}_{panel:03d}",
            "detail": text,
            "confidence": float(block.get("confidence", 0.5)),
        })


def run_full_extraction(db, manga_id: str, cfg, pages: list[int]) -> dict:
    """Extract knowledge from all processed pages into the knowledge DB.

    This is called after the understand_panels stage completes.
    Returns a summary of what was extracted.
    """
    total_chars = 0
    total_locs = 0
    total_events = 0
    total_objects = 0
    pages_processed = 0

    for page in pages:
        # Skip if already extracted
        if db.checkpoint_status(manga_id, "extract", page) == "completed":
            continue

        result = ingest_page_to_knowledge_db(db, manga_id, cfg, page)
        if result.get("status") == "ok":
            pages_processed += 1
            total_chars += result["characters"]
            total_locs += result["locations"]
            total_events += result["events"]
            total_objects += result["objects"]

    return {
        "pages_processed": pages_processed,
        "total_pages": len(pages),
        "characters": total_chars,
        "locations": total_locs,
        "events": total_events,
        "objects": total_objects,
    }
