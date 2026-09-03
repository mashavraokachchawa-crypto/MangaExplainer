"""Source-aware verification: reconcile internet and PDF evidence.

Never silently picks one source over another.  When internet and PDF
disagree, records a conflict with both values and both sources.  The PDF is
the primary source for what's *currently being processed*; the internet
provides context, names, and background.

Flow:
  - register_evidence() stores each observed fact with its source.
  - verify() compares new evidence against existing; flags a CONFLICT when
    two non-user sources disagree materially.
  - user_correction() always wins (highest priority).
"""
from __future__ import annotations

import logging

LOG = logging.getLogger("mangaexplainer.verification")


def register_evidence(db, manga_id: str, entity_type: str, entity_key: str,
                      value: str, source_type: str, source_url: str = "",
                      pdf_page: int | None = None, confidence: float = 0.5,
                      detail: str = "") -> dict:
    """Register one piece of evidence about an entity.

    Returns a dict with either {"ok": True} or {"conflict": {...}}.

    ``entity_type`` ∈ {character, location, chapter, event, metadata}
    ``entity_key``  is the canonical id (e.g. character_id, field name).
    ``value``       is the observed value for this entity/field.
    ``source_type`` ∈ {pdf, internet, user, ocr, vlm}
    """
    # Record the evidence
    db.add_source_evidence(manga_id, {
        "entity_type": entity_type,
        "entity_key": entity_key,
        "source_type": source_type,
        "source_url": source_url,
        "pdf_page": pdf_page,
        "detail": detail or value,
        "confidence": confidence,
    })

    # Look for an existing conflict on the same entity+field
    conflicts = db.get_unresolved_conflicts(manga_id) or []

    # Check for a material disagreement with the value stored in the entity
    material = _materially_different(db, manga_id, entity_type, entity_key,
                                      value)
    if not material:
        return {"ok": True, "recorded": True, "conflict": None}

    # There's a disagreement.  If a conflict is already open for this
    # entity+field, and it's from a different source, it stays open.
    for conflict in conflicts:
        if (conflict["entity_type"] == entity_type
                and conflict["entity_key"] == entity_key):
            return {"ok": False, "recorded": True,
                    "conflict": conflict,
                    "reason": "existing_conflict"}

    # Open a NEW conflict
    conflict_id = db.add_conflict(manga_id, {
        "entity_type": entity_type,
        "entity_key": entity_key,
        "field_name": "value",
        "value_a": material.get("existing", ""),
        "source_a": material.get("existing_source", ""),
        "value_b": value,
        "source_b": source_type,
    })
    LOG.info("recorded source conflict %s/%s: '%s' (%s) vs '%s' (%s)",
             entity_type, entity_key,
             material.get("existing"), material.get("existing_source"),
             value, source_type)
    return {"ok": False, "recorded": True,
            "conflict": {"id": conflict_id,
                          "entity_type": entity_type,
                          "entity_key": entity_key,
                          "value_a": material.get("existing"),
                          "source_a": material.get("existing_source"),
                          "value_b": value,
                          "source_b": source_type},
            "reason": "new_conflict"}


def _materially_different(db, manga_id: str, entity_type: str,
                          entity_key: str, value: str):
    """Return existing value + source if it differs materially, else None."""
    existing = _get_entity_value(db, manga_id, entity_type, entity_key)
    if existing is None:
        return None
    existing_value, existing_source = existing
    # No real disagreement when either side is empty/unknown
    if not str(existing_value or "").strip():
        return None
    if not str(value or "").strip():
        return None
    if str(existing_value).strip().lower() == str(value).strip().lower():
        return None
    return {"existing": str(existing_value), "existing_source": existing_source}


def _get_entity_value(db, manga_id: str, entity_type: str, entity_key: str):
    """Fetch the current stored value for an entity — best effort."""
    try:
        if entity_type == "metadata":
            manga = db.get_manga(manga_id)
            if manga and entity_key in manga:
                return manga.get(entity_key), _source_of_metadata(db, manga_id, entity_key)
        elif entity_type == "character":
            char = db.get_character(manga_id, entity_key)
            if char:
                return char.get("name") or "", _source_of_character(db, manga_id, entity_key)
        elif entity_type == "chapter":
            return None, ""
    except Exception:
        return None
    return None


def _source_of_metadata(db, manga_id, field):
    evs = db.get_source_evidence(manga_id, "metadata", field)
    if evs:
        return evs[-1].get("source_type", "pdf")
    return "pdf"


def _source_of_character(db, manga_id, char_id):
    evs = db.get_source_evidence(manga_id, "character", char_id)
    if evs:
        return evs[0].get("source_type", "pdf")
    return "pdf"


def resolve_conflict_prefer(db, manga_id: str, conflict_id: int,
                            preferred_source: str):
    """Resolve a conflict by preferring one source (internet or pdf).

    Modern approach: the PDF is primary for the current content, but user
    corrections always win.  Returns the resolved value.
    """
    conflicts = db.get_unresolved_conflicts(manga_id) or []
    conflict = next((c for c in conflicts if c["id"] == conflict_id), None)
    if not conflict:
        return None

    # User source always wins
    if conflict["source_a"] == "user":
        resolved = conflict["value_a"]
        resolved_by = "user"
    elif conflict["source_b"] == "user":
        resolved = conflict["value_b"]
        resolved_by = "user"
    elif conflict["source_a"] == preferred_source:
        resolved = conflict["value_a"]
        resolved_by = "auto"
    elif conflict["source_b"] == preferred_source:
        resolved = conflict["value_b"]
        resolved_by = "auto"
    else:
        # Neither is preferred — keep the PDF one (primary for current content)
        resolved = conflict["value_a"] if conflict["source_a"] == "pdf" else conflict["value_b"]
        resolved_by = "auto"

    db.resolve_conflict(conflict_id, resolved, resolved_by)
    return {"resolved": resolved, "resolved_by": resolved_by}


def user_correction(db, manga_id: str, entity_type: str, entity_key: str,
                    corrected_value: str, field: str = "value"):
    """Apply a user correction — the highest-priority signal.

    Resolves any open conflict for this entity and records the user's value
    as the canonical one.
    """
    # Resolve any open conflict
    conflicts = db.get_unresolved_conflicts(manga_id) or []
    for conflict in conflicts:
        if (conflict["entity_type"] == entity_type
                and conflict["entity_key"] == entity_key):
            db.resolve_conflict(conflict["id"], corrected_value, "user")

    # Record the user value as a new source evidence
    db.add_source_evidence(manga_id, {
        "entity_type": entity_type,
        "entity_key": entity_key,
        "source_type": "user",
        "detail": corrected_value,
        "confidence": 1.0,
    })

    # Update the underlying entity value if applicable
    _apply_user_value(db, manga_id, entity_type, entity_key, corrected_value)

    return {"ok": True, "entity_type": entity_type, "entity_key": entity_key,
            "value": corrected_value}


def _apply_user_value(db, manga_id, entity_type, entity_key, value):
    try:
        if entity_type == "metadata":
            db.upsert_manga({entity_key: value}, manga_id)
        elif entity_type == "character":
            db.update_character_note(manga_id, entity_key, identity_note=str(value))
    except Exception:
        pass
