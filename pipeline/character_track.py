"""Character tracking & identification across pages and chapters.

Handles:
- Merging "Unknown Male 1" with later-identified "Guts"
- Cross-page character consistency (reusing knowledge from earlier pages)
- Possible-identity scoring (never force a false identification)
- User correction priority

Built on top of the knowledge_db characters table.
"""
from __future__ import annotations

import logging
import re

LOG = logging.getLogger("mangaexplainer.character_track")

_KNOWN_BASE = {"unknown", "unk", "n/a", "none", "tbd", "unknowns", "(unknown)", "? "}


def normalize_name(name: str) -> str:
    """Normalize a character name for matching.

    - strips quotes/brackets/genitive suffixes
    - collapses whitespace
    - lowercases for comparison (but keeps title-case for display)
    """
    if not name:
        return ""
    text = str(name).strip()
    # Remove possessive/genitive
    text = re.sub(r"['\u2019]s\b", "", text)
    # Remove parenthetical asides like "Guts (warrior)"
    text = re.sub(r"\s*\([^)]*\)", "", text)
    # Remove quotes
    text = text.strip("\"'“”‘’「」『』")
    return " ".join(text.split())


def is_unknown(name: str) -> bool:
    """True if the name is a generic unknown placeholder."""
    if not name:
        return True
    n = normalize_name(name).lower()
    return n in _KNOWN_BASE or n.startswith("unknown")


def fuzzy_similar(a: str, b: str) -> bool:
    """True if two character names refer to the same person (loose match)."""
    na, nb = normalize_name(a).lower(), normalize_name(b).lower()
    if not na or not nb:
        return False
    if na == nb:
        return True
    # Substring containment for longer names
    if len(na) >= 6 and len(nb) >= 6:
        if na in nb or nb in na:
            return True
    # First-name match for multi-word names
    if len(na.split()) > 1 or len(nb.split()) > 1:
        if na.split()[0] == nb.split()[0] and na.split()[0].isalpha():
            return True
    return False


class CharacterTracker:
    """Cross-page character tracking backed by a knowledge DB connection."""

    def __init__(self, db, manga_id: str):
        self.db = db
        self.manga_id = manga_id

    def resolve(self, name: str, confidence: float = 0.5) -> str | None:
        """Resolve a name to a canonical character_id.

        Tries exact alias match, then fuzzy match against known characters.
        Returns None if unresolvable (caller creates a new character).
        """
        if is_unknown(name):
            return None
        char_id = self.db.resolve_character(self.manga_id, name)
        if char_id:
            return char_id

        # Fuzzy match against existing characters
        for char in self.db.get_characters(self.manga_id):
            if fuzzy_similar(name, char["name"]):
                # Register alias so future lookups are exact
                self.db._register_alias(
                    self.manga_id, char["character_id"], name,
                    source=f"pdf_match:{char['character_id']}",
                )
                self.db.commit()
                return char["character_id"]
            for alias in char.get("aliases") or []:
                if fuzzy_similar(name, alias):
                    self.db._register_alias(
                        self.manga_id, char["character_id"], name,
                        source="pdf_match",
                    )
                    self.db.commit()
                    return char["character_id"]
        return None

    def track_appearance(self, name: str, page: int, description: str = "",
                          role: str = "", source: str = "pdf",
                          confidence: float = 0.5) -> str:
        """Record a character's appearance on a page, merging if already known."""
        if is_unknown(name):
            # Unknown — record as a distinct unknown placeholder but flag
            # possible identity for later merging.
            return self._track_unknown(page, description, source)

        char_id = self.resolve(name, confidence)
        char_id = char_id or self.db.add_character(self.manga_id, {
            "name": name,
            "description": description,
            "role": role,
            "first_page": page,
            "last_page": page,
            "appearance_count": 1,
            "confidence": confidence,
            "source": source,
        })

        # Bump appearance & spans
        self.db.add_character(self.manga_id, {
            "character_id": char_id,
            "name": name,
            "page_last": page,
            "last_page": page,
            "first_page": page,
            "description": description,
            "role": role,
            "confidence": confidence,
            "source": source,
        })
        return char_id

    def _track_unknown(self, page: int, description: str, source: str) -> str:
        """Track an unknown placeholder, keyed by page so it stays isolated."""
        # Unknown characters are grouped per-page under a placeholder id
        # so they don't multiply; possible identities get merged later.
        key = f"unknown_page_{page:03d}"
        char_id = f"char_unknown_{page:03d}"
        existing = self.db.get_character(self.manga_id, char_id)
        if not existing:
            self.db.add_character(self.manga_id, {
                "character_id": char_id,
                "name": f"Unknown (page {page})",
                "description": description or "An unidentified character.",
                "role": "unknown",
                "first_page": page,
                "last_page": page,
                "appearance_count": 1,
                "confidence": 0.2,  # intentionally low
                "state": "uncertain",
                "identity_note": "Possible identity not yet determined.",
                "source": source,
            })
        else:
            # Reinforce + maybe identify
            self.db.add_character(self.manga_id, {
                "character_id": char_id,
                "last_page": page,
                "appearance_count": existing["appearance_count"] + 1,
            })
            # If the VLM provides a description, use it to suggest identity
            if description and description != "unknown":
                self.db.update_character_note(
                    self.manga_id, char_id, description,
                    extra={"last_page": page},
                )
        return char_id

    def merge_unknowns(self, resolved_name: str, unknown_char_ids: list[str],
                       confidence: float = 0.6):
        """Merge one or more unknown placeholders into a resolved character.

        After the user (or internet research) identifies "Unknown Male 1" as
        "Guts", this merges all matching unknown placeholder records into the
        canonical Guts character.
        """
        char_id = self.resolve(resolved_name)
        if not char_id:
            char_id = self.db.add_character(self.manga_id, {
                "name": resolved_name,
                "confidence": confidence,
                "source": "user" if confidence >= 0.9 else "internet",
            })

        for unknown_id in unknown_char_ids:
            unknown = self.db.get_character(self.manga_id, unknown_id)
            if not unknown:
                continue
            # Transfer appearances to the resolved character
            self.db.add_character(self.manga_id, {
                "character_id": char_id,
                "appearance_count": unknown["appearance_count"],
                "first_page": unknown.get("first_page"),
                "last_page": unknown.get("last_page"),
                "confidence": confidence,
            })
            # Delete the unknown record
            self.db.conn.execute(
                "DELETE FROM characters WHERE manga_id=? AND character_id=?",
                (self.manga_id, unknown_id),
            )
            # Re-point any alias
            self.db._register_alias(
                self.manga_id, char_id, unknown["name"],
                source=f"user_merge_from:{unknown_id}",
            )
        self.db.commit()
        return char_id

    def attempt_identification(self, name: str, page: int,
                               candidates: list[dict]) -> str | None:
        """Given a name and a list of possible identities, score them.

        candidates: list of {"name":..., "description":..., "confidence":...}
        Returns the best matching canonical character_id, or None.
        """
        if is_unknown(name):
            return None
        name_norm = normalize_name(name).lower()

        scored = []
        for cand in candidates:
            cand_name = normalize_name(cand.get("name", "")).lower()
            score = 0.0
            if cand_name == name_norm:
                score = 1.0
            elif fuzzy_similar(name, cand.get("name", "")):
                score = 0.7
            else:
                # Check description overlap
                desc_norm = normalize_name(cand.get("description", "")).lower()
                if name_norm and name_norm in desc_norm:
                    score = 0.6
            if score > 0:
                scored.append((score, cand))

        if not scored:
            return None

        # Best score wins; return None if best is too ambiguous (<0.6)
        scored.sort(key=lambda x: -x[0])
        best_score, best = scored[0]
        if best_score < 0.6:
            return None
        return self.resolve(best.get("name", ""))
