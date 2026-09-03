"""Tests for character tracking / identification (character_track)."""
import tempfile
from pathlib import Path

import pytest

from pipeline.knowledge_db import open_knowledge_db
from pipeline.character_track import (
    CharacterTracker,
    fuzzy_similar,
    is_unknown,
    normalize_name,
)


@pytest.fixture()
def tracker():
    tmp = Path(tempfile.mkdtemp())
    db = open_knowledge_db(tmp)
    mid = db.upsert_manga({"title": "Berserk"})
    tracker = CharacterTracker(db, mid)
    yield tracker
    db.close()


def test_normalize_name():
    assert normalize_name("  Guts  ") == "Guts"
    assert normalize_name("Griffith's army") == "Griffith army"
    assert normalize_name('“Guts” (warrior)') == "Guts"


def test_is_unknown():
    assert is_unknown("unknown")
    assert is_unknown("Unknown")
    assert is_unknown("")
    assert is_unknown("n/a")
    assert not is_unknown("Guts")
    assert not is_unknown("A character")


def test_fuzzy_similar():
    assert fuzzy_similar("Guts", "Guts")                    # exact
    assert fuzzy_similar("KING GRIFFITH", "Griffith")       # token containment, case-insens
    assert fuzzy_similar("The Black Swordsman", "Black Swordsman")  # first-name + containment
    assert fuzzy_similar("King Griffith", "Griffith")       # first-name match
    assert not fuzzy_similar("Guts-san", "Guts")            # distinct wrapper
    assert not fuzzy_similar("Gattsu", "Guts")              # different name
    assert not fuzzy_similar("Griffith", "Guts")            # unrelated


def test_track_new_character(tracker):
    cid = tracker.track_appearance("Guts", page=1, description="Swordsman", confidence=0.8)
    assert cid.startswith("char_")
    char = tracker.db.get_character(tracker.manga_id, cid)
    assert char["name"] == "Guts"


def test_track_same_character_merges(tracker):
    tracker.track_appearance("Guts", page=1)
    tracker.track_appearance("Guts", page=2)
    chars = tracker.db.get_characters(tracker.manga_id)
    assert len(chars) == 1
    assert chars[0]["appearance_count"] >= 2


def test_track_unknown_isolated(tracker):
    tracker.track_appearance("unknown", page=1)
    tracker.track_appearance("unknown", page=1)
    # Unknowns grouped by page, only one page-1 placeholder
    datas = tracker.db.conn.execute(
        "SELECT * FROM characters WHERE manga_id=?", (tracker.manga_id,)
    ).fetchall()
    assert len(datas) == 1


def test_resolve_fuzzy_matches_existing(tracker):
    tracker.track_appearance("The Black Swordsman", page=1)
    cid = tracker.resolve("Black Swordsman")
    assert cid is not None


def test_merge_unknowns(tracker):
    tracker.track_appearance("unknown", page=1)
    unknown_chars = [c["character_id"] for c in tracker.db.get_characters(tracker.manga_id)
                     if c["name"].startswith("Unknown")]
    cid = tracker.merge_unknowns("Guts", unknown_chars, confidence=0.9)
    chars = tracker.db.get_characters(tracker.manga_id)
    # The unknown is deleted; Guts remains (or was created)
    names = [c["name"] for c in chars]
    assert "Guts" in names


def test_attempt_identification(tracker):
    cid = tracker.track_appearance("Guts", page=1)
    best = tracker.attempt_identification("Guts", 1, [
        {"name": "Guts", "description": "The protagonist"},
        {"name": "Griffith", "description": "The antagonist"},
    ])
    assert best == cid


def test_attempt_identification_ambiguous_returns_none(tracker):
    tracker.track_appearance("Guts", page=1)
    best = tracker.attempt_identification("Mysterious", 1, [
        {"name": "Guts", "description": "Swordsman"},
        {"name": "Griffith", "description": "Leader"},
    ])
    assert best is None
