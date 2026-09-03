"""Tests for the persistent manga knowledge database (knowledge_db)."""
import sqlite3
import tempfile
from pathlib import Path

import pytest

from pipeline.knowledge_db import open_knowledge_db, DB_FILENAME


@pytest.fixture()
def db():
    tmp = Path(tempfile.mkdtemp())
    database = open_knowledge_db(tmp)
    yield database
    database.close()


def test_create_manga(db):
    mid = db.upsert_manga({"title": "Berserk", "author": "Miura"})
    assert mid.startswith("manga_")
    manga = db.get_manga(mid)
    assert manga["title"] == "Berserk"
    assert manga["author"] == "Miura"


def test_upsert_merges_without_overwrite(db):
    mid = db.upsert_manga({"title": "Berserk", "author": "Miura", "status": ""})
    db.upsert_manga({"title": "Berserk", "status": "ongoing"})
    manga = db.get_manga(mid)
    # Empty existing title/author are NOT overwritten with empty; status filled
    assert manga["title"] == "Berserk"
    assert manga["status"] == "ongoing"


def test_chapter_crud(db):
    mid = db.upsert_manga({"title": "Berserk"})
    c1 = db.add_chapter(mid, {
        "chapter_number": 1, "title": "Chapter One",
        "pdf_page_start": 1, "pdf_page_end": 50, "confidence": 0.9,
    })
    c2 = db.add_chapter(mid, {
        "chapter_number": 2, "title": "Chapter Two",
        "pdf_page_start": 51, "pdf_page_end": 100, "confidence": 0.8,
    })
    chapters = db.get_chapters(mid)
    assert len(chapters) == 2
    assert chapters[0]["pdf_page_start"] == 1
    assert chapters[1]["pdf_page_start"] == 51

    assert db.chapter_for_page(mid, 25)["id"] == c1
    assert db.chapter_for_page(mid, 60)["id"] == c2
    assert db.chapter_for_page(mid, 200) is None


def test_character_merge_and_alias(db):
    mid = db.upsert_manga({"title": "Berserk"})
    cid = db.add_character(mid, {"name": "Guts", "role": "protagonist", "first_page": 1})
    assert cid.startswith("char_")

    # Reinforce on a later page
    db.add_character(mid, {
        "character_id": cid, "name": "Guts", "last_page": 50,
        "description": "A swordsman", "appearance_count": 2,
    })
    char = db.get_character(mid, cid)
    assert char["appearance_count"] == 2  # 1 initial + 1 reinforce
    assert char["first_page"] == 1

    # Resolve via alias
    assert db.resolve_character(mid, "Guts") == cid


def test_character_alias_resolution(db):
    mid = db.upsert_manga({"title": "Berserk"})
    cid = db.add_character(mid, {"name": "Mercenary", "aliases": ["Gattsu"], "first_page": 1})
    assert db.resolve_character(mid, "Mercenary") == cid
    assert db.resolve_character(mid, "Gattsu") == cid


def test_events_queries(db):
    mid = db.upsert_manga({"title": "Berserk"})
    db.add_event(mid, {"event_type": "page", "page_number": 1,
                       "characters": ["Guts"], "description": "Enters",
                       "importance": 0.9})
    db.add_event(mid, {"event_type": "page", "page_number": 2,
                       "characters": ["Griffith"], "description": "Fights",
                       "importance": 0.4})
    all_events = db.get_events(mid)
    assert len(all_events) == 2
    page1 = db.get_events(mid, page=1)
    assert len(page1) == 1
    assert page1[0]["description"] == "Enters"
    important = db.get_events(mid, min_importance=0.5)
    assert len(important) == 1


def test_summary_crud(db):
    mid = db.upsert_manga({"title": "Berserk"})
    cid = db.add_chapter(mid, {"chapter_number": 1, "pdf_page_start": 1, "pdf_page_end": 5})
    sid = db.add_summary(mid, {
        "summary_type": "chapter_detail", "chapter_id": cid,
        "text": "Chapter about...", "important_events": ["Event A", "Event B"],
    })
    summaries = db.get_summaries(mid, "chapter_detail", cid)
    assert len(summaries) == 1
    assert summaries[0]["important_events"] == ["Event A", "Event B"]


def test_source_evidence(db):
    mid = db.upsert_manga({"title": "Berserk"})
    db.add_source_evidence(mid, {
        "entity_type": "character", "entity_key": "char_1",
        "source_type": "vlm", "pdf_page": 1, "detail": "Guts present",
        "confidence": 0.9,
    })
    ev = db.get_source_evidence(mid, "character", "char_1")
    assert len(ev) == 1
    assert ev[0]["source_type"] == "vlm"
    assert ev[0]["pdf_page"] == 1


def test_conflicts_flow(db):
    mid = db.upsert_manga({"title": "Berserk", "status": "ongoing"})
    cid = db.add_conflict(mid, {
        "entity_type": "metadata", "entity_key": "status",
        "field_name": "value", "value_a": "ongoing", "source_a": "internet",
        "value_b": "completed", "source_b": "pdf",
    })
    conflicts = db.get_unresolved_conflicts(mid)
    assert len(conflicts) == 1
    db.resolve_conflict(cid, "ongoing", "user")
    assert len(db.get_unresolved_conflicts(mid)) == 0


def test_research_cache(db):
    mid = db.upsert_manga({"title": "Berserk"})
    db.cache_research(mid, "book_ref", "http://x", {"title": "Berserk"})
    cached = db.get_cached_research(mid, "book_ref", "http://x")
    assert cached == {"title": "Berserk"}


def test_checkpoints(db):
    mid = db.upsert_manga({"title": "Berserk"})
    assert db.checkpoint_status(mid, "extract", 1) == "pending"
    db.checkpoint_set(mid, "extract", "running", page=1)
    assert db.checkpoint_status(mid, "extract", 1) == "running"
    db.checkpoint_set(mid, "extract", "completed", page=1)
    assert db.checkpoint_status(mid, "extract", 1) == "completed"
    # different key
    assert db.checkpoint_status(mid, "extract", 2) == "pending"


def test_stats(db):
    mid = db.upsert_manga({"title": "Berserk"})
    db.add_character(mid, {"name": "Guts"})
    db.add_event(mid, {"event_type": "page", "page_number": 1, "description": "x"})
    stats = db.stats(mid)
    assert stats["manga"] == 1
    assert stats["characters"] == 1
    assert stats["events"] == 1


def test_wipe_manga(db):
    mid = db.upsert_manga({"title": "Berserk"})
    db.add_character(mid, {"name": "Guts"})
    db.wipe_manga(mid)
    assert db.stats(mid)["characters"] == 0
    assert db.get_manga(mid) is None


def test_persistence_across_reopen():
    """Data survives closing and reopening the database."""
    tmp = Path(tempfile.mkdtemp())
    db = open_knowledge_db(tmp)
    mid = db.upsert_manga({"title": "Berserk"})
    db.add_character(mid, {"name": "Guts"})
    db.close()

    db2 = open_knowledge_db(tmp)
    assert db2.get_manga(mid)["title"] == "Berserk"
    assert len(db2.get_characters(mid)) == 1
    db2.close()


def test_db_file_location():
    """The DB file is stable at state/manga_knowledge.db."""
    assert DB_FILENAME == "manga_knowledge.db"
