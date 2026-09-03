"""Tests for source-aware verification / conflict resolution (verification)."""
import tempfile
from pathlib import Path

import pytest

from pipeline.knowledge_db import open_knowledge_db
from pipeline import verification as V


@pytest.fixture()
def db():
    tmp = Path(tempfile.mkdtemp())
    database = open_knowledge_db(tmp)
    yield database
    database.close()


def test_register_new_evidence_ok(db):
    mid = db.upsert_manga({"title": "Berserk", "status": "ongoing"})
    r = V.register_evidence(db, mid, "metadata", "title", "Berserk", "internet")
    assert r["ok"] is True
    assert r.get("recorded") is True


def test_register_same_value_no_conflict(db):
    mid = db.upsert_manga({"title": "Berserk", "status": "ongoing"})
    V.register_evidence(db, mid, "metadata", "status", "ongoing", "internet")
    V.register_evidence(db, mid, "metadata", "status", "ongoing", "pdf")
    conflicts = db.get_unresolved_conflicts(mid)
    assert len(conflicts) == 0


def test_metadata_initial_source_is_pdf_when_created(db):
    mid = db.upsert_manga({"title": "Berserk", "author": "Miura"})
    src = V._source_of_metadata(db, mid, "author")
    assert src == "pdf"  # created by the pipeline, assumed pdf

def test_conflicting_value_creates_conflict(db):
    mid = db.upsert_manga({"title": "Berserk", "status": "ongoing"})
    # title already 'Berserk'; register a differing internet value -> conflict
    r = V.register_evidence(db, mid, "metadata", "title", "Berserk 2", "internet")
    assert r["ok"] is False
    assert r["reason"] == "new_conflict"
    conflicts = db.get_unresolved_conflicts(mid)
    assert len(conflicts) == 1


def test_resolve_conflict_prefer_source(db):
    mid = db.upsert_manga({"title": "Berserk", "status": "ongoing"})
    cid = db.add_conflict(mid, {
        "entity_type": "metadata", "entity_key": "status",
        "field_name": "value", "value_a": "ongoing", "source_a": "internet",
        "value_b": "completed", "source_b": "pdf",
    })
    res = V.resolve_conflict_prefer(db, mid, cid, preferred_source="pdf")
    assert res["resolved_by"] == "auto"
    assert len(db.get_unresolved_conflicts(mid)) == 0


def test_resolve_user_source_wins(db):
    mid = db.upsert_manga({"title": "Berserk", "status": "ongoing"})
    cid = db.add_conflict(mid, {
        "entity_type": "metadata", "entity_key": "status",
        "field_name": "value", "value_a": "canceled", "source_a": "user",
        "value_b": "ongoing", "source_b": "pdf",
    })
    res = V.resolve_conflict_prefer(db, mid, cid, preferred_source="pdf")
    assert res["resolved"] == "canceled"  # user wins over pdf
    assert res["resolved_by"] == "user"


def test_user_correction_sets_value_and_resolves_conflicts(db):
    mid = db.upsert_manga({"title": "Berserk", "status": "ongoing"})
    db.add_conflict(mid, {
        "entity_type": "metadata", "entity_key": "status",
        "field_name": "value", "value_a": "ongoing", "source_a": "internet",
        "value_b": "completed", "source_b": "pdf",
    })
    res = V.user_correction(db, mid, "metadata", "status", "canceled")
    assert res["ok"] is True
    assert db.get_manga(mid)["status"] == "canceled"
    assert len(db.get_unresolved_conflicts(mid)) == 0


def test_materially_different_empty_is_not_conflict(db):
    mid = db.upsert_manga({"title": "Berserk"})  # status default ""
    diff = V._materially_different(db, mid, "metadata", "status", "ongoing")
    assert diff is None  # empty existing not a real conflict


def test_materially_different_real_disagreement(db):
    mid = db.upsert_manga({"title": "Berserk", "status": "ongoing"})
    diff = V._materially_different(db, mid, "metadata", "status", "finished")
    assert diff is not None
    assert diff["existing"] == "ongoing"
