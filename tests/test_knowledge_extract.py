"""Tests for the PDF->DB knowledge extraction bridge (knowledge_extract)."""
import json
import tempfile
from pathlib import Path

import pytest

from pipeline.knowledge_db import open_knowledge_db
from pipeline import knowledge_extract as KE


class FakeNode:
    def __init__(self, **kw):
        self.__dict__.update(kw)


@pytest.fixture()
def ctx():
    tmp = Path(tempfile.mkdtemp())
    analysis = tmp / "analysis"
    analysis.mkdir()
    db = open_knowledge_db(tmp)
    mid = db.upsert_manga({"title": "Berserk"})

    page_data = {
        "panels": [
            {
                "panel_id": "p1",
                "visual": {
                    "characters": [
                        {"name": "Guts", "description": "A large swordsman",
                         "role": "protagonist", "action": "swinging", "emotion": "angry"},
                        {"name": " ", "description": "mysterious figure",
                         "action": "", "emotion": ""},
                    ],
                    "environment": "dark forest",
                    "important_event": "Guts battles a demon",
                    "objects": [{"name": "Dragon Slayer", "description": "huge blade"}],
                    "confidence": 0.9,
                },
            }
        ]
    }
    (analysis / "page_001_knowledge.json").write_text(json.dumps(page_data), "utf-8")

    cfg = FakeNode(output=FakeNode(analysis_dir=str(analysis)))
    afford = {"tmp": tmp, "analysis": analysis, "db": db, "mid": mid, "cfg": cfg}
    yield afford
    db.close()


def test_extract_page_knowledge_loads(ctx):
    data = KE._extract_page_knowledge(ctx["cfg"], 1)
    assert data is not None
    assert len(data["panels"]) == 1


def test_extract_page_knowledge_missing_returns_none(ctx):
    assert KE._extract_page_knowledge(ctx["cfg"], 99) is None


def test_extract_characters_from_page(ctx):
    data = KE._extract_page_knowledge(ctx["cfg"], 1)
    chars = KE.extract_characters_from_page(data, 1)
    # Guts named + unknown placeholder
    names = [c["name"] for c in chars]
    assert "Guts" in names
    assert any(n.startswith("Unknown_") for n in names)


def test_extract_locations_from_page(ctx):
    data = KE._extract_page_knowledge(ctx["cfg"], 1)
    locs = KE.extract_locations_from_page(data, 1)
    assert any(l["name"] == "dark forest" for l in locs)


def test_extract_events_from_page(ctx):
    data = KE._extract_page_knowledge(ctx["cfg"], 1)
    events = KE.extract_events_from_page(data, 1)
    assert any("Guts battles" in e["description"] for e in events)


def test_extract_objects_from_page(ctx):
    from pipeline.knowledge_extract import extract_objects_from_page
    data = KE._extract_page_knowledge(ctx["cfg"], 1)
    objs = extract_objects_from_page(data, 1)
    assert any(o["name"] == "Dragon Slayer" for o in objs)


def test_ingest_page_roundtrip(ctx):
    db, mid, cfg = ctx["db"], ctx["mid"], ctx["cfg"]
    result = KE.ingest_page_to_knowledge_db(db, mid, cfg, 1)
    assert result["status"] == "ok"
    assert result["characters"] >= 2  # Guts + unknown
    assert result["locations"] >= 1
    assert result["events"] >= 1
    assert result["objects"] >= 1
    # checkpoint marked
    assert db.checkpoint_status(mid, "extract", 1) == "completed"


def test_ingest_page_missing_knowledge(ctx):
    db, mid, cfg = ctx["db"], ctx["mid"], ctx["cfg"]
    result = KE.ingest_page_to_knowledge_db(db, mid, cfg, 99)
    assert result["status"] == "no_knowledge"


def test_run_full_extraction(ctx):
    db, mid, cfg = ctx["db"], ctx["mid"], ctx["cfg"]
    # Also write page 2 with knowledge to test multi-page
    (ctx["analysis"] / "page_002_knowledge.json").write_text(json.dumps({
        "panels": [{"panel_id": "q1",
                    "visual": {"characters": [{"name": "Casca", "description": "x",
                                               "role": "fighter"}],
                               "environment": "camp", "important_event": "Casca arrives",
                               "confidence": 0.7}}],
    }), "utf-8")
    result = KE.run_full_extraction(db, mid, cfg, [1, 2])
    assert result["pages_processed"] == 2
    assert result["characters"] >= 3
