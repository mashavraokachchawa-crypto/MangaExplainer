"""Tests for hierarchical story memory (story_memory)."""
import tempfile
from pathlib import Path

import pytest

from pipeline.knowledge_db import open_knowledge_db
from pipeline.story_memory import (
    retrieve_relevant_memory,
    get_story_memory,
    summarize_chapter,
    generate_page_summary,
    _short_summary,
)


@pytest.fixture()
def ctx():
    tmp = Path(tempfile.mkdtemp())
    db = open_knowledge_db(tmp)
    mid = db.upsert_manga({"title": "Berserk", "author": "Miura",
                           "genres": ["Action"], "synopsis": "A dark tale."})
    cid = db.add_chapter(mid, {"chapter_number": 1, "title": "Hawks",
                               "pdf_page_start": 1, "pdf_page_end": 20})
    cache = {"manga_id": mid, "chapter_id": cid}
    yield db, mid, cid, cache
    db.close()


def test_generate_page_summary_deterministic(ctx):
    db, mid, cid, cache = ctx
    db.add_event(mid, {"event_type": "page", "page_number": 3,
                       "characters": ["Guts", "Griffith"],
                       "description": "Guts meets Griffith",
                       "importance": 0.8})
    s = generate_page_summary(db, mid, 3, cid)
    assert "Guts" in s["text"]


def test_summarize_chapter_no_llm_uses_deterministic(ctx):
    db, mid, cid, cache = ctx
    db.add_event(mid, {"event_type": "page", "page_number": 1,
                       "chapter_id": cid,
                       "characters": ["Guts"], "description": "Battle begins",
                       "importance": 0.9})
    chapter = db.get_chapters(mid)[0]
    out = summarize_chapter(db, mid, chapter, llm=None, cfg=None)
    assert isinstance(out, dict)
    assert out["short"]
    assert out["detail"]


def test_short_summary_from_events(ctx):
    events = [
        {"description": "Guts fights the monster", "importance": 0.9},
        {"description": "Casca flees", "importance": 0.5},
    ]
    short = _short_summary(events, {"title": "Hawks"})
    assert "Guts" in short


def test_retrieve_relevant_memory_bounded(ctx):
    db, mid, cid, cache = ctx
    db.add_event(mid, {"event_type": "page", "page_number": 1,
                       "characters": ["Guts"], "description": "Event"})
    block = retrieve_relevant_memory(db, mid, page=1, chapter_id=cid, max_chars=2000)
    assert "MANGA:" in block
    assert "Berserk" in block
    assert len(block) <= 2000


def test_retrieve_manga_only_when_no_page(ctx):
    db, mid, cid, cache = ctx
    block = retrieve_relevant_memory(db, mid)
    assert "MANGA:" in block
    assert "THIS PAGE EVENTS:" not in block


def test_get_story_memory_hierarchy(ctx):
    db, mid, cid, cache = ctx
    db.add_event(mid, {"event_type": "page", "page_number": 1,
                       "characters": ["Guts"], "description": "x"})
    mem = get_story_memory(db, mid, page=1, chapter_id=cid)
    assert "manga_level" in mem
    assert "chapter_level" in mem
    assert "page_level" in mem


def test_summarize_chapter_with_llm_prompt(ctx):
    """With an llm object, we call it with the chapter summary prompt."""
    db, mid, cid, cache = ctx
    calls = {}

    class FakeLLM:
        def generate(self, prompt, **kw):
            calls["prompt"] = prompt
            return "A generated chapter summary"

    chapter = db.get_chapters(mid)[0]
    out = summarize_chapter(db, mid, chapter, llm=FakeLLM(), cfg=None)
    assert out["detail"] == "A generated chapter summary"
