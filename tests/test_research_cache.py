"""Tests for internet research caching (research_cache)."""
import tempfile
from pathlib import Path

import pytest

from pipeline.knowledge_db import open_knowledge_db
from pipeline.research_cache import (
    ResearchCache,
    store_book_reference,
    store_character_list,
)


@pytest.fixture()
def ctx():
    tmp = Path(tempfile.mkdtemp())
    db = open_knowledge_db(tmp)
    mid = db.upsert_manga({"title": "Berserk"})
    cache = ResearchCache(db, mid)
    yield db, mid, cache
    db.close()


def test_fetch_caches_on_miss(ctx):
    db, mid, cache = ctx
    calls = []

    def fetcher():
        calls.append(1)
        return {"title": "Berserk"}

    r1 = cache.fetch("book_ref:Berserk", fetcher)
    r2 = cache.fetch("book_ref:Berserk", fetcher)
    assert r1["title"] == "Berserk"
    assert r2["title"] == "Berserk"
    assert len(calls) == 1  # second call served from cache


def test_fetch_force_refreshes(ctx):
    db, mid, cache = ctx
    calls = []

    def fetcher(v):
        calls.append(1)
        return {"v": v}

    cache.fetch("k", lambda: fetcher(1))
    cache.fetch("k", lambda: fetcher(2), force=True)
    assert len(calls) == 2


def test_fetch_returns_none_on_exception(ctx):
    db, mid, cache = ctx

    def boom():
        raise RuntimeError("network")

    assert cache.fetch("k", boom) is None


def test_store_book_reference(ctx):
    db, mid, cache = ctx
    book = {"title": "Berserk", "author": "Miura", "genres": ["Action"]}
    store_book_reference(db, mid, book, cache)
    row = db.get_cached_research(mid, "book_ref:Berserk")
    assert row is not None
    assert row["title"] == "Berserk"


def test_store_character_list(ctx):
    db, mid, cache = ctx
    chars = [{"name": "Guts", "role": "Protagonist"},
             {"name": "Griffith", "role": "Antagonist"}]
    count = store_character_list(db, mid, chars, cache)
    assert count == 2
    assert len(db.get_characters(mid)) == 2


def test_store_character_list_empty(ctx):
    db, mid, cache = ctx
    assert store_character_list(db, mid, [], cache) == 0
