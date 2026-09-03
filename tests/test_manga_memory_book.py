"""Tests: durable BOOK memory in the Manga Memory Engine.

Covers the store wiring (book.json, kind aliases, counts), BookMemory upsert +
refresh semantics, and that book records flow into retrieval + the prompt
context block as a BOOK-tagged line.
"""
import json
from pathlib import Path

import pytest

from config.loader import Config
from pipeline.manga_memory.book import BookMemory
from pipeline.manga_memory.context_builder import build_memory_block
from pipeline.manga_memory.models import VerificationState
from pipeline.manga_memory.retrieval import MemoryRetriever
from pipeline.manga_memory.store import FILE_BY_KIND, memory_info, open_memory


def make_cfg(tmp_path):
    data = {
        "input": {"pdf": str(tmp_path / "input" / "manga.pdf")},
        "output": {
            "dir": str(tmp_path / "output"),
            "pages_dir": str(tmp_path / "pages"),
            "panels_dir": str(tmp_path / "panels"),
            "clean_dir": str(tmp_path / "panels_clean"),
            "ocr_dir": str(tmp_path / "ocr"),
            "analysis_dir": str(tmp_path / "analysis"),
            "scenes_dir": str(tmp_path / "scenes"),
            "script_dir": str(tmp_path / "script"),
            "audio_dir": str(tmp_path / "audio"),
            "shots_dir": str(tmp_path / "shots"),
            "crops_dir": str(tmp_path / "crops"),
        },
        "images": {"format": "jpg"},
        "pipeline": {
            "state": {"dir": str(tmp_path / "state")},
            "cache": {"dir": str(tmp_path / "state" / "cache")},
        },
        "logging": {"log_dir": str(tmp_path / "logs")},
        "memory": {"window_size": 10},
    }
    return Config(data, tmp_path)


def _info(**over):
    base = {
        "source": "mangadex",
        "title": "One Piece",
        "authors": ["Eiichiro Oda"],
        "genres": ["Action", "Adventure"],
        "demographic": "shounen",
        "status": "ongoing",
        "year": "1997",
        "synopsis": "Luffy dreams of becoming Pirate King.",
        "url": "https://mangadex.org/title/abcd-1234",
    }
    base.update(over)
    return base


def test_store_file_and_counts(tmp_path):
    cfg = make_cfg(tmp_path)
    memory = open_memory(cfg, lazy=True).load_all()
    BookMemory(memory.store_for("book")).remember(_info())
    memory.save_all()
    # book.json exists on disk
    path = Path(tmp_path / "state" / "manga_memory" / FILE_BY_KIND["book"])
    assert path.is_file()
    doc = json.loads(path.read_text("utf-8"))
    assert doc["records"][0]["kind"] == "book"
    assert memory_info(cfg)["books"] == 1


def test_remember_is_verified_and_merged(tmp_path):
    cfg = make_cfg(tmp_path)
    memory = open_memory(cfg, lazy=True).load_all()
    book = BookMemory(memory.store_for("book"))
    rec = book.remember(_info())
    memory.save_all()
    assert rec.state == VerificationState.VERIFIED
    assert rec.confidence >= 0.9
    assert rec.key == "book::one-piece"
    # a later refresh with the same title + a synopsis fills the gap, keeps url
    rec2 = book.remember(_info(synopsis="", url="https://mangadex.org/title/abcd-1234"))
    assert rec2.key == rec.key
    assert rec2.value["synopsis"] == "Luffy dreams of becoming Pirate King."
    assert rec2.value["url"] == "https://mangadex.org/title/abcd-1234"


def test_retrieval_and_context_include_book(tmp_path):
    cfg = make_cfg(tmp_path)
    memory = open_memory(cfg, lazy=True).load_all()
    BookMemory(memory.store_for("book")).remember(_info())
    memory.save_all()
    # retriever surfaces it
    found = [r.key for r in MemoryRetriever(memory, task="narration").retrieve(limit=10)]
    assert "book::one-piece" in found
    # prompt context block renders a BOOK line
    block = build_memory_block(memory, task="narration")
    assert "[BOOK]" in block
    assert "Eiichiro Oda" in block
    assert "One Piece" in block


def test_remember_requires_title(tmp_path):
    cfg = make_cfg(tmp_path)
    memory = open_memory(cfg, lazy=True).load_all()
    with pytest.raises(ValueError):
        BookMemory(memory.store_for("book")).remember({"year": 1997})


def test_kind_alias_accepts_books(tmp_path):
    from pipeline.manga_memory.store import _normalize_kind
    assert _normalize_kind("books") == "book"
    assert _normalize_kind("book") == "book"