"""Tests: compulsory "What manga is this?" — name -> internet -> durable memory.

Replaced the old reader-question loop. Covers the synopsis character-name
extractor (internet_ref._extract_names) and that the fetched characters persist
into the Manga Memory CHARACTER store alongside the BOOK record.
"""
import json
from pathlib import Path

from config.loader import Config
from pipeline.internet_ref import _extract_names, fetch_book_ref, book_ref_to_text
from pipeline.manga_memory.book import BookMemory
from pipeline.manga_memory.character import CharacterMemory
from pipeline.manga_memory.store import open_memory


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
        "synopsis": "Luffy dreams of becoming Pirate King. Luffy sets sail "
                    "with his crew. Zoro joins Luffy on the Grand Line. "
                    "Nami navigates, Sanji cooks, and Luffy chases treasure.",
        "url": "https://mangadex.org/title/abcd-1234",
    }
    base.update(over)
    return base


def test_extract_names_picks_recurring_person_like_caps():
    # Luffy recurs, Zoro recurs; Nami appears once. Author name "Eiichiro Oda"
    # / "Pirate King" stop words must NOT leak into characters.
    text = ("Luffy dreams of becoming Pirate King. Luffy sets sail with his "
            "crew. Zoro joins Luffy on the Grand Line. Nami navigates. "
            "Zoro draws his sword and Luffy laughs.")
    names = _extract_names(text)
    assert "Luffy" in names
    assert "Zoro" in names
    assert "Nami" not in names  # single mention -> not a remembered character
    assert not {"Pirate", "King", "Grand", "Line"} & set(names)
    # lone non-recurring words never qualify
    assert _extract_names("Alpha meets Beta once and nothing recurs.") == []


def test_extract_names_never_raises_and_caps():
    assert _extract_names("") == []
    assert _extract_names("a" * 5) == []
    assert _extract_names("short") == []
    # only single occurrences -> nothing
    assert _extract_names("Alpha meets Beta once.") == []
    many = "Mono appears. Mono again. Mono thrice. Mono four. Mono five. Mono six. Mono seven. Mono eight. Mono nine. Mono ten. Mono eleven. Mono twelve. Mono thirteen. Mono fourteen. Mono fifteen. Mono sixteen. Mono seventeen. Mono eighteen. Mono nineteen. Mono twenty."
    assert len(_extract_names(many)) <= 15


def test_fetch_book_ref_adds_characters_from_synopsis():
    # a real-shaped synopsis that recurs the protagonist
    info = {"source": "mangadex", "title": "Berserk",
            "synopsis": "Guts wields a massive sword. Guts hunts apostles. "
                        "Guts protects Casca. Guts is the Black Swordsman."}
    out = fetch_book_ref.__wrapped__ if hasattr(fetch_book_ref, "__wrapped__") else None
    # note: fetch_book_ref hits the network; exercise the attach logic directly
    from pipeline.internet_ref import _with_characters
    enriched = _with_characters(info)
    assert enriched.get("characters")
    assert "Guts" in enriched["characters"]


def test_book_remember_keeps_characters(tmp_path):
    cfg = make_cfg(tmp_path)
    memory = open_memory(cfg, lazy=True).load_all()
    info = _info(characters=["Luffy", "Zoro", "Nami", "Luffy"])
    rec = BookMemory(memory.store_for("book")).remember(info)
    memory.save_all()
    # deduped by _clean
    assert rec.value["characters"] == ["Luffy", "Zoro", "Nami"]
    assert rec.value["genres"] == ["Action", "Adventure"]


def test_book_fetch_learns_characters_into_store(tmp_path):
    cfg = make_cfg(tmp_path)
    info = _info(characters=["Luffy", "Zoro", "Nami", "Luffy"])
    memory = open_memory(cfg, lazy=True).load_all()
    BookMemory(memory.store_for("book")).remember(info)
    chars = CharacterMemory(memory.store_for("character"))
    for name in info["characters"]:
        if not chars.record(name):
            chars.learn(name, source="internet:book:mangadex", confidence=0.9)
    memory.save_all()
    assert memory.store_for("book").count() == 1
    assert memory.store_for("character").count() == 3
    assert chars.record("Luffy") is not None


def test_book_ref_text_human_line():
    line = book_ref_to_text(_info())
    assert "One Piece" in line
    assert "Eiichiro Oda" in line
    assert "[Action, Adventure]" in line