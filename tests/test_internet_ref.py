"""Tests: automatic book reference fetching (pipeline.internet_ref).

Covers MangaDex parsing, the Wikipedia fallback, total failure -> None, and the
compact one-line rendering. All network calls are stubbed at the ``_get_json``
boundary so the tests are hermetic.
"""
import pytest

from config.loader import Config
from pipeline import internet_ref


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
    }
    return Config(data, tmp_path)


def _mangadex_payload():
    return {
        "data": [{
            "id": "abcd-1234",
            "attributes": {
                "title": {"en": "One Piece"},
                "altTitles": [{"ja": "ワンピース"}],
                "description": {"en": "Luffy dreams of becoming Pirate King. " * 10},
                "year": 1997,
                "status": "ongoing",
                "publicationDemographic": "shounen",
                "originalLanguage": "ja",
                "tags": [
                    {"id": "t1", "attributes": {"name": {"en": "Action"}}},
                    {"id": "t2", "attributes": {"name": {"en": "Adventure"}}},
                ],
            },
            "relationships": [
                {"type": "author", "attributes": {"name": {"en": "Eiichiro Oda"}}},
                {"type": "cover_art", "attributes": {"fileName": "cover-001.png"}},
            ],
        }],
    }


@pytest.fixture(autouse=True)
def _stub_http(monkeypatch):
    """Route _get_json by URL: MangaDex serves the payload, Wikipedia fails."""
    calls = {"n": 0}

    def fake(url, timeout):
        calls["n"] += 1
        if url.startswith(internet_ref.MANGA_DEX):
            return _mangadex_payload()
        return None  # wikipedia never reached in the happy path

    monkeypatch.setattr(internet_ref, "_get_json", fake)
    yield calls


def test_mangadex_hit(monkeypatch):
    info = internet_ref.fetch_book_ref("One Piece")
    assert info is not None
    assert info["source"] == "mangadex"
    assert info["title"] == "One Piece"
    assert info["authors"] == ["Eiichiro Oda"]
    assert info["genres"] == ["Action", "Adventure"]
    assert info["demographic"] == "shounen"
    assert info["status"] == "ongoing"
    assert info["year"] == "1997"
    assert info["url"] == "https://mangadex.org/title/abcd-1234"
    assert info["cover_url"].startswith("https://uploads.mangadex.org/covers/")


def test_mangadex_prefers_exact_title(monkeypatch):
    """The top hit may be a spinoff; an exact title match in the results wins."""
    spinoff = _mangadex_payload()
    canonical = _mangadex_payload()
    spinoff["data"][0]["attributes"]["title"] = {"en": "One Piece Academy"}
    canonical["data"][0]["attributes"]["title"] = {"en": "One Piece"}
    payload = {"data": spinoff["data"] + canonical["data"]}

    def fake(url, timeout):
        return payload

    monkeypatch.setattr(internet_ref, "_get_json", fake)
    info = internet_ref.fetch_book_ref("One Piece")
    assert info["title"] == "One Piece"


def test_wikipedia_fallback(monkeypatch):
    def fake(url, timeout):
        if url.startswith(internet_ref.MANGA_DEX):
            return {"data": []}
        if "action=query" in url:
            return {"query": {"search": [{"title": "Nausicaa of the Valley of the Wind"}]}}
        if "/page/summary/" in url:
            return {
                "title": "Nausicaa of the Valley of the Wind",
                "extract": "Nausicaa is a 1982 manga by Hayao Miyazaki. " * 20,
                "content_urls": {
                    "desktop": {"page": "https://en.wikipedia.org/wiki/Nausicaa_(manga)"}
                },
                "thumbnail": {"source": "https://upload.wikimedia.org/th.jpg"},
            }
        return None

    monkeypatch.setattr(internet_ref, "_get_json", fake)
    info = internet_ref.fetch_book_ref("Nausicaa")
    assert info is not None
    assert info["source"] == "wikipedia"
    assert "Nausicaa" in info["title"]
    assert "Miyazaki" in info["synopsis"]
    assert "wikipedia.org/wiki/Nausicaa" in info["url"]


def test_both_sources_fail_returns_none(monkeypatch):
    def boom(url, timeout):
        raise OSError("no network")

    monkeypatch.setattr(internet_ref, "_get_json", boom)
    assert internet_ref.fetch_book_ref("Something Not Real") is None


def test_empty_title_returns_none():
    assert internet_ref.fetch_book_ref("   ") is None
    assert internet_ref.fetch_book_ref("") is None


def test_book_ref_to_text():
    info = {
        "title": "One Piece",
        "authors": ["Eiichiro Oda"],
        "genres": ["Action", "Adventure"],
        "year": "1997",
        "status": "ongoing",
    }
    text = internet_ref.book_ref_to_text(info)
    assert "One Piece" in text
    assert "1997" in text
    assert "Eiichiro Oda" in text
    assert internet_ref.book_ref_to_text(None) == ""