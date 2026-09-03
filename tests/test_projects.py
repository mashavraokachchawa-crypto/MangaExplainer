"""Tests: project registry (home-screen projects) + character imagery helpers.

Covers the lightweight multi-project store (state/projects.json): upsert,
toggle persistence, delete, and corruption tolerance; plus the offline-safe
slugs / image-store layout of the character-image fetcher (network calls are
best-effort and never exercised here — we only test the pure helpers).
"""
from pathlib import Path

from pipeline import project_registry as pr
from pipeline import character_images as ci


def _tmp(tmp_path):
    return str(Path(tmp_path))


def test_upsert_and_list_newest_first(tmp_path):
    sd = _tmp(tmp_path)
    pr.upsert_project(sd, {"slug": "one-piece", "name": "One Piece",
                           "toggles": pr.default_toggles()})
    import time
    time.sleep(0.01)
    pr.upsert_project(sd, {"slug": "berserk", "name": "Berserk",
                           "toggles": pr.default_toggles()})
    projects = pr.list_projects(sd)
    assert [p["slug"] for p in projects] == ["berserk", "one-piece"]
    assert pr.projects_path(sd).is_file()


def test_toggle_persists(tmp_path):
    sd = _tmp(tmp_path)
    pr.upsert_project(sd, {"slug": "one-piece", "name": "One Piece",
                           "toggles": pr.default_toggles()})
    rec = pr.set_toggle(sd, "one-piece", "music", True)
    assert rec["toggles"]["music"] is True
    assert rec["toggles"]["tts"] is True
    # re-read from disk
    loaded = pr.get_project(sd, "one-piece")
    assert loaded["toggles"]["music"] is True


def test_set_toggle_unknown_key_or_slug_is_noop(tmp_path):
    sd = _tmp(tmp_path)
    assert pr.set_toggle(sd, "nope", "tts", True) is None
    pr.upsert_project(sd, {"slug": "a", "name": "A", "toggles": pr.default_toggles()})
    assert pr.set_toggle(sd, "a", "bogus", True) is None
    assert pr.get_project(sd, "a")["toggles"] == {"tts": True, "music": False}


def test_delete_project(tmp_path):
    sd = _tmp(tmp_path)
    pr.upsert_project(sd, {"slug": "a", "name": "A"})
    assert pr.delete_project(sd, "a") is True
    assert pr.delete_project(sd, "a") is False  # already gone
    assert pr.list_projects(sd) == []


def test_corrupt_or_missing_file_tolerated(tmp_path):
    sd = _tmp(tmp_path)
    assert pr.list_projects(sd) == []  # missing file -> []
    pr.projects_path(sd).write_text("{ not json", encoding="utf-8")
    assert pr.list_projects(sd) == []
    assert pr.get_project(sd, "x") is None


def test_slug_and_image_layout():
    assert ci._slugify("One Piece") == "one-piece"
    assert ci._slugify("  Berserk!!  ") == "berserk"
    # image store lives under state/manga_memory/images/<slug>
    assert ci.image_store_dir("state").name == "images"
    assert ci.image_store_dir("state").parent.name == "manga_memory"


def test_slug_default_when_empty():
    assert ci._slugify("") == "manga"