"""Offline tests for the PDF identity scan + clear-all backend behaviour.

These avoid opening/rendering the whole PDF (that needs an input file, and
we must not depend on the user's real manga.pdf). We test the pure title
heuristics (`_strip_volume_markers`, `_pick_title`) and that `clear_all`
resets the registry + memory + images via a throwaway state dir.
"""
import json
from pathlib import Path

from pipeline import pdf_scan


def test_strip_volume_markers_basic():
    assert pdf_scan._strip_volume_markers(
        "Berserk v01 (2003) (Digital) (Cyborgzx-repack)") == "Berserk"
    assert pdf_scan._strip_volume_markers("Naruto - Volume 1") == "Naruto"
    assert pdf_scan._strip_volume_markers("Attack on Titan (2012)") == "Attack on Titan"
    assert pdf_scan._strip_volume_markers("My Hero Academia - Volume 38") == "My Hero Academia"


def test_extract_volume():
    assert pdf_scan.extract_volume("Berserk v01 (2003) (Digital) (Cyborgzx-repack)") == 1
    assert pdf_scan.extract_volume("Berserk Volume 3") == 3
    assert pdf_scan.extract_volume("Naruto - Vol. 12") == 12
    assert pdf_scan.extract_volume("One Piece, #105") == 105


def test_extract_volume_none_when_absent():
    assert pdf_scan.extract_volume("One Piece") == 0
    assert pdf_scan.extract_volume("") == 0
    assert pdf_scan.extract_volume("The Berserk") == 0


def test_strip_volume_markers_keeps_author_disambiguation():
    assert pdf_scan._strip_volume_markers("Monster (Naoki Urasawa) v2") == "Monster (Naoki Urasawa)"


def test_strip_volume_markers_drops_header_noise():
    assert pdf_scan._strip_volume_markers("Dragon Ball Z vol03 chapter 120") == "Dragon Ball Z"
    assert pdf_scan._strip_volume_markers("OceanofPDF.com Berserk Chapter 301") == "Berserk"
    assert pdf_scan._strip_volume_markers("One Piece, Vol. 105") == "One Piece"


def test_pick_title_prefers_rich_metadata():
    lines = ["copyright (c) 1997", "OceanofPDF.com", "A Long Winded Sentence That Is A Blurb"]
    title, source = pdf_scan._pick_title("Berserk v01 (2003) (Digital)", lines)
    assert title == "Berserk"
    assert source == "metadata"


def test_pick_title_falls_back_to_embedded_text():
    # no metadata title -> a short Title Case line wins over footer noise
    title, source = pdf_scan._pick_title("", ["OceanofPDF.com", "One Piece"])
    assert title == "One Piece"
    assert source == "embedded-text"


def test_pick_title_empty_when_only_noise():
    title, source = pdf_scan._pick_title("", ["www.example.com", "copyright 2020", "volume 3"])
    assert title == ""
    assert source == ""


def test_scan_pdf_missing_file():
    r = pdf_scan.scan_pdf("definitely/missing.pdf")
    assert r["ok"] is False
    assert r["page_count"] == 0
    assert "not found" in r["reason"]


def test_clear_all_wipes_projects_memory_images():
    """clear_all (via a fresh state dir) empties registry + memory + images."""
    import shutil
    from webui import _clear_all
    from pipeline import project_registry
    from pipeline.character_images import image_store_dir

    state_dir = Path("state") / "_test_clear"
    shutil.rmtree(state_dir, ignore_errors=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    # seed a project + a memory record + an image file
    project_registry.upsert_project(state_dir, {
        "slug": "berserk", "name": "Berserk",
        "toggles": {"tts": True, "music": False},
    })
    base = state_dir / "manga_memory"
    base.mkdir(parents=True, exist_ok=True)
    (base / "characters.json").write_text(
        json.dumps({"records": [{"key": "guts", "value": {}}], "version": 1}),
        encoding="utf-8")
    img = image_store_dir(state_dir) / "berserk" / "cover.jpg"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"fake")

    class _Cfg:
        pass
    st = _Cfg()
    st.dir = str(state_dir)
    pl = _Cfg()
    pl.state = st
    cfg = _Cfg()
    cfg.pipeline = pl
    res = _clear_all(cfg)

    assert res["projects"] == 1
    assert res["images"] >= 1
    assert project_registry.list_projects(state_dir) == []
    char = json.loads((base / "characters.json").read_text())
    assert char["records"] == []

    shutil.rmtree(state_dir, ignore_errors=True)