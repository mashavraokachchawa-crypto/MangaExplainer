"""Tests: project + page-window memory (context_memory).

Covers the two independent memories: a durable PROJECT memory that survives
a resume / a new PDF in the same project (characters, places, objects), and
a rolling last-10-pages WINDOW used as short-range narration context.
"""
import json
from pathlib import Path

from config.loader import Config
from pipeline.context_memory import (
    DEFAULT_WINDOW_SIZE,
    PageWindow,
    ProjectMemory,
    build_page_summary,
    memory_info,
    remember_project,
    script_context,
)


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


def knowledge(characters, environment=None, objects=None, event=None, text=None):
    """Lightweight visual-shaped doc: fine for direct remember_page parsing."""
    panels = []
    for i, (name, desc, role) in enumerate(characters, 1):
        char = {"name": name}
        if desc:
            char["description"] = desc
        if role:
            char["role"] = role
        panels.append({
            "panel_id": f"p{i:03d}",
            "ocr": {"text": text or ""},
            "visual": {
                "characters": [char],
                "environment": environment,
                "objects": objects or [],
                "important_event": event,
                "actions": [],
            },
        })
    return {"page": 1, "panels": panels}


def valid_knowledge(tmp_path, page, characters, environment=None,
                    event=None, text=None):
    """Schema-valid page knowledge (real panel files) for the disk path."""
    paths = []
    panels = []
    for i, (name, desc, role) in enumerate(characters, 1):
        image = Path(tmp_path) / "panels" / f"page_{page:03d}" / f"panel_{i:03d}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"\xff\xd8\xff\xe0")
        paths.append(image)
        char = {"name": name}
        if desc:
            char["description"] = desc
        if role:
            char["role"] = role
        panels.append({
            "panel_id": f"p{page:03d}_{i:03d}",
            "page": page,
            "reading_order": i,
            "image": str(image),
            "bbox": [0, 0, 200, 150],
            "ocr": {"text": text or ""},
            "visual": {
                "characters": [char],
                "environment": environment,
                "objects": [],
                "important_event": event,
                "actions": [],
            },
            "previous_panel": "", "next_panel": "", "scene_id": i,
        })
    return {
        "page": page, "reading_direction": "rtl",
        "panel_count": len(panels), "panels": panels,
    }


def write_knowledge(tmp_path, page, doc):
    path = Path(tmp_path) / "analysis" / f"page_{page:03d}_knowledge.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_build_page_summary_keeps_real_info_skips_unknown():
    doc = knowledge(
        [("Alice", "a black-haired girl", "protagonist")],
        environment="a ruined castle", objects=None,
        event="Alice meets Bob", text="Where is the sword?")
    s = build_page_summary(1, doc)
    assert "Alice" in s and "Where is the sword?" in s and "Alice meets Bob" in s
    assert "unknown" not in s

    empty = build_page_summary(1, {"panels": [
        {"panel_id": "p1", "ocr": {"text": ""},
         "visual": {"characters": [{"name": "unknown"}],
                    "important_event": "unknown"}}]})
    assert "no panel information" in empty


def test_project_memory_merges_characters():
    mem = ProjectMemory()
    doc1 = knowledge([("Alice", "a black-haired girl", "protagonist")],
                     environment="a park", objects=[{"name": "sword"}],
                     event="Alice arrives")
    doc2 = knowledge([("Alice", "a black-haired girl", "protagonist"),
                      ("Bob", "a tall man", "ally")],
                     environment="a park", objects=[{"name": "sword"}],
                     event="Alice meets Bob")
    mem.remember_page(1, doc1)
    mem.remember_page(2, doc2)

    alice = mem.characters["Alice"]
    assert alice["appearances"] == 2
    assert alice["first_seen_page"] == 1
    assert alice["descriptions"] == ["a black-haired girl"]
    assert alice["roles"] == ["protagonist"]

    bob = mem.characters["Bob"]
    assert bob["appearances"] == 1 and bob["first_seen_page"] == 2

    # page 2 has two panels (one per character), so place/object got two hits
    assert mem.places["a park"]["appearances"] == 3
    assert mem.objects["sword"]["appearances"] == 3
    assert mem.total_pages_seen == 2

    prompt = mem.to_prompt()
    assert "Alice" in prompt and "protagonist" in prompt
    assert "PLACES" in prompt and "a park" in prompt
    assert "OBJECTS" in prompt and "sword" in prompt


def test_project_memory_survives_new_pdf(tmp_path):
    cfg = make_cfg(tmp_path)
    doc = knowledge([("Alice", "a black-haired girl", "protagonist")],
                    environment="a forest", event="Alice finds the sword")

    remember_project(cfg, 1, knowledge=doc, pdf_name="volume_1.pdf")
    # "project resumed with a NEW pdf" -> fresh loads still hold the facts
    remember_project(cfg, 2, knowledge=doc, pdf_name="volume_2.pdf")
    remember_project(cfg, 3, knowledge=doc, pdf_name="volume_2.pdf")

    mem = ProjectMemory.load(cfg)
    assert mem.first_pdf == "volume_1.pdf"   # first identity is kept
    assert mem.total_pages_seen == 3
    assert mem.characters["Alice"]["appearances"] == 3

    info = memory_info(cfg)
    assert info["characters"] == 1
    assert info["total_pages_seen"] == 3


def test_page_window_keeps_last_10(max_size=10):
    win = PageWindow()
    for page in range(1, 25):
        win.add_page(page, knowledge([("Alice", None, None)],
                                     environment="the palace"))
    keep = sorted(int(k) for k in win.pages)
    assert keep == list(range(25 - DEFAULT_WINDOW_SIZE, 25))
    prompt = win.to_prompt()
    assert "99" not in prompt                      # old pages fell out
    assert "24" in prompt and "RECENT STORY" in prompt


def test_facts_dedupe_and_cap():
    mem = ProjectMemory()
    mem.add_fact("the sword is cursed", page=1)
    mem.add_fact("the sword is cursed", page=1)    # exact dupe dropped
    assert mem.facts == ["page p.1: the sword is cursed"]
    for i in range(60):
        mem.add_fact(f"event {i}")
    assert len(mem.facts) <= 40
    assert "event 59" in mem.facts


def test_script_context_feeds_memory_and_window(tmp_path):
    cfg = make_cfg(tmp_path)
    doc = knowledge([("Alice", "the princess", "protagonist")],
                    environment="the throne room",
                    event="Alice takes the throne", text="My crown.")
    remember_project(cfg, 5, knowledge=doc, pdf_name="manga.pdf")

    memory_block, window_block = script_context(cfg)
    assert "Alice" in memory_block
    assert "the throne room" in memory_block
    assert "Page 5" in window_block
    assert "RECENT STORY CONTEXT" in window_block


def test_remember_project_is_safe_without_knowledge(tmp_path):
    cfg = make_cfg(tmp_path)
    assert remember_project(cfg, 1) is False        # no knowledge file -> no-op
    assert remember_project(cfg, 1, knowledge="garbage") is False
    assert script_context(cfg) == ("", "")
    assert memory_info(cfg) is not None


def test_remember_project_from_disk_knowledge(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(tmp_path, 1, valid_knowledge(
        tmp_path, 1, [("Bob", "the guide", "ally")],
        environment="the harbor", event="Bob points the way", text="Ahoy!"))
    assert remember_project(cfg, 1) is True
    mem = ProjectMemory.load(cfg)
    assert "Bob" in mem.characters
    assert "the harbor" in mem.places
    win = PageWindow.load(cfg)
    assert "Bob" in win.to_prompt()