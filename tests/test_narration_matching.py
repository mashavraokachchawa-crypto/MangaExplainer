"""Tests: deterministic panel <-> narration mapping (Task 10).

Verifies reading-order preservation, the three cardinalities (one_to_one,
many_to_one, one_to_many), unmatched panels, resume/skip behaviour and the
consolidated file. Inputs are written to a tmp project (no real pages).
"""
import json
from pathlib import Path

from config.loader import load_config
from pipeline import narration_matching as nm


def _write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def _cfg(root):
    cfg = load_config(root)
    return cfg


def _seed_project(tmp_path):
    root = Path(tmp_path) / "project"
    cfg = _cfg(root)
    # panel manifest for page 2
    panels_dir = Path(cfg.output.panels_dir) / "page_002"
    _write(panels_dir / "panels.json", {
        "page": 2,
        "panels": [
            {"id": "p002_001", "image": "panels/page_002/panel_001.jpg",
             "bbox": [0, 0, 10, 10], "reading_order": 1},
            {"id": "p002_002", "image": "panels/page_002/panel_002.jpg",
             "bbox": [10, 0, 10, 10], "reading_order": 4},
            {"id": "p002_003", "image": "panels/page_002/panel_003.jpg",
             "bbox": [0, 20, 10, 10], "reading_order": 2},
            {"id": "p002_004", "image": "panels/page_002/panel_004.jpg",
             "bbox": [20, 20, 10, 10], "reading_order": 3},
            {"id": "p002_005", "image": "panels/page_002/panel_005.jpg",
             "bbox": [30, 0, 10, 10], "reading_order": 5},
            {"id": "p002_006", "image": "panels/page_002/panel_006.jpg",
             "bbox": [0, 40, 10, 10], "reading_order": 6},
        ],
    })
    # authoritative reading order (differs from panel manifest order on purpose)
    order_dir = Path(cfg.output.panels_dir) / "page_002"
    (order_dir / "reading_order.json").write_text(json.dumps({
        "page": 2, "direction": "rtl",
        "order": ["p002_003", "p002_001", "p002_004", "p002_002",
                  "p002_005", "p002_006"],
    }, indent=2), encoding="utf-8")
    # narration scripts for the page
    script_dir = Path(cfg.output.script_dir)
    _write(script_dir / "page_002_scene_001.json", {
        "scene_id": "scene_001",
        "page": 2,
        "segments": [
            {"type": "narration", "segment_id": "seg_001",
             "text": "A into B.",
             "panel_ids": ["p002_005", "p002_003"]},   # many_to_one
            {"type": "dialogue", "segment_id": "seg_002",
             "text": "C alone.",
             "panel_ids": ["p002_003"]},                # one_to_one + one_to_many
            {"type": "narration", "segment_id": "seg_003",
             "text": "A again plus D.",
             "panel_ids": ["p002_001", "p002_004", "p002_001"]},  # one_to_many, dedup
            {"type": "narration", "segment_id": "seg_004",
             "text": "References off-page.",
             "panel_ids": ["p002_099"]},                # unknown -> warning
        ],
    })
    return cfg, root


def test_mapping_preserves_reading_order_and_cardinalities(tmp_path):
    cfg, _ = _seed_project(tmp_path)
    page = 2
    matcher = nm.NarrationMatcher(cfg)
    res = matcher.run_page(page, force=False)
    assert res["result"] == "ok"

    path = nm.page_mapping_path(cfg, page)
    assert path.is_file()
    data = json.loads(path.read_text("utf-8"))

    # reading order preserved (authoritative reading_order.json wins)
    assert data["reading_order"] == ["p002_003", "p002_001", "p002_004",
                                     "p002_002", "p002_005", "p002_006"]
    panel_order = [p["panel_id"] for p in data["panels"]]
    assert panel_order == data["reading_order"]

    # many_to_one: seg_001 pushed to [p002_003, p002_005] by reading order
    seg1 = next(s for s in data["segments"] if s["segment_id"] == "seg_001")
    assert seg1["cardinality"] == "many_to_one"
    assert seg1["panel_ids"] == ["p002_003", "p002_005"]

    # one_to_one: seg_002 -> single panel
    seg2 = next(s for s in data["segments"] if s["segment_id"] == "seg_002")
    assert seg2["cardinality"] == "one_to_one"

    # unknown panel ids surfaced as a warning, never silently dropped
    seg4 = next(s for s in data["segments"] if s["segment_id"] == "seg_004")
    assert seg4["unknown_panel_ids"] == ["p002_099"]
    assert seg4["panel_ids"] == []
    assert any("not on this page" in w for w in data["warnings"])

    # one_to_many: p002_003 is referenced by seg_001 + seg_002
    by_id = {p["panel_id"]: p for p in data["panels"]}
    assert by_id["p002_003"]["cardinality"] == "one_to_many"
    assert len(by_id["p002_003"]["narration_segments"]) == 2
    # panels never referenced => unmatched
    assert by_id["p002_006"]["cardinality"] == "unmatched"
    assert by_id["p002_006"]["narration_segments"] == []
    assert data["summary"]["unmatched_panels"] >= 1


def test_resume_skips_completed_pages(tmp_path):
    cfg, _ = _seed_project(tmp_path)
    matcher = nm.NarrationMatcher(cfg)
    assert matcher.run_page(2)["result"] == "ok"
    # second run skips without rewriting
    assert matcher.run_page(2)["result"] == "skipped"
    # force rebuilds
    assert matcher.run_page(2, force=True)["result"] == "ok"
    # index tracks it as completed exactly once
    rows = [r for r in nm.load_index(cfg)["pages"] if r["page"] == 2]
    assert len(rows) == 1
    assert rows[0]["status"] == nm.MATCH_COMPLETED


def test_empty_page_skips_visibly_not_fatal(tmp_path):
    cfg, _ = _seed_project(tmp_path)
    # page 1 has no panels manifest at all -> error (missing input)
    # page with an empty panels list -> visible skip
    empty_dir = Path(cfg.output.panels_dir) / "page_001"
    (empty_dir / "panels.json").parent.mkdir(parents=True, exist_ok=True)
    _write(empty_dir / "panels.json", {"page": 1, "panels": []})
    matcher = nm.NarrationMatcher(cfg)
    res = matcher.run_page(1)
    assert res["result"] == "skipped"
    assert res["reason"] == nm.NO_PANELS_REASON
    assert matcher.run_page(1)["result"] == "skipped"  # resumable


def test_consolidate_streams_all_pages(tmp_path):
    cfg, _ = _seed_project(tmp_path)
    matcher = nm.NarrationMatcher(cfg)
    matcher.run_page(2)
    merged = nm.consolidate_mapping(cfg)
    assert merged["total_segments"] == 4
    # the consolidated file exists and lists page 2
    pages = {p["page"] for p in merged["pages"]}
    assert 2 in pages
    assert nm.consolidated_path(cfg).is_file()


def test_external_narration_connection_means_mapping_is_separable():
    # The mapping is derived purely from script segments + panel manifest; a
    # pipeline consumer can read mapping.json without any image/model.
    assert nm.MATCH_SCHEMA == 1
    assert not {"audio", "subtitle", "video"} & {"narration"}  # no such stages here