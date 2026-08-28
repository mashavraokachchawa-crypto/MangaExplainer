"""Tests: rule-based scene reconstruction (synthetic knowledge fixtures)."""

import json
from pathlib import Path

import pytest

from config.loader import Config
from pipeline.knowledge import knowledge_path
from pipeline.scene_builder import (
    SceneError,
    SceneProcessor,
    boundary_score,
    build_scenes,
)
from state import State

ROOT = Path(__file__).resolve().parent.parent
STAGE_NAMES = ["extract_pages", "segment_panels", "run_ocr", "analyze_panels"]


def make_cfg(tmp_path, scenes=None):
    scenes = scenes or {}
    data = {
        "input": {"pdf": str(tmp_path / "unused.pdf")},
        "output": {
            "dir": str(tmp_path / "output"),
            "pages_dir": str(tmp_path / "pages"),
            "panels_dir": str(tmp_path / "panels"),
            "ocr_dir": str(tmp_path / "ocr"),
            "analysis_dir": str(tmp_path / "analysis"),
            "scenes_dir": str(tmp_path / "scenes"),
        },
        "images": {"format": "jpg", "render_scale": 1.0, "jpeg_quality": 80},
        "panels": {"jpeg_quality": 85, "min_area": 3000, "max_panels": 40},
        "reading": {"direction": "rtl", "row_overlap_ratio": 0.5},
        "ocr": {"engine": "auto", "language": "eng+jpn", "psm": 11, "timeout_seconds": 30},
        "vlm": {"enabled": True, "provider": "mock", "model": "", "device": "cpu",
                "max_image_size": 768, "max_new_tokens": 256, "timeout_seconds": 120},
        "scenes": {
            "threshold": 0.45,
            "weights": {"location_change": 0.45, "character_change": 0.55,
                        "event_change": 0.5, "narrative_transition": 0.5},
            "continuity": {"character": 0.35, "location": 0.35, "action": 0.25,
                           "dialogue": 0.3, "event": 0.3},
            "transition_keywords": ["meanwhile", "later", "翌日", "しかし", "そして"],
            "summary_max_items": 6,
        },
        "video": {"resolution": "1920x1080", "fps": 30},
        "pipeline": {"batch_size": 1, "state": {"dir": str(tmp_path / "state")},
                     "cache": {"dir": str(tmp_path / "state" / "cache")}},
        "memory": {"guard_mb": 3072},
        "logging": {"level": "INFO", "console_level": "WARNING",
                    "log_dir": str(tmp_path / "logs"), "max_bytes": 1048576,
                    "backup_count": 3},
    }
    data["scenes"].update(scenes)
    return Config(data, tmp_path)


def record(cfg, page=1, index=1, characters=None, location=None, actions=None,
           important_event=None, ocr_text=None, confidence=0.8, bad=False):
    image = Path(cfg.output.panels_dir) / f"page_{page:03d}" / f"panel_{index:03d}.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    if not image.exists():
        image.write_bytes(b"\xff\xd8\xff\xe0fake")
    visual = {
        "characters": [{"name": n, "description": "", "action": "", "emotion": ""}
                       for n in (characters or [])],
        "environment": location or "unknown",
        "actions": actions or [],
        "objects": [],
        "visual_effects": [],
        "important_event": important_event or "unknown",
        "composition": "unknown",
        "story_relevance": "unknown",
        "confidence": confidence,
    }
    rec = {
        "panel_id": f"p{page:03d}_{index:03d}",
        "page": page,
        "reading_order": index,
        "image": str(image),
        "bbox": [8, 8, 90, 80],
        "ocr": {"text": ocr_text or "", "blocks": []},
        "visual": visual,
        "previous_panel": None,
        "next_panel": None,
        "scene_id": None,
    }
    if bad:
        rec.pop("panel_id")
    return rec


def write_knowledge(cfg, records, page=1, valid=True, raw=None):
    path = knowledge_path(cfg, page)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "page": page,
        "source": str(path.parent / f"page_{page:03d}.jpg"),
        "panel_count": len(records),
        "reading_direction": "rtl",
        "panels": records,
    }
    if raw is not None:
        path.write_text(raw, encoding="utf-8")
    else:
        path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def run(cfg, page=1, force=False):
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = SceneProcessor(cfg).run_page(page, state, force=force)
    return result, state


def read_scenes(cfg, page=1):
    return json.loads((Path(cfg.output.scenes_dir) / f"page_{page:03d}_scenes.json").read_text("utf-8"))


def scene_panel_sets(cfg, page=1):
    return [scene["panel_ids"] for scene in read_scenes(cfg, page)["scenes"]]


# --------------------------------------------------------------- required


def test_single_panel_scene(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [record(cfg, index=1)])
    result, state = run(cfg)
    assert result["result"] == "ok"
    assert result["scene_count"] == 1
    assert scene_panel_sets(cfg) == [["p001_001"]]
    doc = read_scenes(cfg)
    assert doc["scenes"][0]["scene_id"] == "scene_001"
    assert doc["scenes"][0]["page_start"] == 1 and doc["scenes"][0]["page_end"] == 1
    assert state.pages.get("page_001") == "scenes_completed"


def test_multiple_panels_same_scene(tmp_path):
    cfg = make_cfg(tmp_path)
    records = [
        record(cfg, index=1, characters=["guts"], location="hill", important_event="battle"),
        record(cfg, index=2, characters=["guts"], location="hill", important_event="battle"),
        record(cfg, index=3, characters=["guts"], location="hill", important_event="battle"),
    ]
    write_knowledge(cfg, records)
    result, _ = run(cfg)
    assert result["scene_count"] == 1
    assert scene_panel_sets(cfg) == [["p001_001", "p001_002", "p001_003"]]


def test_different_locations(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [
        record(cfg, index=1, location="forest"),
        record(cfg, index=2, location="castle"),
    ])
    result, _ = run(cfg)
    assert result["scene_count"] == 2
    assert scene_panel_sets(cfg) == [["p001_001"], ["p001_002"]]
    decisions = result["boundaries"]
    assert decisions and decisions[0]["boundary"] is True
    assert decisions[0]["score"] >= 0.45


def test_different_characters(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [
        record(cfg, index=1, characters=["guts"]),
        record(cfg, index=2, characters=["griffith"]),
    ])
    result, _ = run(cfg)
    assert result["scene_count"] == 2
    assert scene_panel_sets(cfg) == [["p001_001"], ["p001_002"]]


def test_continuous_dialogue(tmp_path):
    cfg = make_cfg(tmp_path)
    # character AND location change signals present, but the strong shared
    # dialogue + same-event continuity keeps it one scene
    write_knowledge(cfg, [
        record(cfg, index=1, characters=["guts"], location="castle",
               important_event="confrontation", ocr_text="「お前は」"),
        record(cfg, index=2, characters=["griffith"], location="castle",
               important_event="confrontation", ocr_text="「グリフィスだ」"),
    ])
    result, _ = run(cfg)
    assert result["scene_count"] == 1
    assert scene_panel_sets(cfg) == [["p001_001", "p001_002"]]


def test_event_transition(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [
        record(cfg, index=1, important_event="the party rests"),
        record(cfg, index=2, important_event="a monster attack"),
    ])
    result, _ = run(cfg)
    assert result["scene_count"] == 2
    assert scene_panel_sets(cfg) == [["p001_001"], ["p001_002"]]
    assert result["boundaries"][0]["boundary"] is True


def test_narrative_transition_keyword(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [
        record(cfg, index=1, important_event="talking"),
        record(cfg, index=2, ocr_text="Meanwhile, far away..."),
    ])
    result, _ = run(cfg)
    assert result["scene_count"] == 2


def test_threshold_controls_grouping(tmp_path):
    cfg = make_cfg(tmp_path, scenes={"threshold": 0.99})
    write_knowledge(cfg, [
        record(cfg, index=1, location="forest"),
        record(cfg, index=2, location="castle"),
    ])
    result, _ = run(cfg)
    assert result["scene_count"] == 1  # threshold too high -> no split


def test_empty_page(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [])
    result, state = run(cfg)
    assert result["result"] == "ok"
    assert result["scene_count"] == 0
    assert read_scenes(cfg)["scenes"] == []
    assert state.pages.get("page_001") == "scenes_completed"


def test_missing_knowledge_file(tmp_path):
    cfg = make_cfg(tmp_path)
    result, state = run(cfg)
    assert result["result"] == "error"
    assert "missing" in result["message"].lower()
    assert state.pages.get("page_001") is None


def test_invalid_panel_data(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [record(cfg, index=1, bad=True)])
    result, state = run(cfg)
    assert result["result"] == "error"
    assert "invalid" in result["message"].lower()
    assert state.pages.get("page_001") is None
    # direct build_scenes path also rejects
    with pytest.raises(SceneError):
        build_scenes([{"not": "a record"}], cfg.scenes)


def test_invalid_knowledge_json(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [], raw="{not json")
    result, _ = run(cfg)
    assert result["result"] == "error"


def test_checkpoint_skip_and_force(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [record(cfg, index=1)])
    result, _ = run(cfg)
    assert result["result"] == "ok"
    result2, _ = run(cfg)
    assert result2["result"] == "skipped"
    result3, _ = run(cfg, force=True)
    assert result3["result"] == "ok"


def test_knowledge_links_and_debug_files(tmp_path):
    cfg = make_cfg(tmp_path)
    records = [
        record(cfg, index=1, characters=["guts"], location="castle"),
        record(cfg, index=2, characters=["guts"], location="castle"),
        record(cfg, index=3, location="elsewhere"),
    ]
    write_knowledge(cfg, records)
    result, _ = run(cfg)
    assert result["scene_count"] == 2
    # knowledge file updated in place: scene links + linear prev/next
    doc = json.loads(knowledge_path(cfg, 1).read_text("utf-8"))
    links = {r["panel_id"]: r for r in doc["panels"]}
    assert links["p001_001"]["previous_panel"] is None
    assert links["p001_001"]["next_panel"] == "p001_002"
    assert links["p001_001"]["scene_id"] == "scene_001"
    assert links["p001_002"]["scene_id"] == "scene_001"
    assert links["p001_003"]["scene_id"] == "scene_002"
    assert links["p001_003"]["previous_panel"] == "p001_002"
    assert links["p001_003"]["next_panel"] is None
    # debug files written
    scenes_dir = Path(cfg.output.scenes_dir)
    assert (scenes_dir / "page_001_scene_debug.json").is_file()
    txt = (scenes_dir / "page_001_scene_debug.txt").read_text("utf-8")
    assert "SCENE" in txt and "p001_001" in txt


# ------------------------------------------------------------------ units


def test_boundary_score_units():
    scenes_cfg = make_cfg(Path("/tmp/scene_unit")).scenes
    loc_a = {"panel_id": "p", "visual": {"environment": "a"}}
    loc_b = {"panel_id": "p", "visual": {"environment": "b"}}
    assert boundary_score(loc_a, loc_b, scenes_cfg) == 0.45
    same = {"panel_id": "p", "visual": {"environment": "a", "important_event": "e"}}
    assert boundary_score(same, dict(same), scenes_cfg) == 0.0
    unknown = {"panel_id": "p", "visual": None}
    assert boundary_score(unknown, unknown, scenes_cfg) == 0.0
    assert boundary_score({}, {}, scenes_cfg) == 0.0


def test_boundary_score_uses_config_not_hardcoded():
    low = make_cfg(Path("/tmp/scene_cfg"), scenes={"threshold": 0.99}).scenes
    high = make_cfg(Path("/tmp/scene_cfg"), scenes={"threshold": 0.01}).scenes
    records = [
        {"panel_id": "a", "page": 1, "visual": {"environment": "x"}},
        {"panel_id": "b", "page": 1, "visual": {"environment": "y"}},
    ]
    grouped_low = build_scenes(records, low)[0]
    grouped_high = build_scenes(records, high)[0]
    assert [s["panel_ids"] for s in grouped_low] == [["a", "b"]]
    assert [s["panel_ids"] for s in grouped_high] == [["a"], ["b"]]