"""Tests: script -> panel visual timeline (plan_shots stage).

All fixtures are synthetic JSON; the planner never touches images or AI
models. Covers the ten required scenarios plus scoring/camera/cutting extras.
"""

import json
from pathlib import Path

import pytest

from config.loader import Config
from pipeline.visual_planner import (
    CAMERAS,
    VISUAL_INTENTS,
    VisualPlanner,
    camera_plan,
    review_path,
    score_panel,
    timeline_path,
)
from state import State

ROOT = Path(__file__).resolve().parent.parent
STAGE_NAMES = ["extract_pages", "segment_panels", "run_ocr", "analyze_panels",
               "build_scenes", "write_script", "generate_audio", "plan_shots"]


def make_cfg(tmp_path, shots=None):
    data = {
        "input": {"pdf": str(tmp_path / "unused.pdf")},
        "output": {
            "dir": str(tmp_path / "output"),
            "pages_dir": str(tmp_path / "pages"),
            "panels_dir": str(tmp_path / "panels"),
            "ocr_dir": str(tmp_path / "ocr"),
            "analysis_dir": str(tmp_path / "analysis"),
            "scenes_dir": str(tmp_path / "scenes"),
            "script_dir": str(tmp_path / "script"),
            "audio_dir": str(tmp_path / "audio"),
            "shots_dir": str(tmp_path / "shots"),
        },
        "images": {"format": "jpg", "render_scale": 1.0, "jpeg_quality": 80},
        "panels": {"jpeg_quality": 85, "min_area": 3000, "max_panels": 40},
        "reading": {"direction": "rtl", "row_overlap_ratio": 0.5},
        "ocr": {"engine": "auto", "language": "eng+jpn", "psm": 11, "timeout_seconds": 30},
        "scenes": {"threshold": 0.45, "weights": {}, "continuity": {}, "transition_keywords": [], "summary_max_items": 6},
        "llm": {"enabled": True, "provider": "mock", "model": "", "device": "cpu", "max_context": 4096, "max_new_tokens": 512, "temperature": 0.7, "timeout_seconds": 120},
        "tts": {"enabled": True, "engine": "auto", "voice": "en", "sample_rate": 22050, "rate_wpm": 150, "pitch_base": 50, "timeout_seconds": 60},
        "shots": {
            "match_weights": {
                "character": 0.25, "action": 0.15, "event": 0.20,
                "object": 0.10, "ocr": 0.15, "story_relevance": 0.15,
            },
            "review_threshold": 0.55, "tie_epsilon": 0.02,
            "direct_match_floor": 0.90, "secondary_panel_epsilon": 0.15,
            "long_segment_threshold": 9.0, "max_shots_per_segment": 3,
            "zoom_in_end": 1.12, "zoom_out_end": 0.92,
        },
        "video": {"resolution": "1920x1080", "fps": 30},
        "pipeline": {"batch_size": 1, "state": {"dir": str(tmp_path / "state")}, "cache": {"dir": str(tmp_path / "state" / "cache")}},
        "memory": {"guard_mb": 3072},
        "logging": {"level": "INFO", "console_level": "WARNING", "log_dir": str(tmp_path / "logs"), "max_bytes": 1048576, "backup_count": 3},
    }
    if shots:
        data["shots"].update(shots)
    return Config(data, tmp_path)


def panel(panel_id, order, characters=None, actions=None, events=None,
          objects=None, ocr_text=None, story=None, scene_id=None, visual_confidence=0.7):
    image = Path(f"/tmp/unused/panels/{panel_id}.jpg")
    return {
        "panel_id": panel_id, "page": 1, "reading_order": order,
        "image": str(image), "bbox": [8, 8, 90, 80],
        "ocr": {"text": ocr_text or "", "blocks": []},
        "visual": {
            "characters": [{"name": n, "description": "", "action": "", "emotion": ""} for n in (characters or [])],
            "environment": "unknown",
            "actions": list(actions or []),
            "objects": list(objects or []),
            "visual_effects": [],
            "important_event": (events or [""])[0] if events else "unknown",
            "composition": "unknown",
            "story_relevance": story if story is not None else "unknown",
            "confidence": visual_confidence,
        },
        "previous_panel": None, "next_panel": None, "scene_id": scene_id,
    }


def write_knowledge(cfg, panels, page=1):
    from pipeline.knowledge import knowledge_path

    panels_dir = Path(cfg.output.panels_dir) / f"page_{page:03d}"
    panels_dir.mkdir(parents=True, exist_ok=True)
    for record in panels:
        image = Path(record["image"])
        image.parent.mkdir(parents=True, exist_ok=True)
        image.touch()

    path = knowledge_path(cfg, page)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "page": page, "source": str(path.parent / "unused.jpg"),
        "panel_count": len(panels), "reading_direction": "rtl",
        "panels": panels,
    }), encoding="utf-8")
    return path


def write_script(cfg, segments, page=1, scene=1, scene_id="scene_001", raw=None):
    path = Path(cfg.output.script_dir) / f"page_{page:03d}_scene_{scene:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        path.write_text(raw, encoding="utf-8")
    else:
        path.write_text(json.dumps({
            "scene_id": scene_id, "page": page,
            "segments": segments,
        }), encoding="utf-8")
    return path


def seg(segment_id, text, seconds=4.0, panel_ids=None, intent="full_panel",
        camera="static", seg_type="narration", **extra):
    out = {
        "segment_id": segment_id, "type": seg_type, "text": text,
        "panel_ids": list(panel_ids) if panel_ids is not None else [],
        "estimated_seconds": seconds, "visual_intent": intent, "camera": camera,
        "importance": 0.7,
    }
    out.update(extra)
    return out


def run(cfg, page=1, scene=1, force=False):
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = VisualPlanner(cfg).run_scene(page, scene, state, force=force)
    return result, state


def read_timeline(cfg, page=1, scene=1):
    return json.loads(timeline_path(cfg, page, scene).read_text("utf-8"))


# ----------------------------------------------------------- required: 1-10
# 1. direct panel match


def test_direct_panel_match(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [
        panel("p001_001", 2, characters=["Guts"], actions=["advances"],
              events=["night fall"], objects=["sword"], ocr_text="Hello World"),
        panel("p001_002", 1, characters=["Casca"]),
    ])
    write_script(cfg, [seg("seg_001", "He advances at night.", 4.0,
                           panel_ids=["p001_001"], intent="full_panel", camera="static")])
    result, state = run(cfg)
    assert result["result"] == "ok"
    shot = result["shots"][0]
    assert shot["primary_panel"] == "p001_001"
    assert shot["panel_ids"] == ["p001_001"]
    assert shot["match_score"] >= 0.90
    assert shot["needs_review"] is False
    assert shot["visual_intent"] == "full_panel"
    assert state.pages.get("page_001_scene_001") == "visual_plan_completed"


def test_automatic_fallback_match(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [
        panel("p001_001", 1, characters=["Casca"], actions=["resting"]),
        panel("p001_002", 2, characters=["Guts"], actions=["advances"],
              events=["night fall"], objects=["sword"]),
        panel("p001_003", 3, characters=["Griffith"], actions=["riding"]),
    ])
    write_script(cfg, [seg("seg_001", "Guts advances through the night.", 4.0,
                           panel_ids=[])])
    result, _ = run(cfg)
    assert result["result"] == "ok"
    shot = result["shots"][0]
    assert shot["primary_panel"] == "p001_002"  # best-scoring panel
    assert shot["match_score"] >= 0.55
    assert shot["needs_review"] is False


def test_low_confidence_match(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [
        panel("p001_001", 1, characters=["Casca"]),
        panel("p001_002", 2, characters=["Griffith"]),
        panel("p001_003", 3, characters=["Rickert"]),
    ])
    write_script(cfg, [seg("seg_001", "The sky itself begins to weep.", 4.0,
                           panel_ids=[])])
    result, _ = run(cfg)
    assert result["result"] == "ok"
    shot = result["shots"][0]
    assert shot["needs_review"] is True
    assert shot["match_score"] < 0.55


def test_invalid_panel_id(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [
        panel("p001_001", 1, characters=["Guts"], actions=["advances"],
              events=["night fall"]),
        panel("p001_002", 2, characters=["Casca"]),
    ])
    write_script(cfg, [seg("seg_001", "Guts advances at night.", 4.0,
                           panel_ids=["p999_999"])])
    result, _ = run(cfg)
    assert result["result"] == "ok"  # falls back, does not crash
    shot = result["shots"][0]
    assert shot["primary_panel"] == "p001_001"  # auto-matched to best
    assert "p999_999" in result["dropped_panel_ids"]


def test_visual_intent_fallback(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [panel("p001_001", 1, characters=["Guts"])])
    write_script(cfg, [seg("seg_001", "Arid whispers carry.", 4.0,
                           panel_ids=["p001_001"], intent="extreme_zoom")])
    result, _ = run(cfg)
    assert result["shots"][0]["visual_intent"] == "full_panel"
    assert result["shots"][0]["visual_intent"] in VISUAL_INTENTS


def test_camera_validation(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [panel("p001_001", 1, characters=["Guts"])])
    write_script(cfg, [seg("seg_001", "Quiet days.", 4.0, panel_ids=["p001_001"],
                           camera="helicopter")])
    result, _ = run(cfg)
    assert result["shots"][0]["camera"]["type"] == "static"

    write_script(cfg, [seg("seg_001", "Quiet days.", 4.0, panel_ids=["p001_001"],
                           camera="slow_zoom_in")], scene=1)
    result2, _ = run(cfg, force=True)
    shot = result2["shots"][0]["camera"]
    assert shot["type"] == "slow_zoom_in"
    assert shot["start"] == 1.0 and shot["end"] == 1.12


def test_panel_reuse(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [panel("p001_001", 1, characters=["Guts"])])
    write_script(cfg, [
        seg("seg_001", "The hero stands.", 4.0, panel_ids=["p001_001"]),
        seg("seg_002", "The hero speaks.", 4.0, panel_ids=["p001_001"]),
    ])
    result, _ = run(cfg)
    shots = result["shots"]
    assert shots[0]["reuse_count"] == 1
    assert shots[1]["reuse_count"] == 2
    assert shots[0]["primary_panel"] == shots[1]["primary_panel"] == "p001_001"


def test_timeline_duration(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [panel("p001_001", 1, characters=["Guts"])])
    write_script(cfg, [
        seg("seg_001", "Chapter one.", 4.0, panel_ids=["p001_001"]),
        seg("seg_002", "Chapter two.", 3.5, panel_ids=["p001_001"]),
        seg("seg_003", "Chapter three.", 2.0, panel_ids=["p001_001"]),
    ])
    result, _ = run(cfg)
    durations = [shot["estimated_duration"] for shot in result["shots"]]
    assert durations == [4.0, 3.5, 2.0]
    assert sum(durations) == pytest.approx(9.5)


def test_checkpoint(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [panel("p001_001", 1, characters=["Guts"])])
    write_script(cfg, [seg("seg_001", "Once more.", 4.0, panel_ids=["p001_001"])])
    first, state = run(cfg)
    assert first["result"] == "ok"
    assert state.pages.get("page_001_scene_001") == "visual_plan_completed"
    second, _ = run(cfg)
    assert second["result"] == "skipped"


def test_force_regeneration(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [panel("p001_001", 1, characters=["Guts"])])
    write_script(cfg, [seg("seg_001", "Two moons.", 4.0, panel_ids=["p001_001"],
                           camera="static")])
    run(cfg)
    before = read_timeline(cfg)["generated_at"]
    write_script(cfg, [seg("seg_001", "Two moons.", 4.0, panel_ids=["p001_001"],
                           camera="slow_zoom_in")])
    result, _ = run(cfg, force=True)
    assert result["result"] == "ok"
    assert read_timeline(cfg)["shots"][0]["camera"]["type"] == "slow_zoom_in"
    assert read_timeline(cfg)["generated_at"] != before


# ------------------------------------------------------------- extra coverage


def test_review_report_contents(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [panel("p001_001", 1, characters=["Guts"])])
    write_script(cfg, [seg("seg_001", "Night.", 4.0, panel_ids=["p001_001"],
                           intent="face_closeup", camera="pan_right")])
    run(cfg)
    text = review_path(cfg, 1, 1).read_text("utf-8")
    assert "SCENE SCENE_001" in text or "SCENE 001" in text.upper()
    assert "SHOT 001" in text
    assert "Narration: Night." in text
    assert "Panel: P001_001" in text
    assert "Match: " in text
    assert "Visual: face_closeup" in text
    assert "Camera: pan_right" in text
    assert "Review: NO" in text


def test_missing_script(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [panel("p001_001", 1, characters=["Guts"])])
    result, state = run(cfg)
    assert result["result"] == "error"
    assert "script file not found" in result["message"]
    assert state.pages.get("page_001_scene_001") is None


def test_missing_knowledge(tmp_path):
    cfg = make_cfg(tmp_path)
    write_script(cfg, [seg("seg_001", "Night.", 4.0, panel_ids=["p001_001"])])
    result, state = run(cfg)
    assert result["result"] == "error"
    assert "knowledge" in result["message"]
    assert state.pages.get("page_001_scene_001") is None


def test_empty_segments_rejected(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [panel("p001_001", 1, characters=["Guts"])])
    write_script(cfg, [], raw='{"scene_id":"scene_001","page":1,"segments":[]}')
    result, state = run(cfg)
    assert result["result"] == "error"
    assert "no segments" in result["message"] or "segments" in result["message"]
    assert state.pages.get("page_001_scene_001") is None


def test_multiple_explicit_panels_kept_in_order(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [
        panel("p001_001", 3, characters=["Guts"]),
        panel("p001_002", 1, characters=["Casca"]),
        panel("p001_003", 2, characters=["Griffith"]),
    ])
    write_script(cfg, [seg("seg_001", "All converge.", 5.0,
                           panel_ids=["p001_002", "p001_003"],
                           intent="multi_panel")])
    result, _ = run(cfg)
    shot = result["shots"][0]
    assert shot["panel_ids"] == ["p001_002", "p001_003"]  # reading order kept
    assert shot["primary_panel"] in ("p001_002", "p001_003")
    assert shot["visual_intent"] == "multi_panel"


def test_secondary_panel_included_when_close(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [
        panel("p001_001", 1, characters=["Guts"], actions=["advances"],
              events=["night fall"]),
        panel("p001_002", 2, characters=["Casca"], actions=["advances"],
              events=["night fall"], objects=["campfire"]),
    ])
    write_script(cfg, [seg("seg_001", "They advance under falling night.",
                           6.0, panel_ids=[])])
    result, _ = run(cfg)
    shot = result["shots"][0]
    assert len(shot["panel_ids"]) == 2
    assert shot["primary_panel"] != shot["panel_ids"][1]


def test_long_segment_splits_into_multiple_shots(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [
        panel("p001_001", 1, characters=["Guts"],
              actions=["advances"], events=["war"],
              objects=["sword"], story=0.9),
    ])
    write_script(cfg, [seg("seg_001", "The long night of the hero.", 20.0,
                           panel_ids=["p001_001"], intent="full_panel",
                           camera="slow_zoom_in")])
    result, _ = run(cfg)
    shots = result["shots"]
    assert len(shots) == 3
    intents = [shot["visual_intent"] for shot in shots]
    assert intents[0] == "full_panel"
    assert "object_closeup" in intents
    assert "character_closeup" in intents
    assert shots[-1].get("segment_id") == "seg_001"
    total = sum(shot["estimated_duration"] for shot in shots)
    assert total == pytest.approx(20.0, abs=0.05)


def test_short_segment_not_split(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [
        panel("p001_001", 1, characters=["Guts"], objects=["sword"]),
    ])
    write_script(cfg, [seg("seg_001", "A beat.", 4.0, panel_ids=["p001_001"])])
    result, _ = run(cfg)
    assert len(result["shots"]) == 1


def test_all_cameras_valid():
    cfg = make_cfg(Path("/tmp/visual_units"))
    for camera in CAMERAS:
        plan = camera_plan(camera, cfg)
        assert plan["type"] == camera
        assert plan["start"] <= plan["end"] or camera == "slow_zoom_out"
    assert camera_plan("nothing", cfg)["type"] == "static"


def test_score_panel_components():
    weight = {"character": 1.0, "action": 0.0, "event": 0.0, "object": 0.0,
              "ocr": 0.0, "story_relevance": 0.0}
    hit = panel("p001_001", 1, characters=["Guts"])
    miss = panel("p001_002", 2, characters=["Casca"])
    assert score_panel(hit, "Guts advances", weight)[0] == 1.0
    assert score_panel(miss, "Guts advances", weight)[0] == 0.0


def test_deterministic_output(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [
        panel("p001_001", 1, characters=["Guts"]),
        panel("p001_002", 2, characters=["Casca"]),
    ])
    write_script(cfg, [seg("seg_001", "Guts glares.", 5.0, panel_ids=[])])
    a, _ = run(cfg)
    b, _ = run(cfg, force=True)
    assert a["shots"] == b["shots"]