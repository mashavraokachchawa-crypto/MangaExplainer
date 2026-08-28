"""Tests: shot -> 16:9 cinematic crop planner (crop stage).

Synthetic panel images + knowledge/timeline JSON fixtures; the planner never
invokes AI. Covers the nine required scenarios plus checkpoint/force, config
resolution, debug output and close-up/fallback behaviour.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from config.loader import Config
from pipeline.crop_planner import (
    CropPlanner,
    compute_crop,
    parse_resolution,
    snap_box,
)
from state import State

ROOT = Path(__file__).resolve().parent.parent
STAGE_NAMES = ["extract_pages", "segment_panels", "crop_panels", "run_ocr",
               "analyze_panels", "build_scenes", "write_script",
               "generate_audio", "plan_shots", "render_video"]


def make_cfg(tmp_path, crops=None):
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
            "crops_dir": str(tmp_path / "crops"),
        },
        "images": {"format": "jpg", "render_scale": 1.0, "jpeg_quality": 80},
        "panels": {"jpeg_quality": 85, "min_area": 3000, "max_panels": 40},
        "reading": {"direction": "rtl", "row_overlap_ratio": 0.5},
        "ocr": {"engine": "auto", "language": "eng+jpn", "psm": 11, "timeout_seconds": 30},
        "scenes": {"threshold": 0.45, "weights": {}, "continuity": {}, "transition_keywords": [], "summary_max_items": 6},
        "llm": {"enabled": True, "provider": "mock", "model": "", "device": "cpu", "max_context": 4096, "max_new_tokens": 512, "temperature": 0.7, "timeout_seconds": 120},
        "tts": {"enabled": True, "engine": "auto", "voice": "en", "sample_rate": 22050, "rate_wpm": 150, "pitch_base": 50, "timeout_seconds": 60},
        "shots": {"match_weights": {}, "review_threshold": 0.55, "tie_epsilon": 0.02, "direct_match_floor": 0.9, "secondary_panel_epsilon": 0.15, "long_segment_threshold": 9.0, "max_shots_per_segment": 3, "zoom_in_end": 1.12, "zoom_out_end": 0.92},
        "crops": {
            "resolution": "1280x720", "format": "jpg", "jpeg_quality": 90,
            "safe_padding": 0.06, "critical_weight": 0.6,
            "min_blob_area": 80, "max_regions": 24, "debug": True,
        },
        "video": {"resolution": "1920x1080", "fps": 30},
        "pipeline": {"batch_size": 1, "state": {"dir": str(tmp_path / "state")}, "cache": {"dir": str(tmp_path / "state" / "cache")}},
        "memory": {"guard_mb": 3072},
        "logging": {"level": "INFO", "console_level": "WARNING", "log_dir": str(tmp_path / "logs"), "max_bytes": 1048576, "backup_count": 3},
    }
    if crops:
        data["crops"].update(crops)
    return Config(data, tmp_path)


def make_panel_image(path, width, height, marks=None):
    """White panel with optional gray marks (regions visible to CV)."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    for (x, y, w, h) in marks or []:
        image[y:y + h, x:x + w] = (120, 120, 120)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return path


def panel_record(panel_id, order, image, width, height, ocr_blocks=None,
                 faces=None, characters=None, scene_id=None):
    record = {
        "panel_id": panel_id, "page": 1, "reading_order": order,
        "image": str(image), "bbox": [0, 0, width, height],
        "ocr": {"text": (ocr_blocks[0]["text"] if ocr_blocks else ""),
                "blocks": ocr_blocks or []},
        "visual": {
            "characters": [
                {"name": (c.get("name") or ""), "description": "",
                 "action": "", "emotion": ""} for c in (characters or [])
            ],
            "faces": [{"name": ""} for _ in (faces or [])],
            "environment": "unknown",
            "actions": [], "objects": [], "visual_effects": [],
            "important_event": "unknown", "composition": "unknown",
            "story_relevance": "unknown", "confidence": 0.7,
        },
        "previous_panel": None, "next_panel": None, "scene_id": scene_id,
    }
    if faces:
        for face, entry in zip(faces, record["visual"]["faces"]):
            entry["bbox"] = face
    if characters:
        for char, entry in zip(characters, record["visual"]["characters"]):
            if "bbox" in char:
                entry["bbox"] = char["bbox"]
                entry["is_face"] = bool(char.get("is_face"))
    return record


def write_knowledge(cfg, panels, page=1):
    from pipeline.knowledge import knowledge_path

    path = knowledge_path(cfg, page)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "page": page, "source": str(path.parent / "unused.jpg"),
        "panel_count": len(panels), "reading_direction": "rtl",
        "panels": panels,
    }), encoding="utf-8")
    return path


def write_timeline(cfg, shots, page=1, scene=1, scene_id="scene_001"):
    from pipeline.visual_planner import timeline_path

    path = timeline_path(cfg, page, scene)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "scene_id": scene_id, "page": page, "scene": scene,
        "shots": shots,
    }), encoding="utf-8")
    return path


def shot(shot_id="shot_001", panel="p001_001", intent="smart_crop",
         camera="static"):
    return {
        "shot_id": shot_id, "segment_id": "seg_001", "primary_panel": panel,
        "visual_intent": intent, "camera": {"type": camera, "start": 1.0, "end": 1.0},
        "estimated_duration": 4.0, "transition": "cut",
        "match_score": 0.9, "needs_review": False, "reuse_count": 1,
    }


def run(cfg, page=1, scene=1, force=False):
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = CropPlanner(cfg).run_scene(page, scene, state, force=force)
    return result, state


def read_shot_json(cfg, shot_id, page=1, scene=1):
    path = (Path(cfg.output.shots_dir) / f"page_{page:03d}_scene_{scene:03d}"
            / f"{shot_id}.json")
    return json.loads(path.read_text("utf-8"))


def assert_in_bounds(box, width, height):
    x, y, w, h = box
    assert x >= 0 and y >= 0
    assert w > 0 and h > 0
    assert x + w <= width + 1 and y + h <= height + 1


def contains(outer, inner):
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return ox <= ix and oy <= iy and ix + iw <= ox + ow + 1 and iy + ih <= oy + oh + 1


AR = 16.0 / 9.0


def test_parse_resolution():
    assert parse_resolution("1280x720") == (1280, 720)
    assert parse_resolution("1920:1080") == (1920, 1080)
    assert parse_resolution([640, 480]) == (640, 480)
    with pytest.raises(Exception):
        parse_resolution("big")


# -------------------------------------------------- required: portrait/square


def test_portrait_panel(tmp_path):
    cfg = make_cfg(tmp_path)
    image = make_panel_image(tmp_path / "panels" / "page_001" / "panel_001.jpg", 300, 600)
    write_knowledge(cfg, [panel_record("p001_001", 1, image, 300, 600)])
    write_timeline(cfg, [shot(intent="smart_crop")])
    result, _ = run(cfg)
    assert result["result"] == "ok"
    crop = result["shots"][0]["crop"]
    assert_in_bounds((crop["x"], crop["y"], crop["width"], crop["height"]), 300, 600)
    # A 16:9 window inside a portrait panel is width-limited.
    assert crop["width"] == 300
    assert abs(crop["height"] - 300 / AR) <= 1


# ------------------------------------------------------------- required: face
# ------------------------------------------------------------- edge handling


def test_face_near_edge(tmp_path):
    cfg = make_cfg(tmp_path)
    image = make_panel_image(tmp_path / "panels" / "page_001" / "panel_001.jpg", 640, 360)
    write_knowledge(cfg, [panel_record(
        "p001_001", 1, image, 640, 360, faces=[[5, 5, 40, 50]])])
    write_timeline(cfg, [shot(intent="face_closeup")])
    result, _ = run(cfg)
    assert result["result"] == "ok"
    entry = result["shots"][0]
    assert entry["strategy"] == "16_9"
    assert entry["letterbox"] is False
    crop = (entry["crop"]["x"], entry["crop"]["y"], entry["crop"]["width"], entry["crop"]["height"])
    assert_in_bounds(crop, 640, 360)
    assert contains(crop, (5, 5, 40, 50))
    assert abs(entry["aspect_ratio"] - AR) < 0.01


def test_character_near_edge(tmp_path):
    cfg = make_cfg(tmp_path)
    image = make_panel_image(tmp_path / "panels" / "page_001" / "panel_001.jpg", 640, 360)
    write_knowledge(cfg, [panel_record(
        "p001_001", 1, image, 640, 360,
        characters=[{"name": "Guts", "bbox": [590, 300, 45, 55]}])])
    write_timeline(cfg, [shot(intent="character_closeup")])
    result, _ = run(cfg)
    entry = result["shots"][0]
    assert entry["strategy"] == "16_9"
    crop = (entry["crop"]["x"], entry["crop"]["y"], entry["crop"]["width"], entry["crop"]["height"])
    assert_in_bounds(crop, 640, 360)
    assert contains(crop, (590, 300, 45, 55))
    assert abs(entry["aspect_ratio"] - AR) < 0.01


def test_multiple_important_regions(tmp_path):
    cfg = make_cfg(tmp_path)
    image = make_panel_image(tmp_path / "panels" / "page_001" / "panel_001.jpg", 800, 400)
    write_knowledge(cfg, [panel_record(
        "p001_001", 1, image, 800, 400,
        ocr_blocks=[
            {"text": "A", "bbox": [10, 20, 120, 30], "confidence": 0.9, "type": "unknown"},
            {"text": "B", "bbox": [600, 320, 100, 28], "confidence": 0.9, "type": "unknown"},
        ])])
    write_timeline(cfg, [shot(intent="smart_crop")])
    result, _ = run(cfg)
    entry = result["shots"][0]
    crop = (entry["crop"]["x"], entry["crop"]["y"], entry["crop"]["width"], entry["crop"]["height"])
    assert_in_bounds(crop, 800, 400)
    assert contains(crop, (10, 20, 120, 30)) and contains(crop, (600, 320, 100, 28))


def test_ocr_near_edge(tmp_path):
    cfg = make_cfg(tmp_path)
    image = make_panel_image(tmp_path / "panels" / "page_001" / "panel_001.jpg", 800, 400)
    write_knowledge(cfg, [panel_record(
        "p001_001", 1, image, 800, 400,
        ocr_blocks=[{"text": "Edgy", "bbox": [770, 350, 25, 30], "confidence": 0.9, "type": "unknown"}])])
    write_timeline(cfg, [shot(intent="smart_crop")])
    result, _ = run(cfg)
    entry = result["shots"][0]
    crop = (entry["crop"]["x"], entry["crop"]["y"], entry["crop"]["width"], entry["crop"]["height"])
    assert_in_bounds(crop, 800, 400)
    assert contains(crop, (770, 350, 25, 30))


def test_crop_outside_bounds(tmp_path):
    cfg = make_cfg(tmp_path)
    image = make_panel_image(tmp_path / "panels" / "page_001" / "panel_001.jpg", 800, 400)
    # Text block whose box lies fully outside the panel - must be ignored.
    write_knowledge(cfg, [panel_record(
        "p001_001", 1, image, 800, 400,
        ocr_blocks=[{"text": "Ghost", "bbox": [900, 0, 20, 10], "confidence": 0.9, "type": "unknown"}])])
    write_timeline(cfg, [shot(intent="smart_crop")])
    result, _ = run(cfg)
    assert result["result"] == "ok"
    entry = result["shots"][0]
    crop = (entry["crop"]["x"], entry["crop"]["y"], entry["crop"]["width"], entry["crop"]["height"])
    assert_in_bounds(crop, 800, 400)
    assert abs(entry["aspect_ratio"] - AR) < 0.01
    assert entry["region_count"] == 0  # ghost box was dropped


def test_invalid_bbox(tmp_path):
    cfg = make_cfg(tmp_path)
    image = make_panel_image(tmp_path / "panels" / "page_001" / "panel_001.jpg", 640, 360)
    write_knowledge(cfg, [panel_record(
        "p001_001", 1, image, 640, 360,
        faces=[[-40, -20, 30, 30]])])  # negative bbox -> invalid
    write_timeline(cfg, [shot(intent="face_closeup")])
    result, _ = run(cfg)
    assert result["result"] == "ok"
    entry = result["shots"][0]
    crop = (entry["crop"]["x"], entry["crop"]["y"], entry["crop"]["width"], entry["crop"]["height"])
    assert_in_bounds(crop, 640, 360)
    assert abs(entry["aspect_ratio"] - AR) < 0.01


# ------------------------------------------------------------- safe strategy


def test_wide_safe_strategy_when_16x9_would_remove_content(tmp_path):
    cfg = make_cfg(tmp_path)
    image = make_panel_image(tmp_path / "panels" / "page_001" / "panel_001.jpg", 400, 800)
    write_knowledge(cfg, [panel_record(
        "p001_001", 1, image, 400, 800,
        ocr_blocks=[
            {"text": "Top", "bbox": [10, 10, 150, 30], "confidence": 0.9, "type": "unknown"},
            {"text": "Bottom", "bbox": [10, 760, 150, 30], "confidence": 0.9, "type": "unknown"},
        ])])
    write_timeline(cfg, [shot(intent="smart_crop")])
    result, _ = run(cfg)
    entry = result["shots"][0]
    assert entry["strategy"] == "safe_wider"
    assert entry["letterbox"] is True
    crop = (entry["crop"]["x"], entry["crop"]["y"], entry["crop"]["width"], entry["crop"]["height"])
    assert_in_bounds(crop, 400, 800)
    assert contains(crop, (10, 10, 150, 30)) and contains(crop, (10, 760, 150, 30))


# ----------------------------------------------------------- full panel etc.


def test_full_panel_preserves_everything(tmp_path):
    cfg = make_cfg(tmp_path)
    image = make_panel_image(tmp_path / "panels" / "page_001" / "panel_001.jpg", 300, 600)
    write_knowledge(cfg, [panel_record(
        "p001_001", 1, image, 300, 600,
        ocr_blocks=[{"text": "Any", "bbox": [10, 10, 40, 20], "confidence": 0.9, "type": "unknown"}])])
    write_timeline(cfg, [shot(intent="full_panel")])
    result, _ = run(cfg)
    entry = result["shots"][0]
    assert entry["strategy"] == "full_panel"
    assert entry["crop"] == {"x": 0, "y": 0, "width": 300, "height": 600}
    assert entry["letterbox"] is True


def test_full_panel_native_16x9_is_not_letterboxed(tmp_path):
    cfg = make_cfg(tmp_path)
    image = make_panel_image(tmp_path / "panels" / "page_001" / "panel_001.jpg", 640, 360)
    write_knowledge(cfg, [panel_record("p001_001", 1, image, 640, 360)])
    write_timeline(cfg, [shot(intent="full_panel")])
    result, _ = run(cfg)
    entry = result["shots"][0]
    assert entry["strategy"] == "full_panel"
    assert entry["letterbox"] is False


# ------------------------------------------------------------- checkpoint etc.


def test_checkpoint_and_force(tmp_path):
    cfg = make_cfg(tmp_path)
    image = make_panel_image(tmp_path / "panels" / "page_001" / "panel_001.jpg", 640, 360)
    write_knowledge(cfg, [panel_record("p001_001", 1, image, 640, 360)])
    write_timeline(cfg, [shot(intent="smart_crop")])
    first, state = run(cfg)
    assert first["result"] == "ok"
    assert state.pages.get("page_001_scene_001") == "crops_completed"
    second, _ = run(cfg)
    assert second["result"] == "skipped"
    third, _ = run(cfg, force=True)
    assert third["result"] == "ok"


def test_debug_and_crop_images_written(tmp_path):
    cfg = make_cfg(tmp_path)
    image = make_panel_image(tmp_path / "panels" / "page_001" / "panel_001.jpg", 640, 360,
                             marks=[(40, 40, 90, 70)])
    write_knowledge(cfg, [panel_record("p001_001", 1, image, 640, 360)])
    write_timeline(cfg, [shot(intent="smart_crop")])
    result, _ = run(cfg)
    entry = result["shots"][0]
    crop_dir = Path(cfg.output.crops_dir) / "page_001_scene_001"
    assert (crop_dir / "shot_001.jpg").is_file()
    assert (crop_dir / "shot_001_debug.jpg").is_file()
    payload = read_shot_json(cfg, "shot_001")
    assert payload["target"]["width"] == 1280 and payload["target"]["height"] == 720
    assert payload["strategy"] in ("16_9", "safe_wider", "full_panel")
    # Native crop output: no unnecessary upscale/downscale.
    out = cv2.imread(str(crop_dir / "shot_001.jpg"))
    crop = payload["crop"]
    assert out.shape[1] == crop["width"] and out.shape[0] == crop["height"]


def test_resolution_configurable(tmp_path):
    cfg = make_cfg(tmp_path, crops={"resolution": "1920x1080"})
    image = make_panel_image(tmp_path / "panels" / "page_001" / "panel_001.jpg", 640, 360)
    write_knowledge(cfg, [panel_record("p001_001", 1, image, 640, 360)])
    write_timeline(cfg, [shot(intent="smart_crop")])
    result, _ = run(cfg)
    assert result["target"] == {"width": 1920, "height": 1080, "aspect": "1920:1080"}
    payload = read_shot_json(cfg, "shot_001")
    assert payload["target"]["width"] == 1920


def test_unknown_intent_falls_back_to_smart_crop(tmp_path):
    cfg = make_cfg(tmp_path)
    image = make_panel_image(tmp_path / "panels" / "page_001" / "panel_001.jpg", 640, 360)
    write_knowledge(cfg, [panel_record("p001_001", 1, image, 640, 360)])
    write_timeline(cfg, [shot(intent="extreme_zoom")])
    result, _ = run(cfg)
    assert result["shots"][0]["intent"] == "smart_crop"


def test_missing_timeline_errors(tmp_path):
    cfg = make_cfg(tmp_path)
    result, _ = run(cfg)
    assert result["result"] == "error"
    assert "plan" in result["message"]


def test_missing_knowledge_errors(tmp_path):
    cfg = make_cfg(tmp_path)
    write_timeline(cfg, [shot()])
    result, _ = run(cfg)
    assert result["result"] == "error"
    assert "knowledge" in result["message"]


def test_missing_panel_image_reports_shot_error(tmp_path):
    cfg = make_cfg(tmp_path)
    # Exists on disk (passes knowledge validation) but is not decodable.
    bad = tmp_path / "panels" / "page_001" / "broken.jpg"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"not a real jpeg")
    write_knowledge(cfg, [panel_record("p001_001", 1, bad, 10, 10)])
    write_timeline(cfg, [shot()])
    result, _ = run(cfg)
    assert result["result"] == "error"
    assert any("shot_001" in error for error in result["shot_errors"])
    assert state_not_marked(cfg)


def state_not_marked(cfg):
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    return state.item_done("page_001_scene_001", "crops_completed") is False


# ------------------------------------------------------------- pure functions


def test_compute_crop_frame_math():
    box, strategy, letterbox = compute_crop(300, 600, "smart_crop", [])
    assert strategy == "16_9"
    assert letterbox is False
    assert abs(aspect(box) - AR) < 0.001
    assert_in_bounds(box, 300, 600)


def aspect(box):
    return box[2] / box[3]