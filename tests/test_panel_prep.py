"""Tests: panel preparation for video rendering (Task 16).

Builds tiny synthetic panel images + narration scripts in a temp tree and
verifies visuals/panels_manifest.json: dimensions, original aspect ratio,
panel_id, narration connection, and that images are never converted/re-written.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import cv2
from config.loader import Config
from pipeline.panel_prep import (
    NoPanelData,
    build_narration_index,
    prepare_panels_manifest,
    visuals_manifest_path,
)
from state import State

ROOT = Path(__file__).resolve().parent.parent


def make_cfg(tmp_path):
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
        "llm": {
            "enabled": True, "provider": "mock", "model": "", "device": "cpu",
            "max_context": 4096, "max_new_tokens": 512, "temperature": 0.7,
            "timeout_seconds": 120,
        },
        "tts": {
            "enabled": True, "engine": "auto", "provider": "mock", "voice": "en",
            "reference_audio": str(tmp_path / "input" / "voice_reference.mp3"),
            "sample_rate": 24000, "format": "wav",
            "rate_wpm": 150, "pitch_base": 50, "timeout_seconds": 60,
        },
        "video": {"resolution": "1920x1080", "fps": 30},
        "pipeline": {"batch_size": 1, "state": {"dir": str(tmp_path / "state")}, "cache": {"dir": str(tmp_path / "state" / "cache")}},
        "memory": {"guard_mb": 3072},
        "logging": {"level": "INFO", "console_level": "WARNING", "log_dir": str(tmp_path / "logs"), "max_bytes": 1048576, "backup_count": 3},
    }
    return Config(data, tmp_path)


def write_panel_image(tmp_path, panel_path, width, height):
    panel_path = Path(tmp_path) / panel_path if not Path(panel_path).is_absolute() \
        else Path(panel_path)
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (120, 120, 120)
    cv2.imwrite(str(panel_path), img)
    return panel_path.stat().st_mtime_ns


def write_knowledge(tmp_path, page, panels):
    abs_tmp = Path(tmp_path)
    path = abs_tmp / "analysis" / f"page_{page:03d}_knowledge.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "page": page,
        "reading_direction": "rtl",
        "panels": panels,
        "panel_count": len(panels),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_script(tmp_path, page, scene, segments):
    abs_tmp = Path(tmp_path)
    path = abs_tmp / "script" / f"page_{page:03d}_scene_{scene:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"page": page, "scene_id": scene,
                                "segments": segments}), encoding="utf-8")


def knowledge_panel(panel_id, page, image, reading_order=1):
    return {
        "panel_id": panel_id, "page": page, "reading_order": reading_order,
        "image": image,
        "bbox": [0, 0, 100, 120],
        "ocr": {"text": "x", "language": "en"},
        "visual": {"characters": [], "actions": []},
        "previous_panel": None, "next_panel": None, "scene_id": "scene_999",
    }


def build_demo(tmp_path):
    """3 panels on page 1 + 1 narration script linking them."""
    cfg = make_cfg(tmp_path)
    # panel images (width, height) - different aspect ratios
    write_panel_image(tmp_path, "panels/page_001/panel_001.jpg", 400, 500)
    write_panel_image(tmp_path, "panels/page_001/panel_002.jpg", 480, 360)
    write_panel_image(tmp_path, "panels/page_001/panel_003.jpg", 300, 300)

    write_knowledge(tmp_path, 1, [
        knowledge_panel("p001_001", 1, "panels/page_001/panel_001.jpg", 1),
        knowledge_panel("p001_002", 1, "panels/page_001/panel_002.jpg", 2),
        knowledge_panel("p001_003", 1, "panels/page_001/panel_003.jpg", 3),
    ])

    write_script(tmp_path, 1, 1, [
        {"segment_id": "seg_001", "text": "First panel narration.",
         "panel_ids": ["p001_001"]},
        {"segment_id": "seg_002", "text": "Second panel, a wide shot.",
         "panel_ids": ["p001_002"]},
        {"segment_id": "seg_003", "text": "Third narration.",
         "panel_ids": ["p001_003"]},
    ])
    return cfg


# ------------------------------------------------------------------ tests


def test_detects_dimensions_and_preserves_aspect_ratio(tmp_path):
    cfg = build_demo(tmp_path)
    manifest = prepare_panels_manifest(cfg, tmp_path)

    by_id = {e["panel_id"]: e for e in manifest}
    assert by_id["p001_001"]["width"] == 400
    assert by_id["p001_001"]["height"] == 500
    assert by_id["p001_001"]["aspect_ratio"] == pytest.approx(400 / 500)

    assert by_id["p001_002"]["width"] == 480
    assert by_id["p001_002"]["height"] == 360
    assert by_id["p001_002"]["aspect_ratio"] == pytest.approx(480 / 360)

    assert by_id["p001_003"]["width"] == 300
    assert by_id["p001_003"]["height"] == 300
    assert by_id["p001_003"]["aspect_ratio"] == pytest.approx(1.0)


def test_panel_id_assigned(tmp_path):
    cfg = build_demo(tmp_path)
    manifest = prepare_panels_manifest(cfg, tmp_path)
    ids = [e["panel_id"] for e in manifest]
    assert ids == ["p001_001", "p001_002", "p001_003"]


def test_connects_to_correct_narration_segment(tmp_path):
    cfg = build_demo(tmp_path)
    manifest = prepare_panels_manifest(cfg, tmp_path)
    by_id = {e["panel_id"]: e for e in manifest}
    assert by_id["p001_001"]["narration_segments"][0]["segment_id"] == "seg_001"
    assert by_id["p001_002"]["narration_segments"][0]["segment_id"] == "seg_002"
    assert by_id["p001_003"]["narration_segments"][0]["segment_id"] == "seg_003"


def test_no_image_conversion_or_rewrite(tmp_path):
    cfg = build_demo(tmp_path)
    panel_path = tmp_path / "panels" / "page_001" / "panel_001.jpg"
    before = panel_path.stat().st_mtime_ns

    prepare_panels_manifest(cfg, tmp_path)

    assert panel_path.stat().st_mtime_ns == before  # untouched
    # manifest references the original image path, not a new copy
    doc = json.loads(visuals_manifest_path(tmp_path).read_text("utf-8"))
    resolved = Path(doc[0]["image"])
    if not resolved.is_absolute():
        resolved = tmp_path / resolved
    assert resolved == tmp_path / "panels/page_001/panel_001.jpg"


def test_manifest_written_to_visuals(tmp_path):
    cfg = build_demo(tmp_path)
    prepare_panels_manifest(cfg, tmp_path)
    path = visuals_manifest_path(tmp_path)
    assert path.is_file()
    doc = json.loads(path.read_text("utf-8"))
    assert len(doc) == 3
    keys = {
        "panel_id", "page", "image", "width", "height", "aspect_ratio",
        "bbox", "reading_order", "narration_segments",
    }
    assert keys.issubset(doc[0].keys())


def test_no_panel_data_raises(tmp_path):
    cfg = make_cfg(tmp_path)
    with pytest.raises(NoPanelData):
        prepare_panels_manifest(cfg, tmp_path)


def test_panel_detector_fallback(tmp_path):
    cfg = make_cfg(tmp_path)
    write_panel_image(tmp_path, "panels/page_007/panel_001.jpg", 640, 360)
    pd = tmp_path / "panels" / "page_007" / "panels.json"
    pd.parent.mkdir(parents=True, exist_ok=True)
    pd.write_text(json.dumps({
        "page": 7,
        "source": "pages/page_007.jpg",
        "panels": [{"id": "p007_001", "image": "panels/page_007/panel_001.jpg",
                    "bbox": [0, 0, 640, 360], "area": 230400, "confidence": 0.9}],
    }), encoding="utf-8")
    manifest = prepare_panels_manifest(cfg, tmp_path)
    assert manifest[0]["panel_id"] == "p007_001"
    assert manifest[0]["width"] == 640
    assert manifest[0]["height"] == 360
    assert manifest[0]["aspect_ratio"] == pytest.approx(640 / 360)


def test_explicit_page_selection(tmp_path):
    cfg = build_demo(tmp_path)
    manifest = prepare_panels_manifest(cfg, tmp_path, page_nums=[1])
    assert [e["panel_id"] for e in manifest] == [
        "p001_001", "p001_002", "p001_003"
    ]


def test_panels_without_narration_link_have_no_segments(tmp_path):
    cfg = build_demo(tmp_path)
    manifest = prepare_panels_manifest(cfg, tmp_path,
                                       narration_index={})
    by_id = {e["panel_id"]: e for e in manifest}
    assert "narration_segments" not in by_id["p001_001"]


def test_build_narration_index(tmp_path):
    cfg = build_demo(tmp_path)
    index = build_narration_index(cfg, tmp_path)
    assert "p001_001" in index
    assert index["p001_001"][0]["segment_id"] == "seg_001"
    assert index["p001_002"][0]["text"] == "Second panel, a wide shot."
