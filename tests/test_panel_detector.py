"""Tests: manga panel detection on a single page (OpenCV, no AI model)."""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from config.loader import Config
from pipeline import geometry
from pipeline.panel_detector import PanelDetector
from state import State

ROOT = Path(__file__).resolve().parent.parent
STAGE_NAMES = ["segment_panels"]


def draw_panel(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 4)
    cx, cy = x + w // 2, y + h // 2
    cv2.circle(img, (cx, cy), min(w, h) // 4, (40, 40, 40), -1)


def synthetic_page(path, page_size=(600, 400), panels=True):
    height, width = page_size
    img = np.full((height, width, 3), 255, np.uint8)
    if panels:
        draw_panel(img, 30, 30, 160, 170)
        draw_panel(img, 220, 30, 150, 170)
        draw_panel(img, 30, 230, 160, 170)
        draw_panel(img, 220, 230, 150, 170)
        draw_panel(img, 30, 430, 160, 150)
        draw_panel(img, 220, 410, 150, 170)
    cv2.imwrite(str(path), img)


def make_cfg(tmp_path, panels_params=None):
    panels_params = dict(panels_params or {})
    data = {
        "input": {"pdf": str(tmp_path / "unused.pdf")},
        "output": {
            "dir": str(tmp_path / "output"),
            "pages_dir": str(tmp_path / "pages"),
            "panels_dir": str(tmp_path / "panels"),
        },
        "images": {
            "format": "jpg",
            "render_scale": 1.0,
            "resolution": "1200x1800",
            "max_pixels": 2500000,
            "jpeg_quality": 80,
        },
        "video": {"resolution": "1920x1080", "fps": 30},
        "pipeline": {
            "batch_size": 1,
            "state": {"dir": str(tmp_path / "state")},
            "cache": {"dir": str(tmp_path / "state" / "cache")},
        },
        "memory": {"guard_mb": 3072},
        "logging": {
            "level": "INFO",
            "console_level": "WARNING",
            "log_dir": str(tmp_path / "logs"),
            "max_bytes": 1048576,
            "backup_count": 3,
        },
        "panels": {
            "strategy": "gutter_flood",
            "border_kernel": 3,
            "close_kernel": 3,
            "min_area": 3000,
            "min_area_ratio": 0.0015,
            "max_area_ratio": 0.95,
            "min_side": 20,
            "max_aspect_ratio": 4.0,
            "min_confidence": 0.6,
            "duplicate_iou": 0.5,
            "drop_edge_touching": True,
            "max_panels": 40,
            "pad_pixels": 2,
            "jpeg_quality": 85,
        },
    }
    data["panels"].update(panels_params)
    return Config(data, tmp_path)


@pytest.fixture
def cfg(tmp_path):
    pages = tmp_path / "pages"
    pages.mkdir()
    synthetic_page(pages / "page_001.jpg")
    return make_cfg(tmp_path)


def test_page_missing(tmp_path):
    cfg = make_cfg(tmp_path)
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = PanelDetector(cfg).detect_page(1, state)
    assert result["status"] == "error"
    assert "not found" in result["error"]


def test_valid_page_detection(cfg):
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = PanelDetector(cfg).detect_page(1, state)
    assert result["status"] == "detected"
    assert result["count"] == 6


def test_blank_page_detects_zero_panels(tmp_path):
    pages = tmp_path / "pages"
    pages.mkdir()
    synthetic_page(pages / "page_001.jpg", panels=False)
    cfg = make_cfg(tmp_path)
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = PanelDetector(cfg).detect_page(1, state)
    assert result["status"] == "detected"
    assert result["count"] == 0


def test_detection_outputs_exist(cfg):
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = PanelDetector(cfg).detect_page(1, state)
    out = Path(cfg.output.panels_dir) / "page_001"
    assert out.is_dir()
    for index in range(1, result["count"] + 1):
        assert (out / f"panel_{index:03d}.jpg").is_file()
        assert (out / f"panel_{index:03d}.jpg").stat().st_size > 0
    assert (out / "debug.jpg").is_file()
    assert (out / "panels.json").is_file()


def test_panels_json_is_valid(cfg):
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    PanelDetector(cfg).detect_page(1, state)
    manifest = Path(cfg.output.panels_dir) / "page_001" / "panels.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["page"] == 1
    assert data["source"] == "pages/page_001.jpg"
    assert isinstance(data["panels"], list)
    assert len(data["panels"]) == 6
    for panel in data["panels"]:
        assert panel["id"].startswith("p001_")
        assert panel["image"] == f"panels/page_001/{Path(panel['image']).name}"
        assert len(panel["bbox"]) == 4
        assert panel["area"] > 0
        assert 0.0 < panel["confidence"] <= 1.0


def test_bounding_boxes_valid_within_bounds(cfg):
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    PanelDetector(cfg).detect_page(1, state)
    manifest = Path(cfg.output.panels_dir) / "page_001" / "panels.json"
    img = cv2.imread(str(Path(cfg.output.pages_dir) / "page_001.jpg"))
    height, width = img.shape[:2]
    data = json.loads(manifest.read_text(encoding="utf-8"))
    for panel in data["panels"]:
        x, y, w, h = panel["bbox"]
        assert w > 0 and h > 0
        assert x >= 0 and y >= 0
        assert x + w <= width
        assert y + h <= height


def test_duplicate_and_nested_boxes_removed():
    params = {
        "min_area": 100,
        "min_area_ratio": 0.0,
        "max_area_ratio": 0.95,
        "min_side": 5,
        "max_aspect_ratio": 5.0,
        "min_confidence": 0.0,
        "duplicate_iou": 0.5,
        "drop_edge_touching": False,
        "max_panels": 40,
    }
    boxes = [
        {"box": [10, 10, 100, 100], "area": 10000, "confidence": 0.9},
        {"box": [12, 12, 100, 100], "area": 9600, "confidence": 0.85},
        {"box": [150, 150, 40, 40], "area": 1600, "confidence": 0.9},
        {"box": [146, 146, 200, 200], "area": 40000, "confidence": 0.95},
    ]
    cleaned = geometry.filter_boxes(boxes, 400, 600, params)
    assert len(cleaned) == 2
    assert [10, 10, 100, 100] in [item["box"] for item in cleaned]
    assert [146, 146, 200, 200] in [item["box"] for item in cleaned]


def test_checkpoint_skip(cfg):
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    detector = PanelDetector(cfg)
    first = detector.detect_page(1, state)
    assert first["status"] == "detected"
    second = detector.detect_page(1, state)
    assert second["status"] == "skipped"
    fresh = State(STAGE_NAMES, cfg.pipeline.state.dir)
    assert fresh.is_page_done("panels_page_001")


def test_force_redetects(cfg):
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    detector = PanelDetector(cfg)
    assert detector.detect_page(1, state)["status"] == "detected"
    forced = detector.detect_page(1, state, force=True)
    assert forced["status"] == "detected"
    assert forced["count"] == 6


def test_cli_panels_wiring(tmp_path):
    pages = tmp_path / "pages"
    pages.mkdir()
    synthetic_page(pages / "page_001.jpg")
    conf = tmp_path / "cli.yaml"
    conf.write_text(
        f"""input:
  pdf: {tmp_path}/unused.pdf
output:
  dir: {tmp_path}/output
  pages_dir: {tmp_path}/pages
  panels_dir: {tmp_path}/panels
images:
  format: jpg
  render_scale: 1.0
  resolution: "1200x1800"
  max_pixels: 2500000
  jpeg_quality: 80
panels:
  strategy: gutter_flood
  min_area: 3000
  duplicate_iou: 0.5
video:
  resolution: "1920x1080"
  fps: 30
pipeline:
  batch_size: 1
  state:
    dir: {tmp_path}/state
  cache:
    dir: {tmp_path}/state/cache
memory:
  guard_mb: 3072
logging:
  level: INFO
  console_level: WARNING
  log_dir: {tmp_path}/logs
  max_bytes: 1048576
  backup_count: 3
""",
        encoding="utf-8",
    )

    def run(*args):
        return subprocess.run(
            [sys.executable, "main.py", "panels", *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )

    first = run("--page", "1", "--config", str(conf))
    assert first.returncode == 0, first.stdout + first.stderr
    assert "detected 6 panel" in first.stdout
    assert (tmp_path / "panels" / "page_001" / "panels.json").is_file()
    second = run("--page", "1", "--config", str(conf))
    assert second.returncode == 0
    assert "skipped" in second.stdout
    forced = run("--page", "1", "--config", str(conf), "--force")
    assert forced.returncode == 0
    assert "detected 6 panel" in forced.stdout