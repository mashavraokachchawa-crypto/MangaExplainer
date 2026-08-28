"""Tests: manga panel reading order from synthetic JSON bounding boxes."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from config.loader import Config
from pipeline.reading_order import ReadingOrder
from state import State

ROOT = Path(__file__).resolve().parent.parent
STAGE_NAMES = ["segment_panels"]

GRID_RTL = [
    {"id": "p1", "bbox": [32, 32, 157, 167]},
    {"id": "p2", "bbox": [222, 32, 147, 167]},
    {"id": "p3", "bbox": [32, 232, 157, 167]},
    {"id": "p4", "bbox": [222, 232, 147, 167]},
    {"id": "p5", "bbox": [32, 432, 157, 150]},
    {"id": "p6", "bbox": [222, 432, 147, 150]},
]

DIFFERENT_SIZES = [
    {"id": "p1", "bbox": [260, 30, 120, 300]},
    {"id": "p2", "bbox": [30, 30, 200, 170]},
    {"id": "p3", "bbox": [30, 230, 200, 210]},
    {"id": "p4", "bbox": [260, 350, 120, 90]},
]

SPANNING_COLUMNS = [
    {"id": "p1", "bbox": [200, 20, 140, 160]},
    {"id": "p2", "bbox": [20, 20, 140, 400]},
    {"id": "p3", "bbox": [200, 200, 140, 160]},
    {"id": "p4", "bbox": [200, 380, 140, 160]},
]

OVERLAPPING = [
    {"id": "q1", "bbox": [250, 30, 130, 180]},
    {"id": "q2", "bbox": [120, 50, 150, 160]},
    {"id": "q3", "bbox": [30, 240, 340, 160]},
]


def make_cfg(tmp_path, direction="rtl"):
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
        "panels": {"jpeg_quality": 85},
        "reading": {
            "direction": direction,
            "row_overlap_ratio": 0.5,
            "weights": {
                "row_overlap": 2.0,
                "horizontal": 1.5,
                "vertical": 0.3,
                "distance": 0.5,
                "size": 0.0,
            },
        },
    }
    return Config(data, tmp_path)


def write_manifest(cfg, panels, page=1):
    out = Path(cfg.output.panels_dir) / f"page_{page:03d}"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "page": page,
        "source": f"pages/page_{page:03d}.jpg",
        "panels": panels,
    }
    (out / "panels.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return out


def run(cfg, panels, page=1, force=False):
    out_dir = write_manifest(cfg, panels, page)
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    return ReadingOrder(cfg).detect_page(page, state, force=force), out_dir


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_rtl_normal_layout(tmp_path):
    cfg = make_cfg(tmp_path, direction="rtl")
    result, out = run(cfg, GRID_RTL)
    assert result["status"] == "detected"
    assert result["order"] == ["p2", "p1", "p4", "p3", "p6", "p5"]
    saved = read_json(out / "reading_order.json")
    assert saved["page"] == 1
    assert saved["direction"] == "rtl"
    assert saved["order"] == ["p2", "p1", "p4", "p3", "p6", "p5"]


def test_ltr_layout(tmp_path):
    cfg = make_cfg(tmp_path, direction="ltr")
    result, _ = run(cfg, GRID_RTL)
    assert result["order"] == ["p1", "p2", "p3", "p4", "p5", "p6"]


def test_panels_json_receives_reading_order(tmp_path):
    cfg = make_cfg(tmp_path)
    result, out = run(cfg, GRID_RTL)
    updated = read_json(out / "panels.json")
    ranks = {p["id"]: p["reading_order"] for p in updated["panels"]}
    assert ranks == {"p1": 2, "p2": 1, "p3": 4, "p4": 3, "p5": 6, "p6": 5}
    for panel in updated["panels"]:
        assert isinstance(panel["reading_order"], int)


def test_different_panel_sizes(tmp_path):
    cfg = make_cfg(tmp_path)
    result, _ = run(cfg, DIFFERENT_SIZES)
    assert result["order"] == ["p1", "p2", "p3", "p4"]


def test_panel_spanning_multiple_rows(tmp_path):
    cfg = make_cfg(tmp_path)
    result, _ = run(cfg, SPANNING_COLUMNS)
    assert result["order"] == ["p1", "p2", "p3", "p4"]


def test_overlapping_and_nearby_panels(tmp_path):
    cfg = make_cfg(tmp_path)
    result, _ = run(cfg, OVERLAPPING)
    assert result["order"] == ["q1", "q2", "q3"]


def test_single_panel_page(tmp_path):
    cfg = make_cfg(tmp_path)
    result, out = run(cfg, [{"id": "p1", "bbox": [30, 30, 340, 540]}])
    assert result["status"] == "detected"
    assert result["order"] == ["p1"]
    assert result["ordered"] == 1
    assert (out / "reading_order.json").is_file()
    assert (out / "reading_order_debug.jpg").is_file()


def test_empty_panel_list(tmp_path):
    cfg = make_cfg(tmp_path)
    result, out = run(cfg, [])
    assert result["status"] == "detected"
    assert result["order"] == []
    assert result["count"] == 0
    saved = read_json(out / "reading_order.json")
    assert saved["order"] == []


def test_invalid_bounding_box_is_ignored(tmp_path):
    cfg = make_cfg(tmp_path)
    panels = [
        {"id": "p1", "bbox": [30, 30, 340, 540]},
        {"id": "bad", "bbox": [30, 30, 0, 100]},
    ]
    result, out = run(cfg, panels)
    assert result["status"] == "detected"
    assert result["order"] == ["p1"]
    assert result["ignored"] == ["bad"]
    updated = read_json(out / "panels.json")
    by_id = {p["id"]: p for p in updated["panels"]}
    assert by_id["p1"]["reading_order"] == 1
    assert by_id["bad"]["reading_order"] is None
    saved = read_json(out / "reading_order.json")
    assert saved["order"] == ["p1"]
    assert saved["ignored"] == ["bad"]


def test_missing_panels_json_is_an_error(tmp_path):
    cfg = make_cfg(tmp_path)
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = ReadingOrder(cfg).detect_page(1, state)
    assert result["status"] == "error"
    assert "panels" in result["error"].lower()


def test_unknown_direction_rejected(tmp_path):
    with pytest.raises(ValueError):
        ReadingOrder(make_cfg(tmp_path, direction="rtl")).compute_order(
            GRID_RTL, direction="sideways"
        )
    with pytest.raises(ValueError):
        ReadingOrder(make_cfg(tmp_path, direction="up"))


def test_checkpoint_skip(tmp_path):
    cfg = make_cfg(tmp_path)
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    detector = ReadingOrder(cfg)
    out_dir = write_manifest(cfg, GRID_RTL)
    first = detector.detect_page(1, state)
    assert first["status"] == "detected"
    second = detector.detect_page(1, state)
    assert second["status"] == "skipped"
    assert state.is_page_done("order_page_001")
    assert (out_dir / "reading_order.json").is_file()


def test_force_recomputes(tmp_path):
    cfg = make_cfg(tmp_path)
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    detector = ReadingOrder(cfg)
    write_manifest(cfg, GRID_RTL)
    assert detector.detect_page(1, state)["status"] == "detected"
    forced = detector.detect_page(1, state, force=True)
    assert forced["status"] == "detected"
    assert forced["order"] == ["p2", "p1", "p4", "p3", "p6", "p5"]


def test_cli_order_wiring(tmp_path):
    panels_dir = tmp_path / "panels" / "page_001"
    panels_dir.mkdir(parents=True)
    (panels_dir / "panels.json").write_text(
        json.dumps(
            {"page": 1, "source": "pages/page_001.jpg", "panels": GRID_RTL},
            indent=2,
        ),
        encoding="utf-8",
    )
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
  jpeg_quality: 85
reading:
  direction: rtl
  row_overlap_ratio: 0.5
  weights:
    row_overlap: 2.0
    horizontal: 1.5
    vertical: 0.3
    distance: 0.5
    size: 0.0
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
            [sys.executable, "main.py", "order", *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )

    first = run("--page", "1", "--config", str(conf))
    assert first.returncode == 0, first.stdout + first.stderr
    assert "ordered 6/6" in first.stdout
    assert "p2" in first.stdout
    assert (panels_dir / "reading_order.json").is_file()
    assert (panels_dir / "reading_order_debug.jpg").is_file()
    second = run("--page", "1", "--config", str(conf))
    assert second.returncode == 0
    assert "skipped" in second.stdout
    forced = run("--page", "1", "--config", str(conf), "--force")
    assert forced.returncode == 0
    assert "ordered 6/6" in forced.stdout