"""Tests: OCR of one manga panel with synthetic images and a fake provider."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from config.loader import Config
from pipeline import ocr_provider
from pipeline.ocr_processor import (
    OcrProcessor,
    classify_text_block,
    preprocess_image,
)
from pipeline.ocr_provider import OCREngineUnavailable, create_provider
from state import State

ROOT = Path(__file__).resolve().parent.parent
STAGE_NAMES = ["run_ocr"]


class FakeProvider:
    name = "fake"
    blocks = []

    def __init__(self, blocks=None, raise_on_recognize=None):
        self.blocks = list(blocks or self.blocks)
        self.raised = raise_on_recognize

    def recognize(self, image_path):
        if self.raised is not None:
            raise self.raised
        return list(self.blocks)


ONE_BLOCK = [
    {
        "text": "「こんにちは」",
        "bbox": [10, 16, 120, 30],
        "confidence": 0.93,
    }
]


def make_cfg(tmp_path, preprocess=None, ocr=None):
    preprocess = preprocess or {}
    ocr = ocr or {}
    data = {
        "input": {"pdf": str(tmp_path / "unused.pdf")},
        "output": {
            "dir": str(tmp_path / "output"),
            "pages_dir": str(tmp_path / "pages"),
            "panels_dir": str(tmp_path / "panels"),
            "ocr_dir": str(tmp_path / "ocr"),
        },
        "images": {"format": "jpg", "render_scale": 1.0, "jpeg_quality": 80},
        "panels": {"jpeg_quality": 85},
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
        "preprocess": {
            "enabled": True,
            "grayscale": True,
            "contrast": 1.4,
            "denoise": "none",
            "denoise_kernel": 3,
            "resize_scale": 1.5,
            "resize_max_pixels": 800000,
            "threshold": "none",
        },
        "ocr": {
            "engine": "auto",
            "language": "eng+jpn",
            "psm": 11,
            "timeout_seconds": 30,
        },
    }
    data["preprocess"].update(preprocess)
    data["ocr"].update(ocr)
    return Config(data, tmp_path)


def write_panel(cfg, page=1, panel=1, kind="normal"):
    out = Path(cfg.output.panels_dir) / f"page_{page:03d}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"panel_{panel:03d}.jpg"
    if kind == "normal":
        img = np.full((400, 300, 3), 255, np.uint8)
        cv2.rectangle(img, (8, 10), (260, 220), (40, 40, 40), -1)
    elif kind == "blank":
        img = np.full((400, 300, 3), 255, np.uint8)
    elif kind == "corrupt":
        path.write_bytes(b"this is not a jpeg")
        return path
    cv2.imwrite(str(path), img)
    return path


def run(cfg, page=1, panel=1, provider=None, force=False):
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = OcrProcessor(cfg, provider=provider).run_panel(page, panel, state, force=force)
    return result, state


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_valid_panel(tmp_path):
    cfg = make_cfg(tmp_path, preprocess={"resize_scale": 1.0})
    write_panel(cfg, kind="normal")
    result, state = run(cfg, provider=FakeProvider(ONE_BLOCK))
    assert result["status"] == "detected"
    assert result["engine"] == "fake"
    assert result["count"] == 1
    block = result["blocks"][0]
    assert block["text"] == "「こんにちは」"
    assert block["type"] == "dialogue"
    assert block["confidence"] == 0.93
    assert result["combined_text"] == "「こんにちは」"
    assert state.ocr_done("page_001_panel_001")


def test_missing_panel(tmp_path):
    cfg = make_cfg(tmp_path)
    result, state = run(cfg, provider=FakeProvider(ONE_BLOCK))
    assert result["status"] == "error"
    assert "not found" in result["error"]
    assert not state.ocr_done("page_001_panel_001")


def test_blank_panel_and_no_detected_text(tmp_path):
    cfg = make_cfg(tmp_path, preprocess={"resize_scale": 1.0})
    write_panel(cfg, kind="blank")
    result, state = run(cfg, provider=FakeProvider([]))
    assert result["status"] == "detected"
    assert result["count"] == 0
    assert result["blocks"] == []
    assert result["combined_text"] == ""
    assert state.ocr_done("page_001_panel_001")
    saved = read_json(Path(cfg.output.ocr_dir) / "page_001_panel_001.json")
    assert saved["text_blocks"] == []
    assert saved["combined_text"] == ""


def test_ocr_json_validity(tmp_path):
    cfg = make_cfg(tmp_path, preprocess={"resize_scale": 1.0})
    write_panel(cfg, kind="normal")
    run(cfg, provider=FakeProvider(ONE_BLOCK))
    path = Path(cfg.output.ocr_dir) / "page_001_panel_001.json"
    saved = read_json(path)
    assert saved["page"] == 1
    assert saved["panel"] == 1
    assert saved["engine"] == "fake"
    assert isinstance(saved["text_blocks"], list)
    assert isinstance(saved["combined_text"], str)
    assert saved["combined_text"] == saved["text_blocks"][0]["text"]


def test_bounding_box_validity(tmp_path):
    cfg = make_cfg(tmp_path, preprocess={"resize_scale": 1.0})
    write_panel(cfg, kind="normal")
    big = [{"text": "x", "bbox": [500, 600, 200, 300], "confidence": 0.5}]
    result, _ = run(cfg, provider=FakeProvider(big))
    x, y, w, h = result["blocks"][0]["bbox"]
    assert x >= 0 and y >= 0
    assert w >= 1 and h >= 1
    assert x + w <= 300 and y + h <= 400


def test_preprocessing_resize_maps_bbox_back(tmp_path):
    cfg = make_cfg(tmp_path, preprocess={"resize_scale": 2.0})
    write_panel(cfg, kind="normal")
    block = [{"text": "hi", "bbox": [20, 40, 60, 20], "confidence": 0.8}]
    result, _ = run(cfg, provider=FakeProvider(block))
    x, y, w, h = result["blocks"][0]["bbox"]
    assert (x, y, w, h) == (10, 20, 30, 10)


def test_checkpoint_skip(tmp_path):
    cfg = make_cfg(tmp_path, preprocess={"resize_scale": 1.0})
    write_panel(cfg, kind="normal")
    detector = OcrProcessor(cfg, provider=FakeProvider(ONE_BLOCK))
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    first = detector.run_panel(1, 1, state)
    assert first["status"] == "detected"
    second = detector.run_panel(1, 1, state)
    assert second["status"] == "skipped"
    check = json.loads(
        (Path(cfg.pipeline.state.dir) / "checkpoints.json").read_text(encoding="utf-8")
    )
    assert check["pages"]["page_001_panel_001"] == "ocr_completed"


def test_force_recomputes(tmp_path):
    cfg = make_cfg(tmp_path, preprocess={"resize_scale": 1.0})
    write_panel(cfg, kind="normal")
    detector = OcrProcessor(cfg, provider=FakeProvider(ONE_BLOCK))
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    assert detector.run_panel(1, 1, state)["status"] == "detected"
    forced = detector.run_panel(1, 1, state, force=True)
    assert forced["status"] == "detected"
    assert forced["count"] == 1


def test_ocr_engine_unavailable(tmp_path, monkeypatch):
    write_panel(make_cfg(tmp_path), kind="normal")
    cfg = make_cfg(tmp_path, ocr={"engine": "auto"})
    monkeypatch.setattr(shutil, "which", lambda *a, **k: None)
    monkeypatch.setattr(ocr_provider.PaddleOCRProvider, "available", staticmethod(lambda: False))
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = OcrProcessor(cfg).run_panel(1, 1, state)
    assert result["status"] == "error"
    assert "unavailable" in result["error"].lower()
    assert not state.ocr_done("page_001_panel_001")


def test_ocr_timeout_is_an_error(tmp_path):
    cfg = make_cfg(tmp_path, preprocess={"resize_scale": 1.0})
    write_panel(cfg, kind="normal")
    result, state = run(
        cfg,
        provider=FakeProvider(raise_on_recognize=ocr_provider.OCRTimeout("tesseract timed out after 30s")),
    )
    assert result["status"] == "error"
    assert "timed out" in result["error"]
    assert not state.ocr_done("page_001_panel_001")


def test_corrupt_image_is_an_error(tmp_path):
    cfg = make_cfg(tmp_path, preprocess={"resize_scale": 1.0})
    write_panel(cfg, kind="corrupt")
    result, state = run(cfg, provider=FakeProvider(ONE_BLOCK))
    assert result["status"] == "error"
    assert "corrupt" in result["error"] or "read" in result["error"]


def test_invalid_page_or_panel_number(tmp_path):
    cfg = make_cfg(tmp_path)
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    processor = OcrProcessor(cfg, provider=FakeProvider(ONE_BLOCK))
    assert processor.run_panel(0, 1, state)["status"] == "error"
    assert processor.run_panel(1, 0, state)["status"] == "error"


def test_preprocess_steps_unit():
    img = np.full((100, 200, 3), 200, np.uint8)
    cv2.rectangle(img, (10, 10), (60, 60), (10, 10, 10), -1)
    out, scale = preprocess_image(img, {"enabled": True, "grayscale": True,
                                         "contrast": 1.4, "denoise": "none",
                                         "resize_scale": 2.0,
                                         "resize_max_pixels": 1000000,
                                         "threshold": "none"})
    assert len(out.shape) == 2
    assert out.shape[0] == 200 and out.shape[1] == 400
    assert scale == 2.0
    thr_out, _ = preprocess_image(out, {"enabled": True, "grayscale": False,
                                         "contrast": 1.0, "denoise": "none",
                                         "resize_scale": 1.0, "threshold": "otsu"})
    assert len(thr_out.shape) == 2
    assert set(np.unique(thr_out)) <= {0, 255}


def test_classification_rules():
    assert classify_text_block("「わかる」") == "dialogue"
    assert classify_text_block('"hello"') == "dialogue"
    assert classify_text_block("ドンッ") == "sfx"
    assert classify_text_block("!!") == "sfx"
    assert classify_text_block("こんにちは") == "unknown"
    assert classify_text_block("") == "unknown"
    assert classify_text_block(" ") == "unknown"


def test_create_provider_raises_when_no_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda *a, **k: None)
    monkeypatch.setattr(ocr_provider.PaddleOCRProvider, "available", staticmethod(lambda: False))
    cfg = make_cfg(tmp_path, ocr={"engine": "auto"})
    with pytest.raises(OCREngineUnavailable):
        create_provider(cfg)
    with pytest.raises(ocr_provider.OCRProviderError):
        create_provider(make_cfg(tmp_path, ocr={"engine": "bogus"}))


def test_cli_ocr_wiring(tmp_path):
    cfg = make_cfg(tmp_path, ocr={"engine": "dummy"}, preprocess={"resize_scale": 1.0})
    write_panel(cfg, kind="normal")
    Path(cfg.output.ocr_dir).mkdir(parents=True, exist_ok=True)
    conf = tmp_path / "cli.yaml"
    conf.write_text(
        f"""input:
  pdf: {tmp_path}/unused.pdf
output:
  dir: {tmp_path}/output
  pages_dir: {tmp_path}/pages
  panels_dir: {tmp_path}/panels
  ocr_dir: {tmp_path}/ocr
images:
  format: jpg
  render_scale: 1.0
  jpeg_quality: 80
panels:
  jpeg_quality: 85
ocr:
  engine: dummy
  language: eng
  psm: 11
  timeout_seconds: 30
preprocess:
  enabled: true
  grayscale: true
  contrast: 1.4
  denoise: none
  denoise_kernel: 3
  resize_scale: 1.0
  resize_max_pixels: 800000
  threshold: none
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
            [sys.executable, "main.py", "ocr", *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )

    first = run("--page", "1", "--panel", "1", "--config", str(conf))
    assert first.returncode == 0, first.stdout + first.stderr
    assert "engine          : dummy" in first.stdout
    assert (Path(cfg.output.ocr_dir) / "page_001_panel_001.json").is_file()
    assert (Path(cfg.output.ocr_dir) / "page_001_panel_001_debug.jpg").is_file()
    assert not list(Path(cfg.output.ocr_dir).glob("manga_ocr_*.png"))
    second = run("--page", "1", "--panel", "1", "--config", str(conf))
    assert second.returncode == 0
    assert "skipped" in second.stdout
    forced = run("--page", "1", "--panel", "1", "--config", str(conf), "--force")
    assert forced.returncode == 0
    assert "block(s)" in forced.stdout