"""Tests: single-page PDF extraction (PyMuPDF), low-RAM semantics."""

import subprocess
import sys
from pathlib import Path

import pytest

import pymupdf as fitz
from config.loader import Config
from pipeline.pdf_extractor import PdfExtractor
from state import State

ROOT = Path(__file__).resolve().parent.parent
STAGE_NAMES = ["extract_pages"]


def tiny_pdf(path, pages=2):
    doc = fitz.open()
    for index in range(pages):
        page = doc.new_page(width=200, height=300)
        page.draw_rect(fitz.Rect(10, 10, 190, 100), color=0, fill=0.5)
        page.insert_text((20, 140), f"page {index + 1}", fontsize=20)
    doc.save(str(path))
    doc.close()


def make_cfg(tmp_path, pdf=None):
    data = {
        "input": {"pdf": str(pdf or (tmp_path / "manga.pdf"))},
        "output": {"dir": str(tmp_path / "output"), "pages_dir": str(tmp_path / "pages")},
        "images": {
            "format": "jpg",
            "render_scale": 1.0,
            "resolution": "1200x1800",
            "max_pixels": 2500000,
            "jpeg_quality": 75,
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
    }
    return Config(data, tmp_path)


@pytest.fixture
def cfg(tmp_path):
    pdf = tmp_path / "tiny.pdf"
    tiny_pdf(pdf)
    return make_cfg(tmp_path, pdf)


def test_valid_page_extraction(cfg):
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = PdfExtractor(cfg).extract_page(1, state)
    assert result["status"] == "extracted"
    out = Path(cfg.output.pages_dir) / "page_001.jpg"
    assert out.is_file()
    assert out.stat().st_size > 0
    assert result["width"] > 0 and result["height"] > 0
    assert state.is_page_done("page_001")
    assert not state.is_page_done("page_002")


def test_extract_other_page_independently(cfg):
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    extractor = PdfExtractor(cfg)
    assert extractor.extract_page(1, state)["status"] == "extracted"
    assert extractor.extract_page(2, state)["status"] == "extracted"
    assert (Path(cfg.output.pages_dir) / "page_002.jpg").is_file()


def test_invalid_page_number(cfg):
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = PdfExtractor(cfg).extract_page(99, state)
    assert result["status"] == "error"
    assert "out of range" in result["error"]
    assert not (Path(cfg.output.pages_dir) / "page_099.jpg").exists()
    assert not state.is_page_done("page_099")


def test_page_zero_rejected(tmp_path):
    pdf = tmp_path / "tiny.pdf"
    tiny_pdf(pdf)
    cfg = make_cfg(tmp_path, pdf)
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = PdfExtractor(cfg).extract_page(0, state)
    assert result["status"] == "error"
    assert ">= 1" in result["error"]


def test_missing_pdf(tmp_path):
    cfg = make_cfg(tmp_path, pdf=tmp_path / "does_not_exist.pdf")
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = PdfExtractor(cfg).extract_page(1, state)
    assert result["status"] == "error"
    assert "not found" in result["error"].lower()
    assert not state.is_page_done("page_001")


def test_invalid_pdf(tmp_path):
    bogus = tmp_path / "bogus.pdf"
    bogus.write_bytes(b"this is not a pdf at all")
    cfg = make_cfg(tmp_path, pdf=bogus)
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = PdfExtractor(cfg).extract_page(1, state)
    assert result["status"] == "error"


def test_existing_page_checkpoint_and_skip(cfg):
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    extractor = PdfExtractor(cfg)
    first = extractor.extract_page(1, state)
    assert first["status"] == "extracted"
    assert state.is_page_done("page_001")
    second = extractor.extract_page(1, state)
    assert second["status"] == "skipped"
    fresh = State(STAGE_NAMES, cfg.pipeline.state.dir)
    third = PdfExtractor(cfg).extract_page(1, fresh)
    assert third["status"] == "skipped"


def test_output_validation_recovers_from_corrupt_file(cfg):
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    extractor = PdfExtractor(cfg)
    assert extractor.extract_page(1, state)["status"] == "extracted"
    out = Path(cfg.output.pages_dir) / "page_001.jpg"
    out.write_bytes(b"garbage" * 512)
    recover = extractor.extract_page(1, state)
    assert recover["status"] == "extracted"
    assert recover["width"] > 0 and recover["height"] > 0


def test_cli_extract_wiring(tmp_path):
    pdf = tmp_path / "tiny.pdf"
    tiny_pdf(pdf)
    conf = tmp_path / "cli.yaml"
    conf.write_text(
        f"""input:
  pdf: {pdf}
output:
  dir: {tmp_path}/output
  pages_dir: {tmp_path}/pages
images:
  format: jpg
  render_scale: 1.0
  resolution: "1200x1800"
  max_pixels: 2500000
  jpeg_quality: 75
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

    def run(page):
        return subprocess.run(
            [sys.executable, "main.py", "extract", "--page", str(page), "--config", str(conf)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )

    first = run(2)
    assert first.returncode == 0, first.stdout + first.stderr
    assert "extracted" in first.stdout
    assert (tmp_path / "pages" / "page_002.jpg").is_file()
    second = run(2)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "skipped" in second.stdout