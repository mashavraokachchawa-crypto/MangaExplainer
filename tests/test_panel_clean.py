"""Tests: panel cleaning (banner strips, text boxes, SFX stamps).

Builds synthetic panel images with the three overlay types and verifies
detection, inpainting, manifest/debug output, and the clean_panel_source
fallback that OCR / the analysis processor rely on.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from config.loader import Config
from pipeline.panel_clean import (
    clean_panel,
    clean_panel_source,
    detect_and_clean,
)

W, H = 360, 300


def make_cfg(tmp_path, enabled=True):
    data = {
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
        "images": {"format": "jpg", "render_scale": 1.0, "jpeg_quality": 80},
        "panels": {
            "jpeg_quality": 85, "min_area": 1000, "max_panels": 40,
            "clean": {
                "enabled": enabled,
                "strip_ratio": 0.92,
                "strip_max_height_ratio": 0.18,
                "strip_row_dev": 24.0,
                "min_textbox_area_ratio": 0.02,
                "max_textbox_area_ratio": 0.55,
                "min_sfx_area_ratio": 0.015,
                "bbox_pad": 6,
                "inpaint_radius": 4,
                "write_clean": True,
                "debug": True,
            },
        },
        "pipeline": {"state": {"dir": str(tmp_path / "state")}},
        "logging": {"log_dir": str(tmp_path / "logs")},
    }
    return Config(data, tmp_path)


def synthetic_panel():
    """A textured panel carrying a banner, a caption box and an SFX stamp."""
    rng = np.random.default_rng(7)
    img = rng.integers(90, 150, (H, W, 3), dtype=np.uint8)

    # banner strip along the top edge: uniform, no texture
    cv2.rectangle(img, (0, 0), (W, 33), (40, 40, 40), -1)

    # caption/text box: bright fill, closed dark border
    x, y, bw, bh = 56, 146, 128, 62
    cv2.rectangle(img, (x - 4, y - 4), (x + bw + 4, y + bh + 4), (26, 26, 26), -1)
    cv2.rectangle(img, (x, y), (x + bw, y + bh), (245, 245, 245), -1)
    cv2.line(img, (x + 12, y + 18), (x + bw - 12, y + 18), (20, 20, 20), 4)

    # SFX stamp: a big near-black mass (punched holes keep it under the
    # solidity cap so it reads as a stamp, not a dead rectangle, and it
    # stays one connected component above min size)
    cv2.rectangle(img, (20, 245), (96, 291), (16, 16, 16), -1)
    for cx, cy in ((38, 258), (62, 276), (80, 253), (46, 284)):
        cv2.circle(img, (cx, cy), 4, (110, 110, 110), -1)

    return img


def write_panel(tmp_path, img):
    src = Path(tmp_path) / "panels" / "page_001" / "panel_001.jpg"
    src.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(src), img)
    assert ok
    return src


def region_center(r):
    return (r["x"] + r["w"] / 2, r["y"] + r["h"] / 2)


def near(a, b, tol=26):
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def test_detect_finds_banner_textbox_sfx():
    img = synthetic_panel()
    settings = make_cfg(Path(".")).panels.get("clean")
    cleaned, regions = detect_and_clean(img, settings)

    kinds = [r["kind"] for r in regions]
    assert "banners" in kinds
    assert "textboxes" in kinds
    assert "sfx" in kinds

    banner = next(r for r in regions if r["kind"] == "banners")
    assert banner["y"] < 40  # attached to the top edge

    box = next(r for r in regions if r["kind"] == "textboxes")
    assert near(region_center(box), (56 + 64, 146 + 31))

    sfx = next(r for r in regions if r["kind"] == "sfx")
    assert near(region_center(sfx), (60, 270))

    # inpainting actually changed the pixels inside each found region
    changed = int(np.count_nonzero(cv2.absdiff(cleaned, img)))
    assert changed > 0


def test_clean_panel_writes_clean_debug_manifest(tmp_path):
    cfg = make_cfg(tmp_path)
    write_panel(tmp_path, synthetic_panel())

    res = clean_panel(cfg, tmp_path, 1, 1, force=False)
    assert res["status"] == "cleared"
    assert res["removed"]["banners"] >= 1
    assert res["removed"]["textboxes"] >= 1
    assert res["removed"]["sfx"] >= 1

    cleaned = Path(res["cleaned"])
    assert cleaned.is_file()
    assert Path(res["debug"]).is_file()

    manifest = Path(tmp_path) / "panels_clean" / "page_001" / "clean_manifest.json"
    assert manifest.is_file()
    doc = json.loads(manifest.read_text("utf-8"))
    assert "1" in doc
    assert doc["1"]["panel"] == 1
    assert doc["1"]["removed"]["banners"] >= 1


def test_clean_panel_skips_when_already_done(tmp_path):
    cfg = make_cfg(tmp_path)
    write_panel(tmp_path, synthetic_panel())
    clean_panel(cfg, tmp_path, 1, 1, force=False)
    res = clean_panel(cfg, tmp_path, 1, 1, force=False)
    assert res["status"] == "skipped"
    assert res["reason"] == "already cleaned"


def test_clean_panel_disabled_is_noop(tmp_path):
    cfg = make_cfg(tmp_path, enabled=False)
    write_panel(tmp_path, synthetic_panel())
    res = clean_panel(cfg, tmp_path, 1, 1, force=False)
    assert res["status"] == "skipped"
    assert res["reason"] == "clean disabled in config"
    assert not (Path(tmp_path) / "panels_clean").exists()


def test_clean_panel_source_fallback(tmp_path):
    cfg = make_cfg(tmp_path)
    write_panel(tmp_path, synthetic_panel())
    # before cleaning: cleans don't exist -> None (callers fall back to raw)
    assert clean_panel_source(cfg, 1, 1) is None

    clean_panel(cfg, tmp_path, 1, 1, force=False)
    path = clean_panel_source(cfg, 1, 1)
    assert path is not None
    assert path.name == "panel_001.jpg"
    assert "panels_clean" in str(path)

    # disabled cleaning -> always None
    cfg.disabled = make_cfg(tmp_path, enabled=False)
    assert clean_panel_source(cfg.disabled, 1, 1) is None


def test_clean_panel_missing_input_is_safe(tmp_path):
    cfg = make_cfg(tmp_path)
    res = clean_panel(cfg, tmp_path, 1, 1, force=False)
    assert res["status"] == "skipped"
    assert "not found" in (res.get("reason") or "")