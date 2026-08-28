"""Tests: one-panel VLM analysis with a mock provider (no real VLM needed)."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from config.loader import Config
from pipeline.analysis_processor import AnalysisProcessor, sanitize_analysis
from pipeline.prompts import build_analysis_prompt
from pipeline.vlm_provider import (
    MockVLMProvider,
    VLMNotConfigured,
    VLMProviderError,
    VLMTimeout,
    VLMFailure,
    create_vlm_provider,
    extract_json,
)
from state import State

ROOT = Path(__file__).resolve().parent.parent
STAGE_NAMES = ["analyze_panels"]

VALID_RESPONSE = {
    "characters": [
        {
            "name": "unknown",
            "description": "a swordsman in dark armor",
            "action": "raising a sword",
            "emotion": "determined",
        }
    ],
    "environment": "a battlefield",
    "actions": ["raising a sword", "stepping forward"],
    "objects": ["sword"],
    "visual_effects": ["speed lines"],
    "important_event": "the swordsman prepares to strike",
    "composition": "medium shot focused on the swordsman",
    "story_relevance": "unknown",
    "confidence": 0.85,
}


def make_cfg(tmp_path, vlm=None, analysis_dir="analysis"):
    vlm = vlm or {}
    data = {
        "input": {"pdf": str(tmp_path / "unused.pdf")},
        "output": {
            "dir": str(tmp_path / "output"),
            "pages_dir": str(tmp_path / "pages"),
            "panels_dir": str(tmp_path / "panels"),
            "ocr_dir": str(tmp_path / "ocr"),
            "analysis_dir": str(tmp_path / analysis_dir),
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
        "ocr": {
            "engine": "auto",
            "language": "eng+jpn",
            "psm": 11,
            "timeout_seconds": 30,
        },
        "vlm": {
            "enabled": True,
            "provider": "mock",
            "model": "",
            "device": "cpu",
            "max_image_size": 768,
            "max_new_tokens": 256,
            "timeout_seconds": 120,
        },
    }
    data["vlm"].update(vlm)
    return Config(data, tmp_path)


def write_panel(cfg, page=1, panel=1, kind="normal"):
    out = Path(cfg.output.panels_dir) / f"page_{page:03d}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"panel_{panel:03d}.jpg"
    if kind == "corrupt":
        path.write_bytes(b"definitely not a jpeg")
        return path
    img = np.full((400, 300, 3), 200, np.uint8)
    cv2.rectangle(img, (10, 10), (280, 380), (50, 50, 50), -1)
    cv2.imwrite(str(path), img)
    return path


def write_ocr(cfg, page=1, panel=1, text="「こんにちは」この勇者"):
    ocr_dir = Path(cfg.output.ocr_dir)
    ocr_dir.mkdir(parents=True, exist_ok=True)
    key = f"page_{page:03d}_panel_{panel:03d}"
    (ocr_dir / f"{key}.json").write_text(
        json.dumps({"page": page, "panel": panel, "combined_text": text}),
        encoding="utf-8",
    )


def run(cfg, page=1, panel=1, provider=None, force=False):
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = AnalysisProcessor(cfg, provider=provider).run_panel(
        page, panel, state, force=force
    )
    return result, state


def load_out(cfg, page=1, panel=1):
    key = f"page_{page:03d}_panel_{panel:03d}"
    return json.loads((Path(cfg.output.analysis_dir) / f"{key}.json").read_text("utf-8"))


def mock(cfg, response=None, raise_on_analyze=None):
    return MockVLMProvider(cfg, response=response, raise_on_analyze=raise_on_analyze)


# ---------------------------------------------------------------------- schema


def test_valid_mock_vlm_response(tmp_path):
    cfg = make_cfg(tmp_path)
    write_panel(cfg)
    result, state = run(cfg, provider=mock(cfg, response=VALID_RESPONSE))
    assert result["result"] == "ok"
    assert result["provider"] == "mock"
    doc = load_out(cfg)
    assert doc["page"] == 1 and doc["panel"] == 1
    assert doc["provider"] == "mock" and doc["model"]
    assert doc["analysis"]["characters"][0]["name"] == "unknown"
    assert doc["analysis"]["environment"] == "a battlefield"
    assert doc["analysis"]["confidence"] == 0.85
    assert state.pages.get("page_001_panel_001") == "vlm_completed"


def test_json_validation_fills_missing_and_coerces(tmp_path):
    cfg = make_cfg(tmp_path)
    write_panel(cfg)
    partial = {
        "characters": [{"name": "グリフィス", "description": "white armor"}],
        "environment": 42,
        "objects": "sword",
        "confidence": "0.5",
    }
    result, _ = run(cfg, provider=mock(cfg, response=partial))
    assert result["result"] == "ok"
    doc = load_out(cfg)
    analysis = doc["analysis"]
    assert analysis["characters"][0]["action"] == "unknown"
    assert analysis["environment"] == "42" or analysis["environment"] == "unknown"
    assert isinstance(analysis["objects"], list)
    assert analysis["confidence"] == 0.5
    assert analysis["composition"] == "unknown"
    assert analysis["actions"] == []


# ----------------------------------------------------------------- config errors


def test_missing_model_configuration(tmp_path):
    cfg = make_cfg(tmp_path, vlm={"provider": "local", "model": ""})
    write_panel(cfg)
    result, state = run(cfg)
    assert result["result"] == "error"
    assert "configure" in result["message"].lower()
    assert state.pages.get("page_001_panel_001") is None
    assert not (Path(cfg.output.analysis_dir) / "page_001_panel_001.json").exists()


def test_vlm_disabled(tmp_path):
    cfg = make_cfg(tmp_path, vlm={"enabled": False})
    write_panel(cfg)
    result, state = run(cfg)
    assert result["result"] == "error"
    assert "disabled" in result["message"]
    assert state.pages.get("page_001_panel_001") is None


def test_unknown_vlm_provider(tmp_path):
    cfg = make_cfg(tmp_path, vlm={"provider": "spacex"})
    write_panel(cfg)
    result, _ = run(cfg)
    assert result["result"] == "error"
    assert "unknown vlm.provider" in result["message"]


# --------------------------------------------------------------- input errors


def test_missing_panel(tmp_path):
    cfg = make_cfg(tmp_path)
    result, state = run(cfg)
    assert result["result"] == "error"
    assert "not found" in result["message"]
    assert state.pages.get("page_001_panel_001") is None


def test_invalid_image(tmp_path):
    cfg = make_cfg(tmp_path)
    write_panel(cfg, kind="corrupt")
    result, state = run(cfg, provider=mock(cfg))
    assert result["result"] == "error"
    assert "invalid panel image" in result["message"]
    assert state.pages.get("page_001_panel_001") is None


def test_invalid_page_panel_number(tmp_path):
    cfg = make_cfg(tmp_path)
    result, _ = run(cfg, page=0, panel=1)
    assert result["result"] == "error"
    assert "invalid" in result["message"]


# -------------------------------------------------------------- OCR context


def test_ocr_context_loaded_into_prompt(tmp_path):
    cfg = make_cfg(tmp_path)
    write_panel(cfg)
    write_ocr(cfg, text="「こんにちは」この勇者")
    provider = mock(cfg, response=VALID_RESPONSE)
    run(cfg, provider=provider)
    assert "「こんにちは」この勇者" in provider.last_prompt
    assert "potentially imperfect" in provider.last_prompt


def test_no_ocr_file_means_no_ocr_in_prompt(tmp_path):
    cfg = make_cfg(tmp_path)
    write_panel(cfg)
    provider = mock(cfg, response=VALID_RESPONSE)
    run(cfg, provider=provider)
    assert "<ocr_text>" not in (provider.last_prompt or "")


# ------------------------------------------------------------- checkpointing


def test_skip_when_already_analyzed(tmp_path):
    cfg = make_cfg(tmp_path)
    write_panel(cfg)
    result, state = run(cfg, provider=mock(cfg, response=VALID_RESPONSE))
    assert result["result"] == "ok"
    result2, state2 = run(cfg, provider=mock(cfg, response=VALID_RESPONSE))
    assert result2["result"] == "skipped"
    assert state2.pages.get("page_001_panel_001") == "vlm_completed"


def test_force_reanalyzes(tmp_path):
    cfg = make_cfg(tmp_path)
    write_panel(cfg)
    run(cfg, provider=mock(cfg, response=VALID_RESPONSE))
    result, _ = run(cfg, provider=mock(cfg, response=VALID_RESPONSE), force=True)
    assert result["result"] == "ok"


# ------------------------------------------------------------ error handling


def test_malformed_vlm_response_saved_raw(tmp_path):
    cfg = make_cfg(tmp_path)
    write_panel(cfg)
    result, state = run(cfg, provider=mock(cfg, response="```json\n{ this is not json"))
    assert result["result"] == "error"
    assert "no valid JSON" in result["message"]
    raw_file = Path(cfg.logging.log_dir) / "vlm" / "page_001_panel_001_raw.txt"
    assert raw_file.is_file()
    assert "this is not json" in raw_file.read_text("utf-8")
    assert state.pages.get("page_001_panel_001") is None
    assert not (Path(cfg.output.analysis_dir) / "page_001_panel_001.json").exists()


def test_json_extraction_repairs_fenced_and_wrapped(tmp_path):
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('prefix {"b": 2} suffix') == {"b": 2}
    assert extract_json('{"c": [1, 2]}') == {"c": [1, 2]}
    assert extract_json("not json at all") is None
    assert extract_json("") is None


def test_analysis_not_dict_saves_raw(tmp_path):
    cfg = make_cfg(tmp_path)
    write_panel(cfg)
    result, state = run(cfg, provider=mock(cfg, response="99"))
    assert result["result"] == "error"
    assert (Path(cfg.logging.log_dir) / "vlm" / "page_001_panel_001_raw.txt").is_file()
    assert state.pages.get("page_001_panel_001") is None


def test_timeout_result(tmp_path):
    cfg = make_cfg(tmp_path)
    write_panel(cfg)
    result, state = run(cfg, provider=mock(cfg, raise_on_analyze=VLMTimeout("waited too long")))
    assert result["result"] == "error"
    assert "timed out" in result["message"].lower()
    assert state.pages.get("page_001_panel_001") is None


def test_insufficient_memory_result(tmp_path):
    cfg = make_cfg(tmp_path)
    write_panel(cfg)
    result, state = run(cfg, provider=mock(cfg, raise_on_analyze=MemoryError("oom")))
    assert result["result"] == "error"
    assert "memory" in result["message"].lower()
    assert state.pages.get("page_001_panel_001") is None


def test_inference_failure_result(tmp_path):
    cfg = make_cfg(tmp_path)
    write_panel(cfg)
    result, state = run(cfg, provider=mock(cfg, raise_on_analyze=VLMFailure("boom")))
    assert result["result"] == "error"
    assert "failed" in result["message"].lower()
    assert state.pages.get("page_001_panel_001") is None


# ------------------------------------------------------------------- provider


def test_create_provider_mock_and_local():
    cfg = make_cfg(Path("/tmp/unused"), vlm={"provider": "mock"})
    assert isinstance(create_vlm_provider(cfg), MockVLMProvider)
    cfg2 = make_cfg(Path("/tmp/unused"), vlm={"provider": "local", "model": ""})
    with pytest.raises(VLMNotConfigured):
        create_vlm_provider(cfg2)
    cfg3 = make_cfg(Path("/tmp/unused"), vlm={"provider": "nope"})
    with pytest.raises(VLMProviderError):
        create_vlm_provider(cfg3)


def test_local_provider_rejects_missing_path(tmp_path):
    from pipeline.vlm_provider import LocalVLMProvider

    cfg = make_cfg(tmp_path, vlm={"provider": "local", "model": str(tmp_path / "missing-model")})
    provider = LocalVLMProvider(cfg)
    with pytest.raises(Exception):
        provider.load()
        provider.analyze_image("x", "y")


# -------------------------------------------------------------------- prompt


def test_prompt_builder_omits_and_adds_context():
    base = build_analysis_prompt()
    assert "invent" in base and "JSON" in base
    assert "<ocr_text>" not in base
    full = build_analysis_prompt(
        ocr_context="文字", previous_panel="prev summary", next_panel="next summary"
    )
    assert "<ocr_text>" in full and "文字" in full
    assert "prev summary" in full and "next summary" in full
    assert "potentially imperfect" in full


def test_sanitize_analysis_not_dict():
    assert sanitize_analysis(None) is None
    assert sanitize_analysis([1, 2]) is None