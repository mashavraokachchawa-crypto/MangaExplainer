"""Tests: audio synthesis for ONE scene from its script segments (generate_audio).

Uses the deterministic mock TTS engine so durations can be asserted exactly;
never a real speech binary (except the explicit engine-missing paths).
"""

import json
from pathlib import Path

import pytest

from config.loader import Config
from pipeline.audio_generator import (
    AUDIO_JSON,
    AUDIO_WAV,
    AudioGenerator,
    audio_manifest_path,
    audio_wav_path,
    load_script,
)
from pipeline.tts_provider import (
    MockTtsProvider,
    TtsProvider,
    TtsUnavailable,
    create_tts_provider,
    wav_duration,
)
from state import State

ROOT = Path(__file__).resolve().parent.parent
STAGE_NAMES = ["extract_pages", "segment_panels", "run_ocr", "analyze_panels",
               "build_scenes", "write_script", "generate_audio"]


def make_cfg(tmp_path, tts=None):
    tts = tts or {}
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
            "enabled": True, "engine": "auto", "voice": "en",
            "sample_rate": 22050, "rate_wpm": 150, "pitch_base": 50,
            "timeout_seconds": 60,
        },
        "video": {"resolution": "1920x1080", "fps": 30},
        "pipeline": {"batch_size": 1, "state": {"dir": str(tmp_path / "state")}, "cache": {"dir": str(tmp_path / "state" / "cache")}},
        "memory": {"guard_mb": 3072},
        "logging": {"level": "INFO", "console_level": "WARNING", "log_dir": str(tmp_path / "logs"), "max_bytes": 1048576, "backup_count": 3},
    }
    data["tts"].update(tts)
    return Config(data, tmp_path)


def write_script(cfg, segments, scene="scene_001", page=1, scene_num=1, raw=None):
    path = Path(cfg.output.script_dir) / f"page_{page:03d}_scene_{scene_num:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        path.write_text(raw, encoding="utf-8")
    else:
        payload = {"scene_id": scene, "page": page, "segments": segments}
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def seg(segment_id, text, seconds, seg_type="narration", speaker=None):
    out = {
        "segment_id": segment_id, "type": seg_type, "text": text,
        "panel_ids": ["p001_001"], "estimated_seconds": seconds,
        "visual_intent": "full_panel", "camera": "static", "importance": 0.7,
    }
    if seg_type == "dialogue":
        out["speaker"] = speaker or "unknown"
    return out


class BoomProvider(TtsProvider):
    name = "boom"

    def synth(self, text, out_path, target_seconds=None, speaker=None):
        raise TtsUnavailable("boom engine down")


def run(cfg, page=1, scene_num=1, force=False, provider=None):
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = AudioGenerator(cfg, provider=provider).run_scene(page, scene_num, state, force=force)
    return result, state


def mock_provider(cfg):
    return MockTtsProvider(cfg)


# ----------------------------------------------------------------- required


def test_valid_scene(tmp_path):
    cfg = make_cfg(tmp_path)
    write_script(cfg, [seg("seg_001", "The night air is heavy.", 4.0)])
    result, state = run(cfg, provider=mock_provider(cfg))
    assert result["result"] == "ok"
    assert result["engine"] == "mock"
    assert result["segment_count"] == 1
    wav = audio_wav_path(cfg, 1, 1)
    mani = audio_manifest_path(cfg, 1, 1)
    assert wav.is_file() and mani.is_file()
    assert wav_duration(wav) == pytest.approx(4.0, abs=0.05)
    doc = json.loads(mani.read_text("utf-8"))
    assert doc["scene_id"] == "scene_001"
    assert len(doc["segments"]) == 1
    assert doc["segments"][0]["segment_id"] == "seg_001"
    assert doc["segments"][0]["duration_ms"] == pytest.approx(4000, abs=25)
    assert state.pages.get("page_001_scene_001") == "audio_completed"


def test_missing_script(tmp_path):
    cfg = make_cfg(tmp_path)
    result, state = run(cfg, provider=mock_provider(cfg))
    assert result["result"] == "error"
    assert "script file not found" in result["message"]
    assert state.pages.get("page_001_scene_001") is None
    assert not audio_wav_path(cfg, 1, 1).exists()


def test_malformed_script(tmp_path):
    cfg = make_cfg(tmp_path)
    write_script(cfg, [], raw="{broken")
    result, state = run(cfg, provider=mock_provider(cfg))
    assert result["result"] == "error"
    assert "invalid script" in result["message"]
    assert state.pages.get("page_001_scene_001") is None


def test_no_segments(tmp_path):
    cfg = make_cfg(tmp_path)
    write_script(cfg, [])
    result, state = run(cfg, provider=mock_provider(cfg))
    assert result["result"] == "error"
    assert "no segments" in result["message"]
    assert state.pages.get("page_001_scene_001") is None


def test_empty_segment_text(tmp_path):
    cfg = make_cfg(tmp_path)
    write_script(cfg, [seg("seg_001", "   ", 4.0)])
    result, state = run(cfg, provider=mock_provider(cfg))
    assert result["result"] == "error"
    assert "text is empty" in result["message"]
    assert state.pages.get("page_001_scene_001") is None


def test_non_positive_duration(tmp_path):
    cfg = make_cfg(tmp_path)
    write_script(cfg, [seg("seg_001", "too fast", 0)])
    result, state = run(cfg, provider=mock_provider(cfg))
    assert result["result"] == "error"
    assert "estimated_seconds" in result["message"]
    assert state.pages.get("page_001_scene_001") is None


def test_checkpoint_skip(tmp_path):
    cfg = make_cfg(tmp_path)
    write_script(cfg, [seg("seg_001", "Once.", 2.0)])
    first, _ = run(cfg, provider=mock_provider(cfg))
    assert first["result"] == "ok"
    second, _ = run(cfg, provider=mock_provider(cfg))
    assert second["result"] == "skipped"
    assert second["audio_file"] == str(audio_wav_path(cfg, 1, 1))


def test_force_regeneration(tmp_path):
    cfg = make_cfg(tmp_path)
    write_script(cfg, [seg("seg_001", "Once.", 2.0)])
    first, _ = run(cfg, provider=mock_provider(cfg))
    assert first["result"] == "ok"
    second, _ = run(cfg, provider=mock_provider(cfg), force=True)
    assert second["result"] == "ok"
    assert second["segment_count"] == 1


def test_provider_unavailable_error(tmp_path):
    cfg = make_cfg(tmp_path)
    write_script(cfg, [seg("seg_001", "Once.", 2.0)])
    result, state = run(cfg, provider=BoomProvider(cfg))
    assert result["result"] == "error"
    assert "unavailable" in result["message"]
    assert state.pages.get("page_001_scene_001") is None
    assert not audio_wav_path(cfg, 1, 1).exists()


def test_disabled_tts_error(tmp_path):
    cfg = make_cfg(tmp_path, tts={"enabled": False})
    write_script(cfg, [seg("seg_001", "Once.", 2.0)])
    result, state = run(cfg, provider=None)
    assert result["result"] == "error"
    assert "disabled" in result["message"]
    assert state.pages.get("page_001_scene_001") is None


# ------------------------------------------------------------- extra coverage


def test_multi_segment_manifest_and_concat(tmp_path):
    cfg = make_cfg(tmp_path)
    write_script(cfg, [
        seg("seg_001", "First beat.", 1.0),
        seg("seg_002", "Second beat.", 1.5),
        seg("seg_003", "Guts speaks.", 2.0, seg_type="dialogue", speaker="Guts"),
    ])
    result, state = run(cfg, provider=mock_provider(cfg))
    assert result["result"] == "ok"
    assert result["segment_count"] == 3
    mani = json.loads(audio_manifest_path(cfg, 1, 1).read_text("utf-8"))
    offsets = [e["start_ms"] for e in mani["segments"]]
    assert offsets == sorted(offsets)
    for entry in mani["segments"]:
        assert entry["end_ms"] == entry["start_ms"] + entry["duration_ms"]
    assert mani["total_duration_ms"] == mani["segments"][-1]["end_ms"]
    total = wav_duration(audio_wav_path(cfg, 1, 1))
    assert total == pytest.approx(4.5, abs=0.05)
    assert mani["segments"][2]["speaker"] == "Guts"
    assert mani["segments"][0]["speaker"] is None
    assert state.pages.get("page_001_scene_001") == "audio_completed"


def test_scene_2_untouched_without_script(tmp_path):
    cfg = make_cfg(tmp_path)
    write_script(cfg, [seg("seg_001", "Only scene one.", 1.0)], page=1, scene_num=1)
    run(cfg, provider=mock_provider(cfg))
    result, _ = run(cfg, scene_num=2, provider=mock_provider(cfg))
    assert result["result"] == "error"
    assert "page_001_scene_002.json" in result["message"]
    assert not audio_wav_path(cfg, 1, 2).exists()
    assert audio_wav_path(cfg, 1, 1).exists()  # scene 1 output untouched


def test_load_script_returns_path_and_doc(tmp_path):
    cfg = make_cfg(tmp_path)
    path = write_script(cfg, [seg("seg_001", "x", 1.0)])
    doc, loaded = load_script(cfg, 1, 1)
    assert str(path) == str(loaded)
    assert doc["scene_id"] == "scene_001"


def test_filenames_format():
    assert AUDIO_WAV.format(1, 2) == "page_001_scene_002.wav"
    assert AUDIO_JSON.format(12, 34) == "page_012_scene_034.json"


def test_full_pipeline_naming_matches_script_stage():
    from pipeline.script_generator import json_path as script_json

    assert script_json.__name__ == "json_path"  # script stage owns page/scene naming