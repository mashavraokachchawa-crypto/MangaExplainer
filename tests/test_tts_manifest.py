"""Tests: flat narration-manifest TTS generation (Tasks 14 & 15).

Uses the deterministic MockTtsProvider only - never the real Pocket TTS
library. Verifies the audio/segment_NNN.wav naming, exact-text preservation,
manifest contents, and cumulative start/end timing.
"""

import json
from pathlib import Path

import pytest

from config.loader import Config
from pipeline.pocket_tts import MockTtsProvider, wav_duration
from pipeline.tts_manifest import (
    NarrationManifestRunner,
    load_narration_segments,
    manifest_path,
    segment_wav_path,
)
from state import State

ROOT = Path(__file__).resolve().parent.parent
STAGE_NAMES = ["extract_pages", "segment_panels", "run_ocr", "analyze_panels",
               "build_scenes", "write_script", "generate_audio"]


def make_cfg(tmp_path, reference=None, provider="mock"):
    tts = {"enabled": True, "engine": "auto", "provider": provider, "voice": "en",
           "sample_rate": 24000, "format": "wav", "rate_wpm": 150,
           "pitch_base": 50, "timeout_seconds": 60}
    if reference is not None:
        tts["reference_audio"] = str(reference)
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
        "tts": tts,
        "video": {"resolution": "1920x1080", "fps": 30},
        "pipeline": {"batch_size": 1, "state": {"dir": str(tmp_path / "state")}, "cache": {"dir": str(tmp_path / "state" / "cache")}},
        "memory": {"guard_mb": 3072},
        "logging": {"level": "INFO", "console_level": "WARNING", "log_dir": str(tmp_path / "logs"), "max_bytes": 1048576, "backup_count": 3},
    }
    return Config(data, tmp_path)


def narration_script(tmp_path, segments):
    path = tmp_path / "script" / "page_001_scene_001.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scene_id": "scene_001", "page": 1,
                                "segments": segments}), encoding="utf-8")
    return path


def seg(segment_id, text, seconds=1.0, seg_type="narration"):
    return {
        "segment_id": segment_id, "type": seg_type, "text": text,
        "panel_ids": ["p001_001"], "estimated_seconds": seconds,
        "visual_intent": "full_panel", "camera": "static", "importance": 0.7,
    }


def sample_segments():
    return [
        seg("seg_001", "The night air is heavy.", 1.0),
        seg("seg_002", "He turns the corner.", 1.5),
        seg("seg_003", "Someone is waiting.", 1.2),
    ]


def runner(cfg):
    return NarrationManifestRunner(cfg, provider=MockTtsProvider(cfg))


# --------------------------------------------------------------- tests


def test_generates_flat_segment_wavs(tmp_path):
    cfg = make_cfg(tmp_path)
    out = Path(cfg.output.audio_dir)
    manifest = runner(cfg).generate(sample_segments(), out)

    names = sorted(p.name for p in out.glob("segment_*.wav"))
    assert names == ["segment_001.wav", "segment_002.wav", "segment_003.wav"]

    for i, fm in enumerate(manifest):
        idx = i + 1
        assert fm["audio_path"] == str(segment_wav_path(out, i))
        assert out / f"segment_{idx:03d}.wav" == Path(fm["audio_path"])
        assert Path(fm["audio_path"]).is_file()
        assert fm["segment_id"] == f"seg_{idx:03d}"


def test_manifest_contents(tmp_path):
    cfg = make_cfg(tmp_path)
    out = Path(cfg.output.audio_dir)
    segs = sample_segments()
    manifest = runner(cfg).generate(segs, out)

    assert len(manifest) == 3
    for i, (fm, s) in enumerate(zip(manifest, segs)):
        assert fm["segment_id"] == s["segment_id"]
        assert fm["text"] == s["text"]  # exact narration text preserved
        assert fm["audio_path"] == str(segment_wav_path(out, i))
        assert isinstance(fm["duration"], float) and fm["duration"] > 0


def test_manifest_written_to_disk(tmp_path):
    cfg = make_cfg(tmp_path)
    out = Path(cfg.output.audio_dir)
    runner(cfg).generate(sample_segments(), out)

    doc = json.loads(manifest_path(out).read_text("utf-8"))
    assert isinstance(doc, list) and len(doc) == 3
    keys = set(doc[0].keys())
    assert keys == {"segment_id", "text", "audio_path", "duration"}


def test_sequential_manifest_start_and_end_times(tmp_path):
    cfg = make_cfg(tmp_path)
    out = Path(cfg.output.audio_dir)
    manifest = runner(cfg).run(sample_segments(), out)

    # durations come directly from the WAVs (mock honors estimated_seconds)
    assert manifest[0]["start_time"] == 0.0
    assert manifest[1]["start_time"] == pytest.approx(manifest[0]["end_time"], abs=0.05)
    assert manifest[2]["start_time"] == pytest.approx(manifest[1]["end_time"], abs=0.05)
    for entry in manifest:
        assert entry["end_time"] == pytest.approx(
            entry["start_time"] + entry["duration"], abs=0.05
        )


def test_timing_only_does_not_regenerate(tmp_path):
    cfg = make_cfg(tmp_path)
    out = Path(cfg.output.audio_dir)
    manifest = runner(cfg).generate(sample_segments(), out)

    # sabotage one WAV's mtime + record it; timing-only must not rewrite audio
    target = out / "segment_002.wav"
    before = target.stat().st_mtime_ns

    import time
    time.sleep(0.02)
    manifest2 = runner(cfg).finalize_timing(out)

    assert manifest2[1]["segment_id"] == "seg_002"
    assert manifest2[1]["start_time"] == pytest.approx(manifest[0]["duration"], abs=0.05)
    assert target.stat().st_mtime_ns == before  # untouched -> not regenerated


def test_finalize_timing_preserves_order(tmp_path):
    cfg = make_cfg(tmp_path)
    out = Path(cfg.output.audio_dir)
    segs = sample_segments()
    runner(cfg).generate(segs, out)
    updated = runner(cfg).finalize_timing(out)
    assert [e["segment_id"] for e in updated] == ["seg_001", "seg_002", "seg_003"]


def test_generate_skips_existing_without_force(tmp_path):
    cfg = make_cfg(tmp_path)
    out = Path(cfg.output.audio_dir)
    m1 = runner(cfg).generate(sample_segments(), out)

    # Re-run without force: existing WAVs reused, durations preserved.
    m2 = runner(cfg).generate(sample_segments(), out, force=False)
    assert m2 == m1
    assert (out / "segment_001.wav").exists()


def test_generate_force_regenerates(tmp_path):
    cfg = make_cfg(tmp_path)
    out = Path(cfg.output.audio_dir)
    runner(cfg).generate(sample_segments(), out)
    m2 = runner(cfg).generate(sample_segments(), out, force=True)
    assert m2[0]["duration"] > 0


def test_empty_text_raises(tmp_path):
    cfg = make_cfg(tmp_path)
    out = Path(cfg.output.audio_dir)
    with pytest.raises(ValueError):
        runner(cfg).generate([seg("seg_X", "   ")], out)


def test_missing_segment_id_raises(tmp_path):
    cfg = make_cfg(tmp_path)
    out = Path(cfg.output.audio_dir)
    with pytest.raises(ValueError):
        runner(cfg).generate([{"text": "no id"}], out)


def test_load_narration_segments(tmp_path):
    script = narration_script(tmp_path, sample_segments())
    segments = load_narration_segments(script)
    assert [s["segment_id"] for s in segments] == ["seg_001", "seg_002", "seg_003"]


def test_load_narration_segments_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_narration_segments(tmp_path / "nope.json")
