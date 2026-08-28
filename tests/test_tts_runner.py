"""Tests: Pocket TTS narration generation, ONE segment at a time (Task 13).

Uses the deterministic MockTtsProvider only - never the real Pocket TTS
library (which needs torch / the model / internet). Reference voice is
validated/preprocessed offline.
"""

import json
import wave
from pathlib import Path

import pytest

from config.loader import Config
from pipeline.pocket_tts import (
    MockTtsProvider,
    PocketTtsNotConfigured,
    PocketTtsUnavailable,
    create_pocket_tts_provider,
    wav_duration,
)
from pipeline.tts_provider import TtsError
from pipeline.tts_runner import (
    TtsRunner,
    _seg_key,
    scene_audio_dir,
    segment_wav_path,
    segments_to_process,
    timing_json_path,
)
from pipeline.voice_reference import (
    ReferenceAudioMissing,
    ReferenceAudioValidationError,
    preprocess_reference,
    validate_reference,
)
from state import State

ROOT = Path(__file__).resolve().parent.parent
STAGE_NAMES = ["extract_pages", "segment_panels", "run_ocr", "analyze_panels",
               "build_scenes", "write_script", "generate_audio"]


def make_cfg(tmp_path, tts=None, reference=None):
    tts = tts or {}
    if reference is not None:
        tts = {**tts, "reference_audio": str(reference)}
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
    data["tts"].update(tts)
    return Config(data, tmp_path)


def write_script(cfg, segments, page=1, scene_num=1):
    path = Path(cfg.output.script_dir) / f"page_{page:03d}_scene_{scene_num:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"scene_id": f"scene_{scene_num:03d}", "page": page, "segments": segments}
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


def sample_segments():
    return [
        seg("seg_001", "The night air is heavy.", 1.0),
        seg("seg_002", "He turns the corner.", 1.5),
        seg("seg_003", "Someone is waiting.", 1.2),
    ]


def make_wav_reference(path, seconds=2.0, rate=24000):
    import array

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(array.array("h", [5000] * int(seconds * rate)).tobytes())
    return path


def fresh_state(cfg):
    return State(STAGE_NAMES, Path(cfg.pipeline.state.dir) / "state.json")


def run(cfg, page=1, scene_num=1, force=False, segment=None, provider=None,
        state=None):
    state = state or fresh_state(cfg)
    provider = provider or MockTtsProvider(cfg)
    runner = TtsRunner(cfg, provider=provider)
    result = runner.run_scene(page, scene_num, state, force=force, segment=segment)
    return result, state


# ------------------------------------------------------------------ tests


def test_generates_all_segment_wavs_and_timing_json(tmp_path):
    cfg = make_cfg(tmp_path)
    write_script(cfg, sample_segments())
    result, state = run(cfg)

    out = scene_audio_dir(cfg, 1, 1)
    names = sorted(p.name for p in out.iterdir())
    assert names == ["segment_000.wav", "segment_001.wav", "segment_002.wav",
                     "timing.json"]

    for i, s in enumerate(sample_segments()):
        wav = segment_wav_path(cfg, 1, 1, i)
        assert wav.is_file()
        assert wav_duration(wav) == pytest.approx(s["estimated_seconds"], abs=0.05)

    doc = json.loads(timing_json_path(cfg, 1, 1).read_text("utf-8"))
    assert doc["page"] == 1 and doc["scene"] == 1
    assert doc["sample_rate"] == 24000
    assert [x["segment_id"] for x in doc["segments"]] == ["seg_001", "seg_002", "seg_003"]
    assert result["segments_generated"] == 3 and result["segments_skipped"] == 0


def test_checkpoint_keys_use_expected_format(tmp_path):
    cfg = make_cfg(tmp_path)
    write_script(cfg, sample_segments())
    result, state = run(cfg)
    keys = set(state.as_dict()["pages"].keys())
    assert keys == {
        "page_001_scene_001_seg_001",
        "page_001_scene_001_seg_002",
        "page_001_scene_001_seg_003",
    }
    assert all(value == "tts_completed" for value in
               [state.as_dict()["pages"][k] for k in keys])


def test_rerun_skips_completed_segments(tmp_path):
    cfg = make_cfg(tmp_path)
    write_script(cfg, sample_segments())
    result1, state = run(cfg)
    assert result1["segments_generated"] == 3

    result2, state = run(cfg, state=state)
    assert result2["summary"] == "skipped"
    assert result2["segments_generated"] == 0
    assert result2["segments_skipped"] == 3


def test_force_regenerates_all(tmp_path):
    cfg = make_cfg(tmp_path)
    write_script(cfg, sample_segments())
    result1, state = run(cfg)
    assert result1["segments_generated"] == 3

    result2, state = run(cfg, state=state, force=True)
    assert result2["segments_generated"] == 3
    assert result2["segments_skipped"] == 0


def test_segment_option_regenerates_only_one(tmp_path):
    cfg = make_cfg(tmp_path)
    write_script(cfg, sample_segments())
    # Fresh state + --segment 2 => only the 2nd segment is generated.
    result1, state = run(cfg, segment=2)
    assert result1["segments_generated"] == 1
    names = sorted(p.name for p in scene_audio_dir(cfg, 1, 1).iterdir())
    assert names == ["segment_001.wav", "timing.json"]

    # --segment 2 --force over the same state => regenerates exactly that one.
    result2, state = run(cfg, state=state, segment=2, force=True)
    assert result2["segments_generated"] == 1
    assert result2["segments_skipped"] == 0
    timing = json.loads(timing_json_path(cfg, 1, 1).read_text("utf-8"))
    assert [x["wav"] for x in timing["segments"]] == ["segment_001.wav"]


def test_segment_out_of_range_raises(tmp_path):
    cfg = make_cfg(tmp_path)
    write_script(cfg, sample_segments())
    state = fresh_state(cfg)
    with pytest.raises(ValueError):
        TtsRunner(cfg, provider=MockTtsProvider(cfg)).run_scene(
            1, 1, state, segment=9)


def test_missing_script_raises(tmp_path):
    cfg = make_cfg(tmp_path)
    with pytest.raises(TtsError):
        run(cfg)


def test_no_segments_raises(tmp_path):
    cfg = make_cfg(tmp_path)
    write_script(cfg, [])
    with pytest.raises(RuntimeError):
        run(cfg)


def test_empty_segment_text_raises(tmp_path):
    cfg = make_cfg(tmp_path)
    write_script(cfg, [seg("seg_001", "   ", 1.0)])
    with pytest.raises(RuntimeError):
        run(cfg)


def test_reference_missing_reports_warning_but_generates(tmp_path):
    cfg = make_cfg(tmp_path)
    write_script(cfg, sample_segments())
    result, state = run(cfg)
    assert result["reference"]["missing"] is True
    assert "warning" in result
    assert result["segments_generated"] == 3


def test_reference_present_is_validated(tmp_path):
    ref = make_wav_reference(tmp_path / "voice_reference" / "ref.wav", seconds=2.0)
    cfg = make_cfg(tmp_path, reference=ref)
    write_script(cfg, sample_segments())
    result, state = run(cfg)
    assert result["reference"]["missing"] is False
    assert result["reference"]["format"] == "wav"
    assert result["reference"]["sample_rate"] == 24000
    assert result["reference"]["duration"] == pytest.approx(2.0, abs=0.05)
    assert "warning" not in result


def test_mock_provider_synth_writes_expected_duration(tmp_path):
    cfg = make_cfg(tmp_path)
    provider = MockTtsProvider(cfg)
    out = tmp_path / "out" / "s.wav"
    dur = provider.synth("Hello there", out, target_seconds=2.5)
    assert out.is_file()
    assert dur == pytest.approx(2.5, abs=0.05)
    assert wav_duration(out) == pytest.approx(2.5, abs=0.05)
    assert provider.sample_rate == 24000


def test_factory_raises_when_disabled(tmp_path):
    cfg = make_cfg(tmp_path, tts={"enabled": False})
    with pytest.raises(PocketTtsNotConfigured):
        create_pocket_tts_provider(cfg)


def test_factory_pocket_requested_resolves_real_or_raises(tmp_path):
    from pipeline.pocket_tts import PocketTtsProvider

    cfg = make_cfg(tmp_path, tts={"provider": "pocket_tts"})
    installed = PocketTtsProvider.available()
    if installed:
        # Real package present: the factory must hand back a real provider.
        provider = create_pocket_tts_provider(cfg)
        assert isinstance(provider, PocketTtsProvider)
    else:
        # Package absent: explicit request must raise, never silently mock.
        with pytest.raises(PocketTtsUnavailable):
            create_pocket_tts_provider(cfg)


def test_factory_auto_falls_back_to_mock_when_pocket_absent(tmp_path, monkeypatch):
    from pipeline.pocket_tts import PocketTtsProvider

    monkeypatch.setattr(PocketTtsProvider, "available", staticmethod(lambda: False))
    cfg = make_cfg(tmp_path, tts={"provider": "auto"})
    from pipeline.pocket_tts import MockTtsProvider

    provider = create_pocket_tts_provider(cfg)
    assert isinstance(provider, MockTtsProvider)


def test_validate_reference_missing_raises(tmp_path):
    with pytest.raises(ReferenceAudioMissing):
        validate_reference(tmp_path / "nope.wav")


def test_validate_reference_silent_detected(tmp_path):
    import array

    silent = tmp_path / "silent.wav"
    with wave.open(str(silent), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(array.array("h", [0] * 24000).tobytes())
    info = validate_reference(silent)
    assert info["silent"] is True


def test_preprocess_copies_conforming_wav(tmp_path):
    ref = make_wav_reference(tmp_path / "ref.wav")
    work = preprocess_reference(ref, tmp_path / "work")
    assert Path(work["path"]).is_file()
    assert work["preprocessed"] is False


def test_result_exposes_audio_dir_and_timing(tmp_path):
    cfg = make_cfg(tmp_path)
    write_script(cfg, sample_segments())
    result, _ = run(cfg)
    assert Path(result["audio_dir"]) == scene_audio_dir(cfg, 1, 1)
    assert Path(result["timing_json"]) == timing_json_path(cfg, 1, 1)
    assert Path(result["timing_json"]).is_file()


def test_provider_falls_back_when_clone_weights_unavailable(tmp_path, monkeypatch):
    """Reference present but cloning gated => fall back to builtin, never fake."""
    from pipeline.pocket_tts import (
        PocketTtsProvider,
        ReferenceAudioError,
    )

    ref = make_wav_reference(tmp_path / "voice_reference.mp3")
    cfg = make_cfg(tmp_path, reference=ref)
    prov = PocketTtsProvider(cfg)

    # Do not load the real model: stub the minimal internals.
    def fake_require_deps():
        prov._model = object()  # enough for _resolve_voice_state
        prov._voice_state_cache = _VoiceCacheStub(tmp_path)

    monkeypatch.setattr(prov, "_require_deps", fake_require_deps)
    # Simulate the license-gated clone failure.
    def boom(ref, key, model):
        raise ReferenceAudioError("cloning weights are license-gated")

    monkeypatch.setattr(prov, "_build_and_cache_voice", boom)
    monkeypatch.setattr(
        prov, "_resolve_builtin_voice",
        lambda voice, model: ("builtin-voice", voice),
    )

    state = prov._resolve_voice_state(ref)
    assert state[0] == "builtin-voice"
    assert prov._conditioning == "builtin"
    assert prov._conditioning_unavailable is not None
    assert "cloning weights" in prov._conditioning_unavailable


class _VoiceCacheStub:
    def __init__(self, base):
        tmp = base / "state" / "cache" / "voice"
        self.base = tmp
        tmp.mkdir(parents=True, exist_ok=True)

    def get(self, key):
        p = self.base / f"voice-{key}.safetensors"
        return str(p) if p.exists() else None

    def path_for(self, key):
        return self.base / f"voice-{key}.safetensors"
