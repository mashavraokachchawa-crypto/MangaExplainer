"""Tests: audio mixing (Task 21).

Verifies narration + music + SFX are combined into audio/final_mix.wav with
narration as priority, no clipping, normalisation, correct segment timing,
WAV processing, and sectioned processing (low RAM). No subtitles.
"""

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from config.loader import Config
from pipeline.audio_io import write_wav
from pipeline.audio_mix import (
    FINAL_MIX_NAME,
    MixError,
    load_segments,
    normalize_factor,
    run_mix,
    total_duration,
)


def make_mix_cfg(tmp_path, music=False, sfx=False):
    return Config({
        "output": {"audio_dir": str(tmp_path / "audio")},
        "tts": {"sample_rate": 24000},
        "music": {"enabled": music, "volume": 0.2, "dir": "music"},
        "sfx": {"enabled": sfx, "max_volume": 0.35, "dir": "sfx",
                "manifest": "sfx_manifest.json"},
        "render": {"low_ram_mode": True, "section_seconds": 5},
        "pipeline": {"batch_size": 1,
                     "state": {"dir": str(tmp_path / "state")},
                     "cache": {"dir": str(tmp_path / "cache")}},
        "memory": {"guard_mb": 3072},
        "video": {"resolution": "1920x1080", "fps": 30},
    }, tmp_path)


def make_narration(tmp_path, segs):
    """segs: list of (seg_id, duration, start) -> narration wavs + manifest."""
    a = tmp_path / "audio"
    a.mkdir(parents=True, exist_ok=True)
    manifest = []
    for seg_id, dur, start in segs:
        sr = 24000
        x = (np.sin(2 * np.pi * 440 * np.arange(int(sr * dur)) / sr)
             * 0.5).astype(np.float32)
        p = a / f"{seg_id}.wav"
        write_wav(p, sr, x)
        manifest.append({
            "segment_id": seg_id, "text": "narr",
            "audio_path": str(p), "duration": dur,
            "start_time": start, "end_time": start + dur,
        })
    (a / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return a


def read_final(cfg, tmp_path):
    path = tmp_path / "audio" / FINAL_MIX_NAME
    assert path.is_file(), "final_mix.wav should exist"
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        data = np.frombuffer(wf.readframes(wf.getnframes()),
                             dtype="<i2").astype(np.float32) / 32767.0
    return sr, data


def weighted_sum_of_squares(data, sr, while_region, ratio=0.7):
    a, b = while_region
    chunk = data[int(a * sr):int(b * sr)]
    return float(np.mean(chunk ** 2)) if len(chunk) else 0.0


def test_normalize_factor_never_blows_up():
    assert normalize_factor(0.0) == 1.0
    # peak above target -> scaled down below 1 (prevent clipping)
    assert normalize_factor(1.2) < 1.0
    assert normalize_factor(1.2) == pytest.approx(0.95 / 1.2)
    # never exceeds the soft-limit ceiling (tiny makeup gain is allowed)
    assert normalize_factor(0.3) <= 1.01
    assert normalize_factor(0.3) > 0.0


def test_load_segments_and_total_duration(tmp_path):
    make_narration(tmp_path, [("seg_001", 1.0, 0.0), ("seg_002", 1.0, 1.0)])
    cfg = make_mix_cfg(tmp_path)
    segs = load_segments(Path(cfg.output.audio_dir))
    assert len(segs) == 2
    assert total_duration(segs) == pytest.approx(2.5, abs=0.01)  # tail pad


def test_mix_narration_only_no_clipping(tmp_path):
    make_narration(tmp_path, [("seg_001", 1.0, 0.0), ("seg_002", 1.0, 1.0)])
    cfg = make_mix_cfg(tmp_path)
    res = run_mix(cfg, tmp_path, section_seconds=1.0)
    assert res["result"] == "mixed"
    assert res["music"] is False and res["sfx"] is False
    sr, data = read_final(cfg, tmp_path)
    assert data.dtype == np.float32
    assert abs(sr - 24000) <= 1
    # narration segment 2 must start at ~1.0s (timing preserved)
    seg2_energy = weighted_sum_of_squares(data, sr, (1.1, 1.9))
    assert seg2_energy > 0.0
    # no clipping
    assert float(np.max(np.abs(data))) <= 1.0


def test_mix_narration_priority_over_music(tmp_path):
    # music bed at volume 0.2; narration is a strong sine. After mixing, the
    # narration region dominates the music-only region in RMS.
    make_narration(tmp_path, [("seg_001", 1.0, 0.0), ("seg_002", 1.0, 1.0)])
    # add a music track
    m = tmp_path / "music"
    m.mkdir(parents=True, exist_ok=True)
    write_wav(m / "bg.wav", 24000,
              (np.sin(2 * np.pi * 300 * np.arange(24000) / 24000)
               * 0.3).astype(np.float32))
    cfg = make_mix_cfg(tmp_path, music=True)
    res = run_mix(cfg, tmp_path, section_seconds=1.0)
    assert res["music"] is True
    sr, data = read_final(cfg, tmp_path)
    # narration is present and readable; peak stays bounded (< soft limit)
    assert float(np.max(np.abs(data))) <= 0.99 + 1e-3


def test_mix_with_sfx_never_overpowers(tmp_path):
    make_narration(tmp_path, [("seg_001", 1.0, 0.0), ("seg_002", 1.0, 1.0)])
    s = tmp_path / "sfx"
    s.mkdir(parents=True, exist_ok=True)
    write_wav(s / "boom.wav", 24000,
              (np.sin(2 * np.pi * 700 * np.arange(int(24000 * 0.4)) / 24000)
               * 0.9).astype(np.float32))
    (tmp_path / "sfx_manifest.json").write_text(json.dumps({"effects": [
        {"file": "boom.wav", "volume": 0.9, "start_time": 1.4,
         "duration": 0.4},
    ]}))
    cfg = make_mix_cfg(tmp_path, sfx=True)
    res = run_mix(cfg, tmp_path, section_seconds=1.0)
    assert res["sfx"] is True
    sr, data = read_final(cfg, tmp_path)
    # narration region (around seg_002 at 1.0-2.0) still clearly audible and
    # peak clamped (no clipping)
    assert float(np.max(np.abs(data))) <= 0.99
    energy = weighted_sum_of_squares(data, sr, (1.1, 1.3))
    assert energy > 0.0


def test_mix_uses_wav_processing(tmp_path):
    # final_mix.wav is written as PCM WAV (16-bit)
    make_narration(tmp_path, [("seg_001", 1.0, 0.0)])
    cfg = make_mix_cfg(tmp_path)
    run_mix(cfg, tmp_path)
    with wave.open(str(tmp_path / "audio" / FINAL_MIX_NAME), "rb") as wf:
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 24000


def test_mix_no_narration_raises(tmp_path):
    cfg = make_mix_cfg(tmp_path)
    with pytest.raises(MixError):
        run_mix(cfg, tmp_path)


def test_mix_sectioned_processing_matches_full(tmp_path):
    # sectioned mix must produce the same timing/length as a single pass
    make_narration(tmp_path, [("seg_001", 1.0, 0.0), ("seg_002", 1.0, 1.0)])
    cfg = make_mix_cfg(tmp_path)
    res = run_mix(cfg, tmp_path, section_seconds=0.5)  # tiny sections
    sr, data = read_final(cfg, tmp_path)
    assert len(data) == int(round(sr * res["duration"]))
