"""Tests: panel-specific sound effects (Task 20).

Covers configurable volume, start time, duration, local files, panel/segment
links, and the hard rule that an SFX never overpowers narration.
"""

import json
import numpy as np
import pytest

from config.loader import Config
from pipeline.audio_io import write_wav
from pipeline.sfx import (
    SfxUnavailable,
    aggregate_sfx,
    build_sfx_segment,
    event_absolute_start,
    limit_against_narration,
    load_events,
    sfx_config,
)


def make_cfg(tmp_path, **sfx):
    cfg = sfx_config(Config({
        "output": {"audio_dir": str(tmp_path / "audio")},
        "tts": {"sample_rate": 24000},
        "sfx": dict({"enabled": True, "max_volume": 0.35,
                     "dir": "sfx", "manifest": "sfx_manifest.json"}, **sfx),
        "pipeline": {"batch_size": 1,
                     "state": {"dir": str(tmp_path / "state")},
                     "cache": {"dir": str(tmp_path / "cache")}},
        "memory": {"guard_mb": 3072},
    }, tmp_path))


def write_sfx(tmp_path, name, sr=24000, seconds=0.5, freq=500.0):
    d = tmp_path / "sfx"
    d.mkdir(parents=True, exist_ok=True)
    path = d / name
    x = (np.sin(2 * np.pi * freq * np.arange(int(sr * seconds)) / sr)
         * 0.9).astype(np.float32)
    write_wav(path, sr, x)
    return path


def write_manifest(tmp_path, events):
    (tmp_path / "sfx_manifest.json").write_text(
        json.dumps({"effects": events}), encoding="utf-8")


def test_sfx_disabled():
    from pathlib import Path
    m = sfx_config(Config({"sfx": {"enabled": False}}, Path(".")))
    assert m is None


def test_event_start_time_priority():
    # explicit start_time wins; otherwise tied to segment start + offset
    seg_start = 5.0
    assert event_absolute_start(
        {"start_time": 12.0, "offset": 1.0}, seg_start) == 12.0
    assert event_absolute_start(
        {"offset": 1.0}, seg_start) == 6.0
    assert event_absolute_start({}, seg_start) == 5.0


def test_limit_never_overpowers_narration():
    # with a loud narration peak, sfx capped at half of it
    assert limit_against_narration(0.9, 0.8, 0.35) <= 0.4
    assert limit_against_narration(0.9, 0.8, 0.35) >= 0.0
    # even a modest narration peak never lets sfx exceed the cap
    assert limit_against_narration(0.9, 0.05, 0.35) <= 0.35


def test_build_sfx_segment_duration_snip(tmp_path):
    write_sfx(tmp_path, "pop.wav", seconds=0.5)
    data = build_sfx_segment({"file": str(tmp_path / "sfx" / "pop.wav"),
                              "volume": 0.5, "duration": 0.2}, 24000)
    assert len(data) == int(round(0.2 * 24000))
    peak = float(np.max(np.abs(data)))
    assert 0.0 < peak <= 0.5 + 1e-3


def test_load_events_panel_specific(tmp_path):
    write_sfx(tmp_path, "pop.wav")
    write_manifest(tmp_path, [
        {"panel_id": "p001_001", "file": "pop.wav", "volume": 0.7,
         "start_time": 1.5, "duration": 0.3},
    ])
    cfgm = sfx_config(Config({}, tmp_path))  # not used
    events = load_events(tmp_path, {
        "dir": "sfx", "manifest": "sfx_manifest.json"})
    assert len(events) == 1
    ev = events[0]
    assert ev["panel_id"] == "p001_001"
    assert ev["start_time"] == 1.5
    assert ev["duration"] == 0.3
    assert ev["file"].endswith("pop.wav")


def test_load_events_missing_manifest_raises(tmp_path):
    from pathlib import Path
    with pytest.raises(SfxUnavailable):
        load_events(tmp_path, {"dir": "sfx",
                               "manifest": "sfx_manifest.json"})


def test_aggregate_sfx_places_at_time(tmp_path):
    write_sfx(tmp_path, "pop.wav", seconds=0.5)
    write_manifest(tmp_path, [
        {"file": "pop.wav", "volume": 0.5, "start_time": 1.0,
         "duration": 0.5},
    ])
    events = load_events(tmp_path, {"dir": "sfx",
                                    "manifest": "sfx_manifest.json"})
    placed = aggregate_sfx(tmp_path, {"max_volume": 0.35}, events,
                           {}, 24000, narration_peak=0.9)
    assert 1.0 * 24000 in placed
    arr = placed[1.0 * 24000]
    assert len(arr) == int(round(0.5 * 24000))
    # bounded so it stays under narration
    assert float(np.max(np.abs(arr))) <= 0.35 + 1e-3
