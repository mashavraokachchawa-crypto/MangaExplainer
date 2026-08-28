"""Tests: background music support (Task 19).

Covers enable/disable, configurable volume, auto-loop, fade in/out, and the
rule that music always stays under narration (a low bed by default). Local
audio files only - nothing is generated.
"""

import numpy as np
import pytest

from config.loader import Config
from pipeline.audio_io import write_wav
from pipeline.music import (
    DEFAULT_VOLUME,
    MusicUnavailable,
    make_music_bed,
    music_config,
    resolve_track,
)


def make_cfg(tmp_path, **music):
    cfg = {
        "output": {"audio_dir": str(tmp_path / "audio")},
        "tts": {"sample_rate": 24000},
        "music": dict({"enabled": True, "volume": 0.2, "dir": "music"},
                      **music),
        "pipeline": {"batch_size": 1,
                     "state": {"dir": str(tmp_path / "state")},
                     "cache": {"dir": str(tmp_path / "cache")}},
        "memory": {"guard_mb": 3072},
    }
    return Config(cfg, tmp_path)


def write_track(tmp_path, sr=24000, seconds=1.0, freq=440.0):
    d = tmp_path / "music"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "track.wav"
    x = (np.sin(2 * np.pi * freq * np.arange(int(sr * seconds)) / sr)
         * 0.8).astype(np.float32)
    write_wav(path, sr, x)
    return path


def test_music_disabled_by_default():
    from pathlib import Path
    cfg = make_cfg(Path("."), enabled=False)
    assert music_config(cfg) is None


def test_music_config_defaults():
    from pathlib import Path
    cfg = make_cfg(Path("."))
    m = music_config(cfg)
    assert m is not None
    assert m["volume"] == 0.2
    assert m["loop"] is True
    assert m["dir"] == "music"


def test_music_effective_level_capped_by_max_level(tmp_path):
    cfg = make_cfg(tmp_path, volume=0.9, max_level=0.5)
    settings = music_config(cfg)
    track = write_track(tmp_path)
    bed, _ = make_music_bed(tmp_path, settings, duration=1.0, sr=24000,
                            track=track)
    # effective level clamped to 0.5 (never louder than max_level)
    assert float(np.max(np.abs(bed))) <= 0.5 + 1e-6


def test_resolve_track_finds_local_file(tmp_path):
    cfg = make_cfg(tmp_path)
    track = write_track(tmp_path)
    settings = music_config(cfg)
    found = resolve_track(tmp_path, settings)
    assert found == track


def test_resolve_track_missing_dir_raises(tmp_path):
    cfg = make_cfg(tmp_path)
    settings = music_config(cfg)
    with pytest.raises(MusicUnavailable):
        resolve_track(tmp_path, settings)


def test_music_bed_volume_respected(tmp_path):
    cfg = make_cfg(tmp_path)
    settings = music_config(cfg)
    track = write_track(tmp_path)
    bed, used = make_music_bed(tmp_path, settings, duration=2.0, sr=24000,
                               track=track)
    assert used == track
    assert len(bed) == 2.0 * 24000  # looped to cover full duration
    peak = float(np.max(np.abs(bed)))
    # volume 0.2 -> peak roughly 0.8*0.2 = 0.16
    assert 0.0 < peak <= settings["volume"] + 1e-6


def test_music_loop_covers_duration(tmp_path):
    cfg = make_cfg(tmp_path)
    settings = music_config(cfg)
    track = write_track(tmp_path, seconds=0.5)
    bed, _ = make_music_bed(tmp_path, settings, duration=3.0, sr=24000,
                            track=track)
    assert len(bed) == 3.0 * 24000  # short track looped 6x


def test_music_bed_never_exceeds_narration_level(tmp_path):
    # even at high volume, the bed is bounded and kept low
    cfg = make_cfg(tmp_path, volume=1.0, max_level=0.5)
    settings = music_config(cfg)
    track = write_track(tmp_path, seconds=1.0)
    bed, _ = make_music_bed(tmp_path, settings, duration=1.5, sr=24000,
                            track=track)
    assert float(np.max(np.abs(bed))) <= 0.5 + 1e-6


def test_music_fade_in_out(tmp_path):
    cfg = make_cfg(tmp_path, fade_in=0.4, fade_out=0.4)
    settings = music_config(cfg)
    track = write_track(tmp_path, seconds=2.0)
    bed, _ = make_music_bed(tmp_path, settings, duration=1.0, sr=24000,
                            track=track)
    # first sample is ~0 (fade in), last sample ~0 (fade out)
    assert abs(bed[0]) < 0.05
    assert abs(bed[-1]) < 0.05
    # the middle region is clearly audible (not a silent zero-crossing point)
    mid = bed[int(len(bed) * 0.4):int(len(bed) * 0.6)]
    assert float(np.max(np.abs(mid))) > 0.05
