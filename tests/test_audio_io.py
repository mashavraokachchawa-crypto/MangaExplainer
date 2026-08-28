"""Tests: shared low-level WAV helpers (audio_io).

Round-trips and utility behaviour used by the music, sfx and mixing stages.
"""

import numpy as np
import pytest

from pipeline.audio_io import (
    apply_fades,
    peak_db,
    read_wav,
    resample,
    seconds_to_samples,
    write_wav,
)


def test_wav_roundtrip(tmp_path):
    sr = 24000
    x = (np.sin(2 * np.pi * 440 * np.arange(sr) / sr) * 0.5).astype(np.float32)
    p = tmp_path / "t.wav"
    write_wav(p, sr, x)
    got_sr, got = read_wav(p)
    assert got_sr == sr
    assert len(got) == sr
    # amplitude preserved within ~1 quantization step
    assert np.abs(np.max(np.abs(got)) - 0.5) < 0.01


def test_write_wav_clips(tmp_path):
    p = tmp_path / "c.wav"
    write_wav(p, 8000, np.array([10.0, -10.0, 0.1], dtype=np.float32))
    _, got = read_wav(p)
    assert np.max(got) <= 0.98 + 1e-6


def test_read_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_wav(tmp_path / "nope.wav")


def test_resample_identity_and_length():
    x = np.sin(2 * np.pi * 100 * np.arange(1000) / 1000).astype(np.float32)
    assert resample(x, 24000, 24000) is x  # identity fast-path
    out = resample(x, 1000, 2000)
    assert len(out) == 2000


def test_seconds_to_samples():
    assert seconds_to_samples(1.0, 24000) == 24000
    assert seconds_to_samples(0.0, 24000) == 0


def test_apply_fades_boundaries():
    x = np.ones(24000, dtype=np.float32)
    y = apply_fades(x, 24000, fade_in=0.5, fade_out=0.5)
    assert y[0] == pytest.approx(0.0, abs=0.02)
    assert y[-1] == pytest.approx(0.0, abs=0.02)
    assert abs(y[12000] - 1.0) < 0.02  # middle untouched (fades don't overlap)


def test_apply_fades_long_fades_overlap_ok():
    # fades longer than the buffer still decay the edges and never clip
    x = np.ones(24000, dtype=np.float32)
    y = apply_fades(x, 24000, fade_in=1.0, fade_out=1.0)
    assert y[0] == pytest.approx(0.0, abs=0.02)
    assert y[-1] == pytest.approx(0.0, abs=0.02)
    assert float(np.max(y)) <= 1.0


def test_peak_db():
    assert peak_db(np.zeros(10)) < -100.0
    assert peak_db(np.array([0.5, -0.5], dtype=np.float32)) == pytest.approx(
        -6.0206, abs=0.1)
