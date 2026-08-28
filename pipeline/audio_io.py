"""Shared, lightweight WAV helpers for the audio stages (Tasks 19-21).

Pure numpy + stdlib wave so the mixer runs on a low-RAM CPU box without
heavy audio frameworks. All times are in seconds if a sample_rate is given.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

INT16_MAX = 32767.0


def read_wav(path):
    """Read a mono/stereo 16-bit PCM WAV -> (sample_rate, float32 mono array).

    Audio is down-mixed to mono and normalised to [-1, 1]. Raises ValueError
    on unsupported formats; missing files raise FileNotFoundError.
    """
    path = Path(path)
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    if sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / INT16_MAX
    else:
        raise ValueError(f"unsupported sample width {sampwidth} bytes in {path}")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return sr, data


def write_wav(path, sr, data, max_amplitude=0.98):
    """Write float32 mono data in [-1,1] to a 16-bit PCM WAV, clipped first."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(data, -max_amplitude, max_amplitude)
    pcm = (clipped * INT16_MAX).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())


def resample(data, src_sr, dst_sr):
    """Cheap linear-interpolation resample; identity when rates are equal."""
    if src_sr == dst_sr or src_sr <= 0 or dst_sr <= 0:
        return data
    n_out = int(round(len(data) * dst_sr / src_sr))
    if n_out == 0:
        return data[:0]
    x = np.linspace(0.0, len(data) - 1, n_out) if len(data) > 1 else \
        np.zeros(n_out)
    return np.interp(x, np.arange(len(data)), data).astype(np.float32)


def silence(samples, sr):
    return np.zeros(samples, dtype=np.float32)


def seconds_to_samples(t_sec, sr):
    return max(0, int(round(t_sec * sr)))


def apply_fades(data, sr, fade_in=0.0, fade_out=0.0):
    """Linear fade in/out in-place on a copy. Fades are clamped to the length."""
    out = data.copy()
    n = len(out)
    if fade_in > 0 and n:
        nf = min(seconds_to_samples(fade_in, sr), n)
        if nf > 1:
            out[:nf] *= np.linspace(0.0, 1.0, nf, dtype=np.float32)
    if fade_out > 0 and n:
        nf = min(seconds_to_samples(fade_out, sr), n)
        if nf > 1:
            out[-nf:] *= np.linspace(1.0, 0.0, nf, dtype=np.float32)
    return out


def peak_db(data, eps=1e-9):
    return 20.0 * np.log10(max(float(np.max(np.abs(data))), eps))
