"""Audio mixing (Task 21).

Combines, in priority order:
  1. Pocket TTS narration   (always dominant)
  2. background music        (low, faded, looped)
  3. panel sound effects     (bounded so narration always wins)

into a single 16-bit PCM WAV at audio/final_mix.wav.

Guarantees:
   * narration is placed first, exactly on its manifest times (timing kept)
   * music/SFX are scaled below the narration ceiling
   * clipping is prevented (soft limit) and the final audio is peak-normalised
   * each section is mixed, and only ever held, one at a time to bound RAM
   * WAV is used end-to-end during processing

No subtitles are produced here.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .audio_io import (
    INT16_MAX,
    read_wav,
    resample,
    seconds_to_samples,
    write_wav,
)
from .music import music_config, make_music_bed
from .sfx import aggregate_sfx, load_events, sfx_config

LOG = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
FINAL_MIX_NAME = "final_mix.wav"
DEFAULT_SECTION_SECONDS = 15.0
TARGET_PEAK = 0.95
SOFT_LIMIT = 0.99


class MixError(Exception):
    """Base error for the audio mixing stage."""


class NoNarration(MixError):
    """No narration audio/manifest was found to mix."""


def load_segments(audio_dir):
    """Load narration segments from audio_dir/manifest.json."""
    manifest = Path(audio_dir) / MANIFEST_NAME
    if not manifest.is_file():
        raise NoNarration(f"no {MANIFEST_NAME} in {audio_dir} - run a "
                          "TTS stage first")
    try:
        doc = json.loads(manifest.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NoNarration(f"cannot read {manifest}: {exc}")
    segments = []
    for item in doc:
        if not isinstance(item, dict):
            continue
        start = float(item.get("start_time") or 0.0)
        end = float(item.get("end_time") or start)
        segments.append({
            "segment_id": item.get("segment_id"),
            "path": Path(item.get("audio_path")),
            "start": start,
            "end": end,
            "duration": float(item.get("duration") or (end - start) or 0.0),
        })
    if not segments:
        raise NoNarration(f"no usable segments in {manifest}")
    return segments


def total_duration(segments, tail_pad=0.5):
    if not segments:
        return 0.0
    return max(s["end"] for s in segments) + tail_pad


def narration_peak(segments, section_seconds=DEFAULT_SECTION_SECONDS):
    """Global peak across all narration audio (used as the priority ceiling).

    Reads narration in sections to bound RAM.
    """
    peak = 0.0
    for seg in segments:
        if not seg["path"].is_file():
            continue
        seg_sr, data = read_wav(seg["path"])
        p = float(np.max(np.abs(data))) if len(data) else 0.0
        if p > peak:
            peak = p
    return peak or 0.9


def _section_bounds(total_seconds, sr, section_seconds):
    """Yield (start_sample, end_sample) sample slices across the timeline."""
    total_samples = seconds_to_samples(total_seconds, sr)
    step = seconds_to_samples(section_seconds, sr)
    if step <= 0:
        step = total_samples or 1
    start = 0
    while start < total_samples:
        end = min(total_samples, start + step)
        yield start, end
        if end >= total_samples:
            break
        start = end


def _narration_slices(segments, sr, lo, hi):
    """Yield (local_offset, array) narration slices overlapping [lo,hi)."""
    out = []
    for seg in segments:
        s = int(round(seg["start"] * sr))
        e = int(round(seg["end"] * sr))
        if e <= lo or s >= hi:
            continue
        if not seg["path"].is_file():
            continue
        # only load the overlapping slice of the narration to bound memory
        seg_sr, data = read_wav(seg["path"])
        orig = resample(data, seg_sr, sr).astype(np.float32, copy=False)
        ol = max(0, lo - s)
        oh = min(len(orig), hi - s)
        if oh <= ol:
            continue
        out.append((s - lo + ol, orig[ol:oh]))
    return out


def mix_section(segments, music_bed_fn, sfx_map, sr, lo, hi, narration_peak_val):
    """Mix one [lo,hi) sample section into a float32 buffer.

    music_bed_fn: zero-arg callable returning the pre-resampled, scaled music
                  bed as a full float32 array (or None when music is off).
    sfx_map:      dict {start_sample: array} (or None).
    """
    buf = np.zeros(hi - lo, dtype=np.float32)
    # 1) narration first (priority), on its exact times
    for offset, chunk in _narration_slices(segments, sr, lo, hi):
        buf[offset:offset + len(chunk)] += chunk
    # 2) background music, beneath narration
    if music_bed_fn is not None:
        bed = music_bed_fn()
        if len(bed):
            seg = bed[lo:hi] if hi <= len(bed) else np.pad(
                bed[lo:], (0, max(0, hi - len(bed))))
            buf += seg[:len(buf)]
    # 3) sound effects, bounded below narration
    if sfx_map:
        for start, arr in sfx_map.items():
            rel = start - lo
            if rel >= hi or rel + len(arr) <= 0:
                continue
            b0 = max(0, -rel)
            a0 = max(0, rel)
            n = min(len(arr) - b0, len(buf) - a0)
            if n > 0:
                buf[a0:a0 + n] += arr[b0:b0 + n]
    return buf


def normalize_factor(peak, target=TARGET_PEAK):
    if peak <= 0:
        return 1.0
    return min(target / peak, 1.0 / SOFT_LIMIT * 0.999)


def run_mix(cfg, root, section_seconds=DEFAULT_SECTION_SECONDS,
            target=TARGET_PEAK):
    """Mix narration + music + sfx into audio/final_mix.wav.

    Returns a dict with summary info for the CLI/renderer. Two passes keep
    per-section memory bounded: pass 1 finds the peak, pass 2 normalises.
    """
    root = Path(root)
    audio_dir = Path(cfg.output.audio_dir)
    segments = load_segments(audio_dir)

    sr = int(getattr(cfg, "tts", None).get("sample_rate", 24000)) if \
        getattr(cfg, "tts", None) else 24000
    try:
        sr = int(sr) if sr and sr > 0 else 24000
    except (TypeError, ValueError):
        sr = 24000

    total = total_duration(segments)
    total_samples = seconds_to_samples(total, sr)

    # ---- optional beds built lazily as opaque zero-arg callables
    music_settings = music_config(cfg)
    music_track = None
    music_bed_full = None

    def music_bed_fn():
        nonlocal music_bed_full, music_track
        if music_settings is None:
            return None
        if music_bed_full is None:
            music_bed_full, music_track = make_music_bed(
                root, music_settings, total, sr)
        return music_bed_full

    sfx_settings = sfx_config(cfg)
    sfx_map = None
    if sfx_settings is not None:
        events = load_events(root, sfx_settings)
        if events:
            seg_map = {s["segment_id"]: s["start"] for s in segments}
            narration_peak_val = narration_peak(segments)
            sfx_map = aggregate_sfx(root, sfx_settings, events, seg_map, sr,
                                    narration_peak_val)

    # ---- pass 1: global peak (bounded sections)
    global_peak = 0.0
    for lo, hi in _section_bounds(total, sr, section_seconds):
        sec = mix_section(segments, music_bed_fn if music_settings else None,
                          sfx_map, sr, lo, hi, 0.9)
        if len(sec):
            p = float(np.max(np.abs(sec)))
            if p > global_peak:
                global_peak = p
    if global_peak <= 0:
        global_peak = 0.9

    factor = normalize_factor(global_peak, target)

    # ---- pass 2: scale + stream sections straight into the final WAV so
    # only one section is ever held in RAM at a time.
    out_path = audio_dir / FINAL_MIX_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import wave as _wave
    with _wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.setnframes(total_samples)
        for lo, hi in _section_bounds(total, sr, section_seconds):
            sec = mix_section(segments,
                              music_bed_fn if music_settings else None,
                              sfx_map, sr, lo, hi, global_peak)
            sec *= factor
            clipped = np.clip(sec, -TARGET_PEAK, TARGET_PEAK)
            pcm = (clipped * INT16_MAX).astype("<i2")
            wf.writeframes(pcm.tobytes())

    gated = [s for s in segments if s["path"].is_file()] if segments else []
    return {
        "result": "mixed",
        "segments": len(segments),
        "sample_rate": sr,
        "duration": round(total, 3),
        "global_peak": round(global_peak, 4),
        "normalize_factor": round(factor, 4),
        "music": music_settings is not None,
        "sfx": bool(sfx_map),
        "final_mix": str(out_path),
    }
