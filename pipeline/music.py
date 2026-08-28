"""Background music support (Task 19).

Produces an optional music BED for the narration timeline: a local audio file
looped as needed (auto-loop), faded in/out, scaled to a configurable volume,
and always kept well below the narration level so speech stays clearly
audible.

Design:
   * music.enabled - master on/off
   * music.volume   - 0..1 bed level (kept low by default, never >1)
   * music.fade_in / music.fade_out - seconds of smooth fade
   * music.dir      - directory of local tracks
   * music.file     - optional explicit track relative to music.dir
   * music.loop     - loop the track to cover the full narration duration
   * quiet ceiling  - even at volume=1 the bed never exceeds music.max_level
                       (a fraction of full scale), so narration always wins

No AI is used to generate music; only local audio files are ever read.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .audio_io import apply_fades, read_wav, resample

LOG = logging.getLogger(__name__)

DEFAULT_VOLUME = 0.20        # narration stays clearly audible by default
DEFAULT_MAX_LEVEL = 0.50     # hard ceiling on bed amplitude (fraction of FS)
DEFAULT_FADE = 0.5           # seconds
DEFAULT_MUSIC_DIR = "music"


class MusicUnavailable(Exception):
    """No local music could be loaded (disabled, missing, or unreadable)."""


def music_config(cfg):
    """Normalised music settings; returns None when music is disabled."""
    music = getattr(cfg, "music", None)
    if music is None:
        return None
    enabled = bool(music.get("enabled", False))
    if not enabled:
        return None
    return {
        "volume": float(music.get("volume", DEFAULT_VOLUME)),
        "max_level": float(music.get("max_level", DEFAULT_MAX_LEVEL)),
        "fade_in": float(music.get("fade_in", DEFAULT_FADE)),
        "fade_out": float(music.get("fade_out", DEFAULT_FADE)),
        "loop": bool(music.get("loop", True)),
        "dir": str(music.get("dir", DEFAULT_MUSIC_DIR)),
        "file": music.get("file") or None,
        "max_duration": float(music.get("max_duration", 0.0)),
    }


def resolve_track(root, settings, explicit=None):
    """Pick a local music file under root/settings['dir'].

    Prefers an explicit file (settings['file'] or the explicit arg); otherwise
    scans the dir for common audio extensions and picks the first. Returns an
    absolute Path or raises MusicUnavailable.
    """
    root = Path(root)
    music_dir = root / settings["dir"]
    candidate = explicit or settings.get("file")
    if candidate:
        p = Path(candidate)
        if not p.is_absolute():
            p = music_dir / p
        if p.is_file():
            return p
        raise MusicUnavailable(f"music track not found: {p}")
    if not music_dir.is_dir():
        raise MusicUnavailable(f"music dir not found: {music_dir}")
    for ext in ("wav", "mp3", "flac", "ogg", "aac", "m4a"):
        found = sorted(music_dir.glob(f"*.{ext}"))
        if found:
            return found[0]
    raise MusicUnavailable(f"no music files in {music_dir}")


def make_music_bed(root, settings, duration, sr, explicit=None, track=None):
    """Build the music bed covering `duration` seconds at sample rate `sr`.

    Returns (bed_float32_mono, track_path) for the mixer. The bed is already
    scaled to settings['volume'] with fades applied and looped as required.
    When no music is usable this raises MusicUnavailable so the caller can
    continue without a bed.
    """
    if track is None:
        track = resolve_track(root, settings, explicit)
    track = Path(track)
    track_sr, raw = read_wav(track)
    if settings.get("max_duration"):
        raw = raw[: int(round(settings["max_duration"] * track_sr))]
    data = resample(raw, track_sr, sr)
    data = data.astype(np.float32, copy=False)
    if len(data) == 0:
        raise MusicUnavailable(f"music track has no audio: {track}")

    total = int(round(duration * sr))
    if settings["loop"]:
        bed = np.resize(data, total) if total else np.zeros(0, dtype=np.float32)
    else:
        if total <= len(data):
            bed = data[:total]
        else:
            bed = np.concatenate([data, np.zeros(total - len(data),
                                                 dtype=np.float32)])
    bed = apply_fades(bed, sr, settings["fade_in"], settings["fade_out"])
    level = min(settings["volume"], settings["max_level"])
    bed *= level
    return bed.astype(np.float32), track
