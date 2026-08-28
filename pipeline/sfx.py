"""Sound effects support (Task 20).

Places optional, panel-specific short sound effects onto the timeline: each
event has a local audio file, a volume, a start time and a duration. Volume is
bounded by a configurable ceiling so an SFX never overpowers narration, and a
hard relative cap keeps any SFX well below the loudest narration peak.

An SFX event:
    {
      "panel_id": "p001_001",   # optional - purely descriptive
      "segment_id": "seg_001",  # optional - link to narration segment
      "file": "sfx/thunder.wav",
      "volume": 0.6,            # 0..1 (further bounded by sfx.max_volume)
      "start_time": 12.5,       # seconds into the final mix (or offset below)
      "duration": 2.0,          # seconds to keep the SFX audible (snipped)
      "offset": 0.0             # optional: seconds into the narration segment
    }

SFX come from LOCAL files only; nothing is generated with AI. No subtitles.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .audio_io import apply_fades, read_wav, resample

LOG = logging.getLogger(__name__)

DEFAULT_MAX_VOLUME = 0.35      # SFX never louder than this fraction of FS
DEFAULT_DURATION = 0.0         # no forced length unless requested
DEFAULT_SFX_FILE = "sfx/sfx_manifest.json"


class SfxUnavailable(Exception):
    """SFX are disabled (or no usable events/audio found)."""


def sfx_config(cfg):
    """Normalised SFX settings; returns None when disabled."""
    sfx = getattr(cfg, "sfx", None)
    if sfx is None:
        return None
    enabled = bool(sfx.get("enabled", False))
    if not enabled:
        return None
    return {
        "max_volume": float(sfx.get("max_volume", DEFAULT_MAX_VOLUME)),
        "dir": str(sfx.get("dir", "sfx")),
        "manifest": str(sfx.get("manifest", DEFAULT_SFX_FILE)),
        "fade": float(sfx.get("fade", 0.02)),
    }


def load_events(root, settings):
    """Load the SFX event manifest (list of dicts) from disk.

    Returns a list of events, each with an absolute 'file'. An empty list is a
    valid result (no SFX); a missing/broken manifest raises SfxUnavailable.
    """
    root = Path(root)
    manifest = Path(settings["manifest"])
    if not manifest.is_absolute():
        manifest = root / manifest
    if not manifest.is_file():
        # an empty manifest is fine (verbose off by default) - still resolve
        # files lazily; but a configured manifest that's missing is an error.
        raise SfxUnavailable(f"sfx manifest not found: {manifest}")
    try:
        doc = json.loads(manifest.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SfxUnavailable(f"cannot read sfx manifest {manifest}: {exc}")
    if isinstance(doc, dict):
        doc = doc.get("effects") or doc.get("sfx") or []
    if not isinstance(doc, list):
        raise SfxUnavailable("sfx manifest must be a list of events")
    events = []
    for raw in doc:
        if not isinstance(raw, dict):
            continue
        rel = raw.get("file")
        if not rel:
            continue
        f = Path(rel)
        if not f.is_absolute():
            f = root / settings["dir"] / rel
        events.append({
            "panel_id": raw.get("panel_id"),
            "segment_id": raw.get("segment_id"),
            "file": str(f),
            "volume": float(raw.get("volume", 1.0)),
            "start_time": float(raw.get("start_time", 0.0)),
            "duration": float(raw.get("duration", DEFAULT_DURATION)),
            "offset": float(raw.get("offset", 0.0)) if raw.get("offset") is
            not None else 0.0,
        })
    return events


def event_absolute_start(event, segment_start):
    """Resolve an SFX time to the absolute mix position.

    If the event has an explicit 'start_time' that wins. Otherwise it is
    placed 'offset' seconds after the linked narration segment start, so events
    tied to a panel follow that panel's narration timing.
    """
    if event.get("start_time") is not None:
        return float(event["start_time"])
    if event.get("offset") is not None:
        return float(segment_start) + float(event["offset"])
    return float(segment_start)


def limit_against_narration(sfx_level, narration_peak, cfg_max):
    """Bound an SFX level so narration always stays dominant.

    Returns the effective level, never exceeding cfg_max and never louder
    than a fraction (0.5) of the narration peak (so speech wins).
    """
    ceiling = min(float(cfg_max), (0.5 * float(narration_peak)) if
                  float(narration_peak) > 0 else float(cfg_max))
    return min(float(sfx_level), max(ceiling, 0.0))


def build_sfx_segment(event, sr):
    """Load + prepare a single SFX wave at sample rate sr.

    Returns a float32 mono array already scaled and sniped to the requested
    'duration' (0 = keep the whole file), with a tiny fade to avoid clicks.
    """
    data_sr, raw = read_wav(event["file"])
    data = resample(raw, data_sr, sr).astype(np.float32, copy=False)
    duration = float(event.get("duration") or 0.0)
    if duration > 0:
        n = int(round(duration * sr))
        data = data[:n] if len(data) >= n else \
            np.pad(data, (0, n - len(data)))
    fade = 0.02
    data = apply_fades(data, sr, fade, fade)
    data *= float(event["volume"])
    return data


def aggregate_sfx(root, settings, events, segment_times, sr, narration_peak):
    """Place all SFX events into a sparse map of (start_sample -> array).

    segment_times: {segment_id: start_seconds} from the narration manifest.
    narration_peak: peak amplitude of the narration timeline (for the cap).
    Returns {start_sample: array} (already bounded so SFX never overpowers).
    """
    placed = {}
    for ev in events:
        start = event_absolute_start(ev, segment_times.get(ev.get("segment_id"),
                                                           0.0))
        level = limit_against_narration(ev["volume"], narration_peak,
                                        settings["max_volume"])
        data = build_sfx_segment(ev, sr)
        data = data * (level / max(float(ev["volume"]), 1e-9)) if \
            float(ev["volume"]) > 0 else data * 0.0
        start_sample = int(round(start * sr))
        placed.setdefault(start_sample, np.zeros(0, dtype=np.float32))
        placed[start_sample] = _add(placed[start_sample], data)
    return placed


def _add(a, b):
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    if len(a) >= len(b):
        a[:len(b)] += b
        return a
    out = np.zeros(len(b), dtype=np.float32)
    out[:len(a)] = a
    out[len(a):] += b[len(a):]
    return out
