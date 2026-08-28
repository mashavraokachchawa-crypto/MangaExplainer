"""Flat narration-manifest TTS generation (Task 14 + Task 15).

Produces the plain, playback-oriented output requested by Tasks 14 & 15:

    audio/
        segment_001.wav     <- one file per narration segment, in order
        segment_002.wav
        ...
        manifest.json

Task 14 - generate():
    * synthesizes ONE WAV per narration segment, strictly one at a time,
    * names them audio/segment_001.wav, segment_002.wav ... by order index,
    * preserves each segment's segment_id and EXACT narration text,
    * writes audio/manifest.json with {segment_id, text, audio_path, duration}.

Task 15 - finalize_timing():
    * reads the already-generated WAVs (never regenerates),
    * computes each segment's exact audio duration,
    * preserves segment order,
    * adds cumulative start_time / end_time to audio/manifest.json.

The provider is resolved via pipeline.pocket_tts and released (GC forced)
between segments to respect the ~4 GB CPU machine. Reference voice is handled
honestly: if the reference file is missing OR cloning is license-gated, we use
the built-in catalog voice and report it - never a faked clone.
"""
import gc
import json
import logging
from pathlib import Path

from .pocket_tts import (
    create_pocket_tts_provider,
    wav_duration,
)

LOG = logging.getLogger("mangaexplainer")

SEG_WAV = "segment_{0:03d}.wav"
MANIFEST = "manifest.json"


def segment_wav_path(out_dir, index):
    """index is 0-based; file is named segment_{index+1:03d}.wav (Task 14)."""
    return Path(out_dir) / SEG_WAV.format(index + 1)


def manifest_path(out_dir):
    return Path(out_dir) / MANIFEST


def validate_segment(seg, index):
    seg_id = str(seg.get("segment_id") or "").strip()
    text = str(seg.get("text") or "").strip()
    if not seg_id:
        raise ValueError(f"segment #{index}: segment_id is empty")
    if not text:
        raise ValueError(f"segment #{index} ({seg_id!r}): text is empty")
    return seg_id, text


class NarrationManifestRunner:
    """Task 14 + 15 manifest runner (lightweight, one segment at a time)."""

    def __init__(self, cfg, provider=None):
        self.cfg = cfg
        self.provider = provider
        self._owned_provider = provider is None
        self._provider = None

    # -- Task 14 -----------------------------------------------------------

    def generate(self, segments, out_dir, state=None, force=False):
        """Synthesize one WAV per segment into out_dir; returns the manifest.

        segments: list of narration segment dicts in pipeline order.
        out_dir:  directory for audio/segment_NNN.wav + audio/manifest.json.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest = []
        provider = self._provider or create_pocket_tts_provider(self.cfg)

        try:
            for index, seg in enumerate(segments):
                seg_id, text = validate_segment(seg, index)
                out_path = segment_wav_path(out_dir, index)

                if not force and out_path.is_file() and out_path.stat().st_size > 0:
                    duration = wav_duration(out_path)
                else:
                    provider.synth(text, out_path, target_seconds=None)
                    gc.collect()
                    duration = wav_duration(out_path)

                entry = {
                    "segment_id": seg_id,
                    "text": text,
                    "audio_path": str(out_path),
                    "duration": round(duration, 4),
                }
                if state is not None:
                    state.mark_item_done(f"narration_{seg_id}", "tts_completed")
                manifest.append(entry)
        finally:
            if self._owned_provider:
                try:
                    self._provider = None
                    provider.release()
                except Exception:
                    pass
                gc.collect()

        self.write_manifest(out_dir, manifest)
        return manifest

    # -- Task 15 -----------------------------------------------------------

    def finalize_timing(self, out_dir):
        """Compute exact durations + cumulative start/end times in MANIFEST.

        Reads existing WAVs; never regenerates audio. Preserves segment order.
        Returns the updated manifest list.
        """
        out_dir = Path(out_dir)
        manifest_path_ = manifest_path(out_dir)
        if not manifest_path_.is_file():
            raise FileNotFoundError(
                f"no {MANIFEST} at {out_dir} - run generation (Task 14) first"
            )
        manifest = json.loads(manifest_path_.read_text("utf-8"))
        if not isinstance(manifest, list):
            raise ValueError("manifest.json must be a list of segment entries")

        start = 0.0
        count = 0
        updated = []
        for entry in manifest:
            if not isinstance(entry, dict):
                raise ValueError("manifest.json entries must be objects")
            audio_path = Path(entry.get("audio_path") or "")
            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"audio missing for {entry.get('segment_id')!r}: {audio_path}"
                )
            duration = wav_duration(audio_path)
            entry["duration"] = round(duration, 4)
            entry["start_time"] = round(start, 4)
            start += duration
            entry["end_time"] = round(start, 4)
            updated.append(entry)
            count += 1

        manifest_path_.write_text(
            json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return updated

    # -- helpers -----------------------------------------------------------

    def write_manifest(self, out_dir, manifest):
        manifest_path(out_dir).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def run(self, segments, out_dir, state=None, force=False):
        """Task 14 + 15 in one pass: generate, then compute start/end times."""
        manifest = self.generate(segments, out_dir, state=state, force=force)
        return self.finalize_timing(out_dir)


def load_narration_segments(json_path):
    """Read a narration script JSON, returning its ordered segments list."""
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"narration script not found: {path}")
    data = json.loads(path.read_text("utf-8"))
    segments = data.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"no segments in {path}")
    return segments
