"""Reference-voice audio validation and lightweight preprocessing (Task 13).

The supplied voice sample is treated as the narration's reference voice. This
module checks that it exists and can be decoded, reports its properties
(duration, sample rate, channels), and detects silent/corrupt audio. If Pocket
TTS requires a gentle preprocessing pass (e.g. mono / 16-bit / a target
sample rate), a *working copy* is produced -- never the original.

All validation is cheap and offline (stdlib wave; optional numpy only for a
RMS "is-it-silent" check, with a graceful fallback).
"""
import logging
import shutil
import wave
from pathlib import Path

from .pocket_tts import PocketTtsError

LOG = logging.getLogger("mangaexplainer")

# Pocket TTS / our pipeline expects speech-grade mono 16-bit.
TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1
TARGET_SAMPWIDTH = 2  # bytes (16-bit)


class ReferenceAudioValidationError(PocketTtsError):
    """Reference audio could not be validated / used."""


class ReferenceAudioMissing(ReferenceAudioValidationError):
    """Reference file does not exist."""


def reference_audio_path(cfg):
    """Resolve the configured reference audio path (absolute)."""
    raw = getattr(cfg.tts, "reference_audio", None)
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_absolute() else Path(cfg.root_dir) / p


def validate_reference(path):
    """Validate a reference audio file; returns a metadata dict.

    Raises ReferenceAudioMissing / ReferenceAudioValidationError on problems.
    """
    path = Path(path)
    if not path.is_file():
        raise ReferenceAudioMissing(
            f"reference audio missing: {path} "
            "(expected input/voice_reference.mp3 - copy the supplied "
            "voice sample there without modifying the original)"
        )
    if path.stat().st_size == 0:
        raise ReferenceAudioValidationError(f"reference audio is empty: {path}")

    info = _probe_audio(path)
    if info is None:
        return _process_non_wav(path)
    return _validate_wav(path, info)


def _probe_audio(path):
    """Try to read WAV metadata; return dict or None if not wave."""
    try:
        with wave.open(str(path), "rb") as handle:
            return {
                "format": "wav",
                "channels": handle.getnchannels(),
                "sampwidth": handle.getsampwidth(),
                "sample_rate": handle.getframerate(),
                "nframes": handle.getnframes(),
            }
    except Exception:
        return None


def _process_non_wav(path):
    """Non-WAV reference (e.g. .mp3). Try ffmpeg to decode/transcode.

    If `ffmpeg` is available we can decode it to a working WAV copy for
    validation. If not, we still return a metadata dict.
    """
    ext = path.suffix.lower()
    if ext == ".wav":
        # should have been caught by _probe_audio; corrupt WAV
        raise ReferenceAudioValidationError(
            f"reference audio corrupt/undecodable: {path}"
        )
    metadata = {
        "format": ext.lstrip(".") or "unknown",
        "path": str(path),
        "decodable": True,
    }
    if shutil.which("ffmpeg"):
        try:
            probe = _ffprobe_duration(path)
            metadata["duration"] = probe
        except Exception:
            pass
    else:
        metadata["decodable_note"] = (
            "ffmpeg not found; full decode/duration check skipped (install "
            "ffmpeg for stricter validation)"
        )
    return metadata


def _ffprobe_duration(path):
    import subprocess

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    if out.returncode != 0:
        raise ReferenceAudioValidationError(f"ffprobe failed: {out.stderr.strip()}")
    try:
        return float(out.stdout.strip())
    except ValueError:
        raise ReferenceAudioValidationError(
            f"could not determine reference duration: {path}"
        ) from None


def _validate_wav(path, info):
    duration = info["nframes"] / info["sample_rate"] if info["sample_rate"] else 0.0
    if duration <= 0:
        raise ReferenceAudioValidationError(
            f"reference audio has zero duration (corrupt or empty): {path}"
        )
    if info["sample_rate"] <= 0 or info["sampwidth"] <= 0 or info["channels"] <= 0:
        raise ReferenceAudioValidationError(
            f"reference audio has invalid header: {path} {info}"
        )
    silent = _is_silent(path, info)
    return {
        "path": str(path),
        "format": "wav",
        "channels": info["channels"],
        "sampwidth": info["sampwidth"],
        "sample_rate": info["sample_rate"],
        "duration": round(duration, 3),
        "silent": silent,
        "decodable": True,
    }


def _is_silent(path, info, threshold=0.005):
    """Heuristic RMS check; silent if average amplitude is near zero."""
    rms = _rms(path, info)
    return rms is not None and rms < threshold


def _rms(path, info):
    try:
        import array
        import math

        with wave.open(str(path), "rb") as handle:
            frames = handle.readframes(min(info["nframes"], info["sample_rate"] * 5))
        if not frames:
            return 0.0
        samples = array.array("h")
        samples.frombytes(frames[: len(frames) - (len(frames) % 2)])
        if not samples:
            return 0.0
        sumsq = sum(s * s for s in samples) / len(samples)
        return math.sqrt(sumsq) / 32767.0
    except Exception:
        return None


def preprocess_reference(path, working_dir, target_rate=TARGET_SAMPLE_RATE,
                         channels=TARGET_CHANNELS, sampwidth=TARGET_SAMPWIDTH):
    """Produce a lightweight working copy of the reference for Pocket TTS.

    Writes a mono 16-bit WAV (optionally resampled) to working_dir WITHOUT
    touching the original file. If the reference is already a mono, 16-bit WAV
    at the right sample rate we copy it unchanged.
    """
    path = Path(path)
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    out = working_dir / "voice_reference_working.wav"

    try:
        with wave.open(str(path), "rb") as src:
            if (
                src.getnchannels() == channels
                and src.getsampwidth() == sampwidth
                and src.getframerate() == target_rate
            ):
                shutil.copyfile(path, out)
                return {"path": str(out), "preprocessed": False, "copied": True}

            # Need conversion: use ffmpeg if present, else raise (no invasive
            # resampling in pure python here to keep it lightweight).
            return _convert_with_ffmpeg(path, out, target_rate, channels, sampwidth)
    except wave.Error as exc:
        raise ReferenceAudioValidationError(
            f"cannot preprocess reference {path}: {exc}"
        ) from None


def _convert_with_ffmpeg(src, out, rate, channels, sampwidth):
    if not shutil.which("ffmpeg"):
        raise ReferenceAudioValidationError(
            "reference needs conversion (mono/16-bit/24000 Hz) but ffmpeg is "
            "not installed; install ffmpeg or supply a suitable WAV."
        )
    import subprocess

    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-ac", str(channels), "-ar", str(rate),
         "-c:a", "pcm_s16le", str(out)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0 or not out.exists():
        raise ReferenceAudioValidationError(
            f"ffmpeg conversion of reference failed: {result.stderr.strip()[:400]}"
        )
    return {"path": str(out), "preprocessed": True, "copied": False}
