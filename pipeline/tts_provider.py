"""TTS speech provider for the narration audio stage.

Two engines, both low-RAM and offline:
- "espeak": wraps the system espeak-ng (or espeak) binary - real synthetic
  speech with no model, no download, seconds of CPU.
- "mock": deterministic sine-tone WAV generator (stdlib wave + array) for
  tests and offline smoke runs. Duration follows the script's
  estimated_seconds so timing can be asserted without a speech engine.

"auto" (default) picks espeak when its binary is on PATH and falls back to
the mock engine otherwise, mirroring how vlm/llm resolve mock vs local.
"""
import array
import shutil
import subprocess
import wave
from pathlib import Path

SAMPLE_RATE = 22050
_BASE_FREQ = 220.0
_SPEAKER_FREQ_STEP = 25.0
_SPEAKER_VARIETIES = 7


class TtsError(Exception):
    pass


class TtsUnavailable(TtsError):
    pass


class TtsNotConfigured(TtsError):
    pass


def _seed_for(speaker):
    if not speaker:
        return 0
    value = sum((index + 1) * ord(ch) for index, ch in enumerate(str(speaker)))
    return value % _SPEAKER_VARIETIES


def _mock_frames(text, target_seconds, sample_rate, speaker):
    """Deterministic sine wave frames for the given duration."""
    seconds = max(0.05, float(target_seconds or 1.0))
    total = int(seconds * sample_rate)
    freq = _BASE_FREQ + _seed_for(speaker) * _SPEAKER_FREQ_STEP
    phase = 0.0
    step = 2.0 * 3.141592653589793 * freq / sample_rate
    frames = array.array("h")
    frames.extend(
        int(0.2 * 32767 * __import__("math").sin(phase + step * i))
        for i in range(total)
    )
    return frames.tobytes()


def _write_wav(path, frames, sample_rate=SAMPLE_RATE):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(frames)
    return len(frames) / (sample_rate * 2)


def wav_duration(path):
    """Actual duration in seconds of a mono 16-bit WAV file."""
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate() or SAMPLE_RATE
        return frames / rate


class TtsProvider:
    name = "tts"

    def __init__(self, cfg):
        self.cfg = cfg

    @property
    def sample_rate(self):
        return SAMPLE_RATE

    @staticmethod
    def available():
        return False

    def release(self):
        pass

    def synth(self, text, out_path, target_seconds=None, speaker=None):
        raise NotImplementedError


class MockTtsProvider(TtsProvider):
    """Deterministic WAV generator for tests / offline smoke runs."""

    name = "mock"

    @property
    def sample_rate(self):
        return int(self.cfg.tts.sample_rate)

    @staticmethod
    def available():
        return True

    def synth(self, text, out_path, target_seconds=None, speaker=None):
        rate = self.sample_rate
        frames = _mock_frames(text, target_seconds, rate, speaker)
        return _write_wav(out_path, frames, sample_rate=rate)


class EspeakProvider(TtsProvider):
    """Real offline speech via the system espeak-ng / espeak binary."""

    name = "espeak"

    @property
    def binary(self):
        return shutil.which("espeak-ng") or shutil.which("espeak")

    @staticmethod
    def available():
        return bool(shutil.which("espeak-ng") or shutil.which("espeak"))

    def synth(self, text, out_path, target_seconds=None, speaker=None):
        binary = self.binary
        if not binary:
            raise TtsUnavailable(
                "espeak-ng is not installed (install it with e.g. "
                "'sudo apt install espeak-ng') or set tts.engine=mock"
            )
        args = [binary, "-w", str(out_path)]
        voice = str(self.cfg.tts.voice or "en").strip() or "en"
        rate = int(self.cfg.tts.rate_wpm or 150)
        args += ["-v", voice, "-s", str(rate)]
        if speaker:
            pitch = int(self.cfg.tts.pitch_base or 50) + _seed_for(speaker) * 5
            args += ["-p", str(max(0, min(99, pitch)))]
        args += [str(text)]
        try:
            subprocess.run(args, check=True, capture_output=True, timeout=60)
        except subprocess.CalledProcessError as exc:
            raise TtsError(
                f"espeak failed (exit {exc.returncode}): "
                f"{exc.stderr.decode('utf-8', 'replace') or exc.stdout.decode('utf-8', 'replace')}"
            ) from None
        except subprocess.TimeoutExpired:
            raise TtsError("espeak timed out synthesizing a segment") from None
        return wav_duration(out_path)


PROVIDERS = {
    "espeak": EspeakProvider,
    "mock": MockTtsProvider,
}


def create_tts_provider(cfg):
    """Resolve the configured TTS engine; raises TtsError."""
    if not bool(getattr(cfg, "tts", None) and cfg.tts.enabled):
        raise TtsNotConfigured(
            "TTS is disabled (tts.enabled=false in config). Set enabled=true "
            "to generate narration audio."
        )
    engine = str(getattr(cfg, "tts", None) and getattr(cfg.tts, "engine", "auto") or "auto").lower()
    if engine == "auto":
        if EspeakProvider.available():
            return EspeakProvider(cfg)
        import logging
        logging.getLogger("mangaexplainer").warning(
            "espeak-ng not found on PATH; using mock TTS (sine tones, no speech). "
            "Install espeak-ng or set tts.engine explicitly."
        )
        return MockTtsProvider(cfg)
    if engine not in PROVIDERS:
        raise TtsError(
            f"unknown tts.engine {engine!r} "
            f"(expected one of {', '.join(PROVIDERS)}, auto)"
        )
    cls = PROVIDERS[engine]
    if engine == "espeak" and not cls.available():
        raise TtsUnavailable(
            "tts.engine=espeak but espeak-ng is not installed "
            "(install it, or set tts.engine=mock)"
        )
    return cls(cfg)