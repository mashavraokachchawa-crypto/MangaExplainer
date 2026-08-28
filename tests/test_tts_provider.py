"""Tests: TTS speech provider abstraction (espeak binary / mock engine)."""

from pathlib import Path

import pytest

from config.loader import Config
from pipeline.tts_provider import (
    MockTtsProvider,
    TtsError,
    TtsNotConfigured,
    TtsUnavailable,
    create_tts_provider,
    wav_duration,
)

ROOT = Path(__file__).resolve().parent.parent


def make_cfg(tts=None):
    data = {
        "tts": {
            "enabled": True, "engine": "auto", "voice": "en",
            "sample_rate": 22050, "rate_wpm": 150, "pitch_base": 50,
            "timeout_seconds": 60,
        }
    }
    if tts:
        data["tts"].update(tts)
    return Config(data, ROOT)


def read_wav(path):
    import array
    import wave

    with wave.open(str(path), "rb") as handle:
        return {
            "channels": handle.getnchannels(),
            "sampwidth": handle.getsampwidth(),
            "framerate": handle.getframerate(),
            "nframes": handle.getnframes(),
            "frames": handle.readframes(handle.getnframes()),
        }


def test_mock_provider_synthesizes_valid_wav(tmp_path):
    cfg = make_cfg()
    provider = MockTtsProvider(cfg)
    out = tmp_path / "a.wav"
    duration = provider.synth("Hello.", out, target_seconds=2.0)
    meta = read_wav(out)
    assert meta["channels"] == 1
    assert meta["sampwidth"] == 2
    assert meta["framerate"] == 22050
    assert meta["nframes"] > 0
    assert duration == pytest.approx(2.0, abs=0.02)
    assert wav_duration(out) == pytest.approx(2.0, abs=0.02)
    assert provider.available()


def test_mock_duration_tracks_requested_seconds(tmp_path):
    provider = MockTtsProvider(make_cfg())
    for seconds in (0.5, 1.0, 4.25):
        out = tmp_path / f"{seconds}.wav"
        provider.synth("x", out, target_seconds=seconds)
        assert wav_duration(out) == pytest.approx(seconds, abs=0.05)


def test_mock_speaker_changes_frequency(tmp_path):
    provider = MockTtsProvider(make_cfg())
    seen = set()
    for name in ("Guts", "Griffith", "Casca", "Chitch", "Rickert"):
        out = tmp_path / f"{name}.wav"
        provider.synth("x", out, target_seconds=1.0, speaker=name)
        seen.add(read_wav(out)["frames"])
    assert len(seen) >= 2


def test_espeak_binary_when_absent_raises_unavailable(tmp_path):
    cfg = make_cfg({"engine": "espeak"})
    if MockTtsProvider(cfg).available():
        pytest.skip("espeak-ng installed on this machine")
    with pytest.raises(TtsUnavailable):
        create_tts_provider(cfg)


def test_auto_falls_back_to_mock_when_no_espeak(monkeypatch, tmp_path):
    monkeypatch.setattr("pipeline.tts_provider.EspeakProvider.available",
                        staticmethod(lambda: False))
    provider = create_tts_provider(make_cfg({"engine": "auto"}))
    assert isinstance(provider, MockTtsProvider)
    assert provider.available()


def test_auto_prefers_espeak_when_installed(monkeypatch, tmp_path):
    monkeypatch.setattr("pipeline.tts_provider.EspeakProvider.available",
                        staticmethod(lambda: True))
    provider = create_tts_provider(make_cfg({"engine": "auto"}))
    assert provider.name == "espeak"


def test_unknown_engine_raises():
    with pytest.raises(TtsError, match="unknown tts.engine"):
        create_tts_provider(make_cfg({"engine": "wat"}))


def test_disabled_raises_not_configured():
    with pytest.raises(TtsNotConfigured, match="disabled"):
        create_tts_provider(make_cfg({"enabled": False}))


def test_mock_honors_custom_sample_rate(tmp_path):
    cfg = make_cfg({"sample_rate": 8000})
    provider = MockTtsProvider(cfg)
    out = tmp_path / "r.wav"
    provider.synth("hi", out, target_seconds=1.0)
    assert read_wav(out)["framerate"] == 8000
    assert provider.sample_rate == 8000