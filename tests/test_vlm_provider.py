"""Tests: VLM default/fallback provider resolution (Task 10)."""

from pathlib import Path

import pytest

from config.loader import Config
from pipeline.vlm_provider import (
    FallbackVLMProvider,
    LocalVLMProvider,
    MockVLMProvider,
    OllamaVLMProvider,
    VLMNotConfigured,
    VLMProviderError,
    VLMUnavailable,
    create_vlm_provider,
)

ROOT = Path(__file__).resolve().parent.parent

SMOLVLM = "/home/madhav/manga_tools/models/SmolVLM-256M-Instruct"


def make_cfg(vlm=None):
    data = {
        "vlm": {
            "enabled": True,
            "provider": "local",
            "model": SMOLVLM,
            "fallback": "ollama",
            "ollama_url": "http://127.0.0.1:11434",
            "ollama_model": "moondream",
            "device": "cpu",
            "max_image_size": 768,
            "max_new_tokens": 256,
            "timeout_seconds": 120,
        }
    }
    if vlm:
        data["vlm"].update(vlm)
    return Config(data, ROOT)


def test_default_resolves_local_smolvlm_with_ollama_fallback():
    provider = create_vlm_provider(make_cfg())
    assert isinstance(provider, FallbackVLMProvider)
    assert isinstance(provider.primary, LocalVLMProvider)
    assert isinstance(provider.fallback, OllamaVLMProvider)
    assert provider.primary.model == SMOLVLM
    assert provider.fallback.model == "moondream"
    assert provider.name == "local"


def test_no_fallback_returns_single_local_provider():
    provider = create_vlm_provider(make_cfg({"fallback": ""}))
    assert isinstance(provider, LocalVLMProvider)


def test_ollama_only_resolution():
    provider = create_vlm_provider(
        make_cfg({"provider": "ollama", "fallback": ""})
    )
    assert isinstance(provider, OllamaVLMProvider)
    assert provider.model == "moondream"


def test_unknown_fallback_raises():
    with pytest.raises(VLMProviderError, match="vlm.fallback"):
        create_vlm_provider(make_cfg({"fallback": "spacex"}))


def test_local_without_model_and_no_fallback_raises_not_configured():
    with pytest.raises(VLMNotConfigured, match="vlm.model"):
        create_vlm_provider(make_cfg({"model": "", "fallback": ""}))


def test_runtime_fallback_when_primary_error():
    cfg = make_cfg()
    primary = MockVLMProvider(cfg, raise_on_analyze=VLMUnavailable("nope"))
    fallback = MockVLMProvider(cfg, response="fallback-ok")
    wrapper = FallbackVLMProvider(primary, fallback)
    assert wrapper.analyze_image("x.jpg", "prompt") == "fallback-ok"
    assert wrapper.name == "mock"
    assert wrapper.model == cfg.vlm.model


def test_primary_success_does_not_touch_fallback():
    cfg = make_cfg()
    primary = MockVLMProvider(cfg, response="primary-ok")
    fallback = MockVLMProvider(cfg, raise_on_analyze=VLMUnavailable("unused"))
    wrapper = FallbackVLMProvider(primary, fallback)
    assert wrapper.analyze_image("x.jpg", "prompt") == "primary-ok"
    assert wrapper.name == "mock"


def test_gemini_parses_multiple_api_keys(monkeypatch):
    from pipeline import vlm_provider as v

    cfg = make_cfg({"provider": "gemini", "api_key": "KEY_A, KEY_B,KEY_C"})
    provider = v.GeminiVLMProvider(cfg)
    assert provider.api_keys == ["KEY_A", "KEY_B", "KEY_C"]
    assert provider.api_key == "KEY_A"
    assert provider.model == "gemini-flash-lite-latest"

    # rotation cycles through keys
    assert provider._next_key() == "KEY_B"
    assert provider._next_key() == "KEY_C"
    assert provider._next_key() == "KEY_A"


def test_gemini_reads_key_from_environment(monkeypatch):
    from pipeline import vlm_provider as v

    monkeypatch.setenv("GEMINI_API_KEY", "ENV_KEY")
    cfg = make_cfg({"provider": "gemini", "api_key": ""})
    provider = v.GeminiVLMProvider(cfg)
    assert provider.api_keys == ["ENV_KEY"]


def test_gemini_without_key_raises_not_configured():
    from pipeline import vlm_provider as v

    cfg = make_cfg({"provider": "gemini", "api_key": ""})
    provider = v.GeminiVLMProvider(cfg)
    with pytest.raises(VLMNotConfigured):
        provider.analyze_image("x.jpg", "prompt")