"""Tests: text-only LLM provider resolution and cleaning."""

import sys
from pathlib import Path

import pytest

from config.loader import Config
from pipeline.llm_provider import (
    LLMNotConfigured,
    LLMProviderError,
    LLMUnavailable,
    MockLLMProvider,
    LocalLLMProvider,
    clean_text,
    create_llm_provider,
)

ROOT = Path(__file__).resolve().parent.parent


def make_cfg(llm=None):
    data = {
        "llm": {
            "enabled": True,
            "provider": "local",
            "model": "",
            "device": "cpu",
            "max_context": 4096,
            "max_new_tokens": 512,
            "temperature": 0.7,
            "timeout_seconds": 120,
        }
    }
    if llm:
        data["llm"].update(llm)
    return Config(data, ROOT)


def test_clean_text_strips_markdown_fences():
    assert clean_text("```text\nNarration: The storm hit.\n```") == "The storm hit."


def test_clean_text_strips_quotes_and_prefix():
    assert clean_text('"Narration: Silent."') == "Silent."
    assert clean_text("narration: they waited.") == "they waited."


def test_clean_text_handles_garbage():
    assert clean_text(None) == ""
    assert clean_text("   ") == ""
    assert clean_text(123) == ""


def test_create_mock_provider_generates():
    provider = create_llm_provider(make_cfg({"provider": "mock"}), response="A line.")
    assert isinstance(provider, MockLLMProvider)
    assert provider.model == "mock-model"
    assert provider.generate("prompt") == "A line."
    assert provider.last_prompt == "prompt"


def test_mock_uses_configured_model_name():
    provider = create_llm_provider(
        make_cfg({"provider": "mock", "model": "tiny"}), response="A line."
    )
    assert provider.model == "tiny"


def test_unknown_provider_raises():
    cfg = make_cfg({"provider": "wat"})
    with pytest.raises(LLMProviderError, match="unknown llm.provider"):
        create_llm_provider(cfg)


def test_disabled_raises_not_configured():
    cfg = make_cfg({"enabled": False, "provider": "mock"})
    with pytest.raises(LLMNotConfigured, match="disabled"):
        create_llm_provider(cfg)


def test_local_empty_model_raises_not_configured():
    cfg = make_cfg({"provider": "local", "model": ""})
    with pytest.raises(LLMNotConfigured, match="llm.model"):
        create_llm_provider(cfg)


def test_local_missing_transformers_raises_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    cfg = make_cfg({"provider": "local", "model": "/some/local/model"})
    provider = create_llm_provider(cfg)
    assert isinstance(provider, LocalLLMProvider)
    with pytest.raises(LLMUnavailable, match="transformers"):
        provider.generate("narrate")


def test_local_model_path_missing_raises_unavailable(monkeypatch):
    # transformers importable but the model path does not exist
    fake = type("fake", (), {"AutoModelForCausalLM": object, "AutoTokenizer": object})
    monkeypatch.setitem(sys.modules, "transformers", fake)
    cfg = make_cfg({"provider": "local", "model": "/definitely/not/here"})
    provider = create_llm_provider(cfg)
    with pytest.raises(LLMUnavailable, match="not found locally"):
        provider.generate("narrate")


def test_mock_raise_on_generate_propagates():
    provider = create_llm_provider(
        make_cfg({"provider": "mock"}),
        raise_on_generate=LLMUnavailable("boom"),
    )
    with pytest.raises(LLMUnavailable, match="boom"):
        provider.generate("prompt")


def test_release_is_safe():
    provider = create_llm_provider(make_cfg({"provider": "mock"}))
    provider.release()


def test_default_provider_is_local():
    cfg = make_cfg({"model": "some/local/path"})
    assert create_llm_provider(cfg).name == "local"


def test_local_provider_reads_generation_settings():
    cfg = make_cfg({"provider": "local", "model": "some/path", "temperature": 0.3, "max_context": 1024})
    provider = create_llm_provider(cfg)
    assert provider.max_context == 1024
    assert provider.temperature == 0.3