"""Modular text-only LLM providers for scene narration.

Same provider contract philosophy as vlm_provider but for prose: every
provider implements ``generate(prompt, timeout) -> raw text``, and the app
never depends on one specific LLM - ``create_llm_provider(cfg)`` resolves the
configured one so the backend can be swapped later (local causal LM, hosted
API, ...).

Low-RAM contract:
- narration is generated ONE scene at a time (never a whole script in memory).
- the local provider loads the model lazily, stays on CPU, constrains the
  generation to llm.max_new_tokens and never auto-downloads a model
  (local_files_only=True - a clear error is raised when the path is missing).
- call provider.release() after generation to drop the model references.
"""
import gc
import json
import logging
import re
from pathlib import Path

LOG = logging.getLogger("mangaexplainer")


class LLMProviderError(Exception):
    pass


class LLMNotConfigured(LLMProviderError):
    pass


class LLMUnavailable(LLMProviderError):
    pass


class LLMTimeout(LLMProviderError):
    pass


class LLMFailure(LLMProviderError):
    pass


_CODE_FENCE = re.compile(r"```[a-zA-Z]*\s*(.*?)```", re.DOTALL)
_PREFIX = re.compile(r"^(narration|narrator)\s*:\s*", re.IGNORECASE)


def clean_text(raw):
    """Coerce an LLM narration response to plain prose."""
    if not isinstance(raw, str):
        return ""
    text = raw.strip()
    match = _CODE_FENCE.search(text)
    if match:
        text = match.group(1).strip()
    text = text.strip('"').strip()
    text = _PREFIX.sub("", text).strip()
    return text


class LocalLLMProvider:
    """Lightweight local causal language model via transformers.

    Model resolution is intentionally strict: transformers is loaded with
    local_files_only=True so a download NEVER happens automatically. Point
    ``llm.model`` at a local directory (or a previously cached model id).
    """

    name = "local"

    def __init__(self, cfg):
        self.model = str(cfg.llm.model or "").strip()
        self.device = str(cfg.llm.device or "cpu").lower()
        self.max_context = int(cfg.llm.max_context)
        self.max_new_tokens = int(cfg.llm.max_new_tokens)
        self.timeout_seconds = int(cfg.llm.timeout_seconds)
        self.temperature = float(getattr(cfg.llm, "temperature", 0.0) or 0.0)
        self._refs = None

    @staticmethod
    def available():
        return True

    def _load(self):
        if self._refs is not None:
            return self._refs
        if not self.model:
            raise LLMNotConfigured(
                "LLM model not configured: set llm.model in config/config.yaml "
                "(provide a local model path or a model already present in the "
                "transformers cache - models are never downloaded automatically)"
            )
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise LLMUnavailable(
                "transformers is not installed; the 'local' LLM provider needs "
                "it (pip install transformers --no-deps and a lightweight model)"
            ) from exc
        if not Path(self.model).is_dir():
            raise LLMUnavailable(
                f"LLM model path not found locally: {self.model!r}. Configure "
                "llm.model to an existing local directory (download the model "
                "separately - it is never fetched automatically)"
            )
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model, local_files_only=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
        except Exception as exc:
            raise LLMUnavailable(
                f"failed to load LLM model from {self.model!r}: {exc}"
            ) from None
        model = model.to(self.device)
        model.eval()
        self._refs = (tokenizer, model)
        return self._refs

    def release(self):
        self._refs = None
        gc.collect()

    def generate(self, prompt, timeout=None):
        tokenizer, model = self._load()
        duration = int(timeout or self.timeout_seconds)
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            if self.device.startswith("cuda"):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            kwargs = {"max_new_tokens": self.max_new_tokens}
            if self.temperature > 0.0:
                kwargs.update(do_sample=True, temperature=self.temperature)
            else:
                kwargs["do_sample"] = False
            outputs = model.generate(**inputs, **kwargs)
            del inputs
            gc.collect()
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except MemoryError:
            raise
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMFailure(f"LLM inference failed: {exc}") from None
        return response.strip()


class MockLLMProvider:
    """Deterministic provider for automated tests / offline smoke runs."""

    name = "mock"

    def __init__(self, cfg, response=None, raise_on_generate=None):
        self.model = str(getattr(cfg.llm, "model", "") or "mock-model")
        self.response = response
        if self.response is None:
            self.response = json.dumps({
                "segments": [{
                    "text": "The night air is heavy. Every step forward feels "
                            "like a year, and the silence does not last long.",
                    "panel_ids": ["p001_001"],
                    "estimated_seconds": 4.0,
                    "visual_intent": "full_panel",
                    "camera": "static",
                    "importance": 0.7,
                }]
            })
        self.raise_on_generate = raise_on_generate
        self.last_prompt = None

    @staticmethod
    def available():
        return True

    def release(self):
        pass

    def generate(self, prompt, timeout=None):
        self.last_prompt = prompt
        if self.raise_on_generate is not None:
            raise self.raise_on_generate
        return str(self.response)


PROVIDERS = {
    "local": LocalLLMProvider,
    "mock": MockLLMProvider,
}


def create_llm_provider(cfg, response=None, raise_on_generate=None):
    """Resolve the configured LLM provider; raises LLMProviderError."""
    name = str(getattr(cfg.llm, "provider", "") or "local").lower()
    if name not in PROVIDERS:
        raise LLMProviderError(
            f"unknown llm.provider {name!r} (expected one of {', '.join(PROVIDERS)})"
        )
    if not bool(cfg.llm.enabled):
        raise LLMNotConfigured(
            "LLM is disabled (llm.enabled=false in config). Set enabled=true and "
            "configure llm.model to generate narration."
        )
    cls = PROVIDERS[name]
    if name == "mock":
        return MockLLMProvider(cfg, response=response, raise_on_generate=raise_on_generate)
    if not str(cfg.llm.model or "").strip():
        raise LLMNotConfigured(
            "LLM model not configured: set llm.model in config/config.yaml "
            "(provide a local model path or a model already present in the "
            "transformers cache - models are never downloaded automatically)"
        )
    return LocalLLMProvider(cfg)