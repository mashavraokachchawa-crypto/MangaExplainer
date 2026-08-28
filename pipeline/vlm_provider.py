"""Modular VLM providers for one-panel manga analysis.

The application never depends on one specific VLM: create_vlm_provider(cfg)
returns whichever provider is configured, and every provider implements the
same analyze_image(image_path, prompt) -> raw text contract, so the engine
can be swapped later (local vision-language model, hosted API, ...).

Low-RAM contract:
- one panel at a time
- the local provider loads the model lazily, keeps CPU inference, constrains
  the input image to vlm.max_image_size and never auto-downloads a model
  (local_files_only=True - a clear error is raised when the path is missing).
- call provider.release() after inference to drop the model references.
"""
import gc
import json
import logging
import os
import re
from pathlib import Path

LOG = logging.getLogger("mangaexplainer")


class VLMProviderError(Exception):
    pass


class VLMNotConfigured(VLMProviderError):
    pass


class VLMUnavailable(VLMProviderError):
    pass


class VLMTimeout(VLMProviderError):
    pass


class VLMFailure(VLMProviderError):
    pass


def extract_json(text):
    """Best-effort JSON recovery from a VLM text response; None on failure."""
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = text.strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _load_image_small(path, max_size):
    try:
        from PIL import Image
    except ImportError as exc:
        raise VLMUnavailable("Pillow is required for the local VLM provider") from exc
    try:
        image = Image.open(str(path)).convert("RGB")
    except Exception as exc:
        raise VLMFailure(f"cannot read image for VLM: {path} ({exc})") from None
    width, height = image.size
    longest = max(width, height)
    if longest > max_size:
        scale = max_size / longest
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale)))
        )
    return image


class LocalVLMProvider:
    """Lightweight local vision-language model via transformers.

    Model resolution is intentionally strict: transformers is loaded with
    local_files_only=True so a download NEVER happens automatically. Point
    ``vlm.model`` at a local directory (or a previously cached model id).
    """

    name = "local"

    def __init__(self, cfg):
        self.model = str(cfg.vlm.model or "").strip()
        self.device = str(cfg.vlm.device or "cpu").lower()
        self.max_image_size = max(64, int(cfg.vlm.max_image_size))
        self.max_new_tokens = int(cfg.vlm.max_new_tokens)
        self.timeout_seconds = int(cfg.vlm.timeout_seconds)
        self._refs = None

    @staticmethod
    def available():
        return True

    def _load(self):
        if self._refs is not None:
            return self._refs
        if not self.model:
            raise VLMNotConfigured(
                "VLM model not configured: set vlm.model in config/config.yaml "
                "(provide a local model path or a model already present in the "
                "transformers cache - models are never downloaded automatically)"
            )
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:
            raise VLMUnavailable(
                "transformers is not installed; the 'local' VLM provider needs "
                "it (pip install transformers --no-deps and a lightweight model)"
            ) from exc
        if not Path(self.model).is_dir():
            raise VLMUnavailable(
                f"VLM model path not found locally: {self.model!r}. Configure "
                "vlm.model to an existing local directory (download the model "
                "separately - it is never fetched automatically)"
            )
        try:
            processor = AutoProcessor.from_pretrained(self.model, local_files_only=True)
            model = AutoModelForVision2Seq.from_pretrained(
                self.model,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
        except Exception as exc:
            raise VLMUnavailable(
                f"failed to load VLM model from {self.model!r}: {exc}"
            ) from None
        model = model.to(self.device)
        model.eval()
        self._refs = (processor, model)
        return self._refs

    def release(self):
        self._refs = None
        gc.collect()

    def analyze_image(self, image_path, prompt, timeout=None):
        processor, model = self._load()
        max_size = self.max_image_size
        image = _load_image_small(image_path, max_size)
        try:
            messages = [{"role": "user", "content": [{"image": image}, {"text": prompt}]}]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=text, images=image, return_tensors="pt")
            if self.device.startswith("cuda") and not str(getattr(inputs, "device", "cpu")).startswith("cuda"):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            del inputs
            gc.collect()
            response = processor.decode(outputs[0], skip_special_tokens=True)
        except MemoryError:
            raise
        except VLMProviderError:
            raise
        except Exception as exc:
            raise VLMFailure(f"VLM inference failed: {exc}") from None
        return response.strip()


class MockVLMProvider:
    """Deterministic provider for automated tests / offline smoke runs."""

    name = "mock"

    def __init__(self, cfg, response=None, raise_on_analyze=None):
        self.model = str(getattr(cfg.vlm, "model", "") or "mock-model")
        self.response = response
        self.raise_on_analyze = raise_on_analyze
        if self.response is None:
            self.response = {
                "characters": [
                    {
                        "name": "unknown",
                        "description": "a central figure",
                        "action": "unknown",
                        "emotion": "unknown",
                    }
                ],
                "environment": "unknown",
                "actions": ["unknown"],
                "objects": [],
                "visual_effects": [],
                "important_event": "unknown",
                "composition": "unknown",
                "story_relevance": "unknown",
                "confidence": 0.7,
            }
        self.last_prompt = None

    @staticmethod
    def available():
        return True

    def release(self):
        pass

    def analyze_image(self, image_path, prompt, timeout=None):
        self.last_prompt = prompt
        if self.raise_on_analyze is not None:
            raise self.raise_on_analyze
        if isinstance(self.response, str):
            return self.response
        return json.dumps(self.response, ensure_ascii=False)


PROVIDERS = {
    "local": LocalVLMProvider,
    "mock": MockVLMProvider,
}


def create_vlm_provider(cfg, response=None, raise_on_analyze=None):
    """Resolve the configured VLM provider; raises VLMProviderError."""
    name = str(getattr(cfg.vlm, "provider", "") or "local").lower()
    if name not in PROVIDERS:
        raise VLMProviderError(
            f"unknown vlm.provider {name!r} (expected one of {', '.join(PROVIDERS)})"
        )
    if not bool(cfg.vlm.enabled):
        raise VLMNotConfigured(
            "VLM is disabled (vlm.enabled=false in config). Set enabled=true and "
            "configure vlm.model to run analysis."
        )
    cls = PROVIDERS[name]
    if name == "mock":
        return MockVLMProvider(cfg, response=response, raise_on_analyze=raise_on_analyze)
    if not str(cfg.vlm.model or "").strip():
        raise VLMNotConfigured(
            "VLM model not configured: set vlm.model in config/config.yaml "
            "(provide a local model path or a model already present in the "
            "transformers cache - models are never downloaded automatically)"
        )
    return LocalVLMProvider(cfg)