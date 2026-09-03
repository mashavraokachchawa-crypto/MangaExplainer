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
import urllib.error
import urllib.request
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


def _backoff(seconds):
    """Sleep for `seconds` unless pytest forces an instant retry loop."""
    import time

    if os.environ.get("VLM_FAST_RETRY", "") == "1":
        return
    time.sleep(max(0.0, seconds))


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
            import transformers as _tf

            model_cls = getattr(
                _tf, "AutoModelForImageTextToText", None
            ) or getattr(_tf, "AutoModelForVision2Seq", None)
            if model_cls is None:
                raise ImportError(
                    "no vision model class "
                    "(AutoModelForImageTextToText/AutoModelForVision2Seq) "
                    "in transformers"
                )
            from transformers import AutoProcessor
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
            model = model_cls.from_pretrained(
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
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": prompt}],
                }
            ]
            # transformers 5.x renamed the old ProcessorMixin.apply_chat_template
            # path but kept it working; fall back to the tokenizer if the
            # processor no longer exposes it.
            template_fn = getattr(processor, "apply_chat_template", None)
            if template_fn is None and getattr(processor, "tokenizer", None) is not None:
                template_fn = processor.tokenizer.apply_chat_template
            if template_fn is None:
                raise VLMFailure(
                    "no apply_chat_template found on the processor; the model "
                    "is not a chat-style vision-language model"
                )
            text = template_fn(messages, add_generation_prompt=True)
            inputs = processor(text=text, images=[image], return_tensors="pt")
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


class OllamaVLMProvider:
    """Vision-language panel analysis served by a local Ollama server.

    Ollama runs the vision model in its OWN process (llama.cpp), so the
    weights never enter the pipeline process - a vision model cannot OOM or
    crash the render pipeline here. Requires a running ``ollama serve`` with
    a vision model already pulled (``ollama pull moondream``).
    """

    name = "ollama"

    def __init__(self, cfg):
        import base64

        self._b64 = base64
        self.base_url = str(
            cfg.vlm.get("ollama_url") or "http://127.0.0.1:11434"
        ).rstrip("/")
        self.model = str(cfg.vlm.get("ollama_model") or "").strip()
        self.max_new_tokens = int(cfg.vlm.max_new_tokens)
        self.timeout_seconds = int(cfg.vlm.timeout_seconds)

    @staticmethod
    def available():
        return True

    def release(self):
        pass

    def analyze_image(self, image_path, prompt, timeout=None):
        if not self.model:
            raise VLMNotConfigured(
                "Ollama VLM model not configured: set vlm.ollama_model in "
                "config/config.yaml (e.g. moondream) and run `ollama pull`"
            )
        try:
            with open(str(image_path), "rb") as handle:
                image_b64 = self._b64.b64encode(handle.read()).decode("ascii")
        except OSError as exc:
            raise VLMFailure(f"cannot read image for VLM: {exc}") from None
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            "stream": False,
            "format": "json",
            "options": {"num_predict": self.max_new_tokens},
        }
        body = json.dumps(payload).encode("utf-8")
        duration = int(timeout or self.timeout_seconds)
        attempt = 0
        while True:
            try:
                request = urllib.request.Request(
                    self.base_url + "/api/chat",
                    data=body,
                    method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=duration) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = json.loads(exc.read().decode("utf-8")).get("error")
                except Exception:
                    pass
                if (
                    attempt == 0
                    and exc.code == 400
                    and (
                        "format" in (detail or "")
                        or "structured" in (detail or "")
                    )
                ):
                    payload.pop("format", None)
                    body = json.dumps(payload).encode("utf-8")
                    attempt = 1
                    continue
                message = detail or exc.reason
                raise VLMFailure(
                    f"ollama VLM API error {exc.code}: {message}"
                ) from None
            except urllib.error.URLError as exc:
                raise VLMUnavailable(
                    f"cannot reach Ollama server at {self.base_url}: {exc.reason}."
                    " Start it with `ollama serve`."
                ) from None
            except (TimeoutError, ConnectionError, OSError) as exc:
                raise VLMTimeout(
                    f"ollama VLM request timed out or failed: {exc}"
                ) from None
            except Exception as exc:
                raise VLMFailure(f"ollama VLM request failed: {exc}") from None
        if not isinstance(data, dict):
            raise VLMFailure("ollama returned a non-JSON response")
        if data.get("error"):
            raise VLMFailure(f"ollama VLM error: {data['error']}")
        message = data.get("message") or {}
        response = str(message.get("content") or "").strip()
        if not response:
            raise VLMFailure("ollama VLM returned an empty response")
        return response


class GeminiVLMProvider:
    """Hosted Google Gemini vision API for one-panel manga analysis.

    Keeps this box's 4 GB RAM free (no local model loads) and is fast enough
    for the 1445-panel corpus. Uses the public ``generativelanguage`` API key
    endpoint; the key is read from the ``GEMINI_API_KEY`` environment variable
    or ``vlm.api_key`` in config.

    A real API key must be present: we never silently fall back to plain text
    or mark analysis "unknown" - that is exactly the fake output this provider
    exists to replace.
    """

    name = "gemini"

    BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, cfg):
        import base64

        self._b64 = base64
        self.model = str(cfg.vlm.get("gemini_model") or "gemini-flash-lite-latest").strip()
        raw_keys = (
            str(cfg.vlm.get("api_key") or "").strip()
            or os.environ.get("GEMINI_API_KEY", "").strip()
        )
        # Support multiple keys (comma separated) so a rate-limited or spent
        # key is rotated past and work never stops.
        self.api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()] if raw_keys else []
        self._key_i = 0
        self.api_key = self.api_keys[0] if self.api_keys else ""
        self.max_new_tokens = int(cfg.vlm.max_new_tokens)
        self.timeout_seconds = int(cfg.vlm.timeout_seconds)
        self.max_image_size = max(64, int(cfg.vlm.max_image_size))
        self.max_retries = max(1, int(cfg.vlm.get("gemini_retries") or 3))

    @staticmethod
    def available():
        return bool(os.environ.get("GEMINI_API_KEY", "").strip())

    def release(self):
        pass

    def _next_key(self):
        """Rotate to the next API key (circular). Returns the key string."""
        if not self.api_keys:
            return ""
        self._key_i = (self._key_i + 1) % len(self.api_keys)
        return self.api_keys[self._key_i]

    def analyze_image(self, image_path, prompt, timeout=None):
        if not self.api_keys:
            raise VLMNotConfigured(
                "Gemini VLM provider needs an API key. Set the GEMINI_API_KEY "
                "environment variable (or vlm.api_key in config/config.yaml). "
                "A free key from aistudio.google.com/apikey is sufficient."
            )
        image = _load_image_small(image_path, self.max_image_size)
        if image.mode != "RGB":
            image = image.convert("RGB")
        import io as _io

        buf = _io.BytesIO()
        image.save(buf, format="PNG")
        image_b64 = self._b64.b64encode(buf.getvalue()).decode("ascii")

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": image_b64,
                            }
                        },
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": self.max_new_tokens,
                "responseMimeType": "application/json",
            },
        }
        body = json.dumps(payload).encode("utf-8")
        duration = int(timeout or self.timeout_seconds)

        # Try each key; for transient errors (429 rate limit, 503 overload,
        # 5xx) rotate to the next key, then retry with backoff so a single
        # saturated key never stops the whole corpus.
        keys = list(self.api_keys)
        start_at = self._key_i if 0 <= self._key_i < len(keys) else 0
        ordered = keys[start_at:] + keys[:start_at]

        last_exc = None
        for ki, key in enumerate(ordered):
            self._key_i = keys.index(key)
            self.api_key = key
            url = f"{self.BASE}/{self.model}:generateContent?key={key}"
            for attempt in range(self.max_retries):
                try:
                    request = urllib.request.Request(
                        url, data=body, method="POST",
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(request, timeout=duration) as response:
                        data = json.loads(response.read().decode("utf-8"))
                    text = self._extract_text(data)
                    return text
                except urllib.error.HTTPError as exc:
                    detail = ""
                    try:
                        detail = json.loads(exc.read().decode("utf-8")).get("error", {}).get("message", "")
                    except Exception:
                        pass
                    code = exc.code
                    msg = detail or exc.reason
                    transient = code in (429, 500, 502, 503, 504)
                    bad_key = (
                        code in (401, 403)
                        or (code == 400 and "key" in (msg or "").lower())
                    )
                    if transient:
                        last_exc = VLMFailure(
                            f"Gemini API error {code}: {msg}"
                        )
                        if attempt < self.max_retries - 1:
                            _backoff(0.5 * (2 ** attempt))
                            continue
                        break  # exhausted retries for this key -> try next key
                    if bad_key:
                        # This key is invalid/spent: move to the next key
                        # immediately (work never stops because one key died).
                        last_exc = VLMFailure(f"Gemini API error {code}: {msg}")
                        break  # try next key
                    # Terminal error for this request (e.g. model not found) is
                    # the same for every key - raise immediately.
                    raise VLMFailure(f"Gemini API error {code}: {msg}") from None
                except urllib.error.URLError as exc:
                    last_exc = VLMUnavailable(f"cannot reach Gemini API: {exc.reason}")
                    if attempt < self.max_retries - 1:
                        _backoff(0.5 * (2 ** attempt))
                        continue
                    break
                except (TimeoutError, ConnectionError, OSError) as exc:
                    last_exc = VLMTimeout(f"Gemini request timed out or failed: {exc}")
                    if attempt < self.max_retries - 1:
                        _backoff(0.5 * (2 ** attempt))
                        continue
                    break
                except Exception as exc:
                    raise VLMFailure(f"Gemini request failed: {exc}") from None
        if last_exc is not None:
            raise last_exc
        raise VLMFailure("Gemini request failed (no keys retried)")

    def _extract_text(self, data):
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            try:
                reason = data.get("promptFeedback", {}).get("blockReason")
            except Exception:
                reason = None
            raise VLMFailure(
                "Gemini returned no usable response"
                + (f" (blocked: {reason})" if reason else "")
            )
        text = str(text or "").strip()
        if not text:
            raise VLMFailure("Gemini returned an empty response")
        return text


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


class FallbackVLMProvider:
    """Primary provider with a configured fallback.

    ``analyze_image`` tries the primary first and retries once on the fallback
    when the primary is unavailable, unconfigured, or fails (e.g. local
    SmolVLM unavailable -> Ollama gets the same panel). After a fallback the
    wrapper adopts the fallback's name/model so logs and analysis metadata
    report which provider actually ran.
    """

    name = "fallback"

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
        self.name = primary.name
        self.model = getattr(primary, "model", "") or ""

    def analyze_image(self, image_path, prompt, timeout=None):
        try:
            return self.primary.analyze_image(image_path, prompt, timeout=timeout)
        except MemoryError:
            raise
        except VLMProviderError as exc:
            LOG.warning(
                "primary VLM %s failed (%s) - falling back to %s",
                self.primary.name, exc, self.fallback.name,
            )
            response = self.fallback.analyze_image(
                image_path, prompt, timeout=timeout
            )
            self.name = self.fallback.name
            self.model = getattr(self.fallback, "model", "") or ""
            return response

    def release(self):
        self.primary.release()
        self.fallback.release()


PROVIDERS = {
    "local": LocalVLMProvider,
    "ollama": OllamaVLMProvider,
    "gemini": GeminiVLMProvider,
    "omniroute": "omniroute",  # resolves lazily to avoid a heavy import cycle
    "mock": MockVLMProvider,
}


def _make(name, cfg, response=None, raise_on_analyze=None):
    """Build a single provider (model presence is validated lazily at load)."""
    if name == "mock":
        return MockVLMProvider(cfg, response=response, raise_on_analyze=raise_on_analyze)
    if name == "ollama":
        return OllamaVLMProvider(cfg)
    if name == "gemini":
        return GeminiVLMProvider(cfg)
    if name == "omniroute":
        from .omniroute_provider import create_omniroute_vlm

        return create_omniroute_vlm(cfg)
    return LocalVLMProvider(cfg)


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
    fallback_name = str(getattr(cfg.vlm, "fallback", "") or "").strip().lower()
    if fallback_name not in ("", "omniroute") and fallback_name not in PROVIDERS:
        raise VLMProviderError(
            f"unknown vlm.fallback {fallback_name!r} "
            f"(expected one of {', '.join(PROVIDERS)})"
        )
    primary = _make(name, cfg, response=response, raise_on_analyze=raise_on_analyze)
    if name != "mock" and fallback_name and fallback_name != name:
        return FallbackVLMProvider(
            primary,
            _make(fallback_name, cfg, response=response, raise_on_analyze=raise_on_analyze),
        )
    if name == "local" and not str(cfg.vlm.model or "").strip():
        raise VLMNotConfigured(
            "VLM model not configured: set vlm.model in config/config.yaml "
            "(provide a local model path or a model already present in the "
            "transformers cache - models are never downloaded automatically)"
        )
    return primary