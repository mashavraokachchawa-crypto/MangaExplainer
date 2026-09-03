"""OmniRoute adapter — route VLM/LLM calls through the OmniRoute gateway.

OmniRoute runs as a standalone AI gateway at ``http://localhost:20128`` (its
own server, SQLite-backed). MangaExplainer sends OpenAI-compatible
``/v1/chat/completions`` requests and OmniRoute handles provider selection,
API-key rotation, failover, retries, and circuit breakers across its pool.

This module provides:

  * :class:`OmniRouteVLMProvider` — vision (image + text) analysis.
    Same contract as the other VLM providers in ``pipeline/vlm_provider.py``
    (``analyze_image(image_path, prompt) -> text``). Sends the panel image as
    an OpenAI ``image_url`` (base64 data URI) with a JSON-instructed prompt.

  * :class:`OmniRouteLLMProvider` — text generation.
    Same contract as the other LLM providers in ``pipeline/llm_provider.py``
    (``generate(prompt) -> text``).

Both are pure HTTP (no local model loads), so they keep the 4 GB box's RAM
free — same benefit as the ollama/gemini providers but with OmniRoute's
routing/failover on top.

Configuration (all under ``vlm`` / ``llm`` in config.yaml)::

    vlm:
      provider: omniroute
      omniroute_url: http://127.0.0.1:20128
      omniroute_model: ""            # optional pin; default auto-routing
      omniroute_api_key: ""          # OmniRoute access key if it enforces one
      omniroute_task: understanding  # routing hint; see openai
    llm:
      provider: omniroute
      omniroute_url: http://127.0.0.1:20128
      omniroute_model: ""            # optional pin; default auto-routing
      omniroute_api_key: ""

Security: the API key (if any) is read from config (masked in the UI) or the
``OMNIROUTE_API_KEY`` env var; it is never logged.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path

from .vlm_provider import extract_json

LOG = logging.getLogger("mangaexplainer.omniroute")

DEFAULT_URL = "http://127.0.0.1:20128"

# OmniRoute-task routing hints -> group names it understands on the model
# string (used only when the caller passes a task and no pinned model).


def _base_url(cfg, section: str) -> str:
    return str(
        getattr(cfg, section).get("omniroute_url") or os.environ.get(
            "OMNIROUTE_BASE_URL", DEFAULT_URL
        )
    ).rstrip("/")


def _api_key(cfg, section: str) -> str:
    return str(
        getattr(cfg, section).get("omniroute_api_key") or ""
    ).strip() or os.environ.get("OMNIROUTE_API_KEY", "").strip()


def _model(cfg, section: str, task: str = "") -> str:
    m = str(getattr(cfg, section).get("omniroute_model") or "").strip()
    if m:
        return m
    return ""  # let OmniRoute auto-select


class OmniRouteError(Exception):
    pass


class OmniRouteUnavailable(OmniRouteError):
    pass


class OmniRouteTimeout(OmniRouteError):
    pass


class OmniRouteFailure(OmniRouteError):
    pass


def _headers(cfg, section: str) -> dict:
    headers = {"Content-Type": "application/json"}
    key = _api_key(cfg, section)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _post(base_url: str, api_key: str, body: dict, timeout: int) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    url = base_url.rstrip("/") + "/v1/chat/completions"
    try:
        request = urllib.request.Request(
            url, data=data, method="POST", headers=headers
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return _parse_body(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("error", {}).get("message", "")
            if not detail:
                detail = json.loads(exc.read().decode("utf-8")).get("error", "")
        except Exception:
            pass
        if exc.code == 401 or exc.code == 403:
            raise OmniRouteFailure(
                f"OmniRoute auth {exc.code}: {detail or exc.reason}"
            ) from None
        raise OmniRouteFailure(
            f"OmniRoute API error {exc.code}: {detail or exc.reason}"
        ) from None
    except urllib.error.URLError as exc:
        raise OmniRouteUnavailable(
            f"cannot reach OmniRoute at {base_url}: {exc.reason}. "
            "Start it with `omniroute serve`."
        ) from None
    except (TimeoutError, ConnectionError, OSError) as exc:
        raise OmniRouteTimeout(f"OmniRoute request timed out or failed: {exc}") from None
    except Exception as exc:
        raise OmniRouteFailure(f"OmniRoute request failed: {exc}") from None


def _first_message_text(data: dict) -> str:
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise OmniRouteFailure("OmniRoute returned no usable response") from None
    text = str(text or "").strip()
    if not text:
        raise OmniRouteFailure("OmniRoute returned an empty response")
    return text


def _parse_body(raw: str) -> dict:
    """Parse an OmniRoute response body, tolerating SSE streaming.

    OmniRoute streams by default (Content-Type: text/event-stream), returning
    lines like ``data: {"object":"chat.completion.chunk",...}`` ended by
    ``data: [DONE]``. We request ``stream: false`` so normal JSON comes back,
    but if a provider ignores that we still collapse the stream into the final
    completion so callers never see an opaque JSONDecodeError.
    """
    raw = (raw or "").strip()
    if not raw:
        raise OmniRouteFailure("OmniRoute returned an empty body")
    if raw.lstrip().startswith("{"):
        try:
            return json.loads(raw)
        except ValueError:
            raise OmniRouteFailure("OmniRoute returned malformed JSON") from None
    # SSE stream: collect each chunk's content delta and finish_reason.
    text = ""
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if not data or data == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except ValueError:
            # ignore the trailing [DONE] or any non-JSON control frames
            continue
        try:
            delta = chunk["choices"][0]["delta"]
        except (KeyError, IndexError, TypeError):
            continue
        piece = delta.get("content")
        if piece:
            text += piece
    if text.strip():
        return {"choices": [{"message": {"content": text}}]}
    raise OmniRouteFailure("OmniRoute stream contained no usable text")



class OmniRouteVLMProvider:
    """Vision analysis through OmniRoute's OpenAI-compatible endpoint."""

    name = "omniroute"

    def __init__(self, cfg):
        self.base_url = _base_url(cfg, "vlm")
        self.api_key = _api_key(cfg, "vlm")
        self.model = _model(cfg, "vlm", task="understanding")
        self.max_new_tokens = int(cfg.vlm.max_new_tokens)
        self.timeout_seconds = int(cfg.vlm.timeout_seconds)
        self.max_image_size = max(64, int(cfg.vlm.max_image_size))

    @staticmethod
    def available() -> bool:
        return True

    def release(self):
        pass

    def analyze_image(self, image_path, prompt, timeout=None):
        from .vlm_provider import _load_image_small

        image = _load_image_small(image_path, self.max_image_size)
        import io as _io

        buf = _io.BytesIO()
        image.save(buf, format="JPEG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}",
                    "detail": "low",
                },
            },
            {"type": "text", "text": prompt},
        ]
        payload = {
            "model": self.model or "auto/vision",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_new_tokens,
            "temperature": 0.2,
            # OmniRoute streams by default (SSE); request a single JSON object
            # so the standard json.loads path in _post() can parse it.
            "stream": False,
        }
        duration = int(timeout or self.timeout_seconds)
        data = _post(self.base_url, self.api_key, payload, duration)
        return _first_message_text(data)


class OmniRouteLLMProvider:
    """Text narration through OmniRoute's OpenAI-compatible endpoint."""

    name = "omniroute"

    def __init__(self, cfg):
        self.base_url = _base_url(cfg, "llm")
        self.api_key = _api_key(cfg, "llm")
        self.model = _model(cfg, "llm", task="narration")
        self.max_new_tokens = int(cfg.llm.max_new_tokens)
        self.max_context = int(cfg.llm.max_context)
        self.timeout_seconds = int(cfg.llm.timeout_seconds)
        self.temperature = float(getattr(cfg.llm, "temperature", 0.0) or 0.0)

    @staticmethod
    def available() -> bool:
        return True

    def release(self):
        pass

    def generate(self, prompt, timeout=None):
        payload = {
            "model": self.model or "auto/chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            # OmniRoute streams by default (SSE); request a single JSON object
            # so the standard json.loads path in _post() can parse it.
            "stream": False,
        }
        duration = int(timeout or self.timeout_seconds)
        data = _post(self.base_url, self.api_key, payload, duration)
        return _first_message_text(data)


def create_omniroute_vlm(cfg):
    return OmniRouteVLMProvider(cfg)


def create_omniroute_llm(cfg):
    return OmniRouteLLMProvider(cfg)


def omniroute_status(cfg) -> dict:
    """Cheap health probe for the dashboard; never raises for UI health.

    OmniRoute's /v1/models eagerly enumerates every provider (slow), so we
    probe reachability against the server root instead and only report the
    model count when it happens to be fast.
    """
    import urllib.request as _ur

    base_url = _base_url(cfg, "vlm")
    try:
        request = _ur.Request(base_url.rstrip("/") + "/", method="HEAD")
        # Do not follow redirects: reaching ANY HTTP response proves the server
        # is up. A 307 to /login or 200 both mean "reachable".
        with _ur.urlopen(request, timeout=5) as resp:
            reachable = True
        return {"reachable": reachable, "base_url": base_url, "models": None}
    except _ur.HTTPError as exc:
        # HTTPError for a non-2xx still means the server answered.
        return {"reachable": True, "base_url": base_url, "models": None,
                "http": exc.code}
    except Exception as exc:
        return {"reachable": False, "base_url": base_url, "error": str(exc)}
