"""Pocket TTS served over HTTP - the low-RAM narration path (offline-first).

The same Kyutai pocket-tts model the pipeline already uses is exposed as a
separate lightweight (uvicorn) HTTP server:

    pocket-tts serve --quantize --port <port>

The *webui* and the *pipeline* then POST each narration segment to
``/tts`` one at a time instead of loading torch inside their own process.
That keeps the dashboard + browser light on this ~3 GB machine; the model's
~1 GB working set lives in the server process only, and ``--quantize`` (int8)
shrinks it further.

What lives here:
  * ``PocketServerProvider`` - a drop-in ``TtsProvider`` that speaks HTTP.
  * ``start_server / stop_server / probe_server`` - lifecycle + health.
  * ``builtin_voices()`` - the predefined voice catalog for the UI.

The provider is used automatically when ``tts.provider`` is ``pocket_server``
(or when the pipeline runs with that configured) and stays *even over a plain
WAV upload* for reference-voice cloning (POST field ``voice_wav``).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .pocket_tts import PocketTtsError, TtsProvider, wav_duration

LOG = logging.getLogger("mangaexplainer")

DEFAULT_VOICE = "alba"

# Known predefined voices (subset; the server accepts any name in
# _ORIGINS_OF_PREDEFINED_VOICES, plus http(s)/hf:// URLs).
BUILTIN_VOICES = [
    "alba", "cosette", "marius", "javert", "jean", "anne", "anna", "vera",
    "fantine", "charles", "paul", "eponine", "azelma", "george", "mary",
    "jane", "michael", "eve", "bill_boerst", "peter_yearsley", "stuart_bell",
    "caro_davy", "giovanni", "lola", "juergen", "rafael", "peter", "george",
]

SERVER_STATE = "tts_server.json"      # under state/
SERVER_LOG = "tts_server.log"         # under state/

_SERVER_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# config resolution
# ---------------------------------------------------------------------------


def server_config(cfg):
    """Normalized tts server settings (all absent values have defaults)."""
    tts = getattr(cfg, "tts", None)
    if tts is None:
        return {"url": "", "port": 8000, "auto_start": False, "quantize": True}
    port = int(getattr(tts, "server_port", 8000) or 8000)
    url = str(getattr(tts, "server_url", "") or "").strip().rstrip("/")
    if not url:
        url = f"http://127.0.0.1:{port}"
    return {
        "url": url,
        "port": port,
        "auto_start": bool(getattr(tts, "server_auto_start", False)),
        "quantize": bool(getattr(tts, "server_quantize", True)),
    }


def _state_path(cfg):
    try:
        base = Path(cfg.pipeline.state.dir)
    except Exception:
        base = Path(getattr(cfg, "root_dir", ".")) / "state"
    base.mkdir(parents=True, exist_ok=True)
    return base / SERVER_STATE


def _log_path(cfg):
    try:
        base = Path(cfg.pipeline.state.dir)
    except Exception:
        base = Path(getattr(cfg, "root_dir", ".")) / "state"
    base.mkdir(parents=True, exist_ok=True)
    return base / SERVER_LOG


def read_server_state(cfg):
    path = _state_path(cfg)
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text("utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def write_server_state(cfg, **kw):
    doc = read_server_state(cfg)
    doc.update(kw)
    doc["updated_at"] = time.time()
    try:
        tmp = _state_path(cfg).with_name(_state_path(cfg).name + ".tmp")
        tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        os.replace(tmp, _state_path(cfg))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# HTTP client + provider
# ---------------------------------------------------------------------------


def _urlopen(url, data=None, timeout=60.0, headers=None):
    import urllib.request
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    return urllib.request.urlopen(req, timeout=timeout)


def probe_server(cfg, timeout=2.0):
    """Quick /health check of the configured Pocket TTS server."""
    info = server_config(cfg)
    start = time.time()
    try:
        with _urlopen(info["url"] + "/health", timeout=timeout) as resp:
            if resp.status != 200:
                return {"reachable": False, "url": info["url"],
                        "status": resp.status, "error": "non-200 from /health"}
            try:
                json.loads(resp.read().decode("utf-8"))
            except Exception:
                pass
            return {"reachable": True, "url": info["url"],
                    "latency_ms": round((time.time() - start) * 1000)}
    except Exception as exc:
        return {"reachable": False, "url": info["url"], "error": str(exc)}


def server_voices():
    """Built-in Pocket TTS voice names for the voice picker (best-effort)."""
    try:
        from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES
        names = list(_ORIGINS_OF_PREDEFINED_VOICES)
        if names:
            return names
    except Exception:
        pass
    return list(BUILTIN_VOICES)


class PocketServerProvider(TtsProvider):
    """TTS provider that POSTs segments to a running pocket-tts ``serve``."""

    name = "pocket_server"

    def __init__(self, cfg, voice=None, reference=None):
        super().__init__(cfg)
        info = server_config(cfg)
        if not info["url"]:
            raise PocketTtsError("tts.server_url is empty - cannot reach server")
        self.base_url = info["url"]
        tts = getattr(cfg, "tts", None)
        self.voice = voice or str(
            getattr(tts, "default_voice", "") or DEFAULT_VOICE)
        self._reference = reference  # resolved lazily from cfg when None
        self.timeout = float(
            getattr(tts, "timeout_seconds", 300) or 300)
        self._conditioning = None  # "builtin" | "reference"
        self._conditioning_unavailable = None  # set when a ref could not be used

    @property
    def sample_rate(self):
        # pocket-tts serves 24 kHz PCM WAV.
        return 24000

    @staticmethod
    def available():
        # Availability is decided by whether a server responds, not by the
        # client package being installed - the model runs server-side.
        return True

    @staticmethod
    def supports_reference_conditioning():
        return True

    @property
    def reference(self):
        if self._reference is not None:
            return Path(self._reference) if self._reference else None
        ref = self.reference_audio
        # ignore pathological files (e.g. a stub upload) - use built-in voice.
        if ref and ref.is_file() and ref.stat().st_size > 1024:
            return ref
        self._conditioning_unavailable = (self._conditioning_unavailable or
            "reference voice missing or too small - using built-in voice")
        return None

    @property
    def conditioning(self):
        ref = self.reference
        if ref is not None:
            self._conditioning = "reference"
        else:
            self._conditioning = "builtin"
        return self._conditioning

    def synth(self, text, out_path, target_seconds=None, speaker=None):
        text = str(text or "").strip()
        if not text:
            raise PocketTtsError("Cannot synthesize empty narration text.")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ref = self.reference
        if ref is not None:
            fields, files = {}, {"voice_wav": (ref.name, ref.read_bytes())}
            self._conditioning = "reference"
        else:
            fields, files = {"voice_url": self.voice}, {}
            self._conditioning = "builtin"
        fields["text"] = text
        boundary = "----MangaExplainer" + uuid.uuid4().hex
        body = _multipart_body(fields, files, boundary)

        import urllib.request
        req = urllib.request.Request(
            self.base_url + "/tts", data=body, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status != 200:
                    raise PocketTtsError(
                        f"Pocket TTS server returned HTTP {resp.status}")
                audio = resp.read()
        except PocketTtsError:
            raise
        except Exception as exc:
            raise PocketTtsError(
                f"Pocket TTS server request failed: {exc}") from None

        if not audio or len(audio) < 44:
            raise PocketTtsError("Pocket TTS server returned empty audio.")
        out_path.write_bytes(audio)
        duration = wav_duration(out_path)
        if duration <= 0:
            raise PocketTtsError(
                f"server produced an unreadable WAV at {out_path}")
        return duration

    def release(self):
        self._model = None  # placeholder; server holds the model


def _multipart_body(fields, files, boundary):
    """Minimal multipart/form-data body (no external deps)."""
    lines = []
    for name, value in fields.items():
        lines.append(f"--{boundary}\r\n".encode("utf-8"))
        lines.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        lines.append(str(value).encode("utf-8") + b"\r\n")
    for name, (filename, content) in files.items():
        lines.append(f"--{boundary}\r\n".encode("utf-8"))
        lines.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{Path(filename).name}"\r\n'.encode("utf-8"))
        lines.append(b"Content-Type: application/octet-stream\r\n\r\n")
        lines.append(content + b"\r\n")
    lines.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(lines)


# ---------------------------------------------------------------------------
# server lifecycle
# ---------------------------------------------------------------------------


def start_server(cfg, wait_timeout=200.0):
    """Make sure a Pocket TTS server is running; auto-start it if configured.

    Returns {"ok": True, "url": ..., "started": bool, "detail": str} or an
    error dict. Safe to call from a webui request thread; a module lock keeps
    two threads from racing a double launch.
    """
    info = server_config(cfg)
    alive = probe_server(cfg, timeout=1.5)
    if alive["reachable"]:
        return {"ok": True, "url": info["url"], "started": False,
                "detail": "server already running"}

    if not info["auto_start"]:
        return {"ok": False, "url": info["url"], "started": False,
                "error": ("no Pocket TTS server on " + info["url"] +
                          " (tts.server_auto_start=false); start it with: "
                          "pocket-tts serve --quantize")}

    with _SERVER_LOCK:
        # re-check under the lock in case another thread just started it
        if probe_server(cfg, timeout=1.5)["reachable"]:
            return {"ok": True, "url": info["url"], "started": False,
                    "detail": "server already running"}

        executable = shutil.which("pocket-tts")
        if not executable:
            # guaranteed-consistent fallback: python -m pocket_tts
            executable = sys.executable
            argv = [sys.executable, "-m", "pocket_tts", "serve"]
        else:
            argv = [executable, "serve"]
        argv += ["--host", "127.0.0.1", "--port", str(info["port"])]
        if info["quantize"]:
            argv.append("--quantize")

        LOG.info("starting pocket-tts serve: %s", " ".join(argv))
        try:
            logf = open(_log_path(cfg), "ab")
            proc = subprocess.Popen(
                argv, stdout=logf, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        except Exception as exc:
            return {"ok": False, "url": info["url"], "started": False,
                    "error": f"could not start pocket-tts server: {exc}"}

        write_server_state(
            cfg, url=info["url"], pid=proc.pid, port=info["port"],
            cmd=" ".join(argv), auto_start=info["auto_start"],
            quantize=info["quantize"], status="starting")

        waited = 0.0
        step = 3.0
        while waited < wait_timeout:
            if proc.poll() is not None and probe_server(cfg, timeout=1.0)["reachable"]:
                break  # server detached (setsid) but alive
            if probe_server(cfg, timeout=1.0).get("reachable"):
                write_server_state(cfg, status="running")
                return {"ok": True, "url": info["url"], "started": True,
                        "detail": (f"started pocket-tts serve on {info['url']} "
                                   f"(quantize={'on' if info['quantize'] else 'off'})")}
            if proc.poll() is not None:
                break
            time.sleep(step)
            waited += step

        if proc.poll() is None:
            # still booting (model load) - leave it on disk, report honestly
            write_server_state(cfg, status="booting")
            return {"ok": False, "url": info["url"], "started": True,
                    "error": ("pocket-tts serve is still loading (weights take a "
                              "minute); it will be ready shortly. Try again "
                              "in a few seconds.")}
        write_server_state(cfg, status="exited")
        tail = _log_tail(cfg, n=4)
        return {"ok": False, "url": info["url"], "started": True,
                "error": "pocket-tts server exited on startup." + ("" if not tail else f" Last log:\n{tail}")}


def stop_server(cfg):
    """Stop the server recorded in state (by PID). Idempotent."""
    state = read_server_state(cfg)
    pid = state.get("pid")
    if not pid:
        return {"ok": True, "stopped": False, "detail": "no server recorded"}
    try:
        import signal
        os.kill(int(pid), signal.SIGTERM)
        write_server_state(cfg, status="stopped")
        return {"ok": True, "stopped": True, "detail": f"sent SIGTERM to pid {pid}"}
    except ProcessLookupError:
        write_server_state(cfg, status="stopped")
        return {"ok": True, "stopped": False, "detail": f"pid {pid} already gone"}
    except Exception as exc:
        return {"ok": False, "stopped": False, "error": str(exc)}


def server_info(cfg):
    """One-shot status block for the dashboard (used by /api/status)."""
    info = server_config(cfg)
    probe = probe_server(cfg, timeout=1.5)
    state = read_server_state(cfg)
    return {
        "url": info["url"],
        "configured": bool(info["url"]),
        "auto_start": info["auto_start"],
        "quantize": info["quantize"],
        "reachable": probe["reachable"],
        "latency_ms": probe.get("latency_ms"),
        "error": probe.get("error"),
        "pid": state.get("pid"),
        "state_status": state.get("status") or ("running" if probe["reachable"] else "stopped"),
        "voices": server_voices(),
        "default_voice": str(getattr(getattr(cfg, "tts", None), "default_voice", "") or DEFAULT_VOICE),
        "log_url": "/media/tts_server_log" if _log_path(cfg).is_file() else None,
    }


def _log_tail(cfg, n=4):
    path = _log_path(cfg)
    if not path.is_file():
        return ""
    try:
        with open(path, "rb") as handle:
            lines = handle.read().decode("utf-8", "replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""