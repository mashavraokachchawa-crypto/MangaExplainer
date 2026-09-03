"""MangaExplainer web UI (Task 31).

A zero-dependency local web dashboard built only on the Python standard
library (http.server). It wraps the existing CLI (main.main) so every
pipeline action — full run, resume, per-stage tools, PDF selection, quality
check, clean, and the voice demos — is one click in the browser.

It never adds subtitles and never adds new AI functionality; it is a thin,
user-friendly layer over the existing pipeline.

Run:
    python webui.py [--host 127.0.0.1] [--port 8000] [--config config/config.yaml]
"""
from __future__ import annotations

import argparse
import io
import ipaddress
import json
import logging
import math
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request
import wave
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

LOG = logging.getLogger("mangaexplainer.webui")

_CFG_PATH = None


# ---------------------------------------------------------------------------
# In-process command runner (one job at a time, non-blocking the UI)
# ---------------------------------------------------------------------------


class Job:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.action = None
        self.output = []
        self.exit_code = None
        self.started = None

    def snapshot(self):
        with self.lock:
            elapsed = None
            if self.running and self.started is not None:
                elapsed = round(time.time() - self.started, 1)
            return {
                "running": self.running,
                "action": self.action,
                "exit_code": self.exit_code,
                "output": list(self.output[-80:]),
                "started": self.started,
                "elapsed": elapsed,
            }


JOB = Job()


def _emit(line):
    with JOB.lock:
        JOB.output.append(str(line).rstrip())


def _run_cli(argv, label):
    """Run a CLI command in-process, capturing stdout+stderr."""
    with JOB.lock:
        JOB.running = True
        JOB.action = label
        JOB.output = []
        JOB.exit_code = None
        JOB.started = time.time()
    try:
        import main as cli
        orig_out, orig_err = sys.stdout, sys.stderr
        buf = io.StringIO()
        sys.stdout = buf
        sys.stderr = buf
        try:
            code = cli.main(argv)
        finally:
            sys.stdout = orig_out
            sys.stderr = orig_err
        for chunk in buf.getvalue().splitlines():
            _emit(chunk)
        with JOB.lock:
            JOB.exit_code = code
    except Exception as exc:  # surface unexpected errors in the UI
        _emit(f"[error] {type(exc).__name__}: {exc}")
        with JOB.lock:
            JOB.exit_code = 1
    finally:
        with JOB.lock:
            JOB.running = False


def _start_job(argv, label):
    if JOB.running:
        return False, "A command is already running. Wait for it to finish."
    threading.Thread(target=_run_cli, args=(argv, label), daemon=True).start()
    return True, label


# ---------------------------------------------------------------------------
# Voice demo helpers (sample line + narration playback)
# ---------------------------------------------------------------------------

SAMPLE_TEXT = "This is a voice demo from your MangaExplainer. Enjoy the story."


def _sample_audio_path(cfg):
    return Path(cfg.output.audio_dir) / "voice_demo.wav"


def _mock_tone_provider(cfg):
    from pipeline.tts_provider import MockTtsProvider
    return MockTtsProvider(cfg)


def gen_voice_sample(cfg):
    """Synthesize a short sample line using the configured TTS provider.

    Uses the real provider when available (espeak-ng) and the deterministic
    mock provider otherwise, so the demo always produces playable audio.
    """
    from pipeline.tts_provider import create_tts_provider, TtsNotConfigured
    out = _sample_audio_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    provider = None
    try:
        provider = create_tts_provider(cfg)
    except TtsNotConfigured:
        provider = _mock_tone_provider(cfg)
    try:
        duration = provider.synth(SAMPLE_TEXT, str(out), target_seconds=3.0,
                                  speaker="demo")
    except Exception:
        # Last-resort mock tone so the demo always plays.
        provider = _mock_tone_provider(cfg)
        duration = provider.synth(SAMPLE_TEXT, str(out), target_seconds=3.0,
                                  speaker="demo")
    return {
        "path": str(out),
        "engine": provider.name,
        "duration": round(duration, 2),
        "text": SAMPLE_TEXT,
    }


def _measure_wav(path):
    try:
        with wave.open(str(path), "rb") as handle:
            return round(handle.getnframes() / (handle.getframerate() or 1), 2)
    except Exception:
        return 0.0


def narration_audio(cfg):
    """Point the UI at narration audio: final mix, else the segment files."""
    out = Path(cfg.output.audio_dir)
    candidates = {}
    if (out / "final_mix.wav").exists():
        candidates["final_mix.wav"] = _measure_wav(out / "final_mix.wav")
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text("utf-8"))
        except Exception:
            manifest = []
        for entry in manifest:
            # Manifest entries carry the real audio filename (audio_path);
            # fall back to {segment_id}.wav for older runs.
            apath = Path(entry.get("audio_path") or "")
            seg = apath if apath.is_file() else out / f"{entry.get('segment_id', '')}.wav"
            if seg.is_file():
                candidates[seg.name] = _measure_wav(seg)
    return candidates


# ---------------------------------------------------------------------------
# Music / audio helpers (uploadable background track for the audio-mix stage)
# ---------------------------------------------------------------------------

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a"}
IMG_EXTS = {".png", ".jpg", ".jpeg"}


def _gen_tone_wav(path, seconds=8.0, sr=22050, freq=220.0, volume=0.30):
    """Write a small soft placeholder tone so the mix stage has a track."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sr)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = bytearray()
        for i in range(n):
            t = i / sr
            env = max(0.0, min(1.0, i / (0.5 * sr), (n - i) / (0.5 * sr)))
            v = int(32767 * volume * env * math.sin(2 * math.pi * freq * t))
            frames += struct.pack("<h", v)
        w.writeframes(bytes(frames))
    return seconds


def _music_settings(cfg):
    from pipeline.music import music_config
    try:
        return music_config(cfg)
    except Exception:
        return None


def _music_track_path(cfg):
    """Path where the upload / mix expects the background track file."""
    settings = _music_settings(cfg) or {}
    music_dir = ROOT / settings.get("dir", "music")
    configured = settings.get("file") or "track.wav"
    p = Path(configured)
    if not p.is_absolute():
        p = music_dir / p
    return p


def _ensure_music_track(cfg):
    """Make sure a track exists for the mix stage; generate a placeholder if not."""
    path = _music_track_path(cfg)
    if path.is_file():
        return path, True
    _gen_tone_wav(path)
    return path, False


def _save_uploaded_audio(cfg, data, filename):
    """Save an uploaded audio file into the music dir.

    Keeps the user's filename (validated audio extension) so the mixer's
    scan picks it up; otherwise writes the configured track path.
    """
    settings = _music_settings(cfg) or {}
    music_dir = ROOT / settings.get("dir", "music")
    music_dir.mkdir(parents=True, exist_ok=True)
    if filename:
        name = Path(filename).name
        if name and Path(name).suffix.lower() in AUDIO_EXTS:
            dest = music_dir / name
            dest.write_bytes(data)
            return dest, name
    configured = settings.get("file") or "track.wav"
    dest = music_dir / configured
    dest.write_bytes(data)
    return dest, dest.name


def _voice_reference_path(cfg):
    """Where an uploaded demo voice becomes the TTS reference voice."""
    try:
        from pipeline.voice_reference import reference_audio_path
        p = reference_audio_path(cfg)
        if p:
            return Path(p)
    except Exception:
        pass
    return ROOT / "input" / "voice_reference.mp3"


def _voice_info(cfg):
    path = _voice_reference_path(cfg)
    exists = path.is_file()
    info = {"path": str(path), "exists": exists,
            "url": "/media/voice_reference" if exists else None,
            "name": path.name}
    if exists:
        info["bytes"] = path.stat().st_size
        if path.suffix.lower() == ".wav":
            info["duration"] = _measure_wav(path)
    return info


def _music_info(cfg):
    settings = _music_settings(cfg)
    enabled = settings is not None  # music_config returns None when music is off
    settings = settings or {}
    track = None
    if enabled:
        path = _music_track_path(cfg)
        if path.is_file():
            track = str(path)
    music_dir = ROOT / (settings.get("dir", "music") if settings else "music")
    files = sorted(
        p.name for p in music_dir.iterdir()
        if p.is_file() and (p.suffix.lower() in AUDIO_EXTS) and p.name[:1] != "."
    ) if music_dir.is_dir() else []
    return {
        "enabled": enabled,
        "volume": settings.get("volume"),
        "dir": settings.get("dir", "music"),
        "track": track,
        "track_exists": bool(track),
        "url": "/media/track" if track else None,
        "files": files,
        "urls": [f"/media/music_file/{n}" for n in files],
    }


# ---------------------------------------------------------------------------
# Music download / fetch (the "get a background track" ways)
# ---------------------------------------------------------------------------

# Curated public-domain / CC0 tracks (stable Wikimedia Commons uploads).
MUSIC_PRESETS = [
    {
        "key": "cloud_rap",
        "name": "Cloud rap beat (CC0)",
        "kind": "short ambient beat",
        "url": "https://upload.wikimedia.org/wikipedia/commons/2/2f/"
               "Cloud_rap_beat.ogg",
    },
    {
        "key": "chinese_ensemble",
        "name": "Chinese instrumental ensemble (public domain)",
        "kind": "instrumental ensemble",
        "url": "https://upload.wikimedia.org/wikipedia/commons/f/f7/"
               "Chinese_Vocal_and_Instrumental_Ensemble.ogg",
    },
]
MUSIC_FETCH_LIMIT = 60 * 1024 * 1024  # cap on a single downloaded track
MUSIC_FETCH_TIMEOUT = 30  # seconds
FFMPEG_HINT = "/home/madhav/bin/ffmpeg"


def _download_media(url, max_bytes, timeout=MUSIC_FETCH_TIMEOUT):
    """Download audio bytes with SSRF guards; returns (data, content_type).

    Only http(s), only public IPs, only audio content types, size-capped.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("only http:// and https:// URLs are allowed")
    try:
        raw_ip = socket.gethostbyname(parsed.hostname)
    except OSError as exc:
        raise ValueError(f"could not resolve host {parsed.hostname!r}") from exc
    try:
        addr = ipaddress.ip_address(raw_ip)
    except ValueError:
        raise ValueError(f"unresolved host {parsed.hostname!r}") from None
    if (addr.is_private or addr.is_loopback or addr.is_link_local
            or addr.is_multicast or addr.is_reserved or addr.is_unspecified):
        raise ValueError(f"refusing non-public host {parsed.hostname!r}")
    request = urllib.request.Request(url, headers={
        "User-Agent": "MangaExplainer/1.0 (+https://github.com/)",
        "Accept": "audio/*,application/ogg,application/x-wav",
    })
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if not (content_type.startswith("audio/")
                    or content_type in ("application/ogg", "application/x-wav")):
                raise ValueError(
                    f"URL did not return audio (Content-Type: {content_type or '?'})")
            declared = int(resp.headers.get("Content-Length") or 0)
            if declared > max_bytes:
                raise ValueError(
                    f"track too large ({declared} bytes, cap {max_bytes})")
            data = bytearray()
            while len(data) <= max_bytes:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                data.extend(chunk)
            if len(data) > max_bytes:
                raise ValueError(f"track too large (cap {max_bytes} bytes)")
            if not data:
                raise ValueError("download came back empty")
            return bytes(data), content_type
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(f"download failed: {exc}") from exc


def _ffmpeg_path():
    found = shutil.which("ffmpeg")
    if found:
        return found
    hint = Path(FFMPEG_HINT)
    return str(hint) if hint.is_file() else None


def _save_music_download(cfg, data, suggested_name):
    """Save fetched audio into the music dir; transcode non-WAV via ffmpeg.

    Non-WAV downloads are converted to mono 44.1 kHz WAV (the mixer only
    decodes WAV), named from the source track and returned as (path, name).
    """
    settings = _music_settings(cfg) or {}
    music_dir = ROOT / settings.get("dir", "music")
    music_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9 _.-]+", "_", Path(suggested_name).stem or "track")
    stem = (stem.strip(" .") or "track")[:60]
    src = music_dir / f".{stem}.src"
    try:
        src.write_bytes(data)
        if data[:4] == b"RIFF":
            final = music_dir / f"{stem}.wav"
            final.write_bytes(data)
            return final, final.name
        ffmpeg = _ffmpeg_path()
        if ffmpeg is None:
            raise ValueError(
                "that download is not a WAV and ffmpeg is missing, so it can't "
                "be converted — upload a WAV instead (or add ffmpeg to PATH)")
        final = music_dir / f"{stem}.wav"
        run = subprocess.run(
            [ffmpeg, "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "44100",
             str(final)],
            timeout=MUSIC_FETCH_TIMEOUT + 10, check=False,
            capture_output=True, text=True,
        )
        if run.returncode != 0 or not final.is_file():
            raise ValueError(
                f"ffmpeg could not decode that audio ({run.stderr.strip()[-200:]})")
        return final, final.name
    finally:
        try:
            src.unlink(missing_ok=True)
        except OSError:
            pass


def _music_fetch(cfg, data):
    """Resolve a preset or pasted URL, download it, and drop it in music/."""
    url = (data.get("url") or "").strip()
    preset_key = (data.get("preset") or "").strip()
    preset_name = None
    if not url and preset_key:
        for preset in MUSIC_PRESETS:
            if preset["key"] == preset_key:
                url, preset_name = preset["url"], preset["name"]
                break
    if not url:
        raise ValueError("no music URL — pick a free preset or paste a link")
    raw, content_type = _download_media(url, MUSIC_FETCH_LIMIT)
    name = url.rsplit("/", 1)[-1].split("?")[0] or "track"
    final, saved = _save_music_download(cfg, raw, Path(name).stem or "track")
    _emit(f"[music] fetched {len(raw)} bytes -> {final}")
    origin = f" ({preset_name})" if preset_name else " from URL"
    return (f"Downloaded background track{saved!r}{origin}. It will play on "
            "the next mix once music is enabled.")


# SFX (optional) helpers + live-image media routing
# ---------------------------------------------------------------------------

SFX_EXTS = {".wav", ".mp3", ".ogg"}


def _sfx_dir(cfg):
    from pipeline.sfx import sfx_config
    settings = sfx_config(cfg)
    raw = (settings or {}).get("dir") or "sfx"
    node = getattr(cfg, "sfx", None)
    if isinstance(node, dict):
        raw = (settings or {}).get("dir") or node.get("dir", "sfx")
    path = Path(raw)
    return (path.resolve() if path.is_absolute() else (ROOT / path).resolve())


def _gen_sfx_wav(path, seconds, freq, freq_end=None, volume=0.5):
    """Small SFX placeholder: a brief tone (sweeps up to freq_end if given)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sr = 22050
    n = max(int(seconds * sr), 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = bytearray()
        for i in range(n):
            t = i / sr
            env = max(0.0, min(1.0, i / (0.05 * sr), (n - i) / (0.05 * sr)))
            f = freq if freq_end is None else freq + (freq_end - freq) * (i / n)
            v = int(32767 * volume * env * math.sin(2 * math.pi * f * t))
            frames += struct.pack("<h", v)
        w.writeframes(bytes(frames))
    return str(path)


def _gen_sfx_demo(cfg):
    """Write placeholder SFX wavs + a manifest so the mix stage has effects."""
    sdir = _sfx_dir(cfg)
    sdir.mkdir(parents=True, exist_ok=True)
    whoosh = _gen_sfx_wav(sdir / "whoosh.wav", 0.45, 300, freq_end=900, volume=0.5)
    thud = _gen_sfx_wav(sdir / "thud.wav", 0.30, 85, volume=0.55)
    click = _gen_sfx_wav(sdir / "click.wav", 0.12, 1400, volume=0.4)
    events = [
        {"file": "whoosh.wav", "volume": 0.8, "duration": 0.45, "offset": 0.1},
        {"file": "thud.wav", "volume": 0.9, "duration": 0.30, "offset": 0.05},
        {"file": "click.wav", "volume": 0.7, "duration": 0.12, "offset": 0.0},
    ]
    (sdir / "sfx_manifest.json").write_text(
        json.dumps(events, indent=2), encoding="utf-8")
    return sdir


def _sfx_manifest_events(cfg):
    sdir = _sfx_dir(cfg)
    manifest = sdir / "sfx_manifest.json"
    if not manifest.is_file():
        return []
    try:
        doc = json.loads(manifest.read_text("utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return []
    if isinstance(doc, dict):
        doc = doc.get("events") or doc.get("effects") or doc.get("sfx") or []
    return doc if isinstance(doc, list) else []


def _sfx_info(cfg):
    from pipeline.sfx import sfx_config
    settings = sfx_config(cfg)
    enabled = settings is not None
    sdir = _sfx_dir(cfg)
    events = _sfx_manifest_events(cfg)
    if sdir.is_dir():
        files = sorted(
            p.name for p in sdir.iterdir()
            if p.is_file() and p.suffix.lower() in SFX_EXTS
        )
    else:
        files = []
    return {
        "enabled": enabled,
        "dir": str(sdir),
        "manifest": str(sdir / "sfx_manifest.json")
        if (sdir / "sfx_manifest.json").is_file() else None,
        "events": len(events),
        "files": files[:20],
        "urls": [f"/media/sfx/{n}" for n in files[:20]],
    }


_LIVE_DIR_KEYS = ("pages_dir", "panels_dir", "clean_dir", "ocr_dir",
                  "crops_dir", "shots_dir")


def _live_dirs(cfg):
    """basename -> absolute dir for the artifact folders the live view shows."""
    out = {}
    for key in _LIVE_DIR_KEYS:
        node = getattr(getattr(cfg, "output", None), key, None)
        if not node:
            continue
        base = Path(node).name
        if base:
            out[base] = Path(node)
    return out


def _live_image_url(cfg, rel):
    """Map a repo-relative artifact path to a safe /media/live_image/ URL."""
    if not rel:
        return None
    parts = Path(rel).parts
    if not parts:
        return None
    if parts[0] not in _live_dirs(cfg):
        return None
    if any(p in ("", "..") or p.startswith(".") or ":" in p for p in parts):
        return None
    return "/media/live_image/" + "/".join(parts)


def _memory_info(cfg):
    from pipeline.context_memory import memory_info as _mi
    info = _mi(cfg) or {}
    try:
        from pipeline.manga_memory.store import memory_info as _mmi
        info["manga"] = _mmi(cfg) or {}
    except Exception:
        info["manga"] = {}
    return info


def _manga_records(cfg):
    """Dump all Manga Memory Engine records for the UI (never raises)."""
    try:
        from pipeline.manga_memory.store import open_memory

        memory = open_memory(cfg, lazy=True).load_all()
        out = {"characters": [], "world": [], "story": [], "corrections": [], "books": []}
        for rec in memory.store_for("character").all():
            out["characters"].append(rec.to_dict())
        for rec in memory.store_for("world").all():
            out["world"].append(rec.to_dict())
        for rec in memory.store_for("story").all():
            out["story"].append(rec.to_dict())
        for rec in memory.store_for("correction").all():
            out["corrections"].append(rec.to_dict())
        for rec in memory.store_for("book").all():
            out["books"].append(rec.to_dict())
        return out
    except Exception as exc:
        return {"error": str(exc)}


def _apply_correction(cfg, data):
    """Add a user correction to the Manga Memory Engine (never raises)."""
    target = (data.get("target") or "").strip()
    correction = (data.get("correction") or "").strip()
    kind = (data.get("kind") or "fact").strip() or "fact"
    if not target or not correction:
        raise ValueError("both 'target' and 'correction' are required")
    try:
        from pipeline.manga_memory.store import open_memory
        from pipeline.manga_memory.user_corrections import UserCorrectionMemory

        memory = open_memory(cfg, lazy=True).load_all()
        uc = UserCorrectionMemory(memory.store_for("correction"))
        uc.add(target, correction, kind=kind, source="user")
        memory.save_all()
        return target, correction, kind
    except Exception:
        LOG.exception("memory correction failed")
        raise


def _delete_memory_record(cfg, data):
    """Delete a single memory record by kind + key (never raises)."""
    kind = (data.get("kind") or "").strip()
    key = (data.get("key") or "").strip()
    if not kind or not key:
        raise ValueError("both 'kind' and 'key' are required")
    try:
        from pipeline.manga_memory.store import open_memory

        memory = open_memory(cfg, lazy=True).load_all()
        store = memory.store_for(kind)
        if store is None:
            raise ValueError(f"unknown memory kind {kind!r}")
        removed = store.delete(key)
        memory.save_all()
        return key, removed
    except Exception:
        LOG.exception("memory delete failed")
        raise


def _kb_live(cfg):
    """SQLite knowledge-base dump for the Memory Explorer (never raises).

    Resolves the manga the active project / PDF points at and returns
    stats + structured characters/locations/events/chapters/summaries/
    conflicts + each entity's source evidence, so the UI can show source
    badges, appearance counts, and open verification conflicts.
    """
    try:
        from pipeline.knowledge_db import open_knowledge_db

        state_dir = _state_dir(cfg)
        db = open_knowledge_db(state_dir)
        try:
            mangas = db.list_manga()
            if not mangas:
                return {"status": "empty", "mangas": 0}

            # Resolve the target manga: active project name / pdf stem / newest
            target = None
            try:
                from pipeline import project_registry
                active = _active_project(cfg)
                if active and active.get("name"):
                    nm = str(active["name"]).strip().lower()
                    target = next((m for m in mangas
                                   if str(m.get("title", "")).strip().lower() == nm), None)
            except Exception:
                pass
            if target is None:
                pdf = Path(cfg.input.pdf).stem.strip().lower()
                target = next((m for m in mangas
                               if str(m.get("title", "")).strip().lower() == pdf), None)
            if target is None:
                target = mangas[0]

            manga_id = target["manga_id"]
            stats = db.stats(manga_id)

            # Characters with source + appearance info
            characters = []
            for c in db.get_characters(manga_id):
                evs = db.get_source_evidence(manga_id, "character", c["character_id"]) or []
                characters.append({
                    "id": c["character_id"],
                    "name": c["name"],
                    "role": c.get("role") or "",
                    "description": c.get("description") or "",
                    "first_page": c.get("first_page"),
                    "last_page": c.get("last_page"),
                    "appearance_count": c.get("appearance_count") or 0,
                    "confidence": c.get("confidence") or 0,
                    "source": c.get("source") or (evs[-1].get("source_type") if evs else "pdf"),
                    "sources": sorted({e.get("source_type") for e in evs}) if evs else [c.get("source") or "pdf"],
                    "aliases": c.get("aliases") or [],
                })

            locations = [{"id": l.get("location_id") or l.get("id"),
                          "name": l["name"],
                          "type": l.get("location_type") or "",
                          "first_page": l.get("first_page"),
                          "appearance_count": l.get("appearance_count") or 0,
                          "source": l.get("source") or "pdf"}
                         for l in db.get_locations(manga_id)]

            events = [{"id": e["id"], "page": e.get("page_number"),
                       "chapter_id": e.get("chapter_id"),
                       "characters": e.get("characters") or [],
                       "location": e.get("location") or "",
                       "description": e.get("description") or "",
                       "importance": e.get("importance") or 0,
                       "source": e.get("source") or "pdf"}
                      for e in db.get_events(manga_id)]

            chapters = [{"id": c["id"], "chapter_number": c.get("chapter_number"),
                         "title": c.get("title") or "",
                         "pdf_page_start": c.get("pdf_page_start"),
                         "pdf_page_end": c.get("pdf_page_end"),
                         "confidence": c.get("confidence") or 0}
                        for c in db.get_chapters(manga_id)]

            summaries = [{"id": s["id"], "summary_type": s.get("summary_type"),
                          "chapter_id": s.get("chapter_id"),
                          "page_number": s.get("page_number"),
                          "text": (s.get("text") or "")[:600],
                          "important_events": s.get("important_events") or []}
                         for s in db.get_summaries(manga_id)]

            conflicts = []
            for c in db.get_unresolved_conflicts(manga_id):
                conflicts.append({
                    "id": c["id"], "entity_type": c.get("entity_type"),
                    "entity_key": c.get("entity_key"),
                    "value_a": c.get("value_a"), "source_a": c.get("source_a"),
                    "value_b": c.get("value_b"), "source_b": c.get("source_b"),
                })

            checkpoints = []
            for cp in db.get_checkpoints(manga_id):
                checkpoints.append({
                    "stage": cp.get("stage"),
                    "key": cp.get("key_value"),
                    "status": cp.get("status"),
                    "detail": cp.get("detail") or "",
                    "started": cp.get("started_at"),
                    "completed": cp.get("completed_at"),
                })

            return {
                "status": "ok",
                "manga": target,
                "stats": stats,
                "characters": characters,
                "locations": locations,
                "events": events,
                "chapters": chapters,
                "summaries": summaries,
                "conflicts": conflicts,
                "checkpoints": checkpoints,
            }
        finally:
            try:
                db.close()
            except Exception:
                pass
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def _live(cfg):
    from pipeline.progress import read_progress
    doc = read_progress(state_dir=cfg.pipeline.state.dir)
    if not isinstance(doc, dict):
        return {"live": None}
    doc = dict(doc)
    doc["image_url"] = _live_image_url(cfg, doc.get("image"))
    doc["memory"] = _memory_info(cfg)
    return doc


# ---------------------------------------------------------------------------
# Long-running UI tasks (TTS generation/preview, server start) - not CLI jobs
# ---------------------------------------------------------------------------


def _run_task(label, fn, argv_note=None):
    """Run a callable in the JOB worker thread, streaming _emit() output."""
    with JOB.lock:
        JOB.running = True
        JOB.action = label
        JOB.output = []
        JOB.exit_code = None
        JOB.started = time.time()
    try:
        rc = fn()
        with JOB.lock:
            JOB.exit_code = rc if isinstance(rc, int) else (0 if rc in (None, True) else 1)
    except Exception as exc:
        _emit(f"[error] {type(exc).__name__}: {exc}")
        import traceback
        _emit(traceback.format_exc().splitlines()[-1])
        with JOB.lock:
            JOB.exit_code = 1
    finally:
        with JOB.lock:
            JOB.running = False


def _start_task(label, fn):
    if JOB.running:
        return False, "A command is already running. Wait for it to finish."
    threading.Thread(target=_run_task, args=(label, fn), daemon=True).start()
    return True, label


def _server_preferred_provider(cfg):
    """Resolve a TTS provider for the UI: Pocket TTS server first.

    The dashboard must stay light: narration work runs in the pocket-tts serve
    process, never inside webui. Falls back to the configured in-process
    provider only when the server is genuinely unavailable AND auto-start is
    off (or the named provider is not the server), so the UI never pretends.
    """
    from pipeline import pocket_server
    from pipeline.pocket_tts import create_pocket_tts_provider

    info = pocket_server.server_config(cfg)
    if info["auto_start"]:
        res = pocket_server.start_server(cfg)
        if res.get("ok"):
            _emit(f"[tts] Pocket TTS server ready on {res['url']} "
                  f"({'started' if res['started'] else 'already running'})")
            return pocket_server.PocketServerProvider(cfg)
        raise RuntimeError(
            "Pocket TTS server unavailable - " +
            (res.get("error") or res.get("detail") or info["url"])
        )
    if pocket_server.probe_server(cfg, timeout=1.0).get("reachable"):
        return pocket_server.PocketServerProvider(cfg)
    provider = create_pocket_tts_provider(cfg)  # in-process fallback (honest)
    _emit(f"[tts] server unreachable and auto-start off; using "
          f"in-process provider '{provider.name}'")
    return provider


def _tts_info(cfg):
    """Status block for http://host/api/tts (and /api/status.tts)."""
    from pipeline import pocket_server
    try:
        server = pocket_server.server_info(cfg)
    except Exception:
        server = {"reachable": False, "url": ""}
    provider_name = str(getattr(getattr(cfg, "tts", None), "provider", "") or "")
    return {
        "provider": provider_name or "pocket_tts",
        "voice": str(getattr(getattr(cfg, "tts", None), "default_voice", "") or "alba"),
        "reference": _voice_info(cfg),
        "server": server,
        "audio": list(narration_audio(cfg).keys()),
        "has_manifest": (Path(cfg.output.audio_dir) / "manifest.json").is_file(),
        "loading": JOB.running and (JOB.action or "").startswith("tts_"),
    }


def _load_script_docs(cfg):
    """All scene script JSONs (page -> [scene docs]) with editable segments."""
    docs = {}
    script_dir = Path(cfg.output.script_dir)
    import re
    pat = re.compile(r"page_(\d+)_scene_(\d+)\.json$")
    if script_dir.is_dir():
        for p in sorted(script_dir.glob("page_*_scene_*.json")):
            m = pat.search(p.name)
            if not m:
                continue
            page, scene = int(m.group(1)), int(m.group(2))
            try:
                data = json.loads(p.read_text("utf-8"))
            except Exception:
                continue
            if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
                continue
            docs.setdefault(page, {}).setdefault(scene, {
                "page": page, "scene": scene,
                "scene_id": data.get("scene_id") or f"scene_{scene:03d}",
                "segments": [{
                    "index": i,
                    "segment_id": s.get("segment_id", f"s{i+1:03d}"),
                    "text": s.get("text", ""),
                    "estimated_seconds": s.get("estimated_seconds"),
                } for i, s in enumerate(data["segments"])],
            })
    return docs


def _script_audio_state(cfg):
    """Map segment positions to existing audio files (for per-segment playback)."""
    out = Path(cfg.output.audio_dir)
    manifest_path = out / "manifest.json"
    mapping = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text("utf-8"))
            for i, entry in enumerate(manifest):
                apath = Path(entry.get("audio_path") or "")
                f = apath if apath.is_file() else out / f"segment_{i+1:03d}.wav"
                if f.is_file():
                    mapping[i] = f.name
        except Exception:
            pass
    return mapping


def _tts_generate_all(cfg):
    """Sequential server-backed narration for every scene script.

    Streams live per-segment Progress.detail (the transparent processing view)
    as well as worker output lines, writes output/audio/segment_NNN.wav +
    manifest.json with exact timings. Honest: stops on the first failure.
    """
    from pipeline.tts_manifest import (
        NarrationManifestRunner,
        load_narration_segments,
    )
    from pipeline import pocket_server
    from pipeline.progress import Progress

    progress = Progress(ROOT, state_dir=cfg.pipeline.state.dir)
    segments = []
    script_dir = Path(cfg.output.script_dir)
    candidates = sorted(script_dir.glob("*_scene_*.json"))
    if not candidates:
        raise RuntimeError("no narration scripts in script/ - run the script "
                           "stage first (write_script)")
    for script in candidates:
        try:
            segments.extend(load_narration_segments(script))
        except Exception as exc:
            raise RuntimeError(f"cannot read {script.name}: {exc}")
    if not segments:
        raise RuntimeError("no narration segments found in script/")

    provider = _server_preferred_provider(cfg)
    progress.begin("pocket_tts", "Pocket TTS")
    progress.phase("pocket_tts", f"Synthesizing {len(segments)} narration segments")
    _emit(f"[tts] generating {len(segments)} segments via {provider.name} "
          f"({'reference voice' if provider.conditioning == 'reference' else 'built-in voice'})")

    def _on_progress(done, total, seg_id=None, text=None, duration=None,
                     audio_path=None):
        progress.step("pocket_tts", done, total, phase=f"Segment {done} of {total}")
        if seg_id:
            progress.detail(
                "pocket_tts", segment=done, segment_count=total,
                segment_id=seg_id,
                text=(str(text or "")[:140] +
                      ("…" if text and len(text) > 140 else "")),
                duration_seconds=round(float(duration), 3) if duration else None,
                audio_path=str(audio_path) if audio_path else None,
            )
        _emit(f"[tts {done}/{total}] {seg_id}: {round(duration or 0, 2)}s")

    runner = NarrationManifestRunner(cfg, provider=provider)
    runner.generate(segments, cfg.output.audio_dir, force=False,
                    on_progress=_on_progress)
    progress.phase("pocket_tts", "Computing segment timings")
    runner.finalize_timing(cfg.output.audio_dir)
    progress.phase("pocket_tts", "All narration segments ready")
    _emit("[tts] manifest.json written with exact timings")
    return 0


def _tts_preview_segment(cfg, data):
    """Synthesize a single script segment as output/audio/preview.wav."""
    from pipeline.progress import Progress
    from pipeline import pocket_server

    page = _atoi(data.get("page"), 1)
    scene = _atoi(data.get("scene"), 1)
    index = _atoi(data.get("index"), 0)
    override = (data.get("text") or "").strip()
    script = Path(cfg.output.script_dir) / f"page_{page:03d}_scene_{scene:03d}.json"
    if not script.is_file():
        raise RuntimeError(f"no script at {script.name}")
    doc = json.loads(script.read_text("utf-8"))
    segs = doc.get("segments") or []
    if index < 0 or index >= len(segs):
        raise RuntimeError(f"segment index {index} out of range (0..{len(segs)-1})")
    seg = segs[index]
    text = override or (seg.get("text") or "")
    if not text.strip():
        raise RuntimeError("segment has no narration text to preview")

    provider = _server_preferred_provider(cfg)
    progress = Progress(ROOT, state_dir=cfg.pipeline.state.dir)
    progress.begin("pocket_tts", "Pocket TTS preview")
    progress.phase("pocket_tts", f"Previewing segment {index+1} of scene "
                                 f"{scene:03d} ({provider.name})")
    _emit(f"[tts preview] synthesizing '{text[:60]}…' via {provider.name}")
    out = Path(cfg.output.audio_dir) / "preview.wav"
    duration = provider.synth(text, str(out))
    progress.phase("pocket_tts", f"Preview ready ({round(duration, 2)}s)")
    progress.detail("pocket_tts", segment=index + 1,
                    segment_id=seg.get("segment_id", f"s{index+1:03d}"),
                    text=(text[:140] + "…" if len(text) > 140 else text),
                    duration_seconds=round(float(duration), 3),
                    audio_path=str(out))
    _emit(f"[tts preview] {round(duration, 2)}s -> output/audio/preview.wav")
    return 0


def _script_doc(cfg, page, scene):
    """Load ONE scene script doc (for saving)."""
    script = Path(cfg.output.script_dir) / f"page_{page:03d}_scene_{scene:03d}.json"
    if not script.is_file():
        raise ValueError(f"no script {script.name}")
    return script, json.loads(script.read_text("utf-8"))


def _save_script_segment(cfg, data):
    """Persist an edited segment's narration text back into its script JSON."""
    page = _atoi(data.get("page"), None)
    scene = _atoi(data.get("scene"), None)
    index = _atoi(data.get("index"), None)
    text = (data.get("text") or "").strip()
    if page is None or scene is None or index is None or not text:
        raise ValueError("page, scene, index and non-empty text are required")
    script, doc = _script_doc(cfg, page, scene)
    segs = doc.get("segments") or []
    if index < 0 or index >= len(segs):
        raise ValueError(f"segment index {index} out of range (0..{len(segs)-1})")
    doc["markup_version"] = doc.get("markup_version", "") + " [edited]"
    segs[index]["text"] = text
    script.write_text(json.dumps(doc, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    _emit(f"[script] edited scene {scene:03d} segment {index + 1} -> {script.name}")
    return script.name, index, text


def _book_fetch(cfg, data, cast=None):
    """Fetch the manga's important facts from the internet and store them as
    durable memory.

    This is the app's single compulsory onboarding step: the user types the
    manga's name once and everything useful is remembered — the book reference
    (VERIFIED book record) plus character names (learned into the CHARACTER
    store so the narrator stays consistent). ``cast`` is an optional list of
    ``{"name", "role"}`` dicts from :func:`internet_ref.fetch_characters` that
    enriches the character records with a real one-line role each. Returns a
    one-line summary of what was remembered.
    """
    from pipeline import internet_ref
    from pipeline.manga_memory.book import BookMemory
    from pipeline.manga_memory.character import CharacterMemory
    from pipeline.manga_memory.store import open_memory

    title = (data.get("title") or "").strip()
    if not title:
        raise ValueError("manga name is empty")
    info = internet_ref.fetch_book_ref(title, timeout=10)
    if not info:
        raise ValueError(
            f"could not fetch info about {title!r} - check the spelling / "
            "your internet connection, then try again")
    memory = open_memory(cfg, lazy=True).load_all()
    rec = BookMemory(memory.store_for("book")).remember(
        info, source="internet:" + (info.get("source") or "unknown"))

    chars = CharacterMemory(memory.store_for("character"))
    learned = 0
    roles_by_name = {}
    for c in cast or []:
        if isinstance(c, dict) and c.get("name"):
            roles_by_name[str(c["name"]).strip().lower()] = str(c.get("role") or "").strip()
    for name in (info.get("characters") or []):
        name = str(name or "").strip()
        if not name:
            continue
        if not chars.record(name):
            role = roles_by_name.get(name.lower())
            chars.learn(
                name,
                source="internet:book:" + (info.get("source") or "unknown"),
                description=role or (
                    f"character of {info.get('title')} "
                    f"({info.get('genres') and ', '.join(info.get('genres')) or 'manga'})"),
                role=role or None,
                confidence=0.9,
            )
            learned += 1
    # also learn any rich-cast names not already in the fetched names
    for c in cast or []:
        if not isinstance(c, dict) or not c.get("name"):
            continue
        name = str(c["name"]).strip()
        if name and not chars.record(name):
            role = str(c.get("role") or "").strip()
            chars.learn(
                name,
                source="internet:cast:" + (info.get("source") or "unknown"),
                description=role or (
                    f"character of {info.get('title')} "
                    f"({info.get('genres') and ', '.join(info.get('genres')) or 'manga'})"),
                role=role or None,
                confidence=0.9,
            )
            learned += 1
    memory.save_all()
    _emit(f"[book] \"{info.get('title')}\" fetched from {info.get('source')} "
          f"-> durable book memory ({rec.key}, {learned} character(s))")
    summary = internet_ref.book_ref_to_text(info)
    source_url = info.get("url") or ""
    return summary + (f" - {source_url}" if source_url else "")


# ---------------------------------------------------------------------------
# Projects (home screen) — magazine name identity, character images, toggles
# ---------------------------------------------------------------------------

IMG_URL_PREFIX = "/media/project_img/"


def _state_dir(cfg) -> str:
    return str(cfg.pipeline.state.dir)


def _project_image_url(slug, file) -> str | None:
    if not slug or not file:
        return None
    return f"{IMG_URL_PREFIX}{slug}/{file}"


def _projects_live(cfg):
    """All saved projects, each enriched with image URLs + memory counts."""
    from pipeline import project_registry
    from pipeline.character_images import image_store_dir

    state_dir = _state_dir(cfg)
    projects = project_registry.list_projects(state_dir)
    memory = _manga_records(cfg)
    counts = {}
    if isinstance(memory, dict) and not memory.get("error"):
        counts = {
            "books": len(memory.get("books") or []),
            "characters": len(memory.get("characters") or []),
            "world": len(memory.get("world") or []),
            "story": len(memory.get("story") or []),
            "corrections": len(memory.get("corrections") or []),
        }

    base = image_store_dir(state_dir)
    out = []
    for p in projects:
        slug = p.get("slug") or ""
        cover = _project_image_url(slug, p.get("cover_file"))
        char_imgs = {
            name: _project_image_url(slug, f)
            for name, f in (p.get("characters") or {}).items() if f
        }
        out.append({
            "slug": slug,
            "name": p.get("name") or slug,
            "cover_file": p.get("cover_file"),
            "cover_url": cover,
            "characters": [(n, char_imgs.get(n)) for n in (p.get("character_names") or [])]
            if p.get("character_names")
            else [(n, u) for n, u in char_imgs.items()],
            "character_images": char_imgs,
            "cast": p.get("cast") if isinstance(p.get("cast"), list) else None,
            "series_cast": p.get("series_cast")
            if isinstance(p.get("series_cast"), list) else None,
            "volume": p.get("volume") or None,
            "toggles": p.get("toggles") or project_registry.default_toggles(),
            "book": p.get("book"),
            "created_at": p.get("created_at"),
            "updated_at": p.get("updated_at"),
            "memory": counts,
        })
    return {"projects": out, "count": len(out), "state_dir": state_dir}


def _project_new(cfg, data, volume=None):
    """One-click new project: name -> fetch facts + character images + remember.

    Combines the existing book-fetch (durable memory + characters) with the new
    image layer (Wikipedia portraits + cover fallback) and registers the project
    in the registry so it appears on the home screen.

    When ``volume`` (from the scanned PDF's title, e.g. "Berserk v01") is given,
    the cast/characters are scoped to that collected volume (see
    :func:`internet_ref.scope_cast_to_volume`) so a vol-1 PDF isn't annotated
    with the whole series' later-arc roster; the full series list is still
    preserved on the project record under ``series_cast``.
    """
    from pipeline import internet_ref, project_registry
    from pipeline.character_images import _slugify, ensure_images

    title = (data.get("title") or "").strip()
    if not title:
        raise ValueError("manga name is empty")
    info = internet_ref.fetch_book_ref(title, timeout=10)
    if not info:
        raise ValueError(
            f"could not fetch info about {title!r} - check the spelling / "
            "your internet connection, then try again")
    # one parse of the character-list page -> cast + verified character images
    series_cast, portraits = internet_ref.fetch_characters_with_portraits(
        title, timeout=12)
    cast = internet_ref.scope_cast_to_volume(series_cast, volume)

    # 1. durable memory + learned character names (with real roles when found)
    summary = _book_fetch(cfg, {"title": title}, cast=cast)

    state_dir = _state_dir(cfg)
    info_title = info.get("title") or title
    cast_names = [str(m.get("name")).strip()
                  for m in cast if isinstance(m, dict) and m.get("name")]
    chars = cast_names or [
        str(c).strip() for c in (info.get("characters") or []) if str(c).strip()
    ]

    # 2. images (best-effort, never blocks the flow)
    imgs = ensure_images(
        info_title, chars, cover_url=info.get("cover_url"),
        state_dir=state_dir, timeout=6.0, portraits=portraits)

    # 3. register the project (full roster kept as series_cast when scoped)
    record = project_registry.upsert_project(state_dir, {
        "slug": imgs["slug"],
        "name": info_title,
        "cover_file": imgs["cover_file"],
        "characters": imgs["characters"],
        "character_names": chars,
        "cast": [{"name": c["name"], "role": c.get("role", "")} for c in cast]
                 if cast_names else None,
        "series_cast": [{"name": c["name"], "role": c.get("role", "")}
                        for c in series_cast] if series_cast else None,
        "volume": volume or None,
        "toggles": project_registry.default_toggles(),
        "book": info,
    })
    _emit(f"[project] registered '{info_title}' ({imgs['slug']}) "
          f"cover={imgs['cover_file'] or 'none'} "
          f"char-images={len(imgs.get('characters') or {})}"
          + (f" vol={volume}" if volume else ""))
    return summary, record


def _project_scan(cfg, data):
    """Auto-build a project by scanning the PDF (no typing).

    Reads the input PDF's metadata + first-page text probe via
    :func:`pipeline.pdf_scan.scan_pdf` to guess the title, then reuses
    :func:`_project_new` to fetch facts + cast + images from the internet and
    remember them. Returns ``(summary, record, scan)`` where ``scan`` is the raw
    pdf_scan dict (so the UI can show what was detected). Raises ValueError when
    the title can't be determined or the internet enrichment fails.
    """
    from pipeline import pdf_scan

    pdf = cfg.input.pdf
    scan = pdf_scan.scan_pdf(pdf)
    if not scan.get("page_count"):
        raise ValueError(
            scan.get("reason") or f"could not read the PDF at {pdf}")
    title = (scan.get("title") or "").strip() or (data.get("title") or "").strip()
    if not title:
        raise ValueError(
            "could not detect the manga name from the PDF — the cover is "
            "scanned/image-only. Type the name in the box to continue.")
    summary, record = _project_new(cfg, {"title": title},
                                   volume=scan.get("volume") or None)
    return summary, record, scan


def _project_toggle(cfg, data):
    """Persist one toggle (tts / music) for a project; returns the new state."""
    from pipeline import project_registry
    slug = (data.get("slug") or "").strip()
    key = (data.get("key") or "").strip()
    value = bool(data.get("value"))
    if not slug:
        raise ValueError("project slug is required")
    rec = project_registry.set_toggle(_state_dir(cfg), slug, key, value)
    if rec is None:
        raise ValueError(f"no project '{slug}' to toggle")
    return rec.get("toggles") or {}


def _project_delete(cfg, data):
    """Remove a project from the registry (keeps memory + images on disk)."""
    from pipeline import project_registry
    slug = (data.get("slug") or "").strip()
    if not slug:
        raise ValueError("project slug is required")
    return project_registry.delete_project(_state_dir(cfg), slug)


def _clear_all(cfg):
    """Desktop 'clear all': wipe projects, durable memory and fetched images.

    Keeps the input PDF and the config. Returns a summary dict.
    """
    import shutil
    from pipeline import project_registry
    from pipeline.manga_memory import store as memstore
    from pipeline.character_images import image_store_dir

    state_dir = _state_dir(cfg)

    # projects
    removed = 0
    for p in project_registry.list_projects(state_dir):
        if project_registry.delete_project(state_dir, p.get("slug")):
            removed += 1

    # durable memory (all kinds -> empty record lists)
    cleared = 0
    for kind, fname in memstore.FILE_BY_KIND.items():
        f = Path(state_dir) / memstore.DIRNAME / fname
        try:
            if f.is_file():
                data = json.loads(f.read_text(encoding="utf-8") or "{}")
                if isinstance(data, dict):
                    data["records"] = []
                    f.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
                    cleared += 1
        except Exception as exc:
            _emit(f"[clear] could not reset {fname}: {exc}")

    # fetched images
    img_dir = image_store_dir(state_dir)
    img_removed = 0
    if img_dir.is_dir():
        for child in img_dir.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
                img_removed += 1
            except Exception as exc:
                _emit(f"[clear] could not remove {child}: {exc}")

    # the active-project overlay too (so a cleared app doesn't force toggles)
    _write_run_overlay(cfg)

    _emit(f"[clear] removed {removed} project(s), reset {cleared} memory "
          f"file(s), removed {img_removed} image item(s)")
    return {
        "projects": removed,
        "memory_files": cleared,
        "images": img_removed,
    }


_ARTIFACT_DIR_KEYS = (
    "pages_dir", "panels_dir", "clean_dir", "ocr_dir",
    "analysis_dir", "scenes_dir", "script_dir", "audio_dir",
    "shots_dir", "crops_dir", "matching_dir",
)

# Generated files directly under the output root (not a config-keyed *_dir).
_ARTIFACT_ROOT_FILES = ("final_video.mp4",)
_ARTIFACT_ROOT_DIRS = ("motion", "visuals", "tmp")


def _clear_cache(cfg):
    """Desktop 'clear cache': wipe the generated run artifacts (pages, panels,
    cleaned panels, OCR, analysis, crops, scenes, script, audio, shots,
    matching + the final video / render scraps).

    Keeps the input PDF, config, project memory, fetched images and any
    uploaded reference voice clip. Returns a summary dict.
    """
    import shutil

    output_root = Path(cfg.output.dir) if cfg.output.dir else Path("output")
    removed_items = 0
    removed_dirs = 0

    def _wipe(path: Path):
        nonlocal removed_items, removed_dirs
        if not path.exists():
            return
        for child in path.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                    removed_dirs += 1
                else:
                    child.unlink()
                    removed_items += 1
            except Exception as exc:
                _emit(f"[cache] could not remove {child}: {exc}")

    # every config-keyed artifact dir (clear contents, keep the dirs)
    out_node = cfg.output
    for key in _ARTIFACT_DIR_KEYS:
        path = getattr(out_node, key, None)
        if path:
            _wipe(Path(path))

    # final video + render scraps directly under the output root
    for name in _ARTIFACT_ROOT_FILES:
        f = output_root / name
        try:
            if f.is_file():
                f.unlink()
                removed_items += 1
        except Exception as exc:
            _emit(f"[cache] could not remove {f}: {exc}")
    for name in _ARTIFACT_ROOT_DIRS:
        _wipe(output_root / name)

    _emit(f"[cache] removed {removed_items} file(s) and {removed_dirs} folder(s) "
          "of generated artifacts")
    return {"files": removed_items, "dirs": removed_dirs}


_OVERLAY_PATH = "projects-active.yaml"


def _active_project(cfg):
    """The single 'active' project = the most recently updated one.

    Since the pipeline itself is single (one config, one PDF), the home screen
    picks one project to forward its toggles to the run. Out of the projects
    the user has opened, the newest is treated as active.
    """
    from pipeline import project_registry
    projects = project_registry.list_projects(_state_dir(cfg))
    if not projects:
        return None
    return projects[0]


def _write_run_overlay(cfg):
    """Persist the active project's toggles into a small config the runner uses.

    The pipeline honours ``tts.enabled`` / ``music.enabled`` straight from the
    config file; we write an overlay with just those two keys and point ``--config``
    at it. ``load_config`` merges it over defaults, so nothing else changes.
    """
    from pipeline import project_registry
    proj = _active_project(cfg)
    if not proj:
        # no project -> nothing to enforce; remove any stale overlay so a
        # cleared app does NOT keep forcing old toggles.
        path = ROOT / _OVERLAY_PATH
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    toggles = proj.get("toggles") or project_registry.default_toggles()
    overlay = {
        "tts": {"enabled": bool(toggles.get("tts", True))},
        "music": {"enabled": bool(toggles.get("music", False))},
    }
    import yaml
    path = ROOT / _OVERLAY_PATH
    path.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
    return str(path)


def _run_config():
    """Config path a pipeline action should run against (overlay-aware)."""
    try:
        return _write_run_overlay(_load_cfg())
    except Exception:
        return _CFG_PATH or None


# ---------------------------------------------------------------------------
# Mode detection (offline demo vs real models) + config selection
# ---------------------------------------------------------------------------


def _mode(cfg):
    def provider(section, key="provider", default=""):
        node = getattr(cfg, section, None)
        if node is None:
            return default
        value = node.get(key, default)
        return value if isinstance(value, str) else default

    vlm = provider("vlm")
    llm = provider("llm")
    tts = provider("tts")
    ocr = getattr(cfg, "ocr", None).get("engine", "auto") if getattr(cfg, "ocr", None) else "auto"
    engines = {"vlm": vlm or "local", "llm": llm or "local",
               "ocr": ocr or "auto", "tts": tts or "auto"}
    offline = any(v in ("mock", "dummy") for v in engines.values())
    return {
        "offline": offline,
        "note": ("offline demo mode: mock/dummy providers get real work done "
                 "with placeholder values (perfect for testing the dashboard)")
        if offline else
        ("real mode: configured providers will be used (needs the engines "
         "installed on this machine)"),
        "engines": engines,
    }


def _resolve_config(config):
    if not config:
        return None
    p = Path(config)
    if not p.is_absolute():
        p = ROOT / p
    return str(p)


def _pick_config(config):
    """Choose which config the dashboard should run actions against.

    An explicit --config is always respected. Otherwise the default real-model
    config is attempted; if the default targets providers that cannot work in
    this environment and the offline demo config exists, the dashboard falls
    back to the demo config so that every button actually does something.
    """
    if config:
        return _resolve_config(config)
    demo = ROOT / "config" / "config.task29.yaml"
    try:
        from config.loader import load_config
        default_cfg = load_config(ROOT, None)
    except Exception:
        return str(demo) if demo.exists() else None
    if _mode(default_cfg)["offline"]:
        return None  # the default config already is a demo config
    if demo.exists():
        return str(demo)
    return None


# ---------------------------------------------------------------------------
# Pipeline / artifact state helpers
# ---------------------------------------------------------------------------


def _load_cfg():
    from config.loader import load_config
    return load_config(ROOT, _CFG_PATH or None)


def _checkpoint_state(cfg):
    from state import State
    from pipeline.run_pipeline import PIPELINE_STAGES, STAGE_NAMES
    from pipeline.progress import read_progress
    labels = dict(PIPELINE_STAGES)
    state = State(STAGE_NAMES, Path(cfg.pipeline.checkpoints.dir))
    stages = []
    for row in state.details():
        stages.append({
            "name": row["name"],
            "label": labels.get(row["name"], row["name"]),
            "status": row["status"],
            "started_at": row.get("started_at"),
            "completed_at": row.get("completed_at"),
        })
    rows = {r["name"]: r["status"] for r in state.summary()}
    current = next((s["name"] for s in stages if s["status"] == "running"), None)
    failed = [s["name"] for s in stages if s["status"] == "failed"]
    return {
        "total": len(STAGE_NAMES),
        "completed": state.completed_count(),
        "rows": rows,
        "stages": stages,
        "current": current or None,
        "failed": failed,
        "next": state.next_pending(),
        "done": state.is_complete(),
        "live": read_progress(state_dir=cfg.pipeline.state.dir),
    }


def _artifacts(cfg):
    out = Path(cfg.output.dir)
    audio = Path(cfg.output.audio_dir)
    return {
        "final_video": str(out / "final_video.mp4")
        if (out / "final_video.mp4").exists() else None,
        "narration": list(narration_audio(cfg).keys()),
        "voice_sample": str(_sample_audio_path(cfg))
        if _sample_audio_path(cfg).exists() else None,
        "pages": sorted(p.name for p in Path(cfg.output.pages_dir).glob("page_*"))
        if Path(cfg.output.pages_dir).is_dir() else [],
        "pdf": str(cfg.input.pdf),
    }


_PAGE_IMG = re.compile(r"page_(\d+)\.(?:png|jpe?g)$", re.I)
_PANEL_IMG = re.compile(r"panel_(\d+)\.(?:png|jpe?g)$", re.I)


def _numbered(path):
    """Natural-order key for page/panel files: page_002.png sorts after page_010."""
    m = _PAGE_IMG.match(path.name) or _PANEL_IMG.match(path.name)
    return int(m.group(1)) if m else 10**9


def _extract_live(cfg, full=False):
    """Snapshot of pages + panels currently on disk, in extraction order.

    Fed to the "Extraction, live" filmstrip so page/panel extraction is
    visible one-by-one as it lands, whether a job is running or not.

    full=False returns just page metadata + panel counts (cheap idle poll);
    full=True additionally returns every panel crop URL (thumbnails).
    """
    pages_dir = Path(cfg.output.pages_dir)
    panels_dir = Path(cfg.output.panels_dir)
    pages = []
    if pages_dir.is_dir():
        for page_file in sorted(pages_dir.iterdir(), key=_numbered):
            if _PAGE_IMG.match(page_file.name) is None:
                continue
            num = int(_PAGE_IMG.match(page_file.name).group(1))
            panels = []
            page_panels = panels_dir / f"page_{num:03d}"
            if page_panels.is_dir():
                for panel_file in sorted(page_panels.iterdir(), key=_numbered):
                    if _PANEL_IMG.match(panel_file.name) is None:
                        continue
                    panels.append({
                        "name": panel_file.name,
                        "url": f"/media/panel/page_{num:03d}/{panel_file.name}",
                    })
            pages.append({
                "num": num,
                "file": page_file.name,
                "url": f"/media/page/{page_file.name}",
                "pans": len(panels),
                "panels": panels if full else [],
            })
    return {
        "pages": pages,
        "total_pages": len(pages),
        "running": JOB.running,
        "action": JOB.action,
    }


def _clean_live(cfg, full=False):
    """Snapshot of cleaned panels currently on disk, in page/panel order.

    Mirrors _extract_live: the "Panel cleaning" filmstrip shows each cleaned
    panel (and its debug overlay when present) filling in as the cleaning
    stage works, page by page, whether a job is running or not.
    """
    pages = []
    clean_dir = Path(cfg.output.clean_dir)
    if clean_dir.is_dir():
        for page_dir in sorted(clean_dir.iterdir()):
            if not page_dir.is_dir() or not page_dir.name.startswith("page_"):
                continue
            num = page_dir.name[len("page_"):].split(".", 1)[0]
            if not num.isdigit():
                continue
            cleaned = []
            for panel_file in sorted(page_dir.iterdir(), key=_numbered):
                if _PANEL_IMG.match(panel_file.name) is None:
                    continue
                debug = page_dir / (panel_file.stem + "_debug.jpg")
                cleaned.append({
                    "name": panel_file.name,
                    "url": f"/media/clean/{page_dir.name}/{panel_file.name}",
                    "debug": f"/media/clean/{page_dir.name}/{debug.name}"
                             if debug.is_file() else None,
                })
            pages.append({
                "num": int(num),
                "file": page_dir.name,
                "pans": len(cleaned),
                "panels": cleaned if full else [],
            })
    return {
        "pages": pages,
        "total_pages": len(pages),
        "running": JOB.running,
        "action": JOB.action,
    }


_OCR_DEBUG = re.compile(r"page_(\d+)_panel_(\d+)_debug\.(?:png|jpe?g)$", re.I)


def _understand_live(cfg, full=False):
    """Snapshot of panel understanding currently on disk, in page/panel order.

    The understanding stage writes an OCR overlay per panel
    (ocr/page_NNN_panel_YYY_debug.jpg — the panel with detected dialogue/text
    boxes drawn on it) plus a per-panel analysis json
    (analysis/page_NNN_panel_YYY.json). This mirrors _extract_live/_clean_live:
    the "Understanding, live" filmstrip shows each panel's overlay filling in as
    understand_panels works, whether a job is running or not.
    """
    ocr_dir = Path(cfg.output.ocr_dir)
    analysis_dir = Path(cfg.output.analysis_dir)
    overlays = {}
    if ocr_dir.is_dir():
        for f in ocr_dir.iterdir():
            m = _OCR_DEBUG.match(f.name)
            if not m:
                continue
            page, panel = int(m.group(1)), int(m.group(2))
            overlays.setdefault(page, {})[panel] = f.name

    # load per-panel analysis metadata (VLM output) for tooltips/notes
    meta = {}
    if analysis_dir.is_dir():
        for f in analysis_dir.iterdir():
            if not f.is_file() or not f.name.startswith("page_") \
                    or "_panel_" not in f.name or not f.name.endswith(".json"):
                continue
            try:
                d = json.load(open(f, encoding="utf-8"))
                page = int(d.get("page") or 0)
                panel = int(d.get("panel") or 0)
                summary = ""
                a = d.get("analysis") or {}
                if isinstance(a, dict):
                    env = a.get("environment") or ""
                    chars = a.get("characters") or []
                    if isinstance(chars, list) and chars:
                        summary = ", ".join(str(c) for c in chars[:3])
                    elif env and env not in ("unknown", ""):
                        summary = str(env)
                blurb = [d.get("model") or "", summary or d.get("important_event") or ""]
                meta.setdefault(page, {})[panel] = "".join(
                    (" | " + p) if p else "" for p in blurb if p)
            except Exception:
                continue

    pages = []
    for page in sorted(set(overlays) | set(meta)):
        panel_map = overlays.get(page) or {}
        meta_map = meta.get(page) or {}
        if not (panel_map or meta_map):
            continue
        panels = [
            {
                "page": page,
                "panel": p,
                "name": panel_map.get(p),
                "url": f"/media/ocr/{panel_map[p]}" if panel_map.get(p) else None,
                "note": meta_map.get(p),
            }
            for p in sorted(set(panel_map) | set(meta_map))
        ]
        pages.append({
            "num": page,
            "file": f"page_{page:03d}",
            "pans": len(panels),
            "panels": panels if full else [],
        })

    return {
        "pages": pages,
        "total_pages": len(pages),
        "running": JOB.running,
        "action": JOB.action,
    }


def _status(cfg):
    try:
        from pipeline.omniroute_provider import omniroute_status
        orstatus = omniroute_status(cfg)
    except Exception:
        orstatus = {"reachable": False}
    return {
        "config": _CFG_PATH or "config/config.yaml",
        "mode": _mode(cfg),
        "pdf": str(cfg.input.pdf),
        "pdf_exists": Path(cfg.input.pdf).is_file(),
        "progress": _checkpoint_state(cfg),
        "music": _music_info(cfg),
        "sfx": _sfx_info(cfg),
        "voice": _voice_info(cfg),
        "memory": _memory_info(cfg),
        "omniroute": orstatus,
        "tts": _tts_info(cfg),
        "script": _script_stats(cfg),
        "artifacts": _artifacts(cfg),
    }


def _script_stats(cfg):
    docs = _load_script_docs(cfg)
    segments = sum(len(scene["segments"])
                   for per_page in docs.values()
                   for scene in per_page.values())
    return {"docs": len(docs), "segments": segments,
            "audio": list(narration_audio(cfg).keys())}


# ---------------------------------------------------------------------------
# Action dispatch: UI action -> CLI argv
# ---------------------------------------------------------------------------


def _atoi(v, default=None):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _argv_for(cfg, action, data):
    _cfg = _run_config()
    base = ["--config", _cfg] if _cfg else []
    page = _atoi(data.get("page"), 1)
    scene = _atoi(data.get("scene"), 1)
    force = bool(data.get("force"))

    flag = ["--force"] if force else []
    simple = {
        "start": [*base, "start", *flag],
        "resume": [*base, "resume"],
        "status": [*base, "status"],
        "check": [*base, "check"],
        "clean": [*base, "clean"],
        "clean-cache": [*base, "clean-cache"],
        "export": [*base, "export"],
        "mix": [*base, "mix"],
        "render": [*base, "render"],
        "quality-check": [*base, "quality-check"],
        "tts-narration": [*base, "tts-narration"],
        "panels-prep": [*base, "panels-prep"],
    }
    if action in simple:
        return simple[action]

    page_only = {"extract", "panels", "order", "knowledge", "scenes"}
    if action in page_only:
        return [*base, action, "--page", str(page), *flag]
    if action in {"ocr", "analyze"}:
        panel = _atoi(data.get("panel"), 1)
        return [*base, action, "--page", str(page), "--panel", str(panel), *flag]
    if action in {"script", "audio", "plan", "crops"}:
        return [*base, action, "--page", str(page), "--scene", str(scene), *flag]
    if action == "tts":
        seg = _atoi(data.get("segment"), None)
        argv = [*base, "tts", "--page", str(page), "--scene", str(scene)]
        if seg:
            argv += ["--segment", str(seg)]
        if force:
            argv += ["--force"]
        return argv
    if action == "motion":
        return [*base, "motion", "--page", str(page), *flag]
    if action == "voice_sample":
        gen_voice_sample(cfg)
        return []
    return None


def _dispatch_action(cfg, data):
    action = (data.get("action") or "").strip()
    if action == "refresh":
        return True, None
    if action == "voice_sample":
        result = gen_voice_sample(cfg)
        return True, f"Sample voice generated ({result['engine']}, {result['duration']}s)"
    if action == "music_track":
        path, existed = _ensure_music_track(cfg)
        return True, (f"Music track ready: {path}"
                      if existed else
                      f"Generated placeholder music track -> {path}")
    if action == "music_fetch":
        try:
            return True, _music_fetch(cfg, data)
        except ValueError as exc:
            return False, f"could not fetch music: {exc}"
    if action == "book_fetch":
        try:
            return True, _book_fetch(cfg, data)
        except ValueError as exc:
            return False, f"could not fetch the book reference: {exc}"
    if action == "project_new":
        try:
            summary, _rec = _project_new(cfg, data)
            return True, "New project ready — " + summary
        except ValueError as exc:
            return False, f"could not create project: {exc}"
    if action == "project_scan":
        try:
            summary, _rec, scan = _project_scan(cfg, data)
            src = {"metadata": "PDF metadata", "embedded-text": "PDF cover text"}.get(
                scan.get("source"), "PDF")
            return True, (
                f"Detected '{scan.get('title')}' from {src} ({scan.get('page_count')} pages) — "
                f"{summary}")
        except ValueError as exc:
            return False, f"could not build from PDF: {exc}"
    if action == "project_toggle":
        try:
            state = _project_toggle(cfg, data)
            return True, "Project option updated: " + ", ".join(
                f"{k}={('on' if v else 'off')}" for k, v in state.items())
        except ValueError as exc:
            return False, str(exc)
    if action == "project_delete":
        try:
            removed = _project_delete(cfg, data)
            return True, ("Project removed" if removed
                          else f"No project to remove")
        except ValueError as exc:
            return False, str(exc)
    if action == "clear_all":
        try:
            res = _clear_all(cfg)
            return True, (f"Cleared: {res['projects']} project(s), "
                          f"{res['memory_files']} memory file(s), "
                          f"{res['images']} image item(s). PDF kept.")
        except ValueError as exc:
            return False, f"could not clear: {exc}"
    if action == "clear_cache":
        try:
            res = _clear_cache(cfg)
            return True, (f"Cleared cache: {res['files']} file(s), "
                          f"{res['dirs']} folder(s) of generated artifacts. "
                          "Projects, memory, PDF and voice kept.")
        except ValueError as exc:
            return False, f"could not clear cache: {exc}"
    if action == "sfx_demo":
        sdir = _gen_sfx_demo(cfg)
        count = len(_sfx_manifest_events(cfg))
        return True, (f"Generated {count} placeholder SFX (whoosh/thud/click) "
                      f"-> {sdir}")
    if action == "reset":
        removed = _reset_pipeline(cfg)
        return True, (f"Pipeline state reset — cleared {removed} checkpoint "
                      "file(s). Click Run full pipeline to start from scratch.")
    if action == "memory_correct":
        try:
            target, correction, kind = _apply_correction(cfg, data)
            return True, (f"Memory corrected: {kind} '{target}' -> '{correction}' "
                          "(recorded as user correction, highest priority)")
        except ValueError as exc:
            return False, str(exc)
        except Exception as exc:
            return False, f"could not save correction: {exc}"
    if action == "memory_delete":
        try:
            key, removed = _delete_memory_record(cfg, data)
            return True, (f"Deleted memory record '{key}'"
                          if removed else f"No record '{key}' found")
        except ValueError as exc:
            return False, str(exc)
        except Exception as exc:
            return False, f"could not delete record: {exc}"
    if action == "tts_start":
        return _start_task("tts_start", lambda: _start_pocket_tts_server(cfg))
    if action == "tts_stop":
        try:
            from pipeline import pocket_server
            res = pocket_server.stop_server(cfg)
            return True, (res.get("detail") or res.get("error")
                          or "Pocket TTS server stopped")
        except Exception as exc:
            return False, f"could not stop server: {exc}"
    if action == "tts_preview":
        return _start_task("tts_preview",
                           lambda: _tts_preview_segment(cfg, data))
    if action == "tts_generate":
        return _start_task("tts_generate", lambda: _tts_generate_all(cfg))
    if action == "script_save":
        try:
            fname, index, text = _save_script_segment(cfg, data)
            return True, (f"Saved narration for #{index + 1} in {fname}")
        except ValueError as exc:
            return False, str(exc)
    argv = _argv_for(cfg, action, data)
    if argv is None:
        return False, f"unknown action: {action}"
    return _start_job(argv, action)


def _start_pocket_tts_server(cfg):
    from pipeline import pocket_server
    res = pocket_server.start_server(cfg)
    if res.get("ok"):
        _emit(f"[tts] Pocket TTS server ready on {res['url']} "
              f"({'started' if res['started'] else 'already running'})")
    else:
        _emit(f"[tts] {res.get('error') or res.get('detail') or 'unavailable'}")
    return 0 if res.get("ok") else 1


def _reset_pipeline(cfg):
    """Clear checkpoint/state files so the next run starts from scratch."""
    removed = 0
    for key in ("pipeline.checkpoints.dir", "pipeline.state.dir"):
        node = cfg
        parts = key.split(".")
        for part in parts:
            node = getattr(node, part, None)
            if node is None:
                break
        if node is None:
            continue
        folder = Path(node)
        if not folder.is_dir():
            continue
        for p in folder.rglob("*.json"):
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
        try:
            (folder / "checkpoints.json").unlink()
        except OSError:
            pass
    return removed


def _extract_multipart_field(body, boundary, field_name):
    """Return the raw bytes of the named part from a multipart/form-data body.

    Minimal stdlib parser (avoids the deprecated `cgi` module). Returns None if
    the field is not found.
    """
    delimiter = b"--" + boundary.encode("utf-8")
    parts = body.split(delimiter)
    marker = f'name="{field_name}"'.encode("utf-8")
    for part in parts:
        if marker not in part:
            continue
        head, sep, payload = part.partition(b"\r\n\r\n")
        if not sep:
            continue
        data = payload
        if data.endswith(b"\r\n"):
            data = data[:-2]
        # strip trailing closing delimiter if present
        if data.endswith(b"--"):
            data = data[:-2]
        return data
    return None


def _multipart_filename(body, boundary, field_name):
    """Client-provided filename of the named upload field (for display only)."""
    import re
    mark = f'name="{field_name}"'.encode("utf-8")
    for part in body.split(b"--" + boundary.encode("utf-8")):
        if mark not in part:
            continue
        m = re.search(rb'filename="([^"]+)"', part)
        if m:
            try:
                return Path(m.group(1).decode("utf-8", "replace")).name
            except Exception:
                return None
        return None
    return None


# ---------------------------------------------------------------------------
# HTTP handler + server
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path):
        path = Path(path)
        if not path.is_file():
            self._send_json({"error": "not found"}, 404)
            return
        suffix = path.suffix.lower()
        ctype = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".ogg": "audio/ogg",
            ".mp4": "video/mp4",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".pdf": "application/pdf",
        }.get(suffix, "application/octet-stream")
        size = path.stat().st_size
        # Honour a single HTTP Range request. Browsers rely on 206 Partial
        # Content to stream/seek <video>/<audio>; answering a Range with a
        # full 200 body makes some players refuse to play at all.
        range_hdr = self.headers.get("Range", "").strip()
        if range_hdr and range_hdr.startswith("bytes="):
            try:
                start_s, _, end_s = range_hdr[len("bytes="):].partition("-")
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else size - 1
                if start < 0:
                    start = max(size + start, 0)  # suffix range bytes=-N
                if end >= size:
                    end = size - 1
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Range",
                                 f"bytes {start}-{end}/{size}")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(length))
                self.end_headers()
                with open(path, "rb") as fh:
                    fh.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = fh.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                return
            except ValueError:
                pass  # malformed range -> serve full body below
        with open(path, "rb") as fh:
            body = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        try:
            if route in ("/", "/index.html"):
                self._serve_index()
                return
            if route == "/api/status":
                self._send_json(_status(_load_cfg()))
                return
            if route == "/api/job":
                self._send_json(JOB.snapshot())
                return
            if route == "/api/extract":
                full = "full=1" in parsed.query.split("&")
                self._send_json(_extract_live(_load_cfg(), full=full))
                return
            if route == "/api/clean":
                full = "full=1" in parsed.query.split("&")
                self._send_json(_clean_live(_load_cfg(), full=full))
                return
            if route == "/api/understand":
                full = "full=1" in parsed.query.split("&")
                self._send_json(_understand_live(_load_cfg(), full=full))
                return
            if route == "/api/memory":
                self._send_json(_manga_records(_load_cfg()))
                return
            if route == "/api/kb":
                self._send_json(_kb_live(_load_cfg()))
                return
            if route == "/api/projects":
                self._send_json(_projects_live(_load_cfg()))
                return
            if route == "/api/scan":
                cfg = _load_cfg()
                from pipeline import pdf_scan
                self._send_json({
                    **pdf_scan.scan_pdf(cfg.input.pdf),
                    "pdf": str(cfg.input.pdf),
                    "pdf_exists": bool(cfg.input.pdf and Path(cfg.input.pdf).is_file()),
                })
                return
            if route == "/api/omniroute":
                cfg = _load_cfg()
                from pipeline.omniroute_provider import omniroute_status
                self._send_json(omniroute_status(cfg))
                return
            if route == "/api/live":
                self._send_json(_live(_load_cfg()))
                return
            if route == "/api/script":
                cfg = _load_cfg()
                self._send_json({
                    "docs": _load_script_docs(cfg),
                    "audio": _script_audio_state(cfg),
                })
                return
            if route == "/api/tts":
                self._send_json(_tts_info(_load_cfg()))
                return
            if route == "/media/tts_server_log":
                cfg = _load_cfg()
                from pipeline import pocket_server
                self._send_file(str(pocket_server._log_path(cfg)))
                return
            if route.startswith("/media/live_image/"):
                rel = unquote(parsed.path[len("/media/live_image/"):])
                cfg = _load_cfg()
                parts = Path(rel).parts
                dirs = _live_dirs(cfg)
                if not parts or parts[0] not in dirs:
                    self._send_json({"error": "bad live image path"}, 400)
                    return
                base = dirs[parts[0]].resolve()
                target = (base / Path(*parts[1:])).resolve()
                if target != base and not str(target).startswith(str(base) + os.sep):
                    self._send_json({"error": "bad live image path"}, 400)
                    return
                self._send_file(str(target))
                return
            if route == "/media/sfx/":
                self._send_json({"error": "missing file name"}, 400)
                return
            if route.startswith("/media/sfx/"):
                name = Path(unquote(parsed.path[len("/media/sfx/"):]))
                if len(name.parts) != 1 or name.suffix.lower() not in SFX_EXTS:
                    self._send_json({"error": "bad sfx path"}, 400)
                    return
                cfg = _load_cfg()
                self._send_file(str(_sfx_dir(cfg) / name.name))
                return
            if route == "/media/voice_sample":
                self._send_file(_sample_audio_path(_load_cfg()))
                return
            if route == "/media/music_file/":
                self._send_json({"error": "missing file name"}, 400)
                return
            if route.startswith("/media/music_file/"):
                name = Path(unquote(parsed.path[len("/media/music_file/"):]))
                if len(name.parts) != 1 or name.suffix.lower() not in AUDIO_EXTS:
                    self._send_json({"error": "bad music path"}, 400)
                    return
                cfg = _load_cfg()
                settings = _music_settings(cfg) or {}
                music_dir = ROOT / settings.get("dir", "music")
                self._send_file(str(music_dir / name.name))
                return
            if route == "/media/track":
                cfg = _load_cfg()
                path, _existed = _ensure_music_track(cfg)
                self._send_file(str(path))
                return
            if route == "/media/final_video":
                cfg = _load_cfg()
                self._send_file(str(Path(cfg.output.dir) / "final_video.mp4"))
                return
            if route == "/media/voice_reference":
                cfg = _load_cfg()
                path = _voice_reference_path(cfg)
                if not path.is_file():
                    self._send_json(
                        {"error": "no voice reference uploaded yet"}, 404)
                    return
                self._send_file(str(path))
                return
            if route == "/media/page/":
                self._send_json({"error": "missing file name"}, 400)
                return
            if route.startswith("/media/page/"):
                name = Path(unquote(parsed.path[len("/media/page/"):]))
                cfg = _load_cfg()
                self._send_file(str(Path(cfg.output.pages_dir) / name))
                return
            if route == "/media/panel/":
                self._send_json({"error": "missing file name"}, 400)
                return
            if route.startswith("/media/panel/"):
                name = Path(unquote(parsed.path[len("/media/panel/"):]))
                # accept only page_NNN/panel_NNN.ext relative to panels_dir
                if len(name.parts) != 2 or not _PANEL_IMG.match(name.name):
                    self._send_json({"error": "bad panel path"}, 400)
                    return
                cfg = _load_cfg()
                self._send_file(str(Path(cfg.output.panels_dir) / name))
                return
            if route == "/media/clean/":
                self._send_json({"error": "missing file name"}, 400)
                return
            if route.startswith("/media/clean/"):
                name = Path(unquote(parsed.path[len("/media/clean/"):]))
                # accept only page_NNN/panel_NNN(_debug).ext under clean_dir
                if len(name.parts) != 2 or not name.name.startswith("panel_") \
                        or name.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    self._send_json({"error": "bad clean path"}, 400)
                    return
                cfg = _load_cfg()
                self._send_file(str(Path(cfg.output.clean_dir) / name))
                return
            if route == "/media/ocr/":
                self._send_json({"error": "missing file name"}, 400)
                return
            if route.startswith("/media/ocr/"):
                name = Path(unquote(parsed.path[len("/media/ocr/"):]))
                # accept only page_XXX_panel_YYY_debug.ext under ocr_dir
                if len(name.parts) != 1 or _OCR_DEBUG.match(name.name) is None:
                    self._send_json({"error": "bad ocr path"}, 400)
                    return
                cfg = _load_cfg()
                self._send_file(str(Path(cfg.output.ocr_dir) / name))
                return
            if route == "/media/narration/":
                self._send_json({"error": "missing file name"}, 400)
                return
            if route.startswith("/media/narration/"):
                name = Path(unquote(parsed.path[len("/media/narration/"):]))
                cfg = _load_cfg()
                self._send_file(str(Path(cfg.output.audio_dir) / name))
                return
            if route == IMG_URL_PREFIX:
                self._send_json({"error": "missing slug/file"}, 400)
                return
            if route.startswith(IMG_URL_PREFIX):
                rest = Path(unquote(parsed.path[len(IMG_URL_PREFIX):]))
                if (len(rest.parts) != 2 or not rest.parts[0]
                        or rest.suffix.lower() not in IMG_EXTS):
                    self._send_json({"error": "bad project image path"}, 400)
                    return
                slug, fname = rest.parts[0], rest.parts[1]
                if Path(fname).name != fname:
                    self._send_json({"error": "bad project image path"}, 400)
                    return
                from pipeline import project_registry
                from pipeline.character_images import image_store_dir
                cfg = _load_cfg()
                base = image_store_dir(_state_dir(cfg)).resolve()
                target = (base / slug / fname).resolve()
                if target != base and not str(target).startswith(str(base) + os.sep):
                    self._send_json({"error": "bad project image path"}, 400)
                    return
                if not target.is_file():
                    self._send_json({"error": "no such image"}, 404)
                    return
                self._send_file(str(target))
                return
            self._send_json({"error": "unknown route"}, 404)
        except Exception as exc:
            LOG.exception("GET %s failed", route)
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def _serve_index(self):
        path = ROOT / "ui" / "index.html"
        if not path.is_file():
            self._send_json({"error": "ui/index.html missing"}, 500)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        parsed = urlparse(self.path)
        route = parsed.path
        try:
            if route == "/api/run":
                data = self._read_json()
                ok, message = _dispatch_action(_load_cfg(), data)
                self._send_json({"started": ok, "message": message},
                                200 if ok else 409)
                return
            if route == "/api/pdf":
                self._handle_pdf_upload(_load_cfg())
                return
            if route == "/api/audio":
                self._handle_audio_upload(_load_cfg())
                return
            if route == "/api/voice":
                self._handle_voice_upload(_load_cfg())
                return
            if route == "/api/sfx":
                self._handle_sfx_upload(_load_cfg())
                return
            self._send_json({"error": "unknown route"}, 404)
        except Exception as exc:
            LOG.exception("POST %s failed", route)
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _handle_pdf_upload(self, cfg):
        content_type = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length") or 0)
        if "boundary=" not in content_type:
            self._send_json({"error": "expected multipart/form-data"}, 400)
            return
        boundary = content_type.split("boundary=", 1)[1].strip().strip('"')
        body = self.rfile.read(length)
        data = _extract_multipart_field(body, boundary, "pdf")
        if data is None:
            self._send_json({"error": "no file in 'pdf' field"}, 400)
            return
        dest = Path(cfg.input.pdf)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        _emit(f"[pdf] uploaded {len(data)} bytes -> {dest}")
        self._send_json({"pdf": str(dest), "bytes": len(data)})

    def _handle_audio_upload(self, cfg):
        content_type = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length") or 0)
        if "boundary=" not in content_type:
            self._send_json({"error": "expected multipart/form-data"}, 400)
            return
        boundary = content_type.split("boundary=", 1)[1].strip().strip('"')
        body = self.rfile.read(length)
        data = _extract_multipart_field(body, boundary, "track")
        if data is None:
            self._send_json({"error": "no file in 'track' field"}, 400)
            return
        filename = _multipart_filename(body, boundary, "track")
        try:
            dest, saved_name = _save_uploaded_audio(cfg, data, filename)
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)
            return
        _emit(f"[audio] uploaded {len(data)} bytes -> {dest}")
        self._send_json({
            "path": str(dest),
            "name": saved_name,
            "bytes": len(data),
            "music": _music_info(cfg),
        })

    def _handle_sfx_upload(self, cfg):
        """Save an uploaded sound-effect file into the sfx dir (keeps name)."""
        content_type = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length") or 0)
        if "boundary=" not in content_type:
            self._send_json({"error": "expected multipart/form-data"}, 400)
            return
        boundary = content_type.split("boundary=", 1)[1].strip().strip('"')
        body = self.rfile.read(length)
        data = _extract_multipart_field(body, boundary, "fx")
        if data is None or len(data) == 0:
            self._send_json({"error": "no file in 'fx' field"}, 400)
            return
        filename = _multipart_filename(body, boundary, "fx")
        name = Path(filename or "").name
        if not name or Path(name).suffix.lower() not in SFX_EXTS:
            self._send_json({"error": "SFX must be a .wav / .mp3 / .ogg file"},
                            400)
            return
        dest = _sfx_dir(cfg) / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.write_bytes(data)
        except OSError as exc:
            self._send_json({"error": f"could not save SFX: {exc}"}, 500)
            return
        _emit(f"[sfx] uploaded {len(data)} bytes -> {dest}")
        self._send_json({
            "path": str(dest),
            "name": name,
            "bytes": len(data),
            "sfx": _sfx_info(cfg),
        })

    def _handle_voice_upload(self, cfg):
        """Save an uploaded voice clip as the TTS reference voice."""
        content_type = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length") or 0)
        if "boundary=" not in content_type:
            self._send_json({"error": "expected multipart/form-data"}, 400)
            return
        boundary = content_type.split("boundary=", 1)[1].strip().strip('"')
        body = self.rfile.read(length)
        data = _extract_multipart_field(body, boundary, "voice")
        if data is None:
            self._send_json({"error": "no file in 'voice' field"}, 400)
            return
        if len(data) == 0:
            self._send_json({"error": "the uploaded voice clip is empty"},
                            400)
            return
        dest = _voice_reference_path(cfg)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.write_bytes(data)
        except OSError as exc:
            self._send_json({"error": f"could not save voice: {exc}"}, 500)
            return
        _emit(f"[voice] uploaded reference voice {len(data)} bytes -> {dest}")
        self._send_json({
            "path": str(dest),
            "name": dest.name,
            "bytes": len(data),
            "voice": _voice_info(cfg),
        })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _free_port(port, tries=25):
    for candidate in range(port, port + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", candidate))
                return candidate
            except OSError:
                continue
    return port


def serve(host="127.0.0.1", port=8000, config=None, open_browser=True):
    global _CFG_PATH
    _CFG_PATH = _pick_config(config)
    import main as cli
    cfg = _load_cfg()
    cli.ensure_dirs(cfg)  # ensure the workspace folders exist
    port = _free_port(port)
    httpd = ThreadingHTTPServer((host, port), Handler)
    display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = f"http://{display_host}:{port}"
    used_config = _CFG_PATH or "config/config.yaml"
    print("=" * 58)
    print("MangaExplainer Web UI")
    print(f"  open in your browser : {url}/")
    print(f"  config               : {used_config}")
    mode = _mode(cfg)
    print(f"  mode                 : {'OFFLINE DEMO' if mode['offline'] else 'REAL'}")
    print("  press Ctrl+C to stop (closes the dashboard only, pipeline "
          "artifacts stay)")
    print("=" * 58)
    if open_browser:
        def _open():
            time.sleep(1.5)  # let the server breathe before opening the browser
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=_open, daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        httpd.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python webui.py",
                                     description="MangaExplainer web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default=None,
                        help="alternative config file")
    args = parser.parse_args(argv)
    serve(args.host, args.port, args.config)


if __name__ == "__main__":
    main()
