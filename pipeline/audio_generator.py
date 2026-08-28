"""Generate narration audio for ONE scene from its script segments.

Input  : script/page_001_scene_001.json   (from the write_script stage)
Output : audio/page_001_scene_001.wav     (concatenated narration, reading order)
         audio/page_001_scene_001.json    (manifest with per-segment offsets)

Design keeps the RAM budget: one scene in memory at a time, one TTS engine
held (never VLM + LLM + TTS simultaneously), and runtime resources released
after every scene. Segments are synthesized one by one to temporary WAV
files, then concatenated into the final single audio file - no audio for
panels, scenes or pages other than the requested one is ever touched.
"""
import gc
import json
import logging
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path

from .tts_provider import (
    TtsError,
    TtsNotConfigured,
    TtsUnavailable,
    create_tts_provider,
)

LOG = logging.getLogger("mangaexplainer")

AUDIO_JSON = "page_{0:03d}_scene_{1:03d}.json"
AUDIO_WAV = "page_{0:03d}_scene_{1:03d}.wav"


def validate_page_number(page):
    if not isinstance(page, int) or page < 1:
        raise ValueError("page must be a positive integer")


def validate_scene_number(scene):
    if not isinstance(scene, int) or scene < 1:
        raise ValueError("scene must be a positive integer")


def audio_wav_path(cfg, page, scene):
    return Path(cfg.output.audio_dir) / AUDIO_WAV.format(page, scene)


def audio_manifest_path(cfg, page, scene):
    return Path(cfg.output.audio_dir) / AUDIO_JSON.format(page, scene)


def load_script(cfg, page, scene):
    """Read the scene's script JSON; raises TtsError on any problem."""
    path = Path(cfg.output.script_dir) / f"page_{page:03d}_scene_{scene:03d}.json"
    if not path.is_file():
        raise TtsError(
            f"script file not found: {path} (run the 'script' stage first)"
        )
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle), path
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        raise TtsError(f"invalid script file {path}: {exc}") from None


def _validate_segment(segment, index):
    if not isinstance(segment, dict) or not str(segment.get("text") or "").strip():
        raise TtsError(f"segment #{index}: text is empty")
    seconds = segment.get("estimated_seconds")
    try:
        ok = float(seconds) > 0
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise TtsError(
            f"segment #{index}: estimated_seconds must be positive, got {seconds!r}"
        )
    return segment


def _read_frames(path):
    with wave.open(str(path), "rb") as handle:
        return handle.readframes(handle.getnframes())


def _write_combined(path, frames, rate):
    out = wave.open(str(path), "wb")
    out.setnchannels(1)
    out.setsampwidth(2)
    out.setframerate(int(rate or 22050))
    out.writeframes(frames)
    out.close()


class AudioGenerator:
    def __init__(self, cfg, provider=None):
        self.cfg = cfg
        self.provider = provider
        self._release_on_run = provider is None

    def run_scene(self, page, scene, state, force=False):
        try:
            return self._run(page, scene, state, force)
        except Exception:
            gc.collect()
            raise

    def _run(self, page, scene, state, force):
        try:
            validate_page_number(page)
            validate_scene_number(scene)
            key = f"page_{page:03d}_scene_{scene:03d}"
            cfg = self.cfg

            doc, script_path = load_script(cfg, page, scene)
            segments = doc.get("segments") if isinstance(doc, dict) else None
            if not isinstance(segments, list) or not segments:
                return self._error(
                    page, scene,
                    f"script {script_path} has no segments - re-run the script stage",
                )
            for index, segment in enumerate(segments, 1):
                _validate_segment(segment, index)

            out_wav = audio_wav_path(cfg, page, scene)
            out_manifest = audio_manifest_path(cfg, page, scene)
            out_wav.parent.mkdir(parents=True, exist_ok=True)
            out_manifest.parent.mkdir(parents=True, exist_ok=True)

            if (
                state
                and not force
                and state.item_done(key, "audio_completed")
                and out_wav.is_file()
            ):
                return self._skip(page, scene, out_wav)

            provider = self._resolve_provider(cfg, page, scene)
            if isinstance(provider, dict):
                return provider

            try:
                try:
                    rendered = self._synthesize(provider, segments)
                    rate = int(provider.sample_rate)
                    frames = b"".join(_read_frames(path) for _, path in rendered)
                    _write_combined(out_wav, frames, rate)
                finally:
                    if provider is not None and self._release_on_run:
                        try:
                            provider.release()
                        except Exception:
                            pass
                    gc.collect()

                manifest = self._manifest(page, scene, doc, provider, rate, rendered)
                self._write_manifest(out_manifest, manifest)
                if state:
                    state.mark_item_done(key, "audio_completed")

                total_ms = manifest["total_duration_ms"]
                LOG.info(
                    "page %s scene %s audio: %d segment(s) via %s -> %s (%.1f s)",
                    page, scene, len(segments), provider.name, out_wav, total_ms / 1000.0,
                )
                return {
                    "result": "ok",
                    "page": page,
                    "scene": scene,
                    "scene_id": doc.get("scene_id"),
                    "engine": provider.name,
                    "audio_file": str(out_wav),
                    "manifest_file": str(out_manifest),
                    "segment_count": len(segments),
                    "total_duration_ms": total_ms,
                    "sample_rate": rate,
                    "segments": manifest["segments"],
                }
            except MemoryError:
                return self._error(
                    page, scene, "insufficient memory during TTS synthesis"
                )
            except TtsUnavailable as exc:
                return self._error(page, scene, f"TTS unavailable: {exc}")
            except TtsNotConfigured as exc:
                return self._error(page, scene, str(exc))
            except TtsError as exc:
                return self._error(page, scene, f"TTS failed: {exc}")
            except Exception as exc:
                LOG.exception("audio generation failure")
                return self._error(page, scene, f"audio error: {exc}")
        except TtsError as exc:
            return self._error(page, scene, str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            LOG.exception("audio orchestration failure")
            return self._error(page, scene, f"audio error: {exc}")
        finally:
            gc.collect()

    # -------------------------------------------------------------- helpers

    def _resolve_provider(self, cfg, page, scene):
        if self.provider is not None:
            return self.provider
        try:
            return create_tts_provider(cfg)
        except TtsNotConfigured as exc:
            return self._error(page, scene, str(exc))
        except TtsError as exc:
            return self._error(page, scene, f"TTS error: {exc}")

    def _synthesize(self, provider, segments):
        """Render every segment to a temp WAV; returns [(seg, path), ...]."""
        rendered = []
        try:
            for segment in segments:
                seconds = float(segment["estimated_seconds"])
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                    temp = Path(handle.name)
                try:
                    provider.synth(
                        str(segment["text"]),
                        temp,
                        target_seconds=seconds,
                        speaker=segment.get("speaker") if segment.get("type") == "dialogue" else None,
                    )
                except Exception:
                    temp.unlink(missing_ok=True)
                    raise
                rendered.append((segment, temp))
            return rendered
        except Exception:
            for _, temp in rendered:
                temp.unlink(missing_ok=True)
            raise

    def _manifest(self, page, scene, doc, provider, rate, rendered):
        entries = []
        offset_ms = 0
        for segment, temp in rendered:
            duration_ms = int(round(_wav_ms(temp)))
            temp.unlink(missing_ok=True)
            entries.append({
                "segment_id": segment["segment_id"],
                "type": segment.get("type", "narration"),
                "text": segment["text"],
                "speaker": segment.get("speaker") if segment.get("type") == "dialogue" else None,
                "estimated_seconds": float(segment["estimated_seconds"]),
                "start_ms": offset_ms,
                "end_ms": offset_ms + duration_ms,
                "duration_ms": duration_ms,
            })
            offset_ms += duration_ms
        return {
            "scene_id": doc.get("scene_id"),
            "page": page,
            "scene": scene,
            "engine": provider.name,
            "sample_rate": rate,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "segments": entries,
            "total_duration_ms": offset_ms,
        }

    def _write_manifest(self, path, manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
            handle.flush()
            __import__("os").fsync(handle.fileno())
        __import__("os").replace(tmp, path)

    def _skip(self, page, scene, out_wav):
        return {
            "result": "skipped", "page": page, "scene": scene,
            "audio_file": str(out_wav),
        }

    def _error(self, page, scene, message):
        LOG.error("page %s scene %s audio failed: %s", page, scene, message)
        return {
            "result": "error", "page": page, "scene": scene, "message": message,
        }


def _wav_ms(path):
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate() or 22050
    return frames / rate * 1000.0