"""LLM-based explanation script for ONE reconstructed scene (write_script).

Reads the page's scenes (scenes/<page>_scenes.json), selects EXACTLY ONE scene
by number, prepares a compact context from that scene + the page knowledge
(dialogue + per-panel facts), and asks the configured LLM provider for a
segmented narration script. No audio, no video, no images - TTS/screens come
later.

Low-RAM contract (4 GB machine):
- ONE scene at a time: the scene's JSON, its panels and the knowledge page are
  the only data ever held; never the whole page/volume, never all scenes.
- one provider held at a time - never VLM + LLM + TTS simultaneously.
- provider.release() + gc.collect() after every generation.

Outputs (script/ directory):
    page_001_scene_001.json   <- structured script (segments + visual intent)
    page_001_scene_001.txt    <- plain narration/dialogue for TTS

Checkpoint: page_001_scene_001: script_completed (skip unless --force).

Validation happens BEFORE saving (see VALIDATION RULES): invalid LLM output is
reported, the raw response is salvaged to logs/llm/<key>_raw.txt, and the
checkpoint is never written.
"""
import gc
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from .knowledge import KnowledgeError, load_page_knowledge, validate_page_number
from .llm_provider import (
    LLMFailure,
    LLMNotConfigured,
    LLMProviderError,
    LLMTimeout,
    LLMUnavailable,
    clean_text,
    create_llm_provider,
)
from .prompts import build_script_prompt
from .vlm_provider import extract_json

LOG = logging.getLogger("mangaexplainer")

SCRIPT_JSON_FILENAME = "page_{:03d}_scene_{:03d}.json"
SCRIPT_TXT_FILENAME = "page_{:03d}_scene_{:03d}.txt"

VISUAL_INTENTS = {
    "full_panel", "smart_crop", "character_closeup", "face_closeup",
    "object_closeup", "action_crop", "multi_panel",
}
CAMERAS = {
    "static", "slow_zoom_in", "slow_zoom_out",
    "pan_left", "pan_right", "pan_up", "pan_down",
}
SEGMENT_TYPES = {"narration", "dialogue"}
DEFAULT_VISUAL_INTENT = "full_panel"
DEFAULT_CAMERA = "static"

VALIDATION_RULES = (
    "scene id exists; every referenced panel id exists; every segment has "
    "text; estimated duration > 0; visual intent and camera are valid"
)


class ScriptError(Exception):
    """Raised on missing/invalid input (never silently repaired)."""


class SegmentError(Exception):
    """Raised when a normalized segment fails validation."""


def json_path(cfg, page, scene):
    return Path(cfg.output.script_dir) / SCRIPT_JSON_FILENAME.format(page, scene)


def txt_path(cfg, page, scene):
    return Path(cfg.output.script_dir) / SCRIPT_TXT_FILENAME.format(page, scene)


# ------------------------------------------------------------- scene loading


def load_scenes(cfg, page):
    path = Path(cfg.output.scenes_dir) / f"page_{page:03d}_scenes.json"
    if not path.is_file():
        raise ScriptError(
            f"missing required data: scenes file {path} "
            "(run: python main.py scenes --page %s)" % page
        )
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ScriptError(f"invalid scenes file: {exc}") from None
    if not isinstance(doc.get("scenes"), list):
        raise ScriptError("invalid scenes file: 'scenes' is not a list")
    return doc


def select_scene(doc, scene):
    """Return the scene dict for scene number N (1-based) or raise ScriptError."""
    if not isinstance(scene, int) or isinstance(scene, bool) or scene < 1:
        raise ScriptError(f"invalid scene number {scene!r}: must be an int >= 1")
    scenes = doc.get("scenes", [])
    if scene > len(scenes):
        raise ScriptError(
            f"scene {scene} not found: scenes file has {len(scenes)} scene(s)"
        )
    scene_doc = scenes[scene - 1]
    if not isinstance(scene_doc, dict) or not isinstance(scene_doc.get("scene_id"), str):
        raise ScriptError(f"scene {scene} is not a valid scene object")
    panels = scene_doc.get("panel_ids")
    if not isinstance(panels, list) or not panels:
        raise ScriptError(f"scene {scene} has no panels")
    return scene_doc


# ------------------------------------------------------------ panel context


def _panel_context(knowledge):
    """Compact {panel_id: {characters, event, dialogue}} from the knowledge layer."""
    context = {}
    for record in knowledge.get("panels", []):
        pid = record["panel_id"]
        characters, event, dialogue = [], None, ""
        visual = record.get("visual")
        if isinstance(visual, dict):
            for char in visual.get("characters") or []:
                if not isinstance(char, dict):
                    continue
                name = str(char.get("name") or "").strip()
                if name and name.lower() != "unknown":
                    characters.append(name)
            value = str(visual.get("important_event") or "").strip()
            event = value if value and value.lower() != "unknown" else None
        ocr = record.get("ocr")
        if isinstance(ocr, dict):
            dialogue = str(ocr.get("text") or "").strip()
        context[pid] = {
            "characters": sorted(set(characters)),
            "event": event,
            "dialogue": dialogue,
        }
    return context


# ------------------------------------------------------------ segmentation


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def normalize_segment(raw, scene_panel_ids, default_importance=0.0):
    """Validate + normalize ONE LLM segment dict; raises SegmentError.

    - text: required, non-empty after cleaning
    - panel_ids: required list, every id must exist in the scene
    - estimated_seconds: required, positive float
    - visual_intent / camera: required to be valid; missing -> safe default
    - importance: optional float clamped to [0, 1] (default: scene confidence)
    """
    if not isinstance(raw, dict):
        raise SegmentError("segment is not an object")
    text = clean_text(raw.get("text"))
    if not text:
        raise SegmentError("segment has no text")

    panel_ids = raw.get("panel_ids")
    if not isinstance(panel_ids, list):
        raise SegmentError("segment panel_ids must be a list")
    panels = []
    for pid in panel_ids:
        if not isinstance(pid, str) or pid not in scene_panel_ids:
            raise SegmentError(f"segment references unknown panel id {pid!r}")
        if pid not in panels:
            panels.append(pid)

    try:
        seconds = float(raw.get("estimated_seconds"))
    except (TypeError, ValueError):
        raise SegmentError("segment estimated_seconds must be a number") from None
    if not seconds > 0:
        raise SegmentError("segment estimated_seconds must be positive")
    seconds = round(seconds, 1)

    intent = raw.get("visual_intent", DEFAULT_VISUAL_INTENT)
    if not isinstance(intent, str) or intent not in VISUAL_INTENTS:
        raise SegmentError(
            f"invalid visual_intent {intent!r} "
            f"(allowed: {', '.join(sorted(VISUAL_INTENTS))})"
        )
    camera = raw.get("camera", DEFAULT_CAMERA)
    if not isinstance(camera, str) or camera not in CAMERAS:
        raise SegmentError(
            f"invalid camera {camera!r} (allowed: {', '.join(sorted(CAMERAS))})"
        )

    importance = raw.get("importance", default_importance)
    try:
        importance = _clamp(float(importance))
    except (TypeError, ValueError):
        raise SegmentError("segment importance must be a number") from None

    seg_type = raw.get("type", "narration")
    if not isinstance(seg_type, str) or seg_type not in SEGMENT_TYPES:
        raise SegmentError(
            f"invalid segment type {seg_type!r} (allowed: narration, dialogue)"
        )

    cleaned = {
        "segment_id": "",
        "type": seg_type,
        "text": text,
        "panel_ids": panels,
        "estimated_seconds": seconds,
        "visual_intent": intent,
        "camera": camera,
        "importance": round(importance, 3),
    }
    if seg_type == "dialogue":
        speaker = str(raw.get("speaker") or "unknown").strip() or "unknown"
        cleaned["speaker"] = speaker
    return cleaned


def normalize_segments(raw_segments, scene_panel_ids, default_importance=0.0):
    """Normalize + id all segments in LLM order; raises SegmentError."""
    if not isinstance(raw_segments, list) or not raw_segments:
        raise SegmentError("LLM returned no segments")
    segments = []
    for index, raw in enumerate(raw_segments, 1):
        segment = normalize_segment(raw, scene_panel_ids, default_importance)
        segment["segment_id"] = f"seg_{index:03d}"
        segments.append(segment)
    return segments


def build_txt(segments):
    """Plain narration/dialogue in reading order, formatted for TTS."""
    blocks = []
    for segment in segments:
        if segment["type"] == "dialogue":
            speaker = segment.get("speaker") or "unknown"
            label = (
                f"Dialogue ({speaker}):"
                if speaker.lower() not in ("", "unknown")
                else "Dialogue:"
            )
        else:
            label = "Narrator:"
        blocks.append(label + "\n" + segment["text"])
    return "\n\n".join(blocks) + "\n"


# ---------------------------------------------------------------- processor


class ScriptGenerator:
    def __init__(self, cfg, provider=None):
        self.cfg = cfg
        self.provider = provider
        self._release_on_run = provider is None

    def run_scene(self, page, scene, state, force=False):
        try:
            validate_page_number(page)
            key = f"page_{page:03d}_scene_{scene:03d}"
            cfg = self.cfg
            out_json = json_path(cfg, page, scene)
            out_txt = txt_path(cfg, page, scene)

            doc = load_scenes(cfg, page)
            scene_doc = select_scene(doc, scene)

            knowledge = self._load_knowledge(page)
            if isinstance(knowledge, dict) and knowledge.get("result") == "error":
                return knowledge
            context = _panel_context(knowledge)
            dialogue = {pid: info["dialogue"] for pid, info in context.items()}

            if state and not force and self._done(state, key) and out_json.is_file():
                return self._skip(page, scene, out_json)

            provider = self._resolve_provider(cfg, page, scene)
            if isinstance(provider, dict):
                return provider

            prompt = build_script_prompt(scene_doc, dialogue, context)

            # Durable project memory + recent-pages window + rich memory
            # archive (all optional): the narrator stays consistent across the
            # whole volume, and on resume / a new PDF it still remembers what
            # it already learned (incl. user corrections and story events).
            try:
                from .context_memory import manga_memory_block, script_context
                memory_block, window_block = script_context(cfg)
                rich_block = manga_memory_block(
                    cfg, page=page, task="narration", limit=8,
                    extra_text=" ".join(context.get(pid, {}).get("characters", [])
                                        for pid in (scene_doc.get("panel_ids") or []))
                    if context else None,
                )
                if memory_block or window_block or rich_block:
                    prompt = build_script_prompt(
                        scene_doc, dialogue, context,
                        memory_block=memory_block, window_block=window_block,
                        manga_memory_block=rich_block)
            except Exception:  # memory is best-effort; never fail scripting
                logging.getLogger(__name__).warning(
                    "script context memory unavailable; continuing without it",
                    exc_info=True,
                )
            raw = None
            try:
                raw = provider.generate(
                    prompt, timeout=int(cfg.llm.timeout_seconds)
                )
            except MemoryError:
                return self._error(page, scene, "insufficient memory during LLM narration")
            except LLMTimeout as exc:
                return self._error(page, scene, f"LLM timed out: {exc}")
            except LLMUnavailable as exc:
                return self._error(page, scene, f"LLM unavailable: {exc}")
            except LLMFailure as exc:
                return self._error(page, scene, f"LLM inference failed: {exc}")
            except LLMProviderError as exc:
                return self._error(page, scene, f"LLM error: {exc}")
            except Exception as exc:
                LOG.warning("unexpected LLM error", exc_info=True)
                return self._error(page, scene, f"LLM narration failed: {exc}")
            finally:
                if provider is not None and self._release_on_run:
                    try:
                        provider.release()
                    except Exception:
                        pass
                gc.collect()

            raw_ref = self._save_raw(cfg, key, raw)

            data = extract_json(raw)
            if data is None:
                return self._error(
                    page, scene,
                    "LLM returned no valid JSON; raw response saved for inspection",
                    raw=raw_ref,
                )
            raw_segments = data.get("segments") if isinstance(data, dict) else data
            try:
                segments = normalize_segments(
                    raw_segments,
                    set(scene_doc["panel_ids"]),
                    default_importance=float(scene_doc.get("confidence") or 0.0),
                )
            except SegmentError as exc:
                return self._error(
                    page, scene,
                    f"invalid LLM output: {exc}; raw response saved for inspection "
                    f"(validation rules: {VALIDATION_RULES})",
                    raw=raw_ref,
                )

            provider_name = provider.name
            model_name = getattr(provider, "model", "") or provider_name
            payload = {
                "scene_id": scene_doc["scene_id"],
                "page": page,
                "segments": segments,
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "provider": provider_name,
                "model": model_name,
            }
            self._write_json(out_json, payload)
            self._write_txt(out_txt, segments)
            if state:
                state.mark_item_done(key, "script_completed")

            referenced = []
            for segment in segments:
                for pid in segment["panel_ids"]:
                    if pid not in referenced:
                        referenced.append(pid)
            LOG.info(
                "page %s scene %s script: %d segment(s) via %s (%s) -> %s",
                page, scene, len(segments), provider_name, model_name, out_json,
            )
            return {
                "result": "ok",
                "page": page,
                "scene": scene,
                "scene_id": scene_doc["scene_id"],
                "script_json": str(out_json),
                "script_txt": str(out_txt),
                "provider": provider_name,
                "model": model_name,
                "segment_count": len(segments),
                "text_length": sum(len(s["text"]) for s in segments),
                "referenced_panels": referenced,
                "segments": [
                    {
                        "segment_id": s["segment_id"],
                        "type": s["type"],
                        "text": s["text"],
                        "estimated_seconds": s["estimated_seconds"],
                        "visual_intent": s["visual_intent"],
                        "camera": s["camera"],
                    }
                    for s in segments
                ],
            }
        except ScriptError as exc:
            return self._error(page, scene, str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            LOG.exception("script orchestration failure")
            return self._error(page, scene, f"script error: {exc}")
        finally:
            gc.collect()

    # ------------------------------------------------------------ providers
    def _resolve_provider(self, cfg, page, scene):
        if self.provider is not None:
            return self.provider
        if not bool(cfg.llm.enabled):
            return self._error(
                page, scene,
                "LLM disabled: set llm.enabled=true and llm.model in config/config.yaml",
            )
        try:
            return create_llm_provider(cfg)
        except LLMNotConfigured as exc:
            return self._error(page, scene, str(exc))
        except LLMProviderError as exc:
            return self._error(page, scene, str(exc))

    # ---------------------------------------------------------------- inputs
    def _load_knowledge(self, page):
        try:
            return load_page_knowledge(self.cfg, page)
        except KnowledgeError as exc:
            return self._error(
                page, 0,
                f"missing required data: page knowledge file for page {page} "
                f"(run: python main.py knowledge --page {page}) - {exc}",
            )

    # ---------------------------------------------------------------- writes
    @staticmethod
    def _write_json(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    @staticmethod
    def _write_txt(path, segments):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(build_txt(segments))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    def _save_raw(self, cfg, key, raw):
        raw_dir = Path(cfg.logging.log_dir) / "llm"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_file = raw_dir / f"{key}_raw.txt"
        raw_file.write_text(str(raw), encoding="utf-8")
        return str(raw_file)

    def _skip(self, page, scene, out_json):
        LOG.info("page %s scene %s script already written (skip) -> %s", page, scene, out_json)
        return {
            "result": "skipped",
            "page": page,
            "scene": scene,
            "script_json": str(out_json),
        }

    def _error(self, page, scene, message, **extra):
        LOG.error("page %s scene %s script failed: %s", page, scene, message)
        result = {
            "result": "error",
            "page": page,
            "scene": scene,
            "message": message,
        }
        result.update(extra)
        return result

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _done(state, key):
        try:
            return state.item_done(key, "script_completed")
        except (AttributeError, TypeError):
            return False