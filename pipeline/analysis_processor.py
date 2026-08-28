"""One-panel VLM analysis orchestration.

Sequential low-RAM flow per panel:

    panel image -> load (validate, free) -> build prompt (with OCR context)
    -> resolve VLM provider -> inference -> release provider -> parse &
    validate JSON -> save analysis/<panel>.json -> checkpoint vlm_completed
    -> gc

Only ONE image and ONE provider are ever held at a time - never OCR + VLM
simultaneously, never a whole page/volume. A failed analysis never creates a
checkpoint, and malformed responses are salvaged into logs/vlm/<name>_raw.txt.
"""
import gc
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import cv2

from .prompts import build_analysis_prompt
from .vlm_provider import (
    VLMNotConfigured,
    VLMFailure,
    VLMProviderError,
    VLMTimeout,
    VLMUnavailable,
    create_vlm_provider,
    extract_json,
)

LOG = logging.getLogger("mangaexplainer")

VLM_COMPLETED = "vlm_completed"


def _to_str(value):
    if value is None:
        return "unknown"
    if isinstance(value, str):
        return value if value.strip() else "unknown"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _to_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [_to_str(v) for v in value]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [_to_str(value)]


def _character(value):
    if not isinstance(value, dict):
        value = {}
    return {
        "name": _to_str(value.get("name")),
        "description": _to_str(value.get("description")),
        "action": _to_str(value.get("action")),
        "emotion": _to_str(value.get("emotion")),
    }


def _to_confidence(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def sanitize_analysis(data):
    """Coerce a parsed VLM response into the required analysis schema."""
    if not isinstance(data, dict):
        return None
    characters = data.get("characters")
    if not isinstance(characters, list):
        characters = [_character(characters)] if characters else []
    try:
        characters = [_character(c) for c in characters]
    except Exception:
        characters = []
    return {
        "characters": characters,
        "environment": _to_str(data.get("environment")),
        "actions": _to_list(data.get("actions")),
        "objects": _to_list(data.get("objects")),
        "visual_effects": _to_list(data.get("visual_effects")),
        "important_event": _to_str(data.get("important_event")),
        "composition": _to_str(data.get("composition")),
        "story_relevance": _to_str(data.get("story_relevance")),
        "confidence": _to_confidence(data.get("confidence")),
    }


class AnalysisProcessor:
    def __init__(self, cfg, provider=None):
        self.cfg = cfg
        self.provider = provider
        self._release_on_run = provider is None

    def run_panel(self, page, panel, state, force=False):
        try:
            cfg = self.cfg
            if page < 1 or panel < 1:
                return self._error(page, panel, "invalid page/panel number")
            self._ensure_dirs(cfg)

            key = f"page_{page:03d}_panel_{panel:03d}"
            panel_dir = Path(cfg.output.panels_dir) / f"page_{page:03d}"
            panel_file = panel_dir / f"panel_{panel:03d}.jpg"
            out_file = Path(cfg.output.analysis_dir) / f"{key}.json"

            if not panel_file.is_file():
                return self._error(
                    page, panel, "panel image not found", source=str(panel_file)
                )
            if state and not force and self._done(state, key):
                return self._skip(page, panel, str(out_file))

            provider = self._resolve_provider(cfg, page, panel)
            if isinstance(provider, dict):
                return provider

            image = self._load_image(panel_file)
            if image is None:
                return self._error(page, panel, "invalid panel image")
            del image
            gc.collect()

            ocr_context = self._load_ocr_context(cfg, page, panel)
            prompt = build_analysis_prompt(ocr_context=ocr_context)

            model_name = getattr(provider, "model", "") or provider.name
            try:
                raw = provider.analyze_image(
                    str(panel_file), prompt, timeout=int(cfg.vlm.timeout_seconds)
                )
            except MemoryError:
                return self._error(page, panel, "insufficient memory during VLM inference")
            except VLMTimeout as exc:
                return self._error(page, panel, f"VLM timed out: {exc}")
            except VLMUnavailable as exc:
                return self._error(page, panel, f"VLM unavailable: {exc}")
            except VLMFailure as exc:
                return self._error(page, panel, f"VLM inference failed: {exc}")
            except VLMProviderError as exc:
                return self._error(page, panel, f"VLM error: {exc}")
            except Exception as exc:
                LOG.warning("unexpected VLM error", exc_info=True)
                return self._error(page, panel, f"VLM inference failed: {exc}")
            finally:
                if provider is not None and self._release_on_run:
                    try:
                        provider.release()
                    except Exception:
                        pass
                gc.collect()

            data = extract_json(raw)
            if data is None:
                raw_name = self._save_raw(cfg, key, raw)
                return self._error(
                    page,
                    panel,
                    "VLM returned no valid JSON; raw response saved for inspection",
                    raw=raw_name,
                )
            analysis = sanitize_analysis(data)
            if analysis is None:
                raw_name = self._save_raw(cfg, key, raw)
                return self._error(
                    page,
                    panel,
                    "VLM JSON not in expected shape; raw response saved for inspection",
                    raw=raw_name,
                )

            self._write_analysis(
                out_file, page, panel, str(panel_file),
                provider.name, model_name, analysis,
            )
            if state:
                state.mark_item_done(key, VLM_COMPLETED)
            return self._ok(
                page, panel, provider.name, model_name, analysis, str(out_file)
            )

        except Exception as exc:
            LOG.exception("analysis orchestration failure")
            return self._error(page, panel, f"analysis error: {exc}")
        finally:
            gc.collect()

    # ------------------------------------------------------------ providers
    def _resolve_provider(self, cfg, page, panel):
        if self.provider is not None:
            return self.provider
        if not bool(cfg.vlm.enabled):
            return self._error(
                page,
                panel,
                "VLM disabled: set vlm.enabled=true and vlm.model in config/config.yaml",
            )
        try:
            return create_vlm_provider(cfg)
        except VLMNotConfigured as exc:
            return self._error(page, panel, str(exc))
        except VLMProviderError as exc:
            return self._error(page, panel, str(exc))

    # ---------------------------------------------------------------- output
    def _ok(self, page, panel, provider_name, model_name, analysis, out_file):
        LOG.info(
            "analyzed page %s panel %s via %s (%s) -> %s",
            page, panel, provider_name, model_name, out_file,
        )
        return {
            "result": "ok",
            "page": page,
            "panel": panel,
            "provider": provider_name,
            "model": model_name,
            "output": out_file,
            "confidence": analysis.get("confidence"),
            "important_event": analysis.get("important_event"),
            "characters": len(analysis.get("characters", [])),
        }

    def _skip(self, page, panel, out_file):
        LOG.info("page %s panel %s already analyzed (skip) -> %s", page, panel, out_file)
        return {
            "result": "skipped",
            "page": page,
            "panel": panel,
            "output": out_file,
        }

    def _error(self, page, panel, message, **extra):
        LOG.error("page %s panel %s analysis failed: %s", page, panel, message)
        result = {
            "result": "error",
            "page": page,
            "panel": panel,
            "message": message,
        }
        result.update(extra)
        return result

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _ensure_dirs(cfg):
        Path(cfg.output.analysis_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.logging.log_dir, "vlm").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _done(state, key):
        try:
            return state.item_done(key, VLM_COMPLETED)
        except (AttributeError, TypeError):
            return False

    @staticmethod
    def _load_image(panel_file):
        try:
            return cv2.imread(str(panel_file), cv2.IMREAD_COLOR)
        except Exception:
            return None

    @staticmethod
    def _load_ocr_context(cfg, page, panel):
        key = f"page_{page:03d}_panel_{panel:03d}"
        ocr_file = Path(cfg.output.ocr_dir) / f"{key}.json"
        try:
            data = json.loads(ocr_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        text = (data.get("combined_text") or "").strip()
        return text or None

    @staticmethod
    def _save_raw(cfg, key, raw):
        raw_dir = Path(cfg.logging.log_dir) / "vlm"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_file = raw_dir / f"{key}_raw.txt"
        raw_file.write_text(str(raw), encoding="utf-8")
        return str(raw_file)

    @staticmethod
    def _write_analysis(out_file, page, panel, panel_file, provider_name, model_name, analysis):
        doc = {
            "page": page,
            "panel": panel,
            "image": str(panel_file),
            "provider": provider_name,
            "model": model_name,
            "analyzed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "analysis": analysis,
        }
        tmp = out_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out_file)