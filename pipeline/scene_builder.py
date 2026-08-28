"""Rule-based scene reconstruction from the page knowledge layer.

Groups panels in READING ORDER into logical scenes. Lightweight and fully
deterministic: no VLM/LLM is used - every signal comes from the structured
knowledge records (characters, location, actions, events, OCR text).

Between every consecutive panel pair a BOUNDARY SCORE in ~[0, 1] is computed:

    + location_change, character_change, event_change, narrative_transition
    - character/location/action/dialogue/event continuity

If score >= scenes.threshold (config, not hard-coded) a new scene starts.
Unknown/missing signals are strictly neutral - sparse data never fabricates
scene breaks, and never collapses genuinely distinct scenes either (only
what is visible is used). Scenes may eventually span pages: ids are assigned
from a caller-supplied offset so page N+1 can continue numbering.

Output: scenes/<page>_scenes.json, scenes/<page>_scene_debug.json,
scenes/<page>_scene_debug.txt. The page knowledge file is updated in place
(panel -> scene_id / previous_panel / next_panel), never rewriting unrelated
panels. No images, no VLM, no LLM are ever loaded.
"""
import gc
import json
import logging
import os
from pathlib import Path

from .knowledge import (
    KnowledgeError,
    knowledge_path,
    load_page_knowledge,
    validate_page_number,
)

LOG = logging.getLogger("mangaexplainer")

SCENES_FILENAME = "page_{:03d}_scenes.json"
DEBUG_FILENAME = "page_{:03d}_scene_debug.json"
DEBUG_TEXT_FILENAME = "page_{:03d}_scene_debug.txt"


class SceneError(Exception):
    """Raised on invalid/missing scene input (never silently repaired)."""


# ------------------------------------------------------------- signal access
# Each helper returns None when the signal is unknown, so missing/sparse data
# is strictly neutral in the scoring.
def _characters(record):
    visual = record.get("visual")
    if not isinstance(visual, dict):
        return None
    names = []
    for char in visual.get("characters") or []:
        if not isinstance(char, dict):
            continue
        name = (char.get("name") or "").strip()
        if name and name.lower() != "unknown":
            names.append(name)
    return sorted(set(names)) or None


def _location(record):
    visual = record.get("visual")
    if not isinstance(visual, dict):
        return None
    loc = (visual.get("environment") or "").strip()
    if loc and loc.lower() != "unknown":
        return loc
    return None


def _actions(record):
    visual = record.get("visual")
    if not isinstance(visual, dict):
        return None
    actions = []
    for action in visual.get("actions") or []:
        value = (str(action) if not isinstance(action, str) else action).strip()
        if value and value.lower() != "unknown":
            actions.append(value)
    return sorted(set(actions)) or None


def _event(record):
    visual = record.get("visual")
    if not isinstance(visual, dict):
        return None
    event = (visual.get("important_event") or "").strip()
    return event if event and event.lower() != "unknown" else None


def _ocr_text(record):
    ocr = record.get("ocr")
    if not isinstance(ocr, dict):
        return None
    text = (ocr.get("text") or "").strip()
    return text or None


def _confidence(record):
    visual = record.get("visual")
    if not isinstance(visual, dict):
        return None
    try:
        value = float(visual.get("confidence"))
    except (TypeError, ValueError):
        return None
    if 0.0 <= value <= 1.0:
        return value
    return None


# -------------------------------------------------------------- boundary score
def _intersection(set_a, set_b):
    if not set_a or not set_b:
        return None
    merge = set(set_a) | set(set_b)
    overlap = set(set_a) & set(set_b)
    return len(overlap) / len(merge) if merge else 0.0


def _transition_flag(prev, cur, keywords):
    prev_text = " ".join(filter(None, (_event(prev) or "", _ocr_text(prev) or "")))
    cur_text = " ".join(filter(None, (_event(cur) or "", _ocr_text(cur) or "")))
    haystacks = (prev_text + " " + cur_text).lower()
    for keyword in keywords:
        if keyword.lower() in haystacks:
            return 1.0
    return 0.0


def boundary_score(prev, cur, scenes_cfg):
    """Structural continuity/discontinuity between two ordered panels."""
    weights = scenes_cfg.weights
    continuity = scenes_cfg.continuity
    keywords = scenes_cfg.transition_keywords or []

    score = 0.0

    loc_prev, loc_cur = _location(prev), _location(cur)
    if loc_prev and loc_cur:
        if loc_prev != loc_cur:
            score += float(weights.location_change)
        else:
            score -= float(continuity.location)
    else:
        if bool(loc_prev) != bool(loc_cur):
            pass  # sparse -> neutral

    char_similarity = _intersection(_characters(prev), _characters(cur))
    if char_similarity is not None:
        score -= float(continuity.character) * char_similarity
        if char_similarity == 0.0:
            score += float(weights.character_change)

    event_prev, event_cur = _event(prev), _event(cur)
    if event_prev and event_cur:
        if event_prev != event_cur:
            score += float(weights.event_change)
        else:
            score -= float(continuity.event)
    else:
        if event_prev and not event_cur and len(keywords):
            score += float(weights.narrative_transition) * _transition_flag(
                prev, cur, keywords
            )

    action_similarity = _intersection(_actions(prev), _actions(cur))
    if action_similarity is not None:
        score -= float(continuity.action) * action_similarity

    prev_dialogue = _ocr_text(prev) is not None
    cur_dialogue = _ocr_text(cur) is not None
    if prev_dialogue and cur_dialogue:
        score -= float(continuity.dialogue)

    return max(0.0, min(1.0, score))


def build_scenes(panels, scenes_cfg, start_index=0):
    """Group ordered panel records (knowledge records) into scene dicts."""
    scenes = []
    current = None
    boundaries = []
    threshold = float(scenes_cfg.threshold)
    for panel in panels:
        if not isinstance(panel, dict) or not isinstance(panel.get("panel_id"), str):
            raise SceneError(f"invalid panel record: {panel!r}")
        if current is None:
            current = {"panels": [panel]}
        else:
            prev = current["panels"][-1]
            score = boundary_score(prev, panel, scenes_cfg)
            decision = score >= threshold
            boundaries.append(
                {
                    "from": prev["panel_id"],
                    "to": panel["panel_id"],
                    "score": round(score, 4),
                    "boundary": decision,
                }
            )
            if decision:
                scenes.append(_finalize_scene(current, scenes_cfg, start_index + len(scenes)))
                current = {"panels": [panel]}
            else:
                current["panels"].append(panel)
    if current is not None:
        scenes.append(_finalize_scene(current, scenes_cfg, start_index + len(scenes)))
    return scenes, boundaries


def _finalize_scene(acc, scenes_cfg, index):
    panels = acc["panels"]
    characters, locations, events, confidences = [], [], [], []
    for record in panels:
        chars = _characters(record)
        if chars:
            characters.extend(chars)
        loc = _location(record)
        if loc:
            locations.append(loc)
        event = _event(record)
        if event:
            events.append(event)
        confidence = _confidence(record)
        if confidence is not None:
            confidences.append(confidence)
    characters = sorted(set(characters))
    locations = sorted(set(locations))
    events = sorted(set(events))
    return {
        "scene_id": f"scene_{index + 1:03d}",
        "page_start": min((record.get("page") or 1) for record in panels),
        "page_end": max((record.get("page") or 1) for record in panels),
        "panel_ids": [record["panel_id"] for record in panels],
        "characters": characters,
        "locations": locations,
        "events": events,
        "summary": _factual_summary(characters, locations, events, scenes_cfg),
        "confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
    }


def _factual_summary(characters, locations, events, scenes_cfg):
    limit = int(getattr(scenes_cfg, "summary_max_items", 6))
    parts = []
    if characters:
        parts.append("Characters: " + ", ".join(characters[:limit]))
    if locations:
        parts.append("Location(s): " + ", ".join(locations[:limit]))
    if events:
        parts.append("Event(s): " + ", ".join(events[:limit]))
    return " ".join(parts)


# ------------------------------------------------------------------ processor
class SceneProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.scenes_dir = Path(cfg.output.scenes_dir)

    def run_page(self, page, state, force=False):
        scenes_cfg = self.cfg.scenes
        try:
            validate_page_number(page)
            key = f"page_{page:03d}"
            knowledge_file = knowledge_path(self.cfg, page)
            scenes_file = self.scenes_dir / SCENES_FILENAME.format(page)

            if not knowledge_file.is_file():
                return self._error(
                    page,
                    f"missing required data: page knowledge file {knowledge_file} "
                    "(run: python main.py knowledge --page %s)" % page,
                )
            try:
                doc = load_page_knowledge(self.cfg, page)
            except KnowledgeError as exc:
                return self._error(page, f"invalid page knowledge: {exc}")

            if not force and state is not None:
                try:
                    if state.item_done(key, "scenes_completed") and scenes_file.is_file():
                        return self._skip(page, scenes_file)
                except (AttributeError, TypeError):
                    pass

            records = doc.get("panels", [])
            if not isinstance(records, list):
                return self._error(page, "invalid page knowledge: 'panels' is not a list")

            try:
                scenes, boundaries = build_scenes(records, scenes_cfg)
            except SceneError as exc:
                return self._error(page, f"invalid panel data: {exc}")

            self.scenes_dir.mkdir(parents=True, exist_ok=True)
            self._write_scenes(scenes_file, page, scenes)
            self._write_debug(page, scenes, boundaries)
            self._update_knowledge_links(knowledge_file, doc, records, scenes)

            if state is not None:
                state.mark_item_done(key, "scenes_completed")

            LOG.info(
                "page %s scenes: %d scene(s), %d panel(s)",
                page, len(scenes), len(records),
            )
            return {
                "result": "ok",
                "page": page,
                "scene_count": len(scenes),
                "panels": [scene["panel_ids"] for scene in scenes],
                "scenes_file": str(scenes_file),
                "boundaries": boundaries,
                "missing": self._unanalysed_missing(records),
            }
        except Exception as exc:
            LOG.exception("scene construction failed")
            return self._error(page, f"scene error: {exc}")
        finally:
            gc.collect()

    @staticmethod
    def _unanalysed_missing(records):
        missing = []
        for record in records:
            if record.get("visual") is None:
                missing.append(f"VLM analysis for {record.get('panel_id')}")
            if record.get("ocr") is None:
                missing.append(f"OCR for {record.get('panel_id')}")
        return missing

    def _write_scenes(self, scenes_file, page, scenes):
        payload = {
            "page": page,
            "scene_count": len(scenes),
            "scenes": scenes,
        }
        self._atomic(scenes_file, payload)

    def _write_debug(self, page, scenes, boundaries):
        debug = {
            "page": page,
            "scene_count": len(scenes),
            "boundary_decisions": boundaries,
            "scenes": scenes,
        }
        self._atomic(self.scenes_dir / DEBUG_FILENAME.format(page), debug)
        lines = []
        for scene in scenes:
            lines.append(f"SCENE {scene['scene_id'].upper()}")
            lines.append(f"  Panels: {', '.join(scene['panel_ids'])}")
            lines.append(f"  Characters: {', '.join(scene['characters']) or '-'}")
            lines.append(f"  Location: {', '.join(scene['locations']) or '-'}")
            lines.append(f"  Events: {', '.join(scene['events']) or '-'}")
            if scene["summary"]:
                lines.append(f"  Summary: {scene['summary']}")
            lines.append(f"  Confidence: {scene['confidence']}")
            lines.append("")
        lines.append("Between-panel boundary decisions (score >= threshold starts a scene):")
        for decision in boundaries:
            marker = "SPLIT" if decision["boundary"] else "same "
            lines.append(
                f"  {decision['from']} -> {decision['to']}: "
                f"{marker} (score {decision['score']})"
            )
        (self.scenes_dir / DEBUG_TEXT_FILENAME.format(page)).write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def _update_knowledge_links(self, knowledge_file, doc, records, scenes):
        scene_by_panel = {
            pid: scene["scene_id"]
            for scene in scenes
            for pid in scene["panel_ids"]
        }
        updated = 0
        for index, record in enumerate(records):
            pid = record["panel_id"]
            expected = {
                "scene_id": scene_by_panel.get(pid),
                "previous_panel": records[index - 1]["panel_id"] if index > 0 else None,
                "next_panel": records[index + 1]["panel_id"]
                if index < len(records) - 1
                else None,
            }
            if (
                record.get("scene_id") != expected["scene_id"]
                or record.get("previous_panel") != expected["previous_panel"]
                or record.get("next_panel") != expected["next_panel"]
            ):
                record.update(expected)
                updated += 1
        if updated or not scene_by_panel:
            tmp = knowledge_file.with_name(knowledge_file.name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(doc, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, knowledge_file)

    @staticmethod
    def _atomic(path, payload):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    def _skip(self, page, scenes_file):
        LOG.info("page %s scenes already built (skip) -> %s", page, scenes_file)
        return {
            "result": "skipped",
            "page": page,
            "scenes_file": str(scenes_file),
        }

    def _error(self, page, message):
        LOG.error("page %s scene construction failed: %s", page, message)
        return {
            "result": "error",
            "page": page,
            "message": message,
        }