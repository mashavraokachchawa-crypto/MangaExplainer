"""Script -> panel visual timeline (plan_shots stage).

Turns the segmented narration of ONE scene into a deterministic visual
timeline the future renderer can execute: for every narration segment it
picks the primary panel (plus an optional secondary), validates the visual
intent and camera instruction, assigns a match score in [0.0, 1.0] and flags
low-confidence assignments with needs_review.

Inputs (one scene at a time, JSON only - never image pixels):
    script/page_001_scene_001.json      segments with panel_ids
    analysis/page_001_knowledge.json    page panel metadata

Outputs:
    shots/page_001_scene_001_timeline.json
    shots/page_001_scene_001_review.txt

Lightweight by design: no VLM, no LLM, no TTS, no video render. Matching is
a deterministic weighted scorer over panel metadata (characters, actions,
events, objects, OCR, story relevance) - no AI model is invoked.
"""
import gc
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger("mangaexplainer")

VISUAL_INTENTS = frozenset({
    "full_panel", "smart_crop", "character_closeup", "face_closeup",
    "object_closeup", "action_crop", "multi_panel",
})
DEFAULT_VISUAL_INTENT = "full_panel"

CAMERAS = frozenset({
    "static", "slow_zoom_in", "slow_zoom_out",
    "pan_left", "pan_right", "pan_up", "pan_down",
})
DEFAULT_CAMERA = "static"

# Transitions removed from the pipeline: panels are hard-cut only.
TRANSITIONS = frozenset({"cut"})
DEFAULT_TRANSITION = "cut"

_NOISE_TOKENS = frozenset({"unknown", "none", "no", "undefined", "null"})
_WORD_RE = re.compile(r"[a-z0-9]+")
TIMELINE_FILENAME = "page_{0:03d}_scene_{1:03d}_timeline.json"
REVIEW_FILENAME = "page_{0:03d}_scene_{1:03d}_review.txt"

DEFAULT_WEIGHTS = {
    "character": 0.25,
    "action": 0.15,
    "event": 0.20,
    "object": 0.10,
    "ocr": 0.15,
    "story_relevance": 0.15,
}


class PlanError(Exception):
    pass


# ------------------------------------------------------------ small utils


def tokens(text):
    """Lowercase alphanumeric tokens, noise words removed."""
    if not text:
        return set()
    words = set(_WORD_RE.findall(str(text).lower()))
    return {w for w in words if w not in _NOISE_TOKENS and len(w) >= 2}


def coverage(needle, hay):
    """Fraction of needle tokens present in hay (0.0 when needle is empty)."""
    return len(needle & hay) / len(needle) if needle else 0.0


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _numeric(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return clamp01(number) if 0.0 <= number <= 1.0 else default


# ------------------------------------------------------------------ paths


def timeline_path(cfg, page, scene):
    return Path(cfg.output.shots_dir) / TIMELINE_FILENAME.format(page, scene)


def review_path(cfg, page, scene):
    return Path(cfg.output.shots_dir) / REVIEW_FILENAME.format(page, scene)


def load_script(cfg, page, scene):
    """Read + validate ONE scene's script JSON."""
    path = Path(cfg.output.script_dir) / f"page_{page:03d}_scene_{scene:03d}.json"
    if not path.is_file():
        raise PlanError(f"script file not found: {path} (run the 'script' stage first)")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        raise PlanError(f"invalid script file {path}: {exc}") from None
    if not isinstance(doc, dict) or not isinstance(doc.get("segments"), list) or not doc["segments"]:
        raise PlanError(f"script file {path} has no segments list")
    for index, segment in enumerate(doc["segments"], 1):
        if not isinstance(segment, dict) or not str(segment.get("text") or "").strip():
            raise PlanError(f"script segment #{index}: text is empty")
    return doc


def load_knowledge(cfg, page):
    """Read + validate ONE page's knowledge JSON."""
    from pipeline.knowledge import load_page_knowledge

    try:
        return load_page_knowledge(cfg, page)
    except Exception as exc:
        raise PlanError(f"cannot read page {page} knowledge: {exc}") from None


# ---------------------------------------------------------------- scoring


def _char_names(panel):
    names = []
    for char in (panel.get("visual") or {}).get("characters") or []:
        if isinstance(char, dict):
            name = str(char.get("name") or "").strip()
            if name.lower() not in ("", "unknown"):
                names.append(name)
    return names


def _meaningful_list(value):
    items = []
    for item in value or []:
        text = str(item).strip()
        if text.lower() not in ("", "unknown"):
            items.append(text)
    return items


def _story_relevance(panel):
    value = (panel.get("visual") or {}).get("story_relevance", "unknown")
    if isinstance(value, str) and value.lower() in ("", "unknown"):
        return 0.0
    return _numeric(value)


def score_panel(panel, text, weights=None):
    """Weighted [0.0, 1.0] relevance of a panel against a segment's text.

    Returns (score, components). Components fall to 0.0 when the metadata is
    missing or unrelated, so an unanalyzed panel scores low and is flagged for
    review instead of silently winning.
    """
    weights = dict(weights or DEFAULT_WEIGHTS)
    text_tokens = tokens(text)
    visual = panel.get("visual") or {}
    ocr = panel.get("ocr") or {}

    char_tokens = set()
    for name in _char_names(panel):
        char_tokens |= tokens(name)
    action_tokens = set()
    for item in _meaningful_list(visual.get("actions")):
        action_tokens |= tokens(item)
    object_tokens = set()
    for item in _meaningful_list(visual.get("objects")):
        object_tokens |= tokens(item)
    event_tokens = tokens(visual.get("important_event"))
    ocr_tokens = tokens(ocr.get("text") if isinstance(ocr, dict) else None)

    def any_match(panel_tokens):
        return 1.0 if panel_tokens and (panel_tokens & text_tokens) else 0.0

    components = {
        "character": any_match(char_tokens),
        "action": any_match(action_tokens),
        "event": any_match(event_tokens),
        "object": any_match(object_tokens),
        "ocr": coverage(ocr_tokens, text_tokens),
        "story_relevance": _story_relevance(panel),
    }
    total = sum(weights.values()) or 1.0
    score = sum(weights[k] * components[k] for k in components) / total
    return clamp01(score), components


def camera_plan(raw_type, cfg):
    """Validate a camera instruction; fall back to static when invalid."""
    plan = {"type": DEFAULT_CAMERA, "start": 1.0, "end": 1.0}
    if not isinstance(raw_type, str) or raw_type not in CAMERAS:
        return plan
    shots_cfg = getattr(cfg, "shots", None)
    zoom_in_end = float(shots_cfg.get("zoom_in_end", 1.12)) if shots_cfg else 1.12
    zoom_out_end = float(shots_cfg.get("zoom_out_end", 0.92)) if shots_cfg else 0.92
    endpoints = {
        "static": (1.0, 1.0),
        "slow_zoom_in": (1.0, zoom_in_end),
        "slow_zoom_out": (1.0, zoom_out_end),
        "pan_left": (1.0, 1.0),
        "pan_right": (1.0, 1.0),
        "pan_up": (1.0, 1.0),
        "pan_down": (1.0, 1.0),
    }
    plan["type"] = raw_type
    plan["start"], plan["end"] = endpoints[raw_type]
    return plan


def valid_intent(raw):
    return raw if isinstance(raw, str) and raw in VISUAL_INTENTS else DEFAULT_VISUAL_INTENT


def valid_transition(raw):
    return raw if isinstance(raw, str) and raw in TRANSITIONS else DEFAULT_TRANSITION


def _positive_duration(value):
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


# ----------------------------------------------------------------- planner


class VisualPlanner:
    def __init__(self, cfg):
        self.cfg = cfg

    def run_scene(self, page, scene, state, force=False):
        try:
            return self._run(page, scene, state, force)
        except Exception:
            gc.collect()
            raise

    # ------------------------------------------------------------- main
    def _run(self, page, scene, state, force):
        try:
            if not isinstance(page, int) or isinstance(page, bool) or page < 1:
                raise PlanError("page must be a positive integer")
            if not isinstance(scene, int) or isinstance(scene, bool) or scene < 1:
                raise PlanError("scene must be a positive integer")
            key = f"page_{page:03d}_scene_{scene:03d}"
            cfg = self.cfg

            out_timeline = timeline_path(cfg, page, scene)
            out_review = review_path(cfg, page, scene)

            script = load_script(cfg, page, scene)
            knowledge = load_knowledge(cfg, page)

            if (
                state
                and not force
                and state.item_done(key, "visual_plan_completed")
                and out_timeline.is_file()
            ):
                return self._skip(page, scene, out_timeline)

            for segment in script["segments"]:
                if not _positive_duration(segment.get("estimated_seconds")):
                    raise PlanError(
                        f"script segment {segment.get('segment_id')!r}: "
                        "estimated_seconds must be positive"
                    )

            by_id = {p["panel_id"]: p for p in knowledge.get("panels") or []}
            shots, dropped = self._plan_shots(page, script, by_id)

            timeline = {
                "scene_id": script.get("scene_id"),
                "page": page,
                "scene": scene,
                "planner_version": 1,
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                "shots": shots,
                "dropped_panel_ids": dropped,
            }
            out_timeline.parent.mkdir(parents=True, exist_ok=True)
            self._write_json(out_timeline, timeline)
            self._write_txt(out_review, script.get("scene_id"), shots, script["segments"])
            if state:
                state.mark_item_done(key, "visual_plan_completed")

            review_count = sum(1 for shot in shots if shot["needs_review"])
            LOG.info(
                "page %s scene %s plan: %d shot(s), %d need(s) review -> %s",
                page, scene, len(shots), review_count, out_timeline,
            )
            return {
                "result": "ok",
                "page": page,
                "scene": scene,
                "scene_id": script.get("scene_id"),
                "timeline_file": str(out_timeline),
                "review_file": str(out_review),
                "shot_count": len(shots),
                "review_count": review_count,
                "match_scores": [shot["match_score"] for shot in shots],
                "needs_review": [shot["shot_id"] for shot in shots if shot["needs_review"]],
                "dropped_panel_ids": dropped,
                "shots": shots,
            }
        except PlanError as exc:
            LOG.error("page %s scene %s plan failed: %s", page, scene, exc)
            return {
                "result": "error", "page": page, "scene": scene, "message": str(exc),
            }
        except Exception as exc:  # pragma: no cover - defensive
            LOG.exception("visual planning failure")
            return {
                "result": "error", "page": page, "scene": scene,
                "message": f"visual planning error: {exc}",
            }
        finally:
            gc.collect()

    # -------------------------------------------------------- shot builder
    def _plan_shots(self, page, script, by_id):
        weights = self._match_weights()
        candidate_pool = self._candidate_pool(script.get("scene_id"), by_id)

        shots = []
        dropped = []
        reuse = {}
        for segment in script["segments"]:
            raw = segment.get("panel_ids")
            explicit = []
            invalid = []
            if isinstance(raw, list):
                seen = set()
                for pid in raw:
                    if not isinstance(pid, str) or pid in seen:
                        continue
                    seen.add(pid)
                    (explicit if pid in by_id else invalid).append(pid)
            dropped.extend(invalid)

            best = self._best_match(segment, explicit, weights, by_id, candidate_pool)
            if best is None:
                raise PlanError(
                    f"segment {segment.get('segment_id')!r} has no usable panel: "
                    "page knowledge contains no panel records"
                )
            panel_ids, primary, score, needs_review = best

            duration = float(segment["estimated_seconds"])
            intent = valid_intent(segment.get("visual_intent"))
            camera = camera_plan(segment.get("camera"), self.cfg)
            transition = "cut"  # transitions removed: panels are hard-cut only
            pieces = self._pieces(
                segment, primary, by_id[primary], duration, intent, camera, weights
            )
            for index, piece in enumerate(pieces, len(shots) + 1):
                shot = {
                    "shot_id": f"shot_{index:03d}",
                    "segment_id": segment.get("segment_id") or f"seg_{index:03d}",
                    "panel_ids": list(panel_ids),
                    "primary_panel": primary,
                    "visual_intent": piece["intent"],
                    "camera": piece["camera"],
                    "estimated_duration": round(piece["duration"], 2),
                    "transition": transition,
                    "match_score": score,
                    "needs_review": needs_review,
                }
                reuse[primary] = reuse.get(primary, 0) + 1
                shot["reuse_count"] = reuse[primary]
                shots.append(shot)

        return shots, dropped

    def _candidate_pool(self, scene_id, by_id):
        knowledge = []
        for pid, panel in by_id.items():
            if scene_id and panel.get("scene_id") == scene_id:
                knowledge.append(panel)
        if not knowledge:
            knowledge = [panel for panel in by_id.values()]
        knowledge.sort(key=lambda p: (p.get("reading_order") is None, p.get("reading_order") or 0))
        return knowledge

    def _best_match(self, segment, explicit, weights, by_id, pool):
        text = str(segment["text"])
        if explicit:
            ranked = []
            for pid in explicit:
                ranked.append((score_panel(by_id[pid], text, weights)[0], pid))
            ranked.sort(key=lambda row: row[0], reverse=True)
            score, primary = ranked[0]
            floor = self._setting("direct_match_floor", 0.90)
            score = max(min(1.0, score), floor)
            ordered = sorted(explicit, key=lambda pid: by_id[pid].get("reading_order") or 0)
            return list(ordered), primary, round(score, 3), False
        if not pool:
            return None
        ranked = []
        for panel in pool:
            pid = panel["panel_id"]
            ranked.append((score_panel(panel, text, weights)[0], pid))
        ranked.sort(key=lambda row: row[0], reverse=True)
        score, primary = ranked[0]
        threshold = self._setting("review_threshold", 0.55)
        tie_epsilon = self._setting("tie_epsilon", 0.02)
        ambiguous = len(ranked) > 1 and abs(ranked[0][0] - ranked[1][0]) < tie_epsilon
        needs_review = bool(score < threshold or ambiguous)

        epsilon = self._setting("secondary_panel_epsilon", 0.15)
        second = None
        if len(ranked) > 1 and score - ranked[1][0] <= epsilon:
            second = ranked[1][1]
        panel_ids = [primary] + ([second] if second else [])
        return panel_ids, primary, round(score, 3), needs_review

    # ------------------------------------------------------- long segments
    def _pieces(self, segment, primary_pid, panel, duration, intent, camera, weights):
        """Split long segments into an establishing shot + close-ups.

        A single panel MAY produce multiple shots (full -> close-up ->
        object), but only when the segment is long enough to deserve it and
        the panel offers something to zoom in on - no unnecessary duplicates.
        """
        cfg = self.cfg
        max_shots = int(self._setting("max_shots_per_segment", 3))
        threshold = self._setting("long_segment_threshold", 9.0)
        if max_shots < 2 or duration < threshold:
            return [{"intent": intent, "camera": camera, "duration": duration}]

        visual = panel.get("visual") or {}
        has_characters = bool(_char_names(panel))
        has_objects = bool(_meaningful_list(visual.get("objects")))
        has_actions = bool(_meaningful_list(visual.get("actions")))

        extras = 0
        sequence = []
        if has_characters:
            sequence.append({"intent": "character_closeup", "camera": camera_plan("slow_zoom_in", cfg)})
            extras += 1
        if has_objects:
            sequence.append({"intent": "object_closeup", "camera": camera_plan("slow_zoom_out", cfg)})
            extras += 1
        if has_actions:
            sequence.append({"intent": "action_crop", "camera": camera_plan("slow_zoom_out", cfg)})
            extras += 1

        if extras == 0:
            return [{"intent": intent, "camera": camera, "duration": duration}]

        sequence = sequence[: max_shots - 1]
        establish_duration = max(3.0, duration * 0.5)
        rest = duration - establish_duration
        each = rest / len(sequence)
        pieces = [{"intent": intent, "camera": camera, "duration": establish_duration}]
        pieces.extend({"intent": piece["intent"], "camera": piece["camera"], "duration": each} for piece in sequence)
        return pieces

    # --------------------------------------------------------------- config
    def _match_weights(self):
        shots_cfg = getattr(self.cfg, "shots", None)
        weights = {}
        if shots_cfg and hasattr(shots_cfg, "match_weights"):
            try:
                weights = shots_cfg.match_weights.to_dict()
            except Exception:
                weights = {}
        for name, value in DEFAULT_WEIGHTS.items():
            weights.setdefault(name, value)
        return {name: max(0.0, float(value)) for name, value in weights.items()}

    def _setting(self, name, default):
        shots_cfg = getattr(self.cfg, "shots", None)
        if shots_cfg is None:
            return default
        try:
            return float(shots_cfg.get(name, default))
        except (TypeError, ValueError):
            return float(default)

    # --------------------------------------------------------------- writes
    def _write_json(self, path, payload):
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    def _write_txt(self, path, scene_id, shots, segments):
        text_by_segment = {s.get("segment_id"): str(s.get("text") or "")
                           for s in segments}
        blocks = [f"SCENE {scene_id or '001'}".upper()]
        for shot in shots:
            blocks.append(
                f"SHOT {shot['shot_id'].split('_', 1)[1]}"
                f"\nNarration: {text_by_segment.get(shot['segment_id'], '')}"
                f"\nPanel: {shot['primary_panel'].upper()}"
                f"\nMatch: {shot['match_score']:.2f}"
                f"\nVisual: {shot['visual_intent']}"
                f"\nCamera: {shot['camera']['type']}"
                f"\nReview: {'YES' if shot['needs_review'] else 'NO'}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n\n".join(blocks) + "\n")

    def _skip(self, page, scene, out_timeline):
        return {
            "result": "skipped", "page": page, "scene": scene,
            "timeline_file": str(out_timeline),
        }