"""Ken Burns camera motion + panel transitions (motion / Tasks 17-18).

Turns the shots timeline (plan_shots) and the panels manifest (panel_prep)
into a lightweight, deterministic RENDER PLAN that a future renderer can
execute: smooth camera motion per shot plus short timed transitions between
shots.

Task 17 - Ken Burns
  For every shot we derive a smooth normalized source-rect path
  (x, y, w, h - fractions of the panel) from the shot's camera instruction:
      slow_zoom_in     coverage grows   (full panel -> 1/end_zoom)
      slow_zoom_out    coverage shrinks (1/start -> full panel)
      pan_left/right   horizontal drift across the panel
      pan_up/down      vertical drift
      static           fixed rect
  Motion is SMOOTH: rect corners are interpolated along an ease-in-out
  (smoothstep) curve across a configurable number of keyframes, the coverage
  is clamped to [min_coverage, 1.0] and the rect is always clamped inside the
  panel so it never bleeds past the artwork.

Task 18 - Transitions
  Between consecutive shots we add a short transition dependent on panel
  timing:
      cut        hard switch (0 duration / no overlap)
      fade       dip to black at the boundary
      crossfade  dissolve overlap where the two shots coexist
  Durations stay short (min(transition_max, fraction * shorter neighbour))
  and are anchored to the shots' cumulative timeline.

Emitted as motion/render_plan.json. No pixels are produced here - no image
conversion, no re-encode, no PNG/JPG writes, no subtitles - it is a JSON
blueprint consumed by the (future) render stage. CPU only, deterministic,
one scene at a time.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

LOG = logging.getLogger(__name__)

RENDER_PLAN_NAME = "render_plan.json"
SHOT_MOTION_NAME = "shot_motion.json"

# Camera types produced by plan_shots (visual_planner.CAMERAS).
CAMERAS = frozenset({
    "static", "slow_zoom_in", "slow_zoom_out",
    "pan_left", "pan_right", "pan_up", "pan_down",
})
DEFAULT_CAMERA = "static"

# Transitions produced by plan_shots.
TRANSITIONS = frozenset({"cut", "fade", "dissolve"})
DEFAULT_TRANSITION = "cut"

# A camera may not cover less than this fraction of the panel (prevents
# zooming into noise and keeps the source rect resolvable).
DEFAULT_MIN_COVERAGE = 0.5
# Pan/vertical shots operate at this fixed coverage so there is room to move.
DEFAULT_PAN_COVERAGE = 0.80
# Keyframes for the smooth motion path (not video frames - a path sample).
DEFAULT_KEYFRAMES = 12
# Longest allowed transition, in seconds.
DEFAULT_TRANSITION_MAX = 0.5
# Transition length as a fraction of the shorter neighbouring shot.
DEFAULT_TRANSITION_FRACTION = 0.15


class MotionError(Exception):
    """Base error for the motion stage."""


class NoTimelineData(MotionError):
    """No usable shots timeline was found."""


class NoPanelData(MotionError):
    """A referenced panel is missing from the panels manifest."""


class NoImageData(MotionError):
    """No panel image could be located for a shot."""


def _f(x):
    return float(x)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def smoothstep(t):
    """Cubic ease-in-out (Perlin smoothstep) on t in [0,1] -> [0,1]."""
    t = clamp(_f(t), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def resolve_panel_image(root, image):
    """Resolve a maybe-relative panel image path against the repo root."""
    root = Path(root)
    path = Path(image)
    if not path.is_absolute():
        path = root / path
    return path


# ---------------------------------------------------------------- motion
def panel_rect_for_conservative(img_w, img_h, coverage):
    """Centred source rect covering the given fraction of a panel image."""
    c = clamp(_f(coverage), 0.0, 1.0)
    w = max(1, int(round(img_w * c)))
    h = max(1, int(round(img_h * c)))
    x = int(round((img_w - w) / 2.0))
    y = int(round((img_h - h) / 2.0))
    return x, y, w, h


def zoom_to_rect(img_w, img_h, z, cx=0.5, cy=0.5):
    """Source rect (px) for zoom factor z centred about (cx, cy) in [0,1]."""
    scale = max(_f(z), 1.0)
    w = max(1, int(round(img_w / scale)))
    h = max(1, int(round(img_h / scale)))
    x = int(round(cx * img_w - w * cx))
    y = int(round(cy * img_h - h * cy))
    x = max(0, min(img_w - w, x))
    y = max(0, min(img_h - h, y))
    return x, y, w, h


def kenburns_path(camera_type, img_w, img_h, duration, num_keyframes=None,
                  min_coverage=DEFAULT_MIN_COVERAGE,
                  pan_coverage=DEFAULT_PAN_COVERAGE):
    """Build a smooth normalized source-rect path for a Ken Burns camera.

    camera_type: one of CAMERAS.
    img_w/img_h: panel dimensions in px (used to stay inside the artwork).
    duration:    shot duration in seconds (narration-driven).
    min_coverage: smallest allowed panel coverage fraction.
    pan_coverage: fixed coverage used by pan/vertical cameras.
    num_keyframes: number of path samples (not video frames).

    Returns dict with 'keyframes' = list of {t, x, y, w, h} normalized to the
    panel (each in [0,1]), plus 'zoom_start'/'zoom_end' px rects for tests.
    The path is smooth because corners are eased and clamped to stay inside.
    """
    if not isinstance(camera_type, str) or camera_type not in CAMERAS:
        camera_type = DEFAULT_CAMERA
    duration = max(_f(duration), 0.0)
    n = max(2, int(num_keyframes) if num_keyframes else DEFAULT_KEYFRAMES)
    zlo = max(clamp(_f(min_coverage), 0.1, 1.0), 0.1)
    pz = clamp(_f(pan_coverage), zlo, 1.0)

    frames = []
    if camera_type == "static":
        x0, y0, w0, h0 = panel_rect_for_conservative(img_w, img_h, 1.0)
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0.0
            frames.append(_norm_rect(x0, y0, w0, h0, img_w, img_h, t))
        return {
            "type": "static",
            "duration": round(duration, 4),
            "keyframes": frames,
        }

    if camera_type == "slow_zoom_in":
        # full panel -> zoomed in (coverage zlo..1.0); zoom factor grows.
        s0 = 1.0
        s1 = 1.0 / zlo
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0.0
            e = smoothstep(t)
            z = s0 + (s1 - s0) * e
            x, y, w, h = zoom_to_rect(img_w, img_h, z)
            frames.append(_norm_rect(x, y, w, h, img_w, img_h, t))
        return _zoom_res(camera_type, img_w, img_h, duration, frames, 1.0, 1.0 / zlo)

    if camera_type == "slow_zoom_out":
        # zoomed in -> full panel (coverage 1.0..zlo).
        s0 = 1.0 / zlo
        s1 = 1.0
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0.0
            e = smoothstep(t)
            z = s0 + (s1 - s0) * e
            x, y, w, h = zoom_to_rect(img_w, img_h, z)
            frames.append(_norm_rect(x, y, w, h, img_w, img_h, t))
        return _zoom_res(camera_type, img_w, img_h, duration, frames, 1.0 / zlo, 1.0)

    # ---- pan / vertical: translate a fixed-coverage rect across the panel.
    # The viewport sweeps the full available travel for a clear sense of
    # motion. pan_right = left-to-right, pan_left = right-to-left,
    # pan_down = top-to-bottom, pan_up = bottom-to-top.
    pw = max(1, int(round(img_w * pz)))
    ph = max(1, int(round(img_h * pz)))
    x_lo, x_hi = 0, max(0, img_w - pw)
    y_lo, y_hi = 0, max(0, img_h - ph)
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0.0
        e = smoothstep(t)
        if camera_type == "pan_right":
            x = round(x_lo + (x_hi - x_lo) * e)
            y = y_lo + (y_hi - y_lo) / 2.0
        elif camera_type == "pan_left":
            x = round(x_hi - (x_hi - x_lo) * e)
            y = y_lo + (y_hi - y_lo) / 2.0
        elif camera_type == "pan_down":
            x = x_lo + (x_hi - x_lo) / 2.0
            y = round(y_lo + (y_hi - y_lo) * e)
        elif camera_type == "pan_up":
            x = x_lo + (x_hi - x_lo) / 2.0
            y = round(y_hi - (y_hi - y_lo) * e)
        else:  # defensive
            x = x_lo + (x_hi - x_lo) / 2.0
            y = y_lo + (y_hi - y_lo) / 2.0
        x = min(max(0, int(x)), x_hi)
        y = min(max(0, int(y)), y_hi)
        frames.append(_norm_rect(x, y, pw, ph, img_w, img_h, t))
    return _zoom_res(camera_type, img_w, img_h, duration, frames, 1.0 / pz, 1.0 / pz)


def _zoom_res(camera_type, img_w, img_h, duration, frames, zs, ze):
    return {
        "type": camera_type,
        "duration": round(duration, 4),
        "zoom_start": round(zs, 4),
        "zoom_end": round(ze, 4),
        "keyframes": frames,
    }


def _norm_rect(x, y, w, h, img_w, img_h, t):
    return {
        "t": round(t, 6),
        "x": round(x / max(1, img_w), 6),
        "y": round(y / max(1, img_h), 6),
        "w": round(w / max(1, img_w), 6),
        "h": round(h / max(1, img_h), 6),
    }


# ---------------------------------------------------------- transitions
def transition_length(a_dur, b_dur, cfg=None):
    """Short transition length dependent on the two neighbouring shot timings."""
    max_len = DEFAULT_TRANSITION_MAX
    fraction = DEFAULT_TRANSITION_FRACTION
    if cfg is not None:
        motion = getattr(cfg, "motion", None)
        if motion is not None:
            try:
                max_len = float(motion.get("transition_max", max_len))
            except (TypeError, ValueError):
                pass
            try:
                fraction = float(motion.get("transition_fraction", fraction))
            except (TypeError, ValueError):
                pass
    a_dur = max(_f(a_dur), 0.0)
    b_dur = max(_f(b_dur), 0.0)
    shorter = min(a_dur, b_dur)
    if shorter <= 0.0:
        return 0.0
    length = min(max_len, fraction * shorter)
    # never consume more than half the shorter shot
    return round(min(length, shorter / 2.0), 4)


def build_transition(prev_dur, next_dur, trans_type, cfg=None):
    """Transition descriptor between two consecutive shots.

    trans_type: 'cut' | 'fade' | 'dissolve' (visual_planner TRANSITIONS).
    Returns dict {type, duration, overlap}. cut always has duration/overlap 0.
    """
    trans_type = trans_type if isinstance(trans_type, str) and trans_type in \
        TRANSITIONS else DEFAULT_TRANSITION
    if trans_type == "cut":
        return {"type": "cut", "duration": 0.0, "overlap": 0.0}
    length = transition_length(prev_dur, next_dur, cfg)
    if trans_type == "fade":
        # dip to black: no overlap, but a short dip either side of the seam.
        return {"type": "fade", "duration": length, "overlap": 0.0}
    # dissolve / crossfade: the two shots overlap in time.
    return {"type": "dissolve", "duration": length, "overlap": length}


# ------------------------------------------------------------ render plan
def _load_timelines(cfg, root, page_nums=None):
    """Yield (page, scene, doc) for timeline files under shots_dir."""
    shots_dir = Path(cfg.output.shots_dir)
    if not shots_dir.is_dir():
        return
    files = sorted(shots_dir.glob("page_*_scene_*_timeline.json"))
    for path in files:
        try:
            doc = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        page = doc.get("page")
        scene = doc.get("scene_id")
        if page is None or scene is None:
            continue
        if page_nums is not None and int(page) not in [int(p) for p in page_nums]:
            continue
        yield int(page), int(scene), doc


def _load_panels_manifest(root):
    """panels_manifest.json -> {panel_id: entry}; [] if none present."""
    path = Path(root) / "visuals" / "panels_manifest.json"
    if not path.is_file():
        return {}
    try:
        docs = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(docs, list):
        return {}
    return {d.get("panel_id"): d for d in docs if isinstance(d, dict)}


def motion_for_shot(shot, panels, root, cfg, keyframes=12):
    """Build a single shot's motion section (Task 17).

    shot:  one entry from a shots timeline (has camera, estimated_duration,
           primary_panel).
    panels: panel_id -> manifest entry (visuals/panels_manifest.json).
    Returns dict used inside the render plan.
    """
    camera_type = shot.get("camera", {}).get("type", DEFAULT_CAMERA) \
        if isinstance(shot.get("camera"), dict) else DEFAULT_CAMERA
    duration = max(_f(shot.get("estimated_duration") or 0.0), 0.0)

    panel = panels.get(str(shot.get("primary_panel"))) or panels.get(
        (shot.get("panel_ids") or [""])[0])
    if panel is None:
        raise NoPanelData(
            f"shot {shot.get('shot_id')}: panel {shot.get('primary_panel')!r} "
            "not found in visuals/panels_manifest.json"
        )
    image_rel = panel.get("image")
    if not image_rel:
        raise NoImageData(f"shot {shot.get('shot_id')}: panel has no image")
    img_path = resolve_panel_image(root, image_rel)
    if not img_path.is_file():
        raise NoImageData(
            f"shot {shot.get('shot_id')}: panel image not found at {img_path}"
        )

    img_w = int(panel.get("width") or 0)
    img_h = int(panel.get("height") or 0)
    if img_w <= 0 or img_h <= 0:
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            raise NoImageData(
                f"shot {shot.get('shot_id')}: cannot read dimensions for {img_path}"
            )
        img_h, img_w = img.shape[:2]

    motion_cfg = getattr(cfg, "motion", None)
    min_cov = DEFAULT_MIN_COVERAGE
    pan_cov = DEFAULT_PAN_COVERAGE
    if motion_cfg is not None:
        try:
            min_cov = float(motion_cfg.get("min_coverage", min_cov))
        except (TypeError, ValueError):
            pass
        try:
            pan_cov = float(motion_cfg.get("pan_coverage", pan_cov))
        except (TypeError, ValueError):
            pass
    path = kenburns_path(
        camera_type, img_w, img_h, duration, num_keyframes=keyframes,
        min_coverage=min_cov, pan_coverage=pan_cov,
    )
    return {
        "shot_id": shot.get("shot_id"),
        "segment_id": shot.get("segment_id"),
        "panel_id": panel.get("panel_id"),
        "image": image_rel,
        "camera": camera_type,
        "duration": round(duration, 4),
        "path": path,
    }


def build_render_plan(cfg, root, page_nums=None, keyframes=12,
                      keyframe_setting=True):
    """Assemble the full ordered render plan for one or more scenes (17+18).

    Returns list of entries: each entry has the shot's motion (Task 17) plus
    a 'transition' to the NEXT entry (Task 18) computed from panel timing.
    """
    root = Path(root)
    panels = _load_panels_manifest(root)
    entries = []
    for page, scene, doc in _load_timelines(cfg, root, page_nums):
        shots = doc.get("shots")
        if not isinstance(shots, list):
            continue
        for idx, shot in enumerate(shots):
            motion = motion_for_shot(shot, panels, root, cfg,
                                     keyframes=keyframes)
            entry = {
                "index": idx,
                "scene": scene,
                "page": page,
                "transition": shot.get("transition", DEFAULT_TRANSITION),
                "motion": motion,
            }
            entries.append(entry)
    return entries


def assemble_transitions(entries, cfg=None):
    """Attach Task 18 transitions between consecutive entries.

    Returns a list of transition descriptors aligned with the entries: the
    i-th transition connects entry i -> entry i+1. There are len(entries)-1
    of them (a trailing shot has no next transition).
    """
    out = []
    for i in range(len(entries) - 1):
        a = entries[i]["motion"]
        b = entries[i + 1]["motion"]
        trans_type = entries[i + 1].get("transition") or DEFAULT_TRANSITION
        trans = build_transition(a["duration"], b["duration"], trans_type, cfg)
        out.append({
            "from_index": entries[i]["index"],
            "from_shot": a["shot_id"],
            "to_index": entries[i + 1]["index"],
            "to_shot": b["shot_id"],
            "type": trans["type"],
            "duration": trans["duration"],
            "overlap": trans["overlap"],
        })
    return out


def motion_dir(cfg, root):
    return Path(root) / "motion"


def render_plan_path(cfg, root):
    return motion_dir(cfg, root) / RENDER_PLAN_NAME


def run_motion(cfg, root, page_nums=None, force=False, keyframes=None):
    """Execute the motion stage; write motion/render_plan.json. Returns dict."""
    root = Path(root)
    entries = build_render_plan(cfg, root, page_nums, keyframes=keyframes)
    if not entries:
        raise NoTimelineData(
            "no shots timeline found (no page_*_scene_*_timeline.json in "
            "shots/). Run the plan/crops stages first to build a timeline."
        )
    transitions = assemble_transitions(entries, cfg)
    plan = {
        "tasks": ["17_ken_burns", "18_transitions"],
        "entries": entries,
        "transitions": transitions,
    }
    out_dir = motion_dir(cfg, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = render_plan_path(cfg, root)
    out_path.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOG.info("wrote %s (%d entries, %d transitions)", out_path,
             len(entries), len(transitions))
    return {
        "result": "computed",
        "entries": len(entries),
        "transitions": len(transitions),
        "plan_file": str(out_path),
    }
