"""Shot -> 16:9 cinematic crop planner (crop stage).

For every shot in the visual timeline it turns the referenced manga panel into
a cinematic frame: it computes a crop rectangle (x, y, width, height) that
preserves the panel's important content - faces first, then characters,
objects, action, dialogue/text and visual effects - without stretching or
distorting the artwork.

Inputs (one page scene at a time):
    shots/page_001_scene_001_timeline.json   <- visual planner
    analysis/page_001_knowledge.json         <- panel metadata + image paths
    panels/page_001/panel_002.jpg ...        <- real panel images

Outputs:
    shots/page_001_scene_001/shot_001.json    crop rectangle + plan
    crops/page_001_scene_001/shot_001.jpg     the cropped panel (native size)
    crops/page_001_scene_001/shot_001_debug.jpg  panel + crop + regions

Important content is located with:
    - OCR bounding boxes (dialogue/text)
    - VLM character/face/object boxes when the analyzer provides pixel boxes
    - lightweight edge/blob computer vision (Canny + connected components),
      which detects drawn figures even without model metadata
    No AI model is called here; a video is not rendered; pixels are never
    upscaled (the future renderer does that). One shot at a time in memory.
"""
import gc
import json
import logging
import os
import re
from pathlib import Path

import cv2
import numpy as np

LOG = logging.getLogger("mangaexplainer")

TARGET_ASPECT = 16.0 / 9.0
EPSILON = 1e-6

# A 16:9 window smaller than this fraction of a panel's shorter side is too
# microscopic to be a usable frame (upscaling it to 720p/1080p just produces
# blur). Below this we keep the whole panel (letterboxed to 16:9) instead of
# emitting a useless sliver.
MIN_CROP_FRACTION = 0.6

SUPPORTED_INTENTS = frozenset({
    "full_panel", "character_closeup", "face_closeup",
    "object_closeup", "action_crop", "smart_crop", "multi_panel",
})

# Priority: lower rank = more important. Used when regions compete.
KIND_RANK = {
    "face": 0,
    "character": 1,
    "object": 2,
    "action": 3,
    "text": 4,
    "effect": 5,
    "draw": 6,
}

REGION_COLORS = {
    "face": (0, 0, 255),        # red
    "character": (0, 140, 255), # orange
    "object": (255, 0, 255),    # magenta
    "action": (0, 255, 255),    # yellow
    "text": (255, 0, 0),        # blue
    "effect": (255, 255, 255),  # white
    "draw": (64, 64, 64),       # gray
}
CROP_COLOR = (0, 255, 0)        # green
CROP_COLOR_WIDE = (0, 255, 0)   # green base

# Canonical ratios tried when a 16:9 window cannot contain every critical
# region: a wider safe strategy is better than destroying artwork.
CANONICAL_RATIOS = (16.0 / 9.0, 4.0 / 3.0, 3.0 / 2.0, 1.0, 2.0 / 3.0, 3.0 / 4.0)

_RESOLUTION_RE = re.compile(r"^(\d+)\s*[xX:]\s*(\d+)$")


class CropError(Exception):
    pass


# ------------------------------------------------------------------ helpers


def parse_resolution(value):
    """Parse '1280x720' (or '1280:720') into (width, height)."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        width, height = int(value[0]), int(value[1])
    else:
        match = _RESOLUTION_RE.match(str(value).strip())
        if not match:
            raise CropError(f"invalid crops.resolution {value!r}: expected 'WxH'")
        width, height = int(match.group(1)), int(match.group(2))
    if width < 2 or height < 2:
        raise CropError(f"invalid crops.resolution {value!r}: too small")
    return width, height


def aspect_ratio(box):
    _, _, w, h = box
    return (w / h) if h else TARGET_ASPECT


def intersect_area(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    return ix * iy


def clamp_box(x, y, w, h, width, height):
    """Clamp a box so it never leaves the panel; keep size >= 1 px."""
    w = max(1.0, min(float(w), float(width)))
    h = max(1.0, min(float(h), float(height)))
    x = max(0.0, min(float(x), float(width) - w))
    y = max(0.0, min(float(y), float(height) - h))
    return (x, y, w, h)


def snap_box(box):
    """Round a float box to integer pixels inside the panel."""
    x, y, w, h = box
    return (
        int(round(x)), int(round(y)),
        int(round(w)), int(round(h)),
    )


def _min_crop_side(width, height):
    """Smallest usable crop side (px) for a panel of (width, height)."""
    return MIN_CROP_FRACTION * min(width, height)


def _too_small(box, width, height):
    """True if a crop box is too tiny to be a usable video frame."""
    return min(box[2], box[3]) < _min_crop_side(width, height) - EPSILON


def is_valid_region(region, width, height):
    x, y, w, h = region["bbox"]
    if w <= 0 or h <= 0:
        return False
    if x < 0 or y < 0 or x + w > width + EPSILON or y + h > height + EPSILON:
        return False
    return True


# ------------------------------------------------------------- CV detection


def detect_draw_regions(image, cfg=None):
    """Lightweight CV: find drawn detail (figures), no models involved.

    Runs Canny edges and connected-component labeling on a downscaled copy;
    each significant blob becomes a 'draw' region whose weight reflects its
    edge density (solid ink detail is more important than a flat background).
    """
    crops_cfg = getattr(cfg, "crops", None)
    min_area = float(crops_cfg.get("min_blob_area", 80)) if crops_cfg else 80.0
    max_regions = int(crops_cfg.get("max_regions", 24)) if crops_cfg else 24
    height, width = image.shape[:2]
    scale = min(1.0, 768.0 / max(height, width))
    small = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))),
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(edges, 8)
    blobs = []
    for index in range(1, count):
        x, y, w, h, area = [int(value) for value in stats[index]]
        if w < 3 or h < 3 or w * h < min_area:
            continue
        density = min(1.0, area / float(w * h))
        weight = 0.30 + 0.50 * density
        blobs.append({
            "kind": "draw",
            "bbox": (x / scale, y / scale, w / scale, h / scale),
            "weight": round(weight, 3),
        })
    blobs.sort(key=lambda item: item["weight"], reverse=True)
    kept = []
    for blob in blobs:
        x, y, w, h = blob["bbox"]
        overlaps = any(
            0.4 < intersect_area(x, y, x + w, y + h, kx, ky, kx + kw, ky + kh)
            / float(w * h + 1e-9)
            for (kx, ky, kw, kh) in [other["bbox"] for other in kept]
        )
        if not overlaps:
            kept.append(blob)
        if len(kept) >= max_regions:
            break
    return kept


# ---------------------------------------------------------- metadata regions


def _box_or_none(value):
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x, y, w, h = (float(value[0]), float(value[1]),
                      float(value[2]), float(value[3]))
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


def _entry_box(entry):
    """Accept 'bbox', 'box', or {'x','y','w','h'} on a metadata entry."""
    for key in ("bbox", "box", "region"):
        box = entry.get(key)
        if box is not None:
            return _box_or_none(box)
    if all(key in entry for key in ("x", "y", "w", "h")):
        return _box_or_none([entry["x"], entry["y"], entry["w"], entry["h"]])
    return None


def metadata_regions(panel_record):
    """Regions with explicit pixel boxes from OCR/VLM metadata."""
    regions = []
    ocr = panel_record.get("ocr") or {}
    for index, block in enumerate(ocr.get("blocks") or [], 1):
        box = _box_or_none(block.get("bbox"))
        if box is None:
            continue
        regions.append({
            "kind": "text", "bbox": box, "weight": 1.0,
            "label": f"text{index:02d}", "source": "ocr",
        })

    visual = panel_record.get("visual") or {}
    for index, entry in enumerate(visual.get("faces") or [], 1):
        box = _entry_box(entry)
        if box is None:
            continue
        regions.append({
            "kind": "face", "bbox": box, "weight": 1.0,
            "label": f"face{index:02d}", "source": "vlm",
        })
    for index, entry in enumerate(visual.get("characters") or [], 1):
        box = _entry_box(entry)
        if box is None:
            continue
        regions.append({
            "kind": "face" if entry.get("is_face") else "character",
            "bbox": box, "weight": 1.0,
            "label": f"char{index:02d}", "source": "vlm",
        })
    for kind, key in (("object", "objects"), ("action", "actions"),
                      ("effect", "visual_effects")):
        for index, entry in enumerate(visual.get(key) or [], 1):
            if not isinstance(entry, dict):
                continue
            box = _entry_box(entry)
            if box is None:
                continue
            regions.append({
                "kind": kind, "bbox": box, "weight": 1.0,
                "label": f"{kind}{index:02d}", "source": "vlm",
            })
    return regions


def gather_regions(panel_record, image, cfg=None):
    """Metadata regions first, then CV draw regions (deduplicated)."""
    regions = metadata_regions(panel_record)
    labels = {region["label"] for region in regions}
    width, height = image.shape[1], image.shape[0]
    regions = [r for r in regions if is_valid_region(r, width, height)]
    draw_counter = 1
    for blob in detect_draw_regions(image, cfg):
        x, y, w, h = blob["bbox"]
        if w <= 0 or h <= 0:
            continue
        label = f"draw{draw_counter:02d}"
        draw_counter += 1
        # Skip draw blobs fully inside a metadata region (that is its detail).
        if any(
            intersect_area(x, y, x + w, y + h, rx, ry, rx + rw, ry + rh)
            >= 0.8 * w * h
            for (rx, ry, rw, rh) in [region["bbox"] for region in regions]
        ):
            continue
        regions.append({
            "kind": "draw", "bbox": (x, y, w, h),
            "weight": blob["weight"], "label": label, "source": "edge",
        })
    regions.sort(key=lambda item: (item["weight"], item["label"]), reverse=True)
    return regions


# ------------------------------------------------------------------ cropping


def _window_candidates(width, height, aspect, align=None):
    """16:9 windows inside a panel.

    Spans the largest aspect-correct window over the short axis and, when an
    anchor box is given, adds windows aligned to contain it (left/center/right
    of the anchor) so a 16:9 crop that includes every critical region is found
    whenever one exists.
    """
    if width / height >= aspect:
        w, h = height * aspect, height
        for spot in (0.0, 0.5, 1.0):
            yield clamp_box(0.0, spot * (height - h), w, h, width, height)
        if align is not None:
            ax, _, aw, _ = align
            if aw <= w + EPSILON:
                for x in (ax, ax + aw - w, ax + aw / 2 - w / 2):
                    yield clamp_box(x, 0.0, w, h, width, height)
    else:
        w, h = width, width / aspect
        for spot in (0.0, 0.5, 1.0):
            yield clamp_box(spot * (width - w), 0.0, w, h, width, height)
        if align is not None:
            _, ay, _, ah = align
            if ah <= h + EPSILON:
                for y in (ay, ay + ah - h, ay + ah / 2 - h / 2):
                    yield clamp_box(0.0, y, w, h, width, height)


def _largest_window(width, height, aspect):
    if width / height >= aspect:
        return (height * aspect, height)
    return (width, width / aspect)


def _window_score(box, regions, critical):
    x, y, w, h = box
    x2, y2, xl, yl = x + w, y + h, x, y
    score = 0.0
    covered = 0
    for region in regions:
        rx, ry, rw, rh = region["bbox"]
        inter = intersect_area(x, y, x2, y2, rx, ry, rx + rw, ry + rh)
        fraction = inter / (rw * rh + EPSILON)
        score += region["weight"] * fraction
        if fraction > 0.999 and region in critical:
            covered += 1
    return score, covered


def _pick_best_window(candidates, regions, critical, centroids):
    best = None
    best_key = None
    for box in candidates:
        score, covered = _window_score(box, regions, critical)
        cut_critical = len(critical) - covered
        cx, cy = (box[0] + box[2] / 2, box[1] + box[3] / 2)
        if centroids:
            dmin = min(
                ((cx - rx - rw / 2) ** 2 + (cy - ry - rh / 2) ** 2)
                for (rx, ry, rw, rh) in centroids
            )
        else:
            dmin = 0.0
        key = (cut_critical, -score, dmin)
        if best_key is None or key < best_key:
            best_key = key
            best = box
    return best


def _critical_of(regions, critical_weight):
    return [region for region in regions if region["weight"] >= critical_weight - EPSILON]


def _union_box(regions, width, height):
    xs = [r["bbox"][0] for r in regions]
    ys = [r["bbox"][1] for r in regions]
    x2s = [r["bbox"][0] + r["bbox"][2] for r in regions]
    y2s = [r["bbox"][1] + r["bbox"][3] for r in regions]
    x, y = min(xs), min(ys)
    w = max(x2s) - x
    h = max(y2s) - y
    return clamp_box(x, y, w, h, width, height)


def _wider_safe_box(critical, width, height, aspect, padding):
    if not critical:
        return None
    pad = padding * min(width, height)
    box = _union_box(critical, width, height)
    x, y = max(0.0, box[0] - pad), max(0.0, box[1] - pad)
    x2 = min(width, box[0] + box[2] + pad)
    y2 = min(height, box[1] + box[3] + pad)
    w, h = x2 - x, y2 - y
    wide = clamp_box(x, y, w, h, width, height)

    # Expand toward the smallest canonical ratio that still fits in the panel
    # and still covers every critical region ("wider safe crop").
    for ratio in CANONICAL_RATIOS:
        if wide[2] / wide[3] > ratio + EPSILON:
            target_h = wide[2] / ratio
            if target_h <= height + EPSILON:
                nw, nh = wide[2], target_h
            else:
                continue
        else:
            target_w = wide[3] * ratio
            if target_w <= width + EPSILON:
                nw, nh = target_w, wide[3]
            else:
                continue
        nx = max(0.0, min((wide[0] + wide[2] / 2) - nw / 2, width - nw))
        ny = max(0.0, min((wide[1] + wide[3] / 2) - nh / 2, height - nh))
        wide = clamp_box(nx, ny, nw, nh, width, height)
        return wide
    return wide


def visual_intent(raw):
    return str(raw).strip() if isinstance(raw, str) and raw in SUPPORTED_INTENTS else "smart_crop"


def focus_regions(intent, regions):
    """Regions a close-up intent concentrates on (with fallbacks)."""
    tiers = {
        "face_closeup": (("face",), ("character",)),
        "character_closeup": (("character", "face"),),
        "object_closeup": (("object",), ("draw",)),
        "action_crop": (("action",), ("draw",)),
    }.get(intent)
    if not tiers:
        return []
    for tier in tiers:
        tier_regions = [region for region in regions if region["kind"] in tier]
        if not tier_regions:
            continue
        tier_regions.sort(
            key=lambda region: (region["weight"], -region["bbox"][2] * region["bbox"][3]),
            reverse=True,
        )
        return tier_regions
    return []


def compute_crop(width, height, intent, regions, cfg=None):
    """Return (x, y, w, h, strategy, letterbox) covering the intent.

    strategy is one of 16_9 (exact cinematic frame), safe_wider (critical
    content does not fit a 16:9 window, so a wider box is kept and must be
    padded at render time) or full_panel (whole panel preserved).
    """
    x, y, w, h = 0.0, 0.0, float(width), float(height)
    letterbox = abs(aspect_ratio((x, y, w, h)) - TARGET_ASPECT) > 0.01
    if intent == "full_panel" or intent == "multi_panel":
        return (x, y, w, h), "full_panel", letterbox

    crops_cfg = getattr(cfg, "crops", None)
    critical_weight = float(crops_cfg.get("critical_weight", 0.6)) if crops_cfg else 0.6
    padding = float(crops_cfg.get("safe_padding", 0.06)) if crops_cfg else 0.06

    focus = focus_regions(intent, regions)
    if intent in ("face_closeup", "character_closeup", "object_closeup", "action_crop"):
        if not focus and regions:
            focus = [max(regions, key=lambda region: region["weight"])]
        if focus:
            region = max(focus, key=lambda item: item["weight"])
            rx, ry, rw, rh = region["bbox"]
            box_w = max(rw, rh * TARGET_ASPECT)
            box_h = box_w / TARGET_ASPECT
            if box_w > width:
                box_w = width
                box_h = box_w / TARGET_ASPECT
            if box_h > height:
                box_h = height
                box_w = box_h * TARGET_ASPECT
            cx = min(max(rx + rw / 2, box_w / 2), width - box_w / 2)
            cy = min(max(ry + rh / 2, box_h / 2), height - box_h / 2)
            box = clamp_box(cx - box_w / 2, cy - box_h / 2, box_w, box_h, width, height)
            inside = (
                box[0] - EPSILON <= rx and ry >= box[1] - EPSILON
                and rx + rw <= box[0] + box[2] + EPSILON
                and ry + rh <= box[1] + box[3] + EPSILON
            )
            if inside:
                if _too_small(box, width, height):
                    # The 16:9 close-up window is microscopic (e.g. 20x11 on a
                    # ~200px panel) -> keep the whole panel, letterboxed.
                    return (x, y, w, h), "full_panel", letterbox
                return box, "16_9", False
            return _wider_safe_box([region], width, height, TARGET_ASPECT, padding), "safe_wider", True

    # smart crop (default): best 16:9 window, widened when it would cut art.
    critical = _critical_of(regions, critical_weight)
    align = None
    if critical:
        align = _union_box(critical, width, height)
        window_w, window_h = _largest_window(width, height, TARGET_ASPECT)
        fits = (
            align[2] <= window_w + EPSILON and align[3] <= window_h + EPSILON
        )
        if not fits:
            return _wider_safe_box(critical, width, height, TARGET_ASPECT, padding), "safe_wider", True

    candidates = list(_window_candidates(width, height, TARGET_ASPECT, align))
    if not candidates:
        return (x, y, w, h), "full_panel", letterbox
    box = _pick_best_window(candidates, regions, critical,
                            [(r["bbox"]) for r in critical])
    if critical:
        critical_box = align
        inside = (
            box[0] - EPSILON <= critical_box[0] and box[1] - EPSILON <= critical_box[1]
            and box[0] + box[2] >= critical_box[0] + critical_box[2] - EPSILON
            and box[1] + box[3] >= critical_box[1] + critical_box[3] - EPSILON
        )
        if not inside:
            return _wider_safe_box(critical, width, height, TARGET_ASPECT, padding), "safe_wider", True
    if _too_small(box, width, height):
        return (x, y, w, h), "full_panel", letterbox
    return box, "16_9", False


def compute_target(cfg):
    return parse_resolution(getattr(getattr(cfg, "crops", None), "resolution", "1280x720"))


# ---------------------------------------------------------------- debug image


def draw_debug(image, box, regions, wide_box=None):
    debug = image.copy()
    for region in regions:
        x, y, w, h = region["bbox"]
        color = REGION_COLORS.get(region["kind"], (200, 200, 200))
        cv2.rectangle(debug, (int(x), int(y)), (int(x + w), int(y + h)), color, 1)
        cv2.putText(debug, region["label"], (int(x), max(0, int(y) - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
    x, y, w, h = box
    cv2.rectangle(debug, (int(x), int(y)), (int(x + w), int(y + h)), CROP_COLOR, 2)
    if wide_box is not None:
        wx, wy, ww, wh = wide_box
        cv2.rectangle(debug, (int(wx), int(wy)), (int(wx + ww), int(wy + wh)),
                      (255, 255, 0), 1)
    return debug


# -------------------------------------------------------------------- planner


class CropPlanner:
    def __init__(self, cfg):
        self.cfg = cfg

    def run_scene(self, page, scene, state, force=False):
        try:
            return self._run(page, scene, state, force)
        except Exception:
            gc.collect()
            raise

    def _run(self, page, scene, state, force):
        try:
            if not isinstance(page, int) or isinstance(page, bool) or page < 1:
                raise CropError("page must be a positive integer")
            if not isinstance(scene, int) or isinstance(scene, bool) or scene < 1:
                raise CropError("scene must be a positive integer")
            cfg = self.cfg
            key = f"page_{page:03d}_scene_{scene:03d}"

            from pipeline.visual_planner import timeline_path
            from pipeline.knowledge import load_page_knowledge

            out_timeline = timeline_path(cfg, page, scene)
            if not out_timeline.is_file():
                raise CropError(f"timeline file not found: {out_timeline} (run the 'plan' stage first)")
            timeline = json.loads(out_timeline.read_text(encoding="utf-8"))
            shots = timeline.get("shots")
            if not isinstance(shots, list):
                raise CropError(f"timeline file {out_timeline} has no shots list")

            scenes_dir = Path(cfg.output.shots_dir) / key
            crops_dir = Path(cfg.output.crops_dir) / key
            target_width, target_height = compute_target(cfg)
            if state and not force and state.item_done(key, "crops_completed") and scenes_dir.is_dir():
                return self._skip(page, scene, scenes_dir)

            knowledge = load_page_knowledge(cfg, page)
            by_id = {panel["panel_id"]: panel for panel in knowledge.get("panels") or []}

            shot_results = []
            errors = []
            for shot in shots:
                try:
                    shot_results.append(self._process_shot(
                        shot, by_id, scenes_dir, crops_dir,
                        (target_width, target_height),
                    ))
                except CropError as exc:
                    errors.append(f"{shot.get('shot_id', '?')}: {exc}")
                finally:
                    gc.collect()

            if errors:
                LOG.error("%s crops: %d shot(s) failed", key, len(errors))
                return {
                    "result": "error", "page": page, "scene": scene,
                    "message": "; ".join(errors), "shot_errors": errors,
                    "shot_count": len(shot_results), "errors": errors,
                }
            if state:
                state.mark_item_done(key, "crops_completed")
            LOG.info("%s crops: %d shot(s) planned -> %s", key, len(shot_results), crops_dir)
            return {
                "result": "ok", "page": page, "scene": scene,
                "scene_id": timeline.get("scene_id"),
                "target": {"width": target_width, "height": target_height,
                           "aspect": f"{target_width}:{target_height}"},
                "shots_dir": str(scenes_dir), "crops_dir": str(crops_dir),
                "shot_count": len(shot_results),
                "shots": shot_results, "errors": [],
            }
        except CropError as exc:
            LOG.error("page %s scene %s crops failed: %s", page, scene, exc)
            return {
                "result": "error", "page": page, "scene": scene, "message": str(exc),
            }
        except Exception as exc:  # pragma: no cover - defensive
            LOG.exception("crop planning failure")
            return {
                "result": "error", "page": page, "scene": scene,
                "message": f"crop planning error: {exc}",
            }
        finally:
            gc.collect()

    def _process_shot(self, shot, by_id, scenes_dir, crops_dir, target):
        shot_id = shot.get("shot_id") or "shot_001"
        primary = shot.get("primary_panel")
        record = by_id.get(primary)
        if record is None:
            raise CropError(f"panel {primary!r} missing from page knowledge")
        image_path = Path(record["image"])
        if not image_path.is_file():
            raise CropError(f"panel image not found: {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise CropError(f"cannot read panel image: {image_path}")
        try:
            height, width = image.shape[:2]
            intent = visual_intent(shot.get("visual_intent"))
            regions = gather_regions(record, image, self.cfg)
            box, strategy, letterbox = compute_crop(width, height, intent, regions, self.cfg)
            sx, sy, sw, sh = snap_box(box)

            target_w, target_h = target
            crop = {"x": sx, "y": sy, "width": sw, "height": sh}
            target = {"width": target_w, "height": target_h,
                      "aspect": f"{target_w}:{target_h}",
                      "aspect_ratio": round(target_w / target_h, 4)}
            payload = {
                "shot_id": shot_id,
                "segment_id": shot.get("segment_id"),
                "primary_panel": primary,
                "visual_intent": intent,
                "camera": shot.get("camera"),
                "panel_image": str(image_path),
                "panel_size": {"width": width, "height": height},
                "crop": crop,
                "strategy": strategy,
                "aspect_ratio": round(aspect_ratio(box), 4),
                "letterbox": letterbox,
                "target": target,
                "regions": [
                    {"id": region["label"], "kind": region["kind"],
                     "weight": region["weight"],
                     "bbox": {"x": int(region["bbox"][0]), "y": int(region["bbox"][1]),
                              "width": int(region["bbox"][2]), "height": int(region["bbox"][3])}}
                    for region in regions
                ],
            }

            cropped = image[sy:sy + sh, sx:sx + sw]
            scenes_dir.mkdir(parents=True, exist_ok=True)
            crops_dir.mkdir(parents=True, exist_ok=True)
            fmt = self._image_format()
            shot_image = crops_dir / f"{shot_id}.{fmt}"
            out_img = self._letterbox_to_target(cropped, target_w, target_h) \
                if letterbox and not self._is_target_aspect(cropped, target_w, target_h) \
                else cropped
            cv2.imwrite(str(shot_image), out_img, self._jpeg_params())
            payload["out_image"] = str(shot_image)

            if self._debug_enabled():
                debug = draw_debug(image, (sx, sy, sw, sh), regions)
                debug_image = crops_dir / f"{shot_id}_debug.{fmt}"
                cv2.imwrite(str(debug_image), debug, self._jpeg_params())
                payload["debug_image"] = str(debug_image)

            json_path = scenes_dir / f"{shot_id}.json"
            self._write_json(json_path, payload)

            return {
                "shot_id": shot_id, "ok": True,
                "panel": primary, "intent": intent,
                "crop": crop, "strategy": strategy, "letterbox": letterbox,
                "aspect_ratio": payload["aspect_ratio"],
                "region_count": len(regions),
                "json": str(json_path), "image": str(shot_image),
                "debug_image": payload.get("debug_image"),
            }
        finally:
            del image

    def _is_target_aspect(self, img, target_w, target_h):
        ih, iw = img.shape[:2]
        cur = iw / ih if ih else TARGET_ASPECT
        return abs(cur - (target_w / target_h)) <= 0.01

    def _letterbox_to_target(self, img, target_w, target_h):
        """Scale `img` to fit inside a target-ratio canvas, centring it.

        The whole artwork is preserved (aspect held) and the letterbox/pillarbox
        bars fill the remainder. This guarantees a true 16:9 frame with no
        distortion, so the renderer's resize-to-16:9 never stretches the art.
        """
        ih, iw = img.shape[:2]
        if iw <= 0 or ih <= 0:
            return img
        scale = min(target_w / float(iw), target_h / float(ih))
        nw = max(1, int(round(iw * scale)))
        nh = max(1, int(round(ih * scale)))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        ox, oy = (target_w - nw) // 2, (target_h - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = resized
        return canvas

    def _image_format(self):
        crops_cfg = getattr(self.cfg, "crops", None)
        return "jpg" if crops_cfg is None else str(crops_cfg.get("format", "jpg"))

    def _jpeg_params(self):
        crops_cfg = getattr(self.cfg, "crops", None)
        quality = int(crops_cfg.get("jpeg_quality", 90)) if crops_cfg else 90
        return [cv2.IMWRITE_JPEG_QUALITY, max(int(quality), 1)]

    def _debug_enabled(self):
        crops_cfg = getattr(self.cfg, "crops", None)
        return bool(crops_cfg.get("debug", True)) if crops_cfg else True

    def _write_json(self, path, payload):
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    def _skip(self, page, scene, scenes_dir):
        return {
            "result": "skipped", "page": page, "scene": scene,
            "shots_dir": str(scenes_dir),
        }