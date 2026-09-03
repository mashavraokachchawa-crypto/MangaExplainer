"""Panel cleaning: remove overlay noise before OCR + VLM understanding.

Raw panel crops carry a lot of "overlay" that is not part of the artwork a
narrator cares about: banner strips (volume/series title bars along an edge),
caption / signage text boxes (solid fill with a closed dark border), and big
bold sound-effect stamps. These confuse both the OCR pass and the VLM's
scene understanding.

This module detects those regions and reconstructs the underlying art with
cv2.inpaint (TELEA), so understanding can look at just the panel. Cleaned
images are written to clean_dir/<page>/panel_NNN.jpg; every detected region
is recorded per page in that page's clean_manifest.json, which is also what
the live dashboard uses to show a before/after while the stage runs.

Detection is deliberately conservative: a region is only cleared when it is
clearly a band / box / large stamp. Everything is governed by panels.clean
config knobs, and nothing here touches the original panels/ images.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

LOG = logging.getLogger("mangaexplainer")

CLEAN_MANIFEST = "clean_manifest.json"


def clean_settings(cfg):
    """Resolve the panels.clean settings, or None when cleaning is off."""
    panels = getattr(cfg, "panels", None)
    if panels is None:
        return None
    node = panels.get("clean")
    if node is None:
        return None
    if not bool(node.get("enabled", True)):
        return None
    return node


def clean_panel_source(cfg, page, panel, ext=None):
    """Return the cleaned panel path if it exists, else None.

    OCR / the analysis processor call this first: when a clean copy is on
    disk they read that instead of the raw panel. Keeping the check here
    means a disabled/empty clean step degrades transparently to raw panels.
    """
    settings = clean_settings(cfg)
    if settings is None:
        return None
    try:
        clean_dir = Path(cfg.output.clean_dir)
        if ext is None:
            ext = str(cfg.images.format).lower().lstrip(".") \
                if getattr(getattr(cfg, "images", None), "format", None) else "jpg"
        path = clean_dir / f"page_{int(page):03d}" / f"panel_{int(panel):03d}.{ext}"
    except (AttributeError, TypeError, ValueError):
        return None
    return path if path.is_file() else None


def clean_panel(cfg, root, page, panel, force=False):
    """Clean one panel and record the regions in its page manifest.

    Returns {status, page, panel, source, cleaned, debug, regions,
             removed, error}. status is one of cleared / unchanged /
    skipped / error. When panels.clean.enabled is False the call is a no-op
    (status "skipped", reason "disabled").
    """
    page = int(page)
    panel = int(panel)
    root_path = Path(root)
    settings = clean_settings(cfg)
    panels_dir = Path(cfg.output.panels_dir)
    image_format = str(cfg.images.format).lower().lstrip(".")
    src = panels_dir / f"page_{page:03d}" / f"panel_{panel:03d}.{image_format}"
    if settings is None:
        return _ok(page, panel, status="skipped",
                   reason="clean disabled in config", source=src)
    if not src.is_file():
        return _ok(page, panel, status="skipped",
                   reason="panel image not found", source=src)

    clean_dir = Path(cfg.output.clean_dir)
    out = clean_dir / f"page_{page:03d}" / f"panel_{panel:03d}.{image_format}"
    debug = clean_dir / f"page_{page:03d}" / f"panel_{panel:03d}_debug.jpg"
    if not force and out.is_file():
        return _ok(page, panel, status="skipped", reason="already cleaned",
                   source=src, cleaned=out, debug=debug)

    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return _ok(page, panel, status="error",
                   reason=f"cannot read panel image {src}", source=src)

    cleaned, regions = detect_and_clean(img, settings)

    removed = {"banners": 0, "textboxes": 0, "sfx": 0}
    for r in regions:
        kind = r.get("kind")
        if kind in removed:
            removed[kind] += 1

    result = _ok(page, panel, status="cleared" if regions else "unchanged",
                 source=src, cleaned=out, debug=debug, regions=regions,
                 removed=removed)

    if not bool(settings.get("write_clean", True)):
        return result

    manifest = _load_manifest(clean_dir, page)
    entry = {"panel": panel, "status": result["status"],
             "source": _rel(src, root_path),
             "regions": regions, "removed": removed}
    manifest[str(panel)] = entry

    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        quality = int(getattr(cfg.panels, "jpeg_quality", 85) or 85)
        if not cv2.imwrite(str(out), cleaned,
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality]):
            result["status"] = "error"
            result["reason"] = f"failed to write clean image {out}"
        else:
            if bool(settings.get("debug", True)):
                cv2.imwrite(str(debug), _debug_view(img, regions))
            _write_manifest(clean_dir, page, manifest)
    except OSError as exc:
        result["status"] = "error"
        result["reason"] = f"IO error: {exc}"
    finally:
        img = None
        cleaned = None
    return result


def _ok(page, panel, status, reason=None, source=None, cleaned=None,
        debug=None, regions=None, removed=None):
    return {
        "status": status,
        "reason": reason,
        "page": page,
        "panel": panel,
        "source": str(source) if source else None,
        "cleaned": str(cleaned) if cleaned else None,
        "debug": str(debug) if debug else None,
        "regions": regions or [],
        "removed": removed or {"banners": 0, "textboxes": 0, "sfx": 0},
    }


def _rel(path, root):
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _load_manifest(clean_dir, page):
    manifest = clean_dir / f"page_{page:03d}" / CLEAN_MANIFEST
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text("utf-8"))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    return {}


def _write_manifest(clean_dir, page, manifest):
    manifest_path = clean_dir / f"page_{page:03d}" / CLEAN_MANIFEST
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), "utf-8")
    except OSError as exc:
        LOG.warning("could not write clean manifest %s: %s", manifest_path, exc)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_and_clean(img, settings):
    """Detect overlay noise in a BGR panel and return (clean, regions).

    settings is the panels.clean dict. Never mutates img.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    area = float(h * w)

    regions = []
    mask = np.zeros((h, w), dtype=np.uint8)

    for y0, y1 in _banner_bands(gray, settings):
        kind = "banners"
        regions.append({"kind": kind, "x": 0, "y": int(y0), "w": w,
                        "h": int(y1 - y0),
                        "area_ratio": round((y1 - y0) * w / area, 4)})
        mask[y0:y1, :] = 255

    boxed_mask = np.zeros((h, w), dtype=np.uint8)
    for box in _text_boxes(gray, settings):
        x, y, bw, bh = box
        boxed_mask[y:y + bh, x:x + bw] = 255
    _apply_inpaint_mask(mask, boxed_mask, settings, regions, "textboxes",
                        gray.shape)

    sfx_mask = _sfx_mask(gray, boxed_mask, settings)
    _apply_inpaint_mask(mask, sfx_mask, settings, regions, "sfx", gray.shape)

    radius = int(settings.get("inpaint_radius", 4) or 4)
    clean = img.copy()
    if regions:
        clean = cv2.inpaint(clean, mask, radius, cv2.INPAINT_TELEA)
    return clean, regions


def _apply_inpaint_mask(mask, sub, settings, regions, kind, shape):
    h, w = shape
    pad = int(settings.get("bbox_pad", 6) or 6)
    k = 2 * pad + 1
    kernel = np.ones((k, k), np.uint8)
    dilated = cv2.dilate(sub, kernel, iterations=1)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    area = float(h * w)
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        regions.append({"kind": kind, "x": int(x), "y": int(y),
                        "w": int(bw), "h": int(bh),
                        "area_ratio": round(a / area, 4)})
    mask[:, :] = cv2.bitwise_or(mask, dilated)


def _banner_bands(gray, settings):
    """Horizontal uniform bands along the top/bottom of a panel.

    A banner is a run of rows at an edge that is almost constant in value and
    noticeably flatter than the artwork around it. Returns [(y0, y1), ...].
    """
    h, w = gray.shape
    max_ratio = float(settings.get("strip_max_height_ratio", 0.18) or 0.18)
    limit = max(8, int(h * max_ratio))
    dev = float(settings.get("strip_row_dev", 24.0) or 24.0)
    min_run = max(6, min(limit, h // 60))
    panel_std = float(np.std(gray))
    tie = max(dev * 0.5, panel_std * 0.22)

    rows = np.array([np.std(gray[r, :]) for r in range(h)], dtype=np.float64)

    def scan(direction):
        rng = range(h) if direction < 0 else range(h - 1, -1, -1)
        bands, start = [], None
        for r in rng:
            bottom_edge = r == 0 if direction < 0 else r == h - 1
            if rows[r] <= tie and (start is not None or bottom_edge
                                   or _near_edge(r, h, limit)):
                if start is None:
                    start = r
            else:
                if start is not None:
                    y0, y1 = min(start, r), max(start, r)
                    if y1 - y0 >= min_run:
                        bands.append((y0, y1))
                    start = None
        if start is not None:
            if direction < 0:
                bands.append((start, h))
            else:
                bands.append((0, start + 1))
        return bands

    bands = scan(-1) + scan(+1)
    # dedupe overlapping hits from both scans
    merged = []
    for y0, y1 in sorted(bands):
        if merged and y0 <= merged[-1][1]:
            prev = merged[-1]
            merged[-1] = (min(prev[0], y0), max(prev[1], y1))
        else:
            merged.append((y0, y1))
    return merged


def _near_edge(r, h, limit):
    return r < limit or r >= h - limit


def _text_boxes(gray, settings):
    """Rectangular caption/text boxes: solid fill with a closed dark border."""
    h, w = gray.shape
    area = float(h * w)
    min_ratio = float(settings.get("min_textbox_area_ratio", 0.02) or 0.02)
    max_ratio = float(settings.get("max_textbox_area_ratio", 0.55) or 0.55)
    min_area = area * min_ratio
    max_area = area * max_ratio

    fill = np.zeros(gray.shape, np.uint8)
    fill[gray > 160] = 255                      # bright/solid fills
    ink = (gray < 130).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    ink_dilated = cv2.dilate(ink, kernel, iterations=2)
    boxes = []
    ncc, _, stats, _ = cv2.connectedComponentsWithStats(fill, 8)
    for i in range(1, ncc):
        x, y, bw, bh, a = stats[i]
        if a < min_area or a > max_area:
            continue
        if not (0.20 <= bw / bh <= 5.0):        # not an elongated strip
            continue
        rect = a / float(bw * bh)
        if rect < 0.72:                          # not boxy enough
            continue
        # a box interior stays near-bright (a caption panel on manga art);
        # motley bright artwork is far more varied
        box_gray = gray[y:y + bh, x:x + bw]
        bright_ratio = float(np.mean(box_gray > 160)) if box_gray.size else 0.0
        if bright_ratio < 0.8:
            continue
        # the border must be closed dark ink
        ring = _box_ring(ink_dilated, x, y, bw, bh)
        if ring < 0.35:
            continue
        boxes.append((int(x), int(y), int(bw), int(bh)))
    return boxes


def _box_ring(ink, x, y, w, h, t=2):
    """Fraction of the band just OUTSIDE a box's edges that is dark ink.

    A caption/text box in manga is a bright fill wrapped in a dark frame, so
    the 2px ring right around the four edges is mostly ink. Plain bright
    shapes in artwork have no such frame and score low.
    """
    h_, w_ = ink.shape
    if w < 4 or h < 4 or x < 0 or y < 0 or x + w > w_ or y + h > h_:
        return 0.0
    top = ink[max(0, y - t):y, max(0, x - t):min(w_, x + w + t)].ravel()
    bottom = ink[y + h:min(h_, y + h + t), max(0, x - t):min(w_, x + w + t)].ravel()
    left = ink[max(0, y - t):min(h_, y + h + t), max(0, x - t):x].ravel()
    right = ink[max(0, y - t):min(h_, y + h + t), x + w:min(w_, x + w + t)].ravel()
    ring = np.concatenate([top, bottom, left, right])
    return float(np.mean(ring > 0)) if ring.size else 0.0


def _sfx_mask(gray, boxed, settings):
    """Large near-black bold strokes lying on the art (outside text boxes).

    Printed manga ink is near-black (belongs to printed manga Ink). SFX
    stamps are coalesced masses of that ink, whereas halftone / textured art
    stays mid-gray. Detect on gray < 80 so artwork texture does not flood the
    ink mask, then keep the big solid blobs.
    """
    h, w = gray.shape
    area = float(h * w)
    min_ratio = float(settings.get("min_sfx_area_ratio", 0.015) or 0.015)
    min_area = area * min_ratio

    ink = (gray < 80).astype(np.uint8) * 255
    ink[boxed > 0] = 0

    ncc, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    out = np.zeros(gray.shape, np.uint8)
    for i in range(1, ncc):
        x, y, bw, bh, a = stats[i]
        if a < min_area:
            continue
        if bw < 4 or bh < 4 or bw / bh > 7 or bh / bw > 7:
            continue
        solidity = a / float(bw * bh)
        if not (0.30 <= solidity <= 0.97):
            continue
        out[y:y + bh, x:x + bw] = 255
    return out


def _debug_view(img, regions):
    """Original panel with color-coded region boxes (red band, green box,
    yellow SFX) plus a translucent fill so the dashboard shows what got cut."""
    view = img.copy()
    colors = {"banners": (0, 0, 255), "textboxes": (0, 180, 0),
              "sfx": (0, 200, 200)}
    overlay = view.copy()
    for r in regions:
        color = colors.get(r.get("kind"), (255, 255, 255))
        x, y, w, h = r.get("x", 0), r.get("y", 0), r.get("w", 1), r.get("h", 1)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)
    return cv2.addWeighted(overlay, 0.25, view, 0.75, 0)