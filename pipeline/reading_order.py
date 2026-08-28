"""Manga panel reading-order inference, tuned for low RAM.

Determines the likely reading sequence of detected panels purely from the
bounding boxes in panels/page_XXX/panels.json - no panel images are loaded.
A greedy walk picks the next panel with a modular, weighted score so the
algorithm can be improved later: row overlap (same visual band), reading
direction alignment (rtl/ltr), downward bias when leaving a row, proximity
and panel size. Within a row, panels advance in the reading direction; once
a row is exhausted the next line starts at its reading-start edge.

The original page image is loaded only when building the final debug
visualization (reading_order_debug.jpg) and is released before returning.
"""

import gc
import json
import logging
import math
import os
from pathlib import Path

import cv2
import numpy as np

LOG = logging.getLogger("mangaexplainer")

DIRECTIONS = ("rtl", "ltr")
SIGN = {"rtl": -1, "ltr": 1}

ROW_OVERLAP = "row_overlap"
HORIZONTAL = "horizontal"
VERTICAL = "vertical"
DISTANCE = "distance"
SIZE = "size"


class _Box:
    __slots__ = (
        "id", "index", "x", "y", "w", "h",
        "left", "top", "right", "bottom", "cx", "cy", "area",
    )

    def __init__(self, pid, x, y, w, h, index):
        self.id = pid
        self.index = index
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
        self.left = self.x
        self.top = self.y
        self.right = self.x + self.w
        self.bottom = self.y + self.h
        self.cx = (self.left + self.right) / 2.0
        self.cy = (self.top + self.bottom) / 2.0
        self.area = self.w * self.h


def _valid_bbox(bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    if any(isinstance(value, bool) for value in bbox):
        return None
    try:
        x, y, w, h = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x, y, w, h)):
        return None
    if w <= 0 or h <= 0:
        return None
    return int(x), int(y), int(w), int(h)


class _Ctx:
    def __init__(self, direction, page_w, page_h, params):
        self.direction = direction
        self.sign = SIGN[direction]
        self.row_threshold = float(params.get("row_overlap_ratio", 0.5))
        weights = params.get("weights") or {}
        self.w_overlap = float(weights.get(ROW_OVERLAP, 2.0))
        self.w_horizontal = float(weights.get(HORIZONTAL, 1.5))
        self.w_vertical = float(weights.get(VERTICAL, 0.3))
        self.w_distance = float(weights.get(DISTANCE, 0.5))
        self.w_size = float(weights.get(SIZE, 0.0))
        self.W = float(max(page_w, 1))
        self.H = float(max(page_h, 1))
        self.D = math.hypot(self.W, self.H)
        self.max_area = 1.0


def _overlap_ratio(a, b):
    overlap = min(a.bottom, b.bottom) - max(a.top, b.top)
    if overlap <= 0:
        return 0.0
    return overlap / min(a.h, b.h)


def _ahead(a, b, sign):
    return sign * (b.cx - a.cx) > 0


def _row_start(unread, sign):
    min_top = min(box.top for box in unread)
    starters = [box for box in unread if box.top == min_top]
    if sign < 0:
        return max(starters, key=lambda box: (box.right, box.left))
    return min(starters, key=lambda box: (box.left, box.right))


def _factor_overlap(a, b):
    return _overlap_ratio(a, b)


def _factor_horizontal(a, b, ctx):
    if ctx.direction == "rtl":
        raw = (a.cx - b.cx) / ctx.W
    else:
        raw = (b.cx - a.cx) / ctx.W
    return max(0.0, min(1.0, (raw + 1.0) / 2.0))


def _factor_vertical(a, b, ctx):
    return max(0.0, min(1.0, (b.cy - a.cy) / ctx.H))


def _factor_distance(a, b, ctx):
    dist = math.hypot(b.cx - a.cx, b.cy - a.cy)
    return max(0.0, 1.0 - dist / ctx.D)


def _factor_size(b, ctx):
    return b.area / ctx.max_area


def _next_score(a, b, ctx):
    score = 0.0
    score += ctx.w_overlap * _factor_overlap(a, b)
    score += ctx.w_horizontal * _factor_horizontal(a, b, ctx)
    score += ctx.w_vertical * _factor_vertical(a, b, ctx)
    score += ctx.w_distance * _factor_distance(a, b, ctx)
    score += ctx.w_size * _factor_size(b, ctx)
    return score


def _pick_next(current, unread, ctx):
    same_row = [
        box for box in unread
        if _overlap_ratio(current, box) >= ctx.row_threshold
    ]
    ahead = [box for box in same_row if _ahead(current, box, ctx.sign)]
    if ahead:
        if ctx.sign < 0:
            return max(ahead, key=lambda box: (_next_score(current, box, ctx), box.right))
        return max(ahead, key=lambda box: (_next_score(current, box, ctx), -box.left))
    return _row_start(unread, ctx.sign)


class ReadingOrder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = Path(cfg.root_dir)
        self.panels_dir = Path(cfg.output.panels_dir)
        self.pages_dir = Path(cfg.output.pages_dir)
        self.image_format = str(cfg.images.format).lower().lstrip(".")
        self.quality = int(cfg.panels.jpeg_quality)
        self.params = cfg.reading.to_dict()
        self.direction = str(self.params.get("direction", "rtl")).lower()
        if self.direction not in DIRECTIONS:
            raise ValueError(
                f"unknown reading direction {self.direction!r} (expected rtl or ltr)"
            )

    def page_out_dir(self, page_num):
        return self.panels_dir / f"page_{page_num:03d}"

    def page_image(self, page_num):
        return self.pages_dir / f"page_{page_num:03d}.{self.image_format}"

    def _rel(self, path):
        try:
            return Path(path).resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return Path(path).as_posix()

    def _error(self, page_num, message):
        LOG.error(message)
        return {
            "status": "error",
            "page": page_num,
            "direction": self.direction,
            "out_dir": None,
            "manifest": None,
            "order_path": None,
            "debug_image": None,
            "count": 0,
            "ordered": 0,
            "ignored": [],
            "order": [],
            "error": message,
        }

    def compute_order(self, panels, direction=None, page_w=0, page_h=0):
        """Pure box-geometry ordering. Returns {"order": [...ids], "ignored": [...]}."""
        direction = (direction or self.direction).lower()
        if direction not in DIRECTIONS:
            raise ValueError(f"unknown reading direction {direction!r} (expected rtl or ltr)")
        sign = SIGN[direction]

        boxes = []
        ignored = []
        for index, panel in enumerate(panels):
            pid = panel.get("id") if isinstance(panel, dict) else None
            if not isinstance(pid, str) or not pid:
                ignored.append(pid if pid else f"#panel-{index}")
                continue
            dims = _valid_bbox(panel.get("bbox"))
            if dims is None:
                ignored.append(pid)
                continue
            boxes.append(_Box(pid, *dims, index))

        ctx = _Ctx(direction, page_w or 0, page_h or 0, self.params)
        if boxes:
            ctx.W = max(box.right for box in boxes)
            ctx.H = max(box.bottom for box in boxes)
            ctx.D = math.hypot(ctx.W, ctx.H)
            ctx.max_area = max(box.area for box in boxes)

        if not boxes:
            return {"order": [], "ignored": ignored}

        unread = list(boxes)
        order = []
        current = _row_start(unread, sign)
        while current is not None:
            order.append(current)
            unread.remove(current)
            if not unread:
                break
            current = _pick_next(current, unread, ctx)

        return {"order": [box.id for box in order], "ignored": ignored}

    def _load_manifest(self, path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError, ValueError):
            return None
        if isinstance(data, dict) and isinstance(data.get("panels"), list):
            return data
        return None

    def _atomic_write(self, path, payload):
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    def _write_manifest(self, manifest, order, ignored):
        ranks = {pid: rank for rank, pid in enumerate(order, 1)}
        for panel in manifest.get("panels", []):
            pid = panel.get("id") if isinstance(panel, dict) else None
            panel["reading_order"] = ranks.get(pid) if pid in ranks else None

    def _save_order_json(self, page_num, out_dir, order, ignored):
        payload = {
            "page": int(page_num),
            "direction": self.direction,
            "order": order,
        }
        if ignored:
            payload["ignored"] = ignored
        path = out_dir / "reading_order.json"
        self._atomic_write(path, payload)
        return path

    def _save_debug(self, page_num, out_dir, manifest, order):
        panels = manifest.get("panels", [])
        by_id = {
            panel["id"]: panel
            for panel in panels
            if isinstance(panel, dict) and isinstance(panel.get("id"), str)
        }
        base = out_dir / "debug.jpg"
        page_file = self.page_image(page_num)
        img = cv2.imread(str(base)) if base.is_file() else None
        if img is None and page_file.is_file():
            img = cv2.imread(str(page_file), cv2.IMREAD_COLOR)
        if img is None:
            img = np.full((1000, 800, 3), 255, np.uint8)
        try:
            for panel in panels:
                if not isinstance(panel, dict):
                    continue
                dims = _valid_bbox(panel.get("bbox"))
                if dims is None:
                    continue
                x, y, w, h = dims
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            for rank, pid in enumerate(order, 1):
                panel = by_id.get(pid)
                if panel is None or not isinstance(panel, dict):
                    continue
                dims = _valid_bbox(panel.get("bbox"))
                if dims is None:
                    continue
                x, y, w, h = dims
                cx, cy = x + w // 2, y + h // 2
                scale = max(0.6, min(2.0, min(w, h) / 80.0))
                text = str(rank)
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3
                )
                pad = 10
                bx0 = cx - tw // 2 - pad
                by0 = cy - th // 2 - 2 * pad // 3
                bx1 = cx + tw // 2 + pad
                by1 = cy + th // 2 + 2 * pad // 3
                cv2.rectangle(img, (bx0, by0), (bx1, by1), (0, 0, 0), -1)
                cv2.putText(
                    img,
                    text,
                    (bx0 + pad, by1 - pad),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    (255, 255, 255),
                    3,
                    lineType=cv2.LINE_AA,
                )
            path = out_dir / "reading_order_debug.jpg"
            cv2.imwrite(
                str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
            )
            return path
        finally:
            img = None
            gc.collect()

    def detect_page(self, page_num, state, force=False):
        page_num = int(page_num)
        if page_num < 1:
            return self._error(page_num, f"invalid page number {page_num} (must be >= 1)")
        out_dir = self.page_out_dir(page_num)
        manifest = out_dir / "panels.json"
        order_path = out_dir / "reading_order.json"
        key = f"order_page_{page_num:03d}"

        if not force and order_path.is_file():
            try:
                with open(order_path, "r", encoding="utf-8") as handle:
                    saved = json.load(handle)
            except (OSError, json.JSONDecodeError, ValueError):
                saved = None
            if isinstance(saved, dict) and isinstance(saved.get("order"), list):
                state.mark_page_done(key)
                LOG.info("reading order page %d already computed; skipped", page_num)
                return {
                    "status": "skipped",
                    "page": page_num,
                    "direction": self.direction,
                    "out_dir": out_dir,
                    "manifest": manifest,
                    "order_path": order_path,
                    "debug_image": out_dir / "reading_order_debug.jpg",
                    "count": len(saved.get("order", [])),
                    "ordered": len(saved.get("order", [])),
                    "ignored": saved.get("ignored", []),
                    "order": saved.get("order", []),
                    "error": None,
                }

        data = self._load_manifest(manifest)
        if data is None:
            return self._error(
                page_num,
                f"panel detection not found at {manifest} "
                f"(run: python main.py panels --page {page_num} first)",
            )

        panels = data.get("panels", [])
        result = self.compute_order(panels)
        order = result["order"]
        ignored = result["ignored"]

        out_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest(data, order, ignored)
        self._atomic_write(manifest, data)
        self._save_order_json(page_num, out_dir, order, ignored)
        debug_path = self._save_debug(page_num, out_dir, data, order)

        state.mark_page_done(key)
        LOG.info(
            "reading order page %d: %d/%d panels (ignored %d, direction %s)",
            page_num, len(order), len(panels), len(ignored), self.direction,
        )
        return {
            "status": "detected",
            "page": page_num,
            "direction": self.direction,
            "out_dir": out_dir,
            "manifest": manifest,
            "order_path": order_path,
            "debug_image": debug_path,
            "count": len(panels),
            "ordered": len(order),
            "ignored": ignored,
            "order": order,
            "error": None,
        }