"""Manga panel detection on a single extracted page, tuned for low RAM.

Strategy "gutter_flood" segments panels via the white gutter network: the
paper (gutter/margin) connected to the page border is flooded, so every
panel becomes the enclosed region still separated from that network
(border ink + interior paper). Panels are written straight to disk one at a
time, followed by panels.json and a debug view. The strategy is pluggable
(rewrite/extend for irregular artwork later).
"""
import gc
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from pipeline import geometry

LOG = logging.getLogger("mangaexplainer")


class GutterFloodStrategy:
    name = "gutter_flood"

    def __init__(self, params):
        self.border_kernel = max(1, int(params.get("border_kernel", 3)))
        self.close_kernel = int(params.get("close_kernel", 3))
        self.close_kernel = 3 if self.close_kernel < 3 else self.close_kernel

    def detect(self, page_bgr):
        gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
        if cv2.countNonZero(gray) == 0:
            return []
        _, paper = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((self.border_kernel, self.border_kernel), np.uint8)
        sealed = cv2.erode(paper, kernel)
        height, width = sealed.shape
        flood = sealed.copy()
        mask = np.zeros((height + 2, width + 2), np.uint8)
        cv2.floodFill(flood, mask, (0, 0), 0)

        enclosed = (flood == 255)
        if not enclosed.any():
            return []

        enclosed_u8 = enclosed.astype(np.uint8) * 255
        adjacent = cv2.dilate(enclosed_u8, np.ones((3, 3), np.uint8))
        ink = (paper == 0).astype(np.uint8) * 255
        panel_mask = cv2.bitwise_or(enclosed_u8, cv2.bitwise_and(ink, adjacent))
        if self.close_kernel >= 3:
            close_k = np.ones((self.close_kernel, self.close_kernel), np.uint8)
            panel_mask = cv2.morphologyEx(panel_mask, cv2.MORPH_CLOSE, close_k)

        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            panel_mask, connectivity=8
        )
        boxes = []
        for label in range(1, labels.max() + 1):
            x, y, cw, ch, area = stats[label]
            if area <= 0:
                continue
            coverage = area / (cw * ch) if cw * ch > 0 else 0.0
            confidence = round(min(0.99, 0.4 + 0.6 * coverage), 3)
            boxes.append(
                {
                    "box": [int(x), int(y), int(cw), int(ch)],
                    "area": int(area),
                    "confidence": confidence,
                }
            )
        return boxes


STRATEGIES = {"gutter_flood": GutterFloodStrategy}


class PanelDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = Path(cfg.root_dir)
        self.pages_dir = Path(cfg.output.pages_dir)
        self.panels_dir = Path(cfg.output.panels_dir)
        self.image_format = str(cfg.images.format).lower().lstrip(".")
        self.params = cfg.panels.to_dict()
        strategy_name = str(self.params.get("strategy", "gutter_flood"))
        if strategy_name not in STRATEGIES:
            raise ValueError(f"unknown panel strategy: {strategy_name}")
        self.strategy = STRATEGIES[strategy_name](self.params)

    def page_file(self, page_num):
        return self.pages_dir / f"page_{page_num:03d}.{self.image_format}"

    def page_out_dir(self, page_num):
        return self.panels_dir / f"page_{page_num:03d}"

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
            "source": None,
            "out_dir": None,
            "manifest": None,
            "count": 0,
            "panels": None,
            "error": message,
        }

    def _load_manifest(self, path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError, ValueError):
            return None
        if isinstance(data, dict) and isinstance(data.get("panels"), list):
            return data
        return None

    def _clear_dir(self, out_dir):
        if not out_dir.is_dir():
            return
        for child in out_dir.iterdir():
            try:
                if child.is_file():
                    child.unlink()
            except OSError:
                pass

    def _save_panels(self, page_num, img, boxes, out_dir):
        pad = int(self.params.get("pad_pixels", 2))
        quality = int(self.params.get("jpeg_quality", 85))
        page_h, page_w = img.shape[:2]
        panels = []
        for index, item in enumerate(boxes, 1):
            x, y, bw, bh = item["box"]
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(page_w, x + bw + pad)
            y1 = min(page_h, y + bh + pad)
            crop = img[y0:y1, x0:x1]
            fname = f"panel_{index:03d}.jpg"
            fpath = out_dir / fname
            ok = cv2.imwrite(
                str(fpath), crop, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            )
            crop = None
            if not ok:
                raise OSError(f"failed to write {fpath}")
            panels.append(
                {
                    "id": f"p{page_num:03d}_{index:03d}",
                    "image": self._rel(fpath),
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "area": (x1 - x0) * (y1 - y0),
                    "confidence": item["confidence"],
                }
            )
        return panels

    def _save_manifest(self, page_num, page_file, panels, manifest):
        payload = {
            "page": page_num,
            "source": self._rel(page_file),
            "panels": panels,
        }
        tmp = manifest.with_name(manifest.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, manifest)

    def _save_debug(self, img, boxes, out_dir):
        quality = int(self.params.get("jpeg_quality", 85))
        viz = img.copy()
        for index, item in enumerate(boxes, 1):
            x, y, w, h = item["box"]
            cv2.rectangle(viz, (x, y), (x + w, y + h), (0, 0, 255), 2)
            label = f"P{index}"
            tx = max(2, int(x) + 4)
            ty = max(16, int(y) + 18)
            cv2.putText(
                viz, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
            )
        cv2.imwrite(
            str(out_dir / "debug.jpg"),
            viz,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        viz = None

    def detect_page(self, page_num, state, force=False):
        page_num = int(page_num)
        if page_num < 1:
            return self._error(page_num, f"invalid page number {page_num} (must be >= 1)")
        page_file = self.page_file(page_num)
        if not page_file.is_file():
            return self._error(
                page_num,
                f"page {page_num} not found at {page_file} "
                f"(run: python main.py extract --page {page_num} first)",
            )

        key = f"panels_page_{page_num:03d}"
        out_dir = self.page_out_dir(page_num)
        manifest = out_dir / "panels.json"

        if not force and manifest.is_file():
            data = self._load_manifest(manifest)
            if data is not None:
                state.mark_page_done(key)
                count = len(data.get("panels", []))
                LOG.info("page %d panels already detected (%d); skipped", page_num, count)
                return {
                    "status": "skipped",
                    "page": page_num,
                    "source": data.get("source"),
                    "out_dir": out_dir,
                    "manifest": manifest,
                    "count": count,
                    "panels": data.get("panels"),
                    "error": None,
                }

        img = cv2.imread(str(page_file), cv2.IMREAD_COLOR)
        if img is None:
            return self._error(page_num, f"cannot read page image {page_file}")

        panels = []
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            if force:
                self._clear_dir(out_dir)
            boxes = self.strategy.detect(img)
            page_h, page_w = img.shape[:2]
            boxes = geometry.filter_boxes(boxes, page_w, page_h, self.params)
            panels = self._save_panels(page_num, img, boxes, out_dir)
            self._save_manifest(page_num, page_file, panels, manifest)
            self._save_debug(img, boxes, out_dir)
        finally:
            img = None
            gc.collect()

        state.mark_page_done(key)
        LOG.info("panel detection page %d: %d panels", page_num, len(panels))
        return {
            "status": "detected",
            "page": page_num,
            "source": self._rel(page_file),
            "out_dir": out_dir,
            "manifest": manifest,
            "count": len(panels),
            "panels": panels,
            "error": None,
        }