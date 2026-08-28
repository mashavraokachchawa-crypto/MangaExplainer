"""OCR of ONE manga panel with a lightweight preprocessing pipeline.

Pipeline per panel (low RAM, one image at a time):

    panels/page_001/panel_001.jpg
      -> load image
      -> optional preprocess (grayscale / contrast / denoise / resize /
         threshold) into a temporary file
      -> OCR provider -> text blocks
      -> save ocr/page_001_panel_001.json (+ classification, combined text)
      -> save ocr/page_001_panel_001_debug.jpg overlay
      -> delete temp file, release image, gc

Empty OCR is a success (text_blocks: [], combined_text: ""). Failures are
never checkpointed as completed.
"""
import gc
import json
import logging
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from pipeline.ocr_provider import (
    OCREngineUnavailable,
    OCRProviderError,
    OCRTimeout,
    create_provider,
)

LOG = logging.getLogger("mangaexplainer")

TYPES = ("dialogue", "narration", "sfx", "unknown")

_QUOTE_MARKERS = (
    "\u300c", "\u300d", "\u300e", "\u300f", "\u201c", "\u201d",
    "\u201e", "\u201f", "\u00ab", "\u00bb", '"',
)
_SFX_SMALL_KANA = "ッっ"
_SFX_VOWEL_MARK = "ー"


def _sfx_like(text):
    if any(char.isspace() for char in text):
        return False
    if len(text) > 8:
        return False
    if any(char in text for char in _SFX_SMALL_KANA):
        return True
    if _SFX_VOWEL_MARK in text:
        return True
    alnum = sum(1 for char in text if char.isalnum())
    if alnum == 0:
        return True
    return alnum / len(text) < 0.4


def classify_text_block(text, bbox=None, panel_size=None):
    """Conservative, rule-based block classification.

    Dialogue/narration/SFX/unknown. Falls back to "unknown" unless a marker
    is unambiguous - never invents a classification.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return "unknown"
    if any(char in cleaned for char in _QUOTE_MARKERS):
        return "dialogue"
    if _sfx_like(cleaned):
        return "sfx"
    return "unknown"


def preprocess_image(img, params):
    """Optional preprocessing; returns (image, effective_scale)."""
    if not bool(params.get("enabled", True)):
        return img, 1.0
    scale = 1.0

    if bool(params.get("grayscale", True)) and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    contrast = float(params.get("contrast", 1.4))
    if contrast != 1.0:
        img = cv2.convertScaleAbs(img, alpha=contrast, beta=0)

    denoise = str(params.get("denoise", "none")).lower()
    kernel = max(1, int(params.get("denoise_kernel", 3)))
    if kernel % 2 == 0:
        kernel += 1
    if denoise in ("gaussian", "median"):
        if denoise == "gaussian":
            img = cv2.GaussianBlur(img, (kernel, kernel), 0)
        else:
            img = cv2.medianBlur(img, kernel)

    factor = float(params.get("resize_scale", 1.5))
    max_pixels = max(1, int(params.get("resize_max_pixels", 800000)))
    height, width = img.shape[:2]
    if factor > 1.0 and height * width < max_pixels:
        capped = min(factor, (max_pixels / (height * width)) ** 0.5)
        if capped > 1.0:
            scale = capped
            img = cv2.resize(
                img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )

    threshold = str(params.get("threshold", "none")).lower()
    if threshold == "otsu" and len(img.shape) == 2:
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img, float(scale)


def _map_bbox(bbox, scale):
    x0 = int(round(bbox[0] / scale))
    y0 = int(round(bbox[1] / scale))
    x1 = int(round((bbox[0] + bbox[2]) / scale))
    y1 = int(round((bbox[1] + bbox[3]) / scale))
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


class OcrProcessor:
    def __init__(self, cfg, provider=None):
        self.cfg = cfg
        self.root = Path(cfg.root_dir)
        self.panels_dir = Path(cfg.output.panels_dir)
        self.ocr_dir = Path(cfg.output.ocr_dir)
        self.image_format = str(cfg.images.format).lower().lstrip(".")
        self.quality = int(cfg.panels.jpeg_quality)
        self.params = cfg.preprocess.to_dict()
        self.provider = provider

    def panel_file(self, page_num, panel_num):
        return (
            self.panels_dir
            / f"page_{page_num:03d}"
            / f"panel_{panel_num:03d}.{self.image_format}"
        )

    def out_json(self, page_num, panel_num):
        return self.ocr_dir / f"page_{page_num:03d}_panel_{panel_num:03d}.json"

    def out_debug(self, page_num, panel_num):
        return self.ocr_dir / f"page_{page_num:03d}_panel_{panel_num:03d}_debug.jpg"

    def _rel(self, path):
        try:
            return Path(path).resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return Path(path).as_posix()

    def _error(self, page_num, panel_num, message):
        LOG.error(message)
        return {
            "status": "error",
            "page": page_num,
            "panel": panel_num,
            "engine": None,
            "source": self._rel(self.panel_file(page_num, panel_num)),
            "out_file": self.out_json(page_num, panel_num),
            "debug_image": self.out_debug(page_num, panel_num),
            "count": 0,
            "blocks": [],
            "combined_text": "",
            "error": message,
        }

    def _atomic_write(self, path, payload):
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    def _save_debug(self, img, blocks, out_path):
        viz = img.copy()
        if len(viz.shape) == 2:
            viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2BGR)
        for block in blocks:
            x, y, w, h = block["bbox"]
            cv2.rectangle(viz, (x, y), (x + w, y + h), (0, 0, 255), 2)
            label = f"[{block['type'].upper()}]"
            tx = max(2, x + 4)
            ty = max(18, y - 8)
            cv2.rectangle(viz, (tx - 2, ty - 16), (tx + max(60, len(label) * 10), ty),
                          (255, 255, 255), -1)
            cv2.putText(
                viz, label, (tx, ty - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
            )
        ok = cv2.imwrite(
            str(out_path), viz, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        )
        viz = None
        if not ok:
            raise OSError(f"failed to write OCR debug image {out_path}")

    def run_panel(self, page_num, panel_num, state, force=False):
        page_num = int(page_num)
        panel_num = int(panel_num)
        if page_num < 1:
            return self._error(
                page_num, panel_num, f"invalid page number {page_num} (must be >= 1)"
            )
        if panel_num < 1:
            return self._error(
                page_num, panel_num, f"invalid panel number {panel_num} (must be >= 1)"
            )

        panel_file = self.panel_file(page_num, panel_num)
        out_file = self.out_json(page_num, panel_num)
        debug_file = self.out_debug(page_num, panel_num)
        key = f"page_{page_num:03d}_panel_{panel_num:03d}"

        if not force and out_file.is_file():
            try:
                with open(out_file, "r", encoding="utf-8") as handle:
                    saved = json.load(handle)
            except (OSError, json.JSONDecodeError, ValueError):
                saved = None
            if (
                isinstance(saved, dict)
                and "text_blocks" in saved
                and "combined_text" in saved
            ):
                state.mark_ocr_done(key)
                LOG.info("OCR %s already done; skipped", key)
                return {
                    "status": "skipped",
                    "page": page_num,
                    "panel": panel_num,
                    "engine": saved.get("engine"),
                    "source": self._rel(panel_file),
                    "out_file": out_file,
                    "debug_image": debug_file,
                    "count": len(saved.get("text_blocks", [])),
                    "blocks": saved.get("text_blocks", []),
                    "combined_text": saved.get("combined_text", ""),
                    "error": None,
                }

        if not panel_file.is_file():
            return self._error(
                page_num,
                panel_num,
                f"panel {panel_file} not found "
                f"(run: python main.py panels --page {page_num} first)",
            )

        img = cv2.imread(str(panel_file), cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return self._error(
                page_num,
                panel_num,
                f"cannot read or decode panel image {panel_file} (corrupt?)",
            )

        provider = self.provider
        if provider is None:
            try:
                provider = create_provider(self.cfg)
            except OCREngineUnavailable as exc:
                img = None
                gc.collect()
                return self._error(
                    page_num, panel_num, f"OCR engine unavailable: {exc}"
                )

        tmp_path = None
        blocks = []
        try:
            self.ocr_dir.mkdir(parents=True, exist_ok=True)
            processed, scale = preprocess_image(img, self.params)
            fd, tmp_name = tempfile.mkstemp(
                prefix="manga_ocr_", suffix=".png", dir=str(self.ocr_dir)
            )
            os.close(fd)
            tmp_path = Path(tmp_name)
            if not cv2.imwrite(str(tmp_path), processed):
                raise OSError(f"failed to write temporary image {tmp_path}")

            raw_blocks = provider.recognize(str(tmp_path))
            for block in raw_blocks:
                text = str(block.get("text", "")).strip()
                bbox = block.get("bbox")
                if not text:
                    continue
                if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                    continue
                bbox = _map_bbox([float(v) for v in bbox], scale)
                panel_h, panel_w = img.shape[:2]
                bbox[0] = max(0, min(bbox[0], panel_w - 1))
                bbox[1] = max(0, min(bbox[1], panel_h - 1))
                bbox[2] = max(1, min(bbox[2], panel_w - bbox[0]))
                bbox[3] = max(1, min(bbox[3], panel_h - bbox[1]))
                blocks.append(
                    {
                        "text": text,
                        "bbox": bbox,
                        "confidence": float(block.get("confidence", 0.0) or 0.0),
                        "type": classify_text_block(text, bbox, (panel_w, panel_h)),
                    }
                )

            combined_text = "\n".join(block["text"] for block in blocks)
            payload = {
                "page": page_num,
                "panel": panel_num,
                "engine": provider.name,
                "source": self._rel(panel_file),
                "text_blocks": blocks,
                "combined_text": combined_text,
            }
            self._atomic_write(out_file, payload)
            self._save_debug(img, blocks, debug_file)
        except OCRTimeout as exc:
            return self._error(page_num, panel_num, str(exc))
        except OCRProviderError as exc:
            return self._error(page_num, panel_num, str(exc))
        except OSError as exc:
            return self._error(page_num, panel_num, f"IO error: {exc}")
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            processed = None
            img = None
            gc.collect()

        state.mark_ocr_done(key)
        LOG.info("OCR page %d panel %d: %d block(s)", page_num, panel_num, len(blocks))
        return {
            "status": "detected",
            "page": page_num,
            "panel": panel_num,
            "engine": provider.name,
            "source": self._rel(panel_file),
            "out_file": out_file,
            "debug_image": debug_file,
            "count": len(blocks),
            "blocks": blocks,
            "combined_text": combined_text,
            "error": None,
        }