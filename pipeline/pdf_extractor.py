"""Single-page PDF image extraction, tuned for a 4 GB RAM machine.

Process: open PDF -> load ONE requested page -> render one pixmap -> encode
to JPEG bytes -> close document -> garbage collect -> write file atomically
-> validate -> checkpoint. The full PDF is never held in memory and at most
one rendered page exists at a time. Extraction is skipped when the target
file already exists and passes validation.
"""
import errno
import gc
import logging
import math
import os
from pathlib import Path

import pymupdf as fitz

LOG = logging.getLogger("mangaexplainer")

PAGE_KEY_FORMAT = "page_{:03d}"


def page_key(page_num):
    return PAGE_KEY_FORMAT.format(page_num)


class PdfExtractor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pdf_path = Path(cfg.input.pdf)
        self.pages_dir = Path(cfg.output.pages_dir)
        self.image_format = str(cfg.images.format).lower().lstrip(".")
        self.jpeg_quality = int(cfg.images.jpeg_quality)
        self.max_pixels = int(cfg.images.max_pixels)
        self.render_scale = float(cfg.images.render_scale)
        self._encode_format = "jpeg" if self.image_format == "jpg" else self.image_format

    def _clamped_scale(self, page):
        area = page.rect.width * page.rect.height
        if area <= 0:
            return self.render_scale
        max_scale = math.sqrt(self.max_pixels / area)
        return max(0.1, min(self.render_scale, max_scale))

    def _validate_image(self, path):
        if not path.is_file() or path.stat().st_size <= 0:
            return False, 0, 0
        pix = None
        try:
            pix = fitz.Pixmap(str(path))
            width, height = pix.width, pix.height
        except Exception as exc:
            LOG.warning("output validation failed for %s: %s", path, exc)
            return False, 0, 0
        finally:
            pix = None
        if width <= 0 or height <= 0:
            return False, 0, 0
        return True, width, height

    def _error_result(self, page_num, path, message):
        return {
            "status": "error",
            "page_num": page_num,
            "page_key": page_key(page_num),
            "path": path,
            "width": 0,
            "height": 0,
            "size_bytes": 0,
            "error": message,
        }

    def extract_page(self, page_num, state):
        page_num = int(page_num)
        if page_num < 1:
            LOG.error("invalid page number: %d (must be >= 1)", page_num)
            return self._error_result(
                page_num, None, f"invalid page number {page_num} (must be >= 1)"
            )
        key = page_key(page_num)
        out_path = self.pages_dir / f"{key}.{self.image_format}"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        valid, width, height = self._validate_image(out_path)
        if valid:
            state.mark_page_done(key)
            LOG.info("page %s already extracted (%dx%d); skipped", key, width, height)
            return {
                "status": "skipped",
                "page_num": page_num,
                "page_key": key,
                "path": out_path,
                "width": width,
                "height": height,
                "size_bytes": out_path.stat().st_size,
                "error": None,
            }

        if not self.pdf_path.is_file():
            message = f"input PDF not found: {self.pdf_path}"
            LOG.error(message)
            return self._error_result(page_num, out_path, message)

        doc, page, pix = None, None, None
        try:
            doc = fitz.open(str(self.pdf_path))
            page_count = doc.page_count
            if page_count <= 0:
                return self._error_result(page_num, out_path, "PDF has no pages")
            if page_num > page_count:
                message = (
                    f"page number {page_num} out of range (PDF has {page_count} pages)"
                )
                LOG.error(message)
                return self._error_result(page_num, out_path, message)
            page = doc.load_page(page_num - 1)
            scale = self._clamped_scale(page)
            matrix = fitz.Matrix(scale, scale).prerotate(page.rotation)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            if pix.colorspace not in (fitz.csRGB, fitz.csGRAY):
                pix = fitz.Pixmap(fitz.csRGB, pix)
            width, height = pix.width, pix.height
            data = pix.tobytes(self._encode_format, jpg_quality=self.jpeg_quality)
        except Exception as exc:
            LOG.exception("failed to render page %d from %s", page_num, self.pdf_path)
            return self._error_result(page_num, out_path, f"rendering failed: {exc}")
        finally:
            page = None
            pix = None
            if doc is not None:
                doc.close()
            doc = None
            gc.collect()

        tmp = out_path.with_name(out_path.name + ".tmp")
        try:
            with open(tmp, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, out_path)
        except OSError as exc:
            if exc.errno == errno.ENOSPC:
                message = f"insufficient disk space to write {out_path}"
            else:
                message = f"failed to write {out_path}: {exc}"
            LOG.error(message)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return self._error_result(page_num, out_path, message)
        finally:
            data = None

        valid, width, height = self._validate_image(out_path)
        if not valid:
            LOG.error(
                "post-extraction validation failed for %s; page not marked complete",
                out_path,
            )
            try:
                out_path.unlink(missing_ok=True)
            except OSError:
                pass
            return self._error_result(
                page_num, out_path, "output validation failed after extraction"
            )

        size_bytes = out_path.stat().st_size
        state.mark_page_done(key)
        LOG.info("page %s extracted: %dx%d (%d bytes)", key, width, height, size_bytes)
        return {
            "status": "extracted",
            "page_num": page_num,
            "page_key": key,
            "path": out_path,
            "width": width,
            "height": height,
            "size_bytes": size_bytes,
            "error": None,
        }