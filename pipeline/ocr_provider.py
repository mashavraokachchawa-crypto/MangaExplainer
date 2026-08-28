"""Modular OCR providers for the MangaExplainer pipeline.

One provider is selected per run (see create_provider). Providers consume a
preprocessed image file and return text blocks with bounding boxes and
confidence when the underlying engine exposes them:

  [{"text": ..., "bbox": [x, y, width, height], "confidence": ...}]

Selection order: Tesseract (if the binary is installed), lightweight
PaddleOCR only if it is importable, and a deterministic "dummy" engine for
tests. None of these is downloaded or installed here; if no engine is
available, create_provider reports OCR engine unavailable.
"""
import csv
import io
import logging
import shutil
import subprocess

LOG = logging.getLogger("mangaexplainer")


class OCRProviderError(Exception):
    pass


class OCREngineUnavailable(OCRProviderError):
    pass


class OCRTimeout(OCRProviderError):
    pass


class OCRFailure(OCRProviderError):
    pass


def _bbox_of_words(words):
    xs0 = [int(item["x"]) for item in words]
    ys0 = [int(item["y"]) for item in words]
    xs1 = [int(item["x"]) + int(item["width"]) for item in words]
    ys1 = [int(item["y"]) + int(item["height"]) for item in words]
    x = min(xs0)
    y = min(ys0)
    x1 = max(xs1)
    y1 = max(ys1)
    return [x, y, x1 - x, y1 - y]


class TesseractProvider:
    name = "tesseract"

    def __init__(self, cfg):
        self.language = str(cfg.ocr.language)
        self.psm = int(cfg.ocr.psm)
        self.timeout = int(cfg.ocr.timeout_seconds)

    @staticmethod
    def available():
        return shutil.which("tesseract") is not None

    def recognize(self, image_path):
        try:
            proc = subprocess.run(
                [
                    "tesseract", str(image_path), "stdout",
                    "-l", self.language,
                    "--psm", str(self.psm),
                    "tsv",
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except FileNotFoundError:
            raise OCREngineUnavailable("tesseract binary not found") from None
        except subprocess.TimeoutExpired:
            raise OCRTimeout(
                f"tesseract timed out after {self.timeout}s"
            ) from None
        if proc.returncode != 0:
            raise OCRFailure(
                f"tesseract failed (exit {proc.returncode}): "
                f"{proc.stderr.strip() or 'unknown error'}"
            )
        return self._parse_tsv(proc.stdout)

    @staticmethod
    def _parse_tsv(text):
        if not text.strip():
            return []
        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        words = []
        for row in reader:
            if row.get("level") != "5":
                continue
            try:
                conf = float(row.get("conf", "-1"))
                text_val = (row.get("text") or "").strip()
            except (TypeError, ValueError):
                continue
            if conf < 0 or not text_val:
                continue
            words.append(
                {
                    "text": text_val,
                    "x": int(float(row["left"])),
                    "y": int(float(row["top"])),
                    "width": int(float(row["width"])),
                    "height": int(float(row["height"])),
                    "confidence": conf / 100.0,
                    "block": row.get("block_num", "0"),
                    "par": row.get("par_num", "0"),
                    "line": row.get("line_num", "0"),
                }
            )
        lines = {}
        for word in words:
            key = (word["block"], word["par"], word["line"])
            lines.setdefault(key, []).append(word)
        blocks = []
        for key in sorted(lines, key=lambda k: (int(k[0]), int(k[1]), int(k[2]))):
            line = lines[key]
            blocks.append(
                {
                    "text": " ".join(item["text"] for item in line),
                    "bbox": _bbox_of_words(line),
                    "confidence": round(
                        sum(item["confidence"] for item in line) / len(line), 3
                    ),
                }
            )
        return blocks


class PaddleOCRProvider:
    name = "paddleocr"

    def __init__(self, cfg):
        self.timeout = int(cfg.ocr.timeout_seconds)

    @staticmethod
    def available():
        try:
            import paddleocr  # noqa: F401

            return True
        except (ImportError, OSError):
            return False

    def recognize(self, image_path):
        try:
            import numpy as np
            import cv2

            import paddleocr
        except (ImportError, OSError) as exc:
            raise OCREngineUnavailable(str(exc)) from None
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise OCRFailure("cannot read image for paddleocr")
            ocr = paddleocr.PaddleOCR(
                use_angle_cls=False,
                lang="japan",
                show_log=False,
            )
            output = ocr.ocr(np.asarray(img), cls=False)
        except OCRFailure:
            raise
        except Exception as exc:
            raise OCRFailure(f"paddleocr failed: {exc}") from None
        result = ocr_result = output
        if isinstance(result, list):
            page_result = result[0] if result and isinstance(result[0], list) else None
            if page_result is None:
                return []
            PADDLE_TIMEOUT = self.timeout
            blocks = []
            for item in page_result:
                try:
                    polygon, text, score = item
                    xs = [float(pt[0]) for pt in polygon]
                    ys = [float(pt[1]) for pt in polygon]
                    blocks.append(
                        {
                            "text": str(text).strip(),
                            "bbox": [
                                int(min(xs)),
                                int(min(ys)),
                                int(max(xs)) - int(min(xs)),
                                int(max(ys)) - int(min(ys)),
                            ],
                            "confidence": round(float(score), 3),
                        }
                    )
                except (TypeError, ValueError):
                    continue
            return blocks
        return []


class DummyProvider:
    """Deterministic in-process engine for tests / offline smoke runs."""

    name = "dummy"

    def __init__(self, cfg):
        pass

    @staticmethod
    def available():
        return True

    def recognize(self, image_path):
        return [
            {
                "text": "Hello World",
                "bbox": [12, 16, 96, 24],
                "confidence": 0.91,
            }
        ]


PROVIDERS = {
    "tesseract": TesseractProvider,
    "paddleocr": PaddleOCRProvider,
    "dummy": DummyProvider,
}


def create_provider(cfg):
    """Select the best available OCR provider or raise OCREngineUnavailable."""
    requested = str(getattr(cfg.ocr, "engine", "auto") or "auto").lower()
    if requested != "auto":
        if requested not in PROVIDERS:
            raise OCRProviderError(
                f"unknown ocr.engine {requested!r} "
                f"(expected auto or one of {', '.join(PROVIDERS)})"
            )
        cls = PROVIDERS[requested]
        if not cls.available():
            raise OCREngineUnavailable(f"OCR engine '{requested}' is not available")
        return cls(cfg)
    # auto: real engines only - never silently fall back to the dummy
    for name in ("tesseract", "paddleocr"):
        cls = PROVIDERS[name]
        if cls.available():
            LOG.info("using OCR engine: %s", name)
            return cls(cfg)
    raise OCREngineUnavailable(
        "no OCR engine available (install tesseract, or set ocr.engine=dummy)"
    )