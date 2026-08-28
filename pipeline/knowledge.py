"""Disk-first manga knowledge layer.

Combines the artifacts of earlier stages - page metadata, detected panels,
bounding boxes, reading order, OCR results and VLM analysis - into a
normalized, validated, lazily-loadable knowledge store. Uses plain JSON files
on disk; images are referenced by path and never loaded here.

Layout (all inside cfg.output.analysis_dir):

    index.json                    <- lightweight global index (page refs)
    page_001_knowledge.json       <- one knowledge file per processed page
    page_001_panel_001.json       <- VLM analysis (produced by the analyze stage)

This module never reads or holds more than one page's worth of JSON at once,
and never loads manga images unless validation requires it.

Completion rule (checkpoint): a page is only marked knowledge_completed when
every detected panel has both an OCR result and a VLM analysis. Missing
optional data is represented as null / empty and reported - never invented,
never fatal: the knowledge file is still written with what exists so later
stages can use whatever is available. Missing REQUIRED data (page source,
panel manifest, reading order) is an error and produces nothing.
"""
import gc
import json
import logging
import os
import re
from pathlib import Path

LOG = logging.getLogger("mangaexplainer")

KNOWLEDGE_FILENAME = "page_{:03d}_knowledge.json"
INDEX_FILENAME = "index.json"
PANEL_ID_RE = re.compile(r"^p\d{3}_\d{3}$")


class KnowledgeError(Exception):
    """Raised for invalid or missing knowledge data (never silently fixed)."""


# ------------------------------------------------------------- validation


def validate_page_number(page):
    if not isinstance(page, int) or isinstance(page, bool) or page < 1:
        raise KnowledgeError(f"invalid page number {page!r}: must be an int >= 1")
    return page


def validate_panel_id(panel_id):
    if not isinstance(panel_id, str) or not PANEL_ID_RE.match(panel_id):
        raise KnowledgeError(
            f"invalid panel id {panel_id!r}: expected form p<page>_<panel>"
        )
    return panel_id


def validate_bbox(bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise KnowledgeError(f"invalid bbox {bbox!r}: must be [x, y, w, h]")
    values = []
    for value in bbox:
        try:
            values.append(int(float(value)))
        except (TypeError, ValueError):
            raise KnowledgeError(
                f"invalid bbox {bbox!r}: all coordinates must be numeric"
            ) from None
    x, y, w, h = values
    if w <= 0 or h <= 0 or x < 0 or y < 0:
        raise KnowledgeError(
            f"invalid bbox {bbox!r}: width/height must be > 0, x/y >= 0"
        )
    return values


def validate_confidence(confidence):
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        raise KnowledgeError(
            f"invalid confidence {confidence!r}: must be a number"
        ) from None
    if not (0.0 <= value <= 1.0):
        raise KnowledgeError(f"invalid confidence {confidence!r}: must be within [0, 1]")
    return value


def validate_image_path(path):
    if not isinstance(path, str) or not path.strip():
        raise KnowledgeError(f"invalid image path {path!r}: must be a non-empty string")
    as_path = Path(path)
    if not as_path.is_absolute():
        raise KnowledgeError(f"invalid image path {path!r}: must be absolute")
    if not as_path.is_file():
        raise KnowledgeError(f"invalid image path {path!r}: file does not exist")
    return str(as_path)


def validate_reading_order(rank):
    if rank is not None and (not isinstance(rank, int) or rank < 1):
        raise KnowledgeError(
            f"invalid reading_order {rank!r}: must be an int >= 1 or null"
        )
    return rank


def _read_json(path):
    """Strict JSON read with a clear error; never silently mangles data."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise KnowledgeError(f"cannot read valid JSON from {path}: {exc}") from None


def validate_record(record):
    required = (
        "panel_id", "page", "reading_order", "image",
        "bbox", "ocr", "visual",
        "previous_panel", "next_panel", "scene_id",
    )
    if not isinstance(record, dict):
        raise KnowledgeError("panel record must be an object")
    for field in required:
        if field not in record:
            raise KnowledgeError(f"panel record missing required field {field!r}")
    validate_panel_id(record["panel_id"])
    validate_page_number(record["page"])
    validate_reading_order(record["reading_order"])
    validate_image_path(record["image"])
    record["bbox"] = validate_bbox(record["bbox"])
    return record


def validate_knowledge(doc):
    """Full-document validation for a page knowledge file."""
    if not isinstance(doc, dict):
        raise KnowledgeError("page knowledge must be an object")
    page = validate_page_number(doc.get("page"))
    if doc.get("reading_direction") not in ("rtl", "ltr"):
        raise KnowledgeError(
            f"invalid reading_direction {doc.get('reading_direction')!r}"
        )
    panels = doc.get("panels")
    if not isinstance(panels, list):
        raise KnowledgeError("page knowledge missing required list field 'panels'")
    for record in panels:
        validate_record(record)
    if doc.get("panel_count") != len(panels):
        raise KnowledgeError(
            f"panel_count {doc.get('panel_count')} != panels list length {len(panels)}"
        )
    return page


# ------------------------------------------------------------ lazy loading


def knowledge_path(cfg, page):
    return Path(cfg.output.analysis_dir) / KNOWLEDGE_FILENAME.format(page)


def index_path(cfg):
    return Path(cfg.output.analysis_dir) / INDEX_FILENAME


def load_page_knowledge(cfg, page):
    """Lazy: loads ONLY this one page's knowledge file."""
    validate_page_number(page)
    path = knowledge_path(cfg, page)
    if not path.is_file():
        raise KnowledgeError(f"no knowledge file for page {page}: {path}")
    doc = _read_json(path)
    validate_knowledge(doc)
    return doc


def load_index(cfg):
    """Lazy, cheap global reference index (never the full per-page payloads)."""
    path = index_path(cfg)
    if not path.is_file():
        return {"pages": []}
    return _read_json(path)


def index_status(cfg, page):
    try:
        validate_page_number(page)
    except KnowledgeError:
        return None
    for entry in load_index(cfg).get("pages", []):
        if entry.get("page") == page:
            return entry
    return None


# ------------------------------------------------------------------- builder


class KnowledgeBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.analysis_dir = Path(cfg.output.analysis_dir)

    # -------------------------------------------------------- entry point
    def build_page(self, page, state, force=False):
        try:
            validate_page_number(page)
            key = f"page_{page:03d}"
            knowledge_file = knowledge_path(self.cfg, page)

            old_doc = None
            if not force and knowledge_file.is_file():
                try:
                    old_doc = load_page_knowledge(self.cfg, page)
                except KnowledgeError as exc:
                    raise KnowledgeError(
                        f"existing knowledge file is invalid: {exc} "
                        "(fix or rebuild with --force)"
                    ) from None

            sources = self._read_sources(page)
            records = self._build_records(page, sources)
            missing = self._missing(records)
            panel_ids = [record["panel_id"] for record in records]

            changed = []
            if old_doc is not None:
                old_by_id = {r["panel_id"]: r for r in old_doc.get("panels", [])}
                for record in records:
                    if old_by_id.get(record["panel_id"]) != record:
                        changed.append(record["panel_id"])
            else:
                changed = list(panel_ids)

            status = "complete" if not missing else "partial"

            # Checkpointed page whose sources have NOT changed -> cheap skip.
            if (
                not force
                and state is not None
                and self._done(state, key)
                and not changed
            ):
                return self._result(
                    "skipped", page, knowledge_file, records,
                    status="complete", missing=missing, changed=[],
                )

            if not changed and old_doc is not None:
                result = self._result(
                    "unchanged", page, knowledge_file, records,
                    status=status, missing=missing, changed=[],
                )
            else:
                knowledge_file.parent.mkdir(parents=True, exist_ok=True)
                doc = self._write_knowledge(knowledge_file, page, sources, records)
                validate_knowledge(doc)
                self._update_index(page, knowledge_file, status)
                result = self._result(
                    "ok", page, knowledge_file, records,
                    status=status, missing=missing, changed=changed,
                )

            if state is not None and status == "complete":
                state.mark_item_done(key, "knowledge_completed")

            LOG.info(
                "knowledge page %s: %s (%s, %s panel(s))",
                page, result["result"], status, len(panel_ids),
            )
            return result
        except KnowledgeError as exc:
            LOG.error("knowledge page %s build failed: %s", page, exc)
            return self._result(
                "error", page, knowledge_path(self.cfg, page), None,
                status="error", missing=[str(exc)], message=str(exc),
            )
        finally:
            gc.collect()

    # -------------------------------------------------------------- sources
    def _read_sources(self, page):
        page_file = Path(self.cfg.output.pages_dir) / f"page_{page:03d}.jpg"
        manifest_file = (
            Path(self.cfg.output.panels_dir) / f"page_{page:03d}" / "panels.json"
        )
        order_file = (
            Path(self.cfg.output.panels_dir) / f"page_{page:03d}" / "reading_order.json"
        )
        missing = []
        for label, path in (
            ("page source", page_file),
            ("panel manifest", manifest_file),
            ("reading order", order_file),
        ):
            if not path.is_file():
                missing.append(label)
        if missing:
            raise KnowledgeError(
                f"cannot build knowledge for page {page} - missing: "
                + ", ".join(missing)
            )
        manifest = _read_json(manifest_file)
        order_data = _read_json(order_file)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("panels"), list):
            raise KnowledgeError(f"invalid panel manifest {manifest_file}: no panels list")
        if not isinstance(order_data, dict) or not isinstance(order_data.get("order"), list):
            raise KnowledgeError(f"invalid reading order {order_file}: no order list")
        return {
            "page_file": page_file,
            "manifest": manifest,
            "order": order_data,
        }

    def _build_records(self, page, sources):
        manifest = sources["manifest"]
        order = sources["order"].get("order", [])
        direction = sources["order"].get("direction") or (
            self.cfg.reading.direction or "rtl"
        )
        ordered = {pid: rank for rank, pid in enumerate(order, 1)}
        records = []
        for index, panel in enumerate(manifest.get("panels", []), 1):
            if not isinstance(panel, dict) or not isinstance(panel.get("id"), str):
                raise KnowledgeError(f"panel manifest entry {index} is not a valid object")
            panel_id = validate_panel_id(panel["id"])
            bbox = validate_bbox(panel.get("bbox"))
            image = panel.get("image") or str(
                Path(self.cfg.output.panels_dir)
                / f"page_{page:03d}" / f"panel_{index:03d}.jpg"
            )
            image = validate_image_path(image)
            rank = panel.get("reading_order")
            if rank is None:
                rank = ordered.get(panel_id)
            rank = validate_reading_order(rank)

            ocr_record = self._load_ocr(page, index)
            visual_record, visual_missing = self._load_visual(page, index)

            records.append(
                {
                    "panel_id": panel_id,
                    "page": page,
                    "reading_order": rank,
                    "image": image,
                    "bbox": bbox,
                    "ocr": ocr_record,
                    "visual": visual_record,
                    "previous_panel": None,
                    "next_panel": None,
                    "scene_id": None,
                    "_missing": visual_missing,
                }
            )
        records.sort(key=lambda r: (r["reading_order"] is None, r["reading_order"] or 0))
        for record in records:
            record.pop("_missing", None)
        return records

    def _load_ocr(self, page, panel):
        key = f"page_{page:03d}_panel_{panel:03d}"
        path = Path(self.cfg.output.ocr_dir) / f"{key}.json"
        if not path.is_file():
            return None
        data = _read_json(path)
        blocks = data.get("text_blocks", [])
        if not isinstance(blocks, list):
            blocks = []
        return {
            "text": (data.get("combined_text") or ""),
            "blocks": blocks,
        }

    def _load_visual(self, page, panel):
        key = f"page_{page:03d}_panel_{panel:03d}"
        path = self.analysis_dir / f"{key}.json"
        if not path.is_file():
            return None, f"VLM analysis for {key}"
        data = _read_json(path)
        analysis = data.get("analysis")
        if not isinstance(analysis, dict):
            return None, f"invalid VLM analysis in {path}"
        confidence = analysis.get("confidence")
        try:
            validate_confidence(confidence)
        except KnowledgeError as exc:
            raise KnowledgeError(f"{path}: {exc}") from None
        return {
            "characters": analysis.get("characters", []),
            "environment": analysis.get("environment", "unknown"),
            "actions": analysis.get("actions", []),
            "objects": analysis.get("objects", []),
            "visual_effects": analysis.get("visual_effects", []),
            "important_event": analysis.get("important_event", "unknown"),
            "composition": analysis.get("composition", "unknown"),
            "story_relevance": analysis.get("story_relevance", "unknown"),
            "confidence": confidence or 0.0,
        }, None

    @staticmethod
    def _missing(records):
        missing = []
        for record in records:
            pid = record["panel_id"]
            if record["ocr"] is None:
                missing.append(f"OCR for {pid}")
            if record["visual"] is None:
                missing.append(f"VLM analysis for {pid}")
        return missing

    # ---------------------------------------------------------------- writes
    def _write_knowledge(self, knowledge_file, page, sources, records):
        doc = {
            "page": page,
            "source": str(sources["page_file"]),
            "panel_count": len(records),
            "reading_direction": sources["order"].get("direction")
            or self.cfg.reading.direction
            or "rtl",
            "panels": records,
        }
        tmp = knowledge_file.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(doc, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, knowledge_file)
        return doc

    def _update_index(self, page, knowledge_file, status):
        index_file = index_path(self.cfg)
        index = load_index(self.cfg)
        pages = index.setdefault("pages", [])
        entry = {"page": page, "knowledge": str(knowledge_file), "status": status}
        for i, existing in enumerate(pages):
            if existing.get("page") == page:
                pages[i] = entry
                break
        else:
            pages.append(entry)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = index_file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(index, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, index_file)
        return entry

    # ---------------------------------------------------------------- state
    @staticmethod
    def _done(state, key):
        try:
            return state.item_done(key, "knowledge_completed")
        except (AttributeError, TypeError):
            return False

    @staticmethod
    def _status_of(state, key):
        return "complete" if KnowledgeBuilder._done(state, key) else "partial"

    @staticmethod
    def _result(kind, page, knowledge_file, records, status=None, missing=None, changed=None, message=None):
        return {
            "result": kind,
            "page": page,
            "knowledge_file": str(knowledge_file) if knowledge_file is not None else None,
            "panel_count": len(records) if records else 0,
            "status": status,
            "missing": missing or [],
            "changed": changed or [],
            "message": message,
        }