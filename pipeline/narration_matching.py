"""Deterministic panel <-> narration mapping (Task 10).

Reads the existing on-disk panel metadata and narration/script data and
builds a *mapping file* that connects every detected manga panel to the
narration segment(s) that describe it, and every narration segment to its
source panel(s).

Inputs (already produced by earlier stages - never regenerated here):
    panels/page_XXX/panels.json          panel ids, images, bboxes
    panels/page_XXX/reading_order.json   authoritative manga reading order
    script/page_XXX_scene_YYY.json       narration segments with panel_ids

Outputs (matching/ directory, one page at a time):
    matching/page_XXX_mapping.json       per-page panel <-> narration mapping
    matching/index.json                  page status tracker (resume/checkpoint)
    matching/mapping.json                consolidated stream-built mapping

Design rules (all required by Task 10):
    * deterministic - the output is a pure function of the input files; no
      LLM/VLM/model is ever invoked
    * manga reading order is preserved (reading_order.json wins, then the
      per-panel reading_order field, then manifest order)
    * every narration segment carries its source panel_ids (normalized to
      reading order); unknown/cross-page ids are kept but flagged
    * every panel carries the narration segment(s) that reference it
    * one-to-one, many-to-one (many panels -> one segment) and one-to-many
      (one panel -> many segments) are all represented with a cardinality
      label and counted in the summary
    * checkpoint/resume: per-page mapping files + page index; completed pages
      are never recomputed unless --force
    * low RAM: exactly ONE page (its panels + its scripts) is ever held in
      memory; nothing else is loaded

No audio is generated, no subtitles are added and no video is rendered
anywhere in this module.
"""
from __future__ import annotations

import gc
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger("mangaexplainer")

MATCH_SCHEMA = 1
MATCH_COMPLETED = "matching_completed"
MATCH_SKIPPED = "matching_skipped"  # page has no panels to match (visible, non-fatal)
NO_PANELS_REASON = "no panels detected on this page, nothing to match"
UNMATCHED = "unmatched"
ONE_TO_ONE = "one_to_one"
MANY_TO_ONE = "many_to_one"
ONE_TO_MANY = "one_to_many"

PAGE_MAPPING_FILENAME = "page_{:03d}_mapping.json"
INDEX_FILENAME = "index.json"
CONSOLIDATED_FILENAME = "mapping.json"


class MatchingError(Exception):
    """Missing/invalid input or a page that cannot be matched (never silent)."""


# ------------------------------------------------------------------ paths


def matching_dir(cfg):
    return Path(cfg.output.matching_dir)


def page_mapping_path(cfg, page):
    return matching_dir(cfg) / PAGE_MAPPING_FILENAME.format(page)


def index_path(cfg):
    return matching_dir(cfg) / INDEX_FILENAME


def consolidated_path(cfg):
    return matching_dir(cfg) / CONSOLIDATED_FILENAME


def _script_pattern(page):
    return f"page_{page:03d}_scene_*.json"


def _script_number(path):
    try:
        return int(path.with_suffix("").name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0


# ------------------------------------------------------------------ inputs


def load_panels(cfg, page):
    """(panels_list, reading_order, direction) for one page.

    Panels come from panels/page_XXX/panels.json. The reading order comes
    from reading_order.json when present (authoritative manga order), else
    falls back to each panel's own reading_order field ascending, else to
    manifest order.
    """
    panels_dir = Path(cfg.output.panels_dir)
    manifest = panels_dir / f"page_{page:03d}" / "panels.json"
    if not manifest.is_file():
        raise MatchingError(
            f"page {page}: missing panel metadata {manifest} "
            "(run: python main.py panels --page %s)" % page
        )
    try:
        doc = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise MatchingError(f"page {page}: invalid panels file {manifest}: {exc}") from None
    if isinstance(doc, dict):
        panels = doc.get("panels")
        direction = str(doc.get("direction") or "rtl")
    else:
        panels = doc
        direction = "rtl"
    if not isinstance(panels, list):
        raise MatchingError(f"page {page}: panels file has no 'panels' list")
    if not panels:
        # a legitimately empty page (cover/insert/blank: no panels detected);
        # nothing can be matched - the caller records a visible skip. Not a
        # broken input, so not a MatchingError.
        return [], [], direction, {}

    cleaned = []
    for record in panels:
        if not isinstance(record, dict) or not record.get("id"):
            continue
        cleaned.append({
            "panel_id": str(record["id"]),
            "image": record.get("image"),
            "bbox": record.get("bbox"),
            "confidence": record.get("confidence"),
            "reading_order": record.get("reading_order"),
        })

    # authoritative ordering: reading_order.json, else per-panel field, else order.
    order_path = panels_dir / f"page_{page:03d}" / "reading_order.json"
    order = []
    if order_path.is_file():
        try:
            ro_doc = json.loads(order_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise MatchingError(f"page {page}: invalid reading order file: {exc}") from None
        order = [str(pid) for pid in (ro_doc.get("order") or []) if pid]
        if ro_doc.get("direction"):
            direction = str(ro_doc["direction"])
    if not order:
        ordered = sorted(
            (p for p in cleaned if p["reading_order"] is not None),
            key=lambda p: (int(p["reading_order"]), p["panel_id"]),
        )
        order = [p["panel_id"] for p in ordered]
    if not order:
        order = [p["panel_id"] for p in cleaned]

    by_id = {p["panel_id"]: p for p in cleaned}
    return cleaned, order, direction, by_id


def load_segments(cfg, page, by_id):
    """Narration segments for one page, in scene/script order.

    Returns (segments, scripts). Every segment keeps its script panel_ids
    (deduplicated, preserved). panel_ids not present on this page are moved
    to 'unknown_panel_ids' so a cross-page reference can never be silently
    dropped.
    """
    script_dir = Path(cfg.output.script_dir)
    scripts = sorted(
        script_dir.glob(_script_pattern(page)),
        key=_script_number,
    )
    segments = []
    for script in scripts:
        try:
            doc = json.loads(script.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise MatchingError(f"page {page}: invalid script {script}: {exc}") from None
        scene_id = doc.get("scene_id") or script.with_suffix("").name
        for segment in doc.get("segments") or []:
            if not isinstance(segment, dict) or not segment.get("segment_id"):
                continue
            known, unknown = [], []
            for pid in segment.get("panel_ids") or []:
                pid = str(pid)
                if pid in by_id:
                    if pid not in known:
                        known.append(pid)
                elif pid not in unknown:
                    unknown.append(pid)
            segments.append({
                "segment_id": str(segment["segment_id"]),
                "scene_id": scene_id,
                "type": segment.get("type", "narration"),
                "text": str(segment.get("text") or ""),
                "panel_ids": known,
                "unknown_panel_ids": unknown,
                "estimated_seconds": segment.get("estimated_seconds"),
                "visual_intent": segment.get("visual_intent"),
                "camera": segment.get("camera"),
                "importance": segment.get("importance"),
            })
    if not scripts:
        LOG.info("page %s: no narration scripts found (panels stay unmatched)",
                 page)
    return segments, [str(p) for p in scripts]


def _sorted_by_order(panel_ids, order_index):
    """panel_ids ordered by manga reading order; unknown ids appended last."""
    known = sorted(
        (pid for pid in panel_ids if pid in order_index),
        key=lambda pid: order_index[pid],
    )
    return known + [pid for pid in panel_ids if pid not in order_index]


# --------------------------------------------------------------- mapping


def build_page_mapping(page, panels, order, direction, segments, scripts):
    """Deterministic panel <-> narration mapping for ONE page (pure function).

    This is the heart of Task 10: for every narration segment it records the
    source panel(s) (preserving manga reading order), and for every panel it
    records the narration segment(s) that describe it. Return value is the
    mapping dict; nothing is written here.
    """
    order_index = {pid: index for index, pid in enumerate(order)}
    panels_by_id = {p["panel_id"]: p for p in panels}

    # segment side: panel_ids normalized to reading order
    ordered_segments = []
    for segment in segments:
        source_panels = _sorted_by_order(list(segment["panel_ids"]), order_index)
        rec = {
            "segment_id": segment["segment_id"],
            "scene_id": segment["scene_id"],
            "type": segment["type"],
            "text": segment["text"],
            "panel_ids": source_panels,
            "panel_count": len(source_panels),
            "cardinality": (
                ONE_TO_ONE if len(source_panels) == 1 else MANY_TO_ONE
            ),
        }
        for field in (
            "estimated_seconds", "visual_intent", "camera", "importance",
        ):
            if segment.get(field) is not None:
                rec[field] = segment[field]
        if segment.get("unknown_panel_ids"):
            rec["unknown_panel_ids"] = segment["unknown_panel_ids"]
            rec["warnings"] = [f"segment references panel(s) not on this page: "
                               f"{', '.join(segment['unknown_panel_ids'])}"]
        ordered_segments.append(rec)

    # panel side: narration segments grouped by source panel, in script order
    segments_by_panel = {}
    for segment in ordered_segments:
        for pid in segment["panel_ids"]:
            segments_by_panel.setdefault(pid, []).append({
                "scene_id": segment["scene_id"],
                "segment_id": segment["segment_id"],
                "type": segment["type"],
                "text": segment["text"],
                "estimated_seconds": segment.get("estimated_seconds"),
            })

    matched = set(segments_by_panel)
    panel_rows = []
    for pid in order:  # preserve manga reading order
        panel = panels_by_id.get(pid) or {}
        segs = segments_by_panel.get(pid, [])
        card = UNMATCHED
        if len(segs) > 1:
            card = ONE_TO_MANY
        elif len(segs) == 1:
            card = ONE_TO_ONE
        row = {
            "panel_id": pid,
            "reading_order": order_index[pid] + 1,
            "cardinality": card,
            "source_count": len(segs),
        }
        if panel.get("image"):
            row["image"] = panel["image"]
        if panel.get("bbox") is not None:
            row["bbox"] = panel["bbox"]
        if panel.get("confidence") is not None:
            row["confidence"] = panel["confidence"]
        row["narration_segments"] = segs
        panel_rows.append(row)
    for pid in sorted(set(panels_by_id) - set(order)):  # leftovers (safety)
        panel = panels_by_id[pid]
        segs = segments_by_panel.get(pid, [])
        row = {
            "panel_id": pid,
            "reading_order": None,
            "cardinality": (
                ONE_TO_MANY if len(segs) > 1 else
                ONE_TO_ONE if segs else UNMATCHED
            ),
            "source_count": len(segs),
        }
        if panel.get("image"):
            row["image"] = panel["image"]
        if panel.get("bbox") is not None:
            row["bbox"] = panel["bbox"]
        if panel.get("confidence") is not None:
            row["confidence"] = panel["confidence"]
        row["narration_segments"] = segs
        panel_rows.append(row)

    unmatched = [row["panel_id"] for row in panel_rows
                 if row["cardinality"] == UNMATCHED]
    warnings = [seg["warnings"][0] for seg in ordered_segments
                if seg.get("warnings")]
    if not segments:
        warnings.append("no narration scripts exist for this page (all panels "
                        "are unmatched)")

    return {
        "schema": MATCH_SCHEMA,
        "page": int(page),
        "reading_direction": direction,
        "reading_order": list(order),
        # panel -> narration (manga reading order)
        "panels": panel_rows,
        # narration -> source panels (script order)
        "segments": ordered_segments,
        "summary": {
            "panels": len(panel_rows),
            "segments": len(ordered_segments),
            "unmatched_panels": len(unmatched),
            ONE_TO_ONE: sum(1 for r in panel_rows if r["cardinality"] == ONE_TO_ONE),
            MANY_TO_ONE: sum(1 for r in ordered_segments
                             if r["cardinality"] == MANY_TO_ONE),
            ONE_TO_MANY: sum(1 for r in panel_rows if r["cardinality"] == ONE_TO_MANY),
            "warnings": len(warnings),
        },
        "unmatched_panels": unmatched,
        "warnings": warnings,
        "inputs": {"scripts": scripts},
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------- page index


def load_index(cfg):
    """Read the matching index (page status tracker); a fresh one if missing."""
    path = index_path(cfg)
    blank = {"schema": MATCH_SCHEMA, "pages": []}
    if not path.is_file():
        return blank
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return blank
    if not isinstance(doc, dict) or not isinstance(doc.get("pages"), list):
        return blank
    return doc


def _write_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _update_index(cfg, page, status, mapping, reason=None):
    doc = load_index(cfg)
    rows = doc["pages"]
    summary = mapping.get("summary") or {}
    row = {
        "page": int(page),
        "file": PAGE_MAPPING_FILENAME.format(page) if status == MATCH_COMPLETED else None,
        "status": status,
        "panel_count": summary.get("panels", 0),
        "segment_count": summary.get("segments", 0),
        "unmatched_panels": summary.get("unmatched_panels", 0),
        "warnings": summary.get("warnings", 0),
    }
    if reason:
        row["reason"] = reason
    rows[:] = [r for r in rows if r.get("page") != int(page)]
    rows.append(row)
    rows.sort(key=lambda r: r.get("page", 0))
    doc["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_atomic(index_path(cfg), doc)


def _page_already_done(cfg, page):
    for row in load_index(cfg)["pages"]:
        if row.get("page") != int(page):
            continue
        if row.get("status") == MATCH_COMPLETED:
            return page_mapping_path(cfg, page).is_file()
        if row.get("status") == MATCH_SKIPPED:
            return True
    return False


# ------------------------------------------------------------- processor


class NarrationMatcher:
    """Builds the deterministic panel <-> narration mapping.

    Lightweight by design: reads only the small JSON manifests the earlier
    stages produced; never loads an image, a model or a PDF, so it runs on
    the 4 GB box without special handling.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def run_page(self, page, state=None, force=False):
        """Build the mapping for ONE page; returns a result dict.

        {"result": "ok"|"skipped"|"error", "page", "mapping_file",
         "segment_count", "panel_count", ...}
        """
        try:
            if not force and _page_already_done(self.cfg, page):
                return self._skip(page)

            panels, order, direction, by_id = load_panels(self.cfg, page)
            if not panels:
                # legitimately empty page (cover/insert/blank): record a
                # visible, resumable skip - not a failure.
                _update_index(self.cfg, page, MATCH_SKIPPED, {},
                              reason=NO_PANELS_REASON)
                LOG.info("page %s: %s", page, NO_PANELS_REASON)
                return {
                    "result": "skipped",
                    "page": int(page),
                    "reason": NO_PANELS_REASON,
                    "panel_count": 0,
                    "segment_count": 0,
                }
            segments, scripts = load_segments(self.cfg, page, by_id)
            mapping = build_page_mapping(
                page, panels, order, direction, segments, scripts
            )
            out_path = page_mapping_path(self.cfg, page)
            _write_atomic(out_path, mapping)
            _update_index(self.cfg, page, MATCH_COMPLETED, mapping)
            if state is not None:
                try:
                    state.mark_item_done(f"match_{page:03d}", MATCH_COMPLETED)
                except (AttributeError, TypeError):
                    pass

            summary = mapping["summary"]
            LOG.info(
                "page %s matched: %d segment(s) <-> %d panel(s), "
                "%d unmatched (%s)",
                page, summary["segments"], summary["panels"],
                summary["unmatched_panels"], out_path,
            )
            return {
                "result": "ok",
                "page": int(page),
                "mapping_file": str(out_path),
                "panel_count": summary["panels"],
                "segment_count": summary["segments"],
                "unmatched_panels": summary["unmatched_panels"],
                "warnings": mapping["warnings"],
            }
        except MatchingError as exc:
            LOG.error("page %s matching failed: %s", page, exc)
            return {"result": "error", "page": int(page), "message": str(exc)}
        finally:
            gc.collect()

    def run_all(self, state=None, force=False, on_progress=None):
        """Tag every page one at a time, resuming from completed pages.

        Pages are discovered from the panels manifests (panels/page_*). A
        page already recorded as completed in the index is skipped unless
        force=True. Returns a result dict with per-page rows.
        """
        dir_path = Path(self.cfg.output.panels_dir)
        pages = []
        if dir_path.is_dir():
            for d in sorted(dir_path.iterdir()):
                num = d.name.rsplit("_", 1)[-1] if "_" in d.name else d.name
                if d.is_dir() and num.isdigit() and (d / "panels.json").is_file():
                    pages.append(int(num))
        if not pages:
            return {
                "result": "error",
                "pages_done": 0,
                "pages_skipped": 0,
                "pages_failed": 0,
                "pages": [],
                "message": "no panels/page_NNN/panels.json manifests found - "
                           "run the panels/reading-order stages first",
            }

        rows, done, skipped, failed = [], 0, 0, 0
        for index, page in enumerate(pages, 1):
            if on_progress is not None:
                on_progress(index, len(pages), page)
            result = self.run_page(page, state=state, force=force)
            row = {"page": page, "result": result.get("result")}
            if result.get("result") == "error":
                row["message"] = result.get("message")
                failed += 1
            elif result.get("result") == "skipped":
                skipped += 1
                if result.get("reason"):
                    row["reason"] = result.get("reason")
                else:
                    row["mapping_file"] = result.get("mapping_file")
            else:
                done += 1
                row["mapping_file"] = result.get("mapping_file")
                row["segment_count"] = result.get("segment_count")
                row["unmatched_panels"] = result.get("unmatched_panels")
            rows.append(row)

        return {
            "result": "ok" if not failed else "error",
            "pages_done": done,
            "pages_skipped": skipped,
            "pages_failed": failed,
            "pages": rows,
        }

    def _skip(self, page):
        mapping_file = page_mapping_path(self.cfg, page)
        LOG.info("page %s mapping already complete (skip) -> %s", page, mapping_file)
        return {
            "result": "skipped",
            "page": int(page),
            "mapping_file": str(mapping_file),
        }


# ------------------------------------------------------ consolidated file


def consolidate_mapping(cfg):
    """Merge every per-page mapping into one matching/mapping.json.

    Streamed: each page file is read, appended and released immediately, so
    a whole book never sits in memory. Pages are sorted in page order; the
    panel/segment counts roll up the per-page summary.
    """
    out_path = consolidated_path(cfg)
    page_files = sorted(
        matching_dir(cfg).glob("page_*_mapping.json"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not page_files:
        raise MatchingError(
            "no per-page mapping files found in matching/ - run the match "
            "stage first (python main.py match --all)"
        )
    merged = {
        "schema": MATCH_SCHEMA,
        "pages": [],
        "total_panels": 0,
        "total_segments": 0,
        "total_unmatched_panels": 0,
    }
    for path in page_files:
        try:
            page_doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise MatchingError(f"cannot read mapping {path}: {exc}") from None
        summary = page_doc.get("summary") or {}
        merged["pages"].append({
            "page": page_doc.get("page"),
            "file": path.name,
            "reading_direction": page_doc.get("reading_direction"),
            "summary": summary,
        })
        merged["total_panels"] += summary.get("panels", 0)
        merged["total_segments"] += summary.get("segments", 0)
        merged["total_unmatched_panels"] += summary.get("unmatched_panels", 0)
    merged["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_atomic(out_path, merged)
    return merged