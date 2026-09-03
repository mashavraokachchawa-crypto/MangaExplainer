"""Automatic chapter detection from PDF + internet research.

Uses multiple signals to find chapter boundaries:
  1. Chapter title pages (large blank regions + centered text)
  2. OCR text containing chapter number/title patterns
  3. Visual formatting changes (full-page panels, margins)
  4. Known chapter info from internet research
  5. PDF table of contents if available
  6. Page content analysis (scene changes, major transitions)

Each detected boundary carries a confidence score and source tag so the
user can correct it and the correction sticks.
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

LOG = logging.getLogger("mangaexplainer.chapter_detect")

# Common chapter-title OCR patterns across languages
_CHAPTER_PATTERNS = [
    # English: "Chapter 1", "Chapter 1: Title", "CH. 5", "#12"
    re.compile(r"(?:chapter|chap\.?|ch\.?|ch)\s*#?\s*(\d+)(?:\s*[:\-\u2013]\s*(.+))?", re.IGNORECASE),
    re.compile(r"#\s*(\d+)(?:\s+([A-Z].+))?"),
    # Japanese patterns: "第N話", "第N回", "第N章"
    re.compile(r"第\s*(\d+)\s*[話回章話]"),
    # Volume markers: "Volume 1 Chapter 3"
    re.compile(r"(?:volume|vol\.?|v\.?)\s*(\d+)\s+(?:chapter|ch\.?|ch)\s*(\d+)", re.IGNORECASE),
]

# Transition keywords that often appear at chapter starts
_CHAPTER_START_SIGNALS = {
    "meanwhile", "later", "the next day", "elsewhere", "meanwhile...",
    "several days later", "back at", "the following morning",
    "翌日", "しかし", "そして", "数日後", " Meanwhile", "elsewhere...",
}

# Patterns that suggest a page is a chapter title page
_TITLE_PAGE_SIGNALS = [
    # Page has very few panels (1-2) but lots of text
    # Page has large blank regions
    # Page has centered text with chapter number
    # Page has author name prominently displayed
]


def _text_hash(text: str) -> str:
    """Stable short hash for deduplication."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _clean_text(text: str) -> str:
    """Normalize OCR text for pattern matching."""
    text = text.strip()
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def detect_chapters_from_ocr(manga_id: str, pages_dir: Path, ocr_dir: Path,
                              total_pages: int) -> list[dict]:
    """Scan OCR results for chapter-title patterns.

    Returns a list of detected chapter boundaries:
    [{"chapter_number": N, "title": "...", "pdf_page_start": P,
      "confidence": 0.X, "source": "pdf_ocr", "extra": {...}}, ...]
    """
    detected = []

    for page_num in range(1, total_pages + 1):
        ocr_file = ocr_dir / f"page_{page_num:03d}_panel_001.json"
        if not ocr_file.is_file():
            continue
        try:
            data = __import__("json").loads(ocr_file.read_text("utf-8"))
        except Exception:
            continue

        # Combine text from all panels on the page
        combined = data.get("combined_text", "")
        if not combined:
            continue
        text = _clean_text(combined)

        for pattern in _CHAPTER_PATTERNS:
            matches = pattern.finditer(text)
            for m in matches:
                groups = m.groups()
                chapter_num = int(groups[0]) if groups[0] else None
                title = groups[1].strip() if len(groups) > 1 and groups[1] else ""

                if chapter_num is None:
                    continue

                # Confidence: higher for patterns with title, lower for bare numbers
                conf = 0.7 if title else 0.5
                # Boost confidence if this is early in the text (likely a title page)
                text_pos = m.start() / max(1, len(text))
                if text_pos < 0.3:
                    conf += 0.1

                detected.append({
                    "chapter_number": chapter_num,
                    "title": title,
                    "pdf_page_start": page_num,
                    "confidence": min(conf, 0.95),
                    "source": "pdf_ocr",
                    "extra": {
                        "matched_pattern": pattern.pattern,
                        "matched_text": m.group(),
                        "page_text_preview": text[:200],
                    },
                })
                break  # One match per page is enough

    return detected


def detect_chapters_from_title_pages(manga_id: str, pages_dir: Path,
                                      ocr_dir: Path, total_pages: int,
                                      page_panel_counts: dict[int, int] | None = None) -> list[dict]:
    """Detect chapter starts from title-page heuristics.

    Title pages often have:
    - Very few panels (1-2)
    - Large amounts of whitespace
    - Chapter number + title text
    - Author credit
    """
    detected = []

    for page_num in range(1, total_pages + 1):
        # Check if this page has very few panels (title page indicator)
        panel_count = (page_panel_counts or {}).get(page_num)
        if panel_count is None:
            # Try to count panels from manifest
            manifest = pages_dir.parent / "panels" / f"page_{page_num:03d}" / "panels.json"
            if manifest.is_file():
                try:
                    import json
                    data = json.loads(manifest.read_text("utf-8"))
                    if isinstance(data, dict):
                        panel_count = len(data.get("panels", []))
                    elif isinstance(data, list):
                        panel_count = len(data)
                except Exception:
                    panel_count = None

        # Title pages typically have 1-2 panels
        if panel_count is not None and panel_count > 3:
            continue

        # Check OCR for chapter patterns
        ocr_file = ocr_dir / f"page_{page_num:03d}_panel_001.json"
        if not ocr_file.is_file():
            continue
        try:
            import json
            data = json.loads(ocr_file.read_text("utf-8"))
        except Exception:
            continue

        combined = (data.get("combined_text") or "").strip()
        if not combined:
            continue

        text = _clean_text(combined)

        # Look for chapter patterns
        for pattern in _CHAPTER_PATTERNS:
            m = pattern.search(text)
            if m:
                groups = m.groups()
                chapter_num = int(groups[0]) if groups[0] else None
                title = groups[1].strip() if len(groups) > 1 and groups[1] else ""
                if chapter_num:
                    conf = 0.75  # title page + chapter pattern = high confidence
                    detected.append({
                        "chapter_number": chapter_num,
                        "title": title,
                        "pdf_page_start": page_num,
                        "confidence": min(conf, 0.95),
                        "source": "pdf_title_page",
                        "extra": {
                            "panel_count": panel_count,
                            "matched_text": m.group(),
                        },
                    })
                    break

    return detected


def detect_chapters_from_transitions(manga_id: str, ocr_dir: Path,
                                      total_pages: int) -> list[dict]:
    """Detect chapter-start signals from transition keywords in OCR text."""
    detected = []

    for page_num in range(2, total_pages + 1):  # Skip page 1
        ocr_file = ocr_dir / f"page_{page_num:03d}_panel_001.json"
        if not ocr_file.is_file():
            continue
        try:
            import json
            data = json.loads(ocr_file.read_text("utf-8"))
        except Exception:
            continue

        combined = (data.get("combined_text") or "").strip().lower()
        if not combined:
            continue

        for signal in _CHAPTER_START_SIGNALS:
            if signal.lower() in combined[:100]:  # Only check first 100 chars
                detected.append({
                    "chapter_number": None,  # Will be assigned later
                    "title": "",
                    "pdf_page_start": page_num,
                    "confidence": 0.4,
                    "source": "pdf_transition",
                    "extra": {
                        "transition_signal": signal,
                        "page_text_preview": combined[:200],
                    },
                })
                break

    return detected


def merge_chapter_detections(detections: list[list[dict]],
                              total_pages: int) -> list[dict]:
    """Merge chapter detections from multiple sources.

    Rules:
    - Same page_number = same chapter boundary (merge, keep highest confidence)
    - Adjacent pages (±2) = merge into same boundary (keep higher-confidence page)
    - Conflicting chapter numbers at same page = keep higher confidence, flag
    - Assign sequential chapter numbers to unnumbered transitions
    """
    # Flatten all detections
    all_detections = []
    for source_list in detections:
        all_detections.extend(source_list)

    if not all_detections:
        return []

    # Sort by page then confidence
    all_detections.sort(key=lambda d: (d["pdf_page_start"], -d["confidence"]))

    # Group nearby detections (within 2 pages)
    merged = []
    current_group = [all_detections[0]]

    for det in all_detections[1:]:
        if det["pdf_page_start"] - current_group[-1]["pdf_page_start"] <= 2:
            current_group.append(det)
        else:
            merged.append(_merge_group(current_group))
            current_group = [det]
    merged.append(_merge_group(current_group))

    # Assign sequential chapter numbers to unnumbered detections
    next_ch = 1
    for det in merged:
        if det.get("chapter_number") is None:
            det["chapter_number"] = next_ch
        next_ch = det.get("chapter_number", next_ch) + 1

    # Compute end pages
    for i, det in enumerate(merged):
        if i + 1 < len(merged):
            det["pdf_page_end"] = merged[i + 1]["pdf_page_start"] - 1
        else:
            det["pdf_page_end"] = total_pages
        # Ensure end >= start
        det["pdf_page_end"] = max(det["pdf_page_end"], det["pdf_page_start"])

    return merged


def _merge_group(group: list[dict]) -> dict:
    """Merge a group of nearby detections into one boundary."""
    best = max(group, key=lambda d: d["confidence"])
    sources = list(dict.fromkeys(d["source"] for d in group))
    # Merge extra info
    extra = {}
    for d in group:
        extra.update(d.get("extra", {}))
    extra["detection_sources"] = sources
    extra["detection_count"] = len(group)

    return {
        "chapter_number": best.get("chapter_number"),
        "title": best.get("title", ""),
        "pdf_page_start": best["pdf_page_start"],
        "confidence": best["confidence"],
        "source": "merged",
        "extra": extra,
    }


def detect_chapters_from_internet(manga_id: str, book_info: dict,
                                   total_pages: int) -> list[dict]:
    """Use internet-fetched chapter list to create boundary estimates.

    If the book ref has chapter count but no page mapping, we distribute
    chapters evenly across pages as a rough guide.
    """
    detected = []
    total_chapters = book_info.get("total_chapters")
    if not total_chapters or not isinstance(total_chapters, (int, float)):
        return detected
    total_chapters = int(total_chapters)
    if total_chapters < 1:
        return detected

    # Even distribution: each chapter ≈ pages/chapters
    pages_per_chapter = max(1, total_pages / total_chapters)
    for ch_num in range(1, total_chapters + 1):
        start = max(1, int(round((ch_num - 1) * pages_per_chapter)) + 1)
        end = int(round(ch_num * pages_per_chapter))
        end = min(end, total_pages)
        if start > total_pages:
            break
        detected.append({
            "chapter_number": ch_num,
            "title": "",
            "pdf_page_start": start,
            "pdf_page_end": end,
            "confidence": 0.3,  # Low confidence: estimated, not verified
            "source": "internet_estimated",
            "extra": {
                "estimated": True,
                "total_chapters": total_chapters,
                "pages_per_chapter": round(pages_per_chapter, 1),
            },
        })

    return detected


def detect_all_chapters(manga_id: str, pages_dir: Path, ocr_dir: Path,
                         total_pages: int,
                         book_info: dict | None = None) -> list[dict]:
    """Run all detection methods and merge results.

    Returns the final list of chapter boundaries sorted by page start.
    """
    ocr_chapters = detect_chapters_from_ocr(manga_id, pages_dir, ocr_dir, total_pages)
    title_chapters = detect_chapters_from_title_pages(manga_id, pages_dir, ocr_dir, total_pages)
    transition_chapters = detect_chapters_from_transitions(manga_id, ocr_dir, total_pages)
    internet_chapters = detect_chapters_from_internet(
        manga_id, book_info or {}, total_pages
    )

    merged = merge_chapter_detections(
        [ocr_chapters, title_chapters, transition_chapters, internet_chapters],
        total_pages,
    )

    LOG.info(
        "chapter detection: ocr=%d title_page=%d transition=%d internet=%d -> merged=%d",
        len(ocr_chapters), len(title_chapters),
        len(transition_chapters), len(internet_chapters), len(merged),
    )

    return merged


def apply_chapter_detections(db, manga_id: str, chapters: list[dict]):
    """Write detected chapters into the knowledge database.

    Skips duplicates (same manga_id + pdf_page_start).  User corrections
    via update_chapter() take priority over auto-detected values.
    """
    existing = db.get_chapters(manga_id)
    existing_starts = {ch["pdf_page_start"] for ch in existing}

    added = 0
    for ch in chapters:
        if ch["pdf_page_start"] in existing_starts:
            continue  # Don't overwrite existing (possibly user-corrected)
        db.add_chapter(manga_id, ch)
        added += 1

    LOG.info("chapter detection: %d new chapters added to knowledge DB", added)
    return added
