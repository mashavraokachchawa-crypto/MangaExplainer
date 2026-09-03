"""Lightweight PDF identity scan for the auto-project flow.

Reads the PDF's metadata (title/author) plus a shallow text probe of the first
few pages (skip the rasterized-cover case) to guess the manga's title WITHOUT
reading or rendering the whole document. Only pages 1..N are probed for
embedded text; if the PDF is fully scanned (images only) the metadata title is
used. Never OCRs, never renders the full book, never loads the whole file:
pymupdf parses lazily and we close the doc after the probe.

Returns a small dict of candidate + sources so the app can enrich via the
internet (fetch_book_ref / fetch_characters) and build the project.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pymupdf as fitz

LOG = logging.getLogger("mangaexplainer")

# Fallback probe pages (1-relative) to look for a printed title line on pages
# that carry embedded text (digital releases sometimes put the series title on
# the cover/title pages as selectable text while the art pages are scanned).
_PROBE_PAGES = 6
_PROBE_CHAR_CAP = 2000


def _strip_volume_markers(text: str) -> str:
    """Cut volume/chapter/edition/package info trailing the real title."""
    t = text.strip()
    t = re.sub(r"\.pdf\s*$", "", t, flags=re.I)
    t = re.sub(r"\bOceanofPDF\.com\b", "", t, flags=re.I)
    # volume / chapter markers: everything from the first marker onward is info
    t = re.sub(
        r"\s*(?:v\s?\d+|volume\s?\d+|vol\.?\s?\d+|ch(?:apter)?\.?\s?[\d\-]+).*$",
        "",
        t,
        flags=re.I,
    )
    # trailing parenthetical package labels: (2003) (Digital) (repack) ...
    t = re.sub(
        r"\s*(?:\((?:\d{4})?\)\s*)*" + r"(?:\((?:[^()]*(?:digital|repack|calibre|scan|cbz|omnibus)[^()]*)\)\s*)*$",
        "",
        t,
        flags=re.I,
    )
    # any residual standalone (year) groups at the very end
    t = re.sub(r"\s*\(\s*\d{4}\s*\)\s*$", "", t)
    while t != (t2 := re.sub(r"[\s\-_,;:.!?]+\s*$", "", t)):
        t = t2
    return t.strip()


def _page_text_probe(doc, page_count: int) -> list[str]:
    """Return deduplicated text snippets from the first few pages."""
    lines: list[str] = []
    for idx in range(min(_PROBE_PAGES, page_count)):
        try:
            text = doc.load_page(idx).get_text(sort=True) or ""
        except Exception:
            continue
        seen = 0
        for ln in text.splitlines():
            ln = ln.strip()
            if len(ln) >= 3 and ln not in lines:
                lines.append(ln)
                seen += 1
            if seen >= 40 or sum(len(x) for x in lines) > _PROBE_CHAR_CAP:
                break
        if sum(len(x) for x in lines) > _PROBE_CHAR_CAP:
            break
    return lines


def _pick_title(metadata_title: str, probe_lines: list[str]) -> tuple[str, str]:
    """Decide the best title + source ('metadata' | 'embedded-text' | '')."""
    candidates: list[tuple[int, str, str]] = []  # (priority, title, source)
    if metadata_title:
        clean = _strip_volume_markers(metadata_title)
        if clean:
            candidates.append((10, clean, "metadata"))

    # Embedded text: look for a short line near the top of the doc that isn't a
    # running footer/watermark. Prefer lines that look like a title (few words,
    # all-caps or Title Case, no copyright/volume noise).
    dirty = {l.lower() for l in probe_lines}
    skip_words = ("oceanofpdf", "copyright", "www.", "http", "©", ".com",
                  "scanlated", "credits", "translation", "edition", "volume",
                  "chapter")
    for ln in probe_lines:
        low = ln.lower()
        if any(w in low for w in skip_words):
            continue
        wc = len(ln.split())
        # a title line is short; a full sentence (>=8 words) is not a title
        if 1 <= wc <= 7 and len(ln) <= 60:
            candidates.append((5, ln.strip(" \t\"'«»‹›"), "embedded-text"))

    if not candidates:
        return "", ""
    candidates.sort(key=lambda c: (c[0], -len(c[1])), reverse=True)
    best = candidates[0]
    return best[1], best[2]


_VOLUME_RE = re.compile(
    r"\b(?:v\.?\s*|vol(?:ume)?\.?\s*)\d+|(?:^|[(\s,#-])#\s*\d+",
    re.IGNORECASE)


def extract_volume(metadata_or_title: str) -> int:
    """Best-effort volume number from a title like ``Berserk v01`` / ``Vol. 3``.

    Returns the integer volume (1-based) or 0 when the string carries no volume
    marker (e.g. an omnibus or a whole-series scan).
    """
    m = _VOLUME_RE.search(str(metadata_or_title or ""))
    if not m:
        return 0
    digits = re.search(r"\d+", m.group(0))
    return int(digits.group(0)) if digits else 0


def scan_pdf(pdf_path: str | Path) -> dict:
    """Best-effort identity scan of a manga PDF.

    Reads metadata + a shallow text probe of the first pages. Returns
    ``{"ok", "title", "author", "source", "page_count", "reason"}``. ``title``
    may be empty when nothing usable is found (fully scanned cover, no
    metadata) — the caller should then prompt the user to type the name.
    """
    path = Path(pdf_path)
    if not path.is_file():
        return {"ok": False, "title": "", "author": "", "source": "",
                "page_count": 0, "reason": f"pdf not found: {path}"}

    doc = None
    try:
        doc = fitz.open(str(path))
        page_count = int(doc.page_count) if doc.page_count else 0
        metadata = dict(doc.metadata or {})
    except Exception as exc:
        LOG.warning("pdf_scan: could not open %s: %s", path, exc)
        return {"ok": False, "title": "", "author": "", "source": "",
                "page_count": 0, "reason": f"could not open pdf: {exc}"}
    finally:
        if doc is not None:
            doc.close()
        doc = None

    # Re-open for the text probe (lazy; doc already validated above).
    author = (metadata.get("author") or "").strip() or "".strip()
    probe: list[str] = []
    try:
        doc = fitz.open(str(path))
        probe = _page_text_probe(doc, page_count)
    except Exception as exc:
        LOG.warning("pdf_scan: text probe failed: %s", exc)
    finally:
        if doc is not None:
            doc.close()

    title, source = _pick_title(metadata.get("title") or "", probe)
    volume = extract_volume(metadata.get("title") or "")
    if not title:
        # if there is a page_count but no text and no metadata title, surface
        # that (fully scanned, cover reads only as image).
        return {
            "ok": bool(page_count),
            "title": "",
            "author": author or "",
            "source": "",
            "page_count": page_count,
            "reason": "no recognizable title in metadata or first pages"
                      if page_count else "pdf has no pages",
        }
    return {
        "ok": True,
        "title": title,
        "author": author or "",
        "source": source,
        "page_count": page_count,
        "volume": volume or None,
        "reason": "",
    }