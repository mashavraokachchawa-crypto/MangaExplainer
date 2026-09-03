"""Knowledge orchestrator (unique module to avoid clashing with knowledge.py).

Connects the existing PDF analysis pipeline into the persistent knowledge
database.  This is the entry point used by webui.py and main.py.

Workflow:
   1. identify_manga()   — from PDF scan data, create/ensure a manga record
   2. run_research()     — internet research with caching (book ref, characters)
   3. extract_knowledge()— extract per-page knowledge after understanding
   4. detect_chapters()  — chapter boundaries from multiple signals
   5. summarize()        — chapter summaries
   6. retrieval for prompts — bound relevant memory

NOTE: ``knowledge.py`` in this package is the *per-page knowledge builder*
(an existing part of the pipeline).  This module is the higher-level knowledge
*database* engine — deliberately different concerns, deliberately distinct name.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .knowledge_db import open_knowledge_db
from .research_cache import ResearchCache

LOG = logging.getLogger("mangaexplainer.knowledge_engine")


def get_state_dir(cfg) -> Path:
    try:
        return Path(cfg.pipeline.state.dir)
    except (AttributeError, TypeError):
        return Path("state")


class MangaKnowledge:
    """Facade over the knowledge DB + research cache for one manga project."""

    def __init__(self, cfg, manga_id: str | None = None):
        self.cfg = cfg
        self.state_dir = get_state_dir(cfg)
        self.db = open_knowledge_db(self.state_dir)
        self.manga_id = manga_id
        self.cache = ResearchCache(self.db, self.manga_id) if self.manga_id else None

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------ identity
    def ensure_manga(self, data: dict) -> str:
        """Create/update the manga record; returns the manga_id."""
        self.manga_id = self.db.upsert_manga(data, self.manga_id)
        self.cache = ResearchCache(self.db, self.manga_id)
        return self.manga_id

    def identify_from_pdf_scan(self, scan: dict, pdf_path: str = "") -> str:
        """Create a manga record from a PDF identity scan (pdf_scan.scan_pdf)."""
        manga_data = {
            "title": scan.get("title", "") or "Untitled",
            "author": scan.get("author", ""),
            "pdf_path": pdf_path,
            "pdf_page_count": scan.get("page_count"),
        }
        return self.ensure_manga(manga_data)

    # ------------------------------------------------------------ research
    def run_research(self, force: bool = False) -> dict:
        """Run internet research with caching: book ref + character list."""
        if not self.manga_id:
            return {"status": "no_manga"}

        manga = self.db.get_manga(self.manga_id)
        title = (manga or {}).get("title", "")
        if not title or title == "Untitled":
            return {"status": "no_title"}

        from .internet_ref import fetch_book_ref, fetch_characters

        result = {"book_ref": None, "characters": None, "cached": False}

        def _book():
            return fetch_book_ref(title)
        book = self.cache.fetch(f"book_ref:{title}", _book, force=force)
        if book:
            from .research_cache import store_book_reference
            store_book_reference(self.db, self.manga_id, book, self.cache)
            result["book_ref"] = book.get("title") or book.get("source")
            result["cached"] = self.cache.get(f"book_ref:{title}") is not None

        def _chars():
            return {"characters": fetch_characters(title)}
        chars = self.cache.fetch(f"characters:{title}", _chars, force=force)
        if chars and chars.get("characters"):
            from .research_cache import store_character_list
            count = store_character_list(
                self.db, self.manga_id, chars["characters"], self.cache
            )
            result["characters"] = count

        return result

    # ------------------------------------------------------------ extract
    def extract_knowledge(self, pages: list[int]) -> dict:
        """Extract structured knowledge from analyzed pages into the DB."""
        if not self.manga_id:
            return {"pages_processed": 0, "error": "no_manga"}
        from .knowledge_extract import run_full_extraction
        return run_full_extraction(self.db, self.manga_id, self.cfg, pages)

    def extract_page(self, page: int) -> dict:
        """Extract a single page's knowledge into the DB (incremental)."""
        if not self.manga_id:
            return {"status": "no_manga"}
        from .knowledge_extract import ingest_page_to_knowledge_db
        return ingest_page_to_knowledge_db(self.db, self.manga_id, self.cfg, page)

    # ------------------------------------------------------------ chapters
    def detect_chapters(self, total_pages: int | None = None,
                        force: bool = False) -> list[dict]:
        """Detect chapter boundaries and store them in the DB."""
        if not self.manga_id:
            return []
        from .chapter_detect import detect_all_chapters, apply_chapter_detections

        if total_pages is None:
            manga = self.db.get_manga(self.manga_id)
            total_pages = (manga or {}).get("pdf_page_count") or 0
        if total_pages <= 0:
            return []

        chapters = detect_all_chapters(
            self.manga_id,
            Path(self.cfg.output.pages_dir),
            Path(self.cfg.output.ocr_dir),
            total_pages,
            self.db.get_manga(self.manga_id),
        )
        apply_chapter_detections(self.db, self.manga_id, chapters)
        return chapters

    # ------------------------------------------------------------ summaries
    def summarize_chapter(self, chapter: dict, llm=None) -> dict:
        """Generate multi-level summaries for one chapter."""
        if not self.manga_id:
            return {}
        from .story_memory import summarize_chapter as _sc
        return _sc(self.db, self.manga_id, chapter, llm, self.cfg)

    def summarize_all_chapters(self, llm=None) -> list[dict]:
        """Summarize every detected chapter."""
        if not self.manga_id:
            return []
        from .story_memory import summarize_chapter as _sc
        out = []
        for chapter in self.db.get_chapters(self.manga_id):
            out.append(_sc(self.db, self.manga_id, chapter, llm, self.cfg))
        return out

    # ------------------------------------------------------------ retrieval
    def relevant_memory(self, page: int | None = None,
                        chapter_id: int | None = None,
                        max_chars: int = 3000) -> str:
        """Build a bounded, hierarchical memory block for the LLM."""
        if not self.manga_id:
            return ""
        from .story_memory import retrieve_relevant_memory
        return retrieve_relevant_memory(
            self.db, self.manga_id, page, chapter_id, max_chars
        )

    # ------------------------------------------------------------ stats
    def stats(self) -> dict:
        if not self.manga_id:
            return {}
        return self.db.stats(self.manga_id)
