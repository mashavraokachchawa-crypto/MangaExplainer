"""Persistent structured knowledge database for MangaExplainer.

SQLite-backed, zero external dependencies (stdlib only).  One database file
per manga project lives at ``state/manga_knowledge.db``.  The schema is
versioned so future migrations are automatic.

Design goals
============
- Every record tracks **source** (pdf_page_NNN | internet | user) and
  **confidence** so the system never silently overwrites PDF evidence with
  internet guesses.
- Complex nested data (visual descriptions, panel analysis, event details)
  is stored as compact JSON blobs via TEXT columns.
- Character deduplication: characters get a stable ``character_id``; aliases,
  unknowns, and later identifications all merge into one profile.
- Hierarchical story memory: manga → arc → chapter → page → panel.
- Batch-friendly: open once per pipeline stage, close when done; never hold
  the database open across long sleeps.

Low-RAM: only one row is ever held at a time during ingestion; SQLite's
WAL mode keeps concurrent reads possible while the pipeline writes.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger("mangaexplainer.knowledge_db")

DB_VERSION = 1
DB_FILENAME = "manga_knowledge.db"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_dumps(obj) -> str:
    if obj is None:
        return "null"
    return json.dumps(obj, ensure_ascii=False)


def _json_loads(text):
    if text is None or text == "null":
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Schema (versioned)
# ---------------------------------------------------------------------------

_SCHEMA_V1 = """
-- Manga identity & metadata (one row per manga)
CREATE TABLE IF NOT EXISTS manga (
    manga_id        TEXT PRIMARY KEY,
    title           TEXT NOT NULL DEFAULT '',
    alt_titles      TEXT NOT NULL DEFAULT '[]',     -- JSON array
    japanese_title  TEXT NOT NULL DEFAULT '',
    author          TEXT NOT NULL DEFAULT '',
    illustrator     TEXT NOT NULL DEFAULT '',
    publisher       TEXT NOT NULL DEFAULT '',
    magazine        TEXT NOT NULL DEFAULT '',
    genres          TEXT NOT NULL DEFAULT '[]',      -- JSON array
    demographic     TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT '',         -- ongoing/completed/etc
    total_chapters  INTEGER,
    total_volumes   INTEGER,
    synopsis        TEXT NOT NULL DEFAULT '',
    cover_url       TEXT NOT NULL DEFAULT '',
    pdf_path        TEXT NOT NULL DEFAULT '',
    pdf_page_count  INTEGER,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

-- Chapter boundaries detected from PDF + internet
CREATE TABLE IF NOT EXISTS chapters (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    manga_id        TEXT NOT NULL REFERENCES manga(manga_id),
    chapter_number  INTEGER,         -- detected or user-provided
    title           TEXT NOT NULL DEFAULT '',
    pdf_page_start  INTEGER NOT NULL,
    pdf_page_end    INTEGER NOT NULL,
    confidence      REAL NOT NULL DEFAULT 0.5,
    source          TEXT NOT NULL DEFAULT 'pdf',  -- pdf | internet | user
    extra           TEXT NOT NULL DEFAULT '{}',    -- JSON: detection signals
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    UNIQUE(manga_id, pdf_page_start)
);

-- Structured character profiles (one row per character)
CREATE TABLE IF NOT EXISTS characters (
    manga_id        TEXT NOT NULL REFERENCES manga(manga_id),
    character_id    TEXT NOT NULL,   -- stable auto-id: char_XXXX
    name            TEXT NOT NULL DEFAULT '',
    alt_names       TEXT NOT NULL DEFAULT '[]',   -- JSON array
    aliases         TEXT NOT NULL DEFAULT '[]',   -- JSON array
    description     TEXT NOT NULL DEFAULT '',
    visual_traits   TEXT NOT NULL DEFAULT '{}',   -- JSON: hair, build, attire...
    role            TEXT NOT NULL DEFAULT '',
    personality     TEXT NOT NULL DEFAULT '',
    abilities       TEXT NOT NULL DEFAULT '[]',   -- JSON array
    first_page      INTEGER,
    last_page       INTEGER,
    appearance_count INTEGER NOT NULL DEFAULT 0,
    confidence      REAL NOT NULL DEFAULT 0.5,
    state           TEXT NOT NULL DEFAULT 'auto',  -- auto|verified|user_corrected|uncertain|conflicted
    identity_note   TEXT NOT NULL DEFAULT '',       -- "Possible identity: X 55%, Y 30%"
    extra           TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (manga_id, character_id)
);

-- Character aliases resolved from unknowns
CREATE TABLE IF NOT EXISTS character_aliases (
    manga_id        TEXT NOT NULL,
    character_id    TEXT NOT NULL,
    alias           TEXT NOT NULL,
    source          TEXT NOT NULL DEFAULT 'pdf',
    confidence      REAL NOT NULL DEFAULT 1.0,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (manga_id, alias)
);

-- Relationships between characters
CREATE TABLE IF NOT EXISTS relationships (
    manga_id        TEXT NOT NULL,
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    character_a_id  TEXT NOT NULL,
    character_b_id  TEXT NOT NULL,
    relationship    TEXT NOT NULL DEFAULT '',  -- "enemy", "ally", "lover", etc.
    description     TEXT NOT NULL DEFAULT '',
    source          TEXT NOT NULL DEFAULT 'pdf',
    confidence      REAL NOT NULL DEFAULT 0.5,
    first_page      INTEGER,
    created_at      TEXT NOT NULL,
    UNIQUE(manga_id, character_a_id, character_b_id, relationship)
);

-- Locations / places
CREATE TABLE IF NOT EXISTS locations (
    manga_id        TEXT NOT NULL,
    location_id     TEXT NOT NULL,
    name            TEXT NOT NULL DEFAULT '',
    description     TEXT NOT NULL DEFAULT '',
    location_type   TEXT NOT NULL DEFAULT '',   -- city, building, forest, etc.
    first_page      INTEGER,
    last_page       INTEGER,
    appearance_count INTEGER NOT NULL DEFAULT 0,
    confidence      REAL NOT NULL DEFAULT 0.5,
    source          TEXT NOT NULL DEFAULT 'pdf',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (manga_id, location_id)
);

-- Organizations / groups
CREATE TABLE IF NOT EXISTS organizations (
    manga_id        TEXT NOT NULL,
    org_id          TEXT NOT NULL,
    name            TEXT NOT NULL DEFAULT '',
    description     TEXT NOT NULL DEFAULT '',
    members         TEXT NOT NULL DEFAULT '[]',  -- JSON array of character_ids
    first_page      INTEGER,
    confidence      REAL NOT NULL DEFAULT 0.5,
    source          TEXT NOT NULL DEFAULT 'pdf',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (manga_id, org_id)
);

-- Powers / abilities / items
CREATE TABLE IF NOT EXISTS abilities (
    manga_id        TEXT NOT NULL,
    ability_id      TEXT NOT NULL,
    name            TEXT NOT NULL DEFAULT '',
    description     TEXT NOT NULL DEFAULT '',
    ability_type    TEXT NOT NULL DEFAULT '',  -- power, technique, item, etc.
    character_id    TEXT,                      -- who has this ability
    confidence      REAL NOT NULL DEFAULT 0.5,
    source          TEXT NOT NULL DEFAULT 'pdf',
    created_at      TEXT NOT NULL,
    PRIMARY KEY (manga_id, ability_id)
);

-- Story events (hierarchical: page → chapter → arc)
CREATE TABLE IF NOT EXISTS events (
    manga_id        TEXT NOT NULL,
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type      TEXT NOT NULL DEFAULT 'page',  -- page | chapter | arc | manga
    page_number     INTEGER,
    chapter_id      INTEGER REFERENCES chapters(id),
    characters      TEXT NOT NULL DEFAULT '[]',    -- JSON array of character_ids
    location        TEXT NOT NULL DEFAULT '',
    description     TEXT NOT NULL DEFAULT '',
    importance      REAL NOT NULL DEFAULT 0.5,     -- 0..1
    source          TEXT NOT NULL DEFAULT 'pdf',
    confidence      REAL NOT NULL DEFAULT 0.5,
    extra           TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL
);

-- Hierarchical summaries
CREATE TABLE IF NOT EXISTS summaries (
    manga_id        TEXT NOT NULL,
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    summary_type    TEXT NOT NULL,     -- page | chapter_short | chapter_medium | chapter_detail | arc
    page_number     INTEGER,
    chapter_id      INTEGER,
    text            TEXT NOT NULL DEFAULT '',
    important_events TEXT NOT NULL DEFAULT '[]',
    new_characters  TEXT NOT NULL DEFAULT '[]',
    unresolved      TEXT NOT NULL DEFAULT '[]',
    connections     TEXT NOT NULL DEFAULT '[]',
    confidence      REAL NOT NULL DEFAULT 0.7,
    created_at      TEXT NOT NULL
);

-- Source evidence: tracks where every piece of info came from
CREATE TABLE IF NOT EXISTS source_evidence (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    manga_id        TEXT NOT NULL,
    entity_type     TEXT NOT NULL,     -- character | location | chapter | event | metadata
    entity_key      TEXT NOT NULL,     -- character_id, location_id, etc.
    source_type     TEXT NOT NULL,     -- pdf | internet | user | ocr | vlm
    source_url      TEXT NOT NULL DEFAULT '',
    pdf_page        INTEGER,
    panel_id        TEXT NOT NULL DEFAULT '',
    detail          TEXT NOT NULL DEFAULT '',  -- the actual evidence text
    confidence      REAL NOT NULL DEFAULT 0.5,
    created_at      TEXT NOT NULL
);

-- Conflict tracking
CREATE TABLE IF NOT EXISTS conflicts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    manga_id        TEXT NOT NULL,
    entity_type     TEXT NOT NULL,
    entity_key      TEXT NOT NULL,
    field_name      TEXT NOT NULL,
    value_a         TEXT NOT NULL,
    source_a        TEXT NOT NULL,
    value_b         TEXT NOT NULL,
    source_b        TEXT NOT NULL,
    resolution      TEXT NOT NULL DEFAULT '',  -- empty = unresolved
    resolved_value  TEXT NOT NULL DEFAULT '',
    resolved_by     TEXT NOT NULL DEFAULT '',  -- user | auto
    created_at      TEXT NOT NULL,
    resolved_at     TEXT
);

-- Timeline entries (ordered chronological facts)
CREATE TABLE IF NOT EXISTS timeline (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    manga_id        TEXT NOT NULL,
    chapter_id      INTEGER,
    page_number     INTEGER,
    time_label      TEXT NOT NULL DEFAULT '',  -- "Chapter 5", "3 days later"
    event_text      TEXT NOT NULL DEFAULT '',
    importance      REAL NOT NULL DEFAULT 0.5,
    confidence      REAL NOT NULL DEFAULT 0.5,
    source          TEXT NOT NULL DEFAULT 'pdf',
    created_at      TEXT NOT NULL
);

-- Terminology / jargon unique to this manga
CREATE TABLE IF NOT EXISTS terminology (
    manga_id        TEXT NOT NULL,
    term            TEXT NOT NULL,
    definition      TEXT NOT NULL DEFAULT '',
    category        TEXT NOT NULL DEFAULT '',  -- power_system, race, place, etc.
    source          TEXT NOT NULL DEFAULT 'pdf',
    confidence      REAL NOT NULL DEFAULT 0.5,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (manga_id, term)
);

-- Research cache (avoid re-fetching the same internet queries)
CREATE TABLE IF NOT EXISTS research_cache (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    manga_id        TEXT NOT NULL,
    query           TEXT NOT NULL,
    source_url      TEXT NOT NULL DEFAULT '',
    result_json     TEXT NOT NULL DEFAULT '{}',  -- full fetch result
    fetched_at      TEXT NOT NULL,
    expires_at      TEXT,  -- null = never expires
    UNIQUE(manga_id, query, source_url)
);

-- Processing checkpoint for the knowledge pipeline itself
CREATE TABLE IF NOT EXISTS processing_checkpoints (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    manga_id        TEXT NOT NULL,
    stage           TEXT NOT NULL,     -- identify | research | extract | chapter_detect | etc.
    key_value       TEXT NOT NULL,     -- page number (as text) or "*" for whole-manga
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending | running | completed | failed
    detail          TEXT NOT NULL DEFAULT '',
    started_at      TEXT,
    completed_at    TEXT,
    UNIQUE(manga_id, stage, key_value)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_chapters_manga ON chapters(manga_id);
CREATE INDEX IF NOT EXISTS idx_characters_manga ON characters(manga_id);
CREATE INDEX IF NOT EXISTS idx_characters_name ON characters(manga_id, name);
CREATE INDEX IF NOT EXISTS idx_events_manga ON events(manga_id);
CREATE INDEX IF NOT EXISTS idx_events_page ON events(manga_id, page_number);
CREATE INDEX IF NOT EXISTS idx_events_chapter ON events(manga_id, chapter_id);
CREATE INDEX IF NOT EXISTS idx_summaries_manga ON summaries(manga_id);
CREATE INDEX IF NOT EXISTS idx_summaries_chapter ON summaries(manga_id, chapter_id);
CREATE INDEX IF NOT EXISTS idx_source_evidence_entity ON source_evidence(manga_id, entity_type, entity_key);
CREATE INDEX IF NOT EXISTS idx_conflicts_entity ON conflicts(manga_id, entity_type, entity_key);
CREATE INDEX IF NOT EXISTS idx_locations_manga ON locations(manga_id);
CREATE INDEX IF NOT EXISTS idx_terminology_manga ON terminology(manga_id);
CREATE INDEX IF NOT EXISTS idx_research_cache_manga ON research_cache(manga_id, query);
CREATE INDEX IF NOT EXISTS idx_timeline_manga ON timeline(manga_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_manga ON processing_checkpoints(manga_id, stage);
"""


# ---------------------------------------------------------------------------
# Database connection manager
# ---------------------------------------------------------------------------

class KnowledgeDB:
    """Persistent SQLite-backed knowledge store for one manga project.

    Usage::

        db = KnowledgeDB.open(state_dir)
        db.upsert_manga({...})
        db.add_character(manga_id, {...})
        ...
        db.close()

    Or use as a context manager::

        with KnowledgeDB.open(state_dir) as db:
            ...

    Always close when done — SQLite file locks are held until the connection
    is closed.
    """

    def __init__(self, path: Path):
        self.path = path
        self._conn: sqlite3.Connection | None = None

    @classmethod
    def open(cls, state_dir: Path) -> "KnowledgeDB":
        state_dir = Path(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        db = cls(state_dir / DB_FILENAME)
        db._connect()
        return db

    def _connect(self):
        self._conn = sqlite3.connect(
            str(self.path),
            timeout=30,
            isolation_level="DEFERRED",
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._apply_schema()

    def _apply_schema(self):
        cur = self._conn.cursor()
        cur.executescript(_SCHEMA_V1)
        # Version marker
        cur.execute(
            "CREATE TABLE IF NOT EXISTS _meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        row = cur.execute("SELECT value FROM _meta WHERE key='version'").fetchone()
        if row is None:
            cur.execute(
                "INSERT INTO _meta(key, value) VALUES('version', ?)",
                (str(DB_VERSION),),
            )
        self._conn.commit()

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("KnowledgeDB is closed")
        return self._conn

    def commit(self):
        self.conn.commit()

    # -------------------------------------------------------------------
    # MANGA
    # -------------------------------------------------------------------

    def upsert_manga(self, data: dict, manga_id: str | None = None) -> str:
        """Create or update the manga record.  Returns the manga_id."""
        mid = manga_id or data.get("manga_id") or self._make_manga_id(data)
        now = _now()
        existing = self.conn.execute(
            "SELECT manga_id FROM manga WHERE manga_id=?", (mid,)
        ).fetchone()
        if existing is None:
            self.conn.execute(
                """INSERT INTO manga(
                    manga_id, title, alt_titles, japanese_title,
                    author, illustrator, publisher, magazine,
                    genres, demographic, status, total_chapters, total_volumes,
                    synopsis, cover_url, pdf_path, pdf_page_count,
                    created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    mid,
                    data.get("title", ""),
                    _json_dumps(data.get("alt_titles") or []),
                    data.get("japanese_title", ""),
                    data.get("author", ""),
                    data.get("illustrator", ""),
                    data.get("publisher", ""),
                    data.get("magazine", ""),
                    _json_dumps(data.get("genres") or []),
                    data.get("demographic", ""),
                    data.get("status", ""),
                    data.get("total_chapters"),
                    data.get("total_volumes"),
                    data.get("synopsis", ""),
                    data.get("cover_url", ""),
                    data.get("pdf_path", ""),
                    data.get("pdf_page_count"),
                    now,
                    now,
                ),
            )
        else:
            # Merge: only overwrite empty fields with non-empty new values
            fields_to_merge = {
                "title": data.get("title", ""),
                "alt_titles": _json_dumps(data.get("alt_titles") or []),
                "japanese_title": data.get("japanese_title", ""),
                "author": data.get("author", ""),
                "illustrator": data.get("illustrator", ""),
                "publisher": data.get("publisher", ""),
                "magazine": data.get("magazine", ""),
                "genres": _json_dumps(data.get("genres") or []),
                "demographic": data.get("demographic", ""),
                "status": data.get("status", ""),
                "total_chapters": data.get("total_chapters"),
                "total_volumes": data.get("total_volumes"),
                "synopsis": data.get("synopsis", ""),
                "cover_url": data.get("cover_url", ""),
                "pdf_path": data.get("pdf_path", ""),
                "pdf_page_count": data.get("pdf_page_count"),
            }
            # Only update fields that are non-empty/non-None
            sets = []
            vals = []
            for field, new_val in fields_to_merge.items():
                if new_val is not None and new_val != "" and new_val != "[]":
                    sets.append(f"{field} = ?")
                    vals.append(new_val)
            if sets:
                sets.append("updated_at = ?")
                vals.append(now)
                vals.append(mid)
                self.conn.execute(
                    f"UPDATE manga SET {', '.join(sets)} WHERE manga_id=?",
                    vals,
                )
        self.conn.commit()
        return mid

    def get_manga(self, manga_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM manga WHERE manga_id=?", (manga_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_manga(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM manga ORDER BY updated_at DESC"
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def _make_manga_id(self, data: dict) -> str:
        import re, hashlib
        title = data.get("title", "unknown")
        t = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:32]
        h = hashlib.md5(title.encode()).hexdigest()[:6]
        return f"manga_{t}_{h}"

    # -------------------------------------------------------------------
    # CHAPTERS
    # -------------------------------------------------------------------

    def add_chapter(self, manga_id: str, data: dict) -> int:
        """Insert a chapter boundary.  Returns the chapter id."""
        now = _now()
        cur = self.conn.execute(
            """INSERT INTO chapters(
                manga_id, chapter_number, title, pdf_page_start, pdf_page_end,
                confidence, source, extra, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                manga_id,
                data.get("chapter_number"),
                data.get("title", ""),
                data["pdf_page_start"],
                data["pdf_page_end"],
                data.get("confidence", 0.5),
                data.get("source", "pdf"),
                _json_dumps(data.get("extra") or {}),
                now,
                now,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_chapters(self, manga_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM chapters WHERE manga_id=? ORDER BY pdf_page_start",
            (manga_id,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def chapter_for_page(self, manga_id: str, page: int) -> dict | None:
        row = self.conn.execute(
            """SELECT * FROM chapters WHERE manga_id=?
                AND pdf_page_start <= ? AND pdf_page_end >= ?
                ORDER BY pdf_page_start LIMIT 1""",
            (manga_id, page, page),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def update_chapter(self, chapter_id: int, data: dict):
        now = _now()
        sets = []
        vals = []
        for key in ("chapter_number", "title", "pdf_page_start", "pdf_page_end",
                     "confidence", "source"):
            if key in data:
                sets.append(f"{key} = ?")
                vals.append(data[key])
        if "extra" in data:
            sets.append("extra = ?")
            vals.append(_json_dumps(data["extra"]))
        if sets:
            sets.append("updated_at = ?")
            vals.append(now)
            vals.append(chapter_id)
            self.conn.execute(
                f"UPDATE chapters SET {', '.join(sets)} WHERE id=?",
                vals,
            )
            self.conn.commit()

    def delete_chapter(self, chapter_id: int):
        self.conn.execute("DELETE FROM chapters WHERE id=?", (chapter_id,))
        self.conn.commit()

    # -------------------------------------------------------------------
    # CHARACTERS
    # -------------------------------------------------------------------

    def add_character(self, manga_id: str, data: dict) -> str:
        """Insert or merge a character.  Returns the character_id."""
        char_id = data.get("character_id") or self._make_char_id(manga_id, data)
        now = _now()
        existing = self.conn.execute(
            "SELECT character_id FROM characters WHERE manga_id=? AND character_id=?",
            (manga_id, char_id),
        ).fetchone()
        if existing is None:
            self.conn.execute(
                """INSERT INTO characters(
                    manga_id, character_id, name, alt_names, aliases,
                    description, visual_traits, role, personality, abilities,
                    first_page, last_page, appearance_count, confidence,
                    state, identity_note, extra, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    manga_id,
                    char_id,
                    data.get("name", ""),
                    _json_dumps(data.get("alt_names") or []),
                    _json_dumps(data.get("aliases") or []),
                    data.get("description", ""),
                    _json_dumps(data.get("visual_traits") or {}),
                    data.get("role", ""),
                    data.get("personality", ""),
                    _json_dumps(data.get("abilities") or []),
                    data.get("first_page"),
                    data.get("last_page"),
                    data.get("appearance_count", 1),
                    data.get("confidence", 0.5),
                    data.get("state", "auto"),
                    data.get("identity_note", ""),
                    _json_dumps(data.get("extra") or {}),
                    now,
                    now,
                ),
            )
        else:
            # Merge: enrich existing record
            sets = ["updated_at = ?"]
            vals = [now]
            # Only overwrite empty fields
            for field in ("name", "description", "role", "personality", "identity_note"):
                new_val = data.get(field)
                if new_val and new_val != "unknown":
                    sets.append(f"{field} = ?")
                    vals.append(new_val)
            # Merge arrays (alt_names, aliases, abilities)
            for arr_field in ("alt_names", "aliases", "abilities"):
                new_arr = data.get(arr_field) or []
                if new_arr:
                    old_row = self.conn.execute(
                        f"SELECT {arr_field} FROM characters WHERE manga_id=? AND character_id=?",
                        (manga_id, char_id),
                    ).fetchone()
                    old_arr = _json_loads(old_row[arr_field] if old_row else "[]") or []
                    merged = list(dict.fromkeys(old_arr + new_arr))[:20]
                    sets.append(f"{arr_field} = ?")
                    vals.append(_json_dumps(merged))
            # Update visual traits (merge keys)
            new_traits = data.get("visual_traits") or {}
            if new_traits:
                old_row = self.conn.execute(
                    "SELECT visual_traits FROM characters WHERE manga_id=? AND character_id=?",
                    (manga_id, char_id),
                ).fetchone()
                old_traits = _json_loads(old_row["visual_traits"] if old_row else "{}") or {}
                old_traits.update({k: v for k, v in new_traits.items() if v})
                sets.append("visual_traits = ?")
                vals.append(_json_dumps(old_traits))
            # Update page range
            if data.get("first_page") is not None:
                sets.append("first_page = MIN(COALESCE(first_page, ?), ?)")
                vals.extend([data["first_page"], data["first_page"]])
            if data.get("last_page") is not None:
                sets.append("last_page = MAX(COALESCE(last_page, 0), ?)")
                vals.append(data["last_page"])
            # Bump appearance count
            sets.append("appearance_count = appearance_count + 1")
            # Merge confidence
            if data.get("confidence", 0) > 0:
                sets.append("confidence = MAX(confidence, ?)")
                vals.append(data["confidence"])
            vals.append(manga_id)
            vals.append(char_id)
            self.conn.execute(
                f"UPDATE characters SET {', '.join(sets)} WHERE manga_id=? AND character_id=?",
                vals,
            )
        # Register alias
        name = data.get("name", "")
        if name and name.lower() not in ("unknown", "unk", "n/a", ""):
            self._register_alias(manga_id, char_id, name, data.get("source", "pdf"))
        for alias in (data.get("aliases") or []):
            if alias and alias != name:
                self._register_alias(manga_id, char_id, alias, data.get("source", "pdf"))
        self.conn.commit()
        return char_id

    def _register_alias(self, manga_id, char_id, alias, source):
        try:
            self.conn.execute(
                """INSERT OR REPLACE INTO character_aliases(
                    manga_id, character_id, alias, source, confidence, created_at)
                VALUES (?,?,?,?,1.0,?)""",
                (manga_id, char_id, alias, source, _now()),
            )
        except sqlite3.IntegrityError:
            pass

    def resolve_character(self, manga_id: str, name: str) -> str | None:
        """Resolve any alias/name to the canonical character_id, or None."""
        if not name or name.lower() in ("unknown", "unk", "n/a"):
            return None
        row = self.conn.execute(
            "SELECT character_id FROM character_aliases WHERE manga_id=? AND alias=?",
            (manga_id, name),
        ).fetchone()
        if row:
            return row["character_id"]
        # Fuzzy: exact name match
        row = self.conn.execute(
            "SELECT character_id FROM characters WHERE manga_id=? AND name=?",
            (manga_id, name),
        ).fetchone()
        if row:
            return row["character_id"]
        return None

    def update_character_note(self, manga_id: str, character_id: str,
                              identity_note: str = "", extra: dict | None = None):
        """Update a character's identity note and/or extra JSON."""
        if identity_note:
            self.conn.execute(
                "UPDATE characters SET identity_note=?, updated_at=? WHERE manga_id=? AND character_id=?",
                (identity_note, _now(), manga_id, character_id),
            )
        if extra:
            row = self.conn.execute(
                "SELECT extra FROM characters WHERE manga_id=? AND character_id=?",
                (manga_id, character_id),
            ).fetchone()
            old_extra = {}
            if row is not None:
                old_extra = _json_loads(row["extra"]) or {}
            old_extra.update(extra)
            self.conn.execute(
                "UPDATE characters SET extra=?, updated_at=? WHERE manga_id=? AND character_id=?",
                (_json_dumps(old_extra), _now(), manga_id, character_id),
            )
        self.conn.commit()

    def get_character(self, manga_id: str, character_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM characters WHERE manga_id=? AND character_id=?",
            (manga_id, character_id),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_characters(self, manga_id: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT * FROM characters WHERE manga_id=?
               ORDER BY appearance_count DESC, first_page ASC""",
            (manga_id,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_character_aliases(self, manga_id: str, character_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM character_aliases WHERE manga_id=? AND character_id=?",
            (manga_id, character_id),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def _make_char_id(self, manga_id, data) -> str:
        import hashlib
        name = data.get("name", "unknown")
        h = hashlib.md5(f"{manga_id}:{name}".encode()).hexdigest()[:8]
        return f"char_{h}"

    # -------------------------------------------------------------------
    # RELATIONSHIPS
    # -------------------------------------------------------------------

    def add_relationship(self, manga_id: str, data: dict) -> int:
        cur = self.conn.execute(
            """INSERT OR REPLACE INTO relationships(
                manga_id, character_a_id, character_b_id,
                relationship, description, source, confidence,
                first_page, created_at)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                manga_id,
                data["character_a_id"],
                data["character_b_id"],
                data.get("relationship", ""),
                data.get("description", ""),
                data.get("source", "pdf"),
                data.get("confidence", 0.5),
                data.get("first_page"),
                _now(),
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_relationships(self, manga_id: str, character_id: str | None = None) -> list[dict]:
        if character_id:
            rows = self.conn.execute(
                """SELECT * FROM relationships WHERE manga_id=?
                   AND (character_a_id=? OR character_b_id=?)
                   ORDER BY confidence DESC""",
                (manga_id, character_id, character_id),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM relationships WHERE manga_id=? ORDER BY confidence DESC",
                (manga_id,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # -------------------------------------------------------------------
    # LOCATIONS
    # -------------------------------------------------------------------

    def add_location(self, manga_id: str, data: dict) -> str:
        loc_id = data.get("location_id") or f"loc_{hash(data.get('name','')) & 0xFFFF:04x}"
        now = _now()
        existing = self.conn.execute(
            "SELECT location_id FROM locations WHERE manga_id=? AND location_id=?",
            (manga_id, loc_id),
        ).fetchone()
        if existing is None:
            self.conn.execute(
                """INSERT INTO locations(
                    manga_id, location_id, name, description, location_type,
                    first_page, last_page, appearance_count, confidence,
                    source, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    manga_id, loc_id,
                    data.get("name", ""), data.get("description", ""),
                    data.get("location_type", ""),
                    data.get("first_page"), data.get("last_page"),
                    data.get("appearance_count", 1),
                    data.get("confidence", 0.5),
                    data.get("source", "pdf"),
                    now, now,
                ),
            )
        else:
            self.conn.execute(
                """UPDATE locations SET
                    last_page = MAX(COALESCE(last_page, 0), ?),
                    appearance_count = appearance_count + 1,
                    updated_at = ?
                WHERE manga_id=? AND location_id=?""",
                (data.get("last_page", 0), now, manga_id, loc_id),
            )
        self.conn.commit()
        return loc_id

    def get_locations(self, manga_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM locations WHERE manga_id=? ORDER BY appearance_count DESC",
            (manga_id,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # -------------------------------------------------------------------
    # ORGANIZATIONS
    # -------------------------------------------------------------------

    def add_organization(self, manga_id: str, data: dict) -> str:
        org_id = data.get("org_id") or f"org_{hash(data.get('name','')) & 0xFFFF:04x}"
        now = _now()
        self.conn.execute(
            """INSERT OR REPLACE INTO organizations(
                manga_id, org_id, name, description, members,
                first_page, confidence, source, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                manga_id, org_id,
                data.get("name", ""), data.get("description", ""),
                _json_dumps(data.get("members") or []),
                data.get("first_page"),
                data.get("confidence", 0.5),
                data.get("source", "pdf"),
                now, now,
            ),
        )
        self.conn.commit()
        return org_id

    def get_organizations(self, manga_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM organizations WHERE manga_id=?", (manga_id,)
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # -------------------------------------------------------------------
    # ABILITIES
    # -------------------------------------------------------------------

    def add_ability(self, manga_id: str, data: dict) -> str:
        ab_id = data.get("ability_id") or f"ab_{hash(data.get('name','')) & 0xFFFF:04x}"
        now = _now()
        self.conn.execute(
            """INSERT OR REPLACE INTO abilities(
                manga_id, ability_id, name, description, ability_type,
                character_id, confidence, source, created_at)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                manga_id, ab_id,
                data.get("name", ""), data.get("description", ""),
                data.get("ability_type", ""),
                data.get("character_id"),
                data.get("confidence", 0.5),
                data.get("source", "pdf"),
                now,
            ),
        )
        self.conn.commit()
        return ab_id

    def get_abilities(self, manga_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM abilities WHERE manga_id=?", (manga_id,)
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # -------------------------------------------------------------------
    # EVENTS
    # -------------------------------------------------------------------

    def add_event(self, manga_id: str, data: dict) -> int:
        cur = self.conn.execute(
            """INSERT INTO events(
                manga_id, event_type, page_number, chapter_id,
                characters, location, description, importance,
                source, confidence, extra, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                manga_id,
                data.get("event_type", "page"),
                data.get("page_number"),
                data.get("chapter_id"),
                _json_dumps(data.get("characters") or []),
                data.get("location", ""),
                data.get("description", ""),
                data.get("importance", 0.5),
                data.get("source", "pdf"),
                data.get("confidence", 0.5),
                _json_dumps(data.get("extra") or {}),
                _now(),
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_events(self, manga_id: str, page: int | None = None,
                   chapter_id: int | None = None, event_type: str | None = None,
                   min_importance: float = 0.0) -> list[dict]:
        """Query events with flexible filters. Returns newest first for page=None."""
        conditions = ["manga_id = ?"]
        vals: list = [manga_id]
        if page is not None:
            conditions.append("page_number = ?")
            vals.append(page)
        if chapter_id is not None:
            conditions.append("chapter_id = ?")
            vals.append(chapter_id)
        if event_type:
            conditions.append("event_type = ?")
            vals.append(event_type)
        if min_importance > 0:
            conditions.append("importance >= ?")
            vals.append(min_importance)
        where = " AND ".join(conditions)
        order = "page_number ASC, id ASC" if page is None else "id ASC"
        rows = self.conn.execute(
            f"SELECT * FROM events WHERE {where} ORDER BY {order}",
            vals,
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # -------------------------------------------------------------------
    # SUMMARIES
    # -------------------------------------------------------------------

    def add_summary(self, manga_id: str, data: dict) -> int:
        cur = self.conn.execute(
            """INSERT INTO summaries(
                manga_id, summary_type, page_number, chapter_id,
                text, important_events, new_characters, unresolved,
                connections, confidence, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                manga_id,
                data["summary_type"],
                data.get("page_number"),
                data.get("chapter_id"),
                data.get("text", ""),
                _json_dumps(data.get("important_events") or []),
                _json_dumps(data.get("new_characters") or []),
                _json_dumps(data.get("unresolved") or []),
                _json_dumps(data.get("connections") or []),
                data.get("confidence", 0.7),
                _now(),
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_summaries(self, manga_id: str, summary_type: str | None = None,
                      chapter_id: int | None = None) -> list[dict]:
        conditions = ["manga_id = ?"]
        vals: list = [manga_id]
        if summary_type:
            conditions.append("summary_type = ?")
            vals.append(summary_type)
        if chapter_id is not None:
            conditions.append("chapter_id = ?")
            vals.append(chapter_id)
        where = " AND ".join(conditions)
        rows = self.conn.execute(
            f"SELECT * FROM summaries WHERE {where} ORDER BY page_number ASC, id ASC",
            vals,
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # -------------------------------------------------------------------
    # SOURCE EVIDENCE
    # -------------------------------------------------------------------

    def add_source_evidence(self, manga_id: str, data: dict) -> int:
        cur = self.conn.execute(
            """INSERT INTO source_evidence(
                manga_id, entity_type, entity_key, source_type,
                source_url, pdf_page, panel_id, detail,
                confidence, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                manga_id,
                data["entity_type"],
                data["entity_key"],
                data["source_type"],
                data.get("source_url", ""),
                data.get("pdf_page"),
                data.get("panel_id", ""),
                data.get("detail", ""),
                data.get("confidence", 0.5),
                _now(),
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_source_evidence(self, manga_id: str, entity_type: str,
                           entity_key: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT * FROM source_evidence
               WHERE manga_id=? AND entity_type=? AND entity_key=?
               ORDER BY confidence DESC, created_at ASC""",
            (manga_id, entity_type, entity_key),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # -------------------------------------------------------------------
    # CONFLICTS
    # -------------------------------------------------------------------

    def add_conflict(self, manga_id: str, data: dict) -> int:
        cur = self.conn.execute(
            """INSERT INTO conflicts(
                manga_id, entity_type, entity_key, field_name,
                value_a, source_a, value_b, source_b,
                resolution, resolved_value, resolved_by,
                created_at, resolved_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                manga_id,
                data["entity_type"],
                data["entity_key"],
                data["field_name"],
                data["value_a"],
                data["source_a"],
                data["value_b"],
                data["source_b"],
                data.get("resolution", ""),
                data.get("resolved_value", ""),
                data.get("resolved_by", ""),
                _now(),
                None,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def resolve_conflict(self, conflict_id: int, resolved_value: str,
                        resolved_by: str = "user"):
        self.conn.execute(
            """UPDATE conflicts SET
                resolution='resolved', resolved_value=?, resolved_by=?, resolved_at=?
            WHERE id=?""",
            (resolved_value, resolved_by, _now(), conflict_id),
        )
        self.conn.commit()

    def get_unresolved_conflicts(self, manga_id: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT * FROM conflicts WHERE manga_id=?
               AND resolution='' ORDER BY created_at DESC""",
            (manga_id,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # -------------------------------------------------------------------
    # TIMELINE
    # -------------------------------------------------------------------

    def add_timeline_entry(self, manga_id: str, data: dict) -> int:
        cur = self.conn.execute(
            """INSERT INTO timeline(
                manga_id, chapter_id, page_number, time_label,
                event_text, importance, confidence, source, created_at)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                manga_id,
                data.get("chapter_id"),
                data.get("page_number"),
                data.get("time_label", ""),
                data.get("event_text", ""),
                data.get("importance", 0.5),
                data.get("confidence", 0.5),
                data.get("source", "pdf"),
                _now(),
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_timeline(self, manga_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM timeline WHERE manga_id=? ORDER BY page_number ASC, id ASC",
            (manga_id,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # -------------------------------------------------------------------
    # TERMINOLOGY
    # -------------------------------------------------------------------

    def add_terminology(self, manga_id: str, data: dict):
        self.conn.execute(
            """INSERT OR REPLACE INTO terminology(
                manga_id, term, definition, category, source, confidence, created_at)
            VALUES (?,?,?,?,?,?,?)""",
            (
                manga_id,
                data["term"],
                data.get("definition", ""),
                data.get("category", ""),
                data.get("source", "pdf"),
                data.get("confidence", 0.5),
                _now(),
            ),
        )
        self.conn.commit()

    def get_terminology(self, manga_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM terminology WHERE manga_id=? ORDER BY term",
            (manga_id,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # -------------------------------------------------------------------
    # RESEARCH CACHE
    # -------------------------------------------------------------------

    def cache_research(self, manga_id: str, query: str, source_url: str,
                       result: dict):
        self.conn.execute(
            """INSERT OR REPLACE INTO research_cache(
                manga_id, query, source_url, result_json, fetched_at)
            VALUES (?,?,?,?,?)""",
            (manga_id, query, source_url, _json_dumps(result), _now()),
        )
        self.conn.commit()

    def get_cached_research(self, manga_id: str, query: str,
                            source_url: str = "") -> dict | None:
        if source_url:
            row = self.conn.execute(
                """SELECT result_json FROM research_cache
                   WHERE manga_id=? AND query=? AND source_url=?""",
                (manga_id, query, source_url),
            ).fetchone()
        else:
            row = self.conn.execute(
                """SELECT result_json FROM research_cache
                   WHERE manga_id=? AND query=? ORDER BY fetched_at DESC LIMIT 1""",
                (manga_id, query),
            ).fetchone()
        return _json_loads(row["result_json"]) if row else None

    # -------------------------------------------------------------------
    # PROCESSING CHECKPOINTS
    # -------------------------------------------------------------------

    def checkpoint_status(self, manga_id: str, stage: str,
                          page: int | None = None) -> str:
        key = str(page) if page is not None else "*"
        row = self.conn.execute(
            """SELECT status FROM processing_checkpoints
               WHERE manga_id=? AND stage=? AND key_value=?""",
            (manga_id, stage, key),
        ).fetchone()
        return row["status"] if row else "pending"

    def checkpoint_set(self, manga_id: str, stage: str, status: str,
                       page: int | None = None, detail: str = ""):
        now = _now()
        key = str(page) if page is not None else "*"
        existing = self.conn.execute(
            """SELECT id FROM processing_checkpoints
               WHERE manga_id=? AND stage=? AND key_value=?""",
            (manga_id, stage, key),
        ).fetchone()
        if existing is None:
            self.conn.execute(
                """INSERT INTO processing_checkpoints(
                    manga_id, stage, key_value, status, detail,
                    started_at, completed_at)
                VALUES (?,?,?,?,?,?,?)""",
                (manga_id, stage, key, status, detail,
                 now if status == "running" else None,
                 now if status == "completed" else None),
            )
        else:
            sets = ["status = ?", "detail = ?"]
            vals = [status, detail]
            if status == "running":
                sets.append("started_at = ?")
                vals.append(now)
            elif status in ("completed", "failed"):
                sets.append("completed_at = ?")
                vals.append(now)
            vals.append(existing["id"])
            self.conn.execute(
                f"UPDATE processing_checkpoints SET {', '.join(sets)} WHERE id=?",
                vals,
            )
        self.conn.commit()

    def get_checkpoints(self, manga_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM processing_checkpoints WHERE manga_id=? ORDER BY id",
            (manga_id,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # -------------------------------------------------------------------
    # STATS (for dashboard)
    # -------------------------------------------------------------------

    def stats(self, manga_id: str) -> dict:
        """Compact counts for the Memory Explorer dashboard."""
        def _count(table):
            row = self.conn.execute(
                f"SELECT COUNT(*) as n FROM {table} WHERE manga_id=?",
                (manga_id,),
            ).fetchone()
            return row["n"] if row else 0
        return {
            "manga": 1 if self.get_manga(manga_id) else 0,
            "chapters": _count("chapters"),
            "characters": _count("characters"),
            "locations": _count("locations"),
            "organizations": _count("organizations"),
            "abilities": _count("abilities"),
            "events": _count("events"),
            "summaries": _count("summaries"),
            "relationships": _count("relationships"),
            "timeline": _count("timeline"),
            "terminology": _count("terminology"),
            "conflicts": len(self.get_unresolved_conflicts(manga_id)),
            "source_evidence": _count("source_evidence"),
            "research_cache": _count("research_cache"),
        }

    # -------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------

    def _row_to_dict(self, row) -> dict:
        if row is None:
            return {}
        d = dict(row)
        # Deserialize JSON fields
        for key in ("alt_titles", "genres", "aliases", "alt_names",
                     "abilities", "members", "characters", "important_events",
                     "new_characters", "unresolved", "connections"):
            if key in d and isinstance(d[key], str):
                d[key] = _json_loads(d[key]) or ([] if key != "visual_traits" else {})
        for key in ("visual_traits", "extra", "result_json"):
            if key in d and isinstance(d[key], str):
                d[key] = _json_loads(d[key]) or {}
        return d

    def wipe_manga(self, manga_id: str):
        """Delete ALL data for a manga.  Dangerous — used by Clear All."""
        for table in (
            "timeline", "terminology", "source_evidence", "conflicts",
            "research_cache", "processing_checkpoints", "summaries",
            "events", "abilities", "organizations", "relationships",
            "character_aliases", "characters", "locations", "chapters", "manga",
        ):
            self.conn.execute(f"DELETE FROM {table} WHERE manga_id=?", (manga_id,))
        self.conn.commit()


# ---------------------------------------------------------------------------
# Convenience openers
# ---------------------------------------------------------------------------

def open_knowledge_db(state_dir: Path | str) -> KnowledgeDB:
    """Open the knowledge database at state_dir/manga_knowledge.db."""
    return KnowledgeDB.open(Path(state_dir))
