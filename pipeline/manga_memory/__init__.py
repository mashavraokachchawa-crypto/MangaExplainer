"""Manga Memory Engine — persistent, project-scoped memory for MangaExplainer.

Two-tier design mirrors :mod:`pipeline.context_memory` but adds a richer,
queryable memory layer:

1. DURABLE ENTITY MEMORY (type 1)
   Character / World / Story / User-Correction records persisted as JSON in
   ``state/manga_memory/``. Unlike the flat ``project_memory.json`` tables,
   these are first-class records with confidence states, provenance, and
   merge semantics, so later pages (or a later volume) can recall and
   reconcile what an earlier page established.

2. RECENT-PAGES WINDOW (type 2)
   Reuses ``pipeline.context_memory.PageWindow`` unchanged — the last N
   understood pages. Not rebuilt here.

The engine is strictly optional and never breaks the pipeline: every read
returns empty/safe defaults on a corrupt/missing file, every write is atomic,
and no provider error propagates into the render pipeline.
"""
