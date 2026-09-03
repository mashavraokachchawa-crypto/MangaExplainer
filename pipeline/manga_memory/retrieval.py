"""Lazy, task-aware retrieval from the Manga Memory Engine.

Unlike a full text index, retrieval here is deliberately simple and cheap:
the memory corpus is small (a single manga volume), so we can afford linear
scans. What matters is *which* records are relevant to the current task and
current page, ranked by a blend of:

  - relevance keyword match (task ``kind`` provides the vocabulary)
  - effective confidence + verification state
  - recency (page proximity / last seen)

All reads are optional and never raise.
"""
from __future__ import annotations

import keyword
import re
import unicodedata
from typing import Iterable

from .confidence import effective_confidence
from .models import MemoryRecord

STOP = {
    "the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "for",
    "with", "by", "from", "is", "are", "was", "this", "that", "it",
}

_TASK_VOCAB = {
    "narration": ["character", "place", "object", "event", "name", "location"],
    "understanding": ["character", "description", "role", "object", "place"],
    "script": ["thread", "event", "character", "location", "relation"],
    "crop": ["character", "object", "place", "focus"],
    "summary": ["thread", "event", "character", "location"],
}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    return re.sub(r"[^a-z0-9\u4e00-\u9fff\u3040-\u30ff]+", " ", text)


def _tokens(text: str) -> set[str]:
    return {t for t in _norm(text).split() if t and t not in STOP}


def _text_of(rec: MemoryRecord) -> str:
    value = rec.value
    if isinstance(value, dict):
        return " ".join(str(v) for v in value.values())
    return str(value)


def _relevance(rec: MemoryRecord, vocab: set[str]) -> float:
    if not vocab:
        return 0.0
    hay = _tokens(_text_of(rec)) | _tokens(rec.key)
    hits = len(vocab & hay)
    return hits / max(1, len(vocab))


class MemoryRetriever:
    def __init__(self, memory, page: int | None = None, task: str = "narration"):
        self.memory = memory
        self.page = page
        self.task = task
        self._cache_vocab = None

    def _vocab(self) -> set[str]:
        if self._cache_vocab is None:
            vocab = set(_TASK_VOCAB.get(self.task, ["character", "event"]))
            for kind in ("character", "world", "story", "correction", "book"):
                for rec in self.memory.store_for(kind).all():
                    vocab |= _tokens(_text_of(rec))
            self._cache_vocab = vocab
        return self._cache_vocab

    # ------------------------------------------------------------- scoring
    def _score(self, rec: MemoryRecord) -> float:
        rel = _relevance(rec, self._vocab())
        conf = effective_confidence(rec)
        page_boost = self._page_boost(rec)
        return rel * 0.5 + conf * 0.35 + page_boost * 0.15

    def _page_boost(self, rec: MemoryRecord) -> float:
        if self.page is None or rec.page is None:
            return 0.0
        distance = abs(self.page - rec.page)
        return max(0.0, 1.0 - distance / 30.0)

    # ------------------------------------------------------------- retrieval
    def retrieve(
        self,
        kind: str | None = None,
        *,
        limit: int = 8,
        min_confidence: float = 0.0,
        extra_text: str | None = None,
    ) -> list[MemoryRecord]:
        """Best records across (optionally one) kind, ranked and capped."""
        kinds = [kind] if kind else ("character", "world", "story", "correction", "book")
        pool: list[MemoryRecord] = []
        context_vocab = self._vocab()
        if extra_text:
            context_vocab |= _tokens(extra_text)
            self._cache_vocab = context_vocab
        seen = set()
        for k in kinds:
            for rec in self.memory.store_for(k).all():
                if rec.key in seen:
                    continue
                seen.add(rec.key)
                if effective_confidence(rec) < min_confidence:
                    continue
                pool.append(rec)
        pool.sort(key=self._score, reverse=True)
        return pool[:limit]

    def retrieve_keyword(self, text: str, kind: str | None = None, limit: int = 5) -> list[MemoryRecord]:
        """Find records whose text matches keywords from ``text``."""
        wanted = _tokens(text)
        if not wanted:
            return []
        kinds = [kind] if kind else ("character", "world", "story", "correction", "book")
        results = []
        for k in kinds:
            for rec in self.memory.store_for(k).all():
                if wanted & _tokens(_text_of(rec)):
                    results.append(rec)
        results.sort(key=lambda r: len(wanted & _tokens(_text_of(r))), reverse=True)
        return results[:limit]

    def by_character(self, name: str, limit: int = 6) -> list[MemoryRecord]:
        """Records (events, facts) mentioning this character alias."""
        from .character import CharacterMemory

        canonical = CharacterMemory(self.memory.store_for("character")).canonical_name(name) or name
        wanted = {canonical, name}
        wanted_norm = {_norm(w) for w in wanted if w}
        out = []
        for k in ("story", "world", "correction"):
            for rec in self.memory.store_for(k).all():
                if wanted_norm & _tokens(_text_of(rec)):
                    out.append(rec)
        out.sort(key=lambda r: (r.page is not None, r.page or 0), reverse=True)
        return out[:limit]
