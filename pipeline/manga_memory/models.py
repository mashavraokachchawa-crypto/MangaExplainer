"""Data models for Manga Memory Engine records.

These are plain mutable objects serialized to/from JSON. They are deliberately
lightweight (no ORM, no pydantic) to match the rest of the application's
pure-JSON-on-disk storage.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


# ------------------------------------------------------------------- states


class VerificationState:
    """Confidence states a memory record can be in.

    AUTO             — written by the pipeline (VLM/OCR) without review
    VERIFIED         — confirmed by the pipeline/reviewer
    USER_CORRECTED   — a user click corrected or overrode an auto value
    UNCERTAIN        — low-confidence, needs review
    CONFLICTED       — two sources disagreed; awaiting a resolution
    """

    AUTO = "auto"
    VERIFIED = "verified"
    USER_CORRECTED = "user_corrected"
    UNCERTAIN = "uncertain"
    CONFLICTED = "conflicted"

    # Order matters: a record's effective state is the "strongest" present.
    _RANK = {
        USER_CORRECTED: 4,
        VERIFIED: 3,
        CONFLICTED: 2,
        UNCERTAIN: 1,
        AUTO: 0,
    }

    @classmethod
    def strongest(cls, states):
        ranked = [(cls._RANK.get(s, -1), s) for s in states]
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return ranked[0][1] if ranked else cls.AUTO


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------- record


class MemoryRecord:
    """One memory entry: a fact about the manga's world.

    ``source`` is free-form provenance, e.g. ``analysis/page_001_panel_003``
    or ``user``. ``extra`` holds type-specific fields.
    """

    def __init__(
        self,
        *,
        kind: str,
        key: str,
        value: Any = None,
        source: str = "",
        page: int | None = None,
        confidence: float = 1.0,
        state: str = VerificationState.AUTO,
        extra: dict | None = None,
    ):
        self.kind = kind
        self.key = key
        self.value = value
        self.source = source
        self.page = page
        self.confidence = float(confidence)
        self.state = state
        self.extra = extra or {}
        self.created_at = utcnow()
        self.updated_at = self.created_at
        self.seen_count = 1

    # -- hints ---------------------------------------------------------
    @property
    def is_user(self) -> bool:
        return self.state == VerificationState.USER_CORRECTED

    @property
    def is_verified(self) -> bool:
        return self.state in (
            VerificationState.VERIFIED,
            VerificationState.USER_CORRECTED,
        )

    def touch(self, page: int | None = None):
        self.updated_at = utcnow()
        self.seen_count = self.seen_count + 1
        if page is not None:
            self.page = page

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "key": self.key,
            "value": self.value,
            "source": self.source,
            "page": self.page,
            "confidence": self.confidence,
            "state": self.state,
            "extra": dict(self.extra),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "seen_count": self.seen_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryRecord":
        rec = cls(
            kind=str(data.get("kind") or ""),
            key=str(data.get("key") or ""),
            value=data.get("value"),
            source=str(data.get("source") or ""),
            page=data.get("page"),
            confidence=float(data.get("confidence") or 0.0),
            state=str(data.get("state") or VerificationState.AUTO),
            extra=data.get("extra") or {},
        )
        rec.created_at = str(data.get("created_at") or utcnow())
        rec.updated_at = str(data.get("updated_at") or rec.created_at)
        try:
            rec.seen_count = int(data.get("seen_count") or 1)
        except (TypeError, ValueError):
            rec.seen_count = 1
        return rec
