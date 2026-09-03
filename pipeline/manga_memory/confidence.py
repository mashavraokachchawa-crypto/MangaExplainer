"""Confidence & verification helpers for memory records.

A record's effective confidence is the product of its stated confidence and a
factor derived from its verification state — a user correction or a VERIFIED
record always outranks a plain AUTO one at equal raw confidence.
"""
from __future__ import annotations

from .models import MemoryRecord, VerificationState

_STATE_MULTIPLIER = {
    VerificationState.USER_CORRECTED: 1.0,
    VerificationState.VERIFIED: 0.98,
    VerificationState.AUTO: 0.85,
    VerificationState.UNCERTAIN: 0.5,
    VerificationState.CONFLICTED: 0.35,
}


def effective_confidence(rec: MemoryRecord) -> float:
    base = max(0.0, min(1.0, rec.confidence))
    mult = _STATE_MULTIPLIER.get(rec.state, 0.8)
    return base * mult


def is_conflicted(rec: MemoryRecord) -> bool:
    return rec.state == VerificationState.CONFLICTED


def is_uncertain(rec: MemoryRecord) -> bool:
    return rec.state == VerificationState.UNCERTAIN


def is_user(rec: MemoryRecord) -> bool:
    return rec.state == VerificationState.USER_CORRECTED
