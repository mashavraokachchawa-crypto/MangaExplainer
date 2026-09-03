"""Character memory — first-class character records with alias resolution.

Built on top of :class:`pipeline.manga_memory.store.MemoryStore` with
character-specific helpers: alias lookup, appearance counting, and name
canonicalization so later pages match a previously-canonical name.
"""
from __future__ import annotations

from .models import MemoryRecord, VerificationState
from .store import MemoryStore

ALIAS_SEP = "|"


class CharacterMemory:
    def __init__(self, store: MemoryStore):
        self.store = store
        if self.store.kind != "character":
            raise ValueError("CharacterMemory needs the 'character' store")

    # ------------------------------------------------------------- canonical
    def canonical_name(self, name: str) -> str | None:
        """Return the stored canonical name for any alias (or None)."""
        if not name:
            return None
        for key, rec in self.store.items():
            seen = {key} | set(_aliases(rec))
            if name in seen:
                return key
        return None

    def record(self, key: str) -> MemoryRecord | None:
        return self.store.get(key)

    @staticmethod
    def _aliases_of(rec: MemoryRecord) -> set[str]:
        return _aliases(rec)

    # ---------------------------------------------------------------- absorb
    def learn(
        self,
        name: str,
        *,
        source: str = "",
        page: int | None = None,
        description: str | None = None,
        role: str | None = None,
        aliases: list[str] | None = None,
        confidence: float = 1.0,
    ) -> MemoryRecord:
        """Learn (or reinforce) a character from one panel observation."""
        if not name:
            return None
        rec = self.store.get(name)
        if rec is None:
            rec = MemoryRecord(
                kind="character",
                key=name,
                value={"name": name, "descriptions": [description] if description else [],
                       "roles": []},
                source=source,
                page=page,
                confidence=confidence,
                state=VerificationState.AUTO,
                extra={"aliases": []},
            )
            self.store.records[name] = rec
        # reinforce / enrich
        value = rec.value if isinstance(rec.value, dict) else {"name": name}
        descs = value.setdefault("descriptions", [])
        if description and description not in descs:
            descs.append(description)
        if role and role not in value.get("roles", []):
            value.setdefault("roles", []).append(role)
        value["name"] = name
        value.pop("description", None)
        rec.value = value
        rec.touch(page=page)
        # merge aliases
        current = set(_aliases(rec))
        for alias in aliases or []:
            alias = alias.strip()
            if alias and alias != name:
                current.add(alias)
        rec.extra["aliases"] = sorted(current)
        return rec

    def appearances(self, key: str) -> int:
        rec = self.store.get(key)
        return rec.seen_count if rec else 0

    def by_role(self, role: str) -> list[MemoryRecord]:
        out = []
        for rec in self.store.all():
            value = rec.value or {}
            if role in value.get("roles", []):
                out.append(rec)
        return out

    def all(self) -> list[MemoryRecord]:
        return self.store.all()


def _aliases(rec: MemoryRecord):
    try:
        extra = rec.extra or {}
        raw = extra.get("aliases") or []
        if raw and isinstance(raw, str):
            raw = raw.split(ALIAS_SEP)
        return [str(a).strip() for a in raw if str(a).strip()]
    except Exception:
        return []


def make_character_store(base_store: MemoryStore) -> CharacterMemory:
    return CharacterMemory(base_store)
