"""Automatic book reference lookup: fetch the manga's important facts from the internet.

The reader types the manga's name once; this module resolves it and returns the
canonical reference the app stores as durable memory. Sources, in order:

1. **MangaDex** (api.mangadex.org) - the manga index of record: localised title,
   authors, genres (tags), demographic (shonen/seinen/...), status (ongoing/
   completed/...), publication year, synopsis, and cover art.
2. **English Wikipedia** (search + REST summary) - text fallback when MangaDex
   has no match.

Everything is read-only, timeout-bounded, and **never raises**: on any failure
``fetch_book_ref`` returns ``None`` so the UI can tell the reader to check the
spelling or their internet connection. Uses only the standard library
(``urllib``), matching the rest of the pipeline.

Returned dict (all keys optional):

    {"source": "mangadex"|"wikipedia", "title": str, "authors": [str],
     "genres": [str], "demographic": str, "status": str, "year": str,
     "language": str, "synopsis": str, "url": str, "cover_url": str|None}
"""
from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from collections import Counter
from typing import Optional

LOG = logging.getLogger("mangaexplainer.internet_ref")

USER_AGENT = "MangaExplainer/1.0 (book-reference fetcher; local tool)"
MANGA_DEX = "https://api.mangadex.org"
WIKI_API = "https://en.wikipedia.org/w/api.php"


def fetch_book_ref(title: str, timeout: float = 12.0) -> Optional[dict]:
    """Resolve ``title`` to a canonical book reference, or None on failure."""
    title = str(title or "").strip()
    if not title:
        return None
    for fetcher in (_mangadex, _wikipedia):
        try:
            info = fetcher(title, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - a fetcher must never raise
            LOG.debug("book ref '%s': %s failed: %s", title, fetcher.__name__, exc)
            info = None
        if info:
            return _with_characters(info)
    return None


def _with_characters(info: dict) -> dict:
    """Attach a best-effort character-name list extracted from the fetched text.

    MangaDex's search endpoint does not expose structured cast lists, so we
    derive character names from the synopsis / Wikipedia extract with a simple,
    never-raising heuristic: pick capitalized person-like noun phrases that
    recur. Weaker than a real cast, but it gives the narrator correct names to
    stay consistent with and populates the CHARACTER memory group.
    """
    if not isinstance(info, dict):
        return info
    corpus = "\n".join(_ for _ in (
        info.get("synopsis"), info.get("title"), "") if _)
    names = _extract_names(corpus)
    if names:
        info["characters"] = names
    return info


# words that are almost never a person's name (stops the extractor),
# plus words to treat as the *first* token of a person name.
_ROLE_HINT = {"artists", "artist", "illustrator", "author", "story", "artist is",
              "created by", "written by", "illustrated by", "by"}
_PERSON_HINT = {"who", "she", "he", "him", "her", "himself", "herself",
                "is", "was", "and", "are", "his"}


def _extract_names(text: str, limit: int = 15) -> list[str]:
    """Pull recurring capitalized person-like names from running prose."""
    text = str(text or "")
    # strip parentheticals + quotes so "(a)"/"\"\"\" don't pollute tokens
    text = re.sub(r"[()\[\]'\"\u2018\u2019\u201c\u201d]", " ", text)
    if len(text) < 20:
        return []

    tokens = re.findall(r"\b[A-Z][a-zA-Z]+\b", text)
    # person names are almost always exactly one or two capitalized words, and
    # those preceded by a role hint ("by", "illustrated by") are the author, not
    # a character — drop them so "by Eiichiro Oda" doesn't list the author.
    freq: dict[str, int] = {}
    for i, tok in enumerate(tokens):
        prev = (tokens[i - 1].lower() if i else "")
        if prev in {r.lower() for r in _ROLE_HINT}:
            continue
        freq[tok] = freq.get(tok, 0) + 1

    # a real character is usually mentioned more than once; single mentions are
    # usually scene objects or Japanese place glosses.
    ranked = sorted(
        ((c, t) for t, c in freq.items()),
        reverse=True,
    )
    names = []
    seen_lower = set()
    for count, token in ranked:
        low = token.lower()
        if low in seen_lower:
            continue
        if count < 2:
            continue
        names.append(token)
        seen_lower.add(low)
        if len(names) >= limit:
            break
    return names


def book_ref_to_text(info: dict) -> str:
    """One compact human line describing a fetched reference."""
    if not info:
        return ""
    bits = []
    if info.get("year"):
        bits.append(str(info["year"]))
    if info.get("status"):
        bits.append(str(info["status"]))
    head = " («" + " · ".join(bits) + "»)" if bits else ""
    authors = ", ".join(info.get("authors") or []) or "author unknown"
    genres = ", ".join(info.get("genres") or [])
    return f"{info.get('title') or '?'}{head} — by {authors}" + (
        f" [{genres}]" if genres else ""
    )


# --------------------------------------------------------------------------
# Character cast extraction (dedicated "List of X characters" pages)
# --------------------------------------------------------------------------


# Level-3 headings on a character-list page are individuals UNLESS they contain
# deeper member subsections (teams / orgs / titles / locations). This tiny set
# mops up org groupings that happen to have no member subsections of their own.
_GROUP_NO_MEMBERS = {
    "sound four", "kara", "divine trees", "tailed beasts", "ototsuki clan",
}

# level-2 headings that mark the Naruto-style "cast groupings" layout. Skip
# structured parsing on pages whose L2 headings are factions/locations (One
# Piece) and fall through to the wikilink cast fallback instead.
_CAST_GROUP_L2 = {
    "main characters", "major characters", "heroes", "protagonists",
    "secondary characters", "supporting characters", "antagonists",
    "villains", "cast", "characters", "overview",
}


_FILE_LINK_RE = re.compile(
    r"\[\[(?:file|image)\s*:\s*([^\]|\n]+?)(?:\|[^\]]*)?\]\]", re.IGNORECASE)


def _portrait_from_block(body: str) -> str | None:
    """First character-specific image filename in a section's wikitext.

    Ignores site furniture / group / cast-roster images so we never present a
    shared picture as a specific character's portrait.
    """
    for m in _FILE_LINK_RE.finditer(body or ""):
        fname = m.group(1).strip()
        low = fname.replace("_", " ").lower()
        if any(k in low for k in ("icon", "logo", "symbol", "map",
                                  "sprite", "group", "cast", "crop", "cast roster")):
            continue
        return fname
    return None


def fetch_characters_with_portraits(
    title: str, timeout: float = 12.0
) -> tuple[list[dict], dict[str, str]]:
    """One parse of the character-list page -> ``(cast, portraits)``.

    Fetches the "List of <title> characters" wikitext **once** and extracts both
    the cast (name + role, as :func:`fetch_characters`) and any per-character
    portrait filename from the same character's own section. Reusing one page
    keeps Wikipedia calls low (we had 429s fetching per character) and
    guarantees a portrait belongs to the *correct* character — never a bare-name
    global lookup that can surface an unrelated real person.

    ``portraits`` maps character name -> image filename (absent when that
    character has no portrait on the page). Returns ``([], {})`` on any failure.
    """
    if not str(title or "").strip():
        return [], {}
    try:
        list_title = _find_character_list(title, timeout)
    except Exception:  # noqa: BLE001 - network/rate-limit failure
        return [], {}
    if not list_title:
        return [], {}
    try:
        data = _get_json(
            (f"{WIKI_API}?action=parse&page={urllib.parse.quote(list_title)}"
             "&format=json&prop=wikitext"), timeout=timeout)
    except Exception:  # noqa: BLE001
        return [], {}
    if not data:
        return [], {}
    body = ((data.get("parse") or {}).get("wikitext") or {}).get("*") or ""
    if not body:
        return [], {}

    blocks = _split_blocks(body)
    grouped = any(lvl == 2 and _clean_heading(h).lower().strip(" ") in _CAST_GROUP_L2
                  for lvl, h, _ in blocks)
    cast: list[dict] = []
    portraits: dict[str, str] = {}
    if grouped:
        seen: set[str] = set()
        for i in range(len(blocks)):
            lvl, htext, hbody = blocks[i]
            if lvl != 3:
                continue
            name = _clean_heading(htext)
            if not name or len(name) < 2 or name.lower() in seen or \
                    name.lower() in _GROUP_NO_MEMBERS:
                continue
            j = i + 1
            while j < len(blocks) and blocks[j][0] > 3:
                j += 1
            if j > i + 1:
                continue
            role = _first_sentence(hbody, name)
            cast.append({"name": name, "role": role})
            seen.add(name.lower())
        cast = _dedup_cast(cast)
        for lvl, htext, hbody in blocks:
            if lvl != 3:
                continue
            name = _clean_heading(htext)
            if not name:
                continue
            fname = _portrait_from_block(hbody)
            if fname:
                portraits[name] = fname
    else:
        # faction/TOC-style pages: fall back to the wikilink roster (names only)
        cast = [{"name": n, "role": ""} for n in _wikilink_cast_body(body)]
    if len(cast) < 2 and not portraits:
        return [], portraits
    return cast, portraits


def fetch_characters(title: str, timeout: float = 12.0) -> list[dict]:
    """Best-effort real character cast (name + role) for a manga title.

    Prefers the dedicated Wikipedia "List of <title> characters" page, where
    individual characters are level-3 headings and each section's first sentence
    is a natural role/summary (the Naruto-style layout). For faction/TOC-style
    pages (e.g. One Piece) where the cast is buried under deep locations, falls
    back to frequency-ranked ``[[Name]]`` wikilinks from the same page, then to
    recurring names in the fetched book synopsis. Each fallback is weaker but
    still yields real character names. Never raises.
    """
    title = str(title or "").strip()
    if not title:
        return []

    list_title = _find_character_list(title, timeout)
    if list_title:
        members = _cast_from_sections(list_title, timeout)
        if len(members) >= 2:
            return members
        # list page exists but structure didn't parse -> wikilink fallback
        try:
            names = _wikilink_cast(list_title, timeout)
        except Exception:  # noqa: BLE001 - never raise
            names = []
        if len(names) >= 2:
            return [{"name": n, "role": ""} for n in names]

    # last resort: scrape recurring names off the fetched book reference
    try:
        info = fetch_book_ref(title, timeout=timeout)
    except Exception:  # noqa: BLE001 - never raise
        info = None
    corpus = ((info or {}).get("synopsis") or "") + " " + ((info or {}).get("title") or "")
    names = _extract_names(corpus, limit=15)
    return [{"name": n, "role": ""} for n in names]


def scope_cast_to_volume(cast: list[dict], volume: Optional[int]) -> list[dict]:
    """Keep only the characters plausibly present in a given collected-volume PDF.

    The Wikipedia "List of <manga> characters" pages write characters in roughly
    chronological appearance order (the founding/opening cast first, later-arc
    characters far down the page, e.g. Berserk lists Guts/Griffith/Casca/Judeau/
    the Band of the Hawk before Farnese/Serpico/Isidro/Schierke who are from the
    Conviction/Falcon arcs). So a compiled-volume edition (the app's usual input,
    e.g. "Berserk v01") should not be annotated with the whole series' roster.

    There is no reliable per-character "debuts in volume N" datum, so this uses a
    documented **early-block heuristic**: keep the first ``9`` characters (a
    typical opening cast) plus ``4`` more for each volume beyond the first
    (widening into early-arc casts), clamped to the full roster. ``volume`` of 0/
    None keeps the entire cast untouched. Never raises; best-effort by design.
    """
    if not cast:
        return list(cast)
    n = int(volume or 0)
    if n < 1:
        return list(cast)
    limit = min(len(cast), 9 + (n - 1) * 4)
    if limit >= len(cast):
        return list(cast)
    return list(cast[:limit])


def _wikilink_cast(list_title: str, timeout: float) -> list[str]:
    """Ranked ``[[Name]]`` wikilinks from the character page (cast fallback)."""
    data = _get_json(
        f"{WIKI_API}?action=parse&page={urllib.parse.quote(list_title)}"
        "&format=json&prop=wikitext",
        timeout=timeout,
    )
    body = (data.get("parse") or {}).get("wikitext") or {}
    body = body.get("*") or ""
    if not body:
        return []
    return _wikilink_cast_body(body)


def _wikilink_cast_body(body: str) -> list[str]:
    """Ranked ``[[Name]]`` wikilinks from character-page wikitext."""
    if not body:
        return []
    # drop the tail sections (reception / references / video-game / easter-egg)
    for pat in ("\n==Reception", "\n==References", "\n==Notes", "\n==Video game",
                "\n==Easter Egg", "\n==External links", "\n==Footer"):
        i = body.find(pat)
        if i != -1:
            body = body[:i]

    freq: Counter = Counter()
    for m in re.finditer(r"\[\[([^\[\]]+)\]\]", body):
        raw = m.group(1)
        if ":" in raw or raw.lower().startswith("file"):
            continue
        target = (raw.split("|")[0] or "").strip()
        label = (raw.split("|")[-1] if "|" in raw else target) or target
        label = label.strip()
        if "(" in target or not target:
            continue  # disambiguation / empty target
        if re.search(
            r"\bvoic(?:ed|e)?\b|dubb?ed?\b|voice actress|dub (?:part|cast|x)?|"
            r"(?:English|Japanese) (?:dub|voice|ver)|provided by|"
            r"(?:television|radio|company|uhder|anime series)",
            body[max(0, m.start() - 70):m.start() + 90], re.I,
        ):
            continue
        if len(label) < 3 or label.lower() in ("one piece", "naruto", "manga", "anime"):
            continue
        freq[label] += 1

    # keep multi-word names and frequent tokens; thin out obvious media/orgs
    stop = {"the", "to", "of", "in", "and", "for", "an", "a", "on",
            "japan", "japanese", "english", "world", "series", "island",
            "mermen", "straw hats", "straw hat", "anime", "manga"}
    media = ("netflix", "hollywood", "television", "tv", "production",
             "company", "media", "news wire", "hewlett", "buzzfeed")
    out = []
    for label, c in freq.most_common(40):
        if c < 2:
            continue  # prefer recurring names, but single mentions are still cast
        low = label.lower()
        if low in stop or len(label.split()) > 4:
            continue
        if any(tok in low for tok in media):
            continue
        out.append(label)
    return out


def _find_character_list(title: str, timeout: float) -> str:
    """Search for the dedicated character-list page; returns its title or ''."""
    for probe in (
        f"List of {title} characters",
        f"{title} characters",
    ):
        data = _get_json(
            f"{WIKI_API}?action=query&format=json&list=search&srsearch="
            f"{urllib.parse.quote(probe)}&srlimit=3&srnamespace=0",
            timeout=timeout,
        )
        hits = ((data or {}).get("query") or {}).get("search") or []
        for hit in hits:
            name = str(hit.get("title") or "").strip()
            # prefer a page that name-matches <title>"; avoid spinoff/series pages
            if _norm_title(name) == _norm_title(probe) or (
                "characters" in _norm_title(name) and title.lower() in name.lower()
            ):
                return name
    return ""


def _cast_from_sections(list_title: str, timeout: float) -> list[dict]:
    """Extract 'character heading -> first-sentence role' from a list page.

    On these pages the *groupings* are level-2 headings (Main / Secondary /
    Antagonists / Supporting / Other characters) whose children are level-3
    headings. A level-3 heading is an individual character unless it *has*
    deeper (level-4+) member subsections — those are teams/orgs/titles/locations
    (e.g. 'Akatsuki', 'Team 8', 'Hokage', 'Sunagakure'). A tiny denylist mops up
    org headings that lack member subsections ('Sound Four', 'Kara', 'Tailed
    Beasts').

    Fetches the SECTIONS index (to judge the leaf structure) and the FULL
    wikitext in a single batch via ``prop=sections|wikitext`` — never one call
    per character, to stay within Wikipedia rate limits.
    """
    url = (f"{WIKI_API}?action=parse&page={urllib.parse.quote(list_title)}"
           "&format=json&prop=wikitext")
    data = _get_json(url, timeout=timeout)
    if not data:
        return []
    body = ((data.get("parse") or {}).get("wikitext") or {}).get("*") or ""
    if not body:
        return []

    # split the wikitext into (level, heading, body) blocks, preserving order
    blocks = _split_blocks(body)

    # only run the structured pass when this is a Naruto-style "cast-grouping"
    # page (a level-2 heading names a cast group). On faction/location TOC pages
    # (One Piece) the level-3 headings are orgs/locations, not people.
    if not any(lvl == 2 and _clean_heading(h).lower().strip(" ") in _CAST_GROUP_L2
               for lvl, h, _ in blocks):
        return []

    # level-3 headings are individual characters unless they contain deeper
    # (level-4+) subsections; skip those group/org/title/location headings.
    seen: set[str] = set()
    cast: list[dict] = []
    for i in range(len(blocks)):
        lvl, htext, hbody = blocks[i]
        if lvl != 3:
            continue
        name = _clean_heading(htext)
        if name.lower() in seen or not name or len(name) < 2:
            continue
        if name.lower() in _GROUP_NO_MEMBERS:
            continue
        # does this block contain a deeper (lvl>=4) block before the next lvl<=3?
        j = i + 1
        while j < len(blocks) and blocks[j][0] > 3:
            j += 1
        if j > i + 1:
            continue  # has member subsections -> team/org/title/location
        role = _first_sentence(hbody, name)
        cast.append({"name": name, "role": role})
        seen.add(name.lower())
    return _dedup_cast(cast)


_HEADING_LINE_RE = re.compile(r"^\s*(={2,6})\s*(.*?)\s*\1\s*$")


def _split_blocks(body: str) -> list[tuple]:
    """Split wikitext into (level, heading_text, block_body) tuples in order."""
    blocks: list[tuple] = []
    pending = None
    buf: list[str] = []
    for line in body.splitlines():
        m = _HEADING_LINE_RE.match(line)
        if m:
            if pending is not None:
                blocks.append((pending[0], pending[1], "\n".join(buf)))
            pending = (len(m.group(1)), m.group(2).strip())
            buf = []  # body starts after the heading line
            continue
        buf.append(line)
    if pending is not None:
        blocks.append((pending[0], pending[1], "\n".join(buf)))
    return blocks


def _clean_heading(line: str) -> str:
    """Strip ''italics'' / <i> wrappers from a section heading -> plain name."""
    text = re.sub(r"<[^>]+>", "", line)
    text = text.replace("''", "").replace("[[", " ").replace("]]", " ")
    text = re.sub(r"\[\[([^|\]]*\|)?([^\]]*)\]\]", r"\2", text)  # links -> label
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip()


def _first_sentence(block: str, name: str) -> str:
    """First descriptive sentence of a character block, pruned of boilerplate."""
    block = re.sub(r"\{\{[^{}]*\}\}", " ", block)          # {{templates}}
    block = re.sub(r"<ref[^>]*>.*?</ref>", " ", block, flags=re.S)
    block = re.sub(r"<[^>]+>", " ", block)
    block = re.sub(r"\[\[([^|\]]*\|)?([^\]]*)\]\]", r"\2", block)  # wikilinks -> label
    block = block.replace("''", "")  # ''italic'' markers
    block = re.sub(r"[\[\]\"\u2018\u2019\u201c\u201d]", " ", block)
    block = re.sub(r"\s+", " ", block).strip()

    for s in re.split(r"(?<=[.!?])\s+", block):
        s = s.strip()
        if not len(s) > 5:
            continue
        # skip boilerplate like voice-actor credits mid-list
        if re.search(r"\bvoiced by\b", s, re.I):
            continue
        front = s
        if name:
            low = name.lower()
            for lead in (name, name.split()[0] if " " in name else ""):
                if lead and front.lower().startswith(lead.lower()):
                    front = front[len(lead):].lstrip(",;: ")
            m = re.match(r"^(?:is|are|was|been|being)\s+", front, re.I)
            if m:
                front = front[m.end():]
        return front[:160]
    return ""


def _clean_heading(line: str) -> str:
    """Strip ''italics'' / <i> wrappers from a section heading -> plain name."""
    text = re.sub(r"<[^>]+>", "", line)
    text = text.replace("''", "").replace("[[", " ").replace("]]", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip()


def _dedup_cast(cast: list[dict]) -> list[dict]:
    """Remove captialised duplicates differing only by case, keep the longer."""
    by_lower: dict[str, dict] = {}
    for c in cast:
        low = c["name"].lower()
        prev = by_lower.get(low)
        if prev is None or len(c["name"]) > len(prev["name"]):
            by_lower[low] = c
    return list(by_lower.values())


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------


def _get_json(url: str, timeout: float) -> Optional[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _loc(value) -> str:
    """Localised string or plain string -> an English/preferred plain string."""
    if not value:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        if value.get("en"):
            return str(value["en"]).strip()
        for _k, _v in value.items():
            if isinstance(_v, str) and _v.strip():
                return _v.strip()
    return ""


# --------------------------------------------------------------------------
# MangaDex
# --------------------------------------------------------------------------


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _mangadex(title: str, timeout: float) -> Optional[dict]:
    query = urllib.parse.urlencode([
        ("title", title),
        ("limit", "8"),
        ("includes[]", "author"),
        ("includes[]", "cover_art"),
    ])
    data = _get_json(f"{MANGA_DEX}/manga?{query}", timeout=timeout)
    if not data:
        return None
    results = data.get("data") or []
    if not results or not isinstance(results[0], dict):
        return None

    # Prefer an exact title match, then a prefix match, then the top hit:
    # the first hit is often a spinoff ("One Piece Academy" vs "One Piece").
    wanted = _norm_title(title)

    def _name(entry):
        attrs = entry.get("attributes") or {}
        return _loc(attrs.get("title")) or _loc(attrs.get("altTitle")) or title

    entry = None
    for cand in results:
        if _norm_title(_name(cand)) == wanted:
            entry = cand
            break
    if entry is None:
        for cand in results:
            if _norm_title(_name(cand)).startswith(wanted):
                entry = cand
                break
    if entry is None:
        entry = results[0]

    entry_id = str(entry.get("id") or "")
    attributes = entry.get("attributes") or {}
    name = _name(entry)
    if not name or name.lower() in ("n/a", "unknown", "?", ""):
        return None

    authors: list[str] = []
    genres: list[str] = []
    cover = None
    for rel in entry.get("relationships") or []:
        rtype = rel.get("type")
        rattrs = rel.get("attributes") or {}
        if rtype == "author" and _loc(rattrs.get("name")):
            authors.append(_loc(rattrs.get("name")))
        elif rtype == "cover_art" and (rattrs.get("fileName")):
            cover = f"https://uploads.mangadex.org/covers/{entry_id}/{rattrs['fileName']}.256.jpg"
    for tag in attributes.get("tags") or []:
        g = _loc(((tag.get("attributes") or {}).get("name")))
        if g:
            genres.append(g)

    synopsis = _loc(attributes.get("description"))
    return {
        "source": "mangadex",
        "title": name,
        "authors": authors,
        "genres": genres,
        "demographic": _loc(attributes.get("publicationDemographic")),
        "status": str(attributes.get("status") or "").strip(),
        "year": str(attributes.get("year") or "").strip(),
        "language": str(attributes.get("originalLanguage") or "").strip(),
        "synopsis": (synopsis or "")[:800],
        "url": f"https://mangadex.org/title/{entry_id}" if entry_id else "",
        "cover_url": cover,
    }


# --------------------------------------------------------------------------
# Wikipedia fallback
# --------------------------------------------------------------------------


def _wikipedia(title: str, timeout: float) -> Optional[dict]:
    search = _get_json(
        f"{WIKI_API}?action=query&format=json&list=search&srsearch="
        f"{urllib.parse.quote(title)}&srlimit=1&srnamespace=0",
        timeout=timeout,
    )
    found = None
    via_search = search and search.get("query") and search["query"].get("search")
    if via_search:
        for hit in via_search:
            if str(hit.get("title") or "").strip():
                found = str(hit["title"]).strip()
                break
    if not found:
        return None
    summary = _get_json(
        "https://en.wikipedia.org/api/rest_v1/page/summary/"
        + urllib.parse.quote(found.replace(" ", "_")),
        timeout=timeout,
    )
    if not summary:
        return None
    return {
        "source": "wikipedia",
        "title": _loc(summary.get("title")) or found,
        "authors": [],
        "genres": [],
        "demographic": "",
        "status": "",
        "year": "",
        "language": "",
        "synopsis": (str(summary.get("extract") or "")[:800]),
        "url": (summary.get("content_urls") or {}).get("desktop", {}).get("page")
        if isinstance(summary.get("content_urls"), dict) else "",
        "cover_url": (summary.get("thumbnail") or {}).get("source")
        if isinstance(summary.get("thumbnail"), dict) else None,
    }