"""Character / cover imagery for a manga, fetched free with no API key.

The app's memory is strengthened by pictures: for each fetched character we
try to grab a portrait and for the manga a cover. Sources are free and
keyless:

  * Wikipedia REST summary per character -> ``thumbnail`` portrait when known
    (works for famous characters like Luffy, Guts, Naruto).
  * MangaDex cover art for the whole series (already in the book reference),
    used as the cover fallback and as a stand-in when a character has no
    portrait.

Design:
  * stdlib only (urllib), never raises -> errors collapse to ``None``.
  * SSRF-safe: http(s) + public-IP check + size cap + image Content-Type check.
  * Files stored under ``state/manga_memory/images/<slug>/`` so the webui can
    serve them through the existing /media passthrough without touching the FFI.

Module is deliberately independent of the webui (which has its own downloader)
so it can be used by CLI/test code too.
"""
from __future__ import annotations

import ipaddress
import json
import re
import socket
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "MangaExplainer/1.0 (memory image fetcher; local tool)"
WIKIPEDIA_REST = "https://en.wikipedia.org/api/rest_v1/page/summary/"
WIKI_API = "https://en.wikipedia.org/w/api.php"
MANGA_DEX = "https://api.mangadex.org"

MAX_BYTES = 4 * 1024 * 1024  # 4 MB per image


class _Blocked(ValueError):
    pass


def _slugify(title: str) -> str:
    t = re.sub(r"[^A-Za-z0-9]+", "-", (title or "").strip().lower()).strip("-")
    return t[:48] or "manga"


def _safe_download(url: str, timeout: float = 8.0) -> bytes | None:
    """Download image bytes with SSRF guards; None on any failure."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return None
    try:
        raw_ip = socket.gethostbyname(parsed.hostname)
    except OSError:
        return None
    try:
        addr = ipaddress.ip_address(raw_ip)
    except ValueError:
        return None
    if (addr.is_private or addr.is_loopback or addr.is_link_local
            or addr.is_multicast or addr.is_reserved or addr.is_unspecified):
        return None
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "image/*,application/octet-stream",
    })
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if not content_type.startswith("image/"):
                return None
            declared = int(resp.headers.get("Content-Length") or 0)
            if declared > MAX_BYTES:
                return None
            data = bytearray()
            while len(data) <= MAX_BYTES:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                data.extend(chunk)
            if not data or len(data) > MAX_BYTES:
                return None
            return bytes(data)
    except (OSError, ValueError):
        return None


def _wikipedia_image_bare(name: str, timeout: float = 8.0) -> str | None:
    """Wikipedia REST summary thumbnail URL for a bare name (unverified)."""
    query = urllib.parse.quote(name.strip())
    req = urllib.request.Request(
        WIKIPEDIA_REST + query,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    thumb = data.get("thumbnail") or {}
    src = thumb.get("source")
    return src if isinstance(src, str) else None


def _fetch_wikitext(page: str, timeout: float) -> str:
    """Fetch a page's raw wikitext (empty on failure)."""
    url = (f"{WIKI_API}?action=parse&page={urllib.parse.quote(page)}"
           "&format=json&prop=wikitext")
    try:
        with urllib.request.urlopen(
                urllib.request.Request(
                    url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}),
                timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except (OSError, ValueError):
        return ""
    if not isinstance(data, dict):
        return ""
    return ((data.get("parse") or {}).get("wikitext") or {}).get("*") or ""


# any [[File:...]] / [[Image:...]] link (thumb framings, plain links)
_FILE_LINK_RE = re.compile(
    r"\[\[(?:file|image)\s*:\s*([^\]|\n]+?)(?:\|[^\]]*)?\]\]",
    re.IGNORECASE)


def _portrait_from_block(body: str) -> str | None:
    """First non-shared image filename in a character section's wikitext."""
    for m in _FILE_LINK_RE.finditer(body or ""):
        fname = m.group(1).strip()
        # skip icon/shared/group images that are usual site furniture
        low = fname.replace("_", " ").lower()
        if any(k in low for k in ("icon", "logo", "symbol", "map",
                                  "sprite", "group", "cast", " character roster")):
            continue
        return fname
    return None


def franchise_character_image(
    title: str, name: str, timeout: float = 10.0
) -> str | None:
    """Resolve a *verified* portrait: the character's own image on the correct
    "List of <title> characters" page.

    A bare Wikipedia search can return the wrong ''real'' person or a different
    franchise, so we only trust images that appear inside that manga's character
    listing. Returns the resolved ``Special:FilePath`` URL or None when the
    character has no portrait on that page (honest: show a name card instead of
    a mismatched photo). Never raises.
    """
    from .internet_ref import _find_character_list, _split_blocks, _clean_heading
    try:
        list_title = _find_character_list(title, timeout) or ""
    except Exception:  # noqa: BLE001
        return None
    if not list_title:
        return None
    body = _fetch_wikitext(list_title, timeout)
    if not body:
        return None
    name_low = _norm_id(name)
    best = None
    for _lvl, heading, block in _split_blocks(body):
        if _norm_id(_clean_heading(heading)) != name_low:
            continue
        fname = _portrait_from_block(block)
        if fname:
            return ("https://en.wikipedia.org/wiki/Special:FilePath/"
                    + urllib.parse.quote(fname.replace(" ", "_")))
        best = _lvl  # found the section; keep track if a later dup has one
    return None


def _norm_id(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())


def fetch_character_image(
    name: str, timeout: float = 8.0, title: str | None = None
) -> bytes | None:
    """Return portrait bytes for ``name`` or None.

    Uses the franchise-scoped lookup (correct manga character only) when
    ``title`` is provided. When scoped to a title we NEVER fall back to a
    bare-name lookup: a bare name can surface an unrelated real-person photo
    (e.g. a "Rickert" from another game), and the user wants only *verified*
    portraits. Without a title (caller does not know the manga) the lenient
    bare lookup is used, and its output is untrustworthy by design.
    """
    url = None
    if title:
        try:
            url = franchise_character_image(title, name, timeout)
        except Exception:  # noqa: BLE001
            url = None
        if not url:
            return None  # verified-only: never substitute an unverified image
    else:
        url = _wikipedia_image_bare(name, timeout)
    if not url:
        return None
    return _safe_download(url, timeout)


def _mangadex_cover(title: str, timeout: float = 8.0) -> str | None:
    """Best-effort MangaDex cover URL (same exact-title logic as internet_ref)."""
    from .internet_ref import fetch_book_ref  # reuse the exact-title matcher
    try:
        ref = fetch_book_ref(title, timeout=timeout)
    except Exception:
        return None
    if not ref:
        return None
    return ref.get("cover_url")


def fetch_cover(title: str, timeout: float = 8.0) -> bytes | None:
    """Return cover bytes for ``title`` or None."""
    url = _mangadex_cover(title, timeout)
    if not url:
        return None
    return _safe_download(url, timeout)


def image_store_dir(state_dir: str | Path) -> Path:
    """Where per-project images live: ``state/manga_memory/images/<slug>``."""
    return Path(state_dir) / "manga_memory" / "images"


def save_image(path: Path, data: bytes, ext: str = ".jpg") -> Path | None:
    """Persist image bytes to ``path`` (creates parents); returns path or None."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path
    except OSError:
        return None


def ensure_images(
    title: str,
    characters: list[str],
    cover_url: str | None = None,
    state_dir: str | Path = "state",
    timeout: float = 8.0,
    portraits: dict[str, str] | None = None,
) -> dict:
    """Fetch and store images for a manga + its characters.

    ``portraits`` (optional) maps a character name -> an already-resolved image
    filename (from :func:`internet_ref.fetch_characters_with_portraits`); such a
    portrait belongs to the correct manga's character page, so we skip the
    unreliable bare-name lookup entirely. Cover falls back to ``cover_url``.
    Returns ``{slug, cover_file, characters: {name: file}}`` — file names are
    stored relative to the image store dir; missing images are simply absent
    (best-effort, never raises).
    """
    slug = _slugify(title)
    basedir = image_store_dir(state_dir) / slug
    out = {"slug": slug, "cover_file": None, "characters": {}}

    if cover_url:
        data = _safe_download(cover_url, timeout)
        if data:
            path = save_image(basedir / "cover.jpg", data, ".jpg")
            if path:
                out["cover_file"] = path.name

    for name in characters:
        name = str(name or "").strip()
        if not name:
            continue
        data = None
        fname = (portraits or {}).get(name) or (portraits or {}).get(
            name.lower())
        if fname:
            url = ("https://en.wikipedia.org/wiki/Special:FilePath/"
                   + urllib.parse.quote(str(fname).replace(" ", "_")))
            data = _safe_download(url, timeout)
        else:
            data = fetch_character_image(name, timeout, title=title)
        if not data:
            continue
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "char"
        path = save_image(basedir / f"{safe}.jpg", data, ".jpg")
        if path:
            out["characters"][name] = path.name
    return out