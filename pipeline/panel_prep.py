"""Panel preparation for video rendering (Task 16).

Collects the existing manga panel/image data and produces
visuals/panels_manifest.json - WITHOUT generating new images and WITHOUT
writing or converting any image file.

For every panel the manifest records:
    * panel_id              (the panel's global id, e.g. p001_001)
    * image                 (repository-relative path to the original file)
    * width, height         (exact pixel dimensions read from the image)
    * aspect_ratio          (width/height - the ORIGINAL ratio, never forced)
    * bbox                  (original crop/panel box where known)
    * narration_segments    (the correct narration segment(s) from the script
                             that reference this panel via their panel_ids)

Design notes:
    * "Preserve original aspect ratio" -> we store width/height and the exact
      ratio derived from them; we do not resize, pad, or choose a target AR.
    * "Avoid unnecessary image conversion" -> we only DECODE each image once
      with cv2 to read its dimensions; we never re-encode or write images.
    * Sequences only; nothing is processed in parallel (low-RAM CPU box).

Panel discovery (first source that yields data wins):
    1. per-page knowledge: analysis/page_XXX_knowledge.json
       (panel_id + image + bbox + scene_id + reading_order)
    2. panel detector:     panels/page_XXX/panels.json  (id + image + bbox)

Narration connection is a reverse index built from every narration script
(script/page_XXX_scene_YYY.json): a panel is connected to each narration
segment whose panel_ids list includes it.
"""
import json
import logging
from pathlib import Path

LOG = logging.getLogger("mangaexplainer")

PANELS_MANIFEST_NAME = "panels_manifest.json"
PANEL_ID_RE = r"^p\d{3}_\d{3}$"
VISUALS_DIR = "visuals"  # repo-relative


class PanelPrepError(Exception):
    """Base class for panel preparation errors."""


class NoPanelData(PanelPrepError):
    """No panel/image data was found to prepare."""


def visuals_manifest_dir(root):
    return Path(root) / VISUALS_DIR


def visuals_manifest_path(root):
    return visuals_manifest_dir(root) / PANELS_MANIFEST_NAME


def _resolve_path(root, path):
    p = Path(path)
    return p if p.is_absolute() else Path(root) / p


def _image_dimensions(path):
    """Width, height of an image WITHOUT re-encoding/writing it."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise PanelPrepError(f"cannot decode image: {path}")
    h, w = img.shape[:2]
    del img
    return int(w), int(h)


# --------------------------------------------------------------- narration


def _script_scripts_to_scan(cfg, root):
    script_dir = Path(getattr(cfg.output, "script_dir"))
    if script_dir.is_dir():
        return sorted(script_dir.glob("*_scene_*.json"))
    return []


def build_narration_index(cfg, root):
    """panel_id -> list of dicts(page, scene, segment_id, text)."""
    index = {}
    for script in _script_scripts_to_scan(cfg, root):
        try:
            data = json.loads(script.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        page = data.get("page")
        scene = data.get("scene_id")
        for seg in data.get("segments") or []:
            if not isinstance(seg, dict):
                continue
            segment_id = seg.get("segment_id")
            panel_ids = seg.get("panel_ids")
            if not segment_id or not panel_ids:
                continue
            for pid in panel_ids:
                index.setdefault(str(pid), []).append({
                    "page": page,
                    "scene": scene,
                    "segment_id": segment_id,
                    "text": seg.get("text", ""),
                })
    return index


# ------------------------------------------------------------ panel sources


def _from_knowledge(cfg, root, page_num):
    """Load per-page knowledge records; [] if none for this page."""
    path = _resolve_path(root, Path(cfg.output.analysis_dir)) / \
        f"page_{page_num:03d}_knowledge.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text("utf-8"))
    out = []
    for rec in data.get("panels") or []:
        out.append({
            "panel_id": rec["panel_id"],
            "page": rec["page"],
            "image": rec["image"],
            "bbox": rec.get("bbox"),
            "reading_order": rec.get("reading_order"),
            "scene_id": rec.get("scene_id"),
        })
    return out


def _from_panel_detector(cfg, root, page_num):
    """Load panels.json records from the panel-detector stage; [] if none."""
    path = _resolve_path(root, Path(cfg.output.panels_dir)) / \
        f"page_{page_num:03d}" / "panels.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text("utf-8"))
    out = []
    for rec in data.get("panels") or []:
        out.append({
            "panel_id": rec["id"],
            "page": data.get("page", page_num),
            "image": rec["image"],
            "bbox": rec.get("bbox"),
            "reading_order": None,
            "scene_id": None,
        })
    return out


def discover_panels(cfg, root, page_nums=None):
    """Ordered list of panel dicts from knowledge, then panel detector.

    Panels are deduplicated by panel_id (first source that has the record
    wins). page_nums: explicit list, or auto-discover available pages.
    """
    panels_by_id = {}
    if page_nums is None:
        page_nums = _discover_pages(cfg, root)
    for page_num in page_nums:
        records = _from_knowledge(cfg, root, page_num)
        if not records:
            records = _from_panel_detector(cfg, root, page_num)
        for rec in records:
            pid = rec["panel_id"]
            if pid not in panels_by_id:
                panels_by_id[pid] = rec
    return [panels_by_id[pid] for pid in sorted(panels_by_id)]


def _discover_pages(cfg, root):
    pages = set()
    analysis_dir = _resolve_path(root, Path(cfg.output.analysis_dir))
    if analysis_dir.is_dir():
        for p in analysis_dir.glob("page_*_knowledge.json"):
            try:
                pages.add(int(p.name.split("_")[1]))
            except (IndexError, ValueError):
                pass
    panels_dir = _resolve_path(root, Path(cfg.output.panels_dir))
    if panels_dir.is_dir():
        for d in panels_dir.iterdir():
            if d.is_dir() and (d / "panels.json").is_file():
                try:
                    pages.add(int(d.name.split("_")[-1]))
                except (IndexError, ValueError):
                    pass
    return sorted(pages)


# ------------------------------------------------------------ manifest


def prepare_panels_manifest(cfg, root, page_nums=None,
                            narration_index=None):
    """Build visuals/panels_manifest.json; returns the manifest list.

    cfg:     project Config
    root:    repository root (for relative path resolution)
    page_nums: optional explicit pages; default auto-discover
    narration_index: optional panel->segments map; default built from scripts
    """
    root = Path(root)
    panels = discover_panels(cfg, root, page_nums)
    if not panels:
        raise NoPanelData(
            "no panel/image data found (no analysis/page_*_knowledge.json nor "
            "panels/page_*/panels.json with images). Run the panels/analysis "
            "stages first, or provide panel images."
        )
    if narration_index is None:
        narration_index = build_narration_index(cfg, root)

    manifest = []
    seen = set()
    for panel in panels:
        panel_id = panel["panel_id"]
        if panel_id in seen:
            continue
        seen.add(panel_id)
        image_path = _resolve_path(root, panel["image"])
        width, height = _image_dimensions(image_path)
        entry = {
            "panel_id": panel_id,
            "page": panel["page"],
            "image": panel["image"],
            "width": width,
            "height": height,
            "aspect_ratio": round(width / height, 6) if height else None,
        }
        if panel.get("bbox") is not None:
            entry["bbox"] = panel["bbox"]
        if panel.get("reading_order") is not None:
            entry["reading_order"] = panel["reading_order"]
        segs = narration_index.get(panel_id) or []
        if segs:
            entry["narration_segments"] = segs
        manifest.append(entry)

    out_dir = visuals_manifest_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / PANELS_MANIFEST_NAME
    out_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOG.info("wrote %s (%d panels)", out_path, len(manifest))
    return manifest
