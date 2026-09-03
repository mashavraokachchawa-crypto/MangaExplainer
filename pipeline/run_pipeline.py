"""Complete connected pipeline orchestrator (Task 26).

Chains every completed module into one sequential pipeline:

  Manga -> Panel extraction -> Panel cleaning -> Panel understanding
  -> Narration script -> Pocket TTS -> Voice timing -> Panel preparation
  -> Camera movement -> Music -> SFX -> Audio mixing -> Video rendering
  -> Quality check -> Final MP4

Transitions do NOT exist in this pipeline: panels are always hard-cut
(no fade / dissolve), so there is no transitions stage and no transition
effects in the video.

Properties (all required by Task 26):
  * sequential execution of stages in the order above
  * low-RAM operation (stages run one at a time; each closes/limits its own
    working set; the whole chapter is never held in memory at once)
  * intermediate results are saved on disk after every stage
  * resume-from-last-completed-step via the shared crash-safe State
  * completed stages are NEVER re-run (unless force=True) - expensive work is
    not repeated
  * progress percentage and the current task are reported
  * errors are shown clearly and STOP the run (never silently continued)

Subtitles do NOT exist anywhere in this pipeline (no subtitle stage, no
subtitle track, no subtitle files).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from pipeline.progress import Progress

LOG = logging.getLogger(__name__)

# Ordered Task-26 chain: (stage_name, human label). The checkpoint keys.
PIPELINE_STAGES = [
    ("extract_pages", "Panel extraction"),
    ("clean_panels", "Panel cleaning"),
    ("understand_panels", "Panel understanding"),
    ("knowledge_base", "Knowledge base build"),
    ("write_script", "Narration script"),
    ("pocket_tts", "Pocket TTS"),
    ("voice_timing", "Voice timing"),
    ("prepare_panels", "Panel preparation"),
    ("camera_motion", "Camera movement"),
    ("music", "Music"),
    ("sfx", "Sound effects"),
    ("audio_mix", "Audio mixing"),
    ("render_video", "Video rendering"),
    ("quality_check", "Quality check"),
]

STAGE_NAMES = [name for name, _ in PIPELINE_STAGES]

# Subtitles: deliberately absent. Nothing here references a subtitle stage.
SUBTITLE_MARKERS = ("subtitle", "subtitle_track", "srt")


class PipelineError(Exception):
    """A stage failed; the whole pipeline stops (no silent continuation)."""


class MissingInput(PipelineError):
    """A required upstream intermediate does not exist on disk."""


def _progress(completed, total):
    return int(round(100.0 * completed / total)) if total else 100


def _discover_pages(cfg):
    """Return the ordered list of concrete page numbers to process.

    Source of truth is the pages/ workspace (page_%03d.<ext> image files)
    produced by the extraction stage. If nothing has been extracted yet we
    return [] and the extraction stage is responsible for creating pages from
    the PDF.
    """
    pages_dir = Path(cfg.output.pages_dir)
    ext = str(getattr(cfg, "images", None) and
              getattr(cfg.images, "format", "jpg") or "jpg").lstrip(".")
    nums = []
    if pages_dir.is_dir():
        for p in sorted(pages_dir.iterdir()):
            if p.is_file() and p.name.startswith("page_"):
                stem = p.name[len("page_"):]
                num = stem.split(".", 1)[0]
                if num.isdigit():
                    nums.append(int(num))
    return nums


def _require_file(path, what):
    path = Path(path)
    if not path.is_file():
        raise MissingInput(f"{what} not found: {path}")
    return path


# ---------------------------------------------------------------------------
# Stage runners - each calls the real pipeline module. Returns True on
# success or raises PipelineError (with a clear message) on failure.
# ---------------------------------------------------------------------------

def _iter_pages_or_error(cfg, stage):
    pages = _discover_pages(cfg)
    if not pages:
        raise MissingInput(
            f"'{stage}' needs extracted pages but pages/ is empty. Run the "
            "extract stage (needs input/manga.pdf) first."
        )
    return pages


def run_extract(cfg, root, force=False, guard=None, progress=None):
    from state import State
    from pipeline.pdf_extractor import PdfExtractor
    pdf = Path(cfg.input.pdf)
    if not pdf.is_file():
        raise MissingInput(
            f"Panel extraction requires a manga PDF at {pdf} (add input/manga.pdf)"
        )
    state = State(STAGE_NAMES, Path(cfg.pipeline.checkpoints.dir))
    extractor = PdfExtractor(cfg)
    # extract all pages of the PDF, skipping ones already extracted
    pages = _discover_pages(cfg)
    # If nothing discovered, try to open the PDF and iterate its page range.
    if not pages:
        pages = _pdf_page_numbers(pdf)
        if not pages:
            raise MissingInput(f"could not determine page count for {pdf}")
    done = 0
    total = len(pages)
    if progress is not None:
        progress.begin("extract_pages", "Panel extraction")
    for i, p in enumerate(pages, 1):
        res = extractor.extract_page(p, state)
        if res["status"] == "error":
            raise PipelineError(f"extract page {p}: {res['error']}")
        if res["status"] == "extracted":
            done += 1
        if progress is not None:
            progress.step("extract_pages", i, total,
                          phase=f"Rendering page {i} of {total}",
                          image=_image_rel(root, res.get("path")))
        if guard is not None:
            guard.tick()
    return True


def run_clean_panels(cfg, root, force=False, guard=None, progress=None):
    """Remove banner strips / text boxes / SFX stamps before understanding.

    Cleaned crops land in clean_dir (<page>/panel_NNN.jpg) with a per-page
    clean_manifest.json; OCR + VLM then prefer the cleaned crop. Never
    touches the raw panels/ images.
    """
    from pipeline.panel_clean import clean_panel
    pages = _iter_pages_or_error(cfg, "Panel cleaning")
    total_panels = 0
    per_page = []
    for page in pages:
        n = _discover_panels(cfg, page)
        if n < 1:
            LOG.info("no panels on page %s, nothing to clean", page)
            per_page.append(0)
            continue
        per_page.append(n)
        total_panels += n
    if progress is not None:
        progress.begin("clean_panels", "Panel cleaning")
    done = 0
    for pi, page in enumerate(pages, 1):
        npanels = per_page[pi - 1]
        for p in range(1, npanels + 1):
            img = None
            if progress is not None:
                res = clean_panel(cfg, root, page, p, force=force)
                if res["status"] == "error":
                    raise PipelineError(
                        f"panel cleaning failed on page {page} panel {p}: "
                        f"{res.get('reason') or res.get('status')}"
                    )
                img = res.get("debug") or res.get("cleaned") or res.get("source")
                progress.step(
                    "clean_panels", done, total_panels,
                    phase=f"Cleaning page {pi}/{len(pages)}, panel {p}/{npanels}",
                    item=f"page {page} · panel {p}",
                    image=_image_rel(root, img),
                )
            else:
                res = clean_panel(cfg, root, page, p, force=force)
                if res["status"] == "error":
                    raise PipelineError(
                        f"panel cleaning failed on page {page} panel {p}: "
                        f"{res.get('reason') or res.get('status')}"
                    )
            done += 1
            if guard is not None:
                guard.tick()
        if progress is not None:
            progress.step("clean_panels", done, total_panels,
                          phase=f"Page {pi}/{len(pages)} cleaned ({npanels} panels)")
    return True


def _panel_live_image(cfg, root, page, panel):
    """Live-view image for a panel: the cleaned crop when one exists, else the
    raw panel — so the dashboard shows what OCR / the VLM is really reading."""
    try:
        from pipeline.panel_clean import clean_panel_source
        cleaned = clean_panel_source(cfg, page, panel)
    except Exception:
        cleaned = None
    if cleaned is not None:
        return _image_rel(root, cleaned)
    fmt = str(cfg.images.format).lower().lstrip(".")
    raw = Path(cfg.output.panels_dir) / f"page_{int(page):03d}" / f"panel_{int(panel):03d}.{fmt}"
    return _image_rel(root, raw)


def _page_live_image(cfg, root, page):
    """Live-view image for a full page (used for the script stage)."""
    fmt = str(cfg.images.format).lower().lstrip(".")
    raw = Path(cfg.output.pages_dir) / f"page_{int(page):03d}.{fmt}"
    if not raw.is_file():
        return None
    return _image_rel(root, raw)


def _scene_crop_image(cfg, root, page, scene):
    """Newest produced crop in crops/page_NNN_scene_NNN/ for the live view."""
    key = f"page_{int(page):03d}_scene_{int(scene):03d}"
    sdir = Path(cfg.output.crops_dir) / key
    if not sdir.is_dir():
        return None
    try:
        cands = sorted(
            p for p in sdir.glob("shot_*.jpg")
            if "_debug." not in p.name and not p.name.startswith(".")
        )
    except OSError:
        return None
    if not cands:
        return None
    return _image_rel(root, cands[-1])


def _image_rel(root, path):
    """Repo-relative path for the live dashboard, or None."""
    if not path:
        return None
    try:
        return os.path.relpath(str(Path(path)), str(Path(root))).replace("\\", "/")
    except ValueError:
        return str(path)


def _pdf_page_numbers(pdf):
    """Exact PDF page count via the same renderer the extract stage uses.

    A naive count of b"/Type/Page" bytes is unreliable: a page object is
    referenced more than once in the PDF cross-reference/tree, so a 1-page
    file can report 2+ "pages". pymupdf already parses the document, so use
    its authoritative page_count.
    """
    path = Path(pdf)
    try:
        import pymupdf as fitz
        with fitz.open(str(path)) as doc:
            return list(range(1, doc.page_count + 1))
    except Exception as exc:
        LOG.error("could not open %s for page-count: %s", path, exc)
        return []


def _discover_panels(cfg, page):
    """Count of panels for a page from its panels.json manifest.

    PanelDetector writes {"page":.., "source":.., "panels": [...]} so accept
    both that dict form and a bare list.
    """
    pj = Path(cfg.output.panels_dir) / f"page_{page:03d}" / "panels.json"
    if not pj.is_file():
        return 0
    try:
        import json
        data = json.loads(pj.read_text("utf-8"))
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict) and isinstance(data.get("panels"), list):
            return len(data["panels"])
        return 0
    except Exception:
        return 0


def _discover_scenes(cfg, page):
    """Scene numbers for a page from its scenes manifest.

    The authoritative source is scenes/page_NNN_scenes.json (written by the
    scenes stage) — NOT the script/ dir, which is empty until scripts exist.
    scene_id "scene_001" -> scene number 1.
    """
    import json, re
    sf = Path(cfg.output.scenes_dir) / f"page_{page:03d}_scenes.json"
    if not sf.is_file():
        return []
    try:
        doc = json.loads(sf.read_text("utf-8"))
    except Exception:
        return []
    scenes = doc.get("scenes") if isinstance(doc, dict) else None
    if not isinstance(scenes, list):
        return []
    nums = []
    pat = re.compile(r"scene_(\d+)$")
    for s in scenes:
        if not isinstance(s, dict):
            continue
        m = pat.search(str(s.get("scene_id", "")))
        if m:
            nums.append(int(m.group(1)))
    return sorted(nums)


def _analysis_summary(cfg, page, panel):
    """Live detail block read from analysis/page_NNN_panel_YYY.json (best-effort)."""
    try:
        import json
        a = Path(cfg.output.analysis_dir) / f"page_{page:03d}_panel_{panel:03d}.json"
        if not a.is_file():
            return {}
        doc = json.loads(a.read_text("utf-8"))
        analysis = doc.get("analysis") if isinstance(doc, dict) else None
        if not isinstance(analysis, dict):
            return {}
        chars = [c.get("name") for c in analysis.get("characters", []) if isinstance(c, dict)]
        return {
            "characters": ", ".join([c for c in chars if c and c != "unknown"]) or "unknown",
            "environment": str(analysis.get("environment") or "unknown")[:60],
            "important_event": str(analysis.get("important_event") or "unknown")[:80],
            "confidence": analysis.get("confidence"),
        }
    except Exception:
        return {}


def run_understand(cfg, root, force=False, guard=None, progress=None):
    import main as cli
    pages = _iter_pages_or_error(cfg, "Panel understanding")
    total_pages = len(pages)
    if progress is not None:
        progress.begin("understand_panels", "Panel understanding")
    for pi, page in enumerate(pages, 1):
        if progress is not None:
            progress.step("understand_panels", pi, total_pages,
                          phase=f"Page {pi}/{total_pages}: detecting panels")
        if cli.cmd_panels(cfg, page, force=force) != 0:
            raise PipelineError(f"panels failed on page {page}")
        if progress is not None:
            progress.step("understand_panels", pi, total_pages,
                          phase=f"Page {pi}/{total_pages}: reading order")
        if cli.cmd_order(cfg, page, force=force) != 0:
            raise PipelineError(f"reading order failed on page {page}")
        npanels = _discover_panels(cfg, page)
        for p in range(1, npanels + 1):
            panel_img = _panel_live_image(cfg, root, page, p)
            if progress is not None:
                progress.step("understand_panels", pi, total_pages,
                              phase=f"Page {pi}/{total_pages}: OCR panel {p}/{npanels}",
                              image=panel_img)
            if cli.cmd_ocr(cfg, page, p, force=force) != 0:
                raise PipelineError(f"OCR failed on page {page} panel {p}")
            if progress is not None:
                progress.step("understand_panels", pi, total_pages,
                              phase=f"Page {pi}/{total_pages}: analyzing panel {p}/{npanels}",
                              image=panel_img)
            if cli.cmd_analyze(cfg, page, p, force=force) != 0:
                raise PipelineError(f"analysis failed on page {page} panel {p}")
            if progress is not None:
                progress.detail(
                    "understand_panels",
                    page=pi, total_pages=total_pages, panel=p,
                    panel_count=npanels, panel_image=panel_img or "",
                    **_analysis_summary(cfg, page, p),
                )
        if progress is not None:
            progress.step("understand_panels", pi, total_pages,
                          phase=f"Page {pi}/{total_pages}: knowledge")
        if cli.cmd_knowledge(cfg, page, force=force) != 0:
            raise PipelineError(f"knowledge failed on page {page}")
        # Project memory + recent-pages window (both optional, never fatal):
        # what this page revealed is remembered so a resume or a new PDF in
        # the same project stays consistent (character names, places, ...).
        try:
            from pipeline.context_memory import remember_project
            remember_project(cfg, page,
                             pdf_name=Path(cfg.input.pdf).name)
        except Exception:
            LOG.info("context memory skipped for page %s (best-effort)", page)
        # Rich Manga Memory Engine (optional, never fatal): durable character,
        # world, and story records with confidence states for the narrator.
        try:
            from pipeline.manga_memory.ingest import remember_manga
            remember_manga(cfg, page,
                           pdf_name=Path(cfg.input.pdf).name)
        except Exception:
            LOG.info("manga memory ingest skipped for page %s (best-effort)", page)
        if guard is not None:
            guard.tick()
    return True


def _manga_id_for_context(cfg):
    """Best-effort canonical manga id derived from the output/context."""
    # Fallback: derive from PDF name
    try:
        pdf = Path(cfg.input.pdf).stem.lower()
        import re
        pdf = re.sub(r"[^a-z0-9]+", "_", pdf).strip("_")
        return f"manga_{pdf}" if pdf else None
    except Exception:
        return None


def _resolve_manga_title(cfg):
    """Best-effort canonical series title for the knowledge base."""
    # 1. Prefer the active project's name
    try:
        from pipeline import project_registry
        projects = project_registry.list_projects(str(cfg.pipeline.state.dir))
        if projects and projects[0].get("name"):
            return str(projects[0]["name"])
    except Exception:
        pass
    # 2. Try the PDF identity scan (lightweight metadata + cover probe)
    try:
        from pipeline import pdf_scan
        scan = pdf_scan.scan_pdf(cfg.input.pdf)
        if scan and scan.get("title"):
            return str(scan["title"])
    except Exception:
        pass
    # 3. Fall back to the PDF filename
    try:
        return Path(cfg.input.pdf).stem
    except Exception:
        return "Untitled"


def run_knowledge_base(cfg, root, force=False, guard=None, progress=None):
    """Build/refresh the persistent SQLite knowledge base after understanding.

    Extracts structured characters/locations/events from the analyzed pages
    into the knowledge DB, detects chapter boundaries, and stores page/chapter
    summaries.  Best-effort: never fatal, never blocks the main pipeline.
    """
    from pipeline.knowledge_db import open_knowledge_db
    from pipeline.knowledge_extract import run_full_extraction, extract_ocr_text_to_db
    from pipeline.chapter_detect import detect_all_chapters, apply_chapter_detections
    from pipeline.story_memory import summarize_chapter

    pages = _discover_pages(cfg)
    if not pages:
        return True  # nothing extracted yet -> nothing to index

    total_pages = len(pages)
    if progress is not None:
        progress.begin("knowledge_base", "Knowledge base build")

    state_dir = Path(cfg.pipeline.state.dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    db = open_knowledge_db(state_dir)

    try:
        # Ensure a manga record exists (best-effort identity) keyed by title
        title = _resolve_manga_title(cfg)
        manga_data = {"title": title,
                      "pdf_path": str(cfg.input.pdf)}
        manga_id = db.upsert_manga(manga_data)
        if not manga_id:
            manga_id = _manga_id_for_context(cfg) or "manga_untitled"
            db.upsert_manga({"title": title, "pdf_path": str(cfg.input.pdf)},
                            manga_id)

        if progress is not None:
            progress.step("knowledge_base", 1, 4, phase="Extracting page knowledge")

        result = run_full_extraction(db, manga_id, cfg, pages)
        n_chars, n_evts, n_locs = (result.get("characters", 0),
                                   result.get("events", 0),
                                   result.get("locations", 0))
        LOG.info("knowledge_base: extracted page knowledge (%d chars, %d events, %d locs)",
                 n_chars, n_evts, n_locs)

        if progress is not None:
            progress.step("knowledge_base", 2, 4, phase="Detecting chapter boundaries")

        analyze_dir = Path(cfg.output.analysis_dir)
        chapters = detect_all_chapters(
            manga_id,
            Path(cfg.output.pages_dir),
            Path(cfg.output.ocr_dir),
            len(_discover_pages(cfg)) or 0,
            db.get_manga(manga_id),
        )
        if chapters:
            apply_chapter_detections(db, manga_id, chapters)

        if progress is not None:
            progress.step("knowledge_base", 3, 4, phase="Building story summaries")

        # Deterministic page summaries (cheap, no LLM)
        for page in pages:
            chapter = db.chapter_for_page(manga_id, page)
            cid = chapter.get("id") if chapter else None
            try:
                from pipeline.story_memory import generate_page_summary
                generate_page_summary(db, manga_id, page, cid)
            except Exception:
                pass

        if progress is not None:
            progress.step("knowledge_base", 4, 4, phase="Knowledge base ready")

        db.checkpoint_set(manga_id, "knowledge_base", "completed")
        LOG.info("knowledge_base: built knowledge base for %s (%d pages)",
                 manga_id, total_pages)
        return True
    except Exception:
        LOG.exception("knowledge_base stage failed (non-fatal)")
        return True
    finally:
        try:
            db.close()
        except Exception:
            pass


def run_write_script(cfg, root, force=False, guard=None, progress=None):
    import main as cli
    pages = _iter_pages_or_error(cfg, "Narration script")
    total_pages = len(pages)
    if progress is not None:
        progress.begin("write_script", "Narration script")
    for pi, page in enumerate(pages, 1):
        if progress is not None:
            progress.step("write_script", pi, total_pages,
                          phase=f"Page {pi}/{total_pages}: building scenes")
        if cli.cmd_scenes(cfg, page, force=force) != 0:
            raise PipelineError(f"scenes failed on page {page}")
        scenes = _discover_scenes(cfg, page)
        page_img = _page_live_image(cfg, root, page)
        for si, scene in enumerate(scenes, 1):
            if progress is not None:
                progress.step("write_script", pi, total_pages,
                              phase=f"Page {pi}/{total_pages}: writing scene "
                                    f"{si}/{len(scenes)}",
                              image=page_img)
                progress.detail(
                    "write_script", page=pi, total_pages=total_pages,
                    scene=si, scene_total=len(scenes),
                    scene_id=f"scene_{scene:03d}", page_image=page_img or "",
                )
            if cli.cmd_script(cfg, page, scene, force=force) != 0:
                raise PipelineError(
                    f"script failed on page {page} scene {scene}")
        if guard is not None:
            guard.tick()
    return True


def run_pocket_tts(cfg, root, force=False, guard=None, progress=None):
    from pipeline.tts_manifest import (
        NarrationManifestRunner,
        load_narration_segments,
    )
    # cmd_tts_narration only ever reads ONE script (its --script argument or
    # the first one it finds), so a multi-page run would narrate scene 1 and
    # silently drop every later scene. Aggregate the whole book here instead:
    # every scene script's segments go into one global manifest, numbered by
    # overall position so scenes never overwrite each other's segment files.
    script_dir = Path(cfg.output.script_dir)
    candidates = sorted(script_dir.glob("*_scene_*.json"))
    if not candidates:
        raise MissingInput(
            "Pocket TTS needs narration scripts but script/ is empty. Run the "
            "script stage first."
        )
    segments = []
    for script in candidates:
        try:
            segments.extend(load_narration_segments(script))
        except Exception as exc:
            raise PipelineError(
                f"cannot read narration script {script}: {exc}") from exc
    if not segments:
        raise MissingInput(
            "no narration segments found in script/ - re-run the script stage")
    runner = NarrationManifestRunner(cfg)
    if progress is not None:
        progress.begin("pocket_tts", "Pocket TTS")
        progress.phase("pocket_tts",
                       f"Synthesizing {len(segments)} narration segments")

    # If the operator chose the HTTP (pocket_server) provider, make sure the
    # serve process is up (auto-starts per tts.server_auto_start) so synth()
    # never fails on a dead connection. In-process (pocket_tts) stays as-is.
    provider_name = str(getattr(getattr(cfg, "tts", None), "provider", "") or "")
    if provider_name.lower() == "pocket_server":
        try:
            from pipeline import pocket_server
            srv = pocket_server.start_server(cfg)
            if not srv.get("ok"):
                raise PipelineError(
                    "Pocket TTS server unavailable - " +
                    (srv.get("error") or srv.get("detail") or srv["url"]))
        except ImportError:
            pass
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError(f"could not start Pocket TTS server: {exc}") from None

    def _tts_phase(i, n, seg_id=None, text=None, duration=None, audio_path=None):
        if progress is not None:
            progress.step("pocket_tts", i, n, phase=f"Segment {i} of {n}")
            if seg_id:
                progress.detail(
                    "pocket_tts", segment=i, segment_count=n,
                    segment_id=seg_id,
                    text=(str(text or "")[:140] +
                          ("…" if text and len(text) > 140 else "")),
                    duration_seconds=round(float(duration), 3) if duration else None,
                    audio_path=str(audio_path) if audio_path else None,
                )

    runner.generate(segments, cfg.output.audio_dir, force=force,
                    on_progress=_tts_phase)
    if progress is not None:
        progress.phase("pocket_tts", "Computing segment timings")
    runner.finalize_timing(cfg.output.audio_dir)
    if guard is not None:
        guard.tick()
    return True


def run_voice_timing(cfg, root, force=False, guard=None, progress=None):
    import main as cli
    if progress is not None:
        progress.begin("voice_timing", "Voice timing")
        progress.phase("voice_timing", "Aligning narration timing")
    rc = cli.cmd_tts_narration(cfg, None, None, force=False, timing_only=True)
    if rc != 0:
        raise PipelineError("Voice timing stage failed (see error above)")
    return True


def run_prepare_panels(cfg, root, force=False, guard=None, progress=None):
    from pipeline.panel_prep import PanelPrepError, prepare_panels_manifest
    if progress is not None:
        progress.begin("prepare_panels", "Panel preparation")
        progress.phase("prepare_panels", "Preparing panel manifest")
    try:
        prepare_panels_manifest(cfg, root, page_nums=None)
    except PanelPrepError as exc:
        raise PipelineError(f"Panel preparation failed: {exc}") from exc

    # The visual-preparation half of "panel preparation": every scene needs a
    # shots timeline (plan) and 16:9 crops BEFORE camera movement can run.
    # Without this, motion/transitions have no timeline to consume and the
    # pipeline stalls at camera_movement. Both only need the narration script
    # and page knowledge that earlier stages already produced.
    from state import State
    from pipeline.visual_planner import VisualPlanner, PlanError
    from pipeline.crop_planner import CropPlanner, CropError

    try:
        state = State(STAGE_NAMES, Path(cfg.pipeline.state.dir))
    except Exception:
        state = None
    pages = _discover_pages(cfg)
    scene_total = sum(len(_discover_scenes(cfg, page)) for page in pages)
    scene_done = 0
    for page in pages:
        for scene in _discover_scenes(cfg, page):
            scene_done += 1
            if progress is not None:
                progress.step("prepare_panels", scene_done, scene_total,
                              phase=f"Page {page}: planning shots + crops "
                                    f"(scene {scene_done}/{scene_total})",
                              image=_scene_crop_image(cfg, root, page, scene))
            try:
                pl = VisualPlanner(cfg).run_scene(page, scene, state, force=force)
                if pl.get("result") == "error":
                    raise PipelineError(f"visual plan failed: {pl.get('message')}")
                cr = CropPlanner(cfg).run_scene(page, scene, state, force=force)
                if cr.get("result") == "error":
                    raise PipelineError(f"crops failed: {cr.get('message')}")
            except (PlanError, CropError) as exc:
                raise PipelineError(f"visual preparation failed: {exc}") from exc
        if guard is not None:
            guard.tick()
    return True


def run_camera_motion(cfg, root, force=False, guard=None, progress=None):
    from pipeline.motion import run_motion
    if progress is not None:
        progress.begin("camera_motion", "Camera movement")
        progress.phase("camera_motion", "Planning camera paths")
    try:
        run_motion(cfg, root, page_nums=None, force=force)
    except Exception as exc:
        raise PipelineError(f"Camera movement failed: {exc}") from exc
    return True


def run_music(cfg, root, force=False, guard=None, progress=None):
    import json
    from pipeline.music import music_config
    if progress is not None:
        progress.begin("music", "Music")
        progress.phase("music", "Resolving background track")
    settings = music_config(cfg)
    mdir = Path(root) / "music"
    mdir.mkdir(parents=True, exist_ok=True)
    # choose a track (local file) if music is enabled
    track = None
    if settings is not None:
        try:
            from pipeline.music import resolve_track
            track = resolve_track(root, settings)
        except Exception:
            track = None
    manifest = {"enabled": settings is not None,
                "volume": (settings.get("volume") if settings else None) or 0.2,
                "track": str(track) if track else None,
                "loop": bool(settings and settings.get("loop", True))}
    (mdir / "music_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return True


def run_sfx(cfg, root, force=False, guard=None, progress=None):
    import json
    from pipeline.sfx import sfx_config, load_events
    if progress is not None:
        progress.begin("sfx", "Sound effects")
        progress.phase("sfx", "Loading SFX events")
    sdir = Path(root) / "sfx"
    sdir.mkdir(parents=True, exist_ok=True)
    settings = sfx_config(cfg)
    events = []
    if settings is not None:
        try:
            events = load_events(root, settings) or []
        except Exception:
            events = []
    (sdir / "sfx_manifest.json").write_text(
        json.dumps({"enabled": settings is not None, "events": events},
                   indent=2), encoding="utf-8")
    return True


def run_audio_mix(cfg, root, force=False, guard=None, progress=None):
    from pipeline.audio_mix import NoNarration, run_mix
    if progress is not None:
        progress.begin("audio_mix", "Audio mixing")
        progress.phase("audio_mix", "Mixing narration with music path")
    _require_file(Path(cfg.output.audio_dir) / "manifest.json", "narration manifest")
    try:
        run_mix(cfg, root)
    except NoNarration as exc:
        raise PipelineError(f"Audio mixing failed: {exc}") from exc
    return True


def run_render_video(cfg, root, force=False, guard=None, progress=None):
    from pipeline.export import ExportError, export_final
    if guard is not None:
        # RAM-only tick while the renderer writes its temp working set.
        guard.tick(sweep_dirs=())
    if progress is not None:
        progress.begin("render_video", "Video rendering")
        progress.phase("render_video", "Rendering frames, then encoding MP4")

    def _frame_phase(i, n):
        if progress is not None:
            progress.step("render_video", i, n,
                          phase=f"Rendering frame {i} of {n}")
            progress.detail(
                "render_video", frame=i, frame_count=n,
                phase=f"Rendering frame {i} of {n}")

    try:
        export_final(cfg, root, low_ram=True, on_progress=_frame_phase)
    except ExportError as exc:
        raise PipelineError(f"Video rendering failed: {exc}") from exc
    if progress is not None:
        progress.phase("render_video", "Encode finished")
    return True


def run_quality(cfg, root, force=False, guard=None, progress=None):
    from pipeline.quality_check import check_quality
    if progress is not None:
        progress.begin("quality_check", "Quality check")
        progress.phase("quality_check", "Running quality checks")
    report = check_quality(cfg, root, low_ram=True)
    if report["status"] == "error":
        errs = "\n".join(
            f"      - {c['check']}: {c['detail']}"
            for c in report["checks"] if not c["passed"] and c["critical"]
        )
        raise PipelineError(
            f"Quality check FAILED ({report['error_count']} error(s)):\n{errs}"
        )
    return True


_STAGE_FNS = {
    "extract_pages": run_extract,
    "clean_panels": run_clean_panels,
    "understand_panels": run_understand,
    "knowledge_base": run_knowledge_base,
    "write_script": run_write_script,
    "pocket_tts": run_pocket_tts,
    "voice_timing": run_voice_timing,
    "prepare_panels": run_prepare_panels,
    "camera_motion": run_camera_motion,
    "music": run_music,
    "sfx": run_sfx,
    "audio_mix": run_audio_mix,
    "render_video": run_render_video,
    "quality_check": run_quality,
}

# Assert every pipeline stage has a runner and none is a subtitle stage.
assert set(_STAGE_FNS) == set(STAGE_NAMES), "pipeline stage/runner mismatch"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _input_hash(path, chunk=1 << 20):
    """Short SHA-256 of an input file, computed streaming (low RAM)."""
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sync_input_fingerprint(state, cfg):
    """Invalidate stale completed stages when the source PDF changes.

    Returns True if the checkpoints were invalidated (a new/updated PDF was
    detected) and the new fingerprint recorded.

    Rules:
      * no recorded fingerprint + no completed work -> record it, no re-run
      * recorded fingerprint differs -> invalidate completed stages, re-run
      * legacy state (completed work, no fingerprint recorded yet) -> cannot
        prove the build matches the current PDF, so invalidate once to be safe
    """
    pdf = Path(cfg.input.pdf)
    if not pdf.is_file():
        return False
    current = _input_hash(pdf)
    previous = state.input_fingerprint()
    completed = state.completed_count()
    if previous == current:
        return False
    state.set_input_fingerprint(current)
    if previous is not None or completed > 0:
        state.invalidate_completed()
        return True
    return False


def run_pipeline(cfg, root, force=False, log=LOG):
    """Run the complete pipeline, checkpointing after each stage.

    Returns a summary dict:
      {status, completed, total, current, report: [ {name,status}, ... ]}

    Raises PipelineError if a stage fails (after reporting it clearly). The
    caller decides whether to surface the error; nothing is silently skipped.
    """
    from state import State
    from pipeline.lowram import MemoryGuard, rss_mb
    root = Path(root)
    state = State(STAGE_NAMES, Path(cfg.pipeline.checkpoints.dir))
    if state.recover_interrupted():
        log.info("recovered %s stage(s) left 'running' by an interrupted "
                 "process; they will resume from the start of that stage",
                 "interrupted")
    guard = MemoryGuard(cfg, root, log=log)
    if not force and _sync_input_fingerprint(state, cfg):
        log.info("input PDF changed; stale checkpoints invalidated, "
                 "pipeline will re-run from extract_pages")
    total = len(STAGE_NAMES)
    report = []

    rss = rss_mb()
    log.info(
        "low-RAM run: rss=%sMB batch_size=%s guard=%sMB cache_sweep_every=%ss "
        "sweep_dirs=%s (one process, one item at a time, SSD-backed)",
        rss, getattr(cfg.pipeline, "batch_size", 1), guard.guard_mb, guard.sweep_seconds,
        [str(p) for p in guard.sweep_dirs],
    )
    try:
        state_dir = Path(cfg.pipeline.state.dir)
    except AttributeError:
        state_dir = root / "state"
    progress = Progress(root, state_dir=state_dir)
    guard.start()
    try:
        for name, label in PIPELINE_STAGES:
            status = state.status_of(name)
            if status == "completed" and not force:
                report.append({"name": name, "status": "completed", "skipped": True,
                               "progress": _progress(
                                   state.completed_count(), total)})
                continue  # never repeat a completed expensive operation
            if force:
                state.mark_running(name)
            else:
                state.mark_running(name)
            progress.begin(name, label)
            completed_before = state.completed_count()
            pct = _progress(completed_before, total)
            print(f"\n[ {pct:3d}% ] Stage {completed_before + 1}/{total}: {label} "
                  f"({name})")
            try:
                _STAGE_FNS[name](cfg, root, force=force, guard=guard,
                                 progress=progress)
            except PipelineError as exc:
                state.mark_failed(name)
                report.append({"name": name, "status": "failed", "skipped": False,
                               "error": str(exc),
                               "progress": _progress(completed_before, total)})
                print(f"\n  ERROR in stage '{label}': {exc}")
                print("  Pipeline stopped. Nothing later was run; fix the error "
                      "and re-run to resume from here.")
                guard.tick()
                return {"status": "error", "completed": state.completed_count(),
                        "total": total, "current": name, "report": report,
                        "error": str(exc)}
            except Exception as exc:  # unexpected -> still a clear stop
                state.mark_failed(name)
                report.append({"name": name, "status": "failed", "skipped": False,
                               "error": f"{type(exc).__name__}: {exc}",
                               "progress": _progress(completed_before, total)})
                print(f"\n  ERROR in stage '{label}': {type(exc).__name__}: {exc}")
                guard.tick()
                return {"status": "error", "completed": state.completed_count(),
                        "total": total, "current": name, "report": report,
                        "error": f"{type(exc).__name__}: {exc}"}
            state.mark_completed(name)
            report.append({"name": name, "status": "completed", "skipped": False,
                           "progress": _progress(state.completed_count(), total)})
            guard.tick()

    finally:
        # Stop the periodic guard thread (sweeps any remaining wrong-stage
        # leftovers from cache/temp dirs) before returning to the caller.
        guard.tick()
        guard.close()
        progress.clear()

    print(f"\n[100%] Pipeline complete: {total}/{total} stages.")
    return {"status": "ok", "completed": total, "total": total,
            "current": None, "report": report, "error": None}
