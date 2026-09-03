#!/usr/bin/env python3
"""MangaExplainer CLI - low-RAM manga explanation video pipeline skeleton.

Simple interface (Task 28):
    python main.py start          run the complete chain: Manga -> final MP4
    python main.py resume         continue from the last completed step
    python main.py status         show pipeline progress and next step
    python main.py render         render the video
    python main.py check          quality-check the output video
    python main.py clean          clear cache/derived artifacts (keeps checkpoints)

Advanced/per-stage commands:
    python main.py clean-cache    delete derived artifacts/cache (keeps checkpoints)
    python main.py ocr --page N --panel M
                                  OCR a single detected panel
    python main.py analyze --page N --panel M
                                  VLM analysis of a single detected panel
    python main.py knowledge --page N
                                  build/update the page knowledge file
    python main.py scenes --page N
                                  group the page's panels into scenes
    python main.py script --page N --scene M
                                  write narration script for one scene
    python main.py audio --page N --scene M
                                  synthesize narration audio for one scene
    python main.py plan --page N --scene M
                                  build the visual timeline for one scene
    python main.py crops --page N --scene M
                                  compute 16:9 cinematic crops for one scene
    python main.py match --page N [--all] [--force]
                                  deterministic panel <-> narration mapping
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.loader import load_config
from logger import setup_logging
from pipeline.stages import STAGES

STAGE_NAMES = [s.name for s in STAGES]


def ensure_dirs(cfg):
    paths = {
        Path(cfg.output.dir),
        Path(cfg.pipeline.state.dir),
        Path(cfg.pipeline.checkpoints.dir),
        Path(cfg.logging.log_dir),
        Path(cfg.pipeline.cache.dir),
        Path(cfg.input.pdf).parent,
        ROOT / "input",
    }
    for stage in STAGES:
        paths.add(ROOT / stage.output_dir)
    paths.add(Path(cfg.output.pages_dir))
    paths.add(Path(cfg.output.panels_dir))
    paths.add(Path(cfg.output.ocr_dir))
    paths.add(Path(cfg.output.analysis_dir))
    paths.add(Path(cfg.output.scenes_dir))
    paths.add(Path(cfg.output.script_dir))
    paths.add(Path(cfg.output.audio_dir))
    paths.add(Path(cfg.output.shots_dir))
    paths.add(Path(cfg.output.crops_dir))
    paths.add(Path(cfg.output.matching_dir))
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def cmd_status(cfg):
    ensure_dirs(cfg)
    from state import State
    from pipeline.run_pipeline import STAGE_NAMES, PIPELINE_STAGES
    state = State(STAGE_NAMES, cfg.pipeline.checkpoints.dir)
    total = len(STAGE_NAMES)
    rows = {r["name"]: r["status"] for r in state.summary()}
    print("MangaExplainer - status")
    print(f"  input PDF        : {cfg.input.pdf}")
    print(f"  output dir       : {cfg.output.dir}")
    print(f"  checkpoint file  : {state.path}")
    print(f"  progress         : {state.completed_count()}/{total} stages "
          f"({round(100 * state.completed_count() / total)}%)")
    width = max(len(name) for name in STAGE_NAMES)
    for name, label in PIPELINE_STAGES:
        print(f"    {name:<{width}}  {rows.get(name, 'pending')}  ({label})")
    nxt = state.next_pending()
    print(f"  next stage       : {nxt or '(none - complete)'}")
    return 0


def cmd_resume(cfg):
    ensure_dirs(cfg)
    setup_logging(cfg)
    from pipeline.run_pipeline import run_pipeline
    print("MangaExplainer - resume (continue from last completed step; "
          "completed work is never redone)")
    result = run_pipeline(cfg, ROOT, force=False)
    if result["status"] == "error":
        print(f"\n  ERROR: {result.get('error')}")
        return 1
    return 0


def _clean_artifacts(cfg):
    """Clear cache + stage output artifacts; preserve valid checkpoints.

    Returns (cleaned, removed_files, preserved) so callers can report.
    Task 27: data/checkpoints is never auto-cleaned, so a stopped run can
    always resume.
    """
    targets = [Path(cfg.pipeline.cache.dir)] + [ROOT / s.output_dir
                                                for s in STAGES]
    removed_files = 0
    cleaned = []
    for target in targets:
        if target.resolve() == ROOT or not target.exists():
            continue
        for path in target.rglob("*"):
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
                removed_files += 1
        for path in sorted(
            (p for p in target.rglob("*") if p.is_dir()),
            key=lambda p: len(p.parts),
            reverse=True,
        ):
            try:
                path.rmdir()
            except OSError:
                pass
        cleaned.append(str(target))
    checkpoint_dir = Path(cfg.pipeline.checkpoints.dir)
    preserved = [str(p) for p in checkpoint_dir.rglob("checkpoints.json")]
    return cleaned, removed_files, preserved


def cmd_clean_cache(cfg):
    ensure_dirs(cfg)
    log = setup_logging(cfg)
    cleaned, removed_files, preserved = _clean_artifacts(cfg)
    print("MangaExplainer - clean-cache")
    print(f"  cleared         : {', '.join(cleaned)}")
    print(f"  files deleted   : {removed_files}")
    if preserved:
        print("  checkpoints kept: " + ", ".join(preserved) +
              "  (never auto-deleted, resume-safe)")
    else:
        print("  checkpoints kept: none present yet (data/checkpoints/ preserved)")
    print("  protected       : input/, config/, state/, data/checkpoints/, "
          "logs/, pipeline/, tests/, tools/")
    log.info("clean-cache done: %d files removed (checkpoints preserved)",
             removed_files)
    return 0


def cmd_extract(cfg, page_num):
    ensure_dirs(cfg)
    setup_logging(cfg)
    from pipeline.pdf_extractor import PdfExtractor
    from state import State

    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = PdfExtractor(cfg).extract_page(page_num, state)
    print("MangaExplainer - extract")
    print(f"  page            : {result['page_num']}")
    print(f"  input PDF       : {cfg.input.pdf}")
    print(f"  target          : {result['path']}")
    if result["status"] == "error":
        print(f"  result          : ERROR - {result['error']}")
        return 1
    action = "extracted" if result["status"] == "extracted" else "skipped (already complete)"
    print(f"  result          : {action}")
    print(f"  dimensions      : {result['width']} x {result['height']} px")
    print(f"  size            : {result['size_bytes']} bytes")
    return 0


def cmd_panels(cfg, page_num, force):
    ensure_dirs(cfg)
    setup_logging(cfg)
    from pipeline.panel_detector import PanelDetector
    from state import State

    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = PanelDetector(cfg).detect_page(page_num, state, force=force)
    print("MangaExplainer - panels")
    print(f"  page            : {result['page']}")
    print(f"  source          : {result.get('source')}")
    print(f"  output dir      : {result.get('out_dir')}")
    if result["status"] == "error":
        print(f"  result          : ERROR - {result['error']}")
        return 1
    if result["status"] == "skipped":
        print(f"  result          : skipped (already detected)")
    else:
        print(f"  result          : detected {result['count']} panel(s)")
    print(f"  manifest        : {result.get('manifest')}")
    panels = result.get("panels") or []
    for panel in panels:
        x, y, w, h = panel["bbox"]
        print(
            f"    {panel['id']}  bbox=[{x},{y},{w},{h}] "
            f"conf={panel['confidence']}"
        )
    return 0


def cmd_order(cfg, page_num, force):
    ensure_dirs(cfg)
    setup_logging(cfg)
    from pipeline.reading_order import ReadingOrder
    from state import State

    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = ReadingOrder(cfg).detect_page(page_num, state, force=force)
    print("MangaExplainer - reading order")
    print(f"  page            : {result['page']}")
    print(f"  direction       : {result['direction']}")
    print(f"  output dir      : {result.get('out_dir')}")
    if result["status"] == "error":
        print(f"  result          : ERROR - {result['error']}")
        return 1
    if result["status"] == "skipped":
        print(f"  result          : skipped (already computed)")
    else:
        print(
            f"  result          : ordered {result['ordered']}/{result['count']} panel(s)"
        )
    print(f"  manifest        : {result.get('manifest')}")
    print(f"  order file      : {result.get('order_path')}")
    print(f"  debug image     : {result.get('debug_image')}")
    for rank, pid in enumerate(result["order"], 1):
        print(f"    {rank:>2}. {pid}")
    if result["ignored"]:
        print(f"  ignored (bad bbox): {', '.join(str(i) for i in result['ignored'])}")
    return 0


def cmd_ocr(cfg, page_num, panel_num, force):
    ensure_dirs(cfg)
    setup_logging(cfg)
    from pipeline.ocr_processor import OcrProcessor
    from state import State

    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = OcrProcessor(cfg).run_panel(page_num, panel_num, state, force=force)
    print("MangaExplainer - OCR")
    print(f"  page            : {result['page']}")
    print(f"  panel           : {result['panel']}")
    print(f"  engine          : {result.get('engine')}")
    print(f"  source          : {result.get('source')}")
    if result["status"] == "error":
        print(f"  result          : ERROR - {result['error']}")
        return 1
    if result["status"] == "skipped":
        print(f"  result          : skipped (already completed)")
    else:
        print(f"  result          : {result['count']} text block(s)")
    print(f"  output          : {result.get('out_file')}")
    print(f"  debug image     : {result.get('debug_image')}")
    for index, block in enumerate(result["blocks"], 1):
        x, y, w, h = block["bbox"]
        print(
            f"    {index}. [{block['type']}] conf={block['confidence']} "
            f"bbox=[{x},{y},{w},{h}] {block['text']!r}"
        )
    print(f"  combined text   : {result['combined_text']!r}")
    return 0


def cmd_analyze(cfg, page_num, panel_num, force):
    ensure_dirs(cfg)
    setup_logging(cfg)
    from pipeline.analysis_processor import AnalysisProcessor
    from state import State

    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = AnalysisProcessor(cfg).run_panel(page_num, panel_num, state, force=force)
    print("MangaExplainer - VLM analysis")
    print(f"  page            : {result['page']}")
    print(f"  panel           : {result['panel']}")
    print(f"  provider        : {result.get('provider')}")
    print(f"  model           : {result.get('model')}")
    print(f"  source          : {result.get('source')}")
    if result["result"] == "error":
        print(f"  result          : ERROR - {result['message']}")
        return 1
    if result["result"] == "skipped":
        print(f"  result          : skipped (already completed)")
        print(f"  output          : {result.get('output')}")
        return 0
    print(f"  result          : analyzed")
    print(f"  output          : {result.get('output')}")
    print(f"  confidence      : {result.get('confidence')}")
    print(f"  characters      : {result.get('characters')}")
    print(f"  important event : {result.get('important_event')!r}")
    return 0


def cmd_knowledge(cfg, page_num, force):
    ensure_dirs(cfg)
    setup_logging(cfg)
    from pipeline.knowledge import KnowledgeBuilder, load_index
    from state import State

    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = KnowledgeBuilder(cfg).build_page(page_num, state, force=force)
    print("MangaExplainer - knowledge")
    print(f"  page            : {result['page']}")
    print(f"  result          : {result['result']}")
    if result["result"] == "error":
        print(f"  message         : ERROR - {result['message']}")
        for item in result.get("missing") or []:
            print(f"  Missing         : - {item}")
        return 1
    print(f"  knowledge file  : {result.get('knowledge_file')}")
    print(f"  panel count     : {result.get('panel_count')}")
    print(f"  status          : {result.get('status')}")
    if result.get("changed"):
        print(f"  changed panels  : {', '.join(result['changed'])}")
    if result.get("missing"):
        print("  Missing (unavailable):")
        for item in result["missing"]:
            print(f"    - {item}")
    index = load_index(cfg)
    entry = next((e for e in index["pages"] if e["page"] == page_num), None)
    print(f"  index           : {entry}")
    # Remember what this page revealed (best-effort; never blocks the stage):
    # project memory + recent-pages window so character names survive resumes
    # and even a new PDF in the same project.
    try:
        from pipeline.context_memory import remember_project
        remembered = remember_project(cfg, page_num)
        if remembered:
            print("  memory          : project + last-pages context updated")
    except Exception:
        pass
    return 0


def cmd_scenes(cfg, page_num, force):
    ensure_dirs(cfg)
    setup_logging(cfg)
    from pipeline.scene_builder import SceneProcessor
    from state import State

    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = SceneProcessor(cfg).run_page(page_num, state, force=force)
    print("MangaExplainer - scenes")
    print(f"  page            : {result['page']}")
    if result["result"] == "error":
        print(f"  result          : ERROR - {result['message']}")
        return 1
    if result["result"] == "skipped":
        print(f"  result          : skipped (already completed)")
        print(f"  scenes file     : {result.get('scenes_file')}")
        return 0
    print(f"  result          : built")
    print(f"  scenes file     : {result.get('scenes_file')}")
    print(f"  scene count     : {result.get('scene_count')}")
    for rank, panels in enumerate(result["panels"], 1):
        print(f"    scene {rank:02d} : {', '.join(panels)}")
    for decision in result.get("boundaries", []):
        marker = "SPLIT" if decision["boundary"] else "same "
        print(
            f"    boundary {decision['from']} -> {decision['to']}: "
            f"{marker} (score {decision['score']})"
        )
    if result.get("missing"):
        print("  Missing (unavailable):")
        for item in result["missing"]:
            print(f"    - {item}")
    return 0


def cmd_script(cfg, page_num, scene_num, force):
    ensure_dirs(cfg)
    setup_logging(cfg)
    from pipeline.script_generator import ScriptGenerator
    from state import State

    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = ScriptGenerator(cfg).run_scene(page_num, scene_num, state, force=force)
    print("MangaExplainer - script")
    print(f"  page           : {result['page']}")
    print(f"  scene          : {result['scene']}")
    if result["result"] == "error":
        print(f"  result         : ERROR - {result['message']}")
        if result.get("raw"):
            print(f"  raw output     : {result['raw']}")
        return 1
    if result["result"] == "skipped":
        print(f"  result         : skipped (already completed)")
        print(f"  script file    : {result.get('script_json')}")
        return 0
    print(f"  result         : written")
    print(f"  scene id       : {result.get('scene_id')}")
    print(f"  provider       : {result.get('provider')} ({result.get('model')})")
    print(f"  segments       : {result.get('segment_count')}")
    print(f"  text length    : {result.get('text_length')} chars")
    print(f"  referenced     : {result.get('referenced_panels')}")
    print(f"  script json    : {result.get('script_json')}")
    print(f"  script txt     : {result.get('script_txt')}")
    for entry in result.get("segments", []):
        print(
            f"    {entry['segment_id']} [{entry['type']}] {entry['estimated_seconds']}s "
            f"{entry['visual_intent']}/{entry['camera']} {entry['text']!r}"
        )
    return 0


def cmd_audio(cfg, page_num, scene_num, force):
    ensure_dirs(cfg)
    setup_logging(cfg)
    from pipeline.audio_generator import AudioGenerator
    from state import State

    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = AudioGenerator(cfg).run_scene(page_num, scene_num, state, force=force)
    print("MangaExplainer - audio")
    print(f"  page           : {result['page']}")
    print(f"  scene          : {result['scene']}")
    if result["result"] == "error":
        print(f"  result         : ERROR - {result['message']}")
        return 1
    if result["result"] == "skipped":
        print(f"  result         : skipped (already completed)")
        print(f"  audio file     : {result.get('audio_file')}")
        return 0
    print(f"  result         : generated")
    print(f"  scene id       : {result.get('scene_id')}")
    print(f"  engine         : {result.get('engine')}")
    print(f"  segments       : {result.get('segment_count')}")
    print(f"  total duration : {result.get('total_duration_ms')} ms "
          f"({result.get('total_duration_ms', 0) / 1000.0:.1f} s) @ {result.get('sample_rate')} Hz")
    print(f"  audio file     : {result.get('audio_file')}")
    print(f"  manifest file  : {result.get('manifest_file')}")
    for entry in result.get("segments", []):
        print(
            f"    {entry['segment_id']} [{entry['type']}] "
            f"{entry['start_ms']}-{entry['end_ms']}ms "
            f"({entry['duration_ms']}ms) {entry['text']!r}"
        )
    return 0


def cmd_tts(cfg, page_num, scene_num, force, segment):
    ensure_dirs(cfg)
    setup_logging(cfg)
    from pipeline.tts_runner import TtsRunner
    from pipeline.pocket_tts import (
        PocketTtsError,
        PocketTtsUnavailable,
        PocketTtsNotConfigured,
    )
    from state import State

    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    runner = TtsRunner(cfg)
    print("MangaExplainer - tts (Pocket TTS narration)")
    try:
        result = runner.run_scene(
            page_num, scene_num, state, force=force, segment=segment
        )
    except PocketTtsNotConfigured as exc:
        print(f"  result         : ERROR - {exc}")
        print("  enable tts.enabled=true (or use --yes) in config/config.yaml")
        return 1
    except PocketTtsUnavailable as exc:
        print(f"  result         : ERROR - {exc}")
        print("  install with: pip install pocket-tts   (CPU wheels)")
        print("  for offline testing set tts.provider=mock in config/config.yaml")
        return 1
    except PocketTtsError as exc:
        print(f"  result         : ERROR - {exc}")
        return 1
    except ValueError as exc:
        print(f"  result         : ERROR - {exc}")
        return 1

    print(f"  page           : {result['page']}")
    print(f"  scene          : {result['scene']}")
    print(f"  scene key      : {result['scene_key']}")
    print(f"  result         : {result['summary']}")
    print(f"  segments total : {result['segments_total']}")
    print(f"  segments gen   : {result['segments_generated']}")
    print(f"  segments skip  : {result['segments_skipped']}")
    if result.get("warning"):
        print(f"  warning        : {result['warning']}")
    if result.get("conditioning"):
        print(f"  conditioning   : {result['conditioning']}")
    if result.get("conditioning_unavailable"):
        print(f"  conditioning   : UNAVAILABLE -> {result['conditioning_unavailable']}")
    print(f"  audio dir      : {result['audio_dir']}")
    print(f"  timing json    : {result['timing_json']}")
    return 0


def cmd_tts_narration(cfg, script, out_dir, force, timing_only):
    ensure_dirs(cfg)
    setup_logging(cfg)
    from pipeline.tts_manifest import (
        NarrationManifestRunner,
        load_narration_segments,
    )
    from pipeline.pocket_tts import (
        PocketTtsError,
        PocketTtsUnavailable,
        PocketTtsNotConfigured,
    )

    if script is None:
        # discover the first narration script in the configured script_dir
        candidates = sorted(Path(cfg.output.script_dir).glob("*_scene_*.json"))
        if not candidates:
            print("  result         : ERROR - no narration script found; pass "
                  "--script PATH")
            return 1
        script = str(candidates[0])
    script = Path(script)
    if out_dir is None:
        out_dir = str(Path(cfg.output.audio_dir))
    out_dir = Path(out_dir)

    print("MangaExplainer - tts-narration (Pocket TTS, Tasks 14/15)")
    print(f"  script         : {script}")

    runner = NarrationManifestRunner(cfg)

    if timing_only:
        # Task 15: recompute durations + start/end times from existing WAVs only.
        try:
            manifest = runner.finalize_timing(out_dir)
        except Exception as exc:
            print(f"  result         : ERROR - {exc}")
            return 1
        print(f"  result         : timing updated ({len(manifest)} segments)")
    else:
        # Task 14: generate one WAV per segment, then Task 15 timing.
        try:
            segments = load_narration_segments(script)
        except Exception as exc:
            print(f"  result         : ERROR - {exc}")
            return 1
        try:
            manifest = runner.run(segments, out_dir, force=force)
        except PocketTtsNotConfigured as exc:
            print(f"  result         : ERROR - {exc}")
            return 1
        except PocketTtsUnavailable as exc:
            print(f"  result         : ERROR - {exc}")
            print("  install with: pip install pocket-tts")
            return 1
        except PocketTtsError as exc:
            print(f"  result         : ERROR - {exc}")
            return 1
        print(f"  result         : generated ({len(manifest)} segments)")

    print(f"  output dir     : {out_dir}")
    print(f"  manifest       : {out_dir / 'manifest.json'}")
    for entry in manifest:
        print(f"    {entry['segment_id']:<10} "
              f"dur={entry['duration']:.2f}s "
              f"start={entry.get('start_time', 0):.2f} "
              f"end={entry.get('end_time', 0):.2f}")
    return 0


def cmd_panels_prep(cfg, pages):
    ensure_dirs(cfg)
    setup_logging(cfg)
    from pipeline.panel_prep import (
        PanelPrepError,
        prepare_panels_manifest,
        visuals_manifest_path,
    )
    print("MangaExplainer - panels-prep (Task 16)")
    try:
        manifest = prepare_panels_manifest(cfg, ROOT, page_nums=pages)
    except PanelPrepError as exc:
        print(f"  result         : ERROR - {exc}")
        return 1
    print(f"  result         : prepared ({len(manifest)} panels)")
    print(f"  manifest       : {visuals_manifest_path(ROOT)}")
    for entry in manifest:
        segs = entry.get("narration_segments") or []
        seg_ids = ",".join(s["segment_id"] for s in segs) or "-"
        print(
            f"    {entry['panel_id']:<10} {entry['width']}x{entry['height']} "
            f"ar={entry['aspect_ratio']}\t-> {seg_ids}"
        )
    return 0


def cmd_plan(cfg, page_num, scene_num, force):
    ensure_dirs(cfg)
    setup_logging(cfg)
    from pipeline.visual_planner import VisualPlanner
    from state import State

    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = VisualPlanner(cfg).run_scene(page_num, scene_num, state, force=force)
    print("MangaExplainer - plan")
    print(f"  page           : {result['page']}")
    print(f"  scene          : {result['scene']}")
    if result["result"] == "error":
        print(f"  result         : ERROR - {result['message']}")
        return 1
    if result["result"] == "skipped":
        print(f"  result         : skipped (already completed)")
        print(f"  timeline file  : {result.get('timeline_file')}")
        return 0
    print(f"  result         : planned")
    print(f"  scene id       : {result.get('scene_id')}")
    print(f"  shots          : {result.get('shot_count')}")
    print(f"  review needed  : {result.get('review_count')} "
          f"({', '.join(result.get('needs_review') or []) or 'none'})")
    dropped = result.get("dropped_panel_ids") or []
    if dropped:
        print(f"  dropped panels : {', '.join(dropped)}")
    print(f"  timeline file  : {result.get('timeline_file')}")
    print(f"  review file    : {result.get('review_file')}")
    for shot in result.get("shots", []):
        marker = "REVIEW" if shot["needs_review"] else "    "
        print(
            f"    {shot['shot_id']} [{marker}] {shot['visual_intent']}/"
            f"{shot['camera']['type']} {shot['estimated_duration']:.1f}s "
            f"score={shot['match_score']:.2f} reuse={shot['reuse_count']} "
            f"{shot['primary_panel']} -> {', '.join(shot['panel_ids'])}"
        )
    return 0


def cmd_crops(cfg, page_num, scene_num, force):
    ensure_dirs(cfg)
    setup_logging(cfg)
    from pipeline.crop_planner import CropPlanner
    from state import State

    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = CropPlanner(cfg).run_scene(page_num, scene_num, state, force=force)
    print("MangaExplainer - crops")
    print(f"  page           : {result['page']}")
    print(f"  scene          : {result['scene']}")
    if result["result"] == "error":
        print(f"  result         : ERROR - {result['message']}")
        for item in result.get("shot_errors") or []:
            print(f"    - {item}")
        return 1
    if result["result"] == "skipped":
        print(f"  result         : skipped (already completed)")
        print(f"  shots dir      : {result.get('shots_dir')}")
        return 0
    target = result.get("target") or {}
    print(f"  result         : computed")
    print(f"  scene id       : {result.get('scene_id')}")
    print(f"  shots          : {result.get('shot_count')}")
    print(f"  target frame   : {target.get('width')}x{target.get('height')} "
          f"({target.get('aspect')})")
    print(f"  shots dir      : {result.get('shots_dir')}")
    print(f"  crops dir      : {result.get('crops_dir')}")
    for entry in result.get("shots", []):
        crop = entry.get("crop") or {}
        mode = "LETTERBOX" if entry.get("letterbox") else "16:9     "
        print(
            f"    {entry['shot_id']} [{mode}] {entry['strategy']} "
            f"{crop.get('width')}x{crop.get('height')} @({crop.get('x')},{crop.get('y')}) "
            f"{entry['panel']} {entry['intent']} regions={entry.get('region_count')} -> "
            f"{entry.get('image')}"
        )
    return 0


def cmd_motion(cfg, page_num, scene_num, force, keyframes):
    ensure_dirs(cfg)
    setup_logging(cfg)
    from pipeline.motion import (
        MotionError,
        NoTimelineData,
        NoPanelData,
        NoImageData,
        run_motion,
        render_plan_path,
    )
    print("MangaExplainer - motion (Ken Burns + transitions, Tasks 17/18)")
    if keyframes is None:
        motion_cfg = getattr(cfg, "motion", None)
        keyframes = int(motion_cfg.get("keyframes", 12)) if motion_cfg else 12
    page_nums = [page_num] if page_num is not None else None
    if page_num is not None and scene_num is not None:
        # a single scene still requires a full timeline file; scene filter is
        # handled at render time, so we plan the whole scene's file here.
        pass
    try:
        result = run_motion(cfg, ROOT, page_nums=page_nums, force=force,
                            keyframes=keyframes)
    except NoTimelineData as exc:
        print(f"  result         : ERROR - {exc}")
        return 1
    except NoPanelData as exc:
        print(f"  result         : ERROR - {exc}")
        return 1
    except NoImageData as exc:
        print(f"  result         : ERROR - {exc}")
        return 1
    except MotionError as exc:
        print(f"  result         : ERROR - {exc}")
        return 1
    print(f"  result         : {result['result']}")
    print(f"  entries        : {result['entries']} shots")
    print(f"  transitions    : {result['transitions']}")
    print(f"  plan file      : {result['plan_file']}")
    return 0


def cmd_mix(cfg, section_seconds):
    ensure_dirs(cfg)
    setup_logging(cfg)
    from pipeline.audio_mix import MixError, run_mix
    print("MangaExplainer - mix (narration + music + sfx, Task 21)")
    if section_seconds is None:
        render_cfg = getattr(cfg, "render", None)
        section_seconds = float(render_cfg.get("section_seconds", 15)) \
            if render_cfg else 15
    try:
        result = run_mix(cfg, ROOT, section_seconds=section_seconds)
    except MixError as exc:
        print(f"  result         : ERROR - {exc}")
        return 1
    print(f"  result         : {result['result']}")
    print(f"  segments       : {result['segments']}")
    print(f"  sample rate    : {result['sample_rate']}")
    print(f"  duration       : {result['duration']}s")
    print(f"  global peak    : {result['global_peak']}")
    print(f"  normalize      : x{result['normalize_factor']}")
    print(f"  background music : {'on' if result['music'] else 'off'}")
    print(f"  sound effects    : {'on' if result['sfx'] else 'off'}")
    print(f"  final mix      : {result['final_mix']}")
    return 0


def cmd_render(cfg, out_path, low_ram, fps):
    ensure_dirs(cfg)
    setup_logging(cfg)
    from pipeline.video_render import RenderError, render_video
    print("MangaExplainer - render (video, Tasks 22/23)")
    try:
        result = render_video(cfg, ROOT, out_path=out_path, low_ram=low_ram,
                              fps_override=fps)
    except RenderError as exc:
        print(f"  result         : ERROR - {exc}")
        return 1
    print(f"  result         : {result['result']}")
    print(f"  resolution     : {result['resolution']}")
    print(f"  fps            : {result['fps']}")
    print(f"  duration       : {result['duration']}s")
    print(f"  frames         : {result['frames']}")
    print(f"  low RAM mode   : {'on' if result['low_ram'] else 'off'}")
    print(f"  codec          : {result['codec']}")
    print(f"  output         : {result['output']}")
    return 0


def cmd_export(cfg, low_ram, fps):
    ensure_dirs(cfg)
    setup_logging(cfg)
    from pipeline.export import ExportError, export_final
    print("MangaExplainer - export (final MP4 + video_info.json, Task 24)")
    try:
        result = export_final(cfg, ROOT, low_ram=low_ram, fps_override=fps)
    except ExportError as exc:
        print(f"  result         : ERROR - {exc}")
        return 1
    print(f"  result         : {result['result']}")
    print(f"  video          : {result['video']}")
    print(f"  video_info     : {result['video_info']}")
    info = result["info"]
    print(f"  duration       : {info.get('duration')}s")
    print(f"  resolution     : {info.get('resolution')}")
    print(f"  fps            : {info.get('fps')}")
    print(f"  video codec    : {info.get('video_codec')}")
    print(f"  audio codec    : {info.get('audio_codec')}")
    print(f"  file size      : {info.get('file_size')} bytes")
    return 0


def cmd_quality(cfg, video):
    ensure_dirs(cfg)
    setup_logging(cfg)
    from pipeline.quality_check import QualityCheckError, check_quality, report_path
    print("MangaExplainer - quality-check (Task 25)")
    try:
        report = check_quality(cfg, ROOT, video_path=video)
    except QualityCheckError as exc:
        print(f"  result         : ERROR - {exc}")
        return 1
    status = report["status"]
    print(f"  result         : {status.upper()}")
    print(f"  errors         : {report.get('error_count', 0)}")
    for c in report["checks"]:
        mark = "PASS" if c["passed"] else ("FAIL" if c["critical"] else "warn")
        print(f"    [{mark}] {c['check']:22s} {c['detail']}")
    print(f"  report         : {report_path(ROOT)}")
    # Task 25: an error must be clearly reported, not silently continued.
    if status == "error":
        return 1
    if status == "warning":
        return 0
    return 0


def cmd_match(cfg, page_num, all_pages, force):
    ensure_dirs(cfg)
    setup_logging(cfg)
    from pipeline.narration_matching import (
        NarrationMatcher,
        consolidated_path,
        consolidate_mapping,
    )
    matcher = NarrationMatcher(cfg)
    print("MangaExplainer - match (panel <-> narration mapping, Task 10)")
    if page_num is not None and not all_pages:
        result = matcher.run_page(page_num, force=force)
        if result["result"] == "error":
            print(f"  page           : {result['page']}")
            print(f"  result         : ERROR - {result['message']}")
            return 1
        print(f"  page           : {result['page']}")
        if result["result"] == "skipped":
            print("  result         : skipped (already matched)")
        else:
            print(f"  result         : {result['result']} "
                  f"({result.get('segment_count', 0)} segment(s) <-> "
                  f"{result.get('panel_count', 0)} panel(s), "
                  f"{result.get('unmatched_panels', 0)} unmatched)")
        print(f"  mapping file   : {result.get('mapping_file')}")
        for warning in result.get("warnings") or []:
            print(f"  warning        : {warning}")
        return 0

    result = matcher.run_all(force=force)
    print(f"  mode           : {'all pages (force rebuild)' if force else 'all pages (resume)'}")
    if result["pages"]:
        print(f"  result         : matched {result['pages_done']}, "
              f"skipped {result['pages_skipped']}, failed {result['pages_failed']}")
        for row in result["pages"]:
            marker = "MATCHED" if row["result"] == "ok" else row["result"].upper()
            extra = ""
            if row.get("segment_count") is not None:
                extra = f"  ({row['segment_count']} segs, {row.get('unmatched_panels')} unmatched)"
            if row.get("reason"):
                extra = f"  - {row['reason']}"
            if row.get("message"):
                extra = f"  - {row['message']}"
            print(f"    page {row['page']:<4} [{marker}]{extra}")
    else:
        print(f"  result         : ERROR - {result.get('message')}")
        return 1
    if result["pages_done"] and not result["pages_failed"]:
        try:
            merged = consolidate_mapping(cfg)
            print(f"  consolidated   : {consolidated_path(cfg)}")
            print(f"    panels      : {merged['total_panels']}")
            print(f"    segments    : {merged['total_segments']}")
            print(f"    unmatched   : {merged['total_unmatched_panels']}")
        except Exception as exc:
            print(f"  consolidated   : skipped - {exc}")
    return 0 if not result["pages_failed"] else 1


def cmd_pipeline(cfg, force):
    ensure_dirs(cfg)
    setup_logging(cfg)
    from pipeline.run_pipeline import run_pipeline
    print("MangaExplainer - pipeline (Task 26, complete chain: Manga -> final MP4)")
    print("  (low-RAM, sequential, checkpointed; resumes from last completed "
          "step; no subtitles)")
    result = run_pipeline(cfg, ROOT, force=force)
    print("\nPipeline summary:")
    for row in result["report"]:
        status = (row.get("status") or "unknown").upper()
        skipped = " (skipped)" if row.get("skipped") else ""
        print(f"    {row['name']:<18} {status}{skipped}")
    if result["status"] == "error":
        print(f"\n  PIPELINE ERROR: {result.get('error')}")
        print("  Stopped. Fix the error and re-run `pipeline` to resume from "
              f"here ({result.get('current')}).")
        return 1
    return 0


def cmd_pipeline_resume(cfg):
    ensure_dirs(cfg)
    setup_logging(cfg)
    from pipeline.run_pipeline import run_pipeline
    print("MangaExplainer - pipeline resume (continue from last completed step)")
    result = run_pipeline(cfg, ROOT, force=False)
    if result["status"] == "error":
        print(f"\n  PIPELINE ERROR: {result.get('error')}")
        return 1
    return 0


def cmd_pipeline_status(cfg):
    ensure_dirs(cfg)
    from state import State
    from pipeline.run_pipeline import STAGE_NAMES, PIPELINE_STAGES
    state = State(STAGE_NAMES, Path(cfg.pipeline.checkpoints.dir))
    total = len(STAGE_NAMES)
    rows = {r["name"]: r["status"] for r in state.summary()}
    print("MangaExplainer - pipeline status (Task 26 chain)")
    print(f"  checkpoint     : {state.path}")
    print(f"  progress       : {state.completed_count()}/{total} stages "
          f"({round(100 * state.completed_count() / total)}%)")
    width = max(len(name) for name in STAGE_NAMES)
    for name, label in PIPELINE_STAGES:
        print(f"    {name:<{width}}  {rows.get(name, 'pending')}  ({label})")
    nxt = state.next_pending()
    print(f"  next stage     : {nxt or '(none - complete)'}")
    return 0


def cmd_start(cfg, force=False):
    ensure_dirs(cfg)
    setup_logging(cfg)
    from pipeline.run_pipeline import run_pipeline
    print("MangaExplainer - start (complete chain: Manga -> final MP4)")
    if force:
        print("  mode            : FORCE — rebuilding every stage from scratch")
    result = run_pipeline(cfg, ROOT, force=force)
    if result["status"] == "error":
        print(f"\n  START ERROR: {result.get('error')}")
        print("  Stopped. Fix the error and re-run `start`/`resume` to "
              "continue from here (completed work is skipped).")
        return 1
    return 0


def cmd_check(cfg):
    ensure_dirs(cfg)
    setup_logging(cfg)
    from pipeline.quality_check import QualityCheckError, check_quality
    print("MangaExplainer - check")
    try:
        report = check_quality(cfg, ROOT, low_ram=True)
    except QualityCheckError as exc:
        print(f"  result         : ERROR - {exc}")
        return 1
    status = report["status"]
    print(f"  result         : {status.upper()}")
    print(f"  errors         : {report.get('error_count', 0)}")
    for c in report["checks"]:
        mark = "PASS" if c["passed"] else ("FAIL" if c["critical"] else "warn")
        print(f"    [{mark}] {c['check']:22s} {c['detail']}")
    return 0 if status in ("ok", "warning") else 1


def cmd_clean(cfg):
    ensure_dirs(cfg)
    setup_logging(cfg)
    # Shares the artifact cleaner with clean-cache; valid checkpoints preserved.
    cleaned, removed_files, preserved = _clean_artifacts(cfg)
    print("MangaExplainer - clean")
    print(f"  cleared         : {', '.join(cleaned)}")
    print(f"  files deleted   : {removed_files}")
    if preserved:
        print("  checkpoints kept: " + ", ".join(preserved) +
              "  (never auto-deleted, resume-safe)")
    else:
        print("  checkpoints kept: none present yet (data/checkpoints/ preserved)")
    print("  protected       : input/, config/, state/, data/checkpoints/, "
          "logs/, pipeline/, tests/, tools/")
    return 0


def cmd_ui(port, config):
    """Launch the user-friendly browser dashboard for MangaExplainer.

    Runs the zero-dependency web UI in-process (no separate process), so you
    can drive the whole pipeline — full run, per-stage tools, PDF, voice
    demo, and the result video — from one page in the browser.
    """
    import webui
    webui.serve(host="127.0.0.1", port=port, config=config)


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        metavar="PATH",
        default=argparse.SUPPRESS,
        help="alternative config file (default: config/config.yaml)",
    )
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Low-RAM manga explanation video pipeline (skeleton stage).",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.add_parser("status", parents=[common], help="show configuration and pipeline progress")
    sub.add_parser("resume", parents=[common], help="resume the pipeline from the last checkpoint")
    sub.add_parser(
        "clean-cache",
        parents=[common],
        help="delete derived artifacts and cache (checkpoints are preserved)",
    )
    extract = sub.add_parser(
        "extract", parents=[common], help="extract a single page from the PDF to an image"
    )
    extract.add_argument(
        "--page", type=int, required=True, help="1-based page number to extract"
    )
    panels = sub.add_parser(
        "panels", parents=[common], help="detect panels on a single extracted page"
    )
    panels.add_argument(
        "--page", type=int, required=True, help="1-based page number to process"
    )
    panels.add_argument(
        "--force", action="store_true", help="re-detect even if already done"
    )
    order = sub.add_parser(
        "order",
        parents=[common],
        help="determine the manga reading order of detected panels",
    )
    order.add_argument(
        "--page", type=int, required=True, help="1-based page number to process"
    )
    order.add_argument(
        "--force", action="store_true", help="recompute even if already done"
    )
    ocr = sub.add_parser(
        "ocr", parents=[common], help="OCR a single detected panel"
    )
    ocr.add_argument(
        "--page", type=int, required=True, help="1-based page number"
    )
    ocr.add_argument(
        "--panel", type=int, required=True, help="1-based panel number"
    )
    ocr.add_argument(
        "--force", action="store_true", help="re-run even if already done"
    )
    analyze = sub.add_parser(
        "analyze",
        parents=[common],
        help="VLM analysis of a single detected panel",
    )
    analyze.add_argument(
        "--page", type=int, required=True, help="1-based page number"
    )
    analyze.add_argument(
        "--panel", type=int, required=True, help="1-based panel number"
    )
    analyze.add_argument(
        "--force", action="store_true", help="re-run even if already analyzed"
    )
    knowledge = sub.add_parser(
        "knowledge",
        parents=[common],
        help="build/update the page knowledge file for one page",
    )
    knowledge.add_argument(
        "--page", type=int, required=True, help="1-based page number"
    )
    knowledge.add_argument(
        "--force", action="store_true", help="force full rebuild of the page"
    )
    scenes = sub.add_parser(
        "scenes",
        parents=[common],
        help="group a page's panels into logical scenes",
    )
    scenes.add_argument(
        "--page", type=int, required=True, help="1-based page number"
    )
    scenes.add_argument(
        "--force", action="store_true", help="rebuild scenes even if already done"
    )
    script = sub.add_parser(
        "script",
        parents=[common],
        help="write narration script for ONE page scene",
    )
    script.add_argument(
        "--page", type=int, required=True, help="1-based page number"
    )
    script.add_argument(
        "--scene", type=int, required=True, help="1-based scene number"
    )
    script.add_argument(
        "--force", action="store_true", help="rewrite script even if already done"
    )
    audio = sub.add_parser(
        "audio",
        parents=[common],
        help="synthesize narration audio for ONE page scene",
    )
    audio.add_argument(
        "--page", type=int, required=True, help="1-based page number"
    )
    audio.add_argument(
        "--scene", type=int, required=True, help="1-based scene number"
    )
    audio.add_argument(
        "--force", action="store_true", help="regenerate audio even if already done"
    )
    tts = sub.add_parser(
        "tts",
        parents=[common],
        help="generate Pocket TTS narration audio (one segment at a time)",
    )
    tts.add_argument(
        "--page", type=int, required=True, help="1-based page number"
    )
    tts.add_argument(
        "--scene", type=int, required=True, help="1-based scene number"
    )
    tts.add_argument(
        "--segment", type=int, default=None,
        help="1-based segment index; regenerate only that segment"
    )
    tts.add_argument(
        "--force", action="store_true",
        help="regenerate audio even if a checkpoint says it is done"
    )
    tts_narration = sub.add_parser(
        "tts-narration",
        parents=[common],
        help="generate flat audio/segment_NNN.wav + audio/manifest.json "
             "(Tasks 14/15)",
    )
    tts_narration.add_argument(
        "--script", metavar="PATH", default=None,
        help="narration script JSON (default: first *_scene_*.json in "
             "t.output.script_dir)",
    )
    tts_narration.add_argument(
        "--out-dir", metavar="PATH", default=None,
        help="output directory for segment_NNN.wav + manifest.json "
             "(default: t.output.audio_dir)",
    )
    tts_narration.add_argument(
        "--timing-only", action="store_true",
        help="Task 15: recompute durations + start/end times from existing "
             "WAVs without regenerating audio",
    )
    tts_narration.add_argument(
        "--force", action="store_true",
        help="regenerate audio even if segment WAVs already exist",
    )
    panels_prep = sub.add_parser(
        "panels-prep",
        parents=[common],
        help="prepare each panel for video rendering -> "
             "visuals/panels_manifest.json (Task 16)",
    )
    panels_prep.add_argument(
        "--page", type=int, action="append", default=None,
        help="1-based page number to include (repeatable); "
             "default: auto-discover available pages",
    )
    plan = sub.add_parser(
        "plan",
        parents=[common],
        help="build the visual timeline (script -> panels) for ONE page scene",
    )
    plan.add_argument(
        "--page", type=int, required=True, help="1-based page number"
    )
    plan.add_argument(
        "--scene", type=int, required=True, help="1-based scene number"
    )
    plan.add_argument(
        "--force", action="store_true", help="re-plan even if already done"
    )
    crops = sub.add_parser(
        "crops",
        parents=[common],
        help="compute 16:9 cinematic crops for ONE page scene",
    )
    crops.add_argument(
        "--page", type=int, required=True, help="1-based page number"
    )
    crops.add_argument(
        "--scene", type=int, required=True, help="1-based scene number"
    )
    crops.add_argument(
        "--force", action="store_true", help="recompute even if already done"
    )
    match_p = sub.add_parser(
        "match",
        parents=[common],
        help="build the deterministic panel <-> narration mapping for one "
             "page, or for all pages",
    )
    match_p.add_argument(
        "--page", type=int, default=None,
        help="1-based page number to match (default: --all)",
    )
    match_p.add_argument(
        "--all", action="store_true",
        help="match every page with a panels manifest, one at a time "
             "(resumes past completed pages)",
    )
    match_p.add_argument(
        "--force", action="store_true",
        help="rebuild even if a page is already matched",
    )
    motion = sub.add_parser(
        "motion",
        parents=[common],
        help="smooth Ken Burns camera path + short panel transitions -> "
             "motion/render_plan.json (Tasks 17/18)",
    )
    motion.add_argument(
        "--page", type=int, default=None,
        help="1-based page number to include (default: all pages with a "
             "shots timeline)",
    )
    motion.add_argument(
        "--scene", type=int, default=None,
        help="1-based scene number (informational; plan is per scene file)",
    )
    motion.add_argument(
        "--keyframes", type=int, default=None,
        help="keyframes for the smooth motion path (default: config "
             "motion.keyframes)",
    )
    motion.add_argument(
        "--force", action="store_true", help="regenerate even if the plan "
                                             "already exists",
    )
    mix = sub.add_parser(
        "mix",
        parents=[common],
        help="mix narration + optional music + SFX into "
             "audio/final_mix.wav (Task 21)",
    )
    mix.add_argument(
        "--section-seconds", type=float, default=None, metavar="SEC",
        help="mix section size in seconds (bounds RAM; default config "
             "render.section_seconds)",
    )
    render = sub.add_parser(
        "render",
        parents=[common],
        help="render the final video from panels + motion + transitions + "
             "final audio (Tasks 22/23)",
    )
    render.add_argument(
        "--out", metavar="PATH", default=None,
        help="output video path (default: output/MangaExplainer_video.mp4)",
    )
    render.add_argument(
        "--low-ram", action="store_true", default=None, help=argparse.SUPPRESS,
    )
    render.add_argument(
        "--no-low-ram", action="store_false", dest="low_ram", default=None,
        help="disable LOW_RAM_MODE (process more freely)",
    )
    render.add_argument(
        "--fps", type=float, default=None,
        help="override frames per second (default: config video.fps)",
    )
    export_ = sub.add_parser(
        "export",
        parents=[common],
        help="export the final MP4 + output/video_info.json "
             "(H.264 + AAC, Task 24)",
    )
    export_.add_argument(
        "--fps", type=float, default=None,
        help="override frames per second (default: config video.fps)",
    )
    export_.add_argument(
        "--no-low-ram", action="store_false", dest="low_ram", default=None,
        help="disable LOW_RAM_MODE",
    )
    quality = sub.add_parser(
        "quality-check",
        parents=[common],
        help="verify output/final_video.mp4 -> output/quality_report.json "
             "(Task 25)",
    )
    quality.add_argument(
        "--video", metavar="PATH", default=None,
        help="video to check (default: output/final_video.mp4)",
    )
    pipeline_p = sub.add_parser(
        "pipeline",
        parents=[common],
        help="run the complete chain: Manga -> ... -> final MP4 (Task 26)",
    )
    pipeline_p.add_argument(
        "--force", action="store_true",
        help="re-run already-completed stages (default: resume, never repeat "
             "completed work)",
    )
    sub.add_parser(
        "pipeline-resume",
        parents=[common],
        help="resume the complete pipeline from the last completed stage",
    )
    sub.add_parser(
        "pipeline-status",
        parents=[common],
        help="show progress of the complete pipeline (Task 26 chain)",
    )
    # --- Task 28: simple user-facing CLI --------------------------------
    start = sub.add_parser(
        "start",
        parents=[common],
        help="run the complete pipeline (Manga -> final MP4); by default "
             "skips completed work, with --force rebuilds everything",
    )
    start.add_argument(
        "--force", action="store_true",
        help="rebuild every stage from scratch, ignoring completed checkpoints",
    )
    sub.add_parser(
        "check",
        parents=[common],
        help="quality-check the output video (output/final_video.mp4)",
    )
    sub.add_parser(
        "clean",
        parents=[common],
        help="clear cache and derived artifacts (checkpoints are preserved)",
    )
    ui = sub.add_parser(
        "ui",
        parents=[common],
        help="open the user-friendly browser dashboard",
    )
    ui.add_argument(
        "--port", type=int, default=8000,
        help="port for the local web dashboard (default: 8000)",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    cfg = load_config(ROOT, getattr(args, "config", None))
    if args.command == "extract":
        return cmd_extract(cfg, args.page)
    if args.command == "panels":
        return cmd_panels(cfg, args.page, args.force)
    if args.command == "order":
        return cmd_order(cfg, args.page, args.force)
    if args.command == "ocr":
        return cmd_ocr(cfg, args.page, args.panel, args.force)
    if args.command == "analyze":
        return cmd_analyze(cfg, args.page, args.panel, args.force)
    if args.command == "knowledge":
        return cmd_knowledge(cfg, args.page, args.force)
    if args.command == "scenes":
        return cmd_scenes(cfg, args.page, args.force)
    if args.command == "script":
        return cmd_script(cfg, args.page, args.scene, args.force)
    if args.command == "audio":
        return cmd_audio(cfg, args.page, args.scene, args.force)
    if args.command == "tts":
        return cmd_tts(cfg, args.page, args.scene, args.force, args.segment)
    if args.command == "tts-narration":
        return cmd_tts_narration(
            cfg, args.script, args.out_dir, args.force, args.timing_only
        )
    if args.command == "panels-prep":
        return cmd_panels_prep(cfg, args.page)
    if args.command == "plan":
        return cmd_plan(cfg, args.page, args.scene, args.force)
    if args.command == "crops":
        return cmd_crops(cfg, args.page, args.scene, args.force)
    if args.command == "match":
        return cmd_match(cfg, args.page, args.all, args.force)
    if args.command == "motion":
        return cmd_motion(cfg, args.page, args.scene, args.force,
                          args.keyframes)
    if args.command == "mix":
        return cmd_mix(cfg, args.section_seconds)
    if args.command == "render":
        return cmd_render(cfg, args.out, args.low_ram, args.fps)
    if args.command == "export":
        return cmd_export(cfg, args.low_ram, args.fps)
    if args.command == "quality-check":
        return cmd_quality(cfg, args.video)
    if args.command == "pipeline":
        return cmd_pipeline(cfg, args.force)
    if args.command == "pipeline-resume":
        return cmd_pipeline_resume(cfg)
    if args.command == "pipeline-status":
        return cmd_pipeline_status(cfg)
    if args.command == "ui":
        return cmd_ui(args.port, getattr(args, "config", None))
    commands = {
        "status": cmd_status,
        "resume": cmd_resume,
        "start": cmd_start,
        "check": cmd_check,
        "clean": cmd_clean,
        "clean-cache": cmd_clean_cache,
    }
    if args.command == "start":
        return commands["start"](cfg, getattr(args, "force", False))
    return commands[args.command](cfg)


if __name__ == "__main__":
    sys.exit(main())