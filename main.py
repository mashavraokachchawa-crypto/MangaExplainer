#!/usr/bin/env python3
"""MangaExplainer CLI - low-RAM manga explanation video pipeline skeleton.

Commands:
    python main.py status         show configuration and pipeline progress
    python main.py resume         resume from the last checkpoint (crash-safe)
    python main.py clean-cache    delete derived artifacts and checkpoints
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
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.loader import load_config
from logger import setup_logging
from pipeline.runner import PipelineRunner
from pipeline.stages import STAGES

STAGE_NAMES = [s.name for s in STAGES]


def ensure_dirs(cfg):
    paths = {
        Path(cfg.output.dir),
        Path(cfg.pipeline.state.dir),
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
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _print_stage_table(state):
    width = max(len(name) for name in STAGE_NAMES)
    rows = {row["name"]: row["status"] for row in state.summary()}
    print(f"  stages ({len(STAGE_NAMES)}):")
    for name in STAGE_NAMES:
        print(f"    {name:<{width}}  {rows.get(name, 'pending')}")


def cmd_status(cfg):
    ensure_dirs(cfg)
    from state import State

    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    print("MangaExplainer pipeline - status")
    print(f"  input PDF        : {cfg.input.pdf}")
    print(f"  output dir       : {cfg.output.dir}")
    print(f"  image resolution : {cfg.images.resolution}")
    print(f"  video resolution : {cfg.video.resolution} @ {cfg.video.fps} fps")
    print(f"  batch size       : {cfg.pipeline.batch_size}  (low-RAM mode)")
    print(f"  cache dir        : {cfg.pipeline.cache.dir}")
    print(f"  memory guard     : {cfg.memory.guard_mb} MB")
    print(f"  checkpoint file  : {state.path}")
    if state.exists():
        print(
            f"  pipeline state   : {state.completed_count()}/{len(STAGE_NAMES)} stages completed"
        )
    else:
        print("  pipeline state   : fresh (no checkpoint yet)")
    _print_stage_table(state)
    nxt = state.next_pending()
    print(f"  next stage       : {nxt or '(none - complete)'}")
    return 0


def cmd_resume(cfg):
    ensure_dirs(cfg)
    log = setup_logging(cfg)
    from state import State

    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    runner = PipelineRunner(cfg, state)
    nxt = state.next_pending()
    if nxt is None:
        print("resume: all stages are complete; nothing to do.")
        return 0
    print(f"resume: picking up from stage '{nxt}'")
    ok, detail = runner.resume()
    log.info("resume attempt -> ok=%s stage=%s detail=%s", ok, nxt, detail)
    print(f"resume: {detail}")
    return 0


def cmd_clean_cache(cfg):
    ensure_dirs(cfg)
    log = setup_logging(cfg)
    targets = [Path(cfg.pipeline.cache.dir)] + [ROOT / s.output_dir for s in STAGES]
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
    checkpoint = Path(cfg.pipeline.state.dir) / "checkpoints.json"
    removed_state = False
    if checkpoint.exists():
        checkpoint.unlink()
        removed_state = True
    print("MangaExplainer - clean-cache")
    print(f"  cleared         : {', '.join(cleaned)}")
    print(f"  files deleted   : {removed_files}")
    print(f"  checkpoint reset: {'yes' if removed_state else 'already clean'}")
    print("  protected       : input/, config/, state/, logs/, pipeline/, tests/, tools/")
    log.info("clean-cache done: %d files removed", removed_files)
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
        help="delete derived artifacts, cache and checkpoints",
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
    commands = {
        "status": cmd_status,
        "resume": cmd_resume,
        "clean-cache": cmd_clean_cache,
    }
    return commands[args.command](cfg)


if __name__ == "__main__":
    sys.exit(main())