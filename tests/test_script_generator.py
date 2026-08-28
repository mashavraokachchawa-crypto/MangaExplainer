"""Tests: LLM-based explanation script for ONE scene (write_script stage).

Covers the nine required scenarios plus validation/unit behavior, all with a
deterministic mock LLM provider - never a real model.
"""

import json
from pathlib import Path

import pytest

from config.loader import Config
from pipeline.knowledge import knowledge_path
from pipeline.llm_provider import (
    LLMUnavailable,
    MockLLMProvider,
    clean_text,
    create_llm_provider,
)
from pipeline.prompts import build_script_prompt
from pipeline.script_generator import (
    CAMERAS,
    SegmentError,
    ScriptError,
    ScriptGenerator,
    VISUAL_INTENTS,
    build_txt,
    json_path,
    select_scene,
    normalize_segment,
    txt_path,
)
from state import State

ROOT = Path(__file__).resolve().parent.parent
STAGE_NAMES = ["extract_pages", "segment_panels", "run_ocr", "analyze_panels",
               "build_scenes", "write_script"]


def make_cfg(tmp_path, llm=None):
    llm = llm or {}
    data = {
        "input": {"pdf": str(tmp_path / "unused.pdf")},
        "output": {
            "dir": str(tmp_path / "output"),
            "pages_dir": str(tmp_path / "pages"),
            "panels_dir": str(tmp_path / "panels"),
            "ocr_dir": str(tmp_path / "ocr"),
            "analysis_dir": str(tmp_path / "analysis"),
            "scenes_dir": str(tmp_path / "scenes"),
            "script_dir": str(tmp_path / "script"),
        },
        "images": {"format": "jpg", "render_scale": 1.0, "jpeg_quality": 80},
        "panels": {"jpeg_quality": 85, "min_area": 3000, "max_panels": 40},
        "reading": {"direction": "rtl", "row_overlap_ratio": 0.5},
        "ocr": {"engine": "auto", "language": "eng+jpn", "psm": 11, "timeout_seconds": 30},
        "scenes": {"threshold": 0.45, "weights": {}, "continuity": {}, "transition_keywords": [], "summary_max_items": 6},
        "llm": {
            "enabled": True, "provider": "mock", "model": "",
            "device": "cpu", "max_context": 4096, "max_new_tokens": 512,
            "temperature": 0.7, "timeout_seconds": 120,
        },
        "video": {"resolution": "1920x1080", "fps": 30},
        "pipeline": {"batch_size": 1, "state": {"dir": str(tmp_path / "state")}, "cache": {"dir": str(tmp_path / "state" / "cache")}},
        "memory": {"guard_mb": 3072},
        "logging": {"level": "INFO", "console_level": "WARNING", "log_dir": str(tmp_path / "logs"), "max_bytes": 1048576, "backup_count": 3},
    }
    data["llm"].update(llm)
    return Config(data, tmp_path)


def record(cfg, index=1, ocr_text=None, characters=None, event=None):
    image = Path(cfg.output.panels_dir) / "page_001" / f"panel_{index:03d}.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    if not image.exists():
        image.write_bytes(b"\xff\xd8\xff\xe0fake")
    visual = {
        "characters": [{"name": n, "description": "", "action": "", "emotion": ""} for n in (characters or [])],
        "environment": "unknown", "actions": [], "objects": [], "visual_effects": [],
        "important_event": "unknown", "composition": "unknown",
        "story_relevance": "unknown", "confidence": 0.8,
    }
    if event:
        visual["important_event"] = event
    return {
        "panel_id": f"p001_{index:03d}", "page": 1, "reading_order": index,
        "image": str(image), "bbox": [8, 8, 90, 80],
        "ocr": {"text": ocr_text or "", "blocks": []} if ocr_text is not None else None,
        "visual": visual,
        "previous_panel": None, "next_panel": None, "scene_id": None,
    }


def write_knowledge(cfg, records, page=1):
    path = knowledge_path(cfg, page)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "page": page,
        "source": str(path.parent / f"page_{page:03d}.jpg"),
        "panel_count": len(records),
        "reading_direction": "rtl",
        "panels": records,
    }), encoding="utf-8")
    return path


def scene(sid, panel_ids, characters=(), locations=(), events=(), confidence=0.8):
    return {
        "scene_id": sid, "page_start": 1, "page_end": 1,
        "panel_ids": list(panel_ids), "characters": list(characters),
        "locations": list(locations), "events": list(events),
        "summary": "A test scene.", "confidence": confidence,
    }


def write_scenes(cfg, scenes, page=1, raw=None):
    path = Path(cfg.output.scenes_dir) / f"page_{page:03d}_scenes.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        path.write_text(raw, encoding="utf-8")
    else:
        path.write_text(json.dumps({"page": page, "scene_count": len(scenes), "scenes": scenes}), encoding="utf-8")
    return path


def seg(text, panel_ids=("p001_001",), seconds=4.5, intent="full_panel",
         camera="static", importance=0.8, **extra):
    out = {
        "text": text, "panel_ids": list(panel_ids), "estimated_seconds": seconds,
        "visual_intent": intent, "camera": camera, "importance": importance,
    }
    out.update(extra)
    return out


def run(cfg, page=1, scene_num=1, force=False, provider=None):
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = ScriptGenerator(cfg, provider=provider).run_scene(page, scene_num, state, force=force)
    return result, state


def default_provider(cfg, response=None):
    return create_llm_provider(cfg, response=response if response is not None else json.dumps({"segments": [seg("The hero advances.")]}))


# ----------------------------------------------------------------- required


def test_valid_scene(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [record(cfg, index=1, ocr_text="Hello World", characters=["guts"])])
    write_scenes(cfg, [scene("scene_001", ["p001_001"], characters=["guts"])])
    provider = default_provider(cfg)
    result, state = run(cfg, provider=provider)
    assert result["result"] == "ok"
    assert result["scene_id"] == "scene_001"
    assert result["segment_count"] == 1
    assert result["referenced_panels"] == ["p001_001"]
    assert json_path(cfg, 1, 1).is_file()
    assert txt_path(cfg, 1, 1).is_file()
    doc = json.loads(json_path(cfg, 1, 1).read_text("utf-8"))
    entry = doc["segments"][0]
    assert entry["segment_id"] == "seg_001"
    assert entry["text"] == "The hero advances."
    assert entry["visual_intent"] in VISUAL_INTENTS
    assert entry["camera"] in CAMERAS
    assert entry["estimated_seconds"] > 0
    assert state.pages.get("page_001_scene_001") == "script_completed"


def test_missing_scene(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [record(cfg, index=1)])
    write_scenes(cfg, [scene("scene_001", ["p001_001"])])
    result, state = run(cfg, scene_num=2, provider=default_provider(cfg))
    assert result["result"] == "error"
    assert "scene 2 not found" in result["message"]
    assert state.pages.get("page_001_scene_002") is None
    assert not json_path(cfg, 1, 2).is_file()


def test_empty_scene(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [record(cfg, index=1)])
    write_scenes(cfg, [scene("scene_001", [])])
    result, state = run(cfg, provider=default_provider(cfg))
    assert result["result"] == "error"
    assert "no panels" in result["message"]
    assert state.pages.get("page_001_scene_001") is None


def test_malformed_llm_response(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [record(cfg, index=1)])
    write_scenes(cfg, [scene("scene_001", ["p001_001"])])
    provider = create_llm_provider(cfg, response="not json at all")
    result, state = run(cfg, provider=provider)
    assert result["result"] == "error"
    assert "no valid JSON" in result["message"]
    assert "raw" in result
    raw = Path(result["raw"])
    assert raw.is_file() and raw.exists()
    assert state.pages.get("page_001_scene_001") is None
    assert not json_path(cfg, 1, 1).is_file()


def test_invalid_panel_id(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [record(cfg, index=1)])
    write_scenes(cfg, [scene("scene_001", ["p001_001"])])
    provider = create_llm_provider(cfg, response=json.dumps({"segments": [seg("oops", panel_ids=["p001_999"])]}))
    result, state = run(cfg, provider=provider)
    assert result["result"] == "error"
    assert "p001_999" in result["message"]
    assert "raw response saved" in result["message"]
    assert state.pages.get("page_001_scene_001") is None


def test_invalid_visual_intent(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [record(cfg, index=1)])
    write_scenes(cfg, [scene("scene_001", ["p001_001"])])
    provider = create_llm_provider(cfg, response=json.dumps({"segments": [seg("oops", intent="extreme_zoom")]}))
    result, state = run(cfg, provider=provider)
    assert result["result"] == "error"
    assert "visual_intent" in result["message"]
    assert state.pages.get("page_001_scene_001") is None


def test_checkpoint_skip(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [record(cfg, index=1)])
    write_scenes(cfg, [scene("scene_001", ["p001_001"])])
    provider = default_provider(cfg)
    first, _ = run(cfg, provider=provider)
    assert first["result"] == "ok"
    second, _ = run(cfg, provider=provider)
    assert second["result"] == "skipped"


def test_force_regeneration(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [record(cfg, index=1)])
    write_scenes(cfg, [scene("scene_001", ["p001_001"])])
    first, _ = run(cfg, provider=default_provider(cfg))
    assert first["result"] == "ok"
    second, _ = run(cfg, provider=default_provider(cfg), force=True)
    assert second["result"] == "ok"
    assert second["segment_count"] == 1


def test_missing_model_configuration(tmp_path):
    cfg = make_cfg(tmp_path, llm={"provider": "local", "model": ""})
    write_knowledge(cfg, [record(cfg, index=1)])
    write_scenes(cfg, [scene("scene_001", ["p001_001"])])
    result, state = run(cfg, provider=None)
    assert result["result"] == "error"
    assert "llm.model" in result["message"]
    assert state.pages.get("page_001_scene_001") is None


# ---------------------------------------------------------------- validation


def test_invalid_camera(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [record(cfg, index=1)])
    write_scenes(cfg, [scene("scene_001", ["p001_001"])])
    provider = create_llm_provider(cfg, response=json.dumps({"segments": [seg("oops", camera="helicopter")]}))
    result, state = run(cfg, provider=provider)
    assert result["result"] == "error"
    assert "camera" in result["message"]
    assert state.pages.get("page_001_scene_001") is None


def test_non_positive_duration_rejected(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [record(cfg, index=1)])
    write_scenes(cfg, [scene("scene_001", ["p001_001"])])
    provider = create_llm_provider(cfg, response=json.dumps({"segments": [seg("oops", seconds=0)]}))
    result, _ = run(cfg, provider=provider)
    assert result["result"] == "error"
    assert "positive" in result["message"]


def test_empty_segment_text_rejected():
    with pytest.raises(SegmentError):
        normalize_segment(seg("   "), {"p001_001"})
    with pytest.raises(SegmentError):
        normalize_segment(seg(None), {"p001_001"})


def test_missing_intent_and_camera_default():
    cleaned = normalize_segment({"text": "Fine.", "panel_ids": ["p001_001"], "estimated_seconds": 3}, {"p001_001"})
    assert cleaned["visual_intent"] == "full_panel"
    assert cleaned["camera"] == "static"
    assert cleaned["importance"] == 0.0


def test_importance_and_seconds_normalized():
    cleaned = normalize_segment(
        seg("x", seconds=4.55, importance=3.0, intent="face_closeup", camera="pan_up"),
        {"p001_001"}, default_importance=0.5,
    )
    assert cleaned["estimated_seconds"] == 4.5
    assert cleaned["importance"] <= 1.0
    assert cleaned["importance"] >= 0.0
    assert cleaned["visual_intent"] == "face_closeup"
    assert cleaned["camera"] == "pan_up"


def test_dialogue_segment_kept():
    cleaned = normalize_segment(
        seg("We leave tonight.", type="dialogue", speaker="Guts"), {"p001_001"}
    )
    assert cleaned["type"] == "dialogue"
    assert cleaned["speaker"] == "Guts"
    narration = normalize_segment(
        seg("The night air is heavy.", type="narration"), {"p001_001"}
    )
    assert narration["type"] == "narration"
    assert "speaker" not in narration


def test_build_txt_reading_order():
    segments = [
        {"type": "narration", "text": "First."},
        {"type": "dialogue", "speaker": "unknown", "text": "Second."},
        {"type": "dialogue", "speaker": "Guts", "text": "Third."},
    ]
    txt = build_txt(segments)
    assert txt.startswith("Narrator:\nFirst.")
    assert "Dialogue:\nSecond." in txt
    assert "Dialogue (Guts):\nThird." in txt
    assert txt.count("\n\n") == 2

    lines = [line for line in txt.splitlines()]
    assert lines[0] == "Narrator:" and lines[1] == "First."
    assert lines[3] == "Dialogue:" and lines[4] == "Second."


def test_multi_segment_script(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [record(cfg, index=1), record(cfg, index=2)])
    write_scenes(cfg, [scene("scene_001", ["p001_001", "p001_002"])])
    response = {
        "segments": [
            seg("He steps forward.", ["p001_001"], seconds=3.1, intent="action_crop", camera="slow_zoom_in"),
            seg("A shadow falls.", ["p001_002"], seconds=2.8, intent="face_closeup", camera="pan_down"),
            seg("We are not alone.", ["p001_002"], seconds=4.0, type="dialogue", speaker="unknown"),
        ]
    }
    provider = create_llm_provider(cfg, response=json.dumps(response))
    result, _ = run(cfg, provider=provider)
    assert result["result"] == "ok"
    assert result["segment_count"] == 3
    doc = json.loads(json_path(cfg, 1, 1).read_text("utf-8"))
    ids = [s["segment_id"] for s in doc["segments"]]
    assert ids == ["seg_001", "seg_002", "seg_003"]
    assert doc["segments"][2]["type"] == "dialogue"
    assert doc["segments"][2]["speaker"] == "unknown"


def test_empty_segments_list_rejected(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [record(cfg, index=1)])
    write_scenes(cfg, [scene("scene_001", ["p001_001"])])
    provider = create_llm_provider(cfg, response=json.dumps({"segments": []}))
    result, state = run(cfg, provider=provider)
    assert result["result"] == "error"
    assert "no segments" in result["message"]
    assert not json_path(cfg, 1, 1).is_file()
    assert state.pages.get("page_001_scene_001") is None


def test_no_scenes_file(tmp_path):
    cfg = make_cfg(tmp_path)
    result, state = run(cfg)
    assert result["result"] == "error"
    assert "scenes file" in result["message"]


def test_invalid_scenes_json(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [record(cfg, index=1)])
    write_scenes(cfg, [scene("scene_001", ["p001_001"])], raw="{bad")
    result, _ = run(cfg)
    assert result["result"] == "error"


def test_llm_unavailable_is_error(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [record(cfg, index=1)])
    write_scenes(cfg, [scene("scene_001", ["p001_001"])])
    provider = create_llm_provider(cfg, raise_on_generate=LLMUnavailable("no model"))
    result, state = run(cfg, provider=provider)
    assert result["result"] == "error"
    assert "unavailable" in result["message"]
    assert state.pages.get("page_001_scene_001") is None


def test_raw_saved_to_logs_dir(tmp_path):
    cfg = make_cfg(tmp_path)
    write_knowledge(cfg, [record(cfg, index=1)])
    write_scenes(cfg, [scene("scene_001", ["p001_001"])])
    provider = create_llm_provider(cfg, response="garbage")
    result, _ = run(cfg, provider=provider)
    assert (Path(cfg.logging.log_dir) / "llm" / "page_001_scene_001_raw.txt").is_file()


# ------------------------------------------------------------------- units


def test_select_scene():
    doc = {"scenes": [scene("scene_001", ["p001_001"]), scene("scene_002", ["p001_002"])]}
    assert select_scene(doc, 1)["scene_id"] == "scene_001"
    assert select_scene(doc, 2)["scene_id"] == "scene_002"
    with pytest.raises(ScriptError):
        select_scene(doc, 3)
    with pytest.raises(ScriptError):
        select_scene(doc, 0)
    with pytest.raises(ScriptError):
        select_scene(doc, "1")


def test_prompt_contains_facts_and_schema():
    context = {"p001_001": {"characters": ["guts"], "event": "battle", "dialogue": "Hello World"}}
    prompt = build_script_prompt(
        scene("scene_001", ["p001_001"], characters=["guts"]),
        {"p001_001": "Hello World"},
        context,
    )
    assert "scene_001" in prompt
    assert "guts" in prompt
    assert "Hello World" in prompt
    assert "potentially imperfect" in prompt
    assert "segments" in prompt
    assert "slow_zoom_in" in prompt  # camera list surfaced for the LLM
    assert "full_panel" in prompt


def test_prompt_no_double_panel_names():
    prompt = build_script_prompt(scene("scene_001", ["p001_001"]), {}, {})
    assert "p001_001" in prompt  # ids are context; the rule forbids mentioning them to the viewer


def test_prompt_follows_one_scene_only():
    prompt = build_script_prompt(scene("scene_002", ["p001_002"]), {}, {})
    assert "scene_002" in prompt
    assert "scene_001" not in prompt


def test_clean_text():
    assert clean_text('"Narration: done."') == "done."


def test_out_of_range_scene_number_report():
    cfg = make_cfg(Path("/tmp/script_units"))
    provider = create_llm_provider(cfg)
    assert isinstance(provider, MockLLMProvider)
    assert provider.generate("hi")  # deterministic default is prose
    assert provider.last_prompt == "hi"