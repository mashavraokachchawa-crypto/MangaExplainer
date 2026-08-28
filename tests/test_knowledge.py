"""Tests: disk-first manga knowledge database (synthetic fixtures only)."""

import json
from pathlib import Path

import pytest

from config.loader import Config
from pipeline.knowledge import (
    KnowledgeBuilder,
    KnowledgeError,
    index_path,
    index_status,
    knowledge_path,
    load_index,
    load_page_knowledge,
    validate_bbox,
    validate_confidence,
    validate_knowledge,
    validate_panel_id,
)
from state import State

ROOT = Path(__file__).resolve().parent.parent
STAGE_NAMES = ["extract_pages", "segment_panels", "run_ocr", "analyze_panels"]


def make_cfg(tmp_path, analysis_dir="analysis"):
    data = {
        "input": {"pdf": str(tmp_path / "unused.pdf")},
        "output": {
            "dir": str(tmp_path / "output"),
            "pages_dir": str(tmp_path / "pages"),
            "panels_dir": str(tmp_path / "panels"),
            "ocr_dir": str(tmp_path / "ocr"),
            "analysis_dir": str(tmp_path / analysis_dir),
        },
        "images": {"format": "jpg", "render_scale": 1.0, "jpeg_quality": 80},
        "panels": {
            "jpeg_quality": 85,
            "min_area": 3000,
            "max_panels": 40,
            "pad_pixels": 2,
        },
        "reading": {"direction": "rtl", "row_overlap_ratio": 0.5},
        "ocr": {"engine": "auto", "language": "eng+jpn", "psm": 11, "timeout_seconds": 30},
        "vlm": {
            "enabled": True,
            "provider": "mock",
            "model": "",
            "device": "cpu",
            "max_image_size": 768,
            "max_new_tokens": 256,
            "timeout_seconds": 120,
        },
        "video": {"resolution": "1920x1080", "fps": 30},
        "pipeline": {
            "batch_size": 1,
            "state": {"dir": str(tmp_path / "state")},
            "cache": {"dir": str(tmp_path / "state" / "cache")},
        },
        "memory": {"guard_mb": 3072},
        "logging": {
            "level": "INFO",
            "console_level": "WARNING",
            "log_dir": str(tmp_path / "logs"),
            "max_bytes": 1048576,
            "backup_count": 3,
        },
    }
    return Config(data, tmp_path)


def _image(tmp_path, rel):
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
    return str(path)


def build_fixture(cfg, page=1, panels=3, ocr=True, vlm=True, direction="rtl",
                  corrupt=None, bad_confidence=False, bad_bbox=False,
                  page_image=True, manifest=True, order=True):
    tmp = cfg.root_dir
    _image(tmp, f"pages/page_{page:03d}.jpg") if page_image else None
    panels_dir = Path(cfg.output.panels_dir) / f"page_{page:03d}"
    panels_dir.mkdir(parents=True, exist_ok=True)
    panel_ids = [f"p{page:03d}_{i:03d}" for i in range(1, panels + 1)]
    entries = []
    for i, pid in enumerate(panel_ids, 1):
        img = f"{panels_dir}/panel_{i:03d}.jpg"
        _image(tmp, Path(img))
        entry = {
            "id": pid,
            "image": str(img),
            "bbox": [-5, 0, 90, 80] if bad_bbox else [8, 8, 90, 80],
            "area": 7200,
            "confidence": 0.9,
            "reading_order": i,
        }
        entries.append(entry)

    if manifest:
        (panels_dir / "panels.json").write_text(
            json.dumps({"page": page, "source": f"{tmp}/pages/page_{page:03d}.jpg", "panels": entries}),
            encoding="utf-8",
        )
    if corrupt == "panels":
        (panels_dir / "panels.json").write_text("{not json!!", encoding="utf-8")
    if order:
        (panels_dir / "reading_order.json").write_text(
            json.dumps({"page": page, "direction": direction, "order": panel_ids}),
            encoding="utf-8",
        )
    if corrupt == "order":
        (panels_dir / "reading_order.json").write_text("broken", encoding="utf-8")

    if ocr:
        _write_ocr(cfg, page, panels, corrupt)
    if vlm:
        _write_vlm(cfg, page, panels, bad_confidence)
    return panel_ids


def _write_ocr(cfg, page, panels, corrupt=None):
    for i in range(1, panels + 1):
        key = f"page_{page:03d}_panel_{i:03d}"
        path = Path(cfg.output.ocr_dir) / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "page": page,
            "panel": i,
            "engine": "mock",
            "combined_text": f"text {i}"
            if not (corrupt == "ocr" and i == 1)
            else "",
            "text_blocks": [] if (corrupt == "ocr" and i == 1) else [
                {"text": f"text {i}", "bbox": [1, 1, 10, 10], "confidence": 0.9}
            ],
        }
        if corrupt == "ocr" and i == 1:
            path.write_text("{oops", encoding="utf-8")
        else:
            path.write_text(json.dumps(payload), encoding="utf-8")


def _write_vlm(cfg, page, panels, bad_confidence=False):
    for i in range(1, panels + 1):
        key = f"page_{page:03d}_panel_{i:03d}"
        path = Path(cfg.output.analysis_dir) / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        analysis = {
            "characters": [{"name": "unknown", "description": "figure",
                            "action": "unknown", "emotion": "unknown"}],
            "environment": "unknown",
            "actions": [],
            "objects": [],
            "visual_effects": [],
            "important_event": "unknown",
            "composition": "unknown",
            "story_relevance": "unknown",
            "confidence": 2.0 if bad_confidence else 0.8,
        }
        path.write_text(
            json.dumps(
                {"page": page, "panel": i, "image": "x", "provider": "mock",
                 "model": "mock", "analysis": analysis}
            ),
            encoding="utf-8",
        )


def run(cfg, page=1, force=False):
    state = State(STAGE_NAMES, cfg.pipeline.state.dir)
    result = KnowledgeBuilder(cfg).build_page(page, state, force=force)
    return result, state


def read_knowledge(cfg, page=1):
    return json.loads(knowledge_path(cfg, page).read_text("utf-8"))


# ------------------------------------------------------------ required tests


def test_valid_page_knowledge(tmp_path):
    cfg = make_cfg(tmp_path)
    build_fixture(cfg, panels=3)
    result, state = run(cfg)
    assert result["result"] == "ok"
    assert result["panel_count"] == 3
    doc = read_knowledge(cfg)
    validate_knowledge(doc)
    assert doc["page"] == 1
    assert doc["panel_count"] == 3
    assert doc["reading_direction"] == "rtl"
    assert doc["source"].endswith("pages/page_001.jpg")
    assert [r["panel_id"] for r in doc["panels"]] == ["p001_001", "p001_002", "p001_003"]
    record = doc["panels"][0]
    assert record["reading_order"] == 1
    assert record["bbox"] == [8, 8, 90, 80]
    assert record["ocr"]["text"] == "text 1"
    assert record["visual"]["confidence"] == 0.8
    assert record["previous_panel"] is None
    assert record["next_panel"] is None
    assert record["scene_id"] is None
    assert state.pages.get("page_001") == "knowledge_completed"


def test_missing_ocr_represented_null(tmp_path):
    cfg = make_cfg(tmp_path)
    build_fixture(cfg, panels=2, ocr=False)
    result, _ = run(cfg)
    assert result["result"] == "ok"
    assert result["status"] == "partial"
    assert "OCR for p001_001" in result["missing"]
    doc = read_knowledge(cfg)
    assert doc["panels"][0]["ocr"] is None
    assert doc["panels"][0]["visual"] is not None


def test_missing_vlm_represented_null(tmp_path):
    cfg = make_cfg(tmp_path)
    build_fixture(cfg, panels=2, vlm=False)
    result, _ = run(cfg)
    assert result["result"] == "ok"
    assert result["status"] == "partial"
    assert "VLM analysis for p001_002" in result["missing"]
    doc = read_knowledge(cfg)
    assert doc["panels"][0]["visual"] is None
    assert doc["panels"][0]["ocr"] is not None


def test_missing_panel_metadata(tmp_path):
    cfg = make_cfg(tmp_path)
    build_fixture(cfg, panels=2, manifest=False)
    result, state = run(cfg)
    assert result["result"] == "error"
    assert "panel manifest" in result["message"]
    assert not knowledge_path(cfg, 1).exists()
    assert state.pages.get("page_001") is None


def test_invalid_json(tmp_path):
    cfg = make_cfg(tmp_path)
    build_fixture(cfg, panels=2, corrupt="panels")
    result, state = run(cfg)
    assert result["result"] == "error"
    assert "cannot read valid JSON" in result["message"]
    assert state.pages.get("page_001") is None
    cfg2 = make_cfg(tmp_path)
    build_fixture(cfg2, panels=2, corrupt="ocr")
    result2, _ = run(cfg2)
    assert result2["result"] == "error"
    assert "cannot read valid JSON" in result2["message"]


def test_invalid_bbox(tmp_path):
    cfg = make_cfg(tmp_path)
    build_fixture(cfg, panels=2, bad_bbox=True)
    result, state = run(cfg)
    assert result["result"] == "error"
    assert "bbox" in result["message"]
    assert state.pages.get("page_001") is None


def test_invalid_confidence(tmp_path):
    cfg = make_cfg(tmp_path)
    build_fixture(cfg, panels=2, bad_confidence=True)
    result, state = run(cfg)
    assert result["result"] == "error"
    assert "confidence" in result["message"]
    assert state.pages.get("page_001") is None


def test_incremental_update_only_changed_panel(tmp_path):
    cfg = make_cfg(tmp_path)
    build_fixture(cfg, panels=2)
    result, _ = run(cfg)
    assert result["result"] == "ok"
    before = read_knowledge(cfg)
    # OCR for panel 1 changes
    key = "page_001_panel_001.json"
    path = Path(cfg.output.ocr_dir) / key
    payload = json.loads(path.read_text("utf-8"))
    payload["combined_text"] = "changed text 1"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result2, _ = run(cfg)
    assert result2["result"] == "ok"
    assert result2["changed"] == ["p001_001"]
    after = read_knowledge(cfg)
    assert after["panels"][0]["ocr"]["text"] == "changed text 1"
    # unrelated panel untouched (identical content)
    assert after["panels"][1] == before["panels"][1]

    # no changes at all -> checkpointed page short-circuits, no rewrite
    mtime = knowledge_path(cfg, 1).stat().st_mtime_ns
    result3, _ = run(cfg)
    assert result3["result"] == "skipped"
    assert knowledge_path(cfg, 1).stat().st_mtime_ns == mtime


def test_only_target_page_rewritten(tmp_path):
    cfg = make_cfg(tmp_path)
    build_fixture(cfg, page=1, panels=2)
    run(cfg, page=1)
    page2_file = knowledge_path(cfg, 2)
    page2_file.parent.mkdir(parents=True, exist_ok=True)
    page2_file.write_text(
        json.dumps({"page": 2, "source": "x", "panel_count": 0,
                    "reading_direction": "rtl", "panels": []}),
        encoding="utf-8",
    )
    mtime1 = knowledge_path(cfg, 1).stat().st_mtime_ns
    mtime2 = page2_file.stat().st_mtime_ns
    run(cfg, page=1)  # unchanged -> page 1 not rewritten
    assert knowledge_path(cfg, 1).stat().st_mtime_ns == mtime1
    assert page2_file.stat().st_mtime_ns == mtime2


def test_global_index_update(tmp_path):
    cfg = make_cfg(tmp_path)
    build_fixture(cfg, page=1, panels=2)
    run(cfg, page=1)
    index = load_index(cfg)
    assert len(index["pages"]) == 1
    entry = index["pages"][0]
    assert entry == {"page": 1, "knowledge": str(knowledge_path(cfg, 1)), "status": "complete"}
    build_fixture(cfg, page=2, panels=1)
    run(cfg, page=2)
    index = load_index(cfg)
    assert [e["page"] for e in index["pages"]] == [1, 2]
    assert index_status(cfg, 1)["status"] == "complete"
    assert index_status(cfg, 2)["status"] == "complete"


def test_lazy_page_loading(tmp_path):
    cfg = make_cfg(tmp_path)
    build_fixture(cfg, page=1, panels=2)
    run(cfg, page=1)
    # page 1 loads fine even though page 2 never exists
    assert load_page_knowledge(cfg, 1)["page"] == 1
    with pytest.raises(KnowledgeError):
        load_page_knowledge(cfg, 2)
    assert load_index(cfg)["pages"]


def test_checkpoint_behavior(tmp_path):
    cfg = make_cfg(tmp_path / "a")
    build_fixture(cfg, panels=2)
    result, state = run(cfg)
    assert result["result"] == "ok"
    assert state.pages.get("page_001") == "knowledge_completed"
    result2, _ = run(cfg)
    assert result2["result"] == "skipped"
    # partial page must NOT be marked complete
    cfg3 = make_cfg(tmp_path / "b")
    build_fixture(cfg3, panels=2, vlm=False)
    result3, state3 = run(cfg3)
    assert result3["status"] == "partial"
    assert state3.pages.get("page_001") is None


def test_force_rebuild(tmp_path):
    cfg = make_cfg(tmp_path)
    build_fixture(cfg, panels=2)
    run(cfg)
    result, _ = run(cfg, force=True)
    assert result["result"] == "ok"
    assert result["changed"] == ["p001_001", "p001_002"]


# ------------------------------------------------------- validation units


def test_validate_units():
    assert validate_panel_id("p001_012") == "p001_012"
    with pytest.raises(KnowledgeError):
        validate_panel_id("bad")
    assert validate_bbox([1, 2, 3, 4]) == [1, 2, 3, 4]
    assert validate_bbox((1.9, 2, 3, 4)) == [1, 2, 3, 4]
    with pytest.raises(KnowledgeError):
        validate_bbox([1, 2, 3])
    with pytest.raises(KnowledgeError):
        validate_bbox([0, 0, -3, 4])
    with pytest.raises(KnowledgeError):
        validate_bbox(["a", 2, 3, 4])
    assert validate_confidence(0.0) == 0.0
    assert validate_confidence(1.0) == 1.0
    with pytest.raises(KnowledgeError):
        validate_confidence(1.5)
    with pytest.raises(KnowledgeError):
        validate_confidence("high")


def test_knowledge_loader_validates(tmp_path):
    cfg = make_cfg(tmp_path)
    doc = {
        "page": 1, "source": "x", "panel_count": 0,
        "reading_direction": "rtl", "panels": [],
    }
    validate_knowledge(doc)
    with pytest.raises(KnowledgeError):
        validate_knowledge({**doc, "panel_count": 1})
    with pytest.raises(KnowledgeError):
        validate_knowledge({**doc, "reading_direction": "up"})