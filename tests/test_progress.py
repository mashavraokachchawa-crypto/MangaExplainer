"""Unit tests for the live per-step progress reporter (pipeline/progress.py)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.progress import PROGRESS_FILE, Progress, read_progress


def test_begin_writes_stage(tmp_path):
    p = Progress(tmp_path, state_dir=tmp_path, throttle=0.0)
    p.begin("extract_pages", "Panel extraction")
    doc = json.loads((tmp_path / PROGRESS_FILE).read_text("utf-8"))
    assert doc["stage"] == "extract_pages"
    assert doc["label"] == "Panel extraction"
    assert doc["phase"] is None
    assert doc["pct"] == 0


def test_step_writes_counts_and_pct(tmp_path):
    p = Progress(tmp_path, state_dir=tmp_path, throttle=0.0)
    p.begin("extract_pages", "Panel extraction")
    p.step("extract_pages", 3, 10, phase="Rendering page 3 of 10")
    doc = json.loads((tmp_path / PROGRESS_FILE).read_text("utf-8"))
    assert doc["done"] == 3 and doc["total"] == 10
    assert doc["pct"] == 30
    assert doc["phase"] == "Rendering page 3 of 10"


def test_phase_only_step(tmp_path):
    p = Progress(tmp_path, state_dir=tmp_path, throttle=0.0)
    p.begin("camera_motion", "Camera movement")
    p.phase("camera_motion", "Planning camera paths")
    doc = read_progress(state_dir=tmp_path)
    assert doc["stage"] == "camera_motion"
    assert doc["phase"] == "Planning camera paths"
    assert doc["done"] is None and doc["total"] is None


def test_read_progress_none_when_missing(tmp_path):
    assert read_progress(state_dir=tmp_path) is None


def test_clear(tmp_path):
    p = Progress(tmp_path, state_dir=tmp_path, throttle=0.0)
    p.begin("music", "Music")
    assert (tmp_path / PROGRESS_FILE).is_file()
    p.clear()
    assert not (tmp_path / PROGRESS_FILE).exists()
    assert read_progress(state_dir=tmp_path) is None


def test_bad_json_returns_none(tmp_path):
    (tmp_path / PROGRESS_FILE).write_text("{not json!!", encoding="utf-8")
    assert read_progress(state_dir=tmp_path) is None


def test_read_progress_defaults_to_own_state_dir(tmp_path):
    p = Progress(tmp_path)
    p.phase("sfx", "Loading SFX events")
    assert read_progress(root=tmp_path)["stage"] == "sfx"


def test_override_state_dir(tmp_path):
    p = Progress(tmp_path, state_dir=tmp_path / "custom")
    p.begin("render_video", "Video rendering")
    doc = read_progress(state_dir=tmp_path / "custom")
    assert doc["stage"] == "render_video"
    assert read_progress(root=tmp_path) is None  # default target untouched


def test_pct_caps_and_rounds(tmp_path):
    p = Progress(tmp_path, state_dir=tmp_path, throttle=0.0)
    p.begin("understand_panels", "Understand")
    p.step("understand_panels", 5, 3)  # beyond total
    doc = json.loads((tmp_path / PROGRESS_FILE).read_text("utf-8"))
    assert doc["pct"] == 100
    p.step("understand_panels", 1, 4, item="p")  # partial
    doc = json.loads((tmp_path / PROGRESS_FILE).read_text("utf-8"))
    assert doc["pct"] == 25
    assert doc["item"] == "p"


def test_throttle_blocks_intermediate_writes(tmp_path):
    """With a large throttle the file keeps the begin() state until it elapses."""
    p = Progress(tmp_path, state_dir=tmp_path, throttle=100)
    p.begin("voice_timing", "Voice timing")
    p.step("voice_timing", 7, 12)
    p.step("voice_timing", 9, 12)
    doc = read_progress(state_dir=tmp_path)
    assert doc["done"] is None and doc["total"] is None  # throttled out


def test_throttled_loop_stays_consistent(tmp_path):
    """A fast inner frame loop cannot corrupt the file; it always reads back."""
    p = Progress(tmp_path, state_dir=tmp_path)  # default 0.4s throttle
    p.begin("render_video", "Video rendering")
    for i in range(1, 2001):
        p.step("render_video", i, 2000, phase=f"Rendering frame {i} of 2000")
    doc = json.loads((tmp_path / PROGRESS_FILE).read_text("utf-8"))
    assert doc["stage"] == "render_video"
    assert 0 <= doc["pct"] <= 100
    assert read_progress(state_dir=tmp_path)["stage"] == "render_video"


def test_label_defaults_to_stage(tmp_path):
    p = Progress(tmp_path, state_dir=tmp_path, throttle=0.0)
    p.phase("voice_timing", "Aligning narration timing")
    doc = read_progress(state_dir=tmp_path)
    assert doc["label"] == "voice_timing"


def test_step_does_not_require_begin(tmp_path):
    p = Progress(tmp_path, state_dir=tmp_path, throttle=0.0)
    p.step("write_script", 2, 8)
    doc = read_progress(state_dir=tmp_path)
    assert doc["stage"] == "write_script"
    assert doc["pct"] == 25