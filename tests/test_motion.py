"""Tests: Ken Burns motion + panel transitions (Tasks 17/18).

Builds a tiny synthetic panel image + a shots timeline + a panels manifest
in a temp tree and verifies motion/render_plan.json: smooth clamped camera
paths (zoom in/out, L-R/R-L pan, vertical), narration-driven durations, short
timing-dependent transitions (cut/fade/dissolve), no image conversion, no
subtitles.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import cv2
from config.loader import Config
from pipeline.motion import (
    DEFAULT_TRANSITION,
    DEFAULT_TRANSITION_FRACTION,
    DEFAULT_TRANSITION_MAX,
    MotionError,
    NoImageData,
    NoPanelData,
    NoTimelineData,
    assemble_transitions,
    build_render_plan,
    build_transition,
    kenburns_path,
    render_plan_path,
    run_motion,
    smoothstep,
    transition_length,
)

ROOT = Path(__file__).resolve().parent.parent


def make_cfg(tmp_path, motion=None):
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
            "audio_dir": str(tmp_path / "audio"),
            "shots_dir": str(tmp_path / "shots"),
            "crops_dir": str(tmp_path / "crops"),
        },
        "images": {"format": "jpg", "render_scale": 1.0, "jpeg_quality": 80},
        "video": {"resolution": "1920x1080", "fps": 30},
        "pipeline": {"batch_size": 1,
                     "state": {"dir": str(tmp_path / "state")},
                     "cache": {"dir": str(tmp_path / "state" / "cache")}},
        "memory": {"guard_mb": 3072},
        "logging": {"level": "INFO", "console_level": "WARNING",
                    "log_dir": str(tmp_path / "logs"),
                    "max_bytes": 1048576, "backup_count": 3},
    }
    if motion is not None:
        data["motion"] = motion
    return Config(data, tmp_path)


def make_panel_image(tmp_path, panel_id, width=1200, height=1800):
    d = tmp_path / "panels" / "page_001"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{panel_id}.jpg"
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (90, 90, 90)
    cv2.imwrite(str(path), img)
    return path


def write_panels_manifest(tmp_path, *panels):
    d = tmp_path / "visuals"
    d.mkdir(parents=True, exist_ok=True)
    (d / "panels_manifest.json").write_text(
        json.dumps(list(panels), indent=2), encoding="utf-8"
    )
    return d / "panels_manifest.json"


def shot(shot_id, panel, dur, camera="static", transition=None):
    s = {
        "shot_id": shot_id,
        "segment_id": "seg_001",
        "primary_panel": panel,
        "panel_ids": [panel],
        "visual_intent": "full_panel",
        "camera": {"type": camera, "start": 1.0, "end": 1.0},
        "estimated_duration": dur,
        "match_score": 0.9,
        "needs_review": False,
        "reuse_count": 1,
    }
    if transition is not None:
        s["transition"] = transition
    return s


def write_timeline(tmp_path, page=1, scene=1, shots=None, transitions=None):
    d = tmp_path / "shots"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"page_{page:03d}_scene_{scene:03d}_timeline.json"
    payload = {"page": page, "scene_id": scene, "shots": shots or []}
    if transitions is not None:
        payload["transitions"] = transitions
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def standard_setup(tmp_path):
    """Panel image + panels manifest + a 3-shot timeline with cameras."""
    make_panel_image(tmp_path, "p001_001", 1200, 1800)
    make_panel_image(tmp_path, "p001_002", 1200, 1800)
    make_panel_image(tmp_path, "p001_003", 1200, 1800)
    write_panels_manifest(
        tmp_path,
        {"panel_id": "p001_001", "page": 1,
         "image": "panels/page_001/p001_001.jpg",
         "width": 1200, "height": 1800, "aspect_ratio": 0.666667},
        {"panel_id": "p001_002", "page": 1,
         "image": "panels/page_001/p001_002.jpg",
         "width": 1200, "height": 1800, "aspect_ratio": 0.666667},
        {"panel_id": "p001_003", "page": 1,
         "image": "panels/page_001/p001_003.jpg",
         "width": 1200, "height": 1800, "aspect_ratio": 0.666667},
    )
    write_timeline(
        tmp_path, 1, 1,
        shots=[
            shot("shot_001", "p001_001", 4.0, "slow_zoom_in"),
            shot("shot_002", "p001_002", 3.0, "pan_right"),
            shot("shot_003", "p001_003", 2.5, "static", transition="fade"),
        ],
    )
    return make_cfg(tmp_path)


# ------------------------------------------------------------- smoothstep
def test_smoothstep_bounds_and_shape():
    assert smoothstep(0.0) == 0.0
    assert smoothstep(1.0) == 1.0
    assert smoothstep(0.5) == 0.5
    # monotone non-decreasing on a grid
    prev = -1.0
    for i in range(101):
        v = smoothstep(i / 100.0)
        assert v >= prev
        prev = v
    # clamped outside [0,1]
    assert smoothstep(-1) == 0.0
    assert smoothstep(2) == 1.0


# --------------------------------------------------------- Task 17 - Ken Burns
@pytest.mark.parametrize("camera", [
    "static", "slow_zoom_in", "slow_zoom_out",
    "pan_left", "pan_right", "pan_up", "pan_down",
])
def test_kenburns_path_all_cameras_clamped_smooth(camera):
    img_w, img_h = 1200, 1800
    path = kenburns_path(camera, img_w, img_h, 4.0, num_keyframes=20)
    assert path["type"] == camera
    assert path["duration"] == 4.0
    kf = path["keyframes"]
    assert len(kf) == 20
    # timestamps strictly increasing, rects stay inside the panel (no bleed)
    prev_t = -1.0
    for k in kf:
        assert k["t"] > prev_t - 1e-9
        prev_t = k["t"]
        assert k["x"] >= 0 and k["y"] >= 0
        assert k["x"] + k["w"] <= 1.0 + 1e-6
        assert k["y"] + k["h"] <= 1.0 + 1e-6
        assert k["w"] > 0 and k["h"] > 0


def test_kb_min_coverage_respected():
    path = kenburns_path("slow_zoom_in", 1200, 1800, 4.0,
                         num_keyframes=30, min_coverage=0.5)
    min_w = min(k["w"] for k in path["keyframes"])
    assert min_w >= 0.5 - 1e-6  # never zooms in past min_coverage


def test_kb_smoothness_no_jumps():
    path = kenburns_path("pan_right", 1200, 1800, 4.0, num_keyframes=50)
    xs = [k["x"] for k in path["keyframes"]]
    # a smooth L-R pan is monotone non-decreasing (no jitter/backtrack)
    assert all(xs[i] <= xs[i + 1] + 1e-9 for i in range(len(xs) - 1))
    assert xs[0] < xs[-1]  # it actually moves


def test_kb_zoom_in_start_full_panel_end_zoomed():
    path = kenburns_path("slow_zoom_in", 1200, 1800, 4.0, num_keyframes=5)
    first, last = path["keyframes"][0], path["keyframes"][-1]
    assert first["w"] == 1.0 and first["h"] == 1.0  # starts full panel
    assert last["w"] < 1.0 and last["h"] < 1.0        # zooms in
    # centred about the panel
    assert abs(first["x"] + first["w"] / 2.0 - 0.5) < 1e-6
    assert abs(last["x"] + last["w"] / 2.0 - 0.5) < 1e-6


def test_kb_unknown_camera_falls_back_to_static():
    path = kenburns_path("extreme_zoom", 1200, 1800, 3.0, num_keyframes=4)
    assert path["type"] == "static"
    assert all(k["w"] == 1.0 for k in path["keyframes"])


def test_kb_uses_narration_duration():
    # duration comes through unchanged ("narration determines panel duration")
    path = kenburns_path("slow_zoom_in", 1200, 1800, 7.25, num_keyframes=6)
    assert path["duration"] == 7.25


def test_kb_non_positive_duration_safe():
    path = kenburns_path("slow_zoom_in", 1200, 1800, 0.0, num_keyframes=4)
    assert path["duration"] == 0.0
    assert len(path["keyframes"]) == 4


# ------------------------------------------------------- Task 18 - transitions
def test_transition_cut_is_zero_length():
    t = build_transition(4.0, 4.0, "cut")
    assert t["type"] == "cut"
    assert t["duration"] == 0.0
    assert t["overlap"] == 0.0


def test_transition_length_short_and_timing_dependent():
    # long shots -> capped at transition_max (short)
    assert transition_length(20.0, 20.0) <= DEFAULT_TRANSITION_MAX
    # short shots -> fraction of the shorter one
    length = transition_length(2.0, 2.0)
    assert length <= DEFAULT_TRANSITION_FRACTION * 2.0
    # never more than half the shorter adjacent shot
    assert transition_length(1.0, 1.0) <= 0.5


def test_transition_fade_and_dissolve():
    fade = build_transition(4.0, 4.0, "fade")
    assert fade["type"] == "fade"
    assert fade["overlap"] == 0.0
    assert 0 < fade["duration"] <= DEFAULT_TRANSITION_MAX
    diss = build_transition(4.0, 4.0, "dissolve")
    assert diss["type"] == "dissolve"
    assert diss["duration"] == diss["overlap"]
    assert 0 < diss["duration"] <= DEFAULT_TRANSITION_MAX


def test_transition_default_cut_and_config_tuning():
    assert build_transition(4.0, 4.0, "warp")["type"] == DEFAULT_TRANSITION
    cfg = make_cfg(Path("."), motion={"transition_max": 0.3,
                                      "transition_fraction": 0.5})
    diss = build_transition(10.0, 10.0, "dissolve", cfg)
    assert diss["duration"] == 0.3  # capped by tuned max
    assert diss["overlap"] == 0.3


# ---------------------------------------------------- end-to-end render plan
def test_build_render_plan_entries_and_motion(tmp_path):
    cfg = standard_setup(tmp_path)
    entries = build_render_plan(cfg, tmp_path, keyframes=12)
    assert len(entries) == 3
    first = entries[0]
    assert first["index"] == 0
    assert first["motion"]["shot_id"] == "shot_001"
    assert first["motion"]["panel_id"] == "p001_001"
    assert first["motion"]["duration"] == 4.0
    assert first["transition"] == "cut"  # not specified -> default
    # Ken Burns path present and smooth
    assert first["motion"]["path"]["type"] == "slow_zoom_in"
    assert len(first["motion"]["path"]["keyframes"]) == 12
    # entry 3 requested a fade transition
    assert entries[2]["transition"] == "fade"


def test_assemble_transitions_alignment(tmp_path):
    cfg = standard_setup(tmp_path)
    entries = build_render_plan(cfg, tmp_path)
    trans = assemble_transitions(entries, cfg)
    assert len(trans) == 2  # 3 entries -> 2 transitions
    assert trans[0]["from_shot"] == "shot_001"
    assert trans[0]["to_shot"] == "shot_002"
    assert trans[0]["type"] == "cut"
    assert trans[1]["from_shot"] == "shot_002"
    assert trans[1]["to_shot"] == "shot_003"
    assert trans[1]["type"] == "fade"
    assert trans[1]["duration"] > 0
    assert trans[1]["overlap"] == 0.0


def test_no_subtitles_or_image_conversion(tmp_path):
    cfg = standard_setup(tmp_path)
    img_path = tmp_path / "panels" / "page_001" / "p001_001.jpg"
    before = img_path.stat().st_mtime_ns
    run_motion(cfg, tmp_path)
    assert img_path.stat().st_mtime_ns == before  # panels untouched
    plan = json.loads(render_plan_path(cfg, tmp_path).read_text("utf-8"))
    raw = render_plan_path(cfg, tmp_path).read_text("utf-8")
    # entries reference original image paths, never generated copies
    assert "panels/page_001/p001_001.jpg" in raw
    # no subtitles anywhere
    assert "subtitle" not in raw.lower()
    # no rendered video/png frames produced
    assert plan["tasks"] == ["17_ken_burns", "18_transitions"]


def test_run_motion_writes_plan(tmp_path):
    cfg = standard_setup(tmp_path)
    res = run_motion(cfg, tmp_path)
    assert res["result"] == "computed"
    assert res["entries"] == 3
    assert res["transitions"] == 2
    plan_path = render_plan_path(cfg, tmp_path)
    assert plan_path.is_file()
    plan = json.loads(plan_path.read_text("utf-8"))
    assert len(plan["entries"]) == 3
    assert len(plan["transitions"]) == 2
    # timing chain: durations carried from narration
    durations = [e["motion"]["duration"] for e in plan["entries"]]
    assert durations == [4.0, 3.0, 2.5]


def test_no_timeline_raises(tmp_path):
    cfg = make_cfg(tmp_path)
    with pytest.raises(NoTimelineData):
        run_motion(cfg, tmp_path)


def test_missing_panel_in_manifest_raises(tmp_path):
    make_panel_image(tmp_path, "p001_001", 1200, 1800)
    write_panels_manifest(
        tmp_path,
        {"panel_id": "p001_001", "page": 1,
         "image": "panels/page_001/p001_001.jpg",
         "width": 1200, "height": 1800, "aspect_ratio": 0.666667},
    )
    write_timeline(tmp_path, 1, 1,
                   shots=[shot("shot_001", "p001_002", 4.0, "static")])
    cfg = make_cfg(tmp_path)
    with pytest.raises(NoPanelData):
        run_motion(cfg, tmp_path)


def test_missing_image_file_raises(tmp_path):
    # manifest references an image that does not exist on disk
    write_panels_manifest(
        tmp_path,
        {"panel_id": "p001_001", "page": 1,
         "image": "panels/page_001/p001_001.jpg",
         "width": 1200, "height": 1800, "aspect_ratio": 0.666667},
    )
    write_timeline(tmp_path, 1, 1,
                   shots=[shot("shot_001", "p001_001", 4.0, "static")])
    cfg = make_cfg(tmp_path)
    with pytest.raises(NoImageData):
        run_motion(cfg, tmp_path)


def test_dimensions_reader_fallback(tmp_path):
    # manifest omits width/height -> must read the image dimensions with cv2
    make_panel_image(tmp_path, "p001_001", 640, 360)
    write_panels_manifest(
        tmp_path,
        {"panel_id": "p001_001", "page": 1,
         "image": "panels/page_001/p001_001.jpg"},  # no width/height
    )
    write_timeline(tmp_path, 1, 1,
                   shots=[shot("shot_001", "p001_001", 2.0, "pan_right")])
    cfg = make_cfg(tmp_path)
    entries = build_render_plan(cfg, tmp_path, keyframes=8)
    assert entries[0]["motion"]["path"]["type"] == "pan_right"
    # 640x360 panel, pan_coverage 0.8 -> rect height 0.8
    assert entries[0]["motion"]["path"]["keyframes"][0]["h"] == 0.8
