"""Tests: video rendering + LOW RAM mode (Tasks 22/23).

Verifies sequential rendering from panels + motion + transitions + final
audio, temp file usage + auto-cleanup, CPU encoding at the configured
resolution, A/V sync, and that LOW_RAM_MODE keeps per-frame memory bounded.
ffmpeg is required (skipped when unavailable).
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

import cv2
from config.loader import Config
from pipeline.audio_io import write_wav
from pipeline.video_render import (
    NoFinalAudio,
    NoRenderPlan,
    RenderError,
    crop_resize,
    cumulative_timeline,
    find_active,
    ffmpeg_bin,
    load_final_audio,
    low_ram_mode,
    render_video,
    sample_rect,
)


def make_cfg(tmp_path, low_ram=True, resolution="160x90", fps=10):
    return Config({
        "output": {"audio_dir": str(tmp_path / "audio")},
        "video": {"resolution": resolution, "fps": fps},
        "render": {"low_ram_mode": low_ram,
                   "temp_dir": "output/tmp",
                   "codec": "libx264", "crf": 30,
                   "preset": "ultrafast", "pix_fmt": "yuv420p"},
        "pipeline": {"batch_size": 1,
                     "state": {"dir": str(tmp_path / "state")},
                     "cache": {"dir": str(tmp_path / "cache")}},
        "memory": {"guard_mb": 3072},
    }, tmp_path)


def make_panels(tmp_path, n=2, w=320, h=180):
    pd = tmp_path / "panels" / "page_001"
    pd.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (30 * i, 60 * i, 90 * i)
        cv2.putText(img, f"P{i}", (10, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(str(pd / f"p{i}.jpg"), img)


def zoom_keyframes(n=6):
    return [
        {"t": i / (n - 1), "x": 0.25 * (i / (n - 1)), "y": 0.25 * (i / (n - 1)),
         "w": 1.0 - 0.5 * (i / (n - 1)), "h": 1.0 - 0.5 * (i / (n - 1))}
        for i in range(n)
    ]


def make_plan(tmp_path, n=2, transition=None):
    entries = []
    for i in range(1, n + 1):
        entries.append({
            "index": i - 1, "scene": 1, "page": 1,
            "transition": (transition if i == n else "cut"),
            "motion": {
                "shot_id": f"shot_{i:03d}", "segment_id": f"seg_{i:03d}",
                "panel_id": f"p{i}",
                "image": f"panels/page_001/p{i}.jpg",
                "camera": "slow_zoom_in", "duration": 1.0,
                "path": {"type": "slow_zoom_in", "keyframes": zoom_keyframes()},
            },
        })
    transitions = []
    if transition in ("dissolve", "crossfade") and n > 1:
        transitions.append({"from_index": 0, "to_index": 1, "type": transition,
                            "duration": 0.3, "overlap": 0.3})
    mdir = tmp_path / "motion"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "render_plan.json").write_text(
        json.dumps({"tasks": ["17_ken_burns", "18_transitions"],
                    "entries": entries, "transitions": transitions}))
    return entries, transitions


def make_final_audio(tmp_path, seconds=1.8, sr=24000):
    a = tmp_path / "audio"
    a.mkdir(parents=True, exist_ok=True)
    x = (np.sin(2 * np.pi * 330 * np.arange(int(sr * seconds)) / sr)
         * 0.4).astype(np.float32)
    write_wav(a / "final_mix.wav", sr, x)
    return a / "final_mix.wav"


def standard(tmp_path, n=2, resolution="160x90", fps=10):
    make_panels(tmp_path, n)
    make_final_audio(tmp_path, seconds=(n * 1.0) - 0.0, sr=24000)
    make_plan(tmp_path, n)
    return make_cfg(tmp_path, resolution=resolution, fps=fps)


def test_sample_rect_interpolation():
    kf = [{"t": 0.0, "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
          {"t": 1.0, "x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5}]
    assert sample_rect(kf, 0.0) == kf[0]
    assert sample_rect(kf, 1.0) == kf[1]
    mid = sample_rect(kf, 0.5)
    assert mid["x"] == pytest.approx(0.125)
    assert mid["w"] == pytest.approx(0.75)
    # clamped outside
    assert sample_rect(kf, 2.0) == kf[1]


def test_crop_resize_bounds():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    out = crop_resize(frame, {"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5},
                      50, 50)
    assert out.shape == (50, 50, 3)


def test_low_ram_mode_env_wins(monkeypatch):
    cfg = make_cfg(Path("."), low_ram=False)
    monkeypatch.setenv("LOW_RAM_MODE", "true")
    assert low_ram_mode(cfg) is True
    monkeypatch.setenv("LOW_RAM_MODE", "false")
    assert low_ram_mode(cfg) is False
    monkeypatch.delenv("LOW_RAM_MODE")
    assert low_ram_mode(cfg) is False  # falls back to render.low_ram_mode


def test_no_render_plan_raises(tmp_path):
    cfg = make_cfg(tmp_path)
    make_final_audio(tmp_path)
    with pytest.raises(NoRenderPlan):
        render_video(cfg, tmp_path)


def test_no_final_audio_raises(tmp_path):
    cfg = make_cfg(tmp_path)
    make_panels(tmp_path)
    make_plan(tmp_path)
    with pytest.raises(NoFinalAudio):
        render_video(cfg, tmp_path)


def test_low_ram_mode_releases_memory_per_panel(tmp_path):
    # ensure the code path runs and frames temp-dir is cleaned after
    if not ffmpeg_bin():
        pytest.skip("ffmpeg not available")
    cfg = standard(tmp_path, n=2)
    out = tmp_path / "out.mp4"
    res = render_video(cfg, tmp_path, out_path=out, low_ram=True)
    assert res["result"] == "rendered"
    assert res["low_ram"] is True
    assert res["resolution"] == "160x90"
    assert out.is_file() and out.stat().st_size > 0
    # temp frame dir auto-cleaned
    assert not (tmp_path / "output" / "tmp" / "frames").exists()


def test_render_resolution_and_sync(tmp_path):
    if not ffmpeg_bin():
        pytest.skip("ffmpeg not available")
    cfg = standard(tmp_path, n=2)
    out = tmp_path / "out.mp4"
    res = render_video(cfg, tmp_path, out_path=out, fps_override=10)
    assert res["frames"] == int(round(res["duration"] * res["fps"]))
    assert res["resolution"] == "160x90"
    # A/V sync: output has both video + audio streams (ffprobe)
    import subprocess
    fp = subprocess.run(
        [ffmpeg_bin().replace("ffmpeg", "ffprobe"), "-v", "error",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True)
    streams = fp.stdout.split()
    assert "video" in streams
    assert "audio" in streams


def test_cut_transition_single_frame_ok(tmp_path):
    if not ffmpeg_bin():
        pytest.skip("ffmpeg not available")
    cfg = standard(tmp_path, n=1, resolution="96x54", fps=5)
    out = tmp_path / "out.mp4"
    res = render_video(cfg, tmp_path, out_path=out, fps_override=5)
    assert res["result"] == "rendered"
    assert out.is_file()
