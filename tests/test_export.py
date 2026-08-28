"""Tests: final export (Task 24).

Verifies export produces output/final_video.mp4 (H.264 + AAC, correct FPS /
resolution, synced narration/music/SFX) plus output/video_info.json, with no
temporary files left in the output. ffmpeg required (skipped when missing).
"""

import json
from pathlib import Path

import numpy as np
import pytest

import cv2
from config.loader import Config
from pipeline.audio_io import write_wav
from pipeline.export import (
    EXPORT_AUDIO_CODEC,
    EXPORT_CODEC,
    ExportError,
    collect_video_info,
    export_final,
    final_video_path,
    video_info_path,
)

try:
    from pipeline.export import ffmpeg_bin, ffprobe_bin
    HAVE_FFMPEG = bool(ffmpeg_bin()) and bool(ffprobe_bin())
except Exception:
    HAVE_FFMPEG = False


def make_cfg(tmp_path, resolution="160x90", fps=10):
    return Config({
        "output": {"audio_dir": str(tmp_path / "audio")},
        "audio_dir": str(tmp_path / "audio"),
        "video": {"resolution": resolution, "fps": fps},
        "render": {"low_ram_mode": True, "temp_dir": "output/tmp",
                   "codec": "libx264", "crf": 30,
                   "preset": "ultrafast", "pix_fmt": "yuv420p"},
        "pipeline": {"batch_size": 1,
                     "state": {"dir": str(tmp_path / "state")},
                     "cache": {"dir": str(tmp_path / "cache")}},
        "memory": {"guard_mb": 3072},
    }, tmp_path)


def zoom_keyframes(n=6):
    return [{"t": i / (n - 1),
             "x": 0.25 * (i / (n - 1)), "y": 0.25 * (i / (n - 1)),
             "w": 1.0 - 0.5 * (i / (n - 1)),
             "h": 1.0 - 0.5 * (i / (n - 1))} for i in range(n)]


def make_panels(tmp_path, n=2, w=160, h=90):
    pd = tmp_path / "panels" / "page_001"
    pd.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (40 * i, 60 * i, 80 * i)
        cv2.imwrite(str(pd / f"p{i}.jpg"), img)


def make_plan(tmp_path, n=2):
    mdir = tmp_path / "motion"
    mdir.mkdir(parents=True, exist_ok=True)
    entries = [{
        "index": i - 1, "scene": 1, "page": 1, "transition": "cut",
        "motion": {
            "shot_id": f"shot_{i:03d}", "segment_id": f"seg_{i:03d}",
            "panel_id": f"p{i}", "image": f"panels/page_001/p{i}.jpg",
            "camera": "slow_zoom_in", "duration": 0.9,
            "path": {"type": "slow_zoom_in", "keyframes": zoom_keyframes()},
        },
    } for i in range(1, n + 1)]
    (mdir / "render_plan.json").write_text(
        json.dumps({"tasks": ["17_ken_burns", "18_transitions"],
                    "entries": entries, "transitions": []}))
    return entries


def make_final_audio(tmp_path, seconds=1.7, sr=24000):
    a = tmp_path / "audio"
    a.mkdir(parents=True, exist_ok=True)
    x = (np.sin(2 * np.pi * 330 * np.arange(int(sr * seconds)) / sr)
         * 0.4).astype(np.float32)
    write_wav(a / "final_mix.wav", sr, x)
    return a / "final_mix.wav"


def standard(tmp_path, n=2):
    make_panels(tmp_path, n, 160, 90)
    make_final_audio(tmp_path, seconds=n * 0.9)
    make_plan(tmp_path, n)
    return make_cfg(tmp_path)


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not available")
def test_export_produces_final_video_and_info(tmp_path):
    cfg = standard(tmp_path)
    out = export_final(cfg, tmp_path, fps_override=10)
    assert out["result"] == "exported"
    vp = final_video_path(tmp_path)
    assert vp.is_file() and vp.stat().st_size > 0
    info_path = video_info_path(tmp_path)
    assert info_path.is_file()
    info = json.loads(info_path.read_text("utf-8"))
    assert info["video_codec"].startswith("h264")
    assert info["audio_codec"].startswith("aac")
    assert info["resolution"] == "160x90"
    assert info["fps"] == 10
    assert info["duration"] > 0
    assert info["file_size"] > 0


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not available")
def test_export_no_temp_files_in_output(tmp_path):
    cfg = standard(tmp_path)
    export_final(cfg, tmp_path, fps_override=10)
    # only the two deliverables remain in output/
    out_dir = tmp_path / "output"
    names = [p.name for p in out_dir.iterdir() if p.is_file()]
    assert "final_video.mp4" in names
    assert "video_info.json" in names
    # no stray frame jpgs or temp videos
    assert not any(n.startswith("frame_") or n.endswith(".jpg") for n in names)
    assert not any("noaudio" in n or n.startswith("_") for n in names)


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not available")
def test_collect_video_info_matches_actual(tmp_path):
    cfg = standard(tmp_path)
    export_final(cfg, tmp_path, fps_override=10)
    info = collect_video_info(tmp_path)
    assert info["resolution"] == "160x90"
    assert info["fps"] == 10
    assert info["video_codec"] == "h264"
    assert info["audio_codec"] == "aac"
    # duration consistent for 2 panels at 0.9s each (small padding tolerance)
    assert abs(info["duration"] - 1.8) < 0.3


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not available")
def test_export_sync_narration_music_sfx(tmp_path):
    # audio stream length must match video length (A/V sync from the mix)
    cfg = standard(tmp_path)
    export_final(cfg, tmp_path, fps_override=10)
    info = collect_video_info(tmp_path)
    import subprocess
    from pipeline.export import ffprobe_bin
    fp = subprocess.run(
        [ffprobe_bin(), "-v", "error", "-show_entries",
         "format=duration", "-of", "default=nk=1:nw=1",
         str(final_video_path(tmp_path))],
        capture_output=True, text=True)
    fmt_dur = float(fp.stdout.strip())
    assert abs(fmt_dur - info["duration"]) < 0.1


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not available")
def test_export_error_no_plan(tmp_path):
    cfg = make_cfg(tmp_path)
    make_final_audio(tmp_path)
    with pytest.raises(ExportError):
        export_final(cfg, tmp_path)
