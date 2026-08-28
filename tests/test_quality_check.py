"""Tests: video quality check (Task 25).

Verifies output/quality_report.json is produced and covers: video exists,
video can be opened, audio exists, A/V duration synchronised, no corrupted
frames, no missing panels, no missing narration, no subtitle track, readable
output. Errors are reported clearly (never silently continued).
"""

import json
import subprocess
from pathlib import Path

import pytest

from config.loader import Config
from pipeline.quality_check import (
    QualityCheckError,
    check_quality,
    read_narration_segments,
    report_path,
)

try:
    from pipeline.export import ffmpeg_bin, ffprobe_bin
    HAVE_FFMPEG = bool(ffmpeg_bin()) and bool(ffprobe_bin())
except Exception:
    HAVE_FFMPEG = False

# Reuse the export fixtures to build a real, valid final_video.mp4.
from tests.test_export import (  # noqa: E402
    make_final_audio,
    make_panels,
    make_plan,
    standard as export_standard,
)

from pipeline.export import export_final, final_video_path  # noqa: E402


def make_cfg(tmp_path):
    return Config({
        "output": {"audio_dir": str(tmp_path / "audio")},
        "video": {"resolution": "160x90", "fps": 10},
        "render": {"low_ram_mode": True, "temp_dir": "output/tmp",
                   "codec": "libx264", "crf": 30,
                   "preset": "ultrafast", "pix_fmt": "yuv420p"},
        "pipeline": {"batch_size": 1,
                     "state": {"dir": str(tmp_path / "state")},
                     "cache": {"dir": str(tmp_path / "cache")}},
        "memory": {"guard_mb": 3072},
    }, tmp_path)


def write_narration_manifest(tmp_path, ends):
    a = tmp_path / "audio"
    a.mkdir(parents=True, exist_ok=True)
    (a / "manifest.json").write_text(json.dumps(ends), encoding="utf-8")


def test_read_narration_segments(tmp_path):
    write_narration_manifest(tmp_path, [
        {"segment_id": "seg_001", "start_time": 0.0, "duration": 1.0},
        {"segment_id": "seg_002", "end_time": 2.0},
    ])
    segs = read_narration_segments(tmp_path / "audio")
    assert segs is not None
    assert len(segs) == 2
    assert max(s["end"] for s in segs) == 2.0


def test_read_narration_segments_missing(tmp_path):
    assert read_narration_segments(tmp_path / "audio") is None


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not available")
def test_quality_check_passes_on_valid_export(tmp_path):
    cfg = make_cfg(tmp_path)
    # build a valid final video
    export_standard(tmp_path)
    export_final(cfg, tmp_path, fps_override=10)
    write_narration_manifest(tmp_path, [
        {"segment_id": "seg_001", "start_time": 0.0, "end_time": 0.9},
        {"segment_id": "seg_002", "start_time": 0.9, "end_time": 1.8},
    ])
    report = check_quality(cfg, tmp_path)
    assert report["status"] == "ok"
    summary = report["summary"]
    # every required check passed
    for key in ("video_exists", "video_open", "audio_exists", "av_sync",
                "no_corrupt_frames", "no_missing_panels",
                "no_missing_narration", "no_subtitle_track",
                "output_readable"):
        assert summary.get(key) is True, f"{key} should pass: {report['checks']}"
    # report written to output/quality_report.json
    rp = report_path(tmp_path)
    assert rp.is_file()
    assert json.loads(rp.read_text("utf-8"))["status"] == "ok"


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not available")
def test_quality_check_detects_missing_video(tmp_path):
    cfg = make_cfg(tmp_path)
    report = check_quality(cfg, tmp_path)
    assert report["status"] == "error"
    assert report["summary"]["video_exists"] is False
    rp = report_path(tmp_path)
    assert rp.is_file()
    assert json.loads(rp.read_text("utf-8"))["status"] == "error"


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not available")
def test_quality_check_detects_missing_narration(tmp_path):
    cfg = make_cfg(tmp_path)
    export_standard(tmp_path)  # final video ~1.8s
    export_final(cfg, tmp_path, fps_override=10)
    # narration manifest claims audio runs much longer than the video
    write_narration_manifest(tmp_path, [
        {"segment_id": "seg_001", "start_time": 0.0, "end_time": 50.0},
    ])
    report = check_quality(cfg, tmp_path)
    assert report["summary"].get("no_missing_narration") is False
    assert report["status"] == "error"
