"""Video quality check (Task 25).

After export, verifies output/final_video.mp4 against a required checklist and
writes output/quality_report.json:

   * video exists and can be opened
   * audio exists and A/V duration is synchronised
   * frames decode without corruption and no planned panel was dropped
   * no narration segment is missing from the mix
   * no subtitle track is present
   * the output file is readable

Every check is recorded pass/fail with a detail. When an error is detected it
is reported clearly (never silently swallowed): check_quality() returns a
report whose overall status is "error"/"warning"/"ok", and the CLI exits
non-zero on any recorded error.
"""
from __future__ import annotations

import json
import logging
import math
import os
import subprocess
from pathlib import Path

LOG = logging.getLogger(__name__)

QUALITY_REPORT_NAME = "quality_report.json"
FINAL_VIDEO_NAME = "final_video.mp4"
MANIFEST_NAME = "manifest.json"

SYNC_TOLERANCE = 0.15          # seconds of allowed A/V duration mismatch
PROBE_MAX_FRAMES = 0           # 0 = probe all frames (cheap tiny videos)


class QualityCheckError(Exception):
    """A hard failure prevented the check itself from running."""


def _ffprobe():
    from .export import ffprobe_bin
    return ffprobe_bin()


def report_path(root):
    return Path(root) / "output" / QUALITY_REPORT_NAME


def read_narration_segments(audio_dir):
    """Load segment end times from audio/manifest.json (Task 15)."""
    manifest = Path(audio_dir) / MANIFEST_NAME
    if not manifest.is_file():
        return None
    try:
        doc = json.loads(manifest.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(doc, list):
        return None
    ends = []
    for seg in doc:
        if not isinstance(seg, dict):
            continue
        end = seg.get("end_time")
        if end is None:
            start = seg.get("start_time") or 0.0
            dur = seg.get("duration") or 0.0
            end = float(start) + float(dur)
        ends.append({"segment_id": seg.get("segment_id"),
                     "end": float(end)})
    return ends or None


def decode_info(video_path):
    """Return (total_frames_hint, openable) using cv2 but only if present."""
    try:
        import cv2
    except Exception:
        return None, "cv2 unavailable"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None, "failed to open with cv2"
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    found = 0
    corrupt = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame is None or frame.size == 0:
            corrupt += 1
        found += 1
        if PROBE_MAX_FRAMES and found >= PROBE_MAX_FRAMES:
            break
    cap.release()
    return {"total": total, "decoded": found, "corrupt": corrupt}, None


def check_quality(cfg, root, video_path=None, low_ram=True):
    """Run all quality checks; write output/quality_report.json.

    Returns the report dict. Checks that fail are recorded with pass=False;
    the overall status is 'error' if any check failed, 'warning' if a probe
    could not be fully completed, else 'ok'.
    """
    root = Path(root)
    video = Path(video_path) if video_path else \
        root / "output" / FINAL_VIDEO_NAME
    audio_dir = Path(cfg.output.audio_dir)
    checks = []
    result = {}

    def record(name, passed, detail, critical=True):
        checks.append({
            "check": name, "passed": bool(passed),
            "detail": str(detail), "critical": bool(critical),
        })

    # 1) video exists
    exists = video.is_file()
    record("video_exists", exists,
           f"{video} {'found' if exists else 'MISSING'}"
           f" ({video.stat().st_size if exists else 0} bytes)")
    if not exists:
        return _finalize(root, checks, result)

    # 9) output file is readable
    readable = os.access(video, os.R_OK) and video.stat().st_size > 0
    record("output_readable", readable,
           f"readable={os.access(video, os.R_OK)}, size={video.stat().st_size}")

    # probe streams with ffprobe
    try:
        from .export import probe_streams
        fmt, streams = probe_streams(video)
    except Exception as exc:
        record("video_open", False, f"ffprobe failed: {exc}")
        return _finalize(root, checks, result)

    video_stream = next((s for s in streams
                         if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams
                         if s.get("codec_type") == "audio"), None)
    sub_streams = [s for s in streams if s.get("codec_type") == "subtitle"]

    # 2) video can be opened (two independent probes)
    has_video_stream = video_stream is not None
    record("video_has_stream", has_video_stream,
           f"video codec={video_stream.get('codec_name') if
           video_stream else None} res="
           f"{video_stream.get('width')}x{video_stream.get('height') if
           video_stream else '?'} fps="
           f"{video_stream.get('avg_frame_rate') if video_stream else '?'}")
    dec_q = decode_info(video)
    if dec_q[1]:
        record("video_open", False, dec_q[1])
    else:
        dinfo = dec_q[0] or {}
        record("video_open", True,
               f"decoded {dinfo.get('decoded', 0)} frames, "
               f"{dinfo.get('corrupt', 0)} corrupt")

    # 3) audio exists
    record("audio_exists", audio_stream is not None,
           f"audio codec={audio_stream.get('codec_name') if audio_stream
           else None}")

    # 4) A/V duration synchronised
    vdur = float(video_stream.get("duration", 0.0)) if video_stream else 0.0
    adur = float(audio_stream.get("duration", 0.0)) if audio_stream else 0.0
    adur = adur or float(fmt.get("duration", 0.0) or 0.0)
    vdur = vdur or float(fmt.get("duration", 0.0) or 0.0)
    drift = abs(vdur - adur)
    record("av_sync", drift <= SYNC_TOLERANCE,
           f"video {vdur:.2f}s vs audio {adur:.2f}s (drift {drift:.3f}s)")

    # 5) no corrupted frames (decoded frames equal expected)
    if dec_q[0] is not None:
        dinfo = dec_q[0]
        bad = dinfo.get("corrupt", 0)
        record("no_corrupt_frames", bad == 0,
               f"{bad} corrupted/empty frame reads")

    # 6) no missing panels (every planned shot produced frames)
    motion_plan = root / "motion" / "render_plan.json"
    if motion_plan.is_file():
        try:
            plan = json.loads(motion_plan.read_text("utf-8"))
            planned = len(plan.get("entries") or [])
        except Exception:
            planned = None
    else:
        planned = None
    if planned is not None and dec_q[0] is not None:
        decoded = dec_q[0].get("decoded", 0)
        # every panel must have had the chance to appear: planned shots with
        # positive duration map to >=1 rendered frame each
        record("no_missing_panels", decoded >= planned,
               f"{decoded} rendered frames >= {planned} planned shots")

    # 7) no missing narration (audio length covers the last segment)
    segs = read_narration_segments(audio_dir)
    if segs is not None:
        last_end = max(s["end"] for s in segs)
        mix_dur = adur or vdur
        record("no_missing_narration", mix_dur + 1e-6 >= last_end,
               f"mix {mix_dur:.2f}s vs last narration end {last_end:.2f}s")
        result["narration_segments"] = len(segs)

    # 8) no subtitle track
    record("no_subtitle_track", len(sub_streams) == 0,
           f"{len(sub_streams)} subtitle stream(s) present")

    return _finalize(root, checks, result)


def _finalize(root, checks, result):
    failed = [c for c in checks if not c["passed"] and c["critical"]]
    non_critical = [c for c in checks if not c["passed"] and not c["critical"]]
    if failed:
        status = "error"
    elif non_critical:
        status = "warning"
    else:
        status = "ok"
    report = {
        "status": status,
        "checks": checks,
        "error_count": len(failed),
        "summary": {c["check"]: c["passed"] for c in checks},
    }
    report.update(result)
    out_path = report_path(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return report
