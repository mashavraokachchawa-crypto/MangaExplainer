"""Final export (Task 24).

Produces the finished MP4 as output/final_video.mp4 with:
   * H.264 video          (libx264)
   * AAC audio
   * correct FPS and resolution (from video config)
   * narration synced with the mix
   * music/SFX mixed correctly (from pipeline/audio_mix.final_mix.wav)
   * no subtitles
   * no temporary files left in the output

Also writes output/video_info.json describing the exported file: duration,
resolution, FPS, video/audio codecs and file size.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

LOG = logging.getLogger(__name__)

FINAL_VIDEO_NAME = "final_video.mp4"
VIDEO_INFO_NAME = "video_info.json"

# Locked to the deliverable so the export is always H.264 + AAC.
EXPORT_CODEC = "libx264"
EXPORT_AUDIO_CODEC = "aac"


class ExportError(Exception):
    """Base error for the final export stage."""


class NoSourceAsset(ExportError):
    """The render plan or final audio is missing so nothing can be exported."""


def ffmpeg_bin():
    cand = os.environ.get("FFMPEG")
    if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
        return cand
    for c in ("ffmpeg", "/home/madhav/bin/ffmpeg", "/usr/bin/ffmpeg"):
        if Path(c).is_file() and os.access(c, os.X_OK):
            return c
        try:
            import shutil
            if shutil.which(c):
                return c
        except Exception:
            pass
    raise ExportError("ffmpeg not found")


def ffprobe_bin():
    fp = ffmpeg_bin()
    alt = fp.replace("ffmpeg", "ffprobe")
    if alt != fp and os.access(alt, os.X_OK):
        return alt
    for c in ("ffprobe", "/home/madhav/bin/ffprobe", "/usr/bin/ffprobe"):
        if Path(c).is_file() and os.access(c, os.X_OK):
            return c
        try:
            import shutil
            if shutil.which(c):
                return c
        except Exception:
            pass
    raise ExportError("ffprobe not found")


def probe_streams(video_path):
    """Return (format_entry, streams_list) via ffprobe, or raise ExportError."""
    fp = ffprobe_bin()
    cmd = [fp, "-v", "error", "-show_format", "-show_streams",
           "-of", "json", str(video_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ExportError(f"ffprobe failed on {video_path}: "
                          f"{proc.stderr[-400:]}")
    try:
        doc = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ExportError(f"cannot parse ffprobe output: {exc}")
    return (doc.get("format") or {}), (doc.get("streams") or [])


def final_video_path(root):
    return Path(root) / "output" / FINAL_VIDEO_NAME


def video_info_path(root):
    return Path(root) / "output" / VIDEO_INFO_NAME


def collect_video_info(root):
    """Collect the descriptor fields for output/video_info.json."""
    path = final_video_path(root)
    if not path.is_file():
        raise ExportError(f"no final video to describe: {path}")
    fmt, streams = probe_streams(path)
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    size = path.stat().st_size
    dur = fmt.get("duration") or video.get("duration")
    width = video.get("width")
    height = video.get("height")
    fps = _parse_frame_rate(video.get("avg_frame_rate"))
    return {
        "duration": round(float(dur), 3) if dur else None,
        "resolution":
            f"{width}x{height}" if width and height else None,
        "fps": fps,
        "video_codec": video.get("codec_name"),
        "audio_codec": audio.get("codec_name"),
        "file_size": size,
    }


def _parse_frame_rate(value):
    """Parse an ffprobe frame-rate ("10/1" or 30.0) to seconds^-1 or None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if float(value) > 0 else None
    if isinstance(value, str) and "/" in value:
        try:
            num, den = value.split("/")
            return float(num) / float(den) if float(den) else None
        except (ValueError, ZeroDivisionError):
            return None
    try:
        v = float(value)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def export_final(cfg, root, out_path=None, low_ram=None, fps_override=None,
                 cleanup=True, on_progress=None):
    """Render the final MP4 (H.264 + AAC) and write output/video_info.json.

    Delegates to the Task 22/23 renderer but pins the codecs and the output
    path to output/final_video.mp4, then records the probe info. Temp files
    are removed automatically by the renderer; this stage only leaves the two
    deliverables in output/.

    on_progress: optional callable(done, total) forwarded to the renderer.
    """
    from .video_render import render_video
    root = Path(root)
    target = Path(out_path) if out_path else final_video_path(root)
    try:
        result = render_video(cfg, root, out_path=target, low_ram=low_ram,
                              cleanup=cleanup, codec=EXPORT_CODEC,
                              fps_override=fps_override,
                              on_progress=on_progress)
    except Exception as exc:
        raise ExportError(f"render failed during export: {exc}") from exc

    # Deliverable rule: no temporary files/dirs may remain in output/.
    # The renderer removes frame files but leaves empty temp dirs; clean them.
    if cleanup:
        render_cfg = getattr(cfg, "render", None) or {}
        temp = root / str(render_cfg.get("temp_dir", "output/tmp"))
        try:
            for d in (temp / "frames", temp):
                if d.is_dir():
                    try:
                        d.rmdir()
                    except OSError:
                        pass
        except OSError:
            pass

    info = collect_video_info(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    info_path = video_info_path(root)
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    return {
        "result": "exported",
        "video": str(target),
        "video_info": str(info_path),
        "info": info,
    }
