"""Video rendering (Tasks 22 + 23).

Renders the final video from:
   * manga panels            (visuals/panels_manifest.json image paths)
   * panel motion (Task 17)  (motion/render_plan.json Ken Burns keyframes)
   * final audio (Task 21)   (audio/final_mix.wav)
into an H.264 MP4 at the configured resolution (default 1920x1080).

Transitions were REMOVED from the pipeline (panels are hard-cut only). The
renderer still tolerates legacy plans that carry a "transitions" key so old
artifacts keep rendering, but new plans never produce one and every shot is
a cut.
Design (lightweight, low-RAM):
   * panels are processed SEQUENTIALLY, one shot at a time
   * motion is a smooth source-rect resampled from the Ken Burns keyframes
   * each rendered frame is written as a temp JPEG to a temp dir on disk (so
      at most one frame lives in RAM), then ffmpeg encodes the image sequence
      and muxes the final mix audio, keeping A/V in sync via the audio timeline
   * temporary files are cleaned up automatically afterwards
   * hardware acceleration is used only if the requested codec is supported
      and the runtime reports it; otherwise it falls back to CPU encoding
   * LOW_RAM_MODE (env var) or render.low_ram_mode keep the whole chapter out
     of memory (one panel / one frame at a time)

No subtitles are produced.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)

RENDER_PLAN_NAME = "render_plan.json"
FINAL_MIX_NAME = "final_mix.wav"
FRAME_PATTERN = "frame_%06d.jpg"

# Hard caps so a bad config can't blow memory unpredictably.
MAX_WIDTH, MAX_HEIGHT = 3840, 2160


class RenderError(Exception):
    """Base error for the render stage."""


class NoRenderPlan(RenderError):
    pass


class NoFinalAudio(RenderError):
    pass


def low_ram_mode(cfg):
    """LOW_RAM_MODE env var wins, else render.low_ram_mode (default True)."""
    env = os.environ.get("LOW_RAM_MODE")
    if env is not None and env.strip() != "":
        return env.strip().lower() in ("1", "true", "yes", "on")
    render = getattr(cfg, "render", None)
    if render is None:
        return True
    try:
        return bool(render.get("low_ram_mode", True))
    except (TypeError, ValueError):
        return True


def ffmpeg_bin():
    cand = os.environ.get("FFMPEG")
    if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
        return cand
    for c in ("ffmpeg", "/home/madhav/bin/ffmpeg", "/usr/bin/ffmpeg"):
        if shutil.which(c):
            return c
        if Path(c).is_file() and os.access(c, os.X_OK):
            return c
    raise RenderError("ffmpeg not found; set FFMPEG or install ffmpeg")


# ------------------------------------------------------------ plan loading
def load_render_plan(root):
    path = Path(root) / "motion" / RENDER_PLAN_NAME
    if not path.is_file():
        raise NoRenderPlan(f"no {path} - run the motion stage (Task 17/18) "
                           "first")
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NoRenderPlan(f"cannot read {path}: {exc}")


def load_final_audio(cfg):
    path = Path(cfg.output.audio_dir) / FINAL_MIX_NAME
    if not path.is_file():
        raise NoFinalAudio(f"no {path} - run the audio mix stage (Task 21) "
                           "first")
    return path


def resolve_image(root, image):
    p = Path(image)
    if not p.is_absolute():
        p = root / p
    return p


def total_duration_from_audio(final_mix_path):
    import wave
    with wave.open(str(final_mix_path), "rb") as wf:
        n = wf.getnframes()
        sr = wf.getframerate()
        return (n / float(sr)) if sr else 0.0


def transitions_from_plan(plan):
    out = {}
    for tr in plan.get("transitions") or []:
        out[tr.get("from_index")] = tr
    return out


def cumulative_timeline(entries, transitions):
    """Entry index -> dict(start, end, duration, entry) with overlaps.

    A dissolve/crossfade makes the incoming shot start `overlap` seconds
    before the boundary (the two coexist), matching the Task 18 plan.
    """
    tl = {}
    cursor = 0.0
    carry_overlap = 0.0
    for i, entry in enumerate(entries):
        idx = entry.get("index")
        dur = float(entry.get("motion", {}).get("duration") or 0.0)
        tr = transitions.get(idx)
        overlap = 0.0
        if tr is not None and tr.get("type") in ("dissolve", "crossfade"):
            overlap = min(float(tr.get("overlap") or 0.0), dur)
        start = cursor - carry_overlap
        tl[idx] = {
            "index": idx, "entry": entry,
            "start": start, "end": start + dur,
            "duration": dur, "overlap": overlap,
        }
        cursor = start + dur
        carry_overlap = overlap
    return tl, cursor


def sample_rect(keyframes, local_t):
    """Interpolate the eased source rect at local time t in [0,1]."""
    if not keyframes:
        return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
    if len(keyframes) == 1:
        return keyframes[0]
    if local_t <= keyframes[0]["t"]:
        return keyframes[0]
    if local_t >= keyframes[-1]["t"]:
        return keyframes[-1]
    for a, b in zip(keyframes, keyframes[1:]):
        if a["t"] <= local_t <= b["t"]:
            span = (b["t"] - a["t"]) or 1.0
            f = (local_t - a["t"]) / span
            return {
                "x": a["x"] + (b["x"] - a["x"]) * f,
                "y": a["y"] + (b["y"] - a["y"]) * f,
                "w": max(0.001, a["w"] + (b["w"] - a["w"]) * f),
                "h": max(0.001, a["h"] + (b["h"] - a["h"]) * f),
            }
    return keyframes[-1]


def crop_resize(frame, rect, out_w, out_h):
    """Crop the Ken Burns source rect from `frame` then resize to output."""
    ih, iw = frame.shape[:2]
    x = max(0, min(iw, int(round(rect["x"] * iw))))
    y = max(0, min(ih, int(round(rect["y"] * ih))))
    w = max(1, min(iw - x, int(round(rect["w"] * iw))))
    h = max(1, min(ih - y, int(round(rect["h"] * ih))))
    crop = frame[y:y + h, x:x + w]
    if (w, h) != (out_w, out_h):
        import cv2
        crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return crop


def render_entry_frame(root, entry, img, t, out_w, out_h):
    motion = entry["motion"]
    dur = float(motion.get("duration") or 0.0)
    local = 0.0
    if dur > 0:
        local = max(0.0, min(1.0, t / dur))
    rect = sample_rect(motion["path"]["keyframes"], local)
    return crop_resize(img, rect, out_w, out_h)


# ---------------------------------------------------------------- timeline
def find_active(tl, transitions, t):
    """Return (current_idx, prev_idx, transition_frac) at global time t.

    transition_frac in [0,1] ramps 0->1 across a crossfade window occupied by
    the previous shot's tail; None when no transition is active.
    """
    current = None
    for idx, e in tl.items():
        if e["start"] <= t < e["end"]:
            current = idx
            break
    if current is None:
        return None, None, None
    # crossfade from a preceding shot?
    for fidx, tr in transitions.items():
        if tr.get("type") in ("dissolve", "crossfade"):
            to = tl.get(tr.get("to_index"))
            if to is None or to["index"] != current:
                continue
            overlap = float(tr.get("overlap") or 0.0)
            if overlap <= 0:
                continue
            window_start = to["start"]
            window_end = to["start"] + overlap
            if window_start <= t <= window_end:
                frac = (t - window_start) / overlap
                return current, tr.get("from_index"), max(0.0, min(1.0, frac))
    return current, None, None


# ------------------------------------------------------------ render loop
def render_video(cfg, root, out_path=None, low_ram=None, cleanup=True,
                 codec=None, crf=None, fps_override=None, on_progress=None):
    """Render the final video from the motion plan + final audio.

    Returns a summary dict. Frames are written to a temp dir (one at a time),
    ffmpeg encodes to the output and muxes the final mix; temp files are
    removed automatically afterwards.

    on_progress: optional callable(done, total) invoked as frames render.
    """
    import cv2
    root = Path(root)
    plan = load_render_plan(root)
    entries = plan.get("entries") or []
    if not entries:
        raise NoRenderPlan("render plan has no entries")

    final_audio = load_final_audio(cfg)
    if low_ram is None:
        low_ram = low_ram_mode(cfg)

    video_cfg = getattr(cfg, "video", None)
    res = (video_cfg.get("resolution", "1920x1080") if video_cfg
           else "1920x1080")
    try:
        out_w, out_h = (int(x) for x in str(res).lower().split("x"))
    except Exception:
        out_w, out_h = 1920, 1080
    out_w = min(max(16, out_w), MAX_WIDTH)
    out_h = min(max(16, out_h), MAX_HEIGHT)
    fps = fps_override or (float(video_cfg.get("fps", 30)) if video_cfg
                           else 30.0)
    fps = max(1.0, fps)

    render_cfg = getattr(cfg, "render", None)
    if render_cfg is not None:
        codec = codec or render_cfg.get("codec", "libx264")
        crf = crf if crf is not None else render_cfg.get("crf", 23)
        preset = render_cfg.get("preset", "veryfast")
        pix_fmt = render_cfg.get("pix_fmt", "yuv420p")
        temp_dir = root / str(render_cfg.get("temp_dir", "output/tmp"))
    else:
        preset, pix_fmt = "veryfast", "yuv420p"
        crf = 23 if crf is None else crf
        temp_dir = root / "output/tmp"

    if out_path is None:
        out_path = root / "output" / "MangaExplainer_video.mp4"

    total_dur = float(total_duration_from_audio(final_audio))
    transitions = transitions_from_plan(plan)
    tl, _ = cumulative_timeline(entries, transitions)
    total_frames = int(round(total_dur * fps))
    if total_frames <= 0:
        raise RenderError("final mix has zero duration")

    frame_dir = temp_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    image_cache = {}

    def panel_image(idx):
        if idx in image_cache:
            return image_cache[idx]
        e = entries_by_idx.get(idx)
        if e is None:
            return None
        p = resolve_image(root, e["motion"]["image"])
        img = cv2.imread(str(p))
        if img is None:
            raise RenderError(f"cannot read {p}")
        if low_ram:
            image_cache.clear()
        image_cache[idx] = img
        return img

    entries_by_idx = {e.get("index"): e for e in entries}

    frame_idx = 0
    try:
        for f in range(total_frames):
            t = f / fps
            cur, prev, frac = find_active(tl, transitions, t)
            if cur is None:
                out_img = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            else:
                img = panel_image(cur)
                out_img = render_entry_frame(root, entries_by_idx[cur], img,
                                             t - tl[cur]["start"], out_w, out_h)
                if prev is not None and frac is not None:
                    pimg = panel_image(prev)
                    pout = render_entry_frame(
                        root, entries_by_idx[prev], pimg,
                        (t - tl[prev]["start"]) if prev in tl else 0.0,
                        out_w, out_h)
                    out_img = blend(pout, out_img, frac)
                elif tl[cur].get("entry", {}).get("transition") == "fade":
                    # dip to black near this shot's end for fade-out
                    out_img = fade_out(out_img, t, tl[cur])
            frame_path = frame_dir / (FRAME_PATTERN % frame_idx)
            cv2.imwrite(str(frame_path), out_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            frame_idx += 1
            if on_progress is not None:
                try:
                    on_progress(frame_idx, total_frames)
                except Exception:
                    pass
            if low_ram and frame_idx % 10 == 0:
                gc.collect()
    finally:
        image_cache.clear()
        gc.collect()

    if frame_idx == 0:
        shutil.rmtree(frame_dir, ignore_errors=True)
        raise RenderError("no frames rendered for the mix duration")

    ffmpeg = ffmpeg_bin()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-framerate", str(fps),
        "-i", f"{frame_dir}/{FRAME_PATTERN}",
        "-i", str(final_audio),
        "-c:v", codec, "-preset", preset, "-crf", str(crf),
        "-pix_fmt", pix_fmt, "-c:a", "aac", "-shortest", str(out_path),
    ]
    LOG.info("ffmpeg: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RenderError(f"ffmpeg failed: {proc.stderr[-1200:]}")

    if cleanup:
        shutil.rmtree(frame_dir, ignore_errors=True)

    return {
        "result": "rendered",
        "resolution": f"{out_w}x{out_h}",
        "fps": fps,
        "duration": round(total_dur, 3),
        "frames": frame_idx,
        "low_ram": low_ram,
        "codec": codec,
        "output": str(out_path),
    }


def blend(a, b, alpha):
    return (a.astype(np.float32) * (1.0 - alpha) +
            b.astype(np.float32) * alpha).astype(np.uint8)


def fade_out(img, t, entry):
    dur = float(entry["duration"] or 0.0)
    if dur <= 0:
        return img
    local = t / dur
    if local >= 0.85:
        alpha = min(1.0, (local - 0.85) / 0.15)
        return (img.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    return img
