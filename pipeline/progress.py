"""Live per-step progress for the pipeline (zero dependencies).

The web dashboard polls /api/status every ~0.8 s while a job runs. This tiny
reporter keeps the "what is the current step doing" view in one small JSON
file (state/progress.json) that stage runners update as they walk their items
- pages, panels, scenes, segments, frames.

* writes are atomic (tmp file + os.replace) so a reader never sees a torn file
* writes are throttled (~0.4 s) so a fast inner loop cannot burn disk I/O on
  a low-RAM machine
* it is a LIVE VIEW ONLY: checkpoints remain the source of truth for what is
  actually complete, this file just says what the running step is doing now
"""
import json
import os
import threading
import time
from pathlib import Path

PROGRESS_FILE = "progress.json"


def _pct(done, total):
    return min(100, round(100.0 * done / total)) if total else 100


class Progress:
    """Small file-backed reporter handed to the stage runners."""

    def __init__(self, root, state_dir=None, throttle=0.4):
        self._root = Path(root)
        base = Path(state_dir) if state_dir else self._root / "state"
        self._path = base / PROGRESS_FILE
        self._lock = threading.Lock()
        self._last_write = 0.0
        self._throttle = throttle
        self._details = {}  # stage -> {key: value} surfaced live
        self._last = {}     # carries done/total/image between writes

    def path(self):
        return self._path

    def begin(self, stage, label=None, image=None):
        self._last = {}
        self._details.pop(stage, None)
        self._write({
            "stage": stage,
            "label": label or stage,
            "phase": None,
            "done": None,
            "total": None,
            "pct": 0,
            "item": None,
            "image": image,
            "details": None,
            "updated_at": time.time(),
        }, force=True)

    def step(self, stage, done, total, phase=None, item=None, image=None):
        self._last.update({"done": done, "total": total,
                           "pct": _pct(done, total), "item": item,
                           "image": image, "phase": phase})
        self._write({
            "stage": stage,
            "label": None,
            "phase": phase,
            "done": done,
            "total": total,
            "pct": _pct(done, total),
            "item": item,
            "image": image,
            "details": self._details.get(stage),
            "updated_at": time.time(),
        })

    def phase(self, stage, text, image=None):
        """Countless step (e.g. one-shot stages): just describe what is running."""
        self._last.update({"phase": text, "image": image})
        self._write({
            "stage": stage,
            "label": None,
            "phase": text,
            "done": self._last.get("done"),
            "total": self._last.get("total"),
            "pct": self._last.get("pct"),
            "item": self._last.get("item"),
            "image": image,
            "details": self._details.get(stage),
            "updated_at": time.time(),
        })

    def detail(self, stage, **kwargs):
        """Attach live key/value info to the current stage (transparent work).

        Kept separate from step() so a stage can stream rich per-item detail
        (e.g. what panel OCR just read, which scene is being written, segment
        text + duration) without disturbing its done/total counter. The last
        step's own detail dict (a {key: value} block) is what the dashboard's
        expandable rows render.
        """
        cur = dict(self._details.get(stage) or {})
        cur.update(kwargs)
        self._details[stage] = cur
        self._write({
            "stage": stage,
            "label": None,
            "phase": kwargs.get("phase") or self._last.get("phase"),
            "done": self._last.get("done"),
            "total": self._last.get("total"),
            "pct": self._last.get("pct"),
            "item": self._last.get("item"),
            "image": kwargs.get("image") or self._last.get("image"),
            "details": dict(cur),
            "updated_at": time.time(),
        })

    def clear(self):
        with self._lock:
            try:
                self._path.unlink(missing_ok=True)
            except OSError:
                pass

    def _write(self, data, force=False):
        now = time.time()
        with self._lock:
            if not force and now - self._last_write < self._throttle:
                return
            self._last_write = now
            if data.get("label") is None:
                data["label"] = data.get("stage")
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self._path.with_name(self._path.name + ".tmp")
                with open(tmp, "wb") as handle:
                    handle.write(body)
                os.replace(tmp, self._path)
            except OSError:
                pass


def read_progress(state_dir=None, root=None):
    """Return the live progress dict, or None when nothing is running."""
    base = Path(state_dir) if state_dir else Path(root or ".") / "state"
    path = base / PROGRESS_FILE
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None