"""Crash-safe checkpoint state for the MangaExplainer pipeline.

The state is a small JSON file in the state/ directory, written atomically
(temp file + os.replace) so a crash can never corrupt it. Only "completed"
stages count as done; a stage left "running" after a crash is treated as
interrupted and re-run on resume.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

STATE_FILE = "checkpoints.json"
STATE_VERSION = 1
PENDING, RUNNING, COMPLETED, FAILED = "pending", "running", "completed", "failed"
INPUT_FINGERPRINT_KEY = "input_fingerprint"
PAGE_EXTRACTED = "extracted"
OCR_COMPLETED = "ocr_completed"
VLM_COMPLETED = "vlm_completed"
KNOWLEDGE_COMPLETED = "knowledge_completed"
SCENES_COMPLETED = "scenes_completed"
SCRIPT_COMPLETED = "script_completed"
AUDIO_COMPLETED = "audio_completed"
VISUAL_PLAN_COMPLETED = "visual_plan_completed"
CROPS_COMPLETED = "crops_completed"
CROPS_COMPLETED = "crops_completed"


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _blank(stage_names):
    now = _now()
    return {
        "version": STATE_VERSION,
        "created_at": now,
        "updated_at": now,
        "pages": {},
        "stages": [
            {
                "name": name,
                "status": PENDING,
                "attempts": 0,
                "started_at": None,
                "completed_at": None,
            }
            for name in stage_names
        ],
    }


def _new_row(name):
    return {
        "name": name,
        "status": PENDING,
        "attempts": 0,
        "started_at": None,
        "completed_at": None,
    }


class State:
    def __init__(self, stage_names, state_dir):
        self._names = list(stage_names)
        self._dir = Path(state_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / STATE_FILE
        self._exists = self._path.exists()
        self._data = self._load()

    @property
    def path(self):
        return self._path

    @property
    def pages(self) -> dict:
        return self._data.get("pages") or {}

    def exists(self):
        return self._exists

    def _load(self):
        if not self._path.exists():
            return _blank(self._names)
        try:
            with open(self._path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError, ValueError):
            backup = self._path.with_name(self._path.name + ".corrupt")
            try:
                os.replace(self._path, backup)
            except OSError:
                pass
            return _blank(self._names)
        if not isinstance(data, dict):
            backup = self._path.with_name(self._path.name + ".corrupt")
            try:
                os.replace(self._path, backup)
            except OSError:
                pass
            return _blank(self._names)
        if not isinstance(data.get("pages"), dict):
            data["pages"] = {}
        rows = {
            row["name"]: row
            for row in data.get("stages", [])
            if isinstance(row, dict) and row.get("name")
        }
        stages = []
        for name in self._names:
            row = rows.get(name) or _new_row(name)
            row.pop("error", None)
            stages.append(row)
        data["stages"] = stages
        data["version"] = STATE_VERSION
        return data

    def _save(self):
        self._data["updated_at"] = _now()
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(self._data, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self._path)
        self._exists = True

    def reload(self):
        self._data = self._load()

    def _stage(self, name):
        for row in self._data["stages"]:
            if row["name"] == name:
                return row
        raise KeyError(name)

    def status_of(self, name):
        return self._stage(name)["status"]

    def mark_running(self, name):
        row = self._stage(name)
        row["status"] = RUNNING
        row["started_at"] = _now()
        row["attempts"] = int(row.get("attempts") or 0) + 1
        self._save()

    def mark_completed(self, name):
        row = self._stage(name)
        row["status"] = COMPLETED
        row["completed_at"] = _now()
        self._save()

    def mark_failed(self, name):
        row = self._stage(name)
        row["status"] = FAILED
        self._save()

    def mark_pending(self, name):
        row = self._stage(name)
        row["status"] = PENDING
        self._save()

    # -- input fingerprint ---------------------------------------------------
    # Records a short hash of the source PDF so a *new* PDF (a fresh upload)
    # invalidates stale completed checkpoints and forces the pipeline to
    # re-run from extract_pages instead of pretending it is already done.

    def input_fingerprint(self):
        return self._data.get(INPUT_FINGERPRINT_KEY)

    def set_input_fingerprint(self, fingerprint):
        self._data[INPUT_FINGERPRINT_KEY] = fingerprint
        self._save()

    def invalidate_completed(self):
        """Mark every completed stage pending so the pipeline re-runs them."""
        changed = False
        for row in self._data["stages"]:
            if row["status"] == COMPLETED:
                row["status"] = PENDING
                row["started_at"] = None
                row["completed_at"] = None
                changed = True
        if changed:
            self._save()
        return changed

    def recover_interrupted(self):
        """Demote any stage left 'running' by a dead process to 'pending'.

        A stage is marked 'running' and then completed/failed synchronously in
        the SAME process. If we ever load this state and see a 'running' stage,
        the previous process died before finishing it (a crash, an OOM kill, a
        SIGKILL'ed server) — it never completed AND never failed. Leaving it
        'running' would make the dashboard show a permanent zombie stage that
        blocks a clean resume, so we normalize it back to 'pending' so the next
        run picks it up from the top of that stage.

        No-op if nothing is in the interrupted state. Returns True if any row
        was recovered.
        """
        changed = False
        for row in self._data["stages"]:
            if row["status"] == RUNNING:
                row["status"] = PENDING
                row["started_at"] = None
                row["completed_at"] = None
                changed = True
        if changed:
            self._save()
        return changed

    def next_pending(self):
        for row in self._data["stages"]:
            if row["status"] in (PENDING, RUNNING, FAILED):
                return row["name"]
        return None

    def is_complete(self):
        return all(row["status"] == COMPLETED for row in self._data["stages"])

    def completed_count(self):
        return sum(1 for row in self._data["stages"] if row["status"] == COMPLETED)

    def summary(self):
        return [
            {"name": row["name"], "status": row["status"]}
            for row in self._data["stages"]
        ]

    def details(self):
        """Per-stage rows with timestamps (name, status, started_at, completed_at)."""
        return [dict(row) for row in self._data["stages"]]

    def page_status(self, page_key):
        pages = self._data.get("pages") or {}
        return PAGE_EXTRACTED if pages.get(page_key) == PAGE_EXTRACTED else "pending"

    def is_page_done(self, page_key):
        return self.page_status(page_key) == PAGE_EXTRACTED

    def mark_page_done(self, page_key):
        pages = self._data.setdefault("pages", {})
        pages[page_key] = PAGE_EXTRACTED
        self._save()

    def ocr_done(self, panel_key):
        pages = self._data.get("pages") or {}
        return pages.get(panel_key) == OCR_COMPLETED

    def mark_ocr_done(self, panel_key):
        pages = self._data.setdefault("pages", {})
        pages[panel_key] = OCR_COMPLETED
        self._save()

    def item_done(self, key, value):
        pages = self._data.get("pages") or {}
        return pages.get(key) == value

    def mark_item_done(self, key, value):
        pages = self._data.setdefault("pages", {})
        pages[key] = value
        self._save()

    def as_dict(self):
        return json.loads(json.dumps(self._data))