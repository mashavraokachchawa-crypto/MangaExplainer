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
PAGE_EXTRACTED = "extracted"
OCR_COMPLETED = "ocr_completed"
VLM_COMPLETED = "vlm_completed"
KNOWLEDGE_COMPLETED = "knowledge_completed"
SCENES_COMPLETED = "scenes_completed"
SCRIPT_COMPLETED = "script_completed"


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