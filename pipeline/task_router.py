"""Task-based routing preferences for OmniRoute.

OmniRoute decides the concrete provider/model/key for a request. MangaExplainer
only needs to express *what kind of task* it is (vision understanding, text
narration, etc.) as a routing hint so OmniRoute can bias selection. When a
pinned model is configured it wins; otherwise the task hint maps to OmniRoute's
``auto/<category>`` virtual model selection.
"""
from __future__ import annotations

# Task id -> OmniRoute virtual model / category hint.
# OmniRoute understands `auto/<category>:<tier>` model ids (see its docs);
# these are the categories MangaExplainer actually uses.
TASK_TO_MODEL = {
    "understanding": "auto/vision",   # per-panel VLM analysis (image)
    "narration": "auto/chat",          # scene narration prose (text)
    "script": "auto/chat",             # multi-segment script assembly (text)
    "ocr_assist": "auto/vision",       # OCR fallback / correction (image)
    "summary": "auto/chat",            # arc / volume summaries (text)
}

# Fallback category if a task is unknown.
DEFAULT_TASK_CATEGORY = "auto/chat"


def resolve_model(task: str, cfg, section: str = "llm") -> str:
    """Return the OmniRoute model id to request for a task.

    A pinned ``omniroute_model`` in config wins over the task default, so an
    operator can force e.g. ``gpt-4o`` regardless of task. Otherwise we map
    the task to OmniRoute's ``auto/<category>`` selector.
    """
    pinned = str(getattr(cfg, section).get("omniroute_model") or "").strip()
    if pinned:
        return pinned
    return TASK_TO_MODEL.get(task, DEFAULT_TASK_CATEGORY)


def model_for_task(task: str) -> str:
    return TASK_TO_MODEL.get(task, DEFAULT_TASK_CATEGORY)


def is_omniroute_provider(cfg, section: str) -> bool:
    try:
        return str(getattr(cfg, section).get("provider") or "").strip().lower() == "omniroute"
    except Exception:
        return False
