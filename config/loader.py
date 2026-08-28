"""Configuration loading for MangaExplainer.

Defaults live in DEFAULT_CONFIG and are merged with config/config.yaml so a
missing file never breaks the pipeline. Relative paths are resolved against
the project root. A single integer batch_size >= 1 is enforced to keep the
pipeline low-RAM (default: 1 = one item at a time).
"""
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import yaml

DEFAULT_CONFIG = {
    "input": {"pdf": "input/manga.pdf"},
    "output": {"dir": "output", "pages_dir": "pages", "panels_dir": "panels", "ocr_dir": "ocr", "analysis_dir": "analysis", "scenes_dir": "scenes", "script_dir": "script", "audio_dir": "audio", "shots_dir": "shots"},
    "images": {
        "format": "jpg",
        "render_scale": 1.5,
        "resolution": "1200x1800",
        "max_pixels": 2500000,
        "jpeg_quality": 80,
    },
    "panels": {
        "strategy": "gutter_flood",
        "border_kernel": 3,
        "close_kernel": 3,
        "min_area": 3000,
        "min_area_ratio": 0.0015,
        "max_area_ratio": 0.95,
        "min_side": 20,
        "max_aspect_ratio": 4.0,
        "min_confidence": 0.6,
        "duplicate_iou": 0.5,
        "drop_edge_touching": True,
        "max_panels": 40,
        "pad_pixels": 2,
        "jpeg_quality": 85,
    },
    "reading": {
        "direction": "rtl",
        "row_overlap_ratio": 0.5,
        "weights": {
            "row_overlap": 2.0,
            "horizontal": 1.5,
            "vertical": 0.3,
            "distance": 0.5,
            "size": 0.0,
        },
    },
    "ocr": {
        "engine": "auto",
        "language": "eng+jpn",
        "psm": 11,
        "timeout_seconds": 30,
    },
    "preprocess": {
        "enabled": True,
        "grayscale": True,
        "contrast": 1.4,
        "denoise": "none",
        "denoise_kernel": 3,
        "resize_scale": 1.5,
        "resize_max_pixels": 800000,
        "threshold": "none",
    },
    "vlm": {
        "enabled": True,
        "provider": "local",
        "model": "",
        "device": "cpu",
        "max_image_size": 768,
        "max_new_tokens": 256,
        "timeout_seconds": 120,
    },
    "llm": {
        "enabled": True,
        "provider": "local",
        "model": "",
        "device": "cpu",
        "max_context": 4096,
        "max_new_tokens": 512,
        "temperature": 0.7,
        "timeout_seconds": 120,
    },
    "scenes": {
        "threshold": 0.45,
        "weights": {
            "location_change": 0.45,
            "character_change": 0.55,
            "event_change": 0.5,
            "narrative_transition": 0.5,
        },
        "continuity": {
            "character": 0.35,
            "location": 0.35,
            "action": 0.25,
            "dialogue": 0.3,
            "event": 0.3,
        },
        "transition_keywords": [
            "meanwhile",
            "later",
            "suddenly",
            "翌日",
            "しかし",
            "そして",
        ],
        "summary_max_items": 6,
    },
    "video": {"resolution": "1920x1080", "fps": 30},
    "tts": {
        "enabled": True,
        "engine": "auto",
        "voice": "en",
        "sample_rate": 22050,
        "rate_wpm": 150,
        "pitch_base": 50,
        "timeout_seconds": 60,
    },
    "shots": {
        "match_weights": {
            "character": 0.25,
            "action": 0.15,
            "event": 0.20,
            "object": 0.10,
            "ocr": 0.15,
            "story_relevance": 0.15,
        },
        "review_threshold": 0.55,
        "tie_epsilon": 0.02,
        "direct_match_floor": 0.90,
        "secondary_panel_epsilon": 0.15,
        "long_segment_threshold": 9.0,
        "max_shots_per_segment": 3,
        "zoom_in_end": 1.12,
        "zoom_out_end": 0.92,
    },
    "pipeline": {
        "batch_size": 1,
        "state": {"dir": "state"},
        "cache": {"dir": "state/cache"},
    },
    "memory": {"guard_mb": 3072},
    "logging": {
        "level": "INFO",
        "console_level": "WARNING",
        "log_dir": "logs",
        "max_bytes": 1048576,
        "backup_count": 3,
    },
}

_PATH_KEYS = (
    ("input", "pdf"),
    ("output", "dir"),
    ("output", "pages_dir"),
    ("output", "panels_dir"),
    ("output", "ocr_dir"),
    ("output", "analysis_dir"),
    ("output", "scenes_dir"),
    ("output", "script_dir"),
    ("output", "audio_dir"),
    ("output", "shots_dir"),
    ("pipeline", "state", "dir"),
    ("pipeline", "cache", "dir"),
    ("logging", "log_dir"),
)


class _Node:
    def __init__(self, data):
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name):
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(name) from None
        return _Node(value) if isinstance(value, dict) else value

    def get(self, key, default=None):
        value = self._data.get(key, default)
        return _Node(value) if isinstance(value, dict) else value

    def to_dict(self):
        return deepcopy(self._data)


class Config:
    def __init__(self, data, root_dir):
        self._data = data
        self._root_dir = Path(root_dir)

    def __getattr__(self, name):
        if name not in self._data:
            raise AttributeError(name)
        value = self._data[name]
        return _Node(value) if isinstance(value, dict) else value

    @property
    def root_dir(self):
        return self._root_dir

    def to_dict(self):
        return deepcopy(self._data)


def _merge(base, override):
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            _merge(base[key], value)
        else:
            base[key] = value


def load_config(root_dir=None, config_path=None):
    root = Path(root_dir).resolve() if root_dir else Path(__file__).resolve().parent.parent
    data = deepcopy(DEFAULT_CONFIG)
    path = Path(config_path) if config_path else root / "config" / "config.yaml"
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            user = yaml.safe_load(handle) or {}
        _merge(data, user)
    for keys in _PATH_KEYS:
        node = data
        for key in keys[:-1]:
            node = node[key]
        raw = Path(node[keys[-1]])
        node[keys[-1]] = (raw if raw.is_absolute() else root / str(raw)).resolve()
    batch_size = data["pipeline"]["batch_size"]
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"pipeline.batch_size must be an integer >= 1, got {batch_size!r}")
    return Config(data, root)