"""Tests: project directories and CLI entry point exist."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_DIRS = [
    "input",
    "pages",
    "panels",
    "crops",
    "ocr",
    "analysis",
    "scenes",
    "script",
    "audio",
    "shots",
    "music",
    "output",
    "state",
    "logs",
    "config",
    "pipeline",
    "tests",
    "tools",
]


def test_required_directories_exist():
    missing = [name for name in REQUIRED_DIRS if not (ROOT / name).is_dir()]
    assert not missing, f"missing directories: {missing}"


def test_main_entry_exists():
    assert (ROOT / "main.py").is_file()


def test_stage_workspace_directories_covered():
    from pipeline.stages import STAGES

    for stage in STAGES:
        assert (ROOT / stage.output_dir).is_dir(), f"{stage.output_dir} missing"