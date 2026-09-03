"""Tests: CLI entry point starts and the simple Task 28 interface works."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def run_cli(*args):
    return subprocess.run(
        [PYTHON, "main.py", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_help_exits_zero_and_lists_commands():
    result = run_cli("--help")
    assert result.returncode == 0, result.stderr
    # Task 28 simple command set is exposed
    for token in ("start", "resume", "status", "render", "check", "clean"):
        assert token in result.stdout


def test_status_exits_zero():
    result = run_cli("status")
    assert result.returncode == 0, result.stderr
    assert "progress" in result.stdout.lower()
    assert "checkpoint" in result.stdout.lower()


def test_status_lists_all_pipeline_stages():
    from pipeline.run_pipeline import STAGE_NAMES
    result = run_cli("status")
    assert result.returncode == 0
    assert "extract_pages" in result.stdout
    assert "render_video" in result.stdout
    # every pipeline stage is listed, no subtitle stage
    assert f"/{len(STAGE_NAMES)} stages" in result.stdout
    assert "subtitle" not in result.stdout.lower()


def test_no_command_shows_usage():
    result = run_cli()
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_simple_commands_run_without_traceback():
    # The Task 28 surface must not crash with a Python traceback, even when
    # upstream inputs are missing (they report a clear error instead).
    # NOTE: 'render' and 'clean' are intentionally not in this loop. On a
    # fully-built workspace 'render' legitimately renders the whole video for
    # minutes (>120s), and 'clean' deletes the real stage outputs. render's
    # missing-input failure path is unit-covered in test_video_render.py, and
    # clean has no error branch to exercise here.
    for cmd in ("start", "resume", "check"):
        result = run_cli(cmd)
        assert "Traceback" not in result.stderr, (cmd, result.stderr)
        assert "Traceback" not in result.stdout, (cmd, result.stdout)


def test_status_has_no_subtitle_reference():
    result = run_cli("status")
    assert "subtitle" not in result.stdout.lower()