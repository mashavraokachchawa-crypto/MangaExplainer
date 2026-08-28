"""Tests: CLI entry point starts and the status command works."""

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
        timeout=60,
    )


def test_help_exits_zero_and_lists_commands():
    result = run_cli("--help")
    assert result.returncode == 0, result.stderr
    for token in ("status", "resume", "clean-cache", "audio", "plan", "crops"):
        assert token in result.stdout


def test_status_exits_zero():
    result = run_cli("status")
    assert result.returncode == 0, result.stderr
    assert "input pdf" in result.stdout.lower()
    assert "batch size" in result.stdout.lower()


def test_status_lists_all_stages():
    result = run_cli("status")
    assert result.returncode == 0
    assert "extract_pages" in result.stdout
    assert "render_video" in result.stdout
    assert "stages (12)" in result.stdout


def test_no_command_shows_usage():
    result = run_cli()
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()