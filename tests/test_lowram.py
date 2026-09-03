"""Tests: low-RAM housekeeping (periodic cache sweep + RSS guard)."""

import os
import time
from pathlib import Path

from config.loader import Config
from pipeline.lowram import MemoryGuard, rss_mb

ROOT = Path(__file__).resolve().parent.parent


def make_cfg(root, memory=None, render=None):
    data = {
        "memory": {
            "guard_mb": 3072,
            "sweep_seconds": 60,
            "sweep_dirs": ["state/cache", "output/tmp"],
        },
        "render": {"low_ram_mode": True, "temp_dir": "output/tmp"},
        "pipeline": {
            "batch_size": 1,
            "state": {"dir": str(root / "state")},
            "cache": {"dir": str(root / "state" / "cache")},
            "checkpoints": {"dir": str(root / "data" / "checkpoints")},
        },
    }
    if memory:
        data["memory"].update(memory)
    if render:
        data["render"].update(render)
    return Config(data, root)


def _age(path, seconds):
    future = time.time() - seconds
    os.utime(path, (future, future))


def test_rss_mb_returns_int_or_none():
    value = rss_mb()
    assert value is None or isinstance(value, int)


def test_sweep_dir_resolution_and_defaults(tmp_path):
    guard = MemoryGuard(make_cfg(tmp_path), tmp_path)
    expected = {str((tmp_path / "state" / "cache").resolve()),
                str((tmp_path / "output" / "tmp").resolve())}
    assert {str(p) for p in guard.sweep_dirs} == expected
    assert guard.sweep_seconds == 60
    assert guard.guard_mb == 3072


def test_sweep_removes_stale_keeps_fresh(tmp_path):
    root = tmp_path / "run"
    cache = root / "state" / "cache"
    stale = cache / "stale.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("x")
    _age(stale, 10_000)
    fresh = cache / "fresh.txt"
    fresh.write_text("y")  # mtime now -> younger than the safety window
    old_dir = root / "output" / "tmp" / "old_section"
    old_dir.mkdir(parents=True)
    old = old_dir / "f.jpg"
    old.write_text("z")
    _age(old, 10_000)
    guard = MemoryGuard(make_cfg(root, memory={"sweep_seconds": 60}), root)
    guard.sweep(tuple(guard.sweep_dirs), force=True)
    assert not stale.exists()
    assert fresh.exists()
    assert not old.exists()
    assert not old_dir.exists()  # empty dir pruned


def test_sweep_never_touches_protected_voice_cache(tmp_path):
    root = tmp_path / "run"
    voice = root / "state" / "cache" / "voice"
    voice.mkdir(parents=True)
    emb = voice / "emb.safetensors"
    emb.write_text("w")
    _age(emb, 10_000)
    guard = MemoryGuard(make_cfg(root), root)
    guard.sweep(tuple(guard.sweep_dirs), force=True)
    assert emb.exists()  # SSD-as-memory cache is protected


def test_sweep_never_touches_stage_outputs_or_checkpoints(tmp_path):
    root = tmp_path / "run"
    panels = root / "panels"
    panels.mkdir(parents=True)
    old_panel = panels / "page_001_panel_001.jpg"
    old_panel.write_text("p")
    _age(old_panel, 10_000)
    checks = root / "data" / "checkpoints"
    checks.mkdir(parents=True)
    ck = checks / "checkpoints.json"
    ck.write_text("{}")
    _age(ck, 10_000)
    guard = MemoryGuard(make_cfg(root), root)
    guard.sweep(tuple(guard.sweep_dirs), force=True)
    assert old_panel.exists()
    assert ck.exists()


def test_tick_respects_sweep_cadence(tmp_path):
    root = tmp_path / "run"
    cache = root / "state" / "cache"
    cache.mkdir(parents=True)
    guard = MemoryGuard(make_cfg(root, memory={"sweep_seconds": 600}), root)
    stale = cache / "stale.txt"
    stale.write_text("x")
    _age(stale, 10_000)
    guard.tick()          # first tick always sweeps
    assert not stale.exists()
    stale2 = cache / "stale2.txt"
    stale2.write_text("y")
    _age(stale2, 10_000)
    guard.tick()          # not due yet -> no disk sweep
    assert stale2.exists()


def test_tick_ram_only_when_empty_sweep_dirs(tmp_path):
    root = tmp_path / "run"
    cache = root / "state" / "cache"
    cache.mkdir(parents=True)
    stale = cache / "keep.txt"
    stale.write_text("x")
    _age(stale, 10_000)
    guard = MemoryGuard(make_cfg(root), root)
    guard.tick(sweep_dirs=())  # RAM-only: nothing swept
    assert stale.exists()


def test_sweep_disabled_with_zero_seconds(tmp_path):
    root = tmp_path / "run"
    cache = root / "state" / "cache"
    cache.mkdir(parents=True)
    stale = cache / "stale.txt"
    stale.write_text("x")
    _age(stale, 10_000)
    guard = MemoryGuard(make_cfg(root, memory={"sweep_seconds": 0}), root)
    guard.tick()
    assert guard.sweep_seconds == 0
    assert stale.exists()
    assert guard._thread is None  # timer loop not started


def test_timer_dirs_exclude_render_temp(tmp_path):
    root = tmp_path / "run"
    render = tmp_path / "render_cfg"
    guard = MemoryGuard(make_cfg(root, render={"temp_dir": "output/tmp"}), root)
    assert (root / "output" / "tmp").resolve() == guard._temp_dir.resolve()
    safe = guard._safe_timer_dirs()
    assert (root / "state" / "cache").resolve() in safe
    assert (root / "output" / "tmp").resolve() not in safe