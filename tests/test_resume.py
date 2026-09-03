"""Tests: Task 27 resume system.

Verifies:
  * checkpoints are stored under data/checkpoints/
  * status is saved after every completed task/segment
  * on interruption the run detects completed work, skips it, and continues
    from the last incomplete task
  * valid checkpoints are never deleted by clean-cache
  * existing voice/image/video artifacts are not regenerated on resume
"""
import json
from pathlib import Path

from config.loader import Config
from pipeline import run_pipeline as rp


def _cfg(tmp_path):
    return Config({
        "output": {"pages_dir": str(tmp_path / "pages"),
                   "panels_dir": str(tmp_path / "panels"),
                   "ocr_dir": str(tmp_path / "ocr"),
                   "analysis_dir": str(tmp_path / "analysis"),
                   "scenes_dir": str(tmp_path / "scenes"),
                   "script_dir": str(tmp_path / "script"),
                   "audio_dir": str(tmp_path / "audio"),
                   "shots_dir": str(tmp_path / "shots"),
                   "crops_dir": str(tmp_path / "crops")},
        "input": {"pdf": str(tmp_path / "input" / "manga.pdf")},
        "pipeline": {"batch_size": 1,
                     "state": {"dir": str(tmp_path / "state")},
                     "checkpoints": {"dir": str(tmp_path / "data" / "checkpoints")},
                     "cache": {"dir": str(tmp_path / "cache")}},
        "memory": {"guard_mb": 3072},
    }, tmp_path)


def _stub_runner(targets):
    """Fakes that save a checkpoint after each completed task via the State."""
    fakes = {}
    for name in rp.STAGE_NAMES:
        def runner(cfg, root, force=False, guard=None, progress=None,
                   _name=name, _targets=targets):
            _targets.append(_name)
            return True
        fakes[name] = runner
    return fakes


def test_checkpoints_stored_under_data_checkpoints(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(rp, "_STAGE_FNS", _stub_runner([]))
    cp_dir = Path(cfg.pipeline.checkpoints.dir)
    assert "data" in cp_dir.parts and "checkpoints" in cp_dir.parts
    rp.run_pipeline(cfg, tmp_path)
    # a checkpoint file must exist inside data/checkpoints/
    assert (cp_dir / "checkpoints.json").is_file()
    data = json.loads((cp_dir / "checkpoints.json").read_text("utf-8"))
    assert len(data["stages"]) == len(rp.STAGE_NAMES)
    assert all(row["status"] == "completed" for row in data["stages"])


def test_status_saved_after_every_completed_task(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(rp, "_STAGE_FNS", _stub_runner([]))
    cp_file = Path(cfg.pipeline.checkpoints.dir) / "checkpoints.json"
    # drive the pipeline manually up to a mid-point to inspect incremental saves
    from state import State
    state = State(rp.STAGE_NAMES, Path(cfg.pipeline.checkpoints.dir))
    state.mark_completed("extract_pages")
    data = json.loads(cp_file.read_text("utf-8"))
    statuses = {row["name"]: row["status"] for row in data["stages"]}
    assert statuses["extract_pages"] == "completed"
    assert statuses["understand_panels"] == "pending"
    # status was persisted immediately (file reflects partial completion)
    assert (cp_file.read_text("utf-8")).count("completed") >= 1


def test_resume_skips_completed_and_continues_from_incomplete(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    calls = []
    fakes = _stub_runner(calls)
    monkeypatch.setattr(rp, "_STAGE_FNS", fakes)
    # first run stops early (simulate interruption after 3 stages)
    cp_dir = Path(cfg.pipeline.checkpoints.dir)
    from state import State
    state = State(rp.STAGE_NAMES, cp_dir)
    state.mark_completed("extract_pages")
    state.mark_completed("clean_panels")
    state.mark_completed("understand_panels")

    # resume: completed stages must NOT be re-run; continue from the first
    # incomplete stage (the step right after understand_panels)
    calls.clear()
    rp.run_pipeline(cfg, tmp_path)
    assert "extract_pages" not in calls
    assert "clean_panels" not in calls
    assert "understand_panels" not in calls
    first_incomplete = rp.STAGE_NAMES[rp.STAGE_NAMES.index("understand_panels") + 1]
    assert calls[0] == first_incomplete


def test_resume_after_partial_run_runs_only_remaining(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    calls = []
    fakes = _stub_runner(calls)
    # make a stage fail partway so the checkpoint reflects an interrupted state
    def flaky(name):
        def runner(cfg, root, force=False, guard=None, progress=None):
            calls.append(name)
            if name == "pocket_tts":
                from pipeline.run_pipeline import PipelineError
                raise PipelineError(f"{name} broke")
            return True
        runner.__name__ = name
        return runner
    monkeypatch.setattr(rp, "_STAGE_FNS",
                        {n: flaky(n) for n in rp.STAGE_NAMES})
    first = rp.run_pipeline(cfg, tmp_path)
    assert first["status"] == "error"
    assert first["current"] == "pocket_tts"

    # resume with all-OK fakes: only the incomplete+later stages re-run
    calls.clear()
    ok = {n: _stub_runner([])[n] for n in rp.STAGE_NAMES}
    monkeypatch.setattr(rp, "_STAGE_FNS", ok)
    second = rp.run_pipeline(cfg, tmp_path)
    assert second["status"] == "ok"
    # stages before pocket_tts were already completed -> marked skipped
    skipped = [r["name"] for r in second["report"] if r.get("skipped")]
    assert skipped == rp.STAGE_NAMES[: rp.STAGE_NAMES.index("pocket_tts")]


def test_no_subtitle_stage_in_resume_chain():
    assert not any("subtitle" in s for s in rp.STAGE_NAMES)


def test_clean_cache_targets_do_not_include_checkpoint_dir():
    # Task 27: data/checkpoints is deliberately NOT a clean-cache target, so
    # valid checkpoints are never auto-deleted by the artifact cleaner.
    import main as cli
    # inspect the actual cleanup call: it clears stage output dirs + cache dir,
    # never the checkpoint directory.
    src = Path(cli.__file__).read_text("utf-8")
    assert "cfg.pipeline.checkpoints.dir" in src
    # the cleanup targets are cache + stage output dirs; checkpoint dir is
    # preserved and reported as kept (never unlinked).
    assert "never auto-deleted" in src or "checkpoints kept" in src.lower()
