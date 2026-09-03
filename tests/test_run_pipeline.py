"""Tests: complete connected pipeline orchestrator (Task 26).

Verifies sequential execution, progress %, resume-from-last-step (never
repeating completed stages), clear error stop (no silent continuation), and
that no subtitle stage exists anywhere in the chain.
"""

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


def test_stage_list_matches_task26_and_has_no_subtitles():
    expected = [
        "extract_pages", "clean_panels", "understand_panels", "knowledge_base",
        "write_script", "pocket_tts", "voice_timing", "prepare_panels",
        "camera_motion", "music", "sfx", "audio_mix", "render_video",
        "quality_check",
    ]
    assert rp.STAGE_NAMES == expected
    assert len(rp.STAGE_NAMES) == 14
    # no subtitle stage/track anywhere in the chain
    assert not any("subtitle" in s for s in rp.STAGE_NAMES)
    assert not any(s in " ".join(rp.STAGE_NAMES) for s in
                   ("srt", "subtitle_track"))
    # every stage has a runner
    assert set(rp._STAGE_FNS) == set(rp.STAGE_NAMES)
    # transitions were removed: panels are hard-cut only
    assert "transitions" not in rp.STAGE_NAMES
    assert "transitions" not in rp._STAGE_FNS


def test_pipeline_runs_stages_in_order_with_progress(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    order = []

    def fake_factory(name):
        def runner(cfg, root, force=False, guard=None, progress=None):
            order.append(name)
            return True
        return runner

    fakes = {name: fake_factory(name) for name in rp.STAGE_NAMES}
    monkeypatch.setattr(rp, "_STAGE_FNS", fakes)

    result = rp.run_pipeline(cfg, tmp_path)

    assert result["status"] == "ok"
    assert result["completed"] == len(rp.STAGE_NAMES)
    assert order == rp.STAGE_NAMES  # sequential order
    # progress went 0..100 across the report
    progresses = [row["progress"] for row in result["report"]
                  if not row.get("skipped")]
    assert progresses[0] == round(100.0 / len(rp.STAGE_NAMES))
    assert progresses[-1] == 100
    # every stage completed and none failed
    assert all(row["status"] == "completed" for row in result["report"])


def test_resume_never_repeats_completed_stages(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    calls = []

    def fake_factory(name):
        def runner(cfg, root, force=False, guard=None, progress=None):
            calls.append(name)
            return True
        return runner

    fakes = {name: fake_factory(name) for name in rp.STAGE_NAMES}
    monkeypatch.setattr(rp, "_STAGE_FNS", fakes)

    # first full run
    rp.run_pipeline(cfg, tmp_path)
    first_calls = list(calls)

    # second run: completed stages must NOT run again
    calls.clear()
    rp.run_pipeline(cfg, tmp_path)
    assert calls == []  # nothing repeated

    # force re-runs everything
    calls.clear()
    rp.run_pipeline(cfg, tmp_path, force=True)
    assert calls == rp.STAGE_NAMES


def test_error_stops_with_clear_report(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    order = []

    def fake_factory(name):
        def runner(cfg, root, force=False, guard=None, progress=None):
            order.append(name)
            if name == "audio_mix":
                from pipeline.run_pipeline import PipelineError
                raise PipelineError(f"{name} broken")
            return True
        return runner

    fakes = {name: fake_factory(name) for name in rp.STAGE_NAMES}
    monkeypatch.setattr(rp, "_STAGE_FNS", fakes)

    result = rp.run_pipeline(cfg, tmp_path)

    assert result["status"] == "error"
    assert result["current"] == "audio_mix"
    assert "broken" in result["error"]
    # it stopped at the failing stage: ran everything up to and including it,
    # and nothing AFTER it
    assert order == rp.STAGE_NAMES[: rp.STAGE_NAMES.index("audio_mix") + 1]
    failed = [row for row in result["report"]
              if row["status"] == "failed"]
    assert len(failed) == 1 and failed[0]["name"] == "audio_mix"
    # no silent continuation: the final stage never ran
    assert "quality_check" not in order


def test_resume_continues_from_last_completed(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    calls = []

    def fake_factory(fail_at=None):
        def runner(cfg, root, force=False, guard=None, progress=None):
            calls.append(runner._name)
            if fail_at is not None and runner._name == fail_at:
                from pipeline.run_pipeline import PipelineError
                raise PipelineError(f"{runner._name} boom")
            return True
        return runner

    # first run fails at stage 8 (camera_motion)
    fakes = {}
    for name in rp.STAGE_NAMES:
        r = fake_factory(fail_at="camera_motion")
        r._name = name
        fakes[name] = r
    monkeypatch.setattr(rp, "_STAGE_FNS", fakes)
    rp.run_pipeline(cfg, tmp_path)

    # resume: now everything succeeds
    calls.clear()
    fakes_ok = {}
    for name in rp.STAGE_NAMES:
        r = fake_factory()
        r._name = name
        fakes_ok[name] = r
    monkeypatch.setattr(rp, "_STAGE_FNS", fakes_ok)
    result = rp.run_pipeline(cfg, tmp_path)

    # stages 1-7 were completed before the failure -> must NOT be repeated
    resumed = [rp.STAGE_NAMES.index(n) for n in calls]
    assert resumed == list(range(rp.STAGE_NAMES.index("camera_motion"), len(rp.STAGE_NAMES)))
    assert result["status"] == "ok"
    assert result["completed"] == len(rp.STAGE_NAMES)
    # the report marks everything before the failure stage as skipped
    skipped = [r["name"] for r in result["report"] if r.get("skipped")]
    assert skipped == rp.STAGE_NAMES[: rp.STAGE_NAMES.index("camera_motion")]


def _write_pdf(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_changed_pdf_invalidates_completed_stages(tmp_path, monkeypatch):
    """A re-uploaded PDF forces the pipeline to re-run from extract_pages."""
    cfg = _cfg(tmp_path)
    pdf = _write_pdf(Path(cfg.input.pdf), b"pdf-version-1")
    calls = []

    def fake_factory(name):
        def runner(cfg, root, force=False, guard=None, progress=None):
            calls.append(name)
            return True
        return runner

    fakes = {name: fake_factory(name) for name in rp.STAGE_NAMES}
    monkeypatch.setattr(rp, "_STAGE_FNS", fakes)

    # first run over version 1 -> all stages complete
    rp.run_pipeline(cfg, tmp_path)
    assert calls == rp.STAGE_NAMES
    assert calls  # all stages ran

    # unchanged PDF -> second run repeats nothing
    calls.clear()
    rp.run_pipeline(cfg, tmp_path)
    assert calls == []

    # NEW PDF uploaded -> completed stages invalidated, full re-run
    _write_pdf(pdf, b"pdf-version-2-different")
    calls.clear()
    rp.run_pipeline(cfg, tmp_path)
    assert calls == rp.STAGE_NAMES


def test_same_pdf_does_not_rerun(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _write_pdf(Path(cfg.input.pdf), b"stable-pdf")
    calls = []

    def fake_factory(name):
        def runner(cfg, root, force=False, guard=None, progress=None):
            calls.append(name)
            return True
        return runner

    fakes = {name: fake_factory(name) for name in rp.STAGE_NAMES}
    monkeypatch.setattr(rp, "_STAGE_FNS", fakes)
    rp.run_pipeline(cfg, tmp_path)
    calls.clear()
    rp.run_pipeline(cfg, tmp_path)
    assert calls == []


def test_legacy_state_without_fingerprint_reruns_once(tmp_path, monkeypatch):
    """Pre-fingerprint state that says 'complete' cannot prove it matches the
    current PDF, so the first run of the new code invalidates and rebuilds."""
    from state import State as S
    from config.loader import Config
    base = tmp_path
    cfg = Config({
        "input": {"pdf": str(base / "input" / "manga.pdf")},
        "pipeline": {"checkpoints": {"dir": str(base / "data" / "checkpoints")}},
    }, base)
    _write_pdf(Path(cfg.input.pdf), b"legacy-pdf")

    # build legacy state: all 13 completed, no fingerprint key
    st = S(rp.STAGE_NAMES, Path(cfg.pipeline.checkpoints.dir))
    for name in rp.STAGE_NAMES:
        st.mark_completed(name)
    assert st.input_fingerprint() is None

    calls = []
    fakes = {n: (lambda cfg, root, force=False, guard=None, progress=None,
                 _n=n: calls.append(_n) or True)
             for n in rp.STAGE_NAMES}
    monkeypatch.setattr(rp, "_STAGE_FNS", fakes)

    out = rp.run_pipeline(cfg, base)
    assert out["status"] == "ok"
    # legacy completed work was invalidated -> everything re-ran
    assert calls == rp.STAGE_NAMES
    # fingerprint now recorded for the current PDF
    st2 = S(rp.STAGE_NAMES, Path(cfg.pipeline.checkpoints.dir))
    assert st2.input_fingerprint() is not None
    # and a third run is a clean no-op
    calls.clear()
    rp.run_pipeline(cfg, base)
    assert calls == []
