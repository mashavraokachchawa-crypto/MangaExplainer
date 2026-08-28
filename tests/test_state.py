"""Tests: checkpoint state save/load and crash-resume semantics."""

from state import State

NAMES = ["alpha", "beta"]


def test_state_saves_and_loads(tmp_path):
    first = State(NAMES, str(tmp_path))
    first.mark_completed("alpha")
    assert (tmp_path / "checkpoints.json").exists()
    second = State(NAMES, str(tmp_path))
    assert second.status_of("alpha") == "completed"
    assert second.next_pending() == "beta"
    assert not second.is_complete()


def test_state_roundtrip_complete(tmp_path):
    first = State(NAMES, str(tmp_path))
    first.mark_running("alpha")
    first.mark_completed("alpha")
    first.mark_completed("beta")
    second = State(NAMES, str(tmp_path))
    assert second.is_complete()
    assert second.completed_count() == 2
    assert second.next_pending() is None


def test_running_stage_survives_crash_as_pending(tmp_path):
    first = State(NAMES, str(tmp_path))
    first.mark_running("alpha")
    second = State(NAMES, str(tmp_path))
    assert second.status_of("alpha") == "running"
    assert second.next_pending() == "alpha"


def test_corrupt_checkpoint_is_recovered(tmp_path):
    target = tmp_path / "checkpoints.json"
    target.write_text("{not valid json", encoding="utf-8")
    state = State(NAMES, str(tmp_path))
    assert state.next_pending() == "alpha"
    assert (tmp_path / "checkpoints.json.corrupt").exists()


def test_unknown_stage_is_skipped_in_order(tmp_path):
    first = State(["zeta", "alpha", "beta"], str(tmp_path))
    first.mark_completed("zeta")
    second = State(["alpha", "beta"], str(tmp_path))
    assert second.next_pending() == "alpha"