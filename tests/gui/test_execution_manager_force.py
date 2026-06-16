"""Tests for force-start admission and the memory-guard master switch.

These exercise the scheduler's admission decision (`_pick_next_pending`)
in isolation: the manager is constructed but never `.start()`ed, so no
scheduler thread, watchdog, or subprocess runs. `_memory_admits` is
monkeypatched to simulate a system the live free-RAM check considers full.

The fixture seeds `_running_count = 1` (one job already running) and a
generous `max_workers` so the "always run at least one" rule and the
thread-slot ceiling don't pre-empt the memory check under test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flextool.gui.data_models import ProjectSettings
from flextool.gui.execution_manager import (
    ExecutionJob,
    ExecutionManager,
    JobStatus,
    JobType,
)


def _make_manager(tmp_path: Path) -> ExecutionManager:
    return ExecutionManager(project_path=tmp_path, settings=ProjectSettings())


def _pending_scenario(job_id: int, name: str) -> ExecutionJob:
    return ExecutionJob(
        job_id=job_id,
        job_type=JobType.SCENARIO,
        scenario_name=name,
        output_subdir=name,
        status=JobStatus.PENDING,
    )


@pytest.fixture
def mgr(tmp_path: Path) -> ExecutionManager:
    m = _make_manager(tmp_path)
    m._jobs = [_pending_scenario(1, "a"), _pending_scenario(2, "b")]
    m._max_workers = 5          # don't let the thread ceiling bind first
    m._running_count = 1        # something already runs ⇒ skip "always run one"
    # Simulate a system the live free-RAM check always rejects.
    m._memory_admits = lambda candidate, source, est: False  # type: ignore[assignment]
    return m


class TestForceStartAdmission:
    def test_unforced_is_held_when_memory_full(self, mgr: ExecutionManager) -> None:
        with mgr._lock:
            chosen = mgr._pick_next_pending()
        assert chosen is None
        assert mgr._memory_limited is True
        assert mgr._running_count == 1

    def test_first_job_always_admitted_when_nothing_runs(
        self, mgr: ExecutionManager
    ) -> None:
        # With nothing running, the first scenario starts even though the
        # memory check would reject it — a single oversized scenario gets its
        # shot and a tight machine never deadlocks.
        mgr._running_count = 0
        with mgr._lock:
            chosen = mgr._pick_next_pending()
        assert chosen is not None and chosen.job_id == 1
        assert chosen.status is JobStatus.RUNNING
        assert mgr._memory_limited is False
        assert mgr._running_count == 1

    def test_forced_job_bypasses_reserve_check(self, mgr: ExecutionManager) -> None:
        mgr.set_force_start(2, True)
        with mgr._lock:
            chosen = mgr._pick_next_pending()
        assert chosen is not None and chosen.job_id == 2
        assert chosen.status is JobStatus.RUNNING
        assert mgr._memory_limited is False
        assert mgr._running_count == 2

    def test_forced_job_still_respects_thread_limit(self, mgr: ExecutionManager) -> None:
        mgr.set_force_start(2, True)
        mgr._running_count = mgr._max_workers  # pool full
        with mgr._lock:
            chosen = mgr._pick_next_pending()
        assert chosen is None
        assert mgr._thread_limited is True
        assert mgr._memory_limited is False

    def test_unforcing_restores_memory_limit(self, mgr: ExecutionManager) -> None:
        mgr.set_force_start(2, True)
        mgr.set_force_start(2, False)
        with mgr._lock:
            chosen = mgr._pick_next_pending()
        assert chosen is None
        assert mgr._memory_limited is True

    def test_set_force_start_ignores_non_pending(self, mgr: ExecutionManager) -> None:
        mgr._jobs[0].status = JobStatus.RUNNING
        mgr.set_force_start(1, True)
        assert mgr._jobs[0].force_start is False

    def test_set_force_start_ignores_non_scenario(self, tmp_path: Path) -> None:
        m = _make_manager(tmp_path)
        aux = ExecutionJob(job_id=9, job_type=JobType.OUTPUT_ACTION,
                           status=JobStatus.PENDING, display_name="plots")
        m._jobs = [aux]
        m.set_force_start(9, True)
        assert aux.force_start is False


class TestMemoryGuardSwitch:
    def test_guard_off_does_not_bypass_admission(self, mgr: ExecutionManager) -> None:
        # The guard governs killing only; an unforced job on a full system
        # is still held back regardless of the switch.
        mgr.memory_guard_enabled = False
        with mgr._lock:
            chosen = mgr._pick_next_pending()
        assert chosen is None
        assert mgr._memory_limited is True
        assert mgr._running_count == 1

    def test_guard_off_force_still_admits(self, mgr: ExecutionManager) -> None:
        # Force start remains the deliberate way past the memory limit,
        # independent of the guard switch.
        mgr.memory_guard_enabled = False
        mgr.set_force_start(2, True)
        with mgr._lock:
            chosen = mgr._pick_next_pending()
        assert chosen is not None and chosen.job_id == 2
        assert mgr._memory_limited is False

    def test_guard_default_on(self, tmp_path: Path) -> None:
        assert _make_manager(tmp_path).memory_guard_enabled is True

    def test_guard_state_surfaced_in_status(self, mgr: ExecutionManager) -> None:
        assert mgr.get_execution_status()["memory_guard"] is True
        mgr.memory_guard_enabled = False
        assert mgr.get_execution_status()["memory_guard"] is False
