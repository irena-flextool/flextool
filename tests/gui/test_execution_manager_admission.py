"""Tests for the live free-RAM admission model and warmed-detection.

These exercise the scheduler's admission building blocks in isolation: the
manager is constructed but never `.start()`ed, so no scheduler thread,
watchdog, or subprocess runs. ``psutil.virtual_memory`` is monkeypatched so
free RAM is deterministic.

Covers:
  * ``_compute_memory_budget_for_job`` source ladder (explicit > history > auto)
    and the 1.05 history multiplier.
  * ``_predicted_available_gb`` reserving the unrealised growth of warming jobs
    while ignoring warmed ones.
  * ``_memory_admits`` budgeting against live free RAM, the auto-stagger gate,
    and the history/explicit batch path.
  * ``_pick_next_pending`` admitting a batch of history-backed jobs up to the
    point one no longer fits, and the max_workers ceiling.
  * ``MemoryWatchdog._maybe_mark_warmed`` plateau + timeout backstop.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

import flextool.gui.execution_manager as em
from flextool.gui.data_models import ExecutionLimits, ProjectSettings, ScenarioRun
from flextool.gui.execution_manager import (
    ExecutionJob,
    ExecutionManager,
    JobStatus,
    JobType,
    MemoryWatchdog,
)

GB = 1024 ** 3


class _FakeVM:
    def __init__(self, available_gb: float, total_gb: float = 64.0) -> None:
        self.available = int(available_gb * GB)
        self.total = int(total_gb * GB)


@pytest.fixture
def patch_ram(monkeypatch):
    """Return a setter that pins ``psutil.virtual_memory`` to a fixed free-GB."""
    def _set(available_gb: float, total_gb: float = 64.0) -> None:
        monkeypatch.setattr(
            em.psutil, "virtual_memory",
            lambda: _FakeVM(available_gb, total_gb),
        )
    return _set


def _make_manager(tmp_path: Path, reserve_gb: float = 2.0) -> ExecutionManager:
    settings = ProjectSettings()
    settings.execution_limits = ExecutionLimits(system_reserve_gb=reserve_gb)
    return ExecutionManager(project_path=tmp_path, settings=settings)


def _scenario(job_id: int, name: str, status: JobStatus = JobStatus.PENDING) -> ExecutionJob:
    return ExecutionJob(
        job_id=job_id,
        job_type=JobType.SCENARIO,
        scenario_name=name,
        output_subdir=name,
        status=status,
    )


# --------------------------------------------------------------------------- #
# Budget source ladder
# --------------------------------------------------------------------------- #

class TestBudgetSource:
    def test_explicit_cap_wins(self, tmp_path: Path) -> None:
        m = _make_manager(tmp_path)
        m.settings.execution_limits.memory_cap_per_job_gb = 7.0
        m.settings.scenario_resource_history["a"] = ScenarioRun(
            peak_rss_mb=2048.0, runtime_s=1.0, last_run="",
        )
        gb, source = m._compute_memory_budget_for_job(_scenario(1, "a"))
        assert source == "explicit"
        assert gb == 7.0

    def test_history_beats_auto_and_uses_1_05(self, tmp_path: Path) -> None:
        m = _make_manager(tmp_path)
        m.settings.scenario_resource_history["a"] = ScenarioRun(
            peak_rss_mb=4096.0, runtime_s=1.0, last_run="",
        )
        gb, source = m._compute_memory_budget_for_job(_scenario(1, "a"))
        assert source == "history"
        assert gb == pytest.approx(4.0 * 1.05)

    def test_auto_fallback_fair_share(self, tmp_path: Path, patch_ram) -> None:
        patch_ram(available_gb=10.0, total_gb=64.0)
        m = _make_manager(tmp_path, reserve_gb=4.0)
        m._max_workers = 6
        gb, source = m._compute_memory_budget_for_job(_scenario(1, "a"))
        assert source == "auto"
        assert gb == pytest.approx((64.0 - 4.0) / 6)


# --------------------------------------------------------------------------- #
# Predicted-available accounting
# --------------------------------------------------------------------------- #

class TestPredictedAvailable:
    def test_warming_job_reserves_shortfall(self, tmp_path: Path) -> None:
        m = _make_manager(tmp_path)
        warming = _scenario(1, "a", JobStatus.RUNNING)
        warming.memory_cap_gb = 4.0
        warming.peak_rss_mb = 1.0 * 1024  # 1 GB allocated so far → shortfall 3
        m._jobs = [warming]
        assert m._predicted_available_gb(10.0) == pytest.approx(7.0)

    def test_warmed_job_reserves_nothing(self, tmp_path: Path) -> None:
        m = _make_manager(tmp_path)
        warmed = _scenario(1, "a", JobStatus.RUNNING)
        warmed.memory_cap_gb = 4.0
        warmed.peak_rss_mb = 2.0 * 1024
        warmed.footprint_solid = True       # solver running ⇒ RSS is truth
        m._jobs = [warmed]
        assert m._predicted_available_gb(10.0) == pytest.approx(10.0)

    def test_overshooting_warming_job_floors_at_zero(self, tmp_path: Path) -> None:
        m = _make_manager(tmp_path)
        warming = _scenario(1, "a", JobStatus.RUNNING)
        warming.memory_cap_gb = 4.0
        warming.peak_rss_mb = 9.0 * 1024  # already past its estimate → no extra
        m._jobs = [warming]
        assert m._predicted_available_gb(10.0) == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# Admission decision
# --------------------------------------------------------------------------- #

class TestMemoryAdmits:
    def test_history_admits_when_it_fits(self, tmp_path: Path, patch_ram) -> None:
        patch_ram(available_gb=10.0)
        m = _make_manager(tmp_path, reserve_gb=2.0)
        cand = _scenario(2, "b")
        assert m._memory_admits(cand, "history", 4.0) is True   # 10-4=6 >= 2

    def test_history_rejected_when_it_would_breach_reserve(
        self, tmp_path: Path, patch_ram
    ) -> None:
        patch_ram(available_gb=10.0)
        m = _make_manager(tmp_path, reserve_gb=2.0)
        cand = _scenario(2, "b")
        assert m._memory_admits(cand, "history", 9.0) is False  # 10-9=1 < 2

    def test_auto_staggers_while_a_job_is_warming(
        self, tmp_path: Path, patch_ram
    ) -> None:
        patch_ram(available_gb=64.0)  # plenty of RAM
        m = _make_manager(tmp_path)
        warming = _scenario(1, "a", JobStatus.RUNNING)
        warming.memory_cap_gb = 1.0  # tiny — RAM is not the issue here
        m._jobs = [warming]
        # Auto estimate is an unreliable guess: refuse to start a second
        # unknown until the first reveals its real footprint, even with RAM free.
        assert m._memory_admits(_scenario(2, "b"), "auto", 1.0) is False

    def test_auto_admits_when_nothing_is_warming(
        self, tmp_path: Path, patch_ram
    ) -> None:
        patch_ram(available_gb=64.0)
        m = _make_manager(tmp_path)
        warmed = _scenario(1, "a", JobStatus.RUNNING)
        warmed.memory_cap_gb = 1.0
        warmed.footprint_solid = True
        m._jobs = [warmed]
        assert m._memory_admits(_scenario(2, "b"), "auto", 1.0) is True

    def test_history_does_not_stagger_behind_a_warming_job(
        self, tmp_path: Path, patch_ram
    ) -> None:
        patch_ram(available_gb=64.0)
        m = _make_manager(tmp_path)
        warming = _scenario(1, "a", JobStatus.RUNNING)
        warming.memory_cap_gb = 4.0
        warming.peak_rss_mb = 0.0  # reserves its full 4 GB
        m._jobs = [warming]
        # Solid estimate ⇒ budget directly; 64 - 4 (warming) - 4 (cand) >= 2.
        assert m._memory_admits(_scenario(2, "b"), "history", 4.0) is True


# --------------------------------------------------------------------------- #
# Integrated pick: history batch up to budget + thread ceiling
# --------------------------------------------------------------------------- #

class TestPickBatch:
    def test_history_batch_admits_until_one_no_longer_fits(
        self, tmp_path: Path, patch_ram
    ) -> None:
        # 20 GB free, 2 GB reserve, each scenario learned at ~4 GB. Expect 4
        # admitted (4*4=16, leaving 4 >= 2) and the 5th held: predicted after
        # 4 warming jobs is 20-16=4, and 4-4=0 < 2.
        patch_ram(available_gb=20.0)
        m = _make_manager(tmp_path, reserve_gb=2.0)
        m._max_workers = 10
        for i in range(5):
            name = chr(ord("a") + i)
            job = _scenario(i + 1, name)
            m.settings.scenario_resource_history[name] = ScenarioRun(
                peak_rss_mb=(4.0 / 1.05) * 1024, runtime_s=1.0, last_run="",
            )
            m._jobs.append(job)

        admitted = []
        for _ in range(10):
            with m._lock:
                chosen = m._pick_next_pending()
            if chosen is None:
                break
            # Simulate dispatch: the worker would set memory_cap_gb; here the
            # job is still warming (footprint not solid, RSS still 0).
            chosen.memory_cap_gb, _ = m._compute_memory_budget_for_job(chosen)
            admitted.append(chosen.job_id)

        assert admitted == [1, 2, 3, 4]
        assert m._memory_limited is True
        assert m._running_count == 4

    def test_thread_ceiling_binds_before_memory(
        self, tmp_path: Path, patch_ram
    ) -> None:
        patch_ram(available_gb=64.0)
        m = _make_manager(tmp_path)
        m._max_workers = 2
        m._running_count = 2
        m._jobs = [_scenario(1, "a")]
        with m._lock:
            chosen = m._pick_next_pending()
        assert chosen is None
        assert m._thread_limited is True
        assert m._memory_limited is False


# --------------------------------------------------------------------------- #
# Warmed-detection backstop (RSS plateau + timeout)
# --------------------------------------------------------------------------- #

class TestWarmedBackstop:
    def _watchdog(self, tmp_path: Path) -> tuple[ExecutionManager, MemoryWatchdog]:
        m = _make_manager(tmp_path)
        return m, MemoryWatchdog(m)

    def test_plateau_flips_after_two_flat_polls(self, tmp_path: Path) -> None:
        m, wd = self._watchdog(tmp_path)
        job = _scenario(1, "a", JobStatus.RUNNING)
        job.start_time = datetime.now()
        with m._lock:
            wd._maybe_mark_warmed(job, 4000.0)   # first reading, seeds prev
            assert job.footprint_solid is False
            wd._maybe_mark_warmed(job, 4050.0)   # +1.25% ≤ 3% → stable 1
            assert job.footprint_solid is False
            wd._maybe_mark_warmed(job, 4080.0)   # +0.7% ≤ 3% → stable 2 → warmed
            assert job.footprint_solid is True

    def test_growth_resets_the_plateau_counter(self, tmp_path: Path) -> None:
        m, wd = self._watchdog(tmp_path)
        job = _scenario(1, "a", JobStatus.RUNNING)
        job.start_time = datetime.now()
        with m._lock:
            wd._maybe_mark_warmed(job, 4000.0)
            wd._maybe_mark_warmed(job, 4050.0)   # stable 1
            wd._maybe_mark_warmed(job, 6000.0)   # +48% > 3% → reset
            assert job.footprint_solid is False
            assert job._warm_stable_polls == 0

    def test_small_rss_never_plateaus(self, tmp_path: Path) -> None:
        m, wd = self._watchdog(tmp_path)
        job = _scenario(1, "a", JobStatus.RUNNING)
        job.start_time = datetime.now()
        with m._lock:
            for _ in range(5):
                wd._maybe_mark_warmed(job, 100.0)  # < WARM_MIN_RSS_MB floor
            assert job.footprint_solid is False

    def test_timeout_flips_regardless(self, tmp_path: Path) -> None:
        m, wd = self._watchdog(tmp_path)
        job = _scenario(1, "a", JobStatus.RUNNING)
        job.start_time = datetime.now() - timedelta(
            seconds=em.WARM_TIMEOUT_S + 1
        )
        with m._lock:
            wd._maybe_mark_warmed(job, 100.0)
        assert job.footprint_solid is True
