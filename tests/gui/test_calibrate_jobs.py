"""Headless tests for the RP / calibration auxiliary-job launcher.

No real solve runs: the argv builders are monkeypatched to return tiny
``python -c`` fake commands so the worker's streaming, finalisation,
serialization, and watchdog-exemption behaviour can be exercised deterministically
against a real ``ExecutionManager`` (constructed but never ``.start()``ed — no
scheduler thread or watchdog thread runs).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import flextool.gui.calibrate_jobs as cj
import flextool.gui.execution_manager as em
from flextool.gui.calibrate_jobs import (
    CalibJobSpec,
    RpJobSpec,
    _SerialRunner,
    launch_calibration_jobs,
    launch_rp_jobs,
)
from flextool.gui.data_models import ExecutionLimits, ProjectSettings
from flextool.gui.execution_manager import (
    ExecutionJob,
    ExecutionManager,
    JobStatus,
    JobType,
    MemoryWatchdog,
)

GB = 1024 ** 3


def _make_manager(
    tmp_path: Path, *, reserve_gb: float = 2.0, swap_allow_gb: float = 1.0
) -> ExecutionManager:
    settings = ProjectSettings()
    settings.execution_limits = ExecutionLimits(
        system_reserve_gb=reserve_gb, swap_allowance_gb=swap_allow_gb
    )
    return ExecutionManager(project_path=tmp_path, settings=settings)


def _calib_spec(scenario: str) -> CalibJobSpec:
    return CalibJobSpec(
        db_url="sqlite:///m.sqlite",
        scenario=scenario,
        iterations=1,
    )


def _find_job(mgr: ExecutionManager, action_key: str) -> ExecutionJob:
    for j in mgr.get_jobs():
        if j.action_key == action_key:
            return j
    raise AssertionError(f"no job with action_key {action_key!r}")


# --------------------------------------------------------------------------- #
# Streaming + finalisation
# --------------------------------------------------------------------------- #

def _fake_argv(script: str, *extra: str) -> list[str]:
    return [sys.executable, "-c", script, *extra]


def test_streams_stdout_and_finishes_success(tmp_path: Path, monkeypatch) -> None:
    script = "import sys; print('hello from calib'); print('iteration 1/1'); sys.exit(0)"
    monkeypatch.setattr(cj, "build_calibrate_command", lambda *a, **k: _fake_argv(script))
    mgr = _make_manager(tmp_path)
    runner = _SerialRunner("test")

    launch_calibration_jobs(
        mgr, python_exe=sys.executable, project_path=tmp_path,
        jobs=[_calib_spec("base")], runner=runner,
    )
    runner.wait(timeout=30)

    job = _find_job(mgr, "calibrate:base")
    assert job.status == JobStatus.SUCCESS
    assert job.job_type == JobType.OUTPUT_ACTION
    assert job.display_name == "Calibrate: base"
    assert job.watchdog_exempt is True
    assert "hello from calib" in job.stdout_lines
    assert "iteration 1/1" in job.stdout_lines
    # The exact command is logged first, mirroring the scenario runner.
    assert any("-c" in line for line in job.stdout_lines[:2])


def test_nonzero_exit_144_marks_failed(tmp_path: Path, monkeypatch) -> None:
    # exit-144 is the known intermittent calibrator exit; it must FAIL, not raise.
    script = "import sys; print('boom before exit'); sys.exit(144)"
    monkeypatch.setattr(cj, "build_calibrate_command", lambda *a, **k: _fake_argv(script))
    mgr = _make_manager(tmp_path)
    runner = _SerialRunner("test")

    launch_calibration_jobs(
        mgr, python_exe=sys.executable, project_path=tmp_path,
        jobs=[_calib_spec("s1")], runner=runner,
    )
    runner.wait(timeout=30)

    job = _find_job(mgr, "calibrate:s1")
    assert job.status == JobStatus.FAILED
    assert "boom before exit" in job.stdout_lines
    assert "Process exited with code 144" in job.stdout_lines


def test_rp_job_streams_and_succeeds(tmp_path: Path, monkeypatch) -> None:
    script = "import sys; print('rp preprocess done'); sys.exit(0)"
    monkeypatch.setattr(cj, "build_rp_command", lambda *a, **k: _fake_argv(script))
    mgr = _make_manager(tmp_path)
    runner = _SerialRunner("test")

    launch_rp_jobs(
        mgr, python_exe=sys.executable, project_path=tmp_path,
        jobs=[RpJobSpec(db_url="sqlite:///m.sqlite", scenario="base", n_rp=4, period_length=24)],
        runner=runner,
    )
    runner.wait(timeout=30)

    job = _find_job(mgr, "rp:base")
    assert job.status == JobStatus.SUCCESS
    assert job.display_name == "RP: base"
    assert "rp preprocess done" in job.stdout_lines


# --------------------------------------------------------------------------- #
# Concurrency cap: calibration jobs never overlap
# --------------------------------------------------------------------------- #

def test_calibration_jobs_are_serialized(tmp_path: Path, monkeypatch) -> None:
    marker = tmp_path / "timeline.txt"
    # Each fake process records ENTER, sleeps, then EXIT into a shared file.
    # If the launcher serializes correctly the file strictly alternates
    # ENTER / EXIT; any overlap would produce two consecutive ENTERs.
    script = (
        "import sys, time\n"
        "p = sys.argv[1]\n"
        "open(p, 'a').write('ENTER\\n')\n"
        "time.sleep(0.3)\n"
        "open(p, 'a').write('EXIT\\n')\n"
    )
    monkeypatch.setattr(
        cj, "build_calibrate_command",
        lambda *a, **k: _fake_argv(script, str(marker)),
    )
    mgr = _make_manager(tmp_path)
    runner = _SerialRunner("test")

    launch_calibration_jobs(
        mgr, python_exe=sys.executable, project_path=tmp_path,
        jobs=[_calib_spec("a"), _calib_spec("b"), _calib_spec("c")],
        runner=runner,
    )
    runner.wait(timeout=60)

    events = marker.read_text().split()
    assert events == ["ENTER", "EXIT"] * 3, events
    # Sanity: all three jobs exist and succeeded.
    for name in ("a", "b", "c"):
        assert _find_job(mgr, f"calibrate:{name}").status == JobStatus.SUCCESS


# --------------------------------------------------------------------------- #
# Watchdog exemption: a calibration job is never the kill victim
# --------------------------------------------------------------------------- #

class _FakeProc:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.killed = False

    def poll(self):
        return None  # still running

    def kill(self):
        self.killed = True


def _running(job_id: int, *, pid: int, cap_gb: float, exempt: bool) -> ExecutionJob:
    return ExecutionJob(
        job_id=job_id,
        job_type=JobType.OUTPUT_ACTION if exempt else JobType.SCENARIO,
        scenario_name=f"job{job_id}",
        status=JobStatus.RUNNING,
        process=_FakeProc(pid),
        memory_cap_gb=cap_gb,
        watchdog_exempt=exempt,
    )


def test_exempt_calibration_job_is_not_killed(tmp_path: Path, monkeypatch) -> None:
    # RSS by pid: the exempt calibration (pid 20) is the BIGGEST overage
    # (20 GB rss, 0 estimate) and would be the victim without the exemption;
    # the normal scenario (pid 10) is 3 GB rss over a 1 GB estimate.
    rss_by_pid = {10: 3 * GB, 20: 20 * GB}

    class _FakePsProc:
        def __init__(self, pid):
            self._rss = rss_by_pid[pid]

        def memory_info(self):
            return types.SimpleNamespace(rss=self._rss)

        def children(self, recursive=False):
            return []

    monkeypatch.setattr(em.psutil, "Process", _FakePsProc)
    monkeypatch.setattr(
        em.psutil, "virtual_memory",
        lambda: types.SimpleNamespace(available=int(0.5 * GB), total=64 * GB),
    )
    monkeypatch.setattr(
        em.psutil, "swap_memory",
        lambda: types.SimpleNamespace(used=int(5 * GB)),
    )

    mgr = _make_manager(tmp_path, reserve_gb=2.0, swap_allow_gb=1.0)
    normal = _running(1, pid=10, cap_gb=1.0, exempt=False)
    calib = _running(2, pid=20, cap_gb=0.0, exempt=True)
    mgr._jobs = [normal, calib]

    wd = MemoryWatchdog(mgr)
    # Drive exactly one poll of the real _loop, then stop.
    calls = {"n": 0}

    def _wait_once(timeout=None):
        calls["n"] += 1
        return calls["n"] > 1  # False first (run body), True after (exit)

    wd._stop.wait = _wait_once  # type: ignore[method-assign]
    wd._loop()

    # The normal scenario is killed; the exempt calibration is spared even
    # though its overage is far larger. Its peak RSS is still tracked.
    assert normal.process.killed is True
    assert calib.process.killed is False
    assert normal.killed_for_memory is True
    assert calib.killed_for_memory is False
    assert calib.peak_rss_mb > 0
