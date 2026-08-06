"""Reusable launcher for RP-preprocess and calibration auxiliary jobs.

The ``CalibrateDialog`` (a separate task) resolves per-scenario knobs and then
calls :func:`launch_rp_jobs` / :func:`launch_calibration_jobs` from here. This
module is the ONLY place that turns those resolved knobs into running
subprocesses; it does not re-render command lines — it delegates to the single
argv renderers in :mod:`flextool.gui.calibrate_commands`.

Why NOT the scheduler
---------------------
``ExecutionManager``'s scheduler dispatches only ``JobType.SCENARIO`` jobs
(``_pick_next_pending`` filters SCENARIO; ``any_active`` is SCENARIO-only). RP
and calibration runs therefore use the *auxiliary-job* pattern:
``add_auxiliary_job`` registers a tracked, streamable job entry, and THIS module
runs the subprocess in its own background thread — draining stdout line by line
into ``append_stdout`` and finalising with ``finish_job``. Aux jobs are
rendered generically by the execution window (only SCENARIO is special-cased),
so they reuse ``JobType.OUTPUT_ACTION`` and carry a descriptive
``display_name``; no new ``JobType`` is required.

Concurrency cap
---------------
Aux jobs bypass the scheduler's ``max_workers`` gate, so naively threading N
calibrations would launch all N heavy solves at once and thrash memory. Each
job kind is funnelled through a single module-global FIFO worker
(:class:`_SerialRunner`): calibrations run strictly one at a time, and RP jobs
(lighter, but serialized for the same simplicity) likewise. The cap is the
runner, not an incidental side effect — submitting M jobs enqueues M callables
that the one worker thread drains sequentially. Separate runners for RP and
calibration mean the two kinds don't block each other, only themselves.

Watchdog interaction
--------------------
``MemoryWatchdog`` picks its kill victim as the RUNNING job with the largest
``rss - memory_cap_gb`` overage under global memory pressure. A long
calibration has ``memory_cap_gb == 0`` (no reliable estimate), so its overage
would equal its whole RSS — it would be the automatic victim and get killed
mid-run. We take the smallest, least-surprising fix: mark these aux jobs
``watchdog_exempt = True`` (a one-field addition on ``ExecutionJob``). The
watchdog still tracks their peak RSS but never selects them to kill. We chose
this over faking a ``memory_cap_gb`` estimate (Option A) because a made-up
estimate would still be fragile — a calibration that legitimately grows past it
would become the victim again — whereas the exemption states the intent
directly: these user-initiated, non-scenario runs are not memory-managed.

Thread safety
-------------
All subprocess work happens off the Tk thread. ``append_stdout`` and
``finish_job`` lock internally, so the worker can call them freely.
"""
from __future__ import annotations

import logging
import os
import queue
import re
import subprocess
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from flextool.gui.calibrate_commands import (
    build_calibrate_command,
    build_rp_command,
    command_to_display_string,
    overshoot_pct_to_multiplier,
)
from flextool.gui.execution_manager import ExecutionManager, JobType

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Job specifications (what the dialog hands us)
# --------------------------------------------------------------------------- #

@dataclass
class RpJobSpec:
    """One representative-periods preprocess run.

    ``db_url`` / ``scenario`` identify the target; the remaining fields are the
    resolved knobs consumed verbatim by :func:`build_rp_command`. RP writes back
    into the database, so no per-scenario output directories are needed.
    """

    db_url: str
    scenario: str
    n_rp: int
    period_length: int
    force_sustained: bool = False
    force_peak: bool = False
    force_window: int | None = None
    solves: Sequence[str] = field(default_factory=list)
    alternative_name: str | None = None
    alternative_description: str | None = None
    # Optional post-success hook, invoked with this spec's ``scenario`` AFTER
    # the RP subprocess exits 0 and the job is finalised SUCCESS. The dialog
    # uses it to append the freshly written RP alternative onto the scenario's
    # stack (``add_alternative_to_scenario``) — a step that must NOT run before
    # the async subprocess finishes, or "Run calibrations" would not see the
    # new periods. It runs on the RP worker thread (off the Tk thread); any
    # exception it raises is caught and streamed into the job log rather than
    # crashing the worker (see ``_run_aux_subprocess``).
    on_success: Callable[[str], None] | None = None


@dataclass
class CalibJobSpec:
    """One calibration run.

    ``db_url`` / ``scenario`` identify the target. ``overshoot_pct`` is the
    dialog's planning-safety-margin PERCENT (0 = off); the launcher converts it
    to the CLI multiplier via :func:`overshoot_pct_to_multiplier`. The three
    per-scenario directories are derived under the project path by the launcher
    (see :func:`_calib_dirs`) — leave them ``None`` to use the derived defaults,
    or set them to override.
    """

    db_url: str
    scenario: str
    iterations: int
    sizing: str = "uniform"
    overshoot_pct: float = 0.0
    damping_first: float = 1.0
    damping_remaining: float = 0.5
    stall_fraction: float = 0.05
    debug: bool = False
    warm_start_cache_dir: str | None = None
    work_dir: str | None = None
    output_location: str | None = None


# --------------------------------------------------------------------------- #
# Per-scenario directory derivation
# --------------------------------------------------------------------------- #

_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(scenario: str) -> str:
    """Filesystem-safe slug for a scenario name (spaces / slashes → ``_``)."""
    slug = _SAFE_RE.sub("_", scenario).strip("_")
    return slug or "scenario"


def _calib_dirs(
    project_path: Path, scenario: str, *, create: bool = True
) -> tuple[str, str, str]:
    """Return ``(warm_start_cache_dir, work_dir, output_location)`` for a scenario.

    * ``work_dir`` — ``<project>/work/calibrate_<scenario>`` (per-scenario, so
      concurrent-nothing but repeat runs of different scenarios don't collide).
    * ``warm_start_cache_dir`` — ``<work_dir>/warm_start_cache`` (the calibrate
      CLI's own default relative to the work dir; made explicit here so the
      copied command line is self-contained).
    * ``output_location`` — the project root. The calibrate CLI writes solve
      results under ``<output-location>/output_parquet/<scenario>/`` and report
      CSVs under ``<output-location>/calibration_reports/<scenario>/``, matching
      the scenario runner's convention of one shared output root (both trees are
      per-scenario so calibrating several scenarios never overwrites).

    When ``create`` is True (the launch path) all three directories are created
    eagerly so the subprocess never races to ``mkdir`` a parent. The live CLI
    preview passes ``create=False`` so refreshing it on every keystroke does
    not litter empty directories.
    """
    safe = _safe_name(scenario)
    work_dir = project_path / "work" / f"calibrate_{safe}"
    warm_start = work_dir / "warm_start_cache"
    output_location = project_path
    if create:
        for d in (work_dir, warm_start, output_location / "output_parquet"):
            d.mkdir(parents=True, exist_ok=True)
    return str(warm_start), str(work_dir), str(output_location)


# --------------------------------------------------------------------------- #
# Serial FIFO worker (the concurrency cap)
# --------------------------------------------------------------------------- #

class _SerialRunner:
    """A single background thread draining a FIFO queue of callables.

    Guarantees that at most one submitted callable runs at a time and that they
    run in submission order. The worker is a lazily-started daemon thread that
    parks on the queue when idle; it dies with the process. ``submit`` is
    thread-safe. ``wait`` (test helper) blocks until the queue is drained.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def submit(self, fn) -> None:
        self._q.put(fn)
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._loop, name=self._name, daemon=True
                )
                self._thread.start()

    def _loop(self) -> None:
        while True:
            fn = self._q.get()
            try:
                fn()
            except Exception:  # never let the worker thread die
                logger.exception("%s: job callable raised", self._name)
            finally:
                self._q.task_done()

    def wait(self, timeout: float | None = None) -> None:
        """Block until every submitted callable has finished (test helper)."""
        # queue.Queue.join() has no timeout; emulate one so tests can't hang.
        if timeout is None:
            self._q.join()
            return
        deadline = threading.Event()
        t = threading.Timer(timeout, deadline.set)
        t.daemon = True
        t.start()
        try:
            while self._q.unfinished_tasks and not deadline.is_set():
                deadline.wait(0.02)
        finally:
            t.cancel()


# Module-global runners: one per job kind, each serialising its own kind across
# every launch call for the process lifetime.
_CALIBRATE_RUNNER = _SerialRunner("flextool-calibrate")
_RP_RUNNER = _SerialRunner("flextool-rp")


# --------------------------------------------------------------------------- #
# The worker body: run one aux-job subprocess to completion
# --------------------------------------------------------------------------- #

def _run_aux_subprocess(
    mgr: ExecutionManager,
    *,
    display_name: str,
    action_key: str,
    argv: list[str],
    cwd: str,
    on_success: Callable[[], None] | None = None,
) -> None:
    """Register an auxiliary job, run ``argv`` as a subprocess, stream + finalise.

    Robust by construction: any exception is caught, logged into the job's own
    stream, and the job is finalised FAILED — this never raises out of the
    worker thread. A nonzero exit (including the known intermittent exit-144)
    finalises FAILED with the streamed log left intact.

    ``on_success`` (optional) is a zero-arg callable run AFTER the job is
    finalised SUCCESS, on this same worker thread. It is where post-run wiring
    (e.g. appending the RP alternative to the scenario stack) belongs, so it
    cannot run before the subprocess has actually finished. Its own failure is
    caught and streamed into the job log — it never propagates out of the
    worker and never flips the already-recorded SUCCESS.
    """
    job = mgr.add_auxiliary_job(JobType.OUTPUT_ACTION, display_name, action_key)
    # Mark exempt from the memory watchdog (long run, no cap estimate).
    job.watchdog_exempt = True

    # Log the exact command first, mirroring the scenario runner.
    mgr.append_stdout(job.job_id, command_to_display_string(argv))
    mgr.append_stdout(job.job_id, "")

    success = False
    try:
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=cwd,
            env=env,
        )
        with mgr._lock:
            job.process = proc
        assert proc.stdout is not None
        for line in proc.stdout:
            mgr.append_stdout(job.job_id, line.rstrip("\n"))
        return_code = proc.wait()
        success = return_code == 0
        if not success:
            mgr.append_stdout(
                job.job_id, f"Process exited with code {return_code}"
            )
    except Exception as exc:  # never propagate out of the worker
        logger.exception("Auxiliary job '%s' failed", display_name)
        mgr.append_stdout(job.job_id, f"Job failed with exception: {exc}")
        success = False
    finally:
        with mgr._lock:
            job.process = None
        mgr.finish_job(job.job_id, success)

    # Post-success wiring runs only after a clean, finalised success. A
    # failure here is surfaced in the job log (not raised) so a broken scenario
    # hand-off is visible without killing the worker or the other queued jobs.
    if success and on_success is not None:
        try:
            on_success()
        except Exception as exc:
            logger.exception("Post-success hook for '%s' failed", display_name)
            mgr.append_stdout(
                job.job_id, f"Post-run wiring failed: {exc}"
            )


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #

def launch_rp_jobs(
    mgr: ExecutionManager,
    *,
    python_exe: str,
    project_path: Path,
    jobs: Sequence[RpJobSpec],
    runner: _SerialRunner | None = None,
) -> None:
    """Enqueue one serialized RP-preprocess aux job per scenario.

    Returns immediately; the jobs run one at a time on the RP runner. The
    optional ``runner`` override exists for tests (inject a local
    :class:`_SerialRunner` to wait on it deterministically).
    """
    run = runner if runner is not None else _RP_RUNNER
    cwd = str(project_path)
    for spec in jobs:
        argv = build_rp_command(
            python_exe,
            spec.db_url,
            spec.scenario,
            n_rp=spec.n_rp,
            period_length=spec.period_length,
            force_sustained=spec.force_sustained,
            force_peak=spec.force_peak,
            force_window=spec.force_window,
            solves=list(spec.solves),
            alternative_name=spec.alternative_name,
            alternative_description=spec.alternative_description,
        )
        display_name = f"RP: {spec.scenario}"
        action_key = f"rp:{spec.scenario}"
        # Bind the spec's (scenario)->None hook into the zero-arg callable the
        # worker invokes, capturing this spec's scenario name. ``None`` when no
        # hook was supplied.
        hook = spec.on_success
        scenario = spec.scenario
        on_success = (
            (lambda h=hook, s=scenario: h(s)) if hook is not None else None
        )
        run.submit(
            lambda mgr=mgr, dn=display_name, ak=action_key, av=argv,
            os_=on_success: (
                _run_aux_subprocess(
                    mgr, display_name=dn, action_key=ak, argv=av, cwd=cwd,
                    on_success=os_,
                )
            )
        )


def launch_calibration_jobs(
    mgr: ExecutionManager,
    *,
    python_exe: str,
    project_path: Path,
    jobs: Sequence[CalibJobSpec],
    runner: _SerialRunner | None = None,
) -> None:
    """Enqueue one serialized calibration aux job per scenario.

    Returns immediately; calibrations run strictly one at a time on the
    calibration runner. Per-scenario directories are derived (and created)
    under ``project_path`` unless the spec overrides them. The optional
    ``runner`` override exists for tests.
    """
    run = runner if runner is not None else _CALIBRATE_RUNNER
    cwd = str(project_path)
    for spec in jobs:
        warm_start, work_dir, output_location = _calib_dirs(
            project_path, spec.scenario
        )
        argv = build_calibrate_command(
            python_exe,
            spec.db_url,
            spec.scenario,
            iterations=spec.iterations,
            sizing=spec.sizing,
            overshoot=overshoot_pct_to_multiplier(spec.overshoot_pct),
            damping_first=spec.damping_first,
            damping_remaining=spec.damping_remaining,
            stall_fraction=spec.stall_fraction,
            warm_start_cache_dir=spec.warm_start_cache_dir or warm_start,
            work_dir=spec.work_dir or work_dir,
            output_location=spec.output_location or output_location,
            debug=spec.debug,
        )
        display_name = f"Calibrate: {spec.scenario}"
        action_key = f"calibrate:{spec.scenario}"
        run.submit(
            lambda mgr=mgr, dn=display_name, ak=action_key, av=argv: (
                _run_aux_subprocess(
                    mgr, display_name=dn, action_key=ak, argv=av, cwd=cwd
                )
            )
        )
