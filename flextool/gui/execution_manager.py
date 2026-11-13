"""Thread-safe execution manager for running FlexTool scenarios as subprocesses."""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Callable

import psutil

from flextool._child_python import SAFE_PATH_FLAG, child_env
from flextool._safe_text import printable
from flextool.safe_names import describe_name, output_name_problem, require_strictly_inside
from flextool._process_tree import kill_process_tree
from flextool.gui.cli_format import format_cmd_for_log, terminal_command
from flextool.gui.data_models import TIMESTAMP_FORMAT, GlobalSettings, ProjectSettings, ScenarioInfo
from flextool.gui.scenario_key import choose_output_subdir_for_write
from flextool.run_record import (
    RUN_LOG_OWNER_ENV,
    ExpectedRun,
    RunLogOwner,
    now_iso,
    read_status_settled,
    results_usable,
    write_stop_request,
)
from flextool.sqlite_url import sqlite_url_string

logger = logging.getLogger(__name__)


_slice_probe_cache: dict[str, bool] = {}

# Why the GUI stopped a job: the ``kill_reason`` of the job and the error message of its run record.
KILL_REASON_WINDOW = "stopped from the FlexTool window"
KILL_REASON_SOURCE_REMOVED = "its input source was removed"
KILL_REASON_CLOSED = "FlexTool was closed"
KILL_REASON_REPLACED = "replaced by a new run of the same action"
KILL_REASON_PROJECT_DELETED = "the project was deleted"
KILL_REASON_PROJECT_SWITCHED = "the project was switched"

#: How long :meth:`ExecutionManager.cleanup` waits for the stopped jobs to finish their run records.
CLEANUP_WAIT_S = 15.0
#: How long the ``flextool-run`` supervisors of a stopped auxiliary job get to stop their engines and record the
#: runs as killed, before what is left of the job's processes is killed.
AUX_STOP_GRACE_S = 10.0


def _is_run_supervisor(process: psutil.Process) -> bool:
    """Whether *process* is a ``flextool-run`` supervisor (``python -m flextool_run_supervisor``, the console
    script ``flextool-run``)."""
    try:
        cmdline = process.cmdline()
    except (psutil.Error, OSError):
        return False
    if not cmdline:
        return False
    if any("flextool_run_supervisor" in part for part in cmdline):
        return True
    return os.path.basename(cmdline[0]).lower() in ("flextool-run", "flextool-run.exe")


def _kill_quietly(process: psutil.Process) -> None:
    try:
        process.kill()
    except (psutil.Error, OSError):
        pass


def stop_aux_tree(
    proc: subprocess.Popen, stop_file: Path, reason: str, *, grace_s: float = AUX_STOP_GRACE_S,
) -> None:
    """Stop an auxiliary job (a calibration, a representative-period iteration) so that each engine run under it
    is recorded as ``killed`` with *reason*.

    Its ``flextool-run`` supervisors own the engines' run records, so they are not killed with the rest: *reason*
    goes into *stop_file* first (a supervisor started meanwhile then starts nothing), the job's process tree is
    listed, the job's own process and everything that is neither a supervisor nor under one is killed (also the
    processes an engine started, so its pipe closes at once), each supervisor gets *grace_s* to kill its engine
    and finish the record, and then whatever is left is killed.  The folder of *stop_file* (the job's own,
    from ``tempfile.mkdtemp``) is removed at the end.  Blocking: run it in its own thread.
    """
    try:
        write_stop_request(stop_file, reason)
    except OSError as exc:
        logger.warning("Cannot write the stop file %s (%s); killing the job's processes.", stop_file, exc)
        kill_process_tree(proc)
        shutil.rmtree(stop_file.parent, ignore_errors=True)
        return
    try:
        snapshot = psutil.Process(proc.pid).children(recursive=True)
    except (psutil.Error, OSError):
        snapshot = []
    parents: dict[int, int] = {}
    for process in snapshot:
        try:
            parents[process.pid] = process.ppid()
        except (psutil.Error, OSError):
            pass
    supervisors = [process for process in snapshot if _is_run_supervisor(process)]
    supervisor_pids = {process.pid for process in supervisors}

    def under_supervisor(pid: int) -> int:
        """How many levels below a supervisor *pid* is (0: not under one)."""
        depth, current = 0, pid
        while current in parents:
            current = parents[current]
            depth += 1
            if current in supervisor_pids:
                return depth
        return 0

    try:
        proc.kill()
    except OSError:
        pass
    for process in snapshot:
        if process.pid in supervisor_pids:
            continue
        # Not under a supervisor, or below a supervisor's engine (the engine's own children): killed now.
        if under_supervisor(process.pid) != 1:
            _kill_quietly(process)
    if supervisors:
        _gone, alive = psutil.wait_procs(supervisors, timeout=grace_s)
        if alive:
            logger.warning("%d flextool-run supervisor(s) did not stop within %.0f s; their run records stay as "
                           "they are.", len(alive), grace_s)
    for process in snapshot:
        _kill_quietly(process)
    shutil.rmtree(stop_file.parent, ignore_errors=True)


def _wait_done(wait: threading.Event | threading.Thread) -> bool:
    """Whether a worker's ``worker_done`` event is set, or a stop thread has ended."""
    if isinstance(wait, threading.Thread):
        return not wait.is_alive()
    return wait.is_set()


def _slice_exists(name: str) -> bool:
    """Return True if a user-level systemd slice with this name is loaded.

    Result is cached for the process lifetime so repeated job spawns don't
    re-probe. Probe failures (missing systemctl, timeout) cache as False.
    """
    if name in _slice_probe_cache:
        return _slice_probe_cache[name]
    loaded = False
    try:
        result = subprocess.run(
            ["systemctl", "--user", "show", name, "--property=LoadState"],
            capture_output=True, text=True, timeout=2.0, check=False,
        )
        loaded = "LoadState=loaded" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        loaded = False
    _slice_probe_cache[name] = loaded
    if not loaded:
        logger.warning(
            "FLEXTOOL_SLICE=%s requested but slice is not loaded; "
            "running without slice isolation.", name,
        )
    return loaded


def _wrap_linux(
    cmd: list[str],
    estimate_gb: float,
) -> tuple[list[str], dict, Callable[[subprocess.Popen], object | None] | None]:
    """Wrap via systemd-run user scope for slice isolation; no memory cap.

    MemoryHigh throttles allocations (stalls the process) and MemoryMax kills
    it — both interfere with large polars collect_all calls.  The watchdog is
    the sole memory enforcer.  systemd-run is still used so the process can be
    placed in a cgroup slice (FLEXTOOL_SLICE) for scheduling isolation.
    """
    if not shutil.which("systemd-run"):
        return cmd, {}, None
    wrapped = ["systemd-run", "--user", "--scope", "--quiet"]
    slice_name = os.environ.get("FLEXTOOL_SLICE")
    if slice_name and _slice_exists(slice_name):
        wrapped.append(f"--slice={slice_name}")
    wrapped.append("--")
    wrapped.extend(cmd)
    return wrapped, {}, None


def _wrap_windows(
    cmd: list[str],
    estimate_gb: float,
) -> tuple[list[str], dict, Callable[[subprocess.Popen], object | None] | None]:
    """No OS-level memory cap on Windows — MemoryWatchdog is the sole enforcer."""
    return cmd, {}, None


def _wrap_macos(
    cmd: list[str],
    estimate_gb: float,
) -> tuple[list[str], dict, Callable[[subprocess.Popen], object | None] | None]:
    """No OS-level memory cap on macOS — MemoryWatchdog is the sole enforcer."""
    return cmd, {}, None


def _wrap_for_memory_cap(
    cmd: list[str],
    estimate_gb: float,
) -> tuple[list[str], dict, Callable[[subprocess.Popen], object | None] | None]:
    """Optionally wrap a command with a soft memory hint and return Popen extras.

    Returns a triple ``(wrapped_cmd, popen_extras, post_spawn)``:

    * ``wrapped_cmd`` — argv to hand to ``subprocess.Popen``.
    * ``popen_extras`` — kwargs to merge into the ``Popen`` call.
    * ``post_spawn`` — optional callable invoked with the spawned ``Popen``;
      its return value MUST be retained by the caller until the child exits.

    Per-platform mechanism:

    * Linux: transient ``systemd-run --user --scope`` for cgroup slice
      isolation only (``FLEXTOOL_SLICE``). No memory property is set —
      ``MemoryHigh`` throttles allocations and ``MemoryMax`` kills the
      process, both of which stall or abort large solver runs.
    * Windows / macOS: no OS-level wrapping.

    ``MemoryWatchdog`` is the sole memory enforcer on all platforms.
    """
    if estimate_gb < 0:
        estimate_gb = 0
    if sys.platform.startswith("linux"):
        return _wrap_linux(cmd, estimate_gb)
    if sys.platform == "win32":
        return _wrap_windows(cmd, estimate_gb)
    if sys.platform == "darwin":
        return _wrap_macos(cmd, estimate_gb)
    return cmd, {}, None


# --- Solver "warmed" detection ---------------------------------------------
# A scenario job is "warming" from dispatch until its solver has loaded the
# matrix and begun reserving its working memory — the high-water point at
# which its measured RSS is a reliable proxy for its real footprint. Until
# then admission only has an *estimate* to go on (see ExecutionJob.footprint_
# solid and ExecutionManager admission).
#
# These strings are matched case-insensitively as substrings of the job's
# stdout. NONE is load-bearing on its own: the watchdog's RSS-plateau detector
# (see MemoryWatchdog._loop) is the wording-proof, log-level-proof backstop, so
# a HiGHS release that renames any of them — or a quiet ``--solver-log-level``
# that suppresses the block entirely — degrades to a ~10s-later flip rather
# than breaking. ``tests/gui/test_warm_markers.py`` pins them so silent drift
# turns a test red.
_SOLVER_WARM_MARKERS: tuple[str, ...] = (
    "coefficient ranges",          # HiGHS load-time statistics block
    "running highs",               # HiGHS startup banner
    "presolving model",            # HiGHS presolve phase
    "solving lp",                  # HiGHS LP solve phase
    "solving mip",                 # HiGHS MIP solve phase
    "matrix built by polar-high",  # our own engine phase checkpoint
)


def _is_solver_warm_marker(line: str) -> bool:
    """True if *line* shows the solver has begun optimising (memory reserved)."""
    low = line.lower()
    return any(m in low for m in _SOLVER_WARM_MARKERS)


# RSS-plateau backstop (used when no marker is seen — e.g. quiet solver log or
# suppressed phase checkpoints on within-group rolling iters):
#  * consider a job warmed once its RSS grows < WARM_PLATEAU_RATIO between two
#    consecutive polls, for WARM_PLATEAU_POLLS polls in a row, and it has
#    allocated at least WARM_MIN_RSS_MB (HiGHS has actually started);
#  * WARM_TIMEOUT_S is the last-resort anti-deadlock cap.
WARM_PLATEAU_RATIO = 1.03
WARM_PLATEAU_POLLS = 2
WARM_MIN_RSS_MB = 200.0
WARM_TIMEOUT_S = 180.0


class MemoryWatchdog:
    """Polls running scenario jobs every 5s; tracks peak RSS and enforces
    global memory/swap thresholds by killing the job most over its
    budget."""

    POLL_INTERVAL_S = 5.0

    def __init__(self, manager: "ExecutionManager") -> None:
        self._manager = manager
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Captured at start(): swap usage attributable to other processes
        # (kswapd cache of inactive pages, other apps, etc.). Only swap
        # GROWTH past this baseline counts toward swap_allowance_gb.
        self._baseline_swap_used: int = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        try:
            self._baseline_swap_used = psutil.swap_memory().used
        except Exception:
            self._baseline_swap_used = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _maybe_mark_warmed(self, job: "ExecutionJob", rss_mb: float) -> None:
        """Backstop warmed-detection from measured RSS. Holds ``_manager._lock``.

        Flips ``footprint_solid`` once the job's RSS has plateaued (grew less
        than ``WARM_PLATEAU_RATIO`` for ``WARM_PLATEAU_POLLS`` consecutive
        polls, having allocated at least ``WARM_MIN_RSS_MB``) or a hard
        ``WARM_TIMEOUT_S`` has elapsed. The stdout warm-marker (see
        ``_run_job``) usually flips it sooner; this covers quiet solver logs
        and suppressed phase checkpoints so the auto-stagger never deadlocks.
        """
        if job.footprint_solid:
            return
        if job.start_time is not None:
            elapsed = (datetime.now() - job.start_time).total_seconds()
            if elapsed > WARM_TIMEOUT_S:
                job.footprint_solid = True
                return
        prev = job._warm_last_rss_mb
        if rss_mb >= WARM_MIN_RSS_MB and prev > 0 and rss_mb <= prev * WARM_PLATEAU_RATIO:
            job._warm_stable_polls += 1
        else:
            job._warm_stable_polls = 0
        job._warm_last_rss_mb = rss_mb
        if job._warm_stable_polls >= WARM_PLATEAU_POLLS:
            job.footprint_solid = True

    def _loop(self) -> None:
        while not self._stop.wait(self.POLL_INTERVAL_S):
            with self._manager._lock:
                snapshot = [
                    (j, j.process, j.memory_cap_gb, j.watchdog_exempt)
                    for j in self._manager._jobs
                    if j.status == JobStatus.RUNNING and j.process is not None
                ]

            measured: list[tuple[ExecutionJob, int, int]] = []
            for job, proc, est_gb, exempt in snapshot:
                if proc.poll() is not None:
                    continue
                try:
                    parent = psutil.Process(proc.pid)
                    rss = parent.memory_info().rss
                    for child in parent.children(recursive=True):
                        try:
                            rss += child.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                rss_mb = rss / (1024 ** 2)
                with self._manager._lock:
                    if rss_mb > job.peak_rss_mb:
                        job.peak_rss_mb = rss_mb
                    self._maybe_mark_warmed(job, rss_mb)
                # Exempt jobs (calibrator / RP aux jobs) still have their peak
                # RSS tracked above, but are never eligible to be killed: keep
                # them out of the victim-selection candidate list.
                if exempt:
                    continue
                est_bytes = int(est_gb * (1024 ** 3))
                measured.append((job, rss, est_bytes))

            if not measured:
                continue

            # Peaks above are always tracked (they feed the learned budget),
            # but with the memory guard switched off we never kill: the user
            # has accepted the OOM risk for this session. Admission still
            # honours the reserve, so this only disables killing.
            if not self._manager._memory_guard_enabled:
                continue

            limits = self._manager.execution_limits
            reserve_bytes = int(limits.system_reserve_gb * (1024 ** 3))
            swap_allow_bytes = int(limits.swap_allowance_gb * (1024 ** 3))

            try:
                vm = psutil.virtual_memory()
                sm = psutil.swap_memory()
            except Exception:
                continue

            # Both thresholds are safety nets, not independent triggers.
            # We only kill when BOTH are exhausted: the system is short
            # on free RAM AND swap growth has reached the allowance.
            # Plenty of free RAM ⇒ never kill, regardless of swap growth
            # (Linux's kswapd may proactively swap inactive pages even
            # when RAM is plentiful, which is harmless).
            reason = ""
            if vm.available < reserve_bytes:
                swap_growth = max(0, sm.used - self._baseline_swap_used)
                if swap_growth > swap_allow_bytes:
                    reason = (
                        f"system out of memory: {vm.available / 1024**3:.1f} GB free "
                        f"(reserve {limits.system_reserve_gb:.1f} GB) and swap grew "
                        f"{swap_growth / 1024**3:.1f} GB since FlexTool started "
                        f"> allowance {limits.swap_allowance_gb:.1f} GB"
                    )

            if not reason:
                continue

            victim, victim_rss, victim_est = max(
                measured, key=lambda t: t[1] - t[2]
            )
            overage_gb = (victim_rss - victim_est) / 1024**3
            logger.warning(
                "Killing job %d (%s): %s. RSS %.1f GB, estimate %.1f GB (overage %+.1f GB).",
                victim.job_id,
                victim.scenario_name or victim.display_name,
                reason,
                victim_rss / 1024**3,
                victim_est / 1024**3,
                overage_gb,
            )
            with self._manager._lock:
                victim.killed_for_memory = True
                victim.kill_reason = reason
            vproc = victim.process
            if vproc is not None:
                kill_process_tree(vproc)


class JobStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    KILLED = auto()


class JobType(Enum):
    """Distinguishes different kinds of jobs in the execution list."""

    SCENARIO = auto()       # FlexTool scenario run
    CONVERSION = auto()     # xlsx → sqlite pre-conversion
    OUTPUT_ACTION = auto()  # output generation (plots, Excel, CSV, comparison)
    OLD_CONVERT = auto()    # old FlexTool 2.0 import
    MIGRATION = auto()      # database schema / data version migration
    UPDATE = auto()         # FlexTool self-update (git pull / pip upgrade)
    DB_EDITOR = auto()      # Spine DB Editor launch (recorded only on failure)
    ENV_REPAIR = auto()     # native-library compatibility fix (polars compat runtime, highspy pin)


@dataclass
class ExecutionJob:
    """Represents a single job in the execution list.

    For scenario runs, the scenario-specific fields (``scenario_name``,
    ``source_name``, etc.) are populated.  For auxiliary jobs (conversions,
    output actions) only ``display_name`` and ``action_key`` are used.
    """

    job_id: int
    job_type: JobType = JobType.SCENARIO
    display_name: str = ""   # Shown in the tree for non-scenario jobs
    action_key: str = ""     # For replacement: same key → old entry removed
    # --- Scenario-specific fields ---
    scenario_name: str = ""
    source_name: str = ""
    source_number: int = 0
    output_subdir: str = ""  # on-disk folder under output_parquet/
    input_db_url: str = ""   # sqlite:///path/to/input.sqlite
    is_xlsx_source: bool = False
    xlsx_path: Path | None = None
    # --- Common fields ---
    status: JobStatus = JobStatus.PENDING
    stdout_lines: list[str] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None
    process: subprocess.Popen | None = None
    finish_timestamp: str = ""  # DD.MM.YY hh:mm format
    memory_cap_gb: float = 0.0       # snapshot of estimate at dispatch (0 = none)
    peak_rss_mb: float = 0.0         # high-water mark, fed by MemoryWatchdog
    # ``footprint_solid`` flips True once the solver has begun optimising — by a
    # stdout warm-marker (fast path) or the watchdog's RSS-plateau/timeout
    # backstop. Until then the job is "warming": admission only has its
    # estimate (``memory_cap_gb``) to reserve against; afterwards its measured
    # RSS (already reflected in ``vm.available``) is the truth. See admission.
    footprint_solid: bool = False
    _warm_stable_polls: int = 0      # consecutive flat-RSS watchdog polls
    _warm_last_rss_mb: float = 0.0   # RSS at the previous watchdog poll
    killed_for_memory: bool = False  # set by watchdog just before SIGTERM
    kill_reason: str = ""  # populated by watchdog: "exceeded estimate" / "global memory pressure" / "global swap pressure"
    force_start: bool = False  # user override: admit despite the memory-reserve check (watchdog still applies)
    # ``watchdog_exempt`` removes a RUNNING job from the watchdog's kill-victim
    # selection. Long calibrator / RP auxiliary jobs run with no reliable
    # ``memory_cap_gb`` estimate (0), so their ``rss - estimate`` overage would
    # equal their whole RSS and make them the automatic victim under global
    # memory pressure — killed mid-run through no fault of their own. Their peak
    # RSS is still tracked; they are only excluded from being chosen to kill.
    # Set by ``flextool/gui/calibrate_jobs.py``. See MemoryWatchdog._loop.
    watchdog_exempt: bool = False
    native_fault: bool = False  # set on finalize when the process died from a native crash (segfault / illegal instruction)
    # The run folder whose _run_status.json / _run.log this job's engine writes, set when the engine's marker
    # arrives (its record says "running" from then on): the main window refreshes its executed list once then.
    run_record_folder: Path | None = None
    # Set by the job's worker thread as its very last step, after the run record is finished and the job's
    # output folder is free again.  ``cleanup`` waits for it.
    worker_done: threading.Event = field(default_factory=threading.Event)
    # An auxiliary job whose process starts ``flextool-run`` supervisors: the file that asks them to stop
    # (``FLEXTOOL_RUN_STOP_FILE``), and whether a stop through it has started (its stop thread then owns the file).
    stop_file: Path | None = None
    stop_started: bool = False
    # Opaque per-platform handle returned by `_wrap_for_memory_cap` post_spawn
    # (e.g. Windows Job Object). Must outlive the Popen, else the cap is
    # released when the last handle closes. None on platforms that don't need it.
    memory_cap_handle: object | None = None


class ExecutionManager:
    """Thread-safe manager for parallel FlexTool scenario execution."""

    def __init__(
        self,
        project_path: Path,
        settings: ProjectSettings,
        on_status_change: Callable[[ExecutionJob], None] | None = None,
        on_all_finished: Callable[[], None] | None = None,
        global_settings: GlobalSettings | None = None,
        on_run_record_closed: Callable[[ExecutionJob], None] | None = None,
    ):
        """
        Args:
            project_path: Path to the project directory.
            settings: Project settings (auto-generate flags, plot settings).
            on_status_change: Callback when a job's status changes.  Called from
                a worker thread -- the caller should use ``root.after()`` to
                safely update tkinter.
            on_all_finished: Callback when all jobs are done (for post-execution
                hooks like comparison).
            global_settings: Optional global settings (for execution limits).
            on_run_record_closed: Callback, from the job's worker thread, after
                the GUI has finished a scenario job's run record (a kill, a
                failure, a run that ended before its record started).  The
                job's status notify came earlier, while the record still said
                ``running``: this one is when the executed list can show the
                final state.
        """
        self._lock = threading.Lock()
        # Separate lock for settings.yaml saves so worker threads don't
        # serialize on the manager lock during file I/O.
        self._settings_save_lock = threading.Lock()
        self._jobs: list[ExecutionJob] = []
        self._next_id = 0
        # Seed max_workers: prefer the per-project value, fall back to the
        # legacy global preference, finally to the dataclass default of 1.
        if settings.max_workers > 0:
            self._max_workers = max(1, settings.max_workers)
        elif global_settings is not None and global_settings.max_workers > 0:
            self._max_workers = max(1, global_settings.max_workers)
        else:
            self._max_workers = 1
        self._running_count = 0
        # The output folders of scenario jobs from their pick until their worker has finished the run record
        # (also of a job ``remove_jobs_for_source`` took off the list while its worker still runs).
        self._subdirs_in_use: set[str] = set()
        # What ``cleanup`` waits for: each scenario worker's ``worker_done`` and each running aux-job stop thread,
        # whether or not the job is still on the list.
        self._live_waits: set[threading.Event | threading.Thread] = set()
        self._wind_down = False
        self._stopped = False
        self._memory_limited: bool = False  # set by scheduler when admission blocked by RAM
        self._thread_limited: bool = False  # set by scheduler when running == max_workers and pending exist
        # Memory-guard switch (session-only, never persisted). When False the
        # watchdog stops KILLING jobs; it does not change how many jobs start
        # (the reserve still limits admission — ``force_start`` is the
        # deliberate per-job override). So only Allow swap (a watchdog-only
        # threshold) becomes non-binding. Defaults on; re-disable each session.
        self._memory_guard_enabled: bool = True
        self.project_path = project_path
        self.settings = settings
        self._global_settings = global_settings
        self._on_status_change = on_status_change
        self._on_all_finished = on_all_finished
        self._on_run_record_closed = on_run_record_closed
        self._scheduler_thread: threading.Thread | None = None
        self._comparison_scenarios: list[str] = []
        self._pending_select_job_id: int | None = None  # for auto-selecting new jobs in UI

        self._converted_xlsx: set[str] = set()  # source_names already converted
        self._watchdog: MemoryWatchdog | None = None

        # Register cleanup for when the Python process exits
        atexit.register(self.cleanup)

    @property
    def execution_limits(self):
        """Return the active ExecutionLimits from ProjectSettings."""
        from flextool.gui.data_models import ExecutionLimits
        proj = getattr(self.settings, "execution_limits", None)
        return proj if proj is not None else ExecutionLimits()

    def set_global_settings(self, gs: GlobalSettings) -> None:
        self._global_settings = gs

    @property
    def memory_guard_enabled(self) -> bool:
        """Whether the watchdog may kill jobs under memory pressure.

        Session-only; not persisted to settings.yaml. When False the
        watchdog still tracks peak RSS but never kills — the OS is the only
        backstop. Admission (how many jobs start) is unaffected: the reserve
        still limits it, and ``force_start`` is the per-job override.
        """
        return self._memory_guard_enabled

    @memory_guard_enabled.setter
    def memory_guard_enabled(self, value: bool) -> None:
        with self._lock:
            self._memory_guard_enabled = bool(value)

    def _compute_memory_budget_for_job(
        self, job: ExecutionJob | None = None
    ) -> tuple[float, str]:
        """Return ``(per_job_budget_gb, source)``. Budget 0 means none set.

        Resolution order (``source`` in parentheses):
          1. ``"explicit"`` — user-set ``ExecutionLimits.memory_cap_per_job_gb``
             when > 0. A *soft* target, not a guarantee.
          2. ``"history"`` — learned peak from a prior successful run in
             ``ProjectSettings.scenario_resource_history``, ×1.05 for safety.
          3. ``"auto"`` — fair-share fallback ``(system_total - reserve) /
             max_workers``.

        The ``source`` drives admission (see ``_memory_admits``): ``explicit``
        and ``history`` are solid enough to admit a batch up front, whereas
        ``auto`` is an unreliable guess that must be staggered one job at a
        time until its real footprint is observed.
        """
        limits = self.execution_limits
        if limits.memory_cap_per_job_gb > 0:
            return limits.memory_cap_per_job_gb, "explicit"
        if job is not None and job.output_subdir:
            history = self.settings.scenario_resource_history.get(job.output_subdir)
            if history is not None and history.peak_rss_mb > 0:
                return (history.peak_rss_mb / 1024.0) * 1.05, "history"
        try:
            total_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            return 0.0, "auto"
        available = max(0.0, total_gb - limits.system_reserve_gb)
        workers = max(1, self._max_workers)
        return available / workers, "auto"

    def _job_warming(self, job: ExecutionJob) -> bool:
        """A RUNNING scenario job whose solver hasn't begun optimising yet —
        its real memory footprint is still unknown. Holds ``self._lock``."""
        return (
            job.job_type == JobType.SCENARIO
            and job.status == JobStatus.RUNNING
            and not job.footprint_solid
        )

    def _predicted_available_gb(self, vm_available_gb: float) -> float:
        """Free RAM debited by the *unrealised* growth of still-warming jobs.

        ``vm.available`` already reflects every running job's RSS so far. A
        warming job will keep growing toward its estimate, so we reserve the
        shortfall ``max(0, estimate - measured_rss)``. Warmed jobs are fully
        reflected in ``vm.available`` and reserve nothing — their measured RSS
        is the truth, not the estimate. Holds ``self._lock``.
        """
        reserved = 0.0
        for j in self._jobs:
            if not self._job_warming(j) or j.memory_cap_gb <= 0:
                continue
            measured_gb = j.peak_rss_mb / 1024.0
            reserved += max(0.0, j.memory_cap_gb - measured_gb)
        return vm_available_gb - reserved

    def _memory_admits(
        self, candidate: ExecutionJob, source: str, estimate_gb: float
    ) -> bool:
        """Decide whether live free RAM allows admitting *candidate*.

        Must hold ``self._lock``; ``_running_count`` is > 0 here (the first job
        is admitted unconditionally upstream). Admission keys off actual
        ``vm.available``, re-measured every tick, not a static reservation
        against total RAM — so freed memory lets more jobs start.

        * ``auto`` source (no cap, no history): the estimate is an unreliable
          fair-share guess, so *stagger* — refuse while any job is still
          warming, to observe one job's real footprint before starting the
          next unknown.
        * ``explicit`` / ``history`` source: the estimate is solid, so budget
          directly against predicted-available RAM with no staggering; a burst
          of pending jobs is admitted up to the point one no longer fits.
        """
        if estimate_gb <= 0:
            return True  # no estimate; let the watchdog handle it
        if source == "auto" and any(self._job_warming(j) for j in self._jobs):
            return False
        try:
            vm_available_gb = psutil.virtual_memory().available / (1024 ** 3)
        except Exception:
            return True
        limits = self.execution_limits
        predicted = self._predicted_available_gb(vm_available_gb)
        return predicted - estimate_gb >= limits.system_reserve_gb

    def _record_scenario_peak(self, job: ExecutionJob, runtime_s: float) -> None:
        """Persist this job's peak RSS into the project settings history.

        Called once per successful scenario run. Auxiliary jobs are skipped.
        """
        if job.job_type != JobType.SCENARIO:
            return
        if not job.output_subdir or job.peak_rss_mb <= 0:
            return
        from flextool.gui.data_models import ScenarioRun
        record = ScenarioRun(
            peak_rss_mb=float(job.peak_rss_mb),
            runtime_s=float(runtime_s),
            last_run=datetime.now().isoformat(timespec="seconds"),
        )
        with self._lock:
            self.settings.scenario_resource_history[job.output_subdir] = record
        with self._settings_save_lock:
            try:
                from flextool.gui.settings_io import save_project_settings
                save_project_settings(self.project_path, self.settings)
            except Exception:
                logger.warning(
                    "Could not persist scenario_resource_history for '%s'",
                    job.output_subdir, exc_info=True,
                )

    # ------------------------------------------------------------------
    # max_workers property
    # ------------------------------------------------------------------

    @property
    def max_workers(self) -> int:
        return self._max_workers

    @max_workers.setter
    def max_workers(self, value: int) -> None:
        self._max_workers = max(1, value)

    def _resolve_source_path(self, source_name: str) -> Path:
        """Return the absolute path to a source file, honouring external refs."""
        rel = self.settings.external_refs.get(source_name)
        if rel is not None:
            return (self.project_path / rel).resolve()
        return self.project_path / "input_sources" / source_name

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------

    def add_jobs(self, scenarios: list[ScenarioInfo]) -> list[ExecutionJob]:
        """Add scenarios to the execution queue. Returns the created jobs."""
        new_jobs: list[ExecutionJob] = []
        ownership_changed = False
        with self._lock:
            for scenario in scenarios:
                # The name becomes work/, output_parquet/... folder names: never queue one that is not a plain
                # folder name (the main window tells the user before it gets here).
                reason = output_name_problem(scenario.name)
                if reason is not None:
                    logger.warning("Scenario %s is not run: its name %s.", describe_name(scenario.name), reason)
                    continue
                is_xlsx = scenario.source_name.lower().endswith(
                    (".xlsx", ".xls", ".ods")
                )
                source_path = self._resolve_source_path(scenario.source_name)
                if is_xlsx:
                    xlsx_path = source_path
                    stem = Path(scenario.source_name).stem
                    input_db_url = sqlite_url_string(
                        self.project_path / "intermediate" / f"{stem}.sqlite"
                    )
                else:
                    xlsx_path = None
                    input_db_url = sqlite_url_string(source_path)

                prev_owners = dict(self.settings.bare_output_owners)
                output_subdir = choose_output_subdir_for_write(
                    self.project_path,
                    self.settings.bare_output_owners,
                    scenario.source_number,
                    scenario.name,
                )
                if self.settings.bare_output_owners != prev_owners:
                    ownership_changed = True

                job = ExecutionJob(
                    job_id=self._next_id,
                    scenario_name=scenario.name,
                    source_name=scenario.source_name,
                    source_number=scenario.source_number,
                    output_subdir=output_subdir,
                    input_db_url=input_db_url,
                    is_xlsx_source=is_xlsx,
                    xlsx_path=xlsx_path,
                )
                self._next_id += 1
                self._jobs.append(job)
                new_jobs.append(job)

        if ownership_changed:
            try:
                from flextool.gui.settings_io import save_project_settings
                save_project_settings(self.project_path, self.settings)
            except Exception:
                logger.warning(
                    "Could not persist bare_output_owners to settings.yaml",
                    exc_info=True,
                )

        # Upfront, additive seeding of input-DB entity colors into the
        # project plot_settings.yaml (Stage 5).  This runs ONCE here in the
        # GUI thread (the single writer) — scenarios themselves run as
        # parallel subprocesses, so per-scenario writes would race.  We seed
        # OUTSIDE self._lock: the DB fetch + file splice is slow I/O that must
        # not hold the manager lock and block other job operations.  The jobs
        # created above are all SCENARIO jobs (add_jobs only enqueues
        # scenarios; auxiliary jobs go through add_auxiliary_job).
        #
        # Collect the DISTINCT input DB URLs of the newly-added jobs whose
        # backing sqlite file exists on disk.  xlsx sources point at a
        # not-yet-created intermediate/<stem>.sqlite; those are skipped here
        # and picked up on a later batch once the intermediate DB exists
        # (known limitation).  Best-effort / non-fatal: any failure is logged
        # and never breaks job queuing.  GUI-execution path only; the headless
        # CLI path is out of scope.
        input_db_urls = list(
            dict.fromkeys(j.input_db_url for j in new_jobs if j.input_db_url)
        )
        if input_db_urls:
            try:
                from flextool.scenario_comparison.input_entity_colors import (
                    seed_input_entity_colors,
                )
                seed_input_entity_colors(self.project_path, input_db_urls)
            except Exception:
                logger.warning(
                    "Upfront input-DB color seeding failed (non-fatal)",
                    exc_info=True,
                )

        return new_jobs

    # ------------------------------------------------------------------
    # Auxiliary (non-scenario) jobs
    # ------------------------------------------------------------------

    def add_auxiliary_job(
        self,
        job_type: JobType,
        display_name: str,
        action_key: str,
        *,
        insert_before_source: str | None = None,
    ) -> ExecutionJob:
        """Create a non-scenario job (conversion, output action, etc.).

        Auxiliary jobs are **not** managed by the scheduler — the caller is
        responsible for running them in a thread and calling
        :meth:`append_stdout` / :meth:`finish_job`.

        If a finished or running job with the same *action_key* already
        exists, it is killed (if running) and removed first.

        Args:
            job_type: The kind of auxiliary job.
            display_name: What to show in the job tree.
            action_key: Unique key for replacement logic.
            insert_before_source: If given, insert the job right before the
                first SCENARIO job with this ``source_name`` (used for
                conversion jobs).  Otherwise appended at end.
        """
        with self._lock:
            # Kill and remove old job with the same action_key
            for old in list(self._jobs):
                if old.action_key and old.action_key == action_key:
                    self._stop_job_locked(old, KILL_REASON_REPLACED)
                    self._jobs.remove(old)

            job = ExecutionJob(
                job_id=self._next_id,
                job_type=job_type,
                display_name=display_name,
                action_key=action_key,
                status=JobStatus.RUNNING,
                start_time=datetime.now(),
            )
            self._next_id += 1

            # Determine insertion position
            if insert_before_source is not None:
                idx = len(self._jobs)
                for i, j in enumerate(self._jobs):
                    if (
                        j.job_type == JobType.SCENARIO
                        and j.source_name == insert_before_source
                    ):
                        idx = i
                        break
                self._jobs.insert(idx, job)
            else:
                self._jobs.append(job)

            self._pending_select_job_id = job.job_id

        self._notify_status_change(job)
        return job

    def append_stdout(self, job_id: int, line: str) -> None:
        """Thread-safe append of a stdout line to a job (control characters shown as ``\\xNN``)."""
        line = printable(line, keep_newlines=True)
        with self._lock:
            job = self._get_job(job_id)
            if job is not None:
                job.stdout_lines.append(line)

    def finish_job(self, job_id: int, success: bool) -> None:
        """Mark an auxiliary job as finished (SUCCESS or FAILED); a job the GUI stopped stays KILLED."""
        with self._lock:
            job = self._get_job(job_id)
            if job is None:
                return
            job.process = None
            if job.status is JobStatus.KILLED:
                return
            now = datetime.now()
            job.end_time = now
            job.finish_timestamp = now.strftime(TIMESTAMP_FORMAT)
            job.status = JobStatus.SUCCESS if success else JobStatus.FAILED
        self._notify_status_change(job)

    def remove_jobs_for_source(self, source_name: str) -> None:
        """Remove all jobs related to *source_name*.

        This removes:
        - Scenario jobs whose ``source_name`` matches
        - Auxiliary jobs whose ``action_key`` or ``display_name`` contains
          the source name or its stem (e.g. deleting ``model.sqlite`` also
          matches ``old_convert:model.xlsm`` because the stem ``model``
          is the same).

        Running or pending jobs are stopped first (recorded as killed: "its input source was removed").  A
        removed job's worker still finishes its run record, and its output folder stays busy until then.
        """
        stem = Path(source_name).stem
        with self._lock:
            to_remove: list[ExecutionJob] = []
            for job in self._jobs:
                match = False
                if job.job_type == JobType.SCENARIO and job.source_name == source_name:
                    match = True
                elif job.action_key and (
                    source_name in job.action_key or stem in job.action_key
                ):
                    match = True
                elif job.display_name and (
                    source_name in job.display_name or stem in job.display_name
                ):
                    match = True
                if match:
                    self._stop_job_locked(job, KILL_REASON_SOURCE_REMOVED)
                    to_remove.append(job)
            for job in to_remove:
                self._jobs.remove(job)

    def start(self) -> None:
        """Start executing pending jobs up to *max_workers* concurrently.

        Launches a scheduler thread that monitors and dispatches jobs.
        """
        with self._lock:
            if self._scheduler_thread is not None and self._scheduler_thread.is_alive():
                return
            self._wind_down = False
            self._stopped = False

        # Pre-create the shared results.sqlite once, in-process, before any
        # scenario subprocess runs. Each run writes its alternative at the
        # end via the spinedb write-method; spinedb_api gives writers a
        # 1800s busy_timeout so concurrent *appends* to an existing file
        # serialize safely. The unsafe part is the create-from-JSON path,
        # whose guard is a non-atomic os.path.exists() — on a project's
        # first batch every parallel subprocess would otherwise find the
        # file missing and race to build the schema. Doing it here once
        # closes that race without touching the engine writer.
        self._precreate_results_db_if_needed()

        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop, daemon=True
        )
        self._scheduler_thread.start()

        if self._watchdog is None:
            self._watchdog = MemoryWatchdog(self)
        self._watchdog.start()

    def _precreate_results_db_if_needed(self) -> None:
        """Initialize the shared results.sqlite schema once before the batch.

        No-op unless the SpineDB output is enabled and at least one scenario
        job is pending. Idempotent (``ensure_results_db`` short-circuits on
        an existing file). Non-fatal: on failure the scenario subprocesses
        fall back to creating it themselves (the pre-existing behaviour).
        """
        if not getattr(self.settings, "auto_generate_comp_spinedb", False):
            return
        with self._lock:
            has_pending = any(
                j.job_type == JobType.SCENARIO and j.status == JobStatus.PENDING
                for j in self._jobs
            )
        if not has_pending:
            return
        db_url = sqlite_url_string(self.project_path / "results.sqlite")
        try:
            from flextool.process_outputs.write_spinedb import ensure_results_db
            ensure_results_db(db_url)
        except Exception:
            logger.warning(
                "Could not pre-create results SpineDB at %s; scenario runs "
                "will attempt to create it (possible cold-start race).",
                db_url, exc_info=True,
            )

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def _pick_next_pending(self) -> "ExecutionJob | None":
        """Choose the next pending scenario job to dispatch this tick.

        Must be called while holding ``self._lock``. Resets and sets the
        ``_thread_limited`` / ``_memory_limited`` status flags, and on a
        successful pick increments ``_running_count``. Returns the chosen
        job, or ``None`` when nothing can start.

        Admission rules, in order:
          1. Thread slots (``max_workers``) always bind — a full pool sets
             ``_thread_limited`` and nothing starts, even forced jobs. This is
             the upper limit on concurrency; live RAM decides how much of it is
             actually used.
          2. A job flagged ``force_start`` bypasses the memory check and is
             admitted ahead of the queue.
          3. With nothing running, the first scenario is admitted
             unconditionally — even if its estimate says it might not fit — so
             a single oversized scenario always gets its shot and a tight
             machine never deadlocks.
          4. Otherwise the live free-RAM check (``_memory_admits``) binds;
             failing it sets ``_memory_limited``.

        On a successful pick the job is marked RUNNING here, under the lock, so
        the scheduler's immediate re-pick can't double-dispatch the same
        still-PENDING job before its worker thread starts.

        The memory-guard switch does NOT enter here: it only governs the
        watchdog's killing. How many runs start is always limited by RAM, and
        ``force_start`` is the deliberate per-job override.
        """
        self._thread_limited = False
        self._memory_limited = False
        if self._wind_down:
            return None

        candidate: ExecutionJob | None = None
        forced: ExecutionJob | None = None
        # A job whose output folder belongs to a running job, or to a pending job ahead of it, waits: two runs
        # in one folder would overwrite each other's results and run record, and the second would empty the
        # first's work folder.  A job's folder stays busy until its worker has finished the run record, after
        # the job has already ended (or been removed from the list).
        busy = set(self._subdirs_in_use) | {
            j.output_subdir for j in self._jobs
            if j.job_type == JobType.SCENARIO and j.status == JobStatus.RUNNING and j.output_subdir
        }
        for job in self._jobs:
            if job.job_type == JobType.SCENARIO and job.status == JobStatus.PENDING:
                if job.output_subdir and job.output_subdir in busy:
                    continue
                if job.output_subdir:
                    busy.add(job.output_subdir)
                if candidate is None:
                    candidate = job
                if forced is None and job.force_start:
                    forced = job
                if forced is not None:
                    break
        if candidate is None:
            return None

        if self._running_count >= self._max_workers:
            self._thread_limited = True
            return None

        if forced is not None:
            chosen = forced
        elif self._running_count == 0:
            # Always give at least one scenario its shot, fit or not.
            chosen = candidate
        else:
            estimate_gb, source = self._compute_memory_budget_for_job(candidate)
            if self._memory_admits(candidate, source, estimate_gb):
                chosen = candidate
            else:
                self._memory_limited = True
                return None

        # Mark RUNNING under the lock so the scheduler's immediate re-pick
        # doesn't see this job as still-PENDING and dispatch it twice. The
        # worker thread re-asserts this (idempotent) and sets start_time.
        chosen.status = JobStatus.RUNNING
        self._running_count += 1
        if chosen.output_subdir:
            self._subdirs_in_use.add(chosen.output_subdir)
        return chosen

    def _scheduler_loop(self) -> None:
        """Main loop of the scheduler thread.

        The lock is only held for brief critical sections (checking state,
        updating counters).  It is never held while spawning threads,
        running subprocesses, sleeping, or calling callbacks.
        """
        while True:
            # --- brief lock: read state ---
            with self._lock:
                if self._stopped:
                    break

                next_job = self._pick_next_pending()

                # Check if all SCENARIO jobs are done (auxiliary jobs are
                # managed externally and don't block the scheduler).  A job
                # whose worker is still finishing its run record counts: the
                # post-execution step reads those records.
                any_active = bool(self._subdirs_in_use) or any(
                    j.status in (JobStatus.PENDING, JobStatus.RUNNING)
                    for j in self._jobs
                    if j.job_type == JobType.SCENARIO
                )
            # --- lock released ---

            # Spawn the worker thread WITHOUT holding the lock
            if next_job is not None:
                worker = threading.Thread(
                    target=self._run_job, args=(next_job,), daemon=True
                )
                worker.start()
                continue  # Re-check immediately for more slots

            if not any_active:
                # All jobs finished — run post-execution in a separate thread
                # so the scheduler exits promptly and start() can launch a new
                # scheduler if the user submits more jobs.
                def _post():
                    self._run_post_execution()
                    if self._on_all_finished is not None:
                        try:
                            self._on_all_finished()
                        except Exception:
                            logger.exception("on_all_finished callback failed")
                threading.Thread(target=_post, daemon=True).start()
                break

            time.sleep(0.1)

    # ------------------------------------------------------------------
    # Job execution
    # ------------------------------------------------------------------

    def _run_job(self, job: ExecutionJob) -> None:
        """Run a single job in a worker thread.

        The GUI is the run's pipe owner: a :class:`RunLogOwner` (created before the engine starts, holding the
        command line) writes the engine's output into the run's ``_run.log`` and, at the end, finishes a
        ``_run_status.json`` the engine could not finish (a kill, a crash).  The order at the end is fixed: the
        engine has ended, the record is finished (then ``on_run_record_closed``), and only then is the job's output
        folder free for the next job and ``worker_done`` set.  That last step runs whatever failed before it.
        """
        with self._lock:
            self._live_waits.add(job.worker_done)
        owner: RunLogOwner | None = None
        proc: subprocess.Popen | None = None
        launch: tuple[str, list[str]] | None = None  # (launched_at, command) of the engine
        try:
            try:
                with self._lock:
                    # Stopped between its pick and this thread's start (a kill, "delete source", closing
                    # FlexTool): it never starts, and KILLED is never overwritten with RUNNING.
                    if job.status is JobStatus.KILLED:
                        return
                    job.status = JobStatus.RUNNING
                    job.start_time = datetime.now()
                self._notify_status_change(job)

                # xlsx conversion must be done before jobs are dispatched
                if job.is_xlsx_source and job.source_name not in self._converted_xlsx:
                    raise RuntimeError(
                        f"Bug: xlsx source '{job.source_name}' was not pre-converted"
                    )

                # Step 2: Build and run the main FlexTool command
                work_folder = self.project_path / "work" / job.output_subdir

                # Clean work folder so re-executions start fresh (only ever a folder under <project>/work/)
                if work_folder.exists():
                    require_strictly_inside(work_folder, self.project_path / "work")
                    shutil.rmtree(work_folder, ignore_errors=True)
                work_folder.mkdir(parents=True, exist_ok=True)

                cmd = self._build_run_command(job, work_folder)

                estimate_gb, _src = self._compute_memory_budget_for_job(job)
                with self._lock:
                    job.memory_cap_gb = estimate_gb

                wrapped_cmd, popen_extras, post_spawn = _wrap_for_memory_cap(
                    cmd, estimate_gb
                )
                logger.info("Running job %d (%s):\n%s", job.job_id, job.scenario_name, format_cmd_for_log(wrapped_cmd))
                # The command as a terminal would run it (``flextool-run ...``): the first line of the progress
                # output, for copying, and of the run's _run.log.
                cmd_str = format_cmd_for_log(terminal_command(wrapped_cmd))
                owner = RunLogOwner(launcher="gui", preamble=[cmd_str, ""])

                # Log the CLI command as the first line in the progress output
                with self._lock:
                    job.stdout_lines.append(cmd_str)
                    job.stdout_lines.append("")
                self._notify_status_change(job)

                # Run from flextool root so relative paths (templates/, etc.) work
                flextool_root = Path.cwd()  # subprocess cwd — user workspace, formerly the repo root

                # The pipe-owner variables (docs/dev/architecture.md, "Runtime options" item 4): the engine prints
                # the run-record marker instead of copying its own output into _run.log, and writes UTF-8 unbuffered.
                env = child_env(
                    PYTHONUNBUFFERED="1",
                    PYTHONIOENCODING="utf-8",
                    **{RUN_LOG_OWNER_ENV: "1"},
                )
                popen_kwargs = dict(
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                    cwd=str(flextool_root), env=env,
                )
                popen_kwargs.update(popen_extras)
                # Taken just before the start: a record in the folder that started at or after it is this
                # run's own (or a later run's), never one to replace (``record_run_without_marker``).
                launch = (now_iso(), list(cmd))
                proc = subprocess.Popen(wrapped_cmd, **popen_kwargs)
                cap_handle = post_spawn(proc) if post_spawn is not None else None
                killed = False
                with self._lock:
                    # Check if killed while we were setting up
                    if job.status == JobStatus.KILLED:
                        killed = True
                    else:
                        job.process = proc
                        job.memory_cap_handle = cap_handle

                if killed:
                    kill_process_tree(proc)
                    proc.wait()
                    return

                # Read stdout line by line
                assert proc.stdout is not None
                for line in proc.stdout:
                    # Control characters inside a line are shown as \xNN in the log window and _run.log.
                    stripped = printable(line.rstrip("\r\n"))
                    if not owner.feed(stripped):
                        # The engine's run-record marker: its record now says "running".
                        with self._lock:
                            job.run_record_folder = owner.folder
                        self._notify_status_change(job)
                        continue
                    with self._lock:
                        job.stdout_lines.append(stripped)
                        # Fast-path warmed flip: the solver has loaded the matrix
                        # and begun reserving memory, so this job's RSS is now a
                        # reliable footprint. The watchdog's RSS-plateau backstop
                        # covers quiet/suppressed logs where no marker appears.
                        if not job.footprint_solid and _is_solver_warm_marker(stripped):
                            job.footprint_solid = True
                    # Notify so the GUI can update the log view
                    self._notify_status_change(job)

                return_code = proc.wait()

                # Finalise
                now = datetime.now()
                peak_to_record: tuple[ExecutionJob, float] | None = None
                with self._lock:
                    if job.status == JobStatus.KILLED:
                        return
                    job.end_time = now
                    job.finish_timestamp = now.strftime(TIMESTAMP_FORMAT)
                    job.process = None
                    if return_code == 0:
                        job.status = JobStatus.SUCCESS
                        self.settings.scenarios_changed = True
                        runtime_s = (
                            (now - job.start_time).total_seconds()
                            if job.start_time else 0.0
                        )
                        # Capture under lock; persist below (outside lock) to
                        # avoid blocking other workers on YAML I/O.
                        peak_to_record = (job, runtime_s)
                    else:
                        job.status = JobStatus.FAILED
                        if job.killed_for_memory:
                            job.stdout_lines.append(
                                f"[execution_manager] Killed: {job.kill_reason or 'memory pressure'}"
                            )
                        from flextool import env_check
                        if env_check.is_native_fault(return_code):
                            job.native_fault = True
                            job.stdout_lines.append(
                                f"[execution_manager] Process crashed: "
                                f"{env_check.describe_fault(return_code)} "
                                f"(exit code {return_code})"
                            )
                            job.stdout_lines.append(
                                "[execution_manager] This is a native library crash, "
                                "not a model error. It may be an incompatible solver "
                                "wheel (polars or HiGHS) that FlexTool can re-install — "
                                "see the prompt at the next start, or run 'Update "
                                "FlexTool'. If the solver check passes, the cause is "
                                "usually the Python environment itself: a venv built on "
                                "Anaconda/conda, or running from a network/mapped drive. "
                                "Rebuild from a clean python.org Python on a local disk."
                            )
                        else:
                            job.stdout_lines.append(
                                f"[execution_manager] Process exited with code {return_code}"
                            )
                    self._prune_old_jobs(job)
                if peak_to_record is not None:
                    self._record_scenario_peak(*peak_to_record)
                self._notify_status_change(job)

            except Exception:
                logger.exception("Unexpected error running job %d (%s)", job.job_id, job.scenario_name)
                with self._lock:
                    if job.status is not JobStatus.KILLED:
                        job.status = JobStatus.FAILED
                        job.end_time = datetime.now()
                        job.finish_timestamp = job.end_time.strftime(TIMESTAMP_FORMAT)
                self._notify_status_change(job)
            finally:
                self._finish_run_record(job, proc, owner, launch)
        finally:
            # Last: the folder is free for the next job only once its record is finished.
            with self._lock:
                self._running_count -= 1
                if job.output_subdir:
                    self._subdirs_in_use.discard(job.output_subdir)
                self._live_waits.discard(job.worker_done)
            job.worker_done.set()

    def _finish_run_record(
        self,
        job: ExecutionJob,
        proc: subprocess.Popen | None,
        owner: RunLogOwner | None,
        launch: tuple[str, list[str]] | None = None,
    ) -> None:
        """Stop an engine still running, finish its run record and call ``on_run_record_closed``; never raises.

        An engine that ended before it started its record (an early error, a kill in its first seconds) gets a
        minimal ``failed`` / ``killed`` record in the job's folder (``record_run_without_marker``), so the
        folder's earlier results are no longer offered as the latest run's.
        """
        try:
            if proc is not None and proc.poll() is None:
                # The job failed here while its engine still runs: stop it, so no second run starts in its
                # folder and the run record below is finished.
                kill_process_tree(proc)
                proc.wait()
        except Exception:
            logger.exception("Could not stop the engine of job %d (%s)", job.job_id, job.scenario_name)
        if owner is None:
            return
        try:
            with self._lock:
                killed = job.status is JobStatus.KILLED or job.killed_for_memory
                kill_reason = job.kill_reason or KILL_REASON_WINDOW
            expected = None
            if proc is not None and launch is not None and job.output_subdir:
                folder = self.project_path / "output_parquet" / job.output_subdir
                expected = ExpectedRun(
                    folder=folder, pid=proc.pid, launched_at=launch[0], argv=launch[1],
                    input={"db_url": job.input_db_url, "scenario": job.scenario_name},
                    output={"location": str(self.project_path), "subdir": job.output_subdir,
                            "parquet_dir": str(folder)},
                )
            owner.close(
                proc.returncode if proc is not None else None,
                killed=killed,
                kill_reason=kill_reason if killed else None,
                expected=expected,
            )
        except Exception:
            logger.exception("Could not finish the run record of job %d (%s)", job.job_id, job.scenario_name)
            return
        if owner.record_changed and self._on_run_record_closed is not None:
            try:
                self._on_run_record_closed(job)
            except Exception:
                logger.exception("on_run_record_closed callback failed for job %d", job.job_id)

    def _build_run_command(self, job: ExecutionJob, work_folder: Path) -> list[str]:
        """Build the ``cmd_run_flextool`` command line."""
        settings = self.settings

        # Write methods: always include parquet
        write_methods: list[str] = ["parquet"]
        if settings.auto_generate_scen_plots:
            write_methods.append("plot")
        if settings.auto_generate_scen_excels:
            write_methods.append("excel")
        if settings.auto_generate_scen_csvs:
            write_methods.append("csv")
        # SpineDB results database (project-wide results.sqlite). Must run
        # on the native solve path — the writer needs the live s/par
        # namespaces — so it is requested here at solve time rather than
        # via the parquet-replay regen actions. The default target is
        # <output-location>/results.sqlite, i.e. the project root, shared
        # across scenarios (each appended as its own alternative).
        if settings.auto_generate_comp_spinedb:
            write_methods.append("spinedb")

        cmd = [
            sys.executable,
            SAFE_PATH_FLAG, "-m",
            "flextool.cli.cmd_run_flextool",
            job.input_db_url,
            "--scenario-name",
            job.scenario_name,
            "--work-folder",
            str(work_folder),
            "--output-location",
            str(self.project_path),
            "--output-subdir",
            job.output_subdir,
            "--write-methods",
            *write_methods,
            # Recorded in the run's _run_status.json.
            "--run-launcher",
            "gui",
        ]
        if job.is_xlsx_source and job.xlsx_path is not None:
            # The spreadsheet the input database is converted from (each batch): a reader of the run record
            # compares its time, not the database's, to tell whether the results are older than their input.
            cmd.extend(["--input-source-file", str(job.xlsx_path)])

        # Output config file
        single = settings.single_plot_settings
        if single.config_file:
            cmd.extend(["--output-config", single.config_file])

        # Active configs — pass all available if none explicitly set
        active = single.active_configs
        if not active:
            from flextool._resources import package_data_path
            from flextool.gui.config_parser import parse_plot_configs
            if single.config_file:
                config_path = Path(single.config_file)
                if not config_path.is_absolute():
                    config_path = Path.cwd() / config_path
            else:
                config_path = package_data_path("schemas/default_plots.yaml")
            active = parse_plot_configs(config_path) or ["default"]
        cmd.extend(["--active-configs", *active])

        # Plot rows (start_time .. start_time + duration - 1)
        if single.duration > 0:
            first_row = single.start_time
            last_row = single.start_time + single.duration - 1
            cmd.extend(["--plot-rows", str(first_row), str(last_row)])

        if single.only_first_file:
            cmd.append("--only-first-file-per-plot")
        if single.write_plot_data:
            cmd.append("--write-plot-data")

        # "Threads per run" (Execution jobs window): FlexTool's data processing and the solver of this run.
        # ``--threads`` gives HiGHS ``threads=N`` (parallel simplex off) and replaces a commercial solver's
        # thread count.
        cmd.extend(["--threads", str(max(1, int(self.execution_limits.max_cores_per_job)))])

        # The solver option folders, file by file: the project's ``solver_config`` (when it has one), then the
        # workspace's (``FLEXTOOL_SOLVER_CONFIG_DIR`` when set, else ``<workspace>/solver_config``); given
        # explicitly, one flag per folder, so a run never depends on where its input database is (an external
        # source database may sit in another tree).  Project export uses the same folders.  The folders the window
        # found go as --workspace-solver-config-dir, whose files are trusted only when this computer set up their
        # project or workspace; those of ``FLEXTOOL_SOLVER_CONFIG_DIR`` (the user's choice) as --solver-config-dir
        # (item 240 B3).
        from flextool._solver_config_source import gui_solver_config_flags
        cmd.extend(gui_solver_config_flags(self.project_path))

        # "Debug" radio group in the main window: Off / Basic / Full.
        # Basic → --debug=basic only (verbose checkpoints, no tracemalloc).
        # Full  → --debug=full + --csv-dump (tracemalloc + retained
        #         intermediate CSVs; the heavy I/O path).
        debug_level = getattr(settings, "debug_level", "off")
        if debug_level == "basic":
            cmd.append("--debug=basic")
        elif debug_level == "full":
            cmd.append("--debug=full")
            cmd.append("--csv-dump")

        # "Save memory" checkbox enables polar-high's save_memory path
        # (drop polar-side LP source + MPS round-trip pre-solve).
        if settings.save_memory:
            cmd.append("--save-memory")

        # ── Solver options (Solver options dialog in the side menu) ──
        # A flag is appended only for a setting the user changed, so a GUI
        # run and a terminal run of the same database solve alike.  The
        # thread count is sourced solely from the
        # execution_limits.max_cores_per_job append above (the Execution
        # jobs window owns that knob).  The MIP gap is not a GUI setting:
        # each solve's ``solver_mip_gap`` (else highs.opt) governs.
        _sll = getattr(settings, "solver_log_level", "normal")
        if _sll != "normal":
            cmd.append(f"--solver-log-level={_sll}")
        _stl = getattr(settings, "solver_time_limit", 0)
        if isinstance(_stl, int) and _stl > 0:
            cmd.extend(["--solver-time-limit", str(_stl)])
        # Scaling / presolve: ``None`` is the dialog's "Default" entry and
        # sends nothing (the solve's scaling, else full; presolve on).  Any
        # other value was picked by the user and is sent, including "full".
        _scl = getattr(settings, "scaling", None)
        if _scl is not None:
            cmd.append(f"--scaling={_scl}")
        _ps = getattr(settings, "presolve", None)
        if _ps is not None:
            cmd.append(f"--presolve={_ps}")

        return cmd

    # ------------------------------------------------------------------
    # Post-execution (comparison)
    # ------------------------------------------------------------------

    def _run_post_execution(self) -> None:
        """Run comparison plots/Excel after all scenarios finish.

        Accumulates successful scenario names across batches so that when
        the user adds batch 2 after batch 1 has already finished, the
        comparison includes all scenarios from both batches.
        """
        settings = self.settings
        do_comp_plots = settings.auto_generate_comp_plots
        do_comp_excel = settings.auto_generate_comp_excel

        # Gather on-disk subdirs from successful SCENARIO jobs only. Each
        # job records its chosen subdir (bare when uncontested, suffixed
        # on collision), so same-named scenarios from different sources
        # stay distinct.  A folder counts by its latest finished job: when
        # that one failed or was killed, the folder's results are not this
        # run's (its run record says so), so it leaves the comparison, also
        # when an earlier batch had put it there.
        with self._lock:
            latest: dict[str, ExecutionJob] = {}
            for j in self._jobs:
                if j.job_type != JobType.SCENARIO or not j.output_subdir:
                    continue
                if j.status not in (JobStatus.SUCCESS, JobStatus.FAILED, JobStatus.KILLED):
                    continue
                prev = latest.get(j.output_subdir)
                if prev is None or (j.end_time or datetime.min) >= (prev.end_time or datetime.min):
                    latest[j.output_subdir] = j
            failed = {sub for sub, j in latest.items() if j.status is not JobStatus.SUCCESS}
            successful = [
                j.output_subdir
                for j in self._jobs
                if j.job_type == JobType.SCENARIO and j.status == JobStatus.SUCCESS
                and j.output_subdir not in failed
            ]
            self._comparison_scenarios = [s for s in self._comparison_scenarios if s not in failed]
            # Merge in any externally-specified comparison subdirs
            extra = [s for s in self._comparison_scenarios if s not in successful]
            all_scenarios = successful + extra
            # Accumulate across batches
            for subdir in successful:
                if subdir not in self._comparison_scenarios:
                    self._comparison_scenarios.append(subdir)

        if not do_comp_plots and not do_comp_excel:
            return

        # A job that ended well can still leave a folder whose latest results are not usable (a run that
        # succeeded without writing parquet keeps an earlier run's files): such folders are left out, as the
        # output actions' comparison does; they stay in _comparison_scenarios, so a later batch checks them again.
        # A folder without a run record (an older FlexTool) stays in.  Read outside the lock.
        from flextool.gui.scenario_lists import describe_unusable  # lazy: scenario_lists loads tkinter

        left_out: list[tuple[str, str]] = []
        usable: list[str] = []
        for subdir in all_scenarios:
            if (why := output_name_problem(subdir)) is not None:
                # Never passed to the command: it would not be one plain argument (flextool.safe_names).
                left_out.append((describe_name(subdir), f"its folder name {why}"))
                continue
            status = read_status_settled(self.project_path / "output_parquet" / subdir)
            if results_usable(status):
                usable.append(subdir)
            else:
                left_out.append((subdir, describe_unusable(status) or "its results are not usable"))
        all_scenarios = usable
        for subdir, reason in left_out:
            logger.info("Comparison leaves out %s: %s", subdir, reason)

        if len(all_scenarios) < 1:
            logger.info("No successful scenarios with usable results; skipping comparison")
            return

        cmd = [
            sys.executable,
            SAFE_PATH_FLAG, "-m",
            "flextool.cli.cmd_scenario_results",
            "--parquet-base-dir",
            str(self.project_path / "output_parquet"),
            "--alternatives",
            *all_scenarios,
            "--plot-dir",
            str(self.project_path / "output_plot_comparisons"),
        ]

        comp = settings.comparison_plot_settings

        # Self-heal stale settings still pointing at the removed
        # default_comparison_plots.yaml — comparison rendering now lives
        # in default_plots.yaml.
        cfg_file = comp.config_file
        if cfg_file and cfg_file.endswith("default_comparison_plots.yaml"):
            cfg_file = cfg_file.replace(
                "default_comparison_plots.yaml", "default_plots.yaml",
            )

        if cfg_file:
            cmd.extend(["--output-config-path", cfg_file])

        active = comp.active_configs
        if not active:
            from flextool._resources import package_data_path
            from flextool.gui.config_parser import parse_plot_configs
            if cfg_file:
                config_path = Path(cfg_file)
                if not config_path.is_absolute():
                    config_path = Path.cwd() / config_path
            else:
                config_path = package_data_path("schemas/default_plots.yaml")
            active = parse_plot_configs(config_path) or ["default"]
        cmd.extend(["--active-configs", *active])

        if comp.duration > 0:
            first_row = comp.start_time
            last_row = comp.start_time + comp.duration - 1
            cmd.extend(["--plot-rows", str(first_row), str(last_row)])

        if do_comp_plots:
            cmd.append("--dispatch-plots")

        if do_comp_excel:
            cmd.extend(["--write-to-xlsx", "--write-dispatch-xlsx"])

        if comp.only_first_file:
            cmd.append("--only-first-file-per-plot")
        if comp.write_plot_data:
            cmd.append("--write-plot-data")

        logger.info("Running post-execution comparison:\n%s", format_cmd_for_log(cmd))

        # Create an auxiliary job so comparison output is visible in the
        # execution window.
        job = self.add_auxiliary_job(
            JobType.OUTPUT_ACTION,
            "Auto-comparison",
            "output:auto_comparison",
        )
        cmd_str = format_cmd_for_log(cmd)
        for subdir, reason in left_out:
            self.append_stdout(job.job_id, f"Left out {subdir}: {reason}")
        self.append_stdout(job.job_id, cmd_str)
        self.append_stdout(job.job_id, "")

        flextool_root = Path.cwd()  # subprocess cwd — user workspace, formerly the repo root
        success = False

        try:
            env = child_env(PYTHONUNBUFFERED="1")
            estimate_gb, _src = self._compute_memory_budget_for_job(None)
            with self._lock:
                job.memory_cap_gb = estimate_gb
            wrapped_cmd, popen_extras, post_spawn = _wrap_for_memory_cap(
                cmd, estimate_gb
            )
            popen_kwargs = dict(
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(flextool_root),
                env=env,
            )
            popen_kwargs.update(popen_extras)
            proc = subprocess.Popen(wrapped_cmd, **popen_kwargs)
            cap_handle = post_spawn(proc) if post_spawn is not None else None
            with self._lock:
                job.process = proc
                job.memory_cap_handle = cap_handle
            assert proc.stdout is not None
            for line in proc.stdout:
                stripped = line.rstrip("\n")
                self.append_stdout(job.job_id, stripped)
            return_code = proc.wait()
            success = return_code == 0
            if not success:
                self.append_stdout(
                    job.job_id, f"Process exited with code {return_code}"
                )
        except Exception:
            logger.exception("Post-execution comparison failed")
            self.append_stdout(job.job_id, "Comparison failed with exception")
        finally:
            self.finish_job(job.job_id, success)
            # Record which scenarios were compared so the Show button works
            if do_comp_plots:
                settings.comp_plots_scenarios = list(all_scenarios)
            if do_comp_excel:
                settings.comp_excel_scenarios = list(all_scenarios)

    # ------------------------------------------------------------------
    # Control methods
    # ------------------------------------------------------------------

    def pause(self) -> None:
        """Let currently running jobs finish but don't start new ones."""
        with self._lock:
            self._wind_down = True

    def resume(self) -> None:
        """Resume starting new jobs after a pause.

        If the scheduler thread has exited (all jobs were done when paused),
        a new scheduler is started.
        """
        with self._lock:
            self._wind_down = False
        # Re-launch the scheduler if it has exited
        self.start()

    @property
    def is_paused(self) -> bool:
        """Return True if the manager is in wind-down / paused state."""
        with self._lock:
            return self._wind_down

    def _stop_job_locked(self, job: ExecutionJob, reason: str) -> threading.Thread | None:
        """Stop *job* on purpose; the caller holds ``self._lock``.

        A RUNNING job becomes KILLED with *reason* as its ``kill_reason`` (unless one is set already) before its
        process is killed, so its worker records the run as ``killed`` with that reason, never as a crash or an
        out-of-memory kill.  A job still being set up (no process yet) is KILLED too: its worker does not start
        the engine.  A PENDING job becomes KILLED.  Returns the thread that stops an auxiliary job's process
        tree when one was started (it waits, so it never runs under the lock).
        """
        if job.status is JobStatus.PENDING:
            job.status = JobStatus.KILLED
            return None
        if job.status is not JobStatus.RUNNING:
            return None
        job.status = JobStatus.KILLED
        job.kill_reason = job.kill_reason or reason
        job.end_time = datetime.now()
        job.finish_timestamp = job.end_time.strftime(TIMESTAMP_FORMAT)
        if job.process is None:
            return None
        if job.stop_file is not None:
            job.stop_started = True
            thread = threading.Thread(
                target=self._stop_aux_job, args=(job.process, job.stop_file, job.kill_reason),
                name=f"flextool-stop-job-{job.job_id}", daemon=True,
            )
            self._live_waits.add(thread)
            thread.start()
            return thread
        kill_process_tree(job.process)
        return None

    def _stop_aux_job(self, proc: subprocess.Popen, stop_file: Path, reason: str) -> None:
        """The stop thread of an auxiliary job (see :func:`stop_aux_tree`)."""
        try:
            stop_aux_tree(proc, stop_file, reason)
        except Exception:
            logger.exception("Could not stop an auxiliary job's processes")
            kill_process_tree(proc)
        finally:
            with self._lock:
                self._live_waits.discard(threading.current_thread())

    def kill_all(self, reason: str = KILL_REASON_WINDOW) -> None:
        """Kill all running jobs and cancel pending ones; the run records give *reason*."""
        with self._lock:
            self._stopped = True
            for job in self._jobs:
                self._stop_job_locked(job, reason)

    def cleanup(self, progress: Callable[[], None] | None = None) -> None:
        """Stop every job and wait (at most ``CLEANUP_WAIT_S``) for their run records. Called on exit.

        This method is registered with ``atexit`` so it runs when Python
        exits normally (``sys.exit``, end of script).  It is also safe to
        call explicitly from signal handlers or window-close callbacks --
        the method is idempotent.  Running jobs are recorded as killed ("FlexTool
        was closed").  The worker threads are daemon threads, so without the
        wait the process could end before they finish the records; jobs
        already taken off the list are waited for as well.  The wait is done
        without the lock (the workers need it to finish).  *progress*, when
        given, is called about every 0.1 s while waiting (the window keeps
        its event loop going with it).
        """
        if self._watchdog is not None:
            self._watchdog.stop()
        with self._lock:
            self._stopped = True
            for job in self._jobs:
                self._stop_job_locked(job, KILL_REASON_CLOSED)
            waits = list(self._live_waits)
        if not waits:
            return
        logger.info("Waiting up to %.0f s for %d job(s) to finish their run records.", CLEANUP_WAIT_S, len(waits))
        deadline = time.monotonic() + CLEANUP_WAIT_S
        pending = waits
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            step = min(0.1, remaining)
            first = pending[0]
            if isinstance(first, threading.Thread):
                first.join(step)
            else:
                first.wait(step)
            pending = [w for w in pending if not _wait_done(w)]
            if pending and progress is not None:
                try:
                    progress()
                except Exception:
                    logger.debug("cleanup progress callback failed", exc_info=True)
        if pending:
            logger.warning("%d job(s) did not finish their run records within %.0f s of closing.", len(pending),
                           CLEANUP_WAIT_S)

    def kill_job(self, job_id: int) -> None:
        """Kill a specific running job (recorded as "stopped from the FlexTool window")."""
        with self._lock:
            job = self._get_job(job_id)
            if job is not None and job.status is JobStatus.RUNNING:
                self._stop_job_locked(job, KILL_REASON_WINDOW)
        if job:
            self._notify_status_change(job)

    def _prune_old_jobs(self, finished_job: ExecutionJob) -> None:
        """Remove previous jobs for the same scenario after a job finishes.

        Must be called while holding ``self._lock``.

        Only applies to SCENARIO jobs.  Auxiliary jobs are pruned via
        ``action_key`` in :meth:`add_auxiliary_job`.

        Rules:
        - New job succeeded → remove all previous jobs for this scenario.
        - New job failed    → remove previous failed/killed jobs, keep successful ones.
        """
        if finished_job.job_type != JobType.SCENARIO:
            return
        to_remove = []
        for old in self._jobs:
            if old is finished_job:
                continue
            if old.job_type != JobType.SCENARIO:
                continue
            if (old.source_number, old.scenario_name) != (
                finished_job.source_number, finished_job.scenario_name
            ):
                continue
            if old.status in (JobStatus.PENDING, JobStatus.RUNNING):
                continue
            if finished_job.status == JobStatus.SUCCESS:
                to_remove.append(old)
            elif old.status in (JobStatus.FAILED, JobStatus.KILLED):
                to_remove.append(old)
        for old in to_remove:
            self._jobs.remove(old)

    def remove_job(self, job_id: int) -> None:
        """Remove a finished or pending job from the list."""
        with self._lock:
            job = self._get_job(job_id)
            if job and job.status in (
                JobStatus.PENDING,
                JobStatus.SUCCESS,
                JobStatus.FAILED,
                JobStatus.KILLED,
            ):
                self._jobs.remove(job)

    def move_pending_up(self, job_id: int) -> None:
        """Move a pending job one position earlier in the queue."""
        with self._lock:
            job = self._get_job(job_id)
            if job is None or job.status != JobStatus.PENDING:
                return
            idx = self._jobs.index(job)
            if idx <= 0:
                return
            # Swap with the previous item (only if it is also pending)
            prev = self._jobs[idx - 1]
            if prev.status == JobStatus.PENDING:
                self._jobs[idx - 1], self._jobs[idx] = self._jobs[idx], self._jobs[idx - 1]

    def move_pending_down(self, job_id: int) -> None:
        """Move a pending job one position later in the queue."""
        with self._lock:
            job = self._get_job(job_id)
            if job is None or job.status != JobStatus.PENDING:
                return
            idx = self._jobs.index(job)
            if idx >= len(self._jobs) - 1:
                return
            # Swap with the next item (only if it is also pending)
            nxt = self._jobs[idx + 1]
            if nxt.status == JobStatus.PENDING:
                self._jobs[idx], self._jobs[idx + 1] = self._jobs[idx + 1], self._jobs[idx]

    def set_force_start(self, job_id: int, force: bool = True) -> None:
        """Flag (or unflag) a pending scenario job to bypass the memory check.

        Only affects ``PENDING`` ``SCENARIO`` jobs; finished or running jobs
        are ignored. A forced job is admitted by the scheduler ahead of the
        memory-reserve check, but still respects the thread-slot limit and
        the watchdog.
        """
        with self._lock:
            job = self._get_job(job_id)
            if (
                job is not None
                and job.job_type == JobType.SCENARIO
                and job.status == JobStatus.PENDING
            ):
                job.force_start = bool(force)

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_jobs(self) -> list[ExecutionJob]:
        """Return a copy of the job list."""
        with self._lock:
            return list(self._jobs)

    def get_stdout(self, job_id: int) -> list[str]:
        """Get stdout lines for a specific job."""
        with self._lock:
            job = self._get_job(job_id)
            if job is None:
                return []
            return list(job.stdout_lines)

    def has_pending_or_running(self) -> bool:
        """Check if there are any pending or running jobs (any type)."""
        with self._lock:
            return any(
                j.status in (JobStatus.PENDING, JobStatus.RUNNING) for j in self._jobs
            )

    def get_execution_status(self) -> dict:
        """Return a snapshot of the dispatcher's state for the GUI status bar.

        Keys:
            running          int  — number of currently running scenario jobs
            max_threads      int  — current ``max_workers`` setting
            pending          int  — number of queued scenario jobs
            used_gb          float — system RAM in use (total - available)
            total_gb         float — system RAM total
            thread_limited   bool  — pending jobs are being held by max_workers
            memory_limited   bool  — pending jobs are being held by free RAM
                                     (live ``vm.available`` admission check)
            memory_guard     bool  — whether the memory guard is active this session
        """
        with self._lock:
            running = self._running_count
            max_threads = self._max_workers
            pending = sum(
                1 for j in self._jobs
                if j.job_type == JobType.SCENARIO and j.status == JobStatus.PENDING
            )
            thread_limited = self._thread_limited
            memory_limited = self._memory_limited
            memory_guard = self._memory_guard_enabled
        try:
            vm = psutil.virtual_memory()
            total_gb = vm.total / (1024 ** 3)
            used_gb = (vm.total - vm.available) / (1024 ** 3)
        except Exception:
            total_gb = 0.0
            used_gb = 0.0
        return {
            "running": running,
            "max_threads": max_threads,
            "pending": pending,
            "used_gb": used_gb,
            "total_gb": total_gb,
            "thread_limited": thread_limited,
            "memory_limited": memory_limited,
            "memory_guard": memory_guard,
        }

    def has_pending_or_running_scenarios(self) -> bool:
        """Check if there are pending or running scenario jobs specifically."""
        with self._lock:
            return any(
                j.status in (JobStatus.PENDING, JobStatus.RUNNING)
                for j in self._jobs
                if j.job_type == JobType.SCENARIO
            )

    def set_comparison_scenarios(self, scenario_names: list[str]) -> None:
        """Set the list of executed scenarios to include in comparison."""
        with self._lock:
            self._comparison_scenarios = list(scenario_names)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_job(self, job_id: int) -> ExecutionJob | None:
        """Find job by ID. Must be called with lock held."""
        for job in self._jobs:
            if job.job_id == job_id:
                return job
        return None

    def _notify_status_change(self, job: ExecutionJob) -> None:
        """Invoke the on_status_change callback (if set)."""
        if self._on_status_change is not None:
            try:
                self._on_status_change(job)
            except Exception:
                logger.exception(
                    "on_status_change callback failed for job %d", job.job_id
                )
