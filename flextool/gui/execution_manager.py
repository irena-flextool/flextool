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

from flextool.gui.cli_format import format_cmd_for_log
from flextool.gui.data_models import GlobalSettings, ProjectSettings, ScenarioInfo
from flextool.gui.scenario_key import choose_output_subdir_for_write

logger = logging.getLogger(__name__)


_slice_probe_cache: dict[str, bool] = {}


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

    def _loop(self) -> None:
        while not self._stop.wait(self.POLL_INTERVAL_S):
            with self._manager._lock:
                snapshot = [
                    (j, j.process, j.memory_cap_gb)
                    for j in self._manager._jobs
                    if j.status == JobStatus.RUNNING and j.process is not None
                ]

            measured: list[tuple[ExecutionJob, int, int]] = []
            for job, proc, est_gb in snapshot:
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
                est_bytes = int(est_gb * (1024 ** 3))
                measured.append((job, rss, est_bytes))

            if not measured:
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
            try:
                vproc = victim.process
                if vproc is not None:
                    vproc.kill()
            except OSError:
                pass


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
    ENV_REPAIR = auto()     # native-library compatibility fix (polars-lts-cpu swap)


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
    memory_cap_gb: float = 0.0       # snapshot of cap at dispatch (0 = uncapped)
    peak_rss_mb: float = 0.0         # high-water mark, fed by MemoryWatchdog
    killed_for_memory: bool = False  # set by watchdog just before SIGTERM
    kill_reason: str = ""  # populated by watchdog: "exceeded estimate" / "global memory pressure" / "global swap pressure"
    native_fault: bool = False  # set on finalize when the process died from a native crash (segfault / illegal instruction)
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
        self._wind_down = False
        self._stopped = False
        self._memory_limited: bool = False  # set by scheduler when admission blocked by RAM
        self._thread_limited: bool = False  # set by scheduler when running == max_workers and pending exist
        self.project_path = project_path
        self.settings = settings
        self._global_settings = global_settings
        self._on_status_change = on_status_change
        self._on_all_finished = on_all_finished
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

    def _compute_memory_budget_for_job(self, job: ExecutionJob | None = None) -> float:
        """Return per-job memory budget in GB. 0 means no budget set.

        Resolution order:
          1. User-set value (``ExecutionLimits.memory_cap_per_job_gb``) when > 0.
          2. Learned peak from ``ProjectSettings.scenario_resource_history``,
             scaled by 1.5 for safety, when this job has a prior successful run.
          3. Auto fallback: ``(system_total - reserve) / max_workers``.
        """
        limits = self.execution_limits
        if limits.memory_cap_per_job_gb > 0:
            return limits.memory_cap_per_job_gb
        if job is not None and job.output_subdir:
            history = self.settings.scenario_resource_history.get(job.output_subdir)
            if history is not None and history.peak_rss_mb > 0:
                return (history.peak_rss_mb / 1024.0) * 1.5
        try:
            total_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            return 0.0
        available = max(0.0, total_gb - limits.system_reserve_gb)
        workers = max(1, self._max_workers)
        return available / workers

    def _can_admit_memory(self, next_estimate_gb: float) -> bool:
        """Return True if dispatching another job won't breach the memory reserve.

        Sums the budgets (``memory_cap_gb``) of currently-running scenario jobs and
        checks whether ``running_budget + next_estimate <= total - reserve``.

        Must be called while holding ``self._lock``.
        """
        if next_estimate_gb <= 0:
            return True  # no estimate available; let the watchdog handle it
        try:
            total_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            return True
        limits = self.execution_limits
        headroom = max(0.0, total_gb - limits.system_reserve_gb)
        running_budget = sum(
            j.memory_cap_gb for j in self._jobs
            if j.status == JobStatus.RUNNING and j.memory_cap_gb > 0
        )
        return running_budget + next_estimate_gb <= headroom

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
                is_xlsx = scenario.source_name.lower().endswith(
                    (".xlsx", ".xls", ".ods")
                )
                source_path = self._resolve_source_path(scenario.source_name)
                if is_xlsx:
                    xlsx_path = source_path
                    stem = Path(scenario.source_name).stem
                    input_db_url = (
                        "sqlite:///"
                        + str(self.project_path / "intermediate" / f"{stem}.sqlite")
                    )
                else:
                    xlsx_path = None
                    input_db_url = "sqlite:///" + str(source_path)

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
                    if old.status == JobStatus.RUNNING and old.process:
                        try:
                            old.process.kill()
                        except OSError:
                            pass
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
        """Thread-safe append of a stdout line to a job."""
        with self._lock:
            job = self._get_job(job_id)
            if job is not None:
                job.stdout_lines.append(line)

    def finish_job(self, job_id: int, success: bool) -> None:
        """Mark an auxiliary job as finished (SUCCESS or FAILED)."""
        with self._lock:
            job = self._get_job(job_id)
            if job is None:
                return
            now = datetime.now()
            job.end_time = now
            job.finish_timestamp = now.strftime("%d.%m.%y %H:%M")
            job.process = None
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

        Running or pending jobs are killed first.
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
                    if job.status == JobStatus.RUNNING and job.process:
                        try:
                            job.process.kill()
                        except OSError:
                            pass
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
        db_url = "sqlite:///" + str(self.project_path / "results.sqlite")
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

                # Find the first pending scenario job (if any)
                candidate: ExecutionJob | None = None
                for job in self._jobs:
                    if job.job_type == JobType.SCENARIO and job.status == JobStatus.PENDING:
                        candidate = job
                        break

                # Reset both flags; we'll set whichever applies below
                self._thread_limited = False
                self._memory_limited = False

                next_job: ExecutionJob | None = None
                if not self._wind_down and candidate is not None:
                    if self._running_count >= self._max_workers:
                        self._thread_limited = True
                    else:
                        estimate_gb = self._compute_memory_budget_for_job(candidate)
                        if self._can_admit_memory(estimate_gb):
                            next_job = candidate
                            self._running_count += 1
                        else:
                            self._memory_limited = True

                # Check if all SCENARIO jobs are done (auxiliary jobs are
                # managed externally and don't block the scheduler)
                any_active = any(
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
        """Run a single job in a worker thread."""
        try:
            # Mark as RUNNING
            with self._lock:
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

            # Clean work folder so re-executions start fresh
            if work_folder.exists():
                shutil.rmtree(work_folder, ignore_errors=True)
            work_folder.mkdir(parents=True, exist_ok=True)

            cmd = self._build_run_command(job, work_folder)

            estimate_gb = self._compute_memory_budget_for_job(job)
            with self._lock:
                job.memory_cap_gb = estimate_gb

            wrapped_cmd, popen_extras, post_spawn = _wrap_for_memory_cap(
                cmd, estimate_gb
            )
            cmd_str = format_cmd_for_log(wrapped_cmd)
            logger.info("Running job %d (%s):\n%s", job.job_id, job.scenario_name, cmd_str)

            # Log the CLI command as the first line in the progress output
            with self._lock:
                job.stdout_lines.append(cmd_str)
                job.stdout_lines.append("")
            self._notify_status_change(job)

            # Run from flextool root so relative paths (templates/, etc.) work
            flextool_root = Path.cwd()  # subprocess cwd — user workspace, formerly the repo root

            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            popen_kwargs = dict(
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(flextool_root), env=env,
            )
            popen_kwargs.update(popen_extras)
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
                proc.kill()
                proc.wait()
                return

            # Read stdout line by line
            assert proc.stdout is not None
            for line in proc.stdout:
                stripped = line.rstrip("\n")
                with self._lock:
                    job.stdout_lines.append(stripped)
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
                job.finish_timestamp = now.strftime("%d.%m.%y %H:%M")
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
                            "not a model error. It almost always means an installed "
                            "solver library (polars or HiGHS/highspy) does not run "
                            "on this computer. FlexTool can re-install a compatible "
                            "build — see the prompt at the next start, or run "
                            "'Update FlexTool'."
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
                job.status = JobStatus.FAILED
                job.end_time = datetime.now()
                job.finish_timestamp = job.end_time.strftime("%d.%m.%y %H:%M")
            self._notify_status_change(job)
        finally:
            with self._lock:
                self._running_count -= 1

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
            "-m",
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
        ]

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

        cmd.extend(["--highs-threads", str(self.execution_limits.max_cores_per_job)])

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
        # Append each flag only when the user has changed it from the
        # GUI-side default, keeping the engine command line clean on
        # the common path.  HiGHS thread count is sourced solely from
        # the execution_limits.max_cores_per_job append above (the
        # Execution jobs window owns that knob); ``user_bound_scale``
        # is advanced-user territory routed through ``solver_arguments``.
        _sll = getattr(settings, "solver_log_level", "normal")
        if _sll != "normal":
            cmd.append(f"--solver-log-level={_sll}")
        _stl = getattr(settings, "solver_time_limit", 0)
        if isinstance(_stl, int) and _stl > 0:
            cmd.extend(["--solver-time-limit", str(_stl)])
        _mff = getattr(settings, "matrix_file_format", "mps")
        if _mff != "mps":
            cmd.append(f"--matrix-file-format={_mff}")
        _scl = getattr(settings, "scaling", "full")
        if _scl != "full":
            cmd.append(f"--scaling={_scl}")
        _ps = getattr(settings, "presolve", "choose")
        if _ps != "choose":
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
        # stay distinct.
        with self._lock:
            successful = [
                j.output_subdir
                for j in self._jobs
                if j.job_type == JobType.SCENARIO and j.status == JobStatus.SUCCESS
            ]
            # Merge in any externally-specified comparison subdirs
            extra = [s for s in self._comparison_scenarios if s not in successful]
            all_scenarios = successful + extra
            # Accumulate across batches
            for subdir in successful:
                if subdir not in self._comparison_scenarios:
                    self._comparison_scenarios.append(subdir)

        if not do_comp_plots and not do_comp_excel:
            return

        if len(all_scenarios) < 1:
            logger.info("No successful scenarios; skipping comparison")
            return

        cmd = [
            sys.executable,
            "-m",
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

        logger.info("Running post-execution comparison:\n%s", format_cmd_for_log(cmd))

        # Create an auxiliary job so comparison output is visible in the
        # execution window.
        job = self.add_auxiliary_job(
            JobType.OUTPUT_ACTION,
            "Auto-comparison",
            "output:auto_comparison",
        )
        cmd_str = format_cmd_for_log(cmd)
        self.append_stdout(job.job_id, cmd_str)
        self.append_stdout(job.job_id, "")

        flextool_root = Path.cwd()  # subprocess cwd — user workspace, formerly the repo root
        success = False

        try:
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            estimate_gb = self._compute_memory_budget_for_job(None)
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

    def kill_all(self) -> None:
        """Kill all running jobs and cancel pending ones."""
        with self._lock:
            self._stopped = True
            for job in self._jobs:
                if job.status == JobStatus.RUNNING and job.process:
                    try:
                        job.process.kill()
                    except OSError:
                        pass
                    job.status = JobStatus.KILLED
                elif job.status == JobStatus.PENDING:
                    job.status = JobStatus.KILLED

    def cleanup(self) -> None:
        """Kill all running subprocesses. Called on exit.

        This method is registered with ``atexit`` so it runs when Python
        exits normally (``sys.exit``, end of script).  It is also safe to
        call explicitly from signal handlers or window-close callbacks --
        the method is idempotent.
        """
        if self._watchdog is not None:
            self._watchdog.stop()
        with self._lock:
            self._stopped = True
            for job in self._jobs:
                if job.status == JobStatus.RUNNING and job.process:
                    try:
                        job.process.kill()
                        job.process.wait(timeout=5)
                    except Exception:
                        pass

    def kill_job(self, job_id: int) -> None:
        """Kill a specific running job."""
        with self._lock:
            job = self._get_job(job_id)
            if job and job.status == JobStatus.RUNNING and job.process:
                try:
                    job.process.kill()
                except OSError:
                    pass
                job.status = JobStatus.KILLED
                job.end_time = datetime.now()
                job.finish_timestamp = job.end_time.strftime("%d.%m.%y %H:%M")
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
            memory_limited   bool  — pending jobs are being held by memory reserve
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
