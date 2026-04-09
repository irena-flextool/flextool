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

from flextool.gui.cli_format import format_cmd_for_log
from flextool.gui.data_models import ProjectSettings, ScenarioInfo

logger = logging.getLogger(__name__)


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


class ExecutionManager:
    """Thread-safe manager for parallel FlexTool scenario execution."""

    def __init__(
        self,
        project_path: Path,
        settings: ProjectSettings,
        on_status_change: Callable[[ExecutionJob], None] | None = None,
        on_all_finished: Callable[[], None] | None = None,
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
        """
        self._lock = threading.Lock()
        self._jobs: list[ExecutionJob] = []
        self._next_id = 0
        self._max_workers = max(1, (os.cpu_count() or 2) - 1)
        self._running_count = 0
        self._wind_down = False
        self._stopped = False
        self.project_path = project_path
        self.settings = settings
        self._on_status_change = on_status_change
        self._on_all_finished = on_all_finished
        self._scheduler_thread: threading.Thread | None = None
        self._comparison_scenarios: list[str] = []
        self._pending_select_job_id: int | None = None  # for auto-selecting new jobs in UI

        self._converted_xlsx: set[str] = set()  # source_names already converted

        # Register cleanup for when the Python process exits
        atexit.register(self.cleanup)

    # ------------------------------------------------------------------
    # max_workers property
    # ------------------------------------------------------------------

    @property
    def max_workers(self) -> int:
        return self._max_workers

    @max_workers.setter
    def max_workers(self, value: int) -> None:
        self._max_workers = max(1, value)

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------

    def add_jobs(self, scenarios: list[ScenarioInfo]) -> list[ExecutionJob]:
        """Add scenarios to the execution queue. Returns the created jobs."""
        new_jobs: list[ExecutionJob] = []
        with self._lock:
            for scenario in scenarios:
                is_xlsx = scenario.source_name.lower().endswith(
                    (".xlsx", ".xls", ".ods")
                )
                if is_xlsx:
                    xlsx_path = (
                        self.project_path / "input_sources" / scenario.source_name
                    )
                    # The converted sqlite lives in intermediate/
                    stem = Path(scenario.source_name).stem
                    input_db_url = (
                        "sqlite:///"
                        + str(self.project_path / "intermediate" / f"{stem}.sqlite")
                    )
                else:
                    xlsx_path = None
                    input_db_url = (
                        "sqlite:///"
                        + str(
                            self.project_path
                            / "input_sources"
                            / scenario.source_name
                        )
                    )

                job = ExecutionJob(
                    job_id=self._next_id,
                    scenario_name=scenario.name,
                    source_name=scenario.source_name,
                    source_number=scenario.source_number,
                    input_db_url=input_db_url,
                    is_xlsx_source=is_xlsx,
                    xlsx_path=xlsx_path,
                )
                self._next_id += 1
                self._jobs.append(job)
                new_jobs.append(job)
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

        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop, daemon=True
        )
        self._scheduler_thread.start()

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

                # Find the next pending SCENARIO job if capacity is available
                next_job: ExecutionJob | None = None
                if not self._wind_down and self._running_count < self._max_workers:
                    for job in self._jobs:
                        if job.job_type == JobType.SCENARIO and job.status == JobStatus.PENDING:
                            next_job = job
                            break

                if next_job is not None:
                    self._running_count += 1

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
            work_folder = self.project_path / "work" / job.scenario_name

            # Clean work folder so re-executions start fresh
            if work_folder.exists():
                shutil.rmtree(work_folder, ignore_errors=True)
            work_folder.mkdir(parents=True, exist_ok=True)

            cmd = self._build_run_command(job, work_folder)
            cmd_str = format_cmd_for_log(cmd)
            logger.info("Running job %d (%s):\n%s", job.job_id, job.scenario_name, cmd_str)

            # Log the CLI command as the first line in the progress output
            with self._lock:
                job.stdout_lines.append(cmd_str)
                job.stdout_lines.append("")
            self._notify_status_change(job)

            # Run from flextool root so relative paths (templates/, etc.) work
            flextool_root = Path(__file__).resolve().parent.parent.parent

            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(flextool_root),
                env=env,
            )
            killed = False
            with self._lock:
                # Check if killed while we were setting up
                if job.status == JobStatus.KILLED:
                    killed = True
                else:
                    job.process = proc

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
            with self._lock:
                if job.status == JobStatus.KILLED:
                    return
                job.end_time = now
                job.finish_timestamp = now.strftime("%d.%m.%y %H:%M")
                job.process = None
                if return_code == 0:
                    job.status = JobStatus.SUCCESS
                    self.settings.scenarios_changed = True
                else:
                    job.status = JobStatus.FAILED
                    job.stdout_lines.append(
                        f"[execution_manager] Process exited with code {return_code}"
                    )
                self._prune_old_jobs(job)
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
            from flextool.gui.config_parser import parse_plot_configs
            config_file = single.config_file or "templates/default_plots.yaml"
            config_path = Path(config_file)
            if not config_path.is_absolute():
                config_path = Path(__file__).resolve().parent.parent.parent / config_file
            active = parse_plot_configs(config_path) or ["default"]
        cmd.extend(["--active-configs", *active])

        # Plot rows (start_time .. start_time + duration - 1)
        if single.duration > 0:
            first_row = single.start_time
            last_row = single.start_time + single.duration - 1
            cmd.extend(["--plot-rows", str(first_row), str(last_row)])

        if single.only_first_file:
            cmd.append("--only-first-file-per-plot")

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

        # Gather scenario names from successful SCENARIO jobs only
        with self._lock:
            successful = [
                j.scenario_name
                for j in self._jobs
                if j.job_type == JobType.SCENARIO and j.status == JobStatus.SUCCESS
            ]
            # Merge in any externally-specified comparison scenarios
            extra = [s for s in self._comparison_scenarios if s not in successful]
            all_scenarios = successful + extra
            # Accumulate: remember these scenarios so that if a new batch
            # is added later, the next comparison will include them too.
            for name in successful:
                if name not in self._comparison_scenarios:
                    self._comparison_scenarios.append(name)

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

        if comp.config_file:
            cmd.extend(["--output-config-path", comp.config_file])

        active = comp.active_configs
        if not active:
            from flextool.gui.config_parser import parse_plot_configs
            config_file = comp.config_file or "templates/default_comparison_plots.yaml"
            config_path = Path(config_file)
            if not config_path.is_absolute():
                config_path = Path(__file__).resolve().parent.parent.parent / config_file
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

        flextool_root = Path(__file__).resolve().parent.parent.parent
        success = False

        try:
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(flextool_root),
                env=env,
            )
            with self._lock:
                job.process = proc
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
            if old.scenario_name != finished_job.scenario_name:
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
