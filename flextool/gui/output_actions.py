"""Manages output generation actions (plots, Excel, CSV) as background operations."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, TYPE_CHECKING

from flextool.gui.cli_format import format_cmd_for_log
from flextool.gui.config_parser import parse_plot_configs
from flextool.gui.data_models import PlotSettings, ProjectSettings
from flextool.gui.platform_utils import open_file_in_default_app
from flextool.gui.project_utils import get_projects_dir

if TYPE_CHECKING:
    from flextool.gui.execution_manager import ExecutionManager

logger = logging.getLogger(__name__)


class OutputActionManager:
    """Manages output generation actions as background operations.

    Each action creates an auxiliary job in the :class:`ExecutionManager` so
    its output is visible in the ExecutionWindow.  When the command finishes
    the optional *on_complete* callback is invoked **from the worker thread**
    -- the caller should use ``root.after()`` to safely update tkinter.
    """

    def __init__(
        self,
        project_path: Path,
        settings: ProjectSettings,
        execution_mgr: ExecutionManager | None = None,
        on_complete: Callable[[str, bool], None] | None = None,
    ) -> None:
        self.project_path = project_path
        self.settings = settings
        self._execution_mgr = execution_mgr
        self._on_complete = on_complete
        self._running_actions: set[str] = set()
        self._action_gen: dict[str, int] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public query
    # ------------------------------------------------------------------

    def is_running(self, action_name: str) -> bool:
        """Return *True* if the given action is currently running."""
        with self._lock:
            return action_name in self._running_actions

    # ------------------------------------------------------------------
    # Scenario-level actions
    # ------------------------------------------------------------------

    def run_scenario_plots(self, scenario_names: list[str]) -> None:
        action = "scen_plots"
        gen = self._mark_running(action)
        job = self._create_aux_job(action, "Re-plot scenarios")

        def _work() -> bool:
            ok = True
            for name in scenario_names:
                cmd = self._build_write_outputs_cmd(name, ["plot"])
                if not self._run_subprocess(cmd, job):
                    ok = False
            return ok

        self._start_thread(action, gen, _work, job)

    def run_scenario_excel(self, scenario_names: list[str]) -> None:
        action = "scen_excel"
        gen = self._mark_running(action)
        job = self._create_aux_job(action, "Scenarios to Excel")

        def _work() -> bool:
            ok = True
            for name in scenario_names:
                cmd = self._build_write_outputs_cmd(name, ["excel"])
                if not self._run_subprocess(cmd, job):
                    ok = False
            return ok

        self._start_thread(action, gen, _work, job)

    def run_scenario_csvs(self, scenario_names: list[str]) -> None:
        action = "scen_csvs"
        gen = self._mark_running(action)
        job = self._create_aux_job(action, "Scenarios to CSVs")

        def _work() -> bool:
            ok = True
            for name in scenario_names:
                cmd = self._build_write_outputs_cmd(name, ["csv"])
                if not self._run_subprocess(cmd, job):
                    ok = False
            return ok

        self._start_thread(action, gen, _work, job)

    # ------------------------------------------------------------------
    # Comparison actions
    # ------------------------------------------------------------------

    def run_comparison_plots(self, scenario_names: list[str]) -> None:
        action = "comp_plots"
        gen = self._mark_running(action)
        job = self._create_aux_job(action, "Comparison plots")

        def _work() -> bool:
            cmd = self._build_comparison_cmd(scenario_names, plots=True, excel=False)
            ok = self._run_subprocess(cmd, job)
            if ok:
                comp_dir = self.project_path / "output_plot_comparisons"
                first_png = self.find_first_plot(comp_dir)
                if first_png is not None:
                    try:
                        open_file_in_default_app(first_png)
                    except OSError:
                        logger.warning("Could not open comparison plot: %s", first_png)
            return ok

        self._start_thread(action, gen, _work, job)

    def run_comparison_excel(self, scenario_names: list[str]) -> None:
        action = "comp_excel"
        gen = self._mark_running(action)
        job = self._create_aux_job(action, "Comparison to Excel")

        def _work() -> bool:
            cmd = self._build_comparison_cmd(scenario_names, plots=False, excel=True)
            return self._run_subprocess(cmd, job)

        self._start_thread(action, gen, _work, job)

    # ------------------------------------------------------------------
    # File finders (used by Show/Open buttons)
    # ------------------------------------------------------------------

    def find_first_plot(self, directory: Path) -> Path | None:
        if not directory.is_dir():
            return None
        pngs = sorted(directory.rglob("*.png"))
        return pngs[0] if pngs else None

    def find_scenario_excel(self, scenario_name: str) -> Path | None:
        excel_dir = self.project_path / "output_excel"
        if not excel_dir.is_dir():
            return None
        for f in sorted(excel_dir.iterdir()):
            if f.suffix.lower() == ".xlsx" and scenario_name in f.stem:
                return f
        return None

    def find_comparison_excel(self) -> Path | None:
        if not self.project_path.is_dir():
            return None
        xlsxs = sorted(
            f for f in self.project_path.iterdir()
            if f.suffix.lower() == ".xlsx" and f.stem.startswith("compare_")
        )
        return xlsxs[0] if xlsxs else None

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_active_configs(ps: PlotSettings, default_config: str) -> list[str]:
        if ps.active_configs:
            return ps.active_configs
        config_file = ps.config_file or default_config
        config_path = Path(config_file)
        if not config_path.is_absolute():
            config_path = get_projects_dir().parent / config_file
        return parse_plot_configs(config_path) or ["default"]

    # ------------------------------------------------------------------
    # Command builders
    # ------------------------------------------------------------------

    def _build_write_outputs_cmd(
        self, scenario_name: str, write_methods: list[str]
    ) -> list[str]:
        settings = self.settings
        single = settings.single_plot_settings
        parquet_dir = self.project_path / "output_parquet" / scenario_name

        cmd: list[str] = [
            sys.executable, "-m", "flextool.cli.cmd_write_outputs",
            "--scenario-name", scenario_name,
            "--read-parquet-dir", str(parquet_dir),
            "--output-location", str(self.project_path),
            "--write-methods", *write_methods,
        ]

        if single.config_file:
            cmd.extend(["--config-path", single.config_file])

        active = self._resolve_active_configs(single, "templates/default_plots.yaml")
        cmd.extend(["--active-configs", *active])

        if single.duration > 0:
            first_row = single.start_time
            last_row = single.start_time + single.duration - 1
            cmd.extend(["--plot-rows", str(first_row), str(last_row)])

        if single.only_first_file:
            cmd.append("--only-first-file-per-plot")

        return cmd

    def _build_comparison_cmd(
        self,
        scenario_names: list[str],
        *,
        plots: bool,
        excel: bool,
    ) -> list[str]:
        settings = self.settings
        comp = settings.comparison_plot_settings
        parquet_base = self.project_path / "output_parquet"
        plot_dir = self.project_path / "output_plot_comparisons"

        cmd: list[str] = [
            sys.executable, "-m", "flextool.cli.cmd_scenario_results",
            "--parquet-base-dir", str(parquet_base),
            "--alternatives", *scenario_names,
            "--plot-dir", str(plot_dir),
        ]

        if comp.config_file:
            cmd.extend(["--output-config-path", comp.config_file])

        active = self._resolve_active_configs(comp, "templates/default_comparison_plots.yaml")
        cmd.extend(["--active-configs", *active])

        if comp.duration > 0:
            first_row = comp.start_time
            last_row = comp.start_time + comp.duration - 1
            cmd.extend(["--plot-rows", str(first_row), str(last_row)])

        if plots and comp.dispatch_plots:
            cmd.append("--dispatch-plots")

        if excel:
            cmd.extend(["--write-to-xlsx", "--write-dispatch-xlsx"])
            cmd.extend(["--excel-dir", str(self.project_path)])

        if comp.only_first_file:
            cmd.append("--only-first-file-per-plot")

        return cmd

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mark_running(self, action: str) -> int:
        with self._lock:
            self._running_actions.add(action)
            gen = self._action_gen.get(action, 0) + 1
            self._action_gen[action] = gen
            return gen

    def _mark_finished(self, action: str) -> None:
        with self._lock:
            self._running_actions.discard(action)

    def _create_aux_job(self, action: str, title: str):
        """Create an auxiliary job in the ExecutionManager (if available)."""
        if self._execution_mgr is None:
            return None
        from flextool.gui.execution_manager import JobType
        return self._execution_mgr.add_auxiliary_job(
            JobType.OUTPUT_ACTION,
            title,
            f"output:{action}",
        )

    def _start_thread(self, action: str, gen: int, work, job=None) -> None:
        def _wrapper() -> None:
            try:
                success = work()
            except Exception:
                logger.exception("Output action %s failed with exception", action)
                success = False
            finally:
                self._mark_finished(action)

            with self._lock:
                superseded = self._action_gen.get(action, 0) != gen

            if job is not None and not superseded:
                self._execution_mgr.finish_job(job.job_id, success)

            if self._on_complete is not None and not superseded:
                try:
                    self._on_complete(action, success)
                except Exception:
                    logger.exception("on_complete callback failed for %s", action)

        thread = threading.Thread(target=_wrapper, daemon=True)
        thread.start()

    def _run_subprocess(self, cmd: list[str], job=None) -> bool:
        """Run a subprocess to completion. Returns True on success."""
        flextool_root = get_projects_dir().parent
        cmd_str = format_cmd_for_log(cmd)

        if job is not None:
            self._execution_mgr.append_stdout(job.job_id, cmd_str)
            self._execution_mgr.append_stdout(job.job_id, "")

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

            if job is not None:
                with self._execution_mgr._lock:
                    job.process = proc

            assert proc.stdout is not None
            for line in proc.stdout:
                stripped = line.rstrip("\n")
                if job is not None:
                    self._execution_mgr.append_stdout(job.job_id, stripped)

            return_code = proc.wait()
            if return_code != 0:
                msg = f"Process exited with code {return_code}"
                logger.error("Output action %s", msg)
                if job is not None:
                    self._execution_mgr.append_stdout(job.job_id, f"\n{msg}")
                return False
            return True
        except FileNotFoundError:
            msg = f"Could not find executable: {cmd[0]}"
            logger.error(msg)
            if job is not None:
                self._execution_mgr.append_stdout(job.job_id, msg)
            return False
        except Exception:
            logger.exception("Subprocess failed: %s", cmd_str)
            return False
