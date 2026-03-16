"""Manages output generation actions (plots, Excel, CSV) as background operations."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

from flextool.gui.data_models import ProjectSettings
from flextool.gui.platform_utils import open_file_in_default_app

logger = logging.getLogger(__name__)


class OutputActionManager:
    """Manages output generation actions as background operations.

    Each action runs a CLI command in a background thread.  When the command
    finishes the optional *on_complete* callback is invoked **from the worker
    thread** -- the caller should use ``root.after()`` to safely update tkinter.
    """

    def __init__(
        self,
        project_path: Path,
        settings: ProjectSettings,
        on_complete: Callable[[str, bool], None] | None = None,
    ) -> None:
        """
        Args:
            project_path: Path to the project directory.
            settings: Project settings (plot settings, config paths, etc.).
            on_complete: ``callback(action_name, success)`` called when an
                action finishes.  Called from the worker thread.
        """
        self.project_path = project_path
        self.settings = settings
        self._on_complete = on_complete
        self._running_actions: set[str] = set()
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
        """Generate plots for selected executed scenarios.

        Runs ``cmd_write_outputs`` for each scenario with ``--write-methods plot``.
        Uses ``--read-parquet-dir`` to read from existing parquet files.
        Opens the first plot automatically on completion.
        """
        action = "scen_plots"
        if not self._mark_running(action):
            return

        def _work() -> bool:
            ok = True
            for name in scenario_names:
                cmd = self._build_write_outputs_cmd(name, ["plot"])
                logger.info("Running scenario plots for %s: %s", name, " ".join(cmd))
                if not self._run_subprocess(cmd):
                    ok = False
            if ok:
                # Try to open the first plot from the first scenario
                for name in scenario_names:
                    plot_dir = self.project_path / "output_plots" / name
                    first_png = self.find_first_plot(plot_dir)
                    if first_png is not None:
                        try:
                            open_file_in_default_app(first_png)
                        except OSError:
                            logger.warning("Could not open plot: %s", first_png)
                        break
            return ok

        self._start_thread(action, _work)

    def run_scenario_excel(self, scenario_names: list[str]) -> None:
        """Generate Excel files for selected executed scenarios.

        Runs ``cmd_write_outputs`` for each scenario with ``--write-methods excel``.
        """
        action = "scen_excel"
        if not self._mark_running(action):
            return

        def _work() -> bool:
            ok = True
            for name in scenario_names:
                cmd = self._build_write_outputs_cmd(name, ["excel"])
                logger.info("Running scenario Excel for %s: %s", name, " ".join(cmd))
                if not self._run_subprocess(cmd):
                    ok = False
            return ok

        self._start_thread(action, _work)

    def run_scenario_csvs(self, scenario_names: list[str]) -> None:
        """Generate CSV files for selected executed scenarios.

        Runs ``cmd_write_outputs`` for each scenario with ``--write-methods csv``.
        """
        action = "scen_csvs"
        if not self._mark_running(action):
            return

        def _work() -> bool:
            ok = True
            for name in scenario_names:
                cmd = self._build_write_outputs_cmd(name, ["csv"])
                logger.info("Running scenario CSVs for %s: %s", name, " ".join(cmd))
                if not self._run_subprocess(cmd):
                    ok = False
            return ok

        self._start_thread(action, _work)

    # ------------------------------------------------------------------
    # Comparison actions
    # ------------------------------------------------------------------

    def run_comparison_plots(self, scenario_names: list[str]) -> None:
        """Generate comparison plots.

        Runs ``cmd_scenario_results`` with ``--all-plots``.
        Opens the first plot automatically on completion.
        """
        action = "comp_plots"
        if not self._mark_running(action):
            return

        def _work() -> bool:
            cmd = self._build_comparison_cmd(scenario_names, plots=True, excel=False)
            logger.info("Running comparison plots: %s", " ".join(cmd))
            ok = self._run_subprocess(cmd)
            if ok:
                comp_dir = self.project_path / "output_plot_comparisons"
                first_png = self.find_first_plot(comp_dir)
                if first_png is not None:
                    try:
                        open_file_in_default_app(first_png)
                    except OSError:
                        logger.warning("Could not open comparison plot: %s", first_png)
            return ok

        self._start_thread(action, _work)

    def run_comparison_excel(self, scenario_names: list[str]) -> None:
        """Generate comparison Excel.

        Runs ``cmd_scenario_results`` with ``--write-to-xlsx`` and
        ``--write-dispatch-xlsx`` (no plot flags).
        """
        action = "comp_excel"
        if not self._mark_running(action):
            return

        def _work() -> bool:
            cmd = self._build_comparison_cmd(scenario_names, plots=False, excel=True)
            logger.info("Running comparison Excel: %s", " ".join(cmd))
            return self._run_subprocess(cmd)

        self._start_thread(action, _work)

    # ------------------------------------------------------------------
    # File finders (used by Show/Open buttons)
    # ------------------------------------------------------------------

    def find_first_plot(self, directory: Path) -> Path | None:
        """Find the alphabetically first ``.png`` file in *directory* (recursive)."""
        if not directory.is_dir():
            return None
        pngs = sorted(directory.rglob("*.png"))
        return pngs[0] if pngs else None

    def find_scenario_excel(self, scenario_name: str) -> Path | None:
        """Find an Excel file for *scenario_name* in ``output_excel/``."""
        excel_dir = self.project_path / "output_excel"
        if not excel_dir.is_dir():
            return None
        for f in sorted(excel_dir.iterdir()):
            if f.suffix.lower() == ".xlsx" and scenario_name in f.stem:
                return f
        return None

    def find_comparison_excel(self) -> Path | None:
        """Find the first comparison ``.xlsx`` file in the project root."""
        if not self.project_path.is_dir():
            return None
        xlsxs = sorted(
            f for f in self.project_path.iterdir()
            if f.suffix.lower() == ".xlsx" and f.stem.startswith("compare_")
        )
        return xlsxs[0] if xlsxs else None

    # ------------------------------------------------------------------
    # Command builders
    # ------------------------------------------------------------------

    def _build_write_outputs_cmd(
        self, scenario_name: str, write_methods: list[str]
    ) -> list[str]:
        """Build the ``cmd_write_outputs`` command for a single scenario."""
        settings = self.settings
        single = settings.single_plot_settings
        parquet_dir = self.project_path / "output_parquet" / scenario_name

        cmd: list[str] = [
            sys.executable,
            "-m",
            "flextool.cli.cmd_write_outputs",
            "--scenario-name",
            scenario_name,
            "--read-parquet-dir",
            str(parquet_dir),
            "--output-location",
            str(self.project_path),
            "--write-methods",
            *write_methods,
        ]

        if single.config_file:
            cmd.extend(["--config-path", single.config_file])

        if single.active_configs:
            cmd.extend(["--active-configs", *single.active_configs])

        if single.duration > 0:
            first_row = single.start_time
            last_row = single.start_time + single.duration - 1
            cmd.extend(["--plot-rows", str(first_row), str(last_row)])

        return cmd

    def _build_comparison_cmd(
        self,
        scenario_names: list[str],
        *,
        plots: bool,
        excel: bool,
    ) -> list[str]:
        """Build the ``cmd_scenario_results`` command for comparison."""
        settings = self.settings
        comp = settings.comparison_plot_settings
        parquet_base = self.project_path / "output_parquet"
        plot_dir = self.project_path / "output_plot_comparisons"

        cmd: list[str] = [
            sys.executable,
            "-m",
            "flextool.cli.cmd_scenario_results",
            "--parquet-base-dir",
            str(parquet_base),
            "--alternatives",
            *scenario_names,
            "--plot-dir",
            str(plot_dir),
        ]

        if comp.config_file:
            cmd.extend(["--output-config-path", comp.config_file])

        if comp.active_configs:
            cmd.extend(["--active-configs", *comp.active_configs])

        if comp.duration > 0:
            first_row = comp.start_time
            last_row = comp.start_time + comp.duration - 1
            cmd.extend(["--plot-rows", str(first_row), str(last_row)])

        if plots:
            cmd.append("--all-plots")

        if excel:
            cmd.extend(["--write-to-xlsx", "--write-dispatch-xlsx"])
            cmd.extend(["--excel-dir", str(self.project_path)])

        return cmd

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mark_running(self, action: str) -> bool:
        """Mark *action* as running. Returns False if already running."""
        with self._lock:
            if action in self._running_actions:
                logger.info("Action %s is already running, ignoring", action)
                return False
            self._running_actions.add(action)
            return True

    def _mark_finished(self, action: str) -> None:
        with self._lock:
            self._running_actions.discard(action)

    def _start_thread(self, action: str, work: Callable[[], bool]) -> None:
        """Start a daemon thread that runs *work* and fires callbacks."""

        def _wrapper() -> None:
            try:
                success = work()
            except Exception:
                logger.exception("Output action %s failed with exception", action)
                success = False
            finally:
                self._mark_finished(action)

            if self._on_complete is not None:
                try:
                    self._on_complete(action, success)
                except Exception:
                    logger.exception("on_complete callback failed for %s", action)

        thread = threading.Thread(target=_wrapper, daemon=True)
        thread.start()

    @staticmethod
    def _run_subprocess(cmd: list[str]) -> bool:
        """Run a subprocess to completion. Returns True on success."""
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                logger.info("[output_action] %s", line.rstrip("\n"))
            return_code = proc.wait()
            if return_code != 0:
                logger.error("Output action process exited with code %d", return_code)
                return False
            return True
        except FileNotFoundError:
            logger.error("Could not find executable: %s", cmd[0])
            return False
        except Exception:
            logger.exception("Subprocess failed: %s", " ".join(cmd))
            return False
