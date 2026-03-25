"""Manages output generation actions (plots, Excel, CSV) as background operations."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from typing import Callable

from flextool.gui.cli_format import format_cmd_for_log
from flextool.gui.config_parser import parse_plot_configs
from flextool.gui.data_models import PlotSettings, ProjectSettings
from flextool.gui.output_log_window import OutputLogWindow
from flextool.gui.platform_utils import open_file_in_default_app
from flextool.gui.project_utils import get_projects_dir

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
        parent: tk.Misc | None = None,
        on_complete: Callable[[str, bool], None] | None = None,
    ) -> None:
        """
        Args:
            project_path: Path to the project directory.
            settings: Project settings (plot settings, config paths, etc.).
            parent: Parent tkinter widget for creating log windows.
            on_complete: ``callback(action_name, success)`` called when an
                action finishes.  Called from the worker thread.
        """
        self.project_path = project_path
        self.settings = settings
        self._parent = parent
        self._on_complete = on_complete
        self._running_actions: set[str] = set()
        self._log_windows: dict[str, OutputLogWindow] = {}
        self._action_gen: dict[str, int] = {}  # generation counter per action
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
        """Generate plots for selected executed scenarios."""
        action = "scen_plots"
        gen = self._mark_running(action)
        log_window = self._create_log_window(action, "Re-plot scenarios")

        def _work(lw: OutputLogWindow | None) -> bool:
            ok = True
            for name in scenario_names:
                cmd = self._build_write_outputs_cmd(name, ["plot"])
                if not self._run_subprocess(cmd, lw):
                    ok = False
            return ok

        self._start_thread(action, gen, _work, log_window)

    def run_scenario_excel(self, scenario_names: list[str]) -> None:
        """Generate Excel files for selected executed scenarios."""
        action = "scen_excel"
        gen = self._mark_running(action)
        log_window = self._create_log_window(action, "Scenarios to Excel")

        def _work(lw: OutputLogWindow | None) -> bool:
            ok = True
            for name in scenario_names:
                cmd = self._build_write_outputs_cmd(name, ["excel"])
                if not self._run_subprocess(cmd, lw):
                    ok = False
            return ok

        self._start_thread(action, gen, _work, log_window)

    def run_scenario_csvs(self, scenario_names: list[str]) -> None:
        """Generate CSV files for selected executed scenarios."""
        action = "scen_csvs"
        gen = self._mark_running(action)
        log_window = self._create_log_window(action, "Scenarios to CSVs")

        def _work(lw: OutputLogWindow | None) -> bool:
            ok = True
            for name in scenario_names:
                cmd = self._build_write_outputs_cmd(name, ["csv"])
                if not self._run_subprocess(cmd, lw):
                    ok = False
            return ok

        self._start_thread(action, gen, _work, log_window)

    # ------------------------------------------------------------------
    # Comparison actions
    # ------------------------------------------------------------------

    def run_comparison_plots(self, scenario_names: list[str]) -> None:
        """Generate comparison plots."""
        action = "comp_plots"
        gen = self._mark_running(action)
        log_window = self._create_log_window(action, "Comparison plots")

        def _work(lw: OutputLogWindow | None) -> bool:
            cmd = self._build_comparison_cmd(scenario_names, plots=True, excel=False)
            ok = self._run_subprocess(cmd, lw)
            if ok:
                comp_dir = self.project_path / "output_plot_comparisons"
                first_png = self.find_first_plot(comp_dir)
                if first_png is not None:
                    try:
                        open_file_in_default_app(first_png)
                    except OSError:
                        logger.warning("Could not open comparison plot: %s", first_png)
            return ok

        self._start_thread(action, gen, _work, log_window)

    def run_comparison_excel(self, scenario_names: list[str]) -> None:
        """Generate comparison Excel."""
        action = "comp_excel"
        gen = self._mark_running(action)
        log_window = self._create_log_window(action, "Comparison to Excel")

        def _work(lw: OutputLogWindow | None) -> bool:
            cmd = self._build_comparison_cmd(scenario_names, plots=False, excel=True)
            return self._run_subprocess(cmd, lw)

        self._start_thread(action, gen, _work, log_window)

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
    # Config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_active_configs(ps: PlotSettings, default_config: str) -> list[str]:
        """Return active configs from settings, or all configs from the file if empty."""
        if ps.active_configs:
            return ps.active_configs
        # No active configs set — read all available from the config file
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
        """Mark *action* as running.  Returns the generation number.

        If the action is already running, kills its process and closes
        the log window first, then starts a fresh run.
        """
        with self._lock:
            if action in self._running_actions:
                # Kill previous and close its window
                self._stop_and_close_log_window(action)
            self._running_actions.add(action)
            gen = self._action_gen.get(action, 0) + 1
            self._action_gen[action] = gen
            return gen

    def _mark_finished(self, action: str) -> None:
        with self._lock:
            self._running_actions.discard(action)

    def _create_log_window(self, action: str, title: str) -> OutputLogWindow | None:
        """Create a log window on the main thread. Returns None if no parent.

        Also closes any existing log window for the same action.
        """
        if self._parent is None:
            return None
        # Close previous log window for this action (finished or not)
        self._close_log_window(action)
        lw = OutputLogWindow(self._parent, title)
        self._log_windows[action] = lw
        return lw

    def _close_log_window(self, action: str) -> None:
        """Close a log window for *action* if it exists."""
        lw = self._log_windows.pop(action, None)
        if lw is not None:
            try:
                if lw.winfo_exists():
                    lw.destroy()
            except Exception:
                pass

    def _stop_and_close_log_window(self, action: str) -> None:
        """Kill the subprocess and close the log window for *action*."""
        lw = self._log_windows.pop(action, None)
        if lw is not None:
            try:
                if lw.winfo_exists():
                    lw._on_stop_and_close()
            except Exception:
                pass

    def _start_thread(
        self, action: str, gen: int,
        work: Callable[[OutputLogWindow | None], bool],
        log_window: OutputLogWindow | None = None,
    ) -> None:
        """Start a daemon thread that runs *work* and fires callbacks."""

        def _wrapper() -> None:
            try:
                success = work(log_window)
            except Exception:
                logger.exception("Output action %s failed with exception", action)
                success = False
            finally:
                self._mark_finished(action)

            # If this invocation has been superseded, skip callbacks
            with self._lock:
                superseded = self._action_gen.get(action, 0) != gen

            # Notify log window (on main thread)
            if log_window is not None and not superseded:
                try:
                    log_window.after(0, log_window.mark_finished, success)
                except Exception:
                    pass

            if self._on_complete is not None and not superseded:
                try:
                    self._on_complete(action, success)
                except Exception:
                    logger.exception("on_complete callback failed for %s", action)

        thread = threading.Thread(target=_wrapper, daemon=True)
        thread.start()

    def _run_subprocess(
        self, cmd: list[str], log_window: OutputLogWindow | None = None,
    ) -> bool:
        """Run a subprocess to completion. Returns True on success."""
        flextool_root = get_projects_dir().parent

        # Log the command (multi-line for readability, copy-pasteable with \ continuations)
        cmd_str = format_cmd_for_log(cmd)
        if log_window is not None:
            log_window.after(0, log_window.append_line, cmd_str)
            log_window.after(0, log_window.append_line, "")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(flextool_root),
            )

            # Register process with log window so Stop can kill it
            if log_window is not None:
                log_window.after(0, log_window.set_process, proc)

            assert proc.stdout is not None
            for line in proc.stdout:
                stripped = line.rstrip("\n")
                logger.debug("[output_action] %s", stripped)
                if log_window is not None:
                    try:
                        log_window.after(0, log_window.append_line, stripped)
                    except Exception:
                        pass  # Window may have been closed

            return_code = proc.wait()
            if return_code != 0:
                msg = f"Process exited with code {return_code}"
                logger.error("Output action %s", msg)
                if log_window is not None:
                    try:
                        log_window.after(0, log_window.append_line, f"\n{msg}")
                    except Exception:
                        pass
                return False
            return True
        except FileNotFoundError:
            msg = f"Could not find executable: {cmd[0]}"
            logger.error(msg)
            if log_window is not None:
                try:
                    log_window.after(0, log_window.append_line, msg)
                except Exception:
                    pass
            return False
        except Exception:
            logger.exception("Subprocess failed: %s", cmd_str)
            return False
