from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from tkinter import ttk

from flextool.gui.data_models import ExecutedScenarioInfo, ProjectSettings, ScenarioInfo

logger = logging.getLogger(__name__)

# Unicode checkbox characters (mirrors main_window.py constants)
# Using geometric shapes (U+25A1 / U+25A3) which render noticeably larger
# than ballot box characters at the same font size.
CHECK_ON = "\u25a3"   # ▣
CHECK_OFF = "\u25a1"  # □


class AvailableScenarioManager:
    """Manages the available scenarios list with persistent ordering."""

    def __init__(self, settings: ProjectSettings) -> None:
        self.settings = settings
        self._scenarios: list[ScenarioInfo] = []

    def update_scenarios(self, scenarios: list[ScenarioInfo]) -> list[ScenarioInfo]:
        """Update with new scenario list from InputSourceManager.

        Apply persistent ordering from settings.scenario_order.
        New scenarios (not in saved order) go to the end.
        Scenarios no longer available are removed from the order.
        Returns the ordered list.
        """
        saved_order: list[str] = list(self.settings.scenario_order)
        scenario_by_key: dict[str, ScenarioInfo] = {}
        for s in scenarios:
            key = f"{s.source_number}|{s.name}"
            scenario_by_key[key] = s

        ordered: list[ScenarioInfo] = []

        # First: add scenarios that are in the saved order and still available
        for key in saved_order:
            if key in scenario_by_key:
                ordered.append(scenario_by_key.pop(key))

        # Second: append new scenarios (not in saved order) at the end
        for s in scenarios:
            key = f"{s.source_number}|{s.name}"
            if key in scenario_by_key:
                ordered.append(scenario_by_key.pop(key))

        self._scenarios = ordered
        return ordered

    def move_up(self, indices: list[int]) -> list[int]:
        """Move scenarios at given indices up by one position.

        Returns new indices after move. Cannot move past position 0.
        """
        if not indices:
            return []

        indices = sorted(indices)

        # If the first selected item is already at position 0, do nothing
        if indices[0] == 0:
            return indices

        for i, idx in enumerate(indices):
            # Swap with the item above
            self._scenarios[idx - 1], self._scenarios[idx] = (
                self._scenarios[idx],
                self._scenarios[idx - 1],
            )
            indices[i] = idx - 1

        return indices

    def move_down(self, indices: list[int]) -> list[int]:
        """Move scenarios at given indices down by one position.

        Returns new indices after move. Cannot move past the last position.
        """
        if not indices:
            return []

        indices = sorted(indices)
        max_idx = len(self._scenarios) - 1

        # If the last selected item is already at the bottom, do nothing
        if indices[-1] >= max_idx:
            return indices

        # Process from bottom to top to avoid conflicts
        for i in range(len(indices) - 1, -1, -1):
            idx = indices[i]
            self._scenarios[idx], self._scenarios[idx + 1] = (
                self._scenarios[idx + 1],
                self._scenarios[idx],
            )
            indices[i] = idx + 1

        return indices

    def get_order(self) -> list[str]:
        """Return current order as list of 'source_number|scenario_name' keys for persistence."""
        return [f"{s.source_number}|{s.name}" for s in self._scenarios]

    def get_checked_scenarios(self, tree: ttk.Treeview) -> list[ScenarioInfo]:
        """Get scenarios that have checkboxes checked in the treeview."""
        checked: list[ScenarioInfo] = []
        children = tree.get_children()
        for i, item in enumerate(children):
            values = tree.item(item, "values")
            if values and values[0] == CHECK_ON:
                if i < len(self._scenarios):
                    checked.append(self._scenarios[i])
        return checked

    @property
    def scenarios(self) -> list[ScenarioInfo]:
        """The current ordered list of scenarios."""
        return list(self._scenarios)


class ExecutedScenarioManager:
    """Manages the executed scenarios list with output detection."""

    def __init__(self, project_path: Path) -> None:
        self.project_path = project_path

    def scan_executed(self) -> list[ExecutedScenarioInfo]:
        """Scan output_parquet/ for subdirectories.

        Each subdir = one executed scenario.
        Timestamp: directory modification time formatted as DD.MM.YY hh:mm.
        """
        parquet_dir = self.project_path / "output_parquet"
        if not parquet_dir.is_dir():
            return []

        results: list[ExecutedScenarioInfo] = []
        for entry in sorted(parquet_dir.iterdir()):
            if not entry.is_dir():
                continue
            try:
                mtime = entry.stat().st_mtime
                ts = datetime.fromtimestamp(mtime).strftime("%d.%m.%y %H:%M")
            except OSError:
                ts = ""

            results.append(
                ExecutedScenarioInfo(
                    name=entry.name,
                    source_number=0,  # updated below if available
                    timestamp=ts,
                )
            )

        return results

    def check_outputs(
        self, scenario_names: list[str]
    ) -> dict[str, dict[str, bool]]:
        """For each scenario, check which outputs exist.

        Returns dict: {scenario_name: {has_plots: bool, has_excel: bool, has_csvs: bool}}
        """
        result: dict[str, dict[str, bool]] = {}
        for name in scenario_names:
            plots_dir = self.project_path / "output_plots" / name
            has_plots = plots_dir.is_dir() and any(plots_dir.iterdir())

            excel_dir = self.project_path / "output_excel"
            has_excel = False
            if excel_dir.is_dir():
                for f in excel_dir.iterdir():
                    if f.suffix.lower() == ".xlsx" and name in f.stem:
                        has_excel = True
                        break

            csv_dir = self.project_path / "output_csv" / name
            has_csvs = csv_dir.is_dir() and any(csv_dir.iterdir())

            result[name] = {
                "has_plots": has_plots,
                "has_excel": has_excel,
                "has_csvs": has_csvs,
            }
        return result

    def check_comparison_outputs(
        self, scenario_names: list[str]
    ) -> dict[str, bool]:
        """Check if comparison outputs exist for this set of scenarios.

        Returns: {has_comp_plots: bool, has_comp_excel: bool}
        """
        comp_dir = self.project_path / "output_plot_comparisons"

        has_comp_plots = False
        has_comp_excel = False

        if comp_dir.is_dir():
            for f in comp_dir.iterdir():
                if f.is_file():
                    if f.suffix.lower() == ".xlsx":
                        has_comp_excel = True
                    elif f.suffix.lower() in (".png", ".svg", ".pdf", ".html"):
                        has_comp_plots = True
                if has_comp_plots and has_comp_excel:
                    break

        return {
            "has_comp_plots": has_comp_plots,
            "has_comp_excel": has_comp_excel,
        }

    def delete_results(self, scenario_names: list[str]) -> None:
        """Delete all output files for given scenarios.

        Removes: output_parquet/scenario_name/, output_plots/scenario_name/,
        output_excel/*scenario_name*.xlsx, output_csv/scenario_name/
        """
        for name in scenario_names:
            # Remove output_parquet/scenario_name/
            parquet_dir = self.project_path / "output_parquet" / name
            if parquet_dir.is_dir():
                shutil.rmtree(parquet_dir, ignore_errors=True)
                logger.info("Deleted %s", parquet_dir)

            # Remove output_plots/scenario_name/
            plots_dir = self.project_path / "output_plots" / name
            if plots_dir.is_dir():
                shutil.rmtree(plots_dir, ignore_errors=True)
                logger.info("Deleted %s", plots_dir)

            # Remove matching xlsx files from output_excel/
            excel_dir = self.project_path / "output_excel"
            if excel_dir.is_dir():
                for f in excel_dir.iterdir():
                    if f.suffix.lower() == ".xlsx" and name in f.stem:
                        try:
                            f.unlink()
                            logger.info("Deleted %s", f)
                        except OSError as exc:
                            logger.warning("Could not delete %s: %s", f, exc)

            # Remove output_csv/scenario_name/
            csv_dir = self.project_path / "output_csv" / name
            if csv_dir.is_dir():
                shutil.rmtree(csv_dir, ignore_errors=True)
                logger.info("Deleted %s", csv_dir)
