from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from tkinter import ttk

from flextool.gui.data_models import ExecutedScenarioInfo, ProjectSettings, ScenarioInfo
from flextool.gui.scenario_key import (
    format_key,
    release_bare_owner,
    resolve_source_number,
    resolve_subdir_for_read,
)

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

    def get_order(self) -> list[str]:
        """Return current order as list of 'source_number|scenario_name' keys for persistence."""
        return [f"{s.source_number}|{s.name}" for s in self._scenarios]

    def get_checked_scenarios(self, tree: ttk.Treeview) -> list[ScenarioInfo]:
        """Get scenarios that have checkboxes checked in the treeview."""
        checked: list[ScenarioInfo] = []
        scenario_by_key: dict[str, ScenarioInfo] = {
            f"{s.source_number}|{s.name}": s for s in self._scenarios
        }
        for item in tree.get_children():
            values = tree.item(item, "values")
            if values and values[0] == CHECK_ON:
                key = f"{values[1]}|{values[2]}"
                if key in scenario_by_key:
                    checked.append(scenario_by_key[key])
        return checked

    @property
    def scenarios(self) -> list[ScenarioInfo]:
        """The current ordered list of scenarios."""
        return list(self._scenarios)


class ExecutedScenarioManager:
    """Manages the executed scenarios list with output detection."""

    def __init__(
        self,
        project_path: Path,
        settings: ProjectSettings | None = None,
    ) -> None:
        self.project_path = project_path
        # The ownership map lets us recover (source_number, scenario_name)
        # for bare-named folders. An empty map is fine — bare folders then
        # parse as ``source_number = LEGACY_SOURCE_NUMBER``.
        self._bare_owners: dict[str, int] = (
            settings.bare_output_owners if settings is not None else {}
        )
        # Retain the full settings reference so mutating writers (delete_results)
        # can update it in place when releasing ownership.
        self._settings = settings

    def scan_executed(self) -> list[ExecutedScenarioInfo]:
        """Scan output_parquet/ for subdirectories.

        Each subdir is either the bare scenario name (preferred when the
        scenario name is uncontested for this source) or
        ``<scenario_name>_<source_number>`` when another source already
        owns the bare name. Ownership is recorded in project settings and
        is used here to reverse-map bare folders to their source.
        """
        parquet_dir = self.project_path / "output_parquet"
        if not parquet_dir.is_dir():
            return []

        results: list[ExecutedScenarioInfo] = []
        for entry in sorted(parquet_dir.iterdir()):
            if not entry.is_dir():
                continue
            try:
                file_mtimes = [
                    f.stat().st_mtime
                    for f in entry.iterdir()
                    if f.is_file()
                ]
                mtime = max(file_mtimes) if file_mtimes else entry.stat().st_mtime
                ts = datetime.fromtimestamp(mtime).strftime("%d.%m.%y %H:%M")
            except OSError:
                ts = ""

            source_number, scenario_name = resolve_source_number(
                entry.name, self._bare_owners
            )
            results.append(
                ExecutedScenarioInfo(
                    name=scenario_name,
                    source_number=source_number,
                    timestamp=ts,
                )
            )

        return results

    def check_outputs(
        self, scenario_ids: list[tuple[int, str]]
    ) -> dict[str, dict[str, bool]]:
        """For each scenario, check which outputs exist.

        *scenario_ids* is a list of ``(source_number, scenario_name)`` pairs.
        The result dict is keyed by the compound ``"<src#>|<name>"`` key.
        """
        result: dict[str, dict[str, bool]] = {}
        for source_number, name in scenario_ids:
            subdir = resolve_subdir_for_read(self._bare_owners, source_number, name)

            plots_dir = self.project_path / "output_plots" / subdir
            has_plots = plots_dir.is_dir() and any(plots_dir.iterdir())

            excel_dir = self.project_path / "output_excel"
            has_excel = False
            if excel_dir.is_dir():
                for f in excel_dir.iterdir():
                    if f.suffix.lower() == ".xlsx" and subdir in f.stem:
                        has_excel = True
                        break

            csv_dir = self.project_path / "output_csv" / subdir
            has_csvs = csv_dir.is_dir() and any(csv_dir.iterdir())

            result[format_key(source_number, name)] = {
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
                    if f.suffix.lower() in (".png", ".svg", ".pdf", ".html"):
                        has_comp_plots = True
                        break

        # Check for comparison Excel in the project root directory
        if self.project_path.is_dir():
            for f in self.project_path.iterdir():
                if (
                    f.is_file()
                    and f.suffix.lower() == ".xlsx"
                    and f.stem.startswith("compare_")
                ):
                    has_comp_excel = True
                    break

        return {
            "has_comp_plots": has_comp_plots,
            "has_comp_excel": has_comp_excel,
        }

    def delete_results(self, scenario_ids: list[tuple[int, str]]) -> None:
        """Delete all output files for given scenarios.

        Removes ``output_parquet/<subdir>/``, ``output_plots/<subdir>/``,
        ``output_csv/<subdir>/`` and any matching ``output_excel/*<subdir>*.xlsx``.
        When the deleted folder is the bare-named one owned by the given
        source, the ownership record is released so another source can
        claim the bare name later.
        """
        for source_number, name in scenario_ids:
            subdir = resolve_subdir_for_read(self._bare_owners, source_number, name)
            was_bare_owner = subdir == name

            parquet_dir = self.project_path / "output_parquet" / subdir
            if parquet_dir.is_dir():
                shutil.rmtree(parquet_dir, ignore_errors=True)
                logger.info("Deleted %s", parquet_dir)

            plots_dir = self.project_path / "output_plots" / subdir
            if plots_dir.is_dir():
                shutil.rmtree(plots_dir, ignore_errors=True)
                logger.info("Deleted %s", plots_dir)

            excel_dir = self.project_path / "output_excel"
            if excel_dir.is_dir():
                for f in excel_dir.iterdir():
                    if f.suffix.lower() == ".xlsx" and subdir in f.stem:
                        try:
                            f.unlink()
                            logger.info("Deleted %s", f)
                        except OSError as exc:
                            logger.warning("Could not delete %s: %s", f, exc)

            csv_dir = self.project_path / "output_csv" / subdir
            if csv_dir.is_dir():
                shutil.rmtree(csv_dir, ignore_errors=True)
                logger.info("Deleted %s", csv_dir)

            if was_bare_owner:
                release_bare_owner(self._bare_owners, source_number, name)
