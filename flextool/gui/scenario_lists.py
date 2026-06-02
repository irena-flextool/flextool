from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from tkinter import ttk

from flextool.gui.data_models import ExecutedScenarioInfo, ProjectSettings, ScenarioInfo
from flextool.gui.scenario_key import (
    format_key,
    parse_key,
    release_bare_owner,
    resolve_source_number,
    resolve_subdir_for_read,
)

logger = logging.getLogger(__name__)

# Unicode checkbox characters (mirrors main_window.py constants)
# Using geometric shapes (U+25A1 / U+25A3) which render noticeably larger
# than ballot box characters at the same font size.
CHECK_ON = "\u25a0"   # ■
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
            if entry.name.startswith("_"):
                # Skip manifest files/directories (e.g. _axis_bounds.json).
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

        Resolution is robust to drift between ``bare_output_owners`` and
        the actual on-disk layout: we forward-scan ``output_parquet/`` and
        match each entry's ``(source_number, scenario_name)`` against the
        request. The owners map is consulted only as a fallback for the
        idempotent case where the folder is already gone.

        Also strips the deleted scenario from the shared axis-bounds
        manifest (``output_parquet/_axis_bounds.json``) so it
        doesn't continue to influence y-axis ranges in the viewer.
        """
        # Forward-scan output_parquet/ once to map (src#, name) -> subdir
        # AND collect a name-keyed fallback (for cases where the ownership
        # record disagrees with the caller-supplied source_number).
        parquet_root = self.project_path / "output_parquet"
        on_disk: dict[tuple[int, str], str] = {}
        on_disk_by_name: dict[str, list[str]] = {}
        if parquet_root.is_dir():
            for entry in parquet_root.iterdir():
                if not entry.is_dir() or entry.name.startswith("_"):
                    continue
                src, scen_name = resolve_source_number(
                    entry.name, self._bare_owners
                )
                on_disk[(src, scen_name)] = entry.name
                on_disk_by_name.setdefault(scen_name, []).append(entry.name)

        for source_number, name in scenario_ids:
            key = (source_number, name)
            if key in on_disk:
                subdir = on_disk[key]
            elif name in on_disk_by_name and len(on_disk_by_name[name]) == 1:
                # Stale owners record: only one folder matches the
                # scenario name (so it must be the one the caller meant
                # even though the (src#, name) tuple doesn't match the
                # ownership record). Trust the caller, drop the stale
                # record below if applicable.
                subdir = on_disk_by_name[name][0]
                logger.warning(
                    "delete_results: ownership map disagreed for "
                    "(source=%d, scenario=%r); using on-disk folder %r",
                    source_number, name, subdir,
                )
            else:
                # Folder isn't on disk under any matching encoding — fall back
                # to the owners-map resolution (idempotent no-op when the
                # folder has already been deleted) and log a warning so the
                # caller can still flush dangling settings state.
                subdir = resolve_subdir_for_read(
                    self._bare_owners, source_number, name
                )
                logger.warning(
                    "delete_results: no on-disk folder found for "
                    "(source=%d, scenario=%r); falling back to %r",
                    source_number, name, subdir,
                )
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

            # Strip scenario from the shared axis-bounds manifest.  The
            # manifest uses the on-disk folder name (``subdir``) as the
            # scenario key, matching how ``ManifestAccumulator`` stores it.
            try:
                from flextool.plot_outputs.shared_manifest import (
                    remove_scenario_from_manifest,
                )
                remove_scenario_from_manifest(self.project_path, subdir)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to strip %s from shared axis manifest: %s",
                    subdir, exc,
                )

            if was_bare_owner:
                # Release the bare-name claim. If the ownership record was
                # stale (different source), drop it anyway — the folder is
                # gone, so no other reader could honour the old record.
                if not release_bare_owner(self._bare_owners, source_number, name):
                    if name in self._bare_owners:
                        del self._bare_owners[name]


def prune_dangling_scenario_state(
    settings: ProjectSettings,
    available_keys: set[tuple[int, str]],
    executed_subdirs: set[str],
) -> bool:
    """Drop scenario references from settings that no longer have a backing.

    Rule: a scenario stays in settings.yaml as long as it's EITHER an
    available scenario in some input source OR has on-disk results.
    Once both conditions become false, all references in settings.yaml
    are removed.

    Parameters
    ----------
    settings:
        The project settings to mutate in place.
    available_keys:
        Set of ``(source_number, scenario_name)`` pairs that are present
        in some loaded input source's available-scenarios list.
    executed_subdirs:
        Set of on-disk subdir names under ``output_parquet/`` (already
        filtered to skip ``_``-prefixed entries).

    Order of operations
    -------------------
    Snapshot ``bare_output_owners`` at the start and use the snapshot
    consistently throughout for resolving compound keys to subdirs. The
    live ``settings.bare_output_owners`` map is then pruned LAST. This
    avoids the "I just removed the ownership record I was about to use
    to resolve another list" footgun.

    Returns True if any changes were made.
    """
    changed = False

    # Snapshot the ownership map so all resolutions use a consistent view,
    # regardless of the prune order below.
    owners_snapshot: dict[str, int] = dict(settings.bare_output_owners)

    # Set of available scenario names (ignoring source number) for lists
    # that hold bare names rather than compound keys.
    available_names: set[str] = {name for _, name in available_keys}

    # Helper: does (src, name) correspond to an existing on-disk subdir?
    def _has_executed_subdir(source_number: int, scenario_name: str) -> bool:
        subdir = resolve_subdir_for_read(
            owners_snapshot, source_number, scenario_name
        )
        return subdir in executed_subdirs

    # 1. scenario_order: drop names not in any input source's scenarios.
    new_scenario_order = [
        n for n in settings.scenario_order if n in available_names
    ]
    if new_scenario_order != settings.scenario_order:
        settings.scenario_order = new_scenario_order
        changed = True

    # 2. comp_plots / comp_excel / comp_viewer scenarios: drop entries
    #    whose subdir is not in executed_subdirs. These lists hold either
    #    bare scenario names or compound "src#|name" keys depending on
    #    when they were written; handle both forms.
    for attr in ("comp_plots_scenarios", "comp_excel_scenarios", "comp_viewer_scenarios"):
        old: list[str] = list(getattr(settings, attr))
        new: list[str] = []
        for entry in old:
            if "|" in entry:
                src, scen_name = parse_key(entry)
                if _has_executed_subdir(src, scen_name):
                    new.append(entry)
            else:
                # Bare name — keep if it matches any executed subdir
                # directly or via owners_snapshot.
                if entry in executed_subdirs:
                    new.append(entry)
                    continue
                # Try to find any (src, entry) with an executed subdir
                kept = False
                for src, scen_name in available_keys:
                    if scen_name == entry and _has_executed_subdir(src, scen_name):
                        kept = True
                        break
                if not kept:
                    # Last-ditch: try owner-based resolution with each
                    # known source number from owners_snapshot.
                    src_for_name = owners_snapshot.get(entry)
                    if src_for_name is not None and _has_executed_subdir(
                        src_for_name, entry
                    ):
                        kept = True
                if kept:
                    new.append(entry)
        if new != old:
            setattr(settings, attr, new)
            changed = True

    # 3. checked_available_scenarios: compound "src#|name"; drop those
    #    whose (src, name) is not in available_keys.
    new_checked_avail = [
        k for k in settings.checked_available_scenarios
        if parse_key(k) in available_keys
    ]
    if new_checked_avail != settings.checked_available_scenarios:
        settings.checked_available_scenarios = new_checked_avail
        changed = True

    # 4. checked_executed_scenarios: compound "src#|name"; drop those
    #    whose (src, name) does not correspond to any executed_subdir.
    new_checked_exec = [
        k for k in settings.checked_executed_scenarios
        if _has_executed_subdir(*parse_key(k))
    ]
    if new_checked_exec != settings.checked_executed_scenarios:
        settings.checked_executed_scenarios = new_checked_exec
        changed = True

    # 5. scenario_resource_history: keyed by output subdir; drop missing.
    new_resource_history = {
        k: v for k, v in settings.scenario_resource_history.items()
        if k in executed_subdirs
    }
    if new_resource_history != settings.scenario_resource_history:
        settings.scenario_resource_history = new_resource_history
        changed = True

    # 6. executed_scenario_order: bare names; keep only those that
    #    correspond to some executed_subdir.
    new_executed_order: list[str] = []
    for name in settings.executed_scenario_order:
        # Direct match (subdir == bare name).
        if name in executed_subdirs:
            new_executed_order.append(name)
            continue
        # Any source has an executed subdir for this name?
        kept = False
        # Try owners_snapshot first.
        src_for_name = owners_snapshot.get(name)
        if src_for_name is not None and _has_executed_subdir(src_for_name, name):
            kept = True
        if not kept:
            # Try via suffix-encoded subdirs in executed_subdirs.
            for subdir in executed_subdirs:
                src, scen_name = resolve_source_number(subdir, owners_snapshot)
                if scen_name == name:
                    kept = True
                    break
        if kept:
            new_executed_order.append(name)
    if new_executed_order != settings.executed_scenario_order:
        settings.executed_scenario_order = new_executed_order
        changed = True

    # 7. bare_output_owners: drop entries with NO backing in either
    #    available_keys OR on-disk executed_subdirs. Pruned LAST so the
    #    other lists' resolutions used the original ownership map.
    new_owners: dict[str, int] = {}
    for scen_name, src in settings.bare_output_owners.items():
        # Keep if still an available scenario from this source.
        if (src, scen_name) in available_keys:
            new_owners[scen_name] = src
            continue
        # Keep if there's still an on-disk folder for this (src, name).
        # The folder may be encoded as either bare or suffixed, so check
        # both forms.
        if scen_name in executed_subdirs:
            new_owners[scen_name] = src
            continue
        from flextool.gui.scenario_key import format_subdir
        if format_subdir(src, scen_name) in executed_subdirs:
            new_owners[scen_name] = src
            continue
        # No backing anywhere — drop.
    if new_owners != settings.bare_output_owners:
        settings.bare_output_owners = new_owners
        changed = True

    return changed
