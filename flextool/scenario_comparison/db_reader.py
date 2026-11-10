"""Load parquet files from scenario folders and combine into TimeSeriesResults.

Functions:
- read_scenario_folders  : read scenario→folder mapping from Spine DB
- collect_parquet_files  : gather parquet paths grouped by filename
- combine_parquet_files  : concat per-scenario parquets into combined DataFrames
- get_scenario_results   : top-level convenience (returns TimeSeriesResults)
- combine_scenario_parquets : combine per-scenario parquets from disk (for GUI)
- comparison_members / write_comparison_metadata : the ``_metadata.json`` of the comparison files
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
from flextool.update_flextool.db_handle import open_database_mapping

from flextool.lean_parquet import read_lean_parquet, write_lean_parquet
from flextool.scenario_comparison.data_models import TimeSeriesResults


#: The ``format`` of the ``_metadata.json`` :func:`write_comparison_metadata` writes.  Metadata without it (an
#: older writer) is never taken as matching the current runs.
COMPARISON_METADATA_FORMAT = 2


def new_write_id() -> str:
    """The id of one write of the comparison files: in every table's footer and in ``_metadata.json``."""
    import uuid

    return uuid.uuid4().hex


def comparison_members(member_dirs: dict[str, Path]) -> list[dict]:
    """The run each member folder holds, from its run record read strictly:
    ``[{folder, path, run_id, started_at, finished_at, outputs_written_at}]``.

    *path* is the resolved member folder.  ``run_id`` and ``outputs_written_at`` are set only when the record
    exists and ``run_record.results_usable`` holds for it; a running, failed or damaged record, one that stays
    unreadable for 2 s, and a folder without a record give ``None``.  The times are the record's (for display).
    """
    from flextool.run_record import read_status_settled, results_usable, well_formed

    members = []
    for name, folder in member_dirs.items():
        status = read_status_settled(folder)
        usable = status is not None and results_usable(status)
        shown = status if well_formed(status) else {}
        outputs = shown.get("outputs") or {}
        members.append({"folder": name, "path": str(Path(folder).resolve()),
                        "run_id": shown.get("run_id") if usable else None,
                        "started_at": shown.get("started_at"), "finished_at": shown.get("finished_at"),
                        "outputs_written_at": outputs.get("written_at") if usable else None})
    return members


def settled_members(before: list[dict], member_dirs: dict[str, Path]) -> list[dict]:
    """The members as their run records say now (:func:`comparison_members`).  A member keeps its ``run_id`` only
    when *before* (taken before any of its files was read) has the same folder path, the same ``run_id`` and the
    same ``outputs_written_at``: its results were usable then and now, and neither a run nor a rewrite of its
    outputs came between.  Otherwise its ``run_id`` is ``None``, so the comparison files are not taken as its
    results."""
    now = comparison_members(member_dirs)
    then = {m["folder"]: m for m in before}
    for member in now:
        prev = then.get(member["folder"])
        if (prev is None or prev["run_id"] is None or prev["run_id"] != member["run_id"]
                or prev["outputs_written_at"] != member["outputs_written_at"] or prev["path"] != member["path"]):
            member["run_id"] = None
            member["outputs_written_at"] = None
    return now


def write_comparison_metadata(output_dir: Path | str, scenarios: list[str], members: list[dict],
                              written_at: str, write_id: str) -> None:
    """Write ``_metadata.json`` of the comparison files: the scenario folders, the run each was built from
    (``members``), when the writing started (``written_at``), the id the tables of this write carry in their
    footers (``write_id``) and ``format`` (:data:`COMPARISON_METADATA_FORMAT`), as UTF-8 through a temporary
    file."""
    output_dir = Path(output_dir)
    meta = {"scenarios": list(scenarios), "members": members, "written_at": written_at, "write_id": write_id,
            "format": COMPARISON_METADATA_FORMAT}
    fd, tmp = tempfile.mkstemp(dir=str(output_dir), prefix="_metadata_", suffix=".tmp")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        os.replace(tmp, output_dir / "_metadata.json")
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_scenario_folders(db_url: str) -> dict[str, str]:
    """Read the scenario database to get all folder paths.

    Parameters
    ----------
    db_url : str
        Database URL containing scenario information

    Returns
    -------
    dict
        Dictionary mapping scenario names to folder paths
    """
    scenario_folders: dict[str, str] = {}

    with open_database_mapping(db_url) as db_map:
        # At the results stage, scenarios have been flattened to alternatives.
        # Get alternative names from URL filter config (set by Spine Toolbox)
        # or fall back to reading all alternatives from the database.
        filter_configs = db_map.get_filter_configs()
        alternative_names: list[str] = []
        for cfg in filter_configs:
            if cfg.get('type') == 'alternative_filter':
                alternative_names = cfg['alternatives']
                break
        if not alternative_names:
            alternative_names = [
                a["name"] for a in db_map.get_alternative_items()
            ]

        for alt_name in alternative_names:
            param_values = db_map.get_parameter_value_items(
                entity_class_name="scenario",
                entity_name=alt_name,
                parameter_definition_name="output_location",
            )

            if param_values:
                folder_path = param_values[0]["parsed_value"]
                scenario_folders[alt_name] = folder_path

    return scenario_folders


def collect_parquet_files(
    scenario_folders: dict[str, str],
    output_subdir: str = "output_parquet",
) -> dict[str, list[tuple[str, Path]]]:
    """Collect all parquet files from all scenario folders.

    Parameters
    ----------
    scenario_folders : dict
        Dictionary mapping scenario names to folder paths
    output_subdir : str
        Subdirectory within each folder containing parquet files

    Returns
    -------
    dict
        Dictionary mapping filename to list of (scenario_name, file_path) tuples
    """
    files_by_name: dict[str, list[tuple[str, Path]]] = {}

    for scenario_name, folder_path in scenario_folders.items():
        parquet_dir = Path(folder_path) / output_subdir / scenario_name

        if not parquet_dir.exists():
            print(f"Warning: {parquet_dir} does not exist for scenario {scenario_name}")
            continue

        for parquet_file in sorted(parquet_dir.glob("*.parquet")):
            filename = parquet_file.name

            # Skip metadata files (not result variables)
            if filename == 'timeline_breaks.parquet':
                continue

            if filename not in files_by_name:
                files_by_name[filename] = []

            files_by_name[filename].append((scenario_name, parquet_file))

    return files_by_name


def combine_parquet_files(
    files_by_name: dict[str, list[tuple[str, Path]]],
    num_scenarios: int = 0,
    row_order: dict[str, dict[str, pd.Index]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Combine parquet files across scenarios into dataframes.

    The parquet files are expected to have scenario information in a multi-index
    column level (highest level), so they are appended along the column axis.

    Parameters
    ----------
    files_by_name : dict
        Dictionary mapping filename to list of (scenario_name, file_path) tuples
    num_scenarios : int
        Total number of scenarios being combined (for warning about missing files)
    row_order : dict, optional
        Filled with ``{variable: {scenario: row index}}``: each scenario's own rows in
        the order of its file.  The combined row index is the union of the scenarios'
        rows in first-seen order, so a row that only a later scenario has comes after
        the first scenario's rows; pass this to ``TimeSeriesResults.from_dict`` so a
        per-scenario slice keeps its own rows and order.

    Returns
    -------
    dict
        Dictionary mapping filename (without extension) to combined dataframe
    """
    combined_dfs: dict[str, pd.DataFrame] = {}

    for filename, scenario_files in files_by_name.items():
        variable_name = filename.replace('.parquet', '')

        dfs_to_append = []
        own_rows: dict[str, pd.Index] = {}

        for scenario_name, file_path in scenario_files:
            try:
                df = read_lean_parquet(file_path)
                # Deduplicate index (can occur with overlapping solve windows)
                if df.index.duplicated().any():
                    df = df[~df.index.duplicated(keep='last')]
                # Re-tag the scenario column level to the folder identity
                # (the loop key) so two folders holding the same model
                # scenario stay distinct and all downstream slicing keys off
                # the folder name.  No-op (and byte-identical) when the
                # baked-in tag already equals the folder, e.g. the DB/CLI
                # path.  See specs/dispatch_scenario_identity_retag.md.
                if (
                    isinstance(df.columns, pd.MultiIndex)
                    and 'scenario' in df.columns.names
                ):
                    src = list(df.columns.get_level_values('scenario').unique())
                    if src != [scenario_name]:
                        df = df.rename(
                            columns={s: scenario_name for s in src},
                            level='scenario',
                        )
                elif (
                    not isinstance(df.columns, pd.MultiIndex)
                    and df.columns.name == 'scenario'
                ):
                    # Entity-less variables (e.g. period-indexed costs ``_p_``)
                    # carry a single-level column index named 'scenario'.  The
                    # MultiIndex branch above skips these, so two folders that
                    # share a baked model tag (``trade_only_shipping_4`` and
                    # ``_2`` both baked as ``trade_only_shipping``) would emit
                    # duplicate columns after the cross-scenario concat and
                    # crash the downstream ``stack``.  Re-tag here too.
                    src = list(df.columns.unique())
                    if src != [scenario_name]:
                        df = df.rename(columns={s: scenario_name for s in src})
                dfs_to_append.append(df)
                own_rows[scenario_name] = df.index

            except Exception as e:
                print(f"Error reading {file_path} for scenario {scenario_name}: {e}")
                continue

        if dfs_to_append:
            combined_df = pd.concat(dfs_to_append, axis=1)
            combined_dfs[variable_name] = combined_df
            if row_order is not None:
                row_order[variable_name] = own_rows
            if num_scenarios > 0 and len(dfs_to_append) < num_scenarios:
                print(f"Found and combined only {len(dfs_to_append)} out of {num_scenarios} "
                      f"possible files for variable '{variable_name}' (shape: {combined_df.shape})")
        else:
            print(f"Warning: No valid data found for {filename}")

    return combined_dfs


def build_scenario_folders_from_dir(
    base_dir: Path,
    scenario_names: list[str],
) -> dict[str, str]:
    """Build scenario-to-folder mapping from a directory of per-scenario parquet subdirectories.

    The returned mapping has the same format as :func:`read_scenario_folders`.
    Each scenario maps to ``str(base_dir)`` so that the existing
    ``collect_parquet_files`` / ``combine_dispatch_mappings`` helpers resolve
    the final parquet directory as ``base_dir / scenario_name`` (when called
    with ``output_subdir=""``).

    Parameters
    ----------
    base_dir : Path
        Root directory that contains one subdirectory per scenario, each
        holding ``*.parquet`` files directly.
    scenario_names : list[str]
        Scenario (alternative) names whose subdirectories should be included.

    Returns
    -------
    dict[str, str]
        ``{"scenario_name": str(base_dir), ...}`` for every requested scenario
        whose subdirectory exists under *base_dir*.
    """
    base_dir = Path(base_dir)
    scenario_folders: dict[str, str] = {}
    for name in scenario_names:
        scenario_dir = base_dir / name
        if not scenario_dir.is_dir():
            print(f"Warning: expected scenario directory {scenario_dir} does not exist – skipping")
            continue
        scenario_folders[name] = str(base_dir)
    return scenario_folders


def get_scenario_results(
    db_url: str | None = None,
    parquet_subdir: str = 'output_parquet',
    scenario_folders: dict[str, str] | None = None,
) -> tuple[dict[str, str], TimeSeriesResults]:
    """Load and combine all scenario parquet files into TimeSeriesResults.

    Parameters
    ----------
    db_url : str, optional
        Database URL containing scenario information.  Ignored when
        *scenario_folders* is supplied.
    parquet_subdir : str
        Subdirectory within each scenario folder containing parquet files
    scenario_folders : dict[str, str], optional
        Pre-built scenario-name → folder-path mapping (e.g. from
        :func:`build_scenario_folders_from_dir`).  When provided the
        database is not queried.

    Returns
    -------
    tuple[dict[str, str], TimeSeriesResults]
        (scenario_folders, results) — folder mapping and combined time-series data
    """
    if scenario_folders is None:
        if db_url is None:
            raise ValueError("Either db_url or scenario_folders must be provided")
        print(f"Reading scenario information from {db_url}...")
        scenario_folders = read_scenario_folders(db_url)
    print(f"Found {len(scenario_folders)} scenarios: {list(scenario_folders.keys())}. Processing...", flush=True)

    files_by_name = collect_parquet_files(scenario_folders, parquet_subdir)
    print(f"Found {len(files_by_name)} unique result variables from parquet subdirectories")

    num_scenarios = len(scenario_folders)
    row_order: dict[str, dict[str, pd.Index]] = {}
    combined_dfs = combine_parquet_files(files_by_name, num_scenarios=num_scenarios, row_order=row_order)

    return scenario_folders, TimeSeriesResults.from_dict(combined_dfs, scenario_rows=row_order)


def combine_scenario_parquets(
    project_path: Path,
    scenario_names: list[str],
    output_dir: Path | None = None,
) -> Path:
    """Combine per-scenario parquet files into comparison parquets.

    Reads parquets from ``project_path/output_parquet/<scenario>/`` for each
    scenario, concatenates them with scenario as the top column MultiIndex
    level, and writes the combined files to *output_dir* (defaults to
    ``project_path/output_parquet_comparison/``).

    Also writes:
    - ``_metadata.json`` with the scenario list, the run each folder held and when the writing started
    - ``timeline_breaks.parquet`` (merged from all scenarios)

    Parameters
    ----------
    project_path : Path
        Project root directory.
    scenario_names : list[str]
        Scenario names to combine.
    output_dir : Path, optional
        Where to write combined parquets. Defaults to
        ``project_path / "output_parquet_comparison"``.

    Returns
    -------
    Path
        The output directory.
    """
    project_path = Path(project_path)
    parquet_base = project_path / "output_parquet"

    if output_dir is None:
        output_dir = project_path / "output_parquet_comparison"
    output_dir = Path(output_dir)

    # 1. Build scenario folder mapping
    scenario_folders = build_scenario_folders_from_dir(parquet_base, scenario_names)
    if not scenario_folders:
        raise FileNotFoundError(
            f"No scenario directories found under {parquet_base} "
            f"for scenarios: {scenario_names}"
        )
    # The runs the members hold, before any of their files is read (see settled_members).
    member_dirs = {name: parquet_base / name for name in scenario_folders}
    members_before = comparison_members(member_dirs)

    # 2. Collect all parquet files grouped by filename
    files_by_name = collect_parquet_files(scenario_folders, output_subdir="")

    # 3. Combine parquet files across scenarios, and merge the scenarios' timeline breaks
    # (a break of one scenario also cuts the others' lines: the scenarios share one time axis)
    combined_dfs = combine_parquet_files(
        files_by_name, num_scenarios=len(scenario_names)
    )
    from flextool.plot_outputs.format_helpers import load_timeline_breaks, write_timeline_breaks
    break_times = load_timeline_breaks(*(parquet_base / name for name in scenario_folders))

    # 4. Write combined DataFrames to output_dir, each stamped with this write's id
    #    Skip mapping DataFrames (flat columns) — they have duplicate column
    #    names after axis=1 concat and are handled separately via dispatch
    #    mappings.  Only write time-series DataFrames (MultiIndex columns).
    output_dir.mkdir(parents=True, exist_ok=True)
    from flextool.run_record import now_iso
    written_at = now_iso()
    write_id = new_write_id()
    for name, df in combined_dfs.items():
        if df.empty:
            continue
        if not isinstance(df.columns, pd.MultiIndex):
            continue
        write_lean_parquet(df, output_dir / f"{name}.parquet", write_id=write_id)

    # 5. Write metadata: the scenario folders and the runs they held, checked again after every read and write
    write_comparison_metadata(output_dir, list(scenario_folders.keys()),
                              settled_members(members_before, member_dirs), written_at, write_id)

    # 6. Write the merged timeline breaks
    write_timeline_breaks(output_dir, break_times)

    written_count = sum(
        1 for df in combined_dfs.values()
        if not df.empty and isinstance(df.columns, pd.MultiIndex)
    )
    print(
        f"Wrote comparison parquets for {len(scenario_folders)} scenarios "
        f"({written_count} variables) to: {output_dir}"
    )

    # 7. Compute plot plans and availability for the comparison data.
    # Read the merged single-mode YAML and derive comparison configs from
    # each entry's ``scenario_rule`` (or an explicit
    # ``comparison_overrides.map_dimensions_for_plots``); entries with
    # neither have no comparison rendering and are dropped.
    try:
        import yaml
        from flextool.plot_outputs.orchestrator import compute_all_plot_plans
        from flextool.plot_outputs.color_template import resolve_plot_settings_path
        from flextool.plot_outputs.config import flatten_new_format
        from flextool.scenario_comparison.orchestrator import (
            _derive_comparison_settings,
        )
        from flextool.gui.project_utils import get_projects_dir

        config_path = get_projects_dir().parent / "templates" / "default_plots.yaml"
        if config_path.is_file():
            with open(config_path, "r", encoding="utf-8") as f:
                settings = yaml.safe_load(f)
            plot_settings = _derive_comparison_settings(
                flatten_new_format(settings.get("plots", {}))
            )
            bt = break_times if len(break_times) else None
            compute_all_plot_plans(
                combined_dfs, plot_settings, output_dir,
                plot_rows=(0, 167), break_times=bt,
                strip_scenario_level=False,
                color_path=resolve_plot_settings_path(project_path),
            )
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "Comparison plot plan computation failed (non-fatal): %s", exc
        )

    # 7. Return the output directory
    return output_dir
