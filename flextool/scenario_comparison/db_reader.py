"""Load parquet files from scenario folders and combine into TimeSeriesResults.

Functions:
- read_scenario_folders  : read scenario→folder mapping from Spine DB
- collect_parquet_files  : gather parquet paths grouped by filename
- combine_parquet_files  : concat per-scenario parquets into combined DataFrames
- get_scenario_results   : top-level convenience (returns TimeSeriesResults)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from spinedb_api import DatabaseMapping

from flextool.scenario_comparison.data_models import TimeSeriesResults


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

    with DatabaseMapping(db_url) as db_map:
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

    Returns
    -------
    dict
        Dictionary mapping filename (without extension) to combined dataframe
    """
    combined_dfs: dict[str, pd.DataFrame] = {}

    for filename, scenario_files in files_by_name.items():
        variable_name = filename.replace('.parquet', '')

        dfs_to_append = []

        for scenario_name, file_path in scenario_files:
            try:
                df = pd.read_parquet(file_path)
                dfs_to_append.append(df)

            except Exception as e:
                print(f"Error reading {file_path} for scenario {scenario_name}: {e}")
                continue

        if dfs_to_append:
            combined_df = pd.concat(dfs_to_append, axis=1)
            combined_dfs[variable_name] = combined_df
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
    combined_dfs = combine_parquet_files(files_by_name, num_scenarios=num_scenarios)

    return scenario_folders, TimeSeriesResults.from_dict(combined_dfs)
