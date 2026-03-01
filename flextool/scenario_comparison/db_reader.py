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
        filter_configs = db_map.get_filter_configs()
        scenarios: list[str] = []
        if filter_configs:
            alternative_names = filter_configs[0]['alternatives']
            scenarios = alternative_names

        for scenario_name in scenarios:
            param_values = db_map.get_parameter_value_items(
                entity_class_name="scenario",
                entity_name=scenario_name,
                parameter_definition_name="output_location"
            )

            if param_values:
                folder_path = param_values[0]["parsed_value"]
                scenario_folders[scenario_name] = folder_path

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

            if filename not in files_by_name:
                files_by_name[filename] = []

            files_by_name[filename].append((scenario_name, parquet_file))

    return files_by_name


def combine_parquet_files(
    files_by_name: dict[str, list[tuple[str, Path]]],
) -> dict[str, pd.DataFrame]:
    """Combine parquet files across scenarios into dataframes.

    The parquet files are expected to have scenario information in a multi-index
    column level (highest level), so they are appended along the column axis.

    Parameters
    ----------
    files_by_name : dict
        Dictionary mapping filename to list of (scenario_name, file_path) tuples

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
            print(f"Combined {len(dfs_to_append)} files for variable '{variable_name}' "
                  f"(shape: {combined_df.shape})")
        else:
            print(f"Warning: No valid data found for {filename}")

    return combined_dfs


def get_scenario_results(
    db_url: str,
    parquet_subdir: str = 'output_parquet',
) -> tuple[dict[str, str], TimeSeriesResults]:
    """Load and combine all scenario parquet files into TimeSeriesResults.

    Parameters
    ----------
    db_url : str
        Database URL containing scenario information
    parquet_subdir : str
        Subdirectory within each scenario folder containing parquet files

    Returns
    -------
    tuple[dict[str, str], TimeSeriesResults]
        (scenario_folders, results) — folder mapping and combined time-series data
    """
    print(f"Reading scenario information from {db_url}...")
    scenario_folders = read_scenario_folders(db_url)
    print(f"Found {len(scenario_folders)} scenarios: {list(scenario_folders.keys())}")

    print(f"\nCollecting parquet files from {parquet_subdir} subdirectories...")
    files_by_name = collect_parquet_files(scenario_folders, parquet_subdir)
    print(f"Found {len(files_by_name)} unique result variables")

    print("\nCombining parquet files...")
    combined_dfs = combine_parquet_files(files_by_name)

    print(f"Combined {len(combined_dfs)} result variables.")

    return scenario_folders, TimeSeriesResults.from_dict(combined_dfs)
