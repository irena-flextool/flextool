"""Load and combine dispatch mapping parquet files across scenarios.

Exports:
    load_dispatch_mappings  — load all mapping files from one scenario folder
    combine_dispatch_mappings — combine across scenarios → DispatchMappings
    get_group_node_multiindex — build (group, node) MultiIndex for a scenario
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from flextool.scenario_comparison.data_models import DispatchMappings


def load_dispatch_mappings(parquet_dir: Path) -> dict[str, pd.DataFrame | None]:
    """
    Load all dispatch-related mapping files from a single parquet directory.

    Parameters:
    -----------
    parquet_dir : Path
        Directory containing the parquet files for one scenario

    Returns:
    --------
    dict : Dictionary containing all dispatch mapping dataframes (raw, unfiltered)
    """
    parquet_dir = Path(parquet_dir)
    mappings: dict[str, pd.DataFrame | None] = {}

    # File mapping: key -> (filename, required)
    # node_inflow is excluded — it's already in combined_dfs as 'node_inflow__dt'
    file_mapping = {
        'dispatch_groups': ('outputNodeGroup_does_specified_flows.parquet', True),
        'group_node': ('group_node.parquet', True),
        'group_process_node': ('group_process_node.parquet', False),
        # ProcessGroup aggregation files
        'processGroup_Unit_to_group': ('outputNodeGroup__processGroup_Unit_to_group.parquet', False),
        'processGroup_Group_to_unit': ('outputNodeGroup__processGroup_Group_to_unit.parquet', False),
        'processGroup_Connection': ('outputNodeGroup__processGroup_Connection.parquet', False),
        # ProcessGroup member files
        'processGroup_unit_to_node_members': ('outputNodeGroup__processGroup__process__unit__to_node.parquet', False),
        'processGroup_node_to_unit_members': ('outputNodeGroup__processGroup__process__node__to_unit.parquet', False),
        'processGroup_connection_to_node_members': ('outputNodeGroup__processGroup__process__connection__to_node.parquet', False),
        'processGroup_node_to_connection_members': ('outputNodeGroup__processGroup__process__node__to_connection.parquet', False),
        # Not_in_aggregate files
        'not_in_aggregate_unit_to_node': ('outputNodeGroup__process__unit__to_node_Not_in_aggregate.parquet', False),
        'not_in_aggregate_node_to_unit': ('outputNodeGroup__process__node__to_unit_Not_in_aggregate.parquet', False),
        'not_in_aggregate_connection_to_node': ('outputNodeGroup__process__connection__to_node_Not_in_aggregate.parquet', False),
        'not_in_aggregate_node_to_connection': ('outputNodeGroup__process__node__to_connection_Not_in_aggregate.parquet', False),
        'not_in_aggregate_connection': ('outputNodeGroup__connection_Not_in_aggregate.parquet', False),
        # Process fully inside (for internal losses)
        'process_fully_inside': ('outputNodeGroup__process_fully_inside.parquet', False),
    }

    for key, (filename, required) in file_mapping.items():
        filepath = parquet_dir / filename
        if filepath.exists():
            mappings[key] = pd.read_parquet(filepath)
        elif required:
            print(f"Warning: Required file {filepath} not found")
            mappings[key] = None
        else:
            mappings[key] = None

    return mappings


def combine_dispatch_mappings(
    scenario_folders: dict[str, str],
    parquet_subdir: str,
) -> DispatchMappings:
    """
    Load and combine dispatch mapping files across all scenarios.

    Each mapping key's DataFrames are concatenated and indexed by 'scenario'.
    Use ``DispatchMappings.get_for_scenario(field, scenario)`` for per-scenario views.

    Parameters:
    -----------
    scenario_folders : dict
        Mapping of scenario names to folder paths
    parquet_subdir : str
        Subdirectory within each folder containing parquet files

    Returns:
    --------
    DispatchMappings : Combined mapping dataclass with 'scenario' in the index
    """
    # Collect per-key lists: {key: [df, df, ...]}
    collected: dict[str, list[pd.DataFrame]] = {}

    for scen_name, folder_path in scenario_folders.items():
        pq_dir = Path(folder_path) / parquet_subdir / scen_name
        mappings = load_dispatch_mappings(pq_dir)
        for key, df in mappings.items():
            if df is None or df.empty:
                continue
            if key not in collected:
                collected[key] = []
            collected[key].append(df)

    # Concatenate and set scenario as index
    combined: dict[str, pd.DataFrame] = {}
    for key, dfs in collected.items():
        combined_df = pd.concat(dfs, ignore_index=True)
        if 'scenario' in combined_df.columns:
            combined_df = combined_df.set_index('scenario')
        combined[key] = combined_df

    return DispatchMappings(**combined)


def get_group_node_multiindex(
    group_node_df: pd.DataFrame | None,
    scenario: str | None = None,
) -> pd.MultiIndex | None:
    """
    Create a MultiIndex from group_node dataframe for a specific scenario.

    Parameters:
    -----------
    group_node_df : pd.DataFrame
        DataFrame with columns [scenario, group, node]
    scenario : str, optional
        Scenario name to filter by. If None, uses all rows.

    Returns:
    --------
    pd.MultiIndex : MultiIndex with levels [group, node]
    """
    if group_node_df is None:
        return None
    if scenario is None:
        df = group_node_df
    elif 'scenario' in group_node_df.columns:
        df = group_node_df[group_node_df['scenario'] == scenario]
    elif group_node_df.index.name == 'scenario':
        df = group_node_df.xs(scenario)
    else:
        df = group_node_df
    return pd.MultiIndex.from_frame(df[['group', 'node']])
