import pandas as pd
import os
import math
import yaml
from pathlib import Path
from spinedb_api import DatabaseMapping
from spinedb_api.filters.alternative_filter import alternative_filter_config
from spinedb_api.filters.tools import append_filter_config
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from flextool.plot_functions import plot_dict_of_dataframes


# Default color mapping for special columns
DEFAULT_SPECIAL_COLORS = {
    # Positive special columns (at top of legend/plot)
    'LossOfLoad': 'crimson',
    'Discharge': 'aqua',
    'Import': 'indigo',
    # Negative special columns (at bottom of legend/plot)
    'Charge': 'lime',
    'Export': 'purple',
    'internal_losses': 'darkgray',
}

# Special columns that should be POSITIVE (at top of stacked plot, top of legend)
POSITIVE_SPECIAL = ['LossOfLoad', 'Discharge', 'Import']
# Special columns that should be NEGATIVE (at bottom of stacked plot, bottom of legend)
NEGATIVE_SPECIAL = ['Charge', 'Export', 'internal_losses']
# Columns plotted as lines, not stacked areas
LINE_COLUMNS = ['Curtailed', 'Demand']


def get_scenarios_from_config(config: dict) -> list[str]:
    """Extract active scenario names from config, handling both dict and list formats."""
    scenarios = config.get('scenarios', [])
    if isinstance(scenarios, dict):
        return list(scenarios.keys())
    if isinstance(scenarios, list):
        # Old list format: items may be plain strings or single-key dicts
        result = []
        for item in scenarios:
            if isinstance(item, dict):
                result.extend(item.keys())
            else:
                result.append(item)
        return result
    return []


def _yaml_quote(value: str) -> str:
    """Quote a string if it contains YAML-special characters."""
    special = set('(),[]{}&*#?|->!%@`"\'')
    if any(c in special for c in str(value)):
        # Use double quotes, escaping internal double quotes
        escaped = str(value).replace('"', '\\"')
        return f'"{escaped}"'
    return str(value)


def _yaml_color(color: str) -> str:
    """Quote hex color strings starting with #."""
    if str(color).startswith('#'):
        return f'"{color}"'
    return str(color)


def _auto_assign_node_colors(columns) -> dict[str, str]:
    """Auto-assign tab20 colors for node dispatch columns."""
    cmap = plt.cm.tab20(np.linspace(0, 1, 20))
    colors: dict[str, str] = {}
    for i, col in enumerate(columns):
        col_str = str(col)
        if col_str in DEFAULT_SPECIAL_COLORS:
            colors[col_str] = DEFAULT_SPECIAL_COLORS[col_str]
        else:
            colors[col_str] = matplotlib.colors.rgb2hex(cmap[i % 20])
    return colors


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
) -> dict[str, pd.DataFrame]:
    """
    Load and combine dispatch mapping files across all scenarios.

    Each mapping key's DataFrames are concatenated and indexed by 'scenario'.
    Use `.xs(scenario_name)` to get a per-scenario view.

    Parameters:
    -----------
    scenario_folders : dict
        Mapping of scenario names to folder paths
    parquet_subdir : str
        Subdirectory within each folder containing parquet files

    Returns:
    --------
    dict : Mapping of key -> combined DataFrame with 'scenario' in the index
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

    return combined


def get_group_node_multiindex(group_node_df, scenario=None):
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


def get_group_process_multiindex(group_process_df, scenario=None):
    """
    Create a MultiIndex from group_process dataframe for a specific scenario.

    Parameters:
    -----------
    group_process_df : pd.DataFrame
        DataFrame with columns [scenario, group, process]
    scenario : str, optional
        Scenario name to filter by. If None, uses all rows.

    Returns:
    --------
    pd.MultiIndex : MultiIndex with levels [group, process]
    """
    if group_process_df is None:
        return None
    df = group_process_df if scenario is None else group_process_df[group_process_df['scenario'] == scenario]
    return pd.MultiIndex.from_frame(df[['group', 'process']])


def parse_config_with_comments(config_path):
    """
    Parse a YAML config file while tracking commented-out entries.

    Handles both new dict-format scenarios and old list-format scenarios.
    For scenarios: commented_entries['scenarios'] is dict[str, str] (name→color).
    For nodes: commented_entries['nodes'] is set[str].
    No comment tracking for positive/negative sections.

    Returns:
    --------
    tuple : (config_dict, commented_entries)
        - config_dict: parsed YAML content
        - commented_entries: dict mapping section -> commented items
          'scenarios' -> dict[str, str] (name→color)
          'nodes' -> set[str]
    """
    commented_entries: dict[str, dict[str, str] | set[str]] = {
        'scenarios': {},
        'nodes': set(),
    }

    if not os.path.exists(config_path):
        return {}, commented_entries

    with open(config_path, 'r') as f:
        lines = f.readlines()

    current_section = None
    for line in lines:
        # Check for section headers (top-level keys)
        if line and not line.startswith(' ') and not line.startswith('#') and ':' in line:
            current_section = line.split(':')[0].strip()

        stripped = line.strip()
        if not current_section or not stripped.startswith('#'):
            continue

        # Only parse comments in scenarios and nodes sections
        if current_section == 'scenarios':
            # Commented scenario: #name: color or #- name
            item_line = stripped.lstrip('#').strip()
            if item_line.startswith('- '):
                # Old list format: #- name
                name = item_line[2:].strip().strip('"').strip("'")
                if ':' in name:
                    name = name.split(':')[0].strip()
                commented_entries['scenarios'][name] = ''
            elif ':' in item_line:
                # New dict format: #name: color
                parts = item_line.split(':', 1)
                name = parts[0].strip().strip('"').strip("'")
                color = parts[1].strip().strip('"').strip("'") if len(parts) > 1 else ''
                commented_entries['scenarios'][name] = color

        elif current_section == 'nodes':
            item_line = stripped.lstrip('#').strip()
            if item_line.startswith('- '):
                item = item_line[2:].strip().strip('"').strip("'")
                commented_entries['nodes'].add(item)

    # Load the actual YAML (uncommented parts)
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f) or {}

    return config_dict, commented_entries


def write_config_with_comments(config_path, config_dict, commented_entries):
    """
    Write config to YAML file with commented entries preserved.

    New format:
    - time_to_plot: first_timestep, number_of_timesteps
    - scenarios: dict name→color (commented = #name: color)
    - positive: processGroups and processes_not_aggregated as dict name→color
    - negative: processGroups and processes_not_aggregated as dict name→color
    - nodes: list (commented = #- name)

    No nodeGroups section, no colors section.

    Parameters:
    -----------
    config_path : str or Path
        Path to write the config file
    config_dict : dict
        Main config dictionary with active entries
    commented_entries : dict
        'scenarios' -> dict[str, str] (name→color) for commented scenarios
        'nodes' -> set[str] for commented nodes
    """
    lines = []

    lines.append("# The color codes in the config file can be replaced by appropriate colors.")
    lines.append("# You can use named colors from: https://matplotlib.org/stable/gallery/color/named_colors.html")
    lines.append("# Deleting the config files resets the colors.")
    lines.append("")

    # time_to_plot
    if 'time_to_plot' in config_dict:
        value = config_dict['time_to_plot']
        lines.append("time_to_plot:")
        lines.append(f"  first_timestep: {value.get('first_timestep', 0)}")
        lines.append(f"  number_of_timesteps: {value.get('number_of_timesteps', 168)}")
        lines.append("")

    # scenarios (dict format with comments)
    if 'scenarios' in config_dict:
        lines.append("scenarios:")
        scenarios = config_dict['scenarios']
        if isinstance(scenarios, dict):
            for name, color in scenarios.items():
                lines.append(f"  {_yaml_quote(name)}: {_yaml_color(color)}")
        # Commented scenarios
        commented_scens = commented_entries.get('scenarios', {})
        if isinstance(commented_scens, dict):
            for name in sorted(commented_scens.keys()):
                color = commented_scens[name]
                lines.append(f"  #{_yaml_quote(name)}: {_yaml_color(color)}")
        lines.append("")

    # Helper to write a positive/negative section
    def _write_sign_section(section_name: str):
        section = config_dict.get(section_name)
        if section is None:
            return
        lines.append(f"{section_name}:")

        pg = section.get('processGroups')
        if pg and isinstance(pg, dict):
            lines.append("  processGroups:")
            for name, color in pg.items():
                lines.append(f"    {_yaml_quote(name)}: {_yaml_color(color)}")
        else:
            lines.append("  processGroups: {}")

        pna = section.get('processes_not_aggregated')
        if pna and isinstance(pna, dict):
            lines.append("  processes_not_aggregated:")
            for name, color in pna.items():
                lines.append(f"    {_yaml_quote(name)}: {_yaml_color(color)}")
        else:
            lines.append("  processes_not_aggregated: {}")

        lines.append("")

    _write_sign_section('positive')
    _write_sign_section('negative')

    # nodes (list format with comments)
    if 'nodes' in config_dict:
        lines.append("nodes:")
        for item in config_dict['nodes']:
            lines.append(f"  - {item}")
        for item in sorted(commented_entries.get('nodes', set())):
            lines.append(f"  #- {item}")
        lines.append("")

    with open(config_path, 'w') as f:
        f.write('\n'.join(lines))


def compute_process_group_std_order(combined_dfs, combined_mapping_dfs, scenarios, available_process_groups):
    """
    Compute standard deviation for each processGroup and return them ordered by std dev.

    Parameters:
    -----------
    combined_dfs : dict
        Combined result dataframes
    combined_mapping_dfs : dict
        Combined dispatch mapping dataframes (scenario in index)
    scenarios : list
        List of scenario names
    available_process_groups : set
        Set of available processGroup names

    Returns:
    --------
    list : processGroups ordered by std dev (lowest first)
    """
    if not scenarios or not available_process_groups:
        return sorted(available_process_groups)

    # Use the first scenario to compute std dev
    first_scenario = scenarios[0]
    if not combined_mapping_dfs:
        return sorted(available_process_groups)

    # Compute std dev for each processGroup by checking actual data
    pg_std = {}

    for pg in available_process_groups:
        std_sum = 0.0
        count = 0

        # Check unit_outputNode_dt_ee for unit-based processGroups
        if 'unit_outputNode_dt_ee' in combined_dfs:
            unit_output = combined_dfs['unit_outputNode_dt_ee']
            if first_scenario in unit_output.columns.get_level_values('scenario'):
                # Get members for this processGroup
                unit_members_all = combined_mapping_dfs.get('processGroup_unit_to_node_members')
                if unit_members_all is not None and first_scenario in unit_members_all.index:
                    unit_members = unit_members_all.xs(first_scenario)
                    if isinstance(unit_members, pd.Series):
                        unit_members = unit_members.to_frame().T
                else:
                    unit_members = None
                if unit_members is not None and not unit_members.empty:
                    members = unit_members[unit_members['group_aggregate'] == pg]
                    if not members.empty:
                        data = unit_output.xs(first_scenario, axis=1, level='scenario')
                        for _, row in members.iterrows():
                            col_key = (row['unit'], row['node'])
                            if col_key in data.columns:
                                std_sum += data[col_key].std()
                                count += 1

        # Check connection flows
        for conn_df_name in ['connection_leftward_dt_eee', 'connection_rightward_dt_eee']:
            if conn_df_name in combined_dfs:
                conn_data = combined_dfs[conn_df_name]
                if first_scenario in conn_data.columns.get_level_values('scenario'):
                    conn_members_all = combined_mapping_dfs.get('processGroup_connection_to_node_members')
                    if conn_members_all is not None and first_scenario in conn_members_all.index:
                        conn_members = conn_members_all.xs(first_scenario)
                        if isinstance(conn_members, pd.Series):
                            conn_members = conn_members.to_frame().T
                    else:
                        conn_members = None
                    if conn_members is not None and not conn_members.empty:
                        members = conn_members[conn_members['group_aggregate'] == pg]
                        if not members.empty:
                            data = conn_data.xs(first_scenario, axis=1, level='scenario')
                            for _, row in members.iterrows():
                                col_key = (row['process'], row['node'])
                                if col_key in data.columns:
                                    std_sum += data[col_key].abs().std()
                                    count += 1

        pg_std[pg] = std_sum / count if count > 0 else float('inf')

    # Sort by std dev (lowest first)
    return sorted(available_process_groups, key=lambda pg: pg_std.get(pg, float('inf')))


def create_or_update_dispatch_config(plot_dir, combined_dfs, scenarios, combined_mapping_dfs):
    """
    Create or update the dispatch plot configuration file.

    New config format: scenarios as dict name→color, positive/negative with inline
    colors, no separate nodeGroups or colors sections.

    Parameters:
    -----------
    plot_dir : str or Path
        Directory for plots (config will be stored here)
    combined_dfs : dict
        Dictionary of combined result dataframes
    scenarios : list
        List of scenario names
    combined_mapping_dfs : dict
        Combined dispatch mapping dataframes (scenario in index)

    Returns:
    --------
    dict : The configuration dictionary
    """
    config_path = Path(plot_dir) / 'config.yaml'

    # Get available data from parquet
    available_nodes = set()
    if 'node_d_ep' in combined_dfs:
        node_cols = combined_dfs['node_d_ep'].columns
        if isinstance(node_cols, pd.MultiIndex):
            available_nodes = set(node_cols.get_level_values('node').unique())

    # Collect available nodeGroups and processGroups
    available_node_groups = set()
    available_process_groups = set()
    available_processes_not_aggregated = set()

    dispatch_groups_df = combined_mapping_dfs.get('dispatch_groups')
    if dispatch_groups_df is not None and not dispatch_groups_df.empty:
        available_node_groups.update(dispatch_groups_df['group'].unique())

    for pg_key in ['processGroup_Unit_to_group', 'processGroup_Group_to_unit', 'processGroup_Connection']:
        pg_df = combined_mapping_dfs.get(pg_key)
        if pg_df is not None and not pg_df.empty:
            available_process_groups.update(pg_df['group_aggregate'].unique())

    for na_key in ['not_in_aggregate_unit_to_node', 'not_in_aggregate_node_to_unit',
                   'not_in_aggregate_connection_to_node', 'not_in_aggregate_node_to_connection']:
        na_df = combined_mapping_dfs.get(na_key)
        if na_df is not None and not na_df.empty:
            for _, row in na_df.iterrows():
                if 'process' in row and 'node' in row:
                    available_processes_not_aggregated.add(f"({row['process']}, {row['node']})")

    na_conn_df = combined_mapping_dfs.get('not_in_aggregate_connection')
    if na_conn_df is not None and not na_conn_df.empty:
        for _, row in na_conn_df.iterrows():
            available_processes_not_aggregated.add(f"({row['connection']})")

    available_scenarios = set(scenarios)

    # Parse existing config
    existing_config, commented_entries = parse_config_with_comments(config_path)

    # --- 4a: Old format migration ---
    old_format = isinstance(existing_config.get('scenarios'), list)
    old_colors: dict[str, str] = {}
    if old_format:
        # Collect colors from old colors section
        old_colors_section = existing_config.get('colors', {}) or {}
        for sub in ['positive', 'negative']:
            sub_colors = old_colors_section.get(sub, {}) or {}
            for cat in ['processGroups', 'processes_not_aggregated']:
                old_colors.update(sub_colors.get(cat, {}) or {})
        # Old flat processGroups/processes_not_aggregated colors
        old_colors.update(old_colors_section.get('processGroups', {}) or {})
        old_colors.update(old_colors_section.get('processes_not_aggregated', {}) or {})
        # Scenario colors from old format
        old_scenario_colors = old_colors_section.get('scenarios', {}) or {}
    else:
        old_scenario_colors = {}

    # --- 4b: Build scenarios dict ---
    new_config: dict = {}
    new_config['time_to_plot'] = existing_config.get('time_to_plot', {
        'first_timestep': 0,
        'number_of_timesteps': 168
    })

    scenario_colormap = plt.cm.tab10(np.linspace(0, 1, 10))

    if old_format:
        # Migrate old list format → dict with colors
        existing_scenario_list = get_scenarios_from_config(existing_config)
        active_scenarios: dict[str, str] = {}
        for i, name in enumerate(existing_scenario_list):
            if name in available_scenarios:
                color = old_scenario_colors.get(name,
                        matplotlib.colors.rgb2hex(scenario_colormap[i % 10]))
                active_scenarios[name] = color
        commented_scens: dict[str, str] = {}
        # New scenarios not in old config → active
        for name in scenarios:
            if name not in active_scenarios and name not in commented_scens:
                idx = len(active_scenarios) + len(commented_scens)
                active_scenarios[name] = matplotlib.colors.rgb2hex(scenario_colormap[idx % 10])
    else:
        # New dict format: preserve existing active + commented
        existing_scenarios = existing_config.get('scenarios', {}) or {}
        if isinstance(existing_scenarios, dict):
            existing_active_scens = existing_scenarios
        else:
            existing_active_scens = {}

        existing_commented_scens = commented_entries.get('scenarios', {})
        if not isinstance(existing_commented_scens, dict):
            existing_commented_scens = {}

        active_scenarios = {}
        commented_scens = {}

        # Existing active → keep if still available
        for name, color in existing_active_scens.items():
            if name in available_scenarios:
                active_scenarios[name] = color or ''

        # Existing commented → keep if still available
        for name, color in existing_commented_scens.items():
            if name in available_scenarios and name not in active_scenarios:
                commented_scens[name] = color or ''

        # New scenarios not in config → add as active
        all_known = set(active_scenarios) | set(commented_scens)
        for name in scenarios:
            if name not in all_known:
                idx = len(active_scenarios) + len(commented_scens)
                active_scenarios[name] = matplotlib.colors.rgb2hex(scenario_colormap[idx % 10])

    new_config['scenarios'] = active_scenarios
    new_commented: dict[str, dict[str, str] | set[str]] = {
        'scenarios': commented_scens,
        'nodes': set(),
    }

    # --- 4c: Fix positive/negative categorization across ALL nodeGroups ---
    positive_groups: set[str] = set()
    negative_groups: set[str] = set()
    positive_processes: set[str] = set()
    negative_processes: set[str] = set()

    active_scenario_names = list(active_scenarios.keys())

    if active_scenario_names and combined_mapping_dfs and available_node_groups:
        for scenario in active_scenario_names:
            for node_group in sorted(available_node_groups):
                df_sample, _ = prepare_dispatch_data(
                    combined_dfs, combined_mapping_dfs, scenario, node_group
                )
                if df_sample is None or df_sample.empty:
                    continue
                for col in df_sample.columns:
                    if col in LINE_COLUMNS:
                        continue
                    series = df_sample[col]
                    has_pos = (series > 0).any()
                    has_neg = (series < 0).any()

                    is_pg = col in available_process_groups or col in POSITIVE_SPECIAL or col in NEGATIVE_SPECIAL
                    is_pna = col in available_processes_not_aggregated

                    if is_pg:
                        if has_pos:
                            positive_groups.add(col)
                        if has_neg:
                            negative_groups.add(col)
                    elif is_pna:
                        if has_pos:
                            positive_processes.add(col)
                        if has_neg:
                            negative_processes.add(col)
    else:
        # No sample data — use defaults
        positive_groups = {pg for pg in available_process_groups if pg not in NEGATIVE_SPECIAL}
        negative_groups = {pg for pg in NEGATIVE_SPECIAL if pg in available_process_groups}
        positive_processes = set(available_processes_not_aggregated)

    # --- 4d: Assign colors inline ---
    # Collect existing inline colors from config (new format)
    existing_pos = existing_config.get('positive', {}) or {}
    existing_neg = existing_config.get('negative', {}) or {}
    existing_inline_colors: dict[str, str] = {}
    for section in [existing_pos, existing_neg]:
        for cat in ['processGroups', 'processes_not_aggregated']:
            cat_dict = section.get(cat)
            if isinstance(cat_dict, dict):
                existing_inline_colors.update(cat_dict)
    # Merge old colors from migration
    all_existing_colors = {**old_colors, **existing_inline_colors}

    process_group_colormap = plt.cm.tab20(np.linspace(0, 1, 20))
    color_idx = 0

    def _assign_color(name: str) -> str:
        nonlocal color_idx
        if name in all_existing_colors and all_existing_colors[name]:
            return all_existing_colors[name]
        if name in DEFAULT_SPECIAL_COLORS:
            return DEFAULT_SPECIAL_COLORS[name]
        color = matplotlib.colors.rgb2hex(process_group_colormap[color_idx % 20])
        color_idx += 1
        return color

    # Order positive processGroups: special first, then regular by std dev
    pos_special = [c for c in ['LossOfLoad', 'Discharge', 'Import'] if c in positive_groups]
    pos_regular = positive_groups - set(POSITIVE_SPECIAL) - set(NEGATIVE_SPECIAL)
    pos_regular_ordered = compute_process_group_std_order(
        combined_dfs, combined_mapping_dfs, active_scenario_names, pos_regular
    )
    ordered_pos_groups = pos_special + pos_regular_ordered

    # Order negative processGroups: Charge, Export first, regular, then internal_losses
    neg_special_top = [c for c in ['Charge', 'Export'] if c in negative_groups]
    neg_regular = negative_groups - set(POSITIVE_SPECIAL) - set(NEGATIVE_SPECIAL)
    neg_regular_ordered = compute_process_group_std_order(
        combined_dfs, combined_mapping_dfs, active_scenario_names, neg_regular
    )
    neg_special_bottom = ['internal_losses'] if 'internal_losses' in negative_groups else []
    ordered_neg_groups = neg_special_top + neg_regular_ordered + neg_special_bottom

    # Build positive/negative config with inline colors
    pos_pg_dict = {name: _assign_color(name) for name in ordered_pos_groups}
    pos_pna_dict = {name: _assign_color(name) for name in sorted(positive_processes)}
    neg_pg_dict = {name: _assign_color(name) for name in ordered_neg_groups}
    neg_pna_dict = {name: _assign_color(name) for name in sorted(negative_processes)}

    new_config['positive'] = {
        'processGroups': pos_pg_dict,
        'processes_not_aggregated': pos_pna_dict,
    }
    new_config['negative'] = {
        'processGroups': neg_pg_dict,
        'processes_not_aggregated': neg_pna_dict,
    }

    # --- 4e: Build nodes section ---
    existing_nodes = list(existing_config.get('nodes', []) or [])
    existing_commented_nodes = commented_entries.get('nodes', set())
    if not isinstance(existing_commented_nodes, set):
        existing_commented_nodes = set()

    if existing_nodes or existing_commented_nodes:
        # Preserve existing active/commented state
        active_nodes = [n for n in existing_nodes if n in available_nodes]
        commented_nodes = {n for n in existing_commented_nodes if n in available_nodes}
        # New nodes not seen before → add to commented
        all_known_nodes = set(existing_nodes) | existing_commented_nodes
        new_nodes = available_nodes - all_known_nodes
        commented_nodes.update(new_nodes)
    else:
        # First run: first 5 active, rest commented
        sorted_nodes = sorted(available_nodes)
        active_nodes = sorted_nodes[:5]
        commented_nodes = set(sorted_nodes[5:])

    new_config['nodes'] = active_nodes
    new_commented['nodes'] = commented_nodes

    # --- Write the config file (no nodeGroups, no colors section) ---
    write_config_with_comments(config_path, new_config, new_commented)
    print(f"Updated dispatch config at {config_path}")

    return new_config


def plot_dispatch_area(df_dispatch, inflow_series, output_path, title, ylabel="MWh/h",
                       colors=None, timeline=(0, 168), show_plot=False, ylim=None):
    """
    Create a stacked area dispatch plot with demand line.

    Parameters:
    -----------
    df_dispatch : pd.DataFrame
        DataFrame with time index and columns for each generation type
    inflow_series : pd.Series
        Demand/inflow time series to plot as line
    output_path : str or Path
        Path to save the plot
    title : str
        Plot title
    ylabel : str
        Y-axis label
    colors : dict
        Mapping of column names to colors
    timeline : tuple
        (start, end) indices for time range to plot
    show_plot : bool
        Whether to display the plot
    ylim : tuple, optional
        Y-axis limits (min, max)
    """
    if colors is None:
        colors = DEFAULT_SPECIAL_COLORS

    def get_color_for_column(col, colors_dict):
        """Get color for a column, handling _in/_out suffixes."""
        # Direct lookup
        if col in colors_dict and colors_dict[col] is not None:
            return colors_dict[col]
        # Try base name without _in/_out suffix
        if col.endswith('_in') or col.endswith('_out'):
            base_name = col[:-3] if col.endswith('_in') else col[:-4]
            if base_name in colors_dict and colors_dict[base_name] is not None:
                return colors_dict[base_name]
        # Default color
        return 'lightgray'

    # Get plot colors for columns (excluding 'Curtailed' which is plotted as line)
    plot_cols = [col for col in df_dispatch.columns if col != 'Curtailed']
    plot_colors = [get_color_for_column(col, colors) for col in plot_cols]

    # Slice to timeline
    df_plot = df_dispatch.iloc[timeline[0]:timeline[1]]

    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot area chart
    df_plot[plot_cols].plot.area(
        ax=ax,
        stacked=True,
        linewidth=0,
        color=plot_colors,
        legend=False
    )

    # Plot curtailed as dashed line if present
    if 'Curtailed' in df_dispatch.columns:
        curtailed = df_plot['Curtailed']
        ax.plot(curtailed.index, curtailed.values, linestyle='--', color='red', linewidth=1, label='Curtailed')

    # Plot demand line
    if inflow_series is not None:
        inflow_plot = inflow_series.iloc[timeline[0]:timeline[1]]
        ax.plot(inflow_plot.index, inflow_plot.values, linestyle='solid', color='black', linewidth=1.5, label='Demand')

    ax.axhline(y=0, color='black', linestyle=':', linewidth=0.5)

    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if ylim:
        ax.set_ylim(ylim)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='png', bbox_inches='tight', dpi=150)

    if show_plot:
        plt.show()
    plt.close()


def prepare_dispatch_data(combined_dfs, combined_mapping_dfs, scenario, output_node_group,
                          colors=None):
    """
    Prepare dispatch data for a specific outputNodeGroup using the new parquet-based mappings.

    Columns are validated for sign consistency:
    - Positive special columns: LossOfLoad, Discharge, Import
    - Negative special columns: Charge, Export, internal_losses
    - Columns with mixed signs are excluded with a warning

    Parameters:
    -----------
    combined_dfs : dict
        Combined result dataframes
    combined_mapping_dfs : dict
        Combined dispatch mapping dataframes (scenario in index)
    scenario : str
        Scenario name
    output_node_group : str
        Name of the output node group to prepare dispatch for
    colors : dict
        Color mapping for process groups

    Returns:
    --------
    tuple : (df_dispatch, inflow_series) or (None, None) if data not available
    """
    if colors is None:
        colors = DEFAULT_SPECIAL_COLORS

    def _get_mapping(key: str) -> pd.DataFrame | None:
        """Get per-scenario mapping DataFrame via .xs(), or None."""
        df = combined_mapping_dfs.get(key)
        if df is None or df.empty:
            return None
        if df.index.name == 'scenario':
            if scenario not in df.index:
                return None
            result = df.xs(scenario)
            # xs() returns a Series when there's only one matching row; ensure DataFrame
            if isinstance(result, pd.Series):
                return result.to_frame().T
            return result
        return df

    try:
        # Validate that this group should have dispatch plots
        dispatch_groups_df = _get_mapping('dispatch_groups')
        if dispatch_groups_df is None or dispatch_groups_df.empty:
            return None, None

        if output_node_group not in dispatch_groups_df['group'].values:
            return None, None

        # Get nodes in this group
        group_node_df = _get_mapping('group_node')
        if group_node_df is None or group_node_df.empty:
            return None, None

        nodes_in_group = group_node_df[group_node_df['group'] == output_node_group]['node'].tolist()
        if not nodes_in_group:
            return None, None

        # Initialize dispatch dataframe with time index
        # Get time index from unit output or connection data
        time_index = None
        if 'unit_outputNode_dt_ee' in combined_dfs:
            df_temp = combined_dfs['unit_outputNode_dt_ee']
            if scenario in df_temp.columns.get_level_values('scenario'):
                time_index = df_temp.xs(scenario, axis=1, level='scenario').groupby('time').sum().index

        if time_index is None:
            if 'connection_leftward_dt_eee' in combined_dfs:
                df_temp = combined_dfs['connection_leftward_dt_eee']
                if scenario in df_temp.columns.get_level_values('scenario'):
                    time_index = df_temp.xs(scenario, axis=1, level='scenario').groupby('time').sum().index

        if time_index is None:
            return None, None

        df_dispatch = pd.DataFrame(index=time_index)

        # --- Process Unit_to_group (unit outputs to nodes in group) ---
        unit_to_group_df = _get_mapping('processGroup_Unit_to_group')
        unit_to_group_members = _get_mapping('processGroup_unit_to_node_members')

        if unit_to_group_df is not None and not unit_to_group_df.empty:
            group_aggregates = unit_to_group_df[unit_to_group_df['group'] == output_node_group]['group_aggregate'].unique()

            if 'unit_outputNode_dt_ee' in combined_dfs:
                unit_output = combined_dfs['unit_outputNode_dt_ee'].copy()
                if scenario in unit_output.columns.get_level_values('scenario'):
                    unit_output = unit_output.xs(scenario, axis=1, level='scenario').groupby('time').sum()

                    for group_agg in group_aggregates:
                        # Get members for this processGroup
                        if unit_to_group_members is not None and not unit_to_group_members.empty:
                            members = unit_to_group_members[
                                (unit_to_group_members['group'] == output_node_group) &
                                (unit_to_group_members['group_aggregate'] == group_agg)
                            ]

                            flow_sum = pd.Series(0.0, index=time_index)
                            for _, row in members.iterrows():
                                unit = row['unit']
                                node = row['node']
                                col_key = (unit, node)
                                if col_key in unit_output.columns:
                                    flow_sum = flow_sum.add(unit_output[col_key], fill_value=0)

                            if flow_sum.abs().sum() > 0:
                                df_dispatch[group_agg] = flow_sum

        # --- Process Group_to_unit (node inputs to units, negative values) ---
        group_to_unit_df = _get_mapping('processGroup_Group_to_unit')
        group_to_unit_members = _get_mapping('processGroup_node_to_unit_members')

        if group_to_unit_df is not None and not group_to_unit_df.empty:
            group_aggregates = group_to_unit_df[group_to_unit_df['group'] == output_node_group]['group_aggregate'].unique()

            if 'unit_inputNode_dt_ee' in combined_dfs:
                unit_input = combined_dfs['unit_inputNode_dt_ee'].copy()
                if scenario in unit_input.columns.get_level_values('scenario'):
                    unit_input = unit_input.xs(scenario, axis=1, level='scenario').groupby('time').sum()

                    for group_agg in group_aggregates:
                        if group_to_unit_members is not None and not group_to_unit_members.empty:
                            members = group_to_unit_members[
                                (group_to_unit_members['group'] == output_node_group) &
                                (group_to_unit_members['group_aggregate'] == group_agg)
                            ]

                            flow_sum = pd.Series(0.0, index=time_index)
                            for _, row in members.iterrows():
                                unit = row['unit']
                                node = row['node']
                                col_key = (unit, node)
                                if col_key in unit_input.columns:
                                    # Input flows are negative (consumption)
                                    flow_sum = flow_sum.add(-unit_input[col_key], fill_value=0)

                            if flow_sum.abs().sum() > 0:
                                if group_agg in df_dispatch.columns:
                                    df_dispatch[group_agg] = df_dispatch[group_agg].add(flow_sum, fill_value=0)
                                else:
                                    df_dispatch[group_agg] = flow_sum

        # --- Process Connection flows ---
        connection_df = _get_mapping('processGroup_Connection')
        conn_to_node_members = _get_mapping('processGroup_connection_to_node_members')
        conn_from_node_members = _get_mapping('processGroup_node_to_connection_members')

        if connection_df is not None and not connection_df.empty:
            group_aggregates = connection_df[connection_df['group'] == output_node_group]['group_aggregate'].unique()

            # Load connection flow data
            conn_left = None
            conn_right = None
            if 'connection_leftward_dt_eee' in combined_dfs:
                conn_left = combined_dfs['connection_leftward_dt_eee'].copy()
                if scenario in conn_left.columns.get_level_values('scenario'):
                    conn_left = conn_left.xs(scenario, axis=1, level='scenario').groupby('time').sum()
                else:
                    conn_left = None

            if 'connection_rightward_dt_eee' in combined_dfs:
                conn_right = combined_dfs['connection_rightward_dt_eee'].copy()
                if scenario in conn_right.columns.get_level_values('scenario'):
                    conn_right = conn_right.xs(scenario, axis=1, level='scenario').groupby('time').sum()
                else:
                    conn_right = None

            for group_agg in group_aggregates:
                flow_sum = pd.Series(0.0, index=time_index)

                # Connection to node (leftward flows - node receives from connection)
                if conn_to_node_members is not None and not conn_to_node_members.empty and conn_left is not None:
                    members = conn_to_node_members[
                        (conn_to_node_members['group'] == output_node_group) &
                        (conn_to_node_members['group_aggregate'] == group_agg)
                    ]
                    for _, row in members.iterrows():
                        process = row['process']
                        node = row['node']
                        col_key = (process, node)
                        if col_key in conn_left.columns:
                            flow_sum = flow_sum.add(conn_left[col_key], fill_value=0)

                # Node to connection (rightward flows - node sends to connection)
                if conn_from_node_members is not None and not conn_from_node_members.empty and conn_right is not None:
                    members = conn_from_node_members[
                        (conn_from_node_members['group'] == output_node_group) &
                        (conn_from_node_members['group_aggregate'] == group_agg)
                    ]
                    for _, row in members.iterrows():
                        process = row['process']
                        node = row['node']
                        col_key = (process, node)
                        if col_key in conn_right.columns:
                            flow_sum = flow_sum.add(conn_right[col_key], fill_value=0)

                if flow_sum.abs().sum() > 0:
                    if group_agg in df_dispatch.columns:
                        df_dispatch[group_agg] = df_dispatch[group_agg].add(flow_sum, fill_value=0)
                    else:
                        df_dispatch[group_agg] = flow_sum

        # --- Process Not_in_aggregate entries (individual columns) ---
        # Unit to node not in aggregate
        not_agg_unit_to_node = _get_mapping('not_in_aggregate_unit_to_node')
        if not_agg_unit_to_node is not None and not not_agg_unit_to_node.empty:
            entries = not_agg_unit_to_node[not_agg_unit_to_node['group'] == output_node_group]
            if 'unit_outputNode_dt_ee' in combined_dfs:
                unit_output = combined_dfs['unit_outputNode_dt_ee'].copy()
                if scenario in unit_output.columns.get_level_values('scenario'):
                    unit_output = unit_output.xs(scenario, axis=1, level='scenario').groupby('time').sum()

                    for _, row in entries.iterrows():
                        unit = row['unit']
                        node = row['node']
                        col_key = (unit, node)
                        col_name = f"({row['process']}, {node})"
                        if col_key in unit_output.columns:
                            df_dispatch[col_name] = unit_output[col_key]

        # Node to unit not in aggregate
        not_agg_node_to_unit = _get_mapping('not_in_aggregate_node_to_unit')
        if not_agg_node_to_unit is not None and not not_agg_node_to_unit.empty:
            entries = not_agg_node_to_unit[not_agg_node_to_unit['group'] == output_node_group]
            if 'unit_inputNode_dt_ee' in combined_dfs:
                unit_input = combined_dfs['unit_inputNode_dt_ee'].copy()
                if scenario in unit_input.columns.get_level_values('scenario'):
                    unit_input = unit_input.xs(scenario, axis=1, level='scenario').groupby('time').sum()

                    for _, row in entries.iterrows():
                        unit = row['unit']
                        node = row['node']
                        col_key = (unit, node)
                        col_name = f"({row['process']}, {node})"
                        if col_key in unit_input.columns:
                            df_dispatch[col_name] = -unit_input[col_key]

        # Connection to node not in aggregate
        not_agg_conn_to_node = _get_mapping('not_in_aggregate_connection_to_node')
        if not_agg_conn_to_node is not None and not not_agg_conn_to_node.empty:
            entries = not_agg_conn_to_node[not_agg_conn_to_node['group'] == output_node_group]
            if conn_left is not None:
                for _, row in entries.iterrows():
                    process = row['process']
                    node = row['node']
                    col_key = (process, node)
                    col_name = f"({process}, {node})"
                    if col_key in conn_left.columns:
                        df_dispatch[col_name] = conn_left[col_key]

        # Node to connection not in aggregate
        not_agg_node_to_conn = _get_mapping('not_in_aggregate_node_to_connection')
        if not_agg_node_to_conn is not None and not not_agg_node_to_conn.empty:
            entries = not_agg_node_to_conn[not_agg_node_to_conn['group'] == output_node_group]
            if conn_right is not None:
                for _, row in entries.iterrows():
                    process = row['process']
                    node = row['node']
                    col_key = (process, node)
                    col_name = f"({process}, {node})"
                    if col_key in conn_right.columns:
                        if col_name in df_dispatch.columns:
                            df_dispatch[col_name] = df_dispatch[col_name].add(conn_right[col_key], fill_value=0)
                        else:
                            df_dispatch[col_name] = conn_right[col_key]

        # Connection not in aggregate (total flow)
        not_agg_connection = _get_mapping('not_in_aggregate_connection')
        if not_agg_connection is not None and not not_agg_connection.empty:
            entries = not_agg_connection[not_agg_connection['group'] == output_node_group]
            # Sum both directions for each connection
            if conn_left is not None or conn_right is not None:
                for _, row in entries.iterrows():
                    connection = row['connection']
                    col_name = f"({connection})"
                    flow_sum = pd.Series(0.0, index=time_index)

                    if conn_left is not None:
                        matching_cols = [c for c in conn_left.columns if c[0] == connection]
                        for col in matching_cols:
                            if col[1] in nodes_in_group:
                                flow_sum = flow_sum.add(conn_left[col], fill_value=0)

                    if conn_right is not None:
                        matching_cols = [c for c in conn_right.columns if c[0] == connection]
                        for col in matching_cols:
                            if col[1] in nodes_in_group:
                                flow_sum = flow_sum.add(conn_right[col], fill_value=0)

                    if flow_sum.abs().sum() > 0:
                        df_dispatch[col_name] = flow_sum

        # --- Process fully inside (internal losses) ---
        process_fully_inside = _get_mapping('process_fully_inside')
        if process_fully_inside is not None and not process_fully_inside.empty:
            entries = process_fully_inside[process_fully_inside['group'] == output_node_group]
            if not entries.empty and 'connection_losses_dt_eee' in combined_dfs:
                conn_losses = combined_dfs['connection_losses_dt_eee'].copy()
                if scenario in conn_losses.columns.get_level_values('scenario'):
                    conn_losses = conn_losses.xs(scenario, axis=1, level='scenario').groupby('time').sum()

                    loss_sum = pd.Series(0.0, index=time_index)
                    for _, row in entries.iterrows():
                        process = row['process']
                        # Connection losses have columns like (connection,) - process is the connection name
                        if process in conn_losses.columns:
                            loss_sum = loss_sum.add(conn_losses[process], fill_value=0)
                        # Also check if it's in a tuple column format
                        matching_cols = [c for c in conn_losses.columns if (isinstance(c, tuple) and c[0] == process) or c == process]
                        for col in matching_cols:
                            if col != process:  # Avoid double counting
                                loss_sum = loss_sum.add(conn_losses[col], fill_value=0)

                    if loss_sum.abs().sum() > 0:
                        df_dispatch['internal_losses'] = loss_sum

        # --- Get loss of load (slack) ---
        if 'node_slack_up_dt_e' in combined_dfs:
            slack = combined_dfs['node_slack_up_dt_e'].copy()
            if scenario in slack.columns.get_level_values('scenario'):
                slack = slack.xs(scenario, axis=1, level='scenario')
                slack_filtered = slack.loc[:, slack.columns.isin(nodes_in_group)]
                if not slack_filtered.empty:
                    lol = slack_filtered.groupby('time').sum().sum(axis=1)
                    if lol.abs().sum() > 0:
                        df_dispatch['LossOfLoad'] = lol

        # --- Get curtailment ---
        if 'unit_curtailment_outputNode_dt_ee' in combined_dfs:
            curtail = combined_dfs['unit_curtailment_outputNode_dt_ee'].copy()
            if scenario in curtail.columns.get_level_values('scenario'):
                curtail = curtail.xs(scenario, axis=1, level='scenario')
                # Column names might be 'sink' instead of 'node'
                node_level = 'sink' if 'sink' in curtail.columns.names else 'node'
                curtail_filtered = curtail.loc[:, curtail.columns.get_level_values(node_level).isin(nodes_in_group)]
                if not curtail_filtered.empty:
                    curtailed = curtail_filtered.groupby('time').sum().sum(axis=1).clip(lower=0)
                    if curtailed.abs().sum() > 0:
                        df_dispatch['Curtailed'] = curtailed

        # --- Validate sign and categorize columns ---
        # Each column must be either fully positive or fully negative for stacked area plots
        # Columns with mixed signs are warned about and excluded

        positive_cols = []
        negative_cols = []
        excluded_cols = []

        for col in df_dispatch.columns:
            if col in LINE_COLUMNS:
                continue  # Skip line columns (Curtailed, Demand)

            series = df_dispatch[col]
            has_pos = (series > 0).any()
            has_neg = (series < 0).any()

            if has_pos and has_neg:
                # Mixed sign - warn and exclude
                print(f"  Warning: Column '{col}' has mixed positive/negative values - excluding from plot")
                excluded_cols.append(col)
            elif has_neg:
                negative_cols.append(col)
            elif has_pos:
                positive_cols.append(col)
            # If all zeros, skip

        # Remove excluded columns
        if excluded_cols:
            df_dispatch = df_dispatch.drop(columns=excluded_cols)

        # --- Validate special column signs ---
        # Check that special columns have the expected sign
        for col in POSITIVE_SPECIAL:
            if col in negative_cols:
                print(f"  Warning: '{col}' is expected to be positive but has negative values")
        for col in NEGATIVE_SPECIAL:
            if col in positive_cols:
                print(f"  Warning: '{col}' is expected to be negative but has positive values")

        # --- Order columns by std dev within each category ---
        # Separate special columns from regular processGroups
        pos_special = [c for c in positive_cols if c in POSITIVE_SPECIAL]
        pos_regular = [c for c in positive_cols if c not in POSITIVE_SPECIAL]
        neg_special = [c for c in negative_cols if c in NEGATIVE_SPECIAL]
        neg_regular = [c for c in negative_cols if c not in NEGATIVE_SPECIAL]

        # Sort regular columns by std dev (lowest first = closest to origin)
        if pos_regular:
            col_std = {col: df_dispatch[col].std() for col in pos_regular}
            pos_regular = sorted(pos_regular, key=lambda c: col_std.get(c, 0))
        if neg_regular:
            col_std = {col: df_dispatch[col].abs().std() for col in neg_regular}
            neg_regular = sorted(neg_regular, key=lambda c: col_std.get(c, 0))

        # --- Build final column order ---
        # Stacking order (bottom to top of plot):
        # 1. Negative columns: regular (by std dev), then special (Charge, Export, internal_losses)
        # 2. Positive columns: regular (by std dev), then special (Import, Discharge, LossOfLoad)
        #
        # Legend order (matches config, top to bottom):
        # Positive special, positive regular (reversed), negative regular (reversed), negative special

        # Order for stacking (first = bottom of stack)
        ordered_cols = []

        # Negative columns (these go below x-axis)
        # Regular negative (lowest std dev first = closest to x-axis)
        ordered_cols.extend(neg_regular)
        # Special negative: internal_losses, Export, Charge (bottom of legend = top of negative stack)
        for col in ['internal_losses', 'Export', 'Charge']:
            if col in neg_special:
                ordered_cols.append(col)

        # Positive columns (these go above x-axis)
        # Regular positive (lowest std dev first = closest to x-axis)
        ordered_cols.extend(pos_regular)
        # Special positive: Import, Discharge, LossOfLoad (top of legend = top of positive stack)
        for col in ['Import', 'Discharge', 'LossOfLoad']:
            if col in pos_special:
                ordered_cols.append(col)

        # Line overlay: Curtailed (plotted as dashed line, not area)
        if 'Curtailed' in df_dispatch.columns:
            ordered_cols.append('Curtailed')

        if ordered_cols:
            df_dispatch = df_dispatch[ordered_cols]

        # --- Get demand from node_inflow__dt (from combined_dfs, not dispatch mappings) ---
        inflow_series = None
        node_inflow = combined_dfs.get('node_inflow__dt')

        if node_inflow is not None and not node_inflow.empty:
            # node_inflow has MultiIndex columns (scenario, node) - values are negative for demand
            if isinstance(node_inflow.columns, pd.MultiIndex):
                if scenario in node_inflow.columns.get_level_values('scenario'):
                    inflow_data = node_inflow.xs(scenario, axis=1, level='scenario')
                    inflow_filtered = inflow_data.loc[:, inflow_data.columns.isin(nodes_in_group)]
                    if not inflow_filtered.empty:
                        # Values are already negative for demand, so negate to get positive demand
                        inflow_series = -inflow_filtered.groupby('time').sum().sum(axis=1)
            else:
                # Single scenario format
                inflow_filtered = node_inflow.loc[:, node_inflow.columns.isin(nodes_in_group)]
                if not inflow_filtered.empty:
                    inflow_series = -inflow_filtered.groupby('time').sum().sum(axis=1)

        return df_dispatch, inflow_series

    except Exception as e:
        print(f"Error preparing dispatch data for {output_node_group} in {scenario}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def prepare_node_dispatch_data(combined_dfs, scenario: str, node: str):
    """
    Prepare dispatch data for a single node (not a nodeGroup).

    Collects unit outputs/inputs, connection flows, slack (LossOfLoad),
    and demand for the given node.

    Parameters:
    -----------
    combined_dfs : dict
        Combined result dataframes
    scenario : str
        Scenario name
    node : str
        Node name

    Returns:
    --------
    tuple : (df_dispatch, inflow_series) or (None, None) if data not available
    """
    try:
        # Get time index
        time_index = None
        for df_name in ['unit_outputNode_dt_ee', 'connection_leftward_dt_eee']:
            if df_name in combined_dfs:
                df_temp = combined_dfs[df_name]
                if scenario in df_temp.columns.get_level_values('scenario'):
                    time_index = df_temp.xs(scenario, axis=1, level='scenario').groupby('time').sum().index
                    break
        if time_index is None:
            return None, None

        df_dispatch = pd.DataFrame(index=time_index)

        # --- Unit outputs to this node (positive) ---
        if 'unit_outputNode_dt_ee' in combined_dfs:
            unit_output = combined_dfs['unit_outputNode_dt_ee']
            if scenario in unit_output.columns.get_level_values('scenario'):
                data = unit_output.xs(scenario, axis=1, level='scenario').groupby('time').sum()
                # Filter columns where node matches
                for col in data.columns:
                    col_node = col[1] if isinstance(col, tuple) else col
                    if col_node == node:
                        unit_name = col[0] if isinstance(col, tuple) else col
                        series = data[col].clip(lower=0)
                        if series.abs().sum() > 0:
                            df_dispatch[f"{unit_name}_out"] = series

        # --- Unit inputs from this node (negative) ---
        if 'unit_inputNode_dt_ee' in combined_dfs:
            unit_input = combined_dfs['unit_inputNode_dt_ee']
            if scenario in unit_input.columns.get_level_values('scenario'):
                data = unit_input.xs(scenario, axis=1, level='scenario').groupby('time').sum()
                for col in data.columns:
                    col_node = col[1] if isinstance(col, tuple) else col
                    if col_node == node:
                        unit_name = col[0] if isinstance(col, tuple) else col
                        series = -data[col]
                        neg_part = series.clip(upper=0)
                        if neg_part.abs().sum() > 0:
                            df_dispatch[f"{unit_name}_in"] = neg_part

        # --- Connection leftward flows (node receives) ---
        if 'connection_leftward_dt_eee' in combined_dfs:
            conn_left = combined_dfs['connection_leftward_dt_eee']
            if scenario in conn_left.columns.get_level_values('scenario'):
                data = conn_left.xs(scenario, axis=1, level='scenario').groupby('time').sum()
                for col in data.columns:
                    col_node = col[1] if isinstance(col, tuple) else col
                    if col_node == node:
                        conn_name = col[0] if isinstance(col, tuple) else col
                        series = data[col]
                        pos_part = series.clip(lower=0)
                        neg_part = series.clip(upper=0)
                        if pos_part.abs().sum() > 0:
                            col_label = f"{conn_name}_left"
                            if col_label in df_dispatch.columns:
                                df_dispatch[col_label] = df_dispatch[col_label].add(pos_part, fill_value=0)
                            else:
                                df_dispatch[col_label] = pos_part
                        if neg_part.abs().sum() > 0:
                            col_label = f"{conn_name}_left_neg"
                            df_dispatch[col_label] = neg_part

        # --- Connection rightward flows (node sends) ---
        if 'connection_rightward_dt_eee' in combined_dfs:
            conn_right = combined_dfs['connection_rightward_dt_eee']
            if scenario in conn_right.columns.get_level_values('scenario'):
                data = conn_right.xs(scenario, axis=1, level='scenario').groupby('time').sum()
                for col in data.columns:
                    col_node = col[1] if isinstance(col, tuple) else col
                    if col_node == node:
                        conn_name = col[0] if isinstance(col, tuple) else col
                        series = data[col]
                        pos_part = series.clip(lower=0)
                        neg_part = series.clip(upper=0)
                        if pos_part.abs().sum() > 0:
                            col_label = f"{conn_name}_right"
                            if col_label in df_dispatch.columns:
                                df_dispatch[col_label] = df_dispatch[col_label].add(pos_part, fill_value=0)
                            else:
                                df_dispatch[col_label] = pos_part
                        if neg_part.abs().sum() > 0:
                            col_label = f"{conn_name}_right_neg"
                            df_dispatch[col_label] = neg_part

        # --- Loss of load (slack up) ---
        if 'node_slack_up_dt_e' in combined_dfs:
            slack = combined_dfs['node_slack_up_dt_e']
            if scenario in slack.columns.get_level_values('scenario'):
                slack_data = slack.xs(scenario, axis=1, level='scenario')
                if node in slack_data.columns:
                    lol = slack_data[node].groupby('time').sum()
                    if lol.abs().sum() > 0:
                        df_dispatch['LossOfLoad'] = lol

        # --- Validate and order columns ---
        positive_cols = []
        negative_cols = []

        for col in df_dispatch.columns:
            if col in LINE_COLUMNS:
                continue
            series = df_dispatch[col]
            has_pos = (series > 0).any()
            has_neg = (series < 0).any()
            if has_pos and has_neg:
                # Split mixed-sign columns
                pos_part = series.clip(lower=0)
                neg_part = series.clip(upper=0)
                if pos_part.abs().sum() > 0:
                    df_dispatch[f"{col}_pos"] = pos_part
                    positive_cols.append(f"{col}_pos")
                if neg_part.abs().sum() > 0:
                    df_dispatch[f"{col}_neg"] = neg_part
                    negative_cols.append(f"{col}_neg")
                df_dispatch = df_dispatch.drop(columns=[col])
            elif has_neg:
                negative_cols.append(col)
            elif has_pos:
                positive_cols.append(col)

        # Sort by std dev
        if positive_cols:
            col_std = {c: df_dispatch[c].std() for c in positive_cols}
            positive_cols.sort(key=lambda c: col_std.get(c, 0))
        if negative_cols:
            col_std = {c: df_dispatch[c].abs().std() for c in negative_cols}
            negative_cols.sort(key=lambda c: col_std.get(c, 0))

        # Move LossOfLoad to end of positive (top of stack)
        if 'LossOfLoad' in positive_cols:
            positive_cols.remove('LossOfLoad')
            positive_cols.append('LossOfLoad')

        ordered_cols = negative_cols + positive_cols
        if ordered_cols:
            df_dispatch = df_dispatch[ordered_cols]

        # --- Get demand from node_inflow__dt ---
        inflow_series = None
        node_inflow = combined_dfs.get('node_inflow__dt')
        if node_inflow is not None and not node_inflow.empty:
            if isinstance(node_inflow.columns, pd.MultiIndex):
                if scenario in node_inflow.columns.get_level_values('scenario'):
                    inflow_data = node_inflow.xs(scenario, axis=1, level='scenario')
                    if node in inflow_data.columns:
                        inflow_series = -inflow_data[node].groupby('time').sum()
            else:
                if node in node_inflow.columns:
                    inflow_series = -node_inflow[node].groupby('time').sum()

        return df_dispatch, inflow_series

    except Exception as e:
        print(f"Error preparing node dispatch data for {node} in {scenario}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def create_dispatch_plots(combined_dfs, combined_mapping_dfs, config, plot_dir,
                          scenarios=None, show_plot=False, write_xlsx=False):
    """
    Create dispatch plots for all configured nodeGroups and nodes.

    Parameters:
    -----------
    combined_dfs : dict
        Combined result dataframes
    combined_mapping_dfs : dict
        Combined dispatch mapping dataframes (scenario in index)
    config : dict
        Dispatch plot configuration (new format with inline colors)
    plot_dir : str or Path
        Directory to save plots
    scenarios : list, optional
        List of scenarios to plot. If None, uses config['scenarios']
    show_plot : bool
        Whether to display plots
    write_xlsx : bool
        Whether to write dispatch data to Excel
    """
    plot_dir = Path(plot_dir)

    if scenarios is None:
        scenarios = get_scenarios_from_config(config)

    # Merge colors from inline positive/negative sections
    colors = {}
    for section_key in ['positive', 'negative']:
        section = config.get(section_key, {})
        for cat in ['processGroups', 'processes_not_aggregated']:
            cat_dict = section.get(cat, {})
            if isinstance(cat_dict, dict):
                colors.update(cat_dict)

    # Fallback to special colors for any missing
    for col, color in DEFAULT_SPECIAL_COLORS.items():
        if col not in colors:
            colors[col] = color

    timeline = (
        config.get('time_to_plot', {}).get('first_timestep', 0),
        config.get('time_to_plot', {}).get('first_timestep', 0) +
        config.get('time_to_plot', {}).get('number_of_timesteps', 168)
    )

    # Get nodeGroups from data (dispatch_groups mapping), not config
    dispatch_groups_df = combined_mapping_dfs.get('dispatch_groups')
    node_groups = []
    if dispatch_groups_df is not None and not dispatch_groups_df.empty:
        node_groups = list(dispatch_groups_df['group'].unique())

    excel_data = {}

    for scenario in scenarios:
        print(f"Creating dispatch plots for scenario: {scenario}")

        # Plot nodeGroup dispatches
        for ng in node_groups:
            df_dispatch, inflow = prepare_dispatch_data(
                combined_dfs, combined_mapping_dfs, scenario, ng,
                colors=colors
            )

            if df_dispatch is not None and not df_dispatch.empty:
                output_path = plot_dir / f"dispatch_nodeGroup_{ng}_{scenario}.png"
                plot_dispatch_area(
                    df_dispatch, inflow, output_path,
                    title=f"{ng} - {scenario}",
                    colors=colors,
                    timeline=timeline,
                    show_plot=show_plot
                )

                if write_xlsx:
                    excel_data[f"{ng}_{scenario}"] = df_dispatch
            else:
                print(f"  No dispatch data for nodeGroup {ng}")

        # Plot individual node dispatches
        nodes = config.get('nodes', [])
        for node in nodes:
            df_node, inflow_node = prepare_node_dispatch_data(
                combined_dfs, scenario, node
            )
            if df_node is not None and not df_node.empty:
                node_colors = _auto_assign_node_colors(df_node.columns)
                output_path = plot_dir / f"dispatch_node_{node}_{scenario}.png"
                plot_dispatch_area(
                    df_node, inflow_node, output_path,
                    title=f"{node} - {scenario}",
                    colors=node_colors,
                    timeline=timeline,
                    show_plot=show_plot
                )
                if write_xlsx:
                    excel_data[f"node_{node}_{scenario}"] = df_node
            else:
                print(f"  No dispatch data for node {node}")

    # Write Excel file
    if write_xlsx and excel_data:
        excel_path = plot_dir / "dispatch_data.xlsx"
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            for name, df in excel_data.items():
                sheet_name = name[:31]  # Excel sheet name limit
                df.to_excel(writer, sheet_name=sheet_name)
        print(f"Wrote dispatch data to {excel_path}")


def create_summary_plots(combined_dfs, group_node_df, config, plot_dir, scenarios=None, show_plot=False):
    """
    Create summary bar chart plots comparing scenarios.

    Parameters:
    -----------
    combined_dfs : dict
        Combined result dataframes
    group_node_df : pd.DataFrame
        Node-to-group mapping
    config : dict
        Plot configuration
    plot_dir : str or Path
        Directory to save plots
    scenarios : list, optional
        List of scenarios to include
    show_plot : bool
        Whether to display plots
    """
    plot_dir = Path(plot_dir)

    if scenarios is None:
        scenarios = get_scenarios_from_config(config)

    nodes = config.get('nodes', [])

    # Reorder dataframes to match scenario order
    def reindex_scenarios(df, scen_list):
        if df is None or df.empty:
            return df
        try:
            available = [s for s in scen_list if s in df.columns.get_level_values('scenario')]
            if available:
                return df.reindex(available, axis=1, level='scenario')
        except (KeyError, ValueError):
            pass
        return df

    # 1. Generation by type (if nodeGroup_flows_d_gpe available)
    if 'nodeGroup_flows_d_gpe' in combined_dfs:
        try:
            df = combined_dfs['nodeGroup_flows_d_gpe'].copy()
            # Filter to electricity/main group and sum by type
            df_type_TWh = df.stack('item', future_stack=True).groupby('item').sum().T.groupby('scenario').sum().T.div(1000000)
            df_type_TWh = reindex_scenarios(df_type_TWh, scenarios)

            if not df_type_TWh.empty:
                plot_horizontal_bar(
                    df_type_TWh,
                    filename=str(plot_dir / 'generation_by_type.png'),
                    title='Generation by type',
                    figsize=(6, 6),
                    xlabel="TWh",
                    show_plot=show_plot
                )
        except Exception as e:
            print(f"Could not create generation by type plot: {e}")

    # 2. Loss of load plots
    if 'node_slack_up_dt_e' in combined_dfs and group_node_df is not None:
        try:
            df_lol = combined_dfs['node_slack_up_dt_e'].copy()
            df_lol = reindex_scenarios(df_lol, scenarios)

            # Get group_node mapping for first available scenario
            first_scen = scenarios[0] if scenarios else None
            if first_scen:
                group_node = get_group_node_multiindex(group_node_df, first_scen)
                if group_node is not None:
                    # Filter to nodes in nodeGroups
                    valid_nodes = group_node.get_level_values('node')
                    df_lol_filtered = df_lol.loc[:, df_lol.columns.get_level_values('node').isin(valid_nodes)]

                    if not df_lol_filtered.empty:
                        # LoL per nodeGroup
                        df_lol_filtered.columns = df_lol_filtered.columns.join(group_node)
                        df_lol_sum = df_lol_filtered.groupby('period').sum().T.groupby(['group', 'scenario']).sum().stack().unstack('scenario')
                        df_lol_sum = reindex_scenarios(df_lol_sum, scenarios)

                        plot_horizontal_bar(
                            df_lol_sum.div(1000000),
                            filename=str(plot_dir / 'lol_TWh_nodeGroups.png'),
                            title='Loss of load by nodeGroup',
                            figsize=(5, 4),
                            sum_index_level=0,
                            xlabel="TWh",
                            show_plot=show_plot
                        )
        except Exception as e:
            print(f"Could not create loss of load plots: {e}")

    # 3. VRE share plots
    if 'nodeGroup_gd_p' in combined_dfs:
        try:
            df_vre = combined_dfs['nodeGroup_gd_p'].copy()
            if 'vre_share' in df_vre.columns.get_level_values(1):
                df_vre_share = df_vre.xs('vre_share', axis=1, level=1).groupby('group').sum()
                df_vre_share = reindex_scenarios(df_vre_share, scenarios)

                if not df_vre_share.empty:
                    plot_horizontal_bar(
                        df_vre_share * 100,
                        filename=str(plot_dir / 'vre_share_nodeGroups.png'),
                        title='VRE share by nodeGroup',
                        figsize=(5, 4),
                        xlabel="%",
                        show_plot=show_plot
                    )
        except Exception as e:
            print(f"Could not create VRE share plots: {e}")

    # 4. Curtailment plots
    if 'unit_curtailment_outputNode_dt_ee' in combined_dfs:
        try:
            df_curtail = combined_dfs['unit_curtailment_outputNode_dt_ee'].copy()
            df_curtail = reindex_scenarios(df_curtail, scenarios)

            # Curtailment per node
            curtail_by_node = df_curtail.sum(axis=0).groupby(['node', 'scenario']).sum().unstack('scenario')

            # Filter to configured nodes
            if nodes:
                curtail_by_node = curtail_by_node.loc[curtail_by_node.index.isin(nodes)]

            if not curtail_by_node.empty:
                plot_horizontal_bar(
                    curtail_by_node.div(1000000),
                    filename=str(plot_dir / 'curtailment_TWh_nodes.png'),
                    title='Curtailment by node',
                    figsize=(5, 4),
                    xlabel="TWh",
                    show_plot=show_plot
                )
        except Exception as e:
            print(f"Could not create curtailment plots: {e}")

    print(f"Summary plots saved to {plot_dir}")


def plot_horizontal_bar(df, filename=None, title=None, figsize=(10, 6), show_plot=False, subplot=None, stacked=None, sum_index_level=None, n_subplot_cols=1, xlabel=None, ylabel=None):    
    if sum_index_level is not None:
        df = df.groupby(level=sum_index_level).sum()
    n_subplots = 1
    subplot_names = ['']
    if subplot is not None:
        subplot_names = df.columns.get_level_values(level=subplot).unique()
        n_subplots = len(subplot_names)
    n_subplot_rows = math.ceil(n_subplots / n_subplot_cols)
    fig, axes = plt.subplots(nrows=n_subplot_rows, ncols=n_subplot_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, subplot_name in enumerate(subplot_names):
        if isinstance(df.columns, pd.MultiIndex):
            df_sub = df.xs(subplot_name, axis=1, level=subplot)
        else:
            df_sub = df
        ax = axes[i]
        _ = df_sub.plot.barh(ax=ax, legend=False, title=subplot_name, xlabel=xlabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 1), loc='upper left')

    if title:
        fig.suptitle(title, fontweight='bold')
    plt.tight_layout()
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    return ax

def stacked_and_grouped_bar_plot(df, output_dir, filename=None, title=None, ylabel=None, xlabel=None, show_plot=False):
    """
    Create a stacked and grouped bar plot.

    The dataframe should be pre-shaped so that:
    - Index (can be MultiIndex): dimensions to stack within each bar
    - Columns (can be MultiIndex): dimensions that create groups and bars within groups
      - If MultiIndex columns: first level creates groups, remaining levels create bars within groups

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe in the correct shape for plotting
    output_dir : str or Path
        Directory to save the SVG file (relative to current working directory)
    filename : str
        Name of the output file (without extension, .svg will be added)
    title : str, optional
        Plot title
    ylabel : str, optional
        Y-axis label
    xlabel : str, optional
        X-axis label
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        # First level creates groups (x-axis positions)
        groups = df.columns.get_level_values(0).unique()
    else:
        # Single level columns - each column is a separate group
        groups = df.columns

    n_stack = len(df.index)

    # Colors for stacking dimension
    colors = plt.cm.tab20(np.linspace(0, 1, n_stack))

    # Create list of all column labels for sequential bar positioning
    all_columns = []
    if isinstance(df.columns, pd.MultiIndex):
        for group in groups:
            group_data = df[group]
            if isinstance(group_data, pd.Series):
                all_columns.append(group)
            else:
                all_columns.extend([(group, col) for col in group_data.columns])
    else:
        all_columns = list(df.columns)

    # Plot bars sequentially
    for bar_idx, col_key in enumerate(all_columns):
        # Stack the index dimensions
        bottom = 0
        for stack_idx, stack_val in enumerate(df.index):
            value = df.loc[stack_val, col_key]

            ax.bar(bar_idx,
                   value,
                   bottom=bottom,
                   label=f'{stack_val}' if bar_idx == 0 else '',
                   color=colors[stack_idx])
            bottom += value

    # Customize the plot
    ax.set_ylabel(ylabel if ylabel else 'Value')
    ax.set_title(title if title else 'Stacked and Grouped Bar Plot')

    # Set up two-level x-axis labels
    if isinstance(df.columns, pd.MultiIndex):
        # Extract scenario labels from all_columns
        scenario_labels = [col[1] if isinstance(col, tuple) else col for col in all_columns]

        # Calculate group centers
        group_centers = []
        group_lefts = []
        bar_idx = 0
        for group in groups:
            group_data = df[group]
            if isinstance(group_data, pd.Series):
                n_bars_in_group = 1
            else:
                n_bars_in_group = len(group_data.columns)
            # Center of group is at midpoint of its bars
            group_center = bar_idx + (n_bars_in_group - 1) / 2
            group_centers.append(group_center)
            group_lefts.append(bar_idx - 0.5)
            bar_idx += n_bars_in_group
        group_lefts.append(bar_idx - 0.5)

        # Set main x-axis for scenarios (individual bars)
        ax.set_xticks(range(len(all_columns)), labels=scenario_labels)
        # ax.xaxis.set_major_locator(mticker.MultipleLocator(base=1))
        ax.tick_params('x', length=0)

        # Set plot limits
        ax.set_xlim(-0.5, len(all_columns) - 0.5)

        # Add x-axis tick-based separators
        scen_sep_ax  = ax.secondary_xaxis(location=0)
        scen_sep_ax.set_xticks([x - 0.5 for x in range(len(all_columns) + 1)], [''] * (len(all_columns) + 1))
        scen_sep_ax.tick_params('x', length=15)

        # Add secondary x-axis for groups
        group_ax = ax.secondary_xaxis(location=0)
        group_ax.set_xticks(group_centers, labels=groups)
        group_ax.tick_params('x', length=0, pad=20)

        # Set x-axis label on group axis
        if xlabel:
            group_ax.set_xlabel(xlabel)

        group_sep_ax  = ax.secondary_xaxis(location=0)
        group_sep_ax.set_xticks(group_lefts, [''] * (len(groups) + 1))
        group_sep_ax.tick_params('x', length=30)
    else:
        ax.set_xticks(range(len(all_columns)))
        ax.set_xticklabels(all_columns)
        ax.set_xlabel(xlabel if xlabel else 'Groups')

    # Reverse legend order to match stacking order (bottom to top)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title='Period', bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()

    # Save to SVG
    output_path = Path(os.getcwd()) / output_dir / f"{filename}.svg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if filename:
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    if show_plot:
        plt.show()
    plt.close()


def stacked_and_grouped_barh_plot(df, output_dir, filename=None, title=None, xlabel=None, ylabel=None, show_plot=False):
    """
    Create a horizontal stacked and grouped bar plot.

    The dataframe should be pre-shaped so that:
    - Index (can be MultiIndex): dimensions to stack within each bar
    - Columns (can be MultiIndex): dimensions that create groups and bars within groups
      - If MultiIndex columns: first level creates groups, remaining levels create bars within groups

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe in the correct shape for plotting
    output_dir : str or Path
        Directory to save the SVG file (relative to current working directory)
    filename : str
        Name of the output file (without extension, .svg will be added)
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    """

    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        # First level creates groups (y-axis positions)
        groups = df.columns.get_level_values(0).unique()
    else:
        # Single level columns - each column is a separate group
        groups = df.columns

    n_stack = len(df.index)

    # Colors for stacking dimension
    colors = plt.cm.tab20(np.linspace(0, 1, n_stack))

    # Create list of all column labels for sequential bar positioning
    all_columns = []
    if isinstance(df.columns, pd.MultiIndex):
        for group in groups:
            group_data = df[group]
            if isinstance(group_data, pd.Series):
                all_columns.append(group)
            else:
                all_columns.extend([(group, col) for col in group_data.columns])
    else:
        all_columns = list(df.columns)

    fig, ax = plt.subplots(figsize=(14, (0.2 * len(all_columns) + 0.08)))

    # Plot bars sequentially (horizontal)
    for bar_idx, col_key in enumerate(all_columns):
        # Stack the index dimensions
        left = 0
        for stack_idx, stack_val in enumerate(df.index):
            value = df.loc[stack_val, col_key]

            ax.barh(bar_idx,
                    value,
                    left=left,
                    label=f'{stack_val}' if bar_idx == 0 else '',
                    color=colors[stack_idx])
            left += value

    # Customize the plot
    ax.set_xlabel(xlabel if xlabel else 'Value')
    ax.set_title(title if title else 'Stacked and Grouped Bar Plot')

    # Set up two-level y-axis labels
    if isinstance(df.columns, pd.MultiIndex):
        # Extract scenario labels from all_columns
        scenario_labels = [col[1] if isinstance(col, tuple) else col for col in all_columns]

        # Calculate group centers
        group_centers = []
        group_lefts = []
        bar_idx = 0
        for group in groups:
            group_data = df[group]
            if isinstance(group_data, pd.Series):
                n_bars_in_group = 1
            else:
                n_bars_in_group = len(group_data.columns)
            # Center of group is at midpoint of its bars
            group_center = bar_idx + (n_bars_in_group - 1) / 2
            group_centers.append(group_center)
            group_lefts.append(bar_idx - 0.5)
            bar_idx += n_bars_in_group
        group_lefts.append(bar_idx - 0.5)

        # Set main y-axis for scenarios (individual bars)
        ax.set_yticks(range(len(all_columns)), labels=scenario_labels)
        ax.tick_params('y', length=0)

        # Set plot limits
        ax.set_ylim(-0.5, len(all_columns) - 0.5)

        # Calculate padding based on maximum length of scenario labels
        # Get the longest scenario label text
        max_label_length_scens = max(len(str(label)) for label in scenario_labels)
        max_label_length_groups = max(len(str(label)) for label in groups)

        # Estimate padding: roughly 6 points per character
        pad_value_scens = max_label_length_scens * 5.8
        pad_value_groups = pad_value_scens + max_label_length_groups * 5.8

        # Add y-axis tick-based separators
        scen_sep_ax = ax.secondary_yaxis(location=0)
        scen_sep_ax.set_yticks([x - 0.5 for x in range(len(all_columns) + 1)], [''] * (len(all_columns) + 1))
        scen_sep_ax.tick_params('y', length=pad_value_scens)

        # Add secondary y-axis for groups
        group_ax = ax.secondary_yaxis(location=0)
        group_ax.set_yticks(group_centers, labels=groups)
        group_ax.tick_params('y', length=0 , pad=pad_value_scens + 3) # , labelrotation=90
        # group_ax.set_position([0.0, 0.11, 0.12, 0.88])

        # Set y-axis label on group axis
        if ylabel:
            group_ax.set_ylabel(ylabel)

        # Separators for groups
        group_sep_ax = ax.secondary_yaxis(location=0)
        group_sep_ax.set_yticks(group_lefts, [''] * (len(groups) + 1))
        group_sep_ax.tick_params('y', length=pad_value_groups)
        #group_sep_ax.set_position([0.0, 0.11, 0.12, 0.88])

    else:
        ax.set_yticks(range(len(all_columns)))
        ax.set_yticklabels(all_columns)
        ax.set_ylabel(ylabel if ylabel else 'Groups')

    # Reverse legend order to match stacking order (left to right)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title='Period', bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()

    # Save to SVG
    output_path = Path(os.getcwd()) / output_dir / f"{filename}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if filename:
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    if show_plot:
        plt.show()
    plt.close()


def read_scenario_folders(db_url):
    """
    Read the scenario database to get all folder paths.

    Parameters:
    -----------
    db_url : str
        Database URL containing scenario information

    Returns:
    --------
    dict : Dictionary mapping scenario names to folder paths
    """
    scenario_folders = {}

    with DatabaseMapping(db_url) as db_map:
        filter_configs = db_map.get_filter_configs()
        scenarios = []
        if filter_configs:
            alternative_names = filter_configs[0]['alternatives']
            scenarios = alternative_names

        for scenario_name in scenarios:

            # Get the folder parameter value for this scenario
            param_values = db_map.get_parameter_value_items(
                entity_class_name="scenario",
                entity_name=scenario_name,
                parameter_definition_name="output_location"
            )

            if param_values:
                # Get the first (and should be only) folder parameter value
                folder_path = param_values[0]["parsed_value"]
                scenario_folders[scenario_name] = folder_path

    return scenario_folders


def collect_parquet_files(scenario_folders, output_subdir="output_parquet"):
    """
    Collect all parquet files from all scenario folders.

    Parameters:
    -----------
    scenario_folders : dict
        Dictionary mapping scenario names to folder paths
    output_subdir : str
        Subdirectory within each folder containing parquet files

    Returns:
    --------
    dict : Dictionary mapping filename to list of (scenario_name, file_path) tuples
    """
    # Dictionary to store {filename: [(scenario_name, file_path), ...]}
    files_by_name = {}

    for scenario_name, folder_path in scenario_folders.items():
        parquet_dir = Path(folder_path) / output_subdir / scenario_name

        if not parquet_dir.exists():
            print(f"Warning: {parquet_dir} does not exist for scenario {scenario_name}")
            continue

        # Get all parquet files in this directory
        for parquet_file in sorted(parquet_dir.glob("*.parquet")):
            filename = parquet_file.name

            if filename not in files_by_name:
                files_by_name[filename] = []

            files_by_name[filename].append((scenario_name, parquet_file))

    return files_by_name


def combine_parquet_files(files_by_name):
    """
    Combine parquet files across scenarios into dataframes.

    The parquet files are expected to have scenario information in a multi-index
    column level (highest level), so they are appended along the column axis.

    Parameters:
    -----------
    files_by_name : dict
        Dictionary mapping filename to list of (scenario_name, file_path) tuples

    Returns:
    --------
    dict : Dictionary mapping filename (without extension) to combined dataframe
    """
    combined_dfs = {}

    for filename, scenario_files in files_by_name.items():
        # Remove .parquet extension for the variable name
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
            # Append dataframes along columns (axis=1) since scenario is in column multi-index
            combined_df = pd.concat(dfs_to_append, axis=1)
            combined_dfs[variable_name] = combined_df
            print(f"Combined {len(dfs_to_append)} files for variable '{variable_name}' "
                  f"(shape: {combined_df.shape})")
        else:
            print(f"Warning: No valid data found for {filename}")

    return combined_dfs

def get_scenario_results(db_url, parquet_subdir='output_parquet'):
    # Read scenario folders from database
    print(f"Reading scenario information from {db_url}...")
    scenario_folders = read_scenario_folders(db_url)
    print(f"Found {len(scenario_folders)} scenarios: {list(scenario_folders.keys())}")

    # Collect all parquet files
    print(f"\nCollecting parquet files from {parquet_subdir} subdirectories...")
    files_by_name = collect_parquet_files(scenario_folders, parquet_subdir)
    print(f"Found {len(files_by_name)} unique result variables")

    # Combine parquet files
    print("\nCombining parquet files...")
    combined_dfs = combine_parquet_files(files_by_name)

    print(f"Combined {len(combined_dfs)} result variables.")

    # Return the combined dataframes for use in interactive mode
    return scenario_folders, combined_dfs


if __name__ == '__main__':
    import argparse
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(
        description='Read and combine scenario results from multiple folders based on database information')
    parser.add_argument(
        'db_url',
        help='Database URL containing scenario information (e.g., sqlite:///scenarios.db)')
    parser.add_argument(
        '--parquet-subdir',
        default='output_parquet',
        help='Subdirectory containing parquet files (default: output_parquet)')
    parser.add_argument(
        '--output-config-path', default='templates/default_comparison_plots.yaml'
    )
    parser.add_argument('--active-configs', type=str, default='default', nargs="+",
                        help='Which plot configurations from config_path yaml to use. Defaults to default')
    parser.add_argument('--plot-rows', type=int, nargs=2, default=[0, 167],
                        help='First and last row to plot in time series (default: 0 167)')
    parser.add_argument('--write-to-xlsx', action='store_true',
                        help='Write combined results to Excel file')
    parser.add_argument('--write-dispatch-xlsx', action='store_true',
                        help='Write dispatch data to Excel file in plot directory')
    parser.add_argument('--write-to-ods', action='store_true')
    parser.add_argument(
        '--alternatives', metavar='S', type=str, nargs='+',
        help='Add alternative names manually')
    parser.add_argument(
        '--plot-dir', default='output_plot_comparisons',
        help='Directory to plot scenario comparison plots'
    )
    parser.add_argument(
        '--dispatch-plots', action='store_true',
        help='Generate dispatch area plots for nodes and nodeGroups'
    )
    parser.add_argument(
        '--summary-plots', action='store_true',
        help='Generate summary bar chart plots'
    )
    parser.add_argument(
        '--all-plots', action='store_true',
        help='Generate all plot types (dispatch and summary)'
    )
    parser.add_argument(
        '--show-plots', action='store_true',
        help='Display plots interactively (in addition to saving)'
    )

    args = parser.parse_args()
    db_url = args.db_url
    alternatives = args.alternatives
    plot_dir = args.plot_dir
    active_configs = args.active_configs
    plot_rows = args.plot_rows

    if alternatives:
        alternative_filter = alternative_filter_config(alternatives)
        db_url = append_filter_config(db_url, alternative_filter)

    with open(args.output_config_path, 'r') as f:
        settings = yaml.safe_load(f)

    scenario_folders, combined_dfs = get_scenario_results(db_url=db_url, parquet_subdir=args.parquet_subdir)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Get list of scenarios
    scenarios = list(scenario_folders.keys())

    # Load and combine dispatch mappings across all scenarios
    combined_mapping_dfs = combine_dispatch_mappings(scenario_folders, args.parquet_subdir) if scenario_folders else {}

    # Derive group_node_df for summary plots (needs 'scenario' as column)
    group_node_df = None
    group_node_combined = combined_mapping_dfs.get('group_node')
    if group_node_combined is not None and not group_node_combined.empty:
        group_node_df = group_node_combined.reset_index()

    # Create or update dispatch config
    dispatch_config = None
    if args.dispatch_plots or args.summary_plots or args.all_plots:
        dispatch_config = create_or_update_dispatch_config(
            plot_dir, combined_dfs, scenarios, combined_mapping_dfs
        )

    # Generate original comparison plots (from default_comparison_plots.yaml)
    plot_dict_of_dataframes(combined_dfs, plot_dir, settings['plots'], active_settings=active_configs, plot_rows=plot_rows, delete_existing_plots=True)
    print(f'\nPlotted comparison of {len(scenario_folders)} scenarios to folder: {plot_dir}')

    # Generate dispatch plots
    if args.dispatch_plots or args.all_plots:
        if dispatch_config and combined_mapping_dfs:
            print("\nGenerating dispatch plots...")
            create_dispatch_plots(
                combined_dfs, combined_mapping_dfs, dispatch_config, plot_dir,
                scenarios=get_scenarios_from_config(dispatch_config),
                show_plot=args.show_plots,
                write_xlsx=args.write_dispatch_xlsx
            )
        else:
            print("Warning: Cannot generate dispatch plots - missing dispatch mappings")

    # Generate summary plots
    if args.summary_plots or args.all_plots:
        if dispatch_config:
            print("\nGenerating summary plots...")
            create_summary_plots(
                combined_dfs, group_node_df, dispatch_config, plot_dir,
                scenarios=get_scenarios_from_config(dispatch_config),
                show_plot=args.show_plots
            )

    # Write to excel (combined results)
    if args.write_to_xlsx:
        excel_dir = 'output_excel_comparison'
        os.makedirs(excel_dir, exist_ok=True)
        filename = 'compare_' + str(len(scenario_folders)) + '_scens.xlsx'
        excel_path = os.path.join(excel_dir, filename)
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            used_names = set()
            for name, df in combined_dfs.items():
                if (not df.empty) & (len(df) > 0):
                    # Excel sheet names limited to 31 characters
                    sheet_name = name[:31]
                    # Handle duplicates from truncation
                    if sheet_name in used_names:
                        suffix = 1
                        while f"{sheet_name[:28]}_{suffix}" in used_names:
                            suffix += 1
                        sheet_name = f"{sheet_name[:28]}_{suffix}"
                    used_names.add(sheet_name)
                    df.to_excel(writer, sheet_name=sheet_name)

        print(f'\nWrote comparison of {len(scenario_folders)} scenarios to xlsx file: {excel_path}')

    print('\nDone!')