"""Build and update dispatch configuration from data.

Reads parquet-based results and dispatch mappings to generate
a config.yaml with scenario colors, positive/negative process groups,
and node selections.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from flextool.scenario_comparison.constants import (
    DEFAULT_SPECIAL_COLORS, POSITIVE_SPECIAL, NEGATIVE_SPECIAL, LINE_COLUMNS,
)
from flextool.scenario_comparison.config_io import (
    parse_config_with_comments, write_config_with_comments,
)
from flextool.scenario_comparison.data_models import TimeSeriesResults, DispatchMappings


def get_scenarios_from_config(config: dict) -> list[str]:
    """Extract active scenario names from config."""
    scenarios = config.get('scenarios', {})
    if isinstance(scenarios, dict):
        return list(scenarios.keys())
    return []


def compute_process_group_std_order(
    results: TimeSeriesResults,
    mappings: DispatchMappings,
    scenarios: list[str],
    available_process_groups: set[str],
) -> list[str]:
    """Compute standard deviation for each processGroup and return them ordered by std dev.

    Parameters
    ----------
    results : TimeSeriesResults
        Combined time-series result DataFrames
    mappings : DispatchMappings
        Combined dispatch mapping DataFrames
    scenarios : list[str]
        List of scenario names
    available_process_groups : set[str]
        Set of available processGroup names

    Returns
    -------
    list[str]
        processGroups ordered by std dev (lowest first)
    """
    if not scenarios or not available_process_groups:
        return sorted(available_process_groups)

    # Use the first scenario to compute std dev
    first_scenario = scenarios[0]

    # Compute std dev for each processGroup by checking actual data
    pg_std: dict[str, float] = {}

    for pg in available_process_groups:
        std_sum = 0.0
        count = 0

        # Check unit_outputNode_dt_ee for unit-based processGroups
        if results.unit_outputNode_dt_ee is not None:
            unit_output = results.unit_outputNode_dt_ee
            if first_scenario in unit_output.columns.get_level_values('scenario'):
                # Get members for this processGroup
                unit_members_all = mappings.processGroup_unit_to_node_members
                if unit_members_all is not None and not unit_members_all.empty:
                    unit_members = mappings.get_for_scenario(
                        'processGroup_unit_to_node_members', first_scenario
                    )
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
        for conn_df in [results.connection_leftward_dt_eee, results.connection_rightward_dt_eee]:
            if conn_df is not None:
                if first_scenario in conn_df.columns.get_level_values('scenario'):
                    conn_members = mappings.get_for_scenario(
                        'processGroup_connection_to_node_members', first_scenario
                    )
                    if conn_members is not None and not conn_members.empty:
                        members = conn_members[conn_members['group_aggregate'] == pg]
                        if not members.empty:
                            data = conn_df.xs(first_scenario, axis=1, level='scenario')
                            for _, row in members.iterrows():
                                col_key = (row['process'], row['node'])
                                if col_key in data.columns:
                                    std_sum += data[col_key].abs().std()
                                    count += 1

        pg_std[pg] = std_sum / count if count > 0 else float('inf')

    # Sort by std dev (lowest first)
    return sorted(available_process_groups, key=lambda pg: pg_std.get(pg, float('inf')))


def create_or_update_dispatch_config(
    plot_dir: Path,
    results: TimeSeriesResults,
    scenarios: list[str],
    mappings: DispatchMappings,
) -> dict:
    """Create or update the dispatch plot configuration file.

    New config format: scenarios as dict name->color, positive/negative with inline
    colors, no separate nodeGroups or colors sections.

    Parameters
    ----------
    plot_dir : Path
        Directory for plots (config will be stored here)
    results : TimeSeriesResults
        Combined time-series result DataFrames
    scenarios : list[str]
        List of scenario names
    mappings : DispatchMappings
        Combined dispatch mapping DataFrames

    Returns
    -------
    dict
        The configuration dictionary
    """
    from flextool.scenario_comparison.dispatch_data import prepare_dispatch_data

    config_path = Path(plot_dir) / 'config.yaml'

    # Get available data from parquet
    available_nodes: set[str] = set()
    if results.node_d_ep is not None:
        node_cols = results.node_d_ep.columns
        if isinstance(node_cols, pd.MultiIndex):
            available_nodes = set(node_cols.get_level_values('node').unique())

    # Collect available nodeGroups and processGroups
    available_node_groups: set[str] = set()
    available_process_groups: set[str] = set()
    available_processes_not_aggregated: set[str] = set()

    dispatch_groups_df = mappings.dispatch_groups
    if dispatch_groups_df is not None and not dispatch_groups_df.empty:
        available_node_groups.update(dispatch_groups_df['group'].unique())

    for pg_attr in ['processGroup_Unit_to_group', 'processGroup_Group_to_unit', 'processGroup_Connection']:
        pg_df = getattr(mappings, pg_attr, None)
        if pg_df is not None and not pg_df.empty:
            available_process_groups.update(pg_df['group_aggregate'].unique())

    for na_attr in ['not_in_aggregate_unit_to_node', 'not_in_aggregate_node_to_unit',
                    'not_in_aggregate_connection_to_node', 'not_in_aggregate_node_to_connection']:
        na_df = getattr(mappings, na_attr, None)
        if na_df is not None and not na_df.empty:
            for _, row in na_df.iterrows():
                if 'process' in row and 'node' in row:
                    available_processes_not_aggregated.add(f"({row['process']}, {row['node']})")

    na_conn_df = mappings.not_in_aggregate_connection
    if na_conn_df is not None and not na_conn_df.empty:
        for _, row in na_conn_df.iterrows():
            available_processes_not_aggregated.add(f"({row['connection']})")

    available_scenarios = set(scenarios)

    # Parse existing config
    existing_config, commented_entries = parse_config_with_comments(config_path)

    # --- Build scenarios dict ---
    new_config: dict = {}
    new_config['time_to_plot'] = existing_config.get('time_to_plot', {
        'first_timestep': 0,
        'number_of_timesteps': 168
    })

    scenario_colormap = plt.cm.tab10(np.linspace(0, 1, 10))

    existing_scenarios = existing_config.get('scenarios', {}) or {}
    if isinstance(existing_scenarios, dict):
        existing_active_scens = existing_scenarios
    else:
        existing_active_scens = {}

    existing_commented_scens = commented_entries.get('scenarios', {})
    if not isinstance(existing_commented_scens, dict):
        existing_commented_scens = {}

    active_scenarios: dict[str, str] = {}
    commented_scens: dict[str, str] = {}

    # Existing active -> keep if still available
    for name, color in existing_active_scens.items():
        if name in available_scenarios:
            active_scenarios[name] = color or ''

    # Existing commented -> keep if still available
    for name, color in existing_commented_scens.items():
        if name in available_scenarios and name not in active_scenarios:
            commented_scens[name] = color or ''

    # New scenarios not in config -> add as active
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

    # --- Fix positive/negative categorization across ALL nodeGroups ---
    positive_groups: set[str] = set()
    negative_groups: set[str] = set()
    positive_processes: set[str] = set()
    negative_processes: set[str] = set()

    active_scenario_names = list(active_scenarios.keys())

    if active_scenario_names and available_node_groups:
        for scenario in active_scenario_names:
            for node_group in sorted(available_node_groups):
                df_sample, _ = prepare_dispatch_data(
                    results, mappings, scenario, node_group
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

    # --- Assign colors inline ---
    # Collect existing inline colors from config (new format)
    existing_pos = existing_config.get('positive', {}) or {}
    existing_neg = existing_config.get('negative', {}) or {}
    existing_inline_colors: dict[str, str] = {}
    for section in [existing_pos, existing_neg]:
        for cat in ['processGroups', 'processes_not_aggregated']:
            cat_dict = section.get(cat)
            if isinstance(cat_dict, dict):
                existing_inline_colors.update(cat_dict)
    all_existing_colors = existing_inline_colors

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
        results, mappings, active_scenario_names, pos_regular
    )
    ordered_pos_groups = pos_special + pos_regular_ordered

    # Order negative processGroups: Charge, Export first, regular, then internal_losses
    neg_special_top = [c for c in ['Charge', 'Export'] if c in negative_groups]
    neg_regular = negative_groups - set(POSITIVE_SPECIAL) - set(NEGATIVE_SPECIAL)
    neg_regular_ordered = compute_process_group_std_order(
        results, mappings, active_scenario_names, neg_regular
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

    # --- Build nodes section ---
    existing_nodes = list(existing_config.get('nodes', []) or [])
    existing_commented_nodes = commented_entries.get('nodes', set())
    if not isinstance(existing_commented_nodes, set):
        existing_commented_nodes = set()

    if existing_nodes or existing_commented_nodes:
        # Preserve existing active/commented state
        active_nodes = [n for n in existing_nodes if n in available_nodes]
        commented_nodes = {n for n in existing_commented_nodes if n in available_nodes}
        # New nodes not seen before -> add to commented
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
