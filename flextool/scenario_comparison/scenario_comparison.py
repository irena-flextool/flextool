import pandas as pd
import os
import math
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from flextool.scenario_comparison.constants import (
    DEFAULT_SPECIAL_COLORS, POSITIVE_SPECIAL, NEGATIVE_SPECIAL, LINE_COLUMNS,
)



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

    # Save (bbox_inches='tight' in savefig handles layout)
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

        conn_left = None
        conn_right = None
        if connection_df is not None and not connection_df.empty:
            group_aggregates = connection_df[connection_df['group'] == output_node_group]['group_aggregate'].unique()

            # Load connection flow data
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

    # First pass: collect y-axis ranges across all scenarios for consistent scales
    def _compute_ylim(df: pd.DataFrame, timeline: tuple) -> tuple[float, float]:
        """Compute stacked area y-axis range from a dispatch DataFrame."""
        plot_cols = [col for col in df.columns if col != 'Curtailed']
        df_slice = df[plot_cols].iloc[timeline[0]:timeline[1]]
        pos_sum = df_slice.clip(lower=0).sum(axis=1).max()
        neg_sum = df_slice.clip(upper=0).sum(axis=1).min()
        return (neg_sum, pos_sum)

    ng_ylims: dict[str, tuple[float, float]] = {}
    node_ylims: dict[str, tuple[float, float]] = {}
    ng_columns: dict[str, list[str]] = {}
    node_columns: dict[str, list[str]] = {}
    nodes = config.get('nodes', [])

    for scenario in scenarios:
        for ng in node_groups:
            df_dispatch, _ = prepare_dispatch_data(
                combined_dfs, combined_mapping_dfs, scenario, ng,
                colors=colors
            )
            if df_dispatch is not None and not df_dispatch.empty:
                ymin, ymax = _compute_ylim(df_dispatch, timeline)
                if ng in ng_ylims:
                    ng_ylims[ng] = (min(ng_ylims[ng][0], ymin), max(ng_ylims[ng][1], ymax))
                    # Add any new columns preserving existing order
                    for col in df_dispatch.columns:
                        if col not in ng_columns[ng]:
                            ng_columns[ng].append(col)
                else:
                    ng_ylims[ng] = (ymin, ymax)
                    ng_columns[ng] = list(df_dispatch.columns)

        for node in nodes:
            df_node, _ = prepare_node_dispatch_data(
                combined_dfs, scenario, node
            )
            if df_node is not None and not df_node.empty:
                ymin, ymax = _compute_ylim(df_node, timeline)
                if node in node_ylims:
                    node_ylims[node] = (min(node_ylims[node][0], ymin), max(node_ylims[node][1], ymax))
                    for col in df_node.columns:
                        if col not in node_columns[node]:
                            node_columns[node].append(col)
                else:
                    node_ylims[node] = (ymin, ymax)
                    node_columns[node] = list(df_node.columns)

    # Add small margin to y-axis limits
    for key, (ymin, ymax) in ng_ylims.items():
        margin = (ymax - ymin) * 0.05
        ng_ylims[key] = (ymin - margin, ymax + margin)
    for key, (ymin, ymax) in node_ylims.items():
        margin = (ymax - ymin) * 0.05
        node_ylims[key] = (ymin - margin, ymax + margin)

    for scenario in scenarios:
        print(f"Creating dispatch plots for scenario: {scenario}")

        # Plot nodeGroup dispatches
        for ng in node_groups:
            df_dispatch, inflow = prepare_dispatch_data(
                combined_dfs, combined_mapping_dfs, scenario, ng,
                colors=colors
            )

            if df_dispatch is not None and not df_dispatch.empty:
                # Ensure consistent columns across scenarios for same nodeGroup
                if ng in ng_columns:
                    for col in ng_columns[ng]:
                        if col not in df_dispatch.columns:
                            df_dispatch[col] = 0.0
                    df_dispatch = df_dispatch[ng_columns[ng]]
                output_path = plot_dir / f"dispatch_nodeGroup_{ng}_{scenario}.png"
                plot_dispatch_area(
                    df_dispatch, inflow, output_path,
                    title=f"{ng} - {scenario}",
                    colors=colors,
                    timeline=timeline,
                    show_plot=show_plot,
                    ylim=ng_ylims.get(ng)
                )

                if write_xlsx:
                    excel_data[f"{ng}_{scenario}"] = df_dispatch
            else:
                print(f"  No dispatch data for nodeGroup {ng}")

        # Plot individual node dispatches
        for node in nodes:
            df_node, inflow_node = prepare_node_dispatch_data(
                combined_dfs, scenario, node
            )
            if df_node is not None and not df_node.empty:
                # Ensure consistent columns across scenarios for same node
                if node in node_columns:
                    for col in node_columns[node]:
                        if col not in df_node.columns:
                            df_node[col] = 0.0
                    df_node = df_node[node_columns[node]]
                node_colors = _auto_assign_node_colors(df_node.columns)
                output_path = plot_dir / f"dispatch_node_{node}_{scenario}.png"
                plot_dispatch_area(
                    df_node, inflow_node, output_path,
                    title=f"{node} - {scenario}",
                    colors=node_colors,
                    timeline=timeline,
                    show_plot=show_plot,
                    ylim=node_ylims.get(node)
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


def create_basic_plots(combined_dfs, group_node_df, config, plot_dir, scenarios=None, show_plot=False):
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
                        # LoL per nodeGroup — stack to long format and merge with group mapping
                        # (avoids index join which breaks when nodes belong to multiple groups)
                        df_summed = df_lol_filtered.groupby('period').sum()
                        df_long = df_summed.stack(list(range(len(df_summed.columns.names))), future_stack=True).rename('value').reset_index()
                        node_col = next(c for c in df_long.columns if c not in ('period', 'scenario', 'value'))
                        gn = group_node_df[group_node_df['scenario'] == first_scen][['group', 'node']].drop_duplicates()
                        df_merged = df_long.merge(gn, left_on=node_col, right_on='node', how='inner')
                        df_lol_sum = df_merged.groupby(['group', 'period', 'scenario'])['value'].sum().unstack('scenario')
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

            # Curtailment per node (column level may be 'sink' or 'node')
            node_level = 'sink' if 'sink' in df_curtail.columns.names else 'node'
            curtail_by_node = df_curtail.sum(axis=0).groupby([node_level, 'scenario']).sum().unstack('scenario')

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


def plot_horizontal_bar(df, filename=None, title=None, figsize=(10, 6), show_plot=False, subplot=None, stacked=None, sum_index_level=None, n_subplot_cols=1, xlabel=None, ylabel=None, max_items: int = 20):
    if sum_index_level is not None:
        df = df.groupby(level=sum_index_level).sum()

    # Split into multiple files if too many items (rows)
    all_items = df.index.tolist()
    n_items = len(all_items)
    needs_split = n_items > max_items

    if needs_split:
        chunks = [all_items[i:i + max_items] for i in range(0, n_items, max_items)]
    else:
        chunks = [all_items]

    for chunk_idx, item_chunk in enumerate(chunks, start=1):
        df_chunk = df.loc[item_chunk]

        # Scale figsize height proportionally to number of items in this chunk
        chunk_figsize = (figsize[0], figsize[1] * len(item_chunk) / min(n_items, max_items)) if figsize else figsize

        n_subplots = 1
        subplot_names = ['']
        if subplot is not None:
            subplot_names = df_chunk.columns.get_level_values(level=subplot).unique()
            n_subplots = len(subplot_names)
        n_subplot_rows = math.ceil(n_subplots / n_subplot_cols)
        fig, axes = plt.subplots(nrows=n_subplot_rows, ncols=n_subplot_cols, figsize=chunk_figsize, squeeze=False)
        axes = axes.flatten()

        for i, subplot_name in enumerate(subplot_names):
            if isinstance(df_chunk.columns, pd.MultiIndex):
                df_sub = df_chunk.xs(subplot_name, axis=1, level=subplot)
            else:
                df_sub = df_chunk
            ax = axes[i]
            _ = df_sub.plot.barh(ax=ax, legend=False, title=subplot_name, xlabel=xlabel)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 1), loc='upper left')

        chunk_title = f"{title} ({chunk_idx}/{len(chunks)})" if title and needs_split else title
        if chunk_title:
            fig.suptitle(chunk_title, fontweight='bold')
        plt.tight_layout()
        if filename:
            if needs_split:
                base, ext = os.path.splitext(filename)
                chunk_filename = f"{base}_{chunk_idx}{ext}"
            else:
                chunk_filename = filename
            plt.savefig(chunk_filename, bbox_inches='tight')
        if show_plot:
            plt.show()
        plt.close()
    return ax


from flextool.scenario_comparison.db_reader import (  # noqa: E402
    read_scenario_folders,
    collect_parquet_files,
    combine_parquet_files,
    get_scenario_results,
)

from flextool.scenario_comparison.dispatch_mappings import (  # noqa: E402
    load_dispatch_mappings,
    combine_dispatch_mappings,
    get_group_node_multiindex,
)

from flextool.scenario_comparison.config_builder import (  # noqa: E402
    get_scenarios_from_config,
    compute_process_group_std_order,
    create_or_update_dispatch_config,
)
