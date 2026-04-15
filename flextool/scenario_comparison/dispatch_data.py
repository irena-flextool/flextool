"""Prepare per-scenario dispatch DataFrames for plotting.

Public API:
- prepare_dispatch_data      — group-level dispatch (uses DispatchMappings)
- prepare_node_dispatch_data — single-node dispatch

Internal helpers:
- _get_time_index         — find time index from results (S5)
- _slice_scenario_df      — extract scenario columns and sum over time (S4)
- _order_dispatch_columns — validate signs, sort by std dev, reorder (S6)
"""

from __future__ import annotations

import pandas as pd

from flextool.scenario_comparison.constants import (
    LINE_COLUMNS,
    NEGATIVE_SPECIAL,
    POSITIVE_SPECIAL,
)
from flextool.scenario_comparison.data_models import (
    DispatchMappings,
    TimeSeriesResults,
)


# ---------------------------------------------------------------------------
# Shared helpers (S4, S5, S6)
# ---------------------------------------------------------------------------

def _get_time_index(results: TimeSeriesResults, scenario: str) -> pd.Index | None:
    """Find a time index from the first available result DataFrame.

    Tries unit flows and connection flows first, then falls back to
    slack and inflow data so that nodeGroups with only demand / loss-of-load
    still get a valid time index.
    """
    candidates = [
        results.unit_outputNode_dt_ee,
        results.connection_leftward_dt_eee,
        results.node_slack_up_dt_e,
        results.node_inflow__dt,
    ]
    for df in candidates:
        if df is None:
            continue
        if isinstance(df.columns, pd.MultiIndex) and 'scenario' in df.columns.names:
            if scenario in df.columns.get_level_values('scenario'):
                return df.xs(scenario, axis=1, level='scenario').groupby('time').sum().index
        elif df.index.nlevels >= 2:
            # node_inflow__dt may not have a scenario column level
            return df.groupby('time').sum().index
    return None


def _slice_scenario_df(
    df: pd.DataFrame | None, scenario: str,
) -> pd.DataFrame | None:
    """Extract scenario columns and sum over time.  Returns None if scenario absent."""
    if df is None:
        return None
    if scenario not in df.columns.get_level_values('scenario'):
        return None
    return df.xs(scenario, axis=1, level='scenario').groupby('time').sum()


def _order_dispatch_columns(
    df: pd.DataFrame,
    plot_name: str = "",
    config_order: list[str] | None = None,
) -> pd.DataFrame:
    """Validate signs, sort columns, return reordered DataFrame.

    Mixed-sign columns are split into ``<col>_pos`` / ``<col>_neg`` parts
    so they can be stacked correctly in area plots.

    When *config_order* is provided (from config.yaml positive/negative
    sections), columns are ordered to match the config.  Columns not in
    the config fall back to std-dev sorting.
    """
    positive_cols: list[str] = []
    negative_cols: list[str] = []
    plot_label = f" in '{plot_name}'" if plot_name else ""

    for col in list(df.columns):
        if col in LINE_COLUMNS:
            continue
        series = df[col]
        has_pos = (series > 0).any()
        has_neg = (series < 0).any()

        if has_pos and has_neg:
            print(f"  Note: Column '{col}' has mixed positive/negative values{plot_label}"
                  f" - splitting into {col}_pos and {col}_neg")
            pos_part = series.clip(lower=0)
            neg_part = series.clip(upper=0)
            if pos_part.abs().sum() > 0:
                df[f"{col}_pos"] = pos_part
                positive_cols.append(f"{col}_pos")
            if neg_part.abs().sum() > 0:
                df[f"{col}_neg"] = neg_part
                negative_cols.append(f"{col}_neg")
            df = df.drop(columns=[col])
        elif has_neg:
            negative_cols.append(col)
        elif has_pos:
            positive_cols.append(col)

    # --- Validate special column signs ---
    for col in POSITIVE_SPECIAL:
        if col in negative_cols:
            print(f"  Warning: '{col}' is expected to be positive but has negative values")
    for col in NEGATIVE_SPECIAL:
        if col in positive_cols:
            print(f"  Warning: '{col}' is expected to be negative but has positive values")

    if config_order:
        # Use config order: columns present in config come first (in config order),
        # then any remaining columns sorted by std dev.
        # Also match _pos/_neg split columns to their base name in the config.
        config_set = set(config_order)
        ordered_from_config_neg: list[str] = []
        ordered_from_config_pos: list[str] = []
        remaining_neg: list[str] = []
        remaining_pos: list[str] = []
        for col in negative_cols:
            base = col.removesuffix('_neg') if col.endswith('_neg') else col
            if base in config_set or col in config_set:
                ordered_from_config_neg.append(col)
            else:
                remaining_neg.append(col)
        for col in positive_cols:
            base = col.removesuffix('_pos') if col.endswith('_pos') else col
            if base in config_set or col in config_set:
                ordered_from_config_pos.append(col)
            else:
                remaining_pos.append(col)
        # Sort config-matched columns by their position in config_order
        def _config_key(col):
            base = col.removesuffix('_pos').removesuffix('_neg')
            try:
                return config_order.index(base)
            except ValueError:
                try:
                    return config_order.index(col)
                except ValueError:
                    return len(config_order)
        ordered_from_config_neg.sort(key=_config_key)
        ordered_from_config_pos.sort(key=_config_key)
        # Sort remaining by std dev
        if remaining_neg:
            col_std = {col: df[col].abs().std() for col in remaining_neg}
            remaining_neg.sort(key=lambda c: col_std.get(c, 0))
        if remaining_pos:
            col_std = {col: df[col].std() for col in remaining_pos}
            remaining_pos.sort(key=lambda c: col_std.get(c, 0))
        ordered_cols = ordered_from_config_neg + remaining_neg + ordered_from_config_pos + remaining_pos
    else:
        # Fallback: sort by std dev with special columns in fixed positions
        pos_special = [c for c in positive_cols if c in POSITIVE_SPECIAL]
        pos_regular = [c for c in positive_cols if c not in POSITIVE_SPECIAL]
        neg_special = [c for c in negative_cols if c in NEGATIVE_SPECIAL]
        neg_regular = [c for c in negative_cols if c not in NEGATIVE_SPECIAL]

        if pos_regular:
            col_std = {col: df[col].std() for col in pos_regular}
            pos_regular = sorted(pos_regular, key=lambda c: col_std.get(c, 0))
        if neg_regular:
            col_std = {col: df[col].abs().std() for col in neg_regular}
            neg_regular = sorted(neg_regular, key=lambda c: col_std.get(c, 0))

        ordered_cols: list[str] = []
        ordered_cols.extend(neg_regular)
        for col in ['internal_losses', 'Export', 'Charge']:
            if col in neg_special:
                ordered_cols.append(col)
        ordered_cols.extend(pos_regular)
        for col in ['Import', 'Discharge', 'LossOfLoad']:
            if col in pos_special:
                ordered_cols.append(col)

    # Line overlay
    if 'Curtailed' in df.columns:
        ordered_cols.append('Curtailed')

    if ordered_cols:
        df = df[ordered_cols]

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_dispatch_data(
    results: TimeSeriesResults,
    mappings: DispatchMappings,
    scenario: str,
    output_node_group: str,
    colors: dict | None = None,
    config_order: list[str] | None = None,
) -> tuple[pd.DataFrame | None, pd.Series | None]:
    """Prepare dispatch data for a specific outputNodeGroup.

    Columns are validated for sign consistency:
    - Positive special columns: LossOfLoad, Discharge, Import
    - Negative special columns: Charge, Export, internal_losses
    - Columns with mixed signs are excluded with a warning
    """
    from flextool.scenario_comparison.constants import DEFAULT_SPECIAL_COLORS

    if colors is None:
        colors = DEFAULT_SPECIAL_COLORS

    try:
        # Validate that this group should have dispatch plots
        dispatch_groups_df = mappings.get_for_scenario('dispatch_groups', scenario)
        if dispatch_groups_df is None or dispatch_groups_df.empty:
            return None, None

        if output_node_group not in dispatch_groups_df['group'].values:
            return None, None

        # Get nodes in this group
        group_node_df = mappings.get_for_scenario('group_node', scenario)
        if group_node_df is None or group_node_df.empty:
            return None, None

        nodes_in_group = group_node_df[group_node_df['group'] == output_node_group]['node'].tolist()
        if not nodes_in_group:
            return None, None

        # Initialize dispatch dataframe with time index
        time_index = _get_time_index(results, scenario)
        if time_index is None:
            return None, None

        df_dispatch = pd.DataFrame(index=time_index)

        # --- Process Unit_to_group (unit outputs to nodes in group) ---
        unit_to_group_df = mappings.get_for_scenario('processGroup_Unit_to_group', scenario)
        unit_to_group_members = mappings.get_for_scenario('processGroup_unit_to_node_members', scenario)

        if unit_to_group_df is not None and not unit_to_group_df.empty:
            group_aggregates = unit_to_group_df[unit_to_group_df['group'] == output_node_group]['group_aggregate'].unique()

            unit_output = _slice_scenario_df(results.unit_outputNode_dt_ee, scenario)

            if unit_output is not None:
                for group_agg in group_aggregates:
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
        group_to_unit_df = mappings.get_for_scenario('processGroup_Group_to_unit', scenario)
        group_to_unit_members = mappings.get_for_scenario('processGroup_node_to_unit_members', scenario)

        if group_to_unit_df is not None and not group_to_unit_df.empty:
            group_aggregates = group_to_unit_df[group_to_unit_df['group'] == output_node_group]['group_aggregate'].unique()

            unit_input = _slice_scenario_df(results.unit_inputNode_dt_ee, scenario)

            if unit_input is not None:
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
                                flow_sum = flow_sum.add(unit_input[col_key], fill_value=0)

                        if flow_sum.abs().sum() > 0:
                            # If this group already has a positive (Unit_to_group)
                            # column, keep charge as "Charge" so both discharge
                            # and charge are separately visible in the dispatch.
                            if group_agg in df_dispatch.columns:
                                if 'Charge' in df_dispatch.columns:
                                    df_dispatch['Charge'] = df_dispatch['Charge'].add(flow_sum, fill_value=0)
                                else:
                                    df_dispatch['Charge'] = flow_sum
                            else:
                                df_dispatch[group_agg] = flow_sum

        # --- Process Connection flows ---
        connection_df = mappings.get_for_scenario('processGroup_Connection', scenario)
        conn_to_node_members = mappings.get_for_scenario('processGroup_connection_to_node_members', scenario)
        conn_from_node_members = mappings.get_for_scenario('processGroup_node_to_connection_members', scenario)

        conn_left = None
        conn_right = None
        if connection_df is not None and not connection_df.empty:
            group_aggregates = connection_df[connection_df['group'] == output_node_group]['group_aggregate'].unique()

            conn_left = _slice_scenario_df(results.connection_leftward_dt_eee, scenario)
            conn_right = _slice_scenario_df(results.connection_rightward_dt_eee, scenario)

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
        not_agg_unit_to_node = mappings.get_for_scenario('not_in_aggregate_unit_to_node', scenario)
        if not_agg_unit_to_node is not None and not not_agg_unit_to_node.empty:
            entries = not_agg_unit_to_node[not_agg_unit_to_node['group'] == output_node_group]
            unit_output = _slice_scenario_df(results.unit_outputNode_dt_ee, scenario)

            if unit_output is not None:
                for _, row in entries.iterrows():
                    unit = row['unit']
                    node = row['node']
                    col_key = (unit, node)
                    col_name = f"({row['process']}, {node})"
                    if col_key in unit_output.columns:
                        df_dispatch[col_name] = unit_output[col_key]

        # Node to unit not in aggregate
        not_agg_node_to_unit = mappings.get_for_scenario('not_in_aggregate_node_to_unit', scenario)
        if not_agg_node_to_unit is not None and not not_agg_node_to_unit.empty:
            entries = not_agg_node_to_unit[not_agg_node_to_unit['group'] == output_node_group]
            unit_input = _slice_scenario_df(results.unit_inputNode_dt_ee, scenario)

            if unit_input is not None:
                for _, row in entries.iterrows():
                    unit = row['unit']
                    node = row['node']
                    col_key = (unit, node)
                    col_name = f"({row['process']}, {node})"
                    if col_key in unit_input.columns:
                        df_dispatch[col_name] = -unit_input[col_key]

        # Connection to node not in aggregate
        not_agg_conn_to_node = mappings.get_for_scenario('not_in_aggregate_connection_to_node', scenario)
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
        not_agg_node_to_conn = mappings.get_for_scenario('not_in_aggregate_node_to_connection', scenario)
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
        not_agg_connection = mappings.get_for_scenario('not_in_aggregate_connection', scenario)
        if not_agg_connection is not None and not not_agg_connection.empty:
            entries = not_agg_connection[not_agg_connection['group'] == output_node_group]
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
        process_fully_inside = mappings.get_for_scenario('process_fully_inside', scenario)
        if process_fully_inside is not None and not process_fully_inside.empty:
            entries = process_fully_inside[process_fully_inside['group'] == output_node_group]
            if not entries.empty:
                conn_losses = _slice_scenario_df(results.connection_losses_dt_eee, scenario)

                if conn_losses is not None:
                    loss_sum = pd.Series(0.0, index=time_index)
                    for _, row in entries.iterrows():
                        process = row['process']
                        if process in conn_losses.columns:
                            loss_sum = loss_sum.add(conn_losses[process], fill_value=0)
                        matching_cols = [c for c in conn_losses.columns if (isinstance(c, tuple) and c[0] == process) or c == process]
                        for col in matching_cols:
                            if col != process:
                                loss_sum = loss_sum.add(conn_losses[col], fill_value=0)

                    if loss_sum.abs().sum() > 0:
                        df_dispatch['internal_losses'] = loss_sum

        # --- Get loss of load (slack) ---
        if results.node_slack_up_dt_e is not None:
            slack = results.node_slack_up_dt_e
            if scenario in slack.columns.get_level_values('scenario'):
                slack = slack.xs(scenario, axis=1, level='scenario')
                slack_filtered = slack.loc[:, slack.columns.isin(nodes_in_group)]
                if not slack_filtered.empty:
                    lol = slack_filtered.groupby('time').sum().sum(axis=1)
                    if lol.abs().sum() > 0:
                        df_dispatch['LossOfLoad'] = lol

        # --- Get curtailment ---
        if results.unit_curtailment_outputNode_dt_ee is not None:
            curtail = results.unit_curtailment_outputNode_dt_ee
            if scenario in curtail.columns.get_level_values('scenario'):
                curtail = curtail.xs(scenario, axis=1, level='scenario')
                node_level = 'sink' if 'sink' in curtail.columns.names else 'node'
                curtail_filtered = curtail.loc[:, curtail.columns.get_level_values(node_level).isin(nodes_in_group)]
                if not curtail_filtered.empty:
                    curtailed = curtail_filtered.groupby('time').sum().sum(axis=1).clip(lower=0)
                    if curtailed.abs().sum() > 0:
                        df_dispatch['Curtailed'] = curtailed

        # --- Order columns (S6) ---
        df_dispatch = _order_dispatch_columns(df_dispatch, plot_name=f"{output_node_group} ({scenario})", config_order=config_order)

        # --- Get demand from node_inflow__dt ---
        inflow_series = None
        node_inflow = results.node_inflow__dt

        if node_inflow is not None and not node_inflow.empty:
            if isinstance(node_inflow.columns, pd.MultiIndex):
                if scenario in node_inflow.columns.get_level_values('scenario'):
                    inflow_data = node_inflow.xs(scenario, axis=1, level='scenario')
                    inflow_filtered = inflow_data.loc[:, inflow_data.columns.isin(nodes_in_group)]
                    if not inflow_filtered.empty:
                        inflow_series = -inflow_filtered.groupby('time').sum().sum(axis=1)
            else:
                inflow_filtered = node_inflow.loc[:, node_inflow.columns.isin(nodes_in_group)]
                if not inflow_filtered.empty:
                    inflow_series = -inflow_filtered.groupby('time').sum().sum(axis=1)

        return df_dispatch, inflow_series

    except Exception as e:
        print(f"Error preparing dispatch data for {output_node_group} in {scenario}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def prepare_node_dispatch_data(
    results: TimeSeriesResults,
    scenario: str,
    node: str,
) -> tuple[pd.DataFrame | None, pd.Series | None]:
    """Prepare dispatch data for a single node (not a nodeGroup).

    Collects unit outputs/inputs, connection flows, slack (LossOfLoad),
    and demand for the given node.
    """
    try:
        # Get time index (S5)
        time_index = _get_time_index(results, scenario)
        if time_index is None:
            return None, None

        df_dispatch = pd.DataFrame(index=time_index)

        # --- Unit outputs to this node (positive) ---
        unit_output = _slice_scenario_df(results.unit_outputNode_dt_ee, scenario)
        if unit_output is not None:
            for col in unit_output.columns:
                col_node = col[1] if isinstance(col, tuple) else col
                if col_node == node:
                    unit_name = col[0] if isinstance(col, tuple) else col
                    series = unit_output[col].clip(lower=0)
                    if series.abs().sum() > 0:
                        df_dispatch[f"{unit_name}_out"] = series

        # --- Unit inputs from this node (negative) ---
        unit_input = _slice_scenario_df(results.unit_inputNode_dt_ee, scenario)
        if unit_input is not None:
            for col in unit_input.columns:
                col_node = col[1] if isinstance(col, tuple) else col
                if col_node == node:
                    unit_name = col[0] if isinstance(col, tuple) else col
                    series = -unit_input[col]
                    neg_part = series.clip(upper=0)
                    if neg_part.abs().sum() > 0:
                        df_dispatch[f"{unit_name}_in"] = neg_part

        # --- Connection leftward flows (node receives) ---
        conn_left = _slice_scenario_df(results.connection_leftward_dt_eee, scenario)
        if conn_left is not None:
            for col in conn_left.columns:
                col_node = col[1] if isinstance(col, tuple) else col
                if col_node == node:
                    conn_name = col[0] if isinstance(col, tuple) else col
                    series = conn_left[col]
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
        conn_right = _slice_scenario_df(results.connection_rightward_dt_eee, scenario)
        if conn_right is not None:
            for col in conn_right.columns:
                col_node = col[1] if isinstance(col, tuple) else col
                if col_node == node:
                    conn_name = col[0] if isinstance(col, tuple) else col
                    series = conn_right[col]
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
        if results.node_slack_up_dt_e is not None:
            slack = results.node_slack_up_dt_e
            if scenario in slack.columns.get_level_values('scenario'):
                slack_data = slack.xs(scenario, axis=1, level='scenario')
                if node in slack_data.columns:
                    lol = slack_data[node].groupby('time').sum()
                    if lol.abs().sum() > 0:
                        df_dispatch['LossOfLoad'] = lol

        # --- Order columns (S6) ---
        df_dispatch = _order_dispatch_columns(df_dispatch, plot_name=f"{node} ({scenario})")

        # --- Get demand from node_inflow__dt ---
        inflow_series = None
        node_inflow = results.node_inflow__dt
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
