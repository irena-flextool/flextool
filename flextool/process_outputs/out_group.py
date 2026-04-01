"""Output functions for node group results."""
import pandas as pd


def _get_group_nodes_with_inflow(s, g: str) -> list[str]:
    """Return nodes in group g that have an inflow method (not 'no_inflow')."""
    group_nodes = (
        s.group_node[s.group_node.get_level_values('group') == g]
        .get_level_values('node')
        .tolist()
    )
    if hasattr(s, 'node__inflow_method'):
        no_inflow_nodes = {n for (n, method) in s.node__inflow_method if method == 'no_inflow'}
        return [n for n in group_nodes if n not in no_inflow_nodes]
    return group_nodes


def nodeGroup_indicators(par, s, v, r, debug):
    """Node group indicator results by period and time"""

    results = []

    if not list(s.outputNodeGroup_does_specified_flows_node):
        return results

    # Get time steps
    dt_index = s.dt_realize_dispatch  # Should be MultiIndex with (period, time)

    # Calculate timestep-level results first
    results_dt = []

    for g in s.outputNodeGroup_does_specified_flows_node:
        group_nodes = (
            s.group_node[s.group_node.get_level_values('group').isin([g])]
            .get_level_values('node')
            .tolist()
        )
        group_nodes_with_inflow = _get_group_nodes_with_inflow(s, g)

        # 1. pdtNodeInflow (negative sum)
        if group_nodes_with_inflow:
            group_inflow = -par.node_inflow[group_nodes_with_inflow].sum(axis=1)
        else:
            group_inflow = pd.Series(0, index=dt_index)

        # 2. Sum of annualized inflows [MWh]
        annualized_inflow = group_inflow.div(par.complete_period_share_of_year)

        # 3. VRE share (actual flow)
        vre_processes = s.process_VRE.get_level_values('process')
        vre_cols = r.flow_dt.columns[
            r.flow_dt.columns.get_level_values('sink').isin(group_nodes) &
            r.flow_dt.columns.get_level_values('process').isin(vre_processes) &
            r.flow_dt.columns.isin(s.process_source_sink_alwaysProcess)
        ]
        vre_flow_sum = r.flow_dt[vre_cols].sum(axis=1)

        # VRE share calculation (avoid division by zero)
        vre_share = vre_flow_sum / group_inflow.where(group_inflow != 0, pd.NA)

        # 4. Curtailed VRE share
        potential_cols = r.potentialVREgen_dt.columns[
            r.potentialVREgen_dt.columns.get_level_values(1).isin(group_nodes) &
            r.potentialVREgen_dt.columns.get_level_values(0).isin(vre_processes)
        ]
        potential_sum = r.potentialVREgen_dt[potential_cols].sum(axis=1)
        curtailed_vre = (potential_sum - vre_flow_sum).clip(lower=0)
        curtailed_vre_share = curtailed_vre / group_inflow.where(group_inflow != 0, pd.NA)
        curtailed_vre_of_potential_vre = curtailed_vre / potential_sum

        # Filter balance nodes directly from the sets
        balance_set = set(s.node_balance) | set(s.node_balance_period)
        balance_nodes = [n for n in group_nodes if n in balance_set]

        # 5. Upward slack
        if balance_nodes and not v.q_state_up.empty:
            upward_slack = v.q_state_up.mul(par.node_capacity_for_scaling[v.q_state_up.columns]).sum(axis=1).clip(lower=0)
        else:
            upward_slack = pd.Series(0, index=dt_index)

        # 6. Downward slack
        if balance_nodes and not v.q_state_down.empty:
            downward_slack = v.q_state_down.mul(par.node_capacity_for_scaling[v.q_state_down.columns]).sum(axis=1).clip(lower=0)
        else:
            downward_slack = pd.Series(0, index=dt_index)

        # Combine timestep results for this group
        group_result_dt = pd.DataFrame({
            'group': g,
            'period': dt_index.get_level_values('period'),
            'time': dt_index.get_level_values('time'),
            '1. Loss of load': upward_slack.fillna(0).values,
            '2. VRE generation': vre_flow_sum.values,
            '3. Excess load': downward_slack.fillna(0).values,
            '4. Curtailed VRE': curtailed_vre.values,
            '5. Timestep inflow': group_inflow.values,
            '6. Curtailed VRE of potential VRE': curtailed_vre_share.fillna(0).values,
            '7. Annualized inflow': annualized_inflow.values,
            '8. VRE share of demand': vre_share.fillna(0).values,
        })
        results_dt.append(group_result_dt)

    # Combine all groups for timestep level
    result_flat_dt = pd.concat(results_dt, ignore_index=True)

    # Create multi-index version for timestep level
    results_dt_indexed = result_flat_dt.set_index(['group', 'period', 'time'])[
        ['1. Loss of load', '2. VRE generation', '3. Excess load', '4. Curtailed VRE',
         '5. Timestep inflow', '6. Curtailed VRE of potential VRE', '7. Annualized inflow', '8. VRE share of demand']
    ]
    results_dt_indexed.columns.name = "parameter"

    results.append((results_dt_indexed, 'nodeGroup_gdt_p'))

    # Aggregate to period level
    # For shares, we need to recalculate properly:
    # Sum the numerators and denominators separately, then divide
    inflow_d = results_dt_indexed['5. Timestep inflow'].groupby(level=['group', 'period']).sum()
    annualized_inflow_d = results_dt_indexed['7. Annualized inflow'].groupby(level=['group', 'period']).sum()

    # For VRE share: need to sum absolute flows and recalculate
    # Since we have vre_share * inflow = vre_flow, we can recover vre_flow
    vre_flow_dt = results_dt_indexed['8. VRE share of demand'] * results_dt_indexed['5. Timestep inflow']
    vre_flow_d = vre_flow_dt.groupby(level=['group', 'period']).sum()
    vre_share_d = vre_flow_d / inflow_d.where(inflow_d != 0, pd.NA)

    # For curtailed VRE share: similar recovery
    curtailed_vre_share_dt = results_dt_indexed['4. Curtailed VRE'] / results_dt_indexed['5. Timestep inflow']
    curtailed_vre_share_d = curtailed_vre_share_dt.groupby(level=['group', 'period']).mean()
    curtailed_vre_of_potential_vre_d = results_dt_indexed['6. Curtailed VRE of potential VRE'].groupby(level=['group', 'period']).sum()

    # Slack shares: sum absolute values and recalculate
    upward_slack_d = results_dt_indexed['1. Loss of load'].groupby(['group', 'period']).sum()
    upward_slack_share_d = upward_slack_d / inflow_d

    downward_slack_d = results_dt_indexed['3. Excess load'].groupby(['group', 'period']).sum()
    downward_slack_share_d = downward_slack_d / inflow_d

    # Combine period-level results
    results_d_indexed = pd.DataFrame({
        '1. Loss of load share': upward_slack_share_d.fillna(0),
        '2. VRE share of demand': vre_share_d.fillna(0),
        '3. Excess load share': downward_slack_share_d.fillna(0),
        '4. Curtailed VRE of demand': curtailed_vre_share_d.fillna(0),
        '5. Annualized inflow': annualized_inflow_d,
        '6. Curtailed VRE of potential VRE': curtailed_vre_of_potential_vre_d.fillna(0)
    })
    results_d_indexed.columns.name = "parameter"

    results.append((results_d_indexed, 'nodeGroup_gd_p'))

    return results


def nodeGroup_VRE_share(par, s, v, r, debug):
    """VRE share for node groups by period and time"""

    results = []

    # Get timesteps and groups
    timesteps = list(s.dt_realize_dispatch)

    # Filter groups that have nodes with inflow
    groups_with_inflow = [
        g for g in s.outputNodeGroup_does_specified_flows_node
        if _get_group_nodes_with_inflow(s, g)
    ]

    if not groups_with_inflow or not timesteps:
        index = pd.MultiIndex.from_tuples([], names=['period', 'time'])
        results_dt = pd.DataFrame(index=index, columns=groups_with_inflow)
        return [(results_dt, 'nodeGroup_VRE_share_dt_g')]

    # Create index
    index = pd.MultiIndex.from_tuples(timesteps, names=['period', 'time'])
    results_dt = pd.DataFrame(index=index, columns=groups_with_inflow, dtype=float)

    # Get VRE processes
    vre_processes = s.process_VRE.unique()

    # Calculate for each group
    for g in groups_with_inflow:
        group_nodes = _get_group_nodes_with_inflow(s, g)
        node_cols = [col for col in par.node_inflow.columns.get_level_values(0) if col in group_nodes]

        # Total inflow to group nodes (vectorized)
        total_inflow = par.node_inflow[node_cols].sum(axis=1)

        # VRE flow: sum flows from VRE processes to group nodes (vectorized)
        vre_cols = [(p, src, snk) for (p, src, snk) in r.flow_dt.columns
                    if p in vre_processes and (p, snk) in s.process_sink and snk in group_nodes]

        if vre_cols:
            vre_flow = r.flow_dt[vre_cols].sum(axis=1)
        else:
            vre_flow = pd.Series(0.0, index=r.flow_dt.index)

        # Calculate share (avoid division by zero)
        results_dt[g] = (vre_flow / (-total_inflow)).fillna(0.0)

    results.append((results_dt, 'nodeGroup_VRE_share_dt_g'))

    # Return period results
    results_d = results_dt.groupby(level='period').mean()
    results_d.columns.name = 'group'
    results.append((results_d, 'nodeGroup_VRE_share_d_g'))

    return results


def nodeGroup_total_inflow(par, s, v, r, debug):
    """Total inflow (inflow - outflow) to groups by period and time"""

    results = []
    groups = list(s.outputNodeGroup_does_specified_flows_process)

    if not groups:
        return results

    timesteps = list(s.dt_realize_dispatch)
    if not timesteps:
        return results

    # Calculate timestep-level results first
    index = pd.MultiIndex.from_tuples(timesteps, names=['period', 'time'])
    result_multi_dt = pd.DataFrame(index=index, columns=groups, dtype=float)

    # Calculate for each group
    for g in groups:
        # Flows into nodes (process -> node sink)
        # Filter for (g, process, node) in group_process_node where node is the sink
        sink_cols = r.flow_dt.columns[
            r.flow_dt.columns.isin(s.process_source_sink_alwaysProcess) &
            r.flow_dt.columns.to_series().apply(
                lambda col: (g, col[0], col[2]) in s.group_process_node
            )
        ]

        # Flows from nodes (node source -> process)
        # Filter for (g, process, node) in group_process_node where node is the source
        source_cols = r.flow_dt.columns[
            r.flow_dt.columns.isin(s.process_source_sink_alwaysProcess) &
            r.flow_dt.columns.to_series().apply(
                lambda col: (g, col[0], col[1]) in s.group_process_node
            )
        ]

        inflow = r.flow_dt[sink_cols].sum(axis=1) if len(sink_cols) > 0 else 0
        outflow = r.flow_dt[source_cols].sum(axis=1) if len(source_cols) > 0 else 0

        result_multi_dt[g] = inflow - outflow
    result_multi_dt.columns.name = 'group'

    # Return timestep results
    results.append((result_multi_dt, 'nodeGroup_flows_dt_g'))

    # Aggregate to period level
    result_multi_d = result_multi_dt.groupby(level='period').sum()

    # Divide by period shares to annualize
    result_multi_d = result_multi_d.div(par.complete_period_share_of_year, axis=0)
    result_multi_d.columns.name = 'group'

    # Return period results
    results.append((result_multi_d, 'nodeGroup_flows_d_g'))

    return results


def nodeGroup_flows(par, s, v, r, debug):
    """Group output flows for periods and time"""

    results = []

    if s.outputNodeGroup_does_generic_flows.empty or s.dt_realize_dispatch.empty:
        return results

    # Calculate timestep-level results first
    result_multi_dt = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['group', 'type', 'item']))

    # Assign simple mappings for all groups at once (before group loop)
    # Slack upward
    r.group_node_up_slack__dt.columns = pd.MultiIndex.from_arrays([
        r.group_node_up_slack__dt.columns,
        ['slack'] * len(r.group_node_up_slack__dt.columns),
        ['upward'] * len(r.group_node_up_slack__dt.columns)
    ], names=['group', 'type', 'item'])
    result_multi_dt[r.group_node_up_slack__dt.columns] = r.group_node_up_slack__dt

    # Unit aggregates (aggregateUnits to group)
    r.group_output__group_aggregate_Unit_to_group__dt.columns = pd.MultiIndex.from_arrays([
        r.group_output__group_aggregate_Unit_to_group__dt.columns.get_level_values('group'),
        ['from_unitGroup'] * len(r.group_output__group_aggregate_Unit_to_group__dt.columns),
        r.group_output__group_aggregate_Unit_to_group__dt.columns.get_level_values(1)
    ], names=['group', 'type', 'item'])
    result_multi_dt[r.group_output__group_aggregate_Unit_to_group__dt.columns] = r.group_output__group_aggregate_Unit_to_group__dt

    # Units not in aggregate (unit to group) - sum across nodes
    r.group_output__unit_to_node_not_in_aggregate__dt.columns = pd.MultiIndex.from_arrays([
        r.group_output__unit_to_node_not_in_aggregate__dt.columns.get_level_values('group'),
        ['from_unit'] * len(r.group_output__unit_to_node_not_in_aggregate__dt.columns),
        r.group_output__unit_to_node_not_in_aggregate__dt.columns.get_level_values('process')
    ], names=['group', 'type', 'item'])
    result_multi_dt[r.group_output__unit_to_node_not_in_aggregate__dt.columns] = r.group_output__unit_to_node_not_in_aggregate__dt

    # Connection aggregates (from connections to group) - sum across nodes
    r.group_output__from_connection_aggregate__dt.columns = pd.MultiIndex.from_arrays([
        r.group_output__from_connection_aggregate__dt.columns.get_level_values('group'),
        ['from_connectionGroup'] * len(r.group_output__from_connection_aggregate__dt.columns),
        r.group_output__from_connection_aggregate__dt.columns.get_level_values(1)
    ], names=['group', 'type', 'item'])
    result_multi_dt[r.group_output__from_connection_aggregate__dt.columns] = r.group_output__from_connection_aggregate__dt

    # Connections not in aggregate (from connections)
    r.group_output__from_connection_not_in_aggregate__dt.columns = pd.MultiIndex.from_arrays([
        r.group_output__from_connection_not_in_aggregate__dt.columns.get_level_values('group'),
        ['from_connection'] * len(r.group_output__from_connection_not_in_aggregate__dt.columns),
        r.group_output__from_connection_not_in_aggregate__dt.columns.get_level_values('process')
    ], names=['group', 'type', 'item'])
    result_multi_dt[r.group_output__from_connection_not_in_aggregate__dt.columns] = r.group_output__from_connection_not_in_aggregate__dt

    # Connections not in aggregate (to connections)
    r.group_output__to_connection_not_in_aggregate__dt.columns = pd.MultiIndex.from_arrays([
        r.group_output__to_connection_not_in_aggregate__dt.columns.get_level_values('group'),
        ['to_connection'] * len(r.group_output__to_connection_not_in_aggregate__dt.columns),
        r.group_output__to_connection_not_in_aggregate__dt.columns.get_level_values('process')
    ], names=['group', 'type', 'item'])
    result_multi_dt[r.group_output__to_connection_not_in_aggregate__dt.columns] = -r.group_output__to_connection_not_in_aggregate__dt

    # Connection aggregates (to connections) - sum across nodes
    r.group_output__to_connection_aggregate__dt.columns = pd.MultiIndex.from_arrays([
        r.group_output__to_connection_aggregate__dt.columns.get_level_values('group'),
        ['to_connectionGroup'] * len(r.group_output__to_connection_aggregate__dt.columns),
        r.group_output__to_connection_aggregate__dt.columns.get_level_values(1)
    ], names=['group', 'type', 'item'])
    result_multi_dt[r.group_output__to_connection_aggregate__dt.columns] = -r.group_output__to_connection_aggregate__dt

    # Group to aggregate units (negative)
    r.group_output__group_aggregate_Group_to_unit__dt.columns = pd.MultiIndex.from_arrays([
        r.group_output__group_aggregate_Group_to_unit__dt.columns.get_level_values('group'),
        ['to_unitGroup'] * len(r.group_output__group_aggregate_Group_to_unit__dt.columns),
        r.group_output__group_aggregate_Group_to_unit__dt.columns.get_level_values(1)
    ], names=['group', 'type', 'item'])
    result_multi_dt[r.group_output__group_aggregate_Group_to_unit__dt.columns] = -r.group_output__group_aggregate_Group_to_unit__dt

    # Node to unit not in aggregate (negative)
    r.group_output__node_to_unit_not_in_aggregate__dt.columns = pd.MultiIndex.from_arrays([
        r.group_output__node_to_unit_not_in_aggregate__dt.columns.get_level_values('group'),
        ['to_unit'] * len(r.group_output__node_to_unit_not_in_aggregate__dt.columns),
        r.group_output__node_to_unit_not_in_aggregate__dt.columns.get_level_values('process')
    ], names=['group', 'type', 'item'])
    result_multi_dt[r.group_output__node_to_unit_not_in_aggregate__dt.columns] = -r.group_output__node_to_unit_not_in_aggregate__dt

    # Inflow
    r.group_node_inflow_dt.columns = pd.MultiIndex.from_arrays([
        r.group_node_inflow_dt.columns,
        ['inflow'] * len(r.group_node_inflow_dt.columns),
        r.group_node_inflow_dt.columns
    ], names=['group', 'type', 'item'])
    result_multi_dt[r.group_node_inflow_dt.columns] = r.group_node_inflow_dt

    # Internal losses - connections (sum across processes, negate)
    if not r.group_output_Internal_connection_losses__dt.empty:
        r.group_output_Internal_connection_losses__dt = r.group_output_Internal_connection_losses__dt.T.groupby('group').sum().T
        r.group_output_Internal_connection_losses__dt.columns = pd.MultiIndex.from_arrays([
            r.group_output_Internal_connection_losses__dt.columns,
            ['internal_losses'] * len(r.group_output_Internal_connection_losses__dt.columns),
            ['connections'] * len(r.group_output_Internal_connection_losses__dt.columns)
        ], names=['group', 'type', 'item'])
        result_multi_dt[r.group_output_Internal_connection_losses__dt.columns] = r.group_output_Internal_connection_losses__dt

    # Internal losses - units (sum across processes, negate)
    if not r.group_output_Internal_unit_losses__dt.empty:
        r.group_output_Internal_unit_losses__dt = r.group_output_Internal_unit_losses__dt.T.groupby('group').sum().T
        r.group_output_Internal_unit_losses__dt.columns = pd.MultiIndex.from_arrays([
            r.group_output_Internal_unit_losses__dt.columns,
            ['internal_losses'] * len(r.group_output_Internal_unit_losses__dt.columns),
            ['units'] * len(r.group_output_Internal_unit_losses__dt.columns)
        ], names=['group', 'type', 'item'])
        result_multi_dt[r.group_output_Internal_unit_losses__dt.columns] = r.group_output_Internal_unit_losses__dt

    # Internal losses - storages (negate)
    r.group_node_state_losses__dt.columns = pd.MultiIndex.from_arrays([
        r.group_node_state_losses__dt.columns,
        ['internal_losses'] * len(r.group_node_state_losses__dt.columns),
        ['storages'] * len(r.group_node_state_losses__dt.columns)
    ], names=['group', 'type', 'item'])
    result_multi_dt[r.group_node_state_losses__dt.columns] = r.group_node_state_losses__dt

    # Slack downward
    r.group_node_down_slack__dt.columns = pd.MultiIndex.from_arrays([
        r.group_node_down_slack__dt.columns,
        ['slack'] * len(r.group_node_down_slack__dt.columns),
        ['downward'] * len(r.group_node_down_slack__dt.columns)
    ], names=['group', 'type', 'item'])
    result_multi_dt[r.group_node_down_slack__dt.columns] = r.group_node_down_slack__dt

    result_multi_dt.columns.names = ['group', 'type', 'item']
    result_multi_dt = result_multi_dt.sort_index(axis=1, level='group', sort_remaining=False)

    # Return timestep results
    results.append((result_multi_dt, 'nodeGroup_flows_dt_gpe'))

    # Aggregate to period level
    result_multi_d = result_multi_dt.groupby(level='period').sum()

    # Return period results
    results.append((result_multi_d, 'nodeGroup_flows_d_gpe'))

    return results
