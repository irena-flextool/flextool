import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import time
import yaml
from datetime import datetime, timezone
from flextool.read_flextool_outputs import read_variables, read_parameters, read_sets
from flextool.process_results import post_process_results
from flextool.plot_functions import plot_dict_of_dataframes
import warnings


def read_outputs(output_dir):
    """
    Read solver output files.

    Args:
    """

    # Read solver output files
    p = read_parameters(output_dir)
    s = read_sets(output_dir)
    v = read_variables(output_dir)

    return p, s, v

def generic(par, s, v, r):
    results = []
    df = pd.concat([par.discount_factor_operations_yearly, par.discount_factor_investment_yearly], axis=1)
    df.columns = ["operations discount factor","investments discount factor"]
    df.columns.name = "param"
    results.append((df, 'discountFactors_d_p'))

    df = par.entity_annuity
    results.append((df, 'entity_annuity_d_p'))

    return results

def unit_capacity(par, s, v, r):
    """Unit capacity by period"""
    
    # Get all periods and filter to process_unit entities
    periods = list(s.d_realize_dispatch_or_invest)
    processes = list(s.process_unit)
    
    # Create base dataframe with all combinations (period, unit order)
    if processes:
        index = pd.MultiIndex.from_product([processes, periods], names=['unit', 'period'])
    else:
        index = pd.Index(periods, name='period')
    results = pd.DataFrame(index=index)
    results.columns.name = 'parameter'

    # Existing capacity - filter to process_unit only
    existing = par.entity_all_existing[processes].unstack()
    results['existing'] = existing

    # Invested capacity - default to None, overwrite if data exists
    results['invested'] = pd.Series(dtype=float)
    if not v.invest.empty and len(v.invest.columns) > 0:
        ed_unit_invest = s.ed_invest[s.ed_invest.get_level_values('entity').isin(s.process_unit)]
        unit_invest = ed_unit_invest.get_level_values('entity').unique()
        results['invested'] = v.invest.unstack()[ed_unit_invest] * par.entity_unitsize[unit_invest]

    # Divested capacity - default to None, overwrite if data exists
    results['divested'] = pd.Series(dtype=float)
    if not v.divest.empty and len(v.divest.columns) > 0:
        ed_unit_divest = s.ed_divest[s.ed_divest.get_level_values('entity').isin(s.process_unit)]
        unit_divest = ed_unit_divest.get_level_values('entity').unique()
        results['divested'] = v.divest.unstack()[ed_unit_divest] * par.entity_unitsize[unit_divest]

    # Total capacity - filter to process_unit only
    total = r.entity_all_capacity[processes].unstack()
    results['total'] = total
    results = results[['existing', 'invested', 'divested', 'total']]
    return results, 'unit_capacity_ed_p'


def connection_capacity(par, s, v, r):
    """Connection capacity by period"""
    
    # Get all periods and filter to process_connection entities
    periods = list(s.d_realize_dispatch_or_invest)
    connections = list(s.process_connection)
    
    # Create base dataframe with all combinations (period, connection order)
    if connections:
        index = pd.MultiIndex.from_product([connections, periods], names=['connection', 'period'])
    else:
        index = pd.Index(periods, name='period')
    results = pd.DataFrame(index=index)
    results.columns.name = 'parameter'
    
    # Existing capacity - filter to process_connection only
    existing = par.entity_all_existing[connections].unstack()
    results['existing'] = existing
    
    # Invested capacity - default to empty, overwrite if data exists
    results['invested'] = pd.Series(dtype=float)
    if not v.invest.empty and len(v.invest.columns) > 0:
        ed_conn_invest = s.ed_invest[s.ed_invest.get_level_values('entity').isin(s.process_connection)]
        conn_invest = ed_conn_invest.get_level_values('entity').unique()
        results['invested'] = v.invest.unstack()[ed_conn_invest] * par.entity_unitsize[conn_invest]
    
    # Divested capacity - default to empty, overwrite if data exists
    results['divested'] = pd.Series(dtype=float)
    if not v.divest.empty and len(v.divest.columns) > 0:
        ed_conn_divest = s.ed_divest[s.ed_divest.get_level_values('entity').isin(s.process_connection)]
        conn_divest = ed_conn_divest.get_level_values('entity').unique()
        results['invested'] = v.divest.unstack()[ed_conn_divest] * par.entity_unitsize[conn_divest]
    
    # Total capacity - filter to process_connection only
    results['total'] = r.entity_all_capacity[connections].unstack()
    
    # Reorder columns
    results = results[['existing', 'invested', 'divested', 'total']]
    
    return results, 'connection_capacity_ed_p'


def node_capacity(par, s, v, r):
    """Node capacity by period"""

    # Get all periods and filter to node_state entities
    periods = list(s.d_realize_dispatch_or_invest)
    nodes = list(s.node_state)

    # Create base dataframe with all combinations (period, node order)
    if nodes:
        index = pd.MultiIndex.from_product([nodes, periods], names=['node', 'period'])
    else:
        index = pd.Index(periods, name='period')
    results = pd.DataFrame(index=index)
    results.columns.name = 'parameter'

    # Existing capacity - filter to node_state only
    if nodes:
        existing = par.entity_all_existing[nodes].unstack()
        results['existing'] = existing
    else:
        results['existing'] = pd.Series(dtype=float)

    # Invested capacity - default to empty, overwrite if data exists
    results['invested'] = pd.Series(dtype=float)
    if not v.invest.empty and len(v.invest.columns) > 0:
        ed_node_invest = s.ed_invest[s.ed_invest.get_level_values('entity').isin(s.node)]
        node_invest = ed_node_invest.get_level_values('entity').unique()
        results['invested'] = v.invest.unstack()[ed_node_invest] * par.entity_unitsize[node_invest]

    # Divested capacity - default to empty, overwrite if data exists
    results['divested'] = pd.Series(dtype=float)
    if not v.divest.empty and len(v.divest.columns) > 0:
        ed_node_divest = s.ed_divest[s.ed_divest.get_level_values('entity').isin(s.node)]
        node_divest = ed_node_divest.get_level_values('entity').unique()
        results['invested'] = v.invest.unstack()[ed_node_divest] * par.entity_unitsize[node_divest]

    # Total capacity - filter to node_state only
    if nodes:
        results['total'] = r.entity_all_capacity[nodes].unstack()
    else:
        results['total'] = pd.Series(dtype=float)

    results = results[['existing', 'invested', 'divested', 'total']]
    return results, 'node_capacity_ed_p'


def nodeGroup_indicators(par, s, v, r):
    """Node group indicator results by period and time"""

    results = []

    if not list(s.groupOutput_node):
        return results

    # Get time steps
    dt_index = s.dt_realize_dispatch  # Should be MultiIndex with (period, time)

    # Calculate timestep-level results first
    results_dt = []

    for g in s.groupOutput_node:
        # Get nodes in this group
        group_nodes = s.group_node[s.group_node.get_level_values('group').isin([g])].get_level_values('node').tolist()

        # Filter out nodes with 'no_inflow' method
        if hasattr(s, 'node__inflow_method'):
            no_inflow_nodes = [n for (n, method) in s.node__inflow_method if method == 'no_inflow']
            group_nodes_with_inflow = [n for n in group_nodes if n not in no_inflow_nodes]
        else:
            group_nodes_with_inflow = group_nodes

        # 1. pdtNodeInflow (negative sum)
        if group_nodes_with_inflow:
            group_inflow = -par.node_inflow[group_nodes_with_inflow].sum(axis=1)
        else:
            group_inflow = pd.Series(0, index=dt_index)

        # 2. Sum of annualized inflows [MWh]
        period_shares = group_inflow.index.get_level_values('period').map(
            lambda p: par.complete_period_share_of_year[p]
        )
        annualized_inflow = -group_inflow / period_shares

        # 3. VRE share (actual flow)
        vre_processes = s.process_VRE.get_level_values('process').unique()
        if len(vre_processes) > 0:
            vre_cols = r.flow_dt.columns[
                r.flow_dt.columns.get_level_values('sink').isin(group_nodes) &
                r.flow_dt.columns.get_level_values('process').isin(vre_processes) &
                r.flow_dt.columns.isin(s.process_source_sink_alwaysProcess)
            ]
            if len(vre_cols) > 0:
                vre_flow_sum = r.flow_dt[vre_cols].sum(axis=1)
            else:
                vre_flow_sum = pd.Series(0, index=dt_index)
        else:
            vre_flow_sum = pd.Series(0, index=dt_index)

        # VRE share calculation (avoid division by zero)
        vre_share = vre_flow_sum / -group_inflow.where(group_inflow != 0, pd.NA)

        # 4. Curtailed VRE share
        potential_cols = r.potentialVREgen_dt.columns[
            r.potentialVREgen_dt.columns.get_level_values(1).isin(group_nodes) &
            r.potentialVREgen_dt.columns.get_level_values(0).isin(vre_processes)
        ]
        if len(potential_cols) > 0:
            potential_sum = r.potentialVREgen_dt[potential_cols].sum(axis=1)
        else:
            potential_sum = pd.Series(0, index=dt_index)
        curtailed_vre = (potential_sum - vre_flow_sum)
        curtailed_vre_share = curtailed_vre / -group_inflow.where(group_inflow != 0, pd.NA)

        # Filter balance nodes directly from the sets
        balance_nodes = s.node_balance.union(s.node_balance_period)
        balance_set = set(s.node_balance) | set(s.node_balance_period)
        balance_nodes = [n for n in group_nodes if n in balance_set]

        # 5. Upward slack
        if balance_nodes and not v.q_state_up.empty:
            upward_slack = v.q_state_up.mul(par.node_capacity_for_scaling[v.q_state_up.columns]).sum(axis=1)
        else:
            upward_slack = pd.Series(0, index=dt_index)

        # 6. Downward slack
        if balance_nodes and not v.q_state_down.empty:
            downward_slack = v.q_state_down.mul(par.node_capacity_for_scaling[v.q_state_down.columns]).sum(axis=1)
        else:
            downward_slack = pd.Series(0, index=dt_index)

        # Combine timestep results for this group
        group_result_dt = pd.DataFrame({
            'group': g,
            'period': dt_index.get_level_values('period'),
            'time': dt_index.get_level_values('time'),
            'pdtNodeInflow': group_inflow.values,
            'annualized_inflows': annualized_inflow.values,
            'vre_share': vre_share.fillna(0).values,
            'curtailed_vre_share': curtailed_vre_share.fillna(0).values,
            'upward_slack': upward_slack.fillna(0).values,
            'downward_slack': downward_slack.fillna(0).values
        })
        results_dt.append(group_result_dt)

    # Combine all groups for timestep level
    result_flat_dt = pd.concat(results_dt, ignore_index=True)

    # Create multi-index version for timestep level
    results_dt_indexed = result_flat_dt.set_index(['group', 'period', 'time'])[
        ['pdtNodeInflow', 'annualized_inflows', 'vre_share', 'curtailed_vre_share',
         'upward_slack', 'downward_slack']
    ]
    results_dt_indexed.columns.name = "parameter"

    results.append((results_dt_indexed, 'nodeGroup_gdt_p'))

    # Aggregate to period level
    # For shares, we need to recalculate properly:
    # Sum the numerators and denominators separately, then divide
    inflow_d = results_dt_indexed['pdtNodeInflow'].groupby(level=['group', 'period']).sum()
    annualized_d = results_dt_indexed['annualized_inflows'].groupby(level=['group', 'period']).sum()

    # For VRE share: need to sum absolute flows and recalculate
    # Since we have vre_share * inflow = vre_flow, we can recover vre_flow
    vre_flow_dt = results_dt_indexed['vre_share'] * results_dt_indexed['pdtNodeInflow']
    vre_flow_d = vre_flow_dt.groupby(level=['group', 'period']).sum()
    vre_share_d = vre_flow_d / inflow_d.where(inflow_d != 0, pd.NA)

    # For curtailed VRE share: similar recovery
    curtailed_flow_dt = results_dt_indexed['curtailed_vre_share'] * results_dt_indexed['pdtNodeInflow']
    curtailed_flow_d = curtailed_flow_dt.groupby(level=['group', 'period']).sum()
    curtailed_vre_share_d = curtailed_flow_d / inflow_d.where(inflow_d != 0, pd.NA)

    # Slack shares: sum absolute values and recalculate
    upward_slack_d = results_dt_indexed['upward_slack'].groupby(['group', 'period']).sum()
    upward_slack_d = upward_slack_d.div(annualized_d.where(annualized_d != 0, pd.NA))

    downward_slack_d = results_dt_indexed['downward_slack'].groupby(['group', 'period']).sum()
    downward_slack_d = downward_slack_d.div(annualized_d.where(annualized_d != 0, pd.NA))

    # Combine period-level results
    results_d_indexed = pd.DataFrame({
        'sum_annualized_inflows': annualized_d,
        'vre_share': vre_share_d.fillna(0),
        'curtailed_vre_share': curtailed_vre_share_d.fillna(0),
        'upward_slack': upward_slack_d.fillna(0),
        'downward_slack': downward_slack_d.fillna(0)
    })
    results_d_indexed.columns.name = "parameter"

    results.append((results_d_indexed, 'nodeGroup_gd_p'))

    return results


def nodeGroup_VRE_share(par, s, v, r):
    """VRE share for node groups by period and time"""

    results = []

    # Get timesteps and groups
    timesteps = list(s.dt_realize_dispatch)
    
    # Filter groups that have nodes with inflow
    groups_with_inflow = []
    for g in s.groupOutput_node:
        # Get nodes in this group
        group_nodes_df = s.group_node[s.group_node.get_level_values('group') == g]
        if not group_nodes_df.empty:
            # Check if any node has inflow (not marked as 'no_inflow')
            if hasattr(s, 'node__inflow_method'):
                no_inflow_nodes = set(n for (n, method) in s.node__inflow_method if method == 'no_inflow')
                has_inflow = any(n not in no_inflow_nodes for n in group_nodes_df.get_level_values('node'))
            else:
                has_inflow = True
            if has_inflow:
                groups_with_inflow.append(g)
    
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
        # Get nodes in this group with inflow
        group_nodes_df = s.group_node[s.group_node.get_level_values('group') == g]
        if hasattr(s, 'node__inflow_method'):
            no_inflow_nodes = set(n for (n, method) in s.node__inflow_method if method == 'no_inflow')
            group_nodes = [n for n in group_nodes_df.get_level_values('node') if n not in no_inflow_nodes]
        else:
            group_nodes = group_nodes_df.get_level_values('node').tolist()
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


def CO2(par, s, v, r):
    """Annualized CO2 Mt for groups by period"""
    results = []

    # Calculate CO2 emissions in Mt
    total_co2 = ((r.emissions_co2_d * par.years_represented_d) / 1000000).sum(axis=0)
    co2_summary = pd.DataFrame(index=["CO2 [Mt]"], columns=["model_wide"], data=total_co2)
    co2_summary.index.name = 'param_CO2'
    results.append((co2_summary, 'CO2__'))

    # Process co2 emissions
    process_co2 = r.process_emissions_co2_d.groupby(['period']).sum()
    results.append((process_co2, 'process_co2_d_eee'))

    # Group co2 emissions
    results.append((r.group_co2_d, 'CO2_d_g'))
    return results

def nodeGroup_total_inflow(par, s, v, r):
    """Total inflow (inflow - outflow) to groups by period and time"""

    results = []
    groups = list(s.groupOutput_process)

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
    results.append((result_multi_dt, 'total_inflow_dt_g'))

    # Aggregate to period level
    result_multi_d = result_multi_dt.groupby(level='period').sum()

    # Divide by period shares to annualize
    result_multi_d = result_multi_d.div(par.complete_period_share_of_year, axis=0)
    result_multi_d.columns.name = 'group'

    # Return period results
    results.append((result_multi_d, 'total_inflow_d_g'))

    return results


def unit_outputNode(par, s, v, r):
    """Unit output node flow for periods and time"""

    results = []

    if r.flow_dt.empty:
        return results

    # Calculate timestep-level results first
    result_multi_dt = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['unit', 'node']))

    # Filter columns: unit processes that have sinks
    unit_sink_cols = r.flow_dt.columns[
        r.flow_dt.columns.get_level_values(0).isin(s.process_unit) &
        r.flow_dt.columns.to_series().apply(lambda col: (col[0], col[2]) in s.process_sink)
    ]

    for col in unit_sink_cols:
        u, source, sink = col
        result_multi_dt[(u, sink)] = r.flow_dt[col]

    # Return timestep results
    results.append((result_multi_dt, 'unit_outputNode_dt_ee'))

    # Aggregate to period level
    result_multi_d = result_multi_dt.groupby(level='period').sum()

    # Divide by period shares to annualize
    result_multi_d = result_multi_d.div(par.complete_period_share_of_year, axis=0)

    # Return period results
    results.append((result_multi_d, 'unit_outputNode_d_ee'))

    return results

def unit_inputNode(par, s, v, r):
    """Unit input node flow for periods and time"""

    results = []

    if r.flow_dt.empty:
        return results

    # Calculate timestep-level results first
    result_multi_dt = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['unit', 'node']))

    # Filter columns: unit processes that have sources
    unit_source_cols = r.flow_dt.columns[
        r.flow_dt.columns.get_level_values(0).isin(s.process_unit) &
        r.flow_dt.columns.to_series().apply(lambda col: (col[0], col[1]) in s.process_source)
    ]

    for col in unit_source_cols:
        u, source, sink = col
        result_multi_dt[(u, source)] = -r.flow_dt[col]

    # Return timestep results
    results.append((result_multi_dt, 'unit_inputNode_dt_ee'))

    # Aggregate to period level
    result_multi_d = result_multi_dt.groupby(level='period').sum()

    # Divide by period shares to annualize
    result_multi_d = result_multi_d.div(par.complete_period_share_of_year, axis=0)

    # Return period results
    results.append((result_multi_d, 'unit_inputNode_d_ee'))

    return results


def connection(par, s, v, r):
    """Connection flow for periods and time"""

    results = []

    # Return timestep results
    results.append((r.connection_dt, 'connection_dt_eee'))
    results.append((r.connection_losses_dt, 'connection_losses_dt_eee'))

    # Aggregate to period level
    r.connection_d = r.connection_dt.groupby(level='period').sum()
    r.connection_d = r.connection_d.div(par.complete_period_share_of_year, axis=0)
    r.connection_losses_d = r.connection_losses_dt.groupby(level='period').sum()
    r.connection_losses_d = r.connection_losses_d.div(par.complete_period_share_of_year, axis=0)

    # Return period results
    results.append((r.connection_d, 'connection_d_eee'))
    results.append((r.connection_losses_d, 'connection_losses_d_eee'))

    return results 


def connection_wards(par, s, v, r):
    """Connection flow to right node and to left node for periods and time"""

    results = []

    # Return timestep results
    results.append((r.connection_to_left_node__dt, 'connection_leftward_dt_eee'))
    results.append((r.connection_to_right_node__dt, 'connection_rightward_dt_eee'))

    # Return period results
    results.append((r.connection_to_right_node__d, 'connection_rightward_d_eee'))
    results.append((r.connection_to_left_node__d, 'connection_leftward_d_eee'))

    return results


def nodeGroup_flows(par, s, v, r):
    """Group output flows for periods and time"""

    results = []

    if s.groupOutputNodeFlows.empty or s.dt_realize_dispatch.empty:
        return results

    # Calculate timestep-level results first
    result_multi_dt = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['group', 'type', 'item']))

    # Assign simple mappings for all groups at once (before group loop)
    # Slack upward
    temp = r.group_node_up_slack__dt
    temp.columns = pd.MultiIndex.from_tuples([(g, 'slack', 'upward') for g in temp.columns], names=['group', 'type', 'item'])
    result_multi_dt[temp.columns] = temp

    # Process each group
    for g in s.groupOutputNodeFlows:
        # Unit aggregates (aggregateUnits to group)
        unit_to_group_cols = r.group_output__group_aggregate_Unit_to_group__dt.columns[
            r.group_output__group_aggregate_Unit_to_group__dt.columns.get_level_values('group') == g
        ]
        for ga in unit_to_group_cols.droplevel('group').unique():
            result_multi_dt[g, 'from_unitGroup', ga] = r.group_output__group_aggregate_Unit_to_group__dt[g, ga]

        # Units not in aggregate (unit to group) - sum across nodes
        unit_to_node_cols = r.group_output__unit_to_node_not_in_aggregate__dt.columns[
            r.group_output__unit_to_node_not_in_aggregate__dt.columns.get_level_values('group') == g
        ]
        for process, node in unit_to_node_cols.droplevel(['group']).unique():
            result_multi_dt[g, 'from_unit', process] = r.group_output__unit_to_node_not_in_aggregate__dt[process, node, g]

        # Connection aggregates (from connections to group) - sum across nodes
        from_conn_agg_cols = r.group_output__from_connection_aggregate__dt.columns[
            r.group_output__from_connection_aggregate__dt.columns.get_level_values('group') == g
        ]
        for ga in from_conn_agg_cols.droplevel(['group', 'node']).unique():
            result_multi_dt[g, 'from_connection_aggregate', ga] = r.group_output__from_connection_aggregate__dt[g, ga].sum(axis=1)

        # Connections not in aggregate (from connections)
        from_conn_not_agg_cols = r.group_output__from_connection_not_in_aggregate__dt.columns[
            r.group_output__from_connection_not_in_aggregate__dt.columns.get_level_values('group') == g
        ]
        for process, node in from_conn_not_agg_cols.droplevel('group').unique():
            result_multi_dt[g, 'from_connection', process] = r.group_output__from_connection_not_in_aggregate__dt[process, node, g]

        # Connections not in aggregate (to connections)
        to_conn_not_agg_cols = r.group_output__to_connection_not_in_aggregate__dt.columns[
            r.group_output__to_connection_not_in_aggregate__dt.columns.get_level_values('group') == g
        ]
        for process, node in to_conn_not_agg_cols.droplevel('group').unique():
            result_multi_dt[g, 'to_connection', process] = -r.group_output__to_connection_not_in_aggregate__dt[process, node, g]

        # Connection aggregates (to connections) - sum across nodes
        to_conn_agg_cols = r.group_output__to_connection_aggregate__dt.columns[
            r.group_output__to_connection_aggregate__dt.columns.get_level_values('group') == g
        ]
        for ga in to_conn_agg_cols.droplevel(['group', 'node']).unique():
            result_multi_dt[g, 'to_connection_aggregate', ga] = -r.group_output__to_connection_aggregate__dt[g, ga]

        # Group to unit aggregates (negative)
        group_to_unit_cols = r.group_output__group_aggregate_Group_to_unit__dt.columns[
            r.group_output__group_aggregate_Group_to_unit__dt.columns.get_level_values('group') == g
        ]
        for ga in group_to_unit_cols.droplevel('group').unique():
            result_multi_dt[g, 'unit_aggregate', ga] = -r.group_output__group_aggregate_Group_to_unit__dt[g, ga]

        # Node to unit not in aggregate (negative)
        node_to_unit_cols = r.group_output__node_to_unit_not_in_aggregate__dt.columns[
            r.group_output__node_to_unit_not_in_aggregate__dt.columns.get_level_values('group') == g
        ]
        for process, node in node_to_unit_cols.droplevel('group').unique():
            result_multi_dt[g, 'to_unit', process] = -r.group_output__node_to_unit_not_in_aggregate__dt[process, node, g]

    # Inflow
    temp = r.group_node_inflow_dt
    temp.columns = pd.MultiIndex.from_tuples([(g, 'inflow', g) for g in temp.columns], names=['group', 'type', 'item'])
    result_multi_dt[temp.columns] = temp

    # Internal losses - connections (sum across processes, negate)
    temp = r.group_output_Internal_connection_losses__dt.T.groupby('group').sum().T
    temp.columns = pd.MultiIndex.from_tuples([(g, 'internal_losses', 'connections') for g in temp.columns], names=['group', 'type', 'item'])
    result_multi_dt[temp.columns] = temp

    # Internal losses - units (sum across processes, negate)
    temp = r.group_output_Internal_unit_losses__dt.T.groupby('group').sum().T
    temp.columns = pd.MultiIndex.from_tuples([(g, 'internal_losses', 'units') for g in temp.columns], names=['group', 'type', 'item'])
    result_multi_dt[temp.columns] = temp

    # Internal losses - storages (negate)
    temp = r.group_node_state_losses__dt
    temp.columns = pd.MultiIndex.from_tuples([(g, 'internal_losses', 'storages') for g in temp.columns], names=['group', 'type', 'item'])
    result_multi_dt[temp.columns] = temp

    # Slack downward
    temp = r.group_node_down_slack__dt
    temp.columns = pd.MultiIndex.from_tuples([(g, 'slack', 'downward') for g in temp.columns], names=['group', 'type', 'item'])
    result_multi_dt[temp.columns] = temp


    result_multi_dt.columns.names = ['group', 'type', 'item']
    result_multi_dt = result_multi_dt.sort_index(axis=1, level='group', sort_remaining=False)

    # Return timestep results
    results.append((result_multi_dt, 'nodeGroup_flows_dt_gpe'))

    # Aggregate to period level
    result_multi_d = result_multi_dt.groupby(level='period').sum()

    # Return period results
    results.append((result_multi_d, 'nodeGroup_flows_d_gpe'))

    return results

def connection_cf(par, s, v, r):
    """Connection capacity factors for periods"""
    if r.process_sink_flow_d.empty:
        results = pd.DataFrame(index=pd.Index([], name='period'))
        return results, 'connection_cf_d_e'

    complete_hours = par.complete_period_share_of_year * 8760
    connection_cols = r.process_sink_flow_d.columns[r.process_sink_flow_d.columns.get_level_values(0).isin(s.process_connection)]
    connection_capacity = r.entity_all_capacity[connection_cols.droplevel(1).unique()].rename_axis('process', axis=1)
    connection_capacity.columns = connection_capacity.columns.get_level_values(0)
    results = r.connection_dt.abs().groupby('period').sum().div(connection_capacity, level=0).div(complete_hours, axis=0)
    results.columns.names = ['connection']
    return results, 'connection_cf_d_e'

def unit_cf_outputNode(par, s, v, r):
    """Unit capacity factors by output node for periods"""
    if r.process_sink_flow_d.empty:
        results = pd.DataFrame(index=pd.Index([], name='period'))
        return results, 'unit_outputs_cf_d_ee'

    complete_hours = par.complete_period_share_of_year * 8760
    unit_cols = r.process_sink_flow_d.columns[r.process_sink_flow_d.columns.get_level_values(0).isin(s.process_unit)]
    unit_capacity = r.entity_all_capacity[unit_cols.droplevel(1).unique()].rename_axis('process', axis=1)
    unit_capacity.columns = unit_capacity.columns.get_level_values(0)
    results = r.process_sink_flow_d[unit_cols].div(unit_capacity, level=0).div(complete_hours, axis=0)
    results.columns.names = ['unit', 'sink']
    return results, 'unit_outputs_cf_d_ee'

def unit_cf_inputNode(par, s, v, r):
    """Unit capacity factors by input node for periods"""
    if r.process_source_flow_d.empty:
        results = pd.DataFrame(index=pd.Index([], name='period'))
        return results, 'unit_inputs_cf_d_ee'
    # !!! This should account for efficiency losses in direct conversion units (but it does not)
    complete_hours = par.complete_period_share_of_year * 8760
    unit_source = r.process_source_flow_d.columns[r.process_source_flow_d.columns.get_level_values(0).isin(s.process_unit)]
    unit_capacity = r.entity_all_capacity[unit_source.droplevel(1).unique()].rename_axis('process', axis=1)
    unit_capacity.columns = unit_capacity.columns.get_level_values(0)
    results = r.process_source_flow_d[unit_source].div(unit_capacity, level=0).div(complete_hours, axis=0)
    results.columns.names = ['unit', 'source']
    return results, 'unit_inputs_cf_d_ee'


def unit_VRE_curtailment_and_potential(par, s, v, r):
    """Unit VRE curtailment and potential for both periods and timesteps"""
    
    results = []
    vre_processes = s.process_VRE.unique()
    
    # Timestep-level curtailment (absolute values) - calculate first
    if not r.flow_dt.empty and not r.potentialVREgen_dt.empty:
        curtail_dt = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['unit', 'sink']))
        potential_dt = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['unit', 'sink']))
        
        for col in r.flow_dt.columns:
            u, source, sink = col
            if u in vre_processes and (u, sink) in s.process_sink and (u, sink) in r.potentialVREgen_dt.columns:
                curtail_dt[u, sink] = r.potentialVREgen_dt[(u, sink)] - r.flow_dt[col]
                potential_dt[u, sink] = r.potentialVREgen_dt[(u, sink)]
        
        results.append((curtail_dt, 'unit_curtailment_outputNode_dt_ee'))
        results.append((potential_dt, 'unit_VRE_potential_outputNode_dt_ee'))
        
        # Calculate curtailment share at timestep level
        curtail_share_dt = (curtail_dt / potential_dt).where(potential_dt != 0, 0)
        results.append((curtail_share_dt, 'unit_curtailment_share_outputNode_dt_ee'))
        
        # Aggregate to period level
        curtail_period = curtail_dt.groupby(level='period').sum()
        potential_period = potential_dt.groupby(level='period').sum()
        
        # Calculate curtailment share at period level
        curtail_share_period = (curtail_period / potential_period).where(potential_period != 0, 0)

        results.append((curtail_share_period, 'unit_curtailment_share_outputNode_d_ee'))
        results.append((potential_period, 'unit_VRE_potential_outputNode_d_ee'))
    
    return results

def unit_ramps(par, s, v, r):
    """Unit ramps by input and output nodes for timesteps"""
    results = []
    if r.ramp_dtt.empty:
        return results

    # Output node ramps
    process_sink_ramp_output = s.process_sink[s.process_sink.get_level_values(0).isin(s.process_unit)]
    pss_ramp_output = r.ramp_dtt.columns[r.ramp_dtt.columns.droplevel(1).isin(process_sink_ramp_output)]
    ramp_output = r.ramp_dtt[pss_ramp_output].droplevel('t_previous')
    ramp_output.columns = ramp_output.columns.droplevel(1)  # Remove 'source' from (process, source, sink) to get (unit, source)
    ramp_output.columns.names = ['unit', 'sink']
    results.append((ramp_output, 'unit_ramp_outputs_dt_ee'))
    
    # Input node ramps
    process_source_ramp_input = s.process_source[s.process_source.get_level_values(0).isin(s.process_unit)]
    pss_ramp_input = r.ramp_dtt.columns[r.ramp_dtt.columns.droplevel(2).isin(process_source_ramp_input)]
    ramp_input = r.ramp_dtt[pss_ramp_input].droplevel('t_previous')
    ramp_input.columns = ramp_input.columns.droplevel(2)  # Remove 'sink' from (process, source, sink) to get (unit, source)
    ramp_input.columns.names = ['unit', 'source']
    results.append((ramp_input, 'unit_ramp_inputs_dt_ee'))

    return results

def cost_summaries(par, s, v, r):
    """Cost summaries for periods and timesteps"""
    
    results = []
    
    # Common calculations
    discount_ops = par.discount_factor_operations_yearly
    discount_invs = par.discount_factor_investment_yearly
    period_share = par.complete_period_share_of_year
    to_millions = 1000000
    
    # 1. Costs at timestep level (non-annualized)
    costs_dt = pd.DataFrame(index=s.dt_realize_dispatch, dtype=float)
    costs_dt.columns.name = 'category'
    costs_dt['commodity'] = r.cost_commodity_dt.sum(axis=1)
    costs_dt['co2'] = r.cost_co2_dt
    costs_dt['other operational'] = r.cost_process_other_operational_cost_dt.sum(axis=1)
    costs_dt['starts'] = r.cost_startup_dt.sum(axis=1)
    costs_dt['upward slack penalty'] = r.costPenalty_node_state_upDown_dt.xs('up', level='upDown', axis=1).sum(axis=1)
    costs_dt['downward slack penalty'] = r.costPenalty_node_state_upDown_dt.xs('down', level='upDown', axis=1).sum(axis=1)
    costs_dt['inertia slack penalty'] = r.costPenalty_inertia_dt.sum(axis=1)
    costs_dt['non-synchronous slack penalty'] = r.costPenalty_non_synchronous_dt.sum(axis=1)
    try:
        costs_dt['upward reserve slack penalty'] = r.costPenalty_reserve_upDown_dt.xs('up', level='updown', axis=1).sum(axis=1)
    except KeyError:
        costs_dt['upward reserve slack penalty'] = 0
    try:
        costs_dt['downward reserve slack penalty'] = r.costPenalty_reserve_upDown_dt.xs('down', level='updown', axis=1).sum(axis=1)
    except KeyError:
        costs_dt['downward reserve slack penalty'] = 0
    
    results.append((costs_dt, 'costs_dt_p'))
    
    # 2. Annualized, inflation adjusted and years represented (derived from costs_dt)
    dispatch_costs_pure_period = costs_dt.groupby(level='period').sum()
    dispatch_costs_annualized_period = dispatch_costs_pure_period.div(period_share, axis=0) / to_millions
    dispatch_costs_inflation_adjusted = dispatch_costs_annualized_period.mul(discount_ops, axis=0)

    # results.append((dispatch_period.reset_index(), dispatch_period, 'annualized_dispatch_costs_dt', 'tbl_annualized_dispatch_costs_period_t'))
    
    # 3. Discounted and inflation adjusted (with years represented) investment costs (d_realize_invest only)
    investment_costs = pd.DataFrame(index=s.d_realize_invest, dtype=float)
    investment_costs.columns.name = 'category'
    investment_costs['unit investment retirement'] = (r.costInvestUnit_d + r.costDivestUnit_d) / to_millions
    investment_costs['connection investment retirement'] = (r.costInvestConnection_d + r.costDivestConnection_d) / to_millions
    investment_costs['storage investment retirement'] = (r.costInvestState_d + r.costDivestState_d) / to_millions
    investment_costs['fixed cost pre-existing'] = r.costFixedPreExisting_d / to_millions
    investment_costs['fixed cost invested'] = r.costFixedInvested_d / to_millions
    investment_costs['fixed cost reduction due to divested'] = r.costFixedDivested_d / to_millions
    investment_costs['capacity margin penalty'] = r.costPenalty_capacity_margin_d / to_millions

    # Annualize back: Remove inflation adjustment and years represented
    annual_invest_costs = investment_costs.div(discount_invs, axis=0)
    annual_invest_costs['fixed cost pre-existing'] = investment_costs['fixed cost pre-existing'].div(discount_ops, axis=0)
    annual_invest_costs['fixed cost invested'] = investment_costs['fixed cost invested'].div(discount_ops, axis=0)
    annual_invest_costs['fixed cost reduction due to divested'] = investment_costs['fixed cost reduction due to divested'].div(discount_ops, axis=0)
    annual_invest_costs['capacity margin penalty'] = investment_costs['capacity margin penalty'].div(discount_ops, axis=0)

    # results.append((investment_costs.reset_index(), investment_costs, 'annualized_investment_costs_d', 'tbl_annualized_investment_costs_period'))
    
    # 4. Combined summary (investment + dispatch aggregated to period)
    all_periods = s.d_realized_period.union(s.d_realize_invest)
    summary = pd.DataFrame(index=all_periods, dtype=float)
    summary.columns.name = 'parameter'    

    # Without inflation and years (so, pure annual results)
    summary_annualized = annual_invest_costs.join(dispatch_costs_annualized_period)
    results.append((summary_annualized, 'annualized_costs_d_p'))

    # With years_represented adjusted with inflation (same as model)
    summary_inflation_years = investment_costs.join(dispatch_costs_inflation_adjusted)
    results.append((summary_inflation_years, 'costs_discounted_d_p'))

    # With years_represented adjusted with inflation (same as model)
    summary_inflation_years = investment_costs.join(dispatch_costs_inflation_adjusted)
    results.append((summary_inflation_years.sum(axis=0), 'costs_discounted_p_'))

    return results

def reserves(par, s, v, r):
    """Process reserves for timesteps and periods"""
    results = []
    
    # Timestep-level reserves
    results.append((r.reserves_dt, 'process_reserve_upDown_node_dt_eppe'))
    
    # Period-level reserves (average)
    results.append((r.reserves_d, 'process_reserve_average_d_eppe'))

    # Reserve price results
    results.append((v.dual_reserve_balance, 'reserve_prices_dt_ppg'))

    return results

def unit_online_and_startup(par, s, v, r):
    """Unit online status and startups for timesteps and periods"""
    results = []
    
    # 1. Online units dt
    online_units_dt = r.process_online_dt[s.process_unit.intersection(s.process_online)]
    results.append((online_units_dt, 'unit_online_dt_e'))
    
    # 2. Average online status at period level (weighted by step_duration)
    complete_hours = par.complete_period_share_of_year * 8760
    online_units_d = online_units_dt.mul(par.step_duration, axis=0).groupby('period').sum().div(complete_hours, axis=0)
    results.append((online_units_d, 'unit_online_average_d_e'))
    
    # 3. Startups aggregated to period level
    startup_units_d = r.process_startup_dt[s.process_unit.intersection(s.process_online)].groupby('period').sum()
    results.append((startup_units_d, 'unit_startup_d_e'))
    
    return results

def node_summary(par, s, v, r):
    """Node balance summaries for periods and timesteps"""
    results = []

    categories = ['inflow', 'from_units', 'from_connections', 'to_units', 'to_connections', 'self_discharge', 'upward_slack', 'downward_slack']
    nodes_sink = s.node.copy()
    nodes_sink.name = 'sink'
    nodes_source = s.node.copy()
    nodes_source.name = 'source'
    
    # 1. Timestep-level node summary
    node_dt = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_product([s.node, categories], names=['node', 'category']), dtype=float)
    inflow_cols = node_dt.columns[
                    node_dt.columns.get_level_values('node').isin(par.node_inflow.columns) 
                    & node_dt.columns.get_level_values('category').isin(['inflow'])]
    node_dt[inflow_cols] = par.node_inflow
    
    from_units = r.flow_dt[s.process_unit.join(r.flow_dt.columns).join(nodes_sink, how='inner')].T.groupby('sink').sum().T
    from_units_cols = node_dt.columns[
                        node_dt.columns.get_level_values('node').isin(from_units.columns)
                        & node_dt.columns.get_level_values('category').isin(['from_units'])]
    node_dt[from_units_cols] = from_units

    # From connections
    from_connections = r.flow_dt[s.process_connection.join(r.flow_dt.columns).join(nodes_sink, how='inner')].T.groupby('sink').sum().T
    from_connections_cols = node_dt.columns[
                        node_dt.columns.get_level_values('node').isin(from_connections.columns)
                        & node_dt.columns.get_level_values('category').isin(['from_connections'])]
    node_dt[from_connections_cols] = from_connections

    # To units (negative)
    to_units = -r.flow_dt[s.process_unit.join(r.flow_dt.columns).join(nodes_source, how='inner')].T.groupby('source').sum().T
    to_units_cols = node_dt.columns[
                        node_dt.columns.get_level_values('node').isin(to_units.columns)
                        & node_dt.columns.get_level_values('category').isin(['to_units'])]
    node_dt[to_units_cols] = to_units

    # To connections (negative)
    to_connections = -r.flow_dt[s.process_connection.join(r.flow_dt.columns).join(nodes_source, how='inner')].T.groupby('source').sum().T
    to_connections_cols = node_dt.columns[
                        node_dt.columns.get_level_values('node').isin(to_connections.columns)
                        & node_dt.columns.get_level_values('category').isin(['to_connections'])]
    node_dt[to_connections_cols] = to_connections

    # Self discharge
    self_discharge = r.self_discharge_loss_dt[r.self_discharge_loss_dt.columns.intersection(s.node_self_discharge)]
    self_discharge_cols = node_dt.columns[
                        node_dt.columns.get_level_values('node').isin(self_discharge.columns)
                        & node_dt.columns.get_level_values('category').isin(['self_discharge'])]
    node_dt[self_discharge_cols] = self_discharge

    # Upward slack
    balanced_nodes = s.node_balance.union(s.node_balance_period)
    upward_slack_data = v.q_state_up.loc[:, v.q_state_up.columns.get_level_values('node').isin(balanced_nodes)]
    upward_slack_data = upward_slack_data.mul(par.node_capacity_for_scaling[upward_slack_data.columns], axis=1)
    upward_slack_cols = node_dt.columns[
                        node_dt.columns.get_level_values('node').isin(upward_slack_data.columns.get_level_values('node'))
                        & node_dt.columns.get_level_values('category').isin(['upward_slack'])]
    node_dt[upward_slack_cols] = upward_slack_data

    # Downward slack (negative)
    downward_slack_data = -v.q_state_down.loc[:, v.q_state_down.columns.get_level_values('node').isin(balanced_nodes)]
    downward_slack_data = downward_slack_data.mul(par.node_capacity_for_scaling[downward_slack_data.columns], axis=1)
    downward_slack_cols = node_dt.columns[
                        node_dt.columns.get_level_values('node').isin(downward_slack_data.columns.get_level_values('node'))
                        & node_dt.columns.get_level_values('category').isin(['downward_slack'])]
    node_dt[downward_slack_cols] = downward_slack_data

    # Fill any remaining NaN values with 0
    node_dt = node_dt.fillna(0)
    
    results.append((node_dt, 'node_dt_ep'))
        
 # 2. Period-level node summary
    node_d = node_dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0, level=1)
    
    results.append((node_d, 'node_d_ep'))
    
    return results

def node_additional_results(par, s, v, r):
    """Additional node results: prices, state, and slacks"""
    results = []
    
    # 1. Nodal prices
    results.append((v.dual_node_balance, 'node_prices_dt_e'))
    
    # 2. Node state
    node_state = v.state.mul(par.entity_unitsize[s.node_state], level="node")
    results.append((node_state, 'node_state_dt_e'))
    
    # 3. Node upward slack
    upward_slack = v.q_state_up.mul(par.node_capacity_for_scaling[s.node_balance.union(s.node_balance_period)], level=0)
    results.append((upward_slack, 'node_slack_up_dt_e'))

    # 4. Node downward slack
    downward_slack = v.q_state_down.mul(par.node_capacity_for_scaling[s.node_balance.union(s.node_balance_period)], level=0)
    results.append((downward_slack, 'node_slack_down_dt_e'))
    
    return results

def investment_duals(par, s, v, r):
    """Additional node results: prices, state, and slacks"""
    results = []
    
    # 1. v.dual_invest_unit
    dual_invest_unit = v.dual_invest_unit.div(par.entity_unitsize[v.dual_invest_unit.columns])
    results.append((dual_invest_unit, 'dual_invest_unit_d_e'))

    # 2. v.dual_invest_connection
    dual_invest_connection = v.dual_invest_connection.div(par.entity_unitsize[v.dual_invest_connection.columns])
    results.append((dual_invest_connection, 'dual_invest_connection_d_e'))

    # 3. v.dual_invest_node
    dual_invest_node = v.dual_invest_node.div(par.entity_unitsize[v.dual_invest_node.columns])
    results.append((dual_invest_node, 'dual_invest_node_d_e'))

    return results

def inertia_results(par, s, v, r):
    """Inertia results for groups and individual entities"""
    
    results = []
    
    # Helper to get inertia constant
    def get_inertia_constant(p, node, direction):
        if direction == 'source':
            return par.process_source[p, node]['inertia_constant'] if 'inertia_constant' in par.process_source.index else 0.0
        else:  # sink
            return par.process_sink[p, node]['inertia_constant'] if 'inertia_constant' in par.process_sink.index else 0.0
    
    # Helper to get flow/online value
    def get_flow_or_online(p, source, sink, df_index):
        unitsize = par.entity_unitsize[p]
        if p in s.process_online and p in r.process_online_dt.columns:
            return r.process_online_dt[p] * unitsize
        elif (p, source, sink) in v.flow.columns:
            return v.flow[(p, source, sink)] * unitsize
        else:
            return pd.Series(0, index=df_index)
    
    # 1. Group inertia totals
    group_inertia = pd.DataFrame(index=s.dt_realize_dispatch, columns=s.groupInertia, dtype=float)
    
    for g in s.groupInertia:
        total_inertia = pd.Series(0, index=s.dt_realize_dispatch, dtype=float)
        
        # Inertia from sources
        s.group_node_inertia = s.group_node[s.group_node.get_level_values('group').isin([g])]
        s.process_source_inertia = s.process_source[s.process_source.get_level_values(1).isin(s.group_node_inertia.get_level_values('node'))]
        s.pss_inertia = s.process_source_sink_alwaysProcess[s.process_source_sink_alwaysProcess.droplevel(2).isin(s.process_source_inertia)]
        for (p, source, sink) in s.pss_inertia:
            inertia_const = get_inertia_constant(p, source, 'source')
            if inertia_const:
                flow_online = get_flow_or_online(p, source, sink, s.dt_realize_dispatch)
                total_inertia += (flow_online * inertia_const).squeeze()
        
        # Inertia from sinks
        s.process_sink_inertia = s.process_sink[s.process_sink.get_level_values(1).isin(s.group_node_inertia.get_level_values('node'))]
        s.pss_inertia = s.process_source_sink_alwaysProcess[s.process_source_sink_alwaysProcess.droplevel(1).isin(s.process_sink_inertia)]
        for (p, source, sink) in s.pss_inertia:
            inertia_const = get_inertia_constant(p, sink, 'sink')
            if inertia_const:
                flow_online = get_flow_or_online(p, source, sink, s.dt_realize_dispatch)
                total_inertia += (flow_online * inertia_const).squeeze()
        
        group_inertia[g] = total_inertia
    
    results.append((group_inertia, 'nodeGroup_inertia_dt_g'))
    
    # 2. Individual entity inertia
    unit_inertia = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['group', 'process', 'node']), dtype=float)
    
    for g in s.groupInertia:
        # From sources
        for (p, source, sink) in s.process_source_sink_alwaysProcess:
            if (p, source) in s.process_source and (g, source) in s.group_node:
                inertia_const = get_inertia_constant(p, source, 'source')
                if inertia_const:
                    flow_online = get_flow_or_online(p, source, sink, s.dt_realize_dispatch)
                    unit_inertia[g, p, source] = flow_online * inertia_const
        
        # From sinks
        for (p, source, sink) in s.process_source_sink_alwaysProcess:
            if (p, sink) in s.process_sink and (g, sink) in s.group_node:
                inertia_const = get_inertia_constant(p, sink, 'sink')
                if inertia_const:
                    flow_online = get_flow_or_online(p, source, sink, s.dt_realize_dispatch)
                    unit_inertia[g, p, sink] = flow_online * inertia_const
    
    results.append((unit_inertia, 'nodeGroup_unit_node_inertia_dt_gee'))
    
    # 3. Largest flow per group (for inertia constraint)
    largest_flow = pd.DataFrame(index=s.dt_realize_dispatch, dtype=float)
    
    for g in s.groupInertia:
        max_flows = []
        # This is poor processing and probably not working correctly.
        for (p, source, sink) in s.process_source_sink_alwaysProcess:
            if (p, sink) in s.process_sink and (g, sink) in s.group_node:
                unitsize = par.entity_unitsize[p]
                if (p, source, sink) in v.flow.columns:
                    max_flows.append(v.flow[(p, source, sink)] * unitsize)
        
        if max_flows:
            # Take maximum across all processes
            largest_flow[g] = pd.concat(max_flows, axis=1).max(axis=1)
        else:
            largest_flow[g] = 0
    
    results.append((largest_flow, 'nodeGroup_inertia_largest_flow_dt_g'))
    
    return results

def slack_variables(par, s, v, r):
    """Slack variables for reserves, non-synchronous, inertia, and capacity margin"""
    
    results = []
    
    # 1. Reserve slack variables
    reserve_slack = v.q_reserve * par.reserve_upDown_group_reservation
    # pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['reserve', 'updown', 'node_group']), dtype=float)
    # for col in v.q_reserve.columns:
    #     if col in par.reserve_upDown_group_reservation.columns:
    #         reserve_slack[col] = v.q_reserve[col] * par.reserve_upDown_group_reservation[col]
    results.append((reserve_slack, 'nodeGroup_slack_reserve_dt_eeg'))
    
    # 2. Non-synchronous slack variables
    nonsync_slack = v.q_non_synchronous * par.group_capacity_for_scaling[v.q_non_synchronous.columns]
    results.append((nonsync_slack, 'nodeGroup_slack_nonsync_dt_g'))
    
    # 3. Inertia slack variables
    inertia_slack = v.q_inertia * par.group_inertia_limit
    results.append((inertia_slack, 'nodeGroup_slack_inertia_dt_g'))
    
    # 4. Capacity margin slack variables (for investment periods only)
    capmargin_slack = v.q_capacity_margin * par.group_capacity_for_scaling[s.groupCapacityMargin]
    results.append((capmargin_slack, 'nodeGroup_slack_capacity_margin_d_g'))
    
    return results


def print_namespace_structure(namespace, name='r', max_items=3, output_file='namespace_structure.txt'):
    import pandas as pd
    import sys
    
    def format_list(items, max_n=max_items):
        items_list = list(items)
        if len(items_list) <= max_n:
            return items_list
        return items_list[:max_n] + [f'... ({len(items_list)} total)']
    
    with open(output_file, 'a') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        
        for attr_name in dir(namespace):
            if attr_name.startswith('_'):
                continue
            
            obj = getattr(namespace, attr_name)
            print(f"\n{name}.{attr_name}")
            
            if isinstance(obj, pd.DataFrame):
                print("Type: DataFrame")
                print(f"Shape: {obj.shape}")
                print(f"Index: {obj.index.names if hasattr(obj.index, 'names') else 'default'}")
                print(f"Columns: {format_list(obj.columns)}")
                print(f"Dtypes:\n{obj.dtypes}")
                
            elif isinstance(obj, pd.Series):
                print("Type: Series")
                print(f"Shape: {obj.shape}")
                print(f"Index: {obj.index.name or 'default'}")
                print(f"Dtype: {obj.dtype}")
                
            elif isinstance(obj, pd.Index):
                print("Type: Index")
                print(f"Name: {obj.name}")
                print(f"Values: {format_list(obj)}")
                print(f"Dtype: {obj.dtype}")
        
        sys.stdout = original_stdout


def write_summary_csv(par, s, v, r, csv_dir):
    """Write summary CSV file matching the GNU MathProg format"""
    import os

    # Output file path
    fn_summary = os.path.join(csv_dir, 'summary_solve.csv')

    # Get common parameters - these are Series indexed by period
    p_discount_factor_operations_yearly = par.discount_factor_operations_yearly
    p_discount_factor_investment_yearly = par.discount_factor_investment_yearly
    complete_period_share_of_year = par.complete_period_share_of_year

    # Get period sets
    period_in_use = complete_period_share_of_year.index
    d_realized_period = s.d_realized_period
    d_realize_invest = s.d_realize_invest

    # Open file and write all content
    with open(fn_summary, 'w') as f:
        # Header with timestamp
        timestamp = datetime.now(timezone.utc)
        f.write(f'"Diagnostic results from all solves. Output at (UTC): {timestamp}"\n\n')

        # Total cost from solver (M CUR)
        f.write('\n')
        f.write('"Solve","Objective","Total cost from solver, includes all penalty costs"\n')
        for row_idx in v.obj.index:
            f.write(f'{row_idx},{v.obj.loc[row_idx, "objective"] / 1000000:.12g}\n')

        # Total cost (calculated) full horizon (M CUR)
        # Sum over all periods: (operational + penalty) * discount / period_share + invest + divest
        total_cost_full = (
            r.costOper_and_penalty_d
                .add(r.costInvest_d, fill_value=0.0)
                .add(r.costDivest_d, fill_value=0.0)
        ).sum(axis=0) / 1000000

        f.write(f'"Total cost (calculated) full horizon (M CUR)",{total_cost_full:.12g},"Annualized operational, penalty and investment costs"\n')
        f.write(f'"Total cost (calculated) realized periods (M CUR)",{total_cost_full:.12g}\n')

        # Operational costs for realized periods (M CUR)
        operational_costs = r.costOper_d.sum(axis=0) / 1000000
        f.write(f'"Operational costs for realized periods (M CUR)",{operational_costs:.12g}\n')

        # Investment costs for realized periods (M CUR)
        investment_costs = r.costInvest_d.sum(axis=0) / 1000000
        f.write(f'"Investment costs for realized periods (M CUR)",{investment_costs:.12g}\n')

        # Retirement costs (negative salvage value) for realized periods (M CUR)
        retirement_costs = r.costDivest_d.sum(axis=0) / 1000000
        f.write(f'"Retirement costs (negative salvage value) for realized periods (M CUR)",{retirement_costs:.12g}\n')

        # Fixed costs for existing entities (M CUR)
        fixed_costs_pre_existing = r.costFixedPreExisting_d.sum(axis=0) / 1000000
        fixed_costs_invested = r.costFixedInvested_d.sum(axis=0) / 1000000
        fixed_costs_divested = r.costFixedDivested_d.sum(axis=0) / 1000000

        f.write(f'"Fixed costs for pre-existing entities (M CUR)",{fixed_costs_pre_existing:.12g}\n')
        f.write(f'"Fixed costs for invested entities (M CUR)",{fixed_costs_invested:.12g}\n')
        f.write(f'"Fixed cost removal due to divested entities (M CUR)",{fixed_costs_divested:.12g}\n')

        # Penalty (slack) costs for realized periods (M CUR)
        penalty_costs = r.costPenalty_d.sum(axis=0) / 1000000
        f.write(f'"Penalty (slack) costs for realized periods (M CUR)",{penalty_costs:.12g}\n')

        # Period information table
        f.write('\nPeriod')
        for d in period_in_use:
            f.write(f',{d}')
        f.write('\n')

        # Time in use in years
        f.write('"Time in use in years"')
        for d in period_in_use:
            f.write(f',{complete_period_share_of_year[d]:.12g}')
        f.write('\n')

        # Operational discount factor
        f.write('"Operational discount factor"')
        for d in period_in_use:
            f.write(f',{p_discount_factor_operations_yearly[d]:.12g}')
        f.write('\n')

        # Investment discount factor
        f.write('"Investment discount factor"')
        for d in s.d_realize_invest:
            f.write(f',{p_discount_factor_investment_yearly[d]:.12g}')
        f.write('\n\n')

        # Emissions section
        f.write('Emissions\n')
        co2_total = r.emissions_co2_d.sum(axis=0) / 1000000
        f.write(f'"CO2 [Mt]",{co2_total:.6g},"System-wide annualized CO2 emissions for realized periods"\n')

        # Slack variables section
        f.write('\n"Slack variables multiplied by timestep duration (creating or removing energy/matter, ')
        f.write('creating inertia, adding synchronous generation, decreasing capacity margin, creating reserve)"\n')

        # Node state slack - upward (creating energy)
        for node in r.upward_node_slack_d_not_annualized.columns:
            for period in d_realized_period:
                if period in r.upward_node_slack_d_not_annualized.index and r.upward_node_slack_d_not_annualized.loc[period, node] > 0:
                    f.write(f'Created, {node}, {period}, {r.upward_node_slack_d_not_annualized.loc[period, node]:.5g}\n')

        # Node state slack - downward (removing energy)
        for node in r.downward_node_slack_d_not_annualized.columns:
            for period in d_realized_period:
                if period in r.downward_node_slack_d_not_annualized.index and r.downward_node_slack_d_not_annualized.loc[period, node] > 0:
                    f.write(f'Removed, {node}, {period}, {r.downward_node_slack_d_not_annualized.loc[period, node]:.5g}\n')

        # Inertia slack
        for group in r.q_inertia_d_not_annualized.columns:
            for period in d_realized_period:
                if period in r.q_inertia_d_not_annualized.index and r.q_inertia_d_not_annualized.loc[period, group] > 0:
                    f.write(f'Inertia, {group}, {period}, {r.q_inertia_d_not_annualized.loc[period, group]:.5g}\n')

        # Non-synchronous slack
        for group in r.q_non_synchronous_d_not_annualized.columns:
            for period in d_realized_period:
                if period in r.q_non_synchronous_d_not_annualized.index and r.q_non_synchronous_d_not_annualized.loc[period, group] > 0:
                    f.write(f'NonSync, {group}, {period}, {r.q_non_synchronous_d_not_annualized.loc[period, group]:.5g}\n')

        # Capacity margin slack
        for group in r.q_capacity_margin_d_not_annualized.columns:
            for period in d_realize_invest:
                if period in r.q_capacity_margin_d_not_annualized.index and r.q_capacity_margin_d_not_annualized.loc[period, group] > 0:
                    f.write(f'CapMargin, {group}, {period}, {r.q_capacity_margin_d_not_annualized.loc[period, group]:.5g}\n')
        
        # Reserve slack
        for group in r.q_non_synchronous_d_not_annualized.columns:
            for period in d_realized_period:
                if period in r.q_reserves_d_not_annualized.index and r.q_reserves_d_not_annualized.loc[period, group] > 0:
                    f.write(f'Reserve, {group}, {period}, {r.q_reserves_d_not_annualized.loc[period, group]:.5g}\n')


# List of all output functions
ALL_OUTPUTS = [
    generic,
    cost_summaries,
    reserves,
    unit_online_and_startup,
    node_summary,
    node_additional_results,
    investment_duals,
    inertia_results,
    slack_variables,
    unit_capacity,
    connection_capacity,
    node_capacity,
    nodeGroup_indicators,
    nodeGroup_VRE_share,
    CO2,
    nodeGroup_flows,
    unit_outputNode,
    unit_inputNode,
    connection,
    connection_wards,
    nodeGroup_total_inflow,
    connection_cf,
    unit_cf_outputNode,
    unit_cf_inputNode,
    unit_VRE_curtailment_and_potential,
    unit_ramps,
]


# writer.py - handles the actual writing
def write_outputs(scenario_name, output_config_path, output_funcs=None, subdir=None, read_parquet_dir=False, methods=['plot', 'parquet', 'excel', 'db', 'csv'], plot_rows=(0, 167), debug=False, single_result=None):
    """
    Write FlexTool outputs to various formats.

    Args:
        scenario_name: Name of the scenario
        output_config_path: Path to YAML configuration file defining outputs
        output_funcs: list of functions to run, or None for ALL_OUTPUTS
        subdir: Subdirectory for outputs
        read_parquet_dir: Directory to read existing parquet files from
        methods: List of output methods ('plot', 'parquet', 'excel', 'db', 'csv')
        plot_rows: Tuple of first and last row to plot in a time series plots. Default is (0, 167).
        debug: Enable debug output
        single_result: Tuple of (key, csv_name, plot_name, plot_type, subplots_per_row, legend_position)
                       for processing a single result. Overrides config file.
    """
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

    # Load output configuration from YAML or create from single_result
    if single_result:
        # Parse single_result tuple
        key, csv_name, plot_name, plot_type, subplots_per_row, legend_position = single_result

        # Convert string "null" to None
        def parse_value(val):
            if val == "null" or val == "None":
                return None
            # Try to convert to int if it's a numeric string
            try:
                return int(val)
            except (ValueError, TypeError):
                return val

        csv_name = parse_value(csv_name)
        plot_name = parse_value(plot_name)
        plot_type = parse_value(plot_type)
        subplots_per_row = parse_value(subplots_per_row)
        legend_position = parse_value(legend_position)

        # Create single-entry settings dict
        settings = {
            key: [csv_name, plot_name, plot_type, subplots_per_row, legend_position]
        }
    else:
        # Load output configuration from YAML
        with open(output_config_path, 'r') as f:
            settings = yaml.safe_load(f)

    if subdir:
        parquet_dir = os.path.join('output_parquet', subdir)
        csv_dir = os.path.join('output_csv', subdir)
        plot_dir = os.path.join('output_plots', subdir)  # Fixed: was os.path.join(subdir, ' output_plots')
    else:
        parquet_dir = 'output_parquet'
        csv_dir = 'output_csv'
        plot_dir = 'output_plots'


    # Handle single_result mode with special data reading logic
    if single_result:
        key = single_result[0]
        csv_name = settings[key][0]  # Already parsed above
        results = {}

        start = time.perf_counter()

        if read_parquet_dir:
            # Read single parquet file
            parquet_path = os.path.join(read_parquet_dir, f'{key}.parquet')
            if not os.path.exists(parquet_path):
                raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
            results[key] = pd.read_parquet(parquet_path)
            print(f"--- Read single parquet file ({key}): {time.perf_counter() - start:.4f} seconds")
        else:
            # Read single CSV file
            if not csv_name:
                raise ValueError(f"CSV name is required for key '{key}' when not reading from parquet")

            csv_path = os.path.join(csv_dir, csv_name)
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")

            # Read CSV with multi-level index/header inference
            df = pd.read_csv(csv_path, header=0, index_col=0)
            results[key] = df
            print(f"--- Read single CSV file ({csv_name}): {time.perf_counter() - start:.4f} seconds")

        start = time.perf_counter()
    # Otherwise, use normal data reading pipeline
    else:
        start = time.perf_counter()
        # If results already exist as parquet files, just read them without processing
        if read_parquet_dir:
            results = {}
            for filename in os.listdir(read_parquet_dir):
                if filename.endswith('.parquet'):
                    key = filename[:-8]  # Remove '.parquet' extension
                    filepath = os.path.join(read_parquet_dir, filename)
                    results[key] = pd.read_parquet(filepath)
            print(f"--- Read parquet files: {time.perf_counter() - start:.4f} seconds")
            start = time.perf_counter()

        # Read original raw outputs from FlexTool
        else:
            par, s, v = read_outputs('output_raw')
            print(f"--- Read flextool outputs: {time.perf_counter() - start:.4f} seconds")
            start = time.perf_counter()

            # Pre-process results to be closer to what needed for output writing
            r = post_process_results(par, s, v)
            print(f"--- Post processed outputs: {time.perf_counter() - start:.4f} seconds")
            start = time.perf_counter()

            # Call the final processing functions for each category of outputs
            # and make a dict of dataframes to hold final results
            output_funcs = output_funcs or ALL_OUTPUTS

            results = {}
            for func in output_funcs:
                func_results = func(par, s, v, r)

                # Handle both single result (wrapped in list) and multiple results
                if not isinstance(func_results, list):
                    func_results = [func_results]

                for result_df, table_name in func_results:
                    # Use excel_sheet as the key to allow multiple outputs per function
                    results[table_name] = result_df

            print(f"--- Formatted for output: {time.perf_counter() - start:.4f} seconds")
            start = time.perf_counter()

    # Write files for debugging purposes
    if debug:
        open('namespace_structure.txt', 'w').close()
        print_namespace_structure(r, 'r')
        print_namespace_structure(s, 's')
        print_namespace_structure(v, 'v')
        print_namespace_structure(par, 'par')
        print(f"--- Debug outputs: {time.perf_counter() - start:.4f} seconds")
        start = time.perf_counter()

    # Write to parquet
    if 'parquet' in methods and not read_parquet_dir:
        for name, df in results.items():
            if not os.path.exists(parquet_dir):
                os.makedirs(parquet_dir)
            df = pd.concat({scenario_name: df}, axis=1, names=['scenario'])
            df.to_parquet(f'{parquet_dir}/{name}.parquet')

        print(f"--- Wrote to parquet: {time.perf_counter() - start:.4f} seconds")
        start = time.perf_counter()

    # Plot results
    if 'plot' in methods:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        # Don't delete existing plots when processing single result
        delete_plots = not bool(single_result)
        plot_dict_of_dataframes(results, plot_dir, settings, plot_rows=plot_rows, delete_existing_plots=delete_plots)

        print(f"--- Plotted figures: {time.perf_counter() - start:.4f} seconds")
        start = time.perf_counter()

    # Write to csv
    if 'csv' in methods:
        if single_result and read_parquet_dir:
            # Simplified CSV writing for single result from parquet
            if not os.path.exists(csv_dir):
                os.makedirs(csv_dir)

            for table_name, attributes in settings.items():
                if table_name and table_name in results and attributes[0]:
                    csv_filename = attributes[0]
                    df = results[table_name]
                    csv_path = os.path.join(csv_dir, csv_filename)
                    df_copy = df.reset_index()
                    df_copy.columns.names = [None] * df_copy.columns.nlevels
                    df_copy.to_csv(csv_path, index=False, float_format='%.8g')

            print(f"--- Wrote to CSV: {time.perf_counter() - start:.4f} seconds")
            start = time.perf_counter()
        elif not (single_result and read_parquet_dir):
            # Original CSV writing logic with full processing
            if not os.path.exists(csv_dir):
                os.makedirs(csv_dir)
            # Empty csv dir
            for filename in os.listdir(csv_dir):
                file_path = os.path.join(csv_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            write_summary_csv(par, s, v, r, csv_dir)

            for table_name, attributes in settings.items():
                if table_name and table_name in results and attributes[0]:
                    csv_filename = attributes[0]
                    df = results[table_name]
                    if 'solve' not in df.index.names and 'period' in df.index.names: # and csv_filename not in ['costs_discounted.csv']
                        df.index = df.index.join(s.solve_period)
                        names = list(df.index.names)
                        solve_pos = names.index('solve')
                        period_pos = names.index('period')
                        names.pop(solve_pos)
                        if solve_pos < period_pos:
                            period_pos -= 1
                        names.insert(period_pos, 'solve')
                        df.index = df.index.reorder_levels(order=names)

                    #if not df.empty and len(df) > 0:
                    # Write to CSV with proper multi-index column handling
                    csv_path = os.path.join(csv_dir, csv_filename)
                    df = df.reset_index()
                    df.columns.names = [None] * df.columns.nlevels
                    df.to_csv(csv_path, index=False, float_format='%.8g')

            print(f"Wrote to CSV: {time.perf_counter() - start:.4f} seconds")
            start = time.perf_counter()

    # Write to excel
    if 'excel' in methods:
        with pd.ExcelWriter('output_' + scenario_name + '.xlsx') as writer:
            for name, df in results.items():
                if (not df.empty) & (len(df) > 0):
                    df.to_excel(writer, sheet_name=name)

        print(f"Wrote to Excel: {time.perf_counter() - start:.4f} seconds")
        start = time.perf_counter()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Write FlexTool outputs to various formats')
    parser.add_argument('scenario_name', type=str, help='Name of the scenario')
    parser.add_argument('--config', type=str, default='templates/default_plots.yaml',
                        help='Path to output configuration YAML file (default: templates/default_plots.yaml)')
    parser.add_argument('--subdir', type=str, default=None,
                        help='Subdirectory for outputs (default: current directory)')
    parser.add_argument('--read-parquet-dir', type=str, default=False,
                        help='Directory to read existing parquet files from (default: False, reads from raw CSV files)')
    parser.add_argument('--methods', type=str, nargs='+', default=['plot', 'parquet', 'csv'],
                        choices=['plot', 'parquet', 'excel', 'db', 'csv'],
                        help='Output methods to use (default: plot parquet csv)')
    parser.add_argument('--plot-rows', type=int, nargs=2, default=[0, 167],
                        help='First and last row to plot in time series (default: 0 167)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    parser.add_argument('--single-result', type=str, nargs=6,
                        metavar=('KEY', 'CSV_NAME', 'PLOT_NAME', 'PLOT_TYPE', 'SUBPLOTS_PER_ROW', 'LEGEND_POSITION'),
                        help='Process a single result (overrides --config): key csv_name plot_name plot_type subplots_per_row legend_position. Use "null" for None values.')

    args = parser.parse_args()

    write_outputs(
        args.scenario_name,
        args.config,
        output_funcs=None,
        subdir=args.subdir,
        read_parquet_dir=args.read_parquet_dir,
        methods=args.methods,
        plot_rows=tuple(args.plot_rows),
        debug=args.debug,
        single_result=tuple(args.single_result) if args.single_result else None
    )