import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import time
from flextool.read_flextool_outputs import read_variables, read_parameters, read_sets
from flextool.process_results import post_process_results
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
    result_multi = pd.DataFrame(index=index)
    result_multi.columns.name = 'parameter'
    
    # Existing capacity - filter to process_unit only
    existing = par.entity_all_existing[processes].unstack()
    result_multi['existing'] = existing
    
    # Invested capacity - default to None, overwrite if data exists
    result_multi['invested'] = pd.Series(dtype=float)
    if not v.invest.empty and len(v.invest.columns) > 0:
        ed_unit_invest = s.ed_invest[s.ed_invest.get_level_values('entity').isin(s.process_unit)]
        unit_invest = ed_unit_invest.get_level_values('entity').unique()
        result_multi['invested'] = v.invest.unstack()[ed_unit_invest] * par.entity_unitsize[unit_invest]
    
    # Divested capacity - default to None, overwrite if data exists
    result_multi['divested'] = pd.Series(dtype=float)
    if not v.divest.empty and len(v.divest.columns) > 0:
        ed_unit_divest = s.ed_divest[s.ed_divest.get_level_values('entity').isin(s.process_unit)]
        unit_divest = ed_unit_divest.get_level_values('entity').unique()
        result_multi['divested'] = v.divest.unstack()[ed_unit_divest] * par.entity_unitsize[unit_divest]
    
    # Total capacity - filter to process_unit only
    total = r.entity_all_capacity[processes].unstack()
    result_multi['total'] = total
    result_multi = result_multi[['existing', 'invested', 'divested', 'total']]
    result_flat = result_multi.reset_index()
    return result_flat, result_multi, 'unit_capacity_ed_p'


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
    result_multi = pd.DataFrame(index=index)
    result_multi.columns.name = 'parameter'
    
    # Existing capacity - filter to process_connection only
    existing = par.entity_all_existing[connections].unstack()
    result_multi['existing'] = existing
    
    # Invested capacity - default to empty, overwrite if data exists
    result_multi['invested'] = pd.Series(dtype=float)
    if not v.invest.empty and len(v.invest.columns) > 0:
        ed_conn_invest = s.ed_invest[s.ed_invest.get_level_values('entity').isin(s.process_connection)]
        conn_invest = ed_conn_invest.get_level_values('entity').unique()
        result_multi['invested'] = v.invest.unstack()[ed_conn_invest] * par.entity_unitsize[conn_invest]
    
    # Divested capacity - default to empty, overwrite if data exists
    result_multi['divested'] = pd.Series(dtype=float)
    if not v.divest.empty and len(v.divest.columns) > 0:
        ed_conn_divest = s.ed_divest[s.ed_divest.get_level_values('entity').isin(s.process_connection)]
        conn_divest = ed_conn_divest.get_level_values('entity').unique()
        result_multi['invested'] = v.divest.unstack()[ed_conn_divest] * par.entity_unitsize[conn_divest]
    
    # Total capacity - filter to process_connection only
    total = r.entity_all_capacity[connections].unstack()
    result_multi['total'] = total
    
    # Reorder columns
    result_multi = result_multi[['existing', 'invested', 'divested', 'total']]
    
    # Flatten for Excel
    result_flat = result_multi.reset_index()
    
    return result_flat, result_multi, 'connection_capacity_ed_p'


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
    result_multi = pd.DataFrame(index=index)
    result_multi.columns.name = 'parameter'
    
    # Existing capacity - filter to node_state only
    if nodes:
        existing = par.entity_all_existing[nodes].unstack()
        result_multi['existing'] = existing
    else:
        result_multi['existing'] = pd.Series(dtype=float)
    
    # Invested capacity - default to empty, overwrite if data exists
    result_multi['invested'] = pd.Series(dtype=float)
    if not v.invest.empty and len(v.invest.columns) > 0:
        ed_node_invest = s.ed_invest[s.ed_invest.get_level_values('entity').isin(s.node)]
        node_invest = ed_node_invest.get_level_values('entity').unique()
        result_multi['invested'] = v.invest.unstack()[ed_node_invest] * par.entity_unitsize[node_invest]
    
    # Divested capacity - default to empty, overwrite if data exists
    result_multi['divested'] = pd.Series(dtype=float)
    if not v.divest.empty and len(v.divest.columns) > 0:
        ed_node_divest = s.ed_divest[s.ed_divest.get_level_values('entity').isin(s.node)]
        node_divest = ed_node_divest.get_level_values('entity').unique()
        result_multi['invested'] = v.invest.unstack()[ed_node_divest] * par.entity_unitsize[node_divest]
    
    # Total capacity - filter to node_state only
    if nodes:
        result_multi['total'] = r.entity_all_capacity[nodes].unstack()
    else:
        result_multi['total'] = pd.Series(dtype=float)
    
    result_multi = result_multi[['existing', 'invested', 'divested', 'total']]
    result_flat = result_multi.reset_index()
    return result_flat, result_multi, 'node_capacity_ed_p'


def model_co2(par, s, v, r):
    """Model-wide CO2 emissions per period"""
    # Calculate CO2 emissions in Mt
    total_co2 = (r.emissions_co2_d * par.years_represented_d) / 1000000
    result_multi = total_co2.to_frame(name="CO2 [Mt]")
    return result_multi.reset_index(), result_multi, 'CO2_d'


def group_node(par, s, v, r):
    """Group node results by period and time, then aggregate to period only"""

    results = []
    groups = list(s.groupOutput_node)

    if not groups:
        return results

    # Get time steps
    dt_index = s.dt_realize_dispatch  # Should be MultiIndex with (period, time)

    # Calculate timestep-level results first
    results_dt = []

    for g in groups:
        # Get nodes in this group
        group_nodes = s.group_node[s.group_node['group'].isin([g])]['node'].tolist()

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
        annualized_inflow = group_inflow / period_shares

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
        vre_share = vre_flow_sum / group_inflow.where(group_inflow != 0, pd.NA)

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
        curtailed_vre_share = curtailed_vre / group_inflow.where(group_inflow != 0, pd.NA)

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
    result_multi_dt = result_flat_dt.set_index(['group', 'period', 'time'])[
        ['pdtNodeInflow', 'annualized_inflows', 'vre_share', 'curtailed_vre_share',
         'upward_slack', 'downward_slack']
    ]
    result_multi_dt.columns.name = "parameter"

    results.append((result_flat_dt, result_multi_dt, 'nodeGroup_gdt_p'))

    # Aggregate to period level
    result_multi_d = result_multi_dt.groupby(level=['group', 'period']).sum()

    # For shares, we need to recalculate properly:
    # Sum the numerators and denominators separately, then divide
    inflow_d = result_multi_dt['pdtNodeInflow'].groupby(level=['group', 'period']).sum()
    annualized_d = result_multi_dt['annualized_inflows'].groupby(level=['group', 'period']).sum()

    # For VRE share: need to sum absolute flows and recalculate
    # Since we have vre_share * inflow = vre_flow, we can recover vre_flow
    vre_flow_dt = result_multi_dt['vre_share'] * result_multi_dt['pdtNodeInflow']
    vre_flow_d = vre_flow_dt.groupby(level=['group', 'period']).sum()
    vre_share_d = vre_flow_d / inflow_d.where(inflow_d != 0, pd.NA)

    # For curtailed VRE share: similar recovery
    curtailed_flow_dt = result_multi_dt['curtailed_vre_share'] * result_multi_dt['pdtNodeInflow']
    curtailed_flow_d = curtailed_flow_dt.groupby(level=['group', 'period']).sum()
    curtailed_vre_share_d = curtailed_flow_d / inflow_d.where(inflow_d != 0, pd.NA)

    # Slack shares: sum absolute values and recalculate
    upward_slack_d = result_multi_dt['upward_slack'].groupby(['group', 'period']).sum()
    upward_slack_d = upward_slack_d.div(annualized_d.where(annualized_d != 0, pd.NA))

    downward_slack_d = result_multi_dt['downward_slack'].groupby(['group', 'period']).sum()
    downward_slack_d = downward_slack_d.div(annualized_d.where(annualized_d != 0, pd.NA))

    # Combine period-level results
    result_multi_d = pd.DataFrame({
        'sum_annualized_inflows': annualized_d,
        'vre_share': vre_share_d.fillna(0),
        'curtailed_vre_share': curtailed_vre_share_d.fillna(0),
        'upward_slack': upward_slack_d.fillna(0),
        'downward_slack': downward_slack_d.fillna(0)
    })
    result_multi_d.columns.name = "parameter"

    result_flat_d = result_multi_d.reset_index()

    results.append((result_flat_d, result_multi_d, 'nodeGroup_gd_p'))

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


def group_node_VRE_share(par, s, v, r):
    """VRE share for node groups by period and time"""

    results = []

    # Get timesteps and groups
    timesteps = list(s.dt_realize_dispatch)
    
    # Filter groups that have nodes with inflow
    groups_with_inflow = []
    for g in s.groupOutput_node:
        # Get nodes in this group
        group_nodes_df = s.group_node[s.group_node['group'] == g]
        if not group_nodes_df.empty:
            # Check if any node has inflow (not marked as 'no_inflow')
            if hasattr(s, 'node__inflow_method'):
                no_inflow_nodes = set(n for (n, method) in s.node__inflow_method if method == 'no_inflow')
                has_inflow = any(n not in no_inflow_nodes for n in group_nodes_df['node'])
            else:
                has_inflow = True
            if has_inflow:
                groups_with_inflow.append(g)
    
    if not groups_with_inflow or not timesteps:
        index = pd.MultiIndex.from_tuples([], names=['period', 'time'])
        result_multi = pd.DataFrame(index=index, columns=groups_with_inflow)
        return result_multi.reset_index(), result_multi, 'nodeGroup_VRE_share_gdt'
    
    # Create index
    index = pd.MultiIndex.from_tuples(timesteps, names=['period', 'time'])
    result_multi = pd.DataFrame(index=index, columns=groups_with_inflow, dtype=float)
    
    # Get VRE processes
    vre_processes = s.process_VRE.unique()
    
    # Calculate for each group
    for g in groups_with_inflow:
        # Get nodes in this group with inflow
        group_nodes_df = s.group_node[s.group_node['group'] == g]
        if hasattr(s, 'node__inflow_method'):
            no_inflow_nodes = set(n for (n, method) in s.node__inflow_method if method == 'no_inflow')
            group_nodes = [n for n in group_nodes_df['node'] if n not in no_inflow_nodes]
        else:
            group_nodes = group_nodes_df['node'].tolist()
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
        result_multi[g] = (vre_flow / (-total_inflow)).fillna(0.0)
    
    results.append((result_multi.reset_index(), result_multi, 'nodeGroup_VRE_share_dt_g'))

    # Return period results
    result_multi_d = result_multi.groupby(level='period').mean()
    result_multi_d.columns.name = 'group'
    results.append((result_multi_d.reset_index(), result_multi_d, 'nodeGroup_VRE_share_d_g'))

    return results


def group_process_CO2(par, s, v, r):
    """Annualized CO2 Mt for groups by period"""
    
    # Get periods
    periods = list(s.d_realized_period)
    
    # Filter groups that have processes with CO2 emissions
    co2_processes = set(s.process__commodity__node_co2['process'])
    group_processes_df = pd.DataFrame(s.group_process.tolist(), columns=['group', 'process'])
    groups_with_co2 = group_processes_df[group_processes_df['process'].isin(co2_processes)]['group'].unique().tolist()
    
    if not groups_with_co2 or not periods:
        result_multi = pd.DataFrame(index=pd.Index([], name='period'), columns=groups_with_co2)
        return result_multi.reset_index(), result_multi, 'CO2_d_g'
    
    # Create index
    result_multi = pd.DataFrame(index=pd.Index(periods, name='period'), columns=groups_with_co2, dtype=float)
    
    # Get process-commodity-node tuples
    pcn_tuples = list(zip(s.process__commodity__node_co2['process'], 
                         s.process__commodity__node_co2['commodity'], 
                         s.process__commodity__node_co2['node']))
    
    # Calculate for each group
    for g in groups_with_co2:
        # Get processes in this group that have CO2 emissions
        group_processes = [p for (grp, p) in s.group_process if grp == g]
        
        # Filter CO2 columns for processes in this group
        co2_cols = [(p, c, n) for (p, c, n) in pcn_tuples if p in group_processes]
        
        if co2_cols:
            # Sum emissions across processes and convert to Mt (divide by 1,000,000)
            result_multi[g] = r.process_emissions_co2_d[co2_cols].sum(axis=1) / 1000000
        else:
            result_multi[g] = 0.0
    
    return result_multi.reset_index(), result_multi, 'CO2_d_g'

def group_process_node_flow(par, s, v, r):
    """Flow results for groups by period and time, then aggregate to period only"""

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
    results.append((result_multi_dt.reset_index(), result_multi_dt, 'flow_dt_g'))

    # Aggregate to period level
    result_multi_d = result_multi_dt.groupby(level='period').sum()

    # Divide by period shares to annualize
    result_multi_d = result_multi_d.div(par.complete_period_share_of_year, axis=0)
    result_multi_d.columns.name = 'group'

    # Return period results
    results.append((result_multi_d.reset_index(), result_multi_d, 'flow_d_g'))

    return results


def unit_outputNode(par, s, v, r):
    """Unit output node flow for periods and time, then aggregate to period only"""

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
    results.append((result_multi_dt.reset_index(), result_multi_dt, 'unit_outputNode_dt_ee'))

    # Aggregate to period level
    result_multi_d = result_multi_dt.groupby(level='period').sum()

    # Divide by period shares to annualize
    result_multi_d = result_multi_d.div(par.complete_period_share_of_year, axis=0)

    # Return period results
    results.append((result_multi_d.reset_index(), result_multi_d, 'unit_outputNode_d_ee'))

    return results

def unit_inputNode(par, s, v, r):
    """Unit input node flow for periods and time, then aggregate to period only"""

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
    results.append((result_multi_dt.reset_index(), result_multi_dt, 'unit_inputNode_dt_ee'))

    # Aggregate to period level
    result_multi_d = result_multi_dt.groupby(level='period').sum()

    # Divide by period shares to annualize
    result_multi_d = result_multi_d.div(par.complete_period_share_of_year, axis=0)

    # Return period results
    results.append((result_multi_d.reset_index(), result_multi_d, 'unit_inputNode_d_ee'))

    return results


def connection(par, s, v, r):
    """Connection flow for periods and time, then aggregate to period only"""

    results = []

    if r.connection_dt.empty:
        return results

    # Create connection mapping once (reused for both timestep and period)
    conn_map = s.process_source_sink[
        s.process_source_sink['process'].isin(s.process_connection) &
        s.process_source_sink.apply(lambda row: (row['process'], row['sink']) in s.process_sink, axis=1)
    ].set_index('process')

    # Calculate timestep-level results first
    result_multi_dt = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['connection', 'node_left', 'node_right']))

    for c in r.connection_dt.columns:
        if c in conn_map.index:
            row = conn_map.loc[c]
            result_multi_dt[(c, row['source'], row['sink'])] = r.connection_dt[c]

    # Return timestep results
    results.append((result_multi_dt.reset_index(), result_multi_dt, 'connection_dt_eee'))

    # Aggregate to period level
    result_multi_d = result_multi_dt.groupby(level='period').sum()

    # Divide by period shares to annualize
    result_multi_d = result_multi_d.div(par.complete_period_share_of_year, axis=0)

    # Return period results
    results.append((result_multi_d.reset_index(), result_multi_d, 'connection_d_eee'))

    return results 


def connection_rightward(par, s, v, r):
    """Connection flow to right node for periods and time, then aggregate to period only"""

    results = []

    if r.connection_to_right_node__dt.empty:
        return results

    # Create connection mapping once
    conn_map = s.process_source_sink[
        s.process_source_sink['process'].isin(s.process_connection) &
        s.process_source_sink.apply(lambda row: (row['process'], row['sink']) in s.process_sink, axis=1)
    ].set_index('process')

    # Calculate timestep-level results first
    result_multi_dt = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['process', 'connection', 'node']))

    for c in r.connection_to_right_node__dt.columns:
        if c in conn_map.index:
            row = conn_map.loc[c]
            result_multi_dt[(c, row['source'], row['sink'])] = r.connection_to_right_node__dt[c]

    # Return timestep results
    results.append((result_multi_dt.reset_index(), result_multi_dt, 'connection_rightward_dt_eee'))

    # Aggregate to period level
    result_multi_d = result_multi_dt.groupby(level='period').sum()

    # Divide by period shares to annualize
    result_multi_d = result_multi_d.div(par.complete_period_share_of_year, axis=0)

    # Return period results
    results.append((result_multi_d.reset_index(), result_multi_d, 'connection_rightward_d_eee'))

    return results


def connection_leftward(par, s, v, r):
    """Connection flow to left node for periods and time, then aggregate to period only"""

    results = []

    if r.connection_to_left_node__dt.empty:
        return results

    # Create connection mapping once
    conn_map = s.process_source_sink[
        s.process_source_sink['process'].isin(s.process_connection) &
        s.process_source_sink.apply(lambda row: (row['process'], row['sink']) in s.process_sink, axis=1)
    ].set_index('process')

    # Calculate timestep-level results first
    result_multi_dt = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['process', 'connection', 'node']))

    for c in r.connection_to_left_node__dt.columns:
        if c in conn_map.index:
            row = conn_map.loc[c]
            result_multi_dt[(c, row['sink'], row['source'])] = r.connection_to_left_node__dt[c]

    # Return timestep results
    results.append((result_multi_dt.reset_index(), result_multi_dt, 'connection_leftward_dt_eee'))

    # Aggregate to period level
    result_multi_d = result_multi_dt.groupby(level='period').sum()

    # Divide by period shares to annualize
    result_multi_d = result_multi_d.div(par.complete_period_share_of_year, axis=0)

    # Return period results
    results.append((result_multi_d.reset_index(), result_multi_d, 'connection_leftward_d_eee'))

    return results


def group_flows(par, s, v, r):
    """Group output flows for periods and time, then aggregate to period only"""

    results = []

    if s.groupOutputNodeFlows.empty or s.dt_realize_dispatch.empty:
        return results

    # Calculate timestep-level results first
    result_multi_dt = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['group', 'type', 'item']))

    # Process each group
    for g in s.groupOutputNodeFlows:
        # Slack upward
        if g in r.group_node_up_penalties__dt.columns:
            result_multi_dt[g, 'slack', 'upward'] = r.group_node_up_penalties__dt[g]

        # Unit aggregates (unit to group)
        for col in r.group_output__group_aggregate_Unit_to_group__dt.columns:
            if col[0] == g:
                result_multi_dt[g, 'unit_aggregate', col[1]] = r.group_output__group_aggregate_Unit_to_group__dt[col]

        # Units not in aggregate (unit to node)
        for idx, row in s.group_output__process__unit__to_node_Not_in_aggregate.iterrows():
            if row['group'] == g:
                flow_col = (row['process'], row['unit'], row['node'])
                if flow_col in r.flow_dt.columns:
                    result_multi_dt[g, 'unit', row['process']] = r.flow_dt[flow_col]

        # Connection aggregates
        for col in r.group_output__group_aggregate_Connection__dt.columns:
            if col[0] == g:
                result_multi_dt[g, 'connection', col[1]] = r.group_output__group_aggregate_Connection__dt[col]

        # Connections not in aggregate
        for col in r.group_output__connection_not_in_aggregate__dt.columns:
            if col[0] == g:
                result_multi_dt[g, 'connection', col[1]] = r.group_output__connection_not_in_aggregate__dt[col]

        # Storage flows (negative)
        group_nodes_in_state = s.group_node[
            (s.group_node['group'] == g) &
            s.group_node['node'].isin(s.node_state)
        ]['node']
        for n in group_nodes_in_state:
            if n in r.node_state_change_dt.columns:
                result_multi_dt[g, 'storage_flow', g] = -r.node_state_change_dt[n]

        # Group to unit aggregates
        for col in r.group_output__group_aggregate_Group_to_unit__dt.columns:
            if col[0] == g:
                result_multi_dt[g, 'unit_aggregate', col[1]] = r.group_output__group_aggregate_Group_to_unit__dt[col]

        # Node to unit not in aggregate (negative)
        for idx, row in s.group_output__process__node__to_unit_Not_in_aggregate.iterrows():
            if row['group'] == g:
                flow_col = (row['process'], row['node'], row['unit'])
                if flow_col in r.flow_dt.columns:
                    result_multi_dt[g, 'unit', row['process']] = -r.flow_dt[flow_col]

        # Internal losses (negative)
        if g in r.group_output_Internal_connection_losses__dt.columns:
            result_multi_dt[g, 'internal_losses', 'connections'] = -r.group_output_Internal_connection_losses__dt[g]
        if g in r.group_output_Internal_unit_losses__dt.columns:
            result_multi_dt[g, 'internal_losses', 'units'] = -r.group_output_Internal_unit_losses__dt[g]
        if g in r.group_node_state_losses__dt.columns:
            result_multi_dt[g, 'internal_losses', 'storages'] = -r.group_node_state_losses__dt[g]

        # Slack downward
        if g in r.group_node_down_penalties__dt.columns:
            result_multi_dt[g, 'slack', 'downward'] = r.group_node_down_penalties__dt[g]

        # Inflow
        if g in r.group_node_inflow_dt.columns:
            result_multi_dt[g, 'inflow', g] = r.group_node_inflow_dt[g]

    result_multi_dt.columns.names = ['group', 'type', 'item']

    # Return timestep results
    results.append((result_multi_dt.reset_index(), result_multi_dt, 'nodeGroup_flows_dt_gpe'))

    # Aggregate to period level
    result_multi_d = result_multi_dt.groupby(level='period').sum()

    # Return period results
    results.append((result_multi_d.reset_index(), result_multi_d, 'nodeGroup_flows_d_gpe'))

    return results

def connection_cf(par, s, v, r):
    """Connection capacity factors for periods"""
    if r.process_sink_flow_d.empty:
        result_multi = pd.DataFrame(index=pd.Index([], name='period'))
        return result_multi.reset_index(), result_multi, 'connection_cf_d_e'

    complete_hours = par.complete_period_share_of_year * 8760
    connection_cols = r.process_sink_flow_d.columns[r.process_sink_flow_d.columns.get_level_values(0).isin(s.process_connection)]
    connection_capacity = r.entity_all_capacity[connection_cols.droplevel(1).unique()].rename_axis('process', axis=1)
    connection_capacity.columns = connection_capacity.columns.get_level_values(0)
    result_multi = r.connection_dt.abs().groupby('period').sum().div(connection_capacity, level=0).div(complete_hours, axis=0)
    result_multi.columns.names = ['connection']
    return result_multi.reset_index(), result_multi, 'connection_cf_d_e'

def unit_cf_outputNode(par, s, v, r):
    """Unit capacity factors by output node for periods"""
    if r.process_sink_flow_d.empty:
        result_multi = pd.DataFrame(index=pd.Index([], name='period'))
        return result_multi.reset_index(), result_multi, 'unit_outputs_cf_d_ee'

    complete_hours = par.complete_period_share_of_year * 8760
    unit_cols = r.process_sink_flow_d.columns[r.process_sink_flow_d.columns.get_level_values(0).isin(s.process_unit)]
    unit_capacity = r.entity_all_capacity[unit_cols.droplevel(1).unique()].rename_axis('process', axis=1)
    unit_capacity.columns = unit_capacity.columns.get_level_values(0)
    result_multi = r.process_sink_flow_d[unit_cols].div(unit_capacity, level=0).div(complete_hours, axis=0)
    result_multi.columns.names = ['unit', 'sink']
    return result_multi.reset_index(), result_multi, 'unit_outputs_cf_d_ee'

def unit_cf_inputNode(par, s, v, r):
    """Unit capacity factors by input node for periods"""
    if r.process_source_flow_d.empty:
        result_multi = pd.DataFrame(index=pd.Index([], name='period'))
        return result_multi.reset_index(), result_multi, 'unit_inputs_cf_d_ee'
    # !!! This should account for efficiency losses in direct conversion units (but it does not)
    complete_hours = par.complete_period_share_of_year * 8760
    unit_source = r.process_source_flow_d.columns[r.process_source_flow_d.columns.get_level_values(0).isin(s.process_unit)]
    unit_capacity = r.entity_all_capacity[unit_source.droplevel(1).unique()].rename_axis('process', axis=1)
    unit_capacity.columns = unit_capacity.columns.get_level_values(0)
    result_multi = r.process_source_flow_d[unit_source].div(unit_capacity, level=0).div(complete_hours, axis=0)
    result_multi.columns.names = ['unit', 'source']
    return result_multi.reset_index(), result_multi, 'unit_inputs_cf_d_ee'


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
        
        results.append((curtail_dt.reset_index(), curtail_dt, 'unit_curtailment_outputNode_dt_ee'))
        results.append((potential_dt.reset_index(), potential_dt, 'unit_VRE_potential_outputNode_dt_ee'))
        
        # Calculate curtailment share at timestep level
        curtail_share_dt = (curtail_dt / potential_dt).where(potential_dt != 0, 0)
        results.append((curtail_share_dt.reset_index(), curtail_share_dt, 'unit_curtailment_share_outputNode_dt_ee'))
        
        # Aggregate to period level
        curtail_period = curtail_dt.groupby(level='period').sum()
        potential_period = potential_dt.groupby(level='period').sum()
        
        # Calculate curtailment share at period level
        curtail_share_period = (curtail_period / potential_period).where(potential_period != 0, 0)

        results.append((curtail_share_period.reset_index(), curtail_share_period, 'unit_curtailment_share_outputNode_d_ee'))
        results.append((potential_period.reset_index(), potential_period, 'unit_VRE_potential_outputNode_d_ee'))
    
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
    results.append((ramp_output.reset_index(), ramp_output, 'unit_ramp_outputs_dt_ee'))
    
    # Input node ramps
    process_source_ramp_input = s.process_source[s.process_source.get_level_values(0).isin(s.process_unit)]
    pss_ramp_input = r.ramp_dtt.columns[r.ramp_dtt.columns.droplevel([1, 2]).isin(process_source_ramp_input)]
    ramp_input = r.ramp_dtt[pss_ramp_input].droplevel('t_previous')
    ramp_input.columns = ramp_input.columns.droplevel(1)  # Remove 'source' from (process, source, sink) to get (unit, source)
    ramp_input.columns.names = ['unit', 'source']
    results.append((ramp_input.reset_index(), ramp_input, 'unit_ramp_inputNode_dt_ee'))

    return results

def cost_summaries(par, s, v, r):
    """Cost summaries for periods and timesteps"""
    
    results = []
    
    # Common calculations
    discount_ops = par.discount_factor_operations_yearly
    period_share = par.complete_period_share_of_year
    to_millions = 1000000
    
    # 1. Costs at timestep level (non-annualized)
    costs_dt = pd.DataFrame(index=s.dt_realize_dispatch, dtype=float)
    costs_dt['commodity'] = r.cost_commodity_dt.sum(axis=1)
    costs_dt['co2'] = r.cost_co2_dt
    costs_dt['other_operational'] = r.cost_process_other_operational_cost_dt.sum(axis=1)
    costs_dt['starts'] = r.cost_startup_dt.sum(axis=1)
    costs_dt['upward_slack_penalty'] = r.costPenalty_node_state_upDown_dt.xs('up', level='upDown', axis=1).sum(axis=1)
    costs_dt['downward_slack_penalty'] = r.costPenalty_node_state_upDown_dt.xs('down', level='upDown', axis=1).sum(axis=1)
    costs_dt['inertia_slack_penalty'] = r.costPenalty_inertia_dt.sum(axis=1)
    costs_dt['non_synchronous_slack_penalty'] = r.costPenalty_non_synchronous_dt.sum(axis=1)
    try:
        costs_dt['upward_reserve_slack_penalty'] = r.costPenalty_reserve_upDown_dt.xs('up', level='updown', axis=1).sum(axis=1)
    except KeyError:
        costs_dt['upward_reserve_slack_penalty'] = 0
    try:
        costs_dt['downward_reserve_slack_penalty'] = r.costPenalty_reserve_upDown_dt.xs('down', level='updown', axis=1).sum(axis=1)
    except KeyError:
        costs_dt['downward_reserve_slack_penalty'] = 0
    
    results.append((costs_dt.reset_index(), costs_dt, 'costs_dt_p'))
    
    # 2. Annualized dispatch costs (derived from costs_dt)
    dispatch_period = costs_dt.groupby(level='period').sum()
    dispatch_period = dispatch_period.div(period_share, axis=0) / to_millions
    
    # results.append((dispatch_period.reset_index(), dispatch_period, 'annualized_dispatch_costs_dt', 'tbl_annualized_dispatch_costs_period_t'))
    
    # 3. Annualized investment costs (d_realize_invest only)
    investment_costs = pd.DataFrame(index=s.d_realize_invest, dtype=float)
    investment_costs['unit_investment_retirement'] = (r.costInvestUnit_d + r.costDivestUnit_d) / discount_ops / to_millions
    investment_costs['connection_investment_retirement'] = (r.costInvestConnection_d + r.costDivestConnection_d) / discount_ops / to_millions
    investment_costs['storage_investment_retirement'] = (r.costInvestState_d + r.costDivestState_d) / discount_ops / to_millions
    investment_costs['fixed_cost_existing'] = r.costExistingFixed_d / discount_ops / to_millions
    investment_costs['capacity_margin_penalty'] = r.costPenalty_capacity_margin_d.sum(axis=1) / discount_ops / to_millions
    
    # results.append((investment_costs.reset_index(), investment_costs, 'annualized_investment_costs_d', 'tbl_annualized_investment_costs_period'))
    
    # 4. Combined summary (investment + dispatch aggregated to period)
    all_periods = s.d_realized_period.union(s.d_realize_invest)
    summary = pd.DataFrame(index=all_periods, dtype=float)
    summary.columns.name = 'parameter'    
    
    # Investment costs (only for d_realize_invest)
    for col in investment_costs.columns:
        summary[col] = investment_costs[col].reindex(all_periods, fill_value=0)
    
    # Dispatch costs (only for d_realized_period)
    for col in dispatch_period.columns:
        summary[col] = dispatch_period[col].reindex(all_periods, fill_value=0)
    
    results.append((summary.reset_index(), summary, 'annualized_costs_d_p'))
    
    return results

def reserves(par, s, v, r):
    """Process reserves for timesteps and periods"""
    
    results = []
    if v.reserve.empty:
        return results
    
    # Timestep-level reserves
    reserves_dt = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['entity_type', 'process', 'reserve', 'updown', 'node']), dtype=float)
    for col in v.reserve.columns:
        p, r_type, ud, n = col
        entity_type = 'unit' if p in s.process_unit else 'connection'
        unitsize = par.entity_unitsize[p]
        reserves_dt[entity_type, p, r_type, ud, n] = v.reserve[col] * unitsize
    results.append((reserves_dt.reset_index(), reserves_dt, 'process_reserve_upDown_node_dt_peppe'))
    
    # Period-level reserves (average)
    step_duration = par.step_duration
    complete_hours = par.complete_period_share_of_year * 8760
    reserves_d = pd.DataFrame(index=s.d_realized_period, columns=reserves_dt.columns, dtype=float)
    
    for col in reserves_dt.columns:
        # Weighted average by step_duration
        weighted = reserves_dt[col] * step_duration
        reserves_d[col] = weighted.groupby(level='period').sum() / complete_hours
    results.append((reserves_d.reset_index(), reserves_d, 'process_reserve_average_d_peppe'))

    # Reserve price results
    if not v.dual_reserve_balance.empty:
        results.append((v.dual_reserve_balance.reset_index(), v.dual_reserve_balance, 'reserve_prices_dt_ppg'))

    return results

def unit_online_and_startup(par, s, v, r):
    """Unit online status and startups for timesteps and periods"""
    
    results = []
    if r.process_online_dt.empty:
        return results
    
    # 1. Online units dt
    online_units_dt = r.process_online_dt[s.process_unit.intersection(s.process_online)]
    results.append((online_units_dt.reset_index(), online_units_dt, 'unit_online_dt_e'))
    
    # 2. Average online status at period level (weighted by step_duration)
    complete_hours = par.complete_period_share_of_year * 8760
    online_units_d = online_units_dt.mul(par.step_duration, axis=0).groupby('period').sum().div(complete_hours, axis=0)
    results.append((online_units_d.reset_index(), online_units_d, 'unit_online_average_d_e'))
    
    # 3. Startups aggregated to period level
    startup_units_d = r.process_startup_dt[s.process_unit.intersection(s.process_online)].groupby('period').sum()
    results.append((startup_units_d.reset_index(), startup_units_d, 'unit_startup_d_e'))
    
    return results

def node_summary(par, s, v, r):
    """Node balance summaries for periods and timesteps"""
    
    results = []
    categories = ['inflow', 'from_units', 'from_connections', 'to_units', 'to_connections', 'state_change', 'self_discharge', 'upward_slack', 'downward_slack']
    nodes = list(s.node)
    
    # 1. Timestep-level node summary
    node_dt = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_product([s.node, categories], names=['node', 'category']), dtype=float)
    
    for n in nodes:
        # Inflow
        if (n, 'no_inflow') not in s.node__inflow_method and n in s.node_balance.union(s.node_balance_period):
            node_dt[n, 'inflow'] = par.node_inflow[(n,)].values
        else:
            node_dt[n, 'inflow'] = 0
        
        # From units
        from_units_cols = [(p, src, snk) for (p, src, snk) in r.flow_dt.columns if snk == n and p in s.process_unit]
        node_dt[n, 'from_units'] = r.flow_dt[from_units_cols].sum(axis=1) if from_units_cols else 0
        
        # From connections
        from_conn_cols = [(p, src, snk) for (p, src, snk) in r.flow_dt.columns if snk == n and p in s.process_connection]
        node_dt[n, 'from_connections'] = r.flow_dt[from_conn_cols].sum(axis=1) if from_conn_cols else 0
        
        # To units (negative)
        to_units_cols = [(p, src, snk) for (p, src, snk) in r.flow_dt.columns if src == n and p in s.process_unit]
        node_dt[n, 'to_units'] = -r.flow_dt[to_units_cols].sum(axis=1) if to_units_cols else 0
        
        # To connections (negative)
        to_conn_cols = [(p, src, snk) for (p, src, snk) in r.flow_dt.columns if src == n and p in s.process_connection]
        node_dt[n, 'to_connections'] = -r.flow_dt[to_conn_cols].sum(axis=1) if to_conn_cols else 0
        
        # State change
        if n in s.node_state and n in r.node_state_change_dt.columns:
            node_dt[n, 'state_change'] = r.node_state_change_dt[n]
        else:
            node_dt[n, 'state_change'] = 0
        
        # Self discharge
        if n in s.node_self_discharge and n in r.self_discharge_loss_dt.columns:
            node_dt[n, 'self_discharge'] = r.self_discharge_loss_dt[n]
        else:
            node_dt[n, 'self_discharge'] = 0
        
        # Upward slack
        if n in s.node_balance.union(s.node_balance_period) and ('node', n) in v.q_state_up.columns:
            capacity_scaling = par.node_capacity_for_scaling[('node', n)]
            node_dt[n, 'upward_slack'] = v.q_state_up[('node', n)] * capacity_scaling
        else:
            node_dt[n, 'upward_slack'] = 0
        
        # Downward slack (negative)
        if n in s.node_balance.union(s.node_balance_period) and ('node', n) in v.q_state_down.columns:
            capacity_scaling = par.node_capacity_for_scaling[('node', n)]
            node_dt[n, 'downward_slack'] = -v.q_state_down[('node', n)] * capacity_scaling
        else:
            node_dt[n, 'downward_slack'] = 0
    
    results.append((node_dt.reset_index(), node_dt, 'node_dt_ep'))
        
 # 2. Period-level node summary
    node_d = pd.DataFrame(index=s.d_realized_period, columns=pd.MultiIndex.from_product([s.node, categories], names=['node', 'category']), dtype=float)
    
    for n in nodes:
        # Inflow
        if (n, 'no_inflow') not in s.node__inflow_method and n in s.node_balance.union(s.node_balance_period):
            node_d[n, 'inflow'] = r.node_inflow_d[(n,)]
        else:
            node_d[n, 'inflow'] = 0
        
        # From units
        from_units_cols = [(p, src, snk) for (p, src, snk) in r.flow_d.columns if snk == n and p in s.process_unit]
        node_d[n, 'from_units'] = r.flow_d[from_units_cols].sum(axis=1) if from_units_cols else 0
        
        # From connections
        from_conn_cols = [(p, src, snk) for (p, src, snk) in r.flow_d.columns if snk == n and p in s.process_connection]
        node_d[n, 'from_connections'] = r.flow_d[from_conn_cols].sum(axis=1) if from_conn_cols else 0
        
        # To units (negative)
        to_units_cols = [(p, src, snk) for (p, src, snk) in r.flow_d.columns if src == n and p in s.process_unit]
        node_d[n, 'to_units'] = -r.flow_d[to_units_cols].sum(axis=1) if to_units_cols else 0
        
        # To connections (negative)
        to_conn_cols = [(p, src, snk) for (p, src, snk) in r.flow_d.columns if src == n and p in s.process_connection]
        node_d[n, 'to_connections'] = -r.flow_d[to_conn_cols].sum(axis=1) if to_conn_cols else 0
        
        # State change
        if n in s.node_state and n in r.node_state_change_d.columns:
            node_d[n, 'state_change'] = r.node_state_change_d[n]
        else:
            node_d[n, 'state_change'] = 0
        
        # Self discharge
        if n in s.node_self_discharge and n in r.self_discharge_loss_d.columns:
            node_d[n, 'self_discharge'] = r.self_discharge_loss_d[n]
        else:
            node_d[n, 'self_discharge'] = 0
        
        # Upward slack
        if n in s.node_balance.union(s.node_balance_period) and (n, 'up') in r.penalty_node_state_upDown_d.columns:
            node_d[n, 'upward_slack'] = r.penalty_node_state_upDown_d[(n, 'up')]
        else:
            node_d[n, 'upward_slack'] = 0
        
        # Downward slack (negative)
        if n in s.node_balance.union(s.node_balance_period) and (n, 'down') in r.penalty_node_state_upDown_d.columns:
            node_d[n, 'downward_slack'] = -r.penalty_node_state_upDown_d[(n, 'down')]
        else:
            node_d[n, 'downward_slack'] = 0
    
    results.append((node_d.reset_index(), node_d, 'node_d_ep'))
    
    return results

def node_additional_results(par, s, v, r):
    """Additional node results: prices, state, and slacks"""
    results = []
    
    # 1. Nodal prices
    if not v.dual_node_balance.empty:
        results.append((v.dual_node_balance.reset_index(), v.dual_node_balance, 'node_prices_dt_e'))
    
    # 2. Node state
    if not v.state.empty:
        node_state = v.state.mul(par.entity_unitsize[s.node_state], level="node")
        results.append((node_state.reset_index(), node_state, 'node_state_dt_e'))
    
    # 3. Node upward slack
    if not v.q_state_up.empty:
        upward_slack = v.q_state_up.mul(par.node_capacity_for_scaling[s.node_balance.union(s.node_balance_period)], level=0)
        results.append((upward_slack.reset_index(), upward_slack, 'node_slack_up_dt_e'))

    # 4. Node downward slack
    if not v.q_state_down.empty:
        downward_slack = v.q_state_down.mul(par.node_capacity_for_scaling[s.node_balance.union(s.node_balance_period)], level=0)
        results.append((downward_slack.reset_index(), downward_slack, 'node_slack_down_dt_e'))
    
    return results

def investment_duals(par, s, v, r):
    """Additional node results: prices, state, and slacks"""
    results = []
    
    # 1. v.dual_invest_unit
    if not v.dual_invest_unit.empty:
        results.append((v.dual_invest_unit.reset_index(), v.dual_invest_unit, 'dual_invest_unit_d_e'))

    # 2. v.dual_invest_connection
    if not v.dual_invest_connection.empty:
        results.append((v.dual_invest_connection.reset_index(), v.dual_invest_connection, 'dual_invest_connection_d_e'))

    # 3. v.dual_invest_node
    if not v.dual_invest_node.empty:
        results.append((v.dual_invest_node.reset_index(), v.dual_invest_node, 'dual_invest_node_d_e'))

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
        s.group_node_inertia = s.group_node[s.group_node['group'].isin([g])]
        s.process_source_inertia = s.process_source[s.process_source.get_level_values(1).isin(s.group_node_inertia['node'])]
        s.pss_inertia = s.process_source_sink_alwaysProcess[s.process_source_sink_alwaysProcess.droplevel(2).isin(s.process_source_inertia)]
        for (p, source, sink) in s.pss_inertia:
            inertia_const = get_inertia_constant(p, source, 'source')
            if inertia_const:
                flow_online = get_flow_or_online(p, source, sink, s.dt_realize_dispatch)
                total_inertia += (flow_online * inertia_const).squeeze()
        
        # Inertia from sinks
        s.process_sink_inertia = s.process_sink[s.process_sink.get_level_values(1).isin(s.group_node_inertia['node'])]
        s.pss_inertia = s.process_source_sink_alwaysProcess[s.process_source_sink_alwaysProcess.droplevel(1).isin(s.process_sink_inertia)]
        for (p, source, sink) in s.pss_inertia:
            inertia_const = get_inertia_constant(p, sink, 'sink')
            if inertia_const:
                flow_online = get_flow_or_online(p, source, sink, s.dt_realize_dispatch)
                total_inertia += (flow_online * inertia_const).squeeze()
        
        group_inertia[g] = total_inertia
    
    results.append((group_inertia.reset_index(), group_inertia, 'nodeGroup_inertia_dt_g'))
    
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
    
    results.append((unit_inertia.reset_index(), unit_inertia, 'nodeGroup_unit_node_inertia_dt_gee'))
    
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
    
    results.append((largest_flow.reset_index(), largest_flow, 'nodeGroup_inertia_largest_flow_dt_g'))
    
    return results

def slack_variables(par, s, v, r):
    """Slack variables for reserves, non-synchronous, inertia, and capacity margin"""
    
    results = []
    
    # 1. Reserve slack variables
    if not v.q_reserve.empty:
        reserve_slack = v.q_reserve * par.reserve_upDown_group_reservation
        # pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['reserve', 'updown', 'node_group']), dtype=float)
        # for col in v.q_reserve.columns:
        #     if col in par.reserve_upDown_group_reservation.columns:
        #         reserve_slack[col] = v.q_reserve[col] * par.reserve_upDown_group_reservation[col]
        results.append((reserve_slack.reset_index(), reserve_slack, 'nodeGroup_slack_reserve_dt_rug'))
    
    # 2. Non-synchronous slack variables
    if not v.q_non_synchronous.empty:
        nonsync_slack = v.q_non_synchronous * par.group_capacity_for_scaling[s.groupNonSync]
        results.append((nonsync_slack.reset_index(), nonsync_slack, 'nodeGroup_slack_nonsync_dt_g'))
    
    # 3. Inertia slack variables
    if not v.q_inertia.empty:
        inertia_slack = v.q_inertia * par.group_inertia_limit
        results.append((inertia_slack.reset_index(), inertia_slack, 'nodeGroup_slack_inertia_dt_g'))
    
    # 4. Capacity margin slack variables (for investment periods only)
    if not v.q_capacity_margin.empty:
        capmargin_slack = v.q_capacity_margin * par.group_capacity_for_scaling[s.groupCapacityMargin]
        results.append((capmargin_slack.reset_index(), capmargin_slack, 'nodeGroup_slack_capacity_margin_d_g'))
    
    return results


def plot_dict_of_dataframes(results_dict, output_dir='.'):
    """
    Plot dataframes from a dictionary according to key suffixes.
    
    Args:
        results_dict: Dictionary of pandas DataFrames
        output_dir: Directory to save PNG files
    """
    
    for key, df in results_dict.items():
        # print(f"Processing {key}...")
        
        if (not df.empty) & (len(df) > 0):
            if key.endswith('_dt'):
                plot_dt_type(df, key, output_dir)
            elif key.endswith('_d'):
                plot_d_type(df, key, output_dir)
            #else:
            #    plot_other_type(df, key, output_dir)
        
        plt.close('all')  # Clean up


def plot_dt_type(df, key, output_dir):
    """Line plot for _dt type: 168 rows, all columns as lines"""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Take first 168 rows
    df_plot = df.iloc[:168]
    
    # Plot each column as a line
    for col in df_plot.columns:
        if isinstance(col, tuple):
            label = ' - '.join(str(c) for c in col)
        else:
            label = str(col)
        ax.plot(range(len(df_plot)), df_plot[col], label=label, alpha=0.7)
    
    # Set x-axis labels to show multi-index
    if isinstance(df_plot.index, pd.MultiIndex):
        # Create labels from multi-index
        x_labels = [' '.join(str(idx) for idx in row) for row in df_plot.index]
        # Show every nth label to avoid overcrowding
        step = max(1, len(x_labels) // 20)
        ax.set_xticks(range(0, len(x_labels), step))
        ax.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), step)], 
                          rotation=45, ha='right')
        
        # Set x-axis label from index names
        xlabel = ' - '.join(str(name) for name in df_plot.index.names if name)
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(df_plot.index.name or 'Index')
    
    # Set y-axis label from column names
    if isinstance(df.columns, pd.MultiIndex):
        ylabel = ' - '.join(str(name) for name in df.columns.names if name)
    else:
        ylabel = df.columns.name or 'Value'
    ax.set_ylabel(ylabel)
    
    ax.set_title(key)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{key}.svg', bbox_inches='tight')


def plot_d_type(df, key, output_dir):
    """Grouped bar plot for _d type: separate bars for columns, grouped by index"""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Transpose so columns become x-axis groups
    df_plot = df.T
    
    # Create x positions for bar groups
    n_groups = len(df_plot.index)  # number of columns
    n_bars = len(df.index)  # number of periods
    
    # Width of each bar and spacing
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)
    
    # Plot grouped bars
    for i, idx_val in enumerate(df.index):
        values = df_plot[idx_val].values
        label = str(idx_val)
        offset = (i - n_bars/2 + 0.5) * bar_width
        ax.bar(x + offset, values, bar_width, label=label)
    
    # Set x-axis labels
    if isinstance(df.columns, pd.MultiIndex):
        x_labels = ['\n'.join(str(c) for c in col) for col in df.columns]
    else:
        x_labels = [str(col) for col in df.columns]
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    # Set labels
    if isinstance(df.columns, pd.MultiIndex):
        xlabel = ' - '.join(str(name) for name in df.columns.names if name)
    else:
        xlabel = df.columns.name or 'Columns'
    ax.set_xlabel(xlabel)
    
    ylabel = df.index.name or 'Value'
    ax.set_ylabel(ylabel)
    
    ax.set_title(key)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{key}.svg', bbox_inches='tight')


def plot_other_type(df, key, output_dir):
    """Subplots (2xN) for other types: one subplot per column, bars for index rows"""
    n_cols = len(df.columns)
    n_rows = int(np.ceil(n_cols / 2))
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 10))
    fig.suptitle(key, fontsize=16)
    
    # Flatten axes array for easier iteration
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    for idx, col in enumerate(df.columns):
        ax = axes_flat[idx]
        
        # Get data for this column
        data = df[col]
        
        # Create bar plot
        x = np.arange(len(data))
        ax.bar(x, data, width=0.8)
        
        # Set x-axis labels
        if isinstance(df.index, pd.MultiIndex):
            x_labels = ['\n'.join(str(i) for i in row) for row in df.index]
        else:
            x_labels = [str(i) for i in df.index]
        
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        
        # Set subplot title (column name)
        if isinstance(col, tuple):
            col_title = '\n'.join(str(c) for c in col)
        else:
            col_title = str(col)
        ax.set_title(col_title, fontsize=10)
        
        # Set x-axis label from index names
        if isinstance(df.index, pd.MultiIndex):
            xlabel = ' - '.join(str(name) for name in df.index.names if name)
        else:
            xlabel = df.index.name or 'Index'
        ax.set_xlabel(xlabel, fontsize=9)
        
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(n_cols, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{key}.svg', bbox_inches='tight')


# List of all output functions
ALL_OUTPUTS = [
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
    model_co2,
    group_node,
    group_node_VRE_share,
    group_process_CO2,
    group_process_node_flow,
    unit_outputNode,
    unit_inputNode,
    connection,
    connection_rightward,
    connection_leftward,
    group_flows,
    connection_cf,
    unit_cf_outputNode,
    unit_cf_inputNode,
    unit_VRE_curtailment_and_potential,
    unit_ramps,
]


# writer.py - handles the actual writing
def write_outputs(scenario_name, output_funcs=None, output_dir='output_raw', methods=['excel', 'db']):
    """
    output_funcs: list of functions to run, or None for ALL_OUTPUTS
    """
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

    start = time.perf_counter()

    par, s, v = read_outputs(output_dir)

    print(f"--- Read flextool outputs: {time.perf_counter() - start:.4f} seconds")
    start = time.perf_counter()

    r = post_process_results(par, s, v)

    print(f"--- Post processed outputs: {time.perf_counter() - start:.4f} seconds")
    start = time.perf_counter()

    # Usage
    # open('namespace_structure.txt', 'w').close()
    # print_namespace_structure(r, 'r')
    # print_namespace_structure(s, 's')
    # print_namespace_structure(v, 'v')
    # print_namespace_structure(par, 'par')

    output_funcs = output_funcs or ALL_OUTPUTS


    results_multi = {}
    results_flat = {}
    for func in output_funcs:
        func_results = func(par, s, v, r)
        
        # Handle both single result (wrapped in list) and multiple results
        if not isinstance(func_results, list):
            func_results = [func_results]
        
        for result_flat, result_multi, table_name in func_results:
            # Use excel_sheet as the key to allow multiple outputs per function
            results_multi[table_name] = result_multi
            results_flat[table_name] = result_flat

    print(f"--- Formatted for output: {time.perf_counter() - start:.4f} seconds")
    start = time.perf_counter()

    # Write to parquet
    for name, df in results_multi.items():
        if name.endswith(('_d_p', '_d_e', '_d_ep', '_d_peppe', '_d_g', '_d', '_gd_p', \
                          '_ed_p', '_d_ee', '_d_eee', '_d_gpe', \
                          'node_slack_up_dt_e', 'unit_outputNode_dt_ee', 'unit_inputNode_dt_ee', \
                          'connection_dt_eee', 'connection_rightward_dt_eee', 'connection_leftward_dt_eee', \
                          'flow_dt_g', 'unit_curtailment_outputNode_dt_ee')):
            if not os.path.exists('output_parquet'):
                os.makedirs('output_parquet')
            df = pd.concat({scenario_name: df}, axis=1, names=['scenario'])
            df.to_parquet(f'output_parquet/{name}.parquet')

    print(f"--- Wrote to parquet: {time.perf_counter() - start:.4f} seconds")
    start = time.perf_counter()

    plot_dir = './output_plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_dict_of_dataframes(results_multi, output_dir=plot_dir)

    print(f"--- Plotted figures: {time.perf_counter() - start:.4f} seconds")
    start = time.perf_counter()

    # Write to excel
    if 'excel' in methods:
        with pd.ExcelWriter('output' + scenario_name + '.xlsx') as writer:
            for name, df in results_flat.items():
                if (not df.empty) & (len(df) > 0):
                    df.to_excel(writer, sheet_name=name)

    print(f"Wrote to Excel: {time.perf_counter() - start:.4f} seconds")
    start = time.perf_counter()



if __name__ == "__main__":
    write_outputs("foo")