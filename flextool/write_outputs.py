import csv
import pandas as pd
from pathlib import Path
from flextool.read_flextool_outputs import read_variables, read_parameters, read_sets
from flextool.process_results import post_process_results


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
        index = pd.MultiIndex.from_product([periods, processes], names=['period', 'unit'])
    else:
        index = pd.Index(periods, name='period')
    result_multi = pd.DataFrame(index=index)
    
    # Existing capacity - filter to process_unit only
    existing = par.entity_all_existing.droplevel(0, axis=1)[processes].stack()
    existing.index.names = ['period', 'unit']
    result_multi['existing'] = existing
    
    # Invested capacity - default to None, overwrite if data exists
    result_multi['invested'] = pd.Series(dtype=float)
    if not v.invest.empty and len(v.invest.columns) > 0:
        invested = v.invest.loc[:, v.invest.columns.get_level_values(1).isin(s.process_unit)]
        invested = invested.droplevel(0, axis=1).stack()
        invested.index.names = ['period', 'unit']
        result_multi['invested'] = invested
    
    # Divested capacity - default to None, overwrite if data exists
    result_multi['divested'] = pd.Series(dtype=float)
    if not v.divest.empty and len(v.divest.columns) > 0:
        divested = v.divest.loc[:, v.divest.columns.get_level_values(1).isin(s.process_unit)]
        divested = divested.droplevel(0, axis=1).stack()
        divested.index.names = ['period', 'unit']
        result_multi['divested'] = divested
    
    # Total capacity - filter to process_unit only
    total = r.entity_all_capacity.droplevel(0, axis=1)[processes].stack()
    total.index.names = ['period', 'unit']
    result_multi['total'] = total
    result_multi = result_multi[['existing', 'invested', 'divested', 'total']]
    result_flat = result_multi.reset_index()
    return result_flat, result_multi, 'unit_capacity', 'tbl_unit_capacity'


def connection_capacity(par, s, v, r):
    """Connection capacity by period"""
    
    # Get all periods and filter to process_connection entities
    periods = list(s.d_realize_dispatch_or_invest)
    connections = list(s.process_connection)
    
    # Create base dataframe with all combinations (period, connection order)
    if connections:
        index = pd.MultiIndex.from_product([periods, connections], names=['period', 'connection'])
    else:
        index = pd.Index(periods, name='period')
    result_multi = pd.DataFrame(index=index)
    
    # Existing capacity - filter to process_connection only
    existing = par.entity_all_existing.droplevel(0, axis=1)[connections].stack()
    existing.index.names = ['period', 'connection']
    result_multi['existing'] = existing
    
    # Invested capacity - default to empty, overwrite if data exists
    result_multi['invested'] = pd.Series(dtype=float)
    if not v.invest.empty and len(v.invest.columns) > 0:
        invested = v.invest.loc[:, v.invest.columns.get_level_values(1).isin(s.process_connection)]
        invested = invested.droplevel(0, axis=1).stack()
        invested.index.names = ['period', 'connection']
        result_multi['invested'] = invested
    
    # Divested capacity - default to empty, overwrite if data exists
    result_multi['divested'] = pd.Series(dtype=float)
    if not v.divest.empty and len(v.divest.columns) > 0:
        divested = v.divest.loc[:, v.divest.columns.get_level_values(1).isin(s.process_connection)]
        divested = divested.droplevel(0, axis=1).stack()
        divested.index.names = ['period', 'connection']
        result_multi['divested'] = divested
    
    # Total capacity - filter to process_connection only
    total = r.entity_all_capacity.droplevel(0, axis=1)[connections].stack()
    total.index.names = ['period', 'connection']
    result_multi['total'] = total
    
    # Reorder columns
    result_multi = result_multi[['existing', 'invested', 'divested', 'total']]
    
    # Flatten for Excel
    result_flat = result_multi.reset_index()
    
    return result_flat, result_multi, 'connection_capacity', 'tbl_connection_capacity'


def node_capacity(par, s, v, r):
    """Node capacity by period"""
    
    # Get all periods and filter to nodeState entities
    periods = list(s.d_realize_dispatch_or_invest)
    nodes = list(s.nodeState)
    
    # Create base dataframe with all combinations (period, node order)
    if nodes:
        index = pd.MultiIndex.from_product([periods, nodes], names=['period', 'node'])
    else:
        index = pd.Index(periods, name='period')
    result_multi = pd.DataFrame(index=index)
    
    # Existing capacity - filter to nodeState only
    if nodes:
        existing = par.entity_all_existing.droplevel(0, axis=1)[nodes].stack()
        existing.index.names = ['period', 'node']
        result_multi['existing'] = existing
    else:
        result_multi['existing'] = pd.Series(dtype=float)
    
    # Invested capacity - default to empty, overwrite if data exists
    result_multi['invested'] = pd.Series(dtype=float)
    if not v.invest.empty and len(v.invest.columns) > 0:
        invested = v.invest.loc[:, v.invest.columns.get_level_values(1).isin(s.nodeState)]
        if not invested.empty:
            invested = invested.droplevel(0, axis=1).stack()
            invested.index.names = ['period', 'node']
            result_multi['invested'] = invested
    
    # Divested capacity - default to empty, overwrite if data exists
    result_multi['divested'] = pd.Series(dtype=float)
    if not v.divest.empty and len(v.divest.columns) > 0:
        divested = v.divest.loc[:, v.divest.columns.get_level_values(1).isin(s.nodeState)]
        if not divested.empty:
            divested = divested.droplevel(0, axis=1).stack()
            divested.index.names = ['period', 'node']
            result_multi['divested'] = divested
    
    # Total capacity - filter to nodeState only
    if nodes:
        total = r.entity_all_capacity.droplevel(0, axis=1)[nodes].stack()
        total.index.names = ['period', 'node']
        result_multi['total'] = total
    else:
        result_multi['total'] = pd.Series(dtype=float)
    
    result_multi = result_multi[['existing', 'invested', 'divested', 'total']]
    result_flat = result_multi.reset_index()
    return result_flat, result_multi, 'node_capacity', 'tbl_node_capacity'


def model_co2(par, s, v, r):
    """Model-wide CO2 emissions"""
    
    # Calculate total CO2 emissions in Mt
    total_co2 = (r.emissions_co2_d * par.years_represented_d).sum() / 1000000
    
    # Add model_co2 parameter if it exists
    if hasattr(par, 'model_co2') and 'CO2 [Mt]' in par.model_co2:
        total_co2 += par.model_co2['CO2 [Mt]']
    
    # Create result dataframe
    result_multi = pd.DataFrame({
        'model_wide': [total_co2]
    }, index=pd.Index(['CO2 [Mt]'], name='param_co2'))
    
    # Flatten for Excel (already flat)
    result_flat = result_multi.reset_index()
    
    return result_flat, result_multi, 'CO2', 'tbl_model_co2'


def group_node_period(par, s, v, r):
    """Group node results by period"""
    
    periods = list(s.d_realized_period)
    groups = list(s.groupOutput_node)
    
    if not groups:
        return pd.DataFrame(), pd.DataFrame(), 'group_node_period', 'tbl_group_node_period'
    
    # Create group-node mapping as DataFrame for easier merging
    group_node_df = pd.DataFrame(s.group_node.to_list(), columns=['group', 'node'])
    group_node_df = group_node_df[group_node_df['group'].isin(groups)]
    
    results = []
    
    for g in groups:
        # Get nodes in this group
        group_nodes = group_node_df[group_node_df['group'] == g]['node'].tolist()
        
        # 1. Sum of annualized inflows [MWh]
        annualized_inflow = r.node_inflow_d[group_nodes].sum(axis=1)
        period_inflow = par.node_inflow.sum(axis=1).groupby('period').sum()
        
        # 2. VRE share of annual inflow. Filter flows to VRE processes in this group's nodes
        vre_processes = s.process_VRE['process'].unique()
        flow_filtered = r.flow_d.loc[:, (slice(None), slice(None), group_nodes)]
        # Select only VRE process columns
        vre_cols = [(p, src, snk) for p, src, snk in flow_filtered.columns 
                    if p in vre_processes and (p, src, snk) in s.process_source_sink_alwaysProcess]
        if vre_cols:
            vre_flow_sum = flow_filtered[vre_cols].sum(axis=1)
        else:
            vre_flow_sum = pd.Series(0, index=flow_filtered.index)
        vre_share = vre_flow_sum / (-period_inflow)
        
        # 3. Curtailed VRE share. Potential VRE generation for this group's nodes
        potential_cols = [(p, n) for p, n in r.potentialVREgen.columns 
                         if n in group_nodes and p in vre_processes]
        if potential_cols:
            potential_sum = r.potentialVREgen[potential_cols].sum(axis=1)
        else:
            potential_sum = pd.Series(0, index=periods)
        curtailed_vre_share = (potential_sum - vre_flow_sum) / (-period_inflow)
        
        # Filter nodes that are in nodeBalance or nodeBalancePeriod
        balance_nodes = [n for n in group_nodes 
                        if n in set(s.nodeBalance) | set(s.nodeBalancePeriod)]
        # 4. Upward slack. 
        if balance_nodes:
            up_cols = [(n, 'up') for n in balance_nodes if (n, 'up') in r.penalty_nodeState_upDown_d.columns]
            if up_cols:
                upward_slack_sum = r.penalty_nodeState_upDown_d[up_cols].sum(axis=1)
            else:
                upward_slack_sum = pd.Series(0, index=periods)
        else:
            upward_slack_sum = pd.Series(0, index=periods)
        upward_slack = upward_slack_sum / (-annualized_inflow)
        
        # 5. Downward slack
        if balance_nodes:
            down_cols = [(n, 'down') for n in balance_nodes if (n, 'down') in r.penalty_nodeState_upDown_d.columns]
            if down_cols:
                downward_slack_sum = r.penalty_nodeState_upDown_d[down_cols].sum(axis=1)
            else:
                downward_slack_sum = pd.Series(0, index=periods)
        else:
            downward_slack_sum = pd.Series(0, index=periods)
        downward_slack = downward_slack_sum / (-annualized_inflow)
        
        # Combine results for this group
        group_result = pd.DataFrame({
            'group': g,
            'period': periods,
            'sum_annualized_inflows': annualized_inflow.values,
            'vre_share': vre_share.values,
            'curtailed_vre_share': curtailed_vre_share.values,
            'upward_slack': upward_slack.values,
            'downward_slack': downward_slack.values
        })
        results.append(group_result)
    
    # Combine all groups
    result_flat = pd.concat(results, ignore_index=True)
    
    # Create multi-index version
    result_multi = result_flat.set_index(['group', 'period'])[
        ['sum_annualized_inflows', 'vre_share', 'curtailed_vre_share', 
         'upward_slack', 'downward_slack']
    ]
    
    return result_flat, result_multi, 'group_node_period', 'tbl_group_node_period'

def group_node_period_time(par, s, v, r):
    """Group node results by period and time"""
    
    groups = list(s.groupOutput_node)
    
    if not groups:
        return pd.DataFrame(), pd.DataFrame(), 'group_node_period_time', 'tbl_group_node_period_time'
    
    # Create group-node mapping
    group_node_df = pd.DataFrame(s.group_node.to_list(), columns=['group', 'node'])
    group_node_df = group_node_df[group_node_df['group'].isin(groups)]
    
    # Get time steps
    dt_index = s.dt_realize_dispatch  # Should be MultiIndex with (period, time)
    
    results = []
    
    for g in groups:
        # Get nodes in this group
        group_nodes = group_node_df[group_node_df['group'] == g]['node'].tolist()
        
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
        # Need to divide by complete_period_share_of_year for each period
        # Align complete_period_share_of_year with dt_index
        period_shares = group_inflow.index.get_level_values('period').map(
            lambda p: par.complete_period_share_of_year[p]
        )
        annualized_inflow = group_inflow / period_shares
        
        # 3. VRE share (actual flow)
        vre_processes = s.process_VRE['process'].unique()
        # Filter flows to VRE processes in this group's nodes
        flow_filtered = r.flow_dt.loc[:, (slice(None), slice(None), group_nodes)]
        vre_cols = [(p, src, snk) for p, src, snk in flow_filtered.columns 
                    if p in vre_processes and (p, src, snk) in s.process_source_sink_alwaysProcess]
        if vre_cols:
            vre_flow_sum = flow_filtered[vre_cols].sum(axis=1) / group_inflow
        else:
            vre_flow_sum = pd.Series(0, index=flow_filtered.index)
        
        # 4. Curtailed VRE share
        # Potential VRE generation for this group's nodes
        potential_cols = [(p, n) for p, n in r.potentialVREgen_dt.columns 
                         if n in group_nodes and p in vre_processes]
        if potential_cols:
            potential_sum = r.potentialVREgen_dt[potential_cols].sum(axis=1)
        else:
            potential_sum = pd.Series(0, index=dt_index)
        curtailed_vre = (potential_sum - vre_flow_sum) / group_inflow
        
        balance_nodes = [n for n in group_nodes 
                        if n in set(s.nodeBalance) | set(s.nodeBalancePeriod)]
        # 5. Upward slack
        if balance_nodes and not v.q_state_up.empty:
            up_cols = [n for n in balance_nodes if n in v.q_state_up.columns]
            if up_cols:
                upward_slack = (v.q_state_up[up_cols] * par.node_capacity_for_scaling[up_cols]).sum(axis=1)
            else:
                upward_slack = pd.Series(0, index=dt_index)
        else:
            upward_slack = pd.Series(0, index=dt_index)
        
        # 6. Downward slack
        if balance_nodes and not v.q_state_down.empty:
            down_cols = [n for n in balance_nodes if n in v.q_state_down.columns]
            if down_cols:
                downward_slack = (v.q_state_down[down_cols] * par.node_capacity_for_scaling[down_cols]).sum(axis=1)
            else:
                downward_slack = pd.Series(0, index=dt_index)
        else:
            downward_slack = pd.Series(0, index=dt_index)

        # Combine results for this group
        group_result = pd.DataFrame({
            'group': g,
            'period': dt_index.get_level_values('period'),
            'time': dt_index.get_level_values('time'),
            'pdtNodeInflow': group_inflow.values,
            'sum_annualized_inflows': annualized_inflow.values,
            'vre_share': vre_flow_sum.values,
            'curtailed_vre_share': curtailed_vre.values,
            'upward_slack': upward_slack.fillna(0).values,
            'downward_slack': downward_slack.fillna(0).values
        })
        results.append(group_result)
    
    # Combine all groups
    result_flat = pd.concat(results, ignore_index=True)
    
    # Create multi-index version
    result_multi = result_flat.set_index(['group', 'period', 'time'])[
        ['pdtNodeInflow', 'sum_annualized_inflows', 'vre_share', 'curtailed_vre_share', 
         'upward_slack', 'downward_slack']
    ]
    
    return result_flat, result_multi, 'group_node_period_time', 'tbl_group_node_period_time'


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
                print(f"Type: DataFrame")
                print(f"Shape: {obj.shape}")
                print(f"Index: {obj.index.names if hasattr(obj.index, 'names') else 'default'}")
                print(f"Columns: {format_list(obj.columns)}")
                print(f"Dtypes:\n{obj.dtypes}")
                
            elif isinstance(obj, pd.Series):
                print(f"Type: Series")
                print(f"Shape: {obj.shape}")
                print(f"Index: {obj.index.name or 'default'}")
                print(f"Dtype: {obj.dtype}")
                
            elif isinstance(obj, pd.Index):
                print(f"Type: Index")
                print(f"Name: {obj.name}")
                print(f"Values: {format_list(obj)}")
                print(f"Dtype: {obj.dtype}")
        
        sys.stdout = original_stdout


def group_node_VRE_share(par, s, v, r):
    """VRE share for node groups by period and time"""
    
    # Get timesteps and groups
    timesteps = list(s.dt_realize_dispatch)
    
    # Filter groups that have nodes with inflow
    groups_with_inflow = []
    for g in s.groupOutput_node:
        has_inflow = any((grp, n) in s.group_node and (n, 'no_inflow') not in s.node__inflow_method 
                        for (grp, n) in s.group_node if grp == g)
        if has_inflow:
            groups_with_inflow.append(g)
    
    if not groups_with_inflow or not timesteps:
        index = pd.MultiIndex.from_tuples([], names=['period', 'time'])
        result_multi = pd.DataFrame(index=index, columns=groups_with_inflow)
        return result_multi.reset_index(), result_multi, 'group_node_VRE_share', 'tbl_group_node_VRE_share'
    
    # Create index
    index = pd.MultiIndex.from_tuples(timesteps, names=['period', 'time'])
    result_multi = pd.DataFrame(index=index, columns=groups_with_inflow, dtype=float)
    
    # Get VRE processes
    vre_processes = set(s.process_VRE['process'])
    
    # Calculate for each group
    for g in groups_with_inflow:
        # Get nodes in this group with inflow
        group_nodes = [n for (grp, n) in s.group_node 
                      if grp == g and (n, 'no_inflow') not in s.node__inflow_method]
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
    
    return result_multi.reset_index(), result_multi, 'group_node_VRE_share', 'tbl_group_node_VRE_share'

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
        return result_multi.reset_index(), result_multi, 'group_process_CO2', 'tbl_group_process_CO2'
    
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
    
    return result_multi.reset_index(), result_multi, 'group_process_CO2', 'tbl_group_process_CO2'

def group_process_node_flow_period(par, s, v, r):
    """Flow results for groups by period"""
    
    # Get periods
    periods = list(s.d_realized_period)
    groups = list(s.groupOutput_process)
    
    if not groups or not periods:
        result_multi = pd.DataFrame(index=pd.Index([], name='period'), columns=groups)
        return result_multi.reset_index(), result_multi, 'group_process_node_flow_period', 'tbl_group_process_node_flow_period'
    
    # Create index
    result_multi = pd.DataFrame(index=pd.Index(periods, name='period'), columns=groups, dtype=float)
    
    # Get period shares
    period_shares = par.complete_period_share_of_year
    
    # Calculate for each group
    for g in groups:
        # Flows into nodes (process -> node sink)
        sink_cols = [(p, src, snk) for (p, src, snk) in s.process_source_sink_alwaysProcess 
                     if (g, p, snk) in s.group_process_node]
        
        # Flows from nodes (node source -> process)
        source_cols = [(p, src, snk) for (p, src, snk) in s.process_source_sink_alwaysProcess 
                       if (g, p, src) in s.group_process_node]
        
        inflow = r.flow_d[sink_cols].sum(axis=1) if sink_cols else 0
        outflow = r.flow_d[source_cols].sum(axis=1) if source_cols else 0
        
        result_multi[g] = (inflow - outflow) / period_shares
    
    return result_multi.reset_index(), result_multi, 'group_process_node_flow_period', 'tbl_group_process_node_flow_period'

def group_process_node_flow_dt(par, s, v, r):
    """Flow results for groups by period and time"""
    
    # Get timesteps
    timesteps = list(s.dt_realize_dispatch)
    groups = list(s.groupOutput_process)
    
    if not groups or not timesteps:
        index = pd.MultiIndex.from_tuples([], names=['period', 'time'])
        result_multi = pd.DataFrame(index=index, columns=groups)
        return result_multi.reset_index(), result_multi, 'group_process_node_flow_dt', 'tbl_group_process_node_flow_dt'
    
    # Create index
    index = pd.MultiIndex.from_tuples(timesteps, names=['period', 'time'])
    result_multi = pd.DataFrame(index=index, columns=groups, dtype=float)
    
    # Calculate for each group
    for g in groups:
        # Flows into nodes (process -> node sink)
        sink_cols = [(p, src, snk) for (p, src, snk) in s.process_source_sink_alwaysProcess 
                     if (g, p, snk) in s.group_process_node]
        
        # Flows from nodes (node source -> process)
        source_cols = [(p, src, snk) for (p, src, snk) in s.process_source_sink_alwaysProcess 
                       if (g, p, src) in s.group_process_node]
        
        inflow = r.flow_dt[sink_cols].sum(axis=1) if sink_cols else 0
        outflow = r.flow_dt[source_cols].sum(axis=1) if source_cols else 0
        
        result_multi[g] = inflow - outflow
    
    return result_multi.reset_index(), result_multi, 'group_process_node_flow_dt', 'tbl_group_process_node_flow_dt'

def unit_outputNode_period(par, s, v, r):
    """Unit output node flow for periods"""
    if r.process_sink_flow_d.empty:
        result_multi = pd.DataFrame(index=pd.Index([], name='period'))
        return result_multi.reset_index(), result_multi, 'unit_outputNode_period', 'tbl_unit_outputNode_period'
    result_multi = pd.DataFrame(index=s.d_realized_period, columns=pd.MultiIndex.from_tuples([], names=['unit', 'node']))
    for col in r.process_sink_flow_d.columns:
        if col[0] in s.process_unit:
            result_multi[col] = r.process_sink_flow_d[col] / par.complete_period_share_of_year
    return result_multi.reset_index(), result_multi, 'unit_outputNode_period', 'tbl_unit_outputNode_period'

def unit_outputNode_dt(par, s, v, r):
    """Unit output node flow for time"""
    if r.flow_dt.empty:
        result_multi = pd.DataFrame(index=s.dt_realize_dispatch)
        return result_multi.reset_index(), result_multi, 'unit_outputNode_dt', 'tbl_unit_outputNode_dt'
    result_multi = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['unit', 'node']))
    for col in r.flow_dt.columns:
        u, source, sink = col
        if (u, sink) in s.process_sink and u in s.process_unit:
            result_multi[(u, sink)] = r.flow_dt[col]
    return result_multi.reset_index(), result_multi, 'unit_outputNode_dt', 'tbl_unit_outputNode_dt'

def unit_inputNode_period(par, s, v, r):
    """Unit input node flow for periods"""
    if r.process_source_flow_d.empty:
        result_multi = pd.DataFrame(index=pd.Index([], name='period'))
        return result_multi.reset_index(), result_multi, 'unit_inputNode_period', 'tbl_unit_inputNode_period'
    result_multi = pd.DataFrame(index=s.d_realized_period, columns=pd.MultiIndex.from_tuples([], names=['unit', 'node']))
    for col in r.process_source_flow_d.columns:
        if col[0] in s.process_unit:
            result_multi[col] = -r.process_source_flow_d[col] / par.complete_period_share_of_year
    return result_multi.reset_index(), result_multi, 'unit_inputNode_period', 'tbl_unit_inputNode_period'

def unit_inputNode_dt(par, s, v, r):
    """Unit input node flow for time"""
    if r.flow_dt.empty:
        result_multi = pd.DataFrame(index=s.dt_realize_dispatch)
        return result_multi.reset_index(), result_multi, 'unit_inputNode_dt', 'tbl_unit_inputNode_dt'
    result_multi = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['unit', 'node']))
    for col in r.flow_dt.columns:
        u, source, sink = col
        if (u, source) in s.process_source and u in s.process_unit:
            result_multi[(u, source)] = -r.flow_dt[col]
    return result_multi.reset_index(), result_multi, 'unit_inputNode_dt', 'tbl_unit_inputNode_dt'


def connection_period(par, s, v, r):
    """Connection flow for periods"""
    if r.connection_d.empty:
        result_multi = pd.DataFrame(index=pd.Index([], name='period'))
        return result_multi.reset_index(), result_multi, 'connection_period', 'tbl_connection_period'
    result_multi = pd.DataFrame(index=s.d_realized_period, columns=pd.MultiIndex.from_tuples([], names=['connection', 'node_left', 'node_right']))
    conn_map = s.process_source_sink[
        s.process_source_sink['process'].isin(s.process_connection) &
        s.process_source_sink.apply(lambda row: (row['process'], row['sink']) in s.process_sink, axis=1)
    ].set_index('process')
    for c in r.connection_d.columns:
        if c in conn_map.index:
            row = conn_map.loc[c]
            result_multi[c, row['source'], row['sink']] = r.connection_d[c] / par.complete_period_share_of_year
    return result_multi.reset_index(), result_multi, 'connection_period', 'tbl_connection_period'


def connection_dt(par, s, v, r):
    """Connection flow for time"""
    if r.connection_dt.empty:
        result_multi = pd.DataFrame(index=s.dt_realize_dispatch)
        return result_multi.reset_index(), result_multi, 'connection_dt', 'tbl_connection_dt'
    result_multi = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['connection', 'node_left', 'node_right']))
    conn_map = s.process_source_sink[
        s.process_source_sink['process'].isin(s.process_connection) &
        s.process_source_sink.apply(lambda row: (row['process'], row['sink']) in s.process_sink, axis=1)
    ].set_index('process')
    for c in r.connection_dt.columns:
        if c in conn_map.index:
            row = conn_map.loc[c]
            result_multi[(c, row['source'], row['sink'])] = r.connection_dt[c]
    return result_multi.reset_index(), result_multi, 'connection_dt', 'tbl_connection_dt'


def connection_to_right_node_period(par, s, v, r):
    """Connection flow to right node for periods"""
    if r.connection_to_right_node__d.empty:
        result_multi = pd.DataFrame(index=pd.Index([], name='period'))
        return result_multi.reset_index(), result_multi, 'connection_to_right_node_period', 'tbl_connection_to_right_node_period'
    result_multi = pd.DataFrame(index=s.d_realized_period, columns=pd.MultiIndex.from_tuples([], names=['process', 'connection', 'node']))
    conn_map = s.process_source_sink[
        s.process_source_sink['process'].isin(s.process_connection) &
        s.process_source_sink.apply(lambda row: (row['process'], row['sink']) in s.process_sink, axis=1)
    ].set_index('process')
    for c in r.connection_to_right_node__d.columns:
        if c in conn_map.index:
            row = conn_map.loc[c]
            result_multi[(c, row['source'], row['sink'])] = r.connection_to_right_node__d[c] / par.complete_period_share_of_year
    return result_multi.reset_index(), result_multi, 'connection_to_right_node_period', 'tbl_connection_to_right_node_period'


def connection_to_right_node_dt(par, s, v, r):
    """Connection flow to right node for time"""
    if r.connection_to_right_node__dt.empty:
        result_multi = pd.DataFrame(index=s.dt_realize_dispatch)
        return result_multi.reset_index(), result_multi, 'connection_to_right_node_dt', 'tbl_connection_to_right_node_dt'
    result_multi = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['process', 'connection', 'node']))
    conn_map = s.process_source_sink[
        s.process_source_sink['process'].isin(s.process_connection) &
        s.process_source_sink.apply(lambda row: (row['process'], row['sink']) in s.process_sink, axis=1)
    ].set_index('process')
    for c in r.connection_to_right_node__dt.columns:
        if c in conn_map.index:
            row = conn_map.loc[c]
            result_multi[(c, row['source'], row['sink'])] = r.connection_to_right_node__dt[c]
    return result_multi.reset_index(), result_multi, 'connection_to_right_node_dt', 'tbl_connection_to_right_node_dt'


def connection_to_left_node_period(par, s, v, r):
    """Connection flow to left node for periods"""
    if r.connection_to_left_node__d.empty:
        result_multi = pd.DataFrame(index=pd.Index([], name='period'))
        return result_multi.reset_index(), result_multi, 'connection_to_left_node_period', 'tbl_connection_to_left_node_period'
    result_multi = pd.DataFrame(index=s.d_realized_period, columns=pd.MultiIndex.from_tuples([], names=['process', 'connection', 'node']))
    conn_map = s.process_source_sink[
        s.process_source_sink['process'].isin(s.process_connection) &
        s.process_source_sink.apply(lambda row: (row['process'], row['sink']) in s.process_sink, axis=1)
    ].set_index('process')
    for c in r.connection_to_left_node__d.columns:
        if c in conn_map.index:
            row = conn_map.loc[c]
            result_multi[(c, row['sink'], row['source'])] = r.connection_to_left_node__d[c] / par.complete_period_share_of_year
    return result_multi.reset_index(), result_multi, 'connection_to_left_node_period', 'tbl_connection_to_left_node_period'


def connection_to_left_node_dt(par, s, v, r):
    """Connection flow to left node for time"""
    if r.connection_to_left_node__dt.empty:
        result_multi = pd.DataFrame(index=s.dt_realize_dispatch)
        return result_multi.reset_index(), result_multi, 'connection_to_left_node_dt', 'tbl_connection_to_left_node_dt'
    result_multi = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['process', 'connection', 'node']))
    conn_map = s.process_source_sink[
        s.process_source_sink['process'].isin(s.process_connection) &
        s.process_source_sink.apply(lambda row: (row['process'], row['sink']) in s.process_sink, axis=1)
    ].set_index('process')
    for c in r.connection_to_left_node__dt.columns:
        if c in conn_map.index:
            row = conn_map.loc[c]
            result_multi[(c, row['sink'], row['source'])] = r.connection_to_left_node__dt[c]
    return result_multi.reset_index(), result_multi, 'connection_to_left_node_dt', 'tbl_connection_to_left_node_dt'


def group_flows_dt(par, s, v, r):
    """Group output flows for time"""
    
    if s.groupOutputNodeFlows.empty or s.dt_realize_dispatch.empty:
        result_multi = pd.DataFrame(index=s.dt_realize_dispatch)
        return result_multi.reset_index(), result_multi, 'group_flows_dt', 'tbl_group_flows_dt'
    
    result_multi = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['group', 'type', 'item']))
    
    # Process each group
    for g in s.groupOutputNodeFlows:
        # Slack upward
        if g in r.group_node_up_penalties__dt.columns:
            result_multi[g, 'slack', 'upward'] = r.group_node_up_penalties__dt[g]
        
        # Unit aggregates (unit to group)
        for col in r.group_output__group_aggregate_Unit_to_group__dt.columns:
            if col[0] == g:
                result_multi[g, 'unit_aggregate', col[1]] = r.group_output__group_aggregate_Unit_to_group__dt[col]
        
        # Units not in aggregate (unit to node)
        for idx, row in s.group_output__process__unit__to_node_Not_in_aggregate.iterrows():
            if row['group'] == g:
                flow_col = (row['process'], row['unit'], row['node'])
                if flow_col in r.flow_dt.columns:
                    result_multi[g, 'unit', row['process']] = r.flow_dt[flow_col]
        
        # Connection aggregates
        for col in r.group_output__group_aggregate_Connection__dt.columns:
            if col[0] == g:
                result_multi[g, 'connection', col[1]] = r.group_output__group_aggregate_Connection__dt[col]
        
        # Connections not in aggregate
        for col in r.group_output__connection_not_in_aggregate__dt.columns:
            if col[0] == g:
                result_multi[g, 'connection', col[1]] = r.group_output__connection_not_in_aggregate__dt[col]
        
        # Storage flows (negative)
        for (grp, n) in s.group_node:
            if grp == g and n in s.nodeState and n in r.nodeState_change_dt.columns:
                result_multi[g, 'storage_flow', g] = -r.nodeState_change_dt[n]
        
        # Group to unit aggregates
        for col in r.group_output__group_aggregate_Group_to_unit__dt.columns:
            if col[0] == g:
                result_multi[g, 'unit_aggregate', col[1]] = r.group_output__group_aggregate_Group_to_unit__dt[col]
        
        # Node to unit not in aggregate (negative)
        for idx, row in s.group_output__process__node__to_unit_Not_in_aggregate.iterrows():
            if row['group'] == g:
                flow_col = (row['process'], row['node'], row['unit'])
                if flow_col in r.flow_dt.columns:
                    result_multi[g, 'unit', row['process']] = -r.flow_dt[flow_col]
        
        # Internal losses (negative)
        if g in r.group_output_Internal_connection_losses__dt.columns:
            result_multi[g, 'internal_losses', 'connections'] = -r.group_output_Internal_connection_losses__dt[g]
        if g in r.group_output_Internal_unit_losses__dt.columns:
            result_multi[g, 'internal_losses', 'units'] = -r.group_output_Internal_unit_losses__dt[g]
        if g in r.group_node_state_losses__dt.columns:
            result_multi[g, 'internal_losses', 'storages'] = -r.group_node_state_losses__dt[g]
        
        # Slack downward
        if g in r.group_node_down_penalties__dt.columns:
            result_multi[g, 'slack', 'downward'] = r.group_node_down_penalties__dt[g]
        
        # Inflow
        if g in r.group_node_inflow_dt.columns:
            result_multi[g, 'inflow', g] = r.group_node_inflow_dt[g]
    
    result_multi.columns.names = ['group', 'type', 'item']
    
    return result_multi.reset_index(), result_multi, 'group_flows_dt', 'tbl_group_flows_dt'

def group_flows_period(par, s, v, r):
    """Group output flows for periods"""
    
    if s.groupOutputNodeFlows.empty or s.d_realized_period.empty:
        result_multi = pd.DataFrame(index=s.d_realized_period)
        return result_multi.reset_index(), result_multi, 'group_flows_period', 'tbl_group_flows_period'
    
    result_multi = pd.DataFrame(index=s.d_realized_period, columns=pd.MultiIndex.from_tuples([], names=['group', 'type', 'item']))
    
    # Process each group
    for g in s.groupOutputNodeFlows:
        # Slack upward
        if g in r.group_node_up_penalties__d.columns:
            result_multi[g, 'slack', 'upward'] = r.group_node_up_penalties__d[g]
        
        # Unit aggregates (unit to group)
        for col in r.group_output__group_aggregate_Unit_to_group__d.columns:
            if col[0] == g:
                result_multi[g, 'unit_aggregate', col[1]] = r.group_output__group_aggregate_Unit_to_group__d[col]
        
        # Units not in aggregate (unit to node)
        for idx, row in s.group_output__process__unit__to_node_Not_in_aggregate.iterrows():
            if row['group'] == g:
                flow_col = (row['process'], row['unit'], row['node'])
                if flow_col in r.flow_d.columns:
                    result_multi[g, 'unit', row['process']] = r.flow_d[flow_col]
        
        # Connection aggregates
        for col in r.group_output__group_aggregate_Connection__d.columns:
            if col[0] == g:
                result_multi[g, 'connection', col[1]] = r.group_output__group_aggregate_Connection__d[col]
        
        # Connections not in aggregate
        for col in r.group_output__connection_not_in_aggregate__d.columns:
            if col[0] == g:
                result_multi[g, 'connection', col[1]] = r.group_output__connection_not_in_aggregate__d[col]
        
        # Storage flows (negative)
        for (grp, n) in s.group_node:
            if grp == g and n in s.nodeState and n in r.nodeState_change_d.columns:
                result_multi[g, 'storage_flow', g] = -r.nodeState_change_d[n]
        
        # Group to unit aggregates
        for col in r.group_output__group_aggregate_Group_to_unit__d.columns:
            if col[0] == g:
                result_multi[g, 'unit_aggregate', col[1]] = r.group_output__group_aggregate_Group_to_unit__d[col]
        
        # Node to unit not in aggregate (negative)
        for idx, row in s.group_output__process__node__to_unit_Not_in_aggregate.iterrows():
            if row['group'] == g:
                flow_col = (row['process'], row['node'], row['unit'])
                if flow_col in r.flow_d.columns:
                    result_multi[g, 'unit', row['process']] = -r.flow_d[flow_col]
        
        # Internal losses (negative)
        if g in r.group_output_Internal_connection_losses__d.columns:
            result_multi[g, 'internal_losses', 'connections'] = -r.group_output_Internal_connection_losses__d[g]
        if g in r.group_output_Internal_unit_losses__d.columns:
            result_multi[g, 'internal_losses', 'units'] = -r.group_output_Internal_unit_losses__d[g]
        if g in r.group_node_state_losses__d.columns:
            result_multi[g, 'internal_losses', 'storages'] = -r.group_node_state_losses__d[g]
        
        # Slack downward
        if g in r.group_node_down_penalties__d.columns:
            result_multi[g, 'slack', 'downward'] = r.group_node_down_penalties__d[g]
        
        # Inflow
        if g in r.group_node_inflow_d.columns:
            result_multi[g, 'inflow', g] = r.group_node_inflow_d[g]
    
    return result_multi.reset_index(), result_multi, 'group_flows_period', 'tbl_group_flows_period'

def unit_cf_outputNode_period(par, s, v, r):
    """Unit capacity factors by output node for periods"""
    if r.process_sink_flow_d.empty:
        result_multi = pd.DataFrame(index=pd.Index([], name='period'))
        return result_multi.reset_index(), result_multi, 'unit_cf_outputNode_period', 'tbl_unit_cf_outputNode_period'
    result_multi = pd.DataFrame(index=s.d_realized_period, columns=pd.MultiIndex.from_tuples([], names=['unit', 'sink']))
    complete_hours = par.complete_period_share_of_year * 8760
    for col in r.process_sink_flow_d.columns:
        if col[0] in s.process_unit:
            capacity = r.entity_all_capacity[('entity', col[0])]
            result_multi[col] = (r.process_sink_flow_d[col] / complete_hours / capacity).fillna(0)
    return result_multi.reset_index(), result_multi, 'unit_cf_outputNode_period', 'tbl_unit_cf_outputNode_period'


def unit_cf_inputNode_period(par, s, v, r):
    """Unit capacity factors by input node for periods"""
    if r.process_source_flow_d.empty:
        result_multi = pd.DataFrame(index=pd.Index([], name='period'))
        return result_multi.reset_index(), result_multi, 'unit_cf_inputNode_period', 'tbl_unit_cf_inputNode_period'
    result_multi = pd.DataFrame(index=s.d_realized_period, columns=pd.MultiIndex.from_tuples([], names=['unit', 'source']))
    complete_hours = par.complete_period_share_of_year * 8760
    for col in r.process_source_flow_d.columns:
        if col[0] in s.process_unit:
            capacity = r.entity_all_capacity[('entity', col[0])]
            result_multi[col] = (r.process_source_flow_d[col] / complete_hours / capacity).fillna(0)
    return result_multi.reset_index(), result_multi, 'unit_cf_inputNode_period', 'tbl_unit_cf_inputNode_period'


def unit_VRE_curtailment_and_potential(par, s, v, r):
    """Unit VRE curtailment and potential for both periods and timesteps"""
    
    results = []
    vre_processes = set(s.process_VRE['process'])
    
    # Timestep-level curtailment (absolute values) - calculate first
    if not r.flow_dt.empty and not r.potentialVREgen_dt.empty:
        curtail_dt = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['unit', 'sink']))
        potential_dt = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['unit', 'sink']))
        
        for col in r.flow_dt.columns:
            u, source, sink = col
            if u in vre_processes and (u, sink) in s.process_sink and (u, sink) in r.potentialVREgen_dt.columns:
                curtail_dt[u, sink] = r.potentialVREgen_dt[(u, sink)] - r.flow_dt[col]
                potential_dt[u, sink] = r.potentialVREgen_dt[(u, sink)]
        
        results.append((curtail_dt.reset_index(), curtail_dt, 'unit_curtailment_outputNode_dt', 'tbl_unit_curtailment_outputNode_dt'))
        results.append((potential_dt.reset_index(), potential_dt, 'unit_VRE_potential_outputNode_dt', 'tbl_unit_VRE_potential_outputNode_dt'))
        
        # Calculate curtailment share at timestep level
        curtail_share_dt = (curtail_dt / potential_dt).where(potential_dt != 0, 0)
        results.append((curtail_share_dt.reset_index(), curtail_share_dt, 'unit_curtailment_share_outputNode_dt', 'tbl_unit_curtailment_share_outputNode_dt'))
        
        # Aggregate to period level
        curtail_period = curtail_dt.groupby(level='period').sum()
        potential_period = potential_dt.groupby(level='period').sum()
        
        # Calculate curtailment share at period level
        curtail_share_period = (curtail_period / potential_period).where(potential_period != 0, 0)

        results.append((curtail_share_period.reset_index(), curtail_share_period, 'unit_curtailment_share_outputNode_period', 'tbl_unit_curtailment_share_outputNode_period'))
        results.append((potential_period.reset_index(), potential_period, 'unit_VRE_potential_outputNode_period', 'tbl_unit_VRE_potential_outputNode_period'))
    
    return results

def unit_ramps(par, s, v, r):
    """Unit ramps by input and output nodes for timesteps"""
    results = []
    if r.ramp_dtt.empty:
        return results

    # Output node ramps
    ramp_output = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['unit', 'sink']), dtype=float)
    for col in r.ramp_dtt.columns:
        u, source, sink = col
        if (u, sink) in s.process_sink and u in s.process_unit:
            ramp_output[u, sink] = r.ramp_dtt[col].droplevel('t_previous')
    results.append((ramp_output.reset_index(), ramp_output, 'unit_ramp_outputNode_dt', 'tbl_unit_ramp_outputNode_dt'))
    
    # Input node ramps
    ramp_input = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['unit', 'source']), dtype=float)
    for col in r.ramp_dtt.columns:
        u, source, sink = col
        if (u, source) in s.process_source and u in s.process_unit:
            ramp_input[u, source] = r.ramp_dtt[col].droplevel('t_previous')
    results.append((ramp_input.reset_index(), ramp_input, 'unit_ramp_inputNode_dt', 'tbl_unit_ramp_inputNode_dt'))
    
    return results

# List of all output functions
ALL_OUTPUTS = [
    unit_capacity,
    connection_capacity,
    node_capacity,
    model_co2,
    group_node_period,
    group_node_period_time,
    group_node_VRE_share,
    group_process_CO2,
    group_process_node_flow_period,
    group_process_node_flow_dt,
    unit_outputNode_period,
    unit_outputNode_dt,
    unit_inputNode_period,
    unit_inputNode_dt,
    connection_period,
    connection_dt,
    connection_to_right_node_period,
    connection_to_right_node_dt,
    connection_to_left_node_period,
    connection_to_left_node_dt,
    group_flows_dt,
    group_flows_period,
    unit_cf_outputNode_period,
    unit_cf_inputNode_period,
    unit_VRE_curtailment_and_potential,
    unit_ramps,
]


# writer.py - handles the actual writing
def write_outputs(output_funcs=None, output_dir='output_raw', methods=['excel', 'db']):
    """
    output_funcs: list of functions to run, or None for ALL_OUTPUTS
    """
    par, s, v = read_outputs(output_dir)
    r = post_process_results(par, s, v)

    # Usage
    open('namespace_structure.txt', 'w').close()
    print_namespace_structure(r, 'r')
    print_namespace_structure(s, 's')
    print_namespace_structure(v, 'v')
    print_namespace_structure(par, 'par')

    output_funcs = output_funcs or ALL_OUTPUTS


    results_multi = {}
    results_flat = {}
    for func in output_funcs:
        func_results = func(par, s, v, r)
        
        # Handle both single result (wrapped in list) and multiple results
        if not isinstance(func_results, list):
            func_results = [func_results]
        
        for result_flat, result_multi, excel_sheet, db_table in func_results:
            # Use excel_sheet as the key to allow multiple outputs per function
            results_multi[excel_sheet] = (result_multi, excel_sheet, db_table)
            results_flat[excel_sheet] = (result_flat, excel_sheet, db_table)

    # Write to excel
    if 'excel' in methods:
        with pd.ExcelWriter('output.xlsx') as writer:
            for name, (df, sheet, _) in results_flat.items():
                df.to_excel(writer, sheet_name=sheet)
    
    # Write to database
    #if 'db' in methods:
    #    for name, (df, _, table) in results.items():
    #        api.upload(df, table=table)

