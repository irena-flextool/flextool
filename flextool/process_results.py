from types import SimpleNamespace
import pandas as pd

def post_process_results(par, s, v):
    """Calculate post-processing results from variables, parameters, and sets"""
    r = SimpleNamespace()  # Initialize empty namespace for results
    
    par, s, v = drop_levels(par, s, v)

    # hours_in_realized_period
    # Filter dt_realize_dispatch by periods in d_realized_period, then group and sum
    step_duration = par.step_duration
    
    hours_in_realized_period = step_duration.groupby(level='period').sum()
    hours_in_realized_period = hours_in_realized_period.reindex(s.d_realized_period)

    r.hours_in_realized_period = hours_in_realized_period
    r.realized_period_share_of_year = hours_in_realized_period / 8760

    s.nb = s.node_balance.union(s.node_balance_period)

    # entity_all_capacity
    # Existing capacity
    # Add investments
    # r.entity_all_capacity = r.entity_all_capacity.add(v.invest.mul(par.entity_unitsize), fill_value=0.0)
    r.entity_all_capacity = par.entity_all_existing.copy()
    periods = r.entity_all_capacity.index.get_level_values('period')
      # Add investments
    if not v.invest.empty:
        capacity_add = v.invest.mul(par.entity_unitsize[v.invest.columns])
        capacity_add_recursive = pd.DataFrame(columns=v.invest.columns)
        for i, period in enumerate(periods):
            capacity_add_recursive.loc[period] = capacity_add.loc[periods[:i+1]].sum()
        r.entity_all_capacity = r.entity_all_capacity.add(capacity_add_recursive, fill_value=0)
      # Subtract divestments
    if not v.divest.empty:
        capacity_sub = v.divest.mul(par.entity_unitsize[v.divest.columns])
        capacity_sub_recursive = pd.DataFrame(columns=v.divest.columns)
        for i, period in enumerate(periods):
            capacity_sub_recursive.loc[period] = capacity_sub.loc[periods[:i+1]].sum()
        r.entity_all_capacity = r.entity_all_capacity.sub(capacity_sub_recursive, fill_value=0)

    # r_process_Online__dt - just sum the two DataFrames
    r.process_online_dt = v.online_linear.add(v.online_integer, fill_value=0)

    # Calculate r_process__source__sink_Flow_dt
    s.process_source_sink_alwaysProcess = s.process_method_sources_sinks.droplevel(['method', 'orig_source', 'orig_sink'])
    s.process_source_sink_alwaysProcess.names=['process', 'source', 'sink']

    r.flow_dt = pd.DataFrame(
        index=s.dt_realize_dispatch,
        columns=s.process_source_sink_alwaysProcess,
        dtype=float
    )    
    unitsize = par.entity_unitsize
    slope = par.process_slope
    section = par.process_section
    
    for row in s.process_method_sources_sinks:
        p = row[0]
        method = row[1]
        orig_source, orig_sink = row[2], row[3]
        always_source, always_sink = row[4], row[5]
        
        # Get base flow value
        flow_val = v.flow[[(p, orig_source, orig_sink)]] * unitsize[p]
        
        # Apply transformations for method_1var_per_way on source-to-process flows
        if (method in s.method_1var_per_way and 
            p not in s.process_profile and
            orig_source == always_source and orig_sink != always_sink):
            flow_val = flow_val.mul(slope[p], axis=0)
            if p in s.process_unit:
                flow_val /= (par.process_sink_coefficient.loc[p, orig_sink] *
                            par.process_source_coefficient.loc[p, orig_source])
            if (p, 'min_load_efficiency') in s.process__ct_method:
                flow_val = flow_val.add(r.process_online_dt[p] * section[p] * unitsize[p], axis=0)
        r.flow_dt[p, always_source, always_sink] = flow_val

    # r.flow_d - sum over dt_realize_dispatch
    r.flow_d = r.flow_dt[r.flow_dt.index.get_level_values('period').isin(s.d_realized_period)].groupby(level='period').sum()

    # r_process_source_sink_ramp_dtt - difference between t and t_previous
    current_idx = s.dtt.droplevel('t_previous')
    previous_idx = s.dtt.droplevel('time')
    r.ramp_dtt = pd.DataFrame(
        r.flow_dt.reindex(current_idx).values - r.flow_dt.reindex(previous_idx).values,
        index=s.dtt,
        columns=r.flow_dt.columns,
        dtype=float
    )

    # r_node_ramp_dtt - sum ramps for flows into/out of each node
    # Drop t_previous from index first
    ramp_dt_dropped = r.ramp_dtt.droplevel('t_previous')
    nodes_source = s.node_balance.copy()
    nodes_source.name = 'source'
    nodes_sink = s.node_balance.copy()
    nodes_sink.name = 'sink'

    # Flows out of nodes (source == n, positive)
    flows_out = ramp_dt_dropped[nodes_source.join(ramp_dt_dropped.columns).join(nodes_source, how='inner')].T.groupby('source').sum().T
    # Flows into nodes (sink == n, negative)
    flows_in = -ramp_dt_dropped[nodes_sink.join(ramp_dt_dropped.columns).join(nodes_sink, how='inner')].T.groupby('sink').sum().T
    # Combine
    r_node_ramp_dt = flows_out.add(flows_in, fill_value=0).reindex(columns=s.node_balance, fill_value=0.0)

    # Filter just connection flows
    conn_flows = r.flow_dt[r.flow_dt.columns[r.flow_dt.columns.get_level_values('process').isin(s.process_connection)]]

    # Divide into the four flows present in one connection
    conn_to_left = conn_flows[conn_flows.columns[conn_flows.columns.droplevel('source').isin(s.process_source)]]
    left_to_conn = conn_flows[conn_flows.columns[conn_flows.columns.droplevel('sink').isin(s.process_source)]]
    conn_to_right = conn_flows[conn_flows.columns[conn_flows.columns.droplevel('source').isin(s.process_sink)]]
    right_to_conn = conn_flows[conn_flows.columns[conn_flows.columns.droplevel('sink').isin(s.process_sink)]]

    # Drop redundant column index level
    conn_to_left.columns = conn_to_left.columns.droplevel('source')
    left_to_conn.columns = left_to_conn.columns.droplevel('sink')
    conn_to_right.columns = conn_to_right.columns.droplevel('source')
    right_to_conn.columns = right_to_conn.columns.droplevel('sink')

    # Rename source/sink to match with other dataframes using 'node'
    conn_to_left.columns.names = ['process', 'node']
    left_to_conn.columns.names = ['process', 'node']
    conn_to_right.columns.names = ['process', 'node']
    right_to_conn.columns.names = ['process', 'node']

    # Flow to right: where (process, sink) in s.process_sink
    # flow_to_right = conn_to_right.sub(right_to_conn, axis=1)

    # Flow to left: where (process, sink) in s.process_source (negative)
    #flow_left = conn_to_left

    r.connection_dt = conn_to_right.droplevel('node', axis=1).sub(conn_to_left.droplevel('node', axis=1), axis=1)
    r.connection_losses_dt = r.connection_dt.sub(right_to_conn.droplevel('node', axis=1)).sub(left_to_conn.droplevel('node', axis=1))

    # r_connection_to_left_node__dt and r_connection_to_right_node__dt
    # Using same conn_flows, right_cols, left_cols from above
    r.connection_to_left_node__dt = conn_to_left.sub(left_to_conn, axis=1)
    r.connection_to_right_node__dt = conn_to_right.sub(right_to_conn, axis=1)
    r.connection_to_left_node__d = r.connection_to_left_node__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)
    r.connection_to_right_node__d = r.connection_to_right_node__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)
    
    # Combine directional flows
    from_conn = pd.concat([conn_to_left, conn_to_right], axis=1)  # columns: ['process', 'node']
    to_conn = pd.concat([left_to_conn, right_to_conn], axis=1)     # columns: ['process', 'node']

    # r_group_output__from_connection_not_in_aggregate__dt
    from_conn_set = s.group_output__process__connection__to_node_Not_in_aggregate.droplevel('connection')
    group_sets = from_conn.columns.join(from_conn_set, how='inner')
    from_conn_selected = from_conn[group_sets.droplevel('group')]
    from_conn_selected.columns = group_sets
    r.group_output__from_connection_not_in_aggregate__dt = from_conn_selected

    # r_group_output__to_connection_not_in_aggregate__dt
    to_conn_set = s.group_output__process__node__to_connection_Not_in_aggregate.droplevel('connection')
    group_sets = to_conn.columns.join(to_conn_set, how='inner')
    to_conn_selected = to_conn[group_sets.droplevel('group')]
    to_conn_selected.columns = group_sets
    r.group_output__to_connection_not_in_aggregate__dt = to_conn_selected

    # Period aggregations
    r.group_output__from_connection_not_in_aggregate__d = r.group_output__from_connection_not_in_aggregate__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)
    r.group_output__to_connection_not_in_aggregate__d = r.group_output__to_connection_not_in_aggregate__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)

    # r_group_output__from_connection_aggregate__dt
    from_conn_agg_set = s.group_output__group_aggregate__process__connection__to_node.droplevel('connection')
    group_agg_sets = from_conn.columns.join(from_conn_agg_set, how='inner')
    from_conn_agg_selected = from_conn[group_agg_sets.droplevel(['group', 'group_aggregate'])]
    from_conn_agg_selected.columns = group_agg_sets
    r.group_output__from_connection_aggregate__dt = from_conn_agg_selected.T.groupby(level=['group', 'group_aggregate']).sum().T

    # r_group_output__to_connection_aggregate__dt
    to_conn_agg_set = s.group_output__group_aggregate__process__node__to_connection.droplevel('connection')
    group_agg_sets = to_conn_agg_set.join(to_conn.columns)
    to_conn_agg_selected = to_conn[group_agg_sets.droplevel(['group', 'group_aggregate'])]
    to_conn_agg_selected.columns = group_agg_sets
    r.group_output__to_connection_aggregate__dt = to_conn_agg_selected.T.groupby(level=['group', 'group_aggregate']).sum().T

    # Daily aggregations
    r.group_output__from_connection_aggregate__d = r.group_output__from_connection_aggregate__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)
    r.group_output__to_connection_aggregate__d = r.group_output__to_connection_aggregate__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)

    # r_group_output_Internal_connection_losses__dt
    losses_set = s.group_output__process_fully_inside
    group_losses_sets = r.connection_losses_dt.columns.join(losses_set, how='inner')
    losses_selected = r.connection_losses_dt[group_losses_sets.droplevel('group')]
    losses_selected.columns = group_losses_sets
    r.group_output_Internal_connection_losses__dt = losses_selected

    # r_group_output_Internal_connection_losses__d
    r.group_output_Internal_connection_losses__d = r.group_output_Internal_connection_losses__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)


    # r_process_source_sink_flow_d - with step_duration
    r.flow_d = r.flow_dt.mul(step_duration, axis=0).groupby('period').sum()
    
    # r_process_source_flow_d - sum over sinks
    r.process_source_flow_d = r.flow_d.T.groupby(level=['process', 'source']).sum().T.reindex(columns=s.process_source, fill_value=0.0)

    # r_process_sink_flow_d - sum over sources
    r.process_sink_flow_d = r.flow_d.T.groupby(level=['process', 'sink']).sum().T.reindex(columns=s.process_sink, fill_value=0.0)
    
    # r_connection_d - with step_duration
    r_conn_weighted = r.connection_dt.mul(step_duration, axis=0)    
    if not r_conn_weighted.empty:
        r_conn_d = r_conn_weighted[
            r_conn_weighted.index.get_level_values('period').isin(s.d_realized_period)
        ].groupby(level='period').sum()
        r.connection_d = r_conn_d
    else:
        r.connection_d = pd.DataFrame(index=s.d_realized_period)
    

    # Calculate r_node_state_change_dt
    # Filter dt_realize_dispatch
    dt_dispatch_idx = s.dt_realize_dispatch
    # Initialize result
    r_state_change = pd.DataFrame(0.0, index=s.dt_realize_dispatch, columns=s.node_state, dtype=float)
    # Create index mappings from dtttdt
    prev_period_idx = s.dtttdt.droplevel(['time', 't_previous_within_timeset', 'd_previous', 't_previous_within_solve']).set_names(['period', 'time'])
    prev_timeset_idx = s.dtttdt.droplevel(['time', 't_previous', 'd_previous', 't_previous_within_solve']).set_names(['period', 'time'])
    prev_forward_only_idx = s.dtttdt.droplevel(['period', 'time', 't_previous', 't_previous_within_timeset']).set_names(['period', 'time'])

    # Create exclude_idx directly from MultiIndex
    exclude_idx = s.period__time_first[
        s.period__time_first.get_level_values('period').isin(s.period_first_of_solve)
    ]
    
    for n in s.node_state:
        if n not in v.state.columns:
            continue

        state_change = pd.Series(0.0, index=current_idx)

        v_current = v.state[n].squeeze()
        v_prev_period = pd.Series(v.state[n].squeeze().reindex(prev_period_idx).values, index=current_idx)
        v_prev_timeblock = pd.Series(v.state[n].squeeze().reindex(prev_timeset_idx).values, index=current_idx)
        v_forward = pd.Series(v.state[n].squeeze().reindex(prev_forward_only_idx).values, index=current_idx)

        # bind_forward_only: change from start to finish, leaving first timestep empty
        # (uses same timeline as bind_within_timeset)
        if (n, 'bind_forward_only') in s.node__storage_binding_method:
            mask = ~current_idx.isin(exclude_idx)
            state_change += ((v_current - v_forward) * unitsize[n]).where(mask, 0)

        # bind_within_solve: treated as bind_forward_only without exclude first since solve info is not available
        if (n, 'bind_within_solve') in s.node__storage_binding_method:
            state_change += (v_current - v_forward) * unitsize[n]

        # bind_within_period: wraps the change over (difference between last and first
        # timestep in period is assigned to first timestep)
        if (n, 'bind_within_period') in s.node__storage_binding_method:
            state_change += (v_current - v_prev_period) * unitsize[n]

        # bind_within_timeset: continues timeline from one period to next, wraps over
        # the whole set of results
        if (n, 'bind_within_timeset') in s.node__storage_binding_method:
            state_change += (v_current - v_prev_timeblock) * unitsize[n]

        # Assign
        r_state_change[n] = state_change
    
    r.node_state_change_dt = r_state_change

    # r_node_state_change_d
    r.node_state_change_d = r.node_state_change_dt.groupby(level='period').sum()
    
    # r_self_discharge_loss_dt - element-wise multiplication
    r.self_discharge_loss_dt = v.state[par.node_self_discharge_loss.columns] \
                                .mul(par.node_self_discharge_loss, axis=1, level=0) \
                                .mul(par.entity_unitsize[par.node_self_discharge_loss.columns], axis='columns', level=0)
    
    # r_self_discharge_loss_d - multiply by step_duration then sum
    r.self_discharge_loss_d = r.self_discharge_loss_dt.mul(step_duration, axis=0).groupby('period').sum()

    # r_cost_commodity_dt
    # Filter flows with isin
    flow_from_commodity_node = r.flow_dt[r.flow_dt.columns[r.flow_dt.columns.get_level_values(level=1).isin(s.commodity_node.get_level_values(level=1))]]
    flow_to_commodity_node = r.flow_dt[r.flow_dt.columns[r.flow_dt.columns.get_level_values(level=2).isin(s.commodity_node.get_level_values(level=1))]]
    
    commodity_price = par.commodity_price[s.commodity_node.get_level_values('commodity').unique()]
    commodity_price.columns = commodity_price.columns.join(s.commodity_node)
    flow_from_commodity_node.columns.names = ['process', 'node', 'sink']
    # Filter columns with join
    flow_from_commodity_node.columns = flow_from_commodity_node.columns.join(commodity_price.columns)
    flow_from_commodity = flow_from_commodity_node.T.groupby('commodity').sum().T
    r.cost_commodity_dt = flow_from_commodity.mul(commodity_price).mul(par.step_duration, axis=0)
    flow_to_commodity_node.columns.names = ['process', 'source', 'node']
    flow_to_commodity_node.columns = flow_to_commodity_node.columns.join(commodity_price.columns)
    flow_to_commodity = flow_to_commodity_node.T.groupby('commodity').sum().T
    r.sales_commodity_dt = flow_to_commodity.mul(commodity_price).mul(par.step_duration, axis=0)

    # r_process_commodity_d
    r.cost_commodity_d = r.cost_commodity_dt.groupby('period').sum()
    r.sales_commodity_d = r.sales_commodity_dt.groupby('period').sum()
     
    # r_process_emissions_co2_dt
    # Flows out of node: (process, source, sink) where (process, sink) matches (process, node)
    flow_outof_cols = r.flow_dt.columns.copy()
    flow_outof_cols.names = ['process', 'node', 'sink']
    flow_outof_cols = s.process__commodity__node_co2.join(flow_outof_cols)
    flow_outof_cols = flow_outof_cols[~flow_outof_cols.get_level_values('sink').isna()]

    # Flows into node: (process, source, sink) where (process, source) matches (process, node)
    flow_into_cols = r.flow_dt.columns.copy()
    flow_into_cols.names = ['process', 'source', 'node']
    flow_into_cols = s.process__commodity__node_co2.join(flow_into_cols)
    flow_into_cols = flow_into_cols[~flow_into_cols.get_level_values('source').isna()]
    flow_into_cols = flow_into_cols.reorder_levels(order = ['process', 'commodity', 'source', 'node'])

    flow_outof_node = r.flow_dt[flow_outof_cols.droplevel('commodity')]
    flow_outof_node.columns.names = ['process', 'node', 'sink']
    flow_outof_node.columns = flow_outof_node.columns.join(flow_outof_cols)

    flow_into_node = r.flow_dt[flow_into_cols.droplevel('commodity')]
    flow_into_node.columns.names = ['process', 'source', 'node']
    flow_into_node.columns = flow_into_node.columns.join(flow_into_cols)

    # Group by (process, commodity, node) and sum (handles duplicate columns)
    flow_into_node_grouped = flow_into_node.T.groupby(level=[0, 2, 3]).sum().T
    flow_outof_node_grouped = flow_outof_node.T.groupby(level=[0, 1, 3]).sum().T

    # Net flow = into - out
    net_flow = flow_outof_node_grouped.sub(flow_into_node_grouped, fill_value=0)

    # Multiply by step_duration and co2_content
    net_flow_with_duration = net_flow.mul(par.step_duration, axis=0)
    r.process_emissions_co2_dt = net_flow_with_duration.mul(par.commodity_co2_content, axis=1, level='commodity')

    # Add 'type' level to process_emissions_co2_dt columns
    cols_df = r.process_emissions_co2_dt.columns.to_frame(index=False)
    cols_df['type'] = 'unit'  # default
    cols_df.loc[cols_df['process'].isin(s.process_connection), 'type'] = 'connection'
    r.process_emissions_co2_dt.columns = pd.MultiIndex.from_frame(
        cols_df[['type', 'process', 'commodity', 'node']]
    )    

    # r_process_emissions_co2_d - sum and divide by complete_period_share_of_year
    r.process_emissions_co2_d = r.process_emissions_co2_dt.groupby(level='period').sum()
    r.process_emissions_co2_d = r.process_emissions_co2_d.div(par.complete_period_share_of_year, axis=0)
    
    # r_emissions_co2_d - sum processes over period
    r.emissions_co2_d = r.process_emissions_co2_d.sum(axis=1)

    # r_emissions_co2_dt - sum processes over period__time
    r.emissions_co2_dt = r.process_emissions_co2_dt.sum(axis=1)

    # r_group co2
    s.group_node_co2 = s.group_node[s.group_node.get_level_values('group').isin(par.group_co2_price.columns)]
    group_process_co2_columns = r.process_emissions_co2_dt.columns.join(s.group_node_co2)
    r.group_process_emissions_co2_dt = pd.DataFrame(index=r.process_emissions_co2_dt.index, columns=group_process_co2_columns)
    for col in group_process_co2_columns:
        r.group_process_emissions_co2_dt[col] = r.process_emissions_co2_dt[col[:4]]
    # Sum emissions by group (handles multiple columns per group)
    r.group_co2_dt = r.group_process_emissions_co2_dt.T.groupby('group').sum().T
    r.group_co2_d = r.group_co2_dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)
    # Multiply by CO2 prices for each group
    r.group_cost_co2_dt = r.group_co2_dt.mul(par.group_co2_price)
    r.group_cost_co2_d = r.group_co2_d.mul(par.group_co2_price)
    # As CO2 price is set on groups, then the total co2 cost needs to be summed over groups
    r.cost_co2_dt = r.group_cost_co2_dt.sum(axis=1)
    r.cost_co2_d = r.group_cost_co2_d.groupby('period').sum()
    
    # r_cost_process_other_operational_cost_dt
    # Filter flow columns that exist in varCost parameters
    relevant_flows = r.flow_dt.loc[:, r.flow_dt.columns.intersection(par.process_source_sink_varCost.columns)]
    # Multiply by step_duration and varCost
    cost_flows = relevant_flows.mul(par.step_duration, axis=0).mul(par.process_source_sink_varCost, axis=1)
    # Group by process (level 0 of column MultiIndex) and sum
    r.cost_process_other_operational_cost_dt = cost_flows.T.groupby(level=0).sum().T.reindex(columns=s.process, fill_value=0.0)

    # r_process_startup_dt
    r.process_startup_dt = v.startup_linear.add(v.startup_integer, fill_value=0)

    # r_cost_startup_dt
    r.cost_startup_dt = pd.DataFrame(0.0, index=r.process_startup_dt.index, columns=s.process_online, dtype=float)
    # Filter to processes present in all required structures
    valid_processes = s.process_online.intersection(r.process_startup_dt.columns).intersection(par.process_startup_cost.columns)
    if len(valid_processes) > 0:
        # Multiply by entity_unitsize
        cost = r.process_startup_dt[valid_processes].mul(par.entity_unitsize[valid_processes], axis=1)
        # Apply period-specific costs by mapping each row's period to the cost dataframe
        periods = cost.index.get_level_values('period')
        period_costs = par.process_startup_cost.loc[periods, valid_processes]
        period_costs.index = cost.index  # Align indices
        r.cost_startup_dt[valid_processes] = cost.mul(period_costs)

    # Reserves
    r.reserves_dt = v.reserve.mul(par.step_duration, axis=0)
    r.reserves_d = r.reserves_dt.groupby('period').sum() \
        .div(par.complete_period_share_of_year, axis=0)

    # Node slacks
    r.upward_node_slack_dt = v.q_state_up.mul(par.node_capacity_for_scaling[v.q_state_up.columns]).mul(par.step_duration, axis=0)
    r.upward_node_slack_d_not_annualized = r.upward_node_slack_dt.mul(par.step_duration, axis=0) \
        .groupby('period').sum()
    r.upward_node_slack_d = r.upward_node_slack_d_not_annualized.div(par.complete_period_share_of_year, axis=0)
    r.downward_node_slack_dt = v.q_state_down.mul(par.node_capacity_for_scaling[v.q_state_down.columns]).mul(par.step_duration, axis=0)
    r.downward_node_slack_d_not_annualized = r.downward_node_slack_dt.mul(par.step_duration, axis=0) \
        .groupby('period').sum()
    r.downward_node_slack_d = r.downward_node_slack_d_not_annualized.div(par.complete_period_share_of_year, axis=0)
    upward_node_penalty = r.upward_node_slack_dt.mul(par.node_penalty_up[v.q_state_up.columns])
    downward_node_penalty = r.downward_node_slack_dt.mul(par.node_penalty_down[v.q_state_down.columns])
    r.costPenalty_node_state_upDown_dt = pd.concat([upward_node_penalty, downward_node_penalty], axis=1, keys=['up', 'down'], names=['upDown'])
    r.costPenalty_node_state_upDown_dt = r.costPenalty_node_state_upDown_dt.reorder_levels([1, 0], axis=1)
    r.costPenalty_node_state_upDown_d = r.costPenalty_node_state_upDown_dt.groupby(level='period').sum()

    # Inertia slack
    r.q_inertia_dt = v.q_inertia.mul(par.group_inertia_limit)
    r.q_inertia_d_not_annualized = r.q_inertia_dt.mul(par.step_duration, axis=0).groupby('period').sum()
    r.q_inertia_d = r.q_inertia_d_not_annualized.div(par.complete_period_share_of_year, axis=0)
    r.costPenalty_inertia_dt = r.q_inertia_dt.mul(par.group_penalty_inertia)

    # Non-synchronous slack
    r.q_non_synchronous_dt = v.q_non_synchronous.mul(par.group_capacity_for_scaling[s.groupNonSync])
    r.q_non_synchronous_d_not_annualized = r.q_non_synchronous_dt.mul(par.step_duration, axis=0) \
        .groupby('period').sum()
    r.q_non_synchronous_d = r.q_non_synchronous_d_not_annualized.div(par.complete_period_share_of_year, axis=0)
    r.costPenalty_non_synchronous_dt = r.q_non_synchronous_dt.mul(par.group_penalty_non_synchronous)

    # Capacity margin slack
    r.q_capacity_margin_d_not_annualized = v.q_capacity_margin \
        .mul(par.group_capacity_for_scaling[s.groupCapacityMargin])
    r.costPenalty_capacity_margin_d = r.q_capacity_margin_d_not_annualized \
        .mul(par.discount_factor_operations_yearly, axis=0).sum(axis=1)
    
    # Reserve slack
    r.q_reserves_dt = v.q_reserve.mul(par.reserve_upDown_group_reservation[v.q_reserve.columns], axis=1)
    r.q_reserves_d_not_annualized = r.q_reserves_dt.mul(par.step_duration, axis=0).groupby(level='period').sum()
    r.q_reserves_d = r.q_reserves_d_not_annualized.div(par.complete_period_share_of_year, axis=0)
    r.costPenalty_reserve_upDown_dt = v.q_reserve.mul(par.step_duration, axis=0) \
        .mul(par.reserve_upDown_group_penalty, axis=1) \
        .mul(par.reserve_upDown_group_reservation, axis=1)

    # Investment cost for entities
    r.cost_entity_invest_d = v.invest.mul(par.entity_unitsize[v.invest.columns]).mul(par.entity_annual_discounted)
    
    # Divestment cost for entities
    r.cost_entity_divest_d = -v.divest.mul(par.entity_unitsize[v.divest.columns]).mul(par.entity_annual_divest_discounted)
    
    # Fixed cost for entities
    r.cost_entity_fixed_pre_existing = (par.entity_pre_existing * par.entity_fixed_cost * 1000).mul(par.discount_factor_operations_yearly, axis=0)
    r.cost_entity_fixed_invested = (v.invest.mul(par.entity_unitsize[v.invest.columns] * par.entity_lifetime_fixed_cost[v.invest.columns] * 1000))
    r.cost_entity_fixed_divested = -(v.divest.mul(par.entity_unitsize[v.divest.columns] * par.entity_lifetime_fixed_cost_divest[v.divest.columns] * 1000))
    
    # Aggregate costs
    r.costOper_dt = (r.cost_commodity_dt.sum(axis=1) -
                        r.sales_commodity_dt.sum(axis=1) +
                        r.cost_process_other_operational_cost_dt.sum(axis=1) + 
                        r.cost_startup_dt.sum(axis=1) +
                        r.cost_co2_dt
                    )
    r.costPenalty_dt = (r.costPenalty_node_state_upDown_dt.sum(axis=1) + 
                            r.costPenalty_inertia_dt.sum(axis=1) + 
                            r.costPenalty_non_synchronous_dt.sum(axis=1) + 
                            r.costPenalty_reserve_upDown_dt.sum(axis=1))
    
    # Period aggregations
    r.cost_process_other_operational_cost_d = r.cost_process_other_operational_cost_dt.groupby('period').sum()
    
    r.cost_variable_d = r.cost_process_other_operational_cost_d.sum(axis=1)

    r.cost_startup_d = (r.cost_startup_dt[
        r.cost_startup_dt.index.get_level_values('period').isin(s.d_realized_period)
    ].groupby(level='period').sum() if not r.cost_startup_dt.empty 
    else pd.DataFrame(0.0, index=s.d_realized_period, columns=r.cost_startup_dt.columns if hasattr(r.cost_startup_dt, 'columns') else []))

    r.costPenalty_inertia_d = (r.costPenalty_inertia_dt[
        r.costPenalty_inertia_dt.index.get_level_values('period').isin(s.d_realized_period)
    ].groupby(level='period').sum() if not r.costPenalty_inertia_dt.empty 
    else pd.DataFrame(0.0, index=s.d_realized_period, columns=r.costPenalty_inertia_dt.columns if hasattr(r.costPenalty_inertia_dt, 'columns') else []))

    r.costPenalty_non_synchronous_d = (r.costPenalty_non_synchronous_dt[
        r.costPenalty_non_synchronous_dt.index.get_level_values('period').isin(s.d_realized_period)
    ].groupby(level='period').sum() if not r.costPenalty_non_synchronous_dt.empty 
    else pd.DataFrame(0.0, index=s.d_realized_period, columns=r.costPenalty_non_synchronous_dt.columns if hasattr(r.costPenalty_non_synchronous_dt, 'columns') else []))

    r.costPenalty_reserve_upDown_d = (r.costPenalty_reserve_upDown_dt[
        r.costPenalty_reserve_upDown_dt.index.get_level_values('period').isin(s.d_realized_period)
    ].groupby(level='period').sum() if not r.costPenalty_reserve_upDown_dt.empty 
    else pd.DataFrame(0.0, index=s.d_realized_period, columns=r.costPenalty_reserve_upDown_dt.columns if hasattr(r.costPenalty_reserve_upDown_dt, 'columns') else []))

    r.costOper_d = r.costOper_dt.groupby('period').sum() \
                                .mul(par.discount_factor_operations_yearly, axis=0) \
                                .div(par.complete_period_share_of_year, axis=0)

    r.costPenalty_d = r.costPenalty_dt.groupby('period').sum() \
                                .mul(par.discount_factor_operations_yearly, axis=0) \
                                .div(par.complete_period_share_of_year, axis=0)
    
    r.costPenalty_d = r.costPenalty_d.add(r.costPenalty_capacity_margin_d, fill_value=0)

    r.costOper_and_penalty_d = r.costOper_d.add(r.costPenalty_d, fill_value=0)        

    # Investment/divestment aggregations by type
    r.costInvestUnit_d = r.cost_entity_invest_d[[e for e in s.process_unit if e in r.cost_entity_invest_d.columns]].sum(axis=1)
    r.costDivestUnit_d = r.cost_entity_divest_d[[e for e in s.process_unit if e in r.cost_entity_divest_d.columns]].sum(axis=1)
    r.costInvestConnection_d = r.cost_entity_invest_d[[e for e in s.process_connection if e in r.cost_entity_invest_d.columns]].sum(axis=1)
    r.costDivestConnection_d = r.cost_entity_divest_d[[e for e in s.process_connection if e in r.cost_entity_divest_d.columns]].sum(axis=1)
    r.costInvestState_d = r.cost_entity_invest_d[[e for e in s.node_state if e in r.cost_entity_invest_d.columns]].sum(axis=1)
    r.costDivestState_d = r.cost_entity_divest_d[[e for e in s.node_state if e in r.cost_entity_divest_d.columns]].sum(axis=1)
    
    r.costInvest_d = r.costInvestUnit_d + r.costInvestConnection_d + r.costInvestState_d
    r.costDivest_d = r.costDivestUnit_d + r.costDivestConnection_d + r.costDivestState_d
    r.costFixedPreExisting_d = r.cost_entity_fixed_pre_existing.sum(axis=1)
    r.costFixedInvested_d = r.cost_entity_fixed_invested.sum(axis=1)
    r.costFixedDivested_d = r.cost_entity_fixed_divested.sum(axis=1)

    # pdNodeInflow
    r.node_inflow_d = par.node_inflow.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)
    
    # potentialVREgen_dt
    # Filter VRE processes that are in process_sink and have upper_limit profile method
    vre_with_sink = s.process_VRE[s.process_VRE.isin(s.process_sink)]
    vre_node_profile = vre_with_sink.join(s.process__node__profile__profile_method)
    vre_node_profile_upper = vre_node_profile[vre_node_profile.get_level_values('profile_method').isin(['upper_limit'])]
    vre_profiles_in_use = par.profile[vre_node_profile_upper.get_level_values('profile').unique()]
    # Add process, node and profile_method levels to the profiles in use
    vre_profiles_in_use.columns = vre_profiles_in_use.columns.join(vre_node_profile_upper)
    r.potentialVREgen_dt = vre_profiles_in_use.mul(par.process_availability).mul(r.entity_all_capacity, axis=1, level=0).droplevel(axis=1, level=['profile', 'profile_method'])

    # potentialVREgen - aggregate by period
    r.potentialVREgen_d = r.potentialVREgen_dt.groupby('period').sum()

    # Filter unit flows
    unit_flows = r.flow_dt[r.flow_dt.columns[r.flow_dt.columns.get_level_values('process').isin(s.process_unit)]]

    # Split by direction
    unit_to_node = unit_flows[unit_flows.columns[unit_flows.columns.droplevel('source').isin(s.process_sink)]]
    node_to_unit = unit_flows[unit_flows.columns[unit_flows.columns.droplevel('sink').isin(s.process_source)]]

    # Drop redundant levels
    unit_to_node.columns = unit_to_node.columns.droplevel('source')
    node_to_unit.columns = node_to_unit.columns.droplevel('sink')

    # Rename to match with other dataframes
    unit_to_node.columns.names = ['process', 'node']
    node_to_unit.columns.names = ['process', 'node']

    # r_group_output__unit_to_node_not_in_aggregate__dt
    unit_to_node_set = s.group_output__process__unit__to_node_Not_in_aggregate.droplevel('unit')
    group_sets = unit_to_node.columns.join(unit_to_node_set, how='inner')
    unit_to_node_selected = unit_to_node[group_sets.droplevel('group')]
    unit_to_node_selected.columns = group_sets
    r.group_output__unit_to_node_not_in_aggregate__dt = unit_to_node_selected

    # r_group_output__node_to_unit_not_in_aggregate__dt
    node_to_unit_set = s.group_output__process__node__to_unit_Not_in_aggregate.droplevel('unit')
    group_sets = node_to_unit.columns.join(node_to_unit_set, how='inner')
    node_to_unit_selected = node_to_unit[group_sets.droplevel('group')]
    node_to_unit_selected.columns = group_sets
    r.group_output__node_to_unit_not_in_aggregate__dt = node_to_unit_selected

    # Daily aggregations
    r.group_output__unit_to_node_not_in_aggregate__d = r.group_output__unit_to_node_not_in_aggregate__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)
    r.group_output__node_to_unit_not_in_aggregate__d = r.group_output__node_to_unit_not_in_aggregate__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)

    # r_group_output__group_aggregate_Unit_to_group__dt
    unit_to_group_set = s.group_output__group_aggregate__process__unit__to_node.droplevel('unit')
    group_agg_sets = unit_to_node.columns.join(unit_to_group_set, how='inner')
    unit_to_group_selected = unit_to_node[group_agg_sets.droplevel(['group', 'group_aggregate'])]
    unit_to_group_selected.columns = group_agg_sets
    r.group_output__group_aggregate_Unit_to_group__dt = unit_to_group_selected.T.groupby(level=['group', 'group_aggregate']).sum().T

    # r_group_output__group_aggregate_Unit_to_group__d
    r.group_output__group_aggregate_Unit_to_group__d = r.group_output__group_aggregate_Unit_to_group__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)

    # r_group_output__group_aggregate_Group_to_unit__dt (negative)
    group_to_unit_set = s.group_output__group_aggregate__process__node__to_unit.droplevel('unit')
    group_agg_sets = node_to_unit.columns.join(group_to_unit_set, how='inner')
    group_to_unit_selected = node_to_unit[group_agg_sets.droplevel(['group', 'group_aggregate'])]
    group_to_unit_selected.columns = group_agg_sets
    r.group_output__group_aggregate_Group_to_unit__dt = -group_to_unit_selected.T.groupby(level=['group', 'group_aggregate']).sum().T

    # r_group_output__group_aggregate_Group_to_unit__d
    r.group_output__group_aggregate_Group_to_unit__d = r.group_output__group_aggregate_Group_to_unit__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)

    # r_group_output_Internal_unit_losses__dt
    # Filter to only units that are fully inside groups first
    losses_set = s.group_output__process_fully_inside
    group_losses_sets_node_to_unit = losses_set.join(node_to_unit.columns, how='inner')
    group_losses_sets_unit_to_node = losses_set.join(unit_to_node.columns, how='inner')

    # Filter the directional flows to only include relevant units
    node_to_unit_filtered = node_to_unit[group_losses_sets_node_to_unit.droplevel('group')]
    unit_to_node_filtered = unit_to_node[group_losses_sets_unit_to_node.droplevel('group')]
    node_to_unit_filtered.columns = group_losses_sets_node_to_unit
    unit_to_node_filtered.columns = group_losses_sets_unit_to_node

    # Calculate unit losses (input - output) only for filtered units
    unit_losses_dt = node_to_unit_filtered.sub(unit_to_node_filtered, axis=1, fill_value=0.0)
    r.group_output_Internal_unit_losses__dt = unit_losses_dt

    # r_group_output_Internal_unit_losses__d
    r.group_output_Internal_unit_losses__d = r.group_output_Internal_unit_losses__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)

    # r_group_node_inflow_dt
    r_group_inflow = pd.DataFrame(0.0, index=dt_dispatch_idx, columns=s.groupOutputNodeFlows, dtype=float)
    for g in s.groupOutputNodeFlows:
        inflow = pd.Series(0.0, index=dt_dispatch_idx)
        for n in s.node:
            if (g, n) in s.group_node and (n, 'no_inflow') not in s.node__inflow_method:
                if n in par.node_inflow.columns:
                    inflow = par.node_inflow[n].squeeze().add(inflow, axis=0)
        r_group_inflow[g] = inflow
    r.group_node_inflow_dt = r_group_inflow
    
    # r_group_node_inflow_d
    r.group_node_inflow_d = (
        r.group_node_inflow_dt[
            r.group_node_inflow_dt.index.get_level_values('period').isin(s.d_realized_period)
        ].groupby(level='period').sum() if not r.group_node_inflow_dt.empty
        else pd.DataFrame(0.0, index=s.d_realized_period, columns=s.groupOutputNodeFlows))
    
    # r_group_node_state_losses__dt
    r_group_state_losses = pd.DataFrame(0.0, index=dt_dispatch_idx, columns=s.groupOutputNodeFlows, dtype=float)
    for g in s.groupOutputNodeFlows:
        losses = pd.Series(0.0, index=dt_dispatch_idx)
        
        for n in s.node_self_discharge:
            if (g, n) in s.group_node and n in r.self_discharge_loss_dt.columns:
                losses += r.self_discharge_loss_dt[n].reindex(dt_dispatch_idx, fill_value=0)
        
        r_group_state_losses[g] = losses
    
    r.group_node_state_losses__dt = r_group_state_losses
    
    # r_group_node_state_losses__d
    r.group_node_state_losses__d = (
        r.group_node_state_losses__dt[
            r.group_node_state_losses__dt.index.get_level_values('period').isin(s.d_realized_period)
        ].groupby(level='period').sum() if not r.group_node_state_losses__dt.empty
        else pd.DataFrame(0.0, index=s.d_realized_period, columns=s.groupOutputNodeFlows))
    
    # r_group_node slacks
    r.group_node_up_slack__dt = pd.DataFrame(index=s.dt_realize_dispatch)
    r.group_node_down_slack__dt = pd.DataFrame(index=s.dt_realize_dispatch)
    for g in s.groupOutputNodeFlows:
        g_node = s.group_node[s.group_node.get_level_values('group').isin([g])].get_level_values('node')
        r.group_node_up_slack__dt[g] = r.upward_node_slack_dt[g_node].sum(axis=1)
        r.group_node_down_slack__dt[g] = r.downward_node_slack_dt[g_node].sum(axis=1)

    # r.node_up_slack__d = r.group_node_up_slack__dt.groupby('period').sum()
    # r.node_down_slack__d = r.group_node_down_slack__dt.groupby('period').sum()


    # r_storage_usage_dt
    dt_fix_idx = s.dt_fix_storage_timesteps
    r_storage_usage = {}
    
    for n in s.node:
        if (n, 'fix_usage') in s.node__storage_nested_fix_method:
            usage = pd.Series(0.0, index=dt_fix_idx)
            
            for col in r.flow_dt.columns:
                p, source, sink = col
                if source == n:
                    usage += (r.flow_dt[col].reindex(dt_fix_idx, fill_value=0) * 
                            par.step_duration.reindex(dt_fix_idx, fill_value=0))
                if sink == n:
                    usage -= (r.flow_dt[col].reindex(dt_fix_idx, fill_value=0) * 
                            par.step_duration.reindex(dt_fix_idx, fill_value=0))
            
            r_storage_usage[n] = usage
    
    r.storage_usage_dt = (pd.DataFrame(r_storage_usage, dtype=float) 
        if r_storage_usage else pd.DataFrame(0.0, index=dt_fix_idx, columns=[]))
    
    return r

def drop_levels(par, s, v):
    v.flow = v.flow.droplevel('solve')
    v.ramp = v.ramp.droplevel('solve')
    v.reserve = v.reserve.droplevel('solve')
    v.state = v.state.droplevel('solve')
    v.online_linear = v.online_linear.droplevel('solve')
    v.startup_linear = v.startup_linear.droplevel('solve')
    v.shutdown_linear = v.shutdown_linear.droplevel('solve')
    v.online_integer = v.online_integer.droplevel('solve')
    v.startup_integer = v.startup_integer.droplevel('solve')
    v.shutdown_integer = v.shutdown_integer.droplevel('solve')
    v.q_state_up = v.q_state_up.droplevel('solve')
    v.q_state_down = v.q_state_down.droplevel('solve')
    v.q_reserve = v.q_reserve.droplevel('solve')
    v.q_inertia = v.q_inertia.droplevel('solve')
    v.q_non_synchronous = v.q_non_synchronous.droplevel('solve')
    v.q_state_up_group = v.q_state_up_group.droplevel('solve')
    v.q_capacity_margin = v.q_capacity_margin.droplevel('solve')
    v.invest = v.invest.droplevel('solve')
    v.divest = v.divest.droplevel('solve')
    v.dual_invest_connection = v.dual_invest_connection.droplevel('solve')
    v.dual_invest_node = v.dual_invest_node.droplevel('solve')
    v.dual_invest_unit = v.dual_invest_unit.droplevel('solve')

    par.step_duration = par.step_duration.droplevel('solve')
    par.flow_min = par.flow_min.droplevel('solve')
    par.flow_max = par.flow_max.droplevel('solve')
    par.process_availability = par.process_availability.droplevel('solve')
    par.process_source_sink_varCost = par.process_source_sink_varCost.droplevel('solve')
    par.process_slope = par.process_slope.droplevel('solve')
    par.process_section = par.process_section.droplevel('solve')
    par.node_self_discharge_loss = par.node_self_discharge_loss.droplevel('solve')
    par.node_penalty_up = par.node_penalty_up.droplevel('solve')
    par.node_penalty_down = par.node_penalty_down.droplevel('solve')
    par.node_inflow = par.node_inflow.droplevel('solve')
    par.commodity_price = par.commodity_price.droplevel('solve')
    par.group_co2_price = par.group_co2_price.droplevel('solve')
    par.reserve_upDown_group_reservation = par.reserve_upDown_group_reservation.droplevel('solve')
    par.profile = par.profile.droplevel('solve')
    par.years_from_start_d = par.years_from_start_d.droplevel('solve')
    par.years_from_start_d = par.years_from_start_d[~par.years_from_start_d.index.duplicated(keep='first')]
    par.years_represented_d = par.years_represented_d.droplevel('solve')
    par.years_represented_d = par.years_represented_d[~par.years_represented_d.index.duplicated(keep='first')]
    par.entity_max_units = par.entity_max_units.droplevel('solve')
    par.entity_max_units = par.entity_max_units[~par.entity_max_units.index.duplicated(keep='first')]
    par.entity_all_existing = par.entity_all_existing.droplevel('solve')
    par.entity_all_existing = par.entity_all_existing[~par.entity_all_existing.index.duplicated(keep='first')]
    par.process_startup_cost = par.process_startup_cost.droplevel('solve')
    par.process_startup_cost = par.process_startup_cost[~par.process_startup_cost.index.duplicated(keep='first')]
    par.entity_fixed_cost = par.entity_fixed_cost.droplevel('solve')
    par.entity_fixed_cost = par.entity_fixed_cost[~par.entity_fixed_cost.index.duplicated(keep='first')]
    par.entity_lifetime_fixed_cost = par.entity_lifetime_fixed_cost.droplevel('solve')
    par.entity_lifetime_fixed_cost = par.entity_lifetime_fixed_cost[~par.entity_lifetime_fixed_cost.index.duplicated(keep='first')]
    par.entity_lifetime_fixed_cost_divest = par.entity_lifetime_fixed_cost_divest.droplevel('solve')
    par.entity_lifetime_fixed_cost_divest = par.entity_lifetime_fixed_cost_divest[~par.entity_lifetime_fixed_cost_divest.index.duplicated(keep='first')]
    par.node_annual_flow = par.node_annual_flow.droplevel('solve')
    par.node_annual_flow = par.node_annual_flow[~par.node_annual_flow.index.duplicated(keep='first')]
    par.group_penalty_inertia = par.group_penalty_inertia.droplevel('solve')
    par.group_penalty_inertia = par.group_penalty_inertia[~par.group_penalty_inertia.index.duplicated(keep='first')]
    par.group_penalty_non_synchronous = par.group_penalty_non_synchronous.droplevel('solve')
    par.group_penalty_non_synchronous = par.group_penalty_non_synchronous[~par.group_penalty_non_synchronous.index.duplicated(keep='first')]
    par.group_penalty_capacity_margin = par.group_penalty_capacity_margin.droplevel('solve')
    par.group_inertia_limit = par.group_inertia_limit.droplevel('solve')
    par.group_inertia_limit = par.group_inertia_limit[~par.group_inertia_limit.index.duplicated(keep='first')]
    par.group_capacity_margin = par.group_capacity_margin.droplevel('solve')
    par.entity_annual_discounted = par.entity_annual_discounted.droplevel('solve')
    par.entity_annual_divest_discounted = par.entity_annual_divest_discounted.droplevel('solve')
    par.discount_factor_operations_yearly = par.discount_factor_operations_yearly.droplevel('solve')
    par.discount_factor_operations_yearly = par.discount_factor_operations_yearly[~par.discount_factor_operations_yearly.index.duplicated(keep='first')]
    par.discount_factor_investment_yearly = par.discount_factor_investment_yearly.droplevel('solve')
    par.node_capacity_for_scaling = par.node_capacity_for_scaling.droplevel('solve')
    par.node_capacity_for_scaling = par.node_capacity_for_scaling[~par.node_capacity_for_scaling.index.duplicated(keep='first')]
    par.group_capacity_for_scaling = par.group_capacity_for_scaling.droplevel('solve')
    par.group_capacity_for_scaling = par.group_capacity_for_scaling[~par.group_capacity_for_scaling.index.duplicated(keep='first')]
    par.complete_period_share_of_year = par.complete_period_share_of_year.droplevel('solve')
    par.complete_period_share_of_year = par.complete_period_share_of_year[~par.complete_period_share_of_year.index.duplicated(keep='first')]

    s.solve_period = s.period
    s.period = s.period.droplevel('solve')
    s.period__time_first = s.period__time_first.droplevel('solve')
    s.period_first_of_solve = s.period_first_of_solve.droplevel('solve')
    s.period_in_use = s.period_in_use.droplevel('solve').unique()
    s.d_realize_dispatch_or_invest = s.d_realize_dispatch_or_invest.droplevel('solve').unique()
    s.d_realize_invest = s.d_realize_invest.droplevel('solve')
    s.d_realized_period = s.d_realized_period.droplevel('solve').unique()
    s.dt = s.dt.droplevel('solve')
    s.dt_fix_storage_timesteps = s.dt_fix_storage_timesteps.droplevel('solve')
    s.dt_realize_dispatch = s.dt_realize_dispatch.droplevel('solve')
    s.dtt = s.dtt.droplevel('solve')
    s.dtttdt = s.dtttdt.droplevel('solve')
    s.ed_invest = s.ed_invest.droplevel('solve').join(s.d_realize_invest, how='inner')
    s.edd_invest = s.edd_invest.droplevel('solve')
    s.edd_invest.names = ['entity', 'period_invest', 'period']
    s.edd_invest = s.edd_invest.join(s.d_realize_invest, how='inner')
    s.ed_divest = s.ed_divest.droplevel('solve').join(s.d_realize_invest, how='inner')

    return par, s, v