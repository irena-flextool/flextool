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
    
    # entity_all_capacity
      # Existing capacity
    entity_all_capacity = par.entity_all_existing.copy()
    periods = entity_all_capacity.index.get_level_values('period')
      # Add investments
    if not v.invest.empty:
        v_reindexed = v.invest.reindex(columns=pd.MultiIndex.from_product([par.entity_unitsize.index]), fill_value=0)
        capacity_add = v_reindexed.mul(par.entity_unitsize.values, axis=1)
        for i, period in enumerate(periods):
            entity_all_capacity.loc[period] += capacity_add.loc[periods[:i+1]].sum()
      # Subtract divestments
    if not v.divest.empty:
        v_reindexed = v.divest.reindex(columns=pd.MultiIndex.from_product([par.entity_unitsize.index]), fill_value=0)
        capacity_divest = v_reindexed.mul(par.entity_unitsize.values, axis=1)
        for i, period in enumerate(periods):
            entity_all_capacity.loc[period] -= capacity_divest.loc[periods[:i+1]].sum()
    r.entity_all_capacity = entity_all_capacity


    # r_process_Online__dt - just sum the two DataFrames
    r.process_online_dt = v.online_linear.add(v.online_integer, fill_value=0)

    # Calculate r_process__source__sink_Flow_dt
    s.process_source_sink_alwaysProcess = pd.MultiIndex.from_frame(
        s.process_method_sources_sinks[['process', 'always_source', 'always_sink']],
        names=['process', 'source', 'sink']
    )
    r.flow_dt = pd.DataFrame(
        index=s.dt_realize_dispatch,
        columns=s.process_source_sink_alwaysProcess,
        dtype=float
    )    
    unitsize = par.entity_unitsize
    slope = par.process_slope
    section = par.process_section
    
    for _, row in s.process_method_sources_sinks.iterrows():
        p = row['process']
        method = row['method']
        orig_source, orig_sink = row['orig_source'], row['orig_sink']
        always_source, always_sink = row['always_source'], row['always_sink']
        
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
    if not r.flow_dt.empty:
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
    r_node_ramp_dt = pd.DataFrame(index=r.ramp_dtt.index.droplevel('t_previous'), columns=s.node_balance, dtype=float)
    for n in s.node_balance:
        node_ramp = pd.Series(0.0, index=r_node_ramp_dt.index)
        # Flows into node (process, n, sink)
        for col in r.ramp_dtt.columns:
            if col[1] == n:  # source == n
                node_ramp += r.ramp_dtt[col].droplevel('t_previous')
        # Flows out of node (process, source, n) - negative
        for col in r.ramp_dtt.columns:
            if col[2] == n:  # sink == n
                node_ramp -= r.ramp_dtt[col].droplevel('t_previous')
        r_node_ramp_dt[n] = node_ramp

    # r_connection_dt - net flow through connections
    r.connection_dt = pd.DataFrame(index=s.dt_realize_dispatch, columns=s.process_connection, dtype=float)
    for c in s.process_connection:
        conn_flow = pd.Series(0.0, index=r.flow_dt.index)
        # Flow to right: (c, c, n) where (c, n) in process_sink
        for col in r.flow_dt.columns:
            if col[0] == c and col[1] == c and (c, col[2]) in s.process_sink:
                conn_flow += r.flow_dt[col]
        # Flow to left: (c, c, n) where (c, n) in process_source - negative
        for col in r.flow_dt.columns:
            if col[0] == c and col[1] == c and (c, col[2]) in s.process_source:
                conn_flow -= r.flow_dt[col]
        r.connection_dt[c] = conn_flow

    # r_connection_to_left_node__dt and r_connection_to_right_node__dt
    r_conn_left = pd.DataFrame(index=r.flow_dt.index, columns=s.process_connection, dtype=float)
    r_conn_right = pd.DataFrame(index=r.flow_dt.index, columns=s.process_connection, dtype=float)
    
    for c in s.process_connection:
        left_flow = pd.Series(0.0, index=r.flow_dt.index)
        right_flow = pd.Series(0.0, index=r.flow_dt.index)
        
        for col in r.flow_dt.columns:
            if col[0] == c and col[1] == c:
                if (c, col[2]) in s.process_source:
                    left_flow += r.flow_dt[col]
                if (c, col[2]) in s.process_sink:
                    right_flow += r.flow_dt[col]
        
        r_conn_left[c] = left_flow
        r_conn_right[c] = right_flow
    
    r.connection_to_left_node__dt = r_conn_left
    r.connection_to_right_node__dt = r_conn_right
    r.connection_to_left_node__d = r.connection_to_left_node__dt.groupby('period').sum()
    r.connection_to_right_node__d = r.connection_to_right_node__dt.groupby('period').sum()
    
    # r_group_output__connection_Not_in_aggregate__dt
    r.group_output__connection_not_in_aggregate__dt = pd.DataFrame(index=s.dt_realize_dispatch, columns=s.group_output__connection_Not_in_aggregate, dtype=float)
    for _, row in s.group_output__connection_Not_in_aggregate.iterrows():
        g, c = row['group'], row['connection']
        group_flow = pd.Series(0.0, index=r.flow_dt.index)
        
        # Connection to node
        conn_to_node = s.group_output__process__connection__to_node_Not_in_aggregate[
            (s.group_output__process__connection__to_node_Not_in_aggregate['group'] == g) &
            (s.group_output__process__connection__to_node_Not_in_aggregate['connection'] == c)
        ]
        for _, conn_row in conn_to_node.iterrows():
            col = (c, c, conn_row['node'])
            if col in r.flow_dt.columns:
                group_flow += r.flow_dt[col]
    
        # Node to connection
        node_to_conn = s.group_output__process__node__to_connection_Not_in_aggregate[
            (s.group_output__process__node__to_connection_Not_in_aggregate['group'] == g) &
            (s.group_output__process__node__to_connection_Not_in_aggregate['connection'] == c)
        ]
        for _, conn_row in node_to_conn.iterrows():
            col = (c, conn_row['node'], c)
            if col in r.flow_dt.columns:
                group_flow -= r.flow_dt[col]
    
        r.group_output__connection_not_in_aggregate__dt[(g, c)] = group_flow

    # r_group_output__connection_Not_in_aggregate__d
    if not r.group_output__connection_not_in_aggregate__dt.empty:
        r_group_conn_d = r.group_output__connection_not_in_aggregate__dt[
            r.group_output__connection_not_in_aggregate__dt.index.get_level_values('period').isin(s.d_realized_period)
        ].groupby(level='period').sum()
        r.group_output__connection_not_in_aggregate__d = r_group_conn_d
    else:
        r.group_output__connection_not_in_aggregate__d = pd.DataFrame(index=s.d_realized_period, columns=s.group_output__connection_Not_in_aggregate, dtype=float)
    
    # r_process_source_sink_flow_d - with step_duration
    r.flow_d = pd.DataFrame(index=s.d_realized_period, columns=s.process_source_sink_alwaysProcess, dtype=float)
    r.flow_d = r.flow_dt.mul(step_duration, axis=0).groupby('period').sum()
    
    # r_process_source_flow_d - sum over sinks
    r.process_source_flow_d = pd.DataFrame(index=s.d_realized_period, columns=s.process_source, dtype=float)
    for (p, source) in s.process_source:
        flow_sum = pd.Series(0.0, index=s.d_realized_period)
        for col in r.flow_d.columns:
            if col[0] == p and col[1] == source:
                flow_sum += r.flow_d[col]
        r.process_source_flow_d[p, source] = flow_sum
    
    # r_process_sink_flow_d - sum over sources
    r.process_sink_flow_d = pd.DataFrame(index=s.d_realized_period, columns=s.process_sink, dtype=float)
    for (p, sink) in s.process_sink:
        flow_sum = pd.Series(0.0, index=s.d_realized_period)
        for col in r.flow_d.columns:
            if col[0] == p and col[2] == sink:
                flow_sum += r.flow_d[col]
        r.process_sink_flow_d[p, sink] = flow_sum
    
    # r_connection_d - with step_duration
    r_conn_weighted = r.connection_dt.mul(step_duration, axis=0)    
    if not r_conn_weighted.empty:
        r_conn_d = r_conn_weighted[
            r_conn_weighted.index.get_level_values('period').isin(s.d_realized_period)
        ].groupby(level='period').sum()
        r.connection_d = r_conn_d
    else:
        r.connection_d = pd.DataFrame(index=s.d_realized_period)
    
    # r_connection_to_left_node__d and r_connection_to_right_node__d
    if 'output_connection_flow_separate' in s.enable_optional_outputs:
        r_conn_left_weighted = r_conn_left.mul(step_duration, axis=0)
        r_conn_right_weighted = r_conn_right.mul(step_duration, axis=0)
        
        if not r_conn_left_weighted.empty:
            r_conn_left_d = r_conn_left_weighted[
                r_conn_left_weighted.index.get_level_values('period').isin(s.d_realized_period)
            ].groupby(level='period').sum()
            r.connection_to_left_node__d = r_conn_left_d
        
        if not r_conn_right_weighted.empty:
            r_conn_right_d = r_conn_right_weighted[
                r_conn_right_weighted.index.get_level_values('period').isin(s.d_realized_period)
            ].groupby(level='period').sum()
            r.connection_to_right_node__d = r_conn_right_d

    # Calculate r_node_state_change_dt
    # Filter dt_realize_dispatch
    dt_dispatch_idx = s.dt_realize_dispatch
    # Initialize result
    r_state_change = pd.DataFrame(0.0, index=s.dt_realize_dispatch, columns=s.node_state, dtype=float)
    # Create index mappings from dtttdt
    prev_period_idx = s.dtttdt.droplevel(['time', 't_previous_within_timeset', 'd_previous', 't_previous_within_solve']).set_names(['period', 'time'])
    prev_timeset_idx = s.dtttdt.droplevel(['time', 't_previous', 'd_previous', 't_previous_within_solve']).set_names(['period', 'time'])

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
        v_prev_timeset = pd.Series(v.state[n].squeeze().reindex(prev_timeset_idx).values, index=current_idx)

        # bind_forward_only: change from start to finish, leaving first timestep empty
        # (uses same timeline as bind_within_timeset)
        if (n, 'bind_forward_only') in s.node__storage_binding_method:
            mask = ~current_idx.isin(exclude_idx)
            state_change += ((v_current - v_prev_timeset) * unitsize[n]).where(mask, 0)

        # bind_within_solve: treated as bind_within_period since solve info is not available
        if (n, 'bind_within_solve') in s.node__storage_binding_method:
            state_change += (v_current - v_prev_period) * unitsize[n]

        # bind_within_period: wraps the change over (difference between last and first
        # timestep in period is assigned to first timestep)
        if (n, 'bind_within_period') in s.node__storage_binding_method:
            state_change += (v_current - v_prev_period) * unitsize[n]

        # bind_within_timeset: continues timeline from one period to next, wraps over
        # the whole set of results
        if (n, 'bind_within_timeset') in s.node__storage_binding_method:
            state_change += (v_current - v_prev_timeset) * unitsize[n]

        # Assign
        r_state_change[n] = state_change
    
    r.node_state_change_dt = r_state_change

    # r_node_state_change_d - sum over dt_realize_dispatch
    r_state_change_d = r.node_state_change_dt[
        r.node_state_change_dt.index.get_level_values('period').isin(s.d_realized_period)
    ].groupby(level='period').sum()
    r.node_state_change_d = r_state_change_d
    
    # r_self_discharge_loss_dt - element-wise multiplication
    r.self_discharge_loss_dt = pd.DataFrame(index=dt_dispatch_idx, columns=s.node_self_discharge, dtype=float)
    for n in s.node_self_discharge:
        if n in v.state.columns and n in par.node_self_discharge_loss.columns:
            r.self_discharge_loss_dt[n] = (
                v.state[n].reindex(dt_dispatch_idx) * 
                par.node_self_discharge_loss[n].reindex(dt_dispatch_idx) * 
                par.entity_unitsize[n]
            )
    
    # r_self_discharge_loss_d - multiply by step_duration then sum
    r.self_discharge_loss_d = r.self_discharge_loss_dt.mul(step_duration, axis=0).groupby('period').sum()

    # r_cost_commodity_dt
    commodity_node_columns = pd.MultiIndex.from_tuples(list(s.commodity_node), names=['commodity', 'node'])
    r_commodity_cost = pd.DataFrame(index=r.flow_dt.index, columns=commodity_node_columns, dtype=float)
    for (c, n) in s.commodity_node:
        net_flow = pd.Series(0.0, index=r.flow_dt.index)
        # Flows into node (p, n, sink)
        for col in r.flow_dt.columns:
            if col[1] == n:
                net_flow += r.flow_dt[col]
        # Flows out of node (p, source, n) - negative
        for col in r.flow_dt.columns:
            if col[2] == n:
                net_flow -= r.flow_dt[col]
        # Multiply by step_duration and price
        cost = net_flow.copy()
        for idx in cost.index:
            if idx in par.step_duration.index and idx in par.commodity_price.index:
                cost.loc[idx] *= par.step_duration.loc[idx] * par.commodity_price.loc[idx, c]
        r_commodity_cost[(c, n)] = cost
    r.cost_commodity_dt = r_commodity_cost

    # r_process_commodity_d
    process_commodity_columns = pd.MultiIndex.from_frame(s.process__commodity__node)
    r.process_commodity_d = pd.DataFrame(index=s.d_realized_period, columns=process_commodity_columns)
    for _, row in s.process__commodity__node.iterrows():
        p, c, n = row['process'], row['commodity'], row['node']
        net_flow = pd.Series(0.0, index=r.flow_d.index)
        # Flows into node (p, n, sink)
        for col in r.flow_d.columns:
            if col[0] == p and col[1] == n:
                net_flow += r.flow_d[col]
        # Flows out of node (p, source, n) - negative
        for col in r.flow_d.columns:
            if col[0] == p and col[2] == n:
                net_flow -= r.flow_d[col]
        r.process_commodity_d[(p, c, n)] = net_flow
     
    # r_process_emissions_co2_dt
    process_commodity_node_columns = pd.MultiIndex.from_frame(s.process__commodity__node_co2[['process', 'commodity', 'node']])
    r.process_emissions_co2_dt = pd.DataFrame(index=r.flow_dt.index, columns=process_commodity_node_columns, dtype=float)
    for _, row in s.process__commodity__node_co2.iterrows():
        p, c, n = row['process'], row['commodity'], row['node']
        net_flow = pd.Series(0.0, index=r.flow_dt.index)
        # Flows into node (p, n, sink)
        for col in r.flow_dt.columns:
            if col[0] == p and col[1] == n:
                net_flow += r.flow_dt[col]
        # Flows out of node (p, source, n) - negative
        for col in r.flow_dt.columns:
            if col[0] == p and col[2] == n:
                net_flow -= r.flow_dt[col]
        # Multiply by step_duration and co2_content
        emissions = net_flow.copy()
        for idx in emissions.index:
            if idx in par.step_duration.index:
                emissions.loc[idx] *= (par.step_duration.loc[idx] * 
                                    par.commodity_co2_content.loc[c])
        r.process_emissions_co2_dt[(p, c, n)] = emissions
    
    # r_process_emissions_co2_d - sum and divide by complete_period_share_of_year
    r.process_emissions_co2_d = r.process_emissions_co2_dt.groupby(level='period').sum()
    r.process_emissions_co2_d = r.process_emissions_co2_d.div(par.complete_period_share_of_year, axis=0)
    
    # r_emissions_co2_d - sum processes over period
    r.emissions_co2_d = r.process_emissions_co2_d.sum(axis=1)

    # r_emissions_co2_dt - sum processes over period__time
    r.emissions_co2_dt = r.process_emissions_co2_dt.sum(axis=1)

    # r_cost_co2_dt
    r.cost_co2 = pd.DataFrame(index=s.dt_realize_dispatch, dtype=float)
    r.group_emissions_co2_dt = pd.DataFrame(
        index=r.process_emissions_co2_dt.index,
        columns=pd.MultiIndex.from_tuples([(g,) for g in s.group_co2_price['group']], names=['group'])
    )
    for group in s.group_co2_price['group']:
        # Get nodes for this group
        nodes_in_group = s.group_node[s.group_node.get_level_values('group') == group].get_level_values('node')
        prices = par.group_co2_price[group]
        # Sum all columns where node matches
        total = pd.Series(0.0, index=r.process_emissions_co2_dt.index)
        for col in r.process_emissions_co2_dt.columns:
            node = col[2]  # node is third level in (process, commodity, node)
            if node in nodes_in_group:
                total += r.process_emissions_co2_dt[col]
        r.group_emissions_co2_dt[group] = prices.mul(total, axis=0)

    r.cost_co2_dt = r.group_emissions_co2_dt.sum(axis=1)
    
    # r_cost_process_other_operational_cost_dt
    r.cost_process_other_operational_cost_dt = pd.DataFrame(0.0, index=r.flow_dt.index, columns=s.process, dtype=float)
    for p in s.process:
        for col in r.flow_dt.columns:
            if col[0] == p and col in par.process_source_sink_varCost.columns:
                r.cost_process_other_operational_cost_dt[p] += (par.step_duration * 
                                                                par.process_source_sink_varCost[col] * 
                                                                r.flow_dt[col])

    # r_process_startup_dt
    r.process_startup_dt = v.startup_linear.add(v.startup_integer, fill_value=0)
    
    # r_cost_startup_dt
    r.cost_startup_dt = pd.DataFrame(0.0, index=r.process_startup_dt.index, columns=s.process_online, dtype=float)
    for p in s.process_online:
        if p in r.process_startup_dt.columns and p in par.process_startup_cost.columns:
            cost = r.process_startup_dt[p] * par.entity_unitsize[p]
            for d in par.process_startup_cost.index:
                if p in par.process_startup_cost.columns:
                    period_mask = cost.index.get_level_values('period') == d
                    cost.loc[period_mask] *= par.process_startup_cost.loc[d, p]
            r.cost_startup_dt[p] = cost

    # r_costPenalty_node_state_upDown_dt
    nodes = list(s.node_balance) + list(s.node_balance_period)
    node_updown_columns = pd.MultiIndex.from_product([nodes, s.upDown], names=['node', 'upDown'])
    r_penalty_state = pd.DataFrame(
        index=v.q_state_up.index if hasattr(v, 'q_state_up') and not v.q_state_up.empty else v.q_state_down.index,
        columns=node_updown_columns, 
        dtype=float
    )
    for n in nodes:
        for ud in s.upDown:
            if ud == 'up' and n in v.q_state_up.columns:
                penalty = v.q_state_up[n] * par.node_penalty_up[n]
                for d in par.node_capacity_for_scaling.index:
                    if n in par.node_capacity_for_scaling.columns:
                        period_mask = penalty.index.get_level_values('period') == d
                        penalty.loc[period_mask] *= par.node_capacity_for_scaling.loc[d, n].drop_duplicates()
                r_penalty_state[(n, ud)] = penalty
            elif ud == 'down' and n in v.q_state_down.columns:
                penalty = v.q_state_down[n] * par.node_penalty_down[n]
                for d in par.node_capacity_for_scaling.index:
                    if n in par.node_capacity_for_scaling.columns:
                        period_mask = penalty.index.get_level_values('period') == d
                        penalty.loc[period_mask] *= par.node_capacity_for_scaling.loc[d, n].drop_duplicates()
                r_penalty_state[(n, ud)] = penalty
    r.costPenalty_node_state_upDown_dt = r_penalty_state

    # r_penalty_node_state_upDown_d
    if not r.costPenalty_node_state_upDown_dt.empty:
        r.penalty_node_state_upDown_d = r.costPenalty_node_state_upDown_dt[
            r.costPenalty_node_state_upDown_dt.index.get_level_values('period').isin(s.d_realized_period)
        ].groupby(level='period').sum()
    else:
        r.penalty_node_state_upDown_d = None

    # r_costPenalty_inertia_dt
    if not v.q_inertia.empty:
        r.costPenalty_inertia_dt = v.q_inertia \
            * par.group_inertia_limit \
            * par.group_penalty_inertia
    else: 
        r.costPenalty_inertia_dt = pd.DataFrame(index=v.q_inertia.index)

    # r_costPenalty_non_synchronous_dt
    if not v.q_non_synchronous.empty:
        r.costPenalty_non_synchronous_dt \
            = ( v.q_non_synchronous 
                * par.group_capacity_for_scaling[s.groupNonSync] 
                * par.group_penalty_non_synchronous )
    else:
        r.costPenalty_non_synchronous_dt = pd.DataFrame(index=v.q_non_synchronous.index)

    # r_costPenalty_capacity_margin_d
    if not v.q_capacity_margin.empty:
        r.costPenalty_capacity_margin_d \
            = ( v.q_capacity_margin
                * par.group_capacity_for_scaling[s.groupCapacityMargin]
                * par.group_penalty_capacity_margin
              ).mul(par.discount_factor_operations_yearly, axis=0)
    else:
        r.costPenalty_capacity_margin_d = pd.DataFrame(index=v.q_capacity_margin.index)
    
    # r_costPenalty_reserve_upDown_dt
    r.costPenalty_reserve_upDown_dt = pd.DataFrame(index=v.q_reserve.index, columns=v.q_reserve.columns, dtype=float)
    for col in v.q_reserve.columns:
        res, ud, ng = col
        penalty = (v.q_reserve[col].mul(par.step_duration, axis=0) * 
                par.reserve_upDown_group_penalty.loc[(res, ud, ng)] * 
                par.reserve_upDown_group_reservation[col])
        r.costPenalty_reserve_upDown_dt[col] = penalty

    # r_cost_entity_invest_d
    r.cost_entity_invest_d = pd.DataFrame(index=v.invest.index, columns=v.invest.columns, dtype=float)
    for e in v.invest.columns:
        for d in v.invest.index:
            if (e, d) in s.ed_invest:
                r.cost_entity_invest_d.loc[d, e] = (v.invest.loc[d, e] * 
                                        par.entity_unitsize[e] * 
                                        par.entity_annual_discounted.loc[d, e])
    
    # r_cost_entity_divest_d
    r.cost_entity_divest_d = pd.DataFrame(index=v.divest.index, columns=v.divest.columns, dtype=float)
    for e in v.divest.columns:
        for d in v.divest.index:
            if (e, d) in s.ed_divest:
                r.cost_entity_divest_d.loc[d, e] = (-v.divest.loc[d, e] * 
                                        par.entity_unitsize[e] * 
                                        par.entity_annual_divest_discounted.loc[d, e])
    
    # r_cost_entity_existing_fixed
    combined_fixed = pd.concat([par.process_fixed_cost, par.node_fixed_cost], axis=1).rename_axis('entity', axis=1)
    r.cost_entity_existing_fixed = (r.entity_all_capacity * combined_fixed * 1000).mul(par.discount_factor_operations_yearly, axis=0)

    # Aggregate costs
    r.costOper_dt = (r.cost_commodity_dt.sum(axis=1) + 
                        r.cost_process_other_operational_cost_dt.sum(axis=1) + 
                        r.cost_startup_dt.sum(axis=1) +
                        r.cost_co2_dt
                    )
    r.costPenalty_dt = (r.costPenalty_node_state_upDown_dt.sum(axis=1) + 
                            r.costPenalty_inertia_dt.sum(axis=1) + 
                            r.costPenalty_non_synchronous_dt.sum(axis=1) + 
                            r.costPenalty_reserve_upDown_dt.sum(axis=1))
    
    r.costOper_and_penalty_dt = r.costOper_dt + r.costPenalty_dt
    
    # Period aggregations
    r.cost_process_other_operational_cost_d = r.cost_process_other_operational_cost_dt.groupby('period').sum()
    
    r.cost_co2_d = (r.cost_co2_dt.groupby('period').sum() 
    if not r.cost_co2_dt.empty 
    else pd.Series(0.0, index=s.d_realized_period))

    r.cost_variable_d = r.cost_process_other_operational_cost_d.sum(axis=1)

    r.cost_startup_d = (r.cost_startup_dt[
        r.cost_startup_dt.index.get_level_values('period').isin(s.d_realized_period)
    ].groupby(level='period').sum() if not r.cost_startup_dt.empty 
    else pd.DataFrame(0.0, index=s.d_realized_period, columns=r.cost_startup_dt.columns if hasattr(r.cost_startup_dt, 'columns') else []))

    r.costPenalty_node_state_upDown_d = (r.costPenalty_node_state_upDown_dt[
        r.costPenalty_node_state_upDown_dt.index.get_level_values('period').isin(s.d_realized_period)
    ].groupby(level='period').sum() if not r.costPenalty_node_state_upDown_dt.empty 
    else pd.DataFrame(0.0, index=s.d_realized_period, columns=r.costPenalty_node_state_upDown_dt.columns if hasattr(r.costPenalty_node_state_upDown_dt, 'columns') else []))

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

    r.costOper_d = (r.costOper_dt.groupby(level='period').sum() 
    if not r.costOper_dt.empty 
    else pd.Series(0.0, index=s.d_realized_period))

    r.costPenalty_d = (r.costPenalty_dt.groupby(level='period').sum() 
    if not r.costPenalty_dt.empty 
    else pd.Series(0.0, index=s.d_realized_period))

    for d in s.d_realize_invest:
        if d in r.costPenalty_d.index and not r.costPenalty_capacity_margin_d.empty:
            r.costPenalty_d.loc[d] += r.costPenalty_capacity_margin_d.loc[d].sum()

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
    r.costExistingFixed_d = r.cost_entity_existing_fixed.sum(axis=1)

    # pdNodeInflow
    r.node_inflow_d = par.node_inflow.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)
    
    # potentialVREgen_dt
    r.potentialVREgen_dt = pd.DataFrame(index=s.dt_realize_dispatch, columns=s.process_VRE, dtype=float)
    for _, row in s.process_VRE.iterrows():
        p, n = row['process'], row['node']
        if (p, n) not in s.process_sink:
            continue
        # Find matching profile method with upper_limit
        profile_methods = s.process__source__sink__profile__profile_method[
            (s.process__source__sink__profile__profile_method['process'] == p) &
            (s.process__source__sink__profile__profile_method['sink'] == n) &
            (s.process__source__sink__profile__profile_method['method'] == 'upper_limit')
        ]
        if not profile_methods.empty:
            f = profile_methods['profile'].iloc[0]
            if (f in par.profile.columns and 
                p in r.entity_all_capacity.columns and
                p in par.process_availability.columns):
                r.potentialVREgen_dt[p, n] = par.profile[f].squeeze() * par.process_availability[p].squeeze() * r.entity_all_capacity[p].squeeze()
    #r.potentialVREgen_dt = pd.DataFrame(vre_potential_dt, dtype=float) if vre_potential_dt else pd.DataFrame()
    #r.potentialVREgen_dt.columns.names=['process', 'node']
    
    # potentialVREgen - sum over dt_realize_dispatch
    if not r.potentialVREgen_dt.empty:
        r.potentialVREgen = r.potentialVREgen_dt[
            r.potentialVREgen_dt.index.get_level_values('period').isin(s.d_realized_period)
        ].groupby(level='period').sum()
    else:
        r.potentialVREgen = pd.DataFrame(0.0, index=s.d_realized_period, columns=[])

    # r_group_output__group_aggregate_Unit_to_group__dt
    r_unit_to_group = {}
    for _, row in s.group_output__group_aggregate_Unit_to_group.iterrows():
        g, ga = row['group'], row['group_aggregate']
        flow_sum = pd.Series(0.0, index=dt_dispatch_idx)
        
        matches = s.group_output__group_aggregate__process__unit__to_node[
            (s.group_output__group_aggregate__process__unit__to_node['group'] == g) &
            (s.group_output__group_aggregate__process__unit__to_node['group_aggregate'] == ga)
        ]
        for _, match_row in matches.iterrows():
            u, source, sink = match_row['unit'], match_row['source'], match_row['sink']
            if (u, source, sink) in r.flow_dt.columns:
                flow_sum += r.flow_dt[(u, source, sink)].reindex(dt_dispatch_idx, fill_value=0)
        
        r_unit_to_group[(g, ga)] = flow_sum
    
    r.group_output__group_aggregate_Unit_to_group__dt = (pd.DataFrame(r_unit_to_group, dtype=float) 
        if r_unit_to_group else pd.DataFrame(0.0, index=dt_dispatch_idx, columns=[]))
    
    # r_group_output__group_aggregate_Unit_to_group__d
    r.group_output__group_aggregate_Unit_to_group__d = (
        r.group_output__group_aggregate_Unit_to_group__dt[
            r.group_output__group_aggregate_Unit_to_group__dt.index.get_level_values('period').isin(s.d_realized_period)
        ].groupby(level='period').sum() if not r.group_output__group_aggregate_Unit_to_group__dt.empty
        else pd.DataFrame(0.0, index=s.d_realized_period, columns=[]))
    
    # r_group_output__group_aggregate_Group_to_unit__dt
    r_group_to_unit = {}
    for _, row in s.group_output__group_aggregate_Group_to_unit.iterrows():
        g, ga = row['group'], row['group_aggregate']
        flow_sum = pd.Series(0.0, index=dt_dispatch_idx)
        
        matches = s.group_output__group_aggregate__process__node__to_unit[
            (s.group_output__group_aggregate__process__node__to_unit['group'] == g) &
            (s.group_output__group_aggregate__process__node__to_unit['group_aggregate'] == ga)
        ]
        for _, match_row in matches.iterrows():
            u, source, sink = match_row['unit'], match_row['source'], match_row['sink']
            if (u, source, sink) in r.flow_dt.columns:
                flow_sum -= r.flow_dt[(u, source, sink)].reindex(dt_dispatch_idx, fill_value=0)
        
        r_group_to_unit[(g, ga)] = flow_sum
    
    r.group_output__group_aggregate_Group_to_unit__dt = (pd.DataFrame(r_group_to_unit, dtype=float)
        if r_group_to_unit else pd.DataFrame(0.0, index=dt_dispatch_idx, columns=[]))
    
    # r_group_output__group_aggregate_Group_to_unit__d
    r.group_output__group_aggregate_Group_to_unit__d = (
        r.group_output__group_aggregate_Group_to_unit__dt[
            r.group_output__group_aggregate_Group_to_unit__dt.index.get_level_values('period').isin(s.d_realized_period)
        ].groupby(level='period').sum() if not r.group_output__group_aggregate_Group_to_unit__dt.empty
        else pd.DataFrame(0.0, index=s.d_realized_period, columns=[]))
    
    # r_group_output__group_aggregate_Connection__dt
    r_connection = {}
    for _, row in s.group_output__group_aggregate_Connection.iterrows():
        g, ga = row['group'], row['group_aggregate']
        flow_sum = pd.Series(0.0, index=dt_dispatch_idx)
        
        # Connection to node
        conn_to_node = s.group_output__group_aggregate__process__connection__to_node[
            (s.group_output__group_aggregate__process__connection__to_node['group'] == g) &
            (s.group_output__group_aggregate__process__connection__to_node['group_aggregate'] == ga)
        ]
        for _, match_row in conn_to_node.iterrows():
            c, sink = match_row['connection'], match_row['sink']
            if (c, c, sink) in r.flow_dt.columns:
                flow_sum += r.flow_dt[(c, c, sink)].reindex(dt_dispatch_idx, fill_value=0)
        
        # Node to connection
        node_to_conn = s.group_output__group_aggregate__process__node__to_connection[
            (s.group_output__group_aggregate__process__node__to_connection['group'] == g) &
            (s.group_output__group_aggregate__process__node__to_connection['group_aggregate'] == ga)
        ]
        for _, match_row in node_to_conn.iterrows():
            c, source = match_row['connection'], match_row['source']
            if (c, source, c) in r.flow_dt.columns:
                flow_sum -= r.flow_dt[(c, source, c)].reindex(dt_dispatch_idx, fill_value=0)
        
        r_connection[(g, ga)] = flow_sum
    
    r.group_output__group_aggregate_Connection__dt = (pd.DataFrame(r_connection, dtype=float)
        if r_connection else pd.DataFrame(0.0, index=dt_dispatch_idx, columns=[]))
    
    # r_group_output__group_aggregate_Connection__d
    r.group_output__group_aggregate_Connection__d = (
        r.group_output__group_aggregate_Connection__dt[
            r.group_output__group_aggregate_Connection__dt.index.get_level_values('period').isin(s.d_realized_period)
        ].groupby(level='period').sum() if not r.group_output__group_aggregate_Connection__dt.empty
        else pd.DataFrame(0.0, index=s.d_realized_period, columns=[]))
    
    # r_group_output_Internal_connection_losses__dt
    r_conn_losses = pd.DataFrame(0.0, index=dt_dispatch_idx, columns=s.groupOutputNodeFlows, dtype=float)
    for g in s.groupOutputNodeFlows:
        losses = pd.Series(0.0, index=dt_dispatch_idx)
        
        for col in r.flow_dt.columns:
            c, source, sink = col
            if c in s.process_connection and (g, c) in s.group_output__process_fully_inside:
                if (c, source) in s.process_source:
                    losses += r.flow_dt[col].reindex(dt_dispatch_idx, fill_value=0)
                if (c, sink) in s.process_sink:
                    losses -= r.flow_dt[col].reindex(dt_dispatch_idx, fill_value=0)
                if (c, source) in s.process_sink:
                    losses += r.flow_dt[col].reindex(dt_dispatch_idx, fill_value=0)
                if (c, sink) in s.process_source:
                    losses -= r.flow_dt[col].reindex(dt_dispatch_idx, fill_value=0)
        
        r_conn_losses[g] = losses
    
    r.group_output_Internal_connection_losses__dt = r_conn_losses
    
    # r_group_output_Internal_connection_losses__d
    r.group_output_Internal_connection_losses__d = (
        r.group_output_Internal_connection_losses__dt[
            r.group_output_Internal_connection_losses__dt.index.get_level_values('period').isin(s.d_realized_period)
        ].groupby(level='period').sum() if not r.group_output_Internal_connection_losses__dt.empty
        else pd.DataFrame(0.0, index=s.d_realized_period, columns=s.groupOutputNodeFlows))
    
    # r_group_output_Internal_unit_losses__dt
    r_unit_losses = pd.DataFrame(0.0, index=dt_dispatch_idx, columns=s.groupOutputNodeFlows, dtype=float)
    for g in s.groupOutputNodeFlows:
        losses = pd.Series(0.0, index=dt_dispatch_idx)
        
        for col in r.flow_dt.columns:
            u, source, sink = col
            if u in s.process_unit and (g, u) in s.group_output__process_fully_inside:
                if (g, source) in s.group_node:
                    losses += r.flow_dt[col].reindex(dt_dispatch_idx, fill_value=0)
                if (g, sink) in s.group_node:
                    losses -= r.flow_dt[col].reindex(dt_dispatch_idx, fill_value=0)
        
        r_unit_losses[g] = losses
    
    r.group_output_Internal_unit_losses__dt = r_unit_losses
    
    # r_group_output_Internal_unit_losses__d
    r.group_output_Internal_unit_losses__d = (
        r.group_output_Internal_unit_losses__dt[
            r.group_output_Internal_unit_losses__dt.index.get_level_values('period').isin(s.d_realized_period)
        ].groupby(level='period').sum() if not r.group_output_Internal_unit_losses__dt.empty
        else pd.DataFrame(0.0, index=s.d_realized_period, columns=s.groupOutputNodeFlows))
    
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
    
    # r_group_node penalties
    r.group_node_up_penalties__dt = (v.q_state_up * par.node_capacity_for_scaling).mul(par.step_duration, axis=0)
    r.group_node_up_penalties__d = r.group_node_up_penalties__dt.groupby('period').sum()
    r.group_node_down_penalties__dt = (v.q_state_down * par.node_capacity_for_scaling).mul(par.step_duration, axis=0)
    r.group_node_down_penalties__d = r.group_node_down_penalties__dt.groupby('period').sum()
    
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
    par.entity_max_units = par.entity_max_units.droplevel('solve').drop_duplicates()
    par.entity_max_units = par.entity_max_units[~par.entity_max_units.index.duplicated(keep='first')]
    par.entity_all_existing = par.entity_all_existing.droplevel('solve').drop_duplicates()
    par.entity_all_existing = par.entity_all_existing[~par.entity_all_existing.index.duplicated(keep='first')]
    par.process_startup_cost = par.process_startup_cost.droplevel('solve')
    par.process_startup_cost = par.process_startup_cost[~par.process_startup_cost.index.duplicated(keep='first')]
    par.process_fixed_cost = par.process_fixed_cost.droplevel('solve')
    par.process_fixed_cost = par.process_fixed_cost[~par.process_fixed_cost.index.duplicated(keep='first')]
    par.node_fixed_cost = par.node_fixed_cost.droplevel('solve')
    par.node_fixed_cost = par.node_fixed_cost[~par.node_fixed_cost.index.duplicated(keep='first')]
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
    s.ed_invest = s.ed_invest.droplevel('solve')
    s.edd_invest = s.edd_invest.droplevel('solve')
    s.ed_divest = s.ed_divest.droplevel('solve')

    return par, s, v