from types import SimpleNamespace
import pandas as pd

def post_process_results(self):
    """Calculate post-processing results from variables, parameters, and sets"""
    
    drop_levels(self)

    # hours_in_realized_period
    # Filter dt_realize_dispatch by periods in d_realized_period, then group and sum
    step_duration = self.p.step_duration
    
    hours_in_realized_period = step_duration.groupby(level='period').sum()
    hours_in_realized_period = hours_in_realized_period.reindex(self.s.d_realized_period)

    self.r.hours_in_realized_period = hours_in_realized_period
    self.r.realized_period_share_of_year = hours_in_realized_period / 8760
    
    # entity_all_capacity
      # Existing capacity
    entity_all_capacity = self.p.entity_all_existing.copy()
    periods = entity_all_capacity.index.get_level_values('period').unique()
      # Add investments
    if not self.v.invest.empty:
        capacity_add = (self.v.invest * self.p.entity_unitsize.iloc[0]).fillna(0)
        for i, period in enumerate(periods):
            entity_all_capacity.loc[period] += capacity_add.loc[periods[:i+1]].sum()
      # Subtract divestments
    if not self.v.divest.empty:
        capacity_divest = (self.v.divest * self.p.entity_unitsize.iloc[0]).fillna(0)
        for i, period in enumerate(periods):
            entity_all_capacity.loc[period] -= capacity_divest.loc[periods[:i+1]].sum()
    self.r.entity_all_capacity = entity_all_capacity


    # r_process_Online__dt - just sum the two DataFrames
    self.r.process_online_dt = self.v.online_linear.add(self.v.online_integer, fill_value=0)

    # Calculate r_process__source__sink_Flow__dt
    r_flow_dt = pd.DataFrame(index=self.v.flow.index)
    unitsize = self.p.entity_unitsize
    slope = self.p.process_slope.droplevel(0, axis=1)
    section = self.p.process_section.droplevel(0, axis=1)
    
    for _, row in self.s.process_method_sources_sinks.iterrows():
        p = row['process']
        method = row['method']
        orig_source, orig_sink = row['orig_source'], row['orig_sink']
        always_source, always_sink = row['always_source'], row['always_sink']
        
        # Get base flow value
        flow_val = self.v.flow[(p, orig_source, orig_sink)] * unitsize[p]
        
        # Apply transformations for method_1var_per_way on source-to-process flows
        if (method in self.s.method_1var_per_way and 
            p not in self.s.process_profile and
            orig_source == always_source and orig_sink != always_sink):
            flow_val *= slope[p]
            if p in self.s.process_unit:
                flow_val /= (self.p.process_sink_coefficient.loc[p, orig_sink] *
                            self.p.process_source_coefficient.loc[p, orig_source])
            if (p, 'min_load_efficiency') in self.s.process__ct_method:
                flow_val += self.r.process_online_dt['process', p] * section[p] * unitsize[p]
        r_flow_dt[(p, always_source, always_sink)] = flow_val
    self.r.process_source_sink_flow_dt = r_flow_dt

    # r_process__source__sink_Flow__d - sum over dt_realize_dispatch
    if not r_flow_dt.empty:
        r_flow_d = r_flow_dt[r_flow_dt.index.get_level_values('period').isin(self.s.d_realized_period)].groupby(level='period').sum()
        self.r.process_source_sink_flow__d = r_flow_d

    # r_process_source_sink_ramp_dtt - difference between t and t_previous
    current_idx = self.s.dtt.droplevel('t_previous')
    previous_idx = self.s.dtt.droplevel('time')
    r_ramp_dtt = pd.DataFrame(
        r_flow_dt.reindex(current_idx).values - r_flow_dt.reindex(previous_idx).values,
        index=self.s.dtt,
        columns=r_flow_dt.columns
    )
    self.r.process_source_sink_ramp_dtt = r_ramp_dtt

    # r_node_ramp_dtt - sum ramps for flows into/out of each node
    r_node_ramp = pd.DataFrame(index=r_ramp_dtt.index.droplevel('t_previous'), columns=self.s.nodeBalance)
    for n in self.s.nodeBalance:
        node_ramp = pd.Series(0.0, index=r_node_ramp.index)
        # Flows into node (process, n, sink)
        for col in r_ramp_dtt.columns:
            if col[1] == n:  # source == n
                node_ramp += r_ramp_dtt[col].droplevel('t_previous')
        # Flows out of node (process, source, n) - negative
        for col in r_ramp_dtt.columns:
            if col[2] == n:  # sink == n
                node_ramp -= r_ramp_dtt[col].droplevel('t_previous')
        r_node_ramp[n] = node_ramp
    self.r.node_ramp_dtt = r_node_ramp

    # r_connection_dt - net flow through connections
    r_connection = pd.DataFrame(index=r_flow_dt.index, columns=self.s.process_connection)
    for c in self.s.process_connection:
        conn_flow = pd.Series(0.0, index=r_flow_dt.index)
        # Flow to right: (c, c, n) where (c, n) in process_sink
        for col in r_flow_dt.columns:
            if col[0] == c and col[1] == c and (c, col[2]) in self.s.process_sink:
                conn_flow += r_flow_dt[col]
        # Flow to left: (c, c, n) where (c, n) in process_source - negative
        for col in r_flow_dt.columns:
            if col[0] == c and col[1] == c and (c, col[2]) in self.s.process_source:
                conn_flow -= r_flow_dt[col]
        r_connection[c] = conn_flow
    self.r.connection_dt = r_connection

    # r_connection_to_left_node__dt and r_connection_to_right_node__dt
    if 'output_connection_flow_separate' in self.s.enable_optional_outputs:
        
        r_conn_left = pd.DataFrame(index=r_flow_dt.index, columns=self.s.process_connection)
        r_conn_right = pd.DataFrame(index=r_flow_dt.index, columns=self.s.process_connection)
        
        for c in self.s.process_connection:
            left_flow = pd.Series(0.0, index=r_flow_dt.index)
            right_flow = pd.Series(0.0, index=r_flow_dt.index)
            
            for col in r_flow_dt.columns:
                if col[0] == c and col[1] == c:
                    if (c, col[2]) in self.s.process_source:
                        left_flow += r_flow_dt[col]
                    if (c, col[2]) in self.s.process_sink:
                        right_flow += r_flow_dt[col]
            
            r_conn_left[c] = left_flow
            r_conn_right[c] = right_flow
        
        self.r.connection_to_left_node__dt = r_conn_left
        self.r.connection_to_right_node__dt = r_conn_right
    
    # r_group_output__connection_Not_in_aggregate__dt
    r_group_conn = {}
    for _, row in self.s.group_output__connection_Not_in_aggregate.iterrows():
        g, c = row['group'], row['connection']
        group_flow = pd.Series(0.0, index=r_flow_dt.index)
        
        # Connection to node
        conn_to_node = self.s.group_output__process__connection__to_node_Not_in_aggregate[
            (self.s.group_output__process__connection__to_node_Not_in_aggregate['group'] == g) &
            (self.s.group_output__process__connection__to_node_Not_in_aggregate['connection'] == c)
        ]
        for _, conn_row in conn_to_node.iterrows():
            col = (c, c, conn_row['node'])
            if col in r_flow_dt.columns:
                group_flow += r_flow_dt[col]
    
        # Node to connection
        node_to_conn = self.s.group_output__process__node__to_connection_Not_in_aggregate[
            (self.s.group_output__process__node__to_connection_Not_in_aggregate['group'] == g) &
            (self.s.group_output__process__node__to_connection_Not_in_aggregate['connection'] == c)
        ]
        for _, conn_row in node_to_conn.iterrows():
            col = (c, conn_row['node'], c)
            if col in r_flow_dt.columns:
                group_flow -= r_flow_dt[col]
    
        r_group_conn[(g, c)] = group_flow

    self.r.group_output__connection_not_in_aggregate__dt = pd.DataFrame(r_group_conn)
    
    # r_group_output__connection_Not_in_aggregate__d
    if not self.r.group_output__connection_not_in_aggregate__dt.empty:
        r_group_conn_d = self.r.group_output__connection_not_in_aggregate__dt[
            self.r.group_output__connection_not_in_aggregate__dt.index.get_level_values('period').isin(self.s.d_realized_period)
        ].groupby(level='period').sum()
        self.r.group_output__connection_not_in_aggregate__d = r_group_conn_d
    
    # r_process_source_sink_flow_d - with step_duration
    r_flow_weighted = r_flow_dt.copy()
    for idx in r_flow_weighted.index:
        if idx in step_duration.index:
            r_flow_weighted.loc[idx] *= step_duration.loc[idx]
    
    if not r_flow_weighted.empty:
        r_flow_d_weighted = r_flow_weighted[
            r_flow_weighted.index.get_level_values('period').isin(self.s.d_realized_period)
        ].groupby(level='period').sum()
        self.r.process_source_sink_flow_d = r_flow_d_weighted
    
    # r_process_source_flow_d - sum over sinks
    r_source_flow = {}
    for (p, source) in self.s.process_source:
        flow_sum = pd.Series(0.0, index=r_flow_d_weighted.index)
        for col in r_flow_d_weighted.columns:
            if col[0] == p and col[1] == source:
                flow_sum += r_flow_d_weighted[col]
        r_source_flow[(p, source)] = flow_sum
    
    self.r.process_source_flow_d = pd.DataFrame(r_source_flow)
    
    # r_process_sink_flow_d - sum over sources
    r_sink_flow = {}
    for (p, sink) in self.s.process_sink:
        flow_sum = pd.Series(0.0, index=r_flow_d_weighted.index)
        for col in r_flow_d_weighted.columns:
            if col[0] == p and col[2] == sink:
                flow_sum += r_flow_d_weighted[col]
        r_sink_flow[(p, sink)] = flow_sum
    
    self.r.process_sink_flow_d = pd.DataFrame(r_sink_flow)
    
    # r_connection_d - with step_duration
    r_conn_weighted = r_connection.copy()
    for idx in r_conn_weighted.index:
        if idx in step_duration.index:
            r_conn_weighted.loc[idx] *= step_duration.loc[idx]
    
    if not r_conn_weighted.empty:
        r_conn_d = r_conn_weighted[
            r_conn_weighted.index.get_level_values('period').isin(self.s.d_realized_period)
        ].groupby(level='period').sum()
        self.r.connection_d = r_conn_d
    
    # r_connection_to_left_node__d and r_connection_to_right_node__d
    if 'output_connection_flow_separate' in self.s.enable_optional_outputs:
        r_conn_left_weighted = r_conn_left.copy()
        r_conn_right_weighted = r_conn_right.copy()
        
        for idx in r_conn_left_weighted.index:
            if idx in step_duration.index:
                r_conn_left_weighted.loc[idx] *= step_duration.loc[idx]
                r_conn_right_weighted.loc[idx] *= step_duration.loc[idx]
        
        if not r_conn_left_weighted.empty:
            r_conn_left_d = r_conn_left_weighted[
                r_conn_left_weighted.index.get_level_values('period').isin(self.s.d_realized_period)
            ].groupby(level='period').sum()
            self.r.connection_to_left_node__d = r_conn_left_d
        
        if not r_conn_right_weighted.empty:
            r_conn_right_d = r_conn_right_weighted[
                r_conn_right_weighted.index.get_level_values('period').isin(self.s.d_realized_period)
            ].groupby(level='period').sum()
            self.r.connection_to_right_node__d = r_conn_right_d

    # Calculate r_nodeState_change_dt
    v_state = self.v.state.droplevel(0, axis=1)
    # Filter dt_realize_dispatch
    dt_dispatch_idx = self.s.dt_realize_dispatch
    # Initialize result
    r_state_change = pd.DataFrame(0.0, index=self.s.dt_realize_dispatch, columns=self.s.nodeState)
    # Create index mappings from dtttdt
    prev_solve_idx = self.s.dtttdt.droplevel(['period', 'time', 't_previous', 't_previous_within_timeset']).set_names(['period', 'time'])
    prev_period_idx = self.s.dtttdt.droplevel(['time', 't_previous_within_timeset', 'd_previous', 't_previous_within_solve']).set_names(['period', 'time'])
    prev_timeset_idx = self.s.dtttdt.droplevel(['time', 't_previous', 'd_previous', 't_previous_within_solve']).set_names(['period', 'time'])

    # Create exclude_idx directly from MultiIndex
    exclude_idx = self.s.period__time_first[
        self.s.period__time_first.get_level_values('period').isin(self.s.period_first_of_solve)
    ]
    entity_all_capacity = entity_all_capacity.droplevel(0, axis=1)
    
    for n in self.s.nodeState:
        if n not in v_state.columns:
            continue
        
        state_change = pd.Series(0.0, index=current_idx)
        
        v_current = pd.Series(v_state[n].reindex(current_idx).values, index=current_idx)
        v_prev_solve = pd.Series(v_state[n].reindex(prev_solve_idx).values, index=current_idx)
        v_prev_period = pd.Series(v_state[n].reindex(prev_period_idx).values, index=current_idx)
        v_prev_timeset = pd.Series(v_state[n].reindex(prev_timeset_idx).values, index=current_idx)
        
        # Case 1: bind_forward_only
        if (n, 'bind_forward_only') in self.s.node__storage_binding_method:
            mask = ~current_idx.isin(exclude_idx)
            state_change += ((v_current - v_prev_solve) * unitsize[n]).where(mask, 0)
        
        # Cases 2-4 remain the same...
        if ((n, 'bind_within_solve') in self.s.node__storage_binding_method and
            (n, 'fix_start_end') not in self.s.node__storage_start_end_method):
            state_change += (v_current - v_prev_solve) * unitsize[n]
        
        if ((n, 'bind_within_period') in self.s.node__storage_binding_method and
            (n, 'fix_start_end') not in self.s.node__storage_start_end_method):
            state_change += (v_current - v_prev_period) * unitsize[n]
        
        if ((n, 'bind_within_timeset') in self.s.node__storage_binding_method and
            (n, 'fix_start_end') not in self.s.node__storage_start_end_method):
            state_change += (v_current - v_prev_timeset) * unitsize[n]
        
        # Case 5
        if not self.p.nested_model.loc['solveFirst', 'value']:
            mask = current_idx.isin(exclude_idx)
            state_change += (v_current * unitsize[n] - self.p.roll_continue_state.loc[n, 'value']).where(mask, 0)
        
        # Case 6
        if ((n, 'bind_forward_only') in self.s.node__storage_binding_method and
            self.p.nested_model.loc['solveFirst', 'value'] and
            ((n, 'fix_start') in self.s.node__storage_start_end_method or
            (n, 'fix_start_end') in self.s.node__storage_start_end_method)):
            
            for idx_tuple in exclude_idx:
                if idx_tuple in current_idx:
                    state_change.loc[idx_tuple] += (
                        v_current.loc[idx_tuple] * unitsize[n] -
                        self.p.node.droplevel(0, axis=1).loc[('storage_state_start', n)] * entity_all_capacity.loc[idx_tuple[0], n]
                    )

        # Assign
        r_state_change[n] = state_change
    
    self.r.nodeState_change_dt = r_state_change

    # r_nodeState_change_d - sum over dt_realize_dispatch
    r_state_change_d = self.r.nodeState_change_dt[
        self.r.nodeState_change_dt.index.get_level_values('period').isin(self.s.d_realized_period)
    ].groupby(level='period').sum()
    self.r.nodeState_change_d = r_state_change_d
    
    # r_selfDischargeLoss_dt - element-wise multiplication
    r_self_discharge_dt = pd.DataFrame(index=dt_dispatch_idx, columns=self.s.nodeSelfDischarge)
    for n in self.s.nodeSelfDischarge:
        if n in self.v.state.columns and n in self.p.node_self_discharge_loss.columns:
            r_self_discharge_dt[n] = (
                self.v.state[n].reindex(dt_dispatch_idx) * 
                self.p.node_self_discharge_loss[n].reindex(dt_dispatch_idx) * 
                self.p.entity_unitsize['value'][n]
            )
    
    
    # r_selfDischargeLoss_d - multiply by step_duration then sum
    r_self_discharge_weighted = r_self_discharge_dt.copy()
    for idx in r_self_discharge_weighted.index:
        if idx in self.p.step_duration.index:
            r_self_discharge_weighted.loc[idx] *= self.p.step_duration.loc[idx]
    r_self_discharge_d = r_self_discharge_weighted[
        r_self_discharge_weighted.index.get_level_values('period').isin(self.s.d_realized_period)
    ].groupby(level='period').sum()
    self.r.selfDischargeLoss_d = r_self_discharge_d
    self.p.commodity_price = self.p.commodity_price.droplevel(0, axis=1)
    # r_cost_commodity_dt
    r_commodity_cost = {}
    for (c, n) in self.s.commodity_node:
        net_flow = pd.Series(0.0, index=r_flow_dt.index)
        # Flows into node (p, n, sink)
        for col in r_flow_dt.columns:
            if col[1] == n:
                net_flow += r_flow_dt[col]
        # Flows out of node (p, source, n) - negative
        for col in r_flow_dt.columns:
            if col[2] == n:
                net_flow -= r_flow_dt[col]
        # Multiply by step_duration and price
        cost = net_flow.copy()
        for idx in cost.index:
            if idx in self.p.step_duration.index and idx in self.p.commodity_price.index:
                cost.loc[idx] *= self.p.step_duration.loc[idx] * self.p.commodity_price.loc[idx, c]
        r_commodity_cost[(c, n)] = cost
    self.r.cost_commodity_dt = pd.DataFrame(r_commodity_cost)
    
    # r_process_commodity_d
    r_process_commodity = {}
    for _, row in self.s.process__commodity__node.iterrows():
        p, c, n = row['process'], row['commodity'], row['node']
        net_flow = pd.Series(0.0, index=r_flow_d.index)
        # Flows into node (p, n, sink)
        for col in r_flow_d.columns:
            if col[0] == p and col[1] == n:
                net_flow += r_flow_d[col]
        # Flows out of node (p, source, n) - negative
        for col in r_flow_d.columns:
            if col[0] == p and col[2] == n:
                net_flow -= r_flow_d[col]
        r_process_commodity[(p, c, n)] = net_flow
    self.r.process_commodity_d = pd.DataFrame(r_process_commodity)
    
    # r_process_emissions_co2_dt
    r_process_co2 = {}
    for _, row in self.s.process__commodity__node_co2.iterrows():
        p, c, n = row['process'], row['commodity'], row['node']
        net_flow = pd.Series(0.0, index=r_flow_dt.index)
        # Flows into node (p, n, sink)
        for col in r_flow_dt.columns:
            if col[0] == p and col[1] == n:
                net_flow += r_flow_dt[col]
        # Flows out of node (p, source, n) - negative
        for col in r_flow_dt.columns:
            if col[0] == p and col[2] == n:
                net_flow -= r_flow_dt[col]
        # Multiply by step_duration and co2_content
        emissions = net_flow.copy()
        for idx in emissions.index:
            if idx in self.p.step_duration.index:
                emissions.loc[idx] *= (self.p.step_duration.loc[idx] * 
                                    self.p.commodity_co2_content.loc[c])
        r_process_co2[(p, c, n)] = emissions
    self.r.process_emissions_co2_dt = pd.DataFrame(r_process_co2)
    
    # r_process_emissions_co2_d - sum and divide by complete_period_share_of_year
    r_process_co2_d = {}
    for col in self.r.process_emissions_co2_dt.columns:
        emissions_sum = self.r.process_emissions_co2_dt[col][
            self.r.process_emissions_co2_dt.index.get_level_values('period').isin(self.s.d_realized_period)
        ].groupby(level='period').sum()
        # Divide by complete_period_share_of_year
        for d in emissions_sum.index:
            if d in self.p.complete_period_share_of_year.index:
                emissions_sum.loc[d] /= self.p.complete_period_share_of_year.loc[d]
        r_process_co2_d[col] = emissions_sum
    self.r.process_emissions_co2_d = pd.DataFrame(r_process_co2_d)
    
    # r_emissions_co2_dt - sum over processes
    r_co2_dt = {}
    for (c, n) in self.s.commodity_node_co2:
        total_emissions = pd.Series(0.0, index=r_flow_dt.index)
        # Sum all processes for this (c, n)
        for col in self.r.process_emissions_co2_dt.columns:
            if col[1] == c and col[2] == n:  # (p, c, n)
                total_emissions += self.r.process_emissions_co2_dt[col]
        r_co2_dt[(c, n)] = total_emissions
    self.r.emissions_co2_dt = pd.DataFrame(r_co2_dt)
    
    # r_emissions_co2_d - sum and divide by complete_period_share_of_year
    r_co2_d = {}
    for col in self.r.emissions_co2_dt.columns:
        emissions_sum = self.r.emissions_co2_dt[col][
            self.r.emissions_co2_dt.index.get_level_values('period').isin(self.s.d_realized_period)
        ].groupby(level='period').sum()
        # Divide by complete_period_share_of_year
        for d in emissions_sum.index:
            if d in self.p.complete_period_share_of_year.index:
                emissions_sum.loc[d] /= self.p.complete_period_share_of_year.loc[d]
        r_co2_d[col] = emissions_sum
    self.r.emissions_co2_d = pd.DataFrame(r_co2_d)

    # r_cost_co2_dt
    r_co2_cost = {}
    for _, row in self.s.gcndt_co2_price.iterrows():
        g, c, n, d, t = row['group'], row['commodity'], row['node'], row['period'], row['time']
        if (c, n) in self.r.emissions_co2_dt.columns:
            r_co2_cost[(g, c, n, d, t)] = (
                self.r.emissions_co2_dt[(c, n)].loc[(d, t)] * 
                self.p.group_co2_price.loc[(d, t), g]
            )
    self.r.cost_co2_dt = pd.Series(r_co2_cost)
    
    # r_cost_process_other_operational_cost_dt
    r_process_cost = pd.DataFrame(0.0, index=self.r.process_source_sink_flow_dt.index, columns=self.s.process)
    for p in self.s.process:
        for col in self.r.process_source_sink_flow_dt.columns:
            if col[0] == p and col in self.p.process_source_sink_varCost.columns:
                r_process_cost[p] += (self.p.step_duration * 
                                    self.p.process_source_sink_varCost[col] * 
                                    self.r.process_source_sink_flow_dt[col])
    self.r.cost_process_other_operational_cost_dt = r_process_cost

    # r_process_startup_dt
    self.r.process_startup_dt = self.v.startup_linear.add(self.v.startup_integer, fill_value=0)
    
    # r_cost_startup_dt
    r_startup_cost = pd.DataFrame(0.0, index=self.r.process_startup_dt.index, columns=self.s.process_online)
    for p in self.s.process_online:
        if p in self.r.process_startup_dt.columns and p in self.p.process_startup_cost.columns:
            cost = self.r.process_startup_dt[p] * self.p.entity_unitsize[p]
            for d in self.p.process_startup_cost.index:
                if p in self.p.process_startup_cost.columns:
                    period_mask = cost.index.get_level_values('period') == d
                    cost.loc[period_mask] *= self.p.process_startup_cost.loc[d, p]
            r_startup_cost[p] = cost
    self.r.cost_startup_dt = r_startup_cost

    # r_costPenalty_nodeState_upDown_dt
    nodes = list(self.s.nodeBalance) + list(self.s.nodeBalancePeriod)
    node_updown_columns = pd.MultiIndex.from_product([nodes, self.s.upDown], names=['node', 'upDown'])
    r_penalty_state = pd.DataFrame(
        index=self.v.q_state_up.index if hasattr(self.v, 'q_state_up') and not self.v.q_state_up.empty else self.v.q_state_down.index,
        columns=node_updown_columns
    )
    for n in nodes:
        for ud in self.s.upDown:
            if ud == 'up' and n in self.v.q_state_up.columns:
                penalty = self.v.q_state_up[n] * self.p.node_penalty_up[n]
                for d in self.p.node_capacity_for_scaling.index:
                    if n in self.p.node_capacity_for_scaling.columns:
                        period_mask = penalty.index.get_level_values('period') == d
                        penalty.loc[period_mask] *= self.p.node_capacity_for_scaling.loc[d, n]
                r_penalty_state[(n, ud)] = penalty
            elif ud == 'down' and n in self.v.q_state_down.columns:
                penalty = self.v.q_state_down[n] * self.p.node_penalty_down[n]
                for d in self.p.node_capacity_for_scaling.index:
                    if n in self.p.node_capacity_for_scaling.columns:
                        period_mask = penalty.index.get_level_values('period') == d
                        penalty.loc[period_mask] *= self.p.node_capacity_for_scaling.loc[d, n]
                r_penalty_state[(n, ud)] = penalty
    self.r.costPenalty_nodeState_upDown_dt = r_penalty_state

    # r_penalty_nodeState_upDown_d
    if not self.r.costPenalty_nodeState_upDown_dt.empty:
        self.r.penalty_nodeState_upDown_d = self.r.costPenalty_nodeState_upDown_dt[
            self.r.costPenalty_nodeState_upDown_dt.index.get_level_values('period').isin(self.s.d_realized_period)
        ].groupby(level='period').sum()
    else:
        self.r.penalty_nodeState_upDown_d = None

    # r_costPenalty_inertia_dt
    r_penalty_inertia = pd.DataFrame(index=self.v.q_inertia.index, columns=self.s.groupInertia)
    for g in self.s.groupInertia:
        if g in self.v.q_inertia.columns:
            penalty = self.v.q_inertia[g] * self.p.step_duration
            for d in self.p.group_inertia_limit.index:
                if g in self.p.group_inertia_limit.columns:
                    period_mask = penalty.index.get_level_values('period') == d
                    penalty.loc[period_mask] *= (self.p.group_inertia_limit.loc[d, g] * 
                                                self.p.group_penalty_inertia.loc[d, g])
            r_penalty_inertia[g] = penalty
    self.r.costPenalty_inertia_dt = r_penalty_inertia

    # r_costPenalty_non_synchronous_dt
    r_penalty_nonsync = pd.DataFrame(index=self.v.q_non_synchronous.index, columns=self.s.groupNonSync)
    for g in self.s.groupNonSync:
        if g in self.v.q_non_synchronous.columns:
            penalty = self.v.q_non_synchronous[g] * self.p.step_duration
            for d in self.p.group_capacity_for_scaling.index:
                if g in self.p.group_capacity_for_scaling.columns:
                    period_mask = penalty.index.get_level_values('period') == d
                    penalty.loc[period_mask] *= (self.p.group_capacity_for_scaling.loc[d, g] * 
                                                self.p.group_penalty_non_synchronous.loc[d, g])
            r_penalty_nonsync[g] = penalty
    self.r.costPenalty_non_synchronous_dt = r_penalty_nonsync

    # r_costPenalty_capacity_margin_d
    r_penalty_cap_margin = pd.DataFrame(index=self.s.period_invest, columns=self.s.groupCapacityMargin)
    for g in self.s.groupCapacityMargin:
        if g in self.v.q_capacity_margin.columns:
            for d in self.s.period_invest:
                if d in self.v.q_capacity_margin.index:
                    r_penalty_cap_margin.loc[d, g] = (self.v.q_capacity_margin.loc[d, g] * 
                                                    self.p.group_capacity_for_scaling.loc[d, g] * 
                                                    self.p.group_penalty_capacity_margin.loc[d, g] * 
                                                    self.p.discount_factor_operations_yearly.loc[d])
    self.r.costPenalty_capacity_margin_d = r_penalty_cap_margin
    
    # r_costPenalty_reserve_upDown_dt
    r_penalty_reserve = pd.DataFrame(index=self.v.q_reserve.index, columns=self.v.q_reserve.columns)
    for col in self.v.q_reserve.columns:
        r, ud, ng = col
        penalty = (self.v.q_reserve[col] * self.p.step_duration * 
                self.p.reserve_upDown_group_penalty.loc[(r, ud, ng), 'value'] * 
                self.p.reserve_upDown_group_reservation[col])
        r_penalty_reserve[col] = penalty
    self.r.costPenalty_reserve_upDown_dt = r_penalty_reserve

    # r_cost_entity_invest_d
    r_invest_cost = pd.DataFrame(index=self.v.invest.index, columns=self.v.invest.columns)
    for e in self.v.invest.columns:
        for d in self.v.invest.index:
            if (e, d) in self.s.ed_invest:
                r_invest_cost.loc[d, e] = (self.v.invest.loc[d, e] * 
                                        self.p.entity_unitsize['value'][e] * 
                                        self.p.entity_annual_discounted.loc[d, e])
    self.r.cost_entity_invest_d = r_invest_cost
    
    # r_cost_entity_divest_d
    r_divest_cost = pd.DataFrame(index=self.v.divest.index, columns=self.v.divest.columns)
    for e in self.v.divest.columns:
        for d in self.v.divest.index:
            if (e, d) in self.s.ed_divest:
                r_divest_cost.loc[d, e] = (-self.v.divest.loc[d, e] * 
                                        self.p.entity_unitsize['value'][e] * 
                                        self.p.entity_annual_divest_discounted.loc[d, e])
    self.r.cost_entity_divest_d = r_divest_cost
    
    # r_cost_entity_existing_fixed
    r_existing_fixed = pd.DataFrame(index=self.s.period_in_use, columns=self.s.entity)
    for e in self.s.entity:
        for d in self.s.period_in_use:
            capacity = self.p.entity_all_existing.droplevel(0, axis=1).loc[d, e]
            if not self.s.edd_invest.empty:
                for row in self.s.edd_invest.itertuples(index=False):
                    e_inv, d_inv, d_use = row.entity, row.d_invest, row.d
                    if e_inv == e and d_use == d and d_inv != d:
                        capacity += self.v.invest.loc[d_inv, e] * self.p.entity_unitsize['value'][e]
            fixed_cost = 0
            if e in self.s.process:
                fixed_cost = self.p.process_fixed_cost.droplevel(0, axis=1).loc[d, e]
            elif e in self.s.node:
                fixed_cost = self.p.node_fixed_cost.droplevel(0, axis=1).loc[d, e]
            r_existing_fixed.loc[d, e] = capacity * fixed_cost * 1000 * self.p.discount_factor_operations_yearly.loc[d]
    self.r.cost_entity_existing_fixed = r_existing_fixed
    
    # Aggregate costs
    self.r.costOper_dt = (self.r.cost_commodity_dt.sum(axis=1) + 
                        self.r.cost_process_other_operational_cost_dt.sum(axis=1) + 
                        self.r.cost_startup_dt.sum(axis=1))
    for _, row in self.s.gcndt_co2_price.iterrows():
        key = (row['group'], row['commodity'], row['node'], row['period'], row['time'])
        if key in self.r.cost_co2_dt.index:
            self.r.costOper_dt.loc[(row['period'], row['time'])] += self.r.cost_co2_dt.loc[key]
    
    self.r.costPenalty_dt = (self.r.costPenalty_nodeState_upDown_dt.sum(axis=1) + 
                            self.r.costPenalty_inertia_dt.sum(axis=1) + 
                            self.r.costPenalty_non_synchronous_dt.sum(axis=1) + 
                            self.r.costPenalty_reserve_upDown_dt.sum(axis=1))
    
    self.r.costOper_and_penalty_dt = self.r.costOper_dt + self.r.costPenalty_dt
    
    # Period aggregations
    self.r.cost_process_other_operational_cost_d = (self.r.cost_process_other_operational_cost_dt[
        self.r.cost_process_other_operational_cost_dt.index.get_level_values('period').isin(self.s.d_realized_period)
    ].groupby(level='period').sum() if not self.r.cost_process_other_operational_cost_dt.empty
    else pd.Series(0.0, index=self.s.d_realized_period))
    
    self.r.cost_co2_d = (self.r.cost_co2_dt[
        self.r.cost_co2_dt.index.get_level_values(3).isin(self.s.d_realized_period)
    ].groupby(level=3).sum() if not self.r.cost_co2_dt.empty 
    else pd.Series(0.0, index=self.s.d_realized_period))

    self.r.cost_commodity_d = (self.r.cost_commodity_dt[
        self.r.cost_commodity_dt.index.get_level_values('period').isin(self.s.d_realized_period)
    ].groupby(level='period').sum() if not self.r.cost_commodity_dt.empty 
    else pd.DataFrame(0.0, index=self.s.d_realized_period, columns=self.r.cost_commodity_dt.columns if hasattr(self.r.cost_commodity_dt, 'columns') else []))

    self.r.cost_variable_d = (self.r.cost_process_other_operational_cost_d.sum(axis=1) 
    if not self.r.cost_process_other_operational_cost_d.empty 
    else pd.Series(0.0, index=self.s.d_realized_period))

    self.r.cost_startup_d = (self.r.cost_startup_dt[
        self.r.cost_startup_dt.index.get_level_values('period').isin(self.s.d_realized_period)
    ].groupby(level='period').sum() if not self.r.cost_startup_dt.empty 
    else pd.DataFrame(0.0, index=self.s.d_realized_period, columns=self.r.cost_startup_dt.columns if hasattr(self.r.cost_startup_dt, 'columns') else []))

    self.r.costPenalty_nodeState_upDown_d = (self.r.costPenalty_nodeState_upDown_dt[
        self.r.costPenalty_nodeState_upDown_dt.index.get_level_values('period').isin(self.s.d_realized_period)
    ].groupby(level='period').sum() if not self.r.costPenalty_nodeState_upDown_dt.empty 
    else pd.DataFrame(0.0, index=self.s.d_realized_period, columns=self.r.costPenalty_nodeState_upDown_dt.columns if hasattr(self.r.costPenalty_nodeState_upDown_dt, 'columns') else []))

    self.r.costPenalty_inertia_d = (self.r.costPenalty_inertia_dt[
        self.r.costPenalty_inertia_dt.index.get_level_values('period').isin(self.s.d_realized_period)
    ].groupby(level='period').sum() if not self.r.costPenalty_inertia_dt.empty 
    else pd.DataFrame(0.0, index=self.s.d_realized_period, columns=self.r.costPenalty_inertia_dt.columns if hasattr(self.r.costPenalty_inertia_dt, 'columns') else []))

    self.r.costPenalty_non_synchronous_d = (self.r.costPenalty_non_synchronous_dt[
        self.r.costPenalty_non_synchronous_dt.index.get_level_values('period').isin(self.s.d_realized_period)
    ].groupby(level='period').sum() if not self.r.costPenalty_non_synchronous_dt.empty 
    else pd.DataFrame(0.0, index=self.s.d_realized_period, columns=self.r.costPenalty_non_synchronous_dt.columns if hasattr(self.r.costPenalty_non_synchronous_dt, 'columns') else []))

    self.r.costPenalty_reserve_upDown_d = (self.r.costPenalty_reserve_upDown_dt[
        self.r.costPenalty_reserve_upDown_dt.index.get_level_values('period').isin(self.s.d_realized_period)
    ].groupby(level='period').sum() if not self.r.costPenalty_reserve_upDown_dt.empty 
    else pd.DataFrame(0.0, index=self.s.d_realized_period, columns=self.r.costPenalty_reserve_upDown_dt.columns if hasattr(self.r.costPenalty_reserve_upDown_dt, 'columns') else []))

    self.r.costOper_d = (self.r.costOper_dt.groupby(level='period').sum() 
    if not self.r.costOper_dt.empty 
    else pd.Series(0.0, index=self.s.period_in_use))

    self.r.costPenalty_d = (self.r.costPenalty_dt.groupby(level='period').sum() 
    if not self.r.costPenalty_dt.empty 
    else pd.Series(0.0, index=self.s.period_in_use))

    for d in self.s.period_invest:
        if d in self.r.costPenalty_d.index and not self.r.costPenalty_capacity_margin_d.empty:
            self.r.costPenalty_d.loc[d] += self.r.costPenalty_capacity_margin_d.loc[d].sum()

    self.r.costOper_and_penalty_d = self.r.costOper_d.add(self.r.costPenalty_d, fill_value=0)        

    # Investment/divestment aggregations by type
    self.r.costInvestUnit_d = self.r.cost_entity_invest_d[[e for e in self.s.process_unit if e in self.r.cost_entity_invest_d.columns]].sum(axis=1)
    self.r.costDivestUnit_d = self.r.cost_entity_divest_d[[e for e in self.s.process_unit if e in self.r.cost_entity_divest_d.columns]].sum(axis=1)
    self.r.costInvestConnection_d = self.r.cost_entity_invest_d[[e for e in self.s.process_connection if e in self.r.cost_entity_invest_d.columns]].sum(axis=1)
    self.r.costDivestConnection_d = self.r.cost_entity_divest_d[[e for e in self.s.process_connection if e in self.r.cost_entity_divest_d.columns]].sum(axis=1)
    self.r.costInvestState_d = self.r.cost_entity_invest_d[[e for e in self.s.nodeState if e in self.r.cost_entity_invest_d.columns]].sum(axis=1)
    self.r.costDivestState_d = self.r.cost_entity_divest_d[[e for e in self.s.nodeState if e in self.r.cost_entity_divest_d.columns]].sum(axis=1)
    
    self.r.costInvest_d = self.r.costInvestUnit_d + self.r.costInvestConnection_d + self.r.costInvestState_d
    self.r.costDivest_d = self.r.costDivestUnit_d + self.r.costDivestConnection_d + self.r.costDivestState_d
    self.r.costExistingFixed_d = self.r.cost_entity_existing_fixed.sum(axis=1)

    # pdNodeInflow
    r_node_inflow = pd.DataFrame(index=self.s.period_in_use, columns=self.s.node)
    for n in self.s.node:
        for d in self.s.period_in_use:
            inflow = 0
            if n in self.s.nodeBalance and (n, 'no_inflow') not in self.s.node__inflow_method:
                if n in self.p.node_inflow.columns:
                    period_mask = self.p.node_inflow.index.get_level_values('period') == d
                    inflow = self.p.node_inflow.loc[period_mask, n].sum()
            elif n in self.s.nodeBalancePeriod:
                if n in self.p.node_annual_flow.columns and d in self.p.node_annual_flow.index:
                    inflow = self.p.node_annual_flow.loc[d, n]
            r_node_inflow.loc[d, n] = inflow
    self.r.node_inflow = r_node_inflow
    
    # potentialVREgen_dt
    vre_potential_dt = {}
    for _, row in self.s.process_VRE.iterrows():
        p, n = row['process'], row['node']
        if (p, n) not in self.s.process_sink:
            continue
        # Find matching profile method with upper_limit
        profile_methods = self.s.process__source__sink__profile__profile_method[
            (self.s.process__source__sink__profile__profile_method['process'] == p) &
            (self.s.process__source__sink__profile__profile_method['sink'] == n) &
            (self.s.process__source__sink__profile__profile_method['method'] == 'upper_limit')
        ]
        if not profile_methods.empty:
            f = profile_methods['profile'].iloc[0]
            if (f in self.p.profile.columns and 
                p in self.r.entity_all_capacity.columns and
                p in self.p.process_availability.columns):
                potential = self.p.profile[f].reindex(dt_dispatch_idx) * self.p.process_availability[p].reindex(dt_dispatch_idx)
                # Multiply by capacity for each period
                for d in self.r.entity_all_capacity.index:
                    period_mask = potential.index.get_level_values('period') == d
                    potential.loc[period_mask] *= self.r.entity_all_capacity.loc[d, p]
                vre_potential_dt[(p, n)] = potential
    self.r.potentialVREgen_dt = pd.DataFrame(vre_potential_dt) if vre_potential_dt else pd.DataFrame()
    
    # potentialVREgen - sum over dt_realize_dispatch
    if not self.r.potentialVREgen_dt.empty:
        self.r.potentialVREgen = self.r.potentialVREgen_dt[
            self.r.potentialVREgen_dt.index.get_level_values('period').isin(self.s.d_realized_period)
        ].groupby(level='period').sum()
    else:
        self.r.potentialVREgen = pd.DataFrame(0.0, index=self.s.d_realized_period, columns=[])

    # r_group_output__group_aggregate_Unit_to_group__dt
    r_unit_to_group = {}
    for _, row in self.s.group_output__group_aggregate_Unit_to_group.iterrows():
        g, ga = row['group'], row['group_aggregate']
        flow_sum = pd.Series(0.0, index=dt_dispatch_idx)
        
        matches = self.s.group_output__group_aggregate__process__unit__to_node[
            (self.s.group_output__group_aggregate__process__unit__to_node['group'] == g) &
            (self.s.group_output__group_aggregate__process__unit__to_node['group_aggregate'] == ga)
        ]
        for _, match_row in matches.iterrows():
            u, source, sink = match_row['unit'], match_row['source'], match_row['sink']
            if (u, source, sink) in self.r.process_source_sink_flow_dt.columns:
                flow_sum += self.r.process_source_sink_flow_dt[(u, source, sink)].reindex(dt_dispatch_idx, fill_value=0)
        
        r_unit_to_group[(g, ga)] = flow_sum
    
    self.r.group_output__group_aggregate_Unit_to_group__dt = (pd.DataFrame(r_unit_to_group) 
        if r_unit_to_group else pd.DataFrame(0.0, index=dt_dispatch_idx, columns=[]))
    
    # r_group_output__group_aggregate_Unit_to_group__d
    self.r.group_output__group_aggregate_Unit_to_group__d = (
        self.r.group_output__group_aggregate_Unit_to_group__dt[
            self.r.group_output__group_aggregate_Unit_to_group__dt.index.get_level_values('period').isin(self.s.d_realized_period)
        ].groupby(level='period').sum() if not self.r.group_output__group_aggregate_Unit_to_group__dt.empty
        else pd.DataFrame(0.0, index=self.s.d_realized_period, columns=[]))
    
    # r_group_output__group_aggregate_Group_to_unit__dt
    r_group_to_unit = {}
    for _, row in self.s.group_output__group_aggregate_Group_to_unit.iterrows():
        g, ga = row['group'], row['group_aggregate']
        flow_sum = pd.Series(0.0, index=dt_dispatch_idx)
        
        matches = self.s.group_output__group_aggregate__process__node__to_unit[
            (self.s.group_output__group_aggregate__process__node__to_unit['group'] == g) &
            (self.s.group_output__group_aggregate__process__node__to_unit['group_aggregate'] == ga)
        ]
        for _, match_row in matches.iterrows():
            u, source, sink = match_row['unit'], match_row['source'], match_row['sink']
            if (u, source, sink) in self.r.process_source_sink_flow_dt.columns:
                flow_sum -= self.r.process_source_sink_flow_dt[(u, source, sink)].reindex(dt_dispatch_idx, fill_value=0)
        
        r_group_to_unit[(g, ga)] = flow_sum
    
    self.r.group_output__group_aggregate_Group_to_unit__dt = (pd.DataFrame(r_group_to_unit)
        if r_group_to_unit else pd.DataFrame(0.0, index=dt_dispatch_idx, columns=[]))
    
    # r_group_output__group_aggregate_Group_to_unit__d
    self.r.group_output__group_aggregate_Group_to_unit__d = (
        self.r.group_output__group_aggregate_Group_to_unit__dt[
            self.r.group_output__group_aggregate_Group_to_unit__dt.index.get_level_values('period').isin(self.s.d_realized_period)
        ].groupby(level='period').sum() if not self.r.group_output__group_aggregate_Group_to_unit__dt.empty
        else pd.DataFrame(0.0, index=self.s.d_realized_period, columns=[]))
    
    # r_group_output__group_aggregate_Connection__dt
    r_connection = {}
    for _, row in self.s.group_output__group_aggregate_Connection.iterrows():
        g, ga = row['group'], row['group_aggregate']
        flow_sum = pd.Series(0.0, index=dt_dispatch_idx)
        
        # Connection to node
        conn_to_node = self.s.group_output__group_aggregate__process__connection__to_node[
            (self.s.group_output__group_aggregate__process__connection__to_node['group'] == g) &
            (self.s.group_output__group_aggregate__process__connection__to_node['group_aggregate'] == ga)
        ]
        for _, match_row in conn_to_node.iterrows():
            c, sink = match_row['connection'], match_row['sink']
            if (c, c, sink) in self.r.process_source_sink_flow_dt.columns:
                flow_sum += self.r.process_source_sink_flow_dt[(c, c, sink)].reindex(dt_dispatch_idx, fill_value=0)
        
        # Node to connection
        node_to_conn = self.s.group_output__group_aggregate__process__node__to_connection[
            (self.s.group_output__group_aggregate__process__node__to_connection['group'] == g) &
            (self.s.group_output__group_aggregate__process__node__to_connection['group_aggregate'] == ga)
        ]
        for _, match_row in node_to_conn.iterrows():
            c, source = match_row['connection'], match_row['source']
            if (c, source, c) in self.r.process_source_sink_flow_dt.columns:
                flow_sum -= self.r.process_source_sink_flow_dt[(c, source, c)].reindex(dt_dispatch_idx, fill_value=0)
        
        r_connection[(g, ga)] = flow_sum
    
    self.r.group_output__group_aggregate_Connection__dt = (pd.DataFrame(r_connection)
        if r_connection else pd.DataFrame(0.0, index=dt_dispatch_idx, columns=[]))
    
    # r_group_output__group_aggregate_Connection__d
    self.r.group_output__group_aggregate_Connection__d = (
        self.r.group_output__group_aggregate_Connection__dt[
            self.r.group_output__group_aggregate_Connection__dt.index.get_level_values('period').isin(self.s.d_realized_period)
        ].groupby(level='period').sum() if not self.r.group_output__group_aggregate_Connection__dt.empty
        else pd.DataFrame(0.0, index=self.s.d_realized_period, columns=[]))
    
    # r_group_output_Internal_connection_losses__dt
    r_conn_losses = pd.DataFrame(0.0, index=dt_dispatch_idx, columns=self.s.groupOutputNodeFlows)
    for g in self.s.groupOutputNodeFlows:
        losses = pd.Series(0.0, index=dt_dispatch_idx)
        
        for col in self.r.process_source_sink_flow_dt.columns:
            c, source, sink = col
            if c in self.s.process_connection and (g, c) in self.s.group_output__process_fully_inside:
                if (c, source) in self.s.process_source:
                    losses += self.r.process_source_sink_flow_dt[col].reindex(dt_dispatch_idx, fill_value=0)
                if (c, sink) in self.s.process_sink:
                    losses -= self.r.process_source_sink_flow_dt[col].reindex(dt_dispatch_idx, fill_value=0)
                if (c, source) in self.s.process_sink:
                    losses += self.r.process_source_sink_flow_dt[col].reindex(dt_dispatch_idx, fill_value=0)
                if (c, sink) in self.s.process_source:
                    losses -= self.r.process_source_sink_flow_dt[col].reindex(dt_dispatch_idx, fill_value=0)
        
        r_conn_losses[g] = losses
    
    self.r.group_output_Internal_connection_losses__dt = r_conn_losses
    
    # r_group_output_Internal_connection_losses__d
    self.r.group_output_Internal_connection_losses__d = (
        self.r.group_output_Internal_connection_losses__dt[
            self.r.group_output_Internal_connection_losses__dt.index.get_level_values('period').isin(self.s.d_realized_period)
        ].groupby(level='period').sum() if not self.r.group_output_Internal_connection_losses__dt.empty
        else pd.DataFrame(0.0, index=self.s.d_realized_period, columns=self.s.groupOutputNodeFlows))
    
    # r_group_output_Internal_unit_losses__dt
    r_unit_losses = pd.DataFrame(0.0, index=dt_dispatch_idx, columns=self.s.groupOutputNodeFlows)
    for g in self.s.groupOutputNodeFlows:
        losses = pd.Series(0.0, index=dt_dispatch_idx)
        
        for col in self.r.process_source_sink_flow_dt.columns:
            u, source, sink = col
            if u in self.s.process_unit and (g, u) in self.s.group_output__process_fully_inside:
                if (g, source) in self.s.group_node:
                    losses += self.r.process_source_sink_flow_dt[col].reindex(dt_dispatch_idx, fill_value=0)
                if (g, sink) in self.s.group_node:
                    losses -= self.r.process_source_sink_flow_dt[col].reindex(dt_dispatch_idx, fill_value=0)
        
        r_unit_losses[g] = losses
    
    self.r.group_output_Internal_unit_losses__dt = r_unit_losses
    
    # r_group_output_Internal_unit_losses__d
    self.r.group_output_Internal_unit_losses__d = (
        self.r.group_output_Internal_unit_losses__dt[
            self.r.group_output_Internal_unit_losses__dt.index.get_level_values('period').isin(self.s.d_realized_period)
        ].groupby(level='period').sum() if not self.r.group_output_Internal_unit_losses__dt.empty
        else pd.DataFrame(0.0, index=self.s.d_realized_period, columns=self.s.groupOutputNodeFlows))
    
    # r_group_node_inflow__dt
    r_group_inflow = pd.DataFrame(0.0, index=dt_dispatch_idx, columns=self.s.groupOutputNodeFlows)
    for g in self.s.groupOutputNodeFlows:
        inflow = pd.Series(0.0, index=dt_dispatch_idx)
        
        for n in self.s.node:
            if (g, n) in self.s.group_node and (n, 'no_inflow') not in self.s.node__inflow_method:
                if n in self.p.node_inflow.columns:
                    inflow += self.p.node_inflow[n].reindex(dt_dispatch_idx, fill_value=0)
        
        r_group_inflow[g] = inflow
    
    self.r.group_node_inflow__dt = r_group_inflow
    
    # r_group_node_inflow__d
    self.r.group_node_inflow__d = (
        self.r.group_node_inflow__dt[
            self.r.group_node_inflow__dt.index.get_level_values('period').isin(self.s.d_realized_period)
        ].groupby(level='period').sum() if not self.r.group_node_inflow__dt.empty
        else pd.DataFrame(0.0, index=self.s.d_realized_period, columns=self.s.groupOutputNodeFlows))
    
    # r_group_node_state_losses__dt
    r_group_state_losses = pd.DataFrame(0.0, index=dt_dispatch_idx, columns=self.s.groupOutputNodeFlows)
    for g in self.s.groupOutputNodeFlows:
        losses = pd.Series(0.0, index=dt_dispatch_idx)
        
        for n in self.s.nodeSelfDischarge:
            if (g, n) in self.s.group_node and n in self.r.selfDischargeLoss_dt.columns:
                losses += self.r.selfDischargeLoss_dt[n].reindex(dt_dispatch_idx, fill_value=0)
        
        r_group_state_losses[g] = losses
    
    self.r.group_node_state_losses__dt = r_group_state_losses
    
    # r_group_node_state_losses__d
    self.r.group_node_state_losses__d = (
        self.r.group_node_state_losses__dt[
            self.r.group_node_state_losses__dt.index.get_level_values('period').isin(self.s.d_realized_period)
        ].groupby(level='period').sum() if not self.r.group_node_state_losses__dt.empty
        else pd.DataFrame(0.0, index=self.s.d_realized_period, columns=self.s.groupOutputNodeFlows))
    
    # r_group_node_up_penalties__dt
    nodes = list(self.s.nodeBalance) + list(self.s.nodeBalancePeriod)
    r_group_up_penalties = pd.DataFrame(0.0, index=dt_dispatch_idx, columns=self.s.groupOutputNodeFlows)
    for g in self.s.groupOutputNodeFlows:
        penalties = pd.Series(0.0, index=dt_dispatch_idx)
        
        for n in nodes:
            if (g, n) in self.s.group_node and n in self.v.q_state_up.columns:
                penalty = self.v.q_state_up[n] * self.p.step_duration
                for d in self.p.node_capacity_for_scaling.index:
                    if n in self.p.node_capacity_for_scaling.columns:
                        period_mask = penalty.index.get_level_values('period') == d
                        penalty.loc[period_mask] *= self.p.node_capacity_for_scaling.loc[d, n]
                penalties += penalty.reindex(dt_dispatch_idx, fill_value=0)
        
        r_group_up_penalties[g] = penalties
    
    self.r.group_node_up_penalties__dt = r_group_up_penalties
    
    # r_group_node_up_penalties__d
    self.r.group_node_up_penalties__d = (
        self.r.group_node_up_penalties__dt[
            self.r.group_node_up_penalties__dt.index.get_level_values('period').isin(self.s.d_realized_period)
        ].groupby(level='period').sum() if not self.r.group_node_up_penalties__dt.empty
        else pd.DataFrame(0.0, index=self.s.d_realized_period, columns=self.s.groupOutputNodeFlows))
    
    # r_group_node_down_penalties__dt
    r_group_down_penalties = pd.DataFrame(0.0, index=dt_dispatch_idx, columns=self.s.groupOutputNodeFlows)
    for g in self.s.groupOutputNodeFlows:
        penalties = pd.Series(0.0, index=dt_dispatch_idx)
        
        for n in nodes:
            if (g, n) in self.s.group_node and n in self.v.q_state_down.columns:
                penalty = -self.v.q_state_down[n] * self.p.step_duration
                for d in self.p.node_capacity_for_scaling.index:
                    if n in self.p.node_capacity_for_scaling.columns:
                        period_mask = penalty.index.get_level_values('period') == d
                        penalty.loc[period_mask] *= self.p.node_capacity_for_scaling.loc[d, n]
                penalties += penalty.reindex(dt_dispatch_idx, fill_value=0)
        
        r_group_down_penalties[g] = penalties
    self.r.group_node_down_penalties__dt = r_group_down_penalties
    
    # r_group_node_down_penalties__d
    self.r.group_node_down_penalties__d = (
        self.r.group_node_down_penalties__dt[
            self.r.group_node_down_penalties__dt.index.get_level_values('period').isin(self.s.d_realized_period)
        ].groupby(level='period').sum() if not self.r.group_node_down_penalties__dt.empty
        else pd.DataFrame(0.0, index=self.s.d_realized_period, columns=self.s.groupOutputNodeFlows))
    
    # r_storage_usage_dt
    dt_fix_idx = self.s.dt_fix_storage_timesteps
    r_storage_usage = {}
    
    for n in self.s.node:
        if (n, 'fix_usage') in self.s.node__storage_nested_fix_method:
            usage = pd.Series(0.0, index=dt_fix_idx)
            
            for col in self.r.process_source_sink_flow_dt.columns:
                p, source, sink = col
                if source == n:
                    usage += (self.r.process_source_sink_flow_dt[col].reindex(dt_fix_idx, fill_value=0) * 
                            self.p.step_duration.reindex(dt_fix_idx, fill_value=0))
                if sink == n:
                    usage -= (self.r.process_source_sink_flow_dt[col].reindex(dt_fix_idx, fill_value=0) * 
                            self.p.step_duration.reindex(dt_fix_idx, fill_value=0))
            
            r_storage_usage[n] = usage
    
    self.r.storage_usage_dt = (pd.DataFrame(r_storage_usage) 
        if r_storage_usage else pd.DataFrame(0.0, index=dt_fix_idx, columns=[]))

def drop_levels(self):
    self.v.flow = self.v.flow.droplevel('solve')
    self.v.ramp = self.v.ramp.droplevel('solve')
    self.v.reserve = self.v.reserve.droplevel('solve')
    self.v.state = self.v.state.droplevel('solve')
    self.v.online_linear = self.v.online_linear.droplevel('solve')
    self.v.startup_linear = self.v.startup_linear.droplevel('solve')
    self.v.shutdown_linear = self.v.shutdown_linear.droplevel('solve')
    self.v.online_integer = self.v.online_integer.droplevel('solve')
    self.v.startup_integer = self.v.startup_integer.droplevel('solve')
    self.v.shutdown_integer = self.v.shutdown_integer.droplevel('solve')
    self.v.q_state_up = self.v.q_state_up.droplevel('solve')
    self.v.q_state_down = self.v.q_state_down.droplevel('solve')
    self.v.q_reserve = self.v.q_reserve.droplevel('solve')
    self.v.q_inertia = self.v.q_inertia.droplevel('solve')
    self.v.q_non_synchronous = self.v.q_non_synchronous.droplevel('solve')
    self.v.q_state_up_group = self.v.q_state_up_group.droplevel('solve')
    self.v.q_capacity_margin = self.v.q_capacity_margin.droplevel('solve')
    self.v.invest = self.v.invest.droplevel('solve')
    self.v.divest = self.v.divest.droplevel('solve')

    self.p.step_duration = self.p.step_duration.droplevel('solve')
    self.p.flow_min = self.p.flow_min.droplevel('solve')
    self.p.flow_max = self.p.flow_max.droplevel('solve')
    self.p.process_availability = self.p.process_availability.droplevel('solve')
    self.p.process_source_sink_varCost = self.p.process_source_sink_varCost.droplevel('solve')
    self.p.process_slope = self.p.process_slope.droplevel('solve')
    self.p.process_section = self.p.process_section.droplevel('solve')
    self.p.node_self_discharge_loss = self.p.node_self_discharge_loss.droplevel('solve')
    self.p.node_penalty_up = self.p.node_penalty_up.droplevel('solve')
    self.p.node_penalty_down = self.p.node_penalty_down.droplevel('solve')
    self.p.node_inflow = self.p.node_inflow.droplevel('solve')
    self.p.commodity_price = self.p.commodity_price.droplevel('solve')
    self.p.group_co2_price = self.p.group_co2_price.droplevel('solve')
    self.p.reserve_upDown_group_reservation = self.p.reserve_upDown_group_reservation.droplevel('solve')
    self.p.profile = self.p.profile.droplevel('solve')
    self.p.years_d = self.p.years_d.droplevel('solve')
    self.p.entity_max_units = self.p.entity_max_units.droplevel('solve')
    self.p.entity_all_existing = self.p.entity_all_existing.droplevel('solve')
    self.p.process_startup_cost = self.p.process_startup_cost.droplevel('solve')
    self.p.process_fixed_cost = self.p.process_fixed_cost.droplevel('solve')
    self.p.node_fixed_cost = self.p.node_fixed_cost.droplevel('solve')
    self.p.node_annual_flow = self.p.node_annual_flow.droplevel('solve')
    self.p.group_penalty_inertia = self.p.group_penalty_inertia.droplevel('solve')
    self.p.group_penalty_non_synchronous = self.p.group_penalty_non_synchronous.droplevel('solve')
    self.p.group_penalty_capacity_margin = self.p.group_penalty_capacity_margin.droplevel('solve')
    self.p.group_inertia_limit = self.p.group_inertia_limit.droplevel('solve')
    self.p.group_capacity_margin = self.p.group_capacity_margin.droplevel('solve')
    self.p.entity_annual_discounted = self.p.entity_annual_discounted.droplevel('solve')
    self.p.entity_annual_divest_discounted = self.p.entity_annual_divest_discounted.droplevel('solve')
    self.p.discount_factor_operations_yearly = self.p.discount_factor_operations_yearly.droplevel('solve')
    self.p.discount_factor_investment_yearly = self.p.discount_factor_investment_yearly.droplevel('solve')
    self.p.node_capacity_for_scaling = self.p.node_capacity_for_scaling.droplevel('solve')
    self.p.group_capacity_for_scaling = self.p.group_capacity_for_scaling.droplevel('solve')
    self.p.complete_period_share_of_year = self.p.complete_period_share_of_year.droplevel('solve')

    self.s.period = self.s.period.droplevel('solve')
    self.s.period__time_first = self.s.period__time_first.droplevel('solve')
    self.s.period_first_of_solve = self.s.period_first_of_solve.droplevel('solve')
    self.s.period_in_use = self.s.period_in_use.droplevel('solve')
    self.s.period_invest = self.s.period_invest.droplevel('solve')
    self.s.d_realize_dispatch_or_invest = self.s.d_realize_dispatch_or_invest.droplevel('solve')
    self.s.d_realized_period = self.s.d_realized_period.droplevel('solve')
    self.s.dt = self.s.dt.droplevel('solve')
    self.s.dt_fix_storage_timesteps = self.s.dt_fix_storage_timesteps.droplevel('solve')
    self.s.dt_realize_dispatch = self.s.dt_realize_dispatch.droplevel('solve')
    self.s.dtt = self.s.dtt.droplevel('solve')
    self.s.dtttdt = self.s.dtttdt.droplevel('solve')
    self.s.ed_invest = self.s.ed_invest.droplevel('solve')
    self.s.edd_invest = self.s.edd_invest.droplevel('solve')
    self.s.ed_divest = self.s.ed_divest.droplevel('solve')