import pandas as pd


def compute_group_flows(par, s, v, r) -> None:
    """Compute all group-level flow aggregations (connection and unit flows, inflow, losses).

    v_flow in FlexTool is MW (MWh/h — length-of-time independent).  Period
    aggregations here multiply by par.step_duration before summing so that
    `_d` results are MWh (then annualized by dividing by
    complete_period_share_of_year).  par.node_inflow is already MWh/step
    (see mod comment at line 2319), so inflow-derived frames sum as-is.
    """
    step_duration = par.step_duration

    # --- Group connection flows (uses r.from_conn and r.to_conn from compute_connection_flows) ---

    # Helper: create an empty DataFrame with the expected MultiIndex column structure
    _empty_gp = pd.DataFrame(columns=pd.MultiIndex(levels=[[], []], codes=[[], []], names=['group', 'process']))
    _empty_gga = pd.DataFrame(columns=pd.MultiIndex(levels=[[], [], []], codes=[[], [], []], names=['group', 'group_aggregate', 'process']))

    from_conn_set = s.nodeGroupDispatch__process__connection__to_node_Not_in_aggregate.droplevel('connection')
    if not from_conn_set.empty and not r.from_conn.empty:
        group_sets = r.from_conn.columns.join(from_conn_set, how='inner')
        from_conn_selected = r.from_conn[group_sets.droplevel('group')]
        from_conn_selected.columns = group_sets
        r.nodeGroupDispatch__from_connection_not_in_aggregate__dt = from_conn_selected
    else:
        r.nodeGroupDispatch__from_connection_not_in_aggregate__dt = _empty_gp.copy()

    to_conn_set = s.nodeGroupDispatch__process__node__to_connection_Not_in_aggregate.droplevel('connection')
    if not to_conn_set.empty and not r.to_conn.empty:
        group_sets = r.to_conn.columns.join(to_conn_set, how='inner')
        to_conn_selected = r.to_conn[group_sets.droplevel('group')]
        to_conn_selected.columns = group_sets
        r.nodeGroupDispatch__to_connection_not_in_aggregate__dt = to_conn_selected
    else:
        r.nodeGroupDispatch__to_connection_not_in_aggregate__dt = _empty_gp.copy()

    r.nodeGroupDispatch__from_connection_not_in_aggregate__d = r.nodeGroupDispatch__from_connection_not_in_aggregate__dt.mul(step_duration, axis=0).groupby('period').sum().div(par.complete_period_share_of_year, axis=0) if not r.nodeGroupDispatch__from_connection_not_in_aggregate__dt.empty else _empty_gp.copy()
    r.nodeGroupDispatch__to_connection_not_in_aggregate__d = r.nodeGroupDispatch__to_connection_not_in_aggregate__dt.mul(step_duration, axis=0).groupby('period').sum().div(par.complete_period_share_of_year, axis=0) if not r.nodeGroupDispatch__to_connection_not_in_aggregate__dt.empty else _empty_gp.copy()

    from_conn_agg_set = s.nodeGroupDispatch__processGroup__process__connection__to_node.droplevel('connection')
    if not from_conn_agg_set.empty and not r.from_conn.empty:
        group_agg_sets = r.from_conn.columns.join(from_conn_agg_set, how='inner')
        from_conn_agg_selected = r.from_conn[group_agg_sets.droplevel(['group', 'group_aggregate'])]
        from_conn_agg_selected.columns = group_agg_sets
        r.nodeGroupDispatch__from_connection_aggregate__dt = from_conn_agg_selected.T.groupby(level=['group', 'group_aggregate']).sum().T
    else:
        r.nodeGroupDispatch__from_connection_aggregate__dt = _empty_gga.copy()

    to_conn_agg_set = s.nodeGroupDispatch__processGroup__process__node__to_connection.droplevel('connection')
    if not to_conn_agg_set.empty and not r.to_conn.empty:
        group_agg_sets = to_conn_agg_set.join(r.to_conn.columns)
        to_conn_agg_selected = r.to_conn[group_agg_sets.droplevel(['group', 'group_aggregate'])]
        to_conn_agg_selected.columns = group_agg_sets
        r.nodeGroupDispatch__to_connection_aggregate__dt = to_conn_agg_selected.T.groupby(level=['group', 'group_aggregate']).sum().T
    else:
        r.nodeGroupDispatch__to_connection_aggregate__dt = _empty_gga.copy()

    r.nodeGroupDispatch__from_connection_aggregate__d = r.nodeGroupDispatch__from_connection_aggregate__dt.mul(step_duration, axis=0).groupby('period').sum().div(par.complete_period_share_of_year, axis=0) if not r.nodeGroupDispatch__from_connection_aggregate__dt.empty else _empty_gga.copy()
    r.nodeGroupDispatch__to_connection_aggregate__d = r.nodeGroupDispatch__to_connection_aggregate__dt.mul(step_duration, axis=0).groupby('period').sum().div(par.complete_period_share_of_year, axis=0) if not r.nodeGroupDispatch__to_connection_aggregate__dt.empty else _empty_gga.copy()

    losses_set = s.nodeGroupDispatch__process_fully_inside
    if not losses_set.empty and not r.connection_losses_dt.empty:
        group_losses_sets = r.connection_losses_dt.columns.join(losses_set, how='inner')
        losses_selected = r.connection_losses_dt[group_losses_sets.droplevel('group')]
        losses_selected.columns = group_losses_sets
        r.nodeGroupDispatch_Internal_connection_losses__dt = losses_selected
        r.nodeGroupDispatch_Internal_connection_losses__d = r.nodeGroupDispatch_Internal_connection_losses__dt.mul(step_duration, axis=0).groupby('period').sum().div(par.complete_period_share_of_year, axis=0)
    else:
        r.nodeGroupDispatch_Internal_connection_losses__dt = pd.DataFrame()
        r.nodeGroupDispatch_Internal_connection_losses__d = pd.DataFrame()

    # --- Group unit flows ---

    unit_flows = r.flow_dt[r.flow_dt.columns[r.flow_dt.columns.get_level_values('process').isin(s.process_unit)]]

    unit_to_node = unit_flows[unit_flows.columns[unit_flows.columns.droplevel('source').isin(s.process_sink)]]
    node_to_unit = unit_flows[unit_flows.columns[unit_flows.columns.droplevel('sink').isin(s.process_source)]]

    unit_to_node.columns = unit_to_node.columns.droplevel('source')
    node_to_unit.columns = node_to_unit.columns.droplevel('sink')

    unit_to_node.columns.names = ['process', 'node']
    node_to_unit.columns.names = ['process', 'node']

    unit_to_node_set = s.nodeGroupDispatch__process__unit__to_node_Not_in_aggregate.droplevel('unit')
    group_sets = unit_to_node.columns.join(unit_to_node_set, how='inner')
    unit_to_node_selected = unit_to_node[group_sets.droplevel('group')]
    unit_to_node_selected.columns = group_sets
    r.nodeGroupDispatch__unit_to_node_not_in_aggregate__dt = unit_to_node_selected

    node_to_unit_set = s.nodeGroupDispatch__process__node__to_unit_Not_in_aggregate.droplevel('unit')
    group_sets = node_to_unit.columns.join(node_to_unit_set, how='inner')
    node_to_unit_selected = node_to_unit[group_sets.droplevel('group')]
    node_to_unit_selected.columns = group_sets
    r.nodeGroupDispatch__node_to_unit_not_in_aggregate__dt = node_to_unit_selected

    r.nodeGroupDispatch__unit_to_node_not_in_aggregate__d = r.nodeGroupDispatch__unit_to_node_not_in_aggregate__dt.mul(step_duration, axis=0).groupby('period').sum().div(par.complete_period_share_of_year, axis=0)
    r.nodeGroupDispatch__node_to_unit_not_in_aggregate__d = r.nodeGroupDispatch__node_to_unit_not_in_aggregate__dt.mul(step_duration, axis=0).groupby('period').sum().div(par.complete_period_share_of_year, axis=0)

    unit_to_group_set = s.nodeGroupDispatch__processGroup__process__unit__to_node.droplevel('unit')
    group_agg_sets = unit_to_node.columns.join(unit_to_group_set, how='inner')
    unit_to_group_selected = unit_to_node[group_agg_sets.droplevel(['group', 'group_aggregate'])]
    unit_to_group_selected.columns = group_agg_sets
    negatives = (par.entity_unitsize[unit_to_group_selected.columns.get_level_values('process')] < 0).values
    unit_to_group_selected_negative = unit_to_group_selected[unit_to_group_selected.columns[negatives]]
    unit_to_group_selected_positive = unit_to_group_selected[unit_to_group_selected.columns[~negatives]]
    r.nodeGroupDispatch__group_aggregate_Unit_to_group_positive__dt = unit_to_group_selected_positive.T.groupby(level=['group', 'group_aggregate']).sum().T
    r.nodeGroupDispatch__group_aggregate_Unit_to_group_negative__dt = unit_to_group_selected_negative.T.groupby(level=['group', 'group_aggregate']).sum().T

    group_to_unit_set = s.nodeGroupDispatch__processGroup__process__node__to_unit.droplevel('unit')
    group_agg_sets = node_to_unit.columns.join(group_to_unit_set, how='inner')
    group_to_unit_selected = node_to_unit[group_agg_sets.droplevel(['group', 'group_aggregate'])]
    group_to_unit_selected.columns = group_agg_sets
    negatives = (par.entity_unitsize[group_to_unit_selected.columns.get_level_values('process')] < 0).values
    group_to_unit_selected_negative = group_to_unit_selected[group_to_unit_selected.columns[negatives]]
    group_to_unit_selected_positive = group_to_unit_selected[group_to_unit_selected.columns[~negatives]]
    r.nodeGroupDispatch__group_aggregate_Group_to_unit_positive__dt = group_to_unit_selected_positive.T.groupby(level=['group', 'group_aggregate']).sum().T
    r.nodeGroupDispatch__group_aggregate_Group_to_unit_negative__dt = group_to_unit_selected_negative.T.groupby(level=['group', 'group_aggregate']).sum().T

    r.nodeGroupDispatch__group_aggregate_Unit_to_group__dt = r.nodeGroupDispatch__group_aggregate_Unit_to_group_positive__dt.sub(
        r.nodeGroupDispatch__group_aggregate_Group_to_unit_negative__dt, fill_value=0.0)
    r.nodeGroupDispatch__group_aggregate_Group_to_unit__dt = r.nodeGroupDispatch__group_aggregate_Group_to_unit_positive__dt.sub(
        r.nodeGroupDispatch__group_aggregate_Unit_to_group_negative__dt, fill_value=0.0)

    r.nodeGroupDispatch__group_aggregate_Group_to_unit__d = r.nodeGroupDispatch__group_aggregate_Group_to_unit__dt.mul(step_duration, axis=0).groupby('period').sum().div(par.complete_period_share_of_year, axis=0)
    r.nodeGroupDispatch__group_aggregate_Unit_to_group__d = r.nodeGroupDispatch__group_aggregate_Unit_to_group__dt.mul(step_duration, axis=0).groupby('period').sum().div(par.complete_period_share_of_year, axis=0)

    losses_set = s.nodeGroupDispatch__process_fully_inside
    group_losses_sets_node_to_unit = losses_set.join(node_to_unit.columns, how='inner')
    group_losses_sets_unit_to_node = losses_set.join(unit_to_node.columns, how='inner')

    node_to_unit_filtered = node_to_unit[group_losses_sets_node_to_unit.droplevel('group')]
    unit_to_node_filtered = unit_to_node[group_losses_sets_unit_to_node.droplevel('group')]
    node_to_unit_filtered.columns = group_losses_sets_node_to_unit
    unit_to_node_filtered.columns = group_losses_sets_unit_to_node

    unit_losses_dt = node_to_unit_filtered.sub(unit_to_node_filtered, axis=1, fill_value=0.0)
    r.nodeGroupDispatch_Internal_unit_losses__dt = unit_losses_dt
    r.nodeGroupDispatch_Internal_unit_losses__d = r.nodeGroupDispatch_Internal_unit_losses__dt.mul(step_duration, axis=0).groupby('period').sum().div(par.complete_period_share_of_year, axis=0)

    # --- Node inflow by group ---
    dt_dispatch_idx = s.dt_realize_dispatch
    r_group_inflow = pd.DataFrame(0.0, index=dt_dispatch_idx, columns=s.nodeGroupDispatch, dtype=float)
    for g in s.nodeGroupDispatch:
        inflow = pd.Series(0.0, index=dt_dispatch_idx)
        for n in s.node:
            if (g, n) in s.group_node and (n, 'no_inflow') not in s.node__inflow_method:
                if n in par.node_inflow.columns:
                    inflow = par.node_inflow[n].squeeze().add(inflow, axis=0)
        r_group_inflow[g] = inflow
    r.group_node_inflow_dt = r_group_inflow

    r.group_node_inflow_d = (
        r.group_node_inflow_dt[
            r.group_node_inflow_dt.index.get_level_values('period').isin(s.d_realized_period)
        ].groupby(level='period').sum() if not r.group_node_inflow_dt.empty
        else pd.DataFrame(0.0, index=s.d_realized_period, columns=s.nodeGroupDispatch))

    # --- Storage state losses by group ---
    r_group_state_losses = pd.DataFrame(0.0, index=dt_dispatch_idx, columns=s.nodeGroupDispatch, dtype=float)
    for g in s.nodeGroupDispatch:
        losses = pd.Series(0.0, index=dt_dispatch_idx)
        for n in s.node_self_discharge:
            if (g, n) in s.group_node and n in r.self_discharge_loss_dt.columns:
                losses += r.self_discharge_loss_dt[n].reindex(dt_dispatch_idx, fill_value=0)
        r_group_state_losses[g] = losses
    r.group_node_state_losses__dt = r_group_state_losses

    r.group_node_state_losses__d = (
        r.group_node_state_losses__dt[
            r.group_node_state_losses__dt.index.get_level_values('period').isin(s.d_realized_period)
        ].groupby(level='period').sum() if not r.group_node_state_losses__dt.empty
        else pd.DataFrame(0.0, index=s.d_realized_period, columns=s.nodeGroupDispatch))

    # --- Node slack by group ---
    r.group_node_up_slack__dt = pd.DataFrame(index=s.dt_realize_dispatch)
    r.group_node_down_slack__dt = pd.DataFrame(index=s.dt_realize_dispatch)
    for g in s.nodeGroupDispatch:
        g_node = s.group_node[s.group_node.get_level_values('node').isin(s.node_balance) & s.group_node.get_level_values('group').isin([g])].get_level_values('node')
        r.group_node_up_slack__dt[g] = r.upward_node_slack_dt[g_node].sum(axis=1)
        r.group_node_down_slack__dt[g] = r.downward_node_slack_dt[g_node].sum(axis=1)
