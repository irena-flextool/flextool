import pandas as pd


def compute_group_flows(par, s, v, r) -> None:
    """Compute all group-level flow aggregations (connection and unit flows, inflow, losses)."""

    # --- Group connection flows (uses r.from_conn and r.to_conn from compute_connection_flows) ---

    from_conn_set = s.outputNodeGroup__process__connection__to_node_Not_in_aggregate.droplevel('connection')
    group_sets = r.from_conn.columns.join(from_conn_set, how='inner')
    from_conn_selected = r.from_conn[group_sets.droplevel('group')]
    from_conn_selected.columns = group_sets
    r.group_output__from_connection_not_in_aggregate__dt = from_conn_selected

    to_conn_set = s.outputNodeGroup__process__node__to_connection_Not_in_aggregate.droplevel('connection')
    group_sets = r.to_conn.columns.join(to_conn_set, how='inner')
    to_conn_selected = r.to_conn[group_sets.droplevel('group')]
    to_conn_selected.columns = group_sets
    r.group_output__to_connection_not_in_aggregate__dt = to_conn_selected

    r.group_output__from_connection_not_in_aggregate__d = r.group_output__from_connection_not_in_aggregate__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)
    r.group_output__to_connection_not_in_aggregate__d = r.group_output__to_connection_not_in_aggregate__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)

    from_conn_agg_set = s.outputNodeGroup__processGroup__process__connection__to_node.droplevel('connection')
    group_agg_sets = r.from_conn.columns.join(from_conn_agg_set, how='inner')
    from_conn_agg_selected = r.from_conn[group_agg_sets.droplevel(['group', 'group_aggregate'])]
    from_conn_agg_selected.columns = group_agg_sets
    r.group_output__from_connection_aggregate__dt = from_conn_agg_selected.T.groupby(level=['group', 'group_aggregate']).sum().T

    to_conn_agg_set = s.outputNodeGroup__processGroup__process__node__to_connection.droplevel('connection')
    group_agg_sets = to_conn_agg_set.join(r.to_conn.columns)
    to_conn_agg_selected = r.to_conn[group_agg_sets.droplevel(['group', 'group_aggregate'])]
    to_conn_agg_selected.columns = group_agg_sets
    r.group_output__to_connection_aggregate__dt = to_conn_agg_selected.T.groupby(level=['group', 'group_aggregate']).sum().T

    r.group_output__from_connection_aggregate__d = r.group_output__from_connection_aggregate__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)
    r.group_output__to_connection_aggregate__d = r.group_output__to_connection_aggregate__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)

    losses_set = s.outputNodeGroup__process_fully_inside
    group_losses_sets = r.connection_losses_dt.columns.join(losses_set, how='inner')
    losses_selected = r.connection_losses_dt[group_losses_sets.droplevel('group')]
    losses_selected.columns = group_losses_sets
    r.group_output_Internal_connection_losses__dt = losses_selected
    r.group_output_Internal_connection_losses__d = r.group_output_Internal_connection_losses__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)

    # --- Group unit flows ---

    unit_flows = r.flow_dt[r.flow_dt.columns[r.flow_dt.columns.get_level_values('process').isin(s.process_unit)]]

    unit_to_node = unit_flows[unit_flows.columns[unit_flows.columns.droplevel('source').isin(s.process_sink)]]
    node_to_unit = unit_flows[unit_flows.columns[unit_flows.columns.droplevel('sink').isin(s.process_source)]]

    unit_to_node.columns = unit_to_node.columns.droplevel('source')
    node_to_unit.columns = node_to_unit.columns.droplevel('sink')

    unit_to_node.columns.names = ['process', 'node']
    node_to_unit.columns.names = ['process', 'node']

    unit_to_node_set = s.outputNodeGroup__process__unit__to_node_Not_in_aggregate.droplevel('unit')
    group_sets = unit_to_node.columns.join(unit_to_node_set, how='inner')
    unit_to_node_selected = unit_to_node[group_sets.droplevel('group')]
    unit_to_node_selected.columns = group_sets
    r.group_output__unit_to_node_not_in_aggregate__dt = unit_to_node_selected

    node_to_unit_set = s.outputNodeGroup__process__node__to_unit_Not_in_aggregate.droplevel('unit')
    group_sets = node_to_unit.columns.join(node_to_unit_set, how='inner')
    node_to_unit_selected = node_to_unit[group_sets.droplevel('group')]
    node_to_unit_selected.columns = group_sets
    r.group_output__node_to_unit_not_in_aggregate__dt = node_to_unit_selected

    r.group_output__unit_to_node_not_in_aggregate__d = r.group_output__unit_to_node_not_in_aggregate__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)
    r.group_output__node_to_unit_not_in_aggregate__d = r.group_output__node_to_unit_not_in_aggregate__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)

    unit_to_group_set = s.outputNodeGroup__processGroup__process__unit__to_node.droplevel('unit')
    group_agg_sets = unit_to_node.columns.join(unit_to_group_set, how='inner')
    unit_to_group_selected = unit_to_node[group_agg_sets.droplevel(['group', 'group_aggregate'])]
    unit_to_group_selected.columns = group_agg_sets
    negatives = (par.entity_unitsize[unit_to_group_selected.columns.get_level_values('process')] < 0).values
    unit_to_group_selected_negative = unit_to_group_selected[unit_to_group_selected.columns[negatives]]
    unit_to_group_selected_positive = unit_to_group_selected[unit_to_group_selected.columns[~negatives]]
    r.group_output__group_aggregate_Unit_to_group_positive__dt = unit_to_group_selected_positive.T.groupby(level=['group', 'group_aggregate']).sum().T
    r.group_output__group_aggregate_Unit_to_group_negative__dt = unit_to_group_selected_negative.T.groupby(level=['group', 'group_aggregate']).sum().T

    group_to_unit_set = s.outputNodeGroup__processGroup__process__node__to_unit.droplevel('unit')
    group_agg_sets = node_to_unit.columns.join(group_to_unit_set, how='inner')
    group_to_unit_selected = node_to_unit[group_agg_sets.droplevel(['group', 'group_aggregate'])]
    group_to_unit_selected.columns = group_agg_sets
    negatives = (par.entity_unitsize[group_to_unit_selected.columns.get_level_values('process')] < 0).values
    group_to_unit_selected_negative = group_to_unit_selected[group_to_unit_selected.columns[negatives]]
    group_to_unit_selected_positive = group_to_unit_selected[group_to_unit_selected.columns[~negatives]]
    r.group_output__group_aggregate_Group_to_unit_positive__dt = group_to_unit_selected_positive.T.groupby(level=['group', 'group_aggregate']).sum().T
    r.group_output__group_aggregate_Group_to_unit_negative__dt = group_to_unit_selected_negative.T.groupby(level=['group', 'group_aggregate']).sum().T

    r.group_output__group_aggregate_Unit_to_group__dt = r.group_output__group_aggregate_Unit_to_group_positive__dt.sub(
        r.group_output__group_aggregate_Group_to_unit_negative__dt, fill_value=0.0)
    r.group_output__group_aggregate_Group_to_unit__dt = r.group_output__group_aggregate_Group_to_unit_positive__dt.sub(
        r.group_output__group_aggregate_Unit_to_group_negative__dt, fill_value=0.0)

    r.group_output__group_aggregate_Group_to_unit__d = r.group_output__group_aggregate_Group_to_unit__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)
    r.group_output__group_aggregate_Unit_to_group__d = r.group_output__group_aggregate_Unit_to_group__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)

    losses_set = s.outputNodeGroup__process_fully_inside
    group_losses_sets_node_to_unit = losses_set.join(node_to_unit.columns, how='inner')
    group_losses_sets_unit_to_node = losses_set.join(unit_to_node.columns, how='inner')

    node_to_unit_filtered = node_to_unit[group_losses_sets_node_to_unit.droplevel('group')]
    unit_to_node_filtered = unit_to_node[group_losses_sets_unit_to_node.droplevel('group')]
    node_to_unit_filtered.columns = group_losses_sets_node_to_unit
    unit_to_node_filtered.columns = group_losses_sets_unit_to_node

    unit_losses_dt = node_to_unit_filtered.sub(unit_to_node_filtered, axis=1, fill_value=0.0)
    r.group_output_Internal_unit_losses__dt = unit_losses_dt
    r.group_output_Internal_unit_losses__d = r.group_output_Internal_unit_losses__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)

    # --- Node inflow by group ---
    dt_dispatch_idx = s.dt_realize_dispatch
    r_group_inflow = pd.DataFrame(0.0, index=dt_dispatch_idx, columns=s.outputNodeGroup_does_generic_flows, dtype=float)
    for g in s.outputNodeGroup_does_generic_flows:
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
        else pd.DataFrame(0.0, index=s.d_realized_period, columns=s.outputNodeGroup_does_generic_flows))

    # --- Storage state losses by group ---
    r_group_state_losses = pd.DataFrame(0.0, index=dt_dispatch_idx, columns=s.outputNodeGroup_does_generic_flows, dtype=float)
    for g in s.outputNodeGroup_does_generic_flows:
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
        else pd.DataFrame(0.0, index=s.d_realized_period, columns=s.outputNodeGroup_does_generic_flows))

    # --- Node slack by group ---
    r.group_node_up_slack__dt = pd.DataFrame(index=s.dt_realize_dispatch)
    r.group_node_down_slack__dt = pd.DataFrame(index=s.dt_realize_dispatch)
    for g in s.outputNodeGroup_does_generic_flows:
        g_node = s.group_node[s.group_node.get_level_values('node').isin(s.node_balance) & s.group_node.get_level_values('group').isin([g])].get_level_values('node')
        r.group_node_up_slack__dt[g] = r.upward_node_slack_dt[g_node].sum(axis=1)
        r.group_node_down_slack__dt[g] = r.downward_node_slack_dt[g_node].sum(axis=1)
