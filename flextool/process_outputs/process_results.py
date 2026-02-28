from types import SimpleNamespace
import pandas as pd

from flextool.process_outputs.drop_levels import drop_levels
from flextool.process_outputs.calc_capacity_flows import compute_capacity_and_flows
from flextool.process_outputs.calc_connections import compute_connection_flows
from flextool.process_outputs.calc_storage_vre import compute_storage_and_vre


def post_process_results(par, s, v):
    """Calculate post-processing results from variables, parameters, and sets"""
    r = SimpleNamespace()

    par, s, v = drop_levels(par, s, v)
    compute_capacity_and_flows(par, s, v, r)
    compute_connection_flows(par, s, v, r)
    compute_storage_and_vre(par, s, v, r)

    # --- Group connection flows (uses r.from_conn and r.to_conn from compute_connection_flows) ---

    # r_group_output__from_connection_not_in_aggregate__dt
    from_conn_set = s.outputNodeGroup__process__connection__to_node_Not_in_aggregate.droplevel('connection')
    group_sets = r.from_conn.columns.join(from_conn_set, how='inner')
    from_conn_selected = r.from_conn[group_sets.droplevel('group')]
    from_conn_selected.columns = group_sets
    r.group_output__from_connection_not_in_aggregate__dt = from_conn_selected

    # r_group_output__to_connection_not_in_aggregate__dt
    to_conn_set = s.outputNodeGroup__process__node__to_connection_Not_in_aggregate.droplevel('connection')
    group_sets = r.to_conn.columns.join(to_conn_set, how='inner')
    to_conn_selected = r.to_conn[group_sets.droplevel('group')]
    to_conn_selected.columns = group_sets
    r.group_output__to_connection_not_in_aggregate__dt = to_conn_selected

    # Period aggregations
    r.group_output__from_connection_not_in_aggregate__d = r.group_output__from_connection_not_in_aggregate__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)
    r.group_output__to_connection_not_in_aggregate__d = r.group_output__to_connection_not_in_aggregate__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)

    # r_group_output__from_connection_aggregate__dt
    from_conn_agg_set = s.outputNodeGroup__processGroup__process__connection__to_node.droplevel('connection')
    group_agg_sets = r.from_conn.columns.join(from_conn_agg_set, how='inner')
    from_conn_agg_selected = r.from_conn[group_agg_sets.droplevel(['group', 'group_aggregate'])]
    from_conn_agg_selected.columns = group_agg_sets
    r.group_output__from_connection_aggregate__dt = from_conn_agg_selected.T.groupby(level=['group', 'group_aggregate']).sum().T

    # r_group_output__to_connection_aggregate__dt
    to_conn_agg_set = s.outputNodeGroup__processGroup__process__node__to_connection.droplevel('connection')
    group_agg_sets = to_conn_agg_set.join(r.to_conn.columns)
    to_conn_agg_selected = r.to_conn[group_agg_sets.droplevel(['group', 'group_aggregate'])]
    to_conn_agg_selected.columns = group_agg_sets
    r.group_output__to_connection_aggregate__dt = to_conn_agg_selected.T.groupby(level=['group', 'group_aggregate']).sum().T

    # Daily aggregations
    r.group_output__from_connection_aggregate__d = r.group_output__from_connection_aggregate__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)
    r.group_output__to_connection_aggregate__d = r.group_output__to_connection_aggregate__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)

    # r_group_output_Internal_connection_losses__dt
    losses_set = s.outputNodeGroup__process_fully_inside
    group_losses_sets = r.connection_losses_dt.columns.join(losses_set, how='inner')
    losses_selected = r.connection_losses_dt[group_losses_sets.droplevel('group')]
    losses_selected.columns = group_losses_sets
    r.group_output_Internal_connection_losses__dt = losses_selected

    # r_group_output_Internal_connection_losses__d
    r.group_output_Internal_connection_losses__d = r.group_output_Internal_connection_losses__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)

    # --- Commodity costs ---

    # r_cost_commodity_dt
    flow_from_commodity_node = r.flow_dt[r.flow_dt.columns[r.flow_dt.columns.get_level_values(level=1).isin(s.commodity_node.get_level_values(level=1))]]
    flow_to_commodity_node = r.flow_dt[r.flow_dt.columns[r.flow_dt.columns.get_level_values(level=2).isin(s.commodity_node.get_level_values(level=1))]]

    commodity_price = par.commodity_price[s.commodity_node.get_level_values('commodity').unique()]
    commodity_price.columns = commodity_price.columns.join(s.commodity_node)
    flow_from_commodity_node.columns.names = ['process', 'node', 'sink']
    flow_from_commodity_node.columns = flow_from_commodity_node.columns.join(commodity_price.columns)
    flow_from_commodity = flow_from_commodity_node.T.groupby('commodity').sum().T
    r.cost_commodity_dt = flow_from_commodity.mul(commodity_price).mul(par.step_duration, axis=0)
    flow_to_commodity_node.columns.names = ['process', 'source', 'node']
    flow_to_commodity_node.columns = flow_to_commodity_node.columns.join(commodity_price.columns)
    flow_to_commodity = flow_to_commodity_node.T.groupby('commodity').sum().T
    r.sales_commodity_dt = flow_to_commodity.mul(commodity_price).mul(par.step_duration, axis=0)

    r.cost_commodity_d = r.cost_commodity_dt.groupby('period').sum()
    r.sales_commodity_d = r.sales_commodity_dt.groupby('period').sum()

    # --- CO2 emissions ---

    # r_process_emissions_co2_dt
    flow_outof_cols = r.flow_dt.columns.copy()
    flow_outof_cols.names = ['process', 'node', 'sink']
    flow_outof_cols = s.process__commodity__node_co2.join(flow_outof_cols)
    flow_outof_cols = flow_outof_cols[~flow_outof_cols.get_level_values('sink').isna()]

    flow_into_cols = r.flow_dt.columns.copy()
    flow_into_cols.names = ['process', 'source', 'node']
    flow_into_cols = s.process__commodity__node_co2.join(flow_into_cols)
    flow_into_cols = flow_into_cols[~flow_into_cols.get_level_values('source').isna()]
    flow_into_cols = flow_into_cols.reorder_levels(order=['process', 'commodity', 'source', 'node'])

    flow_outof_node = r.flow_dt[flow_outof_cols.droplevel('commodity')]
    flow_outof_node.columns.names = ['process', 'node', 'sink']
    flow_outof_node.columns = flow_outof_node.columns.join(flow_outof_cols)

    flow_into_node = r.flow_dt[flow_into_cols.droplevel('commodity')]
    flow_into_node.columns.names = ['process', 'source', 'node']
    flow_into_node.columns = flow_into_node.columns.join(flow_into_cols)

    flow_into_node_grouped = flow_into_node.T.groupby(level=[0, 2, 3]).sum().T
    flow_outof_node_grouped = flow_outof_node.T.groupby(level=[0, 1, 3]).sum().T

    net_flow = flow_outof_node_grouped.sub(flow_into_node_grouped, fill_value=0)
    net_flow_with_duration = net_flow.mul(par.step_duration, axis=0)
    r.process_emissions_co2_dt = net_flow_with_duration.mul(par.commodity_co2_content, axis=1, level='commodity')

    cols_df = r.process_emissions_co2_dt.columns.to_frame(index=False)
    cols_df['type'] = 'unit'
    cols_df.loc[cols_df['process'].isin(s.process_connection), 'type'] = 'connection'
    r.process_emissions_co2_dt.columns = pd.MultiIndex.from_frame(
        cols_df[['type', 'process', 'commodity', 'node']]
    )

    r.process_emissions_co2_d = r.process_emissions_co2_dt.groupby(level='period').sum()
    r.process_emissions_co2_d = r.process_emissions_co2_d.div(par.complete_period_share_of_year, axis=0)

    r.emissions_co2_d = r.process_emissions_co2_d.sum(axis=1)
    r.emissions_co2_dt = r.process_emissions_co2_dt.sum(axis=1)

    # r_group co2
    s.group_node_co2 = s.group_node[s.group_node.get_level_values('group').isin(s.group_co2_limit.union(s.group_co2_price))]
    group_process_co2_columns = r.process_emissions_co2_dt.columns.join(s.group_node_co2)
    r.group_process_emissions_co2_dt = pd.DataFrame(index=r.process_emissions_co2_dt.index, columns=group_process_co2_columns)
    for col in group_process_co2_columns:
        r.group_process_emissions_co2_dt[col] = r.process_emissions_co2_dt[col[:4]]
    r.group_co2_dt = r.group_process_emissions_co2_dt.T.groupby('group').sum().T
    r.group_co2_d = r.group_co2_dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)
    r.group_cost_co2_dt = r.group_co2_dt.mul(par.group_co2_price)
    r.group_cost_co2_d = r.group_co2_d.mul(par.group_co2_price)
    r.cost_co2_dt = r.group_cost_co2_dt.sum(axis=1)
    r.cost_co2_d = r.group_cost_co2_d.groupby('period').sum()

    # r_cost_process_other_operational_cost_dt
    relevant_flows = r.flow_dt.loc[:, r.flow_dt.columns.intersection(par.process_source_sink_varCost.columns)]
    cost_flows = relevant_flows.mul(par.step_duration, axis=0).mul(par.process_source_sink_varCost, axis=1)
    r.cost_process_other_operational_cost_dt = cost_flows.T.groupby(level=0).sum().T.reindex(columns=s.process, fill_value=0.0)

    # r_process_startup_dt
    r.process_startup_dt = v.startup_linear.add(v.startup_integer, fill_value=0)

    # r_cost_startup_dt
    r.cost_startup_dt = pd.DataFrame(0.0, index=r.process_startup_dt.index, columns=s.process_online, dtype=float)
    valid_processes = s.process_online.intersection(r.process_startup_dt.columns).intersection(par.process_startup_cost.columns)
    if len(valid_processes) > 0:
        cost = r.process_startup_dt[valid_processes].mul(par.entity_unitsize[valid_processes], axis=1)
        periods = cost.index.get_level_values('period')
        period_costs = par.process_startup_cost.loc[periods, valid_processes]
        period_costs.index = cost.index
        r.cost_startup_dt[valid_processes] = cost.mul(period_costs)

    # --- Slacks and reserves ---

    r.reserves_dt = v.reserve.mul(par.step_duration, axis=0)
    r.reserves_d = r.reserves_dt.groupby('period').sum() \
        .div(par.complete_period_share_of_year, axis=0)

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

    r.q_inertia_dt = v.q_inertia.mul(par.group_inertia_limit)
    r.q_inertia_d_not_annualized = r.q_inertia_dt.mul(par.step_duration, axis=0).groupby('period').sum()
    r.q_inertia_d = r.q_inertia_d_not_annualized.div(par.complete_period_share_of_year, axis=0)
    r.costPenalty_inertia_dt = r.q_inertia_dt.mul(par.group_penalty_inertia)

    r.q_non_synchronous_dt = v.q_non_synchronous.mul(par.group_capacity_for_scaling[s.groupNonSync])
    r.q_non_synchronous_d_not_annualized = r.q_non_synchronous_dt.mul(par.step_duration, axis=0) \
        .groupby('period').sum()
    r.q_non_synchronous_d = r.q_non_synchronous_d_not_annualized.div(par.complete_period_share_of_year, axis=0)
    r.costPenalty_non_synchronous_dt = r.q_non_synchronous_dt.mul(par.group_penalty_non_synchronous)

    r.q_capacity_margin_d_not_annualized = v.q_capacity_margin \
        .mul(par.group_capacity_for_scaling[s.groupCapacityMargin])
    r.costPenalty_capacity_margin_d = r.q_capacity_margin_d_not_annualized \
        .mul(par.discount_factor_operations_yearly, axis=0).sum(axis=1)

    r.q_reserves_dt = v.q_reserve.mul(par.reserve_upDown_group_reservation[v.q_reserve.columns], axis=1)
    r.q_reserves_d_not_annualized = r.q_reserves_dt.mul(par.step_duration, axis=0).groupby(level='period').sum()
    r.q_reserves_d = r.q_reserves_d_not_annualized.div(par.complete_period_share_of_year, axis=0)
    r.costPenalty_reserve_upDown_dt = v.q_reserve.mul(par.step_duration, axis=0) \
        .mul(par.reserve_upDown_group_penalty, axis=1) \
        .mul(par.reserve_upDown_group_reservation, axis=1)

    # --- Investment costs ---

    r.cost_entity_invest_d = v.invest.mul(par.entity_unitsize[v.invest.columns]).mul(par.entity_annual_discounted)
    r.cost_entity_divest_d = -v.divest.mul(par.entity_unitsize[v.divest.columns]).mul(par.entity_annual_divest_discounted)
    r.cost_entity_fixed_pre_existing = (par.entity_pre_existing * par.entity_fixed_cost).mul(par.discount_factor_operations_yearly, axis=0)
    r.cost_entity_fixed_invested = (v.invest.mul(par.entity_unitsize[v.invest.columns] * par.entity_lifetime_fixed_cost[v.invest.columns]))
    r.cost_entity_fixed_divested = -(v.divest.mul(par.entity_unitsize[v.divest.columns] * par.entity_lifetime_fixed_cost_divest[v.divest.columns]))

    # --- Aggregate costs ---

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

    # --- Group flows (unit) ---

    unit_flows = r.flow_dt[r.flow_dt.columns[r.flow_dt.columns.get_level_values('process').isin(s.process_unit)]]

    unit_to_node = unit_flows[unit_flows.columns[unit_flows.columns.droplevel('source').isin(s.process_sink)]]
    node_to_unit = unit_flows[unit_flows.columns[unit_flows.columns.droplevel('sink').isin(s.process_source)]]

    unit_to_node.columns = unit_to_node.columns.droplevel('source')
    node_to_unit.columns = node_to_unit.columns.droplevel('sink')

    unit_to_node.columns.names = ['process', 'node']
    node_to_unit.columns.names = ['process', 'node']

    # r_group_output__unit_to_node_not_in_aggregate__dt
    unit_to_node_set = s.outputNodeGroup__process__unit__to_node_Not_in_aggregate.droplevel('unit')
    group_sets = unit_to_node.columns.join(unit_to_node_set, how='inner')
    unit_to_node_selected = unit_to_node[group_sets.droplevel('group')]
    unit_to_node_selected.columns = group_sets
    r.group_output__unit_to_node_not_in_aggregate__dt = unit_to_node_selected

    # r_group_output__node_to_unit_not_in_aggregate__dt
    node_to_unit_set = s.outputNodeGroup__process__node__to_unit_Not_in_aggregate.droplevel('unit')
    group_sets = node_to_unit.columns.join(node_to_unit_set, how='inner')
    node_to_unit_selected = node_to_unit[group_sets.droplevel('group')]
    node_to_unit_selected.columns = group_sets
    r.group_output__node_to_unit_not_in_aggregate__dt = node_to_unit_selected

    r.group_output__unit_to_node_not_in_aggregate__d = r.group_output__unit_to_node_not_in_aggregate__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)
    r.group_output__node_to_unit_not_in_aggregate__d = r.group_output__node_to_unit_not_in_aggregate__dt.groupby('period').sum().div(par.complete_period_share_of_year, axis=0)

    # r_group_output__group_aggregate_Unit_to_group__dt
    unit_to_group_set = s.outputNodeGroup__processGroup__process__unit__to_node.droplevel('unit')
    group_agg_sets = unit_to_node.columns.join(unit_to_group_set, how='inner')
    unit_to_group_selected = unit_to_node[group_agg_sets.droplevel(['group', 'group_aggregate'])]
    unit_to_group_selected.columns = group_agg_sets
    negatives = (par.entity_unitsize[unit_to_group_selected.columns.get_level_values('process')] < 0).values
    unit_to_group_selected_negative = unit_to_group_selected[unit_to_group_selected.columns[negatives]]
    unit_to_group_selected_positive = unit_to_group_selected[unit_to_group_selected.columns[~negatives]]
    r.group_output__group_aggregate_Unit_to_group_positive__dt = unit_to_group_selected_positive.T.groupby(level=['group', 'group_aggregate']).sum().T
    r.group_output__group_aggregate_Unit_to_group_negative__dt = unit_to_group_selected_negative.T.groupby(level=['group', 'group_aggregate']).sum().T

    # r_group_output__group_aggregate_Group_to_unit__dt
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

    # r_group_output_Internal_unit_losses__dt
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

    # r_group_node_inflow_dt
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

    # r_group_node_state_losses__dt
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

    # r_group_node slacks
    r.group_node_up_slack__dt = pd.DataFrame(index=s.dt_realize_dispatch)
    r.group_node_down_slack__dt = pd.DataFrame(index=s.dt_realize_dispatch)
    for g in s.outputNodeGroup_does_generic_flows:
        g_node = s.group_node[s.group_node.get_level_values('node').isin(s.node_balance) & s.group_node.get_level_values('group').isin([g])].get_level_values('node')
        r.group_node_up_slack__dt[g] = r.upward_node_slack_dt[g_node].sum(axis=1)
        r.group_node_down_slack__dt[g] = r.downward_node_slack_dt[g_node].sum(axis=1)

    return r
