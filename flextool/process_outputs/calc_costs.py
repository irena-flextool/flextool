import pandas as pd


def _agg_period_or_empty(df: pd.DataFrame, realized_periods, period_level: str = 'period') -> pd.DataFrame:
    """Group-by period sum, or an all-zeros frame when df is empty."""
    if df.empty:
        cols = df.columns if hasattr(df, 'columns') else []
        return pd.DataFrame(0.0, index=realized_periods, columns=cols)
    return df[df.index.get_level_values(period_level).isin(realized_periods)].groupby(period_level).sum()


def compute_costs(par, s, v, r) -> None:
    """Compute all cost quantities (depends on r.costPenalty_* from compute_slacks)."""
    # --- Commodity costs ---
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

    # Group CO2
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

    # --- Operational costs ---
    relevant_flows = r.flow_dt.loc[:, r.flow_dt.columns.intersection(par.process_source_sink_varCost.columns)]
    cost_flows = relevant_flows.mul(par.step_duration, axis=0).mul(par.process_source_sink_varCost, axis=1)
    r.cost_process_other_operational_cost_dt = cost_flows.T.groupby(level=0).sum().T.reindex(columns=s.process, fill_value=0.0)

    # --- Startup costs ---
    r.process_startup_dt = v.startup_linear.add(v.startup_integer, fill_value=0)
    r.cost_startup_dt = pd.DataFrame(0.0, index=r.process_startup_dt.index, columns=s.process_online, dtype=float)
    valid_processes = s.process_online.intersection(r.process_startup_dt.columns).intersection(par.process_startup_cost.columns)
    if len(valid_processes) > 0:
        cost = r.process_startup_dt[valid_processes].mul(par.entity_unitsize[valid_processes], axis=1)
        periods = cost.index.get_level_values('period')
        period_costs = par.process_startup_cost.loc[periods, valid_processes]
        period_costs.index = cost.index
        r.cost_startup_dt[valid_processes] = cost.mul(period_costs)

    # --- Investment costs ---
    r.cost_entity_invest_d = v.invest.mul(par.entity_unitsize[v.invest.columns]).mul(par.entity_annual_discounted)
    r.cost_entity_divest_d = -v.divest.mul(par.entity_unitsize[v.divest.columns]).mul(par.entity_annual_divest_discounted)
    r.cost_entity_fixed_pre_existing = (par.entity_pre_existing * par.entity_fixed_cost).mul(par.discount_factor_operations_yearly, axis=0)
    r.cost_entity_fixed_invested = (v.invest.mul(par.entity_unitsize[v.invest.columns] * par.entity_lifetime_fixed_cost[v.invest.columns]))
    r.cost_entity_fixed_divested = -(v.divest.mul(par.entity_unitsize[v.divest.columns] * par.entity_lifetime_fixed_cost_divest[v.divest.columns]))

    # --- Aggregate operational costs (depends on r.costPenalty_* from compute_slacks) ---
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

    r.cost_startup_d = _agg_period_or_empty(r.cost_startup_dt, s.d_realized_period)
    r.costPenalty_inertia_d = _agg_period_or_empty(r.costPenalty_inertia_dt, s.d_realized_period)
    r.costPenalty_non_synchronous_d = _agg_period_or_empty(r.costPenalty_non_synchronous_dt, s.d_realized_period)
    r.costPenalty_reserve_upDown_d = _agg_period_or_empty(r.costPenalty_reserve_upDown_dt, s.d_realized_period)

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
