import numpy as np
import pandas as pd

from flextool.process_outputs._annualize import annualize_dt_to_d
from flextool.process_outputs._branch_weight import weight_dt_by_branch


def _agg_period_or_empty(df: pd.DataFrame, realized_periods, period_level: str = 'period') -> pd.DataFrame:
    """Group-by period sum, or an all-zeros frame when df is empty."""
    if df.empty:
        cols = df.columns if hasattr(df, 'columns') else []
        return pd.DataFrame(0.0, index=realized_periods, columns=cols)
    return df[df.index.get_level_values(period_level).isin(realized_periods)].groupby(period_level).sum()


def compute_costs(par, s, v, r) -> None:
    """Compute all cost quantities (depends on r.costPenalty_* from compute_slacks).

    Per-timestep cost terms must carry the same step-level scaling set as
    the LP objective in flextool.mod: ``step_duration * p_rp_cost_weight``.
    Period-level scaling (``p_inflation_factor_operations_yearly /
    complete_period_share_of_year``) is applied when aggregating *_dt →
    *_d below.
    """
    # --- Commodity costs ---
    # In-memory ``commodity_node`` is derived from the FlexData flow_*
    # frames (covers every (c, n) referenced by an arc, regardless of
    # whether a price was authored).  ``par.commodity_price`` only
    # carries columns for commodities with a populated ``p_commodity_price``
    # entry — so intersect here to avoid KeyError when an arc references
    # a price-less commodity (e.g. coal_chp where coal has only a
    # co2_content row).  Legacy behaviour: cost contribution is 0.
    # Commodities priced on a price ladder (price_ladder_annual /
    # price_ladder_cumulative) are excluded here and priced from v_trade
    # below instead (#57) — the LP objective skips them in the flat
    # commodity-price term (model/core.py _obj_commodity), so the flat-price
    # output must too, or the ladder commodity would be double-priced (or,
    # without a flat `price` authored, silently cost nothing at all).
    ladder_commodities = getattr(s, 'commodity_with_ladder', pd.Index([], name='commodity'))
    available_commodities = pd.Index(par.commodity_price.columns).unique().difference(ladder_commodities)
    commodity_node = s.commodity_node[
        s.commodity_node.get_level_values('commodity').isin(available_commodities)
    ]
    flow_from_commodity_node = r.flow_dt[r.flow_dt.columns[r.flow_dt.columns.get_level_values(level=1).isin(commodity_node.get_level_values(level=1))]]
    flow_to_commodity_node = r.flow_dt[r.flow_dt.columns[r.flow_dt.columns.get_level_values(level=2).isin(commodity_node.get_level_values(level=1))]]

    # Per-step scaling factor: step_duration × timestep_weight.  Used by all
    # per-(d, t) cost terms below to match the objective in flextool.mod.
    # Option (A): fold in pdt_branch_weight[d, t] so every operational cost
    # term built from step_x_rp (commodity buy/sell, per-process commodity,
    # other operational) carries the LP's per-(d, t) probability weight.
    # step_x_rp feeds ONLY cost terms here (never a physical quantity), so
    # folding the weight in is safe; deterministic runs weight by 1.0.
    _pdt_bw = getattr(par, "pdt_branch_weight", None)
    step_x_rp = par.step_duration.mul(par.timestep_weight, axis=0)
    step_x_rp = weight_dt_by_branch(step_x_rp, _pdt_bw)

    commodity_price = par.commodity_price[commodity_node.get_level_values('commodity').unique()]
    commodity_price.columns = commodity_price.columns.join(commodity_node)
    flow_from_commodity_node.columns.names = ['process', 'node', 'sink']
    flow_from_commodity_node.columns = flow_from_commodity_node.columns.join(commodity_price.columns)
    flow_from_commodity = flow_from_commodity_node.T.groupby('commodity').sum().T
    r.cost_commodity_dt = flow_from_commodity.mul(commodity_price).mul(step_x_rp, axis=0)
    flow_to_commodity_node.columns.names = ['process', 'source', 'node']
    flow_to_commodity_node.columns = flow_to_commodity_node.columns.join(commodity_price.columns)
    flow_to_commodity = flow_to_commodity_node.T.groupby('commodity').sum().T
    r.sales_commodity_dt = flow_to_commodity.mul(commodity_price).mul(step_x_rp, axis=0)

    # --- Commodity price ladder (#57): price the tiers from v_trade ---
    # v_trade[c, n, d, i] is decided once per period (no time axis), so its
    # cost is annualised the way the LP objective annualises it
    # (inflation_op/period_share · the period's branch weight,
    # model/_commodity_ladder.py objective_terms) and then spread across the
    # period's (d, t) rows proportionally to step_x_rp, so it folds into
    # cost_commodity_dt / cost_commodity_d / costOper_dt / costOper_d (and
    # out_costs.py's per-(d, t) commodity_cost output) the same way every
    # other commodity cost does, with no separate annualisation of its own.
    trade = getattr(v, 'trade', None)
    if trade is not None and not trade.empty and len(ladder_commodities) > 0:
        trade = trade.loc[:, trade.columns.get_level_values('commodity').isin(ladder_commodities)]
    if trade is not None and not trade.empty:
        unitsize = par.commodity_unitsize.reindex(trade.columns.get_level_values('commodity')).fillna(1.0)
        unitsize.index = trade.columns
        mwh = trade.mul(unitsize, axis=1)

        # Tier price: the period-varying annual ladder price where authored, else the period-constant
        # cumulative ladder price (broadcast over every period present in mwh).
        tiers = mwh.columns.droplevel('node')
        ann_price = par.ladder_price_ann.reindex(index=mwh.index, columns=tiers)
        cum_price = pd.DataFrame(
            np.tile(par.ladder_price_cum.reindex(tiers).to_numpy(), (len(mwh.index), 1)),
            index=mwh.index, columns=tiers,
        )
        price = ann_price.fillna(cum_price)
        price.columns = mwh.columns

        # Branch weight is per period (pd_branch_weight keeps its 'solve' level — drop_levels does not
        # touch it, _weight_by_branch below mirrors this same lookup); default 1.0 outside stochastic solves.
        bw_series = getattr(par, 'pd_branch_weight', None)
        bw_map: dict[str, float] = {}
        if bw_series is not None:
            for idx, val in bw_series.items():
                period = idx[-1] if isinstance(idx, tuple) else idx
                bw_map[str(period)] = float(val)
        bw = pd.Series(
            [bw_map.get(str(p), 1.0) for p in mwh.index], index=mwh.index,
        )
        ladder_cost_d = mwh.mul(price).mul(bw, axis=0)
        ladder_cost_d.columns = tiers
        ladder_cost_d = ladder_cost_d.T.groupby('commodity').sum().T

        # Spread each period's ladder cost across its (d, t) rows proportionally to step_x_rp — the
        # model requires a uniform branch weight within a ladder period, so this is a pure
        # step_duration x timestep_weight share and reconciles to ladder_cost_d exactly on re-summing.
        share = step_x_rp.div(step_x_rp.groupby(level='period').transform('sum'))
        ladder_cost_dt = ladder_cost_d.reindex(share.index.get_level_values('period'))
        ladder_cost_dt.index = share.index
        ladder_cost_dt = ladder_cost_dt.mul(share, axis=0)
        r.cost_commodity_dt = r.cost_commodity_dt.add(ladder_cost_dt, fill_value=0.0)

    r.cost_commodity_d = r.cost_commodity_dt.groupby('period').sum()
    r.sales_commodity_d = r.sales_commodity_dt.groupby('period').sum()

    # --- Per-process commodity (fuel) buy & sales (additive output layer) ---
    # Attribute fuel cost to the consuming process.  Same dt-price, step_x_rp
    # scaling and sign convention as the per-commodity totals above; only the
    # collapse axis differs (process, not commodity).  Build a price frame
    # whose columns mirror the flow's full column MultiIndex (price selected
    # by each column's commodity level), then a plain element-wise mul — a
    # level= broadcast raises NotImplementedError on the non-unique commodity
    # key.  Summing these over 'process' reproduces r.cost_commodity_dt /
    # r.sales_commodity_dt exactly (reconciliation invariant).
    _price_by_commodity = commodity_price.droplevel('node', axis=1)
    _price_by_commodity = _price_by_commodity.loc[:, ~_price_by_commodity.columns.duplicated()]

    def _per_process_commodity_cost(flow_frame):
        pf = _price_by_commodity.reindex(
            columns=flow_frame.columns.get_level_values('commodity'))
        pf.columns = flow_frame.columns
        return flow_frame.mul(step_x_rp, axis=0).mul(pf).T.groupby('process').sum().T

    r.cost_commodity_process_dt = _per_process_commodity_cost(flow_from_commodity_node)
    r.sales_commodity_process_dt = _per_process_commodity_cost(flow_to_commodity_node)

    # --- CO2 emissions ---
    flow_outof_cols = r.flow_dt.columns.copy()
    flow_outof_cols.names = ['process', 'node', 'sink']
    flow_outof_cols = s.process__commodity__node_co2.join(flow_outof_cols)
    flow_outof_cols = flow_outof_cols[~flow_outof_cols.get_level_values('sink').isna()]

    # Removal credit (#19, Q14; c18a gap): join against the DEDICATED
    # removal set (keyed by the CREDITING process's own sink arc), not
    # ``process__commodity__node_co2`` (the emission-side set, keyed by the
    # EMITTING process's source arc).  The old code joined into-flows
    # against the emission set, which only matched a flow back into the
    # SAME process's SAME node -- a same-process round trip -- and missed
    # the LP's actual removal credit (``flow_to_co2_*``), which any OTHER
    # process can earn by delivering into the group's node (e.g. DAC/BECCS
    # distinct from whatever emits in the group).  Net result below still
    # subtracts by (process, node, commodity): a different crediting
    # process produces its OWN (negative) column instead of cancelling the
    # emitting process's column -- correct, since results.md's per-process
    # CO2 cost attributes a removal flow's credit to the process that
    # delivers it, not to whichever process happens to emit at that node.
    flow_into_cols = r.flow_dt.columns.copy()
    flow_into_cols.names = ['process', 'source', 'node']
    flow_into_cols = s.process__commodity__node_co2_removal.join(flow_into_cols)
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
    # Emissions are physical MWh, so only step_duration applies here (no
    # timestep_weight).  timestep_weight is added downstream on the monetary
    # group_cost_co2_dt to match the objective's CO2-price term
    # (flextool.mod ~line 2312-2336).
    net_flow_with_duration = net_flow.mul(par.step_duration, axis=0)
    r.process_emissions_co2_dt = net_flow_with_duration.mul(par.commodity_co2_content, axis=1, level='commodity')

    cols_df = r.process_emissions_co2_dt.columns.to_frame(index=False)
    cols_df['type'] = 'unit'
    cols_df.loc[cols_df['process'].isin(s.process_connection), 'type'] = 'connection'
    r.process_emissions_co2_dt.columns = pd.MultiIndex.from_frame(
        cols_df[['type', 'process', 'commodity', 'node']]
    )

    # Reported annual emissions (tonnes) must match the already-weighted CO2
    # cap and CO2 cost: weight each (d, t) by par.timestep_weight (=1.0 with
    # no/uniform timeset_weights → byte-identical) before the period sum.
    # process_emissions_co2_dt carries step_duration but NOT timestep_weight
    # (see net_flow_with_duration above), so this is the only weight applied.
    r.process_emissions_co2_d = annualize_dt_to_d(r.process_emissions_co2_dt, par.timestep_weight, par.complete_period_share_of_year)

    r.emissions_co2_d = r.process_emissions_co2_d.sum(axis=1)
    r.emissions_co2_dt = r.process_emissions_co2_dt.sum(axis=1)

    # Group CO2
    s.group_node_co2 = s.group_node[s.group_node.get_level_values('group').isin(s.group_co2_limit.union(s.group_co2_price))]
    # Inner join: a process/node with no co2-priced/capped group at all (the physical-emissions-only
    # case #_physical_co2_emission_pieces widened process_emissions_co2_dt to cover) has no group to
    # price it against.  Index.join's default how='left' would keep it with group=NaN, and the NaN
    # group's price reindexes to NaN too — summed into cost_process_co2_dt's groupby('process').sum()
    # below, which treats an all-NaN group as 0 (pandas sum() skips NaN) rather than leaving the
    # process out, emitting a spurious all-zero 'co2' cost column/row for every process without a
    # priced group.  group_co2_dt / cost_co2_dt (grouped by 'group') are unaffected either way: a
    # NaN group key is already dropped by groupby('group'), so it never contributed to a real group's
    # total — only the unconditionally-physical emissions outputs above (process_emissions_co2_d etc.)
    # cover the ungrouped case, same as before this join existed.
    group_process_co2_columns = r.process_emissions_co2_dt.columns.join(s.group_node_co2, how='inner')
    r.group_process_emissions_co2_dt = pd.DataFrame(index=r.process_emissions_co2_dt.index, columns=group_process_co2_columns)
    for col in group_process_co2_columns:
        r.group_process_emissions_co2_dt[col] = r.process_emissions_co2_dt[col[:4]]
    r.group_co2_dt = r.group_process_emissions_co2_dt.T.groupby('group').sum().T
    # Reported annual group emissions (tonnes): weight each (d, t) by
    # par.timestep_weight (=1.0 with no/uniform timeset_weights →
    # byte-identical) before the period sum so it matches the CO2 cap/cost.
    # group_co2_dt derives from process_emissions_co2_dt (step_duration only,
    # no timestep_weight), so this is the only weight applied.
    r.group_co2_d = annualize_dt_to_d(r.group_co2_dt, par.timestep_weight, par.complete_period_share_of_year)
    # Monetary CO2 cost = emissions × price.  Emissions already carry
    # step_duration; add timestep_weight to match the objective's CO2-price
    # term (flextool.mod ~line 2312-2336 where it has
    # step_duration × timestep_weight).  Per-period cost_co2 is then
    # aggregated from the rp-weighted cost_dt directly (instead of
    # multiplying the annualized physical emissions by price, which would
    # drop the per-step rp weighting).
    # Option (A): weight the CO2 *cost* by pdt_branch_weight (the LP's CO2
    # price term carries op_factor — model/core.py §4).  The physical CO2 tonnes
    # (group_co2_d / process_emissions_co2_d above) stay UNWEIGHTED.
    r.group_cost_co2_dt = r.group_co2_dt.mul(par.group_co2_price).mul(par.timestep_weight, axis=0)
    r.group_cost_co2_dt = weight_dt_by_branch(r.group_cost_co2_dt, _pdt_bw)
    r.group_cost_co2_d = r.group_cost_co2_dt.groupby('period').sum()
    r.cost_co2_dt = r.group_cost_co2_dt.sum(axis=1)
    r.cost_co2_d = r.group_cost_co2_d.sum(axis=1)

    # --- Per-process CO2 cost (additive output layer) ---
    # group_process_emissions_co2_dt replicates per-process emissions once per
    # priced group the process's node belongs to.  Multiply by the dt group
    # price (column-aligned to the 'group' level) and timestep_weight, then
    # groupby('process').sum() — this SUMS the per-group charges over every
    # priced group the process touches, matching the LP objective and
    # calc_costs.py:104-127.  Summing over 'process' reproduces r.cost_co2_dt.
    _co2_cols = r.group_process_emissions_co2_dt.columns
    _co2_price = par.group_co2_price.reindex(columns=_co2_cols.get_level_values('group'))
    _co2_price.columns = _co2_cols
    r.cost_process_co2_dt = (
        r.group_process_emissions_co2_dt
        .mul(_co2_price)
        .mul(par.timestep_weight, axis=0)
        .T.groupby('process').sum().T
    )
    # Option (A): same per-(d, t) weight as the per-group CO2 cost above so
    # the per-process CO2 cost reconciles to r.cost_co2_dt.
    r.cost_process_co2_dt = weight_dt_by_branch(r.cost_process_co2_dt, _pdt_bw)

    # --- Operational costs ---
    # Mod objective (~line 2347-2376): flow × unitsize × varCost ×
    # step_duration × timestep_weight × inflation / period_share ×
    # pdt_branch_weight.
    #
    # The mod splits the ``other_operational_cost`` contribution into
    # three sums keyed by ``pssdt_varCost_noEff``,
    # ``pssdt_varCost_eff_unit_source``/``_sink``, and
    # ``_eff_connection``.  For ``min_load_efficiency`` processes the
    # source-side sum also contains a ``section × online × unitsize``
    # term (flextool.mod line 2358-2364).  In Python this term is
    # folded into ``r.flow_dt`` on the source-side (p, source, p)
    # tuple by ``calc_capacity_flows.compute_capacity_and_flows``
    # (line 58-59): ``flow_val.add(online × section × unitsize)`` when
    # the process uses ``min_load_efficiency``.  Multiplying that
    # pre-aggregated source-side flow by the per-(p, source, sink)
    # varCost here therefore reproduces both the slope and the section
    # contributions in a single matmul, matching the objective.  See
    # ``test_cost_aggregation_semantics.TestMinLoadEfficiencySectionTerm``
    # for the regression guard.
    relevant_flows = r.flow_dt.loc[:, r.flow_dt.columns.intersection(par.process_source_sink_varCost.columns)]
    cost_flows = relevant_flows.mul(step_x_rp, axis=0).mul(par.process_source_sink_varCost, axis=1)
    r.cost_process_other_operational_cost_dt = cost_flows.T.groupby(level=0).sum().T.reindex(columns=s.process, fill_value=0.0)

    # --- Startup costs ---
    # Mod objective (~line 2337-2346): v_startup × startup_cost × unitsize
    # × timestep_weight × inflation / period_share × pdt_branch_weight.
    # No step_duration: startup is a per-event (not per-hour) cost.
    r.process_startup_dt = v.startup_linear.add(v.startup_integer, fill_value=0)
    r.cost_startup_dt = pd.DataFrame(0.0, index=r.process_startup_dt.index, columns=s.process_online, dtype=float)
    valid_processes = s.process_online.intersection(r.process_startup_dt.columns).intersection(par.process_startup_cost.columns)
    if len(valid_processes) > 0:
        cost = r.process_startup_dt[valid_processes].mul(par.entity_unitsize[valid_processes], axis=1)
        periods = cost.index.get_level_values('period')
        period_costs = par.process_startup_cost.loc[periods, valid_processes]
        period_costs.index = cost.index
        r.cost_startup_dt[valid_processes] = cost.mul(period_costs).mul(par.timestep_weight, axis=0)
    # Option (A): weight the startup *cost* by pdt_branch_weight (the LP's
    # startup_factor carries it — model/core.py §startup).  The physical start
    # count (r.process_startup_dt above) stays UNWEIGHTED.
    r.cost_startup_dt = weight_dt_by_branch(r.cost_startup_dt, _pdt_bw)

    # --- Investment costs ---
    r.cost_entity_invest_d = v.invest.mul(par.entity_unitsize[v.invest.columns]).mul(par.entity_annual_discounted)
    r.cost_entity_divest_d = -v.divest.mul(par.entity_unitsize[v.divest.columns]).mul(par.entity_annual_divest_discounted)
    # Use p_entity_all_existing (= pre_existing + carry-forward from earlier
    # solves − divested) to match the mod objective term at
    # flextool.mod:2395-2398.  Differs from par.entity_pre_existing only for
    # rolling / nested solves where prior-solve v_invest has aged into
    # current-solve existing capacity.
    r.cost_entity_fixed_pre_existing = (par.entity_all_existing * par.entity_fixed_cost).mul(par.inflation_factor_operations_yearly, axis=0)
    # entity_lifetime_fixed_cost* are densified over _entity_universe and
    # solve-keyed (concat-unioned), so they cover every v.invest/v.divest
    # column — direct indexing is safe.
    r.cost_entity_fixed_invested = (v.invest.mul(par.entity_unitsize[v.invest.columns] * par.entity_lifetime_fixed_cost[v.invest.columns]))
    r.cost_entity_fixed_divested = -(v.divest.mul(par.entity_unitsize[v.divest.columns] * par.entity_lifetime_fixed_cost_divest[v.divest.columns]))

    # --- Slice D (I-a, §11.3): probability-weight the invest/divest cost ---
    # Under ``recourse`` the LP objective weights every invest/divest
    # annuity + lifetime-fixed contribution by ``pd_branch_weight[period]``
    # (§9.1).  The post-hoc recompute above uses the raw annuity, so the
    # reported cost would over-state the realized invest by ``1/w_R``.
    # Multiply each invest/divest cost row by its period's branch weight so
    # the cost breakdown reconciles to the reported (expected) objective.
    # Gated on the recourse flag — absent / off → factor 1.0 (byte-parity).
    if getattr(par, "recourse_invest", False):
        _wser = getattr(par, "pd_branch_weight", None)
        _wmap: dict[str, float] = {}
        if _wser is not None:
            for _idx, _val in _wser.items():
                _period = _idx[-1] if isinstance(_idx, tuple) else _idx
                _wmap[str(_period)] = float(_val)

        def _weight_by_branch(df):
            if df is None or df.empty:
                return df
            _periods = df.index.get_level_values('period')
            _factors = np.array([_wmap.get(str(_p), 1.0) for _p in _periods])
            return df.mul(_factors, axis=0)

        r.cost_entity_invest_d = _weight_by_branch(r.cost_entity_invest_d)
        r.cost_entity_divest_d = _weight_by_branch(r.cost_entity_divest_d)
        r.cost_entity_fixed_invested = _weight_by_branch(
            r.cost_entity_fixed_invested)
        r.cost_entity_fixed_divested = _weight_by_branch(
            r.cost_entity_fixed_divested)

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
                                .mul(par.inflation_factor_operations_yearly, axis=0) \
                                .div(par.complete_period_share_of_year, axis=0)
    r.costPenalty_d = r.costPenalty_dt.groupby('period').sum() \
                                .mul(par.inflation_factor_operations_yearly, axis=0) \
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
