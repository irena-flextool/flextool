import numpy as np
import pandas as pd

from flextool.process_outputs._annualize import annualize_dt_to_d
from flextool.process_outputs.calc_capacity_flows import flows_with_startup


def _agg_period_or_empty(df: pd.DataFrame, realized_periods, period_level: str = 'period') -> pd.DataFrame:
    """Group-by period sum, or an all-zeros frame when df is empty."""
    if df.empty:
        cols = df.columns if hasattr(df, 'columns') else []
        return pd.DataFrame(0.0, index=realized_periods, columns=cols)
    return df[df.index.get_level_values(period_level).isin(realized_periods)].groupby(period_level).sum()


def _reported_start_periods(s) -> pd.Index:
    """The periods in which the start of operation of new capacity has its investment costs reported: the
    realized periods, and the period labels of the stochastic fans (inside a fan the investment costs are
    reported, weighted by the branch weights, also without a solve that realizes the fan's periods)."""
    return s.d_realized_period.union(s.d_realize_invest).union(s.fan_label_period)


def _invest_with_fan(v) -> pd.DataFrame:
    """The investment of the reported costs (rows: decision period, columns: entity):

    invest_cost[e,d] = v_invest[e,d] + invest_fan[e,d]
      invest_fan[e,L] = Σ_b pd_branch_weight[b]·v_invest[e,b] over the branch periods b of a fan label L
      (``v_invest_fan``, recourse); 0 elsewhere.  A shared fan investment is the label's own v_invest (its
      branch weights sum to 1); under recourse each branch's own investment carries its branch weight."""
    fan = v.invest_fan
    if fan.empty or len(fan.columns) == 0:
        return v.invest
    index = v.invest.index.union(fan.index, sort=False)
    out = v.invest.reindex(index=index).add(fan.reindex(index=index, columns=v.invest.columns), fill_value=0.0)
    out.index.name = v.invest.index.name
    out.columns.name = v.invest.columns.name
    return out


def _at_start_of_operation(cost: pd.DataFrame, s) -> pd.DataFrame:
    """The cost of new capacity (rows: decision period, columns: entity) moved to the period the capacity starts
    to operate, c(d) (``s.edd_invest_commission``; c(d) = d without a construction lead time), and kept only
    where c(d) is realized or a fan label (:func:`_reported_start_periods`): an investment's cost is reported
    once, by the solve that realizes the period its capacity starts operating in, or in the fan it starts
    operating in."""
    commission = s.edd_invest_commission
    realized = _reported_start_periods(s)
    if cost.empty or (len(commission) == 0 and cost.index.isin(realized).all()):
        return cost
    long = cost.stack(future_stack=True).rename('value').rename_axis(['period_invest', 'entity']).reset_index()
    if len(commission) > 0:
        start = commission.to_frame(index=False).rename(columns={'period': 'start'})
        long = long.merge(start[['entity', 'period_invest', 'start']], on=['entity', 'period_invest'], how='left')
        long['start'] = long['start'].fillna(long['period_invest'])
    else:
        long['start'] = long['period_invest']
    long = long[long['start'].isin(realized)]
    moved = long.groupby(['start', 'entity'], sort=False)['value'].sum(min_count=1).unstack('entity')
    order = [d for d in dict.fromkeys(list(cost.index) + list(realized)) if d in moved.index]
    moved = moved.reindex(index=order, columns=cost.columns)
    moved.index.name = cost.index.name
    moved.columns.name = cost.columns.name
    return moved


def _decisions_long(amount: pd.DataFrame, decision: str) -> pd.DataFrame:
    """``[<decision>, entity, value]`` of a (decision period × entity) frame, without empty cells."""
    long = amount.stack(future_stack=True).rename('value').rename_axis([decision, 'entity']).reset_index()
    return long[long['value'].notna()]


def _per_year_in_window(long: pd.DataFrame, window: pd.MultiIndex, decision: str, columns: pd.Index) -> pd.DataFrame:
    """Rows ``[<decision>, entity, value]`` (a yearly amount of a decision) summed per (period, entity) over the
    periods of each decision's window (``window``: (entity, <decision>, period)): the yearly amounts of every
    decision in effect in a period.  Rows: period, columns: ``columns``."""
    if long.empty or len(window) == 0:
        out = pd.DataFrame(index=pd.Index([], name='period'), columns=columns, dtype=float)
        out.columns.name = columns.name
        return out
    win = window.to_frame(index=False)
    win = win.astype({'entity': str, decision: str, 'period': str})
    long = long.astype({'entity': str, decision: str})
    joined = long.merge(win, on=['entity', decision], how='inner')
    out = joined.groupby(['period', 'entity'], sort=False)['value'].sum().unstack('entity')
    order = list(dict.fromkeys(win['period']))
    out = out.reindex(index=[d for d in order if d in out.index], columns=columns)
    out.index.name = 'period'
    out.columns.name = columns.name
    return out


def _annualized_investment(par, s, v) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(investment, fixed cost invested)`` per year, rows: period, columns: entity (CUR/yr).

    ANN. invest_yr[e,d'] = Σ_{(e,d,d') ∈ edd_operating_window} v_invest[e,d]·unitsize[e]·ed_annuity[e,d]
         fixed_invested_yr[e,d'] = Σ_{(e,d,d') ∈ edd_operating_window} v_invest[e,d]·unitsize[e]·fc[e,c(d)]
      The yearly amounts of every vintage that operates in d'; ed_annuity[e,d] is the annuity at c(d), fc[e,c(d)]
      the fixed cost of the lifetime fixed-cost lump.  Only the decisions whose lump is reported (c(d) realized,
      :func:`_at_start_of_operation`)."""
    columns = v.invest.columns
    mw = _invest_with_fan(v).mul(par.entity_unitsize[columns])
    long = _decisions_long(mw, 'period_invest')
    commission = s.edd_invest_commission
    if len(commission) > 0 and not long.empty:
        start = commission.to_frame(index=False).rename(columns={'period': 'start'})
        long = long.merge(start[['entity', 'period_invest', 'start']], on=['entity', 'period_invest'], how='left')
        long['start'] = long['start'].fillna(long['period_invest'])
    else:
        long = long.assign(start=long['period_invest'])
    long = long[long['start'].isin(_reported_start_periods(s))]
    annuity = _decisions_long(par.entity_annuity_yearly, 'period_invest').rename(columns={'value': 'coef'})
    fixed = _decisions_long(par.entity_fixed_cost, 'start').rename(columns={'value': 'coef'})
    invest = long.merge(annuity, on=['period_invest', 'entity'], how='inner')
    invest = invest.assign(value=invest['value'] * invest['coef'])[['period_invest', 'entity', 'value']]
    fixed = long.merge(fixed, on=['start', 'entity'], how='inner')
    fixed = fixed.assign(value=fixed['value'] * fixed['coef'])[['period_invest', 'entity', 'value']]
    window = s.edd_operating_window
    return (_per_year_in_window(invest, window, 'period_invest', columns),
            _per_year_in_window(fixed, window, 'period_invest', columns))


def _annualized_retirement(par, s, v) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(retirement, fixed cost reduction of divestments)`` per year, rows: period, columns: entity (CUR/yr).

    ANN. retire_yr[e,d'] = −Σ_{(e,d,d') ∈ edd_salvage_window} v_divest[e,d]·unitsize[e]·ed_divest_annuity[e,d]
         fixed_divested_yr[e,d'] = −Σ_{(e,d,d') ∈ edd_retire_window} v_divest[e,d]·unitsize[e]·fc[e,d']
      The yearly amounts of every retirement in d': the salvage annuity over the window it is spread over (the
      retirement period until lifetime later), and the fixed cost of d' the retired capacity no longer pays (from
      the retirement period until the end of life of the existing capacity).  fc[e,d'] falls back to fc[e,d] of
      the retirement period for a window period the solve does not hold; such rows are never reported (the chain
      union keeps each period from the solve that realizes it, which holds its fixed cost)."""
    columns = v.divest.columns
    long = _decisions_long(-v.divest.mul(par.entity_unitsize[columns]), 'period_divest')
    annuity = _decisions_long(par.entity_divest_annuity_yearly, 'period_divest').rename(columns={'value': 'coef'})
    joined = long.merge(annuity, on=['period_divest', 'entity'], how='inner')
    joined = joined.assign(value=joined['value'] * joined['coef'])[['period_divest', 'entity', 'value']]
    retirement = _per_year_in_window(joined, s.edd_salvage_window, 'period_divest', columns)
    window = s.edd_retire_window
    if long.empty or len(window) == 0:
        return retirement, _per_year_in_window(long.iloc[0:0], window, 'period_divest', columns)
    fc = _decisions_long(par.entity_fixed_cost, 'period').astype({'entity': str, 'period': str})
    win = window.to_frame(index=False).astype({'entity': str, 'period_divest': str, 'period': str})
    per = long.astype({'entity': str, 'period_divest': str}).merge(win, on=['entity', 'period_divest'], how='inner')
    per = per.merge(fc.rename(columns={'value': 'fc'}), on=['period', 'entity'], how='left')
    per = per.merge(fc.rename(columns={'period': 'period_divest', 'value': 'fc_r'}), on=['period_divest', 'entity'],
                    how='left')
    per = per.assign(value=per['value'] * per['fc'].fillna(per['fc_r']))
    if per.empty:
        return retirement, _per_year_in_window(long.iloc[0:0], window, 'period_divest', columns)
    fixed = per.groupby(['period', 'entity'], sort=False)['value'].sum().unstack('entity')
    order = list(dict.fromkeys(win['period']))
    fixed = fixed.reindex(index=[d for d in order if d in fixed.index], columns=columns)
    fixed.index.name = 'period'
    fixed.columns.name = columns.name
    return retirement, fixed


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
    # The energy taken from a commodity node: the flows and the start-up input draws (startup_input_method), as
    # the LP's commodity_cost_* terms; a draw is MWh per start-up, its MW-average × step_duration (as a flow).
    taken = flows_with_startup(r)
    flow_from_commodity_node = taken[taken.columns[taken.columns.get_level_values(level=1).isin(commodity_node.get_level_values(level=1))]]
    flow_to_commodity_node = r.flow_dt[r.flow_dt.columns[r.flow_dt.columns.get_level_values(level=2).isin(commodity_node.get_level_values(level=1))]]

    # Per-step scaling factor: step_duration × timestep_weight.  Used by all
    # per-(d, t) cost terms below to match the objective in flextool.mod.
    # The LP also weights these terms by pdt_branch_weight[d, t]; normal outputs hold realized rows only, and a
    # realized (d, t) has branch weight 1, so no branch weight is applied here or below.
    step_x_rp = par.step_duration.mul(par.timestep_weight, axis=0)

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

        ladder_cost_d = mwh.mul(price)
        ladder_cost_d.columns = tiers
        ladder_cost_d = ladder_cost_d.T.groupby('commodity').sum().T

        # Spread each period's ladder cost across its (d, t) rows proportionally to step_x_rp (a pure
        # step_duration x timestep_weight share), which reconciles to ladder_cost_d exactly on re-summing.
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
    # Emissions of the energy taken from a CO2 node: the flows and the start-up input draws (co2_price_startup,
    # the start-up terms of the CO2 caps).
    flow_outof_cols = taken.columns.copy()
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

    flow_outof_node = taken[flow_outof_cols.droplevel('commodity')]
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
    r.group_cost_co2_dt = r.group_co2_dt.mul(par.group_co2_price).mul(par.timestep_weight, axis=0)
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

    # --- Ramp costs ---
    # The LP objective (model/core.py _obj_ramp_cost): (v_ramp_up + v_ramp_down) × unitsize × ramp_cost
    # × timestep_weight × inflation / period_share × pdt_branch_weight.  No step_duration: a change of the flow is
    # a per-event cost, as a start-up.  Up + down, never |up − down| (the objective's value, also when an
    # interior-point solution leaves both slightly > 0).  A coarse process's change sits at the first fine step of
    # its coarse step (0 at the others).  Columns: the processes with a ramp cost.
    ramp_cost = par.process_ramp_cost
    ramp_procs = pd.Index(ramp_cost.columns.get_level_values(0).unique(), name='process')
    r.cost_ramp_dt = pd.DataFrame(0.0, index=r.flow_dt.index, columns=ramp_procs, dtype=float)
    ramp = v.ramp_up.add(v.ramp_down, fill_value=0)
    ramp_arcs = ramp.columns.intersection(ramp_cost.columns) if len(ramp.columns) else ramp_cost.columns[:0]
    if len(ramp_arcs) > 0:
        cost = ramp[ramp_arcs].mul(
            par.entity_unitsize.reindex(ramp_arcs.get_level_values(0)).to_numpy(), axis=1)
        periods = cost.index.get_level_values('period')
        arc_costs = ramp_cost.loc[periods, ramp_arcs]
        arc_costs.index = cost.index
        cost = cost.mul(arc_costs).mul(par.timestep_weight, axis=0)
        r.cost_ramp_dt = (cost.T.groupby(level=0).sum().T
                          .reindex(index=r.flow_dt.index, columns=ramp_procs, fill_value=0.0))

    # --- Investment costs ---
    # The lumps are discounted from the start of the run over the run's periods; the investment lump and the
    # lifetime fixed cost of new capacity are reported in the period the capacity starts to operate, the
    # retirement lumps in the retirement period.
    invest = _invest_with_fan(v)
    r.cost_entity_invest_d = _at_start_of_operation(
        invest.mul(par.entity_unitsize[v.invest.columns]).mul(par.entity_annual_discounted), s)
    r.cost_entity_divest_d = -v.divest.mul(par.entity_unitsize[v.divest.columns]).mul(par.entity_annual_divest_discounted)
    # The fixed cost of existing capacity: the user's existing capacity aged over its lifetime, in every period of
    # the run.  The capacity carried from earlier solves is in the fixed-cost lump of its investment and a
    # retirement in the lump of the retirement, so neither is charged here (a single solve: the same as
    # p_entity_all_existing, the mod objective term at flextool.mod:2395-2398).
    r.cost_entity_fixed_pre_existing = (par.entity_user_existing * par.entity_fixed_cost).mul(
        par.inflation_factor_operations_yearly, axis=0)
    # entity_lifetime_fixed_cost* are densified over _entity_universe and
    # solve-keyed (concat-unioned), so they cover every v.invest/v.divest
    # column — direct indexing is safe.
    r.cost_entity_fixed_invested = _at_start_of_operation(
        invest.mul(par.entity_unitsize[v.invest.columns] * par.entity_lifetime_fixed_cost[v.invest.columns]), s)
    r.cost_entity_fixed_divested = -(v.divest.mul(par.entity_unitsize[v.divest.columns] * par.entity_lifetime_fixed_cost_divest[v.divest.columns]))

    # The annualized investment, retirement and fixed costs of new / retired capacity: the yearly amounts of
    # every vintage and retirement in effect in a period (not the lumps divided by a period's factor).
    r.cost_entity_invest_yearly, r.cost_entity_fixed_invested_yearly = _annualized_investment(par, s, v)
    r.cost_entity_divest_yearly, r.cost_entity_fixed_divested_yearly = _annualized_retirement(par, s, v)

    # Under ``recourse`` the LP weights the invest annuity and lifetime-fixed terms by pd_branch_weight[period]:
    # the fan's investment is the branch-weighted v_invest_fan at the label (_invest_with_fan), whose lumps are
    # the label's (a branch period takes its label's years).  Retirements in the branch periods of a recourse fan
    # are not reported (normal outputs hold no forecast-branch retirement).

    # --- Aggregate operational costs (depends on r.costPenalty_* from compute_slacks) ---
    r.costOper_dt = (r.cost_commodity_dt.sum(axis=1) -
                     r.sales_commodity_dt.sum(axis=1) +
                     r.cost_process_other_operational_cost_dt.sum(axis=1) +
                     r.cost_startup_dt.sum(axis=1) +
                     r.cost_ramp_dt.sum(axis=1) +
                     r.cost_co2_dt
                     )
    r.costPenalty_dt = (r.costPenalty_node_state_upDown_dt.sum(axis=1) +
                        r.costPenalty_inertia_dt.sum(axis=1) +
                        r.costPenalty_non_synchronous_dt.sum(axis=1) +
                        r.costPenalty_reserve_upDown_dt.sum(axis=1))

    r.cost_process_other_operational_cost_d = r.cost_process_other_operational_cost_dt.groupby('period').sum()
    r.cost_variable_d = r.cost_process_other_operational_cost_d.sum(axis=1)

    r.cost_startup_d = _agg_period_or_empty(r.cost_startup_dt, s.d_realized_period)
    r.cost_ramp_d = _agg_period_or_empty(r.cost_ramp_dt, s.d_realized_period)
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

    def _class_sum(frame: pd.DataFrame, entities) -> pd.Series:
        return frame[[e for e in entities if e in frame.columns]].sum(axis=1)

    for kind, entities in (('Unit', s.process_unit), ('Connection', s.process_connection), ('State', s.node_state)):
        setattr(r, f'costInvest{kind}_yearly', _class_sum(r.cost_entity_invest_yearly, entities))
        setattr(r, f'costDivest{kind}_yearly', _class_sum(r.cost_entity_divest_yearly, entities))
    r.costFixedInvested_yearly = r.cost_entity_fixed_invested_yearly.sum(axis=1)
    r.costFixedDivested_yearly = r.cost_entity_fixed_divested_yearly.sum(axis=1)
