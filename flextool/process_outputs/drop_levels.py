from types import SimpleNamespace

import pandas as pd


# Variables: droplevel('solve') only
_V_DROP = [
    'flow', 'ramp_up', 'ramp_down', 'reserve', 'state', 'self_discharge',
    'online_linear', 'startup_linear', 'shutdown_linear',
    'online_integer', 'startup_integer', 'shutdown_integer',
    'q_state_up', 'q_state_down', 'q_reserve',
    'q_inertia', 'q_non_synchronous', 'q_state_up_group', 'q_capacity_margin',
    'invest', 'divest', 'invest_fan',
    # trade (v_trade, commodity price ladder, #57): period-only, same
    # rolling treatment as invest/divest — the committing roll's tier
    # purchase for a period wins over an earlier look-ahead roll's.
    'trade',
    'dual_invest_connection', 'dual_invest_node', 'dual_invest_unit',
    'dual_maxInvest_period', 'dual_maxCumulative',
    'dual_maxInvestGroup_period', 'dual_maxInvestGroup_cumulative',
    # Investment-floor (min-side) duals — keep a ('solve', 'period') index,
    # so every family drops the solve level here rather than collapsing to
    # solve-only.  ``dual_minInvest_total`` / ``dual_minInvestGroup_total``
    # moved to ``_V_SOLVE_ONLY`` below — the rows they mirror
    # (``min{,InvestGroup_}entity_total_path_{p,n}``) are now one row per
    # entity/group over the whole chain horizon (period summed out), like
    # the max-side totals, not per (entity/group, period).
    'dual_minInvest_period', 'dual_minCumulative',
    'dual_minInvestGroup_period', 'dual_minInvestGroup_cumulative',
    # dual_co2_max_period moved to its own hour-weighted reduction below
    # (Q22) -- a period that straddles several rolling windows needs an
    # average over the rolls that realize it, not "last roll wins".
]

# Variables with a solve-only row index — must collapse to a single
# row (last solve wins) instead of droplevel.  Same pattern as
# ``dual_co2_max_total`` below.
_V_SOLVE_ONLY = [
    'dual_maxInvest_total',
    # min{,InvestGroup_}entity_total_path_{p,n} are now one row per
    # entity/group over the chain horizon (no period axis) — see the
    # comment above.  dual_maxInvestGroup_total moves here too (same
    # chain-horizon reshape, max side).
    'dual_minInvest_total',
    'dual_maxInvestGroup_total', 'dual_minInvestGroup_total',
]

# Parameters: droplevel('solve') + deduplicate with keep='last'.
#
# Use this bucket for params whose per-roll values represent successive
# states of the same (entity, period) (e.g. ``entity_all_existing``,
# which is a chain-sum that gets updated as later rolls realise their
# investments — the LAST roll covering a given period carries the
# post-decision value).
_PAR_DROP = [
    'step_duration', 'timestep_weight', 'flow_min', 'flow_max',
    'process_availability', 'process_source_sink_varCost',
    'process_slope', 'process_section',
    'node_self_discharge_loss', 'node_penalty_up', 'node_penalty_down',
    'node_inflow', 'commodity_price', 'group_co2_price', 'group_inertia_limit',
    # ladder_price_ann (commodity price ladder, #57): authored per period, invariant across rolls that
    # touch the same period — same keep='last' bucket as commodity_price for symmetry.
    'ladder_price_ann',
    'reserve_upDown_group_reservation',
    'reserve_upDown_group_shortfall_scale', 'profile',
    # The invest / divest lumps come from the solve that realizes the
    # decision period only (persist_realized_slice); keep='last' ties the
    # rolls of one solve.
    'entity_annual_discounted', 'entity_annual_divest_discounted',
    'entity_annuity_yearly', 'entity_divest_annuity_yearly',
    # The reported costs' discount factors count from the start of the run,
    # so every solve holding a period gives it the same value.
    'inflation_factor_investment_yearly',
    'inflation_factor_operations_yearly',
    # The solve's own operations factor (CO2 price): only the solves that
    # realize the period's dispatch persist it, so the realizing solve's
    # value is the one kept (its rolls carry the same factor).
    'inflation_factor_operations_solve',
    'group_penalty_capacity_margin', 'group_capacity_margin',
    # entity_all_existing carries the cumulative existing-capacity
    # chain (existing + Σ prior invest − Σ prior divest).  The last roll
    # covering a given period writes the post-decision value; the
    # earlier rolls' rows for the same period carry the pre-decision
    # value.  keep='last' picks the post-decision row to match v3.32.0.
    'entity_all_existing',
    # entity_pre_existing is the static baseline before any invest
    # decision.  In practice constant across rolls for a given
    # (entity, period), so either keep choice yields the same value —
    # moved here for symmetry with entity_all_existing per the
    # 2026-05-14 investigation into the unit_capacity__d.csv lag.
    'entity_pre_existing',
    # entity_user_existing — the user's existing capacity aged over its
    # lifetime, the same in every solve (the fixed cost of existing capacity).
    'entity_user_existing',
    # NOTE: complete_period_share_of_year is intentionally NOT in this
    # bucket — it needs an order-INDEPENDENT max-per-period reduction,
    # done explicitly in drop_levels() below.  See the comment there.
]

# Parameters: droplevel('solve') + deduplicate with keep='first'.
#
# Use this bucket ONLY for params that are invariant across rolls (so
# either keep choice would yield the same result).  Do NOT use this
# bucket for params whose value evolves across rolls — see _PAR_DROP
# above and the comment there.
_PAR_DEDUP = [
    'years_from_start_d', 'years_represented_d',
    'entity_max_units',
    'process_startup_cost',
    'process_ramp_cost',
    'entity_fixed_cost', 'entity_lifetime_fixed_cost', 'entity_lifetime_fixed_cost_divest',
    'node_annual_flow',
    'group_penalty_inertia', 'group_penalty_non_synchronous',
    'node_capacity_for_scaling', 'group_capacity_for_scaling',
]


def _reduce_share_per_period(csoy: pd.Series) -> pd.Series:
    """Collapse ``complete_period_share_of_year`` to one value per period.

    ``complete_period_share_of_year`` is the per-period annualisation
    denominator for DISPATCH outputs (energy/cost = Σ power·dur ÷ share).
    It is NOT invariant across sub-solves: an invest sub-solve whose
    period samples a representative window (e.g. 3×24 h) carries
    ``share = 72/8760 ≈ 0.0082``, while the dispatch sub-solve that
    realises the same period over the full year carries ``share = 1``.

    BOTH sub-solves persist a row for the realised period, so a plain
    ``keep='first'/'last'`` dedup is **order-dependent** — and the multi-
    solve union order is NOT guaranteed: when the ``_solve_order.txt``
    creation manifest is absent, ``union_realized_slice`` falls back to
    filesystem glob order, which can place the invest row last and pick
    the 0.0082 window share, inflating every period total by
    ``8760/(sampled hours)`` (≈121.7× for a 3-day window).  This was the
    cause of an intermittent multi-solve period/aggregate over-count that
    appeared on some scenarios/runs but not others.

    The dispatch sub-solve covers the full period, so its share is always
    ≥ any sampling sub-solve's for the same period.  Reducing by **max per
    period** therefore deterministically selects the realising-dispatch
    share regardless of union order, and is byte-identical to the correct
    (manifest-present) ``keep='last'`` runs.  ``sort=False`` preserves the
    first-appearance period order those runs produced.
    """
    csoy = csoy.droplevel('solve')
    return csoy.groupby(level=0, sort=False).max()


# Investment-axis variables (``period_source`` invest / invest_branches in read_highs_solution): a period's value
# comes from the solve that realizes the investment there (``d_realize_invest``), see ``_keep_realizing_solve_rows``.
# The investment duals too (item 266): the v_invest reduced costs and the per-period cap / floor duals.
_V_REALIZING_SOLVE = frozenset({
    'invest', 'divest', 'q_capacity_margin',
    'dual_invest_connection', 'dual_invest_node', 'dual_invest_unit',
    'dual_maxInvest_period', 'dual_maxCumulative', 'dual_maxInvestGroup_period', 'dual_maxInvestGroup_cumulative',
    'dual_minInvest_period', 'dual_minCumulative', 'dual_minInvestGroup_period', 'dual_minInvestGroup_cumulative',
})


def _keep_realizing_solve_rows(obj: pd.DataFrame, d_realize_invest: pd.MultiIndex) -> pd.DataFrame:
    """The rows of the solves that realize the investment of their period, for a period some solve realizes.

    A solve writes a ``v_invest`` / ``v_divest`` / ``vq_capacity_margin`` row (and an investment dual row) for
    every period on its dispatch axis, also a 0 row for a period whose investment an earlier solve realized
    (``realized_invest_periods`` of that solve, outside its ``realized_periods``).  The ``keep='last'`` dedup that
    follows would let that later 0 row win, so the rows of a period realized by some solve are cut to the
    realizing solves' rows first (their last one still wins).  A period no solve realizes keeps all its rows (as
    before).  ``d_realize_invest`` is still solve-indexed here.
    """
    if obj.empty or len(d_realize_invest) == 0:
        return obj

    def solve_period(idx: pd.MultiIndex) -> pd.MultiIndex:
        return pd.MultiIndex.from_arrays(
            [idx.get_level_values('solve').astype(str), idx.get_level_values('period').astype(str)])

    realizing = solve_period(d_realize_invest)
    rows = solve_period(obj.index)
    periods = rows.get_level_values(1)
    keep = ~periods.isin(realizing.get_level_values(1)) | rows.isin(realizing)
    return obj[keep]


def _period_level(idx: pd.Index) -> "pd.Index | None":
    if isinstance(idx, pd.MultiIndex):
        return idx.get_level_values('period') if 'period' in idx.names else None
    return idx if idx.name == 'period' else None


def _drop_unrealized_forecast_periods(par: SimpleNamespace, s: SimpleNamespace, v: SimpleNamespace) -> None:
    """Remove the rows of unrealized forecast-branch periods from every output input.

    A stochastic solve's forecast-branch periods (``s.forecast_period``) are never part of the
    realized timeline unless the solve realizes them (``s.d_realize_dispatch_or_invest``), yet the
    parameters, sets and dual variables of the solve carry rows for them.  Normal outputs hold realized
    rows only, so those rows go here, before any calculation: every ``Series`` / ``DataFrame`` /
    ``Index`` of ``par``, ``s`` and ``v`` with a ``period`` index level is cut with one ``isin``.
    No-op for a solve without forecasts.
    """
    forecast = getattr(s, 'forecast_period', None)
    if forecast is None or len(forecast) == 0:
        return
    realized = set(_period_level(s.d_realize_dispatch_or_invest)) | set(_period_level(s.d_realized_period))
    drop = [p for p in dict.fromkeys(_period_level(forecast)) if p not in realized]
    if not drop:
        return
    for ns in (par, s, v):
        for attr, obj in list(vars(ns).items()):
            if attr == 'forecast_period':
                continue
            if isinstance(obj, (pd.Series, pd.DataFrame)):
                periods = _period_level(obj.index)
            elif isinstance(obj, pd.Index):
                periods = _period_level(obj)
            else:
                continue
            if periods is None:
                continue
            mask = periods.isin(drop)
            if mask.any():
                setattr(ns, attr, obj[~mask])


def _co2_period_price_weighted(
    v_dual: pd.DataFrame, step_duration: pd.Series, dt_realize_dispatch: pd.MultiIndex,
) -> pd.DataFrame:
    """Hour-weighted average of ``dual_co2_max_period`` across rolls (Q22).

    A period that straddles several rolling windows (the roll boundary
    falls inside the period, not at its edge) gets one dual per roll that
    touches it -- each roll's own sub-LP shadow price for its own slice of
    the period.  ``keep='last'`` (the plain ``_V_DROP`` treatment every
    other roll-overlapping dual gets) would report only the LAST roll's
    price and silently drop every earlier roll's realized hours. Q22
    instead weights each roll's price by the hours THAT roll keeps
    (realizes) of the period, and averages.

    Weight[solve, period] = Σ step_duration[solve, period, t] over the
    (period, t) pairs ``dt_realize_dispatch`` marks realized for that
    solve.  A roll that only sees period d as look-ahead (never commits a
    result for it) contributes zero weight and is excluded entirely.

    Deterministic / single-roll solves have exactly one (solve, period)
    row per period, so the weight cancels and this is a no-op -- byte-
    identical to the prior ``keep='last'`` dedup.
    """
    has_solve = 'solve' in v_dual.index.names
    if v_dual.empty or not has_solve:
        return v_dual.droplevel('solve') if has_solve else v_dual
    sd_realized = step_duration[step_duration.index.isin(dt_realize_dispatch)]
    weight = sd_realized.groupby(level=['solve', 'period']).sum()
    weight = weight.reindex(v_dual.index, fill_value=0.0)
    weighted = v_dual.mul(weight, axis=0)
    num = weighted.groupby(level='period').sum()
    den = weight.groupby(level='period').sum()
    out = num.div(den, axis=0)
    # Defensive: a period with zero total realized weight across every
    # roll that touches it shouldn't happen for a genuinely realized
    # period, but fall back to a plain mean rather than a silent 0/0 NaN.
    zero_den = den.index[den == 0.0]
    if len(zero_den) > 0:
        plain = v_dual.groupby(level='period').mean()
        out.loc[zero_den] = plain.loc[zero_den]
    return out


def drop_levels(par: SimpleNamespace, s: SimpleNamespace, v: SimpleNamespace):
    """Strip the 'solve' level from all time-indexed variables, parameters and sets.

    Must be called before any post-processing calculations.
    Returns (par, s, v) with the solve level removed in-place.
    """
    _drop_unrealized_forecast_periods(par, s, v)
    for attr in _V_DROP:
        obj = getattr(v, attr)
        if attr in _V_REALIZING_SOLVE:
            obj = _keep_realizing_solve_rows(obj, s.d_realize_invest)
        obj = obj.droplevel('solve')
        # For rolling scenarios, keep only last solve's value per (period, time)
        if obj.index.duplicated().any():
            obj = obj[~obj.index.duplicated(keep='last')]
        setattr(v, attr, obj)

    # dual_co2_max_period (Q22): hour-weighted average across rolls instead
    # of the plain keep='last' dedup above -- see _co2_period_price_weighted.
    # par.step_duration / s.dt_realize_dispatch are still solve-indexed here
    # (their own drops happen further down).
    v.dual_co2_max_period = _co2_period_price_weighted(
        v.dual_co2_max_period, par.step_duration, s.dt_realize_dispatch,
    )

    # Solve-only indexes (``dual_co2_max_total`` historically, plus
    # ``dual_maxInvest_total`` which became solve-only when its reader
    # arity was fixed, and ``dual_minInvest_total`` since they became one row per entity): collapse
    # to a single row (last solve wins).
    for attr in [*_V_SOLVE_ONLY, 'dual_co2_max_total']:
        obj = getattr(v, attr)
        if not obj.empty:
            last = obj.iloc[[-1]].copy()
            last.index = pd.RangeIndex(1)
            setattr(v, attr, last)
        else:
            setattr(v, attr, obj.reset_index(drop=True))

    # v_angle: drop solve level only when non-empty (may be empty when no DC PF nodes)
    if not v.angle.empty:
        v.angle = v.angle.droplevel('solve')
        if v.angle.index.duplicated().any():
            v.angle = v.angle[~v.angle.index.duplicated(keep='last')]

    for attr in _PAR_DROP:
        obj = getattr(par, attr)
        obj = obj.droplevel('solve')
        if obj.index.duplicated().any():
            obj = obj[~obj.index.duplicated(keep='last')]
        setattr(par, attr, obj)

    for attr in _PAR_DEDUP:
        obj = getattr(par, attr)
        obj = obj.droplevel('solve')
        setattr(par, attr, obj[~obj.index.duplicated(keep='first')])

    # process_node_startup_energy is the same in every solve that has it; a solve whose reduce_unit_detail_method
    # left out the start-ups has no value (NaN): the first value per period and (process, node).
    se = getattr(par, 'process_node_startup_energy', None)
    if se is not None and 'solve' in (se.index.names or ()):
        se = se.droplevel('solve')
        if not se.empty:
            se = se.groupby(level='period', sort=False).first()
        par.process_node_startup_energy = se

    # complete_period_share_of_year — order-independent max-per-period
    # reduction (deterministic across multi-solve union order).  See
    # _reduce_share_per_period for the full rationale.
    par.complete_period_share_of_year = _reduce_share_per_period(
        par.complete_period_share_of_year
    )

    # Sets have varied special handling so are done individually
    s.solve_period = s.d_realized_period
    # (solve, period) of the realized investments, for the solve label of the investment tables (#210).
    s.solve_invest_period = s.d_realize_invest
    # Save per-timestep solve mapping before dropping (for correct re-join in CSV output)
    # For rolling scenarios, keep only the last solve's mapping per (period, time)
    s.solve_period_time = s.dt_realize_dispatch
    if s.solve_period_time.droplevel('solve').duplicated().any():
        s.solve_period_time = s.solve_period_time[~s.solve_period_time.droplevel('solve').duplicated(keep='last')]
    s.period = s.period.droplevel('solve')
    s.period__time_first = s.period__time_first.droplevel('solve')
    s.period_first_of_solve = s.period_first_of_solve.droplevel('solve')
    s.period_in_use = s.period_in_use.droplevel('solve').unique()
    s.d_realize_dispatch_or_invest = s.d_realize_dispatch_or_invest.droplevel('solve').unique()
    s.d_realize_invest = s.d_realize_invest.droplevel('solve').unique()
    s.d_realized_period = s.d_realized_period.droplevel('solve').unique()
    s.dt_realize_dispatch = s.dt_realize_dispatch.droplevel('solve').unique()
    s.dt = s.dt.droplevel('solve').unique()
    s.dt_fix_storage_timesteps = s.dt_fix_storage_timesteps.droplevel('solve')
    # For rolling scenarios, dtt/dtttdt have overlapping timesteps from different solves.
    # Filter to only realized (period, time) pairs, keeping the last solve's mapping.
    s.dtt = s.dtt.droplevel('solve')
    dtt_pt = s.dtt.droplevel('t_previous')
    s.dtt = s.dtt[dtt_pt.isin(s.dt_realize_dispatch) & ~dtt_pt.duplicated(keep='last')]
    s.ramp_link = s.ramp_link.droplevel('solve')
    ramp_pt = s.ramp_link.droplevel(['period_previous', 'time_previous'])
    s.ramp_link = s.ramp_link[ramp_pt.isin(s.dt_realize_dispatch) & ~ramp_pt.duplicated(keep='last')]
    s.dtttdt = s.dtttdt.droplevel('solve')
    dtttdt_pt = s.dtttdt.droplevel(['t_previous', 't_previous_within_timeset', 'd_previous', 't_previous_within_solve'])
    s.dtttdt = s.dtttdt[dtttdt_pt.isin(s.dt_realize_dispatch) & ~dtttdt_pt.duplicated(keep='last')]
    s.ed_invest = s.ed_invest.droplevel('solve').unique().join(s.d_realize_invest, how='inner')
    s.edd_invest = s.edd_invest.droplevel('solve').unique()
    s.edd_invest.names = ['entity', 'period_invest', 'period']
    s.edd_invest = s.edd_invest.join(s.d_realize_invest, how='inner')
    s.ed_divest = s.ed_divest.droplevel('solve').unique().join(s.d_realize_invest, how='inner')
    # The period labels of the stochastic fans (the investment costs inside a fan are reported at its labels).
    s.fan_label_period = s.fan_label_period.droplevel('solve').unique()
    # Lead-time sets are keyed by the realized investment DECISION
    # (period_invest); the period the capacity comes online or is in
    # construction may be a look-ahead period of the deciding solve.  A
    # recourse fan decides at its branch periods; its branch-weighted
    # investment is at the label (``v.invest_fan``), so the labels that hold
    # one are decisions too.
    realized_invest = set(s.d_realize_invest) | set(v.invest_fan.index)
    # The operating / retirement windows of the annualized costs are keyed by their decision likewise.
    for attr, decision in (('edd_invest_commission', 'period_invest'), ('edd_invest_in_construction', 'period_invest'),
                           ('edd_operating_window', 'period_invest'), ('edd_retire_window', 'period_divest'),
                           ('edd_salvage_window', 'period_divest')):
        idx = getattr(s, attr).droplevel('solve').unique()
        setattr(s, attr, idx[idx.get_level_values(decision).isin(realized_invest)])

    return par, s, v
