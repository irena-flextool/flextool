import pandas as pd


def generic(par, s, v, r, debug):
    if debug:
        results = []
        df = pd.concat([par.inflation_factor_operations_yearly, par.inflation_factor_investment_yearly], axis=1)
        df.columns = ["operations discount factor","investments discount factor"]
        df.columns.name = "param"
        results.append((df, 'discountFactors_d_p'))

        df = par.entity_annuity
        results.append((df, 'entity_annuity_d_p'))

        return results


def cost_summaries(par, s, v, r, debug):
    """Cost summaries for periods and timesteps"""

    results = []

    # Common calculations
    discount_ops = par.inflation_factor_operations_yearly
    discount_invs = par.inflation_factor_investment_yearly
    period_share = par.complete_period_share_of_year
    to_millions = 1000000

    # 1. Costs at timestep level (non-annualized)
    costs_dt = pd.DataFrame(index=s.dt_realize_dispatch, dtype=float)
    costs_dt.columns.name = 'category'
    costs_dt['commodity_cost'] = r.cost_commodity_dt.sum(axis=1)
    # Sign convention: store the LP obj contribution.  flextool.mod's
    # commodity term is `+ price × BUY - price × SELL` (lines ~1985-2000),
    # so the obj contribution from sales is `-r.sales_commodity_dt`
    # — revenue shows as a negative value, matching costOper_dt's
    # `- r.sales_commodity_dt` (calc_costs.py:158-163).  Without the
    # negation, sum(costs_dt categories) ≠ costOper_dt + costPenalty_dt
    # by 2×|sales|, which propagates into costs_discounted_p_.
    costs_dt['commodity_sales'] = -r.sales_commodity_dt.sum(axis=1)
    costs_dt['co2'] = r.cost_co2_dt
    costs_dt['other operational'] = r.cost_process_other_operational_cost_dt.sum(axis=1)
    costs_dt['starts'] = r.cost_startup_dt.sum(axis=1)
    costs_dt['upward slack penalty'] = r.costPenalty_node_state_upDown_dt.xs('up', level='upDown', axis=1).sum(axis=1)
    costs_dt['downward slack penalty'] = r.costPenalty_node_state_upDown_dt.xs('down', level='upDown', axis=1).sum(axis=1)
    costs_dt['inertia slack penalty'] = r.costPenalty_inertia_dt.sum(axis=1)
    costs_dt['non-synchronous slack penalty'] = r.costPenalty_non_synchronous_dt.sum(axis=1)
    try:
        costs_dt['upward reserve slack penalty'] = r.costPenalty_reserve_upDown_dt.xs('up', level='updown', axis=1).sum(axis=1)
    except KeyError:
        costs_dt['upward reserve slack penalty'] = 0
    try:
        costs_dt['downward reserve slack penalty'] = r.costPenalty_reserve_upDown_dt.xs('down', level='updown', axis=1).sum(axis=1)
    except KeyError:
        costs_dt['downward reserve slack penalty'] = 0

    results.append((costs_dt, 'costs_dt_p'))

    # 2. Annualized, inflation adjusted and years represented (derived from costs_dt)
    dispatch_costs_pure_period = costs_dt.groupby(level='period').sum()
    dispatch_costs_annualized_period = dispatch_costs_pure_period.div(period_share, axis=0) / to_millions
    # discount_ops = inflation_factor_operations_yearly already sums
    # (1+r)^y × p_years_represented[d,y] over the represented years of d
    # (preprocessing/period_calculated_params.py:287-301), so multiplying
    # by it produces the horizon-weighted NPV directly.  Multiplying by
    # par.years_represented_d on top of that double-counts the year
    # weighting and shows up as sum(costs_discounted) = obj × years_rep.
    dispatch_costs_inflation_adjusted = (
        dispatch_costs_annualized_period
        .mul(discount_ops, axis=0)
    )

    # 3. Discounted and inflation adjusted (with years represented) investment costs.
    # Indexed by d_realize_invest ∪ d_realized_period ∪ period_in_use:
    # invest/divest and fixed_invested/fixed_divested only have rows in
    # d_realize_invest, but fixed-cost-pre-existing applies in every period
    # that the LP objective sees — which is ``period_in_use`` (every period
    # carrying a non-zero ``complete_period_share_of_year``).  Excluding
    # ``period_in_use`` periods that are NOT in d_realized_period dropped
    # fixed_pre_existing rows that the LP DID charge for (e.g. the ``all``
    # scenario realizes p2020 dispatch but charges fixed_pre_existing for
    # p2020+p2025 via ``years_represented``; bare
    # ``d_realize_invest ∪ d_realized_period`` discarded p2025's row on
    # assignment to ``investment_costs``, leaving the per-category sum 1000
    # M-CUR below summary's full-horizon total).
    period_in_use = par.complete_period_share_of_year.index
    investment_index = s.d_realize_invest.union(s.d_realized_period).union(period_in_use)
    investment_costs = pd.DataFrame(index=investment_index, dtype=float)
    investment_costs.columns.name = 'category'
    # Invest + divest are independent per-period series: a scenario may
    # realize investment but no divestment anywhere in the cascade (and
    # vice-versa).  When ``v_divest`` is empty across every sub-solve the
    # divest-cost series is row-less, so a bare ``invest + divest`` would
    # align on the index union and yield NaN at EVERY period the divest
    # series lacks — which ``fillna(0.0)`` below then silently zeroes,
    # discarding the valid investment cost (the ``unit investment &
    # retirement`` cost-path collapse for nested-invest scenarios with no
    # divestment).  ``add(fill_value=0)`` is the correct additive-identity
    # union: a period present in only one operand keeps that operand's
    # value instead of becoming NaN.  Byte-identical when both series carry
    # the same full period index (the divest-active scenarios).
    investment_costs['unit investment & retirement'] = r.costInvestUnit_d.add(r.costDivestUnit_d, fill_value=0) / to_millions
    investment_costs['connection investment & retirement'] = r.costInvestConnection_d.add(r.costDivestConnection_d, fill_value=0) / to_millions
    investment_costs['storage investment & retirement'] = r.costInvestState_d.add(r.costDivestState_d, fill_value=0) / to_millions
    investment_costs['fixed cost pre-existing'] = r.costFixedPreExisting_d / to_millions
    investment_costs['fixed cost invested'] = r.costFixedInvested_d / to_millions
    investment_costs['fixed cost reduction of divestments'] = r.costFixedDivested_d / to_millions
    investment_costs['capacity margin penalty'] = r.costPenalty_capacity_margin_d / to_millions
    investment_costs = investment_costs.fillna(0.0)

    # Annualize back: Remove inflation adjustment and years represented
    annual_invest_costs = investment_costs.div(discount_invs, axis=0)
    annual_invest_costs['fixed cost pre-existing'] = investment_costs['fixed cost pre-existing'].div(discount_ops, axis=0)
    annual_invest_costs['fixed cost invested'] = investment_costs['fixed cost invested'].div(discount_ops, axis=0)
    annual_invest_costs['fixed cost reduction of divestments'] = investment_costs['fixed cost reduction of divestments'].div(discount_ops, axis=0)
    annual_invest_costs['capacity margin penalty'] = investment_costs['capacity margin penalty'].div(discount_ops, axis=0)

    # 4. Combined summary (investment + dispatch aggregated to period)
    all_periods = s.d_realized_period.union(s.d_realize_invest)
    summary = pd.DataFrame(index=all_periods, dtype=float)
    summary.columns.name = 'parameter'

    # Without inflation and years (so, pure annual results)
    summary_annualized = annual_invest_costs.join(dispatch_costs_annualized_period)
    results.append((summary_annualized, 'annualized_costs_d_p'))

    # With years_represented adjusted with inflation (same as model).
    # Outer join: investment_costs is indexed by d_realize_invest (only
    # periods where invest/divest is realized), but dispatch_costs covers
    # d_realized_period (every realized period).  A left join on
    # investment_costs would silently drop dispatch costs for periods
    # without investment activity (e.g. y2020_2029_2x5y dispatches in
    # both p2020 and p2025 but only realizes invest in p2020).
    summary_inflation_years = investment_costs.join(
        dispatch_costs_inflation_adjusted, how='outer'
    ).fillna(0.0)
    results.append((summary_inflation_years, 'costs_discounted_d_p'))

    # With years_represented adjusted with inflation (same as model).
    # Outer join: investment_costs is indexed by d_realize_invest (only
    # periods where invest/divest is realized), but dispatch_costs covers
    # d_realized_period (every realized period).  A left join on
    # investment_costs would silently drop dispatch costs for periods
    # without investment activity (e.g. y2020_2029_2x5y dispatches in
    # both p2020 and p2025 but only realizes invest in p2020).
    summary_inflation_years = investment_costs.join(
        dispatch_costs_inflation_adjusted, how='outer'
    ).fillna(0.0)
    results.append((summary_inflation_years.sum(axis=0), 'costs_discounted_p_'))

    return results


def cost_breakdown_by_entity(par, s, v, r, debug):
    """Per-entity cost break-down at the PERIOD level (unit / connection / node).

    Purely additive over the existing per-entity intermediates in
    ``calc_costs.py``.  Each table mirrors ``cost_summaries`` arithmetic
    EXACTLY, only collapsed per-entity instead of system-wide, so summing a
    per-entity table over its entities reconciles against the system summary
    (``annualized_costs_d_p`` / ``costs_discounted_d_p``).  Six tables: unit,
    connection, node — in annualized (M CUR/a) and discounted (M CUR) flavours.

    Index = (period, entity) MultiIndex with the entity level named
    'unit'/'connection'/'node'; columns = single level named 'category'.
    PERIOD-LEVEL ONLY — no per-timestep tables (this deliberately avoids the
    CSV writer's positional solve-injection on dt-indexed frames).

    Fuel/commodity cost is attributed to the consuming process; multi-group
    CO2 cost is the sum over every priced group a process touches (matching the
    LP objective).
    """
    results = []

    discount_ops = par.inflation_factor_operations_yearly
    discount_invs = par.inflation_factor_investment_yearly
    period_share = par.complete_period_share_of_year
    to_millions = 1000000

    # Entity sets for the three entity types.  Operational/startup/fuel/co2
    # process frames are sliced to the processes of the relevant type; the
    # invest/fixed frames (already per-entity) are sliced the same way.
    entity_specs = (
        ('unit', list(s.process_unit), 'unit'),
        ('connection', list(s.process_connection), 'connection'),
        ('node', list(s.node_state), 'node'),
    )

    def _slice_cols(frame, entities):
        """Columns of ``frame`` restricted to ``entities`` (present only)."""
        return [e for e in entities if e in frame.columns]

    def _op_annualized(frame, entities):
        """Operational dt frame → per-(period, entity) annualized Series
        (M CUR/a).  Matches dispatch_costs_annualized_period."""
        cols = _slice_cols(frame, entities)
        if not cols:
            return None
        per_d = frame[cols].groupby('period').sum().div(period_share, axis=0) / to_millions
        return per_d.stack(future_stack=True)

    def _op_discounted(frame, entities):
        """Operational dt frame → per-(period, entity) discounted Series
        (M CUR).  Matches dispatch_costs_inflation_adjusted."""
        cols = _slice_cols(frame, entities)
        if not cols:
            return None
        per_d = (
            frame[cols].groupby('period').sum()
            .div(period_share, axis=0)
            .mul(discount_ops, axis=0)
            / to_millions
        )
        return per_d.stack(future_stack=True)

    def _inv_discounted(frame, entities):
        """Already-discounted M-path per-entity frame → per-(period, entity)
        Series (M CUR).  Matches investment_costs (/1e6 only)."""
        cols = _slice_cols(frame, entities)
        if not cols:
            return None
        return (frame[cols] / to_millions).stack(future_stack=True)

    def _inv_annualized(frame, entities, *, discount):
        """Already-discounted M-path per-entity frame → per-(period, entity)
        annualized Series (M CUR/a): remove the supplied discount factor.
        Matches annual_invest_costs."""
        cols = _slice_cols(frame, entities)
        if not cols:
            return None
        return (frame[cols].div(discount, axis=0) / to_millions).stack(future_stack=True)

    for kind, entities, level_name in entity_specs:
        # Build (annualized_pieces, discounted_pieces) dicts keyed by category.
        ann_pieces: dict = {}
        dis_pieces: dict = {}

        # --- Operational (process / node-state) ---
        # commodity_sales is NEGATED to match out_costs.py:40 sign convention.
        op_specs = [
            ('commodity_cost', r.cost_commodity_process_dt, 1.0),
            ('commodity_sales', r.sales_commodity_process_dt, -1.0),
            ('co2', r.cost_process_co2_dt, 1.0),
            ('other operational', r.cost_process_other_operational_cost_dt, 1.0),
            ('starts', r.cost_startup_dt, 1.0),
        ]
        for cat, frame, sign in op_specs:
            src = frame if sign == 1.0 else frame.mul(sign)
            a = _op_annualized(src, entities)
            d = _op_discounted(src, entities)
            if a is not None:
                ann_pieces[cat] = a
            if d is not None:
                dis_pieces[cat] = d

        # --- Node-state slack penalties (node table only) ---
        # Same operational annualized/discounted treatment (they flow through
        # costs_dt_p → annualized/discounted in the system path,
        # out_costs.py:44-45).  The slack penalty applies to every balance
        # node carrying a state-slack variable (the frame's own columns) —
        # which is a SUPERSET of s.node_state (storage-state nodes).  Slicing
        # to s.node_state here would drop the penalties of balance-only nodes
        # and break the system reconciliation, so use the frame's own node
        # columns as the entity set instead of ``entities``.
        if kind == 'node':
            slack = r.costPenalty_node_state_upDown_dt
            for cat, ud in (('upward slack penalty', 'up'),
                            ('downward slack penalty', 'down')):
                try:
                    sub = slack.xs(ud, level='upDown', axis=1)
                except KeyError:
                    continue
                slack_nodes = list(sub.columns)
                a = _op_annualized(sub, slack_nodes)
                d = _op_discounted(sub, slack_nodes)
                if a is not None:
                    ann_pieces[cat] = a
                if d is not None:
                    dis_pieces[cat] = d

        # --- Investment / fixed (already discounted M-path) ---
        # Skip categories whose sliced frame is empty (dispatch-only models
        # have empty v.invest / v.divest).
        invest_label = 'storage investment' if kind == 'node' else 'investment'
        divest_label = 'storage retirement' if kind == 'node' else 'retirement'
        inv_specs = [
            (invest_label, r.cost_entity_invest_d, discount_invs),
            (divest_label, r.cost_entity_divest_d, discount_invs),
            ('fixed cost pre-existing', r.cost_entity_fixed_pre_existing, discount_ops),
            ('fixed cost invested', r.cost_entity_fixed_invested, discount_ops),
            ('fixed cost reduction of divestments', r.cost_entity_fixed_divested, discount_ops),
        ]
        for cat, frame, discount in inv_specs:
            d = _inv_discounted(frame, entities)
            a = _inv_annualized(frame, entities, discount=discount)
            if a is not None:
                ann_pieces[cat] = a
            if d is not None:
                dis_pieces[cat] = d

        for pieces, flavour in ((ann_pieces, 'annualized'), (dis_pieces, 'discounted')):
            if not pieces:
                # Empty guard: no category pieces — skip the append.
                continue
            df = pd.concat(pieces, axis=1)
            df.columns.name = 'category'
            # After future_stack the entity level is the last index level;
            # name the (period, entity) levels explicitly.
            df.index = df.index.set_names(['period', level_name])
            df = df.fillna(0.0).sort_index()
            key = f'cost_{kind}_{flavour}_d_ec'
            results.append((df, key))

    return results


def CO2(par, s, v, r, debug):
    """Annualized CO2 Mt for groups by period"""
    results = []

    # Calculate CO2 emissions in Mt
    total_co2 = ((r.emissions_co2_d * par.years_represented_d) / 1000000).sum(axis=0)
    co2_summary = pd.DataFrame(index=["CO2 [Mt]"], columns=["model_wide"], data=total_co2)
    co2_summary.index.name = 'param_CO2'
    co2_summary.columns.name = 'scope'
    results.append((co2_summary, 'CO2__'))

    # Process co2 emissions (annualized — plot rule 'y' weights by years_represented for horizon totals)
    process_co2 = r.process_emissions_co2_d.groupby(['period']).sum()
    results.append((process_co2, 'process_co2_d_eee'))

    # Group co2 emissions (annualized)
    results.append((r.group_co2_d, 'CO2_d_g'))
    return results
