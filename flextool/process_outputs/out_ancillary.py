import pandas as pd


def connection(par, s, v, r, debug):
    """Connection flow for periods and time"""

    results = []

    # Return timestep results
    results.append((r.connection_dt, 'connection_dt_eee'))
    results.append((r.connection_losses_dt, 'connection_losses_dt_eee'))

    # Annualize: r.connection_d / r.connection_losses_d from calc_connections
    # are already in MWh (v_flow × step_duration summed per period); dividing
    # by complete_period_share_of_year scales them to MWh/year.
    r.connection_d = r.connection_d.div(par.complete_period_share_of_year, axis=0)
    r.connection_losses_d = r.connection_losses_d.div(par.complete_period_share_of_year, axis=0)

    # Return period results
    results.append((r.connection_d, 'connection_d_eee'))
    results.append((r.connection_losses_d, 'connection_losses_d_eee'))

    return results


def connection_wards(par, s, v, r, debug):
    """Connection flow to right node and to left node for periods and time"""

    results = []

    # Return timestep results
    results.append((r.connection_to_left_node__dt, 'connection_leftward_dt_eee'))
    results.append((r.connection_to_right_node__dt, 'connection_rightward_dt_eee'))

    # Return period results
    results.append((r.connection_to_right_node__d, 'connection_rightward_d_eee'))
    results.append((r.connection_to_left_node__d, 'connection_leftward_d_eee'))

    return results


def connection_cf(par, s, v, r, debug):
    """Connection capacity factors for periods"""
    complete_hours = par.complete_period_share_of_year * 8760
    connection_cols = r.process_sink_flow_d.columns[r.process_sink_flow_d.columns.get_level_values(0).isin(s.process_connection)]
    connection_capacity = r.entity_all_capacity[connection_cols.droplevel(1).unique()].rename_axis('process', axis=1)
    connection_capacity.columns = connection_capacity.columns.get_level_values(0)
    # v_flow is MW (MWh/h); multiply by step_duration for MWh before the CF ratio.
    results = (
        r.connection_dt.abs()
        .mul(par.step_duration, axis=0)
        .groupby('period').sum()
        .div(connection_capacity, level=0)
        .div(complete_hours, axis=0)
    )
    results.columns.names = ['connection']
    return results, 'connection_cf_d_e'


def reserves(par, s, v, r, debug):
    """Process reserves for timesteps and periods"""
    results = []

    # Timestep-level reserves
    results.append((r.reserves_dt, 'process_reserve_upDown_node_dt_eppe'))

    # Period-level reserves (average)
    results.append((r.reserves_d, 'process_reserve_average_d_eppe'))

    # Reserve price results
    results.append((v.dual_reserve_balance, 'reserve_prices_dt_ppg'))

    return results


def investment_duals(par, s, v, r, debug):
    """Additional node results: prices, state, and slacks"""
    results = []

    # 1. v.dual_invest_unit
    dual_invest_unit = v.dual_invest_unit.div(par.entity_unitsize[v.dual_invest_unit.columns])
    results.append((dual_invest_unit, 'dual_invest_unit_d_e'))

    # 2. v.dual_invest_connection
    dual_invest_connection = v.dual_invest_connection.div(par.entity_unitsize[v.dual_invest_connection.columns])
    results.append((dual_invest_connection, 'dual_invest_connection_d_e'))

    # 3. v.dual_invest_node
    dual_invest_node = v.dual_invest_node.div(par.entity_unitsize[v.dual_invest_node.columns])
    results.append((dual_invest_node, 'dual_invest_node_d_e'))

    # 4. Synthesized effective investment dual per entity type.
    # Complete signed marginal value (objective per MW) of one more MW of
    # investment capacity, folding every binding regime (a/b/c).  Positive
    # means more investment in this entity would lower the objective.
    combined = _synthesize_invest_dual(v, par)
    if not combined.empty:
        units = [c for c in combined.columns if c in s.process_unit]
        connections = [c for c in combined.columns if c in s.process_connection]
        nodes = [c for c in combined.columns if c in s.node]
        if units:
            results.append((combined[units], 'dual_invest_effective_unit_d_e'))
        if connections:
            results.append((combined[connections], 'dual_invest_effective_connection_d_e'))
        if nodes:
            results.append((combined[nodes], 'dual_invest_effective_node_d_e'))

    return results


def _synthesize_invest_dual(v, par) -> "pd.DataFrame":
    """Complete SIGNED effective investment dual, per (entity, period).

    Returns the full marginal value of one more MW of investment capacity in
    an entity, in objective units per MW.  Sign convention (user-confirmed):
    POSITIVE means more investment in this entity would IMPROVE (lower) the
    objective.

    The value folds every binding investment regime (≤1 nonzero per
    entity-period at a non-degenerate optimum).  Each source is first put in
    objective-per-MW units, then NEGATED, then summed:

    - (b) upper-cap binds — the ``<=`` row duals
      (``maxInvest_period``/``maxInvest_total``/``maxCumulative`` and the
      group ``maxInvestGroup_*`` caps).  Their raw HiGHS dual is NEGATIVE;
      negate → POSITIVE (the cap holds the entity back, so more would help).
    - (c) lower-floor binds — the ``>=`` row duals
      (``minInvest_period``/``minInvest_total``/``minCumulative`` and the
      group ``minInvestGroup_*`` floors).  Their raw dual is POSITIVE;
      negate → NEGATIVE (the floor over-forces capacity, so more would hurt).
    - (a) not built — the v_invest COLUMN reduced cost
      (``dual_invest_unit``/``connection``/``node``), correctly scaled by
      ``1/scale_the_objective`` (Increment 2).  It is in objective per
      v_invest-unit, so divide by ``entity_unitsize`` (mirroring the
      standalone path in ``investment_duals``) to reach objective per MW;
      the reduced cost is ≥0, so negate → NEGATIVE (unprofitable to build).
    - (d) interior, binds nothing → 0.

    ASYMMETRY (read off ``drop_levels.py``): ``dual_maxInvest_total`` is in
    ``_V_SOLVE_ONLY`` — it is collapsed to a single-row ``RangeIndex(1)``
    (period-less) and so must be broadcast across the period axis of its
    siblings before combining.  ``dual_minInvest_total`` is in ``_V_DROP`` —
    after dropping the solve level it retains a ``period`` index and is
    ALREADY per-(entity, period), so it is added directly without broadcast.

    All duals are obj/MW; non-binding constraints contribute 0.
    """
    import pandas as pd

    # Collect all entity-level constraint duals, aligned to the same columns
    # and index.  Every contribution is appended already NEGATED so the final
    # sum carries the signed obj/MW convention documented above.
    entity_duals: list[pd.DataFrame] = []

    # A reference (entity, period) frame to broadcast the period-less
    # ``dual_maxInvest_total`` across.  Prefer a per-period sibling that
    # actually carries the period axis.  The per-period constraint-dual
    # families are each independently emission-gated, so when a per-entity
    # ``maxInvest_total`` cap is the ONLY binding family they are all empty.
    # Fall back to the regime-(a) column reduced-cost frames
    # (``dual_invest_unit``/``connection``/``node``): per _parquet_bundle.py
    # these are ALWAYS present with a realized-``period`` index for any
    # investable entity (no emission gate).  Only ``period_ref.index`` is
    # used for the broadcast, so the differing column axis name on these
    # fallbacks is irrelevant.
    period_ref: pd.DataFrame | None = next(
        (
            df
            for df in (
                v.dual_maxInvest_period,
                v.dual_minInvest_period,
                v.dual_maxCumulative,
                v.dual_minCumulative,
                v.dual_minInvest_total,
                v.dual_invest_unit,
                v.dual_invest_connection,
                v.dual_invest_node,
            )
            if not df.empty
        ),
        None,
    )

    # (b) Max-side ``<=`` duals: raw < 0 → negate → POSITIVE.
    #     ``dual_maxInvest_total`` is solve-only → broadcast over periods.
    for df in (v.dual_maxInvest_period, v.dual_maxInvest_total, v.dual_maxCumulative):
        if df.empty:
            continue
        if df is v.dual_maxInvest_total:
            if period_ref is None:
                continue
            df = pd.DataFrame(
                {c: df.iloc[0][c] for c in df.columns},
                index=period_ref.index,
            )
            df.columns.name = 'entity'
        entity_duals.append(df.mul(-1.0))

    # (c) Min-side ``>=`` duals: raw > 0 → negate → NEGATIVE.
    #     ``dual_minInvest_total`` is ALREADY per-(entity, period) — add it
    #     directly, NO broadcast (the max-total asymmetry above).
    for df in (v.dual_minInvest_period, v.dual_minInvest_total, v.dual_minCumulative):
        if df.empty:
            continue
        entity_duals.append(df.mul(-1.0))

    # (b)/(c) group caps/floors: expand group → member entities, then negate.
    if not v.group_entity_invest.empty:
        group_map = v.group_entity_invest  # columns: group, entity
        for group_df in (
            v.dual_maxInvestGroup_period, v.dual_maxInvestGroup_total,
            v.dual_maxInvestGroup_cumulative,
            v.dual_minInvestGroup_period, v.dual_minInvestGroup_total,
            v.dual_minInvestGroup_cumulative,
        ):
            if group_df.empty:
                continue
            for group_name in group_df.columns:
                members = group_map.loc[group_map['group'] == group_name, 'entity'].tolist()
                if not members:
                    continue
                # Create per-entity columns with the group's dual value.
                group_series = group_df[group_name]
                expanded = pd.DataFrame(
                    {entity: group_series for entity in members},
                    index=group_series.index,
                )
                expanded.columns.name = 'entity'
                entity_duals.append(expanded.mul(-1.0))

    # (a) Not-built regime — v_invest COLUMN reduced cost.  Divide by
    #     entity_unitsize (obj/v_invest-unit → obj/MW), then negate.  These
    #     frames have a period index and a 'unit'/'connection'/'node' column
    #     axis; rename to the common 'entity' axis before combining.
    for col_dual in (v.dual_invest_unit, v.dual_invest_connection, v.dual_invest_node):
        if col_dual.empty:
            continue
        per_mw = col_dual.div(par.entity_unitsize[col_dual.columns])
        per_mw = per_mw.rename_axis('entity', axis=1)
        entity_duals.append(per_mw.mul(-1.0))

    if not entity_duals:
        return pd.DataFrame()

    # Align all DataFrames to a common (index, columns) and sum.
    combined = entity_duals[0]
    for df in entity_duals[1:]:
        combined = combined.add(df, fill_value=0.0)
    combined = combined.fillna(0.0)

    return combined


def co2_duals(par, s, v, r, debug):
    """CO2 emission-cap shadow prices in Currency / tCO2 (positive = cost).

    Both sides of co2_max_period and co2_max_total in the mod are divided
    by 1000 (Mt instead of t).  The raw dual is therefore
    Δobj / Δ(scaled-RHS in Mt); to get per-tCO2 we DIVIDE by 1000 (since
    Δ(scaled) = Δ(raw)/1000).  The dual is also in NPV currency because
    the objective discounts operational costs; for the per-period cap we
    divide by inflation_factor_operations_yearly[d] to recover nominal
    Currency/tCO2 at period d (mirroring v_dual_node_balance).  Sign is
    always flipped so binding caps show as positive costs — negatives
    should ring alarm bells.

    The cumulative (total) cap has no single period to un-discount against
    (it spans all periods in the sum), so it is reported in NPV currency
    (same convention as investment duals).  Its scalar per group is
    broadcast across periods for uniform (period × group) shape.

    Caveat: the constraint LHS does not include p_rp_cost_weight whereas
    the objective does.  In rp (representative-period) scenarios the raw
    dual therefore picks up an extra rp_cost_weight factor, inflating
    shadow prices relative to chrono runs.  Fixing that requires either
    adding the weight to the constraint LHS in the mod, or post-dividing
    by a representative value of rp_cost_weight per period.
    """
    import pandas as pd

    results = []
    periods = s.d_realized_period

    # Period-limited shadow price: (period × group) in nominal Currency/tCO2
    if not v.dual_co2_max_period.empty:
        co2_period_price = (-v.dual_co2_max_period / 1000).astype(float)
        co2_period_price = co2_period_price.div(
            par.inflation_factor_operations_yearly, axis=0
        )
        co2_period_price.columns.name = 'group'
        co2_period_price.index.name = 'period'
        results.append((co2_period_price, 'co2_price_period_d_g'))

    # Cumulative limit shadow price: scalar per group, broadcast across periods.
    # Kept in NPV currency (no single period to un-discount against).
    if not v.dual_co2_max_total.empty:
        total_row = v.dual_co2_max_total.iloc[0]  # Series indexed by group
        total_price = (-total_row / 1000).astype(float)
        co2_total_price = pd.DataFrame(
            {g: [total_price[g]] * len(periods) for g in total_price.index},
            index=pd.Index(periods, name='period'),
        )
        co2_total_price.columns.name = 'group'
        results.append((co2_total_price, 'co2_price_total_d_g'))

    return results


def inertia_results(par, s, v, r, debug):
    """Inertia results for groups and individual entities"""

    results = []

    # 1. Calculate unit_inertia for all (process, node) without groups
    unit_inertia = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['process', 'node']), dtype=float)

    # === SOURCE-BASED INERTIA ===
    s.process_source_with_inertia = par.process_source.columns[list(par.process_source.loc['inertia_constant'] > 0)]
    pss_source_inertia = s.process_source_sink_alwaysProcess[
        s.process_source_sink_alwaysProcess.droplevel('sink').isin(s.process_source_with_inertia)
    ]
    s.process_source_with_inertia.names = ['process', 'node']

    # Online processes - group by (process, source) since online_dt is indexed by process only
    pss_source_online_inertia = pss_source_inertia[pss_source_inertia.get_level_values('process').isin(s.process_online)]
    process_online_inertia = pss_source_online_inertia.droplevel('sink').unique()
    process_online_inertia.names = ['process', 'node']
    online_inertia_procs = process_online_inertia.get_level_values('process').unique()

    unit_inertia[process_online_inertia] = ( r.process_online_dt[online_inertia_procs]
        .mul(par.entity_unitsize[online_inertia_procs], axis=1, level=0)
        .mul(par.process_source.loc['inertia_constant'][process_online_inertia]) )

    # Flow processes
    pss_source_flow_inertia = pss_source_inertia[~pss_source_inertia.get_level_values('process').isin(s.process_online)]
    flow_inertia_cols = pss_source_flow_inertia.intersection(r.flow_dt.columns)
    process_flow = flow_inertia_cols.droplevel('sink').unique()
    par_process_source_inertia = par.process_source[process_flow].loc['inertia_constant']
    par_process_source_inertia.index = flow_inertia_cols.join(par_process_source_inertia.index)
    flows_weighted_source = (
        r.flow_dt[flow_inertia_cols]
        .mul(par_process_source_inertia) )

    # Sum across sinks for each (process, source)
    unit_inertia_source_flow = flows_weighted_source.T.groupby(level=['process', 'source']).sum().T
    unit_inertia_source_flow.columns.names = ['process', 'node']
    unit_inertia[unit_inertia_source_flow.columns] = unit_inertia_source_flow

    # === SINK-BASED INERTIA ===
    s.process_sink_with_inertia = par.process_sink.columns[list(par.process_sink.loc['inertia_constant'] > 0)]
    pss_sink_inertia = s.process_source_sink_alwaysProcess[
        s.process_source_sink_alwaysProcess.droplevel('source').isin(s.process_sink_with_inertia)
    ]
    s.process_sink_with_inertia.names = ['process', 'node']

    # Online processes - group by (process, sink) since online_dt is indexed by process only
    pss_sink_online_inertia = pss_sink_inertia[pss_sink_inertia.get_level_values('process').isin(s.process_online)]
    process_online_inertia = pss_sink_online_inertia.droplevel('source').unique()
    process_online_inertia.names = ['process', 'node']
    online_inertia_procs = process_online_inertia.get_level_values('process').unique()

    unit_inertia[process_online_inertia] = ( r.process_online_dt[online_inertia_procs]
        .mul(par.entity_unitsize[online_inertia_procs], axis=1, level=0)
        .mul(par.process_sink.loc['inertia_constant'][process_online_inertia]) )

    # Flow processes
    pss_sink_flow_inertia = pss_sink_inertia[~pss_sink_inertia.get_level_values('process').isin(s.process_online)]
    flow_inertia_cols = pss_sink_flow_inertia.intersection(r.flow_dt.columns)
    process_flow = flow_inertia_cols.droplevel('source').unique()
    par_process_sink_inertia = par.process_sink[process_flow].loc['inertia_constant']
    par_process_sink_inertia.index = flow_inertia_cols.join(par_process_sink_inertia.index)
    flows_weighted_sink = (
        r.flow_dt[flow_inertia_cols]
        .mul(par_process_sink_inertia) )

    # Sum across sources for each (process, sink)
    unit_inertia_sink_flow = flows_weighted_sink.T.groupby(level=['process', 'sink']).sum().T
    unit_inertia_sink_flow.columns.names = ['process', 'node']
    unit_inertia[unit_inertia_sink_flow.columns] = unit_inertia_sink_flow

    # 2. Add group dimension by joining with group_node
    group_unit_inertia = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['group', 'process', 'node']), dtype=float)

    for g in s.groupInertia:
        # Get (process, node) pairs for this group
        group_pn = s.group_node[s.group_node.get_level_values('group') == g].droplevel('group')
        # Filter unit_inertia to these columns
        cols = unit_inertia.columns.intersection(group_pn)
        # Add group level to columns
        group_cols = pd.MultiIndex.from_arrays(
            [[g] * len(cols), cols.get_level_values(0), cols.get_level_values(1)],
            names=['group', 'process', 'node']
        )
        group_unit_inertia[group_cols] = unit_inertia[cols].values

    results.append((group_unit_inertia, 'nodeGroup_unit_node_inertia_dt_gee'))

    # 3. Group inertia - sum by group
    group_inertia = group_unit_inertia.T.groupby(level='group').sum().T
    results.append((group_inertia, 'nodeGroup_inertia_dt_g'))

    # 4. Largest flow per group
    largest_flow = pd.DataFrame(index=s.dt_realize_dispatch, dtype=float)
    largest_flow.columns.name = 'group'

    for g in s.groupInertia:
        group_nodes = s.group_node[s.group_node.get_level_values('group') == g].get_level_values('node')
        process_sink_in_group = s.process_sink[s.process_sink.get_level_values('sink').isin(group_nodes)]
        pss_sink = s.process_source_sink_alwaysProcess[
            s.process_source_sink_alwaysProcess.droplevel('source').isin(process_sink_in_group)
        ]

        flow_cols = pss_sink.intersection(r.flow_dt.columns)
        largest_flow[g] = r.flow_dt[flow_cols].max(axis=1)

    results.append((largest_flow, 'nodeGroup_inertia_largest_flow_dt_g'))

    return results


def slack_variables(par, s, v, r, debug):
    """Slack variables for reserves, non-synchronous, inertia, and capacity margin"""

    results = []

    # 1. Reserve slack variables
    results.append((r.q_reserves_dt, 'nodeGroup_slack_reserve_dt_eeg'))

    # 2. Non-synchronous slack variables
    results.append((r.q_non_synchronous_dt, 'nodeGroup_slack_nonsync_dt_g'))

    # 3. Inertia slack variables
    results.append((r.q_inertia_dt, 'nodeGroup_slack_inertia_dt_g'))

    # 4. Capacity margin slack variables (for investment periods only)
    results.append((r.q_capacity_margin_d_not_annualized, 'nodeGroup_slack_capacity_margin_d_g'))

    # 5. Period-level slack variables (annualized)
    results.append((r.q_reserves_d, 'nodeGroup_slack_reserve_d_eeg'))
    results.append((r.q_inertia_d, 'nodeGroup_slack_inertia_d_g'))
    results.append((r.q_non_synchronous_d, 'nodeGroup_slack_nonsync_d_g'))

    return results


def dc_power_flow(par, s, v, r, debug):
    """DC power flow voltage angles and angle differences per connection"""

    results = []

    # Voltage angles per node (may be empty when no DC PF nodes exist)
    if not r.angle_dt.empty:
        results.append((r.angle_dt, 'dc_angle_dt_e'))

    # Angle difference per connection
    if not r.connection_angle_diff_dt.empty:
        results.append((r.connection_angle_diff_dt, 'dc_angle_diff_dt_e'))

    # DC PF structural sets (Index objects become parquet mapping files)
    if not s.node_dc_power_flow.empty:
        results.append((s.node_dc_power_flow, 'node_dc_power_flow'))
    if not s.connection_dc_power_flow.empty:
        results.append((s.connection_dc_power_flow, 'connection_dc_power_flow'))

    return results if results else None


def input_sets(par, s, v, r, debug):
    """Input sets needed for scenario results"""

    results = []
    results.append((s.group_node, 'group_node'))
    results.append((s.group_process, 'group_process'))
    results.append((s.group_process_node, 'group_process_node'))
    results.append((s.nodeGroupIndicators, 'nodeGroupIndicators'))
    results.append((s.flowGroupIndicators, 'flowGroupIndicators'))
    results.append((s.nodeGroupDispatch, 'nodeGroupDispatch'))
    results.append((s.nodeGroupDispatch__connection_Not_in_aggregate, 'nodeGroupDispatch__connection_Not_in_aggregate'))
    results.append((s.nodeGroupDispatch__process__unit__to_node_Not_in_aggregate, 'nodeGroupDispatch__process__unit__to_node_Not_in_aggregate'))
    results.append((s.nodeGroupDispatch__process__node__to_unit_Not_in_aggregate, 'nodeGroupDispatch__process__node__to_unit_Not_in_aggregate'))
    results.append((s.nodeGroupDispatch__process__connection__to_node_Not_in_aggregate, 'nodeGroupDispatch__process__connection__to_node_Not_in_aggregate'))
    results.append((s.nodeGroupDispatch__process__node__to_connection_Not_in_aggregate, 'nodeGroupDispatch__process__node__to_connection_Not_in_aggregate'))
    results.append((s.nodeGroupDispatch__processGroup_Unit_to_group, 'nodeGroupDispatch__processGroup_Unit_to_group'))
    results.append((s.nodeGroupDispatch__processGroup__process__unit__to_node, 'nodeGroupDispatch__processGroup__process__unit__to_node'))
    results.append((s.nodeGroupDispatch__processGroup_Group_to_unit, 'nodeGroupDispatch__processGroup_Group_to_unit'))
    results.append((s.nodeGroupDispatch__processGroup__process__node__to_unit, 'nodeGroupDispatch__processGroup__process__node__to_unit'))
    results.append((s.nodeGroupDispatch__processGroup_Connection, 'nodeGroupDispatch__processGroup_Connection'))
    results.append((s.nodeGroupDispatch__processGroup__process__connection__to_node, 'nodeGroupDispatch__processGroup__process__connection__to_node'))
    results.append((s.nodeGroupDispatch__processGroup__process__node__to_connection, 'nodeGroupDispatch__processGroup__process__node__to_connection'))
    results.append((s.nodeGroupDispatch__process_fully_inside, 'nodeGroupDispatch__process_fully_inside'))
    results.append((par.node_inflow, 'node_inflow__dt'))

    # Per-period years_represented — used by plot rules 'y' and 'z' to convert
    # annualized outputs to horizon totals or years-weighted averages.
    years_d = par.years_represented_d.rename('years_represented').to_frame()
    years_d.index.name = 'period'
    years_d.columns.name = 'param'
    results.append((years_d, 'years_represented__d'))

    return results
