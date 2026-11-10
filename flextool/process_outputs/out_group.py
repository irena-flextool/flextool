"""Output functions for node group results."""
import pandas as pd

from flextool.process_outputs._annualize import annualize_dt_to_d


def _get_group_nodes_with_inflow(s, g: str) -> list[str]:
    """Return nodes in group g that have an inflow method (not 'no_inflow')."""
    group_nodes = (
        s.group_node[s.group_node.get_level_values('group') == g]
        .get_level_values('node')
        .tolist()
    )
    if hasattr(s, 'node__inflow_method'):
        no_inflow_nodes = {n for (n, method) in s.node__inflow_method if method == 'no_inflow'}
        return [n for n in group_nodes if n not in no_inflow_nodes]
    return group_nodes


def _group_vre_flow(par, s, r, group_nodes) -> pd.Series:
    """VRE generation into the member nodes of a group [MWh/step].

    Sums ``r.flow_dt`` (MW) over the arcs of the VRE processes whose sink is
    one of ``group_nodes`` and multiplies by ``step_duration``.  The node-group
    indicators and ``nodeGroup_VRE_share`` both take the VRE energy from here,
    so '2. VRE share of demand' and ``nodeGroup_VRE_share_d_g`` agree.
    """
    vre_processes = s.process_VRE.get_level_values('process')
    cols = r.flow_dt.columns
    vre_cols = cols[
        cols.get_level_values('sink').isin(group_nodes)
        & cols.get_level_values('process').isin(vre_processes)
        & cols.isin(s.process_source_sink_alwaysProcess)
    ]
    return r.flow_dt[vre_cols].mul(par.step_duration, axis=0).sum(axis=1)


def _realized_timestep_weight(par, dt_index: pd.MultiIndex) -> pd.Series:
    """``par.timestep_weight`` aligned to the realized ``(period, time)`` steps.

    Raises ``ValueError`` when a realized step has no weight.  A plain
    ``.mul(weight, axis=0)`` would outer-join and turn such a step into NaN,
    which ``groupby.sum()`` then skips silently in both the numerator and the
    denominator of a period share.
    """
    w = par.timestep_weight.reindex(dt_index)
    missing = w.index[w.isna()]
    if len(missing):
        shown = ', '.join(f'({d}, {t})' for d, t in list(missing)[:5])
        more = f' and {len(missing) - 5} more' if len(missing) > 5 else ''
        raise ValueError(
            f'Node group results: no timestep weight for the realized time steps {shown}{more}.'
        )
    return w


def _safe_ratio(num, den):
    """``num / den`` with 0 where ``den`` is 0 (never ±inf or NaN)."""
    return (num / den.where(den != 0)).fillna(0.0)


def nodeGroup_indicators(par, s, v, r, debug):
    """Node group indicator results by period and time.

    Time-step table ``nodeGroup_gdt_p``: energies in MWh/step and the shares
    of the step.  Period table ``nodeGroup_gd_p``: every share is a ratio of
    weighted sums over the period's realized steps,
    ``share_X[d] = Σ_t w[d,t]·X[d,t] / Σ_t w[d,t]·Y[d,t]`` with
    ``w = par.timestep_weight``, never a sum or a mean of the step shares:

    - 1. Loss of load share = loss of load / inflow
    - 2. VRE share of demand = VRE generation / inflow
    - 3. Excess load share = excess load / inflow
    - 4. Curtailed VRE of demand = curtailed VRE / inflow
    - 6. Curtailed VRE of potential VRE = curtailed VRE / potential VRE

    The years represented and ``complete_period_share_of_year`` cancel in
    each of these ratios (numerator and denominator are from the same
    period), so a share must not be annualised.  '5. Annualized inflow' is
    the sum of the dt column '7', which is already weighted.
    """

    results = []

    if not list(s.nodeGroupIndicators):
        return results

    # Get time steps
    dt_index = s.dt_realize_dispatch  # Should be MultiIndex with (period, time)
    weight = _realized_timestep_weight(par, dt_index)

    # Calculate timestep-level results first
    results_dt = []
    # Weighted energies per (group, period, time) for the period shares
    weighted_dt = []

    for g in s.nodeGroupIndicators:
        group_nodes = (
            s.group_node[s.group_node.get_level_values('group').isin([g])]
            .get_level_values('node')
            .tolist()
        )
        group_nodes_with_inflow = _get_group_nodes_with_inflow(s, g)

        # 1. pdtNodeInflow (negative sum)
        if group_nodes_with_inflow:
            group_inflow = -par.node_inflow[group_nodes_with_inflow].sum(axis=1)
        else:
            group_inflow = pd.Series(0, index=dt_index)

        # 2. Sum of annualized inflows [MWh].  Weight each (d, t) by
        # par.timestep_weight (=1.0 with no/uniform timeset_weights →
        # byte-identical) so the annualized inflow matches the cost-weighted
        # objective.  group_inflow itself stays unweighted: it is the raw
        # '5. Timestep inflow' output; the period shares weight it below.
        annualized_inflow = group_inflow.mul(par.timestep_weight, axis=0).div(par.complete_period_share_of_year)

        # Flow-derived quantities (r.flow_dt, potentialVREgen_dt) and q_state
        # slacks are MW at the timestep.  Multiply by step_duration so the
        # stored timestep value is MWh/step — matching '5. Timestep inflow'
        # (par.node_inflow is already MWh/step).  This keeps all columns in
        # consistent energy units so shares and period aggregations work for
        # any step_duration.
        step_dur = par.step_duration

        # 3. VRE generation (actual flow) into the group's nodes
        vre_flow_sum = _group_vre_flow(par, s, r, group_nodes)

        # 4. Potential VRE generation and curtailment
        vre_processes = s.process_VRE.get_level_values('process')
        potential_cols = r.potentialVREgen_dt.columns[
            r.potentialVREgen_dt.columns.get_level_values(1).isin(group_nodes) &
            r.potentialVREgen_dt.columns.get_level_values(0).isin(vre_processes)
        ]
        potential_sum = r.potentialVREgen_dt[potential_cols].mul(step_dur, axis=0).sum(axis=1)
        curtailed_vre = (potential_sum - vre_flow_sum).clip(lower=0)

        # Filter balance nodes directly from the sets
        balance_set = set(s.node_balance) | set(s.node_balance_period)
        balance_nodes = [n for n in group_nodes if n in balance_set]

        # 5. Upward slack (MWh/step)
        if balance_nodes and not v.q_state_up.empty:
            # node_capacity_for_scaling is complete over nodeBalance ∪
            # nodeBalancePeriod (⊇ q_state nodes), so direct column indexing
            # is safe.
            upward_slack = v.q_state_up.mul(par.node_capacity_for_scaling[v.q_state_up.columns]).sum(axis=1).clip(lower=0)
            upward_slack = upward_slack.mul(step_dur)
        else:
            upward_slack = pd.Series(0, index=dt_index)

        # 6. Downward slack (MWh/step)
        if balance_nodes and not v.q_state_down.empty:
            downward_slack = v.q_state_down.mul(par.node_capacity_for_scaling[v.q_state_down.columns]).sum(axis=1).clip(lower=0)
            downward_slack = downward_slack.mul(step_dur)
        else:
            downward_slack = pd.Series(0, index=dt_index)

        # Multi-solve handoff: v.* / r.flow_dt span the full union of
        # sub-solve dt axes, while ``dt_realize_dispatch`` is the
        # post-roll "realized" subset.  Reindex every component series
        # to ``dt_index`` so the per-(period, time) DataFrame below
        # never raises "All arrays must be of the same length".
        upward_slack = upward_slack.reindex(dt_index, fill_value=0).fillna(0)
        downward_slack = downward_slack.reindex(dt_index, fill_value=0).fillna(0)
        vre_flow_sum = vre_flow_sum.reindex(dt_index, fill_value=0)
        potential_sum = potential_sum.reindex(dt_index, fill_value=0)
        curtailed_vre = curtailed_vre.reindex(dt_index, fill_value=0)
        group_inflow = group_inflow.reindex(dt_index, fill_value=0)
        annualized_inflow = annualized_inflow.reindex(dt_index, fill_value=0)

        # Step shares: curtailment against the potential, VRE against inflow
        curtailed_vre_share = _safe_ratio(curtailed_vre, potential_sum)
        vre_share = _safe_ratio(vre_flow_sum, group_inflow)

        # Combine timestep results for this group
        period_values = dt_index.get_level_values('period')
        group_result_dt = pd.DataFrame({
            'group': g,
            'period': period_values,
            'time': dt_index.get_level_values('time'),
            '1. Loss of load': upward_slack.values,
            '2. VRE generation': vre_flow_sum.values,
            '3. Excess load': downward_slack.values,
            '4. Curtailed VRE': curtailed_vre.values,
            '5. Timestep inflow': group_inflow.values,
            '6. Curtailed VRE of potential VRE': curtailed_vre_share.values,
            '7. Annualized inflow': annualized_inflow.values,
            '8. VRE share of demand': vre_share.values,
        })
        results_dt.append(group_result_dt)

        w = weight.values
        weighted_dt.append(pd.DataFrame({
            'group': g,
            'period': period_values,
            'lol': upward_slack.values * w,
            'vre': vre_flow_sum.values * w,
            'excess': downward_slack.values * w,
            'curt': curtailed_vre.values * w,
            'inflow': group_inflow.values * w,
            'pot': potential_sum.values * w,
        }))

    # Combine all groups for timestep level
    result_flat_dt = pd.concat(results_dt, ignore_index=True)

    # Create multi-index version for timestep level
    results_dt_indexed = result_flat_dt.set_index(['group', 'period', 'time'])[
        ['1. Loss of load', '2. VRE generation', '3. Excess load', '4. Curtailed VRE',
         '5. Timestep inflow', '6. Curtailed VRE of potential VRE', '7. Annualized inflow', '8. VRE share of demand']
    ]
    results_dt_indexed.columns.name = "parameter"

    results.append((results_dt_indexed, 'nodeGroup_gdt_p'))

    # Period level: ratios of the weighted period sums (see the docstring)
    sums_d = pd.concat(weighted_dt, ignore_index=True).groupby(['group', 'period']).sum()
    annualized_inflow_d = results_dt_indexed['7. Annualized inflow'].groupby(level=['group', 'period']).sum()

    results_d_indexed = pd.DataFrame({
        '1. Loss of load share': _safe_ratio(sums_d['lol'], sums_d['inflow']),
        '2. VRE share of demand': _safe_ratio(sums_d['vre'], sums_d['inflow']),
        '3. Excess load share': _safe_ratio(sums_d['excess'], sums_d['inflow']),
        '4. Curtailed VRE of demand': _safe_ratio(sums_d['curt'], sums_d['inflow']),
        '5. Annualized inflow': annualized_inflow_d,
        '6. Curtailed VRE of potential VRE': _safe_ratio(sums_d['curt'], sums_d['pot']),
    })
    results_d_indexed.columns.name = "parameter"

    results.append((results_d_indexed, 'nodeGroup_gd_p'))

    return results


def nodeGroup_VRE_share(par, s, v, r, debug):
    """VRE share of the inflow of each node group by time step and period.

    Time step: ``vre[d,t] / inflow[d,t]`` (0 where the inflow is 0).
    Period: ``Σ_t w[d,t]·vre[d,t] / Σ_t w[d,t]·inflow[d,t]`` with
    ``w = par.timestep_weight`` — a share of the period's inflow, not a mean
    of the step shares.  The VRE energy comes from ``_group_vre_flow`` (as
    for the indicators), so the period value equals '2. VRE share of demand'.
    """

    results = []

    # Get timesteps and groups
    timesteps = list(s.dt_realize_dispatch)

    # Filter groups that have nodes with inflow
    groups_with_inflow = [
        g for g in s.nodeGroupIndicators
        if _get_group_nodes_with_inflow(s, g)
    ]

    if not groups_with_inflow or not timesteps:
        index = pd.MultiIndex.from_tuples([], names=['period', 'time'])
        results_dt = pd.DataFrame(
            index=index,
            columns=pd.Index(groups_with_inflow, name='group'),
        )
        return [(results_dt, 'nodeGroup_VRE_share_dt_g')]

    # Create index
    index = pd.MultiIndex.from_tuples(timesteps, names=['period', 'time'])
    group_columns = pd.Index(groups_with_inflow, name='group')
    vre_dt = pd.DataFrame(index=index, columns=group_columns, dtype=float)
    inflow_dt = pd.DataFrame(index=index, columns=group_columns, dtype=float)

    for g in groups_with_inflow:
        group_nodes = (
            s.group_node[s.group_node.get_level_values('group') == g]
            .get_level_values('node')
            .tolist()
        )
        inflow_nodes = _get_group_nodes_with_inflow(s, g)
        node_cols = [col for col in par.node_inflow.columns.get_level_values(0) if col in inflow_nodes]
        # Inflow (demand) of the group's nodes [MWh/step], positive for demand
        inflow_dt[g] = (-par.node_inflow[node_cols].sum(axis=1)).reindex(index, fill_value=0.0)
        vre_dt[g] = _group_vre_flow(par, s, r, group_nodes).reindex(index, fill_value=0.0)

    results_dt = _safe_ratio(vre_dt, inflow_dt)
    results.append((results_dt, 'nodeGroup_VRE_share_dt_g'))

    weight = _realized_timestep_weight(par, index)
    vre_d = vre_dt.mul(weight, axis=0).groupby(level='period').sum()
    inflow_d = inflow_dt.mul(weight, axis=0).groupby(level='period').sum()
    results_d = _safe_ratio(vre_d, inflow_d)
    results_d.columns.name = 'group'
    results.append((results_d, 'nodeGroup_VRE_share_d_g'))

    return results


def nodeGroup_total_inflow(par, s, v, r, debug):
    """Total inflow (par.node_inflow) summed over nodes in each group."""

    results = []

    # Use node-based group membership (same as nodeGroup_VRE_share)
    groups = [
        g for g in s.nodeGroupIndicators
        if _get_group_nodes_with_inflow(s, g)
    ]

    if not groups:
        return results

    timesteps = list(s.dt_realize_dispatch)
    if not timesteps:
        return results

    # Calculate timestep-level results
    index = pd.MultiIndex.from_tuples(timesteps, names=['period', 'time'])
    result_multi_dt = pd.DataFrame(index=index, columns=groups, dtype=float)

    for g in groups:
        group_nodes = _get_group_nodes_with_inflow(s, g)
        node_cols = [col for col in par.node_inflow.columns.get_level_values(0)
                     if col in group_nodes]
        if node_cols:
            result_multi_dt[g] = par.node_inflow[node_cols].sum(axis=1)
        else:
            result_multi_dt[g] = 0.0

    result_multi_dt.columns.name = 'group'
    results.append((result_multi_dt, 'nodeGroup_flows_dt_g'))

    # Aggregate to period level.  Weight each (d, t) by par.timestep_weight
    # (=1.0 with no/uniform timeset_weights → byte-identical) before the
    # period sum.  result_multi_dt is group inflow (already MWh/step), so no
    # step_duration.
    result_multi_d = annualize_dt_to_d(result_multi_dt, par.timestep_weight, par.complete_period_share_of_year)
    result_multi_d.columns.name = 'group'
    results.append((result_multi_d, 'nodeGroup_flows_d_g'))

    return results


def nodeGroup_flows(par, s, v, r, debug):
    """Group output flows for periods and time.

    result_multi_dt combines sources that arrive in different units: slacks
    and inflow are already MWh/step, but the v_flow-derived columns
    (unit/connection flows, internal losses) are MW.  We multiply the
    latter by step_duration at assignment so every column is MWh/step at
    the timestep level.  The period-level output (nodeGroup_flows_d_gpe)
    is then divided by complete_period_share_of_year so it reports
    annualized MWh, matching node_d_ep and nodeGroup_flows_d_g.
    """

    results = []

    if s.nodeGroupDispatch.empty or s.dt_realize_dispatch.empty:
        return results

    step_dur = par.step_duration

    def _to_energy(df: pd.DataFrame) -> pd.DataFrame:
        """Multiply a (period, time)-indexed MW frame by step_duration → MWh/step.

        Guards against empty placeholder frames whose index lacks 'period'/'time'
        level names (pandas' align-on-index would raise).
        """
        if df.empty:
            return df
        return df.mul(step_dur, axis=0)

    # Calculate timestep-level results first
    result_multi_dt = pd.DataFrame(index=s.dt_realize_dispatch, columns=pd.MultiIndex.from_tuples([], names=['group', 'type', 'item']))

    # Assign simple mappings for all groups at once (before group loop)
    # Slack upward
    r.group_node_up_slack__dt.columns = pd.MultiIndex.from_arrays([
        r.group_node_up_slack__dt.columns,
        ['slack'] * len(r.group_node_up_slack__dt.columns),
        ['upward'] * len(r.group_node_up_slack__dt.columns)
    ], names=['group', 'type', 'item'])
    result_multi_dt[r.group_node_up_slack__dt.columns] = r.group_node_up_slack__dt

    # Unit aggregates (aggregateUnits to group)
    r.nodeGroupDispatch__group_aggregate_Unit_to_group__dt.columns = pd.MultiIndex.from_arrays([
        r.nodeGroupDispatch__group_aggregate_Unit_to_group__dt.columns.get_level_values('group'),
        ['from_unitGroup'] * len(r.nodeGroupDispatch__group_aggregate_Unit_to_group__dt.columns),
        r.nodeGroupDispatch__group_aggregate_Unit_to_group__dt.columns.get_level_values(1)
    ], names=['group', 'type', 'item'])
    result_multi_dt[r.nodeGroupDispatch__group_aggregate_Unit_to_group__dt.columns] = _to_energy(r.nodeGroupDispatch__group_aggregate_Unit_to_group__dt)

    # Units not in aggregate (unit to group) - sum across nodes
    r.nodeGroupDispatch__unit_to_node_not_in_aggregate__dt.columns = pd.MultiIndex.from_arrays([
        r.nodeGroupDispatch__unit_to_node_not_in_aggregate__dt.columns.get_level_values('group'),
        ['from_unit'] * len(r.nodeGroupDispatch__unit_to_node_not_in_aggregate__dt.columns),
        r.nodeGroupDispatch__unit_to_node_not_in_aggregate__dt.columns.get_level_values('process')
    ], names=['group', 'type', 'item'])
    result_multi_dt[r.nodeGroupDispatch__unit_to_node_not_in_aggregate__dt.columns] = _to_energy(r.nodeGroupDispatch__unit_to_node_not_in_aggregate__dt)

    # Connection aggregates (from connections to group) - sum across nodes
    r.nodeGroupDispatch__from_connection_aggregate__dt.columns = pd.MultiIndex.from_arrays([
        r.nodeGroupDispatch__from_connection_aggregate__dt.columns.get_level_values('group'),
        ['from_connectionGroup'] * len(r.nodeGroupDispatch__from_connection_aggregate__dt.columns),
        r.nodeGroupDispatch__from_connection_aggregate__dt.columns.get_level_values(1)
    ], names=['group', 'type', 'item'])
    result_multi_dt[r.nodeGroupDispatch__from_connection_aggregate__dt.columns] = _to_energy(r.nodeGroupDispatch__from_connection_aggregate__dt)

    # Connections not in aggregate (from connections)
    r.nodeGroupDispatch__from_connection_not_in_aggregate__dt.columns = pd.MultiIndex.from_arrays([
        r.nodeGroupDispatch__from_connection_not_in_aggregate__dt.columns.get_level_values('group'),
        ['from_connection'] * len(r.nodeGroupDispatch__from_connection_not_in_aggregate__dt.columns),
        r.nodeGroupDispatch__from_connection_not_in_aggregate__dt.columns.get_level_values('process')
    ], names=['group', 'type', 'item'])
    result_multi_dt[r.nodeGroupDispatch__from_connection_not_in_aggregate__dt.columns] = _to_energy(r.nodeGroupDispatch__from_connection_not_in_aggregate__dt)

    # Connections not in aggregate (to connections)
    r.nodeGroupDispatch__to_connection_not_in_aggregate__dt.columns = pd.MultiIndex.from_arrays([
        r.nodeGroupDispatch__to_connection_not_in_aggregate__dt.columns.get_level_values('group'),
        ['to_connection'] * len(r.nodeGroupDispatch__to_connection_not_in_aggregate__dt.columns),
        r.nodeGroupDispatch__to_connection_not_in_aggregate__dt.columns.get_level_values('process')
    ], names=['group', 'type', 'item'])
    result_multi_dt[r.nodeGroupDispatch__to_connection_not_in_aggregate__dt.columns] = -_to_energy(r.nodeGroupDispatch__to_connection_not_in_aggregate__dt)

    # Connection aggregates (to connections) - sum across nodes
    r.nodeGroupDispatch__to_connection_aggregate__dt.columns = pd.MultiIndex.from_arrays([
        r.nodeGroupDispatch__to_connection_aggregate__dt.columns.get_level_values('group'),
        ['to_connectionGroup'] * len(r.nodeGroupDispatch__to_connection_aggregate__dt.columns),
        r.nodeGroupDispatch__to_connection_aggregate__dt.columns.get_level_values(1)
    ], names=['group', 'type', 'item'])
    result_multi_dt[r.nodeGroupDispatch__to_connection_aggregate__dt.columns] = -_to_energy(r.nodeGroupDispatch__to_connection_aggregate__dt)

    # Group to aggregate units (negative)
    r.nodeGroupDispatch__group_aggregate_Group_to_unit__dt.columns = pd.MultiIndex.from_arrays([
        r.nodeGroupDispatch__group_aggregate_Group_to_unit__dt.columns.get_level_values('group'),
        ['to_unitGroup'] * len(r.nodeGroupDispatch__group_aggregate_Group_to_unit__dt.columns),
        r.nodeGroupDispatch__group_aggregate_Group_to_unit__dt.columns.get_level_values(1)
    ], names=['group', 'type', 'item'])
    result_multi_dt[r.nodeGroupDispatch__group_aggregate_Group_to_unit__dt.columns] = -_to_energy(r.nodeGroupDispatch__group_aggregate_Group_to_unit__dt)

    # Node to unit not in aggregate (negative)
    r.nodeGroupDispatch__node_to_unit_not_in_aggregate__dt.columns = pd.MultiIndex.from_arrays([
        r.nodeGroupDispatch__node_to_unit_not_in_aggregate__dt.columns.get_level_values('group'),
        ['to_unit'] * len(r.nodeGroupDispatch__node_to_unit_not_in_aggregate__dt.columns),
        r.nodeGroupDispatch__node_to_unit_not_in_aggregate__dt.columns.get_level_values('process')
    ], names=['group', 'type', 'item'])
    result_multi_dt[r.nodeGroupDispatch__node_to_unit_not_in_aggregate__dt.columns] = -_to_energy(r.nodeGroupDispatch__node_to_unit_not_in_aggregate__dt)

    # Inflow
    r.group_node_inflow_dt.columns = pd.MultiIndex.from_arrays([
        r.group_node_inflow_dt.columns,
        ['inflow'] * len(r.group_node_inflow_dt.columns),
        r.group_node_inflow_dt.columns
    ], names=['group', 'type', 'item'])
    result_multi_dt[r.group_node_inflow_dt.columns] = r.group_node_inflow_dt

    # Internal losses - connections (sum across processes, negate)
    if not r.nodeGroupDispatch_Internal_connection_losses__dt.empty:
        r.nodeGroupDispatch_Internal_connection_losses__dt = r.nodeGroupDispatch_Internal_connection_losses__dt.T.groupby('group').sum().T
        r.nodeGroupDispatch_Internal_connection_losses__dt.columns = pd.MultiIndex.from_arrays([
            r.nodeGroupDispatch_Internal_connection_losses__dt.columns,
            ['internal_losses'] * len(r.nodeGroupDispatch_Internal_connection_losses__dt.columns),
            ['connections'] * len(r.nodeGroupDispatch_Internal_connection_losses__dt.columns)
        ], names=['group', 'type', 'item'])
        result_multi_dt[r.nodeGroupDispatch_Internal_connection_losses__dt.columns] = _to_energy(r.nodeGroupDispatch_Internal_connection_losses__dt)

    # Internal losses - units (sum across processes, negate)
    if not r.nodeGroupDispatch_Internal_unit_losses__dt.empty:
        r.nodeGroupDispatch_Internal_unit_losses__dt = r.nodeGroupDispatch_Internal_unit_losses__dt.T.groupby('group').sum().T
        r.nodeGroupDispatch_Internal_unit_losses__dt.columns = pd.MultiIndex.from_arrays([
            r.nodeGroupDispatch_Internal_unit_losses__dt.columns,
            ['internal_losses'] * len(r.nodeGroupDispatch_Internal_unit_losses__dt.columns),
            ['units'] * len(r.nodeGroupDispatch_Internal_unit_losses__dt.columns)
        ], names=['group', 'type', 'item'])
        result_multi_dt[r.nodeGroupDispatch_Internal_unit_losses__dt.columns] = _to_energy(r.nodeGroupDispatch_Internal_unit_losses__dt)

    # Internal losses - storages (negate)
    r.group_node_state_losses__dt.columns = pd.MultiIndex.from_arrays([
        r.group_node_state_losses__dt.columns,
        ['internal_losses'] * len(r.group_node_state_losses__dt.columns),
        ['storages'] * len(r.group_node_state_losses__dt.columns)
    ], names=['group', 'type', 'item'])
    result_multi_dt[r.group_node_state_losses__dt.columns] = _to_energy(r.group_node_state_losses__dt)

    # Slack downward
    r.group_node_down_slack__dt.columns = pd.MultiIndex.from_arrays([
        r.group_node_down_slack__dt.columns,
        ['slack'] * len(r.group_node_down_slack__dt.columns),
        ['downward'] * len(r.group_node_down_slack__dt.columns)
    ], names=['group', 'type', 'item'])
    result_multi_dt[r.group_node_down_slack__dt.columns] = r.group_node_down_slack__dt

    # Tidy: drop all-zero (group, type, item) columns.  The inflow,
    # internal_losses storages and slack families are pre-allocated one
    # column per group (filled with 0.0 even when a group has no inflow /
    # no self-discharge node / no slack), unlike the join-built unit /
    # connection families which only appear when non-empty.  Without this
    # an ``elec`` group that is a single loss-free node still emits an
    # ``internal_losses | storages`` column of zeros, which reads as a
    # spurious participant in the plot legend.  ``result_multi_d`` is
    # derived from ``result_multi_dt`` below, so both stay consistent.
    if result_multi_dt.shape[1] > 0:
        result_multi_dt = result_multi_dt.loc[:, (result_multi_dt != 0).any(axis=0)]

    result_multi_dt.columns.names = ['group', 'type', 'item']
    result_multi_dt = result_multi_dt.sort_index(axis=1, level='group', sort_remaining=False)

    # Return timestep results
    results.append((result_multi_dt, 'nodeGroup_flows_dt_gpe'))

    # Aggregate to period level and annualize to match node_d_ep convention.
    # result_multi_dt is already energy (MWh/step); weight each (d, t) by
    # par.timestep_weight (=1.0 with no/uniform timeset_weights →
    # byte-identical) before the period sum.
    result_multi_d = annualize_dt_to_d(result_multi_dt, par.timestep_weight, par.complete_period_share_of_year)

    # Return period results
    results.append((result_multi_d, 'nodeGroup_flows_d_gpe'))

    return results
