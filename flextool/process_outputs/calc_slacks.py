import pandas as pd

from flextool.process_outputs._annualize import annualize_dt_to_d


def inertia_limit_on_slack_cells(limit: pd.DataFrame, q_inertia: pd.DataFrame) -> pd.DataFrame:
    """``group_inertia_limit`` on the ``(period, time) × group`` cells of ``vq_inertia``, 0 where it has no value.

    The limit is per (period, time); a step without a limit (not authored, or authored 0) has no requirement, so
    its slack is 0 MWs rather than a blank cell.  The 0 must not stand in for a limit that is there but whose index
    does not line up with the slack's (other level names, or time labels of another type or spelling): that would
    report every shortfall as 0.  A group whose limit has a value but none on the slack's cells raises.
    """
    if limit.empty or q_inertia.empty:
        return limit.reindex(index=q_inertia.index, columns=q_inertia.columns).fillna(0.0)
    if list(limit.index.names) != list(q_inertia.index.names):
        raise ValueError(f"group_inertia_limit is indexed by {list(limit.index.names)}, the inertia slack by "
                         f"{list(q_inertia.index.names)}")
    on_cells = limit.reindex(index=q_inertia.index, columns=q_inertia.columns)
    lost = [g for g in q_inertia.columns.intersection(limit.columns)
            if limit[g].notna().any() and not on_cells[g].notna().any()]
    if lost:
        raise ValueError(f"group_inertia_limit of {lost} has values, but none at the (period, time) steps of the "
                         f"inertia slack: the indexes do not line up")
    return on_cells.fillna(0.0)


def compute_slacks(par, s, v, r) -> None:
    """Compute slack and reserve quantities.

    Cost-term conventions (must match the LP objective in flextool.mod):

    * **Per-timestep slack terms** (node state, inertia, non-sync, reserve)
      are cost amounts held across each step.  The mod multiplies them by
      ``step_duration[d, t] * p_rp_cost_weight[d, t] *
      p_inflation_factor_operations_yearly[d] /
      complete_period_share_of_year[d] * pdt_branch_weight[d, t]``.  The
      branch weight is not applied here: normal outputs hold realized rows
      only, and a realized ``(d, t)`` has branch weight 1.  Here
      we store the *per-(d, t)* contribution with ``step_duration`` and
      ``timestep_weight`` already baked in (so it can still be totalled
      per-step by ``costs__dt.csv``).  The remaining period-level factors
      (``inflation`` and ``/ period_share``) are applied at
      ``calc_costs.costPenalty_d`` where ``costPenalty_dt`` is aggregated.

    * **Capacity-margin slack** is a per-period decision (no timestep
      dimension).  The mod term is
      ``vq * group_capacity_for_scaling * penalty_capacity_margin * 1000 *
      p_inflation_factor_operations_yearly``.  No step_duration, no
      timestep_weight, no period_share division (it is already a per-period
      quantity).
    """
    # Per-step scaling factor for all per-(d, t) cost terms:
    # step_duration × timestep_weight.  Combined once for readability.
    step_x_rp = par.step_duration.mul(par.timestep_weight, axis=0)

    # Reserve held at each step, MW: v_reserve counts in units of the process's unit size, like v_flow
    # (model/_reserve.py: MW = v_reserve · unitsize[p]).  A capacity held, not an energy: no step_duration.
    r.reserves_dt = v.reserve.mul(par.entity_unitsize, axis=1, level='process')
    # Average MW held over the period: Σ(MW × step_duration) / total period hours.
    # Reserves are MW capacity held per hour (not consumed energy), so the
    # meaningful per-period figure is the average MW held across the period.
    period_hours = par.complete_period_share_of_year * 8760
    r.reserves_d = r.reserves_dt.mul(par.step_duration, axis=0).groupby('period').sum().div(period_hours, axis=0)

    # upward_node_slack_dt is already MWh per step (q_state_up × capacity × step_duration),
    # so the period aggregate is a plain groupby-sum; no extra step_duration.
    # node_capacity_for_scaling / node_penalty_* are complete over
    # nodeBalance ∪ nodeBalancePeriod (⊇ the q_state nodes) and solve-keyed
    # (concat-unioned), so direct column
    # indexing is safe — no missing-column densify needed here.
    # The annualised ``_d`` reporting series weights each (d, t) by
    # par.timestep_weight (=1.0 with no/uniform timeset_weights →
    # byte-identical) before the period sum, matching the cost-weighted
    # objective.  The ``_not_annualized`` diagnostic sums (used for the raw
    # Created/Removed slack-event reports) stay unweighted.
    r.upward_node_slack_dt = v.q_state_up.mul(par.node_capacity_for_scaling[v.q_state_up.columns]).mul(par.step_duration, axis=0)
    r.upward_node_slack_d_not_annualized = r.upward_node_slack_dt.groupby('period').sum()
    r.upward_node_slack_d = annualize_dt_to_d(r.upward_node_slack_dt, par.timestep_weight, par.complete_period_share_of_year)
    r.downward_node_slack_dt = v.q_state_down.mul(par.node_capacity_for_scaling[v.q_state_down.columns]).mul(par.step_duration, axis=0)
    r.downward_node_slack_d_not_annualized = r.downward_node_slack_dt.groupby('period').sum()
    r.downward_node_slack_d = annualize_dt_to_d(r.downward_node_slack_dt, par.timestep_weight, par.complete_period_share_of_year)
    # Node-state slack penalty: apply timestep_weight (step_duration already in _slack_dt).
    upward_node_penalty = r.upward_node_slack_dt.mul(par.node_penalty_up[v.q_state_up.columns]) \
                                               .mul(par.timestep_weight, axis=0)
    downward_node_penalty = r.downward_node_slack_dt.mul(par.node_penalty_down[v.q_state_down.columns]) \
                                                   .mul(par.timestep_weight, axis=0)
    r.costPenalty_node_state_upDown_dt = pd.concat([upward_node_penalty, downward_node_penalty], axis=1, keys=['up', 'down'], names=['upDown'])
    r.costPenalty_node_state_upDown_dt = r.costPenalty_node_state_upDown_dt.reorder_levels([1, 0], axis=1)
    r.costPenalty_node_state_upDown_d = r.costPenalty_node_state_upDown_dt.groupby(level='period').sum()

    # Inertia penalty: mod uses × step_duration × timestep_weight (objective line ~2380).
    # Per-step cost = q × inertia_limit × penalty × step_duration × timestep_weight.
    r.q_inertia_dt = v.q_inertia.mul(inertia_limit_on_slack_cells(par.group_inertia_limit, v.q_inertia))
    r.q_inertia_d_not_annualized = r.q_inertia_dt.mul(par.step_duration, axis=0).groupby('period').sum()
    r.q_inertia_d = annualize_dt_to_d(r.q_inertia_dt, par.timestep_weight, par.complete_period_share_of_year, par.step_duration)
    r.costPenalty_inertia_dt = r.q_inertia_dt.mul(par.group_penalty_inertia).mul(step_x_rp, axis=0)

    # Non-synchronous penalty: mod uses × step_duration × timestep_weight (objective line ~2382).
    r.q_non_synchronous_dt = v.q_non_synchronous.mul(par.group_capacity_for_scaling[s.groupNonSync])
    r.q_non_synchronous_d_not_annualized = r.q_non_synchronous_dt.mul(par.step_duration, axis=0) \
        .groupby('period').sum()
    r.q_non_synchronous_d = annualize_dt_to_d(
        r.q_non_synchronous_dt, par.timestep_weight,
        par.complete_period_share_of_year, par.step_duration)
    r.costPenalty_non_synchronous_dt = r.q_non_synchronous_dt \
        .mul(par.group_penalty_non_synchronous) \
        .mul(step_x_rp, axis=0)

    # Capacity-margin penalty: per-period event; no step_duration, no rp_weight.
    # Mod (~line 2414-2417): vq × group_capacity × penalty × 1000 × inflation.
    # Penalty is expressed per kW in the template; the ×1000 converts to /MW.
    r.q_capacity_margin_d_not_annualized = v.q_capacity_margin \
        .mul(par.group_capacity_for_scaling[s.groupCapacityMargin])
    r.costPenalty_capacity_margin_d = r.q_capacity_margin_d_not_annualized \
        .mul(par.group_penalty_capacity_margin[s.groupCapacityMargin]) \
        .mul(1000.0) \
        .mul(par.inflation_factor_operations_yearly, axis=0) \
        .sum(axis=1)

    # vq_reserve is a fraction of the group's largest possible requirement
    # (``reserve_upDown_group_shortfall_scale``, the same multiplier the LP
    # uses in the reserve balances and the penalty; the reservation for
    # timeseries groups).  ``reindex``, not ``[...]``: the output static view keeps the reserve groups of every
    # solve, while a solve that left reserves out (``reduce_system_detail_method``) has no scale for them; those
    # cells become NaN, and ``reduced_cells`` empties them anyway.
    r.q_reserves_dt = v.q_reserve.mul(
        par.reserve_upDown_group_shortfall_scale.reindex(columns=v.q_reserve.columns), axis=1)
    r.q_reserves_d_not_annualized = r.q_reserves_dt.mul(par.step_duration, axis=0).groupby(level='period').sum()
    r.q_reserves_d = annualize_dt_to_d(r.q_reserves_dt, par.timestep_weight, par.complete_period_share_of_year, par.step_duration)
    # Reserve slack penalty: step_duration is already here (line below);
    # add timestep_weight to match mod line ~2388.  ``shortfall_scale`` is the same multiplier the LP uses (it
    # equals the reservation for timeseries-only groups), so it is applied once here — not alongside
    # ``reservation``.
    r.costPenalty_reserve_upDown_dt = v.q_reserve.mul(par.step_duration, axis=0) \
        .mul(par.timestep_weight, axis=0) \
        .mul(par.reserve_upDown_group_penalty.reindex(v.q_reserve.columns), axis=1) \
        .mul(par.reserve_upDown_group_shortfall_scale, axis=1)
