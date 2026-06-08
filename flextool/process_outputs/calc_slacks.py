import pandas as pd


def compute_slacks(par, s, v, r) -> None:
    """Compute slack and reserve quantities.

    Cost-term conventions (must match the LP objective in flextool.mod):

    * **Per-timestep slack terms** (node state, inertia, non-sync, reserve)
      are cost amounts held across each step.  The mod multiplies them by
      ``step_duration[d, t] * p_rp_cost_weight[d, t] *
      p_inflation_factor_operations_yearly[d] /
      complete_period_share_of_year[d] * pdt_branch_weight[d, t]``.  Here
      we store the *per-(d, t)* contribution with ``step_duration`` and
      ``rp_cost_weight`` already baked in (so it can still be totalled
      per-step by ``costs__dt.csv``).  The remaining period-level factors
      (``inflation`` and ``/ period_share``) are applied at
      ``calc_costs.costPenalty_d`` where ``costPenalty_dt`` is aggregated.

    * **Capacity-margin slack** is a per-period decision (no timestep
      dimension).  The mod term is
      ``vq * group_capacity_for_scaling * penalty_capacity_margin * 1000 *
      p_inflation_factor_operations_yearly``.  No step_duration, no
      rp_cost_weight, no period_share division (it is already a per-period
      quantity).
    """
    # Per-step scaling factor for all per-(d, t) cost terms:
    # step_duration × rp_cost_weight.  Combined once for readability.
    step_x_rp = par.step_duration.mul(par.rp_cost_weight, axis=0)

    r.reserves_dt = v.reserve.mul(par.step_duration, axis=0)
    # Average MW held over the period: Σ(MW × step_duration) / total period hours.
    # Reserves are MW capacity held per hour (not consumed energy), so the
    # meaningful per-period figure is the average MW held across the period.
    period_hours = par.complete_period_share_of_year * 8760
    r.reserves_d = r.reserves_dt.groupby('period').sum().div(period_hours, axis=0)

    # upward_node_slack_dt is already MWh per step (q_state_up × capacity × step_duration),
    # so the period aggregate is a plain groupby-sum; no extra step_duration.
    # node_capacity_for_scaling / node_penalty_* are complete over nodeBalance
    # (⊇ the q_state nodes) and solve-keyed (concat-unioned), so direct column
    # indexing is safe — no missing-column densify needed here.
    # The annualised ``_d`` reporting series weights each (d, t) by
    # par.rp_cost_weight (=1.0 with no/uniform timeset_weights →
    # byte-identical) before the period sum, matching the cost-weighted
    # objective.  The ``_not_annualized`` diagnostic sums (used for the raw
    # Created/Removed slack-event reports) stay unweighted.
    r.upward_node_slack_dt = v.q_state_up.mul(par.node_capacity_for_scaling[v.q_state_up.columns]).mul(par.step_duration, axis=0)
    r.upward_node_slack_d_not_annualized = r.upward_node_slack_dt.groupby('period').sum()
    r.upward_node_slack_d = r.upward_node_slack_dt.mul(par.rp_cost_weight, axis=0).groupby('period').sum().div(par.complete_period_share_of_year, axis=0)
    r.downward_node_slack_dt = v.q_state_down.mul(par.node_capacity_for_scaling[v.q_state_down.columns]).mul(par.step_duration, axis=0)
    r.downward_node_slack_d_not_annualized = r.downward_node_slack_dt.groupby('period').sum()
    r.downward_node_slack_d = r.downward_node_slack_dt.mul(par.rp_cost_weight, axis=0).groupby('period').sum().div(par.complete_period_share_of_year, axis=0)
    # Node-state slack penalty: apply rp_cost_weight (step_duration already in _slack_dt).
    upward_node_penalty = r.upward_node_slack_dt.mul(par.node_penalty_up[v.q_state_up.columns]) \
                                               .mul(par.rp_cost_weight, axis=0)
    downward_node_penalty = r.downward_node_slack_dt.mul(par.node_penalty_down[v.q_state_down.columns]) \
                                                   .mul(par.rp_cost_weight, axis=0)
    r.costPenalty_node_state_upDown_dt = pd.concat([upward_node_penalty, downward_node_penalty], axis=1, keys=['up', 'down'], names=['upDown'])
    r.costPenalty_node_state_upDown_dt = r.costPenalty_node_state_upDown_dt.reorder_levels([1, 0], axis=1)
    r.costPenalty_node_state_upDown_d = r.costPenalty_node_state_upDown_dt.groupby(level='period').sum()

    # Inertia penalty: mod uses × step_duration × rp_cost_weight (objective line ~2380).
    # Per-step cost = q × inertia_limit × penalty × step_duration × rp_cost_weight.
    r.q_inertia_dt = v.q_inertia.mul(par.group_inertia_limit)
    r.q_inertia_d_not_annualized = r.q_inertia_dt.mul(par.step_duration, axis=0).groupby('period').sum()
    r.q_inertia_d = r.q_inertia_dt.mul(par.step_duration, axis=0).mul(par.rp_cost_weight, axis=0).groupby('period').sum().div(par.complete_period_share_of_year, axis=0)
    r.costPenalty_inertia_dt = r.q_inertia_dt.mul(par.group_penalty_inertia).mul(step_x_rp, axis=0)

    # Non-synchronous penalty: mod uses × step_duration × rp_cost_weight (objective line ~2382).
    r.q_non_synchronous_dt = v.q_non_synchronous.mul(par.group_capacity_for_scaling[s.groupNonSync])
    r.q_non_synchronous_d_not_annualized = r.q_non_synchronous_dt.mul(par.step_duration, axis=0) \
        .groupby('period').sum()
    r.q_non_synchronous_d = r.q_non_synchronous_dt.mul(par.step_duration, axis=0) \
        .mul(par.rp_cost_weight, axis=0).groupby('period').sum() \
        .div(par.complete_period_share_of_year, axis=0)
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

    r.q_reserves_dt = v.q_reserve.mul(par.reserve_upDown_group_reservation[v.q_reserve.columns], axis=1)
    r.q_reserves_d_not_annualized = r.q_reserves_dt.mul(par.step_duration, axis=0).groupby(level='period').sum()
    r.q_reserves_d = r.q_reserves_dt.mul(par.step_duration, axis=0).mul(par.rp_cost_weight, axis=0).groupby(level='period').sum().div(par.complete_period_share_of_year, axis=0)
    # Reserve slack penalty: step_duration is already here (line below);
    # add rp_cost_weight to match mod line ~2388.
    r.costPenalty_reserve_upDown_dt = v.q_reserve.mul(par.step_duration, axis=0) \
        .mul(par.rp_cost_weight, axis=0) \
        .mul(par.reserve_upDown_group_penalty, axis=1) \
        .mul(par.reserve_upDown_group_reservation, axis=1)
