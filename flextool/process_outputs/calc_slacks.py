import pandas as pd


def compute_slacks(par, s, v, r) -> None:
    """Compute slack and reserve quantities."""
    r.reserves_dt = v.reserve.mul(par.step_duration, axis=0)
    # Average MW held over the period: Σ(MW × step_duration) / total period hours.
    # Reserves are MW capacity held per hour (not consumed energy), so the
    # meaningful per-period figure is the average MW held across the period.
    period_hours = par.complete_period_share_of_year * 8760
    r.reserves_d = r.reserves_dt.groupby('period').sum().div(period_hours, axis=0)

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
        .mul(par.inflation_factor_operations_yearly, axis=0).sum(axis=1)

    r.q_reserves_dt = v.q_reserve.mul(par.reserve_upDown_group_reservation[v.q_reserve.columns], axis=1)
    r.q_reserves_d_not_annualized = r.q_reserves_dt.mul(par.step_duration, axis=0).groupby(level='period').sum()
    r.q_reserves_d = r.q_reserves_d_not_annualized.div(par.complete_period_share_of_year, axis=0)
    r.costPenalty_reserve_upDown_dt = v.q_reserve.mul(par.step_duration, axis=0) \
        .mul(par.reserve_upDown_group_penalty, axis=1) \
        .mul(par.reserve_upDown_group_reservation, axis=1)
