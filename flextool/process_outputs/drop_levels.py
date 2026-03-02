from types import SimpleNamespace


# Variables: droplevel('solve') only
_V_DROP = [
    'flow', 'ramp', 'reserve', 'state',
    'online_linear', 'startup_linear', 'shutdown_linear',
    'online_integer', 'startup_integer', 'shutdown_integer',
    'q_state_up', 'q_state_down', 'q_reserve',
    'q_inertia', 'q_non_synchronous', 'q_state_up_group', 'q_capacity_margin',
    'invest', 'divest',
    'dual_invest_connection', 'dual_invest_node', 'dual_invest_unit',
]

# Parameters: droplevel('solve') only
_PAR_DROP = [
    'step_duration', 'flow_min', 'flow_max',
    'process_availability', 'process_source_sink_varCost',
    'process_slope', 'process_section',
    'node_self_discharge_loss', 'node_penalty_up', 'node_penalty_down',
    'node_inflow', 'commodity_price', 'group_co2_price',
    'reserve_upDown_group_reservation', 'profile',
    'entity_annual_discounted', 'entity_annual_divest_discounted',
    'discount_factor_investment_yearly',
    'group_penalty_capacity_margin', 'group_capacity_margin',
]

# Parameters: droplevel('solve') + deduplicate index
_PAR_DEDUP = [
    'years_from_start_d', 'years_represented_d',
    'entity_max_units', 'entity_all_existing', 'entity_pre_existing',
    'process_startup_cost',
    'entity_fixed_cost', 'entity_lifetime_fixed_cost', 'entity_lifetime_fixed_cost_divest',
    'node_annual_flow',
    'group_penalty_inertia', 'group_penalty_non_synchronous',
    'group_inertia_limit',
    'discount_factor_operations_yearly',
    'node_capacity_for_scaling', 'group_capacity_for_scaling',
    'complete_period_share_of_year',
]


def drop_levels(par: SimpleNamespace, s: SimpleNamespace, v: SimpleNamespace):
    """Strip the 'solve' level from all time-indexed variables, parameters and sets.

    Must be called before any post-processing calculations.
    Returns (par, s, v) with the solve level removed in-place.
    """
    for attr in _V_DROP:
        obj = getattr(v, attr)
        setattr(v, attr, obj.droplevel('solve'))

    for attr in _PAR_DROP:
        obj = getattr(par, attr)
        setattr(par, attr, obj.droplevel('solve'))

    for attr in _PAR_DEDUP:
        obj = getattr(par, attr)
        obj = obj.droplevel('solve')
        setattr(par, attr, obj[~obj.index.duplicated(keep='first')])

    # Sets have varied special handling so are done individually
    s.solve_period = s.period
    # Save per-timestep solve mapping before dropping (for correct re-join in CSV output)
    s.solve_period_time = s.dt_realize_dispatch
    s.period = s.period.droplevel('solve')
    s.period__time_first = s.period__time_first.droplevel('solve')
    s.period_first_of_solve = s.period_first_of_solve.droplevel('solve')
    s.period_in_use = s.period_in_use.droplevel('solve').unique()
    s.d_realize_dispatch_or_invest = s.d_realize_dispatch_or_invest.droplevel('solve').unique()
    s.d_realize_invest = s.d_realize_invest.droplevel('solve')
    s.d_realized_period = s.d_realized_period.droplevel('solve').unique()
    s.dt = s.dt.droplevel('solve')
    s.dt_fix_storage_timesteps = s.dt_fix_storage_timesteps.droplevel('solve')
    s.dt_realize_dispatch = s.dt_realize_dispatch.droplevel('solve')
    s.dtt = s.dtt.droplevel('solve')
    s.dtttdt = s.dtttdt.droplevel('solve')
    s.ed_invest = s.ed_invest.droplevel('solve').join(s.d_realize_invest, how='inner')
    s.edd_invest = s.edd_invest.droplevel('solve')
    s.edd_invest.names = ['entity', 'period_invest', 'period']
    s.edd_invest = s.edd_invest.join(s.d_realize_invest, how='inner')
    s.ed_divest = s.ed_divest.droplevel('solve').join(s.d_realize_invest, how='inner')

    return par, s, v
