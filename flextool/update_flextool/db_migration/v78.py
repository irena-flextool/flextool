"""Data version 78: a unit on every parameter description."""
from flextool.update_flextool.db_migration.common import _commit_step, _pdef_or_none


# ─── v78: a unit on every parameter description ────────────────────────
# Frozen tables (never read from the shipped schema).  The bracket at the
# start of a description is the unit of the value in the default
# convention (MW, MWh, CUR); non-numeric parameters carry a marker
# (``[choice]``, ``[yes/no]``, ``[name]``, ``[text]``, ``[count]``,
# ``[version number]``).

# Descriptions rewritten whole: the per-time-step meaning of inflow, the
# investment-solve adder, the bound without the stale commodity-tier use,
# the self-discharge decay, the solve texts that began with "Hours", and
# the reserve penalty (its v70 text said "per hour" after the unit too).
_V78_DESCRIPTIONS: dict[tuple[str, str], str] = {
    ('model', 'max_flow_for_unconstrained_variables'): (
        '[MW] Upper bound for capacities and flows that have no other limit: how much capacity an entity '
        'with `invest_method` `invest_no_limit` or `invest_retire_no_limit` (or `cumulative_limits` without '
        '`cumulative_max_capacity`) can add on top of its `existing` capacity, and the flow on a unit\'s input '
        'or output whose `conversion_flow_coeff` is 0. The value is used as given; with several model '
        'entities the largest value is used. Keep it large enough not to limit the real solution and small '
        'enough to avoid numerical problems. Default 1,000,000.'
    ),
    ('node', 'energy_margin_adder'): (
        '[MWh] per time step. Extra demand (negative inflow) added to every time step of the investment '
        'solve, when `energy_margin_method` = `inflow_adder`. The value is not a total that is divided over '
        'the steps: each step gets the full value. On 24 h steps, a margin of 10 MW needs a value of 240. '
        'FlexTool does not rescale it, including when the investment solve uses `new_stepduration`. Dispatch '
        'solves are not affected. 0 = off. Constant, period or period+time.'
    ),
    ('node', 'energy_margin_method'): (
        "[choice] How the node's investment-stage energy margin is applied. 'none' (default) = off, no "
        "scaling. 'inflow_multiplier' multiplies this node's inflow by 'energy_margin_multiplier' in the "
        "solve that carries investment periods only (dispatch solves are unaffected). 'inflow_adder' instead "
        "adds 'energy_margin_adder' MWh of extra demand (negative inflow) to each time step of that same "
        'investment solve. Both offset representative-period VRE optimism that would otherwise cause '
        'underinvestment.'
    ),
    ('node', 'inflow'): (
        '[MWh] per time step. Energy flowing into the node in each time step (negative is outflow, e.g. '
        'demand). The value is the energy of the whole step, not a rate: on 1 h steps it equals the average '
        'MW, and on 3 h steps it is 3 × the average MW. When the step length of your data changes, rescale '
        'the values yourself, e.g. sum hourly values to get 3 h steps, or halve MW values for 0.5 h steps. '
        'FlexTool does not convert them, and time-series values whose time names are not on the timeline are '
        "ignored. FlexTool sums the values for you only when a solve's `new_stepduration` coarsens the "
        "timeline, and when a group's `new_stepduration` balances the node per coarse step. A constant value "
        'is used as the same energy in every step. Constant, time or stochastic time.'
    ),
    ('node', 'peak_inflow'): (
        '[MWh] per time step. The highest inflow of a time step that the scaled inflow should reach. Use the '
        'same scale and sign as `inflow` (negative for a demand). Used only with `inflow_method` = '
        '`scale_to_annual_and_peak_flow`. If you change the step length, rescale it as you rescale `inflow`. '
        "Each period reaches its own `peak_inflow`; with a solve's `new_stepduration`, `peak_inflow` is "
        'matched per original time step. Constant or period.'
    ),
    ('node', 'self_discharge_loss'): (
        '[-] e.g. 0.01 means 1% per hour. Loss of stored energy over time: the content shrinks by the factor '
        '1/(1 + loss) every hour (with 0.01, by about 0.99% per hour), decaying continuously at ln(1 + loss) '
        'per hour. Constant, period, time or period+time.'
    ),
    ('reserve__upDown__group', 'penalty_reserve'): (
        '[CUR/MW per hour] Penalty for each MW of reserve shortfall (the reserve requirement of the group is '
        'soft). Constant. Default 5000 - applied to every reserve group that has no value of its own.'
    ),
    ('solve', 'new_stepduration'): (
        '[hours] Creates a new `timeline` from the old for this `solve` with this timestep duration. The new '
        'timeline sums `inflow` over the time steps that form each new time step (a constant `inflow` counts '
        'once per time step it combines) and averages the other time series (e.g. `profile`) over them, '
        "weighted by their durations. The `new_stepduration` must be a whole multiple of the timeline's time "
        'steps (on a timeline with time steps of different lengths, every new time step must end where a time '
        'step of the timeline ends); FlexTool stops with an error otherwise.'
    ),
    ('solve', 'rolling_duration'): (
        '[hours] Optional. Duration of rolling, if not stated, assumed to be the whole timeline of the solve'
    ),
    ('solve', 'rolling_solve_horizon'): (
        '[hours] Required if rolling_window solve. How long into the future the roll sees'
    ),
    ('solve', 'rolling_solve_jump'): (
        '[hours] Required if rolling_window solve. Interval between the start points of the rolls. Should be '
        'smaller than the horizon'
    ),
}

# Old bracket -> new bracket (with the example or default that follows it).
_V78_BRACKET_REWRITES: dict[tuple[str, str], tuple[str, str]] = {
    ('commodity', 'co2_content'): ('[CO2 ton per MWh]', '[tCO2/MWh]'),
    ('connection', 'availability'): ('[e.g. 0.9 means 90%]', '[-] e.g. 0.9 means 90%.'),
    ('connection', 'discount_rate'): ('[e.g. 0.05 equals 5%]', '[-] e.g. 0.05 means 5% per year.'),
    ('connection', 'efficiency'): ('[factor, typically 0-1]', '[-] e.g. 0.9 means 90%.'),
    ('connection', 'fixed_cost'): ('[CUR/kW]', '[CUR/kW/a]'),
    ('group', 'co2_price'): ('[CUR/ton]', '[CUR/tCO2]'),
    ('group', 'non_synchronous_limit'): ('[share, e.g. 0.8]', '[-] e.g. 0.8 means 80%.'),
    ('group', 'penalty_capacity_margin'): ('[CUR/kW]', '[CUR/kW/a]'),
    ('group', 'penalty_inertia'): ('[CUR/MWs]', '[CUR/MWs per hour]'),
    ('model', 'inflation_rate'): ('[e.g. 0.02 for 2%]', '[-] e.g. 0.02 means 2% per year.'),
    ('node', 'annual_flow'): ('[MWh]', '[MWh/a]'),
    ('node', 'availability'): ('[e.g. 0.9 means 90%]', '[-] e.g. 0.9 means 90%.'),
    ('node', 'discount_rate'): ('[e.g. 0.05 equals 5%]', '[-] e.g. 0.05 means 5% per year.'),
    ('node', 'energy_margin_multiplier'): ('[factor]', '[-] e.g. 1.1 means 10% more demand.'),
    ('node', 'fixed_cost'): ('[CUR/kWh]', '[CUR/kWh/a]'),
    ('node', 'storage_state_end'): ('[0-1]', '[-] 0-1, share of the storage capacity.'),
    ('node', 'storage_state_reference_value'): ('[0-1]', '[-] 0-1, share of the storage capacity.'),
    ('node', 'storage_state_start'): ('[0-1]', '[-] 0-1, share of the storage capacity.'),
    ('profile', 'profile'): ('[factor, typically 0-1]', '[-] e.g. 0.3 means 30% of capacity.'),
    ('reserve__upDown__connection__node', 'increase_reserve_ratio'): ('[factor]', '[-]'),
    ('reserve__upDown__connection__node', 'large_failure_ratio'): ('[factor]', '[-]'),
    ('reserve__upDown__connection__node', 'max_share'): ('[factor]', '[-]'),
    ('reserve__upDown__connection__node', 'reliability'): ('[factor]', '[-]'),
    ('reserve__upDown__group', 'increase_reserve_ratio'): ('[factor]', '[-]'),
    ('reserve__upDown__group', 'reserve_duration'): ('[h]', '[hours]'),
    ('reserve__upDown__unit__node', 'increase_reserve_ratio'): ('[factor]', '[-]'),
    ('reserve__upDown__unit__node', 'large_failure_ratio'): ('[factor]', '[-]'),
    ('reserve__upDown__unit__node', 'max_share'): ('[factor]', '[-]'),
    ('reserve__upDown__unit__node', 'reliability'): ('[factor]', '[-]'),
    ('solve', 'timeline_hole_multiplier'): ('[unitless]', '[-]'),
    ('unit', 'availability'): ('[e.g. 0.9 means 90%]', '[-] e.g. 0.9 means 90%.'),
    ('unit', 'discount_rate'): ('[e.g. 0.05 equals 5%]', '[-] e.g. 0.05 means 5% per year.'),
    ('unit', 'efficiency'): ('[factor]', '[-]'),
    ('unit', 'efficiency_at_min_load'): ('[e.g. 0.4 means 40%]', '[-] e.g. 0.4 means 40%.'),
    ('unit', 'fixed_cost'): ('[CUR/kW]', '[CUR/kW/a]'),
    ('unit', 'min_load'): ('[0-1]', '[-] 0-1, share of the online capacity.'),
    ('unit__inputNode', 'conversion_flow_coeff'): ('[factor, default 1]', '[-] default 1.'),
    ('unit__inputNode', 'input_share_max'): ('[factor, default 1.0]', '[-] default 1.'),
    ('unit__inputNode', 'input_share_min'): ('[factor, default 0]', '[-] default 0.'),
    ('unit__outputNode', 'capacity_max_coeff'): ('[factor, default 1.0]', '[-] default 1.'),
    ('unit__outputNode', 'capacity_min_coeff'): ('[factor, default 0]', '[-] default 0.'),
    ('unit__outputNode', 'conversion_flow_coeff'): ('[factor, default 1]', '[-] default 1.'),
}

# Older brackets of shipped templates that the steps before v78 left in
# place (a database created from such a template skipped the step that
# rewrote the text), old -> new.  Found by migrating a database made from
# every earlier spinedb_schema.json to v78 and comparing the brackets with
# the shipped schema: the v70-v75 templates kept capacity_min_coeff's
# pre-v70 bracket (and its old default 1.0, now 0), the first v77
# templates the bracket of `existing` without the unit-count reading.
_V78_OLDER_BRACKETS: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {
    ('connection', 'existing'): (('[MW]', '[MW or count]'),),
    ('node', 'existing'): (('[MWh]', '[MWh or count]'),),
    ('unit', 'existing'): (('[MW]', '[MW or count]'),),
    ('unit__outputNode', 'capacity_min_coeff'): (('[factor, default 1.0]', '[-] default 0.'),),
}

# Prepended to a description that has no bracket yet.
_V78_UNIT_PREFIXES: dict[tuple[str, str], str] = {
    ('commodity', 'price_ladder_annual'): '[CUR/MWh; MWh/a]',
    ('commodity', 'price_ladder_cumulative'): '[CUR/MWh; MWh]',
    ('commodity', 'price_method'): '[choice]',
    ('commodity', 'unitsize'): '[MWh]',
    ('connection', 'constraint_capacity_coeff'): '[unit of the constraint per MW]',
    ('connection', 'constraint_cumulative_pre_built_capacity_coeff'): '[unit of the constraint per MW]',
    ('connection', 'constraint_in_construction_capacity_coeff'): '[unit of the constraint per MW]',
    ('connection', 'constraint_invested_capacity_coeff'): '[unit of the constraint per MW]',
    ('connection', 'constraint_retired_capacity_coeff'): '[unit of the constraint per MW]',
    ('connection', 'construction_lead_time_method'): '[choice]',
    ('connection', 'invest_method'): '[choice]',
    ('connection', 'is_DC'): '[yes/no]',
    ('connection', 'lifetime_method'): '[choice]',
    ('connection', 'startup_method'): '[choice]',
    ('connection', 'transfer_method'): '[choice]',
    ('connection__node', 'constraint_flow_coeff'): '[unit of the constraint per MW]',
    ('connection__profile', 'profile_method'): '[choice]',
    ('constraint', 'constant'): "[the constraint's own unit]",
    ('constraint', 'is_enabled'): '[yes/no]',
    ('constraint', 'sense'): '[choice]',
    ('flowGroup', 'flow_aggregator'): '[choice]',
    ('group', 'capacity_margin_method'): '[choice]',
    ('group', 'co2_method'): '[choice]',
    ('group', 'decomposition_method'): '[choice]',
    ('group', 'has_inertia'): '[yes/no]',
    ('group', 'has_non_synchronous'): '[yes/no]',
    ('group', 'include_stochastics'): '[yes/no]',
    ('group', 'invest_method'): '[choice]',
    ('group', 'print_dispatch'): '[yes/no]',
    ('group', 'print_indicators'): '[yes/no]',
    ('group', 'reference_node'): '[name]',
    ('group', 'share_loss_of_load'): '[choice]',
    ('group', 'transfer_method'): '[choice]',
    ('group', 'use_for_representative_periods'): '[yes/no]',
    ('model', 'output_horizon'): '[yes/no]',
    ('model', 'periods_available'): '[name]',
    ('model', 'small_number_threshold'): '[unit of each LP coefficient]',
    ('model', 'solves'): '[name]',
    ('model', 'version'): '[version number]',
    ('node', 'constraint_capacity_coeff'): '[unit of the constraint per MWh]',
    ('node', 'constraint_cumulative_pre_built_capacity_coeff'): '[unit of the constraint per MWh]',
    ('node', 'constraint_in_construction_capacity_coeff'): '[unit of the constraint per MWh]',
    ('node', 'constraint_invested_capacity_coeff'): '[unit of the constraint per MWh]',
    ('node', 'constraint_retired_capacity_coeff'): '[unit of the constraint per MWh]',
    ('node', 'constraint_state_coeff'): '[unit of the constraint per MWh]',
    ('node', 'construction_lead_time_method'): '[choice]',
    ('node', 'inflow_method'): '[choice]',
    ('node', 'invest_method'): '[choice]',
    ('node', 'lifetime_method'): '[choice]',
    ('node', 'node_type'): '[choice]',
    ('node', 'penalty_method'): '[choice]',
    ('node', 'storage_binding_method'): '[choice]',
    ('node', 'storage_nested_fix_method'): '[choice]',
    ('node', 'storage_solve_horizon_method'): '[choice]',
    ('node', 'storage_start_end_method'): '[choice]',
    ('node__profile', 'profile_method'): '[choice]',
    ('reserve__upDown__connection__node', 'is_enabled'): '[yes/no]',
    ('reserve__upDown__group', 'reserve_method'): '[choice]',
    ('reserve__upDown__unit__node', 'is_enabled'): '[yes/no]',
    ('solve', 'benders_in_out_weight'): '[-]',
    ('solve', 'benders_max_iter'): '[count]',
    ('solve', 'benders_tolerance'): '[-]',
    ('solve', 'contains_solves'): '[name]',
    ('solve', 'decomposition'): '[choice]',
    ('solve', 'dispatch_realization_method'): '[choice]',
    ('solve', 'fix_storage_periods'): '[name]',
    ('solve', 'invest_periods'): '[name]',
    ('solve', 'period_timeset'): '[name]',
    ('solve', 'realized_invest_periods'): '[name]',
    ('solve', 'realized_periods'): '[name]',
    ('solve', 'scaling'): '[choice]',
    ('solve', 'shared_invest_periods'): '[name]',
    ('solve', 'shared_operation_periods'): '[name]',
    ('solve', 'solve_mode'): '[choice]',
    ('solve', 'solver'): '[choice]',
    ('solve', 'solver_arguments'): '[text]',
    ('solve', 'solver_mip_gap'): '[-]',
    ('solve', 'solver_precommand'): '[text]',
    ('solve', 'stochastic_branches'): '[-]',
    ('solve', 'stochastic_invest_method'): '[choice]',
    ('solve', 'years_represented'): '[years]',
    ('timeline', 'timestep_duration'): '[hours]',
    ('timeset', 'representative_period_weights'): '[-]',
    ('timeset', 'timeline'): '[name]',
    ('timeset', 'timeset_duration'): '[count of time steps]',
    ('timeset', 'timeset_weights'): '[-]',
    ('unit', 'constraint_capacity_coeff'): '[unit of the constraint per MW]',
    ('unit', 'constraint_cumulative_pre_built_capacity_coeff'): '[unit of the constraint per MW]',
    ('unit', 'constraint_in_construction_capacity_coeff'): '[unit of the constraint per MW]',
    ('unit', 'constraint_invested_capacity_coeff'): '[unit of the constraint per MW]',
    ('unit', 'constraint_retired_capacity_coeff'): '[unit of the constraint per MW]',
    ('unit', 'construction_lead_time_method'): '[choice]',
    ('unit', 'conversion_method'): '[choice]',
    ('unit', 'invest_method'): '[choice]',
    ('unit', 'lifetime_method'): '[choice]',
    ('unit', 'minimum_time_method'): '[choice]',
    ('unit', 'startup_method'): '[choice]',
    ('unit__inputNode', 'constraint_flow_coeff'): '[unit of the constraint per MW]',
    ('unit__inputNode', 'is_non_synchronous'): '[yes/no]',
    ('unit__inputNode', 'ramp_method'): '[choice]',
    ('unit__node__profile', 'profile_method'): '[choice]',
    ('unit__outputNode', 'constraint_flow_coeff'): '[unit of the constraint per MW]',
    ('unit__outputNode', 'is_non_synchronous'): '[yes/no]',
    ('unit__outputNode', 'ramp_method'): '[choice]',
}


def _v78_new_description(cls: str, name: str, description: str) -> str:
    """The v78 description of one definition (unchanged when nothing applies)."""
    text = _V78_DESCRIPTIONS.get((cls, name))
    if text is not None:
        return text
    rewrites = ((_V78_BRACKET_REWRITES[(cls, name)],) if (cls, name) in _V78_BRACKET_REWRITES else ()) \
        + _V78_OLDER_BRACKETS.get((cls, name), ())
    for old, new in rewrites:
        if description == old or description.startswith(old + " "):
            return new + description[len(old):]
    prefix = _V78_UNIT_PREFIXES.get((cls, name))
    if prefix is not None and not description.startswith("["):
        return f"{prefix} {description}" if description else prefix
    return description


def _migrate_v78_unit_prefixes(db) -> None:
    """A unit on every parameter description (v77 -> v78).

    * :data:`_V78_DESCRIPTIONS` replace the whole description.
    * :data:`_V78_BRACKET_REWRITES` replace an exact old bracket at the start
      (example-style and ``[factor…]`` brackets, the per-year and per-hour
      prices, one spelling per unit); :data:`_V78_OLDER_BRACKETS` do the
      same for the older brackets some shipped templates still had.
    * :data:`_V78_UNIT_PREFIXES` go in front of a description that does not
      start with ``[``.

    Definitions a database does not have are skipped.  Idempotent: a
    rerun finds every description already in its v78 form.
    """
    keys = (set(_V78_DESCRIPTIONS) | set(_V78_BRACKET_REWRITES) | set(_V78_OLDER_BRACKETS)
            | set(_V78_UNIT_PREFIXES))
    for cls, name in sorted(keys):
        pdef = _pdef_or_none(db, cls, name)
        if pdef is None:
            continue
        current = pdef["description"] or ""
        new = _v78_new_description(cls, name, current)
        if new != current:
            db.update_parameter_definition(id=pdef["id"], description=new)
    _commit_step(db, "v78: a unit on every parameter description")
