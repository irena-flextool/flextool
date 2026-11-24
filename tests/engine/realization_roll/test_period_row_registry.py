"""Pins of ``PERIOD_ROW_KINDS`` (G1, ``model._period_rows``) against the autoscale constraint registry.

``CONSTRAINT_FAMILIES`` is complete by invariant 1 (every ``add_cstr`` family is registered for autoscale), so a
new constraint family fails here until it is classified: add it to ``PERIOD_ROW_KINDS``, or to
``REVIEWED_STEP_LEVEL`` after checking that its ``over`` has a step column (``t`` / ``b_first``) or no ``d``, and
that its left-hand side sums no period.  The lp_snapshot capture pins the same on every corpus build, from the
``over`` frames themselves.
"""
from __future__ import annotations

from collections import Counter

from flextool.engine.autoscale._layer2_types import CONSTRAINT_FAMILIES, lookup_cstr, resolve_cstr_key
from flextool.engine.model._period_rows import PERIOD_ROW_KINDS, RP_PREFIX

# The CONSTRAINT_FAMILIES keys no PERIOD_ROW_KINDS name resolves to (rp_inter_period_* aside), reviewed: each
# family's over has a step column (t, b_first) or no d (solve totals, leaf paths, the Benders master).
REVIEWED_STEP_LEVEL = frozenset({
    "capacityMargin", "capacityMargin_coarse", "co2_max_total", "co2_max_total_path", "connection_reserve_down",
    "connection_reserve_down_coarse", "connection_reserve_down_exact", "connection_reserve_down_exact_coarse",
    "connection_reserve_down_online", "connection_reserve_down_online_integer_coarse",
    "connection_reserve_down_online_linear_coarse", "connection_reserve_up", "connection_reserve_up_coarse",
    "connection_reserve_up_exact", "connection_reserve_up_exact_coarse", "connection_reserve_up_online",
    "connection_reserve_up_online_integer_coarse", "connection_reserve_up_online_linear_coarse",
    "conversion_indirect", "conversion_indirect_coarse", "conversion_piecewise", "conversion_piecewise_coarse",
    "dc_flow_eq", "dc_flow_eq_coarse", "dc_reference_angle_eq", "inertia_constraint", "inertia_constraint_coarse",
    "ladder_tier_cap_cumulative_roll", "ladder_tier_cap_cumulative_roll_path", "maxCumulative_flow_solve",
    "maxCumulative_flow_solve_path", "maxDivestGroup_entity_total", "maxDivestGroup_entity_total_path",
    "maxDivest_entity_total", "maxDivest_entity_total_n", "maxDivest_entity_total_path",
    "maxDivest_entity_total_path_n", "maxFlow", "maxFlow_back", "maxFlow_back_coarse", "maxFlow_back_online",
    "maxFlow_back_online_integer_coarse", "maxFlow_back_online_linear_coarse", "maxFlow_coarse",
    "maxFlow_exact_forward", "maxFlow_exact_forward_coarse", "maxFlow_exact_reverse", "maxFlow_exact_reverse_coarse",
    "maxFlow_negCap", "maxFlow_negCap_coarse", "maxFlow_online", "maxFlow_online_integer_coarse",
    "maxFlow_online_linear_coarse", "maxFlow_online_shutdown", "maxFlow_online_shutdown_branch",
    "maxFlow_online_shutdown_branch_integer_coarse", "maxFlow_online_shutdown_branch_linear_coarse",
    "maxFlow_online_shutdown_integer_coarse", "maxFlow_online_shutdown_linear_coarse", "maxInputFlow_online",
    "maxInputFlow_online_integer_coarse", "maxInputFlow_online_linear_coarse", "maxInputFlow_online_shutdown",
    "maxInputFlow_online_shutdown_branch", "maxInputFlow_online_shutdown_branch_integer_coarse",
    "maxInputFlow_online_shutdown_branch_linear_coarse", "maxInputFlow_online_shutdown_integer_coarse",
    "maxInputFlow_online_shutdown_linear_coarse", "maxInstant_flow", "maxInstant_flow_coarse", "maxInvest",
    "maxInvestGroup_entity_total", "maxInvestGroup_entity_total_path", "maxInvest_entity_total",
    "maxInvest_entity_total_n", "maxInvest_entity_total_path", "maxInvest_entity_total_path_n", "maxOnline",
    "maxOutputSum", "maxOutputSum_coarse", "maxOutputSum_negCap", "maxOutputSum_negCap_coarse",
    "maxOutputSum_online", "maxOutputSum_online_integer_coarse", "maxOutputSum_online_linear_coarse", "maxShutdown",
    "maxStartup", "maxState", "maxState_rp_start", "minCumulative_flow_solve", "minCumulative_flow_solve_path",
    "minDivestGroup_entity_total_path", "minDivest_entity_total_path", "minFlow_minload",
    "minFlow_minload_integer_coarse", "minFlow_minload_linear_coarse", "minFlow_output_floor",
    "minFlow_output_floor_integer_coarse", "minFlow_output_floor_linear_coarse", "minFlow_reserve",
    "minFlow_reserve_arc", "minFlow_reserve_arc_coarse", "minFlow_reserve_coarse", "minInputShare",
    "minInputShare_coarse", "minInstant_flow", "minInstant_flow_coarse", "minInvestGroup_entity_total_path",
    "minInvest_entity_total_path", "minimum_downtime", "minimum_uptime", "nodeBalanceBlock_eq",
    "nodeBalanceCoarse_eq", "nodeBalance_eq", "node_balance_fix_quantity_eq_lower",
    "non_anticipativity_online_integer", "non_anticipativity_online_linear", "non_anticipativity_reserve",
    "non_anticipativity_rp_block_start", "non_anticipativity_storage_state_first", "non_anticipativity_storage_use",
    "non_anticipativity_storage_use_coarse", "non_sync_constraint", "non_sync_constraint_coarse", "online_change",
    "online_fix_eq", "process_constraint_equal", "process_constraint_equal_coarse",
    "process_constraint_greater_than", "process_constraint_greater_than_coarse", "process_constraint_less_than",
    "process_constraint_less_than_coarse", "profile_flow_fixed", "profile_flow_fixed_coarse",
    "profile_flow_lower_limit", "profile_flow_lower_limit_coarse", "profile_flow_upper_limit",
    "profile_flow_upper_limit_coarse", "profile_state_fixed", "profile_state_lower_limit",
    "profile_state_upper_limit", "ramp", "ramp_change_eq", "ramp_sink_down_reserve_coarse",
    "ramp_sink_up_reserve_coarse", "ramp_source_down_reserve_coarse", "ramp_source_up_reserve_coarse",
    "reserveBalance_down_n_1_eq", "reserveBalance_dynamic_eq", "reserveBalance_timeseries_eq",
    "reserveBalance_up_n_1_eq", "reserve_process_downward", "reserve_process_downward_coarse",
    "reserve_process_upward", "reserve_process_upward_coarse", "reserve_storage_down_headroom",
    "reserve_storage_down_headroom_coarse", "reserve_storage_up_floor", "reserve_storage_up_floor_coarse",
    "reserve_total_upper", "stateConstantCoarse_eq", "stateConstantWithinBlock_eq", "storage_state_end_binding",
    "storage_state_end_match", "storage_state_end_match_share", "storage_state_fan_start",
    "storage_state_roll_anchor", "storage_state_roll_anchor_share", "storage_state_roll_close",
    "storage_state_roll_close_share", "storage_state_roll_match", "storage_state_solve_horizon_reference_value",
    "storage_state_start_binding_cyclic_period", "storage_within_solve_path_end", "trade_capacity",
})


def test_every_period_row_name_is_a_registered_family():
    for name in PERIOD_ROW_KINDS:
        lookup_cstr(name)   # KeyError on a rename
    # Design rev 7.1 section 2.4: 8 path, 1 path_at_step, 1 fragment, 48 invest names.
    assert Counter(PERIOD_ROW_KINDS.values()) == {"path": 8, "path_at_step": 1, "fragment": 1, "invest": 48}
    assert any(k.startswith(RP_PREFIX) for k in CONSTRAINT_FAMILIES)


def test_every_other_family_is_reviewed():
    classified = {resolve_cstr_key(n) for n in PERIOD_ROW_KINDS}
    rest = {k for k in CONSTRAINT_FAMILIES if k not in classified and not k.startswith(RP_PREFIX)}
    new = sorted(rest - REVIEWED_STEP_LEVEL)
    assert not new, (f"constraint families {new}: add each to PERIOD_ROW_KINDS (model/_period_rows.py), or to "
                     "REVIEWED_STEP_LEVEL after checking that its over has a step column and its LHS sums no period")
    assert not REVIEWED_STEP_LEVEL - rest, sorted(REVIEWED_STEP_LEVEL - rest)   # stale or now classified
