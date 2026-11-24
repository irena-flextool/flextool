"""Period-level rows and branch weights under a realization roll: the checks G1 and G2 of ``build_flextool``.

Under a split realization roll the base period ``p`` (a period label with steps) dispatches its jump and each branch
period ``p_f`` the steps after it.  A *period-level* row (one per period, summing every step of the period) built
per period name then holds only a fragment of the path ``p`` + ``p_f``.  G1 makes such a row an internal error
instead of a different LP (default-deny):

* a row on a label with steps of a ``path``, ``path_at_step``, ``invest``, ``rp`` or unclassified period-level
  family;
* a row on a branch period of such a label of a ``path`` / ``path_at_step`` family that the path helper has not
  converted (``_Build.path_converted``).

A family is period-level when its ``over`` has ``d`` and no step column, or when ``PERIOD_ROW_KINDS`` names it,
whatever its columns.  G2: every (d, t) cell of the LP has a branch weight > 0.  Without a label with steps (every
solve but a split realization roll) G1 does nothing and G2 holds.
"""
from __future__ import annotations

import polars as pl

from flextool.engine.autoscale._guard import _strict
from flextool.engine.preprocess._step_relations import labels_with_steps, not_path_aware, unregistered_step_frames

PATH, PATH_AT_STEP, FRAGMENT, INVEST, RP = "path", "path_at_step", "fragment", "invest", "rp"
RP_PREFIX = "rp_inter_period_"   # every rp_inter_period_* family is kind ``rp`` (refused in rolls)
STEP_COLUMNS = frozenset({"t", "b_first"})

_INVEST_NAMES = (
    "capacityMargin_slack_energy",
    "divestedBuilt_n", "divestedBuilt_p",
    "extendedCapacity_n", "extendedCapacity_p",
    "maxExtend_cohort_n", "maxExtend_cohort_p",
    "non_anticipativity_extend_n", "non_anticipativity_extend_p",
    "maxCumulative_capacity_n", "maxCumulative_capacity_p",
    "minCumulative_capacity_n", "minCumulative_capacity_p",
    "maxInvestGroup_entity_cumulative", "minInvestGroup_entity_cumulative",
    "maxInvestGroup_entity_period_n", "maxInvestGroup_entity_period_p",
    "minInvestGroup_entity_period_n", "minInvestGroup_entity_period_p",
    "maxDivestGroup_entity_period_n", "maxDivestGroup_entity_period_p",
    "minDivestGroup_entity_period_n", "minDivestGroup_entity_period_p",
    "maxInvest_entity_period_n", "maxInvest_entity_period_p",
    "minInvest_entity_period_n", "minInvest_entity_period_p",
    "maxDivest_entity_period_n", "maxDivest_entity_period_p",
    "minDivest_entity_period_n", "minDivest_entity_period_p",
    "maxDivest_cohort_n", "maxDivest_cohort_p",
    "maxDivest_var_bound", "maxDivest_var_bound_n",
    "maxInvest_var_bound", "maxInvest_var_bound_n",
    "maxRetired_existing_n", "maxRetired_existing_p",
    "non_anticipativity_divest_built_n", "non_anticipativity_divest_built_p",
    "non_anticipativity_divest_n", "non_anticipativity_divest_p",
    "non_anticipativity_invest_n", "non_anticipativity_invest_p",
    # rows of invest and capacity terms only (core._eq_user_constraints)
    "process_constraint_equal_period", "process_constraint_less_than_period",
    "process_constraint_greater_than_period",
)

PERIOD_ROW_KINDS: dict[str, str] = {
    "co2_max_period": PATH,
    "ladder_tier_cap_annual_roll": PATH,
    "maxCumulative_flow_period": PATH,
    "minCumulative_flow_period": PATH,
    "nodeBalancePeriod_eq": PATH,
    "process_constraint_equal_period_total": PATH,
    "process_constraint_less_than_period_total": PATH,
    "process_constraint_greater_than_period_total": PATH,
    "node_storage_usage_fix_le": PATH_AT_STEP,   # a period sum anchored at a step (over has t)
    "commodity_ladder_balance": FRAGMENT,        # v_trade[d] buys d's own cells: right per name
    **dict.fromkeys(_INVEST_NAMES, INVEST),
}

# Every other constraint family name of the LP (rp_inter_period_* aside), reviewed: its over has a step column
# (t, b_first) or no d (solve totals, leaf paths, the Benders master), and its left-hand side sums no period.  A new
# name fails the pins (tests/engine/realization_roll/test_period_row_registry.py, registry_gaps) until it is added
# here or to PERIOD_ROW_KINDS.
REVIEWED_STEP_LEVEL = frozenset({
    "capacityMargin", "capacityMargin_coarse", "co2_max_total", "co2_max_total_path", "connection_reserve_down",
    "connection_reserve_down_coarse", "connection_reserve_down_exact", "connection_reserve_down_exact_coarse",
    "connection_reserve_down_online_integer", "connection_reserve_down_online_integer_coarse",
    "connection_reserve_down_online_linear", "connection_reserve_down_online_linear_coarse", "connection_reserve_up",
    "connection_reserve_up_coarse", "connection_reserve_up_exact", "connection_reserve_up_exact_coarse",
    "connection_reserve_up_online_integer", "connection_reserve_up_online_integer_coarse",
    "connection_reserve_up_online_linear", "connection_reserve_up_online_linear_coarse", "conversion_indirect",
    "conversion_indirect_coarse", "conversion_piecewise", "conversion_piecewise_coarse", "dc_flow_eq",
    "dc_flow_eq_coarse", "dc_reference_angle_eq", "inertia_constraint", "inertia_constraint_coarse",
    "ladder_tier_cap_cumulative_roll", "ladder_tier_cap_cumulative_roll_path", "maxCumulative_flow_solve",
    "maxCumulative_flow_solve_path", "maxDivestGroup_entity_total_n", "maxDivestGroup_entity_total_p",
    "maxDivestGroup_entity_total_path_n", "maxDivestGroup_entity_total_path_p", "maxDivest_entity_total",
    "maxDivest_entity_total_n", "maxDivest_entity_total_path", "maxDivest_entity_total_path_n", "maxFlow",
    "maxFlow_back", "maxFlow_back_coarse", "maxFlow_back_online_integer", "maxFlow_back_online_integer_coarse",
    "maxFlow_back_online_linear", "maxFlow_back_online_linear_coarse", "maxFlow_coarse", "maxFlow_exact_forward",
    "maxFlow_exact_forward_coarse", "maxFlow_exact_reverse", "maxFlow_exact_reverse_coarse", "maxFlow_negCap",
    "maxFlow_negCap_coarse", "maxFlow_online_integer", "maxFlow_online_integer_coarse", "maxFlow_online_linear",
    "maxFlow_online_linear_coarse", "maxFlow_online_shutdown_branch_integer",
    "maxFlow_online_shutdown_branch_integer_coarse", "maxFlow_online_shutdown_branch_linear",
    "maxFlow_online_shutdown_branch_linear_coarse", "maxFlow_online_shutdown_integer",
    "maxFlow_online_shutdown_integer_coarse", "maxFlow_online_shutdown_linear",
    "maxFlow_online_shutdown_linear_coarse", "maxInputFlow_online_integer", "maxInputFlow_online_integer_coarse",
    "maxInputFlow_online_linear", "maxInputFlow_online_linear_coarse", "maxInputFlow_online_shutdown_branch_integer",
    "maxInputFlow_online_shutdown_branch_integer_coarse", "maxInputFlow_online_shutdown_branch_linear",
    "maxInputFlow_online_shutdown_branch_linear_coarse", "maxInputFlow_online_shutdown_integer",
    "maxInputFlow_online_shutdown_integer_coarse", "maxInputFlow_online_shutdown_linear",
    "maxInputFlow_online_shutdown_linear_coarse", "maxInstant_flow", "maxInstant_flow_coarse", "maxInvest",
    "maxInvestGroup_entity_total_n", "maxInvestGroup_entity_total_p", "maxInvestGroup_entity_total_path_n",
    "maxInvestGroup_entity_total_path_p", "maxInvest_entity_total", "maxInvest_entity_total_n",
    "maxInvest_entity_total_path", "maxInvest_entity_total_path_n", "maxOnline_integer", "maxOnline_linear",
    "maxOutputSum", "maxOutputSum_coarse", "maxOutputSum_negCap", "maxOutputSum_negCap_coarse",
    "maxOutputSum_online_integer", "maxOutputSum_online_integer_coarse", "maxOutputSum_online_linear",
    "maxOutputSum_online_linear_coarse", "maxShutdown_integer", "maxShutdown_linear", "maxStartup_integer",
    "maxStartup_linear", "maxState", "maxState_rp_start", "minCumulative_flow_solve",
    "minCumulative_flow_solve_path", "minDivestGroup_entity_total_path_n", "minDivestGroup_entity_total_path_p",
    "minDivest_entity_total_path_n", "minDivest_entity_total_path_p", "minFlow_minload_integer",
    "minFlow_minload_integer_coarse", "minFlow_minload_linear", "minFlow_minload_linear_coarse",
    "minFlow_output_floor_integer", "minFlow_output_floor_integer_coarse", "minFlow_output_floor_linear",
    "minFlow_output_floor_linear_coarse", "minFlow_reserve", "minFlow_reserve_arc", "minFlow_reserve_arc_coarse",
    "minFlow_reserve_coarse", "minInputShare", "minInputShare_coarse", "minInstant_flow", "minInstant_flow_coarse",
    "minInvestGroup_entity_total_path_n", "minInvestGroup_entity_total_path_p", "minInvest_entity_total_path_n",
    "minInvest_entity_total_path_p", "minimum_downtime_integer", "minimum_downtime_linear", "minimum_uptime_integer",
    "minimum_uptime_linear", "nodeBalanceBlock_eq", "nodeBalanceCoarse_eq", "nodeBalance_eq",
    "node_balance_fix_quantity_eq_lower", "non_anticipativity_online_integer", "non_anticipativity_online_linear",
    "non_anticipativity_reserve", "non_anticipativity_rp_block_start", "non_anticipativity_storage_state_first",
    "non_anticipativity_storage_use", "non_anticipativity_storage_use_coarse", "non_sync_constraint",
    "non_sync_constraint_coarse", "online_change_integer", "online_change_linear", "online_fix_eq_integer",
    "online_fix_eq_linear", "process_constraint_equal", "process_constraint_equal_coarse",
    "process_constraint_greater_than", "process_constraint_greater_than_coarse", "process_constraint_less_than",
    "process_constraint_less_than_coarse", "profile_flow_fixed", "profile_flow_fixed_coarse",
    "profile_flow_lower_limit", "profile_flow_lower_limit_coarse", "profile_flow_upper_limit",
    "profile_flow_upper_limit_coarse", "profile_state_fixed", "profile_state_lower_limit",
    "profile_state_upper_limit", "ramp_change_eq", "ramp_sink_down_constraint", "ramp_sink_down_reserve_coarse",
    "ramp_sink_up_constraint", "ramp_sink_up_reserve_coarse", "ramp_source_down_constraint",
    "ramp_source_down_reserve_coarse", "ramp_source_up_constraint", "ramp_source_up_reserve_coarse",
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


def period_row_kind(name: str) -> str | None:
    """The ``PERIOD_ROW_KINDS`` kind of constraint family ``name`` (``rp`` by prefix), None when unclassified."""
    return RP if name.startswith(RP_PREFIX) else PERIOD_ROW_KINDS.get(name)


def _period_level_columns(over) -> bool:
    return over is not None and "d" in over.columns and not STEP_COLUMNS.intersection(over.columns)


def unclassified_period_families(m) -> list[str]:
    """The constraint families of ``m`` whose ``over`` has ``d`` and no step column and that ``PERIOD_ROW_KINDS``
    does not classify (the registry pin; empty when complete)."""
    return [n for n in m.cstr_names()
            if period_row_kind(n) is None and _period_level_columns(m.cstrs_named(n)[0].over)]


def unreviewed_families(names) -> list[str]:
    """The constraint family names of ``names`` in neither ``PERIOD_ROW_KINDS`` (or the ``rp`` prefix) nor
    ``REVIEWED_STEP_LEVEL`` (the name pin; empty when complete)."""
    return sorted(n for n in set(names) if period_row_kind(n) is None and n not in REVIEWED_STEP_LEVEL)


def unreviewed_message(names: list[str]) -> str:
    return (f"constraint families {names}: add each to PERIOD_ROW_KINDS (model/_period_rows.py), or to "
            "REVIEWED_STEP_LEVEL after checking that its over has a step column and its LHS sums no period")


def registry_gaps(m, d) -> list[str]:
    """The gaps of the realization-roll registries on the LP ``m`` built from ``d``: period-level families not in
    ``PERIOD_ROW_KINDS``, names not reviewed, FlexData frames with two step columns not in ``STEP_RELATIONS``.
    Empty when complete."""
    gaps = []
    if unclassified := unclassified_period_families(m):
        gaps.append(f"period-level families (over with d, no step column) not in PERIOD_ROW_KINDS: {unclassified}")
    if unreviewed := unreviewed_families(m.cstr_names()):
        gaps.append(unreviewed_message(unreviewed))
    if frames := unregistered_step_frames(d):
        gaps.append(f"FlexData frames with two step columns not in STEP_RELATIONS "
                    f"(preprocess/_step_relations.py): {frames}")
    return gaps


def check_registry_pins(b) -> None:
    """The registry pins on every build under ``FLEXTOOL_AUTOSCALE_STRICT=1`` (CI): raise on any
    ``registry_gaps``.  Off otherwise (a user run never stops on a registry gap; the checks G1-G4 still run)."""
    if not _strict():
        return
    if gaps := registry_gaps(b.m, b.d):
        raise RuntimeError(f"Realization-roll registry gaps in solve '{b.solve_name}': " + "; ".join(gaps))


def check_period_rows(b) -> None:
    """G1 on the families of ``b.m`` (module docstring).  No-op without a label with steps."""
    lab = labels_with_steps(b.d)
    if not lab:
        return
    invest = getattr(b.d, "period_invest", None)
    if invest is not None and invest.height > 0:
        # S9: a split realization roll is a roll, and rolls build no investment.
        for p in sorted(set(invest["d"].cast(pl.Utf8).to_list()).intersection(lab)):
            raise not_path_aware("period_invest", p, b.solve_name)
    branches = {f for fs in lab.values() for f in fs}
    for name in b.m.cstr_names():
        over = b.m.cstrs_named(name)[0].over
        kind = period_row_kind(name)
        if over is None or "d" not in over.columns or (kind is None and not _period_level_columns(over)):
            continue
        periods = set(over["d"].cast(pl.Utf8).unique().to_list())
        hit = sorted(periods.intersection(lab)) if kind != FRAGMENT else []
        if kind in (PATH, PATH_AT_STEP) and name not in b.path_converted:
            hit += sorted(periods & branches)
        if hit:
            raise not_path_aware(name, hit[0], b.solve_name)


def check_branch_weights(b) -> None:
    """G2: every (d, t) cell of the LP has a ``pdt_branch_weight`` > 0 (a missing cell has none)."""
    w = b.d.pdt_branch_weight
    if w is None:
        return
    cells = (b.d.dt.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8))
             .join(w.frame.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8), "value"),
                   on=["d", "t"], how="left"))
    bad = cells.filter(~(pl.col("value").fill_nan(None) > 0).fill_null(False))
    if bad.height > 0:
        raise not_path_aware("pdt_branch_weight", bad["d"][0], b.solve_name)
