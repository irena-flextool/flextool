"""End-to-end chain runner — cold rebuild + opt-in WarmProblem warm updates.

Drives a multi-solve scenario sub-solve by sub-solve in flexpy alone,
extracting an in-memory ``SolveHandoff`` after each sub-solve via
:func:`build_handoff_from_flexpy` and routing it into the next
sub-solve's ``FlexData`` via the loader's prior-handoff overlay.

Two execution modes:

* ``warm=False`` (default) — **cold rebuild** every sub-solve.  Each
  sub-solve gets a fresh :class:`polar_high_opt.Problem` + HiGHS instance.
  Backward-compatible with the original ``run_chain`` behaviour.

* ``warm=True`` — **solve-type-aware** runner.  Builds a
  :class:`polar_high_opt.WarmProblem` on the first sub-solve and, on each
  subsequent sub-solve, decides whether to warm-update the live LP
  (RHS / objective coefs only) or cold-rebuild from scratch.  The
  decision is made by comparing a structural fingerprint of each
  sub-solve's :class:`FlexData`: if two consecutive sub-solves share
  the same fingerprint AND the only Params that differ between them
  belong to the "clean-mapping" set (currently ``p_inflow``), the LP
  is warm-updated.  Any structural change OR any unmapped-Param diff
  triggers a cold rebuild and resets the warm state.

  This exploits flexpy's unique advantage over the GMPL pipeline in
  flextool — highspy's ``changeRowsBounds`` / ``changeColsCost`` lets
  us preserve the basis and the LP matrix across consecutive rolling
  sub-solves of, e.g., ``dispatch_fullYear_roll_roll_<i>``.

Today the per-sub-solve snapshots written by flextool's pre-run
(``solve_data_<sub>/``) already contain the prior solve's handoff
state baked into their CSVs (e.g. ``p_entity_previously_invested_capacity``,
``p_roll_continue_state``, ``fix_storage_quantity_<parent>.csv``).
``run_chain`` loads each snapshot AS IS — the snapshot is the source
of truth for handoff state.  The flexpy-derived handoff returned by
:func:`build_handoff_from_flexpy` is captured per sub-solve and
exposed via the returned dict for callers that want to compare
against flextool's writers, but it isn't used to override the
loader's read of the snapshot.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polar_high_opt import Problem, Solution, WarmProblem
from flextool.engine_polars.input import load_flextool, build_handoff_from_flexpy
from flextool.engine_polars.model import build_flextool
from flextool.engine_polars._input_source import _read_csv_file

if TYPE_CHECKING:
    from flextool.engine_polars.input import FlexData


__all__ = ["run_chain", "ChainStep"]


# ---------------------------------------------------------------------------
# Structural-fingerprint fields.
#
# Two consecutive sub-solves are "warm-compatible" iff they emit an LP of
# identical shape — same set of variables (same dims, same row counts) and
# same set of constraints (same row counts).  In flextool the LP shape is
# determined by which "set"-typed FlexData fields are populated and how
# many rows they hold.  We capture that with a tuple of (field_name,
# height) pairs.
#
# The list is intentionally NOT exhaustive — only fields that we have
# evidence affect the LP structure in tested scenarios are listed.  When
# warm=True misclassifies a transition (i.e. the obj diverges from the
# cold rebuild path), add the offending field here.

_STRUCTURAL_FIELDS: tuple[str, ...] = (
    # Time + node sets (the foundation of every LP).
    "dt", "nodeBalance", "nodeBalance_dt",
    # Process topology.
    "process_source_sink", "process_source_sink_eff",
    "process_source_sink_noEff", "pss_dt",
    "flow_to_n", "flow_from_n",
    "flow_from_commodity_eff", "flow_from_commodity_noEff",
    "flow_to_commodity",
    "pd_neg_cap",
    # CO2.
    "flow_from_co2_priced", "flow_from_co2_priced_noEff",
    "group_co2_max_period", "flow_from_co2_capped",
    "flow_from_co2_capped_noEff", "group_d_co2_capped",
    # Indirect (CHP).
    "process_indirect", "process_input_flows",
    "process_output_flows", "process_indirect_dt",
    # User constraints.
    "flow_constraint_idx", "cdt_eq", "cdt_le", "cdt_ge",
    # Profiles.
    "process_profile_upper", "process_profile_lower",
    "process_profile_fixed",
    # Invest / divest.
    "ed_invest_set", "ed_divest_set",
    "pd_invest_set", "pd_divest_set",
    "nd_invest_set", "nd_divest_set",
    "edd_invest_set", "edd_invest_lookback_set", "edd_divest_active",
    "e_invest_total", "e_divest_total",
    "ed_invest_period_set", "ed_divest_period_set",
    # Ramp.
    "process_source_sink_ramp_limit_sink_up",
    "process_source_sink_ramp_limit_sink_down",
    "process_source_sink_ramp_limit_source_up",
    "process_source_sink_ramp_limit_source_down",
    # Online / UC.
    "process_online", "process_online_linear", "process_online_integer",
    "process_minload", "process_min_load_eff",
    "p_online_dt", "pdt_online_linear", "pdt_online_integer",
    "pdt_uptime_set", "pdt_downtime_set",
    "uptime_lookback", "downtime_lookback",
    # Storage.
    "nodeState", "nodeState_dt", "nodeState_first_dt",
    "storage_bind_within_timeset", "storage_bind_forward_only",
    "storage_bind_within_solve", "storage_fix_start",
    "dtttdt", "dtttdt_forward_only",
    "n_fix_storage_quantity", "ndt_fix_storage_quantity",
    "dtt_timeline_matching", "period_branch", "period_last",
    "nodeState_last_dt",
    "nodeStateBlock", "period_block", "period_block_succ",
    "period_block_time", "dtttdt_block_interior",
    "arc_sink_block_dt", "arc_source_block_dt",
    "flow_from_nodeBalance_eff", "flow_from_nodeBalance_noEff",
    "node_profile_upper", "node_profile_lower", "node_profile_fixed",
    # Variable cost partitions.
    "pssdt_varCost_noEff", "pssdt_varCost_eff_unit_source",
    "pssdt_varCost_eff_unit_sink", "pssdt_varCost_eff_connection",
    # Group slack.
    "groupCapacityMargin", "groupInertia", "groupNonSync",
    "group_node", "process_unit",
    "process_sink_inertia", "process_source_inertia",
    "process_sink_nonSync", "process_group_inside_nonSync",
    # Reserves.
    "reserve_upDown_group",
    "reserve_upDown_group_method_timeseries",
    "reserve_upDown_group_method_dynamic",
    "reserve_upDown_group_method_n_1",
    "prundt", "process_reserve_upDown_node_active",
    "process_reserve_upDown_node_increase_reserve_ratio",
    "process_reserve_upDown_node_large_failure_ratio",
    # Cumulative invest / group invest.
    "ed_invest_forbidden_no_investment", "ed_invest_cumulative",
    "group_entity", "g_invest_total", "g_divest_total",
    "g_invest_cumulative", "gd_invest_period", "gd_divest_period",
    "gdt_maxInstantFlow", "gdt_minInstantFlow", "group_process_node",
    # Delays.
    "process_delayed", "process_delayed__duration",
    "process_source_delayed", "process_source_undelayed",
    "process_source_sink_delayed", "process_source_sink_undelayed",
    "dtt__delay_duration",
)


# ---------------------------------------------------------------------------
# Clean-mapping Params for warm updates.
#
# Only Params whose contribution to the LP is exactly "RHS of constraint X"
# OR "objective coefficient of variable Y" via a single-Param algebraic
# pathway can be warm-updated cleanly.  Multi-Param composite expressions
# (e.g. ``vq_up * p_penalty_up * p_node_capacity_for_scaling * op_factor``)
# would require the engine to track which Params feed which LP cells —
# that's WarmProblem's deferred Phase 2.
#
# Each entry is (flexdata_field, kind, target, transform, over_field).
# ``kind`` is either "rhs" (constraint RHS) or "obj" (variable objective
# coefficient).  ``target`` is the constraint or variable name in the
# built Problem.  ``transform`` is None (push the Param as-is) or "neg"
# (push -Param).  ``over_field`` names the FlexData index frame that the
# constraint was built ``over=`` (used to position-align values when the
# new sub-solve has different dim labels but same row counts — the
# rolling-horizon case).  None means push the Param's value column
# directly (used when the constraint axis dim values are stable across
# rolls, e.g. (n,) for storage-anchor handoff).
#
# Adding entries here only widens the set of transitions for which warm
# update is attempted; if a transition's diff falls entirely inside this
# set it stays warm, otherwise it falls back to cold rebuild.

_WARM_PARAMS: tuple[tuple[str, str, str, str | None, str | None], ...] = (
    ("p_inflow", "rhs", "nodeBalance_eq", "neg", "nodeBalance_dt"),
)

# Params that participate in composite LP cells (multi-Param products).
# When ``run_chain(..., warm=True)`` is invoked, the WarmProblem is told to
# track these via :meth:`polar_high_opt.WarmProblem.declare_mutable` so per-cell
# auto-update can fire on transitions where any of them differs between
# sub-solves.  This widens the warm-compatible regime BEYOND the clean-RHS
# subset above to cover:
#   * slack penalties scaled by op-factor and capacity_for_scaling,
#   * commodity-price terms multiplied by step_duration / inflation /
#     period_share / cost_weight,
#   * storage-anchor RHS terms ``p_state_start * p_state_existing_capacity``
#     and ``p_roll_continue_state``,
#   * per-(d,t) fix-storage RHS / inflow time-series.
# The list is kept narrow; additions are cheap (~tens of MB of side-table
# storage in the worst case) but each new entry has to be vetted against
# the auto-update math (numerator vs denominator direction recovery).
_MUTABLE_PARAMS: tuple[str, ...] = (
    "p_inflow",
    "p_penalty_up", "p_penalty_down",
    "p_state_start", "p_roll_continue_state",
    "p_fix_storage_quantity",
    "p_commodity_price",
    "p_step_duration", "p_rp_cost_weight",
    "p_inflation_op", "p_period_share",
    "p_node_capacity_for_scaling",
    "p_state_existing_capacity",
)

# Param fields that we know change between sub-solves but cannot warm-update
# (touched by composite expressions, multi-constraint patterns, or matrix
# coefficients that would require :meth:`WarmProblem.update_coef`).  When
# any of these differ between consecutive sub-solves and the structural
# fingerprint matches, we still cold-rebuild — they're listed here purely
# for clarity / future Phase-2 work.
#
# D1 audit (2026-05-03; see ``audit/handoff_full_parity_gaps.md`` §D1)
# categorised every entry on the
# ``work_multi_fullYear_battery_nested_multi_invest`` 80-roll chain:
#
#   * **No-op-on-tested-fixtures** entries (``presence_count == 0`` AND
#     ``diff_count == 0`` across every observed transition) are split
#     into :data:`_WARM_PARAMS_NO_OP` for documentation — they remain
#     here too so a regression on a fixture that DOES populate them is
#     still caught.
#   * **Gated-by-dormant-feature** entries are listed in
#     :data:`_WARM_PARAM_GATES`; on transitions where every gate field
#     is None the diff is "phantom" (the consuming constraint family
#     was never emitted) and we short-circuit the cold-rebuild fallback.
#   * **Sum-collapse RHS-side** entries (Params reaching the LP only via
#     constraint RHS as composite anonymous Params — e.g.
#     ``p_profile_value`` going through ``p_profile_value
#     · p_process_existing_count [· p_process_availability]`` into the
#     RHS of ``profile_flow_*``, or ``p_roll_continue_state`` rebuilt
#     into a sparse ``(n, d, t)`` Param dropped into nodeBalance LHS
#     constants) remain genuine cold-rebuild triggers.  Hand-coded
#     warm-update exception paths for these are blocked on engine-side
#     RHS source-tracking AND on rolling-horizon t-label-shift handling
#     for ``nodeState_first_dt`` / ``dtt_timeline_matching`` — both
#     explicitly out of scope for D1 (engine refactor; see follow-ups
#     in ``audit/handoff_param_tracked_autoupdate.md``).
_WARM_PARAMS_DEFERRED: tuple[str, ...] = (
    # Slack-penalty composites: vq * p_penalty_* * op_factor * scaling.
    "p_penalty_up", "p_penalty_down",
    # Time-weight composites that touch every (d,t)-keyed obj/lhs term.
    "p_step_duration", "p_inflation_op",
    "p_period_share", "p_rp_cost_weight",
    # Storage handoff / anchor — multi-cstr.
    "p_state_start", "p_roll_continue_state",
    "p_fix_storage_quantity", "p_state_existing_capacity",
    "p_state_unitsize", "p_state_self_discharge", "p_state_upper",
    # Invest handoff — RHS of multiple invest/divest cstrs.
    "p_entity_previously_invested_capacity",
    "p_entity_invested", "p_entity_divested",
    "p_entity_max_units", "p_entity_all_existing",
    "ed_lifetime_fixed_cost", "ed_lifetime_fixed_cost_divest",
    "ed_entity_annual_discounted", "ed_entity_annual_divest_discounted",
    "e_invest_max_total", "e_divest_max_total",
    "ed_invest_max_period", "ed_divest_max_period",
    # Commodity / CO2 — composite obj.
    "p_commodity_price", "p_co2_price", "p_co2_max_period",
    "p_co2_content",
    # Process topology Params used in many cstrs / objs.
    "p_unitsize", "p_flow_upper", "p_flow_upper_existing",
    "p_slope", "p_process_existing_count", "p_process_availability",
    "p_node_availability",
    # Profile Params — drive process_profile_* cstrs.
    "p_profile_value",
    # User constraints.
    "p_flow_constraint_coef", "p_constraint_constant",
    "p_node_constraint_invested_capacity_coefficient",
    "p_process_constraint_invested_capacity_coefficient",
    "p_node_constraint_state_coefficient",
    "p_node_constraint_prebuilt_capacity_coefficient",
    "p_process_constraint_prebuilt_capacity_coefficient",
    # Variable cost partitions.
    "p_pssdt_varCost", "p_pdt_varCost_source",
    "p_pdt_varCost_sink", "p_pdt_varCost_process",
    # Online / UC.
    "p_startup_cost", "p_section", "p_min_load",
    # Ramp speeds.
    "p_ramp_speed_up_sink", "p_ramp_speed_down_sink",
    "p_ramp_speed_up_source", "p_ramp_speed_down_source",
    # Capacity scaling.
    "p_node_capacity_for_scaling", "p_group_capacity_for_scaling",
    # Inflow (split into positive / negative for slack scaling — feeds
    # nodeBalance terms beyond just RHS).
    "p_positive_inflow", "p_negative_inflow", "pdtNodeInflow_per_step",
    # Existing-fixed cost on entities.
    "p_ed_fixed_cost",
    # Group reserves / capacity-margin / inertia.
    "pdGroup_capacity_margin", "pdGroup_penalty_capacity_margin",
    "pdGroup_inertia_limit", "pdGroup_penalty_inertia",
    "pdGroup_non_synchronous_limit", "pdGroup_penalty_non_synchronous",
    "p_inv_group_cap",
    "p_process_sink_inertia_constant", "p_process_source_inertia_constant",
    # Reserves.
    "pdtReserve_upDown_group_reservation",
    "p_reserve_upDown_group_penalty_reserve",
    "p_process_reserve_upDown_node_reliability",
    "p_process_reserve_upDown_node_max_share",
    "p_process_reserve_upDown_node_large_failure_ratio_value",
    "p_process_reserve_upDown_node_increase_reserve_ratio_value",
    # Cumulative invest / group invest Params.
    "ed_invest_min_period", "ed_divest_min_period",
    "e_invest_min_total", "e_divest_min_total",
    "ed_cumulative_max_capacity", "ed_cumulative_min_capacity",
    "p_group_invest_max_period", "p_group_invest_min_period",
    "p_group_retire_max_period", "p_group_retire_min_period",
    "p_group_invest_max_total", "p_group_invest_min_total",
    "p_group_retire_max_total", "p_group_retire_min_total",
    "p_group_invest_max_cumulative", "p_group_invest_min_cumulative",
    "p_group_max_cumulative_flow", "p_group_min_cumulative_flow",
    "pd_max_cumulative_flow", "pd_min_cumulative_flow",
    "pdt_max_instant_flow", "pdt_min_instant_flow",
    # Block / per-arc step durations.
    "p_arc_step_duration_sink", "p_arc_step_duration_source",
    "p_arc_sink_weight", "p_arc_source_weight",
    # Delays.
    "p_process_delay_weight",
)


# Subset of :data:`_WARM_PARAMS_DEFERRED` that the D1 audit observed to
# never differ in any tested chain transition (presence_count == 0 OR
# diff_count == 0 across every observed transition).  Kept as a tuple
# rather than removed so a future fixture that DOES populate one of
# these still gets caught by :data:`_WARM_PARAMS_DEFERRED`'s diff scan.
# This list is documentation only — :func:`_apply_warm_updates` does
# not consult it.
_WARM_PARAMS_NO_OP: tuple[str, ...] = (
    # Reserves — none of the in-tree fixtures exercise reserve scenarios.
    "pdtReserve_upDown_group_reservation",
    "p_reserve_upDown_group_penalty_reserve",
    "p_process_reserve_upDown_node_reliability",
    "p_process_reserve_upDown_node_max_share",
    "p_process_reserve_upDown_node_large_failure_ratio_value",
    "p_process_reserve_upDown_node_increase_reserve_ratio_value",
    # Cumulative invest / group invest — not active on the multi-invest
    # nested fixture.
    "ed_invest_min_period", "ed_divest_min_period",
    "e_invest_min_total", "e_divest_min_total",
    "ed_cumulative_max_capacity", "ed_cumulative_min_capacity",
    "p_group_invest_max_period", "p_group_invest_min_period",
    "p_group_retire_max_period", "p_group_retire_min_period",
    "p_group_invest_max_total", "p_group_invest_min_total",
    "p_group_retire_max_total", "p_group_retire_min_total",
    "p_group_invest_max_cumulative", "p_group_invest_min_cumulative",
    "p_group_max_cumulative_flow", "p_group_min_cumulative_flow",
    "pd_max_cumulative_flow", "pd_min_cumulative_flow",
    "pdt_max_instant_flow", "pdt_min_instant_flow",
    # Block / per-arc step durations — only used in multi-block fixtures.
    "p_arc_step_duration_sink", "p_arc_step_duration_source",
    "p_arc_sink_weight", "p_arc_source_weight",
    # Delays.
    "p_process_delay_weight",
    # Variable-cost partitions — none of the tested fixtures exercise
    # priced flows yet.
    "p_pssdt_varCost", "p_pdt_varCost_source",
    "p_pdt_varCost_sink", "p_pdt_varCost_process",
    # CO2 (not active on this fixture).
    "p_co2_price", "p_co2_max_period", "p_co2_content",
    # Online / UC.
    "p_startup_cost", "p_section", "p_min_load",
    # Ramp speeds.
    "p_ramp_speed_up_sink", "p_ramp_speed_down_sink",
    "p_ramp_speed_up_source", "p_ramp_speed_down_source",
    # Per-process inertia constants.
    "p_process_sink_inertia_constant", "p_process_source_inertia_constant",
    # Divest siblings (only present when divest is active).
    "p_entity_invested", "p_entity_divested",
    "ed_lifetime_fixed_cost_divest", "ed_entity_annual_divest_discounted",
    "e_divest_max_total", "ed_divest_max_period",
)


# Per-Param "structural gates": tuples of FlexData field names whose
# non-None state determines whether the Param can possibly contribute to
# any LP cell on the new sub-solve.  When ALL gates are None on
# ``nxt`` (and, by fingerprint match, also on ``prior``), the consuming
# constraint family was never emitted and the Param's diff is a phantom
# — :func:`_apply_warm_updates` skips the cold-rebuild check.
#
# Conservative by design: only Params whose consuming pathways are
# fully gated by tracked structural fields appear here.  Params with
# unconditional consumers (e.g. ``p_inflow`` always reaches
# ``nodeBalance_eq``) are absent and treated as always-active.
_WARM_PARAM_GATES: dict[str, tuple[str, ...]] = {
    # Group-slack inflow consumers (capacityMargin / inertia /
    # non_sync_constraint).
    "p_positive_inflow":              ("groupNonSync",),
    "p_negative_inflow":              ("groupNonSync",),
    "pdtNodeInflow_per_step":         ("groupCapacityMargin",),
    "pdGroup_capacity_margin":        ("groupCapacityMargin",),
    "pdGroup_penalty_capacity_margin": ("groupCapacityMargin",),
    "pdGroup_inertia_limit":          ("groupInertia",),
    "pdGroup_penalty_inertia":        ("groupInertia",),
    "pdGroup_non_synchronous_limit":  ("groupNonSync",),
    "pdGroup_penalty_non_synchronous": ("groupNonSync",),
    "p_inv_group_cap":                ("groupCapacityMargin", "groupInertia",
                                        "groupNonSync"),
    "p_process_sink_inertia_constant":   ("groupInertia",),
    "p_process_source_inertia_constant": ("groupInertia",),
    "p_group_capacity_for_scaling":   ("groupCapacityMargin", "groupInertia",
                                        "groupNonSync"),
    # Reserves.
    "pdtReserve_upDown_group_reservation":     ("reserve_upDown_group",),
    "p_reserve_upDown_group_penalty_reserve":  ("reserve_upDown_group",),
    "p_process_reserve_upDown_node_reliability": ("prundt",),
    "p_process_reserve_upDown_node_max_share":   ("prundt",),
    "p_process_reserve_upDown_node_large_failure_ratio_value":
        ("process_reserve_upDown_node_large_failure_ratio",),
    "p_process_reserve_upDown_node_increase_reserve_ratio_value":
        ("process_reserve_upDown_node_increase_reserve_ratio",),
    # CO2.
    "p_co2_price":      ("flow_from_co2_priced",),
    "p_co2_max_period": ("group_co2_max_period",),
    "p_co2_content":    ("flow_from_co2_priced", "flow_from_co2_capped",
                          "group_co2_max_period"),
    # Online / UC — only emitted when online sets are populated.
    "p_startup_cost": ("process_online",),
    "p_section":      ("process_min_load_eff",),
    "p_min_load":     ("process_minload",),
    # Ramps.
    "p_ramp_speed_up_sink":    ("process_source_sink_ramp_limit_sink_up",),
    "p_ramp_speed_down_sink":  ("process_source_sink_ramp_limit_sink_down",),
    "p_ramp_speed_up_source":  ("process_source_sink_ramp_limit_source_up",),
    "p_ramp_speed_down_source":
        ("process_source_sink_ramp_limit_source_down",),
    # Cumulative invest / group invest — gated by their respective sets.
    "ed_invest_min_period":           ("ed_invest_period_set",),
    "ed_divest_min_period":           ("ed_divest_period_set",),
    "e_invest_min_total":             ("e_invest_total",),
    "e_divest_min_total":             ("e_divest_total",),
    "ed_cumulative_max_capacity":     ("ed_invest_cumulative",),
    "ed_cumulative_min_capacity":     ("ed_invest_cumulative",),
    "p_group_invest_max_period":      ("gd_invest_period",),
    "p_group_invest_min_period":      ("gd_invest_period",),
    "p_group_retire_max_period":      ("gd_divest_period",),
    "p_group_retire_min_period":      ("gd_divest_period",),
    "p_group_invest_max_total":       ("g_invest_total",),
    "p_group_invest_min_total":       ("g_invest_total",),
    "p_group_retire_max_total":       ("g_divest_total",),
    "p_group_retire_min_total":       ("g_divest_total",),
    "p_group_invest_max_cumulative":  ("g_invest_cumulative",),
    "p_group_invest_min_cumulative":  ("g_invest_cumulative",),
    "p_group_max_cumulative_flow":    ("group_process_node",),
    "p_group_min_cumulative_flow":    ("group_process_node",),
    "pd_max_cumulative_flow":         ("group_process_node",),
    "pd_min_cumulative_flow":         ("group_process_node",),
    "pdt_max_instant_flow":           ("gdt_maxInstantFlow",),
    "pdt_min_instant_flow":           ("gdt_minInstantFlow",),
    # Variable-cost partitions.
    "p_pssdt_varCost":      ("pssdt_varCost_noEff",
                              "pssdt_varCost_eff_unit_source",
                              "pssdt_varCost_eff_unit_sink",
                              "pssdt_varCost_eff_connection"),
    "p_pdt_varCost_source": ("pssdt_varCost_eff_unit_source",
                              "pssdt_varCost_eff_connection"),
    "p_pdt_varCost_sink":   ("pssdt_varCost_eff_unit_sink",
                              "pssdt_varCost_eff_connection"),
    "p_pdt_varCost_process": ("pssdt_varCost_noEff",),
    # Block / per-arc step durations.
    "p_arc_step_duration_sink":   ("arc_sink_block_dt",),
    "p_arc_step_duration_source": ("arc_source_block_dt",),
    "p_arc_sink_weight":          ("arc_sink_block_dt",),
    "p_arc_source_weight":        ("arc_source_block_dt",),
    # Delays.
    "p_process_delay_weight": ("process_delayed",),
    # User constraints.
    "p_flow_constraint_coef":         ("flow_constraint_idx",),
    "p_constraint_constant":          ("cdt_eq", "cdt_le", "cdt_ge"),
    "p_node_constraint_invested_capacity_coefficient":
        ("flow_constraint_idx", "cdt_eq", "cdt_le", "cdt_ge"),
    "p_process_constraint_invested_capacity_coefficient":
        ("flow_constraint_idx", "cdt_eq", "cdt_le", "cdt_ge"),
    "p_node_constraint_state_coefficient":
        ("flow_constraint_idx", "cdt_eq", "cdt_le", "cdt_ge"),
    "p_node_constraint_prebuilt_capacity_coefficient":
        ("flow_constraint_idx", "cdt_eq", "cdt_le", "cdt_ge"),
    "p_process_constraint_prebuilt_capacity_coefficient":
        ("flow_constraint_idx", "cdt_eq", "cdt_le", "cdt_ge"),
}




class _IncompatibleUpdate(Exception):
    """Raised by :func:`_apply_warm_updates` when the difference between
    two consecutive sub-solves' FlexData includes Params outside the
    clean-mapping set, forcing a cold rebuild for the next sub-solve."""


class ChainStep:
    """Per-sub-solve result of :func:`run_chain`.

    Attributes
    ----------
    solve_name : str
        The sub-solve identifier (e.g. ``"y2025_5week"``).
    solution : flexpy.Solution
        The HiGHS solution for this sub-solve.
    handoff : flextool.SolveHandoff
        The flexpy-derived handoff carriers (see
        :func:`flextool.input.build_handoff_from_flexpy`).
    warm_used : bool
        True if this sub-solve was solved by warm-updating the prior
        sub-solve's :class:`WarmProblem` instance; False if it was a
        cold rebuild.  Always False for the first sub-solve and for
        ``warm=False`` runs.
    """

    __slots__ = ("solve_name", "solution", "handoff", "warm_used")

    def __init__(self, solve_name: str, solution: Solution, handoff,
                 warm_used: bool = False):
        self.solve_name = solve_name
        self.solution = solution
        self.handoff = handoff
        self.warm_used = warm_used

    def __repr__(self) -> str:  # pragma: no cover — debug-only
        return (f"ChainStep(solve_name={self.solve_name!r}, "
                f"obj={self.solution.obj!r}, "
                f"warm_used={self.warm_used}, "
                f"handoff_empty={self.handoff.is_empty()})")


def _read_chain_order(work_folder: Path) -> list[str]:
    """Return the ordered list of sub-solve names from
    ``input/model__solve.csv``.  Order matches CSV row order.
    """
    msv = work_folder / "input" / "model__solve.csv"
    if not msv.exists():
        # Fall back to any solve_data_<name>/ dirs sorted alphabetically.
        dirs = sorted(
            d.name[len("solve_data_"):] for d in work_folder.iterdir()
            if d.is_dir() and d.name.startswith("solve_data_")
            and d.name != "solve_data"
        )
        return dirs
    df = _read_csv_file(msv)
    if "solve" in df.columns:
        return df["solve"].cast(pl.Utf8).to_list()
    # Schema fallback — flextool's column may be different in some fixtures.
    return [str(v) for v in df[df.columns[-1]].to_list()]


def _stage_subsolve_workdir(
    work_folder: Path, sub_solve: str, tmpdir: Path,
) -> Path:
    """Build a per-sub-solve view of ``work_folder`` that points
    ``solve_data/`` at the sub-solve's snapshot dir.

    Symlinks ``input/`` and ``output_raw/`` so flexpy's loader sees
    the same shared data the per-sub-solve tests use.
    """
    import os
    for child in ("input", "output_raw"):
        src = work_folder / child
        if src.exists() and not (tmpdir / child).exists():
            os.symlink(src, tmpdir / child)
    sub_dir = work_folder / f"solve_data_{sub_solve}"
    if not sub_dir.exists():
        # Handle the "single-solve" degenerate case — fall through to the
        # canonical solve_data/ directly.
        sub_dir = work_folder / "solve_data"
    if not (tmpdir / "solve_data").exists():
        os.symlink(sub_dir, tmpdir / "solve_data")
    return tmpdir


def _fingerprint(data: "FlexData") -> tuple:
    """Compute a structural fingerprint of a FlexData snapshot.

    Returns a tuple of ``(field_name, height_or_None)`` pairs covering
    every field listed in :data:`_STRUCTURAL_FIELDS`.  Two FlexData
    snapshots with equal fingerprints emit identically-shaped LPs (set
    of vars and cstrs match by row count and dim signature) under the
    current ``build_flextool`` rules.

    ``height_or_None`` is ``None`` when the field is unset and the
    integer ``height`` when it's a polars DataFrame.  Boolean / scalar
    fields contribute their value directly.
    """
    out = []
    for name in _STRUCTURAL_FIELDS:
        v = getattr(data, name, None)
        if v is None:
            out.append((name, None))
        elif isinstance(v, pl.DataFrame):
            out.append((name, int(v.height)))
        elif isinstance(v, bool):
            out.append((name, bool(v)))
        else:
            # Unexpected — be conservative and force-mismatch.
            out.append((name, repr(type(v).__name__)))
    # ``p_nested_solve_first`` is a tri-state flag that swaps a whole
    # constraint family in/out — track it explicitly.
    out.append(("p_nested_solve_first",
                getattr(data, "p_nested_solve_first", None)))
    return tuple(out)


def _param_frame_equal(a, b) -> bool:
    """Return True if two flexpy Params have identical frames.

    Compares dim signature and value column row-by-row.  Robust to
    polars row-order differences via a sort on the dim columns.
    """
    if a is None and b is None:
        return True
    if (a is None) != (b is None):
        return False
    if a.dims != b.dims:
        return False
    af = a.frame
    bf = b.frame
    if af.height != bf.height:
        return False
    if af.height == 0:
        return True
    if a.dims:
        cols = list(a.dims)
        af = af.sort(cols)
        bf = bf.sort(cols)
    return af.equals(bf)


def _param_values_position_equal(a, b) -> bool:
    """Return True if two Params have value columns that are equal at
    matching positions (after sorting each by its full set of dim
    columns).

    Captures the rolling-horizon case where dim labels (e.g. ``t``)
    shift between sub-solves but the per-position values (e.g. constant
    ``3000.0`` slack penalties for every (n,d,t)) stay identical.

    Returns ``False`` if dim signatures differ or row counts differ.
    Returns ``True`` if both Params are ``None``.  Otherwise compares
    the sorted-by-dims value column element-wise within float64
    precision.
    """
    if a is None and b is None:
        return True
    if (a is None) != (b is None):
        return False
    if a.dims != b.dims:
        return False
    af = a.frame
    bf = b.frame
    if af.height != bf.height:
        return False
    if af.height == 0:
        return True
    if a.dims:
        # Sort each frame by its dim columns positionally — the t-labels
        # in `a` and `b` differ for rolling-horizon snapshots, so we
        # can't just compare frames as-is.  Sorting brings the value
        # columns into 1-to-1 positional correspondence assuming the
        # sort orders agree (which they do when both sub-solves have
        # the same number of dim-tuples).  This is a fast O(n log n)
        # numeric check rather than a full frame equality.
        cols = list(a.dims)
        av = af.sort(cols)["value"].to_numpy()
        bv = bf.sort(cols)["value"].to_numpy()
    else:
        av = af["value"].to_numpy()
        bv = bf["value"].to_numpy()
    import numpy as np
    return bool(np.array_equal(av, bv))


def _gate_active(d: "FlexData", fld: str) -> bool:
    """Return True if Param ``fld`` can possibly contribute to an LP
    cell on FlexData ``d`` based on its consuming-feature gates.

    Defaults to True (assume active) for Params absent from
    :data:`_WARM_PARAM_GATES` — gating is opt-in and conservative.
    Returns False ONLY when every gate field is None or an empty
    polars frame on ``d``; that means the consuming constraint family
    was never emitted, so the Param is dormant on this LP and a diff
    in its values can be safely ignored.
    """
    gates = _WARM_PARAM_GATES.get(fld)
    if not gates:
        return True
    for g in gates:
        v = getattr(d, g, None)
        if v is None:
            continue
        # Empty polars frame counts as "gate inactive" — the constraint
        # iterator yields zero rows.
        try:
            if hasattr(v, "height") and v.height == 0:
                continue
        except Exception:
            pass
        return True
    return False


def _apply_warm_updates(warm: WarmProblem,
                        prior: "FlexData", nxt: "FlexData") -> int:
    """Push every changed clean-mapping Param from ``prior`` → ``nxt``
    into ``warm``.

    Returns the count of warm-update calls executed.

    Raises :class:`_IncompatibleUpdate` if any Param in
    :data:`_WARM_PARAMS_DEFERRED` differs between ``prior`` and ``nxt``
    AND that Param is NOT in :data:`_MUTABLE_PARAMS`.  Mutable Params
    are auto-updated via :meth:`polar_high_opt.WarmProblem.update_param`.
    Phantom diffs (Params whose consuming feature is dormant per
    :data:`_WARM_PARAM_GATES`) are skipped — those are the audit-proven
    no-effect cases on this LP.  Mutable Params with zero tracked cells
    (Sum-collapse on the build-side composite Param construction) also
    raise :class:`_IncompatibleUpdate` rather than silently no-op'ing
    via ``update_param`` — the silent-corruption guard.
    """
    # First, scan the deferred (force-cold) list — any difference there
    # is a hard "cold rebuild" signal UNLESS the field is in
    # _MUTABLE_PARAMS, in which case auto-update will handle it below.
    mutable_set = set(_MUTABLE_PARAMS)
    deferred_diffs: dict[str, "object"] = {}
    for fld in _WARM_PARAMS_DEFERRED:
        prior_p = getattr(prior, fld, None)
        next_p = getattr(nxt, fld, None)
        if not _param_values_position_equal(prior_p, next_p):
            if fld in mutable_set:
                deferred_diffs[fld] = next_p
                continue
            if not _gate_active(nxt, fld):
                # Phantom diff — the consuming feature is dormant on
                # this sub-solve (e.g. p_negative_inflow when
                # groupNonSync is None), so the Param can't reach any
                # LP cell.  Safe to skip; ignoring this diff is
                # equivalent to cold-rebuilding and re-evaluating an
                # unused Param.
                continue
            raise _IncompatibleUpdate(
                f"Param {fld!r} differs between sub-solves and is not in "
                f"the clean-mapping set — falling back to cold rebuild.")

    import numpy as np

    n_updates = 0
    for fld, kind, target, transform, over_field in _WARM_PARAMS:
        prior_p = getattr(prior, fld, None)
        next_p = getattr(nxt, fld, None)
        if _param_frame_equal(prior_p, next_p):
            continue
        if next_p is None:
            # Going from "param present" to "param absent" effectively
            # changes the LP shape — treat as cold.
            raise _IncompatibleUpdate(
                f"Param {fld!r} disappeared between sub-solves; "
                f"falling back to cold rebuild.")
        if kind == "rhs":
            # Resolve the new RHS values positionally aligned to the
            # ORIGINAL over frame's row order.  The original over
            # frame's dim labels (e.g. ``t``) differ from ``next_p``'s
            # labels in rolling-horizon scenarios, so we can't rely on
            # WarmProblem.update_rhs's label-based join — it would
            # produce zeros for every row.  Instead we resolve the new
            # value vector against the NEW over frame (which has the
            # new t-labels) and push as a positional ndarray of length
            # row_count.
            if over_field is None:
                push = next_p
                if transform == "neg":
                    push = -next_p
                warm.update_rhs(target, push)
            else:
                new_over = getattr(nxt, over_field, None)
                if new_over is None:
                    raise _IncompatibleUpdate(
                        f"warm-update needs FlexData.{over_field}, but it "
                        f"is None on the new sub-solve")
                # Left-join new_over with next_p on shared dims; values
                # come out aligned to new_over's row order, which by
                # fingerprint match has the same row count as the
                # original LP cstr over.
                shared = [c for c in next_p.dims if c in new_over.columns]
                if not shared:
                    rhs_vec = np.full(new_over.height,
                                      float(next_p.frame["value"][0]),
                                      dtype=np.float64)
                else:
                    j = new_over.join(next_p.frame, on=shared, how="left")
                    rhs_vec = (j["value"].fill_null(0.0)
                                         .to_numpy()
                                         .astype(np.float64, copy=False))
                if transform == "neg":
                    rhs_vec = -rhs_vec
                warm.update_rhs(target, rhs_vec)
        elif kind == "obj":
            push = next_p
            if transform == "neg":
                push = -next_p
            warm.update_obj_coef(target, push)
        else:
            raise _IncompatibleUpdate(
                f"unknown warm-update kind {kind!r} for {fld!r}")
        n_updates += 1

    # Auto-update for declared-mutable Params via the Param-tracked
    # cell map.  This handles every composite-Param diff that the
    # clean-RHS / clean-obj path can't represent.
    for fld, next_p in deferred_diffs.items():
        if next_p is None:
            # Param disappeared — can't auto-update (no values to push).
            raise _IncompatibleUpdate(
                f"mutable Param {fld!r} went None between sub-solves; "
                f"falling back to cold rebuild.")
        if fld not in warm._mutable_params:
            # Param was tracked-mutable but the LP didn't actually
            # carry it (build skipped its branch).  Different LP — treat
            # as cold.
            raise _IncompatibleUpdate(
                f"mutable Param {fld!r} differs but isn't tracked on "
                f"the warm problem; falling back to cold rebuild.")
        # CRITICAL silent-corruption guard (D1 audit, 2026-05-03).
        # ``WarmProblem.update_param`` silently returns when the Param
        # has no tracked cells — that's correct behaviour for Params
        # whose only effect is on a code path that the engine's
        # source-tracker walks (e.g. p_step_duration on dispatch
        # rolls).  But many "mutable" Params reach the LP through
        # composite-anonymous-Param construction in flextool/model.py
        # (Sum-collapse: the new Param is built fresh without a
        # ``name=`` or ``_sources=`` link to the origin Param), and
        # for those the side-table is empty even though the LP DOES
        # depend on the values.  Pushing a no-op there leaves stale
        # coefficients in the live LP and silently corrupts the
        # objective.
        #
        # Fall back to cold rebuild whenever a mutable Param differs
        # but has zero tracked cells.  Loses warm-mode benefit on
        # those transitions but preserves correctness — the only
        # acceptable trade-off given task constraints.
        cells = warm._param_cells.get(fld)
        has_cells = cells is not None and int(cells["rows"].size) > 0
        if not has_cells and not _gate_active(nxt, fld):
            # Param's gates are dormant — diff is phantom, no-op is
            # genuinely safe (the consuming feature was never built).
            continue
        if not has_cells:
            raise _IncompatibleUpdate(
                f"mutable Param {fld!r} differs but the WarmProblem "
                f"recorded zero tracked cells for it (Sum-collapse on "
                f"the build-side composite-Param construction); "
                f"falling back to cold rebuild to avoid silent stale "
                f"LP coefficients.")
        warm.update_param(fld, next_p)
        n_updates += 1

    return n_updates


def _run_chain_native(
    work_folder: Path | str,
    *,
    chain: list[str] | None = None,
    scenario: str | None = None,
) -> dict[str, ChainStep]:
    """Native orchestrator backend for :func:`run_chain`.

    Discovers the scenario DB via:

    * an explicit ``tests.sqlite`` / ``input.sqlite`` under
      ``work_folder``.

    Picks the scenario via, in priority order:

    1. The explicit ``scenario`` kwarg (or the
       ``FLEXPY_NATIVE_SCENARIO`` env var as a fallback).
    2. The scenario whose name matches the directory's ``work_<S>``
       suffix (with a small set of legacy-naming overrides — see
       :data:`_NATIVE_SCENARIO_OVERRIDES`).
    3. Failing both, raises :class:`ValueError` instead of guessing —
       the native path is the consumer of a DB, not a snapshot tree;
       silently picking the first scenario alphabetically silently
       runs the wrong scenario in shared-DB fixtures.

    When the DB is found AND a unique scenario is determined,
    delegates to
    :func:`flextool.engine_polars._orchestration.run_chain_from_db`,
    re-runs flextool's preprocessing into the same work folder, and
    returns the result as ``dict[str, ChainStep]`` (mapping each solve
    to its Solution + handoff + ``warm_used=False``).
    """
    import os
    work = Path(work_folder)
    db_path = None
    for cand in ("tests.sqlite", "input.sqlite"):
        p = work / cand
        if p.exists():
            db_path = p
            break
    if db_path is None:
        raise ValueError(
            f"_run_chain_native: no DB found under {work} "
            f"(looked for tests.sqlite, input.sqlite).  Native "
            f"orchestration requires a DB scenario; for the file-based "
            f"path call run_chain(..., native=False)."
        )

    # Late import to avoid a build-time cycle between chain and _orchestration.
    from flextool.engine_polars._orchestration import run_chain_from_db

    # Resolve the scenario.
    if scenario is None:
        scenario = os.environ.get("FLEXPY_NATIVE_SCENARIO") or None
    if scenario is None:
        scenario = _resolve_native_scenario(db_path, work)
    if scenario is None:
        raise ValueError(
            f"_run_chain_native: cannot determine which scenario to "
            f"run for {work}.  Pass scenario= explicitly, set "
            f"FLEXPY_NATIVE_SCENARIO, or rename the work directory to "
            f"match the scenario name (work_<scenario>).  "
            f"work.name={work.name!r}"
        )

    # Critical: do NOT pass ``work`` directly as ``work_folder`` to
    # ``run_chain_from_db``.  ``FlexToolRunner.write_input`` would
    # overwrite the fixture's ``input/`` directory (rewriting
    # ``input/model__solve.csv`` to whatever scenario we resolved,
    # potentially shrinking a 4-solve cascade fixture down to a single
    # solve).  Instead, run the orchestrator into a private tempdir;
    # the work folder is consulted only for its DB.
    steps = run_chain_from_db(
        db_path, scenario,
    )
    # Adapt to ChainStep shape so callers see the same return type.
    out: dict[str, ChainStep] = {}
    for name, step in steps.items():
        out[name] = ChainStep(
            solve_name=name,
            solution=step.solution,
            handoff=step.handoff,
            warm_used=False,
        )
    return out


# Same overrides used by the parity-sweep fixtures — see
# tests/engine_polars/test_solve_config_parity._discover_fixtures.  The
# keys are work_folder dirnames; values are the scenario names that
# produced those snapshots.
_NATIVE_SCENARIO_OVERRIDES: dict[str, str] = {
    "work_2day_stochastic_dispatch_full_storage": "2_day_stochastic_dispatch",
    "work_commodity_ladder_annual": "coal_ladder_annual",
    "work_commodity_ladder_cumulative": "coal_ladder_cumulative",
    "work_delay_source_coef": "water_pump_delayed",
    "work_inflation_check": "wind_battery_invest_lifetime_renew",
}


def _resolve_native_scenario(db_path: Path, work: Path) -> str | None:
    """Map ``work/`` dirname → scenario name using the same convention
    as the parity tests.  Returns ``None`` when no rule matches.
    """
    import re
    import spinedb_api as api

    if work.name in _NATIVE_SCENARIO_OVERRIDES:
        return _NATIVE_SCENARIO_OVERRIDES[work.name]

    scen_target = work.name.removeprefix("work_") if work.name.startswith("work_") else None
    if scen_target is None:
        return None

    try:
        with api.DatabaseMapping("sqlite:///" + str(db_path)) as db:
            scenarios = sorted(s.name for s in db.query(db.scenario_sq).all())
    except Exception:
        return None

    candidates = [scen_target]
    candidates.append(re.sub(r"(^|_)(\d+)([a-z])", r"\1\2_\3", scen_target))
    candidates.append(re.sub(r"(\d+)_([a-z])", r"\1\2", scen_target))
    if scen_target.endswith("_full_storage"):
        base = scen_target[: -len("_full_storage")]
        candidates.append(re.sub(r"(^|_)(\d+)([a-z])", r"\1\2_\3", base))
        candidates.append(base)
    for cand in candidates:
        if cand in scenarios:
            return cand
    return None


def run_chain(
    work_folder: Path | str,
    *,
    use_handoff_overlay: bool = False,
    warm: bool = False,
    chain: list[str] | None = None,
    native: bool | None = None,
) -> dict[str, ChainStep]:
    """Run a flextool multi-solve chain end-to-end in flexpy.

    Iterates the sub-solves listed in ``input/model__solve.csv`` (or, if
    that file is missing, any ``solve_data_<name>/`` dirs found under
    ``work_folder``).  Each sub-solve:

    1. Loads its per-sub-solve snapshot (``solve_data_<sub>/``).
    2. Builds (or warm-updates) a flexpy LP via :func:`build_flextool`.
    3. Solves with HiGHS.
    4. Captures an in-memory ``SolveHandoff`` via
       :func:`build_handoff_from_flexpy`, threaded forward as the
       ``prior_handoff`` for the next sub-solve.

    The returned dict preserves chain order via Python 3.7+ dict
    insertion semantics; iterate with ``run_chain(...).items()`` to
    walk the chain in sequence.

    Parameters
    ----------
    work_folder : Path | str
        Directory containing ``input/``, ``output_raw/`` and one
        ``solve_data_<sub>/`` per chained sub-solve.
    use_handoff_overlay : bool, default False
        When True, every sub-solve except the first ignores the
        snapshot's pre-written handoff CSVs and instead overlays the
        prior sub-solve's flexpy-extracted ``SolveHandoff`` onto the
        loaded ``FlexData`` via :func:`apply_handoff`.  This makes the
        chain runner a TRUE standalone driver — flextool's per-sub-solve
        snapshots are needed only for STRUCTURE (entity sets, methods,
        profiles, …), and all multi-solve STATE flows in-memory between
        flexpy invocations.  Default ``False`` preserves the original
        behaviour: snapshot CSVs are the source of truth for handoff.
    warm : bool, default False
        When True, attempt warm LP updates between consecutive
        structurally-compatible sub-solves using
        :class:`polar_high_opt.WarmProblem`.  Decisions are recorded per-step
        on :attr:`ChainStep.warm_used`.  Default ``False`` preserves
        the original cold-rebuild behaviour for full backward
        compatibility.
    chain : list[str] | None, default None
        Explicit sub-solve order; overrides the default
        ``input/model__solve.csv`` lookup.  Use to drive a
        ``solve_data_<sub>/`` enumeration that the CSV doesn't cover
        (e.g. rolling-horizon snapshots whose ``model__solve.csv``
        names a parent solve rather than the per-roll snapshot dirs).
    native : bool | None, default None
        Γ.8.D feature flag.  ``True`` delegates to the native
        orchestrator (``_orchestration.run_orchestration``) which
        re-runs flextool's per-solve preprocessing under
        ``work_folder`` and runs HiGHS for every solve in-process via
        the in-memory handoff path.  ``False`` (default) preserves the
        legacy file-symlink-based driver below — the behaviour every
        existing test exercises.  ``None`` (default-default) consults
        the ``FLEXPY_USE_NATIVE_ORCHESTRATION`` env var: ``"1"`` /
        ``"true"`` / ``"yes"`` enable the native path, anything else
        keeps legacy.  R-O7 mitigation: legacy path stays the default
        so the existing test surface stays green.

    Returns
    -------
    dict[str, ChainStep]
        Mapping ``solve_name → ChainStep(solve_name, solution, handoff,
        warm_used)``.
    """
    import os
    import tempfile

    # Feature-flag gate.  ``native=True`` always uses the new path;
    # ``native=False`` always uses legacy; ``native=None`` consults the
    # env var (default ``False``).  See Γ.8.D in
    # ``audit/solve_orchestration_plan.md``.
    if native is None:
        env_val = os.environ.get(
            "FLEXPY_USE_NATIVE_ORCHESTRATION", ""
        ).strip().lower()
        native = env_val in ("1", "true", "yes", "on")
    if native:
        return _run_chain_native(work_folder, chain=chain)

    work = Path(work_folder)
    if chain is None:
        chain = _read_chain_order(work)
    if not chain:
        raise ValueError(
            f"run_chain: no sub-solves found in {work} "
            f"(missing input/model__solve.csv and no solve_data_<sub>/ dirs)"
        )

    results: dict[str, ChainStep] = {}
    prior_handoff = None
    # Warm-mode state, used only when warm=True.
    warm_problem: WarmProblem | None = None
    prior_data: "FlexData | None" = None
    prior_fp: tuple | None = None

    for i, sub_solve in enumerate(chain):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            _stage_subsolve_workdir(work, sub_solve, td)
            # Δ.11 — construct-with-handoff: pass the in-memory
            # SolveHandoff into ``load_flextool`` so the carrier-derived
            # fields are populated during the build.  Replaces the
            # previous post-load ``apply_handoff`` overlay step.
            handoff_arg = (prior_handoff
                              if use_handoff_overlay and i > 0
                                 and prior_handoff is not None
                              else None)
            data = load_flextool(td, handoff=handoff_arg)

            warm_used = False
            if warm:
                fp = _fingerprint(data)
                tried_warm = (i > 0
                              and warm_problem is not None
                              and prior_data is not None
                              and prior_fp == fp)
                if tried_warm:
                    try:
                        _apply_warm_updates(warm_problem, prior_data, data)
                        warm_used = True
                    except _IncompatibleUpdate:
                        warm_problem = None
                if not warm_used:
                    pb = Problem()
                    build_flextool(pb, data)
                    warm_problem = WarmProblem(pb)
                    # Declare every COMMONLY-MUTATING Param as mutable
                    # so the build-time bookkeeping records per-cell
                    # contribution maps for them.  Params not present
                    # on this particular FlexData are silently no-ops
                    # (no source-Param metadata flows through the LP
                    # build, the side-table for that name stays
                    # empty).
                    warm_problem.declare_mutable(*_MUTABLE_PARAMS)
                sol = warm_problem.solve()
                prior_data = data
                prior_fp = fp
            else:
                pb = Problem()
                build_flextool(pb, data)
                sol = pb.solve()

            handoff = build_handoff_from_flexpy(
                sol, td, sub_solve, prior_handoff=prior_handoff,
            )

        results[sub_solve] = ChainStep(sub_solve, sol, handoff,
                                       warm_used=warm_used)
        prior_handoff = handoff

    return results
