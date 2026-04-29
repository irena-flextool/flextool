"""Per-solve preprocessing entry point.

Called from ``orchestration.run_model``'s solve loop, after
``solve_writers``/``blocks.write_block_data_for_solve`` have populated
``solve_data/`` and immediately before ``solver.run`` invokes glpsol.

Two responsibilities:

1. Refresh ALL Python-driven CSVs in ``solve_data/`` whose contents
   mod's ``if p_model['solveFirst']`` printf blocks may have overwritten
   on a previous solve. Refreshes are pure functions of the (immutable
   across solves) ``input/`` CSVs, so they're idempotent and safe to
   re-run every solve. Without this every multi-solve / rolling
   scenario would read stale or wrong-schema CSVs after solve 1.

2. Compute and write the per-solve-specific sets (``per_solve_sets.py``)
   whose inputs come from per-solve CSVs.

Note: ``commodity_ladder_sets`` and ``entity_total_caps`` aren't
refreshed here — they need the DB (closed by the time we get here)
and their outputs aren't overwritten by any mod printf. Their
write_input-time pass is sufficient.
"""
from __future__ import annotations

from pathlib import Path

from flextool.flextoolrunner.runner_state import RunnerState


def run(state: RunnerState, solve_name: str) -> None:
    """Execute per-solve preprocessing for ``solve_name``.

    Idempotent: calling twice with the same ``solve_name`` produces
    the same outputs.
    """
    wf = state.paths.work_folder
    input_dir = wf / "input"
    solve_data_dir = wf / "solve_data"

    # Refresh write_input-scope outputs that don't depend on the DB.
    # All input/ CSVs are immutable across solves, so re-running these
    # is pure recomputation. The output CSVs may have been overwritten
    # by mod's printf blocks on a previous solve.
    from flextool.flextoolrunner.preprocessing import (
        period_param_sets,
        invest_method_sets,
        co2_method_sets,
        simple_projections,
        node_type_sets,
        method_with_fallback_sets,
        nonsync_sets,
        union_sets,
        process_method_sets,
        reserve_method_partitions,
        structural_filters,
        dc_angle_bounds,
        invest_total_sets,
        process_arc_unions,
        period_calculated_params,
        entity_period_calc_params,
        entity_annual_calc_params,
        node_inflow_scaling_params,
        lp_scaling_params,
        invest_divest_sets,
        per_solve_sets,
        reserve_calc_params,
    )
    period_param_sets.write_period_param_sets(input_dir, solve_data_dir)
    invest_method_sets.write_invest_method_sets(input_dir, solve_data_dir)
    co2_method_sets.write_co2_method_sets(input_dir, solve_data_dir)
    simple_projections.write_optional_yes(input_dir, solve_data_dir)
    simple_projections.write_reserve_upDown_group(input_dir, solve_data_dir)
    simple_projections.write_group_loss_share(input_dir, solve_data_dir)
    node_type_sets.write_node_type_sets(input_dir, solve_data_dir)
    method_with_fallback_sets.write_entity_lifetime_method(input_dir, solve_data_dir)
    method_with_fallback_sets.write_process_ct_method(input_dir, solve_data_dir)
    method_with_fallback_sets.write_process_startup_method(input_dir, solve_data_dir)
    method_with_fallback_sets.write_node_inflow_method(input_dir, solve_data_dir)
    method_with_fallback_sets.write_node_storage_binding_method(input_dir, solve_data_dir)
    nonsync_sets.write_process_group_inside_group_nonsync(input_dir, solve_data_dir)
    nonsync_sets.write_process__sink_nonSync(input_dir, solve_data_dir)
    union_sets.write_group_entity(input_dir, solve_data_dir)
    union_sets.write_process_delayed__duration(input_dir, solve_data_dir)
    process_method_sets.write_process_method_projections(input_dir, solve_data_dir)
    process_method_sets.write_process_VRE(input_dir, solve_data_dir)
    process_method_sets.write_process_arc_method_joins(input_dir, solve_data_dir)
    process_method_sets.write_process_profile_method_joins(input_dir, solve_data_dir)
    reserve_method_partitions.write_reserve_partitions(input_dir, solve_data_dir)
    structural_filters.write_connection_param(input_dir, solve_data_dir)
    structural_filters.write_nodegroup_dispatch_node(input_dir, solve_data_dir)
    structural_filters.write_commodity_node_co2(input_dir, solve_data_dir)
    structural_filters.write_process__commodity__node(input_dir, solve_data_dir)
    structural_filters.write_process_coeff_zero_sets(input_dir, solve_data_dir)
    simple_projections.write_def_optional_yes(input_dir, solve_data_dir)
    simple_projections.write_process_delayed(input_dir, solve_data_dir)
    simple_projections.write_process_side(solve_data_dir)
    simple_projections.write_simple_setof_projections(input_dir, solve_data_dir)
    simple_projections.write_period_solve(solve_data_dir)
    simple_projections.write_time_set(input_dir, solve_data_dir)
    simple_projections.write_enable_optional_outputs(solve_data_dir)
    simple_projections.write_node_state_subsets(solve_data_dir)
    simple_projections.write_commodity_tier_sets(input_dir, solve_data_dir)
    dc_angle_bounds.write_dc_angle_bounds(input_dir, solve_data_dir)
    invest_total_sets.write_invest_total_sets(input_dir, solve_data_dir)
    invest_total_sets.write_ci_ladder_cumulative(input_dir, solve_data_dir)
    process_arc_unions.write_process_arc_unions(input_dir, solve_data_dir)
    process_arc_unions.write_group_commodity_node_period_co2_total(input_dir, solve_data_dir)
    process_arc_unions.write_param_in_use_sets(input_dir, solve_data_dir)

    # Per-solve-only sets: inputs in solve_data/ written above by
    # orchestration / solve_writers / blocks before this hook fires.
    # Must run BEFORE period_calculated_params, which depends on
    # period_in_use_set.csv etc. produced here.
    per_solve_sets.write_per_solve_sets(solve_data_dir)

    # L1 batch 13: per-period calculated params (per-solve scope).
    # Reads period_in_use_set.csv etc. that per_solve_sets just wrote.
    period_calculated_params.write_period_calculated_params(input_dir, solve_data_dir)
    # L1 batch 15: pdProcess / pdNode + edEntity_lifetime + ed_fixed_cost.
    entity_period_calc_params.write_entity_period_calc_params(input_dir, solve_data_dir)
    # L1 batch 16: ed_entity_annual + discounted variants + ed_lifetime_fixed_cost.
    entity_annual_calc_params.write_entity_annual_calc_params(input_dir, solve_data_dir)
    # L1 batch 17: node-inflow scaling family (ptNode_inflow + 17 calc params).
    node_inflow_scaling_params.write_node_inflow_scaling_params(input_dir, solve_data_dir)
    # L1 batch 18: LP-scaling row scalers (node + group capacities).
    lp_scaling_params.write_lp_scaling_params(input_dir, solve_data_dir)
    # L2 batch 19: invest/divest entity-period sets + edd_history family.
    invest_divest_sets.write_invest_divest_sets(input_dir, solve_data_dir)
    # L2 batch 20: ed_*_period / ed_cumulative_* family (depends on
    # ed_invest / ed_divest written by batch 19 above).
    entity_period_calc_params.write_ed_period_params(input_dir, solve_data_dir)
    # L2 batch 21: process_source partitioned by process_delayed.
    process_arc_unions.write_process_source_delayed_partition(input_dir, solve_data_dir)
    # L2 batch 22: node__TimeParam_in_use.
    process_arc_unions.write_node_time_param_in_use(input_dir, solve_data_dir)
    # L2 batch 25: process_source_sink_param_t (filter on pt_process keys).
    process_arc_unions.write_process_source_sink_param_t(input_dir, solve_data_dir)
    # L2 batch 26: p_entity_pre_existing (12-branch lifetime-method
    # × entity-kind × virtual-unitsize gate). Reads pdProcess, pdNode,
    # edEntity_lifetime written by write_entity_period_calc_params above.
    entity_period_calc_params.write_p_entity_pre_existing(input_dir, solve_data_dir)
    # L3 batch 27: ed_invest_forbidden_no_investment (ed_invest filtered
    # by no_investment method + expired lifetime). Reads ed_invest from
    # batch 19 and edEntity_lifetime from entity_period_calc_params.
    invest_divest_sets.write_ed_invest_forbidden_no_investment(input_dir, solve_data_dir)
    # L4 batch 28: process_source_sink partitioned by process_delayed.
    process_arc_unions.write_process_source_sink_delayed_partition(input_dir, solve_data_dir)
    # L4 batch 29: process__source__sinkIsNode + 3 method-bucket partitions.
    process_arc_unions.write_process_source_sink_is_node_family(input_dir, solve_data_dir)
    # L4 batch 30: process_source_sink_ramp_limit_*/cost (5 sets — ramp method
    # gate + per-side ramp_speed gate from p_process_source/sink).
    process_arc_unions.write_process_source_sink_ramp_family(input_dir, solve_data_dir)
    # L4 batch 31: process_source_sink_coeff_zero (OR of side-coeff-zero sets).
    process_arc_unions.write_process_source_sink_coeff_zero(input_dir, solve_data_dir)
    # L4 batch 32: process__source__sink__ramp_method (4-tuple, per-side).
    process_arc_unions.write_process_source_sink_ramp_method(input_dir, solve_data_dir)
    # L4 batch 33: nodeGroupDispatch__process_fully_inside.
    process_arc_unions.write_node_group_dispatch_process_fully_inside(input_dir, solve_data_dir)
    # L5/L6 batch 62: 12 remaining nodeGroupDispatch sets — 8 base
    # 4-/5-tuple partitions × {unit/connection, source/sink-in-group,
    # with/without flowAggregator} + 4 (g,p)/(g,ga) projections.
    # Depends on process_source_sink_alwaysProcess (write_process_arc_unions
    # above) and nodeGroupDispatch__process_fully_inside (batch 33 just
    # written above).
    process_arc_unions.write_node_group_dispatch_sets(input_dir, solve_data_dir)
    # L4 batch 34: process__source__sink__param.
    process_arc_unions.write_process_source_sink_param(input_dir, solve_data_dir)
    # L4 batch 35: process__source__sink__param_t (double-underscore mod
    # name; broader than batch 25's process_source_sink_param_t — adds
    # time-variant ORs).
    process_arc_unions.write_process_source_sink_param_with_time(input_dir, solve_data_dir)
    # L4 batch 36: process__source__sink__profile__profile_method_connection.
    process_arc_unions.write_process_source_sink_profile_method_connection(input_dir, solve_data_dir)
    # L4 batch 37: process__sourceIsNode__sink_1way_noSinkOrMoreThan1Source.
    process_arc_unions.write_process_source_is_node_sink_1way_no_sink_or_more_than_1_source(input_dir, solve_data_dir)
    # L4 batch 38: p_entity_divest_cumulative_max — depends on
    # ed_divest_period (batch 19) and ed_divest_max_period (batch 20).
    entity_period_calc_params.write_p_entity_divest_cumulative_max(input_dir, solve_data_dir)
    # L4 batch 39: ed_history_realized_first (gated on p_model['solveFirst']).
    process_arc_unions.write_ed_history_realized_first(input_dir, solve_data_dir)
    # L4 batch 40: process_method_sources_sinks (3-way join, 6-tuple output).
    process_arc_unions.write_process_method_sources_sinks(input_dir, solve_data_dir)
    # L4 batch 41: peedt — cross-product of process_source_sink × dt.
    # Indexes v_flow and p_flow_max/_min / d_flow* in the mod.
    process_arc_unions.write_peedt(input_dir, solve_data_dir)
    # L0 batch 42: pdtProcess — 7-branch hourly process param resolution.
    # Unblocks pdtProcess_section + pssdt_varCost_eff_* + 4 downstream sets.
    entity_period_calc_params.write_pdtProcess(input_dir, solve_data_dir)
    # L0 batch 43: pdtReserve_upDown_group — 4-branch hourly reserve param.
    # Unblocks process_reserve_upDown_node_active + prundt + reserve ratio sets.
    reserve_calc_params.write_pdtReserve_upDown_group(input_dir, solve_data_dir)
    # L1 batch 44: process_reserve_upDown_node_active + prundt — depend on
    # pdtReserve_upDown_group from batch 43.
    reserve_calc_params.write_process_reserve_upDown_node_active_and_prundt(
        input_dir, solve_data_dir
    )
    # L0 batch 45: pdtProcess_source — 6-branch hourly per-source param.
    entity_period_calc_params.write_pdtProcess_source(input_dir, solve_data_dir)
    # L0 batch 46: pdtProcess_sink — 6-branch hourly per-sink param.
    entity_period_calc_params.write_pdtProcess_sink(input_dir, solve_data_dir)
    # L4 batch 47: pdtProcess__source__sink__dt_varCost{,_alwaysProcess} —
    # depend on pdtProcess + pdtProcess_source + pdtProcess_sink.
    entity_period_calc_params.write_pdtProcess__source__sink__dt_varCost_pair(
        input_dir, solve_data_dir
    )
    # L2/L5 batch 48: pssdt_varCost_eff_*/noEff (4 filter sets).
    entity_period_calc_params.write_pssdt_varCost_filters(input_dir, solve_data_dir)
    # L0/L2 batch 49: p_process_reserve_upDown_node_reliability + 2 ratio
    # filter sets + process_large_failure projection. Reads
    # process_reserve_upDown_node_active (batch 44) and p_process_reserve_upDown_node.
    reserve_calc_params.write_process_reserve_filters_and_reliability(
        input_dir, solve_data_dir
    )
    # L0 batch 50: p_*_cap_reduction_*  (4 Morales-Espana cap reduction params).
    entity_period_calc_params.write_cap_reduction_params(input_dir, solve_data_dir)
    # L0 batch 51: pProcess_source_sink + pdtCommodity (simple fallback params).
    entity_period_calc_params.write_pProcess_source_sink(input_dir, solve_data_dir)
    entity_period_calc_params.write_pdtCommodity(input_dir, solve_data_dir)
    # L0 batch 52: pdGroup + pdtGroup + pdCommodity (3-5 branch fallbacks).
    entity_period_calc_params.write_pdGroup(input_dir, solve_data_dir)
    entity_period_calc_params.write_pdtGroup(input_dir, solve_data_dir)
    entity_period_calc_params.write_pdCommodity(input_dir, solve_data_dir)
    # L0 batch 53: pdtNode — 9-branch hourly node param resolution.
    # Same shape as pdtProcess but time-first axis priority and extra
    # class_paramName_default fallback (input/default_values.csv).
    entity_period_calc_params.write_pdtNode(input_dir, solve_data_dir)
    # L8 batch 54: pdtNodeInflow — stochastic / parent-branch fold-in OR
    # additive sum across the 4 inflow scaling methods. Reads pdNode +
    # ptNode_inflow + period_flow_*_multiplier + new_old_slope/section.
    entity_period_calc_params.write_pdtNodeInflow(input_dir, solve_data_dir)
    # L0 batch 55: pdtProfile — 5-branch profile param resolution with
    # 3-way UNION stochastic gate (process_profile / node_profile /
    # process_node_profile groups).
    entity_period_calc_params.write_pdtProfile(input_dir, solve_data_dir)
    # L1 batch 56: pdtProcess_source_sink — 11-branch combined source+sink
    # fallback with connection-only pt_process / p_process branches.
    entity_period_calc_params.write_pdtProcess_source_sink(input_dir, solve_data_dir)
    # L1/L2/L3 batch 57: pdtConversion_rate + pdtProcess_section + pdtProcess_slope.
    # Cascade: conversion_rate = 1/efficiency, section uses min_load + eff_at_min,
    # slope = conversion_rate - (section if p in process_minload else 0).
    # Mod's wide-format printfs for slope+section are retargeted to
    # solve__pdtProcess_{slope,section}.csv to break the dual-writer collision.
    entity_period_calc_params.write_pdtConversion_rate_section_slope(
        input_dir, solve_data_dir
    )
    # L8 batch 58: p_positive_inflow + p_negative_inflow — max/min thresholds
    # on pdtNodeInflow. Reads pdtNodeInflow.csv (batch 54).
    entity_period_calc_params.write_p_positive_negative_inflow(
        input_dir, solve_data_dir
    )
    # L6/L7 batch 59: p_entity existing-capacity chain (5 cascading params).
    # Reads p_entity_pre_existing (batch 12), p_entity_unitsize (batch 18),
    # edd_history, ed_history_realized_first, p_entity_period_existing_capacity
    # (handoff), p_entity_divested (handoff). solveFirst flag from p_model.csv.
    entity_period_calc_params.write_p_entity_existing_chain(
        input_dir, solve_data_dir
    )
    # L9/L10 batch 60: capacity max chain (4 params).
    # p_entity_max_capacity, p_entity_max_units, p_entity_invest_cumulative_max,
    # p_entity_dispatch_capacity_max — all depend on p_entity_all_existing
    # (batch 59). Path-collision: p_entity_max_units.csv retargeted.
    entity_period_calc_params.write_p_entity_capacity_max_chain(
        input_dir, solve_data_dir
    )
    # L4/L5 batch 61: process_source_sink_ramp (5-way union of ramp_*
    # input sets) + 4 dtttdt-filtered ramp limit sets. Reads the 5
    # ramp_*.csv files written by batch 30 + step_previous.csv +
    # steps_in_use.csv + p_process_source/sink.csv.
    process_arc_unions.write_process_source_sink_ramp_unions(
        input_dir, solve_data_dir
    )
    # L0/L1 batch 63: branch weights and odds (5 items).
    # pd_branch_weight + pdt_branch_weight: per-period and per-(d,t)
    # normalization of solve_branch_weight by sibling-branch sum.
    period_calculated_params.write_branch_weights(input_dir, solve_data_dir)
    # p_process_delay_weight: 1 if (p, td) in process_delay_single,
    # else p_process_delay_weighted[p, td].
    process_arc_unions.write_p_process_delay_weight(input_dir, solve_data_dir)
    # gcndt_co2_price (5-tuple set, hourly co2 price gate) and
    # group_commodity_node_period_co2_period (4-tuple, max-period gate).
    process_arc_unions.write_gcndt_co2_price(input_dir, solve_data_dir)
    process_arc_unions.write_group_commodity_node_period_co2_period(
        input_dir, solve_data_dir
    )
    # L0/L1 batch 64: process__*__param family + gdt_{max,min}InstantFlow.
    # 8 set projections/joins around process__*__param__time CSVs +
    # 2 group×dt sets gated by pdtGroup non-zero values.
    process_arc_unions.write_param_t_projections_and_time_params(
        input_dir, solve_data_dir
    )
    process_arc_unions.write_gdt_instant_flow_sets(
        input_dir, solve_data_dir
    )
    # L0/L1 batch 65: 6 small set derivations (ed_history_realized,
    # process__source__sink__profile__profile_method (4-way union),
    # process_sinkIsNode_2way1var, nodeSelfDischarge,
    # pdt_online_linear/integer). Depends on: pdtNode, pdProcess,
    # ed_history_realized_first, process__source__sinkIsNode_2way1var.
    process_arc_unions.write_small_set_derivations(input_dir, solve_data_dir)
    # L1 batch 66: p_state_slack_share + p_storage_state_reference_price.
    # Both have empty domains for the 5 parity baselines, but migrated
    # faithfully (mod's `:= ...` removed, table-data-IN added).
    process_arc_unions.write_p_state_slack_share(input_dir, solve_data_dir)
    process_arc_unions.write_p_storage_state_reference_price(
        input_dir, solve_data_dir
    )
