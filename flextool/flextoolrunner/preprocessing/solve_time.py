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
