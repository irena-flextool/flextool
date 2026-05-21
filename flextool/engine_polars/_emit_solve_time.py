"""Native per-solve preprocessing orchestrator.

Replaces the legacy ``flextool.flextoolrunner.preprocessing.solve_time.run``
(deleted in Step 2.5 item 15).  Every sub-emitter is a native polars
implementation in ``flextool.engine_polars._emit_*`` — the legacy
preprocessing package is gone, and after Phase 3a the cascade threads the
per-sub-solve Provider into every emit_* call directly.

Ordering parity
---------------

The legacy ordering carries dependency contracts between sub-emitters
(e.g. ``per_solve_sets`` produces ``period_in_use_set.csv`` consumed by
``period_calculated_params``; ``entity_period_calc_params`` writes
``pdProcess`` / ``pdNode`` consumed by ``write_p_entity_existing_chain``;
batch 19 ``ed_invest`` / ``ed_divest`` are consumed by batch 20 / 27).
We preserve the exact call order to keep those contracts intact.

Function-signature parity
-------------------------

* positional ``state: RunnerState``
* positional ``solve_name: str``
* keyword-only ``provider: object`` — required; every cascade caller
  (``_native_run_model.native_run_model``) supplies the per-sub-solve
  Provider.  No fallback to ``state.current_provider`` — direct
  threading is now the only path.

Phase 2 of ``specs/provider_consolidation.md`` retired the
``prior_handoff`` parameter — handoff-aware preprocessing sub-emitters
read the prior carrier via ``provider.get(K.HANDOFF_*)``, populated at
iteration start by ``_provider_translators.translate_handoff_to_provider``.
"""
from __future__ import annotations

from flextool.flextoolrunner.runner_state import RunnerState


def run(
    state: RunnerState,
    solve_name: str,
    *,
    provider: object,
) -> None:
    """Native per-solve preprocessing orchestrator.

    Calls the native polars emitters in the canonical order used by the
    legacy preprocessing chain.  *provider* is required and threaded
    into every emit_* invocation.
    """
    wf = state.paths.work_folder
    input_dir = wf / "input"
    solve_data_dir = wf / "solve_data"

    # Native emit modules.
    from flextool.engine_polars import (
        _emit_arc_unions as _arc,
        _emit_calc_params as _calc,
        _emit_dispatchers as _disp,
        _emit_entity_annual as _entity_annual,
        _emit_inflow_scaling as _inflow_scaling,
        _emit_leaf_sets as _leaf,
        _emit_lp_scaling as _lp_scaling,
        _emit_mid_sets as _mid,
        _emit_per_solve as _per_solve,
        _emit_period_calc as _period_calc,
        _emit_period_params as _period,
        _emit_reserve as _reserve,
    )

    # ── Refresh write_input-scope outputs (idempotent, no-DB) ─────────
    # The mod's ``if p_model['solveFirst']`` printf blocks may have
    # overwritten these on a previous solve.
    _leaf.emit_period_param_sets(input_dir, solve_data_dir, provider=provider)
    _leaf.emit_invest_method_sets(input_dir, solve_data_dir, provider=provider)
    _leaf.emit_co2_method_sets(input_dir, solve_data_dir, provider=provider)
    _leaf.emit_optional_yes(input_dir, solve_data_dir, provider=provider)
    _leaf.emit_reserve_upDown_group(input_dir, solve_data_dir, provider=provider)
    _leaf.emit_group_loss_share(input_dir, solve_data_dir, provider=provider)
    _mid.emit_node_type_sets(input_dir, solve_data_dir, provider=provider)
    _mid.emit_entity_lifetime_method(input_dir, solve_data_dir, provider=provider)
    _mid.emit_process_ct_method(input_dir, solve_data_dir, provider=provider)
    _mid.emit_process_startup_method(input_dir, solve_data_dir, provider=provider)
    _mid.emit_node_inflow_method(input_dir, solve_data_dir, provider=provider)
    _mid.emit_node_storage_binding_method(input_dir, solve_data_dir, provider=provider)
    _mid.emit_process_group_inside_group_nonsync(input_dir, solve_data_dir, provider=provider)
    _mid.emit_process__sink_nonSync(input_dir, solve_data_dir, provider=provider)
    _mid.emit_group_entity(input_dir, solve_data_dir, provider=provider)
    _mid.emit_process_delayed__duration(input_dir, solve_data_dir, provider=provider)
    _calc.emit_process_method_projections(input_dir, provider=provider)
    _calc.emit_process_VRE(input_dir, provider=provider)
    _calc.emit_process_arc_method_joins(input_dir, provider=provider)
    _calc.emit_process_profile_method_joins(input_dir, provider=provider)
    _mid.emit_reserve_partitions(input_dir, solve_data_dir, provider=provider)
    _mid.emit_connection_param(input_dir, solve_data_dir, provider=provider)
    _mid.emit_nodegroup_dispatch_node(input_dir, solve_data_dir, provider=provider)
    _mid.emit_commodity_node_co2(input_dir, solve_data_dir, provider=provider)
    _mid.emit_process__commodity__node(input_dir, solve_data_dir, provider=provider)
    _mid.emit_process_coeff_zero_sets(input_dir, solve_data_dir, provider=provider)
    _leaf.emit_def_optional_yes(input_dir, solve_data_dir, provider=provider)
    _leaf.emit_process_delayed(input_dir, solve_data_dir, provider=provider)
    _leaf.emit_process_side(solve_data_dir, provider=provider)
    _leaf.emit_simple_setof_projections(input_dir, solve_data_dir, provider=provider)
    _leaf.emit_period_solve(solve_data_dir, provider=provider)
    _leaf.emit_time_set(input_dir, solve_data_dir, provider=provider)
    _leaf.emit_enable_optional_outputs(solve_data_dir, provider=provider)
    _leaf.emit_node_state_subsets(solve_data_dir, provider=provider)
    _leaf.emit_commodity_tier_sets(input_dir, solve_data_dir, provider=provider)
    _mid.emit_dc_angle_bounds(input_dir, solve_data_dir, provider=provider)
    _mid.emit_invest_total_sets(input_dir, solve_data_dir, provider=provider)
    _mid.emit_ci_ladder_cumulative(input_dir, solve_data_dir, provider=provider)
    _disp.emit_process_arc_unions(input_dir, solve_data_dir, provider=provider)
    _arc.emit_group_commodity_node_period_co2_total(input_dir, solve_data_dir, provider=provider)
    _arc.emit_param_in_use_sets(input_dir, solve_data_dir, provider=provider)

    # ── Per-solve sets (must run BEFORE period_calculated_params) ─────
    _per_solve.emit_per_solve_sets(solve_data_dir, provider=provider)

    # ── L1 batch 13: period_calculated_params ─────────────────────────
    _period_calc.emit_period_calculated_params(input_dir, solve_data_dir, provider=provider)
    # ── L1 batch 15: pdProcess / pdNode + edEntity_lifetime + ed_fixed_cost
    _disp.emit_entity_period_calc_params(input_dir, solve_data_dir, provider=provider)
    # ── L1 batch 16: ed_entity_annual family ──────────────────────────
    _entity_annual.emit_entity_annual_calc_params(input_dir, solve_data_dir, provider=provider)
    # ── L1 batch 17: node-inflow scaling family ───────────────────────
    _inflow_scaling.emit_node_inflow_scaling_params(input_dir, solve_data_dir, provider=provider)
    # ── L1 batch 18: LP-scaling row scalers ───────────────────────────
    _lp_scaling.emit_lp_scaling_params(input_dir, solve_data_dir, provider=provider)
    # ── L2 batch 19: invest/divest + edd_history ──────────────────────
    _per_solve.emit_invest_divest_sets(input_dir, solve_data_dir, provider=provider)
    # ── L2 batch 20: ed_*_period / ed_cumulative_* (needs batch 19) ───
    _period.emit_ed_period_params(input_dir, solve_data_dir, provider=provider)
    # ── L2 batch 21 ───────────────────────────────────────────────────
    _arc.emit_process_source_delayed_partition(input_dir, solve_data_dir, provider=provider)
    # ── L2 batch 22 ───────────────────────────────────────────────────
    _arc.emit_node_time_param_in_use(input_dir, solve_data_dir, provider=provider)
    # ── L2 batch 25 ───────────────────────────────────────────────────
    _arc.emit_process_source_sink_param_t(solve_data_dir, provider=provider)
    # ── L2 batch 26: p_entity_pre_existing ────────────────────────────
    from flextool.engine_polars import _emit_chain_params as _chain
    _chain.emit_p_entity_pre_existing(input_dir, solve_data_dir, provider=provider)
    # ── L3 batch 27: ed_invest_forbidden_no_investment ────────────────
    _per_solve.emit_ed_invest_forbidden_no_investment(input_dir, solve_data_dir, provider=provider)
    # ── L4 batches 28-33 ──────────────────────────────────────────────
    _arc.emit_process_source_sink_delayed_partition(solve_data_dir, provider=provider)
    _arc.emit_process_source_sink_is_node_family(input_dir, solve_data_dir, provider=provider)
    _arc.emit_process_source_sink_ramp_family(input_dir, solve_data_dir, provider=provider)
    _arc.emit_process_source_sink_coeff_zero(solve_data_dir, provider=provider)
    _arc.emit_process_source_sink_ramp_method(input_dir, solve_data_dir, provider=provider)
    _arc.emit_node_group_dispatch_process_fully_inside(input_dir, solve_data_dir, provider=provider)
    # ── L5/L6 batch 62: 12 remaining nodeGroupDispatch sets ───────────
    _arc.emit_node_group_dispatch_sets(input_dir, solve_data_dir, provider=provider)
    # ── L4 batches 34-37 ──────────────────────────────────────────────
    _arc.emit_process_source_sink_param(input_dir, solve_data_dir, provider=provider)
    _arc.emit_process_source_sink_param_with_time(input_dir, solve_data_dir, provider=provider)
    _arc.emit_process_source_sink_profile_method_connection(
        input_dir, solve_data_dir, provider=provider,
    )
    _arc.emit_process_source_is_node_sink_1way_no_sink_or_more_than_1_source(
        input_dir, solve_data_dir, provider=provider,
    )
    # ── L4 batch 38: p_entity_divest_cumulative_max ───────────────────
    _chain.emit_p_entity_divest_cumulative_max(input_dir, solve_data_dir, provider=provider)
    # ── L4 batches 39-41 ──────────────────────────────────────────────
    _arc.emit_ed_history_realized_first(input_dir, solve_data_dir, provider=provider)
    _arc.emit_process_method_sources_sinks(input_dir, solve_data_dir, provider=provider)
    _arc.emit_peedt(solve_data_dir, provider=provider)
    # ── L0 batches 42 / 43 ────────────────────────────────────────────
    from flextool.engine_polars import _emit_pdt_params as _pdt
    _pdt.emit_pdtProcess(input_dir, solve_data_dir, provider=provider)
    _reserve.emit_pdtReserve_upDown_group(input_dir, solve_data_dir, provider=provider)
    # ── L1 batch 44 (needs pdtReserve from batch 43) ──────────────────
    _reserve.emit_process_reserve_upDown_node_active_and_prundt(
        input_dir, solve_data_dir, provider=provider,
    )
    # ── L0 batches 45 / 46 ────────────────────────────────────────────
    _pdt.emit_pdtProcess_source(input_dir, solve_data_dir, provider=provider)
    _pdt.emit_pdtProcess_sink(input_dir, solve_data_dir, provider=provider)
    # ── L4 batch 47 ───────────────────────────────────────────────────
    _period.emit_pdtProcess__source__sink__dt_varCost_pair(
        input_dir, solve_data_dir, provider=provider,
    )
    # ── L2/L5 batch 48 ────────────────────────────────────────────────
    _period.emit_pssdt_varCost_filters(input_dir, solve_data_dir, provider=provider)
    # ── L0/L2 batch 49 ────────────────────────────────────────────────
    _reserve.emit_process_reserve_filters_and_reliability(
        input_dir, solve_data_dir, provider=provider,
    )
    # ── L0 batch 50 ───────────────────────────────────────────────────
    _period.emit_cap_reduction_params(input_dir, solve_data_dir, provider=provider)
    # ── L0 batch 51 ───────────────────────────────────────────────────
    _arc.emit_pProcess_source_sink(input_dir, solve_data_dir, provider=provider)
    _period.emit_pdtCommodity(input_dir, solve_data_dir, provider=provider)
    # ── L0 batch 52 ───────────────────────────────────────────────────
    _period.emit_pdGroup(input_dir, solve_data_dir, provider=provider)
    _period.emit_pdtGroup(input_dir, solve_data_dir, provider=provider)
    _period.emit_pdCommodity(input_dir, solve_data_dir, provider=provider)
    # ── L0 batch 53 ───────────────────────────────────────────────────
    _pdt.emit_pdtNode(input_dir, solve_data_dir, provider=provider)
    # ── L8 batch 54 ───────────────────────────────────────────────────
    _period.emit_pdtNodeInflow(input_dir, solve_data_dir, provider=provider)
    # ── L0 batch 55 ───────────────────────────────────────────────────
    _period.emit_pdtProfile(input_dir, solve_data_dir, provider=provider)
    # ── L1 batch 56 ───────────────────────────────────────────────────
    _period.emit_pdtProcess_source_sink(input_dir, solve_data_dir, provider=provider)
    # ── L1/L2/L3 batch 57 ─────────────────────────────────────────────
    _period.emit_pdtConversion_rate_section_slope(input_dir, solve_data_dir, provider=provider)
    # ── L8 batch 58 ───────────────────────────────────────────────────
    _period.emit_p_positive_negative_inflow(input_dir, solve_data_dir, provider=provider)
    # ── L6/L7 batch 59: p_entity existing-capacity chain — reads
    # ``K.HANDOFF_REALIZED_EXISTING`` / ``K.HANDOFF_REALIZED_INVEST``
    # via the Provider (Phase 2 of provider_consolidation.md retired
    # the ``prior_handoff`` parameter threading).
    _chain.emit_p_entity_existing_chain(
        input_dir, solve_data_dir, provider=provider,
    )
    # ── L9/L10 batch 60: capacity max chain ───────────────────────────
    _chain.emit_p_entity_capacity_max_chain(input_dir, solve_data_dir, provider=provider)
    # ── L4/L5 batch 61: process_source_sink_ramp_unions ───────────────
    _arc.emit_process_source_sink_ramp_unions(solve_data_dir, provider=provider)
    # ── L0/L1 batch 63: branch weights + delay weight + co2 ───────────
    _period_calc.emit_branch_weights(input_dir, solve_data_dir, provider=provider)
    _arc.emit_p_process_delay_weight(input_dir, solve_data_dir, provider=provider)
    _arc.emit_gcndt_co2_price(input_dir, solve_data_dir, provider=provider)
    _arc.emit_group_commodity_node_period_co2_period(input_dir, solve_data_dir, provider=provider)
    # ── L0/L1 batch 64: param_t projections + instant-flow sets ───────
    _arc.emit_param_t_projections_and_time_params(input_dir, solve_data_dir, provider=provider)
    _arc.emit_gdt_instant_flow_sets(solve_data_dir, provider=provider)
    # ── L0/L1 batch 65: small set derivations ─────────────────────────
    _arc.emit_small_set_derivations(solve_data_dir, provider=provider)
    # ── L1 batch 66: state slack share + storage state reference price ─
    _arc.emit_p_state_slack_share(input_dir, solve_data_dir, provider=provider)
    _arc.emit_p_storage_state_reference_price(input_dir, solve_data_dir, provider=provider)
    # ── L2 batch 67: p_flow_min + p_flow_max ──────────────────────────
    _arc.emit_p_flow_min(input_dir, solve_data_dir, provider=provider)
    _arc.emit_p_flow_max(input_dir, solve_data_dir, provider=provider)
