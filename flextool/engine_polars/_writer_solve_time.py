"""Native per-solve preprocessing orchestrator.

Replaces the legacy ``flextool.flextoolrunner.preprocessing.solve_time.run``
(deleted in Step 2.5 item 15).  Every sub-writer is now a native polars
implementation in ``flextool.engine_polars._writer_*`` — the legacy
preprocessing package is gone, so we call the native writers directly
rather than through the old monkey-patch on legacy module attributes.

Ordering parity
---------------

The legacy ordering carries dependency contracts between sub-writers
(e.g. ``per_solve_sets`` produces ``period_in_use_set.csv`` consumed by
``period_calculated_params``; ``entity_period_calc_params`` writes
``pdProcess`` / ``pdNode`` consumed by ``write_p_entity_existing_chain``;
batch 19 ``ed_invest`` / ``ed_divest`` are consumed by batch 20 / 27).
We preserve the exact call order to keep those contracts intact.

Function-signature parity
-------------------------

* positional ``state: RunnerState``
* positional ``solve_name: str``
* keyword-only ``prior_handoff: SolveHandoff | None = None``
* keyword-only ``provider: object | None = None`` — when ``None`` we
  fall back to ``state.current_provider`` (set by the cascade
  orchestrator before each sub-solve).
"""
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from flextool.flextoolrunner.runner_state import RunnerState

if TYPE_CHECKING:
    from flextool.flextoolrunner.solve_handoff import SolveHandoff


def _accepts_provider(fn) -> bool:
    """Return True iff *fn* accepts a ``provider`` keyword argument."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if "provider" in params:
        return True
    for p in params.values():
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def run(
    state: RunnerState,
    solve_name: str,
    *,
    prior_handoff: "SolveHandoff | None" = None,
    provider: "object | None" = None,
) -> None:
    """Native per-solve preprocessing orchestrator.

    Calls the native polars writers in the canonical order used by the
    legacy preprocessing chain.  When *provider* is non-None we thread
    it into every writer that accepts ``provider=``; otherwise the
    writers fall back to their workdir CSV path (off-cascade harness
    only — cascade callers always pass a Provider).
    """
    wf = state.paths.work_folder
    input_dir = wf / "input"
    solve_data_dir = wf / "solve_data"
    if provider is None:
        provider = getattr(state, "current_provider", None)
    _provider = provider

    def _call(fn, *args, **kwargs):
        if _provider is not None and _accepts_provider(fn):
            return fn(*args, provider=_provider, **kwargs)
        return fn(*args, **kwargs)

    # Native writer modules.
    from flextool.engine_polars import (
        _writer_arc_unions as _arc,
        _writer_calc_params as _calc,
        _writer_dispatchers as _disp,
        _writer_entity_annual as _entity_annual,
        _writer_inflow_scaling as _inflow_scaling,
        _writer_leaf_sets as _leaf,
        _writer_lp_scaling as _lp_scaling,
        _writer_mid_sets as _mid,
        _writer_per_solve as _per_solve,
        _writer_period_calc as _period_calc,
        _writer_period_params as _period,
        _writer_reserve as _reserve,
    )

    # ── Refresh write_input-scope outputs (idempotent, no-DB) ─────────
    # The mod's ``if p_model['solveFirst']`` printf blocks may have
    # overwritten these on a previous solve.
    _call(_leaf.write_period_param_sets, input_dir, solve_data_dir)
    _call(_leaf.write_invest_method_sets, input_dir, solve_data_dir)
    _call(_leaf.write_co2_method_sets, input_dir, solve_data_dir)
    _call(_leaf.write_optional_yes, input_dir, solve_data_dir)
    _call(_leaf.write_reserve_upDown_group, input_dir, solve_data_dir)
    _call(_leaf.write_group_loss_share, input_dir, solve_data_dir)
    _call(_mid.write_node_type_sets, input_dir, solve_data_dir)
    _call(_mid.write_entity_lifetime_method, input_dir, solve_data_dir)
    _call(_mid.write_process_ct_method, input_dir, solve_data_dir)
    _call(_mid.write_process_startup_method, input_dir, solve_data_dir)
    _call(_mid.write_node_inflow_method, input_dir, solve_data_dir)
    _call(_mid.write_node_storage_binding_method, input_dir, solve_data_dir)
    _call(_mid.write_process_group_inside_group_nonsync, input_dir, solve_data_dir)
    _call(_mid.write_process__sink_nonSync, input_dir, solve_data_dir)
    _call(_mid.write_group_entity, input_dir, solve_data_dir)
    _call(_mid.write_process_delayed__duration, input_dir, solve_data_dir)
    _call(_calc.write_process_method_projections, input_dir, solve_data_dir)
    _call(_calc.write_process_VRE, input_dir, solve_data_dir)
    _call(_calc.write_process_arc_method_joins, input_dir, solve_data_dir)
    _call(_calc.write_process_profile_method_joins, input_dir, solve_data_dir)
    _call(_mid.write_reserve_partitions, input_dir, solve_data_dir)
    _call(_mid.write_connection_param, input_dir, solve_data_dir)
    _call(_mid.write_nodegroup_dispatch_node, input_dir, solve_data_dir)
    _call(_mid.write_commodity_node_co2, input_dir, solve_data_dir)
    _call(_mid.write_process__commodity__node, input_dir, solve_data_dir)
    _call(_mid.write_process_coeff_zero_sets, input_dir, solve_data_dir)
    _call(_leaf.write_def_optional_yes, input_dir, solve_data_dir)
    _call(_leaf.write_process_delayed, input_dir, solve_data_dir)
    _call(_leaf.write_process_side, solve_data_dir)
    _call(_leaf.write_simple_setof_projections, input_dir, solve_data_dir)
    _call(_leaf.write_period_solve, solve_data_dir)
    _call(_leaf.write_time_set, input_dir, solve_data_dir)
    _call(_leaf.write_enable_optional_outputs, solve_data_dir)
    _call(_leaf.write_node_state_subsets, solve_data_dir)
    _call(_leaf.write_commodity_tier_sets, input_dir, solve_data_dir)
    _call(_mid.write_dc_angle_bounds, input_dir, solve_data_dir)
    _call(_mid.write_invest_total_sets, input_dir, solve_data_dir)
    _call(_mid.write_ci_ladder_cumulative, input_dir, solve_data_dir)
    _call(_disp.write_process_arc_unions, input_dir, solve_data_dir)
    _call(_arc.write_group_commodity_node_period_co2_total, input_dir, solve_data_dir)
    _call(_arc.write_param_in_use_sets, input_dir, solve_data_dir)

    # ── Per-solve sets (must run BEFORE period_calculated_params) ─────
    _call(_per_solve.write_per_solve_sets, solve_data_dir)

    # ── L1 batch 13: period_calculated_params ─────────────────────────
    _call(_period_calc.write_period_calculated_params, input_dir, solve_data_dir)
    # ── L1 batch 15: pdProcess / pdNode + edEntity_lifetime + ed_fixed_cost
    _call(_disp.write_entity_period_calc_params, input_dir, solve_data_dir)
    # ── L1 batch 16: ed_entity_annual family ──────────────────────────
    _call(_entity_annual.write_entity_annual_calc_params, input_dir, solve_data_dir)
    # ── L1 batch 17: node-inflow scaling family ───────────────────────
    _call(_inflow_scaling.write_node_inflow_scaling_params, input_dir, solve_data_dir)
    # ── L1 batch 18: LP-scaling row scalers ───────────────────────────
    _call(_lp_scaling.write_lp_scaling_params, input_dir, solve_data_dir)
    # ── L2 batch 19: invest/divest + edd_history ──────────────────────
    _call(_per_solve.write_invest_divest_sets, input_dir, solve_data_dir)
    # ── L2 batch 20: ed_*_period / ed_cumulative_* (needs batch 19) ───
    _call(_period.write_ed_period_params, input_dir, solve_data_dir)
    # ── L2 batch 21 ───────────────────────────────────────────────────
    _call(_arc.write_process_source_delayed_partition, input_dir, solve_data_dir)
    # ── L2 batch 22 ───────────────────────────────────────────────────
    _call(_arc.write_node_time_param_in_use, input_dir, solve_data_dir)
    # ── L2 batch 25 ───────────────────────────────────────────────────
    _call(_arc.write_process_source_sink_param_t, input_dir, solve_data_dir)
    # ── L2 batch 26: p_entity_pre_existing ────────────────────────────
    from flextool.engine_polars import _writer_chain_params as _chain
    _call(_chain.write_p_entity_pre_existing, input_dir, solve_data_dir)
    # ── L3 batch 27: ed_invest_forbidden_no_investment ────────────────
    _call(_per_solve.write_ed_invest_forbidden_no_investment, input_dir, solve_data_dir)
    # ── L4 batches 28-33 ──────────────────────────────────────────────
    _call(_arc.write_process_source_sink_delayed_partition, input_dir, solve_data_dir)
    _call(_arc.write_process_source_sink_is_node_family, input_dir, solve_data_dir)
    _call(_arc.write_process_source_sink_ramp_family, input_dir, solve_data_dir)
    _call(_arc.write_process_source_sink_coeff_zero, input_dir, solve_data_dir)
    _call(_arc.write_process_source_sink_ramp_method, input_dir, solve_data_dir)
    _call(_arc.write_node_group_dispatch_process_fully_inside, input_dir, solve_data_dir)
    # ── L5/L6 batch 62: 12 remaining nodeGroupDispatch sets ───────────
    _call(_arc.write_node_group_dispatch_sets, input_dir, solve_data_dir)
    # ── L4 batches 34-37 ──────────────────────────────────────────────
    _call(_arc.write_process_source_sink_param, input_dir, solve_data_dir)
    _call(_arc.write_process_source_sink_param_with_time, input_dir, solve_data_dir)
    _call(_arc.write_process_source_sink_profile_method_connection,
          input_dir, solve_data_dir)
    _call(_arc.write_process_source_is_node_sink_1way_no_sink_or_more_than_1_source,
          input_dir, solve_data_dir)
    # ── L4 batch 38: p_entity_divest_cumulative_max ───────────────────
    _call(_chain.write_p_entity_divest_cumulative_max, input_dir, solve_data_dir)
    # ── L4 batches 39-41 ──────────────────────────────────────────────
    _call(_arc.write_ed_history_realized_first, input_dir, solve_data_dir)
    _call(_arc.write_process_method_sources_sinks, input_dir, solve_data_dir)
    _call(_arc.write_peedt, input_dir, solve_data_dir)
    # ── L0 batches 42 / 43 ────────────────────────────────────────────
    from flextool.engine_polars import _writer_pdt_params as _pdt
    _call(_pdt.write_pdtProcess, input_dir, solve_data_dir)
    _call(_reserve.write_pdtReserve_upDown_group, input_dir, solve_data_dir)
    # ── L1 batch 44 (needs pdtReserve from batch 43) ──────────────────
    _call(_reserve.write_process_reserve_upDown_node_active_and_prundt,
          input_dir, solve_data_dir)
    # ── L0 batches 45 / 46 ────────────────────────────────────────────
    _call(_pdt.write_pdtProcess_source, input_dir, solve_data_dir)
    _call(_pdt.write_pdtProcess_sink, input_dir, solve_data_dir)
    # ── L4 batch 47 ───────────────────────────────────────────────────
    _call(_period.write_pdtProcess__source__sink__dt_varCost_pair,
          input_dir, solve_data_dir)
    # ── L2/L5 batch 48 ────────────────────────────────────────────────
    _call(_period.write_pssdt_varCost_filters, input_dir, solve_data_dir)
    # ── L0/L2 batch 49 ────────────────────────────────────────────────
    _call(_reserve.write_process_reserve_filters_and_reliability,
          input_dir, solve_data_dir)
    # ── L0 batch 50 ───────────────────────────────────────────────────
    _call(_period.write_cap_reduction_params, input_dir, solve_data_dir)
    # ── L0 batch 51 ───────────────────────────────────────────────────
    _call(_arc.write_pProcess_source_sink, input_dir, solve_data_dir)
    _call(_period.write_pdtCommodity, input_dir, solve_data_dir)
    # ── L0 batch 52 ───────────────────────────────────────────────────
    _call(_period.write_pdGroup, input_dir, solve_data_dir)
    _call(_period.write_pdtGroup, input_dir, solve_data_dir)
    _call(_period.write_pdCommodity, input_dir, solve_data_dir)
    # ── L0 batch 53 ───────────────────────────────────────────────────
    _call(_pdt.write_pdtNode, input_dir, solve_data_dir)
    # ── L8 batch 54 ───────────────────────────────────────────────────
    _call(_period.write_pdtNodeInflow, input_dir, solve_data_dir)
    # ── L0 batch 55 ───────────────────────────────────────────────────
    _call(_period.write_pdtProfile, input_dir, solve_data_dir)
    # ── L1 batch 56 ───────────────────────────────────────────────────
    _call(_period.write_pdtProcess_source_sink, input_dir, solve_data_dir)
    # ── L1/L2/L3 batch 57 ─────────────────────────────────────────────
    _call(_period.write_pdtConversion_rate_section_slope, input_dir, solve_data_dir)
    # ── L8 batch 58 ───────────────────────────────────────────────────
    _call(_period.write_p_positive_negative_inflow, input_dir, solve_data_dir)
    # ── L6/L7 batch 59: p_entity existing-capacity chain (handoff-aware)
    _call(_chain.write_p_entity_existing_chain,
          input_dir, solve_data_dir, prior_handoff=prior_handoff)
    # ── L9/L10 batch 60: capacity max chain ───────────────────────────
    _call(_chain.write_p_entity_capacity_max_chain, input_dir, solve_data_dir)
    # ── L4/L5 batch 61: process_source_sink_ramp_unions ───────────────
    _call(_arc.write_process_source_sink_ramp_unions, input_dir, solve_data_dir)
    # ── L0/L1 batch 63: branch weights + delay weight + co2 ───────────
    _call(_period_calc.write_branch_weights, input_dir, solve_data_dir)
    _call(_arc.write_p_process_delay_weight, input_dir, solve_data_dir)
    _call(_arc.write_gcndt_co2_price, input_dir, solve_data_dir)
    _call(_arc.write_group_commodity_node_period_co2_period, input_dir, solve_data_dir)
    # ── L0/L1 batch 64: param_t projections + instant-flow sets ───────
    _call(_arc.write_param_t_projections_and_time_params, input_dir, solve_data_dir)
    _call(_arc.write_gdt_instant_flow_sets, input_dir, solve_data_dir)
    # ── L0/L1 batch 65: small set derivations ─────────────────────────
    _call(_arc.write_small_set_derivations, input_dir, solve_data_dir)
    # ── L1 batch 66: state slack share + storage state reference price ─
    _call(_arc.write_p_state_slack_share, input_dir, solve_data_dir)
    _call(_arc.write_p_storage_state_reference_price, input_dir, solve_data_dir)
    # ── L2 batch 67: p_flow_min + p_flow_max ──────────────────────────
    _call(_arc.write_p_flow_min, input_dir, solve_data_dir)
    _call(_arc.write_p_flow_max, input_dir, solve_data_dir)
