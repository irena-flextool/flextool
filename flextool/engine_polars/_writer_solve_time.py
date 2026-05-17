"""Writer-port Phase 2 (sub-dispatch 8) — ``preprocessing/solve_time.run``.

Native port of the per-solve preprocessing orchestrator
``flextool.flextoolrunner.preprocessing.solve_time.run`` (legacy 328 LOC,
single public ``run`` function).

This module is the closeout of Phase 2: every other preprocessing helper
called from this orchestrator has been ported in sub-dispatches 1-7 of
this phase (and in Phase 1).  The native :func:`run` therefore simply
sequences the same call list in the same order; each individual writer
is intercepted by the override hook in
:func:`flextool.engine_polars._native_input_writer._native_leaf_set_override`
and routes through its native polars implementation.

Function-signature parity
-------------------------

The signature matches the legacy one exactly:

* positional ``state: RunnerState``
* positional ``solve_name: str``
* keyword-only ``prior_handoff: SolveHandoff | None = None``

``state`` is consumed only as a path carrier — we touch
``state.paths.work_folder`` to derive ``input_dir`` / ``solve_data_dir``.
``solve_name`` is accepted (and forwarded internally) for signature
parity but, like in the legacy module, is not used by any of the sub-
writers (per-solve identification flows through CSVs in
``solve_data/``).  ``prior_handoff`` is threaded into
``entity_period_calc_params.write_p_entity_existing_chain`` exactly as
legacy.

Ordering parity
---------------

The legacy ordering carries dependency contracts between sub-writers
(e.g. ``per_solve_sets`` produces ``period_in_use_set.csv`` consumed by
``period_calculated_params``; ``entity_period_calc_params`` writes
``pdProcess`` / ``pdNode`` consumed by ``write_p_entity_existing_chain``;
batch 19 ``ed_invest`` / ``ed_divest`` are consumed by batch 20 / 27).
We preserve the exact call order to keep those contracts intact.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from flextool.flextoolrunner.runner_state import RunnerState

if TYPE_CHECKING:
    from flextool.flextoolrunner.solve_handoff import SolveHandoff


def _accepts_provider(fn) -> bool:
    """Return True iff *fn* accepts a ``provider`` keyword argument.

    Used by the dispatcher to thread ``provider=`` only into writer
    entry points that have been migrated to the Provider-first reader
    pattern.  Step 2 will delete this helper once every writer in the
    legacy preprocessing modules accepts ``provider=`` unconditionally.
    """
    import inspect
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if "provider" in params:
        return True
    # Accept **kwargs as a permissive sink.
    for p in params.values():
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def _call(fn, *args, provider=None, **kwargs):
    """Invoke *fn* with ``provider=`` only when its signature accepts it.

    Transitional helper for the S1-g-4 migration sweep — see the
    handoff doc.  Keeps the dispatcher diff small while still threading
    the live Provider into the writer chain.
    """
    if provider is not None and _accepts_provider(fn):
        return fn(*args, provider=provider, **kwargs)
    return fn(*args, **kwargs)


def run(
    state: RunnerState,
    solve_name: str,
    *,
    prior_handoff: "SolveHandoff | None" = None,
    provider: "object | None" = None,
) -> None:
    """Native per-solve preprocessing orchestrator.

    Mirrors :func:`flextool.flextoolrunner.preprocessing.solve_time.run`
    call-for-call.  Each sub-writer is dispatched through the legacy
    module attribute name so the override hook installed by
    :func:`flextool.engine_polars._native_input_writer._native_leaf_set_override`
    can route to the native implementation when present (and falls
    through to legacy code for any helper not yet ported).

    Step 1-g-4 — the optional *provider* kwarg is the live
    :class:`FlexDataProvider` for the current sub-solve, pulled from
    ``state.current_provider`` by the cascade orchestrator.  We thread
    it into every writer call via :func:`_call`, which checks each
    writer's signature and passes ``provider=`` only when the writer
    accepts it.  Writers that haven't been migrated yet still read
    from the legacy seed funnel; Step 2 deletes both paths.
    """
    wf = state.paths.work_folder
    input_dir = wf / "input"
    solve_data_dir = wf / "solve_data"
    if provider is None:
        provider = getattr(state, "current_provider", None)

    # Bind the dispatcher's provider into a closure that wraps every
    # ``module.write_X(...)`` call below.  Each call site is a thin
    # ``_call(module.write_X, ...)`` wrapper that defers to
    # :func:`_call` (signature-aware), forwarding *provider* exactly
    # when the called writer accepts it.
    _provider = provider

    def _call(fn, *args, **kwargs):  # noqa: E306 — re-bound per invocation
        if _provider is not None and _accepts_provider(fn):
            return fn(*args, provider=_provider, **kwargs)
        return fn(*args, **kwargs)

    # Import the legacy preprocessing modules; the override hook
    # monkey-patches their ``write_*`` attributes for the duration of
    # the cascade, so calling through these module references in turn
    # invokes the native implementations.
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

    # ── Refresh write_input-scope outputs (idempotent, no-DB) ─────────
    # The mod's ``if p_model['solveFirst']`` printf blocks may have
    # overwritten these on a previous solve.
    _call(period_param_sets.write_period_param_sets, input_dir, solve_data_dir)
    _call(invest_method_sets.write_invest_method_sets, input_dir, solve_data_dir)
    _call(co2_method_sets.write_co2_method_sets, input_dir, solve_data_dir)
    _call(simple_projections.write_optional_yes, input_dir, solve_data_dir)
    _call(simple_projections.write_reserve_upDown_group, input_dir, solve_data_dir)
    _call(simple_projections.write_group_loss_share, input_dir, solve_data_dir)
    _call(node_type_sets.write_node_type_sets, input_dir, solve_data_dir)
    _call(method_with_fallback_sets.write_entity_lifetime_method, input_dir, solve_data_dir)
    _call(method_with_fallback_sets.write_process_ct_method, input_dir, solve_data_dir)
    _call(method_with_fallback_sets.write_process_startup_method, input_dir, solve_data_dir)
    _call(method_with_fallback_sets.write_node_inflow_method, input_dir, solve_data_dir)
    _call(method_with_fallback_sets.write_node_storage_binding_method, input_dir, solve_data_dir)
    _call(nonsync_sets.write_process_group_inside_group_nonsync, input_dir, solve_data_dir)
    _call(nonsync_sets.write_process__sink_nonSync, input_dir, solve_data_dir)
    _call(union_sets.write_group_entity, input_dir, solve_data_dir)
    _call(union_sets.write_process_delayed__duration, input_dir, solve_data_dir)
    _call(process_method_sets.write_process_method_projections, input_dir, solve_data_dir)
    _call(process_method_sets.write_process_VRE, input_dir, solve_data_dir)
    _call(process_method_sets.write_process_arc_method_joins, input_dir, solve_data_dir)
    _call(process_method_sets.write_process_profile_method_joins, input_dir, solve_data_dir)
    _call(reserve_method_partitions.write_reserve_partitions, input_dir, solve_data_dir)
    _call(structural_filters.write_connection_param, input_dir, solve_data_dir)
    _call(structural_filters.write_nodegroup_dispatch_node, input_dir, solve_data_dir)
    _call(structural_filters.write_commodity_node_co2, input_dir, solve_data_dir)
    _call(structural_filters.write_process__commodity__node, input_dir, solve_data_dir)
    _call(structural_filters.write_process_coeff_zero_sets, input_dir, solve_data_dir)
    _call(simple_projections.write_def_optional_yes, input_dir, solve_data_dir)
    _call(simple_projections.write_process_delayed, input_dir, solve_data_dir)
    _call(simple_projections.write_process_side, solve_data_dir)
    _call(simple_projections.write_simple_setof_projections, input_dir, solve_data_dir)
    _call(simple_projections.write_period_solve, solve_data_dir)
    _call(simple_projections.write_time_set, input_dir, solve_data_dir)
    _call(simple_projections.write_enable_optional_outputs, solve_data_dir)
    _call(simple_projections.write_node_state_subsets, solve_data_dir)
    _call(simple_projections.write_commodity_tier_sets, input_dir, solve_data_dir)
    _call(dc_angle_bounds.write_dc_angle_bounds, input_dir, solve_data_dir)
    _call(invest_total_sets.write_invest_total_sets, input_dir, solve_data_dir)
    _call(invest_total_sets.write_ci_ladder_cumulative, input_dir, solve_data_dir)
    _call(process_arc_unions.write_process_arc_unions, input_dir, solve_data_dir)
    _call(process_arc_unions.write_group_commodity_node_period_co2_total, input_dir, solve_data_dir)
    _call(process_arc_unions.write_param_in_use_sets, input_dir, solve_data_dir)

    # ── Per-solve sets (must run BEFORE period_calculated_params) ─────
    _call(per_solve_sets.write_per_solve_sets, solve_data_dir)

    # ── L1 batch 13: period_calculated_params ─────────────────────────
    _call(period_calculated_params.write_period_calculated_params, input_dir, solve_data_dir)
    # ── L1 batch 15: pdProcess / pdNode + edEntity_lifetime + ed_fixed_cost
    _call(entity_period_calc_params.write_entity_period_calc_params, input_dir, solve_data_dir)
    # ── L1 batch 16: ed_entity_annual family ──────────────────────────
    _call(entity_annual_calc_params.write_entity_annual_calc_params, input_dir, solve_data_dir)
    # ── L1 batch 17: node-inflow scaling family ───────────────────────
    _call(node_inflow_scaling_params.write_node_inflow_scaling_params, input_dir, solve_data_dir)
    # ── L1 batch 18: LP-scaling row scalers ───────────────────────────
    _call(lp_scaling_params.write_lp_scaling_params, input_dir, solve_data_dir)
    # ── L2 batch 19: invest/divest + edd_history ──────────────────────
    _call(invest_divest_sets.write_invest_divest_sets, input_dir, solve_data_dir)
    # ── L2 batch 20: ed_*_period / ed_cumulative_* (needs batch 19) ───
    _call(entity_period_calc_params.write_ed_period_params, input_dir, solve_data_dir)
    # ── L2 batch 21 ───────────────────────────────────────────────────
    _call(process_arc_unions.write_process_source_delayed_partition, input_dir, solve_data_dir)
    # ── L2 batch 22 ───────────────────────────────────────────────────
    _call(process_arc_unions.write_node_time_param_in_use, input_dir, solve_data_dir)
    # ── L2 batch 25 ───────────────────────────────────────────────────
    _call(process_arc_unions.write_process_source_sink_param_t, input_dir, solve_data_dir)
    # ── L2 batch 26: p_entity_pre_existing ────────────────────────────
    _call(entity_period_calc_params.write_p_entity_pre_existing, input_dir, solve_data_dir)
    # ── L3 batch 27: ed_invest_forbidden_no_investment ────────────────
    _call(invest_divest_sets.write_ed_invest_forbidden_no_investment, input_dir, solve_data_dir)
    # ── L4 batches 28-33 ──────────────────────────────────────────────
    _call(process_arc_unions.write_process_source_sink_delayed_partition, input_dir, solve_data_dir)
    _call(process_arc_unions.write_process_source_sink_is_node_family, input_dir, solve_data_dir)
    _call(process_arc_unions.write_process_source_sink_ramp_family, input_dir, solve_data_dir)
    _call(process_arc_unions.write_process_source_sink_coeff_zero, input_dir, solve_data_dir)
    _call(process_arc_unions.write_process_source_sink_ramp_method, input_dir, solve_data_dir)
    _call(process_arc_unions.write_node_group_dispatch_process_fully_inside, input_dir, solve_data_dir)
    # ── L5/L6 batch 62: 12 remaining nodeGroupDispatch sets ───────────
    _call(process_arc_unions.write_node_group_dispatch_sets, input_dir, solve_data_dir)
    # ── L4 batches 34-37 ──────────────────────────────────────────────
    _call(process_arc_unions.write_process_source_sink_param, input_dir, solve_data_dir)
    _call(process_arc_unions.write_process_source_sink_param_with_time, input_dir, solve_data_dir)
    _call(process_arc_unions.write_process_source_sink_profile_method_connection, 
        input_dir, solve_data_dir,
    )
    _call(process_arc_unions.write_process_source_is_node_sink_1way_no_sink_or_more_than_1_source, 
        input_dir, solve_data_dir,
    )
    # ── L4 batch 38: p_entity_divest_cumulative_max ───────────────────
    _call(entity_period_calc_params.write_p_entity_divest_cumulative_max, input_dir, solve_data_dir)
    # ── L4 batches 39-41 ──────────────────────────────────────────────
    _call(process_arc_unions.write_ed_history_realized_first, input_dir, solve_data_dir)
    _call(process_arc_unions.write_process_method_sources_sinks, input_dir, solve_data_dir)
    _call(process_arc_unions.write_peedt, input_dir, solve_data_dir)
    # ── L0 batches 42 / 43 ────────────────────────────────────────────
    _call(entity_period_calc_params.write_pdtProcess, input_dir, solve_data_dir)
    _call(reserve_calc_params.write_pdtReserve_upDown_group, input_dir, solve_data_dir)
    # ── L1 batch 44 (needs pdtReserve from batch 43) ──────────────────
    _call(reserve_calc_params.write_process_reserve_upDown_node_active_and_prundt, 
        input_dir, solve_data_dir,
    )
    # ── L0 batches 45 / 46 ────────────────────────────────────────────
    _call(entity_period_calc_params.write_pdtProcess_source, input_dir, solve_data_dir)
    _call(entity_period_calc_params.write_pdtProcess_sink, input_dir, solve_data_dir)
    # ── L4 batch 47 ───────────────────────────────────────────────────
    _call(entity_period_calc_params.write_pdtProcess__source__sink__dt_varCost_pair, 
        input_dir, solve_data_dir,
    )
    # ── L2/L5 batch 48 ────────────────────────────────────────────────
    _call(entity_period_calc_params.write_pssdt_varCost_filters, input_dir, solve_data_dir)
    # ── L0/L2 batch 49 ────────────────────────────────────────────────
    _call(reserve_calc_params.write_process_reserve_filters_and_reliability, 
        input_dir, solve_data_dir,
    )
    # ── L0 batch 50 ───────────────────────────────────────────────────
    _call(entity_period_calc_params.write_cap_reduction_params, input_dir, solve_data_dir)
    # ── L0 batch 51 ───────────────────────────────────────────────────
    _call(entity_period_calc_params.write_pProcess_source_sink, input_dir, solve_data_dir)
    _call(entity_period_calc_params.write_pdtCommodity, input_dir, solve_data_dir)
    # ── L0 batch 52 ───────────────────────────────────────────────────
    _call(entity_period_calc_params.write_pdGroup, input_dir, solve_data_dir)
    _call(entity_period_calc_params.write_pdtGroup, input_dir, solve_data_dir)
    _call(entity_period_calc_params.write_pdCommodity, input_dir, solve_data_dir)
    # ── L0 batch 53 ───────────────────────────────────────────────────
    _call(entity_period_calc_params.write_pdtNode, input_dir, solve_data_dir)
    # ── L8 batch 54 ───────────────────────────────────────────────────
    _call(entity_period_calc_params.write_pdtNodeInflow, input_dir, solve_data_dir)
    # ── L0 batch 55 ───────────────────────────────────────────────────
    _call(entity_period_calc_params.write_pdtProfile, input_dir, solve_data_dir)
    # ── L1 batch 56 ───────────────────────────────────────────────────
    _call(entity_period_calc_params.write_pdtProcess_source_sink, input_dir, solve_data_dir)
    # ── L1/L2/L3 batch 57 ─────────────────────────────────────────────
    _call(entity_period_calc_params.write_pdtConversion_rate_section_slope, 
        input_dir, solve_data_dir,
    )
    # ── L8 batch 58 ───────────────────────────────────────────────────
    _call(entity_period_calc_params.write_p_positive_negative_inflow, input_dir, solve_data_dir)
    # ── L6/L7 batch 59: p_entity existing-capacity chain (handoff-aware)
    _call(entity_period_calc_params.write_p_entity_existing_chain, 
        input_dir, solve_data_dir, prior_handoff=prior_handoff,
    )
    # ── L9/L10 batch 60: capacity max chain ───────────────────────────
    _call(entity_period_calc_params.write_p_entity_capacity_max_chain, 
        input_dir, solve_data_dir,
    )
    # ── L4/L5 batch 61: process_source_sink_ramp_unions ───────────────
    _call(process_arc_unions.write_process_source_sink_ramp_unions, input_dir, solve_data_dir)
    # ── L0/L1 batch 63: branch weights + delay weight + co2 ───────────
    _call(period_calculated_params.write_branch_weights, input_dir, solve_data_dir)
    _call(process_arc_unions.write_p_process_delay_weight, input_dir, solve_data_dir)
    _call(process_arc_unions.write_gcndt_co2_price, input_dir, solve_data_dir)
    _call(process_arc_unions.write_group_commodity_node_period_co2_period, input_dir, solve_data_dir)
    # ── L0/L1 batch 64: param_t projections + instant-flow sets ───────
    _call(process_arc_unions.write_param_t_projections_and_time_params, input_dir, solve_data_dir)
    _call(process_arc_unions.write_gdt_instant_flow_sets, input_dir, solve_data_dir)
    # ── L0/L1 batch 65: small set derivations ─────────────────────────
    _call(process_arc_unions.write_small_set_derivations, input_dir, solve_data_dir)
    # ── L1 batch 66: state slack share + storage state reference price ─
    _call(process_arc_unions.write_p_state_slack_share, input_dir, solve_data_dir)
    _call(process_arc_unions.write_p_storage_state_reference_price, input_dir, solve_data_dir)
    # ── L2 batch 67: p_flow_min + p_flow_max ──────────────────────────
    _call(process_arc_unions.write_p_flow_min, input_dir, solve_data_dir)
    _call(process_arc_unions.write_p_flow_max, input_dir, solve_data_dir)
