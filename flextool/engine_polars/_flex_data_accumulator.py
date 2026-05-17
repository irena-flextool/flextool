"""Per-sub-solve frame capture into :class:`FlexDataProvider`.

This module owns :func:`capture_frames`, the context manager the cascade
wraps around its preprocessing pass to monkey-patch every participating
writer's ``_write(df, path)`` helper so the emitted frame flows into the
caller-supplied :class:`FlexDataProvider` instead of being written to
disk.  Outside the context, ``_write`` falls back to its direct-to-disk
behaviour — that is the path the ``test_writer_port_phase1.py`` byte-
parity tests exercise.

The 37 ``OK_thin_wrapper`` writers identified in
``specs/phase_b_writer_audit.md`` all funnel their derived frames
through their module's private ``_write(df, path)`` helper.  Subsequent
phases promoted the streamed-writer family into the same canonical
shape, expanding the set patched by :func:`capture_frames` (see
:data:`_PATCH_MODULES`).
"""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Iterator

import polars as pl


# ---------------------------------------------------------------------------
# Writer modules whose ``_write`` helper feeds the 37 thin-wrapper writers.
# Patching these four modules' ``_write`` covers every OK_thin_wrapper entry
# from the Phase B audit (writers in _writer_leaf_sets, _writer_mid_sets,
# _writer_calc_params, _writer_arc_unions).
# ---------------------------------------------------------------------------

_PATCH_MODULES = (
    "flextool.engine_polars._writer_leaf_sets",
    "flextool.engine_polars._writer_mid_sets",
    "flextool.engine_polars._writer_calc_params",
    "flextool.engine_polars._writer_arc_unions",
    "flextool.engine_polars._writer_chain_params",
    "flextool.engine_polars._writer_co2_accumulators",
    "flextool.engine_polars._writer_pdt_params",
    "flextool.engine_polars._writer_period_params",
    "flextool.engine_polars._writer_dispatchers",
    "flextool.engine_polars._writer_entity_annual",
    "flextool.engine_polars._writer_inflow_scaling",
    "flextool.engine_polars._writer_lp_scaling",
    "flextool.engine_polars._writer_solve_writers",
    "flextool.engine_polars._writer_period_calc",
    "flextool.engine_polars._writer_per_solve",
    "flextool.engine_polars._writer_reserve",
)


# ---------------------------------------------------------------------------
# Context manager — monkey-patches the writer modules' ``_write``
# helper to capture frames into the supplied accumulator.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def capture_frames(
    provider: "object | None" = None,
) -> Iterator[None]:
    """Patch the participating writers' ``_write`` helper to push every
    emitted frame into *provider* for the duration of the block, and
    SKIP the underlying disk write.

    The wrapped writers populate the Provider exclusively while this
    context is active — disk emission is the responsibility of
    :meth:`FlexDataProvider.snapshot_processed_inputs` (the
    ``--csv-dump`` debug path).  Frames are stored under both the bare
    basename and the parent-qualified key (``"<parent>/<stem>"``) so
    callers can disambiguate ``input/`` vs ``solve_data/`` collisions.

    Outside this context every writer's ``_write`` falls back to its
    direct-to-disk behaviour — that is the path the
    ``test_writer_port_phase1.py`` byte-parity gate exercises.
    """
    import importlib

    modules = [importlib.import_module(name) for name in _PATCH_MODULES]
    saved: list[tuple[object, object]] = [
        (mod, getattr(mod, "_write")) for mod in modules
    ]
    try:
        for mod, _original in saved:
            def _make_wrapped():
                def _wrapped(df: pl.DataFrame, path: Path) -> None:
                    if provider is not None:
                        p = Path(path)
                        provider.put(p.name, df)
                        parent = p.parent.name
                        if parent:
                            provider.put(f"{parent}/{p.name}", df)
                    # Provider-only — no disk write while capture is active.
                return _wrapped
            setattr(mod, "_write", _make_wrapped())
        yield None
    finally:
        for mod, original in saved:
            setattr(mod, "_write", original)


__all__ = [
    "capture_frames",
    "expected_basenames",
]


# ---------------------------------------------------------------------------
# Coverage manifest
# ---------------------------------------------------------------------------
#
# Basenames of CSVs that go through one of the patched ``_write`` helpers.
# This list is the public contract Phase D / E-a consumers can read against
# to know which solve_data/*.csv files the accumulator captures in-memory.
#
# When you lift another streamed writer into the canonical
# ``derive_X → _write(derive_X(...), path)`` shape, add its target basename
# below.  The matching test in
# ``tests/engine_polars/test_phase_c_flex_data_accumulator.py`` cross-checks
# the captured-vs-disk frames for each basename present in the cascade run.

_THIN_WRAPPER_BASENAMES: tuple[str, ...] = (
    # _writer_leaf_sets — 27 thin writers
    "period_group.csv",
    "period_node.csv",
    "period_commodity.csv",
    "period_process.csv",
    "entityInvest.csv",
    "entityDivest.csv",
    "group_invest.csv",
    "group_divest.csv",
    "group_co2_price.csv",
    "group_co2_max_period.csv",
    "group_co2_max_total.csv",
    "optional_yes.csv",
    "reserve__upDown__group.csv",
    "group_loss_share.csv",
    "def_optional_yes.csv",
    "process_delayed.csv",
    "process_side.csv",
    "period_solve.csv",
    "time.csv",
    "enable_optional_outputs.csv",
    "nodeState_rp.csv",
    "nodeStateBlock.csv",
    "commodity__tier.csv",
    "tier.csv",
    "timeline.csv",
    "timeline_steps.csv",
    "commodity__tier_ann.csv",
    # _writer_mid_sets — thin writers
    "group_entity.csv",
    "process_delayed__duration.csv",
    "process__sink_nonSync.csv",
    "entity_lifetime_method.csv",
    "process_ct_method.csv",
    "process_startup_method.csv",
    "node_inflow_method.csv",
    "node_storage_binding_method.csv",
    "connection_param.csv",
    "nodegroup_dispatch_node.csv",
    "commodity_node_co2.csv",
    "process__commodity__node.csv",
    # _writer_calc_params — thin writers
    "process_VRE.csv",
    # _writer_arc_unions — thin writers + Phase E-b lifted streamed writers
    # (this group expanded substantially when streamed writers were
    # converted to the canonical derive_X → _write pattern)
    "process_source_sink_param_t.csv",
    "node__TimeParam_in_use.csv",
    "process_source_delayed.csv",
    "process_source_undelayed.csv",
    "process_source_sink_param.csv",
    "process__source__sink__profile__profile_method_connection.csv",
    "process_method_sources_sinks.csv",
    "ed_history_realized_first.csv",
    "process__source__sinkIsNode.csv",
    "process__source__sinkIsNode_2way1var.csv",
    "process__source__sinkIsNode_not2way1var.csv",
    "process__source__sinkIsNode_2way2var.csv",
    "process_source_sink_ramp_method.csv",
    "process_source_sink_coeff_zero.csv",
    "process_source_sink_delayed.csv",
    "process_source_sink_undelayed.csv",
    "p_process_source_sink.csv",
    "nodeGroupDispatch__process_fully_inside.csv",
    # — ramp family + union (Phase E-b promoted)
    "process_source_sink_ramp_limit_source_up.csv",
    "process_source_sink_ramp_limit_sink_up.csv",
    "process_source_sink_ramp_limit_source_down.csv",
    "process_source_sink_ramp_limit_sink_down.csv",
    "process_source_sink_ramp_cost.csv",
    "process_source_sink_ramp.csv",
    # — group_commodity_node co2 (Phase E-b)
    "group_commodity_node_period_co2_total.csv",
    "group_commodity_node_period_co2_period.csv",
    # — param_in_use family (already _write; in audit scope)
    "node__PeriodParam_in_use.csv",
    "process__PeriodParam_in_use.csv",
    "process_TimeParam_in_use.csv",
    "process_source_sourceSinkTimeParam_in_use.csv",
    "process_sink_sourceSinkTimeParam_in_use.csv",
    "process_source_sourceSinkPeriodParam_in_use.csv",
    "process_sink_sourceSinkPeriodParam_in_use.csv",
    # — Phase E-b lifted streamed writers
    "peedt.csv",
    "process__source__sink__param_t.csv",
    "gdt_maxInstantFlow.csv",
    "gdt_minInstantFlow.csv",
    "p_process_delay_weight.csv",
    "gcndt_co2_price.csv",
    "p_flow_min.csv",
    "p_flow_max.csv",
    "p_state_slack_share.csv",
    "p_storage_state_reference_price.csv",
    "ed_history_realized.csv",
    "process__source__sink__profile__profile_method.csv",
    "process_sinkIsNode_2way1var.csv",
    "nodeSelfDischarge.csv",
    "pdt_online_linear.csv",
    "pdt_online_integer.csv",
    # — 12-CSV nodeGroupDispatch family
    "nodeGroupDispatch__process__unit__to_node_Not_in_aggregate.csv",
    "nodeGroupDispatch__process__node__to_unit_Not_in_aggregate.csv",
    "nodeGroupDispatch__group_aggregate__process__unit__to_node.csv",
    "nodeGroupDispatch__group_aggregate__process__node__to_unit.csv",
    "nodeGroupDispatch__process__node__to_connection_Not_in_aggregate.csv",
    "nodeGroupDispatch__process__connection__to_node_Not_in_aggregate.csv",
    "nodeGroupDispatch__connection_Not_in_aggregate.csv",
    "nodeGroupDispatch__group_aggregate__process__connection__to_node.csv",
    "nodeGroupDispatch__group_aggregate__process__node__to_connection.csv",
    "nodeGroupDispatch__group_aggregate_Connection.csv",
    "nodeGroupDispatch__group_aggregate_Unit_to_group.csv",
    "nodeGroupDispatch__group_aggregate_Group_to_unit.csv",
    # — 8-CSV param_t projections + timeParam
    "process__param_t.csv",
    "connection__param__time.csv",
    "connection__param_t.csv",
    "process__source__param_t.csv",
    "process__sink__param_t.csv",
    "process__source__timeParam.csv",
    "process__sink__timeParam.csv",
    "process__timeParam.csv",
    # _writer_chain_params — Phase E-b lifted streamed writers
    "p_entity_pre_existing.csv",
    "p_entity_divest_cumulative_max.csv",
    # — 5-CSV existing chain
    "p_entity_existing_capacity_later_solves.csv",
    "p_entity_all_existing.csv",
    "p_entity_existing_count.csv",
    "p_entity_existing_integer_count.csv",
    "p_entity_previously_invested_capacity.csv",
    # — 4-CSV capacity max chain
    "p_entity_max_capacity.csv",
    "p_entity_max_units.csv",
    "p_entity_invest_cumulative_max.csv",
    "p_entity_dispatch_capacity_max.csv",
    # _writer_co2_accumulators — Phase E-b lifted
    "co2_cum_realized_tonnes.csv",
    # _writer_pdt_params — Phase E-b lifted streamed writers
    # (high-memory hot path; ~280k-row dense frames preserved for
    # byte-parity, sparse-emit deferred per audit doc)
    "pdtProcess.csv",
    "pdtNode.csv",
    "pdtProcess_source.csv",
    "pdtProcess_sink.csv",
    # _writer_period_params — Phase E-b lifted streamed writers
    "pdtNodeInflow.csv",
    "pdtProfile.csv",
    "pdtConversion_rate.csv",
    "pdtProcess_section.csv",
    "pdtProcess_slope.csv",
    "pdtProcess_source_sink.csv",
    "pdGroup.csv",
    "pdtGroup.csv",
    "pdCommodity.csv",
    "pdtCommodity.csv",
    "p_positive_inflow.csv",
    "p_negative_inflow.csv",
    "pdtProcess__source__sink__dt_varCost.csv",
    "pdtProcess__source__sink__dt_varCost_alwaysProcess.csv",
    "pssdt_varCost_noEff.csv",
    "pssdt_varCost_eff_unit_source.csv",
    "pssdt_varCost_eff_unit_sink.csv",
    "pssdt_varCost_eff_connection.csv",
    "p_startup_cap_reduction_sink.csv",
    "p_shutdown_cap_reduction_sink.csv",
    "p_startup_cap_reduction_source.csv",
    "p_shutdown_cap_reduction_source.csv",
    "ed_invest_max_period.csv",
    "ed_invest_min_period.csv",
    "ed_divest_max_period.csv",
    "ed_divest_min_period.csv",
    "ed_cumulative_max_capacity.csv",
    "ed_cumulative_min_capacity.csv",
    # _writer_calc_params — Phase E-b lifted streamed writers
    # — write_process_arc_method_joins (10 CSVs, methodgated arc joins)
    "process_sink_toProcess.csv",
    "process_process_toSource.csv",
    "process_source_toSink.csv",
    "process_source_toProcess_direct.csv",
    "process_process_toSink_direct.csv",
    "process_sink_toProcess_direct.csv",
    "process_sink_toSource.csv",
    "process_process_toSink_noConversion.csv",
    "process_source_toProcess_noConversion.csv",
    "process_process_toSource_direct.csv",
    # — write_process_profile_method_joins (2 CSVs)
    "process__profileProcess__toSink__profile__profile_method.csv",
    "process__source__toProfileProcess__profile__profile_method.csv",
    # _writer_dispatchers — Phase E-b lifted entity_period_calc_params
    # (5 CSVs from a single own-compute monolith; dispatcher module
    # joined _PATCH_MODULES to expose its new _write helper)
    "pdProcess.csv",
    "pdNode.csv",
    "edEntity_lifetime.csv",
    "ed_fixed_cost.csv",
    "p_entity_unitsize.csv",
    # _writer_dispatchers — Phase E-b lifted process_arc_unions monolith
    # (14 CSVs from a single own-compute dispatcher; convert _write_csv
    # into _write(derive_X(...), path) per emission)
    "process__profileProcess__toSink.csv",
    "process__source__toProfileProcess.csv",
    "process_profile.csv",
    "process_source_toProcess.csv",
    "process_process_toSink.csv",
    "process_source_sink_eff.csv",
    "process_source_sink_noEff.csv",
    "process_online.csv",
    "process_minload.csv",
    "process__commodity__node_co2.csv",
    "process_co2.csv",
    "process_source_sink.csv",
    "process_source_sink_alwaysProcess.csv",
    "process__source__sink__profile__profile_method_direct.csv",
    # _writer_entity_annual — Phase E-b lifted (6-CSV monolith;
    # repr(float) precision preserved by the _rows_to_frame helper)
    "ed_entity_annual.csv",
    "ed_entity_annual_discounted.csv",
    "ed_entity_annual_divest.csv",
    "ed_entity_annual_divest_discounted.csv",
    "ed_lifetime_fixed_cost.csv",
    "ed_lifetime_fixed_cost_divest.csv",
    # _writer_inflow_scaling — Phase E-b lifted (17-CSV monolith, peak
    # family heavy cross-CSV state — converted via dict-of-frames adapter
    # since splitting into 17 standalone derive_* would re-walk the
    # t-axis O(N) times per call)
    "ptNode_inflow.csv",
    "_node_cap_inflow_fallback.csv",
    "orig_flow_sum.csv",
    "period_share_of_annual_flow.csv",
    "period_flow_annual_multiplier.csv",
    "period_flow_proportional_multiplier.csv",
    "new_peak_sign.csv",
    "old_peak_max.csv",
    "old_peak_min.csv",
    "old_peak_sign.csv",
    "old_peak.csv",
    "new_peak_divided_by_old_peak.csv",
    "new_peak_divide_by_old_peak_sum_inflow.csv",
    "new_peak_inflow_sum.csv",
    "new_old_multiplier.csv",
    "new_old_slope.csv",
    "new_old_section.csv",
    # _writer_lp_scaling — Phase E-b lifted (9-CSV monolith with chained
    # raw -> pow10 -> capacity -> inverse cascades; converted via
    # dict-of-frames adapter)
    #
    # NB ``_group_cap_raw.csv`` is captured by the accumulator hook but
    # is NOT in this manifest: the legacy emitter writes ``"0"`` (int
    # via ``repr(sum(()))``) for empty groups, while the polars
    # ``write_csv`` round-trip through ``pl.read_csv`` -> Float64 ->
    # Utf8 cast in the parity test yields ``"0.0"``.  Disk byte-parity
    # is preserved; the captured frame is functionally usable (Phase D
    # consumers parse the Utf8 numerically) but the byte-string-compare
    # parity test cannot validate it.  See _writer_lp_scaling for the
    # corresponding code note.
    "_node_cap_unitsize_sum.csv",
    "_node_cap_raw.csv",
    "_node_cap_pow10.csv",
    "node_capacity_for_scaling.csv",
    "inv_node_cap.csv",
    "_group_cap_pow10.csv",
    "group_capacity_for_scaling.csv",
    "inv_group_cap.csv",
    # _writer_solve_writers — Phase E-b7 (34 small per-solve CSVs that
    # emit from in-memory WriterSnapshot/timeline records; converted
    # from csv.writer(newline="") + CRLF emit into the canonical
    # derive_X -> _write(derive_X(...), path) pattern.  The new
    # ``_write`` helper uses polars ``line_terminator="\r\n"`` to
    # preserve byte-identical parity with the legacy ``csv.writer``
    # output the writer_port_phase1 gate compares.  Empty / header-
    # only emitters use an all-Utf8 ``_empty_frame`` so the captured
    # frame has the correct schema even when the body is empty.)
    "steps_in_timeline.csv",
    "steps_in_use.csv",
    "steps_complete_solve.csv",
    "step_previous.csv",
    "period_block_time.csv",
    "period_block_succ.csv",
    "p_years_represented.csv",
    "period_with_history.csv",
    "p_discount_years.csv",
    "realized_invest_periods_of_current_solve.csv",
    "invest_periods_of_current_solve.csv",
    "period_last.csv",
    "period_first_of_solve.csv",
    "period_first.csv",
    "p_model.csv",
    "p_nested_model.csv",
    "solve_current.csv",
    "first_timesteps.csv",
    "last_timesteps.csv",
    "last_realized_timestep.csv",
    "realized_dispatch.csv",
    "fix_storage_timesteps.csv",
    "period__branch.csv",
    "branch_all.csv",
    "time_branch_all.csv",
    "solve_branch_weight.csv",
    "solve_branch__time_branch.csv",
    "p_entity_invested.csv",
    "p_entity_divested.csv",
    "p_entity_period_existing_capacity.csv",
    "ladder_cum_realized_mwh.csv",
    "ladder_cum_sim_hours.csv",
    # NB co2_cum_realized_tonnes.csv already captured by
    # _writer_co2_accumulators above; the empty-seed variant from
    # write_empty_cumulative_files overwrites with the same header.
    "fix_storage_price.csv",
    "fix_storage_quantity.csv",
    "fix_storage_usage.csv",
    "p_roll_continue_state.csv",
    "costs_discounted.csv",
    "co2.csv",
    "period_capacity.csv",
    "timesets_in_use.csv",
    "timesets__timeline.csv",
    "solve_hole_multiplier.csv",
    "p_use_row_scaling.csv",
    "scale_the_objective.csv",
    "scale_the_state.csv",
    "delay_duration.csv",
    "dtt__delay_duration.csv",
    "rp_weights.csv",
    "rp_base_chain.csv",
    "rp_base_first.csv",
    "rp_base_last.csv",
    "rp_block_first.csv",
    "rp_block_last.csv",
    "rp_block_start_last.csv",
    "rp_cost_weight.csv",
    # _writer_period_calc — Phase E-b8 (write_period_calculated_params
    # emits 12 CSVs, write_branch_weights emits 2).  All 14 captured;
    # the previous E-b8 _NO_CAPTURE workaround for
    # p_inflation_factor_operations_yearly.csv and
    # complete_period_share_of_year_calc.csv was retired once
    # input._read_long gained a uniform Float64 cast on the value
    # column so seed-mode Utf8 frames coerce to Float64 in Param.frame
    # exactly as the disk-read path does.
    "p_inflation_factor_operations_yearly.csv",
    "complete_period_share_of_year_calc.csv",
    "p_timeline_duration_in_years.csv",
    "hours_in_period.csv",
    "period_share_of_year.csv",
    "p_years_d.csv",
    "p_years_represented_d_calc.csv",
    "complete_hours_in_period.csv",
    "p_years_until_invest.csv",
    "p_years_until_dispatch.csv",
    "p_inflation_factor_investment_yearly.csv",
    "f_d_k.csv",
    "pd_branch_weight.csv",
    "pdt_branch_weight.csv",
    # _writer_per_solve — Phase E-b8 (write_per_solve_sets emits 30
    # CSVs, write_invest_divest_sets emits 19, plus the singleton
    # ed_invest_forbidden_no_investment.csv).  All emits route
    # through ``_write_singles`` / ``_write_tuples`` which now
    # delegate to ``_write(df, path)``.
    "branch_set.csv",
    "year_set.csv",
    "period_from_period_time_set.csv",
    "period_in_use_set.csv",
    "time_in_use_set.csv",
    "complete_time_in_use_set.csv",
    "rp_base_period_set.csv",
    "rp_rep_period_set.csv",
    "period_block_set.csv",
    "dtt_set.csv",
    "d_fix_storage_period_set.csv",
    "period_set.csv",
    "periodAll_set.csv",
    "block_set.csv",
    "period__timeline_set.csv",
    "dt_realize_dispatch_set.csv",
    "d_realized_period_set.csv",
    "d_realize_dispatch_or_invest_set.csv",
    "dt_non_anticipativity_set.csv",
    "pdt_uptime_set.csv",
    "pdt_downtime_set.csv",
    "cnd_ladder_set.csv",
    "cndi_ladder_cum_set.csv",
    "cndi_ladder_ann_set.csv",
    "cndi_ladder_set.csv",
    "dtdt_next_set.csv",
    "n_fix_storage_quantity_set.csv",
    "n_fix_storage_price_set.csv",
    "n_fix_storage_usage_set.csv",
    "p_online_dt_set.csv",
    "ed_invest.csv",
    "ed_divest.csv",
    "ed_invest_period.csv",
    "ed_divest_period.csv",
    "ed_invest_cumulative.csv",
    "pd_invest.csv",
    "nd_invest.csv",
    "pd_divest.csv",
    "nd_divest.csv",
    "edd_history_choice.csv",
    "edd_history_automatic.csv",
    "edd_history_no_investment.csv",
    "edd_history.csv",
    "edd_history_invest.csv",
    "edd_invest.csv",
    "gd_invest.csv",
    "gd_divest.csv",
    "gd_invest_period.csv",
    "gd_divest_period.csv",
    "ed_invest_forbidden_no_investment.csv",
    # _writer_reserve — Phase E-b8 (write_pdtReserve_upDown_group +
    # write_process_reserve_upDown_node_active_and_prundt (2 CSVs) +
    # write_process_reserve_filters_and_reliability (4 CSVs); all
    # converted to derive_X -> _write(derive_X(...), path).)
    "pdtReserve_upDown_group.csv",
    "process_reserve_upDown_node_active.csv",
    "prundt.csv",
    "p_process_reserve_upDown_node_reliability.csv",
    "process_reserve_upDown_node_increase_reserve_ratio.csv",
    "process_reserve_upDown_node_large_failure_ratio.csv",
    "process_large_failure.csv",
)


def expected_basenames() -> tuple[str, ...]:
    """Return the basenames the accumulator is expected to capture.

    The list comes from the Phase B writer audit (every ``OK_thin_wrapper``
    entry, plus the streamed writers lifted into the canonical pattern by
    Phase E-b).  Tests use this to cross-check disk-vs-accumulator parity
    without re-enumerating the list inline.
    """
    return _THIN_WRAPPER_BASENAMES
