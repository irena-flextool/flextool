"""Per-sub-solve frame capture coverage manifest.

This module hosts :func:`expected_basenames` — the public-contract list
of CSV basenames the cascade's emit_* functions push into the
in-memory :class:`FlexDataProvider`.  Tests cross-check this manifest
against the disk-resident frames produced by ``--csv-dump``.

Prior to the writer→emitter refactor this module also exposed a
``capture_frames`` context manager that monkey-patched every
participating writer's ``_write(df, path)`` helper to redirect frames
into a Provider.  Phase 3 of the refactor migrated every call site to
``emit_*(..., provider=...)``; the monkey-patch (and the ``_PATCH_MODULES``
tuple it consumed) is gone.  Only the basename manifest remains.
"""
from __future__ import annotations


__all__ = [
    "expected_basenames",
]


# ---------------------------------------------------------------------------
# Coverage manifest
# ---------------------------------------------------------------------------
#
# Basenames of CSVs the cascade's emit_* functions push into the
# in-memory :class:`FlexDataProvider`.  This list is the public contract
# Phase D / E-a consumers read against to know which solve_data/*.csv
# files the cascade captures.
#
# When a new emit_* lands that adds a previously-uncaptured frame, add
# its target basename below.  The matching test in
# ``tests/engine_polars/test_phase_c_flex_data_accumulator.py`` cross-
# checks the in-Provider-vs-disk frames for each basename present in
# the cascade run.

_THIN_WRAPPER_BASENAMES: tuple[str, ...] = (
    # _emit_leaf_sets — 27 thin writers
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
    # _emit_mid_sets — thin writers.
    #
    # Phase 4 audit (2026-05-21) corrected the spelling of seven
    # entries below to match the actual emit keys produced by the
    # cascade.  The legacy single-underscore names
    # (``entity_lifetime_method.csv`` etc.) had been stale in this
    # manifest since pre-Phase-1; the corresponding writers have
    # always emitted the double-underscore (or camelCase) variant.
    # Tests never consulted this manifest until Phase 4 introduced
    # ``test_csv_dump_post_refactor`` so the drift was undetected.
    "group_entity.csv",
    "process_delayed__duration.csv",
    "process__sink_nonSync.csv",
    "entity__lifetime_method.csv",
    "process__ct_method.csv",
    "process__startup_method.csv",
    "node__inflow_method.csv",
    "node__storage_binding_method.csv",
    "connection__param.csv",
    "nodeGroupDispatch_node.csv",
    "commodity_node_co2.csv",
    "process__commodity__node.csv",
    # _emit_calc_params — thin writers
    "process_VRE.csv",
    # _emit_arc_unions — thin writers + Phase E-b lifted streamed writers
    "process_source_sink_param_t.csv",
    "node__TimeParam_in_use.csv",
    "process_source_delayed.csv",
    "process_source_undelayed.csv",
    "process__source__sink__param.csv",
    "process__source__sink__profile__profile_method_connection.csv",
    "process_method_sources_sinks.csv",
    "ed_history_realized_first.csv",
    "process__source__sinkIsNode.csv",
    "process__source__sinkIsNode_2way1var.csv",
    "process__source__sinkIsNode_not2way1var.csv",
    "process__source__sinkIsNode_2way2var.csv",
    "process__source__sink__ramp_method.csv",
    "process_source_sink_coeff_zero.csv",
    "process_source_sink_delayed.csv",
    "process_source_sink_undelayed.csv",
    "pProcess_source_sink.csv",
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
    # — param_in_use family
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
    # _emit_chain_params — Phase E-b lifted streamed writers
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
    # _emit_co2_accumulators — Phase E-b lifted
    "co2_cum_realized_tonnes.csv",
    # _emit_pdt_params — Phase E-b lifted streamed writers
    "pdtProcess.csv",
    "pdtNode.csv",
    "pdtProcess_source.csv",
    "pdtProcess_sink.csv",
    # _emit_period_params — Phase E-b lifted streamed writers
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
    # _emit_calc_params — Phase E-b lifted streamed writers
    # — process_arc_method_joins (10 CSVs, methodgated arc joins)
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
    # — process_profile_method_joins (2 CSVs)
    "process__profileProcess__toSink__profile__profile_method.csv",
    "process__source__toProfileProcess__profile__profile_method.csv",
    # _emit_dispatchers — Phase E-b lifted entity_period_calc_params
    "pdProcess.csv",
    "pdNode.csv",
    "edEntity_lifetime.csv",
    "ed_fixed_cost.csv",
    "p_entity_unitsize.csv",
    # _emit_dispatchers — Phase E-b lifted process_arc_unions monolith
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
    # _emit_entity_annual — Phase E-b lifted (6-CSV monolith)
    "ed_entity_annual.csv",
    "ed_entity_annual_discounted.csv",
    "ed_entity_annual_divest.csv",
    "ed_entity_annual_divest_discounted.csv",
    "ed_lifetime_fixed_cost.csv",
    "ed_lifetime_fixed_cost_divest.csv",
    # _emit_inflow_scaling — Phase E-b lifted (17-CSV monolith)
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
    # _emit_lp_scaling — Phase E-b lifted (9-CSV monolith)
    #
    # NB ``_group_cap_raw.csv`` is captured by the accumulator hook but
    # is NOT in this manifest: the legacy emitter writes ``"0"`` (int
    # via ``repr(sum(()))``) for empty groups, while the polars
    # ``write_csv`` round-trip through ``pl.read_csv`` -> Float64 ->
    # Utf8 cast in the parity test yields ``"0.0"``.  Disk byte-parity
    # is preserved; the captured frame is functionally usable (Phase D
    # consumers parse the Utf8 numerically) but the byte-string-compare
    # parity test cannot validate it.  See _emit_lp_scaling for the
    # corresponding code note.
    "_node_cap_unitsize_sum.csv",
    "_node_cap_raw.csv",
    "_node_cap_pow10.csv",
    "node_capacity_for_scaling.csv",
    "inv_node_cap.csv",
    "_group_cap_pow10.csv",
    "group_capacity_for_scaling.csv",
    "inv_group_cap.csv",
    # _emit_solve_writers — Phase E-b7 (34 small per-solve CSVs)
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
    # _emit_co2_accumulators above; the empty-seed variant from
    # the cumulative emitters overwrites with the same header.
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
    "rp_cost_weight.csv",
    # _emit_period_calc — Phase E-b8 (12 + 2 CSVs)
    "p_inflation_factor_operations_yearly.csv",
    "complete_period_share_of_year_calc.csv",
    "p_timeline_duration_in_years.csv",
    "hours_in_period.csv",
    "period_share_of_year.csv",
    "p_years_d.csv",
    "complete_hours_in_period.csv",
    "p_years_until_invest.csv",
    "p_years_until_dispatch.csv",
    "p_inflation_factor_investment_yearly.csv",
    "f_d_k.csv",
    "pd_branch_weight.csv",
    "pdt_branch_weight.csv",
    # _emit_per_solve — Phase E-b8 (30 + 19 + 1 CSVs)
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
    # _emit_reserve — Phase E-b8 (1 + 2 + 4 CSVs)
    "pdtReserve_upDown_group.csv",
    "process_reserve_upDown_node_active.csv",
    "prundt.csv",
    "p_process_reserve_upDown_node_reliability.csv",
    "process_reserve_upDown_node_increase_reserve_ratio.csv",
    "process_reserve_upDown_node_large_failure_ratio.csv",
    "process_large_failure.csv",
    # flextoolrunner.blocks — Step 2.5 Phase A: the eight per-solve
    # block frames populated by emit_block_data into the Provider.
    "entity_block.csv",
    "process_side_block.csv",
    "process_block.csv",
    "block_step_duration.csv",
    "overlap_set.csv",
    "block_step_previous.csv",
    "block_period_time_first.csv",
    "block_period_time_last.csv",
)


def expected_basenames() -> tuple[str, ...]:
    """Return the basenames the cascade is expected to capture.

    The list comes from the Phase B writer audit (every ``OK_thin_wrapper``
    entry, plus the streamed writers lifted into the canonical pattern by
    Phase E-b).  Tests use this to cross-check disk-vs-Provider parity
    without re-enumerating the list inline.
    """
    return _THIN_WRAPPER_BASENAMES
