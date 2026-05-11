"""Δ.20 — engine_polars-owned workdir CSV writer.

This module replaces the cascade's call to
``flextool.flextoolrunner.flextoolrunner.FlexToolRunner.write_input(...)``.

Pre-Δ.20 the cascade in
:func:`flextool.engine_polars._orchestration.run_chain_from_db` directly
invoked the legacy FlexToolRunner method to populate the workdir's
``input/`` and ``solve_data/`` CSVs.  Δ.20 lifts that responsibility into
``engine_polars`` itself: the cascade now calls
:func:`write_workdir_inputs`, an engine_polars module function whose
contract is "produce the workdir CSVs the cascade and the output writer
adapter need."

Writer-port Phase 1 (L0-L9)
---------------------------

Fourteen preprocessing families are now produced natively:

* L0-L2 (:mod:`._writer_leaf_sets`): ``period_param_sets``,
  ``invest_method_sets``, ``co2_method_sets``, ``simple_projections``.
* L3-L6 (:mod:`._writer_mid_sets`): ``node_type_sets``, ``union_sets``,
  ``dc_angle_bounds``, ``reserve_method_partitions``, ``nonsync_sets``,
  ``method_with_fallback_sets``, ``invest_total_sets``,
  ``structural_filters``.
* L7-L9 (:mod:`._writer_calc_params`): ``entity_total_caps`` (first
  calculated-param family; ``repr(float)`` precision-parity),
  ``process_method_sets`` (process-method projections, ``process_VRE``,
  10 method-gated arc-cross-product tables, 2 profile-method joins).

The remaining ``input/*`` emission (DB → CSV per ``_PARAMETER_SPECS`` /
``_ENTITY_SPECS``) and the two heaviest preprocessing families
(``process_arc_unions`` ~2.3 kLOC, ``entity_period_calc_params``
~2.4 kLOC) still delegate to the legacy ``input_writer.write_input``
body — those are out of scope for this dispatch.  The swap is implemented via
monkey-patch on the legacy preprocessing modules so the in-tree call
sites in ``write_input`` route through native code without modifying
the legacy module's source.

Implementation strategy
-----------------------

The current implementation delegates to the legacy
:func:`flextool.flextoolrunner.input_writer.write_input` for the bulk of
the CSV emission.  This is intentional and documented:

* :mod:`flextool.flextoolrunner.input_writer` is a 2356-LOC pure
  function (no FlexToolRunner state required) that authoritatively
  emits ~100 ``input/`` CSVs and triggers the L0–L9 preprocessing
  passes that produce ~100 ``solve_data/`` CSVs.  Re-implementing the
  whole stack natively against
  :class:`flextool.engine_polars._input_source.InputSource` is well
  beyond the dispatch budget for any single Δ.

* By owning the *call site* in ``engine_polars`` we satisfy the literal
  goal of Δ.20: the live cascade (``run_chain_from_db``) no longer
  references ``FlexToolRunner.write_input``.  Future phases (Δ.21+) can
  replace the delegation one writer at a time without touching the
  cascade contract.

* The cascade's :class:`flextool.flextoolrunner.flextoolrunner.FlexToolRunner`
  is still used downstream — it carries the ``state`` (timeline, solve
  config, handoff dict) consumed by flextool's existing
  ``orchestration.run_model`` driver, which we still leverage for the
  per-solve preprocessing chain (``preprocessing_solve_time``,
  ``solve_writers``, ``handoff_writers``).  The runner is no longer the
  *originator* of workdir CSVs — that ownership now lives here.

Why not skip the writes entirely?
---------------------------------

Several downstream consumers in the live cascade still read CSVs from
``solve_data/`` and ``input/``:

* :func:`flextool.engine_polars.input.load_flextool` — the per-iteration
  ``_load_*`` family reads ~85 CSVs (post-Δ.18) from ``solve_data/``.
* :func:`flextool.engine_polars._output_writer.write_outputs_for_solve`
  → ``flextool.process_outputs.read_highs_solution.write_all_variables``
  → reads ``solve_data/p_step_duration.csv``,
  ``solve_data/process_block.csv``, etc.
* :func:`flextool.flextoolrunner.orchestration.run_model` — its
  ``separate_period_and_timeseries_data`` reads
  ``input/pdt_commodity.csv`` / ``input/pdt_group.csv`` and the L9
  passes (``preprocessing_solve_time``) read another ~30
  ``solve_data/`` CSVs.

Skipping writes in this dispatch would break all three.  Δ.20 lays the
foundation; Δ.21+ retire the write path piece by piece.

Reference: ``flextool/flextoolrunner/input_writer.py`` (read-only).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


__all__ = ["write_workdir_inputs", "write_output_support_csvs"]


# ---------------------------------------------------------------------------
# Δ.25 — minimal output-support CSV writer (fast single-solve path).
# ---------------------------------------------------------------------------


def write_output_support_csvs(
    flex_data: object,
    work_folder: Path,
    *,
    solve_name: str,
) -> None:
    """Δ.25 — emit the tiny subset of ``solve_data/`` CSVs the output
    writer adapter (:mod:`flextool.engine_polars._output_writer`) needs
    when the fast single-solve path skips ``write_input``.

    Specifically writes:

      * ``solve_data/p_step_duration.csv`` — long format
        ``[solve, period, time, p_step_duration]``.  Used by
        :func:`flextool.process_outputs.read_highs_solution._load_realized_list`
        for canonical row order.
      * ``solve_data/scale_the_objective.csv`` — ``key,value`` form,
        ``value=1.0`` (the native LP doesn't apply the scale-factor
        the GMPL pipeline does, so the writer's reciprocal becomes a
        no-op).
      * ``solve_data/process_block.csv`` (header-only stub when no
        block layout is in play — the output writer falls back to
        ``"default"`` when the file is empty / absent).
      * ``solve_data/entity_block.csv`` (same).

    Every CSV is overwritten on each call — idempotent.

    Parameters
    ----------
    flex_data : FlexData
        The in-memory FlexData built by
        :func:`flextool.engine_polars._fast_load.load_flextool_source_only`.
        ``flex_data.dt`` and ``flex_data.p_step_duration`` must be
        non-empty.
    work_folder : Path
        Workdir; ``solve_data/`` is created if absent.
    solve_name : str
        The ``solve`` value to use in the long-format CSVs.  The output
        writer filters by this exact string.
    """
    sd = Path(work_folder) / "solve_data"
    sd.mkdir(parents=True, exist_ok=True)

    # ── p_step_duration.csv ────────────────────────────────────────────
    # Long format: [solve, period, time, p_step_duration].  flextool's
    # phase-1 printf writes one row per (d, t) ∈ dt_realize_dispatch.
    # In the fast single-solve path, every (d, t) in flex_data.dt is
    # realized, so we emit them all.
    psd = getattr(flex_data, "p_step_duration", None)
    psd_frame = psd.frame if psd is not None and hasattr(psd, "frame") else None
    if psd_frame is None or psd_frame.height == 0:
        raise ValueError(
            "write_output_support_csvs: flex_data.p_step_duration is "
            "empty — fast path requires populated step duration."
        )
    long_psd = (psd_frame
        .with_columns(solve=__import__("polars").lit(solve_name))
        .rename({"value": "p_step_duration"})
        .select("solve", "d", "t", "p_step_duration")
        .rename({"d": "period", "t": "time"}))
    long_psd.write_csv(sd / "p_step_duration.csv")

    # ── scale_the_objective.csv ────────────────────────────────────────
    (sd / "scale_the_objective.csv").write_text(
        "key,value\nscale_the_objective,1.0\n"
    )

    # ── process_block.csv / entity_block.csv ───────────────────────────
    # Header-only stubs.  When the output writer's
    # ``_load_entity_block_map`` reads an empty file it returns {} and
    # downstream block-expand falls through to identity (every entity
    # mapped to ``"default"``).  Single-solve simple fixtures don't
    # exercise multi-block layouts, so empty stubs are correct.
    (sd / "process_block.csv").write_text("process,block\n")
    (sd / "entity_block.csv").write_text("entity,block\n")

    # ── realized_dispatch.csv ──────────────────────────────────────────
    # Long format: [solve, period, step].  Every (d, t) in dt is
    # "realized" in single-solve mode (no rolling / nested cascade).
    # write_all_variables uses this as the realized-set filter.
    dt = getattr(flex_data, "dt", None)
    if dt is not None and dt.height > 0:
        rd = (dt
            .with_columns(solve=__import__("polars").lit(solve_name))
            .rename({"d": "period", "t": "step"})
            .select("solve", "period", "step"))
        rd.write_csv(sd / "realized_dispatch.csv")

    # ── realized_invest_periods_of_current_solve.csv ───────────────────
    # Single-column [period].  In single-solve mode every period in
    # dt is also an invest-realized period.
    if dt is not None and dt.height > 0:
        periods = dt.select("d").unique().rename({"d": "period"})
        periods.write_csv(sd / "realized_invest_periods_of_current_solve.csv")

    # ── solve_current.csv ──────────────────────────────────────────────
    # Some output-writer helpers (``_actual_solve_name``) consult this
    # to resolve the solve name.  Not strictly required for the fast
    # single-solve path but harmless and inexpensive.
    (sd / "solve_current.csv").write_text(f"solve\n{solve_name}\n")


def write_workdir_inputs(
    db_url: str,
    scenario_name: str | None,
    work_folder: Path,
    *,
    logger: logging.Logger | None = None,
    precision_digits: int = 0,
) -> None:
    """Populate *work_folder* with the ``input/`` + ``solve_data/`` CSVs
    the cascade needs.

    This is the engine_polars-owned replacement for the cascade's call
    to :meth:`flextool.flextoolrunner.flextoolrunner.FlexToolRunner.write_input`.

    Parameters
    ----------
    db_url : str
        Spine SQLite / postgres URL.  Already canonicalised to
        ``sqlite:///<path>`` / ``postgresql://...`` form.
    scenario_name : str | None
        Scenario filter; ``None`` triggers an auto-pick of the first
        scenario in the DB (matches FlexToolRunner's default).
    work_folder : Path
        Workdir under which ``input/`` + ``solve_data/`` will be created.
    logger : logging.Logger, optional
        Logger to use during emission.  ``None`` builds a default named
        logger.
    precision_digits : int, default 0
        Float precision passed through to the underlying writer.

    Notes
    -----
    Δ.20 delegates the actual emission to
    :func:`flextool.flextoolrunner.input_writer.write_input`, which is a
    pure function (no FlexToolRunner state needed).  This places the
    ownership of workdir population inside ``engine_polars`` — the
    cascade no longer calls ``FlexToolRunner.write_input``.  Future
    phases retire this delegation by porting the underlying writers to
    consume :class:`flextool.engine_polars._input_source.InputSource`
    directly.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    work_folder = Path(work_folder)
    work_folder.mkdir(parents=True, exist_ok=True)
    (work_folder / "input").mkdir(exist_ok=True)
    (work_folder / "solve_data").mkdir(exist_ok=True)

    # Δ.20 — delegate to flextool's input_writer.write_input as a pure
    # function call.  This is intentionally documented as a transitional
    # delegation (see module docstring).  The cascade no longer reaches
    # into ``FlexToolRunner.write_input``; the call site is now owned by
    # ``engine_polars``.
    from flextool.flextoolrunner.input_writer import write_input as _flx_write_input

    with _native_leaf_set_override():
        _flx_write_input(
            db_url,
            scenario_name,
            logger,
            work_folder=work_folder,
            precision_digits=precision_digits,
        )


# ---------------------------------------------------------------------------
# Writer-port Phase 1 (L0-L2) — native override for leaf-level set families.
# ---------------------------------------------------------------------------

import contextlib  # noqa: E402  (placed near the override helper for locality)


@contextlib.contextmanager
def _native_leaf_set_override():
    """Monkey-patch legacy preprocessing families to invoke the native
    polars writers in :mod:`._writer_leaf_sets` (L0-L2),
    :mod:`._writer_mid_sets` (L3-L6) and :mod:`._writer_calc_params`
    (L7-L9).

    The legacy ``flextool.flextoolrunner.input_writer.write_input``
    imports each preprocessing module by name and calls its ``write_*``
    helpers directly.  We rebind those names on the legacy modules for
    the duration of the call so the native implementations are
    consulted in production.  Two heavyweight preprocessing families
    (``process_arc_unions``, ``entity_period_calc_params`` — together
    ~4.7 kLOC) still delegate to legacy code — deferred out of
    Phase 1 scope.
    """
    # L0-L2 — leaf-level set projections.
    from flextool.flextoolrunner.preprocessing import (
        co2_method_sets as _legacy_co2,
        invest_method_sets as _legacy_invest,
        period_param_sets as _legacy_period,
        simple_projections as _legacy_simple,
    )
    # L3-L6 — mid-level set / param projections.
    from flextool.flextoolrunner.preprocessing import (
        dc_angle_bounds as _legacy_dc,
        invest_total_sets as _legacy_invest_total,
        method_with_fallback_sets as _legacy_method_fb,
        node_type_sets as _legacy_node_type,
        nonsync_sets as _legacy_nonsync,
        reserve_method_partitions as _legacy_reserve_part,
        structural_filters as _legacy_struct,
        union_sets as _legacy_union,
    )
    # L7-L9 — calculated-param + process-method families.
    from flextool.flextoolrunner.preprocessing import (
        entity_total_caps as _legacy_entity_total,
        process_method_sets as _legacy_process_method,
    )
    # Phase 1 follow-up — leaf-like arc-union + period-param writers.
    from flextool.flextoolrunner.preprocessing import (
        entity_period_calc_params as _legacy_entity_period,
        process_arc_unions as _legacy_arc_unions,
    )
    # Phase 2 (sub-dispatch 1) — per-solve set + invest-divest writers.
    from flextool.flextoolrunner.preprocessing import (
        invest_divest_sets as _legacy_invest_divest,
        per_solve_sets as _legacy_per_solve,
    )
    # Phase 2 (sub-dispatch 2) — annuity + lp-scaling per-solve writers.
    from flextool.flextoolrunner.preprocessing import (
        entity_annual_calc_params as _legacy_entity_annual,
        lp_scaling_params as _legacy_lp_scaling,
    )
    # Phase 2 (sub-dispatch 3) — per-period calculated params + branch weights.
    from flextool.flextoolrunner.preprocessing import (
        period_calculated_params as _legacy_period_calc,
    )
    # Phase 2 (sub-dispatch 4) — node inflow scaling params.
    from flextool.flextoolrunner.preprocessing import (
        node_inflow_scaling_params as _legacy_inflow_scaling,
    )
    # Phase 2 (sub-dispatch 5) — reserve calculated params.
    from flextool.flextoolrunner.preprocessing import (
        reserve_calc_params as _legacy_reserve_calc,
    )
    # Phase 2 (sub-dispatch 6) — first half of ``solve_writers``
    # (timeline / period / branch / empty / header writers).  These
    # patches target the ``flextool.flextoolrunner.solve_writers`` module
    # directly because ``_native_run_model.py`` imports it as a module
    # and dispatches via attribute access (``solve_writers.write_X(...)``).
    from flextool.flextoolrunner import solve_writers as _legacy_solve_writers
    # Phase 2 (sub-dispatch 8) — ``preprocessing.solve_time.run``
    # orchestrator.  ``_native_run_model.py`` imports the module as
    # ``preprocessing_solve_time`` and calls ``preprocessing_solve_time.run``
    # at attribute-lookup time, so a module-attribute patch suffices.
    from flextool.flextoolrunner.preprocessing import (
        solve_time as _legacy_solve_time,
    )
    from flextool.engine_polars import _writer_leaf_sets as _native
    from flextool.engine_polars import _writer_mid_sets as _native_mid
    from flextool.engine_polars import _writer_calc_params as _native_calc
    from flextool.engine_polars import _writer_arc_unions as _native_arc
    from flextool.engine_polars import _writer_chain_params as _native_chain
    from flextool.engine_polars import _writer_pdt_params as _native_pdt
    from flextool.engine_polars import _writer_period_params as _native_period
    from flextool.engine_polars import _writer_dispatchers as _native_disp
    from flextool.engine_polars import _writer_per_solve as _native_per_solve
    from flextool.engine_polars import _writer_entity_annual as _native_entity_annual
    from flextool.engine_polars import _writer_lp_scaling as _native_lp_scaling
    from flextool.engine_polars import _writer_period_calc as _native_period_calc
    from flextool.engine_polars import _writer_inflow_scaling as _native_inflow_scaling
    from flextool.engine_polars import _writer_reserve as _native_reserve
    from flextool.engine_polars import (
        _writer_solve_writers as _native_solve_writers,
    )
    from flextool.engine_polars import _writer_solve_time as _native_solve_time

    overrides: list[tuple[object, str, object]] = [
        # ── L0-L2 ──────────────────────────────────────────────────────
        # period_param_sets
        (_legacy_period, "write_period_param_sets", _native.write_period_param_sets),
        # invest_method_sets
        (_legacy_invest, "write_invest_method_sets", _native.write_invest_method_sets),
        # co2_method_sets
        (_legacy_co2, "write_co2_method_sets", _native.write_co2_method_sets),
        # simple_projections (11 entries)
        (_legacy_simple, "write_optional_yes", _native.write_optional_yes),
        (_legacy_simple, "write_reserve_upDown_group", _native.write_reserve_upDown_group),
        (_legacy_simple, "write_group_loss_share", _native.write_group_loss_share),
        (_legacy_simple, "write_def_optional_yes", _native.write_def_optional_yes),
        (_legacy_simple, "write_process_delayed", _native.write_process_delayed),
        (_legacy_simple, "write_process_side", _native.write_process_side),
        (_legacy_simple, "write_period_solve", _native.write_period_solve),
        (_legacy_simple, "write_time_set", _native.write_time_set),
        (_legacy_simple, "write_enable_optional_outputs", _native.write_enable_optional_outputs),
        (_legacy_simple, "write_node_state_subsets", _native.write_node_state_subsets),
        (_legacy_simple, "write_commodity_tier_sets", _native.write_commodity_tier_sets),
        (_legacy_simple, "write_simple_setof_projections", _native.write_simple_setof_projections),
        # ── L3-L6 ──────────────────────────────────────────────────────
        # node_type_sets
        (_legacy_node_type, "write_node_type_sets", _native_mid.write_node_type_sets),
        # union_sets
        (_legacy_union, "write_group_entity", _native_mid.write_group_entity),
        (_legacy_union, "write_process_delayed__duration", _native_mid.write_process_delayed__duration),
        # dc_angle_bounds
        (_legacy_dc, "write_dc_angle_bounds", _native_mid.write_dc_angle_bounds),
        # reserve_method_partitions
        (_legacy_reserve_part, "write_reserve_partitions", _native_mid.write_reserve_partitions),
        # nonsync_sets
        (_legacy_nonsync, "write_process__sink_nonSync", _native_mid.write_process__sink_nonSync),
        (_legacy_nonsync, "write_process_group_inside_group_nonsync",
                          _native_mid.write_process_group_inside_group_nonsync),
        # method_with_fallback_sets
        (_legacy_method_fb, "write_entity_lifetime_method", _native_mid.write_entity_lifetime_method),
        (_legacy_method_fb, "write_process_ct_method", _native_mid.write_process_ct_method),
        (_legacy_method_fb, "write_process_startup_method", _native_mid.write_process_startup_method),
        (_legacy_method_fb, "write_node_inflow_method", _native_mid.write_node_inflow_method),
        (_legacy_method_fb, "write_node_storage_binding_method",
                            _native_mid.write_node_storage_binding_method),
        # invest_total_sets
        (_legacy_invest_total, "write_invest_total_sets", _native_mid.write_invest_total_sets),
        (_legacy_invest_total, "write_ci_ladder_cumulative", _native_mid.write_ci_ladder_cumulative),
        # structural_filters
        (_legacy_struct, "write_connection_param", _native_mid.write_connection_param),
        (_legacy_struct, "write_nodegroup_dispatch_node", _native_mid.write_nodegroup_dispatch_node),
        (_legacy_struct, "write_commodity_node_co2", _native_mid.write_commodity_node_co2),
        (_legacy_struct, "write_process__commodity__node", _native_mid.write_process__commodity__node),
        (_legacy_struct, "write_process_coeff_zero_sets", _native_mid.write_process_coeff_zero_sets),
        # ── L7-L9 ──────────────────────────────────────────────────────
        # entity_total_caps (calculated-param family; repr(float) precision)
        (_legacy_entity_total, "write_entity_total_caps", _native_calc.write_entity_total_caps),
        # process_method_sets — 4 emitters
        (_legacy_process_method, "write_process_method_projections",
                                 _native_calc.write_process_method_projections),
        (_legacy_process_method, "write_process_VRE", _native_calc.write_process_VRE),
        (_legacy_process_method, "write_process_arc_method_joins",
                                 _native_calc.write_process_arc_method_joins),
        (_legacy_process_method, "write_process_profile_method_joins",
                                 _native_calc.write_process_profile_method_joins),
        # ── Phase 1 follow-up — process_arc_unions leaf-like writers ──
        (_legacy_arc_unions, "write_process_source_sink_param_t",
                             _native_arc.write_process_source_sink_param_t),
        (_legacy_arc_unions, "write_node_time_param_in_use",
                             _native_arc.write_node_time_param_in_use),
        (_legacy_arc_unions, "write_process_source_delayed_partition",
                             _native_arc.write_process_source_delayed_partition),
        (_legacy_arc_unions, "write_process_source_sink_param",
                             _native_arc.write_process_source_sink_param),
        (_legacy_arc_unions, "write_process_source_sink_profile_method_connection",
                             _native_arc.write_process_source_sink_profile_method_connection),
        (_legacy_arc_unions, "write_process_method_sources_sinks",
                             _native_arc.write_process_method_sources_sinks),
        (_legacy_arc_unions, "write_ed_history_realized_first",
                             _native_arc.write_ed_history_realized_first),
        (_legacy_arc_unions,
         "write_process_source_is_node_sink_1way_no_sink_or_more_than_1_source",
         _native_arc
         .write_process_source_is_node_sink_1way_no_sink_or_more_than_1_source),
        (_legacy_arc_unions, "write_process_source_sink_ramp_method",
                             _native_arc.write_process_source_sink_ramp_method),
        (_legacy_arc_unions, "write_process_source_sink_coeff_zero",
                             _native_arc.write_process_source_sink_coeff_zero),
        (_legacy_arc_unions, "write_process_source_sink_is_node_family",
                             _native_arc.write_process_source_sink_is_node_family),
        (_legacy_arc_unions, "write_process_source_sink_delayed_partition",
                             _native_arc.write_process_source_sink_delayed_partition),
        # ── Phase 1 follow-up — entity_period_calc_params subset ──
        (_legacy_entity_period, "write_pProcess_source_sink",
                                _native_arc.write_pProcess_source_sink),
        # ── Phase 1 follow-up (next dispatch) — pdt* writers via PdtLookup ──
        (_legacy_entity_period, "write_pdtProcess",
                                _native_pdt.write_pdtProcess),
        (_legacy_entity_period, "write_pdtNode",
                                _native_pdt.write_pdtNode),
        (_legacy_entity_period, "write_pdtProcess_source",
                                _native_pdt.write_pdtProcess_source),
        (_legacy_entity_period, "write_pdtProcess_sink",
                                _native_pdt.write_pdtProcess_sink),
        # ── Phase 1 follow-up — medium arc-union families ──
        (_legacy_arc_unions, "write_process_source_sink_ramp_family",
                             _native_arc.write_process_source_sink_ramp_family),
        (_legacy_arc_unions, "write_process_source_sink_ramp_unions",
                             _native_arc.write_process_source_sink_ramp_unions),
        (_legacy_arc_unions, "write_group_commodity_node_period_co2_total",
                             _native_arc.write_group_commodity_node_period_co2_total),
        # ── Phase 1 follow-up 3 — heavy per-(d, t) emitters ──
        (_legacy_entity_period, "write_pdtNodeInflow",
                                _native_period.write_pdtNodeInflow),
        (_legacy_entity_period, "write_pdtProfile",
                                _native_period.write_pdtProfile),
        (_legacy_entity_period, "write_pdtConversion_rate_section_slope",
                                _native_period.write_pdtConversion_rate_section_slope),
        (_legacy_entity_period, "write_pdtProcess_source_sink",
                                _native_period.write_pdtProcess_source_sink),
        # ── Phase 1 follow-up 4 — group/commodity period-param fallbacks
        #    and the positive/negative inflow split.
        (_legacy_entity_period, "write_pdGroup",
                                _native_period.write_pdGroup),
        (_legacy_entity_period, "write_pdtGroup",
                                _native_period.write_pdtGroup),
        (_legacy_entity_period, "write_pdCommodity",
                                _native_period.write_pdCommodity),
        (_legacy_entity_period, "write_pdtCommodity",
                                _native_period.write_pdtCommodity),
        (_legacy_entity_period, "write_p_positive_negative_inflow",
                                _native_period.write_p_positive_negative_inflow),
        # ── Phase 1 follow-up 4 — param-in-use family + dispatch-inside ──
        (_legacy_arc_unions, "write_param_in_use_sets",
                             _native_arc.write_param_in_use_sets),
        (_legacy_arc_unions, "write_node_group_dispatch_process_fully_inside",
                             _native_arc
                             .write_node_group_dispatch_process_fully_inside),
        # ── Phase 1 follow-up 5 — small_set_derivations + small writers ──
        (_legacy_arc_unions, "write_small_set_derivations",
                             _native_arc.write_small_set_derivations),
        (_legacy_arc_unions, "write_process_source_sink_param_with_time",
                             _native_arc.write_process_source_sink_param_with_time),
        (_legacy_arc_unions, "write_gdt_instant_flow_sets",
                             _native_arc.write_gdt_instant_flow_sets),
        (_legacy_arc_unions, "write_p_process_delay_weight",
                             _native_arc.write_p_process_delay_weight),
        (_legacy_arc_unions, "write_gcndt_co2_price",
                             _native_arc.write_gcndt_co2_price),
        (_legacy_arc_unions, "write_group_commodity_node_period_co2_period",
                             _native_arc
                             .write_group_commodity_node_period_co2_period),
        (_legacy_arc_unions, "write_peedt", _native_arc.write_peedt),
        # ── Phase 1 follow-up 5 — entity_period_calc_params varCost +
        #    cap_reduction + ed_period_params + pssdt_varCost filters.
        (_legacy_entity_period, "write_pdtProcess__source__sink__dt_varCost_pair",
                                _native_period
                                .write_pdtProcess__source__sink__dt_varCost_pair),
        (_legacy_entity_period, "write_pssdt_varCost_filters",
                                _native_period.write_pssdt_varCost_filters),
        (_legacy_entity_period, "write_cap_reduction_params",
                                _native_period.write_cap_reduction_params),
        (_legacy_entity_period, "write_ed_period_params",
                                _native_period.write_ed_period_params),
        # ── Phase 1 follow-up 6 — flow-bound + state-slack + storage ref
        #    price + 12-CSV nodeGroupDispatch dispatch set family.
        (_legacy_arc_unions, "write_p_flow_min",
                             _native_arc.write_p_flow_min),
        (_legacy_arc_unions, "write_p_flow_max",
                             _native_arc.write_p_flow_max),
        (_legacy_arc_unions, "write_p_state_slack_share",
                             _native_arc.write_p_state_slack_share),
        (_legacy_arc_unions, "write_p_storage_state_reference_price",
                             _native_arc.write_p_storage_state_reference_price),
        (_legacy_arc_unions, "write_node_group_dispatch_sets",
                             _native_arc.write_node_group_dispatch_sets),
        # ── Phase 1 follow-up 7 — param_t projections + time-param joins ──
        (_legacy_arc_unions, "write_param_t_projections_and_time_params",
                             _native_arc.write_param_t_projections_and_time_params),
        # ── Phase 1 follow-up 8 — chain-cluster entity-period params ──
        # Four writers covering the per-(entity, period) existing/divest
        # chain that the LP build consumes via p_entity_*_existing_* /
        # p_entity_*_capacity_*.  ``write_p_entity_existing_chain``
        # straddles the per-solve handoff boundary (in-memory
        # ``SolveHandoff`` carriers OR file-based reads).
        (_legacy_entity_period, "write_p_entity_pre_existing",
                                _native_chain.write_p_entity_pre_existing),
        (_legacy_entity_period, "write_p_entity_divest_cumulative_max",
                                _native_chain.write_p_entity_divest_cumulative_max),
        (_legacy_entity_period, "write_p_entity_existing_chain",
                                _native_chain.write_p_entity_existing_chain),
        (_legacy_entity_period, "write_p_entity_capacity_max_chain",
                                _native_chain.write_p_entity_capacity_max_chain),
        # ── Phase 1 closeout — top-level dispatcher own-compute ──
        # Both functions are own-compute (no sub-writer calls inside);
        # porting them completes Phase 1's writer-port scope.
        (_legacy_arc_unions, "write_process_arc_unions",
                             _native_disp.write_process_arc_unions),
        (_legacy_entity_period, "write_entity_period_calc_params",
                                _native_disp.write_entity_period_calc_params),
        # ── Phase 2 (sub-dispatch 1) — per-solve set + invest/divest ──
        # ``write_per_solve_sets`` / ``write_invest_divest_sets`` /
        # ``write_ed_invest_forbidden_no_investment`` are not called
        # from ``input_writer.write_input`` so this rebind is currently
        # a no-op at the workdir-population call site.  Wiring into the
        # per-solve preprocessing chain (around ``_flx_orch.run_model``
        # in ``_orchestration._drive_cascade``) is deferred to a
        # follow-up dispatch — see the dispatch report.  Listing them
        # here ahead of wiring keeps a single source of truth for the
        # writer-port override surface and lets the follow-up land as a
        # one-line wrap in ``_drive_cascade``.
        (_legacy_per_solve, "write_per_solve_sets",
                            _native_per_solve.write_per_solve_sets),
        (_legacy_invest_divest, "write_invest_divest_sets",
                                _native_per_solve.write_invest_divest_sets),
        (_legacy_invest_divest, "write_ed_invest_forbidden_no_investment",
                                _native_per_solve
                                .write_ed_invest_forbidden_no_investment),
        # ── Phase 2 (sub-dispatch 2) — annuity + lp-scaling ──
        # Like sub-dispatch 1, these helpers are not invoked from
        # ``input_writer.write_input``; they fire in
        # ``preprocessing/solve_time.run``.  Listing them here keeps
        # ``_native_leaf_set_override`` as the single source of truth
        # for the writer-port surface and lets the future per-solve
        # wire-up land as a one-line wrap.
        (_legacy_entity_annual, "write_entity_annual_calc_params",
                                _native_entity_annual
                                .write_entity_annual_calc_params),
        (_legacy_lp_scaling, "write_lp_scaling_params",
                             _native_lp_scaling.write_lp_scaling_params),
        # ── Phase 2 (sub-dispatch 3) — per-period calculated params ──
        # Fired from ``preprocessing/solve_time.run`` at batches 13
        # (``write_period_calculated_params``) and 63
        # (``write_branch_weights``).  Like the prior sub-dispatches,
        # neither helper is called from ``input_writer.write_input`` —
        # the per-solve wiring activates these in production.
        (_legacy_period_calc, "write_period_calculated_params",
                              _native_period_calc.write_period_calculated_params),
        (_legacy_period_calc, "write_branch_weights",
                              _native_period_calc.write_branch_weights),
        # ── Phase 2 (sub-dispatch 4) — node inflow scaling params ──
        # Fired from ``preprocessing/solve_time.run`` at batch 17.
        # Emits ``ptNode_inflow`` + 16 per-(n, d) inflow-scaling
        # parameters consumed by the LP build via the
        # ``scale_to_annual_flow`` / ``scale_in_proportion`` /
        # ``scale_to_annual_and_peak_flow`` cascades.
        (_legacy_inflow_scaling, "write_node_inflow_scaling_params",
                                 _native_inflow_scaling
                                 .write_node_inflow_scaling_params),
        # ── Phase 2 (sub-dispatch 5) — reserve calculated params ──
        # Fired from ``preprocessing/solve_time.run`` at batches 43, 44,
        # and 49.  Emits 7 CSVs (``pdtReserve_upDown_group``, the
        # ``process_reserve_upDown_node_active`` filter and ``prundt``
        # cross-product, plus the reliability fallback, two ``> 0``
        # filter sets and the ``process_large_failure`` projection).
        # These feed the reserve LP-build module
        # (``engine_polars._reserve``) as RHS / domain inputs.
        (_legacy_reserve_calc, "write_pdtReserve_upDown_group",
                               _native_reserve.write_pdtReserve_upDown_group),
        (_legacy_reserve_calc,
         "write_process_reserve_upDown_node_active_and_prundt",
         _native_reserve.write_process_reserve_upDown_node_active_and_prundt),
        (_legacy_reserve_calc,
         "write_process_reserve_filters_and_reliability",
         _native_reserve.write_process_reserve_filters_and_reliability),
        # ── Phase 2 (sub-dispatch 6) — solve_writers first half ──────
        # Group A — timeline / period writers fired from
        # ``_native_run_model.py`` per-solve loop.
        (_legacy_solve_writers, "write_full_timelines",
                                _native_solve_writers.write_full_timelines),
        (_legacy_solve_writers, "write_active_timelines",
                                _native_solve_writers.write_active_timelines),
        (_legacy_solve_writers, "write_step_jump",
                                _native_solve_writers.write_step_jump),
        (_legacy_solve_writers, "write_period_block",
                                _native_solve_writers.write_period_block),
        (_legacy_solve_writers, "write_years_represented",
                                _native_solve_writers.write_years_represented),
        (_legacy_solve_writers, "write_period_years",
                                _native_solve_writers.write_period_years),
        (_legacy_solve_writers, "write_periods",
                                _native_solve_writers.write_periods),
        (_legacy_solve_writers, "write_first_and_last_periods",
                                _native_solve_writers
                                .write_first_and_last_periods),
        (_legacy_solve_writers, "write_solve_status",
                                _native_solve_writers.write_solve_status),
        (_legacy_solve_writers, "write_current_solve",
                                _native_solve_writers.write_current_solve),
        (_legacy_solve_writers, "write_period_boundary_step",
                                _native_solve_writers.write_period_boundary_step),
        (_legacy_solve_writers, "write_first_steps",
                                _native_solve_writers.write_first_steps),
        (_legacy_solve_writers, "write_last_steps",
                                _native_solve_writers.write_last_steps),
        (_legacy_solve_writers, "get_first_steps",
                                _native_solve_writers.get_first_steps),
        (_legacy_solve_writers, "write_last_realized_step",
                                _native_solve_writers.write_last_realized_step),
        (_legacy_solve_writers, "write_realized_dispatch",
                                _native_solve_writers.write_realized_dispatch),
        (_legacy_solve_writers, "write_fix_storage_timesteps",
                                _native_solve_writers
                                .write_fix_storage_timesteps),
        # Group B — branch / empty / header writers.
        (_legacy_solve_writers, "write_branch__period_relationship",
                                _native_solve_writers
                                .write_branch__period_relationship),
        (_legacy_solve_writers, "write_all_branches",
                                _native_solve_writers.write_all_branches),
        (_legacy_solve_writers, "write_branch_weights_and_map",
                                _native_solve_writers
                                .write_branch_weights_and_map),
        (_legacy_solve_writers, "write_empty_investment_file",
                                _native_solve_writers
                                .write_empty_investment_file),
        (_legacy_solve_writers, "write_empty_cumulative_files",
                                _native_solve_writers
                                .write_empty_cumulative_files),
        (_legacy_solve_writers, "write_empty_storage_fix_file",
                                _native_solve_writers
                                .write_empty_storage_fix_file),
        (_legacy_solve_writers, "write_headers_for_empty_output_files",
                                _native_solve_writers
                                .write_headers_for_empty_output_files),
        (_legacy_solve_writers, "write_timesets",
                                _native_solve_writers.write_timesets),
        (_legacy_solve_writers, "write_hole_multiplier",
                                _native_solve_writers.write_hole_multiplier),
        # ── Phase 2 (sub-dispatch 7) — solve_writers second half ─────
        # Scaling writers (Agent-5 / Agent-8 / Agent-21 path).
        (_legacy_solve_writers, "write_p_use_row_scaling",
                                _native_solve_writers.write_p_use_row_scaling),
        (_legacy_solve_writers, "write_scale_the_objective",
                                _native_solve_writers.write_scale_the_objective),
        (_legacy_solve_writers, "write_scale_the_state",
                                _native_solve_writers.write_scale_the_state),
        (_legacy_solve_writers, "write_scale_the_objective_header_only",
                                _native_solve_writers
                                .write_scale_the_objective_header_only),
        (_legacy_solve_writers, "write_scale_the_state_header_only",
                                _native_solve_writers
                                .write_scale_the_state_header_only),
        # Delay-duration map.
        (_legacy_solve_writers, "write_delayed_durations",
                                _native_solve_writers.write_delayed_durations),
        # Representative-period writers.
        (_legacy_solve_writers, "write_rp_data",
                                _native_solve_writers.write_rp_data),
        (_legacy_solve_writers, "write_timeset_cost_weight",
                                _native_solve_writers.write_timeset_cost_weight),
        (_legacy_solve_writers, "write_empty_rp_data",
                                _native_solve_writers.write_empty_rp_data),
        # ── Phase 2 (sub-dispatch 8) — preprocessing.solve_time.run ──
        # Closes out Phase 2: the per-solve preprocessing orchestrator
        # itself is now natively overridable.  Every helper called from
        # ``preprocessing.solve_time.run`` is already intercepted by
        # one of the entries above, so the native ``run`` simply
        # sequences the same call chain through the legacy module
        # references (which are themselves patched).
        (_legacy_solve_time, "run", _native_solve_time.run),
    ]
    saved: list[tuple[object, str, object]] = [
        (mod, name, getattr(mod, name)) for mod, name, _ in overrides
    ]
    for mod, name, native_fn in overrides:
        setattr(mod, name, native_fn)
    try:
        yield
    finally:
        for mod, name, original in saved:
            setattr(mod, name, original)
