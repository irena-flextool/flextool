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
    from flextool.engine_polars._flex_data_provider import FlexDataProvider


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
    provider: "FlexDataProvider",
    logger: logging.Logger | None = None,
    precision_digits: int = 0,
) -> None:
    """Populate *work_folder* with the ``input/`` + ``solve_data/`` CSVs
    the cascade needs, and seed *provider* with the spec-driven frames.

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
    provider : FlexDataProvider
        Required cascade-input Provider.  The SpineDBBackend-driven spec
        loops (Step 2.5 Phases 2-4) ``put`` their materialised frames
        here; downstream cascade readers resolve them via
        :meth:`flextool.engine_polars._flex_data_provider.FlexDataProvider.get`
        rather than re-reading the workdir CSVs.
    logger : logging.Logger, optional
        Logger to use during emission.  ``None`` builds a default named
        logger.
    precision_digits : int, default 0
        Float precision passed through to the underlying writer.

    Notes
    -----
    Δ.20 delegates the actual emission to
    :func:`flextool.flextoolrunner.input_writer.write_input`, which is a
    pure function (no FlexToolRunner state needed).  Step 2.5 threads
    the cascade-input *provider* through so the spec loops can
    progressively migrate from disk-writing to Provider-population.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    work_folder = Path(work_folder)
    work_folder.mkdir(parents=True, exist_ok=True)
    (work_folder / "input").mkdir(exist_ok=True)
    (work_folder / "solve_data").mkdir(exist_ok=True)

    # Step 2.5 item 14 — route directly through input_derivation.run.
    # input_derivation dispatches native polars writers itself; the
    # legacy ``_native_leaf_set_override`` monkey-patch wrapper is no
    # longer needed at this call site.  ``capture_frames`` is entered
    # so each writer's ``_write(df, path)`` populates *provider* under
    # the canonical key without touching disk.
    from flextool.input_derivation import run as _input_derivation_run
    from flextool.engine_polars._flex_data_accumulator import capture_frames

    with capture_frames(provider=provider):
        _input_derivation_run(
            db_url,
            provider,
            logger,
            scenario_name=scenario_name,
            work_folder=work_folder,
            precision_digits=precision_digits,
        )


