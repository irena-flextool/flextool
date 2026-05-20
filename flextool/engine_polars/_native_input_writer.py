"""Cascade-input Provider population from a Spine DB.

The live cascade's pre-solve "write_input" responsibility lives here:
:func:`write_workdir_inputs` reads the Spine database, runs the
:mod:`flextool.input_derivation` pipeline, and populates the
caller-supplied :class:`FlexDataProvider` with every derived frame the
cascade's downstream readers consume.

Pure in-memory
--------------

The function is wrapped in
:func:`flextool.engine_polars._flex_data_accumulator.capture_frames`,
which monkey-patches every participating writer's ``_write(df, path)``
helper to push the frame into the Provider and skip the disk write.
No CSVs land on disk through this path; downstream readers
(:func:`flextool.engine_polars.input.load_flextool`,
:func:`flextool.engine_polars._output_writer.write_outputs_for_solve`,
the per-solve preprocessing dispatched by
:func:`flextool.flextoolrunner.orchestration.run_model`) resolve every
input through the Provider.

For the ``--csv-dump`` debug path the cascade calls
:meth:`FlexDataProvider.snapshot_processed_inputs` separately; that is
the only on-disk emission of the derived frames the cascade produces.

This module also hosts :func:`write_output_support_csvs`, a small
helper used by the surgical fast single-solve path
(:func:`flextool.engine_polars.run_single_solve_from_db`) to seed the
tiny subset of ``solve_data/`` CSVs the output writer adapter needs
when the full preprocessing pipeline is bypassed.
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
    memory_recorder=None,
) -> None:
    """Populate *provider* with every cascade-input frame derived from
    the Spine database.

    Pure in-memory: :func:`capture_frames` is entered so each
    participating writer's ``_write(df, path)`` push the frame into
    *provider* under its canonical key without touching disk.

    Parameters
    ----------
    db_url : str
        Spine SQLite / postgres URL.  Already canonicalised to
        ``sqlite:///<path>`` / ``postgresql://...`` form.
    scenario_name : str | None
        Scenario filter; ``None`` triggers an auto-pick of the first
        scenario in the DB.
    work_folder : Path
        Workdir.  Created if absent.  Forwarded to
        :func:`flextool.input_derivation.run` for the not-yet-Provider-
        only writers' fallback path (becomes optional once every
        writer is Provider-only).
    provider : FlexDataProvider
        Required cascade-input Provider.  Every derivation in the
        :mod:`flextool.input_derivation` pipeline ``put``'s its
        materialised frames here; downstream cascade readers resolve
        them via
        :meth:`flextool.engine_polars._flex_data_provider.FlexDataProvider.get`.
    logger : logging.Logger, optional
        Logger to use during emission.  ``None`` builds a default
        named logger.
    precision_digits : int, default 0
        Float precision forwarded to ``SpineDBBackend.parameter_values``.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    work_folder = Path(work_folder)
    work_folder.mkdir(parents=True, exist_ok=True)

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
            memory_recorder=memory_recorder,
        )


