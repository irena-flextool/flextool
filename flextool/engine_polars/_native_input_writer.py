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


__all__ = ["write_workdir_inputs"]


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

    _flx_write_input(
        db_url,
        scenario_name,
        logger,
        work_folder=work_folder,
        precision_digits=precision_digits,
    )
