"""TIER A output writer adapter — Δ.1 dispatch.

Bridges the polars-build :class:`polar_high.Solution` to flextool's
existing post-solve writers in ``flextool.process_outputs``.  These
writers are the same code paths flextool uses today; we feed them the
live ``highspy.Highs`` instance the polars LP just produced + the work
folder so they read the support CSVs (``p_step_duration.csv``,
``process_block.csv``, …) that ``FlexToolRunner.write_input`` /
``preprocessing_solve_time`` are still emitting in the current cascade.

Rationale (from the Δ.1 design):

* ~50-100 LOC of glue beats a 700-LOC re-implementation of the writers.
* Variable-name reconciliation is the only friction point — the polars
  LP carries ``v_invest_p[<entity>,<period>]`` / ``v_invest_n[…]`` /
  ``v_divest_p[…]`` / ``v_divest_n[…]`` whereas flextool's writers expect
  unified ``v_invest[…]`` / ``v_divest[…]`` (option (ii) in the
  dispatch).  We resolve in-place by renaming the live HiGHS column
  names BEFORE delegating — `passColName` accepts a post-solve update,
  verified against highspy 1.x.  The rename is a no-op for variables
  already named correctly (most of them are).
* ``v_ramp`` is absent from the polars LP entirely — flextool's
  extractor handles "no matching columns" gracefully by emitting an
  empty-but-well-shaped parquet (see
  ``read_highs_solution.extract_variable``'s ``if not seen_cols``
  fallback).  No special-casing needed.

The carry-forward dependency is the support CSV cluster
(``p_step_duration.csv``, ``process_block.csv``, ``entity_block.csv``,
…) read by ``read_highs_solution._apply_block_expand`` /
``handoff_writers._load_*``.  Δ.2-Δ.10 retire those one cluster at a
time; Δ.1's job is to consume them via the writers, not to replace.

``periods_already_emitted`` carrier
-----------------------------------
``handoff_writers._bump_period_capacity`` accumulates this set on disk
into ``solve_data/period_capacity.csv``.  Δ.1 removed the in-memory
mirror that previously lived on :class:`SolveHandoff` and put it
where it belongs — :class:`OutputWriterState`, this module's
per-cascade scratch carrier.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polar_high import Solution

    from flextool.engine_polars._solve_handoff import SolveHandoff
    from flextool.engine_polars.input import FlexData

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-cascade writer state
# ---------------------------------------------------------------------------


@dataclass
class OutputWriterState:
    """Cross-solve carrier for the output writer.

    Δ.1 placement: ``periods_already_emitted`` was previously a field on
    :class:`SolveHandoff` (and is still populated there for backward
    compatibility — Δ.2-Δ.10 will retire that mirror).  The canonical
    in-memory location for new consumers is this writer-owned state.

    The set is bumped after each successful adapter call by reading
    back the file ``handoff_writers._bump_period_capacity`` just
    overwrote; that's the same source of truth flextool uses.

    Other writer-level scratch state (e.g. solve_progress.csv append
    pointers if ever ported) belongs here too.
    """

    # Per-period bare-set; periods accumulated by ``_bump_period_capacity``
    # across the cascade.  Empty initially; grows monotonically.
    periods_already_emitted: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Variable-name reconciliation
# ---------------------------------------------------------------------------


# Map from polars-LP variable name (in ``Solution._vars``) to the
# flextool-writer variable name (the prefix the extractor's regex
# expects).  Entries with identical source/target are no-ops; we list
# them only for clarity.
_VAR_RENAME: dict[str, str] = {
    "v_invest_p": "v_invest",
    "v_invest_n": "v_invest",
    "v_divest_p": "v_divest",
    "v_divest_n": "v_divest",
}


def _rename_invest_columns(sol: "Solution") -> None:
    """Rename ``v_invest_p`` / ``v_invest_n`` / ``v_divest_*`` columns
    in-place on the live HiGHS instance to the unified flextool names.

    The polars LP splits each invest/divest decision into two
    non-negative columns (process-side and node-side); flextool's
    writers expect a single ``v_invest[<entity>,<period>]`` /
    ``v_divest[<entity>,<period>]`` family.  Since the entity sets for
    ``_p`` and ``_n`` are disjoint (one is process-only, the other
    node-only, by construction in
    ``flextool/engine_polars/model.py``), the union of the renamed
    columns is itself a well-formed ``v_invest`` family.

    ``passColName`` is invoked per column-id derived from the polars
    Var's ``frame["col_id"]``.  This is O(n_cols_to_rename) — typically
    < 1000 for realistic cases.  No-op if ``sol.highs is None``.
    """
    h = getattr(sol, "highs", None)
    if h is None:
        return  # Solution synthesized outside a real solve — adapter
                # caller is responsible for ensuring this doesn't happen
                # on the cascade path.

    for src_name, dst_name in _VAR_RENAME.items():
        if src_name not in sol._vars:
            continue
        v = sol._vars[src_name]
        if not v.dims:
            continue
        # Build the destination col-name strings — same bracket payload,
        # just a different prefix.  We reuse the existing col_names list
        # we already computed at solve time (sol.col_names) to avoid
        # re-rendering them; the only diff is the prefix.
        ids = v.frame["col_id"].to_numpy().tolist()
        for cid in ids:
            old_name = sol.col_names[cid]
            # old_name is "v_invest_p[entity,period]" — replace the
            # prefix only, keep the bracketed payload verbatim.
            assert old_name is not None and old_name.startswith(src_name + "[")
            new_name = dst_name + old_name[len(src_name):]
            h.passColName(cid, new_name)
            # Mirror onto the Solution's name array so downstream
            # consumers that look at ``sol.col_names`` (e.g. tests) see
            # the unified name too.
            sol.col_names[cid] = new_name


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def write_outputs_for_solve(
    sol: "Solution",
    *,
    work_folder: Path | str,
    solve_name: str,
    prior_handoff: "SolveHandoff | None" = None,
    writer_state: "OutputWriterState | None" = None,
    flex_data: "FlexData | None" = None,
    is_first_solve: bool | None = None,
    scale_the_objective: float | None = None,
) -> None:
    """Adapter — emit TIER A artefacts for one cascade sub-solve.

    Calls flextool's ``process_outputs.read_highs_solution.write_all_variables``
    (~30 variable parquets to ``output_raw/``) and
    ``process_outputs.handoff_writers.write_all_handoffs`` (handoff
    CSVs + 4 capacity CSVs in ``output_raw/``) using the live HiGHS
    instance carried on :class:`Solution`.  The writers consume the
    support CSVs in ``solve_data/`` that flextool's preprocessing has
    already produced — Δ.1's carry-forward dependency.

    ``writer_state`` (optional) accumulates cross-solve scratch
    (``periods_already_emitted``).  When present, the set is updated
    from the freshly-bumped ``solve_data/period_capacity.csv``.

    No-ops gracefully when ``sol.highs is None`` (no live solver
    instance available — typically a synthesized Solution in a unit
    test); callers should pass solutions from a real
    :func:`polar_high.Problem.solve` call.
    """
    h = getattr(sol, "highs", None)
    if h is None:
        # Commercial-solver path — the LiteSolution carries no live HiGHS
        # instance but exposes ``make_shim()`` returning a thin
        # ``highspy.Highs`` API shim built from the SolverResult + LpView.
        # Feed that into ``write_all_variables`` / ``write_all_handoffs``
        # so the same writer code paths work for both the HiGHS path
        # (live ``highspy.Highs``) and the commercial path (LpNamesShim).
        make_shim = getattr(sol, "make_shim", None)
        if make_shim is None:
            _logger.warning(
                "write_outputs_for_solve: Solution carries no live HiGHS "
                "instance (sol.highs is None) AND no make_shim() factory; "
                "skipping output emission for solve '%s'", solve_name,
            )
            return
        h = make_shim()
        commercial_path = True
    else:
        commercial_path = False

    work_folder = Path(work_folder)
    output_dir = work_folder / "output_raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Variable-name shim — flextool's writers see ``v_invest[…]`` /
    # ``v_divest[…]`` after this call.  In-place mutation on the live
    # HiGHS; safe because the Solution is read-only post-solve and the
    # adapter consumes it once.  On the commercial path the LiteSolution
    # already applied the rename at construction time and the shim's
    # ``passColName`` is a no-op — calling here is harmless and lets the
    # code path stay branch-free.
    if not commercial_path:
        _rename_invest_columns(sol)

    # ``scale_the_objective`` — the polars LP now applies the resolved
    # per-solve ``scale_the_objective`` at LP construction (engine_polars/
    # scaling.py auto-apply, commits 19aca81b / 2682cea1 / 4c3b49ca /
    # 8bac7d70).  ``_orchestration._write_scale_csv_and_report`` already
    # writes ``solve_data/scale_the_objective.csv`` with the effective
    # value before this adapter runs; the downstream writers' multiplier
    # (``_resolve_inv_scale_the_objective``) then un-scales objective /
    # dual values back to user-facing units.  An earlier shim here
    # forced the CSV to ``value=1.0`` from the era when the polars LP
    # did NOT scale — that override is now destructive (it canceled
    # the un-scale, leaving ``v_obj`` at the LP-internal magnitude
    # ~1e6× too small).  Removed: trust the upstream CSV.

    # Late imports — keep the adapter's import surface narrow for the
    # 99% of callers that never instantiate it.
    from flextool.process_outputs.read_highs_solution import (
        _actual_solve_name,
        write_all_variables,
    )
    from flextool.process_outputs.handoff_writers import write_all_handoffs

    # Some scenarios use a "complete-solve" name distinct from the
    # roll/solve-current name written into solve_data/ CSVs.  Mirror
    # solver_runner._run_highs_or_cplex's resolution.
    roll_name = _actual_solve_name(work_folder, solve_name)

    sd = work_folder / "solve_data"
    realized_dispatch_csv = sd / "realized_dispatch.csv"
    realized_periods_csv = sd / "realized_invest_periods_of_current_solve.csv"

    try:
        write_all_variables(
            h,
            solve_name=roll_name,
            output_dir=output_dir,
            realized_dispatch_csv=(
                realized_dispatch_csv if realized_dispatch_csv.exists() else None
            ),
            realized_periods_csv=(
                realized_periods_csv if realized_periods_csv.exists() else None
            ),
            # Phase G — route in-memory carriers through to the
            # extractor + custom writers so per-iter file reads
            # (_load_canonical_*, _load_inflation_*, _load_row_scaler,
            # scale_the_objective.csv) can short-circuit.  CSV fallback
            # preserved.
            flex_data=flex_data,
            scale_the_objective=scale_the_objective,
        )
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "write_all_variables failed for solve '%s': %s", solve_name, exc,
        )

    try:
        write_all_handoffs(
            h, solve_name=roll_name, work_folder=work_folder,
            prior_handoff=prior_handoff,
            flex_data=flex_data,
            writer_state=writer_state,
            is_first_solve=is_first_solve,
            scale_the_objective=scale_the_objective,
        )
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "write_all_handoffs failed for solve '%s': %s", solve_name, exc,
        )

    # Phase G — ``writer_state.periods_already_emitted`` is updated
    # in-place by ``handoff_writers._bump_period_capacity`` when the
    # writer_state is threaded through (above).  The previous paranoia
    # re-read of ``solve_data/period_capacity.csv`` was redundant because
    # ``_bump_period_capacity`` is the sole producer and now updates
    # both sinks atomically.  No file re-read here.


__all__ = [
    "OutputWriterState",
    "write_outputs_for_solve",
]
