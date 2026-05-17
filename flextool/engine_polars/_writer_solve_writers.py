"""Writer-port Phase 2 (sub-dispatch 6 + 7) — ``solve_writers.py`` port.

Native re-implementation of the ~36 functions in
``flextool.flextoolrunner.solve_writers`` (legacy 948 LOC total).
Sub-dispatch 6 ported the first 26 helpers (~542 LOC); sub-dispatch 7
extends this module with the remaining nine writers (~406 LOC):
the scaling-flag writer (``write_p_use_row_scaling``), the four
``scale_the_objective`` / ``scale_the_state`` keyed-value writers
(``write_scale_the_*`` + their header-only variants), the
``write_delayed_durations`` chain emitter, and the three
representative-period writers (``write_rp_data``,
``write_timeset_cost_weight``, ``write_empty_rp_data``).  Two
functional groups originally:

Group A — timeline / period writers
-----------------------------------

* :func:`write_full_timelines` — ``steps_in_timeline.csv``
* :func:`write_active_timelines` — ``steps_in_use.csv`` /
  ``steps_complete_solve.csv`` (the ``complete=True`` variant emits the
  ``complete_step_duration`` header)
* :func:`write_step_jump` — ``step_previous.csv``
* :func:`write_period_block` — ``period_block_time.csv`` +
  ``period_block_succ.csv``
* :func:`write_years_represented` — ``p_years_represented.csv``
* :func:`write_period_years` — ``period_with_history.csv`` /
  ``p_discount_years.csv``
* :func:`write_periods` — ``realized_invest_periods_of_current_solve.csv``
  / ``invest_periods_of_current_solve.csv``
* :func:`write_first_and_last_periods` — ``period_last.csv``,
  ``period_first_of_solve.csv``, ``period_first.csv``
* :func:`write_solve_status` — ``p_model.csv`` / ``p_nested_model.csv``
* :func:`write_current_solve` — ``solve_current.csv``
* :func:`write_period_boundary_step` (+ :func:`write_first_steps` /
  :func:`write_last_steps` shims) — ``first_timesteps.csv`` /
  ``last_timesteps.csv``
* :func:`get_first_steps` — pure helper (returns dict, no CSV)
* :func:`write_last_realized_step` — ``last_realized_timestep.csv``
* :func:`write_realized_dispatch` — ``realized_dispatch.csv``
* :func:`write_fix_storage_timesteps` — ``fix_storage_timesteps.csv``

Group B — branch / empty / header writers
------------------------------------------

* :func:`write_branch__period_relationship` — ``period__branch.csv``
* :func:`write_all_branches` — ``branch_all.csv`` +
  ``time_branch_all.csv``
* :func:`write_branch_weights_and_map` — ``solve_branch_weight.csv``
  + ``solve_branch__time_branch.csv``
* :func:`write_empty_investment_file` — ``p_entity_invested.csv`` /
  ``p_entity_divested.csv`` /
  ``p_entity_period_existing_capacity.csv``
* :func:`write_empty_cumulative_files` — three rolling-accumulator
  seed files
* :func:`write_empty_storage_fix_file` — four fix-storage seed files
* :func:`write_headers_for_empty_output_files` — generic
  comma-split header writer
* :func:`write_timesets` — ``input/timesets_in_use.csv`` +
  ``input/timesets__timeline.csv``
* :func:`write_hole_multiplier` — ``solve_hole_multiplier.csv``

Implementation
--------------

These are pure CSV writers — no preprocessing logic, no polars
operations, no data derivation.  Phase E-b7 converted every
``write_*`` into the canonical ``derive_X -> _write(derive_X(...),
path)`` pattern: a :func:`derive_X` companion returns the full
``Utf8`` polars frame and :func:`_write` emits it via
``df.write_csv(path, line_terminator="\r\n")`` — the CRLF terminator
matches the legacy ``csv.writer`` + ``newline=""`` output byte-for-
byte.  This lets the Phase E-b accumulator
(:mod:`._flex_data_accumulator`) capture every CSV through the
shared ``_write`` monkey-patch.

Byte-identical parity is still verified via
``filecmp.cmp(shallow=False)`` in
``tests/engine_polars/test_writer_port_phase1.py``.

Function signatures match the legacy module verbatim because
:mod:`._native_input_writer._native_leaf_set_override` monkey-patches
the legacy module's attributes by name; ``_native_run_model.py``
imports the legacy module once and dispatches via attribute access.

Sub-dispatch 7 group — scaling / delay / representative period
----------------------------------------------------------------

* :func:`write_p_use_row_scaling` — ``p_use_row_scaling.csv`` (Agent-5
  row-scaling opt-in flag, with the
  ``FLEXTOOL_FORCE_ROW_SCALING`` env-var test hook preserved verbatim)
* :func:`write_scale_the_objective` /
  :func:`write_scale_the_objective_header_only` —
  ``solve_data/scale_the_objective.csv`` (auto-scale value-or-header
  variants; ``%.17g`` precision)
* :func:`write_scale_the_state` /
  :func:`write_scale_the_state_header_only` —
  ``solve_data/scale_the_state.csv`` (companion to the objective
  scalar, currently fixed at ``1.0``)
* :func:`write_delayed_durations` — ``solve_data/delay_duration.csv``
  + ``solve_data/dtt__delay_duration.csv`` (source/sink offset map
  with wrap-around at end-of-period)
* :func:`write_rp_data` — eight representative-period CSVs
  (``rp_weights.csv``, ``rp_base_chain.csv``,
  ``rp_base_first.csv`` / ``rp_base_last.csv``,
  ``rp_block_first.csv`` / ``rp_block_last.csv``,
  ``rp_block_start_last.csv``, ``rp_cost_weight.csv``)
* :func:`write_timeset_cost_weight` — ``rp_cost_weight.csv`` from
  per-timestep ``timeset_weights`` (non-RP normalised pathway)
* :func:`write_empty_rp_data` — header-only seeds for the eight
  representative-period CSVs (non-RP models)

The scaling CSVs use ``%.17g`` repr formatting (so that
``writerow([..., f"{float(v):.17g}"])`` produces a value that
round-trips through GMPL's ``table data IN``).  ``write_rp_data``
preserves the legacy ordering and the ``weight > 1e-10`` epsilon
filter.  ``write_timeset_cost_weight`` uses ``%.10g`` for the
normalised float (matches legacy).
"""
from __future__ import annotations

import csv
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl


# ---------------------------------------------------------------------------
# Canonical writer-port emitter — Phase E-b7 accumulator funnel.
#
# All writers in this module funnel their derived frames through
# :func:`_write` so :mod:`._flex_data_accumulator` can capture them via
# its monkey-patch.  Byte parity with the legacy ``csv.writer`` emission
# is preserved by using ``line_terminator="\r\n"`` (the same CRLF
# terminator the ``newline=""`` open + ``csv.writer`` combination
# produces on every platform).
#
# The companion :func:`_to_utf8_frame` builds an all-``Utf8`` polars
# frame from a header tuple and a row list.  Every value is taken
# verbatim if already a string, otherwise stringified via ``str(v)`` —
# this matches ``csv.writer``'s behaviour exactly for the types this
# module emits (``int`` 0/1 flags, ``float`` weights, plain strings).
# Schema is locked to ``Utf8`` per column so polars' write_csv does not
# attempt numeric formatting / quoting.
# ---------------------------------------------------------------------------


def _write(df: pl.DataFrame, path: Path) -> None:
    """Emit *df* as a CRLF-terminated CSV at *path*.

    Identical I/O contract to the ``_write(df, path)`` helper in
    :mod:`._writer_pdt_params`.  Phase E-b7 accumulator
    (:mod:`._flex_data_accumulator`) monkey-patches this name so every
    CSV emission also stashes ``(path.name → df)`` for the sub-solve.

    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(Path(path), line_terminator="\r\n")


def _to_utf8_frame(
    headers: tuple[str, ...],
    rows: list[tuple[Any, ...]],
) -> pl.DataFrame:
    """Build an all-``Utf8`` polars frame from a header tuple + row list.

    Each row cell is taken as-is when already a string, otherwise
    rendered via ``str(v)`` — matching ``csv.writer``'s exact byte
    output for ``int`` / ``float`` cells.
    """
    cols: dict[str, list[str]] = {h: [] for h in headers}
    for row in rows:
        for h, v in zip(headers, row):
            cols[h].append(v if isinstance(v, str) else str(v))
    return pl.DataFrame(cols, schema={h: pl.Utf8 for h in headers})


# ---------------------------------------------------------------------------
# Group A — timeline writers
# ---------------------------------------------------------------------------


def derive_full_timelines(
    stochastic_timesteps: list[tuple[str, str]],
    period__timesets_in_this_solve: list[tuple[str, str]],
    timesets__timeline: dict[str, str],
    timelines: dict[str, list[tuple[str, ...]]],
) -> pl.DataFrame:
    """Build the canonical ``steps_in_timeline`` frame (period, step)."""
    rows: list[tuple[Any, ...]] = []
    for period__timeset in period__timesets_in_this_solve:
        for timeline in timelines:
            for timeset_in_timeline, tt in timesets__timeline.items():
                if period__timeset[1] == timeset_in_timeline:
                    if timeline == tt:
                        for item in timelines[timeline]:
                            rows.append((period__timeset[0], item[0]))
    for step in stochastic_timesteps:
        rows.append((step[0], step[1]))
    return _to_utf8_frame(("period", "step"), rows)


def write_full_timelines(
    stochastic_timesteps: list[tuple[str, str]],
    period__timesets_in_this_solve: list[tuple[str, str]],
    timesets__timeline: dict[str, str],
    timelines: dict[str, list[tuple[str, ...]]],
    filename: str,
) -> None:
    """Emit ``steps_in_timeline.csv`` — every timestep of the bundled
    timelines (in the order they appear in the period→timeset →timeline
    chain), followed by any extra stochastic timesteps for this solve.
    """
    _write(
        derive_full_timelines(
            stochastic_timesteps,
            period__timesets_in_this_solve,
            timesets__timeline,
            timelines,
        ),
        Path(filename),
    )


def derive_active_timelines(
    timeline: dict[str, list[tuple[str, ...]]],
    complete: bool = False,
) -> pl.DataFrame:
    """Build the canonical ``steps_in_use`` / ``steps_complete_solve``
    frame (period, step, [complete_]step_duration).
    """
    header_dur = "complete_step_duration" if complete else "step_duration"
    rows: list[tuple[Any, ...]] = []
    for period_name, period in timeline.items():
        for item in period:
            rows.append((period_name, item.timestep, str(item.duration)))
    return _to_utf8_frame(("period", "step", header_dur), rows)


def write_active_timelines(
    timeline: dict[str, list[tuple[str, ...]]],
    filename: str,
    complete: bool = False,
) -> None:
    """Emit ``steps_in_use.csv`` / ``steps_complete_solve.csv`` — the
    active timestep + step-duration triples for this (or the complete)
    solve.  ``complete=True`` swaps the duration column header to
    ``complete_step_duration``.
    """
    _write(derive_active_timelines(timeline, complete), Path(filename))


_STEP_JUMP_HEADERS = (
    "period", "time", "previous", "previous_within_timeset",
    "previous_period", "previous_within_solve", "jump",
)


def derive_step_jump(step_lengths: list[tuple[str, ...]]) -> pl.DataFrame:
    """Build the canonical ``step_previous`` frame (7 Utf8 columns)."""
    return _to_utf8_frame(_STEP_JUMP_HEADERS, list(step_lengths))


def write_step_jump(
    step_lengths: list[tuple[str, ...]],
    work_folder: Path | None = None,
) -> None:
    """Emit ``solve_data/step_previous.csv``."""
    wf = work_folder if work_folder is not None else Path.cwd()
    _write(derive_step_jump(step_lengths), wf / "solve_data/step_previous.csv")


def derive_period_block_time(
    period_block_time: list[tuple],
) -> pl.DataFrame:
    """Build the ``period_block_time`` frame (period, block_first, step)."""
    return _to_utf8_frame(
        ("period", "block_first", "step"), list(period_block_time),
    )


def derive_period_block_succ(
    period_block_succ: list[tuple],
) -> pl.DataFrame:
    """Build the ``period_block_succ`` frame
    (period, block_first, block_first_next).
    """
    return _to_utf8_frame(
        ("period", "block_first", "block_first_next"),
        list(period_block_succ),
    )


def write_period_block(
    period_block_time: list[tuple],
    period_block_succ: list[tuple],
    work_folder: Path | None = None,
) -> None:
    """Emit the two block-structure CSVs (``period_block_time.csv`` +
    ``period_block_succ.csv``) used by ``bind_intraperiod_blocks``.
    """
    wf = work_folder if work_folder is not None else Path.cwd()
    sd = wf / "solve_data"
    _write(derive_period_block_time(period_block_time),
           sd / "period_block_time.csv")
    _write(derive_period_block_succ(period_block_succ),
           sd / "period_block_succ.csv")


# ---------------------------------------------------------------------------
# Period writers
# ---------------------------------------------------------------------------


def derive_years_represented(
    period__branch: list[tuple[str, str]],
    years_represented: list[tuple[str, str]],
) -> pl.DataFrame:
    """Build the canonical ``p_years_represented`` frame
    (period, years_from_solve, p_years_from_solve, p_years_represented).
    """
    rows: list[tuple[Any, ...]] = []
    year_count: float = 0
    for period__years in years_represented:
        total_represented = float(period__years[1])
        if total_represented <= 0:
            continue
        n_rows = math.ceil(total_represented)
        remaining = total_represented
        for _ in range(n_rows):
            years_to_cover_within_year = min(1.0, remaining)
            rows.append((
                period__years[0], str(year_count), str(year_count),
                str(years_to_cover_within_year),
            ))
            for pd_pair in period__branch:
                if pd_pair[0] in period__years[0] and pd_pair[0] != pd_pair[1]:
                    rows.append((
                        pd_pair[1], str(year_count), str(year_count),
                        str(years_to_cover_within_year),
                    ))
            year_count += years_to_cover_within_year
            remaining -= years_to_cover_within_year
    return _to_utf8_frame(
        ("period", "years_from_solve",
         "p_years_from_solve", "p_years_represented"),
        rows,
    )


def write_years_represented(
    period__branch: list[tuple[str, str]],
    years_represented: list[tuple[str, str]],
    filename: str,
) -> None:
    """Emit ``p_years_represented.csv`` — each period's
    ``years_represented`` R is expanded into ``ceil(R)`` width-1 rows
    plus a trailing fractional remainder row (sub-year periods emit a
    single row of width R; R <= 0 → skip).
    """
    _write(
        derive_years_represented(period__branch, years_represented),
        Path(filename),
    )


def derive_period_years(
    stochastic_branches: list[tuple[str, str]],
    years_represented: list[tuple[str, str]],
) -> pl.DataFrame:
    """Build the canonical ``period_with_history`` / ``p_discount_years``
    frame (period, param).
    """
    rows: list[tuple[Any, ...]] = []
    year_count: float = 0
    for period__year in years_represented:
        rows.append((period__year[0], str(year_count)))
        for pd_pair in stochastic_branches:
            if pd_pair[0] in period__year[0] and pd_pair[0] != pd_pair[1]:
                rows.append((pd_pair[1], str(year_count)))
        year_count += float(period__year[1])
    return _to_utf8_frame(("period", "param"), rows)


def write_period_years(
    stochastic_branches: list[tuple[str, str]],
    years_represented: list[tuple[str, str]],
    filename: str,
) -> None:
    """Emit ``period_with_history.csv`` / ``p_discount_years.csv`` —
    each period mapped to its cumulative ``year_count`` start.
    """
    _write(
        derive_period_years(stochastic_branches, years_represented),
        Path(filename),
    )


def derive_periods(
    solve: str,
    periods_dict: dict[str, list[tuple[str, str]]],
) -> pl.DataFrame:
    """Build a single-column ``period`` frame for the given solve key."""
    rows: list[tuple[Any, ...]] = [
        (period_tuple[1],) for period_tuple in periods_dict.get(solve, [])
    ]
    return _to_utf8_frame(("period",), rows)


def write_periods(
    solve: str,
    periods_dict: dict[str, list[tuple[str, str]]],
    filename: str,
) -> None:
    """Emit a single-column ``period`` CSV for the given solve key."""
    _write(derive_periods(solve, periods_dict), Path(filename))


def _compute_first_and_last_periods(
    active_time_list: dict[str, list[tuple[str, ...]]],
    period__timesets_in_this_solve: list[tuple[str, str]],
    period__branch_list: list[tuple[str, str]],
) -> tuple[list[str], list[str], list[str]]:
    """Shared compute for the three first/last period CSVs.

    Returns ``(period_last, period_first_of_solve_list, period_first_list)``
    each as a flat list of period names.
    """
    period_first_of_solve = list(active_time_list.keys())[0]
    period_last = [list(active_time_list.keys())[-1]]
    time_step_last = active_time_list[period_last[0]][-1].timestep
    for period in active_time_list.keys():
        if (
            active_time_list[period][-1].timestep == time_step_last
            and period != period_last[0]
        ):
            period_last.append(period)

    period_first_of_solve_list = [
        pb[1] for pb in period__branch_list if pb[0] == period_first_of_solve
    ]
    period_first = period__timesets_in_this_solve[0][0]
    period_first_list = [
        pb[1] for pb in period__branch_list if pb[0] == period_first
    ]
    return period_last, period_first_of_solve_list, period_first_list


def derive_period_last(
    active_time_list: dict[str, list[tuple[str, ...]]],
    period__timesets_in_this_solve: list[tuple[str, str]],
    period__branch_list: list[tuple[str, str]],
) -> pl.DataFrame:
    """Build the ``period_last`` frame (period)."""
    period_last, _, _ = _compute_first_and_last_periods(
        active_time_list, period__timesets_in_this_solve, period__branch_list,
    )
    return _to_utf8_frame(("period",), [(p,) for p in period_last])


def derive_period_first_of_solve(
    active_time_list: dict[str, list[tuple[str, ...]]],
    period__timesets_in_this_solve: list[tuple[str, str]],
    period__branch_list: list[tuple[str, str]],
) -> pl.DataFrame:
    """Build the ``period_first_of_solve`` frame (period)."""
    _, period_first_of_solve_list, _ = _compute_first_and_last_periods(
        active_time_list, period__timesets_in_this_solve, period__branch_list,
    )
    return _to_utf8_frame(
        ("period",), [(p,) for p in period_first_of_solve_list],
    )


def derive_period_first(
    active_time_list: dict[str, list[tuple[str, ...]]],
    period__timesets_in_this_solve: list[tuple[str, str]],
    period__branch_list: list[tuple[str, str]],
) -> pl.DataFrame:
    """Build the ``period_first`` frame (period)."""
    _, _, period_first_list = _compute_first_and_last_periods(
        active_time_list, period__timesets_in_this_solve, period__branch_list,
    )
    return _to_utf8_frame(("period",), [(p,) for p in period_first_list])


def write_first_and_last_periods(
    active_time_list: dict[str, list[tuple[str, ...]]],
    period__timesets_in_this_solve: list[tuple[str, str]],
    period__branch_list: list[tuple[str, str]],
    work_folder: Path | None = None,
) -> None:
    """Emit ``period_last.csv``, ``period_first_of_solve.csv`` and
    ``period_first.csv``.  Assumes ``active_time_list`` is in
    chronological period order; multi-branch tail is included by
    matching the final period's last timestep.
    """
    wf = work_folder if work_folder is not None else Path.cwd()
    sd = wf / "solve_data"
    period_last, period_first_of_solve_list, period_first_list = (
        _compute_first_and_last_periods(
            active_time_list,
            period__timesets_in_this_solve,
            period__branch_list,
        )
    )
    _write(
        _to_utf8_frame(("period",), [(p,) for p in period_last]),
        sd / "period_last.csv",
    )
    _write(
        _to_utf8_frame(
            ("period",), [(p,) for p in period_first_of_solve_list],
        ),
        sd / "period_first_of_solve.csv",
    )
    _write(
        _to_utf8_frame(("period",), [(p,) for p in period_first_list]),
        sd / "period_first.csv",
    )


# ---------------------------------------------------------------------------
# Solve status
# ---------------------------------------------------------------------------


def derive_solve_status(
    first_state: bool,
    last_state: bool,
    nested: bool = False,
) -> pl.DataFrame:
    """Build the ``p_model`` / ``p_nested_model`` frame
    (modelParam, p_model | p_nested_model).
    """
    param_col = "p_nested_model" if nested else "p_model"
    rows = [
        ("solveFirst", "1" if first_state else "0"),
        ("solveLast", "1" if last_state else "0"),
    ]
    return _to_utf8_frame(("modelParam", param_col), rows)


def write_solve_status(
    first_state: bool,
    last_state: bool,
    nested: bool = False,
    work_folder: Path | None = None,
) -> None:
    """Emit ``p_model.csv`` (or ``p_nested_model.csv`` when
    ``nested=True``) with ``solveFirst`` / ``solveLast`` integer flags.
    """
    wf = work_folder if work_folder is not None else Path.cwd()
    if not nested:
        path = wf / "solve_data/p_model.csv"
    else:
        path = wf / "solve_data/p_nested_model.csv"
    _write(derive_solve_status(first_state, last_state, nested), path)


def derive_current_solve(solve: str) -> pl.DataFrame:
    """Build the ``solve_current`` frame (solve)."""
    return _to_utf8_frame(("solve",), [(solve,)])


def write_current_solve(solve: str, filename: str) -> None:
    """Emit ``solve_current.csv`` — single-row solve name file."""
    _write(derive_current_solve(solve), Path(filename))


# ---------------------------------------------------------------------------
# Timestep boundary  (S04 consolidation in legacy)
# ---------------------------------------------------------------------------


def derive_period_boundary_step(
    timeline: dict[str, list[tuple[str, ...]]],
    last: bool = False,
) -> pl.DataFrame:
    """Build the (period, step) boundary-step frame — first or last
    timestep of each period.
    """
    rows: list[tuple[Any, ...]] = []
    for period_name, period in timeline.items():
        boundary = period[-1:] if last else period[:1]
        for item in boundary:
            rows.append((period_name, item.timestep))
    return _to_utf8_frame(("period", "step"), rows)


def write_period_boundary_step(
    timeline: dict[str, list[tuple[str, ...]]],
    filename: str,
    *,
    last: bool = False,
) -> None:
    """Emit one (period, step) row per period — the first or last
    timestep depending on ``last``.
    """
    _write(derive_period_boundary_step(timeline, last=last), Path(filename))


def write_first_steps(
    timeline: dict[str, list[tuple[str, ...]]],
    filename: str,
) -> None:
    """Thin wrapper — emit ``first_timesteps.csv``."""
    write_period_boundary_step(timeline, filename, last=False)


def write_last_steps(
    timeline: dict[str, list[tuple[str, ...]]],
    filename: str,
) -> None:
    """Thin wrapper — emit ``last_timesteps.csv``."""
    write_period_boundary_step(timeline, filename, last=True)


def get_first_steps(
    steplists: dict[str, list[Any]],
) -> dict[str, tuple[Any, ...]]:
    """Helper: pull the first step of the current solve (and the first
    step of the next solve in execution order) from each solve's
    steplist.  Returns ``{solve: (own_first[, next_first])}``.

    Pure data transform — no CSV side effects.
    """
    solve_names = list(steplists.keys())
    starts: dict[str, tuple[Any, ...]] = dict()
    for index, name in enumerate(solve_names):
        if index == (len(solve_names) - 1):
            starts[name] = (steplists[name][0],)
        else:
            starts[name] = (
                steplists[solve_names[index]][0],
                steplists[solve_names[index + 1]][0],
            )
    return starts


# ---------------------------------------------------------------------------
# Realized / fix storage
# ---------------------------------------------------------------------------


def derive_last_realized_step(
    realized_timeline: dict[str, list[tuple[str, ...]]],
    solve: str,
    realized_periods: list[tuple[str, str]],
) -> pl.DataFrame:
    """Build the ``last_realized_timestep`` frame (period, step)."""
    rows: list[tuple[Any, ...]] = []
    has_realized_period = False
    last_realized_period: tuple[str, list[tuple[str, ...]]] | None = None
    for period_name, period in realized_timeline.items():
        if any(t[1] == period_name for t in realized_periods):
            last_realized_period = (period_name, period)
            has_realized_period = True
    if has_realized_period and last_realized_period is not None:
        for item in last_realized_period[1][-1:]:
            rows.append((last_realized_period[0], item.timestep))
    return _to_utf8_frame(("period", "step"), rows)


def write_last_realized_step(
    realized_timeline: dict[str, list[tuple[str, ...]]],
    solve: str,
    realized_periods: list[tuple[str, str]],
    filename: str,
) -> None:
    """Emit ``last_realized_timestep.csv`` — single row pointing at the
    last step of the last period that appears in ``realized_periods``
    (empty file when there is none).
    """
    _write(
        derive_last_realized_step(realized_timeline, solve, realized_periods),
        Path(filename),
    )


def derive_realized_dispatch(
    realized_time_list: dict[str, list[tuple[str, ...]]],
    solve: str,
    realized_periods: list[tuple[str, str]],
) -> pl.DataFrame:
    """Build the ``realized_dispatch`` frame (period, step)."""
    rows: list[tuple[Any, ...]] = []
    for period, realized_time in realized_time_list.items():
        if any(t[1] == period for t in realized_periods):
            for i in realized_time:
                rows.append((period, i.timestep))
    return _to_utf8_frame(("period", "step"), rows)


def write_realized_dispatch(
    realized_time_list: dict[str, list[tuple[str, ...]]],
    solve: str,
    realized_periods: list[tuple[str, str]],
    work_folder: Path | None = None,
) -> None:
    """Emit ``realized_dispatch.csv`` — every step of every period whose
    name appears in ``realized_periods``.
    """
    wf = work_folder if work_folder is not None else Path.cwd()
    _write(
        derive_realized_dispatch(realized_time_list, solve, realized_periods),
        wf / "solve_data/realized_dispatch.csv",
    )


def derive_fix_storage_timesteps(
    active_time_list: dict[str, list[tuple[str, ...]]],
    solve: str,
    fix_storage_periods: list[tuple[str, str]],
) -> pl.DataFrame:
    """Build the ``fix_storage_timesteps`` frame (period, step)."""
    rows: list[tuple[Any, ...]] = []
    for period, active_time in active_time_list.items():
        if any(t[1] == period for t in fix_storage_periods):
            for i in active_time:
                rows.append((period, i.timestep))
    return _to_utf8_frame(("period", "step"), rows)


def write_fix_storage_timesteps(
    active_time_list: dict[str, list[tuple[str, ...]]],
    solve: str,
    fix_storage_periods: list[tuple[str, str]],
    work_folder: Path | None = None,
) -> None:
    """Emit ``fix_storage_timesteps.csv`` — every step of every period
    that appears in ``fix_storage_periods``.
    """
    wf = work_folder if work_folder is not None else Path.cwd()
    _write(
        derive_fix_storage_timesteps(
            active_time_list, solve, fix_storage_periods,
        ),
        wf / "solve_data/fix_storage_timesteps.csv",
    )


# ---------------------------------------------------------------------------
# Group B — branch / empty / header writers
# ---------------------------------------------------------------------------


def derive_branch__period_relationship(
    period__branch: list[tuple[str, str]],
) -> pl.DataFrame:
    """Build the ``period__branch`` frame (period, branch)."""
    return _to_utf8_frame(
        ("period", "branch"),
        [(row[0], row[1]) for row in period__branch],
    )


def write_branch__period_relationship(
    period__branch: list[tuple[str, str]],
    filename: str,
) -> None:
    """Emit ``period__branch.csv`` (2-col verbatim copy)."""
    _write(
        derive_branch__period_relationship(period__branch), Path(filename),
    )


def derive_branch_all(
    period__branch_list: dict[str, list[tuple[str, str]]],
) -> pl.DataFrame:
    """Build the ``branch_all`` frame — union of all branches across
    solves (single column ``branch``).
    """
    branches: list[str] = []
    for solve in period__branch_list:
        for row in period__branch_list[solve]:
            if row[1] not in branches:
                branches.append(row[1])
    return _to_utf8_frame(("branch",), [(b,) for b in branches])


_TIME_BRANCH_TIMESERIES_NAMES: tuple[str, ...] = (
    "pbt_node_inflow.csv",
    "pbt_node.csv",
    "pbt_process.csv",
    "pbt_profile.csv",
    "pbt_process_source.csv",
    "pbt_process_sink.csv",
    "pbt_reserve__upDown__group.csv",
)


def derive_time_branch_all(
    solve_branch__time_branch_list: list[tuple[str, str]],
    logger: logging.Logger,
    work_folder: Path,
) -> pl.DataFrame:
    """Build the ``time_branch_all`` frame from the seven ``pbt_*`` input
    CSVs (under ``<work>/input/``) plus any extra time-branches from
    ``solve_branch__time_branch_list``.

    Aborts with ``sys.exit(-1)`` if any ``pbt_*`` row has an empty
    branch name (matches legacy fatal behaviour).
    """
    time_branches: list[str] = []
    for filename in _TIME_BRANCH_TIMESERIES_NAMES:
        with open(
            work_folder / "input" / filename, "r", encoding="utf-8",
        ) as blk:
            filereader = csv.reader(blk, delimiter=",")
            next(filereader)  # header
            while True:
                try:
                    datain = next(filereader)
                    if datain[1] not in time_branches:
                        time_branches.append(datain[1])
                    if datain[1] == "":
                        logger.error(
                            "Empty branch name in timeseries: " + filename
                            + " , check that there is no empty row at the"
                              " end of the array"
                        )
                        sys.exit(-1)
                except StopIteration:
                    break

    for solve__branch in solve_branch__time_branch_list:
        if solve__branch[1] not in time_branches:
            time_branches.append(solve__branch[1])
    return _to_utf8_frame(("time_branch",), [(tb,) for tb in time_branches])


def write_all_branches(
    period__branch_list: dict[str, list[tuple[str, str]]],
    solve_branch__time_branch_list: list[tuple[str, str]],
    logger: logging.Logger,
    work_folder: Path | None = None,
) -> None:
    """Emit ``branch_all.csv`` (union of all branches across solves)
    and ``time_branch_all.csv`` (union of branches from the seven
    ``pbt_*`` input CSVs plus any extra time-branches from
    ``solve_branch__time_branch_list``).

    Aborts with ``sys.exit(-1)`` if any ``pbt_*`` row has an empty
    branch name (matches legacy fatal behaviour).
    """
    wf = work_folder if work_folder is not None else Path.cwd()
    _write(
        derive_branch_all(period__branch_list),
        wf / "solve_data/branch_all.csv",
    )
    _write(
        derive_time_branch_all(
            solve_branch__time_branch_list, logger, wf,
        ),
        wf / "solve_data/time_branch_all.csv",
    )


def derive_solve_branch_weight(
    complete_solve: str,
    active_time_list: dict[str, list[tuple[str, ...]]],
    solve_branch__time_branch_list: list[tuple[str, str]],
    branch_start_time: tuple[str, str] | None,
    period__branch_lists: list[tuple[str, str]],
    stochastic_branches: dict[str, list[Any]],
) -> pl.DataFrame:
    """Build the ``solve_branch_weight`` frame
    (branch, p_branch_weight_input).
    """
    time_branch_weight: dict[str, Any] = defaultdict()
    if branch_start_time is not None:
        for row in stochastic_branches[complete_solve]:
            if (
                branch_start_time[0] == row[0]
                and branch_start_time[1] == row[2]
            ):
                time_branch_weight[row[1]] = row[4]

    rows: list[tuple[Any, ...]] = []
    for solve_branch__time_branch in solve_branch__time_branch_list:
        if (
            (solve_branch__time_branch[0], solve_branch__time_branch[0])
            in period__branch_lists
        ):
            rows.append((solve_branch__time_branch[0], "1.0"))
        elif (
            solve_branch__time_branch[1] in time_branch_weight.keys()
            and solve_branch__time_branch[0] in active_time_list.keys()
        ):
            rows.append((
                solve_branch__time_branch[0],
                str(time_branch_weight[solve_branch__time_branch[1]]),
            ))
    return _to_utf8_frame(("branch", "p_branch_weight_input"), rows)


def derive_solve_branch__time_branch(
    solve_branch__time_branch_list: list[tuple[str, str]],
) -> pl.DataFrame:
    """Build the ``solve_branch__time_branch`` frame (period, branch)."""
    return _to_utf8_frame(
        ("period", "branch"),
        [(s[0], s[1]) for s in solve_branch__time_branch_list],
    )


def write_branch_weights_and_map(
    complete_solve: str,
    active_time_list: dict[str, list[tuple[str, ...]]],
    solve_branch__time_branch_list: list[tuple[str, str]],
    branch_start_time: tuple[str, str] | None,
    period__branch_lists: list[tuple[str, str]],
    stochastic_branches: dict[str, list[Any]],
    work_folder: Path | None = None,
) -> None:
    """Emit ``solve_branch_weight.csv`` (per-branch weight, defaulting
    to ``1.0`` for self-pairings) and ``solve_branch__time_branch.csv``
    (the relationship verbatim).
    """
    wf = work_folder if work_folder is not None else Path.cwd()
    _write(
        derive_solve_branch_weight(
            complete_solve, active_time_list, solve_branch__time_branch_list,
            branch_start_time, period__branch_lists, stochastic_branches,
        ),
        wf / "solve_data/solve_branch_weight.csv",
    )
    _write(
        derive_solve_branch__time_branch(solve_branch__time_branch_list),
        wf / "solve_data/solve_branch__time_branch.csv",
    )


# ---------------------------------------------------------------------------
# Init / empty files
# ---------------------------------------------------------------------------


def _empty_frame(headers: tuple[str, ...]) -> pl.DataFrame:
    """Build a header-only all-Utf8 polars frame."""
    return pl.DataFrame(schema={h: pl.Utf8 for h in headers})


def write_empty_investment_file(work_folder: Path | None = None) -> None:
    """Seed three header-only investment CSVs for the first solve."""
    wf = work_folder if work_folder is not None else Path.cwd()
    sd = wf / "solve_data"
    _write(
        _empty_frame(("entity", "p_entity_invested")),
        sd / "p_entity_invested.csv",
    )
    _write(
        _empty_frame(("entity", "p_entity_divested")),
        sd / "p_entity_divested.csv",
    )
    _write(
        _empty_frame((
            "entity", "period",
            "p_entity_period_existing_capacity",
            "p_entity_period_invested_capacity",
        )),
        sd / "p_entity_period_existing_capacity.csv",
    )


def write_empty_cumulative_files(work_folder: Path | None = None) -> None:
    """Seed three header-only rolling-accumulator CSVs (mod default 0
    collapses every accumulator on a single-period single solve).
    """
    wf = work_folder if work_folder is not None else Path.cwd()
    sd = wf / "solve_data"
    sd.mkdir(exist_ok=True)
    _write(
        _empty_frame(
            ("commodity", "tier", "period", "p_ladder_cum_realized_mwh"),
        ),
        sd / "ladder_cum_realized_mwh.csv",
    )
    _write(
        _empty_frame(("period", "p_ladder_cum_sim_hours")),
        sd / "ladder_cum_sim_hours.csv",
    )
    _write(
        _empty_frame(("group", "period", "p_co2_cum_realized_tonnes")),
        sd / "co2_cum_realized_tonnes.csv",
    )


def write_empty_storage_fix_file(work_folder: Path | None = None) -> None:
    """Seed four header-only fix-storage CSVs for the first solve.

    NOTE: legacy preserves the leading-space column names (e.g.
    ``" period"``) — kept verbatim for byte parity.
    """
    wf = work_folder if work_folder is not None else Path.cwd()
    sd = wf / "solve_data"
    _write(
        _empty_frame(
            ("node", " period", " step", " ndt_fix_storage_price"),
        ),
        sd / "fix_storage_price.csv",
    )
    _write(
        _empty_frame(
            ("node", " period", " step", " ndt_fix_storage_quantity"),
        ),
        sd / "fix_storage_quantity.csv",
    )
    _write(
        _empty_frame(
            ("node", " period", " step", " ndt_fix_storage_usage"),
        ),
        sd / "fix_storage_usage.csv",
    )
    _write(
        _empty_frame(("node", " p_roll_continue_state")),
        sd / "p_roll_continue_state.csv",
    )


def write_headers_for_empty_output_files(filename: str, header: str) -> None:
    """Emit a header-only output CSV given comma-separated column names."""
    _write(_empty_frame(tuple(header.split(","))), Path(filename))


# ---------------------------------------------------------------------------
# Misc writers
# ---------------------------------------------------------------------------


def derive_timesets_in_use(
    timesets_used_by_solves: dict[str, list[tuple[str, str]]],
) -> pl.DataFrame:
    """Build the ``timesets_in_use`` frame (solve, period, timesets)."""
    rows: list[tuple[Any, ...]] = []
    for solve, period_timeset_list in timesets_used_by_solves.items():
        for period, timeset in period_timeset_list:
            rows.append((solve, period, timeset))
    return _to_utf8_frame(("solve", "period", "timesets"), rows)


def derive_timesets__timeline(
    timeset__timeline: dict[str, str],
) -> pl.DataFrame:
    """Build the ``timesets__timeline`` frame (timesets, timeline)."""
    rows = [(ts, tl) for ts, tl in timeset__timeline.items()]
    return _to_utf8_frame(("timesets", "timeline"), rows)


def write_timesets(
    timesets_used_by_solves: dict[str, list[tuple[str, str]]],
    timeset__timeline: dict[str, str],
    work_folder: Path | None = None,
) -> None:
    """Emit ``input/timesets_in_use.csv`` (solve × period × timeset
    long-form) and ``input/timesets__timeline.csv`` (timeset → timeline
    mapping).
    """
    wf = work_folder if work_folder is not None else Path.cwd()
    _write(
        derive_timesets_in_use(timesets_used_by_solves),
        wf / "input/timesets_in_use.csv",
    )
    _write(
        derive_timesets__timeline(timeset__timeline),
        wf / "input/timesets__timeline.csv",
    )


def derive_hole_multiplier(
    solve: str,
    hole_multipliers: dict[str, str],
) -> pl.DataFrame:
    """Build the ``solve_hole_multiplier`` frame
    (solve, p_hole_multiplier).  Empty body when multiplier is falsy.
    """
    rows: list[tuple[Any, ...]] = []
    if hole_multipliers[solve]:
        rows.append((solve, hole_multipliers[solve]))
    return _to_utf8_frame(("solve", "p_hole_multiplier"), rows)


def write_hole_multiplier(
    solve: str,
    hole_multipliers: dict[str, str],
    filename: str,
) -> None:
    """Emit ``solve_hole_multiplier.csv`` — single row when the
    multiplier is truthy, header-only otherwise.
    """
    _write(derive_hole_multiplier(solve, hole_multipliers), Path(filename))


# ---------------------------------------------------------------------------
# Sub-dispatch 7 — scaling writers
# ---------------------------------------------------------------------------


def derive_p_use_row_scaling(
    solve: str,
    use_row_scaling: dict[str, str],
) -> pl.DataFrame:
    """Build the ``p_use_row_scaling`` frame (solve, p_use_row_scaling).

    Honours the ``FLEXTOOL_FORCE_ROW_SCALING`` env-var test hook (Agent 9
    Mode B un-scaling benchmark harness).
    """
    import os as _os

    value_str = (
        use_row_scaling.get(solve, "no")
        if isinstance(use_row_scaling, dict)
        else "no"
    )
    flag = 1 if str(value_str).strip().lower() == "yes" else 0
    if _os.environ.get("FLEXTOOL_FORCE_ROW_SCALING", "").strip().lower() in (
        "1", "yes", "true", "on",
    ):
        flag = 1
    return _to_utf8_frame(
        ("solve", "p_use_row_scaling"), [(solve, str(flag))],
    )


def write_p_use_row_scaling(
    solve: str,
    use_row_scaling: dict[str, str],
    filename: str,
) -> None:
    """Emit ``p_use_row_scaling.csv`` — the Agent-5 row-scaling opt-in
    flag as an integer 0/1.  Default is ``0`` (off) unless the user
    explicitly sets the parameter to ``"yes"`` on the solve entity.
    The row is always written so AMPL finds the current solve's value.

    The ``FLEXTOOL_FORCE_ROW_SCALING`` env-var (``1`` / ``yes`` /
    ``true`` / ``on``) forces ``flag=1`` regardless — Agent 9 test
    hook for the Mode B un-scaling benchmark harness.  No effect in
    production unless the env var is set.
    """
    _write(
        derive_p_use_row_scaling(solve, use_row_scaling), Path(filename),
    )


def derive_scale_the_objective(value: float) -> pl.DataFrame:
    """Build the ``scale_the_objective`` keyed-value frame
    (key=``v``, value=``%.17g``).
    """
    return _to_utf8_frame(
        ("key", "value"), [("v", f"{float(value):.17g}")],
    )


def derive_scale_the_state(value: float) -> pl.DataFrame:
    """Build the ``scale_the_state`` keyed-value frame — companion to
    :func:`derive_scale_the_objective`.
    """
    return _to_utf8_frame(
        ("key", "value"), [("v", f"{float(value):.17g}")],
    )


def derive_scale_the_objective_header_only() -> pl.DataFrame:
    """Build the header-only ``scale_the_objective`` frame."""
    return _empty_frame(("key", "value"))


def derive_scale_the_state_header_only() -> pl.DataFrame:
    """Build the header-only ``scale_the_state`` frame."""
    return _empty_frame(("key", "value"))


def write_scale_the_objective(
    solve_data_dir: Path | str,
    value: float,
) -> Path:
    """Emit ``solve_data/scale_the_objective.csv`` — Agent-8 scaling
    analyser's global objective scalar in a single keyed-row CSV.

    GMPL ``table data IN`` requires a keyed read, so the row is stored
    as ``("v", <scalar>)`` and pulled via
    ``sum{k in _scale_obj_keys} _scale_obj_from_csv[k]`` on the
    single-row file.  Value is formatted with ``%.17g`` for full
    double precision round-trip.

    See :func:`write_scale_the_objective_header_only` for the
    default-mode header-only variant.
    """
    sd = Path(solve_data_dir)
    path = sd / "scale_the_objective.csv"
    _write(derive_scale_the_objective(value), path)
    return path


def write_scale_the_state(
    solve_data_dir: Path | str,
    value: float,
) -> Path:
    """Emit ``solve_data/scale_the_state.csv`` — companion to
    :func:`write_scale_the_objective`.  Currently fixed at ``1.0`` in
    the analyser; the field is reserved for future tuning.  Layout
    matches the objective CSV verbatim.
    """
    sd = Path(solve_data_dir)
    path = sd / "scale_the_state.csv"
    _write(derive_scale_the_state(value), path)
    return path


def write_scale_the_objective_header_only(
    solve_data_dir: Path | str,
) -> Path:
    """Emit header-only ``solve_data/scale_the_objective.csv`` —
    default-mode counterpart to :func:`write_scale_the_objective`.
    The file exists (so ``table data IN`` does not fail) but has no
    data rows; ``_scale_obj_keys`` stays empty and the
    ``default 1e-6`` clause on ``param scale_the_objective`` applies.

    Agent 21 rationale: the analyser's power-of-10 rounding is too
    aggressive for models whose Matrix range is already wide; users
    must opt in to auto-apply explicitly.
    """
    sd = Path(solve_data_dir)
    path = sd / "scale_the_objective.csv"
    _write(derive_scale_the_objective_header_only(), path)
    return path


def write_scale_the_state_header_only(solve_data_dir: Path | str) -> Path:
    """Emit header-only ``solve_data/scale_the_state.csv`` —
    default-mode counterpart to :func:`write_scale_the_state`.  Same
    rationale as :func:`write_scale_the_objective_header_only`: the
    CSV exists but has no data rows, so the ``default 1`` clause on
    ``param scale_the_state`` applies.
    """
    sd = Path(solve_data_dir)
    path = sd / "scale_the_state.csv"
    _write(derive_scale_the_state_header_only(), path)
    return path


# ---------------------------------------------------------------------------
# Sub-dispatch 7 — delay durations
# ---------------------------------------------------------------------------


def _compute_delay_duration_set(
    delay_durations: dict[str, Any],
) -> set[str]:
    """Shared compute for the two delayed-duration CSVs."""
    delay_duration_set: set[str] = set()
    for entity, dur in delay_durations.items():
        if isinstance(dur, list):
            for delay_duration in dur:
                delay_duration_set.add(str(delay_duration[0]))
        else:
            delay_duration_set.add(str(dur))
    return delay_duration_set


def derive_delay_duration(
    delay_durations: dict[str, Any],
) -> pl.DataFrame:
    """Build the ``delay_duration`` frame (delay_duration)."""
    rows = [(str(d),) for d in _compute_delay_duration_set(delay_durations)]
    return _to_utf8_frame(("delay_duration",), rows)


def derive_dtt__delay_duration(
    active_time_list: dict[str, list[tuple[str, ...]]],
    delay_durations: dict[str, Any],
) -> pl.DataFrame:
    """Build the ``dtt__delay_duration`` frame
    (period, time_source, time_sink, delay_duration) with cyclic
    end-of-period wrap-around.
    """
    delay_duration_set = _compute_delay_duration_set(delay_durations)
    rows: list[tuple[Any, ...]] = []
    for period_name, time_steps in active_time_list.items():
        for k, time_step in enumerate(time_steps):
            for delay_duration in delay_duration_set:
                offset = int(float(delay_duration))
                if k + offset < len(time_steps):
                    sink_step = time_steps[k + offset].timestep
                else:
                    sink_step = time_steps[
                        k - len(time_steps) + offset
                    ].timestep
                rows.append((
                    period_name, time_step.timestep,
                    sink_step, str(delay_duration),
                ))
    return _to_utf8_frame(
        ("period", "time_source", "time_sink", "delay_duration"), rows,
    )


def write_delayed_durations(
    active_time_list: dict[str, list[tuple[str, ...]]],
    solve: str,
    delay_durations: dict[str, Any],
    work_folder: Path | None = None,
) -> None:
    """Emit ``solve_data/delay_duration.csv`` (unique delay values used
    by any delayed process) and ``solve_data/dtt__delay_duration.csv``
    (source/sink offset map across every active timestep, with
    end-of-period wrap-around to keep the relationship cyclic).

    ``delay_durations`` may map an entity to either a scalar duration
    or a list of ``(duration, ...)`` tuples — both shapes are
    flattened into the unique duration set.
    """
    wf = work_folder if work_folder is not None else Path.cwd()
    _write(
        derive_delay_duration(delay_durations),
        wf / "solve_data/delay_duration.csv",
    )
    _write(
        derive_dtt__delay_duration(active_time_list, delay_durations),
        wf / "solve_data/dtt__delay_duration.csv",
    )


# ---------------------------------------------------------------------------
# Sub-dispatch 7 — representative period writers
# ---------------------------------------------------------------------------


def _compute_rp_frames(
    rp_weights: dict[str, dict[str, float]],
    timeset_duration_entries: list[tuple[str, float]],
    period_name: str,
) -> dict[str, pl.DataFrame]:
    """Shared compute for the eight representative-period CSVs.

    Returns a dict keyed by output basename — used by both the
    ``derive_*`` accessors and the wrapper writer.
    """
    # RP block boundaries from the timeset_duration entries.
    rp_starts: list[str] = []
    rp_lasts: list[str] = []
    for start_step, count in timeset_duration_entries:
        start_step = str(start_step)
        rp_starts.append(start_step)
        start_idx = int(start_step[1:])  # 't0001' -> 1
        last_idx = start_idx + int(float(count)) - 1
        rp_lasts.append(f"t{last_idx:04d}")

    base_starts = sorted(rp_weights.keys(), key=lambda s: int(s[1:]))
    n_base = len(base_starts)
    n_rp = len(rp_starts)

    out: dict[str, pl.DataFrame] = {}

    # 1. rp_weights.csv — drop near-zero entries.
    rp_weights_rows: list[tuple[Any, ...]] = []
    for base in base_starts:
        for rep, weight in rp_weights[base].items():
            if weight > 1e-10:
                rp_weights_rows.append((base, rep, str(weight)))
    out["rp_weights.csv"] = _to_utf8_frame(
        ("base_start", "rep_start", "weight"), rp_weights_rows,
    )

    # 2. rp_base_chain.csv — predecessor chain (excludes first).
    out["rp_base_chain.csv"] = _to_utf8_frame(
        ("base_start", "prev_base_start"),
        [(base_starts[i], base_starts[i - 1]) for i in range(1, n_base)],
    )

    # 3. rp_base_first.csv / rp_base_last.csv
    out["rp_base_first.csv"] = _to_utf8_frame(
        ("base_start",), [(base_starts[0],)] if n_base > 0 else [],
    )
    out["rp_base_last.csv"] = _to_utf8_frame(
        ("base_start",), [(base_starts[-1],)] if n_base > 0 else [],
    )

    # 4. rp_block_first.csv / rp_block_last.csv
    out["rp_block_first.csv"] = _to_utf8_frame(
        ("period", "step"), [(period_name, s) for s in rp_starts],
    )
    out["rp_block_last.csv"] = _to_utf8_frame(
        ("period", "step"), [(period_name, lst) for lst in rp_lasts],
    )

    # 5. rp_block_start_last.csv
    out["rp_block_start_last.csv"] = _to_utf8_frame(
        ("rep_start", "last_step"),
        [(s, lst) for s, lst in zip(rp_starts, rp_lasts)],
    )

    # 6. rp_cost_weight.csv — normalised per-timestep weight.
    w_r: dict[str, float] = {r: 0.0 for r in rp_starts}
    for base_weights in rp_weights.values():
        for rep, weight in base_weights.items():
            if rep in w_r:
                w_r[rep] += weight
    for rep in w_r:
        w_r[rep] = w_r[rep] * n_rp / n_base if n_base > 0 else 1.0
    cost_rows: list[tuple[Any, ...]] = []
    for start, last in zip(rp_starts, rp_lasts):
        start_idx = int(start[1:])
        last_idx = int(last[1:])
        weight = w_r[start]
        for t_idx in range(start_idx, last_idx + 1):
            cost_rows.append((period_name, f"t{t_idx:04d}", str(weight)))
    out["rp_cost_weight.csv"] = _to_utf8_frame(
        ("period", "time", "weight"), cost_rows,
    )

    return out


def derive_rp_weights(
    rp_weights: dict[str, dict[str, float]],
    timeset_duration_entries: list[tuple[str, float]],
    period_name: str,
) -> pl.DataFrame:
    """Build the ``rp_weights`` frame (base_start, rep_start, weight)."""
    return _compute_rp_frames(
        rp_weights, timeset_duration_entries, period_name,
    )["rp_weights.csv"]


def derive_rp_base_chain(
    rp_weights: dict[str, dict[str, float]],
    timeset_duration_entries: list[tuple[str, float]],
    period_name: str,
) -> pl.DataFrame:
    """Build the ``rp_base_chain`` frame
    (base_start, prev_base_start).
    """
    return _compute_rp_frames(
        rp_weights, timeset_duration_entries, period_name,
    )["rp_base_chain.csv"]


def write_rp_data(
    rp_weights: dict[str, dict[str, float]],
    timeset_duration_entries: list[tuple[str, float]],
    period_name: str,
    work_folder: Path | None = None,
) -> None:
    """Emit the eight representative-period CSVs for the GMPL solver.

    Args:
        rp_weights: ``{base_start: {rep_start: weight}}`` — the full
            weight matrix between base periods and representative
            periods.
        timeset_duration_entries: ``[(start_step, count), ...]`` for
            the RP timeset; ``count`` is interpreted as a float (the
            timeset CSV stores it as a stringified number).
        period_name: FlexTool period name used as the ``period``
            column on the per-step CSVs (e.g. ``"p2025"``).
        work_folder: working directory containing ``solve_data/``.

    Files written (all under ``solve_data/``):

    * ``rp_weights.csv``         — (base, rep, weight) triples, with
      a ``weight > 1e-10`` epsilon filter to drop numerical zeroes.
    * ``rp_base_chain.csv``      — chronological predecessor chain
      (excludes the very first base period).
    * ``rp_base_first.csv``      — single-row first base period.
    * ``rp_base_last.csv``       — single-row last base period.
    * ``rp_block_first.csv`` / ``rp_block_last.csv`` — first/last
      timestep of each RP block.
    * ``rp_block_start_last.csv`` — start → last step mapping.
    * ``rp_cost_weight.csv``     — per-timestep cost weight
      ``w_r = (sum_d W[d,r]) * n_rp / n_base`` (so a uniform weight
      input produces ``w_r = 1`` per step).
    """
    wf = work_folder if work_folder is not None else Path.cwd()
    sd = wf / "solve_data"
    frames = _compute_rp_frames(
        rp_weights, timeset_duration_entries, period_name,
    )
    for basename, df in frames.items():
        _write(df, sd / basename)


def _compute_timeset_cost_weight_rows(
    active_time_list: dict[str, list],
    timesets_used_by_solve: list[tuple[str, str]],
    timeset_weights: dict[str, dict[str, float]],
) -> tuple[list[tuple[Any, ...]], bool]:
    """Shared compute for :func:`derive_timeset_cost_weight` and the
    wrapper writer.  Returns ``(rows, any_written)``.
    """
    rows: list[tuple[Any, ...]] = []
    any_written = False
    for period, timeset in timesets_used_by_solve:
        weights = timeset_weights.get(timeset)
        active_steps = active_time_list.get(period, [])
        if weights is None or not active_steps:
            continue
        raw = [float(weights.get(step.timestep, 0.0)) for step in active_steps]
        total = sum(raw)
        n = len(raw)
        if total <= 0 or n == 0:
            continue
        scale = n / total
        for step, w in zip(active_steps, raw):
            rows.append((period, step.timestep, f"{w * scale:.10g}"))
        any_written = True
    return rows, any_written


def derive_timeset_cost_weight(
    active_time_list: dict[str, list],
    timesets_used_by_solve: list[tuple[str, str]],
    timeset_weights: dict[str, dict[str, float]],
) -> pl.DataFrame:
    """Build the non-RP ``rp_cost_weight`` frame (period, time, weight).

    Empty body when no active timeset on the current solve has
    ``timeset_weights`` defined.
    """
    rows, _ = _compute_timeset_cost_weight_rows(
        active_time_list, timesets_used_by_solve, timeset_weights,
    )
    return _to_utf8_frame(("period", "time", "weight"), rows)


def write_timeset_cost_weight(
    active_time_list: dict[str, list],
    timesets_used_by_solve: list[tuple[str, str]],
    timeset_weights: dict[str, dict[str, float]],
    work_folder: Path | None = None,
) -> bool:
    """Emit ``rp_cost_weight.csv`` from user-supplied per-timestep
    ``timeset_weights`` (non-RP pathway).

    For each (period, timeset) pair, look up the timeset's weight map
    and the period's active step list.  Per-step weights are
    normalised to sum to 1 across the period, then scaled by the
    number of active steps so that a uniform input reproduces the
    default ``weight = 1`` per step.  Timesteps absent from the user
    map are treated as ``0`` before normalisation.

    Returns ``True`` if any rows were written, ``False`` when no
    active timeset on the current solve has ``timeset_weights``
    defined (orchestrator falls back to default unit weights).
    """
    wf = work_folder if work_folder is not None else Path.cwd()
    sd = wf / "solve_data"
    rows, any_written = _compute_timeset_cost_weight_rows(
        active_time_list, timesets_used_by_solve, timeset_weights,
    )
    if not any_written:
        return False
    _write(
        _to_utf8_frame(("period", "time", "weight"), rows),
        sd / "rp_cost_weight.csv",
    )
    return True


_EMPTY_RP_HEADERS: dict[str, tuple[str, ...]] = {
    "rp_weights.csv": ("base_start", "rep_start", "weight"),
    "rp_base_chain.csv": ("base_start", "prev_base_start"),
    "rp_base_first.csv": ("base_start",),
    "rp_base_last.csv": ("base_start",),
    "rp_block_first.csv": ("period", "step"),
    "rp_block_last.csv": ("period", "step"),
    "rp_block_start_last.csv": ("rep_start", "last_step"),
    "rp_cost_weight.csv": ("period", "time", "weight"),
}


def write_empty_rp_data(work_folder: Path | None = None) -> None:
    """Seed the eight representative-period CSVs with header-only
    files (used by non-RP models so ``table data IN`` declarations in
    ``flextool.mod`` find the expected files).
    """
    wf = work_folder if work_folder is not None else Path.cwd()
    sd = wf / "solve_data"
    for filename, headers in _EMPTY_RP_HEADERS.items():
        _write(_empty_frame(headers), sd / filename)
