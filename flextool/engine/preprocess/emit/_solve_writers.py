"""Per-solve preprocessing emitters — timeline / period / branch / RP frames.

This module owns the ~36 ``emit_*`` functions that build the
``solve_data/*.csv`` artefacts each sub-solve consumes:

* Timeline & period: ``steps_in_timeline``, ``steps_in_use``,
  ``steps_complete_solve``, ``step_previous``, ``period_block_time``,
  ``period_block_succ``, ``p_years_represented``,
  ``period_with_history``, ``p_discount_years``,
  ``realized_invest_periods_of_current_solve``,
  ``invest_periods_of_current_solve``, ``period_last``,
  ``period_first_of_solve``, ``period_first``, ``p_model``,
  ``p_nested_model``, ``solve_current``, ``first_timesteps``,
  ``last_timesteps``,
  ``realized_dispatch``, ``fix_storage_timesteps``.
* Branch: ``period__branch``,
  ``solve_branch_weight``, ``solve_branch__time_branch``.
* Empty / seed: ``p_entity_invested``, ``p_entity_divested``,
  ``p_entity_period_existing_capacity``, the three
  rolling-accumulator seeds, the four fix-storage seeds.
* Headers / timesets: header-only seeds, ``timesets_in_use``,
  ``timesets__timeline``.
* Scaling / delay / RP: ``p_use_row_scaling``,
  ``scale_the_objective``, ``delay_duration``,
  ``dtt__delay_duration``, and the two representative-period CSVs
  (``rp_weights``, ``rp_base_chain``, ``rp_base_first``,
  ``rp_base_last``, ``rp_block_first``, ``rp_block_last``,
  ``rp_block_start_last``, ``timestep_weight``).

Each ``emit_*`` builds an all-``Utf8`` polars frame (inline or via a
shared ``_compute_*`` helper) and registers it on the Provider with the
dual ``basename`` + ``parent/basename`` keys downstream readers expect.
When the cascade later flushes to disk (``--csv-dump``), the CRLF line
terminator preserved through the snapshot path matches the legacy
``csv.writer`` + ``newline=""`` byte shape exactly.

The scaling CSVs use ``%.17g`` repr formatting for round-trip safety.
``emit_rp_data`` preserves the canonical ordering and the
``weight > 1e-10`` epsilon filter.  ``emit_timeset_cost_weight`` uses
``%.10g`` for the normalised float.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl

from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.preprocess import _provider_keys as K
from flextool.engine.preprocess._provider_io import _emit


def _emit_path(provider, path: "Path | str", df: pl.DataFrame) -> None:
    """Provider-emit *df* under both ``basename`` and ``parent/basename``.

    The dual-key registration matches downstream readers' lookup
    conventions; either form resolves to the same frame.
    """
    p = Path(path)
    parent = p.parent.name
    key = f"{parent}/{p.name}" if parent else p.name
    _emit(provider, key, df)


# ---------------------------------------------------------------------------
# Frame construction — :func:`_to_utf8_frame` builds an all-``Utf8``
# polars frame from a header tuple and a row list.  Every value is taken
# verbatim if already a string, otherwise stringified via ``str(v)`` —
# matching ``csv.writer``'s behaviour for the types this module emits
# (``int`` 0/1 flags, ``float`` weights, plain strings).  Schema is
# locked to ``Utf8`` per column so polars' ``write_csv`` does no extra
# numeric formatting or quoting at flush time.
# ---------------------------------------------------------------------------


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


def emit_full_timelines(
    stochastic_timesteps: list[tuple[str, str]],
    period__timesets_in_this_solve: list[tuple[str, str]],
    timesets__timeline: dict[str, str],
    timelines: dict[str, list[tuple[str, ...]]],
    filename: str,
    *, provider,
) -> None:
    """Emit ``full_timelines`` to the Provider."""
    _emit_path(provider, filename,
               derive_full_timelines(
                   stochastic_timesteps,
                   period__timesets_in_this_solve,
                   timesets__timeline,
                   timelines,
               ))


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


def emit_active_timelines(
    timeline: dict[str, list[tuple[str, ...]]],
    filename: str,
    complete: bool = False,
    *, provider,
) -> None:
    """Emit ``active_timelines`` to the Provider."""
    _emit_path(provider, filename,
               derive_active_timelines(timeline, complete))


_STEP_JUMP_HEADERS = (
    "period", "time", "previous", "previous_within_timeset",
    "previous_period", "previous_within_solve", "jump",
)


def derive_step_jump(step_lengths: list[tuple[str, ...]]) -> pl.DataFrame:
    """Build the canonical ``step_previous`` frame (7 Utf8 columns)."""
    return _to_utf8_frame(_STEP_JUMP_HEADERS, list(step_lengths))


def emit_step_jump(
    step_lengths: list[tuple[str, ...]],
    *, provider,
) -> None:
    """Emit ``step_jump`` to the Provider."""
    _emit(provider, "solve_data/step_previous.csv",
                   derive_step_jump(step_lengths))


_ROLL_FIRST_STEP_LINK_HEADERS = _STEP_JUMP_HEADERS + (
    "previous_end_rank", "previous_end_root", "previous_end_index", "previous_end_branch",
)


def derive_roll_first_step_link(
    active_time_list: dict,
    complete_step_lengths: list[tuple[str, ...]],
    complete_active_time_list: "dict | None" = None,
    end_of=None,
) -> pl.DataFrame:
    """The complete solve's ``step_previous`` rows (7 Utf8 columns) of the first step of each period of
    ``active_time_list`` (period -> steps with ``.timestep``), in period order, plus where the predecessor ends on
    the base timeline (``previous_end_rank``, ``previous_end_root``, ``previous_end_index``,
    ``previous_end_branch``: ``end_of(previous_period, previous_within_solve, its duration in
    complete_active_time_list)``, ``solve._carry_routing._TimeIndex.end_of``; empty strings when unknown or without
    ``end_of``).  Periods (or first steps) that the complete solve does not have (stochastic branch periods) have
    no row."""
    by_key = {(row[0], row[1]): row for row in complete_step_lengths}
    duration = {(str(period), str(step.timestep)): step.duration
                for period, steps in (complete_active_time_list or {}).items() for step in steps}
    rows = []
    for period, steps in active_time_list.items():
        if not steps:
            continue
        row = by_key.get((period, steps[0].timestep))
        if row is None:
            continue
        pos = None
        if end_of is not None:
            prev = (str(row[4]), str(row[5]))
            if prev in duration:
                pos = end_of(prev[0], prev[1], duration[prev])
        rows.append(tuple(row) + tuple("" if v is None else str(v) for v in (pos or (None,) * 4)))
    return _to_utf8_frame(_ROLL_FIRST_STEP_LINK_HEADERS, rows)


def emit_roll_first_step_link(
    active_time_list: dict,
    complete_step_lengths: list[tuple[str, ...]],
    *, provider,
    complete_active_time_list: "dict | None" = None,
    end_of=None,
) -> None:
    """Emit ``roll_first_step_link`` (the adjacency gate of the ramp carry-in) to the Provider."""
    _emit(provider, "solve_data/roll_first_step_link.csv",
          derive_roll_first_step_link(active_time_list, complete_step_lengths, complete_active_time_list, end_of))


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


def emit_period_block(
    period_block_time: list[tuple],
    period_block_succ: list[tuple],
    *, provider,
) -> None:
    """Emit ``period_block`` to the Provider."""
    _emit(provider, "solve_data/period_block_time.csv",
                   derive_period_block_time(period_block_time))
    _emit(provider, "solve_data/period_block_succ.csv",
                   derive_period_block_succ(period_block_succ))


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
                # Equality, not substring containment (``in``): with
                # prefix-overlapping period names (p1 / p10), ``in``
                # would copy p10's year rows to p1's branches.
                if pd_pair[0] == period__years[0] and pd_pair[0] != pd_pair[1]:
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


def emit_years_represented(
    period__branch: list[tuple[str, str]],
    years_represented: list[tuple[str, str]],
    filename: str,
    *, provider,
) -> None:
    """Emit ``years_represented`` to the Provider."""
    _emit_path(provider, filename,
               derive_years_represented(period__branch, years_represented))


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
            # Equality, not substring containment — see
            # derive_years_represented.
            if pd_pair[0] == period__year[0] and pd_pair[0] != pd_pair[1]:
                rows.append((pd_pair[1], str(year_count)))
        year_count += float(period__year[1])
    return _to_utf8_frame(("period", "param"), rows)


def emit_period_years(
    stochastic_branches: list[tuple[str, str]],
    years_represented: list[tuple[str, str]],
    filename: str,
    *, provider,
) -> None:
    """Emit ``period_years`` to the Provider."""
    _emit_path(provider, filename,
               derive_period_years(stochastic_branches, years_represented))


def derive_periods(
    solve: str,
    periods_dict: dict[str, list[tuple[str, str]]],
) -> pl.DataFrame:
    """Build a single-column ``period`` frame for the given solve key."""
    rows: list[tuple[Any, ...]] = [
        (period_tuple[1],) for period_tuple in periods_dict.get(solve, [])
    ]
    return _to_utf8_frame(("period",), rows)


def emit_periods(
    solve: str,
    periods_dict: dict[str, list[tuple[str, str]]],
    filename: str,
    *, provider,
) -> None:
    """Emit ``periods`` to the Provider."""
    _emit_path(provider, filename, derive_periods(solve, periods_dict))


def emit_stochastic_invest_method(
    method: str,
    filename: str,
    *, provider,
) -> None:
    """Emit the resolved ``solve.stochastic_invest_method`` (Slice D).

    A single-row, single-column (``method``) frame carrying the resolved
    per-solve mode (``"none"`` / ``"recourse"``).  Read run-time via
    ``_derived._recourse_invest_active`` so the derived/model
    layers can gate the recourse feature without a ``SolveConfig``.
    """
    _emit_path(provider, filename,
               _to_utf8_frame(("method",), [(method,)]))


def _compute_first_and_last_periods(
    active_time_list: dict[str, list[tuple[str, ...]]],
    period__timesets_in_this_solve: list[tuple[str, str]],
    period__branch_list: list[tuple[str, str]],
) -> tuple[list[str], list[str], list[str]]:
    """Shared compute for the three first/last period CSVs.

    Returns ``(period_last, period_first_of_solve_list, period_first_list)``
    each as a flat list of period names.

    ``period_last`` is the solve's last period and its stochastic branch
    siblings (the periods of ``period__branch_list`` that share its anchor
    period and end on the same timestep): the periods whose last step ends
    the solve horizon.  An earlier period that merely ends on the same
    timestep name (periods sharing one timeset) is not last (#37).
    """
    period_first_of_solve = list(active_time_list.keys())[0]
    last = list(active_time_list.keys())[-1]
    time_step_last = active_time_list[last][-1].timestep
    anchor = next((pb[0] for pb in period__branch_list if pb[1] == last), last)
    siblings = {pb[1] for pb in period__branch_list if pb[0] == anchor} | {anchor}
    period_last = [last] + [
        period for period in active_time_list.keys()
        if period != last
        and period in siblings
        and active_time_list[period][-1].timestep == time_step_last
    ]

    # The first active key may be a forecast branch period (a fan opening in the solve's first period, whose
    # period label has no steps): list the members of its anchor that have steps, as for period_last.
    first_anchor = next((pb[0] for pb in period__branch_list if pb[1] == period_first_of_solve),
                        period_first_of_solve)
    period_first_of_solve_list = [
        pb[1] for pb in period__branch_list
        if pb[0] == first_anchor and pb[1] in active_time_list
    ]
    period_first = period__timesets_in_this_solve[0][0]
    period_first_list = [
        pb[1] for pb in period__branch_list if pb[0] == period_first
    ]
    return period_last, period_first_of_solve_list, period_first_list


def emit_first_and_last_periods(
    active_time_list: dict[str, list[tuple[str, ...]]],
    period__timesets_in_this_solve: list[tuple[str, str]],
    period__branch_list: list[tuple[str, str]],
    *, provider,
) -> None:
    """Emit ``first_and_last_periods`` to the Provider."""
    period_last, period_first_of_solve_list, period_first_list = (
        _compute_first_and_last_periods(
            active_time_list,
            period__timesets_in_this_solve,
            period__branch_list,
        )
    )
    _emit(provider, "solve_data/period_last.csv",
                   _to_utf8_frame(("period",), [(p,) for p in period_last]))
    _emit(provider, "solve_data/period_first_of_solve.csv",
                   _to_utf8_frame(("period",),
                                  [(p,) for p in period_first_of_solve_list]))
    _emit(provider, "solve_data/period_first.csv",
                   _to_utf8_frame(("period",),
                                  [(p,) for p in period_first_list]))


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


def emit_solve_status(
    first_state: bool,
    last_state: bool,
    nested: bool = False,
    *, provider,
) -> None:
    """Emit ``solve_status`` to the Provider."""
    key = ("solve_data/p_nested_model.csv" if nested
           else "solve_data/p_model.csv")
    _emit(provider, key,
                   derive_solve_status(first_state, last_state, nested))


def derive_current_solve(solve: str) -> pl.DataFrame:
    """Build the ``solve_current`` frame (solve)."""
    return _to_utf8_frame(("solve",), [(solve,)])


def emit_current_solve(solve: str, filename: str, *, provider) -> None:
    """Emit ``current_solve`` to the Provider."""
    _emit_path(provider, filename, derive_current_solve(solve))


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


def emit_period_boundary_step(
    timeline: dict[str, list[tuple[str, ...]]],
    filename: str,
    *,
    last: bool = False,
    provider,
) -> None:
    """Emit ``period_boundary_step`` to the Provider."""
    _emit_path(provider, filename,
               derive_period_boundary_step(timeline, last=last))


def emit_first_steps(
    timeline: dict[str, list[tuple[str, ...]]],
    filename: str,
    *, provider,
) -> None:
    """Emit ``first_steps`` to the Provider."""
    emit_period_boundary_step(timeline, filename, last=False, provider=provider)


def emit_last_steps(
    timeline: dict[str, list[tuple[str, ...]]],
    filename: str,
    *, provider,
) -> None:
    """Emit ``last_steps`` to the Provider."""
    emit_period_boundary_step(timeline, filename, last=True, provider=provider)


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


def emit_realized_dispatch(
    realized_time_list: dict[str, list[tuple[str, ...]]],
    solve: str,
    realized_periods: list[tuple[str, str]],
    *, provider,
) -> None:
    """Emit ``realized_dispatch`` to the Provider."""
    _emit(
        provider, "solve_data/realized_dispatch.csv",
        derive_realized_dispatch(realized_time_list, solve, realized_periods),
    )


def derive_periods_steps(time_list: dict[str, list[tuple[str, ...]]]) -> pl.DataFrame:
    """Build a (period, step) frame of every step of *time_list*, in list order."""
    rows = [(period, i.timestep) for period, steps in time_list.items() for i in steps]
    return _to_utf8_frame(("period", "step"), rows)


def emit_fan_label_steps(
    committed_in_fan: "pl.DataFrame | None",
    label_steps: "pl.DataFrame | None",
    *, provider,
) -> None:
    """Emit ``committed_in_fan`` and ``label_steps`` (period, step) to the Provider.

    A period label has no dispatch, so it is in neither ``steps_in_use`` nor ``realized_dispatch``.
    ``committed_in_fan`` holds the cells the solve commits inside its fan (the labels' realized and fix-storage
    cells before the stochastic expansion); ``label_steps`` every step the labels had.  Both are header-only for a
    solve without a fan.  The non-anticipativity window emitter reads them.
    """
    empty = _to_utf8_frame(("period", "step"), [])
    _emit(provider, "solve_data/committed_in_fan.csv",
          committed_in_fan if committed_in_fan is not None else empty)
    _emit(provider, "solve_data/label_steps.csv", label_steps if label_steps is not None else empty)


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


def emit_fix_storage_timesteps(
    active_time_list: dict[str, list[tuple[str, ...]]],
    solve: str,
    fix_storage_periods: list[tuple[str, str]],
    *, provider,
) -> None:
    """Emit ``fix_storage_timesteps`` to the Provider."""
    _emit(
        provider, "solve_data/fix_storage_timesteps.csv",
        derive_fix_storage_timesteps(
            active_time_list, solve, fix_storage_periods,
        ),
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


def emit_branch__period_relationship(
    period__branch: list[tuple[str, str]],
    filename: str,
    *, provider,
) -> None:
    """Emit ``branch__period_relationship`` to the Provider."""
    _emit_path(provider, filename,
               derive_branch__period_relationship(period__branch))


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


def emit_branch_weights_and_map(
    complete_solve: str,
    active_time_list: dict[str, list[tuple[str, ...]]],
    solve_branch__time_branch_list: list[tuple[str, str]],
    branch_start_time: tuple[str, str] | None,
    period__branch_lists: list[tuple[str, str]],
    stochastic_branches: dict[str, list[Any]],
    *, provider,
) -> None:
    """Emit ``branch_weights_and_map`` to the Provider."""
    _emit(
        provider, "solve_data/solve_branch_weight.csv",
        derive_solve_branch_weight(
            complete_solve, active_time_list, solve_branch__time_branch_list,
            branch_start_time, period__branch_lists, stochastic_branches,
        ),
    )
    _emit(
        provider, "solve_data/solve_branch__time_branch.csv",
        derive_solve_branch__time_branch(solve_branch__time_branch_list),
    )


def branch_time_start_rows(
    period__branch: list[tuple[str, str]],
    branch_start_time: tuple[str, str] | None,
) -> list[tuple[str, str]]:
    """``(period, fan start step)`` rows of a solve's fan.

    *branch_start_time* is the solve's ``(fan period, fan step)`` (one fan per solve, roll and nested
    inner solve; ``None`` without one).  Every anchor period from the fan's period onwards and each of its
    ``<period>_<branch>`` copies gets one row; a pre-reveal trunk period gets none.  Row order: the anchors in
    the order they first appear in the period column of *period__branch*, each anchor followed by its copies
    in their *period__branch* row order (the anchor's own ``(period, period)`` row, if any, adds no second
    row).  A branch period whose own first step authors no stochastic-time value for a cell reads the
    forecast authored at this step.
    """
    if branch_start_time is None:
        return []
    fan_period, fan_step = branch_start_time
    order = list(dict.fromkeys(period for period, _branch in period__branch))
    if fan_period not in order:
        return []
    after = set(order[order.index(fan_period):])
    members: dict[str, list[str]] = {}
    for period, branch in period__branch:
        if period in after:
            members.setdefault(period, []).append(branch)
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for period in order[order.index(fan_period):]:
        for d in (period, *members.get(period, ())):
            if d not in seen:
                seen.add(d)
                rows.append((d, fan_step))
    return rows


def project_branch_time_start_rows(
    parent_rows: list[tuple[str, str]],
    parent_period_branch: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """The parent's ``branch_time_start`` rows of its real period names, for its realization-dispatch sibling.

    The sibling dispatches the real period names with the parent's realized branch (the projected
    ``solve_branch__time_branch`` rows), so it falls back to the same fan start; the ``(p, p)`` self-row
    predicate is the one of ``project_realized_branch_time_branch_rows``.
    """
    period_branch = set(parent_period_branch)
    return [(d, step) for d, step in parent_rows if (d, d) in period_branch]


def derive_branch_time_start(rows: list[tuple[str, str]]) -> pl.DataFrame:
    """Build the ``branch_time_start`` frame (period, step, read_first); header-only without a fan.

    ``read_first`` is ``fan`` on every row: a period from the fan onwards reads the forecast made at the fan's
    analysis time first, in a single solve and in a roll of a rolling solve alike (item 255); see ``_pdt_lookup``
    "Which forecast a branch period reads".
    """
    return _to_utf8_frame(("period", "step", "read_first"), [(d, step, "fan") for d, step in rows])


def emit_branch_time_start(rows: list[tuple[str, str]], *, provider) -> None:
    """Emit ``solve_data/branch_time_start`` to the Provider."""
    _emit(provider, "solve_data/branch_time_start.csv", derive_branch_time_start(rows))


# ---------------------------------------------------------------------------
# Init / empty files
# ---------------------------------------------------------------------------


def _empty_frame(headers: tuple[str, ...]) -> pl.DataFrame:
    """Build a header-only all-Utf8 polars frame."""
    return pl.DataFrame(schema={h: pl.Utf8 for h in headers})


def emit_empty_investment_file(
    *, provider) -> None:
    """Emit ``empty_investment_file`` to the Provider."""
    _emit(provider, "solve_data/p_entity_invested.csv",
                   _empty_frame(("entity", "p_entity_invested")))
    _emit(provider, "solve_data/p_entity_divested.csv",
                   _empty_frame(("entity", "p_entity_divested")))
    _emit(
        provider, "solve_data/p_entity_period_existing_capacity.csv",
        _empty_frame((
            "entity", "period",
            "p_entity_period_existing_capacity",
            "p_entity_period_invested_capacity",
        )),
    )


def emit_empty_storage_fix_file(
    *, provider) -> None:
    """Emit ``empty_storage_fix_file`` to the Provider."""
    _emit(provider, "solve_data/fix_storage_price.csv",
                   _empty_frame(
                       ("node", " period", " step", " ndt_fix_storage_price"),
                   ))
    _emit(provider, "solve_data/fix_storage_quantity.csv",
                   _empty_frame(
                       ("node", " period", " step", " ndt_fix_storage_quantity"),
                   ))
    _emit(provider, "solve_data/fix_storage_usage.csv",
                   _empty_frame(
                       ("node", " period", " step", " ndt_fix_storage_usage"),
                   ))
    _emit(provider, "solve_data/p_roll_continue_state.csv",
                   _empty_frame(("node", " p_roll_continue_state")))


def emit_headers_for_empty_output_files(filename: str, header: str,
                                          *, provider) -> None:
    """Emit ``headers_for_empty_output_files`` to the Provider."""
    _emit_path(provider, filename, _empty_frame(tuple(header.split(","))))


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
    original_timeline: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Build the ``timesets__timeline`` frame (timesets, timeline, original_timeline).

    ``original_timeline`` is the unaggregated timeline a solve-level
    ``new_stepduration`` timeline was built from (``TimelineConfig.original_timeline``),
    else the timeline itself."""
    original_timeline = original_timeline or {}

    def _original(tl: str) -> str:
        seen: set[str] = set()
        while tl in original_timeline and tl not in seen:
            seen.add(tl)
            tl = original_timeline[tl]
        return tl

    rows = [(ts, tl, _original(tl)) for ts, tl in timeset__timeline.items()]
    return _to_utf8_frame(("timesets", "timeline", "original_timeline"), rows)


def emit_timesets(
    timesets_used_by_solves: dict[str, list[tuple[str, str]]],
    timeset__timeline: dict[str, str],
    original_timeline: dict[str, str] | None = None,
    *, provider,
) -> None:
    """Emit ``timesets`` to the Provider."""
    _emit(provider, "input/timesets_in_use.csv",
                   derive_timesets_in_use(timesets_used_by_solves))
    _emit(provider, "input/timesets__timeline.csv",
                   derive_timesets__timeline(timeset__timeline, original_timeline))


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


def emit_p_use_row_scaling(
    solve: str,
    use_row_scaling: dict[str, str],
    filename: str,
    *, provider,
) -> None:
    """Emit ``p_use_row_scaling`` to the Provider."""
    _emit_path(provider, filename,
               derive_p_use_row_scaling(solve, use_row_scaling))


def derive_scale_the_objective(value: float) -> pl.DataFrame:
    """Build the ``scale_the_objective`` keyed-value frame
    (key=``v``, value=``%.17g``).
    """
    return _to_utf8_frame(
        ("key", "value"), [("v", f"{float(value):.17g}")],
    )


def emit_scale_the_objective(
    solve_data_dir: Path | str,
    value: float,
    *, provider,
) -> Path:
    """Emit ``scale_the_objective`` to the Provider."""
    path = Path(solve_data_dir) / "scale_the_objective.csv"
    _emit(provider, "solve_data/scale_the_objective.csv",
                   derive_scale_the_objective(value))
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


def emit_delayed_durations(
    active_time_list: dict[str, list[tuple[str, ...]]],
    solve: str,
    delay_durations: dict[str, Any],
    *, provider,
) -> None:
    """Emit ``delayed_durations`` to the Provider."""
    _emit(provider, "solve_data/delay_duration.csv",
                   derive_delay_duration(delay_durations))
    _emit(provider, "solve_data/dtt__delay_duration.csv",
                   derive_dtt__delay_duration(active_time_list,
                                                delay_durations))


# ---------------------------------------------------------------------------
# Sub-dispatch 7 — representative period writers
# ---------------------------------------------------------------------------


def _compute_rp_frames(
    per_period_inputs: (
        "list[tuple[str, dict[str, dict[str, float]], "
        "list[tuple[str, float]], list[str]]]"
    ),
    period_previous: "list[tuple[str, str]] | None" = None,
) -> dict[str, pl.DataFrame]:
    """The representative-period frames of one solve: every rp period has its own data.

    ``per_period_inputs`` holds one ``(period, rp_weights, timeset_duration_entries, timeline_steps)`` entry
    per period of the solve whose timeset has ``representative_period_weights``, in the solve's period order.
    Two periods may use the same timeset or the same timeline (same base-period names): every frame carries
    the owning ``period``, so their base-period chains, blocks and weights stay apart.
    ``period_previous`` lists ``(period, previous period)`` in solve order (the chain from one period's last
    base period to the next period's first, used by the across-solve variants).

    Output basenames (columns): ``rp_weights.csv`` (period, base_start, rep_start, weight),
    ``rp_base_chain.csv`` (period, base_start, prev_base_start; within-period edges only),
    ``rp_base_first.csv`` / ``rp_base_last.csv`` (period, base_start), ``rp_period_previous.csv``
    (period, prev_period), ``rp_block_first.csv`` / ``rp_block_last.csv`` (period, step),
    ``rp_block_start_last.csv`` (period, rep_start, last_step), ``rp_base_period_set.csv``
    (period, base_start), ``rp_rep_period_set.csv`` (period: the rep starts), ``timestep_weight.csv``
    (period, time, weight).
    """
    chain_rows: list[tuple[Any, ...]] = []
    first_rows: list[tuple[Any, ...]] = []
    last_rows: list[tuple[Any, ...]] = []
    block_first_rows: list[tuple[Any, ...]] = []
    block_last_rows: list[tuple[Any, ...]] = []
    block_sl_rows: list[tuple[Any, ...]] = []
    base_period_rows: list[tuple[Any, ...]] = []
    rep_period_rows: list[tuple[Any, ...]] = []
    weights_rows: list[tuple[Any, ...]] = []
    cost_rows: list[tuple[Any, ...]] = []
    for p_name, p_rp_w, p_ts_dur, p_tl_steps in per_period_inputs:
        sub = _compute_rp_period_frames(p_rp_w, p_ts_dur, p_name, p_tl_steps)
        chain_rows.extend((p_name,) + r for r in sub["rp_base_chain.csv"].iter_rows())
        first_rows.extend((p_name,) + r for r in sub["rp_base_first.csv"].iter_rows())
        last_rows.extend((p_name,) + r for r in sub["rp_base_last.csv"].iter_rows())
        block_first_rows.extend(sub["rp_block_first.csv"].iter_rows())
        block_last_rows.extend(sub["rp_block_last.csv"].iter_rows())
        block_sl_rows.extend((p_name,) + r for r in sub["rp_block_start_last.csv"].iter_rows())
        base_period_rows.extend((p_name,) + r for r in sub["rp_base_period_set.csv"].iter_rows())
        rep_period_rows.extend(sub["rp_rep_period_set.csv"].iter_rows())
        weights_rows.extend((p_name,) + r for r in sub["rp_weights.csv"].iter_rows())
        cost_rows.extend(sub["timestep_weight.csv"].iter_rows())
    out: dict[str, pl.DataFrame] = {}
    out["rp_weights.csv"] = _to_utf8_frame(("period", "base_start", "rep_start", "weight"), weights_rows)
    out["rp_base_chain.csv"] = _to_utf8_frame(("period", "base_start", "prev_base_start"), chain_rows)
    out["rp_base_first.csv"] = _to_utf8_frame(("period", "base_start"), first_rows)
    out["rp_base_last.csv"] = _to_utf8_frame(("period", "base_start"), last_rows)
    out["rp_period_previous.csv"] = _to_utf8_frame(("period", "prev_period"), list(period_previous or []))
    out["rp_block_first.csv"] = _to_utf8_frame(("period", "step"), block_first_rows)
    out["rp_block_last.csv"] = _to_utf8_frame(("period", "step"), block_last_rows)
    out["rp_block_start_last.csv"] = _to_utf8_frame(("period", "rep_start", "last_step"), block_sl_rows)
    out["rp_base_period_set.csv"] = _to_utf8_frame(("period", "base_start"), base_period_rows)
    out["rp_rep_period_set.csv"] = _to_utf8_frame(("period",), rep_period_rows)
    out["timestep_weight.csv"] = _to_utf8_frame(("period", "time", "weight"), cost_rows)
    return out


def _compute_rp_period_frames(
    rp_weights: dict[str, dict[str, float]],
    timeset_duration_entries: list[tuple[str, float]],
    period_name: str,
    timeline_steps: list[str],
) -> dict[str, pl.DataFrame]:
    """The representative-period frames of ONE period (no ``period`` column on the base-period frames).

    ``timeline_steps`` is the ordered list of step IDs for the timeline this timeset belongs to; it anchors
    ``(start, count)`` block entries to real step labels (e.g. ``2050-01-01T00:00:00``) rather than the
    synthetic ``t####`` form.  :func:`_compute_rp_frames` tags the frames with the period.
    """
    step_to_idx: dict[str, int] = {s: i for i, s in enumerate(timeline_steps)}

    # RP block boundaries from the timeset_duration entries.
    rp_starts: list[str] = []
    rp_lasts: list[str] = []
    rp_ranges: list[tuple[int, int]] = []  # inclusive (start_idx, last_idx)
    for start_step, count in timeset_duration_entries:
        start_step = str(start_step)
        if start_step not in step_to_idx:
            raise KeyError(
                f"Representative-period block start {start_step!r} not "
                f"found in timeline (period={period_name!r}). Check that "
                "timeset_duration entries reference real timeline steps."
            )
        start_idx = step_to_idx[start_step]
        last_idx = start_idx + int(float(count)) - 1
        if last_idx >= len(timeline_steps):
            raise IndexError(
                f"Representative-period block {start_step!r} + "
                f"{int(float(count))} steps overruns timeline length "
                f"({len(timeline_steps)}) for period={period_name!r}."
            )
        rp_starts.append(start_step)
        rp_lasts.append(timeline_steps[last_idx])
        rp_ranges.append((start_idx, last_idx))

    # A base period is ordered by its start step on the timeline (the inter-period storage chain links the
    # base periods in that order), so every base start must be a step of the timeline.
    off_timeline = sorted(str(s) for s in rp_weights if s not in step_to_idx)
    if off_timeline:
        span = f", {timeline_steps[0]!r} .. {timeline_steps[-1]!r}" if timeline_steps else ""
        raise FlexToolConfigError(
            f"representative_period_weights of period {period_name!r}: the base period(s) "
            f"{off_timeline} do not start at a time step of the period's timeline "
            f"({len(timeline_steps)} steps{span}).  Each base "
            "period (the first index of representative_period_weights) must be the first step of a "
            "period of the timeline; extend the timeline or correct the base period names."
        )
    base_starts = sorted(rp_weights.keys(), key=lambda s: step_to_idx[s])
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

    # 5b. rp_base_period_set.csv / rp_rep_period_set.csv —
    # ordered-unique projections of the rp_weights base/rep columns.
    # Owned by _compute_rp_frames because emit_per_solve_sets runs
    # BEFORE emit_rp_data in the cascade, so the per-solve emit can't
    # see ``rp_weights`` yet.  Column header ``period`` matches the
    # convention used by ``_emit_singles``-style set frames; the
    # FlexData loader renames to its axis name on load.
    out["rp_base_period_set.csv"] = _to_utf8_frame(
        ("period",), [(b,) for b in base_starts],
    )
    out["rp_rep_period_set.csv"] = _to_utf8_frame(
        ("period",), [(r,) for r in rp_starts],
    )

    # 6. timestep_weight.csv — normalised per-timestep weight.
    w_r: dict[str, float] = {r: 0.0 for r in rp_starts}
    for base_weights in rp_weights.values():
        for rep, weight in base_weights.items():
            if rep in w_r:
                w_r[rep] += weight
    for rep in w_r:
        w_r[rep] = w_r[rep] * n_rp / n_base if n_base > 0 else 1.0
    cost_rows: list[tuple[Any, ...]] = []
    for (start_idx, last_idx), start in zip(rp_ranges, rp_starts):
        weight = w_r[start]
        for t_idx in range(start_idx, last_idx + 1):
            cost_rows.append(
                (period_name, timeline_steps[t_idx], str(weight))
            )
    out["timestep_weight.csv"] = _to_utf8_frame(
        ("period", "time", "weight"), cost_rows,
    )

    return out


# Map from the basename keys produced by :func:`_compute_rp_frames`
# to the canonical Provider keys (no ``.csv`` suffix, per the Phase 3b
# convention in :mod:`flextool.engine.preprocess._provider_keys`).  Used by
# both :func:`emit_rp_data` and :func:`emit_empty_rp_data` so the six
# restored RP basenames + the two pre-existing ones land under the
# named ``K.SOLVE_DATA_*`` constants.
_RP_BASENAME_TO_PROVIDER_KEY: dict[str, str] = {
    "rp_weights.csv": K.SOLVE_DATA_RP_WEIGHTS,
    "rp_base_chain.csv": K.SOLVE_DATA_RP_BASE_CHAIN,
    "rp_base_first.csv": K.SOLVE_DATA_RP_BASE_FIRST,
    "rp_base_last.csv": K.SOLVE_DATA_RP_BASE_LAST,
    "rp_block_first.csv": K.SOLVE_DATA_RP_BLOCK_FIRST,
    "rp_block_last.csv": K.SOLVE_DATA_RP_BLOCK_LAST,
    "rp_block_start_last.csv": K.SOLVE_DATA_RP_BLOCK_START_LAST,
    "rp_base_period_set.csv": K.SOLVE_DATA_RP_BASE_PERIOD_SET,
    "rp_rep_period_set.csv": K.SOLVE_DATA_RP_REP_PERIOD_SET,
    "rp_period_previous.csv": K.SOLVE_DATA_RP_PERIOD_PREVIOUS,
    "timestep_weight.csv": "solve_data/timestep_weight",
}


def emit_rp_data(
    per_period_inputs: (
        "list[tuple[str, dict[str, dict[str, float]], "
        "list[tuple[str, float]], list[str]]]"
    ),
    *, provider,
    period_previous: "list[tuple[str, str]] | None" = None,
    extra_timestep_weight_rows: "list[tuple[Any, ...]] | None" = None,
) -> None:
    """Emit the representative-period frames of a solve (:func:`_compute_rp_frames`) to the Provider.

    ``extra_timestep_weight_rows`` are ``(period, time, weight)`` rows of the solve's periods without
    representative periods (their ``timeset_weights``), appended to ``timestep_weight`` as rendered by
    :func:`_compute_timeset_cost_weight_rows`.
    """
    frames = _compute_rp_frames(per_period_inputs, period_previous)
    if extra_timestep_weight_rows:
        frames["timestep_weight.csv"] = pl.concat(
            [frames["timestep_weight.csv"],
             _to_utf8_frame(("period", "time", "weight"), extra_timestep_weight_rows)],
            how="vertical",
        )
    for basename, df in frames.items():
        _emit(provider, _RP_BASENAME_TO_PROVIDER_KEY[basename], df)


def _with_branch_period_timesets(
    timesets_used_by_solve: list[tuple[str, str]],
    period__branch: "list[tuple[str, str]] | None",
) -> list[tuple[str, str]]:
    """``(period, timeset)`` pairs plus, after each period, its stochastic branch periods with the
    period's timeset (a branch period is a copy of its anchor period).  A branch period that has a
    timeset of its own keeps it."""
    own = {period for period, _ts in timesets_used_by_solve}
    out: list[tuple[str, str]] = []
    for period, timeset in timesets_used_by_solve:
        out.append((period, timeset))
        for pair in period__branch or ():
            anchor, branch = pair[0], pair[1]
            if anchor == period and branch != anchor and branch not in own:
                out.append((branch, timeset))
                own.add(branch)
    return out


def _compute_timeset_cost_weight_rows(
    active_time_list: dict[str, list],
    timesets_used_by_solve: list[tuple[str, str]],
    timeset_weights: dict[str, dict[str, float]],
    period__branch: "list[tuple[str, str]] | None" = None,
) -> tuple[list[tuple[Any, ...]], bool]:
    """Shared compute for :func:`emit_timeset_cost_weight`.
    Returns ``(rows, any_written)``.  The stochastic branch periods of ``period__branch``
    (``(anchor, branch)`` rows) take their anchor period's timeset weights.
    """
    rows: list[tuple[Any, ...]] = []
    any_written = False
    for period, timeset in _with_branch_period_timesets(timesets_used_by_solve, period__branch):
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


def emit_timeset_cost_weight(
    active_time_list: dict[str, list],
    timesets_used_by_solve: list[tuple[str, str]],
    timeset_weights: dict[str, dict[str, float]],
    *, provider,
    period__branch: "list[tuple[str, str]] | None" = None,
) -> bool:
    """Emit ``timeset_cost_weight`` to the Provider."""
    rows, any_written = _compute_timeset_cost_weight_rows(
        active_time_list, timesets_used_by_solve, timeset_weights, period__branch,
    )
    if not any_written:
        return False
    _emit(provider, "solve_data/timestep_weight.csv",
                   _to_utf8_frame(("period", "time", "weight"), rows))
    return True


_EMPTY_RP_HEADERS: dict[str, tuple[str, ...]] = {
    "rp_weights.csv": ("period", "base_start", "rep_start", "weight"),
    "rp_base_chain.csv": ("period", "base_start", "prev_base_start"),
    "rp_base_first.csv": ("period", "base_start"),
    "rp_base_last.csv": ("period", "base_start"),
    "rp_period_previous.csv": ("period", "prev_period"),
    "rp_block_first.csv": ("period", "step"),
    "rp_block_last.csv": ("period", "step"),
    "rp_block_start_last.csv": ("period", "rep_start", "last_step"),
    "rp_base_period_set.csv": ("period", "base_start"),
    "rp_rep_period_set.csv": ("period",),
    "timestep_weight.csv": ("period", "time", "weight"),
}


def emit_empty_rp_data(
    *, provider) -> None:
    """Emit ``empty_rp_data`` to the Provider."""
    for filename, headers in _EMPTY_RP_HEADERS.items():
        _emit(provider, _RP_BASENAME_TO_PROVIDER_KEY[filename],
                       _empty_frame(headers))
