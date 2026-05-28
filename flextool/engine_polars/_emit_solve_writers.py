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
  ``rp_block_start_last``, ``rp_cost_weight``).

Each emitter builds an all-``Utf8`` polars frame via a companion
``derive_*`` helper and registers it on the Provider with the dual
``basename`` + ``parent/basename`` keys downstream readers expect.
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

from flextool.engine_polars import _provider_keys as K
from flextool.engine_polars._emit_provider_io import _emit


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
            if pd_pair[0] in period__year[0] and pd_pair[0] != pd_pair[1]:
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
) -> pl.DataFrame:
    """Build the ``timesets__timeline`` frame (timesets, timeline)."""
    rows = [(ts, tl) for ts, tl in timeset__timeline.items()]
    return _to_utf8_frame(("timesets", "timeline"), rows)


def emit_timesets(
    timesets_used_by_solves: dict[str, list[tuple[str, str]]],
    timeset__timeline: dict[str, str],
    *, provider,
) -> None:
    """Emit ``timesets`` to the Provider."""
    _emit(provider, "input/timesets_in_use.csv",
                   derive_timesets_in_use(timesets_used_by_solves))
    _emit(provider, "input/timesets__timeline.csv",
                   derive_timesets__timeline(timeset__timeline))


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


def derive_scale_the_objective_header_only() -> pl.DataFrame:
    """Build the header-only ``scale_the_objective`` frame."""
    return _empty_frame(("key", "value"))


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


def emit_scale_the_objective_header_only(
    solve_data_dir: Path | str,
    *, provider,
) -> Path:
    """Emit ``scale_the_objective_header_only`` to the Provider."""
    path = Path(solve_data_dir) / "scale_the_objective.csv"
    _emit(provider, "solve_data/scale_the_objective.csv",
                   derive_scale_the_objective_header_only())
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
    rp_weights: dict[str, dict[str, float]],
    timeset_duration_entries: list[tuple[str, float]],
    period_name: str,
    timeline_steps: list[str],
    *,
    variant: str = "within_solve",
    per_period_inputs: (
        "list[tuple[str, dict[str, dict[str, float]], "
        "list[tuple[str, float]], list[str]]] | None"
    ) = None,
) -> dict[str, pl.DataFrame]:
    """Shared compute for the eight representative-period CSVs.

    Returns a dict keyed by output basename — used by both the
    ``derive_*`` accessors and the wrapper writer.

    Output basenames: ``rp_weights.csv``, ``rp_base_chain.csv``,
    ``rp_base_first.csv``, ``rp_base_last.csv``, ``rp_block_first.csv``,
    ``rp_block_last.csv``, ``rp_block_start_last.csv``,
    ``rp_base_period_set.csv``, ``rp_rep_period_set.csv``,
    ``rp_cost_weight.csv``.

    ``timeline_steps`` is the ordered list of step IDs for the timeline
    this timeset belongs to; it anchors ``(start, count)`` block entries
    to real step labels (e.g. ``2050-01-01T00:00:00``) rather than the
    synthetic ``t####`` form.

    Phase E — ``variant`` toggles the RP chain TOPOLOGY:

    * ``"within_solve"`` (default, back-compat): one base-period chain
      spanning every base period in the solve.  ``rp_base_first`` /
      ``rp_base_last`` are singletons (head/tail of the single chain).
      Used by ``bind_within_solve_blended_weights`` (cyclic closure
      across the entire chain) and ``bind_forward_only_blended_weights``
      (no closure).
    * ``"within_period"``: each FlexTool period has its own independent
      chain that closes within itself.  ``per_period_inputs`` carries
      one entry per RP-bearing FlexTool period;
      ``rp_base_chain`` accumulates only WITHIN-period predecessor
      edges, and ``rp_base_first`` / ``rp_base_last`` carry one row per
      period plus a ``period`` column so the downstream cyclic-closure
      constraint (``model.py``'s ``rp_inter_period_cyclic``) can pair
      each period's first base against its OWN period's last base.

    ``per_period_inputs`` is required when ``variant == "within_period"``
    and ignored otherwise; the legacy single-period args
    (``rp_weights`` / ``timeset_duration_entries`` / ``period_name`` /
    ``timeline_steps``) drive the within_solve path.
    """
    if variant not in ("within_solve", "within_period"):
        raise ValueError(
            f"_compute_rp_frames: variant must be 'within_solve' or "
            f"'within_period', got {variant!r}."
        )
    if variant == "within_period":
        if not per_period_inputs:
            raise ValueError(
                "_compute_rp_frames: variant='within_period' requires a "
                "non-empty per_period_inputs list (one entry per "
                "RP-bearing FlexTool period)."
            )
        # Per-period assembly: each entry yields a chain segment fully
        # contained within its FlexTool period.  Cross-period edges are
        # omitted by construction (we never concatenate cross-segment
        # predecessor edges).
        wp_chain_rows: list[tuple[Any, ...]] = []
        wp_first_rows: list[tuple[Any, ...]] = []
        wp_last_rows: list[tuple[Any, ...]] = []
        wp_block_first_rows: list[tuple[Any, ...]] = []
        wp_block_last_rows: list[tuple[Any, ...]] = []
        wp_block_sl_rows: list[tuple[Any, ...]] = []
        wp_base_period_rows: list[tuple[Any, ...]] = []
        wp_rep_period_rows: list[tuple[Any, ...]] = []
        wp_weights_rows: list[tuple[Any, ...]] = []
        wp_cost_rows: list[tuple[Any, ...]] = []
        for (p_name, p_rp_w, p_ts_dur, p_tl_steps) in per_period_inputs:
            p_sub = _compute_rp_frames(
                p_rp_w, p_ts_dur, p_name, p_tl_steps,
            )
            # rp_base_chain: rename to add the owning period column.
            sub_chain = p_sub["rp_base_chain.csv"]
            for row in sub_chain.iter_rows():
                wp_chain_rows.append((row[0], row[1], p_name))
            # rp_base_first / rp_base_last: tag with period.
            sub_first = p_sub["rp_base_first.csv"]
            for row in sub_first.iter_rows():
                wp_first_rows.append((row[0], p_name))
            sub_last = p_sub["rp_base_last.csv"]
            for row in sub_last.iter_rows():
                wp_last_rows.append((row[0], p_name))
            # Pass through unchanged.
            wp_block_first_rows.extend(p_sub["rp_block_first.csv"].iter_rows())
            wp_block_last_rows.extend(p_sub["rp_block_last.csv"].iter_rows())
            wp_block_sl_rows.extend(
                p_sub["rp_block_start_last.csv"].iter_rows())
            wp_base_period_rows.extend(
                p_sub["rp_base_period_set.csv"].iter_rows())
            wp_rep_period_rows.extend(
                p_sub["rp_rep_period_set.csv"].iter_rows())
            wp_weights_rows.extend(p_sub["rp_weights.csv"].iter_rows())
            wp_cost_rows.extend(p_sub["rp_cost_weight.csv"].iter_rows())
        out: dict[str, pl.DataFrame] = {}
        out["rp_weights.csv"] = _to_utf8_frame(
            ("base_start", "rep_start", "weight"), wp_weights_rows,
        )
        # ``period`` column is the period that owns this chain edge /
        # endpoint; consumed by the FlexData loader and propagated as
        # ``d`` so ``model.py``'s ``rp_inter_period_cyclic`` can pair
        # endpoints per period.
        out["rp_base_chain.csv"] = _to_utf8_frame(
            ("base_start", "prev_base_start", "period"), wp_chain_rows,
        )
        out["rp_base_first.csv"] = _to_utf8_frame(
            ("base_start", "period"), wp_first_rows,
        )
        out["rp_base_last.csv"] = _to_utf8_frame(
            ("base_start", "period"), wp_last_rows,
        )
        out["rp_block_first.csv"] = _to_utf8_frame(
            ("period", "step"), wp_block_first_rows,
        )
        out["rp_block_last.csv"] = _to_utf8_frame(
            ("period", "step"), wp_block_last_rows,
        )
        out["rp_block_start_last.csv"] = _to_utf8_frame(
            ("rep_start", "last_step"), wp_block_sl_rows,
        )
        out["rp_base_period_set.csv"] = _to_utf8_frame(
            ("period",), wp_base_period_rows,
        )
        out["rp_rep_period_set.csv"] = _to_utf8_frame(
            ("period",), wp_rep_period_rows,
        )
        out["rp_cost_weight.csv"] = _to_utf8_frame(
            ("period", "time", "weight"), wp_cost_rows,
        )
        return out

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

    base_starts = sorted(
        rp_weights.keys(),
        key=lambda s: step_to_idx.get(s, len(timeline_steps)),
    )
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

    # 6. rp_cost_weight.csv — normalised per-timestep weight.
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
    out["rp_cost_weight.csv"] = _to_utf8_frame(
        ("period", "time", "weight"), cost_rows,
    )

    return out


def derive_rp_weights(
    rp_weights: dict[str, dict[str, float]],
    timeset_duration_entries: list[tuple[str, float]],
    period_name: str,
    timeline_steps: list[str],
) -> pl.DataFrame:
    """Build the ``rp_weights`` frame (base_start, rep_start, weight)."""
    return _compute_rp_frames(
        rp_weights, timeset_duration_entries, period_name, timeline_steps,
    )["rp_weights.csv"]


# Map from the basename keys produced by :func:`_compute_rp_frames`
# to the canonical Provider keys (no ``.csv`` suffix, per the Phase 3b
# convention in :mod:`flextool.engine_polars._provider_keys`).  Used by
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
    "rp_cost_weight.csv": "solve_data/rp_cost_weight",
}


def emit_rp_data(
    rp_weights: dict[str, dict[str, float]],
    timeset_duration_entries: list[tuple[str, float]],
    period_name: str,
    timeline_steps: list[str],
    *, provider,
    variant: str = "within_solve",
    per_period_inputs: (
        "list[tuple[str, dict[str, dict[str, float]], "
        "list[tuple[str, float]], list[str]]] | None"
    ) = None,
) -> None:
    """Emit ``rp_data`` to the Provider.

    Phase E added ``variant`` + ``per_period_inputs`` (see
    :func:`_compute_rp_frames` for semantics).  The caller in
    :mod:`flextool.engine_polars._native_run_model` selects the variant
    based on the binding-method mix of nodes in this solve; mixed
    within_solve/within_period in a single solve is rejected upstream
    with ``FlexToolConfigError`` because the per-solve RP frame family
    can only carry one chain topology.
    """
    frames = _compute_rp_frames(
        rp_weights, timeset_duration_entries, period_name, timeline_steps,
        variant=variant, per_period_inputs=per_period_inputs,
    )
    for basename, df in frames.items():
        _emit(provider, _RP_BASENAME_TO_PROVIDER_KEY[basename], df)


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


def emit_timeset_cost_weight(
    active_time_list: dict[str, list],
    timesets_used_by_solve: list[tuple[str, str]],
    timeset_weights: dict[str, dict[str, float]],
    *, provider,
) -> bool:
    """Emit ``timeset_cost_weight`` to the Provider."""
    rows, any_written = _compute_timeset_cost_weight_rows(
        active_time_list, timesets_used_by_solve, timeset_weights,
    )
    if not any_written:
        return False
    _emit(provider, "solve_data/rp_cost_weight.csv",
                   _to_utf8_frame(("period", "time", "weight"), rows))
    return True


_EMPTY_RP_HEADERS: dict[str, tuple[str, ...]] = {
    "rp_weights.csv": ("base_start", "rep_start", "weight"),
    "rp_base_chain.csv": ("base_start", "prev_base_start"),
    "rp_base_first.csv": ("base_start",),
    "rp_base_last.csv": ("base_start",),
    "rp_block_first.csv": ("period", "step"),
    "rp_block_last.csv": ("period", "step"),
    "rp_block_start_last.csv": ("rep_start", "last_step"),
    "rp_base_period_set.csv": ("period",),
    "rp_rep_period_set.csv": ("period",),
    "rp_cost_weight.csv": ("period", "time", "weight"),
}


def emit_empty_rp_data(
    *, provider) -> None:
    """Emit ``empty_rp_data`` to the Provider."""
    for filename, headers in _EMPTY_RP_HEADERS.items():
        _emit(provider, _RP_BASENAME_TO_PROVIDER_KEY[filename],
                       _empty_frame(headers))
