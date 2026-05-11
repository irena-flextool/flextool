"""Writer-port Phase 2 (sub-dispatch 6) — first half of ``solve_writers.py``.

Native re-implementation of ~26 of the ~36 functions in
``flextool.flextoolrunner.solve_writers`` (legacy 948 LOC total, ~542
LOC ported here).  Two functional groups:

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
operations, no data derivation.  The legacy module uses
:func:`csv.writer` with ``newline=''`` which emits ``CRLF`` line
terminators on every platform; we use the same ``csv.writer`` here
for byte-identical parity (verified via
``filecmp.cmp(shallow=False)`` in
``tests/engine_polars/test_writer_port_phase1.py``).

Function signatures match the legacy module verbatim because
:mod:`._native_input_writer._native_leaf_set_override` monkey-patches
the legacy module's attributes by name; ``_native_run_model.py``
imports the legacy module once and dispatches via attribute access.

Out of scope for this dispatch (deferred to sub-dispatch 7): scaling
writers (``write_p_use_row_scaling``, ``write_scale_the_objective``
and its header-only / state companions), ``write_delayed_durations``,
and the representative-period writers (``write_rp_data``,
``write_timeset_cost_weight``, ``write_empty_rp_data``).
"""
from __future__ import annotations

import csv
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Group A — timeline writers
# ---------------------------------------------------------------------------


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
    with open(filename, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["period", "step"])
        for period__timeset in period__timesets_in_this_solve:
            for timeline in timelines:
                for timeset_in_timeline, tt in timesets__timeline.items():
                    if period__timeset[1] == timeset_in_timeline:
                        if timeline == tt:
                            for item in timelines[timeline]:
                                writer.writerow([period__timeset[0], item[0]])
        for step in stochastic_timesteps:
            writer.writerow([step[0], step[1]])


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
    header_dur = "complete_step_duration" if complete else "step_duration"
    with open(filename, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["period", "step", header_dur])
        for period_name, period in timeline.items():
            for item in period:
                writer.writerow([period_name, item.timestep, str(item.duration)])


def write_step_jump(
    step_lengths: list[tuple[str, ...]],
    work_folder: Path | None = None,
) -> None:
    """Emit ``solve_data/step_previous.csv``."""
    wf = work_folder if work_folder is not None else Path.cwd()
    headers = (
        "period", "time", "previous", "previous_within_timeset",
        "previous_period", "previous_within_solve", "jump",
    )
    with open(wf / "solve_data/step_previous.csv", "w", newline="") as stepfile:
        writer = csv.writer(stepfile, delimiter=",")
        writer.writerow(headers)
        writer.writerows(step_lengths)


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
    with open(sd / "period_block_time.csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(("period", "block_first", "step"))
        writer.writerows(period_block_time)
    with open(sd / "period_block_succ.csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(("period", "block_first", "block_first_next"))
        writer.writerows(period_block_succ)


# ---------------------------------------------------------------------------
# Period writers
# ---------------------------------------------------------------------------


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
    with open(filename, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow([
            "period", "years_from_solve",
            "p_years_from_solve", "p_years_represented",
        ])
        year_count: float = 0
        for period__years in years_represented:
            total_represented = float(period__years[1])
            if total_represented <= 0:
                continue
            rows = math.ceil(total_represented)
            remaining = total_represented
            for _ in range(rows):
                years_to_cover_within_year = min(1.0, remaining)
                writer.writerow([
                    period__years[0], str(year_count), str(year_count),
                    str(years_to_cover_within_year),
                ])
                for pd in period__branch:
                    if pd[0] in period__years[0] and pd[0] != pd[1]:
                        writer.writerow([
                            pd[1], str(year_count), str(year_count),
                            str(years_to_cover_within_year),
                        ])
                year_count += years_to_cover_within_year
                remaining -= years_to_cover_within_year


def write_period_years(
    stochastic_branches: list[tuple[str, str]],
    years_represented: list[tuple[str, str]],
    filename: str,
) -> None:
    """Emit ``period_with_history.csv`` / ``p_discount_years.csv`` —
    each period mapped to its cumulative ``year_count`` start.
    """
    with open(filename, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["period", "param"])
        year_count: float = 0
        for period__year in years_represented:
            writer.writerow([period__year[0], str(year_count)])
            for pd in stochastic_branches:
                if pd[0] in period__year[0] and pd[0] != pd[1]:
                    writer.writerow([pd[1], str(year_count)])
            year_count += float(period__year[1])


def write_periods(
    solve: str,
    periods_dict: dict[str, list[tuple[str, str]]],
    filename: str,
) -> None:
    """Emit a single-column ``period`` CSV for the given solve key."""
    with open(filename, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["period"])
        for period_tuple in periods_dict.get(solve, []):
            writer.writerow([period_tuple[1]])


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
    period_first_of_solve = list(active_time_list.keys())[0]
    period_last = [list(active_time_list.keys())[-1]]
    time_step_last = active_time_list[period_last[0]][-1].timestep

    for period in active_time_list.keys():
        if (
            active_time_list[period][-1].timestep == time_step_last
            and period != period_last[0]
        ):
            period_last.append(period)

    with open(wf / "solve_data/period_last.csv", "w", newline="") as realfile:
        writer = csv.writer(realfile)
        writer.writerow(["period"])
        for period in period_last:
            writer.writerow([period])

    period_first_of_solve_list = []
    for period__branch in period__branch_list:
        if period__branch[0] == period_first_of_solve:
            period_first_of_solve_list.append(period__branch[1])

    with open(
        wf / "solve_data/period_first_of_solve.csv", "w", newline="",
    ) as realfile:
        writer = csv.writer(realfile)
        writer.writerow(["period"])
        for period in period_first_of_solve_list:
            writer.writerow([period])

    period_first = period__timesets_in_this_solve[0][0]
    period_first_list = []
    for period__branch in period__branch_list:
        if period__branch[0] == period_first:
            period_first_list.append(period__branch[1])

    with open(wf / "solve_data/period_first.csv", "w", newline="") as realfile:
        writer = csv.writer(realfile)
        writer.writerow(["period"])
        for period in period_first_list:
            writer.writerow([period])


# ---------------------------------------------------------------------------
# Solve status
# ---------------------------------------------------------------------------


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
        param_col = "p_model"
    else:
        path = wf / "solve_data/p_nested_model.csv"
        param_col = "p_nested_model"
    with open(path, "w", newline="") as p_model_file:
        writer = csv.writer(p_model_file)
        writer.writerow(["modelParam", param_col])
        writer.writerow(["solveFirst", 1 if first_state else 0])
        writer.writerow(["solveLast", 1 if last_state else 0])


def write_current_solve(solve: str, filename: str) -> None:
    """Emit ``solve_current.csv`` — single-row solve name file."""
    with open(filename, "w", newline="") as solvefile:
        writer = csv.writer(solvefile)
        writer.writerow(["solve"])
        writer.writerow([solve])


# ---------------------------------------------------------------------------
# Timestep boundary  (S04 consolidation in legacy)
# ---------------------------------------------------------------------------


def write_period_boundary_step(
    timeline: dict[str, list[tuple[str, ...]]],
    filename: str,
    *,
    last: bool = False,
) -> None:
    """Emit one (period, step) row per period — the first or last
    timestep depending on ``last``.
    """
    with open(filename, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["period", "step"])
        for period_name, period in timeline.items():
            boundary = period[-1:] if last else period[:1]
            for item in boundary:
                writer.writerow([period_name, item.timestep])


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
    with open(filename, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["period", "step"])
        has_realized_period = False
        last_realized_period: tuple[str, list[tuple[str, ...]]] | None = None
        for period_name, period in realized_timeline.items():
            if any(t[1] == period_name for t in realized_periods):
                last_realized_period = (period_name, period)
                has_realized_period = True
        if has_realized_period and last_realized_period is not None:
            for item in last_realized_period[1][-1:]:
                writer.writerow([last_realized_period[0], item.timestep])


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
    with open(
        wf / "solve_data/realized_dispatch.csv", "w", newline="",
    ) as realfile:
        writer = csv.writer(realfile)
        writer.writerow(["period", "step"])
        for period, realized_time in realized_time_list.items():
            if any(t[1] == period for t in realized_periods):
                for i in realized_time:
                    writer.writerow([period, i.timestep])


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
    with open(
        wf / "solve_data/fix_storage_timesteps.csv", "w", newline="",
    ) as realfile:
        writer = csv.writer(realfile)
        writer.writerow(["period", "step"])
        for period, active_time in active_time_list.items():
            if any(t[1] == period for t in fix_storage_periods):
                for i in active_time:
                    writer.writerow([period, i.timestep])


# ---------------------------------------------------------------------------
# Group B — branch / empty / header writers
# ---------------------------------------------------------------------------


def write_branch__period_relationship(
    period__branch: list[tuple[str, str]],
    filename: str,
) -> None:
    """Emit ``period__branch.csv`` (2-col verbatim copy)."""
    with open(filename, "w", newline="") as realfile:
        writer = csv.writer(realfile)
        writer.writerow(["period", "branch"])
        for row in period__branch:
            writer.writerow([row[0], row[1]])


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
    branches: list[str] = []
    for solve in period__branch_list:
        for row in period__branch_list[solve]:
            if row[1] not in branches:
                branches.append(row[1])
    with open(wf / "solve_data/branch_all.csv", "w", newline="") as realfile:
        writer = csv.writer(realfile)
        writer.writerow(["branch"])
        for branch in branches:
            writer.writerow([branch])

    timeseries_names = [
        "pbt_node_inflow.csv",
        "pbt_node.csv",
        "pbt_process.csv",
        "pbt_profile.csv",
        "pbt_process_source.csv",
        "pbt_process_sink.csv",
        "pbt_reserve__upDown__group.csv",
    ]

    time_branches: list[str] = []
    for filename in timeseries_names:
        with open(wf / "input" / filename, "r", encoding="utf-8") as blk:
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
    with open(
        wf / "solve_data/time_branch_all.csv", "w", newline="",
    ) as realfile:
        writer = csv.writer(realfile)
        writer.writerow(["time_branch"])
        for time_branch in time_branches:
            writer.writerow([time_branch])


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
    time_branch_weight: dict[str, Any] = defaultdict()
    if branch_start_time is not None:
        for row in stochastic_branches[complete_solve]:
            if branch_start_time[0] == row[0] and branch_start_time[1] == row[2]:
                time_branch_weight[row[1]] = row[4]

    with open(
        wf / "solve_data/solve_branch_weight.csv", "w", newline="",
    ) as realfile:
        writer = csv.writer(realfile)
        writer.writerow(["branch", "p_branch_weight_input"])
        for solve_branch__time_branch in solve_branch__time_branch_list:
            if (
                (solve_branch__time_branch[0], solve_branch__time_branch[0])
                in period__branch_lists
            ):
                writer.writerow([solve_branch__time_branch[0], "1.0"])
            elif (
                solve_branch__time_branch[1] in time_branch_weight.keys()
                and solve_branch__time_branch[0] in active_time_list.keys()
            ):
                writer.writerow([
                    solve_branch__time_branch[0],
                    str(time_branch_weight[solve_branch__time_branch[1]]),
                ])

    with open(
        wf / "solve_data/solve_branch__time_branch.csv", "w", newline="",
    ) as realfile:
        writer = csv.writer(realfile)
        writer.writerow(["period", "branch"])
        for solve_branch__time_branch in solve_branch__time_branch_list:
            writer.writerow([
                solve_branch__time_branch[0], solve_branch__time_branch[1],
            ])


# ---------------------------------------------------------------------------
# Init / empty files
# ---------------------------------------------------------------------------


def write_empty_investment_file(work_folder: Path | None = None) -> None:
    """Seed three header-only investment CSVs for the first solve."""
    wf = work_folder if work_folder is not None else Path.cwd()
    with open(
        wf / "solve_data/p_entity_invested.csv", "w", newline="",
    ) as firstfile:
        writer = csv.writer(firstfile)
        writer.writerow(["entity", "p_entity_invested"])
    with open(
        wf / "solve_data/p_entity_divested.csv", "w", newline="",
    ) as firstfile:
        writer = csv.writer(firstfile)
        writer.writerow(["entity", "p_entity_divested"])
    with open(
        wf / "solve_data/p_entity_period_existing_capacity.csv",
        "w", newline="",
    ) as firstfile:
        writer = csv.writer(firstfile)
        writer.writerow([
            "entity", "period",
            "p_entity_period_existing_capacity",
            "p_entity_period_invested_capacity",
        ])


def write_empty_cumulative_files(work_folder: Path | None = None) -> None:
    """Seed three header-only rolling-accumulator CSVs (mod default 0
    collapses every accumulator on a single-period single solve).
    """
    wf = work_folder if work_folder is not None else Path.cwd()
    (wf / "solve_data").mkdir(exist_ok=True)
    with open(
        wf / "solve_data/ladder_cum_realized_mwh.csv", "w", newline="",
    ) as f:
        csv.writer(f).writerow(
            ["commodity", "tier", "period", "p_ladder_cum_realized_mwh"]
        )
    with open(
        wf / "solve_data/ladder_cum_sim_hours.csv", "w", newline="",
    ) as f:
        csv.writer(f).writerow(["period", "p_ladder_cum_sim_hours"])
    with open(
        wf / "solve_data/co2_cum_realized_tonnes.csv", "w", newline="",
    ) as f:
        csv.writer(f).writerow(
            ["group", "period", "p_co2_cum_realized_tonnes"]
        )


def write_empty_storage_fix_file(work_folder: Path | None = None) -> None:
    """Seed four header-only fix-storage CSVs for the first solve.

    NOTE: legacy preserves the leading-space column names (e.g.
    ``" period"``) — kept verbatim for byte parity.
    """
    wf = work_folder if work_folder is not None else Path.cwd()
    with open(
        wf / "solve_data/fix_storage_price.csv", "w", newline="",
    ) as firstfile:
        writer = csv.writer(firstfile)
        writer.writerow(
            ["node", " period", " step", " ndt_fix_storage_price"]
        )
    with open(
        wf / "solve_data/fix_storage_quantity.csv", "w", newline="",
    ) as firstfile:
        writer = csv.writer(firstfile)
        writer.writerow(
            ["node", " period", " step", " ndt_fix_storage_quantity"]
        )
    with open(
        wf / "solve_data/fix_storage_usage.csv", "w", newline="",
    ) as firstfile:
        writer = csv.writer(firstfile)
        writer.writerow(
            ["node", " period", " step", " ndt_fix_storage_usage"]
        )
    with open(
        wf / "solve_data/p_roll_continue_state.csv", "w", newline="",
    ) as firstfile:
        writer = csv.writer(firstfile)
        writer.writerow(["node", " p_roll_continue_state"])


def write_headers_for_empty_output_files(filename: str, header: str) -> None:
    """Emit a header-only output CSV given comma-separated column names."""
    with open(filename, "w", newline="") as firstfile:
        writer = csv.writer(firstfile)
        writer.writerow(header.split(","))


# ---------------------------------------------------------------------------
# Misc writers
# ---------------------------------------------------------------------------


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
    headers = ("solve", "period", "timesets")
    with open(
        wf / "input/timesets_in_use.csv", "w", newline="",
    ) as timesetfile:
        writer = csv.writer(timesetfile, delimiter=",")
        writer.writerow(headers)
        for solve, period_timeset_list in timesets_used_by_solves.items():
            for period, timeset in period_timeset_list:
                writer.writerow((solve, period, timeset))

    headers = ("timesets", "timeline")
    with open(
        wf / "input/timesets__timeline.csv", "w", newline="",
    ) as timesetfile:
        writer = csv.writer(timesetfile, delimiter=",")
        writer.writerow(headers)
        for timeset, timeline in timeset__timeline.items():
            writer.writerow((timeset, timeline))


def write_hole_multiplier(
    solve: str,
    hole_multipliers: dict[str, str],
    filename: str,
) -> None:
    """Emit ``solve_hole_multiplier.csv`` — single row when the
    multiplier is truthy, header-only otherwise.
    """
    with open(filename, "w", newline="") as holefile:
        writer = csv.writer(holefile)
        writer.writerow(["solve", "p_hole_multiplier"])
        if hole_multipliers[solve]:
            writer.writerow([solve, hole_multipliers[solve]])
