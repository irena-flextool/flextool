"""
Pure functions that write solve_data/ CSV files.
No state objects — all data is passed as explicit parameters.

Functions are grouped by output file type:
- Timeline writers: write_full_timelines, write_active_timelines, write_step_jump
- Period writers: write_years_represented, write_period_years, write_periods,
                  write_first_and_last_periods
- Solve status: write_solve_status, write_current_solve
- Timestep boundary: write_period_boundary_step (consolidates first/last steps)
- Realized/fix: write_last_realized_step, write_realized_dispatch,
                write_fix_storage_timesteps
- Stochastic: write_branch__period_relationship, write_all_branches,
              write_branch_weights_and_map
- Init files: write_empty_investment_file, write_empty_storage_fix_file,
              write_headers_for_empty_output_files
- Misc: write_timesets, write_hole_multiplier, write_delayed_durations,
        get_first_steps
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
# Timeline writers
# ---------------------------------------------------------------------------

def write_full_timelines(
    stochastic_timesteps: list[tuple[str, str]],
    period__timesets_in_this_solve: list[tuple[str, str]],
    timesets__timeline: dict[str, str],
    timelines: dict[str, list[tuple[str, ...]]],
    filename: str,
) -> None:
    """Write a list of timesteps as defined in timelines."""
    with open(filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['period', 'step'])
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
    """Write a list of timesteps as defined by the active timeline of the current solve."""
    if not complete:
        with open(filename, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['period', 'step', 'step_duration'])
            for period_name, period in timeline.items():
                for item in period:
                    writer.writerow([period_name, item.timestep, str(item.duration)])
    else:
        with open(filename, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['period', 'step', 'complete_step_duration'])
            for period_name, period in timeline.items():
                for item in period:
                    writer.writerow([period_name, item.timestep, str(item.duration)])


def write_step_jump(step_lengths: list[tuple[str, ...]], work_folder: Path | None = None) -> None:
    """Write step_jump.csv according to spec."""
    wf = work_folder if work_folder is not None else Path.cwd()
    headers = ("period", "time", "previous", "previous_within_timeset", "previous_period", "previous_within_solve", "jump")
    with open(wf / "solve_data/step_previous.csv", 'w', newline='') as stepfile:
        writer = csv.writer(stepfile, delimiter=',')
        writer.writerow(headers)
        writer.writerows(step_lengths)


def write_period_block(
    period_block_time: list[tuple],
    period_block_succ: list[tuple],
    work_folder: Path | None = None,
) -> None:
    """Write block-structure CSVs for the bind_intraperiod_blocks method."""
    wf = work_folder if work_folder is not None else Path.cwd()
    sd = wf / "solve_data"
    with open(sd / "period_block_time.csv", 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(("period", "block_first", "step"))
        writer.writerows(period_block_time)
    with open(sd / "period_block_succ.csv", 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
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
    """Write a list of periods with the number of years the period represents.

    Each period's ``years_represented`` value R is expanded into
    ``ceil(R)`` one-row-per-year entries of width 1, plus a trailing
    fractional row carrying the remainder when R is not an integer.
    Sub-year periods (0 < R < 1) emit a single row of width R.  Periods
    with R <= 0 are skipped.
    """
    with open(filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['period', 'years_from_solve', 'p_years_from_solve', 'p_years_represented'])
        year_count: float = 0
        for period__years in years_represented:
            total_represented = float(period__years[1])
            if total_represented <= 0:
                continue
            rows = math.ceil(total_represented)
            remaining = total_represented
            for i in range(rows):
                years_to_cover_within_year = min(1.0, remaining)
                writer.writerow([period__years[0], str(year_count), str(year_count),
                        str(years_to_cover_within_year)])
                for pd in period__branch:
                    if pd[0] in period__years[0] and pd[0] != pd[1]:
                        writer.writerow([pd[1], str(year_count), str(year_count),
                        str(years_to_cover_within_year)])
                year_count += years_to_cover_within_year
                remaining -= years_to_cover_within_year


def write_period_years(
    stochastic_branches: list[tuple[str, str]],
    years_represented: list[tuple[str, str]],
    filename: str,
) -> None:
    """Write a list of timesteps as defined by the active timeline of the current solve."""
    with open(filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['period', 'param'])
        year_count = 0
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
    """Write a list of periods based on the current solve."""
    with open(filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['period'])
        for period_tuple in periods_dict.get(solve, []):
            writer.writerow([period_tuple[1]])


def write_first_and_last_periods(
    active_time_list: dict[str, list[tuple[str, ...]]],
    period__timesets_in_this_solve: list[tuple[str, str]],
    period__branch_list: list[tuple[str, str]],
    work_folder: Path | None = None,
) -> None:
    """Write first and last periods (timewise) for the solve.

    Assumes that the periods are in right order in active_time_list,
    but gets the multiple branches as last.
    """
    wf = work_folder if work_folder is not None else Path.cwd()
    period_first_of_solve = list(active_time_list.keys())[0]
    period_last = []
    period_last.append(list(active_time_list.keys())[-1])
    time_step_last = active_time_list[period_last[0]][-1].timestep

    for period in active_time_list.keys():
        if active_time_list[period][-1].timestep == time_step_last and period != period_last[0]:
            period_last.append(period)

    with open(wf / "solve_data/period_last.csv", 'w', newline='') as realfile:
        writer = csv.writer(realfile)
        writer.writerow(["period"])
        for period in period_last:
            writer.writerow([period])

    period_first_of_solve_list = []
    for period__branch in period__branch_list:
        if period__branch[0] == period_first_of_solve:
            period_first_of_solve_list.append(period__branch[1])

    with open(wf / "solve_data/period_first_of_solve.csv", 'w', newline='') as realfile:
        writer = csv.writer(realfile)
        writer.writerow(["period"])
        for period in period_first_of_solve_list:
            writer.writerow([period])

    period_first = period__timesets_in_this_solve[0][0]
    period_first_list = []
    for period__branch in period__branch_list:
        if period__branch[0] == period_first:
            period_first_list.append(period__branch[1])

    with open(wf / "solve_data/period_first.csv", 'w', newline='') as realfile:
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
    """Write solve_first.csv with information if the current solve is the first to be run."""
    wf = work_folder if work_folder is not None else Path.cwd()
    if not nested:
        with open(wf / "solve_data/p_model.csv", 'w', newline='') as p_model_file:
            writer = csv.writer(p_model_file)
            writer.writerow(["modelParam", "p_model"])
            writer.writerow(["solveFirst", 1 if first_state else 0])
            writer.writerow(["solveLast", 1 if last_state else 0])
    else:
        with open(wf / "solve_data/p_nested_model.csv", 'w', newline='') as p_model_file:
            writer = csv.writer(p_model_file)
            writer.writerow(["modelParam", "p_nested_model"])
            writer.writerow(["solveFirst", 1 if first_state else 0])
            writer.writerow(["solveLast", 1 if last_state else 0])


def write_current_solve(solve: str, filename: str) -> None:
    """Write a file with the current solve name."""
    with open(filename, 'w', newline='') as solvefile:
        writer = csv.writer(solvefile)
        writer.writerow(["solve"])
        writer.writerow([solve])


# ---------------------------------------------------------------------------
# Timestep boundary (S04: consolidated from write_first_steps / write_last_steps)
# ---------------------------------------------------------------------------

def write_period_boundary_step(
    timeline: dict[str, list[tuple[str, ...]]],
    filename: str,
    *,
    last: bool = False,
) -> None:
    """Write the first or last step of each period.

    Args:
        timeline: mapping of period_name → list of (timestep, ...) tuples
        filename: output file path
        last: if True write the last step; if False write the first step
    """
    with open(filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['period', 'step'])
        for period_name, period in timeline.items():
            boundary = period[-1:] if last else period[:1]
            for item in boundary:
                writer.writerow([period_name, item.timestep])


def write_first_steps(
    timeline: dict[str, list[tuple[str, ...]]],
    filename: str,
) -> None:
    """Write the first step of each period (thin wrapper)."""
    write_period_boundary_step(timeline, filename, last=False)


def write_last_steps(
    timeline: dict[str, list[tuple[str, ...]]],
    filename: str,
) -> None:
    """Write the last step of each period (thin wrapper)."""
    write_period_boundary_step(timeline, filename, last=True)


def get_first_steps(
    steplists: dict[str, list[Any]],
) -> dict[str, tuple[Any, ...]]:
    """Get the first step of the current solve and the next solve in execution order."""
    solve_names = list(steplists.keys())
    starts: dict[str, tuple[Any, ...]] = dict()
    for index, name in enumerate(solve_names):
        if index == (len(solve_names) - 1):
            starts[name] = (steplists[name][0],)
        else:
            starts[name] = (steplists[solve_names[index]][0], steplists[solve_names[index + 1]][0])
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
    """Write the last step of the realized timeline."""
    with open(filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['period', 'step'])
        out = []
        has_realized_period = False
        for period_name, period in realized_timeline.items():
            if any(t[1] == period_name for t in realized_periods):
                last_realized_period = (period_name, period)
                has_realized_period = True
        if has_realized_period:
            for item in last_realized_period[1][-1:]:
                out = [period_name, item.timestep]
                writer.writerow(out)


def write_realized_dispatch(
    realized_time_list: dict[str, list[tuple[str, ...]]],
    solve: str,
    realized_periods: list[tuple[str, str]],
    work_folder: Path | None = None,
) -> None:
    """Write the timesteps to be realized for the dispatch decisions."""
    wf = work_folder if work_folder is not None else Path.cwd()
    with open(wf / "solve_data/realized_dispatch.csv", 'w', newline='') as realfile:
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
    """Write the timesteps where the storage is fixed for included solves."""
    wf = work_folder if work_folder is not None else Path.cwd()
    with open(wf / "solve_data/fix_storage_timesteps.csv", 'w', newline='') as realfile:
        writer = csv.writer(realfile)
        writer.writerow(["period", "step"])
        for period, active_time in active_time_list.items():
            if any(t[1] == period for t in fix_storage_periods):
                for i in active_time:
                    writer.writerow([period, i.timestep])


# ---------------------------------------------------------------------------
# Stochastic / branch writers
# ---------------------------------------------------------------------------

def write_branch__period_relationship(
    period__branch: list[tuple[str, str]],
    filename: str,
) -> None:
    """Write the period_branch relationship."""
    with open(filename, 'w', newline='') as realfile:
        writer = csv.writer(realfile)
        writer.writerow(["period", "branch"])
        for row in period__branch:
            writer.writerow([row[0], row[1]])


def write_all_branches(
    period__branch_list: dict[str, list[tuple[str, str]]],
    solve_branch__time_branch_list: list[tuple[str, str]],
    logger: logging.Logger,
    work_folder: Path | None = None,
    *,
    provider: "object | None" = None,
) -> None:
    """Write all branches in all solves.

    The seven ``pbt_*.csv`` source frames are looked up via
    :func:`_provider_open` (Step 2.5-G Phase A): the cascade-input
    Provider supplies them in-cascade, falling back to disk for
    off-cascade callers.  Mirrors
    :func:`flextool.engine_polars._writer_solve_writers.write_all_branches`.
    """
    from flextool.engine_polars._writer_provider_io import (
        _provider_key,
        _provider_open,
    )

    wf = work_folder if work_folder is not None else Path.cwd()
    branches = []
    for solve in period__branch_list:
        for row in period__branch_list[solve]:
            if row[1] not in branches:
                branches.append(row[1])
    with open(wf / "solve_data/branch_all.csv", 'w', newline='') as realfile:
        writer = csv.writer(realfile)
        writer.writerow(["branch"])
        for branch in branches:
            writer.writerow([branch])

    timeseries_names = [
        'pbt_node_inflow.csv',
        'pbt_node.csv',
        'pbt_process.csv',
        'pbt_profile.csv',
        'pbt_process_source.csv',
        'pbt_process_sink.csv',
        'pbt_reserve__upDown__group.csv']

    time_branches = []
    for filename in timeseries_names:
        in_path = wf / 'input' / filename
        handle = _provider_open(provider, _provider_key(in_path), in_path)
        if handle is None:
            # Neither Provider nor disk carries this pbt_* frame.
            # Legacy behaviour was a hard FileNotFoundError; the engine_
            # polars twin treats this as "no rows contributed" and
            # continues — match that contract.
            continue
        with handle as blk:
            filereader = csv.reader(blk, delimiter=',')
            headers = next(filereader)
            while True:
                try:
                    datain = next(filereader)
                    if datain[1] not in time_branches:
                        time_branches.append(datain[1])
                    if datain[1] == "":
                        logger.error("Empty branch name in timeseries: " + filename + " , check that there is no empty row at the end of the array")
                        sys.exit(-1)
                except StopIteration:
                    break

    for solve__branch in solve_branch__time_branch_list:
        if solve__branch[1] not in time_branches:
            time_branches.append(solve__branch[1])
    with open(wf / "solve_data/time_branch_all.csv", 'w', newline='') as realfile:
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
    """Write the weights and which branch is realized.

    Renamed from write_solve_branch__time_branch_list_and_weight (S09).
    """
    wf = work_folder if work_folder is not None else Path.cwd()
    time_branch_weight: dict[str, Any] = defaultdict()
    if branch_start_time is not None:
        for row in stochastic_branches[complete_solve]:
            if branch_start_time[0] == row[0] and branch_start_time[1] == row[2]:
                time_branch_weight[row[1]] = row[4]

    with open(wf / "solve_data/solve_branch_weight.csv", 'w', newline='') as realfile:
        writer = csv.writer(realfile)
        writer.writerow(["branch", "p_branch_weight_input"])
        for solve_branch__time_branch in solve_branch__time_branch_list:
            if (solve_branch__time_branch[0], solve_branch__time_branch[0]) in period__branch_lists:
                writer.writerow([solve_branch__time_branch[0], '1.0'])
            elif solve_branch__time_branch[1] in time_branch_weight.keys() and solve_branch__time_branch[0] in active_time_list.keys():
                writer.writerow([solve_branch__time_branch[0], str(time_branch_weight[solve_branch__time_branch[1]])])

    with open(wf / "solve_data/solve_branch__time_branch.csv", 'w', newline='') as realfile:
        writer = csv.writer(realfile)
        writer.writerow(["period", "branch"])
        for solve_branch__time_branch in solve_branch__time_branch_list:
            writer.writerow([solve_branch__time_branch[0], solve_branch__time_branch[1]])


# ---------------------------------------------------------------------------
# Init / empty files
# ---------------------------------------------------------------------------

def write_empty_investment_file(work_folder: Path | None = None) -> None:
    """Write empty p_entity_invested.csv for the first solve."""
    wf = work_folder if work_folder is not None else Path.cwd()
    with open(wf / "solve_data/p_entity_invested.csv", 'w', newline='') as firstfile:
        writer = csv.writer(firstfile)
        writer.writerow(["entity", "p_entity_invested"])
    with open(wf / "solve_data/p_entity_divested.csv", 'w', newline='') as firstfile:
        writer = csv.writer(firstfile)
        writer.writerow(["entity", "p_entity_divested"])
    with open(wf / "solve_data/p_entity_period_existing_capacity.csv", 'w', newline='') as firstfile:
        writer = csv.writer(firstfile)
        writer.writerow(["entity", "period", "p_entity_period_existing_capacity", "p_entity_period_invested_capacity"])


def write_empty_cumulative_files(work_folder: Path | None = None) -> None:
    """Seed header-only rolling accumulator files for the first solve.

    The mod reads three accumulator CSVs at the start of every solve via
    ``table data IN 'CSV' ...`` blocks:

    * ``solve_data/ladder_cum_realized_mwh.csv`` — ladder tier MWh
    * ``solve_data/ladder_cum_sim_hours.csv`` — per-period realized hours
    * ``solve_data/co2_cum_realized_tonnes.csv`` — CO2 cap realized tonnes

    On the very first roll there is no prior solve to emit them, so we
    write header-only files here — the CSV readers load zero rows, the
    mod's default 0 for every accumulator param makes ``f_d_k[d]``
    collapse to ``horizon_hours / (share * 8760)`` (= 1.0 on a full
    single solve) and every ``cum_realized = 0``, so the rolling caps
    reduce to their pre-refactor form bit-for-bit on a single-period
    single solve.  Later rolls overwrite these via
    :func:`flextool.process_outputs.cumulative_handoffs.write_ladder_rolling_accumulators`
    and
    :func:`flextool.process_outputs.cumulative_handoffs.write_co2_rolling_accumulators`.
    """
    wf = work_folder if work_folder is not None else Path.cwd()
    (wf / "solve_data").mkdir(exist_ok=True)
    with open(wf / "solve_data/ladder_cum_realized_mwh.csv", 'w', newline='') as f:
        csv.writer(f).writerow(
            ["commodity", "tier", "period", "p_ladder_cum_realized_mwh"]
        )
    with open(wf / "solve_data/ladder_cum_sim_hours.csv", 'w', newline='') as f:
        csv.writer(f).writerow(["period", "p_ladder_cum_sim_hours"])
    with open(wf / "solve_data/co2_cum_realized_tonnes.csv", 'w', newline='') as f:
        csv.writer(f).writerow(["group", "period", "p_co2_cum_realized_tonnes"])


def write_empty_storage_fix_file(work_folder: Path | None = None) -> None:
    """Write empty storage fix files."""
    wf = work_folder if work_folder is not None else Path.cwd()
    with open(wf / "solve_data/fix_storage_price.csv", 'w', newline='') as firstfile:
        writer = csv.writer(firstfile)
        writer.writerow(["node", " period", " step", " ndt_fix_storage_price"])
    with open(wf / "solve_data/fix_storage_quantity.csv", 'w', newline='') as firstfile:
        writer = csv.writer(firstfile)
        writer.writerow(["node", " period", " step", " ndt_fix_storage_quantity"])
    with open(wf / "solve_data/fix_storage_usage.csv", 'w', newline='') as firstfile:
        writer = csv.writer(firstfile)
        writer.writerow(["node", " period", " step", " ndt_fix_storage_usage"])
    with open(wf / "solve_data/p_roll_continue_state.csv", 'w', newline='') as firstfile:
        writer = csv.writer(firstfile)
        writer.writerow(["node", " p_roll_continue_state"])


def write_headers_for_empty_output_files(filename: str, header: str) -> None:
    """Write an empty output file with headers."""
    with open(filename, 'w', newline='') as firstfile:
        writer = csv.writer(firstfile)
        writer.writerow(header.split(','))


# ---------------------------------------------------------------------------
# Misc writers
# ---------------------------------------------------------------------------

def write_timesets(
    timesets_used_by_solves: dict[str, list[tuple[str, str]]],
    timeset__timeline: dict[str, str],
    work_folder: Path | None = None,
) -> None:
    """Write timesets_in_use.csv and timeset__timeline.csv."""
    wf = work_folder if work_folder is not None else Path.cwd()
    headers = ("solve", "period", "timesets")
    with open(wf / "input/timesets_in_use.csv", 'w', newline='') as timesetfile:
        writer = csv.writer(timesetfile, delimiter=',')
        writer.writerow(headers)
        for solve, period_timeset_list in timesets_used_by_solves.items():
            for period, timeset in period_timeset_list:
                writer.writerow((solve, period, timeset))

    headers = ("timesets", "timeline")
    with open(wf / "input/timesets__timeline.csv", 'w', newline='') as timesetfile:
        writer = csv.writer(timesetfile, delimiter=',')
        writer.writerow(headers)
        for timeset, timeline in timeset__timeline.items():
            writer.writerow((timeset, timeline))


def write_hole_multiplier(
    solve: str,
    hole_multipliers: dict[str, str],
    filename: str,
) -> None:
    """Write solve hole multiplier."""
    with open(filename, 'w', newline='') as holefile:
        writer = csv.writer(holefile)
        writer.writerow(["solve", "p_hole_multiplier"])
        if hole_multipliers[solve]:
            writer.writerow([solve, hole_multipliers[solve]])


def write_p_use_row_scaling(
    solve: str,
    use_row_scaling: dict[str, str],
    filename: str,
) -> None:
    """Write the Agent-5 row-scaling opt-in flag as an integer 0/1.

    Args:
        solve: current solve name.
        use_row_scaling: solve → "yes"/"no" from the DB (missing → off).
        filename: destination CSV path (usually
            ``solve_data/p_use_row_scaling.csv``).

    File layout (read by ``flextool.mod`` with
    ``table data IN 'CSV' ... : [solve], p_use_row_scaling``)::

        solve,p_use_row_scaling
        <solve>,0|1

    Default is 0 (off) unless the user explicitly sets the parameter to
    ``"yes"`` on the solve entity.  Writes the row unconditionally so
    AMPL always finds the current solve's value — the default path
    through the model is value = 0, identical to pre-Agent-5.
    """
    value_str = use_row_scaling.get(solve, "no") if isinstance(use_row_scaling, dict) else "no"
    flag = 1 if str(value_str).strip().lower() == "yes" else 0
    # Agent 9 test hook: FLEXTOOL_FORCE_ROW_SCALING=1 forces flag=1 for all
    # solves, bypassing both DB user-setting and ScaleAnalyzer.  Used by
    # the Agent 9 benchmark harness to validate Mode B un-scaling; no
    # effect in production unless the env var is set.
    import os as _os
    if _os.environ.get("FLEXTOOL_FORCE_ROW_SCALING", "").strip().lower() in ("1", "yes", "true", "on"):
        flag = 1
    with open(filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["solve", "p_use_row_scaling"])
        writer.writerow([solve, flag])


def write_scale_the_objective(solve_data_dir: Path | str, value: float) -> Path:
    """Emit ``solve_data/scale_the_objective.csv`` (Agent 12, Agent 21).

    Writes the global objective scalar coming from the Agent-8 scaling
    analyser (:class:`flextool.flextoolrunner.scaling.ScaleTable`) into
    a single-row CSV that ``flextool.mod`` reads via ``table data IN``
    into ``scale_the_objective``.  The .mod declares the param with a
    ``default 1e-6`` fallback equivalent to the old
    ``flextool_base.dat`` hardcoding.

    Layout::

        key,value
        v,<scale_the_objective>

    The ``key`` column is present only because GMPL ``table data IN``
    requires a keyed read — the value is pulled with
    ``sum{k in _scale_obj_keys} _scale_obj_from_csv[k]`` on a
    single-row file.

    Agent 21: this writer emits the analyser's recommendation only when
    the caller has opted into auto-scale mode.  The gating lives in the
    orchestrator (:mod:`flextool.flextoolrunner.orchestration`); callers
    of this function unconditionally emit the value.  See
    :func:`write_scale_the_objective_header_only` for the default-mode
    fallback.
    """
    sd = Path(solve_data_dir)
    sd.mkdir(parents=True, exist_ok=True)
    path = sd / "scale_the_objective.csv"
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(["key", "value"])
        writer.writerow(["v", f"{float(value):.17g}"])
    return path


def write_scale_the_state(solve_data_dir: Path | str, value: float) -> Path:
    """Emit ``solve_data/scale_the_state.csv`` (Agent 12, Agent 21).

    Companion to :func:`write_scale_the_objective`.  ``scale_the_state``
    is currently fixed at ``1.0`` — the field is reserved for future
    analyser tuning.  Layout / read convention identical to the
    objective CSV.  See :func:`write_scale_the_state_header_only` for
    the default-mode fallback.
    """
    sd = Path(solve_data_dir)
    sd.mkdir(parents=True, exist_ok=True)
    path = sd / "scale_the_state.csv"
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(["key", "value"])
        writer.writerow(["v", f"{float(value):.17g}"])
    return path


def write_scale_the_objective_header_only(solve_data_dir: Path | str) -> Path:
    """Emit header-only ``solve_data/scale_the_objective.csv`` (Agent 21).

    Default-mode counterpart to :func:`write_scale_the_objective`.  The
    CSV exists (``table data IN`` in ``flextool.mod`` won't fail on a
    missing file) but has no data rows, so ``_scale_obj_keys`` stays
    empty and the ``default 1e-6`` clause on the ``param
    scale_the_objective`` declaration takes over.  This restores the
    pre-Agent-12 behaviour for users who do not pass ``--auto-scale``
    — the analyser's per-solve power-of-10 recommendation is still
    serialised to ``scaling_analysis.json`` for diagnostics, but it
    does NOT get applied to the solve.

    Rationale (Agent 21): Agent 12's unconditional auto-apply tightens
    the HiGHS Cost range by up to ~4 decades on some models (rivendell
    S06 is the canonical failure case — scale picks 1e-10, Cost range
    drops from ``[6e-01, 1e+04]`` to ``[6e-05, 1e+00]``, HiGHS stalls
    on a tiny residual after reaching Pr=0,Du=0).  The analyser's
    power-of-10 rounding is too aggressive for models whose Matrix
    range is already wide — the user must opt in explicitly.
    """
    sd = Path(solve_data_dir)
    sd.mkdir(parents=True, exist_ok=True)
    path = sd / "scale_the_objective.csv"
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(["key", "value"])
    return path


def write_scale_the_state_header_only(solve_data_dir: Path | str) -> Path:
    """Emit header-only ``solve_data/scale_the_state.csv`` (Agent 21).

    Default-mode counterpart to :func:`write_scale_the_state`.  Same
    rationale as :func:`write_scale_the_objective_header_only`: the CSV
    exists but has no data rows, so the ``default 1`` clause on
    ``param scale_the_state`` applies.
    """
    sd = Path(solve_data_dir)
    sd.mkdir(parents=True, exist_ok=True)
    path = sd / "scale_the_state.csv"
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(["key", "value"])
    return path


def write_delayed_durations(
    active_time_list: dict[str, list[tuple[str, ...]]],
    solve: str,
    delay_durations: dict[str, Any],
    work_folder: Path | None = None,
) -> None:
    """Write delay duration data for the solve."""
    wf = work_folder if work_folder is not None else Path.cwd()
    delay_duration_set: set[str] = set()
    for (entity, dur) in delay_durations.items():
        if isinstance(dur, list):
            for delay_duration in dur:
                delay_duration_set.add(str(delay_duration[0]))
        else:
            delay_duration_set.add(str(dur))
    with open(wf / "solve_data/delay_duration.csv", 'w', newline='') as realfile:
        writer = csv.writer(realfile)
        writer.writerow(["delay_duration"])
        for delay_duration in delay_duration_set:
            writer.writerow([str(delay_duration)])
    with open(wf / "solve_data/dtt__delay_duration.csv", 'w', newline='') as realfile:
        writer = csv.writer(realfile)
        writer.writerow(["period", "time_source", "time_sink", "delay_duration"])
        for period_name, time_steps in active_time_list.items():
            for k, time_step in enumerate(time_steps):
                for delay_duration in delay_duration_set:
                    if k + int(float(delay_duration)) < len(time_steps):
                        writer.writerow([period_name, time_step.timestep, time_steps[k + int(float(delay_duration))].timestep, str(delay_duration)])
                    elif k + int(float(delay_duration)) >= len(time_steps):
                        writer.writerow([period_name, time_step.timestep, time_steps[k - len(time_steps) + int(float(delay_duration))].timestep, str(delay_duration)])


# ---------------------------------------------------------------------------
# Representative period CSV writers
# ---------------------------------------------------------------------------

def write_rp_data(
    rp_weights: dict[str, dict[str, float]],
    timeset_duration_entries: list[tuple[str, float]],
    period_name: str,
    work_folder: Path | None = None,
) -> None:
    """Write all representative period CSV files for the GMPL solver.

    Args:
        rp_weights: {base_start: {rep_start: weight}} — the full weight matrix.
        timeset_duration_entries: [(start_step, count), ...] for the RP timeset.
        period_name: The FlexTool period name (e.g. 'p2025').
        work_folder: Working directory for output.
    """
    wf = work_folder if work_folder is not None else Path.cwd()
    sd = wf / "solve_data"

    # Determine RP block boundaries from timeset_duration
    rp_starts: list[str] = []
    rp_lasts: list[str] = []
    for start_step, count in timeset_duration_entries:
        start_step = str(start_step)
        rp_starts.append(start_step)
        # Last timestep: compute from start + count - 1
        # We need the actual timestep name. For t-indexed names, increment the index.
        start_idx = int(start_step[1:])  # e.g. 't0001' -> 1
        last_idx = start_idx + int(float(count)) - 1
        last_step = f"t{last_idx:04d}"
        rp_lasts.append(last_step)

    # Base period starts (sorted chronologically)
    base_starts = sorted(rp_weights.keys(), key=lambda s: int(s[1:]))
    n_base = len(base_starts)
    n_rp = len(rp_starts)

    # 1. rp_weights.csv
    with open(sd / "rp_weights.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["base_start", "rep_start", "weight"])
        for base in base_starts:
            for rep, weight in rp_weights[base].items():
                if weight > 1e-10:
                    writer.writerow([base, rep, weight])

    # 2. rp_base_chain.csv (chronological, excludes first)
    with open(sd / "rp_base_chain.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["base_start", "prev_base_start"])
        for i in range(1, n_base):
            writer.writerow([base_starts[i], base_starts[i - 1]])

    # 3. rp_base_first.csv / rp_base_last.csv
    with open(sd / "rp_base_first.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["base_start"])
        writer.writerow([base_starts[0]])

    with open(sd / "rp_base_last.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["base_start"])
        writer.writerow([base_starts[-1]])

    # 4. rp_block_first.csv / rp_block_last.csv
    with open(sd / "rp_block_first.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["period", "step"])
        for start in rp_starts:
            writer.writerow([period_name, start])

    with open(sd / "rp_block_last.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["period", "step"])
        for last in rp_lasts:
            writer.writerow([period_name, last])

    # 5. rp_block_start_last.csv (maps RP start to last step)
    with open(sd / "rp_block_start_last.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["rep_start", "last_step"])
        for start, last in zip(rp_starts, rp_lasts):
            writer.writerow([start, last])

    # 6. rp_cost_weight.csv — per-timestep cost weight
    # W_r = sum_d W[d,r] for each RP r. Normalized: w_r = W_r * n_rp / n_base
    w_r: dict[str, float] = {r: 0.0 for r in rp_starts}
    for base_weights in rp_weights.values():
        for rep, weight in base_weights.items():
            if rep in w_r:
                w_r[rep] += weight
    # Normalize so uniform weights give w_r = 1
    for rep in w_r:
        w_r[rep] = w_r[rep] * n_rp / n_base if n_base > 0 else 1.0

    with open(sd / "rp_cost_weight.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["period", "time", "weight"])
        for start, last in zip(rp_starts, rp_lasts):
            start_idx = int(start[1:])
            last_idx = int(last[1:])
            weight = w_r[start]
            for t_idx in range(start_idx, last_idx + 1):
                writer.writerow([period_name, f"t{t_idx:04d}", weight])


def write_timeset_cost_weight(
    active_time_list: dict[str, list],
    timesets_used_by_solve: list[tuple[str, str]],
    timeset_weights: dict[str, dict[str, float]],
    work_folder: Path | None = None,
) -> bool:
    """Write rp_cost_weight.csv from timeset_weights when present on any
    active timeset for the current solve (non-RP pathway).

    For each period whose active timeset has a timeset_weights map, the
    per-timestep weights are normalized to sum to 1 across that period's
    active steps, then scaled by n_active_steps so that a uniform input
    reproduces the default weight = 1 per step. Timesteps absent from the
    user map are treated as 0 before normalization.

    Returns True if the file was written, False if no timeset on the solve
    has timeset_weights defined.
    """
    wf = work_folder if work_folder is not None else Path.cwd()
    sd = wf / "solve_data"

    rows: list[tuple[str, str, float]] = []
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
            rows.append((period, step.timestep, w * scale))
        any_written = True

    if not any_written:
        return False

    with open(sd / "rp_cost_weight.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["period", "time", "weight"])
        for row in rows:
            writer.writerow([row[0], row[1], f"{row[2]:.10g}"])
    return True


def write_empty_rp_data(work_folder: Path | None = None) -> None:
    """Write empty RP CSV files (headers only) for non-RP models."""
    wf = work_folder if work_folder is not None else Path.cwd()
    sd = wf / "solve_data"
    empty_files = {
        "rp_weights.csv": ["base_start", "rep_start", "weight"],
        "rp_base_chain.csv": ["base_start", "prev_base_start"],
        "rp_base_first.csv": ["base_start"],
        "rp_base_last.csv": ["base_start"],
        "rp_block_first.csv": ["period", "step"],
        "rp_block_last.csv": ["period", "step"],
        "rp_block_start_last.csv": ["rep_start", "last_step"],
        "rp_cost_weight.csv": ["period", "time", "weight"],
    }
    for filename, headers in empty_files.items():
        with open(sd / filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
