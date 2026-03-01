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
import sys
from collections import OrderedDict, defaultdict
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
    with open(filename, 'w') as outfile:
        outfile.write('period,step\n')
        for period__timeset in period__timesets_in_this_solve:
            for timeline in timelines:
                for timeset_in_timeline, tt in timesets__timeline.items():
                    if period__timeset[1] == timeset_in_timeline:
                        if timeline == tt:
                            for item in timelines[timeline]:
                                outfile.write(period__timeset[0] + ',' + item[0] + '\n')
        for step in stochastic_timesteps:
            outfile.write(step[0] + ',' + step[1] + '\n')


def write_active_timelines(
    timeline: dict[str, list[tuple[str, ...]]],
    filename: str,
    complete: bool = False,
) -> None:
    """Write a list of timesteps as defined by the active timeline of the current solve."""
    if not complete:
        with open(filename, 'w') as outfile:
            outfile.write('period,step,step_duration\n')
            for period_name, period in timeline.items():
                for item in period:
                    outfile.write(period_name + ',' + item[0] + ',' + str(item[2]) + '\n')
    else:
        with open(filename, 'w') as outfile:
            outfile.write('period,step,complete_step_duration\n')
            for period_name, period in timeline.items():
                for item in period:
                    outfile.write(period_name + ',' + item[0] + ',' + str(item[2]) + '\n')


def write_step_jump(step_lengths: list[tuple[str, ...]]) -> None:
    """Write step_jump.csv according to spec."""
    headers = ("period", "time", "previous", "previous_within_timeset", "previous_period", "previous_within_solve", "jump")
    with open("solve_data/step_previous.csv", 'w', newline='\n') as stepfile:
        writer = csv.writer(stepfile, delimiter=',')
        writer.writerow(headers)
        writer.writerows(step_lengths)


# ---------------------------------------------------------------------------
# Period writers
# ---------------------------------------------------------------------------

def write_years_represented(
    period__branch: list[tuple[str, str]],
    years_represented: list[tuple[str, str]],
    filename: str,
) -> None:
    """Write a list of periods with the number of years the period represents."""
    with open(filename, 'w') as outfile:
        outfile.write('period,years_from_solve,p_years_from_solve,p_years_represented\n')
        year_count = 0
        for period__years in years_represented:
            for i in range(int(max(1.0, float(period__years[1])))):
                years_to_cover_within_year = min(1, float(period__years[1]))
                outfile.write(period__years[0] + ',' + str(year_count) + ',' + str(year_count) + ','
                        + str(years_to_cover_within_year) + '\n')
                for pd in period__branch:
                    if pd[0] in period__years[0] and pd[0] != pd[1]:
                        outfile.write(pd[1]+ ',' + str(year_count) + ',' + str(year_count) + ','
                        + str(years_to_cover_within_year) + '\n')
                year_count = year_count + years_to_cover_within_year


def write_period_years(
    stochastic_branches: list[tuple[str, str]],
    years_represented: list[tuple[str, str]],
    filename: str,
) -> None:
    """Write a list of timesteps as defined by the active timeline of the current solve."""
    with open(filename, 'w') as outfile:
        outfile.write('period,param\n')
        year_count = 0
        for period__year in years_represented:
            outfile.write(period__year[0] + ',' + str(year_count) + '\n')
            for pd in stochastic_branches:
                if pd[0] in period__year[0] and pd[0] != pd[1]:
                    outfile.write(pd[1] + ',' + str(year_count) + '\n')
            year_count += float(period__year[1])


def write_periods(
    solve: str,
    periods_dict: dict[str, list[tuple[str, str]]],
    filename: str,
) -> None:
    """Write a list of periods based on the current solve."""
    with open(filename, 'w') as outfile:
        outfile.write('period\n')
        for period_tuple in periods_dict.get(solve, []):
            outfile.write(period_tuple[1] + '\n')


def write_first_and_last_periods(
    active_time_list: dict[str, list[tuple[str, ...]]],
    period__timesets_in_this_solve: list[tuple[str, str]],
    period__branch_list: list[tuple[str, str]],
) -> None:
    """Write first and last periods (timewise) for the solve.

    Assumes that the periods are in right order in active_time_list,
    but gets the multiple branches as last.
    """
    period_first_of_solve = list(active_time_list.keys())[0]
    period_last = []
    period_last.append(list(active_time_list.keys())[-1])
    time_step_last = active_time_list[period_last[0]][-1][0]

    for period in active_time_list.keys():
        if active_time_list[period][-1][0] == time_step_last and period != period_last[0]:
            period_last.append(period)

    with open("solve_data/period_last.csv", 'w') as realfile:
        realfile.write("period\n")
        for period in period_last:
            realfile.write(period + "\n")

    period_first_of_solve_list = []
    for period__branch in period__branch_list:
        if period__branch[0] == period_first_of_solve:
            period_first_of_solve_list.append(period__branch[1])

    with open("solve_data/period_first_of_solve.csv", 'w') as realfile:
        realfile.write("period\n")
        for period in period_first_of_solve_list:
            realfile.write(period + "\n")

    period_first = period__timesets_in_this_solve[0][0]
    period_first_list = []
    for period__branch in period__branch_list:
        if period__branch[0] == period_first:
            period_first_list.append(period__branch[1])

    with open("solve_data/period_first.csv", 'w') as realfile:
        realfile.write("period\n")
        for period in period_first_list:
            realfile.write(period + "\n")


# ---------------------------------------------------------------------------
# Solve status
# ---------------------------------------------------------------------------

def write_solve_status(
    first_state: bool,
    last_state: bool,
    nested: bool = False,
) -> None:
    """Write solve_first.csv with information if the current solve is the first to be run."""
    if not nested:
        with open("input/p_model.csv", 'w') as p_model_file:
            p_model_file.write("modelParam,p_model\n")
            if first_state:
                p_model_file.write("solveFirst,1\n")
            else:
                p_model_file.write("solveFirst,0\n")
            if last_state:
                p_model_file.write("solveLast,1\n")
            else:
                p_model_file.write("solveLast,0\n")
    else:
        with open("solve_data/p_nested_model.csv", 'w') as p_model_file:
            p_model_file.write("modelParam,p_nested_model\n")
            if first_state:
                p_model_file.write("solveFirst,1\n")
            else:
                p_model_file.write("solveFirst,0\n")
            if last_state:
                p_model_file.write("solveLast,1\n")
            else:
                p_model_file.write("solveLast,0\n")


def write_current_solve(solve: str, filename: str) -> None:
    """Write a file with the current solve name."""
    with open(filename, 'w') as solvefile:
        solvefile.write("solve\n")
        solvefile.write(solve + "\n")


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
    with open(filename, 'w') as outfile:
        outfile.write('period,step\n')
        for period_name, period in timeline.items():
            boundary = period[-1:] if last else period[:1]
            for item in boundary:
                outfile.write(period_name + ',' + item[0] + '\n')


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
) -> OrderedDict[str, tuple[Any, ...]]:
    """Get the first step of the current solve and the next solve in execution order."""
    solve_names = list(steplists.keys())
    starts: OrderedDict[str, tuple[Any, ...]] = OrderedDict()
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
    with open(filename, 'w') as outfile:
        outfile.write('period,step\n')
        out = []
        has_realized_period = False
        for period_name, period in realized_timeline.items():
            if any(t[1] == period_name for t in realized_periods):
                last_realized_period = (period_name, period)
                has_realized_period = True
        if has_realized_period:
            for item in last_realized_period[1][-1:]:
                out = [period_name, item[0]]
                outfile.write(out[0] + ',' + out[1] + '\n')


def write_realized_dispatch(
    realized_time_list: dict[str, list[tuple[str, ...]]],
    solve: str,
    realized_periods: list[tuple[str, str]],
) -> None:
    """Write the timesteps to be realized for the dispatch decisions."""
    with open("solve_data/realized_dispatch.csv", 'w') as realfile:
        realfile.write("period,step\n")
        for period, realized_time in realized_time_list.items():
            if any(t[1] == period for t in realized_periods):
                for i in realized_time:
                    realfile.write(period + "," + i[0] + "\n")


def write_fix_storage_timesteps(
    active_time_list: dict[str, list[tuple[str, ...]]],
    solve: str,
    fix_storage_periods: list[tuple[str, str]],
) -> None:
    """Write the timesteps where the storage is fixed for included solves."""
    with open("solve_data/fix_storage_timesteps.csv", 'w') as realfile:
        realfile.write("period,step\n")
        for period, active_time in active_time_list.items():
            if any(t[1] == period for t in fix_storage_periods):
                for i in active_time:
                    realfile.write(period + "," + i[0] + "\n")


# ---------------------------------------------------------------------------
# Stochastic / branch writers
# ---------------------------------------------------------------------------

def write_branch__period_relationship(
    period__branch: list[tuple[str, str]],
    filename: str,
) -> None:
    """Write the period_branch relationship."""
    with open(filename, 'w') as realfile:
        realfile.write("period,branch\n")
        for row in period__branch:
            realfile.write(row[0] + "," + row[1] + "\n")


def write_all_branches(
    period__branch_list: dict[str, list[tuple[str, str]]],
    solve_branch__time_branch_list: list[tuple[str, str]],
    logger: logging.Logger,
) -> None:
    """Write all branches in all solves."""
    branches = []
    for solve in period__branch_list:
        for row in period__branch_list[solve]:
            if row[1] not in branches:
                branches.append(row[1])
    with open("solve_data/branch_all.csv", 'w') as realfile:
        realfile.write("branch\n")
        for branch in branches:
            realfile.write(branch + "\n")

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
        with open('input/' + filename, 'r') as blk:
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
    with open("solve_data/time_branch_all.csv", 'w') as realfile:
        realfile.write("time_branch\n")
        for time_branch in time_branches:
            realfile.write(time_branch + "\n")


def write_branch_weights_and_map(
    complete_solve: str,
    active_time_list: dict[str, list[tuple[str, ...]]],
    solve_branch__time_branch_list: list[tuple[str, str]],
    branch_start_time: tuple[str, str] | None,
    period__branch_lists: list[tuple[str, str]],
    stochastic_branches: dict[str, list[Any]],
) -> None:
    """Write the weights and which branch is realized.

    Renamed from write_solve_branch__time_branch_list_and_weight (S09).
    """
    time_branch_weight: dict[str, Any] = defaultdict()
    if branch_start_time is not None:
        for row in stochastic_branches[complete_solve]:
            if branch_start_time[0] == row[0] and branch_start_time[1] == row[2]:
                time_branch_weight[row[1]] = row[4]

    with open("solve_data/solve_branch_weight.csv", 'w') as realfile:
        realfile.write("branch,p_branch_weight_input\n")
        for solve_branch__time_branch in solve_branch__time_branch_list:
            if (solve_branch__time_branch[0], solve_branch__time_branch[0]) in period__branch_lists:
                realfile.write(solve_branch__time_branch[0] + "," + '1.0' + "\n")
            elif solve_branch__time_branch[1] in time_branch_weight.keys() and solve_branch__time_branch[0] in active_time_list.keys():
                realfile.write(solve_branch__time_branch[0] + "," + str(time_branch_weight[solve_branch__time_branch[1]]) + "\n")

    with open("solve_data/solve_branch__time_branch.csv", 'w') as realfile:
        realfile.write("period,branch\n")
        for solve_branch__time_branch in solve_branch__time_branch_list:
            realfile.write(solve_branch__time_branch[0] + "," + solve_branch__time_branch[1] + "\n")


# ---------------------------------------------------------------------------
# Init / empty files
# ---------------------------------------------------------------------------

def write_empty_investment_file() -> None:
    """Write empty p_entity_invested.csv for the first solve."""
    with open("solve_data/p_entity_invested.csv", 'w') as firstfile:
        firstfile.write("entity,p_entity_invested\n")
    with open("solve_data/p_entity_divested.csv", 'w') as firstfile:
        firstfile.write("entity,p_entity_divested\n")
    with open("solve_data/p_entity_period_existing_capacity.csv", 'w') as firstfile:
        firstfile.write("entity,period,p_entity_period_existing_capacity,p_entity_period_invested_capacity\n")


def write_empty_storage_fix_file() -> None:
    """Write empty storage fix files."""
    with open("solve_data/fix_storage_price.csv", 'w') as firstfile:
        firstfile.write("node, period, step, ndt_fix_storage_price\n")
    with open("solve_data/fix_storage_quantity.csv", 'w') as firstfile:
        firstfile.write("node, period, step, ndt_fix_storage_quantity\n")
    with open("solve_data/fix_storage_usage.csv", 'w') as firstfile:
        firstfile.write("node, period, step, ndt_fix_storage_usage\n")
    with open("solve_data/p_roll_continue_state.csv", 'w') as firstfile:
        firstfile.write("node, p_roll_continue_state\n")


def write_headers_for_empty_output_files(filename: str, header: str) -> None:
    """Write an empty output file with headers."""
    with open(filename, 'w') as firstfile:
        firstfile.write(header + "\n")


# ---------------------------------------------------------------------------
# Misc writers
# ---------------------------------------------------------------------------

def write_timesets(
    timesets_used_by_solves: dict[str, list[tuple[str, str]]],
    timeset__timeline: dict[str, str],
) -> None:
    """Write timesets_in_use.csv and timeset__timeline.csv."""
    headers = ("solve", "period", "timesets")
    with open("input/timesets_in_use.csv", 'w', newline='\n') as timesetfile:
        writer = csv.writer(timesetfile, delimiter=',')
        writer.writerow(headers)
        for solve, period_timeset_list in timesets_used_by_solves.items():
            for period, timeset in period_timeset_list:
                writer.writerow((solve, period, timeset))

    headers = ("timesets", "timeline")
    with open("input/timesets__timeline.csv", 'w', newline='\n') as timesetfile:
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
    with open(filename, 'w') as holefile:
        holefile.write("solve,p_hole_multiplier\n")
        if hole_multipliers[solve]:
            holefile.write(solve + "," + hole_multipliers[solve] + "\n")


def write_delayed_durations(
    active_time_list: dict[str, list[tuple[str, ...]]],
    solve: str,
    delay_durations: dict[str, Any],
) -> None:
    """Write delay duration data for the solve."""
    delay_duration_set: set[str] = set()
    for (entity, dur) in delay_durations.items():
        if isinstance(dur, list):
            for delay_duration in dur:
                delay_duration_set.add(str(delay_duration[0]))
        else:
            delay_duration_set.add(str(dur))
    with open("solve_data/delay_duration.csv", 'w') as realfile:
        realfile.write("delay_duration\n")
        for delay_duration in delay_duration_set:
            realfile.write(str(delay_duration) + "\n")
    with open("solve_data/dtt__delay_duration.csv", 'w') as realfile:
        realfile.write("period,time_source,time_sink,delay_duration\n")
        for period_name, time_steps in active_time_list.items():
            for k, time_step in enumerate(time_steps):
                for delay_duration in delay_duration_set:
                    if k + int(float(delay_duration)) < len(time_steps):
                        row = ','.join([period_name, time_step[0], time_steps[k + int(float(delay_duration))][0], str(delay_duration)])
                        realfile.write(row + "\n")
                    elif k + int(float(delay_duration)) >= len(time_steps):
                        row = ','.join([period_name, time_step[0], time_steps[k - len(time_steps) + int(float(delay_duration))][0], str(delay_duration)])
                        realfile.write(row + "\n")
