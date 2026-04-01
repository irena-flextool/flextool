"""Minimum uptime/downtime precomputation.

Computes the backward-looking window sets for the Rajan-Takriti clique
inequality formulation. For each process with min_uptime/min_downtime,
and each timestep (d, t), the module walks backward through the timeline
accumulating step durations to determine which past timesteps fall within
the minimum time window.

Entry point: write_minimum_time_data(...)
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def write_minimum_time_data(
    active_time_list: dict[str, list[Any]],
    jump_list: list[tuple],
    process_min_uptime: dict[str, float],
    process_min_downtime: dict[str, float],
    work_folder: Path,
) -> None:
    """Write uptime/downtime lookback window CSVs for the current solve.

    For each process that has a positive min_uptime or min_downtime, this
    function walks backward through the timeline from every timestep and
    records which earlier timesteps fall within the minimum-time window.
    The results are written as CSV files consumed by the GMPL model.

    Args:
        active_time_list: Mapping from period name to list of ActiveTimeEntry.
        jump_list: List of 7-tuples from make_step_jump().
        process_min_uptime: Mapping from process name to min_uptime hours.
        process_min_downtime: Mapping from process name to min_downtime hours.
        work_folder: Path to the working directory containing solve_data/.
    """
    # Build step_duration_map: (period, timestep) -> duration in hours
    step_duration_map: dict[tuple[str, str], float] = {}
    ordered_steps: list[tuple[str, str]] = []
    for period, entries in active_time_list.items():
        for entry in entries:
            step_duration_map[(period, entry.timestep)] = float(entry.duration)
            ordered_steps.append((period, entry.timestep))

    # Build backward_map from jump_list
    # jump_list entries: (period, time, previous, prev_within_timeset,
    #                     prev_period, prev_within_solve, jump)
    backward_map: dict[tuple[str, str], tuple[str, str, int]] = {}
    for jl in jump_list:
        d, t = jl[0], jl[1]
        d_prev, t_prev = jl[4], jl[5]
        jump_val = jl[6]
        backward_map[(d, t)] = (d_prev, t_prev, jump_val)

    # Compute and write uptime lookback
    _write_lookback_csv(
        work_folder / "solve_data" / "uptime_lookback.csv",
        process_min_uptime,
        ordered_steps,
        step_duration_map,
        backward_map,
    )

    # Compute and write downtime lookback
    _write_lookback_csv(
        work_folder / "solve_data" / "downtime_lookback.csv",
        process_min_downtime,
        ordered_steps,
        step_duration_map,
        backward_map,
    )

    # Check if any lookback windows were truncated at the start of the solve
    if ordered_steps:
        first_step = ordered_steps[0]
        _warn_truncated_lookback(
            first_step, backward_map, step_duration_map,
            process_min_uptime, "min_uptime",
        )
        _warn_truncated_lookback(
            first_step, backward_map, step_duration_map,
            process_min_downtime, "min_downtime",
        )


def _warn_truncated_lookback(
    first_step: tuple[str, str],
    backward_map: dict[tuple[str, str], tuple[str, str, int]],
    step_duration_map: dict[tuple[str, str], float],
    process_min_times: dict[str, float],
    param_name: str,
) -> None:
    """Warn if the lookback from the first timestep is shorter than required."""
    for process, min_time in sorted(process_min_times.items()):
        if min_time <= 0:
            continue
        accumulated = 0.0
        pos = first_step
        while pos in backward_map:
            d_prev, t_prev, jump_val = backward_map[pos]
            if jump_val != 1:
                break
            prev_pos = (d_prev, t_prev)
            if prev_pos not in step_duration_map:
                break
            accumulated += step_duration_map[prev_pos]
            pos = prev_pos
        if accumulated < min_time:
            logger.warning(
                "Process '%s' has %s=%.1f hours but the solve "
                "window lookback from the first timestep covers only "
                "%.1f hours. Consider increasing rolling window overlap.",
                process, param_name, min_time, accumulated,
            )


def _write_lookback_csv(
    filepath: Path,
    process_min_times: dict[str, float],
    ordered_steps: list[tuple[str, str]],
    step_duration_map: dict[tuple[str, str], float],
    backward_map: dict[tuple[str, str], tuple[str, str, int]],
) -> None:
    """Write a lookback window CSV for either uptime or downtime.

    For each process with a positive minimum time and each timestep (d, t),
    the current timestep is always included in the lookback window. Then
    predecessors are walked backward: the elapsed time from the start of a
    predecessor to the start of the current timestep is accumulated, and
    the predecessor is included as long as that elapsed time is strictly
    less than the minimum time.

    Args:
        filepath: Output CSV path.
        process_min_times: Mapping from process name to minimum time (hours).
        ordered_steps: Ordered list of (period, timestep) tuples.
        step_duration_map: Mapping from (period, timestep) to duration (hours).
        backward_map: Mapping from (period, timestep) to (prev_period, prev_time, jump).
    """
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["process", "period", "time", "period_back", "time_back"])
        for process, min_time in sorted(process_min_times.items()):
            if min_time <= 0:
                continue
            for d, t in ordered_steps:
                # Always include the current timestep itself
                writer.writerow([process, d, t, d, t])
                # Walk backward through predecessors
                accumulated = 0.0
                pos = (d, t)
                while pos in backward_map:
                    d_prev, t_prev, jump_val = backward_map[pos]
                    if jump_val != 1:
                        break  # Gap in timeline, stop lookback
                    prev_pos = (d_prev, t_prev)
                    if prev_pos not in step_duration_map:
                        break  # Previous step not in current solve
                    accumulated += step_duration_map[prev_pos]
                    if accumulated < min_time:
                        writer.writerow([process, d, t, d_prev, t_prev])
                    else:
                        break
                    pos = prev_pos
