"""
StochasticSolver — stochastic branching of time periods.

Handles stochastic branching where multiple future scenarios diverge from a
common point, and the nested timeline matching map used when upper solves
fix storage for lower solves.

Entry point: StochasticSolver.create_stochastic_periods(...)
"""
from __future__ import annotations

import bisect
import sys
from collections import OrderedDict, defaultdict
from typing import Any

from flextool.flextoolrunner.runner_state import RunnerState
from flextool.flextoolrunner.timeline_config import make_step_jump


class StochasticSolver:
    """Handles stochastic branching of time periods and timeline matching."""

    def __init__(self, state: RunnerState) -> None:
        self.state = state
        self.logger = state.logger

    def _get_timeset_for_period(
        self,
        period_timesets: list[tuple[str, str]],
        real_period: str,
    ) -> str | None:
        """Find the timeset corresponding to a real period in period_timesets."""
        for period_timeset in period_timesets:
            if period_timeset[0] == real_period:
                return period_timeset[1]
        return None

    def connect_two_timelines(
        self,
        period: str,
        first_solve: str,
        second_solve: str,
        period__branch: list[tuple[str, str]],
    ) -> tuple[OrderedDict[str, float], OrderedDict[str, float]]:
        """Connect two solve timelines by computing cumulative durations from start.

        Returns two OrderedDicts mapping timestep name to cumulative duration,
        one for each solve's timeline in the given period.
        """
        first_period_timesets = self.state.solve.timesets_used_by_solves[first_solve]
        second_period_timesets = self.state.solve.timesets_used_by_solves[second_solve]

        real_period: str | None = None
        for row in period__branch:
            if row[1] == period:
                real_period = row[0]

        first_timeset = self._get_timeset_for_period(first_period_timesets, real_period)
        second_timeset = self._get_timeset_for_period(second_period_timesets, real_period)

        if first_timeset is None:
            raise ValueError(
                f"Could not find first_timeset for real_period={real_period} "
                f"in first_period_timesets={first_period_timesets}"
            )
        if second_timeset is None:
            raise ValueError(
                f"Could not find second_timeset for real_period={real_period} "
                f"in second_period_timesets={second_period_timesets}"
            )

        first_timeline = self.state.timeline.timesets__timeline[first_timeset]
        second_timeline = self.state.timeline.timesets__timeline[second_timeset]

        first_timeline_duration_from_start: OrderedDict[str, float] = OrderedDict()
        second_timeline_duration_from_start: OrderedDict[str, float] = OrderedDict()
        counter: float = 0
        for timestep in self.state.timeline.timelines[first_timeline]:
            first_timeline_duration_from_start[timestep[0]] = counter
            counter += float(timestep[1])
        counter = 0
        for timestep in self.state.timeline.timelines[second_timeline]:
            second_timeline_duration_from_start[timestep[0]] = counter
            counter += float(timestep[1])

        return first_timeline_duration_from_start, second_timeline_duration_from_start

    def find_previous_timestep(
        self,
        from_active_time_list: OrderedDict[str, list[tuple[str, ...]]],
        period_timestamp: tuple[str, str],
        this_solve: str,
        from_solve: str,
        period__branch: list[tuple[str, str]],
    ) -> str:
        """Find the previous timestep in from_solve that corresponds to period_timestamp in this_solve."""
        this_timeline_duration_from_start, from_timeline_duration_from_start = (
            self.connect_two_timelines(period_timestamp[0], this_solve, from_solve, period__branch)
        )

        real_period: str | None = None
        for row in period__branch:
            if row[1] == period_timestamp[0]:
                real_period = row[0]
        from_start = this_timeline_duration_from_start[period_timestamp[1]]
        last_timestep = from_active_time_list[real_period][0][0]
        previous_timestep = from_active_time_list[real_period][-1][0]  # last is the default
        for timestep in from_active_time_list[real_period]:
            if from_timeline_duration_from_start[timestep[0]] > from_start:
                previous_timestep = last_timestep
                break
            last_timestep = timestep[0]
        return previous_timestep

    def find_next_timestep(
        self,
        from_active_time_list: OrderedDict[str, list[tuple[str, ...]]],
        period_timestamp: tuple[str, str],
        this_solve: str,
        from_solve: str,
    ) -> str:
        """Find the next timestep in from_solve that corresponds to period_timestamp in this_solve."""
        this_timeline_duration_from_start, from_timeline_duration_from_start = (
            self.connect_two_timelines(
                period_timestamp[0], this_solve, from_solve,
                [(period_timestamp[0], period_timestamp[0])]
            )
        )

        from_start = this_timeline_duration_from_start[period_timestamp[1]]
        next_timestep = from_active_time_list[period_timestamp[0]][-1][0]  # last is the default
        for timestep in from_active_time_list[period_timestamp[0]]:
            if from_timeline_duration_from_start[timestep[0]] >= from_start:
                next_timestep = timestep[0]
                break
        return next_timestep

    def write_timeline_matching_map(
        self,
        upper_active_time_list: OrderedDict[str, list[tuple[str, ...]]],
        lower_active_time_list: OrderedDict[str, list[tuple[str, ...]]],
        upper_solve: str,
        lower_solve: str,
        period__branch: list[tuple[str, str]],
    ) -> None:
        """Write the timeline matching map CSV for nested solve storage fixing.

        Maps each lower-solve timestep to the corresponding upper-solve timestep
        using binary search on cumulative timeline durations.
        """
        # Pre-compute period -> real_period mapping (O(n) once instead of O(n) per timestep)
        period_to_real: dict[str, str] = {row[1]: row[0] for row in period__branch}

        # Get the periods that exist in the upper solve's timesets
        upper_period_timesets = self.state.solve.timesets_used_by_solves[upper_solve]
        upper_periods: set[str] = {pt[0] for pt in upper_period_timesets}

        matching_map: OrderedDict[tuple[str, str], str] = OrderedDict()
        for period, lower_active_time in lower_active_time_list.items():
            # Get the real period for this period
            real_period = period_to_real.get(period)

            # Skip periods that don't exist in the upper solve's timesets
            if real_period not in upper_periods:
                continue

            # Call connect_two_timelines ONCE per period, not per timestep
            this_timeline_duration, from_timeline_duration = self.connect_two_timelines(
                period, lower_solve, upper_solve, period__branch
            )

            upper_timesteps = upper_active_time_list[real_period]

            # Pre-compute list of (duration, timestep_name) for binary search
            from_durations: list[float] = []
            from_timestep_names: list[str] = []
            for ts in upper_timesteps:
                from_durations.append(from_timeline_duration[ts[0]])
                from_timestep_names.append(ts[0])

            default_timestep = upper_timesteps[-1][0]  # last is the default

            for timestep in lower_active_time:
                period_timestep = (period, timestep[0])
                from_start = this_timeline_duration[timestep[0]]

                # Binary search: find first index where duration > from_start
                idx = bisect.bisect_right(from_durations, from_start)

                if idx == 0:
                    # All upper durations are > from_start, use first timestep
                    previous_timestep = upper_timesteps[0][0]
                elif idx >= len(from_durations):
                    # All upper durations are <= from_start, use default (last)
                    previous_timestep = default_timestep
                else:
                    # Use the timestep just before the one that exceeds from_start
                    previous_timestep = from_timestep_names[idx - 1]

                matching_map[period_timestep] = previous_timestep

        with open("solve_data/timeline_matching_map.csv", 'w') as realfile:
            realfile.write("period,step,upper_step\n")
            for period_timestep, upper_timestep in matching_map.items():
                realfile.write(period_timestep[0] + "," + period_timestep[1] + "," + upper_timestep + "\n")

    def create_stochastic_periods(
        self,
        stochastic_branches: dict[str, Any],
        solves: list[str],
        complete_solves: dict[str, str],
        active_time_lists: dict[str, OrderedDict],
        fix_storage_time_lists: dict[str, OrderedDict],
        realized_time_lists: dict[str, OrderedDict],
    ) -> tuple[
        defaultdict[str, list],
        defaultdict[str, list],
        dict[str, OrderedDict],
        OrderedDict[str, Any],
        dict[str, OrderedDict],
        dict[str, OrderedDict],
        defaultdict[str, Any],
    ]:
        """Apply stochastic branching to time periods.

        Processes stochastic branches and creates branched versions of periods
        where multiple future scenarios diverge. Branches are added to active_time_lists but
        NOT to realized_time_lists or fix_storage_time_lists, since branches represent
        future scenarios that are optimized over but not committed/realized.

        Returns:
            Tuple of (period__branch_lists, solve_branch__time_branch_lists,
                     active_time_lists, jump_lists, fix_storage_time_lists,
                     realized_time_lists, branch_start_time_lists)
        """
        period__branch_lists: defaultdict[str, list] = defaultdict(list)
        solve_branch__time_branch_lists: defaultdict[str, list] = defaultdict(list)
        jump_lists: OrderedDict[str, Any] = OrderedDict()
        branch_start_time_lists: defaultdict[str, Any] = defaultdict()

        for solve in solves:
            new_realized_time_list: OrderedDict = OrderedDict()
            new_fix_storage_time_list: OrderedDict = OrderedDict()
            new_active_time_list: OrderedDict = OrderedDict()

            info = stochastic_branches[complete_solves[solve]]
            active_time_list = active_time_lists[solve]
            realized_time_list = realized_time_lists[solve]
            fix_storage_time_list = fix_storage_time_lists[solve]

            branched = False
            branches: list[str] = []
            branch_start_time_lists[solve] = None

            # Get first step for validation
            for period, active_time in active_time_list.items():
                first_step = (period, active_time[0][0])
                break

            # Check that the start times of the solves can be found from stochastic_branches
            found_start = False
            for row in info:
                if first_step[1] == row[2] and "yes" == row[3]:
                    found_start = True
            if found_start is False and len(info) != 0:
                self.logger.error(
                    "A realized start time of the solve cannot be found from the stochastic_branches parameter. "
                    "Check that stochastic_branches has a realized : yes, branch for the start of the solve"
                    "and that the possible rolling_jump matches with the branch starts"
                )
                sys.exit(-1)

            # Process each period to create branches
            for period, active_time in active_time_list.items():
                if not branched:
                    period__branch_lists[solve].append((period, period))
                    # Get all start times
                    start_times: defaultdict[str, list] = defaultdict(list)
                    for row in info:
                        if row[0] == period:
                            start_times[row[2]].append((row[1], row[4], row[3]))

                    # Check if any timestep triggers branching
                    for step in active_time:
                        if step[0] in start_times.keys():
                            branched = True
                            branch_start_time_lists[solve] = (period, step[0])

                            # Add active time for base period
                            new_active_time_list[period] = active_time

                            # Copy realized and fix_storage ONLY for base period (not branches)
                            if period in realized_time_list:
                                new_realized_time_list[period] = realized_time_list[period]
                            if period in fix_storage_time_list:
                                new_fix_storage_time_list[period] = fix_storage_time_list[period]

                            # Create branches (branches get active time but NOT realized/fix_storage)
                            for branch__weight__real in start_times[step[0]]:
                                branch = branch__weight__real[0]
                                branches.append(branch)
                                solve_branch = period + "_" + branch

                                # If the weight is zero, do not add to the timeline
                                if (
                                    float(branch__weight__real[1]) != 0.0
                                    and branch != period
                                    and branch__weight__real[2] != "yes"
                                ):
                                    # Branches get active time for optimization
                                    new_active_time_list[solve_branch] = active_time[0:]
                                    # But NOT realized or fix_storage (they're future scenarios)
                                    solve_branch__time_branch_lists[solve].append((solve_branch, branch))

                                period__branch_lists[solve].append((period, solve_branch))

                                # Get timesteps for stochastic tracking
                                for i in active_time[0:]:
                                    self.state.timeline.stochastic_timesteps[solve].append(
                                        (solve_branch, i[0])
                                    )
                            break
                else:
                    # If the jump is longer than the period (continuation after branching)
                    for branch in branches:
                        solve_branch = period + "_" + branch
                        # Branches continue to get active time but not realized/fix_storage
                        period__branch_lists[solve].append((period, solve_branch))
                        solve_branch__time_branch_lists[solve].append((solve_branch, branch))
                        for i in active_time_list[period]:
                            self.state.timeline.stochastic_timesteps[solve].append(
                                (solve_branch, i[0])
                            )

                # Before branching occurs, copy periods as-is
                if not branched:
                    new_active_time_list[period] = active_time_list[period]
                    if period in realized_time_list:
                        new_realized_time_list[period] = realized_time_list[period]
                    if period in fix_storage_time_list:
                        new_fix_storage_time_list[period] = fix_storage_time_list[period]

            # Find the realized branch for this start time
            for period, active_time in active_time_list.items():
                found = 0
                # Before branching
                for row in info:
                    if row[0] == period and row[2] == active_time[0][0] and row[3] == 'yes':
                        found += 1
                        solve_branch__time_branch_lists[solve].append((period, row[1]))
                # After branching
                if found == 0 and branch_start_time_lists[solve] is not None:
                    for row in info:
                        if (
                            row[0] == branch_start_time_lists[solve][0]
                            and row[2] == branch_start_time_lists[solve][1]
                            and row[3] == 'yes'
                        ):
                            found += 1
                            solve_branch__time_branch_lists[solve].append((period, row[1]))
                if (branch_start_time_lists[solve] is not None and found == 0) or found > 1:
                    self.logger.error(
                        "Each period should have one and only one realized branch. Found: "
                        + str(found)
                        + "\n"
                    )
                    sys.exit(-1)

            # Update the time lists for this solve
            realized_time_lists[solve] = new_realized_time_list
            fix_storage_time_lists[solve] = new_fix_storage_time_list
            active_time_lists[solve] = new_active_time_list
            jump_lists[solve] = make_step_jump(
                new_active_time_list,
                period__branch_lists[solve],
                solve_branch__time_branch_lists[solve],
            )

        return (
            period__branch_lists,
            solve_branch__time_branch_lists,
            active_time_lists,
            jump_lists,
            fix_storage_time_lists,
            realized_time_lists,
            branch_start_time_lists,
        )
