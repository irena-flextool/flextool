"""
RollingSolver — handles the recursive solve structure (single, rolling-window,
and nested solves). The most algorithmically complex module.

Entry point: RollingSolver.define_solve_recursive(solve, parent_info, ...)
Read define_solve_recursive → _process_single_solve / _process_rolling_solve →
create_rolling_solves for the full call chain.
"""

import logging
from collections import namedtuple

from flextool.flextoolrunner.runner_state import RunnerState, FlexToolConfigError, SolveResult
from flextool.flextoolrunner.stochastic import StochasticSolver
from flextool.flextoolrunner.timeline_config import get_active_time


ParentSolveInfo = namedtuple('ParentSolveInfo', ['solve', 'roll'])


class RollingSolver:
    def __init__(self, state: RunnerState) -> None:
        self.state = state
        self.logger = state.logger

    def create_rolling_solves(
        self,
        solve: str,
        full_active_time_list: dict,
        jump: float,
        horizon: float,
        start: list | None = None,
        duration: float = -1,
    ) -> tuple[list[str], dict, dict]:
        """
        Splits the solve into an overlapping sequence of solves ("rolls").

        Returns:
            Tuple of (solves, active_time_lists, realized_time_lists)
        """
        active_time_lists: dict = dict()
        realized_time_lists: dict = dict()
        solves: list[str] = []
        starts: list[list] = []
        jumps: list[list] = []
        horizons: list[list] = []
        duration_counter: float = 0
        horizon_counter: float = 0
        jump_counter: float = 0
        started: bool = False
        ended: bool = False

        # search for the start, end and horizon time indexes
        for period, active_time in list(full_active_time_list.items()):
            for i, step in enumerate(active_time):
                if not ended:
                    if started:
                        if duration_counter >= float(duration) and duration != -1:
                            jumps.append(last_index)
                            horizons.append(last_index)
                            ended = True
                            break
                        if jump_counter >= float(jump):
                            jumps.append(last_index)
                            starts.append([period, i])
                            jump_counter -= float(jump)
                        if horizon_counter >= float(horizon):
                            horizons.append(last_index)
                            horizon_counter -= float(jump)
                        horizon_counter += float(step.duration)
                        jump_counter += float(step.duration)
                        duration_counter += float(step.duration)
                        last_index = [period, i]
                    else:
                        if start is None or (start == [period, step.timestep]):
                            starts.append([period, i])
                            started = True
                            horizon_counter += float(step.duration)
                            jump_counter += float(step.duration)
                            duration_counter += float(step.duration)
                            last_index = [period, i]
        if not started:
            message = "Start point not found"
            self.logger.error(message)
            raise FlexToolConfigError(message)
        # if there is start of the roll but not end, the end is the last index of the active time
        diff = len(starts) - len(horizons)
        for i in range(0, diff):
            horizons.append(last_index)
        diff = len(starts) - len(jumps)
        for i in range(0, diff):
            jumps.append(last_index)
        # create the active and realized timesteps from the start and end time indexes
        for index, roll_start in enumerate(starts):
            active: dict = dict()
            realized: dict = dict()
            solve_name = solve + "_roll_" + str(self.state.solve.roll_counter[solve])
            self.state.solve.roll_counter[solve] += 1
            solves.append(solve_name)
            if roll_start[0] == horizons[index][0]:  # if the whole roll is in the same period
                active[roll_start[0]] = full_active_time_list[roll_start[0]][roll_start[1]:horizons[index][1] + 1]
            else:
                started = False
                for period, active_time in list(full_active_time_list.items()):
                    if started:
                        if period == horizons[index][0]:
                            active[period] = full_active_time_list[period][0:horizons[index][1] + 1]
                            break
                        else:
                            active[period] = full_active_time_list[period]
                    elif period == roll_start[0]:
                        active[roll_start[0]] = full_active_time_list[period][roll_start[1]:]
                        started = True
            if roll_start[0] == jumps[index][0]:
                realized[roll_start[0]] = full_active_time_list[roll_start[0]][roll_start[1]:jumps[index][1] + 1]
            else:
                started = False
                for period, active_time in list(full_active_time_list.items()):
                    if started:
                        if period == jumps[index][0]:
                            realized[period] = full_active_time_list[period][0:jumps[index][1] + 1]
                            break
                        else:
                            realized[period] = full_active_time_list[period]
                    elif period == roll_start[0]:
                        realized[period] = full_active_time_list[period][roll_start[1]:]
                        started = True
            active_time_lists[solve_name] = active
            realized_time_lists[solve_name] = realized
        return solves, active_time_lists, realized_time_lists

    @staticmethod
    def _filter_time_list_by_periods(
        full_time_list: dict,
        period_dict: dict,
        solve_name: str,
    ) -> dict:
        """
        Filter a time list to include only periods that are in the given period dict.

        Args:
            full_time_list: dict of {period: [(timestep, idx, value), ...]}
            period_dict: dict where key is solve name and value is list of
                        (period_from, period_included) tuples
            solve_name: Name of the solve to filter for

        Returns:
            dict with only the matching periods
        """
        filtered: dict = dict()
        for period_tuple in period_dict.get(solve_name, []):
            period = period_tuple[1]  # period_included
            if period in full_time_list:
                filtered[period] = full_time_list[period]
        return filtered

    @staticmethod
    def _get_periods_from_parent_time_list(parent_time_list: dict) -> set[str]:
        """
        Extract the set of periods that exist in the parent's time list.
        Used to constrain child solves to only timesteps within parent's scope.

        Args:
            parent_time_list: dict of {period: [(timestep, idx, value), ...]}

        Returns:
            Set of period names
        """
        return set(parent_time_list.keys())

    @staticmethod
    def _filter_time_list_by_parent_scope(
        child_time_list: dict,
        parent_periods: set[str],
    ) -> dict:
        """
        Filter child solve's time list to only include periods that exist in parent.

        Args:
            child_time_list: dict of {period: [(timestep, idx, value), ...]}
            parent_periods: Set of period names from parent

        Returns:
            dict with only periods that exist in parent
        """
        filtered: dict = dict()
        for period, timesteps in child_time_list.items():
            if period in parent_periods:
                filtered[period] = timesteps
        return filtered

    def _process_rolling_solve(
        self,
        solve: str,
        complete_solve_name: str,
        full_active_time_list: dict,
        parent_info: ParentSolveInfo,
        start: list | None,
        duration: float,
    ) -> SolveResult:
        """
        Handle rolling window solve logic, creating multiple roll solves.

        Args:
            solve: Name of the solve
            complete_solve_name: Name of the complete (non-rolled) solve
            full_active_time_list: Full time list for this solve
            parent_info: ParentSolveInfo namedtuple
            start: Starting [period, timestep] if constrained by parent
            duration: Duration constraint from parent

        Returns:
            SolveResult with solves, complete_solves, active_time_lists,
            fix_storage_time_lists, realized_time_lists, parent_roll_lists.
        """
        solves: list[str] = []
        complete_solves: dict = dict()
        active_time_lists: dict = dict()
        fix_storage_time_lists: dict = dict()
        realized_time_lists: dict = dict()
        parent_roll_lists: dict = dict()

        rolling_times = self.state.solve.rolling_times[solve]  # [jump, horizon, duration]
        if duration == -1:
            duration = float(rolling_times[2])

        # Find start timestep if constrained by parent
        period_start_timestep = start
        if start is not None:
            start_timestep = StochasticSolver(self.state).find_next_timestep(
                full_active_time_list, start, parent_info.solve, solve)
            period_start_timestep = [start[0], start_timestep]

        # Create rolling solves
        roll_solves, roll_active_time_lists, roll_realized_time_lists = (
            self.create_rolling_solves(solve, full_active_time_list,
                                       float(rolling_times[0]), float(rolling_times[1]),
                                       period_start_timestep, duration))

        # Track metadata for each roll
        for roll_name in roll_solves:
            complete_solves[roll_name] = complete_solve_name
            parent_roll_lists[roll_name] = parent_info.roll

        active_time_lists.update(roll_active_time_lists)

        # For rolling solves, fix_storage and realized are the same (the "jump" portion)
        fix_storage_time_lists.update(roll_realized_time_lists)
        realized_time_lists.update(roll_realized_time_lists)

        # Mark first solve of this complete solve (for state start constraints)
        if parent_info.roll is not None:
            if parent_info.roll in self.state.solve.first_of_complete_solve:
                self.state.solve.first_of_complete_solve.append(roll_solves[0])
        else:
            self.state.solve.first_of_complete_solve.append(roll_solves[0])

        self.state.solve.last_of_solve.append(roll_solves[-1])

        # Process contained solves
        if solve in self.state.solve.contains_solves:
            contain_solves = self.state.solve.contains_solves[solve]
            if len(contain_solves) > 1:
                message = "More than one solve in a rolling solve, not managed"
                logging.error(message)
                raise FlexToolConfigError(message)

            contains_solve = contain_solves[0]

            for index, roll_name in enumerate(roll_solves):
                solves.append(roll_name)

                # Determine start time for child solve
                # Child should start at first timestep of this roll
                if index != 0:
                    first_period = list(roll_active_time_lists[roll_name].keys())[0]
                    first_timestep = roll_active_time_lists[roll_name][first_period][0].timestep
                    child_start = [first_period, first_timestep]
                else:
                    child_start = None

                # Child duration equals parent's jump (the realized portion)
                child_duration = float(rolling_times[0])

                # Get parent's realized periods for child to use as scope
                parent_realized_periods = set(roll_realized_time_lists[roll_name].keys())

                # Recursively process child solve
                child_parent_info = ParentSolveInfo(solve=solve, roll=roll_name)
                child = self.define_solve_recursive(
                    contains_solve, child_parent_info, parent_realized_periods,
                    child_start, child_duration)

                solves += child.solves
                complete_solves.update(child.complete_solves)
                parent_roll_lists.update(child.parent_roll_lists)
                active_time_lists.update(child.active_time_lists)
                fix_storage_time_lists.update(child.fix_storage_time_lists)
                realized_time_lists.update(child.realized_time_lists)
        else:
            solves += roll_solves

        return SolveResult(
            solves=solves,
            complete_solves=complete_solves,
            active_time_lists=active_time_lists,
            fix_storage_time_lists=fix_storage_time_lists,
            realized_time_lists=realized_time_lists,
            parent_roll_lists=parent_roll_lists,
        )

    def _process_single_solve(
        self,
        solve: str,
        full_active_time_list: dict,
        parent_info: ParentSolveInfo,
        complete_solve_name: str | None = None,
    ) -> SolveResult:
        """
        Handle single (non-rolling) solve logic.

        Args:
            solve: Name of the solve
            full_active_time_list: Full time list for this solve
            parent_info: ParentSolveInfo namedtuple
            complete_solve_name: Original (un-renamed) solve name for complete_solves
                mapping. When a solve is renamed (e.g. storage_fullYear_6h →
                storage_fullYear_6h_p2020), this keeps the mapping pointing to the
                original name which is in real_solves and has solve_period_history.

        Returns:
            SolveResult with solves, complete_solves, active_time_lists,
            fix_storage_time_lists, realized_time_lists, parent_roll_lists.
        """
        if complete_solve_name is None:
            complete_solve_name = solve

        solves: list[str] = [solve]
        complete_solves: dict = dict()
        active_time_lists: dict = dict()
        invest_time_lists: dict = dict()
        fix_storage_time_lists: dict = dict()
        realized_time_lists: dict = dict()
        parent_roll_lists: dict = dict()

        complete_solves[solve] = complete_solve_name
        parent_roll_lists[solve] = parent_info.roll
        active_time_lists[solve] = full_active_time_list

        # Get fix_storage and realized time lists from class attributes
        invest_time_lists[solve] = self._filter_time_list_by_periods(
            full_active_time_list, self.state.solve.invest_periods, solve)
        fix_storage_time_lists[solve] = self._filter_time_list_by_periods(
            full_active_time_list, self.state.solve.fix_storage_periods, solve)
        realized_time_lists[solve] = self._filter_time_list_by_periods(
            full_active_time_list, self.state.solve.realized_periods, solve)

        self.state.solve.first_of_complete_solve.append(solve)
        self.state.solve.last_of_solve.append(solve)

        # Process contained solves
        if solve in self.state.solve.contains_solves:
            contain_solves = self.state.solve.contains_solves[solve]

            # Get parent's scope: union of fix_storage, realized, and realized_invest periods.
            # Invest solves only have invest/realized_invest periods (not fix_storage/realized),
            # so realized_invest_periods is needed to provide scope for their children.
            realized_invest_time_list = self._filter_time_list_by_periods(
                full_active_time_list, self.state.solve.realized_invest_periods, solve)
            parent_scope_periods = (set(fix_storage_time_lists[solve].keys()) |
                                    set(realized_time_lists[solve].keys()) |
                                    set(realized_invest_time_list.keys()))

            # Fallback: when a solve was renamed (e.g. storage_fullYear_6h_p2020),
            # its period attributes aren't copied by duplicate_solve, leaving scope
            # empty. Use the solve's own active periods as scope in that case.
            if not parent_scope_periods:
                parent_scope_periods = set(full_active_time_list.keys())

            for contain_solve in contain_solves:
                child_parent_info = ParentSolveInfo(solve=solve, roll=solve)
                child = self.define_solve_recursive(
                    contain_solve, child_parent_info, parent_scope_periods, None, -1)

                solves += child.solves
                complete_solves.update(child.complete_solves)
                parent_roll_lists.update(child.parent_roll_lists)
                active_time_lists.update(child.active_time_lists)
                fix_storage_time_lists.update(child.fix_storage_time_lists)
                realized_time_lists.update(child.realized_time_lists)

        return SolveResult(
            solves=solves,
            complete_solves=complete_solves,
            active_time_lists=active_time_lists,
            fix_storage_time_lists=fix_storage_time_lists,
            realized_time_lists=realized_time_lists,
            parent_roll_lists=parent_roll_lists,
        )

    def define_solve_recursive(
        self,
        solve: str,
        parent_info: ParentSolveInfo,
        parent_scope_periods: set[str] | None = None,
        start: list | None = None,
        duration: float = -1,
    ) -> SolveResult:
        """
        Recursively define solve structure and determine time period mappings.

        This is the core recursive function that processes nested and rolling solves.

        Args:
            solve: Name of the solve to process
            parent_info: ParentSolveInfo namedtuple with parent solve and roll info
            parent_scope_periods: Set of period names from parent's realized/fix_storage
            start: Optional [period, timestep] to start from (for child solves)
            duration: Duration constraint from parent (for child solves)

        Returns:
            SolveResult with solves, complete_solves, active_time_lists,
            fix_storage_time_lists, realized_time_lists, parent_roll_lists.
        """
        new_name = solve
        if new_name not in self.state.solve.real_solves:
            self.state.solve.real_solves.append(new_name)

        if parent_info.solve:
            joint_current_solve_periods = list(set(self.state.solve.invest_periods[solve] + self.state.solve.fix_storage_periods[solve] + self.state.solve.realized_periods[solve]))
            current_solve_periods = [t[0] for t in joint_current_solve_periods]
            joint_parent_periods = list(set(self.state.solve.invest_periods[parent_info.solve] + self.state.solve.fix_storage_periods[parent_info.solve] + self.state.solve.realized_periods[parent_info.solve]))
            parent_period = set([t[0] for t in joint_parent_periods])
            # Find which child periods overlap with the parent
            matching_periods = [p for p in current_solve_periods if p in parent_period]
            if len(matching_periods) == 1:
                # Single matching period (rolling-window parent): rename child to per-period solve
                current_solve_period = matching_periods[0]
                new_name = solve + "_" + str(current_solve_period)
                self.state.solve.duplicate_solve(solve, new_name, update_model_solves=False)
                self.state.solve.solve_period_years_represented[new_name] = self.state.solve.solve_period_years_represented[solve]

                new_period_timeset_list = []
                for solve2, period__timeset_list in list(self.state.solve.timesets_used_by_solves.items()):
                    if solve2 == solve:
                        for period__timeset in period__timeset_list:
                            if period__timeset[0] == current_solve_period:
                                new_period_timeset_list.append(period__timeset)
                if new_name not in self.state.solve.timesets_used_by_solves.keys():
                    self.state.solve.timesets_used_by_solves[new_name] = new_period_timeset_list
                else:
                    for item in new_period_timeset_list:
                        if item not in self.state.solve.timesets_used_by_solves[new_name]:
                            self.state.solve.timesets_used_by_solves[new_name].append(item)
            # When multiple periods match (single-solve parent with multiple periods),
            # keep the original solve name and all its timesets — the active time
            # will be filtered by parent scope later.

        # Get full active time list for this solve (all timesteps it could use)
        full_active_time_list_own = get_active_time(
            new_name, self.state.solve.timesets_used_by_solves, self.state.timeline.timeset_durations,
            self.state.timeline.timelines, self.state.timeline.timesets__timeline)

        # If this is a child solve, constrain it to parent's scope
        if not parent_scope_periods:
            # Top-level solve: include realized_invest_periods in scope
            # (they contribute to active time but not to fix_storage/realized directly)
            full_active_time_list = full_active_time_list_own
        else:
            full_active_time_list = self._filter_time_list_by_parent_scope(
                full_active_time_list_own, parent_scope_periods)

        # Determine solve mode
        solve_mode = self.state.solve.solve_modes.get(new_name, "single_solve")
        if solve_mode == "rolling_window":
            # Process as rolling window solve
            complete_solve_name = solve
            return self._process_rolling_solve(
                new_name, complete_solve_name, full_active_time_list,
                parent_info, start, duration)
        else:
            # Process as single solve — pass original name as complete_solve_name
            # so renamed solves (e.g. storage_fullYear_6h_p2020) map back to the
            # original which is in real_solves with proper solve_period_history.
            return self._process_single_solve(
                new_name, full_active_time_list, parent_info,
                complete_solve_name=solve)
