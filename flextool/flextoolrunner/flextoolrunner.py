import time
import logging
import copy
import sys
import os
import pandas as pd
import shutil
import spinedb_api as api
from spinedb_api import DatabaseMapping
# from spinedb_api.filters.scenario_filter import scenario_filter_config, scenario_filter_from_dict
from pathlib import Path
from collections import OrderedDict, namedtuple
from collections import defaultdict

from flextool.flextoolrunner.db_reader import check_version
from flextool.flextoolrunner import input_writer
from flextool.flextoolrunner import solve_writers
from flextool.flextoolrunner.solve_config import SolveConfig
from flextool.flextoolrunner.timeline_config import (
    TimelineConfig,
    get_active_time,
    make_step_jump,
    separate_period_and_timeseries_data,
)
from flextool.flextoolrunner.runner_state import PathConfig, RunnerState
from flextool.flextoolrunner.solver_runner import SolverRunner

#return_codes
#0 : Success
#-1: Failure (Defined in the Toolbox)
#1: Infeasible or unbounded problem (not implemented in the toolbox, functionally same as -1. For a possiblity of a graphical depiction)


class FlexToolRunner:
    """
    Define Class to run the model and read and recreate the required config files:
    """

    def __init__(self, input_db_url=None, output_path=None, scenario_name=None, flextool_dir=None, bin_dir=None, root_dir=None):
        logger = logging.getLogger(__name__)
        translation = {39: None}
        # delete highs.log from previous run
        if os.path.exists("./HiGHS.log"):
            os.remove("./HiGHS.log")
        # make a directory for solve data
        if not os.path.exists("./solve_data"):
            os.makedirs("./solve_data")
        # Build PathConfig
        _default_root = Path(__file__).parent.parent.parent
        paths = PathConfig(
            flextool_dir=Path(flextool_dir) if flextool_dir is not None else _default_root / "flextool",
            bin_dir=Path(bin_dir) if bin_dir is not None else _default_root / "bin",
            root_dir=Path(root_dir) if root_dir is not None else _default_root,
            output_path=Path(output_path) if output_path is not None else _default_root,
        )
        # read the data in
        # open connection to input db
        if scenario_name:
            scen_config = api.filters.scenario_filter.scenario_filter_config(scenario_name)
        with (DatabaseMapping(input_db_url) as db):
            if scenario_name:
                api.filters.scenario_filter.scenario_filter_from_dict(db, scen_config)
            else:
                scen_names = db.get_scenario_items()
                if len(scen_names) == 0:
                    logger.error("No scenario found")
                    sys.exit(-1)
                scenario_name=scen_names[0]['name']
            logger.info(" Work dir: " + str(paths.root_dir) + "\nDB URL: " + str(db.sa_url) + "\nScenario name: " + scenario_name + "\nOutput path: " + str(paths.output_path))
            if len(db.get_scenario_alternative_items(scenario_name=scenario_name)) == 0:
                logger.error("No alternatives in the scenario, i.e. empty scenario.")
                sys.exit(-1)

            db.fetch_all("parameter_value")
            check_version(db=db, logger=logger)
            # Solve-level fields — delegated to SolveConfig
            solve = SolveConfig.load_from_db(db=db, logger=logger)
            # Timeline-level fields — delegated to TimelineConfig
            timeline = TimelineConfig.load_from_db(db=db, logger=logger)

        # Post-DB initialization of timeline
        timeline.create_assumptive_parts(solve)
        timeline.create_timeline_from_timestep_duration()

        # Assemble RunnerState — the single cross-cutting state container
        self.state = RunnerState(paths=paths, solve=solve, timeline=timeline, logger=logger)





    #these exist to connect timesteps from two different timelines or aggregated versions of one
    def connect_two_timelines(self,period,first_solve,second_solve, period__branch):
        first_period_timesets = self.state.solve.timesets_used_by_solves[first_solve]
        second_period_timesets = self.state.solve.timesets_used_by_solves[second_solve]
        real_period = None
        for row in period__branch:
            if row[1] == period:
                real_period = row[0]
        first_timeset = None
        for period_timeset in first_period_timesets:
            if period_timeset[0] == real_period:
                first_timeset = period_timeset[1]
        second_timeset = None
        for period_timeset in second_period_timesets:
            if period_timeset[0] == real_period:
                second_timeset = period_timeset[1]

        if first_timeset is None:
            raise ValueError(f"Could not find first_timeset for real_period={real_period} in first_period_timesets={first_period_timesets}")
        if second_timeset is None:
            raise ValueError(f"Could not find second_timeset for real_period={real_period} in second_period_timesets={second_period_timesets}")

        first_timeline = self.state.timeline.timesets__timeline[first_timeset]
        second_timeline = self.state.timeline.timesets__timeline[second_timeset]

        first_timeline_duration_from_start = OrderedDict()
        second_timeline_duration_from_start = OrderedDict()
        counter = 0
        for timestep in self.state.timeline.timelines[first_timeline]:
            first_timeline_duration_from_start[timestep[0]] = counter
            counter += float(timestep[1])
        counter = 0
        for timestep in self.state.timeline.timelines[second_timeline]:
            second_timeline_duration_from_start[timestep[0]] = counter
            counter += float(timestep[1])
    
        return first_timeline_duration_from_start,second_timeline_duration_from_start

    def find_previous_timestep(self, from_active_time_list, period_timestamp, this_solve, from_solve, period__branch):
        
        this_timeline_duration_from_start, from_timeline_duration_from_start = self.connect_two_timelines(period_timestamp[0],this_solve,from_solve, period__branch)

        for row in period__branch:
            if row[1] == period_timestamp[0]:
                real_period = row[0]
        from_start = this_timeline_duration_from_start[period_timestamp[1]]
        last_timestep = from_active_time_list[real_period][0][0]
        previous_timestep = from_active_time_list[real_period][-1][0] #last is the default, as the last timestep can be shorter and cause issues
        for timestep in from_active_time_list[real_period]:
            if from_timeline_duration_from_start[timestep[0]] > from_start:
                previous_timestep = last_timestep 
                break
            last_timestep = timestep[0]
        return previous_timestep

    def find_next_timestep(self, from_active_time_list, period_timestamp, this_solve, from_solve):

        this_timeline_duration_from_start, from_timeline_duration_from_start = self.connect_two_timelines(period_timestamp[0],this_solve,from_solve,[(period_timestamp[0],period_timestamp[0])])

        from_start = this_timeline_duration_from_start[period_timestamp[1]]
        next_timestep = from_active_time_list[period_timestamp[0]][-1][0] #last is the default, as the last timestep can be shorter and cause issues
        for timestep in from_active_time_list[period_timestamp[0]]:
            if from_timeline_duration_from_start[timestep[0]] >= from_start:
                next_timestep = timestep[0]
                break
        return next_timestep

    def write_timeline_matching_map(self, upper_active_time_list, lower_active_time_list, upper_solve, lower_solve, period__branch):
        import bisect

        # Pre-compute period -> real_period mapping (O(n) once instead of O(n) per timestep)
        period_to_real = {row[1]: row[0] for row in period__branch}

        # Get the periods that exist in the upper solve's timesets
        upper_period_timesets = self.state.solve.timesets_used_by_solves[upper_solve]
        upper_periods = {pt[0] for pt in upper_period_timesets}

        matching_map = OrderedDict()
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
            from_durations = []
            from_timestep_names = []
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
                realfile.write(period_timestep[0]+","+period_timestep[1]+","+ upper_timestep+"\n")

    def create_rolling_solves(self, solve, full_active_time_list, jump, horizon, start = None, duration = -1):
        """
        splits the solve to overlapping sequence of solves "rolls" 
        """
        active_time_lists= OrderedDict()    
        realized_time_lists = OrderedDict()
        solves=[]
        starts=[]
        jumps= []
        horizons= []
        duration_counter = 0
        horizon_counter = 0
        jump_counter = 0
        started = False
        ended = False

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
                            starts.append([period,i])
                            jump_counter -= float(jump)
                        if horizon_counter >= float(horizon):
                            horizons.append(last_index)
                            horizon_counter -= float(jump)
                        horizon_counter += float(step[2])
                        jump_counter += float(step[2])
                        duration_counter += float(step[2])
                        last_index = [period,i]
                    else:
                        if start == None or (start == [period, step[0]]):
                            starts.append([period, i])
                            started = True
                            horizon_counter += float(step[2])
                            jump_counter += float(step[2])
                            duration_counter += float(step[2])
                            last_index=[period,i]
        if started == False:
            self.state.logger.error("Start point not found")
            sys.exit(-1)
        # if there is start of the roll but not end, the end is the last index of the active time
        diff = len(starts)-len(horizons)
        for i in range(0,diff):
            horizons.append(last_index)
        diff = len(starts)-len(jumps)
        for i in range(0,diff):
            jumps.append(last_index)
        # create the active and realized timesteps from the start and end time indexes
        for index, roll_start in enumerate(starts): 
            active = OrderedDict()
            realized = OrderedDict()
            solve_name= solve+"_roll_" + str(self.state.solve.roll_counter[solve])
            self.state.solve.roll_counter[solve]+=1
            solves.append(solve_name) 
            if roll_start[0]==horizons[index][0]: #if the whole roll is in the same period
                active[roll_start[0]] = full_active_time_list[roll_start[0]][roll_start[1]:horizons[index][1]+1]
            else:
                started = False
                for period, active_time in list(full_active_time_list.items()):
                    if started:
                        if period == horizons[index][0]:
                            active[period] = full_active_time_list[period][0:horizons[index][1]+1]
                            break
                        else:
                            active[period] = full_active_time_list[period]
                    elif period == roll_start[0]:
                        active[roll_start[0]] = full_active_time_list[period][roll_start[1]:]
                        started = True
            if roll_start[0]==jumps[index][0]:
                realized[roll_start[0]] = full_active_time_list[roll_start[0]][roll_start[1]:jumps[index][1]+1]
            else:
                started = False
                for period, active_time in list(full_active_time_list.items()):
                    if started:
                        if period == jumps[index][0]:
                            realized[period] = full_active_time_list[period][0:jumps[index][1]+1]
                            break
                        else:
                            realized[period] = full_active_time_list[period]
                    elif period == roll_start[0]:
                        realized[period] = full_active_time_list[period][roll_start[1]:]
                        started = True
            active_time_lists[solve_name] = active
            realized_time_lists[solve_name] = realized
        return solves, active_time_lists, realized_time_lists

    # Named tuple for tracking parent solve relationships
    ParentSolveInfo = namedtuple('ParentSolveInfo', ['solve', 'roll'])

    def _filter_time_list_by_periods(self, full_time_list, period_dict, solve_name):
        """
        Filter a time list to include only periods that are in the given period dict.

        Args:
            full_time_list: OrderedDict of {period: [(timestep, idx, value), ...]}
            period_dict: dict where key is solve name and value is list of
                        (period_from, period_included) tuples
            solve_name: Name of the solve to filter for

        Returns:
            OrderedDict with only the matching periods
        """
        filtered = OrderedDict()
        for period_tuple in period_dict.get(solve_name, []):
            period = period_tuple[1]  # period_included
            if period in full_time_list:
                filtered[period] = full_time_list[period]
        return filtered

    def _get_periods_from_parent_time_list(self, parent_time_list):
        """
        Extract the set of periods that exist in the parent's time list.
        Used to constrain child solves to only timesteps within parent's scope.

        Args:
            parent_time_list: OrderedDict of {period: [(timestep, idx, value), ...]}

        Returns:
            Set of period names
        """
        return set(parent_time_list.keys())

    def _filter_time_list_by_parent_scope(self, child_time_list, parent_periods):
        """
        Filter child solve's time list to only include periods that exist in parent.

        Args:
            child_time_list: OrderedDict of {period: [(timestep, idx, value), ...]}
            parent_periods: Set of period names from parent

        Returns:
            OrderedDict with only periods that exist in parent
        """
        filtered = OrderedDict()
        for period, timesteps in child_time_list.items():
            if period in parent_periods:
                filtered[period] = timesteps
        return filtered

    def _process_rolling_solve(self, solve, complete_solve_name, full_active_time_list,
                               parent_info, start, duration):
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
            Tuple of (solves, complete_solves, active_time_lists, fix_storage_time_lists,
                     realized_time_lists, parent_roll_lists)
        """
        solves = []
        complete_solves = OrderedDict()
        active_time_lists = OrderedDict()
        fix_storage_time_lists = OrderedDict()
        realized_time_lists = OrderedDict()
        parent_roll_lists = OrderedDict()

        rolling_times = self.state.solve.rolling_times[solve]  # [jump, horizon, duration]
        if duration == -1:
            duration = float(rolling_times[2])

        # Find start timestep if constrained by parent
        period_start_timestep = start
        if start is not None:
            start_timestep = self.find_next_timestep(
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
                logging.error("More than one solve in a rolling solve, not managed")
                sys.exit(-1)

            contains_solve = contain_solves[0]

            for index, roll_name in enumerate(roll_solves):
                solves.append(roll_name)

                # Determine start time for child solve
                # Child should start at first timestep of this roll
                if index != 0:
                    first_period = list(roll_active_time_lists[roll_name].keys())[0]
                    first_timestep = roll_active_time_lists[roll_name][first_period][0][0]
                    child_start = [first_period, first_timestep]
                else:
                    child_start = None

                # Child duration equals parent's jump (the realized portion)
                child_duration = float(rolling_times[0])

                # Get parent's realized periods for child to use as scope
                parent_realized_periods = set(roll_realized_time_lists[roll_name].keys())

                # Recursively process child solve
                child_parent_info = self.ParentSolveInfo(solve=solve, roll=roll_name)
                (child_solves, child_complete_solves, child_active_time_lists,
                 child_fix_storage_time_lists, child_realized_time_lists,
                 child_parent_roll_lists) = self._define_solve_recursive(
                    contains_solve, child_parent_info, parent_realized_periods,
                    child_start, child_duration)

                solves += child_solves
                complete_solves.update(child_complete_solves)
                parent_roll_lists.update(child_parent_roll_lists)
                active_time_lists.update(child_active_time_lists)
                fix_storage_time_lists.update(child_fix_storage_time_lists)
                realized_time_lists.update(child_realized_time_lists)
        else:
            solves += roll_solves

        return (solves, complete_solves, active_time_lists, fix_storage_time_lists,
                realized_time_lists, parent_roll_lists)

    def _process_single_solve(self, solve, full_active_time_list, parent_info):
        """
        Handle single (non-rolling) solve logic.

        Args:
            solve: Name of the solve
            full_active_time_list: Full time list for this solve
            parent_info: ParentSolveInfo namedtuple

        Returns:
            Tuple of (solves, complete_solves, active_time_lists, fix_storage_time_lists,
                     realized_time_lists, parent_roll_lists)
        """
        solves = [solve]
        complete_solves = OrderedDict()
        active_time_lists = OrderedDict()
        invest_time_lists = OrderedDict()
        fix_storage_time_lists = OrderedDict()
        realized_time_lists = OrderedDict()
        parent_roll_lists = OrderedDict()

        complete_solves[solve] = solve
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

            # Get parent's scope: union of fix_storage and realized periods
            parent_scope_periods = (set(fix_storage_time_lists[solve].keys()) |
                                   set(realized_time_lists[solve].keys()))

            for contain_solve in contain_solves:
                child_parent_info = self.ParentSolveInfo(solve=solve, roll=solve)
                (child_solves, child_complete_solves, child_active_time_lists,
                 child_fix_storage_time_lists, child_realized_time_lists,
                 child_parent_roll_lists) = self._define_solve_recursive(
                    contain_solve, child_parent_info, parent_scope_periods, None, -1)

                solves += child_solves
                complete_solves.update(child_complete_solves)
                parent_roll_lists.update(child_parent_roll_lists)
                active_time_lists.update(child_active_time_lists)
                fix_storage_time_lists.update(child_fix_storage_time_lists)
                realized_time_lists.update(child_realized_time_lists)

        return (solves, complete_solves, active_time_lists, fix_storage_time_lists,
                realized_time_lists, parent_roll_lists)

    def _define_solve_recursive(self, solve, parent_info, parent_scope_periods=None,
                                start=None, duration=-1):
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
            Tuple of (solves, complete_solves, active_time_lists, fix_storage_time_lists,
                     realized_time_lists, parent_roll_lists)
        """
        new_name = solve
        if new_name not in self.state.solve.real_solves:
            self.state.solve.real_solves.append(new_name)

        if parent_info.solve:
            joint_current_solve_periods = list(set(self.state.solve.invest_periods[solve] + self.state.solve.fix_storage_periods[solve] + self.state.solve.realized_periods[solve]))
            current_solve_periods = [t[0] for t in joint_current_solve_periods]
            for current_solve_period in current_solve_periods:
                joint_parent_periods = list(set(self.state.solve.invest_periods[parent_info.solve] + self.state.solve.fix_storage_periods[parent_info.solve] + self.state.solve.realized_periods[parent_info.solve]))
                parent_period = set([t[0] for t in joint_parent_periods])
                if current_solve_period in parent_period:
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
                    # There should be only one parent 'period_from'
                    break

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
            # Process as single solve
            return self._process_single_solve(new_name, full_active_time_list, parent_info)

    def create_stochastic_periods(self, stochastic_branches, solves, complete_solves, active_time_lists, fix_storage_time_lists, realized_time_lists):
        """
        Apply stochastic branching to time periods.

        This function processes stochastic branches and creates branched versions of periods
        where multiple future scenarios diverge. Branches are added to active_time_lists but
        NOT to realized_time_lists or fix_storage_time_lists, since branches represent
        future scenarios that are optimized over but not committed/realized.

        Args:
            stochastic_branches: Branch configuration data
            solves: List of solve names
            complete_solves: Mapping of rolls to their complete solve
            active_time_lists: Dict of {solve: {period: timesteps}}
            fix_storage_time_lists: Dict of {solve: {period: timesteps}} for fixed storage
            realized_time_lists: Dict of {solve: {period: timesteps}} for realized results

        Returns:
            Tuple of (period__branch_lists, solve_branch__time_branch_lists, active_time_lists,
                     jump_lists, fix_storage_time_lists, realized_time_lists, branch_start_time_lists)
        """
        period__branch_lists = defaultdict(list)
        solve_branch__time_branch_lists = defaultdict(list)
        jump_lists = OrderedDict()
        branch_start_time_lists = defaultdict()

        for solve in solves:
            new_realized_time_list = OrderedDict()
            new_fix_storage_time_list = OrderedDict()
            new_active_time_list = OrderedDict()

            info = stochastic_branches[complete_solves[solve]]
            active_time_list = active_time_lists[solve]
            realized_time_list = realized_time_lists[solve]
            fix_storage_time_list = fix_storage_time_lists[solve]

            branched = False
            next_analysis_found = False
            branches = []
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
            if found_start == False and len(info) != 0:
                self.state.logger.error("A realized start time of the solve cannot be found from the stochastic_branches parameter. "+
                              "Check that stochastic_branches has a realized : yes, branch for the start of the solve" +
                               "and that the possible rolling_jump matches with the branch starts")
                sys.exit(-1)

            # Process each period to create branches
            for period, active_time in active_time_list.items():
                realized_end = None
                if not branched:
                    period__branch_lists[solve].append((period, period))
                    # Get all start times
                    start_times = defaultdict(list)
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
                                if float(branch__weight__real[1]) != 0.0 and branch != period and branch__weight__real[2] != "yes":
                                    # Branches get active time for optimization
                                    new_active_time_list[solve_branch] = active_time[0:]
                                    # But NOT realized or fix_storage (they're future scenarios)
                                    solve_branch__time_branch_lists[solve].append((solve_branch, branch))

                                period__branch_lists[solve].append((period, solve_branch))

                                # Get timesteps for stochastic tracking
                                for i in active_time[0:]:
                                    self.state.timeline.stochastic_timesteps[solve].append((solve_branch, i[0]))
                            break
                else:
                    # If the jump is longer than the period (continuation after branching)
                    for branch in branches:
                        solve_branch = period + "_" + branch
                        # Branches continue to get active time but not realized/fix_storage
                        period__branch_lists[solve].append((period, solve_branch))
                        solve_branch__time_branch_lists[solve].append((solve_branch, branch))
                        for i in active_time_list[period]:
                             self.state.timeline.stochastic_timesteps[solve].append((solve_branch, i[0]))

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
                if found == 0 and branch_start_time_lists[solve] != None:
                    for row in info:
                        if row[0] == branch_start_time_lists[solve][0] and row[2] == branch_start_time_lists[solve][1] and row[3] == 'yes':
                            found += 1
                            solve_branch__time_branch_lists[solve].append((period, row[1]))
                if (branch_start_time_lists[solve] != None and found == 0) or found > 1:
                    self.state.logger.error("Each period should have one and only one realized branch. Found: " + str(found) + "\n")
                    sys.exit(-1)

            # Update the time lists for this solve
            realized_time_lists[solve] = new_realized_time_list
            fix_storage_time_lists[solve] = new_fix_storage_time_list
            active_time_lists[solve] = new_active_time_list
            jump_lists[solve] = make_step_jump(new_active_time_list, period__branch_lists[solve], solve_branch__time_branch_lists[solve])

        return period__branch_lists, solve_branch__time_branch_lists, active_time_lists, jump_lists, fix_storage_time_lists, realized_time_lists, branch_start_time_lists

    def run_model(self):
        """
        first read the solve configuration from the input files, then for each solve write the files that are needed
        By that solve into disk. separate the reading into a separate step since the input files need knowledge of multiple solves.
        """
        active_time_lists = OrderedDict()
        jump_lists = OrderedDict()
        solve_period_history = defaultdict(list)
        fix_storage_time_lists = OrderedDict()
        realized_time_lists = OrderedDict()
        complete_solve= OrderedDict()
        parent_roll = OrderedDict()
        period__branch_lists = OrderedDict()
        branch_start_time_lists = defaultdict()
        all_solves=[]

        timer = time.perf_counter()

        try:
            os.mkdir('solve_data')
        except FileExistsError:
            print("solve_data folder existed")
        try:
            os.mkdir('output_raw')
        except FileExistsError:
            print("output_raw folder existed")
        try:
            os.mkdir('output_plots')
        except FileExistsError:
            print("output_plots folder existed")

        if not self.state.solve.model_solve:
            self.state.logger.error("No model. Make sure the 'model' class defines solves [Array].")
            sys.exit(-1)
        solves = next(iter(self.state.solve.model_solve.values()))
        if not solves:
            self.state.logger.error("No solves in model.")
            sys.exit(-1)
        
        for solve in solves:
            # Create ParentSolveInfo for top-level solve (no parent)
            parent_info = self.ParentSolveInfo(solve=None, roll=None)
            solve_solves, solve_complete_solve, solve_active_time_lists, solve_fix_storage_time_lists, solve_realized_time_lists, solve_parent_roll = self._define_solve_recursive(solve, parent_info, None, None, -1)
            all_solves += solve_solves
            complete_solve.update(solve_complete_solve)
            parent_roll.update(solve_parent_roll)
            active_time_lists.update(solve_active_time_lists)
            fix_storage_time_lists.update(solve_fix_storage_time_lists)
            realized_time_lists.update(copy.deepcopy(solve_realized_time_lists))

        # Leave only one realized timestep for each timestep in each period
        already_realized_timesteps = OrderedDict()
        for solve, realized_time_list in reversed(realized_time_lists.items()):
            for period, timesteps in realized_time_list.items():
                if period not in already_realized_timesteps.keys():
                    already_realized_timesteps[period] = []
                for i, timestep in enumerate(timesteps):
                    # If a timestep is found, then assume that all the rest of the timesteps are overlapping (can this fail?)
                    if timestep[0] in already_realized_timesteps[period]:
                        del realized_time_lists[solve][period][i:]
                        break
                    else:
                        already_realized_timesteps[period].append(timestep[0])
                if not timesteps:
                    del realized_time_lists[solve][period]

        for solve in self.state.solve.real_solves:
            #check that period__years_represented has only periods included in the solve
            new_years_represented = []
            for period__year in self.state.solve.solve_period_years_represented[solve]:
                if any(period__year[0] == period__timeset[0] for period__timeset in self.state.solve.timesets_used_by_solves[solve]):
                    new_years_represented.append(period__year)
            self.state.solve.solve_period_years_represented[solve] = new_years_represented
            # get period_history from earlier solves
            for solve_2 in self.state.solve.real_solves:
                if solve_2 == solve:
                    break
                # Combine all period tuples for solve_2 from all period dicts
                all_period_tuples = (
                    self.state.solve.realized_periods.get(solve_2, []) +
                    self.state.solve.invest_periods.get(solve_2, []) +
                    self.state.solve.fix_storage_periods.get(solve_2, []) +
                    self.state.solve.realized_invest_periods.get(solve_2, [])
                )
                for period_tuple in all_period_tuples:
                    this_solve = self.state.solve.solve_period_years_represented[solve_2]
                    for period in this_solve:
                        if period[0] == period_tuple[0] and not any(period[0] == sublist[0] for sublist in solve_period_history[solve]):
                            solve_period_history[solve].append((period[0], period[1]))
            # get period_history from this solve
            for period__year in self.state.solve.solve_period_years_represented[solve]:
                if not any(period__year[0]== sublist[0] for sublist in solve_period_history[solve]):
                    solve_period_history[solve].append((period__year[0], period__year[1]))
            #if not defined, all the periods have the value 1
            if not self.state.solve.solve_period_years_represented[solve]:
                for period__timeset in self.state.solve.timesets_used_by_solves[solve]:
                    if not any(period__timeset[0]== sublist[0] for sublist in solve_period_history[solve]):
                        solve_period_history[solve].append((period__timeset[0], 1))

        period__branch_lists, solve_branch__time_branch_lists, active_time_lists, jump_lists, fix_storage_time_lists, realized_time_lists, branch_start_time_lists = \
            self.create_stochastic_periods(self.state.solve.stochastic_branches, all_solves, complete_solve, active_time_lists, fix_storage_time_lists, realized_time_lists)

        for solve in active_time_lists.keys():
            for period in active_time_lists[solve]:
                if (period,period) in period__branch_lists[solve] and not any(period== sublist[0] for sublist in solve_period_history[complete_solve[solve]]):
                    self.state.logger.error(f"The years_represented is defined, but not to all of the periods ({period}) in the solve")
                    sys.exit(-1)

        timing = time.perf_counter() - timer
        print(f"--- Pre-processing of data: {timing:.4f} seconds ---")
        with open("solve_data/solve_progress.csv", "a") as solve_progress:
            solve_progress.write(',,solve,write_solve_input,setup,total_obj_cost,balance,reserves,rest,constraints,glpsol_input,solver,' \
                'setup2,total_obj_cost2,balance2,reserves2,rest2,constraints2,r_solution,w_raw,w_capacity,glpsol_output,\n')
        timer = timer + timing

        solver = SolverRunner(self.state)
        first = True
        previous_complete_solve = None
        for i, solve in enumerate(all_solves):
            timer_in_solve = time.perf_counter()

            self.state.logger.info("Creating timelines for solve " + solve + " (" + str(i) + ")")
            complete_active_time_lists = get_active_time(complete_solve[solve], self.state.solve.timesets_used_by_solves, self.state.timeline.timeset_durations, self.state.timeline.timelines, self.state.timeline.timesets__timeline)

            # Build a combined period__timeset list that includes history periods
            period__timesets_with_history = list(self.state.solve.timesets_used_by_solves[complete_solve[solve]])
            current_periods = {pt[0] for pt in period__timesets_with_history}

            # Determine the timeset to use for history periods (use the timeset from the current solve)
            # If the current solve has multiple timesets, prefer the first one
            current_timeset = period__timesets_with_history[0][1] if period__timesets_with_history else None

            # Add history periods from solve_period_history
            for history_period, _ in solve_period_history[complete_solve[solve]]:
                if history_period not in current_periods:
                    # Use the same timeset as the current solve for history periods
                    if current_timeset:
                        period__timesets_with_history.append((history_period, current_timeset))
                        current_periods.add(history_period)

            solve_writers.write_full_timelines(self.state.timeline.stochastic_timesteps[solve], period__timesets_with_history, self.state.timeline.timesets__timeline, self.state.timeline.timelines, 'solve_data/steps_in_timeline.csv')
            solve_writers.write_active_timelines(active_time_lists[solve], 'solve_data/steps_in_use.csv')
            solve_writers.write_active_timelines(complete_active_time_lists, 'solve_data/steps_complete_solve.csv', complete=True)
            solve_writers.write_step_jump(jump_lists[solve])
            solve_writers.write_timesets(self.state.solve.timesets_used_by_solves, self.state.timeline.timesets__timeline)
            self.state.logger.info("Creating period data")
            solve_writers.write_period_years(period__branch_lists[solve], solve_period_history[complete_solve[solve]], 'solve_data/period_with_history.csv')
            solve_writers.write_periods(complete_solve[solve], self.state.solve.realized_invest_periods, 'solve_data/realized_invest_periods_of_current_solve.csv')
            #assume that if realized_invest_periods is not defined,but the invest_periods and realized_periods are defined, use realized_periods also as the realized_invest_periods
            if not self.state.solve.realized_invest_periods[complete_solve[solve]] and self.state.solve.invest_periods[complete_solve[solve]] and self.state.solve.realized_periods[complete_solve[solve]]:
                solve_writers.write_periods(complete_solve[solve], self.state.solve.realized_periods, 'solve_data/realized_invest_periods_of_current_solve.csv')
            solve_writers.write_periods(complete_solve[solve], self.state.solve.invest_periods, 'solve_data/invest_periods_of_current_solve.csv')
            solve_writers.write_years_represented(period__branch_lists[solve], self.state.solve.solve_period_years_represented[complete_solve[solve]], 'solve_data/p_years_represented.csv')
            solve_writers.write_period_years(period__branch_lists[solve], self.state.solve.solve_period_years_represented[complete_solve[solve]], 'solve_data/p_discount_years.csv')
            solve_writers.write_current_solve(solve, 'solve_data/solve_current.csv')
            solve_writers.write_hole_multiplier(solve, self.state.solve.hole_multipliers, 'solve_data/solve_hole_multiplier.csv')
            solve_writers.write_first_steps(active_time_lists[solve], 'solve_data/first_timesteps.csv')
            solve_writers.write_last_steps(active_time_lists[solve], 'solve_data/last_timesteps.csv')
            solve_writers.write_last_realized_step(active_time_lists[solve], complete_solve[solve], self.state.solve.realized_periods.get(complete_solve[solve], []), 'solve_data/last_realized_timestep.csv')
            self.state.logger.info("Create realized timeline")
            solve_writers.write_realized_dispatch(realized_time_lists[solve], complete_solve[solve], self.state.solve.realized_periods.get(complete_solve[solve], []))
            solve_writers.write_fix_storage_timesteps(fix_storage_time_lists[solve], complete_solve[solve], self.state.solve.fix_storage_periods.get(complete_solve[solve], []))
            solve_writers.write_delayed_durations(active_time_lists[solve], complete_solve[solve], self.state.solve.delay_durations)
            self.state.logger.info("Possible stochastics")
            solve_writers.write_branch__period_relationship(period__branch_lists[solve], 'solve_data/period__branch.csv')
            solve_writers.write_all_branches(period__branch_lists, solve_branch__time_branch_lists[solve], self.state.logger)
            solve_writers.write_branch_weights_and_map(complete_solve[solve], active_time_lists[solve], solve_branch__time_branch_lists[solve], branch_start_time_lists[solve], period__branch_lists[solve], self.state.solve.stochastic_branches)
            solve_writers.write_first_and_last_periods(active_time_lists[solve], self.state.solve.timesets_used_by_solves[complete_solve[solve]], period__branch_lists[solve])
            separate_period_and_timeseries_data(self.state.timeline.timelines, self.state.solve.timesets_used_by_solves)

            #check if the upper level fixes storages
            if [complete_solve[solve]] in self.state.solve.contains_solves.values() and complete_solve[parent_roll[solve]] in self.state.solve.fix_storage_periods:  # check that the parent_roll exists and has storage fixing
                storage_fix_values_exist = True
            else:
                storage_fix_values_exist = False
            if storage_fix_values_exist:
                self.state.logger.info("Nested timeline matching")
                self.write_timeline_matching_map(active_time_lists[parent_roll[solve]], active_time_lists[solve], complete_solve[parent_roll[solve]], complete_solve[solve], period__branch_lists[solve])
            else:
                with open("solve_data/timeline_matching_map.csv", 'w') as realfile:
                    realfile.write("period,step,upper_step\n")
            #if timeline created from new step_duration, all timeseries have to be averaged or summed for the new timestep
            if previous_complete_solve != complete_solve[solve]:
                self.state.logger.info("Aggregating timeline and parameters for the new step size")
                self.state.timeline.create_averaged_timeseries(complete_solve[solve], self.state.solve, self.state.logger)
            previous_complete_solve = complete_solve[solve]
            if solve in self.state.solve.first_of_complete_solve:
                first_of_nested_level = True
            else:
                first_of_nested_level = False
            if solve in self.state.solve.last_of_solve:
                last_of_nested_level = True
            else:
                last_of_nested_level = False
            #if multiple storage solve levels, get the storage fix of the upper level, (not the fix of the previous roll):
            if storage_fix_values_exist:
                self.state.logger.info("Fetching storage parameters from the upper solve")
                shutil.copy("solve_data/fix_storage_quantity_"+ complete_solve[parent_roll[solve]]+".csv", "solve_data/fix_storage_quantity.csv")
                shutil.copy("solve_data/fix_storage_price_"+ complete_solve[parent_roll[solve]]+".csv", "solve_data/fix_storage_price.csv")
                shutil.copy("solve_data/fix_storage_usage_"+ complete_solve[parent_roll[solve]]+".csv", "solve_data/fix_storage_usage.csv")

            solve_writers.write_solve_status(first_of_nested_level, last_of_nested_level, nested=True)
            last = i == len(solves) - 1
            solve_writers.write_solve_status(first, last)
            if i == 0:
                first = False
                solve_writers.write_empty_investment_file()
                solve_writers.write_empty_storage_fix_file()
                solve_writers.write_headers_for_empty_output_files('solve_data/costs_discounted.csv', 'param_costs,costs_discounted')
                solve_writers.write_headers_for_empty_output_files('solve_data/co2.csv', 'param_co2,model_wide')
                solve_writers.write_headers_for_empty_output_files('solve_data/period_capacity.csv', 'period')
            self.state.logger.info("Starting model creation")

            with open("solve_data/solve_progress.csv", "a") as solve_progress:
                solve_progress.write(',,' + solve + ',' + str(round(time.perf_counter() - timer_in_solve,4)))
            
            exit_status = solver.run(complete_solve[solve])
            if exit_status == 0:
                self.state.logger.info('Success!')
                print("-------------------------------------------------------------------------------------------\n\n")
            else:
                self.state.logger.error(f'Error: {exit_status}')
                sys.exit(-1)

            #if multiple storage solve levels, save the storage fix of this level:
            if complete_solve[solve] in self.state.solve.fix_storage_periods:
                shutil.copy("solve_data/fix_storage_quantity.csv","solve_data/fix_storage_quantity_"+ complete_solve[solve]+".csv")
                shutil.copy("solve_data/fix_storage_price.csv", "solve_data/fix_storage_price_"+ complete_solve[solve]+".csv")
                shutil.copy("solve_data/fix_storage_usage.csv","solve_data/fix_storage_usage_"+ complete_solve[solve]+".csv")

        if len(self.state.solve.model_solve) > 1:
            self.state.logger.error(
                f'Trying to run more than one model - not supported. The results of the first model are retained.')
            sys.exit(-1)
        return 0

    def write_input(self, input_db_url, scenario_name=None) -> None:
        input_writer.write_input(input_db_url, scenario_name, self.state.logger)


def main():
    logging.basicConfig(level=logging.INFO)
    logging.error("Run using run_flextool.py in the root of FlexTool")
    sys.exit(-1)

if __name__ == '__main__':
    main()
