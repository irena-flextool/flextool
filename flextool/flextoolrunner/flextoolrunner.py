import time
import logging
import copy
import sys
import os
import shutil
import spinedb_api as api
from spinedb_api import DatabaseMapping
# from spinedb_api.filters.scenario_filter import scenario_filter_config, scenario_filter_from_dict
from pathlib import Path
from collections import OrderedDict
from collections import defaultdict

from flextool.flextoolrunner.db_reader import check_version
from flextool.flextoolrunner import input_writer
from flextool.flextoolrunner import solve_writers
from flextool.flextoolrunner.solve_config import SolveConfig
from flextool.flextoolrunner.timeline_config import (
    TimelineConfig,
    get_active_time,
    separate_period_and_timeseries_data,
)
from flextool.flextoolrunner.runner_state import PathConfig, RunnerState
from flextool.flextoolrunner.solver_runner import SolverRunner
from flextool.flextoolrunner.stochastic import StochasticSolver
from flextool.flextoolrunner.rolling_solver import RollingSolver, ParentSolveInfo

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
        
        rolling_solver = RollingSolver(self.state)
        for solve in solves:
            # Create ParentSolveInfo for top-level solve (no parent)
            parent_info = ParentSolveInfo(solve=None, roll=None)
            solve_solves, solve_complete_solve, solve_active_time_lists, solve_fix_storage_time_lists, solve_realized_time_lists, solve_parent_roll = rolling_solver.define_solve_recursive(solve, parent_info, None, None, -1)
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

        stochastic_solver = StochasticSolver(self.state)
        period__branch_lists, solve_branch__time_branch_lists, active_time_lists, jump_lists, fix_storage_time_lists, realized_time_lists, branch_start_time_lists = \
            stochastic_solver.create_stochastic_periods(self.state.solve.stochastic_branches, all_solves, complete_solve, active_time_lists, fix_storage_time_lists, realized_time_lists)

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
                stochastic_solver.write_timeline_matching_map(active_time_lists[parent_roll[solve]], active_time_lists[solve], complete_solve[parent_roll[solve]], complete_solve[solve], period__branch_lists[solve])
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
