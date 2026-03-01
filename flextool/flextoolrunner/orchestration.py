"""Main solve loop — coordinates all modules to run the full model.

Entry point: run_model(state, solver) -> int
"""

import copy
import os
import shutil
import sys
import time
from collections import OrderedDict, defaultdict

from flextool.flextoolrunner.runner_state import RunnerState
from flextool.flextoolrunner.solver_runner import SolverRunner
from flextool.flextoolrunner.rolling_solver import RollingSolver, ParentSolveInfo
from flextool.flextoolrunner.stochastic import StochasticSolver
from flextool.flextoolrunner.timeline_config import get_active_time, separate_period_and_timeseries_data
from flextool.flextoolrunner import solve_writers


def run_model(state: RunnerState, solver: SolverRunner) -> int:
    """Run the full solve loop.

    Reads the solve configuration, builds the solve execution order via
    RollingSolver, applies stochastic branching, then for each solve writes
    the required CSV files and invokes the solver.

    Args:
        state: Cross-cutting runner state (paths, solve config, timeline config, logger).
        solver: SolverRunner instance for invoking external solver binaries.

    Returns:
        0 on success; calls sys.exit(-1) on failure.
    """
    active_time_lists: OrderedDict = OrderedDict()
    jump_lists: OrderedDict = OrderedDict()
    solve_period_history: defaultdict[str, list] = defaultdict(list)
    fix_storage_time_lists: OrderedDict = OrderedDict()
    realized_time_lists: OrderedDict = OrderedDict()
    complete_solve: OrderedDict = OrderedDict()
    parent_roll: OrderedDict = OrderedDict()
    period__branch_lists: OrderedDict = OrderedDict()
    branch_start_time_lists: defaultdict = defaultdict()
    all_solves: list = []

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

    if not state.solve.model_solve:
        state.logger.error("No model. Make sure the 'model' class defines solves [Array].")
        sys.exit(-1)
    solves = next(iter(state.solve.model_solve.values()))
    if not solves:
        state.logger.error("No solves in model.")
        sys.exit(-1)

    rolling_solver = RollingSolver(state)
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
    already_realized_timesteps: OrderedDict = OrderedDict()
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

    for solve in state.solve.real_solves:
        #check that period__years_represented has only periods included in the solve
        new_years_represented = []
        for period__year in state.solve.solve_period_years_represented[solve]:
            if any(period__year[0] == period__timeset[0] for period__timeset in state.solve.timesets_used_by_solves[solve]):
                new_years_represented.append(period__year)
        state.solve.solve_period_years_represented[solve] = new_years_represented
        # get period_history from earlier solves
        for solve_2 in state.solve.real_solves:
            if solve_2 == solve:
                break
            # Combine all period tuples for solve_2 from all period dicts
            all_period_tuples = (
                state.solve.realized_periods.get(solve_2, []) +
                state.solve.invest_periods.get(solve_2, []) +
                state.solve.fix_storage_periods.get(solve_2, []) +
                state.solve.realized_invest_periods.get(solve_2, [])
            )
            for period_tuple in all_period_tuples:
                this_solve = state.solve.solve_period_years_represented[solve_2]
                for period in this_solve:
                    if period[0] == period_tuple[0] and not any(period[0] == sublist[0] for sublist in solve_period_history[solve]):
                        solve_period_history[solve].append((period[0], period[1]))
        # get period_history from this solve
        for period__year in state.solve.solve_period_years_represented[solve]:
            if not any(period__year[0]== sublist[0] for sublist in solve_period_history[solve]):
                solve_period_history[solve].append((period__year[0], period__year[1]))
        #if not defined, all the periods have the value 1
        if not state.solve.solve_period_years_represented[solve]:
            for period__timeset in state.solve.timesets_used_by_solves[solve]:
                if not any(period__timeset[0]== sublist[0] for sublist in solve_period_history[solve]):
                    solve_period_history[solve].append((period__timeset[0], 1))

    stochastic_solver = StochasticSolver(state)
    period__branch_lists, solve_branch__time_branch_lists, active_time_lists, jump_lists, fix_storage_time_lists, realized_time_lists, branch_start_time_lists = \
        stochastic_solver.create_stochastic_periods(state.solve.stochastic_branches, all_solves, complete_solve, active_time_lists, fix_storage_time_lists, realized_time_lists)

    for solve in active_time_lists.keys():
        for period in active_time_lists[solve]:
            if (period,period) in period__branch_lists[solve] and not any(period== sublist[0] for sublist in solve_period_history[complete_solve[solve]]):
                state.logger.error(f"The years_represented is defined, but not to all of the periods ({period}) in the solve")
                sys.exit(-1)

    timing = time.perf_counter() - timer
    print(f"--- Pre-processing of data: {timing:.4f} seconds ---")
    with open("solve_data/solve_progress.csv", "a") as solve_progress:
        solve_progress.write(',,solve,write_solve_input,setup,total_obj_cost,balance,reserves,rest,constraints,glpsol_input,solver,' \
            'setup2,total_obj_cost2,balance2,reserves2,rest2,constraints2,r_solution,w_raw,w_capacity,glpsol_output,\n')
    timer = timer + timing

    first = True
    previous_complete_solve = None
    for i, solve in enumerate(all_solves):
        timer_in_solve = time.perf_counter()

        state.logger.info("Creating timelines for solve " + solve + " (" + str(i) + ")")
        complete_active_time_lists = get_active_time(complete_solve[solve], state.solve.timesets_used_by_solves, state.timeline.timeset_durations, state.timeline.timelines, state.timeline.timesets__timeline)

        # Build a combined period__timeset list that includes history periods
        period__timesets_with_history = list(state.solve.timesets_used_by_solves[complete_solve[solve]])
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

        solve_writers.write_full_timelines(state.timeline.stochastic_timesteps[solve], period__timesets_with_history, state.timeline.timesets__timeline, state.timeline.timelines, 'solve_data/steps_in_timeline.csv')
        solve_writers.write_active_timelines(active_time_lists[solve], 'solve_data/steps_in_use.csv')
        solve_writers.write_active_timelines(complete_active_time_lists, 'solve_data/steps_complete_solve.csv', complete=True)
        solve_writers.write_step_jump(jump_lists[solve])
        solve_writers.write_timesets(state.solve.timesets_used_by_solves, state.timeline.timesets__timeline)
        state.logger.info("Creating period data")
        solve_writers.write_period_years(period__branch_lists[solve], solve_period_history[complete_solve[solve]], 'solve_data/period_with_history.csv')
        solve_writers.write_periods(complete_solve[solve], state.solve.realized_invest_periods, 'solve_data/realized_invest_periods_of_current_solve.csv')
        #assume that if realized_invest_periods is not defined,but the invest_periods and realized_periods are defined, use realized_periods also as the realized_invest_periods
        if not state.solve.realized_invest_periods[complete_solve[solve]] and state.solve.invest_periods[complete_solve[solve]] and state.solve.realized_periods[complete_solve[solve]]:
            solve_writers.write_periods(complete_solve[solve], state.solve.realized_periods, 'solve_data/realized_invest_periods_of_current_solve.csv')
        solve_writers.write_periods(complete_solve[solve], state.solve.invest_periods, 'solve_data/invest_periods_of_current_solve.csv')
        solve_writers.write_years_represented(period__branch_lists[solve], state.solve.solve_period_years_represented[complete_solve[solve]], 'solve_data/p_years_represented.csv')
        solve_writers.write_period_years(period__branch_lists[solve], state.solve.solve_period_years_represented[complete_solve[solve]], 'solve_data/p_discount_years.csv')
        solve_writers.write_current_solve(solve, 'solve_data/solve_current.csv')
        solve_writers.write_hole_multiplier(solve, state.solve.hole_multipliers, 'solve_data/solve_hole_multiplier.csv')
        solve_writers.write_first_steps(active_time_lists[solve], 'solve_data/first_timesteps.csv')
        solve_writers.write_last_steps(active_time_lists[solve], 'solve_data/last_timesteps.csv')
        solve_writers.write_last_realized_step(active_time_lists[solve], complete_solve[solve], state.solve.realized_periods.get(complete_solve[solve], []), 'solve_data/last_realized_timestep.csv')
        state.logger.info("Create realized timeline")
        solve_writers.write_realized_dispatch(realized_time_lists[solve], complete_solve[solve], state.solve.realized_periods.get(complete_solve[solve], []))
        solve_writers.write_fix_storage_timesteps(fix_storage_time_lists[solve], complete_solve[solve], state.solve.fix_storage_periods.get(complete_solve[solve], []))
        solve_writers.write_delayed_durations(active_time_lists[solve], complete_solve[solve], state.solve.delay_durations)
        state.logger.info("Possible stochastics")
        solve_writers.write_branch__period_relationship(period__branch_lists[solve], 'solve_data/period__branch.csv')
        solve_writers.write_all_branches(period__branch_lists, solve_branch__time_branch_lists[solve], state.logger)
        solve_writers.write_branch_weights_and_map(complete_solve[solve], active_time_lists[solve], solve_branch__time_branch_lists[solve], branch_start_time_lists[solve], period__branch_lists[solve], state.solve.stochastic_branches)
        solve_writers.write_first_and_last_periods(active_time_lists[solve], state.solve.timesets_used_by_solves[complete_solve[solve]], period__branch_lists[solve])
        separate_period_and_timeseries_data(state.timeline.timelines, state.solve.timesets_used_by_solves)

        #check if the upper level fixes storages
        if [complete_solve[solve]] in state.solve.contains_solves.values() and complete_solve[parent_roll[solve]] in state.solve.fix_storage_periods:  # check that the parent_roll exists and has storage fixing
            storage_fix_values_exist = True
        else:
            storage_fix_values_exist = False
        if storage_fix_values_exist:
            state.logger.info("Nested timeline matching")
            stochastic_solver.write_timeline_matching_map(active_time_lists[parent_roll[solve]], active_time_lists[solve], complete_solve[parent_roll[solve]], complete_solve[solve], period__branch_lists[solve])
        else:
            with open("solve_data/timeline_matching_map.csv", 'w') as realfile:
                realfile.write("period,step,upper_step\n")
        #if timeline created from new step_duration, all timeseries have to be averaged or summed for the new timestep
        if previous_complete_solve != complete_solve[solve]:
            state.logger.info("Aggregating timeline and parameters for the new step size")
            state.timeline.create_averaged_timeseries(complete_solve[solve], state.solve, state.logger)
        previous_complete_solve = complete_solve[solve]
        if solve in state.solve.first_of_complete_solve:
            first_of_nested_level = True
        else:
            first_of_nested_level = False
        if solve in state.solve.last_of_solve:
            last_of_nested_level = True
        else:
            last_of_nested_level = False
        #if multiple storage solve levels, get the storage fix of the upper level, (not the fix of the previous roll):
        if storage_fix_values_exist:
            state.logger.info("Fetching storage parameters from the upper solve")
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
        state.logger.info("Starting model creation")

        with open("solve_data/solve_progress.csv", "a") as solve_progress:
            solve_progress.write(',,' + solve + ',' + str(round(time.perf_counter() - timer_in_solve,4)))

        exit_status = solver.run(complete_solve[solve])
        if exit_status == 0:
            state.logger.info('Success!')
            print("-------------------------------------------------------------------------------------------\n\n")
        else:
            state.logger.error(f'Error: {exit_status}')
            sys.exit(-1)

        #if multiple storage solve levels, save the storage fix of this level:
        if complete_solve[solve] in state.solve.fix_storage_periods:
            shutil.copy("solve_data/fix_storage_quantity.csv","solve_data/fix_storage_quantity_"+ complete_solve[solve]+".csv")
            shutil.copy("solve_data/fix_storage_price.csv", "solve_data/fix_storage_price_"+ complete_solve[solve]+".csv")
            shutil.copy("solve_data/fix_storage_usage.csv","solve_data/fix_storage_usage_"+ complete_solve[solve]+".csv")

    if len(state.solve.model_solve) > 1:
        state.logger.error(
            f'Trying to run more than one model - not supported. The results of the first model are retained.')
        sys.exit(-1)
    return 0
