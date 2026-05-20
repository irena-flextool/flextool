"""Main solve loop — coordinates all modules to run the full model.

Entry point: run_model(state, solver) -> int
"""

import copy
import csv
import os
import shutil
import time
from collections import defaultdict

from flextool.flextoolrunner.blocks import write_block_data_for_solve
# Step 2.5 — preprocessing package deleted; per-solve orchestrator
# moved into engine_polars natively.
from flextool.engine_polars import _writer_solve_time as preprocessing_solve_time
from flextool.flextoolrunner.solve_handoff import capture_post_solve
from flextool.flextoolrunner.minimum_time import write_minimum_time_data
from flextool.flextoolrunner.runner_state import RunnerState, FlexToolConfigError, FlexToolSolveError, SolveResult
from flextool.flextoolrunner.solver_runner import SolverRunner
from flextool.flextoolrunner.recursive_solves import RecursiveSolveBuilder, ParentSolveInfo
from flextool.flextoolrunner.scaling import (
    analyze_solve,
    maybe_auto_apply_row_scaling,
    write_scaling_analysis_json,
)
from flextool.flextoolrunner.scaling_report import write_scaling_report
from flextool.flextoolrunner.stochastic import StochasticSolver
from flextool.flextoolrunner.timeline_config import get_active_time, make_period_block, separate_period_and_timeseries_data
from flextool.flextoolrunner import solve_writers


def run_model(state: RunnerState, solver: SolverRunner) -> int:
    """Run the full solve loop.

    Reads the solve configuration, builds the solve execution order via
    RecursiveSolveBuilder, applies stochastic branching, then for each solve writes
    the required CSV files and invokes the solver.

    Args:
        state: Cross-cutting runner state (paths, solve config, timeline config, logger).
        solver: SolverRunner instance for invoking external solver binaries.

    Returns:
        0 on success.

    Raises:
        FlexToolConfigError: On configuration/data errors.
        FlexToolSolveError: On solver execution errors.
    """
    active_time_lists: dict = {}
    jump_lists: dict = {}
    solve_period_history: defaultdict[str, list] = defaultdict(list)
    fix_storage_time_lists: dict = {}
    realized_time_lists: dict = {}
    complete_solve: dict = {}
    parent_roll: dict = {}
    period__branch_lists: dict = {}
    branch_start_time_lists: defaultdict = defaultdict()
    all_solves: list = []

    timer = time.perf_counter()

    wf = state.paths.work_folder
    try:
        os.mkdir(wf / 'solve_data')
    except FileExistsError:
        state.logger.debug("solve_data folder existed")
    # output_raw/ holds the intermediate HiGHS → parquet dumps (both
    # pathways) plus the legacy phase-3 CSVs (``--use-old-raw-csv`` only).
    try:
        os.mkdir(wf / 'output_raw')
    except FileExistsError:
        state.logger.debug("output_raw folder existed")
    try:
        os.mkdir(wf / 'output_plots')
    except FileExistsError:
        state.logger.debug("output_plots folder existed")

    if not state.solve.model_solve:
        message = "No model. Make sure the 'model' class defines solves [Array]."
        state.logger.error(message)
        raise FlexToolConfigError(message)
    solves = next(iter(state.solve.model_solve.values()))
    if not solves:
        message = "No solves in model."
        state.logger.error(message)
        raise FlexToolConfigError(message)

    solve_builder = RecursiveSolveBuilder(state)
    for solve in solves:
        # Create ParentSolveInfo for top-level solve (no parent)
        parent_info = ParentSolveInfo(solve=None, roll=None)
        result = solve_builder.define_solve_recursive(solve, parent_info, None, None, -1)
        all_solves += result.solves
        complete_solve.update(result.complete_solves)
        parent_roll.update(result.parent_roll_lists)
        active_time_lists.update(result.active_time_lists)
        fix_storage_time_lists.update(result.fix_storage_time_lists)
        realized_time_lists.update(copy.deepcopy(result.realized_time_lists))

    # Leave only one realized timestep for each timestep in each period
    already_realized_timesteps: dict[str, set[str]] = {}
    for solve, realized_time_list in reversed(realized_time_lists.items()):
        for period, timesteps in list(realized_time_list.items()):
            if period not in already_realized_timesteps:
                already_realized_timesteps[period] = set()
            for i, timestep in enumerate(timesteps):
                # If a timestep is found, then assume that all the rest of the timesteps are overlapping (can this fail?)
                if timestep.timestep in already_realized_timesteps[period]:
                    del realized_time_lists[solve][period][i:]
                    break
                else:
                    already_realized_timesteps[period].add(timestep.timestep)
            if not timesteps:
                del realized_time_lists[solve][period]

    # Build period history incrementally instead of O(N²) predecessor scanning
    cumulative_contributions: list[tuple[str, float]] = []
    cumulative_period_names: set[str] = set()

    for solve in state.solve.real_solves:
        #check that period__years_represented has only periods included in the solve
        timeset_periods = {pt[0] for pt in state.solve.timesets_used_by_solves[solve]}
        state.solve.solve_period_years_represented[solve] = [
            py for py in state.solve.solve_period_years_represented[solve]
            if py[0] in timeset_periods
        ]
        # get period_history from earlier solves (already accumulated)
        history_period_names: set[str] = set()
        for period_name, years in cumulative_contributions:
            if period_name not in history_period_names:
                solve_period_history[solve].append((period_name, years))
                history_period_names.add(period_name)
        # get period_history from this solve
        for period__year in state.solve.solve_period_years_represented[solve]:
            if period__year[0] not in history_period_names:
                solve_period_history[solve].append((period__year[0], period__year[1]))
                history_period_names.add(period__year[0])
        #if not defined, all the periods have the value 1
        if not state.solve.solve_period_years_represented[solve]:
            for period__timeset in state.solve.timesets_used_by_solves[solve]:
                if period__timeset[0] not in history_period_names:
                    solve_period_history[solve].append((period__timeset[0], 1))
                    history_period_names.add(period__timeset[0])
        # Compute this solve's contributions and add to cumulative for next solves
        period_dict_names = {
            t[0] for t in (
                state.solve.realized_periods.get(solve, []) +
                state.solve.invest_periods.get(solve, []) +
                state.solve.fix_storage_periods.get(solve, []) +
                state.solve.realized_invest_periods.get(solve, [])
            )
        }
        for period in state.solve.solve_period_years_represented[solve]:
            if period[0] in period_dict_names and period[0] not in cumulative_period_names:
                cumulative_contributions.append((period[0], period[1]))
                cumulative_period_names.add(period[0])

    stochastic_solver = StochasticSolver(state)
    period__branch_lists, solve_branch__time_branch_lists, active_time_lists, jump_lists, fix_storage_time_lists, realized_time_lists, branch_start_time_lists = \
        stochastic_solver.create_stochastic_periods(state.solve.stochastic_branches, all_solves, complete_solve, active_time_lists, fix_storage_time_lists, realized_time_lists)

    for solve in active_time_lists.keys():
        for period in active_time_lists[solve]:
            if (period,period) in period__branch_lists[solve] and not any(period== sublist[0] for sublist in solve_period_history[complete_solve[solve]]):
                message = f"The years_represented is defined, but not to all of the periods ({period}) in the solve"
                state.logger.error(message)
                raise FlexToolConfigError(message)

    timing = time.perf_counter() - timer
    state.logger.debug(f"--- Pre-processing of data: {timing:.4f} seconds ---")
    if state.timing_recorder is not None:
        state.timing_recorder.record('preprocessing_global', seconds=timing,
                                     t_start=time.perf_counter() - timing)
    timer = timer + timing

    # Step 2.5-G Phase A — Provider-aware pdt → {pd, pt} split.  The
    # cascade-input Provider on ``state`` carries the seeded
    # ``input/pdt_commodity`` / ``input/pdt_group`` frames (Phase 2.5-E
    # migrated ``_PARAMETER_SPECS`` off disk).  ``write_input`` populates
    # the slot before ``run_model`` fires; we bootstrap a fresh empty
    # Provider for the corner case where the caller invokes ``run_model``
    # standalone (no ``write_input`` first), in which case
    # ``_provider_open`` falls back to disk for the source frame.
    cascade_input_provider = getattr(state, "cascade_input_provider", None)
    if cascade_input_provider is None:
        from flextool.engine_polars._flex_data_provider import FlexDataProvider
        cascade_input_provider = FlexDataProvider()
        state.cascade_input_provider = cascade_input_provider
    separate_period_and_timeseries_data(
        state.timeline.timelines,
        state.solve.timesets_used_by_solves,
        provider=cascade_input_provider,
        work_folder=wf,
    )

    # Load minimum time data from input/ CSVs for precomputation
    process_min_uptime: dict[str, float] = {}
    process_min_downtime: dict[str, float] = {}
    process_min_uptime_set: set[str] = set()
    process_min_downtime_set: set[str] = set()

    # Read which processes have min uptime/downtime enabled
    min_uptime_csv = wf / "input" / "process_min_uptime.csv"
    if min_uptime_csv.exists():
        with open(min_uptime_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                process_min_uptime_set.add(row["process_min_uptime"])

    min_downtime_csv = wf / "input" / "process_min_downtime.csv"
    if min_downtime_csv.exists():
        with open(min_downtime_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                process_min_downtime_set.add(row["process_min_downtime"])

    # Read the actual min_uptime/min_downtime values from p_process.csv
    p_process_csv = wf / "input" / "p_process.csv"
    if p_process_csv.exists() and (process_min_uptime_set or process_min_downtime_set):
        with open(p_process_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                proc = row["process"]
                param = row["processParam"]
                val = float(row["p_process"]) if row["p_process"] else 0.0
                if param == "min_uptime" and proc in process_min_uptime_set and val > 0:
                    process_min_uptime[proc] = val
                elif param == "min_downtime" and proc in process_min_downtime_set and val > 0:
                    process_min_downtime[proc] = val

    # write_timesets is solve-loop-invariant: its args (timesets_used_by_solves,
    # timesets__timeline) come from `state` set up once in `runner.write_input()`,
    # so the output CSVs (input/timesets_in_use.csv, input/timesets__timeline.csv)
    # are identical every roll. Hoisted out of the per-solve loop to write once.
    solve_writers.write_timesets(state.solve.timesets_used_by_solves,
                                 state.timeline.timesets__timeline,
                                 work_folder=wf)

    first = True
    previous_complete_solve = None
    # Track the most-recent solve whose post-solve capture deposited a
    # ``SolveHandoff`` into ``state.handoffs`` — that entry feeds the
    # *next* iteration's per-solve preprocessing as ``prior_handoff``.
    # When ``state.handoffs is None`` the consume side is a no-op and
    # preprocessing falls back to the file-based path.
    last_captured_solve: str | None = None
    cached_complete_active_time_lists: dict = {}
    for i, solve in enumerate(all_solves):
        timer_in_solve = time.perf_counter()

        state.logger.debug("Creating timelines for solve " + solve + " (" + str(i) + ")")
        cs = complete_solve[solve]
        if cs not in cached_complete_active_time_lists:
            cached_complete_active_time_lists[cs] = get_active_time(cs, state.solve.timesets_used_by_solves, state.timeline.timeset_durations, state.timeline.timelines, state.timeline.timesets__timeline)
        complete_active_time_lists = cached_complete_active_time_lists[cs]

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

        solve_writers.write_full_timelines(state.timeline.stochastic_timesteps[solve], period__timesets_with_history, state.timeline.timesets__timeline, state.timeline.timelines, str(wf / 'solve_data/steps_in_timeline.csv'))
        solve_writers.write_active_timelines(active_time_lists[solve], str(wf / 'solve_data/steps_in_use.csv'))
        solve_writers.write_active_timelines(complete_active_time_lists, str(wf / 'solve_data/steps_complete_solve.csv'), complete=True)
        solve_writers.write_step_jump(jump_lists[solve], work_folder=wf)
        pb_time, pb_succ = make_period_block(active_time_lists[solve])
        solve_writers.write_period_block(pb_time, pb_succ, work_folder=wf)
        state.logger.debug("Creating period data")
        solve_writers.write_period_years(period__branch_lists[solve], solve_period_history[complete_solve[solve]], str(wf / 'solve_data/period_with_history.csv'))
        solve_writers.write_periods(complete_solve[solve], state.solve.realized_invest_periods, str(wf / 'solve_data/realized_invest_periods_of_current_solve.csv'))
        #assume that if realized_invest_periods is not defined,but the invest_periods and realized_periods are defined, use realized_periods also as the realized_invest_periods
        if not state.solve.realized_invest_periods[complete_solve[solve]] and state.solve.invest_periods[complete_solve[solve]] and state.solve.realized_periods[complete_solve[solve]]:
            solve_writers.write_periods(complete_solve[solve], state.solve.realized_periods, str(wf / 'solve_data/realized_invest_periods_of_current_solve.csv'))
        solve_writers.write_periods(complete_solve[solve], state.solve.invest_periods, str(wf / 'solve_data/invest_periods_of_current_solve.csv'))
        # Use years_represented if defined, otherwise default to 1 year per period
        years_rep = state.solve.solve_period_years_represented[complete_solve[solve]]
        if not years_rep:
            years_rep = [
                (pt[0], 1)
                for pt in state.solve.timesets_used_by_solves[complete_solve[solve]]
            ]
        solve_writers.write_years_represented(period__branch_lists[solve], years_rep, str(wf / 'solve_data/p_years_represented.csv'))
        solve_writers.write_period_years(period__branch_lists[solve], years_rep, str(wf / 'solve_data/p_discount_years.csv'))
        solve_writers.write_current_solve(solve, str(wf / 'solve_data/solve_current.csv'))
        solve_writers.write_hole_multiplier(solve, state.solve.hole_multipliers, str(wf / 'solve_data/solve_hole_multiplier.csv'))
        # Agent 8 (LP-scaling): analyse the solve's input CSVs, cache the
        # result per solve name, emit solve_data/scaling_analysis.json for
        # Agent 10's user-facing report, and (optionally) auto-apply the
        # row-scaling recommendation when --auto-scale is active and the
        # user has not explicitly set solve.use_row_scaling.
        # The analyser is intentionally cheap (stdlib only, one CSV pass).
        scale_table = analyze_solve(
            solve_name=solve,
            input_dir=wf / 'input',
            logger=state.logger,
        )
        if getattr(state, "csv_dump", False):
            write_scaling_analysis_json(
                table=scale_table,
                solve_data_dir=wf / 'solve_data',
            )
        auto_scale = getattr(state, 'auto_scale', False)
        applied = maybe_auto_apply_row_scaling(
            solve_name=solve,
            table=scale_table,
            user_setting=state.solve.use_row_scaling.get(solve),
            auto_scale=auto_scale,
            logger=state.logger,
        )
        if applied is not None:
            state.solve.use_row_scaling[solve] = applied

        # Agent 5 (LP-scaling): per-solve opt-in for automatic row scaling.
        # Writes 0/1 to solve_data/p_use_row_scaling.csv; default 0 means the
        # .mod keeps node_capacity_for_scaling / group_capacity_for_scaling
        # at 1 everywhere (pre-Agent-5 behaviour).
        solve_writers.write_p_use_row_scaling(
            solve,
            state.solve.use_row_scaling,
            str(wf / 'solve_data/p_use_row_scaling.csv'),
        )
        # Agent 12 / Agent 21 (LP-scaling): the Agent-8 ScaleTable carries
        # a power-of-10 recommendation for scale_the_objective and a fixed
        # 1.0 for scale_the_state.  Agent 12 originally applied these
        # unconditionally per solve, but that caused rivendell S06 (and
        # similar wide-matrix models) to stall in HiGHS when the recommended
        # 1e-10 objective scale compressed the Cost range by ~4 decades
        # against a Matrix range already at ``[1, 8e+5]``.  Agent 21 gates
        # the application behind the ``--auto-scale`` flag:
        #
        #   * auto_scale=True  → emit the analyser's value (Agent 12 behaviour).
        #   * auto_scale=False → emit a header-only CSV so the .mod's
        #     ``default 1e-6`` / ``default 1`` clauses apply (pre-Agent-12
        #     behaviour restored for the default path).
        #
        # The analyser still runs and emits ``scaling_analysis.json`` above,
        # so the recommendation is visible in diagnostics even when it is
        # not applied.  User explicit overrides in the DB (if added in the
        # future) would go through the same gate.
        if auto_scale:
            solve_writers.write_scale_the_objective(
                wf / 'solve_data',
                scale_table.scale_the_objective,
            )
            solve_writers.write_scale_the_state(
                wf / 'solve_data',
                scale_table.scale_the_state,
            )
        else:
            solve_writers.write_scale_the_objective_header_only(
                wf / 'solve_data',
            )
            solve_writers.write_scale_the_state_header_only(
                wf / 'solve_data',
            )
        solve_writers.write_first_steps(active_time_lists[solve], str(wf / 'solve_data/first_timesteps.csv'))
        solve_writers.write_last_steps(active_time_lists[solve], str(wf / 'solve_data/last_timesteps.csv'))
        solve_writers.write_last_realized_step(active_time_lists[solve], complete_solve[solve], state.solve.realized_periods.get(complete_solve[solve], []), str(wf / 'solve_data/last_realized_timestep.csv'))
        state.logger.debug("Create realized timeline")
        solve_writers.write_realized_dispatch(realized_time_lists[solve], complete_solve[solve], state.solve.realized_periods.get(complete_solve[solve], []), work_folder=wf)
        solve_writers.write_fix_storage_timesteps(fix_storage_time_lists[solve], complete_solve[solve], state.solve.fix_storage_periods.get(complete_solve[solve], []), work_folder=wf)
        solve_writers.write_delayed_durations(active_time_lists[solve], complete_solve[solve], state.solve.delay_durations, work_folder=wf)
        state.logger.debug("Possible stochastics")
        solve_writers.write_branch__period_relationship(period__branch_lists[solve], str(wf / 'solve_data/period__branch.csv'))
        solve_writers.write_all_branches(
            period__branch_lists,
            solve_branch__time_branch_lists[solve],
            state.logger,
            work_folder=wf,
            provider=getattr(state, "cascade_input_provider", None),
        )
        solve_writers.write_branch_weights_and_map(complete_solve[solve], active_time_lists[solve], solve_branch__time_branch_lists[solve], branch_start_time_lists[solve], period__branch_lists[solve], state.solve.stochastic_branches, work_folder=wf)
        solve_writers.write_first_and_last_periods(active_time_lists[solve], state.solve.timesets_used_by_solves[complete_solve[solve]], period__branch_lists[solve], work_folder=wf)

        #check if the upper level fixes storages
        if [complete_solve[solve]] in state.solve.contains_solves.values() and complete_solve[parent_roll[solve]] in state.solve.fix_storage_periods:  # check that the parent_roll exists and has storage fixing
            storage_fix_values_exist = True
        else:
            storage_fix_values_exist = False
        if storage_fix_values_exist:
            state.logger.info("Nested timeline matching")
            stochastic_solver.write_timeline_matching_map(active_time_lists[parent_roll[solve]], active_time_lists[solve], complete_solve[parent_roll[solve]], complete_solve[solve], period__branch_lists[solve], work_folder=wf)
        else:
            with open(wf / "solve_data/timeline_matching_map.csv", 'w') as realfile:
                realfile.write("period,step,upper_step\n")
        #if timeline created from new step_duration, all timeseries have to be averaged or summed for the new timestep
        if previous_complete_solve != complete_solve[solve]:
            state.logger.debug("Aggregating timeline and parameters for the new step size")
            state.timeline.create_averaged_timeseries(
                complete_solve[solve],
                state.solve,
                state.logger,
                work_folder=wf,
                provider=getattr(state, "cascade_input_provider", None),
            )
        previous_complete_solve = complete_solve[solve]

        # Agent 1.1 (flex-temporal + decomposition): derive per-entity
        # temporal blocks and the cross-resolution overlap set, then
        # emit ``solve_data/{entity_block,process_side_block,
        # block_step_duration,overlap_set}.csv``.  In Agent 1.1 these
        # CSVs are inert (no GMPL consumer yet — Agent 1.2 adds
        # declarations); the degenerate "default"-block case emits
        # identity overlap rows that match pre-v51 behaviour.
        try:
            write_block_data_for_solve(
                solve=complete_solve[solve],
                solve_config=state.solve,
                timeline_config=state.timeline,
                work_folder=wf,
                active_time_list=active_time_lists[solve],
                default_jump_list=jump_lists[solve],
            )
        except FlexToolConfigError:
            raise
        except Exception as exc:  # pragma: no cover — defensive only
            state.logger.warning(
                f"blocks: emission failed (non-fatal — not consumed yet): {exc}"
            )

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
            parent_complete = complete_solve[parent_roll[solve]]
            parent_handoff = (
                state.handoffs.get(parent_complete)
                if state.handoffs is not None else None
            )
            if parent_handoff is not None and parent_handoff.fix_storage is not None:
                # Source the parent's fix_storage from the in-memory
                # handoff frame.  The .mod still reads CSV, but the
                # archived ``fix_storage_*_<parent>.csv`` files are no
                # longer the source of truth.
                from flextool.flextoolrunner.solve_handoff import (
                    write_fix_storage_files_from_handoff,
                )
                write_fix_storage_files_from_handoff(
                    parent_handoff.fix_storage, wf / "solve_data",
                )
            else:
                shutil.copy(str(wf / f"solve_data/fix_storage_quantity_{parent_complete}.csv"), str(wf / "solve_data/fix_storage_quantity.csv"))
                shutil.copy(str(wf / f"solve_data/fix_storage_price_{parent_complete}.csv"), str(wf / "solve_data/fix_storage_price.csv"))
                shutil.copy(str(wf / f"solve_data/fix_storage_usage_{parent_complete}.csv"), str(wf / "solve_data/fix_storage_usage.csv"))

        solve_writers.write_solve_status(first_of_nested_level, last_of_nested_level, nested=True, work_folder=wf)
        last = i == len(solves) - 1
        solve_writers.write_solve_status(first, last, work_folder=wf)
        if i == 0:
            first = False
            solve_writers.write_empty_investment_file(work_folder=wf)
            solve_writers.write_empty_storage_fix_file(work_folder=wf)
            solve_writers.write_empty_cumulative_files(work_folder=wf)
            solve_writers.write_headers_for_empty_output_files(str(wf / 'solve_data/costs_discounted.csv'), 'param_costs,costs_discounted')
            solve_writers.write_headers_for_empty_output_files(str(wf / 'solve_data/co2.csv'), 'param_co2,model_wide')
            solve_writers.write_headers_for_empty_output_files(str(wf / 'solve_data/period_capacity.csv'), 'period')

        # Write minimum time lookback data for this solve window
        if process_min_uptime or process_min_downtime:
            write_minimum_time_data(
                active_time_list=active_time_lists[solve],
                jump_list=jump_lists[solve],
                process_min_uptime=process_min_uptime,
                process_min_downtime=process_min_downtime,
                work_folder=wf,
            )
        else:
            # Write empty CSVs so GMPL doesn't fail on missing files
            for fname in ["uptime_lookback.csv", "downtime_lookback.csv"]:
                with open(wf / "solve_data" / fname, "w", newline="") as f:
                    csv.writer(f).writerow(["process", "period", "time", "period_back", "time_back"])

        # Write representative period data if available, or timeset_weights
        # if set on the active timeset (non-RP cost weighting). The two
        # mechanisms are mutually exclusive on a given timeset — error out
        # if both are defined on the same one.
        timesets_used = state.solve.timesets_used_by_solves.get(complete_solve[solve], [])
        active_timeset_names = [ts for _, ts in timesets_used]
        for ts_name in active_timeset_names:
            if ts_name in state.timeline.rp_weights and ts_name in state.timeline.timeset_weights:
                message = (
                    f"Timeset '{ts_name}' has both representative_period_weights and "
                    "timeset_weights set. Pick one: use representative_period_weights "
                    "for RP scenarios and timeset_weights for non-RP per-step weighting."
                )
                state.logger.error(message)
                raise FlexToolConfigError(message)

        rp_written = False
        for ts_name in active_timeset_names:
            if ts_name in state.timeline.rp_weights:
                period_name = None
                for p, ts in timesets_used:
                    if ts == ts_name:
                        period_name = p
                        break
                if period_name:
                    solve_writers.write_rp_data(
                        rp_weights=state.timeline.rp_weights[ts_name],
                        timeset_duration_entries=state.timeline.timeset_durations[ts_name],
                        period_name=period_name,
                        work_folder=wf,
                    )
                    rp_written = True
                    break
        if not rp_written:
            # Still emit the full set of empty RP stubs so the model has every
            # file; then overwrite rp_cost_weight.csv if timeset_weights apply.
            solve_writers.write_empty_rp_data(work_folder=wf)
            solve_writers.write_timeset_cost_weight(
                active_time_list=active_time_lists[solve],
                timesets_used_by_solve=timesets_used,
                timeset_weights=state.timeline.timeset_weights,
                work_folder=wf,
            )

        state.logger.debug("Starting model creation")

        if state.timing_recorder is not None:
            roll_setup_seconds = time.perf_counter() - timer_in_solve
            state.timing_recorder.record(
                'roll_setup',
                solve=solve,
                roll_index=i,
                seconds=roll_setup_seconds,
                t_start=timer_in_solve,
            )

        # Agent 18c (LP-scaling): expose the current solve's cache key to
        # the solver runner so bound-scaling diagnostics land in the
        # same ScaleTable this iteration's scaling_report renders from.
        # ``solve`` here is the per-roll name (the cache key used by
        # ``analyze_solve`` a few hundred lines up); ``complete_solve[solve]``
        # is the parent-solve name that gets passed to ``solver.run``.
        state.current_scale_solve_name = solve
        # Expose the roll-loop index to ``solver_runner`` so its four
        # ``recorder.record(...)`` calls carry ``roll_index=i``,
        # matching the mod-side ``solve / mod_*`` rows ingested below.
        # Without this, rolling scenarios where rolls share a parent-
        # solve name produce indistinguishable Python-side solve rows.
        state.current_roll_index = i
        # Migration hook (Option A): refresh / compute per-solve
        # preprocessing CSVs after solve_writers + blocks have written
        # their per-solve inputs, but before glpsol reads them.
        prior_handoff = (
            state.handoffs.get(last_captured_solve)
            if state.handoffs is not None and last_captured_solve is not None
            else None
        )
        preprocessing_solve_time.run(
            state, complete_solve[solve], prior_handoff=prior_handoff,
        )
        exit_status = solver.run(complete_solve[solve])
        state.current_scale_solve_name = None
        state.current_roll_index = None
        if exit_status == 0:
            state.logger.debug('Success!')
            state.logger.debug("-------------------------------------------------------------------------------------------")
        else:
            message = f'Error: {exit_status}'
            state.logger.error(message)
            raise FlexToolSolveError(message)

        # Pipe the .mod-side per-phase printf rows (setup / total_obj_cost /
        # balance / reserves / rest / r_solution / w_raw / w_capacity) into
        # the unified timings.csv.  The mod writes one row per phase to
        # ``solve_data/mod_phases.csv`` (truncated each glpsol invocation,
        # so the file always reflects the just-finished solve).  We read
        # and re-emit each row as a ``phase='solve', subphase='mod_<name>'``
        # entry so all timing data lives in one place.
        if state.timing_recorder is not None:
            mod_phases_path = wf / "solve_data" / "mod_phases.csv"
            if mod_phases_path.exists():
                try:
                    with open(mod_phases_path) as _mp:
                        _reader = csv.DictReader(_mp)
                        for _row in _reader:
                            try:
                                _seconds = float(_row.get("seconds", "") or 0.0)
                            except ValueError:
                                continue
                            _phase_name = (_row.get("phase") or "").strip()
                            if not _phase_name:
                                continue
                            state.timing_recorder.record(
                                'solve',
                                subphase=f'mod_{_phase_name}',
                                solve=complete_solve[solve],
                                roll_index=i,
                                seconds=_seconds,
                            )
                except Exception as _exc:  # diagnostic only — never fail solve
                    state.logger.debug(
                        f"mod_phases ingest failed for {complete_solve[solve]}: {_exc}"
                    )

        # In-memory solve-to-solve handoff (add-on, opt-in via
        # ``state.handoffs``).  No-op when the slot is None (default,
        # behavior bit-identical to pre-handoff flextool).  See
        # ``solve_handoff.py`` for the carrier dataclass.
        capture_post_solve(state, complete_solve[solve])
        if state.handoffs is not None:
            last_captured_solve = complete_solve[solve]
            # Mirror onto state so post-solve writers in
            # ``solver_runner._run_highs`` (which run BEFORE the next
            # iteration starts) can find the most-recent capture by
            # name without re-implementing iteration tracking.
            state.last_captured_solve = last_captured_solve

        # Agent 10 (LP-scaling): user-facing diagnostic report.  Always
        # generated; cheap; reads the ScaleTable cached by Agent 8, the
        # input CSVs produced earlier this iteration, the HiGHS.log
        # written during ``solver.run`` above, and the slack parquets
        # emitted by the HiGHS → parquet handoff (which runs inside
        # ``solver.run`` for the HiGHS path).  Writes
        # ``solve_data/scaling_report.txt`` and echoes a 3-10 line
        # summary to stdout.  The report's main job is to flag
        # composite-scale mismatches that no linear scaling can fix
        # (e.g. a tiny building unit connected to a continental grid);
        # Agent 11 will validate the full pipeline.
        try:
            write_scaling_report(
                scale_table=scale_table,
                input_dir=wf / 'input',
                solve_data_dir=wf / 'solve_data',
                solve_name=solve,
                highs_log_path=wf / 'HiGHS.log',
                output_raw_dir=wf / 'output_raw',
                applied_row_scaling=state.solve.use_row_scaling.get(solve),
                override_source=(
                    'auto-scale' if applied is not None else 'db'
                ) if state.solve.use_row_scaling.get(solve) is not None else None,
                stdout_summary=True,
                logger=state.logger,
            )
        except Exception as exc:  # diagnostic only — never fail the solve
            state.logger.warning(
                f"scaling_report generation failed (non-fatal): {exc}"
            )

        #if multiple storage solve levels, save the storage fix of this level:
        if complete_solve[solve] in state.solve.fix_storage_periods:
            shutil.copy(str(wf / "solve_data/fix_storage_quantity.csv"), str(wf / f"solve_data/fix_storage_quantity_{complete_solve[solve]}.csv"))
            shutil.copy(str(wf / "solve_data/fix_storage_price.csv"), str(wf / f"solve_data/fix_storage_price_{complete_solve[solve]}.csv"))
            shutil.copy(str(wf / "solve_data/fix_storage_usage.csv"), str(wf / f"solve_data/fix_storage_usage_{complete_solve[solve]}.csv"))

    if len(state.solve.model_solve) > 1:
        message = 'Trying to run more than one model - not supported. The results of the first model are retained.'
        state.logger.error(message)
        raise FlexToolConfigError(message)
    return 0
