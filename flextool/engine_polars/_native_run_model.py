"""Native cascade driver — Phase 3 of the writer port.

This module owns the per-solve cascade loop that used to live in
``flextool.flextoolrunner.orchestration.run_model``.  The native function
:func:`native_run_model` is invoked by :func:`._orchestration._drive_cascade`
in place of the legacy import.

Design decisions
----------------

* **Cascade loop is native; inner preprocessing is not (yet).**
  Phase 3's scope is to own the cascade walk and inline the solve-tree
  expansion / stochastic branching / per-solve setup, *not* to re-port
  ``preprocessing.solve_time`` or the ``solve_writers`` module.  Those
  remain authoritative and are still called as functions — the
  ``_native_leaf_set_override()`` context (wired by
  :func:`._orchestration._drive_cascade`) intercepts the already-native
  preprocessing helpers.
* **Native solve-tree expansion.**  Uses
  :class:`flextool.engine_polars._recursive_solve.RecursiveSolveBuilder`
  + :class:`._stochastic.StochasticSolver` — direct ports of the
  flextoolrunner equivalents.
* **No ``capture_post_solve`` call.**  Legacy ``run_model`` called
  ``solve_handoff.capture_post_solve`` after every per-solve invocation,
  and the cascade monkey-patched it to a no-op (see
  ``_orchestration.py:704``) to keep flexpy-derived handoffs from being
  overwritten by the legacy CSV-based capture.  In the native cascade we
  simply omit the call — semantically identical to the patched no-op,
  but cleaner.  The monkey-patch is still applied in ``_drive_cascade``
  as a belt-and-suspenders guard for any other consumer that might
  still reference ``capture_post_solve`` via this module's globals.
* **Optional state fields tolerated.**  Native :class:`RunnerState`
  lacks ``timing_recorder`` / ``auto_scale``.  The few legacy paths that
  consume those guard with ``getattr(state, name, default)`` so the
  function works equally against either runner state shape.

Reference: ``flextool/flextoolrunner/orchestration.py:31-638``.
"""
from __future__ import annotations

import copy
import csv
import os
import shutil
import time
from collections import defaultdict

# ---------------------------------------------------------------------------
# Imports.  Per Phase 3 scope, we depend on legacy preprocessing /
# solve_writer modules directly — Phase 2 covers the override hook that
# intercepts the already-ported helpers.  Phases 4+ will retire the
# remaining writers.
# ---------------------------------------------------------------------------

from flextool.flextoolrunner.blocks import write_block_data_for_solve
from flextool.flextoolrunner.minimum_time import write_minimum_time_data
from flextool.flextoolrunner.preprocessing import (
    solve_time as preprocessing_solve_time,
)
from flextool.flextoolrunner.runner_state import (
    FlexToolConfigError,
    FlexToolSolveError,
)
from flextool.flextoolrunner.scaling import (
    analyze_solve,
    maybe_auto_apply_row_scaling,
    write_scaling_analysis_json,
)
from flextool.flextoolrunner.scaling_report import write_scaling_report
from flextool.flextoolrunner import solve_writers

# Phase C — per-sub-solve FlexData accumulator.  Built fresh inside the
# per-solve loop and released when the next sub-solve runs; CSV emission
# is unchanged (parallel-write mode).  Phase D will consume the
# accumulator as a ``seed`` to ``load_flextool``; today it is inert.
from flextool.engine_polars._flex_data_accumulator import (
    FlexDataAccumulator,
    capture_frames,
)

# Native solve-tree expansion + stochastic branching + timeline helpers.
from flextool.engine_polars._recursive_solve import (
    ParentSolveInfo,
    RecursiveSolveBuilder,
)
from flextool.engine_polars._stochastic import StochasticSolver
from flextool.engine_polars._timeline import (
    get_active_time,
    make_period_block,
    separate_period_and_timeseries_data,
)


def _fan_out_fix_storage(fix_storage):
    """Phase E-d — fan a wide ``SolveHandoff.fix_storage`` frame
    (``[node, period, time, quantity, price, usage]``) back into the
    three per-metric long frames the in-memory cross-solve carrier
    holds (and the on-disk ``solve_data/fix_storage_*.csv`` files
    expose).  Returns ``{basename: pl.DataFrame}`` for the three
    metrics; metrics with no rows are omitted.

    Mirrors the disk fan-out in
    ``_solve_handoff.write_fix_storage_files_from_handoff`` but keeps
    the result in-memory so the next sub-solve's accumulator can seed
    them via ``_seed_lookup``.
    """
    import polars as pl
    out: dict[str, "pl.DataFrame"] = {}
    for metric, on_disk_col, fname in (
        ("quantity", "p_fix_storage_quantity", "fix_storage_quantity.csv"),
        ("price",    "p_fix_storage_price",    "fix_storage_price.csv"),
        ("usage",    "p_fix_storage_usage",    "fix_storage_usage.csv"),
    ):
        if metric not in fix_storage.columns:
            continue
        sub = (
            fix_storage
            .filter(pl.col(metric).is_not_null())
            .rename({"time": "step", metric: on_disk_col})
            .select("node", "period", "step", on_disk_col)
        )
        out[fname] = sub
    return out


def native_run_model(state, solver) -> int:
    """Drive the per-solve cascade natively.

    Phase 3 replacement for
    ``flextool.flextoolrunner.orchestration.run_model``.  Walks the
    solve tree, applies stochastic branching, writes the per-solve
    inputs the solver needs (via legacy ``solve_writers`` /
    ``preprocessing.solve_time`` — both intercepted where native ports
    exist by the override hook in ``_orchestration._drive_cascade``),
    and invokes the per-solve callback ``solver.run(complete_solve_name)``.

    Args:
        state: ``RunnerState`` carrier — paths, solve config, timeline,
            handoffs, logger.  Either the flextool ``RunnerState`` or
            the native :class:`flextool.engine_polars._solve_state.RunnerState`
            shape; optional fields (``timing_recorder``, ``auto_scale``)
            are tolerated via ``getattr``.
        solver: A :class:`SolverRunner` subclass.  In the native cascade
            this is ``_FlexpyCascadeSolver`` defined inside
            :func:`._orchestration._drive_cascade`.

    Returns:
        0 on success.

    Raises:
        FlexToolConfigError: configuration errors.
        FlexToolSolveError:  any per-solve non-zero exit status.
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
    for sub in ("solve_data", "output_raw", "output_plots"):
        try:
            os.mkdir(wf / sub)
        except FileExistsError:
            state.logger.debug(f"{sub} folder existed")

    if not state.solve.model_solve:
        message = (
            "No model. Make sure the 'model' class defines solves [Array]."
        )
        state.logger.error(message)
        raise FlexToolConfigError(message)
    solves = next(iter(state.solve.model_solve.values()))
    if not solves:
        message = "No solves in model."
        state.logger.error(message)
        raise FlexToolConfigError(message)

    # ------------------------------------------------------------------
    # 1. Expand the solve tree (rolling / nested / stochastic).
    # ------------------------------------------------------------------
    solve_builder = RecursiveSolveBuilder(state)
    for solve in solves:
        parent_info = ParentSolveInfo(solve=None, roll=None)
        result = solve_builder.define_solve_recursive(
            solve, parent_info, None, None, -1,
        )
        all_solves += result.solves
        complete_solve.update(result.complete_solves)
        parent_roll.update(result.parent_roll_lists)
        active_time_lists.update(result.active_time_lists)
        fix_storage_time_lists.update(result.fix_storage_time_lists)
        realized_time_lists.update(copy.deepcopy(result.realized_time_lists))

    # Dedupe realized timesteps across solves: keep only the first
    # occurrence of any (period, timestep) pair.  Iterates in reverse so
    # the earliest solve owns the realization.
    already_realized_timesteps: dict[str, set[str]] = {}
    for solve, realized_time_list in reversed(realized_time_lists.items()):
        for period, timesteps in list(realized_time_list.items()):
            if period not in already_realized_timesteps:
                already_realized_timesteps[period] = set()
            for i, timestep in enumerate(timesteps):
                # If we hit one already taken, the rest are assumed
                # overlapping too (preserved verbatim from legacy).
                if timestep.timestep in already_realized_timesteps[period]:
                    del realized_time_lists[solve][period][i:]
                    break
                else:
                    already_realized_timesteps[period].add(timestep.timestep)
            if not timesteps:
                del realized_time_lists[solve][period]

    # ------------------------------------------------------------------
    # 2. Per-real-solve period history accumulation (O(N) instead of O(N²)).
    # ------------------------------------------------------------------
    cumulative_contributions: list[tuple[str, float]] = []
    cumulative_period_names: set[str] = set()

    for solve in state.solve.real_solves:
        timeset_periods = {
            pt[0] for pt in state.solve.timesets_used_by_solves[solve]
        }
        state.solve.solve_period_years_represented[solve] = [
            py for py in state.solve.solve_period_years_represented[solve]
            if py[0] in timeset_periods
        ]
        history_period_names: set[str] = set()
        for period_name, years in cumulative_contributions:
            if period_name not in history_period_names:
                solve_period_history[solve].append((period_name, years))
                history_period_names.add(period_name)
        for period__year in state.solve.solve_period_years_represented[solve]:
            if period__year[0] not in history_period_names:
                solve_period_history[solve].append(
                    (period__year[0], period__year[1])
                )
                history_period_names.add(period__year[0])
        # Default to one year per period when years_represented is empty.
        if not state.solve.solve_period_years_represented[solve]:
            for period__timeset in state.solve.timesets_used_by_solves[solve]:
                if period__timeset[0] not in history_period_names:
                    solve_period_history[solve].append(
                        (period__timeset[0], 1)
                    )
                    history_period_names.add(period__timeset[0])
        period_dict_names = {
            t[0] for t in (
                state.solve.realized_periods.get(solve, [])
                + state.solve.invest_periods.get(solve, [])
                + state.solve.fix_storage_periods.get(solve, [])
                + state.solve.realized_invest_periods.get(solve, [])
            )
        }
        for period in state.solve.solve_period_years_represented[solve]:
            if (
                period[0] in period_dict_names
                and period[0] not in cumulative_period_names
            ):
                cumulative_contributions.append((period[0], period[1]))
                cumulative_period_names.add(period[0])

    # ------------------------------------------------------------------
    # 3. Stochastic branching expansion.
    # ------------------------------------------------------------------
    stochastic_solver = StochasticSolver(state)
    (
        period__branch_lists,
        solve_branch__time_branch_lists,
        active_time_lists,
        jump_lists,
        fix_storage_time_lists,
        realized_time_lists,
        branch_start_time_lists,
    ) = stochastic_solver.create_stochastic_periods(
        state.solve.stochastic_branches,
        all_solves,
        complete_solve,
        active_time_lists,
        fix_storage_time_lists,
        realized_time_lists,
    )

    for solve in active_time_lists.keys():
        for period in active_time_lists[solve]:
            if (period, period) in period__branch_lists[solve] and not any(
                period == sublist[0]
                for sublist in solve_period_history[complete_solve[solve]]
            ):
                message = (
                    f"The years_represented is defined, but not to all "
                    f"of the periods ({period}) in the solve"
                )
                state.logger.error(message)
                raise FlexToolConfigError(message)

    timing = time.perf_counter() - timer
    state.logger.debug(f"--- Pre-processing of data: {timing:.4f} seconds ---")
    timing_recorder = getattr(state, "timing_recorder", None)
    if timing_recorder is not None:
        timing_recorder.record(
            "preprocessing_global",
            seconds=timing,
            t_start=time.perf_counter() - timing,
        )
    timer = timer + timing

    separate_period_and_timeseries_data(
        state.timeline.timelines,
        state.solve.timesets_used_by_solves,
        work_folder=wf,
    )

    # ------------------------------------------------------------------
    # 4. Minimum up/down-time precomputation reads.
    # ------------------------------------------------------------------
    process_min_uptime: dict[str, float] = {}
    process_min_downtime: dict[str, float] = {}
    process_min_uptime_set: set[str] = set()
    process_min_downtime_set: set[str] = set()

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

    p_process_csv = wf / "input" / "p_process.csv"
    if p_process_csv.exists() and (
        process_min_uptime_set or process_min_downtime_set
    ):
        with open(p_process_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                proc = row["process"]
                param = row["processParam"]
                val = float(row["p_process"]) if row["p_process"] else 0.0
                if (
                    param == "min_uptime"
                    and proc in process_min_uptime_set
                    and val > 0
                ):
                    process_min_uptime[proc] = val
                elif (
                    param == "min_downtime"
                    and proc in process_min_downtime_set
                    and val > 0
                ):
                    process_min_downtime[proc] = val

    # Phase E-d — cascade-wide accumulator seeded once, then forwarded
    # into each per-sub-solve accumulator.  Captures the
    # ``solve_writers.write_timesets`` output (``input/timesets_in_use.csv``
    # + ``input/timesets__timeline.csv``) so the per-sub-solve preprocessing
    # readers can find them in seed-mode without touching disk.
    cascade_input_accumulator: "FlexDataAccumulator | None" = getattr(
        state, "cascade_input_accumulator", None,
    )
    if cascade_input_accumulator is None:
        cascade_input_accumulator = FlexDataAccumulator(solve_name="__cascade_input__")
        state.cascade_input_accumulator = cascade_input_accumulator

    # Solve-loop-invariant timesets — hoisted out of the per-solve loop.
    with capture_frames(cascade_input_accumulator):
        solve_writers.write_timesets(
            state.solve.timesets_used_by_solves,
            state.timeline.timesets__timeline,
            work_folder=wf,
        )

    # ------------------------------------------------------------------
    # 5. Per-solve loop.
    # ------------------------------------------------------------------
    first = True
    previous_complete_solve = None
    last_captured_solve: str | None = None
    cached_complete_active_time_lists: dict = {}

    for i, solve in enumerate(all_solves):
        timer_in_solve = time.perf_counter()

        state.logger.debug(
            f"Creating timelines for solve {solve} ({i})"
        )
        cs = complete_solve[solve]
        if cs not in cached_complete_active_time_lists:
            cached_complete_active_time_lists[cs] = get_active_time(
                cs,
                state.solve.timesets_used_by_solves,
                state.timeline.timeset_durations,
                state.timeline.timelines,
                state.timeline.timesets__timeline,
            )
        complete_active_time_lists = cached_complete_active_time_lists[cs]

        # Combined period/timeset list including history periods.
        period__timesets_with_history = list(
            state.solve.timesets_used_by_solves[complete_solve[solve]]
        )
        current_periods = {pt[0] for pt in period__timesets_with_history}
        current_timeset = (
            period__timesets_with_history[0][1]
            if period__timesets_with_history else None
        )
        for history_period, _ in solve_period_history[complete_solve[solve]]:
            if history_period not in current_periods:
                if current_timeset:
                    period__timesets_with_history.append(
                        (history_period, current_timeset)
                    )
                    current_periods.add(history_period)

        # Phase E-d — Build the per-sub-solve accumulator early and
        # install it as the active capture target so every
        # ``_writer_*._write`` call inside this iteration's body funnels
        # through the accumulator.  Manual enter/exit (via the
        # underlying contextmanager) keeps the iteration body's
        # existing indentation untouched while still patching the writer
        # ``_write`` helpers for the duration of the per-iter scope.
        #
        # Cross-sub-solve carriers (fix_storage_* and friends that the
        # cascade historically propagated via ``shutil.copy`` between
        # sub-solves) are pre-seeded into the new accumulator from
        # ``state.cross_solve_carriers`` at the start of the iteration.
        # The next iteration's accumulator only retains the latest
        # frames, so memory does not grow with cascade depth (per
        # handoff doc decision #11).
        sub_solve_accumulator = FlexDataAccumulator(
            solve_name=complete_solve[solve],
        )
        # Seed cascade-wide ``input/*.csv`` frames so per-iter readers
        # find them in seed-mode (e.g. ``input/timesets_in_use.csv``).
        for _key, _frame in cascade_input_accumulator.frames.items():
            sub_solve_accumulator.frames[_key] = _frame
        # Seed cross-sub-solve carriers from the prior tail.  Used for
        # the rolling-cascade fix_storage_* propagation which previously
        # depended on ``dump_csvs`` having written the prior solve's
        # ``solve_data/fix_storage_*.csv`` to disk.
        carriers = getattr(state, "cross_solve_carriers", None)
        if carriers is None:
            carriers = {}
            state.cross_solve_carriers = carriers
        for _key, _frame in carriers.get("__last__", {}).items():
            sub_solve_accumulator.frames[_key] = _frame
        # Per-parent archive — pull the parent's tail frames keyed under
        # ``parent_complete``.  Used by nested cascades that consume an
        # upper-level parent's fix_storage_*.
        _parent_solve_for_carriers = parent_roll.get(solve)
        _parent_complete_for_carriers = (
            complete_solve.get(_parent_solve_for_carriers)
            if _parent_solve_for_carriers else None
        )
        if (
            _parent_complete_for_carriers
            and _parent_complete_for_carriers in carriers
        ):
            for _key, _frame in carriers[_parent_complete_for_carriers].items():
                sub_solve_accumulator.frames[_key] = _frame

        # Manual enter on ``capture_frames``.  We exit AFTER
        # preprocessing_solve_time.run completes (search ``# Phase E-d
        # capture exit`` below); the iteration body in between behaves
        # exactly as before.
        _capture_ctx = capture_frames(sub_solve_accumulator)
        _capture_ctx.__enter__()
        # Phase E-d — also install the accumulator as the active seed
        # so per-iter readers (``_writer_per_solve._read_csv``, the
        # ``csv.reader`` sites that route through ``_seed_open``) hit
        # the in-memory frames produced earlier in this iteration's
        # writer chain.  Restored to the prior seed value on capture
        # exit.
        from flextool.engine_polars._input_source import (
            _install_seed as _install_seed_E_d,
            _active_seed as _initial_seed,  # captured at import time, see below
        )
        # Save the live seed (read fresh — ``_active_seed`` module-level
        # is the canonical truth) and install the per-sub-solve accumulator.
        import flextool.engine_polars._input_source as _is_mod_E_d
        _prior_seed_E_d = _is_mod_E_d._active_seed
        _install_seed_E_d(sub_solve_accumulator)

        solve_writers.write_full_timelines(
            state.timeline.stochastic_timesteps[solve],
            period__timesets_with_history,
            state.timeline.timesets__timeline,
            state.timeline.timelines,
            str(wf / "solve_data/steps_in_timeline.csv"),
        )
        solve_writers.write_active_timelines(
            active_time_lists[solve],
            str(wf / "solve_data/steps_in_use.csv"),
        )
        solve_writers.write_active_timelines(
            complete_active_time_lists,
            str(wf / "solve_data/steps_complete_solve.csv"),
            complete=True,
        )
        solve_writers.write_step_jump(jump_lists[solve], work_folder=wf)
        pb_time, pb_succ = make_period_block(active_time_lists[solve])
        solve_writers.write_period_block(pb_time, pb_succ, work_folder=wf)

        state.logger.debug("Creating period data")
        solve_writers.write_period_years(
            period__branch_lists[solve],
            solve_period_history[complete_solve[solve]],
            str(wf / "solve_data/period_with_history.csv"),
        )
        solve_writers.write_periods(
            complete_solve[solve],
            state.solve.realized_invest_periods,
            str(wf / "solve_data/realized_invest_periods_of_current_solve.csv"),
        )
        # If realized_invest_periods is empty but both invest_periods and
        # realized_periods are defined, fall back to realized_periods.
        if (
            not state.solve.realized_invest_periods[complete_solve[solve]]
            and state.solve.invest_periods[complete_solve[solve]]
            and state.solve.realized_periods[complete_solve[solve]]
        ):
            solve_writers.write_periods(
                complete_solve[solve],
                state.solve.realized_periods,
                str(
                    wf
                    / "solve_data/realized_invest_periods_of_current_solve.csv"
                ),
            )
        solve_writers.write_periods(
            complete_solve[solve],
            state.solve.invest_periods,
            str(wf / "solve_data/invest_periods_of_current_solve.csv"),
        )

        years_rep = state.solve.solve_period_years_represented[
            complete_solve[solve]
        ]
        if not years_rep:
            years_rep = [
                (pt[0], 1)
                for pt in state.solve.timesets_used_by_solves[
                    complete_solve[solve]
                ]
            ]
        solve_writers.write_years_represented(
            period__branch_lists[solve],
            years_rep,
            str(wf / "solve_data/p_years_represented.csv"),
        )
        solve_writers.write_period_years(
            period__branch_lists[solve],
            years_rep,
            str(wf / "solve_data/p_discount_years.csv"),
        )
        solve_writers.write_current_solve(
            solve, str(wf / "solve_data/solve_current.csv")
        )
        solve_writers.write_hole_multiplier(
            solve,
            state.solve.hole_multipliers,
            str(wf / "solve_data/solve_hole_multiplier.csv"),
        )

        # ---- LP scaling analyser (cached per solve name) ----
        scale_table = analyze_solve(
            solve_name=solve,
            input_dir=wf / "input",
            logger=state.logger,
        )
        write_scaling_analysis_json(
            table=scale_table,
            solve_data_dir=wf / "solve_data",
        )
        auto_scale = getattr(state, "auto_scale", False)
        applied = maybe_auto_apply_row_scaling(
            solve_name=solve,
            table=scale_table,
            user_setting=state.solve.use_row_scaling.get(solve),
            auto_scale=auto_scale,
            logger=state.logger,
        )
        if applied is not None:
            state.solve.use_row_scaling[solve] = applied

        solve_writers.write_p_use_row_scaling(
            solve,
            state.solve.use_row_scaling,
            str(wf / "solve_data/p_use_row_scaling.csv"),
        )
        if auto_scale:
            solve_writers.write_scale_the_objective(
                wf / "solve_data",
                scale_table.scale_the_objective,
            )
            solve_writers.write_scale_the_state(
                wf / "solve_data",
                scale_table.scale_the_state,
            )
        else:
            solve_writers.write_scale_the_objective_header_only(
                wf / "solve_data",
            )
            solve_writers.write_scale_the_state_header_only(
                wf / "solve_data",
            )

        solve_writers.write_first_steps(
            active_time_lists[solve],
            str(wf / "solve_data/first_timesteps.csv"),
        )
        solve_writers.write_last_steps(
            active_time_lists[solve],
            str(wf / "solve_data/last_timesteps.csv"),
        )
        solve_writers.write_last_realized_step(
            active_time_lists[solve],
            complete_solve[solve],
            state.solve.realized_periods.get(complete_solve[solve], []),
            str(wf / "solve_data/last_realized_timestep.csv"),
        )

        state.logger.debug("Create realized timeline")
        solve_writers.write_realized_dispatch(
            realized_time_lists[solve],
            complete_solve[solve],
            state.solve.realized_periods.get(complete_solve[solve], []),
            work_folder=wf,
        )
        solve_writers.write_fix_storage_timesteps(
            fix_storage_time_lists[solve],
            complete_solve[solve],
            state.solve.fix_storage_periods.get(complete_solve[solve], []),
            work_folder=wf,
        )
        solve_writers.write_delayed_durations(
            active_time_lists[solve],
            complete_solve[solve],
            state.solve.delay_durations,
            work_folder=wf,
        )

        state.logger.debug("Possible stochastics")
        solve_writers.write_branch__period_relationship(
            period__branch_lists[solve],
            str(wf / "solve_data/period__branch.csv"),
        )
        solve_writers.write_all_branches(
            period__branch_lists,
            solve_branch__time_branch_lists[solve],
            state.logger,
            work_folder=wf,
        )
        solve_writers.write_branch_weights_and_map(
            complete_solve[solve],
            active_time_lists[solve],
            solve_branch__time_branch_lists[solve],
            branch_start_time_lists[solve],
            period__branch_lists[solve],
            state.solve.stochastic_branches,
            work_folder=wf,
        )
        solve_writers.write_first_and_last_periods(
            active_time_lists[solve],
            state.solve.timesets_used_by_solves[complete_solve[solve]],
            period__branch_lists[solve],
            work_folder=wf,
        )

        # ---- Storage fixing from upper level ----
        if (
            [complete_solve[solve]] in state.solve.contains_solves.values()
            and complete_solve[parent_roll[solve]] in state.solve.fix_storage_periods
        ):
            storage_fix_values_exist = True
        else:
            storage_fix_values_exist = False

        if storage_fix_values_exist:
            state.logger.info("Nested timeline matching")
            matching_map = stochastic_solver.write_timeline_matching_map(
                active_time_lists[parent_roll[solve]],
                active_time_lists[solve],
                complete_solve[parent_roll[solve]],
                complete_solve[solve],
                period__branch_lists[solve],
            )
            with open(
                wf / "solve_data/timeline_matching_map.csv", "w", newline=""
            ) as realfile:
                writer = csv.writer(realfile)
                writer.writerow(["period", "step", "upper_step"])
                for (period, step), upper_step in matching_map.items():
                    writer.writerow([period, step, upper_step])
        else:
            with open(wf / "solve_data/timeline_matching_map.csv", "w") as realfile:
                realfile.write("period,step,upper_step\n")

        # Aggregate timeseries when step size changes between solves.
        if previous_complete_solve != complete_solve[solve]:
            state.logger.debug(
                "Aggregating timeline and parameters for the new step size"
            )
            state.timeline.create_averaged_timeseries(
                complete_solve[solve], state.solve, state.logger, work_folder=wf,
            )
        previous_complete_solve = complete_solve[solve]

        # ---- Block data (Agent 1.1) ----
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

        first_of_nested_level = solve in state.solve.first_of_complete_solve
        last_of_nested_level = solve in state.solve.last_of_solve

        # Storage fix copy-from-parent (upper-level handoff).
        if storage_fix_values_exist:
            state.logger.info("Fetching storage parameters from the upper solve")
            parent_complete = complete_solve[parent_roll[solve]]
            parent_handoff = (
                state.handoffs.get(parent_complete)
                if state.handoffs is not None else None
            )
            if parent_handoff is not None and parent_handoff.fix_storage is not None:
                from flextool.flextoolrunner.solve_handoff import (
                    write_fix_storage_files_from_handoff,
                )
                write_fix_storage_files_from_handoff(
                    parent_handoff.fix_storage, wf / "solve_data",
                )
            else:
                # Phase E-d — when the parent's archived per-parent
                # CSVs don't exist on disk (csv-emission disabled),
                # fall back to the in-memory cross-solve carrier slot
                # populated at the end of the parent's iteration.
                _parent_carriers = carriers.get(parent_complete, {})
                _parent_archive = (
                    wf / f"solve_data/fix_storage_quantity_{parent_complete}.csv"
                )
                if _parent_archive.exists():
                    shutil.copy(
                        str(wf / f"solve_data/fix_storage_quantity_{parent_complete}.csv"),
                        str(wf / "solve_data/fix_storage_quantity.csv"),
                    )
                    shutil.copy(
                        str(wf / f"solve_data/fix_storage_price_{parent_complete}.csv"),
                        str(wf / "solve_data/fix_storage_price.csv"),
                    )
                    shutil.copy(
                        str(wf / f"solve_data/fix_storage_usage_{parent_complete}.csv"),
                        str(wf / "solve_data/fix_storage_usage.csv"),
                    )
                else:
                    # Seed the in-memory accumulator from the parent's
                    # carrier slot — readers consult ``_seed_lookup``
                    # for these basenames so the disk copy is unneeded.
                    for _bn in (
                        "fix_storage_quantity.csv",
                        "fix_storage_price.csv",
                        "fix_storage_usage.csv",
                    ):
                        _src = _parent_carriers.get(_bn)
                        if _src is not None:
                            sub_solve_accumulator.frames[_bn] = _src

        solve_writers.write_solve_status(
            first_of_nested_level, last_of_nested_level,
            nested=True, work_folder=wf,
        )
        last = i == len(solves) - 1
        solve_writers.write_solve_status(first, last, work_folder=wf)
        if i == 0:
            first = False
            solve_writers.write_empty_investment_file(work_folder=wf)
            solve_writers.write_empty_storage_fix_file(work_folder=wf)
            solve_writers.write_empty_cumulative_files(work_folder=wf)
            solve_writers.write_headers_for_empty_output_files(
                str(wf / "solve_data/costs_discounted.csv"),
                "param_costs,costs_discounted",
            )
            solve_writers.write_headers_for_empty_output_files(
                str(wf / "solve_data/co2.csv"),
                "param_co2,model_wide",
            )
            solve_writers.write_headers_for_empty_output_files(
                str(wf / "solve_data/period_capacity.csv"),
                "period",
            )

        # Minimum up/down-time lookback windows.
        if process_min_uptime or process_min_downtime:
            write_minimum_time_data(
                active_time_list=active_time_lists[solve],
                jump_list=jump_lists[solve],
                process_min_uptime=process_min_uptime,
                process_min_downtime=process_min_downtime,
                work_folder=wf,
            )
        else:
            # Empty header-only files so GMPL doesn't fail on missing files.
            for fname in ("uptime_lookback.csv", "downtime_lookback.csv"):
                with open(wf / "solve_data" / fname, "w", newline="") as f:
                    csv.writer(f).writerow(
                        ["process", "period", "time", "period_back", "time_back"]
                    )

        # ---- Representative-period / timeset weights ----
        timesets_used = state.solve.timesets_used_by_solves.get(
            complete_solve[solve], []
        )
        active_timeset_names = [ts for _, ts in timesets_used]
        for ts_name in active_timeset_names:
            if (
                ts_name in state.timeline.rp_weights
                and ts_name in state.timeline.timeset_weights
            ):
                message = (
                    f"Timeset '{ts_name}' has both "
                    "representative_period_weights and timeset_weights set. "
                    "Pick one: use representative_period_weights for RP "
                    "scenarios and timeset_weights for non-RP per-step "
                    "weighting."
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
                        timeset_duration_entries=state.timeline.timeset_durations[
                            ts_name
                        ],
                        period_name=period_name,
                        work_folder=wf,
                    )
                    rp_written = True
                    break
        if not rp_written:
            solve_writers.write_empty_rp_data(work_folder=wf)
            solve_writers.write_timeset_cost_weight(
                active_time_list=active_time_lists[solve],
                timesets_used_by_solve=timesets_used,
                timeset_weights=state.timeline.timeset_weights,
                work_folder=wf,
            )

        state.logger.debug("Starting model creation")

        if timing_recorder is not None:
            roll_setup_seconds = time.perf_counter() - timer_in_solve
            timing_recorder.record(
                "roll_setup",
                solve=solve,
                roll_index=i,
                seconds=roll_setup_seconds,
                t_start=timer_in_solve,
            )

        # Per-roll diagnostics handles.
        state.current_scale_solve_name = solve
        state.current_roll_index = i

        # Per-solve preprocessing chain (still authoritative — the
        # override hook intercepts already-ported helpers).  Feeds
        # ``prior_handoff`` from the most-recent capture so consume-side
        # reads come from the in-memory handoff dict when present.
        prior_handoff = (
            state.handoffs.get(last_captured_solve)
            if state.handoffs is not None and last_captured_solve is not None
            else None
        )
        _phase_timing = (
            os.environ.get("FLEXTOOL_PHASE_TIMING") == "1"
            and timing_recorder is not None
        )
        _t_preproc_start = time.perf_counter() if _phase_timing else 0.0
        # Phase C — build a fresh FlexData accumulator for THIS sub-solve.
        # Released when the next iteration replaces ``state.current_accumulator``;
        # memory footprint is O(1) per sub-solve.  Phase E-d — the
        # accumulator is now built UPSTREAM of the per-iter
        # solve_writers calls (see top of the loop body) and the
        # ``capture_frames`` patch is already active for this iteration.
        # ``preprocessing_solve_time.run`` runs INSIDE that scope so its
        # writer-port emissions also flow into the accumulator.
        try:
            preprocessing_solve_time.run(
                state, complete_solve[solve], prior_handoff=prior_handoff,
            )
        finally:
            # ---- Phase E-d capture exit ----
            _capture_ctx.__exit__(None, None, None)
            # Restore the prior seed (typically None outside the cascade).
            _install_seed_E_d(_prior_seed_E_d)
        # Stash on state — read by ``_FlexpyCascadeSolver.run`` to attach
        # to the per-sub-solve ``OrchestrationStep``.  Replaces any prior
        # sub-solve's accumulator (per-sub-solve memory discipline).
        state.current_accumulator = sub_solve_accumulator
        # Phase E-d — refresh cross-sub-solve carriers from this
        # iteration's tail.  ONLY the slim cross-solve carrier basenames
        # are retained; the ``__last__`` slot is overwritten so memory
        # never grows beyond the latest sub-solve's carriers.  The
        # per-parent slot (keyed by complete-solve name) is also
        # refreshed for nested cascades whose child solves consume an
        # upper-level parent's fix_storage_*.
        _CROSS_SOLVE_BASENAMES = (
            "fix_storage_quantity.csv",
            "fix_storage_price.csv",
            "fix_storage_usage.csv",
            "p_entity_pre_existing.csv",
            "p_entity_divest_cumulative_max.csv",
            "p_entity_invested.csv",
            "p_entity_divested.csv",
            "p_entity_period_existing_capacity.csv",
            "p_roll_continue_state.csv",
            "co2_cum_realized_tonnes.csv",
            "ladder_cum_sim_hours.csv",
            "ladder_cum_realized_mwh.csv",
            "ed_history_realized.csv",
            "ed_history_realized_first.csv",
            "edd_history.csv",
        )
        _last_carriers: dict[str, "pl.DataFrame"] = {}
        for _basename in _CROSS_SOLVE_BASENAMES:
            _frame = sub_solve_accumulator.get(_basename)
            if _frame is not None:
                _last_carriers[_basename] = _frame
        if _last_carriers:
            carriers["__last__"] = _last_carriers
            carriers[complete_solve[solve]] = _last_carriers
        if _phase_timing:
            timing_recorder.record(
                "per_iter",
                subphase="preprocessing",
                solve=complete_solve[solve],
                roll_index=i,
                seconds=time.perf_counter() - _t_preproc_start,
                t_start=_t_preproc_start,
            )

        # Phase 4 (Gap F) — expose the upper-level (nesting) parent's
        # complete solve name so ``_FlexpyCascadeSolver.run`` can look the
        # parent's :class:`SolveHandoff` up out of ``state.handoffs`` and
        # pass it to ``build_handoff_from_flexpy`` (which uses it to skip
        # the workdir's ``fix_storage_{price,usage}.csv`` reads).  Resets
        # to None for top-level solves.
        _parent_solve = parent_roll.get(solve)
        state.current_parent_complete = (
            complete_solve.get(_parent_solve) if _parent_solve else None
        )

        exit_status = solver.run(complete_solve[solve])
        state.current_parent_complete = None
        state.current_scale_solve_name = None
        state.current_roll_index = None

        if exit_status == 0:
            state.logger.debug("Success!")
            state.logger.debug(
                "------------------------------------------------------------"
                "-------------------------------"
            )
        else:
            message = f"Error: {exit_status}"
            state.logger.error(message)
            raise FlexToolSolveError(message)

        # ---- mod-phase timing ingest (legacy diagnostic) ----
        if timing_recorder is not None:
            mod_phases_path = wf / "solve_data" / "mod_phases.csv"
            if mod_phases_path.exists():
                try:
                    with open(mod_phases_path) as _mp:
                        _reader = csv.DictReader(_mp)
                        for _row in _reader:
                            try:
                                _seconds = float(
                                    _row.get("seconds", "") or 0.0
                                )
                            except ValueError:
                                continue
                            _phase_name = (_row.get("phase") or "").strip()
                            if not _phase_name:
                                continue
                            timing_recorder.record(
                                "solve",
                                subphase=f"mod_{_phase_name}",
                                solve=complete_solve[solve],
                                roll_index=i,
                                seconds=_seconds,
                            )
                except Exception as _exc:  # diagnostic only
                    state.logger.debug(
                        f"mod_phases ingest failed for "
                        f"{complete_solve[solve]}: {_exc}"
                    )

        # In-memory handoff bookkeeping.  We deliberately skip the
        # legacy ``capture_post_solve(state, complete_solve[solve])``
        # call here — the cascade has already deposited a
        # flexpy-derived ``SolveHandoff`` into ``state.handoffs`` from
        # inside ``solver.run``, and re-running the file-based capture
        # would overwrite ``realized_invest`` (and friends) with values
        # read from the prior-handoff-seeded preprocessing CSVs (i.e.
        # the previous solve's state).  Legacy ``run_model`` called
        # ``capture_post_solve`` unconditionally; the cascade then
        # monkey-patched it to a no-op for the same reason
        # (``_orchestration.py:704``).  Omitting the call here is
        # semantically identical to that patch.
        if state.handoffs is not None:
            last_captured_solve = complete_solve[solve]
            state.last_captured_solve = last_captured_solve

        # ---- Scaling report (Agent 10) ----
        # The diagnostic TXT report is gated behind
        # ``FLEXTOOL_SCALING_REPORT=1`` and emitted at most once per base
        # solve name (the ``_roll_N`` suffix is stripped) — matching the
        # cascade-level gate in ``_orchestration._write_scale_csv_and_report``.
        # Across a 72-roll cascade this saves ~44s of redundant work.
        import re as _re
        _report_env = os.environ.get("FLEXTOOL_SCALING_REPORT") == "1"
        _native_base = _re.sub(r"_roll_\d+$", "", solve)
        _native_seen = getattr(state, "_native_report_seen", None)
        if _native_seen is None:
            _native_seen = set()
            state._native_report_seen = _native_seen
        if _report_env and _native_base not in _native_seen:
            _native_seen.add(_native_base)
            try:
                write_scaling_report(
                    scale_table=scale_table,
                    input_dir=wf / "input",
                    solve_data_dir=wf / "solve_data",
                    solve_name=solve,
                    highs_log_path=wf / "HiGHS.log",
                    output_raw_dir=wf / "output_raw",
                    applied_row_scaling=state.solve.use_row_scaling.get(solve),
                    override_source=(
                        ("auto-scale" if applied is not None else "db")
                        if state.solve.use_row_scaling.get(solve) is not None
                        else None
                    ),
                    stdout_summary=True,
                    logger=state.logger,
                )
            except Exception as exc:  # diagnostic only — never fail the solve
                state.logger.warning(
                    f"scaling_report generation failed (non-fatal): {exc}"
                )

        # Save this level's storage fix for child solves to consume.
        # Phase E-d — also refresh the in-memory cross-solve carrier
        # slot from the post-solve fix_storage frames produced by the
        # solver (via SolveHandoff.fix_storage when populated, or via
        # the on-disk file written by ``data.dump_csvs`` when emission
        # is enabled).  The carriers feed the next sub-solve's
        # accumulator (see top of iteration body).
        if complete_solve[solve] in state.solve.fix_storage_periods:
            _q = wf / "solve_data/fix_storage_quantity.csv"
            _p = wf / "solve_data/fix_storage_price.csv"
            _u = wf / "solve_data/fix_storage_usage.csv"
            if _q.exists():
                shutil.copy(
                    str(_q),
                    str(
                        wf
                        / f"solve_data/fix_storage_quantity_{complete_solve[solve]}.csv"
                    ),
                )
                shutil.copy(
                    str(_p),
                    str(
                        wf
                        / f"solve_data/fix_storage_price_{complete_solve[solve]}.csv"
                    ),
                )
                shutil.copy(
                    str(_u),
                    str(
                        wf
                        / f"solve_data/fix_storage_usage_{complete_solve[solve]}.csv"
                    ),
                )
            # Refresh the in-memory carrier from the SolveHandoff (the
            # canonical post-solve carrier).  When the handoff carries
            # ``fix_storage``, fan it out into the per-metric carrier
            # slots so the next iteration's seed lookup succeeds.
            _hf = (
                state.handoffs.get(complete_solve[solve])
                if state.handoffs is not None else None
            )
            if _hf is not None and _hf.fix_storage is not None:
                _per_metric = carriers.setdefault(complete_solve[solve], {})
                _per_metric.update(_fan_out_fix_storage(_hf.fix_storage))
                carriers["__last__"] = _per_metric

    if len(state.solve.model_solve) > 1:
        message = (
            "Trying to run more than one model - not supported. The results "
            "of the first model are retained."
        )
        state.logger.error(message)
        raise FlexToolConfigError(message)
    return 0


__all__ = ["native_run_model"]
