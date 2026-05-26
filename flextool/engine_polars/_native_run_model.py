"""Native cascade driver.

This module owns the per-solve cascade loop.  :func:`native_run_model`
is invoked by :func:`._orchestration._drive_cascade`.

Design decisions
----------------

* **Cascade loop is native; inner preprocessing is not (yet).**
  Scope: own the cascade walk and inline the solve-tree expansion /
  stochastic branching / per-solve setup; not re-port
  ``preprocessing.solve_time`` or the ``solve_writers`` module.  Those
  remain authoritative and are still called as functions — the
  ``_native_leaf_set_override()`` context (wired by
  :func:`._orchestration._drive_cascade`) intercepts the already-native
  preprocessing helpers.
* **Native solve-tree expansion** via
  :class:`flextool.engine_polars._recursive_solve.RecursiveSolveBuilder`
  + :class:`._stochastic.StochasticSolver`.
* **Direct handoff construction.**  The post-solve ``SolveHandoff`` is
  built by ``build_handoff_from_solution`` directly from the polar_high
  ``Solution`` object — no CSV round-trip.
* **Optional state fields tolerated.**  Native :class:`RunnerState`
  may lack ``timing_recorder`` / ``auto_scale``.  Callers that consume
  those guard with ``getattr(state, name, default)``.
"""
from __future__ import annotations

import copy
import csv
import os
import time
from collections import defaultdict

# ---------------------------------------------------------------------------
# Imports.  Per Phase 3 scope, we depend on legacy preprocessing /
# solve_writer modules directly — Phase 2 covers the override hook that
# intercepts the already-ported helpers.  Phases 4+ will retire the
# remaining writers.
# ---------------------------------------------------------------------------

from flextool.engine_polars._blocks import emit_block_data_for_solve
# Step 2.5 — legacy preprocessing package deleted (item 15).  The per-
# solve orchestrator now lives natively at
# :mod:`flextool.engine_polars._emit_solve_time`.
from flextool.engine_polars import (
    _emit_solve_time as preprocessing_solve_time,
)
from flextool.engine_polars._solve_state import (
    FlexToolConfigError,
    FlexToolSolveError,
)
# Step 2.5 — solve_writers calls now resolve to the native polars
# implementations.  The legacy disk-writing module remains in the tree
# but is no longer called from the cascade.
from flextool.engine_polars import _emit_solve_writers as solve_writers
from flextool.engine_polars import _provider_keys
from flextool.engine_polars import _provider_translators

from flextool.engine_polars._flex_data_provider import FlexDataProvider
from flextool.engine_polars._solve_state import compute_level_key

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


def _nodes_with_blended_weights(input_dir, provider) -> list[str]:
    """Return nodes carrying ``bind_within_solve_blended_weights`` per input CSV.

    Reads ``input/node__storage_binding_method.csv`` via the seeded
    Provider (the file may live only in-memory under the per-level
    Provider's input/ namespace, never written to disk).  Returns a
    sorted list of node names; empty if the file is absent or has no
    blended-weights rows.  Path / provider-key conventions mirror
    :func:`flextool.engine_polars._emit_leaf_sets.derive_node_state_subset`.
    """
    from flextool.engine_polars._emit_leaf_sets import _read_csv
    import polars as pl
    sbm = _read_csv(
        input_dir / "node__storage_binding_method.csv",
        ["node", "storage_binding_method"],
        provider=provider,
    )
    if sbm.height == 0:
        return []
    if "storage_binding_method" in sbm.columns:
        method_col = "storage_binding_method"
    elif "method" in sbm.columns:
        method_col = "method"
    else:
        return []
    return sorted(
        sbm.filter(pl.col(method_col) == "bind_within_solve_blended_weights")
           .select("node")
           .unique()
           .get_column("node")
           .to_list()
    )


def _assert_blended_weights_have_rp_weights(
    *, solve, complete_solve_name, roll_index,
    active_timeset_names, rp_weights_keys,
    input_dir, provider, logger,
) -> None:
    """Strict per-solve precondition: blended-weights nodes need RP weights.

    Fires when at least one node carries
    ``bind_within_solve_blended_weights`` but none of the solve's active
    timesets has an entry in ``state.timeline.rp_weights``.  Surfaces
    the misconfiguration at the solve boundary with full context
    (solve name, complete-solve name, roll index, active timesets,
    offending nodes, two concrete fixes), so the user does not have to
    chase the late ``FlexData loader`` backstop in ``input.py``.

    Raises:
        FlexToolConfigError: when the precondition is violated.
    """
    nodes = _nodes_with_blended_weights(input_dir, provider)
    if not nodes:
        return  # no blended-weights nodes — non-RP path is correct
    # If we got here the caller has already established that
    # ``rp_written is False`` (no active timeset has RP weights), so we
    # only need the offending-node list to fail the check.
    head = nodes[:5]
    if len(nodes) > 5:
        node_summary = (
            f"{', '.join(head)}, ... {len(nodes) - 5} more"
        )
    else:
        node_summary = ', '.join(head)
    timeset_summary = (
        ', '.join(active_timeset_names) if active_timeset_names else "(none)"
    )
    rp_weights_summary = (
        ', '.join(sorted(rp_weights_keys)) if rp_weights_keys
        else "(no timeset has representative_period_weights)"
    )
    message = (
        f"Solve '{solve}' (complete '{complete_solve_name}', roll "
        f"{roll_index}) has {len(nodes)} node(s) carrying "
        f"`storage_binding_method = bind_within_solve_blended_weights` "
        f"({node_summary}) but none of its active timeset(s) "
        f"[{timeset_summary}] supply `representative_period_weights`. "
        f"Timesets that DO have RP weights in this run: "
        f"[{rp_weights_summary}]. Without RP weights the model would "
        f"silently degrade to the non-RP path and the deep "
        f"`FlexData loader: nodeState_rp is non-empty ...` invariant "
        f"would trip later. Either (a) attach an alternative to this "
        f"scenario that supplies `representative_period_weights` for "
        f"the active timeset, or (b) change these nodes' "
        f"`storage_binding_method` to a non-RP value "
        f"(bind_within_solve / bind_within_period / bind_within_timeblock / "
        f"bind_forward_only) using a scenario alternative override. "
        f"In Spine DB toolbox: Database editor -> Scenario tree, edit "
        f"the scenario's alternatives list to include an alternative "
        f"that defines `representative_period_weights` on the timeset, "
        f"or override `node.storage_binding_method` per-node via a "
        f"non-RP alternative."
    )
    logger.error(message)
    raise FlexToolConfigError(message)


def native_run_model(state, solver) -> int:
    """Drive the per-solve cascade natively.

    Walks the solve tree, applies stochastic branching, writes the
    per-solve inputs the solver needs (via ``solve_writers`` /
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
            this is ``_PolarHighCascadeSolver`` defined inside
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

    # Step 2.5-E Phase A — Provider-routed pdt -> {pd, pt} split.  The
    # cascade-input Provider on ``state`` already carries the seeded
    # ``input/pdt_commodity`` / ``input/pdt_group`` frames; the two
    # derived shards land back on the Provider under the same parent-
    # qualified keys (``input/pd_*``, ``input/pt_*``).
    cascade_input_provider_seed: "FlexDataProvider | None" = getattr(
        state, "cascade_input_provider", None,
    )
    if cascade_input_provider_seed is None:
        cascade_input_provider_seed = FlexDataProvider()
        state.cascade_input_provider = cascade_input_provider_seed
    separate_period_and_timeseries_data(
        state.timeline.timelines,
        state.solve.timesets_used_by_solves,
        provider=cascade_input_provider_seed,
        work_folder=wf,
    )

    # Step 1-f — cascade-wide Provider seeded once and re-used for every
    # sub-solve's pre-populated frames.  Captures the
    # ``solve_writers.emit_timesets`` output (``input/timesets_in_use.csv``
    # + ``input/timesets__timeline.csv``) so the per-sub-solve preprocessing
    # readers can find them via ``provider.get`` without touching disk.
    cascade_input_provider: "FlexDataProvider | None" = getattr(
        state, "cascade_input_provider", None,
    )
    if cascade_input_provider is None:
        cascade_input_provider = FlexDataProvider()
        state.cascade_input_provider = cascade_input_provider

    # Solve-loop-invariant timesets — hoisted out of the per-solve loop.
    solve_writers.emit_timesets(
        state.solve.timesets_used_by_solves,
        state.timeline.timesets__timeline,
        provider=cascade_input_provider,
    )

    # ------------------------------------------------------------------
    # 5. Per-solve loop.
    # ------------------------------------------------------------------
    first = True
    previous_complete_solve = None
    last_captured_solve: str | None = None
    cached_complete_active_time_lists: dict = {}

    # Pre-compute per-iter level keys.  Same-key consecutive iters are
    # rolling continuations of one base solve; a key change marks a
    # sequential / nested level transition.  Used to gate phase-progress
    # output: within-group rolling iters emit only the "Solve start"
    # marker, while the last iter of each group emits the four phase
    # checkpoints whose deltas aggregate across the whole group.
    level_keys = [
        compute_level_key(
            solve_name=s,
            complete_solve_name=complete_solve[s],
            solve_config=state.solve,
            timeline_config=state.timeline,
        )
        for s in all_solves
    ]

    for i, solve in enumerate(all_solves):
        timer_in_solve = time.perf_counter()

        # Blank line at level-group boundaries only.  Within-group
        # rolling iters print only the marker line, so a blank line
        # between them would fragment the marker block visually.
        if i > 0 and level_keys[i] != level_keys[i - 1]:
            print("", flush=True)

        # Per-sub-solve marker — plain text line (no timer/memory).
        # Always printed so the user can see roll progression even
        # when the four phase checkpoints are suppressed for within-
        # group rolling iters.
        _memrec_iter = getattr(state, "_memory_recorder", None)
        if _memrec_iter is not None and getattr(_memrec_iter, "verbose", True):
            try:
                print(
                    f"Solve start: {complete_solve[solve]}, "
                    f"{i + 1}/{len(all_solves)}",
                    flush=True,
                )
            except OSError:
                pass

        state.logger.debug(
            f"Creating timelines for solve {solve} ({i})"
        )
        cs = complete_solve[solve]

        # Per-level Provider detection (Design A, step A2).  Below we
        # reuse a single FlexDataProvider per distinct level_key so
        # iters at the same level share one Provider instance.
        _level_key = level_keys[i]
        # level_key is debug-only output (verbose mem mode):
        # ``runner.state.logger`` is forced to ERROR level by the
        # orchestration driver, so the regular ERROR-level print would
        # always escape.  Gate behind the same env var that surfaces
        # the full mem-checkpoint trace.
        if os.environ.get("FLEXTOOL_MEMORY_VERBOSE") == "1":
            state.logger.error(
                "level_key for solve %r (complete=%r): %r",
                solve, complete_solve[solve], _level_key,
            )

        # Per-level boundary (Design A, step A3): when the level_key
        # changes from the previous iter, drop any warm-LP carry-over
        # on the cascade solver so we don't accidentally reuse
        # structures keyed on the prior level's matrix shape.  The
        # warm-LP fingerprint check inside ``_PolarHighCascadeSolver.run``
        # will ALSO null these on shape change (a level transition
        # always changes the FlexData shape in practice), but doing
        # it explicitly at the level boundary makes the intent
        # explicit and gives subprocess-per-chain work a clean
        # lifecycle hook.
        _last_level_key = getattr(state, "_last_level_key_seen", None)
        if _last_level_key is not None and _last_level_key != _level_key:
            if hasattr(solver, "_warm_problem"):
                solver._warm_problem = None
            if hasattr(solver, "_prior_data"):
                solver._prior_data = None
        state._last_level_key_seen = _level_key

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

        # Per-level Provider (Design A, step A2).  Two consecutive
        # iters with the same ``_level_key`` reuse the same Provider —
        # e.g. the 72 dispatch rolls on the multi-invest fixture all
        # share one Provider, while each of the 4 invest sub-solves
        # (distinct period windows -> distinct keys) gets its own.
        # On level transition, build a fresh Provider seeded from the
        # cascade-input frames; per-iter writers below overwrite their
        # own keys in place across iters at the same level.
        if not hasattr(state, "_level_providers"):
            state._level_providers = {}
        sub_solve_provider = state._level_providers.get(_level_key)
        if sub_solve_provider is None:
            sub_solve_provider = FlexDataProvider()
            # Seed cascade-wide ``input/*.csv`` frames so per-iter
            # readers find them via ``provider.get``
            # (e.g. ``input/timesets_in_use.csv``).
            for _key, _frame in cascade_input_provider.items():
                sub_solve_provider.put(_key, _frame)
            state._level_providers[_level_key] = sub_solve_provider
        # else: existing Provider — already seeded; per-iter writers
        # below will overwrite their own keys in place.
        # Resolve nesting parent's complete-solve name; reused below by
        # the parent-handoff translator call (see Phase 4.1e).
        _parent_solve_for_carriers = parent_roll.get(solve)
        _parent_complete_for_carriers = (
            complete_solve.get(_parent_solve_for_carriers)
            if _parent_solve_for_carriers else None
        )

        # S1-g-3 — expose the per-sub-solve Provider to writer entry
        # points BEFORE preprocessing runs, so native writers threaded
        # with ``provider=`` (via :func:`_emit_solve_time.run`) can
        # fetch it from ``state.current_provider``.  Writers not yet
        # threaded still resolve their reads via the Provider-as-seed
        # bridge installed above; Step 2 deletes both paths.
        state.current_provider = sub_solve_provider

        # Memory checkpoints — fire only on the first sub-solve so the
        # output stays readable on multi-roll scenarios.  Used to
        # attribute the pre-load_flextool preprocessing-chain memory
        # spike across writer groups.
        _mem_cp = None
        if i == 0:
            _memrec_native = getattr(state, "_memory_recorder", None)
            if _memrec_native is not None:
                def _mem_cp(label: str, user_label: str,
                            _rec=_memrec_native, _log=state.logger) -> None:
                    _rec.checkpoint(label, _log, user_label=user_label)
        if _mem_cp is not None:
            _mem_cp("prep_seeded",
                    "prep: provider seeded")

        solve_writers.emit_full_timelines(
            state.timeline.stochastic_timesteps[solve],
            period__timesets_with_history,
            state.timeline.timesets__timeline,
            state.timeline.timelines,
            str(wf / "solve_data/steps_in_timeline.csv"),
            provider=sub_solve_provider,
        )
        solve_writers.emit_active_timelines(
            active_time_lists[solve],
            str(wf / "solve_data/steps_in_use.csv"),
            provider=sub_solve_provider,
        )
        solve_writers.emit_active_timelines(
            complete_active_time_lists,
            str(wf / "solve_data/steps_complete_solve.csv"),
            complete=True,
            provider=sub_solve_provider,
        )
        solve_writers.emit_step_jump(
            jump_lists[solve],
            provider=sub_solve_provider,
        )
        pb_time, pb_succ = make_period_block(active_time_lists[solve])
        solve_writers.emit_period_block(
            pb_time, pb_succ,
            provider=sub_solve_provider,
        )

        if _mem_cp is not None:
            _mem_cp("prep_timeline_writers_done",
                    "prep: timeline writers done")

        state.logger.debug("Creating period data")
        solve_writers.emit_period_years(
            period__branch_lists[solve],
            solve_period_history[complete_solve[solve]],
            str(wf / "solve_data/period_with_history.csv"),
            provider=sub_solve_provider,
        )
        solve_writers.emit_periods(
            complete_solve[solve],
            state.solve.realized_invest_periods,
            str(wf / "solve_data/realized_invest_periods_of_current_solve.csv"),
            provider=sub_solve_provider,
        )
        # If realized_invest_periods is empty but both invest_periods and
        # realized_periods are defined, fall back to realized_periods.
        if (
            not state.solve.realized_invest_periods[complete_solve[solve]]
            and state.solve.invest_periods[complete_solve[solve]]
            and state.solve.realized_periods[complete_solve[solve]]
        ):
            solve_writers.emit_periods(
                complete_solve[solve],
                state.solve.realized_periods,
                str(
                    wf
                    / "solve_data/realized_invest_periods_of_current_solve.csv"
                ),
                provider=sub_solve_provider,
            )
        sub_solve_provider.put(
            _provider_keys.SOLVE_DATA_INVEST_PERIODS_OF_CURRENT_SOLVE,
            solve_writers.derive_periods(
                complete_solve[solve], state.solve.invest_periods,
            ),
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
        solve_writers.emit_years_represented(
            period__branch_lists[solve],
            years_rep,
            str(wf / "solve_data/p_years_represented.csv"),
            provider=sub_solve_provider,
        )
        solve_writers.emit_period_years(
            period__branch_lists[solve],
            years_rep,
            str(wf / "solve_data/p_discount_years.csv"),
            provider=sub_solve_provider,
        )
        solve_writers.emit_current_solve(
            solve, str(wf / "solve_data/solve_current.csv"),
            provider=sub_solve_provider,
        )

        if _mem_cp is not None:
            _mem_cp("prep_period_writers_done",
                    "prep: period writers done")

        # ---- LP scaling: emit user-set row-scaling state and the
        # header-only objective-scaling CSV.  ``analyze_solve`` runs
        # post-solve from ``_orchestration.py`` for reporting (Γ-scale);
        # auto-row-scaling and objective auto-scaling were removed in
        # Δ.22 along with the ``--auto-scale`` CLI flag.
        solve_writers.emit_p_use_row_scaling(
            solve,
            state.solve.use_row_scaling,
            str(wf / "solve_data/p_use_row_scaling.csv"),
            provider=sub_solve_provider,
        )
        solve_writers.emit_scale_the_objective_header_only(
            wf / "solve_data",
            provider=sub_solve_provider,
        )

        if _mem_cp is not None:
            _mem_cp("prep_scaling_done", "prep: scaling done")

        solve_writers.emit_first_steps(
            active_time_lists[solve],
            str(wf / "solve_data/first_timesteps.csv"),
            provider=sub_solve_provider,
        )
        solve_writers.emit_last_steps(
            active_time_lists[solve],
            str(wf / "solve_data/last_timesteps.csv"),
            provider=sub_solve_provider,
        )

        state.logger.debug("Create realized timeline")
        solve_writers.emit_realized_dispatch(
            realized_time_lists[solve],
            complete_solve[solve],
            state.solve.realized_periods.get(complete_solve[solve], []),
            provider=sub_solve_provider,
        )
        solve_writers.emit_fix_storage_timesteps(
            fix_storage_time_lists[solve],
            complete_solve[solve],
            state.solve.fix_storage_periods.get(complete_solve[solve], []),
            provider=sub_solve_provider,
        )
        solve_writers.emit_delayed_durations(
            active_time_lists[solve],
            complete_solve[solve],
            state.solve.delay_durations,
            provider=sub_solve_provider,
        )

        if _mem_cp is not None:
            _mem_cp("prep_step_writers_done",
                    "prep: step + realized + fix_storage + delayed writers done")

        state.logger.debug("Possible stochastics")
        solve_writers.emit_branch__period_relationship(
            period__branch_lists[solve],
            str(wf / "solve_data/period__branch.csv"),
            provider=sub_solve_provider,
        )
        solve_writers.emit_branch_weights_and_map(
            complete_solve[solve],
            active_time_lists[solve],
            solve_branch__time_branch_lists[solve],
            branch_start_time_lists[solve],
            period__branch_lists[solve],
            state.solve.stochastic_branches,
            provider=sub_solve_provider,
        )
        solve_writers.emit_first_and_last_periods(
            active_time_lists[solve],
            state.solve.timesets_used_by_solves[complete_solve[solve]],
            period__branch_lists[solve],
            provider=sub_solve_provider,
        )

        if _mem_cp is not None:
            _mem_cp("prep_branch_writers_done",
                    "prep: branch + first_and_last_periods writers done")

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
            # Step 2.5-E Phase C — route averaged-timeseries reads /
            # writes through the cascade-input Provider so the
            # ``solve_data/pt_*`` frames persist across every sub-solve
            # belonging to the same ``complete_solve``.  Legacy code
            # relied on the on-disk ``solve_data/pt_*.csv`` files
            # remaining valid for all rolls of one ``complete_solve``;
            # in Provider-land the equivalent is keeping the frames on
            # the cascade-input Provider so each fresh sub-solve
            # Provider seeds them at the top of its iter (line 444).
            state.timeline.create_averaged_timeseries(
                complete_solve[solve], state.solve, state.logger,
                provider=cascade_input_provider,
                work_folder=wf,
            )
            # Mirror the freshly aggregated frames onto the current
            # sub-solve Provider too — this iter's readers must find
            # them without waiting for the next seed pass.
            for _key, _frame in cascade_input_provider.items():
                if _key.startswith("solve_data/pt_") or _key.startswith(
                    "solve_data/pbt_",
                ):
                    sub_solve_provider.put(_key, _frame)
        previous_complete_solve = complete_solve[solve]

        # ---- Block data (Agent 1.1) ----
        # Δ.31: thread the per-sub-solve Provider so the legacy reads
        # find the ``input/*.csv`` frames in memory.  Without this the
        # cascade silently produces a single-block layout (the input/
        # CSVs are kept on the Provider rather than flushed to disk),
        # collapsing every coarse-block fixture (e.g. lh2_three_region)
        # to the default block and breaking the daily-block aggregation.
        try:
            emit_block_data_for_solve(
                solve=complete_solve[solve],
                solve_config=state.solve,
                timeline_config=state.timeline,
                work_folder=wf,
                active_time_list=active_time_lists[solve],
                default_jump_list=jump_lists[solve],
                provider=sub_solve_provider,
                emit_provider=sub_solve_provider,
            )
        except FlexToolConfigError:
            raise
        except Exception as exc:  # pragma: no cover — defensive only
            state.logger.warning(
                f"blocks: emission failed (non-fatal — not consumed yet): {exc}"
            )

        first_of_nested_level = solve in state.solve.first_of_complete_solve
        last_of_nested_level = solve in state.solve.last_of_solve

        # Phase 4.1j — the parent-to-child fix_storage hand-off is now
        # carried exclusively by the typed ``handoff/*`` Provider keys
        # seeded at iteration start by the parent-handoff translator
        # (Phase 4.1e).  The legacy shutil.copy archive path and the
        # in-memory carrier fallback that wrote to the now-dead
        # ``solve_data/fix_storage_*`` Provider keys have been deleted;
        # all downstream consumers (input.py, _derived_params.py,
        # _emit_arc_unions.py, _emit_per_solve.py) consult ``handoff/*``.

        solve_writers.emit_solve_status(
            first_of_nested_level, last_of_nested_level,
            nested=True,
            provider=sub_solve_provider,
        )
        last = i == len(solves) - 1
        solve_writers.emit_solve_status(
            first, last,
            provider=sub_solve_provider,
        )
        if i == 0:
            first = False
            solve_writers.emit_empty_investment_file(
                provider=sub_solve_provider,
            )
            solve_writers.emit_empty_storage_fix_file(
                provider=sub_solve_provider,
            )
            solve_writers.emit_headers_for_empty_output_files(
                str(wf / "solve_data/costs_discounted.csv"),
                "param_costs,costs_discounted",
                provider=sub_solve_provider,
            )
            solve_writers.emit_headers_for_empty_output_files(
                str(wf / "solve_data/co2.csv"),
                "param_co2,model_wide",
                provider=sub_solve_provider,
            )
            solve_writers.emit_headers_for_empty_output_files(
                str(wf / "solve_data/period_capacity.csv"),
                "period",
                provider=sub_solve_provider,
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
                    timeline_name = state.timeline.timesets__timeline[ts_name]
                    timeline_steps = [
                        step
                        for step, _dur in state.timeline.timelines.get(
                            timeline_name, []
                        )
                    ]
                    solve_writers.emit_rp_data(
                        rp_weights=state.timeline.rp_weights[ts_name],
                        timeset_duration_entries=state.timeline.timeset_durations[
                            ts_name
                        ],
                        period_name=period_name,
                        timeline_steps=timeline_steps,
                        provider=sub_solve_provider,
                    )
                    rp_written = True
                    break
        if not rp_written:
            # Strict precondition: if any node in this solve carries
            # ``bind_within_solve_blended_weights`` but the active timeset has
            # no RP weights, surface the misconfiguration here (with
            # full solve / scenario context) rather than letting the
            # deep ``FlexData loader`` backstop in ``input.py`` trip on
            # internal frame names.  Phase 5 of the storage_binding_method
            # single-valued cleanup.
            _assert_blended_weights_have_rp_weights(
                solve=solve,
                complete_solve_name=complete_solve[solve],
                roll_index=i,
                active_timeset_names=active_timeset_names,
                rp_weights_keys=list(state.timeline.rp_weights.keys()),
                input_dir=wf / "input",
                provider=sub_solve_provider,
                logger=state.logger,
            )
            solve_writers.emit_empty_rp_data(
                provider=sub_solve_provider,
            )
            solve_writers.emit_timeset_cost_weight(
                active_time_list=active_time_lists[solve],
                timesets_used_by_solve=timesets_used,
                timeset_weights=state.timeline.timeset_weights,
                provider=sub_solve_provider,
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
        # override hook intercepts already-ported helpers).  The prior
        # SolveHandoff is fanned into ``handoff/<field>`` Provider keys
        # via :func:`_provider_translators.translate_handoff_to_provider`
        # so cascade consumers go through ``provider.get(K.HANDOFF_X)``
        # — no ``prior_handoff`` parameter is threaded through the
        # cascade (Phase 2 of specs/provider_consolidation.md).
        prior_handoff = (
            state.handoffs.get(last_captured_solve)
            if state.handoffs is not None and last_captured_solve is not None
            else None
        )
        _provider_translators.translate_handoff_to_provider(
            prior_handoff, sub_solve_provider,
        )
        # Phase 4.1e — when nested, parent solve's handoff shadows the
        # sequential prior in ``handoff/*`` keys: both translator calls
        # write the same Provider keys, and parent's call lands after
        # sequential's so parent's values win where both are populated.
        # Reuses ``_parent_complete_for_carriers`` resolved earlier in
        # the same iteration scope.  Guarded by ``if parent_handoff is
        # not None`` so non-nested cascades skip the second call
        # entirely; calling the translator with ``None`` would write
        # empty frames via its empty-schema fallback, obliterating the
        # sequential prior's just-written data.
        parent_handoff = (
            state.handoffs.get(_parent_complete_for_carriers)
            if state.handoffs is not None and _parent_complete_for_carriers is not None
            else None
        )
        if parent_handoff is not None:
            _provider_translators.translate_handoff_to_provider(
                parent_handoff, sub_solve_provider,
            )
        # Phase 5b — external overrides shadow handoff via the
        # ``override/*`` Provider layer.  The callable is set by
        # external code wrapping the runner (file-watch / ZeroMQ /
        # etc.); default is None (no overrides).  Lands AFTER the
        # sequential + parent handoff translators so override values
        # take precedence in ``read_handoff_frame`` consumers
        # (Phase 5a infrastructure).  If the callable raises, the
        # exception propagates — external code owns clean error
        # reporting.
        _override_provider = getattr(state, "override_provider", None)
        if _override_provider is not None:
            overrides = _override_provider()
            if overrides:
                _provider_translators.translate_overrides_to_provider(
                    overrides, sub_solve_provider,
                )
                state.logger.info(
                    f"[override] applied {len(overrides)} keys at "
                    f"iter={i} solve={complete_solve[solve]}"
                )
                state.logger.debug(
                    f"[override] keys: {sorted(overrides.keys())}"
                )
        _phase_timing = (
            os.environ.get("FLEXTOOL_PHASE_TIMING") == "1"
            and timing_recorder is not None
        )
        _t_preproc_start = time.perf_counter() if _phase_timing else 0.0
        if _mem_cp is not None:
            _mem_cp("prep_before_solve_time_dispatcher",
                    "prep: block + rp/timeset + status + empty writers done")
        # Step 1-f — preprocessing emits directly into
        # ``sub_solve_provider`` via the threaded ``provider=`` keyword;
        # the Provider is the sole in-memory carrier.
        preprocessing_solve_time.run(
            state, complete_solve[solve],
            provider=sub_solve_provider,
        )
        if _mem_cp is not None:
            _mem_cp("prep_solve_time_dispatcher_done",
                    "prep: _emit_solve_time.run (per-solve sets + params dispatcher) done")
        # Step 1-f — Provider stash on state.  Read by
        # ``_PolarHighCascadeSolver.run`` to thread through into
        # ``load_flextool`` / ``write_outputs_for_solve`` /
        # ``build_handoff_from_solution``.  Replaces any prior
        # sub-solve's Provider (per-sub-solve memory discipline).
        state.current_provider = sub_solve_provider
        if _phase_timing:
            timing_recorder.record(
                "per_iter",
                subphase="preprocessing",
                solve=complete_solve[solve],
                roll_index=i,
                seconds=time.perf_counter() - _t_preproc_start,
                t_start=_t_preproc_start,
            )

        # Phase 6b — opt-in source-tagging audit dump.  When
        # ``FLEXTOOL_AUDIT_SOURCES=1`` is set in the environment, append
        # every Provider key carrying a non-None source tag to
        # ``<work_folder>/audit_sources.log``.  The override translator
        # tags its writes with ``source="external_override"`` (Phase 6a);
        # other writes leave the source slot empty, so the log captures
        # exactly the externally-injected entries for the just-completed
        # preprocessing pass.  Append mode accumulates across sub-solves.
        if os.environ.get("FLEXTOOL_AUDIT_SOURCES") == "1":
            _provider_translators.dump_provider_sources(
                sub_solve_provider,
                wf / "audit_sources.log",
                complete_solve[solve],
            )

        # Phase 4 (Gap F) — expose the upper-level (nesting) parent's
        # complete solve name so ``_PolarHighCascadeSolver.run`` can look the
        # parent's :class:`SolveHandoff` up out of ``state.handoffs`` and
        # pass it to ``build_handoff_from_solution`` (which uses it to skip
        # the workdir's ``fix_storage_{price,usage}.csv`` reads).  Resets
        # to None for top-level solves.
        _parent_solve = parent_roll.get(solve)
        state.current_parent_complete = (
            complete_solve.get(_parent_solve) if _parent_solve else None
        )

        # Tell the solver whether to emit the four per-iter phase
        # checkpoints (FlexData built / Matrix built / Solver /
        # Outputs written).  Fires on the last iter of each level group
        # so the recorded deltas aggregate across all rolls in the group.
        state.emit_phase_checkpoints_this_iter = (
            i == len(all_solves) - 1
            or level_keys[i + 1] != _level_key
        )
        exit_status = solver.run(complete_solve[solve])
        state.emit_phase_checkpoints_this_iter = False
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

        # In-memory handoff bookkeeping.  ``solver.run`` deposits a
        # polar_high-derived ``SolveHandoff`` into ``state.handoffs`` via
        # ``build_handoff_from_solution``.  Phase 3 of
        # specs/provider_consolidation.md deleted the legacy
        # ``capture_post_solve`` constructor — nothing reads from disk
        # to overwrite that handoff.
        if state.handoffs is not None:
            last_captured_solve = complete_solve[solve]
            state.last_captured_solve = last_captured_solve
            # Phase 4.1a — refresh ``handoff/*`` Provider keys from the
            # post-solve ``SolveHandoff`` so the current iteration's
            # Provider exposes the FINAL cumulative state of this roll
            # (not just the prior-roll state seeded at iteration start).
            # ``csv_dump`` snapshots the Provider after the last roll;
            # without this refresh the on-disk
            # ``handoff/cumulative_commodity.csv`` would reflect the
            # second-to-last roll instead of the final one.  The next
            # iteration's iteration-start translator is unaffected: it
            # reads from ``state.handoffs[last_captured_solve]`` directly.
            _latest_handoff = state.handoffs.get(last_captured_solve)
            if _latest_handoff is not None:
                _provider_translators.translate_handoff_to_provider(
                    _latest_handoff, sub_solve_provider,
                )

        # ---- Scaling report — handled by the orchestrator.
        # The diagnostic TXT report is emitted by
        # ``_orchestration._write_scale_csv_and_report`` (called from
        # the cascade post-solve path at ``_orchestration.py:1352``).
        # The duplicate native call site was removed in Tier 4
        # Commit 4 since the orchestrator's call uses the modern
        # ``flex_data`` / ``Solution`` API and emits the same
        # ``solve_data/scaling_report.txt`` artifact with identical
        # base-name deduplication.

        # Phase 4.1l — the post-solve fix_storage carrier refresh that
        # used to fan the wide ``SolveHandoff.fix_storage`` frame into
        # per-metric ``solve_data/fix_storage_*`` Provider keys is
        # retired.  The narrow ``handoff/fix_storage_{quantity,price,
        # usage}`` keys are seeded directly by the iteration-start
        # parent-handoff translator from the three narrow SolveHandoff
        # fields; no post-solve fan-out is required.

        # Phase 4.1a — the ladder rolling accumulators
        # (``cumulative_commodity`` / ``cum_sim_hours``) cross sub-solves
        # via the iteration-start handoff translator
        # (``translate_handoff_to_provider``).  No fan-out needed here:
        # the next iteration reads ``handoff/cumulative_commodity`` /
        # ``handoff/cum_sim_hours`` directly from the typed SolveHandoff.

    if len(state.solve.model_solve) > 1:
        message = (
            "Trying to run more than one model - not supported. The results "
            "of the first model are retained."
        )
        state.logger.error(message)
        raise FlexToolConfigError(message)
    return 0


__all__ = ["native_run_model"]
