"""Run-scoped timelines: the realization-roll plans, the timelines of the complete solves, the realized-branch
projection of a realized-dispatch sibling, the run's time steps and its existing cohorts."""
from __future__ import annotations

import polars as pl
from flextool.engine.preprocess.params._run_calendar import run_calendar_from_frame
from flextool.engine.solve import _realization_tail
from flextool.engine.input._timeline import _new_step_of_base_steps


def _roll_tail_plans(solve_cfg, rolls: dict, timelines: dict) -> dict:
    """``{sub-solve: RollTail}``: the realization-roll plans of the run (``_realization_tail.roll_tail_plan`` of
    each gated solve's rolls *rolls* on its timeline *timelines*, :func:`_solve_timelines`).  A sub-solve without a
    plan reads its complete solve's ``stochastic_branches`` rows."""
    plans: dict = {}
    for solve, windows in rolls.items():
        plans.update(_realization_tail.roll_tail_plan(solve_cfg, windows, timelines[solve]))
    return plans


def _solve_timelines(state, solves, complete_solve: dict, active_time_lists: dict,
                     wanted) -> dict[str, list[tuple[str, str]]]:
    """``{complete solve: [(period, step), ...]}`` for each complete solve in *wanted*: the timeline its realized
    branch is read on, flattened (``_stochastic.realized_analysis_times``).

    Per period, every step of the timeline of the period's timeset (also those the solve does not use), so a
    realized series made before the solve's first step counts; the steps the rolls of *solves* use when they are
    not all on it.  Periods in the order the rolls first use them.
    """
    by_solve: dict[str, dict[str, dict[str, None]]] = {}
    for s in solves:
        if complete_solve[s] in wanted:
            by_period = by_solve.setdefault(complete_solve[s], {})
            for p, ats in active_time_lists[s].items():
                by_period.setdefault(p, {}).update(dict.fromkeys(e.timestep for e in ats))
    timelines: dict[str, list[tuple[str, str]]] = {}
    for c, by_period in by_solve.items():
        timeset_of = dict(state.solve.timesets_used_by_solves.get(c, []))
        line: dict[tuple[str, str], None] = {}
        for p, active in by_period.items():
            tl = state.timeline.timesets__timeline.get(timeset_of.get(p))
            full = [step for step, *_ in state.timeline.timelines.get(tl, [])]
            line.update(dict.fromkeys((p, step) for step in (full if set(active) <= set(full) else active)))
        timelines[c] = list(line)
    return timelines


def project_realized_branch_time_branch_rows(
    parent_time_branch_list: list[tuple[str, str]],
    parent_period_branch_list: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """W4c — select the parent's REALIZED-branch ``solve_branch__time_branch``
    rows for an auto-spawned realized-dispatch sibling.

    A realized-dispatch sibling is deterministic (``stochastic_branches``
    dropped), so the stochastic pass leaves its
    ``solve_branch__time_branch`` scaffolding EMPTY and the per-solve folds
    that collapse a stochastic 3d_map drop it (wind -> 0).  This re-selects
    the parent's already-materialized realized-branch rows so the folds
    read the realized branch unchanged.

    *parent_time_branch_list* is the parent's
    ``solve_branch__time_branch_lists`` entry — realized rows are
    ``(period, realized_branch)`` (first element == the REAL period name),
    forecast rows are ``(<period>_<branch>, branch)``.
    *parent_period_branch_list* is the parent's ``period__branch_lists``
    entry — the realized branch owns a ``(period, period)`` self-row per
    period.  A row is kept iff its period has a ``(p, p)`` self-row in the
    period-branch list (the SAME realized-row predicate
    ``derive_solve_branch_weight`` uses), which selects the realized rows
    and excludes the synthetic forecast rows.  A MULTI-period realized
    window therefore yields one realized row per realized period (the
    continuation loop appends one ``(period, realized_branch)`` per
    continued period).
    """
    period_branch_set = set(parent_period_branch_list)
    return [
        (period, branch)
        for (period, branch) in parent_time_branch_list
        if (period, period) in period_branch_set
    ]


def run_time_steps(all_solves: list, active_time_lists: dict, complete_solve: dict, solve_config,
                   timeline_config, period__branch_lists: "dict | None" = None) -> "dict[str, pl.DataFrame]":
    """The original time steps each solve of the run uses: complete solve name -> ``(d, t)`` (Utf8, timeline order).

        steps[c] = ∪ { (d, b) : s ∈ all_solves, complete_solve[s] = c, e ∈ active_time_lists[s][d],
                                 b ∈ base(e.timestep) }

    every roll with its look-ahead; ``base`` is the step itself, or for a solve with ``new_stepduration`` the time
    steps of the original timeline that the solve step combines (the timeline of the period's timeset in
    ``timesets_used_by_solves``; a stochastic branch period takes the timeline of the period it branches from,
    ``period__branch_lists``).  Read by the run's input checks before the first solve (``check_run_inputs`` of
    the solver, which builds it only when a group authors a limit as a map).
    """
    timelines = getattr(timeline_config, "timelines", None) or {}
    original_of = getattr(timeline_config, "original_timeline", None) or {}
    timeline_of_timeset = getattr(timeline_config, "timesets__timeline", None) or {}
    used = getattr(solve_config, "timesets_used_by_solves", None) or {}
    base_cache: dict[str, dict[str, list[str]]] = {}

    def _base_of(timeline: "str | None") -> "dict[str, list[str]] | None":
        original = original_of.get(timeline) if timeline is not None else None
        if original is None or original not in timelines or timeline not in timelines:
            return None
        if timeline not in base_cache:
            out: dict[str, list[str]] = {}
            for t, new in _new_step_of_base_steps(timelines[original], timelines[timeline]):
                out.setdefault(new, []).append(t)
            base_cache[timeline] = out
        return base_cache[timeline]

    anchor: dict[str, str] = {}
    for solve in all_solves:
        for period, branch in (period__branch_lists or {}).get(solve, []) or []:
            if branch != period:
                anchor.setdefault(str(branch), str(period))
    pairs: dict[str, dict[tuple[str, str], None]] = {}
    for solve in all_solves:
        cs = complete_solve.get(solve, solve)
        timeline_of_period = {str(d): timeline_of_timeset.get(ts) for d, ts in used.get(cs, ())}
        seen = pairs.setdefault(cs, {})
        for d, entries in (active_time_lists.get(solve) or {}).items():
            base = _base_of(timeline_of_period.get(str(d), timeline_of_period.get(anchor.get(str(d), ""))))
            for e in entries:
                for b in (base.get(e.timestep, [e.timestep]) if base is not None else (e.timestep,)):
                    seen[(str(d), str(b))] = None
    return {cs: pl.DataFrame({"d": [d for d, _ in p], "t": [t for _, t in p]},
                             schema={"d": pl.Utf8, "t": pl.Utf8})
            for cs, p in pairs.items()}


def _run_existing_cohort(state, run_calendar: pl.DataFrame) -> pl.DataFrame:
    """The existing cohorts of the run (:func:`._cohorts.run_existing_cohorts`) from the run's reader
    (``state.input_source``) on the run calendar, in the cascade's axis vocabulary (as the per-solve loads)."""
    from flextool.engine.input._axis_enums import reset_global_axis_enums, set_global_axis_enums
    from flextool.engine.preprocess.params._cohorts import run_existing_cohorts
    source = getattr(state, "input_source", None)
    if source is None:
        raise ValueError("native_run_model needs the run's input-database reader (state.input_source) to place "
                         "the existing capacity of the run.")
    enums = getattr(getattr(state, "cascade_input_provider", None), "axis_enums", None)
    token = set_global_axis_enums(enums) if enums is not None else None
    try:
        return run_existing_cohorts(source, run_calendar_from_frame(run_calendar))
    finally:
        if token is not None:
            reset_global_axis_enums(token)
