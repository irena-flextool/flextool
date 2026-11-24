"""The realization roll R_k of a rolling stochastic solve: which solves get one, and the plan of each roll.

A rolling stochastic solve that runs ``dispatch_realization_method`` ``within_solve`` or ``separate_dispatch``
realizes each jump ``[s_k, s_{k+1})`` on its ``realized: yes`` branch, read as one timeline (the rule of a
``deterministic`` solve), and looks ahead from ``s_{k+1}`` (the end of the jump) with the forecasts made there.  ``within_solve``: the realization roll is the roll itself; ``separate_dispatch``:
the roll ``k`` of the engine-added realization solve, after the roll's fully stochastic solve (1).

* :func:`realization_roll_method` is the gate: the one predicate that says whether a solve has realization rolls
  (also read by ``_orchestration._spawn_realized_dispatch_siblings``, which notes a solve the delayed units alone
  keep from them, :func:`delay_gated_units`).
* :func:`roll_tail_plan` makes the :class:`RollTail` of each roll of a gated solve from the authored
  ``stochastic_branches`` rows (never changed) and raises the configuration errors of a missing forecast or a
  missing, ambiguous or unreachable ``realized: yes`` branch.
* :func:`roll_branch_rows` gives each planned roll the forecast rows of its tail (made at ``s_{k+1}``), read by the
  stochastic expansion and the branch-weight emit through :func:`plan_branch_rows` (no plan: the complete solve's
  rows, as before).  The realized branch of the jump comes from the plan (``RollTail.realized``), not from rows.
* :func:`check_tail_weights` (G2b): every forecast of a planned roll has its own branch weight.
* :func:`announce_realization_notices`: W-0635 / W-0636, once the plans and the expansion have passed their checks.

The plans are made in ``native_run_model`` through the seam ``_run_model._roll_tail_plans``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, NamedTuple

import polars as pl

from flextool.engine.input._solve_state import FlexToolConfigError

RollMethod = Literal["within_solve", "separate_dispatch"]
RollShape = Literal["split", "seam", "deterministic"]


@dataclass(frozen=True)
class RollTail:
    """The plan of one realization roll R_k (one sub-solve).

    ``jump_steps`` are the ``(period, step)`` cells of the jump ``[s_k, s_{k+1})`` in timeline order: dispatched once,
    under their own period names, on the realized data.  ``tail_start`` is ``(p, s_{k+1})``, the first step after the
    jump in the roll's horizon (``None`` without one).  ``cohort`` (C_k*) are the forecasts made at ``tail_start``
    (``stochastic_branches`` rows of period ``p`` with analysis time ``s_{k+1}``, weight > 0, not realized), in
    database order.  ``shape``: ``split`` when ``s_{k+1}`` lies inside ``p`` (the jump holds steps of ``p``),
    ``seam`` when it is ``p``'s first step in the roll, ``deterministic`` when the horizon ends with the jump (no tail,
    ``deterministic`` is then True and ``cohort`` empty).

    The jump (every period of a ``deterministic`` roll) reads ``realized``, the one ``realized: yes`` branch of the
    complete solve, flattened into one timeline: one analysis-time tier per entry of ``realized_times``, the one
    nearest to the roll start ``s_k`` first, the earlier one first on a tie (``_stochastic.realized_analysis_times``,
    the rule of ``dispatch_realization_method`` ``deterministic``).
    """

    cohort: tuple[str, ...]
    shape: RollShape
    deterministic: bool
    jump_steps: tuple[tuple[str, str], ...]
    tail_start: tuple[str, str] | None
    realized: str
    realized_times: tuple[str, ...]


class RollWindow(NamedTuple):
    """The time window of realization roll ``name``: ``active`` is the roll's horizon and ``jump`` its realized part
    (``{period: [ActiveTimeEntry, ...]}`` before the stochastic expansion), ``solve`` the stochastic complete solve
    whose ``stochastic_branches`` the roll reads.  For ``separate_dispatch`` ``name`` is the roll of the realization
    solve and the window is that of the stochastic roll it realizes."""

    name: str
    solve: str
    active: dict
    jump: dict


def _top_level_solves(solve_cfg: Any) -> set[str]:
    top: set[str] = set()
    for solves in (getattr(solve_cfg, "model_solve", None) or {}).values():
        top.update(solves)
    return top


def _contained_solves(solve_cfg: Any) -> set[str]:
    contained: set[str] = set()
    for children in (getattr(solve_cfg, "contains_solves", None) or {}).values():
        if isinstance(children, (list, tuple, set)):
            contained.update(children)
        elif children is not None:
            contained.add(children)
    return contained


def delayed_units(solve_cfg: Any) -> list[str]:
    """The units of the model with a ``delay`` (``SolveConfig.delay_durations``), sorted."""
    return sorted(str(u) for u in (getattr(solve_cfg, "delay_durations", None) or {}))


def realization_roll_method(solve_cfg: Any, solve: str, *, ignore_delays: bool = False) -> RollMethod | None:
    """``"within_solve"`` or ``"separate_dispatch"`` when *solve* has realization rolls, else ``None``.

    True only for a top-level solve (in ``model.solves``, neither nested in nor containing other solves) with an
    explicit ``solve_mode`` ``rolling_window``, a non-empty ``stochastic_branches`` and
    ``dispatch_realization_method`` ``within_solve`` or ``separate_dispatch``, that is no recourse solve
    (``stochastic_invest_method``) and has no author-declared realization solve
    (``_realization.configured_realization_solve``; the engine-added one is not author-declared).
    ``deterministic`` (absent or explicit; such a solve has no ``stochastic_branches`` left,
    ``SolveConfig.apply_deterministic_realization``) and single solves get ``None``.  The engine-added realization
    solve of a ``separate_dispatch`` solve (``realized_dispatch_parent``) takes its parent's answer.  Pass a solve,
    not a roll: a roll name is in no ``model.solves`` and gets ``None``.  A model with delayed units
    (:func:`delayed_units`) gets ``None`` for every solve unless *ignore_delays* (:func:`delay_gated_units`).
    """
    from flextool.engine.solve import _realization

    parent = (getattr(solve_cfg, "realized_dispatch_parent", None) or {}).get(solve)
    if parent is not None:
        if solve not in (getattr(solve_cfg, "derived_solves", None) or {}):
            return None  # an author-declared realization solve
        method = realization_roll_method(solve_cfg, parent, ignore_delays=ignore_delays)
        return method if method == "separate_dispatch" else None
    if (getattr(solve_cfg, "solve_modes", None) or {}).get(solve) != "rolling_window":
        return None
    if not (getattr(solve_cfg, "stochastic_branches", None) or {}).get(solve):
        return None
    method = solve_cfg.dispatch_realization_method_for(solve)
    if method not in ("within_solve", "separate_dispatch"):
        return None
    if solve_cfg.stochastic_invest_method_for(solve) == "recourse":
        return None
    # Nested solves get no plan (D11): only a contained solve has a storage-usage fix from its parent
    # (node_storage_usage_fix_le / node_balance_fix_quantity_eq_lower), and its per-path cap is computed nowhere
    # (_nested_end_rows stops at the jump end).  Lifting this needs that cap first.
    if solve not in _top_level_solves(solve_cfg) or solve in _contained_solves(solve_cfg):
        return None
    if (getattr(solve_cfg, "contains_solves", None) or {}).get(solve):
        return None
    # Delayed units get no plan (D12): where a forecast tail's last delayed flows wrap to (the label's first steps,
    # one cell that every path of a split roll shares) is not decided, and whether a roll splits is known only from
    # its time lists, after the realization solve is spawned.  So any delayed unit in the model keeps today's
    # realization (W-0594 names the solve).
    if not ignore_delays and delayed_units(solve_cfg):
        return None
    configured = _realization.configured_realization_solve(solve_cfg, solve)
    if configured is not None and configured not in (getattr(solve_cfg, "derived_solves", None) or {}):
        return None
    return method


def delay_gated_units(solve_cfg: Any, solve: str) -> list[str]:
    """The delayed units that alone keep *solve* from realization rolls (:func:`realization_roll_method` gates it
    only for them), else ``[]``."""
    if realization_roll_method(solve_cfg, solve) is not None:
        return []
    if realization_roll_method(solve_cfg, solve, ignore_delays=True) is None:
        return []
    return delayed_units(solve_cfg)


def announce_realization_notices(solve_cfg: Any) -> None:
    """Notes W-0635 / W-0636 for the solves that the database upgrade listed (v83 part 3) and that run with the new
    realization.  Called by ``native_run_model`` once the plans and the stochastic expansion have passed their
    checks: a solve stopped by a configuration error never gets the notice (its error explains the change).

    W-0635: a solve with ``within_solve`` realization rolls.  W-0636: a ``separate_dispatch`` solve with an
    engine-added realization solve: its solve (1) changed, and with ``separate_dispatch`` realization rolls its
    realization also looks ahead.  A rolling one without them (delayed units, recourse, a realization solve the
    author declared) keeps the earlier realization; the full notice says so for it.
    """
    from flextool.engine import _behaviour_changes

    derived = getattr(solve_cfg, "derived_solves", None) or {}
    spawned = {p for s, p in (getattr(solve_cfg, "realized_dispatch_parent", None) or {}).items() if s in derived}
    modes = getattr(solve_cfg, "solve_modes", None) or {}
    for solve in sorted(_top_level_solves(solve_cfg)):
        method = realization_roll_method(solve_cfg, solve)
        detail = ""
        if method == "within_solve":
            change = "rolling_within_solve_one_solve"
        elif solve in spawned and solve_cfg.dispatch_realization_method_for(solve) == "separate_dispatch":
            change = "separate_dispatch_realization_looks_ahead"
            if modes.get(solve) == "rolling_window" and method != "separate_dispatch":
                detail = (f"Solve '{solve}' keeps the earlier realization: its realization solve dispatches each "
                          "jump alone, without the look-ahead; only its first solve changed.")
        else:
            continue
        if solve in (_behaviour_changes.marker_json(change).get("solves") or []):
            _behaviour_changes.notice(change, "solves", [solve], detail)


def realization_rolls(
    solve_cfg: Any,
    solves: Iterable[str],
    complete_solve: Mapping[str, str],
    active_time_lists: Mapping[str, dict],
    realized_time_lists: Mapping[str, dict],
) -> dict[str, list[RollWindow]]:
    """``{stochastic complete solve: [RollWindow, ...]}`` of every gated solve, rolls in run order.

    ``within_solve``: each roll of the solve.  ``separate_dispatch``: each roll of its realization solve, with the
    window of the stochastic roll it realizes (``realized_dispatch_parent`` per roll), not its own (cut to the jump).
    """
    rdp = getattr(solve_cfg, "realized_dispatch_parent", None) or {}
    out: dict[str, list[RollWindow]] = {}
    gate: dict[str, RollMethod | None] = {}
    for s in solves:
        cs = complete_solve[s]
        if cs not in gate:
            gate[cs] = realization_roll_method(solve_cfg, cs)
        method = gate[cs]
        if method == "within_solve":
            out.setdefault(cs, []).append(RollWindow(s, cs, active_time_lists[s], realized_time_lists[s]))
        elif method == "separate_dispatch" and cs in rdp and rdp.get(s) in active_time_lists:
            uc = rdp[s]
            out.setdefault(rdp[cs], []).append(RollWindow(s, rdp[cs], active_time_lists[uc], realized_time_lists[uc]))
    return out


def _steps(time_list: dict) -> list[tuple[str, str]]:
    return [(str(p), str(e.timestep)) for p, entries in time_list.items() for e in entries]


def _forecasts_at(rows: list, cell: tuple[str, str]) -> tuple[str, ...]:
    """The forecasts made at ``(period, analysis time)`` *cell*: weight > 0, not realized, database order."""
    return tuple(dict.fromkeys(
        str(r[1]) for r in rows
        if (str(r[0]), str(r[2])) == cell and str(r[3]) != "yes" and float(r[4]) != 0.0))


def _no_forecast_error(solve: str, k: int, jump: list, cell: tuple[str, str], where: str, why: str,
                       fix: str) -> FlexToolConfigError:
    return FlexToolConfigError(
        f"Solve '{solve}', roll {k} (time steps {jump[0][1]} to {jump[-1][1]}): no forecast of stochastic_branches "
        f"is made at {cell[1]}, {where}. Since FlexTool 4.7.0 {why}. Add forecasts with analysis_time {cell[1]} "
        f"for period '{cell[0]}', or set rolling_solve_jump so that {fix}.")


def roll_tail_plan(solve_cfg: Any, rolls: list[RollWindow], timeline: list[tuple[str, str]]) -> dict[str, RollTail]:
    """The :class:`RollTail` of each roll in *rolls* (the rolls of one stochastic complete solve, in run order).

    *timeline* is the ``(period, step)`` timeline the solve's realized branch is read on
    (``_run_model._solve_timelines``).  ``{}`` when the solve is not gated (:func:`realization_roll_method`).  Raises
    :class:`FlexToolConfigError` when the solve has no ``realized: yes`` row, when its ``realized: yes`` rows name more
    than one branch or none lies on *timeline*, when a roll with a tail has no forecast made at the end of its jump,
    and (``separate_dispatch``) when no forecast is made at the start of a roll (its solve (1) branches there).  The
    authored rows are read, never changed.
    """
    from flextool.engine.solve._stochastic import realized_analysis_times

    if not rolls:
        return {}
    solve = rolls[0].solve
    if any(r.solve != solve for r in rolls):
        raise ValueError("roll_tail_plan: the rolls of one complete solve only")
    method = realization_roll_method(solve_cfg, solve)
    if method is None:
        return {}
    name = (getattr(solve_cfg, "map2d_duplicate_of", None) or {}).get(solve, (solve,))[0]
    rows = list((getattr(solve_cfg, "stochastic_branches", None) or {}).get(solve) or [])
    realized_rows = [r for r in rows if str(r[3]) == "yes"]
    if not realized_rows:
        raise FlexToolConfigError(
            f"Solve '{name}' is a rolling stochastic solve with dispatch_realization_method {method}, but its "
            "stochastic_branches have no branch with realized: yes. Since FlexTool 4.7.0 such a solve dispatches "
            "each jump with the data of the realized branch. Mark the branch that holds the realized data with "
            "realized: yes.")
    realized_names = list(dict.fromkeys(str(r[1]) for r in realized_rows))
    if len(realized_names) > 1:
        raise FlexToolConfigError(
            f"Solve '{name}': the realized: yes rows of its stochastic_branches name more than one branch "
            f"({', '.join(realized_names)}). Since FlexTool 4.7.0 a rolling stochastic solve dispatches each jump "
            "with the data of its realized branch, read as one timeline: name the same branch in every "
            "realized: yes row.")
    on_line = set(timeline)
    if not any((str(r[0]), str(r[2])) in on_line for r in realized_rows):
        raise FlexToolConfigError(
            f"Solve '{name}' is a rolling stochastic solve with dispatch_realization_method {method}, but no "
            "realized: yes row of its stochastic_branches starts at a time step of the timeline of its periods, "
            "so its jumps have no realized series to read. Give the realized: yes rows the analysis_time of a time "
            "step of that timeline.")
    realized = realized_names[0]
    plan: dict[str, RollTail] = {}
    for k, roll in enumerate(rolls):
        steps = _steps(roll.active)
        jump = _steps(roll.jump)
        if not steps or not jump:
            continue
        if steps[:len(jump)] != jump:
            raise RuntimeError(
                f"Internal error: the jump of roll '{roll.name}' is not the start of its horizon. Please report this.")
        if method == "separate_dispatch" and not _forecasts_at(rows, steps[0]):
            raise _no_forecast_error(
                name, k, jump, steps[0], "the start of the roll",
                "the first solve of each roll of a rolling separate_dispatch solve branches into the forecasts "
                "made at the start of the roll", "each roll starts at a time step with forecasts")
        times = tuple(realized_analysis_times(rows, timeline, steps[0]))
        if len(jump) == len(steps):
            plan[roll.name] = RollTail((), "deterministic", True, tuple(jump), None, realized, times)
            continue
        tail_start = steps[len(jump)]
        cohort = _forecasts_at(rows, tail_start)
        if not cohort:
            raise _no_forecast_error(
                name, k, jump, tail_start, "the end of its jump",
                "a rolling stochastic solve realizes each jump and then looks ahead with the forecasts made at "
                "the end of the jump", "each jump ends at a time step with forecasts")
        shape: RollShape = "split" if roll.jump.get(tail_start[0]) else "seam"
        plan[roll.name] = RollTail(cohort, shape, False, tuple(jump), tail_start, realized, times)
    return plan


def roll_branch_rows(solve_cfg: Any, rolls: Mapping[str, list[RollWindow]],
                     plan: Mapping[str, RollTail]) -> dict[str, list]:
    """``{sub-solve: rows}`` for every planned roll: the authored ``stochastic_branches`` rows of its complete solve
    made at ``tail_start`` that are a forecast of C_k*, in database order and not re-dated (``[]`` for a
    ``deterministic`` roll).  No ``realized: yes`` row: the jump reads ``RollTail.realized`` (one source of the
    realized branch).  Keyed on the sub-solve, so a realization solve's roll gets its rows."""
    branches = getattr(solve_cfg, "stochastic_branches", None) or {}
    out: dict[str, list] = {}
    for windows in rolls.values():
        for w in windows:
            tail = plan.get(w.name)
            if tail is None:
                continue
            if tail.deterministic:
                out[w.name] = []
                continue
            out[w.name] = [r for r in branches.get(w.solve) or []
                           if (str(r[0]), str(r[2])) == tail.tail_start
                           and str(r[1]) in tail.cohort and str(r[3]) != "yes"]
    return out


def plan_branch_rows(solve_cfg: Any, sub_solve: str) -> list | None:
    """The ``stochastic_branches`` rows the planned roll *sub_solve* reads (:func:`roll_branch_rows`), or ``None``
    without a plan (the reader then takes its complete solve's rows, as before)."""
    return (getattr(solve_cfg, "roll_branch_rows", None) or {}).get(sub_solve)


def check_tail_weights(sub_solve: str, tail: RollTail, solve_branch__time_branch: list[tuple[str, str]],
                       weights: pl.DataFrame) -> None:
    """G2b: every forecast of the planned roll *sub_solve* reads its branch through a period that has its own
    ``solve_branch_weight`` row (*weights*, the emitted frame).  A missing row would read weight 1.0
    (``params/_branch.py``), making the forecasts silently equally likely."""
    if tail.deterministic:
        return
    weighted = set(weights["branch"].to_list()) if weights.height else set()
    for f in tail.cohort:
        periods = [p for p, b in solve_branch__time_branch if b == f] or [f"{tail.tail_start[0]}_{f}"]
        for p in periods:
            if p not in weighted:
                raise RuntimeError(
                    f"Internal error: solve_branch_weight on period '{p}' of roll '{sub_solve}' is not path-aware "
                    f"(forecast '{f}' has no branch weight). Please report this.")


__all__ = [
    "RollTail",
    "RollWindow",
    "announce_realization_notices",
    "check_tail_weights",
    "delay_gated_units",
    "delayed_units",
    "plan_branch_rows",
    "realization_roll_method",
    "realization_rolls",
    "roll_branch_rows",
    "roll_tail_plan",
]
