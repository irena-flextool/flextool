"""The realization roll R_k of a rolling stochastic solve: which solves get one, and the plan of each roll.

A rolling stochastic solve that runs ``dispatch_realization_method`` ``within_solve`` or ``separate_dispatch``
realizes each jump ``[s_k, s_{k+1})`` on the ``realized: yes`` data and looks ahead from ``s_{k+1}`` (the end of the
jump) with the forecasts made there.  ``within_solve``: the realization roll is the roll itself; ``separate_dispatch``:
the roll ``k`` of the engine-added realization solve, after the roll's fully stochastic solve (1).

* :func:`realization_roll_method` is the gate: the one predicate that says whether a solve has realization rolls
  (also read by ``_orchestration._spawn_realized_dispatch_siblings``).
* :func:`roll_tail_plan` makes the :class:`RollTail` of each roll of a gated solve from the authored
  ``stochastic_branches`` rows (never changed) and raises the configuration errors of a missing forecast or a
  missing ``realized: yes`` branch.
* :func:`roll_branch_rows` gives each planned roll the ``stochastic_branches`` rows it reads (the forecasts made at
  ``s_{k+1}`` and the ``realized: yes`` row there), read by the stochastic expansion and the branch-weight emit
  through :func:`plan_branch_rows` (no plan: the complete solve's rows, as before).
* :func:`check_tail_weights` (G2b): every forecast of a planned roll has its own branch weight.

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
    """

    cohort: tuple[str, ...]
    shape: RollShape
    deterministic: bool
    jump_steps: tuple[tuple[str, str], ...]
    tail_start: tuple[str, str] | None


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


def realization_roll_method(solve_cfg: Any, solve: str) -> RollMethod | None:
    """``"within_solve"`` or ``"separate_dispatch"`` when *solve* has realization rolls, else ``None``.

    True only for a top-level solve (in ``model.solves``, neither nested in nor containing other solves) with an
    explicit ``solve_mode`` ``rolling_window``, a non-empty ``stochastic_branches`` and
    ``dispatch_realization_method`` ``within_solve`` or ``separate_dispatch``, that is no recourse solve
    (``stochastic_invest_method``) and has no author-declared realization solve
    (``_realization.configured_realization_solve``; the engine-added one is not author-declared).
    ``deterministic`` (absent or explicit; such a solve has no ``stochastic_branches`` left,
    ``SolveConfig.apply_deterministic_realization``) and single solves get ``None``.  The engine-added realization
    solve of a ``separate_dispatch`` solve (``realized_dispatch_parent``) takes its parent's answer.
    """
    from flextool.engine.solve import _realization

    parent = (getattr(solve_cfg, "realized_dispatch_parent", None) or {}).get(solve)
    if parent is not None:
        if solve not in (getattr(solve_cfg, "derived_solves", None) or {}):
            return None  # an author-declared realization solve
        method = realization_roll_method(solve_cfg, parent)
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
    if solve not in _top_level_solves(solve_cfg) or solve in _contained_solves(solve_cfg):
        return None
    if (getattr(solve_cfg, "contains_solves", None) or {}).get(solve):
        return None
    configured = _realization.configured_realization_solve(solve_cfg, solve)
    if configured is not None and configured not in (getattr(solve_cfg, "derived_solves", None) or {}):
        return None
    return method


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
        f"is made at {cell[1]}, {where}. Since FlexTool 4.7.0 {why}. Add forecasts with analysis time {cell[1]} "
        f"for period '{cell[0]}', or set rolling_solve_jump so that {fix}.")


def roll_tail_plan(solve_cfg: Any, rolls: list[RollWindow]) -> dict[str, RollTail]:
    """The :class:`RollTail` of each roll in *rolls* (the rolls of one stochastic complete solve, in run order).

    ``{}`` when the solve is not gated (:func:`realization_roll_method`).  Raises :class:`FlexToolConfigError` when a
    ``within_solve`` solve has no ``realized: yes`` branch, when a roll with a tail has no forecast made at the end
    of its jump, and (``separate_dispatch``) when no forecast is made at the start of a roll (its solve (1) branches
    there).  The authored rows are read, never changed.
    """
    if not rolls:
        return {}
    solve = rolls[0].solve
    if any(r.solve != solve for r in rolls):
        raise ValueError("roll_tail_plan: the rolls of one complete solve only")
    method = realization_roll_method(solve_cfg, solve)
    if method is None:
        return {}
    rows = list((getattr(solve_cfg, "stochastic_branches", None) or {}).get(solve) or [])
    if method == "within_solve" and not any(str(r[3]) == "yes" for r in rows):
        raise FlexToolConfigError(
            f"Solve '{solve}' is a rolling stochastic solve with dispatch_realization_method within_solve, but its "
            "stochastic_branches have no branch with realized: yes. Since FlexTool 4.7.0 such a solve dispatches "
            "each jump with the data of the realized branch. Mark the branch that holds the realized data with "
            "realized: yes.")
    plan: dict[str, RollTail] = {}
    for k, roll in enumerate(rolls):
        steps = _steps(roll.active)
        jump = _steps(roll.jump)
        if not steps or not jump:
            continue
        if method == "separate_dispatch" and not _forecasts_at(rows, steps[0]):
            raise _no_forecast_error(
                solve, k, jump, steps[0], "the start of the roll",
                "the first solve of each roll of a rolling separate_dispatch solve branches into the forecasts "
                "made at the start of the roll", "each roll starts at a time step with forecasts")
        end = steps.index(jump[-1]) + 1 if jump[-1] in steps else len(steps)
        if end >= len(steps):
            plan[roll.name] = RollTail((), "deterministic", True, tuple(jump), None)
            continue
        tail_start = steps[end]
        cohort = _forecasts_at(rows, tail_start)
        if not cohort:
            raise _no_forecast_error(
                solve, k, jump, tail_start, "the end of its jump",
                "a rolling stochastic solve realizes each jump and then looks ahead with the forecasts made at "
                "the end of the jump", "each jump ends at a time step with forecasts")
        shape: RollShape = "split" if tail_start[0] in roll.jump else "seam"
        plan[roll.name] = RollTail(cohort, shape, False, tuple(jump), tail_start)
    return plan


def roll_branch_rows(solve_cfg: Any, rolls: Mapping[str, list[RollWindow]],
                     plan: Mapping[str, RollTail]) -> dict[str, list]:
    """``{sub-solve: rows}`` for every planned roll: the authored ``stochastic_branches`` rows of its complete solve
    made at ``tail_start`` that are a forecast of C_k* or ``realized: yes``, in database order and not re-dated
    (``[]`` for a ``deterministic`` roll).  Keyed on the sub-solve, so a realization solve's roll gets its rows."""
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
                           and (str(r[1]) in tail.cohort or str(r[3]) == "yes")]
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
    "check_tail_weights",
    "plan_branch_rows",
    "realization_roll_method",
    "realization_rolls",
    "roll_branch_rows",
    "roll_tail_plan",
]
