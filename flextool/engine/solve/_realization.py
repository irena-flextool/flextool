"""Realization of a stochastic fan: which solve realizes the committed window of a stochastic solve.

A stochastic solve optimises over forecasts only; its operational decisions inside a fan have no realized answer of
their own.  Realized operations come from a *realization solve*: a deterministic solve that runs right after the
stochastic solve over the same committed window, with the investments of the stochastic solve fixed and its
non-anticipativity-tied unit commitment and reserves inherited.  The realization solve is either

* engine-spawned (``<solve>__realized_dispatch``): for ``dispatch_realization_method = separate_dispatch``, or for a
  ``realized: yes`` designation in ``stochastic_branches`` whose fan reaches the solve's committed window; or
* author-declared: a deterministic solve that immediately follows the stochastic solve in ``model.solves``, has the
  same solve mode and roll structure, and realizes every committed period of the fan.

Both kinds are registered in ``SolveConfig.realized_dispatch_parent`` (realization solve -> stochastic solve).  This
module holds the config-time predicates that decide the registration and the per-sub-solve
:class:`RealizationPlan`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from flextool.engine.preprocess.params._branch import phantom_periods

_MODULE_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RealizationPlan:
    """How the fan of one sub-solve is realized.

    ``labels`` are the period labels of the sub-solve: the original names of the periods that open a fan
    (``period__branch`` self-row with at least one forecast member).  ``realization_solve`` is the sub-solve that
    realizes this sub-solve's committed window (the key of ``realized_dispatch_parent`` whose value is this
    sub-solve), or ``None`` when no realization solve exists.
    """

    labels: frozenset[str]
    realization_solve: str | None


def build_realization_plan(
    solve: str,
    period__branch: Iterable[tuple[str, str]],
    realized_dispatch_parent: Mapping[str, str],
) -> RealizationPlan:
    """The :class:`RealizationPlan` of *solve* from its ``period__branch`` rows and the realization registry.

    Per-roll registry entries (``<sibling>_roll_<k> -> <parent>_roll_<k>``) are matched like single-solve ones; when
    several entries name *solve* the first in registry order is taken (one realization solve per sub-solve).
    """
    labels = frozenset(phantom_periods({(str(d), str(b)) for d, b in period__branch}))
    realization_solve = next((s for s, p in realized_dispatch_parent.items() if p == solve), None)
    return RealizationPlan(labels=labels, realization_solve=realization_solve)


def _branch_rows(solve_cfg: Any, solve: str) -> list:
    rows = (getattr(solve_cfg, "stochastic_branches", None) or {}).get(solve)
    return list(rows or [])


def _period_order(solve_cfg: Any, solve: str) -> list[str]:
    timesets = (getattr(solve_cfg, "timesets_used_by_solves", None) or {}).get(solve) or []
    return [str(p) for p, _ts in timesets]


def _realized_period_names(solve_cfg: Any, solve: str) -> list[str]:
    realized = (getattr(solve_cfg, "realized_periods", None) or {}).get(solve) or []
    return [str(p_in) for _p_from, p_in in realized]


def _committed_from(solve_cfg: Any, solve: str, reveal_periods: set[str]) -> list[str]:
    """Realized periods of *solve* at or after the earliest of *reveal_periods* in the solve's period order.

    A fan opened at a period continues through every later period of the solve, so a realized period at or after
    the reveal period is committed inside that fan.  A reveal period missing from the period order counts only
    itself.
    """
    realized = _realized_period_names(solve_cfg, solve)
    order = _period_order(solve_cfg, solve)
    positions = [order.index(p) for p in reveal_periods if p in order]
    if not positions:
        return [p for p in realized if p in reveal_periods]
    first = min(positions)
    return [p for p in realized if (p in order and order.index(p) >= first) or p in reveal_periods]


def fan_committed_periods(solve_cfg: Any, solve: str) -> list[str]:
    """The realized periods of *solve* that lie inside one of its fans (in realized-period order); [] when
    *solve* declares no ``stochastic_branches``."""
    rows = _branch_rows(solve_cfg, solve)
    if not rows:
        return []
    return _committed_from(solve_cfg, solve, {str(r[0]) for r in rows})


def designation_reaches_committed_window(solve_cfg: Any, solve: str) -> bool:
    """True when a ``realized: yes`` row of *solve*'s ``stochastic_branches`` opens a fan that reaches a committed
    window of the solve.

    A single solve commits its ``realized_periods``; a rolling solve commits each roll's jump, and the jumps of the
    rolls tile the realized periods, so in both modes the fan reaches a committed window when a realized period lies
    at or after the designated row's reveal period.
    """
    for row in _branch_rows(solve_cfg, solve):
        if str(row[3]).strip().lower() != "yes":
            continue
        if _committed_from(solve_cfg, solve, {str(row[0])}):
            return True
    return False


def configured_realization_solve(solve_cfg: Any, parent: str) -> str | None:
    """The author-declared realization solve of *parent*, or ``None``.

    It is the solve right after *parent* in ``model.solves`` when that solve is deterministic (no
    ``stochastic_branches``), contains no solves, has the same ``solve_mode`` and roll structure
    (``rolling_times``) as *parent*, and realizes every committed period of *parent*'s fans.  A *parent* without a
    fan reaching a committed window has no realization solve.
    """
    committed = fan_committed_periods(solve_cfg, parent)
    if not committed:
        return None
    model_solve = getattr(solve_cfg, "model_solve", None) or {}
    follower = None
    for solves in model_solve.values():
        if parent in solves:
            idx = list(solves).index(parent)
            if idx + 1 < len(solves):
                follower = solves[idx + 1]
            break
    if follower is None or _branch_rows(solve_cfg, follower):
        return None
    contains = getattr(solve_cfg, "contains_solves", None) or {}
    if contains.get(follower):
        return None
    modes = getattr(solve_cfg, "solve_modes", None) or {}
    if modes.get(follower, "single_solve") != modes.get(parent, "single_solve"):
        return None
    rolling = getattr(solve_cfg, "rolling_times", None) or {}
    if list(rolling.get(follower) or []) != list(rolling.get(parent) or []):
        return None
    if not set(committed) <= set(_realized_period_names(solve_cfg, follower)):
        return None
    return follower


def warn_recourse_designation(solve: str) -> None:
    """Warn that a ``realized: yes`` designation of a recourse-investment solve gets no realization solve.

    Under recourse every forecast branch invests on its own after the reveal, so there is no realized post-reveal
    investment to fix in a realization solve.  The designation still selects the branch a deterministic solve
    reads stochastic data from.
    """
    _MODULE_LOGGER.warning(
        "Solve '%s' has a realized: yes designation in stochastic_branches, but it uses "
        "stochastic_invest_method = recourse: no realization solve is added (there is no realized post-reveal "
        "investment to fix).  The designation only selects the branch deterministic solves read stochastic data "
        "from.", solve)
