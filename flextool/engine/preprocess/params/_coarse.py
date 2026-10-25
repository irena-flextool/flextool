"""Coarse time steps of the nodes in a ``new_stepduration`` group.

A group whose ``new_stepduration`` is longer than the solve's time step makes its balance and storage nodes
*coarse*: every flow, online status, reserve and angle stays on the solve's steps, and each coarse node is
balanced once per coarse step by summing its fine balance terms over the step (``nodeBalanceCoarse_eq``); a
coarse storage node keeps one state per coarse step (``stateConstantCoarse_eq``).

:func:`coarse_node_frames` cuts the solve's steps into coarse steps.  It is the only source of the coarse
steps; the per-block timelines of the block layout (``solve_data/block_step_duration`` etc.) do not feed the
model.  The steps of each period are taken in the solve's order and split into runs of consecutive steps; a
run ends

* where the next step does not continue the timeset block (``t_previous_within_timeset`` is not the step
  before it: a gap in the timeline, the start of a timeset block, the start of a period or of a roll),
* at the first step of a representative block (``rp_block_first``),
* after the last realized step of the period (a roll's realized end, so the state a roll hands over is the
  state at its realized end).

Inside a run the steps are chunked from the run's start: a coarse step closes once its fine step durations
add up to the group's step.  A run that is not a multiple of the group step ends with a shorter coarse step.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import polars as pl

from flextool.engine.input._solve_state import FlexToolConfigError

_LOG = logging.getLogger(__name__)
_EPS = 1e-9
# Warnings already given in this run (one per message, not one per roll); cleared at the start of every run
# (``reset_warnings``, called by ``run_orchestration``).
_WARNED: set[str] = set()


def reset_warnings() -> None:
    """Forget the warnings given so far, so a new run in the same process shows them again."""
    _WARNED.clear()


def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        _LOG.warning(msg)


@dataclass(frozen=True)
class CoarseFrames:
    """``nodeCoarse`` ``(n)``, ``node_coarse_step`` ``(n, d, b_first, t)`` (fine step t of period d lies in the
    coarse step named by its first fine step b_first) and ``node_coarse_interior`` ``(n, d, t, t_previous)``
    (consecutive fine steps of one coarse step, storage nodes only; ``None`` without such rows).  Utf8 keys."""

    nodeCoarse: pl.DataFrame
    node_coarse_step: pl.DataFrame
    node_coarse_interior: "pl.DataFrame | None"


def _utf8(df: "pl.DataFrame | None", cols: tuple[str, ...]) -> "pl.DataFrame | None":
    if df is None or df.height == 0 or not all(c in df.columns for c in cols):
        return None
    return df.select(pl.col(c).cast(pl.Utf8) for c in cols)


def coarse_groups(node_group: dict[str, str], group_step: dict[str, float], solve_step: float,
                  solve: str = "") -> dict[str, float]:
    """The groups of ``node_group`` whose step is longer than ``solve_step`` -> their step.

    A group step that is not a multiple of the solve's step raises ``FlexToolConfigError``.  A group step
    shorter than the solve's step is not coarse (warning); one equal to it is not coarse either."""
    out: dict[str, float] = {}
    for g in sorted(set(node_group.values())):
        step = group_step.get(g)
        if step is None:
            continue
        step = float(step)
        if step < solve_step - _EPS:
            _warn_once(f"group {g!r}: new_stepduration {step:g} h is shorter than the time step {solve_step:g} h "
                       f"of the solve{f' {solve!r}' if solve else ''}; its nodes are balanced at the solve's time "
                       "step.")
            continue
        if step <= solve_step + _EPS:
            continue
        ratio = step / solve_step
        if abs(ratio - round(ratio)) > 1e-6:
            raise FlexToolConfigError(
                f"group {g!r}: new_stepduration {step:g} h is not a multiple of the time step {solve_step:g} h of "
                f"the solve{f' {solve!r}' if solve else ''}.  A coarse group must combine a whole number of the "
                "solve's time steps; use a multiple of the solve's step (or change the solve's new_stepduration).")
        out[g] = step
    return out


def coarse_node_frames(*, node_group: dict[str, str], group_step: dict[str, float], solve_step: float,
                       dt: "pl.DataFrame | None", step_duration: "pl.DataFrame | None",
                       dtttdt: "pl.DataFrame | None", rp_block_first: "pl.DataFrame | None" = None,
                       realized: "pl.DataFrame | None" = None, balance_nodes: "pl.DataFrame | None" = None,
                       state_nodes: "pl.DataFrame | None" = None, block_nodes: "pl.DataFrame | None" = None,
                       solve: str = "") -> "CoarseFrames | None":
    """Coarse steps of the balance / storage nodes of every coarse group (see the module docstring).

    ``node_group`` node -> group (``default`` for none), ``group_step`` group -> new_stepduration (h),
    ``solve_step`` the solve's step (h).  ``dt`` ``(d, t)`` in solve order, ``step_duration`` ``(d, t, value)``,
    ``dtttdt`` ``(d, t, t_previous_within_timeset)``, ``rp_block_first`` ``(d, t)``, ``realized`` ``(d, t)``
    the realized steps (``None``: every step), ``balance_nodes`` / ``state_nodes`` / ``block_nodes`` ``(n)``:
    the nodes with a per-step balance, the storage nodes and the ``bind_intraperiod_blocks`` nodes.
    Returns ``None`` when no node is coarse."""
    groups = coarse_groups(node_group, group_step, solve_step, solve)
    if not groups:
        return None
    dt_u = _utf8(dt, ("d", "t"))
    bal = set(_utf8(balance_nodes, ("n",))["n"].to_list()) if _utf8(balance_nodes, ("n",)) is not None else set()
    nodes_by_group: dict[str, list[str]] = {}
    for n, g in sorted(node_group.items()):
        if g in groups and n in bal:
            nodes_by_group.setdefault(g, []).append(n)
    if dt_u is None or not nodes_by_group:
        return None
    blocks = set(_utf8(block_nodes, ("n",))["n"].to_list()) if _utf8(block_nodes, ("n",)) is not None else set()
    both = sorted(n for ns in nodes_by_group.values() for n in ns if n in blocks)
    if both:
        raise FlexToolConfigError(
            f"node(s) {both} use storage_binding_method bind_intraperiod_blocks and are in a group with a "
            "new_stepduration longer than the solve's time step.  bind_intraperiod_blocks already balances the "
            "node once per timeset block; remove the node from the coarse group or choose another binding "
            "method.")

    dur = {}
    sd = _utf8(step_duration, ("d", "t"))
    if sd is not None and step_duration is not None and "value" in step_duration.columns:
        dur = {(d, t): float(v) for d, t, v in zip(sd["d"].to_list(), sd["t"].to_list(),
                                                   step_duration["value"].cast(pl.Float64).to_list())}
    prev_in_block = {}
    lag = _utf8(dtttdt, ("d", "t", "t_previous_within_timeset"))
    if lag is not None:
        prev_in_block = {(d, t): p for d, t, p in lag.iter_rows()}
    rp_first = set()
    rpf = _utf8(rp_block_first, ("d", "t"))
    if rpf is not None:
        rp_first = set(rpf.iter_rows())
    has_rp = bool(rp_first)

    steps_by_period: dict[str, list[str]] = {}
    for d, t in dt_u.iter_rows():
        steps_by_period.setdefault(d, []).append(t)
    realized_end: set[tuple[str, str]] = set()
    rz = _utf8(realized, ("d", "t"))
    if rz is not None:
        rz_set = set(rz.iter_rows())
        for d, ts in steps_by_period.items():
            last = None
            for i, t in enumerate(ts):
                if (d, t) in rz_set:
                    last = i
            if last is not None and last < len(ts) - 1:
                realized_end.add((d, ts[last]))

    # Runs of consecutive steps per period, the same for every group.
    runs: list[tuple[str, list[str]]] = []
    for d, ts in steps_by_period.items():
        cur: list[str] = []
        for i, t in enumerate(ts):
            start = (i == 0 or (d, t) in rp_first or (d, ts[i - 1]) in realized_end
                     or (lag is not None and prev_in_block.get((d, t)) != ts[i - 1]))
            if start and cur:
                runs.append((d, cur))
                cur = []
            cur.append(t)
        if cur:
            runs.append((d, cur))

    step_rows: list[tuple[str, str, str, str]] = []
    interior_rows: list[tuple[str, str, str, str]] = []
    states = set(_utf8(state_nodes, ("n",))["n"].to_list()) if _utf8(state_nodes, ("n",)) is not None else set()
    for g in sorted(nodes_by_group):
        gstep = groups[g]
        chunks: list[tuple[str, list[str]]] = []
        for d, run in runs:
            acc, cur = 0.0, []
            for t in run:
                acc += dur.get((d, t), solve_step)
                cur.append(t)
                if acc > gstep + _EPS:
                    raise FlexToolConfigError(
                        f"group {g!r}: the time step {t!r} of period {d!r} crosses the end of a "
                        f"{gstep:g} h coarse step (the solve's steps there do not add up to {gstep:g} h).  Use a "
                        "new_stepduration that the solve's time steps divide exactly.")
                if acc >= gstep - _EPS:
                    chunks.append((d, cur))
                    acc, cur = 0.0, []
            if cur:
                if has_rp:
                    raise FlexToolConfigError(
                        "Coarse time resolution is not aligned with the representative periods.\n\n"
                        f"The group {g!r} sets a coarser time resolution (new_stepduration {gstep:g} h) on its "
                        "nodes, but that step length does not evenly divide the length of a representative "
                        f"period (period {d!r}, coarse step starting {cur[0]!r}).\n\n"
                        "Representative periods are individual days/weeks sampled from across the year; a coarse "
                        "time step cannot reach across the boundary between two of them.\n\n"
                        "Fix: choose a new_stepduration that divides the representative-period length exactly "
                        "(e.g. for 24-hour representative days use 2, 3, 4, 6, 8, 12 or 24 hours), or remove the "
                        "coarse resolution from these nodes.")
                chunks.append((d, cur))
        for n in nodes_by_group[g]:
            for d, ts in chunks:
                for i, t in enumerate(ts):
                    step_rows.append((n, d, ts[0], t))
                    if i > 0 and n in states:
                        interior_rows.append((n, d, t, ts[i - 1]))
    if not step_rows:
        return None
    node_coarse_step = pl.DataFrame(step_rows, schema={"n": pl.Utf8, "d": pl.Utf8, "b_first": pl.Utf8,
                                                       "t": pl.Utf8}, orient="row")
    interior = (pl.DataFrame(interior_rows, schema={"n": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8,
                                                    "t_previous": pl.Utf8}, orient="row")
                if interior_rows else None)
    return CoarseFrames(nodeCoarse=node_coarse_step.select("n").unique(maintain_order=True),
                        node_coarse_step=node_coarse_step, node_coarse_interior=interior)


def coarse_membership_warnings(process_group: list[tuple[str, str]], process_nodes: dict[str, set[str]],
                               node_group: dict[str, str], groups: dict[str, float]) -> list[str]:
    """Warnings for units / connections listed in a coarse group (Q-c15-A): listing has no effect on time
    resolution (flows and online status stay at the solve's step).  Warned where it would have mattered:
    the process touches a node outside that group."""
    out: list[str] = []
    for g, p in sorted(set(process_group)):
        if g not in groups:
            continue
        outside = sorted(n for n in process_nodes.get(p, set()) if node_group.get(n) != g)
        if outside:
            out.append(
                f"{p!r} is listed in group {g!r} (new_stepduration {groups[g]:g} h) and connects to node(s) "
                f"{outside} outside that group.  Listing a unit or connection in a new_stepduration group has no "
                "effect on time resolution: its flows and online status stay at the solve's time step, and each "
                "coarse node sums the flows over its coarse step.")
    return out
