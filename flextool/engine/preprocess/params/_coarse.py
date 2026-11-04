"""Coarse time steps of the nodes in a ``new_stepduration`` group, and of the processes between them.

A group whose ``new_stepduration`` is longer than the solve's time step makes its balance and storage nodes
*coarse*: each coarse node is balanced once per coarse step by summing its fine balance terms over the step
(``nodeBalanceCoarse_eq``); a coarse storage node keeps one state per coarse step (``stateConstantCoarse_eq``).
Online status, reserves and angles stay on the solve's steps.

A unit or connection whose every balance node is coarse is a *coarse process* (:func:`coarse_process_frames`):
its flows have one value per coarse step of its nodes (a rate that stays the same over the step).  A process
with a fine balance node, or with no balance node, keeps the solve's step, and so does a process with a
feature whose rows are per fine step and couple the flow to other per-step terms (online variables, reserves,
a delay, DC power flow, an exact connection, minimum load, a user constraint flow term, a flow group instant
limit, a capacity-margin / inertia / non-synchronous group): those rows would not hold for a flow that cannot
change within the coarse step.

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


def _info_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        _LOG.info(msg)


@dataclass(frozen=True)
class CoarseProcessFrames:
    """``processCoarse`` ``(p)``: the coarse processes; ``process_coarse_step`` ``(p, d, b_first, t, w)``: fine step
    t of period d lies in the coarse step b_first (its first fine step) of process p, ``w = Δt / Δ_b`` its share
    of the coarse step's duration Δ_b = Σ_{t ∈ b} Δt.  ``step_hours`` p -> the process's coarse step (h);
    ``kept_fine`` p -> why a process whose every balance node is coarse keeps the solve's step.  Utf8 keys."""

    processCoarse: pl.DataFrame
    process_coarse_step: pl.DataFrame
    step_hours: dict[str, float]
    kept_fine: dict[str, str]


def coarse_process_frames(*, process_nodes: dict[str, set[str]], balance_nodes: set[str],
                          node_step: dict[str, float], node_coarse_step: "pl.DataFrame | None",
                          step_duration: "pl.DataFrame | None", excluded: dict[str, str],
                          solve_step: float) -> "CoarseProcessFrames | None":
    """The coarse processes and their coarse steps (module docstring).

    ``process_nodes`` p -> every node of p (each source and sink, both directions of a connection);
    ``balance_nodes`` the nodes with a per-step balance (``nodeBalance``) — the other nodes (commodity / price
    nodes, ``balance_within_period`` nodes) are neutral; ``node_step`` coarse node -> its group's step (h);
    ``node_coarse_step`` ``(n, d, b_first, t)`` (:func:`coarse_node_frames`); ``step_duration`` ``(d, t,
    value)``; ``excluded`` p -> the feature that keeps p at the solve's step; ``solve_step`` (h), the duration
    of a step missing from ``step_duration``.

    A process is coarse when it has a balance node, every balance node is coarse and the shortest coarse step s
    of its nodes divides every other one; it then takes the coarse steps of a node with step s (the coarse
    steps of all groups are chunked from the same run starts, so they nest).  Returns ``None`` when no process
    is coarse and none is kept fine."""
    ncs = _utf8(node_coarse_step, ("n", "d", "b_first", "t"))
    if ncs is None or not node_step:
        return None
    coarse: dict[str, tuple[float, str]] = {}
    kept_fine: dict[str, str] = {}
    for p in sorted(process_nodes):
        bal = sorted(n for n in process_nodes[p] if n in balance_nodes)
        if not bal or any(n not in node_step for n in bal):
            continue
        steps = {n: float(node_step[n]) for n in bal}
        s = min(steps.values())
        ratios = [v / s for v in steps.values()]
        if any(abs(r - round(r)) > 1e-6 for r in ratios):
            kept_fine[p] = ("the coarse steps of its nodes ("
                            + ", ".join(f"{v:g} h" for v in sorted(set(steps.values())))
                            + ") do not divide each other")
            continue
        if p in excluded:
            kept_fine[p] = excluded[p]
            continue
        coarse[p] = (s, next(n for n in bal if abs(steps[n] - s) <= _EPS))
    if not coarse and not kept_fine:
        return None
    schema = {"p": pl.Utf8, "d": pl.Utf8, "b_first": pl.Utf8, "t": pl.Utf8, "w": pl.Float64}
    if not coarse:
        return CoarseProcessFrames(processCoarse=pl.DataFrame(schema={"p": pl.Utf8}),
                                   process_coarse_step=pl.DataFrame(schema=schema), step_hours={},
                                   kept_fine=kept_fine)
    pick = pl.DataFrame({"p": list(coarse), "n": [coarse[p][1] for p in coarse]},
                        schema={"p": pl.Utf8, "n": pl.Utf8})
    sd = _utf8(step_duration, ("d", "t"))
    if sd is not None and step_duration is not None and "value" in step_duration.columns:
        dur = sd.with_columns(__dt=step_duration["value"].cast(pl.Float64))
    else:
        dur = pl.DataFrame(schema={"d": pl.Utf8, "t": pl.Utf8, "__dt": pl.Float64})
    steps = (pick.join(ncs.with_row_index("__i"), on="n", how="inner")
             .join(dur, on=["d", "t"], how="left")
             .with_columns(pl.col("__dt").fill_null(float(solve_step)))
             .with_columns((pl.col("__dt") / pl.col("__dt").sum().over("p", "d", "b_first")).alias("w"))
             .sort("p", "__i")
             .select("p", "d", "b_first", "t", "w"))
    return CoarseProcessFrames(processCoarse=pl.DataFrame({"p": list(coarse)}, schema={"p": pl.Utf8}),
                               process_coarse_step=steps,
                               step_hours={p: coarse[p][0] for p in coarse}, kept_fine=kept_fine)


def coarse_process_messages(frames: "CoarseProcessFrames | None") -> None:
    """Info lines (once per run): the coarse processes per step length, and each process whose nodes are all
    coarse but that keeps the solve's step, with the reason."""
    if frames is None:
        return
    by_step: dict[float, list[str]] = {}
    for p, s in sorted(frames.step_hours.items()):
        by_step.setdefault(s, []).append(p)
    for s, procs in sorted(by_step.items()):
        _info_once(f"The flows of {', '.join(repr(p) for p in procs)} use the {s:g} h time step of their nodes "
                   "(new_stepduration): one flow per coarse step, with availability, profiles and efficiency "
                   "averaged over the step.")
    for p, why in sorted(frames.kept_fine.items()):
        _info_once(f"{p!r} connects only nodes of coarse new_stepduration groups but keeps the solve's time step: "
                   f"{why}.")


def varying_within_coarse_steps(frame: "pl.DataFrame | None", steps: pl.DataFrame,
                                keys: tuple[str, ...] = ()) -> pl.DataFrame:
    """The distinct ``(p, keys…)`` (Utf8) of ``frame`` (``keys…, [d], [t], value``; it carries ``p`` or is joined to
    ``steps`` ``(p, d, b_first, t)`` on ``d`` / ``t`` only) that take more than one value within some coarse step
    of p.  A frame without ``t`` is constant within every step."""
    out_keys = list(dict.fromkeys(["p", *keys]))
    empty = pl.DataFrame(schema={k: pl.Utf8 for k in out_keys})
    if frame is None or frame.height == 0 or "t" not in frame.columns or "value" not in frame.columns:
        return empty
    on = [c for c in ("p", "d", "t") if c in frame.columns]
    group = ["p", "d", "b_first", *[k for k in keys if k != "p"]]
    sel = list(dict.fromkeys([*on, *keys]))
    f = frame.select(*(pl.col(c).cast(pl.Utf8) for c in sel), pl.col("value").cast(pl.Float64))
    st = steps.select(*(pl.col(c).cast(pl.Utf8) for c in ("p", "d", "b_first", "t")))
    spread = (st.join(f, on=on, how="inner")
              .group_by(group)
              .agg((pl.col("value").max() - pl.col("value").min()).alias("__spread"),
                   pl.col("value").abs().max().alias("__size")))
    hit = spread.filter(pl.col("__spread") > 1e-9 * pl.max_horizontal(pl.lit(1.0), pl.col("__size")))
    return hit.select(out_keys).unique().sort(out_keys)


def varies_within_coarse_steps(frame: "pl.DataFrame | None", steps: pl.DataFrame,
                               keys: tuple[str, ...] = ()) -> set[str]:
    """The processes of :func:`varying_within_coarse_steps`."""
    return set(varying_within_coarse_steps(frame, steps, keys)["p"].to_list())
