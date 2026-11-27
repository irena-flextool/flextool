"""The storage of nested solves: the upper solve's planned levels, the end rows that restate an upper-level fix,
the model-start and match rows, the fix_usage totals and the start a lower level borrows from its upper solve."""
from __future__ import annotations

import polars as pl
from flextool.engine.solve import _carry_routing

from flextool.engine.solve._run_model._storage_frames import (
    _MODULE_LOGGER,
    _RollingStorage,
    _SBM_KEY,
    _SSE_KEY,
    _frame_nodes,
    _node_methods,
    _warn_storages,
)


def _upper_steps(parent: str, active_time_lists: dict, time_index) -> list:
    """``(period rank, base timeline, start, stop, period, step)`` of the upper sub-solve ``parent``'s steps in its
    order (start / stop: base-timeline indices, :meth:`_carry_routing._TimeIndex.span`); forecast branch periods and
    steps that cannot be placed are left out."""
    out: list = []
    for period, steps in (active_time_lists.get(parent) or {}).items():
        for entry in steps:
            end = time_index.end_of(str(period), str(entry.timestep), entry.duration)
            if end is None or end[3] is not None:
                continue
            span = time_index.span(str(period), str(entry.timestep), entry.duration)
            if span is not None:
                out.append((*span, str(period), str(entry.timestep)))
    return out


def _node_levels(parent_handoff, node: str) -> dict:
    """``{(period, step): MWh}``: the upper solve's planned level of ``node`` at the end of each step
    (``storage_level``)."""
    level = getattr(parent_handoff, "storage_level", None)
    if level is None or level.height == 0:
        return {}
    hit = level.filter(pl.col("node").cast(pl.Utf8) == node)
    return {(str(p), str(t)): float(v) for p, t, v in zip(hit["period"].to_list(), hit["step"].to_list(),
                                                          hit["value"].to_list())}


def _upper_start_level(parent_handoff, node: str) -> "float | None":
    """The level (MWh) the upper solve's first step starts ``node`` from (``cycle_start_state``)."""
    frame = getattr(parent_handoff, "cycle_start_state", None)
    if frame is None or frame.height == 0:
        return None
    hit = frame.filter(pl.col("node").cast(pl.Utf8) == node)
    return float(hit["value"][0]) if hit.height > 0 else None


def _upper_level_interp(*, node: str, at: tuple, steps: list, parent_handoff, time_index,
                        levels: "dict | None" = None) -> "tuple[float, bool] | None":
    """``(level, aligned)``: the level (MWh) the upper solve planned for ``node`` at the instant ``at`` (``(period
    rank, base-timeline index)``) and whether an upper step ends (or the upper solve starts) exactly there.

    The upper solve holds its flows constant within a step, so its level is linear inside the step: inside a step
    the level is interpolated between the step's start (the end of the step before it, or the upper solve's start
    level for its first step) and its end, by the hours elapsed (owner question Q6).  Where the step's start is not
    known (the first step of a later period) the containing step's end is used.  None when the upper solve has no
    level there."""
    levels = _node_levels(parent_handoff, node) if levels is None else levels
    for k, (rank, root, start, stop, period, step) in enumerate(steps):
        if rank != at[0]:
            continue
        if k == 0 and start == at[1]:
            value = _upper_start_level(parent_handoff, node)
            return None if value is None else (value, True)
        if stop == at[1]:
            value = levels.get((period, step))
            return None if value is None else (value, True)
        if start < at[1] < stop:
            end = levels.get((period, step))
            if end is None:
                return None
            if k == 0:
                begin = _upper_start_level(parent_handoff, node)
            else:
                prev = steps[k - 1]
                begin = levels.get((prev[4], prev[5])) if prev[0] == rank and prev[3] == start else None
            if begin is None:
                return end, False
            total = time_index.hours(root, start, stop)
            if not total > 0:
                return end, False
            return begin + (end - begin) * time_index.hours(root, start, at[1]) / total, False
    return None


def _upper_level_at(*, node: str, parent: str, parent_handoff, active_time_lists: dict, time_index,
                    first: "tuple[str, str]", warned: set, solve: str) -> "float | None":
    """The level (MWh) the upper sub-solve ``parent`` planned for ``node`` at the start of ``first`` (the lower
    sub-solve's first step): its start level when ``first`` starts with the upper solve (``cycle_start_state``),
    else its level at the end of the upper step that ends where ``first`` starts (``storage_level``); when ``first``
    starts inside an upper step, the level interpolated inside that step (:func:`_upper_level_interp`).  None when
    the upper solve has no such level."""
    at = time_index.position(*first)
    if at is None:
        return None
    hit = _upper_level_interp(node=node, at=at, steps=_upper_steps(parent, active_time_lists, time_index),
                              parent_handoff=parent_handoff, time_index=time_index)
    return None if hit is None else hit[0]


def _nested_end_rows(*, solve: str, parent: "str | None", parent_handoff, active_time_lists: dict, time_index,
                     instance: tuple, store: "_RollingStorage", period_branch: list, rolling: bool = False) -> list:
    """``storage_roll_end`` rows that restate an upper-level storage fix at the horizon end of ``solve``.

    * ``fix_quantity`` (MWh): where the horizon ends inside an upper step, the upper level interpolated at that
      instant (:func:`_upper_level_interp`) replaces the containing step's end level
      (node_balance_fix_quantity_eq_lower).
    * ``usage_cap`` (MWh): the fix_usage cap of the horizon's last period (node_storage_usage_fix_le) counts the
      upper usage from the start of that period in the solve instance up to the horizon end, each upper window
      pro rata to the hours of it that lie in that span, less what the earlier sub-solves of the instance realised
      in the period (``realised_storage_usage``): the cap is cumulative over the rolls, so the realised rolls
      together use at most what the upper solve planned.  Written only where it differs from the sum of the whole
      upper windows that overlap the horizon (the cap of a sub-solve aligned with the upper steps that starts the
      period).
    * ``usage_cap_period`` (MWh, with its period): in a roll of a rolling solve (``rolling``), the same cumulative
      cap for each period of the roll before the horizon's last one, up to the end of the period, so the rolls
      that realise a period's last steps (their horizon reaching into the next period) are capped in it too and
      the realised rolls of every period use at most what the upper solve planned for it.
    """
    if parent is None or parent_handoff is None:
        return []
    lower = active_time_lists.get(solve) or {}
    last = next(((p, steps) for p, steps in reversed(list(lower.items())) if steps), None)
    if last is None:
        return []
    period, steps = last
    end = time_index.end_of(str(period), str(steps[-1].timestep), steps[-1].duration)
    first = time_index.position(str(period), str(steps[0].timestep))
    if end is None or first is None:
        return []
    h = (end[0], end[2])
    upper = _upper_steps(parent, active_time_lists, time_index)
    if not upper:
        return []
    rows: list = []
    for node in sorted(_frame_nodes(getattr(parent_handoff, "fix_storage_quantity", None))):
        hit = _upper_level_interp(node=node, at=h, steps=upper, parent_handoff=parent_handoff,
                                  time_index=time_index)
        if hit is not None and not hit[1]:
            rows.append((node, "fix_quantity", float(hit[0])))
    usage = getattr(parent_handoff, "fix_storage_usage", None)
    if usage is not None and usage.height > 0:
        anchor = {str(b): str(p) for p, b in period_branch if b != p}
        spans = {(u[4], u[5]): u for u in upper}
        value_col = "p_fix_storage_usage" if "p_fix_storage_usage" in usage.columns else "value"
        upper_usage = usage.select(
            pl.col("node").cast(pl.Utf8), pl.col("period").cast(pl.Utf8), pl.col("step").cast(pl.Utf8),
            pl.col(value_col).cast(pl.Float64)).rows()

        def cumulative(base: str, first_at, end_at) -> dict:
            """``{node: (whole, cumulative)}``: the upper usage of the windows of the period (rank ``end_at[0]``)
            overlapping [first_at, end_at) (whole: the default cap of the model), and the cumulative cap: the upper
            usage from the period's start in the instance up to ``end_at``, pro rata, less what the instance's
            earlier sub-solves realised in the period."""
            start = store.usage_start.setdefault((instance, base), first_at)
            per_node: dict = {}
            for node, p_up, t_up, value in upper_usage:
                span = spans.get((p_up, t_up))
                if span is None or span[0] != end_at[0] or value is None:
                    continue
                rank, root, a, b = span[:4]
                whole = float(value) if (a < end_at[1] and b > first_at[1]) else 0.0
                lo, hi = max(a, start[1] if start[0] == rank else a), min(b, end_at[1])
                total = time_index.hours(root, a, b)
                part = float(value) * time_index.hours(root, lo, hi) / total if hi > lo and total > 0 else 0.0
                old, new = per_node.get(node, (0.0, 0.0))
                per_node[node] = (old + whole, new + part)
            return {node: (old, new - store.usage_realised.get((instance, base, node), 0.0))
                    for node, (old, new) in per_node.items()}

        # The periods of a roll before its horizon's last one (the realised part and the look-ahead of a roll that
        # crosses into the next period): their own cumulative cap up to the period's end in the roll, so the rolls
        # that realise a period's tail are capped too (review M3).  Rolling solves only: a solve that is not
        # rolling keeps the cap of its last period.
        if rolling:
            for p_mid, mid_steps in lower.items():
                if p_mid == period:
                    break
                if not mid_steps:
                    continue
                mid_first = time_index.position(str(p_mid), str(mid_steps[0].timestep))
                mid_end = time_index.end_of(str(p_mid), str(mid_steps[-1].timestep), mid_steps[-1].duration)
                if mid_first is None or mid_end is None or mid_end[3] is not None:
                    continue
                for node, (_old, new) in sorted(cumulative(anchor.get(str(p_mid), str(p_mid)), mid_first,
                                                           (mid_end[0], mid_end[2])).items()):
                    rows.append((node, "usage_cap_period", new, str(p_mid)))
        for node, (old, new) in sorted(cumulative(anchor.get(str(period), str(period)), first, h).items()):
            if abs(new - old) > 1e-9 * max(1.0, abs(old)):
                rows.append((node, "usage_cap", new))
    if rows:
        from flextool.engine import _behaviour_changes
        _behaviour_changes.notice("nested_storage_fix_cumulative", "nodes", sorted({r[0] for r in rows}),
                                  solve=solve)
    return rows


def _record_model_start(store: "_RollingStorage", handoff, routed_state, level: int = 0) -> None:
    """Record the level (MWh) each storage node started the model from at nesting ``level``, in a sub-solve that
    starts with the model (``cycle_start_state``; the nodes it did not take from a carry).  Per level: each nesting
    level runs its own chain, so a lower level's matching end compares with its own start, not the upper one's."""
    frame = getattr(handoff, "cycle_start_state", None)
    if frame is None or frame.height == 0:
        return
    carried = _frame_nodes(routed_state)
    for node, value in zip(frame["node"].cast(pl.Utf8).to_list(), frame["value"].to_list()):
        if node not in carried and value is not None:
            store.model_start.setdefault((level, node), float(value))


def _sequence_match_rows(*, provider, routed_state, store: "_RollingStorage", at_model_end: bool,
                         upper_fixed: set, solve: str = "", level: int = 0) -> list:
    """``match`` rows (node, MWh) of a sub-solve that is not rolling, reaches the end of the model and continues a
    forward chain (bind_forward_only, bind_forward_only_blended_weights) with a free start (fix_nothing) from a
    carry: it ends the chain at least at the level the model started the node from (the matching end over a
    sequence of solves; model.core storage_state_roll_match / rp_inter_period_start_match), at its nesting
    ``level`` (:meth:`_RollingStorage.model_start_of`)."""
    if not at_model_end or routed_state is None:
        return []
    sbm = _node_methods(provider, _SBM_KEY, "storage_binding_method")
    sse = _node_methods(provider, _SSE_KEY, "storage_start_end_method")
    shm = _node_methods(provider, "input/node__storage_solve_horizon_method", "storage_solve_horizon_method")
    rows = []
    for node in sorted(_frame_nodes(routed_state)):
        start = store.model_start_of(level, node)
        if (sbm.get(node, "bind_forward_only") in ("bind_forward_only", "bind_forward_only_blended_weights")
                and sse.get(node, "fix_nothing") == "fix_nothing" and node not in upper_fixed
                and shm.get(node) != "use_reference_value"
                and start is not None):
            rows.append((node, "match", start))
    if rows:
        from flextool.engine import _behaviour_changes
        _behaviour_changes.notice("storage_end_matches_start", "nodes", [r[0] for r in rows], solve=solve)
    return rows


def _record_usage(store: "_RollingStorage", instance: tuple, handoff, period_branch: list) -> None:
    """Add a solved sub-solve's realised fix_usage energy (``realised_storage_usage``) to its instance's total."""
    frame = getattr(handoff, "realised_storage_usage", None)
    if frame is None or frame.height == 0:
        return
    anchor = {str(b): str(p) for p, b in period_branch if b != p}
    for node, period, value in frame.select(pl.col("node").cast(pl.Utf8), pl.col("period").cast(pl.Utf8),
                                            "value").iter_rows():
        if value is not None:
            key = (instance, anchor.get(period, period), node)
            store.usage_realised[key] = store.usage_realised.get(key, 0.0) + float(value)


def _borrow_upper_start(*, state, solve: str, complete_solve_name: str, parent: "str | None", parent_handoff,
                        provider, routed_state: "pl.DataFrame | None", active_time_lists: dict, time_index,
                        model_start, is_rp: bool, store: _RollingStorage) -> "pl.DataFrame | None":
    """``routed_state`` with the start a nested lower level borrows from its upper solve (borrowed end ⇒ borrowed
    start).

    A storage node whose horizon ends an upper-level solve fixes (``storage_nested_fix_method`` fix_quantity,
    fix_usage or fix_price: a node of the parent's ``fix_storage_*`` handoff) and that has no state realised before
    this sub-solve at its level or deeper starts from the level the upper solve planned for the same instant
    (:func:`_upper_level_at`): the lower level's accounting is then the upper's.  Only a node that continues from a
    start: one that chains forward in this sub-solve (bind_forward_only), or a bind_within_solve, bind_within_period or
    bind_within_timeblock (used as bind_within_period there) node of a rolling roll (no representative periods)
    under fix_quantity / fix_usage, which
    then continues from it (:func:`_anchor_cyclic_in_rolling`, :func:`_scope_rows`: its cycle would otherwise wrap
    to the upper target at the roll's end); a node that cycles on itself is closed already.  An explicit fix_start
    (fix_start, fix_start_end) at the model start is used instead (the same parameter as the upper solve's start;
    the two are compared once the sub-solve is solved).  Returns the routed state (MWh, sorted by node) or None.
    """
    if parent is None or parent_handoff is None:
        return routed_state
    quantity_usage = (_frame_nodes(getattr(parent_handoff, "fix_storage_quantity", None))
                      | _frame_nodes(getattr(parent_handoff, "fix_storage_usage", None)))
    nested = quantity_usage | _frame_nodes(getattr(parent_handoff, "fix_storage_price", None))
    if not nested:
        return routed_state
    carried = _frame_nodes(routed_state)
    sbm = _node_methods(provider, _SBM_KEY, "storage_binding_method")
    sse = _node_methods(provider, _SSE_KEY, "storage_start_end_method")
    rolling = state.solve.solve_modes.get(complete_solve_name) == "rolling_window"
    first = _carry_routing.first_step(active_time_lists.get(solve) or {})
    if first is None:
        return routed_state
    at_start = model_start is not None and time_index.position(*first) == model_start
    borrowed: dict[str, float] = {}
    compare: list[str] = []
    for node in sorted(nested - carried):
        method = sbm.get(node, "bind_forward_only")
        if not (method == "bind_forward_only"
                or (rolling and not is_rp and node in quantity_usage
                    and method in ("bind_within_solve", "bind_within_period"))):
            continue
        if sse.get(node) in ("fix_start", "fix_start_end") and at_start:
            compare.append(node)
            continue
        value = _upper_level_at(node=node, parent=parent, parent_handoff=parent_handoff,
                                active_time_lists=active_time_lists, time_index=time_index, first=first,
                                warned=store.warned, solve=solve)
        if value is not None:
            borrowed[node] = value
    if compare:
        upper = getattr(parent_handoff, "cycle_start_state", None)
        if upper is not None and upper.height > 0:
            store.borrowed_check[solve] = (compare, dict(zip(upper["node"].cast(pl.Utf8).to_list(),
                                                             upper["value"].to_list())))
    if not borrowed:
        return routed_state
    from flextool.engine import _behaviour_changes
    names = sorted(borrowed)
    _behaviour_changes.notice("nested_storage_start_borrowed", "nodes", names, solve=complete_solve_name)
    key = ("borrowed", complete_solve_name)
    if key not in store.warned:
        store.warned.add(key)
        _MODULE_LOGGER.info(
            "Solve '%s' is nested in solve '%s', which fixes %s: they start from the level the upper solve planned "
            "for the start of this solve.", complete_solve_name, parent, _behaviour_changes.storages_named(names))
        _MODULE_LOGGER.debug("Solve '%s': borrowed start for %s.", complete_solve_name,
                             ", ".join(f"{n} {borrowed[n]:.6g} MWh" for n in names))
    rows = {} if routed_state is None else dict(zip(routed_state["node"].cast(pl.Utf8).to_list(),
                                                    routed_state["value"].to_list()))
    rows.update(borrowed)
    nodes = sorted(rows)
    return pl.DataFrame({"node": nodes, "value": [float(rows[n]) for n in nodes]},
                        schema={"node": pl.Utf8, "value": pl.Float64})


def _check_borrowed_start(store: _RollingStorage, solve: str, complete_solve_name: str, handoff) -> None:
    """Warn when a nested first sub-solve that starts from fix_start starts from another level than its upper
    solve (:func:`_borrow_upper_start`): the capacity or the share differs between the two."""
    nodes, upper = store.borrowed_check.pop(solve, ((), {}))
    frame = getattr(handoff, "cycle_start_state", None)
    if not nodes or frame is None or frame.height == 0:
        return
    own = dict(zip(frame["node"].cast(pl.Utf8).to_list(), frame["value"].to_list()))
    off = [(n, own[n], upper[n]) for n in nodes
           if own.get(n) is not None and upper.get(n) is not None
           and abs(float(own[n]) - float(upper[n])) > 1e-6 * max(1.0, abs(float(upper[n])))]
    if off:
        _warn_storages(
            "W-0583", complete_solve_name, [n for n, _a, _b in off],
            "start from fix_start vs upper plan: " + ", ".join(f"{n} {a:.6g} MWh (upper {b:.6g})" for n, a, b in off))
