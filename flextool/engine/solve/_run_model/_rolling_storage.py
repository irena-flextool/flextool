"""The storage of rolling_window solves: whole periods of a roll, the bind_within_solve carry, anchor and close of
a roll (the ``storage_roll_end`` rows), the per-period scopes and their start levels, bind_within_timeblock in a
roll, and the rolling checks."""
from __future__ import annotations

import polars as pl
from flextool.engine.input._solve_state import FlexToolConfigError

from flextool.engine.solve._run_model._storage_frames import (
    _MODULE_LOGGER,
    _RollingStorage,
    _SBM_KEY,
    _SSE_KEY,
    _method_col,
    _node_methods,
    _restore_methods,
    _storages_named,
    _warn_storages,
)


def _roll_period_whole(active_time_list: dict, complete_active_time_list: dict,
                       period_branch: list) -> "pl.DataFrame":
    """``(period, whole)`` of a rolling_window sub-solve: True where the roll holds every step that the complete
    solve has in the period (a forecast branch period is compared with the steps of the period it branches from).

    The ``bind_intraperiod_blocks`` blocks of a whole period loop within the roll, as in a solve that is not rolling;
    the blocks of a period the roll holds in part chain forward and carry the storage level from roll to roll
    (owner decision Q-c15e-A).  In a split realization roll the label keeps the jump steps and each branch period
    the rest of the period: a path (the label's steps, then one branch period's) is compared, so the label and its
    branch periods are whole together.
    """
    anchor = {b: p for p, b in period_branch if b != p}
    rows: list[tuple[str, bool]] = []
    for period, steps in active_time_list.items():
        base = anchor.get(period, period)
        full = complete_active_time_list.get(base) or []
        names = _path_steps(active_time_list, period_branch, period)
        rows.append((str(period), bool(steps) and names == [s.timestep for s in full]))
    return pl.DataFrame(rows, schema={"period": pl.Utf8, "whole": pl.Boolean}, orient="row")


def _path_steps(active_time_list: dict, period_branch: list, period: str) -> list:
    """The step names of ``period`` in the roll, on its path through a split realization roll: a branch period after
    its label's (jump) steps, a label before the steps of its first branch period (every branch period holds the same
    steps).  Elsewhere the period's own steps."""
    own = [s.timestep for s in active_time_list.get(period) or []]
    label = next((p for p, b in period_branch if b == period and p != period), None)
    if label is not None and active_time_list.get(label):
        return [s.timestep for s in active_time_list[label]] + own
    branch = next((b for p, b in period_branch if p == period and b != period and active_time_list.get(b)), None)
    return own + [s.timestep for s in active_time_list[branch]] if branch is not None and own else own


def _warn_rolling_blocks_chained(state, complete_solve_name: str, solve: str, active_time_list: dict,
                                 complete_active_time_list: dict, provider, warned: set) -> None:
    """Warn once per rolling_window solve whose rolls start inside a period while it has ``bind_intraperiod_blocks``
    storage nodes (its rolling_solve_jump is not a whole number of periods).

    Such a period is held in part by the rolls around its inside start, so its blocks chain forward and carry the
    storage level from roll to roll instead of looping as in a solve that is not rolling (the period still closes at
    its end, :func:`_scope_rows`).  ``solve_data/
    nodeStateBlock`` (the block nodes of this sub-solve) is emitted by ``preprocessing_solve_time.run``, so this runs
    after it.  Routed through the module logger: ``state.logger`` is silenced to ERROR during the cascade.
    """
    if complete_solve_name in warned or state.solve.solve_modes.get(complete_solve_name) != "rolling_window":
        return
    if not active_time_list or not provider.has("solve_data/nodeStateBlock"):
        return
    nsb = provider.get("solve_data/nodeStateBlock")
    if nsb is None or nsb.height == 0:
        return
    first_period = next(iter(active_time_list))
    steps = active_time_list[first_period]
    full = complete_active_time_list.get(first_period) or []
    if not steps or not full or steps[0].timestep == full[0].timestep:
        return
    warned.add(complete_solve_name)
    col = "node" if "node" in nsb.columns else nsb.columns[0]
    names = sorted(str(n) for n in nsb[col].unique().to_list())
    _MODULE_LOGGER.warning(
        "Solve '%s' is a rolling_window solve whose rolling_solve_jump is not a whole number of periods (roll '%s' "
        "starts at step '%s' inside period '%s'). The bind_intraperiod_blocks %s loop their timeset "
        "blocks only in the periods a roll holds whole; in a period a roll holds in part the blocks chain forward, "
        "the storage level is carried from roll to roll and the roll(s) holding the period's end close it at the "
        "level it started from.",
        complete_solve_name, solve, steps[0].timestep, first_period, _storages_named(names))


# period: only for a kind that holds per period (usage_cap_period); None otherwise.
_ROLL_END_SCHEMA = {"node": pl.Utf8, "kind": pl.Utf8, "value": pl.Float64, "period": pl.Utf8}


# bind_within_timeblock reaches a rolling solve as bind_within_period (:func:`_timeblock_binds_within_period_in_roll`).
_SCOPE_METHODS = ("bind_within_period", "bind_intraperiod_blocks")


def _first_scope(active_time_list: dict, complete_active_time_list: dict,
                 period_branch: list) -> "tuple[tuple[str, str], bool, bool] | None":
    """``(scope, continued, holds_end)`` of the cyclic scope (the period) that holds a rolling roll's first step.

    ``scope`` is ``(period, the period's first step)`` in the complete solve (a forecast branch period by the period
    it branches from).
    ``continued``: the roll starts after the scope's first step (an earlier roll began the scope); ``holds_end``:
    the roll holds the scope's last step.  None when the roll's first step is not in the complete solve.
    """
    first = next(((p, steps) for p, steps in (active_time_list or {}).items() if steps), None)
    if first is None:
        return None
    period, steps = first
    anchor = {b: p for p, b in period_branch if b != p}
    base = str(anchor.get(period, period))
    full = complete_active_time_list.get(base) or complete_active_time_list.get(period) or []
    names = [str(e.timestep) for e in full]
    s0 = str(steps[0].timestep)
    if s0 not in names:
        return None
    start, end = 0, len(full) - 1
    held = {str(t) for t in _path_steps(active_time_list, period_branch, period)}
    return (base, names[start]), names[start] != s0, names[end] in held


def _put_roll_end(provider, rows: list) -> None:
    from flextool.engine.preprocess import _provider_keys
    rows = [tuple(r) + (None,) * (len(_ROLL_END_SCHEMA) - len(r)) for r in rows]
    provider.put(_provider_keys.SOLVE_DATA_STORAGE_ROLL_END,
                 pl.DataFrame(rows, schema=_ROLL_END_SCHEMA, orient="row"))


def _anchor_cyclic_in_rolling(*, state, solve: str, complete_solve_name: str, instance: tuple, level_key,
                              provider, routed_state: "pl.DataFrame | None", closing: bool, instance_at_model_end: bool,
                              nested_fixed: set, store: _RollingStorage, is_rp: bool,
                              active_time_list: "dict | None" = None, complete_active_time_list: "dict | None" = None,
                              period_branch: "list | None" = None, extra_rows: "list | None" = None,
                              nested_any: "set | None" = None, level: int = 0) -> None:
    """Carry, anchor and close the bind_within_solve storage nodes of a rolling_window roll (no representative
    periods).

    A roll that continues a node from a carried state (``routed_state``: the previous roll, the previous instance,
    or the start a nested level borrows from its upper solve, :func:`_borrow_upper_start`) chains it forward from
    that state: its method is rewritten to bind_forward_only for the roll, in the per-solve
    ``input/node__storage_binding_method`` (read by the per-solve preprocessing), and the horizon end gets
    ``solve_data/storage_roll_end``:

    * a roll whose horizon does not reach the end of the instance: ``anchor`` at the carried level (the model
      ends the horizon at least there, ``model.core._storage_state._eq_storage_roll_end``);
    * a closing roll (``closing``: its horizon holds the last step of the instance): ``close`` at the level the
      instance's first roll started from (:class:`_RollingStorage` ``start``).

    An upper-level fix at the horizon end (fix_quantity, fix_usage; fix_price for the anchor only) and fix_end at
    the last step of the model take precedence in the model.  A first roll with no carry keeps its cycle (its wrap,
    pinned by fix_start), and its start level is read from its handoff once solved, except under an upper-level
    fix_quantity / fix_usage (``nested_fixed``) with fix_start: that roll starts from fix_start and chains forward
    (its wrap target would be the upper target), with ``anchor_share`` / ``close_share``.  A fix_end node without
    fix_start whose instance reaches the model end gets ``wrap_end``: its first roll's wrap is pinned at
    storage_state_end.  ``solve_data/storage_roll_end`` is written for every sub-solve (empty when unused): the
    level Provider is shared by the iters of a level.

    The same rule holds per period (bind_within_period, bind_intraperiod_blocks; bind_within_timeblock, which a
    rolling solve uses as bind_within_period, :func:`_timeblock_binds_within_period_in_roll`): a roll whose first
    step lies inside a period that an earlier roll began (a realised roll boundary inside the period,
    :func:`_first_scope`) continues that period from the carry (kind ``scope_carry``, for bind_within_period; the
    blocks of a period held in part chain forward already), and where the period's part in the roll ends gets
    ``scope_close`` (the roll holds the period's last step, or reaches the end of the instance) at the level the
    period started from, else ``scope_anchor`` at the carried level.  The periods that begin in the roll keep their
    cycle (a period held whole loops as before; the blocks of a bind_intraperiod_blocks period held in part chain
    forward with the horizon method at the roll end), and their start levels are read from the handoff
    (``scope_start_state``) once solved.  ``extra_rows`` (the
    nested-fix rows of :func:`_nested_end_rows`) are written with the rows of any solve.

    A bind_forward_only node with a free start (fix_nothing, no upper-level fix) whose first roll does not reach the
    end of the model gets ``match_first`` there (its horizon end at least at the level it starts from, the matching
    end of a solve that is not rolling) and its start level is recorded; the closing rolls of every instance that
    reaches the end of the model and continues the node from a carry end it at least at the level the node started
    from: the instance's own free start, else the level the model started the node from at this nesting ``level``
    (:meth:`_RollingStorage.model_start_of`; a later rolling solve of a sequence, a later instance of a nested
    rolling solve) (``match``).
    """
    extra = list(extra_rows or [])
    nested_any = set(nested_any or ()) | set(nested_fixed)
    rolling = state.solve.solve_modes.get(complete_solve_name) == "rolling_window"
    if not rolling or is_rp or not provider.has(_SBM_KEY):
        _put_roll_end(provider, extra)
        return
    sbm = _restore_methods(store, level_key, provider)
    col = _method_col(sbm, "storage_binding_method") if sbm.height > 0 else None
    if col is None:
        _put_roll_end(provider, extra)
        return
    targets = sorted({str(n) for n in sbm.filter(pl.col(col) == "bind_within_solve")["node"].to_list()})
    sse = _node_methods(provider, _SSE_KEY, "storage_start_end_method")
    scoped = {str(n): str(m) for n, m in zip(sbm["node"].to_list(), sbm[col].to_list())
              if m in _SCOPE_METHODS and not (m == "bind_intraperiod_blocks"
                                               and sse.get(str(n)) == "fix_start_end")}
    # The matching end of a forward chain whose start the first roll chooses (fix_nothing; logic doc E-match).
    shm = _node_methods(provider, "input/node__storage_solve_horizon_method", "storage_solve_horizon_method")
    free_forward = sorted(str(n) for n, m in zip(sbm["node"].to_list(), sbm[col].to_list())
                          if m == "bind_forward_only" and sse.get(str(n), "fix_nothing") == "fix_nothing"
                          and str(n) not in nested_any and shm.get(str(n)) != "use_reference_value")
    if not targets and not scoped and not free_forward:
        _put_roll_end(provider, extra)
        return
    carry = ({} if routed_state is None else
             dict(zip(routed_state["node"].cast(pl.Utf8).to_list(), routed_state["value"].to_list())))
    first_roll = instance not in store.seen
    store.seen.add(instance)
    starts = store.start.setdefault(instance, {})
    rows: list = []
    matched: list[str] = []
    match_pending: list[str] = []
    for node in free_forward:
        if node not in carry and first_roll and not (closing and instance_at_model_end):
            # The first roll's horizon ends at least at the level it chose to start from, and that level is
            # recorded for the rolls reaching the end of the model.
            rows.append((node, "match_first", None))
            match_pending.append(node)
            matched.append(node)
        elif node in carry and closing and instance_at_model_end:
            start = starts.get(node)
            if start is None:
                start = store.model_start_of(level, node)
            if start is not None:
                rows.append((node, "match", float(start)))
                matched.append(node)
    if matched:
        from flextool.engine import _behaviour_changes
        _behaviour_changes.notice("storage_end_matches_start", "nodes", matched, solve=complete_solve_name)
    if scoped and active_time_list:
        rows += _scope_rows(complete_solve_name=complete_solve_name, solve=solve, instance=instance, scoped=scoped,
                            carry=carry, closing=closing, first_roll=first_roll, store=store,
                            active_time_list=active_time_list,
                            complete_active_time_list=complete_active_time_list or {},
                            period_branch=period_branch or [], sse=sse, provider=provider,
                            nested_fixed=nested_fixed)
    forward: list[str] = []
    pending: list[str] = []
    unknown_start: list[str] = []
    no_start: list[str] = []
    for node in targets:
        fix_start = sse.get(node) in ("fix_start", "fix_start_end")
        if node in carry:
            forward.append(node)
            if first_roll:
                starts[node] = float(carry[node])
            if closing:
                start = starts.get(node)
                if start is None:
                    if (instance, node) not in store.warned:
                        store.warned.add((instance, node))
                        unknown_start.append(node)
                    start = float(carry[node])
                rows.append((node, "close", start))
            else:
                rows.append((node, "anchor", float(carry[node])))
        elif node in nested_fixed and fix_start:
            forward.append(node)
            rows.append((node, "close_share" if closing else "anchor_share", None))
            pending.append(node)
        else:
            if first_roll:
                pending.append(node)
            if sse.get(node) == "fix_end" and instance_at_model_end:
                rows.append((node, "wrap_end", None))
            if node in nested_fixed and (instance, node) not in store.warned:
                store.warned.add((instance, node))
                no_start.append(node)
    if unknown_start:
        _warn_storages("W-0580", complete_solve_name, unknown_start,
                       f"closing roll {solve} ends at the carried level: {', '.join(unknown_start)}")
    if no_start:
        _warn_storages("W-0581", complete_solve_name, no_start,
                       f"first roll {solve} cycles on itself: {', '.join(no_start)}")
    if forward:
        is_fwd = pl.col("node").cast(pl.Utf8).is_in(forward)
        provider.put(_SBM_KEY, sbm.with_columns(
            pl.when(is_fwd).then(pl.lit("bind_forward_only")).otherwise(pl.col(col)).alias(col)))
        from flextool.engine import _behaviour_changes
        _behaviour_changes.notice("rolling_cyclic_storage_anchored", "nodes", forward, solve=complete_solve_name)
        key = ("logged", complete_solve_name)
        if key not in store.warned:
            store.warned.add(key)
            _MODULE_LOGGER.info(
                "Solve '%s' is a rolling_window solve: %s with bind_within_solve continue each roll from the level "
                "realised by the previous roll; a roll ends its horizon at least at the level it started from, and "
                "the roll(s) reaching the end of the solve end at the level the first roll started from.",
                complete_solve_name, _behaviour_changes.storages_named(forward))
            _MODULE_LOGGER.debug("Solve '%s': rolling carry for %s.", complete_solve_name, ", ".join(forward))
    pending += match_pending
    if pending:
        store.pending[solve] = (instance, pending)
    _put_roll_end(provider, rows + extra)


def _scope_rows(*, complete_solve_name: str, solve: str, instance: tuple, scoped: dict, carry: dict, closing: bool,
                first_roll: bool, store: _RollingStorage, active_time_list: dict, complete_active_time_list: dict,
                period_branch: list, sse: dict, provider, nested_fixed: set) -> list:
    """The ``storage_roll_end`` rows of the per-period scopes of a rolling roll
    (:func:`_anchor_cyclic_in_rolling`): for each node of ``scoped`` ({node: method}) whose first period the roll
    continues from the carry, ``scope_carry`` (bind_within_period) and ``scope_close`` (at the
    level the scope started from) or ``scope_anchor`` (at the carried level; a bind_intraperiod_blocks node whose
    storage_solve_horizon_method is use_reference_value keeps that guidance instead).  The roll's scopes are left
    pending for :func:`_record_scope_starts`."""
    rows: list = []
    names: list[str] = []
    missing: list[str] = []
    shm = _node_methods(provider, "input/node__storage_solve_horizon_method", "storage_solve_horizon_method")
    for node, method in sorted(scoped.items()):
        scope = _first_scope(active_time_list, complete_active_time_list, period_branch)
        if scope is None:
            continue
        (period, step), continued, holds_end = scope
        guided = method == "bind_intraperiod_blocks" and shm.get(node) == "use_reference_value"
        if (first_roll and not continued and node in nested_fixed and not holds_end
                and method != "bind_intraperiod_blocks"):
            # The first roll of a nested level begins a period that its horizon cuts: its first step
            # would wrap to the horizon end, which the upper solve fixes, so the cycle would start at the upper level
            # of that moment.  It starts instead from the level the upper solve starts from (borrowed, in the carry)
            # or from fix_start, and continues the scope forward from there.
            key = (instance, node, period, step)
            if node in carry:
                store.scope_start[key] = float(carry[node])
                rows += [(node, "scope_carry", None), (node, "scope_anchor", float(carry[node]))]
                names.append(node)
            elif sse.get(node) in ("fix_start", "fix_start_end"):
                store.scope_from_start.setdefault(solve, []).append((key, node))
                rows.append((node, "scope_carry", None))
                names.append(node)
            continue
        if not continued and method == "bind_intraperiod_blocks" and not (holds_end or closing):
            # The roll begins the period at its first step and holds it in part, so its blocks chain forward from
            # the roll's start: the horizon end is anchored at that start, as a roll that loops ends where it
            # began (a free start, and a node guided by use_reference_value, keep the horizon method).
            if guided:
                continue
            if node in carry:
                rows.append((node, "scope_anchor", float(carry[node])))
                names.append(node)
            elif sse.get(node) == "fix_start":
                rows.append((node, "anchor_share", None))
                names.append(node)
            continue
        if not continued or node not in carry:
            continue
        key = (instance, node, period, step)
        start = store.scope_start.get(key)
        if start is None:
            if not first_roll and (key, "missing") not in store.warned:
                store.warned.add((key, "missing"))
                missing.append(node)
                _MODULE_LOGGER.debug("Solve '%s': start level of storage node '%s' in period '%s' (from step '%s') "
                                     "not known in roll %s.", complete_solve_name, node, period, step, solve)
            start = float(carry[node])
            store.scope_start[key] = start
        if method != "bind_intraperiod_blocks":
            rows.append((node, "scope_carry", None))
        if holds_end or closing:
            rows.append((node, "scope_close", float(start)))
        elif not guided:
            rows.append((node, "scope_anchor", float(carry[node])))
        names.append(node)
    store.scope_pending[solve] = (instance, sorted(scoped))
    if missing:
        _warn_storages("W-0582", complete_solve_name, missing,
                       f"roll {solve} continues the period from the carried level: {', '.join(missing)}")
    if names:
        from flextool.engine import _behaviour_changes
        _behaviour_changes.notice("rolling_cyclic_storage_anchored", "nodes", names, solve=complete_solve_name)
        from flextool import run_warnings
        run_warnings.note("W-0533", names, solve=complete_solve_name)
    return rows


def _record_scope_starts(store: _RollingStorage, solve: str, handoff, period_branch: list) -> None:
    """Record the start levels of the scopes that began in ``solve`` (its handoff's ``scope_start_state``)."""
    from_start = store.scope_from_start.pop(solve, [])
    if from_start:
        cycle = getattr(handoff, "cycle_start_state", None)
        levels = ({} if cycle is None or cycle.height == 0 else
                  dict(zip(cycle["node"].cast(pl.Utf8).to_list(), cycle["value"].to_list())))
        for key, node in from_start:
            if levels.get(node) is not None:
                store.scope_start[key] = float(levels[node])
    instance, nodes = store.scope_pending.pop(solve, (None, ()))
    if instance is None:
        return
    frame = getattr(handoff, "scope_start_state", None)
    if frame is None or frame.height == 0:
        return
    anchor = {str(b): str(p) for p, b in period_branch if b != p}
    wanted = set(nodes)
    # The latest roll that began a scope wins: a roll that holds a scope's first step only in its look-ahead is
    # followed by a roll that begins it again, and the roll that continues the scope follows the one that realised
    # its first step.
    for node, period, step, value in frame.select(pl.col("node").cast(pl.Utf8), pl.col("period").cast(pl.Utf8),
                                                  pl.col("step").cast(pl.Utf8), "value").iter_rows():
        if node in wanted and value is not None:
            store.scope_start[(instance, node, anchor.get(period, period), step)] = float(value)


def _record_roll_starts(store: _RollingStorage, solve: str, handoff) -> None:
    """Fill the start levels :func:`_anchor_cyclic_in_rolling` left pending for ``solve`` from its handoff
    (``cycle_start_state``)."""
    instance, nodes = store.pending.pop(solve, (None, ()))
    if instance is None:
        return
    frame = getattr(handoff, "cycle_start_state", None)
    if frame is None or frame.height == 0:
        return
    levels = dict(zip(frame["node"].cast(pl.Utf8).to_list(), frame["value"].to_list()))
    starts = store.start.setdefault(instance, {})
    for node in nodes:
        if node in levels and levels[node] is not None:
            starts.setdefault(node, float(levels[node]))


def _timeblock_binds_within_period_in_roll(*, state, complete_solve_name: str, active_timeset_names: list,
                                           provider) -> None:
    """A rolling_window solve (without representative periods) uses bind_within_timeblock as bind_within_period
    (owner decision, Phase 5).

    A rolling solve runs over a continuous timeline (:func:`_reject_rolling_with_gaps`), so each period of it is one
    timeblock and the two methods mean the same cycle; a roll, however, cuts the timeblock, and the per-period rule of
    bind_within_period (:func:`_anchor_cyclic_in_rolling`, :func:`_scope_rows`) carries the level from roll to roll,
    anchors each horizon end and closes the period at the level it started from (capped at capacity).  Rewrites the
    per-solve ``input/node__storage_binding_method`` in place, after :func:`_fix_start_end_chains_forward` (a
    fix_start_end node already chains forward); the rolls of the level share the frame and
    :func:`_restore_methods` keeps the converted one.  One printed line per solve, naming how many nodes and the
    first.  A solve that is not rolling (an investment or storage solve above it) keeps bind_within_timeblock.
    """
    if state.solve.solve_modes.get(complete_solve_name) != "rolling_window":
        return
    if any(ts in state.timeline.rp_weights for ts in active_timeset_names):
        return  # rolling over representative periods is an error (_reject_rolling_rp)
    if not provider.has(_SBM_KEY):
        return
    sbm = provider.get(_SBM_KEY)
    col = _method_col(sbm, "storage_binding_method") if sbm is not None and sbm.height > 0 else None
    if col is None:
        return
    is_tb = pl.col(col).cast(pl.Utf8) == "bind_within_timeblock"
    names = sorted(str(n) for n in sbm.filter(is_tb)["node"].unique().to_list())
    if not names:
        return
    provider.put(_SBM_KEY, sbm.with_columns(
        pl.when(is_tb).then(pl.lit("bind_within_period")).otherwise(pl.col(col)).alias(col)))
    from flextool import run_warnings
    run_warnings.note("W-0532", names, solve=complete_solve_name)


def _reject_rolling_with_gaps(*, state, complete_solve_name: str, complete_active_time_list: dict) -> None:
    """Raise for a rolling_window solve over a timeline with gaps: a period with several timeblocks (a sampled
    timeline, e.g. sampled weeks) (owner decision, Phase 5).

    A roll cuts and joins the timeblocks, so the storage level would carry across the gaps, without the losses of the
    hours between them, and the cyclic storage methods would lose their meaning.  Timeblocks are maximal runs of
    consecutive timeline steps (:func:`flextool.engine.input._timeline.make_step_jump`); a solve with
    new_stepduration counts the aggregated steps, which are consecutive on the aggregated timeline.
    """
    if state.solve.solve_modes.get(complete_solve_name) != "rolling_window":
        return
    for period, steps in (complete_active_time_list or {}).items():
        blocks = 1 + sum(1 for a, b in zip(steps, steps[1:]) if b.index - a.index != 1) if steps else 0
        if blocks > 1:
            raise FlexToolConfigError(
                f"Rolling solve '{complete_solve_name}' runs over a timeline with gaps (period '{period}' has "
                f"{blocks} timeblocks). Rolling solves need a continuous timeline; use a single solve for sampled "
                "timeblocks.")
