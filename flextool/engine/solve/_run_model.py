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
  :class:`flextool.engine.solve._recursive_solve.RecursiveSolveBuilder`
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
import logging
import os
import re
import time
from collections import defaultdict

import polars as pl

# ---------------------------------------------------------------------------
# Imports.  Per Phase 3 scope, we depend on legacy preprocessing /
# solve_writer modules directly — Phase 2 covers the override hook that
# intercepts the already-ported helpers.  Phases 4+ will retire the
# remaining writers.
# ---------------------------------------------------------------------------

from flextool import run_record
from flextool.engine.preprocess.emit._blocks import emit_block_data_for_solve
# Step 2.5 — legacy preprocessing package deleted (item 15).  The per-
# solve orchestrator now lives natively at
# :mod:`flextool.engine.preprocess.emit._solve_time`.
from flextool.engine.preprocess.emit import _solve_time as preprocessing_solve_time
from flextool.engine.input._solve_state import (
    FlexToolConfigError,
    FlexToolSolveError,
)
# Step 2.5 — solve_writers calls now resolve to the native polars
# implementations.  The legacy disk-writing module remains in the tree
# but is no longer called from the cascade.
from flextool.engine.preprocess.emit import _solve_writers as solve_writers
from flextool.engine.preprocess.emit._pdt_lookup import READ_FIRST_FLAT
from flextool.engine.preprocess import _provider_keys
from flextool.engine.preprocess import _provider_translators

from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
from flextool.engine.preprocess.params._run_calendar import (
    RUN_CALENDAR_KEY,
    RUN_EXISTING_COHORT_KEY,
    build_run_calendar,
    run_calendar_from_frame,
)
from flextool.engine.solve._co2_total_scope import (
    check_co2_total_sees_all_periods,
    co2_total_groups,
    solve_period_views,
    weightless_periods,
)
from flextool.engine.solve._realized_overlap import (
    check_dispatch_after_investments,
    effective_realized_invest_periods,
    find_realized_invest_overlaps,
    resolve_realized_dispatch_overlap,
    warn_realized_dispatch_overlaps,
    warn_realized_invest_overlaps,
)
from flextool.engine.input._solve_state import compute_level_key

# Native solve-tree expansion + stochastic branching + timeline helpers.
from flextool.engine.solve._recursive_solve import (
    ParentSolveInfo,
    RecursiveSolveBuilder,
)
from flextool.engine.solve._stochastic import StochasticSolver, realized_analysis_times
from flextool.engine.solve import _carry_routing
from flextool.engine.solve import _realization
from flextool.engine.solve import _realization_tail
from flextool.engine.input._timeline import (
    _new_step_of_base_steps,
    get_active_time,
    make_step_jump,
    make_period_block,
    separate_period_and_timeseries_data,
)

# Advisories that must survive the cascade's silencing of ``state.logger`` (raised to ERROR during the run).
_MODULE_LOGGER = logging.getLogger(__name__)


# Map from RP-flavoured storage_binding_method to its non-RP equivalent.
# Used by :func:`_downgrade_rp_methods_for_non_rp_solve` to strip the
# ``_blended_weights`` suffix when a solve's active timeset carries no
# ``representative_period_weights`` entry.  The same storage entity is
# allowed to drive an RP investment solve AND a chronological dispatch
# solve back-to-back; per-solve downgrade keeps both paths alive.
_RP_METHOD_DOWNGRADE: dict[str, str] = {
    "bind_within_solve_blended_weights":  "bind_within_solve",
    "bind_within_period_blended_weights": "bind_within_period",
    "bind_forward_only_blended_weights":  "bind_forward_only",
}


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


_BLENDED_METHODS = tuple(_RP_METHOD_DOWNGRADE)


def _blended_nodes(provider) -> list[str]:
    """Nodes of the per-solve ``input/node__storage_binding_method`` with a ``*_blended_weights`` method."""
    key = "input/node__storage_binding_method"
    if not provider.has(key):
        return []
    sbm = provider.get(key)
    if sbm.height == 0:
        return []
    col = "storage_binding_method" if "storage_binding_method" in sbm.columns else "method"
    return sorted(str(n) for n in sbm.filter(pl.col(col).is_in(_BLENDED_METHODS))["node"].unique().to_list())


def _check_rp_solve(*, complete_solve_name: str, rp_periods: list[str], period_order: list[str], provider) -> None:
    """Raise for a representative-period solve whose rp data would not be defined.

    A solve that mixes periods with and without representative periods while it has ``*_blended_weights`` nodes:
    the blocks of the other periods would be outside the base-period chain.  The forecast branch periods of a
    stochastic solve are rp periods with the timeset of the period they branch from.  A rolling_window solve on
    representative periods never reaches this check: :func:`_reject_rolling_rp` raises before.
    """
    blended = _blended_nodes(provider)
    non_rp = [p for p in period_order if p not in rp_periods]
    if blended and non_rp:
        raise FlexToolConfigError(
            f"Solve '{complete_solve_name}' mixes periods with representative periods ({', '.join(rp_periods)}) "
            f"and without ({', '.join(non_rp)}) while node(s) {', '.join(blended)} use a *_blended_weights "
            "storage_binding_method; the blocks of the periods without representative periods would be outside "
            "the base-period storage chain. Give every period of the solve a timeset with "
            "representative_period_weights, or use separate solves.")


def _reject_rolling_rp(*, state, complete_solve_name: str, active_timeset_names: list) -> None:
    """Raise for a rolling_window solve on representative periods: rolling over representative periods is not
    supported (owner decision; before, its *_blended_weights nodes bound forward only, with a warning).

    The representative periods stand for the whole period through their weights, and the base-period chain of the
    blended storage methods spans the whole period, so a roll cannot realise a part of them: the storage would carry
    from roll to roll without the weights and without a closure.
    """
    if state.solve.solve_modes.get(complete_solve_name) != "rolling_window":
        return
    rp = sorted({ts for ts in active_timeset_names if ts in state.timeline.rp_weights})
    if not rp:
        return
    raise FlexToolConfigError(
        f"Solve '{complete_solve_name}' is a rolling_window solve on representative periods (timeset(s) "
        f"{', '.join(rp)} have representative_period_weights): rolling over representative periods is not "
        "supported. Use solve_mode single_solve for the solve on representative periods (e.g. an investment solve) "
        "and roll a separate solve over a timeset without representative_period_weights.")


# Plain storage_binding_methods that link the representative periods of a solve to each other without their
# weights, and the *_blended_weights variant that weights the transfer.  Used by :func:`_warn_plain_cyclic_in_rp_solve`.
_RP_UNWEIGHTED_CYCLES: dict[str, str] = {
    "bind_within_solve":       "bind_within_solve_blended_weights",
    "bind_within_period":      "bind_within_period_blended_weights",
    "bind_intraperiod_blocks": "bind_within_period_blended_weights",
}


def _warn_plain_cyclic_in_rp_solve(*, complete_solve_name: str, active_timeset_names: list, rp_weights: dict,
                                   provider) -> None:
    """Warn about a solve on representative periods with a bind_within_solve, bind_within_period or
    bind_intraperiod_blocks storage node (owner decision 2026-11-19; before a configuration error).

    In such a solve the representative periods follow each other on the solve's timeline (each one a timeset block),
    so these methods chain v_state from the last step of one representative period to the first step of the next one
    and close the cycle over all of them, unweighted, while the flows of each step count with the weight of its
    representative period.  That is reasonable when the representative periods have equal weights, or when the
    timing of the storage's inflows does not matter (e.g. pumped hydro, a large heat storage); otherwise energy
    stored in a period of low weight can serve a period of high weight.  The *_blended_weights variants weight the
    transfer on the base-period chain.  The weights are the user's to use, so this is a warning, not an error.
    bind_within_timeblock cycles within each timeset block (one representative period), so it is not affected.
    Checked on the method set in the database, before :func:`_fix_start_end_chains_forward`; a fix_start_end node
    is not warned about (its binding method is ignored: that function maps it to the blended forward chain, or in a
    solve that mixes periods with and without representative periods to bind_within_timeblock, so nothing moves
    between the representative periods unweighted).  One printed line per solve (how many nodes and the first); the
    nodes and their methods go to the debug log.
    """
    if not any(ts in rp_weights for ts in active_timeset_names):
        return
    key = "input/node__storage_binding_method"
    if not provider.has(key):
        return
    sbm = provider.get(key)
    if sbm is None or sbm.height == 0:
        return
    col = "storage_binding_method" if "storage_binding_method" in sbm.columns else "method"
    if col not in sbm.columns:
        return  # malformed frame; let the downstream loaders surface it
    sse = _node_methods(provider, _SSE_KEY, "storage_start_end_method")
    hit = (sbm.filter(pl.col(col).cast(pl.Utf8).is_in(list(_RP_UNWEIGHTED_CYCLES)))
           .select(pl.col("node").cast(pl.Utf8), pl.col(col).cast(pl.Utf8)).unique().sort("node")
           .filter(~pl.col("node").is_in([n for n, m in sse.items() if m == "fix_start_end"])))
    if hit.height == 0:
        return
    from flextool import run_warnings
    rp = sorted({ts for ts in active_timeset_names if ts in rp_weights})
    details: dict = {node: f"{m} on representative timeset(s) {', '.join(rp)}; {_RP_UNWEIGHTED_CYCLES[m]} weights "
                           "the transfer" for node, m in hit.iter_rows()}
    run_warnings.warn_grouped("W-0539", hit["node"].to_list(), solve=complete_solve_name, details=details)


def _reject_plain_forward_in_rp_solve(*, complete_solve_name: str, active_timeset_names: list, rp_weights: dict,
                                      provider) -> None:
    """Raise for a solve on representative periods with a storage node on plain bind_forward_only (the default
    storage_binding_method) (owner decision, Phase 5).

    The forward chain runs through the representative periods one after the other, unweighted (as the plain cycles
    of :func:`_warn_plain_cyclic_in_rp_solve` do, which are only a warning), while the flows of each step count with the weight of
    its representative period, so energy stored in a period of low weight can be used in one of high weight.  There
    is no automatic switch to bind_forward_only_blended_weights: the user chooses the blended method.  Checked on
    the method set in the database (a node without one is bind_forward_only), before the engine's own mappings
    (:func:`_fix_start_end_chains_forward` maps a cyclic fix_start_end node in such a solve to the blended chain, or
    to bind_within_timeblock in a solve that also has periods without representative periods).
    The message names how many nodes and the first.
    """
    if not any(ts in rp_weights for ts in active_timeset_names):
        return
    from pathlib import Path
    from flextool.engine.preprocess.emit._mid_sets import derive_node_effective_type
    try:
        types = derive_node_effective_type(Path("input"), provider=provider)
    except Exception:  # noqa: BLE001 — no node data: nothing to check, the loaders report it
        return
    storage = sorted(str(n) for n in types.filter(pl.col("type") == "storage")["node"].to_list())
    if not storage:
        return
    methods = _node_methods(provider, _SBM_KEY, "storage_binding_method")
    hit = [n for n in storage if methods.get(n, "bind_forward_only") == "bind_forward_only"]
    if not hit:
        return
    from flextool.engine import _behaviour_changes
    rp = sorted({ts for ts in active_timeset_names if ts in rp_weights})
    raise FlexToolConfigError(
        f"Solve '{complete_solve_name}' is on representative periods (timeset(s) {', '.join(rp)} have "
        f"representative_period_weights) and {_behaviour_changes.storages_named(hit)} use storage_binding_method "
        "bind_forward_only (also the default): the forward chain runs through the representative periods without "
        "their weights, while the flows of each period count with its weight, so energy stored in a period of low "
        "weight could be used in a period of high weight, for free. Blended weights need to be provided: choose a "
        "*_blended_weights storage_binding_method for these nodes (bind_forward_only_blended_weights, "
        "bind_within_solve_blended_weights or bind_within_period_blended_weights), or bind_within_timeblock to cycle "
        "the storage within each representative period.")


def _rp_chain_reaches_model_end(state, solves: list, complete_solve: dict, at_end: list[bool], index) -> list[bool]:
    """``at_end`` with every representative-period solve (not rolling) that holds the model's last period set: its
    base-period chain runs to the end of that period, so the chain end is the end of the model although its last
    representative block may lie earlier in the year (rp_inter_period_end_fix pins fix_end there)."""
    ends = [index.end(s) for s in solves]
    known = [e for e in ends if e is not None]
    if not known:
        return at_end
    last_rank = max(known)[0]
    out = list(at_end)
    for i, solve in enumerate(solves):
        cs = complete_solve.get(solve, solve)
        if out[i] or ends[i] is None or ends[i][0] != last_rank:
            continue
        if state.solve.solve_modes.get(cs) == "rolling_window":
            continue
        names = [ts for _p, ts in state.solve.timesets_used_by_solves.get(cs, [])]
        if any(ts in state.timeline.rp_weights for ts in names):
            out[i] = True
    return out


_SBM_KEY = "input/node__storage_binding_method"
_SSE_KEY = "input/node__storage_start_end_method"
# period: only for a kind that holds per period (usage_cap_period); None otherwise.
_ROLL_END_SCHEMA = {"node": pl.Utf8, "kind": pl.Utf8, "value": pl.Float64, "period": pl.Utf8}


def _method_col(frame: pl.DataFrame, name: str) -> "str | None":
    """The method column of a per-solve ``input/node__<name>`` frame (``<name>`` or ``method``)."""
    for col in (name, "method"):
        if col in frame.columns:
            return col
    return None


def _node_methods(provider, key: str, name: str) -> dict[str, str]:
    """``{node: method}`` of the per-solve frame ``key`` (empty when absent)."""
    if not provider.has(key):
        return {}
    frame = provider.get(key)
    col = _method_col(frame, name) if frame is not None and frame.height > 0 else None
    if col is None or "node" not in frame.columns:
        return {}
    return {str(n): str(m) for n, m in zip(frame["node"].to_list(), frame[col].to_list()) if m is not None}


def _frame_nodes(frame) -> set[str]:
    """The ``node`` values of a handoff frame (empty for None)."""
    if frame is None or frame.height == 0 or "node" not in frame.columns:
        return set()
    return {str(n) for n in frame["node"].to_list()}


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


class _RollingStorage:
    """Run-scoped bookkeeping of the bind_within_solve storage carry of rolling solves
    (:func:`_anchor_cyclic_in_rolling`).

    ``start[instance][node]``: the level (MWh) the first roll of a solve instance (``(complete solve, parent
    sub-solve)``) started from, which the instance's closing rolls return to.  ``original[level key]``: the per-solve
    ``input/node__storage_binding_method`` frame of the level before a roll rewrote it (the level Provider is shared
    by the rolls of a level, so each roll starts from the methods the user gave).  ``pending[solve]``: ``(instance,
    nodes)`` whose start level is read from the solve's handoff (``cycle_start_state``) once it is solved.
    ``borrowed_check[solve]``: ``(nodes, upper start levels)`` of the nodes a nested first roll starts from
    fix_start while the upper solve fixes them: their start is compared with the upper solve's.
    """

    def __init__(self) -> None:
        self.start: dict = {}
        self.original: dict = {}
        self.pending: dict = {}
        self.seen: set = set()
        self.borrowed_check: dict = {}
        self.warned: set = set()
        # scope_start[(instance, node, period, first step)]: the level (MWh) a bind_within_period /
        # bind_intraperiod_blocks period started from in the roll that began it.
        self.scope_start: dict = {}
        self.scope_pending: dict = {}
        # scope_from_start[solve]: (scope key, node) of the nested first rolls whose scope starts from fix_start: the
        # start level is read from the handoff's cycle_start_state.
        self.scope_from_start: dict = {}
        # usage_start[(instance, period)]: where the period starts in the instance (the first sub-solve holding it);
        # usage_realised[(instance, period, node)]: the fix_usage energy the instance's sub-solves realised in it.
        self.usage_start: dict = {}
        self.usage_realised: dict = {}
        # model_start[(nesting level, node)]: the level (MWh) the node started the model from at that nesting level
        # (the first sub-solve of the level at the model start, :func:`_record_model_start`).
        self.model_start: dict = {}

    def model_start_of(self, level: int, node: str) -> "float | None":
        """The level (MWh) ``node`` started the model from at nesting ``level``; a level that did not start it
        (no sub-solve of it at the model start) takes the nearest level that did, the deeper one first."""
        if (level, node) in self.model_start:
            return self.model_start[(level, node)]
        known = sorted((abs(lv - level), -lv) for lv, n in self.model_start if n == node)
        return self.model_start[(-known[0][1], node)] if known else None


def _restore_methods(store: _RollingStorage, level_key, provider) -> "pl.DataFrame | None":
    """The per-solve ``input/node__storage_binding_method`` of the level as the downgrades left it, before a roll's
    rewrite (:func:`_anchor_cyclic_in_rolling`): put back in the shared level Provider and returned (None when the
    Provider has none)."""
    if not provider.has(_SBM_KEY):
        return None
    if level_key not in store.original:
        store.original[level_key] = provider.get(_SBM_KEY)
    sbm = store.original[level_key]
    provider.put(_SBM_KEY, sbm)
    return sbm


def _storages_named(names) -> str:
    from flextool.engine import _behaviour_changes
    return _behaviour_changes.storages_named(names)


def _warn_storages(code: str, complete_solve_name: str, names: list, details: str) -> None:
    """A coded storage warning: one line per code and complete solve, naming how many and the first
    (:func:`flextool.run_warnings.warn_grouped`); *details* (the roll, the period) go to the debug log."""
    from flextool import run_warnings
    run_warnings.warn_grouped(code, names, solve=complete_solve_name, details=details)


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


def _rp_period_previous(jump_list: list, active_time_list: dict, rp_periods: list[str],
                        period_order: list[str]) -> list[tuple[str, str]]:
    """``(period, previous period)`` of the across-solve representative-period chain.

    The previous period of an rp period is the ``previous_period`` of its first step in the solve's step-jump
    list (the predecessor the bind_within_solve / bind_forward_only state lag uses), kept when it is another rp
    period.  The solve's first period has none (its jump wraps to the last period: that is the
    bind_within_solve_blended_weights closure, built in the model).

    In a stochastic solve the pairs form a tree (``make_step_jump``): the first forecast branch period of a fan
    after the trunk period (the shared period before the split) continues from the trunk period, a later branch
    period from the previous period of the same branch.  The branch periods of a fan in the solve's first period
    self-cycle in the jump list, so they have no previous period: they are the roots of the tree (as
    ``period_first_of_solve``).
    """
    prev_of = {(row[0], row[1]): row[4] for row in jump_list}
    out: list[tuple[str, str]] = []
    for period in rp_periods:
        steps = active_time_list.get(period) or []
        if period == period_order[0] or not steps:
            continue
        prev = prev_of.get((period, steps[0].timestep))
        if prev is not None and prev != period and prev in rp_periods:
            out.append((period, str(prev)))
    return out


def _downgrade_rp_methods_for_non_rp_solve(
    *, solve, complete_solve_name, roll_index,
    active_timeset_names, rp_weights, provider, logger,
) -> None:
    """Silently downgrade RP storage-binding methods on non-RP solves.

    Replaces the Phase 5 strict check.  When the solve's active timeset
    has NO entry in ``state.timeline.rp_weights``, rewrite the per-solve
    Provider's ``input/node__storage_binding_method`` frame in place,
    mapping every ``*_blended_weights`` row to its non-RP equivalent
    (see :data:`_RP_METHOD_DOWNGRADE`).  When at least one active
    timeset DOES have RP weights, do nothing — the RP path is correct.

    The rewrite touches ONLY the per-solve provider (in-memory); the
    on-disk DB and any upstream input CSVs stay unchanged.  Downstream
    in this same solve, ``preprocessing_solve_time.run`` will derive
    ``solve_data/node__storage_binding_method`` from the rewritten
    ``input/...`` key, so all later consumers see the downgraded values
    for THIS solve only.

    Args:
        solve: per-iter solve name (used in log line).
        complete_solve_name: fully-qualified solve name (debug aid).
        roll_index: integer roll within ``complete_solve_name`` (debug aid).
        active_timeset_names: list of timeset names active in this solve.
        rp_weights: ``state.timeline.rp_weights`` dict (timeset -> weights).
        provider: the per-solve :class:`FlexDataProvider`.
        logger: solve logger.  One info-level line is emitted per
            (old, new) downgrade-mapping that fires.
    """
    # Any active timeset has RP weights → RP path is correct, no-op.
    for ts_name in active_timeset_names:
        if ts_name in rp_weights:
            return

    key = "input/node__storage_binding_method"
    if not provider.has(key):
        return  # no per-solve binding-method frame at all → nothing to do
    sbm = provider.get(key)
    if sbm.height == 0:
        return
    if "storage_binding_method" in sbm.columns:
        method_col = "storage_binding_method"
    elif "method" in sbm.columns:
        method_col = "method"
    else:
        return  # malformed frame; let downstream loaders surface it

    # Count downgrades per (old, new) pair so the log line is precise.
    per_pair_counts: dict[tuple[str, str], int] = {}
    for old, new in _RP_METHOD_DOWNGRADE.items():
        n = (sbm.filter(pl.col(method_col) == old)
                .select("node")
                .unique()
                .height)
        if n > 0:
            per_pair_counts[(old, new)] = n
    if not per_pair_counts:
        return  # nothing to downgrade

    # In-place rewrite via a single replace_strict expression.
    sbm_new = sbm.with_columns(
        pl.col(method_col).replace_strict(
            _RP_METHOD_DOWNGRADE, default=pl.col(method_col),
        ).alias(method_col),
    )
    provider.put(key, sbm_new)

    ts_summary = (
        ", ".join(active_timeset_names) if active_timeset_names else "(none)"
    )
    for (old, new), n in per_pair_counts.items():
        logger.info(
            f"Solve '{solve}' has no representative_period_weights "
            f"for active timeset {ts_summary}; downgrading {n} node(s) "
            f"from {old} to {new}."
        )


# With storage_start_end_method fix_start_end the storage_binding_method is ignored, except bind_forward_only
# (reference/nodes.md, "Meaning"): a cyclic method chains forward instead.  Used by :func:`_fix_start_end_chains_forward`.
_FIX_START_END_FORWARD: dict[str, str] = {
    "bind_within_timeblock":              "bind_forward_only",
    "bind_within_period":                 "bind_forward_only",
    "bind_within_solve":                  "bind_forward_only",
    "bind_within_solve_blended_weights":  "bind_forward_only_blended_weights",
    "bind_within_period_blended_weights": "bind_forward_only_blended_weights",
}


# In a solve on representative periods a plain forward chain would link the representative periods without their
# weights (a configuration error, :func:`_reject_plain_forward_in_rp_solve`): a fix_start_end node with a cyclic method
# there chains forward on the base-period chain, where storage_state_start and storage_state_end act (owner decision,
# Phase 5).  These nodes are not warned about by :func:`_warn_plain_cyclic_in_rp_solve` (fix_start_end ignores the
# binding method, and the blended chain weights the transfer).
_FIX_START_END_FORWARD_RP: dict[str, str] = {
    **_FIX_START_END_FORWARD,
    "bind_within_timeblock":   "bind_forward_only_blended_weights",
    "bind_within_period":      "bind_forward_only_blended_weights",
    "bind_within_solve":       "bind_forward_only_blended_weights",
    "bind_intraperiod_blocks": "bind_forward_only_blended_weights",
}

# A solve that mixes periods with and without representative periods has no base-period chain over all of them
# (a *_blended_weights method is a configuration error there, :func:`_check_rp_solve`) and a plain forward chain
# would run through the representative periods unweighted, so a fix_start_end node with a cyclic method cycles
# within each timeblock (bind_within_timeblock: each representative period is a timeblock, no energy moves between
# them): storage_state_start pins the first timeblock's cycle, storage_state_end the last step of the model
# (review M6; before, bind_within_timeblock mapped to the blended chain and the solve stopped with the error about
# a *_blended_weights method the user never set).
_FIX_START_END_MIXED_RP: dict[str, str] = {
    "bind_within_period":                 "bind_within_timeblock",
    "bind_within_solve":                  "bind_within_timeblock",
    "bind_intraperiod_blocks":            "bind_within_timeblock",
    "bind_within_solve_blended_weights":  "bind_within_timeblock",
    "bind_within_period_blended_weights": "bind_within_timeblock",
}


def _fix_start_end_chains_forward(*, solve: str, provider, logger, complete_solve_name: str | None = None,
                                  logged: set | None = None, rp: bool = False, mixed: bool = False) -> None:
    """Map the cyclic storage_binding_method of fix_start_end nodes to its forward-only equivalent, per solve.

    With ``storage_start_end_method`` *fix_start_end* the binding method is ignored (exception bind_forward_only,
    reference/nodes.md): the start and the end are pinned separately, so the chain must not close on itself (a cycle's
    start is its end, and with storage_state_start ≠ storage_state_end the two pins would conflict).  Rewrites the
    per-solve ``input/node__storage_binding_method`` frame in place (see :data:`_FIX_START_END_FORWARD`), like
    :func:`_downgrade_rp_methods_for_non_rp_solve`, after it (a blended node of a solve without representative
    periods: blended → plain → forward-only).  Every ``storage_bind_*`` set of the solve comes from that frame.
    ``bind_intraperiod_blocks`` stays: it is a time aggregation, not a cycle scope; such a node keeps its blocks and
    the model chains them forward (``model._self_discharge.block_forward_nodes``).  In a solve on representative
    periods (``rp``) the forward chain is the base-period chain (:data:`_FIX_START_END_FORWARD_RP`): a
    bind_within_timeblock fix_start_end node there becomes bind_forward_only_blended_weights, so its start and end
    are pinned on the weighted chronological level, not on a chain through the representative periods.  In a solve
    that also has periods without representative periods (``mixed``) such a node cycles within each timeblock
    instead (:data:`_FIX_START_END_MIXED_RP`).  One info line per mapping (the
    shared frame is mapped once, so later rolls log nothing); the bind_intraperiod_blocks line keeps its method, so
    it is logged once per (complete solve, node) through ``logged``.
    """
    sbm_key, sse_key = "input/node__storage_binding_method", "input/node__storage_start_end_method"
    if not (provider.has(sbm_key) and provider.has(sse_key)):
        return
    sbm, sse = provider.get(sbm_key), provider.get(sse_key)
    if sbm.height == 0 or sse.height == 0:
        return
    sbm_col = "storage_binding_method" if "storage_binding_method" in sbm.columns else "method"
    sse_col = "storage_start_end_method" if "storage_start_end_method" in sse.columns else "method"
    if sbm_col not in sbm.columns or sse_col not in sse.columns:
        return  # malformed frame; let the downstream loaders surface it
    nodes = sorted(str(n) for n in sse.filter(pl.col(sse_col) == "fix_start_end")["node"].unique().to_list())
    if not nodes:
        return
    mapping = (_FIX_START_END_MIXED_RP if rp and mixed else _FIX_START_END_FORWARD_RP if rp
               else _FIX_START_END_FORWARD)
    hit = (sbm.filter(pl.col("node").cast(pl.Utf8).is_in(nodes) & pl.col(sbm_col).is_in(list(mapping)))
           .select(pl.col("node").cast(pl.Utf8), pl.col(sbm_col).cast(pl.Utf8)).unique().sort("node"))
    if hit.height > 0:
        is_hit = pl.col("node").cast(pl.Utf8).is_in(hit["node"].to_list())
        provider.put(sbm_key, sbm.with_columns(
            pl.when(is_hit)
            .then(pl.col(sbm_col).replace_strict(mapping, default=pl.col(sbm_col)))
            .otherwise(pl.col(sbm_col))
            .alias(sbm_col)))
        for node, method in hit.iter_rows():
            logger.info(f"Solve '{solve}': storage node '{node}' has storage_start_end_method fix_start_end, so its "
                        f"storage_binding_method {method} is ignored; it "
                        + ("cycles within each timeblock" if mapping[method] == "bind_within_timeblock"
                           else "chains forward") + f" ({mapping[method]}).")
    blocks = (sbm.filter(pl.col("node").cast(pl.Utf8).is_in(nodes) & (pl.col(sbm_col) == "bind_intraperiod_blocks")
                         & pl.lit("bind_intraperiod_blocks" not in mapping))
              ["node"].cast(pl.Utf8).unique().sort().to_list())
    for node in blocks:
        if logged is not None:
            key = (complete_solve_name or solve, node)
            if key in logged:
                continue
            logged.add(key)
        logger.info(f"Solve '{solve}': storage node '{node}' has storage_start_end_method fix_start_end, so its "
                    "bind_intraperiod_blocks blocks chain forward instead of looping within each period.")


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
            the native :class:`flextool.engine.input._solve_state.RunnerState`
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
    # A period_total user constraint must hold in some solve of the run (rolls that split periods leave it out).
    solve_builder.check_period_total_enforced()

    # ------------------------------------------------------------------
    # 1b. OC1 rolling separate_dispatch — INTERLEAVE the dispatch sibling's
    # rolls with its UC parent's rolls so the run order is
    # ``[uc_roll_0, disp_roll_0, uc_roll_1, disp_roll_1, ...]``.  The sibling
    # was spawned ROLLING (mirroring the parent — spawn_realized_dispatch_
    # sibling), so expansion produced two parallel per-roll sequences
    # concatenated (``[uc_0, uc_1, uc_2, disp_0, disp_1, disp_2]``); zip them
    # by roll index.  This gives, via the linear last_captured_solve chain:
    # uc_k's predecessor = disp_{k-1} (uc rolls forward on the realized
    # dispatch), disp_k's predecessor = uc_k (the commitment pin).  Also
    # record the PER-ROLL realized_dispatch_parent so the W4c projection +
    # the fix_online producer key off the specific UC roll.  No-op for
    # single-shot separate_dispatch (no ``_roll_`` names) and for plain
    # rolling (no realized_dispatch_parent).
    _rdp = getattr(state.solve, "realized_dispatch_parent", {})
    if _rdp:
        _roll_re = re.compile(r"^(?P<base>.+)_roll_(?P<idx>\d+)$")
        _by_base: dict = {}
        for _nm in all_solves:
            _m = _roll_re.match(_nm)
            if _m:
                _by_base.setdefault(_m.group("base"), {})[
                    int(_m.group("idx"))] = _nm
        _uc_to_disp: dict = {}
        for _sib_base, _par_base in list(_rdp.items()):
            _sib_rolls = _by_base.get(_sib_base)
            _par_rolls = _by_base.get(_par_base)
            # A realization roll (``_realization_tail``) keeps the UC roll's whole horizon: it realizes the jump and
            # looks ahead after it with the forecasts made at the jump end.
            _rolls_ahead = (
                _realization_tail.realization_roll_method(state.solve, _sib_base) == "separate_dispatch")
            if _sib_rolls and _par_rolls:
                for _idx, _uc in _par_rolls.items():
                    _dp = _sib_rolls.get(_idx)
                    if _dp is not None:
                        _uc_to_disp[_uc] = _dp
                        _rdp[_dp] = _uc           # per-roll parent map
                        # A realized-dispatch roll that is no realization roll
                        # (``_rolls_ahead`` False: delayed units, recourse, ...,
                        # ``_realization_tail.realization_roll_method``) realizes
                        # ONLY its JUMP.  Its SOLVE
                        # window is the realized (jump) time list instead of the
                        # parent's full roll horizon: without forecasts after the
                        # jump, the look-ahead has no realized data and no online
                        # pin, so solving it forced spurious unserved load.
                        # ``realized_time_lists[_dp]`` here is the start→next-jump
                        # slice (untrimmed at this point); copy it as the active
                        # window.  The fix_online pin and the disp→disp realized
                        # carry already align on the jump.  A realization roll
                        # keeps the horizon and looks ahead with the forecasts
                        # made at the end of its jump (its plan, step 2.9).
                        if _dp in realized_time_lists and not _rolls_ahead:
                            active_time_lists[_dp] = copy.deepcopy(
                                realized_time_lists[_dp])
        if _uc_to_disp:
            _rebuilt: list = []
            _emitted: set = set()
            _disp_set = set(_uc_to_disp.values())
            for _nm in all_solves:
                if _nm in _emitted or _nm in _disp_set:
                    continue
                _rebuilt.append(_nm)
                _emitted.add(_nm)
                _dp = _uc_to_disp.get(_nm)
                if _dp is not None:
                    _rebuilt.append(_dp)
                    _emitted.add(_dp)
            # append any disp rolls not yet emitted (defensive; shouldn't happen)
            for _dp in _disp_set:
                if _dp not in _emitted:
                    _rebuilt.append(_dp)
                    _emitted.add(_dp)
            all_solves = _rebuilt

    # The periods of each sub-solve, before the stochastic pass names the periods of a fan by their branches.
    _sub_solve_periods = {_s: list(active_time_lists.get(_s) or {}) for _s in all_solves}

    # NOTE: the realized-dispatch overlap resolution (last-wins) runs
    # AFTER the stochastic pass below (step 3.5), where
    # ``realized_time_lists`` is authoritative — the stochastic pass
    # re-derives it.

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
        # The period of each ``(key, period)`` tuple: the key is the row name of a solve split by a 2-D period
        # map (any name), so reading it lost the history of a split solve whose rows are not period names.
        period_dict_names = {
            t[1] for t in (
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

    # The run calendar: the year of every period of the run (the start of operation of an investment with a
    # construction lead time is rounded on it).  The same frame goes to every sub-solve.
    run_calendar = build_run_calendar(solve_period_history, state.solve.real_solves)
    # The existing cohorts of the run: fixed when the run starts (the run start, the lifetime there, the end of life
    # rounded on the run calendar), the same frame for every sub-solve.
    run_existing_cohort = _run_existing_cohort(state, run_calendar)
    # The reported costs of a run of several solves count from the start of the run, each once.
    if len(state.solve.real_solves) > 1:
        from flextool.engine import _behaviour_changes
        _behaviour_changes.notice("chained_solve_costs_from_run_start", "solves", state.solve.real_solves)

    # ------------------------------------------------------------------
    # 2.9 Realization-roll plans (``_realization_tail``): per roll of a gated solve, from the authored rows (never
    # changed), before the expansion reads them.
    # ------------------------------------------------------------------
    _tail_rolls = _realization_tail.realization_rolls(
        state.solve, all_solves, complete_solve, active_time_lists, realized_time_lists)
    _tail_timelines = _solve_timelines(state, all_solves, complete_solve, active_time_lists, set(_tail_rolls))
    state.solve.roll_tail_plan = _roll_tail_plans(state.solve, _tail_rolls, _tail_timelines)
    state.solve.roll_branch_rows = _realization_tail.roll_branch_rows(
        state.solve, _tail_rolls, state.solve.roll_tail_plan)

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
        fan_label_time_lists,
    ) = stochastic_solver.create_stochastic_periods(
        state.solve.stochastic_branches,
        all_solves,
        complete_solve,
        active_time_lists,
        fix_storage_time_lists,
        realized_time_lists,
    )
    # The plans (2.9) and the expansion have passed their checks: the realization notices of the listed solves.
    _realization_tail.announce_realization_notices(state.solve)

    # ------------------------------------------------------------------
    # 3.1 Realization plan per sub-solve.  A period label has no dispatch, so
    # the cells a sub-solve commits inside its fan (its pre-stochastic
    # realized and fix-storage cells of the labels) are kept as
    # ``committed_in_fan``: the non-anticipativity window of a dispatch solve
    # and the trigger of the committed-window check.  The labels'
    # pre-stochastic steps are kept as ``label_steps``.
    # ------------------------------------------------------------------
    _rdp_registry = getattr(state.solve, "realized_dispatch_parent", {}) or {}
    committed_in_fan_frames: dict[str, pl.DataFrame] = {}
    label_step_frames: dict[str, pl.DataFrame] = {}
    for _s in all_solves:
        _times = fan_label_time_lists.get(_s)
        if _times is None:
            continue
        _cs = complete_solve[_s]
        committed_in_fan_frames[_s] = pl.concat([
            solve_writers.derive_realized_dispatch(
                _times.realized, _cs, state.solve.realized_periods.get(_cs, [])),
            solve_writers.derive_fix_storage_timesteps(
                _times.fix_storage, _cs, state.solve.fix_storage_periods.get(_cs, [])),
        ]).unique(maintain_order=True)
        label_step_frames[_s] = solve_writers.derive_periods_steps(_times.active)
    realization_plans = {
        _s: _realization.build_realization_plan(
            _s, period__branch_lists.get(_s, []), _rdp_registry,
            committed_in_fan=committed_in_fan_frames.get(_s))
        for _s in all_solves
    }

    # ------------------------------------------------------------------
    # 3.5 Resolve realized-dispatch overlap across solves (LAST-WINS).
    #
    # ``realized_time_lists`` is authoritative only HERE — the stochastic
    # pass above re-derives it.  When two solves realize the same
    # (period, timestep), the LATER solve in cascade order wins: it keeps
    # the cell and the earlier solve's claim is dropped.  This is the
    # maintainer-directed resolution ("the lower level / later solve
    # wins") and makes each solve's persisted realized slice disjoint, so
    # the downstream multi-solve output union stays a clean concat.
    #
    # Only the timesteps a later solve actually claims are dropped; the
    # rest of the earlier solve's window is kept in order (it may become
    # non-contiguous — every downstream consumer of
    # ``solve_data/realized_dispatch`` is set/membership based, and the
    # roll-end storage state reads the last step the solve still
    # realizes).  Only periods a solve really
    # realizes (its ``realized_periods`` — the same filter
    # ``derive_realized_dispatch`` applies) take part.  The resolved
    # lists flow into ``solve_data/realized_dispatch.csv`` and thus each
    # roll's ``flex_data.realized_dispatch`` that drives per-roll
    # persistence.  See :mod:`._realized_overlap`.
    #
    # Each drop is reported as one warning per (earlier solve, later
    # solve, period).  Within-solve rolls realize disjoint jump windows
    # and never overlap.  The auto-spawned realized-dispatch sibling
    # (``realized_dispatch_parent``) re-realizes its parent's window by
    # design, so that pair is silent.
    # ------------------------------------------------------------------
    _realized_period_names = {
        _s: {
            t[1]
            for t in state.solve.realized_periods.get(complete_solve[_s], [])
        }
        for _s in realized_time_lists
    }
    _dispatch_overlaps = resolve_realized_dispatch_overlap(
        realized_time_lists, _realized_period_names,
    )
    _rdp_map = getattr(state.solve, "realized_dispatch_parent", {}) or {}
    warn_realized_dispatch_overlaps(
        _dispatch_overlaps,
        silent_pairs={(_par, _sib) for _sib, _par in _rdp_map.items()},
    )

    # Investment-period overlap: a period's investments should be
    # realized by one solve only.  Warning only (the maintainer decides
    # whether to escalate).  One entry per distinct complete solve, in
    # cascade order — rolls of one solve share its configuration and are
    # not compared with each other.
    _invest_claims: list[tuple[str, list[str]]] = []
    _seen_complete: set[str] = set()
    for _s in all_solves:
        _cs = complete_solve[_s]
        if _cs in _seen_complete:
            continue
        _seen_complete.add(_cs)
        _invest_claims.append((_cs, effective_realized_invest_periods(
            _cs,
            realized_invest_periods=state.solve.realized_invest_periods,
            invest_periods=state.solve.invest_periods,
            realized_periods=state.solve.realized_periods,
        )))
    warn_realized_invest_overlaps(find_realized_invest_overlaps(_invest_claims))
    # A period is dispatched only after its investments are realized, per sub-solve (a nested solve renamed for one
    # period has that period), each under its complete solve: an error before anything solves.
    check_dispatch_after_investments(state.solve, [
        (complete_solve[_s], _sub_solve_periods.get(_s, [])) for _s in all_solves])
    # A total CO2 limit needs every solve that has it to see all the model's periods (D54): an error before anything
    # solves.
    check_co2_total_sees_all_periods(
        solve_period_views(all_solves, complete_solve, parent_roll, _sub_solve_periods, _rdp_map),
        co2_total_groups(getattr(state, "cascade_input_provider", None), getattr(state, "input_source", None)),
        weightless_periods(state.solve.solve_period_years_represented))

    # ------------------------------------------------------------------
    # 3.6 W4c — project each stochastic parameter onto the REALIZED branch
    # for auto-spawned realized-dispatch siblings.
    #
    # A W4b realized-dispatch sibling (``<solve>__realized_dispatch``) is
    # deterministic and drops ``stochastic_branches``, so the stochastic
    # pass above leaves its ``solve_branch__time_branch_lists`` EMPTY.  The
    # per-solve folds that collapse a stochastic 3d_map
    # ``(branch, time_start, time) -> (d, t)`` — ``_profiles``'s
    # ``_profile_stochastic_lf`` and ``_pdt_lookup.PdtLookup.get`` — then
    # find no branch rows, drop the 3d_map, and the zero-fallback fills 0
    # (wind -> 0 -> garbage).  Re-populate the sibling's branch rows from
    # the parent's ALREADY-materialized realized-branch rows: the
    # ``(p, p) in period__branch_lists[parent]`` predicate selects the
    # realized self-rows (the SAME predicate ``derive_solve_branch_weight``
    # uses at ``_solve_writers.py``) and excludes synthetic forecast
    # rows (``<p>_high`` ...).  A MULTI-period realized window carries one
    # realized-branch row per realized period (the continuation loop in
    # ``_stochastic.py`` appends ``(period, realized_branch)`` for each), so
    # the filter keeps each — the folds then select the realized branch
    # unchanged, in every realized period.
    #
    # This is purely a scaffolding injection: no new var/cstr/param, no
    # ``_param_shapes`` routing change, and the sibling stays
    # NON-genuinely-stochastic (``is_genuinely_stochastic`` keys on
    # ``period_branch_full`` / ``period_in_use_set``, which are unchanged
    # here, NOT ``solve_branch__time_branch``) -> zero branch LP variables.
    # Gated on ``realized_dispatch_parent`` so it is a strict no-op when no
    # sibling is spawned (byte-parity on deterministic 2d-profile fixtures);
    # parent-absent / parent-empty cases are handled gracefully.
    # ------------------------------------------------------------------
    realized_dispatch_parent = getattr(
        state.solve, "realized_dispatch_parent", {}
    ) or {}
    # A planned realization roll (``roll_tail_plan``) gets its realized branch from its plan (3.8) and keeps the
    # forecast rows of its tail.
    _planned = getattr(state.solve, "roll_tail_plan", None) or {}
    for sibling, parent in realized_dispatch_parent.items():
        if sibling not in solve_branch__time_branch_lists or sibling in _planned:
            continue
        if (
            parent not in solve_branch__time_branch_lists
            or parent not in period__branch_lists
        ):
            continue
        # Cut to the sibling's own periods: a single solve's realization solve
        # dispatches its realized periods only, a realization roll its jump
        # (item 255).  The NA-tied online pin needs no cut: the model pins
        # only the sibling's declared cells (online_fix_eq).
        _sibling_periods = set(active_time_lists.get(sibling, {}))
        solve_branch__time_branch_lists[sibling] = [
            row for row in project_realized_branch_time_branch_rows(
                solve_branch__time_branch_lists[parent],
                period__branch_lists[parent],
            )
            if row[0] in _sibling_periods
        ]

    # 3.7 ``dispatch_realization_method`` deterministic: the stochastic pass
    # above solved the solve as a deterministic one (its stochastic_branches
    # rows are set aside, ``SolveConfig.apply_deterministic_realization``).
    # It reads its realized branch flattened into one timeline: every period
    # of every roll reads the realized branch (``solve_branch__time_branch``)
    # through one analysis-time tier per ``realized: yes`` row, the one
    # nearest to the roll's start first (``branch_time_start`` rows with
    # ``read_first`` = ``flat``, ``_pdt_lookup.ts_tiers``), so each step reads
    # the nearest realized series that covers it; a step none covers is an
    # input error (``_check_pre_reveal_reads_have_forecast``).  Rolls only cut
    # that timeline.  No fan: no fan start, no branch weight but the self row's.
    # Distance to the roll's start is counted on each period's full timeline
    # (every step of the timeline of the period's timeset, also those the
    # solve does not use), so a realized series made before the solve's
    # first step is read; a step it does not cover is left to the coverage
    # check.
    deterministic_branches = getattr(state.solve, "deterministic_branches", None) or {}
    flat_time_start_lists: dict[str, list[tuple[str, str, str]]] = {}

    def _read_realized_flat(s: str, periods: list, branch: str, times) -> None:
        # Periods *periods* of sub-solve *s* read *branch* through one ``flat`` tier per analysis time of *times*.
        solve_branch__time_branch_lists[s] = [
            *(row for row in solve_branch__time_branch_lists[s] if row[0] not in periods),
            *((d, branch) for d in periods)]
        flat_time_start_lists[s] = [(d, ts, READ_FIRST_FLAT) for d in periods for ts in times]

    _det_timelines = _solve_timelines(state, all_solves, complete_solve, active_time_lists, deterministic_branches)
    for _s in all_solves:
        _info = deterministic_branches.get(complete_solve[_s])
        _steps = [(_p, _e.timestep) for _p, _ats in active_time_lists[_s].items() for _e in _ats]
        if not _info or not _steps:
            continue
        _times = realized_analysis_times(_info, _det_timelines[complete_solve[_s]], _steps[0])
        if not _times:
            _name = state.solve.map2d_duplicate_of.get(complete_solve[_s], (complete_solve[_s],))[0]
            raise FlexToolConfigError(
                f"Solve '{_name}' has stochastic_branches and dispatch_realization_method deterministic, but no "
                "realized: yes row of its stochastic_branches starts at a time step of the timeline of its "
                "periods, so it has no realized series to read. Give the realized: yes rows the analysis_time "
                "of a time step of that timeline, or set dispatch_realization_method within_solve or "
                "separate_dispatch.")
        branch_start_time_lists[_s] = None
        _read_realized_flat(_s, list(active_time_lists[_s]), next(row[1] for row in _info if row[3] == "yes"),
                            _times)
    # 3.8 A realization roll (``roll_tail_plan``) reads its realized branch flattened in the same way over its jump
    # periods (every period of a ``deterministic`` roll): one source, the plan (``RollTail.realized`` /
    # ``realized_times``).  Its tail keeps the forecast rows of the expansion.
    for _s, _tail in (getattr(state.solve, "roll_tail_plan", None) or {}).items():
        _read_realized_flat(_s, list(dict.fromkeys(_d for _d, _t in _tail.jump_steps)), _tail.realized,
                            _tail.realized_times)

    # The fan start of each solve (``solve_data/branch_time_start``): a branch
    # period whose own first step authors no stochastic-time value for a cell
    # reads the forecast authored at the fan start.  A realization-dispatch
    # sibling reads the parent's realized branch, so it takes the parent's
    # rows of the real period names.
    # Rows (period, time) here; the flat merge below adds (period, time, read_first) triples, so the W4c projection
    # (which unpacks pairs) runs before it.
    branch_time_start_lists: dict[str, list[tuple[str, str] | tuple[str, str, str]]] = {
        s: solve_writers.branch_time_start_rows(
            period__branch_lists[s], branch_start_time_lists.get(s))
        for s in period__branch_lists
    }
    for sibling, parent in realized_dispatch_parent.items():
        if sibling not in branch_time_start_lists or parent not in branch_time_start_lists or sibling in _planned:
            continue
        _sibling_periods = set(active_time_lists.get(sibling, {}))
        branch_time_start_lists[sibling] = [
            row for row in solve_writers.project_branch_time_start_rows(
                branch_time_start_lists[parent], period__branch_lists[parent])
            if row[0] in _sibling_periods
        ]
    # The flat rows replace the fan rows of the periods that read the realized branch (3.7, 3.8).
    for _s, _flat in flat_time_start_lists.items():
        _periods = {row[0] for row in _flat}
        branch_time_start_lists[_s] = [
            *_flat, *(row for row in branch_time_start_lists.get(_s, []) if row[0] not in _periods)]

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
        state.timeline.original_timeline,
        provider=cascade_input_provider,
    )

    # ------------------------------------------------------------------
    # 4b. Storage carry routing (run-scoped): each sub-solve continues from the latest state realised before it at
    # its nesting level or deeper (_carry_routing).
    # ------------------------------------------------------------------
    _time_args = dict(
        complete_solve=complete_solve,
        timesets_used_by_solves=state.solve.timesets_used_by_solves,
        timesets__timeline=state.timeline.timesets__timeline,
        timelines=state.timeline.timelines,
        original_timeline=state.timeline.original_timeline,
    )
    _siblings = getattr(state.solve, "realized_dispatch_parent", {}) or {}
    _time_index = _carry_routing.make_time_index(all_solves, active_time_lists, period__branch_lists, **_time_args)
    carry_router = _carry_routing.CarryRouter(
        parent_roll,
        siblings=_siblings,
        position=_time_index.position,
        end_of=_time_index.end_of,
    )

    # The sub-solves whose horizon holds the last step of the entire model (fix_end / fix_start_end pin the end
    # state in each of them, owner decision Q-c15f-D).
    storage_solve_at_model_end = _carry_routing.reaches_model_end(
        all_solves, active_time_lists, period__branch_lists, siblings=_siblings, **_time_args)
    # The representative-period solves whose base-period chain ends at the end of the model (fix_end pins the chain
    # end there, rp_inter_period_end_fix).
    storage_rp_chain_at_model_end = _rp_chain_reaches_model_end(
        state, all_solves, complete_solve, storage_solve_at_model_end, _time_index)
    # The sub-solves whose horizon holds the last step of their solve instance (the closing rolls of a rolling solve:
    # bind_within_solve ends there at the level the instance started from), whether an instance reaches the model end,
    # and where the model starts (the first step of the run: an explicit fix_start applies there).
    storage_solve_at_solve_end = _carry_routing.reaches_solve_end(
        all_solves, active_time_lists, period__branch_lists, parent_roll=parent_roll, **_time_args)
    _instance_of = {s: (complete_solve.get(s, s), parent_roll.get(s)) for s in all_solves}
    _instance_at_model_end: dict = {}
    for _s, _at_end in zip(all_solves, storage_solve_at_model_end):
        _instance_at_model_end[_instance_of[_s]] = _instance_at_model_end.get(_instance_of[_s], False) or _at_end
    _first_positions = [p for p in (_time_index.position(*f) for f in
                                    (_carry_routing.first_step(active_time_lists.get(s) or {}) for s in all_solves)
                                    if f is not None) if p is not None]
    _model_start = min(_first_positions) if _first_positions else None
    _rolling_storage = _RollingStorage()

    # ------------------------------------------------------------------
    # 4c. Input checks against every time step of the run, before the first solve: a rolling run stops before
    # its first roll, not at the roll that reaches a gap (a time map of a group limit that leaves out a step).
    # ------------------------------------------------------------------
    _check_run_inputs = getattr(solver, "check_run_inputs", None)
    if _check_run_inputs is not None:
        _check_run_inputs(lambda: run_time_steps(all_solves, active_time_lists, complete_solve, state.solve,
                                                 state.timeline, period__branch_lists))

    # ------------------------------------------------------------------
    # 5. Per-solve loop.
    # ------------------------------------------------------------------
    first = True
    previous_complete_solve = None
    last_captured_solve: str | None = None
    cached_complete_active_time_lists: dict = {}
    cached_complete_step_jumps: dict = {}
    _warned_rolling_blocks: set = set()
    _logged_fix_start_end_blocks: set = set()

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
    # Phase 2 — expose the full pre-planned sequence of level_keys onto
    # ``state`` so the per-solve callback (``_PolarHighCascadeSolver.run``)
    # can answer the "is there another solve of THIS level coming?"
    # question used by the warm-path per-level retention slim.  Empty
    # tuple is safe (no upcoming iter exists for the empty case).
    state._all_level_keys = tuple(level_keys)
    # Each complete solve's last sub-solve in the run (the order is final here): a rolling solve's gathered
    # solver-outcome note (W-0540) prints after that one.
    state._last_sub_solve_index = {complete_solve[s]: i for i, s in enumerate(all_solves)}

    for i, solve in enumerate(all_solves):
        timer_in_solve = time.perf_counter()

        # Between-solves memory snapshot — fires before every roll/solve
        # (i > 0).  The previous iter now always emits the four phase
        # checkpoints (ending in ``Outputs written``), so this shows
        # what's left in memory at each solve boundary and cross-solve
        # retention is visible per roll in the standard log.  The Δ
        # columns are taken vs. the previous emitted line (``Outputs
        # written``), so positive Δrss here is memory the previous
        # solve's outputs phase did not release.
        _memrec_iter = getattr(state, "_memory_recorder", None)
        if (
            i > 0
            and _memrec_iter is not None
        ):
            try:
                _memrec_iter.checkpoint(
                    "solve_cleanup", state.logger,
                    user_label="Solve cleanup",
                )
            except Exception:
                pass

        # Per-sub-solve marker — plain text line (no timer/memory).
        # Always printed so the user can see roll progression even
        # when the four phase checkpoints are suppressed for within-
        # group rolling iters.
        # The run record keeps the marker text as the solve's log anchor.
        _log_anchor = None
        if _memrec_iter is not None and getattr(_memrec_iter, "verbose", True):
            _log_anchor = (
                f"Solve start: {complete_solve[solve]}, "
                f"{i + 1}/{len(all_solves)}"
            )
            try:
                print(_log_anchor, flush=True)
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
        if cs not in cached_complete_step_jumps:
            # The un-rolled solve's own step_previous (every period an anchor): which step precedes the
            # first step of each roll's periods, and whether it is adjacent (ramp carry-in gate).
            cached_complete_step_jumps[cs] = make_step_jump(
                complete_active_time_lists,
                [(p, p) for p in complete_active_time_lists],
                [],
            )

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
            # Axis-Enum mode (FLEXTOOL_AXIS_ENUMS=1) is run-wide: every sub-solve loads with the cascade
            # provider's vocabulary (None in production, so the sub-solve stays Utf8).
            sub_solve_provider.axis_enums = cascade_input_provider.axis_enums
            sub_solve_provider.contract = cascade_input_provider.contract
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
        # with ``provider=`` (via :func:`_solve_time.run`) can
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
        # A period label has no steps, but its first step is where a roll's fan continues from the previous
        # roll: its forecast branch periods start there (the carry matches the label, then its branches).
        _label_times = fan_label_time_lists.get(solve)
        # Each predecessor also carries where it ends on the base timeline: a solve continues the previous one's
        # last realized step by position in time when their step lengths differ.
        solve_writers.emit_roll_first_step_link(
            {**_label_times.active, **active_time_lists[solve]} if _label_times is not None
            else active_time_lists[solve],
            cached_complete_step_jumps[cs],
            provider=sub_solve_provider,
            complete_active_time_list=complete_active_time_lists,
            end_of=_time_index.end_of,
        )
        # Which periods a roll holds whole (bind_intraperiod_blocks loop there and chain elsewhere); written for
        # every sub-solve (header-only when not rolling): the level Provider is shared by the iters of a level.
        sub_solve_provider.put(
            _provider_keys.SOLVE_DATA_ROLL_PERIOD_WHOLE,
            _roll_period_whole(active_time_lists[solve], complete_active_time_lists, period__branch_lists[solve])
            if state.solve.solve_modes.get(cs) == "rolling_window"
            else _roll_period_whole({}, {}, []))
        # Written for every sub-solve, as the level Provider is shared by the iters of a level.
        sub_solve_provider.put(RUN_CALENDAR_KEY, run_calendar)
        sub_solve_provider.put(RUN_EXISTING_COHORT_KEY, run_existing_cohort)
        # Whether this sub-solve leaves the period_total user constraints out (a roll of a rolling solve that
        # does not realize whole periods in one roll); written for every sub-solve, the Provider is shared.
        sub_solve_provider.put(_provider_keys.SOLVE_DATA_USER_CONSTRAINT_PERIOD_TOTAL_OFF,
                               pl.DataFrame({"value": [solve in state.solve.period_total_off_solves]},
                                            schema={"value": pl.Boolean}))
        sub_solve_provider.put(_provider_keys.SOLVE_DATA_STORAGE_SOLVE_AT_MODEL_END,
                               pl.DataFrame({"value": [storage_solve_at_model_end[i]]}, schema={"value": pl.Boolean}))
        sub_solve_provider.put(_provider_keys.SOLVE_DATA_STORAGE_RP_CHAIN_AT_MODEL_END,
                               pl.DataFrame({"value": [storage_rp_chain_at_model_end[i]]},
                                            schema={"value": pl.Boolean}))
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
        # realized_invest_periods, or (when it is empty and the solve has
        # both invest_periods and realized_periods) the realized_periods
        # that are also invest_periods (#319).  One source with the
        # investment-overlap warning: ``effective_realized_invest_periods``.
        _cs_name = complete_solve[solve]
        solve_writers.emit_periods(
            _cs_name,
            {_cs_name: [
                (p, p) for p in effective_realized_invest_periods(
                    _cs_name,
                    realized_invest_periods=state.solve.realized_invest_periods,
                    invest_periods=state.solve.invest_periods,
                    realized_periods=state.solve.realized_periods,
                )
            ]},
            str(wf / "solve_data/realized_invest_periods_of_current_solve.csv"),
            provider=sub_solve_provider,
        )
        sub_solve_provider.put(
            _provider_keys.SOLVE_DATA_INVEST_PERIODS_OF_CURRENT_SOLVE,
            solve_writers.derive_periods(
                complete_solve[solve], state.solve.invest_periods,
            ),
        )
        # The recourse switch (investments branch after the reveal: SolveConfig.recourse_invest), so the
        # derived/model layers can gate the recourse feature through the provider/workdir (they hold no
        # SolveConfig).  Absent CSV == flag-off.
        solve_writers.emit_stochastic_invest_method(
            "recourse" if state.solve.recourse_invest(complete_solve[solve]) else "none",
            str(wf / "solve_data/stochastic_invest_method.csv"),
            provider=sub_solve_provider,
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

        # ---- LP scaling: emit user-set row-scaling state.  The objective
        # scale (``solve_data/scale_the_objective``) is emitted by the
        # orchestrator when it resolves the value the LP is built with
        # (``_orchestration._emit_provider_obj_scale``).
        solve_writers.emit_p_use_row_scaling(
            solve,
            state.solve.use_row_scaling,
            str(wf / "solve_data/p_use_row_scaling.csv"),
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
        solve_writers.emit_fan_label_steps(
            committed_in_fan_frames.get(solve),
            label_step_frames.get(solve),
            provider=sub_solve_provider,
        )
        # Solve (1) of separate_dispatch (single or rolling: the stochastic solve a realization solve follows) ties
        # across its forecasts only the units online of the _fix_in_uc units (``_non_anticipativity.na_role``).
        _plan = realization_plans.get(solve)
        solve_writers.emit_na_online_fix_in_uc(
            _plan is not None and _plan.realization_solve is not None
            and state.solve.dispatch_realization_method_for(complete_solve[solve]) == "separate_dispatch",
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
        _branch_weights = solve_writers.emit_branch_weights_and_map(
            complete_solve[solve],
            active_time_lists[solve],
            solve_branch__time_branch_lists[solve],
            branch_start_time_lists[solve],
            period__branch_lists[solve],
            state.solve.stochastic_branches,
            provider=sub_solve_provider,
            branch_rows=_realization_tail.plan_branch_rows(state.solve, solve),
        )
        _tail = state.solve.roll_tail_plan.get(solve)
        if _tail is not None:
            # G2b: every forecast of a realization roll has its own branch weight.
            _realization_tail.check_tail_weights(
                solve, _tail, solve_branch__time_branch_lists[solve], _branch_weights)
        # Every period from the fan onwards reads the forecast made at the
        # fan's analysis time, in a single solve and in a roll alike, never a
        # forecast made later (``_pdt_lookup`` "Which forecast a branch
        # period reads").
        # The periods that read a realized branch flattened (3.7, 3.8) have ``flat`` rows.
        solve_writers.emit_branch_time_start(
            branch_time_start_lists.get(solve, []),
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

        matching_rows: list[tuple[str, str, str]] = []
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
                wf / "solve_data/timeline_matching_map.csv", "w", encoding="utf-8", newline=""
            ) as realfile:
                writer = csv.writer(realfile)
                writer.writerow(["period", "step", "upper_step"])
                for (period, step), upper_step in matching_map.items():
                    writer.writerow([period, step, upper_step])
                    matching_rows.append((str(period), str(step), str(upper_step)))
        else:
            with open(wf / "solve_data/timeline_matching_map.csv", "w", encoding="utf-8") as realfile:
                realfile.write("period,step,upper_step\n")
        # The same rows in the Provider: the fix_price reference-price derive reads the Provider only (the disk file
        # stays for the csv dump); written every iter, the level Provider being shared by the iters of a level.
        sub_solve_provider.put(
            _provider_keys.SOLVE_DATA_TIMELINE_MATCHING_MAP,
            pl.DataFrame(matching_rows, schema={"period": pl.Utf8, "step": pl.Utf8, "upper_step": pl.Utf8},
                         orient="row"))

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
            aggregated_keys = state.timeline.create_averaged_timeseries(
                complete_solve[solve], state.solve, state.logger,
                provider=cascade_input_provider,
                work_folder=wf,
            )
            # Mirror every frame the aggregation wrote (pt_* / pbt_* /
            # pdt_* / the energy-margin adder / the base steps / the
            # full-timeline inflow) onto the current sub-solve Provider
            # too: this iter's readers must find them, and a level
            # Provider seeded before this aggregation (or cached from an
            # earlier iter) would otherwise keep the earlier frames.
            for _key in aggregated_keys:
                sub_solve_provider.put(_key, cascade_input_provider.get(_key))
        previous_complete_solve = complete_solve[solve]

        # ---- Block data (node -> new_stepduration group) ----
        # Δ.31: thread the per-sub-solve Provider so the reads find the ``input/*.csv`` frames in memory.  The
        # model's coarse-group frames are built from this layout, so a failure here is an error.
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

        first_of_nested_level = solve in state.solve.first_of_complete_solve
        last_of_nested_level = solve in state.solve.last_of_solve

        # Phase 4.1j — the parent-to-child fix_storage hand-off is now
        # carried exclusively by the typed ``handoff/*`` Provider keys
        # seeded at iteration start by the parent-handoff translator
        # (Phase 4.1e).  The legacy shutil.copy archive path and the
        # in-memory carrier fallback that wrote to the now-dead
        # ``solve_data/fix_storage_*`` Provider keys have been deleted;
        # all downstream consumers (flex_data.py, _derived.py,
        # _arc_unions.py, _per_solve.py) consult ``handoff/*``.

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
        # Phase C — silent degrade.  Replaces the Phase 5 strict check.
        # When the active timeset has no representative_period_weights,
        # rewrite the per-solve provider's node__storage_binding_method
        # so the three ``*_blended_weights`` variants degrade to their
        # non-RP equivalents for THIS solve only.  Same storage entity
        # can now legitimately drive both an RP-active investment solve
        # and a chronological dispatch solve back-to-back.
        _downgrade_rp_methods_for_non_rp_solve(
            solve=solve,
            complete_solve_name=complete_solve[solve],
            roll_index=i,
            active_timeset_names=active_timeset_names,
            rp_weights=state.timeline.rp_weights,
            provider=sub_solve_provider,
            logger=state.logger,
        )
        # Rolling over representative periods is not supported; cycles that link the representative periods
        # without their weights are warned about.
        _reject_rolling_rp(state=state, complete_solve_name=complete_solve[solve],
                           active_timeset_names=active_timeset_names)
        _reject_rolling_with_gaps(state=state, complete_solve_name=complete_solve[solve],
                                  complete_active_time_list=complete_active_time_lists)
        _warn_plain_cyclic_in_rp_solve(complete_solve_name=complete_solve[solve],
                                       active_timeset_names=active_timeset_names,
                                       rp_weights=state.timeline.rp_weights, provider=sub_solve_provider)
        _reject_plain_forward_in_rp_solve(complete_solve_name=complete_solve[solve],
                                          active_timeset_names=active_timeset_names,
                                          rp_weights=state.timeline.rp_weights, provider=sub_solve_provider)
        # fix_start_end ignores a cyclic binding method: after the downgrades, on the same frame.
        _fix_start_end_chains_forward(solve=solve, provider=sub_solve_provider, logger=state.logger,
                                      complete_solve_name=complete_solve[solve], logged=_logged_fix_start_end_blocks,
                                      rp=any(ts in state.timeline.rp_weights for ts in active_timeset_names),
                                      mixed=not all(ts in state.timeline.rp_weights for ts in active_timeset_names))
        # A rolling solve uses bind_within_timeblock as bind_within_period (each period is one timeblock there).
        _timeblock_binds_within_period_in_roll(state=state, complete_solve_name=complete_solve[solve],
                                               active_timeset_names=active_timeset_names,
                                               provider=sub_solve_provider)
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

        # Representative periods: every period of the solve whose timeset has representative_period_weights
        # gets its own rp data (weights, blocks, base-period chain), in the solve's period order; periods
        # without representative periods keep their timeset_weights.  The across-solve blended variants chain
        # the periods in the order of the solve's step jumps (rp_period_previous), like bind_within_solve /
        # bind_forward_only chain v_state.
        # A forecast branch period has the timeset of the period it branches from (one shared rp mapping for
        # every branch); a fan's period label has no steps, is not in the period order and gets no rp data.
        _period_order = list(active_time_lists[solve])
        _timeset_of = dict(solve_writers._with_branch_period_timesets(
            timesets_used, period__branch_lists.get(solve, [])))
        rp_periods = [p for p in _period_order if _timeset_of.get(p) in state.timeline.rp_weights]
        if rp_periods:
            _check_rp_solve(
                complete_solve_name=complete_solve[solve],
                rp_periods=rp_periods,
                period_order=_period_order,
                provider=sub_solve_provider,
            )
            per_period_inputs = []
            for period_name in rp_periods:
                ts_name = _timeset_of[period_name]
                timeline_name = state.timeline.timesets__timeline[ts_name]
                timeline_steps = [step for step, _dur in state.timeline.timelines.get(timeline_name, [])]
                per_period_inputs.append((
                    period_name,
                    state.timeline.rp_weights[ts_name],
                    state.timeline.timeset_durations[ts_name],
                    timeline_steps,
                ))
            non_rp_time = {p: active_time_lists[solve][p] for p in _period_order if p not in rp_periods}
            extra_weight_rows, _ = solve_writers._compute_timeset_cost_weight_rows(
                non_rp_time, timesets_used, state.timeline.timeset_weights,
                period__branch_lists.get(solve, []),
            )
            solve_writers.emit_rp_data(
                per_period_inputs,
                provider=sub_solve_provider,
                period_previous=_rp_period_previous(
                    jump_lists[solve], active_time_lists[solve], rp_periods, _period_order),
                extra_timestep_weight_rows=extra_weight_rows,
            )
        else:
            # The per-solve downgrade above has already rewritten any blended-weights node to its non-RP
            # equivalent in this solve's provider.  Emit the empty RP scaffolding so downstream consumers
            # don't see missing keys.
            solve_writers.emit_empty_rp_data(
                provider=sub_solve_provider,
            )
            solve_writers.emit_timeset_cost_weight(
                active_time_list=active_time_lists[solve],
                timesets_used_by_solve=timesets_used,
                timeset_weights=state.timeline.timeset_weights,
                provider=sub_solve_provider,
                period__branch=period__branch_lists.get(solve, []),
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
        # OC1 rolling separate_dispatch — a dispatch roll takes TWO sources:
        # its PIN (commitment + fixed capacity) from its UC roll (the
        # sequential predecessor), and its ROLL-FORWARD (realized trajectory)
        # from the PREVIOUS dispatch roll.  The handoff dict is keyed by the
        # COMPLETE name, and each roll overwrites its base slot, so the
        # sibling's own complete-name slot still holds disp_{k-1}'s handoff
        # until THIS roll overwrites it at solve end — that is the
        # roll-forward source (None on the first dispatch roll).  Two PARTIAL
        # translates (PIN_FIELDS / ROLL_FORWARD_FIELDS) compose without
        # clobbering.  uc_k itself rolls forward on disp_{k-1} for free
        # (its sequential predecessor IS disp_{k-1} in the interleaved order).
        _rdp_map = getattr(state.solve, "realized_dispatch_parent", {}) or {}
        _is_dispatch_roll = solve in _rdp_map
        if _is_dispatch_roll:
            _provider_translators.translate_handoff_to_provider(
                prior_handoff, sub_solve_provider,
                fields=_provider_translators.PIN_FIELDS,
            )
            _rf_handoff = (
                state.handoffs.get(complete_solve[solve])
                if state.handoffs is not None else None
            )
            _provider_translators.translate_handoff_to_provider(
                _rf_handoff, sub_solve_provider,
                fields=_provider_translators.ROLL_FORWARD_FIELDS,
            )
        else:
            _provider_translators.translate_handoff_to_provider(
                prior_handoff, sub_solve_provider,
            )
        # When nested, the parent solve's handoff supplies what the parent imposes on this sub-solve
        # (PIN_FIELDS: realised investment / existing / divest, fixed online, the nested storage fix): a partial
        # translate that lands after the sequential one.  The roll-forward carriers are not the parent's (a storage
        # or investment parent realises no dispatch, so its frames are empty or stale): they are routed below.
        parent_handoff = (
            state.handoffs.get(_parent_complete_for_carriers)
            if state.handoffs is not None and _parent_complete_for_carriers is not None
            else None
        )
        if parent_handoff is not None and not _is_dispatch_roll:
            _provider_translators.translate_handoff_to_provider(
                parent_handoff, sub_solve_provider,
                fields=_provider_translators.PIN_FIELDS,
            )
        # The roll-forward carriers tied to the realised end are routed, not taken from the sequential / parent
        # handoff: the latest state realised before this sub-solve at its nesting level or deeper (_carry_routing;
        # none → the sub-solve is the first of its level: storage starts from fix_start, units online start free
        # and ramps have no row at the first step).  Both storage keys hold the same frame; the override translate below still wins.  The
        # running totals (cumulative_*) stay with the sequential predecessor, which passes them on.
        carry_router.start(solve)
        _first_step = _carry_routing.first_step(active_time_lists[solve])
        # A nested level starts the nodes its upper solve fixes from the upper plan (borrowed end ⇒ borrowed start):
        # the shallower-state warning of the router does not name them.
        _upper_fixed = set()
        if parent_handoff is not None:
            _upper_fixed = {str(n) for _f in ("fix_storage_quantity", "fix_storage_usage", "fix_storage_price")
                            for n in (_frame_nodes(getattr(parent_handoff, _f, None)))}
        _routed_state = carry_router.roll_end_state(solve, _first_step, quiet=_upper_fixed)
        _is_rp_solve = any(ts in state.timeline.rp_weights for ts in active_timeset_names)
        if state.solve.solve_modes.get(complete_solve[solve]) == "rolling_window" and not _is_rp_solve:
            _restore_methods(_rolling_storage, _level_key, sub_solve_provider)
        _routed_state = _borrow_upper_start(
            state=state, solve=solve, complete_solve_name=complete_solve[solve], parent=parent_roll.get(solve),
            parent_handoff=parent_handoff, provider=sub_solve_provider, routed_state=_routed_state,
            active_time_lists=active_time_lists, time_index=_time_index, model_start=_model_start,
            is_rp=_is_rp_solve, store=_rolling_storage)
        # An upper-level fix restated at the horizon end: the level interpolated inside an upper step, the fix_usage
        # cap cumulative over the instance.
        _nested_rows = _nested_end_rows(
            solve=solve, parent=parent_roll.get(solve), parent_handoff=parent_handoff,
            active_time_lists=active_time_lists, time_index=_time_index, instance=_instance_of[solve],
            store=_rolling_storage, period_branch=period__branch_lists[solve],
            rolling=state.solve.solve_modes.get(complete_solve[solve]) == "rolling_window")
        # A later solve of a sequence that reaches the end of the model returns the free start the model began
        # forward chains at (the rolls of a rolling solve do so in _anchor_cyclic_in_rolling).
        if state.solve.solve_modes.get(complete_solve[solve]) != "rolling_window":
            _nested_rows += _sequence_match_rows(
                provider=sub_solve_provider, routed_state=_routed_state, store=_rolling_storage,
                at_model_end=storage_solve_at_model_end[i] or storage_rp_chain_at_model_end[i],
                upper_fixed=_upper_fixed, solve=complete_solve[solve], level=carry_router.nesting_level(solve))
        # Rolling bind_within_solve: carry from roll to roll, the horizon-end anchor and the closing rolls' closure.
        _anchor_cyclic_in_rolling(
            state=state, solve=solve, complete_solve_name=complete_solve[solve], instance=_instance_of[solve],
            level_key=_level_key, provider=sub_solve_provider, routed_state=_routed_state,
            closing=storage_solve_at_solve_end[i], instance_at_model_end=_instance_at_model_end[_instance_of[solve]],
            nested_fixed=(_frame_nodes(getattr(parent_handoff, "fix_storage_quantity", None))
                          | _frame_nodes(getattr(parent_handoff, "fix_storage_usage", None)))
            if parent_handoff is not None else set(),
            store=_rolling_storage, is_rp=_is_rp_solve, active_time_list=active_time_lists[solve],
            complete_active_time_list=complete_active_time_lists, period_branch=period__branch_lists[solve],
            extra_rows=_nested_rows, nested_any=_upper_fixed, level=carry_router.nesting_level(solve))
        for _key in (_provider_keys.HANDOFF_ROLL_END_STATE, _provider_keys.HANDOFF_UPWARD_ROLL_END_STATE):
            sub_solve_provider.put(
                _key, _routed_state if _routed_state is not None
                else _provider_translators._empty_handoff_frame(_key))
        _step_carriers = carry_router.step_carriers(solve, _first_step)
        _provider_translators.translate_handoff_to_provider(
            _carry_routing.RoutedCarriers(_step_carriers), sub_solve_provider,
            fields=frozenset(_carry_routing.STEP_FIELDS),
        )
        # The processes the routed carriers' producer left without online variables: a solve that keeps online
        # variables starts the ramp and the commitment of those units cold (preprocess._reduce_detail).
        state.current_carry_online_left_out = _step_carriers["online_left_out"]
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
                    "prep: _solve_time.run (per-solve sets + params dispatcher) done")
        _warn_rolling_blocks_chained(state, complete_solve[solve], solve, active_time_lists[solve],
                                     complete_active_time_lists, sub_solve_provider, _warned_rolling_blocks)
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

        # Tell the solver to emit the four per-iter phase checkpoints
        # (FlexData built / Matrix built / Solver / Outputs written) on
        # EVERY iter, so each roll/solve reports its own deltas rather
        # than aggregating across the whole level group.  This makes the
        # per-roll time/memory breakdown visible between solves (the
        # ``Solve cleanup`` snapshot above closes each boundary).
        state.emit_phase_checkpoints_this_iter = True
        # Phase 2 — expose the current iter's position in the level_keys
        # sequence so the per-solve callback can decide whether to retain
        # or drop this level's ``Solution.highs`` / ``flex_data_provider``
        # after the per-iter writers run.
        state._current_iter_index = i
        state._current_level_key = _level_key
        # A committed window reaching into a forecasts-only fan must be realized by some solve when a later solve
        # continues from it (checked on the built FlexData, before the LP).
        state.current_committed_window_check = (solve, realization_plans.get(solve), i < len(all_solves) - 1)
        _run_record = run_record.active()
        _run_record.solve_started(solve, complete_solve[solve], i, _log_anchor)
        exit_status = solver.run(complete_solve[solve])
        _run_record.solve_finished(exit_status)
        state.current_committed_window_check = None
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
                    with open(mod_phases_path, encoding="utf-8") as _mp:
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
            carry_router.record(solve, state.handoffs.get(last_captured_solve))
            _record_roll_starts(_rolling_storage, solve, state.handoffs.get(last_captured_solve))
            _record_scope_starts(_rolling_storage, solve, state.handoffs.get(last_captured_solve),
                                 period__branch_lists[solve])
            _record_usage(_rolling_storage, _instance_of[solve], state.handoffs.get(last_captured_solve),
                          period__branch_lists[solve])
            if _first_step is not None and _time_index.position(*_first_step) == _model_start:
                _record_model_start(_rolling_storage, state.handoffs.get(last_captured_solve), _routed_state,
                                    level=carry_router.nesting_level(solve))
            _check_borrowed_start(_rolling_storage, solve, complete_solve[solve],
                                  state.handoffs.get(last_captured_solve))
            # The solve's own (produced) handoff goes to the ``produced_handoff/*`` keys: the ``handoff/*`` keys
            # keep the carriers this solve consumed, so a LP rebuilt from the post-solve Provider (or from its csv
            # dump) is the LP that was solved — not one that starts from the solve's own end state.  The dump still
            # shows the final cumulative totals of the roll (``produced_handoff/cumulative_commodity.csv``).  The
            # next iteration's translator reads ``state.handoffs`` directly.
            _latest_handoff = state.handoffs.get(last_captured_solve)
            if _latest_handoff is not None:
                _provider_translators.translate_produced_handoff_to_provider(
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
