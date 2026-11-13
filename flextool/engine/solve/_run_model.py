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
from flextool.engine.preprocess import _provider_keys
from flextool.engine.preprocess import _provider_translators

from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
from flextool.engine.preprocess.params._run_calendar import RUN_CALENDAR_KEY, build_run_calendar
from flextool.engine.solve._realized_overlap import (
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
from flextool.engine.solve._stochastic import StochasticSolver
from flextool.engine.solve import _carry_routing
from flextool.engine.solve import _realization
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


def _roll_period_whole(active_time_list: dict, complete_active_time_list: dict,
                       period_branch: list) -> "pl.DataFrame":
    """``(period, whole)`` of a rolling_window sub-solve: True where the roll holds every step that the complete
    solve has in the period (a forecast branch period is compared with the steps of the period it branches from).

    The ``bind_intraperiod_blocks`` blocks of a whole period loop within the roll, as in a solve that is not rolling;
    the blocks of a period the roll holds in part chain forward and carry the storage level from roll to roll
    (owner decision Q-c15e-A).
    """
    anchor = {b: p for p, b in period_branch if b != p}
    rows: list[tuple[str, bool]] = []
    for period, steps in active_time_list.items():
        full = complete_active_time_list.get(anchor.get(period, period)) or []
        rows.append((str(period), bool(steps) and [s.timestep for s in steps] == [s.timestep for s in full]))
    return pl.DataFrame(rows, schema={"period": pl.Utf8, "whole": pl.Boolean}, orient="row")


def _warn_rolling_blocks_chained(state, complete_solve_name: str, solve: str, active_time_list: dict,
                                 complete_active_time_list: dict, provider, warned: set) -> None:
    """Warn once per rolling_window solve whose rolls start inside a period while it has ``bind_intraperiod_blocks``
    storage nodes (its rolling_solve_jump is not a whole number of periods).

    Such a period is held in part by the rolls around its inside start, so its blocks chain forward and carry the
    storage level from roll to roll instead of looping as in a solve that is not rolling.  ``solve_data/
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
        "starts at step '%s' inside period '%s'). The bind_intraperiod_blocks storage node(s) %s loop their timeset "
        "blocks only in the periods a roll holds whole; in a period a roll holds in part the blocks chain forward "
        "and the storage level is carried from roll to roll.",
        complete_solve_name, solve, steps[0].timestep, first_period, ", ".join(names))


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
    stochastic solve are rp periods with the timeset of the period they branch from.  A rolling_window solve never
    reaches this check with blended nodes: :func:`_downgrade_blended_for_rolling_rp` has made them
    bind_forward_only before (so in a rolling solve that downgrade pre-empts this raise).
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


# A rolling_window solve on representative periods: every *_blended_weights method binds forward only (owner
# decision, item 126): the base-period chain spans the whole period and cannot be cut into rolls, so the storage
# state is passed directly from roll to roll.  Used by :func:`_downgrade_blended_for_rolling_rp`.
_RP_ROLLING_DOWNGRADE: dict[str, str] = {
    "bind_within_solve_blended_weights":  "bind_forward_only",
    "bind_within_period_blended_weights": "bind_forward_only",
    "bind_forward_only_blended_weights":  "bind_forward_only",
}


def _downgrade_blended_for_rolling_rp(*, state, complete_solve_name: str, active_timeset_names: list, provider,
                                      warned: set) -> None:
    """Map every ``*_blended_weights`` storage_binding_method of a rolling_window solve on representative periods
    to bind_forward_only (see :data:`_RP_ROLLING_DOWNGRADE`), with one warning per complete solve.

    Rewrites the per-solve ``input/node__storage_binding_method`` frame in place, like
    :func:`_downgrade_rp_methods_for_non_rp_solve` (which handles a solve without representative periods; this one
    runs after it and before :func:`_fix_start_end_chains_forward`).  The downgraded nodes chain v_state forward
    and carry it from roll to roll (roll_continue), like any bind_forward_only node.  Runs before
    :func:`_check_rp_solve`, so a rolling solve that mixes periods with and without representative periods does not
    raise for these nodes.  Routed through the module logger: ``state.logger`` is silenced during the cascade.
    """
    if state.solve.solve_modes.get(complete_solve_name) != "rolling_window":
        return
    if not any(ts in state.timeline.rp_weights for ts in active_timeset_names):
        return
    key = "input/node__storage_binding_method"
    if not provider.has(key):
        return
    sbm = provider.get(key)
    if sbm.height == 0:
        return
    col = "storage_binding_method" if "storage_binding_method" in sbm.columns else "method"
    if col not in sbm.columns:
        return  # malformed frame; let the downstream loaders surface it
    hit = sbm.filter(pl.col(col).is_in(list(_RP_ROLLING_DOWNGRADE)))
    if hit.height == 0:
        return
    provider.put(key, sbm.with_columns(
        pl.col(col).replace_strict(_RP_ROLLING_DOWNGRADE, default=pl.col(col)).alias(col)))
    if complete_solve_name in warned:
        return
    warned.add(complete_solve_name)
    names = sorted(str(n) for n in hit["node"].unique().to_list())
    _MODULE_LOGGER.warning(
        "Solve '%s' is a rolling_window solve on representative periods and node(s) %s use a *_blended_weights "
        "storage_binding_method; storage_binding_method is not applied in a rolling solve: they bind forward only "
        "and the storage state is passed directly from roll to roll.",
        complete_solve_name, ", ".join(names))


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


def _fix_start_end_chains_forward(*, solve: str, provider, logger, complete_solve_name: str | None = None,
                                  logged: set | None = None) -> None:
    """Map the cyclic storage_binding_method of fix_start_end nodes to its forward-only equivalent, per solve.

    With ``storage_start_end_method`` *fix_start_end* the binding method is ignored (exception bind_forward_only,
    reference/nodes.md): the start and the end are pinned separately, so the chain must not close on itself (a cycle's
    start is its end, and with storage_state_start ≠ storage_state_end the two pins would conflict).  Rewrites the
    per-solve ``input/node__storage_binding_method`` frame in place (see :data:`_FIX_START_END_FORWARD`), like
    :func:`_downgrade_rp_methods_for_non_rp_solve`, after it (a blended node of a solve without representative
    periods: blended → plain → forward-only).  Every ``storage_bind_*`` set of the solve comes from that frame.
    ``bind_intraperiod_blocks`` stays: it is a time aggregation, not a cycle scope; such a node keeps its blocks and
    the model chains them forward (``model._self_discharge.block_forward_nodes``).  One info line per mapping (the
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
    hit = (sbm.filter(pl.col("node").cast(pl.Utf8).is_in(nodes) & pl.col(sbm_col).is_in(list(_FIX_START_END_FORWARD)))
           .select(pl.col("node").cast(pl.Utf8), pl.col(sbm_col).cast(pl.Utf8)).unique().sort("node"))
    if hit.height > 0:
        is_hit = pl.col("node").cast(pl.Utf8).is_in(hit["node"].to_list())
        provider.put(sbm_key, sbm.with_columns(
            pl.when(is_hit)
            .then(pl.col(sbm_col).replace_strict(_FIX_START_END_FORWARD, default=pl.col(sbm_col)))
            .otherwise(pl.col(sbm_col))
            .alias(sbm_col)))
        for node, method in hit.iter_rows():
            logger.info(f"Solve '{solve}': storage node '{node}' has storage_start_end_method fix_start_end, so its "
                        f"storage_binding_method {method} is ignored; it chains forward "
                        f"({_FIX_START_END_FORWARD[method]}).")
    blocks = (sbm.filter(pl.col("node").cast(pl.Utf8).is_in(nodes) & (pl.col(sbm_col) == "bind_intraperiod_blocks"))
              ["node"].cast(pl.Utf8).unique().sort().to_list())
    for node in blocks:
        if logged is not None:
            key = (complete_solve_name or solve, node)
            if key in logged:
                continue
            logged.add(key)
        logger.info(f"Solve '{solve}': storage node '{node}' has storage_start_end_method fix_start_end, so its "
                    "bind_intraperiod_blocks blocks chain forward instead of looping within each period.")


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
            if _sib_rolls and _par_rolls:
                for _idx, _uc in _par_rolls.items():
                    _dp = _sib_rolls.get(_idx)
                    if _dp is not None:
                        _uc_to_disp[_uc] = _dp
                        _rdp[_dp] = _uc           # per-roll parent map
                        # OC1 fix — the realized-dispatch roll realizes ONLY
                        # its JUMP (the committed/realized section).  Restrict
                        # its SOLVE window to the realized (jump) time list
                        # instead of inheriting the parent's full roll horizon
                        # (jump + look-ahead): the look-ahead has no realized
                        # data and no online pin, so solving it forced
                        # spurious unserved load.  ``realized_time_lists[_dp]``
                        # here is the start→next-jump slice (untrimmed at this
                        # point); copy it as the active window.  The fix_online
                        # pin and the disp→disp realized carry already align on
                        # the jump.
                        if _dp in realized_time_lists:
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
    # The reported costs of a run of several solves count from the start of the run, each once.
    if len(state.solve.real_solves) > 1:
        from flextool.engine import _behaviour_changes
        _behaviour_changes.notice("chained_solve_costs_from_run_start", "solves", state.solve.real_solves)

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

    # ------------------------------------------------------------------
    # 3.1 Realization plan per sub-solve.  A period label has no dispatch, so
    # the cells a sub-solve commits inside its fan (its pre-stochastic
    # realized and fix-storage cells of the labels) are kept as
    # ``committed_in_fan``: the non-anticipativity window of an unset
    # ``non_anticipativity_periods`` and the trigger of the committed-window
    # check.  The labels' pre-stochastic steps are kept for a curated window.
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
    for sibling, parent in realized_dispatch_parent.items():
        if sibling not in solve_branch__time_branch_lists:
            continue
        if (
            parent not in solve_branch__time_branch_lists
            or parent not in period__branch_lists
        ):
            continue
        # Cut to the sibling's own periods: a single solve's realization solve
        # dispatches its realized periods only, a realization roll its jump
        # (item 255).  The NA-tied online / reserve pins need no cut: the
        # model pins only the sibling's declared cells (online_fix_eq,
        # reserve_fix_eq).
        _sibling_periods = set(active_time_lists.get(sibling, {}))
        solve_branch__time_branch_lists[sibling] = [
            row for row in project_realized_branch_time_branch_rows(
                solve_branch__time_branch_lists[parent],
                period__branch_lists[parent],
            )
            if row[0] in _sibling_periods
        ]

    # The fan start of each solve (``solve_data/branch_time_start``): a branch
    # period whose own first step authors no stochastic-time value for a cell
    # reads the forecast authored at the fan start.  A realization-dispatch
    # sibling reads the parent's realized branch, so it takes the parent's
    # rows of the real period names.
    branch_time_start_lists: dict[str, list[tuple[str, str]]] = {
        s: solve_writers.branch_time_start_rows(
            period__branch_lists[s], branch_start_time_lists.get(s))
        for s in period__branch_lists
    }
    for sibling, parent in realized_dispatch_parent.items():
        if sibling not in branch_time_start_lists or parent not in branch_time_start_lists:
            continue
        _sibling_periods = set(active_time_lists.get(sibling, {}))
        branch_time_start_lists[sibling] = [
            row for row in solve_writers.project_branch_time_start_rows(
                branch_time_start_lists[parent], period__branch_lists[parent])
            if row[0] in _sibling_periods
        ]

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
    _warned_rolling_rp_blended: set = set()

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
        # Whether this sub-solve leaves the period_total user constraints out (a roll of a rolling solve that
        # does not realize whole periods in one roll); written for every sub-solve, the Provider is shared.
        sub_solve_provider.put(_provider_keys.SOLVE_DATA_USER_CONSTRAINT_PERIOD_TOTAL_OFF,
                               pl.DataFrame({"value": [solve in state.solve.period_total_off_solves]},
                                            schema={"value": pl.Boolean}))
        sub_solve_provider.put(_provider_keys.SOLVE_DATA_STORAGE_SOLVE_AT_MODEL_END,
                               pl.DataFrame({"value": [storage_solve_at_model_end[i]]}, schema={"value": pl.Boolean}))
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
        # Slice D — emit the RESOLVED stochastic_invest_method so the
        # derived/model layers can gate the recourse feature through the
        # provider/workdir (they hold no SolveConfig).  Absent CSV ==
        # flag-off (byte-parity-safe default).
        solve_writers.emit_stochastic_invest_method(
            state.solve.stochastic_invest_method_for(complete_solve[solve]),
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
        # Every period from the fan onwards reads the forecast made at the
        # fan's analysis time, in a single solve and in a roll alike, never a
        # forecast made later (``_pdt_lookup`` "Which forecast a branch
        # period reads").
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
                wf / "solve_data/timeline_matching_map.csv", "w", newline=""
            ) as realfile:
                writer = csv.writer(realfile)
                writer.writerow(["period", "step", "upper_step"])
                for (period, step), upper_step in matching_map.items():
                    writer.writerow([period, step, upper_step])
                    matching_rows.append((str(period), str(step), str(upper_step)))
        else:
            with open(wf / "solve_data/timeline_matching_map.csv", "w") as realfile:
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
        # A rolling solve on representative periods: blended nodes bind forward only (item 126), with a warning.
        _downgrade_blended_for_rolling_rp(state=state, complete_solve_name=complete_solve[solve],
                                          active_timeset_names=active_timeset_names,
                                          provider=sub_solve_provider, warned=_warned_rolling_rp_blended)
        # fix_start_end ignores a cyclic binding method: after the downgrades, on the same frame.
        _fix_start_end_chains_forward(solve=solve, provider=sub_solve_provider, logger=state.logger,
                                      complete_solve_name=complete_solve[solve], logged=_logged_fix_start_end_blocks)
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
        # none → the sub-solve is the first of its level: storage starts from fix_start, online and ramps from the
        # solve's own wrap).  Both storage keys hold the same frame; the override translate below still wins.  The
        # running totals (cumulative_*) stay with the sequential predecessor, which passes them on.
        carry_router.start(solve)
        _first_step = _carry_routing.first_step(active_time_lists[solve])
        _routed_state = carry_router.roll_end_state(solve, _first_step)
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
            carry_router.record(solve, state.handoffs.get(last_captured_solve))
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
