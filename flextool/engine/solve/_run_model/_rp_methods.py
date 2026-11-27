"""Storage-binding methods on representative periods: the checks of a representative-period solve, the
across-solve representative-period chain, the downgrade on solves without representative periods and the
forward chain of fix_start_end nodes."""
from __future__ import annotations

import polars as pl
from flextool.engine.input._solve_state import FlexToolConfigError

from flextool.engine.solve._run_model._storage_frames import _SBM_KEY, _SSE_KEY, _node_methods


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
