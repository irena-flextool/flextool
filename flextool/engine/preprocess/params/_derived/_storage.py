"""§3.9.2, §3.9.5–§3.9.8 — storage: ``nodeStateBlock``, the state capacity and upper bound, the storage use
reference value, the rolling-handoff storage state, timeline matching, ``period_branch`` and
:func:`apply_derived_e` (Γ.3.E)."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

from flextool.engine.input._axis_enums import alias_to_axis, rename_to_axis, schema_dtype

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource
    from flextool.engine.preprocess._solve_context import SolveContext

from flextool.engine.preprocess.params._derived._scope import (
    _enums,
    _provider_has_key,
    _provider_read,
    _read_active_solve,
    _try_entities,
    _try_param,
)
from flextool.engine.preprocess.params._derived._unitsize import UNITSIZE_DEFAULT, _node_unitsize_lf
from flextool.engine.preprocess.params._derived._periods import _expand_branch_periods, _period_in_use_set
from flextool.engine.preprocess.params._derived._existing_capacity import p_entity_all_existing_from_source
from flextool.engine.preprocess.params._derived._lookbacks import apply_mintime_lookbacks
from flextool.engine.preprocess.params._derived._dtttdt import (
    _roll_first_step_link,
    _rolling_realized_steps,
    _rp_block_first_utf8,
    dtttdt_block_interior_from_dtttdt,
    dtttdt_block_lag_from_dtttdt,
    dtttdt_forward_only_from_dtttdt,
    dtttdt_with_jump_from_source,
    dtttdt_within_solve_from_dtttdt,
    period_block_family_from_source,
    read_period_first_of_solve,
)


# ---------------------------------------------------------------------------
# §3.9.2 — nodeStateBlock (bind_intraperiod_blocks nodes)
# ---------------------------------------------------------------------------


def _node_storage_binding_method_with_fallback(source: "InputSource"
                                                       ) -> pl.DataFrame | None:
    """Per-node storage binding method with the
    ``method_with_fallback_sets.write_node_storage_binding_method``
    fallback rule applied.

    Mirrors flextool's preprocessing: explicit rows are kept verbatim;
    every node lacking an explicit row gets the default
    ``bind_forward_only``.  Returns ``[n, method]`` frame.
    """
    nodes = _try_entities(source, "node")
    if nodes is None or nodes.height == 0:
        return None
    explicit = _try_param(source, "node", "storage_binding_method")
    if explicit is None:
        explicit_rows = pl.DataFrame(schema={"n": schema_dtype(_enums, "n"),
                                              "method": pl.Utf8})
    else:
        explicit_rows = (explicit
            .pipe(rename_to_axis, {"name": "n", "value": "method"})
            .select("n", "method"))
    explicit_n = set(explicit_rows["n"].to_list())
    fallback_n = [n for n in nodes["name"].to_list() if n not in explicit_n]
    if fallback_n:
        fb = (pl.DataFrame({"n": fallback_n,
                              "method": ["bind_forward_only"] * len(fallback_n)})
                .with_columns(pl.col("n").cast(schema_dtype(_enums, "n"),
                                                  strict=False)))
    else:
        fb = pl.DataFrame(
            schema={"n": schema_dtype(_enums, "n"),
                    "method": pl.Utf8})
    out = pl.concat([explicit_rows, fb]).unique().sort("n", "method")
    return out


def nodeStateBlock_from_source(source: "InputSource") -> pl.DataFrame | None:
    """The ``nodeStateBlock`` set: nodes with ``storage_binding_method == 'bind_intraperiod_blocks'``, balanced
    once per timeset block (``nodeBalanceBlock_eq``).  ``None`` when there are none.  (Nodes of a coarse
    new_stepduration group are ``nodeCoarse``, see ``params._coarse``.)"""
    sbm = _try_param(source, "node", "storage_binding_method")
    if sbm is None:
        return None
    intraperiod = (sbm.lazy()
                   .filter(pl.col("value") == "bind_intraperiod_blocks")
                   .select(alias_to_axis("name", "n"))
                   .collect())
    if intraperiod.height == 0:
        return None
    return pl.DataFrame({"n": sorted(set(intraperiod["n"].to_list()))})


# ---------------------------------------------------------------------------
# §3.9.5 — p_state_existing_capacity / p_state_upper
# ---------------------------------------------------------------------------


def p_state_existing_capacity_from_source(source: "InputSource",
                                                active_solve: str | None,
                                                nodeState_df: pl.DataFrame
                                                  | None,
                                                workdir: Path | None = None,
                                                *,
                                                ctx: "SolveContext | None"
                                                  = None,
                                                provider: "object | None" = None,
                                                ) -> "Param | None":
    """``p_state_existing_capacity`` per (n, d) — node existing capacity
    restricted to nodes carrying state (``nodeState`` set).

    Algorithm: take ``p_entity_all_existing`` for nodes only, restrict to
    nodes in ``nodeState_df``.  Falls back to the lifetime-gated existing
    capacity (existing × virtual_unitsize when it is set) of the state
    nodes when the chained Param isn't available.  Broadcasts onto the full
    period set including stochastic-branch siblings (workdir-aware).
    """
    if nodeState_df is None or nodeState_df.height == 0:
        return None
    state_n = set(nodeState_df["n"].to_list())
    # Full period set (realized + invest + stochastic-branch siblings).
    piu_base = _period_in_use_set(source, active_solve, ctx=ctx, provider=provider) if active_solve \
                else []
    full_periods = _expand_branch_periods(piu_base, workdir, ctx=ctx, provider=provider)
    pae = p_entity_all_existing_from_source(source, active_solve, workdir, ctx=ctx, provider=provider)
    if pae is not None and pae.frame.height > 0:
        base = (pae.frame.lazy()
                  .pipe(rename_to_axis, {"e": "n"})
                  .filter(pl.col("n").is_in(list(state_n)))
                  .select("n", "d", "value")
                  .collect())
        # Broadcast each (n, d) to its branch siblings sharing the
        # anchor's existing value.
        if full_periods and len(full_periods) > base["d"].n_unique():
            anchor_branches = _expand_branch_periods(
                base["d"].unique().to_list(), workdir, ctx=ctx, provider=provider)
            # Build (anchor → list of d) mapping
            extra_rows: list[tuple[str, str, float]] = []
            covered = set(base["d"].to_list())
            for d in anchor_branches:
                if d in covered:
                    continue
                # Find anchor whose row to copy: any anchor it expanded
                # from.  Walk period__branch.csv to find anchor mapping.
                anchor_for_d = _branch_anchor(d, workdir, ctx=ctx, provider=provider)
                if anchor_for_d is None:
                    continue
                rows_for_anchor = base.filter(
                    pl.col("d") == anchor_for_d)
                for r in rows_for_anchor.iter_rows(named=True):
                    extra_rows.append((r["n"], d, r["value"]))
            if extra_rows:
                add = pl.DataFrame(
                    extra_rows, schema=["n", "d", "value"], orient="row")
                base = pl.concat([base, add])
        df = base.sort("n", "d")
        if df.height > 0:
            return Param(("n", "d"), df)
    # Fallback (no period_in_use for p_entity_all_existing): the lifetime-gated existing capacity
    # (existing × virtual_unitsize when it is set) of the state nodes with an explicit existing, over the full
    # period set or, without one, the periods of an existing period map.
    ex = _try_param(source, "node", "existing")
    if ex is None or ex.height == 0:
        return None
    ex_nodes = set(ex["name"].cast(pl.Utf8).to_list()) & state_n
    periods = full_periods
    if not periods and "period" in ex.columns:
        periods = sorted({str(d) for d in ex["period"].drop_nulls().to_list()})
    if not ex_nodes or not periods:
        return None
    from flextool.engine.preprocess.params._existing import p_entity_pre_existing_lf
    df = (p_entity_pre_existing_lf(source, active_solve, periods, workdir=workdir, provider=provider)
            .filter(pl.col("e").cast(pl.Utf8).is_in(sorted(ex_nodes)))
            .pipe(rename_to_axis, {"e": "n"})
            .select("n", "d", "value")
            .sort("n", "d")
            .collect())
    if df.height == 0:
        return None
    return Param(("n", "d"), df)


def _branch_anchor(branch: str, workdir: Path | None,
                      *, ctx: "SolveContext | None" = None,
                      provider: "object | None" = None,
                      ) -> str | None:
    """Return the anchor period whose ``period__branch.csv`` row maps
    to *branch*.  None if no mapping found.

    Path B Cat B: when ``ctx`` is provided, the in-memory
    :pyattr:`SolveContext.period_branch` frame is consulted first.
    Falls back to the CSV-read path when ``ctx`` is None or the typed
    accessor is empty.
    """
    if ctx is not None:
        pb = ctx.period_branch
        if pb.height > 0:
            cand = pb.filter(pl.col("b") == branch)
            if cand.height > 0:
                return cand["d_anchor"][0]
            # Typed accessor is authoritative when populated; don't fall
            # through to the CSV path even if branch isn't in it.
            return None
    if workdir is None:
        return None
    p = Path(workdir) / "solve_data" / "period__branch.csv"
    if not _provider_has_key(provider, p):  # Phase E-h — seed-aware
        return None
    try:
        df = _provider_read(provider, p)
    except Exception:
        return None
    if df.height == 0:
        return None
    cand = df.filter(pl.col("branch") == branch)
    if cand.height > 0:
        return cand["period"][0]
    return None


def p_state_upper_from_source(source: "InputSource",
                                  state_existing_capacity: "Param | None",
                                  nodeState_df: pl.DataFrame | None,
                                  ) -> "Param | None":
    """``p_state_upper[n, d] = state_existing_capacity[n, d] /
    p_entity_unitsize[n]``, with the unitsize fallback chain
    ``virtual_unitsize OR existing OR 1000``.  Mirrors
    ``entity_period_calc_params.py:186-202`` and ``flex_data/_load_storage.py::_load_storage``.
    """
    if (state_existing_capacity is None
            or state_existing_capacity.frame.height == 0
            or nodeState_df is None or nodeState_df.height == 0):
        return None
    cap = state_existing_capacity.frame
    us_lf = _node_unitsize_lf(source)
    df = (cap.lazy()
              .join(us_lf, on="n", how="left")
              .with_columns(us=pl.col("us").fill_null(UNITSIZE_DEFAULT))
              .with_columns(value=pl.col("value").cast(pl.Float64)
                                                   / pl.col("us"))
              .select("n", "d", "value")
              .sort("n", "d")
              .collect())
    if df.height == 0:
        return None
    return Param(("n", "d"), df)


# ---------------------------------------------------------------------------
# §3.9.6 — storage_use_reference_value (multi-method exclusion chain)
# ---------------------------------------------------------------------------


def storage_use_reference_value_from_source(source: "InputSource",
                                                  nodeStateBlock_df: pl.DataFrame
                                                    | None,
                                                  *,
                                                  rolling: bool = False,
                                                  ) -> pl.DataFrame | None:
    """``storage_use_reference_value`` — nodes with horizon method
    ``use_reference_value`` after multi-method exclusion.

    Mirror of ``flex_data/_load_storage.py::_load_storage``.  Filter
    ``node.storage_solve_horizon_method == 'use_reference_value'``, then
    anti-join against:

      * ``storage_start_end_method ∈ {fix_end, fix_start_end}``
      * ``storage_binding_method ∈ {bind_within_solve, bind_within_period,
                                      bind_within_timeblock, bind_intraperiod_blocks}``
        (the end state is tied to the start); the caller also drops the
        nodes of the solve's own plain cyclic sets, which hold the
        blended-weights nodes a non-RP solve downgrades to those methods
      * ``nodeStateBlock`` (synthesised + explicit).

    With *rolling* (a rolling_window sub-solve) the bind_intraperiod_blocks nodes keep the pin: it applies where the
    roll's last period chains forward (``model.core._storage_state._eq_storage_reference_value``).
    """
    sshm = _try_param(source, "node", "storage_solve_horizon_method")
    if sshm is None:
        return None
    base = (sshm.lazy()
                 .filter(pl.col("value") == "use_reference_value")
                 .select(alias_to_axis("name", "n"))
                 .unique()
                 .collect())
    if base.height == 0:
        return None
    # Exclusion sets.
    sse = _try_param(source, "node", "storage_start_end_method")
    excl_lists: list[list[str]] = []
    if sse is not None:
        e = (sse.lazy()
                 .filter(pl.col("value").is_in(["fix_end", "fix_start_end"]))
                 .select(alias_to_axis("name", "n"))
                 .collect())
        if e.height > 0:
            excl_lists.append(e["n"].to_list())
    sbm_full = _node_storage_binding_method_with_fallback(source)
    if sbm_full is not None:
        b = (sbm_full.lazy()
                  .filter(pl.col("method").is_in([
                      "bind_within_solve", "bind_within_period", "bind_within_timeblock"]
                      + ([] if rolling else ["bind_intraperiod_blocks"])))
                  .select("n")
                  .collect())
        if b.height > 0:
            excl_lists.append(b["n"].to_list())
    if not rolling and nodeStateBlock_df is not None and nodeStateBlock_df.height > 0:
        excl_lists.append(nodeStateBlock_df["n"].to_list())
    excl = set()
    for lst in excl_lists:
        excl.update(lst)
    out = base.filter(~pl.col("n").is_in(list(excl)))
    if out.height == 0:
        return None
    return out.unique().sort("n")


# ---------------------------------------------------------------------------
# §3.9.7 — p_roll_continue_state / p_fix_storage_quantity (rolling handoff)
# ---------------------------------------------------------------------------


def p_roll_continue_state_from_handoff(
    *,
    provider: "object | None" = None,
) -> "Param | None":
    """Read the rolling-handoff ``p_roll_continue_state`` from the
    canonical ``handoff/roll_end_state`` Provider key
    (schema ``[node, value]``).

    Phase 4.2-1b — the legacy ``solve_data/p_roll_continue_state``
    CSV read path is gone; the typed handoff translator pipeline
    (extended in Phase 4.2-0) populates the handoff key from
    :pyattr:`SolveHandoff.roll_end_state` at iteration start.
    """
    from flextool.engine.preprocess import _provider_keys as K
    from flextool.engine.preprocess._provider_translators import read_handoff_frame
    df = read_handoff_frame(provider, K.HANDOFF_ROLL_END_STATE)
    if df is None or df.height == 0:
        return None
    df = (df.pipe(rename_to_axis, {"node": "n"})
            .with_columns(value=pl.col("value")
                                    .cast(pl.Float64, strict=False)
                                    .fill_null(0.0))
            .select("n", "value"))
    if df.height == 0:
        return None
    return Param(("n",), df)


def p_fix_storage_quantity_from_handoff(
    *,
    provider: "object | None" = None,
) -> "Param | None":
    """Read the rolling-handoff ``p_fix_storage_quantity`` from the
    canonical ``handoff/fix_storage_quantity`` Provider key
    (schema ``[node, period, step, p_fix_storage_quantity]``).

    Phase 4.1d — the legacy ``solve_data/fix_storage_quantity`` read
    path is gone; the translator populates the handoff key from
    :pyattr:`SolveHandoff.fix_storage_quantity` at iteration start.
    """
    from flextool.engine.preprocess import _provider_keys as K
    from flextool.engine.preprocess._provider_translators import read_handoff_frame
    df = read_handoff_frame(provider, K.HANDOFF_FIX_STORAGE_QUANTITY)
    if df is None or df.height == 0:
        return None
    df = (df.pipe(rename_to_axis, {"period": "d", "step": "t", "node": "n",
                       "p_fix_storage_quantity": "value"})
            .with_columns(value=pl.col("value").cast(pl.Float64))
            .select("n", "d", "t", "value"))
    if df.height == 0:
        return None
    return Param(("n", "d", "t"), df)


# ---------------------------------------------------------------------------
# §3.9.8 — dtt_timeline_matching / period_branch
# ---------------------------------------------------------------------------


def dtt_timeline_matching_from_workdir(workdir: Path | None,
                                              *,
                                              provider: "object | None" = None,
                                              ) -> pl.DataFrame | None:
    """Per-solve timeline matching map ``(d, t, t_upper)`` between a
    sub-solve's fine timesteps and the upper-level coarse timesteps.
    Read from ``solve_data/timeline_matching_map.csv``.
    """
    if workdir is None:
        return None
    p = Path(workdir) / "solve_data" / "timeline_matching_map.csv"
    if not _provider_has_key(provider, p):  # Phase E-h — seed-aware
        return None
    df = _provider_read(provider, p)
    if df.height == 0:
        return None
    return (df
            .pipe(rename_to_axis, {"period": "d", "step": "t", "upper_step": "t_upper"})
            .select("d", "t", "t_upper")
            .unique(maintain_order=True))


def period_branch_from_source(source: "InputSource",
                                workdir: Path | None,
                                *,
                                ctx: "SolveContext | None" = None,
                                provider: "object | None" = None,
                                ) -> pl.DataFrame | None:
    """``period_branch`` rolling-handoff helper — (d_upper, d) anchor →
    branch map per audit §3.9.8.

    Spine source: ``period__branch`` 1-dim entity class is currently a
    flat name list with no values; the canonical per-solve mapping is
    written by ``solve_writers`` to ``solve_data/period__branch.csv``
    with columns ``(period, branch)``.  Read that file when present
    (workdir-aware path).

    Path B Cat B: when ``ctx`` is provided, the in-memory
    :pyattr:`SolveContext.period_branch` frame is consulted first
    (canonical schema ``[d_anchor, b]`` → remapped to the legacy
    ``[d_upper, d]`` shape this helper exposes).
    """
    if ctx is not None:
        pb = ctx.period_branch
        if pb.height > 0:
            # ctx.period_branch schema: [d_anchor, b]; downstream consumer
            # expects (d_upper, d) where d_upper == branch, d == anchor.
            return (pb
                    .pipe(rename_to_axis, {"d_anchor": "d", "b": "d_upper"})
                    .select("d_upper", "d")
                    .unique(maintain_order=True))
    if workdir is not None:
        p = Path(workdir) / "solve_data" / "period__branch.csv"
        if _provider_has_key(provider, p):  # Phase E-h — seed-aware
            df = _provider_read(provider, p)
            if df.height > 0:
                return (df
                        .pipe(rename_to_axis, {"period": "d", "branch": "d_upper"})
                        .select("d_upper", "d")
                        .unique(maintain_order=True))
    return None


# ---------------------------------------------------------------------------
# Public — Γ.3.E field list (selective scope-checking).
# ---------------------------------------------------------------------------


E_PUBLIC_FIELDS: tuple[str, ...] = (
    "dtttdt",
    "dtttdt_forward_only",
    "dtttdt_within_solve",
    "dtttdt_block_interior",
    "dtttdt_block_lag",
    "nodeStateBlock",
    "period_block",
    "period_block_succ",
    "period_block_time",
    "p_state_upper",
    "p_state_existing_capacity",
    "storage_use_reference_value",
    "p_roll_continue_state",
    "p_fix_storage_quantity",
    "dtt_timeline_matching",
    "period_branch",
)


def apply_derived_e(
    flex_data: object,
    source: "InputSource",
    workdir: Path,
    *,
    ctx: "SolveContext | None" = None,
    provider: "object | None" = None,
) -> None:
    """Apply Γ.3.E storage block algebra (§3.9), mutating ``flex_data``
    in place.

    Order — dependency-driven:
      1. dtttdt + auxiliary lag frames (§3.9.1).
      2. period_block / _succ / _time: the timeset blocks of bind_intraperiod_blocks nodes (§3.9.3).
      3. nodeStateBlock: the bind_intraperiod_blocks nodes (§3.9.2).
      4. nodeState_last_dt.
      5. p_state_existing_capacity / p_state_upper (§3.9.5).
      6. storage_use_reference_value (§3.9.6).
      7. p_roll_continue_state / p_fix_storage_quantity (§3.9.7).
      8. dtt_timeline_matching / period_branch (§3.9.8).

    The coarse new_stepduration groups are not handled here: ``load_flextool`` derives their frames for every
    solve (``params._coarse.coarse_node_frames``).
    """
    from flextool.engine.preprocess.params._block import load_block_bundle, nodeState_last_dt_lf

    active_solve = ctx.solve_name if ctx is not None else _read_active_solve(workdir, provider=provider)
    nodeState_df = getattr(flex_data, "nodeState", None)
    has_state = (nodeState_df is not None
                  and getattr(nodeState_df, "height", 0) > 0)
    # Δ.9 — single BlockBundle per solve (replaces 4× CSV reads).
    # No block frames (fixtures without explicit blocks) → ``None``; any
    # other failure (including a missing provider) propagates.
    # Phase 2 multi-block fast-path: thread the in-memory layout (built
    # source-only by the fast loader and stashed on ``FlexData``) so
    # ``load_block_bundle`` returns a populated bundle even when the
    # workdir's solve_data/ is empty.
    fd_layout = getattr(flex_data, "block_layout", None)
    block_bundle = load_block_bundle(
        workdir, block_layout=fd_layout, provider=provider,
    )

    # Δ.12b — assignment is unconditional for storage-only Params and
    # the rolling-handoff carriers.  ``None`` is the explicit
    # "feature inactive / no handoff state" signal.

    # 1. dtttdt -------------------------------------------------------
    # Δ.12c-fix2 gap #5 close — ``_dt_period_active_steps`` now falls
    # back to reading ``solve_data/steps_in_use.csv`` for synthetic
    # rolling-horizon / nested-invest sub-solves whose names aren't in
    # the Spine DB.  The fallback uses ``timeline.timestep_duration``
    # for rank/timeline assignment.  Conditional assignment is retained
    # because fixtures genuinely lacking timeline data on the source AND
    # the workdir CSV (very rare; not present in the engine
    # corpus) still need to fall through to the seed-loaded value.
    dtttdt_jump = dtttdt_with_jump_from_source(source, active_solve, workdir, ctx=ctx, provider=provider,
                                               step_duration=getattr(flex_data, "p_step_duration", None))
    dtttdt_db = None if dtttdt_jump is None else dtttdt_jump.drop("jump")
    if dtttdt_db is not None:
        # A split realization roll: the placeholders at its label's and branch periods' first steps are null
        # (before the 1b frames below are derived from dtttdt), as in the seed (flex_data._readers._read_step_previous).
        from flextool.engine.preprocess._step_relations import null_split_dtttdt
        from flextool.engine.preprocess.params._branch import period_branch_full_df
        split_pb = period_branch_full_df(workdir, provider=provider) if workdir is not None else None
        dtttdt_db = null_split_dtttdt(dtttdt_db, split_pb)
        flex_data.dtttdt = dtttdt_db
        # Ramp predecessors: the adjacent (jump == 1) rows, without representative-block starts; the online
        # stretch wrap is derived from them (ramp_link_fields sets both).
        from flextool.engine.preprocess.params._ramp_links import ramp_link_fields, ramp_link_from_jump
        rp_first = _rp_block_first_utf8(workdir, provider)
        for _name, _value in ramp_link_fields(
                flex_data.dt, ramp_link_from_jump(dtttdt_jump, rp_first), split_pb, rp_first,
                solve=active_solve).items():
            setattr(flex_data, _name, _value)
    # Minimum up/down-time look-backs along the final ramp predecessor links (the seed links when the source
    # timeline walk gives none).
    apply_mintime_lookbacks(flex_data, source, _roll_first_step_link(workdir, provider), active_solve)

    # 2. period_block family (storage-only). --------------------------
    pb_family = None
    if has_state:
        pb_family = period_block_family_from_source(
            source, active_solve, workdir, ctx=ctx, provider=provider,
            realized=_rolling_realized_steps(flex_data, workdir, provider))
    period_block_time_for_arc: pl.DataFrame | None = None
    if pb_family is not None:
        flex_data.period_block = pb_family["period_block"]
        flex_data.period_block_succ = pb_family["period_block_succ"]
        flex_data.period_block_time = pb_family["period_block_time"]
        period_block_time_for_arc = pb_family["period_block_time"]

    # 1b. dtttdt-derived lag frames (storage-only). ------------------
    if has_state:
        fwd = dtttdt_forward_only_from_dtttdt(
            dtttdt_db, read_period_first_of_solve(Path(workdir) / "solve_data", provider=provider)
            if workdir is not None else None)
        if fwd is not None:
            # The per-solve forward-only set counts too: it holds the fix_start_end nodes whose cyclic method the
            # solve ignores (_run_model._fix_start_end_chains_forward), which the DB does not call forward-only.
            sbm_full = _node_storage_binding_method_with_fallback(source)
            per_solve_fo = getattr(flex_data, "storage_bind_forward_only", None)
            if ((sbm_full is not None
                    and sbm_full.filter(
                        pl.col("method") == "bind_forward_only").height > 0)
                    or (per_solve_fo is not None and per_solve_fo.height > 0)):
                flex_data.dtttdt_forward_only = fwd
        # bind_within_solve: the first step of each root of a fan in the solve's first period continues from the
        # last step of its own path (the state-change lag of the bind_within_solve nodes).  This value is
        # authoritative: it overwrites the seed value of flex_data._load_storage._load_storage, which uses the same function and
        # must agree with it.
        ws_nodes = getattr(flex_data, "storage_bind_within_solve", None)
        flex_data.dtttdt_within_solve = (
            dtttdt_within_solve_from_dtttdt(
                getattr(flex_data, "dtttdt", None),
                read_period_first_of_solve(Path(workdir) / "solve_data", provider=provider)
                if workdir is not None else None)
            if ws_nodes is not None and ws_nodes.height > 0 else None)
        flex_data.dtttdt_block_interior = dtttdt_block_interior_from_dtttdt(
            dtttdt_db, period_block_time_for_arc)
        flex_data.dtttdt_block_lag = dtttdt_block_lag_from_dtttdt(
            getattr(flex_data, "dtttdt", None), period_block_time_for_arc,
            getattr(flex_data, "period_block_chain", None), getattr(flex_data, "period_branch_full", None))

    # 3. nodeStateBlock = the bind_intraperiod_blocks nodes (storage-only) -
    nsb_db = None
    if has_state:
        nsb_db = nodeStateBlock_from_source(source)
        flex_data.nodeStateBlock = nsb_db

    # 4. nodeState_last_dt: the default block's last step of each period (every node lives on the solve's
    # steps; a coarse node's state is constant within its last coarse step).
    if has_state and block_bundle is not None:
        nsld = nodeState_last_dt_lf(nodeState_df, block_bundle).collect()
        if nsld.height > 0:
            flex_data.nodeState_last_dt = nsld

    # 5. p_state_existing_capacity / p_state_upper --------------------
    if has_state:
        # A4-followup (BUG fix): pass ``provider`` so the helper can
        # read the workdir ``period__branch.csv`` / ``period_in_use_set.csv``
        # for stochastic-branch broadcast.  Without the provider the
        # Provider-only ``_provider_has_key`` returns False (Step 2.5
        # removed the disk-fallback), ``ctx`` is typically empty at this
        # cascade stage, and the helper falls through to the
        # ``solve.realized_periods`` set — which does NOT include
        # stochastic-branch siblings.  Result: ``p_state_existing_capacity``
        # carries only the realized-period row → downstream constraints
        # that join on (n, d) lose branch rows, e.g.
        # ``storage_state_solve_horizon_reference_value`` becomes an
        # empty Param for branches and the LP leaves their v_state at 0
        # while the realized branch is properly pinned.
        pse = p_state_existing_capacity_from_source(
            source, active_solve, nodeState_df, workdir,
            ctx=ctx, provider=provider)
        flex_data.p_state_existing_capacity = pse
        flex_data.p_state_upper = p_state_upper_from_source(
            source, pse, nodeState_df)
        # §F.2 p_state_unitsize (Δ.10 cluster F) — per-node unitsize
        # restricted to nodeState.
        from flextool.engine.preprocess.params._arithmetic import p_state_unitsize_from_source
        flex_data.p_state_unitsize = p_state_unitsize_from_source(
            source, nodeState_df)

    # 6. storage_use_reference_value (storage-only) -------------------
    if has_state:
        use_ref = storage_use_reference_value_from_source(
            source, nsb_db, rolling=bool(getattr(flex_data, "solve_is_rolling", False)))
        # The solve's plain cyclic sets come from the per-solve frame, so they include the blended-weights
        # nodes this solve downgraded (no representative periods): those cycle too and get no end pin.
        # A bind_within_solve node of a rolling roll that continues from a carried state chains forward in the roll
        # (_run_model._anchor_cyclic_in_rolling) and its end is tied to its start by storage_roll_end: no pin either.
        from flextool.engine.preprocess.flex_data._readers import _roll_end_tied_nodes
        for cyclic in (flex_data.storage_bind_within_solve, flex_data.storage_bind_within_period,
                       _roll_end_tied_nodes(getattr(flex_data, "storage_roll_end", None))):
            if use_ref is not None and cyclic is not None and cyclic.height > 0:
                names = [str(n) for n in cyclic["n"].to_list()]
                use_ref = use_ref.filter(~pl.col("n").cast(pl.Utf8).is_in(names))
        flex_data.storage_use_reference_value = use_ref if use_ref is not None and use_ref.height > 0 else None

    # 7. rolling-handoff (storage-only — read-from-disk handoff) ------
    # Path B Cat A: ``_load_storage`` (``flex_data/_load_storage.py``)
    # already reads ``p_roll_continue_state.csv`` and
    # ``fix_storage_quantity.csv`` and assigns the resulting Params to
    # ``flex_data.p_roll_continue_state`` / ``p_fix_storage_quantity``
    # at load time.  Re-reading the same CSV here duplicates the IO and
    # the polars allocation.  Audit `specs/in_cascade_csv_audit.md`
    # §Category A entries of the two re-reads (`p_roll_continue_state_from_handoff`, `p_fix_storage_quantity_from_handoff`).
    # Prefer the already-populated FlexData fields when available; fall
    # back to the workdir reader for legacy/test callers that bypass
    # ``_load_storage`` or when the seed produced None (e.g. CSV absent
    # at seed time but produced later by a prior solve).
    if has_state:
        if getattr(flex_data, "p_roll_continue_state", None) is None:
            flex_data.p_roll_continue_state = p_roll_continue_state_from_handoff(provider=provider)
        if getattr(flex_data, "p_fix_storage_quantity", None) is None:
            flex_data.p_fix_storage_quantity = p_fix_storage_quantity_from_handoff(provider=provider)
        flex_data.dtt_timeline_matching = dtt_timeline_matching_from_workdir(workdir, provider=provider)
        flex_data.period_branch = period_branch_from_source(source, workdir, ctx=ctx, provider=provider)
