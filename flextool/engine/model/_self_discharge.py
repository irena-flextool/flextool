"""Storage self-discharge: one construction of the loss for the balance rows and for the outputs.

  # ─── self-discharge (in nodeBalance_eq, nodeBalanceCoarse_eq, nodeBalanceBlock_eq) ──
  #   − (α[n,d,t]·v_state[n,d,t] + β[n,d,t]·S⁻[n,d,t])·state_unitsize[n]
  #   h = Δt·ln(1 + self_discharge_loss[n,d,t]),  α = h/(1 − e^{−h}) − 1,  β = 1 − h/(e^{h} − 1)
  #   S⁻ = the state the step starts from: v_state[n,d′,t⁻] by the binding method, v_state_rp_start at a
  #        representative block's first step, the fix_start start state / the roll carry at an open chain's first
  #        step, v_state[n,d,t] where the chain is open with neither (the step has no state change)
  #   coarse steps / blocks b: h_b = Σ_{t∈b} Δt·ln(1 + loss[n,d,t]), Δ_b = Σ_{t∈b} Δt; α spread over the
  #        fine steps by Δt/Δ_b, β at the first fine step only (the flow weighting assumes the loss constant in b)
  # The content decays continuously at ln(1 + loss) per hour and the flows of a step are spread evenly over the
  # step: with this loss the balance row gives S_t = e^{−h}·S⁻ + (1 − e^{−h})/h · E_t/su (E_t the net energy
  # into the storage in the step), exact for every modelled step whatever its length.  α, β ≥ 0, α + β = h.

The model folds the loss into each state-change term: ``(S⁻·(1 − β) − S·(1 + α))·su`` (``keep_prev`` /
``keep_now``), so the β weight is always on the state the step's state change starts from.  The output writer
(``process_outputs.read_highs_solution.write_v_self_discharge``) computes the same loss from the solve's own
solution with :func:`self_discharge_energy`, on the lags of :func:`state_links` — the lags the model's state
change uses (``model.core._state_change_terms``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from polar_high import Param

from flextool.engine.model._common import _has_rows

# Below this h the closed forms lose digits to cancellation; the series is used instead (truncation < 1e-19
# relative at h = 1e-2).
_SERIES_BELOW = 1e-2


def alpha_beta(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(α, β)`` of the decay ``h`` (≥ 0) of a step: α = h/(1 − e^{−h}) − 1, β = h − α."""
    h = np.asarray(h, dtype=float)
    alpha = np.zeros_like(h)
    small = h < _SERIES_BELOW
    hs = h[small]
    alpha[small] = hs / 2 + hs ** 2 / 12 - hs ** 4 / 720 + hs ** 6 / 30240
    hl = h[~small]
    alpha[~small] = hl / (-np.expm1(-hl)) - 1.0
    return alpha, h - alpha


@dataclass(frozen=True)
class LossCoefficients:
    """Self-discharge weights over (n, d, t) of every storage node (1 / 0 for nodes without a loss)."""
    frame: pl.DataFrame        # (n, d, t, alpha, beta)
    keep_now: Param            # 1 + α
    keep_prev: Param           # 1 − β
    h: Param                   # α + β: the loss of a step whose state is constant over it (open first step)


def _grouped_steps(d) -> "pl.DataFrame | None":
    """``(_n, _d, _t, _b)`` (Utf8) of the storage nodes balanced per coarse step or per timeset block."""
    parts = []
    coarse = getattr(d, "node_coarse_step", None)
    if _has_rows(coarse):
        parts.append(coarse.select(pl.col("n").cast(pl.Utf8).alias("_n"), pl.col("d").cast(pl.Utf8).alias("_d"),
                                   pl.col("t").cast(pl.Utf8).alias("_t"), pl.col("b_first").cast(pl.Utf8).alias("_b")))
    nsb, pbt = getattr(d, "nodeStateBlock", None), getattr(d, "period_block_time", None)
    if _has_rows(nsb) and _has_rows(pbt):
        parts.append(nsb.select(pl.col("n").cast(pl.Utf8).alias("_n"))
                     .join(pbt.select(pl.col("d").cast(pl.Utf8).alias("_d"), pl.col("t").cast(pl.Utf8).alias("_t"),
                                      pl.col("b_first").cast(pl.Utf8).alias("_b")), how="cross"))
    if not parts:
        return None
    return pl.concat(parts, how="vertical").unique(subset=["_n", "_d", "_t"], keep="first")


def _loss_frame(d) -> "pl.DataFrame | None":
    """``(_n, _d, _t, loss)`` (Utf8 keys) of ``self_discharge_loss`` — per (n, d, t), or ``(_n, loss)`` for a
    per-node constant (a Param over n only); None when absent.

    Raises FlexToolConfigError when a node has more than one value for a key: summed duplicates would scale the
    storage's whole state change (and not only the loss).
    """
    loss = d.p_state_self_discharge
    if loss is None or loss.frame.height == 0:
        return None
    frame = loss.frame
    keys = [c for c in ("n", "d", "t") if c in frame.columns]
    out = frame.select([pl.col(c).cast(pl.Utf8).alias(f"_{c}") for c in keys]
                       + [pl.col("value").cast(pl.Float64).alias("loss")])
    dup = out.group_by([f"_{c}" for c in keys]).len().filter(pl.col("len") > 1)
    if dup.height > 0:
        from flextool.engine.input._solve_state import FlexToolConfigError

        names = ", ".join(sorted({str(x) for x in dup["_n"].to_list()}))
        raise FlexToolConfigError(
            f"self_discharge_loss of node(s) {names} has more than one value for the same step "
            f"(keys {', '.join(keys)}); expected one value per node and step")
    return out


def loss_coefficients(d, state_dt: "pl.DataFrame | None") -> "LossCoefficients | None":
    """α, β over ``state_dt`` (the (n, d, t) of v_state); None when no storage node has a positive loss.

    λ[n,d,t] = ln(1 + self_discharge_loss[n,d,t]) (a per-node constant or a per-step value), h = Δt·λ.  A coarse
    step / block b has h_b = Σ_{t∈b} Δt·λ_t (exact for idle decay); its α_b is spread over the fine steps by
    Δt/Δ_b and β_b sits on the first fine step — the weighting of the flows assumes λ constant within b.
    """
    dur = d.p_step_duration
    if dur is None or not _has_rows(state_dt):
        return None
    loss_frame = _loss_frame(d)
    if loss_frame is None or loss_frame.filter(pl.col("loss") > 0.0).height == 0:
        return None
    on = [c for c in ("_n", "_d", "_t") if c in loss_frame.columns]
    base = (state_dt.select("n", "d", "t")
            .with_columns(_n=pl.col("n").cast(pl.Utf8), _d=pl.col("d").cast(pl.Utf8), _t=pl.col("t").cast(pl.Utf8))
            .join(loss_frame, on=on, how="left")
            .join(dur.frame.select(pl.col("d").cast(pl.Utf8).alias("_d"), pl.col("t").cast(pl.Utf8).alias("_t"),
                                   pl.col("value").cast(pl.Float64).alias("dur")), on=["_d", "_t"], how="left")
            .with_columns(pl.col("loss").fill_null(0.0))
            .sort("_n", "_d", "_t"))
    if base["dur"].null_count() > 0:
        raise ValueError("self-discharge: step_duration is missing for some storage steps")
    lam = np.log1p(base["loss"].to_numpy())
    dur_np = base["dur"].to_numpy()
    alpha, beta = alpha_beta(dur_np * lam)
    base = base.with_columns(alpha=pl.Series(alpha), beta=pl.Series(beta), lam=pl.Series(lam))
    grouped = _grouped_steps(d)
    if grouped is not None:
        g = base.join(grouped, on=["_n", "_d", "_t"], how="inner")
        if g.height > 0:
            grp = ("_n", "_d", "_b")
            # h_b = Σ Δt·λ_t; with λ constant in b the same value is Δ_b·λ (kept in that form, so a constant
            # loss gives the same coefficients whether it was authored as a constant or as a map).
            g = (g.sort("_n", "_d", "_b", "_t")
                 .with_columns(span=pl.col("dur").sum().over(grp),
                               h_b=pl.when(pl.col("lam").min().over(grp) == pl.col("lam").max().over(grp))
                               .then(pl.col("dur").sum().over(grp) * pl.col("lam"))
                               .otherwise((pl.col("dur") * pl.col("lam")).sum().over(grp))))
            ag, bg = alpha_beta(g["h_b"].to_numpy())
            g = g.with_columns(
                alpha=pl.Series(ag) * pl.col("dur") / pl.col("span"),
                beta=pl.when(pl.col("_t") == pl.col("_b")).then(pl.Series(bg)).otherwise(0.0))
            base = (base.join(g.select("_n", "_d", "_t", pl.col("alpha").alias("_ga"), pl.col("beta").alias("_gb")),
                              on=["_n", "_d", "_t"], how="left")
                    .with_columns(alpha=pl.coalesce("_ga", "alpha"), beta=pl.coalesce("_gb", "beta"))
                    .drop("_ga", "_gb"))
    frame = base.select("n", "d", "t", "alpha", "beta")
    keys = ("n", "d", "t")
    return LossCoefficients(
        frame=frame,
        keep_now=Param(keys, frame.select("n", "d", "t", value=1.0 + pl.col("alpha"))),
        keep_prev=Param(keys, frame.select("n", "d", "t", value=1.0 - pl.col("beta"))),
        h=Param(keys, frame.select("n", "d", "t", value=pl.col("alpha") + pl.col("beta"))),
    )


# ---------------------------------------------------------------------------
# The previous state of each storage step (shared by the model's state change and the output writer)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateLink:
    """One state-change branch: the nodes of a binding method and where their steps' previous state lies.

    ``lag`` is the lag frame and ``col`` its previous-step column: with ``cross_period`` the previous step is
    ``(d_previous, t_previous_within_solve)`` of ``lag`` (it may lie in another period), else ``(d, col)``.
    ``all_steps``: every step of the solve has a lag row (the model then takes the current state unrestricted).
    """
    key: str
    nodes: pl.DataFrame
    lag: pl.DataFrame
    col: str
    cross_period: bool
    all_steps: bool


def state_links(d, *, has_rp: bool, has_nsb: bool) -> list[StateLink]:
    """The state-change branches of ``model.core._state_change_terms``, one per storage_binding_method; each node
    is in one branch only (``_validate_storage_binding_disjoint``)."""
    links: list[StateLink] = []
    if d.storage_bind_within_timeblock is not None:
        # bind_within_timeblock: t⁻ wraps around within the timeblock.
        links.append(StateLink("state_change", d.storage_bind_within_timeblock, d.dtttdt,
                               "t_previous_within_timeset", False, True))
    if _has_rows(d.storage_bind_forward_only) and _has_rows(d.dtttdt_forward_only):
        # bind_forward_only: t⁻ crosses period boundaries; the first step of the solve has no state change
        # (dtttdt_forward_only omits it).
        links.append(StateLink("state_change_fo", d.storage_bind_forward_only, d.dtttdt_forward_only,
                               "t_previous_within_solve", True, False))
    if _has_rows(d.storage_bind_within_solve):
        # bind_within_solve: t⁻ crosses period boundaries and the first step of the solve wraps to the last one.
        links.append(StateLink("state_change_ws", d.storage_bind_within_solve, d.dtttdt,
                               "t_previous_within_solve", True, True))
    if _has_rows(d.storage_bind_within_period) and _has_rows(d.dtttdt_within_period):
        # bind_within_period: t⁻ is the previous step of the period; the first step of each period wraps to the
        # period's own last step.
        links.append(StateLink("state_change_wp", d.storage_bind_within_period, d.dtttdt_within_period,
                               "t_previous_within_period", False, True))
    if has_nsb and _has_rows(d.dtttdt_block_lag):
        # bind_intraperiod_blocks: t⁻ from dtttdt_block_lag (the blocks of a period form a loop; in a rolling solve
        # the blocks of a period the roll holds only in part chain forward, with no row at the solve's first step).
        # The fix_start_end block nodes chain forward in every period (block_forward_lag).
        forward = block_forward_nodes(d)
        looping = without_nodes(d.nodeStateBlock, forward)
        if looping is not None:
            links.append(StateLink("state_change_block", looping, d.dtttdt_block_lag,
                                   "t_previous_within_solve", True, False))
        forward_lag = block_forward_lag(d) if forward is not None else None
        if forward_lag is not None:
            links.append(StateLink("state_change_block_forward", forward, forward_lag,
                                   "t_previous_within_solve", True, False))
    if has_rp:
        # Blended weights: t⁻ is the previous step inside a representative block; the block's first step starts
        # from v_state_rp_start (rp_start_steps).
        interior = d.dtttdt.join(d.rp_block_first.select("d", "t"), on=["d", "t"], how="anti")
        links.append(StateLink("state_change_rp_interior", d.nodeState_rp, interior,
                               "t_previous_within_timeset", False, False))
    return links


def previous_steps(links: list[StateLink]) -> pl.DataFrame:
    """``(n, d, t, d_prev, t_prev)`` (Utf8): the step whose state each linked storage step starts from."""
    parts = []
    for link in links:
        lag = link.lag
        d_prev = pl.col("d_previous") if link.cross_period else pl.col("d")
        frame = lag.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8),
                           d_prev.cast(pl.Utf8).alias("d_prev"), pl.col(link.col).cast(pl.Utf8).alias("t_prev"))
        parts.append(link.nodes.select(pl.col("n").cast(pl.Utf8)).join(frame, how="cross"))
    if not parts:
        return pl.DataFrame(schema={c: pl.Utf8 for c in ("n", "d", "t", "d_prev", "t_prev")})
    return pl.concat(parts, how="vertical").select("n", "d", "t", "d_prev", "t_prev")


def self_discharge_energy(d, *, state: pl.DataFrame, rp_start: "pl.DataFrame | None", start: "pl.DataFrame | None",
                          carry: "pl.DataFrame | None", open_first: "pl.DataFrame | None") -> "pl.DataFrame | None":
    """The self-discharge of every storage step of a solved solve, MWh per step: ``(n, d, t, value)``.

    ``state``: the solve's v_state ``(n, d, t, value)`` (state units).  The previous state S⁻ of a step: the
    linked step's state (:func:`state_links`), ``rp_start`` ``(n, d, t, value)`` (v_state_rp_start at the
    representative blocks' first steps), ``start`` / ``carry`` ``(n, d, t, value)`` (start energy / carried
    state in MWh at an open chain's first step), ``open_first`` ``(n, d, t)`` (an open first step: S⁻ = S).
    Coarse-step and block totals are spread over their fine steps by Δt/Δ_b (a constant rate within the step).
    None when no storage node has a positive loss.
    """
    def utf8(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns([pl.col(c).cast(pl.Utf8) for c in ("n", "d", "t") if c in frame.columns])

    state = utf8(state)
    coef = loss_coefficients(d, state.select("n", "d", "t"))
    if coef is None:
        return None
    su = utf8(d.p_state_unitsize.frame).select("n", pl.col("value").alias("su"))
    w = utf8(coef.frame).join(state.rename({"value": "S"}), on=["n", "d", "t"], how="left").join(su, on="n")
    links = state_links(d, has_rp=_has_rows(getattr(d, "nodeState_rp", None)),
                        has_nsb=_has_rows(getattr(d, "nodeStateBlock", None)))
    prev = (previous_steps(links)
            .join(state.select(pl.col("n"), pl.col("d").alias("d_prev"), pl.col("t").alias("t_prev"),
                               pl.col("value").alias("S_prev")), on=["n", "d_prev", "t_prev"], how="left")
            .select("n", "d", "t", (pl.col("S_prev")).alias("prev_state")))
    pieces = [prev]
    if rp_start is not None and rp_start.height > 0:
        pieces.append(utf8(rp_start).select("n", "d", "t", pl.col("value").alias("prev_state")))
    if open_first is not None and open_first.height > 0:
        pieces.append(utf8(open_first).select("n", "d", "t")
                      .join(state, on=["n", "d", "t"], how="left").select("n", "d", "t",
                                                                          pl.col("value").alias("prev_state")))
    prev_state = pl.concat(pieces, how="vertical")
    energy_pieces = [c for c in (start, carry) if c is not None and c.height > 0]
    prev_energy = (pl.concat([utf8(c).select("n", "d", "t", pl.col("value").alias("prev_energy"))
                              for c in energy_pieces], how="vertical")
                   if energy_pieces else pl.DataFrame(schema={"n": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8,
                                                              "prev_energy": pl.Float64}))
    w = (w.join(prev_state, on=["n", "d", "t"], how="left")
         .join(prev_energy, on=["n", "d", "t"], how="left")
         .with_columns(prev=pl.coalesce(pl.col("prev_state") * pl.col("su"), pl.col("prev_energy"), pl.lit(0.0)))
         .with_columns(value=pl.col("alpha") * pl.col("S") * pl.col("su") + pl.col("beta") * pl.col("prev")))
    grouped = _grouped_steps(d)
    if grouped is not None:
        dur = d.p_step_duration.frame.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8),
                                             pl.col("value").alias("dur"))
        g = (grouped.rename({"_n": "n", "_d": "d", "_t": "t", "_b": "b"})
             .join(w.select("n", "d", "t", "value"), on=["n", "d", "t"], how="inner")
             .join(dur, on=["d", "t"], how="left")
             .sort("n", "d", "b", "t")
             .with_columns(spread=pl.col("value").sum().over("n", "d", "b") * pl.col("dur")
                           / pl.col("dur").sum().over("n", "d", "b")))
        w = (w.join(g.select("n", "d", "t", "spread"), on=["n", "d", "t"], how="left")
             .with_columns(value=pl.coalesce("spread", "value")))
    out = w.select("n", "d", "t", "value").sort("n", "d", "t")
    # One loss per storage step: a step given two previous states (two links, a link and an rp start, a start and a
    # carry) would otherwise be reported once by the writer and silently differ from the LP's row.
    dup = out.group_by("n", "d", "t").len().filter(pl.col("len") > 1)
    if dup.height > 0:
        raise ValueError(f"self-discharge output: {dup.height} storage step(s) have more than one previous state, "
                         f"e.g. {dup.select('n', 'd', 't').row(0)}")
    return out


# ---------------------------------------------------------------------------
# Where a storage chain starts (shared by the model's start terms and the output writer)
# ---------------------------------------------------------------------------


def carried_nodes(d) -> "pl.DataFrame | None":
    """Storage nodes (n) with a realised state from before this solve (the nodes of ``p_roll_continue_state``): they
    continue from it and are not pinned by fix_start.  The cascade routes the carry (``solve._carry_routing``): the
    latest state realised before the solve at its nesting level or deeper.  None in the first solve of the model and
    of each nested level (and in a single solve)."""
    carry = d.p_roll_continue_state
    if carry is None or carry.frame.height == 0:
        return None
    return carry.frame.select("n").unique(maintain_order=True)


def without_nodes(frame: "pl.DataFrame | None", nodes: "pl.DataFrame | None") -> "pl.DataFrame | None":
    """``frame`` without the rows of ``nodes`` (on n); None when nothing is left."""
    if frame is None:
        return None
    if _has_rows(nodes):
        frame = frame.join(nodes.select(pl.col("n").cast(frame.schema["n"], strict=False)), on="n", how="anti")
    return frame if frame.height > 0 else None


def node_union(frames) -> "pl.DataFrame | None":
    """The nodes (n) of the frames that have rows, or None."""
    parts = [f.select("n") for f in frames if _has_rows(f)]
    if not parts:
        return None
    return pl.concat(parts, how="vertical").unique(maintain_order=True)


def block_forward_nodes(d) -> "pl.DataFrame | None":
    """bind_intraperiod_blocks nodes (n) with storage_start_end_method fix_start_end: the binding method's loop is
    ignored (reference.md), so their blocks chain forward in every period of the solve (:func:`block_forward_lag`).
    A node has one storage_start_end_method, so storage_fix_start ∩ storage_fix_end is the fix_start_end set.  None
    when there are none."""
    nsb = getattr(d, "nodeStateBlock", None)
    fix_start, fix_end = getattr(d, "storage_fix_start", None), getattr(d, "storage_fix_end", None)
    if not (_has_rows(nsb) and _has_rows(fix_start) and _has_rows(fix_end)):
        return None
    out = (nsb.select(pl.col("n").cast(pl.Utf8))
           .join(fix_start.select(pl.col("n").cast(pl.Utf8)), on="n", how="semi")
           .join(fix_end.select(pl.col("n").cast(pl.Utf8)), on="n", how="semi")
           .select(pl.col("n").cast(nsb.schema["n"], strict=False))
           .unique(maintain_order=True))
    return out if out.height > 0 else None


def block_forward_lag(d) -> "pl.DataFrame | None":
    """``(d, t, d_previous, t_previous_within_solve)``: the block state lag of :func:`block_forward_nodes`, every
    period chained forward (``dtttdt_block_lag`` with every period of the blocks in the chained set: the
    bind_forward_only lag on the block steps, so a period's first block continues from the previous step of the
    solve and the solve's first step has no row)."""
    from flextool.engine.preprocess.params._derived import dtttdt_block_lag_from_dtttdt

    pbt = getattr(d, "period_block_time", None)
    if not (_has_rows(getattr(d, "dtttdt", None)) and _has_rows(pbt)):
        return None
    return dtttdt_block_lag_from_dtttdt(d.dtttdt, pbt, pbt.select("d").unique(maintain_order=True))


def block_open_nodes(d) -> "pl.DataFrame | None":
    """bind_intraperiod_blocks nodes (n) whose blocks chain forward from the solve's first step: the fix_start_end
    block nodes (:func:`block_forward_nodes`) and, in a rolling solve, the block nodes of a roll that holds the period
    of its first step only in part (``period_block_chain``).  None when there are none (every block node then loops
    in the period of the first step)."""
    nsb, chain, first = getattr(d, "nodeStateBlock", None), getattr(d, "period_block_chain", None), d.nodeState_first_dt
    forward = block_forward_nodes(d)
    if not (_has_rows(nsb) and _has_rows(chain) and _has_rows(first)):
        return forward
    names = [str(x) for x in chain["d"].to_list()]
    out = (first.join(nsb.select("n"), on="n", how="semi")
           .filter(pl.col("d").cast(pl.Utf8).is_in(names))
           .select("n").unique(maintain_order=True))
    return node_union((forward, out))


def block_cyclic_nodes(d) -> "pl.DataFrame | None":
    """bind_intraperiod_blocks nodes (n) whose first step wraps to the end of its period (the blocks of that period
    loop): every block node in a solve that is not rolling, and in a roll that holds the first period whole."""
    nsb = getattr(d, "nodeStateBlock", None)
    if not _has_rows(nsb):
        return None
    open_ = block_open_nodes(d)
    out = nsb.select("n") if open_ is None else nsb.select("n").join(open_, on="n", how="anti")
    return out if out.height > 0 else None


def block_tail_ends(d) -> "pl.DataFrame | None":
    """``(n, d, t)`` of ``nodeState_last_dt``: the end of a chained part-period tail of a bind_intraperiod_blocks
    node in a rolling solve.  The roll holds the period only in part (``period_block_chain``: its blocks chain
    forward), the period is not one of the solve's last periods, and no block state link continues from its last
    step (the next period of the roll is held whole and loops on itself).  Nothing follows that state within the
    roll, so the node's storage_solve_horizon_method applies there as at the roll end.  None when there are none."""
    nsb, chain, last = getattr(d, "nodeStateBlock", None), getattr(d, "period_block_chain", None), d.nodeState_last_dt
    if not (_has_rows(nsb) and _has_rows(chain) and _has_rows(last)):
        return None
    chained = [str(x) for x in chain["d"].to_list()]
    last_periods = [str(x) for x in d.period_last["d"].to_list()] if _has_rows(d.period_last) else []
    links = [link for link in state_links(d, has_rp=False, has_nsb=True)
             if link.key in ("state_change_block", "state_change_block_forward")]
    continued = previous_steps(links).select("n", pl.col("d_prev").alias("_d"), pl.col("t_prev").alias("_t"))
    out = (last.with_columns(_n=pl.col("n").cast(pl.Utf8), _d=pl.col("d").cast(pl.Utf8),
                             _t=pl.col("t").cast(pl.Utf8))
           .join(nsb.select(pl.col("n").cast(pl.Utf8).alias("_n")), on="_n", how="semi")
           .filter(pl.col("_d").is_in(chained) & ~pl.col("_d").is_in(last_periods))
           .join(continued.rename({"n": "_n"}), on=["_n", "_d", "_t"], how="anti")
           .select("n", "d", "t")
           .unique()
           .sort("n", "d", "t"))
    return out if out.height > 0 else None


def cyclic_nodes(d) -> "pl.DataFrame | None":
    """Storage nodes whose chain closes on itself within the solve (n): bind_within_timeblock,
    bind_within_period, bind_within_solve and the bind_intraperiod_blocks nodes whose first period loops
    (:func:`block_cyclic_nodes`).  Their first step wraps to a later step of the solve, so they take neither a start
    term nor the roll carry."""
    return node_union((d.storage_bind_within_timeblock, d.storage_bind_within_period,
                        d.storage_bind_within_solve, block_cyclic_nodes(d)))


def open_start_nodes(d) -> "pl.DataFrame | None":
    """Storage nodes whose chain starts open at the solve's first step (n): bind_forward_only and the
    bind_intraperiod_blocks nodes that chain forward from it (:func:`block_open_nodes`: fix_start_end, or a roll
    that holds its first period only in part).
    Their first step starts from the fix_start state in the first solve and from the roll carry in a continuation
    solve."""
    return node_union((d.storage_bind_forward_only, block_open_nodes(d)))


def fix_start_first(d) -> "pl.DataFrame | None":
    """``(n, d, t)``: the first steps that start from the fix_start state (open-start fix_start nodes without a
    carry: the first solve of the model and of each nested level); None when there are none."""
    open_start = open_start_nodes(d)
    if not (open_start is not None
            and _has_rows(d.storage_fix_start)
            and d.p_state_start is not None
            and d.p_state_existing_capacity is not None
            and d.p_state_unitsize is not None
            and d.nodeState_first_dt is not None):
        return None
    nodes = without_nodes(open_start.join(d.storage_fix_start, on="n", how="inner"), carried_nodes(d))
    if nodes is None:
        return None
    first = d.nodeState_first_dt.join(nodes, on="n", how="inner")
    return first if first.height > 0 else None


def roll_first(d) -> "pl.DataFrame | None":
    """``(n, d, t)``: the first steps that start from the carried state (the carried nodes but the cyclic ones and
    the representative-period ones); None when there are none.

    A cyclic node cycles within the solve; a representative-period (blended weights) node starts each block from
    v_state_rp_start, and its realised state is a state inside a representative block, not a chronological level to
    continue from, so it takes no carry (and, being carried, no fix_start pin either)."""
    carried = carried_nodes(d)
    if carried is None or d.p_state_unitsize is None or d.nodeState_first_dt is None:
        return None
    first = d.nodeState_first_dt.join(carried.select(pl.col("n").cast(d.nodeState_first_dt.schema["n"],
                                                                       strict=False)), on="n", how="semi")
    first = without_nodes(first, cyclic_nodes(d))
    return without_nodes(first, getattr(d, "nodeState_rp", None))


def open_first(d) -> "pl.DataFrame | None":
    """``(n, d, t)``: first steps of open-start nodes that start from neither the fix_start state nor the carry
    (the step has no state change: its state is constant over the step)."""
    open_start = open_start_nodes(d)
    if open_start is None or d.nodeState_first_dt is None:
        return None
    first = d.nodeState_first_dt.join(open_start, on="n", how="inner")
    for covered in (fix_start_first(d), roll_first(d)):
        if covered is not None:
            first = first.join(covered.select("n", "d", "t"), on=["n", "d", "t"], how="anti")
    return first if first.height > 0 else None
