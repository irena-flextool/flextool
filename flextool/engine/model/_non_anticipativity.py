"""Non-anticipativity: decisions taken before a forecast is revealed are equal in every forecast branch.

Owns the dispatch ties non_anticipativity_storage_use, _storage_state_first, _online_{integer,linear} and
_reserve, which hold over the tied steps dt_non_anticipativity between the forecast branches of each anchor
period, and the investment ties non_anticipativity_{invest,divest}_{p,n}, which hold over the window
pd_non_anticipativity of a recourse-investment solve.  The module declares no variables.
"""

import logging
from typing import TYPE_CHECKING

import polars as pl
from polar_high import Param, Sum, Where

from flextool.engine.input._axis_enums import cast_dim, rename_to_axis
from flextool.engine.model._common import _has_rows, _var_view

if TYPE_CHECKING:
    from flextool.engine.model.core import _Build

_LOG = logging.getLogger(__name__)


def _forecast_star(d) -> "tuple[pl.DataFrame | None, pl.DataFrame | None]":
    """``(star, tie_dt)``: the forecast pairs ``(d, b)`` to tie and the tied steps ``(d, t)``.

    period_branch_full pairs every anchor (trunk) period ``d`` with the branch periods ``b`` of its fan;
    the self-pair ``b == d`` and branches not in period_in_use (metadata-only rows such as a realized
    marker) are dropped.  Per anchor period the forecasts are sorted and the first is the reference
    forecast ``d_ref``; the star is {(d_ref, b) : b another forecast of that anchor}.  An anchor with
    fewer than two forecasts has nothing to tie.  The tie is between forecasts only, never to the trunk,
    so it holds with no realized branch at all.  A single global ``d_ref`` would be wrong: it would couple
    different anchor periods at mismatched steps.
    dt_non_anticipativity is re-keyed the same way (anchor ``d`` → that anchor's ``d_ref``, same ``t``),
    because every family joins the star and the tied steps on ``d``.  ``tie_dt`` is None when
    dt_non_anticipativity has no rows.  ``star`` is None when there is nothing to tie.
    """
    # d (period axis) and b (branch axis) are different Enum vocabularies: compare and join them as Utf8.
    db_pairs = (d.period_branch_full
                .with_columns(pl.col("d").cast(pl.Utf8), pl.col("b").cast(pl.Utf8))
                .filter(pl.col("d") != pl.col("b")))
    if d.period_in_use_set is not None:
        piu = d.period_in_use_set.rename({"d": "b"}).with_columns(pl.col("b").cast(pl.Utf8))
        db_pairs = db_pairs.join(piu, on="b", how="inner")
    if db_pairs.height == 0:
        return None, None

    tie_raw = d.dt_non_anticipativity
    tie_u = tie_raw.with_columns(pl.col("d").cast(pl.Utf8)) if _has_rows(tie_raw) else None
    stars: list[pl.DataFrame] = []
    ties: list[pl.DataFrame] = []
    for d_trunk in sorted(db_pairs.select("d").unique().to_series().to_list()):
        forecasts = sorted(db_pairs.filter(pl.col("d") == d_trunk).select("b").unique().to_series().to_list())
        if len(forecasts) < 2:
            continue
        d_ref = forecasts[0]
        stars.append(pl.DataFrame({"d": [d_ref] * (len(forecasts) - 1), "b": forecasts[1:]}))
        if tie_u is not None:
            ties.append(tie_u.filter(pl.col("d") == d_trunk).with_columns(pl.lit(d_ref).alias("d")))
    if not stars:
        return None, None
    star = pl.concat(stars).with_columns(cast_dim(pl.col("d"), None, "d"), cast_dim(pl.col("b"), None, "b"))
    tie_dt = pl.concat(ties).with_columns(cast_dim(pl.col("d"), None, "d")) if ties else None
    return star, tie_dt


def _stochastic_nodes(d) -> "pl.DataFrame | None":
    """Nodes ``n`` in a group of groupStochastic; None when there are none."""
    gn, gs = d.group_node, d.groupStochastic
    if not _has_rows(gn) or not _has_rows(gs):
        return None
    nodes = gn.join(gs, on="g", how="inner").select("n").unique()
    return nodes if nodes.height > 0 else None


def add_dispatch_constraints(b: "_Build") -> None:
    """Tie storage use, units online and reserves across the forecast branches of each anchor period."""
    d = b.d
    if not _has_rows(d.period_branch_full) or not (
            _has_rows(d.dt_non_anticipativity) or b.has_online_lin or b.has_online_int):
        return
    star, tie_dt = _forecast_star(d)
    if star is None:
        return
    m = b.m
    v_flow, v_state = b.v_flow, b.v_state
    v_online_integer, v_online_linear = b.v_online_integer, b.v_online_linear
    v_reserve = b.v_reserve
    has_tie_dt = _has_rows(tie_dt)

    # Steps of the online ties: the tied steps, or, when there are none, every online step of the anchor
    # periods.  Unit commitment over the committed section is decided before the forecast is known.  A
    # separate_dispatch parent hands its committed window to the dispatch sibling and has no tied steps;
    # its whole horizon is the committed section.
    online_dt = tie_dt
    if not has_tie_dt and (v_online_integer is not None or v_online_linear is not None) \
            and _has_rows(d.p_online_dt):
        jump_steps = d.p_online_dt.select("d", "t").unique().join(star.select("d").unique(), on="d", how="inner")
        if jump_steps.height > 0:
            online_dt = jump_steps
    if not has_tie_dt and not _has_rows(online_dt):
        return
    stoch_nodes = _stochastic_nodes(d)

    # ─── EQ. ─── non_anticipativity_storage_use ──────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # net[n,d,t] − net[n,b,t] = 0
    #   net[n,d,t] = Σ_{(p,s,n) ∈ process_source_sink} v_flow[p,s,n,d,t] · unitsize[p] · Δt[d,t]
    #              − Σ_{(p,n,k) ∈ process_source_sink_eff}   v_flow[p,n,k,d,t] · unitsize[p] · slope[p,d,t] · Δt[d,t]
    #              − Σ_{(p,n,k) ∈ process_source_sink_noEff} v_flow[p,n,k,d,t] · unitsize[p] · Δt[d,t]
    #   ∀ n ∈ nodes of groupStochastic, (d,b) ∈ forecast star, (d,t) ∈ tied steps
    # The net flow into a stochastic node is the same in every forecast over
    # the tied steps.  It ties the flows, not v_state: the balance slacks of
    # the node could otherwise absorb a flow difference.
    if has_tie_dt and v_flow is not None and stoch_nodes is not None and d.p_unitsize is not None \
            and d.p_step_duration is not None and _has_rows(d.process_source_sink):
        v_flow_at_b = _var_view(v_flow, "__nab", {"d": "b"})
        step_dur_at_b = Param(("b", "t"), d.p_step_duration.frame.rename({"d": "b"}))
        sink_idx = (d.process_source_sink
                    .join(stoch_nodes.pipe(rename_to_axis, {"n": "sink"}), on="sink", how="inner")
                    .with_columns(n=pl.col("sink")))
        src_eff_idx = None
        if d.process_source_sink_eff is not None:
            src_eff_idx = (d.process_source_sink_eff
                           .join(stoch_nodes.pipe(rename_to_axis, {"n": "source"}), on="source", how="inner")
                           .with_columns(n=pl.col("source")))
        src_noEff_idx = None
        if d.process_source_sink_noEff is not None:
            src_noEff_idx = (d.process_source_sink_noEff
                             .join(stoch_nodes.pipe(rename_to_axis, {"n": "source"}), on="source", how="inner")
                             .with_columns(n=pl.col("source")))
        has_eff = _has_rows(src_eff_idx) and d.p_slope is not None
        arcs = ("p", "source", "sink")

        lhs_d = []
        if sink_idx.height > 0:
            lhs_d.append(Sum(Where(v_flow * d.p_unitsize, sink_idx) * d.p_step_duration, over=arcs))
        if has_eff:
            lhs_d.append(-Sum(Where(v_flow * d.p_unitsize * d.p_slope, src_eff_idx) * d.p_step_duration, over=arcs))
        if _has_rows(src_noEff_idx):
            lhs_d.append(-Sum(Where(v_flow * d.p_unitsize, src_noEff_idx) * d.p_step_duration, over=arcs))

        lhs_b = []
        if sink_idx.height > 0:
            lhs_b.append(-Sum(Where(v_flow_at_b * d.p_unitsize, sink_idx) * step_dur_at_b, over=arcs))
        if has_eff:
            slope_at_b = Param(("p", "b", "t"), d.p_slope.frame.rename({"d": "b"}))
            lhs_b.append(Sum(Where(v_flow_at_b * d.p_unitsize * slope_at_b, src_eff_idx) * step_dur_at_b,
                             over=arcs))
        if _has_rows(src_noEff_idx):
            lhs_b.append(Sum(Where(v_flow_at_b * d.p_unitsize, src_noEff_idx) * step_dur_at_b, over=arcs))

        if lhs_d and lhs_b:
            over = (stoch_nodes
                    .join(star, how="cross")
                    .join(tie_dt, on="d", how="inner")
                    .select("n", "d", "b", "t").unique())
            if over.height > 0:
                lhs_terms = {f"d_{i}": e for i, e in enumerate(lhs_d)}
                lhs_terms.update({f"b_{i}": e for i, e in enumerate(lhs_b)})
                m.add_cstr(
                    "non_anticipativity_storage_use",
                    over      = over,
                    sense     = "==",
                    lhs_terms = lhs_terms,
                    rhs_terms = {},
                )

    # ─── EQ. ─── non_anticipativity_storage_state_first ──────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t_first(d)] − v_state[n,b,t_first(d)] = 0
    #   ∀ n ∈ nodes of groupStochastic, (d,b) ∈ forecast star, t_first(d) = first tied step of d,
    #     only where the tied steps of d are fewer than the v_state steps of (n,d)
    # The net-charge tie above fixes the shape of the state over the tied
    # steps but leaves a constant offset between forecasts; this row removes
    # it.  Where the tied steps cover the whole horizon the state is already
    # bound at both ends (start and reference value) and the row is
    # redundant, so it is emitted only when untied steps follow (rolling).
    # The offset is pinned at the optimum, not by construction: interior
    # drift stays feasible where balance slack or forecast-specific inflow
    # absorbs it.  v_online and v_reserve need no analogue: their ties pin
    # the variable itself at every tied step.
    if has_tie_dt and v_state is not None and stoch_nodes is not None:
        vstate_dt = v_state.frame.select("n", "d", "t").unique()
        tied_steps = tie_dt.select("d", "t").unique()
        first_per_d = tied_steps.group_by("d").agg(pl.col("t").min().alias("t"))
        tied_counts = tied_steps.group_by("d").agg(pl.len().alias("tied_count"))
        full_counts = vstate_dt.group_by("n", "d").agg(pl.len().alias("full_count"))
        nd_keep = (stoch_nodes
                   .join(tie_dt.select("d").unique(), how="cross")
                   .join(tied_counts, on="d", how="inner")
                   .join(full_counts, on=["n", "d"], how="inner")
                   .filter(pl.col("tied_count") < pl.col("full_count"))
                   .select("n", "d").unique())
        if nd_keep.height > 0:
            over = (nd_keep
                    .join(first_per_d, on="d", how="inner")
                    .join(star, on="d", how="inner")
                    .select("n", "d", "b", "t").unique())
            # Keep only cells with a v_state column on both sides.  over.b is on the branch axis and the
            # state's d on the period axis: join the b side as Utf8.
            over = over.join(vstate_dt, on=["n", "d", "t"], how="inner")
            vbt = vstate_dt.rename({"d": "b"}).with_columns(pl.col("b").cast(pl.Utf8))
            over = (over
                    .with_columns(pl.col("b").cast(pl.Utf8))
                    .join(vbt, on=["n", "b", "t"], how="inner")
                    .with_columns(cast_dim(pl.col("b"), None, "b"))
                    .select("n", "d", "b", "t").unique())
            if over.height > 0:
                m.add_cstr(
                    "non_anticipativity_storage_state_first",
                    over      = over,
                    sense     = "==",
                    lhs_terms = {"v_d":  v_state,
                                 "v_b": -_var_view(v_state, "__nab_first", {"d": "b"})},
                    rhs_terms = {},
                )

    # ─── EQ. ─── non_anticipativity_online_{integer,linear} ──────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_online[p,d,t] − v_online[p,b,t] = 0
    #   ∀ p ∈ process_online_{integer,linear}, (d,b) ∈ forecast star, (d,t) ∈ online tie steps,
    #     (p,d,t) and (p,b,t) ∈ p_online_dt
    # Units online are the same in every forecast over the committed section.
    for kind, v_online, online_set in (("integer", v_online_integer, d.process_online_integer),
                                       ("linear", v_online_linear, d.process_online_linear)):
        if v_online is None or not _has_rows(online_set):
            continue
        over = (online_set
                .join(star, how="cross")
                .join(online_dt, on="d", how="inner")
                .select("p", "d", "b", "t").unique())
        if d.p_online_dt is not None:
            pdt = d.p_online_dt.select("p", "d", "t").unique()
            over = (over
                    .join(pdt, on=["p", "d", "t"], how="inner")
                    .join(pdt.rename({"d": "b"}), on=["p", "b", "t"], how="inner")
                    .select("p", "d", "b", "t").unique())
        if over.height > 0:
            m.add_cstr(
                f"non_anticipativity_online_{kind}",
                over      = over,
                sense     = "==",
                lhs_terms = {"v_d":  v_online,
                             "v_b": -_var_view(v_online, "__nab", {"d": "b"})},
                rhs_terms = {},
            )

    # ─── EQ. ─── non_anticipativity_reserve ──────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_reserve[p,r,ud,n,d,t] − v_reserve[p,r,ud,n,b,t] = 0
    #   ∀ (p,r,ud,n) ∈ process_reserve_upDown_node_active, (d,b) ∈ forecast star, (d,t) ∈ tied steps
    # Reserve offered is the same in every forecast over the tied steps.
    if has_tie_dt and v_reserve is not None and _has_rows(d.process_reserve_upDown_node_active) \
            and _has_rows(d.reserve_upDown_group):
        over = (d.process_reserve_upDown_node_active
                .join(star, how="cross")
                .join(tie_dt, on="d", how="inner")
                .select("p", "r", "ud", "n", "d", "b", "t").unique())
        if over.height > 0:
            m.add_cstr(
                "non_anticipativity_reserve",
                over      = over,
                sense     = "==",
                lhs_terms = {"v_d":  v_reserve,
                             "v_b": -_var_view(v_reserve, "__nab", {"d": "b"})},
                rhs_terms = {},
            )


def add_invest_constraints(b: "_Build") -> None:
    """Tie investment and divestment across the branches of the shared-investment window."""
    d = b.d
    window = d.pd_non_anticipativity                       # (d, b)
    if not _has_rows(window):
        return
    if not getattr(d, "recourse_invest", False):
        # Without recourse investment there is one investment per period for all branches: nothing to tie.
        _LOG.info(
            "shared_invest_periods is set but "
            "stochastic_invest_method is not 'recourse'; investment is "
            "already shared across all periods, so the invest-NA window "
            "has no effect (use 'recourse' for shared-first-period / "
            "recourse-later)."
        )
        return
    m = b.m

    # ─── EQ. ─── non_anticipativity_{invest,divest}_{p,n} ────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_invest[e,d] − v_invest[e,b] = 0,   v_divest[e,d] − v_divest[e,b] = 0
    #   ∀ (d,b) ∈ pd_non_anticipativity, (e,d) and (e,b) ∈ {pd,nd}_{invest,divest}_set,
    #     e = process (p) or node (n)
    # Investment and divestment inside the window are the same in every
    # branch.  Divestment must be tied too: under recourse it has a column
    # per branch like investment, and a free divestment would let the net
    # capacity existing + invest − divest differ across branches.
    # pd_non_anticipativity.b is on the branch axis and the invest sets' d on
    # the period axis: the (e,b) check joins b as Utf8.  It drops every pair
    # when the branches have no invest column of their own.
    for name, v, inv_set, ent in (
        ("non_anticipativity_invest_p", b.v_invest_p, d.pd_invest_set, "p"),
        ("non_anticipativity_invest_n", b.v_invest_n, d.nd_invest_set, "n"),
        ("non_anticipativity_divest_p", b.v_divest_p, d.pd_divest_set, "p"),
        ("non_anticipativity_divest_n", b.v_divest_n, d.nd_divest_set, "n"),
    ):
        if v is None or not _has_rows(inv_set):
            continue
        cells = inv_set.select(ent, "d").unique()
        over = cells.join(window, on="d", how="inner")
        if over.height == 0:
            continue
        cells_at_b = cells.rename({"d": "b"}).with_columns(pl.col("b").cast(pl.Utf8))
        over = (over
                .with_columns(pl.col("b").cast(pl.Utf8))
                .join(cells_at_b, on=[ent, "b"], how="inner")
                .with_columns(cast_dim(pl.col("d"), None, "d"), cast_dim(pl.col("b"), None, "b"))
                .select(ent, "d", "b").unique())
        if over.height == 0:
            continue
        m.add_cstr(
            name,
            over      = over,
            sense     = "==",
            lhs_terms = {"v_d":  v,
                         "v_b": -_var_view(v, "__nab", {"d": "b"})},
            rhs_terms = {},
        )
