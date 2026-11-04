"""Non-anticipativity: decisions taken before a forecast is revealed are equal in every forecast branch.

Owns the dispatch ties non_anticipativity_storage_use, _storage_state_first, _online_{integer,linear} and
_reserve, which hold over the tied steps dt_non_anticipativity between the forecast branches of each anchor
period, and the investment ties non_anticipativity_{invest,divest}_{p,n}, which hold over the window
pd_non_anticipativity of a recourse-investment solve.  The module declares no variables.
"""

import logging
from typing import TYPE_CHECKING

import polars as pl

from flextool.engine.input._axis_enums import cast_dim
from flextool.engine.model import _balance_terms
from flextool.engine.model._common import _has_rows, _var_view
from flextool.engine.preprocess.params._branch import forecast_star_pairs

if TYPE_CHECKING:
    from flextool.engine.model.core import _Build

_LOG = logging.getLogger(__name__)


def reference_forecasts(d) -> "list[tuple[str, list[str]]]":
    """``[(anchor, forecasts)]``: per anchor period (the period label that opens a fan), its sorted forecast branch periods.

    period_branch_full pairs every anchor period ``d`` with the branch periods ``b`` of its fan; the self-pair
    ``b == d`` and branches not in period_in_use (metadata-only rows such as a realized marker) are dropped.
    Anchors are listed in sorted order; the first forecast of an anchor is its reference forecast ``d_ref``.
    """
    if not _has_rows(d.period_branch_full):
        return []
    # d (period axis) and b (branch axis) are different Enum vocabularies: compare and join them as Utf8.
    db_pairs = (d.period_branch_full
                .with_columns(pl.col("d").cast(pl.Utf8), pl.col("b").cast(pl.Utf8))
                .filter(pl.col("d") != pl.col("b")))
    if d.period_in_use_set is not None:
        piu = d.period_in_use_set.rename({"d": "b"}).with_columns(pl.col("b").cast(pl.Utf8))
        db_pairs = db_pairs.join(piu, on="b", how="inner")
    return [(d_anchor, sorted(db_pairs.filter(pl.col("d") == d_anchor).select("b").unique().to_series().to_list()))
            for d_anchor in sorted(db_pairs.select("d").unique().to_series().to_list())]


def online_tie_cells(d) -> "pl.DataFrame | None":
    """``(anchor, d_ref, t)`` Utf8: the steps over which units online are tied, at each anchor's reference forecast.

    The steps are those of non_anticipativity_online: the tied steps dt_non_anticipativity of the anchors.  Units
    online are tied only where committed: with no tied step they are free in every forecast.  An anchor with a
    single forecast has no tie rows but the same committed steps (its one forecast is the tied value).  None when
    there is no anchor or no tied step.
    """
    refs = [(a, f[0]) for a, f in reference_forecasts(d) if f]
    if not refs:
        return None
    ref = pl.DataFrame({"anchor": [a for a, _ in refs], "d_ref": [r for _, r in refs]})
    if not _has_rows(d.dt_non_anticipativity):
        return None
    tie = (d.dt_non_anticipativity
           .select(pl.col("d").cast(pl.Utf8).alias("anchor"), pl.col("t").cast(pl.Utf8))
           .unique())
    cells = ref.join(tie, on="anchor", how="inner")
    return cells.select("anchor", "d_ref", "t").sort("anchor", "t") if cells.height > 0 else None


def reserve_tie_cells(d) -> "pl.DataFrame | None":
    """``(anchor, d_ref, t)`` Utf8: the tied steps dt_non_anticipativity of each anchor, at its reference forecast.

    The steps of non_anticipativity_reserve (reserve is tied only over the tied steps).  None when nothing is tied.
    """
    refs = [(a, f[0]) for a, f in reference_forecasts(d) if f]
    if not refs or not _has_rows(d.dt_non_anticipativity):
        return None
    ref = pl.DataFrame({"anchor": [a for a, _ in refs], "d_ref": [r for _, r in refs]})
    tie = (d.dt_non_anticipativity
           .select(pl.col("d").cast(pl.Utf8).alias("anchor"), pl.col("t").cast(pl.Utf8))
           .unique())
    cells = ref.join(tie, on="anchor", how="inner")
    return cells.select("anchor", "d_ref", "t").sort("anchor", "t") if cells.height > 0 else None


def _forecast_star(d) -> "tuple[pl.DataFrame | None, pl.DataFrame | None]":
    """``(star, tie_dt)``: the forecast pairs ``(d, b)`` to tie and the tied steps ``(d, t)``.

    Per anchor period (:func:`reference_forecasts`) the first forecast is the reference forecast ``d_ref``; the
    star is {(d_ref, b) : b another forecast of that anchor}.  An anchor with fewer than two forecasts has nothing
    to tie.  The tie is between forecasts only, never to the period label, so it holds with no realized branch at all.
    A single global ``d_ref`` would be wrong: it would couple different anchor periods at mismatched steps.
    dt_non_anticipativity is re-keyed the same way (anchor ``d`` → that anchor's ``d_ref``, same ``t``),
    because every family joins the star and the tied steps on ``d``.  ``tie_dt`` is None when
    dt_non_anticipativity has no rows.  ``star`` is None when there is nothing to tie.
    """
    anchors = reference_forecasts(d)
    if not anchors:
        return None, None

    tie_raw = d.dt_non_anticipativity
    tie_u = tie_raw.with_columns(pl.col("d").cast(pl.Utf8)) if _has_rows(tie_raw) else None
    pairs = forecast_star_pairs(dict(anchors))
    if not pairs:
        return None, None
    ties: list[pl.DataFrame] = []
    if tie_u is not None:
        for d_anchor, forecasts in anchors:
            if len(forecasts) >= 2:
                ties.append(tie_u.filter(pl.col("d") == d_anchor).with_columns(pl.lit(forecasts[0]).alias("d")))
    star = (pl.DataFrame({"d": [a for a, _ in pairs], "b": [b for _, b in pairs]}, schema={"d": pl.Utf8, "b": pl.Utf8})
            .with_columns(cast_dim(pl.col("d"), None, "d"), cast_dim(pl.col("b"), None, "b")))
    tie_dt = pl.concat(ties).with_columns(cast_dim(pl.col("d"), None, "d")) if ties else None
    return star, tie_dt


def _stochastic_nodes(d) -> "pl.DataFrame | None":
    """Nodes ``n`` in a group of groupStochastic; None when there are none."""
    gn, gs = d.group_node, d.groupStochastic
    if not _has_rows(gn) or not _has_rows(gs):
        return None
    nodes = gn.join(gs, on="g", how="inner").select("n").unique()
    return nodes if nodes.height > 0 else None


def _balance_nodes(d, nodes: "pl.DataFrame | None") -> "pl.DataFrame | None":
    """The rows of ``nodes`` that are balance nodes (``d.nodeBalance``); None when none are."""
    if nodes is None or not _has_rows(d.nodeBalance):
        return None
    names = [str(n) for n in d.nodeBalance["n"].to_list()]
    out = nodes.filter(pl.col("n").cast(pl.Utf8).is_in(names))
    return out if out.height > 0 else None


def add_dispatch_constraints(b: "_Build") -> None:
    """Tie storage use, units online and reserves across the forecast branches of each anchor period."""
    d = b.d
    if not _has_rows(d.period_branch_full) or not _has_rows(d.dt_non_anticipativity):
        return
    star, tie_dt = _forecast_star(d)
    if star is None or not _has_rows(tie_dt):
        return
    m = b.m
    v_state = b.v_state
    v_online_integer, v_online_linear = b.v_online_integer, b.v_online_linear
    v_reserve = b.v_reserve
    stoch_nodes = _stochastic_nodes(d)

    # ─── EQ. ─── non_anticipativity_storage_use ──────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # net[n,d,t] − net[n,b,t] = 0
    #   net[n,d,t] = Σ_{(p,s,n) ∈ flow_to_n} v_flow[p,s,n,d,t] · unitsize[p] · Δt[d,t]
    #              − Σ_{(p,n,k) ∈ flow_from_nodeBalance_eff}
    #                  (v_flow[p,n,k,d,t] · slope[p,d,t] + v_online[p,d,t] · section[p,d,t]) · unitsize[p] · Δt[d,t]
    #              − Σ_{(p,n,k) ∈ flow_from_nodeBalance_noEff} v_flow[p,n,k,d,t] · unitsize[p] · Δt[d,t]
    #              − Σ_{(p,s,n) ∈ twoway1var_arcs} v_flow_back[p,s,n,d,t] · unitsize[p] · Δt[d,t]
    #              + Σ_{(p,n,k) ∈ twoway1var_arcs} v_flow_back[p,n,k,d,t] · unitsize[p] · Δt[d,t]
    #   ∀ n ∈ nodes of groupStochastic ∩ nodeBalance, (d,b) ∈ forecast star, (d,t) ∈ tied steps
    #   section only for p ∈ process_min_load_eff (v_online = v_online_linear or v_online_integer)
    # The net flow into a stochastic node is the same in every forecast over the tied steps: exactly the flow
    # terms of nodeBalance_eq, built by the same helper (_balance_terms.balance_flow_terms).  It ties the flows,
    # not v_state, and leaves out the balance slacks: they could otherwise absorb a flow difference.  The
    # section input counts because efficiency and min_load may differ between forecasts.  A node of the group
    # without an energy balance (e.g. a commodity node) has no balance flow terms to tie.  A coarse process has
    # one flow column per coarse step, so a tie at one of its steps holds the whole step; it is coarse only where
    # each of its coarse steps is wholly tied or wholly untied (flex_data.coarse_tie_straddles).
    tie_nodes = _balance_nodes(d, stoch_nodes)
    if b.v_flow is not None and tie_nodes is not None and d.p_unitsize is not None \
            and d.p_step_duration is not None:
        flow_d = _balance_terms.balance_flow_terms(b, nodes=tie_nodes)
        flow_b = _balance_terms.balance_flow_terms(b, at_branch=True, nodes=tie_nodes)
        if flow_d and flow_b:
            over = (tie_nodes
                    .join(star, how="cross")
                    .join(tie_dt, on="d", how="inner")
                    .select("n", "d", "b", "t").unique())
            if over.height > 0:
                lhs_terms = {f"d_{k}": e for k, e in flow_d.items()}
                lhs_terms.update({f"b_{k}": -e for k, e in flow_b.items()})
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
    if v_state is not None and stoch_nodes is not None:
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
    #   ∀ p ∈ process_online_{integer,linear}, (d,b) ∈ forecast star, (d,t) ∈ tied steps,
    #     (p,d,t) and (p,b,t) ∈ p_online_dt
    # Units online are the same in every forecast over the tied steps, the committed section decided before the
    # forecast is known (inside a fan: the committed cells of its period label, solve_data/committed_in_fan).
    # With no tied step (shared_operation_periods = [], or a rolling fan only in the look-ahead) they are free.
    for kind, v_online, online_set in (("integer", v_online_integer, d.process_online_integer),
                                       ("linear", v_online_linear, d.process_online_linear)):
        if v_online is None or not _has_rows(online_set):
            continue
        over = (online_set
                .join(star, how="cross")
                .join(tie_dt, on="d", how="inner")
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
    if v_reserve is not None and _has_rows(d.process_reserve_upDown_node_active) \
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
    #     e = process (p) or node (n); the divest sets hold the periods with pre-existing capacity only
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
