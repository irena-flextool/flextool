"""Investment floors, cumulative capacity limits, group investment limits and flow-group limits.

Owns the rows that bound investment from below per entity (per period and over the horizon), bound the
cumulative capacity (existing + invested − divested) of an entity, bound what a group of entities invests
or divests (per period, over the horizon, cumulatively), and bound the flow into a flow group (over the
solve, per period, per step).  The per-entity investment maxima are core rows in model/core.py.  The module
declares no variables and adds no objective terms: it reads v_invest_*, v_divest_* and v_flow from the
build context, and its rows are emitted only when some of its data is populated (has_feature).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polar_high import Param, Sum, Where
from polar_high.engine import Expr, Var

from flextool.engine.input._axis_enums import align_join_dtypes, rename_to_axis
from flextool.engine.model._common import (
    _entities_of_kind,
    _entity_rows,
    _has_rows,
    _var_view,
    chain_prior,
    check_max_cap,
    group_prior,
    leaf_rows,
    minus_prior,
)

if TYPE_CHECKING:
    from flextool.engine.model.core import _Build


# ---------------------------------------------------------------------------
# Feature flag (set on the build context by model._prepare)

# Any of these fields populated and non-empty turns the module on; each family is further gated on its
# own data.
_GATE_FIELDS: tuple[str, ...] = (
    "e_invest_min_total", "e_divest_min_total",
    "ed_invest_min_period", "ed_divest_min_period",
    "ed_invest_forbidden_no_investment",
    "ed_invest_cumulative", "ed_cumulative_max_capacity", "ed_cumulative_min_capacity",
    "gd_invest_period", "gd_divest_period",
    "g_invest_total", "g_divest_total", "g_invest_cumulative",
    "group_entity",
    "p_group_max_cumulative_flow", "p_group_min_cumulative_flow",
    "pd_max_cumulative_flow", "pd_min_cumulative_flow",
    "pdt_max_instant_flow", "pdt_min_instant_flow",
    "group_process_node",
)


def has_feature(d) -> bool:
    """True when any of the module's data fields (sets or Params) is populated and non-empty."""
    for f in _GATE_FIELDS:
        v = getattr(d, f, None)
        if v is None:
            continue
        frame = getattr(v, "frame", v)           # a Param carries its rows in .frame
        if getattr(frame, "height", 0) > 0:
            return True
    return False


# ---------------------------------------------------------------------------
# Helpers


def _unitsize(d, axis: str) -> Param | None:
    """unitsize[p] for the process axis ``p``, state_unitsize[n] for the node axis ``n``; None when missing."""
    if axis == "p":
        return d.p_unitsize
    return None if d.p_state_unitsize is None else Param(("n",), d.p_state_unitsize.frame)


def _on_var_axis(frame: pl.DataFrame, axis: str, var: Var) -> pl.DataFrame:
    """``frame`` (keyed on the entity-union axis e) renamed to ``axis`` and restricted to the entities of ``var``.

    The entity-union Enum is cast to the narrower process / node Enum of ``var`` first, so that the
    membership test compares matching dtypes.
    """
    out = frame.rename({"e": axis})
    dtype = var.frame.schema[axis]
    if out.schema[axis] != dtype:
        out = out.with_columns(pl.col(axis).cast(dtype, strict=False))
    return out.filter(pl.col(axis).is_in(var.frame[axis].unique()))


def _group_flow(b: _Build) -> Expr | None:
    """Net flow into each flow group, MW, indexed (g, d, t); None when no group sees a flow.

    F[g,d,t] = Σ_{(p,s,k,n) ∈ flow_to_n, (g,p,n) ∈ gpn} v_flow[p,s,k,d,t] · unitsize[p]
             − Σ_{(p,s,k,n) ∈ flow_from_n ∩ pss_eff, (g,p,n) ∈ gpn} v_flow[p,s,k,d,t] · unitsize[p] · slope[p,d,t]
             − Σ_{(p,s,k,n) ∈ flow_from_n ∩ pss_noEff, (g,p,n) ∈ gpn} v_flow[p,s,k,d,t] · unitsize[p]
      gpn = group_process_node;  flow_to_n / flow_from_n carry the sink / source node as n
    """
    d, v_flow = b.d, b.v_flow
    gpn = d.group_process_node
    if v_flow is None or not _has_rows(gpn):
        return None
    parts: list[Expr] = []
    if d.process_source_sink is not None and _has_rows(d.flow_to_n):
        arcs = d.flow_to_n.join(gpn, on=["p", "n"], how="inner")
        if arcs.height > 0:
            parts.append(Sum(Where(v_flow * d.p_unitsize, arcs), over=("p", "source", "sink", "n")))
    if d.process_source_sink_eff is not None and _has_rows(d.flow_from_n):
        arcs = (d.flow_from_n.join(d.process_source_sink_eff, on=["p", "source", "sink"], how="inner")
                .join(gpn, on=["p", "n"], how="inner"))
        if arcs.height > 0:
            parts.append(-Sum(Where(v_flow * d.p_unitsize * d.p_slope, arcs), over=("p", "source", "sink", "n")))
    if d.process_source_sink_noEff is not None and _has_rows(d.flow_from_n):
        arcs = (d.flow_from_n.join(d.process_source_sink_noEff, on=["p", "source", "sink"], how="inner")
                .join(gpn, on=["p", "n"], how="inner"))
        if arcs.height > 0:
            parts.append(-Sum(Where(v_flow * d.p_unitsize, arcs), over=("p", "source", "sink", "n")))
    if not parts:
        return None
    total = parts[0]
    for part in parts[1:]:
        total = total + part
    return total


def _instant_flow_support(cap_frame: pl.DataFrame, dt: pl.DataFrame) -> pl.DataFrame:
    """The (g, d, t) rows of an instant-flow limit: the cap's own axes, the missing ones filled from ``dt``.

    The cap comes from ``resolve_param_shape`` / ``broadcast_to_period_time`` keyed by the authored axes:
    a scalar gives (g,), a period map (g, d), a time map (g, t), a period-time map (g, d, t).  Deriving
    the rows from the resolved cap keeps them in step with the RHS, so the limit binds at every active
    step of every authored shape (including Spine's default ``"x"`` index name and constant maps).
    """
    has_d = "d" in cap_frame.columns
    has_t = "t" in cap_frame.columns
    if has_d and has_t:
        over = cap_frame.select("g", "d", "t")
    elif has_d:
        over = cap_frame.select("g", "d").unique().join(dt, on="d", how="inner")
    elif has_t:
        over = cap_frame.select("g", "t").unique().join(dt, on="t", how="inner")
    else:
        over = cap_frame.select("g").unique().join(dt, how="cross")
    return over.select("g", "d", "t").unique()


# ---------------------------------------------------------------------------
# Constraints


def add_constraints(b: _Build) -> None:
    """Every family of the module, in emit order, when some of its data is populated."""
    if not b.has_cumulative_invest:
        return
    _eq_fix_no_investment(b)
    _eq_min_entity_period(b)
    _eq_min_invest_total(b)
    _eq_min_divest_total(b)
    _eq_cumulative_capacity(b)
    _eq_group_entity_period(b)
    _eq_group_invest_total(b)
    _eq_group_divest_total(b)
    _eq_group_invest_cumulative(b)
    _eq_cumulative_flow_solve(b)
    _eq_cumulative_flow_period(b)
    _eq_instant_flow(b)


def _eq_fix_no_investment(b: _Build) -> None:
    """fix_v_invest_no_investment_eq_{p,n}: no investment in the cells where investment is forbidden."""
    forbid = b.d.ed_invest_forbidden_no_investment
    if not _has_rows(forbid):
        return

    # ─── EQ. ─── fix_v_invest_no_investment_eq_{p,n} ─────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_invest_p[p,d] = 0   ∀ (p,d) ∈ ed_invest_forbidden_no_investment ∩ pd_invest_set
    # v_invest_n[n,d] = 0   ∀ (n,d) ∈ ed_invest_forbidden_no_investment ∩ nd_invest_set
    # Pins investment to zero where the entity's invest method forbids it.
    # Never emits rows: ed_invest_set (the v_invest_* domain) already excludes the forbidden cells, so the
    # row domain is empty.
    for name, var, axis in (("fix_v_invest_no_investment_eq_p", b.v_invest_p, "p"),
                            ("fix_v_invest_no_investment_eq_n", b.v_invest_n, "n")):
        if var is None:
            continue
        over, var_idx = align_join_dtypes(forbid.rename({"e": axis}), var.frame.select(axis, "d"), (axis, "d"))
        over = over.join(var_idx, on=[axis, "d"], how="inner")
        if over.height == 0:
            continue
        b.m.add_cstr(
            name,
            over      = over,
            sense     = "==",
            lhs_terms = {"invest": var},
            rhs_terms = {"zero":   0.0},
        )


def _eq_min_entity_period(b: _Build) -> None:
    """min{Invest,Divest}_entity_period_{p,n}: capacity built or retired per period, from below."""
    d = b.d
    if d.ed_invest_min_period is None and d.ed_divest_min_period is None:
        return

    # ─── EQ. ─── min{Invest,Divest}_entity_period_{p,n} ──────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_invest_p[p,d] · unitsize[p]        ≥ invest_min_period[p,d]   ∀ (p,d) ∈ ed_invest_period_set ∩ pd_invest_set
    # v_invest_n[n,d] · state_unitsize[n]  ≥ invest_min_period[n,d]   ∀ (n,d) ∈ ed_invest_period_set ∩ nd_invest_set
    # v_divest_p[p,d] · unitsize[p]        ≥ divest_min_period[p,d]   ∀ (p,d) ∈ ed_divest_period_set ∩ pd_divest_set
    # v_divest_n[n,d] · state_unitsize[n]  ≥ divest_min_period[n,d]   ∀ (n,d) ∈ ed_divest_period_set ∩ nd_divest_set
    # Forces a minimum capacity to be built or retired in a period; the floor twin of
    # max{Invest,Divest}_entity_period in model/core.py.  Capacity in MW (MWh for storage nodes).
    for name, period_set, cap, var, axis, kind in (
        ("minInvest_entity_period_p", d.ed_invest_period_set, d.ed_invest_min_period, b.v_invest_p, "p", "invest"),
        ("minInvest_entity_period_n", d.ed_invest_period_set, d.ed_invest_min_period, b.v_invest_n, "n", "invest"),
        ("minDivest_entity_period_p", d.ed_divest_period_set, d.ed_divest_min_period, b.v_divest_p, "p", "divest"),
        ("minDivest_entity_period_n", d.ed_divest_period_set, d.ed_divest_min_period, b.v_divest_n, "n", "divest"),
    ):
        unitsize = _unitsize(d, axis)
        if not _has_rows(period_set) or cap is None or var is None or unitsize is None:
            continue
        over = period_set.rename({"e": axis}).join(var.frame.select(axis, "d"), on=[axis, "d"], how="inner")
        if over.height == 0:
            continue
        b.m.add_cstr(
            name,
            over      = over,
            sense     = ">=",
            lhs_terms = {kind:  var * unitsize},
            rhs_terms = {"cap": Param((axis, "d"), cap.frame.rename({"e": axis}))},
        )


def _eq_min_invest_total(b: _Build) -> None:
    """minInvest_entity_total_path_{p,n}: capacity invested over the chain, from below."""
    d = b.d
    e_set, cap = d.e_invest_total, d.e_invest_min_total
    if not _has_rows(e_set) or cap is None:
        return
    d_leaf = getattr(d, "d_leaf", None)
    prior = chain_prior(d, "invest")

    # ─── EQ. ─── minInvest_entity_total_path_{p,n} ───────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{d ∈ D(leaf)} v_invest_p[p,d] · unitsize[p]  ≥  max(0, invest_min_total[p] − prior_inv[p])
    #   ∀ p ∈ e_invest_total ∩ P(v_invest_p) with invest_min_total (, leaf)
    #   prior_inv[p] = p_entity_invested[p]  (capacity realized by all earlier solves of the chain)
    #   _n: the node analogue with v_invest_n and state_unitsize[n]
    # The part of the minimum not met by earlier solves of the chain must be met in this solve
    # (Q3).  One row per entity over the chain horizon, replacing the old per-period "invested and
    # still alive" row; edd_invest_set is no longer read.  Always named _path, even with one
    # scenario leaf: a single row summed over alternative branches would let one branch meet the
    # minimum for all, so the per-leaf row shape is load-bearing, not cosmetic (c16a critique #2).
    for name, var, axis in (("minInvest_entity_total_path_p", b.v_invest_p, "p"),
                            ("minInvest_entity_total_path_n", b.v_invest_n, "n")):
        unitsize = _unitsize(d, axis)
        if var is None or unitsize is None:
            continue
        members = _entities_of_kind(d, e_set, axis).pipe(rename_to_axis, {"e": axis})
        if members.height == 0:
            continue
        over_e = members.join(var.frame.select(axis).unique(), on=axis, how="semi")
        if over_e.height == 0:
            continue
        cap_param = Param((axis,), minus_prior(cap.frame.pipe(rename_to_axis, {"e": axis}), axis, prior))
        terms = over_e.join(var.frame.select(axis, "d").unique(), on=axis, how="inner")
        _, over, mask = leaf_rows(terms, axis, var, d_leaf)
        b.m.add_cstr(
            name,
            over      = over,
            sense     = ">=",
            lhs_terms = {"invest_total": Sum(Where(var * unitsize, mask), over=("d",))},
            rhs_terms = {"cap":          cap_param},
        )


def _eq_min_divest_total(b: _Build) -> None:
    """minDivest_entity_total_path_{p,n}: capacity retired over the chain, from below."""
    d = b.d
    e_set, cap = d.e_divest_total, d.e_divest_min_total
    if not _has_rows(e_set) or cap is None:
        return
    d_leaf = getattr(d, "d_leaf", None)
    prior = chain_prior(d, "divest")

    # ─── EQ. ─── minDivest_entity_total_path_{p,n} ───────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{d ∈ D(leaf)} v_divest_p[p,d] · unitsize[p]  ≥  max(0, divest_min_total[p] − prior_div[p])
    #   ∀ p ∈ e_divest_total ∩ P(v_divest_p) with divest_min_total (, leaf)
    #   prior_div[p] = p_entity_divested[p]  (capacity retired by earlier solves of the chain, at
    #     realized non-branch periods only)
    #   _n: the node analogue with v_divest_n and state_unitsize[n]
    # Forces the part of the minimum not retired by earlier solves of the chain to be retired in
    # this solve (Q3).  Always named _path, for the same reason as minInvest (c16a critique #2).
    for name, var, axis in (("minDivest_entity_total_path_p", b.v_divest_p, "p"),
                            ("minDivest_entity_total_path_n", b.v_divest_n, "n")):
        unitsize = _unitsize(d, axis)
        if var is None or unitsize is None:
            continue
        members = _entities_of_kind(d, e_set, axis).pipe(rename_to_axis, {"e": axis})
        if members.height == 0:
            continue
        over_e = members.join(var.frame.select(axis).unique(), on=axis, how="semi")
        if over_e.height == 0:
            continue
        cap_param = Param((axis,), minus_prior(cap.frame.pipe(rename_to_axis, {"e": axis}), axis, prior))
        terms = over_e.join(var.frame.select(axis, "d").unique(), on=axis, how="inner")
        _, over, mask = leaf_rows(terms, axis, var, d_leaf)
        b.m.add_cstr(
            name,
            over      = over,
            sense     = ">=",
            lhs_terms = {"divest_total": Sum(Where(var * unitsize, mask), over=("d",))},
            rhs_terms = {"cap":          cap_param},
        )


def _eq_cumulative_capacity(b: _Build) -> None:
    """{max,min}Cumulative_capacity_{p,n}: total capacity of an entity per period, from above and below."""
    d = b.d
    idx, edd = d.ed_invest_cumulative, d.edd_invest_set
    if not _has_rows(idx) or edd is None:
        return
    existing = d.p_entity_all_existing

    # ─── EQ. ─── {max,min}Cumulative_capacity_{p,n} ──────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{d′ : (p,d′,d) ∈ edd_invest_set} v_invest_p[p,d′] · unitsize[p]
    #   − Σ_{d′ : (p,d′,d) ∈ edd_divest_active} v_divest_p[p,d′] · unitsize[p]
    #   ≤ cumulative_max_capacity[p,d] − existing[p,d]        (maxCumulative_capacity_p)
    #   ≥ cumulative_min_capacity[p,d] − existing[p,d]        (minCumulative_capacity_p)
    #   max: ∀ (p,d) ∈ ed_invest_cumulative ∩ v_invest_p entities, p ∈ authors(cumulative_max_capacity) —
    #        a period the author's own map leaves out is a configuration error (FA-12), not "no limit"
    #        (c16b critique change 2)
    #   min: ∀ (p,d) ∈ ed_invest_cumulative ∩ v_invest_p entities ∩ cumulative_min_capacity.frame
    #   _n: the node analogue with v_invest_n / v_divest_n and state_unitsize[n] — ``edd_divest_active``
    #       is built only from process divestments (``pd_divest_set``; no node equivalent exists), so
    #       the _n rows keep subtracting only the row's own period's divestment (pre-existing
    #       limitation, unchanged by #29; tracking node divest history needs its own data plumbing).
    # Keeps the capacity of an entity in a period (existing + invested and still alive − every
    # retirement up to and including d, #29, process axis only) within its cumulative limits,
    # MW (MWh for storage nodes).  Existing capacity is a constant on the right-hand side.
    for name, sense, cap, var, divest, axis, param_name in (
        ("maxCumulative_capacity_p", "<=", d.ed_cumulative_max_capacity, b.v_invest_p, b.v_divest_p, "p",
         "cumulative_max_capacity"),
        ("maxCumulative_capacity_n", "<=", d.ed_cumulative_max_capacity, b.v_invest_n, b.v_divest_n, "n",
         "cumulative_max_capacity"),
        ("minCumulative_capacity_p", ">=", d.ed_cumulative_min_capacity, b.v_invest_p, b.v_divest_p, "p", None),
        ("minCumulative_capacity_n", ">=", d.ed_cumulative_min_capacity, b.v_invest_n, b.v_divest_n, "n", None),
    ):
        unitsize = _unitsize(d, axis)
        if cap is None or var is None or unitsize is None:
            continue
        all_rows = _on_var_axis(idx, axis, var)
        if all_rows.height == 0:
            continue
        cap_frame = cap.frame.rename({"e": axis})
        rows_src, cap_dt = align_join_dtypes(all_rows, cap_frame, (axis,))
        if sense == "<=":
            # Max rows: entities that author the cap at all (any period), restricted to periods the
            # entity actually has the variable for; FA-12 then catches a period that same entity's
            # own map leaves out (c16b critique change 1).
            rows = rows_src.join(cap_dt.select(axis).unique(), on=axis, how="semi")
            if rows.height == 0:
                continue
            check_max_cap(rows, cap_dt, axis, True, param_name=param_name)
        else:
            # Min rows: only the periods an explicit value is given for (a missing min is simply "no
            # floor", not an error).
            rows = rows_src.join(cap_dt.select(axis, "d").unique(), on=[axis, "d"], how="semi")
            if rows.height == 0:
                continue
        lifetime, rows_aligned = align_join_dtypes(edd.rename({"e": axis}), rows, (axis,))
        lifetime = lifetime.join(rows_aligned, on=[axis, "d"], how="inner")
        invest = _var_view(var, "__cumcap_at", {"d": "d_invest"})
        lhs: dict = {"invest_cum": Sum(Where(invest * unitsize, lifetime), over=("d_invest",))}
        if divest is not None:
            if axis == "p" and d.edd_divest_active is not None:
                # #29: every earlier divestment still active at d, not just d's own — the maxFlow
                # measure (core.py divest_in_dispatch_p).  ``edd_divest_active`` only ever carries
                # process rows (built solely from ``pd_divest_set``), so this branch is process-only.
                view = _var_view(divest, "__cumcap_div", {"d": "d_divest"}, to_axis=True)
                active = _entity_rows(d.edd_divest_active, "p", rows, axis)
                active, rows_div = align_join_dtypes(active, rows.select(axis, "d"), (axis, "d"))
                active = active.join(rows_div, on=[axis, "d"], how="semi")
                lhs["divest"] = -Sum(Where(view * unitsize, active), over=("d_divest",))
            else:
                lhs["divest"] = -Where(divest * unitsize, rows)
        rhs: dict = {"cap": Param((axis, "d"), cap_frame)}
        if existing is not None:
            rhs["minus_existing"] = -Param((axis, "d"), existing.frame.rename({"e": axis}))
        b.m.add_cstr(
            name,
            over      = rows,
            sense     = sense,
            lhs_terms = lhs,
            rhs_terms = rhs,
        )


def _eq_group_entity_period(b: _Build) -> None:
    """{max,min}{Invest,Divest}Group_entity_period_{p,n}: what a group builds or retires per period."""
    d = b.d
    group_entity = d.group_entity
    if not _has_rows(group_entity):
        return

    # ─── EQ. ─── {max,min}{Invest,Divest}Group_entity_period_{p,n} ───────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{p : (g,p) ∈ group_entity} v_invest_p[p,d] · unitsize[p]  ≤ group_invest_max_period[g,d]   (max…_p)
    #                                                             ≥ group_invest_min_period[g,d]   (min…_p)
    #   ∀ (g,d) ∈ gd_invest_period
    #   Divest: v_divest_p with group_retire_{max,min}_period over gd_divest_period
    #   _n: the node analogues with v_invest_n / v_divest_n and state_unitsize[n]
    # Bounds the capacity a group of entities builds or retires in one period, MW (MWh for storage
    # nodes); processes and storage nodes of a group get separate rows.  Group retire_{max,min}_period is
    # not a schema parameter, so the divest rows are reachable only with off-schema data.
    for kind, sense, gd_idx, cap, sides, param_name in (
        ("invest", "<=", d.gd_invest_period, d.p_group_invest_max_period,
         (("maxInvestGroup_entity_period_p", b.v_invest_p, "p"),
          ("maxInvestGroup_entity_period_n", b.v_invest_n, "n")), "invest_max_period"),
        ("invest", ">=", d.gd_invest_period, d.p_group_invest_min_period,
         (("minInvestGroup_entity_period_p", b.v_invest_p, "p"),
          ("minInvestGroup_entity_period_n", b.v_invest_n, "n")), None),
        ("divest", "<=", d.gd_divest_period, d.p_group_retire_max_period,
         (("maxDivestGroup_entity_period_p", b.v_divest_p, "p"),
          ("maxDivestGroup_entity_period_n", b.v_divest_n, "n")), "retire_max_period"),
        ("divest", ">=", d.gd_divest_period, d.p_group_retire_min_period,
         (("minDivestGroup_entity_period_p", b.v_divest_p, "p"),
          ("minDivestGroup_entity_period_n", b.v_divest_n, "n")), None),
    ):
        if not _has_rows(gd_idx) or cap is None:
            continue
        for name, var, axis in sides:
            unitsize = _unitsize(d, axis)
            if var is None or unitsize is None:
                continue
            members = group_entity.rename({"e": axis}).filter(pl.col(axis).is_in(var.frame[axis].unique()))
            if members.height == 0:
                continue
            terms = (members.join(var.frame.select(axis, "d"), on=axis, how="inner")
                     .join(gd_idx, on=["g", "d"], how="inner"))
            if terms.height == 0:
                continue
            if sense == "<=":
                # FA-12, c16a critique #4: only on (g, d) whose members have the variable.
                check_max_cap(terms.select("g", "d").unique(), cap.frame, "g", True,
                              param_name=param_name)
            b.m.add_cstr(
                name,
                over      = gd_idx,
                sense     = sense,
                lhs_terms = {kind:  Sum(Where(var * unitsize, terms.select("g", axis, "d")), over=(axis,))},
                rhs_terms = {"cap": cap},
            )


def _eq_group_invest_total(b: _Build) -> None:
    """{max,min}InvestGroup_entity_total[_path]_{p,n}: capacity a group has invested over the chain."""
    d = b.d
    group_entity, g_set = d.group_entity, d.g_invest_total
    if not _has_rows(group_entity) or not _has_rows(g_set):
        return
    d_leaf = getattr(d, "d_leaf", None)
    prior = chain_prior(d, "invest")

    # ─── EQ. ─── {max,min}InvestGroup_entity_total[_path]_{p,n} ─────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{p:(g,p)∈group_entity} Σ_{d∈D(leaf)} v_invest_p[p,d]·unitsize[p]
    #   ≤ max(0, group_invest_max_total[g] − Σ_{p∈g} prior_inv[p])   (maxInvestGroup_entity_total[_path]_p)
    #   ≥ max(0, group_invest_min_total[g] − Σ_{p∈g} prior_inv[p])   (minInvestGroup_entity_total_path_p)
    #   ∀ g ∈ g_invest_total (, leaf)
    #   prior_inv[p] = p_entity_invested[p]  (capacity realized by all earlier solves of the
    #     chain); the group's prior sums every member of the axis kind, not only those with the
    #     variable in THIS solve — their earlier investment still counts (c16a design §2)
    #   _n: storage-node members, state_unitsize[n]
    #   max: _path (one row per (g, leaf)) only under recourse, like the entity max side.
    #   min: always _path — a plain per-group row, summed across alternative branches, would let
    #     one branch meet the floor for every branch (Q3, same reason as the entity min side)
    # Changed from per-(g,d) "invested and still alive" rows to one row per group (per leaf);
    # processes and storage nodes of a group get separate rows.
    for sense, cap, var, axis, legacy_name, path_name, always_path, param_name in (
        ("<=", d.p_group_invest_max_total, b.v_invest_p, "p",
         "maxInvestGroup_entity_total_p", "maxInvestGroup_entity_total_path_p", False,
         "invest_max_total"),
        ("<=", d.p_group_invest_max_total, b.v_invest_n, "n",
         "maxInvestGroup_entity_total_n", "maxInvestGroup_entity_total_path_n", False,
         "invest_max_total"),
        (">=", d.p_group_invest_min_total, b.v_invest_p, "p",
         None, "minInvestGroup_entity_total_path_p", True, None),
        (">=", d.p_group_invest_min_total, b.v_invest_n, "n",
         None, "minInvestGroup_entity_total_path_n", True, None),
    ):
        unitsize = _unitsize(d, axis)
        if cap is None or var is None or unitsize is None:
            continue
        members = (_entities_of_kind(d, group_entity, axis).pipe(rename_to_axis, {"e": axis})
                   .filter(pl.col("g").is_in(g_set["g"].unique())))
        if members.height == 0:
            continue
        terms = members.join(var.frame.select(axis, "d").unique(), on=axis, how="inner")
        if terms.height == 0:
            continue
        if sense == "<=":
            # FA-12, c16a critique #4: only on g whose members have the variable.
            check_max_cap(terms.select("g").unique(), cap.frame, "g", False, param_name=param_name)
        cap_g = Param(("g",), minus_prior(cap.frame, "g", group_prior(group_entity, prior, d, axis)))
        is_path, over, mask = leaf_rows(terms, "g", var, d_leaf)
        name = path_name if (always_path or is_path) else legacy_name
        b.m.add_cstr(
            name,
            over      = over,
            sense     = sense,
            lhs_terms = {"invest_grp": Sum(Where(var * unitsize, mask), over=(axis, "d"))},
            rhs_terms = {"cap":        cap_g},
        )


def _eq_group_divest_total(b: _Build) -> None:
    """{max,min}DivestGroup_entity_total[_path]_{p,n}: capacity a group retires over the chain."""
    d = b.d
    group_entity, g_set = d.group_entity, d.g_divest_total
    if not _has_rows(group_entity) or not _has_rows(g_set):
        return
    d_leaf = getattr(d, "d_leaf", None)
    prior = chain_prior(d, "divest")

    # ─── EQ. ─── {max,min}DivestGroup_entity_total[_path]_{p,n} ─────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{p:(g,p)∈group_entity} Σ_{d∈D(leaf)} v_divest_p[p,d]·unitsize[p]
    #   ≤ max(0, group_retire_max_total[g] − Σ_{p∈g} prior_div[p])   (maxDivestGroup_entity_total[_path]_p)
    #   ≥ max(0, group_retire_min_total[g] − Σ_{p∈g} prior_div[p])   (minDivestGroup_entity_total_path_p)
    #   ∀ g ∈ g_divest_total (, leaf)
    #   prior_div[p] = p_entity_divested[p]  (capacity retired by earlier solves of the chain, at
    #     realized non-branch periods only); the group's prior sums every member of the axis kind
    #   _n: storage-node members, state_unitsize[n]
    #   max: _path (one row per (g, leaf)) only under recourse, as before — the cap applies to
    #     each scenario path instead of adding up divestments of alternative branches.
    #   min: always _path (new — Q3, same reason as every other min-total family: a plain
    #     per-group row, summed across alternative branches, would let one branch meet the floor
    #     for every branch)
    # Bounds the capacity a group retires over the chain; processes and storage nodes of a group
    # get separate rows.  Group retire_{max,min}_total is not a schema parameter, so these rows
    # are reachable only with off-schema data.
    for sense, cap, var, axis, legacy_name, path_name, always_path, param_name in (
        ("<=", d.p_group_retire_max_total, b.v_divest_p, "p",
         "maxDivestGroup_entity_total_p", "maxDivestGroup_entity_total_path_p", False,
         "retire_max_total"),
        ("<=", d.p_group_retire_max_total, b.v_divest_n, "n",
         "maxDivestGroup_entity_total_n", "maxDivestGroup_entity_total_path_n", False,
         "retire_max_total"),
        (">=", d.p_group_retire_min_total, b.v_divest_p, "p",
         None, "minDivestGroup_entity_total_path_p", True, None),
        (">=", d.p_group_retire_min_total, b.v_divest_n, "n",
         None, "minDivestGroup_entity_total_path_n", True, None),
    ):
        unitsize = _unitsize(d, axis)
        if cap is None or var is None or unitsize is None:
            continue
        members = (_entities_of_kind(d, group_entity, axis).pipe(rename_to_axis, {"e": axis})
                   .filter(pl.col("g").is_in(g_set["g"].unique())))
        if members.height == 0:
            continue
        terms = members.join(var.frame.select(axis, "d").unique(), on=axis, how="inner")
        if terms.height == 0:
            continue
        if sense == "<=":
            # FA-12, c16a critique #4: only on g whose members have the variable.
            check_max_cap(terms.select("g").unique(), cap.frame, "g", False, param_name=param_name)
        cap_g = Param(("g",), minus_prior(cap.frame, "g", group_prior(group_entity, prior, d, axis)))
        is_path, over, mask = leaf_rows(terms, "g", var, d_leaf)
        name = path_name if (always_path or is_path) else legacy_name
        b.m.add_cstr(
            name,
            over      = over,
            sense     = sense,
            lhs_terms = {"divest_grp": Sum(Where(var * unitsize, mask), over=(axis, "d"))},
            rhs_terms = {"cap":        cap_g},
        )


def _eq_group_invest_cumulative(b: _Build) -> None:
    """{max,min}InvestGroup_entity_cumulative: existing plus invested capacity of a group per period."""
    d = b.d
    group_entity, g_set, edd = d.group_entity, d.g_invest_cumulative, d.edd_invest_set
    if not _has_rows(group_entity) or not _has_rows(g_set) or edd is None:
        return
    in_groups = pl.col("g").is_in(g_set["g"].unique())
    # (g, d): the periods in which some member of the group has investments alive.
    base_over = (group_entity.rename({"e": "_e"}).filter(in_groups)
                 .join(edd.rename({"e": "_e"}), on="_e", how="inner")
                 .select("g", "d").unique())
    if base_over.height == 0:
        return

    # ─── EQ. ─── {max,min}InvestGroup_entity_cumulative ──────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{p : (g,p) ∈ group_entity} Σ_{d′ : (p,d′,d) ∈ edd_invest_set} v_invest_p[p,d′] · unitsize[p]
    #   + Σ_{n : (g,n) ∈ group_entity} Σ_{d′ : (n,d′,d) ∈ edd_invest_set} v_invest_n[n,d′] · state_unitsize[n]
    #   ≤ cumulative_max_capacity[g,d] − Σ_{e : (g,e) ∈ group_entity} existing[e,d]   (max…)
    #   ≥ cumulative_min_capacity[g,d] − Σ_{e : (g,e) ∈ group_entity} existing[e,d]   (min…)
    #   ∀ g ∈ g_invest_cumulative (invest_method = cumulative_limits, #5/B-Q3), d with a cap value
    #   and some member alive in edd_invest_set
    # Keeps the capacity of a group in a period (existing + invested and still alive) within its
    # cumulative limits (schema "[MW or MWh]"); processes and storage nodes share one row.  Rows are
    # restricted to (g, d) the cap's own map actually covers (#5: now a real period map, not a
    # dummy scalar) — a period the map leaves out has no limit, matching B-Q2's default for this
    # family (no FA-12 check here, unlike the entity-level cumulative row).  Divestments are not yet
    # counted (#30b, next commit); capacity invested by earlier solves of a chain is not counted
    # either (pre-existing, separate item).
    for name, sense, cap in (("maxInvestGroup_entity_cumulative", "<=", d.pd_group_cumulative_max_capacity),
                             ("minInvestGroup_entity_cumulative", ">=", d.pd_group_cumulative_min_capacity)):
        if cap is None:
            continue
        cap_frame, over = align_join_dtypes(cap.frame, base_over, ("g",))
        over = over.join(cap_frame.select("g", "d").unique(), on=["g", "d"], how="semi")
        if over.height == 0:
            continue
        lhs: dict = {}
        for key, var, axis in (("invest_p", b.v_invest_p, "p"), ("invest_n", b.v_invest_n, "n")):
            unitsize = _unitsize(d, axis)
            if var is None or unitsize is None:
                continue
            terms = (_on_var_axis(group_entity, axis, var).filter(in_groups)
                     .join(_on_var_axis(edd, axis, var), on=axis, how="inner"))
            if terms.height == 0:
                continue
            invest = _var_view(var, "__grp_cum_at", {"d": "d_invest"})
            lhs[key] = Sum(Where(invest * unitsize, terms), over=(axis, "d_invest"))
        if not lhs:
            continue
        rhs: dict = {"cap": cap}
        if d.p_entity_all_existing is not None:
            group_existing = (d.p_entity_all_existing.frame
                              .join(group_entity, on="e", how="inner")
                              .group_by(["g", "d"]).agg(pl.col("value").sum())
                              .filter(in_groups))
            if group_existing.height > 0:
                rhs["minus_existing"] = -Param(("g", "d"), group_existing.select("g", "d", "value"))
        b.m.add_cstr(
            name,
            over      = over,
            sense     = sense,
            lhs_terms = lhs,
            rhs_terms = rhs,
        )


def _eq_cumulative_flow_solve(b: _Build) -> None:
    """{max,min}Cumulative_flow_solve: net flow into a flow group summed over the solve."""
    d = b.d
    flow = _group_flow(b)
    if flow is None:
        return
    hours_in_solve = float(d.p_step_duration.frame["value"].sum())

    # ─── EQ. ─── {max,min}Cumulative_flow_solve ──────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(d,t)} F[g,d,t] · Δt[d,t]  ≤ max_cumulative_flow[g] · Σ_{(d,t)} Δt[d,t]   (max…)
    #                               ≥ min_cumulative_flow[g] · Σ_{(d,t)} Δt[d,t]   (min…)
    #   ∀ g with a max / min_cumulative_flow
    #   F = net flow into the group (_group_flow), Δt = step_duration
    # Limits the energy flowing into a flow group over the solve to an average power times the hours of
    # the solve.  The steps are weighted by step_duration only (no rp or years-represented weights);
    # eff source flows carry the slope but no conversion_flow_coeff, and the min-load section input is
    # not counted (known limitation).
    for name, sense, cap in (("maxCumulative_flow_solve", "<=", d.p_group_max_cumulative_flow),
                             ("minCumulative_flow_solve", ">=", d.p_group_min_cumulative_flow)):
        if cap is None or cap.frame.height == 0:
            continue
        b.m.add_cstr(
            name,
            over      = cap.frame.select("g").unique(),
            sense     = sense,
            lhs_terms = {"flow_int": Sum(flow * d.p_step_duration, over=("d", "t"))},
            rhs_terms = {"cap":      Param(("g",), cap.frame.with_columns(value=pl.col("value") * hours_in_solve))},
        )


def _eq_cumulative_flow_period(b: _Build) -> None:
    """{max,min}Cumulative_flow_period: net flow into a flow group summed over each period."""
    d = b.d
    flow = _group_flow(b)
    if flow is None:
        return
    hours_in_period = (d.p_step_duration.frame
                       .group_by("d").agg(pl.col("value").sum())
                       .rename({"value": "hours_in_period"}))

    # ─── EQ. ─── {max,min}Cumulative_flow_period ─────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_t F[g,d,t] · Δt[d,t]  ≤ max_cumulative_flow[g,d] · Σ_t Δt[d,t]   (max…)
    #                         ≥ min_cumulative_flow[g,d] · Σ_t Δt[d,t]   (min…)
    #   ∀ (g,d) with a max / min_cumulative_flow
    #   F = net flow into the group (_group_flow), Δt = step_duration
    # Limits the energy flowing into a flow group in each period to an average power times the hours of
    # the period.  Same weighting and flow terms as the _solve rows.
    for name, sense, cap in (("maxCumulative_flow_period", "<=", d.pd_max_cumulative_flow),
                             ("minCumulative_flow_period", ">=", d.pd_min_cumulative_flow)):
        if cap is None or cap.frame.height == 0:
            continue
        rhs = (cap.frame.join(hours_in_period, on="d", how="inner")
               .with_columns(value=pl.col("value") * pl.col("hours_in_period"))
               .select("g", "d", "value"))
        b.m.add_cstr(
            name,
            over      = cap.frame.select("g", "d").unique(),
            sense     = sense,
            lhs_terms = {"flow_int": Sum(flow * d.p_step_duration, over=("t",))},
            rhs_terms = {"cap":      Param(("g", "d"), rhs)},
        )


def _eq_instant_flow(b: _Build) -> None:
    """{max,min}Instant_flow: net flow into a flow group in every step."""
    d = b.d
    flow = _group_flow(b)
    if flow is None:
        return
    dt = d.p_step_duration.frame.select("d", "t").unique()

    # ─── EQ. ─── {max,min}Instant_flow ───────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # F[g,d,t]  ≤ max_instant_flow[g,d,t]   (maxInstant_flow)
    #           ≥ min_instant_flow[g,d,t]   (minInstant_flow)
    #   ∀ (g,d,t) ∈ _instant_flow_support(cap, dt)
    #   F = net flow into the group (_group_flow), MW
    # Limits the net power flowing into a flow group in every step.  The cap may be authored as a
    # scalar, period map, time map or period-time map; the rows cover every active step of its shape.
    for name, sense, cap in (("maxInstant_flow", "<=", d.pdt_max_instant_flow),
                             ("minInstant_flow", ">=", d.pdt_min_instant_flow)):
        if cap is None or cap.frame.height == 0:
            continue
        over = _instant_flow_support(cap.frame, dt)
        if over.height == 0:
            continue
        b.m.add_cstr(
            name,
            over      = over,
            sense     = sense,
            lhs_terms = {"flow": flow},
            rhs_terms = {"cap":  cap},
        )
