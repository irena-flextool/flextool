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
from flextool.engine.input._param_shapes import drop_no_limit_rows, without_no_limit_rows
from flextool.engine.model import _coarse_flows, _lead_time, _retire_built
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


def _on_var_axis(frame: pl.DataFrame, axis: str, var: "Var | pl.DataFrame") -> pl.DataFrame:
    """``frame`` (keyed on the entity-union axis e) renamed to ``axis`` and restricted to the entities of ``var``
    (a variable, or the ``[axis, d]`` cells of the retirement columns, _retire_built.retirement_cells).

    The entity-union Enum is cast to the narrower process / node Enum of ``var`` first, so that the
    membership test compares matching dtypes.
    """
    cells = var if isinstance(var, pl.DataFrame) else var.frame
    out = frame.rename({"e": axis})
    dtype = cells.schema[axis]
    if out.schema[axis] != dtype:
        out = out.with_columns(pl.col(axis).cast(dtype, strict=False))
    return out.filter(pl.col(axis).is_in(cells[axis].unique()))


def _group_flow(b: _Build) -> Expr | None:
    """Net flow into each flow group, MW, indexed (g, d, t); None when no group sees a flow.

    F[g,d,t] = Σ_{(p,s,k,n) ∈ flow_to_n, (g,p,n) ∈ gpn} v_flow[p,s,k,d,t] · unitsize[p]
             − Σ_{(p,s,k,n) ∈ flow_from_n ∩ pss_eff, (g,p,n) ∈ gpn}
                 (v_flow[p,s,k,d,t]·unitsize[p]·slope[p,d,t]
                  + v_online[p,d,t]·section[p,d,t]·unitsize[p])   [min_load_efficiency units only]
             − Σ_{(p,s,k,n) ∈ flow_from_n ∩ pss_noEff, (g,p,n) ∈ gpn} v_flow[p,s,k,d,t] · unitsize[p]
      gpn = group_process_node;  flow_to_n / flow_from_n carry the sink / source node as n
    The eff-source term is the same fuel measure as ``nodeBalance_eq`` and the CO2/commodity cost
    terms (``_eff_fuel_input``): eff sources carry no ``conversion_flow_coeff``, and a
    min_load_efficiency unit's section fuel (the units online, linear or integer) is drawn too.
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
            from flextool.engine.model.core import _eff_fuel_input

            for _key, term in _eff_fuel_input(b, arcs):
                parts.append(-Sum(term, over=("p", "source", "sink", "n")))
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
    _eq_min_entity_period(b)
    _eq_min_invest_total(b)
    _eq_min_divest_total(b)
    _eq_cumulative_capacity(b)
    _check_group_retire_minimums(b)
    _eq_group_entity_period(b)
    _eq_group_invest_total(b)
    _eq_group_divest_total(b)
    _eq_group_invest_cumulative(b)
    _eq_cumulative_flow_solve(b)
    _eq_cumulative_flow_period(b)
    _eq_instant_flow(b)


def _check_group_retire_minimums(b: _Build) -> None:
    """A group's minimum retirement (``retire_min_total`` / ``retire_min_period``) must fit in what its members can
    retire: their pre-existing capacity (``maxRetired_existing_{p,n}``) and their retirable model-built capacity
    (``maxDivest_cohort_{p,n}``, up to its cohorts' bounds), so a larger minimum is a configuration error, not an
    infeasible LP."""
    d = b.d
    if (getattr(d, "p_group_retire_min_total", None) is None
            and getattr(d, "p_group_retire_min_period", None) is None):
        return
    from flextool.engine.preprocess.params._retire_built import check_minimums_inputs
    from flextool.engine.preprocess.params._retire_existing import (
        active_windows,
        check_group_retirement_minimums,
        ed_values,
    )
    _, built_total, built_alive = check_minimums_inputs(getattr(d, "cohort_divest_built", None),
                                                        getattr(d, "cd_divest_built", None))
    check_group_retirement_minimums(
        group_entity=d.group_entity,
        min_total=d.p_group_retire_min_total,
        min_period=d.p_group_retire_min_period,
        windows=active_windows([b.cap.retired_frame("p"), b.cap.retired_frame("n")]),
        pre=ed_values(b.cap.existing_cohort_size_e),
        prior=chain_prior(d, "divest"),
        carried=ed_values(getattr(d, "p_entity_retired_carried", None)),
        built_total=built_total, built_alive=built_alive)


def _eq_min_entity_period(b: _Build) -> None:
    """min{Invest,Divest}_entity_period_{p,n}: capacity built or retired per period, from below."""
    d = b.d
    if d.ed_invest_min_period is None and d.ed_divest_min_period is None:
        return

    # ─── EQ. ─── min{Invest,Divest}_entity_period_{p,n} ──────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,o,d) ∈ edd_invest_commission} v_invest_p[p,o] · unitsize[p] + carried_commission[p,d]
    #                                      ≥ invest_min_period[p,d]   ∀ (p,d) with a decision coming online in d,
    #                                                                   p ∈ ed_invest_period_set
    # (n: v_invest_n with state_unitsize[n], n ∈ ed_invest_period_set ∩ nd_invest_set)
    # R_p[p,d] · unitsize[p]        ≥ divest_min_period[p,d]   ∀ (p,d) ∈ ed_divest_period_set ∩ (pd_divest_set ∪ B_p)
    # R_n[n,d] · state_unitsize[n]  ≥ divest_min_period[n,d]   ∀ (n,d) ∈ ed_divest_period_set ∩ (nd_divest_set ∪ B_n)
    #   R[e,d] = v_divest[e,d] + Σ_c v_divest_built[e,c,d] (existing and model-built capacity retired in d);
    #   B = the (e,d) of cd_divest_built (_retire_built)
    # Forces a minimum capacity to come online or be retired in a period; the floor twin of
    # max{Invest,Divest}_entity_period in model/core.py.  Capacity in MW (MWh for storage nodes).  Without a
    # construction lead time c(o) = o: v_invest_p[p,d] · unitsize[p] ≥ invest_min_period[p,d] over
    # ed_invest_period_set ∩ pd_invest_set.  With one, a period of an entity with a lead time where nothing can
    # come online (no decision early enough in the solve) and the carried capacity is below the floor is a
    # configuration error.
    for name, period_set, cap, var, axis, kind in (
        ("minInvest_entity_period_p", d.ed_invest_period_set, d.ed_invest_min_period, b.v_invest_p, "p", "invest"),
        ("minInvest_entity_period_n", d.ed_invest_period_set, d.ed_invest_min_period, b.v_invest_n, "n", "invest"),
        ("minDivest_entity_period_p", d.ed_divest_period_set, d.ed_divest_min_period, b.v_divest_p, "p", "divest"),
        ("minDivest_entity_period_n", d.ed_divest_period_set, d.ed_divest_min_period, b.v_divest_n, "n", "divest"),
    ):
        unitsize = _unitsize(d, axis)
        # A retirement floor also counts the retirements of model-built capacity (_retire_built).
        built = _retire_built.decided(b, axis) if kind == "divest" else None
        if not _has_rows(period_set) or cap is None or (var is None and built is None) or unitsize is None:
            continue
        if kind == "invest" and _lead_time.active(d):
            entities = _on_var_axis(period_set, axis, var).join(var.frame.select(axis, "d"), on=[axis, "d"], how="semi")
            _lead_time.period_limit(b, name, axis, var, unitsize, entities, cap, ">=", "invest_min_period")
            continue
        if built is None:
            over = _on_var_axis(period_set, axis, var).join(var.frame.select(axis, "d"), on=[axis, "d"], how="inner")
            lhs: dict = {kind: var * unitsize}
        else:
            cells = _retire_built.retirement_cells(b, axis, var)
            over = _on_var_axis(period_set, axis, cells).join(cells, on=[axis, "d"], how="inner")
            lhs = {kind: var * unitsize} if var is not None else {}
            lhs["divest_built"] = built * unitsize
        if over.height == 0:
            continue
        b.m.add_cstr(
            name,
            over      = over,
            sense     = ">=",
            lhs_terms = lhs,
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
    # still alive" row; the capacity in service is not read.  Always named _path, even with one
    # scenario leaf: a single row summed over alternative branches would let one branch meet the
    # minimum for all, so the per-leaf row shape is load-bearing, not cosmetic.
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
    # Σ_{d ∈ D(leaf)} R_p[p,d] · unitsize[p]  ≥  max(0, divest_min_total[p] − prior_div[p])
    #   ∀ p ∈ e_divest_total ∩ P(v_divest_p ∪ v_divest_built_p) with divest_min_total (, leaf)
    #   R[e,d] = v_divest[e,d] + Σ_c v_divest_built[e,c,d] (existing and model-built capacity retired in d)
    #   prior_div[p] = p_entity_divested[p]  (capacity retired by earlier solves of the chain, at
    #     realized non-branch periods only)
    #   _n: the node analogue with v_divest_n and state_unitsize[n]
    # Forces the part of the minimum not retired by earlier solves of the chain to be retired in
    # this solve (Q3).  Always named _path, for the same reason as minInvest.
    for name, var, axis in (("minDivest_entity_total_path_p", b.v_divest_p, "p"),
                            ("minDivest_entity_total_path_n", b.v_divest_n, "n")):
        unitsize = _unitsize(d, axis)
        built = _retire_built.decided(b, axis)
        if (var is None and built is None) or unitsize is None:
            continue
        members = _entities_of_kind(d, e_set, axis).pipe(rename_to_axis, {"e": axis})
        if members.height == 0:
            continue
        cells = var.frame if built is None else _retire_built.retirement_cells(b, axis, var)
        over_e = members.join(cells.select(axis).unique(), on=axis, how="semi")
        if over_e.height == 0:
            continue
        cap_param = Param((axis,), minus_prior(cap.frame.pipe(rename_to_axis, {"e": axis}), axis, prior))
        terms = over_e.join(cells.select(axis, "d").unique(), on=axis, how="inner")
        _, over, mask = leaf_rows(terms, axis, var if built is None else cells, d_leaf)
        lhs: dict = {"divest_total": Sum(Where(var * unitsize, mask), over=("d",))} if var is not None else {}
        if built is not None:
            lhs["divest_total_built"] = Sum(Where(built * unitsize, mask), over=("d",))
        b.m.add_cstr(
            name,
            over      = over,
            sense     = ">=",
            lhs_terms = lhs,
            rhs_terms = {"cap":          cap_param},
        )


def _eq_cumulative_capacity(b: _Build) -> None:
    """{max,min}Cumulative_capacity_{p,n}: total capacity of an entity per period, from above and below."""
    d = b.d
    idx, edd = d.ed_invest_cumulative, b.cap.alive
    if not _has_rows(idx) or not _has_rows(edd):
        return
    existing = b.cap.existing_mw_e

    # ─── EQ. ─── {max,min}Cumulative_capacity_{p,n} ──────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # invested[p,d] · unitsize[p]
    #   − retired[p,d] · unitsize[p]
    #   ≤ cumulative_max_capacity[p,d] − existing[p,d]        (maxCumulative_capacity_p)
    #   ≥ cumulative_min_capacity[p,d] − existing[p,d]        (minCumulative_capacity_p)
    #   max: ∀ (p,d) ∈ ed_invest_cumulative ∩ v_invest_p entities, p ∈ authors(cumulative_max_capacity) —
    #        a period the author's own map leaves out is a configuration error (FA-12), not "no limit";
    #        a period whose cumulative_max_capacity is inf has no row (no limit there)
    #       
    #   min: ∀ (p,d) ∈ ed_invest_cumulative ∩ v_invest_p entities ∩ cumulative_min_capacity.frame
    #   _n: the node analogue with v_invest_n / v_divest_n and state_unitsize[n].
    # Keeps the capacity of an entity in a period (existing + invested and still alive − every
    # retirement up to and including d, #29, processes and storage nodes) within its cumulative
    # limits, MW (MWh for storage nodes).  Existing capacity is a constant on the right-hand side.
    # Capacity in service: invested[e,d], retired[e,d] (and prebuilt[e,d]) of _capacity, the retirements of
    # existing capacity counting until its end of life (none for reinvest_automatic).
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
            # own map leaves out.
            rows = rows_src.join(cap_dt.select(axis).unique(), on=axis, how="semi")
            if rows.height == 0:
                continue
            check_max_cap(rows, cap_dt, axis, True, param_name=param_name)
            # A period with cumulative_max_capacity = inf has no row (no limit there).
            rows = without_no_limit_rows(rows, Param((axis, "d"), cap_frame), "upper")
            if rows.height == 0:
                continue
        else:
            # Min rows: the periods with a value other than 0 (a min of 0 is no floor).  A period
            # map that leaves out an investment period never gets here: pass 1b refuses it
            # (``require_period_coverage``).
            rows = rows_src.join(cap_dt.select(axis, "d").unique(), on=[axis, "d"], how="semi")
            if rows.height == 0:
                continue
        lifetime, rows_aligned = align_join_dtypes(edd.rename({"e": axis}), rows, (axis,))
        lifetime = lifetime.join(rows_aligned, on=[axis, "d"], how="inner")
        invest = _var_view(var, "__cumcap_at", {"d": "d_invest"})
        lhs: dict = {"invest_cum": Sum(Where(invest * unitsize, lifetime), over=("d_invest",))}
        if divest is not None:
            # #29: every earlier divestment still active at d, not just d's own — the maxFlow measure
            # (retired[e,d], _capacity).
            view = _var_view(divest, "__cumcap_div", {"d": "d_divest"}, to_axis=True)
            active = _entity_rows(b.cap.retired_frame(axis), axis, rows, axis)
            active, rows_div = align_join_dtypes(active, rows.select(axis, "d"), (axis, "d"))
            active = active.join(rows_div, on=[axis, "d"], how="semi")
            lhs["divest"] = -Sum(Where(view * unitsize, active), over=("d_divest",))
        built = _retire_built.retired(b, axis)
        if built is not None:
            # The model-built capacity retired so far (_retire_built).
            built_rows, rows_b = align_join_dtypes(_retire_built.retired_frame(b, axis), rows.select(axis, "d"),
                                                   (axis, "d"))
            built_rows = built_rows.join(rows_b, on=[axis, "d"], how="semi")
            if built_rows.height > 0:
                lhs["divest_built"] = -Where(built * unitsize, built_rows)
        rhs: dict = {"cap": drop_no_limit_rows(Param((axis, "d"), cap_frame), "upper")}
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
    #   ∀ (g,d) ∈ gd_invest_period, restricted to the (g,d) some member actually has the variable for; a max row
    #   is left out where the group's limit is inf (no limit in that period)
    #   Divest: R_p[p,d] = v_divest_p[p,d] + Σ_c v_divest_built_p[p,c,d] (existing and model-built capacity retired
    #   in d) with group_retire_{max,min}_period over gd_divest_period (same restriction)
    #   _n: the node analogues with v_invest_n / v_divest_n and state_unitsize[n]
    # With a construction lead time the invest rows count capacity coming online in d:
    #   Σ_{p : (g,p) ∈ group_entity} (Σ_{(p,o,d) ∈ edd_invest_commission} v_invest_p[p,o] · unitsize[p]
    #                                 + carried_commission[p,d])  ≤ / ≥ group_invest_{max,min}_period[g,d]
    #   over the (g,d) where a member's decision comes online, g with a per-period limit (c(o) = o without a lead
    #   time: the form above).
    # Bounds the capacity a group of entities builds or retires in one period, MW (MWh for storage
    # nodes); processes and storage nodes of a group get separate rows.  Group retire_{max,min}_period is
    # not a schema parameter, so the divest rows are reachable only with off-schema data.  Under recourse
    # ``gd_invest_period`` / ``gd_divest_period`` list a group's own invest/divest period's branch periods
    # too (#67); a branch period with no member actually invest/divest-eligible there (e.g. a branch's own
    # 0 beating the anchor's cost, #82) gets no row, not a vacuous 0 ≥ a positive floor.
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
            # A group's retirement also counts the retirements of model-built capacity (_retire_built).
            built = _retire_built.decided(b, axis) if kind == "divest" else None
            if (var is None and built is None) or unitsize is None:
                continue
            cells = var.frame.select(axis, "d") if built is None else _retire_built.retirement_cells(b, axis, var)
            members = _on_var_axis(group_entity, axis, cells)
            if members.height == 0:
                continue
            # #82-adjacent: row domain is TERMS' (g, d), not the full gd_idx -- gd_idx may list a
            # branch period of a group's invest period (#67) that no member actually has the
            # variable for (e.g. a member whose no_reinvestment lifetime window has ended; before #218 also a
            # branch-native 0 beating the anchor's invest_cost, #82). A row
            # there would be vacuous on the LHS (empty sum) while the RHS min/max still carries
            # the branch's (inherited-or-own) cap, turning an unconstrained branch into an
            # infeasible one for the ">=" (min) sense.
            if kind == "invest" and _lead_time.active(d):
                _group_period_limit_online(b, name, axis, var, unitsize, members, gd_idx, cap, sense, param_name)
                continue
            terms = (members.join(cells, on=axis, how="inner")
                     .join(gd_idx, on=["g", "d"], how="inner"))
            if terms.height == 0:
                continue
            if sense == "<=":
                # FA-12: only on (g, d) whose members have the variable.
                check_max_cap(terms.select("g", "d").unique(), cap.frame, "g", True,
                              param_name=param_name)
                # A period with the group limit inf has no row (no limit there).
                terms = without_no_limit_rows(terms, cap, "upper")
                if terms.height == 0:
                    continue
            lhs: dict = ({kind: Sum(Where(var * unitsize, terms.select("g", axis, "d")), over=(axis,))}
                         if var is not None else {})
            if built is not None:
                lhs["divest_built"] = Sum(Where(built * unitsize, terms.select("g", axis, "d")), over=(axis,))
            b.m.add_cstr(
                name,
                over      = terms.select("g", "d").unique(),
                sense     = sense,
                lhs_terms = lhs,
                rhs_terms = {"cap": drop_no_limit_rows(cap, "upper")},
            )


def _group_period_limit_online(b: _Build, name: str, axis: str, var, unitsize, members: pl.DataFrame,
                               gd_idx: pl.DataFrame, cap: Param, sense: str, param_name: str | None) -> None:
    """The group per-period invest row with a construction lead time: the members' capacity coming online in d
    (decisions of this solve and capacity carried from earlier solves) against the group limit, over the (g, d)
    where a member's decision comes online, for the groups of ``gd_idx``."""
    d = b.d
    rows = _lead_time.commission_rows(d, var, axis)
    rows, mem = align_join_dtypes(rows, members.select("g", axis), [axis])
    rows = rows.join(mem, on=axis, how="inner")
    # The groups with a per-period limit; the row period is where the capacity comes online, which need not be an
    # investment period of the group.
    rows, gd = align_join_dtypes(rows, gd_idx.select("g").unique(), ["g"])
    rows = rows.join(gd, on="g", how="semi")
    if rows.height == 0:
        return
    over = rows.select("g", "d").unique().sort("g", "d")
    if sense == "<=":
        check_max_cap(over, cap.frame, "g", True, param_name=param_name)  # FA-12
        over = without_no_limit_rows(over, cap, "upper")
        if over.height == 0:
            return
        cap = drop_no_limit_rows(cap, "upper")
    lhs: dict = {"invest": _lead_time.decided_sum(var, unitsize, rows.select("g", axis, "d_invest", "d"),
                                                  "__at_online_g", (axis, "d_invest"))}
    car = _lead_time.carried(d, "commission", axis, members.select(axis).unique())
    if car is not None:
        car, mem = align_join_dtypes(car, members.select("g", axis), [axis])
        # Fixed summation order over the members (the join leaves row order unspecified).
        car_g = (car.join(mem, on=axis, how="inner").sort("g", "d", axis)
                 .group_by("g", "d", maintain_order=True).agg(pl.col("value").sum()))
        car_g, over_k = align_join_dtypes(car_g, over, ["g", "d"])
        car_g = car_g.join(over_k, on=["g", "d"], how="semi")
        if car_g.height > 0:
            lhs["carried"] = Param(("g", "d"), car_g.select("g", "d", "value"))
    b.m.add_cstr(
        name,
        over      = over,
        sense     = sense,
        lhs_terms = lhs,
        rhs_terms = {"cap": cap},
    )


def _eq_group_invest_total(b: _Build) -> None:
    """maxInvestGroup_entity_total[_path]_{p,n} / minInvestGroup_entity_total_path_{p,n}: capacity a group has
    invested over the chain."""
    d = b.d
    group_entity, g_set = d.group_entity, d.g_invest_total
    if not _has_rows(group_entity) or not _has_rows(g_set):
        return
    d_leaf = getattr(d, "d_leaf", None)
    prior = chain_prior(d, "invest")

    # ─── EQ. ─── maxInvestGroup_entity_total[_path]_{p,n} / minInvestGroup_entity_total_path_{p,n} ───
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{p:(g,p)∈group_entity} Σ_{d∈D(leaf)} v_invest_p[p,d]·unitsize[p]
    #   ≤ max(0, group_invest_max_total[g] − Σ_{p∈g} prior_inv[p])   (maxInvestGroup_entity_total[_path]_p)
    #   ≥ max(0, group_invest_min_total[g] − Σ_{p∈g} prior_inv[p])   (minInvestGroup_entity_total_path_p)
    #   ∀ g ∈ g_invest_total (, leaf)
    #   prior_inv[p] = p_entity_invested[p]  (capacity realized by all earlier solves of the
    #     chain); the group's prior sums every member of the axis kind, not only those with the
    #     variable in THIS solve — their earlier investment still counts
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
            # FA-12: only on g whose members have the variable.
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
    """maxDivestGroup_entity_total[_path]_{p,n} / minDivestGroup_entity_total_path_{p,n}: capacity a group
    retires over the chain."""
    d = b.d
    group_entity, g_set = d.group_entity, d.g_divest_total
    if not _has_rows(group_entity) or not _has_rows(g_set):
        return
    d_leaf = getattr(d, "d_leaf", None)
    prior = chain_prior(d, "divest")

    # ─── EQ. ─── maxDivestGroup_entity_total[_path]_{p,n} / minDivestGroup_entity_total_path_{p,n} ───
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{p:(g,p)∈group_entity} Σ_{d∈D(leaf)} R_p[p,d]·unitsize[p]
    #   R[e,d] = v_divest[e,d] + Σ_c v_divest_built[e,c,d] (existing and model-built capacity retired in d)
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
        built = _retire_built.decided(b, axis)
        if cap is None or (var is None and built is None) or unitsize is None:
            continue
        members = (_entities_of_kind(d, group_entity, axis).pipe(rename_to_axis, {"e": axis})
                   .filter(pl.col("g").is_in(g_set["g"].unique())))
        if members.height == 0:
            continue
        cells = var.frame if built is None else _retire_built.retirement_cells(b, axis, var)
        terms = members.join(cells.select(axis, "d").unique(), on=axis, how="inner")
        if terms.height == 0:
            continue
        if sense == "<=":
            # FA-12: only on g whose members have the variable.
            check_max_cap(terms.select("g").unique(), cap.frame, "g", False, param_name=param_name)
        cap_g = Param(("g",), minus_prior(cap.frame, "g", group_prior(group_entity, prior, d, axis)))
        is_path, over, mask = leaf_rows(terms, "g", var if built is None else cells, d_leaf)
        name = path_name if (always_path or is_path) else legacy_name
        lhs: dict = {"divest_grp": Sum(Where(var * unitsize, mask), over=(axis, "d"))} if var is not None else {}
        if built is not None:
            lhs["divest_grp_built"] = Sum(Where(built * unitsize, mask), over=(axis, "d"))
        b.m.add_cstr(
            name,
            over      = over,
            sense     = sense,
            lhs_terms = lhs,
            rhs_terms = {"cap":        cap_g},
        )


def _eq_group_invest_cumulative(b: _Build) -> None:
    """{max,min}InvestGroup_entity_cumulative: existing plus invested capacity of a group per period."""
    d = b.d
    group_entity, g_set, edd = d.group_entity, d.g_invest_cumulative, b.cap.alive
    if not _has_rows(group_entity) or not _has_rows(g_set) or not _has_rows(edd):
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
    # Σ_{p : (g,p) ∈ group_entity} invested[p,d] · unitsize[p]
    #   + Σ_{n : (g,n) ∈ group_entity} invested[n,d] · state_unitsize[n]
    #   − Σ_{p : (g,p) ∈ group_entity} retired[p,d] · unitsize[p]
    #   − Σ_{n : (g,n) ∈ group_entity} retired[n,d] · state_unitsize[n]
    #   ≤ cumulative_max_capacity[g,d] − Σ_{e : (g,e) ∈ group_entity} existing[e,d]   (max…)
    #   ≥ cumulative_min_capacity[g,d] − Σ_{e : (g,e) ∈ group_entity} existing[e,d]   (min…)
    #   ∀ g ∈ g_invest_cumulative (invest_method = cumulative_limits, #5/B-Q3), d with a cap value
    #   and some member alive (invest) or retirable (divest) in d; a max row is left out where
    #   cumulative_max_capacity is inf (no limit in that period)
    # Keeps the capacity of a group in a period (existing + invested and still alive − every
    # retirement up to and including d, #30b) within its cumulative limits (schema
    # "[MW or MWh]"); processes and storage nodes share one row.  Max rows: groups that author
    # the cap at all (any period), restricted to periods the group actually has a candidate
    # (g, d) row for; FA-12 (check_max_cap) then catches a period that same group's own map
    # leaves out — mirrors the entity-level cumulative row (previously this
    # family had no FA-12 check at all, unlike the entity-level row).  Min rows: the periods
    # with a value other than 0 (a min of 0 is no floor).  A period map that leaves out an
    # investment period of the solve cannot reach here: pass 1b refuses it
    # (``require_period_coverage``).  Capacity invested by earlier solves of a chain is not
    # counted (pre-existing, separate item).
    # Capacity in service: invested[e,d], retired[e,d] (and prebuilt[e,d]) of _capacity, the retirements of
    # existing capacity counting until its end of life (none for reinvest_automatic).
    for name, sense, cap, param_name in (
        ("maxInvestGroup_entity_cumulative", "<=", d.pd_group_cumulative_max_capacity,
         "cumulative_max_capacity"),
        ("minInvestGroup_entity_cumulative", ">=", d.pd_group_cumulative_min_capacity, None),
    ):
        if cap is None:
            continue
        cap_frame, over = align_join_dtypes(cap.frame, base_over, ("g",))
        if sense == "<=":
            over = over.join(cap_frame.select("g").unique(), on="g", how="semi")
            if over.height == 0:
                continue
            check_max_cap(over, cap_frame, "g", True, param_name=param_name)
            # A period with cumulative_max_capacity = inf has no row (no limit there).
            over = without_no_limit_rows(over, cap, "upper")
            if over.height == 0:
                continue
            cap = drop_no_limit_rows(cap, "upper")
        else:
            over = over.join(cap_frame.select("g", "d").unique(), on=["g", "d"], how="semi")
            if over.height == 0:
                continue
        lhs: dict = {}
        for key, var, divest, axis in (("invest_p", b.v_invest_p, b.v_divest_p, "p"),
                                        ("invest_n", b.v_invest_n, b.v_divest_n, "n")):
            unitsize = _unitsize(d, axis)
            if var is None or unitsize is None:
                continue
            terms = (_on_var_axis(group_entity, axis, var).filter(in_groups)
                     .join(_on_var_axis(edd, axis, var), on=axis, how="inner"))
            if terms.height > 0:
                invest = _var_view(var, "__grp_cum_at", {"d": "d_invest"})
                lhs[key] = Sum(Where(invest * unitsize, terms), over=(axis, "d_invest"))
            active_frame = b.cap.retired_frame(axis)
            if divest is not None:
                div_members = _on_var_axis(group_entity, axis, divest).filter(in_groups)
                active_axis = _entity_rows(active_frame, axis, div_members, axis)
                active_axis, div_members = align_join_dtypes(active_axis, div_members, (axis,))
                div_terms = div_members.join(active_axis, on=axis, how="inner")
                if div_terms.height > 0:
                    view = _var_view(divest, "__grp_cum_div", {"d": "d_divest"}, to_axis=True)
                    lhs[f"{key}_divest"] = -Sum(Where(view * unitsize, div_terms), over=(axis, "d_divest"))
            built = _retire_built.retired(b, axis)
            if built is not None:
                # The model-built capacity retired so far (_retire_built).
                counted = _retire_built.retired_frame(b, axis)
                built_members = _on_var_axis(group_entity, axis, counted).filter(in_groups)
                built_terms = built_members.join(counted, on=axis, how="inner")
                if built_terms.height > 0:
                    lhs[f"{key}_divest_built"] = -Sum(Where(built * unitsize, built_terms), over=(axis,))
        if not lhs:
            continue
        rhs: dict = {"cap": cap}
        if b.cap.existing_mw_e is not None:
            group_existing = (b.cap.existing_mw_e.frame
                              .join(group_entity, on="e", how="inner")
                              # Fixed summation order over the members.
                              .sort("g", "d", "e")
                              .group_by(["g", "d"], maintain_order=True).agg(pl.col("value").sum())
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
    """{max,min}Cumulative_flow_solve[_path]: net flow into a flow group, weighted like an
    operational cost, summed over the solve (or, under recourse, over each scenario path)."""
    d = b.d
    flow = _group_flow(b)
    if flow is None:
        return
    caps = [(name, sense, cap) for name, sense, cap in
            (("maxCumulative_flow_solve", "<=", d.p_group_max_cumulative_flow),
             ("minCumulative_flow_solve", ">=", d.p_group_min_cumulative_flow))
            if cap is not None and cap.frame.height > 0]
    if not caps:
        return
    from flextool.engine.model.core import _check_covers_dt

    _check_covers_dt(d, "p_timestep_weight", "flow-group cumulative flow limits")
    _check_covers_dt(d, "p_period_share", "flow-group cumulative flow limits", positive=True)
    _check_covers_dt(d, "p_years_represented_d", "flow-group cumulative flow limits")

    # r[d] = (years_represented[d] / period_share[d]), normalised to 1.0 at its own maximum, so
    # equal periods (tw=1, one years_represented per period) give r=1 and the row stays
    # byte-identical to the legacy unweighted-hours measure (co2_max_total uses the same ratio,
    # without the normalisation it does not need: its RHS is per-g already, this one isn't).
    ratio = (d.p_years_represented_d.frame.rename({"value": "years"})
             .join(d.p_period_share.frame.rename({"value": "share"}), on="d", how="inner")
             .with_columns(ratio=pl.col("years") / pl.col("share")))
    r_max = float(ratio["ratio"].max())
    r_frame = ratio.with_columns(value=pl.col("ratio") / r_max).select("d", "value")
    r_param = Param(("d",), r_frame)
    w_y_frame = (d.p_step_duration.frame.rename({"value": "sd"})
                 .join(d.p_timestep_weight.frame.rename({"value": "tw"}), on=["d", "t"], how="inner")
                 .join(r_frame.rename({"value": "r"}), on="d", how="inner")
                 .with_columns(value=pl.col("sd") * pl.col("tw") * pl.col("r"))
                 .select("d", "t", "value"))
    d_leaf = getattr(d, "d_leaf", None)
    dt_periods = d.p_step_duration.frame.select("d").unique()

    # ─── EQ. ─── {max,min}Cumulative_flow_solve[_path] ───────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{d,t} F[g,d,t]·w_y[d,t]  ≤ max_cumulative_flow[g] · Σ_{d,t} w_y[d,t]   (max…)
    #                            ≥ min_cumulative_flow[g] · Σ_{d,t} w_y[d,t]   (min…)
    #   ∀ g with a max / min_cumulative_flow (, per leaf under recourse) — an inf max / −inf min is no limit (no row)
    #   F = net flow into the group (_group_flow)
    #   w_y[d,t] = step_duration[d,t]·timestep_weight[d,t]·r[d]
    #   r[d] = (years_represented[d]/period_share[d]) / max_d′(years_represented[d′]/period_share[d′])
    # Limits the energy flowing into a flow group over the solve to an average power times the
    # weighted hours of the solve — the same weighting every operational cost term already
    # carries (op_factor), so a representative period or an unequal roll window does not
    # under/over-count its hours.  Under recourse each scenario path gets its own row (_path, one
    # per (g, leaf), leaf_rows): a plain per-group row, summed across branches, would let one
    # branch's surplus cover another's deficit.  Constraints carry no stochastic branch weight —
    # each branch must respect the limit on its own, unlike the objective (owner decision).
    for name, sense, cap in caps:
        terms = cap.frame.select("g").unique().join(dt_periods, how="cross").select("g", "d")
        is_path, over, mask = leaf_rows(terms, "g", b.v_flow, d_leaf)
        row_name = f"{name}_path" if is_path else name
        leaf_cols = ["g", "leaf"] if is_path else ["g"]
        mask_cols = ["g", "d", "leaf"] if is_path else ["g", "d"]
        rhs_hours = (mask.select(mask_cols).join(w_y_frame, on="d", how="inner")
                     # Fixed summation order (the join leaves row order unspecified).
                     .sort(*mask_cols, "t")
                     .group_by(leaf_cols, maintain_order=True).agg(pl.col("value").sum()))
        cap_rhs = Param(
            tuple(leaf_cols),
            cap.frame.select("g", "value").rename({"value": "cap_v"})
               .join(rhs_hours, on="g", how="inner")
               .with_columns(value=pl.col("cap_v") * pl.col("value"))
               .select(*leaf_cols, "value"))
        b.m.add_cstr(
            row_name,
            over      = over,
            sense     = sense,
            lhs_terms = {"flow_int": Sum(Where(flow * d.p_step_duration * d.p_timestep_weight * r_param, mask),
                                         over=("d", "t"))},
            rhs_terms = {"cap":      cap_rhs},
        )


def _eq_cumulative_flow_period(b: _Build) -> None:
    """{max,min}Cumulative_flow_period: net flow into a flow group, weighted like an operational
    cost, summed over each period."""
    d = b.d
    flow = _group_flow(b)
    if flow is None:
        return
    caps = [cap for cap in (d.pd_max_cumulative_flow, d.pd_min_cumulative_flow)
            if cap is not None and cap.frame.height > 0]
    if not caps:
        return
    from flextool.engine.model.core import _check_covers_dt

    _check_covers_dt(d, "p_timestep_weight", "flow-group cumulative flow limits")

    hours_in_period = (d.p_step_duration.frame.rename({"value": "sd"})
                       .join(d.p_timestep_weight.frame.rename({"value": "tw"}), on=["d", "t"], how="inner")
                       .sort("d", "t")  # fixed summation order
                       .group_by("d", maintain_order=True)
                       .agg((pl.col("sd") * pl.col("tw")).sum().alias("hours_in_period")))

    # ─── EQ. ─── {max,min}Cumulative_flow_period ─────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_t F[g,d,t]·sd[d,t]·tw[d,t]  ≤ max_cumulative_flow[g,d] · Σ_t sd[d,t]·tw[d,t]   (max…)
    #                               ≥ min_cumulative_flow[g,d] · Σ_t sd[d,t]·tw[d,t]   (min…)
    #   ∀ (g,d) with a max / min_cumulative_flow (the reader leaves out an inf max / −inf min: no limit in d)
    #   F = net flow into the group (_group_flow); sd = step_duration, tw = timestep_weight
    # Limits the energy flowing into a flow group in each period to an average power times the
    # weighted hours of the period — the same timestep_weight every operational cost term already
    # carries (op_factor), so a representative-period step does not count as a full unweighted
    # hour.  A branch period that authors no value of its own reads its anchor's value
    # (broadcast_to_period's branch-period inheritance); it still gets its own (g,d) row here, not
    # merged with the anchor's — the continuation fan always starts at a solve's first step
    # (_branch.py, d_leaf_lf), so no period is ever split into an anchor fragment and a branch
    # fragment within one solve, and there is nothing to merge.
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
            lhs_terms = {"flow_int": Sum(flow * d.p_step_duration * d.p_timestep_weight, over=("t",))},
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
    #   ∀ (g,d,t) ∈ _instant_flow_support(cap, dt); the reader leaves out an inf max / −inf min (no limit there)
    #   F = net flow into the group (_group_flow), MW
    # Limits the net power flowing into a flow group in every step.  The cap may be authored as a
    # scalar, period map, time map or period-time map; the rows cover every active step of its shape.
    # Row rule (coarse steps): a flow group whose processes are all coarse gets one row per coarse step,
    # {max,min}Instant_flow_coarse = the row's Δt-weighted mean over the step (_coarse_flows.add_cstr): the step's
    # constant net flow against the step-mean cap (named in the coarse-flow behaviour notice when the cap varies
    # within a step).  A group with a fine process keeps its rows per solve step.
    members = None
    if b.has_coarse_proc:
        from flextool.engine.preprocess.flex_data import coarse_row_members

        members = coarse_row_members(d, "instant_flow")
    for name, sense, cap in (("maxInstant_flow", "<=", d.pdt_max_instant_flow),
                             ("minInstant_flow", ">=", d.pdt_min_instant_flow)):
        if cap is None or cap.frame.height == 0:
            continue
        over = _instant_flow_support(cap.frame, dt)
        if over.height == 0:
            continue
        _coarse_flows.add_row_rule_cstr(
            b, name,
            over      = over,
            sense     = sense,
            lhs_terms = {"flow": flow},
            rhs_terms = {"cap":  cap},
            partition = None if members is None else _coarse_flows.row_partition(b, members, over),
        )
