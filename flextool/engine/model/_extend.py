"""Life extension: the extension columns, the extended capacity, their bound, the ties and the cost.

A cohort of an entity that gives ``life_extension_cost`` and ``life_extension_years`` and that ends its life at the
start of an investment period ``d`` of the solve has an extension column ``v_extend[e,c,d]`` (``cd_extend``,
``preprocess/params/_extend.py``): the units of the cohort still in service at its end of life that are kept
``life_extension_years`` more.  The extension is a cohort of its own (``extend:<d>:<c>``), priced as one plant
(``params._extend.extension_prices``) and retirable as model-built capacity (``_retire_built``).  One aggregate
column per (entity, period) holds the extended units in service there, so every capacity row gains one term
whatever the number of cohorts (:func:`extended`: the ``extended[e,d]`` term of the capacity rows, never an
investment term).

Owns the variables ``v_extend_{p,n}`` / ``v_extended_{p,n}``, the rows ``extendedCapacity_{p,n}``,
``maxExtend_cohort_{p,n}`` and ``non_anticipativity_extend_{p,n}`` and the objective term ``extend_cost_{p,n}``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polar_high import Param, Sum, Where

from flextool.engine.input._axis_enums import align_join_dtypes, rename_to_axis, schema_dtype
from flextool.engine.model import _capacity, _retire_built
from flextool.engine.model._common import _entity_rows, _has_rows, _var_view
from flextool.engine.model._retire_built import _cost, _members, _unitsize
from flextool.engine.preprocess.params._cohorts import KIND_EXISTING, KIND_INVEST

if TYPE_CHECKING:
    from flextool.engine.model.core import _Build


def options(d, axis: str) -> pl.DataFrame | None:
    """``[axis, cohort, d]``: the extension options of the processes (``axis="p"``) or storage nodes (``"n"``),
    ``cohort`` the cohort extended; ``None`` without one."""
    frame, members = getattr(d, "cd_extend", None), _members(d, axis)
    if not _has_rows(frame) or not _has_rows(members):
        return None
    rows = _entity_rows(frame, "e", members, axis)
    return rows if rows.height > 0 else None


def _counting(d, axis: str) -> pl.DataFrame:
    """``[axis, cohort, d_extend, d]``: the units extended in ``d_extend`` are in service in ``d``."""
    return _entity_rows(d.cdd_extend, "e", _members(d, axis), axis)


def _parents(d, axis: str) -> pl.DataFrame:
    """``cohort_extend`` on ``axis``: one row per option, the parent's kind, decision period, last period in service
    before the extension, carried size."""
    return _entity_rows(d.cohort_extend, "e", _members(d, axis), axis)


def prepare(b: "_Build") -> None:
    """Set the feature flags ``has_extend_p`` / ``has_extend_n``."""
    b.has_extend_p = options(b.d, "p") is not None
    b.has_extend_n = options(b.d, "n") is not None and b.d.p_state_unitsize is not None


def add_variables(b: "_Build") -> None:
    """Declare the extension columns and the extended capacity in service."""
    m, d = b.m, b.d

    # ─── VAR. ─── v_extend_p / v_extend_n ───────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_extend_p[p,c,d] ≥ 0   ∀ (p,c,d) ∈ cd_extend, p a process   (n: storage nodes, state_unitsize)
    #   cd_extend = {(e,c,d) : e gives life_extension_cost and life_extension_years at d; c a cohort of e that ends
    #     its life (reinvest_choice / no_reinvestment: the existing cohort, an investment of this solve, or one
    #     carried from an earlier solve) and is not an extension; d ∈ period_invest, |y[d] − y_eol(c)| ≤ YEAR_TOL,
    #     d on c's lineage ((d_build(c), d) ∈ dd_forward for an investment of this solve);
    #     y_eol(x) = round_end_of_life(y[d] + life_extension_years[e,d], end_of_life_method[e]) > y[d]}
    # Units of cohort c still in service at its end of life that are kept for life_extension_years more: a cohort
    # of its own, x(c,d) = extend:<d>:<c>, online in d.  No invest_method is needed.
    # ─── VAR. ─── v_extended_p / v_extended_n ───────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_extended_p[p,d] ≥ 0   ∀ (p,d) ∈ {(e,d) : (e,c,d′,d) ∈ cdd_extend}   (n: storage nodes)
    #   cdd_extend = {(e,c,d′,d) : (e,c,d′) ∈ cd_extend, (d′,d) ∈ dd_forward, y[d] < y_eol(x(c,d′))}
    # The extended units in service in d: the extended[e,d] term of every capacity row (_capacity), never a term
    # of an investment.
    for axis in ("p", "n"):
        if not getattr(b, f"has_extend_{axis}"):
            continue
        rows = options(d, axis)
        setattr(b, f"v_extend_{axis}", m.add_var(f"v_extend_{axis}", (axis, "cohort", "d"), rows))
        counted = _counting(d, axis).select(axis, "d").unique().sort(axis, "d")
        setattr(b, f"v_extended_{axis}", m.add_var(f"v_extended_{axis}", (axis, "d"), counted))


def extended(b: "_Build", axis: str):
    """``v_extended[axis, d]`` (units): the extended capacity in service in d; ``None`` without an extension."""
    return getattr(b, f"v_extended_{axis}", None)


def extended_frame(b: "_Build", axis: str) -> pl.DataFrame:
    """``[axis, d]``: the rows of :func:`extended` (empty without an extension)."""
    var = extended(b, axis)
    if var is None:
        return pl.DataFrame(schema={axis: schema_dtype(None, axis), "d": schema_dtype(None, "d")})
    return var.frame.select(axis, "d")


def decided(b: "_Build", axis: str):
    """``Σ_c v_extend[axis, c, d]`` (units), dims ``(axis, d)``: the capacity extended in d (it starts to operate
    there); ``None`` without an extension."""
    var = getattr(b, f"v_extend_{axis}", None)
    if var is None:
        return None
    return Sum(var, over=("cohort",))


def decided_frame(b: "_Build", axis: str) -> pl.DataFrame:
    """``[axis, d]``: the (entity, period) with an extension option (empty without one)."""
    var = getattr(b, f"v_extend_{axis}", None)
    if var is None:
        return pl.DataFrame(schema={axis: schema_dtype(None, axis), "d": schema_dtype(None, "d")})
    return var.frame.select(axis, "d").unique().sort(axis, "d")


def capacity_term(b: "_Build", axis: str, members: pl.DataFrame, coef=None):
    """``extended[e,d]·coef`` over the (entity, period) of :func:`extended` whose entity is in ``members``
    (``[axis]``), dims ``(axis, d)``; ``None`` without such a row.  The extended capacity in service, to add where
    a capacity row counts the capacity."""
    var = extended(b, axis)
    if var is None:
        return None
    rows = extended_frame(b, axis)
    keys = members.select(pl.col(axis).cast(rows.schema[axis], strict=False)).unique()
    rows = rows.join(keys, on=axis, how="semi")
    if rows.height == 0:
        return None
    return Where(var if coef is None else var * coef, rows)


def add_constraints(b: "_Build") -> None:
    """extendedCapacity_{p,n} and maxExtend_cohort_{p,n}."""
    m, d = b.m, b.d
    for axis in ("p", "n"):
        var = getattr(b, f"v_extend_{axis}", None)
        if var is None:
            continue
        counting = _counting(d, axis)
        ext = getattr(b, f"v_extended_{axis}")
        at = _var_view(var, "__ext_at", {"d": "d_extend"}, to_axis=True)

        # ─── EQ. ─── extendedCapacity_{p,n} ──────────────────────────────────
        # ─────────────────────────────────────────────────────────────────────
        # v_extended[e,d] − Σ_{(e,c,d′,d) ∈ cdd_extend} v_extend[e,c,d′] = 0
        #   ∀ (e,d) with some (e,c,d′,d) ∈ cdd_extend;  e a process (_p) or a storage node (_n)
        # The extended capacity in service in d: every extension decided on d's lineage up to d whose extended end
        # of life is after d.
        m.add_cstr(
            f"extendedCapacity_{axis}",
            over      = ext.frame.select(axis, "d"),
            sense     = "==",
            lhs_terms = {"extended": ext,
                         "extend":   -Sum(Where(at, counting), over=("cohort", "d_extend"))},
            rhs_terms = {},
        )

        # ─── EQ. ─── maxExtend_cohort_{p,n} ──────────────────────────────────
        # ─────────────────────────────────────────────────────────────────────
        # Σ_{d′ : (e,c,d′) ∈ cd_extend, (d′,d) ∈ dd_forward} v_extend[e,c,d′]·us
        #   + retired_c[e,d_last]·us − [c decided here] v_invest[e,d_build(c)]·us
        #   ≤ [c carried] remaining[e,c] + [c existing] size_x[e,d_last]
        #   ∀ (e,c,d) ∈ cd_extend;  us = unitsize (process) / state_unitsize (storage node)
        #   d_last = the last period of c in service before d on d's lineage (cohort_extend)
        #   retired_c[e,d_last] = Σ_{d″ : (e,d″,d_last) ∈ retired} v_divest[e,d″] (c existing) or
        #     Σ_{d″ : (e,c,d″,d_last) ∈ cdd_divest_built} v_divest_built[e,c,d″] (c model-built): the retirements of
        #     c still counting at its end of life
        #   size_x[e,d_last] = pre_existing[e,d_last] − retired_carried[e,d_last] (d_last in the solve), else the
        #     existing capacity in service at the end of life that the run's existing map and the realized
        #     retirements give (cohort_extend size); remaining[e,c] = the carried size of c less its realized
        #     retirements
        # Only what is still in service at the end of life can be extended (a retired part never is), and a cohort
        # is extended once: the extensions of c decided before d on d's lineage count against it too.
        us = _unitsize(d, axis)
        rows = var.frame.select(axis, "cohort", "d")
        parents = _parents(d, axis)
        rows_k, parents = align_join_dtypes(rows, parents, [axis, "d"])
        parents = parents.join(rows_k, on=[axis, "cohort", "d"], how="semi")
        earlier = (rows.pipe(rename_to_axis, {"d": "d_extend"})
                       .join(d.dd_forward.pipe(rename_to_axis, {"d_retire": "d_extend"}), on="d_extend", how="inner")
                       .join(rows.select(axis, "cohort", "d"), on=[axis, "cohort", "d"], how="semi"))
        lhs: dict = {"extend": Sum(Where(at * us, earlier), over=("d_extend",))}
        terms = _parent_terms(b, axis, parents, us)
        lhs.update(terms)
        rhs: dict = {}
        rhs_frame = _parent_rhs(b, axis, parents)
        if rhs_frame.height > 0:
            rhs["capacity"] = Param((axis, "cohort", "d"), rhs_frame)
        m.add_cstr(
            f"maxExtend_cohort_{axis}",
            over      = rows,
            sense     = "<=",
            lhs_terms = lhs,
            rhs_terms = rhs,
        )


def _parent_terms(b: "_Build", axis: str, parents: pl.DataFrame, us) -> dict:
    """The parent's terms of maxExtend_cohort (its retirements still counting at its end of life, and its
    investment when it is decided in this solve), dims ``(axis, cohort, d)``."""
    d = b.d
    out: dict = {}
    last = parents.filter(pl.col("d_last").is_not_null())
    existing = last.filter(pl.col("kind") == KIND_EXISTING).select(axis, "cohort", "d", "d_last")
    divest = getattr(b, f"v_divest_{axis}", None)
    if divest is not None and existing.height > 0:
        reach = b.cap.retired_frame(axis).pipe(rename_to_axis, {"d": "d_last"})
        existing, reach = align_join_dtypes(existing, reach, [axis, "d_last"])
        reach = existing.join(reach, on=[axis, "d_last"], how="inner").select(axis, "cohort", "d", "d_divest")
        if reach.height > 0:
            view = _var_view(divest, "__ext_div", {"d": "d_divest"}, to_axis=True)
            out["retired_existing"] = Sum(Where(view * us, reach), over=("d_divest",))
    built = last.filter(pl.col("kind") != KIND_EXISTING).select(axis, "cohort", "d", "d_last")
    divest_built = getattr(b, f"v_divest_built_{axis}", None)
    if divest_built is not None and built.height > 0:
        counted = (_entity_rows(d.cdd_divest_built, "e", _members(d, axis), axis)
                   .pipe(rename_to_axis, {"d": "d_last"}))
        built, counted = align_join_dtypes(built, counted, [axis, "d_last"])
        reach = built.join(counted, on=[axis, "cohort", "d_last"], how="inner").select(axis, "cohort", "d",
                                                                                    "d_divest")
        if reach.height > 0:
            view = _var_view(divest_built, "__ext_divb", {"d": "d_divest"}, to_axis=True)
            out["retired_built"] = Sum(Where(view * us, reach), over=("d_divest",))
    here = parents.filter((pl.col("kind") == KIND_INVEST) & pl.col("d_build").is_not_null())
    if here.height > 0:
        invest = getattr(b, f"v_invest_{axis}", None)
        columns = (pl.DataFrame(schema={axis: schema_dtype(None, axis), "d_build": schema_dtype(None, "d")})
                   if invest is None else invest.frame.select(axis, "d").pipe(rename_to_axis, {"d": "d_build"}))
        link, columns = align_join_dtypes(here.select(axis, "cohort", "d", "d_build"), columns, [axis, "d_build"])
        unbuilt = link.join(columns, on=[axis, "d_build"], how="anti")
        if unbuilt.height > 0:
            raise ValueError(f"maxExtend_cohort_{axis}: the cohorts {sorted(set(unbuilt['cohort'].to_list()))} are "
                             "decided in this solve but have no investment column; their extension would be "
                             "unbounded.")
        view = _var_view(invest, "__ext_inv", {"d": "d_build"}, to_axis=True)
        out["invested_neg"] = -Sum(Where(view * us, link), over=("d_build",))
    return out


def _parent_rhs(b: "_Build", axis: str, parents: pl.DataFrame) -> pl.DataFrame:
    """``[axis, cohort, d, value]`` (MW, MWh): the right-hand side of maxExtend_cohort, rows where it is not 0."""
    parts = [parents.select(axis, "cohort", "d", value=pl.col("remaining").fill_null(0.0)
                            + pl.col("size").fill_null(0.0))]
    pre = _capacity.existing_not_retired(b)
    existing = parents.filter((pl.col("kind") == KIND_EXISTING) & pl.col("d_last").is_not_null())
    if pre is not None and existing.height > 0:
        sizes = _entity_rows(pre, "e", existing, axis).pipe(rename_to_axis, {"d": "d_last"})
        existing, sizes = align_join_dtypes(existing, sizes, [axis, "d_last"])
        parts.append(existing.join(sizes, on=[axis, "d_last"], how="inner").select(axis, "cohort", "d", "value"))
    out = parts[0]
    for part in parts[1:]:
        out, part = align_join_dtypes(out, part, [axis, "d"])
        out = pl.concat([out, part])
    return (out.group_by(axis, "cohort", "d", maintain_order=True).agg(pl.col("value").sum())
               .filter(pl.col("value") != 0.0).sort(axis, "cohort", "d"))


def add_non_anticipativity(b: "_Build", window: pl.DataFrame) -> None:
    """non_anticipativity_extend_{p,n} over the invest-NA window ``window`` (``(d, b)``, ``b`` on the period axis)
    of a recourse-investment solve."""
    d = b.d

    # ─── EQ. ─── non_anticipativity_extend_{p,n} ─────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_extend[e,c,d] − v_extend[e,c^b,b] = 0
    #   ∀ (d,b) ∈ pd_non_anticipativity, (e,c,d) and (e,c^b,b) ∈ cd_extend
    #   c^b = c for a cohort not decided in a forecast of the window (the existing cohort, a cohort decided before
    #     the fan, or a carried one); otherwise the cohort of e of the same kind decided in the forecast of b's
    #     branch at the same anchor period as c's decision (non_anticipativity_divest_built; a parent is never an
    #     extension, so at most one parent per kind and decision period)
    # An extension inside the window is the same in every forecast branch.
    window, groups = _retire_built.forecast_window(window)
    for axis in ("p", "n"):
        var = getattr(b, f"v_extend_{axis}", None)
        if var is None:
            continue
        cohorts = (_parents(d, axis).select(axis, "cohort", "kind", "d_build")
                   .unique(subset=[axis, "cohort"], maintain_order=True)
                   .with_columns(parent=pl.lit(None, dtype=pl.Utf8)))
        over = _retire_built.tied_pairs(var.frame.select(axis, "cohort", "d"),
                                        _retire_built.with_parents(d, cohorts, axis), groups, window, axis)
        if over is None:
            continue
        b.m.add_cstr(
            f"non_anticipativity_extend_{axis}",
            over      = over,
            sense     = "==",
            lhs_terms = {"v_d":  var,
                         "v_b": -_var_view(var, "__nae", {"cohort": "cohort_b", "d": "b"})},
            rhs_terms = {},
        )


def objective_terms(b: "_Build") -> list:
    """The ``extend_cost_{p,n}`` terms."""
    d = b.d
    terms: list = []
    for axis in ("p", "n"):
        var = getattr(b, f"v_extend_{axis}", None)
        if var is None:
            continue

        # ─── OBJ. ─── extend_cost_{p,n} ──────────────────────────────────────
        # ─────────────────────────────────────────────────────────────────────
        # Σ_{(e,c,d) ∈ cd_extend} v_extend[e,c,d]·us[e]·(a_b[x]·S̄[x,d] + fc[x,d])·bw[d]
        #   x = x(c,d) the extension cohort, a_b[x] = a_cap[x] − a_salv[x]:
        #   a_cap[x] = pay[x] = (E + S_c)·af(r, L_x),  a_salv[x] = S_x·(1+r)^−L_x·af(r, L_x)
        #     E = 1000·life_extension_cost[e,d], L_x = life_extension_years[e,d], r = discount_rate[e,d] (the
        #     extension's own loan, n_pay = L_x), S_c = 1000·net_salvage_value[e,d] (the parent's salvage at its end
        #     of life, given up), S_x = 1000·net_salvage_value[e,d_eol(x)] (the last period's value after the last
        #     period start), every value on d's lineage; af(r,n) = r/(1 − (1+r)^−n)·(1+r)^(off_inv − 1)
        #   S̄[x,d] = Σ_{d′ : (e,c,d,d′) ∈ cdd_extend} inflation_inv[d′]
        #   fc[x,d] = Σ_{same d′} 1000·fixed_cost[e,d′]·inflation_op[d′]
        #   us = unitsize (state_unitsize for _n), bw = pd_branch_weight under recourse (else 1)
        # An extension pays the level yearly price of its own contract (the extension cost plus the salvage value
        # given up, less the salvage at the extended end of life, financed at discount_rate over
        # life_extension_years) and the fixed cost of each year in service inside the horizon.  Extending and
        # building new capacity of the same life cost alike when E = invest_cost − net_salvage_value (flat).
        us = _unitsize(d, axis)
        for param in (_cost(d, d.cd_extend_cost, axis), _cost(d, d.cd_extend_fixed, axis)):
            if param is not None:
                terms.append(Sum(var * us * param))
    return terms


__all__ = ["add_constraints", "add_non_anticipativity", "add_variables", "capacity_term", "decided",
           "decided_frame", "extended", "extended_frame", "objective_terms", "options", "prepare"]
