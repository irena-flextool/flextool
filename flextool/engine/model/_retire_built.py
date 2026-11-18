"""Retirement of model-built capacity: the retirement columns, their reach and bound, the ties and the cost.

A model-built cohort (an investment of this solve, or one an earlier solve of the run realized and carried) of an
entity whose ``invest_method`` allows retirement (any ``lifetime_method``; a ``reinvest_automatic`` cohort is the root
of its chain of replacements) has a retirement column per investment period it is in service in (``cd_divest_built``, ``preprocess/params/_retire_built.py``).  One
aggregate column per (entity, period) holds the model-built capacity retired so far that still counts there, so
every capacity row gains one term whatever the number of cohorts (``_capacity``: ``retired[e,d]`` adds
``v_divested_built[e,d]``).  The retirement of existing capacity keeps ``v_divest`` (``maxRetired_existing``).

Owns the variables ``v_divest_built_{p,n}`` / ``v_divested_built_{p,n}``, the rows ``divestedBuilt_{p,n}``,
``maxDivest_cohort_{p,n}`` and ``non_anticipativity_divest_built_{p,n}``, the objective term
``divest_built_cost_{p,n}``, and the per-period retirement sum the retirement limits read (:func:`decided`).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polar_high import Param, Sum, Where

from flextool.engine.input._axis_enums import align_join_dtypes, cast_dim, rename_to_axis, schema_dtype
from flextool.engine.model._common import _entity_rows, _has_rows, _var_view
from flextool.engine.preprocess.params._cohorts import KIND_EXTEND

if TYPE_CHECKING:
    from flextool.engine.model.core import _Build


def _members(d, axis: str) -> pl.DataFrame | None:
    return d.process_source_sink if axis == "p" else d.nodeState


def _unitsize(d, axis: str) -> Param:
    return d.p_unitsize if axis == "p" else Param(("n",), d.p_state_unitsize.frame)


def options(d, axis: str) -> pl.DataFrame | None:
    """``[axis, cohort, d]``: the retirement options of the model-built cohorts of the processes (``axis="p"``) or
    storage nodes (``"n"``); ``None`` without one."""
    frame, members = getattr(d, "cd_divest_built", None), _members(d, axis)
    if not _has_rows(frame) or not _has_rows(members):
        return None
    rows = _entity_rows(frame, "e", members, axis)
    return rows if rows.height > 0 else None


def _counting(d, axis: str) -> pl.DataFrame:
    """``[axis, cohort, d_divest, d]``: a retirement of ``cohort`` decided in ``d_divest`` counts in ``d``."""
    return _entity_rows(d.cdd_divest_built, "e", _members(d, axis), axis)


def prepare(b: "_Build") -> None:
    """Set the feature flags ``has_divest_built_p`` / ``has_divest_built_n``."""
    b.has_divest_built_p = options(b.d, "p") is not None
    b.has_divest_built_n = options(b.d, "n") is not None and b.d.p_state_unitsize is not None


def add_variables(b: "_Build") -> None:
    """Declare the retirement columns of model-built capacity and the capacity retired so far."""
    m, d = b.m, b.d

    # ─── VAR. ─── v_divest_built_p / v_divest_built_n ───────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_divest_built_p[p,c,d] ≥ 0   ∀ (p,c,d) ∈ cd_divest_built, p a process   (n: storage nodes, state_unitsize)
    #   cd_divest_built = {(e,c,d) : c a model-built cohort of e (an investment of this solve, or carried from an
    #     earlier solve of the run), e ∈ entityDivest with a lifetime_method (bounded and reinvest_automatic
    #     cohorts; an automatic cohort is in service to the end of the horizon), d ∈ period_invest, (c,d) ∈
    #     cohort_alive (along the scenario lineage)}
    # Units (of unitsize) of cohort c retired in period d; a retirement is permanent: it counts in every later
    # period of the lineage while the cohort is in service (divestedBuilt).
    # ─── VAR. ─── v_divested_built_p / v_divested_built_n ───────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_divested_built_p[p,d] ≥ 0   ∀ (p,d) ∈ {(e,d) : (e,c,d′,d) ∈ cdd_divest_built}   (n: storage nodes)
    #   cdd_divest_built = {(e,c,d′,d) : (e,c,d′) ∈ cd_divest_built, (d′,d) ∈ dd_forward, (c,d) ∈ cohort_alive}
    # Model-built units of the entity retired so far and still counting in d (their cohort in service in d); one
    # column per (entity, period), the retired[e,d] term of every capacity row (_capacity).
    for axis in ("p", "n"):
        if not getattr(b, f"has_divest_built_{axis}"):
            continue
        rows = options(d, axis)
        setattr(b, f"v_divest_built_{axis}", m.add_var(f"v_divest_built_{axis}", (axis, "cohort", "d"), rows))
        counted = _counting(d, axis).select(axis, "d").unique().sort(axis, "d")
        setattr(b, f"v_divested_built_{axis}", m.add_var(f"v_divested_built_{axis}", (axis, "d"), counted))


def retired(b: "_Build", axis: str):
    """``v_divested_built[axis, d]`` (units): the model-built capacity retired so far that counts in d; ``None``
    without retirable model-built capacity."""
    return getattr(b, f"v_divested_built_{axis}", None)


def retired_frame(b: "_Build", axis: str) -> pl.DataFrame:
    """``[axis, d]``: the rows of :func:`retired` (empty without retirable model-built capacity)."""
    var = retired(b, axis)
    if var is None:
        return pl.DataFrame(schema={axis: schema_dtype(None, axis), "d": schema_dtype(None, "d")})
    return var.frame.select(axis, "d")


def decided(b: "_Build", axis: str):
    """``Σ_c v_divest_built[axis, c, d]`` (units), dims ``(axis, d)``: the model-built capacity retired in d, the
    retirement the per-period and total retirement limits sum; ``None`` without retirable model-built capacity."""
    var = getattr(b, f"v_divest_built_{axis}", None)
    if var is None:
        return None
    return Sum(var, over=("cohort",))


def decided_frame(b: "_Build", axis: str) -> pl.DataFrame:
    """``[axis, d]``: the (entity, period) with a model-built retirement option (empty without one)."""
    var = getattr(b, f"v_divest_built_{axis}", None)
    if var is None:
        return pl.DataFrame(schema={axis: schema_dtype(None, axis), "d": schema_dtype(None, "d")})
    return var.frame.select(axis, "d").unique().sort(axis, "d")


def retirement_cells(b: "_Build", axis: str, var) -> pl.DataFrame:
    """``[axis, d]``: the (entity, period) with a retirement column, of existing capacity (``var``, v_divest; None
    without) or of model-built capacity (:func:`decided_frame`): the rows of a retirement limit."""
    built = decided_frame(b, axis)
    if var is None:
        return built
    own, built = align_join_dtypes(var.frame.select(axis, "d"), built, [axis, "d"])
    return pl.concat([own, built]).unique(maintain_order=True)


def add_constraints(b: "_Build") -> None:
    """divestedBuilt_{p,n} and maxDivest_cohort_{p,n}."""
    m, d = b.m, b.d
    for axis in ("p", "n"):
        var = getattr(b, f"v_divest_built_{axis}", None)
        if var is None:
            continue
        counting = _counting(d, axis)
        divested = getattr(b, f"v_divested_built_{axis}")
        at = _var_view(var, "__built_at", {"d": "d_divest"}, to_axis=True)

        # ─── EQ. ─── divestedBuilt_{p,n} ─────────────────────────────────────
        # ─────────────────────────────────────────────────────────────────────
        # v_divested_built[e,d] − Σ_{(e,c,d′,d) ∈ cdd_divest_built} v_divest_built[e,c,d′] = 0
        #   ∀ (e,d) with some (e,c,d′,d) ∈ cdd_divest_built;  e a process (_p) or a storage node (_n)
        # The model-built capacity retired so far and still counting in d: the retirements of every cohort of e
        # decided in d′ on the lineage up to d (dd_forward) while that cohort is in service in d.  After a cohort's
        # end of life its retirements stop counting with it.
        m.add_cstr(
            f"divestedBuilt_{axis}",
            over      = divested.frame.select(axis, "d"),
            sense     = "==",
            lhs_terms = {"divested": divested,
                         "retired":  -Sum(Where(at, counting), over=("cohort", "d_divest"))},
            rhs_terms = {},
        )

        # ─── EQ. ─── maxDivest_cohort_{p,n} ──────────────────────────────────
        # ─────────────────────────────────────────────────────────────────────
        # Σ_{d′ : (e,c,d′,d) ∈ cdd_divest_built} v_divest_built[e,c,d′]·us[e]
        #   − [c decided in this solve] v_invest[e,d_build(c)]·us[e]
        #   − [c a life extension of this solve] v_extend[e,parent(c),d_build(c)]·us[e]  ≤  [c carried] remaining[e,c]
        #   ∀ (e,c,d) ∈ cd_divest_built;  us = unitsize (process) / state_unitsize (storage node)
        #   remaining[e,c] = the carried size of c less the retirements earlier solves realized (realized_retire)
        # A cohort never retires more than it has (#70: retired ≤ remaining), in every period it can retire in, so
        # also where two periods of a lineage start in the same year.
        us = _unitsize(d, axis)
        rows = var.frame.select(axis, "cohort", "d")
        reach, rows_k = align_join_dtypes(counting, rows, [axis, "d"])
        reach = reach.join(rows_k, on=[axis, "cohort", "d"], how="semi")
        lhs: dict = {"retired": Sum(Where(at * us, reach), over=("d_divest",))}
        cohorts = _entity_rows(d.cohort_divest_built, "e", _members(d, axis), axis)
        here = cohorts.filter(pl.col("d_build").is_not_null())
        extended_here = here.filter(pl.col("kind") == KIND_EXTEND).select(axis, "cohort", "parent", "d_build")
        built_here = here.filter(pl.col("kind") != KIND_EXTEND).select(axis, "cohort", "d_build")
        if extended_here.height > 0:
            lhs["extended_neg"] = _extension_link(b, axis, rows, extended_here, us)
        if built_here.height > 0:
            invest = getattr(b, f"v_invest_{axis}", None)
            link, built_here = align_join_dtypes(rows, built_here, [axis])
            link = link.join(built_here, on=[axis, "cohort"], how="inner")
            columns = (pl.DataFrame(schema={axis: schema_dtype(None, axis), "d_build": schema_dtype(None, "d")})
                       if invest is None
                       else invest.frame.select(axis, "d").pipe(rename_to_axis, {"d": "d_build"}))
            link_k, columns = align_join_dtypes(link, columns, [axis, "d_build"])
            unbuilt = link_k.join(columns, on=[axis, "d_build"], how="anti")
            if unbuilt.height > 0:
                raise ValueError(
                    f"maxDivest_cohort_{axis}: the cohorts {sorted(set(unbuilt['cohort'].to_list()))} are decided in "
                    "this solve but have no investment column; their retirement would be unbounded.")
            built = _var_view(invest, "__cohort_build", {"d": "d_build"}, to_axis=True)
            lhs["invested_neg"] = -Sum(Where(built * us, link), over=("d_build",))
        remaining = cohorts.filter(pl.col("remaining") != 0.0).select(axis, "cohort", value=pl.col("remaining"))
        rhs: dict = {}
        if remaining.height > 0:
            rhs["remaining"] = Param((axis, "cohort"), remaining)
        m.add_cstr(
            f"maxDivest_cohort_{axis}",
            over      = rows,
            sense     = "<=",
            lhs_terms = lhs,
            rhs_terms = rhs,
        )


def _extension_link(b: "_Build", axis: str, rows: pl.DataFrame, extended: pl.DataFrame, us: Param):
    """``−v_extend[e, parent(c), d_build(c)]·us`` of the retirement rows ``rows`` (``[axis, cohort, d]``) of the
    life-extension cohorts ``extended`` (``[axis, cohort, parent, d_build]``) decided in this solve: what the
    extension holds."""
    var = getattr(b, f"v_extend_{axis}", None)
    link, extended = align_join_dtypes(rows, extended, [axis])
    link = link.join(extended, on=[axis, "cohort"], how="inner")
    columns = (pl.DataFrame(schema={axis: schema_dtype(None, axis), "parent": pl.Utf8,
                                    "d_build": schema_dtype(None, "d")})
               if var is None
               else var.frame.select(axis, pl.col("cohort").alias("parent"), "d")
                    .pipe(rename_to_axis, {"d": "d_build"}))
    link_k, columns = align_join_dtypes(link, columns, [axis, "d_build"])
    unbuilt = link_k.join(columns, on=[axis, "parent", "d_build"], how="anti")
    if unbuilt.height > 0:
        raise ValueError(f"maxDivest_cohort_{axis}: the life extensions {sorted(set(unbuilt['cohort'].to_list()))} "
                         "are decided in this solve but have no extension column; their retirement would be "
                         "unbounded.")
    view = _var_view(var, "__cohort_extend", {"cohort": "parent", "d": "d_build"}, to_axis=True)
    return -Sum(Where(view * us, link_k), over=("parent", "d_build"))


def _forecast_groups(window: pl.DataFrame) -> pl.DataFrame:
    """``[d_build, ref]``: each forecast period of a tied window pair ``(d_ref, b)`` (``b`` on the period axis) and
    its reference forecast ``d_ref`` (the forecasts of one anchor period share it)."""
    pairs = window.lazy().select(pl.col("d").alias("ref"), pl.col("b").alias("member"))
    return (pl.concat([pairs.select(member=pl.col("ref"), ref=pl.col("ref")), pairs.select("member", "ref")])
              .pipe(rename_to_axis, {"member": "d_build"})
              .unique().sort("d_build").collect())


def add_non_anticipativity(b: "_Build", window: pl.DataFrame) -> None:
    """non_anticipativity_divest_built_{p,n} over the invest-NA window ``window`` (``(d, b)``, ``b`` on the period
    axis) of a recourse-investment solve."""
    d = b.d

    # ─── EQ. ─── non_anticipativity_divest_built_{p,n} ───────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_divest_built[e,c,d] − v_divest_built[e,c^b,b] = 0
    #   ∀ (d,b) ∈ pd_non_anticipativity, (e,c,d) and (e,c^b,b) ∈ cd_divest_built
    #   c^b = c for a cohort not decided in a forecast of the window (decided before the fan, or carried);
    #     the cohort of e decided in the forecast of b's branch at the same anchor period as c's decision otherwise
    #     (the forecasts of a window anchor share its reference forecast)
    # A retirement inside the window is the same in every forecast branch, of the same cohort (a cohort decided
    # before the window) or of its twins (the forecasts' copies of one shared decision).
    window, groups = forecast_window(window)
    for axis in ("p", "n"):
        var = getattr(b, f"v_divest_built_{axis}", None)
        if var is None:
            continue
        cohorts = _entity_rows(d.cohort_divest_built, "e", _members(d, axis), axis).select(axis, "cohort", "d_build")
        over = tied_pairs(var.frame.select(axis, "cohort", "d"), cohorts, groups, window, axis)
        if over is None:
            continue
        b.m.add_cstr(
            f"non_anticipativity_divest_built_{axis}",
            over      = over,
            sense     = "==",
            lhs_terms = {"v_d":  var,
                         "v_b": -_var_view(var, "__nab", {"cohort": "cohort_b", "d": "b"})},
            rhs_terms = {},
        )


def forecast_window(window: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """``(window, groups)``: the invest-NA window ``(d, b)`` with ``b`` on the period axis, and its forecast groups
    (:func:`_forecast_groups`), the inputs of :func:`tied_pairs`."""
    window = window.with_columns(cast_dim(pl.col("b"), None, "d"))
    return window, _forecast_groups(window)


def tied_pairs(rows: pl.DataFrame, cohorts: pl.DataFrame, groups: pl.DataFrame, window: pl.DataFrame,
               axis: str) -> pl.DataFrame | None:
    """``[axis, cohort, d, cohort_b, b]``: the pairs of the columns ``rows`` (``[axis, cohort, d]``, one per cohort
    and period) a non-anticipativity row ties over the window ``window`` (``(d, b)``, ``b`` on the period axis)::

        c^b = c for a cohort not decided in a forecast of the window (``cohorts``' ``d_build`` null or outside it);
              the cohort of the same entity decided in the forecast of b's branch at the same anchor period as c's
              decision otherwise (``groups``: the forecasts of a window anchor share its reference forecast)

    ``None`` without a pair."""
    rows, cohorts = align_join_dtypes(rows, cohorts, [axis])
    rows = rows.join(cohorts, on=[axis, "cohort"], how="left")
    # The cohort's group: the reference forecast of its decision period (null outside the window).
    rows, groups_k = align_join_dtypes(rows, groups, ["d_build"])
    keyed = rows.join(groups_k, on="d_build", how="left")
    rows_w, window_k = align_join_dtypes(keyed, window, ["d"])
    left = rows_w.join(window_k, on="d", how="inner")
    if left.height == 0:
        return None
    right = (keyed.select(axis, pl.col("cohort").alias("cohort_b"), pl.col("d").alias("b"),
                          pl.col("ref").alias("ref_b")))
    same = (left.filter(pl.col("ref").is_null())
                .join(right.select(axis, "cohort_b", "b"), left_on=[axis, "cohort", "b"],
                      right_on=[axis, "cohort_b", "b"], how="inner")
                .with_columns(cohort_b=pl.col("cohort")))
    twin = (left.filter(pl.col("ref").is_not_null())
                .join(right.filter(pl.col("ref_b").is_not_null()), left_on=[axis, "b", "ref"],
                      right_on=[axis, "b", "ref_b"], how="inner"))
    over = (pl.concat([same.select(axis, "cohort", "d", "cohort_b", "b"),
                       twin.select(axis, "cohort", "d", "cohort_b", "b")])
              .unique().sort(axis, "cohort", "d", "b"))
    return over if over.height > 0 else None


def _cost(d, param: "Param | None", axis: str) -> "Param | None":
    """``param`` (``[e, cohort, d]``) on ``axis``, times ``pd_branch_weight`` under recourse investment (every branch
    carries its own retirement columns, so the cost is the expected cost); None stays None."""
    if param is None:
        return None
    frame = _entity_rows(param.frame, "e", _members(d, axis), axis)
    if getattr(d, "recourse_invest", False) and d.pd_branch_weight is not None:
        frame, weights = align_join_dtypes(frame, d.pd_branch_weight.frame.rename({"value": "w"}), ["d"])
        frame = (frame.join(weights, on="d", how="left")
                      .with_columns(value=pl.col("value") * pl.col("w").fill_null(1.0)).drop("w"))
    return Param((axis, "cohort", "d"), frame.select(axis, "cohort", "d", "value"))


def objective_terms(b: "_Build") -> list:
    """The ``divest_built_cost_{p,n}`` terms."""
    d = b.d
    terms: list = []
    for axis in ("p", "n"):
        var = getattr(b, f"v_divest_built_{axis}", None)
        if var is None:
            continue

        # ─── OBJ. ─── divest_built_cost_{p,n} ────────────────────────────────
        # ─────────────────────────────────────────────────────────────────────
        # Σ_{(e,c,d) ∈ cd_divest_built} v_divest_built[e,c,d]·us[e]·(settle[c,d] − a_b[c]·S̄[c,d] − fcs[c,d])·bw[d]
        #   (reinvest_choice / no_reinvestment); a reinvest_automatic cohort c, the root of its chain:
        #   settle[c,d] − a_b[c]·S̄[c,d] := [m > 0 or g = g1]·settle[g,d] − Σ_{g′ ≥ g} a_b[g′]·S̄[g′,d]
        #     g the generation of c in service in d, m = y[d] − y_on[g]; settle / S̄ of each generation as below with
        #     its own pay, n, r, y_on, y_eol, a_b of the plant as constructed in d_on(g) (on d's lineage in a fan)
        #   settle[c,d] = pay[c]·S[y_on, min(y[d], y_on + n)) + D(d)·(B[c,m] − S_d)
        #   B[c,m] = pay[c]·(1+r)^(1−off_inv)·ā(r, n − m) for m < n, else 0;  m = y[d] − y_on;
        #     ā(r,x) = (1 − (1+r)^−x)/r (x at r = 0)
        #   a_b[c] = a_cap[c] − a_salv[c] (invest_cost_p: the level yearly price, the salvage credit netted in);
        #     pay = A (invest_cost_p), n = payback_time (lifetime_technical without one), r = discount_rate, all of
        #     the plant as constructed (the cohort's columns, carried to the later solves)
        #   S[a,b) = Σ_{year rows of the run in [a,b)} σ·w·(1+social_discount_rate)^−(base + w·off_inv), σ the row's share
        #   D(d) = (1+social_discount_rate)^−(years from the run start to the start of d); S_d = 1000·net_salvage_value[e,d]
        #   S̄[c,d] = Σ_{d′ ∈ history ∪ solve, y_on ≤ y[d′] < y_eol, on d's lineage} inflation_inv[d′]
        #   fcs[c,d] = Σ_{d″ ∈ solve, y[d] ≤ y[d″] < y_eol, on d's lineage} 1000·fixed_cost[e,d″]·inflation_op[d″]
        #   us = unitsize (state_unitsize for _n), bw = pd_branch_weight under recourse (else 1)
        # A retired MW of a built cohort pays its contract as it actually ran: the capital payments due before d,
        # then at the start of d the capital still owed (the loan balance), less the net salvage value received
        # then.  The level price its years in service were charged (v_invest's, or an earlier solve's for a
        # carried cohort, history included) is taken back with its salvage credits, and the fixed cost from d on
        # is saved.  Building and retiring at once costs (invest_cost − net_salvage_value)·1000·D(d).  Retiring a
        # reinvest_automatic cohort also cancels every replacement still to come; a replacement retired at its own
        # start is never built (no settlement).
        us = _unitsize(d, axis)
        cost = _cost(d, d.cd_divest_built_cost, axis)
        fixed = _cost(d, d.cd_divest_built_fixed, axis)
        if cost is not None:
            terms.append(Sum(var * us * cost))
        if fixed is not None:
            terms.append(-Sum(var * us * fixed))
    return terms


__all__ = ["add_constraints", "add_non_anticipativity", "add_variables", "decided", "decided_frame",
           "forecast_window", "objective_terms", "options", "prepare", "retired", "retired_frame",
           "retirement_cells", "tied_pairs"]
