"""§3.7.1, §3.7.4 — investment sets: the invest / retire method sets, ``ed_invest_set`` /
``ed_divest_set``, the retirable existing pairs and ``p_entity_max_units``."""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

from flextool.engine.input._axis_enums import alias_to_axis, align_join_dtypes, axis_lazyframe, schema_dtype
from flextool.engine.preprocess.params._period_values import _ed_period_value_map
from flextool.engine.preprocess.params._cohorts import cohort_years, existing_cohorts, no_reinvestment_closed
from flextool.engine.preprocess.params._existing import (
    DIVEST_NOT_ALLOWED as _DIVEST_NOT_ALLOWED,
    INVEST_METHODS_ALL as _INVEST_METHODS_ALL,
    INVEST_NOT_ALLOWED as _INVEST_NOT_ALLOWED,
)

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource
    from flextool.engine.preprocess._solve_context import SolveContext

from flextool.engine.preprocess.params._derived._scope import _enums, _try_entities, _try_param, _try_param_explicit
from flextool.engine.preprocess.params._derived._unitsize import UNITSIZE_DEFAULT, _entity_unitsize_lf
from flextool.engine.preprocess.params._derived._periods import (
    _expand_invest_branch_periods,
    _period_in_use_set,
    _solve_periods,
)
from flextool.engine.preprocess.params._derived._existing_capacity import p_entity_all_existing_from_source


def _entityInvest_set(source: "InputSource") -> pl.LazyFrame:
    """Entities (units + nodes + connections) whose
    ``invest_method`` ∉ ``_INVEST_NOT_ALLOWED``.  Returns a lazy frame
    with column ``e``.  Mirrors flextool's ``entityInvest`` set.
    """
    return _entity_invest_filter(source, _INVEST_METHODS_INVEST_ALLOWED)


def _entityDivest_set(source: "InputSource") -> pl.LazyFrame:
    """Entities whose ``invest_method`` ∉ ``_DIVEST_NOT_ALLOWED``.
    Mirrors flextool's ``entityDivest`` set.
    """
    return _entity_invest_filter(source, _INVEST_METHODS_DIVEST_ALLOWED)


# The invest/retire method lists (_INVEST_METHODS_ALL / _INVEST_NOT_ALLOWED /
# _DIVEST_NOT_ALLOWED) are imported from params/_existing.py, their one definition.
_INVEST_METHODS_INVEST_ALLOWED: frozenset[str] = (
    _INVEST_METHODS_ALL - _INVEST_NOT_ALLOWED
)


_INVEST_METHODS_DIVEST_ALLOWED: frozenset[str] = (
    _INVEST_METHODS_ALL - _DIVEST_NOT_ALLOWED
)


# Invest methods that lift every investment limit ("no_limit removes all
# limits even if there are limiting values set").  Both the invest-only
# and the invest+retire flavours qualify; the max-capacity cascade adds
# ``max_flow_for_unconstrained_variables`` for them.
INVEST_METHODS_INVEST_NO_LIMIT: frozenset[str] = frozenset((
    "invest_no_limit", "invest_retire_no_limit",
))


# Mirror ``invest_total_sets.py:25-32``.
_INVEST_METHODS_INVEST_TOTAL: frozenset[str] = frozenset((
    "invest_total", "invest_period_total",
    "invest_retire_total", "invest_retire_period_total",
))


_INVEST_METHODS_DIVEST_TOTAL: frozenset[str] = frozenset((
    "retire_total", "retire_period_total",
    "invest_retire_total", "invest_retire_period_total",
))


# Mirror ``invest_divest_sets.py:33-40``.
_INVEST_METHODS_INVEST_PERIOD: frozenset[str] = frozenset((
    "invest_period", "invest_period_total",
    "invest_retire_period", "invest_retire_period_total",
))


_INVEST_METHODS_DIVEST_PERIOD: frozenset[str] = frozenset((
    "retire_period", "retire_period_total",
    "invest_retire_period", "invest_retire_period_total",
))


def _entity_invest_method(source: "InputSource") -> pl.LazyFrame | None:
    """Pull the per-entity invest_method as a (e, method) lazy frame.

    Spine source: ``unit.invest_method`` / ``node.invest_method`` /
    ``connection.invest_method``.  Returns None when no entity has an
    explicit method (default is ``not_allowed``).
    """
    parts: list[pl.LazyFrame] = []
    for ec in ("unit", "node", "connection"):
        df = _try_param(source, ec, "invest_method")
        if df is None:
            continue
        parts.append(df.lazy().select(
            alias_to_axis("name", "e"),
            pl.col("value").alias("method"),
        ))
    if not parts:
        return None
    return pl.concat(parts).unique()


def _entity_invest_filter(source: "InputSource",
                           allowed: frozenset[str]) -> pl.LazyFrame:
    """Lazy frame ``[e]`` of entities whose ``invest_method ∈ allowed``.
    """
    eim = _entity_invest_method(source)
    if eim is None:
        return pl.DataFrame(schema={"e": schema_dtype(_enums, "e")}).lazy()
    return (eim.filter(pl.col("method").is_in(list(allowed)))
                .select("e").unique())


def ed_invest_set_from_source(source: "InputSource",
                                active_solve: str | None,
                                workdir: Path | None = None,
                                *,
                                ctx: "SolveContext | None" = None,
                                provider: "object | None" = None,
                                ) -> pl.DataFrame | None:
    """Compute the (entity, period) pairs where invest is allowed.

    ::

        ed_invest = { (e, d) : e ∈ entityInvest, d ∈ period_invest_of_solve }

    Every entity whose ``invest_method`` allows investing gets its investment variable in every
    investment period of the solve, also without an ``invest_cost`` (or with a 0 in a period):
    such capacity has no annuity, its ``fixed_cost`` still counts.  (Before, a pair needed a
    non-zero annuity or a user-constraint capacity coefficient; :func:`zero_cost_invest_notice`
    names the pairs that rule left out.)

    The pairs of entities with ``lifetime_method = no_reinvestment`` at or after the end of the first life of
    their existing capacity (:func:`._cohorts.no_reinvestment_closed`) are left out, so no investment variable
    exists for them: every set of investment variables is a partition of this set.
    """
    period_invest = _expand_invest_branch_periods(
        _solve_periods(source, active_solve, "invest_periods"),
        workdir, ctx=ctx, provider=provider)
    if not period_invest:
        # Empty invest_periods → empty ed_invest (the dispatch-only solve case).
        return pl.DataFrame(schema={"e": schema_dtype(_enums, "e"),
                                     "d": schema_dtype(_enums, "d")})

    out = (_entityInvest_set(source)
              .join(axis_lazyframe({"d": period_invest}), how="cross")
              .select("e", "d")
              .unique()
              .sort("e", "d")
              .collect())
    # Drop the pairs the no_reinvestment lifetime gate closes.
    years = cohort_years(source, active_solve, workdir, provider=provider)
    forbidden = no_reinvestment_closed(
        existing_cohorts(source, active_solve, workdir, provider=provider, years=years), years,
        _period_in_use_set(source, active_solve, workdir, provider=provider))
    if forbidden and out.height > 0:
        keep = [(e, d) for (e, d) in out.iter_rows()
                  if (e, d) not in forbidden]
        if not keep:
            out = pl.DataFrame(schema={"e": schema_dtype(_enums, "e"),
                                        "d": schema_dtype(_enums, "d")})
        else:
            # Preserve the axis Enum dtypes on the surviving (e, d) rows
            # so downstream partitions / joins see the axis-typed columns
            # rather than Utf8 from the Python-list rebuild.
            out = axis_lazyframe({
                "e": [e for (e, _d) in keep],
                "d": [d for (_e, d) in keep],
            }).collect()
    return out


def ed_divest_set_from_source(source: "InputSource",
                                active_solve: str | None,
                                workdir: Path | None = None,
                                *,
                                ctx: "SolveContext | None" = None,
                                provider: "object | None" = None,
                                existing_only: bool = True,
                                ) -> pl.DataFrame | None:
    """Mirror of ``ed_invest_set`` for divest::

        ed_divest = { (e, d) :
                       e ∈ entityDivest, d ∈ period_invest_of_solve,
                       pre_existing[e, d] > 0 }

    Also without a ``salvage_value``: retiring still saves the ``fixed_cost`` of the capacity.
    Only capacity that existed before the run can be retired: a period where the entity has no pre-existing
    capacity (none given, or past its end of life) has no retirement (:func:`retirable_existing_pairs`).
    ``existing_only=False`` gives the retirement options before that filter.
    """
    period_invest = _expand_invest_branch_periods(
        _solve_periods(source, active_solve, "invest_periods"),
        workdir, ctx=ctx, provider=provider)
    if not period_invest:
        return pl.DataFrame(schema={"e": schema_dtype(_enums, "e"),
                                     "d": schema_dtype(_enums, "d")})
    out = (_entityDivest_set(source)
              .join(axis_lazyframe({"d": period_invest}), how="cross")
              .select("e", "d")
              .unique()
              .sort("e", "d")
              .collect())
    return retirable_existing_pairs(out, source, active_solve, workdir, provider=provider) if existing_only else out


def retirable_existing_pairs(pairs: pl.DataFrame,
                             source: "InputSource",
                             active_solve: str | None,
                             workdir: Path | None,
                             *,
                             provider: "object | None" = None,
                             ) -> pl.DataFrame:
    """The ``[e, d]`` rows of ``pairs`` where the entity has pre-existing capacity (``p_entity_pre_existing`` > 0:
    the user's ``existing`` before the run, until its end of life), the only capacity a retirement can remove."""
    from flextool.engine.preprocess.params._existing import p_entity_pre_existing_lf
    if pairs.height == 0:
        return pairs
    periods = pairs["d"].cast(pl.Utf8).unique().sort().to_list()
    retirable = (p_entity_pre_existing_lf(source, active_solve, periods, workdir=workdir, provider=provider)
                   .filter(pl.col("value") > 0.0)
                   .select("e", "d")
                   .unique()
                   .collect())
    pairs, retirable = align_join_dtypes(pairs, retirable, ["e", "d"])
    return pairs.join(retirable, on=["e", "d"], how="semi").sort("e", "d")


# ---------------------------------------------------------------------------
# §3.7.4 — p_entity_max_units
# ---------------------------------------------------------------------------


def _entity_class_membership(source: "InputSource") -> dict[str, set[str]]:
    """Return ``{"unit": {...}, "node": {...}, "connection": {...}}``."""
    out: dict[str, set[str]] = {}
    for ec in ("unit", "node", "connection"):
        ents = _try_entities(source, ec)
        out[ec] = set() if ents is None else set(ents["name"].to_list())
    return out


def _entity_method_pairs(source: "InputSource") -> set[tuple[str, str]]:
    """Mirror flextool's ``entity__invest_method`` 2-col CSV — return
    ``{(e, method)}``.  Uses ``parameter`` (not _explicit) since
    ``invest_method`` has a non-broadcasting "not_allowed" default that
    SpineDbReader doesn't synthesise.
    """
    out: set[tuple[str, str]] = set()
    for ec in ("unit", "node", "connection"):
        df = _try_param(source, ec, "invest_method")
        if df is None:
            continue
        for e, m in df.select("name", "value").iter_rows():
            out.add((str(e), str(m)))
    return out


def _e_explicit_param(source: "InputSource", parameter_name: str
                       ) -> pl.LazyFrame:
    """Union explicit per-entity scalar parameter across unit + node +
    connection — returns ``[e, value]`` lazy frame (empty if no rows).
    """
    parts: list[pl.LazyFrame] = []
    for ec in ("unit", "node", "connection"):
        df = _try_param_explicit(source, ec, parameter_name)
        if df is None or df.height == 0:
            continue
        parts.append(df.lazy().select(
            alias_to_axis("name", "e"),
            pl.col("value").cast(pl.Float64),
        ))
    if not parts:
        return pl.LazyFrame(schema={"e": schema_dtype(_enums, "e"),
                                     "value": pl.Float64})
    return pl.concat(parts).unique(subset=["e"], keep="last")


def p_entity_max_units_from_source(source: "InputSource",
                                       ed_invest: pl.DataFrame | None,
                                       active_solve: str | None = None,
                                       workdir: Path | None = None,
                                       p_entity_all_existing: "Param | None" = None,
                                       *,
                                       ctx: "SolveContext | None" = None,
                                       provider: "object | None" = None,
                                       commission: "pl.DataFrame | None" = None,
                                       ) -> "Param | None":
    """Per-(e, d) maximum new-unit count = ``max_capacity / unitsize``.

    ``commission`` (``[e, d, d_c]``, the decisions with a construction lead time) reads the per-period
    invest_max_period of a decision at the period ``d_c`` its capacity starts to operate in: the per-period
    limits count capacity coming online.

    Mirrors flextool's ``entity_period_calc_params.py:1718-1761``:
      ``p_entity_max_capacity[e, d]`` is the per-(entity, period)
      capacity ceiling factoring in existing capacity, per-period and
      total invest caps, and the no-limit blanket cap
      (``invest_no_limit`` / ``invest_retire_no_limit``).
      ``p_entity_max_units[e, d] = max_capacity / unitsize`` (or 0 when
      unitsize is 0).  Caller filters value > 0.

    The returned Param mirrors the CSV-loader: only rows where
    ``value > 0`` after the divide.

    The ``p_entity_all_existing`` argument lets the caller pass the
    CSV-loaded lifetime cascade frame (which factors in
    ``previously_invested`` and lifetime-cumulative chains for
    multi-period invest fixtures).  When ``None`` we fall back to
    :func:`p_entity_all_existing_from_source` — sufficient for the
    no-prior-handoff single-year case.
    """
    # Gate: both ed_invest AND ed_divest must be empty for the
    # corresponding CSV ``_load_invest`` to blank-out.  We get
    # ``ed_invest`` here; the caller checks ed_divest analogously.
    if ed_invest is None or ed_invest.height == 0:
        return None

    piu = _period_in_use_set(source, active_solve, workdir, ctx=ctx, provider=provider)
    if not piu:
        return None
    period_set = set(piu)

    # Entities universe ∪ classes.
    cls_members = _entity_class_membership(source)
    entities = sorted(cls_members["unit"] | cls_members["node"]
                      | cls_members["connection"])
    if not entities:
        return None

    # Per-(e, method) pairs.
    method_pairs = _entity_method_pairs(source)
    has_no_limit_e = {e for (e, m) in method_pairs
                      if m in INVEST_METHODS_INVEST_NO_LIMIT}
    methods_by_e: dict[str, set[str]] = {}
    for (e, m) in method_pairs:
        methods_by_e.setdefault(e, set()).add(m)

    # entityInvest = entities with any allowed-invest method.
    entityInvest = {e for e, ms in methods_by_e.items()
                    if ms - _INVEST_NOT_ALLOWED}
    # e_invest_total = subset filtered by INVEST_TOTAL methods.
    e_invest_total = {e for e in entityInvest
                       if methods_by_e.get(e, set()) & _INVEST_METHODS_INVEST_TOTAL}

    # Unitsize cascade per entity.
    us_lf = _entity_unitsize_lf(source)
    us_df = us_lf.collect()
    us_map: dict[str, float] = dict(us_df.rows())

    # Existing capacity per (e, d).  Prefer the supplied lifetime cascade
    # frame (CSV-loaded ``p_entity_all_existing``), fall back to the
    # raw-existing helper.
    if p_entity_all_existing is not None:
        existing_map: dict[tuple[str, str], float] = {
            (e, d): v
            for e, d, v in p_entity_all_existing.frame.iter_rows()
        }
    else:
        pae_param = p_entity_all_existing_from_source(source, active_solve,
                                                         workdir=workdir,
                                                         ctx=ctx, provider=provider)
        if pae_param is None:
            existing_map = {}
        else:
            existing_map = {(e, d): v
                            for e, d, v in pae_param.frame.iter_rows()}

    # ─── PARAM ─── p_entity_max_units ───────────────────────────────────────────
    # p_entity_max_units[e,d] = max_capacity[e,d] / unitsize[e]       ∀ d ∈ period_in_use, kept where > 0
    # max_capacity[e,d] = cmc[e,d]                                      if (e,d) ∈ ed_invest_cumulative
    #                   = existing[e,d] + imp[e,d]                      if (e,d) ∈ ed_invest_period, e ∉ e_invest_total
    #                                   + eim[e]                        if e ∈ e_invest_total, (e,d) ∉ ed_invest_period
    #                                   + max(imp[e,d], eim[e])         if both
    #                                   + max_flow_for_unconstrained    if invest_method ∈ *_no_limit
    #   cmc[e,d] = inf (no limit in d): existing[e,d] + max_flow_for_unconstrained;  imp[e,d] = inf: imp :=
    #   max_flow_for_unconstrained (the bound of a *_no_limit method)
    # imp[e,d] / cmc[e,d] = authored[e,d]  (period map)
    #                     | authored[e]    (constant)
    #                     | authored[e, anchor(d)]  (recourse branch period without its own value)
    #                     | 0
    # imp / cmc are resolved for every d ∈ period_in_use (not only invest periods).
    # With a construction lead time imp[e,d] is imp[e,c(d)], at the period c(d) the decision's capacity starts to
    # operate in.
    #
    # invest_max_period[e, d]: constant or period map via the shape resolver, default 0.
    imp_map = _ed_period_value_map(source, "invest_max_period", piu)
    if commission is not None and commission.height > 0:
        imp_map = dict(imp_map)
        for _e, _o, _dc in commission.select(
                pl.col("e").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), pl.col("d_c").cast(pl.Utf8)).iter_rows():
            imp_map[(_e, _o)] = imp_map.get((_e, _dc), 0.0)
    # ed_invest_period: rows in ed_invest_pairs whose method ∈ INVEST_PERIOD.
    ed_invest_period = {(e, d) for (e, d) in ed_invest.iter_rows()
                         if methods_by_e.get(e, set()) & _INVEST_METHODS_INVEST_PERIOD}

    # invest_max_total[e]: explicit values, default 0.
    eim_df = _e_explicit_param(source, "invest_max_total").collect()
    eim_map: dict[str, float] = dict(eim_df.iter_rows())

    # cumulative_max_capacity[e, d]: constant or period map via the shape resolver, default 0.
    cmc_map = _ed_period_value_map(source, "cumulative_max_capacity", piu)
    # Entities that author cumulative_max_capacity at all (any period) — an entity
    # absent here never set the parameter, as opposed to an authored-but-zero value.
    cmc_authors = {e for (e, _d) in cmc_map}
    # ed_invest_cumulative: ed_invest pairs with cumulative_limits method.
    ed_invest_cumulative = {(e, d) for (e, d) in ed_invest.iter_rows()
                             if "cumulative_limits" in methods_by_e.get(e, set())}

    # p_max_flow_for_unconstrained_variables: the largest value over the
    # model entities, used as given (also below 1e6); 1e6 when none is set.
    # DB stores the model param as ``max_flow_for_unconstrained_variables``
    # (no ``p_`` prefix — that prefix is only the CSV-emit alias).
    p_unc = 1000000.0  # default per ``entity_period_calc_params.py:1689``.
    df_unc = _try_param(source, "model", "max_flow_for_unconstrained_variables")
    if df_unc is not None and df_unc.height > 0:
        try:
            p_unc = float(df_unc["value"].max())
        except Exception:
            pass
    # invest_max_period = inf is no limit in that period: the variable takes the no-limit bound
    # (max_flow_for_unconstrained_variables), as under an *_no_limit method; the per-period row is left out
    # (core._invest._eq_invest_period_caps).  cumulative_max_capacity = inf likewise: existing + the no-limit bound.
    imp_map = {k: (p_unc if math.isinf(v) and v > 0 else v) for k, v in imp_map.items()}
    cmc_unlimited = {k for k, v in cmc_map.items() if math.isinf(v) and v > 0}

    # Compute max_capacity per (e, d in period_in_use) and divide by unitsize.
    rows: list[tuple[str, str, float]] = []
    for e in entities:
        us = us_map.get(e, UNITSIZE_DEFAULT)
        in_total = e in e_invest_total
        has_no_limit = e in has_no_limit_e
        for d in piu:
            if d not in period_set:
                continue  # defensive; piu is the period_in_use list
            if (e, d) in ed_invest_cumulative:
                if e in cmc_authors:
                    # A period map that leaves out an investment period of the solve never gets here:
                    # pass 1b refuses it (``require_period_coverage``).  The 0.0 is only reached for
                    # a period the solve does not invest in.
                    v = (existing_map.get((e, d), 0.0) + p_unc if (e, d) in cmc_unlimited
                         else cmc_map.get((e, d), 0.0))
                else:
                    # A cumulative_limits entity that never authors
                    # cumulative_max_capacity (e.g. only cumulative_min_capacity, a floor with no
                    # ceiling) is unconstrained above, not pinned to v_invest = 0 — the floor would
                    # otherwise be unmeetable.  The real ceiling (if any) is still enforced by
                    # {max,min}Cumulative_capacity_{p,n} (_cumulative_invest.py); this value only
                    # bounds the variable itself.
                    v = existing_map.get((e, d), 0.0) + p_unc
            else:
                v = existing_map.get((e, d), 0.0)
                in_period = (e, d) in ed_invest_period
                imp = imp_map.get((e, d), 0.0)
                eim = eim_map.get(e, 0.0)
                if in_period and not in_total:
                    v += imp
                if in_total and not in_period:
                    v += eim
                if in_period and in_total:
                    v += max(imp, eim)
                if has_no_limit:
                    v += p_unc
            mu = (v / us) if us != 0.0 else 0.0
            if mu > 0.0:
                rows.append((e, d, float(mu)))
    if not rows:
        return None
    out = pl.DataFrame(rows, schema=["e", "d", "value"], orient="row").sort("e", "d")
    return Param(("e", "d"), out)
