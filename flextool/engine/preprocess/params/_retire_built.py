"""Retirement of model-built capacity: the retirement options of each built cohort, their bounds and the checks.

A cohort built by the model (an investment decision of this solve, or one an earlier solve of the run realized and
handed on in ``realized_cohort``) can be retired when its entity's ``invest_method`` allows retirement and its
``lifetime_method`` is ``reinvest_choice`` or ``no_reinvestment`` (``reinvest_automatic`` capacity is renewed and
has no retirement of its own).  A retirement is decided in an investment period ``d`` in which the cohort is in
service and counts in every later period of the lineage while the cohort is in service (``dd_forward``, the
cohort's alive periods).  The frames (``FlexData`` fields, :func:`built_retirement_options`):

* ``cd_divest_built`` ``[e, cohort, d]`` — the retirement options: the columns of ``v_divest_built``;
* ``cdd_divest_built`` ``[e, cohort, d_divest, d]`` — a retirement of ``cohort`` decided in ``d_divest`` counts in
  ``d``;
* ``cohort_divest_built`` — one row per retirable cohort: ``d_build`` (the decision period of a cohort of this solve,
  the key of its ``v_invest``; null for a carried one), ``remaining`` (MW, MWh for storage: the carried size less
  the retirements earlier solves realized; 0 for a cohort of this solve), ``bound`` (the most the cohort can hold:
  ``remaining`` plus, for a cohort of this solve, ``p_entity_max_units · unitsize`` of its decision; +inf when that
  is unknown), its years ``y_on`` / ``y_eol`` and its prices (:data:`._cohorts.PRICE_COLUMNS`; a cohort of this
  solve gets them from the cohort table when ``_npv.apply_invest_lumps`` has set them).

A solve that places no retirement reach (no ``dd_forward``: a roll or a nested sub-solve) has no option.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import polars as pl

from flextool.engine.input._axis_enums import (
    alias_to_axis,
    align_join_dtypes,
    axis_lazyframe,
    rename_to_axis,
    schema_dtype,
)
from flextool.engine.preprocess.params._cohorts import BOUNDED_METHODS, KIND_INVEST, PRICE_COLUMNS

_INF = float("inf")
#: Capacities (MW, MWh) closer than this are equal.
_TOL = 1e-6


def cohort_columns() -> dict[str, pl.DataType]:
    """The columns of ``cohort_divest_built`` (module docstring)."""
    return {"e": schema_dtype(None, "e"), "cohort": pl.Utf8, "d_build": schema_dtype(None, "d_build"),
            "remaining": pl.Float64, "bound": pl.Float64, "y_on": pl.Float64, "y_eol": pl.Float64,
            **{c: pl.Float64 for c in PRICE_COLUMNS}}


@dataclass(frozen=True)
class BuiltRetirement:
    """The retirement frames of the model-built cohorts of a solve (module docstring); every frame has rows."""

    options: pl.DataFrame
    counting: pl.DataFrame
    cohorts: pl.DataFrame


def _carried_rows(carried: pl.DataFrame | None, retire: pl.DataFrame | None, entities: pl.DataFrame) -> pl.DataFrame:
    """``[e, cohort, remaining, y_on, y_eol, prices…]`` of the carried cohorts (handoff ``realized_cohort``) that can
    be retired: an investment cohort of a ``reinvest_choice`` / ``no_reinvestment`` entity of ``entities`` (``[e]``),
    ``remaining`` its size less its carried retirements (``realized_retire``)."""
    schema = {"e": schema_dtype(None, "e"), "cohort": pl.Utf8, "remaining": pl.Float64, "y_on": pl.Float64,
              "y_eol": pl.Float64, **{c: pl.Float64 for c in PRICE_COLUMNS}}
    if carried is None or carried.height == 0:
        return pl.DataFrame(schema=schema)
    rows = (carried.lazy()
                   .filter((pl.col("kind") == KIND_INVEST) & pl.col("method").is_in(list(BOUNDED_METHODS)))
                   .select(alias_to_axis("entity", "e"), pl.col("cohort"), pl.col("size").cast(pl.Float64),
                           pl.col("y_on").cast(pl.Float64), pl.col("y_eol").cast(pl.Float64).fill_null(_INF),
                           *(pl.col(c).cast(pl.Float64) for c in PRICE_COLUMNS)))
    rows, ents = align_join_dtypes(rows, entities.lazy(), ["e"])
    rows = rows.join(ents, on="e", how="semi")
    if retire is not None and retire.height > 0:
        retired = (retire.lazy()
                         .select(alias_to_axis("entity", "e"), pl.col("cohort"), pl.col("value").cast(pl.Float64))
                         .group_by("e", "cohort", maintain_order=True).agg(pl.col("value").sum().alias("retired")))
        rows, retired = align_join_dtypes(rows, retired, ["e"])
        rows = rows.join(retired, on=["e", "cohort"], how="left", maintain_order="left")
    else:
        rows = rows.with_columns(retired=pl.lit(0.0))
    return rows.select("e", "cohort", remaining=pl.col("size") - pl.col("retired").fill_null(0.0),
                       *(c for c in schema if c not in ("e", "cohort", "remaining"))).collect()


def built_retirement_options(cohort: pl.DataFrame | None,
                             cohort_alive: pl.DataFrame | None,
                             dd_forward: pl.DataFrame | None,
                             offered: pl.DataFrame | None,
                             carried: pl.DataFrame | None,
                             carried_retire: pl.DataFrame | None,
                             years: Mapping[str, float],
                             period_in_use: list[str],
                             ) -> BuiltRetirement | None:
    """The :class:`BuiltRetirement` frames of a solve, ``None`` without a retirement option of model-built capacity.

    ``cohort`` / ``cohort_alive`` / ``dd_forward`` are the solve's cohort frames, ``offered`` ``[e, d]`` the
    retirement options of entityDivest over the solve's investment periods (before the existing-capacity filter),
    ``carried`` / ``carried_retire`` the handoff ``realized_cohort`` / ``realized_retire``, ``years`` the period years
    of the run (:func:`._cohorts.cohort_years`) and ``period_in_use`` the solve's periods.  A carried cohort is in
    service in d when ``y_on ≤ y[d] < y_eol`` (the comparison of :func:`._cohorts.existing_capacity`)::

        options  = {(e, c, d) : (e, d) ∈ offered, (c, d) alive}
        counting = {(e, c, d′, d) : (e, c, d′) ∈ options, (d′, d) ∈ dd_forward, (c, d) alive}
    """
    if dd_forward is None or dd_forward.height == 0 or offered is None or offered.height == 0:
        return None
    entities = offered.select("e").unique()
    alive_parts: list[pl.DataFrame] = []
    here = pl.DataFrame(schema={"e": schema_dtype(None, "e"), "cohort": pl.Utf8,
                                "d_build": schema_dtype(None, "d_build"), "y_on": pl.Float64, "y_eol": pl.Float64,
                                **{c: pl.Float64 for c in PRICE_COLUMNS}})
    if cohort is not None and cohort.height > 0 and cohort_alive is not None and cohort_alive.height > 0:
        own, ents = align_join_dtypes(
            cohort.lazy().filter((pl.col("kind") == KIND_INVEST) & pl.col("method").is_in(list(BOUNDED_METHODS))
                                 & pl.col("y_on").is_not_null()),
            entities.lazy(), ["e"])
        here = (own.join(ents, on="e", how="semi")
                   .select("e", "cohort", "d_build", "y_on", pl.col("y_eol").fill_null(_INF), *PRICE_COLUMNS)
                   .collect())
        if here.height > 0:
            alive_parts.append(cohort_alive.lazy()
                               .filter(pl.col("kind") == KIND_INVEST)
                               .select("e", "cohort", "d")
                               .join(here.lazy().select("e", "cohort"), on=["e", "cohort"], how="semi")
                               .collect())
    carried_rows = _carried_rows(carried, carried_retire, entities)
    if carried_rows.height > 0:
        periods = (axis_lazyframe({"d": [str(d) for d in period_in_use]})
                   .with_columns(y=pl.Series("y", [float(years.get(str(d), 0.0)) for d in period_in_use],
                                             dtype=pl.Float64)))
        alive_parts.append(carried_rows.lazy().select("e", "cohort", "y_on", "y_eol")
                                       .join(periods, how="cross")
                                       .filter((pl.col("y_on").is_null() | (pl.col("y") >= pl.col("y_on")))
                                               & (pl.col("y") < pl.col("y_eol")))
                                       .select("e", "cohort", "d").collect())
    if not alive_parts:
        return None
    alive = alive_parts[0]
    for part in alive_parts[1:]:
        alive, part = align_join_dtypes(alive, part, ["e", "d"])
        alive = pl.concat([alive, part]).unique(maintain_order=True)
    alive, offered_u = align_join_dtypes(alive, offered.select("e", "d").unique(), ["e", "d"])
    options = alive.join(offered_u, on=["e", "d"], how="semi")
    if options.height == 0:
        return None
    counting = (options.lazy().pipe(rename_to_axis, {"d": "d_divest"})
                       .join(dd_forward.lazy().pipe(rename_to_axis, {"d_retire": "d_divest"}), on="d_divest",
                             how="inner")
                       .join(alive.lazy(), on=["e", "cohort", "d"], how="semi")
                       .select("e", "cohort", "d_divest", "d")
                       .sort("e", "cohort", "d_divest", "d").collect())
    cohorts = _cohort_table(here, carried_rows, options)
    return BuiltRetirement(options=options.sort("e", "cohort", "d"), counting=counting, cohorts=cohorts)


def _cohort_table(here: pl.DataFrame, carried: pl.DataFrame, options: pl.DataFrame) -> pl.DataFrame:
    """``cohort_divest_built`` (module docstring) of the cohorts with a retirement option.  A cohort both decided in
    this solve and carried (the same decision realized again) has both a ``d_build`` and a ``remaining``; its years
    and prices are this solve's."""
    keys = options.select("e", "cohort").unique(maintain_order=True)
    keys, own = align_join_dtypes(keys, here, ["e"])
    own = own.join(keys, on=["e", "cohort"], how="semi")
    keys, old = align_join_dtypes(keys, carried, ["e"])
    old = old.join(keys, on=["e", "cohort"], how="semi")
    both = (own.join(old.select("e", "cohort", "remaining"), on=["e", "cohort"], how="full", coalesce=True)
               .join(old.drop("remaining"), on=["e", "cohort"], how="left", suffix="_carried"))
    out = both.select(
        "e", "cohort", "d_build", remaining=pl.col("remaining").fill_null(0.0), bound=pl.lit(None, dtype=pl.Float64),
        y_on=pl.coalesce("y_on", "y_on_carried"), y_eol=pl.coalesce("y_eol", "y_eol_carried"),
        **{c: pl.when(pl.col("d_build").is_not_null()).then(pl.col(c)).otherwise(pl.col(f"{c}_carried"))
           for c in PRICE_COLUMNS})
    return out.select(list(cohort_columns())).sort("e", "cohort")


def with_bounds(cohorts: pl.DataFrame, flex_data: object, unitsize: Mapping[str, float]) -> pl.DataFrame:
    """``cohorts`` (``cohort_divest_built``) with ``bound``, the most a cohort can hold (MW, MWh): ``remaining``
    plus, for a cohort of this solve, the bound of its investment: ``p_entity_max_units[e, d_build] · unitsize[e]``
    (``maxInvest_var_bound``: a decision without a row may not invest), and for an entity with a per-period
    investment limit at most ``invest_max_period`` of the period the cohort comes online (``ed_invest_max_period``;
    no limit where it has no row).  +inf without ``p_entity_max_units`` or a unitsize."""
    out = cohorts.lazy()
    max_units = getattr(flex_data, "p_entity_max_units", None)
    if max_units is not None:
        units = (max_units.frame.lazy().select("e", "d", units=pl.col("value").cast(pl.Float64))
                                       .pipe(rename_to_axis, {"d": "d_build"}))
        out, units = align_join_dtypes(out, units, ["e", "d_build"])
        out = (out.join(units, on=["e", "d_build"], how="left", maintain_order="left")
                  .with_columns(units=pl.col("units").fill_null(0.0)))
    else:
        out = out.with_columns(units=pl.lit(None, dtype=pl.Float64))
    sizes = axis_lazyframe({"e": list(unitsize)}).with_columns(
        us=pl.Series("us", [float(v) for v in unitsize.values()], dtype=pl.Float64))
    out, sizes = align_join_dtypes(out, sizes, ["e"])
    out = out.join(sizes, on="e", how="left", maintain_order="left")
    period_cap = _period_caps(flex_data)
    if period_cap is not None:
        out, period_cap = align_join_dtypes(out, period_cap.lazy(), ["e"])
        out = out.join(period_cap, on=["e", "cohort"], how="left", maintain_order="left")
    else:
        out = out.with_columns(cap=pl.lit(None, dtype=pl.Float64))
    invest = pl.col("units") * pl.col("us")
    invest = pl.when(pl.col("cap").is_not_null()).then(pl.min_horizontal(invest, pl.col("cap"))).otherwise(invest)
    out = out.with_columns(bound=pl.when(pl.col("d_build").is_null()).then(pl.col("remaining"))
                           .when(pl.col("units").is_null() | pl.col("us").is_null()).then(_INF)
                           .otherwise(pl.col("remaining") + invest))
    return out.select(cohorts.columns).collect()


def _period_caps(flex_data: object) -> pl.DataFrame | None:
    """``[e, cohort, cap]``: ``invest_max_period`` (MW, MWh) of the period each investment cohort of the solve comes
    online in, for the entities with a per-period investment limit (``ed_invest_period_set``); None without."""
    cohort = getattr(flex_data, "cohort", None)
    caps = getattr(flex_data, "ed_invest_max_period", None)
    period_set = getattr(flex_data, "ed_invest_period_set", None)
    if cohort is None or caps is None or period_set is None or period_set.height == 0:
        return None
    online = (cohort.lazy().filter(pl.col("kind") == KIND_INVEST)
                    .select("e", "cohort", alias_to_axis("d_on", "d")))
    frame = caps.frame if hasattr(caps, "frame") else caps
    capped, period_set = align_join_dtypes(frame.lazy().select("e", "d", cap=pl.col("value").cast(pl.Float64)),
                                           period_set.lazy().select("e", "d"), ["e", "d"])
    capped = capped.join(period_set, on=["e", "d"], how="semi")
    online, capped = align_join_dtypes(online, capped, ["e", "d"])
    return online.join(capped, on=["e", "d"], how="inner").select("e", "cohort", "cap").collect()


def require_cohort_prices(cohorts: pl.DataFrame | None) -> None:
    """Raise :class:`FlexToolConfigError` when a retirable model-built cohort lacks one of the prices its retirement
    is settled with (a cohort carried from an earlier solve, or a user override of the carrier, without them):
    a missing price is never read as 0."""
    from flextool.engine.input._solve_state import FlexToolConfigError
    if cohorts is None or cohorts.height == 0:
        return
    missing = cohorts.filter(pl.any_horizontal(pl.col(c).is_null() for c in (*PRICE_COLUMNS, "y_on")))
    if missing.height == 0:
        return
    shown = []
    for row in missing.head(5).iter_rows(named=True):
        cols = [c for c in (*PRICE_COLUMNS, "y_on") if row[c] is None]
        shown.append(f"{row['e']} cohort {row['cohort']} ({', '.join(cols)})")
    raise FlexToolConfigError(
        "Model-built capacity can be retired, but its cohort lacks the prices its retirement is settled with: "
        + "; ".join(shown) + ". A cohort carried from an earlier solve (realized_cohort) must carry the columns "
        + ", ".join(PRICE_COLUMNS) + " and y_on.")


def check_minimums_inputs(cohorts: pl.DataFrame | None, options: pl.DataFrame | None
                          ) -> tuple[set[tuple[str, str]], dict[str, float], dict[tuple[str, str], float]]:
    """``(options, total, alive)`` for the minimum-retirement checks of the entities with retirable model-built
    capacity: the ``(e, d)`` with a retirement option, ``total[e]`` the most the cohorts of e can retire in the solve
    (Σ bound, +inf when a bound is unknown) and ``alive[(e, d)]`` the bounds of the cohorts with an option in d."""
    if cohorts is None or cohorts.height == 0 or options is None or options.height == 0:
        return set(), {}, {}
    bounds = cohorts.select("e", "cohort", pl.col("bound").fill_null(_INF))
    total = {str(e): float(v) for e, v in bounds.group_by("e").agg(pl.col("bound").sum()).iter_rows()}
    rows, bounds = align_join_dtypes(options.select("e", "cohort", "d"), bounds, ["e"])
    alive = {(str(e), str(d)): float(v) for e, d, v in
             rows.join(bounds, on=["e", "cohort"], how="left").group_by("e", "d").agg(pl.col("bound").sum())
                 .iter_rows()}
    return {(str(e), str(d)) for e, d in rows.select("e", "d").unique().iter_rows()}, total, alive


def warn_build_to_retire(pairs: pl.DataFrame) -> None:
    """Warn, once per run and entity, where building a cohort and retiring it earns money: ``pairs`` ``[e, cohort,
    total]`` (Utf8), ``total`` the objective coefficient of a MW built (its price and fixed cost) plus the smallest
    coefficient of retiring it, CUR/MW; a negative total lets the model build to retire, up to the investment
    limit (an LP without one is unbounded)."""
    from flextool.engine import _behaviour_changes
    names = sorted({str(e) for e, total in pairs.select("e", "total").iter_rows() if total < -_TOL})
    if not names:
        return
    _behaviour_changes.warn_once_each(
        "build_to_retire", names,
        lambda fresh: "Warning: building and retiring " + ", ".join(fresh) + " earns money: the net_salvage_value "
        "received at retirement is worth more than what the capacity costs until then (invest_cost, fixed_cost, "
        "discount_rate). The model builds as much as the investment limits allow just to retire it; without a "
        "limit the problem is unbounded. Check net_salvage_value against invest_cost.")


def built_retirable_notice(cohorts: pl.DataFrame | None) -> None:
    """The ``built_capacity_retirable`` notice for the entities its marker names (an upgraded database) whose
    model-built capacity has a retirement option in the solve."""
    from flextool.engine import _behaviour_changes as bc
    if cohorts is None or cohorts.height == 0 or not bc.has_marker(BUILT_CAPACITY_RETIRABLE):
        return
    names = {str(n) for n in bc.marker_json(BUILT_CAPACITY_RETIRABLE).get("entities") or []}
    hit = sorted(names & {str(e) for e in cohorts["e"].unique().to_list()})
    if hit:
        bc.notice(BUILT_CAPACITY_RETIRABLE, "entities", hit)


#: The behaviour change of the retirement of model-built capacity (``_behaviour_changes``).
BUILT_CAPACITY_RETIRABLE = "built_capacity_retirable"


__all__ = ["BUILT_CAPACITY_RETIRABLE", "BuiltRetirement", "built_retirable_notice", "built_retirement_options",
           "check_minimums_inputs", "cohort_columns", "require_cohort_prices", "warn_build_to_retire",
           "with_bounds"]
