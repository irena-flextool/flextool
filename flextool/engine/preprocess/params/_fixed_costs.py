"""The lifetime fixed costs: the invest-side lump ``ed_lifetime_fixed_cost`` (the fixed cost of a decision's years in
service, :func:`lifetime_fixed_cost_invest_lf`), the side a retirement saves ``ed_lifetime_fixed_cost_divest``
(:func:`lifetime_fixed_cost_divest_lf`, :func:`apply_retirement_fixed_cost`) and the raw per-period fixed cost
``p_ed_fixed_cost`` (:func:`p_ed_fixed_cost_from_source`).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polar_high import Param

from flextool.engine.input._axis_enums import alias_to_axis, align_join_dtypes, schema_dtype
from flextool.engine.input._lifetime_roles import LIFETIME_TECHNICAL, NO_REINVESTMENT
from flextool.engine.preprocess.params._discount import _factor_frame, _walk_ops_factor
from flextool.engine.preprocess.params._existing import DIVEST_NOT_ALLOWED
from flextool.engine.preprocess.params._lifetime_inputs import (
    _all_entities_lf,
    _commission_walk_lf,
    _ed_fixed_cost_raw_lf,
    _ed_lifetime_on_lf,
    _entity_invest_set_lf,
    _lifetime_method_with_default_lf,
    _register_invest_lumps,
    _REGISTERED_RETIREMENT_LUMPS,
    _window_life_to_end_of_life,
    period_walk_iterator,
)
from flextool.engine.preprocess.params._period_values import at_online_period

# Substrate handle for the cascade-wide axis enum vocabulary (``None``: ``cast_dim`` / ``schema_dtype`` fall
# back to the live ContextVar ``_LIVE_AXIS_ENUMS_CTX``; see ``_npv``).
_enums: "dict | None" = None

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource


def _retire_window_lf(source: "InputSource",
                      active_solve: str | None,
                      ed_anchor_lf: pl.LazyFrame,
                      period_in_use: list[str],
                      period_universe: list[str],
                      *,
                      lineage: pl.DataFrame | None,
                      workdir: Path | None,
                      provider: "object | None",
                      ) -> pl.LazyFrame:
    """``[e, d, d_all]`` (Utf8): the solve's periods d_all a retirement decided in d counts in,
    yr[d] ≤ yr[d_all] < EOL[e] (the retirement reach of :func:`._cohorts.retire_active`), along the walk lineage
    of a fan."""
    walk = period_walk_iterator(source, active_solve, ed_anchor_lf, period_in_use, period_universe,
                                bounded=False, life_lf=None, factor_side=None, lineage=lineage)
    walk = walk.select(pl.col("e").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), pl.col("d_all").cast(pl.Utf8))
    # The periods past the end of life of the entity's existing capacity: a retirement no longer counts there.
    from flextool.engine.preprocess.params._cohorts import cohort_years, existing_cohorts, existing_expired
    years = cohort_years(source, active_solve, workdir, provider=provider)
    pairs = sorted(existing_expired(existing_cohorts(source, active_solve, workdir, provider=provider, years=years),
                                    years, [str(d) for d in period_in_use]))
    expired = pl.DataFrame({"e": [e for e, _ in pairs], "d_all": [d for _, d in pairs]},
                           schema={"e": pl.Utf8, "d_all": pl.Utf8})
    return walk.join(expired.lazy(), on=["e", "d_all"], how="anti")


def lifetime_fixed_cost_invest_lf(
        source: "InputSource",
        active_solve: str | None,
        period_with_history: list[str],
        period_in_use: list[str],
        period_universe: list[str],
        *,
        lineage: pl.DataFrame | None = None,
        commission: pl.DataFrame | None = None,
        year_rows: pl.DataFrame | None = None,
        provider: "object | None" = None,
        ) -> pl.LazyFrame:
    """Variant 3: ``ed_lifetime_fixed_cost`` (invest-side lifetime FC).

    Per (e ∈ entity, d ∈ period_with_history)::

        Σ_{d_all ∈ period_in_use, window(d, d_all)} fc[e, d_all] × ops_factor[d_all]
          = fc[e, c(d)] × Σ_{window} ops_factor[d_all] + Σ_{window} (fc[e, d_all] − fc[e, c(d)]) × ops_factor[d_all]

    the fixed cost of each period the capacity operates in, as existing capacity pays it.  The second form is how it
    is summed: the first sum is the walk's own factor sum and the second is exactly 0.0 where the fixed cost is the
    same in every period, so such a lump is the one of a fixed cost read once.  ``commission`` (``[e, d, d_c,
    yr_c]``): the window of a decision with a construction lead time starts at ``yr_c``, the period ``d_c`` its
    capacity comes online, and ends at the end of life of its cohort, the lifetime that of ``d_c``
    (:func:`._cohorts.invest_cohorts`, :func:`._period_values.at_online_period`).

    Same window dispatch as :func:`._npv.npv_invest_discounted_lf`
    (choice/no_invest → bounded; automatic → unbounded), with
    ops_factor (not inv_factor).  The bounded window ends at the cohort's
    rounded end of life ``y_eol = round_end_of_life(y_on + lifetime_technical,
    end_of_life_method)`` (:func:`._lifetime_inputs._window_life_to_end_of_life`): the fixed cost
    stops with the capacity, in the periods of ``cohort_alive``.

    ``lineage`` — see :func:`._walks.period_walk_iterator`; the
    :func:`._branch.annuity_window_lineage_df` frame in a stochastic fan, ``None`` otherwise.
    """
    if not period_with_history:
        return pl.LazyFrame(schema={
            "e": schema_dtype(_enums, "e"),
            "d": schema_dtype(_enums, "d"),
            "value": pl.Float64,
        })

    # Build (e, d) over all_entities × period_with_history.
    all_e_lf = _all_entities_lf(source)
    pwh_lf = pl.LazyFrame({"d": period_with_history}).with_columns(
        alias_to_axis("d", "d"))
    ed_anchor_lf = all_e_lf.join(pwh_lf, how="cross")

    fc_lf = at_online_period(ed_anchor_lf, lambda lf: _ed_fixed_cost_raw_lf(source, lf), commission, "fc")

    elm_lf = _lifetime_method_with_default_lf(source, all_e_lf)
    has_choice_or_no = (elm_lf
                          .filter(pl.col("method").is_in(
                              ["reinvest_choice", NO_REINVESTMENT]))
                          .select("e").unique()
                          .with_columns(has_choice=pl.lit(True)))
    has_automatic = (elm_lf
                       .filter(pl.col("method") == "reinvest_automatic")
                       .select("e").unique()
                       .with_columns(has_automatic=pl.lit(True)))

    # The lifetime of the period the capacity comes online (the cohort's end of life, _cohorts.invest_cohorts).
    life_lf = _ed_lifetime_on_lf(source, ed_anchor_lf, LIFETIME_TECHNICAL, commission)
    life_lf = _window_life_to_end_of_life(source, active_solve, life_lf, commission, provider=provider)
    commission_lf = _commission_walk_lf(commission)

    def walks(bounded: bool, factor_side: str | None) -> pl.LazyFrame:
        return period_walk_iterator(source, active_solve, ed_anchor_lf, period_in_use, period_universe,
                                    bounded=bounded, life_lf=life_lf, factor_side=factor_side, lineage=lineage,
                                    commission_lf=commission_lf, year_rows=year_rows)

    bounded = walks(True, "ops").rename({"factor": "factor_b"})
    unbounded = walks(False, "ops").rename({"factor": "factor_u"})
    ops = _walk_ops_factor(source, active_solve, period_universe, year_rows)
    corr_b = _fixed_cost_changes(source, walks(True, None), fc_lf, ops).rename({"corr": "corr_b"})
    corr_u = _fixed_cost_changes(source, walks(False, None), fc_lf, ops).rename({"corr": "corr_u"})

    out = (ed_anchor_lf
              .join(fc_lf, on=["e", "d"], how="left")
              .join(bounded, on=["e", "d"], how="left")
              .join(unbounded, on=["e", "d"], how="left")
              .join(corr_b, on=["e", "d"], how="left")
              .join(corr_u, on=["e", "d"], how="left")
              .join(has_choice_or_no, on="e", how="left")
              .join(has_automatic, on="e", how="left")
              .with_columns(
                  fc=pl.col("fc").fill_null(0.0),
                  factor_b=pl.col("factor_b").fill_null(0.0),
                  factor_u=pl.col("factor_u").fill_null(0.0),
                  corr_b=pl.col("corr_b").fill_null(0.0),
                  corr_u=pl.col("corr_u").fill_null(0.0),
                  has_choice=pl.col("has_choice").fill_null(False),
                  has_automatic=pl.col("has_automatic").fill_null(False),
              )
              .with_columns(
                  value=(
                      pl.when(pl.col("has_choice"))
                          .then(pl.col("fc") * pl.col("factor_b") + pl.col("corr_b"))
                          .otherwise(0.0)
                      + pl.when(pl.col("has_automatic"))
                          .then(pl.col("fc") * pl.col("factor_u") + pl.col("corr_u"))
                          .otherwise(0.0)
                  ),
                  _corr=(pl.when(pl.col("has_choice")).then(pl.col("corr_b")).otherwise(0.0)
                         + pl.when(pl.col("has_automatic")).then(pl.col("corr_u")).otherwise(0.0)),
              ))
    _notice_fixed_cost_per_operating_period(out)
    return out.select("e", "d", "value")


def _fixed_cost_changes(source: "InputSource", walk: pl.LazyFrame, fc_on: pl.LazyFrame,
                        ops: pl.LazyFrame) -> pl.LazyFrame:
    """``[e, d, corr]``: Σ over the window ``walk`` (``[e, d, d_all]``) of (fc[e, d_all] − fc_on[e, d]) ×
    factor[d_all] (``ops`` ``[d_all, factor]``; a period without a factor counts 1, as in the walk)."""
    periods = walk.select("e", pl.col("d_all").alias("d")).unique()
    fc_all = _ed_fixed_cost_raw_lf(source, periods).select("e", pl.col("d").alias("d_all"),
                                                           pl.col("fc").alias("_fc_all"))
    rows, fc_all = align_join_dtypes(walk, fc_all, ["e", "d_all"])
    rows = rows.join(fc_all, on=["e", "d_all"], how="left")
    rows, ops = align_join_dtypes(rows, ops, ["d_all"])
    rows = rows.join(ops, on="d_all", how="left")
    rows, fc_on = align_join_dtypes(rows, fc_on.select("e", "d", pl.col("fc").alias("_fc_on")), ["e", "d"])
    return (rows.join(fc_on, on=["e", "d"], how="left")
                .group_by("e", "d")
                .agg(((pl.col("_fc_all").fill_null(0.0) - pl.col("_fc_on").fill_null(0.0))
                      * pl.col("factor").fill_null(1.0)).sum().alias("corr")))


FIXED_COST_PER_OPERATING_PERIOD = "fixed_cost_per_operating_period"


def _notice_fixed_cost_per_operating_period(lumps: pl.LazyFrame) -> None:
    """The ``fixed_cost_per_operating_period`` notice for the entities its marker names (an upgraded database whose
    investing entities have a fixed_cost that differs between periods) whose lifetime fixed cost now differs from
    the fixed cost of the online period over the same years: ``lumps`` ``[e, d, value, _corr, ...]``, ``_corr``
    the difference (:func:`_fixed_cost_changes`).  Built only while the marker names entities."""
    from flextool.engine import _behaviour_changes as bc

    if not bc.has_marker(FIXED_COST_PER_OPERATING_PERIOD):
        return
    names = {str(n) for n in bc.marker_json(FIXED_COST_PER_OPERATING_PERIOD).get("entities") or []}
    if not names:
        return
    changed = (lumps.filter(pl.col("_corr").abs()
                            > 1e-12 * pl.max_horizontal(pl.lit(1.0), (pl.col("value") - pl.col("_corr")).abs()))
                    .select("e").unique().collect()["e"].to_list())
    hit = sorted(names & {str(e) for e in changed})
    if hit:
        bc.notice(FIXED_COST_PER_OPERATING_PERIOD, "entities", hit)


def lifetime_fixed_cost_divest_lf(
        source: "InputSource",
        active_solve: str | None,
        period_invest: list[str],
        period_in_use: list[str],
        period_universe: list[str],
        *,
        lineage: pl.DataFrame | None = None,
        workdir: Path | None = None,
        provider: "object | None" = None,
        year_rows: pl.DataFrame | None = None,
        ) -> pl.LazyFrame:
    """Variant 4: ``ed_lifetime_fixed_cost_divest`` (divest-side FC).

    Per (e ∈ entityDivest, d ∈ period_invest)::

        Σ_{d_all ∈ window(e, d)} fc[e, d_all] × ops_factor[d_all]

    the fixed cost the retired capacity no longer pays: each period's fixed cost of the existing capacity with
    the factor the objective charges it with (``p_inflation_op``), over the periods the retirement counts in
    (:func:`_retire_window_lf`), so a retired MW pays exactly no fixed cost there.

    ``lineage`` — see :func:`._walks.period_walk_iterator`; the
    :func:`._branch.annuity_window_lineage_df` frame in a stochastic fan, ``None`` otherwise.
    """
    if not period_invest:
        return pl.LazyFrame(schema={
            "e": schema_dtype(_enums, "e"),
            "d": schema_dtype(_enums, "d"),
            "value": pl.Float64,
        })
    entity_divest_set = _entity_invest_set_lf(source, DIVEST_NOT_ALLOWED)
    pi_lf = pl.LazyFrame({"d": period_invest}).with_columns(
        alias_to_axis("d", "d"))
    ed_anchor_lf = entity_divest_set.join(pi_lf, how="cross")
    window = _retire_window_lf(source, active_solve, ed_anchor_lf, period_in_use, period_universe,
                               lineage=lineage, workdir=workdir, provider=provider).collect()
    fc_all = (_ed_fixed_cost_raw_lf(source, window.lazy().select(alias_to_axis("e", "e"),
                                                                 alias_to_axis("d_all", "d")).unique())
              .select(pl.col("e").cast(pl.Utf8), pl.col("d").cast(pl.Utf8).alias("d_all"), "fc"))
    saved = (window.lazy()
               .join(fc_all, on=["e", "d_all"], how="left")
               .join(_factor_frame(source, active_solve, period_in_use, period_universe, provider, "ops_factor",
                                   year_rows),
                     on="d_all", how="left")
               .group_by("e", "d")
               .agg((pl.col("fc").fill_null(0.0) * pl.col("factor").fill_null(0.0)).sum().alias("saved")))
    keys = ed_anchor_lf.select("e", "d", _e=pl.col("e").cast(pl.Utf8), _d=pl.col("d").cast(pl.Utf8))
    return (keys
              .join(saved.rename({"e": "_e", "d": "_d"}), on=["_e", "_d"], how="left")
              .with_columns(value=pl.col("saved").fill_null(0.0))
              .select("e", "d", "value"))


def p_ed_fixed_cost_from_source(source: "InputSource",
                                    period_with_history: list[str],
                                    ) -> "Param | None":
    """Public entry: ``p_ed_fixed_cost[e, d]`` over ``period_with_history``.

    Replaces the workdir read of ``solve_data/ed_fixed_cost.csv``.
    Drops zero-valued rows (mirrors the loader's ``filter(value != 0)``).
    """
    if not period_with_history:
        return None
    all_e_lf = _all_entities_lf(source)
    pwh_lf = pl.LazyFrame({"d": period_with_history}).with_columns(
        alias_to_axis("d", "d"))
    ed_lf = all_e_lf.join(pwh_lf, how="cross")
    fc_lf = _ed_fixed_cost_raw_lf(source, ed_lf).rename({"fc": "value"})
    out = (fc_lf
              .filter(pl.col("value") != 0.0)
              .select("e", "d", "value")
              .sort("e", "d")
              .collect())
    if out.height == 0:
        return None
    return Param(("e", "d"), out)


def ed_lifetime_fixed_cost_from_source(
        source: "InputSource",
        active_solve: str | None,
        period_with_history: list[str],
        period_in_use: list[str],
        period_universe: list[str],
        *,
        lineage: pl.DataFrame | None = None,
        commission: pl.DataFrame | None = None,
        year_rows: pl.DataFrame | None = None,
        provider: "object | None" = None,
        ) -> "Param | None":
    """Public entry: ``ed_lifetime_fixed_cost[e, d]``.

    Unfiltered ``entity × period_with_history`` — see the rationale on
    :func:`._npv.ed_entity_annual_discounted_from_source`.

    ``lineage`` — see :func:`._walks.period_walk_iterator`; the
    :func:`._branch.annuity_window_lineage_df` frame in a stochastic fan, ``None`` otherwise.
    """
    if not period_with_history:
        return None
    out = (lifetime_fixed_cost_invest_lf(
              source, active_solve,
              period_with_history, period_in_use, period_universe,
              lineage=lineage, commission=commission, year_rows=year_rows, provider=provider)
              .select("e", "d", "value")
              .sort("e", "d")
              .collect())
    if out.height == 0:
        return None
    return Param(("e", "d"), out)


def ed_lifetime_fixed_cost_divest_from_source(
        source: "InputSource",
        active_solve: str | None,
        period_invest: list[str],
        period_in_use: list[str],
        period_universe: list[str],
        *,
        lineage: pl.DataFrame | None = None,
        workdir: Path | None = None,
        provider: "object | None" = None,
        year_rows: pl.DataFrame | None = None,
        ) -> "Param | None":
    """Public entry: ``ed_lifetime_fixed_cost_divest[e, d]``.

    Unfiltered ``entityDivest × period_invest`` — see the rationale on
    :func:`._npv.ed_entity_annual_discounted_from_source`.

    ``lineage`` — see :func:`._walks.period_walk_iterator`; the
    :func:`._branch.annuity_window_lineage_df` frame in a stochastic fan, ``None`` otherwise.
    """
    if not period_invest:
        return None
    out = (lifetime_fixed_cost_divest_lf(
              source, active_solve,
              period_invest, period_in_use, period_universe,
              lineage=lineage, workdir=workdir, provider=provider, year_rows=year_rows)
              .select("e", "d", "value")
              .sort("e", "d")
              .collect())
    if out.height == 0:
        return None
    return Param(("e", "d"), out)


def apply_retirement_fixed_cost(flex_data: object,
                                source: "InputSource",
                                active_solve: str,
                                period_invest: list[str],
                                period_in_use: list[str],
                                period_universe: list[str],
                                *,
                                lineage: pl.DataFrame | None,
                                year_rows: pl.DataFrame | None,
                                workdir: Path | None,
                                provider: "object | None" = None,
                                ) -> None:
    """Set ``ed_lifetime_fixed_cost_divest`` (the fixed cost a retirement of existing capacity saves,
    :func:`lifetime_fixed_cost_divest_lf`; ``None`` without a retirement option) and register its ``solve_data``
    frame (:func:`._lifetime_inputs._register_invest_lumps`).  The only derivation of the lump: a Spine solve
    (:func:`._lifetime_costs.apply_npv`) and a synthetic sub-solve with a divest set
    (:func:`._lifetime_costs.apply_run_fields_of_synthetic_solve`) call it alike."""
    flex_data.ed_lifetime_fixed_cost_divest = ed_lifetime_fixed_cost_divest_from_source(
        source, active_solve, period_invest, period_in_use, period_universe,
        lineage=lineage, workdir=workdir, provider=provider, year_rows=year_rows)
    _register_invest_lumps(flex_data, provider, _REGISTERED_RETIREMENT_LUMPS)


__all__ = [
    "p_ed_fixed_cost_from_source",
    "ed_lifetime_fixed_cost_from_source",
    "ed_lifetime_fixed_cost_divest_from_source",
]
