"""The run-level fields of the reported costs (outputs only; no model term reads them): the run's investment lump
``ed_entity_annual_discounted_run`` and the investment cost lines ``edl_invest_lines_run``
(:func:`apply_run_cost_fields`, the vintages :func:`_vintage_lines`), the run's discount factors
``p_inflation_op_run`` / ``p_inflation_inv_run`` (:func:`apply_run_factors`) and the pre-existing capacity
``p_entity_pre_existing`` (:func:`apply_pre_existing_of_reported_costs`).  The ``OUT.`` block below is the formula.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polar_high import Param

from flextool.engine.input._axis_enums import alias_to_axis
from flextool.engine.input._lifetime_roles import LIFETIME_TECHNICAL
from flextool.engine.preprocess.params._cohorts import BOUNDED_METHODS
from flextool.engine.preprocess.params._discount import _inflation_factors_lf, _run_year_rows
from flextool.engine.preprocess.params._existing import INVEST_NOT_ALLOWED
from flextool.engine.preprocess.params._lifetime_inputs import (
    _ed_lifetime_on_lf,
    _ed_param,
    _entity_invest_set_lf,
    _lifetime_method_with_default_lf,
    _utf8_ed,
)
from flextool.engine.preprocess.params._npv import (
    _chain_years,
    _level_prices_lf,
    _lumps_of,
    chain_generations,
    priced_chain_rows,
)
from flextool.engine.preprocess.params._report_lines import event_period, stack_lines

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource


# ---------------------------------------------------------------------------
# Run-level fields of the reported costs (outputs only; no model term reads them)
# ---------------------------------------------------------------------------
#
# A solve discounts its own costs from the start of the run (:func:`._discount.solve_year_rows`), but its lumps cover
# only its own horizon: an investment is paid over the solve's periods, not over the periods of the whole run.  The
# reported costs use the fields below, discounted from the start of the run and summed over the periods of the
# whole run; on the solve's own periods the factors equal the solve's own (p_inflation_op).
#
# OUT. p_inflation_op_run[d] = Σ_y pyr[d,y]·(1+r)^-(yr_run[d] + Σ_{y'<y} pyr[d,y'] + pyr[d,y]·offset_ops)
#      p_inflation_inv_run[d] = the same with offset_inv
#   Discount factors of a period from the start of the run, over the solve's periods and the other run periods.
#   yr_run, pyr: the solve's own periods from its period_with_history and years_represented, the other run
#   periods from the run calendar.
# Window. The run periods d in which the capacity of decision (e, d_invest) operates: yr_run[d] ≥ yr_run[c(d_invest)],
#   and < y_eol = round_end_of_life(yr_run[c(d_invest)] + lifetime_technical[e, c(d_invest)], end_of_life_method[e])
#   for reinvest_choice / no_reinvestment (unbounded for reinvest_automatic): the periods its cohort is in service;
#   c(d) is the period the capacity comes online (d without a construction lead time), whose values describe the
#   plant as constructed.  In a stochastic fan only the pairs of the walk lineage (each calendar period once).
# OUT. edl_invest_lines_run[e,d,line,d_report] = the investment lump of a MW of decision (e, d) over its run
#   operating window split by event (_vintage_lines): investment a_cap·S̄ at its start of operation, salvage end of
#   life −a_salv·S̄ at its end of life (reinvest_choice / no_reinvestment; a_cap, a_salv of _level_prices_lf); for
#   reinvest_automatic each plant g of the chain (priced_chain_rows): investment (g1) / replacement a_cap[g]·S̄[g] at
#   its online period and salvage replaced −a_salv[g]·S̄[g] at its end of life (the last run period for a life to or
#   past the horizon end); S̄ the plant's Σ p_inflation_inv_run over its periods in service.  Σ_lines =
#   ed_entity_annual_discounted_run[e,d].
# OUT. ed_entity_annual_discounted_run[e,d] = Σ_{kind} price_kind[e,d]·Σ_{d' ∈ window(e,d,kind)} p_inflation_inv_run[d']
#   price_bounded = a_cap − a_salv, price_unbounded = the price of the generation in service in d' (the annuity
#   without generations): the objective's investment lump over the run periods.  The salvage settlement of a
#   retirement is reported as the objective charges it (ed_divest_salvage, discounted from the start of the run):
#   a one-off at the start of the retirement period.
# Notes: when the run is the solve itself (no run calendar, or every run period is in use and the solve starts
#   the run) the factors and the lump ARE the per-solve fields (byte-identical); only the lines are built.  An
#   entity with both a bounded and an unbounded lifetime method adds both sums, as the per-solve lump does.


def _run_window(anchors: pl.DataFrame,
                yr: dict[str, float],
                lineage: pl.DataFrame | None,
                period_in_use: list[str],
                ) -> pl.DataFrame:
    """``[e, d, kind, d_all]`` of the anchors ``[e, d, kind, yr_avail, life]`` (``kind`` ``bounded`` /
    ``unbounded``): the periods ``d_all`` from ``yr_avail`` on (bounded: before ``yr_avail + life``), with the
    lineage pairs only (:func:`._walks._apply_lineage_filter`) when a fan gives a lineage."""
    from flextool.engine.preprocess.params._run_calendar import YEAR_TOL
    from flextool.engine.preprocess.params._walks import _apply_lineage_filter
    periods = pl.DataFrame({"d_all": list(yr), "yr_dall": [float(v) for v in yr.values()]},
                           schema={"d_all": pl.Utf8, "yr_dall": pl.Float64})
    walk = (anchors.lazy()
              .join(periods.lazy(), how="cross")
              .filter(pl.col("yr_dall") >= pl.col("yr_avail") - YEAR_TOL)
              .filter((pl.col("kind") == "unbounded")
                      | (pl.col("yr_dall") < pl.col("yr_avail") + pl.col("life") - YEAR_TOL)))
    if lineage is not None:
        walk = _apply_lineage_filter(walk, lineage, [str(d) for d in period_in_use], pl.Utf8)
    return walk.select("e", "d", "kind", "d_all").collect()


def _lifetime_kinds(source: "InputSource", entities: pl.DataFrame) -> pl.DataFrame:
    """``[e, kind]``: ``bounded`` for an entity with a reinvest_choice / no_reinvestment lifetime method, ``unbounded``
    for one with reinvest_automatic (the default); both for an entity with both."""
    elm = _utf8_ed(_lifetime_method_with_default_lf(
        source, entities.lazy().select(alias_to_axis("e", "e"))), ("e",))
    return (pl.concat([
                elm.filter(pl.col("method").is_in(list(BOUNDED_METHODS))).select("e", kind=pl.lit("bounded")),
                elm.filter(pl.col("method") == "reinvest_automatic").select("e", kind=pl.lit("unbounded"))])
              .unique())


def _anchor_lf(pairs: pl.DataFrame) -> pl.LazyFrame:
    return pairs.lazy().select(alias_to_axis("e", "e"), alias_to_axis("d", "d"))


def _lump(walk: pl.DataFrame, coef: pl.DataFrame, factor: pl.DataFrame, anchors: pl.DataFrame) -> pl.DataFrame:
    """``[e, d, value]`` over ``anchors``: ``coef[e, d] × Σ_{d_all ∈ window} factor[d_all]``; with a ``kind`` column in
    ``coef`` (one price per lifetime kind of ``walk``) ``Σ_{kind} coef[e, d, kind] × Σ_{d_all ∈ window(kind)}
    factor[d_all]``."""
    keys = ["e", "d", "kind"] if "kind" in coef.columns else ["e", "d"]
    sums = (walk.join(factor, on="d_all", how="left")
                .group_by(keys)
                .agg(pl.col("factor").fill_null(1.0).sum().alias("sum")))
    parts = (sums.join(coef, on=keys, how="left")
                 .group_by("e", "d")
                 .agg((pl.col("coef").fill_null(0.0) * pl.col("sum")).sum().alias("value")))
    return (anchors.join(parts, on=["e", "d"], how="left")
                   .select("e", "d", value=pl.col("value").fill_null(0.0)))


def apply_run_cost_fields(flex_data: object,
                          source: "InputSource",
                          active_solve: str | None,
                          period_in_use: list[str],
                          period_invest: list[str],
                          workdir: Path | None,
                          *,
                          lineage: pl.DataFrame | None,
                          years: tuple[dict[str, float], dict[str, list[float]]] | None,
                          provider: "object | None" = None,
                          ) -> None:
    """Set the run-level investment lump and the investment cost lines of the reported costs (see the ``OUT.``
    block above) on ``flex_data``; the factors (``p_inflation_*_run``) must be set already.  ``years`` is
    :func:`._discount.run_years` (``None``: the lump is the per-solve one, and the windows walk the solve's own
    years)."""
    if years is None:
        from flextool.engine.preprocess.params._existing import _solve_walk_years
        walk_yr = _solve_walk_years(source, active_solve, workdir, provider) or {}
        yr = {str(d): walk_yr[str(d)] for d in period_in_use if str(d) in walk_yr}
    else:
        yr = years[0]
    commission = getattr(flex_data, "ed_invest_commission", None)
    if commission is not None and commission.height > 0:
        start = commission.select(pl.col("e").cast(pl.Utf8), pl.col("d").cast(pl.Utf8),
                                  pl.col("d_c").cast(pl.Utf8))
    else:
        start = pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8, "d_c": pl.Utf8})
        commission = None
    yr_df = pl.DataFrame({"d_c": list(yr), "yr_avail": [float(v) for v in yr.values()]},
                         schema={"d_c": pl.Utf8, "yr_avail": pl.Float64})

    def anchors_of(set_field: str, disallowed: frozenset[str], periods: list[str]) -> pl.DataFrame:
        base = _utf8_ed(_entity_invest_set_lf(source, disallowed), ("e",)).select("e")
        pairs = base.join(pl.DataFrame({"d": [str(d) for d in periods]}, schema={"d": pl.Utf8}), how="cross")
        offered = getattr(flex_data, set_field, None)
        if offered is not None and offered.height > 0:
            pairs = pl.concat([pairs, offered.select(pl.col("e").cast(pl.Utf8), pl.col("d").cast(pl.Utf8))])
        return pairs.unique().sort("e", "d")

    def year_of_start(pairs: pl.DataFrame, with_commission: bool) -> pl.DataFrame:
        out = pairs.join(start, on=["e", "d"], how="left") if with_commission else pairs.with_columns(
            d_c=pl.lit(None, dtype=pl.Utf8))
        out = out.with_columns(d_c=pl.coalesce("d_c", "d")).join(yr_df, on="d_c", how="left")
        missing = out.filter(pl.col("yr_avail").is_null())
        if missing.height > 0:
            raise ValueError(
                f"Solve '{active_solve}': the periods {sorted(set(missing['d_c'].to_list()))} have no year from the "
                "start of the run; the costs of their investments and retirements cannot be discounted.")
        return out.select("e", "d", "yr_avail")

    def factor_of(field: str) -> pl.DataFrame:
        param = getattr(flex_data, field, None)
        if param is None:
            return pl.DataFrame(schema={"d_all": pl.Utf8, "factor": pl.Float64})
        return _utf8_ed(param.frame.lazy(), ("d",)).select(pl.col("d").alias("d_all"),
                                                           pl.col("value").alias("factor"))

    factor_inv = factor_of("p_inflation_inv_run")

    # Investment side: vintages over the operating window from their start of operation.
    inv = anchors_of("ed_invest_set", INVEST_NOT_ALLOWED, period_invest)
    # The vintages: the decisions the solve offers (the lumps also cover the rest of the invest periods, as their
    # per-solve twins, but a decision not offered has no start of operation in the solve).
    offered_inv = getattr(flex_data, "ed_invest_set", None)
    vintages = inv if offered_inv is None or offered_inv.height == 0 else (
        offered_inv.select(pl.col("e").cast(pl.Utf8), pl.col("d").cast(pl.Utf8)).unique().sort("e", "d"))
    if inv.height > 0:
        # The price per year in service of each lifetime kind: the level yearly price a_cap less the level salvage
        # credit a_salv for reinvest_choice / no_reinvestment, the annuity for reinvest_automatic (the objective's,
        # npv_invest_discounted_lf; a reinvest_automatic chain at the price of each generation, below).
        kinds = _lifetime_kinds(source, inv.select("e").unique())
        ann = _utf8_ed(_level_prices_lf(source, active_solve, _anchor_lf(inv), commission, provider=provider))
        level = pl.col("a_cap") - pl.col("a_salv")
        price = (ann.join(kinds, on="e", how="inner")
                    .select("e", "d", "kind",
                            coef=pl.when(pl.col("kind") == "bounded").then(level).otherwise(pl.col("ann"))))
        # The two parts of that price: the capital (a_cap, the annuity for reinvest_automatic) and the salvage
        # credit (−a_salv; none for reinvest_automatic): the reported investment and salvage lines.
        bounded = pl.col("kind") == "bounded"
        split = (ann.join(kinds, on="e", how="inner")
                    .select("e", "d", "kind", inv=pl.when(bounded).then(pl.col("a_cap")).otherwise(pl.col("ann")),
                            salv=pl.when(bounded).then(-pl.col("a_salv")).otherwise(0.0)))
    if inv.height > 0:
        # The lifetime of the period the capacity comes online (the cohort's end of life).
        life = _utf8_ed(_ed_lifetime_on_lf(source, _anchor_lf(inv), LIFETIME_TECHNICAL, commission))
        anchors = (year_of_start(inv, True)
                     .join(life, on=["e", "d"], how="left")
                     .with_columns(life=pl.col("life").fill_null(0.0))
                     .join(_lifetime_kinds(source, inv.select("e").unique()), on="e", how="inner"))
        # The operating window ends at the cohort's rounded end of life: life = y_eol − yr_avail (_run_window
        # compares yr[d′] < yr_avail + life − YEAR_TOL, so the period starting at y_eol is out).
        from flextool.engine.preprocess.params._cohorts import (
            end_of_life_years,
            eol_methods_of,
            existing_cohorts,
            run_cohort_calendar,
        )
        eol = eol_methods_of(existing_cohorts(source, active_solve, workdir, provider=provider))
        y_eol = end_of_life_years(anchors.rename({"yr_avail": "y_on"}),
                                  run_cohort_calendar(source, active_solve, yr, provider), eol)
        anchors = anchors.with_columns(life=y_eol - anchors["yr_avail"])
        walk = _run_window(anchors, yr, lineage, period_in_use)
        chain_rows = _run_chain_rows(source, active_solve, walk, factor_inv, yr, flex_data, provider)
        ends = anchors.select("e", "d", "kind", y_eol=pl.col("yr_avail") + pl.col("life"))
        flex_data.edl_invest_lines_run = _vintage_lines(
            walk.join(vintages, on=["e", "d"], how="semi"), split, chain_rows, factor_inv, start, ends,
            chain_generations(getattr(flex_data, "cohort", None)), yr)
        if years is not None:
            # The price over the periods in service, as the objective's investment lump: a reinvest_automatic
            # chain at the price of the generation in service in each period (single price: the same product).
            chain = _lumps_of(chain_rows)
            single = chain.filter(pl.col("single")).select("e", "d", kind=pl.lit("unbounded"), _a_b="a_b")
            price = (price.join(single, on=["e", "d", "kind"], how="left")
                          .with_columns(coef=pl.coalesce("_a_b", "coef")).drop("_a_b"))
            multi = chain.filter(~pl.col("single")).select("e", "d", _chain="value")
            rest = walk.join(multi.select("e", "d").with_columns(kind=pl.lit("unbounded")),
                             on=["e", "d", "kind"], how="anti")
            lump = _lump(rest, price, factor_inv, inv)
            lump = (lump.join(multi, on=["e", "d"], how="left")
                        .with_columns(value=pl.col("value") + pl.col("_chain").fill_null(0.0)).drop("_chain"))
            flex_data.ed_entity_annual_discounted_run = _ed_param(lump)

    if years is None:
        _run_lumps_are_the_solves(flex_data)


def _run_chain_rows(source: "InputSource",
                    active_solve: str | None,
                    walk: pl.DataFrame,
                    factor: pl.DataFrame,
                    yr: dict[str, float],
                    flex_data: object,
                    provider: "object | None") -> pl.DataFrame | None:
    """:func:`._npv.priced_chain_rows` of the investment decisions of the solve's cohort table over the run's operating
    window ``walk`` (``[e, d, kind, d_all]``: its ``unbounded`` rows), the factors ``factor``
    (``p_inflation_inv_run``) and the run years ``yr``; ``None`` without a chain."""
    from flextool.engine.preprocess.params._cohorts import salvage_periods
    gens = chain_generations(getattr(flex_data, "cohort", None))
    rows = walk.filter(pl.col("kind") == "unbounded").select("e", "d", "d_all")
    if gens.height == 0 or rows.height == 0:
        return None
    years = _chain_years(source, active_solve, provider)
    last = salvage_periods(source, active_solve, years, provider, [None])[0]
    return priced_chain_rows(source, rows, factor, {**years, **yr}, gens, last)


def _vintage_lines(walk: pl.DataFrame,
                   split: pl.DataFrame,
                   chain_rows: pl.DataFrame | None,
                   factor: pl.DataFrame,
                   start: pl.DataFrame,
                   ends: pl.DataFrame,
                   gens: pl.DataFrame,
                   years: dict[str, float]) -> pl.DataFrame | None:
    """``[e, d, line, d_report, value]`` (Utf8, :mod:`._cost_lines`) of the vintages of ``walk`` (``[e, d, kind,
    d_all]``, their run operating window): the investment lump of a MW of each decision (the sum of
    :func:`._npv.chain_lumps` and the bounded and single-price lumps) split by event, each part Σ price·factor over the
    periods of its plant (``factor`` ``[d_all, factor]``)::

        investment          @ c(d)          = a_cap[e,c]·S̄          (the annuity for reinvest_automatic)
        salvage end of life @ d_eol         = −a_salv[e,c]·S̄         (reinvest_choice / no_reinvestment)
        investment / replacement @ d_on(g) = a_cap[g]·S̄[g]         (each plant g of a reinvest_automatic chain;
                                                                     replacement from its second plant on)
        salvage replaced    @ d_eol(g)      = −a_salv[g]·S̄[g]

    ``split`` ``[e, d, kind, inv, salv]`` the two parts of each price, ``start`` ``[e, d, d_c]`` the start of
    operation, ``ends`` ``[e, d, kind, y_eol]`` the end of life of a decision, ``gens`` :func:`._npv.chain_generations`
    (their ``y_eol``), ``chain_rows`` :func:`._npv.priced_chain_rows`; d_eol the period of ``years`` its end of life
    starts (the last period when it lives to or past the end of the horizon, :func:`._report_lines.event_period`).  None
    without a row."""
    from flextool.engine.preprocess.params import _cost_lines as L
    chained = (pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8}) if chain_rows is None
               else chain_rows.select("e", "d").unique())
    plain = (walk.join(chained.with_columns(kind=pl.lit("unbounded")), on=["e", "d", "kind"], how="anti")
                 .join(factor, on="d_all", how="left")
                 .group_by("e", "d", "kind", maintain_order=True)
                 .agg(pl.col("factor").fill_null(1.0).sum().alias("sum"))
                 .join(split, on=["e", "d", "kind"], how="left")
                 .join(start, on=["e", "d"], how="left")
                 .join(ends, on=["e", "d", "kind"], how="left")
                 .with_columns(d_start=pl.coalesce("d_c", "d")))
    plain = event_period(plain, "y_eol", years, "d_eol")
    parts = [stack_lines(plain, ("e", "d"), [
        (L.INVESTMENT, pl.col("inv").fill_null(0.0) * pl.col("sum"), pl.col("d_start")),
        (L.SALVAGE_END_OF_LIFE, pl.col("salv").fill_null(0.0) * pl.col("sum"), pl.col("d_eol"))])]
    if chain_rows is not None and chain_rows.height > 0:
        plants = (chain_rows.group_by("e", "d", "cohort", maintain_order=True)
                            .agg(pl.col("a_cap").first(), pl.col("a_salv").first(), pl.col("d_price").first(),
                                 pl.col("factor").sum().alias("sum"))
                            .join(gens.select("e", "cohort", "y_eol"), on=["e", "cohort"], how="left"))
        plants = event_period(plants, "y_eol", years, "d_eol")
        first = pl.col("cohort").str.ends_with(":1")
        parts += [stack_lines(plants.filter(first), ("e", "d"), [
                      (L.INVESTMENT, pl.col("a_cap") * pl.col("sum"), pl.col("d_price"))]),
                  stack_lines(plants.filter(~first), ("e", "d"), [
                      (L.REPLACEMENT, pl.col("a_cap") * pl.col("sum"), pl.col("d_price"))]),
                  stack_lines(plants, ("e", "d"), [
                      (L.SALVAGE_REPLACED, -pl.col("a_salv") * pl.col("sum"), pl.col("d_eol"))])]
    lines = (pl.concat(parts, how="vertical_relaxed")
               .group_by("e", "d", "line", "d_report", maintain_order=True).agg(pl.col("value").sum())
               .sort("e", "d", "line", "d_report"))
    return lines if lines.height > 0 else None


def _run_lumps_are_the_solves(flex_data: object) -> None:
    """The run is the solve: the run lumps are the solve's own lumps."""
    flex_data.ed_entity_annual_discounted_run = getattr(flex_data, "ed_entity_annual_discounted", None)


def apply_run_factors(flex_data: object,
                      source: "InputSource",
                      active_solve: str | None,
                      years: tuple[dict[str, float], dict[str, list[float]]] | None,
                      solve_inv_factor: "Param | None",
                      ) -> None:
    """Set ``p_inflation_op_run`` / ``p_inflation_inv_run`` (see the ``OUT.`` block above); with ``years`` None
    they are the solve's own factors (``p_inflation_op`` and ``solve_inv_factor``)."""
    if years is None:
        flex_data.p_inflation_op_run = flex_data.p_inflation_op
        flex_data.p_inflation_inv_run = solve_inv_factor
        return
    yr, widths = years
    factors = _inflation_factors_lf(source, active_solve, sorted(yr),
                                    year_rows=_run_year_rows(yr, widths)).sort("d").collect()
    flex_data.p_inflation_op_run = Param(("d",), factors.select("d", value=pl.col("ops_factor")))
    flex_data.p_inflation_inv_run = Param(("d",), factors.select("d", value=pl.col("inv_factor")))


def apply_pre_existing_of_reported_costs(flex_data: object,
                                         source: "InputSource",
                                         active_solve: str | None,
                                         period_in_use: list[str],
                                         workdir: Path | None,
                                         *,
                                         provider: "object | None" = None) -> None:
    """Set ``p_entity_pre_existing`` (e, d) over ``entity × period_in_use``: the existing capacity given by the user
    while its existing cohort is in service (:func:`._existing.p_entity_pre_existing_lf`, the cohort table), 0
    elsewhere, without the capacity carried from earlier solves and without their retirements.  The model's
    maximum retirement of existing capacity and the reported fixed cost of existing capacity are charged on it: the
    carried capacity's fixed cost is in the lump of the solve that invested in it, a retirement's saving in the lump
    of the retirement."""
    from flextool.engine.preprocess.params._existing import p_entity_pre_existing_lf
    frame = (p_entity_pre_existing_lf(source, active_solve, [str(d) for d in period_in_use], workdir=workdir,
                                      provider=provider)
               .select(alias_to_axis("e", "e"), alias_to_axis("d", "d"), "value")
               .sort("e", "d").collect())
    flex_data.p_entity_pre_existing = Param(("e", "d"), frame) if frame.height > 0 else None


__all__ = [
    "apply_run_cost_fields",
    "apply_run_factors",
]
