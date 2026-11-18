"""The salvage and replacement prices of existing capacity: the settlement of a retirement of existing capacity and
its level salvage credit (:func:`existing_salvage`, ``ed_divest_salvage`` / ``ed_existing_salvage_credit``,
:func:`apply_retirement_salvage`) and the prices of the replacements of an existing ``reinvest_automatic`` chain
(:func:`existing_replacement_prices`, :func:`apply_existing_replacement`).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import polars as pl
from polar_high import Param

from flextool.engine.input._axis_enums import align_join_dtypes
from flextool.engine.input._lifetime_roles import NET_SALVAGE_VALUE
from flextool.engine.preprocess._annuity import default_discount_rate
from flextool.engine.preprocess.params._cohorts import within_life
from flextool.engine.preprocess.params._discount import (
    _retirement_discount_lf,
    _run_inv_factors,
    _solve_inflation_scalars,
    run_years,
)
from flextool.engine.preprocess.params._existing import DIVEST_NOT_ALLOWED
from flextool.engine.preprocess.params._lifetime_inputs import _ed_param, _entity_invest_set_lf, _utf8_ed
from flextool.engine.preprocess.params._npv import (
    _chain_years,
    _labels_on_lineage,
    _notice_automatic_replacement_cost,
    _notice_net_salvage_once,
    _set_generation_prices,
    chain_generations,
    generation_prices,
)
from flextool.engine.preprocess.params._period_values import ed_values_on
from flextool.engine.preprocess.params._report_lines import event_period, report_years, stack_lines
from flextool.engine.preprocess.params._retire_settlement import chain_retirement_costs

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource


class ExistingSalvage(NamedTuple):
    """The salvage terms of existing capacity (:func:`existing_salvage`)."""

    divest: pl.DataFrame
    credit: pl.DataFrame
    prices: pl.DataFrame
    divest_lines: pl.DataFrame
    credit_lines: pl.DataFrame


#: The columns of :func:`existing_salvage`'s ``prices``: the cohort price columns of the existing capacity.
_EXISTING_PRICE_COLUMNS = ("r_loan", "a_salv", "nsv_eol")


def existing_salvage(source: "InputSource",
                     active_solve: str | None,
                     period_invest: list[str],
                     period_in_use: list[str],
                     period_universe: list[str],
                     period_with_history: list[str],
                     *,
                     lineage: pl.DataFrame | None,
                     year_rows: pl.DataFrame | None,
                     workdir: Path | None,
                     provider: "object | None",
                     run_factor: "Param | None" = None,
                     ) -> "ExistingSalvage":
    """``(divest_salvage, credit, prices, divest_lines, credit_lines)`` (:class:`ExistingSalvage`) of the existing
    capacity of the entities that may retire (entityDivest):
    the salvage term of the retirement coefficient, the level salvage credit of the objective constant and the
    existing cohort's prices (``divest_salvage_p`` / ``existing_salvage`` OBJ headers)::

        divest_salvage[e,d] = [y[d] < y_eol_x]·(D(d)·S_d − a_x·S̄_x[e,d]) − chain[e,d]
                                                                (e ∈ entityDivest, d ∈ period_invest)
        credit[e,d]         = a_x·inflation_inv[d]             (d ∈ period_in_use, y0 ≤ y[d] < y_eol_x)
        a_x  = salvage_credit(S_x, r_x, R_x, off_inv) = S_x·(1+r_x)^−R_x·ann(r_x, R_x)·(1+r_x)^(off_inv − 1)
        R_x  = y_eol_x − y0 when the life ends inside the run horizon (y_end_x ≤ H), else y_end_x − y0; 0 when
               the plant serves no period (y_eol_x ≤ y0); +inf (a_x = 0) without a lifetime
        chain[e,d] = :func:`._retire_settlement.chain_retirement_costs` of the replacements of a
               reinvest_automatic entity's existing capacity (its generations g ≥ 2, priced as one plant each; 0
               for other entities)
        S̄_x[e,d] = Σ_{d′ ∈ history ∪ period_in_use, y0 ≤ y[d′] < y_eol_x, on d's lineage} inflation_inv[d′]
        S_d  = 1000·net_salvage_value[e,d];  S_x = 1000·net_salvage_value[e, d_eol_x] (the last run period's value
               after the last period start, :func:`._cohorts.salvage_periods`);  r_x = discount_rate[e, d0]

    For a reinvest_automatic entity the existing plant is the first generation of its existing chain (the plant in
    service at the run start, :func:`._cohorts.chain_rows`): y_end_x / y_eol_x are its own, and a retirement after
    its end of life retires a replacement (``chain``: the replacement's contract as it ran, and the level price of
    it and of every later replacement given back).
    y0 the run start (period d0), y_eol_x / y_end_x the rounded / unrounded end of life of the existing cohort, H
    the end of the run horizon, D(d) :func:`._discount._retirement_discount_lf`, inflation_inv the investment factor
    from the start of the run (``run_factor``, ``p_inflation_inv_run``: the history periods' too; the solve's own
    factor, equal on its periods, where it has none).  The existing capacity's investment is sunk; it holds its salvage
    value, received at its end of life, as a level credit a_x over its service years from the run start (the rounded
    span: the credits sum to its salvage once, never more).  Retiring it in d receives S_d at the start of d and
    gives back every credit of the cohort (history included, along d's lineage of a stochastic fan, each calendar
    period once), so a retired MW earns exactly S_d·D(d).  Rows of ``divest_salvage`` only where S_d or a_x is not
    0; ``credit`` only where a_x is not 0; ``prices`` ``[e, r_loan, a_salv, nsv_eol]`` (r_x, a_x, S_x) of every
    entity of entityDivest with an existing cohort.  All frames Utf8 keyed.

    The line frames (``[e, d, line, d_report, value]``, :mod:`._cost_lines`): ``divest_lines`` the events of
    −divest_salvage (the cost of a MW retired in d): ``salvage economic`` @ d = −D(d)·S_d, the credits given back
    a_x·S̄_x[e,d] @ the existing plant's end of life (``salvage end of life``; ``salvage replaced`` for a
    reinvest_automatic plant) and the lines of ``chain`` (:func:`._retire_settlement.chain_retirement_costs`);
    ``credit_lines`` the credit of a MW of existing capacity in d, −credit[e,d] as ``salvage pre-existing`` @ the
    plant's end of life."""
    from flextool.engine.preprocess.params import _cost_lines as L
    from flextool.engine.preprocess._annuity import salvage_credit
    from flextool.engine.preprocess.params._cohorts import (
        REINVEST_AUTOMATIC,
        _run_calendar_of,
        cohort_years,
        existing_cohorts,
        existing_generations,
        salvage_periods,
    )
    from flextool.engine.preprocess.params._derived import run_start_period
    from flextool.engine.preprocess.params._run_calendar import YEAR_TOL
    empty_ed = pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8, "value": pl.Float64})
    empty_prices = pl.DataFrame(schema={"e": pl.Utf8, **{c: pl.Float64 for c in _EXISTING_PRICE_COLUMNS}})
    no_lines = stack_lines(pl.DataFrame(), ("e", "d"), [])
    nothing = ExistingSalvage(empty_ed, empty_ed, empty_prices, no_lines, no_lines)
    divesting = _utf8_ed(_entity_invest_set_lf(source, DIVEST_NOT_ALLOWED), ("e",)).select("e").unique()
    years = cohort_years(source, active_solve, workdir, provider=provider)
    start = run_start_period(years)
    if divesting.height == 0 or start is None:
        return nothing
    y0 = float(years[start])
    horizon = _run_calendar_of(source, active_solve, years, provider).horizon_end
    first = (existing_generations(source, active_solve, workdir, provider=provider, years=years)
             .filter(pl.col("cohort").str.ends_with(":1"))
             .select(pl.col("e").cast(pl.Utf8), _y_end=pl.col("y_end"), _y_eol=pl.col("y_eol")))
    automatic = pl.col("method") == REINVEST_AUTOMATIC
    cohorts = (existing_cohorts(source, active_solve, workdir, provider=provider, years=years)
               .select(pl.col("e").cast(pl.Utf8), "method", "y_end", "y_eol")
               .join(divesting, on="e", how="semi")
               .join(first, on="e", how="left")
               .with_columns(y_end=pl.when(automatic).then(pl.col("_y_end")).otherwise(pl.col("y_end")),
                             y_eol=pl.when(automatic).then(pl.col("_y_eol")).otherwise(pl.col("y_eol")))
               .drop("_y_end", "_y_eol"))
    if cohorts.height == 0:
        return nothing
    span = (pl.when(pl.col("y_end").is_null() | pl.col("y_eol").is_null()).then(float("inf"))
              .when(automatic & (pl.col("y_eol") <= y0 + YEAR_TOL)).then(0.0)
              .when(pl.lit(horizon is not None) & (pl.col("y_end") <= (horizon or 0.0) + YEAR_TOL))
              .then(pl.col("y_eol") - y0)
              .otherwise(pl.col("y_end") - y0))
    cohorts = cohorts.with_columns(span=span, d_salv=pl.Series(
        "d_salv", salvage_periods(source, active_solve, years, provider, cohorts["y_eol"].to_list()), dtype=pl.Utf8))
    nsv_x = (ed_values_on(source, NET_SALVAGE_VALUE, cohorts.lazy().select("e", d=pl.col("d_salv")), fill=0.0)
             .select(pl.col("e").cast(pl.Utf8), s_x=pl.col("value") * 1000.0).collect())
    rate_x = (ed_values_on(source, "discount_rate", cohorts.lazy().select("e", d=pl.lit(str(start))),
                           fill=default_discount_rate(source))
              .select(pl.col("e").cast(pl.Utf8), r_x=pl.col("value")).collect())
    offset = _solve_inflation_scalars(source)[1]
    cohorts = (cohorts.join(nsv_x, on="e", how="left").join(rate_x, on="e", how="left")
                      .with_columns(a_x=salvage_credit(pl.col("s_x"), pl.col("r_x"), pl.col("span"), offset)))
    prices = cohorts.select("e", r_loan=pl.col("r_x"), a_salv=pl.col("a_x"), nsv_eol=pl.col("s_x")).sort("e")

    factors, walk_periods, universe = _run_inv_factors(source, active_solve, period_in_use, period_universe,
                                                       period_with_history, years, provider=provider,
                                                       year_rows=year_rows, run_factor=run_factor)
    in_use = {str(d) for d in period_in_use}
    alive = (cohorts.filter(pl.col("a_x") != 0.0).select("e", "y_eol", "a_x")
                    .join(walk_periods, how="cross")
                    .filter(within_life(pl.col("y"), pl.lit(y0), pl.col("y_eol")))
                    .join(factors, on="d_all", how="left"))
    missing = alive.filter(pl.col("factor").is_null())
    if missing.height > 0:
        raise ValueError(f"Solve '{active_solve}': the periods {sorted(set(missing['d_all'].to_list()))} have no "
                         "investment discount factor; the salvage credit of existing capacity cannot be discounted.")
    credit = (alive.filter(pl.col("d_all").is_in(list(in_use)))
                   .select("e", pl.col("d_all").alias("d"), value=pl.col("a_x") * pl.col("factor")))

    anchors = divesting.join(pl.DataFrame({"d": [str(d) for d in period_invest]}, schema={"d": pl.Utf8}),
                             how="cross")
    take_back = anchors.join(alive, on="e", how="inner")
    if lineage is not None:
        pairs = lineage.select(pl.col("d").cast(pl.Utf8), pl.col("d_other").cast(pl.Utf8).alias("d_all"))
        both = pl.col("d").is_in(list(in_use)) & pl.col("d_all").is_in(list(in_use))
        take_back = pl.concat([take_back.filter(~both),
                               take_back.filter(both).join(pairs, on=["d", "d_all"], how="semi")])
    take_back = (take_back.group_by("e", "d")
                          .agg((pl.col("a_x") * pl.col("factor")).sum().alias("taken")))
    s_d = (ed_values_on(source, NET_SALVAGE_VALUE, anchors.lazy(), fill=0.0)
           .select(pl.col("e").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), s_d=pl.col("value") * 1000.0).collect())
    disc = _retirement_discount_lf(source, active_solve, universe, provider=provider, year_rows=year_rows).collect()
    # The existing plant is retired only while it is in service; later, a replacement of a reinvest_automatic chain.
    chain_years = _chain_years(source, active_solve, provider)
    y_d = pl.DataFrame({"d": list(chain_years), "y_d": [float(v) for v in chain_years.values()]},
                       schema={"d": pl.Utf8, "y_d": pl.Float64})
    in_service = (anchors.join(cohorts.select("e", "y_eol"), on="e", how="inner").join(y_d, on="d", how="left")
                         .filter(pl.col("y_d").is_null() | within_life(pl.col("y_d"), None, pl.col("y_eol")))
                         .select("e", "d"))
    parts = (in_service.join(s_d, on=["e", "d"], how="left")
                       .join(take_back, on=["e", "d"], how="left")
                       .join(disc, on="d", how="left")
                       .with_columns(s_d=pl.col("s_d").fill_null(0.0), taken=pl.col("taken").fill_null(0.0),
                                     disc=pl.col("disc").fill_null(1.0)))
    divest = parts.select("e", "d", value=pl.col("disc") * pl.col("s_d") - pl.col("taken"))
    reported = report_years(source, active_solve, period_in_use, workdir, provider)
    ends = event_period(cohorts.select("e", "method", "y_eol"), "y_eol", reported, "d_eol")
    end_line = pl.when(pl.col("method") == REINVEST_AUTOMATIC).then(pl.lit(L.SALVAGE_REPLACED)).otherwise(
        pl.lit(L.SALVAGE_END_OF_LIFE))
    timed = parts.join(ends, on="e", how="left")
    own_lines = pl.concat([
        stack_lines(timed, ("e", "d"), [(L.SALVAGE_ECONOMIC, -pl.col("disc") * pl.col("s_d"), pl.col("d"))]),
        *(stack_lines(timed.filter(end_line == line), ("e", "d"), [(line, pl.col("taken"), pl.col("d_eol"))])
          for line in (L.SALVAGE_END_OF_LIFE, L.SALVAGE_REPLACED))])
    credit_lines = stack_lines(credit.join(ends, on="e", how="left"), ("e", "d"),
                               [(L.SALVAGE_PRE_EXISTING, -pl.col("value"), pl.col("d_eol"))])
    replacements = (existing_generations(source, active_solve, workdir, provider=provider, years=years)
                    .filter(~pl.col("cohort").str.ends_with(":1")))
    chain, chain_lines = chain_retirement_costs(
        source, active_solve, chain_generations(replacements, "existing"),
        anchors.join(cohorts.filter(automatic).select("e"), on="e", how="semi")
               .with_columns(root=pl.lit(f"existing:{start}")),
        period_in_use, period_universe, period_with_history, lineage=lineage, year_rows=year_rows, workdir=workdir,
        provider=provider, run_factor=run_factor, per_plant=True)
    changed = pl.concat([chain.filter(pl.col("value") != 0.0).select("e"),
                         cohorts.filter(automatic & (pl.col("a_x") != 0.0)).select("e")]).unique()
    _notice_automatic_replacement_cost(changed.lazy().with_columns(changed=pl.lit(True)))
    divest = (pl.concat([divest, chain.select("e", "d", value=-pl.col("value"))])
                .group_by("e", "d", maintain_order=True).agg(pl.col("value").sum())
                .filter(pl.col("value") != 0.0)
                .sort("e", "d"))
    divest_lines = (pl.concat([own_lines, chain_lines.drop("root")], how="vertical_relaxed")
                      .group_by("e", "d", "line", "d_report", maintain_order=True).agg(pl.col("value").sum())
                      .sort("e", "d", "line", "d_report"))
    return ExistingSalvage(divest, credit.sort("e", "d"), prices, divest_lines,
                           credit_lines.sort("e", "d", "line", "d_report"))


def existing_replacement_prices(source: "InputSource",
                                active_solve: str | None,
                                periods: list[str],
                                years: dict[str, float],
                                *,
                                lineage: pl.DataFrame | None,
                                workdir: Path | None,
                                provider: "object | None") -> pl.DataFrame:
    """``[e, d_on, d, value, pay, nsv_eol, a_cap, a_salv, y_eol]`` (Utf8 keys): the price per year in service
    (CUR/MW, MWh for storage; a_b = a_cap − a_salv of :func:`._npv.generation_prices` as one plant, ``per_plant``) of
    the replacement of a reinvest_automatic entity's existing capacity in service in each period ``d`` of ``periods``
    (``years`` their run years), ``d_on`` the period it comes online in, ``y_eol`` its end of life; the existing
    plant itself (the first generation, sunk) has none.  A replacement
    starting inside a stochastic fan takes the values of ``d``'s branch (:func:`._npv._labels_on_lineage`)."""
    from flextool.engine.preprocess.params._cohorts import existing_generations, salvage_periods
    empty = pl.DataFrame(schema={"e": pl.Utf8, "d_on": pl.Utf8, "d": pl.Utf8, "value": pl.Float64})
    gens = chain_generations(existing_generations(source, active_solve, workdir, provider=provider), "existing")
    gens = gens.filter(~pl.col("cohort").str.ends_with(":1"))
    yd = pl.DataFrame({"d": [d for d in periods if d in years], "y_d": [float(years[d]) for d in periods if d in years]},
                      schema={"d": pl.Utf8, "y_d": pl.Float64})
    if gens.height == 0 or yd.height == 0:
        return empty
    rows = (gens.join(yd, how="cross")
                .filter(within_life(pl.col("y_d"), pl.col("y_on"), pl.col("y_eol"))))
    if rows.height == 0:
        return empty
    last = salvage_periods(source, active_solve, years, provider, [None])[0]
    rows = _labels_on_lineage(rows, years, lineage, last)
    prices = generation_prices(source, rows.select("e", "cohort", d=pl.col("d_price"), d_salv=pl.col("d_salv"))
                               .unique(maintain_order=True), per_plant=True)
    return (rows.join(prices.rename({"d": "d_price"}), on=["e", "cohort", "d_price", "d_salv"], how="left")
                .select("e", d_on=pl.col("d_price"), d="d", value=pl.col("a_b").fill_null(0.0),
                        pay=pl.col("pay").fill_null(0.0), nsv_eol=pl.col("nsv_eol").fill_null(0.0),
                        a_cap=pl.col("a_cap").fill_null(0.0), a_salv=pl.col("a_salv").fill_null(0.0),
                        y_eol="y_eol")
                .sort("e", "d"))


def _existing_replacement_lines(prices: pl.DataFrame, factor: "Param | None",
                                years: dict[str, float]) -> pl.DataFrame | None:
    """``[e, d, line, d_report, value]`` (Utf8, :mod:`._cost_lines`) per MW of existing reinvest_automatic capacity in
    the run period d (``prices``: :func:`existing_replacement_prices` over the run's periods; ``factor``
    ``p_inflation_inv_run``): ``investment pre-existing`` @ d_on = a_cap[g]·inflation_inv[d] and ``salvage
    pre-existing`` @ the end of life of g = −a_salv[g]·inflation_inv[d], g the replacement in service in d (the
    objective constant ``existing_replacement`` over the run); None without a replacement."""
    from flextool.engine.preprocess.params import _cost_lines as L
    if prices.height == 0 or factor is None:
        return None
    f = factor.frame.select(pl.col("d").cast(pl.Utf8), factor=pl.col("value"))
    timed = event_period(prices.join(f, on="d", how="inner"), "y_eol", years, "d_eol")
    lines = stack_lines(timed, ("e", "d"), [
        (L.INVESTMENT_PRE_EXISTING, pl.col("a_cap") * pl.col("factor"), pl.col("d_on")),
        (L.SALVAGE_PRE_EXISTING, -pl.col("a_salv") * pl.col("factor"), pl.col("d_eol"))])
    return lines.sort("e", "d", "line", "d_report") if lines.height > 0 else None


def warn_free_replacement_salvage(prices: pl.DataFrame, existing: "Param | None") -> None:
    """Warn, once per run and entity, where existing reinvest_automatic capacity (``existing`` ``[e, d, value]`` > 0
    in some period) is replaced by a free plant (``prices`` :func:`existing_replacement_prices`: no invest_cost,
    ``pay`` 0) that has a net_salvage_value (``nsv_eol`` not 0): the replacement keeps its salvage value (a plant
    priced at −salvage credit), so keeping the capacity earns that salvage and a retirement gives it up."""
    from flextool.engine import _behaviour_changes
    if prices.height == 0 or existing is None:
        return
    held = {str(e) for e, v in existing.frame.select(pl.col("e").cast(pl.Utf8), "value").iter_rows() if v and v > 0}
    free = prices.filter((pl.col("pay") == 0.0) & (pl.col("nsv_eol") != 0.0))
    names = sorted({str(e) for e in free["e"].to_list()} & held)
    if names:
        _behaviour_changes.warn_once_each(
            "free_replacement_salvage", names,
            lambda fresh: "Warning: " + ", ".join(fresh) + ": the existing capacity (lifetime_method "
            "reinvest_automatic) is replaced at its end of life by a plant without an invest_cost that still has a "
            "net_salvage_value: the replacement is free and keeps its salvage value, so keeping the capacity earns "
            "that salvage value and a retirement gives it up. Set invest_cost if the replacement, and so the "
            "retirement, should be priced.")


def apply_existing_replacement(flex_data: object,
                               source: "InputSource",
                               active_solve: str | None,
                               period_in_use: list[str],
                               *,
                               lineage: pl.DataFrame | None,
                               workdir: Path | None,
                               provider: "object | None" = None) -> None:
    """Set ``ed_existing_replacement`` (``[e, d]``: the price per year of the replacement of existing
    reinvest_automatic capacity in service in d times the solve's investment factor inflation_inv[d], CUR/MW; the
    objective constant ``existing_replacement`` with include_existing_fixed_cost), its reported twin
    ``edl_existing_replacement_lines`` (the cost lines per MW of existing capacity in the run's periods,
    :func:`_existing_replacement_lines`) and the existing generations' prices in ``flex_data.cohort``.  ``None``
    without a replacement in the periods."""
    years = _chain_years(source, active_solve, provider)
    run = run_years(source, active_solve, list(period_in_use), workdir, provider=provider)
    run_periods = list(run[0]) if run is not None else [str(d) for d in period_in_use]
    reported = existing_replacement_prices(source, active_solve, run_periods,
                                           {**years, **(run[0] if run is not None else {})}, lineage=lineage,
                                           workdir=workdir, provider=provider)
    flex_data.edl_existing_replacement_lines = _existing_replacement_lines(
        reported, getattr(flex_data, "p_inflation_inv_run", None),
        report_years(source, active_solve, list(period_in_use), workdir, provider))
    prices = existing_replacement_prices(source, active_solve, [str(d) for d in period_in_use], years,
                                         lineage=lineage, workdir=workdir, provider=provider)
    warn_free_replacement_salvage(prices, getattr(flex_data, "p_entity_pre_existing", None))
    factor = getattr(flex_data, "p_inflation_inv_run", None)
    if prices.height == 0 or factor is None:
        flex_data.ed_existing_replacement = None
    else:
        f = factor.frame.select(pl.col("d").cast(pl.Utf8), factor=pl.col("value"))
        flex_data.ed_existing_replacement = _ed_param(
            prices.join(f, on="d", how="inner").select("e", "d", value=pl.col("value") * pl.col("factor"))
                  .filter(pl.col("value") != 0.0))
    from flextool.engine.preprocess.params._cohorts import existing_generations
    gens = chain_generations(existing_generations(source, active_solve, workdir, provider=provider), "existing")
    _set_generation_prices(flex_data, source, active_solve, gens.filter(~pl.col("cohort").str.ends_with(":1")),
                           provider=provider, per_plant=True)


def apply_retirement_salvage(flex_data: object,
                             source: "InputSource",
                             active_solve: str,
                             period_invest: list[str],
                             period_in_use: list[str],
                             period_universe: list[str],
                             period_with_history: list[str],
                             *,
                             lineage: pl.DataFrame | None,
                             year_rows: pl.DataFrame | None,
                             workdir: Path | None,
                             provider: "object | None" = None,
                             ) -> None:
    """Set the salvage terms of the retirement of existing capacity (:func:`existing_salvage`):
    ``ed_divest_salvage`` (the salvage part of the ``v_divest`` coefficient), ``ed_existing_salvage_credit`` (the
    level salvage credit of the objective constant), the existing cohort's prices (``r_loan``,
    ``a_salv``, ``nsv_eol``) and the cost lines of the reported costs (``edl_divest_lines``: per MW retired,
    ``edl_existing_credit_lines``: per MW of existing capacity; :func:`existing_salvage`).  The only derivation of these
    terms: a Spine solve (:func:`._lifetime_costs.apply_npv`) and a synthetic sub-solve
    (:func:`._lifetime_costs.apply_run_fields_of_synthetic_solve`) call it alike."""
    salvage = existing_salvage(
        source, active_solve, period_invest, period_in_use, period_universe, period_with_history,
        lineage=lineage, year_rows=year_rows, workdir=workdir, provider=provider,
        run_factor=getattr(flex_data, "p_inflation_inv_run", None))
    divest, credit, prices = salvage.divest, salvage.credit, salvage.prices
    flex_data.edl_divest_lines = salvage.divest_lines if salvage.divest_lines.height > 0 else None
    flex_data.edl_existing_credit_lines = salvage.credit_lines if salvage.credit_lines.height > 0 else None
    _notice_net_salvage_once(divest["e"].unique().to_list())
    flex_data.ed_divest_salvage = _ed_param(divest) if divest.height > 0 else None
    flex_data.ed_existing_salvage_credit = _ed_param(credit) if credit.height > 0 else None
    _set_existing_cohort_prices(flex_data, prices)


def _set_existing_cohort_prices(flex_data: object, prices: pl.DataFrame) -> None:
    """``flex_data.cohort`` with the prices ``[e, r_loan, a_salv, nsv_eol]`` (:func:`existing_salvage`) on the existing
    cohort of each entity of ``prices``."""
    cohort = getattr(flex_data, "cohort", None)
    if cohort is None or cohort.height == 0 or prices.height == 0:
        return
    from flextool.engine.preprocess.params._cohorts import KIND_EXISTING
    new = prices.rename({c: f"_{c}" for c in _EXISTING_PRICE_COLUMNS})
    left, new = align_join_dtypes(cohort, new, ["e"])
    existing = pl.col("kind") == KIND_EXISTING
    flex_data.cohort = (left
                        .join(new, on="e", how="left", maintain_order="left")
                        .with_columns(pl.when(existing & pl.col(f"_{c}").is_not_null()).then(pl.col(f"_{c}"))
                                      .otherwise(pl.col(c)).alias(c) for c in _EXISTING_PRICE_COLUMNS)
                        .select(cohort.columns))
