"""The retirement settlement: a retired MW of a model-built cohort (:func:`built_retirement_costs`,
``cd_divest_built_cost`` / ``cd_divest_built_fixed``) or of a ``reinvest_automatic`` chain
(:func:`chain_retirement_costs`) pays its contract as it ran, the level price given back and the fixed cost saved
(:func:`_settlements`); :func:`apply_built_retirement_costs` sets them on ``flex_data``.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polar_high import Param

from flextool.engine.input._axis_enums import alias_to_axis, align_join_dtypes, schema_dtype
from flextool.engine.input._lifetime_roles import NET_SALVAGE_VALUE
from flextool.engine.preprocess.params._cohorts import PRICE_COLUMNS, within_life
from flextool.engine.preprocess.params._discount import (
    _factor_frame,
    _payment_rows,
    _retirement_discount_lf,
    _run_inv_factors,
    _solve_inflation_scalars,
)
from flextool.engine.preprocess.params._lifetime_inputs import _ed_fixed_cost_raw_lf
from flextool.engine.preprocess.params._npv import (
    _chain_years,
    _labels_on_lineage,
    _notice_automatic_replacement_cost,
    chain_generations,
    generation_prices,
)
from flextool.engine.preprocess.params._period_values import ed_values_on
from flextool.engine.preprocess.params._report_lines import event_period, report_years, stack_lines

# Substrate handle for the cascade-wide axis enum vocabulary (``None``: ``cast_dim`` / ``schema_dtype`` fall
# back to the live ContextVar ``_LIVE_AXIS_ENUMS_CTX``; see ``_npv``).
_enums: "dict | None" = None

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource


def built_retirement_costs(source: "InputSource",
                           active_solve: str | None,
                           cohorts: pl.DataFrame,
                           options: pl.DataFrame,
                           period_in_use: list[str],
                           period_universe: list[str],
                           period_with_history: list[str],
                           *,
                           lineage: pl.DataFrame | None,
                           year_rows: pl.DataFrame | None,
                           workdir: Path | None,
                           provider: "object | None",
                           run_factor: "Param | None" = None,
                           ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """``(cost, fixed, lines)`` ``[e, cohort, d, value]`` of the retirement options ``options`` (``cd_divest_built``) of the
    model-built cohorts ``cohorts`` (``cohort_divest_built`` with their prices), CUR/MW (MWh for storage): the
    ``divest_built_cost_{p,n}`` OBJ header::

        cost[e,c,d]  = set[c,d] − (a_cap[c] − a_salv[c])·S̄[c,d]
        set[c,d]     = pay[c]·S[y_on, min(y[d], y_on + n)) + D(d)·(B[c,m] − S_d)
        B[c,m]       = pay[c]·(1+r)^(1−off_inv)·ā(r, n − m) for m < n, else 0;  m = y[d] − y_on
        S̄[c,d]       = Σ_{d′ ∈ history ∪ solve, y_on ≤ y[d′] < y_eol, on d's lineage} inflation_inv_run[d′]
        fixed[e,c,d] = Σ_{d″ ∈ solve, y[d] ≤ y[d″] < y_eol, on d's lineage} 1000·fixed_cost[e,d″]·inflation_op[d″]

    pay, n = n_pay, r = r_loan, a_cap, a_salv, y_on, y_eol the cohort's (``_cohorts.PRICE_COLUMNS``); ā(r, x) =
    (1 − (1+r)^−x)/r (x at r = 0); S[a, b) = Σ σ·contrib over the year rows of the run
    (:func:`._discount._payment_rows`), σ the share of the row inside [a, b); D(d) the discount of the start of d
    (:func:`._discount._retirement_discount_lf`); S_d = 1000·net_salvage_value[e, d]; years of the run
    (:func:`._cohorts.cohort_years`), the lineage pairs of the investment walks.  A retired MW pays its contract as it
    ran: the payments due before d, then at the start of d the capital still owed, less its salvage value received then;
    the level price its years in service were charged is taken back (history included: a carried cohort's price was
    charged by an earlier solve) and the fixed cost from d on is saved.

    ``lines`` ``[e, cohort, d, line, d_report, value]`` (Utf8, :mod:`._cost_lines`) split ``cost`` into its events::

        retirement           @ d       = cash[c,d] − a_cap[c]·S̄[c,d]     (the contract's payments less the capital
                                                                          its years in service were charged)
        salvage economic     @ d       = −D(d)·S_d                        (the salvage value received)
        salvage end of life  @ d_eol   = a_salv[c]·S̄[c,d]                (the salvage credits given back, where the
                                                                          plant's end-of-life credit is reported)

    ``cash = pay·S[y_on, min(y[d], y_on + n)) + D(d)·B``, d_eol the period of the plant's end of life
    (:func:`._report_lines.event_period` on :func:`._report_lines.report_years`)."""
    from flextool.engine.preprocess.params import _cost_lines as L
    empty = pl.DataFrame(schema={"e": schema_dtype(_enums, "e"), "cohort": pl.Utf8, "d": schema_dtype(_enums, "d"),
                                 "value": pl.Float64})
    parts = _settlements(source, active_solve, cohorts, options, period_in_use, period_universe, period_with_history,
                         lineage=lineage, year_rows=year_rows, workdir=workdir, provider=provider,
                         run_factor=run_factor)
    if parts is None:
        return empty, empty, stack_lines(pl.DataFrame(), ("e", "cohort", "d"), [])
    keys = ["e", "cohort", "d"]
    out = parts.with_columns(value=pl.col("settle") - (pl.col("a_cap") - pl.col("a_salv")) * pl.col("taken"))
    timed = event_period(out, "y_eol", report_years(source, active_solve, period_in_use, workdir, provider),
                         "d_eol")
    lines = stack_lines(timed, ("e", "cohort", "d"), [
        (L.RETIREMENT, pl.col("cash") - pl.col("a_cap") * pl.col("taken"), pl.col("d")),
        (L.SALVAGE_ECONOMIC, -pl.col("sold"), pl.col("d")),
        (L.SALVAGE_END_OF_LIFE, pl.col("a_salv") * pl.col("taken"), pl.col("d_eol"))])
    return out.select(*keys, "value"), out.select(*keys, value=pl.col("saved")), lines


def _settlements(source: "InputSource",
                 active_solve: str | None,
                 cohorts: pl.DataFrame,
                 options: pl.DataFrame,
                 period_in_use: list[str],
                 period_universe: list[str],
                 period_with_history: list[str],
                 *,
                 lineage: pl.DataFrame | None,
                 year_rows: pl.DataFrame | None,
                 workdir: Path | None,
                 provider: "object | None",
                 run_factor: "Param | None" = None,
                 ) -> pl.DataFrame | None:
    """``[e, cohort, d, y_d, y_on, y_eol, a_cap, a_salv, settle, cash, sold, taken, saved]`` of the retirement options
    ``options`` (``[e, cohort, d]``) of the plants ``cohorts`` (``[e, cohort, y_on, y_eol, pay, n_pay, r_loan, a_cap,
    a_salv]``), the parts of :func:`built_retirement_costs`: ``settle[c,d]`` the cash of a MW retired in d as its
    contract ran (``set``; ``cash = pay·paid + D(d)·B`` its payments, ``sold = D(d)·S_d`` the salvage value received,
    ``settle = cash − sold``), ``taken[c,d] = S̄[c,d]`` the investment factors of the plant's periods in service
    (history included, on d's lineage) and ``saved[c,d]`` the fixed cost it saves from d on; ``None`` without an
    option."""
    from flextool.engine.input._axis_enums import cast_dim
    from flextool.engine.preprocess.params._cohorts import cohort_years
    from flextool.engine.preprocess.params._run_calendar import YEAR_TOL
    from flextool.engine.preprocess.params._walks import _apply_lineage_filter
    if options is None or options.height == 0:
        return None
    years = cohort_years(source, active_solve, workdir, provider=provider)
    off_inv = _solve_inflation_scalars(source)[1]
    factors, walk_periods, universe = _run_inv_factors(source, active_solve, period_in_use, period_universe,
                                                       period_with_history, years, provider=provider,
                                                       year_rows=year_rows, run_factor=run_factor)
    in_use = [str(d) for d in period_in_use]
    in_use_frame = pl.DataFrame({"d_all": in_use}, schema={"d_all": pl.Utf8})
    d_dtype = options.schema["d"]
    on_d = pl.col("d_all").pipe(cast_dim, _enums, "d")
    year_of = (pl.DataFrame({"d": list(years), "y_d": [float(y) for y in years.values()]},
                            schema={"d": pl.Utf8, "y_d": pl.Float64})
                 .with_columns(pl.col("d").pipe(cast_dim, _enums, "d")))
    prices = cohorts.select("e", "cohort", "y_on", "y_eol", "pay", "n_pay", "r_loan", "a_cap", "a_salv")
    anchors, year_of = align_join_dtypes(options.select("e", "cohort", "d"), year_of, ["d"])
    anchors = (anchors.join(prices, on=["e", "cohort"], how="left", maintain_order="left")
                      .join(year_of, on="d", how="left", maintain_order="left"))
    missing = anchors.filter(pl.col("y_d").is_null())
    if missing.height > 0:
        raise ValueError(f"Solve '{active_solve}': the retirement periods {sorted(set(missing['d'].to_list()))} have "
                         "no year of the run; the retirement of model-built capacity cannot be settled.")

    # The periods each cohort is in service in (history included), on the lineage of the retirement period.
    walk = (anchors.select("e", "cohort", "d", "y_d", "y_on", "y_eol")
                   .join(walk_periods.with_columns(on_d.alias("d_all")), how="cross")
                   .filter(within_life(pl.col("y"), pl.col("y_on"), pl.col("y_eol"))))
    if lineage is not None:
        walk = _apply_lineage_filter(walk.lazy(), lineage, in_use, d_dtype).collect()
    walk = walk.join(factors.with_columns(on_d.alias("d_all")), on="d_all", how="left")
    no_factor = walk.filter(pl.col("factor").is_null())
    if no_factor.height > 0:
        raise ValueError(f"Solve '{active_solve}': the periods {sorted(set(no_factor['d_all'].to_list()))} have no "
                         "investment discount factor; the retirement of model-built capacity cannot be settled.")
    keys = ["e", "cohort", "d"]
    taken = walk.group_by(keys, maintain_order=True).agg(pl.col("factor").sum().alias("taken"))

    # The fixed cost saved: the solve's periods from d on, while the cohort is in service.
    saved_rows = (walk.join(in_use_frame.with_columns(on_d.alias("d_all")), on="d_all", how="semi")
                      .filter(pl.col("y") >= pl.col("y_d") - YEAR_TOL))
    fc = (_ed_fixed_cost_raw_lf(source, saved_rows.lazy().select("e", pl.col("d_all").alias("d")).unique())
          .select("e", pl.col("d").alias("d_all"), "fc").collect())
    ops = (_factor_frame(source, active_solve, period_in_use, period_universe, provider, "ops_factor", year_rows)
           .with_columns(on_d.alias("d_all")).collect())
    saved_rows, fc = align_join_dtypes(saved_rows, fc, ["e", "d_all"])
    saved = (saved_rows.join(fc, on=["e", "d_all"], how="left")
                       .join(ops.rename({"factor": "ops"}), on="d_all", how="left")
                       .group_by(keys, maintain_order=True)
                       .agg((pl.col("fc").fill_null(0.0) * pl.col("ops").fill_null(0.0)).sum().alias("saved")))

    # The payments due before d (history included), the capital still owed at the start of d and the salvage.
    rows = _payment_rows(source, active_solve, period_in_use, universe, set(walk_periods["d_all"].to_list()),
                         workdir, provider=provider, year_rows=year_rows)
    end = pl.min_horizontal(pl.col("y_d"), pl.col("y_on") + pl.col("n_pay"))
    share = ((pl.min_horizontal(end, pl.col("base") + pl.col("width"))
              - pl.max_horizontal(pl.col("y_on"), pl.col("base"))) / pl.col("width")).clip(0.0, 1.0)
    paid = (anchors.select(*keys, "y_d", "y_on", "n_pay").join(rows, how="cross")
                   .filter(pl.col("width") > 0.0)
                   .group_by(keys, maintain_order=True).agg((share * pl.col("contrib")).sum().alias("paid")))
    disc = (_retirement_discount_lf(source, active_solve, universe, provider=provider, year_rows=year_rows)
            .with_columns(pl.col("d").pipe(cast_dim, _enums, "d")).collect())
    s_d = (ed_values_on(source, NET_SALVAGE_VALUE, anchors.lazy().select("e", "cohort", "d"), fill=0.0)
           .select(*keys, s_d=pl.col("value") * 1000.0).collect())
    left = pl.col("n_pay") - (pl.col("y_d") - pl.col("y_on"))
    r = pl.col("r_loan")
    remaining = pl.when(r.abs() < 1e-9).then(left).otherwise((1.0 - (1.0 + r).pow(-left)) / r)
    balance = (pl.when(left > YEAR_TOL)
                 .then(pl.col("pay") * (1.0 + r).pow(1.0 - off_inv) * remaining)
                 .otherwise(0.0))
    anchors, disc = align_join_dtypes(anchors, disc, ["d"])
    out = (anchors.join(paid, on=keys, how="left", maintain_order="left")
                  .join(taken, on=keys, how="left", maintain_order="left")
                  .join(saved, on=keys, how="left", maintain_order="left")
                  .join(s_d, on=keys, how="left", maintain_order="left")
                  .join(disc, on="d", how="left", maintain_order="left")
                  .with_columns(paid=pl.col("paid").fill_null(0.0), taken=pl.col("taken").fill_null(0.0),
                                saved=pl.col("saved").fill_null(0.0), s_d=pl.col("s_d").fill_null(0.0),
                                disc=pl.col("disc").fill_null(1.0))
                  .with_columns(settle=pl.col("pay") * pl.col("paid") + pl.col("disc") * (balance - pl.col("s_d")),
                                cash=pl.col("pay") * pl.col("paid") + pl.col("disc") * balance,
                                sold=pl.col("disc") * pl.col("s_d"))
                  .sort(keys))
    return out.select(*keys, "y_d", "y_on", "y_eol", "a_cap", "a_salv", "settle", "cash", "sold", "taken", "saved")


def chain_retirement_costs(source: "InputSource",
                           active_solve: str | None,
                           gens: pl.DataFrame,
                           options: pl.DataFrame,
                           period_in_use: list[str],
                           period_universe: list[str],
                           period_with_history: list[str],
                           *,
                           lineage: pl.DataFrame | None,
                           year_rows: pl.DataFrame | None,
                           workdir: Path | None,
                           provider: "object | None",
                           run_factor: "Param | None" = None,
                           per_plant: bool = False,
                           first_settles: bool = False,
                           ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """``(cost, lines)``: ``cost`` ``[e, root, d, value]`` (Utf8 keys, CUR/MW, MWh for storage) of retiring a MW of the ``reinvest_automatic``
    chain ``root`` in d (``options`` ``[e, root, d]``), the plants of the chain being ``gens``
    (:func:`._npv.chain_generations`), without the fixed cost it saves::

        value[c,d] = [m > 0 or (g = g1 and first_settles)]·set[g,d] − Σ_{g′ ∈ gens, y_eol[g′] > y[d]} a_b[g′]·S̄[g′,d]

    g the generation in service in d, m = y[d] − y_on[g]; ``set`` / ``S̄`` of :func:`_settlements` (the cash of the
    plant's contract as it ran to d, the investment factors of its periods in service, history included, on d's
    lineage) and a_b = a_cap − a_salv (:func:`._npv.generation_prices` of the plant as constructed in its online period
    on d's lineage, ``per_plant`` for the replacement of existing capacity).  A retired MW pays its current plant's
    contract (a replacement retired at its own start was never bought) and gives back the level price of its plant
    and of every replacement still to come.

    ``lines`` ``[e, root, d, line, d_report, value]`` (Utf8, :mod:`._cost_lines`) split ``cost`` as
    :func:`built_retirement_costs` does, per plant g′: ``retirement`` @ d = [settles]·cash − a_cap[g′]·S̄[g′,d] of
    the plant in service in d (y_on[g′] <= y[d]), ``replacement`` @ the online period of g′ on d's lineage
    (``d_price``) = −a_cap[g′]·S̄[g′,d] of a replacement still to come (the capital of a plant never built, taken
    back where the investment lines put it), ``salvage economic`` @ d = −[settles]·D(d)·S_d, ``salvage replaced`` @
    the end of life of g′ = a_salv[g′]·S̄[g′,d] (a plant of a chain is replaced at its end of life)."""
    from flextool.engine.preprocess.params import _cost_lines as L
    from flextool.engine.input._axis_enums import cast_dim
    from flextool.engine.preprocess.params._cohorts import salvage_periods
    from flextool.engine.preprocess.params._run_calendar import YEAR_TOL
    empty = pl.DataFrame(schema={"e": pl.Utf8, "root": pl.Utf8, "d": pl.Utf8, "value": pl.Float64})
    no_lines = stack_lines(pl.DataFrame(), ("e", "root", "d"), [])
    if gens.height == 0 or options is None or options.height == 0:
        return empty, no_lines
    years = _chain_years(source, active_solve, provider)
    last = salvage_periods(source, active_solve, years, provider, [None])[0]
    opts = options.select(pl.col("e").cast(pl.Utf8), pl.col("root").cast(pl.Utf8), pl.col("d").cast(pl.Utf8))
    yd = pl.DataFrame({"d": list(years), "y_d": [float(v) for v in years.values()]},
                      schema={"d": pl.Utf8, "y_d": pl.Float64})
    rows = (opts.join(yd, on="d", how="inner")
                .join(gens.select("e", "root", "cohort", "y_on", "y_eol", "d_on", "d_eol"), on=["e", "root"],
                      how="inner")
                .filter(within_life(pl.col("y_d"), None, pl.col("y_eol"))))
    if rows.height == 0:
        return empty, no_lines
    rows = _labels_on_lineage(rows, years, lineage, last)
    prices = generation_prices(source, rows.select("e", "cohort", d=pl.col("d_price"), d_salv=pl.col("d_salv"))
                               .unique(maintain_order=True), per_plant=per_plant)
    rows = (rows.join(prices.rename({"d": "d_price"}), on=["e", "cohort", "d_price", "d_salv"], how="left")
                .with_columns(plant=pl.concat_str("cohort", "d_price", "d_salv", separator="|")))
    plants = rows.select("e", pl.col("plant").alias("cohort"), "y_on", "y_eol", "pay", "n_pay", "r_loan", "a_cap",
                         "a_salv").unique(subset=["e", "cohort"], maintain_order=True)
    plant_options = (rows.select("e", pl.col("plant").alias("cohort"), "d").unique(maintain_order=True)
                         .with_columns(pl.col("e").pipe(cast_dim, _enums, "e"), pl.col("d").pipe(cast_dim, _enums, "d")))
    parts = _settlements(source, active_solve, plants.with_columns(pl.col("e").pipe(cast_dim, _enums, "e")),
                         plant_options, period_in_use, period_universe, period_with_history, lineage=lineage,
                         year_rows=year_rows, workdir=workdir, provider=provider, run_factor=run_factor)
    if parts is None:
        return empty, no_lines
    parts = parts.select(pl.col("e").cast(pl.Utf8), pl.col("cohort").alias("plant"), pl.col("d").cast(pl.Utf8),
                         "settle", "cash", "sold", "taken")
    current = within_life(pl.col("y_d"), pl.col("y_on"), None)
    starts = pl.col("cohort").str.ends_with(":1") & pl.lit(first_settles)
    settles = current & ((pl.col("y_d") - pl.col("y_on") > YEAR_TOL) | starts)
    joined = rows.join(parts, on=["e", "plant", "d"], how="left").sort("e", "root", "d", "y_on")
    cost = (joined.group_by("e", "root", "d", maintain_order=True)
                  .agg((pl.when(settles).then(pl.col("settle").fill_null(0.0)).otherwise(0.0)
                        - (pl.col("a_cap") - pl.col("a_salv")) * pl.col("taken").fill_null(0.0)).sum().alias("value"))
                  .sort("e", "root", "d"))
    timed = event_period(joined.with_columns(_settles=settles), "y_eol",
                         report_years(source, active_solve, period_in_use, workdir, provider), "d_eol")
    taken = pl.col("taken").fill_null(0.0)
    lines = stack_lines(timed.with_columns(_current=current), ("e", "root", "d"), [
        (L.RETIREMENT, pl.when(pl.col("_current")).then(
            pl.when(pl.col("_settles")).then(pl.col("cash").fill_null(0.0)).otherwise(0.0)
            - pl.col("a_cap") * taken).otherwise(0.0), pl.col("d")),
        (L.REPLACEMENT, pl.when(pl.col("_current")).then(0.0).otherwise(-pl.col("a_cap") * taken),
         pl.col("d_price")),
        (L.SALVAGE_ECONOMIC, -pl.when(pl.col("_settles")).then(pl.col("sold").fill_null(0.0)).otherwise(0.0),
         pl.col("d")),
        (L.SALVAGE_REPLACED, pl.col("a_salv") * taken, pl.col("d_eol"))])
    return cost, lines


def apply_built_retirement_costs(flex_data: object,
                                 source: "InputSource",
                                 active_solve: str,
                                 period_in_use: list[str],
                                 period_universe: list[str],
                                 period_with_history: list[str],
                                 *,
                                 lineage: pl.DataFrame | None,
                                 year_rows: pl.DataFrame | None,
                                 workdir: Path | None,
                                 provider: "object | None" = None,
                                 ) -> None:
    """Set the retirement costs of model-built capacity (:func:`built_retirement_costs`): ``cd_divest_built_cost``
    (the settlement less the level price taken back), ``cd_divest_built_fixed`` (the fixed cost saved) and
    ``cdl_divest_built_lines`` (the cost lines of the reported costs, per MW retired), after the
    cohort prices of this solve's investments (:func:`._npv.apply_invest_lumps`) are on ``cohort_divest_built``.  A
    retirable cohort without its prices is an error (:func:`._retire_built.require_cohort_prices`), never a 0.  Warns
    where building a cohort and retiring it earns money (:func:`._retire_built.warn_build_to_retire`).  The only
    derivation of these terms: a Spine solve (:func:`._lifetime_costs.apply_npv`) and a synthetic sub-solve
    (:func:`._lifetime_costs.apply_run_fields_of_synthetic_solve`) call it alike."""
    from flextool.engine.preprocess.params._retire_built import require_cohort_prices, warn_build_to_retire
    options = getattr(flex_data, "cd_divest_built", None)
    cohorts = getattr(flex_data, "cohort_divest_built", None)
    if options is None or options.height == 0 or cohorts is None or cohorts.height == 0:
        flex_data.cd_divest_built_cost = flex_data.cd_divest_built_fixed = flex_data.cdl_divest_built_lines = None
        return
    cohorts = _with_cohort_prices(cohorts, getattr(flex_data, "cohort", None))
    flex_data.cohort_divest_built = cohorts
    gens = _built_chain_generations(flex_data, provider)
    roots = cohorts.select(pl.col("e").cast(pl.Utf8), "cohort").join(
        gens.select("e", pl.col("root").alias("cohort")).unique(), on=["e", "cohort"], how="semi")
    left = cohorts.with_columns(_e=pl.col("e").cast(pl.Utf8))
    require_cohort_prices(left.join(roots.rename({"e": "_e"}), on=["_e", "cohort"], how="anti").drop("_e"))
    cost, fixed, lines = built_retirement_costs(
        source, active_solve, cohorts, options, period_in_use, period_universe, period_with_history,
        lineage=lineage, year_rows=year_rows, workdir=workdir, provider=provider,
        run_factor=getattr(flex_data, "p_inflation_inv_run", None))
    if roots.height > 0:
        # A reinvest_automatic cohort: its chain's settlement (the plant in service, and every replacement to come).
        chained = options.select(pl.col("e").cast(pl.Utf8), pl.col("cohort").alias("root"), pl.col("d").cast(pl.Utf8))
        chained = chained.join(roots.rename({"cohort": "root"}), on=["e", "root"], how="semi")
        chain, chain_lines = chain_retirement_costs(
            source, active_solve, gens, chained, period_in_use, period_universe, period_with_history,
            lineage=lineage, year_rows=year_rows, workdir=workdir, provider=provider,
            run_factor=getattr(flex_data, "p_inflation_inv_run", None), first_settles=True)
        chained_keys = chain.select("e", pl.col("root").alias("cohort"), "d")
        lines = pl.concat([lines.join(chained_keys, on=["e", "cohort", "d"], how="anti"),
                           chain_lines.rename({"root": "cohort"})], how="vertical_relaxed")
        chain, cost = align_join_dtypes(
            chain.lazy().select(alias_to_axis("e", "e"), pl.col("root").alias("cohort"), alias_to_axis("d", "d"),
                                _chain=pl.col("value")), cost.lazy(), ["e", "d"])
        cost = (cost.join(chain, on=["e", "cohort", "d"], how="left", maintain_order="left")
                    .with_columns(value=pl.coalesce("_chain", "value")).drop("_chain").collect())
        _notice_automatic_replacement_cost(roots.lazy().select("e").unique().with_columns(changed=pl.lit(True)))
    flex_data.cd_divest_built_cost = Param(("e", "cohort", "d"), cost)
    flex_data.cd_divest_built_fixed = Param(("e", "cohort", "d"), fixed)
    flex_data.cdl_divest_built_lines = lines.sort("e", "cohort", "d", "line", "d_report")
    warn_build_to_retire(_build_and_retire_totals(flex_data, cohorts, cost, fixed))


def _built_chain_generations(flex_data: object, provider: "object | None") -> pl.DataFrame:
    """:func:`._npv.chain_generations` of the solve's investment decisions (``flex_data.cohort``) and of the chains
    earlier solves of the run realized (handoff ``realized_cohort``: their generations as the deciding solve placed
    them)."""
    from flextool.engine.preprocess import _provider_keys as K
    from flextool.engine.preprocess._provider_translators import read_handoff_frame
    own = chain_generations(getattr(flex_data, "cohort", None))
    carried = read_handoff_frame(provider, K.HANDOFF_REALIZED_COHORT) if provider is not None else None
    if carried is None or carried.height == 0 or "root" not in carried.columns:
        return own
    renamed = carried.rename({"entity": "e", "period_decided": "d_build", "period_online": "d_on",
                              "period_eol": "d_eol"})
    carried_gens = chain_generations(renamed.with_columns(pl.col(c).cast(pl.Float64) for c in ("y_on", "y_eol")))
    return pl.concat([own, carried_gens.join(own.select("e", "cohort"), on=["e", "cohort"], how="anti")])


def _with_cohort_prices(cohorts: pl.DataFrame, cohort: pl.DataFrame | None) -> pl.DataFrame:
    """``cohorts`` (``cohort_divest_built``) with the prices of each cohort decided in this solve from the cohort
    table ``cohort`` (set by :func:`._npv._set_cohort_prices`); a carried cohort keeps the prices it was carried
    with."""
    from flextool.engine.preprocess.params._cohorts import KIND_EXTEND, KIND_INVEST
    if cohort is None or cohort.height == 0:
        return cohorts
    own = (cohort.filter(pl.col("kind").is_in([KIND_INVEST, KIND_EXTEND]))
                 .select("e", "cohort", *(pl.col(c).alias(f"_{c}") for c in PRICE_COLUMNS)))
    left, own = align_join_dtypes(cohorts, own, ["e"])
    here = pl.col("d_build").is_not_null()
    return (left.join(own, on=["e", "cohort"], how="left", maintain_order="left")
                .with_columns(pl.when(here).then(pl.col(f"_{c}")).otherwise(pl.col(c)).alias(c)
                              for c in PRICE_COLUMNS)
                .select(cohorts.columns))


def _build_and_retire_totals(flex_data: object, cohorts: pl.DataFrame, cost: pl.DataFrame,
                             fixed: pl.DataFrame) -> pl.DataFrame:
    """``[e, cohort, total]``: for each cohort decided in this solve, the objective coefficient of a MW built (its
    price and lifetime fixed cost, ``ed_entity_annual_discounted`` + ``ed_lifetime_fixed_cost`` of its decision; of
    a life extension ``cd_extend_cost`` + ``cd_extend_fixed`` of its option) plus the smallest coefficient of
    retiring it (``cost − fixed``), CUR/MW."""
    from flextool.engine.preprocess.params._cohorts import KIND_EXTEND
    retire = (cost.join(fixed.rename({"value": "fixed"}), on=["e", "cohort", "d"], how="inner")
                  .group_by("e", "cohort").agg((pl.col("value") - pl.col("fixed")).min().alias("retire")))
    here = cohorts.filter(pl.col("d_build").is_not_null())
    built = here.filter(pl.col("kind") != KIND_EXTEND).select("e", "cohort", "d_build")
    extended = here.filter(pl.col("kind") == KIND_EXTEND).select("e", "cohort", "d_build", "parent")
    parts = []
    for frame, names, keys in ((built, ("ed_entity_annual_discounted", "ed_lifetime_fixed_cost"), ["d_build"]),
                               (extended, ("cd_extend_cost", "cd_extend_fixed"), ["parent", "d_build"])):
        frame = frame.with_columns(total=pl.lit(0.0))
        for name in names:
            param = getattr(flex_data, name, None)
            if param is None or frame.height == 0:
                continue
            values = param.frame.rename({"d": "d_build", "value": "_v"})
            if "cohort" in values.columns:
                values = values.rename({"cohort": "parent"})
            frame, values = align_join_dtypes(frame, values, ["e", "d_build"])
            frame = (frame.join(values, on=["e", *keys], how="left")
                          .with_columns(total=pl.col("total") + pl.col("_v").fill_null(0.0)).drop("_v"))
        parts.append(frame.select("e", "cohort", "total"))
    built = pl.concat(parts)
    built, retire = align_join_dtypes(built, retire, ["e"])
    return (built.join(retire, on=["e", "cohort"], how="inner")
                 .select("e", "cohort", total=pl.col("total") + pl.col("retire")))
