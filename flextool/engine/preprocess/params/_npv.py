"""Cluster A — annual integration / NPV (Δ.5).

Lazy-polars port of flextool's NPV / lifetime-fixed-cost / inflation
cascade.  Replaces the eager-Python implementation that previously lived
in :mod:`._derived` (``ed_entity_annual_family_from_source`` plus
its supporting per-(entity, period) lookup).

The cluster is split by job.  This module holds the investment price (``ed_entity_annual_discounted``, the cohort
prices, :func:`apply_invest_lumps`); ``_lifetime_inputs`` the per-entity inputs, ``_discount`` the discount factors
(``p_inflation_op``), ``_fixed_costs`` the fixed costs (``p_ed_fixed_cost``, ``ed_lifetime_fixed_cost*``),
``_retire_settlement`` / ``_existing_salvage`` the retirement terms, ``_extend`` the life-extension prices,
``_run_costs`` the run-level fields of the reported costs and ``_lifetime_costs`` the apply boundary that sets them
all (:func:`._lifetime_costs.apply_npv`).

The cluster covers these FlexData fields:

* ``p_inflation_op``                           — per-period operations
  inflation factor.  Replaces the workdir read of
  ``solve_data/p_inflation_factor_operations_yearly.csv``.
* ``p_ed_fixed_cost``                          — per-entity per-period
  raw fixed cost (× 1000 for nodes/processes).  Replaces the workdir
  read of ``solve_data/ed_fixed_cost.csv``.
* ``ed_entity_annual_discounted``              — invest-side NPV.
  Replaces ``solve_data/ed_entity_annual_discounted.csv``.
* ``ed_divest_salvage`` / ``ed_existing_salvage_credit`` — the salvage
  settlement of a retirement of existing capacity and the level salvage
  credit of existing capacity (:func:`._existing_salvage.existing_salvage`).
* ``cd_divest_built_cost`` / ``cd_divest_built_fixed`` — the settlement of a
  retired MW of a model-built cohort less the level price given back, and the
  fixed cost it saves (:func:`._retire_settlement.built_retirement_costs`).
* ``ed_lifetime_fixed_cost``                   — invest-side lifetime
  cumulative fixed cost.  Replaces ``solve_data/ed_lifetime_fixed_cost.csv``.
* ``ed_lifetime_fixed_cost_divest``            — divest-side lifetime
  cumulative fixed cost.  Replaces ``solve_data/ed_lifetime_fixed_cost_divest.csv``.

and, for the reported costs only (no model term reads them), the run-level
investment lump ``ed_entity_annual_discounted_run`` discounted from the start of the run over the periods of the
whole run and the cost lines ``edl_*`` / ``cdl_*`` (:mod:`._cost_lines`: each lump and coefficient split by the
event it is made of, in the period it happens; ``_run_costs`` "Run-level fields of the reported costs").  The reported
fixed costs of new and retired capacity come from the capacity of each cohort in each period
(``process_outputs.calc_costs.fixed_costs_in_service``), not from lumps.

Architecture
------------

Per the user-locked decisions for derived clusters (audit handoff at
Δ.5):

1. **Lazy polars throughout.**  Helpers return ``pl.LazyFrame`` chains;
   the public entry points materialise via ``Param(...)`` at the
   ``apply_derived_f`` boundary.
2. **Convention over typed BuildPipeline.**  The dependency graph is
   shallow: ``period_walk_iterator`` → 4 NPV helpers; ``p_ed_fixed_cost``
   feeds the 2 lifetime-fixed-cost helpers.  Plain function calls.
3. **4 NPV variants split into 4 helpers** sharing
   :func:`._lifetime_inputs.period_walk_iterator` for the per-period accumulation walk.

Algorithm:

  For each (entity e, period d) in the relevant invest/divest/all-entity
  domain, the NPV value is::

      annuity[e, d] × Σ_{d_all ∈ period_in_use, window(d, d_all, e)}
                                       inflation_factor[d_all]

  where:

  * ``annuity`` = sum-over-allowed-methods of
    ``invest_value × 1000 × annuity_factor(r, n, off)`` (:mod:`.._annuity`:
    ``r / (1 - (1+r)^-n) · (1+r)^(off − 1)``, ``1/n`` at ``r = 0``; ``off`` the
    ``discount_offset_investment``), ``n`` the payback time
    (``payback_time``, else the entity's ``lifetime_technical``).  A missing
    ``r`` takes the schema default (0.05); a missing payback time where the
    cost is not 0 is an error.  The investment of ``reinvest_choice`` /
    ``no_reinvestment`` takes the level yearly price instead,
    ``a_cap − a_salv``: ``a_cap = annuity × G(n) / G(L)`` (:func:`.._annuity.level_factor`,
    ``L`` the ``lifetime_technical``) less the level salvage credit ``a_salv``
    (:func:`_level_prices_lf`).
  * ``window(d, d_all, e)``:
      - investment price and lifetime fixed cost, ``reinvest_choice`` / ``no_reinvestment``: ``pdy[d_all] ∈
        [pdy[d], y_eol)``, ``y_eol = round_end_of_life(pdy[d] + lifetime_technical[e, d], end_of_life_method[e])``
        (the periods the cohort is in service; :func:`._lifetime_inputs._window_life_to_end_of_life`).
      - ``reinvest_automatic``: ``pdy[d_all] ≥ pdy[d]``.
      - the fixed cost a retirement saves: the retirement's periods until the end of life of the existing
        capacity (:func:`._fixed_costs._retire_window_lf`).
  * The salvage terms of a retirement of existing capacity are no annuity window:
    :func:`._existing_salvage.existing_salvage`.
  * ``inflation_factor[d_all]``:
      - investment side → ``p_inflation_factor_investment_yearly[d_all]``.
      - operations side (``ed_lifetime_fixed_cost`` only) →
        ``p_inflation_factor_operations_yearly[d_all]``.
      - ``ed_lifetime_fixed_cost_divest`` is asymmetric and uses the
        investment factor (mirrors flextool.mod L1651).

The annuity formula and the discount-window walk are 1:1 ports of the
flextool source; deviations are documented inline.

Cross-references
----------------

* :mod:`._derived._inflation_yearly_from_source` — the yearly
  inflation cascade computation used to be eager-Python; we keep that
  helper's algorithm but expose a lazy-frame variant
  (:func:`._discount._inflation_factors_lf`) for use here.
* :func:`._derived._p_years_d_lf` — already lazy; reused as-is.
* :func:`._derived._period_in_use_set` and
  :func:`._derived._solve_periods` — list builders used to
  bound the period domain.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

from flextool.engine.input._axis_enums import (
    alias_to_axis,
    align_join_dtypes,
    schema_dtype,
)
from flextool.engine.preprocess._annuity import (
    annuity_factor,
    default_discount_rate,
    level_factor,
    require_lifetime,
)
from flextool.engine.preprocess.params._cohorts import PRICE_COLUMNS, within_life
from flextool.engine.input._lifetime_roles import (
    LIFETIME_TECHNICAL,
    NET_SALVAGE_VALUE,
    NO_REINVESTMENT,
    PAYBACK_TIME,
)
from flextool.engine.preprocess.params._period_values import ed_values_on, ed_values_online
from flextool.engine.preprocess.params._existing import INVEST_NOT_ALLOWED
from flextool.engine.preprocess.params._discount import _inflation_factors_lf, _solve_inflation_scalars
from flextool.engine.preprocess.params._lifetime_inputs import (
    _commission_walk_lf,
    _ed_lifetime_on_lf,
    _entity_class_lf,
    _entity_class_lookup_lf,
    _entity_invest_set_lf,
    _entity_method_lf,
    _lifetime_method_with_default_lf,
    _register_invest_lumps,
    _utf8_ed,
    _window_life_to_end_of_life,
    period_walk_iterator,
)
from flextool.engine.preprocess.params._fixed_costs import ed_lifetime_fixed_cost_from_source

# Substrate handle for the cascade-wide axis enum vocabulary.
# Bare ``None`` here; ``cast_dim`` / ``schema_dtype`` in
# ``_axis_enums`` fall back to ``_LIVE_AXIS_ENUMS_CTX`` (the live
# ContextVar) when this is ``None``, so substrate sites pick up
# activation set by ``load_flextool`` automatically.
_enums: "dict | None" = None

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource


# ---------------------------------------------------------------------------
# Per-method annuity (lazy)
# ---------------------------------------------------------------------------


def _notice_discount_rate_zero() -> None:
    """The ``discount_rate_zero_is_zero`` notice for the entities its marker names (an upgraded database whose
    explicit discount_rate 0 was read as 5 % before schema v79)."""
    from flextool.engine import _behaviour_changes as bc

    if bc.has_marker("discount_rate_zero_is_zero"):
        names = bc.marker_json("discount_rate_zero_is_zero").get("entities") or []
        if names:
            bc.notice("discount_rate_zero_is_zero", "entities", sorted(str(n) for n in names))


ANNUITY_OVER_PAYBACK_TIME = "annuity_over_payback_time"
ANNUITY_TIMING = "annuity_timing"
NET_SALVAGE_ONCE = "net_salvage_once"


def _notice_net_salvage_once(entities) -> None:
    """The ``net_salvage_once`` notice for the entities its marker names (an upgraded database where a retirement
    took the salvage annuity and a new plant's yearly cost had no salvage) among ``entities`` (those whose salvage
    terms are not 0 in this solve)."""
    from flextool.engine import _behaviour_changes as bc

    if not bc.has_marker(NET_SALVAGE_ONCE):
        return
    names = {str(n) for n in bc.marker_json(NET_SALVAGE_ONCE).get("entities") or []}
    hit = sorted(names & {str(e) for e in entities})
    if hit:
        bc.notice(NET_SALVAGE_ONCE, "entities", hit)


def _notice_annuity_timing(rows: pl.LazyFrame, offset: float) -> None:
    """The ``annuity_timing`` notice for the entities its marker names (an upgraded database whose annuity was the
    year-end annuity at every ``discount_offset_investment``) whose annuity now differs: ``rows`` ``[e, cost,
    disc]`` (the cost and discount_rate of each annuity of the solve); it differs where the offset is not 1 and
    both are non-zero.  Built only while the marker names entities."""
    from flextool.engine import _behaviour_changes as bc

    if offset == 1.0 or not bc.has_marker(ANNUITY_TIMING):
        return
    names = {str(n) for n in bc.marker_json(ANNUITY_TIMING).get("entities") or []}
    if not names:
        return
    changed = (rows.filter((pl.col("cost") != 0) & (pl.col("disc") != 0))
                   .select("e").unique().collect()["e"].to_list())
    hit = sorted(names & {str(e) for e in changed})
    if hit:
        bc.notice(ANNUITY_TIMING, "entities", hit)


def _notice_annuity_over_payback_time(paid: pl.LazyFrame, before) -> None:
    """The ``annuity_over_payback_time`` notice for the entities its marker names (an upgraded database whose
    reinvest_choice / no_reinvestment annuity was paid for every period that started before the end of the
    lifetime) whose investment cost now differs: ``paid`` (``[e, d, value]``, the investment lumps of the entities
    with reinvest_choice / no_reinvestment in this solve) against ``before()`` (the same, the annuity over whole
    periods over lifetime_technical).  Built only while the marker names entities."""
    from flextool.engine import _behaviour_changes as bc

    if not bc.has_marker(ANNUITY_OVER_PAYBACK_TIME):
        return
    names = {str(n) for n in bc.marker_json(ANNUITY_OVER_PAYBACK_TIME).get("entities") or []}
    if not names:
        return
    changed = (paid.join(before().rename({"value": "_before"}), on=["e", "d"], how="left")
                   .filter((pl.col("value") - pl.col("_before").fill_null(0.0)).abs()
                           > 1e-12 * pl.max_horizontal(pl.lit(1.0), pl.col("_before").abs()))
                   .select("e").unique().collect()["e"].to_list())
    hit = sorted(names & {str(e) for e in changed})
    if hit:
        bc.notice(ANNUITY_OVER_PAYBACK_TIME, "entities", hit)


def _per_method_annuity_lf(
        source: "InputSource",
        ed_lf: pl.LazyFrame,
        cost_param_name: str,
        disallowed_methods: frozenset[str],
        commission: pl.DataFrame | None = None,
        *,
        level: bool = False,
        per_plant: bool = False,
        ) -> pl.LazyFrame:
    """Lazy ``(e, d, ann)`` frame computing the per-method annuity sum; with ``level`` ``(e, d, ann, a_cap, disc,
    life, technical, method_count)`` (the interest rate, the payback time and the lifetime the price is of, and the
    number of invest methods the annuity is summed over).  ``per_plant``: the annuity of one plant (a method count
    of 1 for every unit, connection and node, also one without an invest method).

    ``commission`` (``[e, d, d_c]``, the decisions with a construction lead time): the cost, the interest rate and
    the payback time (``lifetime_technical`` for an entity without ``payback_time``) of such a decision are those
    of the period ``d_c`` it comes online (:func:`._period_values.ed_values_online`).

    ``level``: also the level yearly price of the plant over its real lifetime (``invest_cost_p`` OBJ header),
    ``lifetime_technical`` ``L`` read at the same period::

        a_cap[e, d] = ann[e, d]                                     |n − L| ≤ YEAR_TOL, or no L
                    = ann[e, d] · G(n) / G(L)  (:func:`.._annuity.level_factor`, i = model.social_discount_rate)

    Algorithm (port of
    ``entity_annual_calc_params.py:177-217``):

      For each (e, d) ∈ ed_lf:
        ann[e, d] = Σ_{m ∈ invest_methods[e] \\ disallowed}
                       cost[e, d] · 1000 · annuity_factor(discount_rate[e, d], payback_time[e, d], off_inv)
        (when e ∈ unit ∪ connection ∪ node)

    The inner computation does NOT depend on ``m`` — every allowed
    method contributes the same annuity value.  So we collapse:
    ``ann[e, d] = method_count[e] × annuity[e, d]`` where
    ``method_count[e] = |{m : (e, m) ∈ entity__invest_method,
                                m ∉ disallowed}|``.
    """
    # Per-entity allowed-method count.
    methods_lf = _entity_method_lf(source, "invest_method")
    method_count_lf = (methods_lf
                          .filter(~pl.col("method").is_in(list(disallowed_methods)))
                          .group_by("e")
                          .agg(pl.col("method").count().alias("method_count")))

    # Class membership gate: e ∈ unit ∪ connection ∪ node (mirrors the
    # if/elif blocks in entity_annual_calc_params._per_method_annuity_*).
    cls_lf = _entity_class_lookup_lf(_entity_class_lf(source))
    in_class = (pl.col("is_process").fill_null(False)
                | pl.col("is_node").fill_null(False))

    # Per-(e, d) cost / discount_rate / lifetime, read through the shape resolver.
    ed_keys = ed_lf.select("e", "d")

    def at_start(parameter: str, fill: float, col: str) -> pl.LazyFrame:
        return (ed_values_online(source, parameter, ed_keys.unique(maintain_order=True), fill, commission)
                  .rename({"value": col}))

    ed_with_class = ed_keys.join(cls_lf, on="e", how="left")
    ed_with_cost = ed_with_class.join(at_start(cost_param_name, 0.0, "cost"), on=["e", "d"], how="left",
                                      maintain_order="left")
    # A missing discount_rate is the schema default (0.05); an explicit 0 stays 0 (straight-line annuity).
    ed_with_disc = ed_with_cost.join(at_start("discount_rate", default_discount_rate(source), "disc"),
                                     on=["e", "d"], how="left", maintain_order="left")
    # No lifetime default: a null (or ≤ 0) lifetime where the cost is not 0 is an input error.  Checked here,
    # not only in the validator: a period map can lack a period (the one the payback time is read at).
    ed_with_life = (ed_values_online(source, PAYBACK_TIME, ed_with_disc, None, commission)
                       .rename({"value": "life"}))
    bad = (ed_with_life
              .join(method_count_lf, on="e", how="inner")
              .filter(in_class & (pl.col("cost") != 0)
                      & (pl.col("life").is_null() | (pl.col("life") <= 0)))
              .select(pl.col("e").cast(pl.Utf8), pl.col("d").cast(pl.Utf8))
              .head(5)
              .collect())
    require_lifetime(bad.iter_rows(), cost_param_name, PAYBACK_TIME)
    _notice_discount_rate_zero()

    # Annuity: cost [CUR/kW] × 1000 × annuity_factor(r, n, discount_offset_investment).
    offset = _solve_inflation_scalars(source)[1]
    _notice_annuity_timing(ed_with_life.join(method_count_lf, on="e", how="inner").filter(in_class)
                           .select("e", "cost", "disc"), offset)
    annuity_expr = (pl.col("cost") * 1000.0
                    * annuity_factor(pl.col("disc"), pl.col("life"), offset))
    out = (ed_with_life
              .join(method_count_lf, on="e", how="left")
              .with_columns(
                  method_count=(pl.lit(1, dtype=pl.UInt32) if per_plant
                                else pl.col("method_count").fill_null(0)),
                  ann_per_method=pl.when(pl.col("cost") == 0).then(0.0).otherwise(annuity_expr),
              )
              .with_columns(
                  ann=pl.when(in_class)
                          .then(pl.col("ann_per_method")
                                  * pl.col("method_count").cast(pl.Float64))
                          .otherwise(0.0),
              ))
    if not level:
        return out.select("e", "d", "ann")
    from flextool.engine.preprocess.params._run_calendar import YEAR_TOL
    rate = _solve_inflation_scalars(source)[0]
    technical = (ed_values_online(source, LIFETIME_TECHNICAL, ed_keys.unique(maintain_order=True), None, commission)
                   .rename({"value": "technical"}))
    out, technical = align_join_dtypes(out, technical, ["e", "d"])
    out = out.join(technical, on=["e", "d"], how="left", maintain_order="left").collect()
    # The level price equals the annuity where the payback time is the lifetime (byte parity with the annuity).
    # The factor of each (payback time, lifetime) pair is the float :func:`level_factor`, so equal inputs give an
    # equal price whatever the frame's layout.
    plain = (pl.col("technical").is_null() | (pl.col("technical") <= 0) | pl.col("life").is_null()
             | ((pl.col("life") - pl.col("technical")).abs() <= YEAR_TOL) | (pl.col("ann") == 0.0))
    pairs = out.filter(~plain).select("life", "technical").unique(maintain_order=True)
    factors = pl.DataFrame({"life": pairs["life"], "technical": pairs["technical"],
                            "_level": [level_factor(n, life, rate) for n, life in pairs.iter_rows()]},
                           schema={"life": pl.Float64, "technical": pl.Float64, "_level": pl.Float64})
    return (out.lazy()
               .join(factors.lazy(), on=["life", "technical"], how="left", maintain_order="left")
               .with_columns(a_cap=pl.when(plain).then(pl.col("ann")).otherwise(pl.col("ann") * pl.col("_level")))
               .select("e", "d", "ann", "a_cap", "disc", "life", "technical", "method_count"))


# ---------------------------------------------------------------------------
# 4 NPV variant helpers (each producing one lazy frame).
# ---------------------------------------------------------------------------


def _level_prices_lf(source: "InputSource",
                     active_solve: str | None,
                     ed_anchor_lf: pl.LazyFrame,
                     commission: pl.DataFrame | None,
                     *,
                     provider: "object | None") -> pl.LazyFrame:
    """Lazy ``[e, d, ann, a_cap, a_salv, disc, life, nsv_eol]`` of each investment decision ``(e, d)`` of
    ``ed_anchor_lf``: the annuity ``ann`` = A (``pay``), the level yearly price ``a_cap`` and the level salvage credit
    of the plant as constructed (its period maps read at the period it comes online, :func:`_per_method_annuity_lf`
    with ``level``)::

        a_salv = salvage_credit(S, r, L, off_inv) = S·(1+r)^−L·ann(r, L)·(1+r)^(off_inv − 1)
        S      = 1000·net_salvage_value[e, d_eol]   (d_eol the period its rounded end of life starts; the last run
                 period after the last period start, :func:`._cohorts.salvage_periods`)

    ``disc`` the discount_rate r, ``life`` the payback time n and ``nsv_eol`` S.  The salvage at the end of the real
    lifetime ``L`` = lifetime_technical is levelled over the ``L`` service years at ``r``: the credit the yearly
    price of a ``reinvest_choice`` / ``no_reinvestment`` plant is lowered by (``invest_cost_p`` OBJ header).  S and
    a_salv are 0.0 without a net_salvage_value (and for an entity without a lifetime_technical)."""
    from flextool.engine.preprocess._annuity import salvage_credit
    from flextool.engine.preprocess.params._cohorts import cohort_years, salvage_periods
    keys = ed_anchor_lf.select("e", "d").unique(maintain_order=True)
    prices = _per_method_annuity_lf(source, keys, "invest_cost", INVEST_NOT_ALLOWED, commission,
                                    level=True).collect()
    eol = _window_life_to_end_of_life(source, active_solve,
                                      _ed_lifetime_on_lf(source, keys, LIFETIME_TECHNICAL, commission), commission,
                                      provider=provider, keep_eol=True).collect()
    years = cohort_years(source, active_solve, None, provider=provider)
    eol = eol.with_columns(d_salv=pl.Series("d_salv", salvage_periods(source, active_solve, years, provider,
                                                                      eol["y_eol"].to_list()), dtype=pl.Utf8))
    salvage = (ed_values_on(source, NET_SALVAGE_VALUE,
                            eol.lazy().select(pl.col("e").cast(pl.Utf8), pl.col("d_salv").alias("d"),
                                              _d=pl.col("d").cast(pl.Utf8)),
                            fill=0.0)
               .select(pl.col("e").cast(pl.Utf8), pl.col("_d").alias("d"), nsv_eol=pl.col("value") * 1000.0)
               .collect())
    prices, salvage = align_join_dtypes(prices, salvage, ["e", "d"])
    offset = _solve_inflation_scalars(source)[1]
    return (prices.lazy()
              .join(salvage.lazy(), on=["e", "d"], how="left", maintain_order="left")
              .with_columns(nsv_eol=pl.col("nsv_eol").fill_null(0.0))
              .with_columns(a_salv=salvage_credit(pl.col("nsv_eol"), pl.col("disc"), pl.col("technical"), offset)
                            * pl.col("method_count").cast(pl.Float64))
              .select("e", "d", "ann", "a_cap", "a_salv", "disc", "life", "nsv_eol"))


# ---------------------------------------------------------------------------
# The replacement generations of a reinvest_automatic chain
# ---------------------------------------------------------------------------

#: The columns of a generation frame (:func:`chain_generations`).
_GENERATION_COLUMNS = ("e", "d_build", "root", "cohort", "d_on", "d_eol", "y_on", "y_eol")


def chain_generations(cohort: pl.DataFrame | None, kind: str = "invest") -> pl.DataFrame:
    """``[e, d_build, root, cohort, d_on, d_eol, y_on, y_eol]`` (Utf8 labels) of the generations of the ``reinvest_automatic``
    chains of ``cohort`` whose root is of ``kind`` (``invest``: one per investment decision, ``d_build`` its decision
    period; ``existing``: the existing chains, ``d_build`` null); empty without one."""
    from flextool.engine.preprocess.params._cohorts import KIND_GENERATION
    schema = {c: (pl.Float64 if c.startswith("y_") else pl.Utf8) for c in _GENERATION_COLUMNS}
    if cohort is None or cohort.height == 0 or "root" not in cohort.columns:
        return pl.DataFrame(schema=schema)
    return (cohort.filter((pl.col("kind") == KIND_GENERATION) & pl.col("root").str.starts_with(f"{kind}:")
                          & pl.col("y_on").is_not_null())
                  .select(*(pl.col(c).cast(t) for c, t in schema.items())))


def _chain_years(source: "InputSource", active_solve: str | None, provider: "object | None") -> dict[str, float]:
    """``{period: run year}`` of the solve's periods (:func:`._cohorts.cohort_years`) and of the branch periods of a
    stochastic fan (each its anchor's year: a generation's years are lineage-free)."""
    from flextool.engine.preprocess.params._branch import active_branch_anchor_pairs
    from flextool.engine.preprocess.params._cohorts import cohort_years
    years = cohort_years(source, active_solve, None, provider=provider)
    pairs = active_branch_anchor_pairs()
    if pairs is not None:
        for anchor, br in pairs.select(pl.col("anchor").cast(pl.Utf8), pl.col("br").cast(pl.Utf8)).iter_rows():
            if anchor in years and br not in years:
                years[br] = years[anchor]
    return years


def generation_prices(source: "InputSource",
                      rows: pl.DataFrame,
                      *,
                      per_plant: bool = False) -> pl.DataFrame:
    """``rows`` ``[e, cohort, d, d_salv, ...]`` (Utf8: a generation, the period it comes online in and the period
    its salvage takes the net_salvage_value of) with the prices of the plant as constructed in ``d``
    (:data:`._cohorts.PRICE_COLUMNS` and ``a_b``, CUR/MW a year in service)::

        pay   = A = invest_cost[e,d]·1000·af(r, n),  r = discount_rate[e,d],  n = payback_time[e,d] (else L)
        a_cap = A·G(n)/G(L) (A when |n − L| ≤ YEAR_TOL),  L = lifetime_technical[e,d]
        a_salv = S·(1+r)^−L·af(r, L),  S = nsv_eol = 1000·net_salvage_value[e, d_salv]
        a_b   = a_cap − a_salv

    (:func:`_per_method_annuity_lf` with ``level``; the invest_cost of an entity with several invest methods
    counts once per method, as for its investment, or once with ``per_plant``: the replacement of existing
    capacity, whose entity may not invest).  0 without an invest_cost; no salvage without a lifetime."""
    from flextool.engine.preprocess._annuity import salvage_credit
    keys = rows.select("e", "d").unique(maintain_order=True)
    prices = _utf8_ed(_per_method_annuity_lf(
        source, keys.lazy().select(alias_to_axis("e", "e"), alias_to_axis("d", "d")), "invest_cost",
        INVEST_NOT_ALLOWED, None, level=True, per_plant=per_plant))
    salv = rows.select("e", "d_salv").unique(maintain_order=True)
    nsv = _utf8_ed(ed_values_on(source, NET_SALVAGE_VALUE,
                                salv.lazy().select(alias_to_axis("e", "e"), alias_to_axis("d_salv", "d")),
                                fill=0.0)).select("e", pl.col("d").alias("d_salv"), s=pl.col("value") * 1000.0)
    offset = _solve_inflation_scalars(source)[1]
    count = pl.col("method_count").cast(pl.Float64).fill_null(0.0)
    return (rows.join(prices, on=["e", "d"], how="left", maintain_order="left")
                .join(nsv, on=["e", "d_salv"], how="left", maintain_order="left")
                .with_columns(nsv_eol=pl.col("s").fill_null(0.0), a_cap=pl.col("a_cap").fill_null(0.0))
                .with_columns(a_salv=salvage_credit(pl.col("nsv_eol"), pl.col("disc"), pl.col("technical"), offset)
                              * count)
                .with_columns(pay=pl.col("ann").fill_null(0.0), n_pay=pl.col("life"), r_loan=pl.col("disc"),
                              a_b=pl.col("a_cap") - pl.col("a_salv"))
                .select(*rows.columns, *PRICE_COLUMNS, "a_b"))


def _generation_walk(walk: pl.DataFrame, years: dict[str, float], gens: pl.DataFrame,
                     last_period: str | None, root: str = "d") -> pl.DataFrame:
    """``[e, <root>, d_all, cohort, d_price, d_salv]``: each period ``d_all`` of a chain's walk (``walk`` ``[e, <root>,
    d_all]``, Utf8, along one lineage) with the generation ``cohort`` in service there (``y_on ≤ y[d_all] < y_eol``,
    ``years`` the run years), the period ``d_price`` it comes online in and the period ``d_salv`` of its salvage, both
    on the walk's own lineage: the walk's period at ``y_on`` (``y_eol``) where it has one (a generation starting
    inside a stochastic fan takes the values of its branch), else the generation's own ``d_on`` (``d_eol``, the
    last period of the run after the last period start).  ``gens`` is :func:`chain_generations` with ``d_build``
    renamed to ``<root>``."""
    from flextool.engine.preprocess.params._run_calendar import YEAR_TOL
    yr = pl.DataFrame({"d_all": list(years), "y": [float(v) for v in years.values()]},
                      schema={"d_all": pl.Utf8, "y": pl.Float64})
    walked = walk.join(yr, on="d_all", how="inner")
    rows = (walked.join(gens, on=["e", root], how="inner")
                  .filter(within_life(pl.col("y"), pl.col("y_on"), pl.col("y_eol"))))
    at = walked.select("e", root, pl.col("d_all").alias("_at"), pl.col("y").alias("_y"))

    def label(year: str, out: str) -> pl.DataFrame:
        return (gens.select("e", root, "cohort", year).join(at, on=["e", root], how="inner")
                    .filter((pl.col("_y") - pl.col(year)).abs() <= YEAR_TOL)
                    .sort("e", root, "cohort", "_at")
                    .unique(subset=["e", root, "cohort"], keep="first", maintain_order=True)
                    .select("e", root, "cohort", pl.col("_at").alias(out)))

    return (rows.join(label("y_on", "_on"), on=["e", root, "cohort"], how="left")
                .join(label("y_eol", "_eol"), on=["e", root, "cohort"], how="left")
                .select("e", root, "d_all", "cohort", d_price=pl.coalesce("_on", "d_on"),
                        d_salv=pl.coalesce("_eol", "d_eol", pl.lit(last_period, dtype=pl.Utf8))))


def priced_chain_rows(source: "InputSource",
                      walk: pl.DataFrame,
                      factor: pl.DataFrame,
                      years: dict[str, float],
                      gens: pl.DataFrame,
                      last_period: str | None) -> pl.DataFrame:
    """``[e, d, d_all, cohort, d_price, d_salv, a_b, factor, ...]``: each period of the walk ``walk`` (``[e, d,
    d_all]``, Utf8) of a chain of ``gens`` (:func:`chain_generations`) with the generation in service there
    (:func:`_generation_walk`), its price (:func:`generation_prices`, at the values of the walk's lineage) and the
    period's factor (``factor`` ``[d_all, factor]``; 1.0 where it has none).  Only the decisions with generations;
    a period of their walk that no generation covers is an error."""
    keyed = gens.rename({"d_build": "d"})
    walk = walk.join(keyed.select("e", "d").unique(), on=["e", "d"], how="semi")
    rows = _generation_walk(walk, years, keyed, last_period)
    covered = rows.group_by("e", "d").agg(pl.len().alias("_n"))
    whole = walk.group_by("e", "d").agg(pl.len().alias("_w")).join(covered, on=["e", "d"], how="left")
    gap = whole.filter(pl.col("_n").fill_null(0) != pl.col("_w"))
    if gap.height > 0:
        raise ValueError(f"The replacement generations of {sorted(gap['e'].unique().to_list())} do not cover the "
                         "periods their capacity is in service in; the chain cannot be priced.")
    return (generation_prices(source, rows.rename({"d": "_dec", "d_price": "d"}))
            .rename({"d": "d_price", "_dec": "d"})
            .join(factor, on="d_all", how="left")
            .with_columns(factor=pl.col("factor").fill_null(1.0))
            .sort("e", "d", "cohort", "d_all"))


def chain_lumps(source: "InputSource",
                walk: pl.DataFrame,
                factor: pl.DataFrame,
                years: dict[str, float],
                gens: pl.DataFrame,
                last_period: str | None) -> pl.DataFrame:
    """``[e, d, single, a_b, value]`` of the chains of ``gens`` over their walk (:func:`priced_chain_rows`): each
    generation's price summed over its periods in service::

        value[e,d] = Σ_g a_b[g]·Σ_{d′ ∈ walk(e,d), y_on[g] ≤ y[d′] < y_eol[g]} factor[d′]

    ``single`` when every period of the walk is priced alike: then ``a_b`` is that one price and the caller
    multiplies it by its own sum of the walk's factors (the chain of constant values charges today's annuity
    over the same periods, bit for bit)."""
    if gens.height == 0 or walk.height == 0:
        return _lumps_of(None)
    return _lumps_of(priced_chain_rows(source, walk, factor, years, gens, last_period))


def _lumps_of(priced: pl.DataFrame | None) -> pl.DataFrame:
    """:func:`chain_lumps` of the rows ``priced`` (:func:`priced_chain_rows`; ``None``: no chain)."""
    if priced is None or priced.height == 0:
        return pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8, "single": pl.Boolean, "a_b": pl.Float64,
                                    "value": pl.Float64})
    parts = (priced.group_by("e", "d", "cohort", maintain_order=True)
                   .agg(pl.col("a_b").first(), pl.col("a_b").n_unique().alias("_k"),
                        pl.col("factor").sum().alias("_sum")))
    return (parts.group_by("e", "d", maintain_order=True)
                 .agg(single=(pl.col("a_b").n_unique() == 1) & (pl.col("_k").max() == 1),
                      a_b=pl.col("a_b").first(),
                      value=(pl.col("a_b") * pl.col("_sum")).sum())
                 .sort("e", "d"))


def npv_invest_discounted_lf(
        source: "InputSource",
        active_solve: str | None,
        period_invest: list[str],
        period_in_use: list[str],
        period_universe: list[str],
        *,
        lineage: pl.DataFrame | None = None,
        commission: pl.DataFrame | None = None,
        year_rows: pl.DataFrame | None = None,
        provider: "object | None" = None,
        generations: pl.DataFrame | None = None,
        ) -> pl.LazyFrame:
    """Variant 1: ``ed_entity_annual_discounted`` (invest-side NPV).

    Per (e ∈ entityInvest, d ∈ period_invest)::

        price[e, d] × Σ_{d_all ∈ period_in_use, window(d, d_all)} inv_factor[d_all]

    where ``window`` and ``price`` switch on ``e``'s lifetime_method: ``reinvest_choice`` / ``no_reinvestment`` →
    the periods the cohort of the decision is in service (``yr[c] ≤ yr[d_all] < y_eol``, the window of the lifetime
    fixed cost, :func:`._lifetime_inputs._window_life_to_end_of_life`) at the level yearly price ``a_cap`` of the plant
    over its real lifetime (:func:`_per_method_annuity_lf` with ``level``); ``reinvest_automatic`` → unbounded, each
    period at the price of the generation of the decision's chain in service there (``generations``,
    :func:`chain_generations`; :func:`chain_lumps`), at the annuity for a decision without generation rows.
    An entity may have multiple methods; each contributes a separate sum.  See ``entity_annual_calc_params.py:222-251``.

    ``commission`` (``[e, d, d_c, yr_c]``, the decisions with a construction lead time): the price of such a
    decision is that of the period ``d_c`` it comes online (invest cost, interest rate, payback time and lifetime
    there) and its window starts at ``yr_c``, so it is paid only over the years the capacity operates within the
    solve.

    ``lineage`` — see :func:`._walks.period_walk_iterator`; the
    :func:`._branch.annuity_window_lineage_df` frame in a stochastic fan, ``None`` otherwise.

    The level price of ``reinvest_choice`` / ``no_reinvestment`` is lowered by the level salvage credit ``a_salv``
    (:func:`_level_prices_lf`): the bounded value is ``(a_cap − a_salv) · Σ inv_factor``.

    Returns lazy ``[e, d, value, a_cap, pay, n_pay, r_loan, a_salv, nsv_eol]`` (``value`` and the cohort's
    :data:`._cohorts.PRICE_COLUMNS`): ``a_cap`` the capital part of the
    price per year in service (the annuity for ``reinvest_automatic``), ``a_salv`` the salvage credit (0 for
    ``reinvest_automatic``), and for ``reinvest_choice`` / ``no_reinvestment`` the loan's yearly payment ``pay`` (A),
    its years ``n_pay``, its interest rate ``r_loan`` and the salvage value ``nsv_eol`` at the end of life (null for
    ``reinvest_automatic``).
    """
    if not period_invest:
        return pl.LazyFrame(schema={
            "e": schema_dtype(_enums, "e"),
            "d": schema_dtype(_enums, "d"),
            "value": pl.Float64,
            **{c: pl.Float64 for c in PRICE_COLUMNS},
        })
    # entityInvest = projection of entity__invest_method via the
    # method-allowed gate (mirror of preprocessing/invest_method_sets).
    entity_invest_set = _entity_invest_set_lf(source, INVEST_NOT_ALLOWED)
    pi_lf = pl.LazyFrame({"d": period_invest}).with_columns(
        alias_to_axis("d", "d"))
    ed_anchor_lf = entity_invest_set.join(pi_lf, how="cross")

    # Annuity, level price and salvage credit, of the period a decision with a lead time comes online.
    ann_lf = _level_prices_lf(source, active_solve, ed_anchor_lf, commission, provider=provider)

    # Lifetime methods: per-entity dispatch.
    all_e_lf = ed_anchor_lf.select("e").unique()
    elm_lf = _lifetime_method_with_default_lf(source, all_e_lf)
    # Choice/no_reinvestment → bounded; automatic → unbounded.  An entity
    # may have multiple methods; each contributes additively (mirrors
    # the if/if structure at L237/244).
    has_choice_or_no = (elm_lf
                          .filter(pl.col("method").is_in(
                              ["reinvest_choice", NO_REINVESTMENT]))
                          .select("e").unique()
                          .with_columns(has_choice=pl.lit(True)))
    has_automatic = (elm_lf
                       .filter(pl.col("method") == "reinvest_automatic")
                       .select("e").unique()
                       .with_columns(has_automatic=pl.lit(True)))

    # The bounded walk runs over the periods the cohort is in service: from the start of operation to the rounded
    # end of life of the lifetime of the period a decision with a lead time comes online.
    pwh_lf = pl.LazyFrame({"d": list({d for d in period_invest})}).with_columns(
        alias_to_axis("d", "d"))
    ed_pairs_lf = all_e_lf.join(pwh_lf, how="cross")

    technical_lf = _ed_lifetime_on_lf(source, ed_pairs_lf, LIFETIME_TECHNICAL, commission)
    life_lf = _window_life_to_end_of_life(source, active_solve, technical_lf, commission, provider=provider)
    commission_lf = _commission_walk_lf(commission)

    bounded = period_walk_iterator(
        source, active_solve, ed_anchor_lf,
        period_in_use, period_universe,
        bounded=True, life_lf=life_lf, factor_side="inv",
        lineage=lineage, commission_lf=commission_lf, year_rows=year_rows)
    unbounded = period_walk_iterator(
        source, active_solve, ed_anchor_lf,
        period_in_use, period_universe,
        bounded=False, life_lf=life_lf, factor_side="inv",
        lineage=lineage, commission_lf=commission_lf, year_rows=year_rows)

    # Combine.
    bounded = bounded.rename({"factor": "factor_b"})
    unbounded = unbounded.rename({"factor": "factor_u"})
    combined = (ed_anchor_lf
                  .join(ann_lf, on=["e", "d"], how="left")
                  .join(bounded, on=["e", "d"], how="left")
                  .join(unbounded, on=["e", "d"], how="left")
                  .join(has_choice_or_no, on="e", how="left")
                  .join(has_automatic, on="e", how="left")
                  .with_columns(
                      ann=pl.col("ann").fill_null(0.0),
                      a_cap=pl.col("a_cap").fill_null(0.0),
                      a_salv=pl.col("a_salv").fill_null(0.0),
                      factor_b=pl.col("factor_b").fill_null(0.0),
                      factor_u=pl.col("factor_u").fill_null(0.0),
                      has_choice=pl.col("has_choice").fill_null(False),
                      has_automatic=pl.col("has_automatic").fill_null(False),
                  )
                  .with_columns(
                      value=(
                          pl.when(pl.col("has_choice"))
                              .then((pl.col("a_cap") - pl.col("a_salv")) * pl.col("factor_b"))
                              .otherwise(0.0)
                          + pl.when(pl.col("has_automatic"))
                              .then(pl.col("ann") * pl.col("factor_u"))
                              .otherwise(0.0)
                      ),
                  ))
    # reinvest_automatic: each period at the price of the generation of the chain in service there.
    chain = _automatic_chain_lumps(source, active_solve, ed_anchor_lf, period_in_use, period_universe, life_lf,
                                   generations, lineage=lineage, commission_lf=commission_lf, year_rows=year_rows,
                                   provider=provider)
    if chain is not None:
        combined, chain = align_join_dtypes(combined, chain, ["e", "d"])
        auto = (pl.when(pl.col("_single")).then(pl.col("_a_b") * pl.col("factor_u"))
                  .when(pl.col("_chain").is_not_null()).then(pl.col("_chain"))
                  .otherwise(pl.col("ann") * pl.col("factor_u")))
        combined = (combined.join(chain, on=["e", "d"], how="left")
                            .with_columns(_single=pl.col("_single").fill_null(False))
                            .with_columns(value=(pl.when(pl.col("has_choice"))
                                                   .then((pl.col("a_cap") - pl.col("a_salv")) * pl.col("factor_b"))
                                                   .otherwise(0.0)
                                                 + pl.when(pl.col("has_automatic")).then(auto).otherwise(0.0)))
                            .drop("_single", "_a_b", "_chain"))
        _notice_automatic_replacement_cost(
            combined.filter(pl.col("has_automatic"))
                    .select("e", changed=(pl.col("value") - pl.col("ann") * pl.col("factor_u")).abs()
                            > 1e-12 * pl.max_horizontal(pl.lit(1.0), (pl.col("ann") * pl.col("factor_u")).abs())))
    # Only the entities whose investment takes the bounded window in this solve (the reinvest_automatic sum is
    # unchanged).  The value before: the annuity over whole periods that start before the end of the lifetime of
    # the start of operation.
    _notice_annuity_over_payback_time(
        combined.join(has_choice_or_no.select("e"), on="e", how="semi")
                .select("e", "d", value=pl.col("a_cap") * pl.col("factor_b")),
        lambda: (period_walk_iterator(source, active_solve, ed_anchor_lf, period_in_use, period_universe,
                                      bounded=True, life_lf=technical_lf, factor_side="inv", lineage=lineage,
                                      commission_lf=commission_lf, year_rows=year_rows)
                 .join(ann_lf, on=["e", "d"], how="left")
                 .select("e", "d", value=pl.col("factor") * pl.col("ann").fill_null(0.0))))
    bounded = pl.col("has_choice")
    return combined.select("e", "d", "value",
                           a_cap=pl.when(bounded).then(pl.col("a_cap")).otherwise(pl.col("ann")),
                           pay=pl.when(bounded).then(pl.col("ann")),
                           n_pay=pl.when(bounded).then(pl.col("life")),
                           r_loan=pl.when(bounded).then(pl.col("disc")),
                           a_salv=pl.when(bounded).then(pl.col("a_salv")).otherwise(0.0),
                           nsv_eol=pl.when(bounded).then(pl.col("nsv_eol")))


def _automatic_chain_lumps(source: "InputSource",
                           active_solve: str | None,
                           ed_anchor_lf: pl.LazyFrame,
                           period_in_use: list[str],
                           period_universe: list[str],
                           life_lf: pl.LazyFrame,
                           generations: pl.DataFrame | None,
                           *,
                           lineage: pl.DataFrame | None,
                           commission_lf: pl.LazyFrame | None,
                           year_rows: pl.DataFrame | None,
                           provider: "object | None") -> pl.LazyFrame | None:
    """``[e, d, _single, _a_b, _chain]`` (:func:`chain_lumps` renamed) of the investment decisions of
    ``ed_anchor_lf`` with replacement generations, over the unbounded walk of the investment lump (its lineage and
    construction lead time) and its investment factors; ``None`` without generations."""
    if generations is None or generations.height == 0:
        return None
    from flextool.engine.preprocess.params._cohorts import salvage_periods
    from flextool.engine.preprocess.params._derived import _fan_walk_provider
    walk = _utf8_ed(period_walk_iterator(source, active_solve, ed_anchor_lf, period_in_use, period_universe,
                                         bounded=False, life_lf=life_lf, factor_side=None, lineage=lineage,
                                         commission_lf=commission_lf, year_rows=year_rows),
                    ("e", "d", "d_all"))
    factor = _utf8_ed(_inflation_factors_lf(source, active_solve, period_universe, provider=_fan_walk_provider(),
                                            year_rows=year_rows).select(pl.col("d").alias("d_all"),
                                                                        pl.col("inv_factor").alias("factor")),
                      ("d_all",))
    years = _chain_years(source, active_solve, provider)
    last = salvage_periods(source, active_solve, years, provider, [None])[0]
    lumps = chain_lumps(source, walk, factor, years, generations, last)
    return (lumps.lazy().select(alias_to_axis("e", "e"), alias_to_axis("d", "d"), _single=pl.col("single"),
                                _a_b=pl.col("a_b"), _chain=pl.col("value")))


AUTOMATIC_REPLACEMENT_COST = "automatic_replacement_cost"


def _notice_automatic_replacement_cost(changed: pl.LazyFrame) -> None:
    """The ``automatic_replacement_cost`` notice for the entities its marker names (an upgraded database whose
    reinvest_automatic capacity was charged the annuity of its first plant for ever) whose investment cost now
    differs: ``changed`` ``[e, changed]``.  Built only while the marker names entities."""
    from flextool.engine import _behaviour_changes as bc
    if not bc.has_marker(AUTOMATIC_REPLACEMENT_COST):
        return
    names = {str(n) for n in bc.marker_json(AUTOMATIC_REPLACEMENT_COST).get("entities") or []}
    if not names:
        return
    hit = sorted(names & {str(e) for e in changed.filter(pl.col("changed")).select("e").collect()["e"].to_list()})
    if hit:
        bc.notice(AUTOMATIC_REPLACEMENT_COST, "entities", hit)


def _labels_on_lineage(rows: pl.DataFrame,
                       years: dict[str, float],
                       lineage: pl.DataFrame | None,
                       last_period: str | None) -> pl.DataFrame:
    """``rows`` (``[..., d, y_on, y_eol, d_on, d_eol]``, Utf8: a generation seen from the period ``d``) with
    ``d_price`` and ``d_salv``: the period the generation comes online in (its salvage period) on ``d``'s scenario
    lineage, ``d`` itself or a period paired with it in ``lineage`` (``[d, d_other]``) whose year is ``y_on``
    (``y_eol``); else the generation's own ``d_on`` (``d_eol``, the last period of the run after the last period
    start).  A generation starting inside a stochastic fan takes the values of ``d``'s branch."""
    from flextool.engine.preprocess.params._run_calendar import YEAR_TOL
    own = rows.with_columns(d_price=pl.col("d_on"), d_salv=pl.coalesce("d_eol", pl.lit(last_period, dtype=pl.Utf8)))
    if lineage is None or lineage.height == 0:
        return own
    yr = pl.DataFrame({"_p": list(years), "_y": [float(v) for v in years.values()]},
                      schema={"_p": pl.Utf8, "_y": pl.Float64})
    d_values = rows.select("d").unique()
    pairs = pl.concat([lineage.select(pl.col("d").cast(pl.Utf8), _p=pl.col("d_other").cast(pl.Utf8)),
                       d_values.select("d", _p=pl.col("d"))]).unique().join(yr, on="_p", how="inner")
    keys = [c for c in rows.columns]

    def pick(year: str, out: str) -> pl.DataFrame:
        return (rows.select(keys).join(pairs, on="d", how="inner")
                    .filter((pl.col("_y") - pl.col(year)).abs() <= YEAR_TOL)
                    .sort("_p").unique(subset=keys, keep="first", maintain_order=True)
                    .select(*keys, pl.col("_p").alias(out)))

    return (own.join(pick("y_on", "_on"), on=keys, how="left")
               .join(pick("y_eol", "_eol"), on=keys, how="left")
               .with_columns(d_price=pl.coalesce("_on", "d_price"), d_salv=pl.coalesce("_eol", "d_salv"))
               .drop("_on", "_eol"))


# ---------------------------------------------------------------------------
# Public Param-producing entry points
# ---------------------------------------------------------------------------


def ed_entity_annual_discounted_from_source(
        source: "InputSource",
        active_solve: str | None,
        period_invest: list[str],
        period_in_use: list[str],
        period_universe: list[str],
        *,
        lineage: pl.DataFrame | None = None,
        commission: pl.DataFrame | None = None,
        year_rows: pl.DataFrame | None = None,
        provider: "object | None" = None,
        ) -> "Param | None":
    """Public entry: ``ed_entity_annual_discounted[e, d]``.

    Returns the full ``entityInvest × period_invest`` frame, including
    zero-valued rows (``apply_npv`` registers it as the solve's
    ``solve_data/ed_entity_annual_discounted`` frame, :func:`._lifetime_inputs._register_invest_lumps`).

    ``lineage`` — see :func:`._walks.period_walk_iterator`; the
    :func:`._branch.annuity_window_lineage_df` frame in a stochastic fan, ``None`` otherwise.
    """
    if not period_invest:
        return None
    out = (npv_invest_discounted_lf(
              source, active_solve,
              period_invest, period_in_use, period_universe,
              lineage=lineage, commission=commission, year_rows=year_rows, provider=provider)
              .select("e", "d", "value")
              .sort("e", "d")
              .collect())
    if out.height == 0:
        return None
    return Param(("e", "d"), out)


def apply_invest_lumps(flex_data: object,
                       source: "InputSource",
                       active_solve: str,
                       period_invest: list[str],
                       period_in_use: list[str],
                       period_universe: list[str],
                       period_with_history: list[str],
                       *,
                       lineage: pl.DataFrame | None,
                       year_rows: pl.DataFrame | None,
                       provider: "object | None" = None,
                       ) -> None:
    """Set the objective's investment lumps ``ed_entity_annual_discounted`` (the level yearly price over the years
    in service) and ``ed_lifetime_fixed_cost`` (the fixed cost over the years in service) of ``active_solve``,
    register their ``solve_data`` frames (:func:`._lifetime_inputs._register_invest_lumps`) and write each investment
    cohort's prices into the cohort table (``flex_data.cohort``: ``a_cap``, ``pay``, ``n_pay``, ``r_loan``, ``a_salv``,
    ``nsv_eol``, carried to the later solves by ``realized_cohort``).  These are the only derivation of both lumps: a
    Spine solve (:func:`._lifetime_costs.apply_npv`) and a synthetic sub-solve
    (:func:`._lifetime_costs.apply_run_fields_of_synthetic_solve`) call it alike.  The start of operation of a
    construction lead time is ``flex_data.ed_invest_commission``.  A helper that returns None (no invest period, no
    entity row) leaves the field as it is."""
    commission = getattr(flex_data, "ed_invest_commission", None)
    generations = chain_generations(getattr(flex_data, "cohort", None))
    if period_invest:
        lumps = (npv_invest_discounted_lf(source, active_solve, period_invest, period_in_use, period_universe,
                                          lineage=lineage, commission=commission, year_rows=year_rows,
                                          provider=provider, generations=generations)
                 .sort("e", "d").collect())
        if lumps.height > 0:
            flex_data.ed_entity_annual_discounted = Param(("e", "d"), lumps.select("e", "d", "value"))
            _set_cohort_prices(flex_data, lumps)
            _set_generation_prices(flex_data, source, active_solve, generations, provider=provider)
            warn_salvage_above_cost(getattr(flex_data, "cohort", None))
            _notice_net_salvage_once(lumps.filter(pl.col("a_salv") != 0.0)["e"].cast(pl.Utf8).unique().to_list())
    fixed = ed_lifetime_fixed_cost_from_source(
        source, active_solve, period_with_history, period_in_use, period_universe,
        lineage=lineage, commission=commission, year_rows=year_rows, provider=provider)
    if fixed is not None:
        flex_data.ed_lifetime_fixed_cost = fixed
    _register_invest_lumps(flex_data, provider)


def _set_cohort_prices(flex_data: object, lumps: pl.DataFrame) -> None:
    """``flex_data.cohort`` with the prices (:data:`._cohorts.PRICE_COLUMNS`) of each investment cohort (``invest:<d>``
    of entity ``e``) from ``lumps`` (:func:`npv_invest_discounted_lf`); the existing cohorts keep theirs (no capital in
    the model; their salvage terms: :func:`._existing_salvage._set_existing_cohort_prices`)."""
    cohort = getattr(flex_data, "cohort", None)
    if cohort is None or cohort.height == 0:
        return
    from flextool.engine.preprocess.params._cohorts import KIND_INVEST
    prices = lumps.select("e", pl.col("d").alias("d_build"), *(pl.col(c).alias(f"_{c}") for c in PRICE_COLUMNS))
    left, prices = align_join_dtypes(cohort, prices, ["e", "d_build"])
    invest = pl.col("kind") == KIND_INVEST
    flex_data.cohort = (left
                        .join(prices, on=["e", "d_build"], how="left", maintain_order="left")
                        .with_columns(pl.when(invest).then(pl.col(f"_{c}")).otherwise(pl.col(c)).alias(c)
                                      for c in PRICE_COLUMNS)
                        .select(cohort.columns))


def warn_salvage_above_cost(cohort: pl.DataFrame | None) -> None:
    """Warn, once per run and entity, where a plant the model can build is priced below 0 a year: a cohort of an
    investment decision or a generation of its chain whose salvage credit exceeds its capital price
    (``a_cap − a_salv < 0``: the net_salvage_value at the end of its life is worth more than invest_cost at its
    discount_rate).  Every such plant earns money, so the model builds as much as the investment limits allow."""
    from flextool import run_warnings
    from flextool.engine.preprocess.params._cohorts import KIND_EXISTING
    if cohort is None or cohort.height == 0 or "root" not in cohort.columns:
        return
    rows = cohort.filter(~pl.col("root").str.starts_with(f"{KIND_EXISTING}:") & pl.col("a_cap").is_not_null()
                         & pl.col("a_salv").is_not_null() & (pl.col("a_cap") - pl.col("a_salv") < -1e-9))
    names = sorted({str(e) for e in rows["e"].cast(pl.Utf8).to_list()})
    run_warnings.warn_grouped("W-0527", names, per_run=True)


def _set_generation_prices(flex_data: object,
                           source: "InputSource",
                           active_solve: str | None,
                           generations: pl.DataFrame,
                           *,
                           provider: "object | None",
                           per_plant: bool = False) -> None:
    """``flex_data.cohort`` with the prices (:data:`._cohorts.PRICE_COLUMNS`) of each generation of ``generations``
    (:func:`chain_generations`): the plant as constructed in its ``d_on``, its salvage at ``d_eol`` (the last period
    of the run after the last period start), :func:`generation_prices`.  Carried by ``realized_cohort`` as they
    are; a later solve never prices a carried chain again."""
    cohort = getattr(flex_data, "cohort", None)
    if cohort is None or cohort.height == 0 or generations.height == 0:
        return
    from flextool.engine.preprocess.params._cohorts import salvage_periods
    years = _chain_years(source, active_solve, provider)
    last = salvage_periods(source, active_solve, years, provider, [None])[0]
    rows = generations.select("e", "cohort", d=pl.col("d_on"),
                              d_salv=pl.coalesce("d_eol", pl.lit(last, dtype=pl.Utf8)))
    prices = (generation_prices(source, rows, per_plant=per_plant)
              .select("e", "cohort", *(pl.col(c).alias(f"_{c}") for c in PRICE_COLUMNS)))
    left = cohort.with_columns(_e=pl.col("e").cast(pl.Utf8))
    flex_data.cohort = (left.join(prices.rename({"e": "_e"}), on=["_e", "cohort"], how="left", maintain_order="left")
                            .with_columns(pl.when(pl.col("_a_cap").is_not_null()).then(pl.col(f"_{c}"))
                                          .otherwise(pl.col(c)).alias(c) for c in PRICE_COLUMNS)
                            .select(cohort.columns))


__all__ = [
    "ed_entity_annual_discounted_from_source",
    "apply_invest_lumps",
]

