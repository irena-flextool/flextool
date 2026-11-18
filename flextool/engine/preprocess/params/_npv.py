"""Cluster A — annual integration / NPV (Δ.5).

Lazy-polars port of flextool's NPV / lifetime-fixed-cost / inflation
cascade.  Replaces the eager-Python implementation that previously lived
in :mod:`._derived` (``ed_entity_annual_family_from_source`` plus
its supporting per-(entity, period) lookup).

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
  credit of existing capacity (:func:`existing_salvage`).
* ``cd_divest_built_cost`` / ``cd_divest_built_fixed`` — the settlement of a
  retired MW of a model-built cohort less the level price given back, and the
  fixed cost it saves (:func:`built_retirement_costs`).
* ``ed_lifetime_fixed_cost``                   — invest-side lifetime
  cumulative fixed cost.  Replaces ``solve_data/ed_lifetime_fixed_cost.csv``.
* ``ed_lifetime_fixed_cost_divest``            — divest-side lifetime
  cumulative fixed cost.  Replaces ``solve_data/ed_lifetime_fixed_cost_divest.csv``.

and, for the reported costs only (no model term reads them), their run-level
twins ``*_run`` discounted from the start of the run over the periods of the
whole run, the undiscounted yearly amounts ``edd_annuity`` (an investment's price per year in service, per
period) /
``ed_divest_settlement_yearly`` (a retirement's salvage settlement as a yearly amount of its period),
the operating / retirement windows ``edd_operating_window`` / ``edd_retire_window`` (section "Run-level
fields of the reported costs").

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
   :func:`period_walk_iterator` for the per-period accumulation walk.

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
        (the periods the cohort is in service; :func:`_window_life_to_end_of_life`).
      - ``reinvest_automatic``: ``pdy[d_all] ≥ pdy[d]``.
      - the fixed cost a retirement saves: the retirement's periods until the end of life of the existing
        capacity (:func:`_retire_window_lf`).
  * The salvage terms of a retirement of existing capacity are no annuity window: :func:`existing_salvage`.
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
  (:func:`_inflation_factors_lf`) for use here.
* :func:`._derived._p_years_d_lf` — already lazy; reused as-is.
* :func:`._derived._period_in_use_set` and
  :func:`._derived._solve_periods` — list builders used to
  bound the period domain.
"""
from __future__ import annotations

from pathlib import Path
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
from flextool.engine.preprocess.params._branch import inherit_active_branch_periods
from flextool.engine.preprocess.params._cohorts import BOUNDED_METHODS, PRICE_COLUMNS
from flextool.engine.input._lifetime_roles import (
    LIFETIME_TECHNICAL,
    NET_SALVAGE_VALUE,
    NO_REINVESTMENT,
    PAYBACK_TIME,
)
from flextool.engine.preprocess.params._period_values import at_online_period, ed_values_on, ed_values_online
from flextool.engine.preprocess.params._existing import (
    DIVEST_NOT_ALLOWED as _DIVEST_NOT_ALLOWED,
    INVEST_NOT_ALLOWED as _INVEST_NOT_ALLOWED,
)

# Substrate handle for the cascade-wide axis enum vocabulary.
# Bare ``None`` here; ``cast_dim`` / ``schema_dtype`` in
# ``_axis_enums`` fall back to ``_LIVE_AXIS_ENUMS_CTX`` (the live
# ContextVar) when this is ``None``, so substrate sites pick up
# activation set by ``load_flextool`` automatically.
_enums: "dict | None" = None

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource


# The invest/retire method lists (_INVEST_NOT_ALLOWED / _DIVEST_NOT_ALLOWED)
# are imported from params/_existing.py, their one definition.

# Default lifetime_method when an entity has no explicit row
# (preprocessing/method_with_fallback_sets.py::_LIFETIME_METHOD_DEFAULT).
_LIFETIME_METHOD_DEFAULT = "reinvest_automatic"

# Inflation defaults — mirrors period_calculated_params.py:221-227 and
# _solve_inflation_inputs in _derived.py.
_INFL_RATE_DEFAULT = 0.0
_INFL_OFFSET_INV_DEFAULT = 0.0
_INFL_OFFSET_OPS_DEFAULT = 0.5


# ---------------------------------------------------------------------------
# Building blocks (lazy frames)
# ---------------------------------------------------------------------------


def _entity_class_lf(source: "InputSource") -> pl.LazyFrame:
    """Lazy ``(e, ec)`` frame: every (entity, entity_class) pairing.

    Entity classes: ``unit``, ``node``, ``connection``.  ``unit ∪
    connection`` forms flextool's ``process_set``; ``node`` is its own
    class.  The frame supports the per-entity-class gates of the
    lifetime and annuity readers.
    """
    parts: list[pl.LazyFrame] = []
    for ec in ("unit", "node", "connection"):
        try:
            df = source.entities(ec)
        except KeyError:
            continue
        if df.height == 0:
            continue
        parts.append(df.lazy().select(
            alias_to_axis("name", "e"),
            pl.lit(ec).alias("ec"),
        ))
    if not parts:
        return pl.LazyFrame(schema={"e": schema_dtype(_enums, "e"),
                                     "ec": pl.Utf8})
    return pl.concat(parts, how="vertical")


def _entity_class_lookup_lf(entity_class_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Lazy ``(e, is_process, is_node)`` frame: bool flags per entity.

    ``is_process = ec ∈ {unit, connection}``, ``is_node = ec == 'node'``.
    """
    return (entity_class_lf
              .with_columns(
                  is_process=pl.col("ec").is_in(["unit", "connection"]),
                  is_node=pl.col("ec") == "node",
              )
              .group_by("e")
              .agg(
                  pl.col("is_process").any().alias("is_process"),
                  pl.col("is_node").any().alias("is_node"),
              ))


def _per_entity_param_lf(source: "InputSource",
                            parameter_name: str,
                            ) -> pl.LazyFrame:
    """Lazy ``(e, d, value)`` frame from ``unit/node/connection.<param>``.

    The per-(entity, period) cascade in lazy form (used for ``fixed_cost``;
    lifetime, investment cost, salvage value and interest rate are read
    through :func:`._period_values.ed_values_on`):

    * Per-period (Map) rows produce ``(e, d, value)`` rows directly.
    * Scalar rows produce ``(e, value)`` rows with ``d`` null;
      consumers cross-join with the period universe.

    Returns columns ``[e, d, value, is_scalar]`` — ``is_scalar`` flags
    rows that need broadcasting.  ``d`` is null on scalar rows.

    A stochastic branch period inherits its anchor's explicit row here
    (unconditionally, not only under recourse — #63): see the matching
    comment in :func:`._existing._per_entity_param_lf`.
    """
    parts: list[pl.LazyFrame] = []
    for ec in ("unit", "node", "connection"):
        try:
            df = source.parameter(ec, parameter_name)
        except KeyError:
            continue
        if df.height == 0:
            continue
        cols = df.columns
        # Period-dim detection — see the matching helper in
        # ``_existing._per_entity_param_lf``.  When the source
        # exposes the period axis under a user-renamed ``Map.index_name``
        # (e.g. ``x``), checking only ``"period" in cols`` misses it and
        # the scalar-broadcast branch explodes per-period rows through
        # ``_resolve_per_period_lf``'s ``scalar.join(on="e")``.
        extra = [c for c in cols if c not in ("name", "value")]
        period_col = "period" if "period" in extra else (extra[0] if extra else None)
        if period_col is not None:
            df = inherit_active_branch_periods(
                df, group_keys=("name",), period_col=period_col)
            # Per-ROW scalar detection.  A scalar value stored alongside
            # Map values in the same entity class surfaces with a NULL
            # index ((e.g. ``x = null``) — Spine emits one ``x`` column
            # for the whole class, so a unit with a constant ``existing``
            # while a sibling unit carries a period-Map shows up as a
            # null-period row.  Such rows must broadcast across the
            # period universe (``is_scalar=True``), NOT be treated as an
            # explicit ``(e, d=null)`` row that fails to join the period
            # grid and silently fills 0.  Classifying by ``period_col``'s
            # null-ness per row (rather than per frame) covers the mixed
            # scalar+Map class.
            parts.append(df.lazy().select(
                alias_to_axis("name", "e"),
                alias_to_axis(pl.col(period_col).cast(pl.Utf8, strict=False),
                              "d"),
                pl.col("value").cast(pl.Float64, strict=False),
                pl.col(period_col).is_null().alias("is_scalar"),
            ))
        else:
            parts.append(df.lazy().select(
                alias_to_axis("name", "e"),
                pl.lit(None).cast(schema_dtype(_enums, "d"), strict=False).alias("d"),
                pl.col("value").cast(pl.Float64, strict=False),
                pl.lit(True).alias("is_scalar"),
            ))
    if not parts:
        return pl.LazyFrame(schema={
            "e": schema_dtype(_enums, "e"),
            "d": schema_dtype(_enums, "d"),
            "value": pl.Float64, "is_scalar": pl.Boolean,
        })
    return pl.concat(parts, how="vertical")


def _resolve_per_period_lf(per_param: pl.LazyFrame,
                              ed_lf: pl.LazyFrame,
                              fill: float | None = 0.0,
                              ) -> pl.LazyFrame:
    """Resolve ``pdX[e, param, d]`` cascade in lazy form.

    Order:
      1. Explicit ``(e, d)`` row → take that ``value``.
      2. Else scalar ``(e, ·)`` row → broadcast across ``ed_lf``.
      3. Else → ``fill``.

    Returns ``ed_lf`` with an extra ``value`` column (Float64).
    ``ed_lf`` must carry columns ``[e, d, ...]``.

    ``per_param`` already carries its stochastic-branch rows (#63 —
    :func:`_per_entity_param_lf` inherits them unconditionally), so no
    further branch expansion happens here.
    """
    explicit = (per_param
          .filter(~pl.col("is_scalar"))
          .select("e", "d", pl.col("value").alias("v_explicit")))
    scalar = (per_param
                .filter(pl.col("is_scalar"))
                .select("e", pl.col("value").alias("v_scalar")))
    return (ed_lf
              .join(explicit, on=["e", "d"], how="left")
              .join(scalar, on="e", how="left")
              .with_columns(
                  value=pl.coalesce(
                      pl.col("v_explicit"),
                      pl.col("v_scalar"),
                      pl.lit(fill, dtype=pl.Float64),
                  ),
              )
              .drop("v_explicit", "v_scalar"))


def _entity_method_lf(source: "InputSource",
                          parameter_name: str,
                          ) -> pl.LazyFrame:
    """Lazy ``(e, method)`` frame from ``unit/node/connection.<param>``.

    ``parameter_name`` is one of ``invest_method`` or ``lifetime_method``.
    Each entity may carry multiple method rows (Spine schema allows multi-
    valued methods); we preserve all rows.  The ``invest_method`` default
    is ``not_allowed``; the ``lifetime_method`` default is
    ``reinvest_automatic`` — the latter is applied at the consumer site
    (:func:`_lifetime_method_with_default_lf`).
    """
    parts: list[pl.LazyFrame] = []
    for ec in ("unit", "node", "connection"):
        try:
            df = source.parameter(ec, parameter_name)
        except KeyError:
            continue
        if df.height == 0:
            continue
        parts.append(df.lazy().select(
            alias_to_axis("name", "e"),
            pl.col("value").cast(pl.Utf8, strict=False).alias("method"),
        ))
    if not parts:
        return pl.LazyFrame(schema={"e": schema_dtype(_enums, "e"),
                                     "method": pl.Utf8})
    return pl.concat(parts, how="vertical")


def _lifetime_method_with_default_lf(
        source: "InputSource",
        all_entities_lf: pl.LazyFrame,
        ) -> pl.LazyFrame:
    """Lazy ``(e, method)`` frame for ``lifetime_method`` with default fill.

    Entities without an explicit ``lifetime_method`` row get
    ``reinvest_automatic`` (mirrors
    ``preprocessing/method_with_fallback_sets.py::_LIFETIME_METHOD_DEFAULT``).
    ``all_entities_lf`` carries ``[e]``.
    """
    explicit = _entity_method_lf(source, "lifetime_method")
    explicit_entities = explicit.select("e").unique()
    default_e = (all_entities_lf
                   .join(explicit_entities, on="e", how="anti")
                   .with_columns(method=pl.lit(_LIFETIME_METHOD_DEFAULT)))
    return pl.concat([explicit, default_e], how="vertical")


# ---------------------------------------------------------------------------
# Inflation factors (lazy)
# ---------------------------------------------------------------------------


def _solve_inflation_scalars(source: "InputSource"
                                  ) -> tuple[float, float, float]:
    """Read ``model.social_discount_rate`` / ``model.discount_offset_investment`` /
    ``model.discount_offset_operations`` scalars, falling back to
    flextool's CSV-side defaults when the Spine reader returned
    only-default rows.

    Mirrors :func:`._derived._solve_inflation_inputs` semantics.
    """
    def _explicit_max(par: str, default: float) -> float:
        try:
            df = source.parameter("model", par)
        except KeyError:
            return default
        if df.height == 0:
            return default
        try:
            spine_default = source.parameter_default("model", par)
        except KeyError:
            spine_default = None
        if spine_default is not None:
            try:
                spine_default_f = float(spine_default)
            except (ValueError, TypeError):
                spine_default_f = None
            if spine_default_f is not None:
                vals = df["value"].cast(pl.Float64, strict=False).to_list()
                if vals and all(
                        v is not None and abs(v - spine_default_f) < 1e-12
                        for v in vals):
                    return default
        try:
            return float(df["value"].cast(pl.Float64).max())
        except Exception:
            return default

    return (
        _explicit_max("social_discount_rate", _INFL_RATE_DEFAULT),
        _explicit_max("discount_offset_investment", _INFL_OFFSET_INV_DEFAULT),
        _explicit_max("discount_offset_operations", _INFL_OFFSET_OPS_DEFAULT),
    )


def _years_for_period_lf(source: "InputSource",
                              active_solve: str | None,
                              period_universe: list[str],
                              *,
                              provider: "object | None" = None,
                              ) -> pl.LazyFrame:
    """Build the per-(d, year_label, width) frame from
    ``solve.years_represented`` mirroring
    :func:`._derived._years_for_period_from_source` in lazy form.

    Returns columns ``[d, y, width]`` where ``y`` is a string label.
    Empty if no rows.  ``provider`` (Slice D α-1) revives the canonical
    ``p_years_represented.csv`` arm carrying fan-member year rows.
    """
    # Materialise scalars eagerly — they're solve-level, tiny.
    from flextool.engine.preprocess.params._derived import _years_for_period_from_source
    yfp = _years_for_period_from_source(
        source, active_solve, period_universe, provider=provider)
    rows: list[tuple[str, str, float]] = []
    for d, years in yfp.items():
        for y, w in years:
            rows.append((d, y, w))
    if not rows:
        return pl.LazyFrame(schema={
            "d": schema_dtype(_enums, "d"),
            "y": pl.Utf8, "width": pl.Float64,
        })
    return pl.LazyFrame(rows, schema=["d", "y", "width"], orient="row").with_columns(
        alias_to_axis("d", "d"))


def _inflation_year_rows_lf(source: "InputSource",
                           active_solve: str | None,
                           period_universe: list[str],
                           *,
                           provider: "object | None" = None,
                           year_rows: pl.DataFrame | None = None,
                           ) -> pl.LazyFrame | None:
    """Lazy ``[d, off, width, base, inv_contrib, ops_contrib]``: one row per represented year of a period of
    ``period_universe`` (``d`` on the period axis), ``off`` the years of the period before it, ``width`` its width,
    ``base`` the years from the discount base to its start and ``inv_contrib`` / ``ops_contrib`` its discounted width
    (the terms of :func:`_inflation_factors_lf`).

    Algorithm — port of ``preprocessing/period_calculated_params.py:230-322``:

      1. Scalar inputs: ``rate``, ``offset_invest``, ``offset_operations``.
      2. For each (d, y) ∈ ``years_for_period``:
           ``base[d, y] = Σ_{y' ∈ global, y' < y} pyr[d, y']``  (default 1)
           ``until_inv[d, y] = base + pyr × offset_invest``
           ``until_ops[d, y] = base + pyr × offset_operations``
           ``inv_contrib[d, y] = pyr[d, y] × (1+rate)^(-until_inv[d, y])`` (``ops_contrib`` likewise).

    The cumulative-base computation (step 2) is per-period over the
    GLOBAL year set, so it's shaped naturally as a polars window:

      base[d, y] = Σ_{y'<y in global order} pyr[d, y']
                 = (cumulative_sum of pyr[d, y'] over y'<y, partitioned by d)

    where unbound (d, y) pairs contribute 1 (default ``years_represented``).  ``off`` counts only the period's own
    rows.

    ``year_rows`` (``[d, width, base]``, one row per represented year of a period): the years given explicitly
    instead of from ``years_for_period``; ``base`` is the number of years from the discount base to the start of
    that year, so ``until = base + width × offset``.

    ``None`` when there are no year rows (every period has the factor 1).  The rows keep the order the factor sums
    take them in (the order of ``year_rows`` / of ``years_for_period``).
    """
    rate, off_inv, off_ops = _solve_inflation_scalars(source)
    one_plus_inv = (1.0 / (1.0 + rate)) if rate != -1.0 else 1.0
    if year_rows is not None:
        return (year_rows.lazy()
                  .select(alias_to_axis(pl.col("d").cast(pl.Utf8), "d"), pl.col("width").cast(pl.Float64),
                          pl.col("base").cast(pl.Float64))
                  .with_columns(
                      off=(pl.col("width").cum_sum() - pl.col("width")).over("d", order_by="base"),
                      inv_contrib=pl.col("width") * (one_plus_inv ** (pl.col("base")
                                                                      + pl.col("width") * off_inv)),
                      ops_contrib=pl.col("width") * (one_plus_inv ** (pl.col("base")
                                                                      + pl.col("width") * off_ops)))
                  .select("d", "off", "width", "base", "inv_contrib", "ops_contrib"))
    yfp_lf = _years_for_period_lf(
        source, active_solve, period_universe, provider=provider)

    # Materialise per-period × per-y.  At LP scale this is small (typically
    # ≤ 100 (d, y) rows even for 50-year horizon × 20 representative years),
    # so collecting once is cheap; the rest stays lazy.
    yfp_eager = yfp_lf.collect()
    if yfp_eager.height == 0:
        # No years_represented data — every period gets the trivial 1.0
        # factor (mirrors period_calculated_params.py:300).
        return None
    period_lf = pl.LazyFrame({"d": period_universe}).with_columns(
        alias_to_axis("d", "d"))

    # Build the global year set with numerical sort when possible.
    try:
        global_years = sorted(
            yfp_eager["y"].unique().to_list(), key=lambda y: float(y)
        )
    except ValueError:
        global_years = sorted(yfp_eager["y"].unique().to_list())

    # Cross-join (d, y_global) so missing rows default to width=1.0
    # (mirrors flextool's default of p_years_represented).
    global_y_lf = pl.LazyFrame({"y": global_years}).with_columns(
        y_idx=pl.int_range(0, pl.len(), dtype=pl.Int64)
    )
    full = (period_lf
              .join(global_y_lf, how="cross")
              .join(yfp_eager.lazy(), on=["d", "y"], how="left")
              .with_columns(
                  pyr=pl.col("width").fill_null(1.0),
              ))

    # Cumulative base[d, y] = Σ pyr[d, y'] over y' < y (within d).
    full = (full
              .sort(["d", "y_idx"])
              .with_columns(
                  base=pl.col("pyr").cum_sum().over("d") - pl.col("pyr"),
              ))

    # Restrict to (d, y) actually bound to this period.
    bound = (yfp_eager.lazy()
                .select("d", "y")
                .join(full, on=["d", "y"], how="inner"))

    # Per-(d, y) factor contributions.
    return (bound
              .with_columns(
                  until_inv=pl.col("base") + pl.col("pyr") * off_inv,
                  until_ops=pl.col("base") + pl.col("pyr") * off_ops,
                  off=(pl.col("pyr").cum_sum() - pl.col("pyr")).over("d", order_by="y_idx"),
              ).with_columns(
                  inv_contrib=pl.col("pyr") * (one_plus_inv ** pl.col("until_inv")),
                  ops_contrib=pl.col("pyr") * (one_plus_inv ** pl.col("until_ops")),
              )
              .select("d", "off", pl.col("pyr").alias("width"), "base", "inv_contrib", "ops_contrib"))


def _inflation_factors_lf(source: "InputSource",
                                active_solve: str | None,
                                period_universe: list[str],
                                *,
                                provider: "object | None" = None,
                                year_rows: pl.DataFrame | None = None,
                                ) -> pl.LazyFrame:
    """Lazy ``(d, inv_factor, ops_factor)`` frame: per period the sum of its year rows
    (:func:`_inflation_year_rows_lf`), ``inv_factor[d] = Σ_y inv_contrib[d, y]`` and ``ops_factor[d] = Σ_y
    ops_contrib[d, y]``.  A period without rows (or with zero total width) gets 1.0.
    """
    if not period_universe:
        return pl.LazyFrame(schema={
            "d": schema_dtype(_enums, "d"),
            "inv_factor": pl.Float64, "ops_factor": pl.Float64,
        })
    period_lf = pl.LazyFrame({"d": [str(d) for d in period_universe]}).with_columns(alias_to_axis("d", "d"))
    rows = _inflation_year_rows_lf(source, active_solve, period_universe, provider=provider, year_rows=year_rows)
    if rows is None:
        return period_lf.with_columns(inv_factor=pl.lit(1.0, dtype=pl.Float64),
                                      ops_factor=pl.lit(1.0, dtype=pl.Float64))
    factor_per_d = (rows
                      .group_by("d")
                      .agg(pl.col("inv_contrib").sum().alias("inv_factor"),
                           pl.col("ops_contrib").sum().alias("ops_factor"),
                           pl.col("width").sum().alias("sum_pyr"))
                      # Sum-pyr == 0 → factor = 1.0 (period_calculated_params.py:299-301).
                      .filter(pl.col("sum_pyr") > 0)
                      .select("d", "inv_factor", "ops_factor"))
    # Periods absent from years_represented (or with R=0) get 1.0
    # (mirrors flextool's writer default of 1.0).
    return (period_lf
              .join(factor_per_d, on="d", how="left")
              .with_columns(inv_factor=pl.col("inv_factor").fill_null(1.0),
                            ops_factor=pl.col("ops_factor").fill_null(1.0)))


# ---------------------------------------------------------------------------
# period_walk_iterator — the shared NPV walk
# ---------------------------------------------------------------------------


def period_walk_iterator(
        source: "InputSource",
        active_solve: str | None,
        ed_lf: pl.LazyFrame,
        period_in_use: list[str],
        period_universe: list[str],
        *,
        bounded: bool,
        life_lf: pl.LazyFrame,
        factor_side: str,
        lineage: pl.DataFrame | None = None,
        commission_lf: pl.LazyFrame | None = None,
        year_rows: pl.DataFrame | None = None,
        ) -> pl.LazyFrame:
    """Δ.5-era public alias delegating to :mod:`._walks`.

    Δ.6 lifted the canonical implementation into ``_walks`` so
    Cluster B's invest-history walk could re-use it.  This wrapper
    preserves the boolean ``bounded`` signature for the existing NPV
    callers while internally dispatching the new
    :class:`._walks.WindowMethod` enum.

    See :func:`._walks.period_walk_iterator` for the canonical
    semantics.  ``bounded`` and ``lineage`` are orthogonal — the bool
    selects the window predicate, ``lineage`` restricts the
    ``(d, d_all)`` pair domain (Slice B; always ``None`` until the
    recourse flag lands, Slice C/D); the keyword is forwarded verbatim, as is
    ``year_rows`` (:func:`solve_year_rows`).
    """
    from flextool.engine.preprocess.params._walks import (
        period_walk_iterator as _walk,
        WindowMethod,
    )
    method = WindowMethod.BOUNDED if bounded else WindowMethod.UNBOUNDED_FORWARD
    return _walk(
        source, active_solve, ed_lf,
        period_in_use, period_universe,
        window_method=method, life_lf=life_lf,
        factor_side=factor_side, lineage=lineage, commission_lf=commission_lf,
        year_rows=year_rows)


# ---------------------------------------------------------------------------
# Helper: per-(e, d) lifetime
# ---------------------------------------------------------------------------


def _ed_lifetime_on_lf(source: "InputSource", ed_lf: pl.LazyFrame,
                       parameter: str = LIFETIME_TECHNICAL,
                       commission: pl.DataFrame | None = None) -> pl.LazyFrame:
    """Lazy ``(e, d, life)`` over the given ``(e, d)`` pairs: ``lifetime_technical`` (``parameter``
    :data:`PAYBACK_TIME`: the payback time, ``lifetime_technical`` for an entity without one) of the period each
    decision comes online (:func:`._period_values.ed_values_online`: with ``commission`` ``[e, d, d_c]``, ``d_c``
    for a decision with a construction lead time, else ``d``), 0 where missing; only nodes and processes have a
    lifetime (the others 0)."""
    cls_lf = _entity_class_lookup_lf(_entity_class_lf(source))
    in_class = pl.col("is_process").fill_null(False) | pl.col("is_node").fill_null(False)
    ed_with_class = ed_lf.select("e", "d").join(cls_lf, on="e", how="left")
    resolved = ed_values_online(source, parameter, ed_with_class, 0.0, commission)
    return (resolved
              .with_columns(life=pl.when(in_class).then(pl.col("value")).otherwise(0.0))
              .select("e", "d", "life"))


def _commission_walk_lf(commission: pl.DataFrame | None) -> pl.LazyFrame | None:
    """``[e, d, yr_c]`` for the walks: the cost window of a decision with a lead time starts where its capacity
    starts to operate."""
    if commission is None or commission.height == 0:
        return None
    return commission.lazy().select(pl.col("e").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), "yr_c")


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


def _entity_invest_set_lf(source: "InputSource",
                              disallowed_methods: frozenset[str],
                              ) -> pl.LazyFrame:
    """Lazy ``[e]`` for the canonical entityInvest / entityDivest set.

    Mirrors flextool's
    ``preprocessing/invest_method_sets.py::write_invest_method_sets``:

        entityInvest = {e : ∃m s.t. (e, m) ∈ entity__invest_method
                                AND m ∉ invest_method_not_allowed}

    ``disallowed_methods`` selects between the invest variant
    (:data:`_INVEST_NOT_ALLOWED`) and the divest variant
    (:data:`_DIVEST_NOT_ALLOWED`).  An entity with no explicit
    ``invest_method`` row is **excluded** (default is ``not_allowed``).
    """
    methods_lf = _entity_method_lf(source, "invest_method")
    return (methods_lf
              .filter(~pl.col("method").is_in(list(disallowed_methods)))
              .select("e")
              .unique())


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
    prices = _per_method_annuity_lf(source, keys, "invest_cost", _INVEST_NOT_ALLOWED, commission,
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
_GENERATION_COLUMNS = ("e", "d_build", "cohort", "d_on", "d_eol", "y_on", "y_eol")


def chain_generations(cohort: pl.DataFrame | None, kind: str = "invest") -> pl.DataFrame:
    """``[e, d_build, cohort, d_on, d_eol, y_on, y_eol]`` (Utf8 labels) of the generations of the ``reinvest_automatic``
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
        _INVEST_NOT_ALLOWED, None, level=True, per_plant=per_plant))
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
                  .filter((pl.col("y") >= pl.col("y_on") - YEAR_TOL) & (pl.col("y") < pl.col("y_eol") - YEAR_TOL)))
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
    fixed cost, :func:`_window_life_to_end_of_life`) at the level yearly price ``a_cap`` of the plant over its real
    lifetime (:func:`_per_method_annuity_lf` with ``level``); ``reinvest_automatic`` → unbounded, each period at the
    price of the generation of the decision's chain in service there (``generations``, :func:`chain_generations`;
    :func:`chain_lumps`), at the annuity for a decision without generation rows.
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
    entity_invest_set = _entity_invest_set_lf(source, _INVEST_NOT_ALLOWED)
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


def _factor_frame(source: "InputSource",
                  active_solve: str | None,
                  period_in_use: list[str],
                  period_universe: list[str],
                  provider: "object | None",
                  factor: str,
                  year_rows: pl.DataFrame | None = None,
                  ) -> pl.LazyFrame:
    """``[d_all, factor]`` (Utf8 key): the solve's per-period discount factor (``ops_factor``: ``p_inflation_op``,
    the factor of the existing fixed cost in the objective; ``inv_factor``: its investment twin)."""
    param = p_inflation_op_from_source(source, active_solve, period_in_use, period_universe, provider=provider,
                                       factor=factor, year_rows=year_rows)
    if param is None:
        return pl.LazyFrame(schema={"d_all": pl.Utf8, "factor": pl.Float64})
    return param.frame.lazy().select(pl.col("d").cast(pl.Utf8).alias("d_all"), pl.col("value").alias("factor"))


def _retirement_discount_lf(source: "InputSource",
                            active_solve: str | None,
                            period_universe: list[str],
                            *,
                            provider: "object | None",
                            year_rows: pl.DataFrame | None) -> pl.LazyFrame:
    """Lazy ``[d, disc]`` (``d`` Utf8): D(d) = (1+i)^−base₀(d), the discount of a one-off cash flow at the start of
    period ``d`` (its first year's ``base``, :func:`_inflation_year_rows_lf`; no offset: a retirement settles when
    the capacity leaves, at the start of the period).  1.0 for a period without year rows."""
    rate = _solve_inflation_scalars(source)[0]
    periods = pl.LazyFrame({"d": [str(d) for d in period_universe]}, schema={"d": pl.Utf8})
    rows = _inflation_year_rows_lf(source, active_solve, period_universe, provider=provider, year_rows=year_rows)
    if rows is None:
        return periods.with_columns(disc=pl.lit(1.0))
    start = rows.group_by("d").agg(pl.col("base").min()).select(pl.col("d").cast(pl.Utf8), "base")
    v = (1.0 / (1.0 + rate)) if rate != -1.0 else 1.0
    return (periods.join(start, on="d", how="left")
                   .with_columns(disc=pl.when(pl.col("base").is_null()).then(1.0).otherwise(v ** pl.col("base")))
                   .select("d", "disc"))


#: The columns of :func:`existing_salvage`'s ``prices``: the cohort price columns of the existing capacity.
_EXISTING_PRICE_COLUMNS = ("r_loan", "a_salv", "nsv_eol")


def _run_inv_factors(source: "InputSource",
                     active_solve: str | None,
                     period_in_use: list[str],
                     period_universe: list[str],
                     period_with_history: list[str],
                     years: dict[str, float],
                     *,
                     provider: "object | None",
                     year_rows: pl.DataFrame | None,
                     run_factor: "Param | None",
                     ) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """``(factors, walk_periods, universe)``: the investment factor ``[d_all, factor]`` (Utf8) of every period of
    the solve's history and periods in use, from the start of the run (``run_factor``, ``p_inflation_inv_run``, where
    it has the period; the solve's own factor otherwise, equal on its periods), the years ``[d_all, y]`` of those
    periods (``years``: :func:`._cohorts.cohort_years`) and the period universe the factors are computed over.  The
    take-back of a salvage credit or a level price sums these factors over the periods a cohort is in service.  A
    period of the history that is neither in use nor a period of the run calendar is no period of the run (the
    branch copy of a realized designation, ``<period>_<branch>``, :func:`run_years`): it holds no years of its own."""
    from flextool.engine.preprocess.params._run_calendar import read_run_calendar
    calendar = read_run_calendar(provider)
    in_use = {str(d) for d in period_in_use}
    history = [str(d) for d in period_with_history
               if calendar is None or str(d) in in_use or str(d) in calendar.years]
    periods = list(dict.fromkeys([*history, *(str(d) for d in period_in_use)]))
    universe = list(dict.fromkeys([*(str(d) for d in period_universe), *periods]))
    factors = (_inflation_factors_lf(source, active_solve, universe, provider=provider, year_rows=year_rows)
               .select(pl.col("d").cast(pl.Utf8).alias("d_all"), pl.col("inv_factor").alias("factor")).collect())
    if run_factor is not None:
        run = run_factor.frame.select(pl.col("d").cast(pl.Utf8).alias("d_all"), pl.col("value").alias("_run"))
        factors = (factors.join(run, on="d_all", how="full", coalesce=True)
                          .select("d_all", factor=pl.coalesce("_run", "factor")))
    walk_periods = pl.DataFrame({"d_all": [d for d in periods if d in years],
                                 "y": [float(years[d]) for d in periods if d in years]},
                                schema={"d_all": pl.Utf8, "y": pl.Float64})
    return factors, walk_periods, universe


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
                     ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """``(divest_salvage, credit, prices)`` of the existing capacity of the entities that may retire (entityDivest):
    the salvage term of the retirement coefficient, the level salvage credit of the objective constant and the
    existing cohort's prices (``divest_salvage_p`` / ``existing_salvage`` OBJ headers)::

        divest_salvage[e,d] = D(d)·S_d − a_x·S̄_x[e,d]        (e ∈ entityDivest, d ∈ period_invest)
        credit[e,d]         = a_x·inflation_inv[d]             (d ∈ period_in_use, y0 ≤ y[d] < y_eol_x)
        a_x  = salvage_credit(S_x, r_x, R_x, off_inv) = S_x·(1+r_x)^−R_x·ann(r_x, R_x)·(1+r_x)^(off_inv − 1)
        R_x  = y_eol_x − y0 when the life ends inside the run horizon (y_end_x ≤ H), else y_end_x − y0; +inf (a_x = 0)
               for reinvest_automatic
        S̄_x[e,d] = Σ_{d′ ∈ history ∪ period_in_use, y0 ≤ y[d′] < y_eol_x, on d's lineage} inflation_inv[d′]
        S_d  = 1000·net_salvage_value[e,d];  S_x = 1000·net_salvage_value[e, d_eol_x] (the last run period's value
               after the last period start, :func:`._cohorts.salvage_periods`);  r_x = discount_rate[e, d0]

    y0 the run start (period d0), y_eol_x / y_end_x the rounded / unrounded end of life of the existing cohort, H
    the end of the run horizon, D(d) :func:`_retirement_discount_lf`, inflation_inv the investment factor from the
    start of the run (``run_factor``, ``p_inflation_inv_run``: the history periods' too; the solve's own factor,
    equal on its periods, where it has none).  The existing capacity's investment is sunk; it holds its salvage value,
    received at its end of life, as a level credit a_x over its service years from the run start (the rounded
    span: the credits sum to its salvage once, never more).  Retiring it in d receives S_d at the start of d and
    gives back every credit of the cohort (history included, along d's lineage of a stochastic fan, each calendar
    period once), so a retired MW earns exactly S_d·D(d).  Rows of ``divest_salvage`` only where S_d or a_x is not
    0; ``credit`` only where a_x is not 0; ``prices`` ``[e, r_loan, a_salv, nsv_eol]`` (r_x, a_x, S_x) of every
    entity of entityDivest with an existing cohort.  All frames Utf8 keyed."""
    from flextool.engine.preprocess._annuity import salvage_credit
    from flextool.engine.preprocess.params._cohorts import (
        REINVEST_AUTOMATIC,
        _run_calendar_of,
        cohort_years,
        existing_cohorts,
        salvage_periods,
    )
    from flextool.engine.preprocess.params._derived import run_start_period
    from flextool.engine.preprocess.params._run_calendar import YEAR_TOL
    empty_ed = pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8, "value": pl.Float64})
    empty_prices = pl.DataFrame(schema={"e": pl.Utf8, **{c: pl.Float64 for c in _EXISTING_PRICE_COLUMNS}})
    divesting = _utf8_ed(_entity_invest_set_lf(source, _DIVEST_NOT_ALLOWED), ("e",)).select("e").unique()
    years = cohort_years(source, active_solve, workdir, provider=provider)
    start = run_start_period(years)
    if divesting.height == 0 or start is None:
        return empty_ed, empty_ed, empty_prices
    y0 = float(years[start])
    horizon = _run_calendar_of(source, active_solve, years, provider).horizon_end
    cohorts = (existing_cohorts(source, active_solve, workdir, provider=provider, years=years)
               .select(pl.col("e").cast(pl.Utf8), "method", "y_end", "y_eol")
               .join(divesting, on="e", how="semi"))
    if cohorts.height == 0:
        return empty_ed, empty_ed, empty_prices
    span = (pl.when(pl.col("method") == REINVEST_AUTOMATIC).then(float("inf"))
              .when(pl.col("y_end").is_null() | pl.col("y_eol").is_null()).then(float("inf"))
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
                    .filter((pl.col("y") >= y0 - YEAR_TOL) & (pl.col("y") < pl.col("y_eol") - YEAR_TOL))
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
    divest = (anchors.join(s_d, on=["e", "d"], how="left")
                     .join(take_back, on=["e", "d"], how="left")
                     .join(disc, on="d", how="left")
                     .with_columns(s_d=pl.col("s_d").fill_null(0.0), taken=pl.col("taken").fill_null(0.0),
                                   disc=pl.col("disc").fill_null(1.0))
                     .filter((pl.col("s_d") != 0.0) | (pl.col("taken") != 0.0))
                     .select("e", "d", value=pl.col("disc") * pl.col("s_d") - pl.col("taken"))
                     .sort("e", "d"))
    return divest, credit.sort("e", "d"), prices


def _payment_rows(source: "InputSource",
                  active_solve: str | None,
                  period_in_use: list[str],
                  universe: list[str],
                  periods: set[str],
                  workdir: Path | None,
                  *,
                  provider: "object | None",
                  year_rows: pl.DataFrame | None) -> pl.DataFrame:
    """``[base, width, contrib]``: one row per represented calendar year of the run (the run calendar's periods and
    widths, :func:`run_years`; the solve's own years ``periods`` when the solve is the run), ``base`` its years from
    the start of the run and ``contrib = width·(1+i)^−(base + width·off_inv)`` the discounted width a yearly payment
    of it has (:func:`_inflation_year_rows_lf`).  The forecast copies of a period share its years: each year once."""
    run = run_years(source, active_solve, list(period_in_use), workdir, provider=provider)
    if run is not None:
        rows = _inflation_year_rows_lf(source, active_solve, sorted(run[0]), year_rows=_run_year_rows(*run))
    else:
        rows = _inflation_year_rows_lf(source, active_solve, universe, provider=provider, year_rows=year_rows)
        if rows is not None:
            rows = rows.filter(pl.col("d").is_in(sorted(periods)))
    if rows is None:
        return pl.DataFrame(schema={"base": pl.Float64, "width": pl.Float64, "contrib": pl.Float64})
    return (rows.select("base", "width", contrib=pl.col("inv_contrib"))
                .sort("base").unique(subset=["base"], keep="first", maintain_order=True).collect())


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
                           ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """``(cost, fixed)`` ``[e, cohort, d, value]`` of the retirement options ``options`` (``cd_divest_built``) of the
    model-built cohorts ``cohorts`` (``cohort_divest_built`` with their prices), CUR/MW (MWh for storage): the
    ``divest_built_cost_{p,n}`` OBJ header::

        cost[e,c,d]  = set[c,d] − (a_cap[c] − a_salv[c])·S̄[c,d]
        set[c,d]     = pay[c]·S[y_on, min(y[d], y_on + n)) + D(d)·(B[c,m] − S_d)
        B[c,m]       = pay[c]·(1+r)^(1−off_inv)·ā(r, n − m) for m < n, else 0;  m = y[d] − y_on
        S̄[c,d]       = Σ_{d′ ∈ history ∪ solve, y_on ≤ y[d′] < y_eol, on d's lineage} inflation_inv_run[d′]
        fixed[e,c,d] = Σ_{d″ ∈ solve, y[d] ≤ y[d″] < y_eol, on d's lineage} 1000·fixed_cost[e,d″]·inflation_op[d″]

    pay, n = n_pay, r = r_loan, a_cap, a_salv, y_on, y_eol the cohort's (``_cohorts.PRICE_COLUMNS``); ā(r, x) =
    (1 − (1+r)^−x)/r (x at r = 0); S[a, b) = Σ σ·contrib over the year rows of the run (:func:`_payment_rows`), σ the
    share of the row inside [a, b); D(d) the discount of the start of d (:func:`_retirement_discount_lf`); S_d =
    1000·net_salvage_value[e, d]; years of the run (:func:`._cohorts.cohort_years`), the lineage pairs of the
    investment walks.  A retired MW pays its contract as it ran: the payments due before d, then at the start of d the
    capital still owed, less its salvage value received then; the level price its years in service were charged is
    taken back (history included: a carried cohort's price was charged by an earlier solve) and the fixed cost from d
    on is saved."""
    from flextool.engine.input._axis_enums import cast_dim
    from flextool.engine.preprocess.params._cohorts import cohort_years
    from flextool.engine.preprocess.params._run_calendar import YEAR_TOL
    from flextool.engine.preprocess.params._walks import _apply_lineage_filter
    empty = pl.DataFrame(schema={"e": schema_dtype(_enums, "e"), "cohort": pl.Utf8, "d": schema_dtype(_enums, "d"),
                                 "value": pl.Float64})
    if options is None or options.height == 0:
        return empty, empty
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
                   .filter((pl.col("y") >= pl.col("y_on") - YEAR_TOL) & (pl.col("y") < pl.col("y_eol") - YEAR_TOL)))
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
                  .with_columns(value=pl.col("pay") * pl.col("paid") + pl.col("disc") * (balance - pl.col("s_d"))
                                - (pl.col("a_cap") - pl.col("a_salv")) * pl.col("taken"))
                  .sort(keys))
    return out.select(*keys, "value"), out.select(*keys, value=pl.col("saved"))


def _ed_fixed_cost_raw_lf(source: "InputSource",
                                ed_lf: pl.LazyFrame,
                                ) -> pl.LazyFrame:
    """Lazy ``(e, d, fc)`` — per-entity per-period fixed cost × 1000.

    Mirrors ``preprocessing/entity_period_calc_params.py:149-156``::

        ed_fixed_cost[e, d] = (1000 if e ∈ node else 0) × pdNode[e, fixed_cost, d]
                             + (1000 if e ∈ process else 0) × pdProcess[e, fixed_cost, d]

    Since unit/connection/node are disjoint, the formula simplifies to:
    fc = 1000 × resolve_pdX(fixed_cost, e, d) when e ∈ node ∪ process,
    else 0.
    """
    cls_lf = _entity_class_lookup_lf(_entity_class_lf(source))
    in_class = (pl.col("is_process").fill_null(False)
                | pl.col("is_node").fill_null(False))
    fc_per = _per_entity_param_lf(source, "fixed_cost")
    ed_with_class = ed_lf.join(cls_lf, on="e", how="left")
    return (_resolve_per_period_lf(fc_per, ed_with_class, fill=0.0)
              .with_columns(
                  fc=pl.when(in_class)
                         .then(pl.col("value") * 1000.0)
                         .otherwise(0.0),
              )
              .select("e", "d", "fc"))


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

    Same window dispatch as :func:`npv_invest_discounted_lf`
    (choice/no_invest → bounded; automatic → unbounded), with
    ops_factor (not inv_factor).  The bounded window ends at the cohort's
    rounded end of life ``y_eol = round_end_of_life(y_on + lifetime_technical,
    end_of_life_method)`` (:func:`_window_life_to_end_of_life`): the fixed cost
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


def _walk_ops_factor(source: "InputSource", active_solve: str | None, period_universe: list[str],
                     year_rows: pl.DataFrame | None) -> pl.LazyFrame:
    """``[d_all, factor]``: the operations factor the walks of :func:`period_walk_iterator` sum (the same factors,
    with the boundary's walker Provider of a solve with branch periods)."""
    from flextool.engine.preprocess.params._derived import _fan_walk_provider
    return (_inflation_factors_lf(source, active_solve, period_universe, provider=_fan_walk_provider(),
                                  year_rows=year_rows)
              .select(pl.col("d").alias("d_all"), pl.col("ops_factor").alias("factor")))


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


def _window_life_to_end_of_life(source: "InputSource",
                                 active_solve: str | None,
                                 life_lf: pl.LazyFrame,
                                 commission: pl.DataFrame | None,
                                 *,
                                 provider: "object | None",
                                 keep_eol: bool = False) -> pl.LazyFrame:
    """``life_lf`` ``[e, d, life]`` (the lifetime of decision ``(e, d)``, read at the period it comes online) with
    ``life`` the years from the start of operation ``y_on`` to the rounded end of life of its cohort, so that the
    bounded walk ``yr[d′] < yr_avail + life`` keeps the periods ``y_on ≤ yr[d′] < y_eol``::

        y_on  = y[c(d)]  (years of the run, :func:`._cohorts.cohort_years`; c(d) = d without a lead time)
        y_eol = round_end_of_life(y_on + life, end_of_life_method[e])   (:func:`._cohorts.end_of_life_years`)
        life  = y_eol − y_on − YEAR_TOL / 2

    The half tolerance keeps the period starting at ``y_eol`` out whatever the float rounding of
    ``y_on + (y_eol − y_on)`` (period starts lie more than ``YEAR_TOL`` apart).  Under ``next_seam`` the walk keeps
    the same periods as with the raw lifetime.  A decision whose start of operation has no year keeps its
    lifetime.  ``keep_eol`` also keeps the column ``y_eol`` (null where the start of operation has no year)."""
    from flextool.engine.preprocess.params._cohorts import (
        cohort_years,
        end_of_life_years,
        eol_methods_of,
        existing_cohorts,
        run_cohort_calendar,
    )
    from flextool.engine.preprocess.params._run_calendar import YEAR_TOL
    years = cohort_years(source, active_solve, None, provider=provider)
    if not years:
        return life_lf.with_columns(y_eol=pl.lit(None, dtype=pl.Float64)) if keep_eol else life_lf
    cal = run_cohort_calendar(source, active_solve, years, provider)
    eol = eol_methods_of(existing_cohorts(source, active_solve, provider=provider, years=years))
    life = life_lf.collect()
    starts: dict[tuple[str, str], str] = {}
    if commission is not None and commission.height > 0:
        starts = {(str(e), str(d)): str(d_c)
                  for e, d, d_c in commission.select("e", "d", "d_c").iter_rows() if d_c is not None}
    y_on = [years.get(starts.get((str(e), str(d)), str(d))) for e, d in life.select("e", "d").iter_rows()]
    life = life.with_columns(y_on=pl.Series("y_on", y_on, dtype=pl.Float64))
    y_eol = end_of_life_years(life, cal, eol)
    out = (life.with_columns(y_eol=y_eol)
               .with_columns(life=pl.when(pl.col("y_on").is_null()).then(pl.col("life"))
                                    .otherwise(pl.col("y_eol") - pl.col("y_on") - YEAR_TOL / 2),
                             y_eol=pl.when(pl.col("y_on").is_null()).then(None).otherwise(pl.col("y_eol")))
               .drop("y_on"))
    return (out if keep_eol else out.drop("y_eol")).lazy()


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
    entity_divest_set = _entity_invest_set_lf(source, _DIVEST_NOT_ALLOWED)
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


# ---------------------------------------------------------------------------
# Per-entity-class helpers (lazy)
# ---------------------------------------------------------------------------


def _all_entities_lf(source: "InputSource") -> pl.LazyFrame:
    """Lazy ``[e]`` frame: union of unit + node + connection."""
    return _entity_class_lf(source).select("e").unique()


# ---------------------------------------------------------------------------
# Public Param-producing entry points
# ---------------------------------------------------------------------------


def p_inflation_op_from_source(source: "InputSource",
                                    active_solve: str | None,
                                    period_in_use: list[str],
                                    period_universe: list[str] | None = None,
                                    *,
                                    provider: "object | None" = None,
                                    factor: str = "ops_factor",
                                    year_rows: pl.DataFrame | None = None,
                                    ) -> "Param | None":
    """Public entry: ``p_inflation_op[d]`` over ``period_in_use``.

    ``factor="inv_factor"`` gives the investment-side factor over the same periods instead.  ``year_rows``
    (:func:`solve_year_rows`): the years from the start of the run, when the solve's own count does not give them.

    Lazy port of
    :func:`._derived.p_inflation_op_full_cascade_from_source`.
    Replaces the workdir read of ``solve_data/p_inflation_factor_operations_yearly.csv``
    when wired through :func:`apply_npv`.

    ``provider`` — forwarded to :func:`_inflation_factors_lf` /
    :func:`_years_for_period_lf` so a stochastic branch period reads its
    anchor's ``years_represented`` rows from the canonical
    ``solve_data/p_years_represented.csv`` (the same arm the recourse
    annuity walk in ``._walks.period_walk_iterator`` already uses via
    this same ``_inflation_factors_lf`` helper).  Without it every
    branch period falls back to the source arm's un-copied rows and is
    under-weighted relative to its anchor (#51).  This is a direct
    per-``d`` lookup (no window sum over ``d_all``), so there is no
    double-counting to guard against here — unlike the annuity walk,
    which already applies its own scenario-lineage dedup before
    aggregating the same factor over a lifetime window.
    """
    if not period_in_use:
        return None
    if period_universe is None:
        period_universe = period_in_use
    factors_lf = _inflation_factors_lf(source, active_solve, period_universe,
                                        provider=provider, year_rows=year_rows)
    out = (pl.LazyFrame({"d": period_in_use})
              .with_columns(alias_to_axis("d", "d"))
              .join(factors_lf, on="d", how="left")
              .with_columns(
                  # Unreachable on the solve path: ``_inflation_factors_lf`` gives every period of
                  # ``period_universe`` a factor (1.0 where years_represented has none), and the periods in use
                  # are periods of that universe.  The fill only keeps a caller's universe that misses a period
                  # in use from leaving it a null factor (build_flextool checks the factor covers dt).
                  value=pl.col(factor).fill_null(1.0),
              )
              .select("d", "value")
              .sort("d")
              .collect())
    if out.height == 0:
        return None
    return Param(("d",), out)


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
    ``solve_data/ed_entity_annual_discounted`` frame, :func:`_register_invest_lumps`).

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
    :func:`ed_entity_annual_discounted_from_source`.

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
    :func:`ed_entity_annual_discounted_from_source`.

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


# ---------------------------------------------------------------------------
# Run-level fields of the reported costs (outputs only; no model term reads them)
# ---------------------------------------------------------------------------
#
# A solve discounts its own costs from the start of the run (:func:`solve_year_rows`), but its lumps cover only
# its own horizon: an investment is paid over the solve's periods, not over the periods of the whole run.  The
# reported costs use the fields below, discounted from the start of the run and summed over the periods of the
# whole run; on the solve's own periods the factors equal the solve's own (p_inflation_op).
#
# OUT. p_inflation_op_run[d] = Σ_y pyr[d,y]·(1+r)^-(yr_run[d] + Σ_{y'<y} pyr[d,y'] + pyr[d,y]·offset_ops)
#      p_inflation_inv_run[d] = the same with offset_inv
#   Discount factors of a period from the start of the run, over the solve's periods and the other run periods.
#   yr_run, pyr: the solve's own periods from its period_with_history and years_represented, the other run
#   periods from the run calendar.
# OUT. edd_operating_window[e,d_invest,d] — run periods d in which the capacity of decision (e, d_invest)
#   operates: yr_run[d] ≥ yr_run[c(d_invest)], and < y_eol = round_end_of_life(yr_run[c(d_invest)] +
#   lifetime_technical[e, c(d_invest)], end_of_life_method[e]) for reinvest_choice / no_reinvestment (unbounded
#   for reinvest_automatic): the periods its cohort is in service; c(d) is the period the capacity comes online (d
#   without a construction lead time), whose values describe the plant as constructed.
#   edd_retire_window[e,d_divest,d]: yr_run[d_divest] ≤ yr_run[d] < EOL[e] (the rounded end of life y_eol of the
#   pre-existing capacity, from the start of the run; the end of the run for reinvest_automatic),
#   the window of _cohorts.retire_active.  In a stochastic fan only the pairs of the walk lineage (each calendar
#   period once).
# OUT. edd_annuity[e,d_invest,d] = the price per year in service (CUR/MW/yr, undiscounted) of the plant of the vintage
#   (e, d_invest) in service in d, over its operating window: a_cap[e,c(d_invest)] − a_salv[e,c(d_invest)] for
#   reinvest_choice / no_reinvestment (the level yearly cost of the plant over its lifetime_technical less its level
#   salvage credit, the objective's price, _level_prices_lf); for reinvest_automatic a_b[g] of the generation g in
#   service in d (each replacement at its own price, priced_chain_rows; the annuity of a decision without
#   generations).
#      edd_entity_annual_discounted_run[e,d_invest,d] = Σ over the window periods d' of the plants of the vintage
#   that come online in d of price[d']·p_inflation_inv_run[d']: the investment lump split by the period each plant
#   comes online (c(d_invest), and each generation of a chain at its own online period); the reported investment
#   cost when it happens.
#   ed_divest_settlement_yearly[e,d] = ed_divest_salvage[e,d] / p_inflation_inv[d]: the salvage settlement of a MW
#   retired in d (the salvage value received at the start of d less the salvage credits given back, the objective's
#   term) as a yearly amount of the retirement period d.
# OUT. ed_entity_annual_discounted_run[e,d] = Σ_{kind} price_kind[e,d]·Σ_{(e,d,d') ∈ edd_operating_window(kind)}
#        p_inflation_inv_run[d']
#   price_bounded = a_cap − a_salv, price_unbounded = the price of the generation in service in d' (the annuity
#   without generations): the objective's investment lump over the run periods.
#      ed_lifetime_fixed_cost_run[e,d] = Σ_{(e,d,d') ∈ edd_operating_window} fc[e,d']·p_inflation_op_run[d']
#        (summed as fc[e,c(d)]·Σ factor + Σ (fc[e,d'] − fc[e,c(d)])·factor, the objective's form)
#      ed_lifetime_fixed_cost_divest_run[e,d] = Σ_{(e,d,d') ∈ edd_retire_window} fc[e,d']·p_inflation_op_run[d']
#   The fixed costs are those of each window period with the operations factor (as the fixed cost of existing
#   capacity is charged), as their per-solve twins.  The salvage settlement of a retirement is reported as the
#   objective charges it (ed_divest_salvage, discounted from the start of the run): a one-off at the start of the
#   retirement period.
# Notes: when the run is the solve itself (no run calendar, or every run period is in use and the solve starts
#   the run) the factors and lumps ARE the per-solve fields (byte-identical); only the windows and annuities are
#   built.  An entity with both a bounded and an unbounded lifetime method adds both sums, as the per-solve lumps
#   do; edd_operating_window then holds the union of the two windows.



def _period_history_years(workdir: Path | None, provider: "object | None") -> dict[str, float] | None:
    """``{period: years from the start of the run}`` of the solve's ``period_with_history`` (history first, then the
    solve's own periods; a branch period carries its anchor's year); ``None`` when the solve has none."""
    from flextool.engine.preprocess.params._derived import _provider_has_key, _provider_read
    if workdir is None:
        return None
    path = Path(workdir) / "solve_data" / "period_with_history.csv"
    if not _provider_has_key(provider, path):
        return None
    df = _provider_read(provider, path)
    if df.height == 0 or "period" not in df.columns:
        return None
    col = "param" if "param" in df.columns else df.columns[-1]
    out: dict[str, float] = {}
    for d, y in df.select(pl.col("period").cast(pl.Utf8),
                          pl.col(col).cast(pl.Float64, strict=False)).iter_rows():
        if d is not None and y is not None:
            out.setdefault(str(d), float(y))
    return out or None


def _split_years(represented: float) -> list[float]:
    """Year widths of a period that represents ``represented`` years: ⌈R⌉ years of min(1, remaining), as the
    solve's own ``years_represented`` is split (:func:`._derived._years_for_period_from_source`)."""
    import math
    out: list[float] = []
    remaining = float(represented)
    for _ in range(int(math.ceil(represented)) if represented > 0 else 0):
        width = min(1.0, remaining)
        out.append(width)
        remaining -= width
    return out


def run_years(source: "InputSource",
              active_solve: str | None,
              period_in_use: list[str],
              workdir: Path | None,
              *,
              provider: "object | None" = None,
              ) -> tuple[dict[str, float], dict[str, list[float]]] | None:
    """``(yr, widths)`` of the periods the reported costs of the solve cover, counted from the start of the run:
    ``yr[d]`` the start of ``d``, ``widths[d]`` its represented years in order.  The solve's own periods (its
    ``period_with_history``) take their year from it and their widths from the solve's ``years_represented``;
    the other periods of the run calendar take both from the calendar.

    ``None`` when the solve's own fields are the run fields: no run calendar or no period history, or every run
    period is in use by the solve and the solve starts the run."""
    from flextool.engine.preprocess.params._derived import _years_for_period_from_source
    from flextool.engine.preprocess.params._run_calendar import YEAR_TOL, read_run_calendar
    calendar = read_run_calendar(provider)
    history = _period_history_years(workdir, provider)
    if calendar is None or history is None:
        return None
    in_use = [str(d) for d in period_in_use]
    # A period of the history that is neither in use nor a period of the run calendar is no period of the run:
    # the branch copy of a realized designation (``<period>_<branch>``), whose operations the realization solve
    # owns, holds no years of its own.
    history = {d: y for d, y in history.items() if d in in_use or d in calendar.years}
    outside = [d for d in calendar.years if d not in history and d not in in_use]
    starts = [history[d] for d in in_use if d in history]
    if (not outside and set(history) <= set(in_use) and starts
            and abs(min(starts)) <= YEAR_TOL):
        return None
    yr: dict[str, float] = {}
    for d in [*in_use, *history, *outside]:
        if d in yr:
            continue
        if d in history:
            yr[d] = history[d]
        elif d in calendar.years:
            yr[d] = calendar.years[d]
        else:
            raise ValueError(
                f"Period '{d}' of solve '{active_solve}' has no year from the start of the run (neither in the "
                "solve's period history nor in the run calendar); its costs cannot be discounted.")
    own = _years_for_period_from_source(source, active_solve, in_use, provider=provider)
    widths: dict[str, list[float]] = {}
    for d in yr:
        if d in in_use:
            widths[d] = [float(w) for _, w in own.get(d, [])]
        else:
            widths[d] = _split_years(calendar.years_represented.get(d, 0.0))
    return yr, widths


def _run_year_rows(yr: dict[str, float], widths: dict[str, list[float]]) -> pl.DataFrame:
    """``[d, width, base]``: one row per represented year, ``base`` its years from the start of the run."""
    rows: list[tuple[str, float, float]] = []
    for d, start in yr.items():
        cum = 0.0
        for w in widths.get(d, []):
            rows.append((d, w, start + cum))
            cum += w
    return pl.DataFrame(rows, schema={"d": pl.Utf8, "width": pl.Float64, "base": pl.Float64}, orient="row")


# ---------------------------------------------------------------------------
# The discount base of a solve's own factors (the objective's)
# ---------------------------------------------------------------------------
#
# EQ. base[d,y] = pwh[d] + Σ_{y' of d before y} pyr[d,y']
#     F_inv(d) = Σ_y pyr[d,y]·(1+r)^-(base[d,y] + pyr[d,y]·offset_inv)
#     F_ops(d) = Σ_y pyr[d,y]·(1+r)^-(base[d,y] + pyr[d,y]·offset_ops)
#   Every solve (a real solve, a roll, a nested ``<solve>_<period>`` sub-solve) discounts its investments and its
#   operations from the start of the run, as one solve over the same periods does.  pwh[d] is the start of
#   period d from the start of the run (``period_with_history``: the earlier solves' periods first, a branch period
#   at its anchor's year); pyr the widths of the (complete) solve's own ``years_represented`` rows
#   (``p_years_represented``).  Where an earlier solve gave a period a different width, the history gives each
#   period's start and the solve's own map its widths (the run calendar reports the conflict).
# Notes: the solve's own count (``years_from_solve``, from 0 at the solve's first period, each global year label
#   one year wide where the period has no row) gives the same base for a solve that starts the run with whole-year
#   widths; such a solve keeps that count (byte-identical LP).  It differs for a solve that starts after the run
#   start (its own count restarts at 0), and for fractional years (item 259: a period of 2.5 years put the next
#   period at year 3, not 2.5).  The investment walk of a deterministic solve counts from the database map over
#   periodAll (``_years_for_period_from_source``), so a whole map counted investments from the run start and
#   operations from the solve start; both now take ``base``.


def _years_by_label_bases(years: dict[str, list[tuple[str, float]]]) -> dict[str, list[float]]:
    """``{d: [base of each year]}`` of the solve's own count: over the global set of year labels, the sum of the
    period's widths before the year, 1 for a label the period has no row of (:func:`_inflation_factors_lf` without
    ``year_rows``, ``emit/_period_calc``)."""
    labels: dict[str, None] = {}
    for rows in years.values():
        for y, _ in rows:
            labels.setdefault(y, None)
    try:
        ordered = sorted(labels, key=float)
    except ValueError:
        ordered = sorted(labels)
    out: dict[str, list[float]] = {}
    for d, rows in years.items():
        width = dict(rows)
        pos: dict[str, float] = {}
        cum = 0.0
        for y in ordered:
            pos[y] = cum
            cum += width.get(y, 1.0)
        out[d] = [pos[y] for y, _ in rows]
    return out


def _sorted_years(rows: list[tuple[str, float]]) -> list[tuple[str, float]]:
    try:
        return sorted(rows, key=lambda r: float(r[0]))
    except ValueError:
        return sorted(rows, key=lambda r: r[0])


def year_rows_from_history(history: dict[str, float] | None,
                           years: dict[str, list[tuple[str, float]]],
                           *,
                           solve: str | None = None,
                           also_counted: dict[str, list[tuple[str, float]]] | None = None,
                           ) -> pl.DataFrame | None:
    """``[d, y, width, base]``: the represented years of the solve (``years``: ``{d: [(year label, width)]}``,
    its ``p_years_represented`` rows) with ``base`` their years from the start of the run (the ``EQ.`` block
    above), for :func:`_inflation_factors_lf` (``year_rows``) and ``emit/_period_calc``.

    ``None`` when the solve's own count already gives every year that base (a solve that starts the run with
    whole-year widths): the solve keeps that count.  ``also_counted`` is another count of the same solve (the
    database map over periodAll, which the deterministic investment walk reads); a base it gives differently
    also takes the rows.  ``None`` without a period history.  A period with years but no history is an error
    only when the rows are needed (the gate over the periods with history fails): otherwise the solve keeps its
    own count, as before."""
    from flextool.engine.preprocess.params._run_calendar import YEAR_TOL
    if not history or not years:
        return None
    own = {d: _sorted_years(rows) for d, rows in years.items() if rows}
    missing = [d for d in own if d not in history]
    rows_out: list[tuple[str, str, float, float]] = []
    new_base: dict[str, list[float]] = {}
    for d, rows in own.items():
        if d not in history:
            continue
        cum = 0.0
        bases: list[float] = []
        for y, w in rows:
            bases.append(history[d] + cum)
            rows_out.append((d, y, float(w), history[d] + cum))
            cum += w
        new_base[d] = bases
    counts = [_years_by_label_bases(own)]
    if also_counted:
        other = {d: _sorted_years(rows) for d, rows in also_counted.items() if rows}
        other_bases = _years_by_label_bases(other)
        # Only the periods the map has years of (a branch period has none there: a fan's walk reads the
        # solve's own rows).
        counts.append({d: other_bases[d] for d in own if d in other_bases})
    differs = any(len(count[d]) != len(new_base[d])
                  or any(abs(a - b) > YEAR_TOL for a, b in zip(count[d], new_base[d]))
                  for count in counts for d in count if d in new_base)
    if not differs:
        return None
    if missing:
        raise ValueError(
            f"Period '{missing[0]}' of solve '{solve}' has represented years but no year from the start of the run "
            "(it is not in the solve's period history); its costs cannot be discounted.")
    return pl.DataFrame(rows_out, schema={"d": pl.Utf8, "y": pl.Utf8, "width": pl.Float64, "base": pl.Float64},
                        orient="row")


def notice_discount_from_run_start(solve: str | None, rate: float) -> None:
    """The behaviour-change notice of a solve whose own count of years does not give its years from the start of
    the run (``year_rows`` not None), when inflation discounts its costs: its objective moved to the run's base.
    A roll is named by its rolling solve."""
    import re

    from flextool.engine import _behaviour_changes
    if solve is None or rate == 0.0:
        return
    _behaviour_changes.notice("chained_solve_discount_from_run_start", "solves",
                              [re.sub(r"_roll_\d+$", "", str(solve))])


def _represented_years(provider: "object | None") -> dict[str, list[tuple[str, float]]]:
    """``{d: [(year label, width)]}`` of the solve's ``solve_data/p_years_represented.csv`` (``{}`` without)."""
    from flextool.engine.preprocess.params._derived import _provider_has_key, _provider_read
    path = Path("solve_data") / "p_years_represented.csv"
    if provider is None or not _provider_has_key(provider, path):
        return {}
    df = _provider_read(provider, path)
    if df.height == 0 or not {"period", "years_from_solve", "p_years_represented"} <= set(df.columns):
        return {}
    out: dict[str, list[tuple[str, float]]] = {}
    for d, y, w in df.select(pl.col("period").cast(pl.Utf8), pl.col("years_from_solve").cast(pl.Utf8),
                             pl.col("p_years_represented").cast(pl.Float64, strict=False)).iter_rows():
        if d is not None and y is not None and w is not None:
            out.setdefault(str(d), []).append((str(y), float(w)))
    return out


def solve_year_rows(workdir: Path | None,
                    provider: "object | None",
                    source: "InputSource | None" = None,
                    active_solve: str | None = None,
                    period_universe: list[str] | None = None,
                    ) -> pl.DataFrame | None:
    """:func:`year_rows_from_history` of the solve of ``workdir``: its period history and its represented years;
    with ``source`` the database map over ``period_universe`` (the deterministic investment walk's count) is
    compared too.  ``None``: the solve keeps its own count."""
    history = _period_history_years(workdir, provider)
    years = _represented_years(provider)
    if not history or not years:
        return None
    also = None
    if source is not None and period_universe:
        from flextool.engine.preprocess.params._derived import _years_for_period_from_source
        also = _years_for_period_from_source(source, active_solve, list(period_universe))
    return year_rows_from_history(history, years, solve=active_solve, also_counted=also)


def _utf8_ed(lf: pl.LazyFrame, cols: tuple[str, ...] = ("e", "d")) -> pl.DataFrame:
    return lf.with_columns(pl.col(c).cast(pl.Utf8) for c in cols).collect()


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


def _ed_param(df: pl.DataFrame) -> "Param | None":
    if df.height == 0:
        return None
    return Param(("e", "d"), df.lazy().select(alias_to_axis("e", "e"), alias_to_axis("d", "d"), "value")
                                     .sort("e", "d").collect())


def _window_frame(walk: pl.DataFrame, anchor_name: str) -> pl.DataFrame | None:
    out = (walk.select("e", pl.col("d").alias(anchor_name), pl.col("d_all").alias("d")).unique()
               .lazy().select(alias_to_axis("e", "e"), alias_to_axis(anchor_name, anchor_name),
                              alias_to_axis("d", "d"))
               .sort("e", anchor_name, "d").collect())
    return out if out.height > 0 else None


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


def _fixed_lump(walk: pl.DataFrame, fc_on: pl.DataFrame, factor: pl.DataFrame, anchors: pl.DataFrame,
                source: "InputSource") -> pl.DataFrame:
    """``[e, d, value]`` over ``anchors``: Σ over the window ``walk`` (``[e, d, kind, d_all]``) of fc[e, d_all] ×
    factor[d_all], summed as ``fc_on[e, d] × Σ factor + Σ (fc[e, d_all] − fc_on[e, d]) × factor[d_all]`` (``fc_on``
    ``[e, d, coef]``, the fixed cost of the online period; the lifetime fixed cost of the objective,
    :func:`lifetime_fixed_cost_invest_lf`)."""
    base = _lump(walk, fc_on, factor, anchors)
    fc_all = _utf8_ed(_ed_fixed_cost_raw_lf(
        source, walk.lazy().select(alias_to_axis("e", "e"), alias_to_axis("d_all", "d")).unique()))
    corr = (walk.join(fc_all.rename({"d": "d_all"}), on=["e", "d_all"], how="left")
                .join(fc_on, on=["e", "d"], how="left")
                .join(factor, on="d_all", how="left")
                .group_by("e", "d")
                .agg(((pl.col("fc").fill_null(0.0) - pl.col("coef").fill_null(0.0))
                      * pl.col("factor").fill_null(1.0)).sum().alias("corr")))
    return (base.join(corr, on=["e", "d"], how="left")
                .select("e", "d", value=pl.col("value") + pl.col("corr").fill_null(0.0)))


def apply_run_cost_fields(flex_data: object,
                          source: "InputSource",
                          active_solve: str | None,
                          period_in_use: list[str],
                          period_invest: list[str],
                          period_with_history: list[str],
                          workdir: Path | None,
                          *,
                          lineage: pl.DataFrame | None,
                          years: tuple[dict[str, float], dict[str, list[float]]] | None,
                          provider: "object | None" = None,
                          ) -> None:
    """Set the run-level invest / divest fields of the reported costs (see the ``OUT.`` block above) on
    ``flex_data``; the factors (``p_inflation_*_run``) must be set already.  ``years`` is :func:`run_years`
    (``None``: the lumps are the per-solve ones, and the windows walk the solve's own years)."""
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
    factor_ops = factor_of("p_inflation_op_run")

    # Investment side: vintages over the operating window from their start of operation.
    inv = anchors_of("ed_invest_set", _INVEST_NOT_ALLOWED, period_invest)
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
    # The lifetime fixed cost covers every entity over the period history (as its per-solve twin); without run
    # years only the vintages are walked (for their operating window).  The history periods are those of the run
    # (:func:`run_years`): the branch copy of a realized designation (``<period>_<branch>``) is no period of the
    # run -- no decision is made at it and its operations are reported by the realization solve, at the year of
    # its real period.
    fc_pairs = inv if years is None else pl.concat([
        _utf8_ed(_all_entities_lf(source), ("e",)).select("e").unique().join(
            pl.DataFrame({"d": [str(d) for d in period_with_history if str(d) in yr]}, schema={"d": pl.Utf8}),
            how="cross"),
        inv]).unique().sort("e", "d")
    if fc_pairs.height > 0:
        fc_lf = _anchor_lf(fc_pairs)
        # The lifetime of the period the capacity comes online (the cohort's end of life).
        life = _utf8_ed(_ed_lifetime_on_lf(source, fc_lf, LIFETIME_TECHNICAL, commission))
        anchors = (year_of_start(fc_pairs, True)
                     .join(life, on=["e", "d"], how="left")
                     .with_columns(life=pl.col("life").fill_null(0.0))
                     .join(_lifetime_kinds(source, fc_pairs.select("e").unique()), on="e", how="inner"))
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
        flex_data.edd_operating_window = _window_frame(walk.join(vintages, on=["e", "d"], how="semi"), "d_invest")
        chain_rows = None
        if inv.height > 0:
            chain_rows = _run_chain_rows(source, active_solve, walk.join(inv, on=["e", "d"], how="semi"),
                                         factor_inv, yr, flex_data, provider)
            flex_data.edd_annuity, flex_data.edd_entity_annual_discounted_run = _vintage_yearly_and_starts(
                walk.join(vintages, on=["e", "d"], how="semi"), price, chain_rows, factor_inv, start)
        if years is not None:
            fc = _utf8_ed(at_online_period(
                fc_lf, lambda lf: _ed_fixed_cost_raw_lf(source, lf), commission, "fc")).rename({"fc": "coef"})
            flex_data.ed_lifetime_fixed_cost_run = _ed_param(_fixed_lump(walk, fc, factor_ops, fc_pairs, source))
            if inv.height > 0:
                # The price over the periods in service, as the objective's investment lump: a reinvest_automatic
                # chain at the price of the generation in service in each period (single price: the same product).
                inv_walk = walk.join(inv, on=["e", "d"], how="semi")
                chain = _lumps_of(chain_rows)
                single = chain.filter(pl.col("single")).select("e", "d", kind=pl.lit("unbounded"), _a_b="a_b")
                price = (price.join(single, on=["e", "d", "kind"], how="left")
                              .with_columns(coef=pl.coalesce("_a_b", "coef")).drop("_a_b"))
                multi = chain.filter(~pl.col("single")).select("e", "d", _chain="value")
                rest = inv_walk.join(multi.select("e", "d").with_columns(kind=pl.lit("unbounded")),
                                     on=["e", "d", "kind"], how="anti")
                lump = _lump(rest, price, factor_inv, inv)
                lump = (lump.join(multi, on=["e", "d"], how="left")
                            .with_columns(value=pl.col("value") + pl.col("_chain").fill_null(0.0)).drop("_chain"))
                flex_data.ed_entity_annual_discounted_run = _ed_param(lump)

    # Retirement side: from the retirement period until the rounded end of life ``y_eol`` of the existing capacity
    # (the end of the run for reinvest_automatic), the window of _cohorts.retire_active on the run calendar.
    div = anchors_of("ed_divest_set", _DIVEST_NOT_ALLOWED, period_invest)
    if div.height > 0:
        from flextool.engine.preprocess.params._cohorts import existing_cohorts, existing_ends
        from flextool.engine.preprocess.params._run_calendar import YEAR_TOL
        anchors = (year_of_start(div, False)
                     .with_columns(life=pl.lit(0.0), kind=pl.lit("unbounded")))
        walk = _run_window(anchors, yr, lineage, period_in_use)
        eol = existing_ends(existing_cohorts(source, active_solve, workdir, provider=provider))
        if eol:
            ends = pl.DataFrame({"e": list(eol), "end": [v.y_eol for v in eol.values()]},
                                schema={"e": pl.Utf8, "end": pl.Float64})
            periods = pl.DataFrame({"d_all": list(yr), "y": [float(v) for v in yr.values()]},
                                   schema={"d_all": pl.Utf8, "y": pl.Float64})
            expired = (ends.join(periods, how="cross")
                           .filter(pl.col("y") >= pl.col("end") - YEAR_TOL).select("e", "d_all"))
            walk = walk.join(expired, on=["e", "d_all"], how="anti")
        flex_data.edd_retire_window = _window_frame(walk, "d_divest")
        if years is not None:
            fc_all = _utf8_ed(_ed_fixed_cost_raw_lf(
                source, walk.lazy().select(alias_to_axis("e", "e"), alias_to_axis("d_all", "d")).unique()))
            saved = (walk.join(fc_all.rename({"d": "d_all"}), on=["e", "d_all"], how="left")
                         .join(factor_ops, on="d_all", how="left")
                         .group_by("e", "d")
                         .agg((pl.col("fc").fill_null(0.0) * pl.col("factor").fill_null(0.0)).sum().alias("value")))
            flex_data.ed_lifetime_fixed_cost_divest_run = _ed_param(
                div.join(saved, on=["e", "d"], how="left").select("e", "d", value=pl.col("value").fill_null(0.0)))

    if years is None:
        _run_lumps_are_the_solves(flex_data)


def _run_chain_rows(source: "InputSource",
                    active_solve: str | None,
                    walk: pl.DataFrame,
                    factor: pl.DataFrame,
                    yr: dict[str, float],
                    flex_data: object,
                    provider: "object | None") -> pl.DataFrame | None:
    """:func:`priced_chain_rows` of the investment decisions of the solve's cohort table over the run's operating
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


def _vintage_yearly_and_starts(walk: pl.DataFrame,
                               price: pl.DataFrame,
                               chain_rows: pl.DataFrame | None,
                               factor: pl.DataFrame,
                               start: pl.DataFrame) -> tuple["Param | None", "Param | None"]:
    """``(edd_annuity, edd_entity_annual_discounted_run)`` (``[e, d_invest, d, value]``) of the vintages of
    ``walk`` (``[e, d, kind, d_all]``, their run operating window): the price per year in service of the plant in
    service in each period ``d`` (``price`` ``[e, d, kind, coef]``; a reinvest_automatic chain at its
    generation's ``a_b``, ``chain_rows`` of :func:`priced_chain_rows`), and the investment lump split by the period
    each plant comes online (``d`` the start of operation ``c(d_invest)``, ``start`` ``[e, d, d_c]``; a generation
    of a chain at its own online period), Σ price·factor over its periods (``factor`` ``[d_all, factor]``)."""
    chained = (pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8}) if chain_rows is None
               else chain_rows.select("e", "d").unique())
    plain = (walk.join(chained.with_columns(kind=pl.lit("unbounded")), on=["e", "d", "kind"], how="anti")
                 .join(price, on=["e", "d", "kind"], how="left")
                 .join(factor, on="d_all", how="left")
                 .join(start, on=["e", "d"], how="left")
                 .select("e", "d", "d_all", coef=pl.col("coef").fill_null(0.0),
                         factor=pl.col("factor").fill_null(1.0), d_start=pl.coalesce("d_c", "d")))
    rows = plain if chain_rows is None else pl.concat([
        plain, chain_rows.select("e", "d", "d_all", coef=pl.col("a_b"), factor=pl.col("factor"),
                                 d_start=pl.col("d_price"))])
    yearly = (rows.group_by("e", "d", "d_all").agg(pl.col("coef").sum().alias("value"))
                  .select("e", pl.col("d").alias("d_invest"), pl.col("d_all").alias("d"), "value"))
    starts = (rows.sort("e", "d", "d_all")
                  .group_by("e", "d", "d_start", maintain_order=True)
                  .agg((pl.col("coef") * pl.col("factor")).sum().alias("value"))
                  .select("e", pl.col("d").alias("d_invest"), pl.col("d_start").alias("d"), "value"))
    return _edd_param(yearly), _edd_param(starts)


def _edd_param(df: pl.DataFrame) -> "Param | None":
    if df.height == 0:
        return None
    return Param(("e", "d_invest", "d"),
                 df.lazy().select(alias_to_axis("e", "e"), alias_to_axis("d_invest", "d_invest"),
                                  alias_to_axis("d", "d"), "value").sort("e", "d_invest", "d").collect())


def _run_lumps_are_the_solves(flex_data: object) -> None:
    """The run is the solve: the run lumps are the solve's own lumps."""
    for name in ("ed_entity_annual_discounted", "ed_lifetime_fixed_cost", "ed_lifetime_fixed_cost_divest"):
        setattr(flex_data, f"{name}_run", getattr(flex_data, name, None))


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


def apply_run_fields_of_synthetic_solve(flex_data: object,
                                        source: "InputSource",
                                        workdir: Path,
                                        *,
                                        provider: "object | None" = None,
                                        ) -> None:
    """The investment lumps and the run-level fields of a synthetic sub-solve (a roll of a rolling solve, a nested
    ``<solve>_<period>`` sub-solve), for which ``apply_npv`` does not run.

    A sub-solve with an invest or a divest set gets its investment lumps from :func:`apply_invest_lumps` and its
    salvage terms from :func:`apply_retirement_salvage` over its own periods, as a Spine solve does (the CSV emitter derives neither the
    investment lumps nor the salvage terms); its lifetime fixed cost of retirements and its own discount factors
    stay the loaded ``solve_data`` frames.  Without run years the run
    fields are the sub-solve's own fields."""
    from flextool.engine.preprocess.params._branch import annuity_window_lineage_df
    from flextool.engine.preprocess.params._derived import (
        _expand_invest_branch_periods,
        _period_in_use_set,
        _periodAll_from_source,
        _provider_has_key,
        _provider_read,
        _read_active_solve,
        _read_period_with_history,
        _solve_periods,
    )
    active_solve = _read_active_solve(workdir, provider=provider)
    period_in_use = list(_period_in_use_set(source, active_solve, workdir, provider=provider))
    apply_pre_existing_of_reported_costs(flex_data, source, active_solve, period_in_use, workdir, provider=provider)
    years = run_years(source, active_solve, period_in_use, workdir, provider=provider)
    seed_inv = None
    path = Path(workdir) / "solve_data" / "p_inflation_factor_investment_yearly.csv"
    if years is None and _provider_has_key(provider, path):
        df = _provider_read(provider, path)
        if df.height > 0 and "period" in df.columns:
            col = "value" if "value" in df.columns else df.columns[-1]
            seed = (df.lazy()
                      .select(pl.col("period").cast(pl.Utf8).alias("d"),
                              pl.col(col).cast(pl.Float64, strict=False).alias("value"))
                      .filter(pl.col("d").is_in([str(d) for d in period_in_use]))
                      .select(alias_to_axis("d", "d"), "value").sort("d").collect())
            seed_inv = Param(("d",), seed) if seed.height > 0 else None
    apply_run_factors(flex_data, source, active_solve, years, seed_inv)
    ed_invest = getattr(flex_data, "ed_invest_set", None)
    ed_divest = getattr(flex_data, "ed_divest_set", None)
    if ((ed_invest is None or ed_invest.height == 0) and (ed_divest is None or ed_divest.height == 0)) \
            or active_solve is None:
        if years is None:
            _run_lumps_are_the_solves(flex_data)
        return
    pi = getattr(flex_data, "period_invest", None)
    period_invest = [] if pi is None or pi.height == 0 else [str(d) for d in pi[pi.columns[0]].to_list()]
    period_with_history = _read_period_with_history(workdir, provider=provider) or list(period_in_use)
    lineage = annuity_window_lineage_df(workdir, source, active_solve, provider=provider)
    # The decision periods as apply_npv and ed_invest_set read them (the sub-solve's own invest periods).
    decision_periods = _expand_invest_branch_periods(
        _solve_periods(source, active_solve, "invest_periods"), workdir, provider=provider) or []
    period_universe = list(_periodAll_from_source(source, active_solve, workdir=workdir, provider=provider))
    year_rows = solve_year_rows(workdir, provider, source, active_solve, period_universe)
    if year_rows is not None:
        notice_discount_from_run_start(active_solve, _solve_inflation_scalars(source)[0])
    if ed_invest is not None and ed_invest.height > 0:
        apply_invest_lumps(flex_data, source, active_solve, decision_periods, period_in_use, period_universe,
                           period_with_history, lineage=lineage, year_rows=year_rows, provider=provider)
    if ed_divest is not None and ed_divest.height > 0:
        apply_retirement_salvage(flex_data, source, active_solve, decision_periods, period_in_use, period_universe,
                                 period_with_history, lineage=lineage, year_rows=year_rows, workdir=workdir,
                                 provider=provider)
    apply_built_retirement_costs(flex_data, source, active_solve, period_in_use, period_universe,
                                 period_with_history, lineage=lineage, year_rows=year_rows, workdir=workdir,
                                 provider=provider)
    apply_run_cost_fields(flex_data, source, active_solve, period_in_use, period_invest, period_with_history,
                          workdir, lineage=lineage, years=years, provider=provider)


# ---------------------------------------------------------------------------
# Apply boundary
# ---------------------------------------------------------------------------


def apply_npv(flex_data: object,
                  source: "InputSource",
                  workdir: Path,
                  *,
                  provider: "object | None" = None) -> None:
    """Wire the cluster-A NPV helpers into ``flex_data`` (mutates in place).

    Order:
      1. ``p_inflation_op``                 (depends on years_represented)
      2. ``p_ed_fixed_cost``                (depends on period_with_history)
      3. ``ed_entity_annual_discounted``    (depends on inv_factor + ed_invest)
      4. ``ed_divest_salvage`` / ``ed_existing_salvage_credit``
                                            (depends on inv_factor + ed_divest; in apply_invest_lumps)
      5. ``ed_lifetime_fixed_cost``         (depends on ops_factor + period_with_history)
      6. ``ed_lifetime_fixed_cost_divest``  (depends on inv_factor + ed_divest)

    Next to 1 the run-level factors of the reported costs (:func:`apply_run_factors`), after 6 the run-level
    lumps, annuities and windows (:func:`apply_run_cost_fields`); no model term reads them.

    Δ.12b — assignment is unconditional (no silent fall-through to a
    seed value).  Each helper raises on hard errors and returns
    ``None`` only when the upstream feature is genuinely inactive
    (no_invest short-circuit, no period_with_history, etc.) — that
    ``None`` is the explicit "field stays default" signal.
    """
    from flextool.engine.preprocess.params._derived import (
        _read_active_solve, _solve_periods, _period_in_use_set,
        _periodAll_from_source, _read_period_with_history,
        _expand_invest_branch_periods, _enter_recourse_anchor_scope,
        _recourse_branch_axis_present,
    )

    # Slice D §4 — arm the per-period anchor-map + walker-Provider scope for
    # the annuity / lifetime / fixed-cost resolution below (no-op flag-off).
    _enter_recourse_anchor_scope(workdir, provider=provider)
    active_solve = _read_active_solve(workdir, provider=provider)
    # Lineage of the four NPV window walks: each calendar period counted once along a lineage, under every
    # invest method (recourse: the branch's own path; none: the period label's bookkeeping path, not every
    # forecast copy of the same calendar years).  Built on demand from the provider:
    # ``flex_data.dd_same_scenario`` is populated only by ``apply_branch_cluster`` (derived_g), after
    # apply_npv.  None without a fan (deterministic / rolling) -> walks unfiltered.
    from flextool.engine.preprocess.params._branch import (
        annuity_window_lineage_df,
        assert_recourse_npv_preconditions,
    )
    if _recourse_branch_axis_present():
        assert_recourse_npv_preconditions(
            workdir, source, active_solve, provider=provider)
    _npv_lineage = annuity_window_lineage_df(
        workdir, source, active_solve, provider=provider)
    period_in_use = _period_in_use_set(source, active_solve, workdir, provider=provider)
    period_universe = _periodAll_from_source(source, active_solve, workdir=workdir, provider=provider)
    period_invest = _expand_invest_branch_periods(
        _solve_periods(source, active_solve, "invest_periods"),
        workdir, provider=provider) or []
    period_with_history = (_read_period_with_history(workdir, provider=provider)
                              or list(period_in_use))
    # The solve's years from the start of the run, when its own count does not give them (a solve after the run
    # start, fractional years): every factor below, the objective's and the walks', discounts from the run start.
    year_rows = solve_year_rows(workdir, provider, source, active_solve, list(period_universe))
    if year_rows is not None:
        notice_discount_from_run_start(active_solve, _solve_inflation_scalars(source)[0])

    # 1. p_inflation_op (None == no inflation data → field stays default).
    flex_data.p_inflation_op = p_inflation_op_from_source(
        source, active_solve, period_in_use, period_universe,
        provider=provider, year_rows=year_rows)
    # Reported costs: the discount factors from the start of the run (outputs only).
    run_yrs = run_years(source, active_solve, list(period_in_use), workdir, provider=provider)
    apply_run_factors(flex_data, source, active_solve, run_yrs,
                      None if run_yrs is not None else p_inflation_op_from_source(
                          source, active_solve, period_in_use, period_universe, provider=provider,
                          factor="inv_factor", year_rows=year_rows))

    apply_pre_existing_of_reported_costs(flex_data, source, active_solve, list(period_in_use), workdir,
                                         provider=provider)

    # 2. p_ed_fixed_cost — None == no period_with_history (single-solve
    #    fixtures with no historical periods); legitimate "no data" outcome.
    flex_data.p_ed_fixed_cost = p_ed_fixed_cost_from_source(
        source, period_with_history)

    # 3-6. NPV / lifetime cascade family.  Mirrors the "no_invest"
    # short-circuit in :mod:`._derived.ed_entity_annual_family_from_source`
    # and :func:`flextool.engine.preprocess.flex_data._load_invest`: when
    # neither ed_invest nor ed_divest carries any row, the cluster
    # outputs are all None (the LP has no v_invest / v_divest
    # variables that would consume them).
    ed_invest = getattr(flex_data, "ed_invest_set", None)
    ed_divest = getattr(flex_data, "ed_divest_set", None)
    built = getattr(flex_data, "cd_divest_built", None)
    no_invest = (
        (ed_invest is None or ed_invest.height == 0)
        and (ed_divest is None or ed_divest.height == 0)
        and (built is None or built.height == 0)
    )
    if no_invest or active_solve is None:
        if run_yrs is None:
            _run_lumps_are_the_solves(flex_data)
        return

    # A helper returns None only when its feature is inactive (no period, no entity row); the field then keeps
    # its loaded value (the divest lumps' emitted ``solve_data`` frames).
    def _set_if(field: str, value):
        if value is not None:
            setattr(flex_data, field, value)

    apply_invest_lumps(flex_data, source, active_solve, list(period_invest), list(period_in_use),
                       list(period_universe), list(period_with_history), lineage=_npv_lineage,
                       year_rows=year_rows, provider=provider)
    apply_retirement_salvage(flex_data, source, active_solve, list(period_invest), list(period_in_use),
                             list(period_universe), list(period_with_history), lineage=_npv_lineage,
                             year_rows=year_rows, workdir=workdir, provider=provider)
    apply_built_retirement_costs(flex_data, source, active_solve, list(period_in_use), list(period_universe),
                                 list(period_with_history), lineage=_npv_lineage, year_rows=year_rows,
                                 workdir=workdir, provider=provider)

    _set_if("ed_lifetime_fixed_cost_divest",
            ed_lifetime_fixed_cost_divest_from_source(
                source, active_solve,
                period_invest, period_in_use, period_universe,
                lineage=_npv_lineage, workdir=workdir, provider=provider, year_rows=year_rows))

    # Reported costs: the lumps from the start of the run over the run's periods, and the windows (outputs only).
    apply_run_cost_fields(flex_data, source, active_solve, list(period_in_use), list(period_invest),
                          list(period_with_history), workdir, lineage=_npv_lineage, years=run_yrs,
                          provider=provider)


#: The investment lumps whose ``solve_data`` frames are the values set here (the CSV emitter does not derive them).
_REGISTERED_LUMPS = ("ed_entity_annual_discounted", "ed_lifetime_fixed_cost")


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
    register their ``solve_data`` frames (:func:`_register_invest_lumps`) and write each investment cohort's prices
    into the cohort table (``flex_data.cohort``: ``a_cap``, ``pay``, ``n_pay``, ``r_loan``, ``a_salv``, ``nsv_eol``,
    carried to the later solves by ``realized_cohort``).  These are the only derivation of both lumps: a Spine solve (:func:`apply_npv`) and a
    synthetic sub-solve (:func:`apply_run_fields_of_synthetic_solve`) call it alike.  The start of operation of a
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
    level salvage credit of the objective constant), ``ed_divest_settlement_yearly`` (the reported yearly amount of
    the retirement period: ``ed_divest_salvage / inflation_inv``) and the existing cohort's prices (``r_loan``,
    ``a_salv``, ``nsv_eol``).  The only derivation of these terms: a Spine solve (:func:`apply_npv`) and a synthetic
    sub-solve (:func:`apply_run_fields_of_synthetic_solve`) call it alike."""
    divest, credit, prices = existing_salvage(
        source, active_solve, period_invest, period_in_use, period_universe, period_with_history,
        lineage=lineage, year_rows=year_rows, workdir=workdir, provider=provider,
        run_factor=getattr(flex_data, "p_inflation_inv_run", None))
    _notice_net_salvage_once(divest["e"].unique().to_list())
    flex_data.ed_divest_salvage = _ed_param(divest) if divest.height > 0 else None
    flex_data.ed_existing_salvage_credit = _ed_param(credit) if credit.height > 0 else None
    if divest.height > 0:
        factor = (_inflation_factors_lf(source, active_solve, list(dict.fromkeys(
                      [*(str(d) for d in period_universe), *(str(d) for d in period_invest)])),
                                        provider=provider, year_rows=year_rows)
                  .select(pl.col("d").cast(pl.Utf8), factor=pl.col("inv_factor")).collect())
        flex_data.ed_divest_settlement_yearly = _ed_param(
            divest.join(factor, on="d", how="left")
                  .select("e", "d", value=pl.col("value") / pl.col("factor").fill_null(1.0)))
    else:
        flex_data.ed_divest_settlement_yearly = None
    _set_existing_cohort_prices(flex_data, prices)


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
    (the settlement less the level price taken back) and ``cd_divest_built_fixed`` (the fixed cost saved), after the
    cohort prices of this solve's investments (:func:`apply_invest_lumps`) are on ``cohort_divest_built``.  A
    retirable cohort without its prices is an error (:func:`._retire_built.require_cohort_prices`), never a 0.  Warns
    where building a cohort and retiring it earns money (:func:`._retire_built.warn_build_to_retire`).  The only
    derivation of these terms: a Spine solve (:func:`apply_npv`) and a synthetic sub-solve
    (:func:`apply_run_fields_of_synthetic_solve`) call it alike."""
    from flextool.engine.preprocess.params._retire_built import require_cohort_prices, warn_build_to_retire
    options = getattr(flex_data, "cd_divest_built", None)
    cohorts = getattr(flex_data, "cohort_divest_built", None)
    if options is None or options.height == 0 or cohorts is None or cohorts.height == 0:
        flex_data.cd_divest_built_cost = flex_data.cd_divest_built_fixed = None
        return
    cohorts = _with_cohort_prices(cohorts, getattr(flex_data, "cohort", None))
    flex_data.cohort_divest_built = cohorts
    require_cohort_prices(cohorts)
    cost, fixed = built_retirement_costs(
        source, active_solve, cohorts, options, period_in_use, period_universe, period_with_history,
        lineage=lineage, year_rows=year_rows, workdir=workdir, provider=provider,
        run_factor=getattr(flex_data, "p_inflation_inv_run", None))
    flex_data.cd_divest_built_cost = Param(("e", "cohort", "d"), cost)
    flex_data.cd_divest_built_fixed = Param(("e", "cohort", "d"), fixed)
    warn_build_to_retire(_build_and_retire_totals(flex_data, cohorts, cost, fixed))


def _with_cohort_prices(cohorts: pl.DataFrame, cohort: pl.DataFrame | None) -> pl.DataFrame:
    """``cohorts`` (``cohort_divest_built``) with the prices of each cohort decided in this solve from the cohort
    table ``cohort`` (set by :func:`_set_cohort_prices`); a carried cohort keeps the prices it was carried with."""
    from flextool.engine.preprocess.params._cohorts import KIND_INVEST
    if cohort is None or cohort.height == 0:
        return cohorts
    own = (cohort.filter(pl.col("kind") == KIND_INVEST)
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
    price and lifetime fixed cost, ``ed_entity_annual_discounted`` + ``ed_lifetime_fixed_cost`` of its decision)
    plus the smallest coefficient of retiring it (``cost − fixed``), CUR/MW."""
    retire = (cost.join(fixed.rename({"value": "fixed"}), on=["e", "cohort", "d"], how="inner")
                  .group_by("e", "cohort").agg((pl.col("value") - pl.col("fixed")).min().alias("retire")))
    built = cohorts.filter(pl.col("d_build").is_not_null()).select("e", "cohort", "d_build")
    for name in ("ed_entity_annual_discounted", "ed_lifetime_fixed_cost"):
        param = getattr(flex_data, name, None)
        if param is None:
            built = built.with_columns(pl.lit(0.0).alias(name))
            continue
        frame = param.frame.select("e", pl.col("d").alias("d_build"), pl.col("value").alias(name))
        built, frame = align_join_dtypes(built, frame, ["e", "d_build"])
        built = built.join(frame, on=["e", "d_build"], how="left").with_columns(pl.col(name).fill_null(0.0))
    built, retire = align_join_dtypes(built, retire, ["e"])
    return (built.join(retire, on=["e", "cohort"], how="inner")
                 .select("e", "cohort", total=pl.col("ed_entity_annual_discounted") + pl.col("ed_lifetime_fixed_cost")
                         + pl.col("retire")))


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


def _set_cohort_prices(flex_data: object, lumps: pl.DataFrame) -> None:
    """``flex_data.cohort`` with the prices (:data:`._cohorts.PRICE_COLUMNS`) of each investment cohort (``invest:<d>``
    of entity ``e``) from ``lumps`` (:func:`npv_invest_discounted_lf`); the existing cohorts keep theirs (no capital in
    the model; their salvage terms: :func:`_set_existing_cohort_prices`)."""
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
    from flextool.engine import _behaviour_changes
    from flextool.engine.preprocess.params._cohorts import KIND_EXISTING
    if cohort is None or cohort.height == 0 or "root" not in cohort.columns:
        return
    rows = cohort.filter(~pl.col("root").str.starts_with(f"{KIND_EXISTING}:") & pl.col("a_cap").is_not_null()
                         & pl.col("a_salv").is_not_null() & (pl.col("a_cap") - pl.col("a_salv") < -1e-9))
    names = sorted({str(e) for e in rows["e"].cast(pl.Utf8).to_list()})
    if names:
        _behaviour_changes.warn_once_each(
            "salvage_above_invest_cost", names,
            lambda fresh: "Warning: " + ", ".join(fresh) + ": a plant the model can build has a net_salvage_value "
            "worth more than its invest_cost (the salvage at the end of its life, discounted at discount_rate): "
            "its yearly cost is below 0, so the model builds as much as the investment limits allow. Check "
            "net_salvage_value against invest_cost (also in the periods a replacement comes online).")


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


def _register_invest_lumps(flex_data: object, provider: "object | None") -> None:
    """Register ``ed_entity_annual_discounted`` / ``ed_lifetime_fixed_cost`` of ``flex_data`` under their
    ``solve_data/<name>`` provider keys (``entity, period, value``, the value rendered with ``repr``), so that the
    solve's frames of these lumps are the objective's coefficients: one derivation, with the start of operation of
    a construction lead time and the years in service."""
    if provider is None:
        return
    from flextool.engine.preprocess._provider_io import _emit
    from flextool.engine.preprocess.emit._vectorize import _render_value_column
    for name in _REGISTERED_LUMPS:
        param = getattr(flex_data, name, None)
        if param is None:
            continue
        frame = _utf8_ed(param.frame.lazy())
        _emit(provider, f"solve_data/{name}.csv",
              pl.DataFrame([frame["e"].alias("entity"), frame["d"].alias("period"),
                            _render_value_column(frame["value"])]))


__all__ = [
    "period_walk_iterator",
    "p_inflation_op_from_source",
    "p_ed_fixed_cost_from_source",
    "ed_entity_annual_discounted_from_source",
    "ed_lifetime_fixed_cost_from_source",
    "ed_lifetime_fixed_cost_divest_from_source",
    "apply_npv",
    "apply_invest_lumps",
    "apply_run_cost_fields",
    "apply_run_factors",
    "apply_run_fields_of_synthetic_solve",
    "run_years",
    "notice_discount_from_run_start",
    "solve_year_rows",
    "year_rows_from_history",
]

