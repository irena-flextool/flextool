"""Cluster A — annual integration / NPV (Δ.5).

Lazy-polars port of flextool's NPV / lifetime-fixed-cost / inflation
cascade.  Replaces the eager-Python implementation that previously lived
in :mod:`._derived` (``ed_entity_annual_family_from_source`` plus
its supporting per-(entity, period) lookup).

The cluster covers six FlexData fields:

* ``p_inflation_op``                           — per-period operations
  inflation factor.  Replaces the workdir read of
  ``solve_data/p_inflation_factor_operations_yearly.csv``.
* ``p_ed_fixed_cost``                          — per-entity per-period
  raw fixed cost (× 1000 for nodes/processes).  Replaces the workdir
  read of ``solve_data/ed_fixed_cost.csv``.
* ``ed_entity_annual_discounted``              — invest-side NPV.
  Replaces ``solve_data/ed_entity_annual_discounted.csv``.
* ``ed_entity_annual_divest_discounted``       — divest-side NPV.
  Replaces ``solve_data/ed_entity_annual_divest_discounted.csv``.
* ``ed_lifetime_fixed_cost``                   — invest-side lifetime
  cumulative fixed cost.  Replaces ``solve_data/ed_lifetime_fixed_cost.csv``.
* ``ed_lifetime_fixed_cost_divest``            — divest-side lifetime
  cumulative fixed cost.  Replaces ``solve_data/ed_lifetime_fixed_cost_divest.csv``.

and, for the reported costs only (no model term reads them), their run-level
twins ``*_run`` discounted from the start of the run over the periods of the
whole run, the undiscounted yearly amounts ``ed_annuity`` (an investment's price per year in service) /
``ed_divest_annuity`` (the salvage annuity),
the operating / retirement / salvage windows ``edd_operating_window`` /
``edd_retire_window`` / ``edd_salvage_window`` (section "Run-level fields of the reported costs").

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
    ``inflation_offset_investment``), ``n`` the payback time
    (``payback_time``, else the entity's ``lifetime_technical``).  A missing
    ``r`` takes the schema default (0.05); a missing payback time where the
    cost is not 0 is an error.  The investment of ``reinvest_choice`` /
    ``no_reinvestment`` takes the level yearly price instead,
    ``a_cap = annuity × G(n) / G(L)`` (:func:`.._annuity.level_factor`,
    ``L`` the ``lifetime_technical``).
  * ``window(d, d_all, e)``:
      - investment price and lifetime fixed cost, ``reinvest_choice`` / ``no_reinvestment``: ``pdy[d_all] ∈
        [pdy[d], y_eol)``, ``y_eol = round_end_of_life(pdy[d] + lifetime_technical[e, d], end_of_life_method[e])``
        (the periods the cohort is in service; :func:`_window_life_to_end_of_life`).
      - ``reinvest_automatic``: ``pdy[d_all] ≥ pdy[d]``.
      - divest side: always the bounded form using ``lifetime_technical``.
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
from flextool.engine.preprocess.params._cohorts import BOUNDED_METHODS
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
    """Read ``model.inflation_rate`` / ``model.inflation_offset_invest`` /
    ``model.inflation_offset_operations`` scalars, falling back to
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
        _explicit_max("inflation_rate", _INFL_RATE_DEFAULT),
        _explicit_max("inflation_offset_investment", _INFL_OFFSET_INV_DEFAULT),
        _explicit_max("inflation_offset_operations", _INFL_OFFSET_OPS_DEFAULT),
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
    """Lazy ``[d, off, width, inv_contrib, ops_contrib]``: one row per represented year of a period of
    ``period_universe`` (``d`` on the period axis), ``off`` the years of the period before it, ``width`` its width and
    ``inv_contrib`` / ``ops_contrib`` its discounted width (the terms of :func:`_inflation_factors_lf`).

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
                  .select("d", "off", "width", "inv_contrib", "ops_contrib"))
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
              .select("d", "off", pl.col("pyr").alias("width"), "inv_contrib", "ops_contrib"))


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


def _raw_lifetime_on_lf(source: "InputSource", ed_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Lazy ``(e, d, life)``: ``lifetime_technical`` (not edEntity_lifetime) over the given ``(e, d)`` pairs, read by
    :func:`._period_values.ed_values_on` with the class gate (only nodes and processes; others 0).  The divest-side
    window."""
    cls_lf = _entity_class_lookup_lf(_entity_class_lf(source))
    in_class = (pl.col("is_process").fill_null(False)
                | pl.col("is_node").fill_null(False))
    ed_with_class = ed_lf.join(cls_lf, on="e", how="left")
    return (ed_values_on(source, LIFETIME_TECHNICAL, ed_with_class, fill=0.0)
              .with_columns(
                  life=pl.when(in_class).then(pl.col("value")).otherwise(0.0),
              )
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


def _notice_annuity_timing(rows: pl.LazyFrame, offset: float) -> None:
    """The ``annuity_timing`` notice for the entities its marker names (an upgraded database whose annuity was the
    year-end annuity at every ``inflation_offset_investment``) whose annuity now differs: ``rows`` ``[e, cost,
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
        ) -> pl.LazyFrame:
    """Lazy ``(e, d, ann)`` frame computing the per-method annuity sum; with ``level`` ``(e, d, ann, a_cap)``.

    ``commission`` (``[e, d, d_c]``, the decisions with a construction lead time): the cost, the interest rate and
    the payback time (``lifetime_technical`` for an entity without ``payback_time``) of such a decision are those
    of the period ``d_c`` it comes online (:func:`._period_values.ed_values_online`).

    ``level``: also the level yearly price of the plant over its real lifetime (``invest_cost_p`` OBJ header),
    ``lifetime_technical`` ``L`` read at the same period::

        a_cap[e, d] = ann[e, d]                                     |n − L| ≤ YEAR_TOL, or no L
                    = ann[e, d] · G(n) / G(L)  (:func:`.._annuity.level_factor`, i = model.inflation_rate)

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

    # Annuity: cost [CUR/kW] × 1000 × annuity_factor(r, n, inflation_offset_investment).
    offset = _solve_inflation_scalars(source)[1]
    _notice_annuity_timing(ed_with_life.join(method_count_lf, on="e", how="inner").filter(in_class)
                           .select("e", "cost", "disc"), offset)
    annuity_expr = (pl.col("cost") * 1000.0
                    * annuity_factor(pl.col("disc"), pl.col("life"), offset))
    out = (ed_with_life
              .join(method_count_lf, on="e", how="left")
              .with_columns(
                  method_count=pl.col("method_count").fill_null(0),
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
               .select("e", "d", "ann", "a_cap"))


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
        ) -> pl.LazyFrame:
    """Variant 1: ``ed_entity_annual_discounted`` (invest-side NPV).

    Per (e ∈ entityInvest, d ∈ period_invest)::

        price[e, d] × Σ_{d_all ∈ period_in_use, window(d, d_all)} inv_factor[d_all]

    where ``window`` and ``price`` switch on ``e``'s lifetime_method: ``reinvest_choice`` / ``no_reinvestment`` →
    the periods the cohort of the decision is in service (``yr[c] ≤ yr[d_all] < y_eol``, the window of the lifetime
    fixed cost, :func:`_window_life_to_end_of_life`) at the level yearly price ``a_cap`` of the plant over its real
    lifetime (:func:`_per_method_annuity_lf` with ``level``); ``reinvest_automatic`` → unbounded, at the annuity.
    An entity may have multiple methods; each contributes a separate sum.  See ``entity_annual_calc_params.py:222-251``.

    ``commission`` (``[e, d, d_c, yr_c]``, the decisions with a construction lead time): the price of such a
    decision is that of the period ``d_c`` it comes online (invest cost, interest rate, payback time and lifetime
    there) and its window starts at ``yr_c``, so it is paid only over the years the capacity operates within the
    solve.

    ``lineage`` — see :func:`._walks.period_walk_iterator`; the
    :func:`._branch.annuity_window_lineage_df` frame in a stochastic fan, ``None`` otherwise.

    Returns lazy ``[e, d, value, a_cap]``: ``a_cap`` the price per year in service (the annuity for
    ``reinvest_automatic``).
    """
    if not period_invest:
        return pl.LazyFrame(schema={
            "e": schema_dtype(_enums, "e"),
            "d": schema_dtype(_enums, "d"),
            "value": pl.Float64,
            "a_cap": pl.Float64,
        })
    # entityInvest = projection of entity__invest_method via the
    # method-allowed gate (mirror of preprocessing/invest_method_sets).
    entity_invest_set = _entity_invest_set_lf(source, _INVEST_NOT_ALLOWED)
    pi_lf = pl.LazyFrame({"d": period_invest}).with_columns(
        alias_to_axis("d", "d"))
    ed_anchor_lf = entity_invest_set.join(pi_lf, how="cross")

    # Annuity and level price (invest cost), of the period a decision with a lead time comes online.
    ann_lf = _per_method_annuity_lf(source, ed_anchor_lf, "invest_cost", _INVEST_NOT_ALLOWED, commission,
                                    level=True)

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
                      factor_b=pl.col("factor_b").fill_null(0.0),
                      factor_u=pl.col("factor_u").fill_null(0.0),
                      has_choice=pl.col("has_choice").fill_null(False),
                      has_automatic=pl.col("has_automatic").fill_null(False),
                  )
                  .with_columns(
                      value=(
                          pl.when(pl.col("has_choice"))
                              .then(pl.col("a_cap") * pl.col("factor_b"))
                              .otherwise(0.0)
                          + pl.when(pl.col("has_automatic"))
                              .then(pl.col("ann") * pl.col("factor_u"))
                              .otherwise(0.0)
                      ),
                  ))
    # Only the entities whose investment takes the bounded window in this solve (the reinvest_automatic sum is
    # unchanged).  The value before: the annuity over whole periods that start before the end of the lifetime of
    # the start of operation.
    _notice_annuity_over_payback_time(
        combined.join(has_choice_or_no.select("e"), on="e", how="semi").select("e", "d", "value"),
        lambda: (period_walk_iterator(source, active_solve, ed_anchor_lf, period_in_use, period_universe,
                                      bounded=True, life_lf=technical_lf, factor_side="inv", lineage=lineage,
                                      commission_lf=commission_lf, year_rows=year_rows)
                 .join(ann_lf, on=["e", "d"], how="left")
                 .select("e", "d", value=pl.col("factor") * pl.col("ann").fill_null(0.0))))
    return combined.select("e", "d", "value",
                           a_cap=pl.when(pl.col("has_choice")).then(pl.col("a_cap")).otherwise(pl.col("ann")))


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


def npv_divest_discounted_lf(
        source: "InputSource",
        active_solve: str | None,
        period_invest: list[str],
        period_in_use: list[str],
        period_universe: list[str],
        *,
        lineage: pl.DataFrame | None = None,
        year_rows: pl.DataFrame | None = None,
        ) -> pl.LazyFrame:
    """Variant 2: ``ed_entity_annual_divest_discounted`` (divest-side NPV).

    Per (e ∈ entityDivest, d ∈ period_invest)::

        ann[e, d] × Σ_{d_all : pdy ∈ [pdy_d, pdy_d + life)} inv_factor[d_all]

    where ``life`` is the **raw** ``lifetime_technical`` value (not edEntity_lifetime),
    and the gate is restricted to e ∈ node ∪ process (mirrors
    L266-285 — no fallback for unclassified entities).

    ``lineage`` — see :func:`._walks.period_walk_iterator`; the
    :func:`._branch.annuity_window_lineage_df` frame in a stochastic fan, ``None`` otherwise.

    Returns lazy ``[e, d, value]``.
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

    ann_lf = _per_method_annuity_lf(
        source, ed_anchor_lf, NET_SALVAGE_VALUE, _DIVEST_NOT_ALLOWED)

    # Use raw lifetime (per L271-284), not edEntity_lifetime.
    life_lf = _raw_lifetime_on_lf(source, ed_anchor_lf)

    walk = period_walk_iterator(
        source, active_solve, ed_anchor_lf,
        period_in_use, period_universe,
        bounded=True, life_lf=life_lf, factor_side="inv",
        lineage=lineage, year_rows=year_rows)

    return (ed_anchor_lf
              .join(ann_lf, on=["e", "d"], how="left")
              .join(walk, on=["e", "d"], how="left")
              .with_columns(
                  ann=pl.col("ann").fill_null(0.0),
                  factor=pl.col("factor").fill_null(0.0),
                  value=pl.col("ann") * pl.col("factor"),
              )
              .select("e", "d", "value"))


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
                                 provider: "object | None") -> pl.LazyFrame:
    """``life_lf`` ``[e, d, life]`` (the lifetime of decision ``(e, d)``, read at the period it comes online) with
    ``life`` the years from the start of operation ``y_on`` to the rounded end of life of its cohort, so that the
    bounded walk ``yr[d′] < yr_avail + life`` keeps the periods ``y_on ≤ yr[d′] < y_eol``::

        y_on  = y[c(d)]  (years of the run, :func:`._cohorts.cohort_years`; c(d) = d without a lead time)
        y_eol = round_end_of_life(y_on + life, end_of_life_method[e])   (:func:`._cohorts.end_of_life_years`)
        life  = y_eol − y_on − YEAR_TOL / 2

    The half tolerance keeps the period starting at ``y_eol`` out whatever the float rounding of
    ``y_on + (y_eol − y_on)`` (period starts lie more than ``YEAR_TOL`` apart).  Under ``next_seam`` the walk keeps
    the same periods as with the raw lifetime.  A decision whose start of operation has no year keeps its
    lifetime."""
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
        return life_lf
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
    return (life.with_columns(y_eol=y_eol)
                .with_columns(life=pl.when(pl.col("y_on").is_null()).then(pl.col("life"))
                                     .otherwise(pl.col("y_eol") - pl.col("y_on") - YEAR_TOL / 2))
                .drop("y_on", "y_eol")
                .lazy())


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


def ed_entity_annual_divest_discounted_from_source(
        source: "InputSource",
        active_solve: str | None,
        period_invest: list[str],
        period_in_use: list[str],
        period_universe: list[str],
        *,
        lineage: pl.DataFrame | None = None,
        year_rows: pl.DataFrame | None = None,
        ) -> "Param | None":
    """Public entry: ``ed_entity_annual_divest_discounted[e, d]``.

    Unfiltered ``entityDivest × period_invest`` — see the rationale on
    :func:`ed_entity_annual_discounted_from_source`.

    ``lineage`` — see :func:`._walks.period_walk_iterator`; the
    :func:`._branch.annuity_window_lineage_df` frame in a stochastic fan, ``None`` otherwise.
    """
    if not period_invest:
        return None
    out = (npv_divest_discounted_lf(
              source, active_solve,
              period_invest, period_in_use, period_universe,
              lineage=lineage, year_rows=year_rows)
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
#   the window of _cohorts.retire_active.  edd_salvage_window[e,d_divest,d]: yr_run[d] ∈ [yr_run[d_divest],
#   yr_run[d_divest] + lifetime_technical[e, d_divest]), the window the net salvage value is spread over.  In a
#   stochastic fan only the pairs of the walk lineage (each calendar period once).
# OUT. ed_annuity[e,d] = the price per year in service (CUR/MW/yr, undiscounted) of the vintage: a_cap[e,c(d)] for
#   reinvest_choice / no_reinvestment (the level yearly cost of the plant over its lifetime_technical, the objective's
#   price, _per_method_annuity_lf), the annuity of invest_cost at c(d) over payback_time for reinvest_automatic;
#   ed_divest_annuity[e,d] the annuity of net_salvage_value at d.
# OUT. ed_entity_annual_discounted_run[e,d] = Σ_{kind} price_kind[e,d]·Σ_{(e,d,d') ∈ edd_operating_window(kind)}
#        p_inflation_inv_run[d']
#   price_bounded = a_cap, price_unbounded = the annuity: the objective's investment lump over the run periods.
#      ed_lifetime_fixed_cost_run[e,d] = Σ_{(e,d,d') ∈ edd_operating_window} fc[e,d']·p_inflation_op_run[d']
#        (summed as fc[e,c(d)]·Σ factor + Σ (fc[e,d'] − fc[e,c(d)])·factor, the objective's form)
#      ed_entity_annual_divest_discounted_run[e,d]
#        = ed_divest_annuity[e,d]·Σ_{edd_salvage_window} p_inflation_inv_run[d']
#      ed_lifetime_fixed_cost_divest_run[e,d] = Σ_{(e,d,d') ∈ edd_retire_window} fc[e,d']·p_inflation_op_run[d']
#   The fixed costs are those of each window period with the operations factor (as the fixed cost of existing
#   capacity is charged), as their per-solve twins.
#   The salvage value keeps the annuity over lifetime and the window of the investment annuity (its per-solve
#   twin ed_entity_annual_divest_discounted likewise), so retiring existing capacity and building identical new
#   capacity earns nothing.
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
        # The price per year in service of each lifetime kind: the level yearly price a_cap for reinvest_choice /
        # no_reinvestment, the annuity for reinvest_automatic (the objective's, npv_invest_discounted_lf).
        kinds = _lifetime_kinds(source, inv.select("e").unique())
        ann = _utf8_ed(_per_method_annuity_lf(source, _anchor_lf(inv), "invest_cost", _INVEST_NOT_ALLOWED,
                                              commission, level=True))
        price = (ann.join(kinds, on="e", how="inner")
                    .select("e", "d", "kind",
                            coef=pl.when(pl.col("kind") == "bounded").then(pl.col("a_cap")).otherwise(pl.col("ann"))))
        bounded_e = kinds.filter(pl.col("kind") == "bounded").select("e")
        flex_data.ed_annuity = _ed_param(
            vintages.join(ann.with_columns(bounded=pl.col("e").is_in(bounded_e["e"].implode())), on=["e", "d"],
                          how="left")
                    .select("e", "d", value=pl.when(pl.col("bounded")).then(pl.col("a_cap"))
                                             .otherwise(pl.col("ann")).fill_null(0.0)))
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
        if years is not None:
            fc = _utf8_ed(at_online_period(
                fc_lf, lambda lf: _ed_fixed_cost_raw_lf(source, lf), commission, "fc")).rename({"fc": "coef"})
            flex_data.ed_lifetime_fixed_cost_run = _ed_param(_fixed_lump(walk, fc, factor_ops, fc_pairs, source))
            if inv.height > 0:
                # The price over the periods in service, as the objective's investment lump.
                flex_data.ed_entity_annual_discounted_run = _ed_param(
                    _lump(walk.join(inv, on=["e", "d"], how="semi"), price, factor_inv, inv))

    # Retirement side: from the retirement period until the rounded end of life ``y_eol`` of the existing capacity
    # (the end of the run for reinvest_automatic), the window of _cohorts.retire_active on the run calendar.
    div = anchors_of("ed_divest_set", _DIVEST_NOT_ALLOWED, period_invest)
    if div.height > 0:
        from flextool.engine.preprocess.params._cohorts import existing_cohorts, existing_ends
        from flextool.engine.preprocess.params._run_calendar import YEAR_TOL
        div_lf = _anchor_lf(div)
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
        # The salvage value: its annuity over lifetime, over the window [yr, yr + lifetime) (raw lifetime).
        life = _utf8_ed(_raw_lifetime_on_lf(source, div_lf))
        salvage_walk = _run_window(year_of_start(div, False)
                                   .join(life, on=["e", "d"], how="left")
                                   .with_columns(life=pl.col("life").fill_null(0.0), kind=pl.lit("bounded")),
                                   yr, lineage, period_in_use)
        flex_data.edd_salvage_window = _window_frame(salvage_walk, "d_divest")
        ann = _utf8_ed(_per_method_annuity_lf(source, div_lf, NET_SALVAGE_VALUE, _DIVEST_NOT_ALLOWED)
                       ).rename({"ann": "coef"})
        flex_data.ed_divest_annuity = _ed_param(div.join(ann, on=["e", "d"], how="left")
                                                   .select("e", "d", value=pl.col("coef").fill_null(0.0)))
        if years is not None:
            fc_all = _utf8_ed(_ed_fixed_cost_raw_lf(
                source, walk.lazy().select(alias_to_axis("e", "e"), alias_to_axis("d_all", "d")).unique()))
            saved = (walk.join(fc_all.rename({"d": "d_all"}), on=["e", "d_all"], how="left")
                         .join(factor_ops, on="d_all", how="left")
                         .group_by("e", "d")
                         .agg((pl.col("fc").fill_null(0.0) * pl.col("factor").fill_null(0.0)).sum().alias("value")))
            flex_data.ed_entity_annual_divest_discounted_run = _ed_param(_lump(salvage_walk, ann, factor_inv, div))
            flex_data.ed_lifetime_fixed_cost_divest_run = _ed_param(
                div.join(saved, on=["e", "d"], how="left").select("e", "d", value=pl.col("value").fill_null(0.0)))

    if years is None:
        _run_lumps_are_the_solves(flex_data)


def _run_lumps_are_the_solves(flex_data: object) -> None:
    """The run is the solve: the run lumps are the solve's own lumps."""
    for name in ("ed_entity_annual_discounted", "ed_entity_annual_divest_discounted", "ed_lifetime_fixed_cost",
                 "ed_lifetime_fixed_cost_divest"):
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

    A sub-solve with an invest set gets its annuity and lifetime fixed cost from
    :func:`apply_invest_lumps` over its own periods, as a Spine solve does (the CSV emitter derives neither); its
    divest lumps and its own discount factors stay the loaded ``solve_data`` frames.  Without run years the run
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
    if ed_invest is not None and ed_invest.height > 0:
        # The decision periods as apply_npv and ed_invest_set read them (the sub-solve's own invest periods).
        decision_periods = _expand_invest_branch_periods(
            _solve_periods(source, active_solve, "invest_periods"), workdir, provider=provider) or []
        period_universe = list(_periodAll_from_source(source, active_solve, workdir=workdir, provider=provider))
        year_rows = solve_year_rows(workdir, provider, source, active_solve, period_universe)
        if year_rows is not None:
            notice_discount_from_run_start(active_solve, _solve_inflation_scalars(source)[0])
        apply_invest_lumps(flex_data, source, active_solve, decision_periods, period_in_use, period_universe,
                           period_with_history, lineage=lineage, year_rows=year_rows, provider=provider)
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
      4. ``ed_entity_annual_divest_discounted``
                                            (depends on inv_factor + ed_divest)
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
    no_invest = (
        (ed_invest is None or ed_invest.height == 0)
        and (ed_divest is None or ed_divest.height == 0)
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

    _set_if("ed_entity_annual_divest_discounted",
            ed_entity_annual_divest_discounted_from_source(
                source, active_solve,
                period_invest, period_in_use, period_universe,
                lineage=_npv_lineage, year_rows=year_rows))

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
    register their ``solve_data`` frames (:func:`_register_invest_lumps`) and write each investment cohort's price
    per year in service into the cohort table (``flex_data.cohort.a_cap``, carried to the later solves by
    ``realized_cohort``).  These are the only derivation of both lumps: a Spine solve (:func:`apply_npv`) and a
    synthetic sub-solve (:func:`apply_run_fields_of_synthetic_solve`) call it alike.  The start of operation of a
    construction lead time is ``flex_data.ed_invest_commission``.  A helper that returns None (no invest period, no
    entity row) leaves the field as it is."""
    commission = getattr(flex_data, "ed_invest_commission", None)
    if period_invest:
        lumps = (npv_invest_discounted_lf(source, active_solve, period_invest, period_in_use, period_universe,
                                          lineage=lineage, commission=commission, year_rows=year_rows,
                                          provider=provider)
                 .sort("e", "d").collect())
        if lumps.height > 0:
            flex_data.ed_entity_annual_discounted = Param(("e", "d"), lumps.select("e", "d", "value"))
            _set_cohort_prices(flex_data, lumps)
    fixed = ed_lifetime_fixed_cost_from_source(
        source, active_solve, period_with_history, period_in_use, period_universe,
        lineage=lineage, commission=commission, year_rows=year_rows, provider=provider)
    if fixed is not None:
        flex_data.ed_lifetime_fixed_cost = fixed
    _register_invest_lumps(flex_data, provider)


def _set_cohort_prices(flex_data: object, lumps: pl.DataFrame) -> None:
    """``flex_data.cohort`` with ``a_cap`` of each investment cohort (``invest:<d>`` of entity ``e``) its price per
    year in service from ``lumps`` (``[e, d, value, a_cap]``, :func:`npv_invest_discounted_lf`); the existing
    cohorts keep a null price (no capital in the model)."""
    cohort = getattr(flex_data, "cohort", None)
    if cohort is None or cohort.height == 0:
        return
    from flextool.engine.preprocess.params._cohorts import KIND_INVEST
    prices = lumps.select("e", pl.col("d").alias("d_build"), pl.col("a_cap").alias("_price"))
    left, prices = align_join_dtypes(cohort, prices, ["e", "d_build"])
    flex_data.cohort = (left
                        .join(prices, on=["e", "d_build"], how="left", maintain_order="left")
                        .with_columns(a_cap=pl.when(pl.col("kind") == KIND_INVEST).then(pl.col("_price"))
                                      .otherwise(pl.col("a_cap")))
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
    "ed_entity_annual_divest_discounted_from_source",
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

