"""Cluster A — annual integration / NPV (Δ.5).

Lazy-polars port of flextool's NPV / lifetime-fixed-cost / inflation
cascade.  Replaces the eager-Python implementation that previously lived
in :mod:`._derived_params` (``ed_entity_annual_family_from_source`` plus
its supporting ``_per_method_annuity`` / ``_resolve_pdX`` helpers).

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

Algorithm — port of
``flextool/flextoolrunner/preprocessing/entity_annual_calc_params.py``:

  For each (entity e, period d) in the relevant invest/divest/all-entity
  domain, the NPV value is::

      annuity[e, d] × Σ_{d_all ∈ period_in_use, window(d, d_all, e)}
                                       inflation_factor[d_all]

  where:

  * ``annuity`` = sum-over-allowed-methods of
    ``invest_value × 1000 × r / (1 - (1/(1+r))^n)`` with
    ``r ≤ 0 → 0.05`` and ``n ≤ 0 → 20`` fallbacks.
  * ``window(d, d_all, e)``:
      - ``reinvest_choice`` / ``no_investment``:
        ``pdy[d_all] ∈ [pdy[d], pdy[d] + lifetime[e, d])``.
      - ``reinvest_automatic``: ``pdy[d_all] ≥ pdy[d]``.
      - divest side: always the bounded form using raw ``lifetime``.
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

* :mod:`._derived_params._inflation_yearly_from_source` — the yearly
  inflation cascade computation used to be eager-Python; we keep that
  helper's algorithm but expose a lazy-frame variant
  (:func:`_inflation_factors_lf`) for use here.
* :func:`._derived_params._p_years_d_lf` — already lazy; reused as-is.
* :func:`._derived_params._period_in_use_set` and
  :func:`._derived_params._solve_periods` — list builders used to
  bound the period domain.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

from ._axis_enums import (
    alias_to_axis,
    axis_lazyframe,
    cast_dim,
    get_global_axis_enums,
    schema_dtype,
)

# Substrate handle for the cascade-wide axis enum vocabulary.
# Bare ``None`` here; ``cast_dim`` / ``schema_dtype`` in
# ``_axis_enums`` fall back to ``_LIVE_AXIS_ENUMS_CTX`` (the live
# ContextVar) when this is ``None``, so substrate sites pick up
# activation set by ``load_flextool`` automatically.
_enums: "dict | None" = None

if TYPE_CHECKING:
    from flextool.engine_polars._input_source import InputSource


# Mirror flextool/flextool_base.dat:211-212 (cf. entity_annual_calc_params.py).
_INVEST_NOT_ALLOWED: frozenset[str] = frozenset((
    "not_allowed", "retire_period", "retire_total", "retire_no_limit",
))
_DIVEST_NOT_ALLOWED: frozenset[str] = frozenset((
    "not_allowed", "invest_period", "invest_total", "invest_no_limit",
))

# Default lifetime_method when an entity has no explicit row
# (preprocessing/method_with_fallback_sets.py::_LIFETIME_METHOD_DEFAULT).
_LIFETIME_METHOD_DEFAULT = "reinvest_automatic"

# Inflation defaults — mirrors period_calculated_params.py:221-227 and
# _solve_inflation_inputs in _derived_params.py.
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
    class.  The frame supports the per-entity-class branching that
    ``_resolve_pdX`` does in eager Python.
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

    Mirrors :func:`._derived_params._resolve_pdX` semantics in lazy form:

    * Per-period (Map) rows produce ``(e, d, value)`` rows directly.
    * Scalar rows produce ``(e, value)`` rows with ``d`` null;
      consumers cross-join with the period universe.

    Returns columns ``[e, d, value, is_scalar]`` — ``is_scalar`` flags
    rows that need broadcasting.  ``d`` is null on scalar rows.
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
        # ``_derived_existing._per_entity_param_lf``.  When the source
        # exposes the period axis under a user-renamed ``Map.index_name``
        # (e.g. ``x``), checking only ``"period" in cols`` misses it and
        # the scalar-broadcast branch explodes per-period rows through
        # ``_resolve_per_period_lf``'s ``scalar.join(on="e")``.
        extra = [c for c in cols if c not in ("name", "value")]
        period_col = "period" if "period" in extra else (extra[0] if extra else None)
        if period_col is not None:
            parts.append(df.lazy().select(
                alias_to_axis("name", "e"),
                alias_to_axis(pl.col(period_col).cast(pl.Utf8, strict=False),
                              "d"),
                pl.col("value").cast(pl.Float64, strict=False),
                pl.lit(False).alias("is_scalar"),
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
                              fill: float = 0.0,
                              ) -> pl.LazyFrame:
    """Resolve ``pdX[e, param, d]`` cascade in lazy form.

    Order (mirrors :func:`._derived_params._resolve_pdX`):
      1. Explicit ``(e, d)`` row → take that ``value``.
      2. Else scalar ``(e, ·)`` row → broadcast across ``ed_lf``.
      3. Else → ``fill``.

    Returns ``ed_lf`` with an extra ``value`` column (Float64).
    ``ed_lf`` must carry columns ``[e, d, ...]``.
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

    Mirrors :func:`._derived_params._solve_inflation_inputs` semantics.
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
                              ) -> pl.LazyFrame:
    """Build the per-(d, year_label, width) frame from
    ``solve.years_represented`` mirroring
    :func:`._derived_params._years_for_period_from_source` in lazy form.

    Returns columns ``[d, y, width]`` where ``y`` is a string label.
    Empty if no rows.
    """
    # Materialise scalars eagerly — they're solve-level, tiny.
    from ._derived_params import _years_for_period_from_source
    yfp = _years_for_period_from_source(source, active_solve, period_universe)
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


def _inflation_factors_lf(source: "InputSource",
                                active_solve: str | None,
                                period_universe: list[str],
                                ) -> pl.LazyFrame:
    """Lazy ``(d, inv_factor, ops_factor)`` frame.

    Algorithm — port of ``preprocessing/period_calculated_params.py:230-322``:

      1. Scalar inputs: ``rate``, ``offset_invest``, ``offset_operations``.
      2. For each (d, y) ∈ ``years_for_period``:
           ``base[d, y] = Σ_{y' ∈ global, y' < y} pyr[d, y']``  (default 1)
           ``until_inv[d, y] = base + pyr × offset_invest``
           ``until_ops[d, y] = base + pyr × offset_operations``
      3. ``inv_factor[d] = Σ_y pyr[d, y] × (1+rate)^(-until_inv[d, y])``
         ``ops_factor[d] = Σ_y pyr[d, y] × (1+rate)^(-until_ops[d, y])``.
         When ``Σ_y pyr[d, y] == 0`` → 1.0.

    The cumulative-base computation (step 2) is per-period over the
    GLOBAL year set, so it's shaped naturally as a polars window:

      base[d, y] = Σ_{y'<y in global order} pyr[d, y']
                 = (cumulative_sum of pyr[d, y'] over y'<y, partitioned by d)

    where unbound (d, y) pairs contribute 1 (default ``years_represented``).
    """
    if not period_universe:
        return pl.LazyFrame(schema={
            "d": schema_dtype(_enums, "d"),
            "inv_factor": pl.Float64, "ops_factor": pl.Float64,
        })
    rate, off_inv, off_ops = _solve_inflation_scalars(source)
    one_plus_inv = (1.0 / (1.0 + rate)) if rate != -1.0 else 1.0
    yfp_lf = _years_for_period_lf(source, active_solve, period_universe)

    # Materialise per-period × per-y.  At LP scale this is small (typically
    # ≤ 100 (d, y) rows even for 50-year horizon × 20 representative years),
    # so collecting once is cheap; the rest stays lazy.
    yfp_eager = yfp_lf.collect()
    period_lf = pl.LazyFrame({"d": period_universe}).with_columns(
        alias_to_axis("d", "d"))

    if yfp_eager.height == 0:
        # No years_represented data — every period gets the trivial 1.0
        # factor (mirrors period_calculated_params.py:300).
        return (period_lf
                  .with_columns(
                      inv_factor=pl.lit(1.0, dtype=pl.Float64),
                      ops_factor=pl.lit(1.0, dtype=pl.Float64),
                  ))

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
    bound = bound.with_columns(
        until_inv=pl.col("base") + pl.col("pyr") * off_inv,
        until_ops=pl.col("base") + pl.col("pyr") * off_ops,
    ).with_columns(
        inv_contrib=pl.col("pyr") * (one_plus_inv ** pl.col("until_inv")),
        ops_contrib=pl.col("pyr") * (one_plus_inv ** pl.col("until_ops")),
    )

    factor_per_d = (bound
                      .group_by("d")
                      .agg(
                          pl.col("inv_contrib").sum().alias("inv_factor"),
                          pl.col("ops_contrib").sum().alias("ops_factor"),
                          pl.col("pyr").sum().alias("sum_pyr"),
                      ))

    # Sum-pyr == 0 → factor = 1.0 (period_calculated_params.py:299-301).
    factor_per_d = factor_per_d.with_columns(
        inv_factor=pl.when(pl.col("sum_pyr") > 0)
                       .then(pl.col("inv_factor"))
                       .otherwise(1.0),
        ops_factor=pl.when(pl.col("sum_pyr") > 0)
                       .then(pl.col("ops_factor"))
                       .otherwise(1.0),
    ).select("d", "inv_factor", "ops_factor")

    # Periods absent from years_represented (or with R=0) get 1.0
    # (mirrors flextool's writer default of 1.0).
    return (period_lf
              .join(factor_per_d, on="d", how="left")
              .with_columns(
                  inv_factor=pl.col("inv_factor").fill_null(1.0),
                  ops_factor=pl.col("ops_factor").fill_null(1.0),
              ))


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
        ) -> pl.LazyFrame:
    """Δ.5-era public alias delegating to :mod:`._derived_walks`.

    Δ.6 lifted the canonical implementation into ``_derived_walks`` so
    Cluster B's invest-history walk could re-use it.  This wrapper
    preserves the boolean ``bounded`` signature for the existing NPV
    callers while internally dispatching the new
    :class:`._derived_walks.WindowMethod` enum.

    See :func:`._derived_walks.period_walk_iterator` for the canonical
    semantics.
    """
    from ._derived_walks import (
        period_walk_iterator as _walk,
        WindowMethod,
    )
    method = WindowMethod.BOUNDED if bounded else WindowMethod.UNBOUNDED_FORWARD
    return _walk(
        source, active_solve, ed_lf,
        period_in_use, period_universe,
        window_method=method, life_lf=life_lf,
        factor_side=factor_side)


# ---------------------------------------------------------------------------
# Helper: per-(e, d) lifetime
# ---------------------------------------------------------------------------


def _ed_lifetime_lf(source: "InputSource",
                       all_entities_lf: pl.LazyFrame,
                       periods_lf: pl.LazyFrame,
                       ) -> pl.LazyFrame:
    """Lazy ``(e, d, life)`` mirroring ``edEntity_lifetime`` semantics.

    Mirrors :func:`._derived_params._ed_lifetime_mapping`: for each
    (e, d) ∈ entities × periods, emit ``unit/node/connection.lifetime``
    cascaded via ``_resolve_pdX``.  Non-process / non-node entities
    get 0 (mirrors flextool's class-gated branch).
    """
    ed_lf = all_entities_lf.join(periods_lf, how="cross")
    cls_lf = _entity_class_lookup_lf(_entity_class_lf(source))
    # Restrict to processes + nodes — other classes get life=0.
    ed_lf = ed_lf.join(cls_lf, on="e", how="left")
    in_class = pl.col("is_process").fill_null(False) | pl.col("is_node").fill_null(False)
    life_per = _per_entity_param_lf(source, "lifetime")
    resolved = _resolve_per_period_lf(life_per, ed_lf, fill=0.0)
    return (resolved
              .with_columns(
                  life=pl.when(in_class).then(pl.col("value")).otherwise(0.0),
              )
              .select("e", "d", "life"))


# ---------------------------------------------------------------------------
# Per-method annuity (lazy)
# ---------------------------------------------------------------------------


def _per_method_annuity_lf(
        source: "InputSource",
        ed_lf: pl.LazyFrame,
        cost_param_name: str,
        disallowed_methods: frozenset[str],
        ) -> pl.LazyFrame:
    """Lazy ``(e, d, ann)`` frame computing the per-method annuity sum.

    Algorithm (port of
    ``entity_annual_calc_params.py:177-217``):

      For each (e, d) ∈ ed_lf:
        ann[e, d] = Σ_{m ∈ invest_methods[e] \\ disallowed}
                       _annuity(cost[e, d], discount_rate[e, d], lifetime[e, d])
        (when e ∈ unit ∪ connection ∪ node)

    The inner computation does NOT depend on ``m`` — every allowed
    method contributes the same ``_annuity(...)`` value.  So we collapse:
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

    # Per-(e, d) cost / discount_rate / lifetime via _resolve_pdX.
    cost_per = _per_entity_param_lf(source, cost_param_name)
    disc_per = _per_entity_param_lf(source, "discount_rate")
    life_per = _per_entity_param_lf(source, "lifetime")

    ed_with_class = ed_lf.join(cls_lf, on="e", how="left")
    ed_with_cost = (_resolve_per_period_lf(cost_per, ed_with_class, fill=0.0)
                       .rename({"value": "cost"}))
    ed_with_disc = (_resolve_per_period_lf(disc_per, ed_with_cost, fill=0.0)
                       .rename({"value": "disc"}))
    ed_with_life = (_resolve_per_period_lf(life_per, ed_with_disc, fill=0.0)
                       .rename({"value": "life"}))

    # Annuity formula: invest_value × 1000 × r / (1 - (1/(1+r))^n)
    # with r ≤ 0 → 0.05, n ≤ 0 → 20.
    r_eff = (pl.when(pl.col("disc") > 0).then(pl.col("disc")).otherwise(0.05))
    n_eff = (pl.when(pl.col("life") > 0).then(pl.col("life")).otherwise(20.0))
    annuity_expr = (pl.when(r_eff == 0)
                       .then(0.0)
                       .otherwise(
                           pl.col("cost") * 1000.0 * r_eff
                           / (1.0 - (1.0 / (1.0 + r_eff)) ** n_eff)))
    out = (ed_with_life
              .join(method_count_lf, on="e", how="left")
              .with_columns(
                  method_count=pl.col("method_count").fill_null(0),
                  ann_per_method=annuity_expr,
              )
              .with_columns(
                  ann=pl.when(in_class)
                          .then(pl.col("ann_per_method")
                                  * pl.col("method_count").cast(pl.Float64))
                          .otherwise(0.0),
              )
              .select("e", "d", "ann"))
    return out


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
        ) -> pl.LazyFrame:
    """Variant 1: ``ed_entity_annual_discounted`` (invest-side NPV).

    Per (e ∈ entityInvest, d ∈ period_invest)::

        ann[e, d] × Σ_{d_all ∈ period_in_use, window(d, d_all)} inv_factor[d_all]

    where ``window`` switches between bounded / unbounded based on
    ``e``'s lifetime_method (``reinvest_choice`` / ``no_investment`` →
    bounded; ``reinvest_automatic`` → unbounded).  An entity may have
    multiple methods; each contributes a separate sum.  See
    ``entity_annual_calc_params.py:222-251``.

    Returns lazy ``[e, d, value]``.
    """
    if not period_invest:
        return pl.LazyFrame(schema={
            "e": schema_dtype(_enums, "e"),
            "d": schema_dtype(_enums, "d"),
            "value": pl.Float64,
        })
    # entityInvest = projection of entity__invest_method via the
    # method-allowed gate (mirror of preprocessing/invest_method_sets).
    entity_invest_set = _entity_invest_set_lf(source, _INVEST_NOT_ALLOWED)
    pi_lf = pl.LazyFrame({"d": period_invest}).with_columns(
        alias_to_axis("d", "d"))
    ed_anchor_lf = entity_invest_set.join(pi_lf, how="cross")

    # Annuity (invest cost).
    ann_lf = _per_method_annuity_lf(
        source, ed_anchor_lf, "invest_cost", _INVEST_NOT_ALLOWED)

    # Lifetime methods: per-entity dispatch.
    all_e_lf = ed_anchor_lf.select("e").unique()
    elm_lf = _lifetime_method_with_default_lf(source, all_e_lf)
    # Choice/no_investment → bounded; automatic → unbounded.  An entity
    # may have multiple methods; each contributes additively (mirrors
    # the if/if structure at L237/244).
    has_choice_or_no = (elm_lf
                          .filter(pl.col("method").is_in(
                              ["reinvest_choice", "no_investment"]))
                          .select("e").unique()
                          .with_columns(has_choice=pl.lit(True)))
    has_automatic = (elm_lf
                       .filter(pl.col("method") == "reinvest_automatic")
                       .select("e").unique()
                       .with_columns(has_automatic=pl.lit(True)))

    # life_lf for the bounded walk (uses edEntity_lifetime per L236).
    pwh_lf = pl.LazyFrame({"d": list({d for d in period_invest})}).with_columns(
        alias_to_axis("d", "d"))
    life_lf = _ed_lifetime_lf(source, all_e_lf, pwh_lf)

    bounded = period_walk_iterator(
        source, active_solve, ed_anchor_lf,
        period_in_use, period_universe,
        bounded=True, life_lf=life_lf, factor_side="inv")
    unbounded = period_walk_iterator(
        source, active_solve, ed_anchor_lf,
        period_in_use, period_universe,
        bounded=False, life_lf=life_lf, factor_side="inv")

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
                      factor_b=pl.col("factor_b").fill_null(0.0),
                      factor_u=pl.col("factor_u").fill_null(0.0),
                      has_choice=pl.col("has_choice").fill_null(False),
                      has_automatic=pl.col("has_automatic").fill_null(False),
                  )
                  .with_columns(
                      value=(
                          pl.when(pl.col("has_choice"))
                              .then(pl.col("ann") * pl.col("factor_b"))
                              .otherwise(0.0)
                          + pl.when(pl.col("has_automatic"))
                              .then(pl.col("ann") * pl.col("factor_u"))
                              .otherwise(0.0)
                      ),
                  )
                  .select("e", "d", "value"))
    return combined


def npv_divest_discounted_lf(
        source: "InputSource",
        active_solve: str | None,
        period_invest: list[str],
        period_in_use: list[str],
        period_universe: list[str],
        ) -> pl.LazyFrame:
    """Variant 2: ``ed_entity_annual_divest_discounted`` (divest-side NPV).

    Per (e ∈ entityDivest, d ∈ period_invest)::

        ann[e, d] × Σ_{d_all : pdy ∈ [pdy_d, pdy_d + life)} inv_factor[d_all]

    where ``life`` is the **raw** ``lifetime`` value (not edEntity_lifetime),
    and the gate is restricted to e ∈ node ∪ process (mirrors
    L266-285 — no fallback for unclassified entities).

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
        source, ed_anchor_lf, "salvage_value", _DIVEST_NOT_ALLOWED)

    # Use raw lifetime (per L271-284), not edEntity_lifetime.  Resolve
    # via _resolve_pdX with class gate (only nodes/processes).
    cls_lf = _entity_class_lookup_lf(_entity_class_lf(source))
    in_class = (pl.col("is_process").fill_null(False)
                | pl.col("is_node").fill_null(False))
    life_per = _per_entity_param_lf(source, "lifetime")
    ed_with_class = ed_anchor_lf.join(cls_lf, on="e", how="left")
    life_lf = (_resolve_per_period_lf(life_per, ed_with_class, fill=0.0)
                  .with_columns(
                      life=pl.when(in_class).then(pl.col("value")).otherwise(0.0),
                  )
                  .select("e", "d", "life"))

    walk = period_walk_iterator(
        source, active_solve, ed_anchor_lf,
        period_in_use, period_universe,
        bounded=True, life_lf=life_lf, factor_side="inv")

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
        ) -> pl.LazyFrame:
    """Variant 3: ``ed_lifetime_fixed_cost`` (invest-side lifetime FC).

    Per (e ∈ entity, d ∈ period_with_history)::

        fc[e, d] × Σ_{d_all ∈ period_in_use, window(d, d_all)} ops_factor[d_all]

    Same window dispatch as :func:`npv_invest_discounted_lf`
    (choice/no_invest → bounded; automatic → unbounded), with
    ops_factor (not inv_factor) and edEntity_lifetime as the bound.
    See L292-321.
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

    fc_lf = _ed_fixed_cost_raw_lf(source, ed_anchor_lf)

    elm_lf = _lifetime_method_with_default_lf(source, all_e_lf)
    has_choice_or_no = (elm_lf
                          .filter(pl.col("method").is_in(
                              ["reinvest_choice", "no_investment"]))
                          .select("e").unique()
                          .with_columns(has_choice=pl.lit(True)))
    has_automatic = (elm_lf
                       .filter(pl.col("method") == "reinvest_automatic")
                       .select("e").unique()
                       .with_columns(has_automatic=pl.lit(True)))

    # life_lf via edEntity_lifetime (per L306).
    life_lf = _ed_lifetime_lf(source, all_e_lf, pwh_lf)

    bounded = period_walk_iterator(
        source, active_solve, ed_anchor_lf,
        period_in_use, period_universe,
        bounded=True, life_lf=life_lf, factor_side="ops")
    unbounded = period_walk_iterator(
        source, active_solve, ed_anchor_lf,
        period_in_use, period_universe,
        bounded=False, life_lf=life_lf, factor_side="ops")

    bounded = bounded.rename({"factor": "factor_b"})
    unbounded = unbounded.rename({"factor": "factor_u"})

    return (ed_anchor_lf
              .join(fc_lf, on=["e", "d"], how="left")
              .join(bounded, on=["e", "d"], how="left")
              .join(unbounded, on=["e", "d"], how="left")
              .join(has_choice_or_no, on="e", how="left")
              .join(has_automatic, on="e", how="left")
              .with_columns(
                  fc=pl.col("fc").fill_null(0.0),
                  factor_b=pl.col("factor_b").fill_null(0.0),
                  factor_u=pl.col("factor_u").fill_null(0.0),
                  has_choice=pl.col("has_choice").fill_null(False),
                  has_automatic=pl.col("has_automatic").fill_null(False),
              )
              .with_columns(
                  value=(
                      pl.when(pl.col("has_choice"))
                          .then(pl.col("fc") * pl.col("factor_b"))
                          .otherwise(0.0)
                      + pl.when(pl.col("has_automatic"))
                          .then(pl.col("fc") * pl.col("factor_u"))
                          .otherwise(0.0)
                  ),
              )
              .select("e", "d", "value"))


def lifetime_fixed_cost_divest_lf(
        source: "InputSource",
        active_solve: str | None,
        period_invest: list[str],
        period_in_use: list[str],
        period_universe: list[str],
        ) -> pl.LazyFrame:
    """Variant 4: ``ed_lifetime_fixed_cost_divest`` (divest-side FC).

    Per (e ∈ entityDivest, d ∈ period_invest)::

        fc[e, d] × Σ_{d_all : pdy ∈ [pdy_d, pdy_d + life)} inv_factor[d_all]

    where ``life`` is raw lifetime and ``inv_factor`` (NOT ops_factor)
    is used — mod L1651 asymmetry.  See L325-348.
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

    fc_lf = _ed_fixed_cost_raw_lf(source, ed_anchor_lf)

    # Raw lifetime (gated on class).
    cls_lf = _entity_class_lookup_lf(_entity_class_lf(source))
    in_class = (pl.col("is_process").fill_null(False)
                | pl.col("is_node").fill_null(False))
    life_per = _per_entity_param_lf(source, "lifetime")
    ed_with_class = ed_anchor_lf.join(cls_lf, on="e", how="left")
    life_lf = (_resolve_per_period_lf(life_per, ed_with_class, fill=0.0)
                  .with_columns(
                      life=pl.when(in_class).then(pl.col("value")).otherwise(0.0),
                  )
                  .select("e", "d", "life"))

    walk = period_walk_iterator(
        source, active_solve, ed_anchor_lf,
        period_in_use, period_universe,
        bounded=True, life_lf=life_lf, factor_side="inv")

    return (ed_anchor_lf
              .join(fc_lf, on=["e", "d"], how="left")
              .join(walk, on=["e", "d"], how="left")
              .with_columns(
                  fc=pl.col("fc").fill_null(0.0),
                  factor=pl.col("factor").fill_null(0.0),
                  value=pl.col("fc") * pl.col("factor"),
              )
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
                                    ) -> "Param | None":
    """Public entry: ``p_inflation_op[d]`` over ``period_in_use``.

    Lazy port of
    :func:`._derived_params.p_inflation_op_full_cascade_from_source`.
    Replaces the workdir read of ``solve_data/p_inflation_factor_operations_yearly.csv``
    when wired through :func:`apply_npv`.
    """
    if not period_in_use:
        return None
    if period_universe is None:
        period_universe = period_in_use
    factors_lf = _inflation_factors_lf(source, active_solve, period_universe)
    out = (pl.LazyFrame({"d": period_in_use})
              .with_columns(alias_to_axis("d", "d"))
              .join(factors_lf, on="d", how="left")
              .with_columns(
                  value=pl.col("ops_factor").fill_null(1.0),
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
        ) -> "Param | None":
    """Public entry: ``ed_entity_annual_discounted[e, d]``.

    Returns the full ``entityInvest × period_invest`` frame, including
    zero-valued rows, to mirror the snapshot CSV produced by
    :func:`._emit_entity_annual.write_entity_annual_calc_params`
    (which the loader's ``_read_e_d`` seed retains unfiltered).  The
    loader's ``apply_npv`` only overwrites the seed when this entry
    returns a non-None frame, so emitting the zero rows is required
    for the lazy result to match the seed on fixtures where every
    (e, d) value is zero (e.g. ``work_wind_battery_invest``).
    """
    if not period_invest:
        return None
    out = (npv_invest_discounted_lf(
              source, active_solve,
              period_invest, period_in_use, period_universe)
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
        ) -> "Param | None":
    """Public entry: ``ed_entity_annual_divest_discounted[e, d]``.

    Unfiltered ``entityDivest × period_invest`` — see the rationale on
    :func:`ed_entity_annual_discounted_from_source`.
    """
    if not period_invest:
        return None
    out = (npv_divest_discounted_lf(
              source, active_solve,
              period_invest, period_in_use, period_universe)
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
        ) -> "Param | None":
    """Public entry: ``ed_lifetime_fixed_cost[e, d]``.

    Unfiltered ``entity × period_with_history`` — see the rationale on
    :func:`ed_entity_annual_discounted_from_source`.
    """
    if not period_with_history:
        return None
    out = (lifetime_fixed_cost_invest_lf(
              source, active_solve,
              period_with_history, period_in_use, period_universe)
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
        ) -> "Param | None":
    """Public entry: ``ed_lifetime_fixed_cost_divest[e, d]``.

    Unfiltered ``entityDivest × period_invest`` — see the rationale on
    :func:`ed_entity_annual_discounted_from_source`.
    """
    if not period_invest:
        return None
    out = (lifetime_fixed_cost_divest_lf(
              source, active_solve,
              period_invest, period_in_use, period_universe)
              .select("e", "d", "value")
              .sort("e", "d")
              .collect())
    if out.height == 0:
        return None
    return Param(("e", "d"), out)


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

    Δ.12b — assignment is unconditional (no silent fall-through to a
    seed value).  Each helper raises on hard errors and returns
    ``None`` only when the upstream feature is genuinely inactive
    (no_invest short-circuit, no period_with_history, etc.) — that
    ``None`` is the explicit "field stays default" signal.
    """
    from ._derived_params import (
        _read_active_solve, _solve_periods, _period_in_use_set,
        _periodAll_from_source, _read_period_with_history,
    )

    active_solve = _read_active_solve(workdir, provider=provider)
    period_in_use = _period_in_use_set(source, active_solve, workdir, provider=provider)
    period_universe = _periodAll_from_source(source, active_solve, workdir=workdir, provider=provider)
    period_invest = _solve_periods(source, active_solve, "invest_periods") or []
    period_with_history = (_read_period_with_history(workdir, provider=provider)
                              or list(period_in_use))

    # 1. p_inflation_op (None == no inflation data → field stays default).
    flex_data.p_inflation_op = p_inflation_op_from_source(
        source, active_solve, period_in_use, period_universe)

    # 2. p_ed_fixed_cost — None == no period_with_history (single-solve
    #    fixtures with no historical periods); legitimate "no data" outcome.
    flex_data.p_ed_fixed_cost = p_ed_fixed_cost_from_source(
        source, period_with_history)

    # 3-6. NPV / lifetime cascade family.  Mirrors the "no_invest"
    # short-circuit in :mod:`._derived_params.ed_entity_annual_family_from_source`
    # and :func:`flextool.engine_polars.input._load_invest`: when
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
        return

    # Δ.18 — when the override returns None (e.g. synthetic per-sub-solve
    # ``active_solve`` whose ``invest_periods`` is empty in Spine), preserve
    # the seed-loaded snapshot CSV value rather than overwriting it.  The
    # override remains authoritative when it returns a non-None Param.
    def _set_if(field: str, value):
        if value is not None:
            setattr(flex_data, field, value)

    _set_if("ed_entity_annual_discounted",
            ed_entity_annual_discounted_from_source(
                source, active_solve,
                period_invest, period_in_use, period_universe))

    _set_if("ed_entity_annual_divest_discounted",
            ed_entity_annual_divest_discounted_from_source(
                source, active_solve,
                period_invest, period_in_use, period_universe))

    _set_if("ed_lifetime_fixed_cost",
            ed_lifetime_fixed_cost_from_source(
                source, active_solve,
                period_with_history, period_in_use, period_universe))

    _set_if("ed_lifetime_fixed_cost_divest",
            ed_lifetime_fixed_cost_divest_from_source(
                source, active_solve,
                period_invest, period_in_use, period_universe))


__all__ = [
    "period_walk_iterator",
    "p_inflation_op_from_source",
    "p_ed_fixed_cost_from_source",
    "ed_entity_annual_discounted_from_source",
    "ed_entity_annual_divest_discounted_from_source",
    "ed_lifetime_fixed_cost_from_source",
    "ed_lifetime_fixed_cost_divest_from_source",
    "apply_npv",
]
