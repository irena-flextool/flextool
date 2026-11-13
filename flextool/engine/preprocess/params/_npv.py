"""Cluster A — annual integration / NPV (Δ.5).

Lazy-polars port of flextool's NPV / lifetime-fixed-cost / inflation
cascade.  Replaces the eager-Python implementation that previously lived
in :mod:`._derived` (``ed_entity_annual_family_from_source`` plus
its supporting ``_resolve_pdX`` helper).

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
whole run, the undiscounted annuities ``ed_annuity`` / ``ed_divest_annuity``
and the operating / retirement / salvage windows ``edd_operating_window`` /
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
    ``invest_value × 1000 × annuity_factor(r, n)`` (:mod:`.._annuity`:
    ``r / (1 - (1+r)^-n)``, ``1/n`` at ``r = 0``).  A missing ``r`` takes
    the schema default (0.05); a missing lifetime where the cost is not 0
    is an error.
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
    schema_dtype,
)
from flextool.engine.preprocess._annuity import (
    annuity_factor_expr,
    default_discount_rate,
    require_lifetime,
)
from flextool.engine.preprocess.params._branch import inherit_active_branch_periods
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

    Mirrors :func:`._derived._resolve_pdX` semantics in lazy form:

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

    Order (mirrors :func:`._derived._resolve_pdX`):
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


def _inflation_factors_lf(source: "InputSource",
                                active_solve: str | None,
                                period_universe: list[str],
                                *,
                                provider: "object | None" = None,
                                year_rows: pl.DataFrame | None = None,
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

    ``year_rows`` (``[d, width, base]``, one row per represented year of a period): the years given explicitly
    instead of from ``years_for_period``; ``base`` is the number of years from the discount base to the start of
    that year, so ``until = base + width × offset``.  A period without rows (or with zero total width) gets 1.0.
    """
    if not period_universe:
        return pl.LazyFrame(schema={
            "d": schema_dtype(_enums, "d"),
            "inv_factor": pl.Float64, "ops_factor": pl.Float64,
        })
    rate, off_inv, off_ops = _solve_inflation_scalars(source)
    one_plus_inv = (1.0 / (1.0 + rate)) if rate != -1.0 else 1.0
    if year_rows is not None:
        explicit = (year_rows.lazy()
                      .select(pl.col("d").cast(pl.Utf8), pl.col("width").cast(pl.Float64),
                              pl.col("base").cast(pl.Float64))
                      .with_columns(
                          inv_contrib=pl.col("width") * (one_plus_inv ** (pl.col("base")
                                                                          + pl.col("width") * off_inv)),
                          ops_contrib=pl.col("width") * (one_plus_inv ** (pl.col("base")
                                                                          + pl.col("width") * off_ops)))
                      .group_by("d")
                      .agg(pl.col("inv_contrib").sum().alias("inv_factor"),
                           pl.col("ops_contrib").sum().alias("ops_factor"),
                           pl.col("width").sum().alias("sum_pyr"))
                      .filter(pl.col("sum_pyr") > 0)
                      .select("d", "inv_factor", "ops_factor"))
        return (pl.LazyFrame({"d": [str(d) for d in period_universe]})
                  .join(explicit, on="d", how="left")
                  .with_columns(alias_to_axis("d", "d"),
                                inv_factor=pl.col("inv_factor").fill_null(1.0),
                                ops_factor=pl.col("ops_factor").fill_null(1.0)))
    yfp_lf = _years_for_period_lf(
        source, active_solve, period_universe, provider=provider)

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


def _ed_lifetime_lf(source: "InputSource",
                       all_entities_lf: pl.LazyFrame,
                       periods_lf: pl.LazyFrame,
                       ) -> pl.LazyFrame:
    """Lazy ``(e, d, life)`` mirroring ``edEntity_lifetime`` semantics.

    Mirrors :func:`._derived._ed_lifetime_mapping`: for each
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


def _ed_lifetime_on_lf(source: "InputSource", ed_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Lazy ``(e, d, life)`` of :func:`_ed_lifetime_lf` over the given ``(e, d)`` pairs."""
    cls_lf = _entity_class_lookup_lf(_entity_class_lf(source))
    in_class = pl.col("is_process").fill_null(False) | pl.col("is_node").fill_null(False)
    resolved = _resolve_per_period_lf(_per_entity_param_lf(source, "lifetime"),
                                      ed_lf.select("e", "d").join(cls_lf, on="e", how="left"), fill=0.0)
    return (resolved
              .with_columns(life=pl.when(in_class).then(pl.col("value")).otherwise(0.0))
              .select("e", "d", "life"))


def _raw_lifetime_on_lf(source: "InputSource", ed_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Lazy ``(e, d, life)``: the raw ``lifetime`` (not edEntity_lifetime) over the given ``(e, d)`` pairs, resolved
    via ``_resolve_pdX`` with the class gate (only nodes and processes; others 0).  The divest-side window."""
    cls_lf = _entity_class_lookup_lf(_entity_class_lf(source))
    in_class = (pl.col("is_process").fill_null(False)
                | pl.col("is_node").fill_null(False))
    life_per = _per_entity_param_lf(source, "lifetime")
    ed_with_class = ed_lf.join(cls_lf, on="e", how="left")
    return (_resolve_per_period_lf(life_per, ed_with_class, fill=0.0)
              .with_columns(
                  life=pl.when(in_class).then(pl.col("value")).otherwise(0.0),
              )
              .select("e", "d", "life"))


def _at_start_of_operation(anchor_lf: pl.LazyFrame,
                              compute,
                              commission: pl.DataFrame | None,
                              col: str,
                              ) -> pl.LazyFrame:
    """``compute(anchor_lf)`` (``[e, d, col]``) with the value of a decision with a construction lead time taken at
    the period its capacity starts to operate (``commission`` ``[e, d, d_c]``): invest cost, interest rate,
    lifetime and fixed cost of new capacity are those of the build period.  Without ``commission`` it is
    ``compute(anchor_lf)`` itself."""
    base = compute(anchor_lf)
    if commission is None or commission.height == 0:
        return base
    schema = anchor_lf.collect_schema()
    e_dt, d_dt = schema.get("e", pl.Utf8), schema.get("d", pl.Utf8)
    cmap = (commission.lazy()
              .select(pl.col("e").cast(pl.Utf8).cast(e_dt, strict=False),
                      pl.col("d").cast(pl.Utf8).cast(d_dt, strict=False),
                      pl.col("d_c").cast(pl.Utf8).cast(d_dt, strict=False)))
    at_start = (compute(cmap.select("e", pl.col("d_c").alias("d")).unique())
                  .select("e", pl.col("d").alias("d_c"), pl.col(col).alias("_at_start")))
    return (base
              .join(cmap, on=["e", "d"], how="left")
              .join(at_start, on=["e", "d_c"], how="left")
              .with_columns(pl.coalesce("_at_start", col).alias(col))
              .select("e", "d", col))


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
                       cost[e, d] · 1000 · annuity_factor(discount_rate[e, d], lifetime[e, d])
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

    # Per-(e, d) cost / discount_rate / lifetime via _resolve_pdX.
    cost_per = _per_entity_param_lf(source, cost_param_name)
    disc_per = _per_entity_param_lf(source, "discount_rate")
    life_per = _per_entity_param_lf(source, "lifetime")

    ed_with_class = ed_lf.join(cls_lf, on="e", how="left")
    ed_with_cost = (_resolve_per_period_lf(cost_per, ed_with_class, fill=0.0)
                       .rename({"value": "cost"}))
    # A missing discount_rate is the schema default (0.05); an explicit 0 stays 0 (straight-line annuity).
    ed_with_disc = (_resolve_per_period_lf(disc_per, ed_with_cost,
                                           fill=default_discount_rate(source))
                       .rename({"value": "disc"}))
    # No lifetime default: a null (or ≤ 0) lifetime where the cost is not 0 is an input error.  Checked here,
    # not only in the validator: a period map can lack a period, and the lead-time path calls this again at the
    # commissioning period.
    ed_with_life = (_resolve_per_period_lf(life_per, ed_with_disc, fill=None)
                       .rename({"value": "life"}))
    bad = (ed_with_life
              .join(method_count_lf, on="e", how="inner")
              .filter(in_class & (pl.col("cost") != 0)
                      & (pl.col("life").is_null() | (pl.col("life") <= 0)))
              .select(pl.col("e").cast(pl.Utf8), pl.col("d").cast(pl.Utf8))
              .head(5)
              .collect())
    require_lifetime(bad.iter_rows(), cost_param_name)
    _notice_discount_rate_zero()

    # Annuity: cost [CUR/kW] × 1000 × annuity_factor(r, n).
    annuity_expr = (pl.col("cost") * 1000.0
                    * annuity_factor_expr(pl.col("disc"), pl.col("life")))
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
        *,
        lineage: pl.DataFrame | None = None,
        commission: pl.DataFrame | None = None,
        year_rows: pl.DataFrame | None = None,
        ) -> pl.LazyFrame:
    """Variant 1: ``ed_entity_annual_discounted`` (invest-side NPV).

    Per (e ∈ entityInvest, d ∈ period_invest)::

        ann[e, d] × Σ_{d_all ∈ period_in_use, window(d, d_all)} inv_factor[d_all]

    ``commission`` (``[e, d, d_c, yr_c]``, the decisions with a construction lead time): the annuity of such a
    decision is that of its build period ``d_c`` (invest cost, interest rate and lifetime there) and its window
    starts at ``yr_c``, so it is paid only over the years the capacity operates within the solve.

    where ``window`` switches between bounded / unbounded based on
    ``e``'s lifetime_method (``reinvest_choice`` / ``no_investment`` →
    bounded; ``reinvest_automatic`` → unbounded).  An entity may have
    multiple methods; each contributes a separate sum.  See
    ``entity_annual_calc_params.py:222-251``.

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
    # entityInvest = projection of entity__invest_method via the
    # method-allowed gate (mirror of preprocessing/invest_method_sets).
    entity_invest_set = _entity_invest_set_lf(source, _INVEST_NOT_ALLOWED)
    pi_lf = pl.LazyFrame({"d": period_invest}).with_columns(
        alias_to_axis("d", "d"))
    ed_anchor_lf = entity_invest_set.join(pi_lf, how="cross")

    # Annuity (invest cost), of the build period for a decision with a lead time.
    ann_lf = _at_start_of_operation(
        ed_anchor_lf,
        lambda lf: _per_method_annuity_lf(source, lf, "invest_cost", _INVEST_NOT_ALLOWED),
        commission, "ann")

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
    if commission is None or commission.height == 0:
        life_lf = _ed_lifetime_lf(source, all_e_lf, pwh_lf)
    else:
        life_lf = _at_start_of_operation(
            all_e_lf.join(pwh_lf, how="cross"), lambda lf: _ed_lifetime_on_lf(source, lf), commission, "life")
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


def _retirement_expired(source: "InputSource",
                        active_solve: str | None,
                        periods: list[str],
                        workdir: Path | None,
                        provider: "object | None",
                        ) -> pl.DataFrame:
    """``[e, d]`` (Utf8) of ``periods`` past the end of life of the entity's pre-existing capacity
    (:func:`._derived._lifetime_expired_pairs`): a retirement no longer counts there."""
    from flextool.engine.preprocess.params._derived import _lifetime_expired_pairs
    pairs = sorted(_lifetime_expired_pairs(source, active_solve, workdir, provider=provider,
                                           periods=[str(d) for d in periods]))
    return pl.DataFrame({"e": [e for e, _ in pairs], "d_all": [d for _, d in pairs]},
                        schema={"e": pl.Utf8, "d_all": pl.Utf8})


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
    yr[d] ≤ yr[d_all] < EOL[e] (the window of ``edd_divest_active``), along the walk lineage of a fan."""
    walk = period_walk_iterator(source, active_solve, ed_anchor_lf, period_in_use, period_universe,
                                bounded=False, life_lf=None, factor_side=None, lineage=lineage)
    walk = walk.select(pl.col("e").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), pl.col("d_all").cast(pl.Utf8))
    expired = _retirement_expired(source, active_solve, period_in_use, workdir, provider)
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

    where ``life`` is the **raw** ``lifetime`` value (not edEntity_lifetime),
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
        source, ed_anchor_lf, "salvage_value", _DIVEST_NOT_ALLOWED)

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
        ) -> pl.LazyFrame:
    """Variant 3: ``ed_lifetime_fixed_cost`` (invest-side lifetime FC).

    ``commission`` (``[e, d, d_c, yr_c]``): a decision with a construction lead time pays the fixed cost (and
    takes the lifetime) of its build period ``d_c``, over the years from ``yr_c`` its capacity operates.

    Per (e ∈ entity, d ∈ period_with_history)::

        fc[e, d] × Σ_{d_all ∈ period_in_use, window(d, d_all)} ops_factor[d_all]

    Same window dispatch as :func:`npv_invest_discounted_lf`
    (choice/no_invest → bounded; automatic → unbounded), with
    ops_factor (not inv_factor) and edEntity_lifetime as the bound.
    See L292-321.

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

    fc_lf = _at_start_of_operation(ed_anchor_lf, lambda lf: _ed_fixed_cost_raw_lf(source, lf), commission, "fc")

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

    # life_lf via edEntity_lifetime (per L306); of the build period for a decision with a lead time.
    if commission is None or commission.height == 0:
        life_lf = _ed_lifetime_lf(source, all_e_lf, pwh_lf)
    else:
        life_lf = _at_start_of_operation(ed_anchor_lf, lambda lf: _ed_lifetime_on_lf(source, lf), commission, "life")
    commission_lf = _commission_walk_lf(commission)

    bounded = period_walk_iterator(
        source, active_solve, ed_anchor_lf,
        period_in_use, period_universe,
        bounded=True, life_lf=life_lf, factor_side="ops",
        lineage=lineage, commission_lf=commission_lf, year_rows=year_rows)
    unbounded = period_walk_iterator(
        source, active_solve, ed_anchor_lf,
        period_in_use, period_universe,
        bounded=False, life_lf=life_lf, factor_side="ops",
        lineage=lineage, commission_lf=commission_lf, year_rows=year_rows)

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
        ) -> "Param | None":
    """Public entry: ``ed_entity_annual_discounted[e, d]``.

    Returns the full ``entityInvest × period_invest`` frame, including
    zero-valued rows, to mirror the snapshot CSV produced by
    :func:`._entity_annual.write_entity_annual_calc_params`
    (which the loader's ``_read_e_d`` seed retains unfiltered).  The
    loader's ``apply_npv`` only overwrites the seed when this entry
    returns a non-None frame, so emitting the zero rows is required
    for the lazy result to match the seed on fixtures where every
    (e, d) value is zero (e.g. ``work_wind_battery_invest``).

    ``lineage`` — see :func:`._walks.period_walk_iterator`; the
    :func:`._branch.annuity_window_lineage_df` frame in a stochastic fan, ``None`` otherwise.
    """
    if not period_invest:
        return None
    out = (npv_invest_discounted_lf(
              source, active_solve,
              period_invest, period_in_use, period_universe,
              lineage=lineage, commission=commission, year_rows=year_rows)
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
              lineage=lineage, commission=commission, year_rows=year_rows)
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
#   operates: yr_run[d] ≥ yr_run[c(d_invest)], and < yr_run[c(d_invest)] + lifetime[e, c(d_invest)] for
#   reinvest_choice / no_investment (unbounded for reinvest_automatic); c(d) is the start of operation (d without
#   a construction lead time).  edd_retire_window[e,d_divest,d]: yr_run[d_divest] ≤ yr_run[d] < EOL[e] (the end
#   of life of the pre-existing capacity, from the start of the run; the end of the run for reinvest_automatic),
#   the window of edd_divest_active.  edd_salvage_window[e,d_divest,d]: yr_run[d] ∈ [yr_run[d_divest],
#   yr_run[d_divest] + lifetime[e, d_divest]) (raw lifetime), the window the salvage value is spread over.  In a
#   stochastic fan only the pairs of the walk lineage (each calendar period once).
# OUT. ed_annuity[e,d] = annuity (CUR/MW/yr, undiscounted) of invest_cost at c(d); ed_divest_annuity[e,d] the same
#   of salvage_value at d.
# OUT. ed_entity_annual_discounted_run[e,d] = ed_annuity[e,d]·Σ_{(e,d,d') ∈ edd_operating_window} p_inflation_inv_run[d']
#      ed_lifetime_fixed_cost_run[e,d] = Σ_{(e,d,d') ∈ edd_operating_window} fc[e,c(d)]·p_inflation_op_run[d']
#      ed_entity_annual_divest_discounted_run[e,d] = ed_divest_annuity[e,d]·Σ_{edd_salvage_window} p_inflation_inv_run[d']
#      ed_lifetime_fixed_cost_divest_run[e,d] = Σ_{(e,d,d') ∈ edd_retire_window} fc[e,d']·p_inflation_op_run[d']
#   The divest lifetime fixed cost is the fixed cost of existing capacity it saves: each window period's fixed
#   cost with the operations factor (as the fixed cost of existing capacity is charged), as its per-solve twin.
#   The salvage value keeps the annuity over lifetime and the window of the investment annuity (its per-solve
#   twin ed_entity_annual_divest_discounted likewise), so retiring existing capacity and building identical new
#   capacity earns nothing.
# Notes: when the run is the solve itself (no run calendar, or every run period is in use and the solve starts
#   the run) the factors and lumps ARE the per-solve fields (byte-identical); only the windows and annuities are
#   built.  An entity with both a bounded and an unbounded lifetime method adds both sums, as the per-solve lumps
#   do; edd_operating_window then holds the union of the two windows.

_RUN_BOUNDED_METHODS = ("reinvest_choice", "no_investment")


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
    """``[e, kind]``: ``bounded`` for an entity with a reinvest_choice / no_investment lifetime method, ``unbounded``
    for one with reinvest_automatic (the default); both for an entity with both."""
    elm = _utf8_ed(_lifetime_method_with_default_lf(
        source, entities.lazy().select(alias_to_axis("e", "e"))), ("e",))
    return (pl.concat([
                elm.filter(pl.col("method").is_in(list(_RUN_BOUNDED_METHODS))).select("e", kind=pl.lit("bounded")),
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
    """``[e, d, value]`` over ``anchors``: ``coef[e, d] × Σ_{kind} Σ_{d_all ∈ window(kind)} factor[d_all]``."""
    sums = (walk.join(factor, on="d_all", how="left")
                .group_by("e", "d")
                .agg(pl.col("factor").fill_null(1.0).sum().alias("sum")))
    return (anchors.join(coef, on=["e", "d"], how="left")
                   .join(sums, on=["e", "d"], how="left")
                   .select("e", "d", value=pl.col("coef").fill_null(0.0) * pl.col("sum").fill_null(0.0)))


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
        inv_lf = _anchor_lf(inv)
        ann = _utf8_ed(_at_start_of_operation(
            inv_lf, lambda lf: _per_method_annuity_lf(source, lf, "invest_cost", _INVEST_NOT_ALLOWED),
            commission, "ann")).rename({"ann": "coef"})
        flex_data.ed_annuity = _ed_param(vintages.join(ann, on=["e", "d"], how="left")
                                            .select("e", "d", value=pl.col("coef").fill_null(0.0)))
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
        life = _utf8_ed(_at_start_of_operation(
            fc_lf, lambda lf: _ed_lifetime_on_lf(source, lf), commission, "life"))
        anchors = (year_of_start(fc_pairs, True)
                     .join(life, on=["e", "d"], how="left")
                     .with_columns(life=pl.col("life").fill_null(0.0))
                     .join(_lifetime_kinds(source, fc_pairs.select("e").unique()), on="e", how="inner"))
        walk = _run_window(anchors, yr, lineage, period_in_use)
        inv_walk = walk.join(inv, on=["e", "d"], how="semi")
        flex_data.edd_operating_window = _window_frame(walk.join(vintages, on=["e", "d"], how="semi"), "d_invest")
        if years is not None:
            fc = _utf8_ed(_at_start_of_operation(
                fc_lf, lambda lf: _ed_fixed_cost_raw_lf(source, lf), commission, "fc")).rename({"fc": "coef"})
            flex_data.ed_lifetime_fixed_cost_run = _ed_param(_lump(walk, fc, factor_ops, fc_pairs))
            if inv.height > 0:
                flex_data.ed_entity_annual_discounted_run = _ed_param(_lump(inv_walk, ann, factor_inv, inv))

    # Retirement side: from the retirement period until the end of life of the existing capacity (the end of the
    # run for reinvest_automatic), the window of edd_divest_active on the run calendar.
    div = anchors_of("ed_divest_set", _DIVEST_NOT_ALLOWED, period_invest)
    if div.height > 0:
        from flextool.engine.preprocess.params._derived import _lifetime_gate_sums
        from flextool.engine.preprocess.params._run_calendar import YEAR_TOL
        div_lf = _anchor_lf(div)
        anchors = (year_of_start(div, False)
                     .with_columns(life=pl.lit(0.0), kind=pl.lit("unbounded")))
        walk = _run_window(anchors, yr, lineage, period_in_use)
        eol = _lifetime_gate_sums(source, active_solve, workdir, provider=provider)
        if eol:
            ends = pl.DataFrame({"e": list(eol), "end": [float(v) for v in eol.values()]},
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
        ann = _utf8_ed(_per_method_annuity_lf(source, div_lf, "salvage_value", _DIVEST_NOT_ALLOWED)
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


def apply_pre_existing_of_reported_costs(flex_data: object, provider: "object | None") -> None:
    """Set ``p_entity_pre_existing`` (e, d): the existing capacity given by the user, aged over its lifetime
    (``solve_data/p_entity_pre_existing``), without the capacity carried from earlier solves and without their
    retirements.  The reported fixed cost of existing capacity is charged on it: the carried capacity's fixed
    cost is in the lump of the solve that invested in it, a retirement's saving in the lump of the retirement."""
    from flextool.engine.preprocess.params._derived import _provider_has_key, _provider_read
    path = "solve_data/p_entity_pre_existing.csv"
    if not _provider_has_key(provider, path):
        return
    df = _provider_read(provider, path)
    if df is None or df.height == 0 or not {"entity", "period", "value"} <= set(df.columns):
        return
    frame = (df.lazy()
               .select(alias_to_axis(pl.col("entity").cast(pl.Utf8), "e"),
                       alias_to_axis(pl.col("period").cast(pl.Utf8), "d"),
                       pl.col("value").cast(pl.Utf8).cast(pl.Float64, strict=False).fill_null(0.0))
               .sort("e", "d").collect())
    flex_data.p_entity_pre_existing = Param(("e", "d"), frame) if frame.height > 0 else None


def apply_run_fields_of_synthetic_solve(flex_data: object,
                                        source: "InputSource",
                                        workdir: Path,
                                        *,
                                        provider: "object | None" = None,
                                        ) -> None:
    """The run-level fields of a synthetic sub-solve (a roll of a rolling solve, a nested ``<solve>_<period>``
    sub-solve), whose own cost fields are the per-solve seeds emitted for it (``apply_npv`` does not run).
    Without run years the run fields are those seeds."""
    from flextool.engine.preprocess.params._branch import annuity_window_lineage_df
    from flextool.engine.preprocess.params._derived import (
        _period_in_use_set,
        _provider_has_key,
        _provider_read,
        _read_active_solve,
        _read_period_with_history,
    )
    active_solve = _read_active_solve(workdir, provider=provider)
    apply_pre_existing_of_reported_costs(flex_data, provider)
    period_in_use = list(_period_in_use_set(source, active_solve, workdir, provider=provider))
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
    apply_run_cost_fields(flex_data, source, active_solve, period_in_use, period_invest, period_with_history,
                          workdir, lineage=annuity_window_lineage_df(workdir, source, active_solve,
                                                                      provider=provider),
                          years=years, provider=provider)


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

    apply_pre_existing_of_reported_costs(flex_data, provider)

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

    # Δ.18 — when the override returns None (e.g. synthetic per-sub-solve
    # ``active_solve`` whose ``invest_periods`` is empty in Spine), preserve
    # the seed-loaded snapshot CSV value rather than overwriting it.  The
    # override remains authoritative when it returns a non-None Param.
    def _set_if(field: str, value):
        if value is not None:
            setattr(flex_data, field, value)

    # Construction lead time: the decisions' start of operation (apply_derived_c, which runs before).
    commission = getattr(flex_data, "ed_invest_commission", None)
    _set_if("ed_entity_annual_discounted",
            ed_entity_annual_discounted_from_source(
                source, active_solve,
                period_invest, period_in_use, period_universe,
                lineage=_npv_lineage, commission=commission, year_rows=year_rows))

    _set_if("ed_entity_annual_divest_discounted",
            ed_entity_annual_divest_discounted_from_source(
                source, active_solve,
                period_invest, period_in_use, period_universe,
                lineage=_npv_lineage, year_rows=year_rows))

    _set_if("ed_lifetime_fixed_cost",
            ed_lifetime_fixed_cost_from_source(
                source, active_solve,
                period_with_history, period_in_use, period_universe,
                lineage=_npv_lineage, commission=commission, year_rows=year_rows))

    _set_if("ed_lifetime_fixed_cost_divest",
            ed_lifetime_fixed_cost_divest_from_source(
                source, active_solve,
                period_invest, period_in_use, period_universe,
                lineage=_npv_lineage, workdir=workdir, provider=provider, year_rows=year_rows))

    # Reported costs: the lumps from the start of the run over the run's periods, and the windows (outputs only).
    apply_run_cost_fields(flex_data, source, active_solve, list(period_in_use), list(period_invest),
                          list(period_with_history), workdir, lineage=_npv_lineage, years=run_yrs,
                          provider=provider)


__all__ = [
    "period_walk_iterator",
    "p_inflation_op_from_source",
    "p_ed_fixed_cost_from_source",
    "ed_entity_annual_discounted_from_source",
    "ed_entity_annual_divest_discounted_from_source",
    "ed_lifetime_fixed_cost_from_source",
    "ed_lifetime_fixed_cost_divest_from_source",
    "apply_npv",
    "apply_run_cost_fields",
    "apply_run_factors",
    "apply_run_fields_of_synthetic_solve",
    "run_years",
    "notice_discount_from_run_start",
    "solve_year_rows",
    "year_rows_from_history",
]

