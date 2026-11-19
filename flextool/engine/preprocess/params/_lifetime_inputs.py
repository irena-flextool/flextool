"""The per-entity inputs the lifetime costs share (the investment price, the fixed costs, the reported costs).

The entity classes and the per-entity readers of a parameter (scalar or period map, with its default), the
``invest_method`` / ``lifetime_method`` of each entity, the entities that may invest or retire, the lifetime at the
period a decision comes online and the raw fixed cost; the per-period accumulation walk of the lifetime costs
(:func:`period_walk_iterator`) and its window to the end of life (:func:`_window_life_to_end_of_life`); the
registration of the investment and retirement lumps set as values (:func:`_register_invest_lumps`).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polar_high import Param

from flextool.engine.input._axis_enums import alias_to_axis, schema_dtype
from flextool.engine.input._lifetime_roles import LIFETIME_TECHNICAL
from flextool.engine.preprocess.params._branch import inherit_active_branch_periods
from flextool.engine.preprocess.params._period_values import ed_values_online

# Substrate handle for the cascade-wide axis enum vocabulary (``None``: ``cast_dim`` / ``schema_dtype`` fall
# back to the live ContextVar ``_LIVE_AXIS_ENUMS_CTX``; see ``_npv``).
_enums: "dict | None" = None

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource


# Default lifetime_method when an entity has no explicit row
# (preprocessing/method_with_fallback_sets.py::_LIFETIME_METHOD_DEFAULT).
_LIFETIME_METHOD_DEFAULT = "reinvest_automatic"


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
    ``year_rows`` (:func:`._discount.solve_year_rows`).
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


def _entity_invest_set_lf(source: "InputSource",
                              disallowed_methods: frozenset[str],
                              ) -> pl.LazyFrame:
    """Lazy ``[e]`` for the canonical entityInvest / entityDivest set.

    Mirrors flextool's
    ``preprocessing/invest_method_sets.py::write_invest_method_sets``:

        entityInvest = {e : ∃m s.t. (e, m) ∈ entity__invest_method
                                AND m ∉ invest_method_not_allowed}

    ``disallowed_methods`` selects between the invest variant
    (:data:`INVEST_NOT_ALLOWED`) and the divest variant
    (:data:`DIVEST_NOT_ALLOWED`).  An entity with no explicit
    ``invest_method`` row is **excluded** (default is ``not_allowed``).
    """
    methods_lf = _entity_method_lf(source, "invest_method")
    return (methods_lf
              .filter(~pl.col("method").is_in(list(disallowed_methods)))
              .select("e")
              .unique())


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


# ---------------------------------------------------------------------------
# Per-entity-class helpers (lazy)
# ---------------------------------------------------------------------------


def _all_entities_lf(source: "InputSource") -> pl.LazyFrame:
    """Lazy ``[e]`` frame: union of unit + node + connection."""
    return _entity_class_lf(source).select("e").unique()


def _utf8_ed(lf: pl.LazyFrame, cols: tuple[str, ...] = ("e", "d")) -> pl.DataFrame:
    return lf.with_columns(pl.col(c).cast(pl.Utf8) for c in cols).collect()


def _ed_param(df: pl.DataFrame) -> "Param | None":
    if df.height == 0:
        return None
    return Param(("e", "d"), df.lazy().select(alias_to_axis("e", "e"), alias_to_axis("d", "d"), "value")
                                     .sort("e", "d").collect())


#: The investment lumps whose ``solve_data`` frames are the values set here (no CSV emitter derives them).
_REGISTERED_LUMPS = ("ed_entity_annual_discounted", "ed_lifetime_fixed_cost")
#: The retirement lump whose ``solve_data`` frame is the value set by :func:`._fixed_costs.apply_retirement_fixed_cost`.
_REGISTERED_RETIREMENT_LUMPS = ("ed_lifetime_fixed_cost_divest",)


def _register_invest_lumps(flex_data: object, provider: "object | None",
                           names: tuple[str, ...] = _REGISTERED_LUMPS) -> None:
    """Register the lumps ``names`` of ``flex_data`` (default ``ed_entity_annual_discounted`` /
    ``ed_lifetime_fixed_cost``) under their ``solve_data/<name>`` provider keys (``entity, period, value``, the value
    rendered with ``repr``), so that the solve's frames of these lumps are the objective's coefficients: one
    derivation, with the start of operation of a construction lead time and the years in service."""
    if provider is None:
        return
    from flextool.engine.preprocess._provider_io import _emit
    from flextool.engine.preprocess.emit._vectorize import _render_value_column
    for name in names:
        param = getattr(flex_data, name, None)
        if param is None:
            continue
        frame = _utf8_ed(param.frame.lazy())
        _emit(provider, f"solve_data/{name}.csv",
              pl.DataFrame([frame["e"].alias("entity"), frame["d"].alias("period"),
                            _render_value_column(frame["value"])]))


__all__ = [
    "period_walk_iterator",
]
