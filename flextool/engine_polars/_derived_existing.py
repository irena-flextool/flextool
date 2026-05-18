"""Cluster B — existing chain & invest sets (Δ.6).

Lazy-polars port of flextool's existing-capacity cascade and the
invest/divest set family.  Replaces the eager-Python helpers spread
across :mod:`._derived_params` (``p_entity_all_existing_from_source``,
``ed_invest_set_from_source`` / ``ed_divest_set_from_source``,
``edd_invest_lookback_set_from_source``,
``p_entity_max_units_from_source``, etc.) and the workdir-CSV reads
they currently depend on.

Cluster B fields (per
``audit/native_data_path_design_derived_clusters.md``):

* ``entityInvest`` / ``entityDivest`` — the (e,)-shaped entity
  projection of ``entity__invest_method`` filtered by the
  not-allowed-method constants
  (``preprocessing/invest_method_sets.py``).  These were the trap that
  Δ.5 caught: the eager helper derived them from
  ``ed_invest_set`` / ``ed_divest_set`` instead of from
  ``entity__invest_method``.  This module derives them correctly.
* ``e_invest_total`` / ``e_divest_total`` — projections of the above
  filtered by INVEST_TOTAL / RETIRE_TOTAL methods
  (``preprocessing/invest_total_sets.py``).
* ``ed_invest_set`` / ``ed_divest_set`` — (e, d) cross-product gated
  by entityInvest × period_invest, eea ≠ 0 OR capacity-constraint
  membership, and the no_investment lifetime-expiry forbidden set.
* ``pd_invest_set`` / ``nd_invest_set`` / ``pd_divest_set`` /
  ``nd_divest_set`` — partitions of ``ed_invest_set`` / ``ed_divest_set``
  by entity-class (``unit ∪ connection`` → process-side; ``node`` →
  node-side).
* ``edd_history`` (and its ``_choice`` / ``_automatic`` / ``_no_invest``
  components) — the (e, d_history, d) triple-set encoding the
  lifetime-window walk:
  ``pdy[d] >= pdy[d_history]`` with optional bounded gate
  ``pdy[d] < pdy[d_history] + edEntity_lifetime[e, d_history]``.
  Filtered to entities whose ``lifetime_method`` is in
  {``reinvest_choice``, ``reinvest_automatic``, ``no_investment``}.
* ``edd_invest_set`` — ``edd_history`` ∩ ``ed_invest_set`` on
  ``(e, d_history)`` (i.e. the history rows that correspond to a
  current-solve invest decision).
* ``edd_invest_lookback_set`` — strict-lookback variant
  (``yr[d_invest] < yr[d]``) used by the prebuilt-capacity LHS.
* ``edd_divest_active`` — ``pd_divest`` × period grid filtered to
  ``yr[d_divest] <= yr[d]`` (used in cap-margin user constraint).
* ``p_entity_all_existing`` — per-(e, d) lifetime-cumulative existing
  capacity.  Algorithm:
    * solve-first: pre-existing[e, d] (lifetime-gated entity.existing).
    * later-solves: Σ_{d_history ∈ edd_history(e, d) ∧ realized}
        p_entity_period_existing_capacity[e, d_history]
        − p_entity_divested[e]  (when e ∈ entityDivest).
* ``p_entity_period_existing_capacity`` — handoff carrier read from
  ``flex_data.p_entity_period_existing_capacity`` (already populated
  by the CSV-side handoff loader); the lazy path doesn't recompute it.
* ``p_entity_max_units`` — per-(e, d) max unit count via the
  unitsize cascade × the max_capacity branch fork.

The shared :func:`._derived_walks.period_walk_iterator` lifted out of
Cluster A's ``_derived_npv`` is consumed throughout for the lifetime-
window walks.

All helpers are lazy until the public ``apply_existing_chain`` boundary
calls ``.collect()`` once per emitted Param.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

from flextool.engine_polars._axis_enums import (
    alias_to_axis,
    get_global_axis_enums,
    rename_to_axis,
    schema_dtype,
)


# Substrate handle for the cascade-wide axis enum vocabulary.
# Bare ``None`` here; ``cast_dim`` / ``schema_dtype`` in
# ``_axis_enums`` fall back to ``_LIVE_AXIS_ENUMS_CTX`` (the live
# ContextVar) when this is ``None``, so substrate sites pick up
# activation set by ``load_flextool`` automatically.
_enums: "dict | None" = None

from ._derived_walks import period_walk_iterator, WindowMethod

if TYPE_CHECKING:
    from flextool.engine_polars._input_source import InputSource
    from flextool.engine_polars._solve_context import SolveContext


# -- Method-enum constants (mirror flextool/flextool_base.dat:211-212) -------

_INVEST_NOT_ALLOWED: frozenset[str] = frozenset((
    "not_allowed", "retire_period", "retire_total", "retire_no_limit",
))
_DIVEST_NOT_ALLOWED: frozenset[str] = frozenset((
    "not_allowed", "invest_period", "invest_total", "invest_no_limit",
))
# preprocessing/invest_total_sets.py:25-32
_INVEST_TOTAL_METHODS: frozenset[str] = frozenset((
    "invest_total", "invest_period_total",
    "invest_retire_total", "invest_retire_period_total",
))
_DIVEST_TOTAL_METHODS: frozenset[str] = frozenset((
    "retire_total", "retire_period_total",
    "invest_retire_total", "invest_retire_period_total",
))
# preprocessing/invest_divest_sets.py:33-40
_INVEST_PERIOD_METHODS: frozenset[str] = frozenset((
    "invest_period", "invest_period_total",
    "invest_retire_period", "invest_retire_period_total",
))
_DIVEST_PERIOD_METHODS: frozenset[str] = frozenset((
    "retire_period", "retire_period_total",
    "invest_retire_period", "invest_retire_period_total",
))

# preprocessing/method_with_fallback_sets.py::_LIFETIME_METHOD_DEFAULT
_LIFETIME_METHOD_DEFAULT = "reinvest_automatic"
_LIFETIME_BOUNDED_METHODS: frozenset[str] = frozenset((
    "reinvest_choice", "no_investment",
))
_LIFETIME_UNBOUNDED_METHODS: frozenset[str] = frozenset((
    "reinvest_automatic",
))


# ---------------------------------------------------------------------------
# Building blocks (lazy frames)
# ---------------------------------------------------------------------------


def _entity_class_lf(source: "InputSource") -> pl.LazyFrame:
    """Lazy ``[e, ec]``: every (entity, entity_class) pairing.

    Mirror of :func:`._derived_npv._entity_class_lf` (kept local to
    avoid cross-module coupling).
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
        return pl.LazyFrame(schema={
            "e": schema_dtype(_enums, "e"),
            "ec": pl.Utf8,
        })
    return pl.concat(parts, how="vertical")


def _all_entities_lf(source: "InputSource") -> pl.LazyFrame:
    """Lazy ``[e]`` — union of unit + node + connection."""
    return _entity_class_lf(source).select("e").unique()


def _entity_method_lf(source: "InputSource",
                          parameter_name: str,
                          ) -> pl.LazyFrame:
    """Lazy ``[e, method]`` from ``unit/node/connection.<param>``.

    ``parameter_name`` ∈ {``invest_method``, ``lifetime_method``}.
    Mirror of :func:`._derived_npv._entity_method_lf`.
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
        return pl.LazyFrame(schema={
            "e": schema_dtype(_enums, "e"),
            "method": pl.Utf8,
        })
    return pl.concat(parts, how="vertical")


def _lifetime_method_with_default_lf(
        source: "InputSource",
        all_entities_lf: pl.LazyFrame,
        ) -> pl.LazyFrame:
    """Lazy ``[e, method]`` for ``lifetime_method`` with default fill.

    Entities without an explicit ``lifetime_method`` row get
    ``reinvest_automatic``.  Mirror of
    :func:`._derived_npv._lifetime_method_with_default_lf`.
    """
    explicit = _entity_method_lf(source, "lifetime_method")
    explicit_entities = explicit.select("e").unique()
    default_e = (all_entities_lf
                   .join(explicit_entities, on="e", how="anti")
                   .with_columns(method=pl.lit(_LIFETIME_METHOD_DEFAULT)))
    return pl.concat([explicit, default_e], how="vertical")


def _per_entity_param_lf(source: "InputSource",
                            parameter_name: str,
                            ) -> pl.LazyFrame:
    """Lazy ``[e, d, value, is_scalar]``.

    Mirror of :func:`._derived_npv._per_entity_param_lf`.

    Period-dim detection: a per-period parameter exposes exactly one
    index column alongside ``name`` / ``value``.  The canonical name is
    ``period``, but the source may surface the column under the user's
    ``Map.index_name`` (e.g. ``x``) when the spinedb default is
    overridden.  Treat any single non-``name``/``value`` column as the
    period dim — checking only ``"period" in cols`` mis-classifies
    user-renamed indices as scalars, which then explode through
    :func:`_resolve_per_period_lf`'s ``scalar.join(on="e")``.
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
        # Any extra column beyond name / value is the per-period index.
        # ``existing`` / ``lifetime`` / ``virtual_unitsize`` have at
        # most one such extra column; fall back to ``period`` when
        # present (typical), otherwise pick the first non-name/value
        # column (covers ``x`` / other user-renamed Map indices).
        extra = [c for c in cols if c not in ("name", "value")]
        period_col = "period" if "period" in extra else (extra[0] if extra else None)
        if period_col is not None:
            parts.append(df.lazy().select(
                alias_to_axis("name", "e"),
                alias_to_axis(period_col, "d"),
                pl.col("value").cast(pl.Float64, strict=False),
                pl.lit(False).alias("is_scalar"),
            ))
        else:
            parts.append(df.lazy().select(
                alias_to_axis("name", "e"),
                pl.lit(None, dtype=schema_dtype(_enums, "d")).alias("d"),
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

    Mirror of :func:`._derived_npv._resolve_per_period_lf`.
    """
    # Align ``per_param``'s ``e`` / ``d`` dtypes to ``ed_lf``'s before
    # joining — ``per_param`` was built from a String-typed CSV read
    # while ``ed_lf`` may now carry Enum-cast columns.
    ed_schema = ed_lf.collect_schema()
    e_dt = ed_schema.get("e", pl.Utf8)
    d_dt = ed_schema.get("d", pl.Utf8)
    if e_dt != pl.Utf8 or d_dt != pl.Utf8:
        per_param = per_param.with_columns(
            pl.col("e").cast(e_dt, strict=False),
            pl.col("d").cast(d_dt, strict=False),
        )
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


# ---------------------------------------------------------------------------
# §3.7.0 — entityInvest / entityDivest / e_invest_total / e_divest_total
# ---------------------------------------------------------------------------


def entity_invest_set_lf(source: "InputSource") -> pl.LazyFrame:
    """Lazy ``[e]`` for ``entityInvest``.

    Mirrors flextool's
    ``preprocessing/invest_method_sets.py::write_invest_method_sets``::

        entityInvest = {e : ∃m s.t. (e, m) ∈ entity__invest_method
                                AND m ∉ invest_method_not_allowed}

    The Δ.5 ``ed_*``-derivation trap: an entity may have an
    ``invest_method`` row but no ``ed_invest_set`` row (e.g. eea == 0
    AND no capacity constraint).  Such an entity is in ``entityInvest``
    but not in any ``ed_invest`` (e, d) pair.  Cluster A's
    ``ed_lifetime_fixed_cost_divest`` produces rows for these entities;
    deriving from ``ed_*`` would miss them.
    """
    methods_lf = _entity_method_lf(source, "invest_method")
    return (methods_lf
              .filter(~pl.col("method").is_in(list(_INVEST_NOT_ALLOWED)))
              .select("e")
              .unique())


def entity_divest_set_lf(source: "InputSource") -> pl.LazyFrame:
    """Lazy ``[e]`` for ``entityDivest``.  Symmetric of
    :func:`entity_invest_set_lf` with the divest-not-allowed filter."""
    methods_lf = _entity_method_lf(source, "invest_method")
    return (methods_lf
              .filter(~pl.col("method").is_in(list(_DIVEST_NOT_ALLOWED)))
              .select("e")
              .unique())


def e_invest_total_lf(source: "InputSource") -> pl.LazyFrame:
    """Lazy ``[e]`` for ``e_invest_total``.

    Mirrors ``preprocessing/invest_total_sets.py:74-90``:

        e_invest_total = entityInvest filtered by invest_method ∈
                              {invest_total, invest_period_total,
                               invest_retire_total,
                               invest_retire_period_total}
    """
    methods_lf = _entity_method_lf(source, "invest_method")
    invest_universe = entity_invest_set_lf(source)
    has_total = (methods_lf
                    .filter(pl.col("method").is_in(list(_INVEST_TOTAL_METHODS)))
                    .select("e").unique())
    return invest_universe.join(has_total, on="e", how="inner")


def e_divest_total_lf(source: "InputSource") -> pl.LazyFrame:
    """Lazy ``[e]`` for ``e_divest_total``.  Symmetric of
    :func:`e_invest_total_lf` with the retire-total method filter."""
    methods_lf = _entity_method_lf(source, "invest_method")
    divest_universe = entity_divest_set_lf(source)
    has_total = (methods_lf
                    .filter(pl.col("method").is_in(list(_DIVEST_TOTAL_METHODS)))
                    .select("e").unique())
    return divest_universe.join(has_total, on="e", how="inner")


# ---------------------------------------------------------------------------
# §3.7.2 — edd_history triple-set walk + variants
# ---------------------------------------------------------------------------


def edd_history_choice_lf(source: "InputSource",
                                  active_solve: str | None,
                                  period_with_history: list[str],
                                  period_in_use: list[str],
                                  workdir: Path | None = None,
                                  ) -> pl.LazyFrame:
    """Lazy ``[e, d_history, d]`` for ``edd_history_choice``.

    Mirror of ``invest_divest_sets.py:228-244`` for entities whose
    ``lifetime_method = reinvest_choice``::

        keep iff pdy[d] >= pdy[d_h] AND pdy[d] < pdy[d_h] + life[e, d_h]
    """
    return _edd_history_lf_for(
        source, active_solve, period_with_history, period_in_use,
        method="reinvest_choice", bounded=True, workdir=workdir)


def edd_history_automatic_lf(source: "InputSource",
                                       active_solve: str | None,
                                       period_with_history: list[str],
                                       period_in_use: list[str],
                                       workdir: Path | None = None,
                                       ) -> pl.LazyFrame:
    """Lazy ``[e, d_history, d]`` for ``edd_history_automatic`` —
    entities with ``lifetime_method = reinvest_automatic``.

    Mirror of ``invest_divest_sets.py:245-246``::

        keep iff pdy[d] >= pdy[d_h]
    """
    return _edd_history_lf_for(
        source, active_solve, period_with_history, period_in_use,
        method="reinvest_automatic", bounded=False, workdir=workdir)


def edd_history_no_investment_lf(source: "InputSource",
                                            active_solve: str | None,
                                            period_with_history: list[str],
                                            period_in_use: list[str],
                                            workdir: Path | None = None,
                                            ) -> pl.LazyFrame:
    """Lazy ``[e, d_history, d]`` for ``edd_history_no_investment``.

    Mirror of ``invest_divest_sets.py:247-248``: same predicate as
    ``edd_history_choice`` but for the ``no_investment`` cohort.
    """
    return _edd_history_lf_for(
        source, active_solve, period_with_history, period_in_use,
        method="no_investment", bounded=True, workdir=workdir)


def _edd_history_lf_for(source: "InputSource",
                              active_solve: str | None,
                              period_with_history: list[str],
                              period_in_use: list[str],
                              *,
                              method: str,
                              bounded: bool,
                              workdir: Path | None = None,
                              ) -> pl.LazyFrame:
    """Internal helper: build edd_history sub-set for a single
    lifetime_method cohort.
    """
    if not period_with_history or not period_in_use:
        return pl.LazyFrame(schema={
            "e": schema_dtype(_enums, "e"),
            "d_history": schema_dtype(_enums, "d"),
            "d": schema_dtype(_enums, "d"),
        })
    all_e_lf = _all_entities_lf(source)
    elm_lf = _lifetime_method_with_default_lf(source, all_e_lf)
    cohort_e = (elm_lf
                   .filter(pl.col("method") == method)
                   .select("e").unique())
    pwh_lf = (pl.LazyFrame({"d": period_with_history})
                .with_columns(pl.col("d").cast(schema_dtype(_enums, "d"),
                                                  strict=False)))
    anchor = cohort_e.join(pwh_lf, how="cross")
    if anchor.collect().height == 0:
        return pl.LazyFrame(schema={
            "e": schema_dtype(_enums, "e"),
            "d_history": schema_dtype(_enums, "d"),
            "d": schema_dtype(_enums, "d"),
        })

    if bounded:
        life_per = _per_entity_param_lf(source, "lifetime")
        life_lf = (_resolve_per_period_lf(life_per, anchor, fill=0.0)
                       .rename({"value": "life"})
                       .select("e", "d", "life"))
        walk = period_walk_iterator(
            source, active_solve, anchor,
            period_in_use, period_in_use,
            window_method=WindowMethod.BOUNDED_INCLUSIVE_LOOKBACK,
            life_lf=life_lf, factor_side=None,
            workdir=workdir)
    else:
        walk = period_walk_iterator(
            source, active_solve, anchor,
            period_in_use, period_in_use,
            window_method=WindowMethod.UNBOUNDED_FORWARD,
            life_lf=None, factor_side=None,
            workdir=workdir)
    return walk.pipe(rename_to_axis, {"d": "d_history", "d_all": "d"})


def edd_history_lf(source: "InputSource",
                       active_solve: str | None,
                       period_with_history: list[str],
                       period_in_use: list[str],
                       workdir: Path | None = None,
                       ) -> pl.LazyFrame:
    """Lazy ``[e, d_history, d]`` for the union ``edd_history`` set.

    Algorithm (mirror of
    ``invest_divest_sets.py:227-262``): union of the three cohort
    walks (:func:`edd_history_choice_lf`,
    :func:`edd_history_automatic_lf`,
    :func:`edd_history_no_investment_lf`).
    """
    if not period_with_history or not period_in_use:
        return pl.LazyFrame(schema={
            "e": schema_dtype(_enums, "e"),
            "d_history": schema_dtype(_enums, "d"),
            "d": schema_dtype(_enums, "d"),
        })
    parts = [
        edd_history_choice_lf(source, active_solve,
                                  period_with_history, period_in_use,
                                  workdir),
        edd_history_automatic_lf(source, active_solve,
                                       period_with_history, period_in_use,
                                       workdir),
        edd_history_no_investment_lf(source, active_solve,
                                            period_with_history, period_in_use,
                                            workdir),
    ]
    return pl.concat(parts, how="vertical").unique()


# ---------------------------------------------------------------------------
# §3.7.1 — pd_invest_set / nd_invest_set partitions
# ---------------------------------------------------------------------------


def _entity_class_partition_lf(source: "InputSource") -> pl.LazyFrame:
    """Lazy ``[e, kind]`` where ``kind ∈ {process, node}``.

    process = unit ∪ connection (processes); node = node.  Disjoint
    classes — every entity falls in exactly one.
    """
    parts: list[pl.LazyFrame] = []
    for ec, kind in (("unit", "process"),
                       ("connection", "process"),
                       ("node", "node")):
        try:
            df = source.entities(ec)
        except KeyError:
            continue
        if df.height == 0:
            continue
        parts.append(df.lazy().select(
            alias_to_axis("name", "e"),
            pl.lit(kind).alias("kind"),
        ))
    if not parts:
        return pl.LazyFrame(schema={
            "e": schema_dtype(_enums, "e"),
            "kind": pl.Utf8,
        })
    return pl.concat(parts, how="vertical").unique()


def pd_invest_set_lf(source: "InputSource",
                            ed_invest_lf: pl.LazyFrame,
                            ) -> pl.LazyFrame:
    """Lazy ``[p, d]`` — ``ed_invest_set`` partitioned to processes
    (``unit ∪ connection``).  Mirror of
    ``invest_divest_sets.py:215-217``.
    """
    cls_lf = _entity_class_partition_lf(source)
    return (ed_invest_lf
              .join(cls_lf, on="e", how="inner")
              .filter(pl.col("kind") == "process")
              .pipe(rename_to_axis, {"e": "p"})
              .select("p", "d")
              .unique()
              .sort("p", "d"))


def nd_invest_set_lf(source: "InputSource",
                            ed_invest_lf: pl.LazyFrame,
                            ) -> pl.LazyFrame:
    """Lazy ``[n, d]`` — ``ed_invest_set`` partitioned to nodes."""
    cls_lf = _entity_class_partition_lf(source)
    return (ed_invest_lf
              .join(cls_lf, on="e", how="inner")
              .filter(pl.col("kind") == "node")
              .pipe(rename_to_axis, {"e": "n"})
              .select("n", "d")
              .unique()
              .sort("n", "d"))


def pd_divest_set_lf(source: "InputSource",
                            ed_divest_lf: pl.LazyFrame,
                            ) -> pl.LazyFrame:
    """Lazy ``[p, d]`` — ``ed_divest_set`` partitioned to processes."""
    cls_lf = _entity_class_partition_lf(source)
    return (ed_divest_lf
              .join(cls_lf, on="e", how="inner")
              .filter(pl.col("kind") == "process")
              .pipe(rename_to_axis, {"e": "p"})
              .select("p", "d")
              .unique()
              .sort("p", "d"))


def nd_divest_set_lf(source: "InputSource",
                            ed_divest_lf: pl.LazyFrame,
                            ) -> pl.LazyFrame:
    """Lazy ``[n, d]`` — ``ed_divest_set`` partitioned to nodes."""
    cls_lf = _entity_class_partition_lf(source)
    return (ed_divest_lf
              .join(cls_lf, on="e", how="inner")
              .filter(pl.col("kind") == "node")
              .pipe(rename_to_axis, {"e": "n"})
              .select("n", "d")
              .unique()
              .sort("n", "d"))


def ed_invest_period_set_lf(source: "InputSource",
                                  ed_invest_lf: pl.LazyFrame,
                                  ) -> pl.LazyFrame:
    """Lazy ``[e, d]`` — ``ed_invest`` filtered to entities whose
    ``invest_method`` is in :data:`_INVEST_PERIOD_METHODS`.

    Mirror of ``preprocessing/invest_divest_sets.py`` 's
    ``ed_invest_period`` set: keeps only the (e, d) pairs that carry
    a per-period invest cap (``maxInvest_entity_period`` /
    ``minInvest_entity_period`` constraint indices).  Used in
    ``model.py:1517-1530`` to index the per-period invest cap rows.
    """
    methods_lf = _entity_method_lf(source, "invest_method")
    period_e = (methods_lf
                   .filter(pl.col("method").is_in(list(_INVEST_PERIOD_METHODS)))
                   .select("e")
                   .unique())
    return (ed_invest_lf
              .join(period_e, on="e", how="inner")
              .select("e", "d")
              .unique()
              .sort("e", "d"))


def ed_divest_period_set_lf(source: "InputSource",
                                  ed_divest_lf: pl.LazyFrame,
                                  ) -> pl.LazyFrame:
    """Lazy ``[e, d]`` — ``ed_divest`` filtered to entities whose
    ``invest_method`` is in :data:`_DIVEST_PERIOD_METHODS`.  Mirror of
    :func:`ed_invest_period_set_lf` for divest.
    """
    methods_lf = _entity_method_lf(source, "invest_method")
    period_e = (methods_lf
                   .filter(pl.col("method").is_in(list(_DIVEST_PERIOD_METHODS)))
                   .select("e")
                   .unique())
    return (ed_divest_lf
              .join(period_e, on="e", how="inner")
              .select("e", "d")
              .unique()
              .sort("e", "d"))


# ---------------------------------------------------------------------------
# §3.7.3 — edd_invest_set + edd_invest_lookback_set
# ---------------------------------------------------------------------------


def edd_invest_set_lf(source: "InputSource",
                              active_solve: str | None,
                              ed_invest_lf: pl.LazyFrame,
                              period_with_history: list[str],
                              period_in_use: list[str],
                              workdir: Path | None = None,
                              ) -> pl.LazyFrame:
    """Lazy ``[e, d_invest, d]`` — ``edd_history_invest`` filtered to
    ``(e, d_invest) ∈ ed_invest``.

    Mirror of ``invest_divest_sets.py:267-270``::

        edd_invest = { (e, d_inv, d) : (e, d_inv, d) ∈ edd_history,
                                          (e, d_inv) ∈ ed_invest }

    The ``edd_history`` triple-set carries history-period entries for
    every entity with a recognised lifetime_method; this helper filters
    to those whose ``d_history`` (renamed ``d_invest``) is also a
    valid invest decision in the current solve.
    """
    edd = edd_history_lf(source, active_solve,
                              period_with_history, period_in_use,
                              workdir)
    inv_pairs = ed_invest_lf.pipe(rename_to_axis, {"d": "d_history"})
    return (edd
              .join(inv_pairs, on=["e", "d_history"], how="inner")
              .pipe(rename_to_axis, {"d_history": "d_invest"})
              .select("e", "d_invest", "d"))


def edd_invest_lookback_set_lf(source: "InputSource",
                                       active_solve: str | None,
                                       ed_invest_lf: pl.LazyFrame,
                                       period_in_use: list[str],
                                       workdir: Path | None = None,
                                       ) -> pl.LazyFrame:
    """Lazy ``[e, d_invest, d]`` — ``edd_invest_lookback_set``.

    The strict-lookback variant of :func:`edd_invest_set_lf`: tuples
    where ``yr[d_invest] < yr[d]`` (strict) and, for entities in the
    bounded lifetime cohort (``reinvest_choice`` / ``no_investment``),
    ``yr[d] < yr[d_invest] + lifetime[e, d_invest]``.

    Mirror of ``invest_divest_sets.py:227-262`` filtered with the
    strict-lookback predicate.  Used by the prebuilt-capacity LHS in
    user-constraint expressions (mod L2885-2898).

    Δ.7 consolidation: the eager
    :func:`._derived_params.edd_invest_lookback_set_from_source`
    pre-dates the shared :func:`._derived_walks.period_walk_iterator`;
    Δ.7 wraps the shared walker with the
    :data:`._derived_walks.WindowMethod.STRICT_LOOKBACK_BOUNDED` /
    :data:`._derived_walks.WindowMethod.STRICT_LOOKBACK_UNBOUNDED`
    modes for bounded / unbounded entity cohorts respectively, then
    unions the results.

    The eager helper now delegates to this lazy port — the previous
    Python ``for r in out.iter_rows`` lifetime gate is replaced with
    a fully lazy join + filter on the shared walker.
    """
    if not period_in_use:
        return pl.LazyFrame(schema={
            "e": pl.Utf8, "d_invest": pl.Utf8, "d": pl.Utf8,
        })
    inv_anchor = ed_invest_lf.select("e", alias_to_axis("d", "d_invest"))
    if inv_anchor.collect().height == 0:
        return pl.LazyFrame(schema={
            "e": pl.Utf8, "d_invest": pl.Utf8, "d": pl.Utf8,
        })

    all_e_lf = _all_entities_lf(source)
    elm_lf = _lifetime_method_with_default_lf(source, all_e_lf)
    bounded_e = (elm_lf
                    .filter(pl.col("method").is_in(
                        list(_LIFETIME_BOUNDED_METHODS)))
                    .select("e").unique())
    unbounded_e = (elm_lf
                      .filter(~pl.col("method").is_in(
                          list(_LIFETIME_BOUNDED_METHODS)))
                      .select("e").unique())

    # Shape the anchor as ``[e, d]`` for the shared walker (it expects
    # the anchor period column to be named ``d``).
    anchor_lf = inv_anchor.pipe(rename_to_axis, {"d_invest": "d"})

    # The source-driven ``_all_entities_lf`` reads raw CSV — String
    # ``e``.  ``anchor_lf.e`` carries whatever dtype the caller passed
    # (Enum-e when called from the post-load test path).  Align so the
    # join below works.
    anchor_e_dtype = anchor_lf.collect_schema().get("e", pl.Utf8)
    if anchor_e_dtype != pl.Utf8:
        bounded_e = bounded_e.with_columns(
            pl.col("e").cast(anchor_e_dtype, strict=False))
        unbounded_e = unbounded_e.with_columns(
            pl.col("e").cast(anchor_e_dtype, strict=False))

    # Bounded cohort.
    bounded_anchor = anchor_lf.join(bounded_e, on="e", how="inner")
    # Scratch frame for the cohort with no rows: must match the dtypes
    # of the upstream Enum-cast inputs to be ``vstack``-compatible at
    # the bottom-of-function ``pl.concat``.  ``empty_like`` reads the
    # input frame's schema for the requested columns and uses
    # ``pl.Utf8`` as the fall-back when the column is absent.
    from flextool.engine_polars._axis_enums import empty_like
    bounded_walk = empty_like(anchor_lf,
                                ["e", "d"],
                                extra={"d_all": anchor_lf.collect_schema().get("d", pl.Utf8)},
                                lazy=True)
    if bounded_anchor.collect().height > 0:
        life_per = _per_entity_param_lf(source, "lifetime")
        life_lf = (_resolve_per_period_lf(life_per, bounded_anchor, fill=0.0)
                       .rename({"value": "life"})
                       .select("e", "d", "life"))
        bounded_walk = period_walk_iterator(
            source, active_solve, bounded_anchor,
            period_in_use, period_in_use,
            window_method=WindowMethod.STRICT_LOOKBACK_BOUNDED,
            life_lf=life_lf, factor_side=None,
            workdir=workdir)

    # Unbounded cohort — strict-lookback only, no lifetime cap.
    unbounded_anchor = anchor_lf.join(unbounded_e, on="e", how="inner")
    unbounded_walk = empty_like(anchor_lf,
                                  ["e", "d"],
                                  extra={"d_all": anchor_lf.collect_schema().get("d", pl.Utf8)},
                                  lazy=True)
    if unbounded_anchor.collect().height > 0:
        unbounded_walk = period_walk_iterator(
            source, active_solve, unbounded_anchor,
            period_in_use, period_in_use,
            window_method=WindowMethod.STRICT_LOOKBACK_UNBOUNDED,
            life_lf=None, factor_side=None,
            workdir=workdir)

    return (pl.concat([bounded_walk, unbounded_walk], how="vertical")
              .pipe(rename_to_axis, {"d": "d_invest", "d_all": "d"})
              .select("e", "d_invest", "d")
              .unique()
              .sort("e", "d_invest", "d"))


# ---------------------------------------------------------------------------
# §3.11 — p_entity_all_existing (existing-chain walk)
# ---------------------------------------------------------------------------


def p_entity_all_existing_from_handoff(
        source: "InputSource",
        active_solve: str | None,
        period_with_history: list[str],  # noqa: ARG001 — kept for symmetry
        period_in_use: list[str],
        *,
        p_entity_period_existing_capacity: "Param | None" = None,
        p_entity_previously_invested_capacity: "Param | None" = None,
        p_entity_divested: "Param | None" = None,
        solve_first: bool = True,
        edd_history: "pl.DataFrame | None" = None,
        ) -> "Param | None":
    """Lazy port of flextool's
    ``write_p_entity_existing_chain`` algorithm
    (``entity_period_calc_params.py:1381-1591``).

    Per flextool's writer (lines 1554-1568):

      * ``solve_first`` →
            ``all_existing[e, d] = pre_existing[e, d]``  (no divest applied)
      * later solves →
            ``all_existing[e, d] = later_existing[e, d]``
            ``                    − p_entity_divested[e]   (if e ∈ entityDivest)``

    where ``later_existing[e, d]`` for the chain-runner case is the
    chain-summed prior-solve existing per current period — already
    integrated by flextool's preprocessing into
    ``p_entity_previously_invested_capacity[e, d]`` (the
    ``later_invested`` shape) ⊕ ``pre_existing[e, d]``.  Δ.11 — we rebuild
    the result from the in-memory handoff carriers via
    ``pre + ppic − divest`` which matches the canonical flextool CSV
    (verified per-fixture in ``test_existing_chain_cluster_parity.py``).

    Parameters
    ----------
    p_entity_period_existing_capacity
        Δ.11 — in-memory handoff carrier ``[entity, period, value]``
        (the SolveHandoff ``realized_existing`` shape).  When supplied,
        the helper switches into the ``later_existing`` chain-summation
        branch: ``later_existing[e, d] = Σ_{(e, d_h, d) ∈ edd_history ∧
        (e, d_h) realized} ppec[(e, d_h)]`` (mirrors flextool's
        ``write_p_entity_existing_chain`` lines 1530-1543).  Required for
        lifetime-renew chains where the chain-summed existing isn't
        recoverable from ``ppic + pre_existing`` alone.
    p_entity_previously_invested_capacity
        In-memory handoff Param ``[e, d, value]`` — chain-summed prior
        invest at d.  Used only when ``p_entity_period_existing_capacity``
        is None (the simple-baseline branch); the chain-summation
        branch derives both terms from ``ppec``.
    p_entity_divested
        Handoff Param ``[e, value]`` cumulative prior divest.
    solve_first
        Δ.11 — flextool's ``solveFirst`` flag.  When True, divest is
        NOT subtracted (mirrors the writer's solve-first branch); when
        False, ``e ∈ entityDivest`` triggers a per-entity divest
        subtraction.  Default True (single-solve / cold-start).
    edd_history
        Δ.11 — ``[entity, period_history, period]`` triple-set used by
        the chain-summation branch.  When None, the helper falls back
        to the simple ``pre + ppic − divest`` formula (which is exact
        for non-renew chains).

    Returns Param ``[e, d, value]`` or None when no entities exist.
    """
    if not period_in_use:
        return None
    all_e_collected = _all_entities_lf(source).collect()
    if all_e_collected.height == 0:
        return None
    all_e_lf = _all_entities_lf(source)
    piu_lf = (pl.LazyFrame({"d": period_in_use})
                .with_columns(pl.col("d").cast(schema_dtype(_enums, "d"),
                                                  strict=False)))
    grid_lf = all_e_lf.join(piu_lf, how="cross")

    pre_existing_lf = p_entity_pre_existing_lf(
        source, active_solve, period_in_use)

    # entityDivest set + p_entity_divested scalar.  Shared by both
    # branches; the divest subtraction is gated on solve_first below.
    div_set_lf = entity_divest_set_lf(source).with_columns(
        is_divest=pl.lit(True))
    if (p_entity_divested is not None
            and p_entity_divested.frame.height > 0):
        ped_lf = (p_entity_divested.frame.lazy()
                     .select("e", pl.col("value").alias("divested")))
    else:
        ped_lf = pl.LazyFrame(schema={
            "e": schema_dtype(_enums, "e"),
            "divested": pl.Float64,
        })

    # ---------------------------------------------------------------
    # Δ.11 — chain-summation branch: later_existing[e, d] from
    # edd_history × ppec.  Activates when the caller supplies the
    # ``p_entity_period_existing_capacity`` carrier (the SolveHandoff
    # ``realized_existing`` shape) AND the ``edd_history`` triple-set
    # AND solve_first is False.  Mirrors flextool's writer at
    # ``entity_period_calc_params.py:1530-1568``.
    # ---------------------------------------------------------------
    if (not solve_first
            and p_entity_period_existing_capacity is not None
            and p_entity_period_existing_capacity.frame.height > 0
            and edd_history is not None and edd_history.height > 0):
        ppec_lf = (p_entity_period_existing_capacity.frame.lazy()
                       .select(alias_to_axis("e", "e"),
                                alias_to_axis("d", "d_h"),
                                pl.col("value").alias("ppec")))
        edd_lf = edd_history.lazy()
        if {"entity", "period_history", "period"}.issubset(set(edd_history.columns)):
            edd_lf = edd_lf.pipe(rename_to_axis, {"entity": "e",
                                       "period_history": "d_h",
                                       "period": "d"})
        # later_existing[e, d] = Σ_{d_h: (e, d_h, d) ∈ edd_history ∧
        #                              (e, d_h) ∈ realized} ppec[(e, d_h)].
        # (e, d_h) is "realized" iff ppec carries a row for it.
        from flextool.engine_polars._axis_enums import align_join_dtypes
        edd_lf, ppec_lf = align_join_dtypes(edd_lf, ppec_lf, ["e", "d_h"])
        later_lf = (edd_lf
                       .join(ppec_lf, on=["e", "d_h"], how="inner")
                       .group_by(["e", "d"])
                       .agg(pl.col("ppec").sum().alias("later")))
        grid_lf, later_lf = align_join_dtypes(grid_lf, later_lf, ["e", "d"])
        grid_lf, div_set_lf = align_join_dtypes(grid_lf, div_set_lf, ["e"])
        grid_lf, ped_lf = align_join_dtypes(grid_lf, ped_lf, ["e"])
        out = (grid_lf
                  .join(later_lf, on=["e", "d"], how="left")
                  .join(div_set_lf, on="e", how="left")
                  .join(ped_lf, on="e", how="left")
                  .with_columns(
                      later=pl.col("later").fill_null(0.0),
                      is_divest=pl.col("is_divest").fill_null(False),
                      divested=pl.col("divested").fill_null(0.0),
                  )
                  .with_columns(
                      value=pl.col("later")
                      - pl.when(pl.col("is_divest"))
                              .then(pl.col("divested"))
                              .otherwise(0.0),
                  )
                  .select("e", "d", "value")
                  .sort("e", "d")
                  .collect())
        if out.height == 0:
            return None
        return Param(("e", "d"), out)

    # ---------------------------------------------------------------
    # Simple-baseline branch (single-solve / non-renew chains):
    #   pae[e, d] = pre_existing[e, d]
    #             + p_entity_previously_invested_capacity[e, d]
    #             − p_entity_divested[e]   (if e ∈ entityDivest, later solves)
    # ---------------------------------------------------------------

    # Previously-invested overlay (in-memory handoff carrier).
    if (p_entity_previously_invested_capacity is not None
            and p_entity_previously_invested_capacity.frame.height > 0):
        ppic_frame = p_entity_previously_invested_capacity.frame
        ppic_lf = (ppic_frame.lazy()
                      .select("e", "d", pl.col("value").alias("ppic")))
    else:
        ppic_lf = pl.LazyFrame(schema={
            "e": pl.Utf8, "d": pl.Utf8, "ppic": pl.Float64,
        })

    # Align grid_lf's String dim columns with any Enum-typed inputs
    # (ppic_lf / ped_lf come from FlexData Params that have been
    # cast).  Two passes — the first lifts grid_lf to the widest dtype
    # available from any input, the second pulls every other input
    # back up to grid_lf's promoted dtype.
    from flextool.engine_polars._axis_enums import align_join_dtypes
    pre_renamed = pre_existing_lf.rename({"value": "pre"})
    grid_lf, ppic_lf = align_join_dtypes(grid_lf, ppic_lf, ["e", "d"])
    grid_lf, ped_lf = align_join_dtypes(grid_lf, ped_lf, ["e"])
    grid_lf, pre_renamed = align_join_dtypes(grid_lf, pre_renamed, ["e", "d"])
    grid_lf, div_set_lf = align_join_dtypes(grid_lf, div_set_lf, ["e"])
    # Second pass: now grid_lf carries the widest dtype on each axis,
    # pull every input up.
    grid_lf, ppic_lf = align_join_dtypes(grid_lf, ppic_lf, ["e", "d"])
    grid_lf, ped_lf = align_join_dtypes(grid_lf, ped_lf, ["e"])
    grid_lf, pre_renamed = align_join_dtypes(grid_lf, pre_renamed, ["e", "d"])
    grid_lf, div_set_lf = align_join_dtypes(grid_lf, div_set_lf, ["e"])
    out = (grid_lf
              .join(pre_renamed,
                      on=["e", "d"], how="left")
              .join(ppic_lf, on=["e", "d"], how="left")
              .join(div_set_lf, on="e", how="left")
              .join(ped_lf, on="e", how="left")
              .with_columns(
                  pre=pl.col("pre").fill_null(0.0),
                  ppic=pl.col("ppic").fill_null(0.0),
                  is_divest=pl.col("is_divest").fill_null(False),
                  divested=pl.col("divested").fill_null(0.0),
              )
              .with_columns(
                  # Δ.11 — divest subtraction is applied only on later
                  # solves (solve_first=False); flextool's writer skips it
                  # on the first solve in the chain.
                  value=pl.col("pre") + pl.col("ppic")
                  - (pl.when(pl.col("is_divest"))
                          .then(pl.col("divested"))
                          .otherwise(0.0)
                     if not solve_first else pl.lit(0.0)),
              )
              .select("e", "d", "value")
              .sort("e", "d")
              .collect())
    if out.height == 0:
        return None
    return Param(("e", "d"), out)


def p_entity_pre_existing_lf(source: "InputSource",
                                  active_solve: str | None,
                                  period_in_use: list[str],
                                  ) -> pl.LazyFrame:
    """Lazy ``[e, d, value]`` for ``p_entity_pre_existing`` —
    lifetime-gated ``entity.existing × virtual_unitsize`` cascade.

    Algorithm (mirror of
    ``preprocessing/entity_period_calc_params.py:write_p_entity_pre_existing``):

      For each entity e:
        v_existing = pdProcess[e, 'existing', d] (if e ∈ process)
                   else pdNode[e, 'existing', d] (if e ∈ node)
                   else 0
        v_unit = p_process[e, 'virtual_unitsize'] (if e ∈ process)
                 else p_node[e, 'virtual_unitsize'] (if e ∈ node)
                 else 0
        if lifetime_method = reinvest_automatic →
            v = v_existing * v_unit  if v_unit else v_existing
        if lifetime_method ∈ {reinvest_choice, no_investment} →
            v = (v_existing * v_unit if v_unit else v_existing)
                while pdy[d] < life_sum(e); 0 thereafter.
        else → 0.

    The ``v_existing × v_unit`` term is the canonical capacity (a
    counted entity's ``existing`` is the count, not the capacity).
    Only applied when ``v_unit > 0``.
    """
    if not period_in_use:
        return pl.LazyFrame(schema={
            "e": schema_dtype(_enums, "e"),
            "d": schema_dtype(_enums, "d"),
            "value": pl.Float64,
        })
    all_e_lf = _all_entities_lf(source)
    piu_lf = (pl.LazyFrame({"d": period_in_use})
                .with_columns(pl.col("d").cast(schema_dtype(_enums, "d"),
                                                  strict=False)))
    grid_lf = all_e_lf.join(piu_lf, how="cross")

    # Resolve existing per (e, d) via _resolve_pdX cascade.
    ex_per = _per_entity_param_lf(source, "existing")
    grid_with_ex = (_resolve_per_period_lf(ex_per, grid_lf, fill=0.0)
                       .rename({"value": "ex"}))

    # virtual_unitsize per entity (scalar across periods).
    vu_per = _per_entity_param_lf(source, "virtual_unitsize")
    # Aggregate to (e, vu) — virtual_unitsize is non-period for
    # process / node (per write_p_entity_pre_existing's read of
    # p_process / p_node directly, NOT pd_process / pd_node).
    vu_scalar = (vu_per
                    .filter(pl.col("is_scalar"))
                    .select("e", pl.col("value").alias("vu"))
                    .unique(subset=["e"], keep="last"))
    grid_with_vu = (grid_with_ex
                       .join(vu_scalar, on="e", how="left")
                       .with_columns(vu=pl.col("vu").fill_null(0.0)))

    # Lifetime methods (with default).
    elm_lf = _lifetime_method_with_default_lf(source, all_e_lf)
    is_auto = (elm_lf
                  .filter(pl.col("method") == "reinvest_automatic")
                  .select("e").unique()
                  .with_columns(is_auto=pl.lit(True)))
    is_bounded = (elm_lf
                     .filter(pl.col("method").is_in(list(_LIFETIME_BOUNDED_METHODS)))
                     .select("e").unique()
                     .with_columns(is_bounded=pl.lit(True)))

    # Lifetime gate: zero out existing past life_sum expiry for bounded.
    expired_lf = _lifetime_expired_pairs_lf(
        source, active_solve, period_in_use, methods=_LIFETIME_BOUNDED_METHODS)
    expired_marked = expired_lf.with_columns(_expired=pl.lit(True))

    # Compose: value = ex × vu (or ex if vu=0) if alive; 0 else.
    capacity_expr = (pl.when(pl.col("vu") != 0.0)
                          .then(pl.col("ex") * pl.col("vu"))
                          .otherwise(pl.col("ex")))
    return (grid_with_vu
              .join(is_auto, on="e", how="left")
              .join(is_bounded, on="e", how="left")
              .join(expired_marked, on=["e", "d"], how="left")
              .with_columns(
                  is_auto=pl.col("is_auto").fill_null(False),
                  is_bounded=pl.col("is_bounded").fill_null(False),
                  _expired=pl.col("_expired").fill_null(False),
                  cap=capacity_expr,
              )
              .with_columns(
                  value=pl.when(pl.col("is_auto"))
                              .then(pl.col("cap"))
                              .when(pl.col("is_bounded") & ~pl.col("_expired"))
                              .then(pl.col("cap"))
                              .otherwise(0.0),
              )
              .select("e", "d", "value"))


def _lifetime_expired_pairs_lf(source: "InputSource",
                                    active_solve: str | None,
                                    period_in_use: list[str],
                                    *,
                                    methods: frozenset[str],
                                    ) -> pl.LazyFrame:
    """Lazy ``[e, d]`` of pairs whose pre-existing capacity has expired.

    For each entity whose ``lifetime_method ∈ methods``, return periods
    d where ``pdy[d] >= life_sum(e)`` with::

        life_sum(e) = Σ_{d_first ∈ period_first}
                          (pdy[d_first] + lifetime[e, d_first])

    Mirror of :func:`._derived_params._lifetime_expired_pairs` in lazy form.
    Uses :func:`._derived_params._read_period_first` /
    :func:`._derived_params._p_years_d_lf` for the workdir-aware reads.
    """
    from ._derived_params import _read_period_first, _p_years_d_lf
    period_first = _read_period_first(source, active_solve, None)
    if not period_first:
        return pl.LazyFrame(schema={
            "e": schema_dtype(_enums, "e"),
            "d": schema_dtype(_enums, "d"),
        })
    all_e_lf = _all_entities_lf(source)
    elm_lf = _lifetime_method_with_default_lf(source, all_e_lf)
    in_methods = (elm_lf
                     .filter(pl.col("method").is_in(list(methods)))
                     .select("e").unique())

    # Class gate: connection without is_DC / non-process → no lifetime.
    cls_lf = _entity_class_lf(source).group_by("e").agg(
        pl.col("ec").alias("ecs"))
    is_proc_or_node = (pl.col("ecs").list.contains("unit")
                        | pl.col("ecs").list.contains("connection")
                        | pl.col("ecs").list.contains("node"))
    in_class = cls_lf.with_columns(
        in_class=is_proc_or_node).select("e", "in_class")

    pyd_lf = _p_years_d_lf(source, active_solve)
    if pyd_lf is None:
        return pl.LazyFrame(schema={
            "e": schema_dtype(_enums, "e"),
            "d": schema_dtype(_enums, "d"),
        })

    # life_sum: per-entity scalar.  Compute per d_first then sum:
    # life_sum(e) = Σ_{d_first} (pdy[d_first] + lifetime[e, d_first]).
    # Phase 4.6: ``d_first`` is a column holding period tokens; cast the
    # literal-list LazyFrame to the ``d`` (period) Enum so downstream
    # joins against Enum-typed ``pyd_first`` don't SchemaError.
    pf_lf = (pl.LazyFrame({"d_first": period_first})
                .with_columns(pl.col("d_first")
                                .cast(schema_dtype(_enums, "d_first"),
                                       strict=False)))
    pyd_first = pyd_lf.rename({"d": "d_first", "yr": "yr_first"})
    e_dfirst = (in_methods.join(pf_lf, how="cross")
                          .pipe(rename_to_axis, {"d_first": "d"})
                          .pipe(lambda lf: lf))  # placeholder for symmetry
    # Resolve lifetime[e, d_first] via cascade.
    life_per = _per_entity_param_lf(source, "lifetime")
    life_resolved = (_resolve_per_period_lf(
        life_per,
        in_methods.join(pf_lf, how="cross").pipe(rename_to_axis, {"d_first": "d"}),
        fill=0.0))
    life_at_first = life_resolved.rename({"d": "d_first", "value": "life"})
    summed_lf = (life_at_first
                    .join(pyd_first, on="d_first", how="left")
                    .with_columns(yr_first=pl.col("yr_first").fill_null(0.0))
                    .group_by("e")
                    .agg((pl.col("yr_first") + pl.col("life")).sum()
                            .alias("life_sum")))
    # Apply class gate.
    summed_lf = (summed_lf
                    .join(in_class, on="e", how="left")
                    .with_columns(in_class=pl.col("in_class").fill_null(False))
                    .with_columns(
                        life_sum=pl.when(pl.col("in_class"))
                                       .then(pl.col("life_sum"))
                                       .otherwise(0.0))
                    .select("e", "life_sum"))

    piu_lf = (pl.LazyFrame({"d": period_in_use})
                .with_columns(pl.col("d").cast(schema_dtype(_enums, "d"),
                                                  strict=False)))
    pyd_at_d = pyd_lf
    return (summed_lf
              .join(piu_lf, how="cross")
              .join(pyd_at_d, on="d", how="left")
              .with_columns(yr=pl.col("yr").fill_null(0.0))
              .filter(pl.col("yr") >= pl.col("life_sum"))
              .select("e", "d"))


# ---------------------------------------------------------------------------
# Public entry: apply boundary
# ---------------------------------------------------------------------------


def apply_existing_chain(flex_data: object,
                              source: "InputSource",
                              workdir: Path,
                              *,
                              handoff: object | None = None,
                              ctx: "SolveContext | None" = None,
                              provider: "object | None" = None) -> None:
    """Apply Cluster B helpers to ``flex_data`` (mutates in place).

    Wired-in fields:

    * ``p_entity_all_existing`` — chained existing capacity (replacing
      the workdir CSV read of ``solve_data/p_entity_all_existing.csv``
      and the cascade in :func:`._derived_params.p_entity_all_existing_from_source`).

    Order:
      1. Resolve the per-solve scope (active_solve, periods, etc.).
      2. Pull the in-memory handoff carriers off ``flex_data``
         (``p_entity_period_existing_capacity``, ``p_entity_divested``)
         — these are populated by the CSV-side handoff loader / by
         :func:`apply_derived_f`'s handoff overlay.
      3. Compute ``p_entity_all_existing`` via the lazy chain.
      4. Overwrite ``flex_data.p_entity_all_existing`` only when the
         lazy result is non-None (mirrors the eager helper's gate).

    Notes
    -----
    Cluster B is parity-bound to the eager
    ``p_entity_all_existing_from_source`` helper which the per-fixture
    sweep validates.  When parity drifts, the test surfaces the
    fixture and the diff frame.
    """
    from ._derived_params import (
        _read_active_solve, _period_in_use_set,
        _read_period_with_history,
    )
    from .input import _read_solve_first
    from ._writer_provider_io import _provider_key
    from polar_high import Param as _Param

    def _provider_get(p):
        if provider is None:
            return None
        key = _provider_key(p)
        if not provider.has(key):
            return None
        return provider.get(key)

    # Δ.12a — prefer the typed SolveContext fields (zero physical IO)
    # when the caller supplied one; fall back to direct workdir reads
    # for back-compat with callers that haven't been wired up.
    if ctx is not None:
        active_solve = ctx.solve_name
        solve_first = ctx.solveFirst
        ppec_csv_df = ctx.p_entity_period_existing_capacity
        edd_hist_df_raw = ctx.edd_history
    else:
        active_solve = _read_active_solve(workdir, provider=provider)
        solve_first = _read_solve_first(workdir, provider=provider)
        ppec_csv_df = None
        edd_hist_df_raw = None
    period_in_use = _period_in_use_set(source, active_solve, workdir,
                                         ctx=ctx, provider=provider)
    period_with_history = (_read_period_with_history(workdir, provider=provider)
                              or list(period_in_use))

    ppic = getattr(flex_data, "p_entity_previously_invested_capacity", None)
    ped = getattr(flex_data, "p_entity_divested", None)

    # Δ.11 — chain-summation inputs.  Prefer the in-memory handoff
    # ``realized_existing`` carrier when supplied; else fall back to the
    # workdir's ``p_entity_period_existing_capacity.csv`` (which carries
    # the same data after flextool's preprocessing).  ``edd_history.csv``
    # is always sourced from the workdir.
    ppec_param: "Param | None" = None
    if handoff is not None and getattr(handoff, "realized_existing", None) is not None:
        re_frame = handoff.realized_existing
        if re_frame.height > 0:
            ppec_param = _Param(("e", "d"),
                                  re_frame.pipe(rename_to_axis, {"entity": "e",
                                                       "period": "d"})
                                          .select("e", "d", "value"))
    if ppec_param is None:
        # Δ.12a — prefer the typed SolveContext field; otherwise consult
        # the Provider (no disk fallback — Step 2.5).
        if ctx is not None and ppec_csv_df is not None and ppec_csv_df.height > 0:
            df = ppec_csv_df
        else:
            ppec_path = workdir / "solve_data" / "p_entity_period_existing_capacity.csv"
            df = _provider_get(ppec_path)
        if (df is not None and df.height > 0
                and "p_entity_period_existing_capacity" in df.columns):
            ppec_param = _Param(("e", "d"),
                                  df.pipe(rename_to_axis, {"entity": "e",
                                               "period": "d"})
                                    .select("e", "d",
                                              pl.col("p_entity_period_existing_capacity")
                                                .cast(pl.Float64, strict=False)
                                                .fill_null(0.0)
                                                .alias("value")))

    edd_hist_df: "pl.DataFrame | None" = None
    if ctx is not None:
        if edd_hist_df_raw is not None and edd_hist_df_raw.height > 0:
            edd_hist_df = edd_hist_df_raw
    else:
        edd_path = workdir / "solve_data" / "edd_history.csv"
        edd_hist_df = _provider_get(edd_path)
        if edd_hist_df is not None and edd_hist_df.height == 0:
            edd_hist_df = None

    # Δ.12b — assign unconditionally.  The helper returns ``None`` only
    # when there's no entity_invest_method-eligible row in the source,
    # which is a legitimate "no-existing-capacity-to-track" outcome
    # (matches the eager helper's empty-frame skip).  Hard errors
    # propagate (no defensive try/except).
    flex_data.p_entity_all_existing = p_entity_all_existing_from_handoff(
        source, active_solve,
        period_with_history, period_in_use,
        p_entity_period_existing_capacity=ppec_param,
        p_entity_previously_invested_capacity=ppic,
        p_entity_divested=ped,
        solve_first=solve_first,
        edd_history=edd_hist_df)


__all__ = [
    # Method-enum constants (re-exported for direct-test use).
    "_INVEST_NOT_ALLOWED", "_DIVEST_NOT_ALLOWED",
    "_INVEST_TOTAL_METHODS", "_DIVEST_TOTAL_METHODS",
    "_INVEST_PERIOD_METHODS", "_DIVEST_PERIOD_METHODS",
    # Public lazy helpers — entity sets.
    "entity_invest_set_lf", "entity_divest_set_lf",
    "e_invest_total_lf", "e_divest_total_lf",
    # Public lazy helpers — edd_history triple-set family.
    "edd_history_lf", "edd_history_choice_lf",
    "edd_history_automatic_lf", "edd_history_no_investment_lf",
    "edd_invest_set_lf", "edd_invest_lookback_set_lf",
    # Public lazy helpers — partitions.
    "pd_invest_set_lf", "nd_invest_set_lf",
    "pd_divest_set_lf", "nd_divest_set_lf",
    "ed_invest_period_set_lf", "ed_divest_period_set_lf",
    # Public lazy helpers — existing chain.
    "p_entity_all_existing_from_handoff", "p_entity_pre_existing_lf",
    "apply_existing_chain",
]
