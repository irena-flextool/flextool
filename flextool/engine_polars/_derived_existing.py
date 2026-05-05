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

from polar_high_opt import Param

from ._derived_walks import period_walk_iterator, WindowMethod

if TYPE_CHECKING:
    from flextool.engine_polars._input_source import InputSource


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
            pl.col("name").alias("e"),
            pl.lit(ec).alias("ec"),
        ))
    if not parts:
        return pl.LazyFrame(schema={"e": pl.Utf8, "ec": pl.Utf8})
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
            pl.col("name").alias("e"),
            pl.col("value").cast(pl.Utf8, strict=False).alias("method"),
        ))
    if not parts:
        return pl.LazyFrame(schema={"e": pl.Utf8, "method": pl.Utf8})
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
        if "period" in cols:
            parts.append(df.lazy().select(
                pl.col("name").alias("e"),
                pl.col("period").alias("d"),
                pl.col("value").cast(pl.Float64, strict=False),
                pl.lit(False).alias("is_scalar"),
            ))
        else:
            parts.append(df.lazy().select(
                pl.col("name").alias("e"),
                pl.lit(None, dtype=pl.Utf8).alias("d"),
                pl.col("value").cast(pl.Float64, strict=False),
                pl.lit(True).alias("is_scalar"),
            ))
    if not parts:
        return pl.LazyFrame(schema={
            "e": pl.Utf8, "d": pl.Utf8,
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
# §3.7.2 — edd_history triple-set walk
# ---------------------------------------------------------------------------


def edd_history_lf(source: "InputSource",
                       active_solve: str | None,
                       period_with_history: list[str],
                       period_in_use: list[str],
                       ) -> pl.LazyFrame:
    """Lazy ``[e, d_history, d]`` for the union ``edd_history`` set.

    Algorithm (mirror of
    ``invest_divest_sets.py:227-262``):

      For each entity e with lifetime_method ∈ {reinvest_choice,
      reinvest_automatic, no_investment}:
        For d_h ∈ period_with_history, d ∈ period_in_use:
          * reinvest_choice / no_investment:
              keep iff pdy[d] ∈ [pdy[d_h], pdy[d_h] + life[e, d_h])
          * reinvest_automatic:
              keep iff pdy[d] ≥ pdy[d_h]

    All bounded variants share the lifetime gate (life is the per-(e,
    d_h) ``lifetime`` cascade), so we do two walks (bounded vs.
    unbounded) and union them.

    Returns lazy ``[e, d_history, d]``.
    """
    if not period_with_history or not period_in_use:
        return pl.LazyFrame(schema={
            "e": pl.Utf8, "d_history": pl.Utf8, "d": pl.Utf8,
        })

    all_e_lf = _all_entities_lf(source)
    elm_lf = _lifetime_method_with_default_lf(source, all_e_lf)
    bounded_e = (elm_lf
                    .filter(pl.col("method").is_in(list(_LIFETIME_BOUNDED_METHODS)))
                    .select("e").unique())
    unbounded_e = (elm_lf
                      .filter(pl.col("method").is_in(list(_LIFETIME_UNBOUNDED_METHODS)))
                      .select("e").unique())

    pwh_lf = pl.LazyFrame({"d": period_with_history})

    # Anchor frame for the bounded walk: bounded entities × period_with_history.
    bounded_anchor = bounded_e.join(pwh_lf, how="cross")
    # life_lf: per-(e, d_history) lifetime via _resolve_pdX cascade.
    life_per = _per_entity_param_lf(source, "lifetime")
    bounded_life = (_resolve_per_period_lf(life_per, bounded_anchor, fill=0.0)
                       .rename({"value": "life"})
                       .select("e", "d", "life"))

    bounded_walk = period_walk_iterator(
        source, active_solve, bounded_anchor,
        period_in_use, period_in_use,
        window_method=WindowMethod.BOUNDED_INCLUSIVE_LOOKBACK,
        life_lf=bounded_life,
        factor_side=None)

    unbounded_anchor = unbounded_e.join(pwh_lf, how="cross")
    unbounded_walk = period_walk_iterator(
        source, active_solve, unbounded_anchor,
        period_in_use, period_in_use,
        window_method=WindowMethod.UNBOUNDED_FORWARD,
        life_lf=None,
        factor_side=None)

    # period_walk_iterator returns columns [e, d, d_all]; rename d→d_history,
    # d_all→d for the canonical schema.
    bounded_triples = bounded_walk.rename({"d": "d_history", "d_all": "d"})
    unbounded_triples = unbounded_walk.rename({"d": "d_history", "d_all": "d"})

    return (pl.concat([bounded_triples, unbounded_triples], how="vertical")
              .unique())


# ---------------------------------------------------------------------------
# §3.11 — p_entity_all_existing (existing-chain walk)
# ---------------------------------------------------------------------------


def p_entity_all_existing_from_handoff(
        source: "InputSource",
        active_solve: str | None,
        period_with_history: list[str],  # noqa: ARG001 — kept for symmetry
        period_in_use: list[str],
        *,
        p_entity_period_existing_capacity: "Param | None" = None,  # noqa: ARG001 — alt. carrier shape
        p_entity_previously_invested_capacity: "Param | None" = None,
        p_entity_divested: "Param | None" = None,
        ) -> "Param | None":
    """Lazy port of flextool's
    ``write_p_entity_existing_chain`` algorithm
    (``entity_period_calc_params.py:1381-1591``).

    The canonical formula on the consumer side
    (:func:`._read_capacity` legacy fallback,
    ``input.py:128-155``):

        all_existing[e, d] = pre_existing[e, d]
                          + p_entity_previously_invested_capacity[e, d]
                          − p_entity_divested[e]   (only if e ∈ entityDivest)

    where:

    * ``pre_existing[e, d]`` is ``entity.existing`` lifetime-gated
      (zero past expiry for ``reinvest_choice`` / ``no_investment``).
    * ``p_entity_previously_invested_capacity[e, d]`` is the chain-summed
      prior-solve invest at d (carried via the in-memory handoff;
      :class:`._solve_handoff.SolveHandoff.realized_invest`).
    * ``p_entity_divested[e]`` is the cumulative prior-solve divest
      scalar (also via handoff).

    On the first solve (no handoff carriers populated) all three terms
    after ``pre_existing`` are zero so the formula collapses to
    ``v = pre_existing[e, d]``.

    Parameters
    ----------
    p_entity_period_existing_capacity
        Legacy / placeholder; not consumed in this overload.  Kept on
        the signature for forward-compat with a future helper that
        walks ``edd_history × ppec_handoff`` directly.
    p_entity_previously_invested_capacity
        In-memory handoff Param ``[e, d, value]`` — chain-summed prior
        invest at d.  When None / empty → solve-first branch.
    p_entity_divested
        Handoff Param ``[e, value]`` cumulative prior divest.

    Returns Param ``[e, d, value]`` or None when no entities exist.
    """
    if not period_in_use:
        return None
    all_e_collected = _all_entities_lf(source).collect()
    if all_e_collected.height == 0:
        return None
    all_e_lf = _all_entities_lf(source)
    piu_lf = pl.LazyFrame({"d": period_in_use})
    grid_lf = all_e_lf.join(piu_lf, how="cross")

    pre_existing_lf = p_entity_pre_existing_lf(
        source, active_solve, period_in_use)

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

    # entityDivest set + p_entity_divested scalar.
    div_set_lf = entity_divest_set_lf(source).with_columns(
        is_divest=pl.lit(True))
    if (p_entity_divested is not None
            and p_entity_divested.frame.height > 0):
        ped_lf = (p_entity_divested.frame.lazy()
                     .select("e", pl.col("value").alias("divested")))
    else:
        ped_lf = pl.LazyFrame(schema={"e": pl.Utf8, "divested": pl.Float64})

    out = (grid_lf
              .join(pre_existing_lf.rename({"value": "pre"}),
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
                  value=pl.col("pre") + pl.col("ppic")
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
            "e": pl.Utf8, "d": pl.Utf8, "value": pl.Float64,
        })
    all_e_lf = _all_entities_lf(source)
    piu_lf = pl.LazyFrame({"d": period_in_use})
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
        return pl.LazyFrame(schema={"e": pl.Utf8, "d": pl.Utf8})
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
        return pl.LazyFrame(schema={"e": pl.Utf8, "d": pl.Utf8})

    # life_sum: per-entity scalar.  Compute per d_first then sum:
    # life_sum(e) = Σ_{d_first} (pdy[d_first] + lifetime[e, d_first]).
    pf_lf = pl.LazyFrame({"d_first": period_first})
    pyd_first = pyd_lf.rename({"d": "d_first", "yr": "yr_first"})
    e_dfirst = (in_methods.join(pf_lf, how="cross")
                          .rename({"d_first": "d"})
                          .pipe(lambda lf: lf))  # placeholder for symmetry
    # Resolve lifetime[e, d_first] via cascade.
    life_per = _per_entity_param_lf(source, "lifetime")
    life_resolved = (_resolve_per_period_lf(
        life_per, in_methods.join(pf_lf, how="cross").rename({"d_first": "d"}),
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

    piu_lf = pl.LazyFrame({"d": period_in_use})
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
                              workdir: Path) -> None:
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

    active_solve = _read_active_solve(workdir)
    period_in_use = _period_in_use_set(source, active_solve, workdir)
    period_with_history = (_read_period_with_history(workdir)
                              or list(period_in_use))

    ppic = getattr(flex_data, "p_entity_previously_invested_capacity", None)
    ped = getattr(flex_data, "p_entity_divested", None)

    try:
        pae = p_entity_all_existing_from_handoff(
            source, active_solve,
            period_with_history, period_in_use,
            p_entity_previously_invested_capacity=ppic,
            p_entity_divested=ped)
    except Exception:
        pae = None
    if pae is not None:
        flex_data.p_entity_all_existing = pae


__all__ = [
    # Method-enum constants (re-exported for direct-test use).
    "_INVEST_NOT_ALLOWED", "_DIVEST_NOT_ALLOWED",
    "_INVEST_TOTAL_METHODS", "_DIVEST_TOTAL_METHODS",
    "_INVEST_PERIOD_METHODS", "_DIVEST_PERIOD_METHODS",
    # Public lazy helpers.
    "entity_invest_set_lf", "entity_divest_set_lf",
    "e_invest_total_lf", "e_divest_total_lf",
    "edd_history_lf",
    "p_entity_all_existing_from_handoff", "p_entity_pre_existing_lf",
    "apply_existing_chain",
]
