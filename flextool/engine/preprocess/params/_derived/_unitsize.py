"""The unit size cascades (``virtual_unitsize`` or ``existing`` or :data:`UNITSIZE_DEFAULT`) of entities
and nodes, memoised per source."""
from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import polars as pl

from flextool.engine.input._axis_enums import alias_to_axis, schema_dtype

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource

from flextool.engine.preprocess.params._derived._scope import _enums, _try_entities, _try_param_explicit


# Canonical unitsize fallback for the ``virtual_unitsize OR existing OR
# <fallback>`` cascade.  NOTE: this is a FlexTool *modeling* default, not a
# schema ``default_value`` — the SpineDB schema declares ``virtual_unitsize``
# with default ``null`` (see ``flextool/schemas/spinedb_schema.json``); the
# 1000.0 fallback lives only in flextool's preprocessing
# (``entity_period_calc_params.py``).  Defined here so the cascade tails
# (:func:`_entity_unitsize_lf` / :func:`_node_unitsize_lf`) and the
# structural-completeness projection (``_arithmetic._unitsize_complete``)
# share ONE source of truth and cannot drift.
UNITSIZE_DEFAULT = 1000.0


# Memoization caches for the solve-INVARIANT unitsize cascades.  Both
# ``_entity_unitsize_lf`` and ``_node_unitsize_lf`` compute a pure
# coalesce (``virtual_unitsize OR existing OR 1000``) purely from
# ``source``; the whole-model ``SpineDbReader`` is reused across all
# rolls, so caching the collected cascade keyed on the source object
# collapses N per-roll recomputes to one.  WeakKeyDictionary so the
# entry is dropped when the source is garbage-collected.
_ENTITY_UNITSIZE_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


_NODE_UNITSIZE_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def unitsize_cascade(base: pl.LazyFrame,
                     vu: "pl.LazyFrame | None",
                     ex: "pl.LazyFrame | None",
                     *, key: str) -> pl.LazyFrame:
    """The unitsize cascade, ``[key, us]`` for every row of ``base``::

        unitsize = virtual_unitsize (if explicitly set above 0)
                  OR existing       (if explicitly set non-zero)
                  OR UNITSIZE_DEFAULT (1000.0)

    A ``virtual_unitsize`` of 0 or below is ignored, as if it were not set
    (``validate_basic_sanity`` warns about it).

    ``base`` is ``[key]`` (one row per entity); ``vu`` is ``[key, vu]``
    and ``ex`` is ``[key, ex]`` with only the explicitly set values
    (``None`` when there are none).  ``existing`` is a constant (one row
    per entity) or a period Map (one row per period): it is reduced to
    the per-entity MAX, which is non-zero iff any period is — the cascade
    only checks "non-zero".  ``virtual_unitsize`` is a constant in the
    schema; it is reduced the same way so a stray row cannot duplicate
    an entity.

    The ONE implementation of the cascade: the DB-side
    :func:`_entity_unitsize_lf` / :func:`_node_unitsize_lf` and the
    per-solve ``solve_data/p_entity_unitsize.csv`` emitter
    (``emit._dispatchers._compute_p_entity_unitsize``, which reads
    ``p_process`` / ``pd_process`` and ``p_node`` / ``pd_node``) all call
    it, so the per-solve CSV seed and the DB rebuild cannot disagree.
    """
    out = base
    for name, frame in (("vu", vu), ("ex", ex)):
        if frame is None:
            out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
            continue
        reduced = (frame.select(key, pl.col(name).cast(pl.Float64))
                   .group_by(key).agg(pl.col(name).max()))
        out = out.join(reduced, on=key, how="left")
    return out.with_columns(
        us=pl.when(pl.col("vu").fill_null(0.0) > 0.0)
              .then(pl.col("vu"))
              .when(pl.col("ex").fill_null(0.0) != 0.0)
              .then(pl.col("ex"))
              .otherwise(pl.lit(UNITSIZE_DEFAULT))
    ).select(key, "us")


def _entity_unitsize_lf(source: "InputSource") -> pl.LazyFrame:
    """Per-entity unitsize cascade across unit + node + connection,
    mirroring ``entity_period_calc_params.py:159-202``::

        unitsize = virtual_unitsize (if explicitly set above 0)
                  OR existing       (if explicitly set non-zero)
                  OR 1000.0

    Returns a lazy frame ``[e, us]`` covering every entity in
    ``unit ∪ node ∪ connection``.  Entities with neither
    ``virtual_unitsize`` nor ``existing`` get the canonical fallback
    ``1000.0``.

    For ``existing`` 1d_map(period) values, the per-period MAX is used
    as the cascade input (matches flextool's flat ``p_unit/p_node`` Map
    -agnostic read; the cascade only checks "non-zero", and the per-
    period MAX is non-zero iff any period is).

    Solve-invariant: the cascade depends only on ``source`` (entities +
    per-entity params), so the collected frame is memoized per source
    object (see ``_ENTITY_UNITSIZE_CACHE``).
    """
    try:
        cached = _ENTITY_UNITSIZE_CACHE.get(source)
    except TypeError:
        cached = None
    if cached is not None:
        return cached.lazy()
    base_parts: list[pl.LazyFrame] = []
    us_parts: list[pl.LazyFrame] = []
    ex_parts: list[pl.LazyFrame] = []
    for ec in ("unit", "node", "connection"):
        ents = _try_entities(source, ec)
        if ents is None or ents.height == 0:
            continue
        base_parts.append(ents.lazy().select(
            alias_to_axis("name", "e")))
        us = _try_param_explicit(source, ec, "virtual_unitsize")
        if us is not None and us.height > 0:
            us_parts.append(us.lazy().select(
                alias_to_axis("name", "e"),
                pl.col("value").cast(pl.Float64).alias("vu"),
            ))
        ex = _try_param_explicit(source, ec, "existing")
        if ex is not None and ex.height > 0:
            # ``existing`` may be a scalar (one row per entity) or a
            # 1d_map(period).  The period axis carries Spine's silent-
            # default name ``"x"`` — NOT ``"period"`` — so a column-name
            # check misfires and, via the downstream ``keep="last"``
            # dedup, takes the *last* period's value.  For a retiring
            # unit whose map ends at 0 (e.g. ``220→0`` by expiry) that
            # collapses the cascade input to 0, defaults unitsize to
            # ``1000`` and turns ``existing_count = capacity/unitsize``
            # into a spurious fraction (caps continuous online below 1
            # and forbids integer online entirely).  Take the per-entity
            # MAX over whatever period rows exist — index-name-agnostic
            # per CLAUDE.md Invariant #2; a scalar's single row makes the
            # group_by a no-op.  The cascade only checks "non-zero", and
            # the per-period MAX is non-zero iff any period is.
            # (The per-entity MAX is taken in :func:`unitsize_cascade`.)
            ex_parts.append(ex.lazy().select(
                alias_to_axis("name", "e"),
                pl.col("value").cast(pl.Float64).alias("ex")))
    if not base_parts:
        return pl.LazyFrame(schema={"e": schema_dtype(_enums, "e"),
                                     "us": pl.Float64})
    result = unitsize_cascade(
        pl.concat(base_parts).unique(),
        pl.concat(us_parts) if us_parts else None,
        pl.concat(ex_parts) if ex_parts else None,
        key="e",
    ).collect()
    try:
        _ENTITY_UNITSIZE_CACHE[source] = result
    except TypeError:
        pass
    return result.lazy()


def _node_unitsize_lf(source: "InputSource") -> pl.LazyFrame:
    """Per-node unitsize lazy frame following the flextool preprocess
    formula (entity_period_calc_params.py:186-202): use
    ``virtual_unitsize`` if explicitly set non-zero, else ``existing``
    if explicitly set non-zero, else 1000.0.

    Uses :func:`_try_param_explicit` so default-broadcast rows from
    SpineDbReader are suppressed — only entities with a parameter_value
    row count as "explicitly set".  This mirrors flextool's
    ``p_node.get(name, None)`` semantic: an absent row is *not*
    equivalent to a row with the schema default value.

    Returns a lazy frame with ``[n, us]`` for every node entity.

    Solve-invariant: the cascade depends only on ``source`` (node
    entities + per-node params), so the collected frame is memoized per
    source object (see ``_NODE_UNITSIZE_CACHE``).
    """
    try:
        cached = _NODE_UNITSIZE_CACHE.get(source)
    except TypeError:
        cached = None
    if cached is not None:
        return cached.lazy()
    us = _try_param_explicit(source, "node", "virtual_unitsize")
    ex = _try_param_explicit(source, "node", "existing")
    nodes = _try_entities(source, "node")
    if nodes is None or nodes.height == 0:
        return pl.LazyFrame(schema={"n": schema_dtype(_enums, "n"),
                                     "us": pl.Float64})
    # virtual_unitsize / existing may carry a period dim authored as a Spine
    # Map; the index column inherits the Map's ``index_name`` and may be
    # named ``period``, ``x``, etc.  :func:`unitsize_cascade` collapses any
    # number of rows per node to one (per-node MAX), so the index column
    # is not read.  Existing has no default (schema-None policy); rows
    # present == explicit.
    def _rows(frame, name):
        if frame is None or frame.height == 0:
            return None
        return frame.lazy().select(alias_to_axis("name", "n"),
                                   pl.col("value").cast(pl.Float64).alias(name))

    result = unitsize_cascade(
        nodes.lazy().select(alias_to_axis("name", "n")),
        _rows(us, "vu"), _rows(ex, "ex"), key="n",
    ).collect()
    try:
        _NODE_UNITSIZE_CACHE[source] = result
    except TypeError:
        pass
    return result.lazy()
