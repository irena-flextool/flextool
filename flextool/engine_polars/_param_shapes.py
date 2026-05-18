"""Δ.17c — centralized parameter-shape resolver.

The user's authoritative directive (Δ.17c dispatch, Gap C):

    There is no guarantee that e.g. 2d_map is (period, time). There is
    multi-level reliance on different factors. The source of truth
    should be three things:
    1. read from database the dimensionality of the parameter.
    2. list of parameter specific allowed dimensionalities e.g.
       p_efficiency[period,time].
    3. If parameter has multiple choices for e.g. 1d-map like
       p_efficiency[period], p_efficiency[time], then you need to read
       the dimension index label from the database. If it says period,
       it is period; if it says time, it is time. If does not match,
       error to the user.

This module implements those three things:

* :data:`PARAM_ALLOWED_SHAPES` — per-(entity_class, parameter_name)
  registry of valid shapes.  Each entry lists the shapes flextool's
  preprocessing accepts (mirrors ``flextool/flextoolrunner/preprocessing/
  entity_period_calc_params.py``'s ``write_pdtX`` / ``write_pdX`` cascade
  domains, see docstring on each entry).
* :func:`resolve_param_shape` — DB-driven shape detection: reads the
  parameter's actual nesting depth and per-level index_name labels
  from the DB via :meth:`InputSource.parameter_shape_info`, validates
  against the allow-list, raises :class:`FlexToolConfigError` on
  mismatch.
* :func:`broadcast_to_period_time` — produces ``[entity, d, t, value]``
  by broadcasting the parameter frame over the active solve's
  ``(d, t)`` axis according to the resolved shape.
* :func:`broadcast_to_period` — produces ``[entity, d, value]`` for
  parameters whose allowed shapes are scalar / 1d_map[period] only
  (e.g. ``co2_max_period``).

Why a centralized resolver and not per-helper column checks?
The previous Δ.17b approach inferred the shape from the column names
of the source frame.  That's flawed when:

* A 2d_map column ordering depends on which dim was outermost in the
  Map — *e.g.* ``2d_map[time, period]`` writes a column ``[t, period]``
  but the helper expected ``[period, t]``.  Column-shape detection
  silently produced the wrong broadcast.
* The DB carries an unexpected (non-period, non-time) index_name —
  e.g. ``branch`` — and the column was renamed to the canonical
  default ``period`` by :class:`SpineDbReader._discover_index_cols`.
  The helper saw a "period" column and broadcast over (d, t) when the
  authoring intent was different.

The data-driven resolver avoids both: it inspects the DB-reported
``index_name`` per level and validates against the per-parameter
allow-list.  Mismatches surface as :class:`FlexToolConfigError`.

Computational efficiency
------------------------

:func:`resolve_param_shape` is called once per (entity_class,
parameter_name) per ``apply_direct_params`` invocation — at most ~10
calls per ``load_flextool``, all cached at the source-plugin level.
The resolver itself does no per-row work; it only inspects the
parameter's structural metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

from flextool.engine_polars._solve_state import FlexToolConfigError

if TYPE_CHECKING:
    from flextool.engine_polars._input_source import InputSource


# ---------------------------------------------------------------------------
# Shape enum + per-parameter allow-list
# ---------------------------------------------------------------------------


class Shape(Enum):
    """Recognised parameter shapes for the (period, time)-broadcast family.

    The string values mirror the user's notation in the dispatch /
    open-issues doc (e.g. ``"1d_map[period]"``).
    """

    SCALAR = "scalar"
    MAP_PERIOD = "1d_map[period]"
    MAP_TIME = "1d_map[time]"
    MAP_PERIOD_TIME = "2d_map[period,time]"
    MAP_TIME_PERIOD = "2d_map[time,period]"


# Per-(entity_class, parameter_name) allow-list.  Each entry lists the
# shapes flextool's preprocessing accepts for that parameter, derived
# by reading flextool/flextoolrunner/preprocessing/entity_period_calc_params.py.
#
# The cascade in `write_pdtX` (mod L1227 etc.) is:
#     pbt → pd → pt → p → def1 → 0
# meaning: 2d_map(period, time) preferred, then 1d_map(period), then
# 1d_map(time), then scalar.  All four shapes are valid authoring options
# for parameters in the relevant ``X_TIME_PARAM`` taxonomy.
#
# `write_pdX` cascade (mod L1115) is:
#     pd → period__branch fold → p → 5000-default-set → 0
# meaning: 1d_map(period) or scalar, no time variant.  Used for
# parameters in ``X_PERIOD_PARAM`` but NOT ``X_TIME_PARAM``.
#
# `write_pdtCommodity` cascade is pt → pd → p → 0 (no pbt).
PARAM_ALLOWED_SHAPES: dict[tuple[str, str], set[Shape]] = {
    # ─── group: (period, time) Params ────────────────────────────────────
    # GROUP_TIME_PARAM = {co2_price, max_instant_flow, min_instant_flow}.
    # Cascade: pbt → pd → pt → p (entity_period_calc_params:write_pdtGroup).
    ("group", "co2_price"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME,
    },
    ("group", "max_instant_flow"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME,
    },
    ("group", "min_instant_flow"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME,
    },
    # ─── group: period-only Params (no time variant) ─────────────────────
    # GROUP_PERIOD_PARAM \ GROUP_TIME_PARAM ⊇ {co2_max_period}.
    # Cascade: pd → p (entity_period_calc_params:write_pdGroup).
    ("group", "co2_max_period"): {Shape.SCALAR, Shape.MAP_PERIOD},
    # ─── node: (period, time) Params ─────────────────────────────────────
    # NODE_TIME_PARAM ⊇ {availability, storage_state_reference_value}.
    # Cascade: pbt → pt → pd → p (write_pdtNode, time_first_priority=True).
    ("node", "availability"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME,
    },
    ("node", "storage_state_reference_value"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME,
    },
    # ─── unit / connection: (period, time) Params ────────────────────────
    # PROCESS_TIME_PARAM ⊇ {availability}.
    # Cascade: pbt → pd → pt → p (write_pdtProcess).  ``processes`` in
    # process_arc_unions.py:write_param_in_use_sets reads the ``process.csv``
    # union of unit + connection, so both classes participate in the LP's
    # ``p_process_availability`` (the Spine schema declares the parameter
    # on each class independently).
    ("unit", "availability"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME,
    },
    ("connection", "availability"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME,
    },
    # ─── commodity: time Params ──────────────────────────────────────────
    # commodityTimeParam = {price}.
    # Cascade: pt → pd → p → 0 (write_pdtCommodity, no pbt).
    ("commodity", "price"): {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME,
    },
}


# Allowed shapes for the (entity, period) family — used by
# :func:`broadcast_to_period`.  These parameters never authorise a time
# axis.  Shapes outside ``{SCALAR, MAP_PERIOD}`` raise on detection.
_PERIOD_ONLY_SHAPES: frozenset[Shape] = frozenset(
    (Shape.SCALAR, Shape.MAP_PERIOD)
)


# ---------------------------------------------------------------------------
# ResolvedShape — the resolver's output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedShape:
    """Outcome of :func:`resolve_param_shape`.

    Carries everything a broadcast helper needs:

    * :attr:`shape` — the canonical :class:`Shape`.
    * :attr:`frame` — the parameter frame from
      :meth:`InputSource.parameter_explicit` (or :meth:`parameter`),
      or ``None`` when the parameter has no explicit rows.
    * :attr:`entity_dim_columns` — the entity's dim columns (for 0-dim
      classes this is ``["name"]``).
    * :attr:`period_index_column` — the column holding the period index
      (or ``None`` for scalar / time-only shapes).  Always ``"period"``
      when present (the SpineDbReader normalises ``index_name`` to the
      Map's authored name; the registry validates it's authored as
      "period").
    * :attr:`time_index_column` — the column holding the time index
      (or ``None`` for scalar / period-only shapes).  Always ``"t"``
      when present (per ``SpineDbReader._discover_index_cols`` which
      maps "time" → "t").
    """

    shape: Shape
    frame: "pl.DataFrame | None"
    entity_dim_columns: tuple[str, ...]
    period_index_column: "str | None"
    time_index_column: "str | None"


# ---------------------------------------------------------------------------
# resolve_param_shape — DB-driven shape detection + validation
# ---------------------------------------------------------------------------


def _allowed_shape_names(allowed: "set[Shape]") -> str:
    """Render a stable, sorted comma-separated name list for error messages."""
    return ", ".join(sorted(s.value for s in allowed))


# Labels that the spinedb_api default-fills when the author didn't
# specify ``index_name`` on a Map.  Treated as "silent default" — the
# resolver returns None (the caller falls through to the seed / leaves
# the field unset) instead of raising.  Per the user advice, the error
# path applies to *explicit* mismatches, not authoring oversights.
_SILENT_DEFAULT_LABELS: frozenset[str] = frozenset(("x", ""))


def _normalise_label(n: "str | None") -> "str | None":
    """Lowercase / strip / collapse silent defaults to ``None``."""
    if n is None:
        return None
    if not isinstance(n, str):
        return None
    s = n.strip().lower()
    if not s or s in _SILENT_DEFAULT_LABELS:
        return None
    return s


# Per-shape depth + per-level canonical label, used to disambiguate
# silent-default ``index_name`` labels by consulting the per-parameter
# allow-list.  Each entry: shape → tuple of canonical labels per depth
# level (length 0/1/2).  Mirrors the enum membership.
_SHAPE_LABELS: "dict[Shape, tuple[str, ...]]" = {
    Shape.SCALAR:           (),
    Shape.MAP_PERIOD:       ("period",),
    Shape.MAP_TIME:         ("time",),
    Shape.MAP_PERIOD_TIME:  ("period", "time"),
    Shape.MAP_TIME_PERIOD:  ("time", "period"),
}


def _infer_silent_default_labels(
    raw_labels: "list[str | None]",
    allowed: "set[Shape]",
) -> "list[str | None]":
    """Fill silent-default ``index_name`` slots from the allow-list.

    When the DB authored a Map without ``index_name`` (the spinedb_api
    silent default ``"x"``), the raw label collapses to ``None`` through
    :func:`_normalise_label`.  For parameters whose allow-list has a
    *unique* choice at the observed depth + position, the silent default
    is unambiguous: the registry only permits one shape with that
    depth/position, so the author's intent is recoverable without
    rewriting the DB.

    Example: ``("group", "co2_max_period")`` admits
    ``{SCALAR, MAP_PERIOD}``.  Depth=1 admits only ``MAP_PERIOD`` →
    position-0 label is unambiguously ``"period"``.  Depth=1 for
    ``("commodity", "price")`` admits both ``MAP_PERIOD`` and
    ``MAP_TIME``: position-0 stays ``None`` (genuinely ambiguous; the
    caller falls back).

    Returns a fresh list with disambiguated labels filled in; original
    non-silent labels are passed through unchanged.  The output has the
    same length as ``raw_labels``.
    """
    normalised = [_normalise_label(n) for n in raw_labels]
    depth = len(normalised)
    if all(n is not None for n in normalised):
        # No silent defaults to disambiguate.
        return normalised
    # Per the allow-list, which canonical-label tuples have matching
    # depth?  Only shapes at the observed depth.
    candidates = [
        _SHAPE_LABELS[s] for s in allowed if len(_SHAPE_LABELS[s]) == depth
    ]
    if not candidates:
        # No allowed shape at this depth — leave it for the post-resolve
        # allow-list check / unresolved-shape branch.
        return normalised
    # For each position, the unique label across all candidates fills
    # silent defaults.  Mixed positions stay ambiguous.
    filled: "list[str | None]" = list(normalised)
    for pos in range(depth):
        if filled[pos] is not None:
            continue
        labels_at_pos = {cand[pos] for cand in candidates}
        if len(labels_at_pos) == 1:
            filled[pos] = next(iter(labels_at_pos))
    return filled


def _shape_from_indices(index_names: "list[str | None]",
                        ) -> "Shape | None":
    """Map a depth-ordered list of raw ``index_name`` labels to a
    :class:`Shape`.

    The user advice "If it says period, it is period; if it says time,
    it is time. If does not match, error to the user." is enforced
    here for *explicit* mismatches.  Empty / ``None`` / spinedb_api's
    ``"x"`` silent default propagate as ambiguity → ``return None``,
    letting the caller fall through to whatever non-resolver pathway
    the seed used to fill (see Δ.18+ punch-list to clean up fixtures
    that author Maps without ``index_name``).

    Returns ``None`` when at least one level is a silent default
    (cannot disambiguate without DB-side authoring fix).
    Returns the resolved :class:`Shape` when every level is explicitly
    "period" or "time" / "t".
    Raises :class:`_UnrecognisedIndex` when at least one level carries
    an explicit label outside that set (e.g. "tier", "branch").
    """
    # Canonical normalisation: silent defaults collapse to None.
    normalised = [_normalise_label(n) for n in index_names]

    # Silent defaults at any level → return None (ambiguous; caller
    # falls back).  We DON'T raise here because the spinedb_api default
    # is a silent oversight, not a deliberate mismatch.
    if any(n is None for n in normalised):
        return None

    if len(normalised) == 0:
        return Shape.SCALAR
    if len(normalised) == 1:
        n0 = normalised[0]
        if n0 == "period":
            return Shape.MAP_PERIOD
        if n0 in ("time", "t"):
            return Shape.MAP_TIME
        raise _UnrecognisedIndex(n0, depth=0)
    if len(normalised) == 2:
        n0, n1 = normalised[0], normalised[1]
        if n0 == "period" and n1 in ("time", "t"):
            return Shape.MAP_PERIOD_TIME
        if n0 in ("time", "t") and n1 == "period":
            return Shape.MAP_TIME_PERIOD
        raise _UnrecognisedIndex(f"({n0!r}, {n1!r})", depth=0)
    # Higher depth — not supported here.
    raise _UnrecognisedIndex(
        f"depth {len(normalised)} (got {normalised})", depth=0,
    )


class _UnrecognisedIndex(Exception):
    """Internal sentinel for :func:`_shape_from_indices` — caller maps
    to :class:`FlexToolConfigError` with full context."""

    def __init__(self, label: str, depth: int):
        super().__init__(label)
        self.label = label
        self.depth = depth


def _disambiguate_shape_by_value_domain(
    df: "pl.DataFrame",
    ent_cols: "list[str]",
    resolved_labels: "list[str | None]",
    allowed: "set[Shape]",
    period_filter: "pl.DataFrame | None",
) -> "Shape | None":
    """Value-domain probing fallback for silent-default ``index_name`` Maps.

    When the registry permits multiple shapes at the observed depth
    (e.g. ``("group", "co2_price")`` admits both ``MAP_PERIOD`` and
    ``MAP_TIME``) and the DB carried no ``index_name`` label, we
    cannot decide structurally.  Mirroring the legacy CSV pipeline
    (``_timeline.separate_period_and_timeseries_data``), we look at
    the Map's actual index values: a Map keyed by ``y2019, y2020, ...``
    is :class:`Shape.MAP_PERIOD`; a Map keyed by ``t0001, t0002, ...``
    is :class:`Shape.MAP_TIME`.

    Requires *period_filter* with columns ``d`` (periods) and ``t``
    (timesteps).  Returns ``None`` when probing is impossible (no
    filter, no candidates, mixed values matching neither set).

    Currently only handles depth-1 ambiguity (a single Map level).  For
    depth-2 Maps we don't yet probe — the registry's allow-list usually
    fixes the ordering by allowing only one 2d shape at a time, so the
    silent-default case there is rare; the standard structural pathway
    handles every fixture observed in the cascade gate as of 2026-05.
    """
    if period_filter is None:
        return None
    depth = len(resolved_labels)
    if depth != 1:
        return None
    # Identify the silent-default level (the only position with None
    # in resolved_labels — every other depth-1 case yielded a definite
    # shape above and never reached this fallback).
    if resolved_labels[0] is not None:
        return None
    # The candidate set at depth 1 is the intersection of *allowed* with
    # depth-1 shapes: at most MAP_PERIOD and MAP_TIME.
    candidates = {s for s in allowed
                  if len(_SHAPE_LABELS[s]) == 1
                  and s in (Shape.MAP_PERIOD, Shape.MAP_TIME)}
    if not candidates:
        return None
    # Locate the index column (the first non-entity, non-value column).
    non_ent_cols = [
        c for c in df.columns if c not in ent_cols and c != "value"
    ]
    if len(non_ent_cols) != 1:
        # Defensive: depth-1 frame should carry exactly one index column.
        return None
    idx_col = non_ent_cols[0]
    # Get distinct index values from the frame.
    try:
        idx_values = (
            df.lazy()
              .select(pl.col(idx_col).cast(pl.Utf8))
              .unique()
              .collect()
              .get_column(idx_col)
              .to_list()
        )
    except Exception:
        return None
    if not idx_values:
        return None
    idx_set = {str(v) for v in idx_values if v is not None}
    # Build the period / timestep universes from period_filter.
    pf_cols = period_filter.columns
    periods_set: "set[str] | None" = None
    timesteps_set: "set[str] | None" = None
    if "d" in pf_cols:
        periods_set = set(
            period_filter.lazy()
                         .select(pl.col("d").cast(pl.Utf8))
                         .unique()
                         .collect()
                         .get_column("d")
                         .to_list()
        )
    if "t" in pf_cols:
        timesteps_set = set(
            period_filter.lazy()
                         .select(pl.col("t").cast(pl.Utf8))
                         .unique()
                         .collect()
                         .get_column("t")
                         .to_list()
        )
    # All-or-nothing: every observed index value must lie in exactly
    # one of the two universes (and the corresponding shape must be in
    # the candidate set).  Mixed → genuinely ambiguous → don't guess.
    in_periods = bool(periods_set) and idx_set.issubset(periods_set)
    in_timesteps = bool(timesteps_set) and idx_set.issubset(timesteps_set)
    if in_periods and not in_timesteps and Shape.MAP_PERIOD in candidates:
        return Shape.MAP_PERIOD
    if in_timesteps and not in_periods and Shape.MAP_TIME in candidates:
        return Shape.MAP_TIME
    return None


def _read_index_names_from_source(
    source: "InputSource",
    entity_class: str,
    parameter_name: str,
) -> "list[str | None]":
    """Read the raw per-level ``index_name`` labels from the source.

    Honours the user advice "read from database the dimensionality of
    the parameter" and "read the dimension index label from the
    database".

    Implementation:

    * If the source provides :meth:`parameter_shape_info`, use it (the
      canonical path — see :class:`SpineDbReader.parameter_shape_info`).
    * Otherwise (e.g. :class:`InMemoryReader` in unit tests), inspect
      the parameter frame's columns directly.  This is the legacy path;
      InMemoryReader test fixtures author frames with column names
      already matching the resolved shape.

    Returns a list of raw labels per nesting depth (length 0 for
    scalar; length 1/2 for maps).  Labels may be ``None`` when the DB
    didn't author one — the caller will raise via the
    :class:`_UnrecognisedIndex` path.
    """
    fn = getattr(source, "parameter_shape_info", None)
    if fn is not None:
        return list(fn(entity_class, parameter_name))
    # Fallback path (InMemoryReader): infer from the frame's column
    # names, treating the first non-entity column as depth-0 etc.
    df = _try_parameter_frame(source, entity_class, parameter_name)
    if df is None or df.height == 0:
        return []
    ent_cols = _entity_dim_columns_for_frame(df, source, entity_class)
    cols = [c for c in df.columns if c not in ent_cols and c != "value"]
    # The frame's column names are the post-normalisation labels
    # (period stays "period"; time → "t" via SpineDbReader).  We translate
    # back to canonical labels for the shape resolver.
    out: list[str | None] = []
    for c in cols:
        if c == "period":
            out.append("period")
        elif c in ("t", "time"):
            out.append("time")
        else:
            out.append(c)
    return out


def _try_parameter_frame(
    source: "InputSource",
    entity_class: str,
    parameter_name: str,
) -> "pl.DataFrame | None":
    """Wrap ``source.parameter_explicit`` / ``source.parameter`` with
    None-on-KeyError semantics.  Used by both the InMemory shape-info
    fallback and by :func:`resolve_param_shape` to fetch the data.

    Δ.28 — when ``parameter_explicit`` returns an empty frame (no
    explicit overrides for any entity), fall through to ``parameter``
    so the schema default (e.g. ``availability = 1.0``) is consulted.
    Without this fall-through, the resolver dropped fields whose
    Spine value is purely the default (``p_process_availability`` on
    fixtures where every unit/connection inherits the 1.0 default,
    e.g. ``work_lh2_three_region``) — the slow path's CSV
    preprocessing always writes the default-broadcast rows so the
    fast path was missing data.

    Parameters with ``default_value=None`` (the §4.5 None-skip family)
    behave unchanged: ``parameter`` still returns the explicit-only
    frame, so an empty result remains an empty result and the resolver
    correctly drops the field.
    """
    try:
        explicit = source.parameter_explicit(entity_class, parameter_name)
    except (KeyError, AttributeError):
        explicit = None
    if explicit is not None and explicit.height > 0:
        return explicit
    try:
        return source.parameter(entity_class, parameter_name)
    except KeyError:
        return explicit


def _entity_dim_columns_for_frame(
    df: "pl.DataFrame",
    source: "InputSource",
    entity_class: str,
) -> list[str]:
    """Best-effort recovery of the entity dim columns of *df* by
    consulting :meth:`InputSource.entities` for the class.

    For 0-dim classes the column name is ``"name"``; for n-relationships
    it's the dim-class names with repeats disambiguated.  When
    :meth:`entities` raises KeyError (unknown class), fall back to
    ``["name"]`` — defensive; the registry only references known
    classes.
    """
    try:
        ent = source.entities(entity_class)
        cols = ent.columns
        if cols:
            return list(cols)
    except KeyError:
        pass
    return ["name"]


def resolve_param_shape(
    source: "InputSource",
    entity_class: str,
    parameter_name: str,
    period_filter: "pl.DataFrame | None" = None,
) -> "ResolvedShape | None":
    """DB-driven shape detection + per-parameter allow-list validation.

    Steps:

    1. Look up :data:`PARAM_ALLOWED_SHAPES` for ``(entity_class,
       parameter_name)``.  Missing entry → :class:`FlexToolConfigError`
       (the registry must explicitly list every parameter routed
       through the resolver).
    2. Fetch the parameter frame via
       :meth:`InputSource.parameter_explicit` (falling back to
       :meth:`parameter`).  Empty / None → return ``None`` (the caller
       drops this parameter from FlexData).
    3. Read the raw per-level ``index_name`` labels from the DB.
    4. Map the labels to a :class:`Shape`.  Unrecognised labels OR a
       shape outside the allow-list → :class:`FlexToolConfigError`
       with full context (parameter name, observed labels, allowed
       shapes).

    When the index_name labels are silent-default (e.g. spinedb_api's
    ``"x"``) AND ``period_filter`` is supplied, the resolver probes the
    index column values against the active solve's known periods /
    timesteps (mirroring the legacy CSV pipeline's value-domain
    discrimination in
    :func:`flextool.engine_polars._timeline.separate_period_and_timeseries_data`).
    This recovers Rivendell-style fixtures where ``group.co2_price`` is
    authored as ``Map(period→value)`` but the author omitted
    ``index_name`` so the DB stores the silent ``"x"`` label.  Without
    this probing the resolver returns ``None`` and ``p_co2_price`` is
    silently dropped while the feature gate (driven by topology, not
    data) stays active — the LP build then aborts at
    :func:`flextool.engine_polars.model.build_flextool`'s ``CO2_PRICE``
    invariant check.

    Returns ``None`` only when the parameter has no explicit rows AND
    no scalar default to broadcast (the source plugin's None-skip per
    §4.5).  Otherwise returns a :class:`ResolvedShape` carrying
    everything :func:`broadcast_to_period_time` /
    :func:`broadcast_to_period` need.
    """
    key = (entity_class, parameter_name)
    allowed = PARAM_ALLOWED_SHAPES.get(key)
    if allowed is None:
        raise FlexToolConfigError(
            f"resolve_param_shape: parameter {key!r} is not in "
            f"PARAM_ALLOWED_SHAPES.  Add an explicit allow-list entry "
            f"to flextool/engine_polars/_param_shapes.py.")

    df = _try_parameter_frame(source, entity_class, parameter_name)
    if df is None or df.height == 0:
        return None

    ent_cols = _entity_dim_columns_for_frame(df, source, entity_class)
    raw_labels = _read_index_names_from_source(
        source, entity_class, parameter_name)
    # Δ.17c follow-up — disambiguate silent-default ``index_name`` labels
    # against the per-parameter allow-list.  For parameters whose
    # registry entry permits a unique shape at the observed (depth,
    # position), the silent default is unambiguous and we can recover
    # the author's intent without rewriting the DB.  Required to keep
    # the Spine path on parity with the legacy CSV pipeline, which
    # picks index dimensionality from value-domain probing (period
    # tokens vs. timestep tokens) rather than the index_name label —
    # see :func:`flextool.engine_polars._timeline.separate_period_and_timeseries_data`.
    resolved_labels = _infer_silent_default_labels(raw_labels, allowed)
    try:
        shape = _shape_from_indices(resolved_labels)
    except _UnrecognisedIndex as exc:
        # User advice: "If does not match, error to the user."
        # We're strict only about *explicit* mismatches (e.g. ``branch`` on
        # a parameter whose allow-list is {scalar, period, time}).  Silent
        # defaults (empty / None / ``x``) collapse to ``shape=None`` below
        # via :func:`_shape_from_indices` and trigger the soft-fallback
        # path instead.
        raise FlexToolConfigError(
            f"Parameter ({entity_class!r}, {parameter_name!r}) carries an "
            f"unrecognised dimension index label {exc.label} (depth "
            f"{exc.depth}).  Allowed shapes: {_allowed_shape_names(allowed)}. "
            "Edit the source database so the Map's index_name reads "
            "'period' or 'time' (matching one of the allowed shapes), or "
            "extend PARAM_ALLOWED_SHAPES if a new shape is intended."
        ) from None

    if shape is None:
        # Ambiguous shape — at least one Map level carries the spinedb_api
        # silent default and the per-parameter allow-list permits multiple
        # interpretations at that depth.  Try value-domain probing against
        # the active solve's known periods / timesteps when *period_filter*
        # is supplied — this mirrors the legacy CSV pipeline's
        # ``separate_period_and_timeseries_data`` discriminator (a Map's
        # index values reveal whether it indexes by period or timestep,
        # regardless of the silent ``"x"`` index_name).
        shape = _disambiguate_shape_by_value_domain(
            df, ent_cols, resolved_labels, allowed, period_filter,
        )
        if shape is None:
            # Still ambiguous (no period_filter, or values match neither
            # the period nor the timestep set).  Per Δ.17c policy: don't
            # raise; return None so the caller falls back to its
            # non-resolver pathway (typically the seed CSV).  Δ.18+
            # punch-list: re-author fixtures so every Map level carries
            # an explicit ``index_name``.
            return None
        # Update the resolved labels so the downstream rename block sees
        # the disambiguated label and renames the silent-default column
        # (typically ``"x"``) to ``"period"`` or ``"t"``.
        resolved_labels = list(_SHAPE_LABELS[shape])

    if shape not in allowed:
        # Render observed shape for the message — labels can be empty
        # so reconstruct from the labels list.
        observed = (shape.value if shape is not None
                     else f"index_names={resolved_labels}")
        raise FlexToolConfigError(
            f"Parameter ({entity_class!r}, {parameter_name!r}) was authored "
            f"as {observed} but allowed shapes are: "
            f"{_allowed_shape_names(allowed)}.  Edit the source database "
            "or extend PARAM_ALLOWED_SHAPES."
        )

    # Δ.17c follow-up — when the resolver filled silent-default
    # ``index_name`` labels from the allow-list, the source frame's
    # columns still carry the silent-default names (e.g. ``"x"``) used
    # by :meth:`SpineDbReader._discover_index_cols`.  Rename them to the
    # canonical broadcast keys (``"period"`` / ``"t"``) so downstream
    # broadcasters can locate the index columns.
    df_for_broadcast = df
    if any(_normalise_label(r) is None for r in raw_labels):
        rename_map: "dict[str, str]" = {}
        # raw_labels and resolved_labels are aligned per depth.  The
        # SpineDbReader put non-entity, non-value columns in depth order;
        # walk them in parallel.
        non_ent_cols = [
            c for c in df.columns if c not in ent_cols and c != "value"
        ]
        for col, raw, resolved_lbl in zip(
            non_ent_cols, raw_labels, resolved_labels,
        ):
            if _normalise_label(raw) is not None:
                continue
            if resolved_lbl == "period" and col != "period":
                rename_map[col] = "period"
            elif resolved_lbl in ("time", "t") and col != "t":
                rename_map[col] = "t"
        if rename_map:
            df_for_broadcast = df.rename(rename_map)

    # Resolve the column names per the (possibly renamed) frame.
    period_col = "period" if "period" in df_for_broadcast.columns else None
    time_col = "t" if "t" in df_for_broadcast.columns else None
    return ResolvedShape(
        shape=shape,
        frame=df_for_broadcast,
        entity_dim_columns=tuple(ent_cols),
        period_index_column=period_col,
        time_index_column=time_col,
    )


# ---------------------------------------------------------------------------
# Broadcast helpers — produce per-(entity, d, t) and per-(entity, d) frames
# ---------------------------------------------------------------------------


def broadcast_to_period_time(
    resolved: "ResolvedShape | None",
    entity_dim_alias: str,
    period_filter: "pl.DataFrame | None",
    *,
    filter_zero: bool = False,
) -> "Param | None":
    """Materialise ``Param((entity_dim_alias, "d", "t"), ...)`` from the
    resolved shape, broadcasting as needed against ``period_filter``'s
    ``(d, t)`` axis.

    *period_filter* must carry ``[d, t]`` columns (typically the active
    solve's ``flex_data.dt`` frame).  When the resolved shape requires
    a broadcast (scalar, 1d_map[period], 1d_map[time]) but
    ``period_filter`` is missing or empty, returns ``None`` — the
    caller falls through to whatever non-broadcast pathway it used
    before.

    *filter_zero* mirrors the CSV cascade's "drop rows where the
    explicit value is 0" semantic for fields like ``co2_price``
    (preprocessing emits zero-defaults but downstream gates filter
    them out).

    Returns ``None`` when:

    * *resolved* is ``None``,
    * the broadcast produces an empty frame,
    * a required ``period_filter`` is missing.
    """
    if resolved is None or resolved.frame is None:
        return None
    df = resolved.frame
    if df.height == 0:
        return None
    if period_filter is None or period_filter.height == 0:
        return None
    pf_cols = set(period_filter.columns)
    if not {"d", "t"}.issubset(pf_cols):
        return None

    # Entity-dim alias: rename the source's dim column(s) to a single
    # name (the Param's first dim).  All resolver-routed parameters
    # operate on 0-dim object classes (group, node, unit, commodity)
    # so there's a single ``"name"`` column to alias.
    if resolved.entity_dim_columns != ("name",):
        # Defensive; the registry currently only contains 0-dim classes.
        raise FlexToolConfigError(
            f"broadcast_to_period_time: only 0-dim entity classes are "
            f"supported; got dims {resolved.entity_dim_columns}")
    lf = (df.lazy()
            .rename({"name": entity_dim_alias})
            .filter(pl.col("value").is_not_null()))
    if filter_zero:
        lf = lf.filter(pl.col("value") != 0.0)
    dt_lf = period_filter.lazy().select("d", "t").unique()

    shape = resolved.shape
    if shape == Shape.MAP_PERIOD_TIME:
        # Direct fill — already (d, t)-keyed.  Inner-join on dt to
        # restrict to the active solve's periods.
        out = (lf.rename({"period": "d"})
                  .select(entity_dim_alias, "d", "t", "value")
                  .join(dt_lf, on=["d", "t"], how="inner")
                  .collect())
    elif shape == Shape.MAP_TIME_PERIOD:
        # Same as MAP_PERIOD_TIME — column renames.  The frame already
        # has both ``period`` and ``t`` columns regardless of authoring
        # order (the SpineDbReader unrolls a 2d_map into a flat frame).
        out = (lf.rename({"period": "d"})
                  .select(entity_dim_alias, "d", "t", "value")
                  .join(dt_lf, on=["d", "t"], how="inner")
                  .collect())
    elif shape == Shape.MAP_PERIOD:
        # 1d_map(period) → broadcast across t per (entity, d).
        out = (lf.rename({"period": "d"})
                  .select(entity_dim_alias, "d", "value")
                  .join(dt_lf, on="d", how="inner")
                  .select(entity_dim_alias, "d", "t", "value")
                  .collect())
    elif shape == Shape.MAP_TIME:
        # 1d_map(time) → broadcast across d per (entity, t).
        out = (lf.select(entity_dim_alias, "t", "value")
                  .join(dt_lf, on="t", how="inner")
                  .select(entity_dim_alias, "d", "t", "value")
                  .collect())
    elif shape == Shape.SCALAR:
        # scalar → broadcast across (d, t) per entity.
        out = (lf.select(entity_dim_alias, "value")
                  .join(dt_lf, how="cross")
                  .select(entity_dim_alias, "d", "t", "value")
                  .collect())
    else:  # pragma: no cover — guarded by allow-list check.
        raise FlexToolConfigError(
            f"broadcast_to_period_time: unhandled shape {shape!r}")
    if out.height == 0:
        return None
    return Param((entity_dim_alias, "d", "t"), out.lazy())


def broadcast_to_period(
    resolved: "ResolvedShape | None",
    entity_dim_alias: str,
    period_filter: "pl.DataFrame | None",
    *,
    filter_zero: bool = False,
    filter_null: bool = True,
) -> "Param | None":
    """Materialise ``Param((entity_dim_alias, "d"), ...)`` from the
    resolved shape — for parameters whose allow-list excludes time.

    Handles ``Shape.SCALAR`` (broadcast across periods in
    *period_filter*) and ``Shape.MAP_PERIOD`` (direct fill, optionally
    filtered to the active periods).  Other shapes raise
    :class:`FlexToolConfigError` (the resolver should have rejected
    them already; this is a defensive guard).
    """
    if resolved is None or resolved.frame is None:
        return None
    df = resolved.frame
    if df.height == 0:
        return None

    if resolved.entity_dim_columns != ("name",):
        raise FlexToolConfigError(
            f"broadcast_to_period: only 0-dim entity classes are "
            f"supported; got dims {resolved.entity_dim_columns}")

    lf = df.lazy().rename({"name": entity_dim_alias})
    if filter_null:
        lf = lf.filter(pl.col("value").is_not_null())
    if filter_zero:
        lf = lf.filter(pl.col("value") != 0.0)

    shape = resolved.shape
    if shape == Shape.SCALAR:
        if period_filter is None or period_filter.height == 0:
            return None
        periods = period_filter.lazy().select("d").unique()
        out = (lf.select(entity_dim_alias, "value")
                  .join(periods, how="cross")
                  .select(entity_dim_alias, "d", "value")
                  .collect())
    elif shape == Shape.MAP_PERIOD:
        out = (lf.rename({"period": "d"})
                  .select(entity_dim_alias, "d", "value"))
        if period_filter is not None and period_filter.height > 0:
            out = out.join(
                period_filter.lazy().select("d").unique(),
                on="d", how="inner",
            )
        out = out.collect()
    else:
        raise FlexToolConfigError(
            f"broadcast_to_period: shape {shape.value} is not supported "
            f"for (entity, period) parameters.  Allowed: "
            f"{_allowed_shape_names(_PERIOD_ONLY_SHAPES)}")
    if out.height == 0:
        return None
    return Param((entity_dim_alias, "d"), out.lazy())
