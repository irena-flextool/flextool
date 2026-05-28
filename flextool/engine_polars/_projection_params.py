"""Γ.2 Projection Param helpers.

Each function in this module takes an :class:`InputSource` and returns
the corresponding polar_high frame (or ``None`` if the projection is empty
on the supplied scenario).  A Projection is defined (per the
``audit/db_direct_param_map.md`` spec §1) as a set-algebra layer over
``source.entities(...)`` and ``source.parameter(...)`` results — a
filter, a distinct projection, or a structural join.  No per-row
algorithms, no per-period state cascades.

Lazy-evaluation pattern
-----------------------
``InputSource.parameter()`` and ``.entities()`` already collect at the
source boundary.  Helpers compose those results via :class:`pl.LazyFrame`
chains and ``.collect()`` once at the boundary — so polars can fuse
projections / filter pushdowns across joins.

The helpers are intentionally tiny (5–25 LOC each).  Common patterns:

1. **Filter on a method discriminator.**  Pull the entity-level method
   parameter, filter to a specific value, project the entity column.
   Example: ``process_minload`` is the units with non-zero ``min_load``.

2. **Distinct projection.**  Pull a Map / 1d_map parameter, take
   ``.unique()`` over the entity dimension(s).  Example:
   ``commodity__tier_ann`` is the distinct ``(commodity, tier)`` pairs
   where the commodity carries an annual ladder.

3. **Structural join.**  Inner-join two membership sets.  Example:
   ``flow_from_commodity_eff`` is ``pss_eff ⋈ commodity__node`` on
   ``source = node``.

4. **Union.**  Concatenate two membership sets with column alignment.
   Example: ``group_entity`` unions ``group__node``, ``group__unit``,
   ``group__connection`` into one ``(group, entity)`` frame.

Block-aware filters (``flow_to_n``, ``flow_from_n``, etc.) are NOT
implemented here — they depend on multi-resolution preprocessed
auxiliary tables (``process_side_block``, ``entity_block``,
``overlap_set``) that are themselves Derived (Γ.3) artefacts.

See ``audit/db_direct_param_map.md`` §1, §5, §7.2 for the contract.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from flextool.engine_polars._axis_enums import (
    alias_to_axis,
    cast_dim,
    cast_frame_axes,
    get_global_axis_enums,
    rename_to_axis,
    schema_dtype,
)


def _to_e(expr: pl.Expr) -> pl.Expr:
    """Cast *expr* to the entity-union (``e``) axis enum if active.

    Used at ``pl.col("p").alias("source"/"sink")`` sites where a
    process-typed column is reused as an entity-union slot.  Without
    the cast, ``vertical_relaxed`` concat downstream raises
    ``SchemaError: failed to determine supertype of enum and enum``.
    """
    return cast_dim(expr, get_global_axis_enums(), "e")

if TYPE_CHECKING:
    from flextool.engine_polars._input_source import InputSource


# ---------------------------------------------------------------------------
# Helpers for the common shape of "rename source to polar_high column names"
# ---------------------------------------------------------------------------

def _empty(schema: dict[str, type[pl.DataType]]) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def _try_param(source: "InputSource", entity_class: str,
                parameter_name: str) -> pl.DataFrame | None:
    """Return ``source.parameter(...)`` or ``None`` if unknown / empty."""
    try:
        df = source.parameter(entity_class, parameter_name)
    except KeyError:
        return None
    if df.height == 0:
        return None
    return df


def _try_entities(source: "InputSource", entity_class: str) -> pl.DataFrame | None:
    try:
        df = source.entities(entity_class)
    except KeyError:
        return None
    if df.height == 0:
        return None
    return df


# ---------------------------------------------------------------------------
# §1.2 — nodeBalance (P): node_type ∈ {balance}
# ---------------------------------------------------------------------------

def nodeBalance(source: "InputSource") -> pl.DataFrame:
    """Nodes with ``node_type ∈ {'balance', 'storage'}``.

    Schema: ``[n]``.  flextool's `node_type_sets.py:69` defines
    `nodeBalance` as the nodes whose type is **either** balance or
    storage — both participate in the LHS of `nodeBalance_eq`.
    """
    df = _try_param(source, "node", "node_type")
    if df is None:
        return _empty({"n": pl.Utf8})
    return (df.lazy()
              .filter(pl.col("value").is_in(["balance", "storage"]))
              .pipe(rename_to_axis, {"name": "n"})
              .select("n")
              .sort("n")
              .collect())


# ---------------------------------------------------------------------------
# §1.3 — Process topology Projections
# ---------------------------------------------------------------------------

def process_source_sink(source: "InputSource") -> pl.DataFrame:
    """Union of ``unit__inputNode``, ``unit__outputNode``, and
    ``connection__node__node`` into the canonical ``(p, source, sink)``
    arc table.

    For unit input arcs: ``source = node``, ``sink = unit``.
    For unit output arcs: ``source = unit``, ``sink = node``.
    For connection arcs: a connection's ``node_1 → node_2``, and the
    reverse ``node_2 → node_1`` if the connection is bidirectional —
    here we materialise both directions as the .mod's
    ``process_source_sink`` includes them when the connection is
    bidirectional.  flextool's preprocessor decides directionality
    via ``connection.is_DC`` / transfer-method semantics; for the
    Projection-only port we mirror flextool's CSV: process_source_sink
    is the **union** of the input/output unit arcs and **all**
    (connection, node_1, node_2) and (connection, node_2, node_1)
    pairs.

    Schema: ``[p, source, sink]``.

    This produces the **expanded** form (one arc per input + one arc per
    output) where indirect / no-collapse units appear as ``(p, source, p)``
    and ``(p, p, sink)``.  Use :func:`process_source_sink_collapsed` to
    obtain flextool's preprocessing-side **collapsed** shape (direct-
    method units flattened to ``(p, source, sink)``) — that's the form
    written to ``solve_data/process_source_sink.csv``.
    """
    parts: list[pl.LazyFrame] = []

    uin = _try_entities(source, "unit__inputNode")
    if uin is not None:
        parts.append(uin.lazy().select(
            alias_to_axis("unit", "p"),
            alias_to_axis("node", "source"),
            alias_to_axis("unit", "sink"),
        ))
    uout = _try_entities(source, "unit__outputNode")
    if uout is not None:
        parts.append(uout.lazy().select(
            alias_to_axis("unit", "p"),
            alias_to_axis("unit", "source"),
            alias_to_axis("node", "sink"),
        ))
    cnn = _try_entities(source, "connection__node__node")
    if cnn is not None:
        parts.append(cnn.lazy().select(
            alias_to_axis("connection", "p"),
            alias_to_axis("node_1", "source"),
            alias_to_axis("node_2", "sink"),
        ))

    if not parts:
        return _empty({"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    return pl.concat(parts).unique().sort("p", "source", "sink").collect()


# ---------------------------------------------------------------------------
# Δ.17b Gap B — flextool preprocessing-style collapsed pss
# ---------------------------------------------------------------------------
#
# flextool's preprocessing produces ``solve_data/process_source_sink.csv``
# as the union of 10 sub-sets (see preprocessing/process_arc_unions.py
# and process_method_sets.py).  The shape per process depends on the
# internal method:
#
#   * DIRECT methods (method_1way_1var_*, method_2way_1var_*,
#     method_2way_2var_*): cross-product of (p, source) × (p, sink) →
#     ``(p, source, sink)``.  2-way variants also emit reverse arcs
#     ``(p, sink, source)``.
#   * INDIRECT methods (method_*_nvar_*): ``(p, source, p)`` and
#     ``(p, p, sink)`` — the unit name appears as the intermediate
#     "node" in two arcs.  2way_nvar also emits ``(p, p, source)``.
#   * 1way_1var with no source: ``(p, p, sink)`` (process_process_toSink_noConversion).
#   * 1way_1var with no sink: ``(p, source, p)`` (process_source_toProcess_noConversion).
#   * Connections: always ``(c, node_1, node_2)`` (already 1-arc form).
#
# Internal method comes from the classifier (input_writer.METHODS_MAPPING).
# We reuse the classifier from ``_derived_params.py``.


_METHOD_2WAY_NVAR_LOCAL = frozenset(("method_2way_nvar_off",))
_METHOD_2WAY_2VAR_LOCAL = frozenset((
    "method_2way_2var_off", "method_2way_2var_exclude",
    "method_2way_2var_MIP_exclude",
))
_METHOD_1WAY_1VAR_LOCAL = frozenset((
    "method_1way_1var_off", "method_1way_1var_LP", "method_1way_1var_MIP",
))
_METHOD_DIRECT_LOCAL = frozenset((
    "method_1way_1var_off", "method_1way_1var_LP", "method_1way_1var_MIP",
    "method_2way_1var_off", "method_2way_2var_off",
    "method_2way_2var_exclude", "method_2way_2var_MIP_exclude",
))
_METHOD_INDIRECT_LOCAL = frozenset((
    "method_1way_nvar_off", "method_1way_nvar_LP", "method_1way_nvar_MIP",
    "method_2way_nvar_off",
))


def process_source_sink_collapsed(source: "InputSource",
                                     classified: pl.DataFrame | None = None
                                     ) -> pl.DataFrame:
    """flextool-preprocessing-shaped ``process_source_sink`` — same as
    :func:`process_source_sink_canonical` minus the ``method`` column.
    Schema: ``[p, source, sink]``.

    Δ.17b Gap B closure.  Prior to this, ``input.py:_load_process_topology``
    had to read the three preprocessed ``process_source_sink*.csv`` files
    from disk because the canonical helper produced the *expanded*
    ``process_source_sink`` shape (``unit__inputNode`` ∪ ``unit__outputNode``
    — 2 arcs per direct+indirect unit) — not flextool's collapsed shape.
    This helper produces the same 10-way union flextool's
    ``write_process_arc_unions`` builds.
    """
    del classified  # canonical computes its own
    canonical = process_source_sink_canonical(source)
    if canonical.height == 0:
        return _empty({"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    return (canonical.lazy()
              .select("p", "source", "sink")
              .unique()
              .sort("p", "source", "sink")
              .collect())


# Conversion methods that are "efficiency-based" (eff partition).  The
# .mod calls these the ``method_X_LP`` family; flextool's preprocessor
# also includes ``min_load_efficiency`` and ``constant_efficiency`` for
# the Projection.  See preprocessing/process_method_sets.py.
_EFF_CONVERSION_METHODS = ("min_load_efficiency", "constant_efficiency",
                            "method_X_LP")


def process_source_sink_canonical(source: "InputSource",
                                    pss: pl.DataFrame | None = None) -> pl.DataFrame:
    """Method-tagged canonical frame for the ``process_source_sink``
    family.  Built once and projected to ``_eff`` / ``_noEff`` by the
    consumers below — replaces the prior pattern of computing each
    partition from scratch.

    Schema: ``[p, source, sink, method]`` where ``method ∈ {'eff', 'noEff'}``.

    Δ.17b Gap B: produces flextool's preprocessing-side **collapsed
    shape** (DIRECT methods flattened to ``(p, source, sink)``; INDIRECT
    methods kept as 2-arc form ``(p, source, p)``+``(p, p, sink)``).
    Partition keyed by internal METHOD (DIRECT → eff, else → noEff),
    not by ``conversion_method``.  This matches flextool's
    ``write_process_arc_unions``:

      * ``_eff``  = ``process_source_toSink ∪ process_sink_toSource``
                  (DIRECT-method arcs).
      * ``_noEff`` = 8-way union of indirect / noConversion / profile
                    sub-sets.

    The legacy ``pss`` parameter is accepted for backward compatibility
    but ignored when the classifier is available — the classifier dictates
    both the shape (collapsed vs expanded) and the eff/noEff partition.
    """
    del pss  # legacy parameter

    # Late import to avoid a circular reference.
    from flextool.engine_polars._derived_params import _classify_process_method
    classified = _classify_process_method(source)

    # Connections are always noEff and 1-arc.
    cnn = _try_entities(source, "connection__node__node")
    parts: list[pl.LazyFrame] = []
    if cnn is not None and cnn.height > 0:
        # Connections classify into the eff partition only when they
        # carry a transfer_method that the .mod treats as efficiency-
        # bearing (i.e. DIRECT-family internal methods).  Use the
        # classifier output rather than re-deriving here.
        cls_conns = (classified.lazy()
                       .filter(pl.col("klass") == "connection")
                       .select("p", "method"))
        cnn_lf = cnn.lazy().pipe(
            rename_to_axis,
            {"connection": "p", "node_1": "source", "node_2": "sink"},
        ).select("p", "source", "sink")
        # Join with classifier to get internal method per arc.
        cnn_with_method = cnn_lf.join(cls_conns, on="p", how="left")
        # Forward direction: tag eff/noEff by method.
        cnn_eff = (cnn_with_method
            .filter(pl.col("method").is_in(list(_METHOD_DIRECT_LOCAL)))
            .select("p", "source", "sink",
                    pl.lit("eff").alias("method")))
        cnn_noEff = (cnn_with_method
            .filter(~pl.col("method").is_in(list(_METHOD_DIRECT_LOCAL))
                    | pl.col("method").is_null())
            .select("p", "source", "sink",
                    pl.lit("noEff").alias("method")))
        parts.append(cnn_eff)
        parts.append(cnn_noEff)
        # Reverse direction for 2way_2var connections.
        rev_eff = (cnn_with_method
            .filter(pl.col("method").is_in(list(_METHOD_2WAY_2VAR_LOCAL)))
            .select("p",
                    alias_to_axis("sink", "source"),
                    alias_to_axis("source", "sink"),
                    pl.lit("eff").alias("method")))
        parts.append(rev_eff)
        # 2way_nvar connections — emits indirect 2-arc form.  In practice
        # connections almost never have 2way_nvar; guard via classifier.
        rev_noEff = (cnn_with_method
            .filter(pl.col("method").is_in(list(_METHOD_2WAY_NVAR_LOCAL)))
            .select("p",
                    alias_to_axis("sink", "source"),
                    alias_to_axis("source", "sink"),
                    pl.lit("noEff").alias("method")))
        parts.append(rev_noEff)

    # Per-unit dispatch keyed by classified method.  Build empty-but-typed
    # source/sink LazyFrames when the entity-class is missing, so the
    # noConversion branches still see "no source rows" / "no sink rows"
    # rather than skipping entirely.  Phase 4: dim columns must carry
    # their canonical Enum dtype so joins against Enum-keyed frames
    # don't SchemaError on the empty branch.
    _enums_empty = get_global_axis_enums()

    def _empty_lf(*cols):
        return pl.DataFrame(
            {c: [] for c in cols},
            schema={c: schema_dtype(_enums_empty, c) for c in cols},
        ).lazy()
    if classified is not None and classified.height > 0:
        cls_units = (classified.lazy()
                       .filter(pl.col("klass") == "unit")
                       .select("p", "method"))
        uin = _try_entities(source, "unit__inputNode")
        sources_lf = (
            uin.lazy().pipe(rename_to_axis,
                              {"unit": "p", "node": "source"})
                      .select("p", "source")
        ) if uin is not None else _empty_lf("p", "source")
        uout = _try_entities(source, "unit__outputNode")
        sinks_lf = (
            uout.lazy().pipe(rename_to_axis,
                              {"unit": "p", "node": "sink"})
                       .select("p", "sink")
        ) if uout is not None else _empty_lf("p", "sink")

        # ── DIRECT methods: cross-join → ``_eff`` partition ──
        direct_p = (cls_units
            .filter(pl.col("method").is_in(list(_METHOD_DIRECT_LOCAL)))
            .select("p"))
        direct_arcs = (sources_lf
            .join(direct_p, on="p", how="inner")
            .join(sinks_lf, on="p", how="inner")
            .select("p", "source", "sink",
                    pl.lit("eff").alias("method")))
        parts.append(direct_arcs)

        # 2way_2var: reverse arcs → also ``_eff``.
        two_way_p = (cls_units
            .filter(pl.col("method").is_in(list(_METHOD_2WAY_2VAR_LOCAL)))
            .select("p"))
        rev_arcs = (sources_lf
            .join(two_way_p, on="p", how="inner")
            .join(sinks_lf, on="p", how="inner")
            .select(pl.col("p"),
                    alias_to_axis("sink", "source"),
                    alias_to_axis("source", "sink"),
                    pl.lit("eff").alias("method")))
        parts.append(rev_arcs)

        # ── INDIRECT methods: 2-arc form → ``_noEff`` partition ──
        indirect_p = (cls_units
            .filter(pl.col("method").is_in(list(_METHOD_INDIRECT_LOCAL)))
            .select("p"))
        indirect_inputs = (sources_lf
            .join(indirect_p, on="p", how="inner")
            .select("p", "source", alias_to_axis(pl.col("p"), "sink"),
                    pl.lit("noEff").alias("method")))
        parts.append(indirect_inputs)

        # 2way_nvar: process_process_toSource — (p, p, source).
        two_way_nvar_p = (cls_units
            .filter(pl.col("method").is_in(list(_METHOD_2WAY_NVAR_LOCAL)))
            .select("p"))
        indirect_inputs_rev = (sources_lf
            .join(two_way_nvar_p, on="p", how="inner")
            .select("p", alias_to_axis(pl.col("p"), "source"),
                    alias_to_axis("source", "sink"),
                    pl.lit("noEff").alias("method")))
        parts.append(indirect_inputs_rev)

        indirect_outputs = (sinks_lf
            .join(indirect_p, on="p", how="inner")
            .select("p", alias_to_axis(pl.col("p"), "source"), "sink",
                    pl.lit("noEff").alias("method")))
        parts.append(indirect_outputs)

        # ── 1way_1var DIRECT, missing one side: noConversion ``_noEff`` ──
        one_way_1var_p = (cls_units
            .filter(pl.col("method").is_in(list(_METHOD_1WAY_1VAR_LOCAL)))
            .select("p"))
        sources_exist = sources_lf.select("p").unique()
        sinks_exist = sinks_lf.select("p").unique()
        no_sink_p = one_way_1var_p.join(sinks_exist, on="p", how="anti")
        no_source_p = one_way_1var_p.join(sources_exist, on="p", how="anti")
        no_sink_arcs = (sources_lf
            .join(no_sink_p, on="p", how="inner")
            .select("p", "source", alias_to_axis(pl.col("p"), "sink"),
                    pl.lit("noEff").alias("method")))
        parts.append(no_sink_arcs)
        no_source_arcs = (sinks_lf
            .join(no_source_p, on="p", how="inner")
            .select("p", alias_to_axis(pl.col("p"), "source"), "sink",
                    pl.lit("noEff").alias("method")))
        parts.append(no_source_arcs)

    # Phase 4.6: when activation is on, the empty fallback must use the
    # canonical Enum dtypes so downstream joins against Enum-typed
    # frames don't SchemaError on the empty branch.
    _enums = get_global_axis_enums()
    _empty_schema = {
        "p": schema_dtype(_enums, "p"),
        "source": schema_dtype(_enums, "source"),
        "sink": schema_dtype(_enums, "sink"),
        "method": pl.Utf8,
    }
    if not parts:
        return _empty(_empty_schema)

    # Final dedup: arcs may appear in multiple partitions due to method
    # categories; flextool's union takes a set-union (dedup).  Tie-break:
    # if same (p, source, sink) appears as both eff and noEff (shouldn't
    # in practice but defensive), prefer eff.
    out = (pl.concat(parts, how="vertical_relaxed")
              .unique(subset=["p", "source", "sink", "method"])
              .sort("p", "source", "sink", "method")
              .collect())
    if out.height == 0:
        return _empty(_empty_schema)
    # Cast dim columns to canonical axis enums.  Many partitions emit
    # ``source``/``sink`` via ``pl.col("p").alias(...)`` (cross-axis from
    # the ``p`` enum into the ``source``/``sink`` slot whose contract
    # axis is ``e``).  The concat above is ``vertical_relaxed`` so
    # types fall to a common supertype (string when mixing enum
    # vocabularies); the cast here re-establishes canonical Enum
    # dtypes once at the boundary.
    enums = get_global_axis_enums()
    if enums is not None:
        out = cast_frame_axes(out, enums)
    return out


def process_source_sink_eff(source: "InputSource",
                              pss: pl.DataFrame | None = None) -> pl.DataFrame:
    """``pss`` filtered to processes with an efficiency-based
    ``conversion_method`` (units only; connections never participate
    in eff).

    Schema: ``[p, source, sink]``.

    Δ.3: thin filter over :func:`process_source_sink_canonical`.
    """
    canonical = process_source_sink_canonical(source, pss)
    if canonical.height == 0:
        return _empty({"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    return (canonical.lazy()
              .filter(pl.col("method") == "eff")
              .select("p", "source", "sink")
              .sort("p", "source", "sink")
              .collect())


def process_source_sink_noEff(source: "InputSource",
                                pss: pl.DataFrame | None = None,
                                pss_eff: pl.DataFrame | None = None) -> pl.DataFrame:
    """Complement of ``process_source_sink_eff`` within ``pss`` — the
    arcs whose process is NOT efficiency-based (connections + units
    with ``conversion_method ∉ eff family``).

    Schema: ``[p, source, sink]``.

    Δ.3: thin filter over :func:`process_source_sink_canonical`.  The
    legacy ``pss_eff`` parameter is accepted for backward compatibility
    but no longer used — the canonical frame already encodes the
    partition via the ``method`` column.
    """
    del pss_eff  # legacy parameter; see docstring
    canonical = process_source_sink_canonical(source, pss)
    if canonical.height == 0:
        return _empty({"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    return (canonical.lazy()
              .filter(pl.col("method") == "noEff")
              .select("p", "source", "sink")
              .sort("p", "source", "sink")
              .collect())


# ---------------------------------------------------------------------------
# process_source_sink_ramp_limit_* / _ramp_cost helpers
# ---------------------------------------------------------------------------
#
# Five sets:
#
#   * ``process_source_sink_ramp_limit_source_up`` — (p, src, sink) where
#       (p, src, m) ∈ process__node__ramp_method, m ∈ RAMP_LIMIT_METHOD
#       AND p_process_source[p, src, 'ramp_speed_up'] > 0.
#   * Symmetric for sink/up, source/down, sink/down.
#   * ``process_source_sink_ramp_cost`` — OR of source/sink-side membership
#       in RAMP_COST_METHOD (no speed-gate).
#
# In Spine, the per-arc ramp_method + ramp_speed_*  parameters live on
# ``unit__inputNode`` (source side) and ``unit__outputNode`` (sink side).
# Connections never carry ramp methods — flextool's preprocessing reads
# only ``input/process__node__ramp_method.csv`` which is the union of
# ``unit__inputNode.ramp_method`` ∪ ``unit__outputNode.ramp_method``.
#
# User-supplied MathProg snippets (Δ.17c dispatch, Gap D):
#
#     set process_source_sink_ramp_limit_source_up :=
#         {(p, source, sink) in process_source_sink :
#             ( sum{(p, source, m) in process_node_ramp_method
#                   : m in ramp_limit_method} 1
#               && p_process_source[p, source, 'ramp_speed_up'] > 0
#             )};
#
#     set process_source_sink_ramp_cost :=
#         {(p, source, sink) in process_source_sink :
#             sum{(p, source, m) in process_node_ramp_method
#                  : m in ramp_cost_method} 1
#             || sum{(p, sink, m) in process_node_ramp_method
#                     : m in ramp_cost_method} 1};


# Mirror of :mod:`flextool.input_derivation._method_constants`.
# Kept locally to avoid an import dependency.
_RAMP_LIMIT_METHOD: frozenset[str] = frozenset(("ramp_limit", "both"))
_RAMP_COST_METHOD: frozenset[str] = frozenset(("ramp_cost", "both"))


def _ramp_pairs(source: "InputSource",
                relationship_class: str,
                method_set: "frozenset[str]") -> "pl.LazyFrame":
    """Return ``[unit, node]`` pairs whose ``ramp_method`` value is in
    *method_set*, drawn from *relationship_class* ∈
    ``{unit__inputNode, unit__outputNode}``.

    Empty / missing parameter → empty LazyFrame with schema ``[unit, node]``.
    """
    df = _try_param(source, relationship_class, "ramp_method")
    if df is None:
        return pl.DataFrame(
            schema={"unit": pl.Utf8, "node": pl.Utf8}).lazy()
    return (df.lazy()
              .filter(pl.col("value").is_in(list(method_set)))
              .select("unit", "node"))


def _ramp_speed_positive_pairs(source: "InputSource",
                                  relationship_class: str,
                                  parameter_name: str) -> "pl.LazyFrame":
    """Return ``[unit, node]`` pairs whose ``parameter_name`` value
    (``ramp_speed_up`` / ``ramp_speed_down``) is strictly > 0.
    """
    df = _try_param(source, relationship_class, parameter_name)
    if df is None:
        return pl.DataFrame(
            schema={"unit": pl.Utf8, "node": pl.Utf8}).lazy()
    return (df.lazy()
              .filter(pl.col("value").is_not_null() & (pl.col("value") > 0.0))
              .select("unit", "node"))


def _ramp_limit_set(source: "InputSource",
                     side: str,
                     direction: str) -> pl.DataFrame:
    """Build ``process_source_sink_ramp_limit_<side>_<direction>``.

    *side* — ``"source"`` or ``"sink"`` (the column the relationship-side
    constraint applies to).
    *direction* — ``"up"`` or ``"down"`` (which speed parameter to gate).
    """
    if side == "source":
        rel_class = "unit__inputNode"
        ramp_pairs = _ramp_pairs(source, rel_class, _RAMP_LIMIT_METHOD) \
                        .pipe(rename_to_axis, {"unit": "p", "node": "source"})
        speed_pairs = _ramp_speed_positive_pairs(
            source, rel_class, f"ramp_speed_{direction}") \
                .pipe(rename_to_axis, {"unit": "p", "node": "source"})
        gated = ramp_pairs.join(speed_pairs, on=["p", "source"],
                                  how="inner")
        canonical = process_source_sink_canonical(source).lazy()
        return (canonical
                  .select("p", "source", "sink")
                  .unique()
                  .join(gated, on=["p", "source"], how="inner")
                  .select("p", "source", "sink")
                  .unique()
                  .sort("p", "source", "sink")
                  .collect())
    if side == "sink":
        rel_class = "unit__outputNode"
        ramp_pairs = _ramp_pairs(source, rel_class, _RAMP_LIMIT_METHOD) \
                        .pipe(rename_to_axis, {"unit": "p", "node": "sink"})
        speed_pairs = _ramp_speed_positive_pairs(
            source, rel_class, f"ramp_speed_{direction}") \
                .pipe(rename_to_axis, {"unit": "p", "node": "sink"})
        gated = ramp_pairs.join(speed_pairs, on=["p", "sink"],
                                  how="inner")
        canonical = process_source_sink_canonical(source).lazy()
        return (canonical
                  .select("p", "source", "sink")
                  .unique()
                  .join(gated, on=["p", "sink"], how="inner")
                  .select("p", "source", "sink")
                  .unique()
                  .sort("p", "source", "sink")
                  .collect())
    raise ValueError(f"_ramp_limit_set: unknown side {side!r}")


def process_source_sink_ramp_limit_source_up(
    source: "InputSource",
) -> pl.DataFrame:
    """Δ.17c Gap D — see module docstring above for the MathProg shape."""
    return _ramp_limit_set(source, "source", "up")


def process_source_sink_ramp_limit_source_down(
    source: "InputSource",
) -> pl.DataFrame:
    return _ramp_limit_set(source, "source", "down")


def process_source_sink_ramp_limit_sink_up(
    source: "InputSource",
) -> pl.DataFrame:
    return _ramp_limit_set(source, "sink", "up")


def process_source_sink_ramp_limit_sink_down(
    source: "InputSource",
) -> pl.DataFrame:
    return _ramp_limit_set(source, "sink", "down")


def process_source_sink_ramp_cost(source: "InputSource") -> pl.DataFrame:
    """``{(p, src, sink) ∈ pss : ramp_cost on source-side OR sink-side}``.

    Mirrors mod L1115-1119 + the user's Δ.17c MathProg snippet (no
    speed-gate; pure method membership).
    """
    src_cost = (_ramp_pairs(source, "unit__inputNode", _RAMP_COST_METHOD)
                  .pipe(rename_to_axis, {"unit": "p", "node": "source"}))
    sink_cost = (_ramp_pairs(source, "unit__outputNode", _RAMP_COST_METHOD)
                   .pipe(rename_to_axis, {"unit": "p", "node": "sink"}))
    canonical = process_source_sink_canonical(source).lazy().select(
        "p", "source", "sink").unique()
    via_source = canonical.join(src_cost, on=["p", "source"], how="inner")
    via_sink = canonical.join(sink_cost, on=["p", "sink"], how="inner")
    return (pl.concat([via_source, via_sink], how="vertical_relaxed")
              .select("p", "source", "sink")
              .unique()
              .sort("p", "source", "sink")
              .collect())


def flow_from_commodity_eff(source: "InputSource",
                             pss_eff: pl.DataFrame | None = None) -> pl.DataFrame:
    """``pss_eff ⋈ commodity__node`` on ``source = node`` →
    ``(p, source, sink, c)``.  Empty if either side is empty.
    """
    if pss_eff is None:
        pss_eff = process_source_sink_eff(source)
    if pss_eff.height == 0:
        return _empty({"p": pl.Utf8, "source": pl.Utf8,
                       "sink": pl.Utf8, "c": pl.Utf8})
    cn = _try_entities(source, "commodity__node")
    if cn is None:
        return _empty({"p": pl.Utf8, "source": pl.Utf8,
                       "sink": pl.Utf8, "c": pl.Utf8})
    return (pss_eff.lazy()
              .join(cn.lazy(), left_on="source", right_on="node", how="inner")
              .pipe(rename_to_axis, {"commodity": "c"})
              .select("p", "source", "sink", "c")
              .sort("p", "source", "sink", "c")
              .collect())


def flow_from_commodity_noEff(source: "InputSource",
                                pss_noEff: pl.DataFrame | None = None) -> pl.DataFrame:
    """``pss_noEff ⋈ commodity__node`` on ``source = node``."""
    if pss_noEff is None:
        pss_noEff = process_source_sink_noEff(source)
    if pss_noEff.height == 0:
        return _empty({"p": pl.Utf8, "source": pl.Utf8,
                       "sink": pl.Utf8, "c": pl.Utf8})
    cn = _try_entities(source, "commodity__node")
    if cn is None:
        return _empty({"p": pl.Utf8, "source": pl.Utf8,
                       "sink": pl.Utf8, "c": pl.Utf8})
    return (pss_noEff.lazy()
              .join(cn.lazy(), left_on="source", right_on="node", how="inner")
              .pipe(rename_to_axis, {"commodity": "c"})
              .select("p", "source", "sink", "c")
              .sort("p", "source", "sink", "c")
              .collect())


def flow_to_commodity(source: "InputSource",
                      pss: pl.DataFrame | None = None) -> pl.DataFrame:
    """``pss ⋈ commodity__node`` on ``sink = node`` →
    ``(p, source, sink, c)``.  Sink-side flow into a priced
    commodity node.
    """
    if pss is None:
        pss = process_source_sink(source)
    if pss.height == 0:
        return _empty({"p": pl.Utf8, "source": pl.Utf8,
                       "sink": pl.Utf8, "c": pl.Utf8})
    cn = _try_entities(source, "commodity__node")
    if cn is None:
        return _empty({"p": pl.Utf8, "source": pl.Utf8,
                       "sink": pl.Utf8, "c": pl.Utf8})
    return (pss.lazy()
              .join(cn.lazy(), left_on="sink", right_on="node", how="inner")
              .pipe(rename_to_axis, {"commodity": "c"})
              .select("p", "source", "sink", "c")
              .sort("p", "source", "sink", "c")
              .collect())


# ---------------------------------------------------------------------------
# §1.4 — CO2 group set Projections
# ---------------------------------------------------------------------------

def group_co2_max_period(source: "InputSource") -> pl.DataFrame:
    """Distinct groups with a non-empty ``co2_max_period`` Map.

    Schema: ``[g]``.
    """
    df = _try_param(source, "group", "co2_max_period")
    if df is None:
        return _empty({"g": pl.Utf8})
    return (df.lazy()
              .select(alias_to_axis("name", "g"))
              .unique()
              .sort("g")
              .collect())


# ---------------------------------------------------------------------------
# §1.5 — Indirect-conversion (CHP) Projections
# ---------------------------------------------------------------------------

_INDIRECT_CONVERSION_METHODS = ("method_indirect", "method_indirect_LP")


def process_indirect(source: "InputSource") -> pl.DataFrame:
    """Units with ``conversion_method`` in the indirect family.

    Schema: ``[p]``.
    """
    df = _try_param(source, "unit", "conversion_method")
    if df is None:
        return _empty({"p": pl.Utf8})
    return (df.lazy()
              .filter(pl.col("value").is_in(_INDIRECT_CONVERSION_METHODS))
              .select(alias_to_axis("name", "p"))
              .sort("p")
              .collect())


def process_input_flows(source: "InputSource",
                          pss: pl.DataFrame | None = None,
                          process_indirect_set: pl.DataFrame | None = None) -> pl.DataFrame:
    """Indirect-process input arcs: ``pss`` filtered to ``sink == p``
    AND ``p ∈ process_indirect``.  In the indirect family the unit
    name appears as both ``p`` and ``sink`` (one row per (p, source)).

    Schema: ``[p, source, sink]``.
    """
    if pss is None:
        pss = process_source_sink(source)
    if process_indirect_set is None:
        process_indirect_set = process_indirect(source)
    if pss.height == 0 or process_indirect_set.height == 0:
        return _empty({"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    return (pss.lazy()
              # Cross-axis compare: sink (e) vs p (p).  Per contract
              # p ⊂ e; up-cast p to e so the compare runs in Enum.
              .filter(pl.col("sink") == cast_dim(pl.col("p"), None, "e"))
              .join(process_indirect_set.lazy(), on="p", how="inner")
              .sort("p", "source", "sink")
              .collect())


def process_output_flows(source: "InputSource",
                          pss: pl.DataFrame | None = None,
                          process_indirect_set: pl.DataFrame | None = None) -> pl.DataFrame:
    """Indirect-process output arcs: ``pss`` filtered to ``source == p``
    AND ``p ∈ process_indirect``.

    Schema: ``[p, source, sink]``.
    """
    if pss is None:
        pss = process_source_sink(source)
    if process_indirect_set is None:
        process_indirect_set = process_indirect(source)
    if pss.height == 0 or process_indirect_set.height == 0:
        return _empty({"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    return (pss.lazy()
              # Cross-axis compare: source (e) vs p (p).  Per contract
              # p ⊂ e; up-cast p to e so the compare runs in Enum.
              .filter(pl.col("source") == cast_dim(pl.col("p"), None, "e"))
              .join(process_indirect_set.lazy(), on="p", how="inner")
              .sort("p", "source", "sink")
              .collect())


def process_indirect_dt(process_indirect_set: pl.DataFrame,
                          dt: pl.DataFrame) -> pl.DataFrame:
    """``process_indirect × dt`` cross-product.  Schema: ``[p, d, t]``.
    """
    if process_indirect_set is None or process_indirect_set.height == 0:
        return _empty({"p": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8})
    if dt is None or dt.height == 0:
        return _empty({"p": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8})
    # Defensive re-cast: ensure d/t are canonical Enum on the cross-join
    # output (process_indirect_set carries Enum p).
    return (process_indirect_set.lazy()
              .join(dt.lazy()
                       .with_columns(alias_to_axis(pl.col("d"), "d"),
                                     alias_to_axis(pl.col("t"), "t")),
                    how="cross")
              .sort("p", "d", "t")
              .collect())


# ---------------------------------------------------------------------------
# §1.6 — Constraint sense partitions (cdt_eq / cdt_le / cdt_ge)
# ---------------------------------------------------------------------------

def cdt_filter(source: "InputSource", sense: str,
                dt: pl.DataFrame) -> pl.DataFrame | None:
    """Cross-product of constraints with the named ``sense`` and
    ``dt``.  Returns ``None`` when the partition is empty (so the caller
    can skip emitting the constraint group).

    Senses: ``equal`` / ``less_than`` / ``greater_than``.
    """
    s = _try_param(source, "constraint", "sense")
    if s is None or dt is None or dt.height == 0:
        return None
    rows = (s.lazy()
              .filter(pl.col("value") == sense)
              .select(alias_to_axis("name", "cn"))
              .collect())
    if rows.height == 0:
        return None
    # Defensive re-cast: ensure d/t are canonical Enum on the cross-join
    # output (rows carries Enum cn via alias_to_axis above).
    return (rows.lazy()
              .join(dt.lazy()
                       .with_columns(alias_to_axis(pl.col("d"), "d"),
                                     alias_to_axis(pl.col("t"), "t")),
                    how="cross")
              .sort("cn", "d", "t")
              .collect())


# ---------------------------------------------------------------------------
# §1.7 — Profile filter Projections
# ---------------------------------------------------------------------------

def _profile_filter(source: "InputSource",
                     profile_class: str,
                     dim_renames: dict[str, str],
                     out_cols: tuple[str, ...],
                     method: str) -> pl.DataFrame:
    """Filter a profile-method relationship by ``method`` value.

    ``dim_renames`` maps Spine column names → polar_high column names;
    ``out_cols`` is the projection.
    """
    df = _try_param(source, profile_class, "profile_method")
    if df is None:
        empty_schema = {c: pl.Utf8 for c in out_cols}
        return _empty(empty_schema)
    return (df.lazy()
              .filter(pl.col("value") == method)
              .pipe(rename_to_axis, dim_renames)
              .select(*out_cols)
              .sort(*out_cols)
              .collect())


def process_profile_upper(source: "InputSource") -> pl.DataFrame:
    """Unit / connection profile arcs with ``profile_method=upper_limit``.

    Schema: ``[p, source, sink, profile]``.  Each row is keyed by
    ``(p, source, sink, profile)``.  Built as the union of the
    ``unit__node__profile`` (for unit-side arcs) and
    ``connection__node__profile`` (for connection-side arcs)
    relationships filtered by method.

    For ``unit__node__profile`` the node is either the source (input
    arc) or the sink (output arc); the .mod resolves this from
    ``unit__inputNode`` / ``unit__outputNode``.  We replicate that by
    cross-joining with ``pss_eff`` / ``pss`` on ``(p, node)``.
    """
    return _profile_method_arc(source, "upper_limit")


def process_profile_lower(source: "InputSource") -> pl.DataFrame:
    return _profile_method_arc(source, "lower_limit")


def process_profile_fixed(source: "InputSource") -> pl.DataFrame:
    return _profile_method_arc(source, "fixed")


def _profile_method_arc(source: "InputSource", method: str) -> pl.DataFrame:
    """Internal: join unit/connection profile-method rows back to
    process_source_sink to reconstruct the (p, source, sink, f) arc
    tuples.  Column ``f`` here is flextool's name for the profile.

    For each ``(unit, node, profile)`` row with the requested method,
    emit two rows when the (unit, node) is a unit__inputNode (sink=unit,
    source=node) and one row per unit__outputNode (sink=node,
    source=unit).  Connection profile-method rows are mapped via
    connection__node__node similarly.
    """
    out_schema = {"p": pl.Utf8, "source": pl.Utf8,
                   "sink": pl.Utf8, "f": pl.Utf8}

    parts: list[pl.LazyFrame] = []

    # Unit profiles: unit__node__profile filtered by method, then map
    # node to the (unit, node) arc (input or output side).
    unp = _try_param(source, "unit__node__profile", "profile_method")
    if unp is not None:
        unp_lz = (unp.lazy()
                    .filter(pl.col("value") == method)
                    .select("unit", "node",
                             alias_to_axis("profile", "f")))
        # Input side (node=source, sink=unit).
        uin = _try_entities(source, "unit__inputNode")
        if uin is not None:
            parts.append(
                unp_lz.join(uin.lazy(), on=["unit", "node"], how="inner")
                       .select(
                           alias_to_axis("unit", "p"),
                           alias_to_axis("node", "source"),
                           alias_to_axis("unit", "sink"),
                           "f",
                       ))
        # Output side (source=unit, sink=node).
        uout = _try_entities(source, "unit__outputNode")
        if uout is not None:
            parts.append(
                unp_lz.join(uout.lazy(), on=["unit", "node"], how="inner")
                       .select(
                           alias_to_axis("unit", "p"),
                           alias_to_axis("unit", "source"),
                           alias_to_axis("node", "sink"),
                           "f",
                       ))

    # Connection profile rows (rare in current fixtures but covered for
    # completeness).  connection__node__profile: (connection, node, profile).
    cnp = _try_param(source, "connection__node__profile", "profile_method")
    if cnp is not None:
        cnp_lz = (cnp.lazy()
                    .filter(pl.col("value") == method)
                    .select("connection", "node",
                             alias_to_axis("profile", "f")))
        cnn = _try_entities(source, "connection__node__node")
        if cnn is not None:
            cnn_lz = cnn.lazy()
            parts.append(
                cnp_lz.join(cnn_lz, left_on=["connection", "node"],
                             right_on=["connection", "node_1"], how="inner")
                       .select(
                           alias_to_axis("connection", "p"),
                           alias_to_axis("node", "source"),
                           alias_to_axis("node_2", "sink"),
                           "f",
                       ))
            parts.append(
                cnp_lz.join(cnn_lz, left_on=["connection", "node"],
                             right_on=["connection", "node_2"], how="inner")
                       .select(
                           alias_to_axis("connection", "p"),
                           alias_to_axis("node_1", "source"),
                           alias_to_axis("node", "sink"),
                           "f",
                       ))

    if not parts:
        return _empty(out_schema)
    return (pl.concat(parts)
              .unique()
              .sort("p", "source", "sink", "f")
              .collect())


def node_profile_filter(source: "InputSource", method: str) -> pl.DataFrame:
    """Filter ``node__profile`` by ``profile_method``.

    Schema: ``[n, f]``.  Column ``f`` is flextool's name for the
    profile column.
    """
    df = _try_param(source, "node__profile", "profile_method")
    if df is None:
        return _empty({"n": pl.Utf8, "f": pl.Utf8})
    return (df.lazy()
              .filter(pl.col("value") == method)
              .select(alias_to_axis("node", "n"),
                       alias_to_axis("profile", "f"))
              .sort("n", "f")
              .collect())


def node_profile_upper(source: "InputSource") -> pl.DataFrame:
    return node_profile_filter(source, "upper_limit")


def node_profile_lower(source: "InputSource") -> pl.DataFrame:
    return node_profile_filter(source, "lower_limit")


def node_profile_fixed(source: "InputSource") -> pl.DataFrame:
    return node_profile_filter(source, "fixed")


# ---------------------------------------------------------------------------
# §1.10 — Online / min_load Projections
# ---------------------------------------------------------------------------

def process_online(source: "InputSource") -> pl.DataFrame:
    """Units with ``startup_method ∈ {linear, integer}``.  Schema: ``[p]``."""
    df = _try_param(source, "unit", "startup_method")
    if df is None:
        return _empty({"p": pl.Utf8})
    return (df.lazy()
              .filter(pl.col("value").is_in(["linear", "integer"]))
              .select(alias_to_axis("name", "p"))
              .sort("p")
              .collect())


def process_online_linear(source: "InputSource") -> pl.DataFrame:
    df = _try_param(source, "unit", "startup_method")
    if df is None:
        return _empty({"p": pl.Utf8})
    return (df.lazy()
              .filter(pl.col("value") == "linear")
              .select(alias_to_axis("name", "p"))
              .sort("p")
              .collect())


def process_online_integer(source: "InputSource") -> pl.DataFrame:
    df = _try_param(source, "unit", "startup_method")
    if df is None:
        return _empty({"p": pl.Utf8})
    return (df.lazy()
              .filter(pl.col("value") == "integer")
              .select(alias_to_axis("name", "p"))
              .sort("p")
              .collect())


def process_minload(source: "InputSource") -> pl.DataFrame:
    """Units with ``min_load > 0``.  Schema: ``[p]``.

    ``min_load`` defaults to 0.0 (per §5.3) — entities with the default
    are filtered out.
    """
    df = _try_param(source, "unit", "min_load")
    if df is None:
        return _empty({"p": pl.Utf8})
    return (df.lazy()
              .filter(pl.col("value") > 0)
              .select(alias_to_axis("name", "p"))
              .sort("p")
              .collect())


def process_min_load_eff(source: "InputSource") -> pl.DataFrame:
    """Units with ``conversion_method='min_load_efficiency'``.  Schema: ``[p]``."""
    df = _try_param(source, "unit", "conversion_method")
    if df is None:
        return _empty({"p": pl.Utf8})
    return (df.lazy()
              .filter(pl.col("value") == "min_load_efficiency")
              .select(alias_to_axis("name", "p"))
              .sort("p")
              .collect())


# ---------------------------------------------------------------------------
# §1.11 — Storage Projections
# ---------------------------------------------------------------------------

def nodeState(source: "InputSource") -> pl.DataFrame:
    """Nodes with ``node_type='storage'``.  Schema: ``[n]``.
    """
    df = _try_param(source, "node", "node_type")
    if df is None:
        return _empty({"n": pl.Utf8})
    return (df.lazy()
              .filter(pl.col("value") == "storage")
              .pipe(rename_to_axis, {"name": "n"})
              .select("n")
              .sort("n")
              .collect())


def storage_bind_filter(source: "InputSource", method: str) -> pl.DataFrame:
    """Filter ``node`` set by ``storage_binding_method``."""
    df = _try_param(source, "node", "storage_binding_method")
    if df is None:
        return _empty({"n": pl.Utf8})
    return (df.lazy()
              .filter(pl.col("value") == method)
              .pipe(rename_to_axis, {"name": "n"})
              .select("n")
              .sort("n")
              .collect())


def storage_bind_within_timeblock(source: "InputSource") -> pl.DataFrame:
    return storage_bind_filter(source, "bind_within_timeblock")


def storage_bind_forward_only(source: "InputSource") -> pl.DataFrame:
    return storage_bind_filter(source, "forward_only")


def storage_bind_within_solve(source: "InputSource") -> pl.DataFrame:
    return storage_bind_filter(source, "bind_within_solve")


def storage_bind_within_solve_blended_weights(source: "InputSource") -> pl.DataFrame:
    return storage_bind_filter(source, "bind_within_solve_blended_weights")


def storage_fix_start(source: "InputSource") -> pl.DataFrame:
    """Nodes with ``storage_start_end_method='fix_start'``.  Schema: ``[n]``."""
    df = _try_param(source, "node", "storage_start_end_method")
    if df is None:
        return _empty({"n": pl.Utf8})
    return (df.lazy()
              .filter(pl.col("value") == "fix_start")
              .pipe(rename_to_axis, {"name": "n"})
              .select("n")
              .sort("n")
              .collect())


def n_fix_storage_quantity(source: "InputSource") -> pl.DataFrame:
    """Nodes with ``storage_nested_fix_method='fix_quantity'``.  Schema: ``[n]``."""
    df = _try_param(source, "node", "storage_nested_fix_method")
    if df is None:
        return _empty({"n": pl.Utf8})
    return (df.lazy()
              .filter(pl.col("value") == "fix_quantity")
              .pipe(rename_to_axis, {"name": "n"})
              .select("n")
              .sort("n")
              .collect())


# ---------------------------------------------------------------------------
# §1.14 — Group-level slack Projections
# ---------------------------------------------------------------------------

def _group_yes(source: "InputSource", parameter_name: str) -> pl.DataFrame:
    """Distinct groups where the boolean param is ``yes``.  Schema: ``[g]``."""
    df = _try_param(source, "group", parameter_name)
    if df is None:
        return _empty({"g": pl.Utf8})
    return (df.lazy()
              .filter(pl.col("value") == "yes")
              .select(alias_to_axis("name", "g"))
              .unique()
              .sort("g")
              .collect())


def groupCapacityMargin(source: "InputSource") -> pl.DataFrame:
    return _group_yes(source, "has_capacity_margin")


def groupInertia(source: "InputSource") -> pl.DataFrame:
    return _group_yes(source, "has_inertia")


def groupNonSync(source: "InputSource") -> pl.DataFrame:
    return _group_yes(source, "has_non_synchronous")


def groupStochastic(source: "InputSource") -> pl.DataFrame:
    return _group_yes(source, "include_stochastics")


def group_node(source: "InputSource") -> pl.DataFrame:
    """Membership ``group__node``.  Schema: ``[g, n]``."""
    df = _try_entities(source, "group__node")
    if df is None:
        return _empty({"g": pl.Utf8, "n": pl.Utf8})
    return (df.lazy()
              .pipe(rename_to_axis, {"group": "g", "node": "n"})
              .select("g", "n")
              .sort("g", "n")
              .collect())


def process_unit(source: "InputSource") -> pl.DataFrame:
    """Unit-typed processes (excludes connections).  Schema: ``[p]``."""
    df = _try_entities(source, "unit")
    if df is None:
        return _empty({"p": pl.Utf8})
    return (df.lazy()
              .pipe(rename_to_axis, {"name": "p"})
              .select("p")
              .sort("p")
              .collect())


def process_sink_inertia(source: "InputSource") -> pl.DataFrame:
    """Output-arcs with non-zero ``inertia_constant``.  Schema: ``[p, sink]``."""
    df = _try_param(source, "unit__outputNode", "inertia_constant")
    if df is None:
        return _empty({"p": pl.Utf8, "sink": pl.Utf8})
    return (df.lazy()
              .filter(pl.col("value") != 0)
              .pipe(rename_to_axis, {"unit": "p", "node": "sink"})
              .select("p", "sink")
              .sort("p", "sink")
              .collect())


def process_source_inertia(source: "InputSource") -> pl.DataFrame:
    """Input-arcs with non-zero ``inertia_constant``.  Schema: ``[p, source]``."""
    df = _try_param(source, "unit__inputNode", "inertia_constant")
    if df is None:
        return _empty({"p": pl.Utf8, "source": pl.Utf8})
    return (df.lazy()
              .filter(pl.col("value") != 0)
              .pipe(rename_to_axis, {"unit": "p", "node": "source"})
              .select("p", "source")
              .sort("p", "source")
              .collect())


def process_sink_nonSync(source: "InputSource") -> pl.DataFrame:
    """Output-arcs with ``is_non_synchronous='yes'``.  Schema: ``[p, sink]``."""
    df = _try_param(source, "unit__outputNode", "is_non_synchronous")
    if df is None:
        return _empty({"p": pl.Utf8, "sink": pl.Utf8})
    return (df.lazy()
              .filter(pl.col("value") == "yes")
              .pipe(rename_to_axis, {"unit": "p", "node": "sink"})
              .select("p", "sink")
              .sort("p", "sink")
              .collect())


# ---------------------------------------------------------------------------
# §1.15 — Reserves
# ---------------------------------------------------------------------------

def reserve_upDown_group(source: "InputSource") -> pl.DataFrame:
    """Distinct ``(r, ud, g)`` tuples for reserves with a populated
    ``reserve_method`` (the active reserve subset; entities without a
    method are silently dropped, mirroring flextool's preprocessing).

    Schema: ``[r, ud, g]``.
    """
    df = _try_param(source, "reserve__upDown__group", "reserve_method")
    if df is None:
        return _empty({"r": pl.Utf8, "ud": pl.Utf8, "g": pl.Utf8})
    return (df.lazy()
              .pipe(rename_to_axis, {"reserve": "r", "upDown": "ud", "group": "g"})
              .select("r", "ud", "g")
              .unique()
              .sort("r", "ud", "g")
              .collect())


_RESERVE_METHOD_TO_FIELD = {
    "timeseries": "timeseries_only",
    "dynamic": "dynamic",
    "n_1": "large_failure",
}


def reserve_upDown_group_method(source: "InputSource", method: str) -> pl.DataFrame:
    """Filter ``reserve__upDown__group`` by ``reserve_method`` value.

    ``method`` is the polar_high partition name (``timeseries`` / ``dynamic``
    / ``n_1``); the underlying Spine value is mapped via
    ``_RESERVE_METHOD_TO_FIELD``.
    """
    spine_method = _RESERVE_METHOD_TO_FIELD.get(method, method)
    df = _try_param(source, "reserve__upDown__group", "reserve_method")
    if df is None:
        return _empty({"r": pl.Utf8, "ud": pl.Utf8, "g": pl.Utf8,
                       "method": pl.Utf8})
    return (df.lazy()
              .filter(pl.col("value") == spine_method)
              .pipe(rename_to_axis, {"reserve": "r", "upDown": "ud", "group": "g",
                        "value": "method"})
              .select("r", "ud", "g", "method")
              .sort("r", "ud", "g")
              .collect())


def process_reserve_upDown_node_active(source: "InputSource") -> pl.DataFrame:
    """Union of ``reserve__upDown__unit__node`` and
    ``reserve__upDown__connection__node`` membership.

    Schema: ``[p, r, ud, n]``.
    """
    parts: list[pl.LazyFrame] = []
    u = _try_entities(source, "reserve__upDown__unit__node")
    if u is not None:
        parts.append(u.lazy().select(
            alias_to_axis("unit", "p"),
            alias_to_axis("reserve", "r"),
            alias_to_axis("upDown", "ud"),
            alias_to_axis("node", "n"),
        ))
    c = _try_entities(source, "reserve__upDown__connection__node")
    if c is not None:
        parts.append(c.lazy().select(
            alias_to_axis("connection", "p"),
            alias_to_axis("reserve", "r"),
            alias_to_axis("upDown", "ud"),
            alias_to_axis("node", "n"),
        ))
    if not parts:
        return _empty({"p": pl.Utf8, "r": pl.Utf8, "ud": pl.Utf8, "n": pl.Utf8})
    return (pl.concat(parts).unique().sort("p", "r", "ud", "n").collect())


def _process_reserve_upDown_node_filter(source: "InputSource",
                                          parameter_name: str) -> pl.DataFrame:
    """Union the per-(p, r, ud, n) parameter across the two relationship
    classes, filter to non-zero values, project to membership.
    """
    parts: list[pl.LazyFrame] = []
    u = _try_param(source, "reserve__upDown__unit__node", parameter_name)
    if u is not None:
        parts.append(u.lazy()
                       .filter(pl.col("value") != 0)
                       .select(
                           alias_to_axis("unit", "p"),
                           alias_to_axis("reserve", "r"),
                           alias_to_axis("upDown", "ud"),
                           alias_to_axis("node", "n"),
                       ))
    c = _try_param(source, "reserve__upDown__connection__node", parameter_name)
    if c is not None:
        parts.append(c.lazy()
                       .filter(pl.col("value") != 0)
                       .select(
                           alias_to_axis("connection", "p"),
                           alias_to_axis("reserve", "r"),
                           alias_to_axis("upDown", "ud"),
                           alias_to_axis("node", "n"),
                       ))
    if not parts:
        return _empty({"p": pl.Utf8, "r": pl.Utf8, "ud": pl.Utf8, "n": pl.Utf8})
    return (pl.concat(parts).unique().sort("p", "r", "ud", "n").collect())


def process_reserve_upDown_node_increase_reserve_ratio(source: "InputSource") -> pl.DataFrame:
    return _process_reserve_upDown_node_filter(source, "increase_reserve_ratio")


def process_reserve_upDown_node_large_failure_ratio(source: "InputSource") -> pl.DataFrame:
    return _process_reserve_upDown_node_filter(source, "large_failure_ratio")


# ---------------------------------------------------------------------------
# §1.16 — Cumulative / group-invest Projections
# ---------------------------------------------------------------------------

def _e_param_set(source: "InputSource", parameter_name: str,
                  positive_only: bool = True) -> pl.DataFrame:
    """Distinct entities (across unit/node/connection) where the
    parameter has a (non-zero, positive if ``positive_only``) value.

    Schema: ``[e]``.
    """
    parts: list[pl.LazyFrame] = []
    for cls in ("unit", "node", "connection"):
        df = _try_param(source, cls, parameter_name)
        if df is None:
            continue
        flt = df.lazy()
        if positive_only:
            flt = flt.filter(pl.col("value") > 0)
        parts.append(flt.select(alias_to_axis("name", "e")))
    if not parts:
        return _empty({"e": pl.Utf8})
    return pl.concat(parts).unique().sort("e").collect()


# Mirror ``invest_total_sets.py:25-32``.
_INVEST_METHODS_INVEST_TOTAL: frozenset[str] = frozenset((
    "invest_total", "invest_period_total",
    "invest_retire_total", "invest_retire_period_total",
))
_INVEST_METHODS_DIVEST_TOTAL: frozenset[str] = frozenset((
    "retire_total", "retire_period_total",
    "invest_retire_total", "invest_retire_period_total",
))


def _e_invest_method_filter(source: "InputSource",
                              allowed: frozenset[str]) -> pl.DataFrame:
    """Distinct entities (across unit/node/connection) whose
    ``invest_method`` is in ``allowed``.  Schema: ``[e]``.

    Mirrors ``invest_total_sets.py:74-90``: ``e_invest_total`` is the
    intersection of ``entityInvest`` (allowed-invest method enum) with
    the ``INVEST_TOTAL`` enum subset.
    """
    parts: list[pl.LazyFrame] = []
    for cls in ("unit", "node", "connection"):
        df = _try_param(source, cls, "invest_method")
        if df is None:
            continue
        parts.append(df.lazy()
                       .filter(pl.col("value").is_in(list(allowed)))
                       .select(alias_to_axis("name", "e")))
    if not parts:
        return _empty({"e": pl.Utf8})
    return pl.concat(parts).unique().sort("e").collect()


def e_invest_total(source: "InputSource") -> pl.DataFrame:
    """Entities whose ``invest_method`` ∈ INVEST_TOTAL enum subset.

    Mirrors flextool's ``invest_total_sets.write_invest_total_sets`` —
    filters ``entity__invest_method`` by the four
    ``invest_total / invest_period_total / invest_retire_total /
    invest_retire_period_total`` enum values.
    """
    return _e_invest_method_filter(source, _INVEST_METHODS_INVEST_TOTAL)


def e_divest_total(source: "InputSource") -> pl.DataFrame:
    """Entities whose ``invest_method`` ∈ DIVEST_TOTAL enum subset."""
    return _e_invest_method_filter(source, _INVEST_METHODS_DIVEST_TOTAL)


def ed_invest_cumulative(source: "InputSource",
                          ed_invest: pl.DataFrame | None = None) -> pl.DataFrame:
    """Subset of ``ed_invest`` whose entity carries
    ``cumulative_max_capacity > 0`` OR ``cumulative_min_capacity > 0``
    (the cumulative-cap rows).  Schema: ``[e, d]``.

    NOTE: ``ed_invest`` is itself Derived (Γ.3); this Projection only
    runs when ``ed_invest`` is supplied externally — without it we
    cannot resolve the (e, d) tuples.  When ``ed_invest`` is None we
    return an empty frame and the caller (CSV-side) keeps its native
    value.
    """
    if ed_invest is None or ed_invest.height == 0:
        return _empty({"e": pl.Utf8, "d": pl.Utf8})

    parts: list[pl.LazyFrame] = []
    for parameter_name in ("cumulative_max_capacity", "cumulative_min_capacity"):
        for cls in ("unit", "node", "connection"):
            df = _try_param(source, cls, parameter_name)
            if df is None:
                continue
            parts.append(df.lazy()
                           .filter(pl.col("value") > 0)
                           .select(alias_to_axis("name", "e")))

    if not parts:
        return _empty({"e": pl.Utf8, "d": pl.Utf8})
    cum_e = pl.concat(parts).unique().collect()
    return (ed_invest.lazy()
              .join(cum_e.lazy(), on="e", how="inner")
              .sort("e", "d")
              .collect())


def group_entity(source: "InputSource") -> pl.DataFrame:
    """Union of ``group__node``, ``group__unit``, ``group__connection``.

    Schema: ``[g, e]``.
    """
    parts: list[pl.LazyFrame] = []
    _enums = get_global_axis_enums()
    for cls, dim in (("group__node", "node"),
                      ("group__unit", "unit"),
                      ("group__connection", "connection")):
        df = _try_entities(source, cls)
        if df is None:
            continue
        parts.append(df.lazy().select(
            alias_to_axis("group", "g"),
            # cast across class-specific enum into the entity-union ``e``
            # enum so concat across (group__node, group__unit,
            # group__connection) lines up on a single dtype.
            alias_to_axis(pl.col(dim), "e"),
        ))
    if not parts:
        return _empty({"g": pl.Utf8, "e": pl.Utf8})
    out = pl.concat(parts).unique().sort("g", "e").collect()
    if _enums is not None:
        out = cast_frame_axes(out, _enums)
    return out


def group_process_node(source: "InputSource") -> pl.DataFrame:
    """Union of ``group__unit__node`` and ``group__connection__node``.

    Schema: ``[g, p, n]``.
    """
    parts: list[pl.LazyFrame] = []
    u = _try_entities(source, "group__unit__node")
    if u is not None:
        parts.append(
            u.lazy().pipe(rename_to_axis,
                           {"group": "g", "unit": "p", "node": "n"})
                    .select("g", "p", "n"),
        )
    c = _try_entities(source, "group__connection__node")
    if c is not None:
        parts.append(
            c.lazy().pipe(rename_to_axis,
                           {"group": "g", "connection": "p", "node": "n"})
                    .select("g", "p", "n"),
        )
    if not parts:
        return _empty({"g": pl.Utf8, "p": pl.Utf8, "n": pl.Utf8})
    return pl.concat(parts).unique().sort("g", "p", "n").collect()


def _g_param_set(source: "InputSource", parameter_name: str,
                  positive_only: bool = True) -> pl.DataFrame:
    """Distinct groups with a (positive) value for the named parameter.
    Handles both scalar and 1d_map(period) shapes.  Schema: ``[g]``.
    """
    df = _try_param(source, "group", parameter_name)
    if df is None:
        return _empty({"g": pl.Utf8})
    flt = df.lazy()
    if positive_only:
        flt = flt.filter(pl.col("value") > 0)
    return (flt.select(alias_to_axis("name", "g"))
                .unique()
                .sort("g")
                .collect())


def g_invest_total(source: "InputSource") -> pl.DataFrame:
    return _g_param_set(source, "invest_max_total")


def g_divest_total(source: "InputSource") -> pl.DataFrame:
    return _g_param_set(source, "retire_max_total")


def g_invest_cumulative(source: "InputSource") -> pl.DataFrame:
    return _g_param_set(source, "invest_max_cumulative")


def gdt_maxInstantFlow(source: "InputSource") -> pl.DataFrame:
    """Distinct (g, d, t) where ``group.max_instant_flow`` is set.

    Schema: ``[g, d, t]``.
    """
    df = _try_param(source, "group", "max_instant_flow")
    if df is None:
        return _empty({"g": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8})
    if "period" in df.columns:
        return (df.lazy()
                  .pipe(rename_to_axis, {"name": "g", "period": "d"})
                  .select("g", "d", "t")
                  .unique()
                  .sort("g", "d", "t")
                  .collect())
    # Scalar: no (d, t) — return empty.
    return _empty({"g": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8})


def gdt_minInstantFlow(source: "InputSource") -> pl.DataFrame:
    df = _try_param(source, "group", "min_instant_flow")
    if df is None:
        return _empty({"g": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8})
    if "period" in df.columns:
        return (df.lazy()
                  .pipe(rename_to_axis, {"name": "g", "period": "d"})
                  .select("g", "d", "t")
                  .unique()
                  .sort("g", "d", "t")
                  .collect())
    return _empty({"g": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8})


# ---------------------------------------------------------------------------
# §1.17 — Delayed processes
# ---------------------------------------------------------------------------

def process_delayed(source: "InputSource") -> pl.DataFrame:
    """Units / connections with a non-zero ``delay`` Map.

    ``delay`` is a 1d_map(td) parameter; non-empty rows yield
    membership.  Schema: ``[p]``.
    """
    parts: list[pl.LazyFrame] = []
    for cls in ("unit", "connection"):
        df = _try_param(source, cls, "delay")
        if df is None:
            continue
        parts.append(df.lazy()
                       .filter(pl.col("value") != 0)
                       .select(alias_to_axis("name", "p")))
    if not parts:
        return _empty({"p": pl.Utf8})
    return pl.concat(parts).unique().sort("p").collect()


def process_source_delayed(source: "InputSource",
                            pss: pl.DataFrame | None = None,
                            process_delayed_set: pl.DataFrame | None = None) -> pl.DataFrame:
    """Source-side arcs of delayed processes.  Schema: ``[p, source]``."""
    if pss is None:
        pss = process_source_sink(source)
    if process_delayed_set is None:
        process_delayed_set = process_delayed(source)
    if pss.height == 0 or process_delayed_set.height == 0:
        return _empty({"p": pl.Utf8, "source": pl.Utf8})
    return (pss.lazy()
              .join(process_delayed_set.lazy(), on="p", how="inner")
              .select("p", "source")
              .unique()
              .sort("p", "source")
              .collect())


def process_source_undelayed(source: "InputSource",
                              pss: pl.DataFrame | None = None,
                              process_delayed_set: pl.DataFrame | None = None) -> pl.DataFrame:
    """Source-side arcs of NON-delayed processes.  Schema: ``[p, source]``."""
    if pss is None:
        pss = process_source_sink(source)
    if process_delayed_set is None:
        process_delayed_set = process_delayed(source)
    if pss.height == 0:
        return _empty({"p": pl.Utf8, "source": pl.Utf8})
    return (pss.lazy()
              .join(process_delayed_set.lazy(), on="p", how="anti")
              .select("p", "source")
              .unique()
              .sort("p", "source")
              .collect())


def process_source_sink_delayed(source: "InputSource",
                                  pss: pl.DataFrame | None = None,
                                  process_delayed_set: pl.DataFrame | None = None) -> pl.DataFrame:
    """Full pss tuples of delayed processes.  Schema: ``[p, source, sink]``."""
    if pss is None:
        pss = process_source_sink(source)
    if process_delayed_set is None:
        process_delayed_set = process_delayed(source)
    if pss.height == 0 or process_delayed_set.height == 0:
        return _empty({"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    return (pss.lazy()
              .join(process_delayed_set.lazy(), on="p", how="inner")
              .sort("p", "source", "sink")
              .collect())


def process_source_sink_undelayed(source: "InputSource",
                                    pss: pl.DataFrame | None = None,
                                    process_delayed_set: pl.DataFrame | None = None) -> pl.DataFrame:
    """Full pss tuples of NON-delayed processes.  Schema: ``[p, source, sink]``."""
    if pss is None:
        pss = process_source_sink(source)
    if process_delayed_set is None:
        process_delayed_set = process_delayed(source)
    if pss.height == 0:
        return _empty({"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    return (pss.lazy()
              .join(process_delayed_set.lazy(), on="p", how="anti")
              .sort("p", "source", "sink")
              .collect())


# ---------------------------------------------------------------------------
# §1.18 — DC power flow
# ---------------------------------------------------------------------------

def connection_dc_power_flow(source: "InputSource") -> pl.DataFrame:
    """Connections with ``is_DC='yes'``.  Schema: ``[p]``."""
    df = _try_param(source, "connection", "is_DC")
    if df is None:
        return _empty({"p": pl.Utf8})
    return (df.lazy()
              .filter(pl.col("value") == "yes")
              .select(alias_to_axis("name", "p"))
              .sort("p")
              .collect())


# ---------------------------------------------------------------------------
# §1.19 — Commodity ladder Projections
# ---------------------------------------------------------------------------

_LADDER_METHODS = ("price_ladder_annual", "price_ladder_cumulative")


def commodity_with_ladder(source: "InputSource") -> pl.DataFrame:
    """Commodities with ``price_method`` in the ladder family.  Schema: ``[c]``."""
    df = _try_param(source, "commodity", "price_method")
    if df is None:
        return _empty({"c": pl.Utf8})
    return (df.lazy()
              .filter(pl.col("value").is_in(_LADDER_METHODS))
              .select(alias_to_axis("name", "c"))
              .sort("c")
              .collect())


def commodity_with_ladder_annual(source: "InputSource") -> pl.DataFrame:
    df = _try_param(source, "commodity", "price_method")
    if df is None:
        return _empty({"c": pl.Utf8})
    return (df.lazy()
              .filter(pl.col("value") == "price_ladder_annual")
              .select(alias_to_axis("name", "c"))
              .sort("c")
              .collect())


def commodity_with_ladder_cumulative(source: "InputSource") -> pl.DataFrame:
    df = _try_param(source, "commodity", "price_method")
    if df is None:
        return _empty({"c": pl.Utf8})
    return (df.lazy()
              .filter(pl.col("value") == "price_ladder_cumulative")
              .select(alias_to_axis("name", "c"))
              .sort("c")
              .collect())


def commodity__tier_ann(source: "InputSource") -> pl.DataFrame:
    """Distinct ``(c, i)`` pairs from the annual price ladder.

    Schema: ``[c, i]``.  ``commodity_ladder_annual`` is a structured
    Map(period → Map(tier → Map({"price","quantity"} → f64))).  The
    SpineDbReader split this into ``p_ladder_ann_price`` /
    ``p_ladder_ann_quantity`` keyed on ``(commodity, period, tier)``;
    we project to distinct ``(c, tier)`` pairs.
    """
    # Use the resolved Direct frame as the source — try price first
    # then quantity (either suffices for tier membership).
    for parameter_name in ("price_ladder_annual",):
        df = _try_param(source, "commodity", parameter_name)
        if df is None:
            continue
        cols = df.columns
        if "tier" in cols and "name" in cols:
            return (df.lazy()
                      .pipe(rename_to_axis, {"name": "c", "tier": "i"})
                      .select("c", "i")
                      .unique()
                      .sort("c", "i")
                      .collect())
    return _empty({"c": pl.Utf8, "i": pl.Utf8})


def commodity__tier_cum(source: "InputSource") -> pl.DataFrame:
    """Distinct ``(c, i)`` pairs from the cumulative price ladder."""
    for parameter_name in ("price_ladder_cumulative",):
        df = _try_param(source, "commodity", parameter_name)
        if df is None:
            continue
        cols = df.columns
        if "tier" in cols and "name" in cols:
            return (df.lazy()
                      .pipe(rename_to_axis, {"name": "c", "tier": "i"})
                      .select("c", "i")
                      .unique()
                      .sort("c", "i")
                      .collect())
    return _empty({"c": pl.Utf8, "i": pl.Utf8})


# ---------------------------------------------------------------------------
# Catalog (used by parity tests)
# ---------------------------------------------------------------------------

# Reclassified-Derived list (audit row says "P" but the helper actually
# requires method-aware preprocessing — see flextool's
# preprocessing/process_arc_unions.py 10-way union of method-specific
# arc tuples.  Flagged for Γ.3.):
#   process_source_sink, process_source_sink_eff, process_source_sink_noEff
#   flow_to_commodity, flow_from_commodity_eff, flow_from_commodity_noEff
#   flow_from_co2_priced/_capped (use the eff/noEff partitions)
#   process_indirect, process_input_flows, process_output_flows, process_indirect_dt
#   process_source_delayed, process_source_undelayed,
#   process_source_sink_delayed, process_source_sink_undelayed,
#   pdt_online_linear / pdt_online_integer (depend on p_online_dt — Derived)
#   p_online_dt itself (block-aware — Derived)
#
# These are still implemented in this module (taking pre-built ``pss`` /
# ``process_indirect_set`` / ``process_delayed_set`` arguments) so they
# can be invoked once the prerequisite Derived frames land in Γ.3 — but
# they are NOT wired into ``apply_projection_params``.

# Mapping: FlexData field → callable taking only ``source``.  These are
# clean, source-only Projections — no dependencies on Derived state.
SIMPLE_PROJECTIONS: dict[str, callable] = {
    "nodeBalance": nodeBalance,
    "nodeState": nodeState,
    "process_unit": process_unit,
    "process_minload": process_minload,
    "process_min_load_eff": process_min_load_eff,
    "process_online": process_online,
    "process_online_linear": process_online_linear,
    "process_online_integer": process_online_integer,
    "process_delayed": process_delayed,
    "storage_bind_within_timeblock": storage_bind_within_timeblock,
    "storage_bind_forward_only": storage_bind_forward_only,
    "storage_bind_within_solve": storage_bind_within_solve,
    # Phase C — ``storage_bind_within_solve_blended_weights`` is now
    # derived authoritatively in ``input._load_storage`` from the
    # per-solve ``solve_data/node__storage_binding_method`` CSV, so
    # that the cascade-applied silent downgrade
    # (``_native_run_model._downgrade_rp_methods_for_non_rp_solve``)
    # is honoured instead of being overwritten by the DB-source
    # projection.  The helper :func:`storage_bind_within_solve_blended_weights`
    # remains exported below for callers that want raw DB-source
    # projection (e.g. diagnostics), but is no longer wired here.
    "storage_fix_start": storage_fix_start,
    "n_fix_storage_quantity": n_fix_storage_quantity,
    "groupCapacityMargin": groupCapacityMargin,
    "groupInertia": groupInertia,
    "groupNonSync": groupNonSync,
    "groupStochastic": groupStochastic,
    "group_node": group_node,
    "group_entity": group_entity,
    "group_process_node": group_process_node,
    "group_co2_max_period": group_co2_max_period,
    "process_sink_inertia": process_sink_inertia,
    "process_source_inertia": process_source_inertia,
    "process_sink_nonSync": process_sink_nonSync,
    "node_profile_upper": node_profile_upper,
    "node_profile_lower": node_profile_lower,
    "node_profile_fixed": node_profile_fixed,
    # Δ.29 — process_profile_upper / _lower / _fixed.  The (p, source, sink, f)
    # tuple the slow path's ``_load_profiles`` reconstructs from
    # ``process__source__sink__profile__profile_method.csv`` (itself a
    # 4-way union over ``profileProcess`` connection/direct + indirect
    # source/sink helpers — preprocessing/process_arc_unions.py:1837-1851)
    # is exactly what ``_profile_method_arc`` rebuilds via the
    # ``unit__node__profile`` × ``unit__inputNode`` / ``unit__outputNode``
    # join.  Direct units with a single side (no source OR no sink) — the
    # common VRE case (wind_X with method_1way_1var_off, no source) —
    # land on the output-side branch (``source=unit, sink=node``) which
    # matches the ``no_source_arcs`` partition of
    # :func:`process_source_sink_canonical`.  Connection-side profiles
    # are covered via ``connection__node__profile`` × ``connection__node__node``.
    # Without this wiring on the fast path the wind upper-limit cap is
    # absent → the LP satisfies demand from unconstrained wind → coal
    # flow stays at 0 → obj=0 (Δ.28's gap diagnosis).
    "process_profile_upper": process_profile_upper,
    "process_profile_lower": process_profile_lower,
    "process_profile_fixed": process_profile_fixed,
    "reserve_upDown_group": reserve_upDown_group,
    # process_reserve_upDown_node_active / _increase_reserve_ratio /
    # _large_failure_ratio are reclassified Derived: their CSV-side
    # values are intersected with the per-(d, t) reservation > 0
    # "active" set, which itself depends on pdtReserve_upDown_group_
    # reservation (a Direct-Param-fed cascade, Derived in our scope).
    # See preprocessing/reserve_calc_params.py:237 for the gate.
    "e_invest_total": e_invest_total,
    "e_divest_total": e_divest_total,
    "g_invest_total": g_invest_total,
    "g_divest_total": g_divest_total,
    "g_invest_cumulative": g_invest_cumulative,
    "gdt_maxInstantFlow": gdt_maxInstantFlow,
    "gdt_minInstantFlow": gdt_minInstantFlow,
    "connection_dc_power_flow": connection_dc_power_flow,
    "commodity_with_ladder": commodity_with_ladder,
    "commodity_with_ladder_annual": commodity_with_ladder_annual,
    "commodity_with_ladder_cumulative": commodity_with_ladder_cumulative,
    "commodity__tier_ann": commodity__tier_ann,
    "commodity__tier_cum": commodity__tier_cum,
    # Δ.17c Gap D — process_source_sink × ramp_method partitions.
    "process_source_sink_ramp_limit_source_up":
        process_source_sink_ramp_limit_source_up,
    "process_source_sink_ramp_limit_source_down":
        process_source_sink_ramp_limit_source_down,
    "process_source_sink_ramp_limit_sink_up":
        process_source_sink_ramp_limit_sink_up,
    "process_source_sink_ramp_limit_sink_down":
        process_source_sink_ramp_limit_sink_down,
    "process_source_sink_ramp_cost":
        process_source_sink_ramp_cost,
}


def apply_projection_params(source: "InputSource",
                              flex_data: object) -> None:
    """Apply the DB-direct construction for Γ.2 Projection Params,
    mutating ``flex_data`` in place.

    Δ.12b — defensive ``try / except: continue`` removed; helper
    exceptions propagate.  The
    ``if v is None or v.height == 0: continue`` guard is retained:
    SIMPLE_PROJECTIONS' SET-frame helpers may return empty when the
    source has rows for a class but none match the projection's
    membership filter, in which case we want to keep the seed's
    populated value rather than zero it out.  Per-helper coverage
    audits (Δ.12-drop preparation) will fold this into helper-side
    "produce-or-raise" once each helper's empty-vs-None semantics
    are pinned.
    """
    # First pass: simple no-deps projections.
    for field_name, fn in SIMPLE_PROJECTIONS.items():
        v = fn(source)
        if v is None or (hasattr(v, "height") and v.height == 0):
            continue
        setattr(flex_data, field_name, v)

    # Second pass: projections that need ``dt`` from the CSV-loaded
    # FlexData — they overlay the CSV value of the dependent fields.
    dt = getattr(flex_data, "dt", None)

    # cdt_eq / cdt_le / cdt_ge — only emit when DB has constraints.
    if dt is not None:
        for sense, field_name in (("equal", "cdt_eq"),
                              ("less_than", "cdt_le"),
                              ("greater_than", "cdt_ge")):
            cdt = cdt_filter(source, sense, dt)
            if cdt is not None and cdt.height > 0:
                setattr(flex_data, field_name, cdt)

    # Reserve-method partitions — applied unconditionally; emit only
    # when the DB-side computes a non-empty frame.
    for method, field_name in (
        ("timeseries", "reserve_upDown_group_method_timeseries"),
        ("dynamic", "reserve_upDown_group_method_dynamic"),
        ("n_1", "reserve_upDown_group_method_n_1"),
    ):
        v = reserve_upDown_group_method(source, method)
        if v is not None and v.height > 0:
            setattr(flex_data, field_name, v)
    return {}
