"""Γ.2 Projection Param helpers.

Each function in this module takes an :class:`InputSource` and returns
the corresponding flexpy frame (or ``None`` if the projection is empty
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

if TYPE_CHECKING:
    from flextool.engine_polars._input_source import InputSource


# ---------------------------------------------------------------------------
# Helpers for the common shape of "rename source to flexpy column names"
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
              .rename({"name": "n"})
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
    """
    parts: list[pl.LazyFrame] = []

    uin = _try_entities(source, "unit__inputNode")
    if uin is not None:
        parts.append(uin.lazy().select(
            pl.col("unit").alias("p"),
            pl.col("node").alias("source"),
            pl.col("unit").alias("sink"),
        ))
    uout = _try_entities(source, "unit__outputNode")
    if uout is not None:
        parts.append(uout.lazy().select(
            pl.col("unit").alias("p"),
            pl.col("unit").alias("source"),
            pl.col("node").alias("sink"),
        ))
    cnn = _try_entities(source, "connection__node__node")
    if cnn is not None:
        parts.append(cnn.lazy().select(
            pl.col("connection").alias("p"),
            pl.col("node_1").alias("source"),
            pl.col("node_2").alias("sink"),
        ))

    if not parts:
        return _empty({"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    return pl.concat(parts).unique().sort("p", "source", "sink").collect()


# Conversion methods that are "efficiency-based" (eff partition).  The
# .mod calls these the ``method_X_LP`` family; flextool's preprocessor
# also includes ``min_load_efficiency`` and ``constant_efficiency`` for
# the Projection.  See preprocessing/process_method_sets.py.
_EFF_CONVERSION_METHODS = ("min_load_efficiency", "constant_efficiency",
                            "method_X_LP")


def process_source_sink_eff(source: "InputSource",
                              pss: pl.DataFrame | None = None) -> pl.DataFrame:
    """``pss`` filtered to processes with an efficiency-based
    ``conversion_method`` (units only; connections never participate
    in eff).

    Schema: ``[p, source, sink]``.
    """
    if pss is None:
        pss = process_source_sink(source)
    if pss.height == 0:
        return _empty({"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    cm = _try_param(source, "unit", "conversion_method")
    if cm is None:
        return _empty({"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    eff_p = (cm.lazy()
                .filter(pl.col("value").is_in(_EFF_CONVERSION_METHODS))
                .select(pl.col("name").alias("p")))
    return (pss.lazy()
              .join(eff_p, on="p", how="inner")
              .sort("p", "source", "sink")
              .collect())


def process_source_sink_noEff(source: "InputSource",
                                pss: pl.DataFrame | None = None,
                                pss_eff: pl.DataFrame | None = None) -> pl.DataFrame:
    """Complement of ``process_source_sink_eff`` within ``pss`` — the
    arcs whose process is NOT efficiency-based (connections + units
    with ``conversion_method ∉ eff family``).

    Schema: ``[p, source, sink]``.
    """
    if pss is None:
        pss = process_source_sink(source)
    if pss_eff is None:
        pss_eff = process_source_sink_eff(source, pss)
    if pss.height == 0:
        return _empty({"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    return (pss.lazy()
              .join(pss_eff.lazy(), on=["p", "source", "sink"], how="anti")
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
              .rename({"commodity": "c"})
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
              .rename({"commodity": "c"})
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
              .rename({"commodity": "c"})
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
              .select(pl.col("name").alias("g"))
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
              .select(pl.col("name").alias("p"))
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
              .filter(pl.col("sink") == pl.col("p"))
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
              .filter(pl.col("source") == pl.col("p"))
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
    return (process_indirect_set.lazy()
              .join(dt.lazy(), how="cross")
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
              .select(pl.col("name").alias("c"))
              .collect())
    if rows.height == 0:
        return None
    return (rows.lazy()
              .join(dt.lazy(), how="cross")
              .sort("c", "d", "t")
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

    ``dim_renames`` maps Spine column names → flexpy column names;
    ``out_cols`` is the projection.
    """
    df = _try_param(source, profile_class, "profile_method")
    if df is None:
        empty_schema = {c: pl.Utf8 for c in out_cols}
        return _empty(empty_schema)
    return (df.lazy()
              .filter(pl.col("value") == method)
              .rename(dim_renames)
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
                             pl.col("profile").alias("f")))
        # Input side (node=source, sink=unit).
        uin = _try_entities(source, "unit__inputNode")
        if uin is not None:
            parts.append(
                unp_lz.join(uin.lazy(), on=["unit", "node"], how="inner")
                       .select(
                           pl.col("unit").alias("p"),
                           pl.col("node").alias("source"),
                           pl.col("unit").alias("sink"),
                           "f",
                       ))
        # Output side (source=unit, sink=node).
        uout = _try_entities(source, "unit__outputNode")
        if uout is not None:
            parts.append(
                unp_lz.join(uout.lazy(), on=["unit", "node"], how="inner")
                       .select(
                           pl.col("unit").alias("p"),
                           pl.col("unit").alias("source"),
                           pl.col("node").alias("sink"),
                           "f",
                       ))

    # Connection profile rows (rare in current fixtures but covered for
    # completeness).  connection__node__profile: (connection, node, profile).
    cnp = _try_param(source, "connection__node__profile", "profile_method")
    if cnp is not None:
        cnp_lz = (cnp.lazy()
                    .filter(pl.col("value") == method)
                    .select("connection", "node",
                             pl.col("profile").alias("f")))
        cnn = _try_entities(source, "connection__node__node")
        if cnn is not None:
            cnn_lz = cnn.lazy()
            parts.append(
                cnp_lz.join(cnn_lz, left_on=["connection", "node"],
                             right_on=["connection", "node_1"], how="inner")
                       .select(
                           pl.col("connection").alias("p"),
                           pl.col("node").alias("source"),
                           pl.col("node_2").alias("sink"),
                           "f",
                       ))
            parts.append(
                cnp_lz.join(cnn_lz, left_on=["connection", "node"],
                             right_on=["connection", "node_2"], how="inner")
                       .select(
                           pl.col("connection").alias("p"),
                           pl.col("node_1").alias("source"),
                           pl.col("node").alias("sink"),
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
              .select(pl.col("node").alias("n"),
                       pl.col("profile").alias("f"))
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
              .select(pl.col("name").alias("p"))
              .sort("p")
              .collect())


def process_online_linear(source: "InputSource") -> pl.DataFrame:
    df = _try_param(source, "unit", "startup_method")
    if df is None:
        return _empty({"p": pl.Utf8})
    return (df.lazy()
              .filter(pl.col("value") == "linear")
              .select(pl.col("name").alias("p"))
              .sort("p")
              .collect())


def process_online_integer(source: "InputSource") -> pl.DataFrame:
    df = _try_param(source, "unit", "startup_method")
    if df is None:
        return _empty({"p": pl.Utf8})
    return (df.lazy()
              .filter(pl.col("value") == "integer")
              .select(pl.col("name").alias("p"))
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
              .select(pl.col("name").alias("p"))
              .sort("p")
              .collect())


def process_min_load_eff(source: "InputSource") -> pl.DataFrame:
    """Units with ``conversion_method='min_load_efficiency'``.  Schema: ``[p]``."""
    df = _try_param(source, "unit", "conversion_method")
    if df is None:
        return _empty({"p": pl.Utf8})
    return (df.lazy()
              .filter(pl.col("value") == "min_load_efficiency")
              .select(pl.col("name").alias("p"))
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
              .rename({"name": "n"})
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
              .rename({"name": "n"})
              .select("n")
              .sort("n")
              .collect())


def storage_bind_within_timeset(source: "InputSource") -> pl.DataFrame:
    return storage_bind_filter(source, "bind_within_timeset")


def storage_bind_forward_only(source: "InputSource") -> pl.DataFrame:
    return storage_bind_filter(source, "forward_only")


def storage_bind_within_solve(source: "InputSource") -> pl.DataFrame:
    return storage_bind_filter(source, "bind_within_solve")


def storage_fix_start(source: "InputSource") -> pl.DataFrame:
    """Nodes with ``storage_start_end_method='fix_start'``.  Schema: ``[n]``."""
    df = _try_param(source, "node", "storage_start_end_method")
    if df is None:
        return _empty({"n": pl.Utf8})
    return (df.lazy()
              .filter(pl.col("value") == "fix_start")
              .rename({"name": "n"})
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
              .rename({"name": "n"})
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
              .select(pl.col("name").alias("g"))
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
              .rename({"group": "g", "node": "n"})
              .select("g", "n")
              .sort("g", "n")
              .collect())


def process_unit(source: "InputSource") -> pl.DataFrame:
    """Unit-typed processes (excludes connections).  Schema: ``[p]``."""
    df = _try_entities(source, "unit")
    if df is None:
        return _empty({"p": pl.Utf8})
    return (df.lazy()
              .rename({"name": "p"})
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
              .rename({"unit": "p", "node": "sink"})
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
              .rename({"unit": "p", "node": "source"})
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
              .rename({"unit": "p", "node": "sink"})
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
              .rename({"reserve": "r", "upDown": "ud", "group": "g"})
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

    ``method`` is the flexpy partition name (``timeseries`` / ``dynamic``
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
              .rename({"reserve": "r", "upDown": "ud", "group": "g",
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
            pl.col("unit").alias("p"),
            pl.col("reserve").alias("r"),
            pl.col("upDown").alias("ud"),
            pl.col("node").alias("n"),
        ))
    c = _try_entities(source, "reserve__upDown__connection__node")
    if c is not None:
        parts.append(c.lazy().select(
            pl.col("connection").alias("p"),
            pl.col("reserve").alias("r"),
            pl.col("upDown").alias("ud"),
            pl.col("node").alias("n"),
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
                           pl.col("unit").alias("p"),
                           pl.col("reserve").alias("r"),
                           pl.col("upDown").alias("ud"),
                           pl.col("node").alias("n"),
                       ))
    c = _try_param(source, "reserve__upDown__connection__node", parameter_name)
    if c is not None:
        parts.append(c.lazy()
                       .filter(pl.col("value") != 0)
                       .select(
                           pl.col("connection").alias("p"),
                           pl.col("reserve").alias("r"),
                           pl.col("upDown").alias("ud"),
                           pl.col("node").alias("n"),
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
        parts.append(flt.select(pl.col("name").alias("e")))
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
                       .select(pl.col("name").alias("e")))
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
                           .select(pl.col("name").alias("e")))

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
    for cls, dim in (("group__node", "node"),
                      ("group__unit", "unit"),
                      ("group__connection", "connection")):
        df = _try_entities(source, cls)
        if df is None:
            continue
        parts.append(df.lazy().select(
            pl.col("group").alias("g"),
            pl.col(dim).alias("e"),
        ))
    if not parts:
        return _empty({"g": pl.Utf8, "e": pl.Utf8})
    return pl.concat(parts).unique().sort("g", "e").collect()


def group_process_node(source: "InputSource") -> pl.DataFrame:
    """Union of ``group__unit__node`` and ``group__connection__node``.

    Schema: ``[g, p, n]``.
    """
    parts: list[pl.LazyFrame] = []
    u = _try_entities(source, "group__unit__node")
    if u is not None:
        parts.append(u.lazy().select(
            pl.col("group").alias("g"),
            pl.col("unit").alias("p"),
            pl.col("node").alias("n"),
        ))
    c = _try_entities(source, "group__connection__node")
    if c is not None:
        parts.append(c.lazy().select(
            pl.col("group").alias("g"),
            pl.col("connection").alias("p"),
            pl.col("node").alias("n"),
        ))
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
    return (flt.select(pl.col("name").alias("g"))
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
                  .rename({"name": "g", "period": "d"})
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
                  .rename({"name": "g", "period": "d"})
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
                       .select(pl.col("name").alias("p")))
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
              .select(pl.col("name").alias("p"))
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
              .select(pl.col("name").alias("c"))
              .sort("c")
              .collect())


def commodity_with_ladder_annual(source: "InputSource") -> pl.DataFrame:
    df = _try_param(source, "commodity", "price_method")
    if df is None:
        return _empty({"c": pl.Utf8})
    return (df.lazy()
              .filter(pl.col("value") == "price_ladder_annual")
              .select(pl.col("name").alias("c"))
              .sort("c")
              .collect())


def commodity_with_ladder_cumulative(source: "InputSource") -> pl.DataFrame:
    df = _try_param(source, "commodity", "price_method")
    if df is None:
        return _empty({"c": pl.Utf8})
    return (df.lazy()
              .filter(pl.col("value") == "price_ladder_cumulative")
              .select(pl.col("name").alias("c"))
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
                      .rename({"name": "c", "tier": "i"})
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
                      .rename({"name": "c", "tier": "i"})
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
# they are NOT wired into ``projection_overrides``.

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
    "storage_bind_within_timeset": storage_bind_within_timeset,
    "storage_bind_forward_only": storage_bind_forward_only,
    "storage_bind_within_solve": storage_bind_within_solve,
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
    # process_profile_upper / _lower / _fixed are reclassified Derived:
    # the (p, source, sink) tuple flextool emits is the unit-arc tuple
    # threaded through the 10-way ``process_source_sink`` union (which
    # collapses unit input + output into a single arc keyed by the
    # unit's actual source node).  The DB-direct projection over
    # unit__node__profile loses the input-side source node; only the
    # post-Γ.3 ``process_source_sink`` resolution can restore it.
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
}


def projection_overrides(source: "InputSource",
                          flex_data: object) -> dict[str, object | None]:
    """Compute the DB-direct override dict for Γ.2 Projection Params.

    Returns ``{FlexData_field: replacement_value}`` containing only
    fields whose DB-direct equivalent is structurally well-defined and
    *non-empty*.  When the DB-side computes an empty frame, the
    CSV-loaded value is preserved (mirrors Γ.1's first_wave_overrides
    behaviour: empty DB-side overlays should not blank-out CSV data).

    Per the Γ.3.G architectural shift, no defensive feature gating:
    every overlay applies unconditionally.  The earlier ``_FEATURE_GATED``
    map silenced "the DB path activates a feature the CSV path skipped"
    cases by suppressing the override; that pattern is removed because
    such mismatches signal real divergence between the two paths and
    should fail loudly so Γ.3 helpers cover them.

    The override is applied **after** the CSV path has populated every
    field — so any field not in this dict keeps its CSV-loaded value.
    """
    out: dict[str, object | None] = {}

    # First pass: simple no-deps projections.
    for field, fn in SIMPLE_PROJECTIONS.items():
        try:
            v = fn(source)
        except Exception:  # pragma: no cover — diagnostic surface
            continue
        if v is None or (hasattr(v, "height") and v.height == 0):
            continue
        out[field] = v

    # Second pass: projections that need ``dt`` from the CSV-loaded
    # FlexData — they overlay the CSV value of the dependent fields.
    dt = getattr(flex_data, "dt", None)

    # cdt_eq / cdt_le / cdt_ge — only emit when DB has constraints.
    if dt is not None:
        for sense, field in (("equal", "cdt_eq"),
                              ("less_than", "cdt_le"),
                              ("greater_than", "cdt_ge")):
            try:
                cdt = cdt_filter(source, sense, dt)
            except Exception:  # pragma: no cover
                continue
            if cdt is not None and cdt.height > 0:
                out[field] = cdt

    # Reserve-method partitions — applied unconditionally; emit only
    # when the DB-side computes a non-empty frame.
    for method, field in (
        ("timeseries", "reserve_upDown_group_method_timeseries"),
        ("dynamic", "reserve_upDown_group_method_dynamic"),
        ("n_1", "reserve_upDown_group_method_n_1"),
    ):
        v = reserve_upDown_group_method(source, method)
        if v is not None and v.height > 0:
            out[field] = v

    return out
