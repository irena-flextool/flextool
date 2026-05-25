"""Silent-default index regression — Tier-3 ``pdt_varCost_*`` /
``pdtReserve_upDown_group_reservation`` Map(period→time) helpers.

The four helpers below operate on the LP's variable-cost and reservation
Param families:

* ``p_pdt_varCost_source_from_source`` — ``unit__inputNode.other_operational_cost``
* ``p_pdt_varCost_sink_from_source``   — ``unit__outputNode.other_operational_cost``
* ``p_pdt_varCost_process_from_source`` — ``unit ∪ connection.other_operational_cost``
* ``pdtReserve_upDown_group_reservation_from_source`` — ``reserve__upDown__group.reservation``

Pre-migration, each helper inlined a column-name gate
(``if not {..., "period", "t"}.issubset(cols)``) that silently dropped
parameters whose Map ``index_name`` carried spinedb_api's silent default
(``"x" / "" / None``) instead of the canonical ``"period" / "t"``.  The
guard also rejected SCALAR / MAP_PERIOD / MAP_TIME shapes outright.

Δ.17c-Tier3 routes each helper through
:func:`flextool.engine_polars._param_shapes.resolve_param_shape` +
:func:`flextool.engine_polars._param_shapes.broadcast_to_period_time`.
``broadcast_to_period_time`` was extended in this commit to accept a
per-source-column rename dict, since three of the four parameters live
on relationship classes (``unit__inputNode`` 2-dim, ``unit__outputNode``
2-dim, ``reserve__upDown__group`` 3-dim).

A real-world (Cyprus migration) audit on 2026-05-25 showed the
``other_operational_cost`` family is authored as plain SCALAR floats
on ``unit__outputNode`` and ``reservation`` as SCALAR floats on
``reserve__upDown__group`` — the silent-default Map path is rarely
exercised by production DBs but the column-gate bug existed for any
user who DID author a Map.  The synthetic stubs below lock both the
SCALAR-broadcast and Map(period→time) paths without bundling proprietary
DB fixtures (per CONTRIBUTING.md).
"""
from __future__ import annotations

import polars as pl
import pytest


_PERIOD_NAMES = ["p2024", "p2025", "p2026", "p2027", "p2028", "p2029",
                 "p2030", "p2035", "p2040", "p2045", "p2050"]
_TIMESTEPS = ["t00001", "t00002", "t00003"]


@pytest.fixture(scope="module")
def period_filter() -> pl.DataFrame:
    """A ``[d, t]`` frame covering the scenario's periods × a small
    timestep panel.  The helpers consult both axes (Map(period→time)
    is the canonical Tier-3 shape).
    """
    rows: list[tuple[str, str]] = []
    for d in _PERIOD_NAMES:
        for t in _TIMESTEPS:
            rows.append((d, t))
    return pl.DataFrame({
        "d": [r[0] for r in rows],
        "t": [r[1] for r in rows],
    })


def _collect(param) -> pl.DataFrame:
    """Materialise a ``Param.frame`` whether eager or lazy."""
    frame = param.frame
    return frame.collect() if hasattr(frame, "collect") else frame


def _assert_no_key_duplicates(param, key_cols: tuple[str, ...]) -> None:
    """No two rows share the same key.  The resolver guarantees
    structural dedup; this is the safety net for the relationship-
    class rename pipeline.
    """
    frame = _collect(param)
    frame = frame.with_columns(*[pl.col(c).cast(pl.Utf8) for c in key_cols])
    dupes = (
        frame.group_by(list(key_cols))
             .len()
             .filter(pl.col("len") > 1)
    )
    assert dupes.height == 0, (
        f"Duplicate {key_cols} keys after Tier-3 helper:\n{dupes.head(10)}"
    )


# ---------------------------------------------------------------------------
# SCALAR-shape regression — synthetic stub mirroring real-world authoring
#
# A Cyprus-migration audit (2026-05-25, kept locally) found
# ``unit__outputNode.other_operational_cost`` is authored as plain SCALAR
# floats across 99 rows and ``reserve__upDown__group.reservation`` as
# 2 SCALAR rows.  Pre-migration the helper's column-gate
# (``{"period", "t"}.issubset(cols)``) silently REJECTED SCALAR shapes
# because no ``period`` / ``t`` columns existed.  Post-migration the
# resolver lands on ``Shape.SCALAR`` and ``broadcast_to_period_time``
# returns a (p, sink) Param that polar_high broadcasts against the LHS
# (p, sink, d, t) at LP build time.
#
# This stub is inspired by Cyprus's shape but uses synthetic fixtures
# only (per CONTRIBUTING.md — no proprietary DB content in the repo).
# ---------------------------------------------------------------------------


class _ScalarRelationshipStub:
    """Surfaces a single SCALAR-authored row on an n-dim relationship
    class.  The frame has no period / t columns — the pre-migration
    helper's column-gate would silently REJECT this (Cyprus-observed
    real-world authoring shape).
    """

    def __init__(self, entity_class: str, parameter_name: str,
                 entity_cols: tuple[str, ...],
                 entity_values: tuple[str, ...],
                 value: float) -> None:
        self._entity_class = entity_class
        self._parameter_name = parameter_name
        self._entity_cols = entity_cols
        self._entity_values = entity_values
        self._value = value

    def parameter_explicit(self, entity_class: str, parameter_name: str):
        if (entity_class != self._entity_class
                or parameter_name != self._parameter_name):
            raise KeyError((entity_class, parameter_name))
        data: dict[str, list] = {c: [v] for c, v in
                                  zip(self._entity_cols, self._entity_values)}
        data["value"] = [self._value]
        return pl.DataFrame(data)

    def parameter(self, entity_class: str, parameter_name: str):
        return self.parameter_explicit(entity_class, parameter_name)

    def parameter_shape_info(self, entity_class: str, parameter_name: str):
        # SCALAR — depth-0 (no Map index labels).
        return []

    def entities(self, entity_class: str):
        if entity_class == self._entity_class:
            return pl.DataFrame({c: [v] for c, v in
                                  zip(self._entity_cols, self._entity_values)})
        return pl.DataFrame(schema={c: pl.Utf8 for c in self._entity_cols})


def test_scalar_relationship_unit_outputNode(period_filter) -> None:
    """Inspired by Cyprus's authoring shape: SCALAR ``float`` on
    ``unit__outputNode.other_operational_cost``.  Pre-migration the
    helper's column-gate (``{"period", "t"}.issubset(cols)``) silently
    DROPPED such rows because no period / t columns exist.  Post-
    migration the resolver lands on ``Shape.SCALAR`` and returns a
    ``Param(("p", "sink"))``.
    """
    import flextool.engine_polars._direct_params as dp
    stub = _ScalarRelationshipStub(
        entity_class="unit__outputNode",
        parameter_name="other_operational_cost",
        entity_cols=("unit", "node"),
        entity_values=("stub_unit", "stub_sink"),
        value=12.5,
    )
    result = dp.p_pdt_varCost_sink_from_source(
        stub, period_filter=period_filter)
    assert result is not None, (
        "SCALAR authoring on unit__outputNode.other_operational_cost "
        "was dropped — pre-migration column-gate bug."
    )
    # SCALAR stays at the entity dims; polar_high broadcasts to (d, t)
    # at LP build time.  No dups on the entity key.
    assert result.dims[:2] == ("p", "sink"), result.dims
    _assert_no_key_duplicates(result, result.dims)


def test_scalar_relationship_reserve_upDown_group(period_filter) -> None:
    """Inspired by Cyprus's authoring shape: SCALAR ``float`` on
    ``reserve__upDown__group.reservation``.  Same pre-migration column-
    gate bug; post-migration the resolver emits Param ``(r, ud, g)``.
    ``filter_zero=False`` is preserved (the helper accepts non-zero
    scalars uniformly via the resolver path).
    """
    import flextool.engine_polars._direct_params as dp
    stub = _ScalarRelationshipStub(
        entity_class="reserve__upDown__group",
        parameter_name="reservation",
        entity_cols=("reserve", "upDown", "group"),
        entity_values=("stub_reserve", "up", "stub_group"),
        value=3.0,
    )
    result = dp.pdtReserve_upDown_group_reservation_from_source(
        stub, period_filter=period_filter)
    assert result is not None
    assert result.dims[:3] == ("r", "ud", "g"), result.dims
    _assert_no_key_duplicates(result, result.dims)


# ---------------------------------------------------------------------------
# Silent-default Map regression — synthetic InputSource
#
# spinedb_api's silent-default Map authoring stores the index column as
# ``"x"`` rather than the canonical ``"period"``.  Pre-migration the
# Tier-3 helpers gated on ``{"period", "t"}.issubset(cols)`` and so
# DROPPED such rows entirely.  These stubs lock the post-migration
# resolver path (via ``_infer_silent_default_labels`` + value-domain
# probing) landing on the same period × time → value association.
# ---------------------------------------------------------------------------


class _SilentDefaultMapPeriodTimeStub:
    """Surfaces a 2d_map(period → time) authored with silent-default
    index names on either or both levels.

    The SpineDbReader fallback names the index columns with whatever
    ``index_name`` the author supplied (or its silent default, e.g.
    ``"x"`` and ``"x_1"``).  For the resolver to land on
    ``Shape.MAP_PERIOD_TIME`` it consults
    :meth:`parameter_shape_info` for the raw labels and then
    value-domain probes the authored index values against the active
    solve's period and time vocabularies.
    """

    def __init__(self,
                 entity_class: str,
                 parameter_name: str,
                 entity_cols: tuple[str, ...],
                 entity_values: tuple[str, ...],
                 period_time_values: dict[tuple[str, str], float],
                 index_names: tuple[str, str] = ("x", "x"),
                 frame_index_cols: tuple[str, str] = ("x", "x_1")) -> None:
        self._entity_class = entity_class
        self._parameter_name = parameter_name
        self._entity_cols = entity_cols
        self._entity_values = entity_values
        self._period_time_values = period_time_values
        self._index_names = index_names
        self._frame_index_cols = frame_index_cols

    def parameter_explicit(self, entity_class: str, parameter_name: str):
        if (entity_class != self._entity_class
                or parameter_name != self._parameter_name):
            raise KeyError((entity_class, parameter_name))
        # Build the flat frame the SpineDbReader would emit.  spinedb_api
        # uses ``index_name="x"`` on every silent-default level; in the
        # FLAT polars frame the columns must be unique so the second
        # level becomes ``"x_1"``.  ``parameter_shape_info`` returns the
        # RAW labels (with duplicates intact).
        ix0, ix1 = self._frame_index_cols
        keys = list(self._period_time_values.keys())
        n = len(keys)
        data: dict[str, list] = {c: [v] * n for c, v in
                                  zip(self._entity_cols, self._entity_values)}
        data[ix0] = [k[0] for k in keys]
        data[ix1] = [k[1] for k in keys]
        data["value"] = [self._period_time_values[k] for k in keys]
        return pl.DataFrame(data)

    def parameter(self, entity_class: str, parameter_name: str):
        return self.parameter_explicit(entity_class, parameter_name)

    def parameter_shape_info(self, entity_class: str, parameter_name: str):
        # Silent-default: spinedb_api emits "x" labels.
        return list(self._index_names)

    def entities(self, entity_class: str):
        if entity_class == self._entity_class:
            schema = {c: pl.Utf8 for c in self._entity_cols}
            data = {c: [v] for c, v in
                    zip(self._entity_cols, self._entity_values)}
            return pl.DataFrame(data, schema=schema)
        return pl.DataFrame(schema={c: pl.Utf8 for c in self._entity_cols})


def _pt_values() -> dict[tuple[str, str], float]:
    """A 3-period × 2-timestep authoring fragment.  All keys are inside
    the ``period_filter`` fixture so the resolver's value-domain
    probing recognises them as (period, time)."""
    return {
        ("p2024", "t00001"): 1.5,
        ("p2024", "t00002"): 2.5,
        ("p2025", "t00001"): 3.5,
        ("p2030", "t00003"): 4.5,
    }


def test_p_pdt_varCost_source_silent_default(period_filter) -> None:
    """``unit__inputNode.other_operational_cost`` silent-default Map →
    Param ``(p, source, d, t)``, period→value AND time→value preserved.
    """
    import flextool.engine_polars._direct_params as dp
    pt_values = _pt_values()
    stub = _SilentDefaultMapPeriodTimeStub(
        entity_class="unit__inputNode",
        parameter_name="other_operational_cost",
        entity_cols=("unit", "node"),
        entity_values=("stub_unit", "stub_node"),
        period_time_values=pt_values,
    )
    result = dp.p_pdt_varCost_source_from_source(
        stub, period_filter=period_filter)
    assert result is not None, (
        "p_pdt_varCost_source_from_source returned None for a "
        "silent-default Map(period→time) source."
    )
    assert result.dims == ("p", "source", "d", "t"), result.dims
    _assert_no_key_duplicates(result, ("p", "source", "d", "t"))
    frame = _collect(result).with_columns(
        pl.col("p").cast(pl.Utf8),
        pl.col("source").cast(pl.Utf8),
        pl.col("d").cast(pl.Utf8),
        pl.col("t").cast(pl.Utf8),
    )
    got = {(d, t): v for d, t, v in
            frame.select("d", "t", "value").iter_rows()}
    assert got == pt_values, (
        f"(d, t) → value association lost: got {got}, expected {pt_values}"
    )


def test_p_pdt_varCost_sink_silent_default(period_filter) -> None:
    """``unit__outputNode.other_operational_cost`` silent-default Map →
    Param ``(p, sink, d, t)``.
    """
    import flextool.engine_polars._direct_params as dp
    pt_values = _pt_values()
    stub = _SilentDefaultMapPeriodTimeStub(
        entity_class="unit__outputNode",
        parameter_name="other_operational_cost",
        entity_cols=("unit", "node"),
        entity_values=("stub_unit", "stub_node"),
        period_time_values=pt_values,
    )
    result = dp.p_pdt_varCost_sink_from_source(
        stub, period_filter=period_filter)
    assert result is not None
    assert result.dims == ("p", "sink", "d", "t"), result.dims
    _assert_no_key_duplicates(result, ("p", "sink", "d", "t"))
    frame = _collect(result).with_columns(
        pl.col("p").cast(pl.Utf8),
        pl.col("sink").cast(pl.Utf8),
        pl.col("d").cast(pl.Utf8),
        pl.col("t").cast(pl.Utf8),
    )
    got = {(d, t): v for d, t, v in
            frame.select("d", "t", "value").iter_rows()}
    assert got == pt_values


class _MultiClassStubSource:
    """Surfaces a silent-default Map(period→time) under each of an
    arbitrary set of classes.  Used by the unit + connection union
    regression for ``p_pdt_varCost_process_from_source``.
    """

    def __init__(self,
                 per_class: dict[str, dict[tuple[str, str], float]],
                 parameter_name: str,
                 index_names: tuple[str, str] = ("x", "x"),
                 frame_index_cols: tuple[str, str] = ("x", "x_1")) -> None:
        self._per_class = per_class
        self._parameter_name = parameter_name
        self._index_names = index_names
        self._frame_index_cols = frame_index_cols

    def parameter_explicit(self, entity_class: str, parameter_name: str):
        if (entity_class not in self._per_class
                or parameter_name != self._parameter_name):
            raise KeyError((entity_class, parameter_name))
        pt = self._per_class[entity_class]
        ix0, ix1 = self._frame_index_cols
        keys = list(pt.keys())
        n = len(keys)
        return pl.DataFrame({
            "name": [f"stub_{entity_class}"] * n,
            ix0: [k[0] for k in keys],
            ix1: [k[1] for k in keys],
            "value": [pt[k] for k in keys],
        })

    def parameter(self, entity_class: str, parameter_name: str):
        return self.parameter_explicit(entity_class, parameter_name)

    def parameter_shape_info(self, entity_class: str, parameter_name: str):
        return list(self._index_names)

    def entities(self, entity_class: str):
        if entity_class in self._per_class:
            return pl.DataFrame({"name": [f"stub_{entity_class}"]})
        return pl.DataFrame({"name": []}, schema={"name": pl.Utf8})


def test_p_pdt_varCost_process_multi_class_concat(period_filter) -> None:
    """``unit`` and ``connection`` rows both surface in a single
    ``Param(("p", "d", "t"))``; no dups; values preserved per class.
    """
    import flextool.engine_polars._direct_params as dp
    unit_pt = {("p2024", "t00001"): 1.0, ("p2025", "t00002"): 2.0}
    conn_pt = {("p2024", "t00002"): 3.0, ("p2030", "t00003"): 4.0}
    stub = _MultiClassStubSource(
        per_class={"unit": unit_pt, "connection": conn_pt},
        parameter_name="other_operational_cost",
    )
    result = dp.p_pdt_varCost_process_from_source(
        stub, period_filter=period_filter)
    assert result is not None
    assert result.dims == ("p", "d", "t"), result.dims
    _assert_no_key_duplicates(result, ("p", "d", "t"))
    frame = _collect(result).with_columns(
        pl.col("p").cast(pl.Utf8),
        pl.col("d").cast(pl.Utf8),
        pl.col("t").cast(pl.Utf8),
    )
    # Both classes' stub-entity names must surface.
    names = set(frame.get_column("p").to_list())
    assert names == {"stub_unit", "stub_connection"}, names
    # Values per-class preserved.
    unit_rows = frame.filter(pl.col("p") == "stub_unit").sort("d", "t")
    assert dict(zip(
        list(zip(unit_rows.get_column("d").to_list(),
                  unit_rows.get_column("t").to_list())),
        unit_rows.get_column("value").to_list(),
    )) == unit_pt
    conn_rows = frame.filter(pl.col("p") == "stub_connection").sort("d", "t")
    assert dict(zip(
        list(zip(conn_rows.get_column("d").to_list(),
                  conn_rows.get_column("t").to_list())),
        conn_rows.get_column("value").to_list(),
    )) == conn_pt


def test_pdtReserve_reservation_silent_default(period_filter) -> None:
    """``reserve__upDown__group.reservation`` silent-default Map →
    Param ``(r, ud, g, d, t)``; no dups; values preserved.
    """
    import flextool.engine_polars._direct_params as dp
    pt_values = _pt_values()
    stub = _SilentDefaultMapPeriodTimeStub(
        entity_class="reserve__upDown__group",
        parameter_name="reservation",
        entity_cols=("reserve", "upDown", "group"),
        entity_values=("stub_reserve", "up", "stub_group"),
        period_time_values=pt_values,
    )
    result = dp.pdtReserve_upDown_group_reservation_from_source(
        stub, period_filter=period_filter)
    assert result is not None
    assert result.dims == ("r", "ud", "g", "d", "t"), result.dims
    _assert_no_key_duplicates(result, ("r", "ud", "g", "d", "t"))
    frame = _collect(result).with_columns(
        pl.col("r").cast(pl.Utf8),
        pl.col("ud").cast(pl.Utf8),
        pl.col("g").cast(pl.Utf8),
        pl.col("d").cast(pl.Utf8),
        pl.col("t").cast(pl.Utf8),
    )
    got = {(d, t): v for d, t, v in
            frame.select("d", "t", "value").iter_rows()}
    assert got == pt_values


def test_pdtReserve_reservation_zero_not_filtered(period_filter) -> None:
    """``pdtReserve_upDown_group_reservation_from_source`` uses
    ``filter_zero=False`` because the schema default is 0.0 and explicit
    zero rows carry meaning (legacy CSV path emitted zero rows).  Pin
    that semantic with an authored 0.0 row.
    """
    import flextool.engine_polars._direct_params as dp
    pt_values = {
        ("p2024", "t00001"): 0.0,   # explicit zero — MUST survive
        ("p2025", "t00002"): 1.5,
    }
    stub = _SilentDefaultMapPeriodTimeStub(
        entity_class="reserve__upDown__group",
        parameter_name="reservation",
        entity_cols=("reserve", "upDown", "group"),
        entity_values=("stub_reserve", "up", "stub_group"),
        period_time_values=pt_values,
    )
    result = dp.pdtReserve_upDown_group_reservation_from_source(
        stub, period_filter=period_filter)
    assert result is not None
    frame = _collect(result)
    # Both authored (d, t) tuples must be present, including the zero row.
    n = (frame
            .with_columns(pl.col("d").cast(pl.Utf8),
                            pl.col("t").cast(pl.Utf8))
            .filter((pl.col("d") == "p2024") & (pl.col("t") == "t00001"))
            .height)
    assert n == 1, (
        f"Explicit zero row was filtered out — pdtReserve uses "
        f"filter_zero=False but observed n={n} for (p2024, t00001)."
    )
