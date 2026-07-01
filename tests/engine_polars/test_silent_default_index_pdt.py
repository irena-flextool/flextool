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

A real-world migration audit on 2026-05-25 showed the
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
# A real-world migration audit (2026-05-25, kept locally) found
# ``unit__outputNode.other_operational_cost`` is authored as plain SCALAR
# floats across 99 rows and ``reserve__upDown__group.reservation`` as
# 2 SCALAR rows.  Pre-migration the helper's column-gate
# (``{"period", "t"}.issubset(cols)``) silently REJECTED SCALAR shapes
# because no ``period`` / ``t`` columns existed.  Post-migration the
# resolver lands on ``Shape.SCALAR`` and ``broadcast_to_period_time``
# returns a (p, sink) Param that polar_high broadcasts against the LHS
# (p, sink, d, t) at LP build time.
#
# This stub is inspired by a real-world shape but uses synthetic fixtures
# only (per CONTRIBUTING.md — no proprietary DB content in the repo).
# ---------------------------------------------------------------------------


class _ScalarRelationshipStub:
    """Surfaces a single SCALAR-authored row on an n-dim relationship
    class.  The frame has no period / t columns — the pre-migration
    helper's column-gate would silently REJECT this (observed
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
    """Inspired by a real-world authoring shape: SCALAR ``float`` on
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
    """Inspired by a real-world authoring shape: SCALAR ``float`` on
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
    """Only ``connection`` rows surface in the resulting
    ``Param(("p", "d", "t"))``; no dups; values preserved.

    v56 (commit 62e62e9b) dropped the dead ``("unit",
    "other_operational_cost")`` branch — no ``_specs.py`` writer ever
    emitted unit-level ``other_operational_cost`` to any CSV, so the
    pre-cleanup loop's unit pass was silently empty.  The class loop in
    :func:`p_pdt_varCost_process_from_source` is now single-element
    ``("connection",)``.  Even when a caller supplies a stub that would
    surface unit rows, the helper must ignore them — that's what this
    test pins.  User-facing ``other_operational_cost`` still lives on
    connection, unit__inputNode, and unit__outputNode via the dedicated
    ``p_pdt_varCost_source_from_source`` /
    ``p_pdt_varCost_sink_from_source`` helpers.
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
    # Only the connection stub-entity name must surface — unit branch
    # was dropped in v56 cleanup (commit 62e62e9b).
    names = set(frame.get_column("p").to_list())
    assert names == {"stub_connection"}, names
    # Values per-class preserved for the surviving connection branch.
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


# ---------------------------------------------------------------------------
# Silent-default Map regression — entity-unitsize cascade
#
# ``_entity_unitsize_lf`` computes ``unitsize = virtual_unitsize OR
# existing OR 1000`` per entity, taking the per-period MAX of a
# 1d_map(period) ``existing``.  Pre-fix it gated on
# ``if "period" in ex.columns`` — but spinedb_api authors the Map index
# with its silent default ``"x"``, so the gate MISSED and the ``else``
# branch fed every period row through unchanged.  The downstream
# ``unique(subset=["e"], keep="last")`` then kept the LAST period's
# value.  For a retiring unit whose map decays to 0 at expiry that
# collapsed the cascade input to 0 → unitsize defaulted to 1000 →
# ``existing_count = existing/unitsize`` became a spurious fraction
# (observed live: a 220 MW CCGT capped continuous online at 0.22 and
# could never commit under integer online).  The fix takes the
# per-entity MAX index-name-agnostically.
# ---------------------------------------------------------------------------


class _EntityExistingStub:
    """Surfaces a ``unit``-class ``existing`` parameter whose Map index
    column carries spinedb_api's silent-default name ``"x"`` (NOT
    ``"period"``).  ``virtual_unitsize`` is absent (KeyError), so the
    cascade must fall back to ``existing`` — via its per-period MAX.

    ``existing_by_unit`` maps ``unit -> [(period, value), ...]``.  A
    single ``(None, value)`` entry authors a SCALAR ``existing`` (no
    index column) to exercise the group-by no-op path.
    """

    def __init__(self, existing_by_unit: dict[str, list]) -> None:
        self._existing = existing_by_unit

    def entities(self, entity_class: str):
        if entity_class == "unit":
            return pl.DataFrame({"name": list(self._existing)})
        raise KeyError(entity_class)

    def parameter_explicit(self, entity_class: str, parameter_name: str):
        if entity_class == "unit" and parameter_name == "existing":
            names: list[str] = []
            idx: list[str] = []
            vals: list[float] = []
            scalar = False
            for unit, series in self._existing.items():
                for period, value in series:
                    names.append(unit)
                    if period is None:
                        scalar = True
                    else:
                        idx.append(period)
                    vals.append(value)
            if scalar:
                # SCALAR authoring — no index column at all.
                return pl.DataFrame({"name": names, "value": vals})
            # Map(period) authored with the silent-default ``"x"`` index.
            return pl.DataFrame({"name": names, "x": idx, "value": vals})
        raise KeyError((entity_class, parameter_name))

    def parameter(self, entity_class: str, parameter_name: str):
        return self.parameter_explicit(entity_class, parameter_name)


def _unitsize_map(stub) -> dict[str, float]:
    from flextool.engine_polars._derived_params import _entity_unitsize_lf
    us = _entity_unitsize_lf(stub).collect()
    return {
        e: v for e, v in zip(
            us.get_column("e").cast(pl.Utf8).to_list(),
            us.get_column("us").to_list(),
        )
    }


def test_entity_unitsize_retiring_silent_default_map_uses_nameplate_max():
    """A retiring unit whose silent-default (``"x"``-indexed) ``existing``
    Map decays to 0 must take its per-period MAX (nameplate), NOT the
    last period's 0 → NOT the 1000 fallback."""
    from flextool.engine_polars._derived_params import UNITSIZE_DEFAULT

    stub = _EntityExistingStub({
        # Retires: nameplate 220 early, 0 from p2027 onward (last = 0).
        "retiring_ccgt": [("p2024", 220.0), ("p2025", 220.0),
                          ("p2026", 220.0), ("p2027", 0.0),
                          ("p2050", 0.0)],
        # Never retires: constant 60 (last period non-zero — unaffected
        # by the bug, pins that the fix is byte-identical here).
        "steady_st": [("p2024", 60.0), ("p2050", 60.0)],
    })
    got = _unitsize_map(stub)
    assert got["retiring_ccgt"] == 220.0, (
        "retiring unit's silent-default existing Map collapsed to the "
        f"last-period 0 / {UNITSIZE_DEFAULT} default: {got}"
    )
    assert got["steady_st"] == 60.0, got


def test_entity_unitsize_scalar_existing_preserved():
    """SCALAR ``existing`` (no index column) must still surface its value
    — the per-entity group-by MAX is a no-op on a single row."""
    stub = _EntityExistingStub({"scalar_unit": [(None, 150.0)]})
    got = _unitsize_map(stub)
    assert got["scalar_unit"] == 150.0, got


def test_entity_unitsize_no_existing_falls_back_to_default():
    """A unit with NO ``existing`` and NO ``virtual_unitsize`` keeps the
    canonical 1000 fallback (guards against the fix over-reaching)."""
    from flextool.engine_polars._derived_params import UNITSIZE_DEFAULT

    stub = _EntityExistingStub({"bare_unit": []})
    got = _unitsize_map(stub)
    assert got["bare_unit"] == UNITSIZE_DEFAULT, got
