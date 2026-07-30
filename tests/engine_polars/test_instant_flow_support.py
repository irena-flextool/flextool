"""Regression — flowGroup ``min_instant_flow`` / ``max_instant_flow``
constraint support across every authored shape.

Background (the bug this pins):

The instant-flow obligation's constraint *support* used to be built by a
separate raw-source projection (``_projection_params.gdt_*InstantFlow``)
that detected the index axis by column name:

* ``if "period" in df.columns`` — silently returned an EMPTY support for
  a period map whose Map ``index_name`` carried spinedb_api's
  silent-default ``"x"`` label (Spine Toolbox cannot cleanly distinguish
  period from time maps, so authors routinely leave it ``"x"``), and for
  constants and time maps.  An empty support → the ``>=`` / ``<=``
  constraint was never emitted → **the obligation was ignored**.
* ``.select("g", "d", "t")`` — once the column *was* named ``"period"``,
  it then **crashed** on a pure period map, which has no ``t`` column.

The fix derives the ``(g, d, t)`` support directly from the resolved
``pdt_*_instant_flow`` cap (which routes through
``_param_shapes.resolve_param_shape`` and therefore handles ``"x"`` /
constant / period / time / period+time uniformly), broadcasting the
missing axes against the active ``(d, t)`` grid — exactly how the
cumulative-flow siblings derive their ``over``.

These tests lock both halves: the pure support-broadcast helper for every
cap shape, and the end-to-end cap resolution for an ``"x"``-indexed period
map (with a zero later period, as in the field report).
"""
from __future__ import annotations

import polars as pl

from flextool.engine_polars._cumulative_invest import _instant_flow_support


# ---------------------------------------------------------------------------
# _instant_flow_support — every cap shape broadcasts to full (g, d, t)
# ---------------------------------------------------------------------------

def _dt() -> pl.DataFrame:
    """Active (d, t) grid: two periods × two timesteps."""
    return pl.DataFrame({
        "d": ["p1", "p1", "p2", "p2"],
        "t": ["t1", "t2", "t1", "t2"],
    })


def _rows(over: pl.DataFrame) -> set[tuple[str, str, str]]:
    return set(over.select("g", "d", "t").iter_rows())


def test_support_from_scalar_cap_broadcasts_whole_grid() -> None:
    """SCALAR cap ``(g,)`` — the constant case the old projection
    silently dropped.  Must broadcast over the entire active grid."""
    cap = pl.DataFrame({"g": ["fg"], "value": [500.0]})
    over = _instant_flow_support(cap, _dt())
    assert _rows(over) == {
        ("fg", "p1", "t1"), ("fg", "p1", "t2"),
        ("fg", "p2", "t1"), ("fg", "p2", "t2"),
    }


def test_support_from_period_cap_broadcasts_over_time() -> None:
    """MAP_PERIOD cap ``(g, d)`` — the case that used to CRASH on the
    missing ``t`` column.  Must broadcast each active period over time,
    and only over periods present in the cap."""
    cap = pl.DataFrame({"g": ["fg"], "d": ["p1"], "value": [500.0]})
    over = _instant_flow_support(cap, _dt())
    assert _rows(over) == {("fg", "p1", "t1"), ("fg", "p1", "t2")}, (
        "period-map support must cover only the cap's period, over all "
        "active timesteps"
    )


def test_support_from_time_cap_broadcasts_over_periods() -> None:
    """MAP_TIME cap ``(g, t)`` — broadcast each active period."""
    cap = pl.DataFrame({"g": ["fg"], "t": ["t1"], "value": [500.0]})
    over = _instant_flow_support(cap, _dt())
    assert _rows(over) == {("fg", "p1", "t1"), ("fg", "p2", "t1")}


def test_support_from_period_time_cap_is_identity() -> None:
    """MAP_PERIOD_TIME cap ``(g, d, t)`` — already fully keyed."""
    cap = pl.DataFrame({
        "g": ["fg", "fg"], "d": ["p1", "p2"], "t": ["t1", "t2"],
        "value": [500.0, 600.0],
    })
    over = _instant_flow_support(cap, _dt())
    assert _rows(over) == {("fg", "p1", "t1"), ("fg", "p2", "t2")}


def test_support_multiple_groups() -> None:
    """Two flowGroups, period maps on different periods — supports must
    not cross-contaminate."""
    cap = pl.DataFrame({
        "g": ["fgA", "fgB"], "d": ["p1", "p2"], "value": [1.0, 2.0],
    })
    over = _instant_flow_support(cap, _dt())
    assert _rows(over) == {
        ("fgA", "p1", "t1"), ("fgA", "p1", "t2"),
        ("fgB", "p2", "t1"), ("fgB", "p2", "t2"),
    }


# ---------------------------------------------------------------------------
# End-to-end cap resolution — flowGroup.min_instant_flow authored as a
# 1d Map(period) with spinedb_api's silent-default "x" index_name, with a
# zero later period (the field-report shape).
# ---------------------------------------------------------------------------

class _PeriodMapStub:
    """Surfaces ``flowGroup.<param>`` as a 1d Map(period→value) authored
    with a chosen ``index_name`` (``"x"`` = spinedb_api silent default,
    or the canonical ``"period"``).  Mirrors the SpineDbReader's flat
    frame: one entity column ``name`` plus the index column named by
    ``index_name``.
    """

    def __init__(self, parameter_name: str, group: str,
                 period_values: dict[str, float],
                 index_name: str = "x") -> None:
        self._pn = parameter_name
        self._group = group
        self._pv = period_values
        self._index_name = index_name

    def parameter_explicit(self, entity_class: str, parameter_name: str):
        if entity_class != "flowGroup" or parameter_name != self._pn:
            raise KeyError((entity_class, parameter_name))
        keys = list(self._pv.keys())
        n = len(keys)
        return pl.DataFrame({
            "name": [self._group] * n,
            self._index_name: keys,
            "value": [self._pv[k] for k in keys],
        })

    def parameter(self, entity_class: str, parameter_name: str):
        return self.parameter_explicit(entity_class, parameter_name)

    def parameter_shape_info(self, entity_class: str, parameter_name: str):
        return [self._index_name]

    def entities(self, entity_class: str):
        if entity_class == "flowGroup":
            return pl.DataFrame({"name": [self._group]})
        return pl.DataFrame({"name": []}, schema={"name": pl.Utf8})


def _collect(param) -> pl.DataFrame:
    frame = param.frame
    return frame.collect() if hasattr(frame, "collect") else frame


def _period_filter() -> pl.DataFrame:
    """Two periods, each with two timesteps."""
    return pl.DataFrame({
        "d": ["p2025", "p2025", "p2030", "p2030"],
        "t": ["t01", "t02", "t01", "t02"],
    })


def _resolve_and_support(stub) -> pl.DataFrame:
    import flextool.engine_polars._direct_params as dp
    pf = _period_filter()
    cap = dp.pdt_min_instant_flow_from_source(stub, period_filter=pf)
    assert cap is not None, (
        "min_instant_flow cap resolved to None — the 'x'-indexed period "
        "map was dropped by the resolver."
    )
    return _instant_flow_support(_collect(cap), pf.select("d", "t").unique())


def test_x_indexed_period_map_resolves_and_emits_support() -> None:
    """The field-report case: ``min_instant_flow`` as a period map with
    ``index_name="x"``, non-zero in the first period and zero in the
    later one.  The non-zero period must yield support over its
    timesteps; the explicit-zero period must drop out (a minimum of 0 is
    no obligation)."""
    stub = _PeriodMapStub(
        "min_instant_flow", "fg1",
        {"p2025": 500.0, "p2030": 0.0}, index_name="x",
    )
    over = _resolve_and_support(stub).with_columns(
        pl.col("g").cast(pl.Utf8), pl.col("d").cast(pl.Utf8),
        pl.col("t").cast(pl.Utf8),
    )
    assert _rows(over) == {("fg1", "p2025", "t01"), ("fg1", "p2025", "t02")}, (
        f"expected support only over the non-zero period p2025; got "
        f"{_rows(over)}"
    )


def test_period_indexed_period_map_resolves_and_emits_support() -> None:
    """Same authoring but with the canonical ``index_name="period"`` —
    this is the label that used to make the old projection crash on the
    missing ``t`` column.  Must now resolve cleanly."""
    stub = _PeriodMapStub(
        "min_instant_flow", "fg1",
        {"p2025": 500.0, "p2030": 0.0}, index_name="period",
    )
    over = _resolve_and_support(stub).with_columns(
        pl.col("g").cast(pl.Utf8), pl.col("d").cast(pl.Utf8),
        pl.col("t").cast(pl.Utf8),
    )
    assert _rows(over) == {("fg1", "p2025", "t01"), ("fg1", "p2025", "t02")}


# ---------------------------------------------------------------------------
# Mixed-authoring regression — one flowGroup authors a 1d Map(period) floor
# while a SIBLING flowGroup authors a 2d Map(period, time) floor on the same
# ``min_instant_flow`` parameter.
#
# Reproduces the reported correctness bug on a real grid model: a baseline
# solve floors two thermal units (=150 / =60) as period maps;
# adding a period-time Map floor on a third flowGroup ("All thermal units")
# flipped the WHOLE parameter's resolved shape to MAP_PERIOD_TIME (Spine's
# ``parameter_shape_info`` reports the deepest row; ``_unroll_rows``
# discovers index columns from the widest row).  The period-map rows then
# arrived with a NULL ``t`` and the old ``broadcast_to_period_time``
# MAP_PERIOD_TIME branch inner-joined them away on (d, t) — silently
# DELETING the pre-existing floors.  The floored model became CHEAPER than
# the un-floored baseline (mathematically impossible for a floor), because
# generation dropped two of the three obligations.
#
# ``broadcast_to_period_time`` now carries the same null-index mixed-
# authoring guard the MAP_PERIOD / MAP_TIME branches have: period-only rows
# (null t) broadcast across every active timestep; time-only rows (null d)
# broadcast across every active period; fully-keyed rows keep the (d, t)
# inner-join.  All three floors must survive.
# ---------------------------------------------------------------------------


class _MixedShapeFlowGroupStub:
    """Surfaces ``flowGroup.min_instant_flow`` where some flowGroups are
    authored as a 1d Map(period) and one as a 2d Map(period, time),
    mirroring the flat frame SpineDbReader emits for a mixed-depth
    parameter under one scenario:

    * ``parameter_shape_info`` returns the DEEPEST row's raw labels
      (``["x", "x"]`` — spinedb_api's silent default on both levels).
    * ``parameter_explicit`` returns a flat frame with two index columns
      (``x`` = period, ``x_2`` = time); the period-map rows carry a NULL
      ``x_2`` because the widest (2d) row fixed the column set.
    """

    def __init__(self,
                 period_maps: dict[str, dict[str, float]],
                 period_time_maps: dict[str, dict[tuple[str, str], float]],
                 frame_index_cols: tuple[str, str] = ("x", "x_2")) -> None:
        self._period_maps = period_maps
        self._pt_maps = period_time_maps
        self._ix = frame_index_cols

    def parameter_explicit(self, entity_class: str, parameter_name: str):
        if entity_class != "flowGroup" or parameter_name != "min_instant_flow":
            raise KeyError((entity_class, parameter_name))
        ix0, ix1 = self._ix
        names: list[str] = []
        col0: list[str] = []
        col1: list[str | None] = []
        vals: list[float] = []
        # 1d period maps → NULL second index (the mixed-shape signature).
        for g, pv in self._period_maps.items():
            for d, v in pv.items():
                names.append(g)
                col0.append(d)
                col1.append(None)
                vals.append(v)
        # 2d period-time maps → both indices populated.
        for g, ptv in self._pt_maps.items():
            for (d, t), v in ptv.items():
                names.append(g)
                col0.append(d)
                col1.append(t)
                vals.append(v)
        return pl.DataFrame(
            {"name": names, ix0: col0, ix1: col1, "value": vals},
            schema={"name": pl.Utf8, ix0: pl.Utf8, ix1: pl.Utf8,
                    "value": pl.Float64},
        )

    def parameter(self, entity_class: str, parameter_name: str):
        return self.parameter_explicit(entity_class, parameter_name)

    def parameter_shape_info(self, entity_class: str, parameter_name: str):
        # Deepest row is the 2d Map — two silent-default "x" levels.
        return ["x", "x"]

    def entities(self, entity_class: str):
        if entity_class == "flowGroup":
            groups = list(self._period_maps) + list(self._pt_maps)
            return pl.DataFrame({"name": groups})
        return pl.DataFrame({"name": []}, schema={"name": pl.Utf8})


def _mixed_min_instant_flow_param():
    """Resolve the mixed-authoring ``pdt_min_instant_flow`` cap over a
    2-period × 2-timestep grid and return (Param, dt_grid)."""
    import flextool.engine_polars._direct_params as dp
    pf = _period_filter()
    stub = _MixedShapeFlowGroupStub(
        period_maps={
            # baseline period-scalar floors (as 1d period maps)
            "Thermal Plant A": {"p2025": 150.0, "p2030": 150.0},
            "Thermal Plant B": {"p2025": 60.0, "p2030": 60.0},
        },
        period_time_maps={
            # the added period-time floor that flips the frame shape
            "All thermal units": {
                ("p2025", "t01"): 10.0, ("p2025", "t02"): 10.0,
                ("p2030", "t01"): 10.0, ("p2030", "t02"): 10.0,
            },
        },
    )
    cap = dp.pdt_min_instant_flow_from_source(stub, period_filter=pf)
    assert cap is not None, "mixed-authoring cap resolved to None"
    return cap, pf


def test_mixed_period_and_period_time_floors_all_survive() -> None:
    """The load-bearing assertion: the resolved ``pdt_min_instant_flow``
    Param must retain BOTH the period-map floors AND the period-time
    floor over the full active (d, t) grid.  Pre-fix, the two period-map
    floors were inner-joined away (NULL ``t``) — reproducing the reported
    lower-than-baseline optimum."""
    cap, _ = _mixed_min_instant_flow_param()
    frame = _collect(cap).with_columns(
        pl.col("g").cast(pl.Utf8), pl.col("d").cast(pl.Utf8),
        pl.col("t").cast(pl.Utf8),
    )
    got = {(g, d, t): v for g, d, t, v in
           frame.select("g", "d", "t", "value").iter_rows()}

    full_grid = {("p2025", "t01"), ("p2025", "t02"),
                 ("p2030", "t01"), ("p2030", "t02")}
    # Plant-A period floor (150) broadcast across every active timestep.
    for d, t in full_grid:
        assert got.get(("Thermal Plant A", d, t)) == 150.0, (
            f"Plant-A floor dropped/altered at ({d}, {t}); got {got}"
        )
        assert got.get(("Thermal Plant B", d, t)) == 60.0, (
            f"Plant-B floor dropped/altered at ({d}, {t}); got {got}"
        )
        assert got.get(("All thermal units", d, t)) == 10.0, (
            f"period-time floor missing at ({d}, {t}); got {got}"
        )
    # No spurious extra rows.
    assert len(got) == 3 * len(full_grid), got


def test_mixed_floors_emit_constraint_support_for_all_groups() -> None:
    """Finer-grained localisation: the emitted ``minInstant_flow``
    constraint *support* (``_instant_flow_support`` over the resolved
    cap) must cover all three flowGroups across the full grid — i.e. the
    obligation is generated for the period-map floors, not silently
    skipped."""
    cap, pf = _mixed_min_instant_flow_param()
    over = _instant_flow_support(
        _collect(cap), pf.select("d", "t").unique()
    ).with_columns(
        pl.col("g").cast(pl.Utf8), pl.col("d").cast(pl.Utf8),
        pl.col("t").cast(pl.Utf8),
    )
    groups = {g for g, _, _ in _rows(over)}
    assert groups == {"Thermal Plant A", "Thermal Plant B",
                      "All thermal units"}, (
        f"constraint support lost a flowGroup: {groups}"
    )
    assert len(_rows(over)) == 3 * 4, _rows(over)
