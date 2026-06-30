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
