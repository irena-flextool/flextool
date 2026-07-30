"""Unit tests for the additive energy_margin_adder inflow emitter.

Exercise :func:`emit_energy_margin_adder` directly against an in-memory
:class:`FlexDataProvider` — no DB, no full solve.  Pins the contract
under FlexTool's sign convention (DEMAND is NEGATIVE inflow, SUPPLY is
POSITIVE inflow, so ADDING demand SUBTRACTS from the inflow value):

* invest solve + ``energy_margin_method='inflow_adder'`` ⇒ the node's
  existing ``pdtNodeInflow`` rows are DEEPENED by ``value - adder`` at
  every invest ``(d, t)``;
* a node that had NO ``pdtNodeInflow`` row gets negative demand rows
  CREATED (value ``-adder``) at every invest ``(d, t)``;
* dispatch solve (``invest_periods.get(solve_name)`` falsy) ⇒ the
  ``pdtNodeInflow`` frame is byte-identical (early return);
* no ``inflow_adder`` node OR a zero adder ⇒ byte-identical (early return);
* a period Map adder places per-period values over that period's (d, t);
* the re-rendered / created value cells match the pipeline's
  ``repr(float)`` render (invariant #5) — asserted as exact strings.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import polars as pl

from flextool.engine_polars._emit_energy_margin_adder import (
    emit_energy_margin_adder,
)
from flextool.engine_polars._flex_data_provider import FlexDataProvider

INPUT_DIR = Path("input")
SOLVE_DATA_DIR = Path("solve_data")
PDT_KEY = "solve_data/pdtNodeInflow"

# The invest solve's (d, t) grid — one period ``p``, four timesteps.
STEPS = [("p", "t1"), ("p", "t2"), ("p", "t3"), ("p", "t4")]


def _utf8(cols: dict[str, list[str]]) -> pl.DataFrame:
    return pl.DataFrame(cols, schema={c: pl.Utf8 for c in cols})


def _inflow_frame() -> pl.DataFrame:
    # nodeA: full demand series across the grid (all four timesteps);
    # nodeB: a single demand row.  nodeC has NO row (zero-inflow slack).
    return _utf8({
        "node": ["nodeA", "nodeA", "nodeA", "nodeA", "nodeB"],
        "period": ["p", "p", "p", "p", "p"],
        "time": ["t1", "t2", "t3", "t4", "t1"],
        "value": ["-589.0", "-42.0", "120.0", "0.0", "-80.0"],
    })


def _steps_frame() -> pl.DataFrame:
    return _utf8({
        "period": [d for d, _t in STEPS],
        "time": [t for _d, t in STEPS],
    })


def _provider(
    *,
    method_node: str | None,
    method: str | None,
    adder_frame: pl.DataFrame | None,
    inflow: pl.DataFrame | None = None,
) -> FlexDataProvider:
    prov = FlexDataProvider()
    prov.put(PDT_KEY, _inflow_frame() if inflow is None else inflow)
    prov.put("solve_data/steps_in_use", _steps_frame())
    if method is not None and method_node is not None:
        prov.put(
            "input/node__energy_margin_method",
            _utf8({"node": [method_node],
                   "energy_margin_method": [method]}),
        )
    if adder_frame is not None:
        prov.put("input/energy_margin_adder", adder_frame)
    return prov


def _scalar_adder(node: str, value: float) -> pl.DataFrame:
    return _utf8({"node": [node], "energy_margin_adder": [repr(value)]})


def _state(*, invest: bool) -> SimpleNamespace:
    invest_periods = {"S": [("p",)]} if invest else {}
    return SimpleNamespace(solve=SimpleNamespace(invest_periods=invest_periods))


def _run(prov: FlexDataProvider, *, invest: bool) -> None:
    emit_energy_margin_adder(
        _state(invest=invest), "S", INPUT_DIR, SOLVE_DATA_DIR, provider=prov,
    )


# ---------------------------------------------------------------------------
# (a) default / no-adder → provider untouched (byte-parity early return)
# ---------------------------------------------------------------------------

def test_no_adder_frame_is_byte_identical() -> None:
    prov = _provider(
        method_node="nodeA", method="inflow_adder", adder_frame=None,
    )
    _run(prov, invest=True)
    assert prov.get(PDT_KEY).equals(_inflow_frame())


def test_method_none_is_byte_identical() -> None:
    prov = _provider(
        method_node="nodeA", method="none",
        adder_frame=_scalar_adder("nodeA", 100.0),
    )
    _run(prov, invest=True)
    assert prov.get(PDT_KEY).equals(_inflow_frame())


def test_zero_adder_is_byte_identical() -> None:
    prov = _provider(
        method_node="nodeA", method="inflow_adder",
        adder_frame=_scalar_adder("nodeA", 0.0),
    )
    _run(prov, invest=True)
    assert prov.get(PDT_KEY).equals(_inflow_frame())


# ---------------------------------------------------------------------------
# (d) dispatch (non-invest) solve is untouched
# ---------------------------------------------------------------------------

def test_dispatch_solve_is_byte_identical() -> None:
    prov = _provider(
        method_node="nodeA", method="inflow_adder",
        adder_frame=_scalar_adder("nodeA", 100.0),
    )
    _run(prov, invest=False)
    assert prov.get(PDT_KEY).equals(_inflow_frame())


# ---------------------------------------------------------------------------
# (b) invest + inflow_adder DEEPENS existing rows by the subtracted amount
# ---------------------------------------------------------------------------

def test_invest_scalar_adder_deepens_existing_rows_only_for_target() -> None:
    ADDER = 100.0
    prov = _provider(
        method_node="nodeA", method="inflow_adder",
        adder_frame=_scalar_adder("nodeA", ADDER),
    )
    _run(prov, invest=True)
    out = prov.get(PDT_KEY)

    # Original rows keep their order + height (nodeA rows deepened, nodeB
    # verbatim); no new rows for nodeA since it covers the whole grid.
    assert out.height == _inflow_frame().height
    nodeA = out.filter(pl.col("node") == "nodeA").sort("time")
    # value_new = value_old - adder (adding demand ⇒ MORE negative), and
    # applies to EVERY existing row of the target (supply rows included).
    assert nodeA["value"].to_list() == [
        repr(-589.0 - ADDER),  # -689.0
        repr(-42.0 - ADDER),   # -142.0
        repr(120.0 - ADDER),   # 20.0
        repr(0.0 - ADDER),     # -100.0
    ]
    # nodeB (not an inflow_adder node) is byte-identical.
    nodeB = out.filter(pl.col("node") == "nodeB")
    assert nodeB["value"].to_list() == ["-80.0"]


# ---------------------------------------------------------------------------
# (c) invest + inflow_adder CREATES rows on a node that had none
# ---------------------------------------------------------------------------

def test_invest_scalar_adder_creates_rows_on_zero_inflow_node() -> None:
    ADDER = 250.0
    prov = _provider(
        method_node="nodeC", method="inflow_adder",
        adder_frame=_scalar_adder("nodeC", ADDER),
    )
    _run(prov, invest=True)
    out = prov.get(PDT_KEY)

    # The original five rows are untouched (nodeA / nodeB byte-identical).
    orig = _inflow_frame()
    keep = out.filter(pl.col("node") != "nodeC")
    assert keep.equals(orig)

    # nodeC now carries a CREATED negative-demand row at every invest (d, t).
    created = out.filter(pl.col("node") == "nodeC").sort("time")
    assert created["time"].to_list() == ["t1", "t2", "t3", "t4"]
    assert created["period"].to_list() == ["p", "p", "p", "p"]
    # value = -adder (adding demand ⇒ negative inflow), repr-rendered.
    assert created["value"].to_list() == [repr(-ADDER)] * 4


# ---------------------------------------------------------------------------
# invest + inflow_adder DEEPENS existing rows AND CREATES missing ones
# ---------------------------------------------------------------------------

def test_invest_scalar_adder_deepens_and_creates_for_partial_node() -> None:
    ADDER = 30.0
    # nodeB has only one row (t1) but the grid has four timesteps.
    prov = _provider(
        method_node="nodeB", method="inflow_adder",
        adder_frame=_scalar_adder("nodeB", ADDER),
    )
    _run(prov, invest=True)
    out = prov.get(PDT_KEY)
    nodeB = out.filter(pl.col("node") == "nodeB").sort("time")
    # t1 deepened (-80 - 30 = -110); t2/t3/t4 created (-30).
    assert nodeB["time"].to_list() == ["t1", "t2", "t3", "t4"]
    assert nodeB["value"].to_list() == [
        repr(-80.0 - ADDER), repr(-ADDER), repr(-ADDER), repr(-ADDER),
    ]
    # nodeA untouched.
    nodeA = out.filter(pl.col("node") == "nodeA").sort("time")
    assert nodeA["value"].to_list() == ["-589.0", "-42.0", "120.0", "0.0"]


# ---------------------------------------------------------------------------
# period-Map adder: per-period values placed over that period's (d, t)
# ---------------------------------------------------------------------------

def test_invest_period_map_adder_places_per_period_values() -> None:
    # Two-period grid so the period axis is meaningful.
    steps = [("p1", "t1"), ("p1", "t2"), ("p2", "t1"), ("p2", "t2")]
    inflow = _utf8({
        "node": ["nodeA", "nodeA", "nodeA", "nodeA"],
        "period": ["p1", "p1", "p2", "p2"],
        "time": ["t1", "t2", "t1", "t2"],
        "value": ["-10.0", "-20.0", "-30.0", "-40.0"],
    })
    prov = FlexDataProvider()
    prov.put(PDT_KEY, inflow)
    prov.put(
        "solve_data/steps_in_use",
        _utf8({"period": [d for d, _t in steps],
               "time": [t for _d, t in steps]}),
    )
    prov.put(
        "input/node__energy_margin_method",
        _utf8({"node": ["nodeA"], "energy_margin_method": ["inflow_adder"]}),
    )
    # period Map: p1 → 5 MWh, p2 → 7 MWh (3-col frame, ``period`` index).
    prov.put(
        "input/energy_margin_adder",
        _utf8({
            "node": ["nodeA", "nodeA"],
            "period": ["p1", "p2"],
            "energy_margin_adder": [repr(5.0), repr(7.0)],
        }),
    )
    _run_prov = SimpleNamespace(
        solve=SimpleNamespace(invest_periods={"S": [("p1",), ("p2",)]}),
    )
    emit_energy_margin_adder(
        _run_prov, "S", INPUT_DIR, SOLVE_DATA_DIR, provider=prov,
    )
    out = prov.get(PDT_KEY).sort(["period", "time"])
    # p1 rows deepened by 5, p2 rows by 7.
    assert out["value"].to_list() == [
        repr(-10.0 - 5.0), repr(-20.0 - 5.0),   # p1
        repr(-30.0 - 7.0), repr(-40.0 - 7.0),   # p2
    ]


# ---------------------------------------------------------------------------
# invariant #5: created / deepened cells render exactly as repr(float)
# ---------------------------------------------------------------------------

def test_rendered_values_match_repr_render() -> None:
    ADDER = 1.07
    prov = _provider(
        method_node="nodeA", method="inflow_adder",
        adder_frame=_scalar_adder("nodeA", ADDER),
    )
    _run(prov, invest=True)
    out = prov.get(PDT_KEY)
    nodeA = out.filter(pl.col("node") == "nodeA").sort("time")
    assert nodeA["value"][0] == repr(-589.0 - ADDER)
    assert nodeA["value"][1] == repr(-42.0 - ADDER)
    # Parse-back sanity at tight tolerance.
    assert abs(float(nodeA["value"][0]) - (-590.07)) < 1e-12 * 590.07
