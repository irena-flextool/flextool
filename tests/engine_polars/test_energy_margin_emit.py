"""Unit tests for the energy_margin inflow emitter.

Exercise :func:`emit_energy_margin_inflow` directly against an in-memory
:class:`FlexDataProvider` — no DB, no full solve.  Pins the contract facets
under FlexTool's sign convention (DEMAND is NEGATIVE inflow, SUPPLY is
POSITIVE inflow):

* invest solve + ``energy_margin_method='inflow_multiplier'`` ⇒ that node's NEGATIVE
  (demand) ``pdtNodeInflow`` rows are multiplied (more negative = more
  demand), its positive (supply) and zero rows untouched, other nodes
  untouched;
* dispatch solve (``invest_periods.get(solve_name)`` falsy) ⇒ the
  ``pdtNodeInflow`` frame is byte-identical (early return);
* method ``'none'`` OR effective margin ``1.0`` ⇒ byte-identical (early
  return);
* the re-rendered value cells match the pipeline's ``repr(float)`` render
  (invariant #5) — asserted as exact strings.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import polars as pl

from flextool.engine_polars._emit_energy_margin import emit_energy_margin_inflow
from flextool.engine_polars._flex_data_provider import FlexDataProvider

INPUT_DIR = Path("input")
SOLVE_DATA_DIR = Path("solve_data")
PDT_KEY = "solve_data/pdtNodeInflow"


def _utf8(cols: dict[str, list[str]]) -> pl.DataFrame:
    return pl.DataFrame(cols, schema={c: pl.Utf8 for c in cols})


def _inflow_frame() -> pl.DataFrame:
    # nodeA: two negative (demand) rows + one positive (supply) row + one zero
    # row; nodeB: one negative (demand) row on another node.
    return _utf8({
        "node": ["nodeA", "nodeA", "nodeA", "nodeA", "nodeB"],
        "period": ["p", "p", "p", "p", "p"],
        "time": ["t1", "t2", "t3", "t4", "t1"],
        "value": ["-589.0", "-42.0", "120.0", "0.0", "-80.0"],
    })


def _provider(
    *,
    method: str | None,
    margin: float | None,
) -> FlexDataProvider:
    prov = FlexDataProvider()
    prov.put(PDT_KEY, _inflow_frame())
    if method is not None:
        prov.put(
            "input/node__energy_margin_method",
            _utf8({"node": ["nodeA"], "energy_margin_method": [method]}),
        )
    if margin is not None:
        prov.put(
            "input/energy_margin_multiplier",
            _utf8({"node": ["nodeA"], "energy_margin_multiplier": [repr(margin)]}),
        )
    return prov


def _state(*, invest: bool) -> SimpleNamespace:
    invest_periods = {"S": [("p",)]} if invest else {}
    return SimpleNamespace(solve=SimpleNamespace(invest_periods=invest_periods))


def _run(prov: FlexDataProvider, *, invest: bool) -> None:
    emit_energy_margin_inflow(
        _state(invest=invest), "S", INPUT_DIR, SOLVE_DATA_DIR, provider=prov,
    )


def test_invest_manual_scales_negative_demand_rows_only() -> None:
    prov = _provider(method="inflow_multiplier", margin=1.5)
    _run(prov, invest=True)
    out = prov.get(PDT_KEY)
    # Row order + non-value columns preserved.
    assert out["node"].to_list() == ["nodeA", "nodeA", "nodeA", "nodeA", "nodeB"]
    assert out["time"].to_list() == ["t1", "t2", "t3", "t4", "t1"]
    # nodeA NEGATIVE (demand) rows scaled ×1.5 (more negative = more demand),
    # rendered via repr(float); positive (supply) row, zero row, and nodeB
    # left byte-identical.
    assert out["value"].to_list() == [
        repr(-589.0 * 1.5),  # -883.5  (demand scaled)
        repr(-42.0 * 1.5),   # -63.0   (demand scaled)
        "120.0",             # positive supply → untouched
        "0.0",               # zero → untouched
        "-80.0",             # other node → untouched
    ]


def test_dispatch_solve_is_byte_identical() -> None:
    prov = _provider(method="inflow_multiplier", margin=1.5)
    before = prov.get(PDT_KEY)
    _run(prov, invest=False)
    after = prov.get(PDT_KEY)
    assert after.equals(before)
    assert after.equals(_inflow_frame())


def test_method_none_is_byte_identical() -> None:
    prov = _provider(method="none", margin=1.5)
    _run(prov, invest=True)
    assert prov.get(PDT_KEY).equals(_inflow_frame())


def test_margin_one_is_byte_identical() -> None:
    prov = _provider(method="inflow_multiplier", margin=1.0)
    _run(prov, invest=True)
    assert prov.get(PDT_KEY).equals(_inflow_frame())


def test_missing_margin_value_defaults_to_one() -> None:
    # method=inflow_multiplier but no energy_margin_multiplier value → effective margin 1.0 → no-op.
    prov = _provider(method="inflow_multiplier", margin=None)
    _run(prov, invest=True)
    assert prov.get(PDT_KEY).equals(_inflow_frame())


def test_rendered_values_match_repr_render() -> None:
    # A margin whose product needs a non-trivial repr (round-trip exactness).
    prov = _provider(method="inflow_multiplier", margin=1.07)
    _run(prov, invest=True)
    out = prov.get(PDT_KEY)
    # Negative (demand) rows scaled and rendered exactly as repr(float).
    assert out["value"][0] == repr(-589.0 * 1.07)
    assert out["value"][1] == repr(-42.0 * 1.07)
    # Positive / zero / other-node rows keep their original string verbatim.
    assert out["value"][2] == "120.0"
    assert out["value"][3] == "0.0"
    assert out["value"][4] == "-80.0"
    # Parse-back sanity at tight tolerance.
    assert abs(float(out["value"][0]) - (-630.23)) < 1e-12 * 630.23
