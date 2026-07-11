"""Unit tests for the energy_margin inflow emitter.

Exercise :func:`emit_energy_margin_inflow` directly against an in-memory
:class:`FlexDataProvider` — no DB, no full solve.  Pins the four contract
facets:

* invest solve + ``energy_margin_method='manual'`` ⇒ that node's POSITIVE
  ``pdtNodeInflow`` rows are multiplied, its negative row untouched, other
  nodes untouched;
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
    # nodeA: two positive rows + one negative row; nodeB: one positive row.
    return _utf8({
        "node": ["nodeA", "nodeA", "nodeA", "nodeB"],
        "period": ["p", "p", "p", "p"],
        "time": ["t1", "t2", "t3", "t1"],
        "value": ["100.0", "33.0", "-30.0", "80.0"],
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
            "input/energy_margin",
            _utf8({"node": ["nodeA"], "energy_margin": [repr(margin)]}),
        )
    return prov


def _state(*, invest: bool) -> SimpleNamespace:
    invest_periods = {"S": [("p",)]} if invest else {}
    return SimpleNamespace(solve=SimpleNamespace(invest_periods=invest_periods))


def _run(prov: FlexDataProvider, *, invest: bool) -> None:
    emit_energy_margin_inflow(
        _state(invest=invest), "S", INPUT_DIR, SOLVE_DATA_DIR, provider=prov,
    )


def test_invest_manual_scales_positive_rows_only() -> None:
    prov = _provider(method="manual", margin=1.5)
    _run(prov, invest=True)
    out = prov.get(PDT_KEY)
    # Row order + non-value columns preserved.
    assert out["node"].to_list() == ["nodeA", "nodeA", "nodeA", "nodeB"]
    assert out["time"].to_list() == ["t1", "t2", "t3", "t1"]
    # nodeA positive rows scaled ×1.5, rendered via repr(float); negative
    # row and nodeB left byte-identical.
    assert out["value"].to_list() == [
        repr(100.0 * 1.5),   # 150.0
        repr(33.0 * 1.5),    # 49.5
        "-30.0",             # negative → untouched
        "80.0",              # other node → untouched
    ]


def test_dispatch_solve_is_byte_identical() -> None:
    prov = _provider(method="manual", margin=1.5)
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
    prov = _provider(method="manual", margin=1.0)
    _run(prov, invest=True)
    assert prov.get(PDT_KEY).equals(_inflow_frame())


def test_missing_margin_value_defaults_to_one() -> None:
    # method=manual but no energy_margin value → effective margin 1.0 → no-op.
    prov = _provider(method="manual", margin=None)
    _run(prov, invest=True)
    assert prov.get(PDT_KEY).equals(_inflow_frame())


def test_rendered_values_match_repr_render() -> None:
    # A margin whose product needs a non-trivial repr (round-trip exactness).
    prov = _provider(method="manual", margin=1.07)
    _run(prov, invest=True)
    out = prov.get(PDT_KEY)
    assert out["value"][0] == repr(100.0 * 1.07)
    assert out["value"][1] == repr(33.0 * 1.07)
    # Parse-back sanity at tight tolerance.
    assert abs(float(out["value"][0]) - 107.0) < 1e-12 * 107.0
