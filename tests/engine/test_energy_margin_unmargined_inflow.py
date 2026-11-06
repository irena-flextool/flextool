"""``solve_data/pdtNodeInflow_unmargined``: the inflow before the energy margin of each solve.

The group ``increase_reserve_ratio`` scales the real demand, so the margin emitters record the un-margined
``pdtNodeInflow`` (``emit/_energy_margin._record_unmargined_inflow``).  Consecutive sub-solves can share one
Provider, so the record must follow each solve: absent in models without a margin, written when a margin is
applied (by the multiplier, or by the adder alone), refreshed by every later solve once present (no stale
frame), and an error when ``pdtNodeInflow`` itself is missing.  Unit tests on an in-memory Provider.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
from flextool.engine.preprocess.emit._energy_margin import emit_energy_margin_inflow
from flextool.engine.preprocess.emit._energy_margin_adder import emit_energy_margin_adder

INPUT_DIR = Path("input")
SOLVE_DATA_DIR = Path("solve_data")
PDT_KEY = "solve_data/pdtNodeInflow"
REC_KEY = "solve_data/pdtNodeInflow_unmargined"
STEPS = [("p", "t1"), ("p", "t2")]


def _utf8(cols: dict[str, list[str]]) -> pl.DataFrame:
    return pl.DataFrame(cols, schema={c: pl.Utf8 for c in cols})


def _inflow(values: tuple[str, str]) -> pl.DataFrame:
    return _utf8({"node": ["nodeA", "nodeA"], "period": ["p", "p"], "time": ["t1", "t2"], "value": list(values)})


def _provider(method: str | None, inflow: pl.DataFrame | None) -> FlexDataProvider:
    prov = FlexDataProvider()
    if inflow is not None:
        prov.put(PDT_KEY, inflow)
    prov.put("solve_data/steps_in_use", _utf8({"period": [d for d, _ in STEPS], "time": [t for _, t in STEPS]}))
    if method is not None:
        prov.put("input/node__energy_margin_method", _utf8({"node": ["nodeA"], "energy_margin_method": [method]}))
        prov.put("input/energy_margin_multiplier", _utf8({"node": ["nodeA"], "energy_margin_multiplier": ["1.1"]}))
        prov.put("input/energy_margin_adder", _utf8({"node": ["nodeA"], "energy_margin_adder": ["50.0"]}))
    return prov


def _state(invest: bool) -> SimpleNamespace:
    return SimpleNamespace(solve=SimpleNamespace(invest_periods={"S": [("p",)]} if invest else {}))


def _solve(prov: FlexDataProvider, *, invest: bool) -> None:
    """The two margin emitters in their pipeline order (multiplier, then adder)."""
    emit_energy_margin_inflow(_state(invest), "S", INPUT_DIR, SOLVE_DATA_DIR, provider=prov)
    emit_energy_margin_adder(_state(invest), "S", INPUT_DIR, SOLVE_DATA_DIR, provider=prov)


def test_no_margin_writes_no_record() -> None:
    for method, invest in ((None, True), ("inflow_multiplier", False), ("inflow_adder", False)):
        prov = _provider(method, _inflow(("-100.0", "-200.0")))
        _solve(prov, invest=invest)
        assert not prov.has(REC_KEY), (method, invest)


@pytest.mark.parametrize("method", ["inflow_multiplier", "inflow_adder"])
def test_margin_records_the_unmargined_inflow(method: str) -> None:
    """Invest solve with a margin: the record is the inflow before the margin (verbatim), pdtNodeInflow is
    margined.  The adder alone writes the record."""
    before = _inflow(("-100.0", "-200.0"))
    prov = _provider(method, before)
    _solve(prov, invest=True)
    assert prov.get(REC_KEY).equals(before)
    assert not prov.get(PDT_KEY).equals(before)


def test_record_is_refreshed_by_a_later_solve_without_margin() -> None:
    """An invest solve records its inflow; the next solve on the same Provider (a dispatch solve: no margin)
    has another inflow: the record follows it."""
    prov = _provider("inflow_multiplier", _inflow(("-100.0", "-200.0")))
    _solve(prov, invest=True)
    later = _inflow(("-300.0", "-400.0"))
    prov.put(PDT_KEY, later)
    _solve(prov, invest=False)
    assert prov.get(REC_KEY).equals(later)
    assert prov.get(PDT_KEY).equals(later)


def test_adder_does_not_overwrite_the_multiplier_record() -> None:
    """With both margins in one solve the record is the inflow before both (written by the multiplier)."""
    before = _inflow(("-100.0", "-200.0"))
    prov = _provider("inflow_multiplier", before)
    prov.put("input/node__energy_margin_method",
             _utf8({"node": ["nodeA", "nodeB"], "energy_margin_method": ["inflow_multiplier", "inflow_adder"]}))
    prov.put("input/energy_margin_adder", _utf8({"node": ["nodeB"], "energy_margin_adder": ["50.0"]}))
    _solve(prov, invest=True)
    assert prov.get(REC_KEY).equals(before)
    out = prov.get(PDT_KEY)
    assert out.filter(pl.col("node") == "nodeB").height == len(STEPS)


def test_missing_inflow_raises_when_a_record_is_due() -> None:
    """A Provider with a record but no pdtNodeInflow cannot refresh it: error instead of a stale frame."""
    prov = _provider(None, None)
    prov.put(REC_KEY, _inflow(("-1.0", "-2.0")))
    with pytest.raises(RuntimeError, match="pdtNodeInflow is missing"):
        emit_energy_margin_inflow(_state(False), "S", INPUT_DIR, SOLVE_DATA_DIR, provider=prov)
