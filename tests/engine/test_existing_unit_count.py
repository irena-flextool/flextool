"""With ``virtual_unitsize`` set, ``existing`` is the number of units: capacity = existing × virtual_unitsize.

The existing unit count is capacity / unitsize, so it equals ``existing`` with virtual_unitsize and 1 without it
(unitsize = existing).  These tests pin the provider-less fallbacks of ``params._derived`` (no chained
``p_entity_all_existing``: workdir None) on an in-memory source.  Unit ``u`` (a → b) has existing 2 and
virtual_unitsize 250 (two 250 MW units, 500 MW): count 2.  Its twin ``w`` has existing 500 MW and no
virtual_unitsize: unitsize 500, count 500 / 500 = 1.
"""
from __future__ import annotations

import polars as pl
import pytest

from flextool.engine import InMemoryReader
from flextool.engine.preprocess.params import _derived

PERIOD = "p2020"


def _source(existing_shape: str) -> InMemoryReader:
    if existing_shape == "constant":
        existing = pl.DataFrame({"name": ["u", "w"], "value": [2.0, 500.0]})
    else:
        existing = pl.DataFrame({"name": ["u", "w"], "period": [PERIOD, PERIOD], "value": [2.0, 500.0]})
    return InMemoryReader(entities={
        "unit": pl.DataFrame({"name": ["u", "w"]}),
        "node": pl.DataFrame({"name": ["a", "b"]}),
        "connection": pl.DataFrame(schema={"name": pl.Utf8}),
        "solve": pl.DataFrame({"name": ["s"]}),
        "unit__inputNode": pl.DataFrame({"unit": ["u", "w"], "node": ["a", "a"]}),
        "unit__outputNode": pl.DataFrame({"unit": ["u", "w"], "node": ["b", "b"]}),
    }, parameters={
        ("unit", "existing"): existing,
        ("unit", "virtual_unitsize"): pl.DataFrame({"name": ["u"], "value": [250.0]}),
        ("solve", "realized_periods"): pl.DataFrame({"name": ["s"], "value": [PERIOD]}),
    })


def _values(frame: pl.DataFrame) -> dict[str, float]:
    assert frame["d"].cast(pl.Utf8).to_list() == [PERIOD] * frame.height
    return {str(p): float(v) for p, v in zip(frame["p"].to_list(), frame["value"].to_list())}


@pytest.mark.parametrize("existing_shape", ["constant", "period_map"])
def test_existing_count_fallback_is_existing_with_virtual_unitsize(existing_shape):
    dt = pl.DataFrame({"d": [PERIOD], "t": ["t0001"]})
    count = _derived.p_process_existing_count_from_source(_source(existing_shape), dt, None, None)
    assert count is not None
    assert _values(count.frame) == {"u": 2.0, "w": 1.0}


@pytest.mark.parametrize("existing_shape", ["constant", "period_map"])
def test_flow_upper_existing_fallback_is_existing_count_per_arc(existing_shape):
    pss = pl.DataFrame({"p": ["u", "w"], "source": ["a", "a"], "sink": ["b", "b"]})
    upper = _derived.p_flow_upper_existing_from_source(_source(existing_shape), pss, None, None)
    assert upper is not None
    assert upper.frame.select("p", "source", "sink").rows() == [("u", "a", "b"), ("w", "a", "b")]
    assert _values(upper.frame) == {"u": 2.0, "w": 1.0}
