"""Unit tests for the ``node.penalty_method`` input-derivation emitter.

Exercise :func:`derive_node_penalty_method` directly against an in-memory
:class:`FlexDataProvider` — no DB, no full solve.  Pins the per-node
fallback contract (mirrors ``inflow_method`` / ``storage_binding_method``):

* every node in ``node.csv`` gets a row; nodes without an explicit
  ``node__penalty_method.csv`` entry default to ``'regular'``;
* an explicit ``'off'`` row is preserved for that node only;
* an absent ``node__penalty_method.csv`` ⇒ every node ``'regular'``;
* blank rows in the explicit CSV are dropped (they must not shadow the
  per-node default).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from flextool.engine_polars._emit_mid_sets import derive_node_penalty_method
from flextool.engine_polars._flex_data_provider import FlexDataProvider

INPUT_DIR = Path("input")
NODE_KEY = "input/node"
PM_KEY = "input/node__penalty_method"


def _utf8(cols: dict[str, list[str]]) -> pl.DataFrame:
    return pl.DataFrame(cols, schema={c: pl.Utf8 for c in cols})


def _provider(nodes: list[str],
              explicit: dict[str, str] | None) -> FlexDataProvider:
    prov = FlexDataProvider()
    prov.put(NODE_KEY, _utf8({"node": nodes}))
    if explicit is not None:
        prov.put(PM_KEY, _utf8({
            "node": list(explicit.keys()),
            "penalty_method": list(explicit.values()),
        }))
    return prov


def _as_dict(frame: pl.DataFrame) -> dict[str, str]:
    return dict(zip(frame["node"].to_list(),
                    frame["penalty_method"].to_list()))


def test_explicit_off_with_regular_defaults() -> None:
    prov = _provider(["load_off", "load_reg", "bat"], {"load_off": "off"})
    out = derive_node_penalty_method(INPUT_DIR, provider=prov)
    assert out.columns == ["node", "penalty_method"]
    assert _as_dict(out) == {
        "load_off": "off",
        "load_reg": "regular",
        "bat": "regular",
    }


def test_absent_csv_defaults_all_regular() -> None:
    prov = _provider(["a", "b"], explicit=None)
    out = derive_node_penalty_method(INPUT_DIR, provider=prov)
    assert _as_dict(out) == {"a": "regular", "b": "regular"}


def test_explicit_regular_is_preserved() -> None:
    # An explicit 'regular' is redundant with the default but must round-trip.
    prov = _provider(["a", "b"], {"a": "regular", "b": "off"})
    out = derive_node_penalty_method(INPUT_DIR, provider=prov)
    assert _as_dict(out) == {"a": "regular", "b": "off"}


def test_blank_explicit_rows_dropped_and_defaulted() -> None:
    # A blank node / value pair must not shadow the per-node default.
    prov = FlexDataProvider()
    prov.put(NODE_KEY, _utf8({"node": ["a", "b"]}))
    prov.put(PM_KEY, _utf8({"node": ["", "b"], "penalty_method": ["", "off"]}))
    out = derive_node_penalty_method(INPUT_DIR, provider=prov)
    assert _as_dict(out) == {"a": "regular", "b": "off"}


def test_every_node_gets_exactly_one_row() -> None:
    prov = _provider(["a", "b", "c"], {"b": "off"})
    out = derive_node_penalty_method(INPUT_DIR, provider=prov)
    assert out.height == 3
    assert out.select("node").unique().height == 3
