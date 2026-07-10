"""Regression test for the ``node__storage_nested_fix_method`` emitter.

This guards the exact contract mismatch that made the multi-solve
storage-nested-fix handoff (``fix_quantity`` / ``fix_price`` /
``fix_usage``) inert: the generic ``_PARAMETER_SPECS`` emit produces
``input/node__storage_nested_fix_method`` with the value column named
``storage_nested_fix_method``, but every solve-time consumer
(``_load_handoff_aux_pair`` and the ``build_handoff_from_solution``
producer guards) checks for a column literally named ``method``.

The native emitter ``emit_node_storage_nested_fix_method`` renames the
value column to ``method`` and stores the frame under the
``solve_data/`` key the consumers read.  Without it the loader returned
None and all three method sets stayed empty.

Two assertions:

1. **Emitter unit** — the emitter produces
   ``solve_data/node__storage_nested_fix_method`` with columns exactly
   ``["node", "method"]``.
2. **Loader round-trip** — ``_load_handoff_aux_pair`` with expected
   columns ``("node", "method")`` returns the frame non-None (it
   returned None before the fix).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from flextool.engine_polars import _emit_mid_sets as _mid
from flextool.engine_polars._flex_data_provider import FlexDataProvider
from flextool.engine_polars.input import _load_handoff_aux_pair


def _seed_provider() -> FlexDataProvider:
    provider = FlexDataProvider()
    provider.put(
        "input/node__storage_nested_fix_method",
        pl.DataFrame({"node": ["s"],
                      "storage_nested_fix_method": ["fix_usage"]}),
    )
    provider.put("input/node", pl.DataFrame({"node": ["s"]}))
    return provider


def test_emitter_produces_method_column():
    """The emitter renames the value column to ``method``."""
    provider = _seed_provider()
    input_dir = Path("input")
    solve_data_dir = Path("solve_data")

    _mid.emit_node_storage_nested_fix_method(
        input_dir, solve_data_dir, provider=provider)

    out = provider.get("solve_data/node__storage_nested_fix_method")
    assert out is not None, "Emitter must register the solve_data frame"
    assert out.columns == ["node", "method"], (
        f"Emitted frame must carry the 'method' contract column; "
        f"got columns {out.columns}")
    assert out.to_dicts() == [{"node": "s", "method": "fix_usage"}]


def test_loader_round_trip_non_none():
    """``_load_handoff_aux_pair(("node","method"))`` round-trips non-None.

    This is the assertion that FAILED before the fix (the input/ frame
    carried ``storage_nested_fix_method`` not ``method``, so the loader's
    ``b not in df.columns`` guard returned None) and PASSES after it.
    """
    provider = _seed_provider()
    _mid.emit_node_storage_nested_fix_method(
        Path("input"), Path("solve_data"), provider=provider)

    loaded = _load_handoff_aux_pair(
        Path("solve_data") / "node__storage_nested_fix_method.csv",
        ("node", "method"),
        provider=provider,
    )
    assert loaded is not None, (
        "Loader must return the frame non-None once the value column is "
        "renamed to 'method' — this is the exact gap that made the "
        "nested-fix handoff inert")
    assert "method" in loaded.columns
    assert loaded.to_dicts() == [{"node": "s", "method": "fix_usage"}]


def test_emitter_empty_input_is_header_only():
    """No nested-fix node → height-0 frame (loader short-circuits to None)."""
    provider = FlexDataProvider()
    provider.put(
        "input/node__storage_nested_fix_method",
        pl.DataFrame(schema={"node": pl.Utf8,
                             "storage_nested_fix_method": pl.Utf8}),
    )
    provider.put("input/node", pl.DataFrame({"node": ["s"]}))

    _mid.emit_node_storage_nested_fix_method(
        Path("input"), Path("solve_data"), provider=provider)

    out = provider.get("solve_data/node__storage_nested_fix_method")
    assert out is not None
    assert out.columns == ["node", "method"]
    assert out.height == 0

    loaded = _load_handoff_aux_pair(
        Path("solve_data") / "node__storage_nested_fix_method.csv",
        ("node", "method"),
        provider=provider,
    )
    assert loaded is None, "Empty frame must short-circuit the loader to None"
