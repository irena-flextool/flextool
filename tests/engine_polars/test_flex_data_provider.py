"""Unit tests for :class:`FlexDataProvider` scaffolding (Step 1-a).

These tests target the bare interface contract: get / has / put
roundtrip, missing-key semantics, parent-qualified name fallback, and
snapshot stub callability.  No cascade integration, no orchestration —
the Provider is not yet wired into anything in this step.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars._flex_data_provider import FlexDataProvider


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _df(rows: int = 2) -> pl.DataFrame:
    return pl.DataFrame({"a": list(range(rows)), "b": [float(i) for i in range(rows)]})


# ---------------------------------------------------------------------------
# get / has / put roundtrip
# ---------------------------------------------------------------------------

def test_put_then_get_returns_same_frame():
    p = FlexDataProvider()
    df = _df()
    p.put("p_flow_max", df)
    out = p.get("p_flow_max")
    assert out is df


def test_has_true_after_put():
    p = FlexDataProvider()
    p.put("p_flow_max", _df())
    assert p.has("p_flow_max") is True


def test_get_returns_none_for_missing_name():
    p = FlexDataProvider()
    assert p.get("does_not_exist") is None


def test_has_false_for_missing_name():
    p = FlexDataProvider()
    assert p.has("does_not_exist") is False


def test_put_strips_csv_suffix_on_storage():
    p = FlexDataProvider()
    df = _df()
    p.put("p_flow_max.csv", df)
    # Either form must retrieve.
    assert p.get("p_flow_max") is df
    assert p.get("p_flow_max.csv") is df


def test_put_overwrites_existing_frame():
    p = FlexDataProvider()
    p.put("x", _df(2))
    p.put("x", _df(5))
    out = p.get("x")
    assert out is not None
    assert out.height == 5


def test_keys_lists_stored_names_without_csv_suffix():
    p = FlexDataProvider()
    p.put("a", _df())
    p.put("b.csv", _df())
    keys = sorted(p.keys())
    assert keys == ["a", "b"]


# ---------------------------------------------------------------------------
# Parent-qualified lookup — bidirectional fallback mirroring the seed funnel
# ---------------------------------------------------------------------------

def test_bare_put_resolves_qualified_get():
    """``put('p_flow_max', df)`` then ``get('solve_data/p_flow_max')``
    returns the same frame.  Mirrors the seed funnel's
    ``_seed_or_pick``-style behaviour: callers that pass a qualified
    path fall back to the bare basename when only that was stashed.
    """
    p = FlexDataProvider()
    df = _df()
    p.put("p_flow_max", df)
    assert p.get("solve_data/p_flow_max") is df
    assert p.has("solve_data/p_flow_max") is True


def test_qualified_put_resolves_bare_get():
    """Inverse: ``put('solve_data/p_flow_max', df)`` then
    ``get('p_flow_max')`` returns the frame.  Step 1 needs the bare
    form to resolve too because loaders that haven't been migrated yet
    call ``provider.get(name)`` without a parent qualifier.
    """
    p = FlexDataProvider()
    df = _df()
    p.put("solve_data/p_flow_max", df)
    assert p.get("p_flow_max") is df
    assert p.has("p_flow_max") is True


def test_qualified_put_resolves_exact_qualified_get():
    p = FlexDataProvider()
    df = _df()
    p.put("solve_data/p_flow_max", df)
    assert p.get("solve_data/p_flow_max") is df


def test_qualified_get_with_csv_suffix_is_normalised():
    p = FlexDataProvider()
    df = _df()
    p.put("solve_data/timeline", df)
    assert p.get("solve_data/timeline.csv") is df


def test_two_qualified_puts_keep_distinct_frames():
    """When the same basename lives in two source dirs (the canonical
    ``timeline.csv`` example), each qualified ``put`` stores a distinct
    frame and qualified ``get`` returns the right one.
    """
    p = FlexDataProvider()
    df_input = _df(2)
    df_solve = _df(3)
    p.put("input/timeline", df_input)
    p.put("solve_data/timeline", df_solve)
    assert p.get("input/timeline") is df_input
    assert p.get("solve_data/timeline") is df_solve


def test_bare_get_finds_first_qualified_match():
    """When only qualified puts exist, a bare ``get`` returns the
    first stored match (insertion order).  This is the
    ``_seed_or_pick``-flavoured permissive fallback used by callers
    that haven't been migrated to qualified lookups yet.
    """
    p = FlexDataProvider()
    df_input = _df(2)
    p.put("input/timeline", df_input)
    p.put("solve_data/timeline", _df(3))
    # First insertion wins.
    assert p.get("timeline") is df_input


# ---------------------------------------------------------------------------
# Snapshot stubs — must be callable + no-op
# ---------------------------------------------------------------------------

def test_snapshot_raw_inputs_callable_noop(tmp_path: Path):
    p = FlexDataProvider()
    p.put("p_flow_max", _df())
    # Step 1-a: stub.  Must not raise; must not write files.
    p.snapshot_raw_inputs(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_snapshot_processed_inputs_callable_noop(tmp_path: Path):
    p = FlexDataProvider()
    p.put("p_flow_max", _df())
    p.snapshot_processed_inputs(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_snapshot_methods_accept_nonexistent_path_without_raising(
    tmp_path: Path,
):
    """Stubs should not blow up on a path that does not yet exist —
    Step 2 will materialise the directory inside the snapshot impls.
    """
    p = FlexDataProvider()
    target = tmp_path / "does_not_exist_yet"
    p.snapshot_raw_inputs(target)
    p.snapshot_processed_inputs(target)
