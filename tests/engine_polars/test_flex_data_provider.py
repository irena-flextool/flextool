"""Unit tests for :class:`FlexDataProvider` scaffolding (Step 1-a).

These tests target the bare interface contract: get / has / put
roundtrip, missing-key semantics, parent-qualified name fallback, and
snapshot stub callability.  No cascade integration, no orchestration —
the Provider is not yet wired into anything in this step.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

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
# Parent-qualified lookup — exact-match contract (Phase 4.2-2)
# ---------------------------------------------------------------------------

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


def test_bare_get_does_not_resolve_qualified_put():
    """Phase 4.2-2 contract: a bare ``get`` does NOT fall back to a
    qualified-key match.  The Phase 0a-era bare↔qualified fallback was
    dropped to give one canonical key form and one lookup path; a typo
    or unqualified key returns ``None`` rather than silently grabbing
    the wrong frame.
    """
    p = FlexDataProvider()
    p.put("input/timeline", _df(2))
    p.put("solve_data/timeline", _df(3))
    assert p.get("timeline") is None
    assert p.has("timeline") is False


def test_qualified_get_does_not_resolve_bare_put():
    """Phase 4.2-2 contract: a qualified ``get`` does NOT fall back to
    a bare-key match.  All real producers emit qualified keys; this
    test pins the contract so a producer regression that puts bare
    surfaces as a miss, not as a silent qualified-form hit.
    """
    p = FlexDataProvider()
    p.put("p_flow_max", _df())
    assert p.get("solve_data/p_flow_max") is None
    assert p.has("solve_data/p_flow_max") is False


# ---------------------------------------------------------------------------
# Snapshot stubs — must be callable + no-op
# ---------------------------------------------------------------------------

def test_snapshot_raw_inputs_callable_noop(tmp_path: Path):
    p = FlexDataProvider()
    p.put("p_flow_max", _df())
    # Step 1-a: stub.  Must not raise; must not write files.
    p.snapshot_raw_inputs(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_snapshot_processed_inputs_dumps_every_frame(tmp_path: Path):
    p = FlexDataProvider()
    p.put("p_flow_max", _df())
    p.put("solve_data/realized_invest", _df())
    p.snapshot_processed_inputs(tmp_path)
    # Bare-keyed frames land at the top level; parent-qualified keys go
    # into the matching subdirectory.
    assert (tmp_path / "p_flow_max.csv").exists()
    assert (tmp_path / "solve_data" / "realized_invest.csv").exists()


def test_snapshot_methods_accept_nonexistent_path_without_raising(
    tmp_path: Path,
):
    """Snapshots create the destination directory as needed."""
    p = FlexDataProvider()
    target = tmp_path / "does_not_exist_yet"
    # raw_inputs is a deliberate stub; processed_inputs writes the
    # frames + materialises the directory.
    p.snapshot_raw_inputs(target)
    p.put("p_flow_max", _df())
    p.snapshot_processed_inputs(target)
    assert (target / "p_flow_max.csv").exists()


# ---------------------------------------------------------------------------
# Phase 6a — source tagging (put(..., source=...) + get_source)
# ---------------------------------------------------------------------------

def test_put_without_source_has_no_recorded_source():
    p = FlexDataProvider()
    p.put("p_flow_max", _df())
    assert p.get_source("p_flow_max") is None


def test_put_with_source_records_tag_retrievable_via_get_source():
    p = FlexDataProvider()
    p.put("p_flow_max", _df(), source="external_override")
    assert p.get_source("p_flow_max") == "external_override"


def test_get_source_for_unknown_key_returns_none_without_raising():
    p = FlexDataProvider()
    # Must NOT raise, must return None — get_source is a query, not a
    # contract enforcer.
    assert p.get_source("never_put") is None


def test_get_source_accepts_csv_suffix_form():
    p = FlexDataProvider()
    p.put("p_flow_max", _df(), source="tag_a")
    assert p.get_source("p_flow_max.csv") == "tag_a"


def test_put_without_source_clears_previous_tag_on_overwrite():
    """Overwriting a tagged entry with an untagged put must clear the
    stale tag — the new frame is the new truth, and a None source is a
    deliberate "no provenance recorded" signal."""
    p = FlexDataProvider()
    p.put("x", _df(), source="tag_a")
    assert p.get_source("x") == "tag_a"
    p.put("x", _df())
    assert p.get_source("x") is None


def test_put_with_source_overwrites_previous_tag():
    p = FlexDataProvider()
    p.put("x", _df(), source="tag_a")
    p.put("x", _df(), source="tag_b")
    assert p.get_source("x") == "tag_b"


def test_eviction_clears_source_tag():
    """Phase 6a: source tags are per-frame metadata; eviction must drop
    them alongside the frame so a re-put after eviction starts clean."""
    p = FlexDataProvider(rss_budget_mb=0.0)
    p.register_handler(
        "h",
        reads=["x"],
        groups=["g0"],
    )
    p.precompute_lifetimes(["g0", "g1"])
    p.put("x", _df(), source="external_override")
    assert p.get_source("x") == "external_override"
    evicted = p.release_unused(after="g0")
    assert "x" in evicted
    # Source vacates with the frame.
    assert p.get_source("x") is None


def test_translate_overrides_to_provider_tags_each_entry_external_override():
    """End-to-end: the override translator stamps every write with the
    ``external_override`` source tag so the Phase 6b audit dump can
    discriminate overridden keys from natural-cascade writes.
    """
    from flextool.engine_polars import _provider_keys as K
    from flextool.engine_polars._provider_translators import (
        translate_overrides_to_provider,
    )

    overrides = {
        K.HANDOFF_REALIZED_INVEST: pl.DataFrame(
            {"entity": ["e1"], "period": ["p1"], "value": ["1.0"]}
        ),
        K.HANDOFF_DIVEST_CUMULATIVE: pl.DataFrame(
            {"entity": ["e1"], "value": ["2.0"]}
        ),
    }
    p = FlexDataProvider()
    translate_overrides_to_provider(overrides, p)
    assert p.get_source(K.OVERRIDE_REALIZED_INVEST) == "external_override"
    assert p.get_source(K.OVERRIDE_DIVEST_CUMULATIVE) == "external_override"
    # Untouched keys remain untagged.
    assert p.get_source(K.OVERRIDE_ROLL_END_STATE) is None
