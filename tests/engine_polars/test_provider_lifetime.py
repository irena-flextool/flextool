"""Track B Phase B.1 — FlexDataProvider lifetime + eviction infrastructure.

Exercises:

* ``register_handler`` + ``precompute_lifetimes`` build a sound
  ``_last_needed`` map.
* ``release_unused`` honours ``retain_all`` and the ``rss_budget_mb``
  threshold gate.
* ``get`` on an evicted frame raises :class:`EvictedFrameError`.
* ``reset_lifetimes`` clears the bookkeeping without touching the cache.

No cascade-side plumbing is exercised here — that lands in Phase B.2+.
"""
from __future__ import annotations

import os

import polars as pl
import pytest

from flextool.engine_polars._flex_data_provider import (
    EvictedFrameError,
    FlexDataProvider,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frame(n_rows: int = 100, n_cols: int = 4) -> pl.DataFrame:
    """A throwaway frame with enough cells to make estimated_size() > 0."""
    return pl.DataFrame(
        {f"c{i}": list(range(n_rows)) for i in range(n_cols)}
    )


# ---------------------------------------------------------------------------
# Registration + lifetime computation
# ---------------------------------------------------------------------------


def test_register_handler_stores_reads_and_groups() -> None:
    p = FlexDataProvider()
    p.register_handler("H1", reads=["a", "b"], groups=[(1, "g1"), (1, "g2")])
    p.register_handler("H2", reads=["b", "c"])  # groups=None → single-pass

    assert p._reads["H1"] == ["a", "b"]
    assert p._reads["H2"] == ["b", "c"]
    assert p._handler_groups["H1"] == [(1, "g1"), (1, "g2")]
    assert p._handler_groups["H2"] == []


def test_precompute_lifetimes_picks_last_consumer() -> None:
    p = FlexDataProvider()
    p.register_handler("H_a", reads=["a"], groups=["g0"])
    p.register_handler("H_b", reads=["a", "b"], groups=["g1"])
    p.register_handler("H_c", reads=["c"], groups=["g2"])

    p.precompute_lifetimes(["g0", "g1", "g2"])

    # 'a' is read in g0 and g1; last is g1.
    assert p._last_needed["a"] == ("g1", 1)
    # 'b' is read only in g1.
    assert p._last_needed["b"] == ("g1", 1)
    # 'c' is read only in g2.
    assert p._last_needed["c"] == ("g2", 2)


def test_precompute_pins_groupless_handler_to_last_group() -> None:
    """A handler registered without ``groups=...`` is treated as
    single-pass / 'unknown span' and pinned to the last group.

    This is the conservative default; the corresponding frame names
    are effectively never evicted unless someone refines the
    declaration later.
    """
    p = FlexDataProvider()
    p.register_handler("H_pinned", reads=["pinned"])
    p.register_handler("H_early", reads=["early"], groups=["g0"])

    p.precompute_lifetimes(["g0", "g1", "g2"])
    assert p._last_needed["pinned"] == ("g2", 2)
    assert p._last_needed["early"] == ("g0", 0)


def test_precompute_resets_evicted_set() -> None:
    """Re-running ``precompute_lifetimes`` clears the evicted-frame
    markers (eviction state is per-build-loop, not persistent).
    """
    p = FlexDataProvider(rss_budget_mb=0.0)
    p.put("a", _frame())
    p.register_handler("H", reads=["a"], groups=["g0"])
    p.precompute_lifetimes(["g0", "g1"])
    p.release_unused(after="g0")
    assert p.is_evicted("a")
    # Re-precompute clears markers (but doesn't restore the frame!).
    p.precompute_lifetimes(["g0", "g1"])
    assert not p.is_evicted("a")


# ---------------------------------------------------------------------------
# Eviction — threshold gate + retain_all + EvictedFrameError
# ---------------------------------------------------------------------------


def test_release_unused_drops_dead_frames_when_budget_crossed() -> None:
    p = FlexDataProvider(rss_budget_mb=0.0)  # force the gate open
    p.put("a", _frame())
    p.put("b", _frame())
    p.register_handler("H_a", reads=["a"], groups=["g0"])
    p.register_handler("H_b", reads=["b"], groups=["g1"])
    p.precompute_lifetimes(["g0", "g1"])

    evicted = p.release_unused(after="g0")
    assert evicted == ["a"]
    assert "a" not in p._frames
    assert "b" in p._frames
    assert p.is_evicted("a")
    assert not p.is_evicted("b")


def test_release_unused_noop_when_under_budget() -> None:
    """Threshold gate closed → release_unused is a no-op even with a
    valid lifetime map.
    """
    p = FlexDataProvider(rss_budget_mb=1024 * 1024.0)  # 1 TB budget
    p.put("a", _frame())
    p.register_handler("H_a", reads=["a"], groups=["g0"])
    p.precompute_lifetimes(["g0", "g1"])
    assert p.release_unused(after="g0") == []
    assert "a" in p._frames


def test_release_unused_noop_when_retain_all() -> None:
    """``retain_all`` disables eviction unconditionally (CSV-dump
    mode)."""
    p = FlexDataProvider(rss_budget_mb=0.0, retain_all=True)
    p.put("a", _frame())
    p.register_handler("H_a", reads=["a"], groups=["g0"])
    p.precompute_lifetimes(["g0", "g1"])
    assert p.release_unused(after="g0") == []


def test_release_unused_noop_without_precompute() -> None:
    """Before ``precompute_lifetimes`` runs, ``release_unused`` is a
    no-op (defensive).
    """
    p = FlexDataProvider(rss_budget_mb=0.0)
    p.put("a", _frame())
    assert p.release_unused(after="g0") == []
    assert "a" in p._frames


def test_get_after_eviction_raises_evictedframeerror() -> None:
    p = FlexDataProvider(rss_budget_mb=0.0)
    p.put("a", _frame())
    p.register_handler("H_a", reads=["a"], groups=["g0"])
    p.precompute_lifetimes(["g0", "g1"])
    p.release_unused(after="g0")

    with pytest.raises(EvictedFrameError) as exc:
        p.get("a")
    assert exc.value.frame_name == "a"


def test_evictedframeerror_message_actionable() -> None:
    """The error message names the frame and the responsible item-group
    so the agent reading the failure can pinpoint the drift.
    """
    p = FlexDataProvider(rss_budget_mb=0.0)
    p.put("foo", _frame())
    p.register_handler("H", reads=["foo"], groups=[("g0", "early")])
    p.precompute_lifetimes([("g0", "early"), ("g1", "late")])
    p.release_unused(after=("g0", "early"))
    with pytest.raises(EvictedFrameError) as exc:
        p.get("foo")
    msg = str(exc.value)
    assert "foo" in msg
    # Item-group token serialised into the message.
    assert "g0" in msg or "early" in msg


def test_rss_estimate_mb_sums_live_cache() -> None:
    p = FlexDataProvider()
    assert p.rss_estimate_mb() == 0.0
    p.put("a", _frame(n_rows=10_000, n_cols=8))
    p.put("b", _frame(n_rows=10_000, n_cols=8))
    sz = p.rss_estimate_mb()
    assert sz > 0.0
    # Sanity: two frames should be ~2x one frame's size.
    p_one = FlexDataProvider()
    p_one.put("a", _frame(n_rows=10_000, n_cols=8))
    one_sz = p_one.rss_estimate_mb()
    assert sz == pytest.approx(2 * one_sz, rel=0.1)


def test_env_var_overrides_budget() -> None:
    """``FLEXTOOL_RSS_BUDGET_MB`` env var sets the budget when no
    constructor arg is provided.  Constructor arg wins when both are
    set.
    """
    os.environ["FLEXTOOL_RSS_BUDGET_MB"] = "42"
    try:
        p_env = FlexDataProvider()
        assert p_env.rss_budget_mb == 42.0

        p_arg = FlexDataProvider(rss_budget_mb=7.0)
        assert p_arg.rss_budget_mb == 7.0
    finally:
        del os.environ["FLEXTOOL_RSS_BUDGET_MB"]


def test_reset_lifetimes_clears_bookkeeping_only() -> None:
    p = FlexDataProvider(rss_budget_mb=0.0)
    p.put("a", _frame())
    p.register_handler("H", reads=["a"], groups=["g0"])
    p.precompute_lifetimes(["g0", "g1"])
    p.release_unused(after="g0")
    assert p.is_evicted("a")

    p.reset_lifetimes()
    assert p._reads == {}
    assert p._handler_groups == {}
    assert p._group_order is None
    assert p._last_needed == {}
    assert not p.is_evicted("a")
    # Frame still in cache (eviction is irreversible, but markers are
    # cleared — caller is responsible for re-putting if they want it
    # back).
    assert "a" not in p._frames


def test_unregistered_frame_not_evicted() -> None:
    """Frames not declared by any handler's READS stay forever — they
    are 'unknown lifetime', treated as live.
    """
    p = FlexDataProvider(rss_budget_mb=0.0)
    p.put("not_declared", _frame())
    p.register_handler("H_other", reads=["something_else"], groups=["g0"])
    p.precompute_lifetimes(["g0", "g1"])
    assert p.release_unused(after="g0") == []
    assert "not_declared" in p._frames


def test_qualified_and_bare_keys_share_eviction() -> None:
    """A frame put under ``"solve_data/foo"`` evicted by name → bare
    ``"foo"`` lookup also raises EvictedFrameError."""
    p = FlexDataProvider(rss_budget_mb=0.0)
    p.put("solve_data/foo", _frame())
    p.register_handler("H", reads=["solve_data/foo"], groups=["g0"])
    p.precompute_lifetimes(["g0", "g1"])
    p.release_unused(after="g0")
    with pytest.raises(EvictedFrameError):
        p.get("foo")  # bare lookup picks up the qualified eviction
    with pytest.raises(EvictedFrameError):
        p.get("solve_data/foo")
