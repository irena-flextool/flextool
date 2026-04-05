from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from flextool.gui.plot_cache import PlotCache


def _make_figure() -> Figure:
    """Create a minimal matplotlib Figure for testing."""
    fig = Figure()
    fig.add_subplot(111)
    return fig


class TestPlotCacheBasic:
    """Basic get/put operations."""

    def test_get_missing_returns_none(self):
        cache = PlotCache(max_size=5)
        assert cache.get(("a",)) is None

    def test_put_and_get(self):
        cache = PlotCache(max_size=5)
        fig = _make_figure()
        cache.put(("key1",), fig)
        assert cache.get(("key1",)) is fig

    def test_put_overwrites_existing(self):
        cache = PlotCache(max_size=5)
        fig1 = _make_figure()
        fig2 = _make_figure()
        cache.put(("key1",), fig1)
        cache.put(("key1",), fig2)
        assert cache.get(("key1",)) is fig2


class TestPlotCacheEviction:
    """LRU eviction behaviour."""

    def test_evicts_oldest_when_full(self):
        cache = PlotCache(max_size=3)
        figs = [_make_figure() for _ in range(4)]
        for i, fig in enumerate(figs):
            cache.put((i,), fig)
        # Oldest (0) should have been evicted
        assert cache.get((0,)) is None
        # Others should still be present
        assert cache.get((1,)) is figs[1]
        assert cache.get((2,)) is figs[2]
        assert cache.get((3,)) is figs[3]

    def test_get_refreshes_lru_order(self):
        cache = PlotCache(max_size=3)
        figs = [_make_figure() for _ in range(3)]
        for i, fig in enumerate(figs):
            cache.put((i,), fig)
        # Access key 0 to move it to end (most recently used)
        cache.get((0,))
        # Now adding a new item should evict key 1 (oldest), not key 0
        fig_new = _make_figure()
        cache.put((99,), fig_new)
        assert cache.get((0,)) is figs[0]
        assert cache.get((1,)) is None
        assert cache.get((2,)) is figs[2]
        assert cache.get((99,)) is fig_new

    def test_put_existing_refreshes_lru_order(self):
        cache = PlotCache(max_size=3)
        figs = [_make_figure() for _ in range(3)]
        for i, fig in enumerate(figs):
            cache.put((i,), fig)
        # Re-put key 0 to refresh it
        fig_updated = _make_figure()
        cache.put((0,), fig_updated)
        # Adding a new item should evict key 1 (oldest), not key 0
        fig_new = _make_figure()
        cache.put((99,), fig_new)
        assert cache.get((0,)) is fig_updated
        assert cache.get((1,)) is None


class TestPlotCacheClear:
    """Cache clear behaviour."""

    def test_clear_empties_cache(self):
        cache = PlotCache(max_size=5)
        for i in range(3):
            cache.put((i,), _make_figure())
        cache.clear()
        for i in range(3):
            assert cache.get((i,)) is None

    def test_clear_on_empty_cache(self):
        cache = PlotCache(max_size=5)
        cache.clear()  # Should not raise
