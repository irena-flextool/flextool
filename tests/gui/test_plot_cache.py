from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from flextool.gui.plot_cache import PlotCache, _figure_size_bytes


def _make_figure(size: tuple[float, float] = (2, 2), dpi: int = 50) -> Figure:
    """Create a matplotlib Figure for testing with known size."""
    fig = Figure(figsize=size, dpi=dpi)
    fig.add_subplot(111)
    return fig


class TestPlotCacheBasic:
    """Basic get/put operations."""

    def test_get_missing_returns_none(self):
        cache = PlotCache()
        assert cache.get(("a",)) is None

    def test_put_and_get(self):
        cache = PlotCache()
        fig = _make_figure()
        cache.put(("key1",), fig)
        assert cache.get(("key1",)) is fig

    def test_put_overwrites_existing(self):
        cache = PlotCache()
        fig1 = _make_figure()
        fig2 = _make_figure()
        cache.put(("key1",), fig1)
        cache.put(("key1",), fig2)
        assert cache.get(("key1",)) is fig2


class TestPlotCacheEviction:
    """Memory-based LRU eviction."""

    def test_evicts_oldest_when_over_limit(self):
        fig_bytes = _figure_size_bytes(_make_figure())
        # Allow ~2.5 figures worth of memory
        cache = PlotCache(max_bytes=int(fig_bytes * 2.5))
        figs = [_make_figure() for _ in range(4)]
        for i, fig in enumerate(figs):
            cache.put((i,), fig)
        # Oldest entries should have been evicted
        assert cache.get((0,)) is None
        assert cache.get((3,)) is figs[3]

    def test_get_refreshes_lru_order(self):
        fig_bytes = _figure_size_bytes(_make_figure())
        cache = PlotCache(max_bytes=int(fig_bytes * 3.5))
        figs = [_make_figure() for _ in range(3)]
        for i, fig in enumerate(figs):
            cache.put((i,), fig)
        # Access key 0 to move it to end
        cache.get((0,))
        # Adding a new item should evict key 1, not key 0
        fig_new = _make_figure()
        cache.put((99,), fig_new)
        assert cache.get((0,)) is figs[0]
        assert cache.get((1,)) is None

    def test_put_existing_refreshes_lru_order(self):
        fig_bytes = _figure_size_bytes(_make_figure())
        cache = PlotCache(max_bytes=int(fig_bytes * 3.5))
        figs = [_make_figure() for _ in range(3)]
        for i, fig in enumerate(figs):
            cache.put((i,), fig)
        # Re-put key 0
        fig_updated = _make_figure()
        cache.put((0,), fig_updated)
        # Adding new should evict key 1, not key 0
        fig_new = _make_figure()
        cache.put((99,), fig_new)
        assert cache.get((0,)) is fig_updated
        assert cache.get((1,)) is None


class TestPlotCacheClear:
    """Cache clear behaviour."""

    def test_clear_empties_cache(self):
        cache = PlotCache()
        for i in range(3):
            cache.put((i,), _make_figure())
        cache.clear()
        for i in range(3):
            assert cache.get((i,)) is None
        assert cache._total_bytes == 0

    def test_clear_on_empty_cache(self):
        cache = PlotCache()
        cache.clear()


class TestPlotCacheMemory:
    """Memory tracking."""

    def test_total_bytes_increases(self):
        cache = PlotCache()
        assert cache._total_bytes == 0
        cache.put(("a",), _make_figure())
        assert cache._total_bytes > 0

    def test_max_gb_property(self):
        cache = PlotCache(max_bytes=512 * 1024 * 1024)
        assert cache.max_gb == 0.5
        cache.max_gb = 1.0
        assert cache._max_bytes == 1024 ** 3
