from __future__ import annotations

import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def _figure_size_bytes(fig: Figure) -> int:
    """Estimate memory usage of a matplotlib Figure in bytes."""
    try:
        renderer = fig.canvas.get_renderer()
        # Rendered bitmap size: width * height * 4 (RGBA)
        w, h = int(renderer.width), int(renderer.height)
        return w * h * 4
    except Exception:
        pass
    # Fallback: estimate from figure size in inches * dpi
    dpi = fig.get_dpi() or 100
    w_in, h_in = fig.get_size_inches()
    return int(w_in * dpi) * int(h_in * dpi) * 4


class PlotCache:
    """Memory-bounded LRU cache for matplotlib Figure objects.

    *max_bytes* sets the memory limit (default 0.5 GB).  Figures are
    evicted oldest-first when the total estimated memory exceeds the
    limit.
    """

    def __init__(self, max_bytes: int = 512 * 1024 * 1024):
        self._max_bytes = max_bytes
        self._cache: OrderedDict[tuple, Figure] = OrderedDict()
        self._sizes: dict[tuple, int] = {}
        self._total_bytes: int = 0

    @property
    def max_gb(self) -> float:
        """Current limit in GB."""
        return self._max_bytes / (1024 ** 3)

    @max_gb.setter
    def max_gb(self, value: float) -> None:
        """Set limit in GB and evict if necessary."""
        self._max_bytes = int(value * 1024 ** 3)
        self._evict_to_limit()

    def get(self, key: tuple) -> Figure | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: tuple, fig: Figure) -> None:
        fig_bytes = _figure_size_bytes(fig)

        if key in self._cache:
            # Update existing entry
            old_bytes = self._sizes.get(key, 0)
            self._total_bytes -= old_bytes
            self._cache.move_to_end(key)
            self._cache[key] = fig
            self._sizes[key] = fig_bytes
            self._total_bytes += fig_bytes
        else:
            self._cache[key] = fig
            self._sizes[key] = fig_bytes
            self._total_bytes += fig_bytes

        self._evict_to_limit()

    def _evict_to_limit(self) -> None:
        """Evict oldest entries until total size is within the limit."""
        while self._total_bytes > self._max_bytes and self._cache:
            evicted_key, evicted_fig = self._cache.popitem(last=False)
            evicted_bytes = self._sizes.pop(evicted_key, 0)
            self._total_bytes -= evicted_bytes
            evicted_fig.clf()
            plt.close(evicted_fig)

    def clear(self) -> None:
        for fig in self._cache.values():
            fig.clf()
            plt.close(fig)
        self._cache.clear()
        self._sizes.clear()
        self._total_bytes = 0
