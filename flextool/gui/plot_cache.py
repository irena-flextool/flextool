from __future__ import annotations

from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class PlotCache:
    """Bounded LRU cache for matplotlib Figure objects."""

    def __init__(self, max_size: int = 30):
        self._max_size = max_size
        self._cache: OrderedDict[tuple, Figure] = OrderedDict()

    def get(self, key: tuple) -> Figure | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: tuple, fig: Figure) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = fig
        else:
            if len(self._cache) >= self._max_size:
                _, evicted = self._cache.popitem(last=False)
                evicted.clf()
                plt.close(evicted)
            self._cache[key] = fig

    def clear(self) -> None:
        for fig in self._cache.values():
            fig.clf()
            plt.close(fig)
        self._cache.clear()
