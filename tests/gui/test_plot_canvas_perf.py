"""Performance benchmarks for PlotCanvas figure switching.

Run with: python3 -m pytest tests/gui/test_plot_canvas_perf.py -v -s
The -s flag is needed to see timing output.
"""
from __future__ import annotations

import time
import tkinter as tk

import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from flextool.gui.plot_canvas import PlotCanvas


@pytest.fixture
def tk_app():
    """Create a Tk root and PlotCanvas, sized like a real window."""
    try:
        root = tk.Tk()
        root.geometry("1200x800")
        root.update()
    except tk.TclError:
        pytest.skip("No display available")

    canvas = PlotCanvas(root)
    canvas.pack(fill="both", expand=True)
    root.update_idletasks()

    yield root, canvas

    canvas.cleanup()
    root.destroy()


def _make_simple_bar_figure() -> Figure:
    """Small bar chart (~10 bars)."""
    fig = Figure(figsize=(8, 5), dpi=100)
    ax = fig.add_subplot(111)
    ax.bar(range(10), np.random.rand(10))
    ax.set_title("Simple bar chart")
    return fig


def _make_line_figure(n_points: int = 8760, n_lines: int = 5) -> Figure:
    """Time series line plot."""
    fig = Figure(figsize=(12, 6), dpi=100)
    ax = fig.add_subplot(111)
    x = np.arange(n_points)
    for i in range(n_lines):
        ax.plot(x, np.random.randn(n_points).cumsum(), label=f"Line {i}")
    ax.legend()
    ax.set_title(f"Time series ({n_points} pts x {n_lines} lines)")
    return fig


def _make_complex_figure() -> Figure:
    """Complex multi-subplot figure."""
    fig = Figure(figsize=(14, 10), dpi=100)
    for i in range(6):
        ax = fig.add_subplot(2, 3, i + 1)
        x = np.arange(1000)
        for j in range(8):
            ax.plot(x, np.random.randn(1000).cumsum())
        ax.set_title(f"Subplot {i}")
    fig.suptitle("Complex multi-subplot")
    return fig


def _time_switch(canvas: PlotCanvas, fig: Figure, root: tk.Tk) -> float:
    """Time a single display_figure call including Tk event processing."""
    root.update_idletasks()
    t0 = time.perf_counter()
    canvas.display_figure(fig)
    root.update_idletasks()
    return time.perf_counter() - t0


class TestPlotCanvasPerf:
    """Performance benchmarks for figure switching."""

    def test_simple_bar_switch(self, tk_app):
        root, canvas = tk_app
        fig_a = _make_simple_bar_figure()
        fig_b = _make_simple_bar_figure()

        # Warm up
        canvas.display_figure(fig_a)
        root.update()

        times = []
        for i in range(10):
            fig = fig_a if i % 2 == 0 else fig_b
            times.append(_time_switch(canvas, fig, root))

        avg = sum(times) / len(times)
        print(f"\n  Simple bar switch: avg={avg*1000:.1f}ms "
              f"min={min(times)*1000:.1f}ms max={max(times)*1000:.1f}ms")
        # Should be under 200ms
        assert avg < 0.5, f"Too slow: {avg*1000:.0f}ms avg"

    def test_line_figure_switch(self, tk_app):
        root, canvas = tk_app
        fig_a = _make_line_figure(8760, 5)
        fig_b = _make_line_figure(8760, 5)

        canvas.display_figure(fig_a)
        root.update()

        times = []
        for i in range(10):
            fig = fig_a if i % 2 == 0 else fig_b
            times.append(_time_switch(canvas, fig, root))

        avg = sum(times) / len(times)
        print(f"\n  Line figure switch (8760x5): avg={avg*1000:.1f}ms "
              f"min={min(times)*1000:.1f}ms max={max(times)*1000:.1f}ms")
        assert avg < 1.0, f"Too slow: {avg*1000:.0f}ms avg"

    def test_complex_figure_switch(self, tk_app):
        root, canvas = tk_app
        fig_a = _make_complex_figure()
        fig_b = _make_complex_figure()

        canvas.display_figure(fig_a)
        root.update()

        times = []
        for i in range(6):
            fig = fig_a if i % 2 == 0 else fig_b
            times.append(_time_switch(canvas, fig, root))

        avg = sum(times) / len(times)
        print(f"\n  Complex figure switch (6 subplots, 8 lines each): "
              f"avg={avg*1000:.1f}ms min={min(times)*1000:.1f}ms "
              f"max={max(times)*1000:.1f}ms")
        assert avg < 2.0, f"Too slow: {avg*1000:.0f}ms avg"

    def test_cached_figure_redisplay(self, tk_app):
        """Switching back to same figure (cache hit) should be fast."""
        root, canvas = tk_app
        fig = _make_line_figure(8760, 5)

        canvas.display_figure(fig)
        root.update()

        times = []
        for _ in range(10):
            times.append(_time_switch(canvas, fig, root))

        avg = sum(times) / len(times)
        print(f"\n  Same figure redisplay: avg={avg*1000:.1f}ms "
              f"min={min(times)*1000:.1f}ms max={max(times)*1000:.1f}ms")
        # Same figure should be very fast (just draw, no setup)
        assert avg < 0.5, f"Too slow: {avg*1000:.0f}ms avg"

    def test_draw_count(self, tk_app):
        """Verify that switching figures only calls draw() once."""
        root, canvas = tk_app
        fig_a = _make_simple_bar_figure()
        fig_b = _make_simple_bar_figure()

        canvas.display_figure(fig_a)
        root.update()

        # Monkey-patch draw to count calls
        draw_count = 0
        orig_draw = canvas._canvas.draw

        def counting_draw():
            nonlocal draw_count
            draw_count += 1
            orig_draw()

        canvas._canvas.draw = counting_draw

        canvas.display_figure(fig_b)
        root.update_idletasks()
        # Process any pending after_idle callbacks
        root.update()

        canvas._canvas.draw = orig_draw

        print(f"\n  Draw count on figure switch: {draw_count}")
        assert draw_count == 1, f"Expected 1 draw, got {draw_count}"
