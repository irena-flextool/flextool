"""Performance benchmarks for PlotCanvas figure switching.

Run with: python3 -m pytest tests/gui/test_plot_canvas_perf.py -v -s
The -s flag is needed to see timing output.
"""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk

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


@pytest.fixture
def tk_app_with_controls():
    """Simulate the ResultViewer layout: control frame + PlotCanvas."""
    try:
        root = tk.Tk()
        root.geometry("1200x800")
        root.update()
    except tk.TclError:
        pytest.skip("No display available")

    right = ttk.Frame(root)
    right.pack(fill="both", expand=True)
    right.columnconfigure(0, weight=1)
    right.rowconfigure(1, weight=1)

    # Control frame with variant buttons (simulates real layout)
    control = ttk.Frame(right, padding=(5, 2))
    control.grid(row=0, column=0, sticky="ew")

    variant_frame = ttk.LabelFrame(control, text="Variant", padding=(2, 1))
    variant_frame.grid(row=0, column=0, sticky="ns")
    buttons = []
    for letter in ["d", "t", "g", "a"]:
        btn = ttk.Button(variant_frame, text=letter, width=3)
        btn.pack(side="left", padx=2, pady=1)
        buttons.append(btn)

    canvas = PlotCanvas(right)
    canvas.grid(row=1, column=0, sticky="nsew")
    root.update()

    yield root, canvas, buttons

    canvas.cleanup()
    root.destroy()


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


def _make_simple_bar_figure() -> Figure:
    """Small bar chart."""
    fig = Figure(figsize=(8, 5), dpi=100)
    ax = fig.add_subplot(111)
    ax.bar(range(10), np.random.rand(10))
    ax.set_title("Simple bar chart")
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
    t0 = time.perf_counter()
    canvas.display_figure(fig)
    root.update_idletasks()
    return time.perf_counter() - t0


def _count_draws(canvas: PlotCanvas, action, root: tk.Tk) -> tuple[float, int]:
    """Run *action*, count how many draw() calls happen, return (time, count)."""
    draw_count = 0
    orig_draw = canvas._canvas.draw

    def counting_draw():
        nonlocal draw_count
        draw_count += 1
        orig_draw()

    canvas._canvas.draw = counting_draw  # type: ignore[assignment]
    t0 = time.perf_counter()
    action()
    root.update_idletasks()
    root.update()  # process any after_idle callbacks
    elapsed = time.perf_counter() - t0
    canvas._canvas.draw = orig_draw  # type: ignore[assignment]
    return elapsed, draw_count


class TestPlotCanvasPerf:
    """Performance benchmarks for figure switching."""

    def test_simple_bar_switch(self, tk_app):
        root, canvas = tk_app
        fig_a = _make_simple_bar_figure()
        fig_b = _make_simple_bar_figure()

        canvas.display_figure(fig_a)
        root.update()

        times = []
        for i in range(10):
            fig = fig_a if i % 2 == 0 else fig_b
            times.append(_time_switch(canvas, fig, root))

        avg = sum(times) / len(times)
        print(f"\n  Simple bar switch: avg={avg*1000:.1f}ms "
              f"min={min(times)*1000:.1f}ms max={max(times)*1000:.1f}ms")

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

    def test_same_figure_redisplay(self, tk_app):
        """Same figure object — just draw(), no setup."""
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

    # ------------------------------------------------------------------
    # Draw count tests — verify exactly 1 draw per switch
    # ------------------------------------------------------------------

    def test_draw_count_plain_switch(self, tk_app):
        """Switch figures without any UI changes — should be 1 draw."""
        root, canvas = tk_app
        fig_a = _make_simple_bar_figure()
        fig_b = _make_simple_bar_figure()

        canvas.display_figure(fig_a)
        root.update()

        elapsed, count = _count_draws(
            canvas, lambda: canvas.display_figure(fig_b), root
        )
        print(f"\n  Plain switch: {count} draw(s) in {elapsed*1000:.1f}ms")
        assert count == 1, f"Expected 1 draw, got {count}"

    def test_draw_count_same_figure(self, tk_app):
        """Redisplay same figure — should be 1 draw."""
        root, canvas = tk_app
        fig = _make_simple_bar_figure()

        canvas.display_figure(fig)
        root.update()

        elapsed, count = _count_draws(
            canvas, lambda: canvas.display_figure(fig), root
        )
        print(f"\n  Same figure: {count} draw(s) in {elapsed*1000:.1f}ms")
        assert count <= 2, f"Expected at most 2 draws, got {count}"

    # ------------------------------------------------------------------
    # Scenario-change vs tree-change comparison
    # ------------------------------------------------------------------

    def test_scenario_vs_tree_change(self, tk_app_with_controls):
        """Compare scenario-change (just swap figure) vs tree-change
        (change button styles + swap figure).

        This mimics the real ResultViewer behavior:
        - Scenario change: just display_figure(new_fig)
        - Tree change: update variant buttons + display_figure(new_fig)
        """
        root, canvas, buttons = tk_app_with_controls
        fig_a = _make_line_figure(8760, 5)
        fig_b = _make_line_figure(8760, 5)

        canvas.display_figure(fig_a)
        root.update()

        # --- Scenario change: just swap figure ---
        scenario_times = []
        scenario_draws = []
        for i in range(10):
            fig = fig_a if i % 2 == 0 else fig_b
            elapsed, count = _count_draws(
                canvas, lambda f=fig: canvas.display_figure(f), root
            )
            scenario_times.append(elapsed)
            scenario_draws.append(count)

        # --- Tree change: change button styles THEN swap figure ---
        tree_times = []
        tree_draws = []
        for i in range(10):
            fig = fig_a if i % 2 == 0 else fig_b
            active_idx = i % len(buttons)

            def tree_action(f=fig, idx=active_idx):
                # Simulate _populate_variant_panel + _highlight_variants
                for j, btn in enumerate(buttons):
                    if j == idx:
                        btn.configure(style="Accent.TButton")
                    else:
                        btn.configure(style="TButton")
                    btn.configure(state="normal" if j < 3 else "disabled")
                canvas.display_figure(f)

            elapsed, count = _count_draws(canvas, tree_action, root)
            tree_times.append(elapsed)
            tree_draws.append(count)

        s_avg = sum(scenario_times) / len(scenario_times)
        t_avg = sum(tree_times) / len(tree_times)

        print(f"\n  Scenario change (no UI):     avg={s_avg*1000:.1f}ms "
              f"draws={scenario_draws}")
        print(f"  Tree change (button styles): avg={t_avg*1000:.1f}ms "
              f"draws={tree_draws}")
        print(f"  Overhead from button styles: {(t_avg-s_avg)*1000:.1f}ms")

        # Both paths should have exactly 1 draw per switch
        assert all(d == 1 for d in scenario_draws), f"Scenario draws: {scenario_draws}"
        assert all(d == 1 for d in tree_draws), f"Tree draws: {tree_draws}"
