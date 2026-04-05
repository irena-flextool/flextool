"""Tests for the PlotCanvas widget."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path

import pytest

try:
    import matplotlib
    matplotlib.use("TkAgg")
except Exception:
    pass

from matplotlib.figure import Figure


@pytest.fixture()
def tk_root():
    """Create a withdrawn Tk root; skip if no display is available."""
    try:
        root = tk.Tk()
        root.withdraw()
        yield root
        root.destroy()
    except tk.TclError:
        pytest.skip("No display available")


@pytest.fixture()
def canvas(tk_root):
    from flextool.gui.plot_canvas import PlotCanvas

    pc = PlotCanvas(tk_root)
    pc.pack(fill="both", expand=True)
    tk_root.update_idletasks()
    return pc


class TestPlotCanvasInit:
    """PlotCanvas can be instantiated."""

    def test_creates_widget(self, canvas):
        from flextool.gui.plot_canvas import PlotCanvas

        assert isinstance(canvas, PlotCanvas)

    def test_has_figure(self, canvas):
        assert canvas._figure is not None
        assert isinstance(canvas._figure, Figure)


class TestDisplayPng:
    """display_png loads and shows a PNG file."""

    def test_display_valid_png(self, canvas, tmp_path):
        # Create a small valid PNG using matplotlib
        fig = Figure(figsize=(2, 2))
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1])
        png_path = tmp_path / "test_plot.png"
        fig.savefig(str(png_path))

        canvas.display_png(png_path)
        canvas.master.update_idletasks()

        # After display, the figure should have axes with an image
        axes = canvas._figure.get_axes()
        assert len(axes) == 1
        assert len(axes[0].get_images()) == 1

    def test_display_missing_file(self, canvas, tmp_path):
        missing = tmp_path / "does_not_exist.png"
        canvas.display_png(missing)
        canvas.master.update_idletasks()

        # Should show an error message (text in axes)
        axes = canvas._figure.get_axes()
        assert len(axes) == 1


class TestShowMessage:
    """show_message displays centred text."""

    def test_shows_text(self, canvas):
        canvas.show_message("Hello, world!")
        canvas.master.update_idletasks()

        axes = canvas._figure.get_axes()
        assert len(axes) == 1
        texts = axes[0].texts
        assert any("Hello, world!" in t.get_text() for t in texts)


class TestClear:
    """clear removes all figure content."""

    def test_clear_removes_axes(self, canvas):
        canvas.show_message("Something")
        canvas.master.update_idletasks()
        assert len(canvas._figure.get_axes()) > 0

        canvas.clear()
        canvas.master.update_idletasks()
        assert len(canvas._figure.get_axes()) == 0


class TestDisplayFigure:
    """display_figure replaces the current figure."""

    def test_replaces_figure(self, canvas):
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.plot([1, 2, 3], [4, 5, 6])

        canvas.display_figure(fig)
        canvas.master.update_idletasks()

        assert canvas._figure is fig
