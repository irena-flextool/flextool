"""Tests for the PlotCanvas widget."""

from __future__ import annotations

import tkinter as tk

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


class TestWheelRouting:
    """A wheel event scrolls the plot only over the plot (review G19, C7-m7), by the right amount per platform."""

    def test_a_wheel_scroll_in_another_window_leaves_the_plot(self, tk_root):
        from flextool.gui.plot_canvas import PlotCanvas

        plot_window = tk.Toplevel(tk_root)
        plot_window.geometry("300x200+0+0")
        other = tk.Toplevel(tk_root)
        other.geometry("300x200+400+0")
        try:
            pc = PlotCanvas(plot_window)
            pc.pack(fill="both", expand=True)
            text = tk.Text(other)
            text.pack(fill="both", expand=True)
            text.insert("1.0", "line\n" * 200)
            tk_root.update()
            pc._scroll_canvas.configure(scrollregion=(0, 0, 300, 3000))
            pc._bind_mousewheel(None)     # the pointer was over the plot and its <Leave> never came
            tk_root.update()
            before = pc._scroll_canvas.yview()
            assert before[1] < 1.0
            text.event_generate("<Button-5>", x=20, y=20)
            tk_root.update()
            assert pc._scroll_canvas.yview() == before
            assert text.yview()[0] > 0.0                 # the other window's own scroll still works
            pc._scroll_canvas.event_generate("<Button-5>", x=20, y=20)
            tk_root.update()
            assert pc._scroll_canvas.yview()[0] > before[0]
        finally:
            other.destroy()
            plot_window.destroy()

    @pytest.mark.parametrize(("system", "delta", "units"), [
        ("win32", -30, 1), ("win32", 30, -1), ("win32", 240, -2), ("win32", -120, 1),
        ("aqua", -1, 1), ("aqua", 2, -2),
    ])
    def test_wheel_deltas_scroll_at_least_one_unit(self, canvas, monkeypatch, system, delta, units):
        from types import SimpleNamespace

        from flextool.gui import plot_canvas as module

        monkeypatch.setattr(module, "_windowing_system", lambda _widget: system, raising=False)
        monkeypatch.setattr(canvas, "_pointer_inside", lambda _event: True, raising=False)
        canvas._scroll_canvas.configure(scrollregion=(0, 0, 100, 5000))
        scrolled: list[int] = []
        monkeypatch.setattr(canvas._scroll_canvas, "yview_scroll", lambda n, _what: scrolled.append(n))
        canvas._on_mousewheel(SimpleNamespace(num="??", delta=delta, x_root=0, y_root=0))
        assert scrolled == [units]
