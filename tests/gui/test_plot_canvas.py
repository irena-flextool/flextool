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


class TestFigureDpiFollowsTreeFont:
    """The figure DPI sizes the plot text to ~1.1 x the plot tree's font em, whatever DPI the X server reports.

    ``tk scaling`` stands in for the screen DPI (Tk derives ``winfo_fpixels('1i')`` from it), so one display
    covers 96, 144 and 208 DPI.  sv_ttk's fonts are given in pixels (-14), so their em does not follow the DPI.
    """

    @staticmethod
    def _set_tree_font(root, size):
        import tkinter.font as tkfont
        from tkinter import ttk

        font = tkfont.Font(root=root, name="TestPlotTreeFont", family="DejaVu Sans", size=size)
        ttk.Style(root).configure("PlotTree.Treeview", font=str(font))
        return font

    @pytest.mark.parametrize("screen_dpi", [96, 144, 208])
    @pytest.mark.parametrize("font_size", [-14, 10])
    def test_plot_text_is_one_size_bigger_than_the_tree(self, tk_root, tmp_path, screen_dpi, font_size):
        from flextool.gui.plot_canvas import PlotCanvas

        tk_root.tk.call("tk", "scaling", screen_dpi / 72.0)
        assert abs(tk_root.winfo_fpixels("1i") - screen_dpi) < 0.5
        font = self._set_tree_font(tk_root, font_size)  # noqa: F841 — keep the named font alive
        # Tk keeps the scaling as whole screen millimetres: read back what it took.
        em_px = -font_size if font_size < 0 else font_size * float(tk_root.tk.call("tk", "scaling"))

        pc = PlotCanvas(tk_root)
        pc.pack(fill="both", expand=True)
        tk_root.update_idletasks()
        default_dpi = matplotlib.rcParams["figure.dpi"]

        fig = Figure(figsize=(4.0, 3.0))
        text = fig.text(0.5, 0.5, "Hg", fontsize=matplotlib.rcParams["font.size"])
        pc.display_figure(fig)

        dpi = fig.get_dpi()
        # The plot text's em on the canvas is 1.1 x the tree font's em ...
        assert matplotlib.rcParams["font.size"] * dpi / 72.0 == pytest.approx(1.1 * em_px, rel=1e-9)
        # ... and its drawn height (the "Hg" ink box, ~0.94 em in whole pixels) matches that, not the
        # ~2x of the X server DPI (29 px at 208 DPI against a 14 px tree font before the fix).
        drawn = text.get_window_extent(pc._canvas.get_renderer()).height
        assert abs(drawn - 1.1 * em_px) <= max(1.0, 0.1 * em_px)
        # The figure keeps its inches; its pixel size scales with the DPI.
        assert tuple(fig.get_size_inches()) == pytest.approx((4.0, 3.0))
        assert int(pc._canvas_widget.cget("width")) == int(4.0 * dpi)
        assert int(pc._canvas_widget.cget("height")) == int(3.0 * dpi)
        # Nothing global changed, and a saved PNG keeps the DPI it is saved with.
        assert matplotlib.rcParams["figure.dpi"] == default_dpi
        png = tmp_path / "saved.png"
        fig.savefig(png, dpi=150)
        import matplotlib.image as mpimage

        h, w = mpimage.imread(str(png)).shape[:2]
        assert (w, h) == (600, 450)

    def test_dpi_follows_a_font_change_on_resize(self, tk_root):
        from flextool.gui.plot_canvas import PlotCanvas

        tk_root.tk.call("tk", "scaling", 96 / 72.0)
        font = self._set_tree_font(tk_root, -14)
        pc = PlotCanvas(tk_root)
        pc.pack(fill="both", expand=True)
        tk_root.geometry("400x300")
        tk_root.deiconify()  # a withdrawn window has no size, and _do_resize skips it
        tk_root.update()
        fig = Figure(figsize=(4.0, 3.0))
        pc.display_figure(fig)
        assert fig.get_dpi() == pytest.approx(72.0 * 14 * 1.1 / matplotlib.rcParams["font.size"])

        font.configure(size=-20)
        pc._do_resize()
        assert fig.get_dpi() == pytest.approx(72.0 * 20 * 1.1 / matplotlib.rcParams["font.size"])

    def test_png_is_shown_one_to_one_after_a_dpi_change(self, tk_root, tmp_path):
        from flextool.gui.plot_canvas import PlotCanvas

        font = self._set_tree_font(tk_root, -14)
        pc = PlotCanvas(tk_root)
        pc.pack(fill="both", expand=True)
        tk_root.geometry("400x300")
        tk_root.deiconify()  # a withdrawn window has no size, and _do_resize skips it
        tk_root.update()
        src = Figure(figsize=(3.0, 2.0), dpi=100)
        src.add_subplot(111).plot([0, 1], [0, 1])
        png = tmp_path / "p.png"
        src.savefig(png, dpi=100)

        pc.display_png(png)
        assert int(pc._canvas_widget.cget("width")) == 300
        assert int(pc._canvas_widget.cget("height")) == 200
        font.configure(size=-22)
        pc._do_resize()
        assert int(pc._canvas_widget.cget("width")) == 300
        assert int(pc._canvas_widget.cget("height")) == 200
