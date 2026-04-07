"""PlotCanvas — embeds a matplotlib FigureCanvasTkAgg with navigation toolbar."""

from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np

from flextool.gui.downsampling import downsample_for_display
from flextool.gui.plot_cache import PlotCache

try:
    import matplotlib
    if matplotlib.get_backend().lower() != "tkagg":
        matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.image as mpimage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Background color used to fill unused canvas area
_BG = "#f0f0f0"


class PlotCanvas(ttk.Frame):
    """Embeds a matplotlib FigureCanvasTkAgg with navigation toolbar.

    Matplotlib's own ``<Configure>`` handler is permanently disconnected.
    Instead, this class manages figure sizing itself via a ``<Configure>``
    binding on the *PlotCanvas frame*, which only fires on real window
    resizes — not on internal button/style changes.  This eliminates
    jitter from the resize → draw → resize feedback loop.
    """

    def __init__(self, master: tk.Widget, **kwargs: object) -> None:
        super().__init__(master, **kwargs)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._raw_line_data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._n_out: int = 3000
        self._cache = PlotCache()
        self._last_widget_size: tuple[int, int] = (0, 0)
        self._resize_pending: str | None = None
        self._natural_size_inches: tuple[float, float] = (6.0, 4.0)

        # Create a blank figure that fills the canvas with a solid bg
        self._figure = Figure(facecolor=_BG)
        self._canvas = FigureCanvasTkAgg(self._figure, master=self)
        self._canvas_widget = self._canvas.get_tk_widget()
        self._canvas_widget.configure(background=_BG)
        self._canvas_widget.grid(row=0, column=0, sticky="nsew")

        # ── Permanently disconnect matplotlib's resize handling ──
        # This prevents the <Configure> → resize() → draw_idle() →
        # <Configure> feedback loop that causes jitter.
        self._canvas_widget.unbind("<Configure>")
        self._canvas_widget.unbind("<Map>")

        # Instead, handle resizes ourselves on the PlotCanvas frame.
        # This only fires on real geometry changes (window resize,
        # paned-window sash), not on internal button style changes.
        self.bind("<Configure>", self._on_frame_configure)

        # Monkey-patch the canvas widget's configure to silently ignore
        # width/height changes from matplotlib internals (draw/blit).
        self._orig_tk_configure = self._canvas_widget.configure

        def _no_resize_configure(**kw):
            kw.pop("width", None)
            kw.pop("height", None)
            if kw:
                self._orig_tk_configure(**kw)

        self._canvas_widget.configure = _no_resize_configure  # type: ignore[assignment]

        # NavigationToolbar2Tk calls pack() in its __init__, so it needs
        # a dedicated container frame managed by grid in the outer layout.
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        self._toolbar = NavigationToolbar2Tk(self._canvas, toolbar_frame)
        self._toolbar.update()

    # ------------------------------------------------------------------
    # Resize handling (replaces matplotlib's <Configure> handler)
    # ------------------------------------------------------------------

    def _on_frame_configure(self, event: tk.Event) -> None:
        """Handle real geometry changes with debouncing.

        Schedules a single redraw after 50ms of no further resize events,
        so dragging a window edge doesn't trigger dozens of redraws.
        """
        if self._resize_pending is not None:
            self.after_cancel(self._resize_pending)
        self._resize_pending = self.after(50, self._do_resize)

    def _do_resize(self) -> None:
        """Adapt the current figure to the new widget size and redraw."""
        self._resize_pending = None
        w = self._canvas_widget.winfo_width()
        h = self._canvas_widget.winfo_height()
        if w < 2 or h < 2:
            return

        dpi = self._figure.get_dpi() or 100
        nat_w, nat_h = self._natural_size_inches
        # Scale down if figure exceeds canvas, preserving aspect ratio
        scale = min(w / (nat_w * dpi), h / (nat_h * dpi), 1.0)
        self._figure.set_size_inches(nat_w * scale, nat_h * scale, forward=False)

        # Recreate the internal PhotoImage at the new pixel size
        # (matplotlib's resize handler normally does this)
        try:
            self._canvas._tkcanvas.delete(self._canvas._tkcanvas_image_region)
            self._canvas._tkphoto.configure(width=w, height=h)
            self._canvas._tkcanvas_image_region = (
                self._canvas._tkcanvas.create_image(
                    w // 2, h // 2, image=self._canvas._tkphoto
                )
            )
        except (AttributeError, tk.TclError):
            pass

        self._cancel_pending_draws()
        self._canvas.draw()
        self._cancel_pending_draws()

        # Invalidate PNG cache if the size changed significantly
        old_w, old_h = self._last_widget_size
        if abs(w - old_w) > 4 or abs(h - old_h) > 4:
            self._cache.clear()
            self._last_widget_size = (w, h)

    # ------------------------------------------------------------------
    # Figure display
    # ------------------------------------------------------------------

    def display_figure(self, fig: Figure) -> None:
        """Display a matplotlib Figure on the canvas.

        Respects the figure's original size.  If the figure is larger
        than the canvas, it is scaled down (preserving aspect ratio).
        Only one ``draw()`` call is made — no jitter, no redundant renders.
        """
        if fig is self._figure:
            self._canvas.draw()
            return

        fig.set_facecolor(_BG)
        # Remember the figure's designed size for resize handling
        self._natural_size_inches = tuple(fig.get_size_inches())

        dpi = fig.get_dpi() or 100
        nat_w, nat_h = self._natural_size_inches
        w_px = self._canvas_widget.winfo_width()
        h_px = self._canvas_widget.winfo_height()
        if w_px > 1 and h_px > 1:
            # Scale down if figure exceeds canvas, preserving aspect ratio
            scale = min(w_px / (nat_w * dpi), h_px / (nat_h * dpi), 1.0)
            if scale < 1.0:
                fig.set_size_inches(nat_w * scale, nat_h * scale, forward=False)

        self._figure = fig
        self._canvas.figure = fig
        fig.set_canvas(self._canvas)

        # Clear the tk canvas so leftovers from a previous (larger) figure
        # don't remain visible behind a smaller new figure.
        try:
            tk_canvas = self._canvas._tkcanvas
            w = tk_canvas.winfo_width()
            h = tk_canvas.winfo_height()
            tk_canvas.delete("bg_rect")
            tk_canvas.create_rectangle(0, 0, w, h, fill=_BG, outline="", tags="bg_rect")
        except (AttributeError, tk.TclError):
            pass

        # Reset the toolbar's nav stack for the new figure so that
        # zoom/pan/home/back/forward work correctly.
        self._toolbar.update()

        # Cancel any pending idle draws from set_canvas()/toolbar.update()
        self._cancel_pending_draws()
        self._canvas.draw()
        self._cancel_pending_draws()

    def _cancel_pending_draws(self) -> None:
        """Cancel any ``after_idle(draw)`` scheduled by ``draw_idle()``."""
        idle_id = getattr(self._canvas, "_idle_draw_id", None)
        if idle_id is not None:
            self._canvas_widget.after_cancel(idle_id)
            self._canvas._idle_draw_id = None

    def display_png(self, png_path: Path) -> None:
        """Load and display a PNG file at its natural resolution.

        Uses the cache — switching back to a previously viewed PNG is
        instant.  If the image is larger than the canvas it is scaled
        down (keeping aspect ratio); otherwise shown at 1:1 pixels.
        """
        cache_key = ("png", str(png_path))

        cached = self._cache.get(cache_key)
        if cached is not None:
            self.display_figure(cached)
            return

        try:
            img = mpimage.imread(str(png_path))
        except Exception:
            logger.exception("Failed to load PNG: %s", png_path)
            self.show_message(f"Failed to load image:\n{png_path.name}")
            return

        img_h, img_w = img.shape[:2]

        w_px = self._canvas_widget.winfo_width()
        h_px = self._canvas_widget.winfo_height()
        widget_w = max(w_px, 100)
        widget_h = max(h_px, 100)
        self._last_widget_size = (widget_w, widget_h)

        # Scale down if image exceeds widget, preserving aspect ratio
        scale = min(widget_w / img_w, widget_h / img_h, 1.0)
        disp_w = img_w * scale
        disp_h = img_h * scale

        # Place the image centered inside a figure that fills the widget
        dpi = 100
        fig = Figure(figsize=(widget_w / dpi, widget_h / dpi), dpi=dpi)
        ax_x = (widget_w - disp_w) / (2 * widget_w)
        ax_y = (widget_h - disp_h) / (2 * widget_h)
        ax_w = disp_w / widget_w
        ax_h = disp_h / widget_h
        ax = fig.add_axes([ax_x, ax_y, ax_w, ax_h])
        ax.imshow(img, interpolation="lanczos" if scale < 1.0 else "nearest")
        ax.set_axis_off()

        self._cache.put(cache_key, fig)
        self.display_figure(fig)

    def show_message(self, text: str) -> None:
        """Display a text message (e.g., 'No data available')."""
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.text(
            0.5, 0.5, text,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=14, color="grey",
        )
        ax.set_axis_off()
        self.display_figure(fig)

    # ------------------------------------------------------------------
    # Downsampling for time series
    # ------------------------------------------------------------------

    def display_timeseries_figure(
        self, fig: Figure, n_out: int = 3000
    ) -> None:
        """Display a time-series Figure with downsampling support."""
        self._raw_line_data = {}
        self._n_out = n_out

        for ax_idx, ax in enumerate(fig.axes):
            for line_idx, line in enumerate(ax.get_lines()):
                x_full = np.asarray(line.get_xdata(), dtype=np.float64)
                y_full = np.asarray(line.get_ydata(), dtype=np.float64)
                self._raw_line_data[(ax_idx, line_idx)] = (x_full, y_full)

                x_ds, y_ds = downsample_for_display(x_full, y_full, n_out)
                line.set_xdata(x_ds)
                line.set_ydata(y_ds)

            ax.callbacks.connect("xlim_changed", self._on_xlim_changed)

        self.display_figure(fig)

    def _on_xlim_changed(self, ax) -> None:  # type: ignore[override]
        """Re-downsample visible data when the user zooms or pans."""
        if not self._raw_line_data:
            return

        try:
            ax_idx = list(self._figure.axes).index(ax)
        except ValueError:
            return

        lo, hi = ax.get_xlim()

        for line_idx, line in enumerate(ax.get_lines()):
            key = (ax_idx, line_idx)
            if key not in self._raw_line_data:
                continue
            x_full, y_full = self._raw_line_data[key]

            mask = (x_full >= lo) & (x_full <= hi)
            x_vis = x_full[mask]
            y_vis = y_full[mask]

            if len(x_vis) == 0:
                continue

            x_ds, y_ds = downsample_for_display(x_vis, y_vis, self._n_out)
            line.set_xdata(x_ds)
            line.set_ydata(y_ds)

        self._canvas.draw_idle()

    def cleanup(self) -> None:
        """Release matplotlib resources held by this canvas."""
        self._raw_line_data.clear()
        self._cache.clear()

    def clear(self) -> None:
        """Clear the display."""
        self._raw_line_data = {}
        self._figure.clear()
        self._canvas.draw_idle()
