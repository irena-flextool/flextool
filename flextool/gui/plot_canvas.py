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
    """Embeds a matplotlib FigureCanvasTkAgg with navigation toolbar."""

    def __init__(self, master: tk.Widget, **kwargs: object) -> None:
        super().__init__(master, **kwargs)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._raw_line_data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._n_out: int = 3000
        self._cache = PlotCache()
        # Track last widget size so we know when to re-render cached PNGs
        self._last_widget_size: tuple[int, int] = (0, 0)

        # Create a blank figure that fills the canvas with a solid bg
        self._figure = Figure(facecolor=_BG)
        self._canvas = FigureCanvasTkAgg(self._figure, master=self)
        self._canvas_widget = self._canvas.get_tk_widget()
        self._canvas_widget.configure(background=_BG)
        self._canvas_widget.grid(row=0, column=0, sticky="nsew")

        # Prevent the canvas widget from propagating size requests
        # upward — the grid manager controls size, not matplotlib.
        self.grid_propagate(False)

        # NavigationToolbar2Tk calls pack() in its __init__, so it needs
        # a dedicated container frame managed by grid in the outer layout.
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        self._toolbar = NavigationToolbar2Tk(self._canvas, toolbar_frame)
        self._toolbar.update()

    def display_figure(self, fig: Figure) -> None:
        """Display a matplotlib Figure on the canvas.

        The figure is sized to match the canvas widget (with facecolor
        filling unused area).  The <Configure> callback on the Tk canvas
        is suppressed during the draw to prevent a resize feedback loop
        (which manifests as visible jitter / multi-stage resize).
        """
        if fig is self._figure:
            self._canvas.draw()
            return

        # Match figure size to the current canvas widget size.
        fig.set_facecolor(_BG)
        w_px = self._canvas_widget.winfo_width()
        h_px = self._canvas_widget.winfo_height()
        dpi = fig.get_dpi() or 100
        if w_px > 1 and h_px > 1:
            fig.set_size_inches(w_px / dpi, h_px / dpi, forward=False)

        self._figure = fig
        self._canvas.figure = fig
        fig.set_canvas(self._canvas)

        # Suppress all Tk event handlers that trigger resize feedback:
        # <Configure> → resize() → draw_idle() and
        # <Map> → _update_device_pixel_ratio() → configure() → <Configure>
        # Also monkey-patch the widget's configure to ignore width/height
        # changes that draw()/blit() may trigger internally.
        self._canvas_widget.unbind("<Configure>")
        self._canvas_widget.unbind("<Map>")
        orig_configure = self._canvas_widget.configure

        def _frozen_configure(**kw):
            kw.pop("width", None)
            kw.pop("height", None)
            if kw:
                orig_configure(**kw)

        self._canvas_widget.configure = _frozen_configure  # type: ignore[assignment]
        try:
            self._canvas.draw()
        finally:
            self._canvas_widget.configure = orig_configure  # type: ignore[assignment]
            self._canvas_widget.bind("<Configure>", self._canvas.resize)
            self._canvas_widget.bind(
                "<Map>", self._canvas._update_device_pixel_ratio,
            )

    def display_png(self, png_path: Path) -> None:
        """Load and display a PNG file at its natural resolution.

        Uses the cache — switching back to a previously viewed PNG is
        instant.  If the image is larger than the canvas it is scaled
        down (keeping aspect ratio); otherwise shown at 1:1 pixels.
        """
        cache_key = ("png", str(png_path))

        # Check widget size — if it changed, cached figures are stale
        w_px = self._canvas_widget.winfo_width()
        h_px = self._canvas_widget.winfo_height()
        current_size = (max(w_px, 100), max(h_px, 100))
        if current_size != self._last_widget_size:
            # Widget resized — invalidate PNG cache entries
            self._cache.clear()
            self._last_widget_size = current_size

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
        widget_w, widget_h = current_size

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
        """Display a time-series Figure with downsampling support.

        Stores the full-resolution data for each Line2D in every axes,
        replaces line data with a downsampled version, and installs a
        callback so that zoom/pan operations re-downsample on the fly.
        """
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
