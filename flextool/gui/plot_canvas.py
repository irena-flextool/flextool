"""PlotCanvas — embeds a matplotlib FigureCanvasTkAgg with navigation toolbar."""

from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np

from flextool.gui.downsampling import downsample_for_display

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


class PlotCanvas(ttk.Frame):
    """Embeds a matplotlib FigureCanvasTkAgg with navigation toolbar."""

    def __init__(self, master: tk.Widget, **kwargs: object) -> None:
        super().__init__(master, **kwargs)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._raw_line_data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._n_out: int = 3000

        # Create a blank figure
        self._figure = Figure()
        self._canvas = FigureCanvasTkAgg(self._figure, master=self)
        self._canvas_widget = self._canvas.get_tk_widget()
        self._canvas_widget.configure(background="#f0f0f0")
        self._canvas_widget.grid(row=0, column=0, sticky="nsew")

        # NavigationToolbar2Tk calls pack() in its __init__, so it needs
        # a dedicated container frame managed by grid in the outer layout.
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        self._toolbar = NavigationToolbar2Tk(self._canvas, toolbar_frame)
        self._toolbar.update()

    def display_figure(self, fig: Figure) -> None:
        """Display a matplotlib Figure on the canvas."""
        # Clear old figure first so remnants don't remain when switching
        # to a smaller plot.
        old_fig = self._figure
        if old_fig is not fig:
            old_fig.clear()

        self._figure = fig
        self._canvas.figure = fig
        fig.set_canvas(self._canvas)
        self._canvas.draw()  # force immediate redraw (not draw_idle)
        self._toolbar.update()

    def display_png(self, png_path: Path) -> None:
        """Load and display a PNG file at its natural resolution.

        If the image is larger than the available widget area it is scaled
        down (keeping aspect ratio).  Otherwise it is shown at 1:1 pixels.
        """
        try:
            img = mpimage.imread(str(png_path))
        except Exception:
            logger.exception("Failed to load PNG: %s", png_path)
            self.show_message(f"Failed to load image:\n{png_path.name}")
            return

        img_h, img_w = img.shape[:2]  # pixels

        # Determine available widget area in pixels
        self.update_idletasks()
        widget_w = self._canvas_widget.winfo_width()
        widget_h = self._canvas_widget.winfo_height()
        if widget_w < 10 or widget_h < 10:
            # Widget not yet laid out — use reasonable defaults
            widget_w = max(widget_w, 800)
            widget_h = max(widget_h, 600)

        # Scale down if image exceeds widget, preserving aspect ratio
        scale = min(widget_w / img_w, widget_h / img_h, 1.0)
        disp_w = img_w * scale
        disp_h = img_h * scale

        dpi = 100
        fig = Figure(figsize=(disp_w / dpi, disp_h / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img, interpolation="lanczos" if scale < 1.0 else "nearest")
        ax.set_axis_off()
        self.display_figure(fig)

    def show_message(self, text: str) -> None:
        """Display a text message (e.g., 'No data available')."""
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="grey",
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

                # Initial downsample over the full range
                x_ds, y_ds = downsample_for_display(x_full, y_full, n_out)
                line.set_xdata(x_ds)
                line.set_ydata(y_ds)

            ax.callbacks.connect("xlim_changed", self._on_xlim_changed)

        self.display_figure(fig)

    def _on_xlim_changed(self, ax) -> None:  # type: ignore[override]
        """Re-downsample visible data when the user zooms or pans."""
        if not hasattr(self, "_raw_line_data"):
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

            # Slice to the visible range (with small margin)
            mask = (x_full >= lo) & (x_full <= hi)
            x_vis = x_full[mask]
            y_vis = y_full[mask]

            if len(x_vis) == 0:
                continue

            x_ds, y_ds = downsample_for_display(x_vis, y_vis, self._n_out)
            line.set_xdata(x_ds)
            line.set_ydata(y_ds)

        # Redraw without triggering the callback again
        self._canvas.draw_idle()

    def cleanup(self) -> None:
        """Release matplotlib resources held by this canvas."""
        self._raw_line_data.clear()
        try:
            plt.close(self._figure)
        except Exception:  # noqa: BLE001
            pass

    def clear(self) -> None:
        """Clear the display."""
        self._raw_line_data = {}
        self._figure.clear()
        self._canvas.draw_idle()
