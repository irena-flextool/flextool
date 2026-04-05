"""PlotCanvas — embeds a matplotlib FigureCanvasTkAgg with navigation toolbar."""

from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import ttk

try:
    import matplotlib
    if matplotlib.get_backend().lower() != "tkagg":
        matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.image as mpimage
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class PlotCanvas(ttk.Frame):
    """Embeds a matplotlib FigureCanvasTkAgg with navigation toolbar."""

    def __init__(self, master: tk.Widget, **kwargs: object) -> None:
        super().__init__(master, **kwargs)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Create a blank figure
        self._figure = Figure()
        self._canvas = FigureCanvasTkAgg(self._figure, master=self)
        self._canvas_widget = self._canvas.get_tk_widget()
        self._canvas_widget.grid(row=0, column=0, sticky="nsew")

        # NavigationToolbar2Tk calls pack() in its __init__, so it needs
        # a dedicated container frame managed by grid in the outer layout.
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        self._toolbar = NavigationToolbar2Tk(self._canvas, toolbar_frame)
        self._toolbar.update()

    def display_figure(self, fig: Figure) -> None:
        """Display a matplotlib Figure on the canvas."""
        self._figure = fig
        self._canvas.figure = fig
        fig.set_canvas(self._canvas)
        self._canvas.draw_idle()
        self._toolbar.update()

    def display_png(self, png_path: Path) -> None:
        """Load and display a PNG file."""
        try:
            img = mpimage.imread(str(png_path))
        except Exception:
            logger.exception("Failed to load PNG: %s", png_path)
            self.show_message(f"Failed to load image:\n{png_path.name}")
            return

        fig = Figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img)
        ax.set_axis_off()
        fig.set_layout_engine("tight")
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

    def clear(self) -> None:
        """Clear the display."""
        self._figure.clear()
        self._canvas.draw_idle()
