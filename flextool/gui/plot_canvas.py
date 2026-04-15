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

        # Detect system DPI so matplotlib renders at the correct scale.
        # On Windows with e.g. 150% scaling the system DPI is 144 while
        # matplotlib defaults to 100, making text appear ~30% too small.
        self._system_dpi: float = self._detect_system_dpi()

        # ── Scrollable container ──────────────────────────────────
        # A tk.Canvas with scrollbars wraps the matplotlib widget so
        # that figures larger than the window can be scrolled.
        self._scroll_canvas = tk.Canvas(self, background=_BG, highlightthickness=0)
        self._vscroll = ttk.Scrollbar(self, orient="vertical",
                                       command=self._scroll_canvas.yview)
        self._hscroll = ttk.Scrollbar(self, orient="horizontal",
                                       command=self._scroll_canvas.xview)
        self._scroll_canvas.configure(
            yscrollcommand=self._vscroll.set,
            xscrollcommand=self._hscroll.set,
        )
        self._scroll_canvas.grid(row=0, column=0, sticky="nsew")
        self._vscroll.grid(row=0, column=1, sticky="ns")
        self._hscroll.grid(row=1, column=0, sticky="ew")
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        # Create a blank figure that fills the canvas with a solid bg
        self._figure = Figure(facecolor=_BG)
        self._canvas = FigureCanvasTkAgg(self._figure, master=self._scroll_canvas)
        self._canvas_widget = self._canvas.get_tk_widget()
        self._canvas_widget.configure(background=_BG)
        # Place the matplotlib widget as a window on the scroll canvas
        self._inner_window = self._scroll_canvas.create_window(
            0, 0, window=self._canvas_widget, anchor="nw",
        )

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

        # Mouse-wheel scrolling
        self._scroll_canvas.bind("<Enter>", self._bind_mousewheel)
        self._scroll_canvas.bind("<Leave>", self._unbind_mousewheel)

        # NavigationToolbar2Tk calls pack() in its __init__, so it needs
        # a dedicated container frame managed by grid in the outer layout.
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        self._toolbar = NavigationToolbar2Tk(self._canvas, toolbar_frame)
        self._toolbar.update()

    def _bind_mousewheel(self, event: tk.Event) -> None:
        self._scroll_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self._scroll_canvas.bind_all("<Button-4>", self._on_mousewheel)
        self._scroll_canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self, event: tk.Event) -> None:
        self._scroll_canvas.unbind_all("<MouseWheel>")
        self._scroll_canvas.unbind_all("<Button-4>")
        self._scroll_canvas.unbind_all("<Button-5>")

    @staticmethod
    def _detect_system_dpi() -> float:
        """Return the system DPI, falling back to matplotlib's default.

        On Windows with display scaling (e.g. 150% → 144 DPI) this
        ensures matplotlib renders text at the correct size.  On Linux
        and macOS the system DPI is usually 96 which is close to
        matplotlib's 100 default.
        """
        import sys
        if sys.platform == "win32":
            try:
                import ctypes
                return float(ctypes.windll.user32.GetDpiForSystem())  # type: ignore[attr-defined]
            except Exception:
                pass
        return matplotlib.rcParams.get("figure.dpi", 100)

    def _apply_system_dpi(self, fig: Figure) -> None:
        """Set the figure DPI to the system DPI.

        This rescales the rendering so that point-sized fonts (9pt, 10pt,
        etc.) appear at the correct physical size on high-DPI screens.
        The figure size in inches is preserved; only the pixel count changes.
        """
        if self._system_dpi and fig.get_dpi() != self._system_dpi:
            fig.set_dpi(self._system_dpi)

    def _on_mousewheel(self, event: tk.Event) -> None:
        # Linux uses Button-4/5, Windows/Mac uses MouseWheel
        if event.num == 4:
            self._scroll_canvas.yview_scroll(-3, "units")
        elif event.num == 5:
            self._scroll_canvas.yview_scroll(3, "units")
        elif event.delta:
            self._scroll_canvas.yview_scroll(-event.delta // 120, "units")

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
        """Re-render figure at natural size and update scrollbars."""
        self._resize_pending = None
        w = self._scroll_canvas.winfo_width()
        h = self._scroll_canvas.winfo_height()
        if w < 2 or h < 2:
            return
        self._apply_system_dpi(self._figure)
        self._size_and_draw()
        # Invalidate PNG cache if the size changed significantly
        old_w, old_h = self._last_widget_size
        if abs(w - old_w) > 4 or abs(h - old_h) > 4:
            self._cache.clear()
            self._last_widget_size = (w, h)

    def _size_and_draw(self) -> None:
        """Set the figure to its natural pixel size, update scroll region, draw."""
        dpi = self._figure.get_dpi() or 100
        nat_w, nat_h = self._natural_size_inches
        self._figure.set_size_inches(nat_w, nat_h, forward=False)
        fig_w = int(nat_w * dpi)
        fig_h = int(nat_h * dpi)

        try:
            # Resize the matplotlib widget and its internal PhotoImage
            self._orig_tk_configure(width=fig_w, height=fig_h)
            self._canvas._tkcanvas.delete(self._canvas._tkcanvas_image_region)
            self._canvas._tkphoto.configure(width=fig_w, height=fig_h)
            self._canvas._tkcanvas_image_region = (
                self._canvas._tkcanvas.create_image(
                    0, 0, anchor="nw", image=self._canvas._tkphoto,
                )
            )
            self._scroll_canvas.configure(scrollregion=(0, 0, fig_w, fig_h))
        except (AttributeError, tk.TclError):
            pass

        # Reset the toolbar's nav stack for the new figure
        self._toolbar.update()
        self._cancel_pending_draws()
        self._canvas.draw()
        self._cancel_pending_draws()

        # Show/hide scrollbars based on whether content exceeds viewport
        w = self._scroll_canvas.winfo_width()
        h = self._scroll_canvas.winfo_height()
        self._vscroll.grid() if fig_h > h else self._vscroll.grid_remove()
        self._hscroll.grid() if fig_w > w else self._hscroll.grid_remove()

        # Scroll to top-left
        self._scroll_canvas.xview_moveto(0)
        self._scroll_canvas.yview_moveto(0)

    # ------------------------------------------------------------------
    # Figure display
    # ------------------------------------------------------------------

    def display_figure(self, fig: Figure) -> None:
        """Display a matplotlib Figure at its natural size.

        Scrollbars appear when the figure exceeds the visible area.
        Only one ``draw()`` call is made — no jitter, no redundant renders.
        """
        if fig is self._figure:
            self._canvas.draw()
            return

        fig.set_facecolor(_BG)
        # Remember the figure's designed size for resize handling
        self._natural_size_inches = tuple(fig.get_size_inches())
        # Render at system DPI so fonts match the OS scaling
        self._apply_system_dpi(fig)

        self._figure = fig
        self._canvas.figure = fig
        fig.set_canvas(self._canvas)

        self._size_and_draw()

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

        w_px = self._scroll_canvas.winfo_width()
        h_px = self._scroll_canvas.winfo_height()
        widget_w = max(w_px, 100)
        widget_h = max(h_px, 100)
        self._last_widget_size = (widget_w, widget_h)

        # Display at natural size — scrollbars appear if needed
        disp_w = img_w
        disp_h = img_h

        # Place the image at top-left inside a figure sized to the image.
        # Use system DPI so the image is rendered at the correct scale.
        dpi = self._system_dpi or 100
        fig = Figure(figsize=(disp_w / dpi, disp_h / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img, interpolation="nearest")
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
