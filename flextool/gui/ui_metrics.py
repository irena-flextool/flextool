"""Central font metrics + scaling helpers for the FlexTool GUI.

All windows and dialogs should read sizes through ``get_metrics()`` so a
single ``setup_fonts()`` call at startup controls the whole UI.
"""
from __future__ import annotations

from dataclasses import dataclass
import re as _re
import tkinter as tk
import tkinter.font as tkfont


_GEOMETRY_RE = _re.compile(
    r"^(?P<w>\d+)x(?P<h>\d+)"
    r"(?:(?P<xs>[+-])(?P<x>\d+)(?P<ys>[+-])(?P<y>\d+))?$"
)


def clamp_geometry(
    saved: str,
    screen_w: int,
    screen_h: int,
    *,
    min_w: int,
    min_h: int,
    margin: int = 40,
) -> str | None:
    """Sanitise a saved Tk geometry string against the current screen.

    Returns ``None`` if *saved* is empty or malformed. Otherwise:

    * Width/height are clamped to ``[min_w, screen_w - margin]`` and
      ``[min_h, screen_h - margin]``.
    * If an ``+X+Y`` offset is present, it is kept only when the window
      would fit fully on-screen; otherwise the offset is dropped and the
      window is placed at ``+0+0`` (the WM picks the position).

    The ``margin`` reserves space for taskbars / docks.
    """
    if not saved:
        return None
    m = _GEOMETRY_RE.match(saved.strip())
    if not m:
        return None
    orig_w = int(m["w"])
    orig_h = int(m["h"])
    w = max(min_w, min(orig_w, max(min_w, screen_w - margin)))
    h = max(min_h, min(orig_h, max(min_h, screen_h - margin)))
    clamped_size = (w != orig_w) or (h != orig_h)
    if m["x"] is None:
        return f"{w}x{h}"
    x = int(m["x"])
    y = int(m["y"])
    if m["xs"] == "-":
        x = -x
    if m["ys"] == "-":
        y = -y
    # Drop the offset if the width/height had to be clamped or the window
    # would run off-screen — saved offsets from a different resolution
    # are not trustworthy in those cases.
    if (
        clamped_size
        or x < 0
        or y < 0
        or x + w > screen_w
        or y + h > screen_h - margin
    ):
        return f"{w}x{h}"
    return f"{w}x{h}+{x}+{y}"


def clamp_sash(
    saved_px: int,
    pane_total: int,
    *,
    min_px: int,
    max_frac: float = 0.85,
) -> int:
    """Keep a saved sash position inside the current pane.

    Returns 0 when *saved_px* is 0 or ``pane_total`` is too small to honour
    ``min_px`` — callers should treat 0 as "leave default".
    """
    if saved_px <= 0 or pane_total <= 0:
        return 0
    upper = int(pane_total * max_frac)
    if upper <= min_px:
        return 0
    return max(min_px, min(saved_px, upper))


def rescale_pixels(saved_px: int, saved_cw: int, current_cw: int) -> int:
    """Rescale a pixel value saved under one font metric to the current one.

    Returns ``saved_px`` unchanged when ``saved_cw`` is 0 (unknown), when
    the ratio is within 10%, or when either cw is non-positive. Otherwise
    returns ``round(saved_px * current_cw / saved_cw)``.

    Used to make saved sash positions and (optionally) window dimensions
    survive a DPI change between sessions.
    """
    if saved_px <= 0 or saved_cw <= 0 or current_cw <= 0:
        return saved_px
    ratio = current_cw / saved_cw
    if 0.90 <= ratio <= 1.10:
        return saved_px
    return round(saved_px * ratio)


def rescale_geometry(saved: str, saved_cw: int, current_cw: int) -> str:
    """Apply rescale_pixels to W and H in a Tk geometry string."""
    if not saved or saved_cw <= 0 or current_cw <= 0 or saved_cw == current_cw:
        return saved
    m = _GEOMETRY_RE.match(saved.strip())
    if not m:
        return saved
    new_w = rescale_pixels(int(m["w"]), saved_cw, current_cw)
    new_h = rescale_pixels(int(m["h"]), saved_cw, current_cw)
    if m["x"] is None:
        return f"{new_w}x{new_h}"
    xs = m["xs"]
    ys = m["ys"]
    return f"{new_w}x{new_h}{xs}{m['x']}{ys}{m['y']}"


@dataclass(frozen=True)
class FontMetrics:
    """Snapshot of derived font metrics. Cheap to recompute, immutable."""
    cw: int            # char width — TkDefaultFont.measure("0")
    lh: int            # line height — TkDefaultFont.metrics("linespace")
    em: int            # alias for cw, exposed for readability in spacing code
    row_height: int    # treeview row height (max(24, int(lh * 1.25)))
    bold_font: tkfont.Font  # body font with weight=bold


def setup_fonts(root: tk.Misc, *, body_pt: int = 10, code_pt: int = 10) -> None:
    """Configure the six standard Tk named fonts with role-aware sizes.

    Sets:
      TkDefaultFont / TkTextFont / TkMenuFont = body_pt
      TkHeadingFont                          = body_pt + 1, bold
      TkTooltipFont                          = body_pt - 1
      TkFixedFont                            = code_pt

    All sizes are POINTS (positive numbers), so tk scaling DPI-scales them.

    Idempotent: safe to call again to restyle the whole UI at runtime.
    """
    # body / text / menu — same size, regular weight
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
        try:
            font = tkfont.nametofont(name)
            font.configure(size=body_pt, weight="normal")
        except tk.TclError:
            pass

    # heading — one point larger, bold
    try:
        heading = tkfont.nametofont("TkHeadingFont")
        heading.configure(size=body_pt + 1, weight="bold")
    except tk.TclError:
        pass

    # tooltip — one point smaller, regular
    try:
        tip = tkfont.nametofont("TkTooltipFont")
        tip.configure(size=max(body_pt - 1, 6), weight="normal")
    except tk.TclError:
        pass

    # monospace / code
    try:
        fixed = tkfont.nametofont("TkFixedFont")
        # keep the platform-default family; only override size
        fixed.configure(size=code_pt)
    except tk.TclError:
        pass


def monitor_dpi(widget: tk.Misc) -> float:
    """Return the current monitor's effective DPI as Tk sees it.

    ``winfo_fpixels('1i')`` returns the pixels-per-inch Tk uses for sizing
    fonts and ttk widgets; it reflects the current value of ``tk scaling``,
    not the OS DPI on the monitor the window happens to be on. Wrapping it
    keeps callers from sprinkling magic strings.

    Note: for a true per-monitor DPI on Windows, the code path would be
    Win32 ``MonitorFromWindow`` + ``GetDpiForMonitor``; that is out of
    scope here. Users can set ``FLEXTOOL_DPI=...`` at startup as an
    override when auto-detection picks the wrong monitor.
    """
    try:
        return float(widget.tk.call("winfo", "fpixels", widget._w, "1i"))
    except (tk.TclError, AttributeError):
        return 96.0


def get_metrics(root: tk.Misc | None = None) -> FontMetrics:
    """Compute current font metrics from TkDefaultFont. Safe to call any time."""
    default = tkfont.nametofont("TkDefaultFont")
    cw = default.measure("0")
    lh = default.metrics("linespace")
    bold = default.copy()
    bold.configure(weight="bold")
    return FontMetrics(
        cw=cw,
        lh=lh,
        em=cw,
        row_height=max(24, int(lh * 1.25)),
        bold_font=bold,
    )
