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


def monitor_signature(window: tk.Misc | None = None) -> str:
    """Stable key identifying the current monitor *configuration*.

    Used to remember window placements per monitor layout: a window
    dragged onto the 4K while docked, then the laptop undocked, should not
    have its docked offset reused on the single internal panel. The
    signature changes whenever a monitor is added/removed/moved/resized, so
    each configuration keeps its own saved geometry.

    Built from the sorted ``WxH+X+Y`` rectangles reported by ``screeninfo``
    (order-independent). Falls back to ``virtual:<W>x<H>`` from the Tk
    virtual-desktop size when ``screeninfo`` is unavailable, and finally to
    ``"virtual:unknown"`` so callers always get a usable, hashable key.
    """
    try:
        from screeninfo import get_monitors
        monitors = list(get_monitors())
        if monitors:
            parts = sorted(
                f"{int(m.width)}x{int(m.height)}+{int(m.x)}+{int(m.y)}"
                for m in monitors
            )
            return "|".join(parts)
    except Exception:
        pass
    if window is not None:
        try:
            return f"virtual:{window.winfo_screenwidth()}x{window.winfo_screenheight()}"
        except tk.TclError:
            pass
    return "virtual:unknown"


def primary_monitor_bounds() -> tuple[int, int, int, int] | None:
    """Bounds ``(x, y, w, h)`` of the primary monitor, or ``None``.

    Used to place a window that has no prior position to detect a "current"
    monitor from (e.g. the main window at first launch). Prefers the monitor
    ``screeninfo`` flags ``is_primary``; otherwise the first enumerated one.
    Returns ``None`` when ``screeninfo`` is unavailable or enumerates none,
    so callers fall back to virtual-desktop sizing.
    """
    try:
        from screeninfo import get_monitors
        monitors = list(get_monitors())
    except Exception:
        return None
    if not monitors:
        return None
    chosen = next((m for m in monitors if getattr(m, "is_primary", False)), monitors[0])
    if chosen.width <= 1 or chosen.height <= 1:
        return None
    return (int(chosen.x), int(chosen.y), int(chosen.width), int(chosen.height))


def _monitor_bounds_screeninfo(
    window: tk.Misc,
) -> tuple[int, int, int, int] | None:
    """Per-monitor rectangle via the ``screeninfo`` package (flicker-free).

    Picks the monitor whose rectangle contains the centre of *window*; if
    no monitor claims the centre (e.g. the window straddles a gap), falls
    back to the primary monitor, then the first enumerated one. Returns
    ``None`` when ``screeninfo`` is missing or enumerates no monitors, so
    the caller can drop to the maximise-probe.
    """
    try:
        from screeninfo import get_monitors
    except Exception:
        return None
    try:
        monitors = list(get_monitors())
    except Exception:
        # screeninfo raises on headless / unusual display backends.
        return None
    if not monitors:
        return None
    try:
        window.update_idletasks()
        cx = window.winfo_x() + window.winfo_width() // 2
        cy = window.winfo_y() + window.winfo_height() // 2
    except tk.TclError:
        cx = cy = None
    chosen = None
    if cx is not None:
        for m in monitors:
            if m.x <= cx < m.x + m.width and m.y <= cy < m.y + m.height:
                chosen = m
                break
    if chosen is None:
        chosen = next((m for m in monitors if getattr(m, "is_primary", False)), None)
    if chosen is None:
        chosen = monitors[0]
    if chosen.width <= 1 or chosen.height <= 1:
        return None
    return (int(chosen.x), int(chosen.y), int(chosen.width), int(chosen.height))


def current_monitor_bounds(window: tk.Misc) -> tuple[int, int, int, int] | None:
    """Best-effort bounds ``(x, y, w, h)`` of the monitor showing *window*.

    Tk exposes no per-monitor geometry: ``winfo_screenwidth`` /
    ``winfo_screenheight`` report the FULL virtual desktop spanning *every*
    monitor, so sizing a window to those values stretches it across all
    screens (the dual-monitor "window covers both displays" bug).

    Prefers the ``screeninfo`` package (flicker-free, also yields the
    monitor's ``x``/``y`` origin). When it is unavailable or enumerates no
    monitors, falls back to momentarily maximising *window* — the window
    manager snaps a maximised window to the *current* monitor's usable area
    — then reading the resulting offset/size and restoring normal state.

    *window* should already be positioned (e.g. via ``geometry("+x+y")``) on
    the target monitor before calling, so both paths pick the right display.
    Returns ``None`` when every approach fails, so callers fall back to
    virtual-desktop sizing (no worse than before).

    Note: ``screeninfo`` reports the full monitor rectangle, not the WM work
    area, so callers should still reserve taskbar/dock space themselves; the
    maximise-probe path already excludes it.
    """
    bounds = _monitor_bounds_screeninfo(window)
    if bounds is not None:
        return bounds

    def _set_maximised(on: bool) -> bool:
        # Windows / macOS use state('zoomed'); most Linux WMs honour the
        # '-zoomed' attribute. Try both; report whether either took.
        try:
            window.tk.call("wm", "state", window._w, "zoomed" if on else "normal")
            return True
        except tk.TclError:
            pass
        try:
            window.tk.call("wm", "attributes", window._w, "-zoomed", "1" if on else "0")
            return True
        except tk.TclError:
            return False

    try:
        window.update_idletasks()
        if not _set_maximised(True):
            return None
        window.update_idletasks()
        x = window.winfo_x()
        y = window.winfo_y()
        w = window.winfo_width()
        h = window.winfo_height()
        _set_maximised(False)
        window.update_idletasks()
        if w <= 1 or h <= 1:
            return None
        return (x, y, w, h)
    except tk.TclError:
        return None


def resolve_saved_geometry(
    geom_map: dict[str, str],
    signature: str,
    saved_cw: int,
    current_cw: int,
    screen_w: int,
    screen_h: int,
    *,
    min_w: int,
    min_h: int,
) -> str | None:
    """Pick the geometry saved for *signature* and make it safe to apply.

    Looks up *geom_map* by the current monitor *signature*, falling back to
    a ``"legacy"`` entry migrated from the old single-string format. The
    result is rescaled for any font/DPI change (``rescale_geometry``) and
    clamped to the current screen (``clamp_geometry``). Returns ``None``
    when nothing usable is saved, so callers keep the default placement.
    """
    if not geom_map:
        return None
    saved = geom_map.get(signature) or geom_map.get("legacy", "")
    if not saved:
        return None
    geom = rescale_geometry(saved, saved_cw, current_cw)
    return clamp_geometry(geom, screen_w, screen_h, min_w=min_w, min_h=min_h)


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
      TkHeadingFont                          = body_pt + 2, bold
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

    # heading — one point larger, bold. Bold weight already reads as a
    # heading next to the regular body; +2 looked oversized on Windows
    # (esp. the "Input sources" / "File outputs" section labels), so +1 is
    # enough lift without dominating the surrounding sv_ttk widgets.
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
    # Treeview row height must track the font the tree ACTUALLY renders cell
    # text with — under sv_ttk that's SunValleyBodyFont, whose linespace runs
    # ~20% tighter than TkDefaultFont's at the same visual size. Deriving the
    # row height from TkDefaultFont (the old code) over-padded every row, so
    # rows looked sparse at all DPIs. A flat 1.3× of the real linespace gives
    # a comfortable click target without the gaps; the old max(24, …) px floor
    # is dropped because a fixed pixel doesn't scale and only inflated rows at
    # 96 DPI.
    tree_lh = lh
    try:
        import tkinter.ttk as ttk
        # lookup() returns a Tcl_Obj, not a str; str() yields the clean font
        # name ("SunValleyBodyFont"), whereas passing the Tcl_Obj straight to
        # nametofont raises "named font … does not already exist".
        tree_font_name = str(ttk.Style().lookup("Treeview", "font") or "")
        if tree_font_name:
            tree_lh = tkfont.nametofont(tree_font_name).metrics("linespace")
    except (tk.TclError, RuntimeError):
        pass
    return FontMetrics(
        cw=cw,
        lh=lh,
        em=cw,
        row_height=max(18, round(tree_lh * 1.3)),
        bold_font=bold,
    )
