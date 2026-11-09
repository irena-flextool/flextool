"""Central font metrics + scaling helpers for the FlexTool GUI.

All windows and dialogs should read sizes through ``get_metrics()`` so a
single ``setup_fonts()`` call at startup controls the whole UI.
"""
from __future__ import annotations

from dataclasses import dataclass
import re as _re
import sys
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

    return _maximise_probe(window)


def _set_maximised(window: tk.Misc, on: bool) -> bool:
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


def _maximise_probe(
    window: tk.Misc, *, frame: bool = False
) -> tuple[int, int, int, int] | None:
    """Momentarily maximise *window* and read where the window manager put it.

    The window manager snaps a maximised window to the current monitor's
    usable area (taskbars excluded). Returns the client rectangle, or with
    *frame* the frame rectangle (the title bar and borders measured while
    maximised added back). ``None`` when the window cannot be maximised.
    """
    try:
        window.update_idletasks()
        if not _set_maximised(window, True):
            return None
        window.update_idletasks()
        x = window.winfo_x()
        y = window.winfo_y()
        w = window.winfo_width()
        h = window.winfo_height()
        top = border = 0
        if frame:
            m = _GEOMETRY_RE.match(window.wm_geometry())
            if m and m["x"] is not None and m["xs"] == "+" and m["ys"] == "+":
                border = max(0, window.winfo_rootx() - int(m["x"]))
                top = max(0, window.winfo_rooty() - int(m["y"]))
                x, y = int(m["x"]), int(m["y"])
        _set_maximised(window, False)
        window.update_idletasks()
        if w <= 1 or h <= 1:
            return None
        return (x, y, w + 2 * border, h + top + border)
    except tk.TclError:
        return None


def _intersect(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def _work_area_windows(window: tk.Misc) -> tuple[int, int, int, int] | None:
    """``rcWork`` of the monitor nearest to *window* (Win32 ``GetMonitorInfoW``)."""
    try:
        import ctypes
        from ctypes import wintypes

        class _MonitorInfo(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT),
                ("dwFlags", wintypes.DWORD),
            ]

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        user32.MonitorFromWindow.restype = wintypes.HMONITOR
        user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
        user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.POINTER(_MonitorInfo)]
        hwnd = int(window.winfo_toplevel().wm_frame(), 16)
        monitor = user32.MonitorFromWindow(hwnd, 2)  # MONITOR_DEFAULTTONEAREST
        info = _MonitorInfo()
        info.cbSize = ctypes.sizeof(_MonitorInfo)
        if not monitor or not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
            return None
        r = info.rcWork
        return (int(r.left), int(r.top), int(r.right - r.left), int(r.bottom - r.top))
    except (AttributeError, OSError, ValueError, tk.TclError):
        return None


def _x11_cardinals(name: bytes) -> list[int] | None:
    """The CARDINAL values of the root window property *name* (``_NET_WORKAREA``), read with ``libX11``."""
    try:
        import ctypes
        import ctypes.util

        library = ctypes.util.find_library("X11")
        if not library:
            return None
        x11 = ctypes.CDLL(library)
    except OSError:
        return None
    c_ulong, c_long, c_int = ctypes.c_ulong, ctypes.c_long, ctypes.c_int
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XDefaultRootWindow.restype = c_ulong
    x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
    x11.XInternAtom.restype = c_ulong
    x11.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, c_int]
    x11.XGetWindowProperty.restype = c_int
    x11.XGetWindowProperty.argtypes = [
        ctypes.c_void_p, c_ulong, c_ulong, c_long, c_long, c_int, c_ulong,
        ctypes.POINTER(c_ulong), ctypes.POINTER(c_int), ctypes.POINTER(c_ulong),
        ctypes.POINTER(c_ulong), ctypes.POINTER(ctypes.c_void_p),
    ]
    x11.XFree.argtypes = [ctypes.c_void_p]
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    display = x11.XOpenDisplay(None)
    if not display:
        return None
    try:
        atom = x11.XInternAtom(display, name, 1)
        if not atom:
            return None
        actual_type = c_ulong()
        actual_format = c_int()
        count = c_ulong()
        remaining = c_ulong()
        data = ctypes.c_void_p()
        status = x11.XGetWindowProperty(
            display, x11.XDefaultRootWindow(display), atom, 0, 4096, 0, 6,  # XA_CARDINAL
            ctypes.byref(actual_type), ctypes.byref(actual_format), ctypes.byref(count),
            ctypes.byref(remaining), ctypes.byref(data),
        )
        if status != 0 or not data.value:
            return None
        try:
            if actual_format.value != 32:
                return None
            # Format-32 properties come back as an array of C longs.
            values = ctypes.cast(data, ctypes.POINTER(c_long))
            return [int(values[i]) for i in range(count.value)]
        finally:
            x11.XFree(data)
    finally:
        x11.XCloseDisplay(display)


def _work_area_x11(
    monitor: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    """``_NET_WORKAREA`` of the current desktop, cut to *monitor* (the work area spans all monitors)."""
    values = _x11_cardinals(b"_NET_WORKAREA")
    if not values or len(values) < 4:
        return None
    desktop = (_x11_cardinals(b"_NET_CURRENT_DESKTOP") or [0])[0]
    start = 4 * desktop if len(values) >= 4 * (desktop + 1) else 0
    area = tuple(values[start:start + 4])
    if area[2] <= 1 or area[3] <= 1:
        return None
    if monitor is None:
        return area  # type: ignore[return-value]
    return _intersect(area, monitor)  # type: ignore[arg-type]


def monitor_work_area(window: tk.Misc) -> tuple[int, int, int, int] | None:
    """The work area ``(x, y, w, h)`` of the monitor showing *window*: the
    rectangle a window's *frame* may cover, taskbars, panels and docks
    excluded.

    Windows reads ``rcWork`` of the nearest monitor; X11 reads
    ``_NET_WORKAREA`` from the root window (``libX11`` through ctypes, no
    ``xprop``) and cuts it to the ``screeninfo`` monitor; elsewhere (macOS)
    the maximise probe gives it, converted from the client to the frame
    rectangle; the last resort is the ``screeninfo`` monitor with the
    taskbar margin the Execution window uses (four lines) taken off the
    bottom. ``None`` when nothing is known.
    """
    monitor = _monitor_bounds_screeninfo(window)
    if sys.platform.startswith("win"):
        area = _work_area_windows(window)
        if area is not None:
            return area
    else:
        try:
            system = window.tk.call("tk", "windowingsystem")
        except tk.TclError:
            system = ""
        if system == "x11":
            area = _work_area_x11(monitor)
            if area is not None:
                return area
        elif system == "aqua":
            area = _maximise_probe(window, frame=True)
            if area is not None:
                return area
    if monitor is not None:
        x, y, w, h = monitor
        try:
            margin = get_metrics(window).lh * 4
        except tk.TclError:
            margin = 80
        return (x, y, w, max(1, h - margin))
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


def blend_rgb(base16: tuple[int, int, int], target_hex: str, frac: float) -> str:
    """Blend a 16-bit ``(r, g, b)`` base colour ``frac`` of the way toward
    ``target_hex`` (``#rrggbb``) and return an ``#rrggbb`` string.

    ``base16`` comes from ``winfo_rgb`` (each channel 0–65535), so the
    result tracks the widget's *actual* theme background/foreground:
    blending a near-black dark-mode base toward grey lightens it, while
    blending a near-white light-mode base toward grey darkens it — the
    same call gives the right "touch" of tint in either theme.
    """
    tr = int(target_hex[1:3], 16) * 257
    tg = int(target_hex[3:5], 16) * 257
    tb = int(target_hex[5:7], 16) * 257
    r = int(base16[0] + (tr - base16[0]) * frac)
    g = int(base16[1] + (tg - base16[1]) * frac)
    b = int(base16[2] + (tb - base16[2]) * frac)
    return f"#{r // 256:02x}{g // 256:02x}{b // 256:02x}"


def rgb_hex(base16: tuple[int, int, int]) -> str:
    """A 16-bit ``(r, g, b)`` from ``winfo_rgb`` as ``#rrggbb`` (the target of :func:`blend_rgb`)."""
    return f"#{base16[0] // 256:02x}{base16[1] // 256:02x}{base16[2] // 256:02x}"


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


def scaled_px(widget: tk.Misc, px: int) -> int:
    """*px*, a size in pixels chosen at 96 DPI, at the DPI Tk uses now
    (:func:`monitor_dpi`): twice as many at 192 DPI.  Never below *px*
    (macOS's logical 72 DPI and a lowered ``tk scaling`` keep the 96-DPI
    size)."""
    return max(px, round(px * monitor_dpi(widget) / 96.0))


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
