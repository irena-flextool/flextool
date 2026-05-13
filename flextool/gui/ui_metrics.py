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
