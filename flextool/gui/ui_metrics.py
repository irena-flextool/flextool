"""Central font metrics + scaling helpers for the FlexTool GUI.

All windows and dialogs should read sizes through ``get_metrics()`` so a
single ``setup_fonts()`` call at startup controls the whole UI.
"""
from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
import tkinter.font as tkfont


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
