"""Reusable hover tooltip for Tkinter widgets.

Attach a tooltip to any widget with :func:`attach_tooltip`.  The tooltip
appears after a short delay when the mouse hovers over the widget and
hides when the mouse leaves.
"""
from __future__ import annotations

import tkinter as tk


class HoverTooltip:
    """Manages a single tooltip window for a widget."""

    _DELAY_MS = 400  # delay before showing

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self._widget = widget
        self._text = text
        self._tip: tk.Toplevel | None = None
        self._after_id: str | None = None

        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")

    def _on_enter(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        self._cancel()
        self._after_id = self._widget.after(self._DELAY_MS, self._show)

    def _on_leave(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        self._cancel()
        self._hide()

    def _cancel(self) -> None:
        if self._after_id is not None:
            self._widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self) -> None:
        self._after_id = None
        if self._tip is not None:
            return

        x = self._widget.winfo_rootx() + 10
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4

        self._tip = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_attributes("-topmost", True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tw,
            text=self._text,
            justify="left",
            background="#333333",
            foreground="#ffffff",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=6,
        )
        label.pack()

    def _hide(self) -> None:
        if self._tip is not None:
            try:
                self._tip.destroy()
            except tk.TclError:
                pass
            self._tip = None


def attach_tooltip(widget: tk.Widget, text: str) -> HoverTooltip:
    """Attach a hover tooltip to *widget* showing *text*."""
    return HoverTooltip(widget, text)
