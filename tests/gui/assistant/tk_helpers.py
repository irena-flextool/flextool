"""Helpers of the assistant window's Tk tests."""
from __future__ import annotations

import time
import tkinter as tk
from collections.abc import Callable, Iterator


def pump(root: tk.Misc, until: Callable[[], bool] | None = None, timeout: float = 3.0) -> None:
    """Run the Tk loop until *until()* is true (or for *timeout* seconds when it is None); fails on timeout."""
    deadline = time.monotonic() + timeout
    while True:
        root.update()
        if until is not None and until():
            return
        if time.monotonic() >= deadline:
            if until is None:
                return
            raise AssertionError("timed out waiting in the Tk loop")
        time.sleep(0.01)


def widgets(widget: tk.Misc) -> Iterator[tk.Misc]:
    """Every descendant of *widget*."""
    for child in widget.winfo_children():
        yield child
        yield from widgets(child)


def key(widget: tk.Misc, sequence: str) -> None:
    """Send the key event *sequence* to *widget*: Tk delivers key events to the window that has the focus, so the
    focus is moved there first."""
    widget.focus_force()
    widget.update()
    widget.event_generate(sequence)
    widget.update()
