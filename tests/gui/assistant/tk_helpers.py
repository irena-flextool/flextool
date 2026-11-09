"""Helpers of the assistant window's Tk tests."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import tkinter as tk
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest


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
    focus is moved there first — again until it is there, because the windows of tests running in other processes
    on the same display take the focus too."""
    deadline = time.monotonic() + 2.0
    while True:
        widget.focus_force()
        widget.update()
        if widget.focus_get() is widget or time.monotonic() >= deadline:
            break
        time.sleep(0.01)
    widget.event_generate(sequence)
    widget.update()


@pytest.fixture
def real_input() -> Iterator[None]:
    """For a test that sends real X input with ``xdotool``: skips without it, and holds a lock per display so that
    such tests in other pytest workers on the same xvfb display do not take the pointer and keyboard grab or
    receive this test's clicks and keys."""
    if shutil.which("xdotool") is None:
        pytest.skip("xdotool is not installed")
    fcntl = pytest.importorskip("fcntl")
    display = os.environ.get("DISPLAY", "").replace(":", "_").replace("/", "_")
    path = Path(tempfile.gettempdir()) / f"flextool-gui-real-input{display}.lock"
    with open(path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def xdotool(root: tk.Misc, *args: object) -> None:
    """Send real X input with ``xdotool`` (a menu's global grab sees only real events, not ``event_generate``)."""
    root.update()
    subprocess.run(["xdotool", *map(str, args)], check=True, timeout=5)
    root.update()


def take_focus(widget: tk.Misc) -> None:
    """Give *widget*'s application the keyboard focus, as when the user works in it (again until it holds: the
    windows of tests in other processes on the same display take the focus too).  A menu posted from a window
    that has no focus gets no keys."""
    deadline = time.monotonic() + 2.0
    while True:
        widget.focus_force()
        widget.update()
        if widget.focus_get() is not None or time.monotonic() >= deadline:
            return
        time.sleep(0.01)


def real_click(root: tk.Misc, x: int, y: int) -> None:
    """A real left click at the screen point (x, y)."""
    xdotool(root, "mousemove", x, y)    # not --sync: it waits for a motion that never comes when already there
    pump(root, timeout=0.1)
    xdotool(root, "click", "1")


def click_on(widget: tk.Misc) -> None:
    """A real left click in the middle of *widget*."""
    widget.update()
    real_click(widget, widget.winfo_rootx() + widget.winfo_width() // 2,
               widget.winfo_rooty() + widget.winfo_height() // 2)


def posted(menu: tk.Menu, timeout: float = 3.0) -> None:
    """Wait until *menu* is on the screen."""
    pump(menu, lambda: bool(menu.winfo_ismapped()), timeout=timeout)


def closes(menu: tk.Menu, timeout: float = 2.0) -> bool:
    """Run the Tk loop until *menu* is off the screen; False when it is still up after *timeout* seconds."""
    try:
        pump(menu, lambda: not menu.winfo_ismapped(), timeout=timeout)
    except AssertionError:
        try:
            menu.unpost()
            menu.grab_release()
        except tk.TclError:
            pass
        return False
    return True
