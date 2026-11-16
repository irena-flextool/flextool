"""Posting a context menu at the pointer."""
from __future__ import annotations

import tkinter as tk
from typing import Any


def menu_position(x: int, y: int, width: int, height: int, screen_width: int,
                  screen_height: int) -> tuple[int, int]:
    """The top-left corner for a *width* x *height* menu opened at the pointer (*x*, *y*), never under the pointer.

    One pixel down and right of the pointer; where the menu would cross the right (bottom) screen edge it opens to
    the left of (above) the pointer instead, so Tk does not push it back on screen under the pointer.
    """
    left = x + 1 if x + 1 + width <= screen_width else x - width - 1
    top = y + 1 if y + 1 + height <= screen_height else y - height - 1
    return max(left, 0), max(top, 0)


def popup_at_pointer(menu: tk.Menu, event: Any) -> None:
    """Post *menu* next to the pointer of the click *event* (needs ``x_root`` / ``y_root``), not under it.

    ``tk_popup`` puts the menu's top-left corner at the given point.  With the pointer exactly there, the release of
    a quick right-click lands on the first entry and Tk's ``<ButtonRelease>`` binding of the menu runs it (seen on
    Linux); one pixel off, the release lands outside the menu and the menu stays posted.  Near the bottom-right
    corner of the screen Tk would move the menu back under the pointer, so :func:`menu_position` opens it to the
    left of / above the pointer there.

    No ``grab_release`` afterwards: on X11 the menu's global grab is what closes it on a click elsewhere or Escape
    (a release there kept it open until an entry was chosen, OR-1).
    """
    menu.update_idletasks()
    x, y = menu_position(event.x_root, event.y_root, menu.winfo_reqwidth(), menu.winfo_reqheight(),
                         menu.winfo_screenwidth(), menu.winfo_screenheight())
    menu.tk_popup(x, y)
