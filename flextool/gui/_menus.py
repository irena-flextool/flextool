"""Posting a context menu at the pointer."""
from __future__ import annotations

import tkinter as tk
from typing import Any


def popup_at_pointer(menu: tk.Menu, event: Any) -> None:
    """Post *menu* one pixel down and right of the pointer of the click *event* (needs ``x_root`` / ``y_root``).

    ``tk_popup`` puts the menu's top-left corner at the given point.  With the pointer exactly there, the release of
    a quick right-click lands on the first entry and Tk's ``<ButtonRelease>`` binding of the menu runs it (seen on
    Linux); one pixel off, the release lands outside the menu and the menu stays posted.

    No ``grab_release`` afterwards: on X11 the menu's global grab is what closes it on a click elsewhere or Escape
    (a release there kept it open until an entry was chosen, OR-1).
    """
    menu.tk_popup(event.x_root + 1, event.y_root + 1)
