"""Mouse drag-and-drop reordering for ttk.Treeview rows."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable


class DragReorderController:
    """Adds drag-to-reorder behaviour to a ttk.Treeview.

    Engages only on a ButtonPress-1 inside a row's BODY (cell region),
    so it composes with the CheckTreeController whose click handler
    captures clicks in the check column. The drag is ignored unless the
    pointer actually moves (so a plain click still selects the row via
    default Treeview behaviour).
    """

    _DRAG_THRESHOLD_PX = 4

    def __init__(
        self,
        tree: ttk.Treeview,
        *,
        check_column: str = "check",
        on_reorder: Callable[[list[str]], None] | None = None,
    ) -> None:
        self._tree = tree
        self._check_column = check_column
        self._on_reorder = on_reorder
        self._press_x: int | None = None
        self._press_y: int | None = None
        self._press_iid: str | None = None
        self._dragging: bool = False

        tree.bind("<ButtonPress-1>", self._on_press, add="+")
        tree.bind("<B1-Motion>", self._on_motion, add="+")
        tree.bind("<ButtonRelease-1>", self._on_release, add="+")

    def _click_in_check_column(self, event: tk.Event) -> bool:  # type: ignore[type-arg]
        if self._tree.identify("region", event.x, event.y) != "cell":
            return False
        column = self._tree.identify_column(event.x)
        try:
            idx = int(column.lstrip("#")) - 1
        except ValueError:
            return False
        cols = self._tree["columns"]
        return 0 <= idx < len(cols) and cols[idx] == self._check_column

    def _on_press(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if self._click_in_check_column(event):
            return
        iid = self._tree.identify_row(event.y)
        if not iid:
            return
        self._press_x = event.x
        self._press_y = event.y
        self._press_iid = iid
        self._dragging = False

    def _on_motion(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if self._press_iid is None or self._press_x is None or self._press_y is None:
            return
        if not self._dragging:
            if abs(event.x - self._press_x) < self._DRAG_THRESHOLD_PX and \
               abs(event.y - self._press_y) < self._DRAG_THRESHOLD_PX:
                return
            self._dragging = True
            self._tree.config(cursor="fleur")
        target = self._tree.identify_row(event.y)
        if not target or target == self._press_iid:
            return
        # Move the source row to where the target row currently sits.
        try:
            target_index = self._tree.index(target)
            self._tree.move(self._press_iid, "", target_index)
        except tk.TclError:
            pass

    def _on_release(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        was_dragging = self._dragging
        self._press_x = self._press_y = None
        self._press_iid = None
        self._dragging = False
        self._tree.config(cursor="")
        if was_dragging and self._on_reorder is not None:
            self._on_reorder(list(self._tree.get_children()))
