"""Reusable controller for Treeview widgets with a checkbox column."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable


class CheckTreeController:
    """Wires consistent check/uncheck behaviour onto a ttk.Treeview.

    The tree must have a column dedicated to a check glyph (default
    column id ``"check"``). The controller installs:

    - A ``<Button-1>`` handler that toggles a single row's check **only
      if the click landed inside the check column**. Clicks anywhere
      else fall through to the Treeview's default selection logic, so
      shift/ctrl multi-select keeps working.
    - A ``<space>`` handler that applies the multi-select toggle rule:
        all selected rows checked   -> uncheck all
        all selected rows unchecked -> check all
        mixed                       -> check all

    The controller never persists state itself; it calls back into the
    caller via ``on_toggle(changed_iids)`` so the host widget can save
    settings, refresh the display, etc.
    """

    def __init__(
        self,
        tree: ttk.Treeview,
        *,
        check_column: str = "check",
        checked_glyph: str = "▣",
        unchecked_glyph: str = "□",
        on_toggle: Callable[[list[str]], None] | None = None,
    ) -> None:
        self._tree = tree
        self._check_column = check_column
        self._checked = checked_glyph
        self._unchecked = unchecked_glyph
        self._on_toggle = on_toggle

        tree.bind("<Button-1>", self._on_click, add="+")
        tree.bind("<space>", self._on_space, add="+")
        tree.bind("<Key-space>", self._on_space, add="+")

    # ---- public helpers --------------------------------------------------

    def is_checked(self, iid: str) -> bool:
        try:
            value = self._tree.set(iid, self._check_column)
        except tk.TclError:
            return False
        return value == self._checked

    def set_checked(self, iid: str, value: bool, *, notify: bool = True) -> None:
        glyph = self._checked if value else self._unchecked
        try:
            self._tree.set(iid, self._check_column, glyph)
        except tk.TclError:
            return
        if notify and self._on_toggle is not None:
            self._on_toggle([iid])

    def toggle_selected(self) -> None:
        sel = list(self._tree.selection())
        if not sel:
            return
        all_checked = all(self.is_checked(iid) for iid in sel)
        target = not all_checked  # all checked -> uncheck all; else check all
        changed: list[str] = []
        for iid in sel:
            if self.is_checked(iid) != target:
                self._tree.set(
                    iid,
                    self._check_column,
                    self._checked if target else self._unchecked,
                )
                changed.append(iid)
        if changed and self._on_toggle is not None:
            self._on_toggle(changed)

    # ---- internal handlers ----------------------------------------------

    def _on_click(self, event: tk.Event) -> str | None:  # type: ignore[type-arg]
        # Only intercept clicks that land in the check column on a row.
        # Anything else (heading, body of other columns, blank area) is
        # passed through so default Treeview behaviour (single/extend
        # selection) handles it.
        region = self._tree.identify("region", event.x, event.y)
        if region != "cell":
            return None
        column = self._tree.identify_column(event.x)
        # column is like "#1"; resolve to the column id
        col_index_str = column.lstrip("#")
        try:
            col_index = int(col_index_str) - 1
        except ValueError:
            return None
        columns = self._tree["columns"]
        if col_index < 0 or col_index >= len(columns):
            return None
        if columns[col_index] != self._check_column:
            return None
        iid = self._tree.identify_row(event.y)
        if not iid:
            return None
        new_value = not self.is_checked(iid)
        self.set_checked(iid, new_value, notify=True)
        return "break"  # don't let default click also alter selection

    def _on_space(self, event: tk.Event) -> str | None:  # type: ignore[type-arg]
        self.toggle_selected()
        return "break"


def install_check_tree(
    tree: ttk.Treeview,
    on_toggle: Callable[[list[str]], None] | None = None,
    *,
    check_column: str = "check",
    checked_glyph: str = "▣",
    unchecked_glyph: str = "□",
) -> CheckTreeController:
    """Convenience: build + return a controller in one call."""
    return CheckTreeController(
        tree,
        check_column=check_column,
        checked_glyph=checked_glyph,
        unchecked_glyph=unchecked_glyph,
        on_toggle=on_toggle,
    )
