"""Tests for CheckTreeController, focused on the can_check predicate.

The predicate lets a host mark certain rows as non-checkable (e.g. the
retired "ghost" input-source rows), so neither a check-column click nor a
space-bar toggle can ever tick them.
"""

from __future__ import annotations

import tkinter as tk

import pytest
from tkinter import ttk

from flextool.gui.check_tree import CheckTreeController


@pytest.fixture()
def tree():
    """A Treeview with a check column; skip if no display is available."""
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("No display available")
    root.withdraw()
    tv = ttk.Treeview(root, columns=("check", "name"), show="headings")
    tv.insert("", "end", iid="live", values=("□", "real.sqlite"))
    tv.insert("", "end", iid="ghost:1", values=("", "(source 1)"))
    yield tv
    root.destroy()


def test_space_toggle_skips_non_checkable_rows(tree) -> None:
    ctrl = CheckTreeController(
        tree, checked_glyph="■", unchecked_glyph="□",
        can_check=lambda iid: not iid.startswith("ghost:"),
    )
    tree.selection_set(("live", "ghost:1"))
    ctrl.toggle_selected()

    assert tree.set("live", "check") == "■"   # checkable row toggled on
    assert tree.set("ghost:1", "check") == ""  # ghost row untouched


def test_set_checked_still_forced_for_host_use(tree) -> None:
    # The predicate guards user interaction, not programmatic set_checked —
    # the host stays in control of explicit state changes.
    ctrl = CheckTreeController(
        tree, checked_glyph="■", unchecked_glyph="□",
        can_check=lambda iid: not iid.startswith("ghost:"),
    )
    ctrl.set_checked("live", True, notify=False)
    assert tree.set("live", "check") == "■"
