"""The plot tree of the Result viewer: groups 8 (unit / connection details) and 9 (extras) start collapsed.

Drives the real ``ResultViewer._populate_plot_tree`` / ``_restore_or_select_first_entry`` against a real
``ttk.Treeview`` with a lightweight stand-in for the rest of the viewer.  Must run under ``xvfb-run -a``.
"""

import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace

import pytest

import flextool.gui.result_viewer as rv
from flextool.gui.plot_config_reader import PlotConfigData, PlotEntry, PlotGroup


class _FakeViewer:
    """What ``_populate_plot_tree`` and ``_restore_or_select_first_entry`` touch."""

    _restore_or_select_first_entry = rv.ResultViewer._restore_or_select_first_entry
    _is_entry_disabled = rv.ResultViewer._is_entry_disabled

    def __init__(self, root, last_entry=""):
        self._plot_tree = ttk.Treeview(root)
        self._tree_entry_map = {}
        self._mode = tk.StringVar(root, value="single")
        self._viewer_settings = SimpleNamespace(last_entry=last_entry)

    def _get_config_path_for_mode(self):
        return "unused.yaml"

    def _collect_all_variant_letters(self):
        pass

    def _update_variant_canvas_width(self):
        pass

    def _update_tree_availability(self):
        pass


@pytest.fixture()
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"No display available: {exc}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture()
def ten_groups(monkeypatch):
    groups = [
        PlotGroup(number=str(g), name=f"group {g}",
                  entries=[PlotEntry(number=f"{g}.{e}", short_name=f"e{e}", full_name=f"entry {g}.{e}")
                           for e in range(2)])
        for g in range(10)
    ]
    monkeypatch.setattr(rv, "parse_plot_config", lambda _path: PlotConfigData(groups=groups))


def _open_groups(tree):
    return {iid for iid in tree.get_children() if tree.item(iid, "open")}


def test_groups_8_and_9_start_collapsed(tk_root, ten_groups):
    viewer = _FakeViewer(tk_root)
    rv.ResultViewer._populate_plot_tree(viewer)
    tree = viewer._plot_tree
    assert _open_groups(tree) == {f"group_{g}" for g in range(8)}
    assert not tree.item("group_8", "open") and not tree.item("group_9", "open")
    assert tree.selection() == ("entry_0.0",)


def test_a_remembered_entry_in_group_8_reopens_it(tk_root, ten_groups):
    viewer = _FakeViewer(tk_root, last_entry="8.1")
    rv.ResultViewer._populate_plot_tree(viewer)
    tree = viewer._plot_tree
    assert tree.selection() == ("entry_8.1",)
    assert tree.item("group_8", "open")
    assert not tree.item("group_9", "open")
