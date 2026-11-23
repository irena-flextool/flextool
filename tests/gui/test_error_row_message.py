"""A red ("error") input-source row leads to its database's message in the message list.

Run headless, never on the owner's display:
    xvfb-run -a sh -c 'openbox & sleep 1; python -m pytest tests/gui/test_error_row_message.py --run-gui -n 0'

Expected values come from the rules, not from the code's output: double-click or Enter on a red row whose database
has a message opens the list at that database's first message (the list is ordered attention first, then by
input-source number, so b.sqlite = source 1 comes before a.sqlite = source 2 and a's message is "Message 2 of 2");
a red row without a message, and a row that is not red, still open the editor on double-click; the tooltip of a red
row with a message is the fixed text "FlexTool cannot use this file. Double-click for what to do.".
"""
from __future__ import annotations

import gc
import tkinter as tk
from collections.abc import Iterator
from pathlib import Path
from tkinter import ttk
from types import SimpleNamespace

import pytest

from flextool.gui.dialogs.message_list_window import MessageListController
from flextool.gui.input_sources import PlanningNote
from flextool.gui.main_window import MainWindow
from flextool.gui.message_sources import unreadable_source
from flextool.gui.messages import MESSAGE_SPECS, GuiMessage, database_key
from tests.gui.assistant.tk_helpers import pump

gui = pytest.mark.gui


class _Host(tk.Frame):
    """The main window's input-source list and message list only (the methods under test are MainWindow's)."""

    _on_input_source_dblclick = MainWindow._on_input_source_dblclick
    _on_input_source_return = MainWindow._on_input_source_return
    _row_message_subject = MainWindow._row_message_subject
    _open_row_message = MainWindow._open_row_message
    _on_input_source_motion = MainWindow._on_input_source_motion
    _hide_input_status_tip = MainWindow._hide_input_status_tip

    def __init__(self, top: tk.Toplevel, folder: Path, store: Path) -> None:
        super().__init__(top)
        self.pack(fill="both", expand=True)
        self.edited: list[tuple[str, ...]] = []
        self._input_status_tip = None
        self.project_settings = SimpleNamespace(external_refs={})
        self.input_sources_tree = ttk.Treeview(self, columns=("check", "name", "number", "status"), show="")
        self.input_sources_tree.pack(fill="both", expand=True)
        self.input_sources_tree.tag_configure("error", background="#6b2020")
        rows = [("b.sqlite", 1, "error"), ("a.sqlite", 2, "error"), ("c.sqlite", 3, "error"), ("ok.sqlite", 4, "")]
        sources = []
        for name, number, tag in rows:
            (folder / name).write_bytes(b"")
            self.input_sources_tree.insert("", "end", iid=name, values=("", name, number, "✗" if tag else "✓"),
                                           tags=(tag,) if tag else ())
            sources.append(SimpleNamespace(name=name, number=number, frozen_note=""))
        self.input_source_mgr = SimpleNamespace(_sources=sources, _last_ghosts=[],
                                                resolve_path=lambda s: folder / s.name)
        self._messages = MessageListController(top, None, store_path=store)

    def _on_edit_source(self) -> None:
        self.edited.append(tuple(self.input_sources_tree.selection()))


@pytest.fixture(scope="module")
def root() -> Iterator[tk.Tk]:
    try:
        window = tk.Tk()
    except tk.TclError:
        pytest.skip("No display available")
    window.withdraw()
    yield window
    try:
        window.destroy()
    except tk.TclError:
        pass
    del window
    gc.collect()


@pytest.fixture
def host(root: tk.Tk, tmp_path: Path) -> Iterator[_Host]:
    top = tk.Toplevel(root)
    top.geometry("500x300+40+40")
    folder = tmp_path / "input_sources"
    folder.mkdir()
    h = _Host(top, folder, tmp_path / "gui_messages.json")
    check = MESSAGE_SPECS["value_list_check"]
    notes = [PlanningNote("unreadable", "a.sqlite", folder / "a.sqlite", "no version")]
    b_check = GuiMessage(check, database_key(folder / "b.sqlite"), "b.sqlite", "Please check this database",
                         ("First.",))
    h._messages.open_project("P", sources=[lambda _s: [b_check], unreadable_source(lambda: notes)], folder=tmp_path)
    h._messages.set_ranks(h.input_source_mgr, h.input_source_mgr._sources)
    h._messages.center.recompute("other")
    top.update()
    pump(top, lambda: bool(top.winfo_viewable()))
    yield h
    h._messages.close()
    try:
        top.destroy()
    except tk.TclError:
        pass


def _row_y(h: _Host, iid: str) -> int:
    h.input_sources_tree.update_idletasks()
    _x, y, _w, height = h.input_sources_tree.bbox(iid)
    return y + height // 2


def _event(h: _Host, iid: str) -> SimpleNamespace:
    tree = h.input_sources_tree
    return SimpleNamespace(y=_row_y(h, iid), x=5, x_root=tree.winfo_rootx() + 5,
                           y_root=tree.winfo_rooty() + _row_y(h, iid))


def _position_and_subject(h: _Host) -> tuple[str, str]:
    window = h._messages.window
    assert window is not None
    return (str(window.body.nametowidget("header.position").cget("text")),
            str(window.body.nametowidget("content.subject").cget("text")))


@gui
def test_double_click_on_a_red_row_opens_its_database_message(host) -> None:
    host._on_input_source_dblclick(_event(host, "a.sqlite"))
    assert _position_and_subject(host) == ("Message 2 of 2", "a.sqlite")
    assert host._messages.window.opened_by is None       # like the indicator: no auto-show count
    assert host.edited == []
    host._on_input_source_dblclick(_event(host, "b.sqlite"))    # the open window moves to b's message
    assert _position_and_subject(host) == ("Message 1 of 2", "b.sqlite")
    assert host.edited == []


@gui
def test_enter_on_a_red_row_opens_its_database_message(host) -> None:
    host.input_sources_tree.focus("a.sqlite")
    assert host._on_input_source_return(None) == "break"
    assert _position_and_subject(host) == ("Message 2 of 2", "a.sqlite")


@gui
def test_rows_without_a_message_keep_opening_the_editor(host) -> None:
    host._on_input_source_dblclick(_event(host, "c.sqlite"))     # red, no message
    host._on_input_source_dblclick(_event(host, "ok.sqlite"))    # not red
    assert host.edited == [("c.sqlite",), ("ok.sqlite",)]
    assert host._messages.window is None
    host.input_sources_tree.focus("ok.sqlite")
    assert host._on_input_source_return(None) is None
    assert host._messages.window is None


@gui
def test_the_tooltip_of_a_red_row_with_a_message(host) -> None:
    def tip(iid: str) -> str:
        host._on_input_source_motion(_event(host, iid))
        text = str(host._input_status_tip.children["!label"].cget("text"))
        host._hide_input_status_tip()
        return text

    assert tip("a.sqlite") == "FlexTool cannot use this file. Double-click for what to do."
    assert tip("c.sqlite") == "Could not read scenarios (invalid or missing file)."
