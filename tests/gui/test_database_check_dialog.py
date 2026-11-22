"""The W-0490 "Please check this database" box: its text, "Open checklist" (the box stays open) and "Close".

Run headless:  xvfb-run -a python -m pytest tests/gui/test_database_check_dialog.py --run-gui
"""
from __future__ import annotations

import tkinter as tk

import pytest

from flextool.gui.dialogs.database_check_dialog import TITLE, show_database_check
from flextool.run_warnings import docs_url
from flextool.update_flextool.value_list_history import CODE, Finding, gui_dialog_text

pytestmark = pytest.mark.gui


@pytest.fixture()
def root():
    try:
        window = tk.Tk()
    except tk.TclError:
        pytest.skip("No display available")
    window.withdraw()
    yield window
    window.destroy()


def test_open_checklist_then_close(root: tk.Tk) -> None:
    text = gui_dialog_text([("cyprus.sqlite", Finding(22, ("invest_methods",), 21, 22, "2024-09-19"))])
    opened: list[str] = []
    seen: dict[str, str] = {}
    presses = ["open", "close"]

    def press() -> None:
        dialogs = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
        if not dialogs:
            root.after(20, press)
            return
        dlg = dialogs[-1]
        seen.setdefault("title", dlg.title())
        seen.setdefault("text", str(dlg.nametowidget("text").cget("text")))
        seen.setdefault("buttons", " | ".join(str(b.cget("text")) for b in
                                              dlg.nametowidget("buttons").winfo_children()))
        dlg.nametowidget(f"buttons.{presses.pop(0)}").invoke()
        if presses:
            root.after(20, press)

    root.after(50, press)
    show_database_check(root, text, docs_url(CODE), open_url=opened.append)
    assert seen == {"title": TITLE, "text": text, "buttons": "Open checklist | Close"}
    assert TITLE == "Please check this database"
    assert opened == [docs_url(CODE)], "opened once, and the box stayed open for Close"
    assert not [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
