"""The execution window's and the main window's popup menus close like menus (OR-1, fix group G16).

A ``grab_release()`` right after ``tk_popup`` dropped the global grab that, on X11, closes a posted menu on a click
elsewhere, so these menus stayed up until an entry was chosen.  The tests post each menu through the window's own
method (on a small stand-in for the window), then send REAL X input with ``xdotool`` — a menu's global grab does
not see ``event_generate`` — and check that the menu is off the screen.

Run headless:  env -u DISPLAY xvfb-run -a python -m pytest tests/gui/test_popup_menus_close.py --run-gui
(with a window manager such as openbox running inside the xvfb display).
"""
from __future__ import annotations

import tkinter as tk
from collections.abc import Iterator
from tkinter import ttk
from types import SimpleNamespace

import pytest

from flextool.gui.execution_window import ExecutionWindow
from flextool.gui.main_window import MainWindow
from tests.gui.assistant.tk_helpers import (  # noqa: F401 — real_input is a fixture
    click_on, closes, posted, pump, real_input, take_focus, xdotool)

pytestmark = [pytest.mark.gui, pytest.mark.usefixtures("real_input")]


@pytest.fixture(scope="module")
def tk_root() -> Iterator[tk.Tk]:
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("No display available")
    root.withdraw()
    yield root
    try:
        root.destroy()
    except tk.TclError:
        pass


@pytest.fixture
def window(tk_root: tk.Tk) -> Iterator[tk.Toplevel]:
    top = tk.Toplevel(tk_root)
    top.geometry("500x400+60+60")
    top.update()
    pump(top, lambda: bool(top.winfo_viewable()))
    yield top
    try:
        top.destroy()
    except tk.TclError:
        pass


def _event(widget: tk.Misc) -> SimpleNamespace:
    """A right-click near the top-left corner of *widget* (screen and widget coordinates)."""
    return SimpleNamespace(x=5, y=5, x_root=widget.winfo_rootx() + 5, y_root=widget.winfo_rooty() + 5)


def _away(window: tk.Toplevel) -> tk.Frame:
    """A spot in *window* well away from a menu posted at its top-left corner."""
    spot = tk.Frame(window, width=80, height=60)
    spot.place(relx=1.0, rely=1.0, anchor="se")
    window.update()
    return spot


# --- execution window --------------------------------------------------------------------------------------------
def _log_menu_owner(window: tk.Toplevel) -> SimpleNamespace:
    text = tk.Text(window, height=5)
    text.pack(fill="x")
    menu = tk.Menu(text, tearoff=0)
    menu.add_command(label="Copy")
    menu.add_command(label="Select all")
    window.update()
    return SimpleNamespace(_output_text=text, _output_menu=menu)


def test_the_log_menu_closes_on_a_click_outside(window):
    owner = _log_menu_owner(window)
    away = _away(window)
    take_focus(window)
    ExecutionWindow._show_output_menu(owner, _event(owner._output_text))
    posted(owner._output_menu)
    click_on(away)
    assert closes(owner._output_menu), "the log menu stayed open after a click outside it"


def test_the_log_menu_closes_on_escape(window):
    owner = _log_menu_owner(window)
    take_focus(window)
    ExecutionWindow._show_output_menu(owner, _event(owner._output_text))
    posted(owner._output_menu)
    xdotool(window, "key", "Escape")
    assert closes(owner._output_menu), "Escape left the log menu open"


def _job_menu_owner(window: tk.Toplevel, chosen: list[str]) -> SimpleNamespace:
    tree = ttk.Treeview(window, height=4)
    tree.pack(fill="x")
    window.update()
    return SimpleNamespace(
        _job_tree=tree,
        _context_menu=tk.Menu(tree, tearoff=0),
        _mgr=SimpleNamespace(get_jobs=lambda: [], is_paused=True),   # paused: "Continue executions" is enabled
        _get_selected_job_ids=lambda: [],
        _on_force_selected=lambda: chosen.append("force"),
        _on_pause_toggle=lambda: chosen.append("pause"),
        _on_kill_selected=lambda: chosen.append("kill"),
        _on_remove_selected=lambda: chosen.append("remove"),
    )


def test_the_job_menu_closes_on_a_click_outside(window):
    owner = _job_menu_owner(window, [])
    away = _away(window)
    take_focus(window)
    ExecutionWindow._on_tree_context_menu(owner, _event(owner._job_tree))
    posted(owner._context_menu)
    click_on(away)
    assert closes(owner._context_menu), "the job menu stayed open after a click outside it"


def test_the_job_menu_works_from_the_keyboard(window):
    chosen: list[str] = []
    owner = _job_menu_owner(window, chosen)
    take_focus(window)
    ExecutionWindow._on_tree_context_menu(owner, _event(owner._job_tree))
    posted(owner._context_menu)
    xdotool(window, "key", "Down", "Return")    # the one enabled entry, "Continue executions"
    pump(window, lambda: bool(chosen))
    assert chosen == ["pause"]
    assert closes(owner._context_menu)


# --- main window: the UI-settings menu ---------------------------------------------------------------------------
class _SettingsOwner(tk.Frame):
    """The parts of the main window that ``_on_ui_settings_btn`` uses."""

    _styled_popup_menu = MainWindow._styled_popup_menu
    _on_ui_settings_btn = MainWindow._on_ui_settings_btn

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.global_settings = SimpleNamespace(font_size_pt=10, code_font_size_pt=0)
        self._theme_var = tk.StringVar(master=self, value="os")
        self.ui_settings_btn = ttk.Button(self, text="UI")
        self.ui_settings_btn.pack(side="right")
        self.reset = 0

    def _on_reset_window_layout(self) -> None:
        self.reset += 1

    def _set_font_size(self, _size: int) -> None: ...
    def _on_font_size_custom(self) -> None: ...
    def _set_code_font_size(self, _size: int) -> None: ...
    def _on_code_font_size_custom(self) -> None: ...
    def _on_theme_change(self) -> None: ...


def _settings_menu(owner: _SettingsOwner) -> tk.Menu:
    menus = [w for w in owner.winfo_children() if isinstance(w, tk.Menu)]
    return menus[-1]


def test_the_ui_settings_menu_closes_on_a_click_outside_and_on_escape(window):
    owner = _SettingsOwner(window)
    owner.pack(fill="x")
    tk.Frame(window, height=40).pack(fill="x")
    away = _away(window)
    take_focus(window)
    owner._on_ui_settings_btn()
    menu = _settings_menu(owner)
    posted(menu)
    click_on(away)
    assert closes(menu), "the UI-settings menu stayed open after a click outside it"
    take_focus(window)
    owner._on_ui_settings_btn()
    menu = _settings_menu(owner)
    posted(menu)
    xdotool(window, "key", "Escape")
    assert closes(menu), "Escape left the UI-settings menu open"


def test_the_ui_settings_menu_works_from_the_keyboard(window):
    owner = _SettingsOwner(window)
    owner.pack(fill="x")
    take_focus(window)
    owner._on_ui_settings_btn()
    menu = _settings_menu(owner)
    posted(menu)
    xdotool(window, "key", "Down", "Down", "Down", "Down", "Return")   # the fourth entry, "Reset window layout"
    pump(window, lambda: owner.reset == 1)
    assert closes(menu)
