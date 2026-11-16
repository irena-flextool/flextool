"""The execution window's and the main window's popup menus close like menus (OR-1, fix group G16), and a quick
right-click does not run the first entry of a menu posted at the pointer.

A ``grab_release()`` right after ``tk_popup`` dropped the global grab that, on X11, closes a posted menu on a click
elsewhere, so these menus stayed up until an entry was chosen.  The tests post each menu through the window's own
method (on a small stand-in for the window), then send REAL X input with ``xdotool`` — a menu's global grab does
not see ``event_generate`` — and check that the menu is off the screen.

``tk_popup`` puts a menu's top-left corner at the given point: with a borderless menu posted exactly at the pointer,
the release of a quick right-click landed on entry 0 and ran it (Linux).  The quick-click tests press and release
the right button at one point through the widget's real binding and check that nothing ran and the menu stays up.

Run headless:  env -u DISPLAY xvfb-run -a python -m pytest tests/gui/test_popup_menus_close.py --run-gui
(with a window manager such as openbox running inside the xvfb display).
"""
from __future__ import annotations

import tkinter as tk
from collections.abc import Iterator
from tkinter import ttk
from types import SimpleNamespace

import pytest

from flextool.gui.assistant.chat_window import ChatWindow, styled_menu
from flextool.gui.assistant.transcript import Transcript
from flextool.gui.execution_manager import JobStatus, JobType
from flextool.gui.execution_window import ExecutionWindow
from flextool.gui.main_window import _GHOST_IID_PREFIX, MainWindow
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


# --- a quick right-click at the pointer does not run entry 0 -----------------------------------------------------
def _quick_right_click(window: tk.Toplevel, widget: tk.Misc, menu_of, chosen: list[str]) -> None:
    """Press and release the right button at one point of *widget* (its own ``<Button-3>`` binding posts the menu
    that ``menu_of()`` finds); then nothing has run, the menu is still up, and Escape closes it."""
    take_focus(window)
    xdotool(window, "mousemove", widget.winfo_rootx() + 12, widget.winfo_rooty() + 12)
    pump(window, timeout=0.1)
    xdotool(window, "mousedown", "3")
    pump(window, lambda: (m := menu_of()) is not None and bool(m.winfo_ismapped()))
    xdotool(window, "mouseup", "3")
    pump(window, timeout=0.4)
    menu = menu_of()
    assert chosen == [], f"the release of a quick right-click ran {chosen}"
    assert menu.winfo_ismapped(), "the menu closed on the release of a quick right-click"
    xdotool(window, "key", "Escape")
    assert closes(menu), "Escape left the menu open"
    assert chosen == []


def _last_menu(owner: tk.Misc):
    """The newest ``tk.Menu`` child of *owner* (the main window makes a new one per right-click), or None."""
    menus = [w for w in owner.winfo_children() if isinstance(w, tk.Menu)]
    return menus[-1] if menus else None


class _TreeOwner(tk.Frame):
    """The parts of the main window that its right-click menus of the three trees use, each on one tree."""

    _styled_popup_menu = MainWindow._styled_popup_menu
    _on_input_source_right_click = MainWindow._on_input_source_right_click
    _show_ghost_source_menu = MainWindow._show_ghost_source_menu
    _on_available_right_click = MainWindow._on_available_right_click
    _on_executed_right_click = MainWindow._on_executed_right_click

    def __init__(self, master: tk.Misc, chosen: list[str], handler: str, iid: str = "row") -> None:
        super().__init__(master)
        self.pack(fill="both", expand=True)
        tree = ttk.Treeview(self, height=4, show="tree", columns=("a", "b", "c", "d"))
        tree.insert("", "end", iid=iid, text="row", values=("", "", "", ""))
        tree.pack(fill="both", expand=True)
        self.input_sources_tree = self.available_tree = self.executed_tree = tree
        self.input_source_mgr = None
        tree.bind("<Button-3>", getattr(self, handler))
        for name in ("_on_edit_source", "_on_convert_source", "_on_delete_source", "_on_refresh_sources",
                     "_on_check_selected", "_on_add_selected_to_execution", "_on_executed_space_from_menu",
                     "_on_delete_results"):
            setattr(self, name, lambda n=name: chosen.append(n))
        self._delete_ghost_results = lambda _number: chosen.append("_delete_ghost_results")
        master.update()

    def _frozen_menu_source(self, _item: str) -> None:
        return None


@pytest.mark.parametrize("handler, iid", [
    ("_on_input_source_right_click", "row"),
    ("_on_input_source_right_click", f"{_GHOST_IID_PREFIX}3"),
    ("_on_available_right_click", "row"),
    ("_on_executed_right_click", "row"),
])
def test_a_quick_right_click_on_a_main_window_tree_runs_nothing(window, handler, iid):
    chosen: list[str] = []
    owner = _TreeOwner(window, chosen, handler, iid)
    _quick_right_click(window, owner.input_sources_tree, lambda: _last_menu(owner), chosen)


def test_a_quick_right_click_on_the_job_list_runs_nothing(window):
    chosen: list[str] = []
    owner = _job_menu_owner(window, chosen)
    owner._mgr.is_paused = False                # a pending scenario: entry 0, "Force start", is enabled
    owner._get_selected_job_ids = lambda: ["job"]
    owner._mgr.get_jobs = lambda: [SimpleNamespace(job_id="job", status=JobStatus.PENDING, job_type=JobType.SCENARIO)]
    owner._job_tree.bind("<Button-3>", lambda e: ExecutionWindow._on_tree_context_menu(owner, e))
    _quick_right_click(window, owner._job_tree, lambda: owner._context_menu, chosen)


def test_a_quick_right_click_on_the_log_runs_nothing(window):
    chosen: list[str] = []
    owner = _log_menu_owner(window)
    owner._output_menu.entryconfigure(0, command=lambda: chosen.append("copy"))
    owner._output_text.bind("<Button-3>", lambda e: ExecutionWindow._show_output_menu(owner, e))
    _quick_right_click(window, owner._output_text, lambda: owner._output_menu, chosen)


class _TranscriptOwner(tk.Frame):
    """The parts of the transcript that its question menu (Ask again / Edit) uses."""

    _post_question_menu = Transcript._post_question_menu

    def __init__(self, master: tk.Misc, chosen: list[str]) -> None:
        super().__init__(master)
        self.pack(fill="both", expand=True)
        self.text = tk.Text(self, height=5)
        self.text.insert("end", "a question")
        self.text.pack(fill="both", expand=True)
        self.text.bind("<Button-3>", self._post_question_menu)
        self._question_menu: tk.Menu | None = None
        self._question_action = lambda action, _line: chosen.append(str(action))
        master.update()

    def question_line_at(self, _index: str) -> int:
        return 1


def test_a_quick_right_click_on_a_question_runs_nothing(window):
    chosen: list[str] = []
    owner = _TranscriptOwner(window, chosen)
    _quick_right_click(window, owner.text, lambda: owner._question_menu, chosen)


class _ChatTabsOwner(tk.Frame):
    """The parts of the chat window that the menu of a chat's tab uses."""

    _on_tab_menu_click = ChatWindow._on_tab_menu_click
    _post_tab_menu = ChatWindow._post_tab_menu

    def __init__(self, master: tk.Misc, chosen: list[str]) -> None:
        super().__init__(master)
        self.pack(fill="both", expand=True)
        self.notebook = tk.Frame(self, width=300, height=100)
        self.notebook.pack(fill="both", expand=True)
        self.notebook.bind("<Button-3>", self._on_tab_menu_click)
        self.tab_menu = styled_menu(self)
        self.tabs = [object()]
        self.rename = lambda _tab: chosen.append("rename")
        self.close_chat = lambda _tab: chosen.append("close")
        self.close_others = lambda _tab: chosen.append("close others")
        master.update()

    def _tab_at(self, _x: int, _y: int) -> SimpleNamespace:
        return SimpleNamespace(is_new=False)


def test_a_quick_right_click_on_a_chat_tab_runs_nothing(window):
    chosen: list[str] = []
    owner = _ChatTabsOwner(window, chosen)
    _quick_right_click(window, owner.notebook, lambda: owner.tab_menu, chosen)
