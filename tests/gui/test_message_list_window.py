"""The message list window, its indicator and the deferred auto-show (design §3.4, §4.1–§4.3, §5 "Tk", package M2).

Run headless, never on the owner's display:
    xvfb-run -a sh -c 'openbox & sleep 1; python -m pytest tests/gui/test_message_list_window.py --run-gui \
        --gui-only -n 0'

Expected values come from the design's rules, not from the window's output: "Message i of N" / "1 message",
Previous disabled on the first and Next on the last, Dismiss only on a dismissable message, Close for now records
"seen" (and one ``shown`` per non-attention message only when the project-open auto-show opened the window), no
grab, the main window's ``Alt-c`` does not fire in the list, an auto-show waits while another window holds a grab,
the "other" trigger opens the list only for a new attention message, an open window updates in place.  The docs
address is built here from ``run_warnings.registry.docs_url`` (a W-code goes to its section).
"""
from __future__ import annotations

import gc
import tkinter as tk
from collections.abc import Iterator
from pathlib import Path
from tkinter import ttk
from types import SimpleNamespace

import pytest

from flextool.gui.dialogs import message_list_window as mlw
from flextool.gui.dialogs.message_list_window import MessageListController, MessageListWindow
from flextool.gui.messages import (
    MESSAGE_SPECS, GuiMessage, MessageAction, MessageCenter, MessageStore, database_key,
)
from flextool.run_warnings.registry import docs_url
from tests.gui.assistant.tk_helpers import key, pump

gui = pytest.mark.gui
CHECK = MESSAGE_SPECS["value_list_check"]          # attention, not dismissable
UPGRADED = MESSAGE_SPECS["database_upgraded"]      # not attention, dismissable


def _upgraded_source(store: MessageStore):
    """M3a's shape: one message per pending ``database_upgraded`` entry, instance = its run."""
    for subject, entry in store.pending("database_upgraded").items():
        yield GuiMessage(UPGRADED, subject, entry["label"], "This database was upgraded",
                         ("It now has the current data format.",), instance=entry["run"])


def _db(folder: Path, name: str) -> str:
    (folder / name).write_bytes(b"")        # a key of a missing file in an existing folder is pruned
    return database_key(folder / name)


def _check(subject: str, label: str = "a.sqlite", action: MessageAction | None = None) -> GuiMessage:
    return GuiMessage(CHECK, subject, label, "Please check this database", ("First.", "Second."), action=action)


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
def host(root: tk.Tk) -> Iterator[tk.Toplevel]:
    window = tk.Toplevel(root)
    window.geometry("700x500+40+40")
    window.update()
    pump(window, lambda: bool(window.winfo_viewable()))
    yield window
    try:
        window.destroy()
    except tk.TclError:
        pass


@pytest.fixture
def center3(tmp_path: Path) -> MessageCenter:
    """One attention state message and two upgrade events: three messages, the attention one first."""
    a, b, c = (_db(tmp_path, n) for n in ("a.sqlite", "b.sqlite", "c.sqlite"))
    center = MessageCenter(tmp_path / "gui_messages.json", [lambda _s: [_check(a)], _upgraded_source])
    center.post_event("database_upgraded", b, "b.sqlite", run="r1")
    center.post_event("database_upgraded", c, "c.sqlite", run="r1")
    center.recompute("project_open")
    return center


def _w(window: MessageListWindow, path: str) -> tk.Misc:
    return window.body.nametowidget(path)


def _text(window: MessageListWindow, path: str) -> str:
    return str(_w(window, path).cget("text"))


def _mapped(window: MessageListWindow, path: str) -> bool:
    window.update_idletasks()
    return bool(_w(window, path).winfo_ismapped())


def _paragraphs(window: MessageListWindow) -> list[str]:
    return [str(w.cget("text")) for w in _w(window, "content.text").winfo_children()]


@gui
def test_paging_counts_and_disables_the_ends(host, center3) -> None:
    window = MessageListWindow(host, center3, project="P")
    try:
        pump(window, lambda: bool(window.winfo_viewable()))
        assert window.title() == "FlexTool messages: P"
        assert _text(window, "header.position") == "Message 1 of 3"
        assert _text(window, "content.title") == "Please check this database"
        assert _text(window, "content.subject") == "a.sqlite"
        assert _paragraphs(window) == ["First.", "Second."]
        assert _w(window, "header.previous").instate(["disabled"])
        assert not _w(window, "header.next").instate(["disabled"])
        assert window.focus_get() is _w(window, "header.next")
        _w(window, "header.next").invoke()
        assert _text(window, "header.position") == "Message 2 of 3"
        assert _text(window, "content.subject") == "b.sqlite"
        key(_w(window, "buttons.close"), "<Right>")
        assert _text(window, "header.position") == "Message 3 of 3"
        assert _w(window, "header.next").instate(["disabled"])
        key(_w(window, "buttons.close"), "<Left>")
        assert _text(window, "header.position") == "Message 2 of 3"
    finally:
        window.close()


@gui
def test_one_message_hides_the_pager_and_focuses_close(host, tmp_path) -> None:
    a = _db(tmp_path, "a.sqlite")
    center = MessageCenter(tmp_path / "s.json", [lambda _s: [_check(a)]])
    center.recompute("project_open")
    window = MessageListWindow(host, center)
    try:
        pump(window, lambda: bool(window.winfo_viewable()))
        assert _text(window, "header.position") == "1 message"
        assert not _mapped(window, "header.previous") and not _mapped(window, "header.next")
        assert window.focus_get() is _w(window, "buttons.close")
        assert _text(window, "buttons.close") == "Close for now"
    finally:
        window.close()


@gui
def test_dismiss_only_on_a_dismissable_message_and_it_moves_on(host, center3, tmp_path) -> None:
    changes: list[int] = []
    window = MessageListWindow(host, center3, on_change=lambda: changes.append(1))
    try:
        pump(window, lambda: bool(window.winfo_viewable()))
        assert not _mapped(window, "buttons.dismiss")             # W-0490: no Dismiss
        assert _mapped(window, "buttons.docs")
        window.page(1)
        assert _mapped(window, "buttons.dismiss")                 # an upgrade event
        _w(window, "buttons.dismiss").invoke()
        # b's event is gone from the file; c (the next one) is shown.
        assert database_key(tmp_path / "b.sqlite") not in MessageStore(tmp_path / "gui_messages.json").pending(
            "database_upgraded")
        assert database_key(tmp_path / "c.sqlite") in MessageStore(tmp_path / "gui_messages.json").pending(
            "database_upgraded")
        assert _text(window, "header.position") == "Message 2 of 2"
        assert _text(window, "content.subject") == "c.sqlite"
        assert changes == [1]
        _w(window, "buttons.dismiss").invoke()                    # the last one: the previous one is shown
        assert _text(window, "header.position") == "1 message"
        assert _text(window, "content.subject") == "a.sqlite"
        assert not _mapped(window, "buttons.dismiss")
    finally:
        window.close()


@gui
def test_no_messages_left(host, tmp_path) -> None:
    b = _db(tmp_path, "b.sqlite")
    center = MessageCenter(tmp_path / "s.json", [_upgraded_source])
    center.post_event("database_upgraded", b, "b.sqlite", run="r1")
    center.recompute("refresh")
    window = MessageListWindow(host, center)
    try:
        pump(window, lambda: bool(window.winfo_viewable()))
        _w(window, "buttons.dismiss").invoke()
        assert _text(window, "content.title") == "No messages."
        assert _paragraphs(window) == ["Notices about this project and its databases appear here."]
        assert _text(window, "buttons.close") == "Close"
        assert not any(_mapped(window, f"buttons.{n}") for n in ("docs", "action", "dismiss"))
        assert not _mapped(window, "header.position")
    finally:
        window.close()


@gui
@pytest.mark.parametrize(("opened_by", "shown"), [("project_open", 1), (None, 0), ("refresh", 0)])
def test_close_for_now_marks_seen_and_counts_shown_only_for_the_project_open_show(
        host, center3, tmp_path, opened_by, shown) -> None:
    closed: list[int] = []
    window = MessageListWindow(host, center3, opened_by=opened_by, on_change=lambda: closed.append(1))
    pump(window, lambda: bool(window.winfo_viewable()))
    _w(window, "buttons.close").invoke()
    assert window.closed and closed == [1]
    assert center3.seen == {m.key for m in center3.messages}
    store = MessageStore(tmp_path / "gui_messages.json")
    upgraded = [m for m in center3.messages if not m.spec.attention]
    assert len(upgraded) == 2 and all(store.shown(m) == shown for m in upgraded)
    attention = [m for m in center3.messages if m.spec.attention][0]
    assert store.shown(attention) == 0                     # attention messages have no store entry


@gui
def test_escape_and_the_window_button_close(host, center3) -> None:
    window = MessageListWindow(host, center3)
    pump(window, lambda: bool(window.winfo_viewable()))
    key(_w(window, "buttons.close"), "<Escape>")
    assert window.closed
    window = MessageListWindow(host, center3)
    pump(window, lambda: bool(window.winfo_viewable()))
    window.tk.call(window.protocol("WM_DELETE_WINDOW"))
    assert window.closed


@gui
def test_docs_button_opens_the_section_and_the_window_stays(host, center3, monkeypatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(mlw.message_box, "open_url", opened.append)
    window = MessageListWindow(host, center3)
    try:
        pump(window, lambda: bool(window.winfo_viewable()))
        _w(window, "buttons.docs").invoke()
        assert opened == [docs_url("W-0490")]
        assert not window.closed and window.winfo_viewable()
    finally:
        window.close()


@gui
def test_action_runs_shows_its_note_and_is_hidden_without_one(host, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(mlw, "ACTION_NOTE_MS", 100)
    runs: list[int] = []

    def run() -> str:
        runs.append(1)
        return "Copied"
    a, b = _db(tmp_path, "a.sqlite"), _db(tmp_path, "b.sqlite")
    action = MessageAction("Copy repair command", run, tooltip="Paste into a terminal")
    center = MessageCenter(tmp_path / "s.json", [lambda _s: [_check(a, action=action), _check(b, "b")]])
    center.recompute("project_open")
    window = MessageListWindow(host, center)
    try:
        pump(window, lambda: bool(window.winfo_viewable()))
        assert _text(window, "buttons.action") == "Copy repair command"
        assert window.action_tooltip.text == "Paste into a terminal"
        _w(window, "buttons.action").invoke()
        assert runs == [1] and _text(window, "buttons.action") == "Copied" and not window.closed
        pump(window, lambda: _text(window, "buttons.action") == "Copy repair command", timeout=2)
        window.page(1)
        assert not _mapped(window, "buttons.action")
    finally:
        window.close()


@gui
def test_the_list_takes_no_grab_and_is_shielded(host, center3) -> None:
    fired: list[str] = []
    entry = tk.Entry(host)
    entry.pack()
    host.bind_all("<Alt-Key-c>", lambda _e: fired.append("alt-c"))
    window = MessageListWindow(host, center3)
    try:
        pump(window, lambda: bool(window.winfo_viewable()))
        pump(window, timeout=0.2)
        assert window.grab_current() is None
        key(_w(window, "buttons.close"), "<Alt-Key-c>")
        assert fired == []
        key(entry, "<Alt-Key-c>")                       # the main window still reacts
        assert fired == ["alt-c"]
    finally:
        host.unbind_all("<Alt-Key-c>")
        window.close()


# ── the controller: indicator, deferred auto-show, in-place update, project switch ──────────────────────────────


@pytest.fixture
def controller(host, tmp_path) -> Iterator[MessageListController]:
    button = ttk.Button(host)
    button.pack()
    ctl = MessageListController(host, button, store_path=tmp_path / "gui_messages.json")
    yield ctl
    ctl.close()


@gui
def test_auto_show_waits_while_another_window_holds_the_grab(controller, host, tmp_path) -> None:
    a = _db(tmp_path, "a.sqlite")
    controller.open_project("P", sources=[lambda _s: [_check(a)]])
    form = tk.Toplevel(host)
    form.geometry("200x100+60+60")
    pump(form, lambda: bool(form.winfo_viewable()))
    form.grab_set()
    try:
        controller.recompute("project_open")
        pump(host, timeout=0.8)                  # three retries of 250 ms
        assert controller.window is None and controller._pending is not None
    finally:
        form.grab_release()
        form.destroy()
    pump(host, lambda: controller.window is not None, timeout=2)
    assert controller.window.opened_by == "project_open"
    assert controller.window.grab_current() is None


@gui
def test_other_trigger_lights_the_indicator_and_opens_only_for_new_attention(controller, host, tmp_path) -> None:
    b = _db(tmp_path, "b.sqlite")
    states: list[GuiMessage] = []
    controller.open_project("P", sources=[lambda _s: list(states), _upgraded_source])
    assert str(controller.button.cget("text")) == "Messages"
    assert str(controller.button.cget("style")) == "TButton"
    controller.center.post_event("database_upgraded", b, "b.sqlite", run="r1")
    controller.recompute("other")
    pump(host, timeout=0.4)
    assert controller.window is None
    assert str(controller.button.cget("text")) == "Messages (1)"
    assert str(controller.button.cget("style")) == "Accent.TButton"      # unseen
    states.append(_check(_db(tmp_path, "a.sqlite")))
    controller.recompute("other")
    pump(host, lambda: controller.window is not None, timeout=2)
    assert controller.window.opened_by == "other"
    assert _text(controller.window, "header.position") == "Message 1 of 2"
    assert _text(controller.window, "content.title") == "Please check this database"


@gui
def test_an_open_window_updates_in_place_and_closing_clears_the_accent(controller, host, tmp_path) -> None:
    b, c = _db(tmp_path, "b.sqlite"), _db(tmp_path, "c.sqlite")
    controller.open_project("P", sources=[_upgraded_source])
    controller.center.post_event("database_upgraded", b, "b.sqlite", run="r1")
    controller.recompute("project_open")
    pump(host, lambda: controller.window is not None, timeout=2)
    window = controller.window
    assert _text(window, "header.position") == "1 message"
    controller.center.post_event("database_upgraded", c, "c.sqlite", run="r1")
    controller.recompute("refresh")
    pump(host, timeout=0.3)
    assert controller.window is window and _text(window, "header.position") == "Message 1 of 2"
    assert str(controller.button.cget("text")) == "Messages (2)"
    _w(window, "buttons.close").invoke()
    assert controller.window is None
    assert str(controller.button.cget("style")) == "TButton"     # all seen, none needs attention
    controller.open_from_indicator()                              # reopened from the indicator
    assert controller.window is not None and controller.window.opened_by is None


@gui
def test_a_project_switch_closes_the_list_and_starts_unseen(controller, host, tmp_path) -> None:
    a = _db(tmp_path, "a.sqlite")
    controller.open_project("P", sources=[lambda _s: [_check(a)]])
    controller.recompute("project_open")
    pump(host, lambda: controller.window is not None, timeout=2)
    old_window, old_center = controller.window, controller.center
    controller.open_project("Q", sources=[lambda _s: [_check(a)]])
    assert old_window.closed and old_center.seen == {_check(a).key}
    assert controller.window is None and controller.center is not old_center and controller.center.seen == set()
    controller.recompute("project_open")
    pump(host, lambda: controller.window is not None, timeout=2)
    assert controller.window.title() == "FlexTool messages: Q"


@gui
def test_a_pending_auto_show_is_cancelled_by_close(controller, host, tmp_path) -> None:
    a = _db(tmp_path, "a.sqlite")
    controller.open_project("P", sources=[lambda _s: [_check(a)]])
    controller.recompute("project_open")
    controller.close()
    pump(host, timeout=0.4)
    assert controller.window is None and controller._pending is None


@gui
def test_a_pending_auto_show_whose_message_is_gone_opens_nothing(controller, host, tmp_path) -> None:
    """Rule: the deferred open is for one message; when it is gone by then, the next recompute decides (here: an
    unseen "upgraded" note under trigger "other" would never open the list by itself)."""
    a, b = _db(tmp_path, "a.sqlite"), _db(tmp_path, "b.sqlite")
    states = [_check(a)]
    controller.open_project("P", sources=[lambda _s: list(states), _upgraded_source])
    form = tk.Toplevel(host)
    form.geometry("200x100+60+60")
    pump(form, lambda: bool(form.winfo_viewable()))
    form.grab_set()
    try:
        controller.recompute("other")                       # a new attention message: an open is pending
        states.clear()                                      # the file became readable meanwhile
        controller.center.post_event("database_upgraded", b, "b.sqlite", run="r1")
        controller.center.recompute("other")
    finally:
        form.grab_release()
        form.destroy()
    pump(host, timeout=0.6)
    assert controller.window is None and controller._pending is None


@gui
def test_the_viewable_wait_starts_at_the_first_try_not_at_recompute(controller, host, tmp_path, monkeypatch) -> None:
    """A slow start (the rest of MainWindow.__init__ between recompute and the first try) must not use up the 5 s."""
    a = _db(tmp_path, "a.sqlite")
    clock = [1000.0]
    monkeypatch.setattr(mlw, "time", SimpleNamespace(monotonic=lambda: clock[0]))      # this module only
    controller.open_project("P", sources=[lambda _s: [_check(a)]])
    host.withdraw()
    controller.recompute("project_open")
    clock[0] += 60.0                                        # a minute passes before the first try
    pump(host, timeout=0.6)
    assert controller.window is None and controller._pending is not None
    host.deiconify()
    pump(host, lambda: controller.window is not None, timeout=2)
    assert controller.window.opened_by == "project_open"


# ── main_window wiring (Tk-free: unbound methods on stand-in hosts) ────────────────────────────────────────────


def test_refresh_button_passes_the_refresh_trigger() -> None:
    from flextool.gui.main_window import MainWindow

    calls: list[str] = []
    host = SimpleNamespace(input_source_mgr=object(), _refresh_input_sources=lambda trigger="other": calls.append(
        trigger))
    MainWindow._on_refresh_sources(host)
    assert calls == ["refresh"]


def test_a_host_without_a_message_list_does_nothing() -> None:
    from flextool.gui.main_window import MainWindow

    MainWindow._messages_recompute(SimpleNamespace(), "project_open", [])     # no error


def test_ranks_follow_the_input_source_numbers(tmp_path) -> None:
    mgr = SimpleNamespace(resolve_path=lambda s: tmp_path / s.name)
    ctl = MessageListController.__new__(MessageListController)
    ctl._ranks = {}
    ctl.set_ranks(mgr, [SimpleNamespace(name="x.sqlite", number=3), SimpleNamespace(name="y.sqlite", number=1)])
    assert ctl._rank(database_key(tmp_path / "x.sqlite")) == 3
    assert ctl._rank(database_key(tmp_path / "y.sqlite")) == 1
    assert ctl._rank("db:elsewhere") > 3

