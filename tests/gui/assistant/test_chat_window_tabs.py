"""The chat window's tabs (P2 §13.2 test 6; P2-O13–O17, critique 2 MJ3, MJ4, MJ6, m3).

Sending in "New chat" makes chat 1 and a new "New chat"; the Chats menu shows number, date and first words; chats
that do not fit are hidden and reachable from the menu and by Ctrl+Tab / Ctrl+Shift+Tab (as ``ISO_Left_Tab``) from
the input, the transcript, the tab row and a button; Ctrl+T / Ctrl+W; two chats answer at once; one data notice at
a time; a chat closed mid-answer lets its worker go and an immediate reopen waits for it; a name given during a
tool call survives the close; reopening the window restores the chats; at retention 0 closing asks first.

Run headless:  xvfb-run -a python -m pytest tests/gui/assistant --run-gui
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from flextool.assistant.store import open_store
from flextool.gui.assistant import chat_window as chat_window_module
from flextool.gui.assistant.notices import ConfirmDialog, ConsentDialog, RenameChatDialog
from flextool.gui.assistant.worker import ConsentNeeded, finishing_worker
from tests.assistant.gui_support.worker_rig import LOCAL, REMOTE, Tools, save_provider, say, tool_call
from tests.assistant.llm.fake_backend import Block
from tests.gui.assistant.tk_helpers import (  # noqa: F401 — real_input is a fixture
    click_on, closes, key, posted, pump, real_input, take_focus, widgets, xdotool)
from tests.gui.assistant.window_rig import Backends, Windows, ask, chat_texts, wait_idle

pytestmark = pytest.mark.gui


@pytest.fixture
def tools(monkeypatch: pytest.MonkeyPatch):
    fake = Tools()
    fake.install(monkeypatch)
    yield fake
    fake.release.set()


@pytest.fixture
def backends() -> Backends:
    return Backends()


@pytest.fixture
def windows(tk_root, tmp_path: Path, backends, tools):
    project = tmp_path / "project"
    project.mkdir()
    made = Windows(tk_root, project, backends)
    yield made
    tools.release.set()
    made.close_all()


def _dialogs(window, kind):
    return [w for w in widgets(window) if isinstance(w, kind) and w.winfo_exists()]


def _local(created: str) -> str:
    moment = datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).astimezone()
    return f"{moment.day} {moment.strftime('%b')} {moment.strftime('%H:%M')}"


def _menu_labels(menu) -> list[str]:
    end = menu.index("end")
    return [] if end is None else [menu.entrycget(i, "label") if menu.type(i) != "separator" else "---"
                                   for i in range(end + 1)]


def _fill(window) -> list[str]:
    window._fill_chats_menu()
    return _menu_labels(window.chats_menu)


def _many_chats(window, backends, n: int) -> list:
    tabs = []
    for i in range(n):
        backends.add([say(f"answer {i + 1}")])
        window.go_to_new_chat()
        tab = ask(window, f"question {i + 1}")
        wait_idle(window, tab)
        tabs.append(tab)
    return tabs


# --- new chat, numbers, the menu -------------------------------------------------------------------------------
def test_send_in_new_chat_makes_chat_1_and_the_menu_entry(windows, backends):
    save_provider(LOCAL)
    backends.add([say("The slack comes from the north node.")])
    window = windows.open()
    assert chat_texts(window) == ["New chat"]
    tab = ask(window, "Why is there loss of load in my last run at the northern node of the system?")
    assert chat_texts(window)[-1] == "New chat" and len(chat_texts(window)) == 2
    assert window.current_tab() is tab                    # the focus stays in the sent chat
    wait_idle(window, tab)
    assert chat_texts(window) == ["1", "New chat"]
    pump(window, lambda: tab.created)                     # the saved list has been read
    entry = [label for label in _fill(window) if "Why is" in label]
    words = "Why is there loss of load in my last ru…"   # 39 characters + "…"
    assert entry == [f"◉ 1 · {_local(tab.created)} · {words}"]


def test_twelve_chats_in_a_narrow_window_hide_and_the_menu_shows_them(windows, backends):
    save_provider(LOCAL)
    window = windows.open()
    tabs = _many_chats(window, backends, 12)
    window.geometry("420x600")
    pump(window, lambda: window.hidden_tabs(), timeout=3)
    hidden = window.hidden_tabs()
    assert window.notebook.tab(window.new_tab, "state") == "normal"
    assert str(window.chats_button.cget("text")) == f"Chats ({len(hidden)} more) ▾"
    target = hidden[0]
    labels = _fill(window)
    index = labels.index(next(label for label in labels if f"○ {target.number} ·" in label))
    window.chats_menu.invoke(index)
    pump(window, lambda: window.current_tab() is target)
    assert window.notebook.tab(target, "state") == "normal"
    assert len(tabs) == 12


# --- the menus close like menus (OR-1): real X events, because a menu's global grab ignores event_generate -------
@pytest.mark.usefixtures("real_input")
def test_the_chats_menu_closes_on_a_click_in_the_transcript(windows):
    window = windows.open()
    take_focus(window)
    window.post_chats_menu()
    posted(window.chats_menu)
    click_on(window.current_tab().transcript)
    assert closes(window.chats_menu), "the Chats menu stayed open after a click in the transcript"


@pytest.mark.usefixtures("real_input")
def test_the_chats_menu_closes_on_a_click_in_another_window(windows, host):
    window = windows.open()
    host.geometry("300x200+700+500")      # beside the chat window, away from the menu
    host.deiconify()
    pump(host, lambda: bool(host.winfo_viewable()))
    take_focus(window)
    window.post_chats_menu()
    posted(window.chats_menu)
    click_on(host)
    assert closes(window.chats_menu), "the Chats menu stayed open after a click in another window"


@pytest.mark.usefixtures("real_input")
def test_the_chats_menu_closes_on_escape_and_its_cascades_are_made_once(windows):
    window = windows.open()
    for _ in range(3):
        take_focus(window)
        window.post_chats_menu()
        posted(window.chats_menu)
        xdotool(window, "key", "Escape")
        assert closes(window.chats_menu), "Escape left the Chats menu open"
    assert len(window.chats_menu.winfo_children()) == 2          # Reopen and Delete, not two more per opening


@pytest.mark.usefixtures("real_input")
def test_the_tab_menu_works_from_the_keyboard_and_closes_on_a_click_outside(windows, backends):
    save_provider(LOCAL)
    backends.add([say("one")], [say("two")])
    window = windows.open()
    first = ask(window, "first question")
    wait_idle(window, first)
    window.go_to_new_chat()
    second = ask(window, "second question")
    wait_idle(window, second)
    window.select(first)
    pump(window, lambda: window.current_tab() is first)
    take_focus(window)
    window._post_tab_menu(first, None)
    posted(window.tab_menu)
    click_on(first.transcript)
    assert closes(window.tab_menu), "the tab menu stayed open after a click in the transcript"
    take_focus(window)
    window._post_tab_menu(first, None)                # as Shift+F10 / the Menu key on the tab row
    posted(window.tab_menu)
    xdotool(window, "key", "Down", "Down", "Return")   # Rename chat, Close chat → Enter
    pump(window, lambda: first not in window.tabs)
    assert not window.tab_menu.winfo_ismapped()
    assert second in window.tabs


def test_ctrl_tab_reaches_hidden_chats_from_every_part_of_the_window(windows, backends):
    save_provider(LOCAL)
    window = windows.open()
    tabs = _many_chats(window, backends, 12)
    window.geometry("420x600")
    pump(window, lambda: len(window.hidden_tabs()) >= 3, timeout=3)
    order = window.all_tabs()
    window.select(tabs[0])
    pump(window, timeout=0.05)

    def step(widget_of, sequence: str, expected_step: int) -> None:
        before = window.current_tab()
        key(widget_of(window.current_tab()), sequence)
        pump(window, timeout=0.05)
        assert window.current_tab() is order[(order.index(before) + expected_step) % len(order)]
        assert window.notebook.tab(window.current_tab(), "state") == "normal"

    for _ in range(4):                                      # walks into the hidden chats
        step(lambda t: t.input, "<Control-Tab>", 1)
    step(lambda t: t.transcript.text, "<Control-Tab>", 1)
    step(lambda _t: window.notebook, "<Control-Tab>", 1)
    step(lambda t: t.send_button, "<Control-Tab>", 1)
    step(lambda t: t.input, "<Control-ISO_Left_Tab>", -1)
    step(lambda t: t.transcript.text, "<Control-Shift-ISO_Left_Tab>", -1)
    step(lambda _t: window.notebook, "<Control-ISO_Left_Tab>", -1)
    step(lambda _t: window.chats_button, "<Control-Shift-Tab>", -1)
    step(lambda t: t.input, "<Control-Next>", 1)
    step(lambda t: t.input, "<Control-Prior>", -1)


def test_right_on_the_tab_row_moves_and_keeps_the_focus(windows, backends):
    save_provider(LOCAL)
    window = windows.open()
    tabs = _many_chats(window, backends, 3)
    window.select(tabs[0])
    pump(window, timeout=0.05)
    key(window.notebook, "<Right>")
    key(window.notebook, "<Right>")
    pump(window, timeout=0.05)
    assert window.current_tab() is tabs[2]
    assert window.focus_lastfor() is window.notebook


def test_ctrl_t_and_ctrl_w(windows, backends):
    save_provider(LOCAL)
    window = windows.open()
    tabs = _many_chats(window, backends, 2)
    window.select(tabs[0])
    key(tabs[0].input, "<Control-t>")
    assert window.current_tab() is window.new_tab
    pump(window, timeout=0.05)
    assert window.focus_lastfor() is window.new_tab.input
    window.select(tabs[1])
    key(tabs[1].transcript.text, "<Control-w>")
    pump(window, timeout=0.05)
    assert tabs[1] not in window.tabs and chat_texts(window) == ["1", "New chat"]
    assert window.current_tab() is tabs[0]                  # the most recently used remaining chat
    key(window.new_tab.input, "<Control-w>")                # "New chat" cannot be closed
    assert window.new_tab in window.all_tabs()


# --- several chats at once ----------------------------------------------------------------------------------------
def test_two_chats_answer_at_the_same_time(windows, backends):
    save_provider(LOCAL, parallel=2)
    first_block, second_block = Block(then=say("one")), Block(then=say("two"))
    backends.add([first_block], [second_block])
    window = windows.open()
    first = ask(window, "first")
    window.go_to_new_chat()
    second = ask(window, "second")
    pump(window, lambda: first_block.started.is_set() and second_block.started.is_set(), timeout=5)
    assert first.busy and second.busy
    first_block.release.set()
    second_block.release.set()
    wait_idle(window, first)
    wait_idle(window, second)
    assert "one" in first.transcript.contents() and "two" in second.transcript.contents()


def test_one_notice_at_a_time_and_an_accepted_key_answers_the_others(windows, backends, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    save_provider(REMOTE)
    backends.add([say("one")], [say("two")])
    window = windows.open()
    first = ask(window, "first")
    window.go_to_new_chat()
    second = ask(window, "second")
    pump(window, lambda: window.consents.current is not None and window.consents.queued() == 1, timeout=5)
    assert len(_dialogs(window, ConsentDialog)) == 1
    asked_tab = window.consents.current[0]
    waiting = second if asked_tab is first else first
    assert waiting.status_text == "Waiting: the data notice of another chat is open."
    assert waiting.tab_text().startswith("‖ ")
    assert asked_tab.tab_text().startswith("? ")
    assert window.current_tab() is asked_tab
    dialog = window.consents.current[2]
    pump(window, lambda: "disabled" not in dialog.send_button.state(), timeout=3)
    key_accepted = window.consents.current[1].consent_key
    dialog.send_button.invoke()
    wait_idle(window, first)
    wait_idle(window, second)
    assert "one" in first.transcript.contents() and "two" in second.transcript.contents()
    assert _dialogs(window, ConsentDialog) == []
    # A later request with the same key (a chat prepared before the acceptance was written) is answered at once.
    answers = []
    monkeypatch.setattr(first, "consent_answer", lambda job_id, accepted: answers.append((job_id, accepted)))
    window.consents.request(first, ConsentNeeded(99, key_accepted, "Google Gemini", "notice", "indicator"))
    assert answers == [(99, True)]
    assert _dialogs(window, ConsentDialog) == []


# --- closing and reopening ----------------------------------------------------------------------------------------
def test_a_chat_closed_mid_answer_lets_its_worker_go(windows, backends):
    save_provider(LOCAL)
    block = Block(then=say("late"))
    backends.add([say("Saved first."), block], [say("Again.")])
    window = windows.open()
    tab = ask(window, "first")
    wait_idle(window, tab)
    ask(window, "second", tab)
    assert block.started.wait(5)
    worker, conversation_id = tab.worker, tab.conversation_id
    started = time.monotonic()
    window.close_chat(tab)
    assert worker.exited.wait(2.0)
    assert time.monotonic() - started <= 2.0
    window.reopen(conversation_id)
    pump(window, lambda: any(t.conversation_id == conversation_id and t.has_messages for t in window.tabs), 5)
    reopened = window.tab_for(conversation_id)
    assert "Saved first." in reopened.transcript.contents()
    assert window.current_tab() is reopened


def test_an_immediate_reopen_waits_for_the_closed_chat(windows, backends, tools):
    save_provider(LOCAL)
    backends.add([say("Saved first."), tool_call("slow"), say("after the tool")])
    window = windows.open()
    tab = ask(window, "first")
    wait_idle(window, tab)
    ask(window, "use the tool", tab)
    assert tools.slow_running.wait(5)
    conversation_id = tab.conversation_id
    window.close_chat(tab)
    assert finishing_worker(conversation_id) is not None
    window.reopen(conversation_id)
    pump(window, timeout=0.3)
    assert str(window.note_label.cget("text")) == "Waiting for the closed chat to finish…"
    assert window.tab_for(conversation_id) is None
    tools.release.set()
    pump(window, lambda: window.tab_for(conversation_id) is not None
         and window.tab_for(conversation_id).has_messages, timeout=5)
    assert "Saved first." in window.tab_for(conversation_id).transcript.contents()


def test_a_reopen_gives_up_after_the_wait(windows, backends, tools, monkeypatch):
    monkeypatch.setattr(chat_window_module, "REOPEN_WAIT_S", 0.5)
    save_provider(LOCAL)
    backends.add([say("Saved first."), tool_call("slow"), say("after the tool")])
    window = windows.open()
    tab = ask(window, "first")
    wait_idle(window, tab)
    ask(window, "use the tool", tab)
    assert tools.slow_running.wait(5)
    conversation_id = tab.conversation_id
    window.close_chat(tab)
    window.reopen(conversation_id)
    pump(window, lambda: str(window.note_label.cget("text")) == ("This chat is still finishing a tool call. Try "
                                                                 "again in a moment."), timeout=3)
    assert window.tab_for(conversation_id) is None


def test_a_name_given_during_a_tool_call_survives_the_close(windows, backends, tools):
    save_provider(LOCAL)
    backends.add([say("Saved first."), tool_call("slow"), say("after the tool")])
    window = windows.open()
    tab = ask(window, "first")
    wait_idle(window, tab)
    ask(window, "use the tool", tab)
    assert tools.slow_running.wait(5)
    window.rename(tab)
    (dialog,) = _dialogs(window, RenameChatDialog)
    dialog.var.set("Hydrogen prices")
    dialog.ok()
    assert chat_texts(window)[0].endswith("Hydrogen prices")
    store = open_store()
    pump(window, lambda: any(i.named for i in store.list_conversations(windows.project)), timeout=3)
    worker = tab.worker
    window.close(ask=False)
    tools.release.set()
    assert worker.exited.wait(5)
    (info,) = store.list_conversations(windows.project)
    assert info.named and info.title == "Hydrogen prices"


def test_reopening_the_window_restores_the_chats_and_the_selection(windows, backends):
    save_provider(LOCAL)
    window = windows.open()
    first, second = _many_chats(window, backends, 2)
    window.select(first)
    ids = [first.conversation_id, second.conversation_id]
    workers = window.workers()
    window.close(ask=False)
    window._last_open_chats_write.join(3)
    for worker in workers:
        assert worker.exited.wait(5)
    again = windows.open()
    pump(again, lambda: len([t for t in again.tabs if t.has_messages]) == 2, timeout=5)
    assert [t.conversation_id for t in again.tabs] == ids
    assert chat_texts(again) == ["1", "2", "New chat"]
    pump(again, lambda: again.current_tab().conversation_id == ids[0], timeout=3)
    assert "answer 1" in again.current_tab().transcript.contents()


def test_at_retention_zero_closing_asks_first(windows, backends):
    save_provider(LOCAL, expiry=0)
    backends.add([say("Not kept.")], [say("Neither.")])
    window = windows.open()
    tab = ask(window, "first")
    wait_idle(window, tab)
    window.close_chat(tab)
    (dialog,) = _dialogs(window, ConfirmDialog)
    assert str(dialog.message_label.cget("text")) == "This chat isn't saved. Close it anyway?"
    assert [str(dialog.action_button.cget("text")), str(dialog.cancel_button.cget("text"))] == ["Close", "Cancel"]
    dialog.cancel()
    assert tab in window.tabs
    window.close_chat(tab)
    (dialog,) = _dialogs(window, ConfirmDialog)
    dialog.action_button.invoke()
    assert tab not in window.tabs
    window.go_to_new_chat()
    one = ask(window, "one")
    wait_idle(window, one)
    backends.add([say("Third.")])
    window.go_to_new_chat()
    two = ask(window, "two")
    wait_idle(window, two)
    window.close()
    (dialog,) = _dialogs(window, ConfirmDialog)
    assert str(dialog.message_label.cget("text")) == "2 chats aren't saved. Close them anyway?"
    dialog.cancel()
    assert not window.closing
    labels = _fill(window)
    assert "Reopen a saved chat — Chats are not saved (see Settings)." in labels


def test_a_chat_open_in_another_process_is_not_restored(windows, backends, user_config_in_tmp):
    import multiprocessing

    from tests.assistant import store_workers

    save_provider(LOCAL)
    window = windows.open()
    first, second = _many_chats(window, backends, 2)
    held_id = second.conversation_id
    workers = window.workers()
    window.close(ask=False)
    window._last_open_chats_write.join(3)
    for worker in workers:
        assert worker.exited.wait(5)
    spawn = multiprocessing.get_context("spawn")
    ready, finish = spawn.Event(), spawn.Event()
    proc = spawn.Process(target=store_workers.hold_open,
                         args=(str(user_config_in_tmp), str(windows.project), held_id, ready, finish))
    proc.start()
    try:
        assert ready.wait(60)
        again = windows.open()
        note = "1 chat is open in another FlexTool window or terminal and was not restored."
        pump(again, lambda: note in again.new_tab.transcript.contents(), timeout=10)
        assert [t.conversation_id for t in again.tabs] == [first.conversation_id]
    finally:
        finish.set()
        proc.join(30)


# --- size and place -----------------------------------------------------------------------------------------------
def test_the_window_takes_the_work_area_height_beside_the_execution_window(windows):
    from tkinter import font as tkfont

    from flextool.gui.assistant.geometry import SAMPLE, CHARACTERS, FIXED_CHARACTERS, parse_geometry

    save_provider(LOCAL)
    window = windows.open(work_area=lambda _m: (100, 20, 1600, 900), exec_rect=lambda: (1000, 20, 700, 900))
    pump(window, timeout=0.3)
    x, y, w, h = parse_geometry(window.wm_geometry())
    decoration = window.measure_decoration()
    assert (x, y) == (100, 20)                                    # the side the Execution window leaves free
    assert h + decoration.top + decoration.border == 900
    from flextool.gui.assistant.fonts import CHAT_FONT, FONTS

    avg = tkfont.nametofont(CHAT_FONT).measure(SAMPLE) / len(SAMPLE)          # the transcript's fonts
    fixed = tkfont.nametofont(FONTS["code"]).measure("0")
    assert w >= max(avg * CHARACTERS, fixed * FIXED_CHARACTERS)
    assert w + 2 * decoration.border <= 1600


def test_the_tab_and_transcript_paddings_follow_the_dpi(windows, monkeypatch):
    """The tab-fit padding and the transcript's chrome are pixels at 96 DPI, scaled at the screen's DPI and never
    below the 96-DPI value (review G19, C7-n6)."""
    from flextool.gui import ui_metrics

    save_provider(LOCAL)
    monkeypatch.setattr(ui_metrics, "monitor_dpi", lambda _widget: 192.0)
    window = windows.open(work_area=lambda _m: (0, 0, 1600, 900))
    assert window._placement[4] == 4 * window._cw + 80
    assert window._tab_pad() == 2 * chat_window_module.TAB_PAD
    monkeypatch.setattr(ui_metrics, "monitor_dpi", lambda _widget: 72.0)
    assert window._tab_pad() == chat_window_module.TAB_PAD


def test_a_resize_is_remembered_per_monitor_configuration(windows):
    from types import SimpleNamespace

    from flextool.gui.ui_metrics import monitor_signature

    save_provider(LOCAL)
    settings = SimpleNamespace(assistant_window_geometry={}, assistant_window_layout_cw=0)
    window = windows.open(global_settings=settings)
    window.geometry("700x600+30+40")
    pump(window, lambda: window.wm_geometry() == "700x600+30+40", timeout=3)
    signature = monitor_signature(window)
    window.close(ask=False)
    assert settings.assistant_window_geometry == {signature: "700x600+30+40"}
    assert settings.assistant_window_layout_cw > 0
    again = windows.open(global_settings=settings)
    pump(again, timeout=0.2)
    assert again.wm_geometry().startswith("700x600")


def test_the_work_area_on_x11_is_cut_to_the_monitor(tk_root, monkeypatch):
    from flextool.gui import ui_metrics

    if tk_root.tk.call("tk", "windowingsystem") != "x11":
        pytest.skip("X11 only")
    monkeypatch.setattr(ui_metrics, "_monitor_bounds_screeninfo", lambda _w: (1920, 0, 1920, 1080))
    values = {b"_NET_WORKAREA": [0, 0, 3840, 1040, 0, 30, 3840, 1010], b"_NET_CURRENT_DESKTOP": [1]}
    monkeypatch.setattr(ui_metrics, "_x11_cardinals", lambda name: values.get(name))
    assert ui_metrics.monitor_work_area(tk_root) == (1920, 30, 1920, 1010)
    monkeypatch.setattr(ui_metrics, "_x11_cardinals", lambda name: None)
    lh = ui_metrics.get_metrics(tk_root).lh
    assert ui_metrics.monitor_work_area(tk_root) == (1920, 0, 1920, 1080 - 4 * lh)


# --- the selected chat's model in the header (owner GUI feedback 2026-11-08 (2)) ----------------------------------
def test_the_header_shows_the_model_of_the_selected_chat(windows, backends):
    from flextool.assistant.settings import ProviderSettings

    from flextool.assistant.probes import ModelFacts

    save_provider(LOCAL)
    backends.add([say("One.")])
    window = windows.open(probe=lambda *_a, **_k: ModelFacts(context_window=131_072, source="test"))
    assert str(window.model_label.cget("text")) == "Custom endpoint · fake-model · basic"
    first = ask(window, "First question")
    wait_idle(window, first)
    assert str(window.model_label.cget("text")) == "Custom endpoint · fake-model · basic"
    assert str(window.route_label.cget("text")).startswith("Endpoint on this computer (127.0.0.1:9)")
    # Another model in the settings: the answered chat keeps its own, a new chat shows (and uses) the new one.
    save_provider(ProviderSettings(preset="openai_compatible", base_url="http://127.0.0.2:9/v1",
                                   model="gpt-5", tier="standard"))
    window.settings_changed()
    assert str(window.model_label.cget("text")) == "Custom endpoint · fake-model · basic"
    window.go_to_new_chat()
    pump(window, lambda: window.current_tab() is window.new_tab)
    assert str(window.model_label.cget("text")) == "Custom endpoint · gpt-5 · standard"
    assert str(window.route_label.cget("text")).startswith("Endpoint on this computer (127.0.0.2:9)")
    backends.add([say("Two.")])
    second = ask(window, "Second question")
    wait_idle(window, second)
    assert backends.made[-1].origin[2] == "gpt-5"
    assert str(window.model_label.cget("text")) == "Custom endpoint · gpt-5 · standard"
    window.select(first)
    pump(window, lambda: window.current_tab() is first)
    assert str(window.model_label.cget("text")) == "Custom endpoint · fake-model · basic"
    assert str(window.route_label.cget("text")).startswith("Endpoint on this computer (127.0.0.1:9)")


# --- the Tk thread does no store or cache I/O (review G18, C7-n5 and critique M2) -------------------------------
def test_a_window_opened_right_after_a_rename_restores_the_moved_chats(tk_root, tmp_path, backends, tools,
                                                                       monkeypatch):
    """The rename returns while the store is still held (the chats move on the project-hook thread); the new
    name's window waits for the move before it reads its open chats, so they are restored and not overwritten.
    Before, the rename waited on the Tk thread for the whole time the store was held."""
    import threading

    from flextool.assistant.store import Store
    from flextool.gui import project_utils
    from tests.gui.assistant.window_rig import Windows

    monkeypatch.chdir(tmp_path)
    alpha = tmp_path / "projects" / "Alpha"
    alpha.mkdir(parents=True)
    save_provider(LOCAL)
    first = Windows(tk_root, alpha, backends)
    second = Windows(tk_root, tmp_path / "projects" / "Beta", backends)
    free = threading.Event()
    real = Store.rename_project

    def held(self, old, new):  # noqa: ANN001, ANN202
        assert free.wait(10)
        return real(self, old, new)

    try:
        window = first.open()
        (chat,) = _many_chats(window, backends, 1)
        chat_id = chat.conversation_id
        workers = window.workers()
        window.close(ask=False)
        window._last_open_chats_write.join(3)
        for worker in workers:
            assert worker.exited.wait(5)
        monkeypatch.setattr(Store, "rename_project", held)
        timer = threading.Timer(1.5, free.set)
        timer.start()
        start = time.monotonic()
        project_utils.rename_project("Alpha", "Beta")
        assert time.monotonic() - start < 1.0
        again = second.open()
        pump(again, timeout=0.2)
        assert [t.conversation_id for t in again.tabs if t.conversation_id] == []    # waiting for the move
        pump(again, lambda: [t.conversation_id for t in again.tabs if t.has_messages] == [chat_id], timeout=8)
        assert "answer 1" in again.tabs[0].transcript.contents()
        timer.join()
    finally:
        free.set()
        first.close_all()
        second.close_all()


def test_window_open_reads_no_cache_on_the_tk_thread(windows, monkeypatch):
    """The facts FlexTool last found for the settings' model are a file read: made on a daemon thread, the
    header's level follows when they come.  gpt-5's name gives advanced; a 32 768-token window allows standard
    at most (``tiers.WINDOW_LEVELS``)."""
    import os
    import threading

    from flextool.assistant import probes
    from flextool.assistant.probes import ModelFacts
    from flextool.assistant.settings import ProviderSettings

    save_provider(ProviderSettings(preset="openai_compatible", base_url="http://127.0.0.1:9/v1", model="gpt-5"))
    seen: list[tuple[str, threading.Thread]] = []
    real_mkdir, real_makedirs = os.mkdir, os.makedirs

    def facts(*_args, **_kwargs) -> ModelFacts:
        seen.append(("cached_facts", threading.current_thread()))
        return ModelFacts(context_window=32_768, source="test")

    def mkdir(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        seen.append(("mkdir", threading.current_thread()))
        return real_mkdir(*args, **kwargs)

    def makedirs(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        seen.append(("makedirs", threading.current_thread()))
        return real_makedirs(*args, **kwargs)

    monkeypatch.setattr(probes, "cached_facts", facts)
    monkeypatch.setattr(os, "mkdir", mkdir)
    monkeypatch.setattr(os, "makedirs", makedirs)
    window = windows.open()
    pump(window, lambda: str(window.model_label.cget("text")) == "Custom endpoint · gpt-5 · standard", timeout=5)
    window.settings_changed()
    assert str(window.model_label.cget("text")) == "Custom endpoint · gpt-5 · advanced"    # from the name
    pump(window, lambda: str(window.model_label.cget("text")) == "Custom endpoint · gpt-5 · standard", timeout=5)
    main = threading.main_thread()
    assert [kind for kind, thread in seen if thread is main] == []
    assert len([kind for kind, _t in seen if kind == "cached_facts"]) >= 2
