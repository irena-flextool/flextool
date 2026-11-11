"""The welcome, the data notice and the rename dialog (P2 §13.2 test 8, critique M2 and m8).

* The welcome comes before the chat window and before the settings; closing it records that it was seen, the
  "How this works" link records nothing; opening it sends nothing; its text is the owner-approved text of P2 §8.2,
  copied here.
* The data notice starts with the focus on Cancel; Return, an early click on [Send to …] and the space bar right
  after it opens send nothing; [Send to …] after the guard records the acceptance and the request goes out; a
  proxy change asks again.
* A dialog opened while another window holds the grab gives that grab back when it closes, also when it is
  destroyed with its parent.

Run headless:  xvfb-run -a python -m pytest tests/gui/assistant --run-gui
"""
from __future__ import annotations

import json
import socket
import time
import tkinter as tk
from pathlib import Path

import pytest

from flextool.assistant import endpoint
from flextool.gui.assistant.notices import (
    ConsentDialog,
    RenameChatDialog,
    WelcomeDialog,
    with_welcome,
)
from flextool.gui.assistant.worker import ConsentAnswer, ConsentNeeded, Idle, TurnFinished, TurnRefused
from tests.assistant.gui_support.worker_rig import REMOTE, Rig, Tools, save_provider, say
from tests.assistant.llm.fake_backend import FakeBackend
from tests.gui.assistant.tk_helpers import key, pump

pytestmark = pytest.mark.gui

#: P2 §8.2, the owner-approved welcome (P2-O6), with the settings symbol of Linux.
WELCOME = """How the AI assistant works

Nothing is sent until you ask
Opening FlexTool or this window sends nothing to any AI service. Your data leaves this computer only when you send \
a question in the chat.

What the assistant does
It reads what it needs from your input data, results and run logs with read-only tools and gives advice. It cannot \
change your model.

What you need
An account with an AI provider and its key, which you paste in Settings (⚙), or an AI model (LLM) running on this \
computer. Free plans work, but they limit how many questions you can send and often let the provider use your data \
for training. Paid plans have higher limits and may more often let you limit how your data is used. Check the \
provider's current policies yourself. A model running on this computer keeps your data here, but it needs a powerful \
computer. Some models offered through programs on this computer still run elsewhere, such as Ollama's cloud models; \
FlexTool tells you when that happens.

Assistant level
FlexTool picks a level (basic, standard or advanced) from the model name and its context size. Smaller models get \
simpler tasks, such as finding and explaining values; larger models are also asked to work through problems in \
several steps, such as why a run has slack. You can choose a lower level in Settings."""


@pytest.fixture
def connects(monkeypatch: pytest.MonkeyPatch) -> list:
    """Every connection attempt (none may happen while only the welcome is open)."""
    seen: list = []
    real = socket.socket.connect

    def connect(self, address):  # noqa: ANN001, ANN202 — socket's signature
        seen.append(address)
        return real(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    return seen


def _seen_soon(timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if endpoint.welcome_seen():
            return True
        time.sleep(0.02)
    return False


# --- the welcome --------------------------------------------------------------------------------------------------
def test_the_welcome_is_the_approved_text(host):
    dialog = WelcomeDialog(host, record=False)
    try:
        assert dialog.text() == WELCOME
        assert dialog.title() == "How the AI assistant works"
        buttons = (dialog.chat_button, dialog.settings_button, dialog.close_button)
        assert [str(b.cget("text")) for b in buttons] == ["Open chat", "Open settings", "Close"]
        dialog.update_idletasks()
        assert [b.winfo_x() for b in buttons] == sorted(b.winfo_x() for b in buttons)      # in this order
    finally:
        dialog.finish("close")


def test_the_assistant_dialogs_use_the_chat_fonts(host):
    """The welcome, the data notice and the rename dialog are in the chat's fonts: text 11 pt at the GUI's
    default 10 pt (one point above it), section headings bold at 11, the welcome's title 13 bold (a ``##``
    heading of an answer)."""
    import tkinter.font as tkfont

    from flextool.gui.assistant.notices import ConsentDialog, RenameChatDialog

    base = tkfont.nametofont("TkTextFont", root=host)
    saved = base.cget("size")
    base.configure(size=10)                     # the GUI's default body size
    dialog = WelcomeDialog(host, record=False)
    try:
        title, heading, body = (tkfont.Font(root=host, font=dialog.fonts[i]).actual() for i in range(3))
        assert body["size"] == 11 and body["weight"] == "normal"
        assert heading["size"] == 11 and heading["weight"] == "bold"
        assert title["size"] == 13 and title["weight"] == "bold"
        head, text = dialog.section_labels[0]
        assert str(text.cget("font")) == str(dialog.fonts.body) and str(head.cget("font")) == str(dialog.fonts.heading)
    finally:
        dialog.finish("close")
    notice = ConsentDialog(host, "Example", "The notice.", "Sent to Example.", "1", lambda _a: None)
    rename = None
    try:
        size = tkfont.Font(root=host, font=notice.notice_label.cget("font")).actual("size")
        assert size == 11
        notice.cancel()
        rename = RenameChatDialog(host, "Chat 1", lambda _n: None)
        assert tkfont.Font(root=host, font=rename.entry.cget("font")).actual("size") == 11
    finally:
        notice.cancel()
        if rename is not None:
            rename.cancel()
        base.configure(size=saved)
        import gc

        rename = notice = None
        gc.collect()                            # the dialogs' Tk variables go here, on the Tk thread


def test_the_welcome_comes_before_the_chat_and_close_records_it(host, connects):
    opened: list[str] = []
    dialog = with_welcome(host, lambda: opened.append("chat"))
    assert isinstance(dialog, WelcomeDialog)
    pump(host, lambda: dialog.grab.held)
    assert opened == []
    assert not endpoint.welcome_seen()
    dialog.close_button.invoke()
    pump(host, timeout=0.2)
    assert opened == []                             # Close opens nothing
    assert _seen_soon()
    assert connects == []
    # Seen once: the next open goes straight to the chat.
    assert with_welcome(host, lambda: opened.append("again")) is None
    assert opened == ["again"]


def test_escape_closes_the_welcome_and_opens_nothing(host):
    opened: list[str] = []
    dialog = with_welcome(host, lambda: opened.append("chat"), open_settings=lambda: opened.append("settings"))
    pump(host, lambda: dialog.grab.held)
    key(dialog, "<Escape>")
    assert _seen_soon()
    pump(host, timeout=0.2)
    assert opened == []


def test_open_settings_from_the_welcome(host):
    opened: list[str] = []
    dialog = with_welcome(host, lambda: opened.append("chat"), open_settings=lambda: opened.append("settings"),
                          needed=True)
    dialog.settings_button.invoke()
    assert opened == ["settings"]


def test_open_chat_waits_for_a_provider_in_the_settings(host):
    """No provider saved (always so at the very first open): [Open chat] is grey, says why and does nothing."""
    opened: list[str] = []
    dialog = with_welcome(host, lambda: opened.append("then"), open_chat=lambda: opened.append("chat"),
                          needed=True)
    try:
        assert dialog.chat_button.instate(["disabled"]) and not dialog.chat_enabled
        assert dialog.chat_tooltip is not None and dialog.chat_tooltip._text == "Check the settings first."
        dialog.finish("chat")                       # as a click that got through: still nothing
        assert opened == [] and dialog.winfo_exists()
        assert not dialog.settings_button.instate(["disabled"])
    finally:
        dialog.close_button.invoke()
    assert opened == []


def test_open_chat_with_a_saved_provider(host):
    save_provider(REMOTE)
    opened: list[str] = []
    dialog = with_welcome(host, lambda: opened.append("then"), open_chat=lambda: opened.append("chat"),
                          open_settings=lambda: opened.append("settings"), needed=True)
    assert not dialog.chat_button.instate(["disabled"]) and dialog.chat_tooltip is None
    dialog.chat_button.invoke()
    assert opened == ["chat"]
    # The link shows the same buttons, both enabled.
    again = WelcomeDialog(host, record=False)
    try:
        assert not again.chat_button.instate(["disabled"]) and not again.settings_button.instate(["disabled"])
    finally:
        again.finish("close")


def test_the_link_shows_the_welcome_without_recording(host):
    dialog = WelcomeDialog(host, record=False)
    pump(host, lambda: dialog.grab.held)
    dialog.close_button.invoke()
    pump(host, timeout=0.3)
    assert not endpoint.welcome_seen()


# --- the data notice ----------------------------------------------------------------------------------------------
@pytest.fixture
def tools(monkeypatch: pytest.MonkeyPatch):
    fake = Tools()
    fake.install(monkeypatch)
    yield fake
    fake.release.set()


@pytest.fixture
def remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tools):
    """A chat worker set up for a remote provider (Gemini, key from the environment), and its cleanup."""
    for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    save_provider(REMOTE)
    project = tmp_path / "project"
    project.mkdir()
    made: list[Rig] = []

    def make(script: list) -> Rig:
        rig = Rig(project, [FakeBackend(script)])
        made.append(rig)
        return rig

    yield make
    for rig in made:
        rig.close()


def _notice_for(host: tk.Misc, rig: Rig) -> tuple[ConsentDialog, list[bool]]:
    job = rig.job()
    event = rig.wait_for(ConsentNeeded)
    answers: list[bool] = []

    def answered(accepted: bool) -> None:
        answers.append(accepted)
        rig.worker.submit(ConsentAnswer(job.job_id, accepted))

    dialog = ConsentDialog(host, event.provider_label, event.notice, event.indicator, "1", answered)
    return dialog, answers


def test_the_notice_starts_on_cancel_and_early_keys_send_nothing(host, remote):
    rig = remote([say("never")])
    dialog, answers = _notice_for(host, rig)
    pump(host, lambda: dialog.winfo_ismapped() and dialog.grab.held)
    assert dialog.title() == "Before sending to Google Gemini"
    assert str(dialog.send_button.cget("text")) == "Send to Google Gemini"
    assert str(dialog.chat_label.cget("text")) == "Chat 1"
    assert "generativelanguage.googleapis.com" in str(dialog.notice_label.cget("text"))
    assert str(dialog.indicator_label.cget("text")) == ("Data goes to: Google Gemini "
                                                        "(generativelanguage.googleapis.com).")
    assert dialog.focus_lastfor() is dialog.cancel_button
    key(dialog.cancel_button, "<Return>")
    key(dialog, "<Return>")
    key(dialog, "<KP_Enter>")
    dialog.send_button.invoke()                    # within the first 500 ms: ignored
    pump(host, timeout=0.1)
    assert answers == []
    key(dialog.cancel_button, "<space>")  # the space bar activates the focused button: Cancel
    pump(host, lambda: answers)
    assert answers == [False]
    assert rig.wait_for(TurnRefused).reason == "Nothing was sent."
    rig.wait_for(Idle)
    assert rig.made == []


def test_send_after_the_guard_records_the_acceptance_and_sends(host, remote):
    rig = remote([say("Hello.")])
    dialog, answers = _notice_for(host, rig)
    pump(host, lambda: "disabled" not in dialog.send_button.state(), timeout=3.0)
    dialog.send_button.invoke()
    assert answers == [True]
    assert rig.wait_for(TurnFinished).result.text == "Hello."
    assert len(rig.requests()) == 1
    data = json.loads(endpoint.acks_path().read_text(encoding="utf-8"))
    assert [a["preset"] for a in data["acks"]] == ["gemini"]
    assert data["acks"][0]["proxy"] is None


def test_a_proxy_change_asks_again(host, remote, monkeypatch):
    rig = remote([say("Hello.")])
    dialog, answers = _notice_for(host, rig)
    pump(host, lambda: "disabled" not in dialog.send_button.state())
    dialog.send_button.invoke()
    rig.wait_for(TurnFinished)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.test:3128")
    monkeypatch.setenv("https_proxy", "http://proxy.example.test:3128")
    again = remote([say("never")])
    dialog2, answers2 = _notice_for(host, again)
    assert "proxy.example.test" in str(dialog2.notice_label.cget("text"))
    dialog2.cancel_button.invoke()
    assert answers2 == [False]
    again.wait_for(Idle)
    assert again.made == []


def test_a_notice_destroyed_with_its_parent_answers_no(tk_root, remote):
    parent = tk.Toplevel(tk_root)
    parent.geometry("300x200+40+40")
    parent.update()
    rig = remote([say("never")])
    dialog, answers = _notice_for(parent, rig)
    pump(parent, lambda: dialog.grab.held)
    parent.destroy()
    assert answers == [False]
    rig.wait_for(Idle)
    assert tk_root.grab_current() is None


# --- the grab -----------------------------------------------------------------------------------------------------
def _holder(tk_root: tk.Tk) -> tk.Toplevel:
    other = tk.Toplevel(tk_root)
    other.geometry("200x100+400+400")
    other.update()
    pump(other, lambda: other.winfo_viewable())
    other.grab_set()
    return other


def test_closing_a_dialog_gives_the_earlier_grab_back(tk_root, host):
    other = _holder(tk_root)
    try:
        names: list[str | None] = []
        dialog = RenameChatDialog(host, "3", names.append)
        pump(host, lambda: dialog.grab.held)
        assert tk_root.grab_current() is dialog
        dialog.cancel()
        assert names == [None]
        assert tk_root.grab_current() is other
    finally:
        other.grab_release()
        other.destroy()


def test_a_dialog_destroyed_with_its_parent_gives_the_grab_back(tk_root):
    other = _holder(tk_root)
    parent = tk.Toplevel(tk_root)
    parent.geometry("300x200+40+40")
    parent.update()
    try:
        dialog = WelcomeDialog(parent, record=False)
        pump(parent, lambda: dialog.grab.held)
        parent.destroy()
        assert tk_root.grab_current() is other
    finally:
        other.grab_release()
        other.destroy()


# --- the rename dialog --------------------------------------------------------------------------------------------
def test_rename_keeps_at_most_60_characters_and_empty_means_the_number(host):
    names: list[str | None] = []
    dialog = RenameChatDialog(host, "Hydrogen", names.append)
    pump(host, lambda: dialog.grab.held)
    assert dialog.title() == "Rename chat"
    assert dialog.var.get() == "Hydrogen"
    dialog.entry.delete(0, "end")
    dialog.entry.insert(0, "x" * 70)                # too long: refused whole
    assert dialog.var.get() == ""
    dialog.entry.insert(0, "x" * 60)
    dialog.entry.insert("end", "y")
    assert dialog.var.get() == "x" * 60
    dialog.var.set("  Hydrogen   prices ")
    key(dialog.entry, "<Return>")
    assert names == ["Hydrogen prices"]
    empty = RenameChatDialog(host, "Hydrogen prices", names.append)
    empty.var.set("")
    empty.ok()
    assert names == ["Hydrogen prices", ""]
    escaped = RenameChatDialog(host, "x", names.append)
    key(escaped, "<Escape>")
    assert names == ["Hydrogen prices", "", None]


def test_typing_in_a_dialog_reaches_no_main_window_shortcut(tk_root, host):
    hits: list[str] = []
    tk_root.bind_all("<Key-e>", lambda _e: hits.append("e"))
    tk_root.bind_all("<F9>", lambda _e: hits.append("F9"))
    try:
        dialog = RenameChatDialog(host, "", lambda _name: None)
        pump(host, lambda: dialog.grab.held)
        key(dialog.entry, "<Key-e>")
        key(dialog.entry, "<F9>")
        pump(host, timeout=0.1)
        assert hits == []
        assert dialog.var.get() == "e"
        dialog.cancel()
    finally:
        tk_root.unbind_all("<Key-e>")
        tk_root.unbind_all("<F9>")



# --- a closed dialog and the garbage collector (review G18, C7-m5) ---------------------------------------------
def _retired(root: tk.Misc, var: tk.Variable, name: str) -> bool:
    return var._tk is None and not root.tk.getboolean(root.tk.call("info", "exists", name))


def test_a_closed_dialog_retires_its_variables_and_a_worker_collects_it_at_once(tk_root, host):
    """Closed, the rename dialog lives on in reference cycles (its widgets name it as master).  Its variable is
    retired at the close, on the Tk thread, so a worker that collects the dialog needs no Tcl call.  Before, the
    variable's ``__del__`` on the worker waited for the Tk thread for the whole 3 s wait."""
    import gc
    import weakref

    from tests.gui.assistant.tk_helpers import collect_on_a_worker

    gc.collect()
    gc.disable()
    try:
        names: list[str | None] = []
        dialog = RenameChatDialog(host, "Hydrogen", names.append)
        pump(host, lambda d=dialog: d.grab.held)
        var, name = dialog.var, str(dialog.var)
        dialog.cancel()
        assert names == [None]
        assert _retired(tk_root, var, name)
        ref = weakref.ref(dialog)
        del dialog, var
        pump(host, timeout=0.05)
        assert ref() is not None                    # kept by its cycles: the collector's to free
        took = collect_on_a_worker(tk_root, wait=3.0)
    finally:
        gc.enable()
    assert ref() is None
    assert took < 0.5


def test_a_dialog_destroyed_with_its_parent_retires_its_variables(tk_root):
    parent = tk.Toplevel(tk_root)
    parent.geometry("300x200+40+40")
    parent.update()
    names: list[str | None] = []
    dialog = RenameChatDialog(parent, "Hydrogen", names.append)
    pump(parent, lambda: dialog.grab.held)
    var, name = dialog.var, str(dialog.var)
    parent.destroy()
    assert _retired(tk_root, var, name)
    assert dialog.grab.dialog is None and tk_root.grab_current() is None
