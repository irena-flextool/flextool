"""A chat tab's life and its header (review G17: C7-M1, m1, m2, m3, m11, n1, n2, n3; G17-D1..D4).

The header shows what the chat's worker says it goes with (also when the settings are saved during the first answer
or while the first question waits for the data notice); a chat that opens takes no question; a chat closed before
its worker took the question sends nothing; a handler that fails never stops the window's poll and never leaves a
tab opening for good; a restore always resolves; a stopped job's lines settle by the worker's word; the footer
names the selected chat's provider.

The only synchronisation is ``threading.Event`` gates and ``pump(until)``.

Run headless:  xvfb-run -a python -m pytest tests/gui/assistant --run-gui
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from flextool.assistant.llm_errors import AuthFailed, ErrorContext, internal_fault_message, make_error
from flextool.assistant.probes import ModelFacts
from flextool.assistant.settings import ProviderSettings
from flextool.assistant.store import ConversationLocked, Store, open_store
from flextool.gui.assistant import chat_window as chat_window_module
from flextool.gui.assistant import strings
from flextool.gui.assistant.chat_tab import ChatTab
from flextool.gui.assistant.transcript import Transcript
from flextool.gui.assistant.worker import AssistantWorker
from tests.assistant.gui_support.worker_rig import (
    LOCAL,
    REMOTE,
    Tools,
    conversation_file,
    save_provider,
    say,
    tool_call,
)
from tests.assistant.llm.fake_backend import Block
from tests.gui.assistant.tk_helpers import pump
from tests.gui.assistant.window_rig import Backends, Windows, ask, wait_idle

pytestmark = pytest.mark.gui

LOCAL_MODEL_LINE = "Custom endpoint · fake-model · basic"
LOCAL_ROUTE = "Endpoint on this computer (127.0.0.1:9)"
OLLAMA_OTHER = ProviderSettings(preset="ollama", model="llama-other")
OPUS = ProviderSettings(preset="anthropic", model="claude-opus-5-5")


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


def _header(window) -> tuple[str, str]:
    return str(window.model_label.cget("text")), str(window.route_label.cget("text"))


def _saved_chats(windows, backends, n: int) -> list[str]:
    """*n* chats answered on the LOCAL endpoint, saved, the window closed (its open chats are these)."""
    save_provider(LOCAL)
    window = windows.open(restore=False)
    ids = []
    for i in range(n):
        backends.add([say(f"Saved {i + 1}.")])
        window.go_to_new_chat()
        tab = ask(window, f"question {i + 1}")
        wait_idle(window, tab)
        ids.append(tab.conversation_id)
    workers = window.workers()
    window.close(ask=False)
    window._last_open_chats_write.join(3)
    for worker in workers:
        assert worker.exited.wait(5)
    return ids


def _set_origin(project: Path, conversation_id: str, origin: list[str]) -> None:
    path = conversation_file(project, conversation_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["origin"] = origin
    path.write_text(json.dumps(data), encoding="utf-8")


def _gate_open(monkeypatch: pytest.MonkeyPatch) -> threading.Event:
    """``AssistantWorker._open`` waits until the returned event is set."""
    gate = threading.Event()
    original = AssistantWorker._open

    def gated(self, conversation_id: str) -> None:
        assert gate.wait(10)
        original(self, conversation_id)

    monkeypatch.setattr(AssistantWorker, "_open", gated)
    return gate


def _deferred_workers(window) -> list[AssistantWorker]:
    """The window's next workers start only when the test calls their ``start_now``."""
    made: list[AssistantWorker] = []
    original = window.make_worker

    def make() -> AssistantWorker:
        worker = original()
        worker.start_now = worker.start
        worker.start = lambda: None
        made.append(worker)
        return worker

    window.make_worker = make
    return made


def _accept_notice(window) -> None:
    dialog = window.consents.current[2]
    pump(window, lambda: "disabled" not in dialog.send_button.state(), timeout=3)
    dialog.send_button.invoke()


# --- the header: the worker owns it (C7-M1, G17-D1) ---------------------------------------------------------------
def test_settings_saved_during_the_first_answer_keep_the_header_on_the_chats_model(windows, backends):
    save_provider(LOCAL)
    block = Block(then=say("One."))
    backends.add([block, say("Two.")])
    window = windows.open()
    tab = ask(window, "First question")
    pump(window, lambda: block.started.is_set() and tab.transcript.text.tag_ranges("user_pending") == (),
         timeout=5)                                     # the answer has started
    shown = _header(window)
    assert shown[0] == LOCAL_MODEL_LINE and shown[1].startswith(LOCAL_ROUTE)
    save_provider(OLLAMA_OTHER)
    window.settings_changed()
    assert _header(window) == shown                     # the answer goes on with the chat's model
    block.release.set()
    wait_idle(window, tab)
    assert _header(window) == shown
    ask(window, "Second question", tab)
    wait_idle(window, tab)
    assert [b.origin[2] for b in backends.made] == ["fake-model"]
    assert len(backends.made[0].requests) == 2
    assert _header(window) == shown


def test_settings_saved_while_the_first_question_waits_for_the_notice(windows, backends, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    save_provider(REMOTE)
    backends.add([say("From Gemini.")])
    window = windows.open()
    tab = ask(window, "First question")
    pump(window, lambda: window.consents.current is not None, timeout=5)
    remote = _header(window)
    assert remote[0].startswith("Google Gemini · gemini-2.5-flash")
    save_provider(LOCAL)
    window.settings_changed()
    assert _header(window) == remote                    # the parked question still goes to Gemini
    pump(window, lambda: tab.worker._header_due, timeout=3)
    assert _header(window) == remote
    _accept_notice(window)
    wait_idle(window, tab)
    assert [b.origin[0] for b in backends.made] == ["gemini"]
    assert _header(window) == remote                    # answered: the chat keeps Gemini


def test_a_failed_first_answer_after_a_settings_change_at_the_notice_goes_with_the_new_settings(
        windows, backends, monkeypatch):
    """G17 critique B1: the change made while the question waited for the notice is not lost when the answer
    fails; the header and the next question's route agree, and the new route asks its own notice."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    save_provider(REMOTE)
    refused = make_error(AuthFailed, "auth", ErrorContext(provider="Google Gemini", key_source="GEMINI_API_KEY"))
    backends.add([refused], [say("From Anthropic.")])
    window = windows.open()
    tab = ask(window, "First question")
    pump(window, lambda: window.consents.current is not None, timeout=5)
    remote = _header(window)
    save_provider(OPUS)
    window.settings_changed()
    assert _header(window) == remote
    pump(window, lambda: tab.worker._header_due, timeout=3)
    _accept_notice(window)
    wait_idle(window, tab)
    assert [b.origin[0] for b in backends.made] == ["gemini"]
    model_line, route_line = _header(window)
    assert model_line.startswith("Anthropic · claude-opus-5-5")     # unanswered: the next question takes these
    assert "Anthropic" in route_line
    ask(window, "Second question", tab)
    pump(window, lambda: window.consents.current is not None, timeout=5)
    assert window.consents.current[1].provider_label == "Anthropic"
    _accept_notice(window)
    wait_idle(window, tab)
    assert [b.origin[0] for b in backends.made] == ["gemini", "anthropic"]
    assert _header(window)[0].startswith("Anthropic · claude-opus-5-5")


def test_an_unanswered_chat_shows_the_settings_it_will_use(windows, backends):
    """A guard of G17-D1 (passes before the fix too): a chat without an answer shows the settings it goes with."""
    save_provider(LOCAL)
    refused = make_error(AuthFailed, "auth", ErrorContext(provider="Custom endpoint", key_source="the key file"))
    backends.add([refused], [say("Other.")])
    window = windows.open(probe=lambda *_a, **_k: ModelFacts(context_window=131_072, source="test"))
    tab = ask(window, "First question")
    wait_idle(window, tab)
    save_provider(ProviderSettings(preset="openai_compatible", base_url="http://127.0.0.2:9/v1", model="gpt-5",
                                   tier="standard"))
    window.settings_changed()
    pump(window, lambda: _header(window)[0] == "Custom endpoint · gpt-5 · standard", timeout=3)
    assert _header(window)[1].startswith("Endpoint on this computer (127.0.0.2:9)")
    ask(window, "Second question", tab)
    wait_idle(window, tab)
    assert backends.made[-1].origin[2] == "gpt-5"
    assert _header(window)[0] == "Custom endpoint · gpt-5 · standard"


def test_a_bound_chat_whose_provider_is_unusable_never_shows_the_settings(windows, backends):
    """G17 critique m2: the header says the chat's route can't be shown; it never shows the settings' route."""
    (conversation_id,) = _saved_chats(windows, backends, 1)
    _set_origin(windows.project, conversation_id, ["gone", "http://127.0.0.1:9/v1", "fake-model"])
    window = windows.open(restore=False)
    window.reopen(conversation_id)
    pump(window, lambda: window.tab_for(conversation_id) is not None
         and window.tab_for(conversation_id).has_messages, timeout=5)
    assert _header(window) == ("gone · fake-model", strings.ROUTE_NOT_SHOWN)


def test_a_reopened_chat_on_another_provider_names_it_in_the_footer(windows, backends):
    """G17 critique m4: the worker's route changes the selected chat's provider, so the footer follows."""
    (conversation_id,) = _saved_chats(windows, backends, 1)
    _set_origin(windows.project, conversation_id, ["anthropic", "https://api.anthropic.com", "claude-opus-5-5"])
    window = windows.open(restore=False)
    window.reopen(conversation_id)
    pump(window, lambda: window.tab_for(conversation_id) is not None
         and window.tab_for(conversation_id).has_messages, timeout=5)
    assert _header(window)[0].startswith("Anthropic · claude-opus-5-5")
    pump(window, lambda: str(window.footer_label.cget("text")) == "Anthropic this month: nothing used yet",
         timeout=3)


# --- a chat that opens (C7-m1) --------------------------------------------------------------------------------------
def test_send_while_a_reopened_chat_opens_keeps_the_question(windows, backends, monkeypatch):
    (conversation_id,) = _saved_chats(windows, backends, 1)
    gate = _gate_open(monkeypatch)
    made = len(backends.made)
    window = windows.open(restore=False)
    window.reopen(conversation_id)
    tab = window.tab_for(conversation_id)
    assert tab is not None
    tab.input.insert("1.0", "second question")
    tab.send()
    assert tab.input.get("1.0", "end-1c") == "second question"
    assert tab.status_text == strings.SEND_WHEN_OPEN
    assert _header(window)[1] == strings.OPENING_CHAT
    assert "disabled" in tab.send_button.state()
    assert len(backends.made) == made
    gate.set()
    pump(window, lambda: not tab.busy and tab.has_messages, timeout=5)
    assert "disabled" not in tab.send_button.state()
    assert _header(window)[0] == LOCAL_MODEL_LINE
    backends.add([say("Answer two.")])
    tab.send()
    wait_idle(window, tab)
    contents = tab.transcript.contents()
    assert contents.count("second question") == 1
    assert contents.index("Saved 1.") < contents.index("second question") < contents.index("Answer two.")


# --- closing (C7-m2, critique m3) -------------------------------------------------------------------------------------
def test_closing_a_chat_before_its_worker_takes_the_question_sends_nothing(windows, backends):
    save_provider(LOCAL)
    backends.add([say("Never.")])
    window = windows.open()
    made = _deferred_workers(window)
    tab = ask(window, "regretted question")
    window.close_chat(tab, ask=False)
    (worker,) = made
    worker.start_now()
    assert worker.exited.wait(5)
    assert backends.made == []
    assert open_store().list_conversations(windows.project) == []


def test_a_stop_before_the_worker_takes_the_question_still_numbers_the_chat(windows, backends):
    """G17 critique m3, decided in G17-D2 (passes before the fix too): a Stop is not a Close — the worker still
    gives the chat its number, the question is "Not sent." and comes back to the box."""
    save_provider(LOCAL)
    window = windows.open()
    made = _deferred_workers(window)
    tab = ask(window, "stopped question")
    tab.stop()
    made[0].start_now()
    wait_idle(window, tab)
    assert tab.number == 1 and tab.tab_text() == "1"
    assert "Not sent." in tab.transcript.contents()
    assert tab.input.get("1.0", "end-1c") == "stopped question"
    assert backends.made == []


# --- robustness (C7-m3, m11, critique M1, M2) -------------------------------------------------------------------------
def test_a_handler_error_does_not_stop_the_poll(windows, backends, monkeypatch):
    save_provider(LOCAL, parallel=2)
    backends.add([say("one")], [say("two")])
    original = ChatTab._finished
    raised = []

    def finished_once(self, event) -> None:
        if not raised:
            raised.append(True)
            raise ValueError("a handler fault")
        original(self, event)

    monkeypatch.setattr(ChatTab, "_finished", finished_once)
    window = windows.open()
    first = ask(window, "first")
    wait_idle(window, first)
    window.go_to_new_chat()
    second = ask(window, "second")
    wait_idle(window, second)
    assert raised == [True]
    assert window._poll_id is not None
    assert "two" in second.transcript.contents()


def test_a_restore_that_faults_still_resolves(windows, backends, monkeypatch):
    ok_id, faulty_id, elsewhere_id = _saved_chats(windows, backends, 3)
    original = Store.open

    def open_(self, project_path, conversation_id=None, **kwargs):
        if conversation_id == faulty_id:
            raise PermissionError("denied")
        if conversation_id == elsewhere_id:
            raise ConversationLocked("other_process")
        return original(self, project_path, conversation_id, **kwargs)

    monkeypatch.setattr(Store, "open", open_)
    window = windows.open()
    pump(window, lambda: strings.NOT_RESTORED_ONE in window.new_tab.transcript.contents(), timeout=5)
    assert window._restoring == set()
    assert [t.conversation_id for t in window.tabs] == [ok_id]
    assert str(window.note_label.cget("text")) == internal_fault_message("PermissionError")
    pump(window, lambda: not window.tabs[0].busy, timeout=5)
    window.close_chat(window.tabs[0], ask=False)
    store = open_store()
    pump(window, lambda: store.open_chats(windows.project)["ids"] == [], timeout=5)


def test_closing_a_chat_while_it_is_restored_resolves_the_restore(windows, backends, monkeypatch):
    (conversation_id,) = _saved_chats(windows, backends, 1)
    gate = _gate_open(monkeypatch)
    window = windows.open()
    pump(window, lambda: window.tab_for(conversation_id) is not None, timeout=5)
    tab = window.tab_for(conversation_id)
    worker = tab.worker
    window.close_chat(tab, ask=False)
    gate.set()
    assert worker.exited.wait(5)
    assert window._restoring == set()
    store = open_store()
    pump(window, lambda: store.open_chats(windows.project)["ids"] == [], timeout=5)


def test_a_history_that_does_not_render_still_opens_the_chat(windows, backends, monkeypatch):
    """G17 critique M2: the tab is idle with Send back and the restore resolves, whatever the rendering does."""
    (conversation_id,) = _saved_chats(windows, backends, 1)
    original = Transcript.render_history
    raised = []

    def render_once(self, messages, provider) -> None:
        if not raised:
            raised.append(True)
            raise ValueError("an odd stored message")
        original(self, messages, provider)

    monkeypatch.setattr(Transcript, "render_history", render_once)
    window = windows.open()
    pump(window, lambda: window.tab_for(conversation_id) is not None and not window._restoring, timeout=5)
    tab = window.tab_for(conversation_id)
    assert raised == [True]
    assert not tab.busy and "disabled" not in tab.send_button.state()
    assert _header(window)[1] != strings.OPENING_CHAT
    backends.add([say("Still answers.")])
    ask(window, "next question", tab)
    wait_idle(window, tab)
    assert "Still answers." in tab.transcript.contents()


# --- a stopped job's lines (C7-n1, n2, G17-D4) ------------------------------------------------------------------------
def test_a_stopped_tool_call_line_resolves(windows, backends, tools):
    save_provider(LOCAL)
    backends.add([tool_call("slow"), say("after the tool")])
    window = windows.open()
    tab = ask(window, "use the tool")
    assert tools.slow_running.wait(5)
    pump(window, lambda: "▸ slow" in tab.transcript.contents(), timeout=5)
    tab.stop()
    tools.release.set()
    wait_idle(window, tab)
    (line,) = [ln for ln in tab.transcript.contents().splitlines() if ln.startswith("▸ slow")]
    assert " → " in line and not line.endswith("…")


def test_stop_after_the_question_started_does_not_mark_it_not_sent(windows, backends, monkeypatch):
    from flextool.assistant.session import Session

    save_provider(LOCAL)
    backends.add([say("never")])
    started, go = threading.Event(), threading.Event()
    original = Session.run_turn

    def run_turn(self, text, on_event, cancel):
        started.set()
        assert go.wait(10)
        return original(self, text, on_event, cancel)

    monkeypatch.setattr(Session, "run_turn", run_turn)
    window = windows.open()
    tab = ask(window, "started question")
    assert started.wait(5)                              # TurnStarted is queued, not yet drained
    tab.stop()
    go.set()
    wait_idle(window, tab)
    assert tab.transcript.text.tag_ranges("user_pending") == ()
    assert strings.NOT_SENT not in tab.transcript.contents()
    assert tab.input.get("1.0", "end-1c") == ""
    session = tab.worker.session
    asked = [m.text for m in session.conversation.messages if m.role == "user"]
    assert len(asked) == 1 and asked[0].endswith("\n\nstarted question")


def test_a_question_the_worker_faulted_on_before_its_turn_is_not_sent(windows, backends, monkeypatch):
    """G17-D4: at the worker's Idle, a question it never started is "Not sent." and back in the box."""
    def setup_fails(self, job):
        raise RuntimeError("a fault of FlexTool before the turn")

    monkeypatch.setattr(AssistantWorker, "_setup", setup_fails)
    save_provider(LOCAL)
    window = windows.open()
    tab = ask(window, "faulted question")
    wait_idle(window, tab)
    assert tab.transcript.text.tag_ranges("user_pending") == ()
    assert "faulted question  Not sent." in tab.transcript.contents()
    assert tab.input.get("1.0", "end-1c") == "faulted question"
    assert backends.made == []


# --- the footer (C7-n3, critique m4, m5) ------------------------------------------------------------------------------
def test_the_footer_names_the_selected_chats_provider(windows, backends, monkeypatch):
    first_id, second_id = _saved_chats(windows, backends, 2)
    _set_origin(windows.project, second_id, ["anthropic", "https://api.anthropic.com", "claude-opus-5-5"])
    window = windows.open()
    pump(window, lambda: len(window.tabs) == 2 and not window._restoring
         and all(t.has_messages and not t.busy for t in window.tabs), timeout=5)
    first, second = window.tab_for(first_id), window.tab_for(second_id)
    assert (first.provider, second.provider) == ("Custom endpoint", "Anthropic")
    reads = []
    real = chat_window_module.run_in_thread

    def collect(fn, name="flextool-assistant-store"):
        if name == "flextool-assistant-ledger":
            reads.append(fn)
            return None
        return real(fn, name)

    monkeypatch.setattr(chat_window_module, "run_in_thread", collect)
    window.select(first)
    pump(window, lambda: len(reads) == 1, timeout=3)
    window.select(second)
    pump(window, lambda: len(reads) == 2, timeout=3)
    reads[1]()                                          # the second chat's read answers first
    reads[0]()
    pump(window, lambda: window._calls.empty(), timeout=3)
    assert str(window.footer_label.cget("text")) == "Anthropic this month: nothing used yet"
