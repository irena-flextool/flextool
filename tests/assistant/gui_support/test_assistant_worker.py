"""The worker thread of a chat tab (P2 §13.1 test 4).

Each test runs a real :class:`AssistantWorker` with a scripted backend (:class:`FakeBackend`), a fake probe and
fake tools, and reads its event queue the way the chat window does.  Nothing reaches the network.  Checked: the
data notice comes before any request and a declined one sends nothing; Stop works during the probe, the parked
notice, the first byte and the wait for the tool lock; each question's window snapshot sets the tools' defaults;
the conversation file stays a valid, resumable history when FlexTool closes during a tool call; a delete during a
tool call is not undone by a later save; a fault repairs and saves the history; expiry 0 writes nothing.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from flextool.assistant import _files
from flextool.assistant.llm_adapter import Message, ToolCall, ToolDone, ToolWaiting
from flextool.assistant.probes import ModelFacts
from flextool.assistant.settings import ProviderSettings
from flextool.assistant.store import open_store
from flextool.gui.assistant.worker import (
    ChatReady,
    ConsentAnswer,
    ConsentNeeded,
    ConversationLoaded,
    DeleteConversation,
    Deleted,
    Exited,
    Idle,
    OpenConversation,
    Prepared,
    SettingsChanged,
    SetupProblem,
    Shutdown,
    TurnFinished,
    TurnRefused,
    TurnStarted,
    WorkerFault,
    finishing_worker,
)
from tests.assistant.gui_support.worker_rig import (
    LOCAL,
    REMOTE,
    Rig,
    Tools,
    say,
    tool_call,
)
from tests.assistant.gui_support.worker_rig import conversation_file as _conversation_file
from tests.assistant.gui_support.worker_rig import messages_on_disk as _messages_on_disk
from tests.assistant.gui_support.worker_rig import save_provider as _settings
from tests.assistant.llm import history_check
from tests.assistant.llm.fake_backend import Block, FakeBackend

CLOSED = "Not run: FlexTool was closed before this tool call finished."
STOPPED_CALL = "Stopped by the user."
FAULT = ("The assistant stopped because of an internal fault (RuntimeError). Please report it; the details are in "
         "the FlexTool log.")


@pytest.fixture
def tools(monkeypatch: pytest.MonkeyPatch) -> Tools:
    fake = Tools()
    fake.install(monkeypatch)
    yield fake
    fake.release.set()


@pytest.fixture
def project(tmp_path: Path) -> Path:
    folder = tmp_path / "project"
    folder.mkdir()
    return folder


@pytest.fixture
def rigs():
    made: list[Rig] = []
    yield made
    for rig in made:
        rig.close()


# --- the data notice ----------------------------------------------------------------------------------------------
def test_the_notice_comes_before_any_request_and_a_yes_sends(project, tools, rigs, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _settings(REMOTE)
    rig = Rig(project, [FakeBackend([say("Hello.")])])
    rigs.append(rig)
    job = rig.job()
    notice = rig.wait_for(ConsentNeeded)
    assert notice.provider_label == "Google Gemini"
    assert "generativelanguage.googleapis.com" in notice.notice
    kinds = rig.kinds()
    assert kinds.index("Prepared") < kinds.index("ConsentNeeded")
    assert rig.made == [] and rig.probes == []          # neither the probe nor a request before the answer
    assert rig.worker.busy.is_set()
    rig.worker.submit(ConsentAnswer(job.job_id, True))
    finished = rig.wait_for(TurnFinished)
    assert finished.result.ok and finished.result.text == "Hello."
    rig.wait_for(Idle)
    assert not rig.worker.busy.is_set()
    assert len(rig.requests()) == 1
    # Accepted once: the next chat with the same key is not asked again.
    rig2 = Rig(project, [FakeBackend([say("Again.")])])
    rigs.append(rig2)
    rig2.job()
    rig2.wait_for(TurnFinished)
    assert "ConsentNeeded" not in rig2.kinds()


def test_a_declined_notice_sends_nothing(project, tools, rigs, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _settings(REMOTE)
    rig = Rig(project, [FakeBackend([say("never")])])
    rigs.append(rig)
    job = rig.job()
    rig.wait_for(ConsentNeeded)
    rig.worker.submit(ConsentAnswer(job.job_id, False))
    refused = rig.wait_for(TurnRefused)
    assert refused.reason == "Nothing was sent."
    rig.wait_for(Idle)
    assert rig.made == [] and rig.probes == []


def test_stop_during_a_parked_notice(project, tools, rigs, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _settings(REMOTE)
    rig = Rig(project, [FakeBackend([say("never")])])
    rigs.append(rig)
    job = rig.job()
    rig.wait_for(ConsentNeeded)
    rig.worker.stop_turn(job)
    assert rig.wait_for(TurnRefused).reason == "Stopped."
    rig.wait_for(Idle)
    assert rig.made == [] and rig.probes == []


def test_stop_during_the_probe(project, tools, rigs):
    _settings(LOCAL)
    probing = threading.Event()

    def slow_probe(_preset, _conn, cancel, **_kwargs) -> ModelFacts:
        from flextool.assistant.llm_errors import cancelled

        probing.set()
        if cancel.wait(10):
            raise cancelled()
        return ModelFacts()

    rig = Rig(project, [FakeBackend([say("never")])], probe=slow_probe)
    rigs.append(rig)
    job = rig.job()
    assert probing.wait(3)
    started = time.monotonic()
    rig.worker.stop_turn(job)
    assert rig.wait_for(TurnRefused).reason == "Stopped."
    rig.wait_for(Idle)
    assert time.monotonic() - started < 1.0
    assert rig.made == []


def test_no_provider_is_a_setup_problem(project, tools, rigs):
    rig = Rig(project, [])
    rigs.append(rig)
    rig.job()
    problem = rig.wait_for(SetupProblem)
    assert problem.message == "Choose an AI provider in the settings first."
    assert problem.open_settings
    assert rig.wait_for(TurnRefused).reason == "Not sent."
    rig.wait_for(Idle)


# --- stop, snapshots, history --------------------------------------------------------------------------------------
def test_stop_before_the_first_byte_then_a_valid_next_request(project, tools, rigs):
    _settings(LOCAL)
    block = Block(then=say("late"))
    rig = Rig(project, [FakeBackend([block, say("Second answer.")])])
    rigs.append(rig)
    job = rig.job("first")
    rig.wait_for(TurnStarted)
    assert block.started.wait(3)
    started = time.monotonic()
    rig.worker.stop_turn(job)
    rig.wait_for(Idle)
    assert time.monotonic() - started < 1.0
    finished = rig.wait_for(TurnFinished)
    assert finished.result.error is not None and finished.result.error.code == "cancelled"
    rig.job("second")
    assert rig.wait_for(TurnFinished, count=2).result.text == "Second answer."
    assert len(rig.requests()) == 2
    for req in rig.requests():
        history_check.validate(req)


def test_each_question_uses_its_own_snapshot_as_the_defaults(project, tools, rigs):
    _settings(LOCAL)
    rig = Rig(project, [FakeBackend([tool_call(), say("One."), tool_call(), say("Two.")])])
    rigs.append(rig)
    rig.job("first", scenario="base")
    rig.wait_for(Idle)
    rig.job("second", scenario="wind")
    rig.wait_for(Idle, count=2)
    assert tools.ran == [("echo", "base"), ("echo", "wind")]
    ready = rig.wait_for(ChatReady)
    assert ready.number == 1 and ready.saved


def test_delete_conversation_removes_the_open_chats_file(project, tools, rigs):
    _settings(LOCAL)
    rig = Rig(project, [FakeBackend([say("Saved.")])])
    rigs.append(rig)
    rig.job()
    rig.wait_for(Idle)
    conversation_id = rig.worker.conversation_id
    file = _conversation_file(project, conversation_id)
    assert file.exists()
    rig.worker.submit(DeleteConversation())
    assert rig.wait_for(Deleted).outcome == "deleted"
    assert not file.exists()
    rig.worker.shutdown()
    assert rig.worker.exited.wait(3)
    assert not file.exists()                      # the closing save writes nothing for a deleted chat


def test_shutdown_during_a_tool_call_leaves_a_resumable_checkpoint(project, tools, rigs):
    _settings(LOCAL)
    rig = Rig(project, [FakeBackend([tool_call("slow", say_first="Looking it up."), say("never")])])
    rigs.append(rig)
    rig.job()
    assert tools.slow_running.wait(5)
    conversation_id = rig.worker.conversation_id
    rig.worker.shutdown()
    assert finishing_worker(conversation_id) is rig.worker
    on_disk = _messages_on_disk(project, conversation_id)
    assert on_disk[1]["text"] == "Looking it up."
    assert on_disk[2]["text"] == CLOSED and on_disk[2]["is_error"]
    assert not rig.worker.exited.is_set()
    tools.release.set()
    assert rig.worker.exited.wait(3)
    rig.wait_for(Exited)
    assert finishing_worker(conversation_id) is None
    # Reopened in a new tab: the history is valid and the chat goes on.
    rig2 = Rig(project, [FakeBackend([say("Resumed.")])])
    rigs.append(rig2)
    rig2.worker.submit(OpenConversation(conversation_id))
    loaded = rig2.wait_for(ConversationLoaded)
    assert loaded.number == 1
    rig2.job("go on")
    assert rig2.wait_for(TurnFinished).result.text == "Resumed."
    for req in rig2.requests():
        history_check.validate(req)


def test_a_delete_during_a_tool_call_is_not_undone(project, tools, rigs):
    _settings(LOCAL)
    rig = Rig(project, [FakeBackend([say("First."), tool_call("slow"), say("after")])])
    rigs.append(rig)
    rig.job()
    rig.wait_for(Idle)
    conversation_id = rig.worker.conversation_id
    rig.job("again")
    assert tools.slow_running.wait(5)
    assert open_store().delete_conversation(project, conversation_id) == "deleted"
    tools.release.set()
    rig.wait_for(Idle, count=2)
    rig.worker.shutdown()
    assert rig.worker.exited.wait(3)
    assert not _conversation_file(project, conversation_id).exists()


def test_stop_while_waiting_for_the_tool_lock(project, tools, rigs):
    _settings(LOCAL)
    lock = threading.Lock()
    lock.acquire()
    try:
        rig = Rig(project, [FakeBackend([tool_call("echo"), say("never")])], tool_lock=lock)
        rigs.append(rig)
        job = rig.job()
        rig.wait_for(ToolWaiting)
        started = time.monotonic()
        rig.worker.stop_turn(job)
        rig.wait_for(Idle)
        assert time.monotonic() - started < 1.0
    finally:
        lock.release()
    assert tools.ran == []
    assert not [ev for _jid, ev in rig.events if isinstance(ev, ToolDone)]
    messages = _messages_on_disk(project, rig.worker.conversation_id)
    assert messages[-1]["text"] == STOPPED_CALL


def test_a_fault_repairs_and_saves_the_history(project, tools, rigs):
    _settings(LOCAL)
    store = open_store()
    conv = store.new_conversation(project, ("openai_compatible", "http://127.0.0.1:9/v1", "fake-model"), "medium",
                                  number=7)
    broken_call = ToolCall("callbroken", "echo", {"text": "x"}, json.dumps({"text": "x"}))
    conv.messages = [Message("user", "earlier", turn=1),
                     Message("assistant", "", tool_calls=[broken_call], turn=1)]
    store.save(conv)
    store.close(conv)

    def explode(_req):
        raise RuntimeError("boom")

    rig = Rig(project, [FakeBackend([explode])])
    rigs.append(rig)
    rig.worker.submit(OpenConversation(conv.id))
    assert rig.wait_for(ConversationLoaded).number == 7
    rig.job("now")
    assert rig.wait_for(WorkerFault).message == FAULT
    rig.wait_for(Idle)
    messages = _messages_on_disk(project, conv.id)
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "user"]
    assert messages[2]["text"] == STOPPED_CALL and messages[2]["tool_call_id"] == "callbroken"


def test_expiry_zero_writes_no_conversation(project, tools, rigs):
    _settings(LOCAL, expiry=0)
    rig = Rig(project, [FakeBackend([say("Not kept.")])])
    rigs.append(rig)
    rig.job()
    assert not rig.wait_for(ChatReady).saved
    rig.wait_for(Idle)
    rig.worker.shutdown()
    assert rig.worker.exited.wait(3)
    folder = open_store().project_dir(_files.project_key(project.resolve()))
    assert not folder.exists() or not [p for p in folder.iterdir() if p.suffix == ".json"]


def test_settings_changed_applies_to_the_next_question(project, tools, rigs):
    _settings(LOCAL)
    rig = Rig(project, [FakeBackend([say("One.")]), FakeBackend([say("Two.")])])
    rigs.append(rig)
    rig.job()
    rig.wait_for(Idle)
    rig.worker.submit(SettingsChanged())
    rig.job()
    rig.wait_for(Idle, count=2)
    assert len(rig.made) == 1                     # nothing the settings give changed: no new connection
    _settings(ProviderSettings(preset="openai_compatible", base_url="http://127.0.0.1:9/v1", model="other-model"))
    rig.worker.submit(SettingsChanged())
    rig.backends.append(FakeBackend([say("Three.")]))
    rig.job()
    rig.wait_for(Idle, count=3)
    assert [b.origin[2] for b in rig.made] == ["fake-model", "other-model"]
    assert rig.made[0].closed
    prepared = [ev for _jid, ev in rig.events if isinstance(ev, Prepared)]
    assert [p.model for p in prepared] == ["fake-model", "other-model"]
    for req in rig.requests():
        history_check.validate(req)


def test_shutdown_while_idle_ends_at_once(project, tools, rigs):
    rig = Rig(project, [])
    rigs.append(rig)
    rig.worker.submit(Shutdown())
    assert rig.worker.exited.wait(3)
    rig.wait_for(Exited)
