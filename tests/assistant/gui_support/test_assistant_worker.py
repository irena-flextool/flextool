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
from flextool.assistant.llm_errors import message_for
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
    RouteShown,
    SettingsChanged,
    SetupProblem,
    Shutdown,
    TurnFinished,
    TurnRefused,
    TurnStarted,
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
    assert kinds.index("RouteShown") < kinds.index("ConsentNeeded")
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
    # The session ends the turn on the fault itself (review G11, C5-3): a finished turn with the fault's message.
    finished = rig.wait_for(TurnFinished)
    assert not finished.result.ok and message_for(finished.result.error, "gui") == FAULT
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


def test_settings_changed_applies_to_the_next_question_but_keeps_the_chats_model(project, tools, rigs):
    _settings(LOCAL)
    rig = Rig(project, [FakeBackend([say("One."), say("Two."), say("Three.")]), FakeBackend([say("Four.")])])
    rigs.append(rig)
    rig.job()
    rig.wait_for(Idle)
    rig.worker.submit(SettingsChanged())
    rig.job()
    rig.wait_for(Idle, count=2)
    assert len(rig.made) == 1                     # nothing the settings give changed: no new connection
    # Another model in the settings: the chat a model answered in goes on with its own (a new chat takes it).
    other = ProviderSettings(preset="openai_compatible", base_url="http://127.0.0.1:9/v1", model="other-model")
    _settings(other)
    rig.worker.submit(SettingsChanged())
    rig.job()
    rig.wait_for(Idle, count=3)
    assert [b.origin[2] for b in rig.made] == ["fake-model"]
    # The other settings still apply at the next question (here the parallel requests: a new connection).
    _settings(other, parallel=2)
    rig.worker.submit(SettingsChanged())
    rig.job()
    rig.wait_for(Idle, count=4)
    assert [b.origin[2] for b in rig.made] == ["fake-model", "fake-model"]
    assert rig.made[0].closed
    finished = [ev.result.text for _jid, ev in rig.events if isinstance(ev, TurnFinished)]
    assert finished == ["One.", "Two.", "Three.", "Four."]
    prepared = _shown(rig)
    assert {p.model for p in prepared} == {"fake-model"}
    for req in rig.requests():
        history_check.validate(req)


def test_shutdown_while_idle_ends_at_once(project, tools, rigs):
    rig = Rig(project, [])
    rigs.append(rig)
    rig.worker.submit(Shutdown())
    assert rig.worker.exited.wait(3)
    rig.wait_for(Exited)


# --- the model of a chat (owner GUI feedback 2026-11-08 (2)) -------------------------------------------------------
OPUS = ProviderSettings(preset="anthropic", model="claude-opus-5-5")
FLASH_LITE = ProviderSettings(preset="gemini", model="gemini-3.5-flash-lite")
OPUS_FACTS = ModelFacts(context_window=1_000_000, source="anthropic")


def _shown(rig: Rig) -> list:
    """The routes the worker showed for the header (``RouteShown`` with a route; ``None`` is "the settings")."""
    return [ev.prepared for _jid, ev in rig.events if isinstance(ev, RouteShown) and ev.prepared is not None]


def _answer_notices(rig: Rig, job) -> None:
    """Say yes to every data notice of *job* until it finishes."""
    while True:
        rig.read()
        kinds = [type(ev) for _jid, ev in rig.events]
        asked = sum(1 for k in kinds if k is ConsentNeeded)
        if asked > getattr(rig, "_answered", 0):
            rig._answered = asked
            rig.worker.submit(ConsentAnswer(job.job_id, True))
        if any(isinstance(ev, TurnFinished) and jid == job.job_id for jid, ev in rig.events):
            return
        time.sleep(0.01)


def _two_model_rig(project: Path, backends: list[FakeBackend]) -> Rig:
    def probe(_preset, conn, cancel, **_kwargs) -> ModelFacts:
        return OPUS_FACTS if conn.model == "claude-opus-5-5" else ModelFacts(source="none")

    return Rig(project, backends, probe=probe)


def test_a_chat_keeps_its_model_and_level_after_the_settings_change(project, tools, rigs, monkeypatch):
    """The owner's case: a chat with Opus went on with the model saved later (a basic one) at its next question."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _settings(OPUS)
    rig = _two_model_rig(project, [FakeBackend([say("One.")]), FakeBackend([say("never")])])
    rigs.append(rig)
    _answer_notices(rig, rig.job())
    rig.wait_for(Idle)
    _settings(FLASH_LITE)
    rig.worker.submit(SettingsChanged())
    rig.made[0].script.append(say("Two."))
    _answer_notices(rig, rig.job())
    rig.wait_for(Idle, count=2)
    assert [b.origin[2] for b in rig.made] == ["claude-opus-5-5"]
    assert len(rig.made[0].requests) == 2
    prepared = _shown(rig)
    assert {(p.model, p.level) for p in prepared} == {("claude-opus-5-5", "advanced")}
    assert rig.worker.session is not None and rig.worker.session.tier == "large"
    data = json.loads(_conversation_file(project, rig.worker.conversation_id).read_text(encoding="utf-8"))
    assert data["origin"][2] == "claude-opus-5-5"
    assert "tier" not in data and data.get("level") is None
    # A new chat takes the settings of now.
    rig2 = _two_model_rig(project, [FakeBackend([say("Lite.")])])
    rigs.append(rig2)
    _answer_notices(rig2, rig2.job())
    assert [b.origin[2] for b in rig2.made] == ["gemini-3.5-flash-lite"]
    assert [p.level for p in _shown(rig2)][-1] == "basic"


def test_a_reopened_chat_goes_on_with_its_own_model(project, tools, rigs, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _settings(OPUS)
    rig = _two_model_rig(project, [FakeBackend([say("One.")])])
    rigs.append(rig)
    _answer_notices(rig, rig.job())
    rig.wait_for(Idle)
    conversation_id = rig.worker.conversation_id
    rig.close()
    rigs.remove(rig)
    _settings(FLASH_LITE)
    again = _two_model_rig(project, [FakeBackend([say("Two.")])])
    rigs.append(again)
    again.worker.submit(OpenConversation(conversation_id))
    again.wait_for(ConversationLoaded)
    shown = again.wait_for(RouteShown).prepared         # the tab shows the chat's own model before a question
    assert (shown.model, shown.level, shown.provider) == ("claude-opus-5-5", "advanced", "Anthropic")
    _answer_notices(again, again.job())
    assert [b.origin[2] for b in again.made] == ["claude-opus-5-5"]
    assert again.worker.session is not None and again.worker.session.tier == "large"


def test_an_old_conversation_with_a_stored_low_level_is_not_held_at_it(project, tools, rigs, monkeypatch):
    """Files written before 2026-11-08 (2) keep the level resolved at their last question as "tier"; it was never
    the user's choice, so it reads as automatic."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _settings(OPUS)
    rig = _two_model_rig(project, [FakeBackend([say("One.")])])
    rigs.append(rig)
    _answer_notices(rig, rig.job())
    rig.wait_for(Idle)
    conversation_id = rig.worker.conversation_id
    rig.close()
    rigs.remove(rig)
    path = _conversation_file(project, conversation_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("level", None)
    data["tier"] = "small"
    path.write_text(json.dumps(data), encoding="utf-8")
    again = _two_model_rig(project, [FakeBackend([say("Two.")])])
    rigs.append(again)
    again.worker.submit(OpenConversation(conversation_id))
    again.wait_for(ConversationLoaded)
    _answer_notices(again, again.job())
    assert again.worker.session is not None and again.worker.session.tier == "large"
    assert "tier" not in json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("facts", [ModelFacts(source="none"), OPUS_FACTS], ids=["unknown context", "1M context"])
def test_an_advanced_model_is_advanced_before_and_after_the_probe_in_the_window(project, tools, rigs, monkeypatch,
                                                                                facts):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _settings(OPUS)
    rig = Rig(project, [FakeBackend([say("One.")])], probe=lambda *_a, **_k: facts)
    rigs.append(rig)
    _answer_notices(rig, rig.job())
    levels = [p.level for p in _shown(rig)]
    assert levels and set(levels) == {"advanced"}           # the first one is from before the probe
    assert rig.worker.session is not None and rig.worker.session.tier == "large"
    assert any(layer.startswith("Your level: advanced.") for layer in rig.requests()[0].system)


def test_a_paid_chat_does_not_lift_the_free_plans_one_request(project, tools, rigs):
    """Review G18, C7-m4: the free plan's one request at a time is the free chat's own limit.  Before, it was the
    gate's process-wide ``parallel``, which the next prepared paid chat set back to 3, so the waiting free-plan
    request went out while the first one ran."""
    from flextool.assistant.settings import Settings, save_settings
    from flextool.gui.assistant.gate import GateWait, ProviderGate

    gate = ProviderGate()
    save_settings(Settings(provider=LOCAL, parallel_requests=3, free_tier=True))
    block = Block(then=say("First."))
    a1 = Rig(project, [FakeBackend([block])], gate=gate)
    a2 = Rig(project, [FakeBackend([say("Second.")])], gate=gate)
    rigs += [a1, a2]
    a1.job()
    assert block.started.wait(5)
    a2.job()
    a2.wait_for(GateWait)
    assert a2.requests() == []
    paid = ProviderSettings(preset="openai_compatible", base_url="http://127.0.0.1:8/v1", model="fake-model")
    save_settings(Settings(provider=paid, parallel_requests=3, free_tier=False))
    b = Rig(project, [FakeBackend([say("Paid.")])], gate=gate)
    rigs.append(b)
    b.job()
    b.wait_for(TurnFinished)
    # Two more busy notes after the paid chat's prepare (one a second): before the fix the waiting request has
    # gone out by then, whatever the 50 ms poll's timing.
    seen = len([ev for _j, ev in a2.events if isinstance(ev, GateWait)])
    a2.wait_for(GateWait, count=seen + 2, timeout=5)
    assert a2.requests() == []
    block.release.set()
    a2.wait_for(TurnFinished)
    assert len(a2.requests()) == 1


# --- asking a question again (item 220) ---------------------------------------------------------------------------
def _of(rig: Rig, kind: type, job_id: int) -> list:
    rig.read()
    return [ev for jid, ev in rig.events if jid == job_id and isinstance(ev, kind)]


def test_a_question_asked_again_keeps_the_earlier_ones_and_gets_its_number(project, tools, rigs):
    from flextool.gui.assistant.worker import Rewound

    _settings(LOCAL)
    rig = Rig(project, [FakeBackend([say("One."), say("Two."), say("Two again.")])])
    rigs.append(rig)
    rig.job("first")
    rig.wait_for(Idle)
    rig.job("second")
    rig.wait_for(Idle, count=2)
    kept = list(rig.worker.session.messages[:2])
    again = rig.job("second, changed", rewind=2)
    rig.wait_for(Idle, count=3)
    assert [ev.turn for _jid, ev in rig.events if isinstance(ev, TurnStarted)] == [1, 2, 2]
    (rewound,) = _of(rig, Rewound, again.job_id)
    assert list(rewound.messages) == kept               # what the chat keeps, before the question is sent
    assert rewound.turn == 2
    order = [type(ev).__name__ for jid, ev in rig.events if jid == again.job_id]
    assert order.index("Rewound") < order.index("TurnStarted") < order.index("TurnFinished")
    messages = rig.worker.session.messages
    assert messages[:2] == kept and [m.turn for m in messages] == [1, 1, 2, 2]
    assert messages[2].text == "second, changed" and messages[3].text == "Two again."
    third = rig.requests()[2]
    assert [m.text for m in third.messages if m.role == "assistant"] == ["One."]
    assert len(rig.made) == 1
    for req in rig.requests():
        history_check.validate(req)


def test_a_question_that_cannot_be_asked_again_changes_nothing(project, tools, rigs, monkeypatch):
    from flextool.assistant.session import Session
    from flextool.gui.assistant import strings
    from flextool.gui.assistant.worker import Rewound

    _settings(LOCAL)
    rig = Rig(project, [FakeBackend([say("One.")])])
    rigs.append(rig)
    rig.job("first")
    rig.wait_for(Idle)
    before = list(rig.worker.session.messages)
    gone = rig.job("third", rewind=3)
    rig.wait_for(Idle, count=2)
    assert [ev.reason for ev in _of(rig, TurnRefused, gone.job_id)] == [strings.CANNOT_ASK_AGAIN_GONE]
    assert [list(ev.messages) for ev in _of(rig, Rewound, gone.job_id)] == [before]     # the whole history again
    assert [ev.turn for ev in _of(rig, Rewound, gone.job_id)] == [None]                 # nothing deleted
    assert _of(rig, TurnStarted, gone.job_id) == []
    monkeypatch.setattr(Session, "checkpoint_waiting", property(lambda _self: True))
    waiting = rig.job("first again", rewind=1)
    rig.wait_for(Idle, count=3)
    assert [ev.reason for ev in _of(rig, TurnRefused, waiting.job_id)] == [strings.CANNOT_ASK_AGAIN_WAITING]
    assert _of(rig, TurnStarted, waiting.job_id) == []
    assert rig.worker.session.messages == before and len(rig.requests()) == 1
    assert [m["text"] for m in _messages_on_disk(project, rig.worker.conversation_id)] == [m.text for m in before]


def test_asked_again_from_the_first_question_the_chat_keeps_its_own_model(project, tools, rigs):
    """Review decision D7: a question asked again goes to the chat's own model, from question 2 and from question 1
    alike (no answer is left then, but the chat is prepared from what it was held with), even after the settings
    name another model."""
    _settings(LOCAL)
    rig = Rig(project, [FakeBackend([say("One."), say("Two."), say("Two again."), say("One again.")]),
                        FakeBackend([say("Other.")])])
    rigs.append(rig)
    rig.job("first")
    rig.wait_for(Idle)
    rig.job("second")
    rig.wait_for(Idle, count=2)
    other = ProviderSettings(preset="openai_compatible", base_url="http://127.0.0.1:9/v1", model="other-model")
    _settings(other)
    rig.worker.submit(SettingsChanged())
    rig.job("second again", rewind=2)
    rig.wait_for(Idle, count=3)
    first = rig.job("first again", rewind=1)
    rig.wait_for(Idle, count=4)
    assert [b.origin[2] for b in rig.made] == ["fake-model"]
    assert [m.text for m in rig.worker.session.messages if m.role == "assistant"] == ["One again."]
    assert _of(rig, TurnStarted, first.job_id)[0].turn == 1
    data = json.loads(_conversation_file(project, rig.worker.conversation_id).read_text(encoding="utf-8"))
    assert data["origin"][2] == "fake-model"


def test_a_question_asked_again_as_it_was_is_taken_from_the_chat_with_its_summary_once(project, tools, rigs):
    """The window's text of a question asked again as it was is not what is sent: the worker takes the question
    from the chat (session.question_text without the seed), and the session puts the summary the chat started
    with back once — never twice, whatever the window showed."""
    from flextool.assistant import prompt

    seed = "Summary of the earlier chat:\n- question one"
    _settings(LOCAL)
    rig = Rig(project, [FakeBackend([say("One."), say("One again.")])])
    rigs.append(rig)
    rig.job("first", seed=seed)
    rig.wait_for(Idle)
    rig.job(f"{seed}\n\nfirst", rewind=1, seed=seed, again=True)     # as a window showing the summary would send
    rig.wait_for(Idle, count=2)
    sent = [m.text for m in rig.requests()[1].messages if m.role == "user"]
    assert [prompt.remove_context_blocks(t) for t in sent] == [f"{seed}\n\nfirst"]
    stored = [m for m in rig.worker.session.messages if m.role == "user"]
    assert [prompt.remove_context_blocks(m.text) for m in stored] == [f"{seed}\n\nfirst"]
