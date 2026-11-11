"""Three invariants of a question's turn, whatever ends it (review G11, P1 §6, P3 §4.7).

1. **Well-formed history**: every request the model gets and every saved history passes the history rules
   (:mod:`tests.assistant.llm.history_check`): each call answered once, nothing empty, no two user messages in a
   row, and no session message (note request, fresh part, "tool limit reached") left without its answer or merged
   into a question.
2. **No lost question**: the next question reaches the model in the first request of its turn, and the `/new carry`
   summary lists both questions.
3. **Every exception path ends the turn**: a turn run through :class:`TurnWorker` that is not stopped calls
   ``on_done`` exactly once with a result that has a message for the user; a stopped turn's worker returns.

The matrix: basic (two steps a part, one continuation) and advanced (two steps, no continuation) × a failure
(Stop, a rate limit, a provider error, "too long" twice, an exception that is not a model error) × the request it
hits (the first, a tool step, the note request, the fresh part's first, the tool-less one after the limit).  Then
the cases of one path each: a Stop that a library reports as another error, a save that fails, a conversation
file that a newer FlexTool wrote, and a fault in the middle of a round of calls.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from flextool.assistant import _files, registry
from flextool.assistant.context import make_context
from flextool.assistant.llm_adapter import Message, Notice, Reply, ToolCall, Usage
from flextool.assistant.llm_errors import (
    ContextTooLong,
    ErrorContext,
    ProviderError,
    RateLimited,
    make_error,
    message_for,
)
from flextool.assistant.providers import PRESETS
from flextool.assistant.registry import make_tool
from flextool.assistant.session import SESSION_MESSAGES, Session, TurnCaps, TurnWorker, carry_summary
from flextool.assistant.store import open_store
from flextool.assistant.usage import Ledger
from tests.assistant.llm import history_check
from tests.assistant.llm.fake_backend import Block, FakeBackend

USAGE = Usage(100, 20, 0, 10)
CTX = ErrorContext(provider="Custom endpoint", model="fake-model")
_ids = iter(f"inv{n:05d}" for n in range(100_000))
#: Two steps a part; basic continues once, advanced not at all.
CAPS = {"basic": TurnCaps(max_rounds=2, max_calls_per_round=3, max_continuations=1),
        "advanced": TurnCaps(max_rounds=2, max_calls_per_round=3, max_continuations=0)}
#: The index of each request of question one (basic: part one's two steps, the note request, part two's two
#: steps, the tool-less request; advanced: two steps and the tool-less request).
REQUEST_AT = {"basic": {"first": 0, "tool step": 1, "note request": 2, "fresh part": 3, "tool-less": 5},
              "advanced": {"first": 0, "tool step": 1, "tool-less": 2}}
FAILURES = ("stop", "rate limit", "provider error", "too long twice", "runtime error")


@pytest.fixture(autouse=True)
def tools(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    ran: list[str] = []

    def echo(_ctx, args: dict) -> dict:
        ran.append(args.get("text", ""))
        return {"items": [args.get("text", "")]}

    props = {"text": {"type": "string", "description": "a text"}}
    monkeypatch.setattr(registry, "_tools", {"echo": make_tool(
        name="echo", title="echo", description="The echo tool.", properties=props, handler=echo,
        area="orientation", small_tier=True)})
    return ran


def call(text: str) -> ToolCall:
    return ToolCall(next(_ids), "echo", {"text": text}, json.dumps({"text": text}))


def calls(*texts: str) -> Reply:
    return Reply(Message("assistant", "", tool_calls=[call(t) for t in texts]), "tools", None, USAGE)


def say(text: str) -> Reply:
    return Reply(Message("assistant", text), "end", None, USAGE)


def make(tmp_path: Path, script: list, tier: str) -> tuple[Session, FakeBackend]:
    ctx = make_context(tmp_path / "project", input_source="base.sqlite", front_end="terminal")
    backend = FakeBackend(script)
    store = open_store()
    conv = store.new_conversation(ctx, backend.origin, tier)
    session = Session(ctx, backend, PRESETS["openai_compatible"], conversation=conv, caps=CAPS[tier], tier=tier,
                      store=store, ledger=Ledger(tmp_path / "config"), facts=None, sleep=lambda _s: False)
    return session, backend


def question_one(tier: str) -> list:
    if tier == "basic":
        return [calls("a"), calls("a2"), say("Note one."), calls("b"), calls("b2"), say("answer one")]
    return [calls("a"), calls("a2"), say("answer one")]


def failure_items(failure: str) -> list:
    if failure == "stop":
        return [Block(then=say("never"))]
    if failure == "rate limit":
        return [make_error(RateLimited, "rpm", CTX, retry_after=600.0)]
    if failure == "provider error":
        return [ProviderError("broken", provider="Custom endpoint")]
    if failure == "too long twice":
        return [make_error(ContextTooLong, "context", CTX)] * 2

    def boom(_req):
        raise RuntimeError("boom")
    return [boom]


def run(worker: TurnWorker, text: str, backend: FakeBackend, *, stop: bool) -> list:
    """One question through the worker; the results ``on_done`` got.  *stop*: the user stops the turn while the
    script's :class:`Block` waits."""
    block = next((b for b in backend.script if isinstance(b, Block)), None)
    done: list = []
    finished = threading.Event()

    def on_done(result) -> None:
        done.append(result)
        finished.set()

    worker.start(text, lambda _e: None, on_done)
    if stop:
        assert block is not None and block.started.wait(5)
        worker.stop()
        assert worker.join(5), "a stopped turn's worker did not return"
    else:
        assert finished.wait(5), "on_done was not called"
        assert worker.join(5)
    return done


CASES = [(tier, failure, at) for tier in ("basic", "advanced") for failure in FAILURES for at in REQUEST_AT[tier]]


@pytest.mark.parametrize(("tier", "failure", "at"), CASES)
def test_a_failed_request_leaves_a_valid_history_and_the_next_question(tmp_path, tier, failure, at):
    index = REQUEST_AT[tier][at]
    script = question_one(tier)[:index] + failure_items(failure) + [say("answer two")]
    session, backend = make(tmp_path, script, tier)
    worker = TurnWorker(session)
    try:
        done = run(worker, "question one", backend, stop=failure == "stop")
    finally:
        backend.release_all()
    # 3. The turn ended: a stopped one's result is discarded, any other comes once, with a message.
    if failure == "stop":
        assert done == []
    else:
        assert len(done) == 1 and not done[0].ok and done[0].error is not None
        assert message_for(done[0].error, "terminal").strip()
        if failure == "runtime error":
            assert done[0].error.message_key == "internal"
    # 1. The saved history is valid (FakeBackend checks every request; the autouse fixture its faults).
    history_check.check_stored(session.messages)
    first_of_two = len(backend.requests)
    backend.aborted.clear()
    done = run(worker, "question two", backend, stop=False)
    assert len(done) == 1 and done[0].ok and done[0].text == "answer two"
    history_check.check_stored(session.messages)
    # 2. Question two is in the first request of its turn, and the summary lists both questions.
    assert "question two" in backend.requests[first_of_two].messages[-1].text
    summary = carry_summary(session.messages)
    assert "Question: question one" in summary and "question two" in summary
    assert summary.rstrip().endswith("Answer: answer two")


def test_a_stop_that_a_library_reports_as_another_error_is_stopped(tmp_path):
    """Closing the response can make a library raise any error while it reads: the user stopped, so the result
    is "Stopped.", not an internal fault (critique MINOR 5)."""
    cancel = threading.Event()

    def closed(_req):
        cancel.set()
        raise OSError("the response was closed")

    session, _backend = make(tmp_path, [calls("a"), closed], "basic")
    result = session.run_turn("q", lambda _e: None, cancel)
    assert not result.ok and result.error.code == "cancelled"
    history_check.check_stored(session.messages)


def test_a_fault_in_a_round_of_calls_marks_the_calls_not_run_as_stopped(tmp_path, monkeypatch):
    """A fault after the first of two calls: the second call's result is "Stopped by the user.", not the
    "FlexTool was closed" checkpoint (critique MINOR 6); the turn ends with the internal fault."""
    real = registry.call_tool
    count = {"n": 0}

    def call_tool(ctx, name, args):
        count["n"] += 1
        if count["n"] == 2:
            raise RuntimeError("boom")
        return real(ctx, name, args)

    monkeypatch.setattr(registry, "call_tool", call_tool)
    session, _backend = make(tmp_path, [calls("a", "b"), say("Next.")], "advanced")
    result = session.run_turn("q", lambda _e: None, threading.Event())
    assert not result.ok and result.error.message_key == "internal"
    assert [m.text for m in session.messages if m.role == "tool"] == ['{"items":["a"]}', SESSION_MESSAGES["stopped"]]
    history_check.check_stored(session.messages)
    assert session.run_turn("again", lambda _e: None, threading.Event()).ok


def _failing_writes(monkeypatch: pytest.MonkeyPatch, session: Session, exc: Exception, after: int) -> dict:
    """``_files.write_json_atomic`` fails with *exc* for the conversation's file, from its write number *after* + 1
    on."""
    real = _files.write_json_atomic
    count = {"n": 0}

    def write(path, data, **kwargs):
        if path.name == f"{session.conversation.id}.json":
            count["n"] += 1
            if count["n"] > after:
                raise exc
        return real(path, data, **kwargs)

    monkeypatch.setattr(_files, "write_json_atomic", write)
    return count


def test_a_save_that_fails_after_the_answer_warns_once(tmp_path, monkeypatch):
    """The conversation file cannot be written (a locked file on Windows): the answer still arrives, and one
    notice says it may be missing when the chat is opened again (C5-3)."""
    session, _backend = make(tmp_path, [calls("a"), say("The answer.")], "basic")
    count = _failing_writes(monkeypatch, session, PermissionError(13, "Permission denied"), after=0)
    events: list = []
    result = session.run_turn("q", events.append, threading.Event())
    assert result.ok and result.text == "The answer."
    notices = [e.text for e in events if isinstance(e, Notice)]
    expected = SESSION_MESSAGES["not_saved"].format(reason="[Errno 13] Permission denied")
    assert notices == [expected] and count["n"] >= 2


def test_a_conversation_file_of_a_newer_flextool_makes_the_chat_read_only(tmp_path, monkeypatch):
    """A newer FlexTool wrote the file meanwhile: the answer arrives, the chat is read-only with one notice, and
    the next question is answered without saving and without the notice again."""
    session, backend = make(tmp_path, [calls("a"), say("The answer."), say("Second.")], "basic")
    _failing_writes(monkeypatch, session, _files.NewerFormat(Path("conversation.json"), 99), after=0)
    events: list = []
    result = session.run_turn("q", events.append, threading.Event())
    assert result.ok and result.text == "The answer."
    notices = [e.text for e in events if isinstance(e, Notice)]
    assert notices == [SESSION_MESSAGES["read_only"]]
    assert session.conversation.read_only
    events.clear()
    assert session.run_turn("again", events.append, threading.Event()).ok
    assert not [e for e in events if isinstance(e, Notice)]


def test_a_newer_format_raised_by_a_store_is_said_plainly(tmp_path):
    """A store that raises :class:`NewerFormat` itself: the session makes the chat read-only and says so; the
    answer is not failed."""
    session, _backend = make(tmp_path, [say("The answer.")], "basic")

    class Raising:
        def save(self, conv):
            if conv.read_only:
                return False
            raise _files.NewerFormat(Path("conversation.json"), 99)

    session.store = Raising()
    events: list = []
    result = session.run_turn("q", events.append, threading.Event())
    assert result.ok and [e.text for e in events if isinstance(e, Notice)] == [SESSION_MESSAGES["read_only"]]


def test_a_used_up_script_is_still_a_fault_of_the_test(tmp_path):
    """The session turns the fake backend's "script is used up" into a failed answer like any fault; the fake
    backend records it, so the assistant tests' teardown check fails the test (critique MINOR 7)."""
    from tests.assistant.llm import fake_backend

    session, _backend = make(tmp_path, [], "basic")
    result = session.run_turn("q", lambda _e: None, threading.Event())
    assert not result.ok and result.error.message_key == "internal"
    assert fake_backend.FAULTS == ["the fake backend's script is used up"]
    fake_backend.FAULTS.clear()                 # the one fault this test expects
