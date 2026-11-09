"""The agent loop (P1 §15 test 5).

The session runs against :class:`FakeBackend` with fake tools in the registry, so each test scripts exactly what
the model answers.  Every request the backend gets must pass :func:`history_check.validate` (each call answered
once, in order; nothing empty), also after a cut-off answer, a refusal, a stop or an error.  The prefix oracle:
every replayed thinking block was produced by a request whose system, tools and messages are byte-identical to the
start of the request that replays it — across a turn boundary, a mid-turn overflow, a rejected signature and a
resume with changed instructions.
"""
from __future__ import annotations

import dataclasses
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from flextool.assistant import prompt, registry
from flextool.assistant.context import make_context
from flextool.assistant.llm_adapter import (
    ID_LENGTH,
    Message,
    Notice,
    Reply,
    Request,
    TextDelta,
    ToolCall,
    ToolDone,
    ToolRunning,
    ToolWaiting,
    Usage,
)
from flextool.assistant.llm_errors import (
    Cancelled,
    ContextTooLong,
    ErrorContext,
    HistoryRejected,
    ProviderError,
    Refused,
    Truncated,
    make_error,
)
from flextool.assistant.providers import PRESETS
from flextool.assistant.registry import make_tool
from flextool.assistant.session import Session, TurnCaps, TurnWorker, carry_summary
from flextool.assistant.store import open_store
from flextool.assistant.usage import Ledger
from tests.assistant.llm import history_check
from tests.assistant.llm.fake_backend import Block, FakeBackend

USAGE = Usage(100, 20, 0, 10)
PRESET = PRESETS["openai_compatible"]
LIMIT = ("The tool limit for this question is reached. Answer with what you have, and say what is still "
         "unknown.")


# --- fake tools ---------------------------------------------------------------------------------------------------
class Tools:
    """The tools of these tests: ``echo`` (basic), ``deep`` (not basic) and ``hook`` (runs ``on_hook``)."""

    def __init__(self) -> None:
        self.ran: list[tuple[str, dict]] = []
        self.on_hook: Callable[[], None] = lambda: None

    def handler(self, name: str):
        def run(_ctx, args: dict) -> dict:
            self.ran.append((name, args))
            if name == "hook":
                self.on_hook()
            return {"items": [args.get("text", "")], "tool": name}
        return run

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        props = {"text": {"type": "string", "description": "a text"}}
        tools = {name: make_tool(name=name, title=name, description=f"The {name} tool.", properties=props,
                                 handler=self.handler(name), area="orientation", small_tier=name != "deep")
                 for name in ("echo", "deep", "hook")}
        monkeypatch.setattr(registry, "_tools", tools)


@pytest.fixture
def tools(monkeypatch: pytest.MonkeyPatch) -> Tools:
    fake = Tools()
    fake.install(monkeypatch)
    return fake


# --- scripted replies ---------------------------------------------------------------------------------------------
_ids = iter(f"call{n:05d}" for n in range(100_000))


def call(name: str = "echo", text: str = "x", *, raw: str | None = None) -> ToolCall:
    raw_arguments = json.dumps({"text": text}) if raw is None else raw
    try:
        arguments = json.loads(raw_arguments)
    except ValueError:
        arguments = None
    return ToolCall(next(_ids), name, arguments if isinstance(arguments, dict) else None, raw_arguments)


def calls(*tool_calls: ToolCall, text: str = "", stop: str = "tools", extra: dict | None = None) -> Reply:
    return Reply(Message("assistant", text, tool_calls=list(tool_calls), provider_extra=extra or {}), stop, None,
                 USAGE)


def say(text: str, *, extra: dict | None = None, stop: str = "end") -> Reply:
    return Reply(Message("assistant", text, provider_extra=extra or {}), stop, None, USAGE)


@dataclasses.dataclass
class Rig:
    session: Session
    backend: FakeBackend
    store: object
    ledger: Ledger
    ctx: object
    events: list

    def ask(self, text: str = "question", cancel: threading.Event | None = None):
        return self.session.run_turn(text, self.events.append, cancel or threading.Event())

    def validate_all(self) -> None:
        for req in self.backend.requests:
            history_check.validate(req)


@pytest.fixture
def rig(tmp_path: Path, tools: Tools, fake_backends, user_config_in_tmp: Path):
    def make(script: list, *, tier: str = "medium", caps: TurnCaps | None = None, every: int = 6, keep: int = 2,
             conversation=None, backend: FakeBackend | None = None, ctx=None) -> Rig:
        ctx = ctx or make_context(tmp_path / "project", input_source="base.sqlite", front_end="terminal")
        backend = backend or FakeBackend(script)
        fake_backends.append(backend)
        store = open_store()
        conv = conversation or store.new_conversation(ctx, backend.origin, tier)
        ledger = Ledger(user_config_in_tmp)
        session = Session(ctx, backend, PRESET, conversation=conv, caps=caps or TurnCaps(), tier=tier,
                          store=store, ledger=ledger, facts=None, compaction_every=every, keep_recent_turns=keep,
                          sleep=lambda _s: False)
        return Rig(session, backend, store, ledger, ctx, [])
    return make


def roles(messages) -> list[str]:
    return [m.role for m in messages]


# --- rounds and calls ---------------------------------------------------------------------------------------------
def test_multi_round_with_parallel_calls(rig, tools):
    r = rig([calls(call(text="a"), call(text="b")), calls(call(text="c")), say("All done.")])
    result = r.ask()
    assert result.ok and result.text == "All done." and result.rounds == 3
    assert [args["text"] for _, args in tools.ran] == ["a", "b", "c"]
    assert len(r.backend.requests) == 3
    second = r.backend.requests[1].messages
    assert roles(second) == ["user", "assistant", "tool", "tool"]       # both calls of one answer in one round
    r.validate_all()
    done = [e for e in r.events if isinstance(e, ToolDone)]
    assert [d.name for d in done] == ["echo"] * 3 and all(d.ok for d in done)
    assert done[0].arguments == '{"text": "a"}'
    assert done[0].result_preview == second[2].text and done[0].summary == "1 items"
    assert any(isinstance(e, ToolRunning) and e.args_preview == '{"text": "a"}' for e in r.events)
    assert all(req.tool_choice == "auto" for req in r.backend.requests)
    assert [t.name for t in r.backend.requests[0].tools] == ["deep", "echo", "hook"]


def test_the_window_tool_line_gets_redacted_capped_arguments_and_what_was_sent(rig, tools, tmp_path):
    """``ToolDone.arguments``: redacted, at most 2 000 characters; ``result_preview``: the first 4 000 characters
    of the result as appended for the provider (P2 §4.5, §12)."""
    long_text = f"{tmp_path / 'project'}/data.csv " + "x" * 5000
    r = rig([calls(call(text=long_text)), say("Done.")])
    assert r.ask().ok
    [done] = [e for e in r.events if isinstance(e, ToolDone)]
    assert len(done.arguments) == 2000 and done.arguments.endswith("…")
    assert done.arguments.startswith('{"text": "<project>/data.csv xxx')
    assert str(tmp_path) not in done.arguments
    [result] = [m for m in r.session.messages if m.role == "tool"]
    assert len(result.text) > 4000
    assert done.result_preview == result.text[:4000]
    assert r.backend.requests[1].messages[2].text == result.text


def test_round_cap_then_an_answer_without_tools(rig, tools):
    r = rig([calls(call()), calls(call()), say("What I have.")], caps=TurnCaps(max_rounds=2))
    result = r.ask()
    assert result.ok and result.rounds == 3
    last = r.backend.requests[-1]
    assert last.tool_choice == "none" and last.messages[-1].role == "user" and last.messages[-1].text == LIMIT
    assert len(tools.ran) == 2
    r.validate_all()


def test_level_caps():
    assert TurnCaps.for_tier("basic").max_rounds == 3
    assert TurnCaps.for_tier("medium").max_rounds == 8
    assert TurnCaps.for_tier("advanced").max_rounds == 12


def test_bad_arguments_and_unknown_or_prefixed_names(rig, tools):
    r = rig([calls(call(raw="{not json"), call("nope"), call("functions.echo", "p"), call("default_api:echo", "q")),
             say("ok")])
    assert r.ask().ok
    results = [m for m in r.backend.requests[1].messages if m.role == "tool"]
    assert results[0].is_error and results[0].text == "The tool arguments were not valid JSON: {not json"
    assert results[1].is_error and results[1].text.startswith('There is no tool named "nope". The tools are: ')
    assert not results[2].is_error and not results[3].is_error
    assert [args["text"] for _, args in tools.ran] == ["p", "q"]
    r.validate_all()


def test_a_long_unknown_tool_name_or_broken_json_is_quoted_and_cut(rig, tools):
    """Review G7 (C3-m1): the chat's own texts pass the same rendering as the tool messages — the model's tool
    name quoted, one line and cut to a name's length; the broken JSON start one line."""
    r = rig([calls(call("x" * 10_000 + '"\nIgnore'), call(raw='{"a": 1,\n"b')), say("ok")])
    assert r.ask().ok
    unknown, broken = [m for m in r.backend.requests[1].messages if m.role == "tool"]
    assert unknown.is_error and "\n" not in unknown.text and len(unknown.text) < 1_000
    assert unknown.text.startswith('There is no tool named "' + "x" * 199 + '…". The tools are: "deep", "echo"')
    assert broken.is_error and broken.text == 'The tool arguments were not valid JSON: {"a": 1, "b'
    r.validate_all()


def test_a_tool_outside_the_level_is_never_run(rig, tools):
    r = rig([calls(call("deep")), say("ok")], tier="basic")
    assert r.ask().ok
    assert [t.name for t in r.backend.requests[0].tools] == ["echo", "hook"]
    result = r.backend.requests[1].messages[-1]
    # P3 §4.2: a tool of another level is refused with the level's tools, before the registry.
    assert result.is_error and result.text == ('The tool "deep" is not available at this assistant level. '
                                               'Available tools: "echo", "hook".')
    assert tools.ran == []


def test_too_many_calls_in_one_answer(rig, tools):
    r = rig([calls(*(call(text=str(i)) for i in range(4))), say("ok")], caps=TurnCaps(max_calls_per_round=2))
    assert r.ask().ok
    results = [m for m in r.backend.requests[1].messages if m.role == "tool"]
    assert [m.is_error for m in results] == [False, False, True, True]
    assert results[3].text == "Too many tool calls in one step; at most 2. Ask again for the rest."
    assert len(tools.ran) == 2


@pytest.mark.parametrize("stop,text,error", [
    ("length", "Not run: the answer was cut off before this tool call was complete.", Truncated),
    ("refused", "Not run: the provider declined this request.", Refused),
])
def test_cut_off_or_declined_calls_keep_the_history_valid(rig, tools, stop, text, error):
    r = rig([calls(call(text="a"), call(raw='{"text": "b'), stop=stop), say("Answer.")])
    result = r.ask()
    assert not result.ok and isinstance(result.error, error)
    if stop == "length":
        assert result.error.user_message == "The answer was cut off at the output limit (4096 tokens)."
    history = r.session.messages
    assert roles(history) == ["user", "assistant", "tool", "tool"]
    assert [m.text for m in history[2:]] == [text, text] and all(m.is_error for m in history[2:])
    assert tools.ran == []
    assert r.ask("again").ok
    r.validate_all()                                    # the next request is valid for every provider


def test_empty_reply_is_not_appended(rig):
    r = rig([Reply(None, "end", None, USAGE), say("Now an answer.")])
    result = r.ask()
    assert not result.ok and isinstance(result.error, ProviderError)
    assert result.error.user_message == "The model returned an empty answer."
    assert roles(r.session.messages) == ["user"]
    assert r.ask("second").ok
    assert roles(r.backend.requests[1].messages) == ["user"]        # merged with the unanswered question
    assert r.backend.requests[1].messages[0].text.endswith("question\n\nsecond")
    r.validate_all()


def test_a_tool_call_written_as_text_gets_a_notice(rig):
    r = rig([say('{"name": "echo", "arguments": {"text": "a"}}')])
    assert r.ask().ok
    assert Notice("The model wrote a tool call as text; it may not support tool calls well.") in r.events


# --- cancel -------------------------------------------------------------------------------------------------------
def test_cancel_mid_stream_keeps_the_text_without_calls(rig):
    reply = calls(call(), text="A long answer that streams in many small pieces.",
                  extra={"anthropic_blocks": [{"type": "thinking", "thinking": "t", "signature": "s"}]})
    r = rig([reply, say("Next.")])
    cancel = threading.Event()

    def on_event(event):
        r.events.append(event)
        if isinstance(event, TextDelta):
            cancel.set()

    result = r.session.run_turn("question", on_event, cancel)
    assert not result.ok and isinstance(result.error, Cancelled)
    kept = r.session.messages[-1]
    assert kept.role == "assistant" and kept.text == "A long " and not kept.tool_calls and not kept.provider_extra
    assert sum(isinstance(e, TextDelta) for e in r.events) == 1         # nothing shown after the stop
    assert r.ask("again").ok
    r.validate_all()


def test_cancel_before_the_first_byte(rig):
    block = Block(then=say("never"))
    r = rig([block, say("Next.")])
    worker = TurnWorker(r.session)
    finished: list = []
    worker.start("question", r.events.append, finished.append)
    assert block.started.wait(5)
    started = time.monotonic()
    worker.stop()
    assert worker.join(5) and time.monotonic() - started < 1.0
    assert finished == []                                   # a stopped turn's result is discarded
    assert roles(r.session.messages) == ["user"]
    assert r.ask("again").ok
    r.validate_all()


def test_cancel_between_calls(rig, tools):
    cancel = threading.Event()
    tools.on_hook = cancel.set
    r = rig([calls(call("hook"), call("echo"), call("echo")), say("Next.")])
    result = r.ask(cancel=cancel)
    assert isinstance(result.error, Cancelled)
    results = [m for m in r.session.messages if m.role == "tool"]
    assert [m.text for m in results[1:]] == ["Stopped by the user."] * 2
    assert [name for name, _ in tools.ran] == ["hook"]
    assert r.ask("again").ok
    r.validate_all()


def _noted_tool(monkeypatch, on_run: Callable[[], None] = lambda: None) -> None:
    """``noted``: a result that needs the units note (as the result tools add it)."""
    from flextool.assistant.unit_wording import UNITS_CONVENTION, add_session_notes

    def run(ctx, _args: dict) -> dict:
        on_run()
        return add_session_notes({"items": [1]}, [UNITS_CONVENTION], ctx.notes_sent, ctx.notes_in_flight)
    tools = dict(registry.all_tools())
    tools["noted"] = make_tool(name="noted", title="noted", description="A noted tool.",
                               properties={"text": {"type": "string", "description": "a text"}},
                               handler=run, area="orientation", small_tier=True)
    monkeypatch.setattr(registry, "_tools", tools)


def test_a_note_is_sent_only_when_a_request_carried_it(rig, tools, monkeypatch):
    """Review G7 (C3-n5): a note counts as sent only after an answered request carried it in full.  A question
    stopped after its tool ran sent nothing, so the next question's first result carries the full note again
    (answers-only history: the stopped result is not replayed).  In one round only the first result carries it
    (critique m3); after the answer the later results carry the key."""
    from flextool.assistant.unit_wording import UNITS_CONVENTION, UNITS_NOTE

    cancel = threading.Event()
    _noted_tool(monkeypatch, cancel.set)
    r = rig([calls(call("noted")), calls(call("noted"), call("noted")), calls(call("noted")), say("done")])
    assert isinstance(r.ask(cancel=cancel).error, Cancelled)
    assert r.ctx.notes_sent == set()
    cancel.clear()
    _noted_tool(monkeypatch)
    assert r.ask("again").ok
    first, second = [m for m in r.backend.requests[2].messages if m.role == "tool"]
    assert UNITS_NOTE in first.text and UNITS_NOTE not in second.text and f'"{UNITS_CONVENTION}"' in second.text
    later = [m for m in r.backend.requests[3].messages if m.role == "tool"][-1]
    assert UNITS_NOTE not in later.text and f'"{UNITS_CONVENTION}"' in later.text
    assert r.ctx.notes_sent == {UNITS_CONVENTION}
    r.validate_all()


def test_a_compacted_result_keeps_its_session_notes(rig, tools):
    """G7-D3: the short note of a compacted result keeps the full session notes the result held, so a later
    result that names the note by its key never loses it."""
    from flextool.assistant.unit_wording import UNITS_NOTE

    r = rig([say("x")])
    kept = r.session._keep_notes("[removed]", json.dumps({"items": [], "notes": [UNITS_NOTE]}))
    assert kept == f"[removed] {UNITS_NOTE}"
    assert r.session._keep_notes("[removed]", '{"items": []}') == "[removed]"


def test_cancel_while_waiting_for_the_tool_lock(rig, tools):
    r = rig([calls(call(), call()), say("Next.")])
    waiting = threading.Event()

    def on_event(event):
        r.events.append(event)
        if isinstance(event, ToolWaiting):
            waiting.set()

    r.ctx.lock.acquire()
    try:
        worker = TurnWorker(r.session)
        worker.start("question", on_event, lambda _result: None)
        assert waiting.wait(5)
        worker.stop()
        assert worker.join(5)
    finally:
        r.ctx.lock.release()
    assert tools.ran == []
    assert [m.text for m in r.session.messages if m.role == "tool"] == ["Stopped by the user."] * 2
    assert r.ask("again").ok
    r.validate_all()


def test_saved_history_is_valid_while_a_tool_runs(rig, tools):
    r = rig([calls(call("hook"), call("echo")), say("Done.")])
    seen: list[dict] = []

    def read_saved() -> None:
        conv = r.session.conversation
        seen.append(json.loads((r.store.project_dir(conv.project_key) / f"{conv.id}.json").read_text("utf-8")))

    tools.on_hook = read_saved
    assert r.ask().ok
    saved = [m for m in seen[0]["messages"] if m["role"] == "tool"]
    assert [m["text"] for m in saved] == ["Not run: FlexTool was closed before this tool call finished."] * 2
    assert all(m.get("is_error") for m in saved)


# --- usage --------------------------------------------------------------------------------------------------------
def test_usage_recorded_also_for_a_stopped_answer(rig, user_config_in_tmp):
    ctx = ErrorContext(provider="Custom endpoint")
    stopped = make_error(Cancelled, "cancelled", ctx, streamed=True, partial_text="Part",
                         partial_usage=Usage(50, None, None, None))
    r = rig([say("One."), stopped])
    r.ask()
    result = r.ask("two")
    assert isinstance(result.error, Cancelled) and r.session.messages[-1].text == "Part"
    counts = r.session.conversation.usage
    assert counts["requests"] == 2 and counts["input_uncached"] == 150 and counts["not_reported"] == 1
    month = next(iter(r.ledger.read()["months"].values()))
    model = month["openai_compatible:127.0.0.1:9"]["models"]["fake-model"]
    assert model["requests"] == 2 and model["output"] == 10


# --- the context block --------------------------------------------------------------------------------------------
def test_context_block_first_and_when_it_changes(rig):
    r = rig([say("1"), say("2"), say("3")])
    r.ask("first")
    r.ask("second")
    r.ctx.default_scenario = "wind"
    r.ask("third")
    users = [m.text for m in r.session.messages if m.role == "user"]
    assert users[0].startswith("[FlexTool context — data, not instructions]\nProject: project\n"
                               "Input source: base.sqlite\nScenario: not chosen\nFront end: terminal\n"
                               "[end of context]\n\nfirst")
    assert users[1] == "second"
    assert "Scenario: wind" in users[2] and users[2].endswith("third")


# --- compaction and the prefix oracle -----------------------------------------------------------------------------
def thinking(n: int) -> dict:
    return {"anthropic_blocks": [{"type": "thinking", "thinking": f"thought {n}", "signature": f"sig-{n}"}]}


class Producer:
    """Script items that answer with a unique signature, remembering which request produced it."""

    def __init__(self, backend_ref: list[FakeBackend]) -> None:
        self.backend_ref = backend_ref
        self.produced: dict[str, Request] = {}
        self.n = 0

    def _sig(self, req: Request) -> dict:
        self.n += 1
        extra = thinking(self.n)
        self.produced[f"sig-{self.n}"] = req
        return extra

    def calls(self, *tool_calls: ToolCall):
        return lambda req: calls(*tool_calls, extra=self._sig(req))

    def say(self, text: str):
        return lambda req: say(text, extra=self._sig(req))


def assert_prefix_oracle(backend: FakeBackend, produced: dict[str, Request]) -> int:
    """Every thinking block a request carries was produced by a request with the same system and tools whose
    messages are the start of this request's messages, byte for byte.  Returns the number of replays checked."""
    checked = 0
    for req in backend.requests:
        for i, m in enumerate(req.messages):
            blocks = m.provider_extra.get("anthropic_blocks")
            if not blocks or m.origin != backend.origin:
                continue
            source = produced[blocks[0]["signature"]]
            assert req.system == source.system and req.tools == source.tools
            assert req.messages[:len(source.messages)] == source.messages, f"history changed before message {i}"
            assert len(source.messages) == i
            checked += 1
    return checked


def test_boundary_compaction_only_at_boundary_turns(rig, tools):
    holder: list[FakeBackend] = []
    p = Producer(holder)
    script = []
    for turn in range(1, 6):
        script += [p.calls(call(text=f"t{turn}")), p.say(f"answer {turn}")]
    r = rig(script, every=2, keep=1, tier="advanced")           # the full history (P3 §4.7)
    for turn in range(1, 6):
        assert r.ask(f"q{turn}").ok
    reqs = r.backend.requests
    boundary_requests = {4, 8}          # the first requests of turns 3 and 5 (turn - 1 a multiple of 2)
    for k in range(1, len(reqs)):
        prev, cur = reqs[k - 1].messages, reqs[k].messages
        if k in boundary_requests:
            assert cur[:len(prev)] != prev
        else:
            assert cur[:len(prev)] == prev, f"request {k} is not an append to request {k - 1}"
    # At turn 3, the results of turn 1 are notes; those of turn 2 (the last completed turn) are kept.
    turn3 = reqs[4].messages
    tool_msgs = [m for m in turn3 if m.role == "tool"]
    assert tool_msgs[0].compacted and tool_msgs[0].text == (
        '[Earlier result of echo({"text": "t1"}) removed to keep the conversation short. Call the tool again if '
        "you need it.]")
    assert not tool_msgs[1].compacted
    assert all(not m.provider_extra for m in turn3)
    assert assert_prefix_oracle(r.backend, p.produced) > 0
    r.validate_all()


def test_prefix_oracle_across_overflow_and_rejected_history(rig, tools):
    p = Producer([])
    ctx = ErrorContext(provider="Custom endpoint", model="fake-model")
    r = rig([p.calls(call()), p.say("one"),
             p.calls(call()), make_error(ContextTooLong, "context", ctx), p.calls(call()), p.say("two"),
             p.calls(call()), make_error(HistoryRejected, "history", ctx), p.say("three")], tier="advanced")
    assert r.ask("q1").ok
    result = r.ask("q2")
    assert result.ok
    assert Notice("The conversation is too long for fake-model. FlexTool removed older tool results and tried "
                  "again.") in r.events
    after_overflow = r.backend.requests[4].messages
    assert [m.compacted for m in after_overflow if m.role == "tool"] == [True, False]   # this turn's result kept
    assert all(not m.provider_extra for m in after_overflow)
    assert r.ask("q3").ok
    assert Notice("The provider did not accept earlier reasoning data, so FlexTool sent the conversation without "
                  "it.") in r.events
    assert all(not m.provider_extra for m in r.backend.requests[-1].messages)
    assert assert_prefix_oracle(r.backend, p.produced) > 0
    r.validate_all()


def test_still_too_long_after_the_overflow_recovery(rig):
    ctx = ErrorContext(provider="Custom endpoint", model="fake-model")
    r = rig([make_error(ContextTooLong, "context", ctx), make_error(ContextTooLong, "context", ctx)])
    result = r.ask()
    assert isinstance(result.error, ContextTooLong)
    assert result.error.user_message.startswith("The conversation is still too long for fake-model.")


def test_resume_with_changed_instructions_or_model(rig, tools):
    p = Producer([])
    first = rig([p.calls(call()), p.say("one")], tier="advanced")      # the full history (P3 §4.7)
    assert first.ask("q1").ok
    conv = first.session.conversation
    assert conv.system_hash and conv.tools_hash and any(m.provider_extra for m in conv.messages)
    first.store.close(conv)

    # The same model after a FlexTool upgrade changed the instructions: provider data dropped, ids kept.
    reopened = first.store.open(first.ctx.project_path, conv.id)
    reopened.system_hash = "changed"
    ids = [c.id for m in reopened.messages for c in m.tool_calls]
    backend = FakeBackend([p.say("two")])
    second = rig([], conversation=reopened, backend=backend, ctx=first.ctx, tier="advanced")
    assert second.ask("q2").ok
    req = backend.requests[0]
    assert all(not m.provider_extra for m in req.messages)
    assert [c.id for m in req.messages for c in m.tool_calls] == ids
    assert assert_prefix_oracle(backend, p.produced) == 0
    second.validate_all()

    # Another model: ids re-mapped (9 characters, call and result alike) and provider data dropped.
    other = FakeBackend([say("three")], origin=("fake", "http://127.0.0.1:9/v1", "other-model"))
    third = rig([], conversation=second.session.conversation, backend=other, ctx=first.ctx, tier="advanced")
    assert third.ask("q3").ok
    req = other.requests[0]
    call_ids = [c.id for m in req.messages for c in m.tool_calls]
    assert call_ids and all(len(i) == ID_LENGTH and i not in ids for i in call_ids)
    assert [m.tool_call_id for m in req.messages if m.role == "tool"] == call_ids
    third.validate_all()


def test_new_carry_summary():
    messages = [Message("user", prompt.context_block(make_context("/p").snapshot()) + "\n\nWhat is x?"),
                Message("assistant", "", tool_calls=[ToolCall("a", "echo", {}, "{}")]),
                Message("tool", "{}", tool_call_id="a", tool_name="echo"),
                Message("assistant", "X is " + "y" * 700),
                Message("user", "And z?"), Message("assistant", "Z is 1.")]
    summary = carry_summary(messages)
    lines = summary.splitlines()
    assert lines[0] == "Summary of an earlier conversation (text only, no tool results):"
    assert lines[1] == "Question: What is x?"
    assert len(lines[2]) == len("Answer: ") + 600 and lines[2].endswith("…")
    assert lines[3:] == ["Question: And z?", "Answer: Z is 1."]
