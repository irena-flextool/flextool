"""Asking a question again, as it was or changed (item 220): ``Session.run_turn(..., rewind=k)``.

Question *k* and every later one are deleted with their answers, tool calls and provider data; the messages kept
are not changed, so the request of the question asked again carries the same history (with its provider data) as
when question *k* was first asked, and nothing of the deleted questions.  The context block is sent again when the
questions kept do not hold the current one; a reopened chat's unanswered question is replaced, never merged into;
the usage totals keep the deleted answers' tokens; the chat is saved shortened at once (also with question 1 gone,
then bound to no model).  Each provider's replayed data stays valid on the wire: Anthropic's thinking blocks,
Gemini's thought signatures, Ollama's reasoning text.
"""
from __future__ import annotations

import json
import threading

import pytest

from flextool.assistant import prompt, registry
from flextool.assistant.context import make_context
from flextool.assistant.llm_adapter import REASONING_TEXT_KEY, Message, Reply, ToolCall, Usage, make_backend
from flextool.assistant.llm_errors import ProviderError
from flextool.assistant.providers import PRESETS
from flextool.assistant.registry import make_tool
from flextool.assistant.session import (
    RewindRefused,
    Session,
    TurnCaps,
    next_turn,
    question_text,
    rewind,
)
from flextool.assistant.store import bound_origin, message_to_json, open_store
from flextool.assistant.usage import Ledger, add_usage
from tests.assistant.llm import history_check
from tests.assistant.llm.fake_backend import FakeBackend
from tests.assistant.llm.wire import Recorder, connection

USAGE = Usage(100, 20, 0, 10)
REDO_USAGE = Usage(7, 0, 0, 3)
_ids = iter(f"again{n:05d}" for n in range(100_000))


@pytest.fixture(autouse=True)
def tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """``echo`` (every level) answers ``{"items": [text]}``."""
    props = {"text": {"type": "string", "description": "a text"}}
    tool = make_tool(name="echo", title="echo", description="The echo tool.", properties=props,
                     handler=lambda _ctx, args: {"items": [args.get("text", "")]}, area="orientation",
                     small_tier=True)
    monkeypatch.setattr(registry, "_tools", {"echo": tool})


def call(text: str, extra: dict | None = None) -> ToolCall:
    return ToolCall(next(_ids), "echo", {"text": text}, json.dumps({"text": text}), provider_extra=extra or {})


def calls(*tool_calls: ToolCall, extra: dict | None = None, usage: Usage = USAGE) -> Reply:
    return Reply(Message("assistant", "", tool_calls=list(tool_calls), provider_extra=extra or {}), "tools", None,
                 usage)


def say(text: str, extra: dict | None = None, usage: Usage = USAGE) -> Reply:
    return Reply(Message("assistant", text, provider_extra=extra or {}), "end", None, usage)


def failure() -> ProviderError:
    return ProviderError("broken", provider="Custom endpoint")


def make(tmp_path, script: list, *, tier: str = "advanced", preset: str = "openai_compatible", origin=None,
         conversation=None, ctx=None, store=None, on_checkpoint=None):
    ctx = ctx or make_context(tmp_path / "project", input_source="base.sqlite", front_end="terminal")
    backend = FakeBackend(script, **({"origin": origin} if origin else {}))
    store = store or open_store()
    conv = conversation or store.new_conversation(ctx, backend.origin, tier)
    session = Session(ctx, backend, PRESETS[preset], conversation=conv, caps=TurnCaps(), tier=tier, store=store,
                      ledger=Ledger(tmp_path / "config"), facts=None, sleep=lambda _s: False,
                      compaction_every=10, on_checkpoint=on_checkpoint)     # no compaction in these few questions
    return session, backend, ctx


def ask(session: Session, text: str, **kw):
    return session.run_turn(text, lambda _e: None, threading.Event(), **kw)


def dumped(messages) -> str:
    return json.dumps([message_to_json(m) for m in messages], sort_keys=True, ensure_ascii=False)


def thinking(n: int) -> dict:
    return {"anthropic_blocks": [{"type": "thinking", "thinking": f"thought {n}", "signature": f"sig-{n}"}]}


def three_questions(tmp_path, *, redo: list, tier: str = "advanced", **kw):
    """Questions 1-3, each with a tool call and an answer that carry provider data, then *redo* for what follows."""
    script = []
    for n in (1, 2, 3):
        script += [calls(call(f"t{n}"), extra=thinking(n)), say(f"MARK answer {n}", extra=thinking(10 + n))]
    session, backend, ctx = make(tmp_path, script + redo, tier=tier, **kw)
    for n in (1, 2, 3):
        assert ask(session, f"q{n}").ok
    return session, backend, ctx


# --- the history kept ---------------------------------------------------------------------------------------------
def test_the_question_asked_again_sends_the_same_history_with_its_provider_data(tmp_path):
    """Question 3 asked again: its first request is byte-identical to its first request the first time (the
    history before it, with every thinking block, unchanged; question 3 is no compaction point here)."""
    session, backend, _ctx = three_questions(tmp_path, redo=[say("MARK redo", extra=thinking(99))])
    first_time = backend.requests[4]                 # question 3's first request
    assert first_time.messages[-1].text == "q3"
    kept = [message_to_json(m) for m in session.messages[:8]]
    assert ask(session, "q3", rewind=3).ok
    again = backend.requests[-1]
    assert again.messages == first_time.messages
    assert session.request_sha256(again) == session.request_sha256(first_time)
    assert [message_to_json(m) for m in session.messages[:8]] == kept
    assert [m.text for m in session.messages[8:]] == ["q3", "MARK redo"]
    assert {m.turn for m in session.messages[8:]} == {3}
    history_check.check_stored(session.messages)


def test_the_deleted_questions_never_reach_the_next_request(tmp_path):
    session, backend, _ctx = three_questions(tmp_path, redo=[say("new answer")])
    dropped = session.messages[4:]
    markers = ([m.text for m in dropped if m.role != "tool" and m.text] + [c.id for m in dropped for c in m.tool_calls]
               + ["sig-2", "sig-3", "sig-12", "sig-13", "thought 2"])
    assert ask(session, "a changed question", rewind=2).ok
    sent = dumped(backend.requests[-1].messages)
    for marker in markers:
        assert marker not in sent, marker
    assert "sig-1" in sent and "sig-11" in sent          # question 1's data is replayed
    assert [m.text for m in session.messages[4:]] == ["a changed question", "new answer"]
    assert next_turn(session.messages) == 3


def test_the_context_block_is_sent_again_when_the_questions_kept_lack_it(tmp_path):
    session, backend, ctx = make(tmp_path, [say("one"), say("two"), say("two again"), say("two once more")])
    block_a = prompt.context_block(ctx.snapshot(), ctx.redact)
    assert ask(session, "q1").ok
    ctx.default_scenario = "other"
    block_b = prompt.context_block(ctx.snapshot(), ctx.redact)
    assert ask(session, "q2").ok
    assert session.messages[2].text == f"{block_b}\n\nq2"
    # Question 2 held block B: asked again with B in force, it starts with B again.
    assert ask(session, "q2 changed", rewind=2).ok
    assert backend.requests[-1].messages[-1].text == f"{block_b}\n\nq2 changed"
    # Back to the first choices, the block of question 1 is in force: no block.
    ctx.default_scenario = None
    assert prompt.context_block(ctx.snapshot(), ctx.redact) == block_a
    assert ask(session, "q2 plain", rewind=2).ok
    assert backend.requests[-1].messages[-1].text == "q2 plain"


def test_a_reopened_unanswered_question_is_replaced_not_merged(tmp_path):
    session, _backend, ctx = make(tmp_path, [say("one"), failure()])
    assert ask(session, "q1").ok
    assert not ask(session, "q2 that failed").ok
    conv = session.conversation
    assert [m.role for m in conv.messages] == ["user", "assistant", "user"]
    store = session.store
    store.close(conv)
    reopened = store.open(ctx.project_path, conv.id)
    assert next_turn(reopened.messages) == 2 and next_turn(reopened.messages, rewind=2) == 2
    assert question_text(reopened.messages, 2) == "q2 that failed"
    again, backend, _ = make(tmp_path, [say("two")], conversation=reopened, ctx=ctx, store=store)
    assert ask(again, "q2 changed", rewind=2).ok
    assert backend.requests[0].messages[-1].text == "q2 changed"
    assert [m.text for m in again.messages] == [conv.messages[0].text, "one", "q2 changed", "two"]


def test_a_merged_unanswered_question_is_one_question_with_both_texts(tmp_path):
    session, _backend, _ctx = make(tmp_path, [say("one"), failure(), say("two")])
    assert ask(session, "q1").ok
    assert not ask(session, "first try").ok
    assert ask(session, "second try").ok
    assert question_text(session.messages, 2) == "first try\n\nsecond try"


# --- refusals, usage, saving --------------------------------------------------------------------------------------
@pytest.mark.parametrize("turn", [0, -1, 2, 7])
def test_a_question_the_chat_lacks_is_refused_and_nothing_changes(tmp_path, turn):
    session, backend, _ctx = make(tmp_path, [say("one")])
    assert ask(session, "q1").ok
    before = dumped(session.messages)
    with pytest.raises(RewindRefused) as refused:
        ask(session, "again", rewind=turn)
    assert refused.value.reason == "unknown"
    assert dumped(session.messages) == before and len(backend.requests) == 1
    with pytest.raises(RewindRefused):
        rewind(list(session.messages), turn)


def test_the_terminals_turn_worker_says_a_refusal_in_plain_words(tmp_path):
    """``TurnWorker`` (the terminal's) turns :class:`RewindRefused` into a notice and a result that is neither ok
    nor an error (nothing was asked), never an internal fault."""
    from flextool.assistant.llm_adapter import Notice
    from flextool.assistant.session import TurnWorker

    session, backend, _ctx = make(tmp_path, [say("one")])
    assert ask(session, "q1").ok
    before = dumped(session.messages)
    events: list = []
    done: list = []
    worker = TurnWorker(session)
    worker.start("again", events.append, done.append, rewind=5)
    assert worker.join(5)
    (result,) = done
    assert (result.ok, result.error, result.rounds) == (False, None, 0)
    assert events == [Notice("This chat has no question 5.")]
    assert dumped(session.messages) == before and len(backend.requests) == 1


def test_a_waiting_checkpoint_refuses_asking_again(tmp_path):
    snaps: list = []
    session, _backend, ctx = make(tmp_path, [say("one")], on_checkpoint=snaps.append)
    assert ask(session, "q1").ok
    restored, backend, _ = make(tmp_path, [], ctx=ctx)
    restored.restore(json.loads(json.dumps(snaps[-1])))
    before = dumped(restored.messages)
    with pytest.raises(RewindRefused) as refused:
        ask(restored, "again", rewind=1)
    assert refused.value.reason == "checkpoint"
    assert dumped(restored.messages) == before and backend.requests == []


def test_the_usage_totals_keep_the_deleted_answers(tmp_path):
    session, _backend, _ctx = three_questions(tmp_path, redo=[say("redo", usage=REDO_USAGE)])
    expected = dict(session.conversation.usage)
    add_usage(expected, REDO_USAGE)
    assert ask(session, "q2", rewind=2).ok
    assert session.conversation.usage == expected


def test_the_chat_is_saved_shortened_at_once_and_reopens_so(tmp_path):
    seen: list[list[str]] = []

    def look(_req):
        conv = session.conversation
        file = session.store._paths(conv.project_key, conv.id)[0]
        seen.append([m["text"] for m in json.loads(file.read_text(encoding="utf-8"))["messages"]])
        return say("redo")

    session, _backend, ctx = three_questions(tmp_path, redo=[look])
    assert ask(session, "q2 changed", rewind=2).ok
    # Saved before the question is sent: question 1 only (the new question is saved with its answer).
    assert seen[0] == [m.text for m in session.messages[:4]] and seen[0][-1] == "MARK answer 1"
    conv = session.conversation
    session.store.close(conv)
    reopened = session.store.open(ctx.project_path, conv.id)
    assert [m.text for m in reopened.messages][3:] == ["MARK answer 1", "q2 changed", "redo"]


def test_asking_question_1_again_leaves_a_chat_bound_to_no_model(tmp_path):
    session, _backend, ctx = make(tmp_path, [say("one"), say("two"), failure()])
    assert ask(session, "q1").ok and ask(session, "q2").ok
    assert bound_origin(session.conversation) is not None
    assert not ask(session, "q1 changed", rewind=1).ok
    conv = session.conversation
    block = prompt.context_block(ctx.snapshot(), ctx.redact)
    assert [m.text for m in conv.messages] == [f"{block}\n\nq1 changed"]
    session.store.close(conv)
    reopened = session.store.open(ctx.project_path, conv.id)
    assert [m.text for m in reopened.messages] == [f"{block}\n\nq1 changed"]
    assert bound_origin(reopened) is None


def test_the_summary_a_new_chat_started_with_is_kept_hidden(tmp_path):
    seed = "Summary of an earlier conversation:\nQuestion: a\n\nAnswer: b"
    session, backend, ctx = make(tmp_path, [say("one"), say("one again"), say("two"), say("two again")])
    assert ask(session, "q1", seed=seed).ok
    block = prompt.context_block(ctx.snapshot(), ctx.redact)
    assert session.messages[0].text == f"{block}\n\n{seed}\n\nq1"
    assert question_text(session.messages, 1, seed) == "q1"
    assert question_text(session.messages, 1) == f"{seed}\n\nq1"
    assert ask(session, "q1 changed", rewind=1, seed=seed).ok
    assert backend.requests[-1].messages[-1].text == f"{block}\n\n{seed}\n\nq1 changed"
    assert ask(session, "q2").ok
    assert ask(session, "q2 changed", rewind=2, seed=seed).ok       # question 2 did not start with it
    assert backend.requests[-1].messages[-1].text == "q2 changed"


# --- each provider's replayed data on the wire --------------------------------------------------------------------
def _wire(preset_id: str):
    backend = make_backend(PRESETS[preset_id], connection(preset_id), transport=Recorder([]).transport)
    return backend


def test_anthropic_thinking_of_the_questions_kept_is_replayed(tmp_path):
    wire = _wire("anthropic")

    def blocks(n: int, *rest: dict) -> dict:
        return {"anthropic_blocks": thinking(n)["anthropic_blocks"] + list(rest)}

    script = []
    for n in (1, 2, 3, 4):
        c = call(f"t{n}")
        script += [calls(c, extra=blocks(n, {"type": "tool_use", "id": c.id, "name": "echo", "input": c.arguments})),
                   say(f"answer {n}", extra=blocks(10 + n, {"type": "text", "text": f"answer {n}"}))]
    session, backend, _ctx = make(tmp_path, script, preset="anthropic", origin=wire.origin)
    for n in (1, 2, 3):
        assert ask(session, f"q{n}").ok
    assert ask(session, "q3 changed", rewind=3).ok
    signatures = []
    for req in backend.requests[-2:]:            # question 3's two requests the second time
        body = {"messages": wire.wire_messages(req)}
        history_check.validate_anthropic_body(body)
        signatures.append([b["signature"] for m in body["messages"] for b in m["content"]
                           if b.get("type") == "thinking"])
    assert signatures == [["sig-1", "sig-11", "sig-2", "sig-12"], ["sig-1", "sig-11", "sig-2", "sig-12", "sig-4"]]
    wire.close()


def test_gemini_signatures_of_the_questions_kept_are_replayed(tmp_path):
    wire = _wire("gemini")

    def signed(n: int) -> dict:
        return {"extra_content": {"google": {"thought_signature": f"gsig-{n}"}}}

    script = [calls(call("t1", signed(1))), say("one"), calls(call("t2", signed(2))), say("two"),
              calls(call("t2b", signed(3))), say("two again")]
    session, backend, _ctx = make(tmp_path, script, preset="gemini", origin=wire.origin)
    assert ask(session, "q1").ok and ask(session, "q2").ok
    assert ask(session, "q2 changed", rewind=2).ok
    first, second = (wire.wire_messages(r) for r in backend.requests[-2:])
    history_check.validate_openai_body({"messages": second})
    signatures = [c["extra_content"]["google"]["thought_signature"]
                  for m in second if m["role"] == "assistant" for c in m.get("tool_calls", [])]
    assert signatures == ["gsig-1", "gsig-3"]
    assert "gsig-2" not in json.dumps(first)
    wire.close()


def test_ollama_reasoning_of_a_deleted_question_is_never_sent(tmp_path):
    wire = _wire("ollama")
    key = PRESETS["ollama"].reasoning_wire_key
    script = [calls(call("t1"), extra={REASONING_TEXT_KEY: "R1"}), say("one"),
              calls(call("t2"), extra={REASONING_TEXT_KEY: "R2"}), say("two"),
              calls(call("t2b"), extra={REASONING_TEXT_KEY: "R3"}), say("two again")]
    session, backend, _ctx = make(tmp_path, script, preset="ollama", origin=wire.origin, tier="standard")
    assert ask(session, "q1").ok and ask(session, "q2").ok
    assert ask(session, "q2 changed", rewind=2).ok
    first, second = (wire.wire_messages(r) for r in backend.requests[-2:])
    for body in (first, second):
        history_check.validate_openai_body({"messages": body})
        assert "R1" not in json.dumps(body) and "R2" not in json.dumps(body)
    assert [m.get(key) for m in second if m["role"] == "assistant"] == [None, "R3"]
    wire.close()
