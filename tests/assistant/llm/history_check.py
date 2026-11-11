"""A conversation every provider accepts: each tool call has exactly one result, right after its call and in call
order; no message is empty; tool results are grouped the way the backend's wire format needs.

The session's own user messages (the note request, the first
message of a fresh context, the "tool limit reached" instruction) must be answered: each is the request's last
message or is followed by the model's answer of the same question.  No two user messages follow each other, and no
question holds the session's instruction texts (a question merged into one, review G11 C5-1).

:func:`validate` checks a neutral :class:`Request` and :func:`check_stored` a saved history (where no session
message is left unanswered); :func:`validate_openai_body` and :func:`validate_anthropic_body` check the body that
reached the endpoint.  Each raises ``AssertionError`` with the first problem.
"""
from __future__ import annotations

from flextool.assistant.llm_adapter import Message, Request
from flextool.assistant.session import SESSION_MESSAGES

#: The kinds of the session's own user messages (as P3 §4.7 names them; not taken from the session's code).
CONTROL_KINDS = frozenset({"note_request", "continuation", "limit"})
#: The session's instruction texts no question may hold.
_CONTROL_TEXTS = (SESSION_MESSAGES["limit"], SESSION_MESSAGES["note_request"])


def validate(req: Request) -> None:
    messages = list(req.messages)
    assert messages and messages[0].role == "user", "the conversation must start with a user message"
    _check_control(messages, last_may_wait=True)
    _check_pairs(messages)


def check_stored(messages: list[Message]) -> None:
    """A saved history: the checks of :func:`validate`, and no session message left without its answer."""
    messages = list(messages)
    if not messages:
        return
    assert messages[0].role == "user", "the conversation must start with a user message"
    _check_control(messages, last_may_wait=False)
    _check_pairs(messages)


def _check_control(messages: list[Message], *, last_may_wait: bool) -> None:
    for i, m in enumerate(messages):
        if m.role != "user":
            continue
        following = messages[i + 1] if i + 1 < len(messages) else None
        assert following is None or following.role != "user", f"message {i}: two user messages in a row"
        if m.kind in CONTROL_KINDS:
            if following is None:
                assert last_may_wait, f"message {i}: the session's {m.kind} message has no answer"
            else:
                assert following.role == "assistant" and following.turn == m.turn, (
                    f"message {i}: the session's {m.kind} message is not followed by its answer")
        else:
            for text in _CONTROL_TEXTS:
                assert text not in m.text, f"message {i}: a question holds the session's instruction {text[:30]!r}"


def _check_pairs(messages: list[Message]) -> None:
    seen: set[str] = set()
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.role == "user":
            assert m.text.strip(), f"message {i}: empty user text"
        elif m.role == "assistant":
            assert m.text.strip() or m.tool_calls, f"message {i}: empty assistant message"
            ids = [c.id for c in m.tool_calls]
            assert len(set(ids)) == len(ids), f"message {i}: repeated call ids {ids}"
            assert not seen & set(ids), f"message {i}: call id used before"
            seen |= set(ids)
            results = messages[i + 1:i + 1 + len(ids)]
            assert [r.role for r in results] == ["tool"] * len(ids), f"message {i}: results missing after calls"
            assert [r.tool_call_id for r in results] == ids, f"message {i}: results not in call order"
            for r in results:
                assert r.text.strip(), f"empty tool result for {r.tool_call_id}"
            i += len(ids)
        else:
            raise AssertionError(f"message {i}: a tool result without its call")
        i += 1


def validate_openai_body(body: dict) -> None:
    messages = [m for m in body["messages"] if m["role"] != "system"]
    assert messages and messages[0]["role"] == "user"
    i = 0
    while i < len(messages):
        m = messages[i]
        if m["role"] == "user":
            assert isinstance(m["content"], str) and m["content"].strip(), f"message {i}: empty user content"
        elif m["role"] == "assistant":
            calls = m.get("tool_calls") or []
            assert (m.get("content") or "").strip() or calls, f"message {i}: empty assistant message"
            ids = [c["id"] for c in calls]
            results = messages[i + 1:i + 1 + len(ids)]
            assert [r["role"] for r in results] == ["tool"] * len(ids), f"message {i}: results missing"
            assert [r["tool_call_id"] for r in results] == ids, f"message {i}: results not in call order"
            assert all(r["content"].strip() for r in results)
            i += len(ids)
        else:
            raise AssertionError(f"message {i}: a tool result without its call")
        i += 1


def validate_anthropic_body(body: dict) -> None:
    messages = body["messages"]
    assert messages and messages[0]["role"] == "user"
    for i, m in enumerate(messages):
        if i:
            assert m["role"] != messages[i - 1]["role"], f"message {i}: two {m['role']} messages in a row"
        assert m["content"], f"message {i}: no content"
        for block in m["content"]:
            if block["type"] == "text":
                assert block["text"].strip(), f"message {i}: empty text block"
        if m["role"] == "assistant":
            ids = [b["id"] for b in m["content"] if b["type"] == "tool_use"]
            if ids:
                assert i + 1 < len(messages), f"message {i}: calls without results"
                following = messages[i + 1]["content"]
                results = [b["tool_use_id"] for b in following if b["type"] == "tool_result"]
                assert results == ids, f"message {i}: results {results} do not match calls {ids}"
                assert [b["type"] for b in following[:len(ids)]] == ["tool_result"] * len(ids), (
                    f"message {i + 1}: tool results must come first")
        else:
            previous_calls = ([b["id"] for b in messages[i - 1]["content"] if b["type"] == "tool_use"]
                              if i else [])
            results = [b["tool_use_id"] for b in m["content"] if b["type"] == "tool_result"]
            assert results == previous_calls or not results and not previous_calls, (
                f"message {i}: tool results without their calls")
