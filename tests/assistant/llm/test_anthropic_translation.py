"""The Anthropic Messages backend against synthetic event streams (P1 §3.2)."""
from __future__ import annotations

import threading

import pytest

from flextool.assistant import llm_errors as E
from flextool.assistant.llm_adapter import (
    Message,
    Notice,
    TextDelta,
    Thinking,
    ToolCallStarted,
    drop_provider_extra,
    make_backend,
)
from flextool.assistant.providers import PRESETS
from tests.assistant.llm import history_check
from tests.assistant.llm.wire import (
    Recorder,
    anthropic_sse,
    collect,
    connection,
    json_response,
    payload,
    request,
    sse_response,
)

#: The content blocks of payloads/anthropic/thinking_tool_use.json, written out by hand.
THINKING_TOOL_USE_BLOCKS = [
    {"type": "thinking", "thinking": "Look up west, then the docs.", "signature": "U3ludGhldGljU2lnbmF0dXJlMDE="},
    {"type": "text", "text": "Let me look."},
    {"type": "tool_use", "id": "toolu_synthetic01", "name": "get_entity", "input": {"name": "west"}},
    {"type": "tool_use", "id": "toolu_synthetic02", "name": "search_docs", "input": {"query": "penalty"}},
]


def answer(name: str):
    return sse_response(anthropic_sse(payload("anthropic", name)["events"]))


def backend_for(responses: list, **conn_kw):
    recorder = Recorder(list(responses))
    backend = make_backend(PRESETS["anthropic"], connection("anthropic", **conn_kw), transport=recorder.transport)
    return backend, recorder


def ask() -> list[Message]:
    return [Message("user", "How many units has west?")]


def _count(obj: object, key: str) -> int:
    if isinstance(obj, dict):
        return sum((k == key) + _count(v, key) for k, v in obj.items())
    if isinstance(obj, list):
        return sum(_count(v, key) for v in obj)
    return 0


def _with_results(first, *, error_second: bool = False) -> list[Message]:
    calls = first.message.tool_calls
    return [Message("user", "How many units has west?"), first.message,
            Message("tool", '{"units": 2}', tool_call_id=calls[0].id, tool_name=calls[0].name),
            Message("tool", "There is no such section.", tool_call_id=calls[1].id, tool_name=calls[1].name,
                    is_error=error_second)]


def test_cached_prefix_and_nothing_else_is_sent():
    backend, recorder = backend_for([answer("text_end")])
    backend.stream(request(ask()), lambda e: None, threading.Event())
    body = recorder.last.body
    assert recorder.last.url == "https://api.anthropic.com/v1/messages"
    assert body["system"] == [{"type": "text", "text": "You are the assistant."},
                              {"type": "text", "text": "Layer two.", "cache_control": {"type": "ephemeral"}}]
    assert body["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in body["tools"][0]
    assert body["cache_control"] == {"type": "ephemeral"}
    assert _count(body, "cache_control") == 3       # of the four breakpoints allowed
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "How many units has west?"}]}]
    assert set(body) == {"model", "max_tokens", "messages", "system", "tools", "cache_control", "stream"}
    assert body["max_tokens"] == 1000 and body["stream"] is True


def test_no_cache_marks_for_an_uncached_request():
    """Grading v2 n3: a one-off prompt (``cache=False``, the grader's) carries no cache_control at all, so it pays
    no cache write."""
    import dataclasses

    backend, recorder = backend_for([answer("text_end")])
    backend.stream(dataclasses.replace(request(ask()), cache=False), lambda e: None, threading.Event())
    body = recorder.last.body
    assert _count(body, "cache_control") == 0
    assert body["system"] == [{"type": "text", "text": "You are the assistant."},
                              {"type": "text", "text": "Layer two."}]
    assert set(body) == {"model", "max_tokens", "messages", "system", "tools", "stream"}


def test_tool_choice_none():
    backend, recorder = backend_for([answer("text_end"), answer("text_end")])
    backend.stream(request(ask(), tool_choice="none"), lambda e: None, threading.Event())
    assert recorder.last.body["tool_choice"] == {"type": "none"}
    backend.stream(request(ask()), lambda e: None, threading.Event())
    assert "tool_choice" not in recorder.last.body


def test_thinking_and_tool_use_stream_and_are_replayed_verbatim_to_the_same_model():
    backend, recorder = backend_for([answer("thinking_tool_use"), answer("text_end")])
    events, on_event = collect()
    first = backend.stream(request(ask()), on_event, threading.Event())
    assert first.stop == "tools"
    assert first.message.text == "Let me look."
    assert [(c.id, c.name, c.arguments, c.raw_arguments) for c in first.message.tool_calls] == [
        ("toolu_synthetic01", "get_entity", {"name": "west"}, '{"name": "west"}'),
        ("toolu_synthetic02", "search_docs", {"query": "penalty"}, '{"query": "penalty"}')]
    assert first.message.provider_extra == {"anthropic_blocks": THINKING_TOOL_USE_BLOCKS}
    kinds = [type(e).__name__ for e in events if not isinstance(e, TextDelta)]
    assert kinds == ["Thinking", "ToolCallStarted", "ToolCallStarted"]
    assert [(e.index, e.name) for e in events if isinstance(e, ToolCallStarted)] == [(0, "get_entity"),
                                                                                     (1, "search_docs")]
    assert (first.usage.input_uncached, first.usage.cache_read, first.usage.cache_write, first.usage.output) == (
        1200, 1000, 0, 80)

    second = backend.stream(request(_with_results(first, error_second=True)), lambda e: None, threading.Event())
    assert second.message.text == "West has 2 units." and second.stop == "end"
    body = recorder.last.body
    assert body["messages"][1] == {"role": "assistant", "content": THINKING_TOOL_USE_BLOCKS}
    assert body["messages"][2] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_synthetic01", "content": '{"units": 2}', "is_error": False},
        {"type": "tool_result", "tool_use_id": "toolu_synthetic02", "content": "There is no such section.",
         "is_error": True}]}
    history_check.validate_anthropic_body(body)


def test_another_model_or_dropped_extras_get_the_message_without_thinking():
    backend, _ = backend_for([answer("thinking_tool_use")])
    first = backend.stream(request(ask()), lambda e: None, threading.Event())
    rebuilt = [{"type": "text", "text": "Let me look."},
               {"type": "tool_use", "id": "toolu_synthetic01", "name": "get_entity", "input": {"name": "west"}},
               {"type": "tool_use", "id": "toolu_synthetic02", "name": "search_docs", "input": {"query": "penalty"}}]
    other, recorder = backend_for([answer("text_end")], model="another-model")
    other.stream(request(_with_results(first)), lambda e: None, threading.Event())
    assert recorder.last.body["messages"][1] == {"role": "assistant", "content": rebuilt}
    history_check.validate_anthropic_body(recorder.last.body)
    same, recorder = backend_for([answer("text_end")])
    same.stream(request(drop_provider_extra(_with_results(first))), lambda e: None, threading.Event())
    assert recorder.last.body["messages"][1] == {"role": "assistant", "content": rebuilt}


def test_redacted_and_empty_thinking_blocks_are_kept():
    backend, recorder = backend_for([answer("redacted_and_empty_thinking"), answer("text_end")])
    events, on_event = collect()
    first = backend.stream(request(ask()), on_event, threading.Event())
    assert sum(isinstance(e, Thinking) for e in events) == 2
    blocks = [{"type": "redacted_thinking", "data": "RW5jcnlwdGVkU3ludGhldGlj"},
              {"type": "thinking", "thinking": "", "signature": "U3ludGhldGljU2lnbmF0dXJlMDI="},
              {"type": "text", "text": "Done."}]
    assert first.message.provider_extra["anthropic_blocks"] == blocks
    backend.stream(request([*ask(), first.message, Message("user", "And east?")]), lambda e: None,
                   threading.Event())
    assert recorder.last.body["messages"][1]["content"] == blocks


def test_a_user_message_after_tool_results_joins_their_user_message():
    backend, _ = backend_for([answer("thinking_tool_use")])
    first = backend.stream(request(ask()), lambda e: None, threading.Event())
    other, recorder = backend_for([answer("text_end")])
    messages = [*_with_results(first), Message("user", "The tool limit for this question is reached.")]
    other.stream(request(messages, tool_choice="none"), lambda e: None, threading.Event())
    last = recorder.last.body["messages"][-1]
    assert [b["type"] for b in last["content"]] == ["tool_result", "tool_result", "text"]
    history_check.validate_anthropic_body(recorder.last.body)


def _stop_events(reason: str, details: dict | None = None) -> list[dict]:
    events = payload("anthropic", "text_end")["events"]
    delta = events[-2]["delta"]
    delta["stop_reason"] = reason
    if details is not None:
        delta["stop_details"] = details
    return events


@pytest.mark.parametrize("reason,stop", [("tool_use", "tools"), ("end_turn", "end"), ("stop_sequence", "end"),
                                         ("pause_turn", "end"), ("max_tokens", "length"),
                                         ("model_context_window_exceeded", "context"), ("refusal", "refused")])
def test_every_stop_reason(reason, stop):
    backend, _ = backend_for([sse_response(anthropic_sse(_stop_events(reason)))])
    events, on_event = collect()
    reply = backend.stream(request(ask()), on_event, threading.Event())
    assert reply.stop == stop
    assert not [e for e in events if isinstance(e, Notice)]


def test_an_unknown_stop_reason_ends_with_a_notice():
    backend, _ = backend_for([sse_response(anthropic_sse(_stop_events("a_future_reason")))])
    events, on_event = collect()
    reply = backend.stream(request(ask()), on_event, threading.Event())
    assert reply.stop == "end"
    assert [e.text for e in events if isinstance(e, Notice)] == [
        "The provider ended the answer for a reason FlexTool does not know (a_future_reason)."]


@pytest.mark.parametrize("details,reason", [
    ({"type": "refusal", "category": "cyber", "explanation": None}, "computer security"),
    ({"type": "refusal", "category": "bio", "explanation": None}, "biology or chemistry safety"),
    ({"type": "refusal", "category": "general_harms", "explanation": None}, "possible harm"),
    ({"type": "refusal", "category": "frontier_llm", "explanation": None}, "the provider's usage rules"),
    ({"type": "refusal", "category": None, "explanation": "This asks for something we cannot help with."},
     "This asks for something we cannot help with."),
    ({"type": "refusal", "category": None, "explanation": None}, None),
])
def test_refusal_reasons_in_plain_english(details, reason):
    backend, _ = backend_for([sse_response(anthropic_sse(_stop_events("refusal", details)))])
    reply = backend.stream(request(ask()), lambda e: None, threading.Event())
    assert reply.stop == "refused" and reply.refusal_reason == reason


def test_a_refusal_without_content_is_no_message():
    backend, _ = backend_for([answer("refusal")])
    reply = backend.stream(request(ask()), lambda e: None, threading.Event())
    assert reply.message is None and reply.stop == "refused" and reply.refusal_reason == "computer security"


def test_a_midstream_error_keeps_the_text_and_the_input_usage():
    backend, _ = backend_for([answer("midstream_error")])
    with pytest.raises(E.ServerBusy) as info:
        backend.stream(request(ask()), lambda e: None, threading.Event())
    err = info.value
    assert err.streamed and err.partial_text == "Partial "
    usage = err.partial_usage
    assert (usage.input_uncached, usage.cache_read, usage.cache_write, usage.output) == (900, 800, 0, None)
    assert err.user_message == "Anthropic stopped the answer because it is overloaded. Ask again in a moment."


def test_a_status_error_before_the_stream():
    backend, _ = backend_for([json_response(529, {"type": "error", "error": {"type": "overloaded_error",
                                                                              "message": "Overloaded"}})])
    with pytest.raises(E.ServerBusy) as info:
        backend.stream(request(ask()), lambda e: None, threading.Event())
    assert not info.value.streamed and info.value.partial_usage is None
    assert info.value.user_message == "Anthropic is still overloaded. Ask again in a moment."


def test_cancel_between_events_keeps_the_partial_text():
    cancel = threading.Event()
    events = payload("anthropic", "thinking_tool_use")["events"]

    def parts():
        yield anthropic_sse(events[:8])
        yield anthropic_sse(events[8:])

    def on_event(event):
        if isinstance(event, TextDelta):
            cancel.set()

    backend, _ = backend_for([lambda request: sse_response(parts())])
    with pytest.raises(E.Cancelled) as info:
        backend.stream(request(ask()), on_event, cancel)
    assert info.value.partial_text == "Let me " and info.value.streamed
    assert info.value.partial_usage.input_uncached == 1200


def test_the_model_list_carries_the_context_window():
    models = {"data": [{"type": "model", "id": "example-model", "display_name": "Example", "created_at":
                        "2026-01-01T00:00:00Z", "max_input_tokens": 200000, "max_tokens": 64000}],
              "has_more": False, "first_id": "example-model", "last_id": "example-model"}
    backend, recorder = backend_for([json_response(200, models)])
    assert [(m.id, m.context_window) for m in backend.list_models()] == [("example-model", 200000)]
    assert recorder.last.url.startswith("https://api.anthropic.com/v1/models")


@pytest.mark.parametrize("cut", ["text", "tool"])
def test_a_stream_cut_before_message_stop_is_truncated(cut):
    """The connection broke off mid-answer, before ``message_delta`` gave a stop reason: the answer is incomplete
    (as on the Chat Completions backend), never a finished turn with a half-written answer or tool call."""
    events: list[dict] = [{"type": "message_start", "message": {
        "id": "msg_cut", "type": "message", "role": "assistant", "model": "example-model", "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 1}}}]
    if cut == "text":
        events += [{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                   {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Half"}}]
    else:
        events += [{"type": "content_block_start", "index": 0,
                    "content_block": {"type": "tool_use", "id": "toolu_cut", "name": "get_entity", "input": {}}},
                   {"type": "content_block_delta", "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '{"a":'}}]
    backend, _ = backend_for([sse_response(anthropic_sse(events))])
    seen, on_event = collect()
    with pytest.raises(E.Truncated) as info:
        backend.stream(request(ask()), on_event, threading.Event())
    err = info.value
    assert err.user_message == "The connection ended before the answer was complete."
    assert err.streamed is True and err.partial_text == ("Half" if cut == "text" else "")
    assert err.partial_usage.input_uncached == 10 and err.partial_usage.output is None
    assert not [e for e in seen if isinstance(e, Notice)]
