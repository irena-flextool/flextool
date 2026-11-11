"""Whole (not streamed) answers of the Chat Completions backend: the calibration's ``run --no-stream``.

The payloads have the shape of Google's OpenAI-compatible endpoint answering Gemma without streaming: the message's
content holds the reasoning as a ``<thought>…</thought>`` block before the answer, the message carries
``extra_content.google.thought``, and a tool call carries its thought signature in ``extra_content``.  A whole
answer must give the same reply as the same answer streamed: the text without the reasoning, the tool calls with
the signature to replay, the usage, the stop, and the errors classified alike.
"""
from __future__ import annotations

import dataclasses
import json
import threading

import pytest

from flextool.assistant import llm_errors as E
from flextool.assistant.llm_adapter import Message, TextDelta, ToolCallStarted, make_backend
from flextool.assistant.providers import PRESETS
from flextool.assistant.thoughts import stream_visible
from tests.assistant.llm.wire import Recorder, collect, connection, json_response, openai_sse, request, sse_response

MODEL = "gemma-4-31b-it"
SIGNATURE = {"google": {"thought_signature": "R2VtbWEtc3ludGhldGljLXNpZ25hdHVyZQ=="}}
THOUGHT_MARK = {"google": {"thought": True}}
USAGE = {"completion_tokens": 31, "prompt_tokens": 5120, "total_tokens": 5151}


def _whole(message: dict, finish: str = "stop", usage: dict | None = USAGE) -> dict:
    body = {"choices": [{"finish_reason": finish, "index": 0, "message": {"role": "assistant", **message}}],
            "created": 1794301738, "id": "Ht_yaoHOOJG3nsEP2srYyQw", "model": MODEL, "object": "chat.completion"}
    if usage is not None:
        body["usage"] = usage
    return body


TEXT = _whole({"content": "<thought>The user wants the units of west.\nget_entity gave two.</thought>"
                          "West has two units.", "extra_content": THOUGHT_MARK})
CALL = _whole({"content": "<thought>I should use the `get_entity` tool.</thought>", "extra_content": THOUGHT_MARK,
               "tool_calls": [{"extra_content": SIGNATURE,
                               "function": {"arguments": "{\"name\":\"west\"}", "name": "get_entity"},
                               "id": "function-call-8714", "type": "function"},
                              {"function": {"arguments": "{\"query\":\"units\"}", "name": "search_docs"},
                               "id": "function-call-8715", "type": "function"}]})


def _streamed(body: dict) -> bytes:
    """The same answer as the endpoint streams it: the content in two pieces, then the calls, then the finish."""
    message = body["choices"][0]["message"]
    content = message.get("content") or ""
    half = len(content) // 2
    chunks = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": piece,
                                                  "extra_content": THOUGHT_MARK}}]}
              for piece in (content[:half], content[half:]) if piece]
    for n, call in enumerate(message.get("tool_calls") or []):
        chunks.append({"choices": [{"index": 0, "delta": {"tool_calls": [{**call, "index": n}]}}]})
    chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": body["choices"][0]["finish_reason"]}],
                   "usage": body.get("usage")})
    return openai_sse(chunks)


def backend_for(responses: list, *, stream: bool):
    recorder = Recorder(list(responses))
    conn = dataclasses.replace(connection("gemini", model=MODEL), stream=stream)
    return make_backend(PRESETS["gemini"], conn, transport=recorder.transport), recorder


def ask(backend, messages=None):
    events, sink = collect()
    reply = stream_visible(backend, request(messages or [Message("user", "How many units has west?")]), sink,
                           threading.Event())
    return reply, events


def test_a_whole_request_asks_for_no_stream():
    """The same request to the same address with the same headers, but ``stream: false`` and no stream options."""
    backend, recorder = backend_for([sse_response(_streamed(TEXT))], stream=True)
    ask(backend)
    streamed = recorder.last
    backend, recorder = backend_for([json_response(200, TEXT)], stream=False)
    ask(backend)
    body = recorder.last.body
    assert recorder.last.url == streamed.url
    framing = ("content-length",)               # the bodies differ in length only
    assert {k: v for k, v in recorder.last.headers.items() if k not in framing} == \
        {k: v for k, v in streamed.headers.items() if k not in framing}
    assert {k: v for k, v in body.items() if k not in ("stream", "stream_options")} == \
        {k: v for k, v in streamed.body.items() if k not in ("stream", "stream_options")}
    assert body["stream"] is False
    assert "stream_options" not in body          # refused without streaming; a whole answer carries its usage
    assert streamed.body["stream"] is True and streamed.body["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize("body", [TEXT, CALL], ids=["text", "tool calls"])
def test_a_whole_answer_gives_the_reply_of_the_same_answer_streamed(body):
    whole, whole_events = ask(backend_for([json_response(200, body)], stream=False)[0])
    streamed, streamed_events = ask(backend_for([sse_response(_streamed(body))], stream=True)[0])
    assert whole.stop == streamed.stop and whole.usage == streamed.usage
    assert whole.message.text == streamed.message.text
    assert [(c.id, c.name, c.arguments, c.raw_arguments, c.provider_extra) for c in whole.message.tool_calls] == \
        [(c.id, c.name, c.arguments, c.raw_arguments, c.provider_extra) for c in streamed.message.tool_calls]
    assert whole.message.provider_extra == streamed.message.provider_extra
    shown = "".join(e.text for e in whole_events if isinstance(e, TextDelta))
    assert shown == "".join(e.text for e in streamed_events if isinstance(e, TextDelta))
    assert "<thought>" not in shown


def test_the_whole_text_answer_without_its_reasoning():
    reply, events = ask(backend_for([json_response(200, TEXT)], stream=False)[0])
    assert reply.stop == "end"
    assert reply.message.text == "West has two units."
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "West has two units."
    assert (reply.usage.input_uncached, reply.usage.output) == (5120, 31)


def test_the_whole_tool_calls_keep_the_signature_and_replay_it():
    backend, recorder = backend_for([json_response(200, CALL), json_response(200, TEXT)], stream=False)
    reply, events = ask(backend)
    assert reply.stop == "tools"                 # Google says "stop" although the model called tools
    assert reply.message.text == ""
    assert [e.name for e in events if isinstance(e, ToolCallStarted)] == ["get_entity", "search_docs"]
    first, second = reply.message.tool_calls
    assert (first.id, first.name, first.arguments) == ("function-call-8714", "get_entity", {"name": "west"})
    assert first.provider_extra == {"extra_content": SIGNATURE}
    assert (second.id, second.arguments, second.provider_extra) == ("function-call-8715", {"query": "units"}, {})
    results = [Message("tool", json.dumps({"name": "west", "units": 2}), tool_call_id=first.id,
                       tool_name="get_entity"),
               Message("tool", "[]", tool_call_id=second.id, tool_name="search_docs")]
    ask(backend, [Message("user", "How many units has west?"), reply.message, *results])
    sent = recorder.last.body["messages"]
    assert sent[2]["tool_calls"][0] == {"id": "function-call-8714", "type": "function",
                                        "function": {"name": "get_entity", "arguments": "{\"name\":\"west\"}"},
                                        "extra_content": SIGNATURE}
    assert "extra_content" not in sent[2]["tool_calls"][1]


def test_a_whole_answer_cut_at_the_output_limit():
    body = _whole({"content": "<thought>Long.</thought>West has"}, finish="length")
    reply, _ = ask(backend_for([json_response(200, body)], stream=False)[0])
    assert (reply.stop, reply.message.text) == ("length", "West has")


def test_whole_answer_errors_are_classified_as_streamed_ones():
    """HTTP 429 with Google's array body: a rate limit carrying Google's words; HTTP 200 with an error body: the
    error in it; a body that is no completion: an incomplete answer."""
    quota = [{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                        "message": "Quota exceeded for metric: generate_content_free_tier_input_token_count, "
                                   "limit: 15000"}}]
    backend, _ = backend_for([json_response(429, quota)], stream=False)
    with pytest.raises(E.RateLimited) as caught:
        ask(backend)
    # Gemini quota fixes 2026-11-10: the quota the message names, in words.
    assert caught.value.provider_said == E.ProviderSaid(
        429, "RESOURCE_EXHAUSTED", "quota: input tokens (limit 15000).",
        (E.QuotaViolation("generate_content_free_tier_input_token_count", "", "15000"),))
    busy = [{"error": {"code": 503, "message": "The model is overloaded.", "status": "UNAVAILABLE"}}]
    backend, _ = backend_for([json_response(200, busy)], stream=False)
    with pytest.raises(E.ServerBusy) as caught:
        ask(backend)
    assert not caught.value.streamed
    assert caught.value.provider_said == E.ProviderSaid(503, "UNAVAILABLE", "The model is overloaded.")
    backend, _ = backend_for([json_response(200, {"id": "x"})], stream=False)
    with pytest.raises(E.Truncated):
        ask(backend)
