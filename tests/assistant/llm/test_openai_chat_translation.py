"""The Chat Completions backend against synthetic payloads of every OpenAI-compatible preset (P1 §3.1)."""
from __future__ import annotations

import dataclasses
import socket
import threading
import time

import httpx2
import pytest

from flextool.assistant import llm_errors as E
from flextool.assistant.llm_adapter import (
    ID_LENGTH,
    Message,
    Notice,
    TextDelta,
    ToolCallStarted,
    Waiting,
    make_backend,
    prefix_key,
    remap_ids,
)
from flextool.assistant.providers import PRESETS
from tests.assistant.llm import history_check
from tests.assistant.llm.wire import (
    TOOLS,
    Recorder,
    collect,
    connection,
    json_response,
    openai_sse,
    payload,
    request,
    sse_response,
)

OPENAI_PRESETS = sorted(p.id for p in PRESETS.values() if p.backend == "openai")


def backend_for(preset_id: str, responses: list, **conn_kw):
    recorder = Recorder(list(responses))
    backend = make_backend(PRESETS[preset_id], connection(preset_id, **conn_kw), transport=recorder.transport)
    return backend, recorder


def answer(preset: str, name: str):
    return sse_response(openai_sse(payload(preset, name)["chunks"]))


def ask(text: str = "How many units has west?") -> list[Message]:
    return [Message("user", text)]


@pytest.mark.parametrize("preset_id", OPENAI_PRESETS)
def test_the_body_holds_only_fields_the_preset_allows(preset_id):
    preset = PRESETS[preset_id]
    backend, recorder = backend_for(preset_id, [answer("openai", "text_usage")])
    backend.stream(request(ask()), lambda e: None, threading.Event())
    body = recorder.last.body
    allowed = {"model", "messages", "stream", "tools", preset.max_tokens_param} | preset.allowed_params
    assert set(body) <= allowed
    assert body[preset.max_tokens_param] == 1000
    assert body["messages"][0] == {"role": "system", "content": "You are the assistant.\n\nLayer two."}
    assert [t["function"]["name"] for t in body["tools"]] == ["get_entity", "search_docs"]
    history_check.validate_openai_body(body)
    if preset_id == "mistral":
        assert not {"store", "stream_options", "max_completion_tokens"} & set(body)
    if preset_id == "openai":
        assert body["store"] is False
        assert "max_tokens" not in body
    else:
        assert "store" not in body
    if "stream_options" in preset.allowed_params:
        assert body["stream_options"] == {"include_usage": True}


def test_openrouter_privacy_fields():
    backend, recorder = backend_for("openrouter", [answer("openai", "text_usage")] * 3)
    backend.stream(request(ask()), lambda e: None, threading.Event())
    assert recorder.last.body["provider"] == {"data_collection": "deny", "require_parameters": True}
    backend.stream(request(ask(), tools=()), lambda e: None, threading.Event())
    assert recorder.last.body["provider"] == {"data_collection": "deny"}
    zdr, recorder = backend_for("openrouter", [answer("openai", "text_usage")],
                                knobs={"openrouter.zdr": True})
    zdr.stream(request(ask()), lambda e: None, threading.Event())
    assert recorder.last.body["provider"] == {"data_collection": "deny", "zdr": True, "require_parameters": True}
    relaxed, recorder = backend_for("openrouter", [answer("openai", "text_usage")],
                                    knobs={"openrouter.data_collection": "allow"})
    relaxed.stream(request(ask(), tools=()), lambda e: None, threading.Event())
    assert recorder.last.body["provider"] == {"data_collection": "allow"}


def test_tool_choice_none_and_presets_without_it():
    backend, recorder = backend_for("openai", [answer("openai", "text_usage")])
    backend.stream(request(ask(), tool_choice="none"), lambda e: None, threading.Event())
    assert recorder.last.body["tool_choice"] == "none" and "tools" in recorder.last.body
    generic, recorder = backend_for("openai_compatible", [answer("openai", "text_usage")])
    generic.stream(request(ask(), tool_choice="auto"), lambda e: None, threading.Event())
    assert recorder.last.body["tool_choice"] == "auto"


def test_text_streams_and_usage_counts_cached_tokens():
    backend, _ = backend_for("openai", [answer("openai", "text_usage")])
    events, on_event = collect()
    reply = backend.stream(request(ask()), on_event, threading.Event())
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "The west node has 2 units."
    assert reply.stop == "end" and reply.message.text == "The west node has 2 units."
    assert reply.message.origin == ("openai", "https://api.openai.com/v1", "example-model")
    assert (reply.usage.input_uncached, reply.usage.cache_read, reply.usage.cache_write, reply.usage.output) == (
        310, 2900, None, 412)
    assert reply.usage.reported and reply.usage.cost is None


def test_parallel_calls_with_split_arguments():
    backend, _ = backend_for("openai", [answer("openai", "parallel_calls")])
    events, on_event = collect()
    reply = backend.stream(request(ask()), on_event, threading.Event())
    assert reply.stop == "tools"
    calls = reply.message.tool_calls
    assert [(c.id, c.name, c.arguments) for c in calls] == [("call_synthA", "get_entity", {"name": "west"}),
                                                            ("call_synthB", "search_docs", {"query": "penalty"})]
    assert calls[0].raw_arguments == '{"name": "west"}'
    assert [(e.index, e.name) for e in events if isinstance(e, ToolCallStarted)] == [(0, "get_entity"),
                                                                                     (1, "search_docs")]
    assert reply.message.text == "" and reply.message.provider_extra == {}


def _round_two(reply, results: list[str]) -> list[Message]:
    messages = [Message("user", "How many units has west?"), reply.message]
    for call, text in zip(reply.message.tool_calls, results, strict=True):
        messages.append(Message("tool", text, tool_call_id=call.id, tool_name=call.name))
    return messages


def test_gemini_signature_is_replayed_byte_identical_and_dropped_after_a_switch():
    backend, recorder = backend_for("gemini", [answer("gemini", "signature_call"), answer("gemini", "text")])
    first = backend.stream(request(ask()), lambda e: None, threading.Event())
    assert first.stop == "tools"           # Gemini says "stop" although it called a tool
    signature = payload("gemini", "signature_call")["chunks"][0]["choices"][0]["delta"]["tool_calls"][0][
        "extra_content"]
    assert first.message.tool_calls[0].provider_extra == {"extra_content": signature}
    messages = _round_two(first, ['{"name": "west", "units": 2}'])
    second = backend.stream(request(messages), lambda e: None, threading.Event())
    assert second.message.text == "West has 2 units."
    sent = recorder.last.body["messages"]
    assert sent[2]["tool_calls"][0] == {"id": "function-call-synth1", "type": "function",
                                        "function": {"name": "get_entity", "arguments": '{"name":"west"}'},
                                        "extra_content": signature}
    assert sent[3] == {"role": "tool", "tool_call_id": "function-call-synth1",
                       "content": '{"name": "west", "units": 2}'}
    history_check.validate_openai_body(recorder.last.body)

    # Another provider gets the conversation without the signature and with fresh ids.
    switched = remap_ids(messages)
    other, recorder = backend_for("openai", [answer("openai", "text_usage")])
    other.stream(request(switched), lambda e: None, threading.Event())
    sent = recorder.last.body["messages"]
    call = sent[2]["tool_calls"][0]
    assert "extra_content" not in call
    assert len(call["id"]) == ID_LENGTH and call["id"].isalnum() and call["id"] != "function-call-synth1"
    assert sent[3]["tool_call_id"] == call["id"]
    history_check.validate_openai_body(recorder.last.body)


def test_extras_are_not_replayed_to_another_model_of_the_same_service():
    backend, _ = backend_for("gemini", [answer("gemini", "signature_call")])
    first = backend.stream(request(ask()), lambda e: None, threading.Event())
    other_model, recorder = backend_for("gemini", [answer("gemini", "text")], model="another-model")
    other_model.stream(request(_round_two(first, ["{}"])), lambda e: None, threading.Event())
    assert "extra_content" not in recorder.last.body["messages"][2]["tool_calls"][0]


def test_reasoning_details_are_merged_and_replayed_with_cost_and_cache_writes():
    backend, recorder = backend_for("openrouter", [answer("openrouter", "reasoning_cost"),
                                                   answer("openrouter", "reasoning_cost")])
    reply = backend.stream(request(ask()), lambda e: None, threading.Event())
    assert reply.message.provider_extra == {"reasoning_details": [
        {"type": "reasoning.text", "text": "First I check the node.", "index": 0, "format": "example-v1",
         "signature": "c2lnbmF0dXJlLXN5bnRoZXRpYw=="}]}
    assert (reply.usage.input_uncached, reply.usage.cache_read, reply.usage.cache_write, reply.usage.output,
            reply.usage.cost) == (300, 600, 100, 50, pytest.approx(0.00042))
    messages = [Message("user", "q"), reply.message, Message("user", "and the east?")]
    backend.stream(request(messages), lambda e: None, threading.Event())
    sent = recorder.last.body["messages"][2]
    assert sent == {"role": "assistant", "content": "Done.",
                    "reasoning_details": reply.message.provider_extra["reasoning_details"]}


def test_openrouter_cost_zero_is_a_reported_cost():
    backend, _ = backend_for("openrouter", [answer("openrouter", "tool_call")])
    reply = backend.stream(request(ask()), lambda e: None, threading.Event())
    assert reply.usage.cost == 0.0 and reply.stop == "tools"


def test_no_usage_is_not_reported():
    backend, _ = backend_for("openai_compatible", [answer("openai_compatible", "text")])
    reply = backend.stream(request(ask()), lambda e: None, threading.Event())
    assert reply.usage.reported is False and reply.usage.output is None


def test_refusal_delta():
    backend, _ = backend_for("openai", [answer("openai", "refusal")])
    reply = backend.stream(request(ask()), lambda e: None, threading.Event())
    assert reply.stop == "refused" and reply.message.text == "I can't help with that."


def test_length_stop_keeps_the_partial_call():
    backend, _ = backend_for("openai", [answer("openai", "length")])
    reply = backend.stream(request(ask()), lambda e: None, threading.Event())
    assert reply.stop == "length"
    assert reply.message.text == "Let me look. "
    assert reply.message.tool_calls[0].arguments is None
    assert reply.message.tool_calls[0].raw_arguments == '{"name": "we'


def test_error_chunk_keeps_the_partial_text():
    backend, _ = backend_for("openrouter", [answer("openrouter", "error_chunk")])
    with pytest.raises(E.ServerBusy) as info:
        backend.stream(request(ask()), lambda e: None, threading.Event())
    err = info.value
    assert err.streamed and err.partial_text == "Partial " and err.partial_usage.reported is False


def test_a_stream_without_a_finish_reason_is_truncated():
    backend, _ = backend_for("openai", [sse_response(openai_sse(payload("openai", "no_finish")["chunks"],
                                                                done=False))])
    with pytest.raises(E.Truncated) as info:
        backend.stream(request(ask()), lambda e: None, threading.Event())
    assert info.value.user_message == "The connection ended before the answer was complete."
    assert info.value.partial_text == "Half an ans" and info.value.streamed


def test_an_unknown_finish_reason_ends_with_a_notice():
    chunks = payload("lmstudio", "text")["chunks"]
    chunks[1]["choices"][0]["finish_reason"] = "eos_token"
    backend, _ = backend_for("lmstudio", [sse_response(openai_sse(chunks))])
    events, on_event = collect()
    reply = backend.stream(request(ask()), on_event, threading.Event())
    assert reply.stop == "end"
    assert [e.text for e in events if isinstance(e, Notice)] == [
        "The provider ended the answer for a reason FlexTool does not know (eos_token)."]


def test_mistral_tool_results_carry_the_name_and_errors_say_so():
    backend, recorder = backend_for("mistral", [answer("mistral", "tool_call"), answer("lmstudio", "text")])
    first = backend.stream(request(ask()), lambda e: None, threading.Event())
    assert first.message.tool_calls[0].id == "Ab3dE6gH9"
    messages = [Message("user", "q"), first.message,
                Message("tool", "There is no entity west.", tool_call_id="Ab3dE6gH9", tool_name="get_entity",
                        is_error=True)]
    backend.stream(request(messages), lambda e: None, threading.Event())
    assert recorder.last.body["messages"][3] == {"role": "tool", "tool_call_id": "Ab3dE6gH9",
                                                 "name": "get_entity",
                                                 "content": "Error: There is no entity west."}


def test_a_call_without_an_id_gets_one():
    chunks = payload("ollama", "tool_call")["chunks"]
    del chunks[0]["choices"][0]["delta"]["tool_calls"][0]["id"]
    backend, _ = backend_for("ollama", [sse_response(openai_sse(chunks))])
    reply = backend.stream(request(ask()), lambda e: None, threading.Event())
    call_id = reply.message.tool_calls[0].id
    assert len(call_id) == ID_LENGTH and call_id.isalnum()


def test_an_empty_answer_is_no_message():
    chunks = [{"id": "x", "object": "chat.completion.chunk", "created": 1, "model": "m",
               "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": "stop"}]}]
    backend, _ = backend_for("openai", [sse_response(openai_sse(chunks))])
    reply = backend.stream(request(ask()), lambda e: None, threading.Event())
    assert reply.message is None and reply.stop == "end"


def test_prompt_cache_key_is_the_same_for_two_conversations_of_a_tier():
    system = ("You are the assistant.", "Layer two.")
    key = prefix_key(system, TOOLS)
    backend, recorder = backend_for("openai", [answer("openai", "text_usage"), answer("openai", "text_usage")])
    backend.stream(request(ask("first conversation"), prefix_key=key), lambda e: None, threading.Event())
    backend.stream(request(ask("another one"), prefix_key=prefix_key(system, TOOLS)), lambda e: None,
                   threading.Event())
    keys = [r.body["prompt_cache_key"] for r in recorder.requests]
    assert keys[0] == keys[1] == key and len(key) == 16
    assert prefix_key(("other",), TOOLS) != key
    mistral, recorder = backend_for("mistral", [answer("openai", "text_usage")])
    mistral.stream(request(ask(), prefix_key=key), lambda e: None, threading.Event())
    assert "prompt_cache_key" not in recorder.last.body


def test_status_errors_are_classified_with_the_route_label():
    backend, _ = backend_for("gemini", [json_response(429, [{"error": {
        "code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota",
        "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12s"}]}}])])
    with pytest.raises(E.RateLimited) as info:
        backend.stream(request(ask()), lambda e: None, threading.Event())
    assert info.value.retry_after == 12.0 and info.value.provider == "Google Gemini"
    assert info.value.partial_usage is None and not info.value.streamed


def test_cancel_between_chunks_keeps_the_partial_text():
    cancel = threading.Event()
    chunks = payload("openai", "text_usage")["chunks"]

    def parts():
        yield openai_sse(chunks[:2], done=False)
        yield openai_sse(chunks[2:])

    def on_event(event):
        if isinstance(event, TextDelta):
            cancel.set()

    backend, _ = backend_for("openai", [lambda request: sse_response(parts())])
    with pytest.raises(E.Cancelled) as info:
        backend.stream(request(ask()), on_event, cancel)
    assert info.value.partial_text == "The west node " and info.value.streamed
    assert info.value.user_message == "Stopped."


def test_the_model_list_is_filtered():
    models = {"object": "list", "data": [
        {"id": "example-chat", "object": "model", "created": 1, "owned_by": "x"},
        {"id": "text-embedding-example", "object": "model", "created": 1, "owned_by": "x"},
        {"id": "example-tts", "object": "model", "created": 1, "owned_by": "x"}]}
    backend, recorder = backend_for("openai", [json_response(200, models)])
    # The saved list keeps every id; the shown list hides the non-chat models (test_providers, RF-D12).
    assert [m.id for m in backend.list_models()] == ["example-chat", "example-tts", "text-embedding-example"]
    assert recorder.last.url == "https://api.openai.com/v1/models"
    routed = {"data": [{"id": "a/with-tools", "context_length": 32768, "supported_parameters": ["tools"]},
                       {"id": "b/no-tools", "context_length": 8192, "supported_parameters": ["temperature"]}]}
    backend, _ = backend_for("openrouter", [json_response(200, routed)])
    assert [(m.id, m.context_window) for m in backend.list_models()] == [("a/with-tools", 32768)]
    gemini = {"object": "list", "data": [{"id": "models/example-flash", "object": "model", "owned_by": "google"}]}
    backend, _ = backend_for("gemini", [json_response(200, gemini)])
    assert [m.id for m in backend.list_models()] == ["example-flash"]
    local = {"object": "list", "data": [{"id": "example:7b", "object": "model"},
                                        {"id": "example:120b-cloud", "object": "model"}]}
    backend, _ = backend_for("ollama", [json_response(200, local)])
    assert [(m.id, m.note) for m in backend.list_models()] == [("example:120b-cloud", "runs on Ollama cloud"),
                                                                ("example:7b", None)]
    backend, _ = backend_for("openai", [json_response(401, {"error": {"message": "bad key"}})])
    with pytest.raises(E.AuthFailed):
        backend.list_models()


def test_a_required_key_must_be_there():
    with pytest.raises(E.NoKey) as info:
        make_backend(PRESETS["gemini"], connection("gemini", key=None))
    assert info.value.user_message == ("No key found for Google Gemini. Set the environment variable GEMINI_API_KEY, "
                                       "or store the key with flextool-assistant --set-key.")


def test_abort_wakes_a_request_waiting_for_the_first_byte():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    port = server.getsockname()[1]
    accepted = threading.Event()
    stop_server = threading.Event()

    def serve():
        conn, _ = server.accept()
        accepted.set()
        stop_server.wait(10)
        conn.close()

    threading.Thread(target=serve, daemon=True).start()
    conn = connection("ollama", url=f"http://127.0.0.1:{port}/v1", key=None)
    backend = make_backend(PRESETS["ollama"], conn)
    backend.wait_interval = 0.1
    events, on_event = collect()
    outcome: list = []

    def worker():
        try:
            outcome.append(backend.stream(request(ask()), on_event, threading.Event()))
        except Exception as exc:  # noqa: BLE001 — the outcome is the test's subject
            outcome.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    try:
        thread.start()
        assert accepted.wait(5)
        time.sleep(0.4)
        started = time.monotonic()
        backend.abort()
        thread.join(2)
        assert not thread.is_alive() and time.monotonic() - started < 1.0
        assert isinstance(outcome[0], E.Cancelled)
        assert any(isinstance(e, Waiting) for e in events)
    finally:
        stop_server.set()
        server.close()
        backend.close()


def test_the_read_timeouts():
    from flextool.assistant import net

    assert net.read_timeout_for("https://api.mistral.ai/v1") == 300.0
    assert net.read_timeout_for("http://localhost:11434/v1") == 900.0
    assert net.read_timeout_for("http://192.168.1.20:8000/v1") == 900.0


def _silent_server():
    """A loopback server that accepts and never answers; returns (port, accepted, stop, close)."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    accepted = threading.Event()
    stop_server = threading.Event()

    def serve():
        conn, _ = server.accept()
        accepted.set()
        stop_server.wait(10)
        conn.close()

    threading.Thread(target=serve, daemon=True).start()
    return server.getsockname()[1], accepted, stop_server, server


def test_a_slow_first_byte_shows_the_wait_then_times_out(monkeypatch):
    """The model sends nothing: the waiting line counts up until the read timeout (shortened for the test), then
    the message names it."""
    from flextool.assistant import net

    monkeypatch.setattr(net, "READ_TIMEOUT_LOCAL", 0.8)
    port, accepted, stop_server, server = _silent_server()
    backend = make_backend(PRESETS["ollama"], connection("ollama", url=f"http://127.0.0.1:{port}/v1", key=None))
    backend.wait_interval = 0.2
    events, on_event = collect()
    try:
        started = time.monotonic()
        with pytest.raises(E.TimedOut) as info:
            backend.stream(request(ask()), on_event, threading.Event())
        assert accepted.is_set() and 0.7 < time.monotonic() - started < 3.0
        assert info.value.user_message == "Ollama did not answer within 0.8 seconds."
        waits = [e.seconds for e in events if isinstance(e, Waiting)]
        assert len(waits) >= 2 and waits == sorted(waits)           # the waiting line kept counting
    finally:
        stop_server.set()
        server.close()
        backend.close()


def test_stop_still_works_late_in_a_long_wait(monkeypatch):
    """Stop late in the wait for the first byte (here after 0.6 of 0.8 seconds) ends the request at once."""
    from flextool.assistant import net

    monkeypatch.setattr(net, "READ_TIMEOUT_LOCAL", 0.8)
    port, accepted, stop_server, server = _silent_server()
    backend = make_backend(PRESETS["ollama"], connection("ollama", url=f"http://127.0.0.1:{port}/v1", key=None))
    backend.wait_interval = 0.1
    outcome: list = []

    def worker():
        try:
            outcome.append(backend.stream(request(ask()), lambda e: None, threading.Event()))
        except Exception as exc:  # noqa: BLE001 — the outcome is the test's subject
            outcome.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    try:
        thread.start()
        assert accepted.wait(5)
        time.sleep(0.6)
        stopped = time.monotonic()
        backend.abort()
        thread.join(2)
        assert not thread.is_alive() and time.monotonic() - stopped < 0.5
        assert isinstance(outcome[0], E.Cancelled)
    finally:
        stop_server.set()
        server.close()
        backend.close()


def test_replies_are_new_objects_each_time():
    backend, _ = backend_for("openai", [answer("openai", "text_usage"), answer("openai", "text_usage")])
    a = backend.stream(request(ask()), lambda e: None, threading.Event())
    b = backend.stream(request(ask()), lambda e: None, threading.Event())
    assert a.message is not b.message and dataclasses.asdict(a.message) == dataclasses.asdict(b.message)


# --- an answer typed application/json (C4-m4) ---------------------------------------------------------------------
def test_a_200_json_error_is_classified():
    """A server that rejects a streamed request with HTTP 200 and a JSON error body: the provider's error is
    classified (here by its type), not reported as a broken connection; a JSON body without an error is still an
    incomplete answer."""
    backend, _ = backend_for("openai", [json_response(200, {"error": {"type": "authentication_error",
                                                                      "message": "invalid key"}})])
    with pytest.raises(E.AuthFailed):
        backend.stream(request(ask()), lambda e: None, threading.Event())
    backend, _ = backend_for("openai", [json_response(200, {"id": "x"})])
    with pytest.raises(E.Truncated):
        backend.stream(request(ask()), lambda e: None, threading.Event())


def test_an_event_stream_sent_as_json_still_streams():
    """A pin (it held before C4-m4): an event stream that a server or proxy labels ``application/json`` is read as
    events as they arrive, its first bytes included."""
    chunks = payload("openai", "text_usage")["chunks"]
    parts = [b"\n", openai_sse(chunks[:1], done=False), openai_sse(chunks[1:])]
    response = httpx2.Response(200, headers={"content-type": "application/json"},
                               stream=_Parts(parts))
    backend, _ = backend_for("openai", [response])
    events, on_event = collect()
    reply = backend.stream(request(ask()), on_event, threading.Event())
    expected = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices"))
    assert reply.stop == "end" and reply.message.text == expected
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == expected
    assert reply.usage.output == chunks[-1]["usage"]["completion_tokens"]


class _Parts(httpx2.SyncByteStream):
    def __init__(self, parts: list[bytes]) -> None:
        self._parts = parts

    def __iter__(self):
        yield from self._parts


# --- tool-call fragments without an index (C4-m9) -----------------------------------------------------------------
def _call_chunks(fragments: list[dict]) -> bytes:
    chunks = [{"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m",
               "choices": [{"index": 0, "delta": {"tool_calls": [f]}, "finish_reason": None}]} for f in fragments]
    chunks.append({"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m",
                   "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    return openai_sse(chunks)


@pytest.mark.parametrize("fragments,expected", [
    pytest.param([{"id": "a", "type": "function", "function": {"name": "get_entity", "arguments": '{"name":'}},
                  {"function": {"arguments": '"west"}'}}],
                 [("a", "get_entity", {"name": "west"})], id="continuation"),
    pytest.param([{"id": "a", "type": "function", "function": {"name": "get_entity", "arguments": '{"name":'}},
                  {"id": "a", "function": {"arguments": '"west"}'}}],
                 [("a", "get_entity", {"name": "west"})], id="continuation_with_the_id"),
    pytest.param([{"id": "a", "type": "function", "function": {"name": "get_entity", "arguments": '{"name":"west"}'}},
                  {"id": "b", "type": "function", "function": {"name": "search_docs", "arguments": '{"query":"x"}'}}],
                 [("a", "get_entity", {"name": "west"}), ("b", "search_docs", {"query": "x"})], id="two_calls"),
    pytest.param([{"id": "call_0", "type": "function", "function": {"name": "get_entity",
                                                                     "arguments": '{"name":"west"}'}},
                  {"id": "call_0", "type": "function", "function": {"name": "search_docs",
                                                                     "arguments": '{"query":"x"}'}}],
                 [("call_0", "get_entity", {"name": "west"}), ("call_0", "search_docs", {"query": "x"})],
                 id="two_calls_sharing_an_id"),
])
def test_index_less_fragments_continue_the_last_call(fragments, expected):
    """A server that leaves ``index`` out: a fragment without an id or a name continues the last call; one with a
    new id or a name opens a call, and two calls that share an id (some servers reuse one) stay two."""
    backend, _ = backend_for("openai", [sse_response(_call_chunks(fragments))])
    reply = backend.stream(request(ask()), lambda e: None, threading.Event())
    assert reply.stop == "tools"
    assert [(c.id, c.name, c.arguments) for c in reply.message.tool_calls] == expected


# --- Ollama's reasoning (gpt-oss "error parsing tool call", 2026-11-11) -------------------------------------------
def _ollama_chunk(delta: dict, finish: str | None = None) -> dict:
    return {"id": "chatcmpl-001", "object": "chat.completion.chunk", "created": 1794226502, "model": "example-model",
            "system_fingerprint": "fp_ollama", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


OLLAMA_CALL = {"index": 0, "id": "call_reasoned1", "type": "function",
               "function": {"name": "get_entity", "arguments": '{"name":"west"}'}}
#: Ollama's stream: the reasoning in pieces of ``delta.reasoning`` (the content empty), then the call.
REASONED_CALL = [_ollama_chunk({"role": "assistant", "content": "", "reasoning": "We need "}),
                 _ollama_chunk({"content": "", "reasoning": "the units of west."}),
                 _ollama_chunk({"content": "", "tool_calls": [OLLAMA_CALL]}),
                 _ollama_chunk({}, "tool_calls")]
REASONED_TEXT = [_ollama_chunk({"role": "assistant", "content": "", "reasoning": "Two units."}),
                 _ollama_chunk({"content": "West has 2 units."}), _ollama_chunk({}, "stop")]


def test_ollama_reasoning_is_joined_and_never_shown():
    backend, _ = backend_for("ollama", [sse_response(openai_sse(REASONED_CALL))])
    events, on_event = collect()
    reply = backend.stream(request(ask()), on_event, threading.Event())
    assert reply.stop == "tools" and reply.message.text == ""
    assert reply.message.provider_extra == {"reasoning_text": "We need the units of west."}
    assert not [e for e in events if isinstance(e, TextDelta)]
    assert [(e.index, e.name) for e in events if isinstance(e, ToolCallStarted)] == [(0, "get_entity")]


def test_ollama_reasoning_of_a_whole_answer_is_kept():
    body = {"id": "chatcmpl-002", "object": "chat.completion", "created": 1794226502, "model": "example-model",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "", "reasoning": "Look up west.",
                "tool_calls": [{k: v for k, v in OLLAMA_CALL.items() if k != "index"}]}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 15, "total_tokens": 915}}
    recorder = Recorder([json_response(200, body)])
    conn = dataclasses.replace(connection("ollama"), stream=False)
    backend = make_backend(PRESETS["ollama"], conn, transport=recorder.transport)
    reply = backend.stream(request(ask()), lambda e: None, threading.Event())
    assert reply.stop == "tools" and reply.message.provider_extra == {"reasoning_text": "Look up west."}


def test_ollama_reasoning_is_replayed_only_in_the_tool_round_of_the_same_model():
    """Sent back as ``reasoning`` on the assistant messages after the last user message (the tool round), to the
    same model and endpoint only; once the next question is asked, not at all."""
    backend, recorder = backend_for("ollama", [sse_response(openai_sse(REASONED_CALL)),
                                               sse_response(openai_sse(REASONED_TEXT)),
                                               sse_response(openai_sse(REASONED_TEXT))])
    first = backend.stream(request(ask()), lambda e: None, threading.Event())
    result = Message("tool", '{"units": 2}', tool_call_id="call_reasoned1", tool_name="get_entity")
    round_two = [Message("user", "How many units has west?"), first.message, result]
    final = backend.stream(request(round_two), lambda e: None, threading.Event())
    assert final.message.provider_extra == {"reasoning_text": "Two units."}
    sent = recorder.last.body["messages"]
    assert sent[2] == {"role": "assistant", "content": None, "reasoning": "We need the units of west.",
                       "tool_calls": [{"id": "call_reasoned1", "type": "function",
                                       "function": {"name": "get_entity", "arguments": '{"name":"west"}'}}]}
    assert sent[3] == {"role": "tool", "tool_call_id": "call_reasoned1", "content": '{"units": 2}'}
    history_check.validate_openai_body(recorder.last.body)
    # The next question: both earlier answers are before the last user message.
    backend.stream(request([*round_two, final.message, Message("user", "And east?")]), lambda e: None,
                   threading.Event())
    sent = recorder.last.body["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert not any("reasoning" in m for m in sent)
    # Another model of the same server, and the same model on another server, never get it.
    for kw in ({"model": "another-model"}, {"url": "http://192.0.2.1:11434/v1"}):
        other, recorder = backend_for("ollama", [sse_response(openai_sse(REASONED_TEXT))], **kw)
        other.stream(request(round_two), lambda e: None, threading.Event())
        assert "reasoning" not in recorder.last.body["messages"][2], kw


def test_a_preset_without_the_reasoning_key_keeps_none_of_it():
    """LM Studio has no reasoning key: a ``reasoning`` field in its stream is neither kept nor sent back."""
    backend, recorder = backend_for("lmstudio", [sse_response(openai_sse(REASONED_CALL)),
                                                 sse_response(openai_sse(REASONED_TEXT))])
    first = backend.stream(request(ask()), lambda e: None, threading.Event())
    assert first.message.provider_extra == {}
    result = Message("tool", "{}", tool_call_id="call_reasoned1", tool_name="get_entity")
    backend.stream(request([Message("user", "q"), first.message, result]), lambda e: None, threading.Event())
    assert "reasoning" not in recorder.last.body["messages"][2]
