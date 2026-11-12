"""A provider error that arrives as a plain JSON body instead of server-sent events (calibration 2026-11-10).

Google's OpenAI-compatible endpoint, streaming with tools, sometimes sends a thought event (Gemma's reasoning as a
``<thought>`` block in the content, marked ``extra_content.google.thought``) and then the error as a JSON array
spread over several lines, or answers HTTP 500 with that array, or ends the events after the thought with no finish
reason.  All are server errors and retried, since nothing of the answer was shown; none is "the connection ended".
The chunks have the shape of the endpoint's own (captured 2026-11-10).  The server is a real one on the loopback
interface; the connection is treated as a remote one, so the remote retry policy applies.
"""
from __future__ import annotations

import dataclasses
import json
import socketserver
import threading

import pytest

from flextool.assistant import llm_errors as E
from flextool.assistant.llm_adapter import RetryWait, TextDelta, make_backend
from flextool.assistant.llm_retry import RetryPolicy, call_with_retries
from flextool.assistant.providers import PRESETS
from flextool.assistant.thoughts import stream_visible
from tests.assistant.llm.wire import collect, connection, openai_sse, request

GOOGLE_ERROR = [{"error": {"code": 500, "message": "Internal error encountered.", "status": "INTERNAL"}}]
#: The array as Google lays it out: over several lines, no ``data:`` prefix.
ERROR_LINES = json.dumps(GOOGLE_ERROR, indent=2).encode()


def _chunk(delta: dict, finish: str | None = None, usage: dict | None = None) -> dict:
    choice: dict = {"delta": {**delta, "role": "assistant"}, "index": 0}
    if finish is not None:
        choice["finish_reason"] = finish
    return {"choices": [choice], "created": 1794301738, "id": "Ht_yaoHOOJG3nsEP2srYyQw", "model": "gemma-4-31b-it",
            "object": "chat.completion.chunk",
            "usage": usage or {"completion_tokens": 0, "prompt_tokens": 40, "total_tokens": 54}}


GOOGLE_THOUGHT = {"google": {"thought": True}}
THOUGHT = _chunk({"content": "<thought>The user wants the units of west.\nI should use",
                  "extra_content": GOOGLE_THOUGHT})
THOUGHT_MORE = _chunk({"content": " the `get_entity` tool.", "extra_content": GOOGLE_THOUGHT})
SUCCESS = openai_sse([THOUGHT, _chunk({"content": "</thought>West has two units."}),
                      _chunk({}, "stop", {"completion_tokens": 6, "prompt_tokens": 40, "total_tokens": 80})])


def http(status: int, body: bytes, content_type: str = "text/event-stream") -> bytes:
    """A whole HTTP answer that ends by closing the connection (no length: the body is read to the end)."""
    return (f"HTTP/1.1 {status} X\r\nContent-Type: {content_type}\r\nConnection: close\r\n\r\n".encode("latin-1")
            + body)


class Script:
    """A loopback server that answers its *answers* in turn, one per connection."""

    def __init__(self, answers: list[bytes]) -> None:
        self.answers = list(answers)
        self.count = 0
        script = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                length = 0
                while (line := self.rfile.readline().strip()):
                    name, _, value = line.decode("latin-1").partition(":")
                    if name.strip().lower() == "content-length":
                        length = int(value)
                self.rfile.read(length)
                script.count += 1
                self.wfile.write(script.answers.pop(0))

        self.server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def serve():
    servers: list[Script] = []

    def start(answers: list[bytes]):
        script = Script(answers)
        servers.append(script)
        backend = make_backend(PRESETS["gemini"], connection("gemini", model="gemma-4-31b-it", url=script.url))
        backend.ctx = dataclasses.replace(backend.ctx, local=False)     # Google's remote endpoint
        return script, backend

    yield start
    for script in servers:
        script.close()


def ask(backend):
    events, on_event = collect()
    now = [0.0]                     # a fake clock: the waits pass at once

    def sleep(seconds: float) -> bool:
        now[0] += seconds
        return False

    reply = call_with_retries(lambda: stream_visible(backend, request([]), on_event, threading.Event()),
                              RetryPolicy(jitter=0.0), on_event, threading.Event(), sleep=sleep,
                              clock=lambda: now[0])
    return reply, events


def _retried_to_success(serve, failures: list[bytes]) -> None:
    script, backend = serve([*failures, http(200, SUCCESS)])
    try:
        reply, events = ask(backend)
    finally:
        backend.close()
    assert script.count == len(failures) + 1
    assert reply.stop == "end" and reply.message.text == "West has two units."
    waits = [e for e in events if isinstance(e, RetryWait)]
    assert {(w.attempt, w.of) for w in waits} == {(n, 3) for n in range(1, len(failures) + 1)}
    assert all("overloaded" in w.reason for w in waits)
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "West has two units."


def test_a_json_error_after_a_thought_event_is_a_server_error_retried(serve):
    _retried_to_success(serve, [http(200, openai_sse([THOUGHT], done=False) + ERROR_LINES),
                                http(200, openai_sse([THOUGHT, THOUGHT_MORE], done=False) + ERROR_LINES)])


def test_a_500_with_the_json_error_is_a_server_error_retried(serve):
    _retried_to_success(serve, [http(500, ERROR_LINES, "application/json")])


def test_the_json_error_is_classified_from_its_code(serve):
    """A 429 in the body is a rate limit (its retry delay named), not a broken connection."""
    limited = json.dumps([{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "slow down",
                                     "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                                                  "retryDelay": "7s"}]}}], indent=2).encode()
    _script, backend = serve([http(200, openai_sse([THOUGHT], done=False) + limited)])
    try:
        with pytest.raises(E.RateLimited) as info:
            stream_visible(backend, request([]), lambda e: None, threading.Event())
    finally:
        backend.close()
    assert info.value.retry_after == 7.0 and not info.value.streamed


def test_events_that_end_after_the_thought_are_retried(serve):
    """``[DONE]`` after only reasoning and no finish reason: the provider ended the answer before any of it came;
    retried, with its own words."""
    script, backend = serve([http(200, openai_sse([THOUGHT, THOUGHT_MORE])),
                             http(200, openai_sse([THOUGHT])),
                             http(200, SUCCESS)])
    try:
        reply, events = ask(backend)
    finally:
        backend.close()
    assert script.count == 3 and reply.message.text == "West has two units."
    waits = [e for e in events if isinstance(e, RetryWait)]
    assert waits and all("ended the answer before any of it came" in w.reason for w in waits)
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "West has two units."


def test_a_stream_that_never_ends_after_the_thought_says_so(serve):
    _script, backend = serve([http(200, openai_sse([THOUGHT]))] * 4)
    try:
        with pytest.raises(E.ServerBusy) as info:
            ask(backend)
    finally:
        backend.close()
    assert info.value.message_key == "ended_early" and not info.value.streamed and info.value.partial_text == ""


def test_events_that_end_after_a_tool_call_are_still_cut_off(serve):
    """A tool call came: the answer had started, so the end without a finish reason stays a cut-off answer."""
    call = _chunk({"content": "</thought>", "tool_calls": [{
        "extra_content": {"google": {"thought_signature": "EiYK"}}, "function": {"arguments": "{}", "name":
        "get_entity"}, "id": "call_5305835037512728217", "type": "function"}]},
        usage={"completion_tokens": 5, "prompt_tokens": 40, "total_tokens": 79})
    script, backend = serve([http(200, openai_sse([THOUGHT, call])), http(200, SUCCESS)])
    try:
        with pytest.raises(E.Truncated):
            ask(backend)
    finally:
        backend.close()
    assert script.count == 1


def test_after_visible_text_the_error_is_not_retried(serve):
    """Part of the answer was shown: the error ends the answer as before (the overload message, not retried)."""
    shown = _chunk({"content": "</thought>West has"})
    script, backend = serve([http(200, openai_sse([THOUGHT, shown], done=False) + ERROR_LINES),
                             http(200, SUCCESS)])
    try:
        with pytest.raises(E.ServerBusy) as info:
            ask(backend)
    finally:
        backend.close()
    assert script.count == 1
    assert info.value.streamed and info.value.partial_text == "West has"
    assert info.value.message_key == "busy_streamed"


# --- Ollama: a tool call it could not read (gpt-oss, eval cell G04, 2026-11-11) -----------------------------------
OLLAMA_UNREADABLE = json.dumps({"error": {
    "message": "error parsing tool call: raw='We need latest successful run folder. Probably list_runs "
               "{\"state\":\"succeeded\"}', err=invalid character 'W' looking for beginning of value",
    "type": "api_error", "param": None}}).encode()


def _ollama_chunk(delta: dict, finish: str | None = None) -> dict:
    return {"id": "chatcmpl-001", "object": "chat.completion.chunk", "created": 1794226502, "model": "gpt-oss:20b",
            "system_fingerprint": "fp_ollama", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


REASONING = _ollama_chunk({"role": "assistant", "content": "", "reasoning": "We need the latest run."})
GOOD_CALL = openai_sse([REASONING, _ollama_chunk({"content": "", "tool_calls": [{
    "index": 0, "id": "call_good1", "type": "function", "function": {"name": "get_entity",
                                                                     "arguments": "{\"name\":\"west\"}"}}]}),
    _ollama_chunk({}, "tool_calls")])


@pytest.fixture
def ollama_serve():
    servers: list[Script] = []

    def start(answers: list[bytes]):
        script = Script(answers)
        servers.append(script)
        return script, make_backend(PRESETS["ollama"], connection("ollama", model="gpt-oss:20b", url=script.url))

    yield start
    for script in servers:
        script.close()


@pytest.mark.parametrize("failure", [
    http(500, OLLAMA_UNREADABLE, "application/json"),
    # inside the stream, after reasoning only (never shown, so nothing of the answer was shown)
    http(200, openai_sse([REASONING], done=False) + b"data: " + OLLAMA_UNREADABLE + b"\n\n"),
], ids=["status 500", "in the stream"])
def test_an_unreadable_tool_call_is_asked_again_at_once(ollama_serve, failure):
    """Ollama's "error parsing tool call", as a 500 or inside the stream, is asked again half a second later (not
    the local server's busy backoff), and the next answer's call goes through."""
    script, backend = ollama_serve([failure, http(200, GOOD_CALL)])
    try:
        reply, events = ask(backend)
    finally:
        backend.close()
    assert script.count == 2
    assert reply.stop == "tools" and [c.name for c in reply.message.tool_calls] == ["get_entity"]
    waits = [e for e in events if isinstance(e, RetryWait)]
    assert [(w.seconds_left, w.attempt, w.of) for w in waits] == [(1, 1, 2)]
    assert waits[0].reason == ("Ollama could not read a tool call in the model's answer. Asking again (attempt 1 "
                               "of 2).")
    assert not [e for e in events if isinstance(e, TextDelta)]


def test_three_unreadable_tool_calls_end_the_answer(ollama_serve):
    script, backend = ollama_serve([http(500, OLLAMA_UNREADABLE, "application/json")] * 3)
    try:
        with pytest.raises(E.UnreadableToolCall) as info:
            ask(backend)
    finally:
        backend.close()
    assert script.count == 3 and info.value.message_key == "tool_call_unreadable_final"
    assert "loading a model" not in info.value.user_message
    assert info.value.provider_said.message.startswith("error parsing tool call: raw=")
