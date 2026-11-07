"""Mock HTTP for the backend tests: synthetic server-sent event streams, a handler that records every request,
connections to each preset, and the synthetic payloads under ``payloads/<preset>/``."""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import httpx2

from flextool.assistant.llm_adapter import Connection, Message, Request, ToolSpec
from flextool.assistant.net import route_for
from flextool.assistant.providers import PRESETS

PAYLOADS = Path(__file__).resolve().parent / "payloads"
KEY = "sk-test-key-0123456789"
LOCAL_URLS = {"ollama": "http://localhost:11434/v1", "lmstudio": "http://localhost:1234/v1"}
GENERIC_URL = "https://llm.example.test/v1"


def payload(preset: str, name: str) -> dict:
    return json.loads((PAYLOADS / preset / f"{name}.json").read_text(encoding="utf-8"))


def openai_sse(chunks: Iterable[dict], *, done: bool = True) -> bytes:
    text = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (text + ("data: [DONE]\n\n" if done else "")).encode()


def anthropic_sse(events: Iterable[dict]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def sse_response(content: bytes | Iterator[bytes]) -> httpx2.Response:
    headers = {"content-type": "text/event-stream"}
    if isinstance(content, bytes):
        return httpx2.Response(200, headers=headers, content=content)
    return httpx2.Response(200, headers=headers, stream=_IterStream(content))


class _IterStream(httpx2.SyncByteStream):
    def __init__(self, parts: Iterator[bytes]) -> None:
        self._parts = parts

    def __iter__(self) -> Iterator[bytes]:
        yield from self._parts


def json_response(status: int, body: object, headers: dict | None = None) -> httpx2.Response:
    return httpx2.Response(status, headers={"content-type": "application/json", **(headers or {})},
                           content=json.dumps(body).encode())


@dataclass
class Captured:
    method: str
    url: str
    headers: dict[str, str]
    body: dict | None


@dataclass
class Recorder:
    """A ``MockTransport`` handler: records each request and answers with the next response (a
    ``httpx2.Response`` or a callable taking the request)."""

    responses: list[object] = field(default_factory=list)
    requests: list[Captured] = field(default_factory=list)

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        raw = request.content
        body = json.loads(raw) if raw else None
        self.requests.append(Captured(request.method, str(request.url),
                                      {k.lower(): v for k, v in request.headers.items()}, body))
        if not self.responses:
            raise AssertionError(f"no response left for {request.method} {request.url}")
        response = self.responses.pop(0)
        if callable(response) and not isinstance(response, httpx2.Response):
            response = response(request)
        return response

    @property
    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self)

    @property
    def last(self) -> Captured:
        return self.requests[-1]


def base_url(preset_id: str) -> str:
    return PRESETS[preset_id].base_url or GENERIC_URL


def connection(preset_id: str, *, model: str = "example-model", key: str | None = KEY,
               url: str | None = None, knobs: dict | None = None) -> Connection:
    preset = PRESETS[preset_id]
    address = url or LOCAL_URLS.get(preset_id) or base_url(preset_id)
    return Connection(base_url=address, model=model, api_key=key or "", key_source="system keyring",
                      knobs=dict(knobs or {}), route=route_for(address, preset.label))


TOOLS = (
    ToolSpec("get_entity", "Get one entity.", {"type": "object", "properties": {"name": {"type": "string"}},
                                                "additionalProperties": False}),
    ToolSpec("search_docs", "Search the documentation.",
             {"type": "object", "properties": {"query": {"type": "string"}}, "additionalProperties": False}),
)


def request(messages: list[Message], *, tools=TOOLS, tool_choice="auto", prefix_key: str = "",
            system: tuple[str, ...] = ("You are the assistant.", "Layer two.")) -> Request:
    return Request(system=system, tools=tools, messages=tuple(messages), max_output_tokens=1000,
                   tool_choice=tool_choice, prefix_key=prefix_key)


def collect() -> tuple[list, Callable[[object], None]]:
    events: list = []
    return events, events.append
