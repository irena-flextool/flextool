"""The neutral message and tool model every AI service backend translates to and from (P1 §2).

A conversation is a list of :class:`Message`; a model request is a :class:`Request`; a backend answers with a
:class:`Reply` or raises an :class:`~flextool.assistant.llm_errors.LLMError`, and pushes events (:class:`TextDelta`,
:class:`ToolCallStarted`, …) to the front end while the answer streams.  Data that only one provider understands
(Anthropic's thinking blocks, Gemini's thought signatures, OpenRouter's reasoning details) travels in
``provider_extra``, stamped with the :data:`Origin` that produced it, and is sent back only to the same origin
(:func:`remap_ids`, :func:`drop_provider_extra`).

Only the standard library is imported here; the provider libraries are imported by :func:`make_backend` when a
backend is built.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import secrets
import string
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from flextool.assistant.providers import Preset

#: (preset id, base URL, model): who produced a message's provider data.
Origin = tuple[str, str, str]

Role = Literal["user", "assistant", "tool"]
Stop = Literal["end", "tools", "length", "refused", "context"]
EndpointKind = Literal["loopback", "local_network", "remote"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict | None          # None when raw_arguments is not a JSON object
    raw_arguments: str              # exactly as streamed; replayed byte-identical to the same origin
    provider_extra: dict = field(default_factory=dict)      # e.g. {"extra_content": {"google": {...}}}


@dataclass
class Message:
    role: Role
    text: str = ""                  # user text / assistant visible text / tool result JSON
    tool_calls: list[ToolCall] = field(default_factory=list)                  # assistant only
    tool_call_id: str | None = None                                           # tool only
    tool_name: str | None = None
    is_error: bool = False
    provider_extra: dict = field(default_factory=dict)  # {"anthropic_blocks": [...]}, {"reasoning_details": [...]}
    origin: Origin | None = None
    turn: int = 0
    compacted: bool = False
    kind: str = ""                  # "" or one of MESSAGE_KINDS: the session's own messages


#: The message data an Anthropic answer keeps: its content blocks, replayed *instead of* the text and the tool calls
#: to the same origin (:mod:`flextool.assistant.backends.anthropic_messages`).
ANTHROPIC_BLOCKS_KEY = "anthropic_blocks"

#: The kinds of the session's own messages (P3 §4.7): the request for the model's note, the note, the first message
#: of a fresh context, and the "tool limit reached" instruction before the tool-less request.  Backends send them like
#: any other message.
MESSAGE_KINDS = ("note_request", "note", "continuation", "limit")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict


@dataclass(frozen=True)
class Request:
    system: tuple[str, ...]         # prompt layers; byte-stable for a conversation
    tools: tuple[ToolSpec, ...]     # sorted by name; fixed for a conversation
    messages: tuple[Message, ...]
    max_output_tokens: int
    tool_choice: Literal["auto", "none"] = "auto"
    prefix_key: str = ""            # see :func:`prefix_key`
    cache: bool = True              # False: no prompt-cache marks (a one-off prompt pays no cache write)


@dataclass
class Usage:
    input_uncached: int | None
    cache_read: int | None
    cache_write: int | None
    output: int | None
    cost: float | None = None       # only when the response carries one (OpenRouter usage.cost)
    reported: bool = True           # False: "not reported", never estimated


def not_reported() -> Usage:
    """The usage of a request whose answer carried no token counts."""
    return Usage(None, None, None, None, reported=False)


@dataclass
class Reply:
    message: Message | None         # None when the model returned nothing (no text, no calls): never appended
    stop: Stop
    refusal_reason: str | None
    usage: Usage


@dataclass(frozen=True)
class ModelEntry:
    """One model of a provider's live model list."""

    id: str
    note: str | None = None             # e.g. "runs on Ollama cloud"
    context_window: int | None = None


@dataclass(frozen=True)
class Route:
    """Where the data of a connection goes (P1 §9.1): the endpoint kind (for a model a local server forwards,
    ``remote``), the endpoint host, the proxy host for a remote endpoint, the host a local server forwards to,
    and the label that names the service in every message.  *server_kind* is the kind of the endpoint itself (for a
    forwarded model, where the forwarding server runs); *encrypted* is ``False`` for a plain ``http://`` address
    outside this computer and the local network (:func:`flextool.assistant.net.plain_http_remote`)."""

    kind: EndpointKind
    host: str
    proxy_host: str | None
    forwarded_to: str | None
    label: str
    server_kind: EndpointKind = "remote"
    encrypted: bool = True


@dataclass(frozen=True)
class Connection:
    base_url: str
    model: str
    api_key: str = field(repr=False)
    key_source: str
    knobs: Mapping[str, object]
    route: Route


# --- events pushed to the front end (thread-safe callback) -------------------------------------------------------
@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class Thinking:
    pass


@dataclass(frozen=True)
class ToolCallStarted:
    index: int
    name: str


@dataclass(frozen=True)
class Waiting:
    seconds: int


@dataclass(frozen=True)
class RetryWait:
    seconds_left: int
    attempt: int
    of: int
    reason: str                     # the plain-English line to show, with seconds_left filled in


@dataclass(frozen=True)
class ToolRunning:
    name: str
    args_preview: str


@dataclass(frozen=True)
class ToolDone:
    name: str
    ok: bool
    summary: str
    seconds: float
    arguments: str = ""
    result_preview: str = ""


@dataclass(frozen=True)
class ToolWaiting:
    name: str


@dataclass(frozen=True)
class UsageUpdate:
    usage: Usage


@dataclass(frozen=True)
class Notice:
    text: str


@dataclass(frozen=True)
class Continuing:
    """A question goes on with a fresh context: *part* of at most *of* (P3 §4.7)."""

    part: int
    of: int


Event = (TextDelta | Thinking | ToolCallStarted | Waiting | RetryWait | ToolRunning | ToolDone | ToolWaiting
         | UsageUpdate | Notice | Continuing)
OnEvent = Callable[[Event], None]


class Backend(Protocol):
    origin: Origin

    def stream(self, req: Request, on_event: OnEvent, cancel: threading.Event) -> Reply: ...

    def abort(self) -> None: ...            # closes the in-flight response; callable from any thread

    def list_models(self) -> list[ModelEntry]: ...

    def close(self) -> None: ...


# --- tool-call ids and provider data -----------------------------------------------------------------------------
_ID_ALPHABET = string.ascii_letters + string.digits
ID_LENGTH = 9


def new_tool_call_id() -> str:
    """A generated tool-call id: 9 characters ``[A-Za-z0-9]``, the strictest format a provider asks for."""
    return "".join(secrets.choice(_ID_ALPHABET) for _ in range(ID_LENGTH))


def drop_provider_extra(messages: list[Message] | tuple[Message, ...]) -> list[Message]:
    """Copies of *messages* without any provider data (thinking blocks, signatures, reasoning details)."""
    out = []
    for m in messages:
        calls = [dataclasses.replace(c, provider_extra={}) for c in m.tool_calls]
        out.append(dataclasses.replace(m, tool_calls=calls, provider_extra={}))
    return out


def remap_ids(messages: list[Message] | tuple[Message, ...]) -> list[Message]:
    """Copies of *messages* for another origin: every tool-call id replaced by a generated one (the same for a call
    and its result) and all provider data dropped.  A result whose call is not in *messages* gets its own id."""
    mapping: dict[str, str] = {}

    def mapped(old: str | None) -> str:
        if old is None or old not in mapping:
            new = new_tool_call_id()
            while new in mapping.values():
                new = new_tool_call_id()
            if old is None:
                return new
            mapping[old] = new
        return mapping[old]

    out = []
    for m in drop_provider_extra(messages):
        if m.role == "assistant" and m.tool_calls:
            m = dataclasses.replace(m, tool_calls=[dataclasses.replace(c, id=mapped(c.id)) for c in m.tool_calls])
        elif m.role == "tool":
            m = dataclasses.replace(m, tool_call_id=mapped(m.tool_call_id))
        out.append(m)
    return out


def prefix_key(system: tuple[str, ...], tools: tuple[ToolSpec, ...]) -> str:
    """``sha256(system ‖ tools)[:16]``: the same for every conversation with the same instructions and tools."""
    payload = json.dumps({"system": list(system),
                          "tools": [[t.name, t.description, t.input_schema] for t in tools]},
                         sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def parse_arguments(raw: str) -> dict | None:
    """The arguments of a tool call: the JSON object in *raw* (an empty text is ``{}``), else ``None``."""
    if not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


# --- the silence timer --------------------------------------------------------------------------------------------
class SilenceTimer:
    """Emits :class:`Waiting` every *interval* seconds while nothing arrived from the endpoint.

    :meth:`touch` is called whenever a part of the answer arrives; the seconds in the event count from the last
    one (or from :meth:`start`).  A daemon thread, stopped by :meth:`stop`.
    """

    def __init__(self, on_event: OnEvent, *, interval: float = 5.0, clock: Callable[[], float] = time.monotonic):
        self._on_event = on_event
        self._interval = interval
        self._clock = clock
        self._last = clock()
        self._stopped = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> SilenceTimer:
        self._last = self._clock()
        self._thread = threading.Thread(target=self._run, name="flextool-assistant-wait", daemon=True)
        self._thread.start()
        return self

    def touch(self) -> None:
        with self._lock:
            self._last = self._clock()

    def stop(self) -> None:
        self._stopped.set()

    def _run(self) -> None:
        while not self._stopped.wait(self._interval):
            with self._lock:
                silent = self._clock() - self._last
            if silent >= self._interval and not self._stopped.is_set():
                self._on_event(Waiting(int(silent)))


# --- the provider libraries' loggers ---------------------------------------------------------------------------
#: Loggers of the provider libraries and their HTTP stack; their debug output holds whole request bodies.
SDK_LOGGERS = ("anthropic", "openai", "httpx2", "httpcore2")


def quiet_sdk_loggers(*, verbose: bool = False) -> None:
    """Cap the provider libraries' loggers at WARNING (INFO with *verbose*), never DEBUG, whatever their own
    logging variables say."""
    level = logging.INFO if verbose else logging.WARNING
    for name in SDK_LOGGERS:
        logging.getLogger(name).setLevel(level)


def make_backend(preset: Preset, conn: Connection, *, transport=None, verbose: bool = False) -> Backend:
    """The backend of *preset* for *conn*.  *transport* (an ``httpx2`` transport) replaces the network in tests."""
    if preset.backend == "openai":
        from flextool.assistant.backends.openai_chat import OpenAIChatBackend

        backend: Backend = OpenAIChatBackend(preset, conn, transport=transport)
    elif preset.backend == "anthropic":
        from flextool.assistant.backends.anthropic_messages import (
            AnthropicMessagesBackend,
        )

        backend = AnthropicMessagesBackend(preset, conn, transport=transport)
    else:
        raise ValueError(f"unknown backend {preset.backend!r}")
    quiet_sdk_loggers(verbose=verbose)
    return backend
