"""The Chat Completions backend (P1 §3.1): OpenAI, Mistral, OpenRouter, Gemini, Ollama, LM Studio and any other
OpenAI-compatible base URL.

The request carries only the fields the preset allows; the knobs' fields go in ``extra_body`` (they land at the
top level of the body).  The answer streams: text deltas are shown as they come, tool calls are assembled per
``index`` with their arguments concatenated, provider data the preset captures (Gemini's thought signatures in
``extra_content``, OpenRouter's ``reasoning_details``) is merged and replayed only to the same origin.
"""
from __future__ import annotations

import contextlib
import copy
import threading
from dataclasses import dataclass, field
from typing import Any

from flextool.assistant.backends._common import (
    InFlight,
    api_key_for,
    deep_merge,
    error_context,
)
from flextool.assistant.llm_adapter import (
    Connection,
    Message,
    ModelEntry,
    Notice,
    OnEvent,
    Reply,
    Request,
    SilenceTimer,
    Stop,
    TextDelta,
    ToolCall,
    ToolCallStarted,
    Usage,
    new_tool_call_id,
    not_reported,
    parse_arguments,
)
from flextool.assistant.llm_errors import (
    LLMError,
    Truncated,
    cancelled,
    classify_openai,
    classify_status,
    make_error,
    render,
)
from flextool.assistant.net import build_http_client, key_header_value
from flextool.assistant.providers import Preset, model_entries, request_fields

#: Request fields the library takes as named arguments; any other field goes in ``extra_body``.
_NAMED_FIELDS = frozenset({"store", "stream_options", "tool_choice", "parallel_tool_calls", "prompt_cache_key",
                           "max_tokens", "max_completion_tokens", "tools"})
#: The fields every request has, whatever the preset allows.
CORE_FIELDS = frozenset({"model", "messages", "stream"})
_TOOL_CALL_FIELDS = frozenset({"index", "id", "type", "function"})
#: Reasoning detail fields that arrive in pieces and are concatenated.
_CONCATENATED = ("text", "summary")


def error_prefix(text: str) -> str:
    return text if text.startswith("Error: ") else f"Error: {text}"


@dataclass
class _Call:
    index: int
    id: str | None = None
    name: str = ""
    raw: str = ""
    extra: dict = field(default_factory=dict)
    started: bool = False


@dataclass
class _State:
    text: str = ""
    refusal: bool = False
    streamed: bool = False
    finish: str | None = None
    error_chunk: dict | None = None
    usage: dict | None = None
    calls: dict[int, _Call] = field(default_factory=dict)
    message_extra: dict[str, Any] = field(default_factory=dict)
    reasoning: dict[str, list] = field(default_factory=dict)


def merge_reasoning(items: list, fragments: list) -> None:
    """Merge streamed ``reasoning_details`` *fragments* into *items*: per ``index``, text and summary
    concatenated, every other field taken from the last fragment that has it; a fragment without an index is an
    item of its own."""
    for fragment in fragments:
        if not isinstance(fragment, dict):
            continue
        index = fragment.get("index")
        existing = next((i for i in items if index is not None and i.get("index") == index), None)
        if existing is None:
            items.append(copy.deepcopy(fragment))
            continue
        for key, value in fragment.items():
            if key in _CONCATENATED and isinstance(value, str):
                existing[key] = (existing.get(key) or "") + value
            elif value is not None:
                existing[key] = copy.deepcopy(value)


class OpenAIChatBackend:
    def __init__(self, preset: Preset, conn: Connection, *, transport: Any = None) -> None:
        import openai

        self.preset = preset
        self.conn = conn
        self.origin = (preset.id, conn.base_url, conn.model)
        self.ctx = error_context(preset, conn)
        key = api_key_for(preset, conn, self.ctx)
        self._http = build_http_client(conn, backend="openai", api_key=key, required_headers=preset.required_headers,
                                       transport=transport)
        # The key is also passed as an explicit header: the library then drops an Authorization header that its
        # custom-headers environment variable carries, instead of letting it replace the key.
        self._client = openai.OpenAI(api_key=key, base_url=conn.base_url, max_retries=0,
                                     http_client=self._http.client,
                                     default_headers={"Authorization": key_header_value(key)})
        self._inflight = InFlight()
        self.wait_interval = 5.0            # seconds of silence between two Waiting events

    # --- the request ----------------------------------------------------------------------------------------------
    def wire_messages(self, req: Request) -> list[dict]:
        out: list[dict] = []
        if req.system:
            out.append({"role": "system", "content": "\n\n".join(req.system)})
        for m in req.messages:
            same = m.origin == self.origin
            if m.role == "user":
                out.append({"role": "user", "content": m.text})
            elif m.role == "assistant":
                wire: dict[str, Any] = {"role": "assistant", "content": m.text or None}
                if m.tool_calls:
                    wire["tool_calls"] = [self._wire_call(c, same) for c in m.tool_calls]
                if same:
                    for key in sorted(self.preset.capture_message_keys):
                        if key in m.provider_extra:
                            wire[key] = copy.deepcopy(m.provider_extra[key])
                out.append(wire)
            else:
                wire = {"role": "tool", "tool_call_id": m.tool_call_id,
                        "content": error_prefix(m.text) if m.is_error else m.text}
                if self.preset.tool_message_name:
                    wire["name"] = m.tool_name
                out.append(wire)
        return out

    def _wire_call(self, call: ToolCall, same: bool) -> dict:
        wire: dict[str, Any] = {"id": call.id, "type": "function",
                                "function": {"name": call.name, "arguments": call.raw_arguments or "{}"}}
        if same:
            for key in sorted(self.preset.capture_tool_call_keys):
                if key in call.provider_extra:
                    wire[key] = copy.deepcopy(call.provider_extra[key])
        return wire

    def request_body(self, req: Request) -> dict[str, Any]:
        """The whole request body (what reaches the endpoint)."""
        allowed = self.preset.allowed_params
        body: dict[str, Any] = {"model": self.conn.model, "messages": self.wire_messages(req), "stream": True}
        send_tools = bool(req.tools) and not (req.tool_choice == "none" and "tool_choice" not in allowed)
        if send_tools:
            body["tools"] = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                               "parameters": t.input_schema}} for t in req.tools]
            if "tool_choice" in allowed:
                body["tool_choice"] = req.tool_choice
        body[self.preset.max_tokens_param] = req.max_output_tokens
        if "stream_options" in allowed:
            body["stream_options"] = {"include_usage": True}
        if "prompt_cache_key" in allowed and req.prefix_key:
            body["prompt_cache_key"] = req.prefix_key
        fields = request_fields(self.preset, self.conn.knobs)
        if self.preset.require_parameters_with_tools and send_tools:
            fields.setdefault("provider", {})["require_parameters"] = True
        for name, value in fields.items():
            if name not in allowed:
                raise ValueError(f"the preset {self.preset.id} does not allow the field {name}")
            body[name] = value
        extra = set(body) - CORE_FIELDS - {"tools", self.preset.max_tokens_param} - allowed
        if extra:
            raise ValueError(f"fields outside the preset's allowed ones: {sorted(extra)}")
        return body

    # --- the answer -----------------------------------------------------------------------------------------------
    def stream(self, req: Request, on_event: OnEvent, cancel: threading.Event) -> Reply:
        body = self.request_body(req)
        named = {k: v for k, v in body.items() if k in _NAMED_FIELDS}
        extra_body = {k: v for k, v in body.items() if k not in _NAMED_FIELDS and k not in CORE_FIELDS}
        state = _State()
        self._inflight.begin()
        self._http.begin_request()
        if cancel.is_set():         # a Stop from before the request started
            raise cancelled(self.ctx)
        timer = SilenceTimer(on_event, interval=self.wait_interval).start()
        stream = None
        try:
            try:
                stream = self._client.chat.completions.create(
                    model=body["model"], messages=body["messages"], stream=True, extra_body=extra_body or None,
                    **named)
            except Exception as exc:
                if cancel.is_set() or self._inflight.aborted.is_set():
                    raise cancelled(self.ctx) from exc
                raise classify_openai(exc, self.ctx) from exc
            self._inflight.set(stream.response)
            try:
                for chunk in stream:
                    timer.touch()
                    if cancel.is_set() or self._inflight.aborted.is_set():
                        raise self._cancelled(state)
                    self._consume(chunk.to_dict(), state, on_event)
            except LLMError:
                raise
            except Exception as exc:
                if cancel.is_set() or self._inflight.aborted.is_set():
                    raise self._cancelled(state) from exc
                raise classify_openai(exc, self.ctx, streamed=state.streamed, partial_text=state.text,
                                      partial_usage=self._usage(state)) from exc
            if cancel.is_set() or self._inflight.aborted.is_set():
                raise self._cancelled(state)
        finally:
            timer.stop()
            self._inflight.clear()
            if stream is not None:
                # After a Stop the response may already be closed from the other thread, and a second close can
                # fail; without a Stop a close error surfaces.
                with contextlib.suppress(Exception) if self._inflight.aborted.is_set() else contextlib.nullcontext():
                    stream.close()
        return self._reply(state, on_event)

    def _cancelled(self, state: _State) -> LLMError:
        return cancelled(self.ctx, streamed=state.streamed, partial_text=state.text,
                         partial_usage=self._usage(state))

    def _consume(self, chunk: dict, state: _State, on_event: OnEvent) -> None:
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            state.usage = usage
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str) and content:
                state.text += content
                state.streamed = True
                on_event(TextDelta(content))
            refusal = delta.get("refusal")
            if isinstance(refusal, str) and refusal:
                state.text += refusal
                state.refusal = True
                state.streamed = True
                on_event(TextDelta(refusal))
            for part in delta.get("tool_calls") or []:
                self._consume_call(part, state, on_event)
            for key in self.preset.capture_message_keys:
                value = delta.get(key)
                if isinstance(value, list):
                    merge_reasoning(state.reasoning.setdefault(key, []), value)
                elif isinstance(value, dict):
                    deep_merge(state.message_extra.setdefault(key, {}), value)
                elif value is not None:
                    state.message_extra[key] = copy.deepcopy(value)
            finish = choice.get("finish_reason")
            if finish:
                state.finish = finish
                if finish == "error":
                    state.error_chunk = chunk

    def _consume_call(self, part: dict, state: _State, on_event: OnEvent) -> None:
        index = part.get("index")
        if not isinstance(index, int):
            index = len(state.calls)
        call = state.calls.setdefault(index, _Call(index))
        if part.get("id"):
            call.id = part["id"]
        function = part.get("function") or {}
        if function.get("name") and not call.name:
            call.name = function["name"]
        if call.name and not call.started:
            call.started = True
            state.streamed = True
            on_event(ToolCallStarted(index, call.name))
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            call.raw += arguments
        extra = {k: v for k, v in part.items()
                 if k not in _TOOL_CALL_FIELDS and k in self.preset.capture_tool_call_keys}
        if extra:
            deep_merge(call.extra, extra)

    def _usage(self, state: _State) -> Usage:
        usage = state.usage
        if not usage:
            return not_reported()
        prompt = usage.get("prompt_tokens")
        details = usage.get("prompt_tokens_details") or {}
        cache_read = details.get("cached_tokens") or 0
        cache_write = details.get("cache_write_tokens")
        if cache_write is None:
            cache_write = usage.get("cache_write_tokens")
        uncached = prompt - cache_read - (cache_write or 0) if isinstance(prompt, int) else None
        cost = None
        if self.preset.cost_field:
            value: Any = {"usage": usage}
            for part in self.preset.cost_field.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cost = float(value)
        output = usage.get("completion_tokens")
        return Usage(uncached, cache_read, cache_write, output if isinstance(output, int) else None, cost=cost)

    def _reply(self, state: _State, on_event: OnEvent) -> Reply:
        usage = self._usage(state)
        if state.finish == "error":
            raise classify_status(None, state.error_chunk, None, self.ctx, streamed=state.streamed,
                                  partial_text=state.text, partial_usage=usage,
                                  fallback_text="the answer ended with an error")
        if state.finish is None:
            raise make_error(Truncated, "length_no_finish", self.ctx, streamed=state.streamed,
                             partial_text=state.text, partial_usage=usage)
        calls = []
        for index in sorted(state.calls):
            c = state.calls[index]
            calls.append(ToolCall(id=c.id or new_tool_call_id(), name=c.name, arguments=parse_arguments(c.raw),
                                  raw_arguments=c.raw, provider_extra=c.extra))
        if state.refusal or state.finish == "content_filter":
            stop: Stop = "refused"
        elif state.finish == "length":
            stop = "length"
        elif calls:
            stop = "tools"
        else:
            stop = "end"
            if state.finish not in ("stop", "tool_calls", "function_call"):
                on_event(Notice(render("unknown_stop", self.ctx, reason=state.finish)))
        extra = dict(state.message_extra)
        extra.update(state.reasoning)
        message = None
        if state.text or calls:
            message = Message("assistant", state.text, tool_calls=calls, provider_extra=extra, origin=self.origin)
        return Reply(message, stop, None, usage)

    # --- the rest -------------------------------------------------------------------------------------------------
    def abort(self) -> None:
        # The flag first, so the read the shutdown wakes counts as a Stop; the shutdown before the response
        # close, so the socket is still in the registry.
        self._inflight.aborted.set()
        self._http.abort_connections()
        self._inflight.abort()

    def rate_headers(self) -> dict[str, str] | None:
        """The request-limit headers of the last response, for the pacing (P1 §5.1)."""
        return self._http.take_rate_headers()

    def list_models(self) -> list[ModelEntry]:
        self._http.begin_request()
        try:
            page = self._client.models.list()
            items = [m.to_dict() for m in page]
        except Exception as exc:
            raise classify_openai(exc, self.ctx) from exc
        return model_entries(self.preset.id, items)

    def close(self) -> None:
        self._client.close()
        self._http.close()
