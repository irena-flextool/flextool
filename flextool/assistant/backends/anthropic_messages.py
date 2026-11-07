"""The Anthropic Messages backend (P1 §3.2): a cached prefix and thinking blocks replayed verbatim.

The system layers and the tool list end with a cache breakpoint, and the request asks for automatic caching of
the conversation (three breakpoints of the four allowed).  No thinking or sampling settings are sent.  The
answer's content blocks (thinking, redacted thinking, text, tool use, in order) are kept in ``provider_extra`` and
sent back unchanged to the same origin; to any other origin, or after a compaction point dropped them, the
message is rebuilt from its text and tool calls.  Consecutive tool results go in one user message.
"""
from __future__ import annotations

import copy
import json
import threading
from typing import Any

from flextool.assistant.backends._common import InFlight, api_key_for, error_context
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
    Thinking,
    ToolCall,
    ToolCallStarted,
    Usage,
    not_reported,
    parse_arguments,
)
from flextool.assistant.llm_errors import (
    LLMError,
    cancelled,
    classify_anthropic,
    refusal_reason,
    render,
)
from flextool.assistant.net import build_http_client, key_header_value
from flextool.assistant.providers import Preset, model_entries

EPHEMERAL = {"type": "ephemeral"}
BLOCKS_KEY = "anthropic_blocks"
#: Stop reasons of the Messages API and what the session makes of them.
STOPS: dict[str, Stop] = {"tool_use": "tools", "end_turn": "end", "stop_sequence": "end", "pause_turn": "end",
                          "max_tokens": "length", "model_context_window_exceeded": "context", "refusal": "refused"}
_THINKING_BLOCKS = ("thinking", "redacted_thinking")


class AnthropicMessagesBackend:
    def __init__(self, preset: Preset, conn: Connection, *, transport: Any = None) -> None:
        import anthropic

        self.preset = preset
        self.conn = conn
        self.origin = (preset.id, conn.base_url, conn.model)
        self.ctx = error_context(preset, conn)
        key = api_key_for(preset, conn, self.ctx)
        self._http = build_http_client(conn, backend="anthropic", api_key=key,
                                       required_headers=preset.required_headers, transport=transport)
        # The library sends the key as a bearer token; passing that header explicitly also keeps a header from
        # its custom-headers environment variable from replacing it.
        self._client = anthropic.Anthropic(api_key=key, base_url=conn.base_url, max_retries=0,
                                           http_client=self._http.client, open_telemetry=False,
                                           default_headers={"Authorization": key_header_value(key)})
        self._inflight = InFlight()
        self.wait_interval = 5.0

    # --- the request ----------------------------------------------------------------------------------------------
    def wire_messages(self, req: Request) -> list[dict]:
        out: list[dict] = []

        def add(role: str, blocks: list[dict]) -> None:
            if out and out[-1]["role"] == role:
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": blocks})

        for m in req.messages:
            if m.role == "user":
                add("user", [{"type": "text", "text": m.text}])
            elif m.role == "assistant":
                blocks = m.provider_extra.get(BLOCKS_KEY) if m.origin == self.origin else None
                if blocks:
                    add("assistant", copy.deepcopy(blocks))
                    continue
                rebuilt: list[dict] = []
                if m.text:
                    rebuilt.append({"type": "text", "text": m.text})
                rebuilt += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments or {}}
                            for c in m.tool_calls]
                if rebuilt:
                    add("assistant", rebuilt)
            else:
                add("user", [{"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.text,
                              "is_error": m.is_error}])
        return out

    def request_params(self, req: Request) -> dict[str, Any]:
        """The arguments of ``messages.stream`` (the request body without ``stream``)."""
        params: dict[str, Any] = {"model": self.conn.model, "max_tokens": req.max_output_tokens,
                                  "messages": self.wire_messages(req), "cache_control": dict(EPHEMERAL)}
        if req.system:
            system = [{"type": "text", "text": layer} for layer in req.system]
            system[-1]["cache_control"] = dict(EPHEMERAL)
            params["system"] = system
        if req.tools:
            tools = [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
                     for t in req.tools]
            tools[-1]["cache_control"] = dict(EPHEMERAL)
            params["tools"] = tools
            if req.tool_choice == "none":
                params["tool_choice"] = {"type": "none"}
        return params

    # --- the answer -----------------------------------------------------------------------------------------------
    def stream(self, req: Request, on_event: OnEvent, cancel: threading.Event) -> Reply:
        params = self.request_params(req)
        self._inflight.begin()
        timer = SilenceTimer(on_event, interval=self.wait_interval).start()
        progress = _Progress()
        try:
            try:
                manager = self._client.messages.stream(**params)
                stream = manager.__enter__()
            except Exception as exc:
                if cancel.is_set() or self._inflight.aborted.is_set():
                    raise cancelled(self.ctx) from exc
                raise classify_anthropic(exc, self.ctx) from exc
            progress.opened = True
            self._inflight.set(stream.response)
            try:
                for event in stream:
                    timer.touch()
                    if cancel.is_set() or self._inflight.aborted.is_set():
                        raise self._cancelled(progress)
                    self._consume(event, stream, progress, on_event)
                if cancel.is_set() or self._inflight.aborted.is_set():
                    raise self._cancelled(progress)
                final = stream.get_final_message()
            except LLMError:
                raise
            except Exception as exc:
                if cancel.is_set() or self._inflight.aborted.is_set():
                    raise self._cancelled(progress) from exc
                raise classify_anthropic(exc, self.ctx, streamed=progress.streamed, partial_text=progress.text,
                                         partial_usage=progress.partial_usage()) from exc
            finally:
                manager.__exit__(None, None, None)
        finally:
            timer.stop()
            self._inflight.clear()
        return self._reply(final, progress, on_event)

    def _cancelled(self, progress: _Progress) -> LLMError:
        return cancelled(self.ctx, streamed=progress.streamed, partial_text=progress.text,
                         partial_usage=progress.partial_usage())

    def _consume(self, event: Any, stream: Any, progress: _Progress, on_event: OnEvent) -> None:
        kind = event.type
        if kind == "message_start":
            progress.usage = stream.current_message_snapshot.usage
        elif kind == "text":
            if event.text:
                progress.text += event.text
                progress.streamed = True
                on_event(TextDelta(event.text))
        elif kind == "content_block_start":
            block = event.content_block
            if block.type in _THINKING_BLOCKS:
                on_event(Thinking())
            elif block.type == "tool_use":
                progress.raw[event.index] = ""
                progress.streamed = True
                on_event(ToolCallStarted(progress.calls, block.name))
                progress.calls += 1
        elif kind == "content_block_delta" and event.delta.type == "input_json_delta":
            progress.raw[event.index] = progress.raw.get(event.index, "") + event.delta.partial_json
        elif kind == "message_delta":
            progress.usage = stream.current_message_snapshot.usage

    def _reply(self, final: Any, progress: _Progress, on_event: OnEvent) -> Reply:
        blocks = [b.to_dict() for b in final.content]
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        calls = []
        for index, block in enumerate(blocks):
            if block.get("type") != "tool_use":
                continue
            raw = progress.raw.get(index)
            if raw is None:
                raw = json.dumps(block.get("input") or {}, ensure_ascii=False)
            calls.append(ToolCall(id=block["id"], name=block["name"], arguments=parse_arguments(raw),
                                  raw_arguments=raw))
        reason = final.stop_reason
        stop = STOPS.get(reason or "")
        if stop is None:
            stop = "end"
            on_event(Notice(render("unknown_stop", self.ctx, reason=reason)))
        refusal = None
        if stop == "refused":
            details = final.stop_details
            refusal = refusal_reason(getattr(details, "category", None), getattr(details, "explanation", None))
        usage = final.usage
        reply_usage = Usage(usage.input_tokens, usage.cache_read_input_tokens, usage.cache_creation_input_tokens,
                            usage.output_tokens)
        message = None
        if text or calls:
            message = Message("assistant", text, tool_calls=calls, provider_extra={BLOCKS_KEY: blocks},
                              origin=self.origin)
        return Reply(message, stop, refusal, reply_usage)

    # --- the rest -------------------------------------------------------------------------------------------------
    def abort(self) -> None:
        self._inflight.abort()
        self._http.abort_connections()

    def list_models(self) -> list[ModelEntry]:
        try:
            items = [m.to_dict() for m in self._client.models.list()]
        except Exception as exc:
            raise classify_anthropic(exc, self.ctx) from exc
        return model_entries(self.preset.id, items)

    def close(self) -> None:
        self._client.close()
        self._http.close()


class _Progress:
    """What has arrived of one answer: the text shown, the raw arguments per block index, the usage so far."""

    def __init__(self) -> None:
        self.text = ""
        self.streamed = False
        self.opened = False
        self.calls = 0
        self.raw: dict[int, str] = {}
        self.usage: Any = None

    def partial_usage(self) -> Usage | None:
        """The usage of a request that failed or was stopped mid-answer: the input fields as reported, the output
        not reported."""
        if self.usage is None:
            return not_reported() if self.opened else None
        return Usage(self.usage.input_tokens, self.usage.cache_read_input_tokens,
                     self.usage.cache_creation_input_tokens, None)
