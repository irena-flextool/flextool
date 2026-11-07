"""The agent loop (P1 §6): one question in, model requests and tool calls until the model answers.

A :class:`Session` holds one conversation with one model.  :meth:`Session.run_turn` runs in a worker thread
(:class:`TurnWorker`): it appends the question (with the context block when it changed), asks the model, runs the
tool calls the model asks for — all of one answer in one round, at most ``max_calls_per_round`` — and asks again,
for at most ``max_rounds`` rounds; then it asks for an answer without tools.  The history every request carries
stays valid for every provider whatever happens: a tool call that was not run gets an error result (cut off,
declined by the provider, stopped by the user, too many, FlexTool closed), and a stopped answer keeps its text
without calls.

**Provider data** (thinking blocks, signatures) is replayed only while the history before it is byte-identical to
the request that produced it.  The history is append-only except at the *compaction points* (P1-D7), where the
provider data of every message is dropped:

* a turn boundary every ``compaction_every`` turns (old tool results replaced by a short note, the last
  ``keep_recent_turns`` completed turns kept);
* a context overflow (all tool results before this turn replaced), then one retry;
* a rejected signature (:class:`HistoryRejected`), then one retry;
* a resumed conversation whose instructions or tools changed (``system_hash`` / ``tools_hash``) or that was held
  with another model (ids re-mapped too);
* a switch of model or provider (:meth:`Session.switch`).

Tool calls run under the context's lock (one per process), taken in a loop that checks the cancel event; before
the first call of a round the conversation is saved with every call answered "not run", so the saved file is a
valid history even if FlexTool closes mid-call.  Usage is recorded in the ledger and the conversation for every
request that reached the model, also a failed or stopped one.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from flextool.assistant import caps as caps_module
from flextool.assistant import prompt, registry
from flextool.assistant.errors import message as tool_message
from flextool.assistant.llm_adapter import (
    Backend,
    Event,
    Message,
    Notice,
    OnEvent,
    Reply,
    Request,
    RetryWait,
    Route,
    TextDelta,
    Thinking,
    ToolCall,
    ToolCallStarted,
    ToolDone,
    ToolRunning,
    ToolSpec,
    ToolWaiting,
    Usage,
    UsageUpdate,
    Waiting,
    drop_provider_extra,
    prefix_key,
    remap_ids,
)
from flextool.assistant.llm_errors import (
    MESSAGES,
    ContextTooLong,
    ErrorContext,
    HistoryRejected,
    LLMError,
    ProviderError,
    Refused,
    Truncated,
    cancelled,
    make_error,
    refusal_message,
    render,
)
from flextool.assistant.llm_retry import RetryPolicy, call_with_retries
from flextool.assistant.usage import add_usage, ledger_key

if TYPE_CHECKING:
    from flextool.assistant.context import AssistantContext
    from flextool.assistant.probes import ModelFacts
    from flextool.assistant.providers import Preset
    from flextool.assistant.store import Conversation, Store
    from flextool.assistant.usage import Ledger

#: What the session tells the model and the user, in plain English.
SESSION_MESSAGES: dict[str, str] = {
    "empty": "The model returned an empty answer.",
    "cut_off_call": "Not run: the answer was cut off before this tool call was complete.",
    "declined_call": "Not run: the provider declined this request.",
    "limit_call": "Not run: the tool limit for this question is reached.",
    "too_many_calls": "Too many tool calls in one step; at most {n}. Ask again for the rest.",
    "bad_json": "The tool arguments were not valid JSON: {start}",
    "stopped": "Stopped by the user.",
    "closed": "Not run: FlexTool was closed before this tool call finished.",
    "limit": ("The tool limit for this question is reached. Answer with what you have, and say what is still "
              "unknown."),
    "compacted": ("[Earlier result of {tool}({args}) removed to keep the conversation short. Call the tool again if "
                  "you need it.]"),
    "text_call": "The model wrote a tool call as text; it may not support tool calls well.",
    "summary_head": "Summary of an earlier conversation (text only, no tool results):",
    "empty_result": "The tool returned nothing.",
}

#: Rounds per question by level (internal names; P1 defaults, P3 owns them).
ROUNDS_BY_TIER: dict[str, int] = {"small": 3, "medium": 8, "large": 12}
#: The result caps preset of each level, and the context below which every level gets the small one.
RESULT_CAPS_BY_TIER: dict[str, str] = {"small": "small", "medium": "medium", "large": "large"}
SMALL_CONTEXT = 16_000
#: Prefixes some models put before a tool name.
TOOL_NAME_PREFIXES = ("functions.", "default_api:", "default_api.")
#: Lengths: the arguments named in a compaction note, the tool line, the window's expanded tool line.
COMPACTED_ARGS_MAX = 120
ARGS_PREVIEW_MAX = 80
ARGUMENTS_MAX = 2_000
RESULT_PREVIEW_MAX = 4_000
BAD_JSON_START = 80
#: The `/new carry` summary: characters of each final answer, of the whole summary.
SUMMARY_ANSWER_MAX = 600
SUMMARY_MAX = 6_000
#: How long one wait for the tool lock lasts before the cancel event is checked again.
LOCK_POLL = 0.25

#: Events that belong to an answer still arriving; dropped once the turn is stopped.
_STREAM_EVENTS = (TextDelta, Thinking, ToolCallStarted, Waiting, RetryWait)


@dataclass(frozen=True)
class TurnCaps:
    max_rounds: int = 8
    max_calls_per_round: int = 8
    max_output_tokens: int | None = None        # None: the preset's default

    @classmethod
    def for_tier(cls, tier: str) -> TurnCaps:
        """The caps of a level (internal or user-facing name)."""
        return cls(max_rounds=ROUNDS_BY_TIER[prompt.level_name(tier)])


def result_caps_for(tier: str, facts: ModelFacts | None, override: str | None = None) -> caps_module.Caps:
    """The result caps of a level: *override* (``--result-cap``) or the level's preset, and the small preset
    whenever the model's context is below :data:`SMALL_CONTEXT` tokens."""
    name = override or RESULT_CAPS_BY_TIER[prompt.level_name(tier)]
    context = facts.context if facts is not None else None
    if context is not None and context < SMALL_CONTEXT:
        name = "small"
    return caps_module.preset(name)


@dataclass
class TurnResult:
    text: str                       # the final answer of the turn (the last assistant text)
    ok: bool
    error: LLMError | None
    rounds: int
    usage: list[Usage] = field(default_factory=list)


def text_hash(parts: object) -> str:
    """``sha256`` of *parts* as canonical JSON, first 16 characters."""
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def tools_hash(tools: tuple[ToolSpec, ...]) -> str:
    return text_hash([[t.name, t.description, t.input_schema] for t in tools])


def tool_specs(tier: str) -> tuple[ToolSpec, ...]:
    """The tools a level sends, sorted by name."""
    tools = registry.tools_for_tier(prompt.level_name(tier))
    return tuple(sorted((ToolSpec(t.name, t.description, t.input_schema) for t in tools), key=lambda t: t.name))


def normalise_tool_name(name: str) -> str:
    """*name* without the prefixes some models add (``functions.``, ``default_api:``)."""
    for prefix in TOOL_NAME_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def carry_summary(messages: list[Message]) -> str:
    """The seed of `/new carry`: each earlier question and the first characters of its final answer, made here
    without a request."""
    lines = [SESSION_MESSAGES["summary_head"]]
    question: str | None = None
    answer = ""
    pairs: list[tuple[str, str]] = []
    for m in messages:
        if m.role == "user" and not m.text.startswith(SESSION_MESSAGES["limit"]):
            if question is not None:
                pairs.append((question, answer))
            question = prompt.strip_context_block(m.text)
            answer = ""
        elif m.role == "assistant" and m.text.strip():
            answer = m.text
    if question is not None:
        pairs.append((question, answer))
    for q, a in pairs:
        lines.append(f"Question: {q.strip()}")
        if a.strip():
            lines.append(f"Answer: {caps_module.cap_text(a.strip(), SUMMARY_ANSWER_MAX)}")
    return caps_module.cap_text("\n".join(lines), SUMMARY_MAX)


class _Stopped(Exception):
    """The user stopped the turn between two tool calls or while a call waited for the lock."""


class Session:
    def __init__(self, ctx: AssistantContext, backend: Backend, preset: Preset, *, conversation: Conversation,
                 caps: TurnCaps, tier: str, store: Store | None, ledger: Ledger | None, facts: ModelFacts | None,
                 compaction_every: int = 6, keep_recent_turns: int = 2, route: Route | None = None,
                 retry_policy: RetryPolicy | None = None, sleep: Callable[[float], bool] | None = None,
                 clock: Callable[[], float] = time.monotonic, rng: random.Random | None = None) -> None:
        self.ctx = ctx
        self.conversation = conversation
        self.caps = caps
        self.store = store
        self.ledger = ledger
        self.compaction_every = max(1, compaction_every)
        self.keep_recent_turns = max(0, keep_recent_turns)
        self.retry_policy = retry_policy or RetryPolicy()
        self._sleep = sleep
        self._clock = clock
        self._rng = rng
        self._turn_lock = threading.Lock()
        self._cancel: threading.Event | None = None
        self._last_block = prompt.last_context_block([m.text for m in conversation.messages if m.role == "user"])
        self._attach(backend, preset, tier, facts, route)
        self._resume()

    # --- the model and its instructions ---------------------------------------------------------------------------
    def _attach(self, backend: Backend, preset: Preset, tier: str, facts: ModelFacts | None,
                route: Route | None) -> None:
        from flextool.assistant.endpoint import route_for

        self.backend = backend
        self.preset = preset
        self.tier = prompt.level_name(tier)
        self.facts = facts
        self.route = route or route_for(preset, backend.origin[1], facts=facts)
        front_end = self.ctx.front_end if self.ctx.front_end in prompt.FRONT_END_NOTES else "agent"
        self.system = prompt.system_layers(front_end, self.tier)
        self.tools = tool_specs(self.tier)
        self.tool_names = frozenset(t.name for t in self.tools)
        self.system_hash = text_hash(list(self.system))
        self.tools_hash = tools_hash(self.tools)
        self.prefix_key = prefix_key(self.system, self.tools)
        context = facts.context if facts is not None else None
        if context is not None and context < SMALL_CONTEXT:
            self.ctx.caps = caps_module.preset("small")
        self.error_ctx = ErrorContext(preset_id=preset.id, provider=self.route.label, model=backend.origin[2])
        self.ledger_key = ledger_key(preset.id, backend.origin[1], forwarded_to=self.route.forwarded_to)

    def _resume(self) -> None:
        """The resume rule: provider data is dropped (and ids re-mapped for another model) when the conversation
        was held with other instructions, tools or model."""
        conv = self.conversation
        if conv.messages:
            if conv.origin is not None and tuple(conv.origin) != tuple(self.backend.origin):
                conv.messages = remap_ids(conv.messages)
            elif conv.system_hash != self.system_hash or conv.tools_hash != self.tools_hash:
                conv.messages = drop_provider_extra(conv.messages)
        conv.origin = self.backend.origin
        conv.tier = self.tier
        conv.system_hash = self.system_hash
        conv.tools_hash = self.tools_hash

    def switch(self, backend: Backend, preset: Preset, *, tier: str, facts: ModelFacts | None,
               route: Route | None = None, caps: TurnCaps | None = None) -> None:
        """Go on with another model or provider (or level): ids re-mapped, provider data dropped, instructions and
        tools rebuilt.  Not while a turn runs."""
        with self._turn_lock:
            self.conversation.messages = remap_ids(self.conversation.messages)
            if caps is not None:
                self.caps = caps
            self._attach(backend, preset, tier, facts, route)
            self._resume()

    def abort(self) -> None:
        """Stop the running turn (any thread): the cancel event is set and the response being read is closed."""
        cancel = self._cancel
        if cancel is not None:
            cancel.set()
        self.backend.abort()

    # --- the history ----------------------------------------------------------------------------------------------
    @property
    def messages(self) -> list[Message]:
        return self.conversation.messages

    def _last_turn(self) -> int:
        return max((m.turn for m in self.messages), default=0)

    def _compact(self, keep_after: int) -> None:
        """Replace each tool result of a turn up to *keep_after* by a short note, and drop all provider data."""
        calls: dict[str, ToolCall] = {}
        out: list[Message] = []
        for m in drop_provider_extra(self.messages):
            if m.role == "assistant":
                calls.update({c.id: c for c in m.tool_calls})
            elif m.role == "tool" and m.turn <= keep_after and not m.compacted:
                call = calls.get(m.tool_call_id or "")
                args = caps_module.cap_text(" ".join((call.raw_arguments if call else "").split()),
                                            COMPACTED_ARGS_MAX)
                note = SESSION_MESSAGES["compacted"].format(tool=m.tool_name or (call.name if call else "tool"),
                                                            args=args)
                m = dataclasses.replace(m, text=note, compacted=True)
            out.append(m)
        self.conversation.messages = out

    def _drop_extra(self) -> None:
        self.conversation.messages = drop_provider_extra(self.messages)

    def _save(self) -> None:
        if self.store is not None:
            self.store.save(self.conversation)

    def _append_user(self, text: str, turn: int) -> None:
        block = prompt.context_block(self.ctx.snapshot(), self.ctx.redact)
        if block != self._last_block:
            text = f"{block}\n\n{text}"
            self._last_block = block
        last = self.messages[-1] if self.messages else None
        if last is not None and last.role == "user":
            # The previous question got no answer: one user message, so the history alternates (m12).
            self.messages[-1] = dataclasses.replace(last, text=f"{last.text}\n\n{text}", turn=turn)
        else:
            self.messages.append(Message("user", text, turn=turn))

    # --- usage ----------------------------------------------------------------------------------------------------
    def _record(self, usage: Usage | None, turn_usage: list[Usage], emit: OnEvent) -> None:
        add_usage(self.conversation.usage, usage)
        if self.ledger is not None:
            self.ledger.record(self.ledger_key, self.route.label, self.backend.origin[2], usage)
        recorded = usage if usage is not None else Usage(None, None, None, None, reported=False)
        turn_usage.append(recorded)
        emit(UsageUpdate(recorded))

    def _rejected(self, _err: LLMError) -> None:
        if self.ledger is not None:
            self.ledger.record_rejected(self.ledger_key, self.route.label, self.backend.origin[2])

    # --- one request, with its recoveries -------------------------------------------------------------------------
    def _request(self, tool_choice: Literal["auto", "none"]) -> Request:
        max_tokens = self.caps.max_output_tokens or self.preset.default_max_output_tokens
        return Request(system=self.system, tools=self.tools, messages=tuple(self.messages),
                       max_output_tokens=max_tokens, tool_choice=tool_choice, prefix_key=self.prefix_key)

    def _ask(self, tool_choice: Literal["auto", "none"], turn: int, cancel: threading.Event, emit: OnEvent,
             turn_usage: list[Usage]) -> Reply:
        """One model request with the retries, and the overflow and rejected-signature recoveries (one each)."""
        overflow_done = history_done = False
        while True:
            req = self._request(tool_choice)
            try:
                reply = call_with_retries(lambda: self.backend.stream(req, emit, cancel), self.retry_policy, emit,
                                          cancel, sleep=self._sleep, clock=self._clock, rng=self._rng,
                                          on_rejected=self._rejected)
            except LLMError as err:
                if err.partial_usage is not None or err.streamed:
                    self._record(err.partial_usage, turn_usage, emit)
                if isinstance(err, ContextTooLong) and not err.streamed and not overflow_done:
                    overflow_done = True
                    self._overflow(turn, emit)
                    continue
                if isinstance(err, HistoryRejected) and not err.streamed and not history_done:
                    history_done = True
                    self._drop_extra()
                    emit(Notice(MESSAGES["history"]))
                    continue
                if isinstance(err, ContextTooLong):
                    raise self._still_too_long(err) from err
                raise
            self._record(reply.usage, turn_usage, emit)
            if cancel.is_set():
                # The answer arrived after the turn was stopped: only its text is kept.
                raise cancelled(self.error_ctx, streamed=True,
                                partial_text=reply.message.text if reply.message else "")
            if reply.stop == "context":
                if overflow_done:
                    raise self._still_too_long(None)
                overflow_done = True
                self._overflow(turn, emit)
                continue
            return reply

    def _overflow(self, turn: int, emit: OnEvent) -> None:
        self._compact(keep_after=turn - 1)
        emit(Notice(render("context_retry", self.error_ctx)))

    def _still_too_long(self, cause: LLMError | None) -> LLMError:
        err = make_error(ContextTooLong, "context", self.error_ctx)
        if cause is not None:
            err.partial_usage = cause.partial_usage
        return err

    # --- tool calls -----------------------------------------------------------------------------------------------
    def _result(self, call: ToolCall, text: str, *, is_error: bool, turn: int) -> Message:
        return Message("tool", text or SESSION_MESSAGES["empty_result"], tool_call_id=call.id,
                       tool_name=normalise_tool_name(call.name), is_error=is_error, turn=turn)

    def _answer_all(self, calls: list[ToolCall], key: str, turn: int) -> None:
        self.messages.extend(self._result(c, SESSION_MESSAGES[key], is_error=True, turn=turn) for c in calls)

    def _acquire_tool_lock(self, name: str, cancel: threading.Event, emit: OnEvent) -> None:
        lock = self.ctx.lock
        if lock.acquire(timeout=0):
            return
        emit(ToolWaiting(name))
        while not lock.acquire(timeout=LOCK_POLL):
            if cancel.is_set():
                raise _Stopped
        if cancel.is_set():
            lock.release()
            raise _Stopped

    def _run_call(self, call: ToolCall, index: int, n_allowed: int, turn: int, cancel: threading.Event,
                  emit: OnEvent) -> Message:
        if cancel.is_set():
            raise _Stopped
        name = normalise_tool_name(call.name)
        if index >= n_allowed:
            return self._result(call, SESSION_MESSAGES["too_many_calls"].format(n=n_allowed), is_error=True,
                                turn=turn)
        if call.arguments is None:
            start = call.raw_arguments[:BAD_JSON_START]
            return self._result(call, SESSION_MESSAGES["bad_json"].format(start=self.ctx.redact(start)),
                                is_error=True, turn=turn)
        if name not in self.tool_names:
            text = tool_message("unknown_tool", name=self.ctx.redact(name), names=", ".join(sorted(self.tool_names)))
            return self._result(call, text, is_error=True, turn=turn)
        arguments = self.ctx.redact(json.dumps(call.arguments, ensure_ascii=False))
        self._acquire_tool_lock(name, cancel, emit)
        try:
            emit(ToolRunning(name, caps_module.cap_text(arguments, ARGS_PREVIEW_MAX)))
            started = time.monotonic()
            outcome = registry.call_tool(self.ctx, name, call.arguments)
            seconds = time.monotonic() - started
        finally:
            self.ctx.lock.release()
        result = self._result(call, outcome.text, is_error=not outcome.ok, turn=turn)
        emit(ToolDone(name, outcome.ok, _summary(outcome), seconds,
                      caps_module.cap_text(arguments, ARGUMENTS_MAX), result.text[:RESULT_PREVIEW_MAX]))
        return result

    def _run_calls(self, calls: list[ToolCall], turn: int, cancel: threading.Event, emit: OnEvent) -> None:
        """Run the calls of one answer in order.  The conversation is saved first with every call "not run",
        and again after each call, so the saved history is always valid."""
        first = len(self.messages)
        self._answer_all(calls, "closed", turn)
        self._save()
        n_allowed = self.caps.max_calls_per_round
        for i, call in enumerate(calls):
            try:
                result = self._run_call(call, i, n_allowed, turn, cancel, emit)
            except _Stopped:
                for j in range(i, len(calls)):
                    self.messages[first + j] = self._result(calls[j], SESSION_MESSAGES["stopped"], is_error=True,
                                                            turn=turn)
                self._save()
                raise
            self.messages[first + i] = result
            self._save()

    def _text_call_notice(self, text: str, emit: OnEvent) -> None:
        stripped = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            return
        try:
            value = json.loads(stripped)
        except ValueError:
            return
        if not isinstance(value, dict):
            return
        named = value.get("name")
        if not isinstance(named, str) and isinstance(value.get("function"), dict):
            named = value["function"].get("name")
        if isinstance(named, str) and normalise_tool_name(named) in registry.all_tools():
            emit(Notice(SESSION_MESSAGES["text_call"]))

    # --- one turn -------------------------------------------------------------------------------------------------
    def run_turn(self, user_text: str, on_event: OnEvent, cancel: threading.Event) -> TurnResult:
        """Answer *user_text*.  Runs in a worker thread; *on_event* gets the events (from this thread);
        *cancel* (set by :meth:`abort` or the front end) stops the turn and repairs the history."""
        with self._turn_lock:
            self._cancel = cancel
            try:
                return self._turn(user_text, on_event, cancel)
            finally:
                self._cancel = None

    def _turn(self, user_text: str, on_event: OnEvent, cancel: threading.Event) -> TurnResult:
        def emit(event: Event) -> None:
            if cancel.is_set() and isinstance(event, _STREAM_EVENTS):
                return
            on_event(event)

        last = self.messages[-1] if self.messages else None
        unanswered = last is not None and last.role == "user"
        turn = last.turn if unanswered and last is not None and last.turn else self._last_turn() + 1
        if not unanswered and turn > 1 and (turn - 1) % self.compaction_every == 0:
            self._compact(keep_after=turn - 1 - self.keep_recent_turns)
        self._append_user(user_text, turn)
        turn_usage: list[Usage] = []
        rounds = 0
        final_text = ""
        try:
            while True:
                limit_reached = rounds >= self.caps.max_rounds
                if limit_reached:
                    self.messages.append(Message("user", SESSION_MESSAGES["limit"], turn=turn))
                reply = self._ask("none" if limit_reached else "auto", turn, cancel, emit, turn_usage)
                rounds += 1
                if reply.message is None:
                    self._save()
                    return self._fail(ProviderError(SESSION_MESSAGES["empty"], provider=self.route.label), "",
                                      rounds, turn_usage)
                message = dataclasses.replace(reply.message, turn=turn)
                self.messages.append(message)
                calls = list(message.tool_calls)
                if message.text.strip():
                    final_text = message.text
                if reply.stop in ("length", "refused"):
                    if calls:
                        self._answer_all(calls, "cut_off_call" if reply.stop == "length" else "declined_call",
                                         turn)
                    self._save()
                    return self._fail(self._stop_error(reply), final_text, rounds, turn_usage)
                if not calls:
                    self._text_call_notice(message.text, emit)
                    self._save()
                    return TurnResult(final_text, True, None, rounds, turn_usage)
                if limit_reached:
                    self._answer_all(calls, "limit_call", turn)
                    self._save()
                    return TurnResult(final_text, True, None, rounds, turn_usage)
                self._run_calls(calls, turn, cancel, emit)
                self._save()
        except _Stopped:
            return self._fail(cancelled(self.error_ctx), final_text, rounds, turn_usage)
        except LLMError as err:
            if err.partial_text.strip():
                # A stopped or broken answer keeps what was shown, without calls or provider data.
                self.messages.append(Message("assistant", err.partial_text, turn=turn))
                final_text = err.partial_text
            self._save()
            return self._fail(err, final_text, rounds + (1 if err.streamed else 0), turn_usage)

    def _stop_error(self, reply: Reply) -> LLMError:
        if reply.stop == "length":
            max_tokens = self.caps.max_output_tokens or self.preset.default_max_output_tokens
            return make_error(Truncated, "length", self.error_ctx, message_fields={"tokens": max_tokens})
        err = make_error(Refused, "refused", self.error_ctx, message_fields={"reason_suffix": ""})
        err.user_message = refusal_message(self.error_ctx, reply.refusal_reason)
        err.args = (err.user_message,)
        return err

    @staticmethod
    def _fail(err: LLMError, text: str, rounds: int, usage: list[Usage]) -> TurnResult:
        return TurnResult(text, False, err, rounds, usage)


def _summary(outcome: registry.ToolOutcome) -> str:
    """The short outcome on a tool line: ``5 results``, ``120 rows``, or the start of the error message."""
    if not outcome.ok:
        first = (outcome.message or "").splitlines()[0] if outcome.message else ""
        return caps_module.cap_text(first, ARGS_PREVIEW_MAX)
    payload = outcome.payload or {}
    for key, value in payload.items():
        if isinstance(value, list):
            return f"{len(value)} {key.replace('_', ' ')}"
    return f"{len(outcome.text)} characters"


class TurnWorker:
    """Runs the turns of a session in a daemon thread, one at a time.

    :meth:`stop` returns at once: it sets the turn's cancel event and closes the response being read.  A stopped
    turn's events and result that arrive later are discarded (a generation counter), while the session still
    repairs its history before the next turn starts.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self._generation = 0
        self._lock = threading.Lock()
        self._cancel: threading.Event | None = None
        self._thread: threading.Thread | None = None

    @property
    def generation(self) -> int:
        return self._generation

    def start(self, text: str, on_event: OnEvent, on_done: Callable[[TurnResult], None]) -> int:
        """Run a turn for *text*; *on_event* and *on_done* are called from the worker thread, and only while the
        turn is the current one.  Returns the turn's generation."""
        with self._lock:
            self._generation += 1
            generation = self._generation
            cancel = threading.Event()
            self._cancel = cancel

        def current() -> bool:
            return self._generation == generation

        def events(event: Event) -> None:
            if current():
                on_event(event)

        def run() -> None:
            result = self.session.run_turn(text, events, cancel)
            if current():
                on_done(result)

        thread = threading.Thread(target=run, name="flextool-assistant-turn", daemon=True)
        self._thread = thread
        thread.start()
        return generation

    def stop(self) -> None:
        """Stop the running turn (any thread); its later events and result are discarded."""
        with self._lock:
            self._generation += 1
            cancel = self._cancel
        if cancel is not None:
            cancel.set()
        self.session.abort()

    def join(self, timeout: float | None = None) -> bool:
        """Wait for the worker thread; ``True`` when it has finished."""
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()
