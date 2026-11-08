"""[Test model] in the assistant settings and ``flextool-assistant --test-model`` (owner decisions 2026-11-08).

One tiny request to the chosen model — no tools, no case data, a few output tokens — through the backend, the net
client, the retries and the pacing a question uses (the caller has already shown the data notice when the route
needs it).  :func:`check_model` says how it went:

* ``available``: the model answered; with the time to the first part of the answer and to its end;
* ``refused``: the provider refused the request; its own words are quoted.  A refusal that says the model is not
  available to this account (:func:`not_available`) sets ``unavailable``, and the caller marks the model in the
  saved model list (:mod:`flextool.assistant.model_list_cache`);
* ``timeout``: no answer within the read timeout (300 seconds for a remote service);
* ``stopped``: the user stopped the test;
* ``failed``: any other error (a missing key, a rate limit FlexTool did not wait out, no connection), with its
  message.

The tokens are counted in the usage ledger like a question's, and the rejected attempts too.  Runs on a background
thread, never on the Tk thread.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from flextool.assistant.llm_adapter import Message, Request, TextDelta, Thinking, ToolCallStarted
from flextool.assistant.llm_errors import (
    AuthFailed,
    Cancelled,
    LLMError,
    ModelNotFound,
    NoKey,
    RateLimited,
    ServerBusy,
    TimedOut,
    Unreachable,
    message_for,
)
from flextool.assistant.llm_retry import Pacing, RetryPolicy, call_with_retries
from flextool.assistant.pacing import PROCESS_PACER, Pacer, pace_key

if TYPE_CHECKING:
    from flextool.assistant.llm_adapter import Backend, OnEvent
    from flextool.assistant.providers import Preset
    from flextool.assistant.usage import Ledger

#: The request: no tools and a few output tokens (a model that thinks first may use them up and still answers).
SYSTEM = "This is a connection test. Reply with the single word OK."
QUESTION = "OK?"
MAX_OUTPUT_TOKENS = 16

AVAILABLE = "{model} is available: the answer started after {first} s and ended after {total} s."
AVAILABLE_NO_TEXT = "{model} is available: it answered after {total} s."
REFUSED = "{provider} refused the test request: {text}"
MARKED = ("{model} is now marked “not available to your account” in the model list, until the list is refreshed, "
          "the key changes, a test succeeds or 7 days have passed.")
STOPPED = "Stopped."
#: Words of a refusal that says the model is closed to this account (besides a 404 "not found").
_NOT_AVAILABLE = ("no longer available", "not available")

Outcome = Literal["available", "refused", "timeout", "stopped", "failed"]


@dataclass(frozen=True)
class CheckResult:
    outcome: Outcome
    message: str
    first_byte: float | None = None
    total: float | None = None
    unavailable: bool = False           # the refusal says the model is not available to this account


def not_available(err: LLMError) -> bool:
    """Whether *err* says the model is not available to this account: a "model not found" answer (404), or a
    refusal whose words say "not available" / "no longer available"."""
    if isinstance(err, ModelNotFound):
        return True
    if isinstance(err, (RateLimited, ServerBusy, TimedOut, Unreachable, Cancelled, AuthFailed, NoKey)):
        return False
    text = " ".join(str(err.fields.get("text") or err.user_message).lower().split())
    return any(phrase in text for phrase in _NOT_AVAILABLE)


def _seconds(value: float) -> str:
    return f"{value:.1f}"


def check_model(preset: Preset, backend: Backend, cancel: threading.Event, *, label: str, ledger: Ledger | None,
                ledger_key: str, free_plan: bool = False, front_end: str = "gui", on_event: OnEvent | None = None,
                policy: RetryPolicy | None = None, clock: Callable[[], float] = time.monotonic,
                sleep: Callable[[float], bool] | None = None, pacer: Pacer | None = None) -> CheckResult:
    """Send the test request to *backend* (see the module text).  *label* names the service in the messages;
    *on_event* gets the waits (pacing, retries, the gate) and every other event of the request.  *pacer* is the
    process's unless the backend paces itself (the window's gate)."""
    model = backend.origin[2]
    emit = on_event or (lambda _e: None)
    paced_by_backend = bool(getattr(type(backend), "paces", False))
    pacer = backend.pacer if paced_by_backend else (pacer or PROCESS_PACER)  # type: ignore[attr-defined]
    key = pace_key(backend.origin)
    from flextool.assistant.providers import pacing_seed

    pacer.prepare(key, pacing_seed(preset, free_plan))
    read_headers = getattr(backend, "rate_headers", None)
    pacing = Pacing(key, pacer, here=not paced_by_backend, label=label, free_plan=free_plan,
                    headers=read_headers if callable(read_headers) else (lambda: None))
    request = Request(system=(SYSTEM,), tools=(), messages=(Message("user", QUESTION),),
                      max_output_tokens=MAX_OUTPUT_TOKENS)
    times: dict[str, float] = {}

    def seen(event: object) -> None:
        if isinstance(event, (TextDelta, Thinking, ToolCallStarted)) and "first" not in times:
            times["first"] = clock()
        emit(event)  # type: ignore[arg-type]

    def attempt():  # noqa: ANN202 — the backend's Reply
        times.clear()
        times["start"] = clock()
        return backend.stream(request, seen, cancel)

    def rejected(_err: LLMError) -> None:
        if ledger is not None:
            ledger.record_rejected(ledger_key, label, model)

    try:
        reply = call_with_retries(attempt, policy or RetryPolicy(), emit, cancel, sleep=sleep, clock=clock,
                                  on_rejected=rejected, pacing=pacing)
    except LLMError as err:
        if ledger is not None and (err.partial_usage is not None or err.streamed):
            ledger.record(ledger_key, label, model, err.partial_usage)
        if isinstance(err, Cancelled) or cancel.is_set():
            return CheckResult("stopped", STOPPED)
        if isinstance(err, TimedOut):
            return CheckResult("timeout", message_for(err, front_end))
        if not_available(err) or err.status is not None and 400 <= err.status < 500 and not isinstance(
                err, (RateLimited, AuthFailed)):
            text = str(err.fields.get("text") or "").strip()
            message = REFUSED.format(provider=label, text=text) if text else message_for(err, front_end)
            unavailable = not_available(err)
            if unavailable:
                message = f"{message} {MARKED.format(model=model)}"
            return CheckResult("refused", message, unavailable=unavailable)
        return CheckResult("failed", message_for(err, front_end))
    end = clock()
    if ledger is not None:
        ledger.record(ledger_key, label, model, reply.usage)
    start = times.get("start", end)
    total = end - start
    first = times.get("first")
    if first is None:
        return CheckResult("available", AVAILABLE_NO_TEXT.format(model=model, total=_seconds(total)), None, total)
    return CheckResult("available", AVAILABLE.format(model=model, first=_seconds(first - start),
                                                     total=_seconds(total)), first - start, total)
