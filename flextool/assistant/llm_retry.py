"""FlexTool's own retries of a model request (P1 §5); the provider libraries never retry (``max_retries=0``).

Retried, and only before any part of the answer was shown: a rate limit (:class:`RateLimited`) and an overloaded
service (:class:`ServerBusy`); a remote endpoint that could not be reached (:class:`Unreachable`) once after two
seconds.  A wait longer than ``longest_auto_wait`` is not waited for: the error is raised with the "long wait"
message.  While waiting, :class:`RetryWait` is pushed every second and the cancel event is honoured.

* **Rate limit**: at most ``max_retries`` retries (``free_plan_rate_retries`` with the free plan on), and all their
  waits together at most ``longest_auto_wait``.  The wait is what the provider names (``retry_after``), else the
  reset the answer's request-limit headers name when none remain, else the pacing interval, raised by the rate
  limit (:class:`~flextool.assistant.pacing.Pacer`).  When neither named a wait, the last message says that
  FlexTool now waits between requests (``rpm_paced``).
* **A reported limit of 0** (:class:`AccessInactive`, review G9 / OR-2) is not retried and leaves the pacing alone.
* **Overload**: at most ``max_retries`` retries, waiting ``retry_after`` or a doubling backoff with jitter.
* **Overload of a local or LAN model server** (``err.local``; the server is often loading or swapping a model):
  at most ``local_busy_retries`` retries, a doubling backoff from ``local_busy_backoff_start`` up to
  ``local_busy_backoff_cap`` per wait (``retry_after`` when named), and all these waits together at most
  ``local_busy_longest_wait``.

With a :class:`Pacing`, every attempt first waits for its turn (unless the backend paces itself, the window's
gate) and reports its outcome and the answer's request-limit headers to the pacer.
"""
from __future__ import annotations

import math
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from flextool.assistant.llm_adapter import OnEvent, Reply, RetryWait
from flextool.assistant.llm_errors import (
    MESSAGES,
    Cancelled,
    LLMError,
    RateLimited,
    ServerBusy,
    Unreachable,
    cancelled,
)
from flextool.assistant.pacing import PROCESS_PACER, Pacer


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3
    free_plan_rate_retries: int = 6
    backoff_start: float = 2.0
    backoff_cap: float = 60.0
    longest_auto_wait: float = 90.0
    jitter: float = 0.2
    connection_wait: float = 2.0
    #: A local or LAN model server that answers busy (calibration 2026-11-10): with jitter the eight waits are
    #: at most 1.2 × (2+4+8+16+32+60+60+60) ≈ 290 s, under the 300 s total.
    local_busy_retries: int = 8
    local_busy_backoff_start: float = 2.0
    local_busy_backoff_cap: float = 60.0
    local_busy_longest_wait: float = 300.0

    def backoff(self, n: int, rng: random.Random, *, local: bool = False) -> float:
        """The wait before retry *n* (1-based) when the provider names none; *local*: a local server's backoff."""
        start, cap = ((self.local_busy_backoff_start, self.local_busy_backoff_cap) if local
                      else (self.backoff_start, self.backoff_cap))
        base = min(cap, start * 2 ** (n - 1))
        return max(0.0, base * (1.0 + rng.uniform(-self.jitter, self.jitter)))


def _message(key: str, err: LLMError, **extra: object) -> str:
    fields = {**err.fields, "provider": err.provider or err.fields.get("provider", ""), **extra}
    return MESSAGES[key].format(**fields)


@dataclass
class Pacing:
    """How one session paces its requests: the *pacer* and the *key* (provider and model), whether the waits
    happen here (*here*; ``False`` when the backend waits inside the window's gate), the label the countdown names,
    the free plan, and *headers()*, the request-limit headers of the last answer (``None`` when the backend has
    none)."""

    key: str
    pacer: Pacer = field(default_factory=lambda: PROCESS_PACER)
    here: bool = True
    label: str = ""
    free_plan: bool = False
    headers: Callable[[], dict[str, str] | None] = lambda: None


def call_with_retries(fn: Callable[[], Reply], policy: RetryPolicy, on_event: OnEvent, cancel: threading.Event, *,
                      sleep: Callable[[float], bool] | None = None, clock: Callable[[], float] = time.monotonic,
                      rng: random.Random | None = None,
                      on_rejected: Callable[[LLMError], None] | None = None,
                      pacing: Pacing | None = None) -> Reply:
    """``fn()``, retried by *policy*.  *sleep(seconds)* waits and returns True when cancelled (default
    ``cancel.wait``); *clock* gives the time for the countdown; *on_rejected* is told of every rate-limit or
    overload answer that came before any output (the usage ledger counts them); *pacing* paces the attempts
    (without it, a private pacer that starts unpaced)."""
    wait_for = cancel.wait if sleep is None else sleep
    rng = rng or random.Random()
    pacing = pacing or Pacing(key="", pacer=Pacer())
    rate_retries = policy.free_plan_rate_retries if pacing.free_plan else policy.max_retries
    retries = 0
    rate_waited = 0.0
    busy_waited = 0.0
    connection_retried = False
    while True:
        if cancel.is_set():
            raise cancelled()
        epoch: int | None = None
        if pacing.here:
            epoch = pacing.pacer.wait(pacing.key, cancel, on_event, label=pacing.label, sleep=sleep)
        try:
            reply = fn()
        except LLMError as err:
            if err.streamed or cancel.is_set():
                if cancel.is_set() and not err.streamed and not isinstance(err, Cancelled):
                    raise cancelled(partial_usage=err.partial_usage) from err
                raise
            if isinstance(err, RateLimited):
                if on_rejected is not None:
                    on_rejected(err)
                if err.retry_after is not None and err.retry_after > policy.longest_auto_wait:
                    pacing.pacer.rate_limited(pacing.key, err.retry_after, _headers_of(err, pacing))
                    err.user_message = _message("rpm_long", err, minutes=max(1, math.ceil(err.retry_after / 60)))
                    err.args = (err.user_message,)
                    raise
                headers = _headers_of(err, pacing)
                from_headers = err.retry_after is None and pacing.pacer.header_wait(headers) is not None
                wait = pacing.pacer.rate_limited(pacing.key, err.retry_after, headers,
                                                 epoch=epoch if pacing.here else err.pace_epoch)
                if retries >= rate_retries or rate_waited + wait > policy.longest_auto_wait:
                    # A wait named by the provider or its headers counts requests per minute: "rpm" says so.
                    if err.retry_after is None and not from_headers:
                        err.user_message = _message("rpm_paced", err)
                        err.args = (err.user_message,)
                        err.message_key = "rpm_paced"
                    raise
                retries += 1
                rate_waited += wait
                key = "rpm_retry"
                attempt, of = retries, rate_retries
            elif isinstance(err, ServerBusy):
                if on_rejected is not None:
                    on_rejected(err)
                longest = policy.local_busy_longest_wait if err.local else policy.longest_auto_wait
                if err.retry_after is not None and err.retry_after > longest:
                    err.user_message = _message("rpm_long", err, minutes=max(1, math.ceil(err.retry_after / 60)))
                    err.args = (err.user_message,)
                    raise
                busy_retries = policy.local_busy_retries if err.local else policy.max_retries
                if retries >= busy_retries:
                    raise
                wait = (err.retry_after if err.retry_after is not None
                        else policy.backoff(retries + 1, rng, local=err.local))
                if err.local and busy_waited + wait > policy.local_busy_longest_wait:
                    raise
                retries += 1
                busy_waited += wait
                key = "busy_local_retry" if err.local else "busy_retry"
                attempt, of = retries, busy_retries
            elif isinstance(err, Unreachable) and not err.local and not connection_retried:
                connection_retried = True
                wait = policy.connection_wait
                key = "connection_retry"
                attempt, of = 1, 1
            else:
                raise
            _count_down(wait, err, key, attempt, of, on_event, wait_for, clock)
            continue
        pacing.pacer.succeeded(pacing.key, pacing.headers())
        return reply


def _headers_of(err: LLMError, pacing: Pacing) -> dict[str, str] | None:
    """The request-limit headers of the rate-limit answer *err*: its own, else the backend's last ones.  The
    backend's are taken either way, so they do not count for a later answer."""
    taken = pacing.headers()
    return err.rate_headers or taken


def _count_down(wait: float, err: LLMError, key: str, attempt: int, of: int, on_event: OnEvent,
                sleep: Callable[[float], bool], clock: Callable[[], float]) -> None:
    deadline = clock() + max(0.0, wait)
    while True:
        left = deadline - clock()
        if left <= 0:
            return
        seconds = math.ceil(left)
        on_event(RetryWait(seconds, attempt, of, _message(key, err, seconds=seconds, n=attempt, max=of)))
        if sleep(min(1.0, left)):
            raise cancelled() from err
