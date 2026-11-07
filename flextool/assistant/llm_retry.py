"""FlexTool's own retries of a model request (P1 §5); the provider libraries never retry (``max_retries=0``).

Retried, and only before any part of the answer was shown: a rate limit (:class:`RateLimited`) and an overloaded
service (:class:`ServerBusy`), at most ``max_retries`` times, waiting what the provider asks (``retry_after``) or a
doubling backoff with jitter; a remote endpoint that could not be reached (:class:`Unreachable`) once after two
seconds.  A wait longer than ``longest_auto_wait`` is not waited for: the error is raised with the "long wait"
message.  While waiting, :class:`RetryWait` is pushed every second and the cancel event is honoured.
"""
from __future__ import annotations

import math
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

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


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3
    backoff_start: float = 2.0
    backoff_cap: float = 60.0
    longest_auto_wait: float = 90.0
    jitter: float = 0.2
    connection_wait: float = 2.0

    def backoff(self, n: int, rng: random.Random) -> float:
        """The wait before retry *n* (1-based) when the provider names none."""
        base = min(self.backoff_cap, self.backoff_start * 2 ** (n - 1))
        return max(0.0, base * (1.0 + rng.uniform(-self.jitter, self.jitter)))


def _message(key: str, err: LLMError, **extra: object) -> str:
    fields = {**err.fields, "provider": err.provider or err.fields.get("provider", ""), **extra}
    return MESSAGES[key].format(**fields)


def call_with_retries(fn: Callable[[], Reply], policy: RetryPolicy, on_event: OnEvent, cancel: threading.Event, *,
                      sleep: Callable[[float], bool] | None = None, clock: Callable[[], float] = time.monotonic,
                      rng: random.Random | None = None,
                      on_rejected: Callable[[LLMError], None] | None = None) -> Reply:
    """``fn()``, retried by *policy*.  *sleep(seconds)* waits and returns True when cancelled (default
    ``cancel.wait``); *clock* gives the time for the countdown; *on_rejected* is told of every rate-limit or
    overload answer that came before any output (the usage ledger counts them)."""
    wait_for = cancel.wait if sleep is None else sleep
    rng = rng or random.Random()
    retries = 0
    connection_retried = False
    while True:
        if cancel.is_set():
            raise cancelled()
        try:
            return fn()
        except LLMError as err:
            if err.streamed or cancel.is_set():
                if cancel.is_set() and not err.streamed and not isinstance(err, Cancelled):
                    raise cancelled(partial_usage=err.partial_usage) from err
                raise
            if isinstance(err, (RateLimited, ServerBusy)):
                if on_rejected is not None:
                    on_rejected(err)
                if err.retry_after is not None and err.retry_after > policy.longest_auto_wait:
                    err.user_message = _message("rpm_long", err, minutes=max(1, math.ceil(err.retry_after / 60)))
                    err.args = (err.user_message,)
                    raise
                if retries >= policy.max_retries:
                    raise
                retries += 1
                wait = err.retry_after if err.retry_after is not None else policy.backoff(retries, rng)
                key = "rpm_retry" if isinstance(err, RateLimited) else "busy_retry"
                attempt, of = retries, policy.max_retries
            elif isinstance(err, Unreachable) and not err.local and not connection_retried:
                connection_retried = True
                wait = policy.connection_wait
                key = "connection_retry"
                attempt, of = 1, 1
            else:
                raise
            _count_down(wait, err, key, attempt, of, on_event, wait_for, clock)


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
