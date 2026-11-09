"""One gate per process for the model requests of all chat tabs (P2-D9, §5.5).

Several chats may run at once, but a free tier counts requests per account: three tabs that each send would
collect three rate-limit answers.  :class:`ProviderGate` lets at most ``parallel`` model requests (1–4, the
setting "Questions answered at the same time"; default 1) go to one provider at a time — the provider is P1's
ledger key — and shares a cool-down after a rate-limit or overload answer:

* a wait the provider asks for of at most ``longest_auto_wait`` (90 s) holds the other chats' requests until it
  has passed (the chat that got the answer keeps P1's own retry countdown and is not held);
* a longer wait is not waited for: until it has passed, another chat's request ends at once with P1's "long
  wait" message, without being sent (and without being counted as a rejected attempt).

A chat's backend may carry its own ``limit`` below ``parallel``: the free plan sends one request at a time
(P3 §4.6), so a free-plan chat's :class:`GatedBackend` has ``limit=1`` while ``parallel`` stays the setting
(review G18, C7-m4: one process-wide value let the last prepared chat decide for all).  The limit is compared with
the requests running to the whole provider, so a free-plan chat also waits while a paid chat of the same provider
runs, but a paid chat (``limit`` ``None``: ``parallel``) still enters beside a running free-plan one.

Slots go in arrival order, per provider: a waiting request that is held only because the provider is full keeps
every later request to that provider out, also one whose own limit would let it in, so two chats that keep the
provider busy cannot starve a third (a free-plan chat behind paid ones).  A request held by another chat's
cool-down does not hold the later ones (the chat that got the answer is not held by its own cool-down).

:class:`GatedBackend` wraps a chat's backend so that each model request takes a slot; tool calls run outside the
gate (they are serialised by the process-wide tool lock instead).  Holding its slot, a request then waits for its
turn by the gate's :class:`~flextool.assistant.pacing.Pacer` (P1 §5.1; the process's for :data:`PROCESS_GATE`), so
every chat of one provider and model shares one interval between request starts.  While a request waits, it pushes
:class:`GateWait` about once a second; the wait honours the chat's cancel event.  Standard library only.
"""
from __future__ import annotations

import math
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from flextool.assistant.llm_adapter import Backend, ModelEntry, OnEvent, Reply, Request
from flextool.assistant.llm_errors import MESSAGES, LLMError, RateLimited, ServerBusy, cancelled
from flextool.assistant.llm_retry import RetryPolicy
from flextool.assistant.pacing import PROCESS_PACER, Pacer, pace_key

#: How often a waiting request looks at the gate again.
POLL = 0.05


@dataclass(frozen=True)
class GateWait:
    """A model request waits for the gate: ``reason`` is ``"busy"`` (another chat is using the provider) or
    ``"cool_down"`` (the provider asked to slow down; ``seconds_left``)."""

    reason: str
    seconds_left: int | None


class GateLongWait(LLMError):
    """The provider asked another chat to wait longer than FlexTool waits by itself: nothing was sent."""

    code = "rpm"


@dataclass
class _Provider:
    active: int = 0
    #: The waiting requests in arrival order: ticket → the owner of the request.
    waiting: dict[int, object] = field(default_factory=dict)
    next_ticket: int = 0
    cool_until: float = 0.0
    cool_owner: object = None
    long_until: float = 0.0
    long_owner: object = None


@dataclass
class ProviderGate:
    """The process-wide gate.  ``parallel`` is read at every look at the gate, so a lowered setting also holds
    requests already waiting."""

    parallel: int = 1
    clock: Callable[[], float] = time.monotonic
    longest_auto_wait: float = RetryPolicy().longest_auto_wait
    pacer: Pacer = field(default_factory=Pacer)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _providers: dict[str, _Provider] = field(default_factory=dict, repr=False)

    def _provider(self, key: str) -> _Provider:
        return self._providers.setdefault(key, _Provider())

    def set_parallel(self, value: int) -> None:
        with self._lock:
            self.parallel = max(1, int(value))

    def acquire(self, key: str, limit: int | Callable[[], int] | None, cancel: threading.Event,
                on_wait: Callable[[GateWait], None] | None = None, *, owner: object = None,
                provider_label: str = "") -> None:
        """Take a slot for one request to *key*.  Waits while *limit* (default ``parallel``) requests are running,
        while a request that came earlier waits for a slot (arrival order), or while a short cool-down set by
        another *owner* lasts; raises :class:`GateLongWait` during a long one and
        :class:`~flextool.assistant.llm_errors.Cancelled` when *cancel* is set while waiting."""
        shown: tuple[str, int | None] | None = None
        last_shown = 0.0
        ticket: int | None = None
        try:
            while True:
                if cancel.is_set():
                    raise cancelled()
                with self._lock:
                    now = self.clock()
                    p = self._provider(key)
                    if p.long_until > now and p.long_owner is not owner:
                        raise _long_wait(provider_label, p.long_until - now)
                    if ticket is None:
                        ticket = p.next_ticket
                        p.next_ticket += 1
                        p.waiting[ticket] = owner
                    cap = self.parallel if limit is None else (limit() if callable(limit) else limit)
                    cooling = p.cool_until > now and p.cool_owner is not owner
                    if not cooling and p.active < max(1, int(cap)) and not self._ahead(p, ticket, now):
                        del p.waiting[ticket]
                        ticket = None
                        p.active += 1
                        return
                    wait = GateWait("cool_down", math.ceil(p.cool_until - now)) if cooling else GateWait("busy", None)
                moment = time.monotonic()
                if on_wait is not None and ((wait.reason, wait.seconds_left) != shown or moment - last_shown >= 1.0):
                    shown = (wait.reason, wait.seconds_left)
                    last_shown = moment
                    on_wait(wait)
                if cancel.wait(POLL):
                    raise cancelled()
        finally:
            if ticket is not None:
                with self._lock:
                    self._provider(key).waiting.pop(ticket, None)

    @staticmethod
    def _ahead(p: _Provider, ticket: int, now: float) -> bool:
        """Whether a request that arrived before *ticket* waits for a slot (not for another chat's cool-down)."""
        for earlier, owner in p.waiting.items():
            if earlier >= ticket:
                return False                # dicts keep insertion order: the tickets ascend
            if not (p.cool_until > now and p.cool_owner is not owner):
                return True
        return False

    def release(self, key: str) -> None:
        with self._lock:
            p = self._provider(key)
            p.active = max(0, p.active - 1)

    def cool_down(self, key: str, until: float, *, owner: object = None) -> None:
        """Hold the other chats' requests to *key* until *until* (the gate's clock)."""
        with self._lock:
            p = self._provider(key)
            if until > p.cool_until:
                p.cool_until = until
                p.cool_owner = owner

    def long_wait(self, key: str, until: float, *, owner: object = None) -> None:
        """Until *until*, the other chats' requests to *key* end at once with the "long wait" message."""
        with self._lock:
            p = self._provider(key)
            if until > p.long_until:
                p.long_until = until
                p.long_owner = owner

    def state(self, key: str) -> tuple[int, float, float]:
        """``(running requests, cool-down end, long-wait end)`` of *key* (for tests and the status)."""
        with self._lock:
            p = self._provider(key)
            return p.active, p.cool_until, p.long_until


#: The one gate of this process: every chat tab of every chat window uses it.
PROCESS_GATE = ProviderGate(pacer=PROCESS_PACER)


def _long_wait(label: str, seconds: float) -> GateLongWait:
    provider = label or "The AI service"
    minutes = max(1, math.ceil(seconds / 60))
    return GateLongWait(MESSAGES["rpm_long"].format(provider=provider, minutes=minutes), provider=provider,
                        retry_after=seconds, fields={"provider": provider, "minutes": minutes})


class GatedBackend:
    """A chat's backend whose model requests go through the :class:`ProviderGate` (implements P1's
    :class:`~flextool.assistant.llm_adapter.Backend`).  It paces its requests itself (``paces``), inside its
    slot, so the session leaves the waits to it."""

    paces = True

    def __init__(self, backend: Backend, gate: ProviderGate, key: str, *, label: str = "",
                 limit: int | Callable[[], int] | None = None, policy: RetryPolicy | None = None,
                 rng: random.Random | None = None) -> None:
        self.inner = backend
        self.gate = gate
        self.key = key
        self.label = label
        self.limit = limit
        self.policy = policy or RetryPolicy()
        self._rng = rng or random.Random()
        self._rejected_in_row = 0
        self.pace_key = pace_key(backend.origin)

    @property
    def pacer(self) -> Pacer:
        return self.gate.pacer

    @property
    def origin(self):  # noqa: ANN201 — the inner backend's Origin
        return self.inner.origin

    def stream(self, req: Request, on_event: OnEvent, cancel: threading.Event) -> Reply:
        self.gate.acquire(self.key, self.limit, cancel, on_event, owner=self, provider_label=self.label)
        epoch: int | None = None
        try:
            epoch = self.gate.pacer.wait(self.pace_key, cancel, on_event, label=self.label)
            reply = self.inner.stream(req, on_event, cancel)
        except (RateLimited, ServerBusy) as err:
            err.pace_epoch = epoch              # the retries raise the interval once per burst (review G9)
            if not err.streamed:
                self._rejected_in_row += 1
                self._hold_others(err)
            raise
        except LLMError:
            self._rejected_in_row = 0
            raise
        finally:
            self.gate.release(self.key)
        self._rejected_in_row = 0
        return reply

    def _hold_others(self, err: LLMError) -> None:
        wait = err.retry_after
        if wait is None and isinstance(err, RateLimited):
            # The reset the answer's own headers name when none remain (review G9, C4-m1), else the raised interval.
            wait = self.gate.pacer.header_wait(err.rate_headers)
            if wait is None:
                wait = self.gate.pacer.raised(self.pace_key, err.pace_epoch)
        if wait is None:
            wait = self.policy.backoff(self._rejected_in_row, self._rng)
        until = self.gate.clock() + max(0.0, wait)
        if wait <= self.policy.longest_auto_wait:
            self.gate.cool_down(self.key, until, owner=self)
        else:
            self.gate.long_wait(self.key, until, owner=self)

    def abort(self) -> None:
        self.inner.abort()

    def list_models(self) -> list[ModelEntry]:
        return self.inner.list_models()

    def close(self) -> None:
        self.inner.close()

    def __getattr__(self, name: str):  # noqa: ANN204 — e.g. the Anthropic backend's context_window()
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)
