"""How fast FlexTool sends model requests to one provider and model (P1 §5.1).

One question may take several requests — one for each round of tool calls — and the tool loop sends them back to
back, so a provider that limits requests per second (Mistral's free plan) refuses some of them even for a single
question.  :class:`Pacer` keeps, per provider and model (:func:`pace_key`), the earliest moment the next request may
start, from these sources in this order of precedence:

1. **headers**: the request-limit headers of the last answer (:data:`RATE_HEADERS`; OpenAI's
   ``x-ratelimit-remaining-requests`` / ``x-ratelimit-reset-requests`` and Anthropic's
   ``anthropic-ratelimit-requests-remaining`` / ``-reset``): the remaining requests are spread over the time to the
   reset, and none remaining waits for the reset.  They count while that reset lies ahead;
2. **learned**: after a rate-limit answer without a wait named by the provider (no ``retry-after``), the interval
   between request starts is doubled (first :data:`START` seconds, at most :data:`CAP`); after each answered
   request it shrinks by :data:`DECAY`, never below :data:`START` (a per-second limit does not go away, and 1.1
   seconds between request starts is rarely noticed).  The learned interval is kept in the model-facts cache
   (:func:`flextool.assistant.probes.store_interval`), so the next session starts paced;
3. **seed**: the preset's documented free-plan interval (:attr:`~flextool.assistant.providers.Preset.
   free_plan_interval`), only while the free plan is on;
4. none: no wait.

A wait named by the provider (``retry-after``) is honoured by the retry countdown (:mod:`flextool.assistant.
llm_retry`) and does not change the interval.  :meth:`Pacer.wait` is atomic across threads: the request that passes
marks its start, so two chats of the FlexTool window that share a provider and model (the gate's pacer,
:data:`PROCESS_PACER`) also share one interval.  Standard library only.
"""
from __future__ import annotations

import math
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from flextool.assistant.llm_adapter import OnEvent, RetryWait
from flextool.assistant.llm_errors import MESSAGES, cancelled

#: The first learned interval, the factor after each rate limit, the largest interval, and the factor after each
#: answered request (seconds between request starts).
START = 1.1
FACTOR = 2.0
CAP = 20.0
DECAY = 0.8
#: A header-named wait longer than this is not waited for: the request goes and the provider's answer says why.
LONGEST_HEADER_WAIT = 90.0
#: A learned interval in the cache older than this is not used.
STORED_MAX_AGE = 7 * 24 * 3600.0
#: How often a waiting request looks again and pushes its countdown.
POLL = 1.0

#: The response headers FlexTool reads for pacing (and nothing else).  They are read only, never logged; they
#: hold counts and times, not keys.
RATE_HEADERS = frozenset({
    "retry-after", "retry-after-ms",
    "x-ratelimit-remaining-requests", "x-ratelimit-reset-requests",                 # OpenAI and compatible
    "anthropic-ratelimit-requests-remaining", "anthropic-ratelimit-requests-reset",  # Anthropic
})

_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")


def pace_key(origin: tuple[str, str, str]) -> str:
    """``preset|endpoint|model`` of a backend's origin, the endpoint in one spelling (the model-facts cache key)."""
    from flextool.assistant.net import normalise_endpoint

    preset_id, base_url, model = origin
    try:
        endpoint = normalise_endpoint(base_url)
    except ValueError:
        endpoint = base_url
    return f"{preset_id}|{endpoint}|{model}"


def rate_headers(headers: Mapping[str, Any] | Any) -> dict[str, str]:
    """The :data:`RATE_HEADERS` of *headers* (lower-case names), as text."""
    out: dict[str, str] = {}
    for name in RATE_HEADERS:
        value = headers.get(name)
        if value is not None:
            out[name] = str(value)
    return out


def _duration(text: str) -> float | None:
    """Seconds of an OpenAI-style duration (``1s``, ``6m0s``, ``20ms``, ``2m59.56s``) or a plain number."""
    text = text.strip()
    try:
        value = float(text)
    except ValueError:
        parts = _DURATION.findall(text)
        if not parts or "".join(n + u for n, u in parts) != text:
            return None
        value = sum(float(n) * {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[u] for n, u in parts)
    return value if math.isfinite(value) and value >= 0 else None


def _timestamp(text: str, wall: float) -> float | None:
    """Seconds from *wall* to an RFC 3339 time (Anthropic's reset headers)."""
    try:
        when = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if when.tzinfo is None:
        return None
    return max(0.0, when.timestamp() - wall)


def requests_left(headers: Mapping[str, str], wall: float) -> tuple[int, float] | None:
    """(requests remaining, seconds to the reset) from the request-limit headers, ``None`` when there are none."""
    for remaining_name, reset_name, parse in (
            ("x-ratelimit-remaining-requests", "x-ratelimit-reset-requests", _duration),
            ("anthropic-ratelimit-requests-remaining", "anthropic-ratelimit-requests-reset",
             lambda text: _timestamp(text, wall))):
        remaining, reset = headers.get(remaining_name), headers.get(reset_name)
        if remaining is None or reset is None:
            continue
        try:
            left = int(float(remaining))
        except ValueError:
            continue
        seconds = parse(reset)
        if seconds is not None:
            return max(0, left), seconds
    return None


@dataclass
class _State:
    seed: float | None = None
    learned: float | None = None
    last_start: float | None = None
    header_spacing: float | None = None     # from the headers: seconds between request starts
    header_ready: float | None = None       # from the headers: no request before this (none left until the reset)
    header_until: float = -math.inf         # the headers count until this time (their reset)
    loaded: bool = False


@dataclass
class Pacer:
    """The pacing of this process (see the module text).  *persist* keeps learned intervals in the user cache."""

    clock: Callable[[], float] = time.monotonic
    wall: Callable[[], float] = time.time
    persist: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _states: dict[str, _State] = field(default_factory=dict, repr=False)

    def _state(self, key: str) -> _State:
        state = self._states.get(key)
        if state is None:
            state = self._states[key] = _State()
        if not state.loaded:
            state.loaded = True
            if self.persist:
                from flextool.assistant import probes

                stored = probes.stored_interval(key, max_age=STORED_MAX_AGE, now=self.wall())
                if stored is not None and state.learned is None:
                    state.learned = min(CAP, max(START, stored))
        return state

    def reset(self) -> None:
        """Forget every key (tests)."""
        with self._lock:
            self._states.clear()

    def prepare(self, key: str, seed: float | None) -> None:
        """Set *key*'s seed (``None``: the free plan is off or the preset has none) and read its learned interval."""
        with self._lock:
            self._state(key).seed = seed

    def interval(self, key: str) -> float:
        """The spacing of request starts without header information: learned, else seed, else 0."""
        with self._lock:
            return self._spacing(self._state(key))

    @staticmethod
    def _spacing(state: _State) -> float:
        if state.learned is not None:
            return state.learned
        return state.seed or 0.0

    def _not_before(self, state: _State, now: float) -> float:
        if now < state.header_until:
            if state.header_ready is not None:
                return state.header_ready
            spacing = state.header_spacing or 0.0
        else:
            spacing = self._spacing(state)
        return now if state.last_start is None else state.last_start + spacing

    def raised(self, key: str) -> float:
        """The interval :meth:`rate_limited` would set for a rate limit without a named wait."""
        with self._lock:
            return self._raised(self._state(key))

    @staticmethod
    def _raised(state: _State) -> float:
        base = state.learned if state.learned is not None else state.seed
        return START if not base else min(CAP, max(START, base * FACTOR))

    def wait(self, key: str, cancel: threading.Event, on_event: OnEvent | None = None, *, label: str = "",
             sleep: Callable[[float], bool] | None = None) -> None:
        """Wait until a request to *key* may start, then mark its start.  Pushes a countdown (:class:`RetryWait`,
        attempt 0) for a wait of a second or more; *sleep(seconds)* returns True when cancelled (default
        ``cancel.wait``)."""
        pause = cancel.wait if sleep is None else sleep
        shown: int | None = None
        while True:
            if cancel.is_set():
                raise cancelled()
            with self._lock:
                state = self._state(key)
                now = self.clock()
                left = self._not_before(state, now) - now
                if left <= 0:
                    state.last_start = now
                    return
            seconds = math.ceil(left)
            if on_event is not None and left >= 1.0 and seconds != shown:
                shown = seconds
                text = MESSAGES["pace_wait"].format(provider=label or "The AI service", seconds=seconds)
                on_event(RetryWait(seconds, 0, 0, text))
            if pause(min(POLL, left)):
                raise cancelled()

    def succeeded(self, key: str, headers: Mapping[str, str] | None = None) -> None:
        """An answered request: the learned interval shrinks; the answer's request-limit headers count."""
        store: float | None = None
        with self._lock:
            state = self._state(key)
            if state.learned is not None and state.learned > START:
                state.learned = max(START, state.learned * DECAY)
                store = state.learned
            self._observe(state, headers)
        if store is not None:
            self._store(key, store)

    def rate_limited(self, key: str, retry_after: float | None, headers: Mapping[str, str] | None = None) -> float:
        """A rate-limit answer before any output: the wait before the retry.  With *retry_after* that wait;
        without, the interval is raised and is the wait."""
        store: float | None = None
        with self._lock:
            state = self._state(key)
            state.header_spacing = state.header_ready = None
            state.header_until = -math.inf
            if retry_after is not None:
                return retry_after
            state.learned = store = self._raised(state)
        self._store(key, store)
        return store

    def _observe(self, state: _State, headers: Mapping[str, str] | None) -> None:
        if not headers:
            return
        found = requests_left(headers, self.wall())
        if found is None:
            return
        left, reset = found
        now = self.clock()
        state.header_until = now + reset
        if left <= 0 and reset <= LONGEST_HEADER_WAIT:
            state.header_spacing, state.header_ready = None, now + reset
        else:
            # A reset too far ahead is not waited for: the request goes, and the provider's answer says why.
            state.header_spacing, state.header_ready = (reset / left if left > 0 else 0.0), None

    def _store(self, key: str, seconds: float) -> None:
        if self.persist:
            from flextool.assistant import probes

            probes.store_interval(key, seconds, now=self.wall())


#: The pacing of this process: the terminal chat's and, through the window's gate, every chat tab's.
PROCESS_PACER = Pacer(persist=True)
