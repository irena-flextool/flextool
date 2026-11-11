"""How fast FlexTool sends model requests to one provider and model (P1 §5.1).

One question may take several requests — one for each round of tool calls — and the tool loop sends them back to
back, so a provider that limits requests per second (Mistral's free plan) refuses some of them even for a single
question.  :class:`Pacer` keeps, per provider and model (:func:`pace_key`), the earliest moment the next request may
start, from these sources in this order of precedence:

1. **headers**: the request-limit headers of the last answer (:data:`RATE_HEADERS`; OpenAI's
   ``x-ratelimit-remaining-requests`` / ``x-ratelimit-reset-requests``, Anthropic's
   ``anthropic-ratelimit-requests-remaining`` / ``-reset``, and Mistral's ``x-ratelimit-remaining-req-minute`` /
   ``x-ratelimit-limit-req-minute``, which name no reset: a 60-second window is assumed): the remaining requests
   are spread over the time to the reset, and none remaining waits for the reset.  They count while that reset
   lies ahead, and only ever lengthen the spacing: it is never shorter than the learned interval or the seed below
   (review G9).  A rate-limit answer whose headers say none remain waits for their reset and learns nothing
   (C4-m1).  A reported limit of 0 is not a pace at all: the account's API access is not active
   (:class:`~flextool.assistant.llm_errors.AccessInactive`, OR-2);
2. **learned**: after a rate-limit answer without a wait named by the provider (no ``retry-after``), the interval
   between request starts is doubled (first :data:`START` seconds, at most :data:`CAP`); after each answered
   request it shrinks by :data:`DECAY`, never below :data:`START` (a per-second limit does not go away, and 1.1
   seconds between request starts is rarely noticed) nor below the seed.  The learned interval is kept in the model-facts cache
   (:func:`flextool.assistant.probes.store_interval`), so the next session starts paced;
3. **seed** (:func:`flextool.assistant.providers.pacing_seed`): the preset's interval for every request
   (:attr:`~flextool.assistant.providers.Preset.min_interval`, Mistral's 1.1 s), and while the free plan is on its
   documented free-plan interval (:attr:`~flextool.assistant.providers.Preset.free_plan_interval`, OpenRouter's
   free models); the learned interval never goes below it;
4. none: no wait.

A **floor** (:meth:`Pacer.set_floor`, the calibration's ``run --pace``) is a least spacing of request starts for
one key in this process: the spacing is the largest of the floor and the sources above.  It is not learned, not
kept in the cache and never changes the learned interval.

A wait named by the provider (``retry-after``) is honoured by the retry countdown (:mod:`flextool.assistant.
llm_retry`) and does not change the interval.  :meth:`Pacer.wait` is atomic across threads: the request that passes
marks its start, so two chats of the FlexTool window that share a provider and model (the gate's pacer,
:data:`PROCESS_PACER`) also share one interval.  It returns the key's *epoch*: rate-limit answers to requests
that started in one epoch raise the interval once (two chats refused for one burst double it once), and each raise
starts a new epoch.

The model-facts cache is read in :meth:`Pacer.prepare`, outside the pacer's lock (a key never prepared starts
without a learned interval), and written on a raise and, while the interval shrinks, only when it reaches its
floor or :data:`STORE_EVERY` seconds after the key's last write; the writes of one process go in order (review
G9, C4-n5).  Standard library only.
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
from flextool.assistant.llm_errors import LIMIT_MINUTE, MESSAGES, REMAINING_MINUTE, cancelled

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
#: While the learned interval shrinks, it is written to the cache at most this often (wall seconds), and when it
#: reaches its floor.
STORE_EVERY = 60.0

#: The response headers FlexTool reads for pacing (and nothing else).  They are read only, never logged; they
#: hold counts and times, not keys.
RATE_HEADERS = frozenset({
    "retry-after", "retry-after-ms",
    "x-ratelimit-remaining-requests", "x-ratelimit-reset-requests",                 # OpenAI and compatible
    "anthropic-ratelimit-requests-remaining", "anthropic-ratelimit-requests-reset",  # Anthropic
    LIMIT_MINUTE, REMAINING_MINUTE,                                                  # Mistral
})
#: The window of Mistral's per-minute headers, which name no reset.
MINUTE_WINDOW = 60.0

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


def _count(text: str) -> int | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return int(value) if math.isfinite(value) else None


def requests_left(headers: Mapping[str, str], wall: float) -> tuple[int, float] | None:
    """(requests remaining, seconds to the reset) from the request-limit headers, ``None`` when there are none.
    Mistral's per-minute pair names no reset (:data:`MINUTE_WINDOW`); with a reported limit of 0 it is not read
    (that is no pace: :class:`~flextool.assistant.llm_errors.AccessInactive`)."""
    for remaining_name, reset_name, parse in (
            ("x-ratelimit-remaining-requests", "x-ratelimit-reset-requests", _duration),
            ("anthropic-ratelimit-requests-remaining", "anthropic-ratelimit-requests-reset",
             lambda text: _timestamp(text, wall))):
        remaining, reset = headers.get(remaining_name), headers.get(reset_name)
        if remaining is None or reset is None:
            continue
        left = _count(remaining)
        if left is None:
            continue
        seconds = parse(reset)
        if seconds is not None:
            return max(0, left), seconds
    remaining, limit = headers.get(REMAINING_MINUTE), headers.get(LIMIT_MINUTE)
    if remaining is not None:
        left = _count(remaining)
        if left is not None and (limit is None or _count(limit) != 0):
            return max(0, left), MINUTE_WINDOW
    return None


def header_wait(headers: Mapping[str, str] | None, wall: float) -> float | None:
    """The wait the request-limit headers of a rate-limit answer name: the reset, when no request remains and the
    reset is at most :data:`LONGEST_HEADER_WAIT` away; else ``None``."""
    found = requests_left(headers, wall) if headers else None
    if found is None:
        return None
    left, reset = found
    return reset if left <= 0 and reset <= LONGEST_HEADER_WAIT else None


@dataclass
class _State:
    seed: float | None = None
    learned: float | None = None
    last_start: float | None = None
    header_spacing: float | None = None     # from the headers: seconds between request starts
    header_ready: float | None = None       # from the headers: no request before this (none left until the reset)
    header_until: float = -math.inf         # the headers count until this time (their reset)
    loaded: bool = False
    stored_at: float | None = None          # wall time of the last cache write (or read) of the learned interval
    epoch: int = 0                          # raised by every raise of the learned interval
    floor: float | None = None              # set_floor: no spacing below this (not learned, not stored)


@dataclass
class Pacer:
    """The pacing of this process (see the module text).  *persist* keeps learned intervals in the user cache."""

    clock: Callable[[], float] = time.monotonic
    wall: Callable[[], float] = time.time
    persist: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _store_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _states: dict[str, _State] = field(default_factory=dict, repr=False)

    def _state(self, key: str) -> _State:
        state = self._states.get(key)
        if state is None:
            state = self._states[key] = _State()
        return state

    def reset(self) -> None:
        """Forget every key (tests)."""
        with self._lock:
            self._states.clear()

    def prepare(self, key: str, seed: float | None) -> None:
        """Set *key*'s seed (``None``: the free plan is off or the preset has none) and, the first time, read its
        learned interval from the cache (outside the lock: the file may be slow or locked by another process)."""
        stored: float | None = None
        with self._lock:
            loaded = self._state(key).loaded
        if self.persist and not loaded:
            from flextool.assistant import probes

            stored = probes.stored_interval(key, max_age=STORED_MAX_AGE, now=self.wall())
        with self._lock:
            state = self._state(key)
            state.seed = seed
            if not state.loaded:
                state.loaded = True
                if stored is not None and state.learned is None:
                    state.learned = min(CAP, max(START, stored))
                    state.stored_at = self.wall()

    def set_floor(self, key: str, seconds: float | None) -> None:
        """No two request starts to *key* closer than *seconds* in this process (``None`` or 0: no floor)."""
        with self._lock:
            self._state(key).floor = seconds if seconds and seconds > 0 else None

    def interval(self, key: str) -> float:
        """The spacing of request starts without header information: learned, else seed, else 0."""
        with self._lock:
            return self._spacing(self._state(key))

    @staticmethod
    def _spacing(state: _State) -> float:
        if state.learned is not None:
            return max(state.learned, state.seed or 0.0, state.floor or 0.0)
        return max(state.seed or 0.0, state.floor or 0.0)

    def _not_before(self, state: _State, now: float) -> float:
        # The headers only lengthen the spacing: never below the learned interval or the seed (review G9).
        spacing = self._spacing(state)
        if now < state.header_until:
            if state.header_ready is not None:
                return state.header_ready if state.last_start is None else max(state.header_ready,
                                                                              state.last_start + spacing)
            spacing = max(state.header_spacing or 0.0, spacing)
        return now if state.last_start is None else state.last_start + spacing

    def raised(self, key: str, epoch: int | None = None) -> float:
        """The interval :meth:`rate_limited` would set for a rate limit without a named wait to a request that
        started in *epoch* (an earlier epoch: the interval as it is, already raised for that burst)."""
        with self._lock:
            state = self._state(key)
            return self._spacing(state) if self._stale(state, epoch) else self._raised(state)

    @staticmethod
    def _stale(state: _State, epoch: int | None) -> bool:
        return epoch is not None and epoch != state.epoch

    @staticmethod
    def _raised(state: _State) -> float:
        base = state.learned if state.learned is not None else state.seed
        return START if not base else min(CAP, max(START, base * FACTOR))

    def wait(self, key: str, cancel: threading.Event, on_event: OnEvent | None = None, *, label: str = "",
             sleep: Callable[[float], bool] | None = None) -> int:
        """Wait until a request to *key* may start, then mark its start; returns the epoch it started in (for
        :meth:`rate_limited`).  Pushes a countdown (:class:`RetryWait`, attempt 0) for a wait of a second or more;
        *sleep(seconds)* returns True when cancelled (default ``cancel.wait``)."""
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
                    return state.epoch
            seconds = math.ceil(left)
            if on_event is not None and left >= 1.0 and seconds != shown:
                shown = seconds
                text = MESSAGES["pace_wait"].format(provider=label or "The AI service", seconds=seconds)
                on_event(RetryWait(seconds, 0, 0, text))
            if pause(min(POLL, left)):
                raise cancelled()

    def succeeded(self, key: str, headers: Mapping[str, str] | None = None) -> None:
        """An answered request: the learned interval shrinks; the answer's request-limit headers count.  The
        shrunk interval is written when it reaches its floor or :data:`STORE_EVERY` after the last write."""
        store = False
        with self._lock:
            state = self._state(key)
            floor = max(START, state.seed or 0.0)
            if state.learned is not None and state.learned > floor:
                state.learned = max(floor, state.learned * DECAY)
                store = state.learned <= floor or state.stored_at is None \
                    or self.wall() - state.stored_at >= STORE_EVERY
            self._observe(state, headers)
        if store:
            self._persist(key)

    def header_wait(self, headers: Mapping[str, str] | None) -> float | None:
        """The wait the headers of a rate-limit answer name (:func:`header_wait`); changes nothing."""
        return header_wait(headers, self.wall())

    def rate_limited(self, key: str, retry_after: float | None, headers: Mapping[str, str] | None = None, *,
                     epoch: int | None = None) -> float:
        """A rate-limit answer before any output: the wait before the retry.  With *retry_after* that wait; else
        the reset its *headers* name when none remain (the interval is left alone, C4-m1); else the interval is
        raised and is the wait — once per *epoch* (:meth:`wait`'s): a request that started before the last raise
        waits the interval as it is."""
        with self._lock:
            state = self._state(key)
            state.header_spacing = state.header_ready = None
            state.header_until = -math.inf
            if retry_after is not None:
                return retry_after
            reset = header_wait(headers, self.wall())
            if reset is not None:
                now = self.clock()
                state.header_ready = state.header_until = now + reset
                return reset
            if self._stale(state, epoch):
                return self._spacing(state)
            state.learned = raised = self._raised(state)
            state.epoch += 1
        self._persist(key)
        return raised

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

    def _persist(self, key: str) -> None:
        """Write *key*'s learned interval as it is now.  One write at a time, the value read just before it, so
        the newest value is the one on disk."""
        if not self.persist:
            return
        from flextool.assistant import probes

        with self._store_lock:
            with self._lock:
                state = self._states.get(key)
                seconds = state.learned if state is not None else None
                if seconds is None:
                    return
                state.stored_at = now = self.wall()
            probes.store_interval(key, seconds, now=now)


#: The pacing of this process: the terminal chat's and, through the window's gate, every chat tab's.
PROCESS_PACER = Pacer(persist=True)
