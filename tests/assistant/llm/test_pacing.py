"""Pacing the model requests (P1 §5.1) against a fake rate-limiting server, on a fake clock.

The server is an ``httpx2.MockTransport`` behind the real Chat Completions backend (Mistral's and OpenAI's
presets): it answers 429 with or without ``retry-after``, or a streamed answer with request-limit headers, and
records the fake time of every request.  The expected waits are worked out by hand from the rules in
``flextool/assistant/pacing.py`` (start 1.1 s, ×2, at most 20 s, ×0.8 after an answer, never below 1.1 s).
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from flextool.assistant import llm_errors as E
from flextool.assistant import pacing, probes
from flextool.assistant.llm_adapter import Message, Reply, RetryWait, Usage, make_backend
from flextool.assistant.llm_retry import Pacing, RetryPolicy, call_with_retries
from flextool.assistant.pacing import Pacer, pace_key, requests_left
from flextool.assistant.providers import PRESETS
from tests.assistant.llm.fake_backend import FakeBackend
from tests.assistant.llm.wire import (
    Recorder,
    connection,
    json_response,
    openai_sse,
    payload,
    request,
    sse_response,
)

LIMITED = {"message": "Requests rate limit exceeded"}
NO_JITTER = RetryPolicy(jitter=0.0)
WALL = 1_800_000_000.0


class FakeTime:
    """The fake clock: only the waits move it."""

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> bool:
        self.sleeps.append(seconds)
        self.t += seconds
        return False


class Server:
    """The fake provider: answers in order and keeps the fake time of every request."""

    def __init__(self, clock: FakeTime, answers: list) -> None:
        self.clock = clock
        self.times: list[float] = []
        self.recorder = Recorder([self._timed(a) for a in answers])

    def _timed(self, answer):
        def respond(_request):
            self.times.append(self.clock.now())
            return answer() if callable(answer) else answer
        return respond


def ok(headers: dict | None = None):
    def make():
        response = sse_response(openai_sse(payload("openai", "text_usage")["chunks"]))
        response.headers.update(headers or {})
        return response
    return make


def limited(retry_after: str | None = None):
    return lambda: json_response(429, LIMITED, {"retry-after": retry_after} if retry_after else None)


def rig(answers: list, *, preset_id: str = "mistral", pacer: Pacer | None = None, clock: FakeTime | None = None):
    clock = clock or FakeTime()
    server = Server(clock, answers)
    backend = make_backend(PRESETS[preset_id], connection(preset_id), transport=server.recorder.transport)
    pacer = pacer or Pacer(clock=clock.now, wall=lambda: WALL)
    return clock, server, backend, pacer


def ask(clock: FakeTime, backend, pacer: Pacer, *, free_plan: bool = False, events: list | None = None):
    cancel = threading.Event()
    on_event = (events if events is not None else []).append
    key = pace_key(backend.origin)
    return call_with_retries(lambda: backend.stream(request([Message("user", "hi")]), on_event, cancel),
                             NO_JITTER, on_event, cancel, sleep=clock.sleep, clock=clock.now,
                             pacing=Pacing(key, pacer, label="Mistral", free_plan=free_plan,
                                           headers=backend.rate_headers))


# --- after a rate limit -----------------------------------------------------------------------------------------
def test_retry_after_is_honoured_and_leaves_the_interval():
    clock, server, backend, pacer = rig([limited("4"), ok(), ok()])
    ask(clock, backend, pacer)
    assert server.times == [0.0, 4.0]
    assert pacer.interval(pace_key(backend.origin)) == 0.0          # a named wait is not a speed limit
    ask(clock, backend, pacer)
    assert server.times[-1] == 4.0                                   # nothing learned: no wait


def test_without_retry_after_the_interval_doubles_and_paces_what_follows():
    clock, server, backend, pacer = rig([limited(), limited(), ok(), ok()])
    events: list = []
    ask(clock, backend, pacer, events=events)
    # 429 → 1.1 s; 429 → 2.2 s; answered, so 2.2 × 0.8 = 1.76 s.
    assert server.times == pytest.approx([0.0, 1.1, 3.3])
    key = pace_key(backend.origin)
    assert pacer.interval(key) == pytest.approx(1.76)
    ask(clock, backend, pacer)
    assert server.times[-1] == pytest.approx(3.3 + 1.76)
    first = next(e for e in events if isinstance(e, RetryWait))
    assert first.reason == ("Mistral is limiting how often requests can be sent. Trying again in 2 seconds "
                            "(attempt 1 of 3).")


def test_the_retries_give_up_with_what_happened():
    clock, server, backend, pacer = rig([limited() for _ in range(4)])
    with pytest.raises(E.RateLimited) as info:
        ask(clock, backend, pacer)
    assert len(server.times) == 4                                    # three retries off the free plan
    assert info.value.user_message == ("Mistral limits how many requests per second a free plan may send. "
                                       "FlexTool now waits between requests; ask again.")


def test_the_free_plan_tries_six_times_within_ninety_seconds():
    clock, server, backend, pacer = rig([limited() for _ in range(6)] + [ok()])
    ask(clock, backend, pacer, free_plan=True)
    # Waits 1.1, 2.2, 4.4, 8.8, 17.6, 20 (the cap): 54.1 s, within 90 s.
    gaps = [b - a for a, b in zip(server.times, server.times[1:])]
    assert gaps == pytest.approx([1.1, 2.2, 4.4, 8.8, 17.6, 20.0])
    clock, server, backend, pacer = rig([limited() for _ in range(8)])
    with pytest.raises(E.RateLimited):
        ask(clock, backend, pacer, free_plan=True)
    assert len(server.times) == 7                                    # six retries at most


def test_answered_requests_shrink_the_interval_to_its_floor():
    p = Pacer(clock=lambda: 0.0)
    for expected in (1.1, 2.2, 4.4):
        assert p.rate_limited("k", None) == pytest.approx(expected)
    for expected in (3.52, 2.816, 2.2528, 1.80224, 1.441792, 1.1534336, 1.1, 1.1):
        p.succeeded("k")
        assert p.interval("k") == pytest.approx(expected)
    for _ in range(10):
        p.rate_limited("k", None)
    assert p.interval("k") == pacing.CAP


# --- what starts the pacing -------------------------------------------------------------------------------------
def test_the_learned_interval_is_kept_for_the_next_session(user_cache_in_tmp):
    key = "mistral|https://api.mistral.ai/v1|example-model"
    first = Pacer(clock=lambda: 0.0, wall=lambda: WALL, persist=True)
    first.rate_limited(key, None)
    first.rate_limited(key, None)
    # A probe of the same model afterwards keeps the interval beside its facts.
    probes.remember_facts("mistral", "https://api.mistral.ai/v1", "example-model", probes.ModelFacts(tools=True),
                          clock=lambda: WALL)
    later = Pacer(clock=lambda: 0.0, wall=lambda: WALL + 3600, persist=True)
    assert later.interval(key) == pytest.approx(2.2)
    stale = Pacer(clock=lambda: 0.0, wall=lambda: WALL + 8 * 24 * 3600, persist=True)
    assert stale.interval(key) == 0.0                                # older than a week: not used
    assert Pacer(clock=lambda: 0.0).interval(key) == 0.0             # a pacer that does not persist


@pytest.mark.parametrize("free_plan", [True, False])
def test_the_free_plan_seed_paces_from_the_first_request(free_plan):
    clock, server, backend, pacer = rig([ok(), ok()])
    key = pace_key(backend.origin)
    pacer.prepare(key, PRESETS["mistral"].free_plan_interval if free_plan else None)
    ask(clock, backend, pacer, free_plan=free_plan)
    ask(clock, backend, pacer, free_plan=free_plan)
    assert server.times == ([0.0, 1.1] if free_plan else [0.0, 0.0])
    assert PRESETS["openrouter"].free_plan_interval == 3.0 and PRESETS["gemini"].free_plan_interval is None


def test_the_headers_pace_before_any_rate_limit():
    """No requests left until the reset: the next one waits for it; some left: they are spread over it."""
    clock, server, backend, pacer = rig(
        [ok({"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "6.5s"}),
         ok({"x-ratelimit-remaining-requests": "4", "x-ratelimit-reset-requests": "2s"}), ok()],
        preset_id="openai")
    for _ in range(3):
        ask(clock, backend, pacer)
    assert server.times == pytest.approx([0.0, 6.5, 7.0])            # 6.5 s; then 2 s / 4 = 0.5 s after the start


def test_headers_come_before_the_learned_interval_and_the_seed():
    clock, server, backend, pacer = rig(
        [ok({"x-ratelimit-remaining-requests": "59", "x-ratelimit-reset-requests": "1s"}), ok(), ok()],
        preset_id="openai")
    key = pace_key(backend.origin)
    pacer.prepare(key, 3.0)
    pacer.rate_limited(key, None)                                    # learned: 3.0 × 2 = 6 s
    ask(clock, backend, pacer)
    assert server.times == [0.0]                                     # a first request never waits
    start = server.times[-1]
    ask(clock, backend, pacer)                                       # the headers: 1 s / 59
    assert server.times[-1] - start == pytest.approx(1 / 59)
    ask(clock, backend, pacer)                                       # no headers in that answer: still fresh
    assert server.times[-1] - server.times[-2] == pytest.approx(1 / 59)


def test_the_request_limit_headers_are_read():
    reset = (datetime.fromtimestamp(WALL, timezone.utc) + timedelta(seconds=12)).isoformat().replace("+00:00", "Z")
    assert requests_left({"anthropic-ratelimit-requests-remaining": "3",
                          "anthropic-ratelimit-requests-reset": reset}, WALL) == (3, pytest.approx(12.0))
    assert requests_left({"x-ratelimit-remaining-requests": "59", "x-ratelimit-reset-requests": "6m0s"},
                         WALL) == (59, 360.0)
    assert requests_left({"x-ratelimit-remaining-requests": "1", "x-ratelimit-reset-requests": "20ms"},
                         WALL) == (1, pytest.approx(0.02))
    assert requests_left({"x-ratelimit-remaining-requests": "1"}, WALL) is None
    assert requests_left({"x-ratelimit-reset-requests": "soon", "x-ratelimit-remaining-requests": "1"},
                         WALL) is None
    assert pacing.rate_headers({"authorization": "Bearer x", "Retry-After": "3"} | {"retry-after": "3"}) == \
        {"retry-after": "3"}


# --- two chats, one interval ------------------------------------------------------------------------------------
def test_two_chats_of_the_window_share_one_interval(monkeypatch, fake_backends):
    """Two chat tabs, one provider and model, through one gate: after chat A's rate limit, chat B's next request
    waits the learned interval from A's last start (shortened to 0.3 s for the test; real clock)."""
    from flextool.gui.assistant.gate import GatedBackend, ProviderGate

    monkeypatch.setattr(pacing, "START", 0.3)
    shared = Pacer()
    gate = ProviderGate(parallel=2, pacer=shared)
    rate = E.make_error(E.RateLimited, "rpm", E.ErrorContext(provider="Mistral"))
    done = Reply(Message("assistant", "done"), "end", None, Usage(1, 0, 0, 1))
    origin = ("mistral", "https://api.mistral.ai/v1", "example-model")
    starts: list[tuple[str, float]] = []

    class Timed(FakeBackend):
        def __init__(self, name: str, script: list) -> None:
            super().__init__(script, origin=origin)
            self.name = name

        def stream(self, req, on_event, cancel):
            starts.append((self.name, time.monotonic()))
            return super().stream(req, on_event, cancel)

    a, b = Timed("a", [rate, done]), Timed("b", [done])
    fake_backends.extend([a, b])
    chat_a, chat_b = GatedBackend(a, gate, "mistral", label="Mistral"), GatedBackend(b, gate, "mistral")
    key = pace_key(origin)
    req = request([Message("user", "hi")])
    cancel = threading.Event()
    call_with_retries(lambda: chat_a.stream(req, lambda e: None, cancel), NO_JITTER, lambda e: None, cancel,
                      pacing=Pacing(key, chat_a.pacer, here=False))
    call_with_retries(lambda: chat_b.stream(req, lambda e: None, cancel), NO_JITTER, lambda e: None, cancel,
                      pacing=Pacing(key, chat_b.pacer, here=False))
    (_, a1), (_, a2), (name, b1) = starts
    assert name == "b"
    assert a2 - a1 >= 0.3 - 0.02 and b1 - a2 >= 0.3 - 0.02
    assert shared.interval(key) == pytest.approx(0.3)                # answered twice: never below the start
