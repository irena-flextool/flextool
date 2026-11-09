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
def test_mistral_is_paced_from_the_first_request_free_or_not(free_plan):
    """Owner decision 2026-11-08: Mistral's 1.1 s applies also to a paid account."""
    from flextool.assistant.providers import pacing_seed

    clock, server, backend, pacer = rig([ok(), ok()])
    key = pace_key(backend.origin)
    pacer.prepare(key, pacing_seed(PRESETS["mistral"], free_plan))
    ask(clock, backend, pacer, free_plan=free_plan)
    ask(clock, backend, pacer, free_plan=free_plan)
    assert server.times == [0.0, 1.1]


def test_the_seeds_of_the_presets():
    from flextool.assistant.providers import pacing_seed

    assert [pacing_seed(PRESETS["mistral"], free) for free in (True, False)] == [1.1, 1.1]
    assert [pacing_seed(PRESETS["openrouter"], free) for free in (True, False)] == [3.0, None]
    assert [pacing_seed(PRESETS["gemini"], free) for free in (True, False)] == [None, None]


def test_the_learned_interval_never_goes_below_the_seed():
    p = Pacer(clock=lambda: 0.0)
    p.prepare("k", 3.0)
    assert p.rate_limited("k", None) == pytest.approx(6.0)
    for expected in (4.8, 3.84, 3.072, 3.0, 3.0):
        p.succeeded("k")
        assert p.interval("k") == pytest.approx(expected)
    p.prepare("k", 5.0)                                              # a higher seed: the spacing is the seed
    assert p.interval("k") == pytest.approx(5.0)


def test_the_headers_pace_before_any_rate_limit():
    """No requests left until the reset: the next one waits for it; some left: they are spread over it."""
    clock, server, backend, pacer = rig(
        [ok({"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "6.5s"}),
         ok({"x-ratelimit-remaining-requests": "4", "x-ratelimit-reset-requests": "2s"}), ok()],
        preset_id="openai")
    for _ in range(3):
        ask(clock, backend, pacer)
    assert server.times == pytest.approx([0.0, 6.5, 7.0])            # 6.5 s; then 2 s / 4 = 0.5 s after the start


def test_headers_only_lengthen_the_learned_interval_and_the_seed():
    """Updated (review G9): the headers' spacing counts only when it is longer than the learned interval and the
    seed.  59 left in 1 s (1/59 s) is shorter than the learned 4.8 s; 1 left in 6 s (6 s) is longer."""
    clock, server, backend, pacer = rig(
        [ok({"x-ratelimit-remaining-requests": "59", "x-ratelimit-reset-requests": "1s"}), ok()],
        preset_id="openai")
    key = pace_key(backend.origin)
    pacer.prepare(key, 3.0)
    pacer.rate_limited(key, None)                                    # learned: 3.0 × 2 = 6 s
    ask(clock, backend, pacer)
    assert server.times == [0.0]                                     # a first request never waits
    ask(clock, backend, pacer)                                       # answered: 6 × 0.8 = 4.8 s > 1/59 s
    assert server.times[-1] == pytest.approx(4.8)

    clock, server, backend, pacer = rig(
        [ok({"x-ratelimit-remaining-requests": "1", "x-ratelimit-reset-requests": "6s"}), ok()],
        preset_id="openai")
    pacer.prepare(key, 3.0)
    pacer.rate_limited(key, None)
    ask(clock, backend, pacer)
    ask(clock, backend, pacer)                                       # 6 s / 1 > 4.8 s: the headers win
    assert server.times[-1] == pytest.approx(6.0)


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


# --- a reported limit of 0 and the 429's own headers (review G9: OR-2, C4-m1) ----------------------------------
#: Mistral's answer seen on 2026-11-08 (OR-2) for an account whose API access is not active.
OR2_BODY = {"message": "Rate limit exceeded", "type": "rate_limited", "param": None, "code": "1300"}
OR2_HEADERS = {"x-ratelimit-limit-req-minute": "0", "x-ratelimit-remaining-req-minute": "0"}
#: A paid account at its per-minute limit: the limit is 750, none remain.
PAID_HEADERS = {"x-ratelimit-limit-req-minute": "750", "x-ratelimit-remaining-req-minute": "0"}
INACTIVE = ("Your Mistral account allows 0 requests per minute: its API access is not active yet. Activate a plan "
            "in your Mistral account, then ask again.")


@pytest.fixture
def mistral_server():
    """A loopback server playing Mistral's chat completions; the test sets ``answers`` (one per request)."""
    from tests.assistant.llm.local_server import FakeServer

    server = FakeServer()
    server.answers = []
    server.routes[("POST", "/v1/chat/completions")] = lambda _body: server.answers.pop(0)
    yield server
    server.close()


def _sse_ok(headers: dict | None = None) -> tuple:
    text = openai_sse(payload("openai", "text_usage")["chunks"]).decode()
    return 200, text, {"Content-Type": "text/event-stream", **(headers or {})}


def _loopback_mistral(server, clock: FakeTime, *, persist: bool = False):
    backend = make_backend(PRESETS["mistral"], connection("mistral", url=f"http://127.0.0.1:{server.port}/v1"))
    pacer = Pacer(clock=clock.now, wall=lambda: WALL, persist=persist)
    pacer.prepare(pace_key(backend.origin), 1.1)                      # Mistral's seed (RF-D15)
    return backend, pacer


@pytest.mark.parametrize("free_plan", [False, True])
def test_limit_zero_is_not_retried_and_changes_no_pacing(mistral_server, user_cache_in_tmp, free_plan):
    """OR-2: one request, no countdown, the owner's message, the interval stays at the seed and nothing is
    stored (before: 4 or 7 requests, a learned 17.6–20 s interval stored, the "requests per second" message)."""
    mistral_server.answers = [(429, OR2_BODY, OR2_HEADERS) for _ in range(8)]
    clock = FakeTime()
    backend, pacer = _loopback_mistral(mistral_server, clock, persist=True)
    events: list = []
    try:
        with pytest.raises(E.AccessInactive) as info:
            ask(clock, backend, pacer, free_plan=free_plan, events=events)
    finally:
        backend.close()
    assert len(mistral_server.requests) == 1
    assert not [e for e in events if isinstance(e, RetryWait)]
    assert info.value.user_message == INACTIVE
    assert E.message_for(info.value, "gui") == E.message_for(info.value, "terminal") == INACTIVE
    key = pace_key(backend.origin)
    assert pacer.interval(key) == pytest.approx(1.1)
    assert probes.stored_interval(key, max_age=pacing.STORED_MAX_AGE, now=WALL) is None


def test_a_paid_account_at_its_minute_limit_waits_for_the_window(mistral_server):
    """Limit 750, none remaining: a rate limit, not an inactive account.  The retry waits the assumed 60 s window
    (Mistral names no reset) and the learned interval is left alone (C4-m1; before: retried after 2.2 s)."""
    mistral_server.answers = [(429, OR2_BODY, PAID_HEADERS), _sse_ok()]
    clock = FakeTime()
    backend, pacer = _loopback_mistral(mistral_server, clock)
    times: list[float] = []
    mistral_server.routes[("POST", "/v1/chat/completions")] = \
        lambda _body: (times.append(clock.now()), mistral_server.answers.pop(0))[1]
    try:
        ask(clock, backend, pacer)
    finally:
        backend.close()
    assert times == pytest.approx([0.0, 60.0])
    assert pacer.interval(pace_key(backend.origin)) == pytest.approx(1.1)


def test_a_second_minute_limit_ends_with_wait_a_minute(mistral_server):
    """Two such answers: 60 + 60 s is past the 90 s of retry waits, so the question ends — with "wait a minute",
    not the per-second wording (critique G9 item 7)."""
    mistral_server.answers = [(429, OR2_BODY, PAID_HEADERS) for _ in range(3)]
    clock = FakeTime()
    backend, pacer = _loopback_mistral(mistral_server, clock)
    try:
        with pytest.raises(E.RateLimited) as info:
            ask(clock, backend, pacer)
    finally:
        backend.close()
    assert len(mistral_server.requests) == 2
    assert info.value.user_message == "Mistral is still limiting requests. Wait a minute and ask again."


def test_test_model_shows_the_limit_zero_message(mistral_server, user_config_in_tmp):
    """Test model with the OR-2 answer: "failed" with the owner's message (before: a retried rate limit, or
    "refused the test request: no reason given")."""
    from flextool.assistant.model_check import check_model
    from flextool.assistant.usage import Ledger

    mistral_server.answers = [(429, OR2_BODY, OR2_HEADERS) for _ in range(8)]
    clock = FakeTime()
    backend, pacer = _loopback_mistral(mistral_server, clock)
    try:
        result = check_model(PRESETS["mistral"], backend, threading.Event(), label="Mistral",
                             ledger=Ledger(user_config_in_tmp), ledger_key="mistral", pacer=pacer, clock=clock.now,
                             sleep=clock.sleep, front_end="gui")
    finally:
        backend.close()
    assert (result.outcome, result.message) == ("failed", INACTIVE)
    assert len(mistral_server.requests) == 1


def test_mistral_minute_headers_are_read():
    assert requests_left({"x-ratelimit-remaining-req-minute": "59", "x-ratelimit-limit-req-minute": "60"},
                         WALL) == (59, 60.0)
    assert requests_left(PAID_HEADERS, WALL) == (0, 60.0)
    assert requests_left(OR2_HEADERS, WALL) is None                  # limit 0: no pace, the classifier's case
    assert pacing.rate_headers(OR2_HEADERS | {"x-ratelimit-limit-tokens-minute": "5"}) == OR2_HEADERS


def test_a_429_reset_header_is_the_wait():
    """C4-m1: remaining 0, reset 20 s, no retry-after: the wait is 20 s and nothing is learned (before: 1.1 s)."""
    p = Pacer(clock=lambda: 0.0, wall=lambda: WALL)
    p.prepare("k", 1.1)
    headers = {"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "20s"}
    assert p.rate_limited("k", None, headers) == pytest.approx(20.0)
    assert p.interval("k") == pytest.approx(1.1)
    assert p.header_wait({"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "2m"}) is None


def test_header_spacing_is_held_at_the_seed():
    """600 requests left over 60 s spread to 0.1 s, below the 1.1 s seed: the seed holds (before: 0.1 s)."""
    clock, server, backend, pacer = rig(
        [ok({"x-ratelimit-remaining-requests": "600", "x-ratelimit-reset-requests": "60s"}), ok()],
        preset_id="openai")
    pacer.prepare(pace_key(backend.origin), 1.1)
    ask(clock, backend, pacer)
    ask(clock, backend, pacer)
    assert server.times == pytest.approx([0.0, 1.1])


def test_mistral_headers_do_not_shorten_the_learned_interval():
    """Guard (critique G9 item 2): 700 of 750 left spreads to 0.086 s; the learned interval holds."""
    clock, server, backend, pacer = rig(
        [ok({"x-ratelimit-remaining-req-minute": "700", "x-ratelimit-limit-req-minute": "750"}), ok()])
    key = pace_key(backend.origin)
    pacer.prepare(key, 1.1)
    pacer.rate_limited(key, None)
    pacer.rate_limited(key, None)                                    # learned 2.2, then 4.4
    ask(clock, backend, pacer)                                       # answered: 4.4 × 0.8 = 3.52
    ask(clock, backend, pacer)
    assert server.times == pytest.approx([0.0, 3.52])
