"""FlexTool's own retries (P1 §5), on a fake clock."""
from __future__ import annotations

import threading

import pytest

from flextool.assistant import llm_errors as E
from flextool.assistant.llm_adapter import Message, Reply, RetryWait, Usage
from flextool.assistant.llm_errors import ErrorContext, make_error
from flextool.assistant.llm_retry import RetryPolicy, call_with_retries
from tests.assistant.llm.fake_backend import FakeBackend

CTX = ErrorContext(preset_id="gemini", provider="Example AI", model="m", host="api.example.test", port=443)
OK = Reply(Message("assistant", "done"), "end", None, Usage(1, 0, 0, 1))
NO_JITTER = RetryPolicy(jitter=0.0)


class Clock:
    """A fake clock whose sleep advances it; *cancel_after* sleeps set the cancel event."""

    def __init__(self, cancel: threading.Event, cancel_after: int | None = None) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []
        self.cancel = cancel
        self.cancel_after = cancel_after

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> bool:
        self.sleeps.append(seconds)
        self.now += seconds
        if self.cancel_after is not None and len(self.sleeps) >= self.cancel_after:
            self.cancel.set()
        return self.cancel.is_set()


def rate_limited(retry_after=None, **kw):
    return make_error(E.RateLimited, "rpm", CTX, retry_after=retry_after, **kw)


def busy(**kw):
    return make_error(E.ServerBusy, "busy", CTX, **kw)


def scripted(*items):
    calls = []

    def fn():
        item = items[len(calls)]
        calls.append(item)
        if isinstance(item, Exception):
            raise item
        return item

    return fn, calls


def run(fn, policy=NO_JITTER, cancel_after=None, **kw):
    cancel = threading.Event()
    clock = Clock(cancel, cancel_after)
    events = []
    result = call_with_retries(fn, policy, events.append, cancel, sleep=clock.sleep, clock=clock.clock, **kw)
    return result, clock, events


def run_raises(fn, policy=NO_JITTER, cancel_after=None, **kw):
    cancel = threading.Event()
    clock = Clock(cancel, cancel_after)
    events = []
    with pytest.raises(E.LLMError) as info:
        call_with_retries(fn, policy, events.append, cancel, sleep=clock.sleep, clock=clock.clock, **kw)
    return info.value, clock, events


def test_retry_after_is_honoured_with_a_countdown():
    fn, calls = scripted(rate_limited(3.0), rate_limited(2.0), OK)
    result, clock, events = run(fn)
    assert result is OK and len(calls) == 3
    assert sum(clock.sleeps) == pytest.approx(5.0)
    waits = [e for e in events if isinstance(e, RetryWait)]
    assert [(e.seconds_left, e.attempt, e.of) for e in waits] == [(3, 1, 3), (2, 1, 3), (1, 1, 3), (2, 2, 3),
                                                                   (1, 2, 3)]
    assert waits[0].reason == ("Example AI is limiting how often requests can be sent. Trying again in 3 seconds "
                               "(attempt 1 of 3).")
    assert all(s <= 1.0 for s in clock.sleeps)


def test_backoff_doubles_and_is_capped():
    fn, _ = scripted(busy(), busy(), busy(), OK)
    _, clock, events = run(fn)
    assert sum(clock.sleeps) == pytest.approx(2 + 4 + 8)
    fn, _ = scripted(busy(), busy(), busy(), OK)
    _, clock, _ = run(fn, RetryPolicy(backoff_start=50.0, backoff_cap=60.0, jitter=0.0))
    assert sum(clock.sleeps) == pytest.approx(50 + 60 + 60)
    first = next(e for e in events if isinstance(e, RetryWait))
    assert first.reason == "Example AI is overloaded at the moment. Trying again in 2 seconds (attempt 1 of 3)."


def test_jitter_stays_within_its_band():
    import random

    policy = RetryPolicy()
    rng = random.Random(4)
    for n in (1, 2, 3):
        for _ in range(50):
            wait = policy.backoff(n, rng)
            base = min(60.0, 2.0 * 2 ** (n - 1))
            assert 0.8 * base <= wait <= 1.2 * base


def test_at_most_three_retries_then_the_last_message():
    fn, calls = scripted(*[rate_limited(1.0) for _ in range(5)])
    err, _, _ = run_raises(fn)
    assert len(calls) == 4
    assert isinstance(err, E.RateLimited)
    assert err.user_message == "Example AI is still limiting requests. Wait a minute and ask again."


def test_a_long_wait_is_not_waited_for():
    fn, calls = scripted(rate_limited(130.0), OK)
    err, clock, events = run_raises(fn)
    assert len(calls) == 1 and clock.sleeps == [] and not events
    assert err.user_message == ("Example AI asks to wait about 3 minutes before the next request. Ask again later, "
                                "or switch provider in the settings.")


def test_nothing_is_retried_once_output_was_streamed():
    fn, calls = scripted(busy(streamed=True, partial_text="Half"), OK)
    err, clock, _ = run_raises(fn)
    assert len(calls) == 1 and clock.sleeps == []
    assert err.streamed and err.partial_text == "Half"


@pytest.mark.parametrize("cls,key", [(E.QuotaExhausted, "quota"), (E.AuthFailed, "auth"),
                                     (E.NoCredit, "credit"), (E.ContextTooLong, "context"),
                                     (E.HistoryRejected, "history"), (E.ModelNotFound, "model")])
def test_other_errors_are_never_retried(cls, key):
    fn, calls = scripted(make_error(cls, key, CTX), OK)
    err, clock, _ = run_raises(fn)
    assert isinstance(err, cls) and len(calls) == 1 and clock.sleeps == []


def test_cancel_during_the_countdown():
    fn, calls = scripted(rate_limited(10.0), OK)
    err, clock, _ = run_raises(fn, cancel_after=2)
    assert isinstance(err, E.Cancelled) and err.user_message == "Stopped."
    assert len(calls) == 1 and len(clock.sleeps) == 2


def test_an_unreachable_remote_endpoint_is_tried_once_more_after_two_seconds():
    unreachable = make_error(E.Unreachable, "connection_remote", CTX)
    fn, calls = scripted(unreachable, unreachable, OK)
    err, clock, events = run_raises(fn)
    assert err is unreachable and len(calls) == 2
    assert sum(clock.sleeps) == pytest.approx(2.0)
    assert events[0].reason == "Could not reach api.example.test. Trying again in 2 seconds."
    fn, calls = scripted(unreachable, OK)
    result, _, _ = run(fn)
    assert result is OK


def test_a_local_endpoint_is_never_retried():
    local = ErrorContext(provider="Ollama", host="localhost", port=11434, local=True, server_name="Ollama")
    fn, calls = scripted(make_error(E.Unreachable, "connection_local", local), OK)
    err, clock, _ = run_raises(fn)
    assert isinstance(err, E.Unreachable) and len(calls) == 1 and clock.sleeps == []


def test_rejected_attempts_are_reported():
    rejected = []
    fn, _ = scripted(rate_limited(1.0), busy(), OK)
    run(fn, on_rejected=rejected.append)
    assert [type(e) for e in rejected] == [E.RateLimited, E.ServerBusy]


def test_cancel_before_the_request():
    cancel = threading.Event()
    cancel.set()
    fn, calls = scripted(OK)
    with pytest.raises(E.Cancelled):
        call_with_retries(fn, NO_JITTER, lambda e: None, cancel)
    assert calls == []


def test_retries_through_the_fake_backend(fake_backends):
    from flextool.assistant.llm_adapter import Request, TextDelta

    backend = FakeBackend([rate_limited(1.0), OK])
    fake_backends.append(backend)
    req = Request(system=("s",), tools=(), messages=(Message("user", "hi"),), max_output_tokens=100)
    cancel = threading.Event()
    clock = Clock(cancel)
    events = []
    reply = call_with_retries(lambda: backend.stream(req, events.append, cancel), NO_JITTER, events.append, cancel,
                              sleep=clock.sleep, clock=clock.clock)
    assert reply.message.text == "done" and reply.message.origin == backend.origin
    assert len(backend.requests) == 2
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "done"
