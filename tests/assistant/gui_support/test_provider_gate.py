"""Several chats on one provider (P2 §13.1 test 5, P2-D9, critique 2 MJ1).

At most ``parallel`` model requests go to one provider at a time; a short wait the provider asks of one chat holds
the others; a long one ends the others' questions at once with P1's long-wait message, nothing sent and no
rejected attempt counted; a wait ends when the chat is stopped; tool calls hold no slot.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from flextool.assistant import registry
from flextool.assistant.context import make_context
from flextool.assistant.llm_adapter import Message, Reply, ToolCall, Usage
from flextool.assistant.llm_errors import Cancelled, RateLimited
from flextool.assistant.providers import PRESETS
from flextool.assistant.registry import make_tool
from flextool.assistant.session import Session, TurnCaps
from flextool.assistant.store import open_store
from flextool.assistant.usage import Ledger
from flextool.gui.assistant.gate import GatedBackend, GateLongWait, GateWait, ProviderGate
from tests.assistant.llm.fake_backend import Block, FakeBackend

USAGE = Usage(10, 0, 0, 5)
KEY = "gemini"


def answer(text: str = "Done.") -> Reply:
    return Reply(Message("assistant", text), "end", None, USAGE)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _request(backend: GatedBackend, cancel: threading.Event | None = None) -> tuple[threading.Thread, dict, list]:
    """Send one request through *backend* on a thread; returns the thread, its outcome box and its events."""
    box: dict = {}
    events: list = []

    def run() -> None:
        from flextool.assistant.llm_adapter import Request

        try:
            box["reply"] = backend.stream(Request((), (), (Message("user", "q"),), 100), events.append,
                                          cancel or threading.Event())
        except Exception as exc:  # noqa: BLE001 — handed to the test
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, box, events


def _until(condition, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_limit_one_holds_the_second_chat_until_the_first_is_done(fake_backends):
    gate = ProviderGate(parallel=1)
    block = Block(then=answer("first"))
    first = FakeBackend([block])
    second = FakeBackend([answer("second")])
    fake_backends += [first, second]
    t1, box1, _ = _request(GatedBackend(first, gate, KEY, label="Gemini"))
    assert block.started.wait(3)
    t2, box2, events2 = _request(GatedBackend(second, gate, KEY, label="Gemini"))
    _until(lambda: events2)
    assert events2[0] == GateWait("busy", None)
    assert second.requests == []
    block.release.set()
    t1.join(3)
    t2.join(3)
    assert box1["reply"].message.text == "first" and box2["reply"].message.text == "second"
    assert gate.state(KEY)[0] == 0


def test_limit_two_lets_both_go_and_a_lowered_limit_holds_the_waiting_one(fake_backends):
    gate = ProviderGate(parallel=2)
    blocks = [Block(then=answer("a")), Block(then=answer("b"))]
    backends = [FakeBackend([blocks[0]]), FakeBackend([blocks[1]]), FakeBackend([answer("c")])]
    fake_backends += backends
    t1, _, _ = _request(GatedBackend(backends[0], gate, KEY))
    t2, _, _ = _request(GatedBackend(backends[1], gate, KEY))
    assert blocks[0].started.wait(3) and blocks[1].started.wait(3)
    gate.set_parallel(1)
    blocks[0].release.set()
    t1.join(3)
    t3, box3, events3 = _request(GatedBackend(backends[2], gate, KEY))
    _until(lambda: events3)
    assert backends[2].requests == []            # one is still running and the limit is now 1
    blocks[1].release.set()
    t2.join(3)
    t3.join(3)
    assert box3["reply"].message.text == "c"


def test_a_short_wait_asked_of_one_chat_holds_the_other(fake_backends):
    clock = Clock()
    gate = ProviderGate(parallel=2, clock=clock)
    limited = FakeBackend([RateLimited("slow down", provider="Gemini", retry_after=5)])
    other = FakeBackend([answer("other")])
    fake_backends += [limited, other]
    own = GatedBackend(limited, gate, KEY, label="Gemini")
    _, box, _ = _request(own)
    _until(lambda: box)
    assert isinstance(box["error"], RateLimited)
    assert gate.state(KEY)[1] == pytest.approx(1005.0)
    t2, box2, events2 = _request(GatedBackend(other, gate, KEY, label="Gemini"))
    _until(lambda: events2)
    assert events2[0] == GateWait("cool_down", 5)
    clock.now = 1003.2
    _until(lambda: GateWait("cool_down", 2) in events2)
    assert other.requests == []
    clock.now = 1005.0
    t2.join(3)
    assert box2["reply"].message.text == "other"
    # The chat that got the answer is not held by its own cool-down (P1's countdown is its wait).
    limited.script.append(answer("again"))
    clock.now = 1004.0
    gate.cool_down(KEY, 1010.0, owner=own)
    t3, box3, _ = _request(own)
    t3.join(3)
    assert box3["reply"].message.text == "again"


def test_cool_down_follows_the_429_reset(fake_backends):
    """Review G9, C4-m1: a 429 without retry-after whose headers say none remain for 20 s holds the other chats
    for those 20 s (before: the raised interval, 1.1 s)."""
    clock = Clock()
    gate = ProviderGate(parallel=2, clock=clock)
    err = RateLimited("slow down", provider="OpenAI")
    err.rate_headers = {"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "20s"}
    limited = FakeBackend([err])
    fake_backends.append(limited)
    _, box, _ = _request(GatedBackend(limited, gate, KEY, label="OpenAI"))
    _until(lambda: box)
    assert gate.state(KEY)[1] == pytest.approx(1020.0)


def _session(tmp_path: Path, backend, preset_id: str = "gemini") -> tuple[Session, Ledger]:
    ctx = make_context(tmp_path / "project", front_end="gui")
    ledger = Ledger()
    store = open_store()
    conv = store.new_conversation(ctx, backend.origin, "medium")
    session = Session(ctx, backend, PRESETS[preset_id], conversation=conv, caps=TurnCaps(), tier="medium",
                      store=store, ledger=ledger, facts=None)
    return session, ledger


def test_a_long_wait_ends_the_other_chat_at_once_without_sending(tmp_path, fake_backends):
    clock = Clock()
    gate = ProviderGate(parallel=1, clock=clock)
    limited = FakeBackend([RateLimited("wait an hour", provider="Google Gemini", retry_after=3600)])
    other = FakeBackend([answer()])
    fake_backends += [limited, other]
    first, _ = _session(tmp_path, GatedBackend(limited, gate, KEY, label="Google Gemini"))
    result = first.run_turn("question", lambda ev: None, threading.Event())
    assert result.error is not None and result.error.code == "rpm"
    assert gate.state(KEY)[2] == pytest.approx(4600.0)

    second, ledger = _session(tmp_path, GatedBackend(other, gate, KEY, label="Google Gemini"))
    started = time.monotonic()
    result = second.run_turn("question", lambda ev: None, threading.Event())
    assert time.monotonic() - started < 1.0
    assert isinstance(result.error, GateLongWait)
    assert result.error.user_message == ("Google Gemini asks to wait about 60 minutes before the next request. Ask "
                                         "again later, or switch provider in the settings.")
    assert other.requests == []
    months = ledger.read()["months"]
    rejected = [c.get("rejected_attempts", 0) for m in months.values() for p in m.values()
                for c in p["models"].values()]
    assert rejected == [1]                        # the first chat's answer only, not the second chat's


def test_stop_ends_a_wait(fake_backends):
    gate = ProviderGate(parallel=1)
    block = Block(then=answer())
    first = FakeBackend([block])
    second = FakeBackend([answer()])
    fake_backends += [first, second]
    _request(GatedBackend(first, gate, KEY))
    assert block.started.wait(3)
    cancel = threading.Event()
    t2, box2, events2 = _request(GatedBackend(second, gate, KEY), cancel)
    _until(lambda: events2)
    cancel.set()
    t2.join(1.0)
    assert not t2.is_alive()
    assert isinstance(box2["error"], Cancelled)
    assert second.requests == []
    assert gate.state(KEY)[0] == 1                # the waiting request never took a slot
    block.release.set()


def test_no_slot_is_held_while_a_tool_runs(tmp_path, monkeypatch, fake_backends):
    gate = ProviderGate(parallel=1)
    seen: list[int] = []

    def handler(_ctx, args: dict) -> dict:
        seen.append(gate.state(KEY)[0])
        return {"items": [args["text"]]}

    props = {"text": {"type": "string", "description": "a text"}}
    monkeypatch.setattr(registry, "_tools", {"echo": make_tool(name="echo", title="echo", description="Echo.",
                                                               properties=props, handler=handler,
                                                               area="orientation", small_tier=True)})
    call = ToolCall("call00001", "echo", {"text": "x"}, json.dumps({"text": "x"}))
    backend = FakeBackend([Reply(Message("assistant", "", tool_calls=[call]), "tools", None, USAGE), answer()])
    fake_backends.append(backend)
    session, _ = _session(tmp_path, GatedBackend(backend, gate, KEY))
    result = session.run_turn("question", lambda ev: None, threading.Event())
    assert result.ok
    assert seen == [0]
    assert gate.state(KEY)[0] == 0


def test_an_int_limit_caps_one_backend_below_parallel(fake_backends):
    """Review G18, C7-m4: a free-plan chat's own limit 1 holds it below the gate's ``parallel`` (3), counted against
    every request to the provider; a backend without a limit still enters beside it."""
    gate = ProviderGate(parallel=3)
    b1, bn = Block(then=answer("one")), Block(then=answer("paid"))
    one, paid, two = FakeBackend([b1]), FakeBackend([bn]), FakeBackend([answer("two")])
    fake_backends += [one, paid, two]
    t1, _, _ = _request(GatedBackend(one, gate, KEY, limit=1))
    assert b1.started.wait(3)
    tn, _, _ = _request(GatedBackend(paid, gate, KEY))            # limit None: parallel 3, one running
    assert bn.started.wait(3)
    t2, box2, events2 = _request(GatedBackend(two, gate, KEY, limit=1))
    _until(lambda: events2)
    assert events2[0] == GateWait("busy", None) and two.requests == []
    b1.release.set()
    t1.join(3)
    _until(lambda: len(events2) >= 2)                # still held: the paid request runs and its limit is 1
    assert two.requests == []
    bn.release.set()
    tn.join(3)
    t2.join(3)
    assert box2["reply"].message.text == "two"


def test_slots_go_in_arrival_order(fake_backends):
    """Review G18 critique m1: a free-plan request waiting for a slot keeps a later paid request out (before, the
    paid one entered at once and two busy paid chats could starve it); once it runs, the paid one enters beside
    it (the documented asymmetry)."""
    gate = ProviderGate(parallel=3)
    held, free_block = Block(then=answer("held")), Block(then=answer("free"))
    first, free, later = FakeBackend([held]), FakeBackend([free_block]), FakeBackend([answer("later")])
    fake_backends += [first, free, later]
    t1, _, _ = _request(GatedBackend(first, gate, KEY))
    assert held.started.wait(3)
    tf, _, events_f = _request(GatedBackend(free, gate, KEY, limit=1))
    _until(lambda: events_f)
    tl, box_l, events_l = _request(GatedBackend(later, gate, KEY))
    _until(lambda: events_l)
    assert events_l[0] == GateWait("busy", None)
    assert later.requests == [] and free.requests == []
    held.release.set()
    t1.join(3)
    assert free_block.started.wait(3)
    tl.join(3)
    assert box_l["reply"].message.text == "later"   # entered beside the running free-plan request
    free_block.release.set()
    tf.join(3)
    assert gate.state(KEY)[0] == 0 and gate._provider(KEY).waiting == {}
