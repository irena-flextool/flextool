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
