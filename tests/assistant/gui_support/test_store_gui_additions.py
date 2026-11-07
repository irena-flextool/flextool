"""What the FlexTool window needs from the store and the session besides the terminal's needs (P2 §13.1 test 5b).

The chat numbers, the titles written while a chat saves and the open-chat list are in
``tests/assistant/test_store_and_ledger.py`` (P1 test 8, extended for P2).  Here: chats of several tabs in one
process never run tools at the same time, because every context shares the one process-wide tool lock (P2 §5.5,
critique 2 MJ2), and the acceptances of the data notice written by two processes at the same moment are all kept
(the acknowledgements file is changed under its lock, critique 2 m3c).
"""
from __future__ import annotations

import json
import multiprocessing
import threading
from pathlib import Path

import pytest

from flextool.assistant import endpoint, registry
from flextool.assistant.context import make_context
from flextool.assistant.llm_adapter import Message, Reply, ToolCall, ToolDone, ToolWaiting, Usage
from flextool.assistant.providers import PRESETS
from flextool.assistant.registry import make_tool
from flextool.assistant.session import Session, TurnCaps
from flextool.assistant.store import open_store
from tests.assistant import store_workers
from tests.assistant.llm.fake_backend import FakeBackend

SPAWN = multiprocessing.get_context("spawn")
USAGE = Usage(10, 0, 0, 5)


class SlowTool:
    """A tool that counts how many of its calls run at once; the first call waits for :attr:`release`."""

    def __init__(self) -> None:
        self.guard = threading.Lock()
        self.active = 0
        self.most = 0
        self.order: list[str] = []
        self.first_running = threading.Event()
        self.release = threading.Event()

    def handler(self, _ctx, args: dict) -> dict:
        with self.guard:
            self.active += 1
            self.most = max(self.most, self.active)
            self.order.append(args["text"])
            first = len(self.order) == 1
        try:
            if first:
                self.first_running.set()
                assert self.release.wait(10)
        finally:
            with self.guard:
                self.active -= 1
        return {"items": [args["text"]]}


@pytest.fixture
def slow_tool(monkeypatch: pytest.MonkeyPatch) -> SlowTool:
    fake = SlowTool()
    props = {"text": {"type": "string", "description": "a text"}}
    tool = make_tool(name="slow", title="slow", description="The slow tool.", properties=props,
                     handler=fake.handler, area="orientation", small_tier=True)
    monkeypatch.setattr(registry, "_tools", {"slow": tool})
    return fake


def tool_then_answer(text: str) -> list[Reply]:
    call = ToolCall(f"call-{text}", "slow", {"text": text}, json.dumps({"text": text}))
    return [Reply(Message("assistant", "", tool_calls=[call]), "tools", None, USAGE),
            Reply(Message("assistant", f"Answer {text}."), "end", None, USAGE)]


def chat_session(project: Path, text: str) -> tuple[Session, FakeBackend]:
    """A chat of one tab: its own context (made as each tab makes it), its own backend, saved conversation."""
    ctx = make_context(project, front_end="gui")
    backend = FakeBackend(tool_then_answer(text))
    store = open_store()
    conv = store.new_conversation(ctx, backend.origin, "medium")
    session = Session(ctx, backend, PRESETS["openai_compatible"], conversation=conv, caps=TurnCaps(),
                      tier="medium", store=store, ledger=None, facts=None, sleep=lambda _s: False)
    return session, backend


def test_two_chats_of_one_process_never_run_tools_at_the_same_time(tmp_path, slow_tool):
    project = tmp_path / "Projekti"
    project.mkdir()
    first, first_backend = chat_session(project, "a")
    second, second_backend = chat_session(project, "b")
    assert first.ctx.lock is second.ctx.lock is registry.TOOL_LOCK
    events: dict[str, list] = {"a": [], "b": []}
    waiting = threading.Event()

    def on_second(event) -> None:
        events["b"].append(event)
        if isinstance(event, ToolWaiting):
            waiting.set()

    results: dict[str, object] = {}
    one = threading.Thread(target=lambda: results.update(a=first.run_turn("A?", events["a"].append,
                                                                          threading.Event())))
    two = threading.Thread(target=lambda: results.update(b=second.run_turn("B?", on_second, threading.Event())))
    one.start()
    try:
        assert slow_tool.first_running.wait(5)
        two.start()
        assert waiting.wait(5)                          # the second chat's call waits for the first one's
        assert slow_tool.order == ["a"]
        assert [e.name for e in events["b"] if isinstance(e, ToolWaiting)] == ["slow"]
    finally:
        slow_tool.release.set()
        one.join(10)
        if two.ident is not None:
            two.join(10)
    assert slow_tool.most == 1 and slow_tool.order == ["a", "b"]
    assert results["a"].ok and results["b"].ok
    assert [e.ok for e in events["b"] if isinstance(e, ToolDone)] == [True]
    for backend in (first_backend, second_backend):
        backend.release_all()
    for session in (first, second):
        open_store().close(session.conversation)


def test_acceptances_written_by_two_processes_at_once_are_all_kept(user_config_in_tmp):
    root = user_config_in_tmp
    root.mkdir(parents=True, exist_ok=True)
    count = 12
    readies = [SPAWN.Event(), SPAWN.Event()]
    go, out = SPAWN.Event(), SPAWN.Queue()
    procs = [SPAWN.Process(target=store_workers.record_consents, args=(str(root), tag, count, ready, go, out))
             for tag, ready in zip(("window", "terminal"), readies, strict=True)]
    for p in procs:
        p.start()
    try:
        assert all(ready.wait(60) for ready in readies)
        go.set()
        assert sorted(out.get(timeout=60) for _ in procs) == [count, count]
    finally:
        for p in procs:
            p.join(30)
    assert all(p.exitcode == 0 for p in procs)
    acks = json.loads((root / "acknowledgements.json").read_text(encoding="utf-8"))["acks"]
    expected = {f"https://{tag}-{i}.example.test/v1" for tag in ("window", "terminal") for i in range(count)}
    assert {a["endpoint"] for a in acks} == expected and len(acks) == 2 * count
    preset = PRESETS["openai_compatible"]
    base_url = "https://window-3.example.test/v1"
    key = endpoint.consent_key(preset.id, base_url, endpoint.route_for(preset, base_url))
    assert endpoint.needs_consent(key, root) is False
