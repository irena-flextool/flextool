"""A chat worker with scripted backends, a fake probe and fake tools, read the way the chat window reads it.

Shared by the worker tests (``test_assistant_worker.py``) and the window's Tk tests (``tests/gui/assistant``).
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from flextool.assistant import _files, registry
from flextool.assistant.context import ContextSnapshot
from flextool.assistant.llm_adapter import Message, Reply, Request, ToolCall, Usage
from flextool.assistant.probes import ModelFacts
from flextool.assistant.registry import make_tool
from flextool.assistant.settings import ProviderSettings, Settings, save_settings
from flextool.assistant.store import open_store
from flextool.gui.assistant.gate import ProviderGate
from flextool.gui.assistant.worker import AssistantWorker, TurnJob
from tests.assistant.llm.fake_backend import FakeBackend

USAGE = Usage(100, 0, 0, 10)
LOCAL = ProviderSettings(preset="openai_compatible", base_url="http://127.0.0.1:9/v1", model="fake-model")
REMOTE = ProviderSettings(preset="gemini", model="gemini-2.5-flash")


def say(text: str) -> Reply:
    return Reply(Message("assistant", text), "end", None, USAGE)


_ids = iter(f"call{n:05d}" for n in range(100_000))


def tool_call(name: str = "echo", text: str = "x", *, say_first: str = "") -> Reply:
    call = ToolCall(next(_ids), name, {"text": text}, json.dumps({"text": text}))
    return Reply(Message("assistant", say_first, tool_calls=[call]), "tools", None, USAGE)


class Tools:
    """``echo`` records the context's default scenario; ``slow`` blocks until :attr:`release`."""

    def __init__(self) -> None:
        self.ran: list[tuple[str, str | None]] = []
        self.release = threading.Event()
        self.slow_running = threading.Event()

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        props = {"text": {"type": "string", "description": "a text"}}

        def echo(ctx, args: dict) -> dict:
            self.ran.append(("echo", ctx.default_scenario))
            return {"items": [args.get("text", "")]}

        def slow(ctx, args: dict) -> dict:
            self.ran.append(("slow", ctx.default_scenario))
            self.slow_running.set()
            assert self.release.wait(10)
            return {"items": ["slow result"]}

        monkeypatch.setattr(registry, "_tools", {
            name: make_tool(name=name, title=name, description=f"The {name} tool.", properties=props,
                            handler=fn, area="orientation", small_tier=True)
            for name, fn in (("echo", echo), ("slow", slow))})


class Rig:
    """One worker, its fake backends and probe, and the events read so far."""

    def __init__(self, project: Path, backends: list[FakeBackend], *, probe: Callable | None = None,
                 tool_lock: threading.Lock | None = None, gate: ProviderGate | None = None) -> None:
        self.backends = backends
        self.made: list[FakeBackend] = []
        self.probes: list[str] = []
        self.project = project

        def factory(_preset, conn):
            backend = self.backends.pop(0)
            backend.origin = (_preset.id, conn.base_url, conn.model)
            self.made.append(backend)
            return backend

        def default_probe(_preset, conn, cancel, **_kwargs) -> ModelFacts:
            self.probes.append(conn.model)
            return ModelFacts()

        self.worker = AssistantWorker(project, "Project", gate=gate or ProviderGate(),
                                      tool_lock=tool_lock or threading.Lock(), backend_factory=factory,
                                      probe=probe or default_probe)
        self.events: list[tuple[int | None, object]] = []
        self._next = 0
        self.worker.start()

    def job(self, text: str = "question", scenario: str | None = None, *, rewind: int | None = None,
            seed: str | None = None) -> TurnJob:
        self._next += 1
        snapshot = ContextSnapshot(project="Project", scenario=scenario, front_end="gui")
        job = TurnJob(self._next, text, snapshot, rewind=rewind, seed=seed)
        self.worker.submit(job)
        return job

    def read(self) -> None:
        while True:
            try:
                self.events.append(self.worker.events.get_nowait())
            except Exception:  # noqa: BLE001 — queue.Empty
                return

    def wait_for(self, kind: type, timeout: float = 5.0, *, count: int = 1) -> object:
        deadline = time.monotonic() + timeout
        while True:
            self.read()
            found = [ev for _jid, ev in self.events if isinstance(ev, kind)]
            if len(found) >= count:
                return found[count - 1]
            assert time.monotonic() < deadline, f"no {kind.__name__}; events: {self.kinds()}"
            time.sleep(0.01)

    def kinds(self) -> list[str]:
        return [type(ev).__name__ for _jid, ev in self.events]

    def requests(self) -> list[Request]:
        return [r for b in self.made for r in b.requests]

    def close(self) -> None:
        for backend in self.made + self.backends:
            backend.release_all()
        self.worker.shutdown()
        self.worker.exited.wait(5)




def save_provider(provider: ProviderSettings, *, expiry: int = 30, parallel: int = 1) -> None:
    save_settings(Settings(provider=provider, conversation_expiry_days=expiry, parallel_requests=parallel))


def conversation_file(project: Path, conversation_id: str) -> Path:
    store = open_store()
    return store.project_dir(_files.project_key(project.resolve())) / f"{conversation_id}.json"


def messages_on_disk(project: Path, conversation_id: str) -> list[dict]:
    return json.loads(conversation_file(project, conversation_id).read_text(encoding="utf-8"))["messages"]
