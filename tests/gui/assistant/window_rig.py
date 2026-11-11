"""A chat window over scripted backends, for the window's Tk tests (P2 §13.2 tests 6 and 7)."""
from __future__ import annotations

import threading
import time
import tkinter as tk
from collections.abc import Callable
from pathlib import Path

from flextool.assistant.probes import ModelFacts
from flextool.gui.assistant.chat_tab import ChatTab
from flextool.gui.assistant.chat_window import ChatWindow
from flextool.gui.assistant.gate import ProviderGate
from tests.assistant.llm.fake_backend import FakeBackend
from tests.gui.assistant.tk_helpers import pump

WORK_AREA = (0, 0, 1000, 800)


class Backends:
    """The backends the chats get, in the order their workers ask for one."""

    def __init__(self) -> None:
        self.queue: list[FakeBackend] = []
        self.made: list[FakeBackend] = []
        self.lock = threading.Lock()

    def add(self, *scripts: list) -> list[FakeBackend]:
        made = [FakeBackend(list(script)) for script in scripts]
        with self.lock:
            self.queue += made
        return made

    def factory(self, preset, conn) -> FakeBackend:  # noqa: ANN001 — the worker's (Preset, Connection)
        with self.lock:
            backend = self.queue.pop(0)
            self.made.append(backend)
        backend.origin = (preset.id, conn.base_url, conn.model)
        return backend

    def release_all(self) -> None:
        with self.lock:
            for backend in self.made + self.queue:
                backend.release_all()


def no_probe(_preset, _conn, _cancel, **_kwargs) -> ModelFacts:  # noqa: ANN001
    return ModelFacts()


class Windows:
    """Opens chat windows for one project and closes them (and their workers) at the end."""

    def __init__(self, root: tk.Tk, project: Path, backends: Backends) -> None:
        self.root = root
        self.project = project
        self.backends = backends
        self.opened: list[ChatWindow] = []
        self.tool_lock = threading.Lock()
        self.gate = ProviderGate()

    def open(self, *, probe: Callable = no_probe, **kwargs) -> ChatWindow:
        kwargs.setdefault("work_area", lambda _master: WORK_AREA)
        window = ChatWindow(self.root, self.project, "Project", backend_factory=self.backends.factory,
                            probe=probe, tool_lock=self.tool_lock, gate=self.gate, **kwargs)
        self.opened.append(window)
        pump(window, lambda: bool(window.winfo_viewable()))
        return window

    def close_all(self) -> None:
        self.backends.release_all()
        workers = []
        for window in self.opened:
            workers += window.workers()
            if window.winfo_exists() and not window.closing:
                window.close(ask=False)
        deadline = time.monotonic() + 5
        for worker in workers:
            worker.exited.wait(max(0.0, deadline - time.monotonic()))


def ask(window: ChatWindow, text: str, tab: ChatTab | None = None) -> ChatTab:
    """Type *text* into the current chat (or *tab*) and send it; returns the chat that got it."""
    tab = tab or window.current_tab()
    tab.input.delete("1.0", "end")
    tab.input.insert("1.0", text)
    tab.send()
    return tab


def wait_idle(window: ChatWindow, tab: ChatTab, timeout: float = 5.0) -> None:
    pump(window, lambda: not tab.busy, timeout)


def chat_texts(window: ChatWindow) -> list[str]:
    return [str(window.notebook.tab(t, "text")) for t in window.notebook.tabs()]
