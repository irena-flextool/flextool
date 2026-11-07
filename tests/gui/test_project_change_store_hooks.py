"""Project changes and the AI assistant's saved chats (P2 §11, §13.2 test 12; no display).

A project rename moves its chats (the store gets the resolved paths; a failing store never stops the rename), a
project deletion deletes them (a chat held by another process is kept and expires), and a rename while an
assistant worker still runs is refused, the runtime context file kept.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest

from flextool.assistant.llm_adapter import Message
from flextool.assistant.runtime_context import runtime_path
from flextool.assistant.store import Store, open_store
from flextool.gui import main_window as main_window_module
from flextool.gui import project_utils
from flextool.gui.assistant import entry as entry_module
from flextool.gui.assistant.entry import AssistantEntry
from flextool.gui.data_models import GlobalSettings
from tests.assistant.conftest import fake_keyring, user_cache_in_tmp, user_config_in_tmp  # noqa: F401

ORIGIN = ("ollama", "http://localhost:11434/v1", "m")


@pytest.fixture
def projects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "projects"
    (folder / "Alpha").mkdir(parents=True)
    return folder


def _saved(store: Store, project: Path, text: str) -> str:
    conv = store.new_conversation(project, ORIGIN, "standard")
    conv.messages.append(Message("user", text))
    assert store.save(conv)
    store.close(conv)
    return conv.id


def test_rename_moves_the_chats_with_resolved_paths(projects: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = open_store()
    chat = _saved(store, projects / "Alpha", "first question")
    calls: list[tuple[Path, Path]] = []
    real = Store.rename_project

    def spy(self: Store, old: Path, new: Path) -> int:
        calls.append((old, new))
        return real(self, old, new)

    monkeypatch.setattr(Store, "rename_project", spy)
    project_utils.rename_project("Alpha", "Beta")
    assert calls == [((projects / "Alpha").resolve(), (projects / "Beta").resolve())]
    assert [i.id for i in store.list_conversations(projects / "Beta")] == [chat]
    assert store.list_conversations(projects / "Alpha") == []


def test_a_failing_store_still_renames(projects: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(self: Store, old: Path, new: Path) -> int:
        raise OSError("the store is unreadable")

    monkeypatch.setattr(Store, "rename_project", broken)
    assert project_utils.rename_project("Alpha", "Beta") == projects / "Beta"
    assert (projects / "Beta").is_dir() and not (projects / "Alpha").exists()


def test_deleting_a_project_deletes_its_chats_but_one_held_elsewhere(projects: Path) -> None:
    store = open_store()
    project = projects / "Alpha"
    free = _saved(store, project, "free question")
    held = _saved(store, project, "held question")
    key = store._project(project)[2]
    lock = store.project_dir(key) / f"{held}.json.lock"
    # Held by another live process (the one that started this test).
    lock.write_text(json.dumps({"pid": os.getppid(), "host": socket.gethostname(), "started": None,
                                "token": "t"}), encoding="utf-8")
    resolved = project.resolve()
    for child in project.iterdir():
        child.unlink()
    project.rmdir()
    project_utils.notify_assistant_project_deleted(resolved)
    remaining = [i.id for i in store.list_conversations(resolved)]
    assert remaining == [held]
    assert free not in remaining
    lock.unlink()
    project_utils.notify_assistant_project_deleted(resolved)
    assert store.list_conversations(resolved) == []
    rows = json.loads((store.root / "projects.json").read_text(encoding="utf-8"))["projects"]
    assert key not in rows


def _eventually(condition, timeout: float = 3.0) -> bool:  # noqa: ANN001
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


class FakeMain:
    """What the entry and the combo rename use of the main window, without a display."""

    def __init__(self, name: str) -> None:
        self.current_project = name
        self._title = f"FlexTool — {name}"
        self.global_settings = GlobalSettings(recent_project=name)
        self.switched: list[str] = []
        self.after_calls: list[object] = []

    def after(self, _ms: int, fn) -> str:  # noqa: ANN001
        self.after_calls.append(fn)
        return f"after#{len(self.after_calls)}"

    def after_cancel(self, _id: str) -> None:
        pass

    def title(self, text: str | None = None) -> str:
        if text is not None:
            self._title = text
        return self._title

    def configure(self, **_kwargs) -> None:
        pass

    def update_idletasks(self) -> None:
        pass

    def _switch_project(self, name: str) -> None:
        self.switched.append(name)


class StuckWorker:
    """A closed chat's worker still in a tool call of *project_path*."""

    def __init__(self, project_path: Path) -> None:
        self.project_path = project_path
        self.exited = threading.Event()


def test_a_rename_while_a_worker_runs_is_refused_and_the_runtime_file_kept(
        projects: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(entry_module, "PROJECT_CHANGE_BUDGET_S", 0.2)
    main = FakeMain("Alpha")
    entry = AssistantEntry(main)
    entry.project_opened("Alpha")
    path = runtime_path(projects / "Alpha")
    assert _eventually(path.exists)
    stuck = StuckWorker(projects / "Alpha")
    entry._finishing.append(stuck)
    shown: list[str] = []
    monkeypatch.setattr(main_window_module.simpledialog, "askstring", lambda *a, **k: "Beta")
    monkeypatch.setattr(main_window_module.messagebox, "showinfo", lambda title, text, **k: shown.append(text))
    main._assistant = entry
    main_window_module.MainWindow._on_combo_rename(main, None)
    assert shown == ["The AI assistant is still finishing a tool call. Try again in a moment."]
    assert (projects / "Alpha").is_dir() and not (projects / "Beta").exists()
    assert main.switched == []
    assert entry.writer is not None and entry.context_project == "Alpha"
    assert _eventually(path.exists)                 # the runtime file is written again
    assert main.title() == "FlexTool — Alpha"       # the waiting note is gone from the title
    # Once the worker has ended the rename goes through, and the main window opens the new name.
    stuck.exited.set()
    shown.clear()
    main_window_module.MainWindow._on_combo_rename(main, None)
    assert shown == []
    assert (projects / "Beta").is_dir()
    assert main.switched == ["Beta"]
    assert not path.exists()
    entry._stop_context(wait=2.0)
