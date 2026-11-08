"""The AI assistant's side-menu entry and its lifecycle (P2 §3, §5.7, §6.4; §13.2 test 10).

In the real main window "AI assistant…" + ⚙ sit in row 5 above "Execution jobs…" and are exactly as wide; a
missing package makes both unavailable with the reason; an open project keeps the runtime context file without a
chat window (P2-O1); FlexTool closing withdraws the windows first, purges, removes the file and keeps its budget
while a chat is stuck in a tool call (P2-O2, P2-O11).

Run headless:  xvfb-run -a python -m pytest tests/gui/assistant --run-gui
"""
from __future__ import annotations

import gc
import time
import tkinter as tk
from pathlib import Path

import pytest

from flextool.assistant import endpoint
from flextool.assistant.runtime_context import runtime_path
from flextool.assistant.store import Store
from flextool.gui.assistant.entry import AssistantEntry
from flextool.gui.data_models import GlobalSettings
from flextool.gui.settings_io import load_global_settings, save_global_settings
from tests.assistant.gui_support.worker_rig import LOCAL, Tools, messages_on_disk, save_provider, say, tool_call
from tests.gui.assistant.tk_helpers import pump
from tests.gui.assistant.window_rig import WORK_AREA, Backends, ask, no_probe, wait_idle

pytestmark = pytest.mark.gui

UNAVAILABLE = ("The AI assistant's packages are missing or broken in this installation. Update FlexTool (Update "
               "FlexTool… or pip install --upgrade flextool) to install them.")


def test_the_entry_sits_above_execution_jobs_and_is_as_wide(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from flextool.gui import main_window as main_window_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FLEXTOOL_NO_UPDATE_CHECK", "1")
    monkeypatch.setattr(main_window_module.MainWindow, "_startup", lambda self: None)
    try:
        main = main_window_module.MainWindow()
    except tk.TclError:
        pytest.skip("No display available")
    try:
        main.update()
        entry = main._assistant
        frame = entry.frame
        side_menu = frame.master
        assert frame.grid_info()["row"] == 5
        assert main.execution_menu_btn.grid_info()["row"] == 6
        assert int(side_menu.grid_rowconfigure(4)["weight"]) == 1          # the stretch spacer
        assert str(entry.chat_button.cget("text")) == "AI assistant…"
        assert str(entry.settings_button.cget("text")) in ("⚙", "…")
        pump(main, lambda: frame.winfo_width() == main.execution_menu_btn.winfo_width(), timeout=3)
        assert frame.winfo_rootx() == main.execution_menu_btn.winfo_rootx()
        assert entry.available
    finally:
        main.destroy()
        gc.collect()


class FakeMain(tk.Toplevel):
    """A mapped stand-in for the main window: a project name, the shared settings and the marshalling queue."""

    def __init__(self, master: tk.Misc, name: str = "Project") -> None:
        super().__init__(master)
        self.geometry("600x400+40+40")
        self.title(f"FlexTool — {name}")
        self.current_project = name
        self.global_settings = GlobalSettings()
        self.execution_window = None
        self._result_viewer = None
        self.posted: list = []
        self.update()

    def post_to_main(self, fn, *args) -> None:  # noqa: ANN001
        self.posted.append((fn, args))


@pytest.fixture
def main(tk_root: tk.Tk, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "projects" / "Project").mkdir(parents=True)
    window = FakeMain(tk_root)
    yield window
    try:
        window.destroy()
    except tk.TclError:
        pass


def test_a_missing_package_makes_the_entry_unavailable(main: FakeMain) -> None:
    from flextool.gui.hover_tooltip import HoverTooltip

    entry = AssistantEntry(main, find_spec=lambda name: None if name == "openai" else object())
    frame = entry.build(tk.Frame(main))
    assert not entry.available
    assert entry.chat_button.instate(["disabled"]) and entry.settings_button.instate(["disabled"])
    tips = [w for w in gc.get_objects() if isinstance(w, HoverTooltip)
            and w._widget in (entry.chat_button, entry.settings_button)]
    assert sorted(t._text for t in tips) == [UNAVAILABLE, UNAVAILABLE]
    entry.open_chat()
    entry.open_settings()
    assert entry.window is None and entry.settings_dialog is None
    frame.destroy()


def test_no_runtime_file_without_the_assistant_package(main: FakeMain, tmp_path: Path) -> None:
    entry = AssistantEntry(main, find_spec=lambda name: None if name == "flextool.assistant" else object())
    entry.project_opened("Project")
    assert entry.writer is None
    pump(main, timeout=0.2)
    assert not runtime_path(tmp_path / "projects" / "Project").exists()


def test_an_open_project_keeps_the_runtime_file_without_a_chat_window(main: FakeMain, tmp_path: Path) -> None:
    entry = AssistantEntry(main, publisher_interval_ms=50)
    entry.project_opened("Project")
    path = runtime_path(tmp_path / "projects" / "Project")
    try:
        pump(main, path.exists)
        assert entry.window is None
        assert entry.close_for_project()
        assert not path.exists()
        assert entry.writer is None
    finally:
        entry._stop_context(wait=2.0)


@pytest.fixture
def tools(monkeypatch: pytest.MonkeyPatch):
    fake = Tools()
    fake.install(monkeypatch)
    yield fake
    fake.release.set()


def test_flextool_close_withdraws_purges_and_keeps_its_budget(main: FakeMain, tmp_path: Path, tools: Tools,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    save_provider(LOCAL)
    endpoint.mark_welcome_seen()
    backends = Backends()
    backends.add([say("Saved first."), tool_call("slow"), say("after the tool")])
    entry = AssistantEntry(main, publisher_interval_ms=50, window_options={
        "backend_factory": backends.factory, "probe": no_probe, "work_area": lambda _m: WORK_AREA})
    purged: list[str] = []
    real_purge = Store.purge_expired

    def purge(self: Store, now=None) -> int:  # noqa: ANN001
        purged.append("purged")
        return real_purge(self, now)

    monkeypatch.setattr(Store, "purge_expired", purge)
    order: list[str] = []
    real_entry_purge = entry._purge

    def entry_purge():  # noqa: ANN202
        order.append(main.wm_state())
        return real_entry_purge()

    monkeypatch.setattr(entry, "_purge", entry_purge)
    project = tmp_path / "projects" / "Project"
    path = runtime_path(project)
    workers = []
    try:
        entry.project_opened("Project")
        pump(main, path.exists)
        entry.open_chat()
        window = entry.window
        assert window is not None
        pump(window, lambda: bool(window.winfo_viewable()))
        tab = ask(window, "first")
        wait_idle(window, tab)
        ask(window, "use the tool", tab)
        assert tools.slow_running.wait(5)
        workers = window.workers()
        conversation_id = tab.conversation_id
        started = time.monotonic()
        entry.on_flextool_close(budget=1.0)
        elapsed = time.monotonic() - started
        assert elapsed < 1.5                            # the tool call is still running: the budget ends the wait
        assert order == ["withdrawn"]                   # the windows go first, then the purge
        assert purged == ["purged"]
        assert not path.exists()
        assert entry.window is None
        # What was received is on disk, with the running call recorded as not run (P2-O11).
        texts = [m.get("text") or "" for m in messages_on_disk(project, conversation_id)]
        assert "Saved first." in texts
        assert any(t.startswith("Not run: FlexTool was closed") for t in texts)
        # The chat window's size went into the main window's settings, which FlexTool saves next.
        geometries = main.global_settings.assistant_window_geometry
        assert len(geometries) == 1 and main.global_settings.assistant_window_layout_cw > 0
        save_global_settings(tmp_path / "projects", main.global_settings)
        loaded = load_global_settings(tmp_path / "projects")
        assert loaded.assistant_window_geometry == geometries
        assert loaded.assistant_window_layout_cw == main.global_settings.assistant_window_layout_cw
    finally:
        tools.release.set()
        backends.release_all()
        for worker in workers:
            worker.exited.wait(5)
        entry._stop_context(wait=2.0)


def test_settings_from_the_chat_window_and_save_reaches_the_chats(main: FakeMain,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    from flextool.gui.assistant.settings_dialog import AssistantSettingsDialog

    save_provider(LOCAL)
    endpoint.mark_welcome_seen()
    entry = AssistantEntry(main, window_options={"backend_factory": Backends().factory, "probe": no_probe,
                                                 "work_area": lambda _m: WORK_AREA})
    entry.open_chat()
    window = entry.window
    try:
        assert window is not None and not window.settings_button.instate(["disabled"])
        changed: list[str] = []
        monkeypatch.setattr(window, "settings_changed", lambda: changed.append("changed"))
        window.settings_button.invoke()
        dialog = entry.settings_dialog
        assert isinstance(dialog, AssistantSettingsDialog)
        assert dialog.master is window
        entry.open_settings(window)                    # a second click raises the same dialog
        assert entry.settings_dialog is dialog
        dialog._on_saved()
        assert changed == ["changed"]
        dialog.destroy()
        assert entry.settings_dialog is None
        entry.open_chat()                              # an open window is raised, not opened twice
        assert entry.window is window
    finally:
        entry.close_window()


def _seen_soon(timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while not endpoint.welcome_seen():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.02)
    return True


def _welcome_over(window: tk.Misc):
    from flextool.gui.assistant.notices import WelcomeDialog

    shown = [w for w in window.winfo_children() if isinstance(w, WelcomeDialog)]
    assert len(shown) == 1
    return shown[0]


def test_the_welcome_buttons_from_each_entry_path(main: FakeMain, monkeypatch: pytest.MonkeyPatch) -> None:
    """The chat entry, ⚙ and "How this works" show the same [Open chat] [Open settings] [Close]: Open chat grey
    until a provider is saved, Close opens nothing, each button opens what it names."""
    from flextool.gui.assistant import notices
    from flextool.gui.assistant.settings_dialog import AssistantSettingsDialog

    entry = AssistantEntry(main, window_options={"backend_factory": Backends().factory, "probe": no_probe,
                                                 "work_area": lambda _m: WORK_AREA})
    try:
        # The very first open, from the chat entry: no provider saved.
        entry.open_chat()
        welcome = _welcome_over(main)
        assert welcome.chat_button.instate(["disabled"])
        welcome.close_button.invoke()
        assert entry.window is None and entry.settings_dialog is None and not entry._opening
        assert _seen_soon()
        # ⚙ as if the welcome were not recorded yet: the same buttons; Open settings opens the settings.
        monkeypatch.setattr(notices, "welcome_needed", lambda _root=None: True)
        entry.open_settings()
        welcome = _welcome_over(main)
        assert welcome.chat_button.instate(["disabled"])
        welcome.settings_button.invoke()
        assert isinstance(entry.settings_dialog, AssistantSettingsDialog) and entry.window is None
        entry.settings_dialog.destroy()
        # A provider is saved: ⚙'s welcome opens the chat (without a second welcome).
        save_provider(LOCAL)
        entry.open_settings()
        welcome = _welcome_over(main)
        assert not welcome.chat_button.instate(["disabled"])
        welcome.chat_button.invoke()
        window = entry.window
        assert window is not None
        monkeypatch.undo()
        # "How this works" in the chat window: both buttons enabled, Open settings opens the settings.
        link = window.show_welcome()
        assert not link.chat_button.instate(["disabled"]) and not link.settings_button.instate(["disabled"])
        link.settings_button.invoke()
        assert isinstance(entry.settings_dialog, AssistantSettingsDialog)
        assert entry.settings_dialog.master is window
        entry.settings_dialog.destroy()
        link = window.show_welcome()
        link.chat_button.invoke()
        assert entry.window is window and entry.settings_dialog is None
    finally:
        entry.close_window()
