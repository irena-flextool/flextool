"""The AI assistant's place in the FlexTool window (P2 §3, §5.7, §6.4, §11): the side-menu entry and the lifecycle.

* **Side menu** (§3, §10): "AI assistant…" with ⚙ beside it, one row above "Execution jobs…".  Building it only
  looks the packages up (``importlib.util.find_spec``); when one is missing both buttons are unavailable and say
  why.  The chat window, the dialogs, the SDKs, the keyring and the store are imported on the first click, inside
  ``try``: a failure is a message, never a broken main window.
* **Project open** (P2-O1): while a project is open its runtime context file is kept — also without a chat window
  — by a :class:`~flextool.assistant.runtime_context.RuntimeContextWriter` fed by the 1 s
  :class:`~flextool.gui.assistant.gui_state.ContextPublisher`.  They are imported when ``flextool.assistant`` is
  found, whatever the SDKs (the agent connection needs none, D9).
* **Project change** (§5.7, §11): a switch, rename or deletion closes the chat window (P2-D6) and waits for its
  workers within a budget; a rename or deletion is refused while a worker can still write (a slow tool call).
* **FlexTool close** (P2-O2, P2-O11): the windows are withdrawn, the answers stop without asking and what was
  received is saved (a running tool call is recorded as not run, so the chat can be resumed), the runtime file is
  deleted, a project rename or deletion still moving its chats finishes and the expired conversations are purged
  — within a budget; what is left is a daemon thread.  The Tk thread's waits serve the Tcl calls other threads
  hand it (:func:`pumped_wait`), never the user's events.
* **Start** (§5.7): the purge and the clean-up of runtime files left by processes that are gone run on a daemon
  thread a few seconds after the window opens.
"""
from __future__ import annotations

import _tkinter
import gc
import importlib.util
import logging
import sys
import threading
import time
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING, Any

from flextool.gui.assistant import strings

if TYPE_CHECKING:
    from flextool.assistant.runtime_context import RuntimeContextWriter
    from flextool.gui.assistant.chat_window import ChatWindow
    from flextool.gui.assistant.gui_state import ContextPublisher
    from flextool.gui.assistant.settings_dialog import AssistantSettingsDialog
    from flextool.gui.assistant.worker import AssistantWorker

logger = logging.getLogger(__name__)

#: The packages the side menu looks up (P2 §1): the assistant itself and what a chat needs.
REQUIRED_MODULES = ("flextool.assistant", "anthropic", "openai", "keyring", "platformdirs")
#: Imported on the first click (inside ``try``): the SDKs, the keyring, the store and the session.
FIRST_USE_MODULES = ("anthropic", "openai", "keyring", "flextool.assistant.store", "flextool.assistant.session",
                     "flextool.gui.assistant.chat_window", "flextool.gui.assistant.settings_dialog")
#: The waits of a project change and of FlexTool closing (P2 §5.7).
PROJECT_CHANGE_BUDGET_S = 5.0
CLOSE_BUDGET_S = 3.0
#: The part of the close budget kept for the runtime file and the purge.
CLOSE_RESERVE_S = 0.5
#: How long the runtime-context writer may take to delete its file at a project change.
WRITER_STOP_S = 1.0
STEP_S = 0.05
#: How often a blocking wait of the Tk thread serves the calls other threads hand it.
PUMP_STEP_S = 0.02

FindSpec = Callable[[str], Any]


def _default_find_spec(name: str) -> Any:
    try:
        return importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None


class AssistantEntry:
    """The AI assistant of the main window *main*.

    *config_root* is the assistant's settings folder (``None``: the user's); *find_spec* looks a package up (a
    test seam); *window_options* go to every :class:`~flextool.gui.assistant.chat_window.ChatWindow` (backends,
    probe, work area in the tests); *settings_options* to the settings dialog.
    """

    def __init__(self, main: Any, *, config_root: Path | None = None, find_spec: FindSpec | None = None,
                 window_options: dict[str, Any] | None = None, settings_options: dict[str, Any] | None = None,
                 publisher_interval_ms: int | None = None) -> None:
        self.main = main
        self.config_root = config_root
        self._find_spec = find_spec or _default_find_spec
        self._window_options = dict(window_options or {})
        self._settings_options = dict(settings_options or {})
        self._publisher_interval_ms = publisher_interval_ms
        self.missing = tuple(name for name in REQUIRED_MODULES if self._find_spec(name) is None)
        self.window: ChatWindow | None = None
        self.settings_dialog: AssistantSettingsDialog | None = None
        self.writer: RuntimeContextWriter | None = None
        self.publisher: ContextPublisher | None = None
        self.context_project: str | None = None
        self.frame: ttk.Frame | None = None
        self.chat_button: ttk.Button | None = None
        self.settings_button: ttk.Button | None = None
        self._align_with: tk.Misc | None = None
        self._opening = False
        self._finishing: list[AssistantWorker] = []
        #: The purge thread of the last start or close (for the tests).
        self.purge_thread: threading.Thread | None = None

    @property
    def available(self) -> bool:
        return not self.missing

    # --- the side menu (§3) ---------------------------------------------------------------------------------
    def build(self, side_menu: ttk.Frame, row: int = 5, *, align_with: tk.Misc | None = None) -> ttk.Frame:
        """Put "AI assistant…" and ⚙ into *side_menu* at *row*.  The pair is as wide as *align_with* (the
        ``width=22`` buttons of the column): the column itself can be wider (the Debug row), so the frame is not
        stretched to it but its first column gets the width the ⚙ button leaves."""
        from flextool.gui.assistant.notices import settings_symbol
        from flextool.gui.hover_tooltip import attach_tooltip

        frame = ttk.Frame(side_menu)
        frame.grid(row=row, column=0, sticky="w", pady=(0, 2))
        frame.columnconfigure(0, weight=1)
        self.frame = frame
        self.chat_button = ttk.Button(frame, text=strings.ENTRY_BUTTON, command=self.open_chat)
        self.chat_button.grid(row=0, column=0, sticky="ew")
        self.settings_button = ttk.Button(frame, text=settings_symbol(frame), width=2,
                                          command=lambda: self.open_settings())
        self.settings_button.grid(row=0, column=1, sticky="e")
        if self.available:
            attach_tooltip(self.chat_button, strings.ENTRY_TOOLTIP, wraplength=_wrap(frame))
            attach_tooltip(self.settings_button, strings.ENTRY_SETTINGS_TOOLTIP, wraplength=_wrap(frame))
        else:
            logger.warning("The AI assistant is unavailable: %s not found", ", ".join(self.missing))
            for button in (self.chat_button, self.settings_button):
                button.state(["disabled"])
                attach_tooltip(button, strings.UNAVAILABLE, wraplength=_wrap(frame))
        if align_with is not None:
            self._align_with = align_with
            align_with.bind("<Configure>", lambda _e: self._align(), add="+")
            frame.after_idle(self._align)
        return frame

    def _align(self) -> None:
        if self.frame is None or self._align_with is None or self.settings_button is None:
            return
        try:
            target = self._align_with.winfo_reqwidth()
            gear = self.settings_button.winfo_reqwidth()
            self.frame.columnconfigure(0, minsize=max(0, target - gear))
        except tk.TclError:
            pass

    # --- opening the chat and the settings ----------------------------------------------------------------
    def _project_path(self, name: str) -> Path:
        from flextool.gui.project_utils import get_projects_dir

        return get_projects_dir() / name

    def _import_first_use(self) -> bool:
        """Import what a chat needs; on a failure say so and log the details."""
        try:
            for name in FIRST_USE_MODULES:
                importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 — a broken installation is a message, not a crash
            logger.exception("The AI assistant could not be loaded")
            messagebox.showerror(strings.ENTRY_BUTTON.rstrip("…"),
                                 f"{strings.UNAVAILABLE} ({type(exc).__name__})", parent=self.main)
            return False
        return True

    def open_chat(self, *, welcome: bool = True) -> None:
        """Open the project's chat window (the welcome first when it has not been seen and *welcome* is true), or
        raise it.  The welcome's [Open chat] from the settings path opens it with *welcome* false: the welcome was
        just closed, and recording that it was seen may still be under way."""
        if not self.available or self._opening:
            return
        name = getattr(self.main, "current_project", None)
        if not name:
            messagebox.showinfo("No project", "Please select or create a project first.", parent=self.main)
            return
        window = self.window
        if window is not None and not window.closing and window.winfo_exists():
            window.deiconify()
            window.lift()
            window.focus_input()
            return
        if not self._import_first_use():
            return
        if not welcome:
            self._open_window(name)
            return
        from flextool.gui.assistant.notices import with_welcome

        self._opening = True
        try:
            shown = with_welcome(self.main, lambda: self._open_window(name),
                                 open_chat=lambda: self._open_window(name),
                                 open_settings=lambda: self._open_window(name, settings=True),
                                 config_root=self.config_root)
        except Exception:
            self._opening = False
            raise
        if shown is None:
            self._opening = False
            return

        def gone(event: tk.Event) -> None:  # type: ignore[type-arg]
            if event.widget is shown and self.window is None:
                self._opening = False      # closed without opening the chat (FlexTool closing)

        shown.bind("<Destroy>", gone, add="+")

    def _open_window(self, name: str, *, settings: bool = False) -> None:
        from flextool.gui.assistant.chat_window import ChatWindow
        from flextool.gui.assistant.gui_state import context_snapshot

        self._opening = False
        main = self.main
        if getattr(main, "current_project", None) != name:
            return                                  # the project changed while the welcome was open
        self.window = ChatWindow(main, self._project_path(name), name,
                                 global_settings=getattr(main, "global_settings", None),
                                 open_settings=self.open_settings,
                                 snapshot=lambda: context_snapshot(main, name),
                                 exec_rect=self._exec_rect, on_closed=self._window_closed,
                                 config_root=self.config_root, **self._window_options)
        if settings:
            self._show_settings(self.window)

    def _exec_rect(self) -> tuple[int, int, int, int] | None:
        from flextool.gui.assistant.geometry import parse_geometry

        window = getattr(self.main, "execution_window", None)
        try:
            if window is None or not window.winfo_exists() or not window.winfo_viewable():
                return None
            return parse_geometry(window.wm_geometry())
        except tk.TclError:
            return None

    def _window_closed(self, window: ChatWindow) -> None:
        self._finishing += [w for w in window.workers() if not w.exited.is_set()]
        if self.window is window:
            self.window = None

    def open_settings(self, parent: tk.Misc | None = None) -> None:
        """Open the assistant settings over *parent* (the main window by default; the welcome first when it has
        not been seen).  Works without a project."""
        if not self.available:
            return
        dialog = self.settings_dialog
        if dialog is not None and dialog.alive:
            try:
                dialog.deiconify()
                dialog.lift()
                return
            except tk.TclError:
                self.settings_dialog = None
        if not self._import_first_use():
            return
        from flextool.gui.assistant.notices import with_welcome

        owner = parent or self.main
        with_welcome(owner, lambda: self._show_settings(owner), open_chat=lambda: self.open_chat(welcome=False),
                     config_root=self.config_root)

    def _show_settings(self, parent: tk.Misc) -> None:
        from flextool.gui.assistant.settings_dialog import AssistantSettingsDialog

        try:
            if not parent.winfo_exists():
                parent = self.main
        except tk.TclError:
            parent = self.main
        dialog = AssistantSettingsDialog(parent, getattr(self.main, "post_to_main", None), self._settings_saved,
                                         config_root=self.config_root, **self._settings_options)
        self.settings_dialog = dialog

        def gone(event: tk.Event) -> None:  # type: ignore[type-arg]
            if event.widget is dialog and self.settings_dialog is dialog:
                self.settings_dialog = None

        dialog.bind("<Destroy>", gone, add="+")

    def _settings_saved(self) -> None:
        """The settings dialog's Save: every open chat prepares again at its next question."""
        window = self.window
        if window is not None and not window.closing:
            try:
                window.settings_changed()
            except tk.TclError:
                pass

    # --- the runtime context file (P2-O1) -----------------------------------------------------------------
    def project_opened(self, name: str) -> None:
        """A project is open: keep its runtime context file (also without a chat window)."""
        self._stop_context(wait=0.0)
        if self._find_spec("flextool.assistant") is None:
            return
        try:
            from flextool.assistant.runtime_context import RuntimeContextWriter
            from flextool.gui.assistant.gui_state import ContextPublisher

            writer = RuntimeContextWriter(self._project_path(name), name, config_root=self.config_root)
            writer.start()
            options = {} if self._publisher_interval_ms is None else {"interval_ms": self._publisher_interval_ms}
            publisher = ContextPublisher(self.main, writer, name, **options)
            publisher.start()
        except Exception:  # noqa: BLE001 — an agent then just does not see the window's selections
            logger.exception("The AI assistant's runtime context file could not be started")
            return
        self.writer, self.publisher, self.context_project = writer, publisher, name

    def ensure_open(self, name: str | None) -> None:
        """Restart the runtime context file of *name* when it is not kept (after a refused or failed change)."""
        if name and (self.writer is None or self.context_project != name):
            self.project_opened(name)

    def _stop_context(self, wait: float) -> bool:
        publisher, writer = self.publisher, self.writer
        self.publisher = self.writer = None
        self.context_project = None
        if publisher is not None:
            publisher.stop()
        if writer is not None:
            return writer.stop(wait=wait)
        return True

    # --- workers ------------------------------------------------------------------------------------------
    def live_workers(self) -> list[AssistantWorker]:
        """The workers of this process that have not ended: the open window's and those still finishing."""
        workers: list[AssistantWorker] = []
        if self.window is not None:
            workers += self.window.workers()
        workers += self._finishing
        module = sys.modules.get("flextool.gui.assistant.worker")
        if module is not None:
            workers += module.finishing_workers()
        seen: set[int] = set()
        live = []
        for worker in workers:
            if id(worker) not in seen and not worker.exited.is_set():
                seen.add(id(worker))
                live.append(worker)
        self._finishing = [w for w in self._finishing if not w.exited.is_set()]
        return live

    def _wait(self, workers: list[AssistantWorker], deadline: float) -> bool:
        """Wait until every worker has ended or *deadline* (``time.monotonic``) passes; ``True`` when all ended.
        The Tk thread keeps serving the Tcl calls other threads hand it meanwhile (:func:`pumped_wait`)."""
        collect_tk_garbage()
        for worker in workers:
            if not pumped_wait(self.main, worker.exited.is_set, deadline):
                return False
        return True

    def _close_window(self) -> list[AssistantWorker]:
        """Close the chat window without asking; returns its workers (still finishing)."""
        window = self.window
        if window is None:
            return []
        workers = window.workers()
        if not window.closing:
            try:
                window.close(ask=False)
            except tk.TclError:
                pass
        if self.window is window:
            self._window_closed(window)
        return workers

    def _wait_with_status(self, workers: list[AssistantWorker], deadline: float) -> bool:
        """Wait for *workers*, saying so in the main window's title while it takes time (the main window has no
        status line); the main window cannot be used meanwhile, as in any project change."""
        if all(w.exited.is_set() for w in workers):
            return True
        main = self.main
        try:
            title = main.title()
            main.title(f"{title} — {strings.PROJECT_WAITING}")
            main.configure(cursor="watch")
            main.update_idletasks()
        except tk.TclError:
            title = None
        try:
            return self._wait(workers, deadline)
        finally:
            if title is not None:
                try:
                    main.title(title)
                    main.configure(cursor="")
                except tk.TclError:
                    pass

    def close_for_project(self, budget: float | None = None) -> bool:
        """The open project is changing: close the chat window and the runtime file, and wait for the workers
        within *budget* seconds (:data:`PROJECT_CHANGE_BUDGET_S`).  ``False`` when one is still running (a slow
        tool call)."""
        budget = PROJECT_CHANGE_BUDGET_S if budget is None else budget
        deadline = time.monotonic() + budget
        self._close_window()
        self._stop_context(wait=min(WRITER_STOP_S, budget))
        return self._wait_with_status(self.live_workers(), deadline)

    def before_project_change(self, name: str) -> bool:
        """The Project dialog is about to rename or delete the project *name*: ``False`` refuses it, because a
        worker of that project could still write.  For the open project the chat window closes first; on a
        refusal its runtime file is kept again."""
        current = getattr(self.main, "current_project", None)
        if name == current:
            if self.close_for_project():
                return True
            self.ensure_open(current)
            return False
        target = _resolved(self._project_path(name))
        workers = [w for w in self.live_workers() if _resolved(Path(w.project_path)) == target]
        return self._wait_with_status(workers, time.monotonic() + PROJECT_CHANGE_BUDGET_S)

    def close_window(self) -> None:
        """Close the chat window without asking (the layout reset)."""
        self._close_window()

    # --- start and close ----------------------------------------------------------------------------------
    def _purge(self) -> threading.Thread | None:
        if self._find_spec("flextool.assistant") is None:
            return None
        config_root = self.config_root

        def run() -> None:
            from flextool.assistant import runtime_context
            from flextool.assistant.store import open_store
            from flextool.gui.project_utils import wait_project_hooks

            wait_project_hooks()
            try:
                open_store(config_root).purge_expired()
            except Exception:  # noqa: BLE001 — the purge runs again at the next start and close
                logger.exception("Expired AI assistant conversations could not be deleted")
            try:
                runtime_context.cleanup(config_root)
            except Exception:  # noqa: BLE001
                logger.exception("Old AI assistant runtime files could not be deleted")

        thread = threading.Thread(target=run, name="flextool-assistant-purge", daemon=True)
        thread.start()
        self.purge_thread = thread
        return thread

    def startup_purge(self) -> None:
        """Delete expired conversations and the runtime files of processes that are gone (daemon thread)."""
        self._purge()

    def on_flextool_close(self, budget: float | None = None) -> None:
        """FlexTool is closing: stop every answer without asking, keep what was received, delete the runtime
        file and purge, within *budget* seconds (:data:`CLOSE_BUDGET_S`); whatever is unfinished is left to its
        daemon thread."""
        deadline = time.monotonic() + (CLOSE_BUDGET_S if budget is None else budget)
        for window in (self.main, self.window):
            if window is None:
                continue
            try:
                window.withdraw()
            except tk.TclError:
                pass
        window = self.window
        workers = self._close_window()
        # The runtime file goes at once (its thread deletes it): an agent must not see a window that is closing.
        writer = self.writer
        self._stop_context(wait=0.0)
        open_chats = getattr(window, "_last_open_chats_write", None) if window is not None else None
        if open_chats is not None:
            pumped_wait(self.main, lambda: not open_chats.is_alive(), deadline)
        # The workers get the budget but a reserve for the writer and the purge.
        reserve = min(CLOSE_RESERVE_S, (deadline - time.monotonic()) / 3)
        self._wait(workers + self.live_workers(), deadline - reserve)
        if writer is not None:
            writer.stop(wait=max(0.0, deadline - time.monotonic()))
        # A project rename or deletion still moving or deleting its chats gets the rest of the budget: a daemon
        # killed in the middle of a move can leave a chat under both names (review G18 M2).
        from flextool.gui.project_utils import project_hooks_idle

        pumped_wait(self.main, project_hooks_idle, deadline)
        thread = self._purge()
        if thread is not None:
            thread.join(max(0.0, deadline - time.monotonic()))


#: What the Tk thread serves while it waits: the calls other threads hand it (and file events), never the
#: window events (a click or a key) or the ``after`` timers.
_PUMP_FLAGS = _tkinter.FILE_EVENTS | _tkinter.DONT_WAIT
#: The most queued calls served in one step of a wait.
_PUMP_BURST = 100


def pumped_wait(main: Any, done: Callable[[], bool], deadline: float, step: float = PUMP_STEP_S) -> bool:
    """Wait on the Tk thread until *done()* or *deadline* (``time.monotonic``); ``True`` when *done*.

    Every *step* seconds the Tk thread serves the Tcl calls other threads have handed it (review G18 critique M1):
    a worker that touches a Tk object — a ``Variable`` or ``PhotoImage`` finalizer run by its garbage collection,
    for one — waits for the Tk thread, so a plain wait for that worker would last its whole budget.  Window events
    and timers stay queued (``FILE_EVENTS`` without ``WINDOW_EVENTS`` / ``TIMER_EVENTS``), so nothing the user does
    runs during the wait.  *main* without a Tk interpreter (a test stand-in) is a plain wait."""
    app = getattr(main, "tk", None)
    pump = getattr(app, "dooneevent", None)
    while not done():
        left = deadline - time.monotonic()
        if left <= 0:
            return done()
        if pump is not None:
            try:
                for _ in range(_PUMP_BURST):
                    if not pump(_PUMP_FLAGS):
                        break
            except (tk.TclError, RuntimeError):
                pump = None
        time.sleep(min(step, left))
    return True


def collect_tk_garbage() -> None:
    """Collect the garbage on the Tk thread before it blocks (review G18, C7-m5): finalizers of Tk objects left
    in reference cycles then run here, not on a worker that would wait for the Tk thread (the pump of
    :func:`pumped_wait` covers what becomes garbage during the wait)."""
    gc.collect()


def _resolved(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def _wrap(widget: tk.Misc) -> int:
    from flextool.gui.ui_metrics import get_metrics

    try:
        return get_metrics(widget).cw * 50
    except tk.TclError:
        return 400
