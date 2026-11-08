"""The AI assistant's chat window (P2 §4): one non-modal window per project with its chats as tabs.

Layout (§4.1): where the active chat's data goes, [Chats ▾] [Settings ⚙] [How this works], the relaxed privacy
line, a note line, the ``ttk.Notebook`` of :class:`~flextool.gui.assistant.chat_tab.ChatTab`, and this month's
usage of the active chat's provider.

* **Tabs** (P2-O13, §4.3).  A chat is titled with its number (or the name the user gave), with a state prefix.
  An empty "New chat" tab is always last; sending there makes it a chat and a new "New chat" appears.  Tabs that
  do not fit the width are hidden, least recently used first; the Chats menu lists every open chat (hidden ones
  too), reopens and deletes saved chats, renames and closes.  Ctrl+Tab / Ctrl+Shift+Tab (``ISO_Left_Tab`` on
  X11) and Ctrl+PgDn / Ctrl+PgUp cycle through all chats, hidden ones shown; Ctrl+T goes to "New chat", Ctrl+W
  closes the current chat (Command+T/W on macOS too).
* **Restoring** (P2-O14, §4.3a).  The open chats are kept per project (debounced, on a daemon thread) and
  restored when the window opens again; a chat open in another FlexTool window or terminal is skipped and said
  so.  Answers in progress are never restarted.
* **Closing** (P2-O15, §5.7).  Closing a chat stops its answer, saves what was received and lets its worker
  finish on its own; at retention 0 it asks first.  Reopening a chat whose worker is still finishing waits up to
  5 s.  The window close asks once for all unsaved chats (not when FlexTool closes or the project changes).
* **Data notices** (§5.5): one :class:`~flextool.gui.assistant.notices.ConsentDialog` at a time, in order; a
  key accepted in this window answers later notices with the same key at once.
* **Threads** (P2-D2): the window never calls the network, the keyring or the store on the Tk thread (the
  settings and the route of the saved settings are read once when it opens).  Workers' events arrive through
  their queues, store and ledger results through :meth:`ChatWindow.call_soon`; one ``after(50)`` poll with a
  30 ms budget, round-robin over the tabs, handles both.
* **Size and place** (P2-O11/O12, §4.2): the work area's height of the main window's monitor, about 80
  characters wide, on the side the Execution window leaves free; a resize is remembered per monitor
  configuration in the main window's global settings.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from tkinter import ttk
from typing import TYPE_CHECKING, Any

from flextool.gui.assistant import strings
from flextool.gui.assistant.chat_tab import ChatTab, run_in_thread
from flextool.gui.assistant.gate import PROCESS_GATE, ProviderGate
from flextool.gui.assistant.notices import (
    ConfirmDialog,
    ConsentDialog,
    RenameChatDialog,
    WelcomeDialog,
    install_shield,
    is_aqua,
    settings_symbol,
)

if TYPE_CHECKING:
    from flextool.assistant.context import ContextSnapshot
    from flextool.assistant.store import ConversationInfo, Store
    from flextool.gui.assistant.worker import AssistantWorker, ConsentNeeded, OpenRefused

logger = logging.getLogger(__name__)

POLL_MS = 50
POLL_BUDGET_S = 0.030
#: A tab's width beyond its text (N2).
TAB_PAD = 34
#: Saved chats listed in the Reopen and Delete menus.
MENU_CHATS = 30
#: How long a reopen waits for the closed chat's worker, and how often it looks.
REOPEN_WAIT_S = 5.0
REOPEN_STEP_MS = 100
OPEN_CHATS_DEBOUNCE_MS = 1000
TOOLTIP_DELAY_MS = 400

Rect = tuple[int, int, int, int]


def menu_entry(number: int | None, name: str | None, created: str, first_words: str) -> str:
    """"{number} · {name, if renamed} · {6 Nov 14:02} · {first words of the first question}" (P2-O17)."""
    parts = [str(number) if number is not None else "–"]
    if name:
        parts.append(name)
    when = _local_time(created)
    if when:
        parts.append(when)
    words = " ".join(first_words.split())
    if len(words) > strings.MENU_FIRST_WORDS_MAX:
        words = words[:strings.MENU_FIRST_WORDS_MAX - 1].rstrip() + "…"
    if words:
        parts.append(words)
    return strings.MENU_ENTRY_PARTS_SEPARATOR.join(parts)


def _local_time(created: str) -> str:
    try:
        moment = datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).astimezone()
    except (TypeError, ValueError):
        return ""
    return f"{moment.day} {moment:%b} {moment:%H:%M}"


def styled_menu(parent: tk.Misc) -> tk.Menu:
    """A ``tk.Menu`` in the ttk theme's colours (``MainWindow._styled_popup_menu``)."""
    style = ttk.Style()
    bg = style.lookup("TFrame", "background") or parent.cget("background")
    fg = style.lookup("TLabel", "foreground") or "#000000"
    return tk.Menu(parent, tearoff=0, background=bg, foreground=fg, activebackground=fg, activeforeground=bg,
                   selectcolor=fg, font="TkDefaultFont", borderwidth=0)


class ConsentQueue:
    """The data notices of the window's chats: one dialog at a time, first come first asked; a key accepted here
    answers the later requests with the same key at once."""

    def __init__(self, window: ChatWindow) -> None:
        self.window = window
        self.accepted: set[Any] = set()
        self._queue: deque[tuple[ChatTab, ConsentNeeded]] = deque()
        self.current: tuple[ChatTab, ConsentNeeded, ConsentDialog] | None = None

    def request(self, tab: ChatTab, event: ConsentNeeded) -> None:
        if event.consent_key in self.accepted:
            tab.consent_answer(event.job_id, True)
            return
        self._queue.append((tab, event))
        tab.notice_queued()
        self._next()

    def _live(self, tab: ChatTab, event: ConsentNeeded) -> bool:
        return not tab.closed and tab.active_job is not None and tab.active_job.job_id == event.job_id

    def _next(self) -> None:
        while self.current is None and self._queue:
            tab, event = self._queue.popleft()
            if not self._live(tab, event):
                continue
            if event.consent_key in self.accepted:
                tab.consent_answer(event.job_id, True)
                continue
            window = self.window
            try:
                window.deiconify()
                window.lift()
            except tk.TclError:
                pass
            window.select(tab)
            tab.notice_shown()
            dialog = ConsentDialog(window, event.provider_label, event.notice, event.indicator, tab.label(),
                                   lambda accepted, t=tab, e=event: self._answered(t, e, accepted))
            self.current = (tab, event, dialog)

    def _answered(self, tab: ChatTab, event: ConsentNeeded, accepted: bool) -> None:
        self.current = None
        if accepted:
            self.accepted.add(event.consent_key)
        if self._live(tab, event):
            tab.consent_answer(event.job_id, accepted)
        if not self.window.closing:
            self._next()

    def remove(self, tab: ChatTab, job_id: int | None = None) -> None:
        """Drop *tab*'s requests (only *job_id*'s when given); close its dialog if it is open."""
        self._queue = deque((t, e) for t, e in self._queue
                            if not (t is tab and (job_id is None or e.job_id == job_id)))
        current = self.current
        if current is not None and current[0] is tab and (job_id is None or current[1].job_id == job_id):
            current[2].cancel()

    def close_all(self) -> None:
        self._queue.clear()
        if self.current is not None:
            self.current[2].cancel()

    def queued(self) -> int:
        return len(self._queue)


class ChatWindow(tk.Toplevel):
    """The chat window of the project at *project_path*.

    *open_settings(parent)* opens the assistant settings (the Settings button is unavailable without it);
    *snapshot()* gives the FlexTool window's selections at Send; *exec_rect()* the Execution window's frame
    when it is open; *work_area(window)* the work area (default :func:`ui_metrics.monitor_work_area`);
    *global_settings* is the main window's shared object the remembered size is written into at close.
    *gate*, *tool_lock*, *config_root*, *backend_factory* and *probe* go to every chat's worker.
    """

    def __init__(self, master: tk.Misc, project_path: Path, project_name: str, *,
                 global_settings: Any = None, open_settings: Callable[[tk.Misc], None] | None = None,
                 snapshot: Callable[[], ContextSnapshot | None] | None = None,
                 gate: ProviderGate | None = None, tool_lock: threading.Lock | None = None,
                 config_root: Path | None = None, backend_factory: Callable | None = None,
                 probe: Callable | None = None, work_area: Callable[[tk.Misc], Rect | None] | None = None,
                 exec_rect: Callable[[], Rect | None] | None = None,
                 menu_factory: Callable[[tk.Misc], tk.Menu] | None = None,
                 on_closed: Callable[[ChatWindow], None] | None = None, restore: bool = True) -> None:
        super().__init__(master)
        self.withdraw()
        self.project_path = Path(project_path)
        self.project_name = project_name
        self.title(strings.WINDOW_TITLE.format(project=project_name))
        self.aqua = is_aqua(self)
        self.closing = False
        self.gate = gate or PROCESS_GATE
        self.tool_lock = tool_lock
        self.config_root = config_root
        self._backend_factory = backend_factory
        self._probe = probe
        self._open_settings = open_settings
        self._snapshot = snapshot
        self._global_settings = global_settings
        self._work_area = work_area
        self._exec_rect = exec_rect
        self._menu_factory = menu_factory or styled_menu
        self._on_closed = on_closed
        self._calls: queue.SimpleQueue = queue.SimpleQueue()
        self._poll_id: str | None = None
        self._rr = 0
        self._fit_scheduled = False
        self._open_chats_id: str | None = None
        self._recent: list[ChatTab] = []
        self.tabs: list[ChatTab] = []
        self._saved: list[ConversationInfo] = []
        self._restoring: set[str] = set()
        self._restore_selected: str | None = None
        self._not_restored = 0
        self._store: Store | None = None
        self.consents = ConsentQueue(self)
        self._read_settings()
        install_shield(self)
        self._build()
        self.bind_tab_commands(self)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Destroy>", self._on_destroy, add="+")
        self._place()
        self._poll_id = self.after(POLL_MS, self._poll)
        if restore:
            self._restore_open_chats()
        self.refresh_saved()
        self.refresh_footer()
        self.after_idle(self.focus_input)

    # --- settings read once at open (P2-D2) -----------------------------------------------------------------
    def _read_settings(self) -> None:
        from flextool.assistant import endpoint
        from flextool.assistant.providers import PRESETS, relaxed_knobs
        from flextool.assistant.settings import load_settings

        try:
            settings = load_settings(self.config_root)
        except Exception:  # noqa: BLE001 — an unreadable settings file reads as the defaults
            logger.exception("The AI assistant settings could not be read")
            from flextool.assistant.settings import Settings

            settings = Settings()
        self._expiry = settings.conversation_expiry_days
        self._settings_route = endpoint.route_from_settings(settings)
        preset = PRESETS.get(settings.provider.preset or "")
        self._settings_relaxed = tuple(relaxed_knobs(preset, settings.provider.knobs)) if preset else ()

    def settings_changed(self) -> None:
        """The settings were saved: every chat prepares again at its next question, the header follows."""
        self._read_settings()
        for tab in self.tabs:
            tab.settings_changed()
        self._refresh_header()
        self.refresh_saved()

    @property
    def saving(self) -> bool:
        return self._expiry > 0

    def default_provider(self) -> str:
        return self._settings_route.label if self._settings_route is not None else "the AI service"

    # --- layout ---------------------------------------------------------------------------------------------
    def _build(self) -> None:
        from flextool.gui.ui_metrics import get_metrics

        metrics = get_metrics(self)
        self._cw, self._lh = metrics.cw, metrics.lh
        outer = ttk.Frame(self, padding=(metrics.cw, metrics.lh // 2))
        outer.pack(fill="both", expand=True)
        self.route_label = ttk.Label(outer, text="", anchor="w", justify="left")
        self.route_label.pack(fill="x")
        row = ttk.Frame(outer)
        row.pack(fill="x", pady=(metrics.lh // 3, 0))
        self.chats_button = ttk.Button(row, text=strings.CHATS + strings.CHATS_ARROW, command=self.post_chats_menu)
        self.chats_button.pack(side="left")
        self.settings_button = ttk.Button(row, text=strings.settings_button_text(settings_symbol(self)),
                                          command=self.open_settings)
        self.settings_button.pack(side="left", padx=(metrics.cw // 2, 0))
        if self._open_settings is None:
            self.settings_button.state(["disabled"])
        self.how_button = ttk.Button(row, text=strings.HOW_IT_WORKS, command=self.show_welcome,
                                     style="Toolbutton", takefocus=1)
        self.how_button.pack(side="left", padx=(metrics.cw // 2, 0))
        self.relaxed_label = ttk.Label(outer, text="", anchor="w", justify="left")
        self.relaxed_label.pack(fill="x")
        self.note_label = ttk.Label(outer, text="", anchor="w", justify="left")
        self.note_label.pack(fill="x")
        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True, pady=(metrics.lh // 3, 0))
        self.footer_label = ttk.Label(outer, text="", anchor="w")
        self.footer_label.pack(fill="x", pady=(metrics.lh // 3, 0))
        self.chats_menu = self._menu_factory(self)
        self.chats_menu.configure(postcommand=self._fill_chats_menu)
        self.tab_menu = self._menu_factory(self)
        self.new_tab = ChatTab(self, new=True)
        self.notebook.add(self.new_tab, text=strings.NEW_CHAT)
        self._tips = _TabTips(self)
        nb = self.notebook
        self.bind_tab_commands(nb)
        nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        nb.bind("<Configure>", lambda _e: self.schedule_fit())
        menu_buttons = ("<Button-2>", "<Control-Button-1>") if self.aqua else ("<Button-3>",)
        for sequence in menu_buttons:
            nb.bind(sequence, self._on_tab_menu_click)
        nb.bind("<Button-3>" if self.aqua else "<Button-2>", self._on_middle_click)
        nb.bind("<Double-Button-1>", self._on_double_click)
        for sequence in ("<Shift-F10>", "<Key-Menu>", "<App>"):
            try:
                nb.bind(sequence, lambda _e: (self._post_tab_menu(self.current_tab(), None), "break")[1])
            except tk.TclError:
                pass
        for widget in (self.route_label, self.relaxed_label, self.note_label):
            outer.bind("<Configure>", lambda e, w=widget: w.configure(wraplength=max(100, e.width - 8)), add="+")
        self._refresh_header()

    # --- tab commands (§4.7) ----------------------------------------------------------------------------------
    def bind_tab_commands(self, widget: tk.Misc) -> None:
        """Bind the window's tab commands on *widget*'s own tag (each returns "break")."""
        commands: list[tuple[str, Callable[[], None]]] = [
            ("<Control-Tab>", lambda: self.cycle(1)), ("<Control-Next>", lambda: self.cycle(1)),
            ("<Control-Shift-Tab>", lambda: self.cycle(-1)), ("<Control-Prior>", lambda: self.cycle(-1)),
            ("<Control-ISO_Left_Tab>", lambda: self.cycle(-1)),
            ("<Control-Shift-ISO_Left_Tab>", lambda: self.cycle(-1)),
            ("<Control-t>", self.go_to_new_chat), ("<Control-w>", self.close_current),
        ]
        if self.aqua:
            commands += [("<Command-t>", self.go_to_new_chat), ("<Command-w>", self.close_current)]
        for sequence, command in commands:
            try:
                widget.bind(sequence, lambda _e, c=command: (c(), "break")[1])
            except tk.TclError:
                pass                    # ISO_Left_Tab is an X11 key name (as notebook.tcl guards it)

    def all_tabs(self) -> list[ChatTab]:
        """Every tab in notebook order: the chats, then "New chat"."""
        return list(self.tabs) + [self.new_tab]

    def current_tab(self) -> ChatTab:
        try:
            name = self.notebook.select()
            return self.nametowidget(name) if name else self.new_tab
        except (tk.TclError, KeyError):
            return self.new_tab

    def is_shown(self, tab: ChatTab) -> bool:
        try:
            return self.current_tab() is tab and bool(self.winfo_viewable())
        except tk.TclError:
            return False

    def select(self, tab: ChatTab) -> None:
        if tab.closed:
            return
        self.notebook.select(tab)
        self._touch(tab)
        self._fit_tabs()

    def cycle(self, step: int) -> None:
        order = self.all_tabs()
        current = self.current_tab()
        index = order.index(current) if current in order else len(order) - 1
        self.select(order[(index + step) % len(order)])

    def go_to_new_chat(self) -> None:
        self.select(self.new_tab)
        self.new_tab.input.focus_set()

    def close_current(self) -> None:
        tab = self.current_tab()
        if not tab.is_new:
            self.close_chat(tab)

    def focus_input(self) -> None:
        try:
            self.current_tab().input.focus_set()
        except tk.TclError:
            pass

    def _touch(self, tab: ChatTab) -> None:
        if tab.is_new:
            return
        if tab in self._recent:
            self._recent.remove(tab)
        self._recent.insert(0, tab)

    def _on_tab_changed(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        tab = self.current_tab()
        self._touch(tab)
        tab.shown()
        self._refresh_header()
        self.refresh_footer()
        self.schedule_open_chats()
        self.schedule_fit()
        try:
            focused = self.focus_get()
        except (tk.TclError, KeyError):
            focused = None
        if focused is not self.notebook:
            tab.input.focus_set()

    # --- the tabs that fit (§4.3) -----------------------------------------------------------------------------
    def schedule_fit(self) -> None:
        if not self._fit_scheduled and not self.closing:
            self._fit_scheduled = True
            self.after_idle(self._fit_tabs)

    def _tab_font(self) -> tkfont.Font:
        name = ttk.Style().lookup("TNotebook.Tab", "font") or "TkDefaultFont"
        try:
            return tkfont.nametofont(str(name))
        except tk.TclError:
            return tkfont.nametofont("TkDefaultFont")

    def _fit_tabs(self) -> None:
        self._fit_scheduled = False
        if self.closing:
            return
        nb = self.notebook
        try:
            width = nb.winfo_width()
            if width <= 1:
                width = self.winfo_width()
        except tk.TclError:
            return
        font = self._tab_font()
        budget = width - (font.measure(strings.NEW_CHAT) + TAB_PAD)
        current = self.current_tab()
        order = [t for t in self._recent if t in self.tabs]
        order += [t for t in self.tabs if t not in order]
        if current in order:
            order.remove(current)
            order.insert(0, current)
        shown: set[ChatTab] = set()
        used = 0
        for tab in order:
            needed = font.measure(tab.tab_text()) + TAB_PAD
            if (tab is current) or used + needed <= budget:
                shown.add(tab)
                used += needed
        for tab in self.tabs:
            try:
                hidden = nb.tab(tab, "state") == "hidden"
                if tab in shown and hidden:
                    nb.add(tab)
                elif tab not in shown and not hidden:
                    nb.hide(tab)
            except tk.TclError:
                continue
        self._refresh_chats_button()

    def hidden_tabs(self) -> list[ChatTab]:
        out = []
        for tab in self.tabs:
            try:
                if self.notebook.tab(tab, "state") == "hidden":
                    out.append(tab)
            except tk.TclError:
                continue
        return out

    def _refresh_chats_button(self) -> None:
        hidden = self.hidden_tabs()
        text = strings.CHATS_MORE.format(n=len(hidden)) if hidden else strings.CHATS
        if any(t.needs_attention() for t in hidden):
            text = strings.CHATS_ATTENTION + text
        self.chats_button.configure(text=text + strings.CHATS_ARROW)

    def tab_changed(self, tab: ChatTab, *, text_only: bool = False) -> None:
        """A tab's state, number, name or status changed."""
        if tab.closed or self.closing:
            return
        try:
            if not text_only:
                self.notebook.tab(tab, text=tab.tab_text())
        except tk.TclError:
            return
        if not text_only:
            self.schedule_fit()
            self._refresh_chats_button()
        if tab is self.current_tab():
            self._refresh_header()

    # --- the header and the footer ---------------------------------------------------------------------------
    def _refresh_header(self) -> None:
        from flextool.assistant import endpoint

        tab = self.current_tab()
        if tab.prepared is not None:
            route_text = tab.prepared.indicator
            relaxed = tab.prepared.relaxed
        elif self._settings_route is not None:
            route_text = endpoint.indicator_text(self._settings_route)
            relaxed = self._settings_relaxed
        else:
            route_text = strings.CHOOSE_PROVIDER
            relaxed = ()
        self.route_label.configure(text=route_text)
        self.relaxed_label.configure(text=" ".join(strings.RELAXED.format(report=r.rstrip(".")) for r in relaxed))

    def note(self, text: str) -> None:
        self.note_label.configure(text=text)

    def refresh_footer(self) -> None:
        """This month's usage of the active chat's provider (the ledger read on a daemon thread)."""
        label = self.current_tab().provider

        def read() -> None:
            from flextool.assistant.usage import Ledger

            try:
                data = Ledger(self.config_root).read()
            except Exception:  # noqa: BLE001 — no footer then
                logger.exception("The usage record could not be read")
                return
            self.call_soon(self._show_footer, label, data)

        run_in_thread(read, "flextool-assistant-ledger")

    def _show_footer(self, label: str, data: dict) -> None:
        from flextool.assistant.usage import number

        month = datetime.now(timezone.utc).strftime("%Y-%m")
        tokens_in = tokens_out = rejected = 0
        found = False
        for provider in (data.get("months") or {}).get(month, {}).values():
            if provider.get("label") != label:
                continue
            found = True
            for counts in (provider.get("models") or {}).values():
                tokens_in += sum(int(counts.get(k) or 0) for k in ("input_uncached", "cache_read", "cache_write"))
                tokens_out += int(counts.get("output") or 0)
                rejected += int(counts.get("rejected_attempts") or 0)
        if not found:
            text = strings.FOOTER_NONE.format(provider=label)
        else:
            text = strings.FOOTER.format(provider=label, input=number(tokens_in), output=number(tokens_out))
            if rejected:
                text += strings.FOOTER_REJECTED.format(n=number(rejected))
        self.footer_label.configure(text=text)

    # --- workers and snapshots ---------------------------------------------------------------------------------
    def make_worker(self) -> AssistantWorker:
        from flextool.gui.assistant.worker import AssistantWorker

        return AssistantWorker(self.project_path, self.project_name, gate=self.gate, tool_lock=self.tool_lock,
                               config_root=self.config_root, backend_factory=self._backend_factory,
                               probe=self._probe)

    def take_snapshot(self) -> ContextSnapshot:
        from flextool.assistant.context import gui_snapshot

        if self._snapshot is not None:
            try:
                snapshot = self._snapshot()
                if snapshot is not None:
                    return snapshot
            except Exception:  # noqa: BLE001 — the question still goes, without the window's selections
                logger.exception("The FlexTool window's selections could not be read")
        return gui_snapshot(self.project_name, None)

    def call_soon(self, fn: Callable[..., None], *args: Any) -> None:
        """Run ``fn(*args)`` on the Tk thread at the next poll (any thread may call this)."""
        self._calls.put((fn, args))

    def _store_handle(self) -> Store:
        if self._store is None:
            from flextool.assistant.store import open_store

            self._store = open_store(self.config_root)
        return self._store

    # --- the poll -----------------------------------------------------------------------------------------
    def _poll(self) -> None:
        self._poll_id = None
        if self.closing:
            return
        while True:
            try:
                fn, args = self._calls.get_nowait()
            except queue.Empty:
                break
            try:
                fn(*args)
            except Exception:  # noqa: BLE001 — one failed update must not stop the window
                logger.exception("A chat window update failed")
            if self.closing:
                return
        deadline = time.perf_counter() + POLL_BUDGET_S
        tabs = [t for t in self.all_tabs() if t.worker is not None and not t.closed]
        if tabs:
            start = self._rr % len(tabs)
            self._rr += 1
            for tab in tabs[start:] + tabs[:start]:
                if time.perf_counter() >= deadline:
                    break
                tab.drain(deadline, time.perf_counter)
                if self.closing:
                    return
        self._poll_id = self.after(POLL_MS, self._poll)

    # --- chats --------------------------------------------------------------------------------------------
    def promote(self, tab: ChatTab) -> None:
        """"New chat" got its first question: it is a chat now, and a new "New chat" comes last."""
        if not tab.is_new:
            return
        tab.is_new = False
        tab.close_button.state(["!disabled"])
        tab.transcript.clear_example()
        self.tabs.append(tab)
        self._touch(tab)
        new = ChatTab(self, new=True)
        self.new_tab = new
        self.notebook.add(new, text=strings.NEW_CHAT)
        self.notebook.tab(tab, text=tab.tab_text())
        self.schedule_fit()

    def _add_chat_tab(self) -> ChatTab:
        tab = ChatTab(self)
        tab.close_button.state(["!disabled"])
        self.notebook.insert(self.new_tab, tab, text=tab.tab_text())
        self.tabs.append(tab)
        return tab

    def new_chat(self, seed: str | None = None) -> None:
        """Go to "New chat"; with *seed* (a summary of a chat that got too long) its first question starts with
        the summary."""
        tab = self.new_tab
        if seed:
            tab.seed = seed
            tab.transcript.clear()
            tab.transcript.notice(seed)
        self.go_to_new_chat()

    def tab_for(self, conversation_id: str) -> ChatTab | None:
        return next((t for t in self.tabs if t.conversation_id == conversation_id and not t.closed), None)

    def reopen(self, conversation_id: str, *, select: bool = True) -> None:
        """Open a saved chat in a new tab (or select its tab).  A closed chat whose worker still finishes is
        waited for up to 5 s; then nothing opens and the window says why."""
        from flextool.gui.assistant.worker import finishing_worker

        open_tab = self.tab_for(conversation_id)
        if open_tab is not None:
            if select:
                self.select(open_tab)
            return
        if finishing_worker(conversation_id) is None:
            self._open_saved(conversation_id, select=select)
            return
        self.note(strings.WAITING_CLOSED_CHAT)
        deadline = time.monotonic() + REOPEN_WAIT_S

        def check() -> None:
            if self.closing:
                return
            if finishing_worker(conversation_id) is None:
                self.note("")
                self._open_saved(conversation_id, select=select)
            elif time.monotonic() >= deadline:
                self.note(strings.STILL_FINISHING)
                self._restore_resolved(conversation_id, refused=False)
            else:
                self.after(REOPEN_STEP_MS, check)

        self.after(REOPEN_STEP_MS, check)

    def _open_saved(self, conversation_id: str, *, select: bool) -> None:
        tab = self._add_chat_tab()
        tab.open_conversation(conversation_id)
        if select:
            self.select(tab)
        else:
            self.schedule_fit()

    def chat_loaded(self, tab: ChatTab) -> None:
        self._restore_resolved(tab.conversation_id, refused=False)
        self.tab_changed(tab)
        self.schedule_open_chats()

    def open_refused(self, tab: ChatTab, event: OpenRefused) -> None:
        """A saved chat could not be opened in *tab*: the tab goes, the window says why."""
        restoring = event.conversation_id in self._restoring
        self._remove_tab(tab)
        if restoring and event.reason == "other_process":
            self._restore_resolved(event.conversation_id, refused=True)
        else:
            self._restore_resolved(event.conversation_id, refused=False)
            self.note(event.message)

    def chat_finished(self, tab: ChatTab) -> None:
        self.refresh_saved()
        self.refresh_footer()
        self.schedule_open_chats()
        self._refresh_chats_button()

    def write_title(self, conversation_id: str, name: str) -> None:
        """The chat's name, written at once on a daemon thread (never queued behind its answer)."""
        store = self._store_handle()
        project = self.project_path

        def write() -> None:
            try:
                store.set_title(project, conversation_id, name)
            except Exception:  # noqa: BLE001 — the name stays in the tab
                logger.exception("The chat's name could not be saved")
            self.call_soon(self.refresh_saved)

        run_in_thread(write, "flextool-assistant-title")

    def rename(self, tab: ChatTab | None = None) -> None:
        tab = tab or self.current_tab()
        if tab.is_new or tab.closed:
            return
        RenameChatDialog(self, tab.name or "", lambda name: name is not None and tab.set_name(name))

    # --- closing chats ------------------------------------------------------------------------------------
    @staticmethod
    def _unsaved(tab: ChatTab) -> bool:
        return not tab.saved and tab.has_messages and not tab.is_new

    def _confirm_close(self, tabs: list[ChatTab], then: Callable[[], None]) -> None:
        unsaved = [t for t in tabs if self._unsaved(t)]
        if not unsaved:
            then()
            return
        text = strings.CONFIRM_CLOSE_UNSAVED if len(unsaved) == 1 else \
            strings.CONFIRM_CLOSE_UNSAVED_MANY.format(n=len(unsaved))
        ConfirmDialog(self, strings.CLOSE_CHAT, text, strings.CLOSE, lambda ok: ok and then())

    def close_chat(self, tab: ChatTab, *, ask: bool = True) -> None:
        """Close the chat of *tab*: the answer stops, what was received is saved, its worker finishes alone."""
        if tab.is_new or tab.closed:
            return
        if ask:
            self._confirm_close([tab], lambda: self.close_chat(tab, ask=False))
            return
        self._remove_tab(tab)
        self.schedule_open_chats()
        self.refresh_saved()

    def close_others(self, keep: ChatTab) -> None:
        others = [t for t in self.tabs if t is not keep and not t.closed]

        def close_all() -> None:
            for tab in others:
                self.close_chat(tab, ask=False)
        self._confirm_close(others, close_all)

    def _remove_tab(self, tab: ChatTab) -> None:
        was_current = self.current_tab() is tab
        self.consents.remove(tab)
        tab.shutdown()
        if tab in self.tabs:
            self.tabs.remove(tab)
        if tab in self._recent:
            self._recent.remove(tab)
        try:
            self.notebook.forget(tab)
        except tk.TclError:
            pass
        try:
            tab.destroy()
        except tk.TclError:
            pass
        if was_current and not self.closing:
            following = next((t for t in self._recent if t in self.tabs), None)
            self.select(following or self.new_tab)
            self.focus_input()
        self.schedule_fit()

    # --- saved chats (the Chats menu) -------------------------------------------------------------------------
    def refresh_saved(self) -> None:
        """Read the saved chats on a daemon thread; the Chats menu shows them next time it opens."""
        if not self.saving:
            self._saved = []
            return
        store = self._store_handle()
        project = self.project_path

        def read() -> None:
            try:
                saved = store.list_conversations(project)
            except Exception:  # noqa: BLE001 — the menu keeps the last list
                logger.exception("The saved chats could not be listed")
                return
            self.call_soon(self._set_saved, saved)

        run_in_thread(read, "flextool-assistant-list")

    def _set_saved(self, saved: list[ConversationInfo]) -> None:
        self._saved = list(saved)
        known = {info.id: info for info in saved}
        for tab in self.tabs:
            info = known.get(tab.conversation_id or "")
            if info is not None:
                tab.created = info.created

    def saved_chats(self) -> list[ConversationInfo]:
        return list(self._saved)

    def _tab_entry(self, tab: ChatTab) -> str:
        return menu_entry(tab.number, tab.name, tab.created, tab.first_question)

    def post_chats_menu(self) -> None:
        button = self.chats_button
        try:
            x, y = button.winfo_rootx(), button.winfo_rooty() + button.winfo_height()
            self.chats_menu.tk_popup(x, y)
        finally:
            self.chats_menu.grab_release()

    def _fill_chats_menu(self) -> None:
        menu = self.chats_menu
        menu.delete(0, "end")
        menu.add_command(label=strings.OPEN_CHATS, state="disabled")
        current = self.current_tab()
        for tab in self.tabs:
            mark = "◉ " if tab is current else "○ "
            menu.add_command(label=mark + tab.prefix() + self._tab_entry(tab), command=lambda t=tab: self.select(t))
        menu.add_separator()
        accelerator_new = strings.NEW_CHAT_ACCELERATOR_MAC if self.aqua else strings.NEW_CHAT_ACCELERATOR
        accelerator_close = strings.CLOSE_CHAT_ACCELERATOR_MAC if self.aqua else strings.CLOSE_CHAT_ACCELERATOR
        menu.add_command(label=strings.NEW_CHAT, accelerator=accelerator_new, command=self.go_to_new_chat)
        reopen = self._menu_factory(menu)
        delete = self._menu_factory(menu)
        if not self.saving:
            menu.add_command(label=f"{strings.REOPEN} — {strings.NOT_SAVED}", state="disabled")
            menu.add_command(label=f"{strings.DELETE_SAVED} — {strings.NOT_SAVED}", state="disabled")
        else:
            open_ids = {t.conversation_id for t in self.tabs if t.conversation_id}
            closed = [info for info in self._saved if info.id not in open_ids][:MENU_CHATS]
            for info in closed:
                reopen.add_command(label=self._info_entry(info), command=lambda i=info.id: self.reopen(i))
            if not closed:
                reopen.add_command(label=strings.NO_SAVED_CHATS, state="disabled")
            for info in self._saved[:MENU_CHATS]:
                label = self._info_entry(info) + (f" {strings.OPEN_MARK}" if info.id in open_ids else "")
                delete.add_command(label=label, command=lambda i=info.id: self.delete_saved(i))
            if self._saved:
                delete.add_separator()
            delete.add_command(label=strings.DELETE_ALL_CLOSED, command=self.delete_all_closed,
                               state="normal" if closed else "disabled")
            menu.add_cascade(label=strings.REOPEN, menu=reopen)
            menu.add_cascade(label=strings.DELETE_SAVED, menu=delete)
        menu.add_separator()
        chat = not current.is_new
        menu.add_command(label=strings.RENAME_CHAT, command=lambda: self.rename(current),
                         state="normal" if chat else "disabled")
        menu.add_command(label=strings.CLOSE_CHAT, accelerator=accelerator_close,
                         command=lambda: self.close_chat(current), state="normal" if chat else "disabled")

    @staticmethod
    def _info_entry(info: ConversationInfo) -> str:
        return menu_entry(info.number, info.title if info.named else None, info.created, info.first_question)

    def delete_saved(self, conversation_id: str) -> None:
        tab = self.tab_for(conversation_id)
        text = strings.CONFIRM_DELETE_OPEN if tab is not None else strings.CONFIRM_DELETE
        ConfirmDialog(self, strings.DELETE_TITLE, text, strings.DELETE_BUTTON,
                      lambda ok: ok and self._delete(conversation_id))

    def _delete(self, conversation_id: str) -> None:
        tab = self.tab_for(conversation_id)
        if tab is not None:
            self.close_chat(tab, ask=False)
        store = self._store_handle()
        project = self.project_path

        def run() -> None:
            try:
                outcome = store.delete_conversation(project, conversation_id)
            except Exception:  # noqa: BLE001 — said in the window
                logger.exception("The chat could not be deleted")
                outcome = "failed"
            self.call_soon(self._deleted, outcome)

        run_in_thread(run, "flextool-assistant-delete")

    def _deleted(self, outcome: str) -> None:
        if outcome == "locked_elsewhere":
            self.note(strings.NOT_DELETED_LOCKED)
        self.refresh_saved()

    def delete_all_closed(self) -> None:
        open_ids = {t.conversation_id for t in self.tabs if t.conversation_id}
        targets = [info.id for info in self._saved if info.id not in open_ids]
        if not targets:
            return
        store = self._store_handle()
        project = self.project_path

        def run() -> None:
            locked = 0
            for conversation_id in targets:
                try:
                    if store.delete_conversation(project, conversation_id) == "locked_elsewhere":
                        locked += 1
                except Exception:  # noqa: BLE001 — the rest is still deleted
                    logger.exception("A chat could not be deleted")
            self.call_soon(self._deleted, "locked_elsewhere" if locked else "deleted")

        ConfirmDialog(self, strings.DELETE_TITLE, strings.CONFIRM_DELETE_ALL, strings.DELETE_BUTTON,
                      lambda ok: ok and run_in_thread(run, "flextool-assistant-delete"))

    # --- the tab menu -------------------------------------------------------------------------------------
    def _tab_at(self, x: int, y: int) -> ChatTab | None:
        try:
            index = self.notebook.index(f"@{x},{y}")
            return self.nametowidget(self.notebook.tabs()[index])
        except (tk.TclError, IndexError, KeyError):
            return None

    def _on_tab_menu_click(self, event: tk.Event) -> str | None:  # type: ignore[type-arg]
        tab = self._tab_at(event.x, event.y)
        if tab is None:
            return None
        self._post_tab_menu(tab, (event.x_root, event.y_root))
        return "break"

    def _post_tab_menu(self, tab: ChatTab, where: tuple[int, int] | None) -> None:
        if tab.is_new:
            return
        menu = self.tab_menu
        menu.delete(0, "end")
        menu.add_command(label=strings.RENAME_CHAT, command=lambda: self.rename(tab))
        menu.add_command(label=strings.CLOSE_CHAT, command=lambda: self.close_chat(tab))
        menu.add_command(label=strings.CLOSE_OTHER_CHATS, command=lambda: self.close_others(tab),
                         state="normal" if len(self.tabs) > 1 else "disabled")
        if where is None:
            nb = self.notebook
            where = (nb.winfo_rootx() + 20, nb.winfo_rooty() + 20)
        try:
            menu.tk_popup(*where)
        finally:
            menu.grab_release()

    def _on_middle_click(self, event: tk.Event) -> str | None:  # type: ignore[type-arg]
        tab = self._tab_at(event.x, event.y)
        if tab is None or tab.is_new:
            return None
        self.close_chat(tab)
        return "break"

    def _on_double_click(self, event: tk.Event) -> str | None:  # type: ignore[type-arg]
        tab = self._tab_at(event.x, event.y)
        if tab is None or tab.is_new:
            return None
        self.rename(tab)
        return "break"

    # --- settings, welcome --------------------------------------------------------------------------------
    def open_settings(self) -> None:
        if self._open_settings is not None:
            self._open_settings(self)

    def show_welcome(self) -> WelcomeDialog:
        """"How this works": the welcome again, recording nothing; [Open chat] comes back to this window's input,
        [Open settings] opens the settings as the ⚙ button does."""
        return WelcomeDialog(self, self._welcome_closed, record=False, config_root=self.config_root)

    def _welcome_closed(self, action: str) -> None:
        if self.closing:
            return
        if action == "settings":
            self.open_settings()
        elif action == "chat":
            try:
                self.deiconify()
                self.lift()
                self.focus_input()
            except tk.TclError:
                pass

    # --- open chats (P2-O14) -----------------------------------------------------------------------------------
    def open_chat_ids(self) -> tuple[list[str], str | None]:
        """The saved chats open in tabs, in tab order, and the selected one."""
        ids = [t.conversation_id for t in self.tabs
               if t.conversation_id and t.answered and t.saved and not t.closed]
        current = self.current_tab()
        selected = current.conversation_id if current.conversation_id in ids else None
        return ids, selected

    def schedule_open_chats(self) -> None:
        if self.closing or not self.saving:
            return
        if self._open_chats_id is not None:
            self.after_cancel(self._open_chats_id)
        self._open_chats_id = self.after(OPEN_CHATS_DEBOUNCE_MS, self._write_open_chats)

    def _write_open_chats(self) -> threading.Thread | None:
        self._open_chats_id = None
        if not self.saving or self._restoring:
            return None
        ids, selected = self.open_chat_ids()
        store = self._store_handle()
        project = self.project_path

        def write() -> None:
            try:
                store.set_open_chats(project, ids, selected)
            except Exception:  # noqa: BLE001 — the chats are just not restored next time
                logger.exception("The open chats could not be kept")

        return run_in_thread(write, "flextool-assistant-open-chats")

    def _restore_open_chats(self) -> None:
        if not self.saving:
            return
        store = self._store_handle()
        project = self.project_path

        def read() -> None:
            try:
                data = store.open_chats(project)
            except Exception:  # noqa: BLE001 — nothing is restored
                logger.exception("The open chats could not be read")
                return
            self.call_soon(self._restore, data)

        run_in_thread(read, "flextool-assistant-restore")

    def _restore(self, data: dict) -> None:
        ids = [i for i in data.get("ids") or [] if self.tab_for(i) is None]
        selected = data.get("selected")
        self._restoring = set(ids)
        self._restore_selected = selected
        for conversation_id in ids:
            self.reopen(conversation_id, select=False)
        if not ids:
            self._restoring = set()

    def _restore_resolved(self, conversation_id: str | None, *, refused: bool) -> None:
        if conversation_id is None or conversation_id not in self._restoring:
            return
        self._restoring.discard(conversation_id)
        if refused:
            self._not_restored += 1
        if self._restoring:
            return
        selected = self._restore_selected
        tab = self.tab_for(selected) if selected else None
        self.select(tab or self.new_tab)
        self.focus_input()
        if self._not_restored:
            text = strings.NOT_RESTORED_ONE if self._not_restored == 1 else \
                strings.NOT_RESTORED.format(n=self._not_restored)
            self.new_tab.transcript.notice(text)
            self._not_restored = 0
        self.schedule_open_chats()

    # --- size and place (§4.2) ---------------------------------------------------------------------------------
    def _place(self) -> None:
        from flextool.gui.assistant.geometry import SAMPLE, Decoration, average_char_px, chat_window_geometry
        from flextool.gui.ui_metrics import monitor_signature, monitor_work_area, resolve_saved_geometry

        cw, lh = self._cw, self._lh
        min_size = (cw * 50, lh * 20)
        self.minsize(*min_size)
        screen_w, screen_h = self.winfo_screenwidth(), self.winfo_screenheight()
        saved = None
        settings = self._global_settings
        geometries = getattr(settings, "assistant_window_geometry", None) if settings is not None else None
        if isinstance(geometries, dict) and geometries:
            saved = resolve_saved_geometry(geometries, monitor_signature(self),
                                           int(getattr(settings, "assistant_window_layout_cw", 0) or 0), cw,
                                           screen_w, screen_h, min_w=min_size[0], min_h=min_size[1])
        if saved:
            self.geometry(saved)
            self.deiconify()
            return
        finder = self._work_area or monitor_work_area
        try:
            area = finder(self.master) or (0, 0, screen_w, screen_h)
        except tk.TclError:
            area = (0, 0, screen_w, screen_h)
        from flextool.gui.assistant.fonts import sync_fonts

        fonts = sync_fonts(self)                # the transcript's text and tables, in the chat's fonts
        avg = average_char_px(fonts["base"].measure(SAMPLE))
        fixed = fonts["code"].measure("0")
        chrome = 4 * cw + 40                    # scroll bar, paddings and borders around the transcript
        exec_rect = self._exec_rect() if self._exec_rect is not None else None
        self._placement = (area, avg, fixed, exec_rect, chrome, min_size)
        self._placed = chat_window_geometry(area, Decoration(), avg, fixed, exec_rect, chrome_px=chrome,
                                            min_size=min_size)
        self.geometry(self._placed)
        self.deiconify()
        self.after(200, self._settle_geometry)

    def measure_decoration(self):  # noqa: ANN201 — a geometry.Decoration
        """The title bar and border the window manager put around the client area (0 where there is none)."""
        from flextool.gui.assistant.geometry import Decoration, parse_geometry

        try:
            self.update_idletasks()
            frame = parse_geometry(self.wm_geometry())
            if frame is None:
                return Decoration()
            return Decoration(top=max(0, self.winfo_rooty() - frame[1]), border=max(0, self.winfo_rootx() - frame[0]))
        except tk.TclError:
            return Decoration()

    def _settle_geometry(self) -> None:
        """Once the window manager has framed the window, fit the frame (not only the client) into the work
        area — unless the user has moved or resized it meanwhile."""
        from flextool.gui.assistant.geometry import chat_window_geometry

        if self.closing or not getattr(self, "_placement", None):
            return
        try:
            if self.wm_geometry() != self._placed:
                return
        except tk.TclError:
            return
        decoration = self.measure_decoration()
        if decoration.top or decoration.border:
            area, avg, fixed, exec_rect, chrome, min_size = self._placement
            self._placed = chat_window_geometry(area, decoration, avg, fixed, exec_rect, chrome_px=chrome,
                                                min_size=min_size)
            self.geometry(self._placed)

    def _remember_geometry(self) -> None:
        from flextool.gui.ui_metrics import monitor_signature

        settings = self._global_settings
        if settings is None or not hasattr(settings, "assistant_window_geometry"):
            return
        try:
            geometry = self.wm_geometry()
            signature = monitor_signature(self)
        except tk.TclError:
            return
        geometries = getattr(settings, "assistant_window_geometry", None)
        if not isinstance(geometries, dict):
            geometries = {}
        geometries[signature] = geometry
        settings.assistant_window_geometry = geometries
        if hasattr(settings, "assistant_window_layout_cw"):
            settings.assistant_window_layout_cw = self._cw

    # --- closing the window -------------------------------------------------------------------------------
    def close(self, *, ask: bool = True) -> None:
        """Close the window: with *ask*, unsaved chats are confirmed once (not when FlexTool closes or the
        project changes)."""
        if self.closing:
            return
        if ask:
            self._confirm_close(list(self.tabs), lambda: self.close(ask=False))
            return
        self.closing = True
        if self._open_chats_id is not None:
            try:
                self.after_cancel(self._open_chats_id)
            except tk.TclError:
                pass
            self._open_chats_id = None
        if self.saving and not self._restoring:
            ids, selected = self.open_chat_ids()
            store = self._store_handle()
            project = self.project_path

            def write() -> None:
                try:
                    store.set_open_chats(project, ids, selected)
                except Exception:  # noqa: BLE001 — the chats are just not restored next time
                    logger.exception("The open chats could not be kept")
            # Written before the workers are told to stop, so the list is the one the user saw.
            self._last_open_chats_write = run_in_thread(write, "flextool-assistant-open-chats")
        self.consents.close_all()
        for tab in list(self.tabs):
            tab.shutdown()
        self._remember_geometry()
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except tk.TclError:
                pass
            self._poll_id = None
        self._tips.hide()
        try:
            self.destroy()
        except tk.TclError:
            pass
        if self._on_closed is not None:
            self._on_closed(self)

    def workers(self) -> list[AssistantWorker]:
        return [t.worker for t in self.all_tabs() if t.worker is not None]

    def _on_destroy(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if event.widget is not self or self.closing:
            return
        # Destroyed with its master (FlexTool closing): the workers stop and save on their own.
        self.closing = True
        for tab in list(self.tabs):
            tab.shutdown()


class _TabTips:
    """The tooltip of the tab under the pointer: the chat's full status (or its title)."""

    def __init__(self, window: ChatWindow) -> None:
        self.window = window
        self._after: str | None = None
        self._tip: tk.Toplevel | None = None
        self._tab: ChatTab | None = None
        window.notebook.bind("<Motion>", self._on_motion, add="+")
        window.notebook.bind("<Leave>", lambda _e: self.hide(), add="+")

    def _on_motion(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        tab = self.window._tab_at(event.x, event.y)
        if tab is self._tab:
            return
        self.hide()
        self._tab = tab
        if tab is not None:
            self._after = self.window.after(TOOLTIP_DELAY_MS, lambda: self._show(event.x_root, event.y_root))

    def text_for(self, tab: ChatTab) -> str:
        if tab.is_new:
            return strings.NEW_CHAT
        parts = [menu_entry(tab.number, tab.name, tab.created, tab.first_question)]
        if tab.status_text:
            parts.append(tab.status_text)
        return "\n".join(parts)

    def _show(self, x: int, y: int) -> None:
        self._after = None
        tab = self._tab
        if tab is None or tab.closed or self.window.closing:
            return
        tip = tk.Toplevel(self.window)
        tip.wm_overrideredirect(True)
        ttk.Label(tip, text=self.text_for(tab), padding=(6, 3), relief="solid", borderwidth=1,
                  wraplength=self.window._cw * 50).pack()
        tip.wm_geometry(f"+{x + 12}+{y + 18}")
        self._tip = tip

    def hide(self) -> None:
        if self._after is not None:
            try:
                self.window.after_cancel(self._after)
            except tk.TclError:
                pass
            self._after = None
        if self._tip is not None:
            try:
                self._tip.destroy()
            except tk.TclError:
                pass
            self._tip = None
        self._tab = None
