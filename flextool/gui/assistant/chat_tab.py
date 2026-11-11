"""One chat in the chat window (P2 §4.1, §4.4, §5.3): its transcript, its input box, its status rows and buttons,
and its own worker thread.

The tab never waits: Send takes the question out of the box (P2-O9: the box is cleared only then), shows it in
the transcript at once (dim until the worker starts the turn) and gives it to the worker with the window's
snapshot of this moment; the window's poll hands the worker's events back through :meth:`ChatTab.handle`.  Send is
unavailable until the answer is done, while the box stays editable.  Stop and Escape stop the answer at once in
the window ("Stopped."); the worker repairs the history before the next question.

Keys in the input box (bound on the box's own tag, so they run before the Text class): Enter sends (also with
Shift or Alt), Ctrl+Enter (Command+Enter on macOS) inserts a new line, Tab and Shift+Tab move the focus, Ctrl+A
selects all, Escape stops (or, with no answer coming, cancels an edit), and the window's tab commands (Ctrl+Tab,
Ctrl+Shift+Tab, Ctrl+T, Ctrl+W) work here too.

Asking a question again (item 220): ↻ on a question (or R, or its menu) asks it again as it was; ✎ (or E) puts it
in the input box, the box's text kept aside until Escape gives it back, and Enter sends the changed question.
Either way, when later questions would go too, a dialog asks first ("This also removes the N later questions and
answers.", a question still being answered counted); during an answer the answer is stopped after that, and the
question goes when the worker is idle.  One such action waits at a time: a second one replaces it.  The worker's
``Rewound`` redraws the chat from what it keeps; an edit lasts until its question is sent (``TurnStarted``), so a
question that is not sent can be sent again from the box.
"""
from __future__ import annotations

import enum
import itertools
import logging
import queue
import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import ttk
from typing import TYPE_CHECKING, Any

from flextool.gui.assistant import strings
from flextool.gui.assistant.transcript import Transcript

if TYPE_CHECKING:
    from flextool.gui.assistant.chat_window import ChatWindow
    from flextool.gui.assistant.worker import AssistantWorker, Prepared, TurnJob

logger = logging.getLogger(__name__)

_job_ids = itertools.count(1)

#: What the tab is doing, for its state prefix (P2 §4.3).
WORKING, WAITING, NOTICE, ERROR, UNSEEN, IDLE = "working", "waiting", "notice", "error", "unseen", "idle"
PREFIXES = {WORKING: strings.PREFIX_WORKING, WAITING: strings.PREFIX_WAITING, NOTICE: strings.PREFIX_NOTICE,
            ERROR: strings.PREFIX_ERROR, UNSEEN: strings.PREFIX_UNSEEN, IDLE: ""}


class Phase(enum.Enum):
    """Where the tab is in its life (G17-D2); the state prefix above is only what it shows.

    ``NEW`` the "New chat" tab (no worker) → Send → ``ANSWERING``.  A reopened chat is ``LOADING`` (Send keeps the
    text in the box, the status says so) → ``ConversationLoaded`` → ``IDLE`` (Send enabled again), or
    ``OpenRefused`` → ``CLOSED`` (the tab goes).  ``IDLE`` → Send → ``ANSWERING`` → the worker's ``Idle`` →
    ``IDLE``; a Stop keeps ``ANSWERING`` until then (the question's line waits for the worker's word).  Close in
    any phase → ``CLOSING``: the worker is told which job to stop, also one it has not taken yet, and no handler
    runs for the tab again."""

    NEW = "new"
    LOADING = "loading"
    IDLE = "idle"
    ANSWERING = "answering"
    CLOSING = "closing"
    CLOSED = "closed"


#: P1 error codes whose fix is in the settings (P2 §5.6).
SETTINGS_ERRORS = frozenset({"auth", "no_key", "model", "permission", "quota", "credit", "routing"})


class ChatTab(ttk.Frame):
    """A chat of *window*.  ``is_new`` is the "New chat" tab: no worker, nothing sent yet."""

    def __init__(self, window: ChatWindow, *, new: bool = False, seed: str | None = None) -> None:
        super().__init__(window.notebook, padding=(4, 4))
        self.window = window
        self.is_new = new
        self.number: int | None = None
        self.name: str | None = None
        self.conversation_id: str | None = None
        self.saved = window.saving              # False: chats are not saved (expiry 0)
        self.answered = False                   # a conversation file exists (an answer was saved, or reopened)
        self.has_messages = False
        self.worker: AssistantWorker | None = None
        self.active_job: TurnJob | None = None
        self.phase = Phase.NEW if new else Phase.IDLE
        self._stopped_job: TurnJob | None = None      # stopped in the window, the worker not done with it yet
        self.state = IDLE
        self.unseen = False
        self.status_text = ""
        self.prepared: Prepared | None = None
        self.seed = seed
        self.first_question = ""
        self.created = ""                       # UTC creation time of the saved conversation, when known
        self._step = "prepare"                  # of the active question: "prepare" until TurnStarted, "answer"
        self._job_lines: dict[int, tuple[int, str]] = {}
        self._tool_lines: list[int] = []
        self._pending_name: str | None = None
        self._totals: dict[str, Any] = {}
        self._question_usage: list | None = None      # the Usage of the last question answered in this window
        self._carry_seed: str | None = None     # the summary the chat's first question started with (seed)
        self._editing: int | None = None        # the number of the question being changed in the input box
        self._draft: str | None = None          # what the box held before the edit (given back by Escape)
        self._edit_job: int | None = None       # the job that sends the changed question
        self._pending: Callable[[], None] | None = None   # an action that waits for the worker's Idle
        self._confirm: Any = None               # the open "This also removes …" dialog
        self._build()
        if new:
            self.transcript.show_example(strings.EXAMPLE)
        elif seed:
            self.transcript.notice(seed)
        self.refresh_placeholder()

    # --- layout ---------------------------------------------------------------------------------------------
    def _build(self) -> None:
        from flextool.gui.ui_metrics import get_metrics

        metrics = get_metrics(self)
        cw, lh = metrics.cw, metrics.lh
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.transcript = Transcript(self, on_escape=self.escape, bind_commands=self.window.bind_tab_commands,
                                     on_question=self.question_action)
        self.transcript.grid(row=0, column=0, sticky="nsew")
        from flextool.gui.assistant.fonts import CHAT_FONT, sync_fonts

        sync_fonts(self)                    # the input box is in the chat's base font, as the transcript
        self.input = tk.Text(self, height=3, wrap="word", undo=True, borderwidth=1, highlightthickness=1,
                             padx=6, pady=4, font=CHAT_FONT)
        self.input.grid(row=1, column=0, sticky="ew", pady=(lh // 2, 0))
        self._placeholder = ttk.Label(self.input, text=strings.PLACEHOLDER, cursor="xterm", takefocus=0)
        self._style_placeholder()
        self._placeholder.bind("<Button-1>", lambda _e: self.input.focus_set())
        self.status = ttk.Label(self, text="", anchor="w", justify="left")
        self.status.grid(row=2, column=0, sticky="ew", pady=(lh // 3, 0))
        bottom = ttk.Frame(self)
        bottom.grid(row=3, column=0, sticky="ew", pady=(lh // 3, 0))
        bottom.columnconfigure(0, weight=1)
        self.close_button = ttk.Button(bottom, text=strings.CLOSE_CHAT, command=self.request_close)
        self.close_button.grid(row=0, column=1, padx=(cw, 0))
        self.stop_button = ttk.Button(bottom, text=strings.STOP, command=self.stop, state="disabled")
        self.stop_button.grid(row=0, column=2, padx=(cw // 2, 0))
        self.send_button = ttk.Button(bottom, text=strings.SEND, command=self.send, style="Accent.TButton")
        self.send_button.grid(row=0, column=3, padx=(cw // 2, 0))
        # The token line: under the buttons, outside the transcript, so it never scrolls away.
        self.totals = ttk.Label(self, text="", anchor="w", justify="left")
        self.totals.grid(row=4, column=0, sticky="ew", pady=(lh // 3, 0))
        if self.is_new:
            self.close_button.state(["disabled"])
        self.bind("<Configure>", self._wrap_status, add="+")
        self._bind_input()

    def _style_placeholder(self) -> None:
        from flextool.gui.ui_metrics import blend_rgb, rgb_hex

        try:
            bg = self.input.winfo_rgb(self.input.cget("background"))
            fg = self.input.winfo_rgb(self.input.cget("foreground"))
            self._placeholder.configure(foreground=blend_rgb(fg, rgb_hex(bg), 0.5), background=rgb_hex(bg),
                                        font=self.input.cget("font"))
        except tk.TclError:
            pass

    def _wrap_status(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if event.widget is self:
            self.status.configure(wraplength=max(100, event.width - 16))
            self.totals.configure(wraplength=max(100, event.width - 16))

    def _bind_input(self) -> None:
        box = self.input
        aqua = self.window.aqua
        box.bind("<Return>", self._on_enter)
        box.bind("<KP_Enter>", self._on_enter)
        for sequence in ("<Control-Return>", "<Control-KP_Enter>") + (
                ("<Command-Return>", "<Command-KP_Enter>") if aqua else ()):
            box.bind(sequence, self._newline)
        for sequence in ("<Control-a>", "<Control-A>") + (("<Command-a>",) if aqua else ()):
            box.bind(sequence, self._select_all)
        self.window.bind_tab_commands(box)
        box.bind("<Tab>", lambda _e: self._focus(next_=True))
        box.bind("<Shift-Tab>", lambda _e: self._focus(next_=False))
        try:
            box.bind("<ISO_Left_Tab>", lambda _e: self._focus(next_=False))
        except tk.TclError:
            pass
        box.bind("<Escape>", lambda _e: (self.escape(), "break")[1])
        box.bind("<<Modified>>", self._on_modified)
        box.bind("<KeyRelease>", lambda _e: self.refresh_placeholder(), add="+")
        box.bind("<FocusIn>", lambda _e: self.refresh_placeholder(), add="+")

    def _on_enter(self, _event: tk.Event) -> str:  # type: ignore[type-arg]
        self.send()
        return "break"

    def _newline(self, _event: tk.Event) -> str:  # type: ignore[type-arg]
        self.input.insert("insert", "\n")
        self.input.see("insert")
        return "break"

    def _select_all(self, _event: tk.Event) -> str:  # type: ignore[type-arg]
        self.input.tag_add("sel", "1.0", "end-1c")
        self.input.mark_set("insert", "end-1c")
        return "break"

    def _focus(self, *, next_: bool) -> str:
        target = self.input.tk_focusNext() if next_ else self.input.tk_focusPrev()
        if target is not None:
            target.focus_set()
        return "break"

    def _on_modified(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        if self.input.edit_modified():
            self.input.edit_modified(False)
        self.refresh_placeholder()

    def refresh_placeholder(self) -> None:
        """The placeholder shows in an empty box while the chat has no messages."""
        empty = self.input.get("1.0", "end-1c") == ""
        if empty and not self.has_messages:
            self._placeholder.place(in_=self.input, x=6, y=4)
        else:
            self._placeholder.place_forget()

    @property
    def placeholder_shown(self) -> bool:
        return bool(self._placeholder.winfo_manager())

    # --- what the window shows of the tab ----------------------------------------------------------------
    def label(self) -> str:
        """The chat's number or name ("" until the number is known)."""
        if self.is_new:
            return strings.NEW_CHAT
        if self.name:
            return self.name
        return str(self.number) if self.number is not None else ""

    def prefix(self) -> str:
        """The state prefix of the tab text ("… " working, "‖ " waiting, "? " notice, "! " error, "• " unseen)."""
        if self.state == IDLE and self.unseen:
            return strings.PREFIX_UNSEEN
        return PREFIXES[self.state]

    def tab_text(self) -> str:
        if self.is_new:
            return strings.NEW_CHAT
        title = self.label()
        if len(title) > strings.TAB_NAME_MAX:
            title = title[:strings.TAB_NAME_MAX] + "…"
        prefix = self.prefix()
        return (prefix + title) if title else prefix.strip() or "…"

    def needs_attention(self) -> bool:
        return self.unseen or self.state == NOTICE

    def _set_state(self, state: str, status: str | None = None) -> None:
        self.state = state
        if status is not None:
            self.status_text = status
            self.status.configure(text=status)
        self.window.tab_changed(self)

    def _set_status(self, text: str) -> None:
        self.status_text = text
        self.status.configure(text=text)
        self.window.tab_changed(self, text_only=True)

    def _show_totals(self) -> None:
        """The token line: the last question of this window (when there is one) and the whole chat."""
        from flextool.assistant.usage import format_token_line

        self.totals.configure(text=format_token_line(self._question_usage, self._totals, self.provider))

    @property
    def busy(self) -> bool:
        """Send is unavailable: the chat is opening or answering."""
        return self.phase in (Phase.LOADING, Phase.ANSWERING)

    @property
    def closed(self) -> bool:
        return self.phase in (Phase.CLOSING, Phase.CLOSED)

    @property
    def provider(self) -> str:
        if self.prepared is not None:
            return self.prepared.provider
        return self.window.default_provider()

    # --- the worker ---------------------------------------------------------------------------------------
    def ensure_worker(self) -> AssistantWorker:
        if self.worker is None:
            self.worker = self.window.make_worker()
            self.worker.start()
        return self.worker

    def open_conversation(self, conversation_id: str) -> None:
        """Load a saved chat into this (new) tab."""
        from flextool.gui.assistant.worker import OpenConversation

        self.conversation_id = conversation_id
        self.answered = True
        self.phase = Phase.LOADING
        self.send_button.state(["disabled"])
        self.ensure_worker().submit(OpenConversation(conversation_id))
        self._set_state(WORKING, strings.OPENING_CHAT)

    def consent_answer(self, job_id: int, accepted: bool) -> None:
        from flextool.gui.assistant.worker import ConsentAnswer

        if self.worker is not None:
            self.worker.submit(ConsentAnswer(job_id, accepted))
        if self.active_job is not None and self.active_job.job_id == job_id:
            self._set_state(WORKING, strings.STATUS_PREPARING if accepted else "")

    def notice_queued(self) -> None:
        self._set_state(WAITING, strings.STATUS_NOTICE_QUEUED)

    def notice_shown(self) -> None:
        self._set_state(NOTICE, strings.STATUS_NOTICE_OPEN)

    def settings_changed(self) -> None:
        """The worker prepares again at the next question and says what the header shows then (``RouteShown``):
        the tab never guesses which model a chat goes on with (G17-D1)."""
        from flextool.gui.assistant.worker import SettingsChanged

        if self.worker is not None:
            self.worker.submit(SettingsChanged())

    def shutdown(self) -> None:
        """Stop and let the worker finish on its own (never waits): the question it has not taken yet, too."""
        if self.phase is not Phase.CLOSED:
            self.phase = Phase.CLOSING
        self._drop_pending()
        if self.worker is not None:
            self.worker.shutdown(self.active_job or self._stopped_job)

    # --- user actions ---------------------------------------------------------------------------------------
    def send(self) -> None:
        if self.closed:
            return
        if self.phase is Phase.LOADING:
            self._set_status(strings.SEND_WHEN_OPEN)          # the text stays in the box (P2-O9, G17-D3)
            return
        if self._editing is not None:
            self._send_edit()
            return
        if self.phase is Phase.ANSWERING:
            self._set_status(strings.SEND_WHEN_DONE)
            return
        text = self.input.get("1.0", "end-1c").strip()
        if not text:
            return
        self.input.delete("1.0", "end")
        self._submit(text)

    def _submit(self, text: str, rewind: int | None = None) -> None:
        """Show *text* as a question and give it to the worker (with *rewind*: question *rewind* asked again)."""
        from flextool.gui.assistant.worker import TurnJob

        if self.is_new:
            self.window.promote(self)
        line = self.transcript.append_user(text, pending=True)
        if not self.first_question:
            self.first_question = " ".join(text.split())
        self.has_messages = True
        self.refresh_placeholder()
        if rewind is None:
            seed = self.seed
            if seed:
                self._carry_seed = seed
        else:
            seed = self._carry_seed             # the session puts it back only when the question started with it
        self.seed = None
        job = TurnJob(next(_job_ids), text, self.window.take_snapshot(), rewind=rewind, seed=seed)
        if rewind is not None and self._editing == rewind:
            self._edit_job = job.job_id
        self._job_lines[job.job_id] = (line, text)
        self.active_job = job
        self.phase = Phase.ANSWERING
        self._step = "prepare"
        self._tool_lines.clear()
        self.send_button.state(["disabled"])
        self.stop_button.state(["!disabled"])
        self._set_state(WORKING, strings.STATUS_PREPARING)
        self.ensure_worker().submit(job)

    def escape(self) -> None:
        """Escape: stops the answer; with no answer coming, cancels the edit of a question."""
        if self.phase is Phase.ANSWERING and self.active_job is not None:
            self.stop()
        elif self._editing is not None:
            self.cancel_edit()

    # --- asking a question again (item 220) -----------------------------------------------------------------
    def question_action(self, action: str, turn: int) -> None:
        """↻ (``"again"``) or ✎ (``"edit"``) on question *turn*, from its buttons, keys or menu."""
        from flextool.gui.assistant.transcript import EDIT

        if self.closed or self.phase in (Phase.NEW, Phase.LOADING):
            return
        text = self.transcript.turn_text(turn)
        if not text:
            return
        self._drop_pending()                    # one action waits at a time: this one replaces it
        if action == EDIT:
            self._begin_edit(turn, text)
        else:
            self._confirm_then(turn, strings.ASK_AGAIN, strings.ASK_AGAIN, lambda: self._submit(text, turn))

    def _later(self, turn: int) -> int:
        """The questions after *turn* that go with it: those the chat shows as sent, and the one being prepared."""
        later = self.transcript.later_turns(turn)
        job = self.active_job
        if job is not None and job.job_id in self._job_lines:
            later += 1
        return later

    def _confirm_then(self, turn: int, title: str, action: str, run: Callable[[], None]) -> None:
        """Ask first when later questions would go too; then stop the answer (if one is coming) and *run* when the
        worker is idle."""
        def go() -> None:
            if self.closed:
                return
            if self.phase is Phase.ANSWERING:
                self._pending = run
                self.stop()
            else:
                run()

        later = self._later(turn)
        if later == 0:
            go()
            return
        from flextool.gui.assistant.notices import ConfirmDialog

        message = strings.CONFIRM_LATER_ONE if later == 1 else strings.CONFIRM_LATER.format(n=later)

        def answered(accepted: bool) -> None:
            if self._confirm is not dialog:
                return                          # replaced by a later action, or the tab closed
            self._confirm = None
            if accepted:
                go()

        dialog = ConfirmDialog(self.window, title, message, action, answered)
        self._confirm = dialog

    def _drop_pending(self) -> None:
        """Forget the action that waits (its dialog closes as Cancel)."""
        self._pending = None
        dialog, self._confirm = self._confirm, None
        if dialog is not None:
            try:
                dialog.cancel()
            except tk.TclError:
                pass

    def _begin_edit(self, turn: int, text: str) -> None:
        if self._editing is None:
            self._draft = self.input.get("1.0", "end-1c")
        self._editing = turn
        self._edit_job = None
        self.input.delete("1.0", "end")
        self.input.insert("1.0", text)
        self.input.mark_set("insert", "end-1c")
        self.input.see("insert")
        self.input.focus_set()
        self.refresh_placeholder()
        self._set_status(strings.EDITING.format(n=turn))

    def _send_edit(self) -> None:
        turn = self._editing
        assert turn is not None
        text = self.input.get("1.0", "end-1c").strip()
        if not text:
            return

        def run() -> None:
            if self._editing != turn:
                return                          # cancelled meanwhile
            self.input.delete("1.0", "end")
            self._submit(text, turn)

        self._drop_pending()
        self._confirm_then(turn, strings.EDIT_SEND_TITLE, strings.EDIT_SEND, run)

    def cancel_edit(self) -> None:
        """Escape: the question stays as it was and the box gets back what it held."""
        if self._editing is None:
            return
        self._drop_pending()
        draft = self._draft or ""
        self._editing = self._draft = self._edit_job = None
        self.input.delete("1.0", "end")
        if draft:
            self.input.insert("1.0", draft)
        self.refresh_placeholder()
        if not self.busy:
            self._set_status("")

    def _edit_sent(self, job_id: int) -> None:
        """The changed question was sent: the edit is over and the box gets its earlier text back (when empty)."""
        if self._edit_job != job_id:
            return
        draft = self._draft or ""
        self._editing = self._draft = self._edit_job = None
        if draft and self.input.get("1.0", "end-1c") == "":
            self.input.insert("1.0", draft)
            self.refresh_placeholder()

    def _rewound(self, event: Any) -> None:
        """The worker asks a question again: the chat as it keeps it, then the question on its way."""
        from flextool.assistant.store import first_question

        messages = list(event.messages)
        entry = self._job_lines.get(event.job_id)
        self.transcript.render_history(messages, self.provider)
        if entry is not None:
            self._job_lines[event.job_id] = (self.transcript.append_user(entry[1], pending=True), entry[1])
        self._tool_lines.clear()
        self.first_question = first_question(messages) or (" ".join(entry[1].split()) if entry else "")
        self._question_usage = None
        self._show_totals()
        self.window.tab_changed(self)

    def _started(self, event: Any) -> None:
        entry = self._job_lines.pop(event.job_id, None)
        if entry is not None:
            self.transcript.mark_user(entry[0], "sent")
            self.transcript.set_turn(entry[0], event.turn)
        self._edit_sent(event.job_id)

    def stop(self) -> None:
        """Stop the answer at once in the window; whether the question was sent is the worker's word (its
        ``TurnStarted`` or ``TurnRefused`` of the stopped job, G17-D4)."""
        job = self.active_job
        if self.phase is not Phase.ANSWERING or job is None:
            return
        self.window.consents.remove(self, job.job_id)
        if self.worker is not None:
            self.worker.stop_turn(job)
        self.active_job = None
        self._stopped_job = job
        self.transcript.end_answer()
        self.transcript.notice(strings.STOPPED)
        self.stop_button.state(["disabled"])
        self._set_state(WORKING, "")

    def request_close(self) -> None:
        self.window.close_chat(self)

    def set_name(self, name: str) -> None:
        """The name the user gave ("" goes back to the number); written at once on a daemon thread."""
        self.name = name or None
        if self.conversation_id is not None:
            self.window.write_title(self.conversation_id, name)
        else:
            self._pending_name = name
        self.window.tab_changed(self)

    def _not_sent(self, job_id: int) -> None:
        entry = self._job_lines.pop(job_id, None)
        if entry is None:
            return
        line, text = entry
        self.transcript.mark_user(line, "not_sent")
        if self.input.get("1.0", "end-1c") == "":
            self.input.insert("1.0", text)
            self.refresh_placeholder()

    # --- the worker's events (Tk thread, from the window's poll) -------------------------------------------
    def handle(self, job_id: int | None, event: object) -> None:
        from flextool.assistant import llm_adapter as la
        from flextool.gui.assistant import worker as w

        if isinstance(event, w.Idle):
            self._idle()
            return
        if isinstance(event, w.Exited):
            return
        if isinstance(event, w.ChatReady):
            self._chat_ready(event)
            return
        if isinstance(event, w.ConversationLoaded):
            self._loaded(event)
            return
        if isinstance(event, w.OpenRefused):
            self.window.open_refused(self, event)
            return
        if isinstance(event, w.Deleted):
            return
        if isinstance(event, w.RouteShown):
            self._route_shown(event.prepared)
            return
        active = self.active_job is not None and job_id == self.active_job.job_id
        if isinstance(event, w.TurnFinished) and not active:
            self._question_usage = list(event.result.usage)
            self._totals = dict(event.totals)
            self._show_totals()
            self.answered = self.answered or bool(event.result.rounds)
            return
        if job_id is not None and not active:
            if self._stopped_job is not None and job_id == self._stopped_job.job_id:
                self._stopped_event(event)
            return
        if isinstance(event, w.ConsentNeeded):
            self.window.consents.request(self, event)
        elif isinstance(event, w.Rewound):
            self._rewound(event)
        elif isinstance(event, w.TurnStarted):
            self._started(event)
            self._step = "answer"
            self._set_state(WORKING, strings.STATUS_THINKING)
        elif isinstance(event, w.TurnRefused):
            self._not_sent(event.job_id)
            if event.reason not in (strings.NOT_SENT, strings.STOPPED):
                self.transcript.notice(event.reason)
            self.active_job = None
        elif isinstance(event, w.TurnFinished):
            self._finished(event)
        elif isinstance(event, w.SetupProblem):
            actions = [(strings.ACTION_OPEN_SETTINGS, self.window.open_settings)] if event.open_settings else []
            self.transcript.error(event.message, actions)
            self.state = ERROR
        elif isinstance(event, w.WorkerFault):
            self.transcript.error(event.message, [])
            self.state = ERROR
        elif isinstance(event, la.TextDelta):
            self.transcript.stream(event.text)
            if self.status_text:
                self._set_status("")
        elif isinstance(event, la.Thinking):
            self._set_status(strings.STATUS_THINKING)
        elif isinstance(event, la.Waiting):
            text = strings.STATUS_LOADING if self._step == "prepare" else strings.STATUS_WAITING_MODEL
            self._set_state(WORKING, text.format(s=event.seconds))
        elif isinstance(event, la.RetryWait):
            self._set_state(WAITING, event.reason)
        elif isinstance(event, w.GateWait):
            if event.reason == "cool_down":
                text = strings.STATUS_GATE_COOL_DOWN.format(provider=self.provider, s=event.seconds_left or 0)
            else:
                text = strings.STATUS_GATE_BUSY.format(provider=self.provider)
            self._set_state(WAITING, text)
        elif isinstance(event, la.ToolWaiting):
            self._set_state(WAITING, strings.STATUS_TOOL_WAITING)
        elif isinstance(event, la.ToolRunning):
            self._tool_lines.append(self.transcript.tool_running(event.name, event.args_preview))
            self._set_state(WORKING, strings.STATUS_RUNNING_TOOL.format(tool=event.name))
        elif isinstance(event, la.ToolDone):
            if self._tool_lines:
                self.transcript.tool_done(self._tool_lines.pop(0), event.ok, event.summary, event.seconds,
                                          event.arguments, event.result_preview, self.provider)
            self._set_state(WORKING, strings.STATUS_THINKING)
        elif isinstance(event, la.Notice):
            self.transcript.notice(event.text)
        elif isinstance(event, la.Continuing):
            from flextool.assistant.session import continuing_text

            self.transcript.notice(continuing_text(event.part, event.of))
            self._set_state(WORKING, strings.STATUS_THINKING)

    def _route_shown(self, prepared: Prepared | None) -> None:
        """The worker's word on what the header shows (G17-D1); the footer follows a changed provider."""
        before = self.provider
        self.prepared = prepared
        self.window.tab_changed(self)
        if self.provider != before and self.window.current_tab() is self:
            self.window.refresh_footer()

    def _stopped_event(self, event: object) -> None:
        """An event of the job stopped in the window: only what settles its lines (G17-D4) — the question was
        sent (``TurnStarted``) or not (``TurnRefused``), and a tool line that was running resolves."""
        from flextool.assistant import llm_adapter as la
        from flextool.gui.assistant import worker as w

        if isinstance(event, w.Rewound):
            self._rewound(event)            # the session deletes the questions also when the answer is stopped
        elif isinstance(event, w.TurnStarted):
            self._started(event)
        elif isinstance(event, w.TurnRefused):
            self._not_sent(event.job_id)
        elif isinstance(event, la.ToolDone) and self._tool_lines:
            self.transcript.tool_done(self._tool_lines.pop(0), event.ok, event.summary, event.seconds,
                                      event.arguments, event.result_preview, self.provider)

    def _chat_ready(self, event: Any) -> None:
        self.conversation_id = event.conversation_id
        self.number = event.number
        self.saved = event.saved
        if self._pending_name is not None:
            name, self._pending_name = self._pending_name, None
            self.window.write_title(event.conversation_id, name)
        self.window.tab_changed(self)

    def _loaded(self, event: Any) -> None:
        """The chat is open: the tab is idle and Send is back first, so a history that does not render still
        leaves a usable chat, and the window always hears of it (a restore waiting for it resolves)."""
        self.conversation_id = event.conversation_id
        self.number = event.number
        self.name = event.title
        self.answered = True
        self.has_messages = bool(event.messages)
        if self.phase is Phase.LOADING:
            self.phase = Phase.IDLE
        self.send_button.state(["!disabled"])
        self.state = IDLE
        self.status_text = ""
        self.status.configure(text="")
        try:
            from flextool.assistant.store import first_question

            self.first_question = first_question(list(event.messages))
            self.transcript.render_history(list(event.messages), self.provider)
            if event.notice:
                self.transcript.notice(event.notice)
            self._totals = dict(event.totals)
            self._show_totals()
            self.refresh_placeholder()
        finally:
            self.window.chat_loaded(self)

    def _finished(self, event: Any) -> None:
        from flextool.assistant.llm_errors import message_for

        result = event.result
        self.transcript.end_answer()
        self._question_usage = list(result.usage)
        self._totals = dict(event.totals)
        self._show_totals()
        if result.rounds or result.text:
            self.answered = True
        error = result.error
        if error is not None and error.code != "cancelled":
            actions: list[tuple[str, Callable[[], None]]] = []
            if error.code in SETTINGS_ERRORS:
                actions = [(strings.ACTION_OPEN_SETTINGS, self.window.open_settings)]
            elif error.code == "context":
                summary = event.summary
                actions = [(strings.ACTION_NEW_CHAT, lambda: self.window.new_chat()),
                           (strings.ACTION_NEW_CHAT_SUMMARY, lambda s=summary: self.window.new_chat(seed=s))]
            self.transcript.error(message_for(error, "gui"), actions)
            self.state = ERROR
        self.active_job = None

    def _idle(self) -> None:
        if self.phase is Phase.ANSWERING:
            self.phase = Phase.IDLE
        self.active_job = None
        self._stopped_job = None
        self.send_button.state(["!disabled"])
        self.stop_button.state(["disabled"])
        for job_id in list(self._job_lines):
            self._not_sent(job_id)          # the worker is done and never started it (a fault before the turn)
        state = ERROR if self.state == ERROR else IDLE
        if not self.window.is_shown(self) and state == IDLE and self.has_messages:
            self.unseen = True
        status = strings.EDITING.format(n=self._editing) if self._editing is not None else ""
        self.status_text = status
        self.status.configure(text=status)
        self._set_state(state)
        self.window.chat_finished(self)
        pending, self._pending = self._pending, None
        if pending is not None and not self.closed and self.phase is Phase.IDLE:
            pending()

    def shown(self) -> None:
        """The tab became the selected one."""
        if self.unseen:
            self.unseen = False
            self.window.tab_changed(self)

    def drain(self, deadline: float, clock: Callable[[], float]) -> None:
        """Hand the worker's queued events to :meth:`handle` until *deadline*; consecutive pieces of the answer
        are joined into one."""
        worker = self.worker
        if worker is None:
            return
        from flextool.assistant.llm_adapter import TextDelta

        pending: tuple[int | None, list[str]] | None = None
        while True:
            try:
                job_id, event = worker.events.get_nowait()
            except queue.Empty:
                break
            if isinstance(event, TextDelta):
                if pending is not None and pending[0] == job_id:
                    pending[1].append(event.text)
                else:
                    if pending is not None:
                        self._handle_safely(pending[0], TextDelta("".join(pending[1])))
                    pending = (job_id, [event.text])
                continue
            if pending is not None:
                self._handle_safely(pending[0], TextDelta("".join(pending[1])))
                pending = None
            self._handle_safely(job_id, event)
            if self.closed or clock() >= deadline:
                break
        if pending is not None:
            self._handle_safely(pending[0], TextDelta("".join(pending[1])))

    def _handle_safely(self, job_id: int | None, event: object) -> None:
        if self.closed:
            return
        try:
            self.handle(job_id, event)
        except tk.TclError:
            if not self.closed:
                logger.exception("The chat tab could not show an event")
        except Exception:  # noqa: BLE001 — one event that cannot be shown must not stop the window's poll
            logger.exception("The chat tab could not show an event")


def run_in_thread(fn: Callable[[], Any], name: str = "flextool-assistant-store") -> threading.Thread:
    """Run *fn* on a short daemon thread (store, ledger and title writes never run on the Tk thread)."""
    thread = threading.Thread(target=fn, name=name, daemon=True)
    thread.start()
    return thread
