"""The GUI message list window and its "Messages (N)" indicator (design §3.4, §4.1–§4.3, package M2).

* :class:`MessageListWindow` — the non-modal window "FlexTool messages: <project>": "Message 1 of N", Previous /
  Next, the message's title, its subject (which database or project) in the muted colour, its paragraphs, then
  [Open documentation], the message's extra action, [Dismiss] (only on a dismissable message) and [Close for now].
  It takes no grab and never waits (``ModalBase`` with ``modal=False``): it is shielded from the main window's
  shortcuts and centred on it.  Close for now = the window button = Escape; Left / Right page.  However it closes,
  it calls :meth:`MessageCenter.window_closed` once with every message it listed.
* :class:`MessageListController` — what ``main_window.py`` holds: one :class:`MessageCenter` per open project, the
  indicator button, the one window, and the deferred auto-show (after idle, then once the main window is viewable
  — retried every 250 ms for at most 5 s from the first try that finds it hidden — and no grab is held — retried
  every 250 ms as long as it is; when the message it was for is gone by then, nothing opens).  An open
  window is updated in place by :meth:`MessageListController.recompute`, never raised.

Widget names (tests find them with ``nametowidget``): ``header.position``, ``header.previous``, ``header.next``,
``content.title``, ``content.subject``, ``content.text``, ``buttons.docs``, ``buttons.action``, ``buttons.dismiss``,
``buttons.close`` — all under the window's ``body``.  Tk thread only.
"""
from __future__ import annotations

import logging
import time
import tkinter as tk
from collections.abc import Callable, Iterable
from pathlib import Path
from tkinter import ttk

from flextool.gui.dialogs import message_box
from flextool.gui.dialogs._modal_base import ModalBase
from flextool.gui.hover_tooltip import attach_tooltip
from flextool.gui.messages import (
    STORE_NAME, GuiMessage, MessageCenter, MessageSource, Trigger, database_key,
)

logger = logging.getLogger(__name__)

BODY_LINES = 12
EMPTY_TITLE = "No messages."
EMPTY_TEXT = "Notices about this project and its databases appear here."
INDICATOR_TOOLTIP = "Notices about this project and its databases"
ACTION_NOTE_MS = 2000
DEFER_STEP_MS = 250
VIEWABLE_WAIT_S = 5.0
_UNRANKED = 1 << 30


def window_title(project: str) -> str:
    return f"FlexTool messages: {project}" if project else "FlexTool messages"


def indicator_text(count: int) -> str:
    return f"Messages ({count})" if count else "Messages"


class MessageListWindow(ModalBase):
    """The list of :attr:`center`'s messages, opened at *start_key* (else the first).  *opened_by* is the
    auto-show trigger that opened it (``None``: opened from the indicator); *on_change* is called after a dismiss
    and when the window has closed."""

    def __init__(self, parent: tk.Misc, center: MessageCenter | None, *, project: str = "",
                 opened_by: Trigger | None = None, start_key: str | None = None,
                 on_change: Callable[[], None] | None = None) -> None:
        super().__init__(parent, window_title(project), modal=False)
        self.center = center
        self.opened_by = opened_by
        self._on_change = on_change
        self.listed: set[str] = set()
        self.index = 0
        self.current: GuiMessage | None = None
        self._note_after: str | None = None
        wrap = self.cw * 60
        self.body.columnconfigure(0, minsize=wrap)

        self.header = ttk.Frame(self.body, name="header")
        self.header.grid(row=0, column=0, sticky="ew")
        self.header.columnconfigure(0, weight=1)
        self.position = ttk.Label(self.header, name="position")
        self.position.grid(row=0, column=0, sticky="w")
        self.previous_button = ttk.Button(self.header, name="previous", text="< Previous",
                                          command=lambda: self.page(-1))
        self.previous_button.grid(row=0, column=1, padx=(self.cw, 0))
        self.next_button = ttk.Button(self.header, name="next", text="Next >", command=lambda: self.page(1))
        self.next_button.grid(row=0, column=2, padx=(self.cw, 0))
        self.top_rule = ttk.Separator(self.body)
        self.top_rule.grid(row=1, column=0, sticky="ew", pady=self.lh // 2)

        # A fixed minimum height, so the buttons do not jump when paging; a longer message makes it taller.
        self.content = ttk.Frame(self.body, name="content")
        self.content.grid(row=2, column=0, sticky="new")
        self.body.rowconfigure(2, minsize=self.lh * BODY_LINES)
        self.title_label = ttk.Label(self.content, name="title", font=self.text_fonts["bold"], wraplength=wrap,
                                     justify="left")
        self.title_label.pack(anchor="w")
        self.subject_label = ttk.Label(self.content, name="subject", font=self.text_fonts["base"],
                                       wraplength=wrap, justify="left", state="disabled")
        self.subject_label.pack(anchor="w")
        self.text = ttk.Frame(self.content, name="text")
        self.text.pack(fill="x")
        ttk.Separator(self.body).grid(row=3, column=0, sticky="ew", pady=self.lh // 2)

        # Created in Tab order: Previous, Next (above), docs, action, Dismiss, Close for now.
        row = ttk.Frame(self.body, name="buttons")
        row.grid(row=4, column=0, sticky="ew")
        row.columnconfigure(2, weight=1)
        self.docs_button = ttk.Button(row, name="docs", text="Open documentation", command=self._open_docs)
        self.docs_button.grid(row=0, column=0, padx=(0, self.cw))
        self.action_button = ttk.Button(row, name="action", command=self._run_action)
        self.action_button.grid(row=0, column=1)
        self.action_tooltip = attach_tooltip(self.action_button, "")
        self.dismiss_button = ttk.Button(row, name="dismiss", text="Dismiss", command=self.dismiss_current)
        self.dismiss_button.grid(row=0, column=3, padx=(0, self.cw))
        self.close_button = ttk.Button(row, name="close", text="Close for now", command=self.cancel)
        self.close_button.grid(row=0, column=4)

        self.bind("<Left>", lambda _e: self.page(-1))
        self.bind("<Right>", lambda _e: self.page(1))
        self.bind("<Destroy>", self._on_destroy, add="+")
        self.render(start_key)
        self.show()
        focus = self.initial_focus()
        if focus is not None:
            focus.focus_set()

    # -- content ---------------------------------------------------------------------------------------------------
    @property
    def messages(self) -> tuple[GuiMessage, ...]:
        return self.center.messages if self.center is not None else ()

    def initial_focus(self) -> tk.Misc | None:
        """Next when there is more than one message (and a next one), else Close for now (never Dismiss)."""
        if len(self.messages) > 1 and self.next_button.instate(["!disabled"]):
            return self.next_button
        return self.close_button

    def render(self, key: str | None = None) -> None:
        """Show the message *key* (else the one shown before, else the one at the same place)."""
        msgs = self.messages
        self.listed |= {m.key for m in msgs}
        keys = [m.key for m in msgs]
        key = key if key in keys else (self.current.key if self.current is not None else None)
        if key in keys:
            self.index = keys.index(key)
        self.index = max(0, min(self.index, len(msgs) - 1))
        self.current = msgs[self.index] if msgs else None
        self._cancel_note()
        for child in self.text.winfo_children():
            child.destroy()
        msg = self.current
        if msg is None:
            self.header.grid_remove()
            self.top_rule.grid_remove()
            self.title_label.configure(text=EMPTY_TITLE)
            self.subject_label.configure(text="")
            self.subject_label.pack_forget()
            self._paragraphs((EMPTY_TEXT,))
            for button in (self.docs_button, self.action_button, self.dismiss_button):
                button.grid_remove()
            self.close_button.configure(text="Close")
            return
        self.header.grid()
        self.top_rule.grid()
        if len(msgs) == 1:
            self.position.configure(text="1 message")
            self.previous_button.grid_remove()
            self.next_button.grid_remove()
        else:
            self.position.configure(text=f"Message {self.index + 1} of {len(msgs)}")
            self.previous_button.grid()
            self.next_button.grid()
            self.previous_button.state(["disabled"] if self.index == 0 else ["!disabled"])
            self.next_button.state(["disabled"] if self.index == len(msgs) - 1 else ["!disabled"])
        self.title_label.configure(text=msg.title)
        self.subject_label.configure(text=msg.subject_label)
        if msg.subject_label:
            self.subject_label.pack(anchor="w", after=self.title_label)
        else:
            self.subject_label.pack_forget()
        self._paragraphs(msg.paragraphs)
        self.docs_button.grid()
        if msg.action is not None:
            self.action_button.configure(text=msg.action.label)
            self.action_tooltip.text = msg.action.tooltip
            self.action_button.grid()
        else:
            self.action_button.grid_remove()
        if msg.spec.dismissable:
            self.dismiss_button.grid()
        else:
            self.dismiss_button.grid_remove()
        self.close_button.configure(text="Close for now")

    def _paragraphs(self, paragraphs: Iterable[str]) -> None:
        for paragraph in paragraphs:
            ttk.Label(self.text, text=paragraph, font=self.text_fonts["base"], wraplength=self.cw * 60,
                      justify="left").pack(anchor="w", fill="x", pady=(self.lh // 2, 0))

    # -- buttons ---------------------------------------------------------------------------------------------------
    def page(self, step: int) -> str:
        if self.messages:
            self.index = max(0, min(self.index + step, len(self.messages) - 1))
            self.current = None
            self.render()
        return "break"

    def _open_docs(self) -> None:
        if self.current is not None:
            message_box.open_url(message_box.docs_url(self.current.spec.docs))

    def _run_action(self) -> None:
        """Run the message's action; a text it returns ("Copied") shows on the button for 2 s."""
        msg = self.current
        if msg is None or msg.action is None:
            return
        try:
            note = msg.action.run()
        except Exception:           # an offline drive while resolving a path, a clipboard error
            logger.exception("Message list: the %r action failed", msg.action.label)
            note = "Not copied" if msg.action.label.startswith("Copy") else "Failed"
        if isinstance(note, str) and note:
            self._cancel_note()
            self.action_button.configure(text=note)
            self._note_after = self.after(ACTION_NOTE_MS, lambda: self._restore_action(msg))

    def _restore_action(self, msg: GuiMessage) -> None:
        self._note_after = None
        if self.current is msg and msg.action is not None:
            self.action_button.configure(text=msg.action.label)

    def _cancel_note(self) -> None:
        if self._note_after is not None:
            try:
                self.after_cancel(self._note_after)
            except tk.TclError:
                pass
            self._note_after = None

    def dismiss_current(self) -> None:
        """Dismiss the shown message; show the next one (the previous one at the end), or "No messages."."""
        msg = self.current
        if msg is None or not msg.spec.dismissable or self.center is None:
            return
        self.center.dismiss(msg)
        self.current = None
        self.render()
        if self.current is None or len(self.messages) <= 1:
            self.close_button.focus_set()
        self._changed()

    def cancel(self) -> None:
        """Close for now."""
        self.close()

    # -- closing ---------------------------------------------------------------------------------------------------
    def _changed(self) -> None:
        if self._on_change is not None:
            try:
                self._on_change()
            except Exception:       # the indicator must never break closing the list
                logger.exception("Message list: updating the indicator failed")

    def _on_destroy(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if event.widget is not self:
            return
        self._cancel_note()
        if self.center is not None:
            self.center.window_closed(self.listed, opened_by=self.opened_by)
        self._changed()

    @property
    def closed(self) -> bool:
        try:
            return not self.winfo_exists()
        except tk.TclError:
            return True


class MessageListController:
    """The message list of *main* (the main window): the indicator *button*, a :class:`MessageCenter` per open
    project (:meth:`open_project`), the one :class:`MessageListWindow` and the deferred auto-show."""

    def __init__(self, main: tk.Misc, button: ttk.Button | None, *, store_path: Path | None = None) -> None:
        self.main = main
        self.button = button
        self._store_path = store_path
        self.center: MessageCenter | None = None
        self.project = ""
        self.window: MessageListWindow | None = None
        self._ranks: dict[str, int] = {}
        self._pending: tuple[str, Trigger] | None = None
        self._pending_after: str | None = None
        self._waiting_since: float | None = None
        if button is not None:
            button.configure(text=indicator_text(0), command=self.open_from_indicator)
            attach_tooltip(button, INDICATOR_TOOLTIP)

    def store_path(self) -> Path:
        if self._store_path is not None:
            return self._store_path
        from flextool.user_dirs import user_config_dir

        return user_config_dir() / STORE_NAME

    # -- project ---------------------------------------------------------------------------------------------------
    def open_project(self, project: str, sources: Iterable[MessageSource] = (), *,
                     folder: Path | None = None) -> None:
        """A project was opened (also at GUI start): close the old list, start a new center (nothing seen).
        *folder* is the project's folder: only stored paths inside it are probed when pruning."""
        self.close()
        self.project = project
        try:
            self.center = MessageCenter(self.store_path(), sources, rank=self._rank, project_dir=folder)
        except Exception:           # the message list must never stop a project from opening
            logger.exception("Message list: cannot start for project %s", project)
            self.center = None
        self.update_indicator()

    def _rank(self, subject_key: str) -> int:
        return self._ranks.get(subject_key, _UNRANKED)

    def set_ranks(self, mgr: object, sources: Iterable[object]) -> None:
        """Rank database messages by input-source number (*sources*: ``InputSourceInfo`` of *mgr*)."""
        ranks: dict[str, int] = {}
        for source in sources:
            try:
                ranks[database_key(mgr.resolve_path(source))] = int(source.number)  # type: ignore[attr-defined]
            except (AttributeError, OSError, TypeError, ValueError):
                continue
        self._ranks = ranks

    def recompute(self, trigger: Trigger = "other") -> None:
        if self.center is None:
            return
        try:
            key = self.center.recompute(trigger)
        except Exception:
            logger.exception("Message list: recompute failed")
            return
        if self.window is not None:
            self.window.render()                # in place, never raised
        elif key is not None and self._pending is None:
            self._pending = (key, trigger)
            self._waiting_since = None
            self._schedule(idle=True)
        self.update_indicator()

    # -- the deferred auto-show ------------------------------------------------------------------------------------
    def _schedule(self, *, idle: bool = False) -> None:
        try:
            if idle:
                self._pending_after = self.main.after_idle(self._try_open)
            else:
                self._pending_after = self.main.after(DEFER_STEP_MS, self._try_open)
        except tk.TclError:
            self._pending = None

    def _cancel_pending(self) -> None:
        self._pending = None
        if self._pending_after is not None:
            try:
                self.main.after_cancel(self._pending_after)
            except tk.TclError:
                pass
            self._pending_after = None

    def _try_open(self) -> None:
        self._pending_after = None
        if self._pending is None or self.center is None:
            self._pending = None
            return
        try:
            viewable = bool(self.main.winfo_viewable())
            grabbed = self.main.grab_current() is not None
        except (tk.TclError, KeyError):
            self._pending = None
            return
        if not viewable:        # the clock starts here, not at recompute (a slow first map at GUI start)
            if self._waiting_since is None:
                self._waiting_since = time.monotonic()
            elif time.monotonic() - self._waiting_since >= VIEWABLE_WAIT_S:
                self._pending = None
                return
        if not viewable or grabbed:     # hidden, or another form is modal (wait as long as it holds the grab)
            self._schedule()
            return
        key, trigger = self._pending
        self._pending = None
        if any(m.key == key for m in self.center.messages):     # gone meanwhile: the next recompute decides
            self._open(trigger, key)

    # -- the window ------------------------------------------------------------------------------------------------
    def _open(self, opened_by: Trigger | None, key: str | None) -> None:
        self.window = MessageListWindow(self.main, self.center, project=self.project, opened_by=opened_by,
                                        start_key=key, on_change=self._window_changed)

    def _raise(self) -> None:
        try:
            self.window.deiconify()     # type: ignore[union-attr]
            self.window.lift()          # type: ignore[union-attr]
            self.window.focus_set()     # type: ignore[union-attr]
        except tk.TclError:
            pass

    def open_from_indicator(self) -> None:
        """The indicator: raise the open window, else open it at the first message."""
        if self.window is not None:
            self._raise()
            return
        self._cancel_pending()
        self._open(None, None)

    def message_about(self, subject_key: str) -> str | None:
        """The key of the first listed message about *subject_key* (e.g. a :func:`database_key`), else ``None``."""
        if self.center is None:
            return None
        return next((m.key for m in self.center.messages if m.subject_key == subject_key), None)

    def open_about(self, subject_key: str) -> bool:
        """Open the list (or show the open one) at the first message about *subject_key*, as the indicator
        would (``opened_by=None``).  ``False``, and nothing opens, when no message concerns it."""
        key = self.message_about(subject_key)
        if key is None:
            return False
        if self.window is not None:
            self.window.render(key)
            self._raise()
            return True
        self._cancel_pending()
        self._open(None, key)
        return True

    def _window_changed(self) -> None:
        if self.window is not None and self.window.closed:
            self.window = None
        self.update_indicator()

    def update_indicator(self) -> None:
        if self.button is None:
            return
        count, accent = self.center.indicator() if self.center is not None else (0, False)
        try:
            if self.button.winfo_exists():
                self.button.configure(text=indicator_text(count), style="Accent.TButton" if accent else "TButton")
        except tk.TclError:
            pass

    def close(self) -> None:
        """Close the window (it records what it listed) and cancel a pending open (project switch, main close)."""
        self._cancel_pending()
        window, self.window = self.window, None
        if window is not None:
            window.close()
