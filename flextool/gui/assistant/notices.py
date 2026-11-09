"""The AI assistant's small dialogs (P2 §8): the welcome, the data notice and the chat rename, and what every
modal dialog of the assistant shares.

* :class:`WelcomeDialog` — the owner-approved "How the AI assistant works" (§8.2), with [Open chat]
  [Open settings] [Close].  The first open of the chat window or of the settings shows it first
  (:func:`with_welcome`); any close records that it was seen (on a short daemon thread), then the chosen window
  opens: the chat, the settings, or none for Close.  [Open chat] stays disabled until the settings have been saved
  with a provider.  "How this works" in the chat window shows it again without recording anything.
* :class:`ConsentDialog` — the data notice before the first question goes to a remote route (§8.3), and before
  the settings' model test sends to one.  Focus starts
  on Cancel, Return does not accept, and the accept button ignores activation for the first 500 ms after the
  dialog is mapped, so a key still being typed in the chat cannot accept it.
* :class:`RenameChatDialog` — one entry for the chat's name (§4.3); empty goes back to the number.
* :class:`ConfirmDialog` — one question with an action and Cancel (closing an unsaved chat, deleting a chat).

The dialogs are callback-based (never ``wait_window``), centred on their parent, closed by Escape and the window
button as Cancel, and modal through :class:`ModalGrab`, which takes the grab once the dialog is mapped (retrying for
at most 5 s), and gives a grab another dialog held back when it closes — also when it is destroyed with its parent.
A dialog being destroyed retires its Tk variables on the Tk thread (:func:`retire_variable`), so the garbage
collector never needs the Tk thread for them on a worker thread.
:func:`install_shield` keeps the main window's application-wide shortcuts from firing while the user types in an
assistant window.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from collections.abc import Callable
from pathlib import Path
from tkinter import ttk
from typing import NamedTuple

from flextool.gui.assistant import strings

logger = logging.getLogger(__name__)

#: The main window's ``bind_all`` shortcuts (``main_window.py``) that must not fire from an assistant window.
SHIELDED_KEYS = ("<F9>", "<Alt-Key-c>", "<Key-e>", "<Key-E>", "<Key-v>", "<Key-x>", "<Key-a>", "<Key-A>",
                 "<Control-Key-a>", "<Control-Key-A>")
#: How long the deferred grab is retried, and the step between tries.
GRAB_RETRY_S = 5.0
GRAB_STEP_MS = 20
#: The data notice ignores its accept button for this long after it is mapped.
ACCEPT_GUARD_MS = 500
#: The longest chat name.
NAME_MAX = 60


def install_shield(toplevel: tk.Misc) -> None:
    """Stop the main window's application-wide shortcuts in *toplevel* and everything in it: its path is in every
    descendant's binding tags after the widget's own and its class's, so typing still works."""
    for sequence in SHIELDED_KEYS:
        toplevel.bind(sequence, lambda _e: "break")


def is_aqua(widget: tk.Misc) -> bool:
    try:
        return widget.tk.call("tk", "windowingsystem") == "aqua"
    except tk.TclError:
        return False


def settings_symbol(widget: tk.Misc) -> str:
    """"⚙", or "…" on Windows when the font that has it (Segoe UI Symbol) is missing (P2-O8)."""
    if sys.platform.startswith("win"):
        try:
            families = set(tkfont.families(widget))
        except tk.TclError:
            families = set()
        if "Segoe UI Symbol" not in families:
            return strings.SETTINGS_SYMBOL_FALLBACK
    return strings.SETTINGS_SYMBOL


def centre_on(dialog: tk.Toplevel, parent: tk.Misc) -> None:
    """Place *dialog* in the middle of *parent* (``main_window.py``'s dialogs do the same)."""
    try:
        dialog.update_idletasks()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        dw, dh = dialog.winfo_reqwidth(), dialog.winfo_reqheight()
        dialog.geometry(f"+{max(0, px + (pw - dw) // 2)}+{max(0, py + (ph - dh) // 2)}")
    except tk.TclError:
        pass


def raise_parent(parent: tk.Misc) -> None:
    """Show *parent* again when it is minimised, and raise it, before a dialog of it opens."""
    try:
        top = parent.winfo_toplevel()
        if top.state() == "iconic":
            top.deiconify()
        top.lift()
    except tk.TclError:
        pass


class ModalGrab:
    """The modal grab of *dialog*: taken when the dialog is mapped, the grab it replaced given back on release.

    :meth:`start` remembers the window that holds the grab now (another dialog), then tries ``grab_set`` every
    20 ms until it works or 5 s have passed (a minimised parent never maps the dialog: a log line, no grab).
    :meth:`release` drops the grab and gives the remembered one back when that window still exists; the dialog's
    ``<Destroy>`` does the same, so a dialog destroyed with its parent leaves no grab behind.
    """

    def __init__(self, dialog: tk.Toplevel, *, focus: Callable[[], tk.Misc | None] | None = None) -> None:
        self.dialog = dialog
        self._focus = focus
        self._previous: tk.Misc | None = None
        self._started = 0.0
        self._released = False
        self.held = False
        dialog.bind("<Destroy>", self._on_destroy, add="+")

    def start(self) -> None:
        if self.dialog is None:
            return
        try:
            current = self.dialog.grab_current()
        except (tk.TclError, KeyError):
            current = None
        self._previous = current if current is not None and current is not self.dialog else None
        self._started = time.monotonic()
        self._try()

    def _try(self) -> None:
        dialog = self.dialog
        if self._released or dialog is None or not self._exists(dialog):
            return
        try:
            dialog.grab_set()
        except tk.TclError:
            if time.monotonic() - self._started < GRAB_RETRY_S:
                dialog.after(GRAB_STEP_MS, self._try)
            else:
                logger.info("The %s dialog never became visible, so it is not modal.", dialog.title())
            return
        self.held = True
        target = self._focus() if self._focus is not None else None
        if target is not None:
            try:
                target.focus_set()
            except tk.TclError:
                pass

    @staticmethod
    def _exists(widget: tk.Misc | None) -> bool:
        try:
            return widget is not None and bool(widget.winfo_exists())
        except tk.TclError:
            return False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self.held and self.dialog is not None:
            try:
                self.dialog.grab_release()
            except tk.TclError:
                pass
        self.held = False
        previous = self._previous
        self._previous = None
        # The dialog no longer needs its grab: dropping the references keeps a closed dialog from being held by it.
        self.dialog = None
        self._focus = None
        if previous is not None and self._exists(previous):
            try:
                previous.grab_set()
            except tk.TclError:
                pass

    def _on_destroy(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if self.dialog is not None and event.widget is self.dialog:
            self.release()


class _AssistantDialog(tk.Toplevel):
    """What the assistant's modal dialogs share: transient, shielded, centred, Escape and the window button as
    Cancel, the deferred grab."""

    def __init__(self, parent: tk.Misc, title: str) -> None:
        raise_parent(parent)
        super().__init__(parent)
        self.withdraw()
        self.title(title)
        try:
            top = parent.winfo_toplevel()
            # A transient window follows its master's state: over a withdrawn master it would never be shown.
            if top.winfo_viewable():
                self.transient(top)
        except tk.TclError:
            pass
        self.resizable(False, False)
        install_shield(self)
        from flextool.gui.ui_metrics import get_metrics

        from flextool.gui.assistant.fonts import sync_fonts

        metrics = get_metrics(self)
        self.cw = metrics.cw
        self.lh = metrics.lh
        #: The text of every assistant dialog is in the chat's fonts (:mod:`flextool.gui.assistant.fonts`):
        #: paragraphs and the entry in the base font, headings bold at its size; the buttons keep the theme's.
        self.chat_fonts = sync_fonts(self)
        self.wrap = self.chat_fonts["base"].measure("0") * 60
        self.body = ttk.Frame(self, padding=(metrics.cw * 2, metrics.lh))
        self.body.pack(fill="both", expand=True)
        self.bind("<Escape>", lambda _e: self.cancel())
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.grab = ModalGrab(self, focus=self.initial_focus)
        self._parent = parent
        self.bind("<Destroy>", self._retire, add="+")

    def _retire(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """The dialog is being destroyed (closed, or with its parent): on the Tk thread, give the grab back and
        retire its Tk variables (review G18, C7-m5).  A closed dialog lives on in reference cycles (its widgets
        name it as their master) until the cyclic garbage collector runs, on whichever thread allocates then — a
        worker's.  A variable's ``__del__`` there would call Tcl, which waits for the Tk thread, which may itself
        be waiting for that worker.  A retired variable is unset and its trace commands deleted here, and its
        ``_tk`` is ``None``, so its later ``__del__`` returns at once (tkinter's own guard)."""
        if event.widget is not self:
            return
        self.grab.release()
        for var in _tk_variables(vars(self).values()):
            retire_variable(var)

    def initial_focus(self) -> tk.Misc | None:
        return None

    def show(self) -> None:
        """Centre, map and take the grab (callers build the content first)."""
        centre_on(self, self._parent)
        self.deiconify()
        self.lift()
        self.grab.start()

    def cancel(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        self.grab.release()
        try:
            self.destroy()
        except tk.TclError:
            pass

    def heading(self, parent: tk.Misc, text: str, *, pady: tuple[int, int] = (0, 0)) -> ttk.Label:
        label = ttk.Label(parent, text=text, font=self.chat_fonts["bold"], wraplength=self.wrap, justify="left")
        label.pack(anchor="w", pady=pady)
        return label

    def paragraph(self, parent: tk.Misc, text: str, *, pady: tuple[int, int] = (0, 0)) -> ttk.Label:
        label = ttk.Label(parent, text=text, font=self.chat_fonts["base"], wraplength=self.wrap, justify="left")
        label.pack(anchor="w", fill="x", pady=pady)
        return label

    def buttons(self) -> ttk.Frame:
        row = ttk.Frame(self.body)
        row.pack(fill="x", pady=(self.lh, 0))
        return row


def _tk_variables(values: object) -> list[tk.Variable]:
    """The Tk variables among *values* and one level into the dicts, lists and tuples among them."""
    found: list[tk.Variable] = []
    for value in values:  # type: ignore[attr-defined]
        if isinstance(value, tk.Variable):
            found.append(value)
        elif isinstance(value, dict):
            found += [v for v in value.values() if isinstance(v, tk.Variable)]
        elif isinstance(value, (list, tuple)):
            found += [v for v in value if isinstance(v, tk.Variable)]
    return found


def retire_variable(var: tk.Variable) -> None:
    """Unset *var* in Tcl and delete its trace commands now (Tk thread), so its ``__del__`` needs no Tcl later."""
    if getattr(var, "_tk", None) is None:
        return
    try:
        var.__del__()
    except tk.TclError:
        pass
    var._tk = None


class WelcomeDialog(_AssistantDialog):
    """"How the AI assistant works".  *on_close(action)* gets ``"chat"``, ``"settings"`` or ``"close"``; with
    *record* the first close also records that the welcome was seen (``endpoint.mark_welcome_seen``, on a daemon
    thread).  [Open chat] is enabled when *chat_enabled* is true; ``None`` asks the saved settings whether they
    name a provider (:func:`provider_configured`)."""

    def __init__(self, parent: tk.Misc, on_close: Callable[[str], None] | None = None, *, record: bool = True,
                 config_root: Path | None = None, mark_seen: Callable[[], None] | None = None,
                 chat_enabled: bool | None = None) -> None:
        super().__init__(parent, strings.WELCOME_TITLE)
        self._on_close = on_close
        self._record = record
        self._config_root = config_root
        self._mark_seen = mark_seen
        self._done = False
        symbol = settings_symbol(self)
        self.fonts = fonts = welcome_fonts(self)
        gap = fonts.body.metrics("linespace") // 2
        self.title_label = ttk.Label(self.body, text=strings.WELCOME_TITLE, font=fonts.title,
                                     wraplength=self.wrap, justify="left")
        self.title_label.pack(anchor="w", pady=(0, gap))
        self.section_labels: list[tuple[ttk.Label, ttk.Label]] = []
        for heading, text in strings.WELCOME_SECTIONS:
            head = ttk.Label(self.body, text=heading, font=fonts.heading, wraplength=self.wrap,
                             justify="left")
            head.pack(anchor="w", pady=(gap, 0))
            body = ttk.Label(self.body, text=text.replace(strings.SETTINGS_SYMBOL, symbol), font=fonts.body,
                             wraplength=self.wrap, justify="left")
            body.pack(anchor="w", fill="x")
            self.section_labels.append((head, body))
        row = self.buttons()
        self.close_button = ttk.Button(row, text=strings.CLOSE, command=lambda: self.finish("close"))
        self.close_button.pack(side="right")
        self.settings_button = ttk.Button(row, text=strings.WELCOME_OPEN_SETTINGS,
                                          command=lambda: self.finish("settings"))
        self.settings_button.pack(side="right", padx=(0, self.cw))
        self.chat_enabled = provider_configured(config_root) if chat_enabled is None else chat_enabled
        self.chat_button = ttk.Button(row, text=strings.WELCOME_OPEN_CHAT, command=lambda: self.finish("chat"),
                                      state="normal" if self.chat_enabled else "disabled")
        self.chat_button.pack(side="right", padx=(0, self.cw))
        self.chat_tooltip = None
        if not self.chat_enabled:
            from flextool.gui.hover_tooltip import attach_tooltip

            # A disabled button still gets <Enter>: the tooltip says why it is grey.
            self.chat_tooltip = attach_tooltip(self.chat_button, strings.WELCOME_CHAT_NEEDS_SETTINGS)
        self.bind("<Return>", lambda _e: self.finish("close"))
        self.show()

    def initial_focus(self) -> tk.Misc | None:
        return self.close_button

    def text(self) -> str:
        """The shown text, as :func:`strings.welcome_text` builds it (for the tests)."""
        parts = [str(self.title_label.cget("text"))]
        for head, body in self.section_labels:
            parts += ["", str(head.cget("text")), str(body.cget("text"))]
        return "\n".join(parts)

    def cancel(self) -> None:
        self.finish("close")

    def finish(self, action: str) -> None:
        if self._done or (action == "chat" and not self.chat_enabled):
            return
        self._done = True
        if self._record:
            _record_welcome_seen(self._config_root, self._mark_seen)
        self.close()
        if self._on_close is not None:
            self._on_close(action)


class WelcomeFonts(NamedTuple):
    title: tkfont.Font
    heading: tkfont.Font
    body: tkfont.Font


def welcome_fonts(widget: tk.Misc) -> WelcomeFonts:
    """The welcome's fonts, the chat's (:mod:`flextool.gui.assistant.fonts`): the body in the base font (11 pt at
    the default 10 pt; the font size setting still scales it), the section headings bold at its size, the title as
    a ``##`` heading of an answer."""
    from flextool.gui.assistant.fonts import sync_fonts

    fonts = sync_fonts(widget)
    return WelcomeFonts(fonts["h2"], fonts["bold"], fonts["base"])


def _record_welcome_seen(config_root: Path | None, mark_seen: Callable[[], None] | None) -> threading.Thread:
    def run() -> None:
        try:
            if mark_seen is not None:
                mark_seen()
            else:
                from flextool.assistant import endpoint

                endpoint.mark_welcome_seen(config_root)
        except Exception:  # noqa: BLE001 — the welcome is shown again next time, nothing else is lost
            logger.warning("Could not record that the AI assistant's welcome was shown", exc_info=True)

    thread = threading.Thread(target=run, name="flextool-assistant-welcome", daemon=True)
    thread.start()
    return thread


def welcome_needed(config_root: Path | None = None) -> bool:
    """Whether the welcome comes before the chat window or the settings (a small file read, bounded)."""
    from flextool.assistant import endpoint

    try:
        return not endpoint.welcome_seen(config_root)
    except Exception:  # noqa: BLE001 — an unreadable file: the welcome is shown
        return True


def provider_configured(config_root: Path | None = None) -> bool:
    """Whether the assistant settings have been saved with a provider (a small file read, bounded)."""
    from flextool.assistant.settings import load_settings

    try:
        return bool(load_settings(config_root).provider.preset)
    except Exception:  # noqa: BLE001 — unreadable settings: as if none were saved
        return False


def with_welcome(parent: tk.Misc, then: Callable[[], None], *, open_chat: Callable[[], None] | None = None,
                 open_settings: Callable[[], None] | None = None, config_root: Path | None = None,
                 needed: bool | None = None, chat_enabled: bool | None = None) -> WelcomeDialog | None:
    """Run *then* — or, when the welcome has not been seen, show it: [Open chat] runs *open_chat*, [Open
    settings] *open_settings* (each *then* when not given), Close, Escape and the window button run nothing.
    Returns the dialog when one is shown."""
    if not (welcome_needed(config_root) if needed is None else needed):
        then()
        return None

    def closed(action: str) -> None:
        if action == "chat":
            (open_chat or then)()
        elif action == "settings":
            (open_settings or then)()

    return WelcomeDialog(parent, closed, record=True, config_root=config_root, chat_enabled=chat_enabled)


class ConsentDialog(_AssistantDialog):
    """The data notice of one chat (*chat_label*), or of the settings' model test (*chat_label* ``None``: no chat
    line).  *on_answer(accepted)* is called once: ``True`` from [Send to …], ``False`` from Cancel, Escape, the
    window button or the dialog being destroyed with its parent."""

    def __init__(self, parent: tk.Misc, provider_label: str, notice: str, indicator: str, chat_label: str | None,
                 on_answer: Callable[[bool], None]) -> None:
        super().__init__(parent, strings.NOTICE_TITLE.format(provider=provider_label))
        self._on_answer = on_answer
        self._answered = False
        self._armed = False
        self.notice_label = self.paragraph(self.body, notice)
        self.indicator_label = self.paragraph(self.body, indicator, pady=(self.lh // 2, 0))
        # No chat for the settings' model test (review G18, C7-n4): the notice and its guard are the same.
        self.chat_label = None if chat_label is None else self.paragraph(
            self.body, strings.NOTICE_CHAT.format(number=chat_label), pady=(self.lh // 2, 0))
        row = self.buttons()
        self.cancel_button = ttk.Button(row, text=strings.CANCEL, command=self.cancel)
        self.cancel_button.pack(side="right")
        self.send_button = ttk.Button(row, text=strings.NOTICE_SEND.format(provider=provider_label),
                                      command=self.accept, state="disabled")
        self.send_button.pack(side="right", padx=(0, self.cw))
        self.bind("<Destroy>", self._on_destroy, add="+")
        self.bind("<Map>", self._on_map, add="+")
        self.show()

    def initial_focus(self) -> tk.Misc | None:
        return self.cancel_button

    def _on_map(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if event.widget is self and not self._armed:
            self._armed = True
            self.cancel_button.focus_set()
            self.after(ACCEPT_GUARD_MS, self._enable)

    def _enable(self) -> None:
        try:
            if self.winfo_exists():
                self.send_button.state(["!disabled"])
        except tk.TclError:
            pass

    def accept(self) -> None:
        if "disabled" in self.send_button.state():
            return
        self._answer(True)

    def cancel(self) -> None:
        self._answer(False)

    def _answer(self, accepted: bool) -> None:
        if self._answered:
            return
        self._answered = True
        self.close()
        self._on_answer(accepted)

    def _on_destroy(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if event.widget is self and not self._answered:
            self._answered = True
            self._on_answer(False)


class RenameChatDialog(_AssistantDialog):
    """The chat's name.  *on_done(name)* gets the new name ("" goes back to the number), or ``None`` when the
    dialog was cancelled."""

    def __init__(self, parent: tk.Misc, current: str, on_done: Callable[[str | None], None]) -> None:
        super().__init__(parent, strings.RENAME_TITLE)
        self._on_done = on_done
        self._done = False
        row = ttk.Frame(self.body)
        row.pack(fill="x")
        ttk.Label(row, text=strings.RENAME_LABEL, font=self.chat_fonts["base"]).pack(side="left", padx=(0, self.cw))
        self.var = tk.StringVar(self, value=current[:NAME_MAX])
        check = (self.register(lambda proposed: len(proposed) <= NAME_MAX), "%P")
        self.entry = ttk.Entry(row, textvariable=self.var, width=40, validate="key", validatecommand=check,
                               font=self.chat_fonts["base"])
        self.entry.pack(side="left", fill="x", expand=True)
        self.paragraph(self.body, strings.RENAME_HINT, pady=(self.lh // 2, 0))
        buttons = self.buttons()
        self.ok_button = ttk.Button(buttons, text=strings.OK, command=self.ok)
        self.ok_button.pack(side="right")
        ttk.Button(buttons, text=strings.CANCEL, command=self.cancel).pack(side="right", padx=(0, self.cw))
        self.entry.bind("<Return>", lambda _e: (self.ok(), "break")[1])
        self.entry.bind("<KP_Enter>", lambda _e: (self.ok(), "break")[1])
        self.entry.select_range(0, "end")
        self.show()

    def initial_focus(self) -> tk.Misc | None:
        return self.entry

    def ok(self) -> None:
        self._finish(" ".join(self.var.get().split())[:NAME_MAX])

    def cancel(self) -> None:
        self._finish(None)

    def _finish(self, name: str | None) -> None:
        if self._done:
            return
        self._done = True
        self.close()
        self._on_done(name)


class ConfirmDialog(_AssistantDialog):
    """A question with one action and Cancel (close an unsaved chat, delete a chat, the settings' questions).
    The focus starts on Cancel (*cancel_text*, "Cancel" by default); *on_answer(True)* only from the action
    button."""

    def __init__(self, parent: tk.Misc, title: str, message: str, action: str,
                 on_answer: Callable[[bool], None], *, cancel_text: str = strings.CANCEL) -> None:
        super().__init__(parent, title)
        self._on_answer = on_answer
        self._answered = False
        self.message_label = self.paragraph(self.body, message)
        row = self.buttons()
        self.cancel_button = ttk.Button(row, text=cancel_text, command=self.cancel)
        self.cancel_button.pack(side="right")
        self.action_button = ttk.Button(row, text=action, command=lambda: self._answer(True))
        self.action_button.pack(side="right", padx=(0, self.cw))
        self.bind("<Destroy>", self._on_destroy, add="+")
        self.show()

    def initial_focus(self) -> tk.Misc | None:
        return self.cancel_button

    def cancel(self) -> None:
        self._answer(False)

    def _answer(self, accepted: bool) -> None:
        if self._answered:
            return
        self._answered = True
        self.close()
        self._on_answer(accepted)

    def _on_destroy(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if event.widget is self and not self._answered:
            self._answered = True
            self._on_answer(False)
