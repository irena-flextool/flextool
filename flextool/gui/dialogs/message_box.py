"""FlexTool's one message box (design §4.4, owner answer Q1), on the shared modal base.

* :func:`tell` — something happened (a result, a validation note, an error); one [Close] button.
* :func:`ask` — a question answered by action-named buttons ("Delete", "Replace it", "Stop and close"), never
  Yes/No; returns the chosen answer, ``None`` for the cancel answer, Escape and the window button.
* :func:`name_list` — a list of names for a paragraph, capped at 12 lines + "and N more".
* :func:`all_names` — every name, for a destructive or in-place question (what is deleted or changed is never
  hidden): more than 12 show in a scrolled list.

The layout is the message list's: the title as the heading (and the window title), the paragraphs, an "Open
documentation" button on the left when *docs* is given, the answer buttons on the right.  The box is synchronous
(``wait_window``), so call sites keep ``if message_box.ask(...):``; over a withdrawn parent it is not transient and
still maps, so it never waits on a box that cannot be seen.  Return invokes the focused button; the focus starts on
*default*, else on the cancel answer of a ``"danger"`` question, else on the first button.  Ctrl+C copies the title
and the paragraphs.  Each button's Tk widget name is its answer (``"cancel"`` for ``None``, ``"close"`` for
:func:`tell`, ``"docs"`` for the documentation), in the frame ``buttons`` of the box's body, so a test can
``invoke()`` it by name.

Call sites use the module attributes (``message_box.ask(...)``), never ``from … import ask``, so a test's
``monkeypatch.setattr(message_box, "ask", …)`` reaches them.  Tk thread only.
"""
from __future__ import annotations

import tkinter as tk
import webbrowser
from collections.abc import Callable, Iterable, Sequence
from tkinter import ttk

from flextool.gui.dialogs._modal_base import ModalBase

#: Opens the documentation (a test replaces it).
open_url: Callable[[str], object] = webbrowser.open

DOCS_LABEL = "Open documentation"
CLOSE_LABEL = "Close"
CANCEL_LABEL = "Cancel"
TELL_KINDS = ("info", "warning", "error")
ASK_KINDS = ("question", "danger")
NAME_LIST_LIMIT = 12


def name_list(names: Iterable[object], limit: int = NAME_LIST_LIMIT) -> str:
    """One name per line: at most *limit* of them, then "and N more" for the rest."""
    items = [str(name) for name in names]
    if len(items) <= limit:
        return "\n".join(items)
    return "\n".join([*items[:limit], f"and {len(items) - limit} more"])


class NameList(str):
    """A paragraph of every name, one per line (:func:`all_names`): the box shows more than
    :data:`NAME_LIST_LIMIT` of them in a scrolled list."""


def all_names(names: Iterable[object]) -> NameList:
    """Every one of *names*, one per line, for a question about deleting or changing them."""
    return NameList("\n".join(str(name) for name in names))


def docs_url(docs: str) -> str:
    """The web address of *docs*: a URL as it is, a coded warning (``"W-0490"``) its section, else a
    ``"page.md#anchor"`` of ``docs/``."""
    if docs.startswith(("http://", "https://")):
        return docs
    from flextool.gui.messages import docs_page_url
    from flextool.run_warnings.registry import docs_url as warning_docs_url

    return warning_docs_url(docs) if docs.startswith("W-") else docs_page_url(docs)


def widget_name(answer: str | None) -> str:
    """The Tk widget name of the button that gives *answer*."""
    return "cancel" if answer is None else answer


class MessageBox(ModalBase):
    """The box itself; :func:`tell` and :func:`ask` build it and wait for it.  :attr:`answer` is the answer given
    (``None`` until one is, and for the cancel answer); :attr:`answer_buttons` maps each answer to its button."""

    def __init__(self, parent: tk.Misc, title: str, paragraphs: Sequence[str],
                 buttons: Sequence[tuple[str, str | None]], *, default: str | None = None, kind: str = "info",
                 docs: str | None = None) -> None:
        names = [widget_name(answer) for _label, answer in buttons]
        if not buttons or len(set(names)) != len(names) or "docs" in names:
            raise ValueError(f"message box buttons need distinct answers other than 'docs': {names}")
        if any("." in name or not name[:1].islower() for name in names):
            raise ValueError(f"a button's answer must be a Tk widget name (lower case first, no '.'): {names}")
        super().__init__(parent, title)
        self.kind = kind
        self.answer: str | None = None
        self.text = "\n\n".join([title, *paragraphs])
        self.heading(self.body, title)
        for paragraph in paragraphs:
            if isinstance(paragraph, NameList) and paragraph.count("\n") >= NAME_LIST_LIMIT:
                self._scrolled_names(paragraph)
            else:
                self.paragraph(self.body, paragraph, pady=(self.lh // 2, 0))
        row = ttk.Frame(self.body, name="buttons")
        row.pack(fill="x", pady=(self.lh, 0))
        self.docs_button: ttk.Button | None = None
        if docs:
            url = docs_url(docs)
            self.docs_button = ttk.Button(row, name="docs", text=DOCS_LABEL, command=lambda: open_url(url))
            self.docs_button.pack(side="left", padx=(0, self.cw))
        ttk.Frame(row).pack(side="left", fill="x", expand=True)
        # Created in the given order, so Tab goes through them as they read.
        self.answer_buttons: dict[str | None, ttk.Button] = {}
        for index, (label, answer) in enumerate(buttons):
            button = ttk.Button(row, name=widget_name(answer), text=label,
                                command=lambda a=answer: self.finish(a))
            button.pack(side="left", padx=(self.cw if index else 0, 0))
            self.answer_buttons[answer] = button
        if default is not None and default in self.answer_buttons:
            self._focus_answer: str | None = default
        elif kind == "danger" and None in self.answer_buttons:
            self._focus_answer = None
        else:
            self._focus_answer = buttons[0][1]
        for sequence in ("<Control-Key-c>", "<Control-Key-C>"):
            self.bind(sequence, self.copy_text)
        try:
            if self.tk.call("tk", "windowingsystem") == "aqua":
                self.bind("<Command-Key-c>", self.copy_text)
        except tk.TclError:
            pass
        # A non-modal window of the same parent (the message list) may stay above a later box: raise it once mapped.
        self.bind("<Map>", self._lift_once, add="+")
        self._lifted = False
        self.show()

    def _scrolled_names(self, names: str) -> None:
        """*names* in a read-only list, :data:`NAME_LIST_LIMIT` lines high, scrolled (widget name ``names``)."""
        frame = ttk.Frame(self.body)
        frame.pack(fill="x", pady=(self.lh // 2, 0))
        text = tk.Text(frame, name="names", height=NAME_LIST_LIMIT, width=60, wrap="none",
                       font=self.text_fonts["base"], relief="flat", borderwidth=0, highlightthickness=0)
        bar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=bar.set)
        text.insert("1.0", names)
        text.configure(state="disabled")
        bar.pack(side="right", fill="y")
        text.pack(side="left", fill="x", expand=True)

    def initial_focus(self) -> tk.Misc | None:
        return self.answer_buttons.get(self._focus_answer)

    def _lift_once(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if event.widget is self and not self._lifted:
            self._lifted = True
            self.lift()

    def copy_text(self, _event: tk.Event | None = None) -> str:  # type: ignore[type-arg]
        """Put the title and the paragraphs on the clipboard (as a native Windows box does)."""
        try:
            self.clipboard_clear()
            self.clipboard_append(self.text)
        except tk.TclError:
            pass
        return "break"

    def finish(self, answer: str | None) -> None:
        self.answer = answer
        self.close()

    def cancel(self) -> None:
        self.finish(None)

    def wait(self) -> str | None:
        try:
            self.wait_window()
        except tk.TclError:
            pass
        return self.answer


def _parent_or_root(parent: tk.Misc | None) -> tk.Misc:
    if parent is not None:
        return parent
    root = getattr(tk, "_default_root", None)
    if root is None:
        raise RuntimeError("A FlexTool message box needs a Tk window to open over")
    return root


def tell(parent: tk.Misc | None, title: str, *paragraphs: str, kind: str = "info", docs: str | None = None) -> None:
    """Show *title* and *paragraphs* with one [Close] button; returns when the box is closed."""
    if kind not in TELL_KINDS:
        raise ValueError(f"tell kind must be one of {TELL_KINDS}, not {kind!r}")
    MessageBox(_parent_or_root(parent), title, paragraphs, ((CLOSE_LABEL, "close"),), kind=kind, docs=docs).wait()


def ask(parent: tk.Misc | None, title: str, *paragraphs: str, buttons: Sequence[tuple[str, str | None]],
        default: str | None = None, kind: str = "question", docs: str | None = None) -> str | None:
    """Ask *title*; *buttons* are ``(label, answer)`` pairs with ``None`` the cancel answer.  Returns the answer of
    the pressed button, or ``None`` for Escape and the window button."""
    if kind not in ASK_KINDS:
        raise ValueError(f"ask kind must be one of {ASK_KINDS}, not {kind!r}")
    box = MessageBox(_parent_or_root(parent), title, paragraphs, buttons, default=default, kind=kind, docs=docs)
    return box.wait()
