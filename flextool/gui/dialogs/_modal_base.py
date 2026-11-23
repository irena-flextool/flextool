"""What every FlexTool modal dialog shares (moved from ``assistant/notices.py``, design §4.4): :class:`ModalBase`
and the helpers it uses.

:class:`ModalBase` is a ``Toplevel`` that is transient of its parent only when the parent is viewable (a transient
of a withdrawn window is never mapped, so a box over a hidden main window would never be seen and ``wait_window``
would never return), shielded from the main window's ``bind_all`` shortcuts (:func:`install_shield`), centred on its
parent, closed by Escape and the window button as Cancel, and modal through :class:`ModalGrab`, which takes the grab
only once the dialog is viewable (retrying for at most 5 s), and gives a grab another dialog held back when it
closes — also when it is destroyed with its parent.  With *return_invokes* the Return key invokes the focused
button (``ttk.Button`` does not); *modal* ``False`` skips the grab.  A dialog being destroyed retires its Tk
variables on the Tk thread (:func:`retire_variable`), so the garbage collector never needs the Tk thread for them on
a worker thread.  Tk thread only.
"""
from __future__ import annotations

import logging
import time
import tkinter as tk
import tkinter.font as tkfont
from collections.abc import Callable, Mapping
from tkinter import ttk

logger = logging.getLogger(__name__)

#: The main window's ``bind_all`` shortcuts (``main_window.py``) that must not fire from a dialog.
SHIELDED_KEYS = ("<F9>", "<Alt-Key-c>", "<Key-e>", "<Key-E>", "<Key-v>", "<Key-x>", "<Key-a>", "<Key-A>",
                 "<Control-Key-a>", "<Control-Key-A>")
#: How long the deferred grab is retried, and the step between tries.
GRAB_RETRY_S = 5.0
GRAB_STEP_MS = 20


def install_shield(toplevel: tk.Misc) -> None:
    """Stop the main window's application-wide shortcuts in *toplevel* and everything in it: its path is in every
    descendant's binding tags after the widget's own and its class's, so typing still works."""
    for sequence in SHIELDED_KEYS:
        toplevel.bind(sequence, lambda _e: "break")


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
    """The modal grab of *dialog*: taken once the dialog is viewable, the grab it replaced given back on release.

    :meth:`start` remembers the window that holds the grab now (another dialog), then tries every 20 ms until the
    dialog is viewable and ``grab_set`` works, or 5 s have passed (a dialog that is never mapped: a log line, no
    grab).  ``grab_set`` is never called on a dialog that is not viewable yet (it raises there, and a window manager
    may map the dialog later than xvfb does).  :meth:`release` drops the grab and gives the remembered one back when
    that window still exists; the dialog's ``<Destroy>`` does the same, so a dialog destroyed with its parent leaves
    no grab behind.
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
            if not dialog.winfo_viewable():
                raise tk.TclError("not viewable yet")
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


def theme_fonts(widget: tk.Misc) -> dict[str, tkfont.Font]:
    """FlexTool's own fonts by role: ``"base"`` the body text font, ``"bold"`` the heading font."""
    return {"base": tkfont.nametofont("TkTextFont", root=widget),
            "bold": tkfont.nametofont("TkHeadingFont", root=widget)}


class ModalBase(tk.Toplevel):
    """A modal dialog: transient (when the parent is viewable), shielded, centred, Escape and the window button as
    :meth:`cancel`, the deferred grab.  *fonts* maps ``"base"`` and ``"bold"`` to the fonts of the paragraphs and
    headings (``None``: :func:`theme_fonts`); the buttons keep the theme's.  Subclasses build their content in
    :attr:`body`, then call :meth:`show`."""

    def __init__(self, parent: tk.Misc, title: str, *, fonts: Mapping[str, tkfont.Font] | None = None,
                 modal: bool = True, return_invokes: bool = True) -> None:
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

        metrics = get_metrics(self)
        self.cw = metrics.cw
        self.lh = metrics.lh
        self.text_fonts = dict(fonts) if fonts is not None else theme_fonts(self)
        self.wrap = self.text_fonts["base"].measure("0") * 60
        self.body = ttk.Frame(self, padding=(metrics.cw * 2, metrics.lh))
        self.body.pack(fill="both", expand=True)
        self.bind("<Escape>", lambda _e: self.cancel())
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        if return_invokes:
            self.bind("<Return>", self._invoke_focused)
            self.bind("<KP_Enter>", self._invoke_focused)
        self.modal = modal
        self.grab = ModalGrab(self, focus=self.initial_focus)
        self._parent = parent
        self.bind("<Destroy>", self._retire, add="+")

    def _invoke_focused(self, _event: tk.Event) -> str | None:  # type: ignore[type-arg]
        """Return: invoke the focused button of this dialog (a ``ttk.Button`` only reacts to the space bar)."""
        try:
            focused = self.focus_get()
        except (tk.TclError, KeyError):
            return None
        if isinstance(focused, (ttk.Button, tk.Button)) and focused.winfo_toplevel() is self:
            if not (isinstance(focused, ttk.Button) and focused.instate(["disabled"])):
                focused.invoke()
            return "break"
        return None

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
        if self.modal:
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
        label = ttk.Label(parent, text=text, font=self.text_fonts["bold"], wraplength=self.wrap, justify="left")
        label.pack(anchor="w", pady=pady)
        return label

    def paragraph(self, parent: tk.Misc, text: str, *, pady: tuple[int, int] = (0, 0)) -> ttk.Label:
        label = ttk.Label(parent, text=text, font=self.text_fonts["base"], wraplength=self.wrap, justify="left")
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
