"""FlexTool's message box and the shared modal base (design §4.4, §5 "Tk", package B0).

Run headless, never on the owner's display:
    xvfb-run -a sh -c 'openbox & sleep 1; python -m pytest tests/gui/test_message_box.py --run-gui --gui-only -n 0'

Expected values come from the design's rules: answers are the button names given, Escape / the window button give
``None``, the cancel answer has the widget name "cancel", a ``danger`` question focuses its cancel answer, Ctrl+C
copies "title, blank line, each paragraph", ``name_list`` shows 12 names then "and N more".
"""
from __future__ import annotations

import gc
import time
import tkinter as tk
from collections.abc import Callable, Iterator

import pytest

from flextool.gui import error_handling
from flextool.gui.dialogs import _modal_base, message_box
from flextool.gui.dialogs.message_box import MessageBox
from tests.gui.assistant.tk_helpers import key, pump

pytestmark = pytest.mark.gui

BUTTONS = (("Delete", "delete"), ("Keep them", "keep"), ("Cancel", None))


@pytest.fixture(scope="module")
def root() -> Iterator[tk.Tk]:
    try:
        window = tk.Tk()
    except tk.TclError:
        pytest.skip("No display available")
    window.withdraw()
    yield window
    try:
        window.destroy()
    except tk.TclError:
        pass
    del window
    gc.collect()


@pytest.fixture
def host(root: tk.Tk) -> Iterator[tk.Toplevel]:
    window = tk.Toplevel(root)
    window.geometry("600x400+50+50")
    window.update()
    pump(window, lambda: bool(window.winfo_viewable()))
    yield window
    try:
        window.destroy()
    except tk.TclError:
        pass


def _when_open(parent: tk.Misc, action: Callable[[MessageBox], None], errors: list[BaseException],
               timeout: float = 5.0) -> None:
    """Once a box of *parent* is viewable and holds its grab, run *action* on it (from the Tk loop that the box's
    ``wait_window`` runs).  A failing action or a box that never opens destroys the box, so the test cannot hang."""
    deadline = time.monotonic() + timeout

    def poll() -> None:
        boxes = [w for w in parent.winfo_children() if isinstance(w, MessageBox)]
        box = boxes[-1] if boxes else None
        if box is not None and box.winfo_viewable() and box.grab.held:
            try:
                action(box)
            except BaseException as exc:  # noqa: BLE001 — reported by the test after the box returns
                errors.append(exc)
                box.destroy()
            return
        if time.monotonic() > deadline:
            errors.append(AssertionError("the box never became viewable with its grab"))
            for b in boxes:
                b.destroy()
            return
        parent.after(20, poll)

    parent.after(20, poll)


def _button(box: MessageBox, name: str) -> tk.Misc:
    return box.body.nametowidget(f"buttons.{name}")


@pytest.mark.parametrize(("name", "expected"), [("delete", "delete"), ("keep", "keep"), ("cancel", None)])
def test_ask_returns_the_answer_of_the_button_invoked_by_name(host, name, expected) -> None:
    errors: list[BaseException] = []
    seen: dict[str, object] = {}

    def press(box: MessageBox) -> None:
        row = box.body.nametowidget("buttons")
        seen["labels"] = [str(w.cget("text")) for w in row.winfo_children() if w.winfo_class() == "TButton"]
        seen["title"] = box.title()
        seen["transient"] = str(box.transient())
        _button(box, name).invoke()

    _when_open(host, press, errors)
    answer = message_box.ask(host, "Delete these results?", "Two folders.", buttons=BUTTONS)
    assert not errors, errors
    assert answer == expected
    # Action-named buttons in the given order; the title is the window title; transient of the viewable host.
    assert seen["labels"] == ["Delete", "Keep them", "Cancel"]
    assert seen["title"] == "Delete these results?"
    assert seen["transient"] == str(host)


def test_escape_and_window_button_give_none(host) -> None:
    errors: list[BaseException] = []
    _when_open(host, lambda box: key(box, "<Escape>"), errors)
    assert message_box.ask(host, "Replace it?", buttons=(("Replace it", "replace"), ("Cancel", None))) is None
    assert not errors, errors

    # The window button: WM_DELETE_WINDOW runs cancel.
    _when_open(host, lambda box: box.tk.call(box.protocol("WM_DELETE_WINDOW")), errors)
    assert message_box.ask(host, "Replace it?", buttons=(("Replace it", "replace"), ("Keep", "keep"))) is None
    assert not errors, errors


def test_return_invokes_the_focused_button(host) -> None:
    errors: list[BaseException] = []
    _when_open(host, lambda box: key(_button(box, "keep"), "<Return>"), errors)
    assert message_box.ask(host, "Delete these results?", buttons=BUTTONS) == "keep"
    assert not errors, errors


@pytest.mark.parametrize(("kind", "default", "expected"), [
    ("danger", None, "cancel"),         # the safe answer of a danger question
    ("question", None, "delete"),       # else the first button
    ("danger", "keep", "keep"),         # an explicit default wins
])
def test_initial_focus(host, kind, default, expected) -> None:
    errors: list[BaseException] = []
    focus: list[str] = []

    def check(box: MessageBox) -> None:
        focus.append(str(box.initial_focus()))
        box.cancel()

    _when_open(host, check, errors)
    message_box.ask(host, "Delete these results?", buttons=BUTTONS, kind=kind, default=default)
    assert not errors, errors
    assert focus == [focus[0]] and focus[0].endswith(f".buttons.{expected}")


def test_ctrl_c_copies_title_and_paragraphs(host) -> None:
    errors: list[BaseException] = []
    copied: list[str] = []

    def copy(box: MessageBox) -> None:
        box.clipboard_clear()
        key(_button(box, "close"), "<Control-Key-c>")
        copied.append(box.clipboard_get())
        _button(box, "close").invoke()

    _when_open(host, copy, errors)
    assert message_box.tell(host, "The file was not created", "Disk full.", "The full error is in the log.",
                            kind="error") is None
    assert not errors, errors
    assert copied == ["The file was not created\n\nDisk full.\n\nThe full error is in the log."]


def test_docs_button_opens_the_url_and_the_box_stays(host, monkeypatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(message_box, "open_url", opened.append)
    errors: list[BaseException] = []
    still_open: list[bool] = []

    def docs(box: MessageBox) -> None:
        _button(box, "docs").invoke()
        still_open.append(bool(box.winfo_exists()))
        _button(box, "close").invoke()

    _when_open(host, docs, errors)
    message_box.tell(host, "Read the guide", "It explains this.", docs="https://example.org/guide#part")
    assert not errors, errors
    assert opened == ["https://example.org/guide#part"]
    assert still_open == [True]


def test_box_over_a_withdrawn_root_opens_and_returns(root) -> None:
    """A transient of a withdrawn window is never mapped: the box must not be transient there, so it maps, takes
    its grab and returns."""
    assert not root.winfo_viewable()
    errors: list[BaseException] = []
    transient: list[str] = []

    def press(box: MessageBox) -> None:
        transient.append(str(box.transient()))
        _button(box, "replace").invoke()

    _when_open(root, press, errors)
    answer = message_box.ask(root, "Replace it?", buttons=(("Replace it", "replace"), ("Cancel", None)))
    assert not errors, errors
    assert answer == "replace"
    assert transient == [""]


def test_box_grab_is_taken_only_once_viewable(host, monkeypatch) -> None:
    """Every ``grab_set`` of the box happens on a viewable window (xvfb maps fast, so the call order is checked,
    not only that the grab ends up held)."""
    calls: list[bool] = []
    original = tk.Misc.grab_set

    def recording(self: tk.Misc) -> None:
        if isinstance(self, MessageBox):
            calls.append(bool(self.winfo_viewable()))
        original(self)

    monkeypatch.setattr(tk.Misc, "grab_set", recording)
    errors: list[BaseException] = []
    held: list[object] = []

    def press(box: MessageBox) -> None:
        held.append(box.grab_current())
        _button(box, "close").invoke()

    _when_open(host, press, errors)
    message_box.tell(host, "Settings saved")
    assert not errors, errors
    assert calls and all(calls)
    assert len(held) == 1 and isinstance(held[0], MessageBox)


def test_modal_grab_waits_for_a_withdrawn_dialog_to_be_mapped(host, monkeypatch) -> None:
    """``ModalGrab.start`` on a dialog that is still withdrawn calls no ``grab_set``; mapping it later takes the
    grab."""
    dialog = tk.Toplevel(host)
    dialog.withdraw()
    calls: list[bool] = []
    original = tk.Misc.grab_set

    def recording(self: tk.Misc) -> None:
        if self is dialog:
            calls.append(bool(self.winfo_viewable()))
        original(self)

    monkeypatch.setattr(tk.Misc, "grab_set", recording)
    grab = _modal_base.ModalGrab(dialog)
    try:
        grab.start()
        pump(host, timeout=0.2)
        assert calls == [] and not grab.held
        dialog.deiconify()
        pump(host, lambda: grab.held)
        assert calls == [True]
    finally:
        grab.release()
        dialog.destroy()


def test_name_list_caps_at_twelve() -> None:
    names = [f"scenario_{i}" for i in range(20)]
    assert message_box.name_list(names) == "\n".join(names[:12] + ["and 8 more"])
    assert message_box.name_list(names[:12]) == "\n".join(names[:12])
    assert message_box.name_list(names[:3], limit=2) == "scenario_0\nscenario_1\nand 1 more"


def test_bad_buttons_are_refused(host) -> None:
    with pytest.raises(ValueError):
        message_box.ask(host, "Two cancels?", buttons=(("Cancel", None), ("Close", None)))
    with pytest.raises(ValueError):
        message_box.ask(host, "Capital?", buttons=(("Go", "Go"), ("Cancel", None)))
    with pytest.raises(ValueError):
        message_box.tell(host, "Wrong kind", kind="danger")


def test_safe_callback_shows_the_error_in_a_flextool_box(host) -> None:
    errors: list[BaseException] = []
    seen: list[str] = []

    def read(box: MessageBox) -> None:
        seen.append(box.text)
        _button(box, "close").invoke()

    class Owner(tk.Frame):
        @error_handling.safe_callback
        def handler(self) -> None:
            raise RuntimeError("disk full")

    owner = Owner(host)
    _when_open(owner, read, errors)     # the box opens over the first window argument
    assert owner.handler() is None
    assert not errors, errors
    assert seen == ["Something went wrong\n\ndisk full\n\n"
                    "The full error is in the FlexTool log. If it happens again, please report it."]
    owner.destroy()
