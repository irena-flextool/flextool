"""The frozen-database choice dialog (C-8): "Upgrade a copy" / "Upgrade in place" (with a second confirm) / "Cancel".

Run headless:  xvfb-run -a python -m pytest tests/gui/test_frozen_migration_dialog.py --run-gui
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path

import pytest

from flextool.gui.dialogs.message_box import MessageBox
from flextool.gui.dialogs.migration_consent_dialog import ask_frozen_migration_choice

pytestmark = pytest.mark.gui


@pytest.fixture()
def root():
    try:
        window = tk.Tk()
    except tk.TclError:
        pytest.skip("No display available")
    window.withdraw()
    yield window
    window.destroy()


def _ask(root: tk.Tk, presses: list[str], confirm_answers: list[bool]) -> tuple[str, list[str]]:
    """Open the dialog, press the buttons named in *presses* in turn (each on the box then open; declining the
    second confirm asks again in a new box); returns the choice and the text of the first box."""
    texts: list[str] = []
    seen: set[str] = set()

    def press() -> None:
        boxes = [w for w in root.winfo_children() if isinstance(w, MessageBox) and str(w) not in seen]
        if not boxes or not boxes[-1].winfo_viewable():
            root.after(20, press)
            return
        box = boxes[-1]
        seen.add(str(box))
        if not texts:
            texts.append(box.text)
        box.body.nametowidget(f"buttons.{presses.pop(0)}").invoke()
        if presses:
            root.after(20, press)

    root.after(50, press)
    choice = ask_frozen_migration_choice(
        root, "ref.sqlite", Path("/data/ref.sqlite"), "Reference 1.0", 78, 79, "ref_v79.sqlite",
        confirm_in_place=lambda _parent: confirm_answers.pop(0),
    )
    return choice, texts


@pytest.mark.parametrize(("presses", "answers", "expected"), [
    (["copy"], [], "copy"),
    (["in_place"], [True], "in_place"),
    (["cancel"], [], "cancel"),
    # Declining the second confirm asks again; Cancel then closes it.
    (["in_place", "cancel"], [False], "cancel"),
])
def test_choice_buttons(root: tk.Tk, presses: list[str], answers: list[bool], expected: str) -> None:
    choice, texts = _ask(root, list(presses), list(answers))
    assert choice == expected
    text = " ".join(texts)
    assert "ref.sqlite is frozen ('Reference 1.0', version 78); this FlexTool needs version 79." in text
    assert "ref_v79.sqlite" in text and "new input source" in text
