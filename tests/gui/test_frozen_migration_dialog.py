"""The frozen-database choice dialog (C-8): "Migrate a copy" / "Migrate in place" (with a second confirm) / "Cancel".

Run headless:  xvfb-run -a python -m pytest tests/gui/test_frozen_migration_dialog.py --run-gui
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path

import pytest

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
    """Open the dialog, press the buttons named in *presses* in turn; returns the choice and the dialog text."""
    texts: list[str] = []

    def press() -> None:
        dialogs = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
        if not dialogs:
            root.after(20, press)
            return
        dlg = dialogs[-1]
        if not texts:
            texts.extend(str(w.cget("text")) for w in dlg.winfo_children() if w.winfo_class() == "TLabel")
        name = presses.pop(0)
        dlg.nametowidget(f"{dlg.winfo_children()[-1]}.{name}").invoke()
        if presses:
            root.after(20, press)

    root.after(50, press)
    choice = ask_frozen_migration_choice(
        root, "ref.sqlite", Path("/data/ref.sqlite"), "Reference 1.0", 82, 83, "ref_v83.sqlite",
        confirm_in_place=lambda _parent: confirm_answers.pop(0),
    )
    return choice, texts


@pytest.mark.parametrize(("presses", "answers", "expected"), [
    (["copy"], [], "copy"),
    (["in_place"], [True], "in_place"),
    (["cancel"], [], "cancel"),
    # Declining the second confirm keeps the dialog open; Cancel then closes it.
    (["in_place", "cancel"], [False], "cancel"),
])
def test_choice_buttons(root: tk.Tk, presses: list[str], answers: list[bool], expected: str) -> None:
    choice, texts = _ask(root, list(presses), list(answers))
    assert choice == expected
    text = " ".join(texts)
    assert "ref.sqlite is frozen ('Reference 1.0', v82). This FlexTool needs v83." in text
    assert "ref_v83.sqlite" in text and "new input source" in text
