"""No native message box is left in FlexTool (design §5 "Guard", critique change 9).

Every box goes through ``flextool.gui.dialogs.message_box`` (``message_box.tell`` / ``message_box.ask`` as module
attributes, so a test's monkeypatch reaches them).  An AST scan of all of ``flextool/**/*.py`` fails on any use of
``tkinter.messagebox`` however it is spelled (``import tkinter.messagebox as mb``, ``from tkinter import
messagebox``, ``from tkinter.messagebox import showinfo``, ``tk.messagebox.showinfo``, a ``"tkinter.messagebox"``
import by name), on ``from flextool.gui.dialogs.message_box import ask|tell``, and on any ``simpledialog`` use
outside the kept input prompts below.  ``flextool/env_check.py`` keeps native boxes: it runs before the GUI exists,
with the standard library only (design §4.4).  A plain text grep misses the aliased and attribute spellings.
Tk-free.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "flextool"

#: Files that may use native ``tkinter.messagebox``.
NATIVE_ALLOWED = {"flextool/env_check.py"}

#: The kept input prompts (design decision 12): (file, function, simpledialog function).
PROMPTS_ALLOWED = {
    ("flextool/gui/main_window.py", "_on_combo_rename", "askstring"),
    ("flextool/gui/main_window.py", "_on_freeze_source", "askstring"),
    ("flextool/gui/main_window.py", "_on_font_size_custom", "askinteger"),
    ("flextool/gui/main_window.py", "_on_code_font_size_custom", "askinteger"),
}
PROMPT_FILES = {path for path, _function, _call in PROMPTS_ALLOWED}

_BOX_MODULE = "flextool.gui.dialogs.message_box"


def _findings(source: str, rel: str) -> list[str]:
    """Each forbidden use in *source* (the file *rel*, relative to the repository) as "rel:line: what"."""
    tree = ast.parse(source, filename=rel)
    native_ok = rel in NATIVE_ALLOWED
    found: list[str] = []

    def bad(node: ast.AST, what: str) -> None:
        found.append(f"{rel}:{getattr(node, 'lineno', 0)}: {what}")

    # The function each node is in, for the prompt allow-list.  ``ast.walk`` is breadth-first, so an inner
    # function comes after its outer one and overwrites it: the innermost function wins.
    enclosing: dict[ast.AST, str] = {}
    for func in ast.walk(tree):
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(func):
                if child is not func:
                    enclosing[child] = func.name

    allowed_prompt_nodes: set[int] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "simpledialog"
                and (rel, enclosing.get(node, ""), node.func.attr) in PROMPTS_ALLOWED):
            allowed_prompt_nodes.add(id(node.func.value))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("tkinter.messagebox") and not native_ok:
                    bad(node, f"import {alias.name}")
                if alias.name.startswith("tkinter.simpledialog"):
                    bad(node, f"import {alias.name}")
                if alias.name == _BOX_MODULE and alias.asname is not None:
                    bad(node, f"import {_BOX_MODULE} as {alias.asname}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = {alias.name for alias in node.names}
            if module.startswith("tkinter.messagebox") and not native_ok:
                bad(node, f"from {module} import {', '.join(sorted(names))}")
            if module == "tkinter" and "messagebox" in names and not native_ok:
                bad(node, "from tkinter import messagebox")
            if module.startswith("tkinter.simpledialog"):
                bad(node, f"from {module} import {', '.join(sorted(names))}")
            if module == "tkinter" and "simpledialog" in names and rel not in PROMPT_FILES:
                bad(node, "from tkinter import simpledialog (not a kept prompt's file)")
            if module == _BOX_MODULE and names & {"ask", "tell", "*"}:
                bad(node, f"from {_BOX_MODULE} import {', '.join(sorted(names))} (use message_box.ask / .tell)")
        elif isinstance(node, ast.Attribute):
            if node.attr == "messagebox" and not native_ok:
                bad(node, ".messagebox attribute")
            if node.attr == "simpledialog":
                bad(node, ".simpledialog attribute")
        elif isinstance(node, ast.Name) and node.id == "simpledialog" and isinstance(node.ctx, ast.Load):
            if id(node) not in allowed_prompt_nodes:
                bad(node, "simpledialog use outside the kept prompts")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # An import by name (importlib.import_module, __import__, getattr(tk, "messagebox")).
            if node.value in ("tkinter.messagebox", "messagebox") and not native_ok:
                bad(node, f"string {node.value!r}")
            if node.value in ("tkinter.simpledialog", "simpledialog"):
                bad(node, f"string {node.value!r}")
    return found


def test_no_native_message_box_in_flextool() -> None:
    files = sorted(PACKAGE.rglob("*.py"))
    assert len(files) > 100, "the scan found too few files: wrong root?"
    findings: list[str] = []
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        findings.extend(_findings(path.read_text(encoding="utf-8"), rel))
    assert not findings, "native boxes left (use flextool.gui.dialogs.message_box):\n" + "\n".join(findings)


def test_the_kept_prompts_still_exist() -> None:
    """A stale allow-list entry would let a new prompt in under an old name: each entry must match a call."""
    seen: set[tuple[str, str, str]] = set()
    for rel in PROMPT_FILES:
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        for func in ast.walk(tree):
            if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for node in ast.walk(func):
                    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and isinstance(node.func.value, ast.Name) and node.func.value.id == "simpledialog"):
                        seen.add((rel, func.name, node.func.attr))
    assert PROMPTS_ALLOWED <= seen, PROMPTS_ALLOWED - seen


@pytest.mark.parametrize("snippet", [
    "import tkinter.messagebox",
    "import tkinter.messagebox as mb",
    "from tkinter import messagebox",
    "from tkinter import ttk, messagebox as mb",
    "from tkinter.messagebox import showerror",
    "import tkinter as tk\ntk.messagebox.showinfo('a', 'b')",
    "import tkinter\ntkinter.messagebox.askyesno('a', 'b')",
    "import importlib\nimportlib.import_module('tkinter.messagebox')",
    "import tkinter as tk\ngetattr(tk, 'messagebox')",
    "from tkinter.simpledialog import askstring",
    "from tkinter import simpledialog\nsimpledialog.askstring('a', 'b')",
    "from flextool.gui.dialogs.message_box import ask",
    "from flextool.gui.dialogs.message_box import tell as say",
    "from flextool.gui.dialogs.message_box import *",
    "import flextool.gui.dialogs.message_box as mbx",
])
def test_the_scan_catches_every_spelling(snippet: str) -> None:
    assert _findings(snippet, "flextool/gui/somewhere.py"), snippet


def test_the_scan_allows_the_box_and_the_kept_prompts() -> None:
    assert _findings("from flextool.gui.dialogs import message_box\nmessage_box.tell(None, 'T', 'p')",
                     "flextool/gui/somewhere.py") == []
    assert _findings("from tkinter import messagebox\nmessagebox.showerror('a', 'b')", "flextool/env_check.py") == []
    prompt = "from tkinter import simpledialog\ndef _on_combo_rename(self):\n    simpledialog.askstring('a', 'b')\n"
    assert _findings(prompt, "flextool/gui/main_window.py") == []
    # The same prompt in another function of that file is not allowed.
    assert _findings(prompt.replace("_on_combo_rename", "_on_other"), "flextool/gui/main_window.py")
