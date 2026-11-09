"""The Tk tests of the AI assistant's window (P2 §13.2; opt-in with ``--run-gui``, under ``xvfb-run -a``).

They run under the assistant tests' rules — the user config and cache folders in ``tmp_path``, an in-memory
keyring, no connection off this computer, blocking fake backends released at the end — and share one withdrawn Tk
root per module.  ``host`` is a mapped window to open dialogs over (a transient dialog of a withdrawn window is
never shown).  ``tk_helpers.pump`` runs the Tk loop until a condition holds.
"""
from __future__ import annotations

import gc
import tkinter as tk
from collections.abc import Iterator

import pytest

from tests.assistant.conftest import (  # noqa: F401
    fake_backend_faults,
    fake_keyring,
    user_cache_in_tmp,
    user_config_in_tmp,
)
from tests.assistant.llm.conftest import fake_backends, no_network  # noqa: F401


@pytest.fixture(scope="module")
def tk_root() -> Iterator[tk.Tk]:
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("No display available")
    root.withdraw()
    yield root
    try:
        root.destroy()
    except tk.TclError:
        pass
    # Destroyed widgets still referenced from cycles (dialogs, the closures of their threads) would otherwise be
    # collected later on some worker thread, and the last reference to this root with them: Tcl aborts when an
    # interpreter is deleted off its own thread.  Collect them here, on the Tk thread.
    del root
    gc.collect()


@pytest.fixture
def host(tk_root: tk.Tk) -> Iterator[tk.Toplevel]:
    window = tk.Toplevel(tk_root)
    window.geometry("600x400+50+50")
    window.update()
    yield window
    try:
        if tk_root.grab_current() is not None:
            tk_root.grab_current().grab_release()
    except (tk.TclError, KeyError):
        pass
    try:
        window.destroy()
    except tk.TclError:
        pass
