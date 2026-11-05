"""The Solver options dialog: no MIP-gap control, and "Default" entries for scaling and presolve.

The dialog handler is bound to a plain Tk root carrying the main window's solver variables, without building a
full ``MainWindow``.

Run headless (no Tk on the live display):
    xvfb-run -a ~/venv-spi/bin/python -m pytest tests/gui/test_solver_options_dialog.py -q --run-gui
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import pytest

from flextool.gui import main_window as mw

pytestmark = pytest.mark.gui


@pytest.fixture()
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("No display available")
    r.withdraw()
    r.solver_log_level_var = tk.StringVar(master=r, value="normal")
    r.solver_time_limit_var = tk.IntVar(master=r, value=0)
    r.matrix_file_format_var = tk.StringVar(master=r, value="mps")
    r.scaling_var = tk.StringVar(master=r, value="basic")
    r.presolve_var = tk.StringVar(master=r, value="choose")
    yield r
    r.destroy()


def _widgets(widget):
    for child in widget.winfo_children():
        yield child
        yield from _widgets(child)


def _open_dialog(root) -> tk.Toplevel:
    mw.MainWindow._open_solver_options_dialog(root)
    (dlg,) = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
    return dlg


def test_dialog_has_no_mip_gap_control_and_offers_default_entries(root):
    dlg = _open_dialog(root)
    texts = " ".join(str(w.cget("text")) for w in _widgets(dlg)
                     if isinstance(w, (ttk.Label, ttk.Checkbutton, ttk.Radiobutton)))
    assert "MIP gap" not in texts.replace("the MIP gap is set per solve", "")
    assert not [w for w in _widgets(dlg) if isinstance(w, ttk.Checkbutton)]

    radios = [w for w in _widgets(dlg) if isinstance(w, ttk.Radiobutton)]
    by_var: dict[str, dict[str, str]] = {}
    for w in radios:
        by_var.setdefault(str(w.cget("variable")), {})[str(w.cget("text"))] = str(w.cget("value"))
    scaling = next(v for v in by_var.values() if "Solver only" in v)
    presolve = next(v for v in by_var.values() if "Choose" in v)
    assert scaling["Default (solve's scaling)"] == mw._SOLVER_DEFAULT
    assert set(scaling.values()) == {mw._SOLVER_DEFAULT, "off", "solver_only", "basic", "full"}
    assert presolve["Default (on)"] == mw._SOLVER_DEFAULT
    assert set(presolve.values()) == {mw._SOLVER_DEFAULT, "on", "off", "choose"}
    dlg.destroy()


def test_choosing_default_and_ok_stores_no_value(root):
    dlg = _open_dialog(root)
    for w in _widgets(dlg):
        if isinstance(w, ttk.Radiobutton) and str(w.cget("value")) == mw._SOLVER_DEFAULT:
            w.invoke()
    (ok,) = [w for w in _widgets(dlg) if isinstance(w, ttk.Button) and str(w.cget("text")) == "OK"]
    ok.invoke()
    assert root.scaling_var.get() == mw._SOLVER_DEFAULT
    assert root.presolve_var.get() == mw._SOLVER_DEFAULT
    assert mw._var_to_choice(root.scaling_var.get(), mw.SCALING_CHOICES) is None
    assert mw._var_to_choice(root.presolve_var.get(), mw.PRESOLVE_CHOICES) is None
