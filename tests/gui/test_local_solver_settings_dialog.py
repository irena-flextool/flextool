"""Solver options → On this computer (item 240 B1-B3): the dialog writes this computer's ``solver_settings.json``,
never the project's ``settings.yaml``.

Run headless (no Tk on the live display):
    xvfb-run -a ~/venv-spi/bin/python -m pytest tests/gui/test_local_solver_settings_dialog.py -q --run-gui -n 0
"""
from __future__ import annotations

import json
import stat
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import pytest

from flextool import _local_solver_settings as local
from flextool.gui import main_window as mw
from flextool.gui.dialogs import local_solver_settings_dialog as lsd

pytestmark = [pytest.mark.gui, pytest.mark.skipif(sys.platform == "win32", reason="the wrapper is a POSIX script")]


@pytest.fixture()
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("FLEXTOOL_USER_HOME", str(tmp_path / "user_home"))
    for name in (*local.WRAPPER_ENV.values(), local.ALLOW_UNLISTED_ENV):
        monkeypatch.delenv(name, raising=False)
    workspace = tmp_path / "ws"
    (workspace / "solver_config").mkdir(parents=True)
    (workspace / "solver_config" / "gurobi.opt").write_text("NodefileDir /scratch/nodes\n")
    monkeypatch.chdir(workspace)
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("No display available")
    r.withdraw()
    r.solver_log_level_var = tk.StringVar(master=r, value="normal")
    r.solver_time_limit_var = tk.IntVar(master=r, value=0)
    r.scaling_var = tk.StringVar(master=r, value="basic")
    r.presolve_var = tk.StringVar(master=r, value="choose")
    yield r
    r.destroy()


def _widgets(widget):
    for child in widget.winfo_children():
        yield child
        yield from _widgets(child)


def _program(tmp_path: Path) -> Path:
    path = tmp_path / "bin" / "lic_hold"
    path.parent.mkdir()
    path.write_text("#!/bin/sh\nexec \"$@\"\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _open_from_solver_options(root) -> lsd.LocalSolverSettingsDialog:
    """Open Solver options, then its "On this computer…" button, as a user would (an earlier Solver options
    window is closed first)."""
    for old in [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]:
        old.destroy()
    mw.MainWindow._open_solver_options_dialog(root)
    (options,) = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
    (button,) = [w for w in _widgets(options) if isinstance(w, ttk.Button) and w.cget("text") == "On this computer…"]
    button.invoke()
    (dialog,) = [w for w in options.winfo_children() if isinstance(w, lsd.LocalSolverSettingsDialog)]
    return dialog


def test_ok_writes_this_computers_settings_file_and_not_the_project(root, tmp_path) -> None:
    program = _program(tmp_path)
    dialog = _open_from_solver_options(root)
    assert "gurobi.opt" in " ".join(str(w.cget("text")) for w in _widgets(dialog) if isinstance(w, ttk.Label))
    dialog.program_vars["cplex"].set(str(program))
    dialog.args_texts["cplex"].insert("1.0", "--queue \nopt name\n")  # spaces at the ends are removed
    dialog.allow_var.set(True)
    dialog.trust_var.set(True)
    dialog.ok()
    assert dialog.saved
    data = json.loads(local.settings_path().read_text())
    assert data["wrappers"] == {"cplex": [str(program), "--queue", "opt name"]}
    assert data["allow_unlisted_parameters"] is True
    assert local.trusted_workspaces() == (local.workspace_key(Path.cwd()),)
    assert local.solver_wrapper("cplex").argv == (str(program), "--queue", "opt name")
    assert not list(Path.cwd().rglob("settings.yaml")) and not list(Path.cwd().rglob("solver_settings.json"))
    # Opened again, the dialog shows what was saved; clearing the program and the box removes them.
    dialog = _open_from_solver_options(root)
    assert dialog.wrapper_argv("cplex") == [str(program), "--queue", "opt name"] and dialog.trust_var.get()
    dialog.program_vars["cplex"].set("")
    dialog.trust_var.set(False)
    dialog.ok()
    data = json.loads(local.settings_path().read_text())
    assert data["wrappers"] == {} and data["trusted_workspaces"] == []


def test_a_refused_wrapper_is_shown_and_nothing_is_written(root, tmp_path, monkeypatch) -> None:
    shown = []
    monkeypatch.setattr(lsd.messagebox, "showerror", lambda title, message, **kw: shown.append(message))
    bat = tmp_path / "hold.bat"
    bat.write_text("@echo off\n")
    dialog = _open_from_solver_options(root)
    dialog.program_vars["gurobi"].set(str(bat))
    dialog.trust_var.set(True)
    dialog.ok()
    assert not dialog.saved and dialog.winfo_exists()
    assert len(shown) == 1 and "hold.bat is a script" in shown[0] and "Gurobi program field" in shown[0]
    assert not local.settings_path().exists()
    dialog.cancel()


def test_a_wrapper_set_by_the_environment_is_shown_read_only(root, tmp_path, monkeypatch) -> None:
    program = _program(tmp_path)
    monkeypatch.setenv("FLEXTOOL_SOLVER_WRAPPER_XPRESS", json.dumps([str(program), "-n", "5"]))
    dialog = _open_from_solver_options(root)
    assert dialog.wrapper_argv("xpress") == [str(program), "-n", "5"]
    assert "set by FLEXTOOL_SOLVER_WRAPPER_XPRESS" in [
        str(w.cget("text")) for w in _widgets(dialog) if isinstance(w, ttk.Label)]
    dialog.ok()
    assert "xpress" not in json.loads(local.settings_path().read_text()).get("wrappers", {})


def test_the_open_projects_box_trusts_its_solver_config_only_when_ticked(root, tmp_path) -> None:
    """With a project open the dialog shows a second box for the project's own ``solver_config`` (item 231's
    ladder, item 240 B3), with its edited files; OK writes the project's state only when the box was changed."""
    workspace = Path.cwd()
    project = workspace / "projects" / "P"
    (project / "solver_config").mkdir(parents=True)
    (project / "solver_config" / "cplex.opt").write_text("workdir /scratch/cplex\n")
    root.current_project = "P"
    dialog = _open_from_solver_options(root)
    assert dialog.project == project and dialog.project_trust_var is not None
    texts = " ".join(str(w.cget("text")) for w in _widgets(dialog) if isinstance(w, (ttk.Label, ttk.Checkbutton)))
    assert "Trust this project's solver option files (P)" in texts and "Edited: cplex.opt" in texts
    dialog.ok()  # nothing changed: the project is not written
    assert local.workspace_key(project) not in local.trusted_workspaces()
    dialog = _open_from_solver_options(root)
    dialog.project_trust_var.set(True)
    dialog.ok()
    assert local.is_trusted_workspace(project) and not local.is_trusted_workspace(workspace)
    dialog = _open_from_solver_options(root)
    assert dialog.project_trust_var.get()
    dialog.project_trust_var.set(False)
    dialog.ok()
    assert not local.is_trusted_workspace(project)


def test_ok_keeps_the_entries_it_does_not_show(root, tmp_path) -> None:
    """Review b2b3 L7: with an entry FlexTool cannot read in the list, the boxes show unticked with the problem,
    and OK with the boxes unchanged writes no trust state: every entry is kept as stored.  Ticking the workspace
    box changes only the workspace's entry."""
    other = tmp_path / "other"
    other.mkdir()
    stored = [".", str(other)]
    local.settings_path().parent.mkdir(parents=True)
    local.settings_path().write_text(json.dumps({"format": 1, "trusted_workspaces": stored}))
    dialog = _open_from_solver_options(root)
    assert not dialog.trust_var.get()
    texts = " ".join(str(w.cget("text")) for w in _widgets(dialog) if isinstance(w, ttk.Label))
    assert "cannot read its list of trusted folders" in texts and "not a full folder path" in texts
    dialog.ok()
    assert json.loads(local.settings_path().read_text())["trusted_workspaces"] == stored
    dialog = _open_from_solver_options(root)
    dialog.trust_var.set(True)
    dialog.ok()
    kept = json.loads(local.settings_path().read_text())["trusted_workspaces"]
    assert kept[:2] == stored and kept[2]["path"] == local.workspace_key(Path.cwd())
