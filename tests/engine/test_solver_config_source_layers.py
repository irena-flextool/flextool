"""Item 231: the solver option files come FILE BY FILE from a ladder of folders (project > workspace > FlexTool's
default), and an explicit ladder is closed.

The expected files and header lines are the hand-derived oracle of the design review (cases A, B, C), not the
module's output: ``/ws`` is a FlexTool workspace whose ``solver_config/`` holds an edited ``highs.opt`` and
``gurobi.opt`` and an unedited-template ``cplex.opt`` (no xpress, copt); ``P = /ws/projects/P``; the input database
is ``P/input_sources/db.sqlite``.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest

from flextool._resources import package_data_path
from flextool._solver_config_source import (
    gui_solver_config_dirs,
    read_solver_option_files,
    resolve_solver_config_source,
    snapshot_solver_option_files,
    split_solver_config_dirs,
)
from flextool.run_options import Resolved, RunOptions

_DEFAULT = "default"


def _template(solver: str) -> str:
    return Path(package_data_path(f"solver_config/{solver}.opt.template")).read_text(encoding="utf-8")


@pytest.fixture
def ws(tmp_path: Path) -> dict:
    root = tmp_path / "ws"
    (root / "templates").mkdir(parents=True)
    (root / "templates" / "project_folder.txt").write_text("projects\n")
    cfg = root / "solver_config"
    cfg.mkdir()
    (cfg / "highs.opt").write_text("mip_rel_gap = 0.02\n")
    (cfg / "gurobi.opt").write_text("Threads 4\n")
    (cfg / "cplex.opt").write_text(_template("cplex"), encoding="utf-8")  # not the Windows ANSI code page
    project = root / "projects" / "P"
    (project / "input_sources").mkdir(parents=True)
    db = project / "input_sources" / "db.sqlite"
    db.write_bytes(b"")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    return {"root": root, "cfg": cfg, "project": project, "db": db, "elsewhere": elsewhere}


def _texts(source) -> dict[str, str]:
    """Per solver: the text the run reads, or ``"default"`` for FlexTool's packaged template."""
    files = read_solver_option_files(source)
    return {s: (_DEFAULT if files.origins[s].startswith("FlexTool default") else files.texts[s]) for s in files.texts}


def _gui_source(project: Path, cwd: Path):
    """The source a FlexTool window run of *project* resolves (the flags the GUI sends, no environment)."""
    folders = tuple(str(f) for f in gui_solver_config_dirs(project, environ={}, cwd=cwd))
    options = {**RunOptions.from_env({}).resolved, "solver_config_dir": Resolved(folders, "cli")}
    return resolve_solver_config_source(run_options=RunOptions(resolved=options), cwd=cwd)


def _terminal_source(ws: dict):
    """A terminal run on the project's database, no flag, started from another folder."""
    return resolve_solver_config_source(environ={}, input_db_url=f"sqlite:///{ws['db']}", cwd=ws["elsewhere"])


def test_case_a_project_file_then_workspace_then_default(ws: dict) -> None:
    own = ws["project"] / "solver_config"
    own.mkdir()
    (own / "gurobi.opt").write_text("Threads 2\n")
    expected = {"highs": "mip_rel_gap = 0.02\n", "gurobi": "Threads 2\n", "cplex": _template("cplex"),
                "xpress": _DEFAULT, "copt": _DEFAULT}
    gui = _gui_source(ws["project"], cwd=ws["root"])
    assert _texts(gui) == expected
    assert gui.describe() == (
        f"{own} (--solver-config-dir): gurobi.opt; {ws['cfg']} (--solver-config-dir): highs.opt, cplex.opt; "
        "FlexTool defaults: xpress.opt, copt.opt"
    )
    terminal = _terminal_source(ws)
    assert _texts(terminal) == expected
    assert [(folder.resolve(), reason) for folder, reason in terminal.layers] == [
        (own.resolve(), "project of the input database"), (ws["cfg"].resolve(), "workspace of the input database")]


def test_case_a_variant_project_template_does_not_hide_the_workspace(ws: dict) -> None:
    own = ws["project"] / "solver_config"
    own.mkdir()
    (own / "gurobi.opt").write_text(_template("gurobi"), encoding="utf-8")
    for source in (_gui_source(ws["project"], cwd=ws["root"]), _terminal_source(ws)):
        texts = _texts(source)
        assert texts["gurobi"] == "Threads 4\n"
        assert texts["highs"] == "mip_rel_gap = 0.02\n"
    assert "skipped" in _gui_source(ws["project"], cwd=ws["root"]).describe()


def test_case_b_project_without_files(ws: dict, caplog: pytest.LogCaptureFixture) -> None:
    expected = {"highs": "mip_rel_gap = 0.02\n", "gurobi": "Threads 4\n", "cplex": _template("cplex"),
                "xpress": _DEFAULT, "copt": _DEFAULT}
    # (i) no project folder: the GUI passes the workspace's folder only; the header is today's one-folder line.
    gui = _gui_source(ws["project"], cwd=ws["root"])
    assert gui.layers == ((ws["cfg"], "--solver-config-dir"),)
    assert gui.describe() == f"{ws['cfg']} (--solver-config-dir)"
    assert _texts(gui) == expected
    assert _texts(_terminal_source(ws)) == expected
    # (ii) an empty project folder: same files, the header lists only the workspace's folder and the defaults.
    (ws["project"] / "solver_config").mkdir()
    gui = _gui_source(ws["project"], cwd=ws["root"])
    assert _texts(gui) == expected
    assert gui.describe() == (f"{ws['cfg']} (--solver-config-dir): highs.opt, gurobi.opt, cplex.opt; "
                              "FlexTool defaults: xpress.opt, copt.opt")
    with caplog.at_level(logging.WARNING):
        snapshot_solver_option_files(gui, ws["root"] / "work")
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], caplog.text
    assert _texts(_terminal_source(ws)) == expected


def test_case_c_explicit_ladder_is_closed(ws: dict, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "highs.opt").write_text("threads=1\n")
    (b / "highs.opt").write_text("threads=2\n")
    (b / "cplex.opt").write_text("threads 3\n")
    (ws["elsewhere"] / "solver_config").mkdir()
    (ws["elsewhere"] / "solver_config" / "gurobi.opt").write_text("Threads 9\n")
    (ws["project"] / "solver_config").mkdir()
    (ws["project"] / "solver_config" / "gurobi.opt").write_text("Threads 8\n")
    env = {"FLEXTOOL_SOLVER_CONFIG_DIR": os.pathsep.join([str(a), str(b)])}
    source = resolve_solver_config_source(environ=env, input_db_url=f"sqlite:///{ws['db']}", cwd=ws["elsewhere"])
    assert _texts(source) == {"highs": "threads=1\n", "cplex": "threads 3\n", "gurobi": _DEFAULT,
                              "xpress": _DEFAULT, "copt": _DEFAULT}
    # /a missing: one warning naming /a; highs.opt then comes from /b, the rest unchanged.
    (a / "highs.opt").unlink()
    a.rmdir()
    source = resolve_solver_config_source(environ=env, input_db_url=f"sqlite:///{ws['db']}", cwd=ws["elsewhere"])
    with caplog.at_level(logging.WARNING):
        files = snapshot_solver_option_files(source, tmp_path / "work")
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and str(a) in warnings[0], warnings
    assert files.texts["highs"] == "threads=2\n" and files.texts["cplex"] == "threads 3\n"
    assert files.origins["gurobi"].startswith("FlexTool default")


def test_split_drops_empty_entries_and_repeats() -> None:
    assert split_solver_config_dirs(["/a", "", " ", "/b", "/a"]) == [Path("/a"), Path("/b")]
    assert split_solver_config_dirs(None) == [] and split_solver_config_dirs("") == []


@pytest.mark.skipif(sys.platform == "win32", reason="':' cannot be in a Windows folder name")
def test_one_folder_is_never_split(tmp_path: Path) -> None:
    """A folder name may hold the path separator: one folder given (an argument, one flag) stays one folder."""
    odd = tmp_path / f"x{os.pathsep}y"
    assert split_solver_config_dirs(str(odd)) == [odd]
    odd.mkdir()
    (odd / "highs.opt").write_text("threads=3\n")
    source = resolve_solver_config_source(explicit=str(odd), environ={}, cwd=tmp_path)
    assert read_solver_option_files(source).texts["highs"] == "threads=3\n"


def test_projects_folder_name_case_is_ignored_where_the_system_ignores_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On Windows ``Projects\\P\\input_sources\\db.sqlite`` is a GUI project's database (the resolved path keeps
    the case on disk).  Stand in for Windows' file-name comparison; on Linux and macOS names stay exact."""
    from flextool import _solver_config_source as source_module

    root = tmp_path / "ws"
    (root / "solver_config").mkdir(parents=True)
    (root / "solver_config" / "highs.opt").write_text("mip_rel_gap = 0.02\n")
    project = root / "Projects" / "P"
    (project / "Input_Sources").mkdir(parents=True)
    (project / "solver_config").mkdir()
    (project / "solver_config" / "gurobi.opt").write_text(_template("gurobi"), encoding="utf-8")
    (project / "solver_config" / "copt.opt").write_text("Threads 1\n")
    db = project / "Input_Sources" / "db.sqlite"
    db.write_bytes(b"")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    def terminal():
        return resolve_solver_config_source(environ={}, input_db_url=f"sqlite:///{db}", cwd=elsewhere)

    if sys.platform != "win32":  # Windows itself compares file names ignoring case
        assert not source_module.is_project_solver_config(project / "solver_config")
        assert terminal().layers == ()
    monkeypatch.setattr(source_module, "_normcase", str.lower)
    assert source_module.is_project_solver_config(project / "solver_config")
    assert [(folder.resolve(), reason) for folder, reason in terminal().layers] == [
        ((project / "solver_config").resolve(), "project of the input database"),
        ((root / "solver_config").resolve(), "workspace of the input database")]
    texts = _texts(terminal())
    assert texts["highs"] == "mip_rel_gap = 0.02\n" and texts["copt"] == "Threads 1\n"
    assert texts["gurobi"] == _DEFAULT   # the project's unedited template is skipped
