"""Where a run takes its solver option files from, and the per-run snapshot of them (item 230).

:func:`flextool._solver_config_source.resolve_solver_config_source` picks one folder: explicit argument >
``--solver-config-dir`` > ``FLEXTOOL_SOLVER_CONFIG_DIR`` > the input database's layout (next to the database,
then the workspace of a GUI project database; bounded, never further up) > the Spine Toolbox project folder > the
current folder > FlexTool's packaged defaults.  :func:`snapshot_solver_option_files` reads the files once and
writes them afresh to ``<work_folder>/solve_data/solver_config/`` every run.  Expected values are derived from
those rules by hand (design ``specs/gurobi_options_fix_design.md`` §2.2 / §3 tests 10-13, critique §5 a-b).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from flextool._resources import package_data_path
from flextool._solver_config_source import (
    REASON_ARGUMENT,
    REASON_CWD,
    REASON_DB_FOLDER,
    REASON_DB_WORKSPACE,
    REASON_ENV,
    REASON_FLAG,
    REASON_TOOLBOX,
    SOLVER_NAMES,
    resolve_solver_config_source,
    snapshot_solver_option_files,
)
from flextool.cli.cmd_run_flextool import build_parser
from flextool.run_options import RunOptions

_LOGGER = logging.getLogger("test_solver_config_source")


def _template(solver: str) -> str:
    return Path(package_data_path(f"solver_config/{solver}.opt.template")).read_text(encoding="utf-8")


@pytest.fixture
def layout(tmp_path: Path) -> dict[str, Path]:
    """``ws/solver_config/gurobi.opt`` (Threads 7), a GUI project database ``ws/projects/P/input_sources/db.sqlite``
    and a current folder ``other/`` with its own ``solver_config/gurobi.opt`` (Threads 1)."""
    ws = tmp_path / "ws"
    (ws / "solver_config").mkdir(parents=True)
    (ws / "solver_config" / "gurobi.opt").write_text("Threads 7\n")
    db = ws / "projects" / "P" / "input_sources" / "db.sqlite"
    db.parent.mkdir(parents=True)
    db.write_bytes(b"")
    other = tmp_path / "other"
    (other / "solver_config").mkdir(parents=True)
    (other / "solver_config" / "gurobi.opt").write_text("Threads 1\n")
    empty = tmp_path / "empty"
    empty.mkdir()
    return {"ws": ws, "db": db, "other": other, "empty": empty, "tmp": tmp_path}


def _resolve(*, environ=None, flags=(), **kwargs):
    if flags:
        options = RunOptions.from_cli(build_parser().parse_args(["in.sqlite", *flags]), environ=environ or {})
    else:
        options = RunOptions.from_env(environ or {})
    return resolve_solver_config_source(run_options=options, **kwargs)


def test_explicit_flag_and_environment_come_first(layout) -> None:
    url = f"sqlite:///{layout['db']}"
    env = {"FLEXTOOL_SOLVER_CONFIG_DIR": str(layout["tmp"] / "from_env")}
    flag = ["--solver-config-dir", str(layout["tmp"] / "from_flag")]

    got = _resolve(environ=env, flags=flag, explicit=layout["tmp"] / "arg", input_db_url=url, cwd=layout["other"])
    assert (got.dir, got.reason) == (layout["tmp"] / "arg", REASON_ARGUMENT)
    got = _resolve(environ=env, flags=flag, input_db_url=url, cwd=layout["other"])
    assert (got.dir, got.reason) == (layout["tmp"] / "from_flag", REASON_FLAG)
    got = _resolve(environ=env, input_db_url=url, cwd=layout["other"])
    assert (got.dir, got.reason) == (layout["tmp"] / "from_env", REASON_ENV)


def test_gui_project_database_finds_the_workspace_not_the_current_folder(layout) -> None:
    got = _resolve(input_db_url=f"sqlite:///{layout['db']}", cwd=layout["other"])
    assert (got.dir, got.reason) == (layout["ws"] / "solver_config", REASON_DB_WORKSPACE)


def test_toolbox_url_with_a_filter_query_finds_the_folder_next_to_the_database(layout) -> None:
    """Spine Toolbox appends ``?spinedbfilter=...`` to the Data Store URL and runs in a temporary work folder."""
    db = layout["ws"] / "input_data.sqlite"
    db.write_bytes(b"")
    url = f"sqlite:///{db}?spinedbfilter=%7B%22type%22%3A+%22scenario_filter%22%7D"
    got = _resolve(input_db_url=url, cwd=layout["empty"])
    assert (got.dir, got.reason) == (layout["ws"] / "solver_config", REASON_DB_FOLDER)


def test_the_database_search_does_not_climb_past_the_layout(layout) -> None:
    """A stray ``solver_config`` two levels above a database that is not in a GUI project is not used."""
    stray = layout["tmp"] / "top"
    (stray / "solver_config").mkdir(parents=True)
    (stray / "solver_config" / "gurobi.opt").write_text("Threads 5\n")
    db = stray / "a" / "input_sources" / "db.sqlite"   # input_sources/, but no projects/ above it
    db.parent.mkdir(parents=True)
    db.write_bytes(b"")
    got = _resolve(input_db_url=f"sqlite:///{db}", cwd=layout["empty"])
    assert got.dir is None
    assert got.tried == (db.parent / "solver_config", layout["empty"] / "solver_config")


def test_server_database_uses_toolbox_project_then_current_folder(layout) -> None:
    url = "mysql://host/db"
    folder_file = layout["ws"] / "templates" / "project_folder.txt"
    got = _resolve(input_db_url=url, project_folder_file=folder_file, cwd=layout["other"])
    assert (got.dir, got.reason) == (layout["ws"] / "solver_config", REASON_TOOLBOX)
    got = _resolve(input_db_url=url, cwd=layout["other"])
    assert (got.dir, got.reason) == (layout["other"] / "solver_config", REASON_CWD)
    got = _resolve(input_db_url=url, cwd=layout["empty"])
    assert got.dir is None and got.tried == (layout["empty"] / "solver_config",)


def test_a_folder_without_option_files_does_not_count(layout) -> None:
    (layout["ws"] / "solver_config" / "gurobi.opt").unlink()
    got = _resolve(input_db_url=f"sqlite:///{layout['db']}", cwd=layout["other"])
    assert (got.dir, got.reason) == (layout["other"] / "solver_config", REASON_CWD)


def test_snapshot_is_written_afresh_every_run(layout) -> None:
    source = _resolve(input_db_url=f"sqlite:///{layout['db']}", cwd=layout["other"])
    work = layout["tmp"] / "work"
    first = snapshot_solver_option_files(source, work, _LOGGER)
    assert first.snapshot_dir == work / "solve_data" / "solver_config"
    assert (first.snapshot_dir / "gurobi.opt").read_text() == "Threads 7\n"
    (layout["ws"] / "solver_config" / "gurobi.opt").write_text("Threads 9\n")
    second = snapshot_solver_option_files(source, work, _LOGGER)
    assert (second.snapshot_dir / "gurobi.opt").read_text() == "Threads 9\n"
    assert second.text("gurobi") == "Threads 9\n"
    # A solver missing in the folder gets its packaged default, and says so.
    assert second.text("highs") == _template("highs")
    assert second.origin("highs") == "FlexTool default (highs.opt.template)"
    assert second.origin("gurobi") == str(layout["ws"] / "solver_config" / "gurobi.opt")


def test_no_folder_found_uses_the_defaults_without_planting_a_copy(layout, caplog) -> None:
    """``work_folder == cwd`` and no ``solver_config`` anywhere: no ``<cwd>/solver_config`` appears, the snapshot
    holds the templates and one warning says no folder was found."""
    cwd = layout["empty"]
    source = _resolve(input_db_url="mysql://host/db", cwd=cwd)
    with caplog.at_level(logging.WARNING):
        files = snapshot_solver_option_files(source, cwd, _LOGGER)
    assert not (cwd / "solver_config").exists()
    for solver in SOLVER_NAMES:
        assert (files.snapshot_dir / f"{solver}.opt").read_text() == _template(solver)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "No solver_config folder found" in warnings[0]


def test_an_old_work_folder_copy_is_ignored_with_a_warning(layout, caplog) -> None:
    work = layout["tmp"] / "work"
    (work / "solver_config").mkdir(parents=True)
    (work / "solver_config" / "gurobi.opt").write_text("Threads 3\n")
    source = _resolve(input_db_url=f"sqlite:///{layout['db']}", cwd=layout["other"])
    with caplog.at_level(logging.WARNING):
        files = snapshot_solver_option_files(source, work, _LOGGER)
    assert files.text("gurobi") == "Threads 7\n"
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and str(work / "solver_config") in warnings[0]


def test_the_source_folder_itself_is_not_a_stale_copy(layout, caplog) -> None:
    """A terminal run from the workspace with ``work_folder == cwd``: ``<cwd>/solver_config`` is the source."""
    ws = layout["ws"]
    source = _resolve(input_db_url="mysql://host/db", cwd=ws)
    assert source.dir == ws / "solver_config"
    with caplog.at_level(logging.WARNING):
        snapshot_solver_option_files(source, ws, _LOGGER)
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
