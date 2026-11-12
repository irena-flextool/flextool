"""Where a run takes its solver option files from, and the per-run snapshot of them (item 230).

:func:`flextool._solver_config_source.resolve_solver_config_source` picks one folder: explicit argument >
``--solver-config-dir`` > ``FLEXTOOL_SOLVER_CONFIG_DIR`` > the input database's layout (next to the database,
then the workspace of a GUI project database; bounded, never further up) > the Spine Toolbox project folder > the
current folder > FlexTool's packaged defaults.  :func:`snapshot_solver_option_files` reads the files once and
writes them afresh to ``<work_folder>/solve_data/solver_config/`` every run.  Expected values are derived from
those rules by hand (design ``specs/gurobi_options_fix_design.md`` §2.2 / §3 tests 10-13, critique §5 a-b).
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
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
    SHIPPED_TEMPLATE_SHA256,
    SOLVER_NAMES,
    is_template_copy,
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


#: ``flextool/solver_config/gurobi.opt.template`` as shipped in 4.3.0 (``git show v4.3.0:...``), the file 4.3
#: copied into the folder a run started from.
_GUROBI_TEMPLATE_4_3 = (
    "# Gurobi baseline parameter file (FlexTool default).\n"
    "#\n"
    "# Format: Gurobi .prm \u2014 one \"<ParamName> <value>\" per line.\n"
    "# Lines starting with \"#\" are comments.  This file is consumed via\n"
    "#     gurobi_cl ReadParams=<this-file> ResultFile=<sol> <mps>\n"
    "# by FlexTool's commercial-solver subprocess driver\n"
    "# (flextool/engine/solve/_subprocess_solve.py).\n"
    "#\n"
    "# Any value set here is overridden line-by-line by the scenario's\n"
    "# ``solver_options`` Map (and by the translated convenience knobs\n"
    "# ``solver_time_limit`` / ``solver_mip_gap`` / ``solver_threads``).\n"
    "#\n"
    "# Reference: https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html\n"
    "#\n"
    "# Conservative defaults \u2014 edit to suit your model.\n"
    "Threads 1\n"
    "MIPGap 0.002\n"
    "LogToConsole 1\n"
)


def _plant_4_3_copy(folder: Path) -> Path:
    """What 4.3 left in ``<folder>/solver_config``: the 4.3 gurobi template (written with Windows line ends) and
    the current highs template."""
    planted = folder / "solver_config"
    planted.mkdir(parents=True, exist_ok=True)
    (planted / "gurobi.opt").write_bytes(_GUROBI_TEMPLATE_4_3.replace("\n", "\r\n").encode("utf-8"))
    (planted / "highs.opt").write_text(_template("highs"), encoding="utf-8")
    return planted


def test_the_4_3_gurobi_template_is_a_known_shipped_version() -> None:
    """The embedded 4.3 text is the file whose hash ``git show v4.3.0:...`` gives (checked by hand once)."""
    digest = hashlib.sha256(_GUROBI_TEMPLATE_4_3.encode("utf-8")).hexdigest()
    assert digest == "7911b914b4c8f1dc4eb335be410251111c12d762dcecb18d4ac19cb2a912c690"
    assert digest in SHIPPED_TEMPLATE_SHA256["gurobi"]


def test_a_template_copy_in_the_current_folder_is_skipped_for_the_next_tier(layout, caplog) -> None:
    """Review M1: 4.3 planted the templates in ``runs/solver_config``; the Toolbox project folder holds the
    user's ``gurobi.opt`` (Threads 7).  The run takes the user's file and warns, naming the planted folder."""
    runs = layout["tmp"] / "runs"
    planted = _plant_4_3_copy(runs)
    assert is_template_copy(planted)
    folder_file = layout["ws"] / "templates" / "project_folder.txt"
    source = _resolve(input_db_url="mysql://host/db", project_folder_file=folder_file, cwd=runs)
    assert (source.dir, source.reason, source.skipped) == (layout["ws"] / "solver_config", REASON_TOOLBOX, ())

    # Without the Toolbox folder the planted copy is the only candidate: skipped, the defaults are used.
    source = _resolve(input_db_url=f"sqlite:///{layout['tmp'] / 'models' / 'uc.sqlite'}", cwd=runs)
    assert source.dir is None and source.skipped == (planted,)
    with caplog.at_level(logging.WARNING):
        files = snapshot_solver_option_files(source, runs, _LOGGER)
    assert files.text("gurobi") == _template("gurobi")
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, warnings
    assert warnings[0].startswith(f"Not using {planted}: ") and "FlexTool's default options" in warnings[0]
    assert "unedited FlexTool templates" in source.describe()


def test_a_template_copy_next_to_the_database_run_from_there_is_skipped(layout, caplog) -> None:
    """Run from the database's own folder: the planted copy is also the tier-3 folder; it is still skipped."""
    models = layout["tmp"] / "models"
    planted = _plant_4_3_copy(models)
    db = models / "uc.sqlite"
    db.write_bytes(b"")
    folder_file = layout["ws"] / "templates" / "project_folder.txt"
    source = _resolve(input_db_url=f"sqlite:///{db}", project_folder_file=folder_file, cwd=models)
    assert (source.dir, source.reason, source.skipped) == (layout["ws"] / "solver_config", REASON_TOOLBOX, (planted,))
    with caplog.at_level(logging.WARNING):
        files = snapshot_solver_option_files(source, models, _LOGGER)
    assert files.text("gurobi") == "Threads 7\n"
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and str(layout["ws"] / "solver_config") in warnings[0], warnings


def test_an_edited_file_keeps_the_current_folder(layout) -> None:
    """One edited file is a choice of options: the folder is used (only a pure template copy is skipped)."""
    runs = layout["tmp"] / "runs"
    planted = _plant_4_3_copy(runs)
    (planted / "highs.opt").write_text(_template("highs") + "threads = 4\n", encoding="utf-8")
    assert not is_template_copy(planted)
    source = _resolve(input_db_url="mysql://host/db", cwd=runs)
    assert (source.dir, source.reason, source.skipped) == (planted, REASON_CWD, ())


def test_shipped_template_hashes_cover_git_history() -> None:
    """Every version of ``flextool/solver_config/<solver>.opt.template`` in the git history is in
    :data:`SHIPPED_TEMPLATE_SHA256` (skipped outside a git checkout)."""
    repo = Path(__file__).resolve().parents[2]
    if shutil.which("git") is None or not (repo / ".git").exists():
        pytest.skip("not a git checkout")
    for solver in SOLVER_NAMES:
        commits = subprocess.run(
            ["git", "log", "--all", "--format=%H", "--", f"flextool*/solver_config/{solver}.opt.template"],
            cwd=repo, capture_output=True, text=True, check=True,
        ).stdout.split()
        assert commits, solver
        seen: set[str] = set()
        for commit in commits:
            tree = subprocess.run(
                ["git", "ls-tree", "-r", commit], cwd=repo, capture_output=True, text=True, check=True,
            ).stdout.splitlines()
            for line in tree:
                meta, path = line.split("\t", 1)
                if path.endswith(f"solver_config/{solver}.opt.template"):
                    seen.add(meta.split()[2])
        for blob in seen:
            data = subprocess.run(["git", "cat-file", "-p", blob], cwd=repo, capture_output=True, check=True).stdout
            digest = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
            assert digest in SHIPPED_TEMPLATE_SHA256[solver], (solver, blob)
