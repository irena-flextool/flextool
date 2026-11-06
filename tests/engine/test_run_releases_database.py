"""A FlexTool run leaves no open handle to its input database or to the databases it writes.

Windows cannot delete or replace a file that is still open, so a database a run left open cannot be swapped or
removed by the caller after an in-process run (WinError 32).  ``with DatabaseMapping(...)`` closes only the
session; the engine keeps the SQLite file open until it is disposed.  Each test lists the process's open files
under the test folder right after the run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from flextool.update_flextool.db_migration import migrate_database
from tests.db_utils import json_to_db

pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="open files are listed from /proc"),
]

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "stoch_two_period.json"


def _open_files_under(folder: Path) -> list[str]:
    out = []
    for fd in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target.startswith(str(folder)):
            out.append(target)
    return out


@pytest.fixture
def input_db(tmp_path) -> str:
    url = json_to_db(FIXTURE, tmp_path / "input.sqlite")
    migrate_database(url)
    assert _open_files_under(tmp_path) == [], "building the input database"
    return url


@pytest.mark.parametrize("scenario", ["det", "stoch"])
def test_run_chain_from_db_releases_input_database(input_db, test_solver_config_dir, tmp_path, scenario) -> None:
    from flextool.engine import run_chain_from_db

    run_chain_from_db(
        input_db_url=input_db, scenario_name=scenario, work_folder=tmp_path / "work",
        solver_config_dir=test_solver_config_dir, warm=False,
    )
    assert _open_files_under(tmp_path) == []


def test_cli_run_releases_all_databases(input_db, tmp_path, monkeypatch) -> None:
    # The in-process command line run: input database, output-info database and the spinedb results database.
    from flextool.cli import cmd_run_flextool

    monkeypatch.setattr(sys, "argv", [
        "cmd_run_flextool", input_db, f"sqlite:///{tmp_path / 'info.sqlite'}",
        "--scenario-name", "det", "--output-location", str(tmp_path / "out"), "--work-folder", str(tmp_path / "work"),
        "--write-methods", "csv", "spinedb", "--results-db-url", f"sqlite:///{tmp_path / 'results.sqlite'}",
    ])
    try:
        cmd_run_flextool.main()
    except SystemExit as exc:
        assert exc.code in (None, 0)
    assert (tmp_path / "info.sqlite").is_file() and (tmp_path / "results.sqlite").is_file()
    assert _open_files_under(tmp_path) == []
