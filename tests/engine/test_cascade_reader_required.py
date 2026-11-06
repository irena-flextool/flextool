"""The cascade never runs without its DB reader.

Every sub-solve loads its parameters through one ``SpineDbReader`` that
``_drive_cascade`` builds for the run's scenario.  Two ways used to end in
a run without that reader, which finished normally but built an LP without
the DB overrides:

* the reader's construction failed (DB locked, version mismatch, a bug) and
  the error was swallowed;
* no scenario was given, so no reader was built at all.

Both now stop the run with a :class:`FlexToolConfigError` that names the
database and the scenario.  The CLI stops with exit code 1 and a plain
message when it cannot resolve a scenario.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.input._spinedb_reader import SpineDbReader
from flextool.engine.solve._orchestration import run_chain_from_db

FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]


def test_reader_construction_failure_raises(test_db_url, test_solver_config_dir, tmp_path, monkeypatch) -> None:
    """A SpineDbReader that cannot be built stops the run; the original error is the cause."""

    def _fail(self, *args, **kwargs):
        raise RuntimeError("injected reader failure")

    monkeypatch.setattr(SpineDbReader, "__init__", _fail)
    with pytest.raises(FlexToolConfigError, match="Could not open the input database") as excinfo:
        run_chain_from_db(
            test_db_url, "base", work_folder=tmp_path / "work",
            solver_config_dir=test_solver_config_dir,
        )
    msg = str(excinfo.value)
    assert "'base'" in msg, msg
    assert test_db_url in msg, msg
    assert "injected reader failure" in msg, msg
    assert isinstance(excinfo.value.__cause__, RuntimeError)


@pytest.mark.parametrize("scenario", [None, ""])
def test_missing_scenario_raises(test_db_url, test_solver_config_dir, tmp_path, scenario) -> None:
    """``run_chain_from_db`` without a scenario raises before it touches the DB."""
    with pytest.raises(FlexToolConfigError, match="needs scenario_name"):
        run_chain_from_db(
            test_db_url, scenario, work_folder=tmp_path / "work",
            solver_config_dir=test_solver_config_dir,
        )


def test_cli_without_scenario_exits_with_message(test_db_url, test_solver_config_dir, tmp_path) -> None:
    """The CLI on a DB URL without a scenario filter and without --scenario-name stops with exit code 1."""
    work = tmp_path / "work"
    work.mkdir()
    result = subprocess.run(
        [
            sys.executable, "-m", "flextool.cli.cmd_run_flextool", test_db_url,
            "--work-folder", str(work),
            "--output-location", str(tmp_path),
            "--write-methods", "csv",
        ],
        capture_output=True,
        text=True,
        cwd=str(FLEXTOOL_ROOT),
        timeout=300,
        env={**os.environ, "FLEXTOOL_SOLVER_CONFIG_DIR": str(test_solver_config_dir)},
    )
    out = result.stdout + result.stderr
    assert result.returncode == 1, f"rc={result.returncode}\n{out}"
    assert "No scenario to run" in out, out
    assert "--scenario-name" in out, out
