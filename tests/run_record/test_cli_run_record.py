"""An engine run records itself: ``_run_status.json`` and ``_run.log`` in its ``output_parquet/<subdir>/`` folder.

The input is the ``base`` scenario of ``tests.json``: one solve, ``y2020_2day_dispatch`` (``model.solves`` of
alternative ``init``, ``solve_mode = single_solve``), and a single balance node, ``west``, with demand (negative
inflow, 589 MW in the first hour) and no unit or connection, so only the node's balance slack can meet it.

* The successful run starts under the ``flextool-run`` supervisor (``python -m flextool_run_supervisor``), which
  owns the output pipe and writes ``_run.log`` (``log_capture: "parent"``, launcher ``flextool-run``).
* The infeasible run adds the alternative ``no_slack``, which sets ``west.penalty_method = off``: the node balance
  becomes a hard equality ``0 = demand``, which no solution satisfies.  The model then has no variable at all
  (only the 48 balance rows): HiGHS reports such an LP as ``kModelEmpty`` without reading its rows, and the
  acceptance check finds the rows that cannot hold and reports the solve infeasible.  It is started as ``python -m
  flextool.cli.cmd_run_flextool`` with no pipe owner, so the engine writes the log itself (``log_capture:
  "python"``).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import spinedb_api as api

from flextool.run_record import MARKER, RUN_LOG_FILE, RUN_LOG_OWNER_ENV, read_status_file
from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.db_handle import open_database_mapping
from flextool.update_flextool.db_migration import migrate_database
from tests.db_utils import json_to_db

pytestmark = pytest.mark.solver

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "tests.json"
SCENARIO = "base_no_slack"
SOLVE = "y2020_2day_dispatch"


@pytest.fixture
def base_db(tmp_path) -> str:
    url = json_to_db(FIXTURE, tmp_path / "input.sqlite")
    migrate_database(url)
    return url


def _engine_env() -> dict[str, str]:
    # PYTHONIOENCODING: the supervisor echoes in its console's encoding (cp1252 on a Windows runner: CI of PR
    # #362), and the tests read its output as UTF-8.
    env = dict(os.environ, FLEXTOOL_THREADS="1", PYTHONIOENCODING="utf-8")
    env.pop(RUN_LOG_OWNER_ENV, None)
    env.pop("POLARS_MAX_THREADS", None)
    return env


def test_successful_run_writes_both_files(base_db, tmp_path):
    result = subprocess.run(
        [
            sys.executable, "-m", "flextool_run_supervisor", base_db,
            "--scenario-name", "base",
            "--work-folder", str(tmp_path / "work"),
            "--output-location", str(tmp_path / "out"),
            "--write-methods", "parquet",
        ],
        capture_output=True, text=True, encoding="utf-8", cwd=str(REPO), env=_engine_env(), timeout=600,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    folder = tmp_path / "out" / "output_parquet" / "base"
    status = read_status_file(folder)
    assert status is not None, result.stdout
    assert status["state"] == "succeeded" and status["exit_code"] == 0 and status["error"] is None
    assert status["verdict"] == "optimal"
    assert status["launcher"] == "flextool-run" and status["log_capture"] == "parent"
    assert [s["name"] for s in status["solves"]] == [SOLVE]
    assert [(s["state"], s["acceptance"]) for s in status["solves"]] == [("accepted", "optimal")]
    assert status["outputs"]["parquet_state"] == "written"
    # Item 234: the option files the run read, with a hash of each (the snapshot is in the work folder).
    solver_config = status["solver_config"]
    assert solver_config["snapshot_dir"] == str((tmp_path / "work" / "solve_data" / "solver_config").resolve())
    assert set(solver_config["files"]) == {"highs.opt", "gurobi.opt", "cplex.opt", "xpress.opt", "copt.opt"}
    assert all(len(f["sha256"]) == 64 for f in solver_config["files"].values())
    # write_outputs emptied the folder before writing the parquet; the run record stayed.
    assert list(folder.glob("*.parquet"))
    assert (folder / RUN_LOG_FILE).is_file()

    log_lines = (folder / RUN_LOG_FILE).read_text(encoding="utf-8").splitlines()
    assert log_lines[0] == f"# FlexTool run {status['run_id']} started {status['started_at']}"
    assert not any(line.startswith(MARKER) for line in log_lines)
    assert MARKER not in result.stdout
    # Native HiGHS output reaches the log only through the pipe owner.
    assert any("Running HiGHS" in line for line in log_lines)


@pytest.fixture
def infeasible_db(tmp_path) -> str:
    url = json_to_db(FIXTURE, tmp_path / "input.sqlite")
    migrate_database(url)
    with open_database_mapping(url) as db:
        db.add_alternative(name="no_slack")
        value, type_ = api.to_database("off")
        db.add_parameter_value(
            entity_class_name="node", entity_byname=("west",), parameter_definition_name="penalty_method",
            alternative_name="no_slack", value=value, type=type_,
        )
        db.add_scenario(name=SCENARIO)
        for rank, alternative in enumerate(("init", "west", "no_slack"), start=1):
            db.add_scenario_alternative(scenario_name=SCENARIO, alternative_name=alternative, rank=rank)
        db.commit_session("infeasible variant of base")
    return url


def test_infeasible_run_writes_both_files(infeasible_db, tmp_path):
    env = _engine_env()
    result = subprocess.run(
        [
            sys.executable, "-m", "flextool.cli.cmd_run_flextool", infeasible_db,
            "--scenario-name", SCENARIO,
            "--work-folder", str(tmp_path / "work"),
            "--output-location", str(tmp_path / "out"),
            "--write-methods", "parquet",
        ],
        capture_output=True, text=True, cwd=str(REPO), env=env, timeout=600,
    )
    assert result.returncode == 1, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    folder = tmp_path / "out" / "output_parquet" / SCENARIO
    status = read_status_file(folder)
    assert status is not None, result.stdout
    assert status["state"] == "failed" and status["exit_code"] == 1
    assert status["launcher"] == "direct" and status["log_capture"] == "python"
    assert status["solver"] == "highs"
    assert status["input"]["scenario"] == SCENARIO
    assert status["input"]["alternatives"] == ["init", "west", "no_slack"]
    assert status["input"]["db_version"] == FLEXTOOL_DB_VERSION
    # FLEXTOOL_THREADS=1 without POLARS_MAX_THREADS: polars gets the budget, HiGHS threads=1.
    assert status["threads"] == {"budget": 1, "source": "env", "polars": 1, "polars_source": "budget", "highs": 1}
    assert status["options_resolved"]["threads"]["source"] == "env"

    last = status["solves"][-1]
    assert last["name"] == SOLVE and last["kind"] == "lp"
    assert last["state"] == "rejected" and last["acceptance"] == "rejected"
    assert last["model_status"] == "kInfeasible"
    assert status["verdict"] == "infeasible"
    acceptance_message = last["message"]
    assert acceptance_message.startswith(
        f"solve for {SOLVE} is infeasible: the model has no variables, and 48 of its 48 constraints cannot hold "
        "with nothing in them: nodeBalance_eq[west,p2020,t0001] (0 not in [589, 589]), ")
    assert status["error"]["message"] == acceptance_message

    assert not list(folder.glob("*.parquet"))
    assert status["outputs"]["parquet_state"] == "none"

    log_lines = (folder / RUN_LOG_FILE).read_text(encoding="utf-8").splitlines()
    assert log_lines[0] == f"# FlexTool run {status['run_id']} started {status['started_at']}"
    assert acceptance_message in log_lines
    if last["log_anchor"] is not None:
        assert last["log_anchor"] in log_lines
