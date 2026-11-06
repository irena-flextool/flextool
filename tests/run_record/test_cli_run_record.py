"""An engine run records itself: ``_run_status.json`` and ``_run.log`` in its ``output_parquet/<subdir>/`` folder.

The input is the ``base`` scenario of ``tests.json`` with one more alternative.  ``base`` has a single balance
node, ``west``, with demand (negative inflow, 589 MW in the first hour) and no unit or connection, so only the
node's balance slack can meet it.  The extra alternative ``no_slack`` sets ``west.penalty_method = off``: the node
balance becomes a hard equality ``0 = demand``, which no solution satisfies, so the LP is infeasible.  The run is
started as ``python -m flextool.cli.cmd_run_flextool`` with no pipe owner, so the engine writes the log itself
(``log_capture: "python"``).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import spinedb_api as api

from flextool.engine.solve._acceptance import _FAILURE_MESSAGES
from flextool.run_record import RUN_LOG_FILE, RUN_LOG_OWNER_ENV, read_status_file
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
    env = dict(os.environ, FLEXTOOL_THREADS="1")
    env.pop(RUN_LOG_OWNER_ENV, None)
    env.pop("POLARS_MAX_THREADS", None)
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
    assert last["model_status"] in ("kInfeasible", "kUnboundedOrInfeasible")
    assert status["verdict"] == "infeasible"
    acceptance_message = f"solve for {SOLVE} {_FAILURE_MESSAGES[last['model_status']]}"
    assert last["message"] == acceptance_message
    assert status["error"]["message"] == acceptance_message

    assert not list(folder.glob("*.parquet"))
    assert status["outputs"]["parquet_state"] == "none"

    log_lines = (folder / RUN_LOG_FILE).read_text(encoding="utf-8").splitlines()
    assert log_lines[0] == f"# FlexTool run {status['run_id']} started {status['started_at']}"
    assert acceptance_message in log_lines
    if last["log_anchor"] is not None:
        assert last["log_anchor"] in log_lines
