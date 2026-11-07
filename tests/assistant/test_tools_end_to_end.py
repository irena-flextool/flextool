"""The tools on a real run: ``flextool-run`` solves scenario ``coal`` of ``tests/fixtures/tests.json`` into a project
folder as the FlexTool window lays it out (``output_parquet/coal_1``), then the tools read that run.

* ``list_runs``: the run succeeded and its results are usable;
* ``query_result`` of ``annualized_costs_d_p``: the unit is worded as the default convention (``M CUR/a``);
* ``get_run_log``: the run's one solve was accepted as optimal.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from flextool.assistant.context import make_context
from flextool.assistant.registry import call_tool
from flextool.run_record import RUN_LOG_OWNER_ENV

from tests.assistant.conftest import make_project

pytestmark = pytest.mark.solver

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def solved_project(tmp_path_factory: pytest.TempPathFactory, tests_db: Path) -> Path:
    project = make_project(tmp_path_factory.mktemp("end_to_end"), tests_db)
    env = dict(os.environ, FLEXTOOL_THREADS="1")
    env.pop(RUN_LOG_OWNER_ENV, None)
    db_url = f"sqlite:///{(project / 'input_sources' / 'tests.sqlite').as_posix()}"
    result = subprocess.run(
        [sys.executable, "-m", "flextool_run_supervisor", db_url, "--scenario-name", "coal",
         "--work-folder", str(project / "work"), "--output-location", str(project),
         "--output-subdir", "coal_1", "--write-methods", "parquet"],
        capture_output=True, text=True, encoding="utf-8", cwd=str(REPO), env=env, timeout=600,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"
    return project


def _call(ctx, tool: str, **args) -> dict:
    out = call_tool(ctx, tool, args)
    assert out.ok, out.message
    return json.loads(out.text)


def test_tools_read_a_real_run(solved_project: Path) -> None:
    ctx = make_context(solved_project)

    runs = {r["folder"]: r for r in _call(ctx, "list_runs")["runs"]}
    coal = runs["coal_1"]
    assert (coal["state"], coal["results_usable"], coal["unusable_code"]) == ("succeeded", True, None)

    costs = _call(ctx, "query_result", run="coal_1", table="annualized_costs_d_p")
    assert costs["unit"] == "M CUR/a (default convention)"
    assert costs["rows"]

    log = _call(ctx, "get_run_log", run="coal_1")
    status = log["status"]
    assert status["state"] == "succeeded"
    assert [(s["state"], s["acceptance"]) for s in status["solves"]] == [("accepted", "optimal")]
