"""The grounding project builds, runs and is usable as the questions need it (P3 test 8).

Built once for the module from ``tests/fixtures/tests.json`` and the overrides of
``tests/assistant/grounding_projects.py`` and solved with ``flextool-run`` (small 48-hour solves).  Asserted
through the assistant's own run listing and result tool: the run order, the three successful runs fresh against
the rewritten source, the investment run with invested capacity, the failed run with the period-inflow reason and
the earlier run's result files, and the marker.  The values the questions ask for are the answer key's (an
independent oracle), not this test's.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.assistant import grounding_projects as gp

pytestmark = pytest.mark.solver


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> gp.GroundingProject:
    return gp.build_grounding_project(tmp_path_factory.mktemp("grounding"))


def _runs(project: Path) -> dict[str, dict]:
    from flextool.assistant import registry
    from flextool.assistant.context import make_context

    outcome = registry.call_tool(make_context(project), "list_runs", {"detail": "full"})
    assert outcome.ok, outcome.text
    common = outcome.payload.get("same_in_every_row", {})
    return {r["folder"]: {**common, **r} for r in outcome.payload["runs"]}


def test_runs_in_order_and_the_failed_rewrite(built: gp.GroundingProject) -> None:
    assert [(r.scenario, r.build) for r in built.runs] == list(gp.RUN_ORDER)
    first, second = built.runs[0], built.runs[1]
    assert first.folder == second.folder == "demand_update_test_1"
    assert first.state == "succeeded" and second.state == "failed" and first.run_id != second.run_id
    runs = _runs(built.path)
    failed = runs["demand_update_test_1"]
    assert failed["state"] == "failed" and failed["results_usable"] is False
    assert "keyed by period" in failed["error"] and "demand_update" in failed["error"]
    folder = built.path / "output_parquet" / "demand_update_test_1"
    assert any(folder.glob("*.parquet")), "the first build's result files stay in the folder"


@pytest.mark.parametrize("folder", ["coal_1", "coal_co2_price_1", "invest_2day_test_1"])
def test_later_runs_usable_and_fresh(built: gp.GroundingProject, folder: str) -> None:
    entry = _runs(built.path)[folder]
    assert entry["state"] == "succeeded" and entry["results_usable"] is True
    assert entry["stale_vs_input"] is False
    assert entry["source_number"] == built.source_number == 1


def test_investment_run_invests(built: gp.GroundingProject) -> None:
    from flextool.assistant import registry
    from flextool.assistant.context import make_context

    outcome = registry.call_tool(make_context(built.path), "query_result",
                                 {"run": "invest_2day_test_1", "table": "unit_capacity_ed_p",
                                  "filters": {"parameter": ["invested"]}})
    assert outcome.ok, outcome.text
    assert outcome.payload["rows"] and all(row[-1] > 0 for row in outcome.payload["rows"])


def test_project_folder_and_marker(built: gp.GroundingProject) -> None:
    assert " " in built.path.name and "ä" in built.path.name
    assert built.source == built.path / "input_sources" / gp.SOURCE_NAME
    marker = json.loads((built.path / gp.MARKER_FILE).read_text(encoding="utf-8"))
    assert marker == {"format": gp.MARKER_FORMAT, "set_version": gp.SET_VERSION,
                      "fingerprint": built.fingerprint, "source_sha256": gp.sha256_file(built.source),
                      "solved": True, "runs": {r.folder: r.run_id for r in built.runs}}
    assert sorted(marker["runs"]) == ["coal_1", "coal_co2_price_1", "demand_update_test_1", "invest_2day_test_1"]
    # the folder of the failed second run keeps that run's id, which its status holds
    failed = next(r for r in built.runs if (r.scenario, r.build) == gp.FAILED_RUN)
    assert marker["runs"]["demand_update_test_1"] == failed.run_id
    assert built.fingerprint == gp.fingerprint(marker["source_sha256"], [r.run_id for r in built.runs])
    assert "input_source_numbers" in (built.path / "settings.yaml").read_text(encoding="utf-8")


def test_inputs_only_without_solve(tmp_path: Path) -> None:
    project = gp.build_grounding_project(tmp_path, solve=False)
    assert project.runs == ()
    assert not any((project.path / "output_parquet").iterdir())
    assert (project.path / gp.MARKER_FILE).is_file()
    assert project.fingerprint == gp.fingerprint(project.source_sha256, [])


def test_the_runner_accepts_the_build(built: gp.GroundingProject, evaluator) -> None:
    """C6-m7: the marker checks pass on a genuine solved build (its result folders, the source's sidecars)."""
    assert evaluator.check_marker(built.path)["solved"] is True
