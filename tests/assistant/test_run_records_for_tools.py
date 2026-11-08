"""Runs and input sources from the run record: what list_runs and list_sources report, the message for each
``unusable_code`` (only a run that succeeded is read, O1), that the tools defer to ``run_record`` for that rule,
the comparison files' state, and that no payload or message carries a local path, the host or the user name.

The run records are written by the engine's own recorder (``conftest.record_run``); the expected values are what
those runs recorded.
"""
from __future__ import annotations

import getpass
import json
import os
import shutil
import socket
import time
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from flextool import run_record
from flextool.assistant.context import make_context
from flextool.assistant.errors import ToolError, message
from flextool.assistant.registry import all_tools, call_tool
from flextool.assistant.results import _runs
from flextool.run_record import read_status_file, write_status_atomic

from tests.assistant.conftest import fail_with, make_project, record_run, succeed, write_parquet


def _call(ctx, tool: str, **args) -> dict:
    out = call_tool(ctx, tool, args)
    assert out.ok, out.message
    return json.loads(out.text)


def _installed() -> str:
    from importlib.metadata import version

    return version("flextool")


def test_list_runs_reports_each_run_from_its_record(ctx) -> None:
    out = _call(ctx, "list_runs")
    runs = {r["folder"]: r for r in out["runs"]}
    assert sorted(runs) == ["base_1", "coal_1", "coal_chp_1"]
    coal = runs["coal_1"]
    # coal_1: one optimal solve, parquet written, from input source 1 (tests.sqlite, settings.yaml), scenario coal.
    assert (coal["state"], coal["verdict"], coal["results_usable"], coal["unusable_code"]) == (
        "succeeded", "optimal", True, None)
    assert (coal["source_number"], coal["source_name"], coal["db_scenario"]) == (1, "tests.sqlite", "coal")
    assert coal["flextool_version"] == _installed() and coal["other_flextool_version"] is False
    assert coal["stale_vs_input"] is False
    chp = runs["coal_chp_1"]
    # coal_chp_1: its second run was rejected as infeasible; the error is that solve's message, with the project
    # folder redacted.
    assert (chp["state"], chp["verdict"], chp["results_usable"], chp["unusable_code"]) == (
        "failed", "infeasible", False, "failed")
    assert chp["error"] == "solve y2020 is infeasible; see <project>/work/log"
    legacy = runs["base_1"]
    assert legacy["state"] is None and legacy["results_usable"] is True and "no run record" in legacy["note"]
    assert out["comparison"]["usable"] is True
    assert [m["folder"] for m in out["comparison"]["members"]] == ["coal_1"]
    assert _call(ctx, "list_runs", scenario="coal")["total"] == 1
    # Every folder name ends in _1, the number of tests.sqlite (base_1 too, although it has no run record).
    assert _call(ctx, "list_runs", source="tests.sqlite")["total"] == 3


def test_list_sources(ctx) -> None:
    from flextool.update_flextool import FLEXTOOL_DB_VERSION

    (source,) = _call(ctx, "list_sources")["sources"]
    assert (source["name"], source["number"], source["kind"], source["path"]) == (
        "tests.sqlite", 1, "sqlite", "input_sources/tests.sqlite")
    # tests.json has 69 scenarios; the database is migrated to this FlexTool.
    assert len(source["scenarios"]) == 69 and source["scenarios_note"] is None
    assert (source["database_version"], source["version_status"]) == (FLEXTOOL_DB_VERSION, "current")
    assert source["conversion"] is None and source["behaviour_notices"] == []


def _project_with(tmp_path: Path, tests_db: Path, *runs) -> Path:
    project = make_project(tmp_path, tests_db)
    for folder, outcome in runs:
        record_run(project, folder, "coal", outcome)
    return project


def _running(rec) -> None:
    rec.solve_started("y2020", "y2020", 0, "Solve start: y2020, 1/1")


def _partial(rec) -> None:
    rec.outputs_cleared()
    rec.finish("succeeded", exit_code=0, error=None)


def _not_requested(rec) -> None:
    rec.outputs_not_requested()
    rec.finish("succeeded", exit_code=0, error=None)


def _succeeded_without_writing(rec) -> None:
    rec.finish("succeeded", exit_code=0, error=None)


def _abandon(folder: Path) -> None:
    """The record of a run on another machine whose heartbeat stopped an hour ago."""
    status = read_status_file(folder)
    status["process"]["host"] = "another-machine"
    old = (datetime.now().astimezone() - timedelta(hours=1)).isoformat(timespec="seconds")
    status["heartbeat_at"] = status["started_at"] = old
    write_status_atomic(folder, status)


def _resolve_message(ctx, run: str) -> str:
    with pytest.raises(ToolError) as info:
        _runs.resolve_run(ctx, run)
    return str(info.value)


def test_every_unusable_code_gives_its_message(tmp_path: Path, tests_db: Path) -> None:
    project = _project_with(tmp_path, tests_db, ("running_1", _running), ("gone_1", _running),
                            ("failed_1", succeed), ("partial_1", _partial), ("previous_1", succeed))
    record_run(project, "nowrite_1", "coal", _not_requested, write_methods=("plot", "excel"))
    record_run(project, "failed_1", "coal", fail_with("solve y2020 is infeasible"))
    record_run(project, "previous_1", "coal", _succeeded_without_writing)
    _abandon(project / "output_parquet" / "gone_1")
    ctx = make_context(project)
    codes = {r["folder"]: r["unusable_code"] for r in _call(ctx, "list_runs")["runs"]}
    assert codes == {"failed_1": "failed", "gone_1": "abandoned", "nowrite_1": "no_parquet",
                     "partial_1": "partial", "previous_1": "previous_run", "running_1": "running"}
    assert _resolve_message(ctx, "running_1") == message("run_running", run="running_1")
    assert _resolve_message(ctx, "gone_1") == message("run_abandoned", run="gone_1")
    assert _resolve_message(ctx, "failed_1") == message("run_failed", run="failed_1",
                                                        error="solve y2020 is infeasible")
    assert _resolve_message(ctx, "nowrite_1") == message("run_no_parquet", run="nowrite_1", methods="plot, excel")
    # A succeeded run with no error message: the reason comes from run_record.unusable_reason.
    assert _resolve_message(ctx, "partial_1") == message(
        "run_failed", run="partial_1", error="the run's parquet output is incomplete")
    assert _resolve_message(ctx, "previous_1") == message(
        "run_failed", run="previous_1", error="the parquet files belong to an earlier run")
    for bad in ("../x", "a/b", "a\\b", ""):
        assert _resolve_message(ctx, bad) == message("bad_run_name", run=bad)
    assert _resolve_message(ctx, "nope_1").startswith("There is no result folder 'nope_1'.")


def test_the_rule_is_run_records_own(ctx, monkeypatch) -> None:
    """Changing run_record's predicate changes the tools' answer: they do not restate it."""
    assert _runs.resolve_run(ctx, "coal_1").folder.name == "coal_1"
    monkeypatch.setattr(run_record, "results_usable", lambda status: False)
    monkeypatch.setattr(run_record, "unusable_code", lambda status, effective=None: "partial")
    monkeypatch.setattr(run_record, "unusable_reason", lambda status: "said so by the run record")
    _runs.clear_cache()
    entry = {r["folder"]: r for r in _call(ctx, "list_runs")["runs"]}["coal_1"]
    assert (entry["results_usable"], entry["unusable_code"]) == (False, "partial")
    assert _resolve_message(ctx, "coal_1") == message("run_failed", run="coal_1", error="said so by the run record")


def test_a_changed_input_and_a_legacy_folder_carry_notes(tmp_path: Path, tests_db: Path) -> None:
    project = _project_with(tmp_path, tests_db, ("coal_1", succeed))
    write_parquet(project / "output_parquet" / "old")
    db = project / "input_sources" / "tests.sqlite"
    later = time.time() + 5
    os.utime(db, (later, later))
    ctx = make_context(project)
    resolved = _runs.resolve_run(ctx, "coal_1")
    assert resolved.entry["stale_vs_input"] is True
    assert "The input data changed after this run" in resolved.run_notes[0]
    legacy = _runs.resolve_run(ctx, "old")
    assert legacy.status is None and "no run record" in legacy.run_notes[0]


def test_comparison_falls_back_when_a_member_ran_again_or_only_scenarios_are_recorded(
        tmp_path: Path, tests_db: Path) -> None:
    project = _project_with(tmp_path, tests_db, ("coal_1", succeed))
    first = read_status_file(project / "output_parquet" / "coal_1")["run_id"]
    comparison = project / "output_parquet_comparison"
    comparison.mkdir()
    meta = comparison / "_metadata.json"
    meta.write_text(json.dumps({"scenarios": ["coal_1"], "members": [{"folder": "coal_1", "run_id": first}],
                                "written_at": run_record.now_iso()}), encoding="utf-8")
    ctx = make_context(project)
    assert _runs.comparison_state(ctx)["usable"] is True
    record_run(project, "coal_1", "coal", succeed)              # the member ran again: another run id
    state = _runs.comparison_state(ctx)
    assert state["usable"] is False and state["members"][0]["fresh"] is False
    # The window's metadata-only writer records the scenarios only.
    meta.write_text(json.dumps({"scenarios": ["coal_1"]}), encoding="utf-8")
    state = _runs.comparison_state(ctx)
    assert state["usable"] is False and state["written_at"] is None


def _all_strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in (_all_strings(k) + _all_strings(v))]
    if isinstance(value, list):
        return [s for v in value for s in _all_strings(v)]
    return []


def _set_description(db_path: Path, text: str) -> None:
    from flextool.update_flextool.db_handle import open_database_mapping

    with open_database_mapping(f"sqlite:///{db_path.as_posix()}") as db:
        item = db.get_entity_item(entity_class_name="unit", entity_byname=("coal_plant",))
        db.update_entity_item(id=item["id"], description=text)
        db.commit_session("test: a description with local paths")


def test_no_local_path_host_or_user_name_reaches_a_result(tmp_path: Path, tests_db: Path, monkeypatch) -> None:
    host, user, home = socket.gethostname(), getpass.getuser(), str(Path.home())
    copy = tmp_path / "described.sqlite"
    shutil.copy(tests_db, copy)
    project = make_project(tmp_path / "p", copy)
    leaky = f"notes in {project}/docs/a.txt, {home}/x.txt and /etc/flextool/b.txt on {host} by {user}"
    _set_description(project / "input_sources" / "tests.sqlite", leaky)
    record_run(project, "coal_1", "coal", succeed)
    record_run(project, "coal_chp_1", "coal_chp", succeed)
    record_run(project, "coal_chp_1", "coal_chp", fail_with(f"cannot read {project}/work/x on {host} ({user})"))
    with open(project / "output_parquet" / "coal_chp_1" / run_record.RUN_LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"Reading {project}/input_sources/tests.sqlite as {user} on {host} from {home}/.cache\n")
    ctx = make_context(project)
    calls = [("get_gui_context", {}), ("list_sources", {}), ("list_runs", {}),
             ("get_entity", {"name": ["coal_plant"], "scenario": "coal"}),
             ("list_entities", {"entity_class": "unit"}),
             ("get_parameter", {"entity_class": "unit", "parameter": "existing", "scenario": "coal"}),
             ("get_run_log", {"run": "coal_chp_1"}), ("get_run_log", {"run": "coal_chp_1", "grep": "reading"}),
             ("query_result", {"run": "coal_1", "table": "node_d_ee"}), ("get_slack_summary", {"run": "coal_1"}),
             ("list_result_tables", {"run": "coal_1"})]
    texts = []
    for tool, args in calls:
        out = call_tool(ctx, tool, args)
        assert out.ok, out.message
        texts += _all_strings(json.loads(out.text))
    description = json.loads(call_tool(ctx, "get_entity", {"name": ["coal_plant"], "scenario": "coal"}).text)[
        "matches"][0]["description"]
    assert description == "notes in <project>/docs/a.txt, ~/x.txt and b.txt on <host> by <user>"
    error = {r["folder"]: r for r in json.loads(call_tool(ctx, "list_runs", {}).text)["runs"]}["coal_chp_1"]["error"]
    assert error == "cannot read <project>/work/x on <host> (<user>)"
    # A message, and the fault message.
    texts.append(call_tool(ctx, "get_parameter", {"entity_class": "unit", "parameter": "existing",
                                                  "entity": [f"{project}/x"], "scenario": "coal"}).message)

    def broken(c, a):
        raise OSError(f"{project}/input_sources/tests.sqlite on {host}")
    monkeypatch.setitem(all_tools(), "list_runs", replace(all_tools()["list_runs"], handler=broken))
    texts.append(call_tool(ctx, "list_runs", {}).message)
    for text in texts:
        assert str(project) not in text and home not in text, text
        for word in (host, user):
            assert word.lower() not in text.lower().replace("<host>", "").replace("<user>", ""), text


def test_a_spreadsheet_source_without_its_conversion(tmp_path: Path) -> None:
    project = tmp_path / "p"
    (project / "input_sources").mkdir(parents=True)
    (project / "input_sources" / "model.xlsx").write_bytes(b"not a workbook")
    (project / "settings.yaml").write_text("input_source_numbers:\n  model.xlsx: 2\n", encoding="utf-8")
    (source,) = _call(make_context(project), "list_sources")["sources"]
    assert (source["name"], source["number"], source["kind"]) == ("model.xlsx", 2, "xlsx")
    assert source["conversion"] == {"state": "missing", "converted_at": None}
    assert source["problem"] == message("not_converted") and source["version_status"] is None


# ---------------------------------------------------------------------------------------------------------
# get_run_log


def _two_solves_one_rejected(rec) -> None:
    rec.solve_started("y2020", "y2020", 0, "Solve start: y2020, 1/2")
    rec.solve_classified(model_status="kOptimal", acceptance="optimal", message="", objective=3.0, warm_used=False)
    rec.solve_finished(0)
    rec.solve_started("y2025", "y2025", 1, "Solve start: y2025, 2/2")
    rec.solve_classified(model_status="kInfeasible", acceptance="rejected", message="solve y2025 is infeasible",
                         objective=None, warm_used=False)
    rec.solve_finished(1)
    rec.finish("failed", exit_code=1, error={"type": "FlexToolSolveError", "message": "Error: 1"})


LOG_LINES = [                       # lines 2.. of _run.log (line 1 is the run record's header)
    "Reading the input data",
    "Solve start: y2020, 1/2",
    "Model status: Optimal",
    run_record.MARKER + "a marker line the tools leave out",
    "Solve start: y2025, 2/2",
    "Presolve finds the model infeasible",
    "Model status: Infeasible",
    "Error: solve y2025 is infeasible",
    "Writing nothing",
]


def test_get_run_log_anchor_grep_tail_and_options(tmp_path: Path, tests_db: Path) -> None:
    project = make_project(tmp_path, tests_db)
    record_run(project, "coal_1", "coal", _two_solves_one_rejected)
    folder = project / "output_parquet" / "coal_1"
    with open(folder / run_record.RUN_LOG_FILE, "a", encoding="utf-8") as log:
        log.write("\n".join(LOG_LINES) + "\n")
    status = read_status_file(folder)
    status["options_resolved"] = {
        "scaling": {"value": "basic", "source": "env", "env": "FLEXTOOL_SCALING"},
        "solver_mip_gap": {"value": 0.01, "source": "db", "env": "FLEXTOOL_HIGHS_MIP_GAP"},
        "threads": {"value": 4, "source": "cli", "env": "FLEXTOOL_THREADS"},
        "presolve": {"value": None, "source": "default", "env": "FLEXTOOL_HIGHS_PRESOLVE"},
    }
    write_status_atomic(folder, status)
    ctx = make_context(project)

    out = _call(ctx, "get_run_log", run="coal_1", solve="y2025")
    # The failed run is readable here; its rejected solve comes first.
    st = out["status"]
    assert (st["state"], st["verdict"]) == ("failed", "infeasible")
    assert [s["name"] for s in st["solves"]] == ["y2025", "y2020"]
    assert st["solves"][0]["message"] == "solve y2025 is infeasible"
    assert set(st["threads"]) == {"budget", "source", "polars", "polars_source", "highs"}
    # Wording of where each option came from (run_options.OPTIONS: flag --threads, database solve.solver_mip_gap).
    assert st["options"] == [
        {"option": "scaling", "value": "basic", "set_by": "set by the environment variable FLEXTOOL_SCALING"},
        {"option": "solver_mip_gap", "value": 0.01, "set_by": "from the database parameter solve.solver_mip_gap"},
        {"option": "threads", "value": 4, "set_by": "set by the run flag --threads"},
        {"option": "presolve", "value": None, "set_by": "default"},
    ]
    # The y2025 anchor is line 6 (header, then LOG_LINES[0..4]); the marker line (5) is left out.
    assert out["log"]["from_line"] == 6 and out["log"]["total_lines"] == 9
    assert out["log"]["lines"] == ["6: Solve start: y2025, 2/2", "7: Presolve finds the model infeasible",
                                   "8: Model status: Infeasible", "9: Error: solve y2025 is infeasible",
                                   "10: Writing nothing"]
    # grep: any case, two lines around each match (lines 7, 8 and 9 contain "infeasible").
    out = _call(ctx, "get_run_log", run="coal_1", grep="INFEASIBLE")
    assert [line.split(":", 1)[0] for line in out["log"]["lines"]] == ["4", "6", "7", "8", "9", "10"]
    # The tail: the last two lines.
    out = _call(ctx, "get_run_log", run="coal_1", tail_lines=2)
    assert out["log"]["lines"] == ["9: Error: solve y2025 is infeasible", "10: Writing nothing"]
    message_text = call_tool(ctx, "get_run_log", {"run": "coal_1", "solve": "y2030"}).message
    assert message_text == "The run coal_1 has no solve named 'y2030'. Its solves include: y2020, y2025."


def test_get_run_log_without_a_run_record(ctx) -> None:
    out = call_tool(ctx, "get_run_log", {"run": "base_1"})
    assert not out.ok and out.message == message("no_run_record")


# ---------------------------------------------------------------------------------------------------------
# get_slack_summary


def _slack_tables(folder: Path) -> None:
    import pandas as pd

    from tests.assistant.results_tables import PERIODS, write_result

    periods = pd.Index(PERIODS, name="period")
    # node_slack_up_d_e [MWh/a]: west 12.5 in p2020; east below min_value.
    slack = pd.DataFrame({"west": [12.5, 0.0], "east": [1e-9, 0.0]}, index=periods)
    slack.columns.name = "node"
    write_result(folder, "node_slack_up_d_e", slack, "coal")
    costs = pd.DataFrame({"investment": [3.0, 0.0], "upward slack penalty": [0.125, 0.0]}, index=periods)
    costs.columns.name = "category"
    write_result(folder, "annualized_costs_d_p", costs, "coal")


def test_get_slack_summary_with_feeding_entities(tmp_path: Path, tests_db: Path) -> None:
    from tests.assistant.results_tables import succeed_writing

    project = make_project(tmp_path, tests_db)
    record_run(project, "coal_1", "coal", succeed_writing(_slack_tables))
    record_run(project, "clean_1", "coal", succeed)
    ctx = make_context(project)
    out = _call(ctx, "get_slack_summary", run="coal_1")
    (slack,) = out["slacks"]
    assert (slack["slack"], slack["table"], slack["unit"]) == ("vq_state_up", "node_slack_up_d_e",
                                                                "MWh/a (default convention)")
    assert slack["entries"] == [{"entity": ["west"], "period": "p2020", "value": 12.5}]
    assert slack["total"] == {"p2020": 12.5}
    assert "node.penalty_up" in slack["related_inputs"]
    # tests.json, scenario coal = alternatives init, west, coal: unit and connection entities are inactive by
    # default and only coal_plant is made active (alternative coal); unit__outputNode is active by default, so
    # the one active unit or connection with west as a dimension is unit__outputNode (coal_plant, west).
    assert slack["feeding_entities"] == {"west": [{"entity_class": "unit__outputNode",
                                                   "entity": ["coal_plant", "west"]}]}
    assert out["penalty_costs"]["entries"] == [{"category": "upward slack penalty", "period": "p2020",
                                                "value": 0.125}]
    assert out["penalty_costs"]["unit"] == "M CUR/a (default convention)" and out["none_found"] is False
    assert _call(ctx, "get_slack_summary", run="clean_1")["none_found"] is True


def test_get_slack_summary_needs_a_successful_run(ctx) -> None:
    out = call_tool(ctx, "get_slack_summary", {"run": "coal_chp_1"})
    assert not out.ok and out.message.startswith("The latest run of coal_chp_1 failed")


# ---------------------------------------------------------------------------------------------------------
# Comparison files that record their runs


def _comparison_project(tmp_path: Path, tests_db: Path) -> tuple[Path, dict[str, str]]:
    import pandas as pd

    from tests.assistant.results_tables import PERIODS, succeed_writing, write_result

    def tables(values: list[float], scenario: str):
        def write(folder: Path) -> None:
            frame = pd.DataFrame({("coal_plant", "west"): values}, index=pd.Index(PERIODS, name="period"))
            frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=["unit", "node"])
            write_result(folder, "unit_outputNode_d_ee", frame, scenario)
        return write

    project = make_project(tmp_path, tests_db)
    ids = {"base_1": record_run(project, "base_1", "base", succeed_writing(tables([1000.0, 2000.0], "base"))),
           "high_1": record_run(project, "high_1", "high", succeed_writing(tables([1500.0, 2000.0], "high")))}
    return project, ids


def _compare(ctx) -> dict:
    return _call(ctx, "compare_runs", table="unit_outputNode_d_ee", runs=["comparison"])


def test_comparison_files_record_their_runs_and_are_used_while_fresh(tmp_path: Path, tests_db: Path) -> None:
    from flextool.scenario_comparison.db_reader import combine_scenario_parquets

    project, ids = _comparison_project(tmp_path, tests_db)
    combine_scenario_parquets(project, ["base_1", "high_1"])
    text = (project / "output_parquet_comparison" / "_metadata.json").read_text(encoding="utf-8")
    # The scenario list is written as before; the member runs and the time follow it.
    assert text.startswith('{"scenarios": ["base_1", "high_1"], "members": [')
    meta = json.loads(text)
    assert [(m["folder"], m["run_id"]) for m in meta["members"]] == [("base_1", ids["base_1"]),
                                                                      ("high_1", ids["high_1"])]
    ctx = make_context(project)
    out = _compare(ctx)
    assert out["source"] == "comparison_files"
    assert [(b["folder"], b["run_id"]) for b in out["built_from"]] == [("base_1", ids["base_1"]),
                                                                        ("high_1", ids["high_1"])]
    # high_1 − base_1: p2020 1500 − 1000 = 500, p2025 0.
    assert [tuple(r) for r in out["rows"]] == [("p2020", "coal_plant", "west", 1000.0, 1500.0, 500.0),
                                               ("p2025", "coal_plant", "west", 2000.0, 2000.0, 0.0)]


def test_comparison_falls_back_to_member_folders(tmp_path: Path, tests_db: Path) -> None:
    from types import SimpleNamespace

    from flextool.gui.result_viewer import ResultViewer
    from flextool.scenario_comparison.db_reader import combine_scenario_parquets
    from tests.assistant.results_tables import succeed_writing, write_result

    project, _ = _comparison_project(tmp_path, tests_db)
    combine_scenario_parquets(project, ["base_1", "high_1"])
    meta_path = project / "output_parquet_comparison" / "_metadata.json"
    fresh = meta_path.read_text(encoding="utf-8")
    ctx = make_context(project)

    # A comparison table older than the comparison: the member folders.
    meta = json.loads(fresh)
    meta["written_at"] = (datetime.now().astimezone() + timedelta(hours=1)).isoformat(timespec="seconds")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    out = _compare(ctx)
    assert out["source"] == "run_folders" and out["notes"][0].startswith("The comparison files do not match")

    # The window's metadata-only writer records the scenarios only: the member folders.
    ResultViewer._write_comparison_metadata(SimpleNamespace(_project_path=project), ["base_1", "high_1"])
    assert json.loads(meta_path.read_text(encoding="utf-8")) == {"scenarios": ["base_1", "high_1"]}
    out = _compare(ctx)
    assert out["source"] == "run_folders" and "do not record the runs" in out["notes"][0]

    # A member ran again (other values): the comparison files no longer stand for it.
    meta_path.write_text(fresh, encoding="utf-8")
    assert _compare(ctx)["source"] == "comparison_files"
    import pandas as pd

    def rerun(folder: Path) -> None:
        frame = pd.DataFrame({("coal_plant", "west"): [900.0, 2000.0]}, index=pd.Index(["p2020", "p2025"],
                                                                                       name="period"))
        frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=["unit", "node"])
        write_result(folder, "unit_outputNode_d_ee", frame, "high")
    record_run(project, "high_1", "high", succeed_writing(rerun))
    _runs.clear_cache()
    out = _compare(ctx)
    assert out["source"] == "run_folders" and out["built_from"] is None
    # The values are high_1's new run: 900 − 1000 = −100.
    assert [tuple(r) for r in out["rows"]][0] == ("p2020", "coal_plant", "west", 1000.0, 900.0, -100.0)


def test_the_comparison_orchestrator_records_its_runs(tmp_path: Path, tests_db: Path) -> None:
    from unittest.mock import patch

    from flextool.scenario_comparison.orchestrator import run

    project, ids = _comparison_project(tmp_path, tests_db)
    config = tmp_path / "plots.yaml"
    config.write_text("plots: {}\n", encoding="utf-8")
    comparison = project / "output_parquet_comparison"
    with patch("flextool.scenario_comparison.orchestrator.plot_dict_of_dataframes"):
        run(db_url=None, parquet_subdir="", plot_dir=str(tmp_path / "plots"), output_config_path=str(config),
            active_configs=["default"], plot_rows=[0, 167], write_to_xlsx=False, write_dispatch_xlsx=False,
            write_to_ods=False, show_plots=False, dispatch_plots=False,
            scenario_folders={"base_1": str(project / "output_parquet"), "high_1": str(project / "output_parquet")},
            comparison_parquet_dir=str(comparison))
    meta = json.loads((comparison / "_metadata.json").read_text(encoding="utf-8"))
    assert meta["scenarios"] == ["base_1", "high_1"]
    assert [(m["folder"], m["run_id"]) for m in meta["members"]] == [("base_1", ids["base_1"]),
                                                                      ("high_1", ids["high_1"])]
    assert _compare(make_context(project))["source"] == "comparison_files"


# ---------------------------------------------------------------------------------------------------------
# A run record that cannot be read now, or is damaged: never served, never taken for a missing one


def test_unreadable_record_not_served_and_not_cached(tmp_path: Path, tests_db: Path) -> None:
    project = make_project(tmp_path, tests_db)
    folder = project / "output_parquet" / "coal_1"
    write_parquet(folder)
    (folder / run_record.RUN_STATUS_FILE).write_text('{"format": 1, "state": "failed", "outputs": {"parq',
                                                     encoding="utf-8")
    ctx = make_context(project)
    _runs.clear_cache()
    (entry,) = _call(ctx, "list_runs")["runs"]
    assert (entry["results_usable"], entry["unusable_code"], entry["state"]) == (False, "unreadable", "unreadable")
    assert "note" not in entry
    assert _resolve_message(ctx, "coal_1") == message("run_unreadable", run="coal_1")
    assert str(folder / run_record.RUN_STATUS_FILE) not in _runs._status_cache
    # Cut short long ago: damaged, not "being written".
    then = time.time() - 60
    os.utime(folder / run_record.RUN_STATUS_FILE, (then, then))
    (entry,) = _call(ctx, "list_runs")["runs"]
    assert (entry["results_usable"], entry["unusable_code"]) == (False, "malformed")
    assert _resolve_message(ctx, "coal_1") == message("run_malformed", run="coal_1")


def test_a_status_that_cannot_be_stat_ed_is_unreadable(tmp_path: Path, tests_db: Path, monkeypatch) -> None:
    project = _project_with(tmp_path, tests_db, ("coal_1", succeed))
    ctx = make_context(project)
    _runs.clear_cache()
    status_path = project / "output_parquet" / "coal_1" / run_record.RUN_STATUS_FILE
    real_stat = Path.stat

    def stat(self, *args, **kwargs):
        if self == status_path:
            raise PermissionError(13, "delete pending")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    (entry,) = _call(ctx, "list_runs")["runs"]
    assert (entry["results_usable"], entry["unusable_code"]) == (False, "unreadable")
    assert "note" not in entry
    assert str(status_path) not in _runs._status_cache


def test_list_runs_and_get_run_log_survive_a_status_of_the_wrong_shape(tmp_path: Path, tests_db: Path) -> None:
    project = _project_with(tmp_path, tests_db, ("coal_1", succeed))
    folder = project / "output_parquet" / "coal_1"
    status = read_status_file(folder)
    status["solves"] = [1]
    status["input"] = "x"
    write_status_atomic(folder, status)
    ctx = make_context(project)
    _runs.clear_cache()
    (entry,) = _call(ctx, "list_runs")["runs"]
    assert (entry["results_usable"], entry["unusable_code"], entry["state"]) == (False, "malformed", "malformed")
    out = _call(ctx, "get_run_log", run="coal_1")
    assert out["status"]["state"] == "malformed" and out["status"]["solves"] == []
