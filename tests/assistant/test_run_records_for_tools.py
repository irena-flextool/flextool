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
    ctx = make_context(project)
    calls = [("get_gui_context", {}), ("list_sources", {}), ("list_runs", {}),
             ("get_entity", {"name": ["coal_plant"], "scenario": "coal"}),
             ("list_entities", {"entity_class": "unit"}),
             ("get_parameter", {"entity_class": "unit", "parameter": "existing", "scenario": "coal"})]
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
