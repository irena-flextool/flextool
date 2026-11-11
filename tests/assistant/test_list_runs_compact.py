"""The token-lean tools on a project of many runs (agent feedback 2026-11-09, Cycle 1).

``many_runs_project`` (``conftest.build_many_runs_project``): 67 run folders ``<scenario>_1`` from the first 67
scenarios of ``tests/fixtures/tests.json`` — the first 43 hold a result file but no run record, the next 20
succeeded, the last 4 failed (so 63 have usable results).  The succeeded runs hold ``cost_unit_annualized_d_ec``
(3 periods × 16 units rows, 12 categories).  The size bounds are upper bounds on the text that is sent; the sizes
before Cycle 1 are in the comments (measured at the medium result cap).
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from flextool.assistant.budget import approx_tokens
from flextool.assistant.context import make_context
from flextool.assistant.redact import Redactor
from flextool.assistant.registry import all_tools, call_tool

from tests.assistant.conftest import (
    MANY_NO_RECORD,
    MANY_RUNS,
    MANY_SUCCEEDED,
    make_project,
    many_run_scenarios,
    write_parquet,
)

NO_RECORD_NOTE = ("43 runs have no run record (state no_record, verified false): the run's success is not known; "
                  "results may be from a failed or older run. Say so when you use them.")
#: The fields of a run record (never on a row without one).
RECORD_FIELDS = {"verdict", "started_at", "finished_at", "flextool_version", "other_flextool_version",
                 "stale_vs_input", "unusable_code", "error"}


def _text(ctx, tool: str, **args) -> str:
    out = call_tool(ctx, tool, args)
    assert out.ok, out.message
    return out.text


def _call(ctx, tool: str, **args) -> dict:
    return json.loads(_text(ctx, tool, **args))


@pytest.fixture
def mctx(many_runs_project: Path):
    return make_context(many_runs_project)


def _folders() -> list[str]:
    return [f"{s}_1" for s in many_run_scenarios()]


def test_list_runs_compact_lists_every_run(mctx) -> None:
    """``all: true``: every run, compact.  Before Cycle 1: 44 of the 67 rows at the medium cap (19 661 characters,
    about 9 000 tokens), the no-record note on each of the 43 rows, about ten nulls a row.  Cycle 2 added the
    ``finished`` column (newest first; ``test_list_runs_scan`` checks the order)."""
    text = _text(mctx, "list_runs", all=True)
    out = json.loads(text)
    assert out["total"] == MANY_RUNS == 67 and len(out["runs"]) == 67
    assert out["counts"] == {"no_record": 43, "succeeded": 20, "failed": 4}
    assert list(out["counts"]) == ["no_record", "succeeded", "failed"]          # largest count first
    assert out["results_usable"] == 63
    assert out["columns"] == ["folder", "state", "finished", "results_usable"]
    assert sorted(r[0] for r in out["runs"]) == sorted(_folders())
    assert {r[1] for r in out["runs"][:24]} == {"succeeded", "failed"}         # recorded runs first
    assert {r[1] for r in out["runs"][24:]} == {"no_record"}
    rows = {r[0]: r for r in out["runs"]}
    names = _folders()
    assert [rows[names[0]][i] for i in (1, 3)] == ["no_record", True]
    assert [rows[names[MANY_NO_RECORD]][i] for i in (1, 3)] == ["succeeded", True]
    assert [rows[names[-1]][i] for i in (1, 3)] == ["failed", False]
    assert out["notes"][0] == NO_RECORD_NOTE and text.count("no run record") == 1
    assert "null" not in text and "shown" not in out and "truncated" not in out
    # Cycle 2: + the finished column and the order note (about 190 characters)
    assert len(text) <= 5_000 and approx_tokens(text, "json") <= 2_800


def test_list_runs_full_detail_states_shared_fields_only_when_every_row_has_them(mctx) -> None:
    """Critique B1: a field stated once (``same_in_every_row``) holds for every row; a run without a record never
    seems to have a verdict or a version.  Before: ``detail`` was refused; the large cap sent 28 782 characters
    (about 13 700 tokens)."""
    text = _text(mctx, "list_runs", detail="full", all=True)
    out = json.loads(text)
    assert len(out["runs"]) == 67
    common = out["same_in_every_row"]
    assert common == {"source_number": 1, "source_name": "tests.sqlite"}
    rows = {r["folder"]: r for r in out["runs"]}
    names = _folders()
    assert rows["base_1"] == {"folder": "base_1", "state": "no_record", "results_usable": True, "verified": False}
    for name in names[:MANY_NO_RECORD]:
        assert not RECORD_FIELDS & set(rows[name]), name
    for name in names[MANY_NO_RECORD:MANY_NO_RECORD + MANY_SUCCEEDED]:
        row = rows[name]
        assert (row["state"], row["verdict"], row["results_usable"]) == ("succeeded", "optimal", True)
        assert "verified" not in row and "flextool_version" in row
    failed = rows[names[-1]]
    assert (failed["state"], failed["unusable_code"], failed["error"]) == (
        "failed", "failed", "solve y2020 is infeasible")
    assert "null" not in text and text.count("no run record") == 1
    assert len(text) <= 12_000 and approx_tokens(text, "json") <= 6_000


def test_list_runs_filters(mctx) -> None:
    no_record = _call(mctx, "list_runs", state="no_record")
    assert no_record["total"] == 43 and no_record["counts"] == {"no_record": 43}
    assert "columns" in no_record                                   # more than five runs: compact
    unusable = _call(mctx, "list_runs", results_usable=False)
    # The 4 failed runs, at most five: with full detail.
    assert sorted(r["folder"] for r in unusable["runs"]) == sorted(_folders()[-4:])
    assert unusable["same_in_every_row"]["state"] == "failed"
    y2020 = _call(mctx, "list_runs", folders=["y2020*"])
    assert sorted(r["folder"] for r in y2020["runs"]) == ["y2020_2029_1x10y_1", "y2020_2029_2x5y_1"]
    assert all(r["verdict"] == "infeasible" for r in y2020["runs"])
    # Exact names: full detail, whatever their number; filters combine.
    named = _call(mctx, "list_runs", folders=["base_1", "coal_1", "wind_1"], results_usable=True)
    assert {r["folder"] for r in named["runs"]} == {"base_1", "coal_1", "wind_1"}
    by_name = {r["folder"]: {**named.get("same_in_every_row", {}), **r} for r in named["runs"]}
    assert by_name["base_1"]["verified"] is False and "verdict" not in by_name["base_1"]
    assert by_name["wind_1"]["verdict"] == "optimal"            # index 58 of the scenarios: succeeded
    # Many runs named exactly: still full.
    many = _call(mctx, "list_runs", folders=_folders()[:8])
    assert "columns" not in many and len(many["runs"]) == 8
    # The state is an enum of the run states (gpt-oss invented "success", 2026-11-11); another is refused in plain
    # English with the states of this project's runs, as before the enum.
    refused = call_tool(mctx, "list_runs", {"state": "finished"})
    assert not refused.ok and refused.message == ('There is no run state "finished". The states of this project\'s '
                                                  'runs are: "succeeded", "failed", "no_record".')
    empty = call_tool(mctx, "list_runs", {"state": ""})
    assert not empty.ok and empty.message.startswith('There is no run state "". The states of this project')
    # Any other enum is refused in plain English by the schema check, its values quoted as names.
    detail = call_tool(mctx, "list_runs", {"detail": "summary"})
    assert not detail.ok and detail.message == ('The arguments of "list_runs" are not valid: detail: "summary" is '
                                                'not one of its values, which are "scan", "overview", "full".')
    assert all_tools()["list_runs"].input_schema["properties"]["state"] == {
        "type": "string", "enum": ["succeeded", "failed", "killed", "interrupted", "crashed", "running", "abandoned",
                                   "unknown", "unreadable", "malformed", "no_record"]}
    unknown = call_tool(mctx, "list_runs", {"folders": ["base_2"]})
    assert not unknown.ok and unknown.message.startswith('There is no result folder "base_2".')
    assert all_tools()["list_runs"].input_schema["properties"]["detail"] == {
        "type": "string", "enum": ["scan", "overview", "full"]}


def test_list_runs_pages_explicitly(mctx) -> None:
    """At a 2 000-character cap the rows come in pages; each page names the next offset, and the pages hold every
    run once.  Before: ``offset`` was refused and fit_json cut the list (``truncated``)."""
    small = dataclasses.replace(mctx, caps=dataclasses.replace(mctx.caps, max_chars=2_000))
    small.redactor = mctx.redactor
    order = [r[0] for r in _call(mctx, "list_runs", all=True)["runs"]]
    seen: list[str] = []
    offset = 0
    pages = 0
    while True:
        text = _text(small, "list_runs", offset=offset, all=True)
        out = json.loads(text)
        assert len(text) <= 2_000 and "truncated" not in out
        assert out["total"] == 67 and out["counts"]["no_record"] == 43           # every page: the whole set
        seen += [r[0] for r in out["runs"]]
        pages += 1
        end = offset + len(out["runs"])
        if end == 67:
            assert out["shown"] == f"runs {offset + 1}–67 of 67, newest first"
            break
        assert out["shown"] == (f"runs {offset + 1}–{end} of 67, newest first; pass offset {end} for the next (with "
                                "the same filters; a run that starts or finishes moves to the top), or filter by "
                                "state, results_usable or folders")
        offset = end
    assert seen == order and pages >= 2
    past = _call(small, "list_runs", offset=70, all=True)
    assert past["runs"] == [] and past["shown"] == "no runs from offset 70; there are 67"


def test_paging_is_measured_on_the_redacted_text(tmp_path: Path, tests_db: Path) -> None:
    """Critique M3: the user name in 40 folder names is sent as <user> (three characters longer each).  A page fitted
    on the names as stored would be longer than the cap once redacted, and fit_json would drop rows the page note
    says were sent.  Each page fits as sent, and the pages hold every folder once, newest files first (no run
    records: by the files' times, equal times by name)."""
    project = make_project(tmp_path, tests_db)
    names = [f"s-zz9-{i:02d}_1" for i in range(40)]
    for name in names:
        write_parquet(project / "output_parquet" / name)
    base = make_context(project)
    ctx = dataclasses.replace(base, caps=dataclasses.replace(base.caps, max_chars=1_200))
    ctx.redactor = Redactor(project, user="zz9")
    seen: list[str] = []
    offset = 0
    while offset < 40:
        text = _text(ctx, "list_runs", offset=offset, all=True)
        out = json.loads(text)
        assert len(text) <= 1_200 and "truncated" not in out and "zz9" not in text
        assert out["runs"], out
        seen += [r[0] for r in out["runs"]]
        offset += len(out["runs"])
        if offset < 40:
            assert f"pass offset {offset} for the next" in out["shown"]
    by_time = sorted(names, key=lambda n: (-(project / "output_parquet" / n / "node_d_ee.parquet").stat().st_mtime,
                                           n))
    assert seen == [n.replace("zz9", "<user>") for n in by_time]


def test_compare_runs_overview_is_short(mctx) -> None:
    """Without arguments: per period and category, units added up (36 rows) and the totals per period.  Before:
    each (period, unit, category) value, largest difference first: 10 564 characters, about 7 000 tokens."""
    names = _folders()
    a, b = names[MANY_NO_RECORD], names[MANY_NO_RECORD + 1]
    text = _text(mctx, "compare_runs", table="cost_unit_annualized_d_ec", runs=[a, b])
    out = json.loads(text)
    assert out["columns"] == ["period", "category", a, b, "difference_vs_first"]
    assert out["total_rows"] == 36 and len(out["rows"]) == 36
    assert [r[:2] for r in out["rows"][:2]] == [["p2020", "category 01"], ["p2020", "category 02"]]
    assert out["totals"]["columns"] == ["period", a, b, "difference_vs_first"]
    assert [r[0] for r in out["totals"]["rows"]] == ["p2020", "p2025", "p2030"]
    assert out["applied"]["mode"] == "overview"
    assert len(text) <= 3_200 and approx_tokens(text, "json") <= 2_100


def test_compare_inputs_is_short(mctx) -> None:
    """The largest pair of tests.json: 159 values that differ.  Counts by alternative and class are complete; 30
    rows by default.  Before: 87 rows cut at the character cap, 19 810 characters (about 9 200 tokens)."""
    text = _text(mctx, "compare_inputs", scenario_a="5weeks_battery_intraperiod_blocks", scenario_b="test_a_lot")
    out = json.loads(text)
    assert out["total_values"] == 159
    assert sum(n for per_class in out["by_alternative"].values() for n in per_class.values()) == 159
    assert len(out["differences"]) == 30
    assert out["shown"] == (f"30 of {out['total_rows']} rows (159 values); pass alternative or entity_class, or a "
                            "larger max_rows")
    assert len(text) <= 8_000 and approx_tokens(text, "json") <= 3_800
