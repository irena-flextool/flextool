"""list_runs in Cycle 2 (agent feedback 2026-11-09 items 1 and 2, C1-OD4, C2-OD1 corrected, C2-OD3): the latest
runs first, a scan of the costs across runs, and one run's overview — on ``latest_runs_project``
(``conftest.build_latest_runs_project``: its times, costs, capacities and slacks are listed there).

The expected values are worked by hand from the fixture's numbers (the comments show the sums); the order of the
runs from its times: the newest run ``<s19>_2`` (2026-10-09), the four failed ones (2026-10-08, f = 3 … 0), the
succeeded ones (2026-10-01, j = 19 … 0, j = 10 and 11 at the same time: by folder name), then the 43 runs without
a record by when their files were written, newest first, although those times are newer than every record (a
copied project, critique M2).
"""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path

import pytest

from flextool.assistant.budget import approx_tokens, resolve_budget
from flextool.assistant.context import make_context
from flextool.assistant.registry import call_tool
from flextool.assistant.results import _overview

from tests.assistant.conftest import MANY_NO_RECORD, many_run_scenarios

SCENARIOS = many_run_scenarios()


def succ(j: int) -> str:
    return f"{SCENARIOS[MANY_NO_RECORD + j]}_1"


def fail(f: int) -> str:
    return f"{SCENARIOS[MANY_NO_RECORD + 20 + f]}_1"


NEWEST = f"{SCENARIOS[MANY_NO_RECORD + 19]}_2"          # the scenario of succ(19), run from source 2
#: The recorded runs, newest first.
RECORDED = [NEWEST, fail(3), fail(2), fail(1), fail(0), *(succ(j) for j in range(19, 11, -1)),
            *sorted([succ(10), succ(11)]), *(succ(j) for j in range(9, -1, -1))]
#: The record-less runs by their files' time: i = 42 … 7, then i = 5 and 6 (one time: by name), then 4 … 0.
NO_RECORD = [f"{SCENARIOS[i]}_1" for i in range(42, 6, -1)] + sorted(f"{SCENARIOS[i]}_1" for i in (5, 6)) + \
    [f"{SCENARIOS[i]}_1" for i in range(4, -1, -1)]
ALL = RECORDED + NO_RECORD
ORDER_NOTE = ("43 runs without a readable run record are listed after the others, by when their files were last "
              "written (finished), which copying a project changes: their order is uncertain.")
COST_UNIT = ("M CUR (default convention), discounted, all periods (costs_discounted_p_); compare runs over the same "
             "periods; penalty_cost: slack penalties")


def _local(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M")


@pytest.fixture
def lctx(latest_runs_project: Path):
    return make_context(latest_runs_project)


def _with_caps(ctx, caps):
    out = dataclasses.replace(ctx, caps=caps)
    out.redactor = ctx.redactor
    return out


def _call(ctx, **args) -> dict:
    out = call_tool(ctx, "list_runs", args)
    assert out.ok, out.message
    return json.loads(out.text)


def _text(ctx, **args) -> str:
    out = call_tool(ctx, "list_runs", args)
    assert out.ok, out.message
    return out.text


# --- latest first -----------------------------------------------------------------------------------------------

def test_latest_runs_first(lctx) -> None:
    """Without filters: the newest 20 runs (medium caps), both runs of the scenario run from two sources, counts of
    the older ones by state; recorded runs before the record-less ones whose files look newer.  Before Cycle 2:
    every run by folder name, no ``finished``, ``all`` refused."""
    text = _text(lctx)
    out = json.loads(text)
    assert [r[0] for r in out["runs"]] == RECORDED[:20]
    assert succ(19) in RECORDED[:20] and NEWEST in RECORDED[:20]           # <s>_1 and <s>_2 both listed
    assert out["columns"] == ["folder", "state", "finished", "results_usable"]
    rows = {r[0]: r for r in out["runs"]}
    assert rows[NEWEST] == [NEWEST, "succeeded", _local("2026-10-09T00:00:00+00:00"), True]
    assert rows[fail(1)] == [fail(1), "killed", _local("2026-10-08T01:00:00+00:00"), False]
    assert rows[succ(11)][2] == rows[succ(10)][2] == _local("2026-10-01T10:00:00+00:00")
    assert out["total"] == 68 and out["results_usable"] == 64
    assert out["counts"] == {"no_record": 43, "succeeded": 21, "failed": 3, "killed": 1}
    assert out["older"] == {"no_record": 43, "succeeded": 5}               # succ(4) … succ(0)
    assert "scenarios" not in out and ORDER_NOTE in out["notes"]
    assert out["shown"] == ("the latest 20 of 68 runs, newest first; pass offset 20 for the next (a run that starts "
                            "or finishes moves to the top), or all true for every run")
    assert len(text) <= 2_200 and approx_tokens(text, "json") <= 1_200
    # every run, the same order
    everything = _call(lctx, all=True)
    assert [r[0] for r in everything["runs"]] == ALL and "older" not in everything and "shown" not in everything
    assert {r[0]: r[2] for r in everything["runs"]}[NO_RECORD[0]] == _local("2026-11-01T00:42:00+00:00")


@pytest.mark.parametrize(("level", "window", "free", "rows"), [
    ("small", 16_384, False, 8), ("small", 32_768, False, 8), ("small", 200_000, False, 8),
    ("medium", 65_536, False, 20), ("medium", 200_000, True, 20), ("large", 200_000, False, 20),
])
def test_the_level_sets_how_many_latest_runs(lctx, level, window, free, rows) -> None:
    """Critique M3: the number of latest runs survives every budget path (the fit below 32 768 tokens, the small
    and medium window classes, the free plan).  Before: ``Caps`` had no ``max_runs``; with one defaulted to 20,
    ``_smaller`` and ``fit_caps`` gave 20 at basic."""
    caps = resolve_budget(level, window, "loaded", local=True, free_tier=free).result_caps
    assert caps.max_runs == caps.ceilings.max_runs == rows
    out = _call(_with_caps(lctx, caps))
    assert [r[0] for r in out["runs"]] == RECORDED[:rows]
    assert len(json.dumps(out, ensure_ascii=False, separators=(",", ":"))) <= caps.max_chars


def test_the_latest_runs_page_without_gaps(lctx) -> None:
    """Basic caps at 16 384 tokens: pages of 8 newest first; their offsets give every run once, in the order of
    ``all``.  A page that fits fewer than 8 rows names its own next offset (critique m7)."""
    caps = resolve_budget("small", 16_384, "loaded", local=True).result_caps
    small = _with_caps(lctx, caps)
    seen: list[str] = []
    offset = 0
    while offset < 68:
        out = _call(small, offset=offset)
        assert 0 < len(out["runs"]) <= 8 and "truncated" not in out
        seen += [r[0] for r in out["runs"]]
        offset += len(out["runs"])
    assert seen == ALL
    tight = _with_caps(lctx, dataclasses.replace(caps, max_chars=1_000))
    out = _call(tight)
    k = len(out["runs"])
    assert 0 < k < 8 and [r[0] for r in out["runs"]] == RECORDED[:k]
    assert out["shown"] == (f"the latest {k} of 68 runs, newest first; pass offset {k} for the next (a run that "
                            "starts or finishes moves to the top), or all true for every run")


def test_a_filter_lists_every_match_newest_first(lctx) -> None:
    out = _call(lctx, state="succeeded")
    succeeded = [r for r in RECORDED if r not in (fail(0), fail(1), fail(2), fail(3))]
    assert out["total"] == 21 and [r[0] for r in out["runs"]] == succeeded
    assert "older" not in out and "shown" not in out


# --- scan -------------------------------------------------------------------------------------------------------

@pytest.fixture
def cost_reads(monkeypatch) -> list[str]:
    seen: list[str] = []
    real = _overview.read_costs

    def spy(folder):
        seen.append(folder.name)
        return real(folder)
    monkeypatch.setattr(_overview, "read_costs", spy)
    return seen


def test_scan_rows(lctx, cost_reads) -> None:
    """``detail: "scan"``: each row's discounted total, slack penalties and largest positive cost; no number (and no
    read) for a run whose results may not be used; a usable run without the table is named.  Before: the enum
    refused "scan"."""
    out = _call(lctx, detail="scan")
    assert out["columns"] == ["folder", "state", "finished", "results_usable", "total_cost", "penalty_cost",
                              "largest_cost"]
    rows = {r[0]: r[4:] for r in out["runs"]}
    # 600 − 50 + 80 + 5 + 0.5 + 300 + 2 + 200 + 40 + 30 + 10 + 0.25 = 1 217.75; penalties 300 + 2
    assert rows[NEWEST] == [1217.75, 302.0, ["commodity_cost", 600.0]]
    # j = 7: 170 − 20 + 50 + 40 + 5 = 245; penalties 40 + 5
    assert rows[succ(7)] == [245.0, 45.0, ["commodity_cost", 170.0]]
    assert rows[succ(19)] == [320.0, 0.0, ["commodity_cost", 290.0]]       # 290 − 20 + 50
    for name in (fail(0), fail(1), fail(2), fail(3), succ(5)):
        assert rows[name] == [None, None, None], name
    assert out["no_cost_table"] == succ(5)
    assert out["cost_unit"] == COST_UNIT + "; largest_cost: [category, value] of the largest cost"
    assert not {fail(0), fail(1), fail(2), fail(3)} & set(cost_reads)
    assert set(cost_reads) <= set(RECORDED[:20])


def test_scan_of_every_run(lctx, cost_reads) -> None:
    """``all: true``: the largest cost is the largest positive one, not a larger negative sale (critique m3); a run
    made with another FlexTool is named (m6); a record-less run has no cost table here, so its numbers are null."""
    out = _call(lctx, detail="scan", all=True)
    rows = {r[0]: r[4:] for r in out["runs"]}
    assert rows[succ(4)] == [-310.0, 0.0, ["commodity_cost", 140.0]]        # 140 − 500 + 50
    assert rows[succ(2)] == [190.0, 40.0, ["commodity_cost", 120.0]]        # 120 − 20 + 50 + 40
    assert out["other_flextool_version_in"] == succ(3)
    assert rows[NO_RECORD[0]] == [None, None, None]
    assert out["no_cost_table"].startswith(f"{succ(5)}, ") and out["no_cost_table"].endswith("and 34 more")


def test_scan_reads_about_the_rows_it_sends(lctx, cost_reads) -> None:
    """Critique m5: at 2 000 characters the scan reads the cost tables of about the rows it sends (not of every
    run from the offset on), and names in its header only folders it sends."""
    small = _with_caps(lctx, dataclasses.replace(lctx.caps, max_chars=2_000))
    text = _text(small, detail="scan", all=True)
    out = json.loads(text)
    sent = [r[0] for r in out["runs"]]
    assert len(text) <= 2_000 and 0 < len(sent) < 20
    assert len(cost_reads) <= 2 * len(sent) + 8 < 64
    assert succ(5) not in sent and "no_cost_table" not in out
    assert out["shown"].startswith(f"runs 1–{len(sent)} of 68, newest first; pass offset {len(sent)} for the next")


def test_a_given_detail_wins_over_the_full_rows_of_named_folders(lctx) -> None:
    """Critique M6: named folders get full rows without a detail; ``detail: "scan"`` gives them the compact rows
    with the costs, ``detail: "full"`` the full rows."""
    names = [succ(7), succ(2)]
    assert "columns" not in _call(lctx, folders=names)
    scan = _call(lctx, folders=names, detail="scan")
    assert scan["columns"][4:] == ["total_cost", "penalty_cost", "largest_cost"]
    assert [r[0] for r in scan["runs"]] == [succ(7), succ(2)]                # newest first
    assert [r[4] for r in scan["runs"]] == [245.0, 190.0]
    assert "columns" not in _call(lctx, folders=names, detail="full")


# --- overview ---------------------------------------------------------------------------------------------------

UNIT = "wind_offshore_farm_north_{:02d}"


def test_overview(lctx, cost_reads) -> None:
    """One run in one call (medium caps: 8 entities a class): costs by category, largest signed value first; per
    entity existing (first period), invested (summed), total (last period), never summed over periods or
    entities; the slacks' largest entries.  Before: refused."""
    out = _call(lctx, detail="overview", folders=[NEWEST])
    assert (out["run"], out["state"], out["verdict"], out["results_usable"]) == (NEWEST, "succeeded", "optimal", True)
    assert (out["total_cost"], out["penalty_cost"], out["cost_unit"]) == (1217.75, 302.0, COST_UNIT)
    assert out["cost_split"] == {"columns": ["category", "value"], "rows": [
        ["commodity_cost", 600.0], ["upward slack penalty", 300.0], ["unit investment/retirement", 200.0],
        ["co2", 80.0], ["connection investment/retirement", 40.0], ["storage investment/retirement", 30.0],
        ["fixed cost of existing assets", 10.0], ["other operational", 5.0], ["upward reserve slack penalty", 2.0],
        ["starts", 0.5], ["fixed cost of pre-existing capacity", 0.25], ["commodity_sales", -50.0]]}
    capacity = out["capacity"]
    assert capacity["columns"] == ["entity", "existing", "invested", "total"]
    # Units: the 6 changed (invested 1, 4, 7, 10, 13; divested 16) and the two largest other totals (15: 150,
    # 14: 140), by total; 140 three times: by name.
    assert capacity["unit"] == {
        "table": "unit_capacity_ed_p", "unit": "MW (default convention)",
        "columns": ["entity", "existing", "invested", "total", "divested"],
        "rows": [[UNIT.format(13), 130.0, 100.0, 230.0, None], [UNIT.format(10), 100.0, 100.0, 200.0, None],
                 [UNIT.format(7), 70.0, 100.0, 170.0, None], [UNIT.format(15), 150.0, 0.0, 150.0, None],
                 [UNIT.format(4), 40.0, 100.0, 140.0, None], [UNIT.format(14), 140.0, 0.0, 140.0, None],
                 [UNIT.format(16), 160.0, 0.0, 140.0, 20.0], [UNIT.format(1), 10.0, 100.0, 110.0, None]],
        "shown": "8 of 16; 6 changed"}
    assert [r[0] for r in capacity["connection"]["rows"]] == [f"interconnector_link_{k}_to_{k + 1}"
                                                             for k in range(6, 0, -1)]
    assert "shown" not in capacity["connection"]
    assert capacity["storage"] == {
        "table": "node_capacity_ed_p", "unit": "MWh (default convention)",
        "columns": ["entity", "existing", "invested", "total", "in_construction"],
        "rows": [["pumped_hydro_reservoir_03", 1000.0, 0.0, 1000.0, 0.0],
                 ["battery_storage_west_01", 0.0, 455.5, 455.5, 0.0],
                 ["battery_storage_east_02", 0.0, None, 0.0, 80.0]]}
    assert out["capacity_periods"].startswith("existing: p2020; total, in_construction: p2030; invested, divested: "
                                              "summed over periods")
    assert out["slacks"] == {"columns": ["slack", "table", "unit", "entries", "largest"], "rows": [
        ["vq_state_up", "node_slack_up_d_e", "MWh/a (default convention)", 2, [["west_grid_node"], "p2025", 1520.5]],
        ["vq_reserve", "nodeGroup_slack_reserve_d_eeg", "MWh/a (default convention)", 1,
         [["primary_reserve", "up", "all_nodes_group"], "p2025", 2.25]]]}
    assert out["notes"] == ["Each slack's entries: query_result on its table."]
    # basic caps at 32 768 tokens: 5 entities a class
    small = _with_caps(lctx, resolve_budget("small", 32_768, "loaded", local=True).result_caps)
    unit = _call(small, detail="overview", folders=[NEWEST])["capacity"]["unit"]
    assert [r[0] for r in unit["rows"]] == [UNIT.format(k) for k in (13, 10, 7, 4, 16)]
    assert unit["shown"] == "5 of 16; 6 changed"


def test_the_overview_shortens_in_a_fixed_order(lctx) -> None:
    """Critique M5: at 2 000 characters (basic, 16 384 tokens: 3 entities a class) the overview is cut by its
    steps — categories under 1 % of the costs, entities to 3, slacks to 3, categories to 3, entities to 2 — each
    cut saying what it shows, never the bare ``truncated`` text."""
    caps = resolve_budget("small", 16_384, "loaded", local=True).result_caps
    text = _text(_with_caps(lctx, caps), detail="overview", folders=[NEWEST])
    out = json.loads(text)
    assert len(text) <= 2_000 and "truncated" not in out
    assert out["cost_split"]["rows"] == [["commodity_cost", 600.0], ["upward slack penalty", 300.0],
                                         ["unit investment/retirement", 200.0]]
    assert out["cost_split"]["shown"] == "3 of 12; every one: query_result on costs_discounted_p_"
    assert [r[0] for r in out["capacity"]["unit"]["rows"]] == [UNIT.format(13), UNIT.format(10)]
    assert out["capacity"]["unit"]["shown"] == "2 of 16; 6 changed"
    assert len(out["slacks"]["rows"]) == 2 and out["total_cost"] == 1217.75


def test_the_overview_of_a_run_whose_results_may_not_be_used(lctx, cost_reads) -> None:
    out = _call(lctx, detail="overview", folders=[fail(2)])
    assert out["run"] == fail(2) and out["state"] == "failed" and out["unusable_code"] == "failed"
    assert out["numbers"] == "not shown: the results of this run may not be used"
    assert not {"total_cost", "capacity", "slacks", "cost_split"} & set(out) and cost_reads == []


def test_the_overview_needs_one_run(lctx) -> None:
    latest = ", ".join(f'"{n}"' for n in RECORDED[:5])
    message = ("detail overview needs one result folder: folders with one name, or scenario alone (its newest "
               f"run). The latest folders are: {latest}.")
    for args in ({}, {"folders": [NEWEST, succ(19)]}, {"folders": ["wind*"]}):
        out = call_tool(lctx, "list_runs", {"detail": "overview", **args})
        assert not out.ok and out.message == message, args
    # scenario alone: its newest folder (critique m8: the scenario was run from both sources), the other named
    out = _call(lctx, detail="overview", scenario=SCENARIOS[MANY_NO_RECORD + 19])
    assert out["run"] == NEWEST
    assert out["notes"][0] == (f"The scenario also has the result folders {succ(19)}; this is the newest. Name one "
                               "in folders for its overview.")


def test_the_overview_of_a_run_without_a_record_or_tables(lctx) -> None:
    out = _call(lctx, detail="overview", folders=[NO_RECORD[0]])
    assert out["verified"] is False and out["total_cost"] is None and out["penalty_cost"] is None
    assert out["run_notes"][0].startswith("No run record: the run's success is not known")
    assert out["notes"] == ["This run has no cost table (costs_discounted_p_), so its costs are not shown."]
    assert "capacity" not in out and out["slacks"]["rows"] == []


def test_sizes(lctx) -> None:
    """Upper bounds on what is sent (measured: default 2 095 characters / 1 125 tokens at medium, scan 2 997 /
    1 620, overview 2 684 / 1 396; at basic 16 384 tokens 1 312 / 683, scan 1 767 / 913, overview 1 891 / 949)."""
    basic = _with_caps(lctx, resolve_budget("small", 16_384, "loaded", local=True).result_caps)
    for ctx, args, chars, tokens in (
            (lctx, {"detail": "scan"}, 3_100, 1_700), (lctx, {"detail": "overview", "folders": [NEWEST]}, 2_800, 1_500),
            (basic, {}, 1_400, 750), (basic, {"detail": "scan"}, 1_900, 1_000),
            (basic, {"detail": "overview", "folders": [NEWEST]}, 2_000, 1_050)):
        text = _text(ctx, **args)
        assert len(text) <= chars and approx_tokens(text, "json") <= tokens, (args, len(text))
