"""explain_entity (item 209) on hand-written result tables of runs of ``tests/fixtures/tests.json`` scenarios.

Every expected value below is derived by hand from the frames written here and from the scenario's inputs as
``tests.json`` stores them (stated beside each test), never copied from the tool's output.  The inputs of
coal_plant: multi_year — invest_method invest_period_total, invest_max_total 700, invest_max_period p2020 250,
p2025 350, p2030 400, p2035 0; all — invest_total, invest_max_total 700, retire_max_total 1000; test_a_lot —
invest_no_limit (invest_max_total 700 still stored); coal — no invest_method (the default, not_allowed).  The
database adds a unit named ``west`` (active with the alternative coal) so that ``west`` names a unit and a node.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flextool.assistant.context import make_context
from flextool.assistant.registry import call_tool
from flextool.assistant.results import _entity

from tests.assistant.conftest import FIXTURES, fail_with, make_project, record_run, succeed
from tests.assistant.results_tables import succeed_writing, write_result

PERIODS3 = ["p2020", "p2025", "p2030"]
NAN = np.nan


def _capacity(folder: Path, scenario: str, rows: dict[str, list[tuple]]) -> None:
    """unit_capacity_ed_p: per unit and period (existing, invested, in_construction, divested, total)."""
    index = pd.MultiIndex.from_tuples([(u, p) for u in rows for p in PERIODS3], names=["unit", "period"])
    frame = pd.DataFrame([cell for u in rows for cell in rows[u]], index=index,
                         columns=pd.Index(["existing", "invested", "in_construction", "divested", "total"],
                                          name="parameter"))
    write_result(folder, "unit_capacity_ed_p", frame, scenario)


def _by_period(folder: Path, stem: str, scenario: str, data: dict[tuple, list[float]], names: list[str]) -> None:
    columns = pd.MultiIndex.from_tuples(list(data), names=names) if len(names) > 1 else pd.Index(
        [k[0] for k in data], name=names[0])
    frame = pd.DataFrame(np.array(list(data.values())).T, index=pd.Index(PERIODS3, name="period"), columns=columns)
    write_result(folder, stem, frame, scenario)


def _multi_year_tables(folder: Path) -> None:
    s = "multi_year"
    # coal_plant invests 250 + 350 + 100 = 700 (= invest_max_total); 250 and 350 equal invest_max_period of p2020
    # and p2025, 100 is below p2030's 400.
    _capacity(folder, s, {
        "coal_plant": [(500.0, 250.0, 0.0, NAN, 750.0), (500.0, 350.0, 0.0, NAN, 1100.0),
                       (500.0, 100.0, 0.0, NAN, 1200.0)],
        "wind_plant": [(1000.0, NAN, 0.0, NAN, 1000.0)] * 3,
    })
    _by_period(folder, "unit_outputNode_d_ee", s, {("coal_plant", "west"): [1e6, 1e6, 1e6],
                                                   ("wind_plant", "west"): [2e6, 2e6, 2e6]}, ["unit", "node"])
    _by_period(folder, "unit_inputNode_d_ee", s, {("coal_plant", "coal_market"): [-2.5e6, -2.5e6, -2.5e6]},
               ["unit", "node"])
    _by_period(folder, "unit_outputs_cf_d_ee", s, {("coal_plant", "west"): [0.5, 0.25, 0.75],
                                                   ("wind_plant", "west"): [0.3, 0.3, 0.3]}, ["unit", "sink"])
    # The investment value: positive for coal_plant in p2020 and p2025, negative for wind_plant.
    _by_period(folder, "dual_invest_effective_unit_d_e", s, {("coal_plant",): [5.0, 5.0, 0.0],
                                                             ("wind_plant",): [-3.0, -3.0, -3.0]}, ["entity"])
    # Prices: west 10, 20, 30 in p2020 and 40, 40, 40 in p2025; coal_market 5, 5, 20 in p2020 and 1, 2, 3 in p2025.
    index = pd.MultiIndex.from_tuples([("y2020", p, t) for p in ("p2020", "p2025") for t in ("t01", "t02", "t03")],
                                      names=["solve", "period", "time"])
    prices = pd.DataFrame({"west": [10.0, 20.0, 30.0, 40.0, 40.0, 40.0],
                           "coal_market": [5.0, 5.0, 20.0, 1.0, 2.0, 3.0]}, index=index)
    prices.columns.name = "node"
    write_result(folder, "node_prices_dt_e", prices, s)
    # Costs [M CUR]: coal_plant commodity_cost 100, 80, 60 (total 240), investment 50, 40, 0 (total 90),
    # other operational and retirement 0 in every period; wind_plant other operational 7.
    rows = pd.MultiIndex.from_tuples([(p, u) for p in PERIODS3 for u in ("coal_plant", "wind_plant")],
                                     names=["period", "unit"])
    costs = pd.DataFrame([[100.0, 0.0, 50.0, 0.0], [0.0, 7.0, 0.0, 0.0], [80.0, 0.0, 40.0, 0.0],
                          [0.0, 7.0, 0.0, 0.0], [60.0, 0.0, 0.0, 0.0], [0.0, 7.0, 0.0, 0.0]], index=rows,
                         columns=pd.Index(["commodity_cost", "other operational", "investment", "retirement"],
                                          name="category"))
    write_result(folder, "cost_unit_discounted_d_ec", costs, s)
    slack = pd.DataFrame({"west": [0.0, 12.5, 0.0]}, index=pd.Index(PERIODS3, name="period"))
    slack.columns.name = "node"
    write_result(folder, "node_slack_up_d_e", slack, s)


def _all_tables(folder: Path) -> None:
    # invest_total: 250 + 350 + 100 = 700 = invest_max_total; the period values equal invest_max_period but the
    # method does not use it.  Divested 1000 = retire_max_total, which invest_total does not use either.
    _capacity(folder, "all", {"coal_plant": [(2000.0, 250.0, 0.0, 1000.0, 1250.0), (2000.0, 350.0, 0.0, NAN, 1600.0),
                                             (2000.0, 100.0, 0.0, NAN, 1700.0)]})


def _test_a_lot_tables(folder: Path) -> None:
    # invest_no_limit: 300 + 400 = 700 equals the stored invest_max_total, which the method ignores.
    _capacity(folder, "test_a_lot", {"coal_plant": [(500.0, 300.0, 0.0, NAN, 800.0), (500.0, 400.0, 0.0, NAN, 1200.0),
                                                    (500.0, 0.0, 0.0, NAN, 1200.0)]})
    _by_period(folder, "dual_invest_effective_unit_d_e", "test_a_lot", {("coal_plant",): [0.0, 0.0, 0.0]},
               ["entity"])


def _coal_tables(folder: Path) -> None:
    _capacity(folder, "coal", {"coal_plant": [(500.0, NAN, 0.0, NAN, 500.0)] * 3})


def _db_with_west_unit(root: Path) -> Path:
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    data = json.loads((FIXTURES / "tests.json").read_text(encoding="utf-8"))
    data["entities"].append(["unit", "west", ""])
    data["entity_alternatives"].append(["unit", ["west"], "coal", True])
    path = root / "case.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    migrate_database(json_to_db(path, root / "tests.sqlite"))
    return root / "tests.sqlite"


@pytest.fixture(scope="module")
def explain_project(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("explain")
    project = make_project(root / "p", _db_with_west_unit(root))
    record_run(project, "multi_year_1", "multi_year", succeed_writing(_multi_year_tables))
    record_run(project, "all_1", "all", succeed_writing(_all_tables))
    record_run(project, "test_a_lot_1", "test_a_lot", succeed_writing(_test_a_lot_tables))
    record_run(project, "coal_1", "coal", succeed_writing(_coal_tables))
    record_run(project, "coal_chp_1", "coal_chp", succeed)
    record_run(project, "coal_chp_1", "coal_chp", fail_with("solve y2020 is infeasible"))
    return project


@pytest.fixture
def ectx(explain_project: Path):
    return make_context(explain_project)


def _call(ctx, **args) -> dict:
    out = call_tool(ctx, "explain_entity", args)
    assert out.ok, out.message
    assert len(out.text) <= int(args.get("max_chars") or ctx.caps.max_chars)
    return json.loads(out.text)


def _refused(ctx, **args) -> str:
    out = call_tool(ctx, "explain_entity", args)
    assert not out.ok, out.text
    return out.message


def _hints_with(payload: dict, text: str) -> list[str]:
    return [h for h in payload["hints"] if text in h]


# ---------------------------------------------------------------------------------------------------------
# Critique 1 and 8: capacity rows and the hints


def test_capacity_rows_and_the_period_and_total_hints(ectx) -> None:
    out = _call(ectx, entity="coal_plant", run="multi_year_1")
    assert out["entity_class"] == "unit" and out["scenario"] == "multi_year"
    cap = out["capacity"]
    assert cap["table"] == "unit_capacity_ed_p" and cap["unit"] == "MW"
    assert cap["columns"] == ["period", "existing", "invested", "divested", "in_construction", "total"]
    assert cap["rows"] == [["p2020", 500.0, 250.0, None, 0.0, 750.0], ["p2025", 500.0, 350.0, None, 0.0, 1100.0],
                           ["p2030", 500.0, 100.0, None, 0.0, 1200.0]]
    # invest_period_total uses both limits: the total (700 = 250 + 350 + 100) and the periods p2020 (250) and
    # p2025 (350); p2030 (100 of 400) does not bind.  p2035's 0 is not a period of the run.
    assert _hints_with(out, "invest_max_total") == [
        "Invested 700 MW over the periods, equal to invest_max_total (700): consistent with the total investment "
        "limit binding."]
    period = _hints_with(out, "invest_max_period")
    assert period == [
        "In p2020 invested 250 MW, equal to invest_max_period (250): consistent with the period investment limit "
        "binding.",
        "In p2025 invested 350 MW, equal to invest_max_period (350): consistent with the period investment limit "
        "binding."]


def test_total_limit_without_a_period_hint_and_an_unused_retire_limit(ectx) -> None:
    out = _call(ectx, entity="coal_plant", scenario="all")
    assert out["run"] == "all_1"
    assert len(_hints_with(out, "equal to invest_max_total")) == 1
    # invest_total uses neither the stored period limit nor the retirement limit (divested 1000 = it): said first.
    assert out["hints"][:2] == [
        "invest_max_period is set (a map) but not used: invest_method is invest_total, and only "
        "invest_period, invest_period_total, invest_retire_period, invest_retire_period_total use it.",
        "retire_max_total is set (1000) but not used: invest_method is invest_total, and only "
        "retire_total, retire_period_total, invest_retire_total, invest_retire_period_total use it."]
    assert _hints_with(out, "equal to invest_max_period") == [] and _hints_with(out, "equal to retire_max") == []
    assert out["capacity"]["rows"][0] == ["p2020", 2000.0, 250.0, 1000.0, 0.0, 1250.0]


def test_no_limit_method_gives_no_hint_and_zero_value_none(ectx) -> None:
    """invest_no_limit uses no limit: the stored invest_max_period (a map) and invest_max_total 700 are said to
    be unused, and nothing binds."""
    out = _call(ectx, entity="coal_plant", run="test_a_lot_1")
    assert out["hints"] == [
        "invest_max_period is set (a map) but not used: invest_method is invest_no_limit, and only "
        "invest_period, invest_period_total, invest_retire_period, invest_retire_period_total use it.",
        "invest_max_total is set (700) but not used: invest_method is invest_no_limit, and only "
        "invest_total, invest_period_total, invest_retire_total, invest_retire_period_total use it."]
    assert out["investment_value"]["rows"] == [["p2020", 0.0], ["p2025", 0.0], ["p2030", 0.0]]


def test_default_not_allowed_hint_and_the_default_row(ectx) -> None:
    out = _call(ectx, entity="coal_plant", run="coal_1")
    assert out["hints"] == ["Nothing was invested: invest_method is not_allowed, which allows no investment."]
    rows = {r[0]: r for r in out["inputs"]["rows"]}
    assert rows["invest_method"] == ["invest_method", "not_allowed", None, "default"]
    assert rows["existing"] == ["existing", 500.0, "MW", "coal"]


def test_existing_not_set_and_a_name_of_two_classes(ectx) -> None:
    message = _refused(ectx, entity="west", run="coal_1")
    assert "several classes" in message and '"node"' in message and '"unit"' in message
    out = _call(ectx, entity="west", entity_class="unit", run="coal_1")
    rows = {r[0]: r for r in out["inputs"]["rows"]}
    assert rows["existing"] == ["existing", None, "MW", "not set"]
    assert rows["invest_method"][1:] == ["not_allowed", None, "default"]


def test_retire_and_minimum_hints() -> None:
    """The hint rules alone (critique 8, 9) on hand-made blocks."""
    capacity = {"unit": "MW", "columns": ["period", "existing", "invested", "divested", "in_construction", "total"],
                "rows": [["p1", 100.0, 40.0, 30.0, 0.0, 110.0], ["p2", 100.0, 60.0, None, 0.0, 170.0]]}
    inputs = {"invest_method": "invest_retire_period_total", "invest_min_total": 100.0, "retire_max_total": 30.0,
              "retire_max_period": {"type": "map"}}
    got = _entity.hints(inputs, {"capacity": capacity}, [])
    assert ("Invested 100 MW over the periods, equal to invest_min_total (100): consistent with the minimum "
            "investment forcing it.") in got
    assert ("Retired 30 MW over the periods, equal to retire_max_total (30): consistent with the total retirement "
            "limit binding.") in got
    # Capacity in construction: no total hint (the total counts what came online only).
    building = {**capacity, "rows": [[*capacity["rows"][0][:4], 5.0, 110.0], capacity["rows"][1]]}
    assert not [h for h in _entity.hints({"invest_method": "invest_total", "invest_max_total": 100.0},
                                         {"capacity": building}, []) if "invest_max_total" in h]
    # A period limit of 0 is said; retire_* methods allow no investment.
    zero = _entity.hints({"invest_method": "invest_period", "invest_max_period": 0.0}, {"capacity": capacity}, [])
    assert "In p1 invest_max_period is 0: investment is not allowed in that period." in zero
    assert _entity.hints({"invest_method": "retire_total"}, {"capacity": {**capacity, "rows": [
        ["p1", 100.0, None, 10.0, 0.0, 90.0]]}}, []) == [
        "Nothing was invested: invest_method is retire_total, which allows retirement only."]


# ---------------------------------------------------------------------------------------------------------
# Flows, prices, costs, investment value


def test_flows_per_node_never_summed(ectx) -> None:
    out = _call(ectx, entity="coal_plant", run="multi_year_1")
    # scaled (B1 T6): 1 000 000 MWh/a is ≈1 TWh/a.
    assert out["output_flows"] == {"table": "unit_outputNode_d_ee", "unit": "MWh/a", "columns": ["period", "node",
                                   "value", "scaled"], "rows": [["p2020", "west", 1e6, "≈1 TWh/a"],
                                                                ["p2025", "west", 1e6, "≈1 TWh/a"],
                                                                ["p2030", "west", 1e6, "≈1 TWh/a"]]}
    assert out["input_flows"]["rows"][0] == ["p2020", "coal_market", -2.5e6, "≈-2.5 TWh/a"]
    assert out["capacity_factor"]["rows"] == [["p2020", "west", 0.5], ["p2025", "west", 0.25],
                                              ["p2030", "west", 0.75]]
    assert _entity.TEXTS["annualized_note"] in out["notes"]


def test_prices_mean_min_max_per_node_and_period(ectx) -> None:
    out = _call(ectx, entity="coal_plant", run="multi_year_1")
    prices = out["prices"]
    assert prices["table"] == "node_prices_dt_e" and prices["unit"] == "CUR/MWh"
    # west p2020: (10 + 20 + 30) / 3 = 20, min 10, max 30; p2025 all 40.  coal_market p2020: 30 / 3 = 10, 5, 20;
    # p2025: 6 / 3 = 2, 1, 3.  Nodes: the output node first, then the input node (the unit's links).
    assert prices["rows"] == [["west", "p2020", 20.0, 10.0, 30.0], ["west", "p2025", 40.0, 40.0, 40.0],
                              ["coal_market", "p2020", 10.0, 5.0, 20.0], ["coal_market", "p2025", 2.0, 1.0, 3.0]]
    assert "unweighted" in prices["note"]


def test_costs_are_the_non_zero_categories(ectx) -> None:
    out = _call(ectx, entity="coal_plant", run="multi_year_1")
    costs = out["costs"]
    assert costs["table"] == "cost_unit_discounted_d_ec" and costs["unit"] == "M CUR"
    assert costs["columns"] == ["category", "total", *PERIODS3]
    # commodity_cost 100 + 80 + 60 = 240; investment 50 + 40 + 0 = 90; the zero categories are left out.
    assert costs["rows"] == [["commodity_cost", 240.0, 100.0, 80.0, 60.0], ["investment", 90.0, 50.0, 40.0, 0.0]]
    assert "cost_unit_annualized_d_ec" in costs["note"]


def test_investment_value_hints(ectx) -> None:
    out = _call(ectx, entity="coal_plant", run="multi_year_1")
    assert out["investment_value"]["unit"] == "CUR/kW"
    assert _hints_with(out, "investment value") == [
        "The investment value is positive in p2020, p2025 (dual_invest_effective_unit_d_e): consistent with an upper "
        "limit holding the investment back (the entity's own, a group's (wind_coal) or a cumulative one)."]
    wind = _call(ectx, entity="wind_plant", run="multi_year_1")
    assert _hints_with(wind, "investment value") == [
        "The investment value is negative in p2020, p2025, p2030 (dual_invest_effective_unit_d_e): consistent with "
        "more capacity not paying off (or a minimum investment forcing more than pays off)."]


def test_a_plain_node_has_prices_slacks_and_no_feeders(ectx) -> None:
    out = _call(ectx, entity="west", entity_class="node", run="multi_year_1")
    assert out["prices"]["rows"] == [["west", "p2020", 20.0, 10.0, 30.0], ["west", "p2025", 40.0, 40.0, 40.0]]
    assert out["slack_up"]["rows"] == [["p2020", 0.0], ["p2025", 12.5], ["p2030", 0.0]]
    assert "output_flows" not in out
    linked = {link["entity_class"] for link in out.get("links", [])} | set(out.get("memberships", {}))
    assert not any(cls.startswith(("unit__", "connection__")) for cls in linked)
    assert "node_slack_down_d_e" in out["not_in_run"] and "cost_node_discounted_d_ec" in out["not_in_run"]


# ---------------------------------------------------------------------------------------------------------
# Inputs


def test_inputs_equal_get_parameter(ectx) -> None:
    out = _call(ectx, entity="coal_plant", run="multi_year_1")
    rows = {r[0]: r for r in out["inputs"]["rows"]}
    names = ["invest_method", "invest_max_total", "invest_cost", "existing", "efficiency", "lifetime"]
    cells = call_tool(ectx, "get_parameter", {"entity_class": "unit", "entities": [["coal_plant"]],
                                              "parameters": names, "scenario": "multi_year"})
    table = json.loads(cells.text)
    columns = table["columns"]
    for row in table["rows"]:
        cell = dict(zip(columns, row))
        assert rows[cell["parameter"]][1] == cell["value"], cell["parameter"]
        assert rows[cell["parameter"]][3] == cell["from_alternative"], cell["parameter"]
        assert rows[cell["parameter"]][2] == (cell.get("unit") or table["units"][cell["parameter"]])
    links = {link["entity_class"]: link for link in out["links"]}
    assert links["unit__outputNode"]["entity"] == ["coal_plant", "west"]
    assert out["memberships"]["group__unit"] == [["wind_coal", "coal_plant"]]


# ---------------------------------------------------------------------------------------------------------
# Refusals


def test_a_failed_latest_run_is_refused_as_query_result_refuses_it(ectx) -> None:
    message = _refused(ectx, entity="coal_plant", scenario="coal_chp")
    expected = call_tool(ectx, "query_result", {"run": "coal_chp_1", "table": "node_d_ee"})
    assert not expected.ok and message == expected.message


def test_unknown_names_other_scenarios_and_other_classes(ectx) -> None:
    message = _refused(ectx, entity="coal_plnt", run="multi_year_1")
    assert "No entity named" in message and '"coal_plant"' in message
    message = _refused(ectx, entity="coal_plant", run="multi_year_1", scenario="all")
    assert message.startswith('The result folder "multi_year_1" is a run of scenario "multi_year", not of "all".')
    message = _refused(ectx, entity="wind_coal", run="multi_year_1")
    assert "not a unit, connection or node" in message and '"group"' in message
    message = _refused(ectx, entity="coal_plant")
    assert message.startswith("explain_entity needs a result folder") and '"multi_year_1"' in message
    message = _refused(ectx, entity="coal_plant", scenario="no_such_scenario")
    assert message.startswith('No result folder is a run of scenario "no_such_scenario".')


# ---------------------------------------------------------------------------------------------------------
# The connection's default scenario (smoke check 2026-11-12: the context block said coal, the call was refused)


def test_without_run_and_scenario_the_connections_default_scenario_is_used(explain_project: Path) -> None:
    for ctx in (make_context(explain_project, scenario="coal"),
                make_context(explain_project, input_source="tests.sqlite", scenario="coal")):
        out = _call(ctx, entity="coal_plant")
        assert out["scenario"] == "coal" and out["run"] == "coal_1"
        assert out == _call(make_context(explain_project), entity="coal_plant", scenario="coal")
    # A run or a scenario of the call wins over the default.
    ctx = make_context(explain_project, scenario="coal")
    assert _call(ctx, entity="coal_plant", run="multi_year_1")["scenario"] == "multi_year"
    assert _call(ctx, entity="coal_plant", scenario="all")["scenario"] == "all"
    # A default scenario without a run is refused as the same scenario named in the call.
    message = _refused(make_context(explain_project, scenario="no_such_scenario"), entity="coal_plant")
    assert message == _refused(make_context(explain_project), entity="coal_plant", scenario="no_such_scenario")


def test_the_default_scenario_of_a_default_source_never_applies_to_another_source(explain_project: Path,
                                                                                tmp_path: Path) -> None:
    import shutil

    project = tmp_path / explain_project.name
    shutil.copytree(explain_project, project)
    shutil.copy(project / "input_sources" / "tests.sqlite", project / "input_sources" / "other.sqlite")
    (project / "settings.yaml").write_text("input_source_numbers:\n  tests.sqlite: 1\n  other.sqlite: 2\n",
                                           encoding="utf-8")
    # With the default source, its default scenario applies to it (named or not), never to the other source.
    ctx = make_context(project, input_source="tests.sqlite", scenario="coal")
    assert _call(ctx, entity="coal_plant")["run"] == "coal_1"
    assert _call(ctx, entity="coal_plant", source="tests")["run"] == "coal_1"
    assert _refused(ctx, entity="coal_plant", source="other").startswith("explain_entity needs a result folder")
    # Without a default source, the default scenario applies to a source that has a run of it.
    ctx = make_context(project, scenario="coal")
    assert _call(ctx, entity="coal_plant", source="1")["run"] == "coal_1"
    assert _refused(ctx, entity="coal_plant", source="other").startswith("explain_entity needs a result folder")


def test_call_explain_entity_with_the_scenario_option(explain_project: Path, monkeypatch: pytest.MonkeyPatch,
                                                      capsys: pytest.CaptureFixture) -> None:
    from flextool.assistant import cli, store

    monkeypatch.setattr(store, "purge_at_start", lambda: None)
    assert cli.main(["--project-path", str(explain_project), "--scenario", "coal", "--call", "explain_entity",
                     '{"entity": "coal_plant"}']) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["scenario"] == "coal" and out["run"] == "coal_1"
    assert cli.main(["--project-path", str(explain_project), "--call", "explain_entity",
                     '{"entity": "coal_plant"}']) == 1
    assert "explain_entity needs a result folder" in capsys.readouterr().out


#: Why coal_chp is not part of multi_year: its only entity_alternative is in alternative coal_chp (tests.json),
#: and units are inactive by default.
NOT_MADE_ACTIVE = ("no alternative of the scenario makes it active, and unit entities are inactive unless an "
                   "alternative makes them active")


def test_an_inactive_entity_is_noted_with_its_tables_not_in_run(ectx) -> None:
    """B1, T3: the note that the entity is not part of the scenario comes first; the run has none of its results
    (every table of the run lacks coal_chp), so the note also says it has none."""
    out = _call(ectx, entity="coal_chp", run="multi_year_1")
    assert out["notes"][0] == (f'"coal_chp" is not part of scenario "multi_year" ("{NOT_MADE_ACTIVE}"). Its stored '
                               "values are not used there: it has no capacity, flows or results in this scenario.")
    assert {"unit_capacity_ed_p", "unit_outputNode_d_ee", "cost_unit_discounted_d_ec"} <= set(out["not_in_run"])


def test_an_entity_inactive_in_the_current_data_with_results_in_the_run(ectx, monkeypatch) -> None:
    """Whether the entity is part of the scenario is read from the current input data; when the run has results
    of it (coal_plant's capacity in multi_year_1), the note does not say it has none."""
    from flextool.assistant.inputs import _db

    monkeypatch.setattr(_db.ScenarioView, "active",
                        lambda self, cls, byname: (False, "it is set inactive in alternative 'x'")
                        if byname == ("coal_plant",) else (True, None))
    out = _call(ectx, entity="coal_plant", run="multi_year_1")
    assert "capacity" in out
    assert out["notes"][0] == ('"coal_plant" is not part of scenario "multi_year" ("it is set inactive in '
                               "alternative 'x'\"). Its stored values are not used there.")


# ---------------------------------------------------------------------------------------------------------
# Critique 10: cuts


def _sent(ctx, max_chars: int) -> dict:
    out = call_tool(ctx, "explain_entity", {"entity": "coal_plant", "run": "multi_year_1", "max_chars": max_chars})
    assert out.ok and len(out.text) <= max_chars
    return json.loads(out.text)


def test_the_small_preset_fits_the_whole_result(explain_project: Path) -> None:
    """6 000 characters (the small preset's result) hold coal_plant's whole result: nothing is cut."""
    payload = _sent(make_context(explain_project), 6_000)
    whole = _call(make_context(explain_project), entity="coal_plant", run="multi_year_1")
    assert payload == whole
    assert payload["costs"]["columns"] == ["category", "total", *PERIODS3]
    assert not any(n in payload["notes"] for n in (_entity.TEXTS["periods_cut"], _entity.TEXTS["inputs_cut"]))


def test_a_cut_keeps_valid_json_within_the_limit(explain_project: Path) -> None:
    # One character less than the whole: costs as totals (the first cut that changes this result: it has no
    # related tables), still every block.
    whole = call_tool(make_context(explain_project), "explain_entity", {"entity": "coal_plant", "run": "multi_year_1"})
    payload = _sent(make_context(explain_project), len(whole.text) - 1)
    assert payload["costs"]["columns"] == ["category", "total"]
    assert payload["capacity"]["table"] == "unit_capacity_ed_p" and payload["hints"]
    # 2 000 characters: the final step (header, notes, capacity, hints), saying what was left out.
    payload = _sent(make_context(explain_project), 2_000)
    assert set(payload) == {"run", "entity", "entity_class", "source", "scenario", "capacity", "hints", "notes"}
    assert len(payload["hints"]) == 4
    final = [n for n in payload["notes"] if n.startswith("The result was shortened to fit")]
    assert final and "costs (cost_unit_discounted_d_ec)" in final[0]
    # The capacity of the first and last periods, said so.
    assert [r[0] for r in payload["capacity"]["rows"]] == ["p2020", "p2030"]
    assert _entity.TEXTS["periods_cut"] in payload["notes"]


def test_max_rows_caps_each_block(ectx) -> None:
    out = _call(ectx, entity="coal_plant", run="multi_year_1", max_rows=2)
    assert out["capacity"]["rows"] == [["p2020", 500.0, 250.0, None, 0.0, 750.0],
                                       ["p2025", 500.0, 350.0, None, 0.0, 1100.0]]
    assert out["capacity"]["shown"] == "2 of 3 rows; query_result on unit_capacity_ed_p gives them all."


# ---------------------------------------------------------------------------------------------------------
# Critique 11: a node's unit_of_measure


@pytest.fixture(scope="module")
def uom_explain_ctx(tmp_path_factory):
    from flextool.lean_parquet import write_lean_parquet
    from flextool.process_outputs._output_meta import UnitContext, derive_column_meta
    from tests.assistant.test_unit_of_measure_tools import SCENARIO, _b64  # noqa: F401 — the uom scenario

    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    root = tmp_path_factory.mktemp("uom_explain")
    data = json.loads((FIXTURES / "tests.json").read_text(encoding="utf-8"))
    data["alternatives"].append(["uom", "water in m3"])
    data["parameter_values"].append(["node", "water", "unit_of_measure", _b64("m3"), "uom"])
    base = [sa for sa in data["scenario_alternatives"] if sa[0] == "desal_water_storage"]
    chain = [[SCENARIO, alt, before] for _s, alt, before in base]
    chain[-1][2] = "uom"
    chain.append([SCENARIO, "uom", None])
    data["scenario_alternatives"] += chain
    data["scenarios"].append([SCENARIO, False, ""])
    (root / "case.json").write_text(json.dumps(data), encoding="utf-8")
    migrate_database(json_to_db(root / "case.json", root / "tests.sqlite"))
    ctx = UnitContext(node_units={"water": "m3"}, capacity_nodes={"med_desal": ("water",)})

    def tables(folder: Path) -> None:
        # med_desal into water (m3) and, for this test, into west (MWh): the flows are per node, each in its unit.
        cols = pd.MultiIndex.from_tuples([("med_desal", "water"), ("med_desal", "west")], names=["unit", "node"])
        flows = pd.DataFrame([[30.0, 4.0]] * 3, index=pd.Index(PERIODS3, name="period"), columns=cols)
        meta = {k: v.as_dict() for k, v in derive_column_meta("unit_outputNode_d_ee", flows.columns, ctx=ctx).items()}
        folder.mkdir(parents=True, exist_ok=True)
        write_lean_parquet(pd.concat({SCENARIO: flows}, axis=1, names=["scenario"]),
                           folder / "unit_outputNode_d_ee.parquet", column_metadata=meta)

    project = make_project(root / "p", root / "tests.sqlite")
    record_run(project, f"{SCENARIO}_1", SCENARIO, succeed_writing(tables))
    return make_context(project)


def test_per_node_units_per_row(uom_explain_ctx) -> None:
    out = _call(uom_explain_ctx, entity="med_desal", run="desal_uom_1")
    flows = out["output_flows"]
    # Two units: the block has none, each row its node's (m3 per year for water, MWh per year for west).
    assert flows["unit"] is None and flows["columns"] == ["period", "node", "value", "unit"]
    assert flows["rows"][:2] == [["p2020", "water", 30.0, "m3/a"], ["p2020", "west", 4.0, "MWh/a"]]
    water = _call(uom_explain_ctx, entity="water", run="desal_uom_1")
    rows = {r[0]: r for r in water["inputs"]["rows"]}
    assert rows["unit_of_measure"][1:] == ["m3", None, "uom"]


# ---------------------------------------------------------------------------------------------------------
# Levels


def test_not_offered_at_basic(explain_project: Path) -> None:
    from flextool.assistant import registry

    assert not registry.all_tools()["explain_entity"].small_tier
    assert "explain_entity" not in {t.name for t in registry.tools_for_tier("small")}
    assert "explain_entity" in {t.name for t in registry.tools_for_tier("medium")}
    assert "explain_entity" in {t.name for t in registry.tools_for_tier("large")}


# ---------------------------------------------------------------------------------------------------------
# Fix pass (review of item 209)


_CAPACITY = {"unit": "MW", "columns": ["period", "existing", "invested", "divested", "in_construction", "total"],
             "rows": [["p1", 100.0, 40.0, None, 0.0, 140.0], ["p2", 100.0, 60.0, None, 0.0, 200.0]]}


def test_a_minimum_hints_only_under_the_methods_that_use_it() -> None:
    """The engine builds minInvest_entity_total over the INVEST_TOTAL methods and minInvest_entity_period over
    the INVEST_PERIOD ones (engine/model/_cumulative_invest.py): invested 40 + 60 = 100 equals the stored
    invest_min_total, and p1's 40 the stored invest_min_period."""
    stored = {"invest_min_total": 100.0, "invest_min_period": 40.0}
    unused_total = ("invest_min_total is set (100) but not used: invest_method is {method}, and only "
                    "invest_total, invest_period_total, invest_retire_total, invest_retire_period_total use it.")
    unused_period = ("invest_min_period is set (40) but not used: invest_method is {method}, and only "
                     "invest_period, invest_period_total, invest_retire_period, invest_retire_period_total use it.")
    assert _entity.hints({"invest_method": "invest_no_limit", **stored}, {"capacity": _CAPACITY}, []) == [
        unused_period.format(method="invest_no_limit"), unused_total.format(method="invest_no_limit")]
    period_only = _entity.hints({"invest_method": "invest_period", **stored}, {"capacity": _CAPACITY}, [])
    assert period_only == [unused_total.format(method="invest_period"),
                           "In p1 invested 40 MW, equal to invest_min_period (40): consistent with the minimum "
                           "investment forcing it."]
    total_only = _entity.hints({"invest_method": "invest_total", **stored}, {"capacity": _CAPACITY}, [])
    assert total_only == [unused_period.format(method="invest_total"),
                          "Invested 100 MW over the periods, equal to invest_min_total (100): consistent with the "
                          "minimum investment forcing it."]


def test_construction_only_holds_back_the_investment_total_hint() -> None:
    # 5 MW in construction in p1: the invest_max_total hint is left out, the minimum and retirement ones are not.
    building = {**_CAPACITY, "rows": [["p1", 100.0, 40.0, 30.0, 5.0, 110.0], ["p2", 100.0, 60.0, None, 0.0, 170.0]]}
    got = _entity.hints({"invest_method": "invest_retire_total", "invest_max_total": 100.0, "invest_min_total": 100.0,
                         "retire_max_total": 30.0}, {"capacity": building}, [])
    assert got == ["Invested 100 MW over the periods, equal to invest_min_total (100): consistent with the minimum "
                   "investment forcing it.",
                   "Retired 30 MW over the periods, equal to retire_max_total (30): consistent with the total "
                   "retirement limit binding."]


def test_a_capacity_without_a_unit_has_no_double_space() -> None:
    got = _entity.hints({"invest_method": "invest_total", "invest_max_total": 100.0},
                        {"capacity": {**_CAPACITY, "unit": None}}, [])
    assert got == ["Invested 100 over the periods, equal to invest_max_total (100): consistent with the total "
                   "investment limit binding."]


def test_a_negative_investment_value_hints_only_where_nothing_was_invested() -> None:
    # p1 invested 40 (built: a non-zero value there may be an artefact), p2 nothing (None), p3 0.
    capacity = {**_CAPACITY, "rows": [["p1", 100.0, 40.0, None, 0.0, 140.0], ["p2", 100.0, None, None, 0.0, 140.0],
                                      ["p3", 100.0, 0.0, None, 0.0, 140.0]]}
    value = {"table": "dual_invest_effective_unit_d_e", "rows": [["p1", -2.0], ["p2", -2.0], ["p3", -2.0]]}
    got = _entity.hints({"invest_method": "invest_no_limit"}, {"capacity": capacity, "investment_value": value}, [])
    assert got == ["The investment value is negative in p2, p3 (dual_invest_effective_unit_d_e): consistent with "
                   "more capacity not paying off (or a minimum investment forcing more than pays off)."]


@pytest.mark.parametrize("max_chars", [1000, 600])
def test_the_final_steps_shrink_until_the_result_fits(explain_project: Path, max_chars: int) -> None:
    from flextool.assistant.unit_wording import SESSION_NOTES, STORED_VALUES

    ctx = make_context(explain_project)
    out = call_tool(ctx, "explain_entity", {"entity": "coal_plant", "run": "multi_year_1", "max_chars": max_chars})
    assert out.ok and len(out.text) <= max_chars
    payload = json.loads(out.text)
    assert "truncated" not in payload and "group_by" not in out.text and "top_n" not in out.text
    final = [n for n in payload["notes"] if n.startswith("The result was shortened to fit")]
    assert len(final) == 1
    assert payload["run"] == "multi_year_1" and payload["entity"] == "coal_plant"
    # The session note on stored values is in flight only when the sent payload carries it in full.
    carried = SESSION_NOTES[STORED_VALUES] in payload["notes"]
    assert (STORED_VALUES in ctx.notes_in_flight) is carried
    if max_chars == 1000:
        # The capacity's total of the first and last periods, the hints capped (of 4), only the note on what was
        # left out.
        assert payload["capacity"]["columns"] == ["period", "total"]
        assert payload["capacity"]["rows"] == [["p2020", 750.0], ["p2030", 1200.0]]
        assert payload["hints"][-1].endswith("more hints; a larger max_chars shows them.")
        assert "costs (cost_unit_discounted_d_ec)" in final[0] and "the other notes" in final[0]
        assert payload["notes"] == final
    else:
        # Only the note on what was left out, naming the capacity and the hints too.
        assert payload["notes"] == final and "capacity" not in payload and "hints" not in payload
        assert "capacity (unit_capacity_ed_p)" in final[0] and "hints" in final[0]
        assert not carried


def test_a_tiny_limit_never_gives_query_advice(explain_project: Path) -> None:
    out = call_tool(make_context(explain_project), "explain_entity",
                    {"entity": "coal_plant", "run": "multi_year_1", "max_chars": 120})
    assert out.ok and len(out.text) <= 120
    assert "group_by" not in out.text and "top_n" not in out.text
    json.loads(out.text)


def test_the_whole_result_records_its_session_notes(explain_project: Path) -> None:
    from flextool.assistant.unit_wording import SESSION_NOTES, STORED_VALUES

    ctx = make_context(explain_project)
    payload = _call(ctx, entity="coal_plant", run="multi_year_1")
    assert SESSION_NOTES[STORED_VALUES] in payload["notes"] and STORED_VALUES in ctx.notes_in_flight


def _large_prices(folder: Path, n_steps: int) -> pd.DataFrame:
    """node_prices_dt_e of two periods with *n_steps* time steps each, nodes west and east (deterministic)."""
    times = [f"t{i:06d}" for i in range(n_steps)]
    index = pd.MultiIndex.from_tuples([("y2020", p, t) for p in ("p2020", "p2025") for t in times],
                                      names=["solve", "period", "time"])
    steps = np.arange(2 * n_steps, dtype=float)
    frame = pd.DataFrame({"west": (steps * 7.0) % 101.0, "east": (steps * 3.0) % 53.0 + 1.0}, index=index)
    frame.columns.name = "node"
    write_result(folder, "node_prices_dt_e", frame, "multi_year")
    return frame


def test_prices_of_a_large_table_are_reduced_lazily(tmp_path: Path, monkeypatch) -> None:
    """200 000 rows: the prices read through a lazy scan (read_wide, which loads the row levels whole, is never
    called) and equal pandas' own per-period mean, minimum and maximum of the frame written."""
    from flextool.assistant.results import _query

    frame = _large_prices(tmp_path, 100_000)

    def no_eager(*_a, **_k):
        raise AssertionError("read_wide called: the table was loaded whole")

    monkeypatch.setattr(_query, "read_wide", no_eager)
    got = _entity.read_prices(tmp_path, ["west"])
    by_period = frame["west"].groupby(level="period", sort=False)
    expected = [["west", p, float(m), float(lo), float(hi)]
                for p, m, lo, hi in zip(by_period.mean().index, by_period.mean(), by_period.min(), by_period.max())]
    assert [r[:2] for r in got["rows"]] == [["west", "p2020"], ["west", "p2025"]]
    for row, want in zip(got["rows"], expected):
        assert row[2:] == pytest.approx(want[2:], rel=1e-12)


def test_the_lazy_reduction_equals_the_eager_one_on_a_small_table(tmp_path: Path) -> None:
    """The same numbers as the eager reduction (read_wide, then polars per period) on a small table."""
    import polars as pl

    from flextool.assistant.results import _query

    _large_prices(tmp_path, 50)
    got = _entity.read_prices(tmp_path, ["west", "east"])
    table = _query.open_table(tmp_path, "node_prices_dt_e", where="t")
    want = []
    for node in ("west", "east"):
        physical = _entity._entity_columns(table, "node", node)[0][0]
        wide = _query.read_wide(table, [physical])
        stats = wide.group_by("period", maintain_order=True).agg(
            pl.col(physical).mean(), pl.col(physical).min().alias("lo"), pl.col(physical).max().alias("hi"))
        want += [[node, *row] for row in stats.iter_rows()]
    assert got["rows"] == want


def test_a_table_over_the_cell_limit_is_left_out_with_a_note(ectx, monkeypatch) -> None:
    from flextool.assistant.results import _query

    # multi_year_1's node_prices_dt_e has 6 rows; a limit of 5 cells puts west's column (6 cells) over it.
    monkeypatch.setattr(_query, "LONG_CELLS_MAX", 5)
    out = _call(ectx, entity="west", entity_class="node", run="multi_year_1")
    assert "prices" not in out and out["too_large"] == ["node_prices_dt_e"]
    assert "node_prices_dt_e" not in out["not_in_run"]
    assert _entity.TEXTS["too_large"] in out["notes"]


def test_an_entity_gone_from_changed_inputs_is_explained_from_the_results(explain_project: Path,
                                                                          tmp_path: Path) -> None:
    """The input data changed after the run and no longer has the unit: its class comes from the run's results,
    with the "input data changed" note, instead of entity_not_found."""
    import os
    import time

    project = make_project(tmp_path / "p", explain_project / "input_sources" / "tests.sqlite")

    def tables(folder: Path) -> None:
        _capacity(folder, "multi_year", {"gone_unit": [(10.0, NAN, 0.0, NAN, 10.0)] * 3})

    record_run(project, "multi_year_1", "multi_year", succeed_writing(tables))
    db = project / "input_sources" / "tests.sqlite"
    later = time.time() + 3600
    os.utime(db, (later, later))
    out = _call(make_context(project), entity="gone_unit", run="multi_year_1")
    assert out["entity_class"] == "unit" and out["capacity"]["rows"][0] == ["p2020", 10.0, None, None, 0.0, 10.0]
    assert any(n.startswith("The input data changed after this run") for n in out["notes"])
    assert "inputs" not in out
    # Without the change, a name the input data does not have is refused as before.
    os.utime(db, (time.time() - 7200, time.time() - 7200))
    assert "No entity named" in _refused(make_context(project), entity="gone_unit", run="multi_year_1")


def test_one_scan_of_the_runs_per_call(ectx, monkeypatch) -> None:
    from flextool.assistant.results import _runs

    calls = []
    real = _runs.runs
    monkeypatch.setattr(_runs, "runs", lambda ctx: calls.append(1) or real(ctx))
    _call(ectx, entity="coal_plant", scenario="all")
    assert len(calls) == 1
    calls.clear()
    _call(ectx, entity="coal_plant", run="multi_year_1")
    assert len(calls) == 1


def test_an_unused_limit_is_said_first_under_every_method_that_ignores_it() -> None:
    """B1 T4: cumulative_limits allows investment but uses no invest_max_*; not_allowed uses none; a cumulative
    limit is used only by cumulative_limits.  Nothing invested (capacity 100 existing, no investment)."""
    capacity = {**_CAPACITY, "rows": [["p1", 100.0, None, None, 0.0, 100.0]]}
    got = _entity.hints({"invest_method": "cumulative_limits", "invest_max_total": 50.0,
                         "cumulative_max_capacity": 100.0}, {"capacity": capacity}, [])
    assert got == [
        "invest_max_total is set (50) but not used: invest_method is cumulative_limits, and only invest_total, "
        "invest_period_total, invest_retire_total, invest_retire_period_total use it.",
        "In p1 the total capacity is 100 MW, equal to cumulative_max_capacity (100): consistent with the "
        "cumulative capacity limit binding."]
    got = _entity.hints({"invest_method": "not_allowed", "cumulative_max_capacity": 80.0}, {"capacity": capacity},
                        [])
    assert got == ["cumulative_max_capacity is set (80) but not used: invest_method is not_allowed, and only "
                   "cumulative_limits use it.",
                   "Nothing was invested: invest_method is not_allowed, which allows no investment."]
    # An unknown method (an older database's spelling) says nothing about the limits.
    assert _entity.hints({"invest_method": "invest_sometimes", "invest_max_total": 50.0}, {"capacity": capacity},
                         []) == []
