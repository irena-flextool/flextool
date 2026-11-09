"""Input values as the tools show them, against values read by hand from ``tests/fixtures/tests.json``.

Fixture facts used below (all from tests.json):

* scenario ``coal`` = alternatives init, west, coal (scenario_alternatives, in that rank order);
  ``coal_chp`` = init, west, coal_chp, heat; ``all`` = init, west, coal, wind, heat, coal_chp, network,
  water_pump, reserve_n_1, coal_min_load, capacity_margin, retire, co2_price, coal_invest, reserve, inertia,
  battery, battery_invest, 5weeks.
* unit coal_plant existing: coal 500, retire 2000, also 1x10y 100.0, 2x5y 100, coal_unit_size 2,
  dr_decrease_demand 100.
* the entity classes unit and node have active_by_default false; unit coal_chp has one entity_alternative, in
  alternative coal_chp; node heat one, in alternative heat; class group is active by default.
"""
from __future__ import annotations

import json

import pytest

from flextool.assistant.inputs._values import RUN_TIME_NOTE
from flextool.assistant.registry import call_tool

from tests.assistant.conftest import compare_rows


def _call(ctx, tool: str, **args) -> dict:
    out = call_tool(ctx, tool, args)
    assert out.ok, out.message
    return json.loads(out.text)


def test_winner_and_the_values_it_overrides(ctx) -> None:
    rec = _call(ctx, "get_parameter", entity_class="unit", parameter="existing", entity=["coal_plant"],
                scenario="all")
    # retire (rank 12 of all) is after coal (rank 3): its 2000 wins over coal's 500.
    assert rec["stored"] is True and rec["is_default"] is False
    assert rec["value"] == 2000.0
    assert rec["provenance"]["from_alternative"] == "retire"
    assert rec["provenance"]["overridden"] == [{"alternative": "coal", "value": 500.0}]
    # The other four alternatives that set it are not in scenario all.
    assert rec["provenance"]["also_set_outside_the_scenario"] == ["1x10y", "2x5y", "coal_unit_size",
                                                                   "dr_decrease_demand"]
    # Highest rank first (Cycle 1, C1-OD1): retire (12 of all) before coal (3); init (1) last.
    assert rec["provenance"]["alternatives_highest_first"][-3:] == ["coal", "west", "init"]
    assert "rank_order" not in rec["provenance"]
    # coal_plant has no virtual_unitsize in tests.json, so existing is a capacity in MW.
    assert rec["unit"] == "MW (default convention)"


def test_default_when_no_alternative_sets_it(ctx) -> None:
    # node coal_market is active in coal (entity_alternative in alternative coal) and only node_type is set for
    # it; penalty_up's default in tests.json's parameter_definitions is 10000.0.
    rec = _call(ctx, "get_parameter", entity_class="node", parameter="penalty_up", entity=["coal_market"],
                scenario="coal")
    assert rec["stored"] is False and rec["is_default"] is True
    assert rec["value"] == 10000.0
    assert rec["provenance"]["from_alternative"] is None


def test_inactive_entity_in_scenario(ctx) -> None:
    # coal_chp's only entity_alternative is in alternative coal_chp, which scenario coal does not have; units
    # are inactive by default.  Its existing comes from alternative coal (1200).
    rec = _call(ctx, "get_parameter", entity_class="unit", parameter="existing", entity=["coal_chp"],
                scenario="coal")
    assert rec["active"] is False
    assert rec["value"] == 1200.0 and rec["provenance"]["from_alternative"] == "coal"
    assert any(n.startswith('"coal_chp" exists in the input data but is not active in scenario "coal"; the reason: "')
               for n in rec["notes"])
    listing = _call(ctx, "list_entities", entity_class="unit", scenario="coal")
    # Of the ten units, only coal_plant has an entity_alternative in an alternative of coal (coal).
    assert listing["entities"] == [{"entity": ["coal_plant"], "description": None}]
    assert ["coal_chp"] in listing["inactive_in_scenario"]


def test_series_summary(ctx) -> None:
    # node west inflow (alternative west): 72 steps t0001..t0072, index name "time", min -872, max -423,
    # sum -45230 (mean -628.19…), first -589, last -635.
    rec = _call(ctx, "get_parameter", entity_class="node", parameter="inflow", entity=["west"], scenario="coal",
                max_rows=10)
    assert rec["kind"] == "map" and rec["index_names"] == ["time"] and rec["n"] == 72
    assert rec["min"] == -872.0 and rec["max"] == -423.0
    assert rec["mean"] == pytest.approx(-45230.0 / 72)
    assert rec["first"] == ["t0001", -589.0] and rec["last"] == ["t0072", -635.0]
    assert "table" not in rec
    # (node, inflow) is not in the shape registry; the shipped schema types it float or map rank 1 or 3.
    assert rec["shape"] == "a map, 1 index level (time)"
    assert rec["shape_note"].startswith("FlexTool reads this parameter as a constant, or a map with 1 or 3 index "
                                        "levels.")
    assert rec["unit"] == "MWh per time step (default convention)"
    part = _call(ctx, "get_parameter", entity_class="node", parameter="inflow", entity=["west"], scenario="coal",
                 time_range={"from": "t0001", "to": "t0003"})
    # The first three steps of the same series.
    assert part["table"] == [["t0001", -589.0], ["t0002", -537.0], ["t0003", -506.0]]


def test_array_value(ctx) -> None:
    # model flexTool solves in coal comes from alternative init: an array ["y2020_2day_dispatch"] with index name
    # sequence_index.
    rec = _call(ctx, "get_parameter", entity_class="model", parameter="solves", entity=["flexTool"],
                scenario="coal")
    assert rec["kind"] == "array" and rec["index_names"] == ["sequence_index"]
    assert rec["table"] == [[0, "y2020_2day_dispatch"]]
    assert rec["provenance"]["from_alternative"] == "init"


def test_unnamed_index_is_decided_at_run_time(ctx) -> None:
    # unit coal_plant availability (alternative availability, in scenario test_a_lot) is a map over t0001..t0072
    # stored without an index name; availability takes a constant or a map over periods or over time.
    rec = _call(ctx, "get_parameter", entity_class="unit", parameter="availability", entity=["coal_plant"],
                scenario="test_a_lot", max_rows=5)
    assert rec["index_names"] == ["x"]
    assert rec["shape_note"] == RUN_TIME_NOTE


def test_named_period_index_is_checked(ctx) -> None:
    # unit coal_plant invest_max_period (alternative coal_invest, in scenario all) is a map with the index name
    # period: p2020 250, p2025 350, p2030 400, p2035 0.  The shape registry allows a period map for it.
    rec = _call(ctx, "get_parameter", entity_class="unit", parameter="invest_max_period", entity=["coal_plant"],
                scenario="all")
    assert rec["table"] == [["p2020", 250.0], ["p2025", 350.0], ["p2030", 400.0], ["p2035", 0.0]]
    assert rec["shape"] == "a map over periods (1 level)" and rec["shape_note"] is None


def test_get_entity_across_classes(ctx) -> None:
    # "heat" is a node and a group.  In coal_chp the node is active (entity_alternative in alternative heat) and
    # the group too (class group active by default).  node heat values in coal_chp: inflow -300, node_type
    # "balance", penalty_down 50, penalty_up 100 (all alternative heat).  Entities naming it: group__node
    # (heat, heat) and unit__outputNode (coal_chp, heat).
    out = _call(ctx, "get_entity", name=["heat"], scenario="coal_chp")
    by_class = {m["entity_class"]: m for m in out["matches"]}
    assert set(by_class) == {"node", "group"}
    node = by_class["node"]
    assert node["active"] is True
    assert {p["parameter"]: p["value"] for p in node["parameters"]} == {
        "inflow": -300.0, "node_type": "balance", "penalty_down": 50.0, "penalty_up": 100.0}
    assert {p["from_alternative"] for p in node["parameters"]} == {"heat"}
    assert [(r["entity_class"], r["entity"]) for r in node["relationships"]] == [
        ("group__node", ["heat", "heat"]), ("unit__outputNode", ["coal_chp", "heat"])]
    assert by_class["group"]["active"] is True
    assert [(r["entity_class"], r["entity"]) for r in by_class["group"]["relationships"]] == [
        ("group__node", ["heat", "heat"])]


def _only_in(out: dict, side: str) -> set[tuple]:
    return {(cls, tuple(e)) for cls, (_, names) in out["entities_active_only_in"][side].items() for e in names}


def test_compare_inputs_of_two_scenarios(ctx) -> None:
    out = _call(ctx, "compare_inputs", scenario_a="coal", scenario_b="coal_chp", max_rows=500)
    # coal = init, west, coal; coal_chp = init, west, coal_chp, heat (highest rank first: heat, coal_chp).
    assert out["alternatives_only_a"] == ["coal"]
    assert out["alternatives_only_b"] == ["heat", "coal_chp"]
    diffs = compare_rows(out)
    # commodity coal price 20 is set in alternative coal only: coal makes the scenarios differ.
    price = diffs[("commodity", ("coal",), "price")]
    assert (price["a"], price["b"], price["alternative_a"], price["alternative"]) == (20.0, None, "coal", "coal")
    # unit coal_chp existing: 1200 from coal, 1000 from coal_chp (the unit is active in coal_chp).
    chp = diffs[("unit", ("coal_chp",), "existing")]
    assert (chp["a"], chp["b"]) == (1200.0, 1000.0)
    # The west node inflow comes from alternative west in both: no difference.
    assert ("node", ("west",), "inflow") not in diffs
    assert {("unit", ("coal_chp",)), ("node", ("heat",))} <= _only_in(out, "b")
    assert ("unit", ("coal_plant",)) in _only_in(out, "a")


def test_describe_parameter_uses_the_shipped_schema(ctx) -> None:
    out = _call(ctx, "describe_parameter", entity_class="unit", parameter="invest_cost")
    assert out["unit"] == "CUR/kW (default convention)"
    assert any("multiplies the value by 1000" in n for n in out["notes"])
    assert out["schema_note"] == "This is the parameter as this FlexTool defines it."
    listing = _call(ctx, "describe_parameter", entity_class="node")
    assert "penalty_up" in {p["parameter"] for p in listing["parameters"]}


def test_unknown_names_answer_with_similar_ones(ctx) -> None:
    out = call_tool(ctx, "get_parameter", {"entity_class": "unit", "parameter": "existng",
                                           "entity": ["coal_plant"], "scenario": "coal"})
    assert not out.ok and "existing" in out.message
    out = call_tool(ctx, "get_parameter", {"entity_class": "unit", "parameter": "existing",
                                           "entity": ["coal_plnt"], "scenario": "coal"})
    assert not out.ok and "coal_plant" in out.message
    out = call_tool(ctx, "get_parameter", {"entity_class": "unit", "parameter": "existing",
                                           "entity": ["coal_plant"]})
    # tests.json has 69 scenarios.
    assert not out.ok and out.message.startswith('The input source "tests.sqlite" has 69 scenarios')


def test_get_entity_gives_each_value_the_scenario_sets(ctx) -> None:
    """P3 §4.7: one call answers a lookup.  Fixture facts (tests.json): in scenario coal (init, west, coal) only
    alternative coal sets values of unit coal_plant: efficiency 0.4 and existing 500.  In scenario all,
    invest_max_period comes from coal_invest, a period map p2020 250, p2025 350, p2030 400, p2035 0, and
    existing from retire, 2000.  availability (alternative availability, in scenario test_a_lot) is a map over
    t0001..t0072."""
    def unit(scenario: str) -> dict:
        out = _call(ctx, "get_entity", name=["coal_plant"], scenario=scenario)
        return {m["entity_class"]: m for m in out["matches"]}["unit"]

    coal = unit("coal")
    assert {p["parameter"]: (p["value"], p["from_alternative"]) for p in coal["parameters"]} == {
        "efficiency": (0.4, "coal"), "existing": (500.0, "coal")}
    assert coal["parameters_total"] == 2
    every = {p["parameter"]: p for p in unit("all")["parameters"]}
    assert every["existing"]["value"] == 2000.0 and every["existing"]["from_alternative"] == "retire"
    period = every["invest_max_period"]["value"]
    assert period["table"] == [["p2020", 250.0], ["p2025", 350.0], ["p2030", 400.0], ["p2035", 0.0]]
    assert period["n"] == 4 and every["invest_max_period"]["from_alternative"] == "coal_invest"
    availability = {p["parameter"]: p for p in unit("test_a_lot")["parameters"]}["availability"]["value"]
    assert availability["n"] == 72 and "table" not in availability
    assert {"min", "mean", "max"} <= set(availability)


def test_an_unknown_parameter_name_gets_the_closest_valid_names(ctx) -> None:
    start = 'The entity class "unit" has no parameter "existing_capacity". Did you mean: "existing"?'
    out = call_tool(ctx, "get_parameter", {"entity_class": "unit", "parameter": "existing_capacity",
                                           "entity": ["coal_plant"], "scenario": "coal"})
    assert not out.ok and out.message.startswith(start)
    out = call_tool(ctx, "describe_parameter", {"entity_class": "unit", "parameter": "existing_capacity"})
    assert not out.ok and out.message.startswith(start)


def test_closest_names() -> None:
    from flextool.assistant.errors import closest_names, did_you_mean

    names = ["constraint_capacity_coeff", "cumulative_min_capacity", "efficiency", "existing", "virtual_unitsize"]
    assert closest_names("existing_capacity", names)[0] == "existing"
    assert closest_names("Existing", names) == ["existing"]
    assert closest_names("zzz", names) == []
    assert did_you_mean("existng", ["existing", "efficiency"]).render() == ' Did you mean: "existing"?'
    assert did_you_mean("zzz", names).render() == ""


# ---------------------------------------------------------------------------------------------------------
# time_range selects by position in the series (C3-M3), on synthetic spinedb_api values


def _steps(n: int, name: str = "time"):
    """A map t1…t<n> (unpadded names) whose value at t<k> is k."""
    from spinedb_api import Map

    return Map([f"t{k}" for k in range(1, n + 1)], [float(k) for k in range(1, n + 1)], index_name=name)


def test_time_range_keeps_the_steps_between_its_bounds_in_series_order() -> None:
    from flextool.assistant.inputs._values import summarize

    out = summarize(_steps(200), max_rows=100, time_range=("t1", "t24"))
    # t1…t24 in the series' order (string order would add t100…t199 and t3…t9 would be missing).
    assert out["n"] == 24
    assert out["table"] == [[f"t{k}", float(k)] for k in range(1, 25)]
    summary = summarize(_steps(200), max_rows=10, time_range=("t1", "t24"))
    assert summary["n"] == 24 and summary["mean"] == 12.5          # (1 + 24) / 2
    assert summary["first"] == ["t1", 1.0] and summary["last"] == ["t24", 24.0]


def test_time_range_on_a_time_series_takes_iso_bounds() -> None:
    from spinedb_api import TimeSeriesFixedResolution

    from flextool.assistant.inputs._values import summarize

    series = TimeSeriesFixedResolution("2020-01-01T00:00", "1h", [float(h) for h in range(48)], False, False)
    out = summarize(series, max_rows=100, time_range=("2020-01-01T00:00", "2020-01-01T23:00"))
    # Hours 0…23 of 1 January (value = hour), the index in ISO form.
    assert out["n"] == 24
    assert out["table"][0] == ["2020-01-01T00:00:00", 0.0]
    assert out["table"][-1] == ["2020-01-01T23:00:00", 23.0]
    summary = summarize(series, max_rows=10, time_range=("2020-01-01T00:00", "2020-01-01T23:00"))
    assert summary["n"] == 24 and summary["max"] == 23.0
    assert summary["first"] == ["2020-01-01T00:00:00", 0.0] and summary["last"] == ["2020-01-01T23:00:00", 23.0]
    # The same instants named another way: a date alone is midnight, a space for the T.
    same = summarize(series, max_rows=100, time_range=("2020-01-01", "2020-01-01 23:00"))
    assert same["table"] == out["table"]


def test_time_range_bounds_reversed_or_missing() -> None:
    from flextool.assistant.errors import ToolError
    from flextool.assistant.inputs._values import summarize

    forward = summarize(_steps(200), max_rows=100, time_range=("t1", "t24"))
    backward = summarize(_steps(200), max_rows=100, time_range=("t24", "t1"))
    assert backward["table"] == forward["table"]
    with pytest.raises(ToolError) as caught:
        summarize(_steps(200), max_rows=100, time_range=("t1", "t999"))
    assert caught.value.key == "bad_input_time_step"
    assert str(caught.value) == ('time_range: the value has no index "t999" at that level. Its index runs from "t1" '
                                 'to "t200".')


def test_time_range_on_a_period_time_map_keeps_every_period() -> None:
    from spinedb_api import Map

    from flextool.assistant.inputs._values import summarize

    value = Map(["p1", "p2"], [_steps(30), _steps(30)], index_name="period")
    out = summarize(value, max_rows=100, time_range=("t5", "t7"))
    # Steps t5, t6, t7 of each of the two periods.
    assert out["table"] == [["p1", "t5", 5.0], ["p1", "t6", 6.0], ["p1", "t7", 7.0],
                            ["p2", "t5", 5.0], ["p2", "t6", 6.0], ["p2", "t7", 7.0]]


#: unit coal_plant's parameters with a value in some alternative of tests.json (its parameter_values rows).
COAL_PLANT_STORED = [
    "availability", "conversion_method", "cumulative_max_capacity", "discount_rate", "efficiency",
    "efficiency_at_min_load", "existing", "fixed_cost", "invest_cost", "invest_max_period", "invest_max_total",
    "invest_method", "lifetime", "min_downtime", "min_load", "min_uptime", "minimum_time_method",
    "retire_max_total", "salvage_value", "startup_cost", "startup_method", "virtual_unitsize"]


class _NoScan(dict):
    """The values of a database that may be looked up but never scanned."""

    def _scan(self, *args, **kwargs):
        raise AssertionError("every stored value was scanned")

    __iter__ = keys = items = values = _scan


def test_entity_lookups_use_the_index_of_stored_parameters(ctx, monkeypatch) -> None:
    """C3-n3.  get_entity and stored_parameters read an index built at load, never every stored value.  In
    scenario coal (init, west, coal) only alternative coal sets values of unit coal_plant: efficiency 0.4 and
    existing 500."""
    from flextool.assistant.inputs import _db

    db = _db.load(ctx.project_path / "input_sources" / "tests.sqlite")
    db.view("coal")
    db.values = _NoScan(db.values)
    monkeypatch.setattr(_db.ScenarioView, "winners", _NoScan._scan)
    out = _call(ctx, "get_entity", name=["coal_plant"], scenario="coal")
    unit = {m["entity_class"]: m for m in out["matches"]}["unit"]
    assert [(p["parameter"], p["value"], p["from_alternative"]) for p in unit["parameters"]] == [
        ("efficiency", 0.4, "coal"), ("existing", 500.0, "coal")]
    assert db.stored_parameters("unit", ("coal_plant",)) == COAL_PLANT_STORED


def _scenarios_sharing_values(path):
    """Two scenarios A = [base, x] and B = [base, y] on a copy of tests.json, over new commodities c1…c6 (class
    commodity is active by default): c1–c3 price only in base; c4 price 7.0 in x and in y (same bytes); c5 1 in x
    and 1.0 in y; c6 5.0 in x and 6.0 in y."""
    from spinedb_api import to_database

    from flextool.update_flextool.db_handle import open_database_mapping

    def put(db, entity: str, alternative: str, value) -> None:
        raw, value_type = to_database(value)
        db.add_parameter_value_item(entity_class_name="commodity", entity_byname=(entity,),
                                    parameter_definition_name="price", alternative_name=alternative, value=raw,
                                    type=value_type)

    with open_database_mapping(f"sqlite:///{path.as_posix()}") as db:
        for alt in ("base", "x", "y"):
            db.add_alternative_item(name=alt)
        for scenario, alts in (("A", ("base", "x")), ("B", ("base", "y"))):
            db.add_scenario_item(name=scenario)
            for rank, alt in enumerate(alts, start=1):
                db.add_scenario_alternative_item(scenario_name=scenario, alternative_name=alt, rank=rank)
        for k in range(1, 7):
            db.add_entity_item(entity_class_name="commodity", name=f"c{k}")
        for k in (1, 2, 3):
            put(db, f"c{k}", "base", float(k))
        put(db, "c4", "x", 7.0)
        put(db, "c4", "y", 7.0)
        put(db, "c5", "x", 1)
        put(db, "c5", "y", 1.0)
        put(db, "c6", "x", 5.0)
        put(db, "c6", "y", 6.0)
        db.commit_session("test: two scenarios sharing values")


def test_compare_inputs_parses_only_values_that_differ(tmp_path, tests_db, monkeypatch) -> None:
    """C3-m7.  Shared values (the same item, or the same stored bytes) are never parsed: only c5 (1 vs 1.0,
    equal once parsed) and c6 (5 vs 6, the one difference) are, two parses each."""
    from flextool.assistant.context import make_context
    from flextool.assistant.inputs import _db
    from tests.assistant.conftest import make_project

    project = make_project(tmp_path, tests_db)
    _scenarios_sharing_values(project / "input_sources" / "tests.sqlite")
    ctx = make_context(project)
    _db.load(project / "input_sources" / "tests.sqlite")
    parses = []
    real = _db.ValueItem.parsed

    def counted(self):
        parses.append(self)
        return real(self)

    monkeypatch.setattr(_db.ValueItem, "parsed", counted)
    out = _call(ctx, "compare_inputs", scenario_a="A", scenario_b="B", entity_class="commodity")
    assert [(d["entities"], d["a"], d["b"]) for d in (dict(zip(out["columns"], r)) for r in out["differences"])] == [
        ([["c6"]], 5.0, 6.0)]
    assert len(parses) == 4


def _identical_changes(path):
    """On a copy of tests.json: connections c01–c12 with efficiency 0.98 in alternative base_net and 0.97 in
    loss_up; connections d01, d02 with the same map {p2020: 0.9, p2025: 0.95} in loss_up only; all active in
    base_net.  Scenarios A = [base_net] and B = [base_net, loss_up]."""
    from spinedb_api import Map, to_database

    from flextool.update_flextool.db_handle import open_database_mapping

    def put(db, entity: str, alternative: str, value) -> None:
        raw, value_type = to_database(value)
        db.add_parameter_value_item(entity_class_name="connection", entity_byname=(entity,),
                                    parameter_definition_name="efficiency", alternative_name=alternative,
                                    value=raw, type=value_type)

    with open_database_mapping(f"sqlite:///{path.as_posix()}") as db:
        for alt in ("base_net", "loss_up"):
            db.add_alternative_item(name=alt)
        for scenario, alts in (("A", ("base_net",)), ("B", ("base_net", "loss_up"))):
            db.add_scenario_item(name=scenario)
            for rank, alt in enumerate(alts, start=1):
                db.add_scenario_alternative_item(scenario_name=scenario, alternative_name=alt, rank=rank)
        names = [f"c{k:02d}" for k in range(1, 13)] + ["d01", "d02"]
        for name in names:
            db.add_entity_item(entity_class_name="connection", name=name)
            db.add_entity_alternative_item(entity_class_name="connection", entity_byname=(name,),
                                           alternative_name="base_net", active=True)
        for name in names[:12]:
            put(db, name, "base_net", 0.98)
            put(db, name, "loss_up", 0.97)
        for name in ("d01", "d02"):
            put(db, name, "loss_up", Map(["p2020", "p2025"], [0.9, 0.95], index_name="period"))
        db.commit_session("test: identical changes")


def test_compare_inputs_groups_identical_changes(tmp_path, tests_db) -> None:
    """Agent feedback waste 3: one row per parameter and change, with its entities; counts by alternative and
    class first (critique M1: the alternative that makes the scenarios differ, in both directions; m2: a map
    value groups too).  Before: one row per entity (14 rows), no counts, no alternative argument."""
    from flextool.assistant.context import make_context
    from tests.assistant.conftest import make_project

    project = make_project(tmp_path, tests_db)
    _identical_changes(project / "input_sources" / "tests.sqlite")
    ctx = make_context(project)
    out = _call(ctx, "compare_inputs", scenario_a="A", scenario_b="B", entity_class="connection")
    assert out["alternatives_only_a"] == [] and out["alternatives_only_b"] == ["loss_up"]
    assert out["by_alternative"] == {"loss_up": {"connection": 14}}
    assert (out["total_values"], out["total_rows"]) == (14, 2)
    rows = [dict(zip(out["columns"], r)) for r in out["differences"]]
    first, second = rows
    # c01–c12: 0.98 (base_net) in A, 0.97 (loss_up) in B, all twelve names listed.
    assert (first["alternative"], first["alternative_a"], first["alternative_b"], first["parameter"]) == (
        "loss_up", "base_net", "loss_up", "efficiency")
    assert (first["a"], first["b"], first["n"]) == (0.98, 0.97, 12)
    assert first["entities"] == [[f"c{k:02d}"] for k in range(1, 13)]
    # d01, d02: no value in A, the same map in B.
    assert (second["alternative"], second["alternative_a"], second["a"], second["n"], second["entities"]) == (
        "loss_up", None, None, 2, [["d01"], ["d02"]])
    assert second["b"]["n"] == 2 and (second["b"]["min"], second["b"]["max"]) == (0.9, 0.95)
    assert "shown" not in out
    # The reverse pair: loss_up is a's, and still the alternative that makes them differ.
    reverse = _call(ctx, "compare_inputs", scenario_a="B", scenario_b="A", entity_class="connection")
    assert reverse["by_alternative"] == {"loss_up": {"connection": 14}}
    assert reverse["alternatives_only_a"] == ["loss_up"]
    # alternative narrows: base_net makes none of them differ.
    assert _call(ctx, "compare_inputs", scenario_a="A", scenario_b="B", alternative="base_net")["total_values"] == 0
    assert _call(ctx, "compare_inputs", scenario_a="B", scenario_b="A", alternative="loss_up")["total_values"] == 14
    refused = call_tool(ctx, "compare_inputs", {"scenario_a": "A", "scenario_b": "B", "alternative": "loss_upp"})
    assert not refused.ok and refused.message == ('Neither scenario has the alternative "loss_upp". Their '
                                                  'alternatives include: "loss_up".')
    # max_rows caps the rows, and says so.
    capped = _call(ctx, "compare_inputs", scenario_a="A", scenario_b="B", max_rows=1)
    assert len(capped["differences"]) == 1
    assert capped["shown"] == "1 of 2 rows (14 values); pass alternative or entity_class"
