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
    assert rec["provenance"]["rank_order"][:3] == ["init", "west", "coal"]
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
    assert any(n.startswith("'coal_chp' exists in the input data but is not active in scenario 'coal' (")
               for n in rec["notes"])
    listing = _call(ctx, "list_entities", entity_class="unit", scenario="coal")
    # Of the ten units, only coal_plant has an entity_alternative in an alternative of coal (coal).
    assert listing["entities"] == [["coal_plant"]]
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
    assert rec["shape"] == "map, 1 index level (time)"
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
    assert {p["parameter"]: p["summary"] for p in node["parameters"]} == {
        "inflow": -300.0, "node_type": "balance", "penalty_down": 50.0, "penalty_up": 100.0}
    assert {p["from_alternative"] for p in node["parameters"]} == {"heat"}
    assert [(r["entity_class"], r["entity"]) for r in node["relationships"]] == [
        ("group__node", ["heat", "heat"]), ("unit__outputNode", ["coal_chp", "heat"])]
    assert by_class["group"]["active"] is True
    assert [(r["entity_class"], r["entity"]) for r in by_class["group"]["relationships"]] == [
        ("group__node", ["heat", "heat"])]


def test_compare_inputs_of_two_scenarios(ctx) -> None:
    out = _call(ctx, "compare_inputs", scenario_a="coal", scenario_b="coal_chp", max_rows=500)
    # coal = init, west, coal; coal_chp = init, west, coal_chp, heat.
    assert out["alternatives_only_a"] == ["coal"]
    assert out["alternatives_only_b"] == ["coal_chp", "heat"]
    diffs = {(d["entity_class"], tuple(d["entity"]), d["parameter"]): d for d in out["differences"]}
    # commodity coal price 20 is set in alternative coal only.
    price = diffs[("commodity", ("coal",), "price")]
    assert (price["a"], price["b"], price["from_alternative_a"]) == (20.0, None, "coal")
    # unit coal_chp existing: 1200 from coal, 1000 from coal_chp (the unit is active in coal_chp).
    chp = diffs[("unit", ("coal_chp",), "existing")]
    assert (chp["a"], chp["b"]) == (1200.0, 1000.0)
    # The west node inflow comes from alternative west in both: no difference.
    assert ("node", ("west",), "inflow") not in diffs
    only_b = {(e["entity_class"], tuple(e["entity"])) for e in out["entities_active_only_in"]["b"]}
    assert {("unit", ("coal_chp",)), ("node", ("heat",))} <= only_b
    only_a = {(e["entity_class"], tuple(e["entity"])) for e in out["entities_active_only_in"]["a"]}
    assert ("unit", ("coal_plant",)) in only_a


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
    assert not out.ok and out.message.startswith("The input source 'tests.sqlite' has 69 scenarios")
