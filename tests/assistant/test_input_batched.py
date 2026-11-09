"""get_parameter with lists (P0 §4.3, batching 2026-11-08): several parameters of several entities in one call, as
one table, against values read by hand from ``tests/fixtures/tests.json``.

Fixture facts used below (tests.json):

* scenario ``all`` ranks init, west, coal, wind, heat, coal_chp (6th), network, water_pump, reserve_n_1,
  coal_min_load, capacity_margin, retire (12th), co2_price, coal_invest (14th), reserve, inertia, battery,
  battery_invest, 5weeks; a later alternative wins.
* unit coal_plant: existing coal 500, retire 2000 (also in 1x10y, 2x5y, coal_unit_size, dr_decrease_demand, which
  scenario all lacks); invest_cost coal_invest 400, retire 1200.0; availability only in alternative availability
  (not in all); invest_method coal_invest "invest_total", retire "retire_total"; virtual_unitsize only in
  coal_unit_size (not in all).
* unit coal_chp: existing coal 1200, coal_chp 1000; no invest_cost, availability, invest_method or
  virtual_unitsize.  Its one entity_alternative is in coal_chp (in all), coal_plant's in coal (in all).
* parameter_definitions: unit availability default 1.0; unit existing and invest_cost no default.
* the shipped schema brackets: existing [MW or count], invest_cost [CUR/kW], availability [-].
"""
from __future__ import annotations

import json

from flextool.assistant.registry import call_tool
from flextool.assistant.tools.inputs import MAX_ENTITIES, MAX_PARAMETERS

KW_NOTE = ("Entered per kW (or kWh): FlexTool multiplies the value by 1000 inside the model, to a value per MW "
           "(or MWh). With other units it is per thousandth of the flow unit.")


def _call(ctx, **args) -> dict:
    out = call_tool(ctx, "get_parameter", args)
    assert out.ok, out.message
    return json.loads(out.text)


def _cells(out: dict) -> dict[tuple[str, str], dict]:
    cols = out["columns"]
    return {(row[0][0], row[1]): dict(zip(cols, row)) for row in out["rows"]}


def test_three_parameters_of_two_entities_in_one_table(ctx) -> None:
    out = _call(ctx, entity_class="unit", parameters=["existing", "invest_cost", "availability"],
                entities=[["coal_plant"], ["coal_chp"]], scenario="all")
    assert out["scenario"] == "all" and out["entity_class"] == "unit" and out["total"] == 6
    # Rows in the order asked: entity by entity, the parameters in the order named.
    assert [(r[0], r[1]) for r in out["rows"]] == [
        (["coal_plant"], "existing"), (["coal_plant"], "invest_cost"), (["coal_plant"], "availability"),
        (["coal_chp"], "existing"), (["coal_chp"], "invest_cost"), (["coal_chp"], "availability")]
    cells = _cells(out)
    plant, chp = "coal_plant", "coal_chp"
    # retire (12th) wins over coal (3rd) for coal_plant existing.
    assert (cells[plant, "existing"]["value"], cells[plant, "existing"]["from_alternative"],
            cells[plant, "existing"]["is_default"]) == (2000.0, "retire", False)
    # coal_invest (14th) wins over retire (12th) for invest_cost.
    assert (cells[plant, "invest_cost"]["value"], cells[plant, "invest_cost"]["from_alternative"]) == (
        400.0, "coal_invest")
    # availability: no alternative of all sets it, the default 1.0 is shown.
    assert (cells[plant, "availability"]["value"], cells[plant, "availability"]["from_alternative"],
            cells[plant, "availability"]["is_default"]) == (1.0, None, True)
    # coal_chp (6th) wins over coal (3rd) for coal_chp existing.
    assert (cells[chp, "existing"]["value"], cells[chp, "existing"]["from_alternative"]) == (1000.0, "coal_chp")
    # coal_chp has no invest_cost and invest_cost no default: no value, and the cell says so.
    assert (cells[chp, "invest_cost"]["value"], cells[chp, "invest_cost"]["is_default"],
            cells[chp, "invest_cost"]["shape"]) == (None, False, None)
    assert "No alternative of the scenario sets this value and the parameter has no default." in \
        cells[chp, "invest_cost"]["notes"]
    assert (cells[chp, "availability"]["value"], cells[chp, "availability"]["is_default"]) == (1.0, True)
    # One unit per parameter in the header (neither unit has virtual_unitsize in all, so existing is MW); no unit
    # column, since no cell differs.
    assert out["units"] == {"existing": "MW (default convention)", "invest_cost": "CUR/kW (default convention)",
                            "availability": "dimensionless"}
    assert "unit" not in out["columns"]
    # A note every cell of a parameter shares is in the header once, not in the rows.
    assert KW_NOTE in out["notes_by_parameter"]["invest_cost"]
    assert all(KW_NOTE not in c["notes"] for c in cells.values())
    assert "a ratio or a share" in out["notes_by_parameter"]["availability"]
    # Both units are active in all; nothing unknown.
    assert "inactive_in_scenario" not in out and "unknown_parameters" not in out and "unknown_entities" not in out


def test_the_single_form_is_unchanged(ctx) -> None:
    """parameter and entity, one each: the full value record (with the alternatives it overrides), as before."""
    out = _call(ctx, entity_class="unit", parameter="existing", entity=["coal_plant"], scenario="all")
    assert out["value"] == 2000.0 and out["provenance"]["from_alternative"] == "retire"
    assert out["provenance"]["overridden"] == [{"alternative": "coal", "value": 500.0}]
    assert "rows" not in out
    # Without entity: one line per active entity, as before.
    listing = _call(ctx, entity_class="unit", parameter="existing", scenario="coal")
    assert listing["parameter"] == "existing" and listing["entities"][0]["entity"] == ["coal_plant"]


def test_one_name_where_the_schema_has_a_list(ctx) -> None:
    """parameters and entities also take one name; entities a flat list of names."""
    out = _call(ctx, entity_class="unit", parameters="existing", entities=["coal_plant", "coal_chp"],
                scenario="all")
    assert [(r[0], r[1], r[2]) for r in out["rows"]] == [(["coal_plant"], "existing", 2000.0),
                                                         (["coal_chp"], "existing", 1000.0)]
    # The two names of a unit__outputNode given flat are that one entity (neither name is one by itself); its
    # constraint_flow_coeff comes from alternative coal_chp (tests.json).
    out = _call(ctx, entity_class="unit__outputNode", parameters=["constraint_flow_coeff"],
                entities=["coal_chp", "heat"], scenario="coal_chp")
    assert [(r[0], r[4]) for r in out["rows"]] == [(["coal_chp", "heat"], "coal_chp")]


def test_unknown_names_do_not_fail_the_call(ctx) -> None:
    out = _call(ctx, entity_class="unit", parameters=["existing", "existng"],
                entities=[["coal_plant"], ["coal_plnt"]], scenario="all")
    # The known cell is answered.
    assert [(r[0], r[1], r[2]) for r in out["rows"]] == [(["coal_plant"], "existing", 2000.0)]
    assert out["unknown_parameters"] == [{"parameter": "existng", "did_you_mean": ["existing"]}]
    assert out["unknown_entities"][0]["entity"] == ["coal_plnt"]
    assert out["unknown_entities"][0]["similar"][0] == "coal_plant"


def test_an_inactive_entity_is_named_in_the_header(ctx) -> None:
    # In scenario coal, coal_chp is inactive (its entity_alternative is in coal_chp); its existing is coal's 1200.
    out = _call(ctx, entity_class="unit", parameters=["existing"], entities=[["coal_plant"], ["coal_chp"]],
                scenario="coal")
    assert out["inactive_in_scenario"] == [["coal_chp"]]
    assert _cells(out)["coal_chp", "existing"]["value"] == 1200.0


def test_the_size_cap(ctx) -> None:
    parameters = [f"p{i}" for i in range(MAX_PARAMETERS + 1)]
    out = call_tool(ctx, "get_parameter", {"entity_class": "unit", "parameters": parameters,
                                           "entities": [["coal_plant"]], "scenario": "all"})
    assert not out.ok and out.message == (
        "One get_parameter call reads at most 10 parameters and 20 entities; this one names 11 parameters and 1 "
        "entities. Split it into smaller calls, which can go in the same step.")
    entities = [[f"u{i}"] for i in range(MAX_ENTITIES + 1)]
    out = call_tool(ctx, "get_parameter", {"entity_class": "unit", "parameters": ["existing"],
                                           "entities": entities, "scenario": "all"})
    assert not out.ok and "names 1 parameters and 21 entities" in out.message
    # At the cap it is answered.
    out = call_tool(ctx, "get_parameter", {"entity_class": "unit", "parameters": ["existing"] * 3,
                                           "entities": entities[:MAX_ENTITIES], "scenario": "all"})
    assert out.ok


def test_at_a_small_result_cap_the_notes_give_way_to_the_rows(ctx) -> None:
    """At the result cap of a 16 384-token window (2 000 characters) the six cells all fit: the notes are cut to 160
    or 80 characters first.  At 1 200 characters the notes are left out (and say so); the rows stay whole."""
    from flextool.assistant.caps import Caps
    from flextool.assistant.tools.inputs import NOTES_LEFT_OUT

    ctx.caps = Caps(17, 2_000, 3, 33)
    out = call_tool(ctx, "get_parameter", {"entity_class": "unit", "parameters": ["existing", "invest_cost",
                                                                                   "availability"],
                                           "entities": [["coal_plant"], ["coal_chp"]], "scenario": "all"})
    assert out.ok and len(out.text) <= 2_000
    table = json.loads(out.text)
    assert len(table["rows"]) == 6 and "truncated" not in table
    assert all(len(n) <= 160 for ns in table["notes_by_parameter"].values() for n in ns)
    assert table["notes_by_parameter"]["invest_cost"][0].startswith("Entered per kW (or kWh): FlexTool multiplies")
    ctx.caps = Caps(17, 1_200, 3, 33)
    out = call_tool(ctx, "get_parameter", {"entity_class": "unit", "parameters": ["existing", "invest_cost",
                                                                                   "availability"],
                                           "entities": [["coal_plant"], ["coal_chp"]], "scenario": "coal"})
    table = json.loads(out.text)
    assert len(table["rows"]) == 6 and "notes_by_parameter" not in table
    assert NOTES_LEFT_OUT in table["notes"] and all(row[-1] == [] for row in table["rows"])


def test_a_time_range_bound_missing_in_one_cell(ctx) -> None:
    """C3-M3.  tests.json, alternative init (in scenario coal; class timeset is active by default): timeset 5weeks
    timeset_duration is a map t0001 4, t0009 4, t0013 4, t0025 4, t0037 4; timeset 2day's is t0001 48."""
    out = _call(ctx, entity_class="timeset", parameters=["timeset_duration"], entities=[["5weeks"], ["2day"]],
                scenario="coal", time_range={"from": "t0009", "to": "t0025"})
    cells = _cells(out)
    assert cells["5weeks", "timeset_duration"]["value"]["table"] == [["t0009", 4.0], ["t0013", 4.0],
                                                                      ["t0025", 4.0]]
    two_day = cells["2day", "timeset_duration"]
    assert two_day["value"] is None
    assert ("time_range: the value has no index 't0009' at that level. Its index runs from t0001 to t0001."
            in two_day["notes"])


def test_shapes_are_worded_alike_inside_and_outside_the_registry(ctx) -> None:
    """C3-n2.  tests.json, scenario coal: unit coal_plant existing 500 and efficiency 0.4 (alternative coal), both
    constants; existing is outside the shape registry, efficiency inside."""
    from flextool.engine.input._param_shapes import PARAM_ALLOWED_SHAPES

    assert ("unit", "existing") not in PARAM_ALLOWED_SHAPES and ("unit", "efficiency") in PARAM_ALLOWED_SHAPES
    out = _call(ctx, entity_class="unit", parameters=["existing", "efficiency"], entities=[["coal_plant"]],
                scenario="coal")
    cells = _cells(out)
    assert cells["coal_plant", "existing"]["shape"] == "a constant"
    assert cells["coal_plant", "efficiency"]["shape"] == "a constant"
