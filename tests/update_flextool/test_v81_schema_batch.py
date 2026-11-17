"""Database migration to version 81 (all schema changes of one release in one step).

A v80 database (built from the v75 seed JSON and migrated up to 80, never a checked-in ``.sqlite``) carries
``lifetime`` (a constant, a ``period`` map, an unnamed ``x`` map and a ``time``-labelled map on a unit, constants on a
connection and a node), ``salvage_value`` (a constant and a ``period`` map) and ``lifetime_method`` values
``no_investment`` (two alternatives) and ``reinvest_choice``; the ``node.lifetime_method`` default is
``no_investment``.  The step to 81

* adds ``lifetime_technical`` and ``payback_time`` (group ``investment``) and ``net_salvage_value`` (group
  ``retirement``) on unit, connection and node: a constant or a period map, no default, the shipped description;
* adds the value list ``end_of_life_methods`` (closest_seam, previous_seam, next_seam), and on unit, connection and
  node (group ``retirement``) ``age_at_model_start`` and ``build_year`` (a constant, no default) and
  ``end_of_life_method`` (a choice of that list, default ``closest_seam``); ``model.start_year`` (group ``model``, a
  constant, no default);
* copies every ``lifetime`` value, as stored, to ``lifetime_technical`` only (``payback_time`` gets no value: it
  falls back to ``lifetime_technical``), and every ``salvage_value`` value to ``net_salvage_value``, then removes
  the old definitions;
* renames the ``lifetime_methods`` member ``no_investment`` to ``no_reinvestment`` (list order kept) and rewrites
  every stored value (each one printed) and the node default; the default ``reinvest_automatic`` of unit and
  connection is unchanged;
* rewrites the descriptions of ``lifetime_method``, ``discount_rate``, ``invest_cost``, ``node.unit_of_measure``
  and ``group.penalty_capacity_margin``;
* writes the behaviour-change marker ``annuity_over_payback_time`` naming the entities with ``reinvest_choice`` /
  ``no_reinvestment`` and a non-zero ``invest_cost`` (thermal: reinvest_choice, 400; not wind: no_investment with
  invest_cost 0; not c1: a non-zero invest_cost without a lifetime_method);
* writes the behaviour-change marker ``end_of_life_closest_seam`` naming the entities with ``reinvest_choice`` /
  ``no_reinvestment`` (thermal and wind; not c1 or city, which have no lifetime_method);
* writes the behaviour-change marker ``fixed_cost_per_operating_period`` naming the entities that may invest and
  have a fixed_cost that differs between periods (thermal: invest_no_limit and a map {10, 20} in another
  alternative; not wind: a map {5, 5}; not c1: no invest_method; not city: retire_no_limit).

Nothing else changes, and a rerun commits nothing.  Where a database already has a value under a new name, that
value is kept (and printed).  A database already at 81 that lacks a definition of the step gets it without a migration of its
values, and its own descriptions are kept; one that still has the shipped ``end_of_life_method`` default
``next_seam`` with its shipped description gets the current default and description, both together.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, Map, from_database, to_database
from spinedb_api.exception import NothingToCommit

from flextool._resources import package_data_path
from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool import db_migration as m
from flextool.update_flextool.db_migration import migrate_database
from tests.db_utils import json_to_db

SEED = Path(__file__).resolve().parent / "data" / "v75_seed_stoch_two_period.json"
TESTS_JSON = Path(__file__).resolve().parents[1] / "fixtures" / "tests.json"
CLASSES = ("unit", "connection", "node")
AGE_DEFINITIONS = {(cls, name) for cls in CLASSES for name in ("age_at_model_start", "build_year",
                                                                "end_of_life_method")} | {("model", "start_year")}
NEW_DEFINITIONS = {(cls, name) for cls in CLASSES for name in ("lifetime_technical", "payback_time",
                                                                "net_salvage_value")} | AGE_DEFINITIONS
OLD_DEFINITIONS = {(cls, name) for cls in CLASSES for name in ("lifetime", "salvage_value")}
REWRITTEN = {(cls, name) for cls in CLASSES for name in ("lifetime_method", "discount_rate", "invest_cost")} | {
    ("node", "unit_of_measure"), ("group", "penalty_capacity_margin")}

#: Stored before the step: (class, entity, alternative index, parameter, value).
STORED = [
    ("unit", "wind", 0, "lifetime", 25.0),
    ("unit", "wind", 1, "lifetime", Map(["p2020", "p2025"], [20.0, 30.0], index_name="period")),
    ("unit", "wind", 2, "lifetime", Map(["p2020", "p2025"], [21.0, 31.0])),
    ("unit", "wind", 3, "lifetime", Map(["t0001", "t0002"], [22.0, 32.0], index_name="time")),
    ("connection", "c1", 0, "lifetime", 40.0),
    ("node", "city", 0, "lifetime", 15.0),
    ("unit", "thermal", 0, "salvage_value", 100.0),
    ("unit", "thermal", 1, "salvage_value", Map(["p2020", "p2025"], [50.0, -20.0], index_name="period")),
    ("unit", "wind", 0, "lifetime_method", "no_investment"),
    ("unit", "wind", 1, "lifetime_method", "no_investment"),
    ("unit", "thermal", 0, "lifetime_method", "reinvest_choice"),
    ("unit", "thermal", 0, "invest_cost", 400.0),
    ("unit", "wind", 0, "invest_cost", 0.0),
    ("connection", "c1", 0, "invest_cost", Map(["p2020", "p2025"], [0.0, 50.0], index_name="period")),
    ("unit", "thermal", 0, "invest_method", "invest_no_limit"),
    ("unit", "thermal", 1, "fixed_cost", Map(["p2020", "p2025"], [10.0, 20.0], index_name="period")),
    ("unit", "wind", 0, "invest_method", "invest_no_limit"),
    ("unit", "wind", 0, "fixed_cost", Map(["p2020", "p2025"], [5.0, 5.0], index_name="period")),
    ("connection", "c1", 0, "fixed_cost", Map(["p2020", "p2025"], [1.0, 2.0], index_name="period")),
    ("node", "city", 0, "invest_method", "retire_no_limit"),
    ("node", "city", 0, "fixed_cost", Map(["p2020", "p2025"], [3.0, 4.0], index_name="period")),
]


def _defs(db) -> dict:
    return {(d["entity_class_name"], d["name"]): (d["description"] or "", tuple(d["parameter_type_list"] or ()),
                                                  d["parameter_group_name"], d["parameter_value_list_name"],
                                                  d["default_value"], d["default_type"])
            for d in db.get_items("parameter_definition")}


def _lists(db) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for x in sorted(db.get_items("list_value"), key=lambda x: (x["parameter_value_list_name"], x["index"])):
        out.setdefault(x["parameter_value_list_name"], []).append(x["parsed_value"])
    return out


def _values(db) -> dict:
    return {(v["entity_class_name"], tuple(v["entity_byname"]), v["parameter_definition_name"],
             v["alternative_name"]): (bytes(v["value"]), v["type"]) for v in db.get_items("parameter_value")}


def _shipped() -> tuple[dict, dict[str, list[str]]]:
    """``({(class, name): (description, types, group, value list, default value, default type)}, lists)`` of the
    shipped schema."""
    with open(package_data_path("schemas/spinedb_schema.json"), encoding="utf-8") as f:
        schema = json.load(f)
    lists: dict[str, list[str]] = {}
    for name, value in schema["parameter_value_lists"]:
        lists.setdefault(name, []).append(value)
    return {(r[0], r[1]): r[4] for r in schema["parameter_definitions"]}, lists


@pytest.fixture(scope="module")
def migrated(tmp_path_factory: pytest.TempPathFactory) -> dict:
    url = json_to_db(SEED, tmp_path_factory.mktemp("v81") / "seed.sqlite")
    migrate_database(url, up_to=80)
    with DatabaseMapping(url) as db:
        alts = sorted(a["name"] for a in db.get_items("alternative"))
        assert len(alts) >= 4
        db.add_entity(entity_class_name="connection", name="c1")
        for cls, entity, i, param, value in STORED:
            val, typ = to_database(value)
            db.add_parameter_value(entity_class_name=cls, entity_byname=(entity,), parameter_definition_name=param,
                                   alternative_name=alts[i], value=val, type=typ)
        node_method = db.get_item("parameter_definition", entity_class_name="node", name="lifetime_method")
        val, typ = to_database("no_investment")
        db.update_parameter_definition(id=node_method["id"], default_value=val, default_type=typ)
        db.commit_session("v80 lifetime, salvage and lifetime_method values, a no_investment default")
        before = {"defs": _defs(db), "lists": _lists(db), "values": _values(db)}
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        migrate_database(url, up_to=81)
    with DatabaseMapping(url) as db:
        after = {"defs": _defs(db), "lists": _lists(db), "values": _values(db),
                 "markers": {md["name"]: md["value"] for md in db.get_items("metadata")}}
    return {"url": url, "alts": alts, "before": before, "after": after, "printed": printed.getvalue()}


def test_lifetime_values_go_to_lifetime_technical_only(migrated) -> None:
    """Every stored lifetime value (constant, period map, unnamed map, time-labelled map; unit, connection, node) is
    under lifetime_technical, byte for byte with its type; payback_time has no value anywhere (an entity without
    one pays back over its lifetime_technical); lifetime is gone everywhere."""
    before, after = migrated["before"], migrated["after"]
    old = {k: v for k, v in before["values"].items() if k[2] == "lifetime"}
    assert len(old) == 6 and {k[0] for k in old} == set(CLASSES)
    for (cls, byname, _, alt), stored in old.items():
        assert after["values"][(cls, byname, "lifetime_technical", alt)] == stored
    assert not any(k[2] == "payback_time" for k in after["values"])
    assert not any(k[2] == "lifetime" for k in after["values"])
    assert not any(k[1] == "lifetime" for k in after["defs"])
    alts = migrated["alts"]
    time_map = json.loads(after["values"][("unit", ("wind",), "lifetime_technical", alts[3])][0])
    assert time_map["index_name"] == "time"


def test_salvage_value_is_net_salvage_value(migrated) -> None:
    before, after = migrated["before"], migrated["after"]
    shipped, _ = _shipped()
    old = {k: v for k, v in before["values"].items() if k[2] == "salvage_value"}
    assert len(old) == 2
    for (cls, byname, _, alt), stored in old.items():
        assert after["values"][(cls, byname, "net_salvage_value", alt)] == stored
    for cls in CLASSES:
        assert (cls, "salvage_value") not in after["defs"]
        fields = after["defs"][(cls, "net_salvage_value")]
        assert fields == (shipped[(cls, "net_salvage_value")], ("float", "1d_map"), "retirement", None, None, None)
        assert fields[0] == m._v81_texts()[(cls, "net_salvage_value")]
    assert after["defs"][("unit", "net_salvage_value")][0].startswith("[CUR/kW] Net salvage value: ")
    assert after["defs"][("node", "net_salvage_value")][0].startswith("[CUR/kWh] Net salvage value: ")


def test_new_lifetime_definitions(migrated) -> None:
    after = migrated["after"]["defs"]
    shipped, _ = _shipped()
    for cls in CLASSES:
        for name in ("lifetime_technical", "payback_time"):
            assert after[(cls, name)] == (shipped[(cls, name)], ("float", "1d_map"), "investment", None, None, None)
    assert "No value: lifetime_technical" in after[("unit", "payback_time")][0]


def test_age_and_end_of_life_definitions(migrated) -> None:
    """The age of existing capacity, its build year, the model's start year and the rounding of an end of life: the
    types, groups, defaults and value list of the schema, and the list members in the rounding order."""
    after = migrated["after"]
    shipped, shipped_lists = _shipped()
    assert after["lists"]["end_of_life_methods"] == ["closest_seam", "previous_seam", "next_seam"]
    assert shipped_lists["end_of_life_methods"] == ["closest_seam", "previous_seam", "next_seam"]
    default = to_database("closest_seam")
    for cls in CLASSES:
        for name in ("age_at_model_start", "build_year"):
            assert after["defs"][(cls, name)] == (shipped[(cls, name)], ("float",), "retirement", None, None, None)
        assert after["defs"][(cls, "end_of_life_method")] == (
            shipped[(cls, "end_of_life_method")], ("str",), "retirement", "end_of_life_methods", *default)
    assert after["defs"][("model", "start_year")] == (shipped[("model", "start_year")], ("float",), "model", None,
                                                      None, None)
    assert after["defs"][("unit", "age_at_model_start")][0].startswith("[years] Age of the existing capacity")
    assert after["defs"][("node", "build_year")][0].startswith("[year] Calendar year the existing storage capacity")
    assert "end_of_life_method" in after["defs"][("unit", "lifetime_technical")][0]


def test_no_investment_is_no_reinvestment(migrated) -> None:
    before, after = migrated["before"], migrated["after"]
    _, shipped_lists = _shipped()
    renamed = {"no_investment": "no_reinvestment"}
    assert sorted(before["lists"]["lifetime_methods"]) == ["no_investment", "reinvest_automatic", "reinvest_choice"]
    assert after["lists"]["lifetime_methods"] == [renamed.get(x, x) for x in before["lists"]["lifetime_methods"]]
    assert sorted(after["lists"]["lifetime_methods"]) == sorted(shipped_lists["lifetime_methods"])
    expected = {}
    for key, (value, typ) in before["values"].items():
        if key[2] == "lifetime_method" and json.loads(value) == "no_investment":
            value, typ = to_database("no_reinvestment")
        expected[key] = (bytes(value), typ)
    assert {k: v for k, v in after["values"].items() if k[2] == "lifetime_method"} == {
        k: v for k, v in expected.items() if k[2] == "lifetime_method"}
    for cls in CLASSES:
        assert after["defs"][(cls, "lifetime_method")][1:4] == before["defs"][(cls, "lifetime_method")][1:4]
    for cls in ("unit", "connection"):
        assert after["defs"][(cls, "lifetime_method")][4:] == before["defs"][(cls, "lifetime_method")][4:]
    assert from_database(*before["defs"][("node", "lifetime_method")][4:]) == "no_investment"
    assert from_database(*after["defs"][("node", "lifetime_method")][4:]) == "no_reinvestment"


def test_renamed_values_are_printed(migrated) -> None:
    """Each stored ``no_investment`` value is named in the message (the values follow the renamed member)."""
    alts = migrated["alts"]
    lines = [x for x in migrated["printed"].splitlines() if x.startswith("v81: value names changed: ")]
    assert len(lines) == 1
    assert sorted(lines[0].removeprefix("v81: value names changed: ").split(", ")) == sorted(
        f"unit wind lifetime_method [{alt}]: no_investment -> no_reinvestment" for alt in alts[:2])


def test_annuity_marker_names_bounded_entities_with_an_invest_cost(migrated) -> None:
    markers = migrated["after"]["markers"]
    assert json.loads(markers["flextool_behaviour_change:annuity_over_payback_time"]) == {
        "since": 81, "entities": ["thermal"]}


def test_end_of_life_marker_names_bounded_entities(migrated) -> None:
    markers = migrated["after"]["markers"]
    assert json.loads(markers["flextool_behaviour_change:end_of_life_closest_seam"]) == {
        "since": 81, "entities": ["thermal", "wind"]}


def test_fixed_cost_marker_names_investing_entities_with_a_changing_fixed_cost(migrated) -> None:
    markers = migrated["after"]["markers"]
    assert json.loads(markers["flextool_behaviour_change:fixed_cost_per_operating_period"]) == {
        "since": 81, "entities": ["thermal"]}


def test_descriptions_are_the_shipped_ones(migrated) -> None:
    before, after = migrated["before"]["defs"], migrated["after"]["defs"]
    shipped, _ = _shipped()
    texts = m._v81_texts()
    assert set(texts) == NEW_DEFINITIONS | REWRITTEN
    for key in REWRITTEN:
        assert after[key][0] == shipped[key] == texts[key] != before[key][0], key
        # The node.lifetime_method default is renamed (test_no_investment_is_no_reinvestment).
        fields = slice(1, 4) if key == ("node", "lifetime_method") else slice(1, None)
        assert after[key][fields] == before[key][fields], key
    assert "(1 + r)^-payback_time" in after[("unit", "discount_rate")][0]
    assert "net_salvage_value" in after[("node", "unit_of_measure")][0]
    assert "no_reinvestment" in after[("connection", "lifetime_method")][0]


def test_changes_nothing_else(migrated) -> None:
    before, after = migrated["before"], migrated["after"]
    touched = NEW_DEFINITIONS | OLD_DEFINITIONS | REWRITTEN | {("model", "version")}
    assert ({k: v for k, v in after["defs"].items() if k not in touched}
            == {k: v for k, v in before["defs"].items() if k not in touched})
    assert "end_of_life_methods" not in before["lists"]
    assert ({k: v for k, v in after["lists"].items() if k not in ("lifetime_methods", "end_of_life_methods")}
            == {k: v for k, v in before["lists"].items() if k != "lifetime_methods"})
    moved = {"lifetime", "salvage_value", "lifetime_method", "lifetime_technical", "payback_time",
             "net_salvage_value"}
    assert ({k: v for k, v in after["values"].items() if k[2] not in moved}
            == {k: v for k, v in before["values"].items() if k[2] not in moved})


def _rerun_changes_nothing(db, step, monkeypatch) -> None:
    """Run ``step`` again on ``db`` and check that it changes nothing: the batch commits once, and that commit has
    nothing to commit; a part does not commit, so the session is committed here and must have nothing."""
    commits: list[str] = []

    def commit_must_be_empty(db_, message) -> None:
        with pytest.raises(NothingToCommit):
            db_.commit_session(message)
        commits.append(message)

    monkeypatch.setattr(m, "_commit_step", commit_must_be_empty)
    step(db)
    if step is m._migrate_v81_schema_batch:
        assert len(commits) == 1, commits
    else:
        assert commits == []
        with pytest.raises(NothingToCommit):
            db.commit_session("rerun")


@pytest.mark.parametrize("part", ["_v81_value_lists", "_v81_definitions_part", "_v81_move_values",
                                  "_v81_list_member_renames", "_v81_descriptions", "_v81_annuity_marker",
                                  "_v81_end_of_life_marker", "_v81_fixed_cost_marker", "_migrate_v81_schema_batch",
                                  "_refresh_head_schema"])
def test_rerun_commits_nothing(migrated, part: str, monkeypatch) -> None:
    with DatabaseMapping(migrated["url"]) as db:
        _rerun_changes_nothing(db, getattr(m, part), monkeypatch)
        assert _defs(db) == migrated["after"]["defs"] and _values(db) == migrated["after"]["values"]


def _tests_json_at(tmp_path_factory, tag: str, version: int) -> str:
    """``tests.json`` (current schema) stamped with ``version``, as a database."""
    from tests.engine._unit_db import b64

    data = json.loads(TESTS_JSON.read_text(encoding="utf-8"))
    for pd in data["parameter_definitions"]:
        if pd[0] == "model" and pd[1] == "version":
            pd[2] = b64(version, "float")
    work = tmp_path_factory.mktemp(tag)
    (work / "f.json").write_text(json.dumps(data), encoding="utf-8")
    return json_to_db(work / "f.json", work / "f.sqlite")


def test_collision_keeps_the_new_value(tmp_path_factory, capsys) -> None:
    """A database built from the current template but stamped 80, with a ``lifetime`` definition (values on
    coal_plant, which has a lifetime_technical value in the same alternative, and on csp_field, which has none in
    it) and a ``no_investment`` member beside ``no_reinvestment``: the old value is copied only where
    lifetime_technical has none, nothing goes to payback_time, the old definition goes, and the list has no
    duplicate."""
    url = _tests_json_at(tmp_path_factory, "v81_collision", 80)
    with DatabaseMapping(url) as db:
        coal = db.get_item("parameter_value", entity_class_name="unit", entity_byname=("coal_plant",),
                           parameter_definition_name="lifetime_technical", alternative_name="coal_invest")
        assert coal, "tests.json: coal_plant has a lifetime_technical in alternative coal_invest"
        coal_before = bytes(coal["value"])
        db.add_parameter_definition(entity_class_name="unit", name="lifetime", parameter_type_list=("float",))
        for entity, value in (("coal_plant", 99.0), ("csp_field", 7.0)):
            val, typ = to_database(value)
            db.add_parameter_value(entity_class_name="unit", entity_byname=(entity,),
                                   parameter_definition_name="lifetime", alternative_name="coal_invest", value=val,
                                   type=typ)
        items = db.get_items("list_value", parameter_value_list_name="lifetime_methods")
        val, typ = to_database("no_investment")
        db.add_list_value(parameter_value_list_name="lifetime_methods", value=val, type=typ,
                          index=max(x["index"] for x in items) + 1)
        db.commit_session("an old lifetime beside the new names")
    capsys.readouterr()
    migrate_database(url, up_to=81)
    printed = capsys.readouterr().out
    assert ("v81: unit lifetime_technical already had a value where lifetime had one; kept the lifetime_technical "
            "value, the lifetime value was not copied to lifetime_technical:\n"
            "  unit coal_plant [coal_invest] = 99.0\n") in printed
    assert "payback_time" not in printed
    _, shipped_lists = _shipped()
    with DatabaseMapping(url) as db:
        assert not db.get_items("parameter_definition", entity_class_name="unit", name="lifetime")
        stored_after = db.get_item("parameter_value", entity_class_name="unit", entity_byname=("coal_plant",),
                                   parameter_definition_name="lifetime_technical", alternative_name="coal_invest")
        assert bytes(stored_after["value"]) == coal_before
        assert db.get_item("parameter_value", entity_class_name="unit", entity_byname=("csp_field",),
                           parameter_definition_name="lifetime_technical",
                           alternative_name="coal_invest")["parsed_value"] == 7.0
        assert not [pv for pv in db.get_items("parameter_value", entity_class_name="unit",
                                               parameter_definition_name="payback_time")
                    if pv["entity_byname"] in (("coal_plant",), ("csp_field",))]
        members = _lists(db)["lifetime_methods"]
    assert sorted(members) == sorted(shipped_lists["lifetime_methods"])


def test_current_version_database_gets_a_missing_head_definition(tmp_path_factory) -> None:
    """A database at the current version made before a definition was added to the current version's step (here:
    ``unit.payback_time`` removed, the ``payback_time`` description of node edited) is not current for the
    read-only check; ``migrate_database`` adds the definition with its shipped fields and keeps the edited
    description and every value."""
    assert m.RERUNNABLE_HEAD_STEP[FLEXTOOL_DB_VERSION] is m._migrate_v81_schema_batch
    url = _tests_json_at(tmp_path_factory, "v81_head", FLEXTOOL_DB_VERSION)
    path = Path(url.removeprefix("sqlite:///"))
    assert m._current_version_ro(path, None) == FLEXTOOL_DB_VERSION
    with DatabaseMapping(url) as db:
        db.remove_item("parameter_definition",
                       db.get_item("parameter_definition", entity_class_name="unit", name="payback_time")["id"])
        node = db.get_item("parameter_definition", entity_class_name="node", name="payback_time")
        db.update_parameter_definition(id=node["id"], description="my own text")
        db.commit_session("an older v81 database")
        values_before = {k: v for k, v in _values(db).items() if not (k[0] == "unit" and k[2] == "payback_time")}
    assert m._head_schema_missing_ro(path)
    assert m._current_version_ro(path, None) is None
    migrate_database(url)
    shipped, _ = _shipped()
    with DatabaseMapping(url) as db:
        defs = _defs(db)
        assert defs[("unit", "payback_time")] == (shipped[("unit", "payback_time")], ("float", "1d_map"),
                                                  "investment", None, None, None)
        assert defs[("node", "payback_time")][0] == "my own text"
        assert _values(db) == values_before
    assert not m._head_schema_missing_ro(path)
    assert m._current_version_ro(path, None) == FLEXTOOL_DB_VERSION


def test_current_version_database_gets_the_end_of_life_list_and_definitions(tmp_path_factory) -> None:
    """A database at the current version made before ``end_of_life_methods``, ``end_of_life_method``,
    ``age_at_model_start`` and ``model.start_year`` were added gets them, with their shipped fields, the list before
    the definition that names it; a second migration commits nothing."""
    url = _tests_json_at(tmp_path_factory, "v81_head_eol", FLEXTOOL_DB_VERSION)
    path = Path(url.removeprefix("sqlite:///"))
    with DatabaseMapping(url) as db:
        for cls, name in (("unit", "end_of_life_method"), ("node", "end_of_life_method"),
                          ("connection", "end_of_life_method"), ("unit", "age_at_model_start"),
                          ("model", "start_year")):
            db.remove_item("parameter_definition",
                           db.get_item("parameter_definition", entity_class_name=cls, name=name)["id"])
        db.remove_item("parameter_value_list", db.get_item("parameter_value_list", name="end_of_life_methods")["id"])
        db.commit_session("an older v81 database")
        values_before = _values(db)
    assert m._head_schema_missing_ro(path)
    migrate_database(url)
    shipped, shipped_lists = _shipped()
    with DatabaseMapping(url) as db:
        defs = _defs(db)
        assert _lists(db)["end_of_life_methods"] == shipped_lists["end_of_life_methods"]
        for cls in CLASSES:
            assert defs[(cls, "end_of_life_method")] == (shipped[(cls, "end_of_life_method")], ("str",), "retirement",
                                                         "end_of_life_methods", *to_database("closest_seam"))
        assert defs[("unit", "age_at_model_start")][1:] == (("float",), "retirement", None, None, None)
        assert defs[("model", "start_year")][1:] == (("float",), "model", None, None, None)
        assert _values(db) == values_before
        with pytest.raises(NothingToCommit):
            m._refresh_head_schema(db)
            db.commit_session("rerun")
    assert not m._head_schema_missing_ro(path)


def test_materialized_file_without_a_head_definition_is_stale(tmp_path_factory) -> None:
    """``materialize`` rebuilds a canonical file at the current version that lacks a head definition."""
    from flextool.update_flextool.canonical_databases import _stale_version

    url = _tests_json_at(tmp_path_factory, "v81_stale", FLEXTOOL_DB_VERSION)
    path = Path(url.removeprefix("sqlite:///"))
    assert _stale_version(path) is None
    with DatabaseMapping(url) as db:
        db.remove_item("parameter_definition",
                       db.get_item("parameter_definition", entity_class_name="connection",
                                   name="net_salvage_value")["id"])
        db.commit_session("an older v81 file")
    assert _stale_version(path) == FLEXTOOL_DB_VERSION


def _set_end_of_life_definition(db, cls: str, default: str, description: str) -> None:
    pdef = db.get_item("parameter_definition", entity_class_name=cls, name="end_of_life_method")
    val, typ = to_database(default)
    db.update_parameter_definition(id=pdef["id"], default_value=val, default_type=typ, description=description)


@pytest.mark.parametrize("case", ["shipped", "edited_text", "previous_seam_default"])
def test_current_version_database_with_the_superseded_end_of_life_default(tmp_path_factory, case) -> None:
    """A database at the current version made while ``end_of_life_method`` defaulted to ``next_seam`` (its shipped
    description, verbatim) gets the current default ``closest_seam`` and description on every class; a definition
    whose description or default the user changed keeps both.  Values are never touched.  Afterwards the database
    is current for the migration (``_current_version_ro``) and for the canonical rebuild (``_stale_version``), so it
    is not migrated or rebuilt again, and a second refresh commits nothing."""
    from flextool.update_flextool.canonical_databases import _stale_version

    url = _tests_json_at(tmp_path_factory, f"v81_eol_default_{case}", FLEXTOOL_DB_VERSION)
    path = Path(url.removeprefix("sqlite:///"))
    assert m._current_version_ro(path, None) == FLEXTOOL_DB_VERSION and _stale_version(path) is None
    old = {cls: m._v81_superseded_end_of_life_text(cls) for cls in CLASSES}
    with DatabaseMapping(url) as db:
        for cls in CLASSES:
            _set_end_of_life_definition(db, cls, "next_seam", old[cls])
        if case == "edited_text":
            _set_end_of_life_definition(db, "unit", "next_seam", "my own text")
        elif case == "previous_seam_default":
            _set_end_of_life_definition(db, "unit", "previous_seam", old["unit"])
        db.commit_session("an older v81 database")
        values_before = _values(db)
        user_unit = _defs(db)[("unit", "end_of_life_method")]
    assert m._head_defaults_stale_ro(path)
    assert m._current_version_ro(path, None) is None and _stale_version(path) == FLEXTOOL_DB_VERSION
    migrate_database(url)
    shipped, _ = _shipped()
    with DatabaseMapping(url) as db:
        defs = _defs(db)
        for cls in CLASSES:
            if cls == "unit" and case != "shipped":
                assert defs[(cls, "end_of_life_method")] == user_unit
                continue
            assert defs[(cls, "end_of_life_method")] == (shipped[(cls, "end_of_life_method")], ("str",),
                                                         "retirement", "end_of_life_methods",
                                                         *to_database("closest_seam"))
            assert shipped[(cls, "end_of_life_method")] == m._v81_texts()[(cls, "end_of_life_method")]
        assert _values(db) == values_before
        with pytest.raises(NothingToCommit):
            m._refresh_head_schema(db)
            db.commit_session("rerun")
    assert not m._head_defaults_stale_ro(path)
    assert m._current_version_ro(path, None) == FLEXTOOL_DB_VERSION and _stale_version(path) is None


def _superseded_descriptions() -> list[tuple[str, str]]:
    """The superseded shipped pairs of definitions without a default: only their description changed."""
    return sorted(key for key, (default, _text) in m._v81_superseded().items() if default is None)


@pytest.mark.parametrize("edited", [False, True], ids=["shipped", "edited_text"])
def test_current_version_database_with_a_superseded_description(tmp_path_factory, edited) -> None:
    """A database at the current version whose definitions without a default still have the description FlexTool
    shipped before (verbatim) gets the current description of each; a description the user changed is kept.
    Values are never touched, and afterwards the database is current, so it is not migrated again."""
    keys = _superseded_descriptions()
    assert keys
    url = _tests_json_at(tmp_path_factory, f"v81_superseded_text_{edited}", FLEXTOOL_DB_VERSION)
    path = Path(url.removeprefix("sqlite:///"))
    with DatabaseMapping(url) as db:
        for cls, name in keys:
            pdef = db.get_item("parameter_definition", entity_class_name=cls, name=name)
            text = "my own text" if edited else m._v81_superseded()[(cls, name)][1]
            db.update_parameter_definition(id=pdef["id"], description=text)
        db.commit_session("an older v81 database")
        values_before = _values(db)
    assert m._head_defaults_stale_ro(path) is not edited
    migrate_database(url)
    shipped, _ = _shipped()
    with DatabaseMapping(url) as db:
        defs = _defs(db)
        for cls, name in keys:
            assert defs[(cls, name)][0] == ("my own text" if edited else shipped[(cls, name)])
            assert shipped[(cls, name)] == m._v81_texts()[(cls, name)]
        assert _values(db) == values_before
    assert not m._head_defaults_stale_ro(path)
    assert m._current_version_ro(path, None) == FLEXTOOL_DB_VERSION


#: Entities per class in the large database: each one's old value sits between values of other parameters, so the
#: ids of the old values are not consecutive and their removal is one id range per value (SQLite refuses a DELETE
#: over more than about 1000 ranges).
LARGE = 1100


@pytest.fixture(scope="module")
def large(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """A v80 database with ``LARGE`` units, each with a ``lifetime`` and an ``invest_cost`` value, and ``LARGE``
    nodes, each with a ``salvage_value`` and an ``invest_cost`` value; every other node (even numbers) already has
    a ``net_salvage_value`` (a definition added at 80, as an earlier development build left it), so its old value
    is not copied, the odd ones get theirs."""
    url = json_to_db(SEED, tmp_path_factory.mktemp("v81_large") / "seed.sqlite")
    migrate_database(url, up_to=80)
    with DatabaseMapping(url) as db:
        alt = sorted(a["name"] for a in db.get_items("alternative"))[0]
        db.add_parameter_definition(entity_class_name="node", name="net_salvage_value", parameter_type_list=("float",))

        def add(cls, entity, param, value):
            val, typ = to_database(value)
            db.add_parameter_value(entity_class_name=cls, entity_byname=(entity,), parameter_definition_name=param,
                                   alternative_name=alt, value=val, type=typ)

        for i in range(LARGE):
            db.add_entity(entity_class_name="unit", name=f"u{i}")
            db.add_entity(entity_class_name="node", name=f"n{i}")
        for i in range(LARGE):
            add("unit", f"u{i}", "lifetime", Map(["p2020", "p2025"], [20.0 + i, 30.0], index_name="period")
                if i % 3 == 0 else 20.0 + i)
            add("unit", f"u{i}", "invest_cost", 100.0 + i)
            add("node", f"n{i}", "salvage_value", 10.0 + i)
            if i % 2 == 0:
                add("node", f"n{i}", "net_salvage_value", -1.0 - i)
            add("node", f"n{i}", "invest_cost", 5.0 + i)
        db.commit_session("many interleaved lifetime and salvage_value values")
        before = _values(db)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        migrate_database(url, up_to=81)
    with DatabaseMapping(url) as db:
        after = {"defs": _defs(db), "values": _values(db)}
    return {"url": url, "alt": alt, "before": before, "after": after, "printed": printed.getvalue()}


def test_large_database_moves_every_value(large) -> None:
    """A database whose old values are too scattered for one removal: it reaches 81; every ``lifetime`` value is
    under ``lifetime_technical`` byte for byte, every ``salvage_value`` under ``net_salvage_value`` where that node had
    none (the even nodes keep their own, and each is printed), the old names are gone and every other value is
    unchanged."""
    before, after = large["before"], large["after"]
    old_lifetime = {k: v for k, v in before.items() if k[2] == "lifetime"}
    old_salvage = {k: v for k, v in before.items() if k[2] == "salvage_value"}
    assert len(old_lifetime) >= LARGE and len(old_salvage) >= LARGE
    expected = {k: v for k, v in before.items() if k[2] not in ("lifetime", "salvage_value")}
    for (cls, byname, _, alt), stored in old_lifetime.items():
        expected[(cls, byname, "lifetime_technical", alt)] = stored
    for (cls, byname, _, alt), stored in old_salvage.items():
        expected.setdefault((cls, byname, "net_salvage_value", alt), stored)
    assert after["values"] == expected
    for i in range(0, LARGE, 2):
        assert after["values"][("node", (f"n{i}",), "net_salvage_value", large["alt"])] == (
            bytes(to_database(-1.0 - i)[0]), "float")
        assert f"  node n{i} [{large['alt']}] = {10.0 + i}\n" in large["printed"]
    assert not any(k[1] in ("lifetime", "salvage_value") for k in after["defs"])
    from flextool.update_flextool.frozen import read_db_version_ro

    assert read_db_version_ro(Path(large["url"].removeprefix("sqlite:///"))) == 81


def test_large_database_definitions_are_those_of_a_small_one(large, migrated) -> None:
    """The definitions after the step do not depend on how many values moved (the small database also has a
    ``node.lifetime_method`` default of its own)."""
    own = ("node", "lifetime_method")
    assert ({k: v for k, v in large["after"]["defs"].items() if k != own}
            == {k: v for k, v in migrated["after"]["defs"].items() if k != own})


def test_large_database_rerun_commits_nothing(large, monkeypatch) -> None:
    with DatabaseMapping(large["url"]) as db:
        _rerun_changes_nothing(db, m._migrate_v81_schema_batch, monkeypatch)
        assert _values(db) == large["after"]["values"]
