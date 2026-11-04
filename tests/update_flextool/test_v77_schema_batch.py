"""Database migration to version 77 (schema batch).

Seed: ``data/v75_seed_stoch_two_period.json`` (a frozen version-75 copy of
``tests/fixtures/stoch_two_period.json``; the fixtures themselves are kept at
the current version) migrated to 76, then extended with
the pre-77 content v77 must handle: group ``invest_method`` values
(``not_allowed`` and another method), a collided ``unit__inputNode``
``capacity_max_coeff`` / ``input_share_max`` pair, stale definitions with
values, a ``solve_mode`` ``single_shot`` value, and one widened definition
deleted (it must stay absent).  Then the v77 step runs and the result is
compared with the v76 snapshot.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, export_data, import_data, to_database
from spinedb_api.exception import NothingToCommit

from flextool._resources import package_data_path
from flextool.update_flextool import db_migration as m
from flextool.update_flextool.db_migration import migrate_database

from tests.db_utils import json_to_db

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SEED = Path(__file__).resolve().parent / "data" / "v75_seed_stoch_two_period.json"
MARKER = "flextool_behaviour_change:user_constraint_prebuilt_nets_retirements"
ABSENT = ("node", "energy_margin_adder")  # deleted from the seed: the migration must not re-create it


def _put(db, cls, byname, name, alt, value):
    v, t = to_database(value)
    db.add_parameter_value(entity_class_name=cls, entity_byname=byname, parameter_definition_name=name,
                           alternative_name=alt, value=v, type=t)


def _snapshot(db) -> dict:
    defs = {(d["entity_class_name"], d["name"]): d for d in db.get_items("parameter_definition")}
    values = {(v["entity_class_name"], tuple(v["entity_byname"]), v["parameter_definition_name"],
               v["alternative_name"]): v["parsed_value"] for v in db.get_items("parameter_value")}
    lists = {}
    for lv in db.get_items("list_value"):
        lists.setdefault(lv["parameter_value_list_name"], []).append(lv["parsed_value"])
    return {
        "types": {k: tuple(d["parameter_type_list"] or ()) for k, d in defs.items()},
        "descriptions": {k: d["description"] for k, d in defs.items()},
        "groups": {k: d["parameter_group_name"] for k, d in defs.items()},
        "values": values,
        "lists": lists,
        "list_of_value": {(v["entity_class_name"], tuple(v["entity_byname"]), v["alternative_name"]):
                          v["parameter_value_list_name"] for v in db.get_items("parameter_value")
                          if v["parameter_definition_name"] == "invest_method"
                          and v["entity_class_name"] == "group"},
        "markers": sorted((md["name"], md["value"]) for md in db.get_items("metadata")),
    }


@pytest.fixture(scope="module")
def migrated(tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp = tmp_path_factory.mktemp("v77")
    url = json_to_db(SEED, tmp / "seed.sqlite")
    migrate_database(url, up_to=76)
    with DatabaseMapping(url) as db:
        alt = db.get_items("alternative")[0]["name"]
        unit = db.get_items("entity", entity_class_name="unit")[0]["name"]
        node = db.get_items("entity", entity_class_name="node")[0]["name"]
        db.add_entity(entity_class_name="node", name="n_extra")
        nodes = [node, "n_extra"]
        for g in ("g_old", "g_total", "g_default"):
            db.add_entity(entity_class_name="group", name=g)
        _put(db, "group", ("g_old",), "invest_method", alt, "not_allowed")
        _put(db, "group", ("g_total",), "invest_method", alt, "invest_total")
        # A collided v70 rename: both names present, values on both.
        db.add_parameter_definition(entity_class_name="unit__inputNode", name="input_share_max",
                                    parameter_type_list=("float",))
        free, taken = (unit, nodes[0]), (unit, nodes[1])
        for byname in (free, taken):
            db.add_entity(entity_class_name="unit__inputNode", entity_byname=byname)
        _put(db, "unit__inputNode", free, "capacity_max_coeff", alt, 0.6)
        _put(db, "unit__inputNode", taken, "capacity_max_coeff", alt, 0.7)
        _put(db, "unit__inputNode", taken, "input_share_max", alt, 0.9)
        _put(db, "unit__inputNode", free, "capacity_min_coeff", alt, 0.2)
        solve = db.get_items("entity", entity_class_name="solve")[0]["name"]
        _put(db, "solve", (solve,), "discount_years", alt, 3.0)
        val, typ = to_database("single_shot")
        db.add_item("list_value", parameter_value_list_name="solve_mode", value=val, type=typ, index=99)
        db.commit_session("pre-77 content")
        _put(db, "solve", (solve,), "solve_mode", alt, "single_shot")
        db.remove_item("parameter_definition",
                       db.get_item("parameter_definition", entity_class_name=ABSENT[0], name=ABSENT[1])["id"])
        db.commit_session("pre-77 content")
        before = _snapshot(db)
    migrate_database(url)
    with DatabaseMapping(url) as db:
        after = _snapshot(db)
        data = export_data(db)
        # Re-running every sub-step changes nothing.
        m._v77_types_and_descriptions(db)
        m._v77_new_definitions(db)
        m._v77_group_invest_methods(db)
        m._v77_behaviour_markers(db)
        m._v77_drop_stale_definitions(db)
        try:
            db.commit_session("re-run")
            rerun_idempotent = False
        except NothingToCommit:
            rerun_idempotent = True
    fresh = tmp / "fresh.sqlite"
    with DatabaseMapping(f"sqlite:///{fresh}", create=True) as db:
        _count, import_errors = import_data(db, **data)
        db.commit_session("import")
    return {"before": before, "after": after, "alt": alt, "solve": solve, "free": free, "taken": taken,
            "rerun_idempotent": rerun_idempotent, "import_errors": import_errors}


def test_type_lists_are_the_union_of_old_and_target(migrated) -> None:
    before, after = migrated["before"]["types"], migrated["after"]["types"]
    for key, target in m._V77_TYPE_TARGETS.items():
        if key not in before:
            continue
        assert set(after[key]) == set(before[key]) | set(target), key
    assert "3d_map" in after[("unit", "efficiency")]  # the stochastic type the seed had is kept


def test_absent_definition_is_not_created(migrated) -> None:
    assert ABSENT not in migrated["after"]["types"]


def test_descriptions_are_the_v77_texts(migrated) -> None:
    after = migrated["after"]["descriptions"]
    texts = m._v77_texts()
    for key, text in texts.items():
        if key in after:
            assert after[key] == text, key


def test_new_definitions_and_parameter_groups(migrated) -> None:
    after = migrated["after"]
    for bound in ("max", "min"):
        assert after["types"][("group", f"retire_{bound}_period")] == ("float", "1d_map")
        assert after["types"][("group", f"retire_{bound}_total")] == ("float",)
        assert after["groups"][("group", f"retire_{bound}_period")] == "retirement"
    for cls in ("unit", "node", "connection"):
        for name in ("constraint_capacity_coeff", "constraint_retired_capacity_coeff",
                     "constraint_in_construction_capacity_coeff"):
            assert after["types"][(cls, name)] == ("1d_map",)
            assert after["groups"][(cls, name)] == "constraint"


def test_group_invest_method_values_move_to_the_group_list(migrated) -> None:
    after, alt = migrated["after"], migrated["alt"]
    assert after["lists"]["group_invest_methods"] == list(m._V77_GROUP_INVEST_METHODS)
    assert "not_allowed" in after["lists"]["invest_methods"]  # the entity list is unchanged
    assert after["values"][("group", ("g_old",), "invest_method", alt)] == "not_active"
    assert after["values"][("group", ("g_total",), "invest_method", alt)] == "invest_total"
    assert set(after["list_of_value"].values()) == {"group_invest_methods"}


def test_migrated_database_exports_and_imports_into_a_fresh_one(migrated) -> None:
    assert migrated["import_errors"] == []


def test_behaviour_marker_written(migrated) -> None:
    assert (MARKER, "77") in migrated["after"]["markers"]


def test_capacity_max_coeff_moves_to_input_share_max_where_free(migrated) -> None:
    after, alt = migrated["after"], migrated["alt"]
    assert after["values"][("unit__inputNode", migrated["free"], "input_share_max", alt)] == 0.6
    assert after["values"][("unit__inputNode", migrated["taken"], "input_share_max", alt)] == 0.9
    for name in ("capacity_max_coeff", "capacity_min_coeff"):
        assert ("unit__inputNode", name) not in after["types"]


def test_stale_definitions_dropped(migrated) -> None:
    after = migrated["after"]["types"]
    for key in m._V77_STALE_DEFINITIONS:
        assert key not in after, key


def test_single_shot_becomes_single_solve(migrated) -> None:
    after, alt = migrated["after"], migrated["alt"]
    assert after["values"][("solve", (migrated["solve"],), "solve_mode", alt)] == "single_solve"
    assert "single_shot" not in after["lists"]["solve_mode"]


def test_other_values_unchanged(migrated) -> None:
    before, after = migrated["before"]["values"], migrated["after"]["values"]
    changed = {
        (cls, name) for cls, name in m._V77_STALE_DEFINITIONS
    } | {("group", "invest_method"), ("solve", "solve_mode"), ("unit__inputNode", "input_share_max")}
    for key, value in before.items():
        if (key[0], key[2]) in changed:
            continue
        assert after[key] == value, key


def test_rerun_is_idempotent(migrated) -> None:
    assert migrated["rerun_idempotent"]


def test_stale_values_are_listed(tmp_path: Path, capsys) -> None:
    url = json_to_db(SEED, tmp_path / "seed.sqlite")
    migrate_database(url, up_to=76)
    with DatabaseMapping(url) as db:
        solve = db.get_items("entity", entity_class_name="solve")[0]["name"]
        _put(db, "solve", (solve,), "discount_years", db.get_items("alternative")[0]["name"], 3.0)
        db.commit_session("stale value")
    capsys.readouterr()
    migrate_database(url)
    out = capsys.readouterr().out
    assert "solve.discount_years is no longer a FlexTool parameter" in out
    assert f"solve {solve} [" in out and "= 3.0" in out


def _repo_jsons() -> list[Path]:
    out = [Path(package_data_path("schemas/spinedb_schema.json"))]
    out += sorted(Path(package_data_path("schemas/canonical_databases")).glob("*.json"))
    out += sorted(FIXTURES.glob("*.json"))
    return out


@pytest.mark.parametrize("path", _repo_jsons(), ids=lambda p: p.name)
def test_repository_jsons_carry_no_behaviour_markers(path: Path) -> None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert not any(str(row[0]).startswith("flextool_behaviour_change:") for row in data.get("metadata", []))


def test_construction_lead_time_marker_written(migrated) -> None:
    """A database upgraded from below v77 carries the construction lead time behaviour-change marker."""
    assert ("flextool_behaviour_change:construction_lead_time_commissioning", "77") in migrated["after"]["markers"]
