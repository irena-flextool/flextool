"""A database of the oldest data version migrates to the current one.

The oldest version the migration starts from is 0 (a database without ``model.version``); the bundled
``flextool_template_v2.json`` is such a database.  Its unit flow coefficient is ``coefficient`` until v35 renames it
to ``flow_coefficient``.  The v23 step used to type ``flow_coefficient`` (so it created that definition beside
``coefficient``) and the v35 rename then collided with it; v23 now types ``coefficient``.

A FlexTool 4.0..4.5 migration of such a database failed at v35 after steps 23..34 had committed, under the old stamp
(each step commits; the version is written at the end).  Migrating that database reruns those steps: v23 does not
bring back the names v25 and v32 changed, v25 and v32 recognise their own result, and v35 removes the empty
``flow_coefficient`` the older v23 created.  The end result equals the clean path's.

The databases are built from the bundled template JSONs (never a checked-in ``.sqlite``).  The oracle is relational:
the same ``coefficient`` values written at v24 into a database from the v2 template, or at v25 into one from the
v25 template (which never runs v23), end as the same values at the current version, and the two databases end with
the same parameter definitions as the shipped schema.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import pytest
from spinedb_api import SpineDBAPIError, from_database, to_database

from flextool._resources import package_data_path
from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.db_migration import migrate_database
from flextool.update_flextool.db_handle import open_database_mapping
from flextool.update_flextool.initialize_database import initialize_database

from tests.update_flextool._rerun import restamp

TEMPLATES = package_data_path("schemas/pre_v26")
#: (class, unit, node, coefficient value) written before the coefficient steps.
COEFFICIENTS = [("unit__inputNode", "u1", "n_in", 0.5), ("unit__outputNode", "u1", "n_out", 2.0)]


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _version(db) -> float:
    pdef = db.get_item("parameter_definition", entity_class_name="model", name="version")
    return from_database(pdef["default_value"], pdef["default_type"])


def _build(path: Path, template: str, up_to: int) -> str:
    """A database from *template* migrated to *up_to*, with a unit, two nodes and the :data:`COEFFICIENTS`."""
    url = "sqlite:///" + str(path)
    _quiet(initialize_database, str(TEMPLATES / template), str(path))
    _quiet(migrate_database, url, up_to=up_to)
    with open_database_mapping(url) as db:
        assert _version(db) == up_to
        db.add_alternative(name="a1")
        db.add_entity(entity_class_name="unit", name="u1")
        for name in ("n_in", "n_out"):
            db.add_entity(entity_class_name="node", name=name)
        for cls, unit, node, value in COEFFICIENTS:
            db.add_entity(entity_class_name=cls, element_name_list=(unit, node))
            val, typ = to_database(value)
            db.add_parameter_value(entity_class_name=cls, entity_byname=(unit, node), parameter_definition_name="coefficient",
                                   alternative_name="a1", value=val, type=typ)
        db.commit_session("coefficients")
    return url


def _definitions(url: str) -> dict:
    with open_database_mapping(url) as db:
        return {(d["entity_class_name"], d["name"]): (tuple(d["parameter_type_list"] or ()), d["parameter_value_list_name"])
                for d in db.get_items("parameter_definition")}


def _unit_node_values(url: str) -> dict:
    with open_database_mapping(url) as db:
        assert _version(db) == FLEXTOOL_DB_VERSION
        return {(v["entity_class_name"], tuple(v["entity_byname"]), v["parameter_definition_name"], v["alternative_name"]):
                from_database(v["value"], v["type"])
                for cls in ("unit__inputNode", "unit__outputNode")
                for v in db.get_items("parameter_value", entity_class_name=cls)}


@pytest.fixture(scope="module")
def v25_path(tmp_path_factory) -> dict:
    url = _build(tmp_path_factory.mktemp("v25_template") / "db.sqlite", "flextool_template_v25.json", 25)
    _quiet(migrate_database, url)
    return {"definitions": _definitions(url), "values": _unit_node_values(url)}


def _full_definitions(url: str) -> dict:
    with open_database_mapping(url) as db:
        return {(d["entity_class_name"], d["name"]): (tuple(d["parameter_type_list"] or ()),
                                                      from_database(d["default_value"], d["default_type"]),
                                                      d["parameter_value_list_name"], d["description"],
                                                      d["parameter_group_name"])
                for d in db.get_items("parameter_definition")}


@pytest.fixture(scope="module")
def v2_path(tmp_path_factory) -> dict:
    """The clean path: a v24 database from the v2 template (version 0) migrated to the current version."""
    url = _build(tmp_path_factory.mktemp("v2_template") / "db.sqlite", "flextool_template_v2.json", 24)
    with open_database_mapping(url) as db:
        for cls in ("unit__inputNode", "unit__outputNode"):
            assert not db.get_item("parameter_definition", entity_class_name=cls, name="flow_coefficient")
    _quiet(migrate_database, url)
    return {"url": url, "values": _unit_node_values(url), "definitions": _definitions(url),
            "full_definitions": _full_definitions(url)}


def test_oldest_version_migrates_with_its_coefficients(v2_path, v25_path) -> None:
    url = v2_path["url"]
    values = _unit_node_values(url)
    assert values == v25_path["values"]
    assert len(values) > len(COEFFICIENTS)
    assert _definitions(url) == v25_path["definitions"]
    with open(package_data_path("schemas/spinedb_schema.json"), encoding="utf-8") as f:
        shipped = {(r[0], r[1]) for r in json.load(f)["parameter_definitions"]}
    assert set(v25_path["definitions"]) == shipped


def _leave_failed_v35_state(url: str, stamp: int, stray_default: bytes | None = None) -> None:
    """The state a FlexTool 4.0..4.5 migration left on a pre-v25 database: each step commits on its own and the
    version is written only at the end, so its failed v35 left steps 23..34 applied (with the definitions its v23
    created: flow_coefficient, max_capacity_coefficient and min_capacity_coefficient of type float) under the old
    stamp.  *url* is at 34."""
    with open_database_mapping(url) as db:
        for cls in ("unit__inputNode", "unit__outputNode"):
            for name in ("flow_coefficient", "max_capacity_coefficient", "min_capacity_coefficient"):
                kwargs = {"default_value": stray_default, "default_type": None} if stray_default is not None else {}
                db.add_parameter_definition(entity_class_name=cls, name=name, parameter_type_list=("float",), **kwargs)
        restamp(db, stamp)
        db.commit_session(f"steps 23..34 of a failed migration under stamp {stamp}")


@pytest.mark.parametrize("stamp", [22, 17, 11])
def test_a_failed_older_migration_completes_as_the_clean_one(tmp_path_factory, stamp, v2_path) -> None:
    """Steps 23..34 applied under stamp 22 or 17 (a database of that version) or 11 (a version-0 database: the
    template imports of steps 1..10 write their version, the last one 11; no stamp below 11 is left with v23 applied):
    the migration reruns them (v23 does not bring back the names v25 / v32 changed, v25 and v32 see their own
    result, v35 removes the empty flow_coefficient) and ends with the definitions and values of the clean path."""
    url = _build(tmp_path_factory.mktemp(f"failed_at_v35_{stamp}") / "db.sqlite", "flextool_template_v2.json", 34)
    _leave_failed_v35_state(url, stamp)
    _quiet(migrate_database, url)
    assert _unit_node_values(url) == v2_path["values"]
    assert _full_definitions(url) == v2_path["full_definitions"]


def test_an_empty_flow_coefficient_with_a_null_default_is_removed(tmp_path_factory, v2_path) -> None:
    """A definition without a default that went through a JSON export and import has the default ``null`` (not SQL
    NULL); it is still the empty flow_coefficient (stamp 34: only v35 and later run)."""
    url = _build(tmp_path_factory.mktemp("null_default") / "db.sqlite", "flextool_template_v2.json", 34)
    _leave_failed_v35_state(url, 34, stray_default=to_database(None)[0])
    with open_database_mapping(url) as db:
        stray = db.get_item("parameter_definition", entity_class_name="unit__inputNode", name="flow_coefficient")
        assert stray["default_value"] == b"null"
    _quiet(migrate_database, url)
    assert _unit_node_values(url) == v2_path["values"]


def test_a_flow_coefficient_with_values_stops_v35(tmp_path_factory) -> None:
    """A ``flow_coefficient`` holding a value beside ``coefficient`` is not the empty one of v23: v35 stops with a
    message and the database stays at 34."""
    url = _build(tmp_path_factory.mktemp("both_valued") / "db.sqlite", "flextool_template_v2.json", 34)
    with open_database_mapping(url) as db:
        db.add_parameter_definition(entity_class_name="unit__outputNode", name="flow_coefficient",
                                    parameter_type_list=("float",))
        val, typ = to_database(3.0)
        db.add_parameter_value(entity_class_name="unit__outputNode", entity_byname=("u1", "n_out"),
                               parameter_definition_name="flow_coefficient", alternative_name="a1", value=val, type=typ)
        db.commit_session("a valued flow_coefficient")
    with pytest.raises(SpineDBAPIError, match="both coefficient and flow_coefficient"):
        _quiet(migrate_database, url)
    with open_database_mapping(url) as db:
        assert _version(db) == 34


def test_a_v24_database_without_either_time_structure_stops_v25(tmp_path_factory) -> None:
    """Without ``timeblockSet`` and without the v25 result, v25 stops with a message (it does not skip)."""
    url = _build(tmp_path_factory.mktemp("no_timeblockset") / "db.sqlite", "flextool_template_v2.json", 24)
    with open_database_mapping(url) as db:
        db.remove_entity_class(id=db.get_item("entity_class", name="timeblockSet")["id"])
        db.commit_session("no timeblockSet")
    with pytest.raises(SpineDBAPIError, match="v25: the database has no timeblockSet class"):
        _quiet(migrate_database, url)
    with open_database_mapping(url) as db:
        assert _version(db) == 24


@pytest.mark.parametrize(("cls", "name"), [("unit", "constraint_capacity_coefficient"),
                                           ("solve", "period_timeblockSet")])
def test_a_deleted_v22_definition_is_recreated_by_v23(tmp_path_factory, v2_path, cls, name) -> None:
    """A v22 database whose user deleted a definition that v25 / v32 later renames: its successor does not exist,
    so v23 creates the definition again (as it always did) and the migration ends with the clean path's
    definitions."""
    url = _build(tmp_path_factory.mktemp(f"deleted_{name}") / "db.sqlite", "flextool_template_v2.json", 22)
    with open_database_mapping(url) as db:
        db.remove_parameter_definition(id=db.get_item("parameter_definition", entity_class_name=cls, name=name)["id"])
        db.commit_session(f"deleted {cls}.{name}")
    _quiet(migrate_database, url)
    assert _unit_node_values(url) == v2_path["values"]
    assert set(_full_definitions(url)) == set(v2_path["full_definitions"])
