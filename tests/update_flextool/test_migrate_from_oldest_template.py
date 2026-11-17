"""A database of the oldest data version migrates to the current one.

The oldest version the migration starts from is 0 (a database without ``model.version``); the bundled
``flextool_template_v2.json`` is such a database.  Its unit flow coefficient is ``coefficient`` until v35 renames it
to ``flow_coefficient``.  The v23 step used to type ``flow_coefficient`` (so it created that definition beside
``coefficient``) and the v35 rename then collided with it; v23 now types ``coefficient``, and v35 removes the empty
``flow_coefficient`` that an older FlexTool's v23 left in a database migrated then.

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
from spinedb_api import DatabaseMapping, SpineDBAPIError, from_database, to_database

from flextool._resources import package_data_path
from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.db_migration import migrate_database
from flextool.update_flextool.initialize_database import initialize_database

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
    with DatabaseMapping(url) as db:
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
    with DatabaseMapping(url) as db:
        return {(d["entity_class_name"], d["name"]): (tuple(d["parameter_type_list"] or ()), d["parameter_value_list_name"])
                for d in db.get_items("parameter_definition")}


def _unit_node_values(url: str) -> dict:
    with DatabaseMapping(url) as db:
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


def test_oldest_version_migrates_with_its_coefficients(tmp_path_factory, v25_path) -> None:
    url = _build(tmp_path_factory.mktemp("v2_template") / "db.sqlite", "flextool_template_v2.json", 24)
    with DatabaseMapping(url) as db:
        for cls in ("unit__inputNode", "unit__outputNode"):
            assert not db.get_item("parameter_definition", entity_class_name=cls, name="flow_coefficient")
    _quiet(migrate_database, url)
    values = _unit_node_values(url)
    assert values == v25_path["values"]
    assert len(values) > len(COEFFICIENTS)
    assert _definitions(url) == v25_path["definitions"]
    with open(package_data_path("schemas/spinedb_schema.json"), encoding="utf-8") as f:
        shipped = {(r[0], r[1]) for r in json.load(f)["parameter_definitions"]}
    assert set(v25_path["definitions"]) == shipped


def test_flow_coefficient_left_by_an_older_v23_is_removed(tmp_path_factory, v25_path) -> None:
    """A database migrated to 23..34 by an older FlexTool carries an empty ``flow_coefficient`` (type float) beside
    ``coefficient``: v35 removes it and the ``coefficient`` values carry on as without it."""
    url = _build(tmp_path_factory.mktemp("older_v23") / "db.sqlite", "flextool_template_v2.json", 34)
    with DatabaseMapping(url) as db:
        for cls in ("unit__inputNode", "unit__outputNode"):
            db.add_parameter_definition(entity_class_name=cls, name="flow_coefficient", parameter_type_list=("float",))
        db.commit_session("the flow_coefficient an older v23 created")
    _quiet(migrate_database, url)
    assert _unit_node_values(url) == v25_path["values"]


def test_a_flow_coefficient_with_values_stops_v35(tmp_path_factory) -> None:
    """A ``flow_coefficient`` holding a value beside ``coefficient`` is not the empty one of v23: v35 stops with a
    message and the database stays at 34."""
    url = _build(tmp_path_factory.mktemp("both_valued") / "db.sqlite", "flextool_template_v2.json", 34)
    with DatabaseMapping(url) as db:
        db.add_parameter_definition(entity_class_name="unit__outputNode", name="flow_coefficient",
                                    parameter_type_list=("float",))
        val, typ = to_database(3.0)
        db.add_parameter_value(entity_class_name="unit__outputNode", entity_byname=("u1", "n_out"),
                               parameter_definition_name="flow_coefficient", alternative_name="a1", value=val, type=typ)
        db.commit_session("a valued flow_coefficient")
    with pytest.raises(SpineDBAPIError, match="both coefficient and flow_coefficient"):
        _quiet(migrate_database, url)
    with DatabaseMapping(url) as db:
        assert _version(db) == 34
