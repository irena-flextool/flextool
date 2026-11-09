"""Database migration to version 83 (a solve's ``new_stepduration`` combines every time series by its kind).

Seed: ``data/v75_seed_stoch_two_period.json`` (a frozen version-75 copy of
``tests/fixtures/stoch_two_period.json``) migrated to 82, then to 83.  The step
rewrites the ``node.energy_margin_adder`` description, changes no other
definition and no value, writes the behaviour-change markers only on a database
whose results change (a solve with ``new_stepduration``; with also an
``energy_margin_adder`` value, the adder marker too), and a rerun commits
nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, to_database
from spinedb_api.exception import NothingToCommit

from flextool._resources import package_data_path
from flextool.update_flextool import db_migration as m
from flextool.update_flextool.db_migration import migrate_database

from tests.db_utils import json_to_db

SEED = Path(__file__).resolve().parent / "data" / "v75_seed_stoch_two_period.json"
PREFIX = m.BEHAVIOUR_CHANGE_MARKER_PREFIX


def _defs(db) -> dict:
    return {(d["entity_class_name"], d["name"]): (tuple(d["parameter_type_list"] or ()), d["parameter_group_name"],
                                                  d["description"], d["default_value"])
            for d in db.get_items("parameter_definition")}


def _markers(db) -> dict:
    return {md["name"][len(PREFIX):]: md["value"] for md in db.get_items("metadata")
            if md["name"].startswith(PREFIX)}


def _set(db, cls: str, name: str, param: str, value: float, alt: str) -> None:
    val, typ = to_database(value)
    db.add_parameter_value(entity_class_name=cls, entity_byname=(name,), parameter_definition_name=param,
                           alternative_name=alt, value=val, type=typ)


def _migrate(tmp_path_factory, tag: str, values: dict) -> dict:
    url = json_to_db(SEED, tmp_path_factory.mktemp(tag) / "seed.sqlite")
    migrate_database(url, up_to=82)
    with DatabaseMapping(url) as db:
        alt = sorted(a["name"] for a in db.get_items("alternative"))[0]
        solve = sorted(e["name"] for e in db.get_items("entity", entity_class_name="solve"))[0]
        node = sorted(e["name"] for e in db.get_items("entity", entity_class_name="node"))[0]
        if "solve" in values:
            _set(db, "solve", solve, "new_stepduration", values["solve"], alt)
        if "node" in values:
            _set(db, "node", node, "energy_margin_adder", values["node"], alt)
        if values:
            db.commit_session("test values")
        before, markers_before = _defs(db), _markers(db)
    migrate_database(url, up_to=83)
    with DatabaseMapping(url) as db:
        after, markers = _defs(db), _markers(db)
    return {"url": url, "before": before, "after": after, "markers_before": markers_before, "markers": markers,
            "solve": solve, "node": node}


@pytest.fixture(scope="module")
def plain(tmp_path_factory: pytest.TempPathFactory) -> dict:
    return _migrate(tmp_path_factory, "v83_plain", {})


def test_rewrites_the_adder_description_only(plain) -> None:
    with open(package_data_path("schemas/spinedb_schema.json"), encoding="utf-8") as f:
        schema = json.load(f)
    shipped = {(r[0], r[1]): r[4] for r in schema["parameter_definitions"]}
    key = ("node", "energy_margin_adder")
    assert plain["after"][key][2] == shipped[key] == m._V83_ENERGY_MARGIN_ADDER_DESCRIPTION
    assert "sum of the values of the time steps it combines" in shipped[key]
    assert plain["before"][key][2] != shipped[key]
    assert plain["after"][key][:2] + plain["after"][key][3:] == plain["before"][key][:2] + plain["before"][key][3:]
    skip = {key, ("model", "version")}
    assert ({k: v for k, v in plain["after"].items() if k not in skip}
            == {k: v for k, v in plain["before"].items() if k not in skip})


def test_no_marker_without_new_stepduration(tmp_path_factory, plain) -> None:
    assert plain["markers"] == plain["markers_before"]
    adder_only = _migrate(tmp_path_factory, "v83_adder_only", {"node": 10.0})
    assert adder_only["markers"] == adder_only["markers_before"]


def test_markers_name_the_solves_and_nodes(tmp_path_factory) -> None:
    step = _migrate(tmp_path_factory, "v83_step", {"solve": 2.0})
    assert json.loads(step["markers"]["solve_step_series_combined"]) == {"since": 83, "solves": [step["solve"]]}
    assert "energy_margin_adder_per_original_step" not in step["markers"]
    both = _migrate(tmp_path_factory, "v83_both", {"solve": 2.0, "node": 10.0})
    assert json.loads(both["markers"]["energy_margin_adder_per_original_step"]) == {
        "since": 83, "solves": [both["solve"]], "nodes": [both["node"]]}
    with DatabaseMapping(both["url"]) as db:
        m._migrate_v83_solve_step_series(db)
        with pytest.raises(NothingToCommit):
            db.commit_session("rerun")


def test_rerun_is_idempotent(plain) -> None:
    with DatabaseMapping(plain["url"]) as db:
        m._migrate_v83_solve_step_series(db)
        with pytest.raises(NothingToCommit):
            db.commit_session("rerun")
