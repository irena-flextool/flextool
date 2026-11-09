"""Database migration to version 82 (``group.inertia_limit`` / ``group.non_synchronous_limit`` per time step).

Seed: ``data/v75_seed_stoch_two_period.json`` (a frozen version-75 copy of
``tests/fixtures/stoch_two_period.json``) migrated to 81, then to 82.  The step
widens the two type lists to float / 1d_map / 2d_map and writes the shipped
descriptions, changes nothing else, and a rerun commits nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping
from spinedb_api.exception import NothingToCommit

from flextool._resources import package_data_path
from flextool.update_flextool import db_migration as m
from flextool.update_flextool.db_migration import migrate_database

from tests.db_utils import json_to_db

SEED = Path(__file__).resolve().parent / "data" / "v75_seed_stoch_two_period.json"
CHANGED = [("group", "inertia_limit"), ("group", "non_synchronous_limit")]


def _defs(db) -> dict:
    return {(d["entity_class_name"], d["name"]): (tuple(d["parameter_type_list"] or ()), d["parameter_group_name"],
                                                  d["description"], d["default_value"])
            for d in db.get_items("parameter_definition")}


@pytest.fixture(scope="module")
def migrated(tmp_path_factory: pytest.TempPathFactory) -> dict:
    url = json_to_db(SEED, tmp_path_factory.mktemp("v82") / "seed.sqlite")
    migrate_database(url, up_to=81)
    with DatabaseMapping(url) as db:
        before = _defs(db)
    migrate_database(url, up_to=82)
    with DatabaseMapping(url) as db:
        after = _defs(db)
    return {"url": url, "before": before, "after": after}


def test_widens_the_types_and_rewrites_the_descriptions(migrated) -> None:
    with open(package_data_path("schemas/spinedb_schema.json"), encoding="utf-8") as f:
        schema = json.load(f)
    shipped = {(r[0], r[1]): r[4] for r in schema["parameter_definitions"]}
    for key in CHANGED:
        types, group, description, default = migrated["after"][key]
        assert migrated["before"][key][0] == ("float", "1d_map")
        assert types == ("float", "1d_map", "2d_map")
        assert description == shipped[key]
        assert "Constant, period, time or period+time." in description
        assert migrated["before"][key][2] != description
        assert (group, default) == (migrated["before"][key][1], migrated["before"][key][3])


def test_changes_nothing_else(migrated) -> None:
    skip = set(CHANGED) | {("model", "version")}
    after = {k: v for k, v in migrated["after"].items() if k not in skip}
    before = {k: v for k, v in migrated["before"].items() if k not in skip}
    assert after == before


def test_rerun_is_idempotent(migrated) -> None:
    with DatabaseMapping(migrated["url"]) as db:
        m._migrate_v82_group_limits_by_time(db)
        with pytest.raises(NothingToCommit):
            db.commit_session("rerun")
