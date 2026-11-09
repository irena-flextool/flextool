"""Database migration to version 80 (``ramp_method`` / ``ramp_cost`` descriptions: ``ramp_cost`` is charged).

Seed: ``data/v75_seed_stoch_two_period.json`` (a frozen version-75 copy of
``tests/fixtures/stoch_two_period.json``) migrated to 79, then to 80.  The step
rewrites the four descriptions to the shipped text, changes nothing else, and a
rerun commits nothing.
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
CHANGED = [(cls, name) for cls in ("unit__inputNode", "unit__outputNode") for name in ("ramp_method", "ramp_cost")]


def _defs(db) -> dict:
    return {(d["entity_class_name"], d["name"]): (tuple(d["parameter_type_list"] or ()), d["parameter_group_name"],
                                                  d["description"], d["default_value"])
            for d in db.get_items("parameter_definition")}


@pytest.fixture(scope="module")
def migrated(tmp_path_factory: pytest.TempPathFactory) -> dict:
    url = json_to_db(SEED, tmp_path_factory.mktemp("v80") / "seed.sqlite")
    migrate_database(url, up_to=79)
    with DatabaseMapping(url) as db:
        before = _defs(db)
    migrate_database(url, up_to=80)
    with DatabaseMapping(url) as db:
        after = _defs(db)
    return {"url": url, "before": before, "after": after}


def test_rewrites_the_descriptions(migrated) -> None:
    with open(package_data_path("schemas/spinedb_schema.json"), encoding="utf-8") as f:
        schema = json.load(f)
    shipped = {(r[0], r[1]): r[4] for r in schema["parameter_definitions"]}
    for key in CHANGED:
        description = migrated["after"][key][2]
        assert description == shipped[key]
        assert "not functional" not in description
        assert migrated["before"][key][2] != description
        assert migrated["after"][key][:2] == migrated["before"][key][:2]
        assert migrated["after"][key][3] == migrated["before"][key][3]


def test_changes_nothing_else(migrated) -> None:
    skip = set(CHANGED) | {("model", "version")}
    after = {k: v for k, v in migrated["after"].items() if k not in skip}
    before = {k: v for k, v in migrated["before"].items() if k not in skip}
    assert after == before


def test_rerun_is_idempotent(migrated) -> None:
    with DatabaseMapping(migrated["url"]) as db:
        m._migrate_v80_ramp_cost_descriptions(db)
        with pytest.raises(NothingToCommit):
            db.commit_session("rerun")
