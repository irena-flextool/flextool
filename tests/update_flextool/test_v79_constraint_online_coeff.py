"""Database migration to version 79 (``constraint_online_coeff`` on unit and connection).

Seed: ``data/v75_seed_stoch_two_period.json`` (a frozen version-75 copy of
``tests/fixtures/stoch_two_period.json``) migrated to 78, then to 79.  The step
adds the two definitions (``1d_map``, group ``constraint``, no default, the
shipped description), changes nothing else, and a rerun commits nothing.
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
NEW = [("unit", "constraint_online_coeff"), ("connection", "constraint_online_coeff")]


def _defs(db) -> dict:
    return {(d["entity_class_name"], d["name"]): (tuple(d["parameter_type_list"] or ()), d["parameter_group_name"],
                                                  d["description"], d["default_value"])
            for d in db.get_items("parameter_definition")}


@pytest.fixture(scope="module")
def migrated(tmp_path_factory: pytest.TempPathFactory) -> dict:
    url = json_to_db(SEED, tmp_path_factory.mktemp("v79") / "seed.sqlite")
    migrate_database(url, up_to=78)
    with DatabaseMapping(url) as db:
        before = _defs(db)
    migrate_database(url)
    with DatabaseMapping(url) as db:
        after = _defs(db)
    return {"url": url, "before": before, "after": after}


def test_adds_the_definitions(migrated) -> None:
    with open(package_data_path("schemas/spinedb_schema.json"), encoding="utf-8") as f:
        schema = json.load(f)
    shipped = {(r[0], r[1]): r[4] for r in schema["parameter_definitions"]}
    for key in NEW:
        assert key not in migrated["before"]
        types, group, description, default = migrated["after"][key]
        assert types == ("1d_map",)
        assert group == "constraint"
        assert default is None
        assert description == shipped[key]
        assert description.startswith("[unit of the constraint per unit online]")


def test_changes_nothing_else(migrated) -> None:
    after = {k: v for k, v in migrated["after"].items() if k not in NEW and k != ("model", "version")}
    before = {k: v for k, v in migrated["before"].items() if k != ("model", "version")}
    assert after == before


def test_rerun_is_idempotent(migrated) -> None:
    with DatabaseMapping(migrated["url"]) as db:
        m._migrate_v79_constraint_online_coeff(db)
        with pytest.raises(NothingToCommit):
            db.commit_session("rerun")
