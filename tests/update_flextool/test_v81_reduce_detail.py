"""Database migration to version 81 (``solve.reduce_unit_detail_method`` / ``solve.reduce_system_detail_method``).

Seed: ``data/v75_seed_stoch_two_period.json`` (a frozen version-75 copy of
``tests/fixtures/stoch_two_period.json``) migrated to 80, then to 81.  The step
adds the two value lists and the two definitions (default ``none``, group
``solve_advanced``, the shipped description), changes nothing else, and a rerun
commits nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, from_database
from spinedb_api.exception import NothingToCommit

from flextool._resources import package_data_path
from flextool.update_flextool import db_migration as m
from flextool.update_flextool.db_migration import migrate_database

from tests.db_utils import json_to_db

SEED = Path(__file__).resolve().parent / "data" / "v75_seed_stoch_two_period.json"
NEW = {("solve", "reduce_unit_detail_method"): "reduce_unit_detail_methods",
       ("solve", "reduce_system_detail_method"): "reduce_system_detail_methods"}
UNIT_VALUES = ["none", "integers", "integers_ramps", "integers_min_limits", "integers_ramps_min_limits", "online",
               "online_ramps"]
SYSTEM_VALUES = ["none", "reserves", "inertia", "non_synchronous", "inertia_non_synchronous",
                 "reserves_inertia_non_synchronous", "dc_power_flow",
                 "reserves_inertia_non_synchronous_dc_power_flow"]


def _defs(db) -> dict:
    return {(d["entity_class_name"], d["name"]): (d["parameter_value_list_name"], d["parameter_group_name"],
                                                  d["description"],
                                                  from_database(d["default_value"], d["default_type"]),
                                                  tuple(d["parameter_type_list"] or ()))
            for d in db.get_items("parameter_definition")}


def _lists(db) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for x in sorted(db.get_items("list_value"), key=lambda x: (x["parameter_value_list_name"], x["index"])):
        out.setdefault(x["parameter_value_list_name"], []).append(x["parsed_value"])
    return out


@pytest.fixture(scope="module")
def migrated(tmp_path_factory: pytest.TempPathFactory) -> dict:
    url = json_to_db(SEED, tmp_path_factory.mktemp("v81") / "seed.sqlite")
    migrate_database(url, up_to=80)
    with DatabaseMapping(url) as db:
        before, lists_before = _defs(db), _lists(db)
    migrate_database(url, up_to=81)
    with DatabaseMapping(url) as db:
        after, lists_after = _defs(db), _lists(db)
    return {"url": url, "before": before, "after": after, "lists_before": lists_before, "lists_after": lists_after}


def test_adds_the_definitions(migrated) -> None:
    with open(package_data_path("schemas/spinedb_schema.json"), encoding="utf-8") as f:
        schema = json.load(f)
    shipped = {(r[0], r[1]): r[4] for r in schema["parameter_definitions"]}
    for key, value_list in NEW.items():
        assert key not in migrated["before"]
        vl, group, description, default, types = migrated["after"][key]
        assert types == ("str",)
        assert vl == value_list
        assert group == "solve_advanced"
        assert default == "none"
        assert description == shipped[key]
        assert description.startswith("[choice]")


def test_adds_the_value_lists_in_order(migrated) -> None:
    assert "reduce_unit_detail_methods" not in migrated["lists_before"]
    assert migrated["lists_after"]["reduce_unit_detail_methods"] == UNIT_VALUES
    assert migrated["lists_after"]["reduce_system_detail_methods"] == SYSTEM_VALUES
    assert tuple(UNIT_VALUES) == m.REDUCE_UNIT_DETAIL_METHODS
    assert tuple(SYSTEM_VALUES) == m.REDUCE_SYSTEM_DETAIL_METHODS


def test_completes_a_partial_list(tmp_path_factory) -> None:
    """A list that already holds some values (here only ``none`` and ``online``, at indexes 0 and 5) is completed:
    no value twice, and the added values take indexes after the largest one present."""
    from spinedb_api import to_database

    url = json_to_db(SEED, tmp_path_factory.mktemp("v81_partial") / "seed.sqlite")
    migrate_database(url, up_to=80)
    with DatabaseMapping(url) as db:
        db.add_parameter_value_list(name="reduce_unit_detail_methods")
        for index, value in ((0, "none"), (5, "online")):
            val, typ = to_database(value)
            db.add_list_value(parameter_value_list_name="reduce_unit_detail_methods", value=val, type=typ,
                              index=index)
        db.commit_session("partial list")
    migrate_database(url, up_to=81)
    with DatabaseMapping(url) as db:
        items = sorted(db.get_items("list_value", parameter_value_list_name="reduce_unit_detail_methods"),
                       key=lambda x: x["index"])
    values = [x["parsed_value"] for x in items]
    assert sorted(values) == sorted(UNIT_VALUES) and len(values) == len(UNIT_VALUES)
    assert [x["index"] for x in items] == [0, 5, 6, 7, 8, 9, 10]
    assert values[:2] == ["none", "online"]


def test_changes_nothing_else(migrated) -> None:
    after = {k: v for k, v in migrated["after"].items() if k not in NEW and k != ("model", "version")}
    before = {k: v for k, v in migrated["before"].items() if k != ("model", "version")}
    assert after == before
    lists_after = {k: v for k, v in migrated["lists_after"].items() if k not in NEW.values()}
    assert lists_after == migrated["lists_before"]


def test_rerun_is_idempotent(migrated) -> None:
    with DatabaseMapping(migrated["url"]) as db:
        m._migrate_v81_reduce_detail_methods(db)
        with pytest.raises(NothingToCommit):
            db.commit_session("rerun")
