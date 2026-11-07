"""``make_db`` writes the same import data after ``add_override_alternative`` was taken out of it (P3 test 10).

The fixture JSON ``make_db`` writes next to its database is ``tests.json`` with exactly the rows below appended, in
this order; ``add_override_alternative`` alone gives the same data.  The database build and the migration are
replaced by stand-ins here: the rows are what this test checks, and the many tests that use ``make_db`` build and
solve the databases themselves.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.engine._unit_db import BASE_FIXTURE_JSON, add_override_alternative, b64, make_db

VALUES = [["node", "west", "inflow", b64(-500.0, "float")],
          ["unit", "u_new", "existing", b64(10.0, "float")]]
ENTITIES = [["node", "n_new", "A new node."], ["unit", "u_new", None], ["connection", "c_new", None],
            ["commodity", "k_new", None], ["unit__outputNode", ["u_new", "n_new"], None]]
CHAIN = ["init", "west", "coal"]

#: The rows appended to tests.json, written out by hand.
APPENDED = {
    "alternatives": [["ov", ""]],
    "entities": ENTITIES,
    "entity_alternatives": [["node", ["n_new"], "ov", True], ["unit", ["u_new"], "ov", True],
                            ["connection", ["c_new"], "ov", True]],
    "parameter_values": [["node", "west", "inflow", b64(-500.0, "float"), "ov"],
                         ["unit", "u_new", "existing", b64(10.0, "float"), "ov"]],
    "scenarios": [["ov_test", False, ""]],
    "scenario_alternatives": [["ov_test", "init", "west"], ["ov_test", "west", "coal"], ["ov_test", "coal", "ov"],
                              ["ov_test", "ov", None]],
}


def _expected() -> dict:
    data = json.loads(BASE_FIXTURE_JSON.read_text())
    for key, rows in APPENDED.items():
        data[key].extend(json.loads(json.dumps(rows)))
    return data


@pytest.fixture
def no_database(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """``json_to_db`` and ``migrate_database`` record the URL instead of building a database."""
    import importlib

    db_utils = importlib.import_module("tests.db_utils")
    migration = importlib.import_module("flextool.update_flextool.db_migration")
    built: list[str] = []
    monkeypatch.setattr(db_utils, "json_to_db", lambda json_path, db_path: f"sqlite:///{Path(db_path).as_posix()}")
    monkeypatch.setattr(migration, "migrate_database", lambda url: built.append(url))
    return built


def test_make_db_writes_the_fixture_with_the_override_rows(tmp_path_factory, no_database) -> None:
    url, scenario = make_db(tmp_path_factory, alt="ov", values=VALUES, base_chain=CHAIN, entities=ENTITIES)
    assert scenario == "ov_test"
    assert no_database == [url]
    written = Path(url.removeprefix("sqlite:///")).with_suffix(".json")
    assert json.loads(written.read_text()) == _expected()


def test_add_override_alternative_is_the_same_data() -> None:
    data = json.loads(BASE_FIXTURE_JSON.read_text())
    assert add_override_alternative(data, "ov", VALUES, CHAIN, ENTITIES) == "ov_test"
    assert data == _expected()


def test_without_entities_only_the_alternative_values_and_scenario() -> None:
    data = json.loads(BASE_FIXTURE_JSON.read_text())
    before = json.loads(BASE_FIXTURE_JSON.read_text())
    add_override_alternative(data, "solo", [], ["init"])
    assert data["entities"] == before["entities"]
    assert data["entity_alternatives"] == before["entity_alternatives"]
    assert data["parameter_values"] == before["parameter_values"]
    assert data["alternatives"][-1] == ["solo", ""]
    assert data["scenario_alternatives"][-2:] == [["solo_test", "init", "solo"], ["solo_test", "solo", None]]
