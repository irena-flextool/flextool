"""A large database whose removed values are scattered migrates.

spinedb_api deletes the removed rows of a table in one ``DELETE`` with one ``id`` range per run of consecutive ids,
and SQLite refuses an expression deeper than 1000 ("Expression tree is too large").  The v70 step deletes every
``unit__inputNode.capacity_min_coeff`` value and every v36 leftover output ``capacity_min_coeff`` (min == max != 1).
In the database below each such value sits between values that stay, so each one is its own id run, and there are
more than 1000 of them on each side.

Built from scratch with ``import_data`` (never a checked-in ``.sqlite``); ``model.version`` is pinned to 69 so the
chain runs from the v70 step to the current version.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, from_database, import_data
from spinedb_api.temp_id import resolve

from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.db_migration import migrate_database
from flextool.update_flextool.db_handle import open_database_mapping
from flextool.update_flextool.frozen import read_db_version_ro

from tests.update_flextool._rerun import already_applied_records

#: Units in the large database.  Each one deletes one input value and, on even units, one output leftover: 1650
#: scattered values of one table, the input ones alone above the 1000 runs one SQLite DELETE accepts.
LARGE = 1100


def _run_count(ids) -> int:
    """How many runs of consecutive integers *ids* has."""
    ordered = sorted(set(ids))
    return sum(1 for k, i in enumerate(ordered) if k == 0 or i != ordered[k - 1] + 1)


def _input_min(i: int) -> float:
    return 0.2 + i / 10000


def _output_max(i: int) -> float:
    return 0.5 + i / 10000


def _build_v69_large(url: str) -> None:
    """``LARGE`` units, each with an input and an output.  Per unit, in this order: input ``capacity_max_coeff``
    (moves to ``input_share_max``), input ``capacity_min_coeff`` (deleted), output ``capacity_max_coeff`` (kept) and
    output ``capacity_min_coeff``: equal to the max on even units (a v36 leftover, deleted), 1.0 on odd units
    (authored, kept)."""
    entities = [["node", "fuel"], ["node", "elec"]]
    values = []
    for i in range(LARGE):
        u = f"u{i}"
        entities += [["unit", u], ["unit__inputNode", (u, "fuel")], ["unit__outputNode", (u, "elec")]]
        values += [
            ["unit__inputNode", (u, "fuel"), "capacity_max_coeff", 0.9, "Base"],
            ["unit__inputNode", (u, "fuel"), "capacity_min_coeff", _input_min(i), "Base"],
            ["unit__outputNode", (u, "elec"), "capacity_max_coeff", _output_max(i), "Base"],
            ["unit__outputNode", (u, "elec"), "capacity_min_coeff", _output_max(i) if i % 2 == 0 else 1.0, "Base"],
        ]
    with open_database_mapping(url, create=True) as db:
        _, errors = import_data(
            db,
            entity_classes=[
                ["model", ()], ["node", ()], ["unit", ()],
                ["unit__inputNode", ("unit", "node")],
                ["unit__outputNode", ("unit", "node")],
            ],
            parameter_definitions=[
                ["model", "version", 69.0, None, "Database version."],
                ["unit__inputNode", "capacity_max_coeff", 1.0, None, ""],
                ["unit__inputNode", "capacity_min_coeff", 1.0, None, ""],
                ["unit__outputNode", "capacity_max_coeff", 1.0, None, ""],
                ["unit__outputNode", "capacity_min_coeff", 1.0, None, ""],
            ],
            alternatives=[["Base", ""]],
            entities=entities,
            parameter_values=values,
        )
        assert not errors, errors[:5]
        db.commit_session("v69 database with scattered capacity coefficients")


def _values(db: DatabaseMapping) -> dict:
    return {(pv["entity_class_name"], pv["entity_byname"], pv["parameter_definition_name"]):
            from_database(pv["value"], pv["type"]) for pv in db.get_parameter_value_items()}


@pytest.fixture(scope="module")
def large(tmp_path_factory: pytest.TempPathFactory) -> dict:
    path = (tmp_path_factory.mktemp("v70_large") / "v69.sqlite").resolve()
    url = f"sqlite:///{path}"
    _build_v69_large(url)
    with open_database_mapping(url) as db:
        removed_ids = [resolve(pv["id"]) for pv in db.get_parameter_value_items()
                       if pv["parameter_definition_name"] == "capacity_min_coeff"
                       and (pv["entity_class_name"] == "unit__inputNode"
                            or from_database(pv["value"], pv["type"]) != 1.0)]
        scattered_runs = _run_count(removed_ids)
    with already_applied_records() as records:
        migrate_database(url)
    with open_database_mapping(url) as db:
        after = _values(db)
    return {"path": path, "runs": scattered_runs, "after": after, "records": records}


def test_the_removed_values_are_scattered(large) -> None:
    """The precondition: every removed value is its own id run, more than one DELETE accepts."""
    assert large["runs"] == LARGE + LARGE // 2


def test_large_database_reaches_the_current_version(large) -> None:
    assert read_db_version_ro(large["path"]) == FLEXTOOL_DB_VERSION
    assert not Path(f"{large['path']}.premigration.bak").exists()


def test_large_database_finds_nothing_already_applied(large) -> None:
    """A clean migration trips no rerun guard of a step (``_already_applied``)."""
    assert large["records"] == []


def test_large_database_values_are_those_of_the_v70_rules(large) -> None:
    """Every input max moved to ``input_share_max``; every input min and every even (leftover) output min is gone;
    every output max and odd output min (1.0) is kept as stored."""
    expected = {}
    for i in range(LARGE):
        u = f"u{i}"
        expected[("unit__inputNode", (u, "fuel"), "input_share_max")] = 0.9
        expected[("unit__outputNode", (u, "elec"), "capacity_max_coeff")] = _output_max(i)
        if i % 2:
            expected[("unit__outputNode", (u, "elec"), "capacity_min_coeff")] = 1.0
    assert large["after"] == expected
