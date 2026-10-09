"""Tests for the v71 database migration.

v71 renames the input-side capacity coefficients on ``unit__inputNode``:

* ``capacity_max_coeff`` -> ``input_share_max``
* ``capacity_min_coeff`` -> ``input_share_min``

``unit__outputNode`` keeps ``capacity_max_coeff`` / ``capacity_min_coeff``.
Existing values move unchanged (spinedb_api links values to the definition
by id), and the new definitions carry new descriptions.
"""

from __future__ import annotations

from pathlib import Path

from spinedb_api import DatabaseMapping, from_database, import_data

from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.db_migration import migrate_database

from tests.db_utils import json_to_db

TEST_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TEST_DIR / "fixtures"


def test_v71_version_constant_is_at_least_71() -> None:
    assert FLEXTOOL_DB_VERSION >= 71


def _build_v70_db(url: str) -> None:
    """Minimal v70 DB with input- and output-side capacity coefficients.

    Built from scratch (never a checked-in .sqlite).  ``model.version``
    default is pinned to 70 so ``migrate_database`` runs only the v71
    step.
    """
    with DatabaseMapping(url, create=True) as db:
        count, errors = import_data(
            db,
            entity_classes=[
                ["model", ()],
                ["node", ()],
                ["unit", ()],
                ["unit__inputNode", ("unit", "node")],
                ["unit__outputNode", ("unit", "node")],
            ],
            parameter_definitions=[
                ["model", "version", 70.0, None, "Database version."],
                ["unit__inputNode", "capacity_max_coeff", 1.0, None,
                 "v70 input max coefficient."],
                ["unit__inputNode", "capacity_min_coeff", 1.0, None,
                 "v70 input min coefficient."],
                ["unit__outputNode", "capacity_max_coeff", 1.0, None,
                 "v70 output max coefficient."],
                ["unit__outputNode", "capacity_min_coeff", 1.0, None,
                 "v70 output min coefficient."],
            ],
            alternatives=[["Base", ""], ["alt", ""]],
            entities=[
                ["node", "fuel"],
                ["node", "elec"],
                ["unit", "plant"],
                ["unit__inputNode", ("plant", "fuel")],
                ["unit__outputNode", ("plant", "elec")],
            ],
            parameter_values=[
                ["unit__inputNode", ("plant", "fuel"), "capacity_max_coeff",
                 0.6, "alt"],
                ["unit__inputNode", ("plant", "fuel"), "capacity_min_coeff",
                 0.3, "alt"],
                ["unit__outputNode", ("plant", "elec"), "capacity_max_coeff",
                 0.8, "alt"],
                ["unit__outputNode", ("plant", "elec"), "capacity_min_coeff",
                 0.5, "Base"],
            ],
        )
        assert not errors, f"seed import errors: {errors[:5]}"
        db.commit_session("Seed v70 DB with capacity coefficients")


def _values(db: DatabaseMapping, cls: str) -> dict:
    return {
        (pv["parameter_definition_name"], pv["entity_byname"],
         pv["alternative_name"]): from_database(pv["value"], pv["type"])
        for pv in db.get_parameter_value_items()
        if pv["entity_class_name"] == cls
    }


def _definitions(db: DatabaseMapping, cls: str) -> set[str]:
    return {p["name"] for p in db.get_parameter_definition_items()
            if p["entity_class_name"] == cls}


def test_input_values_move_to_input_share(tmp_path: Path) -> None:
    """Input-side values move to ``input_share_max`` / ``input_share_min``
    unchanged (same entity and alternative); output-side values stay on
    ``capacity_max_coeff`` / ``capacity_min_coeff``."""
    url = f"sqlite:///{(tmp_path / 'v70.sqlite').resolve()}"
    _build_v70_db(url)
    migrate_database(url)

    db = DatabaseMapping(url, create=False)
    try:
        db.fetch_all()
        in_defs = _definitions(db, "unit__inputNode")
        assert {"input_share_max", "input_share_min"} <= in_defs
        assert "capacity_max_coeff" not in in_defs
        assert "capacity_min_coeff" not in in_defs
        assert {"capacity_max_coeff", "capacity_min_coeff"} <= _definitions(
            db, "unit__outputNode")

        assert _values(db, "unit__inputNode") == {
            ("input_share_max", ("plant", "fuel"), "alt"): 0.6,
            ("input_share_min", ("plant", "fuel"), "alt"): 0.3,
        }
        assert _values(db, "unit__outputNode") == {
            ("capacity_max_coeff", ("plant", "elec"), "alt"): 0.8,
            ("capacity_min_coeff", ("plant", "elec"), "Base"): 0.5,
        }

        share_max = db.get_parameter_definition_item(
            entity_class_name="unit__inputNode", name="input_share_max")
        assert from_database(share_max["default_value"],
                             share_max["default_type"]) == 1.0
        assert "full-load" in share_max["description"]
        out_max = db.get_parameter_definition_item(
            entity_class_name="unit__outputNode", name="capacity_max_coeff")
        assert out_max["description"] == "v70 output max coefficient."
    finally:
        db.close()


def test_current_fixture_carries_new_names(tmp_path: Path) -> None:
    """A DB built from the (regenerated) test fixture has the new names
    on ``unit__inputNode`` only, and re-running the migration is a no-op
    that keeps every definition single."""
    url = json_to_db(FIXTURES_DIR / "tests.json", tmp_path / "tests.sqlite")
    migrate_database(url)
    migrate_database(url)
    db = DatabaseMapping(url, create=False)
    try:
        db.fetch_all()
        names = [(p["entity_class_name"], p["name"])
                 for p in db.get_parameter_definition_items()
                 if p["entity_class_name"] in ("unit__inputNode",
                                               "unit__outputNode")]
        assert len(names) == len(set(names))
        assert ("unit__inputNode", "input_share_max") in names
        assert ("unit__inputNode", "input_share_min") in names
        assert ("unit__inputNode", "capacity_max_coeff") not in names
        assert ("unit__outputNode", "capacity_max_coeff") in names
        assert ("unit__outputNode", "capacity_min_coeff") in names
    finally:
        db.close()
